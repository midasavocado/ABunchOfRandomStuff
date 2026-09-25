#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "moderngl", "pillow", "imageio-ffmpeg"]
# ///
"""
LODESTAR — the film renderer.

Every frame is ray-marched on the GPU from GLSL shaders (no footage, no models, no textures
except procedurally generated noise), then run through a filmic post chain and encoded
together with the soundtrack.

    python3 render.py --preset 1080p           # full film, 1920x1080
    python3 render.py --preset 4k              # full film, 3840x2160
    python3 render.py --preset preview         # quick low-res pass
    python3 render.py --still 95.0             # a single PNG frame at t=95s
    python3 render.py --shots 7,8              # re-render only some shots

Rendering is resumable: finished shots are cached in ./render_cache/<preset>/ and skipped.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shots as S  # noqa: E402

FPS = 24
PRESETS = {
    #            width  height samples-scale
    'draft': (640, 360, 0.0),
    'preview': (960, 540, 0.0),
    '720p': (1280, 720, 0.5),
    '1080p': (1920, 1080, 1.0),
    '4k': (3840, 2160, 0.5),
}
ASPECT = 2.39


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def make_noise(n=128, seed=7):
    cache = os.path.join(HERE, 'render_cache', f'noise{n}_{seed}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    filt = np.exp(-0.5 * (TWO_PI_S * kk * 3.2) ** 2)
    out = np.zeros((n, n, n, 4), np.float32)
    for c in range(4):
        w = rng.normal(size=(n, n, n))
        x = np.real(np.fft.ifftn(np.fft.fftn(w) * filt))
        x = (x - x.mean()) / x.std()
        out[..., c] = np.clip(0.5 + 0.22 * x, 0.0, 1.0)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, out)
    return out


TWO_PI_S = 2 * np.pi

VERT = """#version 330 core
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""


class Engine:
    def __init__(self, width, height, sample_scale):
        self.W, self.H = width, height
        self.Ha = int(round(width / ASPECT / 2)) * 2
        self.sample_scale = sample_scale
        if sys.platform.startswith('linux'):
            try:
                self.ctx = moderngl.create_standalone_context(require=330, backend='egl')
            except Exception:
                self.ctx = moderngl.create_standalone_context(require=330)
        else:
            self.ctx = moderngl.create_standalone_context(require=330)
        print(f'  GPU: {self.ctx.info["GL_RENDERER"]}  ({self.ctx.info["GL_VERSION"]})')
        ctx = self.ctx
        self.vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], 'f4').tobytes())
        self.common = open(os.path.join(HERE, 'shaders', 'common.glsl')).read()
        self.post_src = open(os.path.join(HERE, 'shaders', 'post.glsl')).read()
        self.scene_progs = {}
        self.post = {}
        for p in ['ACCUM', 'DOWN', 'BLUR', 'STREAK', 'RAYS', 'FINAL']:
            prog = ctx.program(vertex_shader=VERT,
                               fragment_shader='#version 330 core\n#define PASS_%s\n' % p + self.post_src)
            self.post[p] = (prog, ctx.vertex_array(prog, [(self.vbo, '2f', 'in_pos')]))

        W, Ha = self.W, self.Ha

        def tex(w, h, comps=4, dtype='f2'):
            t = ctx.texture((max(1, w), max(1, h)), comps, dtype=dtype)
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            return t

        self.t_scene = tex(W, Ha, 4, 'f4')
        self.f_scene = ctx.framebuffer([self.t_scene])
        self.t_acc = [tex(W, Ha, 4, 'f4'), tex(W, Ha, 4, 'f4')]
        self.f_acc = [ctx.framebuffer([t]) for t in self.t_acc]
        self.levels = []
        for i in range(1, 6):
            a = tex(W >> i, Ha >> i)
            b = tex(W >> i, Ha >> i)
            self.levels.append((a, ctx.framebuffer([a]), b, ctx.framebuffer([b])))
        self.t_streak = [tex(W >> 3, Ha >> 1), tex(W >> 3, Ha >> 1)]
        self.f_streak = [ctx.framebuffer([t]) for t in self.t_streak]
        self.t_rays = tex(W >> 1, Ha >> 1)
        self.f_rays = ctx.framebuffer([self.t_rays])
        self.t_final = ctx.texture((self.W, self.H), 4, dtype='f1')
        self.f_final = ctx.framebuffer([self.t_final])

        noise = make_noise()
        self.t_noise = ctx.texture3d(noise.shape[:3], 4, noise.astype(np.float16).tobytes(), dtype='f2')
        self.t_noise.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_noise.repeat_x = self.t_noise.repeat_y = self.t_noise.repeat_z = True

        self.t_text = ctx.texture((self.W, self.H), 4, dtype='f1')
        self.t_text.build_mipmaps()
        self.text_key = None
        self.fonts = {}

    # ------------------------------------------------------------------ helpers
    def scene_prog(self, name):
        if name not in self.scene_progs:
            src = open(os.path.join(HERE, 'shaders', name + '.glsl')).read()
            prog = self.ctx.program(vertex_shader=VERT, fragment_shader='#version 330 core\n' + self.common + src)
            self.scene_progs[name] = (prog, self.ctx.vertex_array(prog, [(self.vbo, '2f', 'in_pos')]))
        return self.scene_progs[name]

    @staticmethod
    def set_uniforms(prog, values):
        for k, v in values.items():
            if k not in prog:
                continue
            u = prog[k]
            if isinstance(v, np.ndarray) and v.shape == (3, 3):
                u.value = tuple(float(x) for x in v.flatten(order='F'))
            elif isinstance(v, list) and v and isinstance(v[0], (tuple, list, np.ndarray)):
                u.value = [tuple(float(x) for x in e) for e in v]
            elif isinstance(v, list):
                u.value = [float(x) for x in v]
            elif isinstance(v, (tuple, list, np.ndarray)):
                u.value = tuple(float(x) for x in v)
            else:
                u.value = float(v) if not isinstance(u.value, int) else int(v)

    def font(self, name, size):
        key = (name, size)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(os.path.join(HERE, 'fonts', name), size)
        return self.fonts[key]

    def update_text(self, items):
        """items: list of (text, font, size@1080p, y_center_frac, tracking_em, alpha)."""
        key = repr(items)
        if key == self.text_key:
            return
        self.text_key = key
        s = self.H / 1080.0
        img = Image.new('RGBA', (self.W, self.H), (0, 0, 0, 0))
        for text, fname, size, yc, tracking, alpha in items:
            if alpha <= 0:
                continue
            layer = Image.new('RGBA', (self.W, self.H), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            f = self.font(fname, max(8, int(size * s)))
            track = tracking * size * s
            widths = [f.getlength(ch) for ch in text]
            total = sum(widths) + track * (len(text) - 1)
            x = (self.W - total) / 2
            asc, desc = f.getmetrics()
            y = yc * self.H - (asc + desc) / 2
            for ch, w in zip(text, widths):
                d.text((x, y), ch, font=f, fill=(255, 255, 255, int(255 * alpha)))
                x += w + track
            img = Image.alpha_composite(img, layer)
        self.t_text.write(img.tobytes())
        self.t_text.build_mipmaps()

    def run(self, key, fbo, uniforms, textures):
        prog, vao = self.post[key]
        fbo.use()
        for i, (name, t) in enumerate(textures.items()):
            t.use(location=i)
            if name in prog:
                prog[name].value = i
        self.set_uniforms(prog, uniforms)
        vao.render(moderngl.TRIANGLES)

    # ------------------------------------------------------------------ frame
    def render_frame(self, t, shot, frame_index, samples=None):
        W, Ha = self.W, self.Ha
        if samples is None:
            samples = max(1, int(round(shot.samples * self.sample_scale))) if self.sample_scale > 0 else 1
        shutter = 0.5 / FPS
        prog, vao = self.scene_prog(shot.scene)
        post_u = None
        for k in range(samples):
            u = (k + 0.5) / samples - 0.5 if samples > 1 else 0.0
            ts = t + u * shutter
            U = S.Uniforms()
            shot.fn(ts, ts - shot.start, U)
            if post_u is None or k == samples // 2:
                pu = S.Uniforms()
                shot.fn(t, t - shot.start, pu)
                post_u = pu
            vals = dict(U.scene)
            vals.update(U.camera_uniforms())
            vals['uRes'] = (W, Ha)
            vals['uTime'] = ts
            vals['uLocal'] = ts - shot.start
            if samples > 1:
                vals['uJitter'] = (halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5)
                r = np.sqrt(halton(k + 1, 5))
                a = 2 * np.pi * halton(k + 1, 7)
                vals['uLens'] = (r * np.cos(a), r * np.sin(a))
            else:
                vals['uJitter'] = (0.0, 0.0)
                vals['uLens'] = (0.0, 0.0)
                vals['uAperture'] = 0.0
            self.f_scene.use()
            self.t_noise.use(location=0)
            if 'uNoise' in prog:
                prog['uNoise'].value = 0
            self.set_uniforms(prog, vals)
            # render in strips (keeps each GPU submission short -> no watchdog resets)
            strips = 4 if Ha >= 700 else 1
            sh = (Ha + strips - 1) // strips
            for s in range(strips):
                self.f_scene.scissor = (0, s * sh, W, min(sh, Ha - s * sh))
                vao.render(moderngl.TRIANGLES)
                self.ctx.finish()
            self.f_scene.scissor = None
            dst = k % 2
            self.run('ACCUM', self.f_acc[dst], {'uW': 1.0 / samples, 'uFirst': 1.0 if k == 0 else 0.0},
                     {'uSrc': self.t_scene, 'uPrev': self.t_acc[1 - dst]})
        hdr = self.t_acc[(samples - 1) % 2]
        P = post_u.post

        # bloom pyramid
        src = hdr
        for i, (a, fa, b, fb) in enumerate(self.levels):
            self.run('DOWN', fa, {'uSrcTexel': (1.0 / src.width, 1.0 / src.height), 'uOutRes': a.size,
                                  'uThreshold': P['bloom_threshold'] if i == 0 else 0.0}, {'uSrc': src})
            if i == 0 and P['streak'] > 0:
                # anamorphic streak from the unblurred bright pass: thin vertically, very wide
                self.run('STREAK', self.f_streak[0], {'uSrcTexel': (1.0 / a.width, 1.0 / a.height),
                                                      'uOutRes': self.t_streak[0].size, 'uSpread': 3.0}, {'uSrc': a})
                self.run('STREAK', self.f_streak[1], {'uSrcTexel': (1.0 / self.t_streak[0].width, 1.0 / a.height),
                                                      'uOutRes': self.t_streak[1].size, 'uSpread': 3.0},
                         {'uSrc': self.t_streak[0]})
            self.run('BLUR', fb, {'uSrcTexel': (1.0 / a.width, 1.0 / a.height), 'uOutRes': a.size, 'uDir': (1.0, 0.0)}, {'uSrc': a})
            self.run('BLUR', fa, {'uSrcTexel': (1.0 / a.width, 1.0 / a.height), 'uOutRes': a.size, 'uDir': (0.0, 1.0)}, {'uSrc': b})
            src = a
        # god rays from the bright pass
        l1 = self.levels[0][0]
        self.run('RAYS', self.f_rays, {'uOutRes': self.t_rays.size, 'uLight': P['light'], 'uDecay': P['ray_decay']},
                 {'uSrc': l1})

        self.update_text(P['text'])
        fin = {
            'uFrameRes': (self.W, self.H), 'uActiveH': float(Ha), 'uOutRes': (W, Ha),
            'uExposure': P['exposure'], 'uBloom': P['bloom'], 'uStreak': P['streak'], 'uStreakTint': P['streak_tint'],
            'uRays': P['rays'], 'uLight': P['light'], 'uFlare': P['flare'], 'uFlareTint': P['flare_tint'],
            'uTint': P['tint'], 'uSat': P['sat'], 'uContrast': P['contrast'], 'uShadowTint': P['shadow_tint'],
            'uHighTint': P['high_tint'], 'uVignette': P['vignette'], 'uCA': P['ca'], 'uGrain': P['grain'],
            'uFade': P['fade'], 'uFlash': P['flash'], 'uFlashCol': P['flash_col'],
            'uTextAlpha': 1.0 if P['text'] else 0.0, 'uFrame': float(frame_index),
        }
        self.run('FINAL', self.f_final, fin, {
            'uSrc': hdr, 'uB1': self.levels[1][0], 'uB2': self.levels[2][0], 'uB3': self.levels[3][0],
            'uB4': self.levels[4][0], 'uStreakTex': self.t_streak[1], 'uRaysTex': self.t_rays, 'uText': self.t_text})
        return self.f_final.read(components=3, alignment=1)


# ---------------------------------------------------------------------- encoding
def encoder(path, W, H, crf):
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-tune', 'film',
           '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-x264-params', 'keyint=48:min-keyint=24', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def fmt_time(s):
    s = int(s)
    return f'{s // 3600}:{(s // 60) % 60:02d}:{s % 60:02d}'


def shot_hash(shot):
    import inspect
    files = [os.path.join('shaders', 'common.glsl'), os.path.join('shaders', 'post.glsl'),
             os.path.join('shaders', shot.scene + '.glsl')]
    h = hashlib.sha1()
    for f in files:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    h.update(inspect.getsource(shot.fn).encode())
    h.update(repr((shot.name, shot.start, shot.end, shot.samples)).encode())
    return h.hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='1080p', choices=list(PRESETS))
    ap.add_argument('--shots', default='')
    ap.add_argument('--still', type=float, nargs='*')
    ap.add_argument('--samples', type=int, default=None, help='override samples per frame')
    ap.add_argument('--out', default=None)
    ap.add_argument('--audio', default=os.path.join(HERE, 'lodestar.mp3'))
    ap.add_argument('--no-mux', action='store_true')
    args = ap.parse_args()

    W, H, scale = PRESETS[args.preset]
    print(f'LODESTAR renderer  |  preset {args.preset}  {W}x{H} @ {FPS} fps')
    eng = Engine(W, H, scale)

    if args.still:
        os.makedirs(os.path.join(HERE, 'stills'), exist_ok=True)
        for t in args.still:
            shot = S.shot_at(t)
            t0 = time.time()
            data = eng.render_frame(t, shot, int(t * FPS), samples=args.samples)
            img = Image.frombytes('RGB', (W, H), data)
            p = os.path.join(HERE, 'stills', f'still_{t:07.2f}.png')
            img.save(p)
            print(f'  {p}  [{shot.name}]  {time.time() - t0:.1f}s')
        return

    wanted = set(int(x) for x in args.shots.split(',') if x.strip()) if args.shots else None
    cache = os.path.join(HERE, 'render_cache', args.preset)
    os.makedirs(cache, exist_ok=True)
    crf = 14 if W >= 1920 else 18
    total_frames = sum(s.frame_range()[1] - s.frame_range()[0] for s in S.SHOTS)
    done_frames = 0
    t_start = time.time()
    rendered_frames = 0
    segs = []
    for i, shot in enumerate(S.SHOTS, 1):
        f0, f1 = shot.frame_range()
        seg = os.path.join(cache, f'{i:02d}_{shot.name}_{shot_hash(shot)}.mp4')
        segs.append(seg)
        if os.path.exists(seg) or (wanted is not None and i not in wanted):
            done_frames += f1 - f0
            continue
        print(f'\n[{i:02d}/{len(S.SHOTS)}] {shot.name}  ({(f1 - f0) / FPS:.1f}s, {f1 - f0} frames)')
        tmp = seg + '.part.mp4'
        enc = encoder(tmp, W, H, crf)
        for f in range(f0, f1):
            t = f / FPS
            data = eng.render_frame(t, shot, f)
            enc.stdin.write(data)
            done_frames += 1
            rendered_frames += 1
            el = time.time() - t_start
            rate = el / rendered_frames
            remaining = (total_frames - done_frames) * rate
            pct = 100.0 * done_frames / total_frames
            bar = '#' * int(pct / 2.5) + '-' * (40 - int(pct / 2.5))
            print(f'\r  [{bar}] {pct:5.1f}%  frame {f + 1}/{f1}  {rate:5.2f}s/frame  ETA {fmt_time(remaining)}   ',
                  end='', flush=True)
        enc.stdin.close()
        enc.wait()
        os.replace(tmp, seg)
    print()
    if args.no_mux or wanted is not None:
        return
    # concatenate shots and add the soundtrack
    lst = os.path.join(cache, 'concat.txt')
    with open(lst, 'w') as f:
        for s in segs:
            f.write(f"file '{os.path.basename(s)}'\n")
    out = args.out or os.path.join(HERE, f'LODESTAR_{args.preset}.mp4')
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst]
    if os.path.exists(args.audio):
        cmd += ['-i', args.audio, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '320k']
    cmd += ['-c:v', 'copy', '-movflags', '+faststart', '-shortest', out]
    subprocess.run(cmd, check=True)
    print(f'\nDone!  ->  {out}   (total {fmt_time(time.time() - t_start)})')


if __name__ == '__main__':
    main()
