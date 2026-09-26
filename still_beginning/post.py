"""Per-shot finishing: renders/<sid>/*.png (16-bit) -> grade + bloom + halation + vignette -> edit/<sid>.mov
(H.264 High 4:4:4 10-bit, near-lossless intermediate). Verifies frame count, then deletes the PNGs.
Grain is added only at the final master encode. usage: python3 post.py <sid> [--keep] [--preview]"""
import os, sys, json, subprocess
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import timeline as TL

ROOT = os.path.dirname(os.path.abspath(__file__))
GRADES = os.path.join(ROOT, "grades.json")   # optional per-shot overrides


def load_grade(sid):
    g = dict(bloom=0.05, bloom_thresh=0.72, halation=0.02, vignette=0.18, lift=0.0, gamma=1.0, gain=1.0,
             sat=1.0, warmth=0.0, tint=0.0, contrast=0.0)
    if os.path.exists(GRADES):
        allg = json.load(open(GRADES))
        g.update(allg.get("_default", {}))
        g.update(allg.get(sid, {}))
    return g


def grade(img, g, vig_mask):
    """img: float32 RGB display-referred 0..1 (sRGB encoded from Blender AgX)."""
    x = img
    # bloom: soft glow from highlights (multi-scale)
    if g["bloom"] > 0:
        lum = x.max(axis=2, keepdims=True)
        hi = x * np.clip((lum - g["bloom_thresh"]) / (1 - g["bloom_thresh"] + 1e-6), 0, 1)
        h, w = x.shape[:2]
        acc = np.zeros_like(x)
        small = hi
        for lvl, wt in ((2, 0.35), (4, 0.3), (8, 0.22), (16, 0.13)):
            sm = cv2.resize(hi, (max(1, w // lvl), max(1, h // lvl)), interpolation=cv2.INTER_AREA)
            sm = cv2.GaussianBlur(sm, (0, 0), 2.5)
            acc += cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR) * wt
        x = x + acc * g["bloom"] * 4.0
    # halation: warm red fringe around bright edges (film-like)
    if g["halation"] > 0:
        lum = x.max(axis=2)
        hm = np.clip(lum - 0.85, 0, None)
        hb = cv2.GaussianBlur(hm, (0, 0), max(2.0, x.shape[1] / 900))
        x = x + np.stack([hb * 1.0, hb * 0.35, hb * 0.08], 2) * g["halation"] * 6.0
    # lift / gamma / gain, contrast, warmth, saturation
    x = x * g["gain"] + g["lift"] * (1 - x)
    x = np.clip(x, 0, None) ** (1.0 / g["gamma"])
    if g["contrast"]:
        x = 0.5 + (x - 0.5) * (1 + g["contrast"])
    if g["warmth"] or g["tint"]:
        x = x * np.array([1 + g["warmth"] * 0.5, 1 + g["tint"] * 0.3, 1 - g["warmth"] * 0.5], np.float32)
    if g["sat"] != 1.0:
        l = (x * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
        x = l + (x - l) * g["sat"]
    x = x * vig_mask[..., None]
    # soft shoulder instead of hard clip
    x = np.where(x > 0.9, 0.9 + 0.1 * np.tanh((x - 0.9) / 0.1), x)
    return np.clip(x, 0, 1)


# --------------------------------------------------------------------------- the statement (s28 only)
TITLE_LINES = ("WE\u2019RE JUST", "GETTING STARTED.")
TITLE_FONT = os.path.join(ROOT, "assets", "fonts", "Inter-Medium.ttf")
TITLE_IN = (2745, 2757)        # line 1 with the final D-major arrival (bar 62, 114.375 s), line 2 half a beat later
FADE_OUT = (2856, 2880)        # picture fades to black; the last frame (2879) is black
_TITLE = {}


def title_layers(w, h):
    """Two anti-aliased RGBA layers (one per line), centred in the upper-middle of the frame above the limb.
    Inter Medium, wide tracking (+14 %), warm white; a very soft glow so it sits in the light, not on top of it."""
    key = (w, h)
    if key in _TITLE:
        return _TITLE[key]
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    size = int(h * 0.046)
    font = ImageFont.truetype(TITLE_FONT, size)
    track = size * 0.14
    layers = []
    y0 = int(h * 0.36)
    for i, line in enumerate(TITLE_LINES):
        widths = [font.getlength(ch) for ch in line]
        total = sum(widths) + track * (len(line) - 1)
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        x = (w - total) / 2
        y = y0 + i * int(size * 1.55)
        for ch, cw in zip(line, widths):
            d.text((x, y), ch, font=font, fill=255)
            x += cw + track
        a = np.asarray(im, np.float32) / 255.0
        glow = np.asarray(im.filter(ImageFilter.GaussianBlur(size * 0.35)), np.float32) / 255.0
        layers.append((a, glow))
    _TITLE[key] = layers
    return layers


def apply_title(x, f):
    """x: graded float RGB frame (0..1), f: global frame number."""
    h, w = x.shape[:2]
    col = np.array([0.96, 0.93, 0.88], np.float32)
    for i, (a, glow) in enumerate(title_layers(w, h)):
        t = (f - TITLE_IN[i]) / 30.0
        k = 0.0 if t <= 0 else (1.0 if t >= 1 else t * t * (3 - 2 * t))
        if k <= 0:
            continue
        x = x + glow[..., None] * (0.10 * k) * col                  # soft light halo
        x = x * (1 - a[..., None] * k) + a[..., None] * k * col      # the letters
    t = (f - FADE_OUT[0]) / (FADE_OUT[1] - 1 - FADE_OUT[0])
    if t > 0:
        k = min(1.0, t)
        x = x * (1 - (k * k * (3 - 2 * k)))
    return np.clip(x, 0, 1)


def vignette(h, w, amt):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 * 0.6)
    return (1 - amt * np.clip(r - 0.35, 0, None) ** 2.2).astype(np.float32)


def run(sid, keep=False, preview=False):
    _, a, b, _ = TL.shot(sid)
    hin, hout = TL.handles(sid)                 # handle frames past each cut (the master blends across them)
    a0 = a - hin
    src = os.path.join(ROOT, "preview" if preview else "renders", sid)
    frames = [os.path.join(src, "%04d.png" % f) for f in range(a0, b + hout)]
    missing = [f for f in frames if not os.path.exists(f)]
    if missing:
        print("missing", len(missing)); return False
    g = load_grade(sid)
    im0 = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
    h, w = im0.shape[:2]
    vm = vignette(h, w, g["vignette"])
    os.makedirs(os.path.join(ROOT, "edit"), exist_ok=True)
    out = os.path.join(ROOT, "edit", f"{sid}{'_prev' if preview else ''}.mov")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "6" if not preview else "14", "-pix_fmt", "yuv444p10le",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi, f in enumerate(frames):
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        scale = 65535.0 if im.dtype == np.uint16 else 255.0
        x = grade(im.astype(np.float32) / scale, g, vm)
        if sid == "s28":
            x = apply_title(x, a0 + fi)
        p.stdin.write((x * 65535.0 + 0.5).astype(np.uint16).tobytes())
    p.stdin.close()
    p.wait()
    # verify decoded frame count
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    ok = n.isdigit() and int(n) == len(frames)
    print(sid, "frames", n, "expected", len(frames), "OK" if ok else "FAIL")
    if ok and not keep and not preview:
        # keep first/mid/last stills (8-bit jpg) for review, delete the heavy PNGs
        rv = os.path.join(ROOT, "stills", sid); os.makedirs(rv, exist_ok=True)
        for f in (frames[hin], frames[hin + (b - a) // 2], frames[hin + (b - a) - 1]):
            im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(rv, os.path.basename(f)[:-4] + ".jpg"), (im / 257).astype(np.uint8) if im.dtype == np.uint16 else im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for f in frames:
            os.remove(f)
    return ok


if __name__ == "__main__":
    sid = sys.argv[1]
    run(sid, keep="--keep" in sys.argv, preview="--preview" in sys.argv)
