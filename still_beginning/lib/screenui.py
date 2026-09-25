"""Composite the s06 workstation screen: one engineering viewport + a minimal, text-free UI.
Layout (16:10-ish monitor, rendered 16:9): graphite app frame, thin top bar with tool glyphs, the viewport, and a
right-hand iteration rail with three thumbnails that appear as each design converges (mass bar under each, shrinking),
the final one selected with an amber outline. A thin optimisation progress line runs under the viewport.
usage: python3 screenui.py [W]  -> assets/screen_s06/scr_NNNN.jpg for frames 450..539"""
import os, sys, math
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "assets", "screen_s06")
W = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
H = int(W * 0.5625)
AMBER = (242, 150, 58)
BG = (30, 32, 36)
PANEL = (40, 43, 48)
LINE = (70, 74, 80)
TXT = (150, 155, 162)


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def glyph(dr, x, y, s, kind, col):
    if kind == 0:
        dr.rectangle([x, y, x + s, y + s], outline=col, width=max(1, s // 8))
    elif kind == 1:
        dr.ellipse([x, y, x + s, y + s], outline=col, width=max(1, s // 8))
    elif kind == 2:
        dr.polygon([(x + s / 2, y), (x + s, y + s), (x, y + s)], outline=col)
    else:
        for i in range(3):
            dr.line([x, y + i * s / 2, x + s, y + i * s / 2], fill=col, width=max(1, s // 8))


def frame(f):
    u = W / 2048.0
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    top = int(44 * u)
    rail = int(360 * u)
    pad = int(14 * u)
    # top bar
    dr.rectangle([0, 0, W, top], fill=(24, 26, 29))
    for i in range(7):
        glyph(dr, int((22 + i * 44) * u), int(12 * u), int(20 * u), i % 4, TXT if i != 3 else (200, 204, 210))
    for i in range(3):
        dr.ellipse([W - int((40 + i * 30) * u), int(15 * u), W - int((26 + i * 30) * u), int(29 * u)], fill=(60, 63, 68))
    # viewport
    vx0, vy0, vx1, vy1 = pad, top + pad, W - rail - pad, H - int(70 * u)
    vp = Image.open(os.path.join(D, "vp_%04d.png" % f)).convert("RGB")
    vw, vh = vx1 - vx0, vy1 - vy0
    sw, sh = vp.size
    # cover-fit crop
    sc = max(vw / sw, vh / sh)
    vp = vp.resize((int(sw * sc), int(sh * sc)), Image.LANCZOS)
    ox, oy = (vp.size[0] - vw) // 2, (vp.size[1] - vh) // 2
    im.paste(vp.crop((ox, oy, ox + vw, oy + vh)), (vx0, vy0))
    dr.rectangle([vx0, vy0, vx1, vy1], outline=LINE, width=max(1, int(u)))
    # axis gizmo (bottom-left of viewport)
    gx, gy, gl = vx0 + int(50 * u), vy1 - int(50 * u), int(30 * u)
    dr.line([gx, gy, gx + gl, gy + gl * 0.35], fill=(200, 80, 70), width=max(2, int(2 * u)))
    dr.line([gx, gy, gx - gl * 0.7, gy + gl * 0.3], fill=(90, 170, 90), width=max(2, int(2 * u)))
    dr.line([gx, gy, gx, gy - gl], fill=(80, 120, 210), width=max(2, int(2 * u)))
    # optimisation progress line under the viewport: runs while converging, then fills
    py = vy1 + int(24 * u)
    dr.line([vx0, py, vx1, py], fill=(55, 58, 63), width=max(2, int(3 * u)))
    prog = smooth((f - 458) / 54.0)
    dr.line([vx0, py, vx0 + (vx1 - vx0) * prog, py], fill=(185, 190, 196) if prog < 1 else AMBER, width=max(2, int(3 * u)))
    # small convergence plot (mass vs iteration) at the right end of the progress row
    # iteration rail
    rx0 = W - rail
    dr.rectangle([rx0, top, W, H], fill=PANEL)
    tw = rail - 2 * pad
    th = int(tw * 0.62)
    appear = [452, 490, 512]
    mass = [1.0, 0.58, 0.34]
    for i in range(3):
        y = top + pad + i * (th + int(56 * u))
        a = smooth((f - appear[i]) / 8.0)
        if a <= 0:
            dr.rectangle([rx0 + pad, y, rx0 + pad + tw, y + th], outline=(52, 55, 60), width=max(1, int(u)))
            continue
        tn = Image.open(os.path.join(D, "thumb%d.png" % (i + 1))).convert("RGB").resize((tw, th), Image.LANCZOS)
        base = Image.new("RGB", (tw, th), PANEL)
        im.paste(Image.blend(base, tn, a), (rx0 + pad, y))
        sel = i == 2 and f >= 516
        dim = i < 2 and f >= 516
        if dim:
            ov = Image.new("RGB", (tw, th), PANEL)
            k = 0.45 * smooth((f - 516) / 8.0)
            region = im.crop((rx0 + pad, y, rx0 + pad + tw, y + th))
            im.paste(Image.blend(region, ov, k), (rx0 + pad, y))
        col = AMBER if sel else LINE
        wd = max(2, int((3 if sel else 1) * u))
        dr.rectangle([rx0 + pad - wd, y - wd, rx0 + pad + tw + wd, y + th + wd], outline=col, width=wd)
        # mass bar
        by = y + th + int(16 * u)
        dr.rectangle([rx0 + pad, by, rx0 + pad + tw, by + int(6 * u)], fill=(55, 58, 63))
        dr.rectangle([rx0 + pad, by, rx0 + pad + int(tw * mass[i] * a), by + int(6 * u)], fill=AMBER if sel else (150, 155, 162))
        # stiffness dots (3 = equal), neutral
        for k in range(3):
            cx = rx0 + pad + tw - int((10 + k * 16) * u)
            dr.ellipse([cx - int(4 * u), by + int(18 * u), cx + int(4 * u), by + int(26 * u)], fill=(120, 124, 130))
    return im


if __name__ == "__main__":
    for f in range(450, 540):
        if not os.path.exists(os.path.join(D, "vp_%04d.png" % f)):
            continue
        frame(f).save(os.path.join(D, "scr_%04d.jpg" % f), quality=93)
