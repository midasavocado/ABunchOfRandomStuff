"""Prepare the s07a powder-bed slice: the component (v3) lies on its side (part X = build Z), sliced at x = XS.
Writes assets/slm/slice.png (RGB: R = inside mask of all parts on the plate, G = hatch coordinate 0..1 per part,
B = part id / 8) and assets/slm/contour.json (hero part contours in plate mm, ordered, with the outer loop first).
Run with system python3: python3 lib/slmprep.py"""
import os, sys, json, math
import numpy as np
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdfpart as SP
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "slm")
os.makedirs(OUT, exist_ok=True)
XS = 4.55
PLATE = 120.0                  # mm (square)
RES = 0.025                    # mm / px
N = int(PLATE / RES)
# part placements on the plate: (centre x, centre y, rotation deg) in plate mm (plate centre = 0,0)
PARTS = [(-32, 28, 90), (0, 28, 90), (32, 28, 90), (-32, -28, 90), (0, -28, 90), (32, -28, 90)]
HERO = 1
HATCH_DEG = 67.0


def slice_img(step=0.02):
    ys = np.arange(-7.5, 46.5, step)
    zs = np.arange(-7.8, 7.8, step)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    P = np.stack([np.full(Y.size, XS), Y.ravel(), Z.ravel()], 1)
    d = SP.v3(P).reshape(Y.shape)
    return ys, zs, d


def main():
    ys, zs, d = slice_img()
    # contours (part frame y,z)
    cs = measure.find_contours(d, 0.0)
    loops = []
    for c in cs:
        pts = np.stack([ys[0] + c[:, 0] * (ys[1] - ys[0]), zs[0] + c[:, 1] * (zs[1] - zs[0])], 1)
        loops.append(pts)
    loops.sort(key=lambda p: -len(p))
    img = np.zeros((N, N, 3), np.float32)
    gx = (np.arange(N) + 0.5) * RES - PLATE / 2
    GX, GY = np.meshgrid(gx, gx, indexing='xy')         # image row = y, col = x
    hero_loops = None
    for pid, (cx, cy, rot) in enumerate(PARTS):
        a = math.radians(rot)
        # plate -> part frame: part y along plate direction (cos a, sin a), part z perpendicular
        # the part centre (y=20, z=0) at (cx, cy)
        dx, dy = GX - cx, GY - cy
        py = dx * math.cos(a) + dy * math.sin(a) + 20.0
        pz = -dx * math.sin(a) + dy * math.cos(a)
        iy = np.clip(((py - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
        iz = np.clip(((pz - zs[0]) / (zs[1] - zs[0])).astype(int), 0, len(zs) - 1)
        inb = (py > ys[0]) & (py < ys[-1]) & (pz > zs[0]) & (pz < zs[-1])
        inside = inb & (d[iy, iz] < 0)
        h = math.radians(HATCH_DEG)
        s = dx * math.cos(h) + dy * math.sin(h)
        s = (s - s[inside].min()) / (s[inside].max() - s[inside].min() + 1e-9) if inside.any() else s
        img[..., 0] = np.where(inside, 1.0, img[..., 0])
        img[..., 1] = np.where(inside, s, img[..., 1])
        img[..., 2] = np.where(inside, (pid + 0.5) / 8.0, img[..., 2])
        if pid == HERO:
            hero_loops = []
            for L in loops:
                yy, zz = L[:, 0] - 20.0, L[:, 1]
                X = cx + yy * math.cos(a) - zz * math.sin(a)
                Yp = cy + yy * math.sin(a) + zz * math.cos(a)
                hero_loops.append(np.stack([X, Yp], 1).tolist())
    Image.fromarray((np.flipud(img) * 255).astype(np.uint8)).save(os.path.join(OUT, "slice.png"))
    json.dump(dict(parts=PARTS, hero=HERO, plate=PLATE, loops=hero_loops, xs=XS), open(os.path.join(OUT, "contour.json"), "w"))
    print("loops", [len(l) for l in hero_loops])


if __name__ == "__main__":
    main()
