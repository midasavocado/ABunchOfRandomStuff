"""Cast-shell link meshes for the s07 robot (system python3: numpy + skimage). Each link is modelled in its own joint
frame (see lib/robot.py) so it can be parented to that joint. Units: metres.
usage: python3 sdfrobot.py [voxel_mm]   -> assets/robot/<link>.npz"""
import os, sys, math
import numpy as np
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdfpart import smin, smax, round_cone, round_box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "robot")
H1, A1, L2, A3, L3, L6 = 0.42, 0.10, 0.58, 0.08, 0.56, 0.10


def cyl(P, c, axis, r, hw, rr=0.0):
    """capped cylinder with rounded edge rr; axis in 'xyz'."""
    q = P - np.asarray(c)
    i = 'xyz'.index(axis)
    a = q[:, i]
    rad = np.sqrt(np.sum(np.delete(q, i, axis=1) ** 2, axis=1))
    d_r = rad - (r - rr); d_a = np.abs(a) - (hw - rr)
    return np.minimum(np.maximum(d_r, d_a), 0) + np.sqrt(np.maximum(d_r, 0) ** 2 + np.maximum(d_a, 0) ** 2) - rr


def flat_cone(P, a, b, r1, r2, sy=1.0):
    """round cone with the cross-section squashed along y by sy (<1 = flatter sides)."""
    Q = P.copy(); Q[:, 1] = Q[:, 1] / sy
    a = np.array(a, float); b = np.array(b, float)
    a[1] /= sy; b[1] /= sy
    return round_cone(Q, a, b, r1, r2) * min(1.0, sy)


def turret(P):
    d = cyl(P, (0, 0, 0.045), 'z', 0.152, 0.045, 0.02)
    d = smin(d, flat_cone(P, (0.02, 0, 0.06), (A1, 0, 0.21), 0.14, 0.12, 0.95), 0.05)
    # shoulder yoke: two cheeks around the upper-arm hub (|y| 0.100 .. 0.150)
    ch = cyl(P, (A1, 0, 0.21), 'y', 0.135, 0.150, 0.02)
    d = smin(d, ch, 0.04)
    slot = np.maximum(np.abs(P[:, 1]) - 0.101, -(P[:, 2] - 0.12))
    slot = np.maximum(slot, -(P[:, 0] - A1 + 0.2))
    d = smax(d, -slot, 0.006)
    return d


def upper(P):
    hub = cyl(P, (0, 0, 0), 'y', 0.118, 0.097, 0.015)
    body = flat_cone(P, (0.0, 0, 0.02), (0.0, 0, L2 - 0.02), 0.100, 0.082, 0.82)
    # slight forward belly for a cast look
    body = smin(body, flat_cone(P, (0.018, 0, 0.18), (0.012, 0, 0.42), 0.085, 0.075, 0.8), 0.06)
    d = smin(hub, body, 0.05)
    # elbow yoke (cheeks |y| 0.068 .. 0.097)
    ey = cyl(P, (0, 0, L2), 'y', 0.098, 0.097, 0.015)
    d = smin(d, ey, 0.05)
    slot = np.maximum(np.abs(P[:, 1]) - 0.069, -(P[:, 2] - (L2 - 0.075)))
    d = smax(d, -slot, 0.005)
    return d


def fore(P):
    hub = cyl(P, (0, 0, 0), 'y', 0.090, 0.066, 0.012)
    blk = flat_cone(P, (0.0, 0, 0.0), (0.02, 0, A3), 0.082, 0.078, 0.8)
    body = flat_cone(P, (0.03, 0, A3), (L3 * 0.55, 0, A3), 0.074, 0.062, 0.92)
    d = smin(smin(hub, blk, 0.04), body, 0.05)
    # roll seam at the J4 end: slight recess ring
    d = smax(d, -(np.abs(P[:, 0] - (L3 * 0.55 - 0.004)) - 0.0012), 0.001) if False else d
    return d


def wrist(P):
    tube = flat_cone(P, (0.0, 0, 0), (L3 * 0.45 - 0.02, 0, 0), 0.060, 0.055, 1.0)
    fork = cyl(P, (L3 * 0.45, 0, 0), 'y', 0.056, 0.066, 0.01)
    d = smin(tube, fork, 0.03)
    slot = np.maximum(np.abs(P[:, 1]) - 0.0445, -(P[:, 0] - (L3 * 0.45 - 0.06)))
    d = smax(d, -slot, 0.004)
    return d


def hand(P):
    hub = cyl(P, (0, 0, 0), 'y', 0.050, 0.043, 0.008)
    body = flat_cone(P, (0.0, 0, 0), (L6 - 0.012, 0, 0), 0.048, 0.044, 0.95)
    return smin(hub, body, 0.025)


LINKS = {
    "turret": (turret, (-0.18, -0.17, -0.01), (0.26, 0.17, 0.37)),
    "upper": (upper, (-0.14, -0.12, -0.14), (0.14, 0.12, L2 + 0.12)),
    "fore": (fore, (-0.11, -0.09, -0.11), (L3 * 0.55 + 0.02, 0.09, A3 + 0.09)),
    "wrist": (wrist, (-0.07, -0.08, -0.07), (L3 * 0.45 + 0.07, 0.08, 0.07)),
    "hand": (hand, (-0.06, -0.06, -0.06), (L6 + 0.01, 0.06, 0.06)),
}


def make(name, vox):
    fn, lo, hi = LINKS[name]
    lo = np.array(lo) + vox * 0.37; hi = np.array(hi)
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    xs = lo[0] + np.arange(shape[0]) * vox
    ys = lo[1] + np.arange(shape[1]) * vox
    zs = lo[2] + np.arange(shape[2]) * vox
    vol = np.empty(shape, np.float32)
    step = max(1, int(1.5e6 // (shape[1] * shape[2])))
    for i in range(0, shape[0], step):
        X, Y, Z = np.meshgrid(xs[i:i + step], ys, zs, indexing='ij')
        vol[i:i + step] = fn(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)).reshape(X.shape)
    vol = np.pad(vol, 1, constant_values=1.0)
    V, F, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox,) * 3)
    V = V - vox + lo
    np.savez_compressed(os.path.join(OUT, name + ".npz"), V=V.astype(np.float32), F=F[:, ::-1].astype(np.int32))
    print(name, len(V))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    vox = float(sys.argv[1]) * 0.001 if len(sys.argv) > 1 else 0.0015
    for n in LINKS:
        make(n, vox)
