"""SDF child hand -> marching cubes mesh (run with system python3: numpy + scikit-image).
Right hand. Local frame: wrist centre at origin, fingers along +Y, back of hand +Z (palm faces -Z), thumb on -X.
usage: python3 sdfhand.py <pose> <out.npz> [voxel_mm]"""
import sys, math, json
import numpy as np
from skimage import measure

# ------------------------------------------------------------------ math helpers

def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot(axis, ang):
    axis = norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def sd_round_cone(P, a, b, r1, r2):
    a = np.asarray(a); b = np.asarray(b)
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    w = pa * l2 - y[:, None] * ba[None, :]
    x2 = np.einsum('ij,ij->i', w, w)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    d_b = np.sqrt(x2 + z2) * il2 - r2
    d_a = np.sqrt(x2 + y2) * il2 - r1
    d = np.where(np.sign(z) * a2 * z2 > k, d_b, np.where(np.sign(y) * a2 * y2 < k, d_a, d_mid))
    return d


def sd_ellipsoid(P, c, r):
    q = (P - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_round_box(P, c, b, r):
    q = np.abs(P - np.asarray(c)) - np.asarray(b)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r

# ------------------------------------------------------------------ skeleton

S = 1.0
# name: (mcp position, base direction, phalanx lengths, radii at mcp/pip/dip/tip)
FINGERS = {
    "index":  ((-0.019, 0.066, 0.001), (-0.10, 1, 0), (0.033, 0.020, 0.016), (0.0080, 0.0071, 0.0063, 0.0056)),
    "middle": ((-0.003, 0.069, 0.002), (0.0, 1, 0),   (0.037, 0.023, 0.017), (0.0083, 0.0073, 0.0064, 0.0057)),
    "ring":   ((0.013, 0.066, 0.001),  (0.08, 1, 0),  (0.034, 0.021, 0.016), (0.0078, 0.0068, 0.0060, 0.0053)),
    "pinky":  ((0.027, 0.058, -0.001), (0.18, 1, 0),  (0.026, 0.016, 0.014), (0.0068, 0.0060, 0.0053, 0.0047)),
}
THUMB = ((-0.019, 0.012, -0.004), (0.037, 0.029, 0.024), (0.0120, 0.0097, 0.0082, 0.0070))


def finger_chain(name, flex, splay=0.0):
    """flex: (mcp, pip, dip) radians. Returns joints [mcp, pip, dip, tip], frames (dir, up)."""
    mcp, bdir, lens, rad = FINGERS[name]
    d = norm(bdir)
    up = np.array([0, 0, 1.0])
    d = rot(up, -splay) @ d
    side = norm(np.cross(d, up))
    pts = [np.array(mcp)]
    frames = []
    for L, f in zip(lens, flex):
        R = rot(side, -f)          # flex toward palm (-Z)
        d = R @ d
        up = R @ up
        frames.append((d.copy(), up.copy()))
        pts.append(pts[-1] + d * L)
    return pts, frames, rad


def thumb_chain(yaw, pitch, flex, roll=0.0):
    """Thumb with a 2-DOF CMC: metacarpal direction from yaw (about Z, + toward the fingers' side) and
    pitch (+ toward palm, -Z). roll = opposition (pronation of the thumb). flex = (cmc_extra, mcp, ip)."""
    base, lens, rad = THUMB
    d0 = np.array([-0.62, 0.78, 0.0])
    d = rot((0, 0, 1), -yaw) @ d0
    side0 = norm(np.cross(d, (0, 0, 1)))
    d = rot(side0, -pitch) @ d
    up = norm(np.cross(side0, d))            # dorsal of thumb before opposition
    up = rot(d, roll) @ up
    side = norm(np.cross(d, up))
    pts = [np.array(base)]
    frames = []
    for L, f in zip(lens, flex):
        R = rot(side, -f)
        d = R @ d
        up = R @ up
        frames.append((d.copy(), up.copy()))
        pts.append(pts[-1] + d * L)
    return pts, frames, rad


# ------------------------------------------------------------------ poses

# pencil in hand frame: rests on the thumb-index web, runs forward/down to the paper
PEN_W = np.array([-0.028, 0.044, 0.021])
PEN_D = norm((0.12, 0.70, -0.70))
PEN_R = 0.0038


def fit_pencil():
    """Search the pencil rest point so thumb, index and middle all make contact."""
    global PEN_W
    import itertools, io, contextlib
    base = PEN_W.copy()
    best = None
    for dx, dy, dz in itertools.product((-0.006, -0.002, 0.002), (-0.008, -0.003, 0.002, 0.007), (-0.006, -0.002, 0.002)):
        PEN_W = base + np.array([dx, dy, dz])
        with contextlib.redirect_stdout(io.StringIO()):
            _, e1 = solve_finger_e("index", pen_pt(0.050, (0, 0.3, 1.0)), "pad")
            _, e2 = solve_finger_e("middle", pen_pt(0.058, (0.2, 0, -1.0)), "side", 0.02)
            _, e3 = solve_thumb(pen_pt(0.043, (-1.0, 0, -0.3)))
        tot = e1 + e2 + e3
        if best is None or tot < best[0]:
            best = (tot, PEN_W.copy(), (e1, e2, e3))
    PEN_W = best[1]
    print("pencil fit", best[0], best[2], PEN_W)


def solve_finger_e(*a, **k):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = solve_finger(*a, **k)
    return r, float(buf.getvalue().split()[-1])


def pen_pt(s_, offset_dir):
    """point on the pencil surface at distance s_ from the web point, offset around the shaft."""
    c = PEN_W + PEN_D * s_
    o = np.asarray(offset_dir, float)
    o = o - PEN_D * (o @ PEN_D)
    return c + norm(o) * PEN_R


KNOB_C = np.array([-0.016, 0.066, -0.040])

POSES = {
    "relaxed": dict(index=((0.25, 0.35, 0.2), 0.02), middle=((0.3, 0.4, 0.25), 0.0), ring=((0.35, 0.45, 0.25), -0.03),
                    pinky=((0.4, 0.5, 0.3), -0.08), thumb=(0.1, 0.3, (0.1, 0.2, 0.2), 0.4)),
    # tripod pencil grip: index pad on top, thumb pad on the radial side, pencil resting on the middle finger
    "pencil": lambda: dict(index=("pad", pen_pt(0.050, (0, 0.3, 1.0))),
                   middle=("side", pen_pt(0.058, (0.2, 0, -1.0)), 0.02),
                   ring=("pad", (0.020, 0.080, -0.040), -0.03),
                   pinky=("pad", (0.034, 0.066, -0.036), -0.10),
                   thumb=("pad", pen_pt(0.043, (-1.0, 0, -0.3)))),
    # pinching a focus knob (radius 17 mm, centre KNOB_C, axis = hand-local Z) on its knurled rim
    "knob": lambda: dict(index=("pad", KNOB_C + np.array([0.0, 0.0172, 0.0])),
                         middle=("pad", KNOB_C + np.array([0.0125, 0.0118, -0.001]), 0.02),
                         ring=("pad", (0.022, 0.066, -0.036), -0.03),
                         pinky=("pad", (0.033, 0.055, -0.030), -0.10),
                         thumb=("pad", KNOB_C + np.array([-0.006, -0.0162, -0.001]))),
}


def pad_point(pts, frames, rad):
    d, up = frames[2]
    return pts[2] + (pts[3] - pts[2]) * 0.72 - up * rad[3] * 0.85


def side_point(pts, frames, rad, sign=1.0):
    """point on the radial(+)/ulnar side of the distal phalanx (for the pencil resting on the middle finger)."""
    d, up = frames[2]
    side = norm(np.cross(d, up))
    return pts[2] + (pts[3] - pts[2]) * 0.55 - side * rad[3] * 0.95 * sign - up * rad[3] * 0.2


def solve_finger(name, target, mode="pad", splay0=0.0, dip_ratio=0.7):
    from scipy.optimize import minimize
    target = np.asarray(target)
    def f(x):
        m, p, sp = x
        pts, fr, rad = finger_chain(name, (m, p, p * dip_ratio), sp)
        q = pad_point(pts, fr, rad) if mode == "pad" else side_point(pts, fr, rad)
        return np.sum((q - target) ** 2) * 1e6 + 0.02 * (sp - splay0) ** 2
    best = None
    for m0 in (0.3, 0.8, 1.2):
        r = minimize(f, (m0, 0.6, splay0), bounds=[(-0.2, 1.5), (0.0, 1.75), (splay0 - 0.35, splay0 + 0.35)])
        if best is None or r.fun < best.fun:
            best = r
    m, p, sp = best.x
    err = math.sqrt(max(best.fun, 0) / 1e6) * 1000
    print(name, "ik err mm", round(err, 2))
    return ((m, p, p * dip_ratio), sp)


def solve_thumb(target, pen_axis=None):
    from scipy.optimize import minimize
    target = np.asarray(target)
    def cost(x):
        yw, pt, rl, c, m, i = x
        pts, fr, rad = thumb_chain(yw, pt, (c, m, i), rl)
        e = np.sum((pad_point(pts, fr, rad) - target) ** 2) * 1e6
        # pad should face the pencil: -up (pad normal) toward the target direction from the phalanx centre
        d, up = fr[2]
        if pen_axis is not None:
            e += 2.0 * (1 + (up @ norm(target - (pts[2] + pts[3]) / 2))) ** 2 * 0.0
        return e
    best = None
    for y0 in (0.0, 0.5, 1.0):
        for p0 in (0.3, 0.8):
            r = minimize(cost, (y0, p0, 0.8, 0.1, 0.3, 0.3),
                         bounds=[(-0.4, 1.4), (-0.2, 1.4), (0.0, 1.8), (-0.1, 0.5), (0.0, 1.0), (0.0, 1.2)])
            if best is None or r.fun < best.fun:
                best = r
    yw, pt, rl, c, m, i = best.x
    return (yw, pt, (c, m, i), rl), math.sqrt(max(best.fun, 0) / 1e6) * 1000


def resolve_pose(pose):
    """Pose entries may be ('pad', target) / ('side', target) for IK, thumb may be ('pad', target)."""
    out = {}
    for n in ("index", "middle", "ring", "pinky"):
        e = pose[n]
        if isinstance(e[0], str):
            out[n] = solve_finger(n, e[1], e[0], splay0=e[2] if len(e) > 2 else 0.0)
        else:
            out[n] = e
    t = pose["thumb"]
    if isinstance(t[0], str):
        out["thumb"], err = solve_thumb(t[1])
        print("thumb ik err mm", err)
    else:
        out["thumb"] = t
    return out


def build_sdf(pose):
    pose = resolve_pose(pose)
    fingers = {}
    for n in ("index", "middle", "ring", "pinky"):
        fl, sp = pose[n]
        fingers[n] = finger_chain(n, fl, sp)
    ty, tp, tf, tr = pose["thumb"]
    fingers["thumb"] = thumb_chain(ty, tp, tf, tr)

    def sdf(P):
        x = P[:, 0]; y = P[:, 1]; z = P[:, 2]
        # palm slab with a gentle transverse arch
        Pw = P.copy()
        Pw[:, 2] = z + 18.0 * (x + 0.002) ** 2
        palm = sd_round_box(Pw, (0.002, 0.036, 0.0), (0.019, 0.026, 0.0035), 0.0085)
        # wrist / forearm (elliptical)
        Pf = P.copy(); Pf[:, 0] = x * 0.82
        arm = sd_round_cone(Pf, (0.0, -0.075, -0.001), (0.0, 0.002, 0.0), 0.021, 0.019)
        d = smin(palm, arm, 0.012)
        # thenar / hypothenar pads (palm side)
        d = smin(d, sd_ellipsoid(P, (-0.017, 0.020, -0.006), (0.013, 0.021, 0.010)), 0.008)
        d = smin(d, sd_ellipsoid(P, (0.019, 0.026, -0.005), (0.010, 0.022, 0.008)), 0.008)
        # distal palm pads under the MCPs
        d = smin(d, sd_ellipsoid(P, (0.004, 0.060, -0.005), (0.026, 0.008, 0.006)), 0.006)
        # fingers
        for n in ("index", "middle", "ring", "pinky", "thumb"):
            pts, frames, rad = fingers[n]
            k_base = 0.010 if n != "thumb" else 0.012
            f = None
            for i in range(3):
                a, b = pts[i], pts[i + 1]
                seg = sd_round_cone(P, a, b, rad[i], rad[i + 1])
                f = seg if f is None else smin(f, seg, 0.0025)
            # fingertip pad: slight bulge on the palmar side of the distal phalanx
            dd, up = frames[2]
            padc = pts[2] + (pts[3] - pts[2]) * 0.62 - up * rad[3] * 0.35
            f = smin(f, sd_ellipsoid(P, padc, (rad[3] * 0.9, rad[3] * 0.9, rad[3] * 0.75)), 0.003)
            # dorsal knuckle bump at the MCP (not thumb)
            if n != "thumb":
                d0, up0 = frames[0]
                kn = pts[0] + up0 * rad[0] * 0.35 - d0 * 0.002
                f = smin(f, sd_ellipsoid(P, kn, (rad[0] * 0.62, rad[0] * 0.58, rad[0] * 0.5)), 0.004)
            d = smin(d, f, k_base)
        return d
    return sdf, fingers


def attributes(V, fingers):
    """Per-vertex: R dorsal-crease mask, G palm-side mask, B phase along the nearest joint axis (mm)."""
    n = len(V)
    R = np.zeros(n); G = np.zeros(n); B = np.zeros(n)
    best = np.full(n, 1e9)
    for name, (pts, frames, rad) in fingers.items():
        for j in (1, 2):   # pip, dip (thumb: mcp, ip)
            c = pts[j]
            dvec, up = frames[j - 1]
            dvec2, _ = frames[j]
            axis = norm(dvec + dvec2)
            rel = V - c
            dist = np.linalg.norm(rel, axis=1)
            along = rel @ axis
            dors = rel @ norm(up)
            m = np.clip(1 - dist / (rad[j] * 1.7), 0, 1) * np.clip(dors / rad[j] * 2.0, 0, 1)
            closer = dist < best
            best = np.where(closer, dist, best)
            B = np.where(closer, along * 1000.0, B)
            R = np.maximum(R, m)
    # palm side: normals facing -Z roughly handled in shader; here use position below the mid-plane
    G = np.clip((-V[:, 2] - 0.002) / 0.006, 0, 1) * np.clip((V[:, 1] + 0.01) / 0.02, 0, 1)
    return np.stack([R, G, B], 1)


def nails(fingers):
    out = []
    for name, (pts, frames, rad) in fingers.items():
        dvec, up = frames[2]
        c = pts[2] + (pts[3] - pts[2]) * 0.58 + up * rad[3] * 0.93
        w = rad[3] * (1.45 if name != "thumb" else 1.55)
        L = np.linalg.norm(pts[3] - pts[2]) * 0.62
        out.append(dict(name=name, c=c.tolist(), d=dvec.tolist(), up=up.tolist(), w=float(w), L=float(L)))
    return out


def make(pose_name, voxel=0.0004):
    if pose_name == "pencil":
        fit_pencil()
    pose = POSES[pose_name] if isinstance(pose_name, str) else pose_name
    if callable(pose):
        pose = pose()
    sdf, fingers = build_sdf(pose)
    lo = np.array([-0.075, -0.08, -0.095])
    hi = np.array([0.065, 0.15, 0.05])
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    xs = lo[0] + np.arange(shape[0]) * voxel
    ys = lo[1] + np.arange(shape[1]) * voxel
    zs = lo[2] + np.arange(shape[2]) * voxel
    vol = np.empty(shape, np.float32)
    for i in range(0, shape[0], 16):
        X, Y, Z = np.meshgrid(xs[i:i + 16], ys, zs, indexing='ij')
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        vol[i:i + 16] = sdf(P).reshape(X.shape)
    verts, faces, normals, _ = measure.marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    verts = verts + lo
    # drop the forearm end cap region below the sleeve (keep y > -0.07)
    attr = attributes(verts, fingers)
    return verts.astype(np.float32), faces.astype(np.int32), attr.astype(np.float32), nails(fingers)


if __name__ == "__main__":
    pose = sys.argv[1]
    out = sys.argv[2]
    vox = float(sys.argv[3]) * 0.001 if len(sys.argv) > 3 else 0.0004
    V, F, A, N = make(pose, vox)
    if pose == "pencil":
        N.append(dict(name="__pencil__", c=PEN_W.tolist(), d=PEN_D.tolist(), up=[0, 0, 1], w=PEN_R, L=0.0))
    np.savez_compressed(out, V=V, F=F, A=A, N=json.dumps(N))
    print("verts", len(V), "faces", len(F))
