"""SDF EVA glove -> marching-cubes mesh (.npz). Run with system python3 (numpy, scipy, scikit-image).

Right glove, ADULT metres. Local frame (same convention as sdfhand.py): wrist centre at origin, fingers along +Y,
back of hand +Z (palm faces -Z), thumb on -X. The gauntlet cuff runs to y = CUFF_END (the glove-side disconnect
ring sits there, built in Blender by suit.py). Left glove = mirror in X (done in Blender).

Output npz: V (verts), F (faces), A (per-vertex RGBA: pad mask, dorsalness -1..1, joint-crease weight, region),
R (per-vertex rest-pose position: texture coordinates that stay glued to the fabric while fingers flex),
P (pose json).

usage: python3 sdfglove.py <pose> <out.npz> [voxel_mm]
       python3 sdfglove.py --json '<pose json>' <out.npz> [voxel_mm]
A pose is {"index": [mcp,pip,splay], "middle":..., "ring":..., "pinky":..., "thumb": [yaw,pitch,roll,cmc,mcp,ip],
           "contacts": [ {"type":"cyl","c":[..],"axis":[..],"r":..,"h":..}, ... ]}   (contacts flatten the pads)
"""
import sys, math, json
import numpy as np

S = 1.36            # child skeleton (sdfhand proportions) -> adult
INFL = 0.0040       # glove wall over the skin (m)
PAD_H = 0.0009      # grip-pad relief (m)
CUFF_END = -0.072   # y of the glove-side disconnect ring face (m)
CUFF_R = 0.0450     # cuff radius at the ring (m)

# ------------------------------------------------------------------ math (copied from sdfhand.py, generic)

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


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_round_cone(P, a, b, r1, r2):
    a = np.asarray(a, float); b = np.asarray(b, float)
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
    return np.where(np.sign(z) * a2 * z2 > k, d_b, np.where(np.sign(y) * a2 * y2 < k, d_a, d_mid))


def sd_ellipsoid(P, c, r):
    q = (P - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_oellipsoid(P, c, M, r):
    """ellipsoid with axes = rows of M (3x3 orthonormal) and radii r."""
    Q = (P - np.asarray(c)) @ np.asarray(M).T
    return sd_ellipsoid(Q, (0, 0, 0), r)


def sd_round_box(P, c, b, r):
    q = np.abs(P - np.asarray(c)) - np.asarray(b)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r


def sd_capped_cyl(P, c, axis, r, h):
    """finite cylinder centred at c, unit axis, radius r, half-length h."""
    ax = norm(axis)
    pc = P - np.asarray(c)
    y = pc @ ax
    x = np.linalg.norm(pc - y[:, None] * ax[None, :], axis=1)
    dx = x - r
    dy = np.abs(y) - h
    return np.minimum(np.maximum(dx, dy), 0) + np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)

# ------------------------------------------------------------------ skeleton (child units, sdfhand proportions)

FINGERS = {
    "index":  ((-0.019, 0.066, 0.001), (-0.10, 1, 0), (0.033, 0.020, 0.016), (0.0080, 0.0071, 0.0063, 0.0056)),
    "middle": ((-0.003, 0.069, 0.002), (0.0, 1, 0),   (0.037, 0.023, 0.017), (0.0083, 0.0073, 0.0064, 0.0057)),
    "ring":   ((0.013, 0.066, 0.001),  (0.08, 1, 0),  (0.034, 0.021, 0.016), (0.0078, 0.0068, 0.0060, 0.0053)),
    "pinky":  ((0.027, 0.058, -0.001), (0.18, 1, 0),  (0.026, 0.016, 0.014), (0.0068, 0.0060, 0.0053, 0.0047)),
}
THUMB = ((-0.019, 0.012, -0.004), (0.037, 0.029, 0.024), (0.0120, 0.0097, 0.0082, 0.0070))
NAMES = ("index", "middle", "ring", "pinky")


def finger_chain(name, flex, splay=0.0):
    """adult metres. flex=(mcp,pip,dip). returns pts[4], frames[3] (dir, up), radii[4]"""
    mcp, bdir, lens, rad = FINGERS[name]
    d = norm(bdir)
    up = np.array([0, 0, 1.0])
    d = rot(up, -splay) @ d
    side = norm(np.cross(d, up))
    pts = [np.array(mcp) * S]
    frames = []
    for L, f in zip(lens, flex):
        R = rot(side, -f)
        d = R @ d
        up = R @ up
        frames.append((d.copy(), up.copy()))
        pts.append(pts[-1] + d * L * S)
    return pts, frames, [r * S for r in rad]


def thumb_chain(yaw, pitch, flex, roll=0.0):
    base, lens, rad = THUMB
    d0 = np.array([-0.62, 0.78, 0.0])
    d = rot((0, 0, 1), -yaw) @ d0
    side0 = norm(np.cross(d, (0, 0, 1)))
    d = rot(side0, -pitch) @ d
    up = norm(np.cross(side0, d))
    up = rot(d, roll) @ up
    side = norm(np.cross(d, up))
    pts = [np.array(base) * S]
    frames = []
    for L, f in zip(lens, flex):
        R = rot(side, -f)
        d = R @ d
        up = R @ up
        frames.append((d.copy(), up.copy()))
        pts.append(pts[-1] + d * L * S)
    return pts, frames, [r * S for r in rad]


def chains(pose):
    out = {}
    for n in NAMES:
        m, p, sp = pose[n]
        out[n] = finger_chain(n, (m, p, p * 0.72), sp)
    y, pt, rl, c, m, i = pose["thumb"]
    out["thumb"] = thumb_chain(y, pt, (c, m, i), rl)
    return out


def rest_pose(pose):
    """flat reference pose (same splay) used for rest-coordinates."""
    r = {n: (0.0, 0.0, pose[n][2]) for n in NAMES}
    r["thumb"] = (0.1, 0.25, 0.4, 0.0, 0.0, 0.0)
    return r


def pad_point(pts, frames, rad, seg=2, frac=0.72):
    d, up = frames[seg]
    return pts[seg] + (pts[seg + 1] - pts[seg]) * frac - up * (rad[seg + 1] * 0.85 + INFL + PAD_H * 0.6)


POSES = {
    # pressurised EVA gloves rest half-open
    "relaxed": {"index": (0.30, 0.42, 0.03), "middle": (0.36, 0.48, 0.0), "ring": (0.42, 0.52, -0.03),
                "pinky": (0.48, 0.56, -0.09), "thumb": (0.25, 0.45, 0.7, 0.05, 0.22, 0.25)},
    "open":    {"index": (0.12, 0.18, 0.06), "middle": (0.14, 0.2, 0.0), "ring": (0.18, 0.22, -0.06),
                "pinky": (0.22, 0.26, -0.14), "thumb": (0.05, 0.3, 0.5, 0.0, 0.1, 0.1)},
    "fist":    {"index": (1.2, 1.45, 0.0), "middle": (1.25, 1.5, 0.0), "ring": (1.3, 1.5, -0.02),
                "pinky": (1.3, 1.5, -0.05), "thumb": (0.7, 0.8, 1.1, 0.1, 0.5, 0.5)},
}

# ------------------------------------------------------------------ IK

def solve_finger(name, target, splay0=0.0, seg=2, frac=0.72):
    from scipy.optimize import minimize
    target = np.asarray(target, float)
    def f(x):
        m, p, sp = x
        pts, fr, rad = finger_chain(name, (m, p, p * 0.72), sp)
        q = pad_point(pts, fr, rad, seg, frac)
        return np.sum((q - target) ** 2) * 1e6 + 0.05 * (sp - splay0) ** 2
    best = None
    for m0 in (0.2, 0.7, 1.2):
        r = minimize(f, (m0, 0.6, splay0), bounds=[(-0.25, 1.5), (0.0, 1.8), (splay0 - 0.3, splay0 + 0.3)])
        if best is None or r.fun < best.fun:
            best = r
    return tuple(best.x), math.sqrt(max(best.fun, 0) / 1e6) * 1000


def solve_thumb(target, seg=2, frac=0.72, face=None):
    """face: desired outward normal of the contact surface at target (pad should press against it)."""
    from scipy.optimize import minimize
    target = np.asarray(target, float)
    def cost(x):
        yw, pt, rl, c, m, i = x
        pts, fr, rad = thumb_chain(yw, pt, (c, m, i), rl)
        e = np.sum((pad_point(pts, fr, rad, seg, frac) - target) ** 2) * 1e6
        if face is not None:
            d, up = fr[seg]
            e += 3.0 * (1 - up @ norm(face)) ** 2      # pad normal (-up) opposes the surface normal
        return e
    best = None
    for y0 in (0.0, 0.5, 1.0):
        for p0 in (0.3, 0.9):
            r = minimize(cost, (y0, p0, 0.8, 0.1, 0.3, 0.3),
                         bounds=[(-0.4, 1.4), (-0.2, 1.4), (0.0, 1.8), (-0.1, 0.5), (0.0, 1.0), (0.0, 1.2)])
            if best is None or r.fun < best.fun:
                best = r
    return tuple(best.x), math.sqrt(max(best.fun, 0) / 1e6) * 1000

# ------------------------------------------------------------------ glove SDF

def seg_frame(d, up):
    d = norm(d); up = norm(up - d * (up @ d))
    side = np.cross(d, up)
    return np.array([side, d, up])       # rows: x=side, y=along, z=dorsal


def build(pose):
    C = chains(pose)
    contacts = pose.get("contacts", [])
    prims = []        # (kind, data) for rest mapping: finger segments

    def hand_sdf(P, want_pad=False):
        x = P[:, 0]; y = P[:, 1]; z = P[:, 2]
        # palm slab with transverse arch (adult)
        Pw = P.copy()
        Pw[:, 2] = z + 14.0 * (x + 0.0025) ** 2
        palm = sd_round_box(Pw, (0.0025, 0.046, 0.0), (0.024, 0.033, 0.0045), 0.0105 + INFL)
        # wrist -> cuff: elliptical near the wrist, round at the ring
        Pf = P.copy()
        t = np.clip((y - 0.0) / (CUFF_END), 0, 1)       # 0 at wrist, 1 at ring
        Pf[:, 0] = x * (0.82 + 0.18 * t)
        cuff = sd_round_cone(Pf, (0.0, CUFF_END + 0.004, -0.002), (0.0, 0.006, 0.0), CUFF_R, 0.0285 + INFL)
        # gauntlet roll just above the ring
        cuff = smin(cuff, sd_round_cone(P, (0.0, CUFF_END + 0.010, -0.001), (0.0, CUFF_END + 0.022, -0.001), CUFF_R + 0.0015, CUFF_R - 0.002), 0.006)
        # gauntlet compression folds: few, soft, irregular (domain-warped), strongest near the ring
        th = np.arctan2(z, x)
        reg = np.clip((y - (CUFF_END + 0.010)) / 0.010, 0, 1) * np.clip((-0.006 - y) / 0.016, 0, 1)
        warp = 0.9 * np.sin(th * 2.0 + y * 70.0 + 0.6) + 0.5 * np.sin(th * 5.0 - y * 130.0 + 2.1)
        ph = y * (2 * math.pi / 0.019) + warp
        fold = np.sin(ph) * 0.6 + 0.4 * (1 - np.abs(np.sin(ph * 0.5 + th))) - 0.2
        amp = 0.0009 * np.clip(0.3 + 0.7 * np.sin(th * 1.5 + y * 45.0 + 0.8), 0, 1)
        cuff = cuff - fold * amp * reg
        cuff = np.maximum(cuff, -(y - CUFF_END))          # cut flat at the ring
        d = smin(palm, cuff, 0.016)
        # thenar / hypothenar
        d = smin(d, sd_ellipsoid(P, (-0.021, 0.025, -0.007), (0.017 + INFL, 0.026 + INFL, 0.012 + INFL)), 0.01)
        d = smin(d, sd_ellipsoid(P, (0.024, 0.033, -0.006), (0.012 + INFL, 0.028 + INFL, 0.010 + INFL)), 0.01)
        # dorsal knuckle bar (padded restraint across the MCPs)
        kb = sd_round_cone(P, (-0.026, 0.080, 0.0105), (0.036, 0.072, 0.0085), 0.0062, 0.0055)
        d = smin(d, kb, 0.006)
        pads = None
        for n in ("index", "middle", "ring", "pinky", "thumb"):
            pts, frames, rad = C[n]
            f = None
            for i in range(3):
                a, b = pts[i], pts[i + 1]
                # slightly puffy segments (TMG bladder between joints)
                seg = sd_round_cone(P, a, b, rad[i] + INFL + 0.0004, rad[i + 1] + INFL + 0.0004)
                f = seg if f is None else smin(f, seg, 0.0028)
                dd, up = frames[i]
                M = seg_frame(dd, up)
                L = np.linalg.norm(b - a)
                mid = (a + b) / 2
                # puff: dorsal-lateral bulge in mid segment
                pf = sd_oellipsoid(P, mid + up * 0.0006, M, (rad[i] + INFL + 0.0009, L * 0.42, rad[i] + INFL + 0.0006))
                f = smin(f, pf, 0.002)
                # palmar grip pad on each phalanx
                pc = mid - up * (rad[i + 1] * 0.55)
                pr = (rad[i] * 0.78, L * 0.36, rad[i] * 0.5 + INFL + PAD_H)
                if i == 2:
                    pc = a + (b - a) * 0.62 - up * (rad[3] * 0.45)
                    pr = (rad[3] * 0.9, L * 0.48, rad[3] * 0.62 + INFL + PAD_H)
                pd = sd_oellipsoid(P, pc, M, pr)
                pads = pd if pads is None else np.minimum(pads, pd)
            # blunt rubber fingertip cap
            dd, up = frames[2]
            tip = pts[3] + dd * 0.0006
            f = smin(f, sd_ellipsoid(P, tip - up * 0.0008, (rad[3] + INFL + 0.0006,) * 2 + (rad[3] + INFL + 0.0002,)), 0.003)
            if n != "thumb":
                d0, up0 = frames[0]
                kn = pts[0] + up0 * rad[0] * 0.3
                f = smin(f, sd_ellipsoid(P, kn, (rad[0] * 0.7 + INFL, rad[0] * 0.65 + INFL, rad[0] * 0.55 + INFL)), 0.004)
            d = smin(d, f, 0.012 if n != "thumb" else 0.014)
        # palm grip pads
        pp = sd_round_box(Pw, (0.003, 0.060, -0.0128 - INFL), (0.022, 0.009, 0.0008), 0.0024)
        pp = np.minimum(pp, sd_ellipsoid(P, (-0.021, 0.024, -0.0115 - INFL), (0.012, 0.018, 0.0065)))
        pp = np.minimum(pp, sd_ellipsoid(P, (0.024, 0.032, -0.0108 - INFL), (0.0085, 0.02, 0.0055)))
        pads = np.minimum(pads, pp)
        d = smin(d, pads, 0.0012)
        # contact flattening (soft pads squash against rigid parts; guarantees no intersection)
        for c in contacts:
            if c["type"] == "cyl":
                cd = sd_capped_cyl(P, c["c"], c["axis"], c["r"], c["h"])
            else:
                cd = sd_ellipsoid(P, c["c"], c["r"])
            d = smax(d, -cd + 0.00005, 0.0012)
        if want_pad:
            return d, pads
        return d
    return hand_sdf, C


def rest_coords(V, C, CR):
    """map each vertex into the flat rest pose via its nearest finger segment (soft-blended)."""
    n = len(V)
    cand = [(np.zeros(n), V.copy())]            # palm/cuff: identity, 'distance' 0.004 bias
    dists = [np.full(n, 1e9)]
    # palm distance: approx via y below MCP line
    for name in ("index", "middle", "ring", "pinky", "thumb"):
        pts, fr, rad = C[name]
        pr, frr, radr = CR[name]
        for i in range(3):
            a, b = pts[i], pts[i + 1]
            dseg = sd_round_cone(V, a, b, rad[i], rad[i + 1])
            M = seg_frame(*fr[i]); Mr = seg_frame(*frr[i])
            loc = (V - a) @ M.T
            rp = pr[i] + loc @ Mr
            cand.append((None, rp)); dists.append(dseg)
    D = np.stack(dists, 0)
    # palm pseudo-distance: distance below the knuckle line along fingers
    ymcp = 0.070
    D[0] = np.clip(V[:, 1] - ymcp, 0, None) * 1.5 + 0.004
    thumb_zone = (V[:, 0] < -0.02) & (V[:, 1] < 0.05)
    D[0] = np.where(thumb_zone, np.maximum(D[0], 0.004), D[0])
    dm = D.min(0)
    W = np.exp(-(D - dm[None]) / 0.0012)
    W /= W.sum(0, keepdims=True)
    R = np.zeros_like(V)
    for k, (_, rp) in enumerate(cand):
        R += W[k][:, None] * rp
    return R


def vnormals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def attributes(V, C, hand_sdf, F=None):
    d, pads = hand_sdf(V, want_pad=True)
    pad = np.clip(1 - (pads - 0.0) / 0.0006, 0, 1)
    n = len(V)
    Nv = vnormals(V, F)
    crease = np.zeros(n)
    # soft-blended 'up' (dorsal) field from palm + all segments -> smooth dorsal/palmar mask
    ds = [np.clip(V[:, 1] - 0.070, 0, None) * 1.5 + 0.004]
    ups = [np.tile([0, 0, 1.0], (n, 1))]
    for name, (pts, frames, rad) in C.items():
        for i in range(3):
            a, b = pts[i], pts[i + 1]
            ds.append(sd_round_cone(V, a, b, rad[i], rad[i + 1]))
            ups.append(np.tile(frames[i][1], (n, 1)))
        for j in (1, 2):
            c = pts[j]
            dist = np.linalg.norm(V - c, axis=1)
            crease = np.maximum(crease, np.clip(1 - dist / (rad[j] * 2.2 + INFL), 0, 1))
    D = np.stack(ds, 0)
    W = np.exp(-(D - D.min(0)[None]) / 0.002)
    W /= W.sum(0, keepdims=True)
    U = (W[..., None] * np.stack(ups, 0)).sum(0)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9)
    dors = np.clip((Nv * U).sum(1) * 1.4, -1, 1)
    region = np.clip((-V[:, 1] - 0.01) / 0.02, 0, 1)      # 0 hand .. 1 cuff
    # fingertip rubber caps: distal half of the distal phalanx, all around
    cap = np.zeros(n)
    for name, (pts, frames, rad) in C.items():
        a, b = pts[2], pts[3]
        ab = b - a
        t = ((V - a) @ ab) / (ab @ ab)
        dseg = sd_round_cone(V, a, b, rad[2], rad[3])
        near = dseg < INFL + PAD_H + 0.0025
        cap = np.maximum(cap, np.where(near, np.clip((t - 0.42) / 0.08, 0, 1), 0))
    return np.stack([pad, dors, crease, region], 1), cap


def march(sdf, lo, hi, voxel):
    from skimage import measure
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    B = 8
    nb = (shape + B - 1) // B
    vol = np.full(tuple(nb * B), 1.0, np.float32)
    # coarse pass at block centres
    ci = [lo[k] + (np.arange(nb[k]) * B + B / 2) * voxel for k in range(3)]
    X, Y, Z = np.meshgrid(*ci, indexing='ij')
    Pc = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    dc = sdf(Pc).reshape(X.shape)
    thr = math.sqrt(3) * B / 2 * voxel * 2.5 + 0.005
    todo = np.argwhere(np.abs(dc) < thr)
    inside = dc < 0
    vol = vol.reshape(nb[0], B, nb[1], B, nb[2], B)
    vol[:] = np.where(inside, -1.0, 1.0)[:, None, :, None, :, None]
    vol = vol.reshape(tuple(nb * B))
    off = np.arange(B) * voxel
    chunk = 256
    for s in range(0, len(todo), chunk):
        blk = todo[s:s + chunk]
        base = lo[None, :] + blk * B * voxel
        gx, gy, gz = np.meshgrid(off, off, off, indexing='ij')
        g = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        P = (base[:, None, :] + g[None]).reshape(-1, 3)
        dv = sdf(P).reshape(len(blk), B, B, B)
        for k, (i, j, l) in enumerate(blk):
            vol[i * B:(i + 1) * B, j * B:(j + 1) * B, l * B:(l + 1) * B] = dv[k]
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    return verts + lo, faces


def make(pose, voxel=0.0006):
    if isinstance(pose, str):
        pose = POSES[pose]
    pose = dict(pose)
    sdf, C = build(pose)
    CR = chains(rest_pose(pose))
    lo = (-0.085, CUFF_END - 0.002, -0.115)
    hi = (0.075, 0.205, 0.065)
    V, F = march(sdf, lo, hi, voxel)
    # drop the flat cap at the ring (it is hidden inside the metal ring)
    A, cap = attributes(V, C, sdf, F)
    R = rest_coords(V, C, CR)
    return V.astype(np.float32), F.astype(np.int32), A.astype(np.float32), R.astype(np.float32), pose, cap.astype(np.float32)


def save(out, pose, voxel=0.0006):
    V, F, A, R, pose, cap = make(pose, voxel)
    pj = {k: (list(v) if not isinstance(v, (list, dict)) else v) for k, v in pose.items()}
    np.savez_compressed(out, V=V, F=F, A=A, R=R, CAP=cap, P=json.dumps(pj))
    return len(V), len(F)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--json":
        pose = json.loads(a[1]); a = a[2:]
    else:
        pose = a[0]; a = a[1:]
    out = a[0]
    vox = float(a[1]) * 0.001 if len(a) > 1 else 0.0006
    import time
    t = time.time()
    nv, nf = save(out, pose, vox)
    print("verts", nv, "faces", nf, "%.1fs" % (time.time() - t))
