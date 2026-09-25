"""THE COMPONENT: proximal-phalanx structural link of the prosthetic finger (making sequence s06-s09).
SDF -> marching cubes (system python3: numpy + scikit-image). Units: mm inside, metres in the output.
Local frame: MCP pivot axis = X through the origin, PIP pivot axis = X through (0, L, 0); long axis +Y,
dorsal +Z. Versions: 'v1' conventional machined block, 'v2' lattice-infilled, 'v3' final generative/organic.
usage: python3 sdfpart.py <version|morph:a:b:t> <out.npz> [voxel_mm]
       python3 sdfpart.py batch <outdir> <voxel_mm> <n>      (morph sequence v1->v2->v3 for the screen)"""
import sys, os, math, json
import numpy as np
from skimage import measure

L = 40.0          # MCP -> PIP pivot distance (mm)
R_MCP, W_MCP, BORE_MCP = 5.2, 7.0, 2.05
R_PIP, W_PIP, BORE_PIP = 4.1, 6.0, 1.55
SLOT = 2.55       # half-width of the PIP clevis slot (distal tongue sits inside)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def cyl_x(P, c, r, hw):
    """capped cylinder along X centred at c (y,z), radius r, half width hw."""
    dy = P[:, 1] - c[0]; dz = P[:, 2] - c[1]
    d_r = np.sqrt(dy * dy + dz * dz) - r
    d_x = np.abs(P[:, 0]) - hw
    return np.minimum(np.maximum(d_r, d_x), 0) + np.sqrt(np.maximum(d_r, 0) ** 2 + np.maximum(d_x, 0) ** 2)


def round_cone(P, a, b, r1, r2):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ba = b - a; l2 = ba @ ba; rr = r1 - r2; a2 = l2 - rr * rr; il2 = 1.0 / l2
    pa = P - a; y = pa @ ba; z = y - l2
    w = pa * l2 - y[:, None] * ba[None, :]
    x2 = np.einsum('ij,ij->i', w, w); y2 = y * y * l2; z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    d_b = np.sqrt(x2 + z2) * il2 - r2
    d_a = np.sqrt(x2 + y2) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d_b, np.where(np.sign(y) * a2 * y2 < k, d_a, d_mid))


def chain(P, pts, rads, k=0.8, sx=1.0):
    Q = P.copy(); Q[:, 0] = Q[:, 0] * sx
    d = None
    for i in range(len(pts) - 1):
        a = np.array(pts[i], float); b = np.array(pts[i + 1], float)
        a[0] *= sx; b[0] *= sx
        s = round_cone(Q, a, b, rads[i], rads[i + 1])
        d = s if d is None else smin(d, s, k)
    return d / sx if sx > 1 else d


def ellipsoid(P, c, r):
    q = (P - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def round_box(P, c, b, r):
    q = np.abs(P - np.asarray(c)) - np.asarray(b)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r


def gyroid(P, period=3.4):
    s = 2 * math.pi / period
    x, y, z = P[:, 0] * s, P[:, 1] * s, P[:, 2] * s
    g = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    return g / s * 0.75   # approx distance scale


# ------------------------------------------------------------------ shared features

def bosses(P):
    mcp = cyl_x(P, (0.0, 0.0), R_MCP, W_MCP)
    pip = cyl_x(P, (L, 0.0), R_PIP, W_PIP)
    return mcp, pip


def cut_features(P, d):
    """bores, PIP clevis slot, MCP flats. Applied to every version (they are interfaces, not design space)."""
    y, z = P[:, 1], P[:, 2]
    bore1 = np.sqrt(y * y + z * z) - BORE_MCP
    bore2 = np.sqrt((y - L) ** 2 + z * z) - BORE_PIP
    d = np.maximum(d, -bore1)
    d = np.maximum(d, -bore2)
    # clevis slot at the PIP end: the distal tongue swings inside |x| < SLOT
    slot = np.maximum(np.abs(P[:, 0]) - SLOT, -(y - (L - 5.6)))
    slot = np.maximum(slot, np.sqrt((y - L) ** 2 + z * z) - (R_PIP + 3.0))
    d = np.maximum(d, -slot)
    # MCP: the palm yoke clamps the boss -> the boss faces are machined flat at |x| = W_MCP
    d = np.maximum(d, np.abs(P[:, 0]) - W_MCP)
    return d


def v1(P):
    """Conventional: rectangular machined bar with bosses, a shallow side pocket."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    bar = round_box(P, (0, L * 0.5, 0), (W_MCP - 0.6, L * 0.5, 3.7), 0.6)
    mcp, pip = bosses(P)
    d = np.minimum(np.minimum(bar, mcp), pip)
    pocket = round_box(P, (0, L * 0.5, 0.0), (W_MCP + 1, L * 0.5 - 8.0, 2.0), 0.5)
    pocket = np.maximum(pocket, -(np.abs(x) - (W_MCP - 1.6)))
    d = np.maximum(d, -pocket)
    return cut_features(P, d)


def envelope(P):
    """Design envelope used by v2: tapered, rounded bar hugging the bosses."""
    return chain(P, [(0, 0, 0), (0, 13, 0.5), (0, 27, 0.3), (0, L, 0)], [5.6, 4.9, 4.5, 4.4], k=2.0, sx=0.72)


def v2(P):
    """Lattice-infilled envelope with a thin perimeter frame (first AI iteration: mass removal)."""
    env = envelope(P)
    g = np.abs(gyroid(P, 3.0)) - 0.38
    lattice = np.maximum(env, g)
    skin = np.maximum(env, -(env + 0.55))          # thin shell
    # open the shell on the sides so the lattice reads; keep dorsal/palmar skins
    # keep only a perimeter frame (four edge strips) so the gyroid core reads from every side
    skin = np.maximum(skin, -(np.abs(P[:, 2]) - 3.2))
    skin = np.maximum(skin, -(np.abs(P[:, 0]) - 2.6))
    mcp, pip = bosses(P)
    d = np.minimum(lattice, skin)
    d = smin(d, mcp, 0.6)
    d = smin(d, pip, 0.6)
    return cut_features(P, d)


def spline(ctrl, n=14):
    """Catmull-Rom through control points -> dense polyline (smooth members without lumpy joints)."""
    c = [np.array(p, float) for p in ctrl]
    c = [c[0] * 2 - c[1]] + c + [c[-1] * 2 - c[-2]]
    out = []
    for i in range(1, len(c) - 2):
        p0, p1, p2, p3 = c[i - 1], c[i], c[i + 1], c[i + 2]
        for u in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(c[-2])
    return out


def member(P, ctrl, radii, n=14, k=0.3):
    pts = spline(ctrl, n)
    rs = np.interp(np.linspace(0, 1, len(pts)), np.linspace(0, 1, len(radii)), radii)
    d = None
    for i in range(len(pts) - 1):
        s = round_cone(P, pts[i], pts[i + 1], rs[i], rs[i + 1])
        d = s if d is None else np.minimum(d, s)
    return d


def v3(P):
    """Final generative link: per side a Warren truss (arched top chord, bottom chord, three diagonals),
    two dorsal arch bridges and a palmar X-tie, all flowing into the pivot bosses with generous fillets."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    mcp, pip = bosses(P)
    d = np.minimum(mcp, pip)
    for s in (-1, 1):
        top = member(P, [(s * 5.2, 0.5, 3.2), (s * 4.9, 10, 4.6), (s * 4.55, 22, 4.7), (s * 4.5, 33, 3.7), (s * 4.6, 39.5, 2.4)],
                     [2.05, 1.55, 1.3, 1.35, 1.75])
        bot = member(P, [(s * 5.2, 1.5, -3.3), (s * 4.7, 12, -3.5), (s * 4.4, 24, -3.2), (s * 4.5, 34, -2.6), (s * 4.6, 39.5, -2.0)],
                     [1.8, 1.25, 1.05, 1.15, 1.6])
        m = smin(top, bot, 0.8)
        for (y0, z0), (y1, z1) in (((5.5, -3.3), (14.5, 4.4)), ((14.5, 4.4), (24.5, -3.2)), ((24.5, -3.2), (33.0, 3.6))):
            dg = member(P, [(s * 4.8, y0, z0), (s * 4.55, (y0 + y1) / 2, (z0 + z1) / 2 * 0.9), (s * 4.5, y1, z1)], [1.05, 0.8, 1.0], n=8)
            m = smin(m, dg, 1.1)
        d = smin(d, m, 1.5)
    # dorsal arch bridges between the top chords
    for yb in (14.5, 30.0):
        br = member(P, [(-4.5, yb, 4.2), (0, yb + 0.6, 5.3), (4.5, yb, 4.2)], [1.05, 0.85, 1.05], n=8)
        d = smin(d, br, 1.2)
    # palmar X-tie
    for s in (-1, 1):
        tie = member(P, [(s * 4.4, 12.5, -3.4), (0, 19.5, -3.9), (-s * 4.3, 26.5, -3.2)], [0.9, 0.75, 0.9], n=8)
        d = smin(d, tie, 0.9)
    # coupler-rod lug on the palmar side near the MCP boss
    lug = cyl_x(P, (8.4, -5.0), 1.75, 1.7)
    d = smin(d, lug, 1.3)
    d = np.maximum(d, -(np.sqrt((y - 8.4) ** 2 + (z + 5.0) ** 2) - 0.8))
    return cut_features(P, d)


def distal(P):
    """Distal segment (PIP frame: pivot axis X through origin, pointing +Y): graphite tongue in the clevis +
    white fingertip shell with a palmar silicone pad and a palmar tendon anchor keel."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    tongue = np.minimum(cyl_x(P, (0.0, 0.0), 3.9, 2.35), round_box(P, (0, 5.0, -0.3), (2.35, 5.0, 3.0), 0.3))
    # fingertip: wider than tall, full rounded tip (no claw)
    Q = P.copy(); Q[:, 0] = Q[:, 0] * 0.86
    shell2 = member(Q, [(0, 8.0, 0.3), (0, 15, 0.4), (0, 22, 0.0), (0, 27.0, -0.5)],
                    [6.1, 6.4, 6.1, 5.6], n=10)
    shell = np.maximum(shell2, -(y - 6.2) - 1.4)
    d = smin(tongue, shell, 1.2)
    # pad groove (separates the silicone pad from the white shell) and a dorsal cap line
    groove = np.abs(z + 1.2 - 0.02 * (y - 20) ** 2 * 0.0) - 0.18
    groove = np.maximum(groove, -(y - 11.0))
    d = np.maximum(d, -np.maximum(groove, -(shell + 0.45)))
    # tendon anchor: a small palmar keel under the tongue with a cross hole
    keel = round_box(P, (0, 7.0, -4.6), (1.3, 2.6, 1.3), 0.5)
    d = smin(d, keel, 0.8)
    d = np.maximum(d, -(np.sqrt((y - 7.6) ** 2 + (z + 4.9) ** 2) - 0.55))
    return d


FINGER_MCP = {  # palm frame (mm): MCP pivot centre, splay (deg, + toward the ulnar side, +X)
    "index": ((-24.0, 80.0, 2.0), -5.0), "middle": ((-4.2, 83.0, 2.5), -1.0),
    "ring": ((15.2, 80.5, 2.0), 3.5), "pinky": ((32.5, 72.0, 0.5), 9.0)}


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def palm_body(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    w = 24.0 + 13.5 * smoothstep(-5, 62, y)            # half width: wrist -> knuckles
    cx = 4.5 * smoothstep(0, 60, y)
    qx = (x - cx) / w * 30.0
    za = z + 0.0036 * qx * qx - 1.5 * smoothstep(40, 80, y) * 0.0
    Q = np.stack([qx, y, za], 1)
    body = round_box(Q, (0.0, 36.0, -0.5), (22.0, 37.0, 5.2), 8.0)
    # curved knuckle line (middle MCP most distal)
    yend = 78.0 - 0.0105 * (x + 4.0) ** 2
    body = smax(body, y - yend, 7.0)
    # thenar and hypothenar volumes (palmar side)
    body = smin(body, ellipsoid(P, (-25.0, 30.0, -6.5), (12.5, 21.0, 10.5)), 7.0)
    body = smin(body, ellipsoid(P, (29.0, 36.0, -5.5), (8.5, 22.0, 8.0)), 6.0)
    # wrist collar: elliptic cylinder along Y from y=-9 to 4
    ell = np.sqrt((x / 1.25) ** 2 + (z + 1.0) ** 2) - 20.6
    collar = np.maximum(ell, np.abs(y + 2.5) - 6.5)
    body = smin(body, collar, 7.0)
    return body, za


def palm(P):
    """Palm chassis (right hand; wrist centre origin, fingers +Y, dorsal +Z, thumb on -X): tapered anatomical
    body, white dorsal cover separated by a parting groove, MCP yokes, thenar housing, wrist collar."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    d, za = palm_body(P)
    for name, ((mx, my, mz), sp) in FINGER_MCP.items():
        a = math.radians(sp)
        c, s_ = math.cos(a), math.sin(a)
        lx = (x - mx) * c - (y - my) * s_
        ly = (x - mx) * s_ + (y - my) * c
        lz = z - mz
        # yoke cheeks straddling the link boss (|x| 7.2 .. 8.9) with a rounded nose around the pivot
        nose = np.sqrt(ly ** 2 + lz ** 2) - 5.6
        cheek = np.maximum(np.abs(np.abs(lx) - 8.05) - 0.85, np.minimum(nose, np.maximum(np.abs(lz) - 5.6, ly)))
        cheek = np.maximum(cheek, -(ly + 14.0))
        d = smin(d, cheek, 1.6)
        pocket = np.maximum(np.abs(lx) - 7.3, np.minimum(np.maximum(nose - 0.7, -ly - 8.0),
                                                           np.maximum(-ly, np.abs(lz) - 25)))
        d = np.maximum(d, -pocket)
        d = np.maximum(d, -np.maximum(np.sqrt(ly ** 2 + lz ** 2) - 1.45, np.abs(lx) - 12))
    # parting groove between the dorsal cover and the frame (follows the arched surface at za = 1.6)
    groove = np.maximum(np.abs(za - 1.6) - 0.28, -(d + 0.7))
    groove = np.maximum(groove, -(y + 1.0))
    d = np.maximum(d, -groove)
    # palmar pad groove
    pg = np.maximum(np.abs(z + 10.2) - 0.25, -(d + 0.6))
    pg = np.maximum(pg, 6.0 - y)
    d = np.maximum(d, -pg)
    return d


PALM_LO, PALM_HI = np.array([-46.0, -9.0, -22.0]), np.array([52.0, 97.0, 16.0])
DIST_LO, DIST_HI = np.array([-9.0, -7.0, -9.0]), np.array([9.0, 35.0, 8.0])
REGIONS = {"palm": (PALM_LO, PALM_HI), "distal": (DIST_LO, DIST_HI)}


VERS = dict(v1=v1, v2=v2, v3=v3, distal=distal, palm=palm)
LO = np.array([-8.2137, -6.8071, -7.4043])
HI = np.array([8.2, 45.6, 6.8])


def sample(fn, voxel, LO=LO, HI=HI):
    shape = np.ceil((HI - LO) / voxel).astype(int) + 1
    xs = LO[0] + np.arange(shape[0]) * voxel
    ys = LO[1] + np.arange(shape[1]) * voxel
    zs = LO[2] + np.arange(shape[2]) * voxel
    vol = np.empty(shape, np.float32)
    step = max(1, int(2e6 // (shape[1] * shape[2])))
    for i in range(0, shape[0], step):
        X, Y, Z = np.meshgrid(xs[i:i + step], ys, zs, indexing='ij')
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        vol[i:i + step] = fn(P).reshape(X.shape)
    return vol


def mesh(vol, voxel, LO=LO):
    # pad so the surface is closed
    vol = np.pad(vol, 1, constant_values=1.0)
    verts, faces, normals, _ = measure.marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    verts = verts - voxel + LO
    return (verts * 0.001).astype(np.float32), faces[:, ::-1].astype(np.int32)


def regions(mode, Vm):
    """Per-vertex material masks (R, G, B) from the same analytic boundaries as the grooves."""
    P = Vm * 1000.0
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if mode == "palm":
        _, za = palm_body(P)
        cover = ((za > 1.6) & (y > -1.0)).astype(np.float32)
        pad = ((z < -10.2) & (y > 6.0)).astype(np.float32)
        return np.stack([cover, pad, np.zeros_like(x)], 1)
    if mode == "distal":
        pad = ((z < -1.2) & (y > 11.0)).astype(np.float32)
        tongue = (y < 6.4).astype(np.float32)
        return np.stack([pad, tongue, np.zeros_like(x)], 1)
    return None


def save(out, V, F, mode=None):
    A = regions(mode, V) if mode else None
    if A is not None:
        np.savez_compressed(out, V=V, F=F, A=A)
    else:
        np.savez_compressed(out, V=V, F=F)
    print(out, "verts", len(V), "faces", len(F))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "batch":
        outdir, vox, n = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        os.makedirs(outdir, exist_ok=True)
        vols = {k: sample(f, vox) for k, f in VERS.items()}
        for i in range(n):
            t = i / (n - 1) * 2.0
            if t <= 1.0:
                a, b, u = "v1", "v2", t
            else:
                a, b, u = "v2", "v3", t - 1.0
            u = u * u * (3 - 2 * u)
            vol = vols[a] * (1 - u) + vols[b] * u
            save(os.path.join(outdir, "m%03d.npz" % i), *mesh(vol, vox))
    else:
        out = sys.argv[2]
        vox = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        lo, hi = REGIONS.get(mode, (LO, HI))
        save(out, *mesh(sample(VERS[mode], vox, lo, hi), vox, lo), mode=mode)
