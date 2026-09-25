"""s18: solve the right glove's rim grip on the left wrist-disconnect lock collar, and generate the glove mesh
sequence (approach -> grip). Run with system python3.  usage: python3 lib/s18grip.py [solve|meshes]
Ring frame (= suit.wrist_disconnect local): axis +Y toward the left hand, collar band y in [-0.0405,-0.0072], r=0.0626.
Hand frame (sdfglove): fingers +Y, dorsal +Z, thumb -X.  Placement: dorsal = radial e_r(theta_h), thumb side = +Y_ring."""
import os, sys, json, math, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdfglove as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "suit", "s18")
os.makedirs(OUT, exist_ok=True)
RC = 0.0626          # collar outer radius
RH = 0.0596          # housing
RCL = 0.0563         # clamp


def placement(theta, Rh, t0, cy, tilt=0.0):
    """returns (Rm, org): p_ring = Rm @ p_hand + org.  tilt rotates the hand about its own X (pitch)."""
    er = np.array([math.cos(theta), 0, math.sin(theta)])
    et = np.array([-math.sin(theta), 0, math.cos(theta)])
    Yr = np.array([0, 1.0, 0])
    X = -Yr
    Z = er
    Y = -et
    Rm = np.stack([X, Y, Z], 1)
    if tilt:
        Rm = Rm @ G.rot((1, 0, 0), tilt)
    org = er * Rh + et * t0 + Yr * cy
    return Rm, org


def ring_pt(theta, r, y):
    return np.array([r * math.cos(theta), y, r * math.sin(theta)])


def to_hand(Rm, org, p):
    return Rm.T @ (np.asarray(p) - org)


def solve(theta_h=math.radians(200)):
    best = None
    for Rh, t0, cy, tilt, di, dt in itertools.product((0.092, 0.100, 0.108), (0.0, 0.015, 0.03), (-0.043, -0.048),
                                                      (-0.2, 0.0, 0.25), (0.45, 0.6), (0.85, 1.05)):
        Rm, org = placement(theta_h, Rh, t0, cy, tilt)
        tg = dict(index=ring_pt(theta_h - di, RC + 0.0003, -0.0205),
                  middle=ring_pt(theta_h - di - 0.05, RC + 0.0003, -0.0385),
                  ring=ring_pt(theta_h - di - 0.12, RH + 0.0003, -0.057),
                  pinky=ring_pt(theta_h - di - 0.22, RCL + 0.0012, -0.074),
                  thumb=ring_pt(theta_h + dt, RC + 0.0003, -0.026))
        errs = {}
        pose = {}
        for n in ("index", "middle", "ring", "pinky"):
            x, e = G.solve_finger(n, to_hand(Rm, org, tg[n]), splay0=0.0)
            pose[n] = x; errs[n] = e
        tx, te = G.solve_thumb(to_hand(Rm, org, tg["thumb"]), face=Rm.T @ np.array([math.cos(theta_h + dt), 0, math.sin(theta_h + dt)]))
        pose["thumb"] = tx; errs["thumb"] = te
        cost = errs["index"] * 2 + errs["middle"] * 2 + errs["thumb"] * 2 + errs["ring"] * 0.5 + errs["pinky"] * 0.3
        if best is None or cost < best[0]:
            best = (cost, dict(theta=theta_h, Rh=Rh, t0=t0, cy=cy, tilt=tilt, di=di, dt=dt), pose, errs)
            print("best", round(cost, 2), best[1], {k: round(v, 2) for k, v in errs.items()}, flush=True)
    return best


def contacts(Rm, org, rot=0.0):
    """rigid parts as capped cylinders in hand coords (collar, flange, housing, clamp, glove ring)."""
    ax = Rm.T @ np.array([0, 1.0, 0])
    out = []
    for r, y0, y1 in ((RC, -0.0405, -0.0058), (0.0666, -0.0058, -0.0002), (RH, -0.060, -0.0405), (RCL, -0.077, -0.0588), (0.0494, -0.0002, 0.010)):
        c = to_hand(Rm, org, (0, (y0 + y1) / 2, 0))
        out.append(dict(type="cyl", c=c.tolist(), axis=ax.tolist(), r=r, h=(y1 - y0) / 2))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "solve"
    if mode == "solve":
        th = math.radians(float(sys.argv[2])) if len(sys.argv) > 2 else math.radians(200)
        cost, P, pose, errs = solve(th)
        json.dump(dict(place=P, pose={k: list(v) for k, v in pose.items()}, errs=errs), open(os.path.join(OUT, "grip.json"), "w"), indent=1)
        print("saved", P, errs)


# ------------------------------------------------------------------ shared motion schedule (shot-relative frames)
LATCH = 45            # shot frame of the latch click (global 1665 = bar 38 downbeat)
ANG0 = -38.0          # collar start angle (deg) -> 0 at LATCH
NSTEP = 12            # finger-closure mesh steps


def _sm(t):
    t = min(1.0, max(0.0, t)); return t * t * (3 - 2 * t)


def collar_angle(f):
    if f <= 29:
        return ANG0
    if f >= LATCH:
        return 0.0
    t = (f - 29) / (LATCH - 29)
    s = 0.35 * _sm(t) + 0.65 * t ** 2.2          # eases in, arrives with speed -> hard stop = click
    return ANG0 * (1 - s)


def pin_pop(f, pop=0.0032):
    if f < LATCH:
        return 0.0
    x = f - LATCH
    return pop * (1 - math.exp(-x / 0.55) * math.cos(x * 2.4))


def finger_blend(f):
    if f < 14:
        return 0.0
    if f < 26:
        return _sm((f - 14) / 12.0)
    if f < 56:
        return 1.0
    return 1.0 - 0.55 * _sm((f - 56) / 9.0)


def approach_offset(f):
    """hand placement offset (radial, tangential, axial) in metres relative to the grip."""
    import math as m
    if f < 26:
        t = min(1.0, max(0.0, (f - 4) / 22.0))
        k = (1 - t) ** 2.4
        return (0.11 * k, 0.07 * k, 0.03 * k)
    if f < 57:
        # tiny squeeze settle after contact and a firm press at the click
        sq = -0.0006 * math.exp(-((f - LATCH - 1.5) / 2.0) ** 2)
        return (sq, 0.0, 0.0)
    t = (f - 57) / 30.0
    k = min(1.0, t) ** 1.8
    return (0.10 * k, 0.05 * k, 0.035 * k)


def step_of(f):
    b = finger_blend(f)
    return int(round(b * NSTEP))


def grip_data():
    return json.load(open(os.path.join(OUT, "grip.json")))


def pose_at(b, D=None):
    D = D or grip_data()
    g = D["pose"]
    op = {"index": (0.10, 0.12, 0.05), "middle": (0.12, 0.14, 0.0), "ring": (0.16, 0.18, -0.05), "pinky": (0.2, 0.22, -0.12)}
    pose = {}
    for n in ("index", "middle", "ring", "pinky"):
        pose[n] = [op[n][i] + (g[n][i] - op[n][i]) * b for i in range(3)]
    t_open = [0.05, 0.25, 0.4, 0.0, 0.05, 0.05]
    pose["thumb"] = [t_open[i] + (g["thumb"][i] - t_open[i]) * b for i in range(6)]
    return pose


def placement_at(f, D=None, off=None):
    D = D or grip_data()
    P = D["place"]
    o = approach_offset(f) if off is None else off
    th = P["theta"]
    return placement(th, P["Rh"] + o[0], P["t0"] + o[1], P["cy"] + o[2], P["tilt"])


def make_meshes(vox_hero=0.00045, vox=0.0006):
    D = grip_data()
    # steps 0..NSTEP at the approach frame where that blend occurs (14..26)
    for k in range(NSTEP + 1):
        b = k / NSTEP
        # frame at which finger_blend reaches b during the approach
        f = 14 + 12 * (math.acos(1 - 2 * b) / math.pi) if 0 < b < 1 else (14 if b == 0 else 26)
        Rm, org = placement_at(f, D)
        pose = pose_at(b, D)
        pose["contacts"] = contacts(Rm, org)
        p = os.path.join(OUT, "grip_%02d.npz" % k)
        if os.path.exists(p):
            continue
        n = G.save(p, pose, vox_hero if k == NSTEP else vox)
        print(k, round(f, 2), n, flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "meshes":
    make_meshes()
