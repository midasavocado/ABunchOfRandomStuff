"""SDF child (~9 y, 1.33 m) -> per-part marching-cubes meshes (system python3: numpy + scikit-image).
World frame: Z up, metres. Poses are given as joint positions (dict). Parts: skin (head/neck), hoodie, hood, cuffs,
pants, shoes. usage: python3 sdfchild.py <pose> <out.npz>"""
import sys, math, json
import numpy as np
from skimage import measure
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sdfhand import smin, sd_round_cone, sd_ellipsoid, sd_round_box, norm, rot


def smax(a, b, k):
    return -smin(-a, -b, k)


def to_local(P, origin, R):
    """R: 3x3 whose columns are the local axes (x,y,z) in world."""
    return (P - origin) @ R


# ------------------------------------------------------------------ head (head-local: +Y forward, +Z up, +X child's left)

def sd_head(Q):
    d = sd_ellipsoid(Q, (0, -0.012, 0.028), (0.071, 0.090, 0.083))                       # cranium
    d = smin(d, sd_ellipsoid(Q, (0, 0.030, -0.018), (0.060, 0.062, 0.066)), 0.025)        # midface mass
    d = smin(d, sd_ellipsoid(Q, (0, 0.040, -0.060), (0.046, 0.044, 0.036)), 0.02)         # jaw
    d = smin(d, sd_ellipsoid(Q, (0, 0.072, -0.076), (0.020, 0.018, 0.017)), 0.012)        # chin
    for s in (-1, 1):
        d = smin(d, sd_ellipsoid(Q, (s * 0.036, 0.058, -0.028), (0.026, 0.022, 0.024)), 0.015)   # full cheeks
    d = smin(d, sd_ellipsoid(Q, (0, 0.066, 0.030), (0.056, 0.022, 0.017)), 0.015)          # brow
    for s in (-1, 1):                                                                       # eye sockets
        d = smax(d, -sd_ellipsoid(Q, (s * 0.030, 0.083, 0.012), (0.018, 0.013, 0.012)), 0.006)
    # nose
    d = smin(d, sd_round_cone(Q, np.array([0, 0.080, 0.014]), np.array([0, 0.097, -0.016]), 0.0055, 0.0085), 0.006)
    d = smin(d, sd_ellipsoid(Q, (0, 0.099, -0.020), (0.0105, 0.0095, 0.0095)), 0.005)
    for s in (-1, 1):
        d = smin(d, sd_ellipsoid(Q, (s * 0.0105, 0.090, -0.025), (0.0075, 0.0070, 0.0060)), 0.004)
        d = smax(d, -sd_ellipsoid(Q, (s * 0.0060, 0.093, -0.0295), (0.0030, 0.0035, 0.0018)), 0.0015)  # nostrils
    # lips
    d = smin(d, sd_ellipsoid(Q, (0, 0.086, -0.041), (0.018, 0.0085, 0.0058)), 0.005)
    d = smin(d, sd_ellipsoid(Q, (0, 0.084, -0.050), (0.0155, 0.0085, 0.0060)), 0.005)
    d = smax(d, -sd_ellipsoid(Q, (0, 0.093, -0.0455), (0.016, 0.004, 0.0009)), 0.001)     # mouth line
    # ears
    for s in (-1, 1):
        c = np.array([s * 0.073, -0.006, 0.004])
        ear = sd_ellipsoid(Q, c + np.array([s * 0.004, 0, 0]), (0.009, 0.019, 0.028))
        ear = smax(ear, -sd_ellipsoid(Q, c + np.array([s * 0.010, 0.002, -0.002]), (0.006, 0.012, 0.018)), 0.002)  # concha
        d = smin(d, ear, 0.004)
    return d


def head_frame(neck_top, head_fwd, head_up):
    y = norm(head_fwd)
    z = norm(np.asarray(head_up) - y * (np.asarray(head_up) @ y))
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


# ------------------------------------------------------------------ poses (joint positions)

def two_bone(a, target, l1, l2, pole):
    a = np.asarray(a, float); t = np.asarray(target, float)
    d = t - a
    L = min(np.linalg.norm(d), l1 + l2 - 1e-4)
    dn = norm(d)
    x = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
    h = math.sqrt(max(l1 * l1 - x * x, 0))
    pv = np.asarray(pole, float) - a
    pv = norm(pv - dn * (pv @ dn))
    mid = a + dn * x + pv * h
    end = a + dn * L
    return mid, end


def pose_scope():
    """Standing at a telescope eyepiece (eyepiece ~ (0, -0.46, 1.16) in scope-local coords; child faces -Y? no:
    the child stands behind the eyepiece at +Y side looking down/forward toward -Y is set by the caller)."""
    J = {}
    J["pelvis"] = np.array([0.0, 0.0, 0.74])
    J["chest"] = np.array([0.0, -0.07, 1.02])
    J["neck"] = np.array([0.0, -0.12, 1.12])
    J["head_fwd"] = norm((0.0, -1.0, -0.55))
    J["head_up"] = norm((0.0, -0.45, 1.0))
    J["shoulder_r"] = np.array([-0.155, -0.08, 1.07])
    J["shoulder_l"] = np.array([0.155, -0.08, 1.07])
    J["hand_r"] = np.array([-0.06, -0.33, 1.02])       # on focus knob (set by caller)
    J["hand_l"] = np.array([0.08, -0.36, 1.10])        # resting on tube
    J["hip_r"] = np.array([-0.075, 0.0, 0.72]); J["hip_l"] = np.array([0.075, 0.0, 0.72])
    J["foot_r"] = np.array([-0.11, 0.06, 0.07]); J["foot_l"] = np.array([0.12, -0.10, 0.07])
    return J


def pose_draw():
    """Seated at a table (table top z=0.66), leaning over paper, right hand drawing."""
    J = {}
    J["pelvis"] = np.array([0.0, 0.0, 0.45])
    J["chest"] = np.array([0.0, 0.10, 0.74])
    J["neck"] = np.array([0.0, 0.15, 0.84])
    J["head_fwd"] = norm((0.05, 1.0, -1.05))
    J["head_up"] = norm((0.05, 0.9, 0.8))
    J["shoulder_r"] = np.array([-0.152, 0.12, 0.79])
    J["shoulder_l"] = np.array([0.152, 0.12, 0.79])
    J["hand_r"] = np.array([-0.05, 0.40, 0.70])
    J["hand_l"] = np.array([0.13, 0.42, 0.675])
    J["hip_r"] = np.array([-0.075, 0.0, 0.44]); J["hip_l"] = np.array([0.075, 0.0, 0.44])
    J["knee_r"] = np.array([-0.08, 0.30, 0.44]); J["knee_l"] = np.array([0.08, 0.30, 0.44])
    J["foot_r"] = np.array([-0.08, 0.34, 0.05]); J["foot_l"] = np.array([0.09, 0.30, 0.05])
    return J


POSES = {"scope": pose_scope, "draw": pose_draw}

UPPER, FORE = 0.235, 0.20
THIGH, SHIN = 0.33, 0.32


def solve(J):
    J = dict(J)
    for s in ("r", "l"):
        sh = J["shoulder_" + s]
        pole = sh + np.array([(-0.2 if s == "r" else 0.2), 0.05, -0.35])
        J["elbow_" + s], J["wrist_" + s] = two_bone(sh, J["hand_" + s], UPPER, FORE, pole)
        if "knee_" + s not in J:
            hp = J["hip_" + s]
            J["knee_" + s], J["ankle_" + s] = two_bone(hp, J["foot_" + s] + np.array([0, 0, 0.0]), THIGH, SHIN, hp + np.array([0, -0.5, -0.2]))
        else:
            J["ankle_" + s] = J["foot_" + s]
    return J


# ------------------------------------------------------------------ parts

def folds(P, a, b, amp, wl, phase=0.0):
    """Sleeve/trouser folds: ring-like ripples along segment a->b, strongest near b (the bend)."""
    ab = b - a
    L = np.linalg.norm(ab)
    t = ((P - a) @ ab) / (L * L)
    ang = np.arctan2(*(np.stack([(P - a) @ norm(np.cross(ab, (0.3, 0.2, 1))), (P - a) @ norm(np.cross(ab, np.cross(ab, (0.3, 0.2, 1))))], 0)))
    w = np.clip(t, 0, 1)
    wob = np.sin(ang * 1.0 + phase * 3.1) * 0.9 + np.sin(ang * 3 + phase) * 0.4
    rip = np.sin(t * L / wl * 2 * math.pi * (0.8 + 0.25 * np.sin(ang + phase)) + 2.2 * wob + phase)
    env = np.clip(np.sin(ang * 1.5 + phase * 2) * 0.5 + 0.6, 0, 1)      # folds gather on one side
    return amp * rip * env * (0.2 + 0.8 * w ** 3) * ((t > -0.05) & (t < 1.05))


def body_sdf(J, part):
    Hf = head_frame(J["neck"], J["head_fwd"], J["head_up"])
    head_c = J["neck"] + Hf[:, 2] * 0.085 + Hf[:, 1] * 0.005

    def torso(P, off):
        # chest/abdomen as round cones + shoulder bar; off = clothing thickness
        d = sd_round_cone(P, J["pelvis"] + [0, 0, 0.02], J["chest"], 0.105 + off, 0.115 + off)
        d = smin(d, sd_round_cone(P, J["shoulder_r"], J["shoulder_l"], 0.047 + off, 0.047 + off), 0.09)
        d = smin(d, sd_round_cone(P, J["chest"], J["neck"] - [0, 0, 0.02], 0.10 + off, 0.05 + off), 0.04)
        return d

    def arm(P, s, off, shorten=0.0):
        sh, el, wr = J["shoulder_" + s], J["elbow_" + s], J["wrist_" + s]
        wr2 = wr - norm(wr - el) * shorten
        d = sd_round_cone(P, sh, el, 0.042 + off, 0.035 + off)
        return smin(d, sd_round_cone(P, el, wr2, 0.035 + off, 0.028 + off), 0.02)

    def leg(P, s, off):
        hp, kn, an = J["hip_" + s], J["knee_" + s], J["ankle_" + s]
        d = sd_round_cone(P, hp, kn, 0.062 + off, 0.045 + off)
        return smin(d, sd_round_cone(P, kn, an + [0, 0, 0.03], 0.045 + off, 0.036 + off), 0.02)

    def f(P):
        if part == "skin":
            Q = to_local(P, head_c, Hf)
            d = sd_head(Q)
            d = smin(d, sd_round_cone(P, J["neck"] - Hf[:, 1] * 0.01, head_c - Hf[:, 2] * 0.04 - Hf[:, 1] * 0.015, 0.041, 0.043), 0.02)
            return d
        if part == "hoodie":
            d = torso(P, 0.018)
            for s in ("r", "l"):
                a = arm(P, s, 0.016, shorten=0.055)
                a = a - folds(P, J["elbow_" + s] - norm(J["elbow_" + s] - J["shoulder_" + s]) * 0.08, J["wrist_" + s], 0.0035, 0.028, 1.3 if s == "r" else 0.4)
                d = smin(d, a, 0.03)
            # soft drape at the hem/waist (sparse, irregular)
            zz = P[:, 2] - J["pelvis"][2]
            d = d - 0.0016 * np.sin(zz * 70 + 3 * np.sin(P[:, 0] * 25 + P[:, 1] * 18)) * np.clip(1 - np.abs(zz - 0.06) / 0.12, 0, 1)
            # collar opening (neck hole)
            d = smax(d, -sd_round_cone(P, J["neck"] - [0, 0, 0.05], J["neck"] + [0, 0, 0.2], 0.052, 0.07), 0.01)
            return d
        if part == "hood":
            Q = to_local(P, head_c, Hf)
            outer = sd_ellipsoid(Q, (0, -0.03, 0.03), (0.108, 0.128, 0.118))
            outer = smin(outer, sd_ellipsoid(Q, (0, -0.075, 0.075), (0.05, 0.06, 0.06)), 0.04)   # soft peak at the back
            outer = outer - 0.003 * np.sin(Q[:, 1] * 35 + Q[:, 2] * 20 + np.sin(Q[:, 0] * 30)) * np.clip(-Q[:, 1] / 0.08, 0, 1)
            shell = np.abs(outer) - 0.007
            # face opening with a soft rolled brim
            opening = sd_ellipsoid(Q, (0, 0.105, -0.025), (0.078, 0.10, 0.105))
            shell = smax(shell, -opening, 0.012)
            shell = smin(shell, np.abs(opening) - 0.004 + 0.0 * Q[:, 0], 0.0) if False else shell
            # blend into collar
            shell = smin(shell, sd_round_cone(P, J["neck"] - Hf[:, 1] * 0.03, head_c - Hf[:, 1] * 0.05, 0.075, 0.06), 0.02)
            return shell
        if part == "cuffs":
            d = None
            for s in ("r", "l"):
                el, wr = J["elbow_" + s], J["wrist_" + s]
                a = wr - norm(wr - el) * 0.065
                c = sd_round_cone(P, a, wr + norm(wr - el) * 0.004, 0.034, 0.031)
                c = smax(c, -sd_round_cone(P, a - norm(wr - el) * 0.02, wr + norm(wr - el) * 0.03, 0.024, 0.024), 0.002)
                # rib knit
                ax = norm(wr - el)
                ang = np.arctan2((P - a) @ norm(np.cross(ax, (0, 0, 1))), (P - a) @ norm(np.cross(ax, np.cross(ax, (0, 0, 1)))))
                c = c - 0.0011 * np.sin(ang * 26)
                d = c if d is None else np.minimum(d, c)
            return d
        if part == "pants":
            d = sd_round_cone(P, J["pelvis"] - [0, 0, 0.03], J["pelvis"] + [0, 0, 0.08], 0.118, 0.112)
            for s in ("r", "l"):
                lg = leg(P, s, 0.012)
                lg = lg - folds(P, J["knee_" + s], J["ankle_" + s], 0.0028, 0.035, 0.7 if s == "r" else 2.1)
                d = smin(d, lg, 0.04)
            return d
        if part == "shoes":
            d = None
            for s in ("r", "l"):
                an, kn = J["ankle_" + s], J["knee_" + s]
                fwd = np.array([kn[0] - an[0], kn[1] - an[1], 0.0])
                fwd = norm(fwd) if np.linalg.norm(fwd) > 1e-3 else np.array([0, -1.0, 0])
                fwd = -fwd if (J["pelvis"][1] - an[1]) * 0 else fwd
                toe = an + fwd * 0.15 + np.array([0, 0, -0.035])
                heel = an - fwd * 0.04 + np.array([0, 0, -0.04])
                sh = sd_round_cone(P, heel, toe, 0.045, 0.038)
                sh = smax(sh, -(P[:, 2] - 0.0), 0.004)      # flat sole on the ground
                d = sh if d is None else np.minimum(d, sh)
            return d
        raise KeyError(part)
    return f, head_c, Hf


def mesh_part(f, lo, hi, vox):
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    xs = lo[0] + np.arange(shape[0]) * vox
    ys = lo[1] + np.arange(shape[1]) * vox
    zs = lo[2] + np.arange(shape[2]) * vox
    vol = np.empty(shape, np.float32)
    step = max(1, int(4e6 // (shape[1] * shape[2])))
    for i in range(0, shape[0], step):
        X, Y, Z = np.meshgrid(xs[i:i + step], ys, zs, indexing='ij')
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        vol[i:i + step] = f(P).reshape(X.shape)
    if vol.min() > 0 or vol.max() < 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    v, fc, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox,) * 3)
    return (v + lo).astype(np.float32), fc.astype(np.int32)


def build(pose, parts=("skin", "hoodie", "hood", "cuffs", "pants", "shoes"), vox_face=0.0008, vox_body=0.002, overrides=None):
    J0 = POSES[pose]()
    for k, v in (overrides or {}).items():
        J0[k] = norm(v) if k in ("head_fwd", "head_up") else np.asarray(v, float)
    J = solve(J0)
    out = {}
    pts = np.array([J[k] for k in J if isinstance(J[k], np.ndarray) and J[k].shape == (3,) and k not in ("head_fwd", "head_up")])
    lo_all = pts.min(0) - 0.2; hi_all = pts.max(0) + 0.25
    lo_all[2] = max(lo_all[2], -0.01)
    for part in parts:
        f, head_c, Hf = body_sdf(J, part)
        if part in ("skin", "hood"):
            lo, hi, vox = head_c - 0.16, head_c + 0.16, vox_face
            if part == "skin":
                lo = np.minimum(lo, J["neck"] - 0.08)
        elif part == "cuffs":
            wr = np.stack([J["wrist_r"], J["wrist_l"]])
            lo, hi, vox = wr.min(0) - 0.1, wr.max(0) + 0.1, 0.0009
        elif part == "shoes":
            fe = np.stack([J["ankle_r"], J["ankle_l"]])
            lo, hi, vox = fe.min(0) - 0.25, fe.max(0) + 0.25, 0.0015
            lo[2] = -0.005
        else:
            lo, hi, vox = lo_all, hi_all, vox_body
        v, fc = mesh_part(f, lo, hi, vox)
        out[part + "_V"] = v
        out[part + "_F"] = fc
        print(part, len(v))
    meta = dict(head_c=body_sdf(J, "skin")[1].tolist(), Hf=body_sdf(J, "skin")[2].tolist(),
                joints={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in J.items()})
    out["meta"] = json.dumps(meta)
    return out


if __name__ == "__main__":
    pose, outp = sys.argv[1], sys.argv[2]
    parts = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "all" else ("skin", "hoodie", "hood", "cuffs", "pants", "shoes")
    ov = json.loads(open(sys.argv[4]).read()) if len(sys.argv) > 4 else None
    np.savez_compressed(outp, **build(pose, parts, overrides=ov))
