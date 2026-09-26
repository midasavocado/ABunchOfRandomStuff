"""The camera operator: applied to every shot's camera right before rendering (sb.render_shot), whatever built its
animation. It turns a clean A->B move into a filmed one and makes shots flow into each other:

  * in motion at the cut: the authored move is entered a little after its start and left a little before its end
    (TRIM), so an eased move never sits dead still on a cut;
  * handles: the camera keeps travelling past both cuts with the velocity it had there (linear continuation of
    position and rotation) for the HANDLE frames the edit blends across;
  * an operator's float: slow, organic drift in position (scaled to the subject distance), pan/tilt and a touch of
    roll, per-shot style (steadicam / handheld / drone / macro / weightless / locked);
  * lens breathing: focal length and focus distance ride a hair.
Deterministic (seeded by shot id)."""
import math, random
import bpy
from mathutils import Vector as V, Quaternion, Matrix
import timeline as TL

#            amp (fraction of subject distance), period (frames), rot deg, roll deg
STYLES = {
    "steady":   (0.0035, 80.0, 0.22, 0.15),
    "handheld": (0.0060, 34.0, 0.45, 0.35),
    "drone":    (0.0022, 120.0, 0.18, 0.25),
    "macro":    (0.0045, 64.0, 0.20, 0.10),
    "weightless": (0.0030, 140.0, 0.30, 0.45),
    "locked":   (0.0010, 110.0, 0.06, 0.04),
}
SHOT_STYLE = {
    "s01": "locked", "s03a": "macro", "s03b": "macro", "s04": "weightless", "s05": "macro", "s06b": "macro",
    "s07a": "macro", "s08a": "macro", "s08b": "handheld", "s09a": "handheld", "s09b": "handheld", "s10a": "macro",
    "s10b": "macro", "s12b": "drone", "s13": "drone", "s14": "drone", "s15": "steady", "s16a": "macro",
    "s16c": "handheld", "s17": "steady", "s18": "macro", "s19a": "steady", "s19b": "locked", "s19c": "macro",
    "s20": "locked", "s21": "weightless", "s22": "weightless", "s23e": "weightless", "s23": "steady",
    "s23m": "weightless", "s25": "weightless", "s24c": "drone", "s24": "handheld", "s25z": "drone", "s28": "weightless",
}
# (trim_in, trim_out) as fractions of the authored move; 0 keeps an exact authored frame (first shot / title end)
TRIM_DEFAULT = (0.06, 0.06)
TRIM = {"s01": (0.0, 0.05), "s28": (0.04, 0.0), "s20": (0.0, 0.05), "s09a": (0.0, 0.06)}


def _sample(cam, f):
    """camera world matrix + lens + focus at frame f from its F-curves (fast; no scene evaluation)."""
    sc = bpy.context.scene
    sc.frame_set(f)
    return cam.matrix_world.copy(), cam.data.lens, cam.data.dof.focus_distance


def _noise1(seed, period):
    rs = random.Random(seed)
    comps = [(rs.uniform(0.7, 1.3) / period, rs.uniform(0, 2 * math.pi), rs.uniform(0.5, 1.0)) for _ in range(3)]
    norm = sum(c[2] for c in comps)

    def f(x):
        return sum(a * math.sin(2 * math.pi * w * x + p) for w, p, a in comps) / norm
    return f


def finish(sid, cam=None, style=None):
    sc = bpy.context.scene
    cam = cam or sc.camera
    if cam is None or cam.get("cinema_done"):
        return
    _, a, b, _ = TL.shot(sid)
    hin, hout = TL.handles(sid)
    N = b - 1 - a
    ti, to = TRIM.get(sid, TRIM_DEFAULT)
    st = STYLES[style or SHOT_STYLE.get(sid, "steady")]
    # sample the authored move densely (every frame of the shot)
    fr0 = sc.frame_current
    samples = [_sample(cam, f) for f in range(a, b)]
    sc.frame_set(fr0)

    def at(u):
        """authored state at fractional u in [0,1] (linear interpolation between frame samples)."""
        x = max(0.0, min(1.0, u)) * N
        i = min(int(x), N - 1) if N > 0 else 0
        k = x - i
        m0, l0, d0 = samples[i]; m1, l1, d1 = samples[min(i + 1, N)]
        loc = m0.translation.lerp(m1.translation, k)
        q = m0.to_quaternion().slerp(m1.to_quaternion(), k)
        return loc, q, l0 + (l1 - l0) * k, d0 + (d1 - d0) * k

    span = 1.0 - ti - to

    def state(g):
        t = (g - a) / max(1, N)
        u = ti + t * span
        du = 1.0 / max(1, N)
        if u < ti:                                   # before the entry point: continue its velocity backwards
            p0, q0, l0, d0 = at(ti); p1, q1, _, _ = at(ti + du)
            k = (ti - u) / du
            d = q1.rotation_difference(q0) if False else q0.inverted() @ q1
            ax, ang = d.to_axis_angle()
            return p0 - (p1 - p0) * k, q0 @ Quaternion(ax, -ang * k), l0, d0
        if u > 1.0 - to:                             # after the exit point: continue forwards
            ue = 1.0 - to
            p1, q1, l1, d1 = at(ue); p0, q0, _, _ = at(ue - du)
            k = (u - ue) / du
            d = q0.inverted() @ q1
            ax, ang = d.to_axis_angle()
            return p1 + (p1 - p0) * k, q1 @ Quaternion(ax, ang * k), l1, d1
        return at(u)

    amp, period, rdeg, rolldeg = st
    nx, ny, nz = (_noise1(hash((sid, c)) & 0xffff, period) for c in "xyz")
    npn, ntl, nrl = (_noise1(hash((sid, c)) & 0xffff, period * 1.3) for c in ("pan", "tilt", "roll"))
    nlens = _noise1(hash((sid, "lens")) & 0xffff, period * 2.0)
    # rebuild the camera animation: unparented, world-space keys over the shot + handles
    cam.animation_data_clear()
    cam.data.animation_data_clear()
    cam.parent = None
    cam.constraints.clear() if hasattr(cam.constraints, "clear") else None
    cam.rotation_mode = 'QUATERNION'
    for g in range(a - hin, b + hout):
        p, q, lens, fd = state(g)
        dist = max(0.05, fd)
        R = q.to_matrix()
        right, up = R.col[0], R.col[1]
        off = (right * nx(g) + up * ny(g) + R.col[2] * nz(g) * 0.5) * (amp * dist)
        qo = q @ Quaternion((0, 1, 0), math.radians(rdeg) * npn(g)) @ Quaternion((1, 0, 0), math.radians(rdeg * 0.7) * ntl(g)) \
            @ Quaternion((0, 0, 1), math.radians(rolldeg) * nrl(g))
        cam.location = p + off
        cam.rotation_quaternion = qo
        cam.keyframe_insert("location", frame=g)
        cam.keyframe_insert("rotation_quaternion", frame=g)
        cam.data.lens = lens * (1.0 + 0.004 * nlens(g))
        cam.data.dof.focus_distance = fd
        cam.data.keyframe_insert("lens", frame=g)
        cam.data.dof.keyframe_insert("focus_distance", frame=g)
    cam["cinema_done"] = 1
