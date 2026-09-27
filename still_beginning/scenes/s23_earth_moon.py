"""s23e TO THE MOON (2130-2190): from low Earth orbit (continuing s22) the half-lit Moon hangs just above the blue
limb; the camera leaves - one exponential flight across 384,000 km - the Earth falls away under frame and the Moon
grows until its cratered south-polar terminator fills the view, where s23's settlement waits in the grazing light.
Moon at the origin (lib/luna.py globe); Earth = the analytic world (lib/earth.py), re-baked after the operator."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, earth, luna, cinema
from mathutils import Vector as V, Quaternion

sid = (sb.argv() or ["s23e"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
F0, F1 = S0 - 20, S1 + 20
T0 = time.time()
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.3, look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.0")))

KM = 1000.0
DIST = 384400.0
EHAT = V((0.0, -1.0, 0.0))                                   # Moon -> Earth
SUN = V((0.8, 0.0, 0.6)).normalized()                         # lights the Earth under the camera; the Moon half-lit
E_KM = EHAT * DIST                                           # Earth centre (km, scene axes)
E = earth.build(center_km=tuple(E_KM), sun_dir=tuple(SUN), clouds=0.45, lights=0.0, samples=16, nadir=(10.0, 40.0, 0.0),
                detail=1.0)
E.sun_lamp()
moon = luna.globe("Moon", (0, 0, 0), near=EHAT, pole=(0, 0, 1))

# camera start: 800 km over the Earth, Moon ~8 deg above the visible limb
Re = earth.R
u0 = V((0.0, -0.33, 0.944)).normalized()                     # local up at the start: Moon 19 deg below horizontal = 8 deg over the dipped limb
P0 = (E_KM + u0 * (Re + 800.0)) * KM
D1 = 4200e3                                                  # final distance from the Moon's centre
TERM = V((0.45, -0.6, -0.6)).normalized()                     # the terminator on the southern near side (n.SUN = 0)
DIR1 = (TERM + V((0.0, -0.7, 0.0))).normalized()             # end: above it, from the Earth side
SPOLE = TERM * luna.R * 0.92


def ease(t):
    # dwell at both worlds, rush the empty crossing: a steep sigmoid (the Earth stays in frame for the opening
    # third, the Moon is already growing by the middle)
    k = 7.0
    return 0.5 + 0.5 * math.tanh(k * (t - 0.5)) / math.tanh(k * 0.5)


def pos(t):
    # ease log(d_Earth / d_Moon): slow at both worlds, the crossing rushes; |d_E| + |d_M| ~ the Earth-Moon distance
    Dm = E_KM.length * KM
    dE0, dM0 = (P0 - E_KM * KM).length, P0.length
    r0, r1 = math.log(dE0 / dM0), math.log((Dm - D1) / D1)
    e = ease(t)
    r = r0 + (r1 - r0) * e
    f0 = Dm / (1.0 + math.exp(r0))
    d = Dm / (1.0 + math.exp(r)) * (dM0 / f0) ** (1.0 - e)      # anchored: exactly P0 at the start, D1 at the end
    dr = P0.normalized().slerp(DIR1, sb.smoother(sb.remap(ease(t), 0.45, 1.0)))
    pB = dr * d
    # early: climb away from the Earth (radially, leaning toward the Moon) so the planet shrinks away under frame
    # instead of swinging behind the camera; blend into the lunar approach by mid-flight
    Ec = E_KM * KM
    dEt = dE0 * (60000e3 / dE0) ** sb.remap(e, 0.0, 0.35) * (1.0 + max(0.0, e - 0.35) * 4.0)   # log climb off the Earth
    pA = Ec + ((P0 - Ec).normalized().slerp((u0 + V((0.0, 0.55, 0.0))).normalized(), sb.smoother(sb.remap(e, 0.0, 0.25)))) * dEt
    w = sb.smoother(sb.remap(e, 0.2, 0.6))
    return pA.lerp(pB, w)


def aim_quat(p, t):
    # look at the Moon (dipped at the start so the Earth's limb sits low in frame), drifting onto the south pole
    tgt = V((0, 0, 0)).lerp(SPOLE, sb.smoother(sb.remap(ease(t), 0.55, 1.0)))
    f = (tgt - p).normalized()
    up = (p - E_KM * KM).normalized().lerp(V((0, 0, 1)), sb.smoother(sb.remap(ease(t), 0.0, 0.5))).normalized()
    dip = math.radians(5.0) * (1.0 - sb.smoother(sb.remap(ease(t), 0.0, 0.35)))
    r = f.cross(up).normalized()
    f = (Quaternion(r, -dip) @ f).normalized()
    u = r.cross(f)
    from mathutils import Matrix
    return Matrix((r, u, -f)).transposed().to_quaternion()


cam = sb.camera("Cam", loc=P0, target=(0, 0, 0), lens=35, clip=(10.0, 1e9))
cam.rotation_mode = 'QUATERNION'
for f in range(F0, F1 + 1):
    t = max(0.0, min(1.0, (f - S0) / (S1 - 1 - S0)))
    p = pos(t)
    cam.location = p
    cam.rotation_quaternion = aim_quat(p, t)
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_quaternion", frame=f)
    dm = p.length - luna.R
    cam.data.clip_start = max(10.0, dm * 2e-4)
    cam.data.clip_end = p.length * 3.0 + 1e7
    cam.data.keyframe_insert("clip_start", frame=f); cam.data.keyframe_insert("clip_end", frame=f)
    cam.data.lens = 35.0 + 5.0 * sb.smoother(t)
    cam.data.keyframe_insert("lens", frame=f)
    cam.data.dof.focus_distance = max(1.0, dm)
    cam.data.dof.keyframe_insert("focus_distance", frame=f)
E.track(cam, F0, F1)
cinema.AFTER.append(lambda c, f0, f1: E.track(c, f0, f1))
print("s23e built in %.0fs" % (time.time() - T0))
sb.render_shot(sid)
