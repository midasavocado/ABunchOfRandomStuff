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
SUN = V((1.0, 0.06, 0.04)).normalized()                      # first quarter: terminator through the poles
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
DIR1 = V((0.0, -0.85, -0.52)).normalized()                   # end: Earth side, below the equator, facing the south pole
SPOLE = V((0.0, 0.0, -luna.R * 0.92))


def ease(t):
    return sb.smoother(t) * 0.7 + t * 0.3


def pos(t):
    d0 = P0.length
    e = ease(t)
    d = d0 * (D1 / d0) ** e
    dr = P0.normalized().slerp(DIR1, sb.smoother(sb.remap(e, 0.35, 1.0)))
    return dr * d


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
