"""s23m ONWARD (2265-2295): the bridge from the Moon to Mars. Low over the lunar limb, a red star sits above the
horizon; the camera rises off the Moon, turns to it and rushes out - the grey limb falls away under frame - until
Mars is a lit gibbous world (s25 telescope, then the colony, follow). Moon at the origin; Mars is a scaled globe
placed at a scaled distance with the same angular size and lighting as the real one."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, luna, mars, space
from mathutils import Vector as V, Quaternion, Matrix

sid = (sb.argv() or ["s23m"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
F0, F1 = S0 - 20, S1 + 20
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.3, look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.0")))
SUN = V((0.8, 0.0, 0.6)).normalized()                         # as s23e: the ground under the camera in daylight
space.starfield("Stars", strength=1.0)
sun = sb.light('SUN', "Sun", energy=4.6, color=(1.0, 0.96, 0.92), angle=0.53)
sun.rotation_mode = 'QUATERNION'; sun.rotation_quaternion = SUN.to_track_quat('Z', 'Y')
moon = luna.globe("Moon", (0, 0, 0), near=(0, -1, 0), pole=(0, 0, 1))

MD = V((0.15, 0.9, 0.4)).normalized()                  # direction to Mars
K = 1.0 / 25.0                                         # Mars globe scale (distance and radius scaled together)
MR = mars.R * K
MC = MD * 2.0e9
mglobe = sb.prim("sphere", "Mars", loc=tuple(MC), segments=512, ring_count=256, radius=MR,
                 mat=mars.planet_mat("MarsFar", bump_k=K * 1.0))
mglobe.visible_shadow = False
for p in mglobe.data.polygons:
    p.use_smooth = True
mglobe.rotation_euler = (0.3, -0.5, 1.2)               # present Valles Marineris / Tharsis toward us
# limb air: a real-radius shell scaled down (object coordinates stay in real metres for the atmosphere shader)
shell = sb.prim("sphere", "MarsAtmoS", loc=tuple(MC), segments=256, ring_count=128, radius=mars.R + 42e3)
shell.scale = (K, K, K)
for p in shell.data.polygons:
    p.use_smooth = True
mars.atmosphere("MarsAtmo", sun=SUN, obj=shell)

# start: 60 km over the Moon, Mars 8 deg above the local horizontal
perp = (V((0, 0, 1)) - MD * MD.z).normalized()
u0 = (MD * math.sin(math.radians(8.0)) + perp * math.cos(math.radians(8.0))).normalized()
P0 = u0 * (luna.R + 60e3)
D1 = MR * 7.0                                          # end distance from Mars' centre (disc ~16 deg across)


def ease(t):
    return sb.smoother(t) ** 1.7            # a long, slow departure over the lunar horizon, then the rush


def pos(t):
    # ease log(d_Moon / d_Mars): slow off the Moon, the crossing rushes, slow onto Mars; anchored at P0 and D1
    e = ease(t)
    lift = u0 * (luna.R * 0.08) * sb.smoother(sb.remap(e, 0.0, 0.3))
    Pl = P0 + lift
    Dt = MC.length
    dm0, dM0 = Pl.length - 0.0, (MC - Pl).length
    r0, r1 = math.log(dm0 / dM0), math.log((Dt - D1) / D1)
    r = r0 + (r1 - r0) * e
    f0 = Dt / (1.0 + math.exp(r0))
    d = Dt / (1.0 + math.exp(r)) * (dM0 / f0) ** (1.0 - e)
    return MC + (Pl - MC).normalized() * d


cam = sb.camera("Cam", loc=P0, target=MC, lens=24, clip=(50.0, 5e9))
cam.rotation_mode = 'QUATERNION'
for f in range(F0, F1 + 1):
    t = max(0.0, min(1.0, (f - S0) / (S1 - 1 - S0)))
    p = pos(t)
    look_mars = (MC - p).normalized()
    look_mid = (look_mars + (-u0) * 0.6).normalized()   # at the start: framing both the limb and the red star
    fwd = look_mid.slerp(look_mars, sb.smoother(sb.remap(t, 0.0, 0.55)))
    up = u0
    r = fwd.cross(up).normalized(); u = r.cross(fwd)
    cam.location = p
    cam.rotation_quaternion = Matrix((r, u, -fwd)).transposed().to_quaternion()
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_quaternion", frame=f)
    cam.data.lens = 24.0 + 26.0 * sb.smoother(t)
    cam.data.keyframe_insert("lens", frame=f)
    dmin = min((p.length - luna.R), (p - MC).length - MR)
    cam.data.clip_start = max(10.0, dmin * 1e-3)
    cam.data.clip_end = 5e9
    cam.data.keyframe_insert("clip_start", frame=f)
    cam.data.dof.focus_distance = (MC - p).length
    cam.data.dof.keyframe_insert("focus_distance", frame=f)
sb.render_shot(sid)
