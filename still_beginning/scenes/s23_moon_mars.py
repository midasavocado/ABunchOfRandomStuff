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
SUN = V((1.0, 0.06, 0.04)).normalized()
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
D1 = MR * 5.2                                          # end distance from Mars' centre (disc ~ 22 deg across)


def ease(t):
    return sb.smoother(t) * 0.55 + t * 0.45


def pos(t):
    e = ease(t)
    d0 = (MC - P0).length
    d = d0 * (D1 / d0) ** e
    # leave the Moon along its local up first, then along the line to Mars
    lift = u0 * (luna.R * 0.6) * sb.smoother(sb.remap(e, 0.0, 0.3))
    base = MC + (P0 + lift - MC).normalized() * d
    return base


cam = sb.camera("Cam", loc=P0, target=MC, lens=24, clip=(50.0, 5e9))
cam.rotation_mode = 'QUATERNION'
for f in range(F0, F1 + 1):
    t = max(0.0, min(1.0, (f - S0) / (S1 - 1 - S0)))
    p = pos(t)
    look_mars = (MC - p).normalized()
    look_mid = (look_mars + (-u0) * 0.45).normalized()   # at the start: framing both the limb and the red star
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
