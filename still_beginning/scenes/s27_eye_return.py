"""s27 STILL US (2610-2700): back to the child's eye - dawn now, on the same rooftop (she watched all night).
The camera begins deep in the pupil, where the sunrise arc glows over the skyline (rhyming with the orbital
sunrise of s26), and pulls back out of the eye as the first direct light warms the skin and lashes; near the end
the gaze lifts toward the sky. The reverse of s01's push-in."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, eye
import timeline as TL
from mathutils import Vector as V, Euler, Quaternion

sid = (sb.argv() or ["s27"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=64, mblur=(os.environ.get("SB_MB", "1") == "1"), shutter=0.5,
                look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "-0.25")))
sc.cycles.max_bounces = 12
sc.cycles.transmission_bounces = 12

E = eye.build()
gaze = E["gaze"]
# dawn: the sun just below the skyline, a bright amber band in the east; the zenith already pale blue
sb.world_bluehour(glow_az=195.0, glow=16.0, glow_col=(1.0, 0.42, 0.10), zenith=(0.10, 0.18, 0.40),
                  horizon=(0.45, 0.42, 0.50), glow_width=48, band_height=4.0, strength=2.2)
eye.build_rooftop_env()


def nog(o):
    o.visible_glossy = False
    return o


# warm first light low from the east (the side of the glow) + cool sky above + bounce
nog(sb.light('AREA', "Dawn", loc=(0.26, -0.26, 0.0), target=(0, 0, 0), energy=1.4, color=(1.0, 0.62, 0.3), size=0.2))
nog(sb.light('AREA', "Sky", loc=(-0.03, -0.14, 0.22), target=(0, -0.01, 0), energy=1.0, color=(0.75, 0.84, 1.0), size=0.16))
nog(sb.light('AREA', "Bounce", loc=(0.0, -0.18, -0.2), target=(0, 0, 0), energy=0.2, color=(0.7, 0.62, 0.58), size=0.4))

yl = eye._limbus_y()
f0, f1 = F0, F1 - 1


def gaze_rot(f):
    t = (f - f0) / (f1 - f0)
    k = sb.smoother(sb.remap(t, 0.62, 0.86))          # the gaze lifts to the sky near the end
    ax = math.radians(sb.lerp(-2.0, 7.5, k))
    az = math.radians(sb.lerp(1.0, -2.0, k))
    ax += math.radians(0.12 * math.sin(f * 0.1) + 0.05 * math.sin(f * 0.27 + 1))
    az += math.radians(0.1 * math.sin(f * 0.08 + 2))
    return Euler((ax, 0, az), 'XYZ')


sb.bake(gaze, f0 - 1, f1 + 1, lambda f: dict(rot=gaze_rot(f)))
sk = E["iris"].data.shape_keys.key_blocks["Constrict"]
for f in range(f0 - 1, f1 + 2):
    t = (f - f0) / (f1 - f0)
    sk.value = sb.lerp(0.1, 0.55, sb.smooth(sb.remap(t, 0.2, 0.9))) + 0.02 * math.sin(f * 0.19)   # pupil closes as light grows
    sk.keyframe_insert("value", frame=f)


def pupil(f):
    return gaze_rot(f).to_matrix() @ V((0, yl + 0.0006, 0))


def focus_pt(f):
    return gaze_rot(f).to_matrix() @ V((0, yl + 0.0008, 0.0012))


d_start = V((0.0, -1.0, 0.05)).normalized()
d_end = V((-0.32, -0.9, -0.12)).normalized()      # ends a 3/4 from the nose side (mirror of s01's start)


def ctar(t):
    f = f0 + t * (f1 - f0)
    k = sb.smoother(t)
    return pupil(f).lerp(V((-0.001, -0.011, 0.001)), k)


def cpos(t):
    k = sb.smoother(t)
    q = d_start.rotation_difference(d_end)
    dirv = Quaternion().slerp(q, k) @ d_start
    dist = sb.lerp(0.034, 0.092, k)
    return ctar(t) + dirv * dist


cam = sb.camera("Cam", loc=(0, -0.1, 0), target=(0, 0, 0), lens=100, fstop=32, clip=(0.001, 100))
sb.cam_bake(cam, f0, f1, cpos, ctar, focus=lambda t: (cpos(t) - focus_pt(f0 + t * (f1 - f0))).length)
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
