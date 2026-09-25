import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, eye
from mathutils import Vector as V, Euler, Quaternion

sid = (sb.argv() or ["s01"])[0]
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=64, mblur=(os.environ.get("SB_MB", "1") == "1"), shutter=0.5,
                look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.0")))
sc.cycles.max_bounces = 12
sc.cycles.transmission_bounces = 12

E = eye.build()
gaze = E["gaze"]

# luminous blue hour: bright open sky overhead, last amber afterglow hugging the horizon
sb.world_bluehour(glow_az=200.0, glow=9.0, glow_col=(1.0, 0.30, 0.04), zenith=(0.05, 0.11, 0.33),
                  horizon=(0.22, 0.29, 0.50), glow_width=60, band_height=5.0, strength=2.0)
eye.build_rooftop_env()


def nog(o):
    o.visible_glossy = False
    return o


# soft overhead sky key: casts the lash shadow band across sclera/iris (as in reference photography)
nog(sb.light('AREA', "SkyKey", loc=(-0.03, -0.14, 0.22), target=(0, -0.01, 0), energy=1.5, color=(0.72, 0.82, 1.0), size=0.14))
# warm afterglow low from the side: rim on lid and cheek
nog(sb.light('AREA', "Afterglow", loc=(0.30, -0.22, -0.02), target=(0, 0, 0), energy=0.35, color=(1.0, 0.52, 0.22), size=0.25))
# low bounce from the rooftop
nog(sb.light('AREA', "Bounce", loc=(0.0, -0.18, -0.2), target=(0, 0, 0), energy=0.25, color=(0.55, 0.6, 0.7), size=0.4))

yl = eye._limbus_y()
f0, f1 = 0, 89


def gaze_rot(f):
    t = (f - f0) / (f1 - f0)
    k = sb.smoother(sb.remap(t, 0.46, 0.53))          # quick saccade down toward the eyepiece
    ax = math.radians(sb.lerp(-8.0, -1.0, k))
    az = math.radians(sb.lerp(-6.0, 0.8, k))
    ax += math.radians(0.15 * math.sin(f * 0.11) + 0.05 * math.sin(f * 0.23 + 2))
    az += math.radians(0.12 * math.sin(f * 0.09 + 1))
    return Euler((ax, 0, az), 'XYZ')


sb.bake(gaze, f0, f1, lambda f: dict(rot=gaze_rot(f)))
sk = E["iris"].data.shape_keys.key_blocks["Constrict"]
for f in range(f0, f1 + 1):
    t = (f - f0) / (f1 - f0)
    sk.value = sb.lerp(0.35, 0.0, sb.smooth(sb.remap(t, 0.5, 1.0))) + 0.02 * math.sin(f * 0.21)
    sk.keyframe_insert("value", frame=f)


def pupil(f):
    return gaze_rot(f).to_matrix() @ V((0, yl + 0.0006, 0))


def focus_pt(f):
    return gaze_rot(f).to_matrix() @ V((0, yl + 0.0008, 0.0012))


# camera: low 3/4 from the temple side (cornea dome catching the sky) -> arcs to frontal and pushes in on the pupil
d_start = V((0.30, -0.92, -0.20)).normalized()
d_end = V((0.0, -1.0, 0.07)).normalized()


def ctar(t):
    f = f0 + t * (f1 - f0)
    k = sb.smoother(t)
    start_t = V((0.0015, -0.011, 0.0005))
    return start_t.lerp(pupil(f), k)


def cpos(t):
    k = sb.smoother(t)
    q = d_start.rotation_difference(d_end)
    dirv = Quaternion().slerp(q, k) @ d_start
    dist = sb.lerp(0.085, 0.050, k)
    return ctar(t) + dirv * dist


cam = sb.camera("Cam", loc=(0, -0.1, 0), target=(0, 0, 0), lens=100, fstop=32, clip=(0.001, 100))
sb.cam_bake(cam, f0, f1, cpos, ctar, focus=lambda t: (cpos(t) - focus_pt(f0 + t * (f1 - f0))).length)

sb.frames(0, 90)
sb.save("s01")
sb.render_shot(sid)
