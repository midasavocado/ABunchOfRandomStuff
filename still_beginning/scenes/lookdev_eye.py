"""Eye look-development harness. usage: blender -b -P lookdev_eye.py -- <mode> <out.png>
modes: iris | refl | lashes | face"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, eye
from mathutils import Vector as V, Euler

args = sb.argv()
mode = args[0] if args else "iris"
out = args[1] if len(args) > 1 else "/tmp/lookdev.png"
res = int(os.environ.get("LD_RES", "1600"))
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=64, mblur=False, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.6")))
sc.render.resolution_x = res
sc.render.resolution_y = int(res * 9 / 16)
sc.cycles.samples = int(os.environ.get("LD_SPP", "64"))
sc.cycles.max_bounces = 12
sc.cycles.transmission_bounces = 12
E = eye.build()
E["gaze"].rotation_euler = Euler((math.radians(float(os.environ.get("LD_GX", "0"))), 0, math.radians(float(os.environ.get("LD_GZ", "0")))))

sb.world_bluehour(glow_az=200.0, glow=9.0, glow_col=(1.0, 0.30, 0.04), zenith=(0.05, 0.11, 0.33),
                  horizon=(0.22, 0.29, 0.50), glow_width=60, band_height=5.0, strength=2.0)
eye.build_rooftop_env()


def nog(o):
    o.visible_glossy = False
    return o


key_e = float(os.environ.get("LD_KEY", "2.0"))
nog(sb.light('AREA', "SkyKey", loc=(-0.03, -0.14, 0.22), target=(0, -0.01, 0), energy=key_e, color=(0.72, 0.82, 1.0), size=0.14))
nog(sb.light('AREA', "Afterglow", loc=(0.30, -0.22, -0.02), target=(0, 0, 0), energy=0.7, color=(1.0, 0.52, 0.22), size=0.25))
nog(sb.light('AREA', "Bounce", loc=(0.0, -0.18, -0.2), target=(0, 0, 0), energy=0.25, color=(0.55, 0.6, 0.7), size=0.4))

yl = eye._limbus_y()
if mode == "iris":
    for n in ("Lids", "Lashes", "Caruncle", "TearMeniscus"):
        o = bpy.data.objects.get(n)
        if o: o.hide_render = True
    cam = sb.camera("Cam", loc=(0, -0.045, 0.002), target=(0, yl, 0), lens=100, fstop=45)
    cam.data.dof.focus_distance = 0.045 + yl + 0.0008
elif mode == "refl":
    cam = sb.camera("Cam", loc=(0.012, -0.06, -0.006), target=(0, yl, 0), lens=100, fstop=32)
    cam.data.dof.focus_distance = (V((0.012, -0.06, -0.006)) - V((0, yl, 0))).length
elif mode == "lashes":
    p = V((0.018, -0.075, -0.012))
    cam = sb.camera("Cam", loc=p, target=(0, -0.012, 0.002), lens=100, fstop=22)
    cam.data.dof.focus_distance = (p - V((0, -0.012, 0.002))).length
else:  # face
    p = V((0.025, -0.11, -0.02))
    cam = sb.camera("Cam", loc=p, target=(0.001, -0.012, 0.0), lens=85, fstop=16)
    cam.data.dof.focus_distance = (p - V((0, -0.012, 0.0))).length
sc.render.filepath = out
if os.environ.get('LD_DIAG'):
    exec(open(os.environ['LD_DIAG']).read())
else:
    bpy.ops.render.render(write_still=True)
