"""Lookdev: the prosthetic component (v1/v2/v3) on a neutral studio sweep.
blender -b -P scenes/lookdev_part.py -- <out.png> [which=all|v3] [engine]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, prosthesis as P
from mathutils import Vector as V

a = sb.argv()
out = a[0]
which = a[1] if len(a) > 1 else "all"
eng = a[2] if len(a) > 2 else "EEVEE"
sc = sb.reset()
sb.setup_render(eng, mblur=False, cycles_samples=128)
sc.render.resolution_x, sc.render.resolution_y = (1920, 1080)
if eng == "EEVEE":
    sc.eevee.taa_render_samples = 64

sb.world_gradient(top=(0.05, 0.055, 0.065), horizon=(0.12, 0.12, 0.125), bottom=(0.03, 0.03, 0.03), strength=0.6)
floor = sb.prim("plane", "Floor", loc=(0, 0.02, -0.0075), scale=(0.3, 0.3, 1), mat=sb.mat("Floor", (0.18, 0.18, 0.19), rough=0.6))
if which == "all":
    for i, v in enumerate(("v1", "v2", "v3")):
        r, o = P.component(v, "L" + v)
        r.location = ((i - 1) * 0.026, 0, 0)
    cam = sb.camera("C", loc=(0.05, -0.075, 0.085), target=(0.0, 0.02, 0.0), lens=55, fstop=11)
else:
    r, o = P.component("v3", "L")
    r.rotation_euler = (math.radians(0), math.radians(0), math.radians(25))
    cam = sb.camera("C", loc=(0.055, -0.035, 0.05), target=(-0.001, 0.019, 0.0), lens=100, fstop=11)
for nm, loc, sz, e in (("SB1", (-0.25, 0.05, 0.3), (0.35, 0.08), 6.0), ("SB2", (0.3, 0.3, 0.2), (0.3, 0.05), 4.0), ("SB3", (0.0, -0.4, 0.15), (0.6, 0.1), 2.0)):
    pl = sb.prim("plane", nm, loc=loc, scale=(sz[0] / 2, sz[1] / 2, 1), mat=sb.emit_mat(nm + "M", (1, 0.97, 0.94), e))
    pl.rotation_mode = 'QUATERNION'; pl.rotation_quaternion = sb.look_quat(loc, (0, 0.02, 0)) @ __import__('mathutils').Quaternion((1, 0, 0), math.pi)
    pl.visible_camera = False
sb.light('AREA', "Key", loc=(-0.2, -0.15, 0.25), target=(0, 0.02, 0), energy=18, color=(1, 0.96, 0.9), size=0.25)
sb.light('AREA', "Rim", loc=(0.2, 0.25, 0.1), target=(0, 0.02, 0), energy=14, color=(0.8, 0.88, 1.0), size=0.15)
sb.light('AREA', "Fill", loc=(0.25, -0.2, 0.02), target=(0, 0.02, 0), energy=3, color=(0.9, 0.9, 1.0), size=0.4)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
