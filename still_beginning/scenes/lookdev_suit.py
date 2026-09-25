"""Suit lookdev. blender -b -P scenes/lookdev_suit.py -- <view> <out.png> [W H]
views: forearm | full | fullback | posed | glove"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, suit
from mathutils import Vector as V

a = sb.argv()
view = a[0] if a else "full"
out = a[1] if len(a) > 1 else "/tmp/suit.png"
W, H = (int(a[2]), int(a[3])) if len(a) > 3 else (1600, 900)
sc = sb.reset()
sb.setup_render("EEVEE", samples=32, mblur=False)
sc.render.resolution_x, sc.render.resolution_y = W, H
t0 = time.time()
sb.world_gradient(top=(0.10, 0.11, 0.13), horizon=(0.30, 0.28, 0.25), bottom=(0.08, 0.075, 0.07), strength=1.0)
floor = sb.prim("plane", "Floor", size=20, mat=sb.concrete("Fl", (0.25, 0.24, 0.23)))

if view in ("forearm", "glove"):
    F = suit.forearm("L", "relaxed", glove_voxel=0.6)
    F["root"].location = (0, 0, 0.3)
    F["root"].rotation_euler = (0, 0, math.radians(-90))     # hand toward +X
    floor.location.z = 0.18
    sb.light('AREA', "Key", loc=(-0.3, -0.5, 0.75), target=(0.05, 0, 0.3), energy=40, color=(1.0, 0.86, 0.68), size=0.5)
    sb.light('AREA', "Rim", loc=(0.4, 0.5, 0.5), target=(0.05, 0, 0.3), energy=20, color=(0.8, 0.88, 1.0), size=0.3)
    sb.light('AREA', "Fill", loc=(0.1, -0.6, 0.1), target=(0.05, 0, 0.3), energy=6, color=(1, 0.9, 0.8), size=1.0)
    if view == "forearm":
        cam = sb.camera("C", loc=(0.05, -0.55, 0.52), target=(0.04, 0, 0.3), lens=55, fstop=8)
    else:
        cam = sb.camera("C", loc=(0.22, -0.30, 0.40), target=(0.14, 0, 0.3), lens=85, fstop=8)
else:
    S = suit.build("Astro", visor=os.environ.get("VISOR", "gold"))
    if view == "posed":
        suit.pose(S, l_shoulder=(35, 10, 0), l_elbow=70, r_shoulder=(-15, 12, 0), r_elbow=25,
                  l_hip=(20, 3, 0), l_knee=30, r_hip=(-10, 3, 0), r_knee=10, torso=(8, 0, 10))
    sun = sb.sun(35, 220, energy=4.0)
    sb.light('AREA', "Fill", loc=(3, -3, 2), target=(0, 0, 1), energy=300, color=(0.8, 0.85, 1.0), size=3)
    if view == "helmet":
        cam = sb.camera("C", loc=(0.35, -1.0, 1.78), target=(0, 0, 1.72), lens=85, fstop=5.6)
    elif view == "suithand":
        cam = sb.camera("C", loc=(0.9, -1.1, 1.0), target=(0.33, 0, 0.72), lens=85)
    elif view == "fullback":
        cam = sb.camera("C", loc=(2.2, 3.4, 1.5), target=(0, 0, 1.0), lens=50)
    else:
        cam = sb.camera("C", loc=(1.6, -3.8, 1.35), target=(0, 0, 1.0), lens=50)
print("BUILD %.1fs" % (time.time() - t0))
sc.render.filepath = out
t1 = time.time()
bpy.ops.render.render(write_still=True)
print("RENDER %.1fs" % (time.time() - t1))
