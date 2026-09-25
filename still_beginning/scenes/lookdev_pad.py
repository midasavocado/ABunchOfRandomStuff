"""Lookdev / layout for the launch complex + vehicle.
usage: blender -b -P scenes/lookdev_pad.py -- <view> [...]   views: far, scale, aerial, clamp, engines, tower
Writes lookdev/rocket/pad_<view>.png"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, rocket, launchpad as LP
from mathutils import Vector as V

views = sb.argv() or ["far"]
sc = sb.reset()
sb.setup_render("EEVEE", samples=int(os.environ.get("SB_SAMPLES", "32")), mblur=False)
RK = rocket.build(frost=1.0)
RK["root"].location = (0, 0, LP.ROCKET_Z0)
CX = LP.build_complex()
import launchfx as FX
FX.environment(sc)
cams = dict(
    far=((-235, -470, 4.0), (0, 0, 40.0), 100),
    scale=((-95, -150, 1.7), (-2, 0, 36.0), 35),
    aerial=((-220, -260, 160), (0, 0, 10), 35),
    clamp=((4.6, -2.4, LP.ROCKET_Z0 + 0.5), (3.1, 0.2, LP.ROCKET_Z0 + 0.1), 50),
    engines=((3.6, -5.0, LP.TABLE_Z + 0.9), (0.3, 0.2, LP.ROCKET_Z0 - 0.6), 24),
    tower=((-40, -60, 55), (-2, 8, 55), 50),
)
d = os.path.join(sb.ROOT, "lookdev", "rocket")
os.makedirs(d, exist_ok=True)
tag = os.environ.get("SB_TAG", "")
for v in views:
    loc, tgt, lens = cams[v]
    cam = sb.camera("Cam_" + v, loc=loc, target=tgt, lens=lens, clip=(0.05, 20000))
    FX.volume_range(sc, cam, tgt)
    sc.render.filepath = os.path.join(d, "pad_" + v + ("_" + tag if tag else "") + ".png")
    bpy.ops.render.render(write_still=True)
