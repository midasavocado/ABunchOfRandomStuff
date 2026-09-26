"""Lookdev for the colony ship. usage: blender -b -P scenes/lookdev_ship.py -- <view> [...]
views: full, legs, belly, fire. SB_LEGS=0..1 leg deploy. Writes lookdev/ship/<view>.png"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, ship
from mathutils import Vector as V

views = sb.argv() or ["full"]
sc = sb.reset()
sb.setup_render("EEVEE", samples=int(os.environ.get("SB_SAMPLES", "32")), mblur=False)
SH = ship.build("Ship", legs_deployed=float(os.environ.get("SB_LEGS", "1")))
SH["root"].location = (0, 0, -ship.GROUND_Z)
sb.world_sky(elev=18, azim=-40, strength=0.6)
sb.prim("plane", "Ground", size=3000, mat=sb.painted("Gnd", (0.36, 0.2, 0.11), rough=0.9, grime=0.5))
sb.light('SUN', "Sun", rot=(math.radians(60), 0, math.radians(-40)), energy=4.0, angle=0.5)
cams = dict(full=((-60, -95, 6.0), (0, 0, 22), 50), legs=((-12, -18, 2.0), (0, 0, 3), 30),
            belly=((20, -70, 30), (0, 0, 28), 45), fire=((-60, -95, 6.0), (0, 0, 22), 50))
d = os.path.join(sb.ROOT, "lookdev", "ship")
os.makedirs(d, exist_ok=True)
for v in views:
    if v == "fire":
        SH["root"].location.z = 30
        ship.fly(SH, 1, 1, lambda f: V((0, 0, 30)), lambda f: 1.0)
        sc.frame_set(1)
    loc, tgt, lens = cams[v]
    cam = sb.camera("Cam_" + v, loc=loc, target=tgt, lens=lens, clip=(0.05, 5000))
    sc.render.filepath = os.path.join(d, v + ".png")
    bpy.ops.render.render(write_still=True)
