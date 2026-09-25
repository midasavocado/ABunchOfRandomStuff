"""Lookdev for the launch vehicle. usage: blender -b -P scenes/lookdev_rocket.py -- <view> [<view> ...]
views: full, engines, band, lug, top. Writes lookdev/rocket/<view>.png (SB_RES controls size)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, rocket
from mathutils import Vector as V

views = sb.argv() or ["full"]
sc = sb.reset()
sb.setup_render("EEVEE", samples=int(os.environ.get("SB_SAMPLES", "32")), mblur=False)
Z0 = 11.6
RK = rocket.build(frost=1.0)
RK["root"].location = (0, 0, Z0)
sb.world_bluehour(glow_az=15.0, glow=7.0, zenith=(0.010, 0.026, 0.085), horizon=(0.10, 0.13, 0.22), glow_width=55,
                  band_height=4.0, strength=1.0)
g = sb.prim("plane", "Ground", size=2000, mat=sb.concrete("Conc", (0.30, 0.29, 0.28), scale=0.05))
deck = sb.prim("cube", "Deck", loc=(0, 0, Z0 - 2.6 - 0.75), scale=(9, 9, 0.75), mat=sb.concrete("Deck", (0.35, 0.34, 0.33)))
# floodlights: three banks ~90 m out, slightly warm-white metal halide
for i, (az, e) in enumerate(((-70, 1.2e5), (40, 0.9e5), (170, 0.7e5))):
    a = math.radians(az)
    p = V((math.cos(a) * 90, math.sin(a) * 90, 22))
    sb.light('SPOT', "Flood%d" % i, loc=p, target=(0, 0, Z0 + 30), energy=e, color=(1.0, 0.93, 0.82), size=1.5, spot=40, blend=0.4)
cams = dict(
    full=((-95, -120, 3.0), (0, 0, Z0 + 31), 50),
    engines=((3.8, -5.2, Z0 - 2.4), (0.2, 0.0, Z0 - 0.2), 24),
    band=((-6, -15, Z0 + 46), (0, 0, Z0 + 44.2), 70),
    lug=((1.8, -4.2, Z0 - 1.2), (2.8, 0.3, Z0 + 0.1), 35),
    top=((-18, -40, Z0 + 55), (0, 0, Z0 + 55), 50),
)
d = os.path.join(sb.ROOT, "lookdev", "rocket")
os.makedirs(d, exist_ok=True)
for v in views:
    loc, tgt, lens = cams[v]
    cam = sb.camera("Cam_" + v, loc=loc, target=tgt, lens=lens, clip=(0.05, 3000))
    sc.render.filepath = os.path.join(d, v + ("_" + os.environ.get("SB_TAG") if os.environ.get("SB_TAG") else "") + ".png")
    bpy.ops.render.render(write_still=True)
