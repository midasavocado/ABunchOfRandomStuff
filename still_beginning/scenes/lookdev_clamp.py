"""Lookdev: hero hold-down clamp + vehicle lug (s19c framing). SB_LD_VIEW=s19c|wide|side ; renders lookdev/clamp/<view>.png"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, rocket, launchpad as LP
from mathutils import Vector as V

view = os.environ.get("SB_LD_VIEW", "s19c")
sc = sb.reset()
sb.setup_render(os.environ.get("SB_ENGINE", "CYCLES"), cycles_samples=int(os.environ.get("SB_SAMPLES", "64")), mblur=False)
sc.render.resolution_x, sc.render.resolution_y = (1920, 1080) if sb.RES == "half" else (960, 540)
M = LP.materials()
cl = LP.clamp("Clamp270", 270, M)
LP.clamp_actuator_update(cl, M, "Clamp270")
# the vehicle lug + a slab of the aft skirt
RM = rocket._materials(frost=0.8)
root = sb.empty("RK", loc=(0, 0, LP.ROCKET_Z0))
rocket._lug("Lug270", 270, RM, root)
sk = rocket.lathe("Skirt", [(2.47, 0.0), (2.47, 3.0)], mat=RM["skirt"], parent=root)
deck = sb.prim("plane", "Deck", loc=(0, 0, LP.TABLE_Z), scale=(12, 12, 1), mat=M["deck"])
sb.world_bluehour(glow_az=110.0, glow=6.0, glow_col=(1.0, 0.45, 0.15), zenith=(0.03, 0.06, 0.16), horizon=(0.2, 0.22, 0.32), strength=1.0)
sb.light('AREA', "Flood", loc=(6, -14, 16), target=(0, -3.3, 12.5), energy=9000, color=(1.0, 0.93, 0.84), size=1.5)
sb.light('AREA', "Rim", loc=(-4, -1, 15), target=(0, -3.4, 12.6), energy=900, color=(0.6, 0.7, 1.0), size=2.0)
Z = LP.ROCKET_Z0
pin = V((0.0, -rocket.LUG_PIN, Z + 0.56))
views = {
    "s19c": ((1.6, -5.6, Z + 0.2), pin + V((0.08, -0.18, -0.03)), 55, 2.2),
    "wide": ((3.5, -8.5, Z + 0.5), V((0, -3.4, Z - 1.3)), 35, 8.0),
    "side": ((4.2, -3.7, Z - 0.6), V((0, -3.6, Z - 1.0)), 28, 8.0),
}
loc, tgt, lens, fs = views[view]
cam = sb.camera("Cam", loc=loc, target=tgt, lens=lens, fstop=fs, clip=(0.02, 500))
cam.data.dof.focus_distance = (V(loc) - V(tgt)).length
d = os.path.join(sb.ROOT, "lookdev", "clamp")
os.makedirs(d, exist_ok=True)
sc.render.filepath = os.path.join(d, view + ".png")
bpy.ops.render.render(write_still=True)
