"""Lookdev for lib/mars.py at every scale. views: orbit, high, mid, ground, limb. -> lookdev/mars/<view>.png"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, mars
from mathutils import Vector as V

views = sb.argv() or ["orbit"]
sc = sb.reset()
sb.setup_render("EEVEE", samples=int(os.environ.get("SB_SAMPLES", "32")), mblur=False)
SUN = mars.SUN_DAY
T = mars.Terrain()
cap = mars.cap(T=T, mat=mars.planet_mat("CapM", offset=(0, 0, mars.R), local=True, haze="Haze"))
pl = mars.planet(mat=mars.planet_mat("PlanetM"))
atm, ag = mars.atmosphere(sun=SUN)
sl = sb.light('SUN', 'Sun', energy=4.5, color=(1.0, 0.93, 0.84), angle=0.35)
sl.rotation_mode='QUATERNION'; sl.rotation_quaternion = SUN.to_track_quat('Z', 'Y')
w, smix = mars.world(SUN)
Rk = mars.R
cams = dict(orbit=((mars.QREF - mars.EAST * 0.45 + mars.POLE * 0.2).normalized() * 18000e3 + mars.C, mars.C, 50, 1.0),
            high=(V((0, -0.3, 1)).normalized() * 1200e3 + V((0, 0, 0)), mars.C * 0.3, 35, 1.0),
            mid=(V((0, -30e3, 28e3)), V((0, 20e3, 0)), 35, 0.6),
            ground=(V((-40, -120, 18)), V((0, 400, 30)), 30, 0.0),
            limb=(V((0, -3000e3, 900e3)) , V((0, 2000e3, -800e3)), 35, 1.0))
d = os.path.join(sb.ROOT, "lookdev", "mars")
os.makedirs(d, exist_ok=True)
for v in views:
    loc, tgt, lens, sm = cams[v]
    smix.outputs[0].default_value = sm
    ag.outputs[0].default_value = 1.0 if sm > 0.9 else 0.0
    cap.data.materials[0].node_tree.nodes["Haze"].outputs[0].default_value = 1.0 - sm
    cam = sb.camera("Cam_" + v, loc=loc, target=tgt, lens=lens, clip=(max(0.1, loc.length * 1e-5 if v != "ground" else 0.1), 1e8))
    mars.aim(cam, loc, tgt, up=mars.POLE if v in ("orbit", "high", "limb") else (0, 0, 1))
    sc.render.filepath = os.path.join(d, v + ".png")
    bpy.ops.render.render(write_still=True)
