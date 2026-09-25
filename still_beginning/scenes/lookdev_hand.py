import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, handmesh
from mathutils import Vector as V
a = sb.argv()
npz = a[0]; out = a[1]
view = a[2] if len(a) > 2 else "top"
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=64, mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
sc.cycles.samples = 64
h, parts = handmesh.load(npz)
sb.world_gradient(top=(0.3, 0.35, 0.45), horizon=(0.6, 0.55, 0.5), bottom=(0.2, 0.18, 0.15), strength=0.8)
sb.light('AREA', "Key", loc=(-0.25, -0.1, 0.35), target=(0, 0.05, 0), energy=10, color=(1, 0.93, 0.85), size=0.25)
sb.light('AREA', "Rim", loc=(0.3, 0.35, 0.15), target=(0, 0.05, 0), energy=6, color=(0.8, 0.88, 1.0), size=0.2)
views = {"top": ((0.12, -0.10, 0.33), (0, 0.045, 0)), "side": ((-0.36, 0.06, 0.05), (0, 0.045, 0)), "front": ((-0.12, 0.34, 0.12), (0, 0.05, 0)), "thumb": ((-0.25, 0.22, 0.18), (0, 0.06, 0))}
p, t = views[view]
cam = sb.camera("C", loc=p, target=t, lens=60, fstop=11)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
