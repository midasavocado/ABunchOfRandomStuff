import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, plants
a = sb.argv(); out = a[0]
sc = sb.reset()
sb.setup_render("EEVEE", samples=32, mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
sb.world_gradient(top=(0.25, 0.22, 0.2), horizon=(0.5, 0.4, 0.3), bottom=(0.1, 0.08, 0.06), strength=0.6)
soil = sb.mat("Soil", (0.05, 0.035, 0.025), rough=0.9)
sb.prim("plane", "Soil", size=2, mat=soil)
B = plants.basil("Basil", (0, 0, 0), seed=3)
L = plants.lettuce("Lettuce", (0.25, 0.1, 0), seed=2)
wm = plants.droplet_mat()
for lf in B["leaves"][:6]:
    plants.droplets("Drop", lf, n=3, mat=wm, seed=hash(lf.name) % 1000)
sb.sun(25, 140, energy=4.0, color=(1.0, 0.9, 0.78))
sb.light('AREA', "Grow", loc=(0, 0, 1.0), target=(0, 0, 0), energy=60, color=(1.0, 0.92, 0.85), size=1.0)
cam = sb.camera("C", loc=(0.1, -0.55, 0.38), target=(0.06, 0, 0.17), lens=50, fstop=4)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
