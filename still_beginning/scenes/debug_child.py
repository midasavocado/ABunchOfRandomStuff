import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, childmesh
from mathutils import Vector as V
a = sb.argv(); npz = a[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 600, 800
sc.eevee.taa_render_samples = 16
objs, meta = childmesh.load(npz)
sb.world_color((0.5, 0.5, 0.52), 1.0)
sb.light('SUN', "S", loc=(1, -1, 2), target=(0, 0, 0), energy=3.0)
sb.prim("plane", "G", scale=(3, 3, 1), mat=sb.mat("g", (0.3, 0.3, 0.3)))
hc = V(meta["head_c"])
views = [((1.8, 0, 0.7), (0, 0, 0.65), 1.6), ((0, -1.8, 0.7), (0, 0, 0.65), 1.6), ((-1.6, 1.2, 1.2), (0, 0, 0.65), 1.6),
         (tuple(hc + V((0.25, -0.4, 0.05))), tuple(hc), 0.4)]
for i, (p, t, sz) in enumerate(views):
    cam = sb.camera("C%d" % i, loc=p, target=t, lens=50)
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = sz
    sc.camera = cam
    sc.render.filepath = f"/tmp/_c{i}.png"
    bpy.ops.render.render(write_still=True)
