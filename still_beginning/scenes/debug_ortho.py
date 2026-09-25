import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, handmesh
from mathutils import Vector as V
a = sb.argv(); npz = a[0]; out = a[1]
extra = a[2] if len(a) > 2 else None
sc = sb.reset()
sb.setup_render("EEVEE", mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 600, 600
sc.eevee.taa_render_samples = 8
h, parts = handmesh.load(npz, nails=True)
import json, numpy as np
for n in json.loads(str(np.load(npz)["N"])):
    if n["name"] == "__pencil__":
        extra = ",".join(str(x) for x in n["c"] + n["d"])
if extra:   # optional pencil: "x,y,z,dx,dy,dz"
    v = [float(t) for t in extra.split(",")]
    p0 = V(v[:3]); d = V(v[3:]).normalized()
    pen = sb.prim("cyl", "Pencil", loc=p0 + d * 0.04, vertices=6, radius=0.0038, depth=0.17, mat=sb.mat("pen", (0.9, 0.55, 0.1)))
    pen.rotation_mode = 'QUATERNION'; pen.rotation_quaternion = d.to_track_quat('Z', 'Y')
sb.world_color((0.5, 0.5, 0.52), 1.0)
sb.light('SUN', "S", loc=(1, -1, 2), target=(0, 0, 0), energy=3.0)
views = [((0, 0.03, 0.5), (0, 0.03, 0), "TOP"), ((-0.5, 0.03, 0), (0, 0.03, 0), "THUMB SIDE"), ((0, 0.5, 0.0), (0, 0.03, 0), "FRONT"), ((0.3, -0.25, 0.3), (0, 0.03, 0), "PERSP")]
for i, (p, t, nm) in enumerate(views):
    cam = sb.camera("C%d" % i, loc=p, target=t, lens=50)
    if i < 3:
        cam.data.type = 'ORTHO'; cam.data.ortho_scale = 0.2
    if nm == "TOP":
        cam.rotation_mode = 'XYZ'; cam.rotation_euler = (0, 0, 0)
    sc.camera = cam
    sc.render.filepath = f"/tmp/_o{i}.png"
    bpy.ops.render.render(write_still=True)
