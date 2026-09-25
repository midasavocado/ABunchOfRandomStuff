"""Lookdev: MPFB procedural walk (side view contact sheet)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, folks
from mathutils import Vector as V
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=16, mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 480, 540
sb.world_gradient(top=(0.3, 0.3, 0.32), horizon=(0.4, 0.38, 0.36), bottom=(0.2, 0.2, 0.2), strength=1.0)
sb.prim("plane", "G", size=40, mat=sb.mat("G", (0.3, 0.3, 0.3)))
bm, rig, parts = folks.person("teen")
bpy.context.view_layer.update()
foot = folks.bone_world(rig, "foot.L")
folks.walk(rig, 1, 60, V((0, 0, rig.location.z)), 0.0, speed=1.3)
out = sb.argv()[0]
for i, f in enumerate((10, 14, 18, 22)):
    sc.frame_set(f)
    x = rig.location.x
    cam = sb.camera("C%d" % i, loc=(x, -4.5, 1.0), target=(x, 0, 0.9), lens=40)
    sc.render.filepath = out + "%d.png" % i
    bpy.ops.render.render(write_still=True)
