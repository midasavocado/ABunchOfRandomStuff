import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, mhchild
from mathutils import Vector as V
out = (sb.argv() or ["/tmp/ld_child.png"])[0]
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=64, mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 1600, 900
bm, rig, parts = mhchild.build()
mhchild.fix_eyes(parts)
mhchild.recolor_top(parts)
cuffs = mhchild.add_cuffs(rig, parts)
import math
mhchild.pose_bone(rig, "upperarm01.L", rot=(0, 0, 0)); mhchild.pose_bone(rig, "upperarm01.R", rot=(0, 0, 0))
for side, sgn in (("L", 1), ("R", -1)):
    mhchild.pose_bone(rig, "shoulder01." + side, rot=(0, 0, sgn * -38))
    mhchild.pose_bone(rig, "lowerarm01." + side, rot=(-25, 0, 0))
mhchild.pose_bone(rig, "neck01", rot=(12, 0, 6))
print("PARTS", [p.name for p in parts], "RIG", rig.name if rig else None)
print("BONES", [b.name for b in rig.data.bones][:80] if rig else None)
bpy.context.view_layer.update()
zs = [ (o.matrix_world @ V(b)).z for o in parts for b in o.bound_box]
print("HEIGHT", max(zs) - min(zs))
sb.world_gradient(top=(0.35, 0.4, 0.5), horizon=(0.6, 0.55, 0.5), bottom=(0.2, 0.18, 0.15), strength=0.9)
sb.light('AREA', "Key", loc=(-0.8, -1.2, 1.9), target=(0, 0, 1.1), energy=120, color=(1, 0.93, 0.85), size=0.8)
sb.light('AREA', "Rim", loc=(0.9, 1.0, 1.6), target=(0, 0, 1.1), energy=80, color=(0.8, 0.88, 1.0), size=0.6)
sb.prim("plane", "G", scale=(4, 4, 1), mat=sb.mat("g", (0.3, 0.3, 0.3)))
top = max(zs)
cam = sb.camera("C", loc=(0.35, -1.6, top - 0.1), target=(0, 0, top - 0.25), lens=85, fstop=4)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
cam.location = (1.0, -3.6, 0.9); cam.rotation_mode = 'QUATERNION'; cam.rotation_quaternion = sb.look_quat(cam.location, (0, 0, top * 0.5))
cam.data.lens = 50
sc.render.filepath = out.replace(".png", "_full.png")
bpy.ops.render.render(write_still=True)
