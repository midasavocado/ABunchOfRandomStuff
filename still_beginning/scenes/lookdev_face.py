"""Lookdev: MPFB face expression test. -- <preset> <smile> <out.png>"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, folks
from mathutils import Vector as V
a = sb.argv()
preset, smile, out = a[0], float(a[1]), a[2]
sc = sb.reset()
sb.setup_render("CYCLES", cycles_samples=48, mblur=False)
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sb.world_gradient(top=(0.2, 0.2, 0.22), horizon=(0.3, 0.28, 0.26), bottom=(0.1, 0.1, 0.1), strength=0.6)
bm, rig, parts = folks.person(preset)
import json
ov = json.loads(os.environ.get("SB_FACE", "{}"))
folks.expression(rig, smile=smile)
if os.environ.get("SB_SOULFACE"):
    import soul
    soul.face(parts + [bm], os.environ["SB_SOULFACE"])
if os.environ.get("SB_UNITS"):
    import soul
    soul.set_units(parts + [bm], json.loads(os.environ["SB_UNITS"]))
for b, (loc, rot) in ov.items():
    pb = rig.pose.bones[b]; pb.location = loc; pb.rotation_mode = 'XYZ'; pb.rotation_euler = [math.radians(r) for r in rot]
bpy.context.view_layer.update()
h = (folks.bone_world(rig, "eye.L") + folks.bone_world(rig, "eye.R")) / 2 + V((0, 0, -0.03))
sb.light('AREA', "Key", loc=h + V((-0.8, -1.2, 0.5)), target=h, energy=120, color=(1, 0.9, 0.8), size=1.0)
sb.light('AREA', "Rim", loc=h + V((0.8, 0.8, 0.4)), target=h, energy=60, color=(0.8, 0.88, 1.0), size=0.6)
cam = sb.camera("C", loc=h + V((-0.3, -1.0, 0.02)), target=h, lens=float(os.environ.get("SB_FLENS", "85")), fstop=4)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
