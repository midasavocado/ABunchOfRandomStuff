"""Lookdev: complete prosthetic hand grasping a glass (contact solver test). -- <out_prefix> [views]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, prosthesis as P
from mathutils import Vector as V
a = sb.argv(); out = a[0]
views = (a[1] if len(a) > 1 else "side,top,front").split(",")
sc = sb.reset()
sb.setup_render("EEVEE", mblur=False)
sc.render.resolution_x, sc.render.resolution_y = (1600, 900) if not os.environ.get("SB_LDSMALL") else (960, 540)
sb.world_gradient(top=(0.06, 0.065, 0.075), horizon=(0.14, 0.14, 0.145), bottom=(0.05, 0.05, 0.05), strength=0.8)
H = P.Hand("PH")
R, GC = 0.037, V((0.0, 0.058, -0.054))
axis = V((1, 0, 0))
glass = sb.prim("cyl", "Glass", loc=GC, rot=(0, math.pi / 2, 0), vertices=96, radius=R, depth=0.11,
                mat=sb.mat("GlassClay", (0.5, 0.6, 0.65), rough=0.2, alpha=0.35))
H.set_thumb_frame(d=(0.12, -0.70, -0.70), z=(0.1, -0.70, 0.70))
import numpy as np
clr = lambda pts: P.cyl_clearance(pts, GC, axis, R)
t = H.fingers['thumb']
for th in (0.0, 0.3, 0.6):
    t.set(th, th)
    print('THUMBDBG', th, 1000*clr(P.world_pts(t.link)), 1000*clr(P.world_pts(t.dmesh)), [round(v,3) for v in t.dmesh.matrix_world.translation])
for n, f in H.fingers.items():
    th = P.solve_curl(f, clr, th1_range=(0.0, 1.5), th2_ratio=0.9)
    print("GRASP", n, [round(math.degrees(x), 1) for x in th], "clear mm", round(1000 * min(clr(P.world_pts(f.link)), clr(P.world_pts(f.dmesh))), 3))
print("palm clear mm", 1000 * clr(P.world_pts(H.palm)))
sb.light('AREA', "Key", loc=(-0.3, -0.2, 0.4), target=(0, 0.05, 0), energy=30, color=(1, 0.95, 0.9), size=0.3)
sb.light('AREA', "Rim", loc=(0.3, 0.35, 0.2), target=(0, 0.05, 0), energy=20, color=(0.8, 0.88, 1.0), size=0.2)
sb.light('AREA', "Fill", loc=(0.3, -0.3, -0.1), target=(0, 0.05, 0), energy=6, size=0.5)
V_ = {"side": ((-0.40, 0.06, -0.03), (0, 0.06, -0.03)), "top": ((0.08, 0.02, 0.40), (0, 0.06, -0.02)),
      "front": ((-0.12, 0.40, 0.05), (0, 0.06, -0.03)), "persp": ((-0.25, -0.18, 0.22), (0, 0.06, -0.03)),
      "under": ((0.15, 0.25, -0.35), (0, 0.06, -0.03)), "tip": ((-0.12, 0.16, -0.09), (-0.02, 0.10, -0.06))}
for v in views:
    p, t = V_[v]
    cam = sb.camera("C" + v, loc=p, target=t, lens=50, fstop=16)
    sc.render.filepath = out + v + ".png"
    cam.data.lens = 50 if v != 'tip' else 70
    bpy.ops.render.render(write_still=True)
