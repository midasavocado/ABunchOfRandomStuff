"""s06a (450-506): over the researcher's shoulder - the AI-assisted design system converges on the component on a
real workstation display (block -> lattice -> generative link). s06b (506-540): closer on the screen - the final
choice is selected (amber outline) and the part settles into its 3/4 hero view. Screen content: scenes/s06_screen.py
+ lib/screenui.py -> assets/screen_s06/scr_NNNN.jpg (emissive image sequence)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, office, folks, prosthesis as P, dbgcam
from mathutils import Vector as V, Euler

sid = (sb.argv() or ["s06a"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
F0, F1 = 450, 540

# ------------------------------------------------------------------ set
top_z = office.desk()
scr_m, scr_em = office.screen_mat("ScreenM", os.path.join(sb.ROOT, "assets", "screen_s06", "scr_0450.jpg"), 450, 90, strength=1.0)
screen, scr_cz = office.monitor(scr_m, loc=(0.0, 0.40, top_z))
office.keyboard((-0.02, 0.10, top_z + 0.006))
smouse = office.space_mouse((0.30, 0.14, top_z))
win = office.room(top_z)
# printed earlier iterations lying on the desk (white resin prints), and a mug
resin = sb.mat("Resin", (0.86, 0.86, 0.84), rough=0.35, sss=0.3, sss_radius=(1, 1, 1), sss_scale=0.003, spec=0.4)
for i, (v, loc, rz) in enumerate((("v1", (-0.34, 0.17, top_z + 0.0076), 20), ("v2", (-0.29, 0.25, top_z + 0.0076), -35))):
    r, o = P.component(v, "Print" + v, mat=resin, bushings=False)
    r.location = loc
    r.rotation_euler = (0, 0, math.radians(rz))
mug = sb.lathe("Mug", [(0.0, 0.0), (0.040, 0.0), (0.042, 0.004), (0.043, 0.095), (0.040, 0.095), (0.039, 0.006), (0.0, 0.006)],
               segs=64, mat=sb.mat("MugM", (0.78, 0.77, 0.74), rough=0.25, coat=0.5))
mug.location = (0.52, 0.30, top_z)
notebook = sb.prim("cube", "Notebook", loc=(-0.52, 0.12, top_z + 0.006), scale=(0.105, 0.148, 0.006), mat=sb.mat("NB", (0.12, 0.13, 0.15), rough=0.7))
notebook.rotation_euler = (0, 0, math.radians(8))
pen = sb.prim("cyl", "Pen", loc=(-0.47, 0.10, top_z + 0.016), rot=(0, math.pi / 2, math.radians(30)), vertices=16, radius=0.0045, depth=0.14,
              mat=sb.brushed_metal("PenM", (0.7, 0.7, 0.72), rough=0.25))

# chair (graphite mesh back) - mostly hidden by the researcher
chair = sb.prim("cube", "ChairSeat", loc=(0.02, -0.36, 0.46), scale=(0.24, 0.23, 0.035), mat=sb.fabric("ChairF", (0.08, 0.08, 0.09), weave=900))
sb.bevel(chair, 0.02, 3)

# ------------------------------------------------------------------ researcher (MPFB)
bm, rig, parts = folks.person("researcher")
folks.sit(rig, knee=88, hip=86)
folks.pose(rig, {"spine01": (-2, 0, 0), "spine03": (2, 0, 0), "neck01": (-4, 0, 0), "head": (-2, 0, -4)})
folks.place(rig, (0.02, -0.30, 0.0), 180.0)          # MPFB faces -Y: turn to face the desk (+Y)
bpy.context.view_layer.update()
pel = folks.bone_world(rig, "pelvis.L")
rig.location.z += (0.46 + 0.035 + 0.075) - pel.z
bpy.context.view_layer.update()
# hands: right on the 3D mouse cap, left forearm resting near the keyboard
eR = folks.reach(rig, "R", smouse.location + V((0.0, -0.035, 0.05)), elbow_hint=(0.26, -0.18, 0.86))
eL = folks.reach(rig, "L", (-0.20, 0.02, top_z + 0.04), elbow_hint=(-0.26, -0.18, 0.84))
print("REACH err", eR, eL)
folks.pose(rig, {"wrist.R": (8, 0, -10), "wrist.L": (0, 0, 8)})
for fb in ("finger2", "finger3", "finger4", "finger5"):
    for j in (1, 2, 3):
        folks.pose(rig, {f"{fb}-{j}.R": (25, 0, 0), f"{fb}-{j}.L": (18, 0, 0)})
bpy.context.view_layer.update()


def hand_reach(f):
    # small, purposeful 3D-mouse nudges while the design converges; still at the final choice
    t = (f - F0)
    return 3.0 * math.sin(t * 0.09) * (1 - sb.smooth((f - 505) / 12.0))


for f in range(F0 - 2, F1 + 2):
    pb = rig.pose.bones["wrist.R"]
    pb.rotation_mode = 'XYZ'
    pb.rotation_euler = Euler((math.radians(8 + hand_reach(f)), 0, math.radians(-10)), 'XYZ')
    pb.keyframe_insert("rotation_euler", frame=f)
    hb = rig.pose.bones["head"]
    hb.rotation_mode = 'XYZ'
    hb.rotation_euler = Euler((math.radians(-2 + 1.5 * sb.smooth((f - 500) / 20.0)), 0, math.radians(-4 + 2 * math.sin(f * 0.03))), 'XYZ')
    hb.keyframe_insert("rotation_euler", frame=f)

# ------------------------------------------------------------------ light: overcast daylight window + screen spill
sb.world_gradient(top=(0.10, 0.11, 0.13), horizon=(0.14, 0.145, 0.15), bottom=(0.05, 0.05, 0.05), strength=0.3)
sb.light('AREA', "WindowLight", loc=(-1.05, 1.55, 1.45), target=(-0.2, -0.3, 0.9), energy=80, color=(0.82, 0.88, 1.0), size=1.5, size_y=1.7)
sb.light('AREA', "ScreenSpill", loc=(0.0, 0.36, scr_cz), target=(0.0, -0.5, 1.15), energy=10, color=(0.92, 0.94, 1.0), size=0.7, size_y=0.4)
sb.light('AREA', "Ceiling", loc=(0.3, -0.3, 2.6), target=(0, 0, 0.8), energy=22, color=(1.0, 0.95, 0.88), size=1.2)
sb.light('AREA', "RimBack", loc=(0.9, -1.4, 1.6), target=(0, -0.4, 1.2), energy=25, color=(1.0, 0.86, 0.7), size=0.6)

# ------------------------------------------------------------------ camera
scr_c = V((0.0, 0.40, scr_cz))
cam = sb.camera("Cam", loc=(-0.4, -1.0, 1.3), target=scr_c, lens=50, fstop=2.2, clip=(0.02, 30))
if sid == "s06a":
    s0, s1 = 450, 506

    def pos(t):
        k = sb.smooth(t)
        return V((sb.lerp(-0.30, -0.24, k), sb.lerp(-0.86, -0.70, k), sb.lerp(1.21, 1.20, k)))

    def tg(t):
        return scr_c + V((0.03, 0, -0.02))

    sb.cam_bake(cam, s0, s1, pos, tg, lens=lambda t: sb.lerp(45, 50, t), focus=lambda t: (pos(t) - scr_c).length, fstop=2.4)
else:
    s0, s1 = 506, 540

    def pos(t):
        k = sb.smooth(t)
        return scr_c + V((sb.lerp(-0.22, -0.17, k), sb.lerp(-0.78, -0.68, k), sb.lerp(0.05, 0.04, k)))

    def tg(t):
        return scr_c + V((sb.lerp(0.03, 0.06, sb.smooth(t)), 0, -0.005))

    sb.cam_bake(cam, s0, s1, pos, tg, lens=lambda t: sb.lerp(44, 50, t), focus=lambda t: (pos(t) - tg(t)).length, fstop=2.8)

dbgcam.apply()
import soul
soul.alive_all(sid, breathe=0.7, calm=1.1)
sb.frames(s0, s1)
sb.save("s06")
sb.render_shot(sid)
