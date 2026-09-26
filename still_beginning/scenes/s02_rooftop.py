"""s02 LOOKING: match from the pupil to the telescope objective, pull back to reveal the child at a rooftop
telescope in blue-hour light; a small purposeful focus adjustment."""
import sys, os, math, json, subprocess, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, props, rooftop, skyline2
from mathutils import Vector as V, Matrix, Quaternion, Euler

ROOT = sb.ROOT
sid = (sb.argv() or ["s02"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "-0.2")),
                volumes=True, vol_tile=8, vol_samples=48)
F0, F1 = 90, 179

# ------------------------------------------------------------------ world
sb.world_bluehour(glow_az=8.0, glow=3.2, glow_col=(1.0, 0.28, 0.035), zenith=(0.010, 0.028, 0.12),
                  horizon=(0.09, 0.13, 0.26), glow_width=38, band_height=3.2, strength=1.0)
rooftop.roof_deck()
rooftop.roof_props()
skyline2.build(rmin=24, rmax=900, count=520, ground_z=-14.0, view_dir=math.pi / 2, view_half=math.radians(85))
rooftop.string_lights((-3.4, 1.4, 2.2), (0.6, 0.55, 1.95), n=12, sag=0.25, strength=16, bulb_w=7.0)
rooftop.string_lights((0.6, 0.55, 1.95), (5.4, 2.6, 2.3), n=12, sag=0.3, strength=16, bulb_w=7.0)

# ------------------------------------------------------------------ the child (MPFB, rigged), posed into the eyepiece
import mhchild
bm, rig, parts = mhchild.build()
mhchild.fix_eyes(parts)
import soul
# right eye at the eyepiece: the free left eye half-closes (everyone squints it at a telescope), brows lifted in
# concentration, the start of a smile at what she sees
soul.set_units([o for o in rig.children_recursive if o.type == 'MESH'],
               {"eye-left-closure": 0.55, "eyebrows-*-inner-up": 0.25, "mouth-corner-puller": 0.22, "mouth-parling": 0.15})
mhchild.recolor_top(parts)
mhchild.add_cuffs(rig, parts)
BEND = dict(spine01=5, spine02=6, spine03=6, spine04=4, neck01=11, neck02=9, head=13)
for n, a_ in BEND.items():
    mhchild.pose_bone(rig, n, rot=(a_, 0, 0))
# weight shift: slight knee softness
mhchild.pose_bone(rig, "upperleg01.R", rot=(-4, 0, 0)); mhchild.pose_bone(rig, "lowerleg01.R", rot=(8, 0, 0))
bpy.context.view_layer.update()


def eye_world(side=-1):
    eo = next(o for o in parts if "high-poly" in o.name)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = eo.evaluated_get(dg)
    me = ev.to_mesh()
    pts = [ev.matrix_world @ v.co for v in me.vertices]
    ev.to_mesh_clear()
    cx = sum(p.x for p in pts) / len(pts)
    sel = [p for p in pts if (p.x - cx) * side > 0]
    c = sum(sel, V()) / len(sel)
    # front of the cornea: most -Y-ish point along gaze later; use centre for now
    return c


hb = rig.pose.bones["head"]
R_rest = (rig.matrix_world @ rig.data.bones["head"].matrix_local).to_3x3()
R_pose = (rig.matrix_world @ hb.matrix).to_3x3()
gaze = (R_pose @ R_rest.inverted() @ V((0, -1, 0))).normalized()
eye_r = eye_world(-1) + gaze * 0.012            # cornea front of the right eye

T = props.telescope(alt_deg=32.0, az_deg=180.0, height=0.98, ep_tilt=2.0)
bpy.context.view_layer.update()
# tilt the eyepiece so its axis points straight back along the child's gaze
ep_axis = (T["ep_dir"].matrix_world.to_3x3() @ V((0, 0, 1))).normalized()
want = -gaze
axX = (T["alt"].matrix_world.to_3x3() @ V((1, 0, 0))).normalized()
def _proj(v):
    return (v - axX * v.dot(axX)).normalized()
ang = _proj(ep_axis).angle(_proj(want))
sgn = 1 if axX.dot(_proj(ep_axis).cross(_proj(want))) > 0 else -1
T["ep_dir"].rotation_euler.x += sgn * ang
bpy.context.view_layer.update()
cup = T["eyepiece"].matrix_world.translation.copy()
ep_axis = (T["ep_dir"].matrix_world.to_3x3() @ V((0, 0, 1))).normalized()
T["root"].location += (eye_r + gaze * 0.018) - (cup + ep_axis * 0.006)
bpy.context.view_layer.update()
knob = T["knobs"][0]                      # knob on the child's LEFT (camera side) so the adjustment is seen
knob_c = knob.matrix_world.translation.copy()
knob_axis = (knob.matrix_world.to_3x3() @ V((0, 0, 1))).normalized()
if knob_axis.x < 0:
    knob_axis = -knob_axis


def ik_arm(side, target, pole):
    tgt = sb.empty(f"IK_{side}", loc=target)
    pl = sb.empty(f"POLE_{side}", loc=pole)
    c = rig.pose.bones[f"lowerarm02.{side}"].constraints.new('IK')
    c.target = tgt
    c.pole_target = pl
    c.pole_angle = math.radians(-90)
    c.chain_count = 4
    c.use_tail = True
    return tgt, pl


# left hand: palm on the knob's outer face, fingertips wrapping its knurled rim.
# measure the hand's true fingers/palm axes in the rest pose (robust to the rig's bone rolls)
def hand_frame(side):
    bpy.context.view_layer.update()
    W = rig.matrix_world @ rig.pose.bones[f"wrist.{side}"].head
    F = rig.matrix_world @ rig.pose.bones[f"finger3-3.{side}"].tail
    pb = rig.pose.bones[f"finger3-1.{side}"]
    old = pb.rotation_euler.copy(); pb.rotation_mode = 'XYZ'
    pb.rotation_euler = (math.radians(60), 0, 0)
    bpy.context.view_layer.update()
    F2 = rig.matrix_world @ rig.pose.bones[f"finger3-3.{side}"].tail
    pb.rotation_euler = old
    bpy.context.view_layer.update()
    fd = (F - W).normalized()
    palm = (F2 - F); palm = (palm - fd * palm.dot(fd)).normalized()
    Rb = (rig.matrix_world @ rig.pose.bones[f"wrist.{side}"].matrix).to_3x3()
    return fd, palm, Rb, (F - W).length


def frame(fd, palm):
    sd = fd.cross(palm).normalized()
    return Matrix((sd, fd, palm)).transposed()


fdL, palmL, RbL, handlen = hand_frame("L")
fingers_des = (V((0, -0.55, -1.0)) - knob_axis * V((0, -0.55, -1.0)).dot(knob_axis)).normalized()
palm_des = -knob_axis
R_des = frame(fingers_des, palm_des) @ frame(fdL, palmL).transposed() @ RbL
palm_c = knob_c + knob_axis * 0.028                     # palm 2.8 cm off the knob's axis centre (knob face +9 mm)
wrist_target = palm_c - fingers_des * (handlen * 0.42)
ikL, poleL = ik_arm("L", wrist_target, wrist_target + V((0.35, 0.25, -0.2)))
wrist_rot = sb.empty("WristRotL", loc=wrist_target)
wrist_rot.matrix_world = Matrix.Translation(wrist_target) @ R_des.to_4x4()
cr = rig.pose.bones["wrist.L"].constraints.new('COPY_ROTATION')
cr.target = wrist_rot
for fb, a_ in (("finger2", 28), ("finger3", 32), ("finger4", 38), ("finger5", 44)):
    for j, k in ((1, 0.8), (2, 1.2), (3, 0.9)):
        mhchild.pose_bone(rig, f"{fb}-{j}.L", rot=(a_ * k, 0, 0))
mhchild.pose_bone(rig, "finger1-2.L", rot=(15, 0, 0)); mhchild.pose_bone(rig, "finger1-3.L", rot=(20, 0, 0))
# right hand: resting on the tube behind the focuser
tube_top = T["alt"].matrix_world @ V((0, -0.20, 0.052))
ikR, poleR = ik_arm("R", tube_top + V((-0.045, 0.03, 0.035)), tube_top + V((-0.4, 0.3, -0.25)))
for fb in ("finger2", "finger3", "finger4", "finger5"):
    for j in (1, 2, 3):
        mhchild.pose_bone(rig, f"{fb}-{j}.R", rot=(28, 0, 0))
bpy.context.view_layer.update()
head_c = eye_r
wrist_l = wrist_target

# ------------------------------------------------------------------ the focus adjustment: knob and hand turn together
F_ADJ0, F_ADJ1 = 140, 156
knob_r0 = knob.rotation_euler.copy()
pivot = sb.empty("KnobPivot", loc=knob_c)
bpy.context.view_layer.update()
for e in (ikL, wrist_rot):
    e.parent = pivot
    e.matrix_parent_inverse = pivot.matrix_world.inverted()
for f in range(F0 - 10, F1 + 11):
    t = sb.smoother(sb.remap(f, F_ADJ0, F_ADJ1))
    ang = math.radians(28.0) * t
    pivot.rotation_mode = 'QUATERNION'
    pivot.rotation_quaternion = Quaternion(knob_axis, ang)
    pivot.keyframe_insert("rotation_quaternion", frame=f)
    knob.rotation_mode = 'XYZ'
    knob.rotation_euler = (knob_r0.x, knob_r0.y, knob_r0.z + ang)
    knob.keyframe_insert("rotation_euler", frame=f)

# ------------------------------------------------------------------ lights
# festoon warm fill on the child's side, cool sky from above
for L in (sb.light('AREA', "SkyTop", loc=(0.0, -0.5, 6.0), target=(0, -0.3, 0.9), energy=60, color=(0.55, 0.66, 1.0), size=6.0),
          sb.light('AREA', "GlowRim", loc=(0.7, 2.6, 1.85), target=(0, -0.2, 1.05), energy=120, color=(1.0, 0.46, 0.14), size=1.2),
          sb.light('SPOT', "ScopeKick", loc=(-1.0, -2.2, 1.0), target=(0, -0.9, 1.25), energy=12, color=(0.6, 0.7, 1.0), size=0.3, spot=18)):
    L.visible_camera = False
    L.visible_glossy = L.name != "GlowRim"

rc = sb.light('AREA', "RimCatch", loc=(0.25, -2.0, 2.3), target=(0, -1.0, 1.4), energy=10, color=(1.0, 0.6, 0.3), size=0.25)
rc.visible_camera = False

# ------------------------------------------------------------------ camera: from the objective, pull back and around
bpy.context.view_layer.update()
obj_c, axis = props.objective_world(T)
child_focus = (head_c + wrist_l) * 0.5
LENS = 35.0
d0 = 0.60


def cpos(t):
    k = sb.ease_out(t, 2.4)
    d = sb.lerp(d0, 2.6, k)
    base = obj_c + axis * d
    side = V((1, 0, 0)) * sb.lerp(0.0, 1.25, sb.smoother(sb.remap(t, 0.05, 1.0)))
    drop = V((0, 0, 1)) * sb.lerp(0.0, 1.16 - (obj_c.z + axis.z * 2.6), sb.smoother(sb.remap(t, 0.0, 0.9)))
    return base + side + drop


def ctar(t):
    k = sb.smoother(sb.remap(t, 0.02, 0.85))
    return obj_c.lerp(child_focus + V((0, 0.05, -0.02)), k)


cam = sb.camera("Cam", loc=cpos(0), target=ctar(0), lens=LENS, fstop=2.8, clip=(0.01, 3000))
cam.data.dof.aperture_blades = 9
sb.cam_bake(cam, F0, F1, cpos, ctar,
            focus=lambda t: sb.lerp((cpos(t) - obj_c).length, (cpos(t) - child_focus).length, sb.smoother(sb.remap(t, 0.1, 0.7))),
            fstop=lambda t: sb.lerp(8.0, 2.8, sb.smoother(sb.remap(t, 0.0, 0.5))))

import soul
soul.alive_all(sid, head=0.15, sway=0.0, breathe=0.4, calm=1.3)
sb.frames(F0, F1 + 1)
if os.environ.get('SB_DIAG'):
    exec(open(os.environ['SB_DIAG']).read()); raise SystemExit
sb.save("s02")
sb.render_shot(sid)
