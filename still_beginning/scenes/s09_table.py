"""s09a (720-765, FIRST DROP): at the family table, the wearer's prosthetic hand closes around a glass of water -
fingers settle one after another on the glass - and lifts it. Warm pendant light, candle bokeh, the dusk window.
s09b (765-810): the reveal - the whole family around the dinner table (the film's child beside her mother, the
partner, grandma, the teen at the window end), the mother lifting the glass; the camera eases back and up along
the table; the twilight city glows beyond the window. Backlit, warm, observed rather than posed."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, home, folks, mhchild, prosthesis as P, dbgcam
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s09a"])[0]
_, F0, F1 = sb.TL.shot(sid)[:3]
sc = sb.reset()
# Cycles for the final: glass + water refraction and candle caustic glints must be physically right here
sb.setup_render("CYCLES", cycles_samples=160, mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
sc.cycles.transmission_bounces = 16
sc.cycles.max_bounces = 16
M = home.materials()
home.build(city=True)
Z = home.TABLE_Z
SEATS = {"wearer": (0.35, -0.74, math.pi / 2), "child": (-0.35, -0.74, math.pi / 2), "partner": (0.35, 0.74, -math.pi / 2),
         "grandma": (-0.4, 0.74, -math.pi / 2), "teen": (-1.25, 0.0, 0.0)}
home.set_table(M, list(SEATS.values()))
for n, (x, y, rz) in SEATS.items():
    home.chair("Chair_" + n, (x - math.cos(rz) * 0.08, y - math.sin(rz) * 0.08, 0), rz + math.pi / 2, M)

# ------------------------------------------------------------------ the glass and the prosthetic grasp
GL = V((0.36, -0.36, Z + 0.003))                 # glass base centre (in front of the wearer, right of her plate)
gl, water = home.glass("HeroGlass", GL, M)
R_G, H_G = 0.036, 0.11
H = P.Hand("PH", socket=True, tendons=False)
GC_hand = V((0.0, 0.058, -0.054))               # glass axis point in hand frame (lookdev_phand grasp)
# hand frame in world: glass axis = hand X (thumb up => hand X = world -Z); fingers point across the glass
yaw = math.radians(128.0)                        # fingers wrap from the wearer's right side, forearm from her
hy = V((math.cos(yaw), math.sin(yaw), 0.0))
hx = V((0.0, 0.0, -1.0))
hz = hx.cross(hy)
Rh = Matrix((hx, hy, hz)).transposed()
grip_h = 0.05                                    # grip height above the glass base
root_grasp = GL + V((0, 0, grip_h)) - Rh @ GC_hand
H.root.rotation_mode = 'QUATERNION'
H.root.rotation_quaternion = Rh.to_quaternion()
H.set_thumb_frame(d=(0.12, -0.70, -0.70), z=(0.1, -0.70, 0.70))
H.root.location = root_grasp
WRIST = sb.empty("WristFlex", parent=H.root)
WRIST.rotation_mode = 'XYZ'
for o in (H.socket, H.ring):
    o.parent = WRIST


def flex(f):
    """Wrist flexion (degrees about hand Z): forearm slopes down to the elbow, more as the glass rises."""
    k = sb.smoother((f - 766) / (TOAST_F - 766)) if sid == "s09b" else 0.0
    return 14.0 + 10.0 * k + 4.0 * sb.smoother((f - 750) / 24.0)


bpy.context.view_layer.update()
axis_w = V((0, 0, 1))
clr = lambda pts: P.cyl_clearance(pts, GL + V((0, 0, grip_h)), axis_w, R_G * 1.02)
closed = {}
for n, f in H.fingers.items():
    closed[n] = P.solve_curl(f, clr, th1_range=(0.0, 1.5), th2_ratio=0.9)
open_ = {n: (0.12, 0.1) for n in H.fingers}
open_["thumb"] = (0.0, 0.0)

T_TOUCH = 736
TOAST_F = 792            # hand arrives around the glass (after the downbeat at 720 it is already moving)
ORDER = ["index", "middle", "ring", "pinky", "thumb"]




def lift_vec(f):
    """Glass offset from its table position: a gentle lift (s09a), continuing into a toast toward the table centre."""
    up = 0.06 * sb.smoother((f - 750) / 24.0)
    k = sb.smoother((f - 766) / (TOAST_F - 766))
    clink = 0.004 * math.sin(max(0.0, f - TOAST_F) * 0.9) * math.exp(-max(0.0, f - TOAST_F) * 0.35)
    return V((0.0, 0.05 * k, up + 0.03 * k))


def hand_world(f):
    """Approach from behind/right (along -fingers dir, a little higher), arrive, then lift and toast with the glass."""
    a = sb.smoother((f - 712) / (T_TOUCH - 712))
    back = (1 - a) * 0.09
    up = (1 - a) * 0.025
    L = lift_vec(f)
    return root_grasp - hy * back + V((0, 0, up)) + L, L


def curl(n, f):
    i = ORDER.index(n)
    k = sb.smoother((f - (T_TOUCH - 6 + i * 1.6)) / 9.0)
    a0, b0 = open_[n]
    a1, b1 = closed[n]
    return a0 + (a1 - a0) * k, b0 + (b1 - b0) * k


# ------------------------------------------------------------------ the family
people = {}


def seat_person(key, preset=None, child=False):
    x, y, rz = SEATS[key]
    if child:
        bm, rig, parts = mhchild.build(name="Child")
        mhchild.fix_eyes(parts)
        mhchild.recolor_top(parts)
        mhchild.add_cuffs(rig, parts)
    else:
        bm, rig, parts = folks.person(preset or key)
    folks.sit(rig, knee=88, hip=84)
    folks.place(rig, (x - math.cos(rz) * 0.12, y - math.sin(rz) * 0.12, 0.0), math.degrees(rz) - 90.0 + 180.0)
    bpy.context.view_layer.update()
    pel = folks.bone_world(rig, "pelvis.L")
    rig.location.z += 0.47 + 0.03 + 0.07 - pel.z
    bpy.context.view_layer.update()
    people[key] = (bm, rig, parts)
    return bm, rig, parts


def forearms_on_table(rig, key, lx=0.16, rx=0.16, fwd=0.2, dz=0.05):
    x, y, rz = SEATS[key]
    d = V((math.cos(rz), math.sin(rz), 0)); s = V((-d.y, d.x, 0))
    base = V((x, y, Z + dz))
    folks.reach(rig, "L", base + d * fwd + s * lx, elbow_hint=base + s * (lx + 0.07) - d * 0.05)
    folks.reach(rig, "R", base + d * fwd - s * rx, elbow_hint=base - s * (rx + 0.07) - d * 0.05)
    folks.hand(rig, "L", "relaxed"); folks.hand(rig, "R", "relaxed", seed=1)


# which sign of rotation about hand Z drops the socket end (world -Z)?
WRIST.rotation_euler = (0, 0, math.radians(20)); bpy.context.view_layer.update()
_zp = (H.root.matrix_world @ WRIST.matrix_local @ V((0, -0.2, 0))).z
WRIST.rotation_euler = (0, 0, math.radians(-20)); bpy.context.view_layer.update()
_zm = (H.root.matrix_world @ WRIST.matrix_local @ V((0, -0.2, 0))).z
FLEX_SIGN = 1.0 if _zp < _zm else -1.0
with_people = os.environ.get("SB_NOPEOPLE") != "1"
if with_people:
    # the wearer: MPFB forearm + hand replaced by the prosthesis and a sleeve that follows it
    bm, rig, parts = seat_person("wearer")
    folks.amputate(parts, "R")
    wx, wy, wrz = SEATS["wearer"]
    folks.reach(rig, "L", V((wx - 0.06, wy + 0.1, Z - 0.17)), elbow_hint=V((wx - 0.22, wy - 0.02, Z - 0.1)))   # in her lap
    folks.pose(rig, {"head": (-8, 0, 8), "neck01": (-6, 0, 4)})
    shirt_m = folks.garment_material(parts, "casualsuit") or M["cloth"]
    sleeve_parts = []
    # right arm: IK the wrist onto the prosthetic socket end for each frame (keyed), sleeve rebuilt as a hooked tube
    elbow_e = sb.empty("ElbowT")
    sock_e = sb.empty("SocketT", parent=WRIST)
    sock_e.location = (0, -0.16, -0.004)
    elb_goal = sb.empty("ElbowGoal", parent=WRIST)
    elb_goal.location = (0.01, -0.275, -0.012)
    for f in range(F0 - 2, F1 + 2):
        rw, _ = hand_world(f)
        H.root.location = rw
        WRIST.rotation_euler = (0, 0, math.radians(FLEX_SIGN * flex(f)))
        bpy.context.view_layer.update()
        tgt = elb_goal.matrix_world.translation
        # she leans into the toast: spine + clavicle join the upper arm so the elbow really meets the socket
        folks.reach(rig, "R", tgt, end="lowerarm01", iters=30,
                    bones=[("spine03", (0, 2)), ("spine02", (0,)), ("clavicle.R", (0, 2)), ("upperarm01.R", (0, 1, 2))])
        for b in ("spine03", "spine02", "clavicle.R", "upperarm01.R"):
            rig.pose.bones[b].keyframe_insert("rotation_euler", frame=f)
        bpy.context.view_layer.update()
        elbow_e.location = folks.bone_world(rig, "lowerarm01.R")
        elbow_e.keyframe_insert("location", frame=f)
    # sleeve: fabric tube from the elbow to over the socket end, hooked to both (so it follows the motion)
    cu = bpy.data.curves.new("SleeveR", 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 0.036; cu.bevel_resolution = 8
    cu.use_fill_caps = False
    sp = cu.splines.new('POLY'); sp.points.add(2)
    sl = bpy.data.objects.new("SleeveR", cu); sb.link_obj(sl); cu.materials.append(shirt_m)
    sp.points[0].radius = 1.15; sp.points[1].radius = 1.0; sp.points[2].radius = 0.92
    bpy.context.view_layer.update()
    for i, h in enumerate((elbow_e, None, sock_e)):
        p = elbow_e.matrix_world.translation if i == 0 else (sock_e.matrix_world.translation if i == 2 else
                                                             (elbow_e.matrix_world.translation + sock_e.matrix_world.translation) / 2)
        sp.points[i].co = (*p, 1)
    for i, h in enumerate((elbow_e, sock_e)):
        md = sl.modifiers.new("H%d" % i, 'HOOK'); md.object = h; md.vertex_indices_set([0 if i == 0 else 2])
    md = sl.modifiers.new("Hm", 'HOOK'); md.object = elbow_e; md.vertex_indices_set([1]); md.strength = 0.5
    md2 = sl.modifiers.new("Hm2", 'HOOK'); md2.object = sock_e; md2.vertex_indices_set([1]); md2.strength = 0.5
    cu.resolution_u = 16

    folks.expression(rig, smile=0.5)
    seat_person("child", child=True)
    forearms_on_table(people["child"][1], "child", lx=0.12, rx=0.1, fwd=0.14, dz=0.04)
    folks.expression(people["child"][1], smile=0.8)
    seat_person("partner")
    forearms_on_table(people["partner"][1], "partner", lx=0.18, rx=0.14, fwd=0.2) if False else \
        folks.reach(people["partner"][1], "L", V((SEATS["partner"][0] - 0.16, SEATS["partner"][1] - 0.22, Z + 0.04)))
    folks.expression(people["partner"][1], smile=0.7)
    folks.reach(people["partner"][1], "R", V((SEATS["partner"][0] + 0.14, SEATS["partner"][1] - 0.2, Z + 0.04)))
    seat_person("grandma")
    forearms_on_table(people["grandma"][1], "grandma", lx=0.14, rx=0.16, fwd=0.16)
    folks.expression(people["grandma"][1], smile=0.9)
    seat_person("teen")
    forearms_on_table(people["teen"][1], "teen", lx=0.1, rx=0.08, fwd=0.26)
    folks.expression(people["teen"][1], smile=0.6)
    for key, (bm_, rg, pts) in people.items():
        # everyone's gaze goes to the glass in her hand (the moment); keyed so heads turn with it
        for f in (F0 - 2, (F0 + F1) // 2, F1 + 1):
            bpy.context.scene.frame_set(f)
            wr = people["wearer"][1]
            gp = (folks.bone_world(wr, "eye.L") + folks.bone_world(wr, "eye.R")) / 2
            if key == "wearer":
                gp = V((SEATS["partner"][0], SEATS["partner"][1], 1.2))   # she looks across at her partner
            if key == "grandma" and f > F0:
                gp = gp + V((0.0, 0.0, -0.12))
            folks.look_at(rg, gp)
            for b in ("neck02", "head"):
                rg.pose.bones[b].keyframe_insert("rotation_euler", frame=f)

# keyframe the hand + fingers + glass
for f in range(F0 - 2, F1 + 2):
    rw, lift = hand_world(f)
    H.root.location = rw
    H.root.keyframe_insert("location", frame=f)
    WRIST.rotation_euler = (0, 0, math.radians(FLEX_SIGN * flex(f)))
    WRIST.keyframe_insert("rotation_euler", frame=f)
    for n, fg in H.fingers.items():
        a, b = curl(n, f)
        fg.set(a, b, frame=f)
    for o in (gl, water):
        o.location = GL + lift
        o.keyframe_insert("location", frame=f)

# ------------------------------------------------------------------ cameras
if sid == "s09a":
    tgt = GL + V((0, 0, 0.055))
    CO = V([float(x) for x in os.environ.get("SB_S09_CAM", "-0.3,0.5,0.1").split(",")])
    cam = sb.camera("Cam", loc=tgt + CO, target=tgt, lens=85, fstop=2.0, clip=(0.01, 200))

    def cpos(t):
        k = sb.smooth(t)
        return tgt + CO * sb.lerp(1.0, 0.94, k) + V((0, 0, 0.02 * k))

    sb.cam_bake(cam, F0, F1, cpos, lambda t: tgt + V((0, 0, 0.03 * sb.smooth(t))), focus="target")
else:
    g0 = GL + lift_vec(F0) + V((0, 0, 0.06))
    look0, look1 = g0, V((-0.45, -0.05, Z + 0.3))
    p0, p1 = g0 + V((0.55, 0.3, 0.16)), V((1.95, -0.3, 1.34))

    def ease(t):
        return sb.smoother(min(1.0, t * 1.15)) * 0.85 + t * 0.15

    cam = sb.camera("Cam", loc=p0, target=look0, lens=35, fstop=2.8, clip=(0.02, 2000))
    sb.cam_bake(cam, F0, F1, lambda t: p0.lerp(p1, ease(t)), lambda t: look0.lerp(look1, ease(t)),
                focus=lambda t: (p0.lerp(p1, ease(t)) - g0.lerp(V((-0.2, -0.1, Z + 0.3)), ease(t))).length)

dbgcam.apply()
import soul
soul.alive_all(sid, calm=0.85, per={"Wearer": dict(breathe=0.0, sway=0.0, head=0.6)})
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
