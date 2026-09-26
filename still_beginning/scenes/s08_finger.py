"""s08a (630-675): on the engineer's bench a nitrile-gloved pinch lowers the titanium link (amber bushings) into the
index finger's MCP yoke of the prosthetic hand; it seats with a tiny settle and the fingers let go.
s08b (675-720): the finished hand is tested - the fingers close in a soft wave (index -> pinky, thumb last) and
open again; the engineer watches from across the bench, soft, a small smile. Warm late sun through the window,
a warm task lamp, a real workshop in depth behind every frame."""
import sys, os, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, workshop, prosthesis as P, lab, dbgcam
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s08a"])[0]
_, F0, F1 = sb.TL.shot(sid)[:3]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
W = workshop.build(sun_elev=14.0, sun_az=185.0)
M, Z = W["M"], W["Z"]
workshop.bench_dressing(M, Z)

# ------------------------------------------------------------------ prosthetic hand on a foam cradle
HAND_LOC = V((-0.02, 0.0, Z + 0.052))
H = P.Hand("PH", socket=True, tendons=(sid == "s08a"))
H.root.location = HAND_LOC
H.root.rotation_euler = (0, 0, math.radians(-18))
H.set_thumb_frame(d=(-0.55, 0.62, -0.25), z=(-0.3, -0.1, 0.95))
workshop.hand_fixture("Fixture", HAND_LOC, math.radians(-18), M, socket_y=-0.13, socket_r=0.034, socket_z=-0.001)
# parts tray with the sibling links still on their build plate (s07 continuity) + spare pins
tray = workshop.box("PartTray", (0.2, 0.2, Z + 0.006), (0.16, 0.12, 0.012), M["steel"], bev=0.0015)
for i in range(4):
    r_, lk = P.component("v3", "Spare%d" % i, bushings=(i % 2 == 0))
    r_.location = (0.15 + i * 0.03, 0.18, Z + 0.016)
    r_.rotation_euler = (0, math.pi / 2, math.radians(8 * i))
lamp, lamp_light = workshop.task_lamp("TaskLamp", (0.38, 0.25, Z), M, head_target=(0.0, 0.03, Z + 0.02),
                                     head_off=(0.0, -0.05, 0.28) if sid == "s08a" else (0.2, 0.12, 0.42))

idx = H.fingers["index"]
bpy.context.view_layer.update()


def rest_pose(f):
    """Relaxed resting curl of all fingers (a hand at rest is never flat)."""
    for n, fg in H.fingers.items():
        a = 0.18 if n != "thumb" else 0.1
        fg.set(a, a * 0.8, frame=f)


# ------------------------------------------------------------------ s08a: pressing the MCP pin home
if sid == "s08a":
    for f in range(F0 - 10, F1 + 11):
        rest_pose(f)
    workshop.far_side(M)
    bpy.context.view_layer.update()
    # the MCP hinge pin still stands 6 mm proud on the thumb side; a gloved fingertip presses it home
    pin = idx.pins[0]
    Rb = idx.base.matrix_world.to_3x3().normalized()
    PIN_HALF = 0.0098
    T_IN, T_HOME = 638, 652

    def proud(f):
        return 0.006 * (1.0 - sb.smoother((f - T_IN) / (T_HOME - T_IN)))

    for f in range(F0 - 10, F1 + 11):
        pin.location = (-proud(f), 0, 0)
        pin.keyframe_insert("location", frame=f)
    npz = os.path.join(sb.ROOT, "assets", "glove_tap.npz")
    meta = json.loads(str(np.load(npz)["META"]))
    P_pad = V(meta["tap_k"]) + V((0, 0, meta["tap_r"]))
    glove = lab.glove_hand(npz, "Glove")
    sleeve = lab.lab_sleeve("Sleeve", glove)
    hx, hy, hz = Rb @ V((0, 0, 1)), Rb @ V((0, 1, 0)), Rb @ V((-1, 0, 0))     # pad normal (-hz) pushes along +X(base)
    R0 = Matrix((hx, hy, hz)).transposed()
    R = Matrix.Rotation(math.radians(-14), 3, Rb @ V((1, 0, 0))) @ Matrix.Rotation(math.radians(12), 3, hx) @ R0

    def pin_end(f):
        return idx.base.matrix_world @ V((-PIN_HALF - proud(f), 0, 0))

    def glove_mat(f):
        gap = 0.0035 * (1 - sb.smoother((f - (T_IN - 8)) / 8.0))          # approach and touch
        lift = sb.smoother((f - (T_HOME + 5)) / 12.0)                       # let go, move away
        e = pin_end(f) + (Rb @ V((-1, 0, 0))) * (gap + 0.006 * lift) + V((0, 0, 0.012 * lift))
        return Matrix.Translation(e - R @ P_pad) @ R.to_4x4() @ Matrix.Diagonal((1.25, 1.25, 1.25, 1.0))

    for f in range(F0 - 10, F1 + 11):
        glove.matrix_world = glove_mat(f)
        for k in ("location", "rotation_euler", "scale"):
            glove.keyframe_insert(k, frame=f)
    # camera: from beyond the fingertips, low, looking back along the index at the knuckle and the pin end
    tgt = idx.base.matrix_world @ V((-0.004, 0.004, 0.002))
    CO = V([float(x) for x in os.environ.get("SB_S08_CAM", "0.05,0.16,0.09").split(",")])
    cam = sb.camera("Cam", loc=tgt + CO, target=tgt, lens=90, fstop=3.5, clip=(0.005, 50))

    def cpos(t):
        k = sb.smooth(t)
        return tgt + CO * sb.lerp(1.03, 0.96, k)

    sb.cam_bake(cam, F0, F1, cpos, lambda t: tgt, focus="target")
    s0, s1 = F0, F1

# ------------------------------------------------------------------ s08b: the test
else:
    order = ["index", "middle", "ring", "pinky", "thumb"]

    def curl(n, f):
        i = order.index(n)
        t0 = 680 + i * 3.0
        close = sb.smoother((f - t0) / 12.0)
        open_ = sb.smoother((f - (702 + i * 2.0)) / 12.0)
        c = close * (1 - open_)
        th1 = 0.18 + c * (1.25 if n != "thumb" else 0.65)
        return th1, th1 * (0.95 if n != "thumb" else 0.8)

    for f in range(F0 - 10, F1 + 11):
        for n, fg in H.fingers.items():
            a, b = curl(n, f)
            fg.set(a, b, frame=f)
    # the engineer across the bench (soft), leaning in with forearms on the bench
    import folks
    bm, rig, parts = folks.person("engineer")
    folks.sit(rig, knee=85, hip=80)
    folks.pose(rig, {"spine01": (12, 0, 0), "spine03": (8, 0, 0), "neck01": (-10, 0, 0), "head": (-6, 0, 6)})
    folks.place(rig, (0.0, -0.62, 0.0), 180.0)          # MPFB faces -Y: turn to face the bench (+Y)
    bpy.context.view_layer.update()
    pel = folks.bone_world(rig, "pelvis.L")
    rig.location.z += 0.56 - pel.z
    bpy.context.view_layer.update()
    folks.reach(rig, "R", (0.14, -0.22, Z + 0.03), elbow_hint=(0.26, -0.42, Z + 0.02))
    folks.reach(rig, "L", (-0.16, -0.24, Z + 0.03), elbow_hint=(-0.28, -0.42, Z + 0.02))
    folks.pose(rig, {"wrist.R": (0, 0, 10), "wrist.L": (0, 0, -10)})
    # a small smile grows as the fingers close and open (face stays soft in the background)
    for f in range(F0 - 10, F1 + 11):
        folks.expression(rig, smile=0.25 + 0.75 * sb.smoother((f - 692) / 16.0), brows=0.3 * sb.smoother((f - 692) / 16.0), frame=f)
        for b in ("oris04.L", "oris04.R", "oris03.L", "oris03.R"):
            rig.pose.bones[b].keyframe_insert("location", frame=f)
        for b in ("levator05.L", "levator05.R", "orbicularis04.L", "orbicularis04.R"):
            rig.pose.bones[b].keyframe_insert("rotation_euler", frame=f)
    for fb in ("finger2", "finger3", "finger4", "finger5"):
        for j in (1, 2, 3):
            folks.pose(rig, {f"{fb}-{j}.R": (22, 0, 0), f"{fb}-{j}.L": (22, 0, 0)})
    # stool
    workshop.cyl("Stool", (0.0, -0.66, 0.52), 0.18, 0.04, M["black"], verts=48, bev=0.01)
    workshop.cyl("StoolPost", (0.0, -0.66, 0.26), 0.025, 0.5, M["steel"], verts=24)
    workshop.far_side(M)
    # camera from the wall side (+Y) looking back across the hand at the engineer and the shop behind him
    tgt = HAND_LOC + V((0.0, 0.03, 0.02))
    face = folks.bone_world(rig, "head") + V((0, 0, 0.07))
    hand_c = HAND_LOC + V((0.0, 0.07, 0.0))             # centre of the curling fingers
    CAMD = V((0.3, 0.92, 0.2)).normalized() * 0.26
    c0 = hand_c + CAMD
    dh, df = (hand_c - c0).normalized(), (face - c0).normalized()
    sep = math.acos(max(-1, min(1, dh.dot(df))))
    look = c0 + (dh * 0.58 + df * 0.42).normalized()    # hand in the lower part, face in the upper third
    vfov = sep / 0.62
    lens = 12.0 / math.tan(vfov / 2) * (24.0 / 24.0) * (36.0 / 36.0) * (16 / 9) ** 0 * 1.0
    lens = (36.0 * 9 / 16) / 2 / math.tan(vfov / 2)
    print("S08B framing sep deg", math.degrees(sep), "lens", lens)
    cam = sb.camera("Cam", loc=c0, target=look, lens=lens, fstop=2.2, clip=(0.01, 50))

    def cpos(t):
        return c0 + (hand_c - c0) * (0.06 * sb.smooth(t))
    sb.cam_bake(cam, F0, F1, cpos, lambda t: look, focus=lambda t: (cpos(t) - hand_c).length)
    s0, s1 = F0, F1
    cam = None

    if cam is not None:
        sb.cam_bake(cam, F0, F1, cpos, lambda t: tgt, focus="target")
    s0, s1 = F0, F1

dbgcam.apply()
import soul
soul.alive_all(sid, breathe=0.5)
sb.frames(s0, s1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
exec(open(os.environ["SB_PROBE"]).read()) if os.environ.get("SB_PROBE") else sb.render_shot(sid)
