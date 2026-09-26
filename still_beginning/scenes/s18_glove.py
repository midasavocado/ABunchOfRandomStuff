"""s18 WE TAKE OUR CURIOSITY WITH US (1620-1710). Match cut from the child's ringed-planet drawing: the amber lock
flange of the left wrist disconnect reads as a 1.8:1 ellipse, centred, ~55% frame width, camera pushing in.
The right glove arrives, grips the lock collar, twists it 38 deg; the collar hits its stop and the amber indicator
pin snaps out: LATCH = shot frame 45 = global 1665 (bar 38 downbeat)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, suit, s18grip as GR
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion

sid = (sb.argv() or ["s18"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "-0.25")))
M = suit.materials(dust=0.0, dirt=0.28)

# ---------------------------------------------------------------- world orientation of the left wrist ring
ELEV = math.radians(22)          # camera looks down this much
TILT = math.radians(float(os.environ.get('S18_TILT', '51')))          # angle between view ray and ring axis -> 1/cos = 1.79:1 ellipse
GRIP_SIDE = float(os.environ.get("S18_GRIP", "1"))   # +1: grip at screen-right ansa, -1: screen-left
u = V((0, -math.cos(ELEV), math.sin(ELEV)))          # ring -> camera (camera in front, above)
right = V((1, 0, 0))
A = (Quaternion(right, TILT) @ u).normalized()       # ring axis (toward the left glove): raised forearm, glove up
if A.z < u.z:
    A = (Quaternion(right, -TILT) @ u).normalized()
ansa = u.cross(A).normalized()                       # horizontal, perpendicular to view and axis
Zl = (-ansa * GRIP_SIDE).normalized()                # pin (+Z local) on one ansa, grip (-Z local) on the other
Xl = A.cross(Zl)
Rw = Matrix((Xl, A, Zl)).transposed()
RING = V((0, 0, 1.25))
RW4 = Matrix.Translation(RING) @ Rw.to_4x4()
print("ansa", ansa, "A", A, "Xl.u", Xl.dot(u))

F = suit.forearm("L", "relaxed", M, sleeve_len=0.36, glove_voxel=0.5, name="LFA")
F["root"].matrix_world = RW4
collar = F["collar"]
collar.rotation_mode = 'XYZ'
pin_root = F["pin_root"]
pin0 = pin_root.location.copy()

# ---------------------------------------------------------------- right glove: mesh sequence + right forearm
D = GR.grip_data()
meshes = []
for k in range(GR.NSTEP + 1):
    o = suit.glove(None, "R", mats=M, name="RG%02d" % k, npz=os.path.join(GR.OUT, "grip_%02d.npz" % k))
    meshes.append(o.data)
    if k:
        bpy.data.objects.remove(o)
    else:
        rglove = o
rglove.name = "RGlove"
rfa_root = sb.empty("RFA")
rwd = suit.wrist_disconnect(M, "RWD", segs=160)
rwd["root"].parent = rfa_root
rsl = suit.sleeve(M, 0.34, "RSleeve", seed=5.0)
rsl.parent = rfa_root
rwd["pin_root"].location.z += suit.PIN_POP      # right wrist already locked


_qprev = {}


def keyxf(o, Mw, fr):
    """key location + continuous quaternion from a world matrix (no parent)."""
    loc, q, sc_ = Mw.decompose()
    o.rotation_mode = 'QUATERNION'
    qp = _qprev.get(o.name)
    if qp is not None and qp.dot(q) < 0:
        q = -q
    _qprev[o.name] = q.copy()
    o.location = loc
    o.rotation_quaternion = q
    o.keyframe_insert("location", frame=fr)
    o.keyframe_insert("rotation_quaternion", frame=fr)


def hand_world(f):
    """right-glove world matrix at (float) shot frame f."""
    Rm, org = GR.placement_at(f, D)
    Hm = Matrix([list(Rm[0]) + [org[0]], list(Rm[1]) + [org[1]], list(Rm[2]) + [org[2]], [0, 0, 0, 1]])
    ang = math.radians(GR.collar_angle(f))
    return RW4 @ Matrix.Rotation(ang, 4, 'Y') @ Hm


def mesh_for(f):
    return meshes[GR.step_of(f)]


for fr in range(S0 - 10, S1 + 11):
    f = fr - S0
    Mw = hand_world(f)
    keyxf(rglove, Mw, fr)
    keyxf(rfa_root, Mw @ Matrix.Translation((0, -suit.GLOVE_OFF, 0)), fr)
    collar.rotation_euler = (0, math.radians(GR.collar_angle(f)), 0)
    collar.keyframe_insert("rotation_euler", frame=fr)
    pin_root.location = pin0 + V((0, 0, GR.pin_pop(f, suit.PIN_POP)))
    pin_root.keyframe_insert("location", frame=fr)


def _swap(scene, depsgraph=None):
    f = scene.frame_current + scene.frame_subframe - S0
    me = mesh_for(round(f))
    if rglove.data != me:
        rglove.data = me


bpy.app.handlers.frame_change_pre.append(_swap)

# ---------------------------------------------------------------- the astronaut behind (blurred chest), room
S = suit.build("Astro", visor="gold", mats=M, glove_voxel=1.6, detail=0.7)
for k, o in S["parts"].items():
    if any(t in k for t in ("arm", "glove", "upperarm", "forearm", "hand")):
        o.hide_render = True
S["root"].location = RING + V((0.06, 0.62, -1.42))
S["root"].rotation_euler = (0, 0, math.radians(8))

room = sb.collection("Room")
wall_m = sb.painted("RoomWall", (0.34, 0.31, 0.27), rough=0.6, coat=0.1, grime=0.2)
panel_m = sb.painted("RoomPanel", (0.07, 0.07, 0.075), rough=0.4, coat=0.2)
back = sb.prim("cube", "BackWall", loc=(0, 2.6, 1.4), scale=(3.0, 0.05, 2.0), mat=wall_m)
for i in range(7):
    x = -2.4 + i * 0.8
    sb.prim("cube", "Rib", loc=(x, 2.52, 1.4), scale=(0.035, 0.05, 2.0), mat=panel_m)
floor = sb.prim("plane", "Floor", loc=(0, 0, -0.1), size=12, mat=sb.concrete("RoomFloor", (0.2, 0.19, 0.18)))
ceil = sb.prim("plane", "Ceil", loc=(0, 0, 2.7), size=12, mat=wall_m)
ceil.rotation_euler = (math.pi, 0, 0)
# warm practicals: small lamps and strip lights in the background (become soft bokeh)
warm = sb.emit_mat("Practical", (1.0, 0.62, 0.30), 18.0)
cool = sb.emit_mat("Status", (0.55, 0.75, 1.0), 6.0)
for (x, z, s_) in ((-1.4, 2.05, 0.05), (-0.5, 2.3, 0.04), (0.6, 2.1, 0.05), (1.5, 2.25, 0.04), (-1.0, 1.1, 0.03), (1.2, 0.9, 0.03)):
    sb.prim("sphere", "Lamp", loc=(x, 2.45, z), radius=s_, mat=warm, segments=16, ring_count=8)
for (x, z) in ((-0.2, 1.55), (0.25, 1.35), (0.9, 1.6)):
    sb.prim("cube", "LED", loc=(x, 2.48, z), scale=(0.02, 0.01, 0.006), mat=cool)
strip = sb.prim("cube", "Strip", loc=(0, 2.45, 2.45), scale=(2.4, 0.02, 0.012), mat=warm)

sb.world_color((0.045, 0.036, 0.028), 1.0)
# key: warm softbox upper-left; kicker: reflects in the amber flange face; fill: low bounce
key = sb.light('AREA', "Key", loc=RING + V((-0.9, -0.55, 0.95)), target=RING, energy=65, color=(1.0, 0.84, 0.66), size=0.6)
refl = 2 * u.dot(A) * A - u
kick = sb.light('AREA', "Kick", loc=RING + refl.normalized() * 1.1 + V((0.25, 0, 0)), target=RING, energy=30,
                color=(1.0, 0.72, 0.45), size=0.25, size_y=0.9)
fill = sb.light('AREA', "Fill", loc=RING + V((0.7, -0.6, -0.4)), target=RING, energy=4, color=(0.95, 0.85, 0.75), size=1.2)
strip_l = sb.light('AREA', "CollarStrip", loc=RING + V((0.0, -0.35, 0.9)), target=RING, energy=18, color=(1.0, 0.93, 0.85), size=1.2, size_y=0.08)
rim = sb.light('AREA', "Rim", loc=RING + V((0.4, 0.9, 0.6)), target=RING, energy=25, color=(1.0, 0.75, 0.5), size=0.5)

# ---------------------------------------------------------------- camera: slow push-in continuing s17
LENS = 90.0
FLANGE_W = 2 * 0.0666
D0 = FLANGE_W / 0.55 * LENS / 36.0            # ellipse = 55% of frame width at frame 0
D1 = D0 * 0.80
cam = sb.camera("Cam", loc=RING + u * D0, target=RING, lens=LENS, fstop=11)
cam.data.dof.aperture_blades = 7
up_s = (V((0, 0, 1)) - u * u.z).normalized()


def cpos(t):
    k = 0.55 * t + 0.45 * sb.ease_out(t, 2.0)          # constant push at the cut, settles at the end
    d = D0 + (D1 - D0) * k
    drift = right * (0.004 * t) + up_s * (-0.003 * t)
    return RING + u * d + drift


for fr in range(S0 - 10, S1 + 11):
    t = (fr - S0) / (S1 - S0)
    p = cpos(t)
    AIM = RING + up_s * float(os.environ.get('S18_AIM', '-0.016'))
    fwd = (AIM - p).normalized()
    rgt = fwd.cross(up_s).normalized()
    upv = rgt.cross(fwd)
    keyxf(cam, Matrix.Translation(p) @ Matrix((rgt, upv, -fwd)).transposed().to_4x4(), fr)
    cam.data.dof.focus_distance = (RING - p).dot(fwd) - 0.004
    cam.data.dof.keyframe_insert("focus_distance", frame=fr)

sb.frames(S0, S1)
if os.environ.get("SB_SAVE"):
    sb.save("s18")
sb.render_shot(sid)
