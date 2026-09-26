"""s22 BUILDING BEYOND EARTH (2070-2160): high over the day side, a station's truss spine is being extended. A
long robotic arm (the s07 arm's big sibling: white shells, graphite housings, amber joint rings) carries a new
truss bay toward the end of the spine; an astronaut on a foot restraint guides it in with both hands - the
latches close. Solar arrays and radiators stretch away; the blue planet fills the lower frame. The camera drifts
along the truss, foreground members sliding past, so the scale reads."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, earth, suit, rocket
import launchpad as LP
import timeline as TL
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s22"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render(os.environ.get("SB_ENGINE", "EEVEE"), cycles_samples=96, samples=32, mblur=True, shutter=0.5,
                look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.0")))
rs = random.Random(22)
ALT = 410.0
SUN = sb.sun_dir(28.0, 125.0)
E = earth.build(alt_km=ALT, sun_dir=tuple(SUN), clouds=0.42, lights=0.0, samples=16, nadir=(12.0, 70.0, -30.0), detail=1.0)
E.sun_lamp()
# earthshine: broad cool fill from below
sb.light('AREA', "Earthshine", loc=(0, 0, -60), target=(0, 0, 0), energy=4.0e5, color=(0.55, 0.7, 1.0), size=120.0)

# ---- materials
alu = sb.brushed_metal("TrussAlu", (0.78, 0.78, 0.8), rough=0.28, aniso=0.4, scale=30, scratches=0.3)
white = sb.painted("StationWhite", (0.82, 0.82, 0.8), rough=0.4, coat=0.2, grime=0.08, wear=0.05, scale=2.0)
graph = sb.painted("StationGraphite", (0.04, 0.042, 0.047), rough=0.4, wear=0.3)
amber = sb.painted("StationAmber", (0.75, 0.32, 0.04), rough=0.3, coat=0.4, wear=0.2)
mli = rocket.foil_mat("MLI", (0.82, 0.58, 0.22))
cells = sb.mat("Cells", (0.02, 0.03, 0.07), rough=0.15, coat=1.0, coat_rough=0.02)
nb = sb.NB(cells)
co = nb.coord('Object')
sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
gx = nb.math('LESS_THAN', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[0], 0.16)), 0.06)
gy = nb.math('LESS_THAN', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[1], 0.16)), 0.06)
nb.set('Base Color', nb.mix(nb.math('MAXIMUM', gx, gy), (0.02, 0.03, 0.07, 1), (0.55, 0.55, 0.58, 1)))
radiator = sb.painted("Radiator", (0.86, 0.86, 0.84), rough=0.35, coat=0.3)

# ---- the truss spine along +X (square box truss, 2.4 m bays), station modules at the -X end
B = LP.Beams(22)
BAY, W = 2.4, 1.8
NB_ = 14
X_END = NB_ * BAY
for i in range(NB_ + 1):
    x = i * BAY
    c = [V((x, -W / 2, -W / 2)), V((x, W / 2, -W / 2)), V((x, W / 2, W / 2)), V((x, -W / 2, W / 2))]
    for k in range(4):
        B.tube(c[k], c[(k + 1) % 4], 0.035, 10)
    if i < NB_:
        cn = [p + V((BAY, 0, 0)) for p in c]
        for k in range(4):
            B.tube(c[k], cn[k], 0.05, 12)                                 # longerons
            B.tube(c[k], cn[(k + 1) % 4], 0.03, 8)                        # diagonals
B.build("Spine", alu)
# nodes: amber-banded fittings at every corner
for i in range(NB_ + 1):
    for (y, z) in ((-W / 2, -W / 2), (W / 2, -W / 2), (W / 2, W / 2), (-W / 2, W / 2)):
        n = sb.prim("cube", "Node", loc=(i * BAY, y, z), scale=(0.09, 0.09, 0.09), mat=graph)
        sb.bevel(n, 0.02, 2)
        if i % 3 == 0:
            rocket.box("NodeBand", (i * BAY, y, z), (0.2, 0.2, 0.05), amber, bev=0.005)
# utility trays + cable runs along the spine
for (y, z) in ((0.0, -W / 2 - 0.12), (W / 2 + 0.12, 0.0)):
    rocket.box("Tray", (X_END / 2, y, z), (X_END, 0.3 if z < 0 else 0.06, 0.06 if z < 0 else 0.3), mli, bev=0.01)
# solar array wings + radiators off the spine (long, receding - scale)
for side in (-1, 1):
    for k in range(3):
        x = 3.0 + k * 11.0
        mast = rocket.rod("ArrayMast%d%d" % (side > 0, k), (x, side * W / 2, 0), (x, side * 26.0, 0), 0.08, graph, verts=12)
        for j in range(8):
            pnl = rocket.box("Array", (x, side * (3.5 + j * 2.8), 0), (4.2, 2.6, 0.03), cells, bev=0.004,
                             rot=(math.radians(35 * side), 0, 0))
    rocket.box("Radiator%d" % (side > 0), (X_END * 0.55, side * 3.0, -W - 2.2), (9.0, 0.05, 3.8), radiator, bev=0.01,
               rot=(math.radians(-18 * side), 0, 0))
# pressurised modules at -X (white, MLI patches, handrails)
for k, (L_, r_) in enumerate(((7.0, 2.1), (5.0, 1.6))):
    x0 = -2.0 - k * 7.5
    m_ = sb.prim("cyl", "Module%d" % k, loc=(x0 - L_ / 2, 0, 0), rot=(0, math.pi / 2, 0), vertices=64, radius=r_, depth=L_, mat=white)
    sb.bevel(m_, 0.15, 3)
    for j in range(6):
        rocket.box("MLIPatch", (x0 - 0.8 - j * 1.1, 0, r_ + 0.01), (0.9, 1.4, 0.02), mli, bev=0.01)
    for j in range(3):
        rocket.rod("Handrail", (x0 - 1 - j * 2.0, -0.6, r_ + 0.1), (x0 - 2 - j * 2.0, -0.6, r_ + 0.1), 0.02, amber, verts=8)
# ---- equipment along the spine: ORU boxes in white MLI / gold foil, handrails, lights, cable bundles
rs2 = random.Random(7)
mli_white = sb.fabric("MLIWhite", (0.86, 0.86, 0.84), weave=180, sheen=0.3, fuzz=0.2)
for i in range(1, NB_ - 1):
    x = i * BAY + BAY / 2
    for face in range(4):
        if rs2.random() < 0.55:
            continue
        sz = V((rs2.uniform(0.6, 1.4), rs2.uniform(0.5, 0.9), rs2.uniform(0.3, 0.6)))
        if face == 0:
            c = V((x, 0, W / 2 + sz.z / 2 + 0.05)); dims = (sz.x, sz.y, sz.z)
        elif face == 1:
            c = V((x, 0, -W / 2 - sz.z / 2 - 0.05)); dims = (sz.x, sz.y, sz.z)
        elif face == 2:
            c = V((x, W / 2 + sz.z / 2 + 0.05, 0)); dims = (sz.x, sz.z, sz.y)
        else:
            c = V((x, -W / 2 - sz.z / 2 - 0.05, 0)); dims = (sz.x, sz.z, sz.y)
        rocket.box("ORU", c, dims, rs2.choice([mli_white, mli_white, mli, graph]), bev=0.03)
    rocket.rod("Handrail", (x - 0.8, -W / 2 - 0.12, W / 2 + 0.12), (x + 0.8, -W / 2 - 0.12, W / 2 + 0.12), 0.018, amber, verts=8)
    if i % 3 == 0:
        lt = sb.prim("cyl", "WorkLight", loc=(x, -W / 2 - 0.15, W / 2 - 0.1), vertices=16, radius=0.07, depth=0.12, mat=graph)
for k in range(3):
    sb.tube_along("CableRun%d" % k, [(0.2, W / 2 + 0.07, -0.3 + k * 0.08), (X_END - 0.2, W / 2 + 0.07, -0.3 + k * 0.08)], radius=0.02,
                  mat=sb.mat("CableBlk", (0.03, 0.03, 0.03), rough=0.5))
# a radiator wing close to the working end (in view, behind the astronaut)
for j in range(4):
    rocket.box("RadPanel", (X_END - 7.0, W / 2 + 1.6 + j * 2.1, -1.0), (3.0, 2.0, 0.04), radiator, bev=0.008, rot=(math.radians(-25), 0, 0))
# ---- the new bay arriving (carried by the arm), final position = next bay at X_END..X_END+BAY
Bn = LP.Beams(23)
cn = [V((0, -W / 2, -W / 2)), V((0, W / 2, -W / 2)), V((0, W / 2, W / 2)), V((0, -W / 2, W / 2))]
for k in range(4):
    Bn.tube(cn[k] + V((BAY, 0, 0)), cn[(k + 1) % 4] + V((BAY, 0, 0)), 0.035, 10)
    Bn.tube(cn[k], cn[k] + V((BAY, 0, 0)), 0.05, 12)
    Bn.tube(cn[k], cn[(k + 1) % 4] + V((BAY, 0, 0)), 0.03, 8)
newbay = Bn.build("NewBay", alu)
newbay_root = sb.empty("NewBayRoot")
newbay.parent = newbay_root
DOCK_F = F1 - 24


def bay_mat(f):
    k = sb.smoother((f - (F0 - 40)) / (DOCK_F - (F0 - 40)))
    off = V((0.9, 1.1, 1.6)) * (1 - k)
    rot = Euler((math.radians(6) * (1 - k), math.radians(-4) * (1 - k), math.radians(9) * (1 - k)), 'XYZ').to_matrix().to_4x4()
    return Matrix.Translation(V((X_END, 0, 0)) + off) @ rot


for f in range(F0 - 2, F1 + 3):
    newbay_root.matrix_world = bay_mat(f)
    newbay_root.keyframe_insert("location", frame=f)
    newbay_root.keyframe_insert("rotation_euler", frame=f)
# the big arm: base on the spine at x = X_END - 3 BAY, grapple at the new bay's far face (scaled robot.Robot)
import robot
arm_base = sb.empty("ArmBase", loc=(X_END - 3 * BAY, 0, W / 2 + 0.1))
R_ = robot.Robot("SArm", loc=(0, 0, 0), yaw=0.0)
R_.root.parent = arm_base
arm_base.scale = (7.0, 7.0, 7.0)                    # the 1.3 m factory arm -> ~9 m space arm (same design family)
q = [0.0, -0.4, 1.2, 0.0, 0.8, 0.0]
for f in range(F0 - 2, F1 + 3):
    bpy.context.scene.frame_set(f)
    bm4 = bay_mat(f)
    tgt = robot.tcp_matrix(bm4 @ V((BAY * 0.5, 0, W / 2 + 0.25)), approach=tuple(bm4.to_3x3() @ V((0, 0, -1))),
                           close_dir=tuple(bm4.to_3x3() @ V((1, 0, 0))))
    q, _e = R_.ik(tgt, q, iters=30)
    R_.set(q, frame=f)
# ---- the astronaut on a foot restraint at the spine end, hands on the incoming bay
S = suit.build("Astro", visor="gold", gloves=("relaxed", "relaxed"), dust=0.0, dirt=0.2)
astro = S["root"]
# work platform cantilevered off the spine's side face near the end; the astronaut faces the arriving bay (+X)
PLAT = V((X_END - 1.2, -W / 2 - 1.1, -0.3))
rocket.box("Platform", PLAT, (1.2, 0.9, 0.06), graph, bev=0.01)
for k in (-1, 1):
    rocket.rod("PlatStrut", PLAT + V((0.5 * k, 0.4, 0)), V((X_END - 1.2 + 0.5 * k, -W / 2, -W / 2 + 0.2)), 0.03, alu, verts=8)
rocket.box("FootRestraint", PLAT + V((0.1, 0.0, 0.05)), (0.5, 0.35, 0.05), amber, bev=0.01)
astro.location = PLAT + V((0.1, 0.0, 0.06))
astro.rotation_euler = (0, 0, math.radians(-70))
for f in range(F0 - 2, F1 + 3):
    k = sb.smoother((f - F0) / (F1 - F0))
    suit.pose(S, frame=f, torso=(12 + 6 * math.sin(f * 0.05), 0, -10), l_shoulder=(70 + 10 * k, 20, 0), r_shoulder=(60 + 15 * k, 25, 0),
              l_elbow=40 - 10 * k, r_elbow=45 - 12 * k, l_knee=12, r_knee=15, l_hip=(8, 5, 0), r_hip=(6, 5, 0))
# ---- camera: drifting along the spine, 3/4 above; foreground truss members sweep past
P0, P1 = V((X_END - 9.0, -7.5, 4.2)), V((X_END - 5.5, -6.8, 3.2))
T0, T1 = V((X_END + 0.5, 0.0, 0.6)), V((X_END + 1.0, 0.0, 0.8))
cam = sb.camera("Cam", loc=P0, target=T0, lens=24, fstop=5.6, clip=(0.05, 20000.0))
sb.cam_bake(cam, F0, F1, lambda t: P0.lerp(P1, sb.smoother(t)), lambda t: T0.lerp(T1, sb.smoother(t)),
            focus=lambda t: (P0.lerp(P1, sb.smoother(t)) - T0.lerp(T1, sb.smoother(t))).length)
E.track(cam, F0 - 2, F1 + 2)
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
