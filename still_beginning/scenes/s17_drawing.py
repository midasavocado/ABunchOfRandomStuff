"""s17 HERO ONE (1440-1620, the breakdown - intimate piano): night, the child's room. At her desk under the window
the child draws in her sketchbook by the warm desk lamp: a planet, an orbit, a rocket climbing on a curved arc -
the line grows under her pencil. The city from s02 glitters through the window; her telescope stands beside it.
One continuous slow push: from the room (child silhouetted against the window) over her shoulder into the page,
ending tight on her drawing hand and the amber rib-knit cuff (match cut -> s18's amber wrist ring)."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, props, handmesh, folks, mhchild, dbgcam
import timeline as TL
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s17"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
rs = random.Random(17)
import home, rocket
M = home.materials()

# ------------------------------------------------------------------ room (desk under the window, window faces +Y)
DESK_Z = 0.72
sb.prim("plane", "Floor", scale=(4, 4, 1), mat=M["floor"])
sb.prim("cube", "Ceil", loc=(0, 0, 2.6), scale=(4, 4, 0.05), mat=M["wall"])
wallm = sb.concrete("BedroomWall", (0.46, 0.50, 0.58), scale=1.2, rough=0.9, stains=0.05)     # soft blue-grey paint
for (loc, s) in (((-2.0, 0, 1.3), (0.05, 3, 1.3)), ((2.0, 0, 1.3), (0.05, 3, 1.3)), ((0, -2.5, 1.3), (3, 0.05, 1.3))):
    sb.prim("cube", "Wall", loc=loc, scale=s, mat=wallm)
# window wall (y = 0.55): wall pieces around a wide window, deep sill
WY = 0.55
for (c, s) in (((-1.35, WY, 1.3), (1.3, 0.2, 2.6)), ((1.35, WY, 1.3), (1.3, 0.2, 2.6)), ((0, WY, 0.4), (1.4, 0.2, 0.8)),
               ((0, WY, 2.3), (1.4, 0.2, 0.6))):
    rocket.box("WinWall", c, s, wallm, bev=0.0)
rocket.box("Sill", (0, WY - 0.05, 0.81), (1.5, 0.35, 0.03), M["oak"], bev=0.004)
for x in (-0.7, 0.0, 0.7):
    rocket.box("WinMull", (x, WY, 1.4), (0.05, 0.08, 1.2), M["window_frame"], bev=0.004)
for z in (0.82, 2.0):
    rocket.box("WinRail", (0, WY, z), (1.45, 0.08, 0.05), M["window_frame"], bev=0.004)
pane = sb.prim("plane", "Pane", loc=(0, WY + 0.01, 1.41), size=1.0, mat=M["pane"])
pane.scale = (0.7, 0.6, 1); pane.rotation_euler = (math.pi / 2, 0, 0)
# outside: the same twilight-into-night city as the rooftop (s02), seen from a few floors up
import skyline2
skyline2.build(center=(0, 0), rmin=24, rmax=900, count=480, ground_z=-12.0, view_dir=math.pi / 2, view_half=math.radians(70))
sb.world_bluehour(glow_az=15.0, glow=1.2, glow_col=(0.9, 0.35, 0.12), zenith=(0.004, 0.009, 0.03), horizon=(0.03, 0.04, 0.08),
                  strength=1.0)

# desk + chair + lamp + dressing
rocket.box("Desk", (0, 0.2, DESK_Z - 0.02), (1.3, 0.62, 0.04), M["oak"], bev=0.006)
for sx in (-0.6, 0.6):
    rocket.box("DeskLeg", (sx, 0.2, (DESK_Z - 0.04) / 2), (0.04, 0.55, DESK_Z - 0.04), M["oak_dark"], bev=0.004)
home.chair("Chair", (0.02, -0.42, 0), 0.0, M)
# desk lamp: articulated, warm, on the left
import workshop
WM = workshop.materials()
lamp, L = workshop.task_lamp("DeskLamp", (-0.48, 0.34, DESK_Z), WM, head_target=(-0.02, 0.1, DESK_Z), head_off=(-0.18, 0.08, 0.36))
L.data.energy = 3.5
L.data.color = (1.0, 0.7, 0.42)
WM["lamp"].node_tree.nodes["Emission"].inputs[1].default_value = 6.0
# books, a globe, a paper model rocket, pencils in a cup, crayons, a small plant; drawings pinned on the wall
for k in range(6):
    rocket.box("DeskBook", (0.42 + k * 0.035, 0.42, DESK_Z + 0.12), (0.03, 0.2, rs.uniform(0.2, 0.26)), rs.choice(M["book"]), bev=0.002)
gl = sb.prim("sphere", "Globe", loc=(0.52, 0.3, DESK_Z + 0.2), radius=0.09, segments=48, ring_count=24, mat=home.child_drawing_mat(1))
rocket.rod("GlobeStand", (0.52, 0.3, DESK_Z), (0.52, 0.3, DESK_Z + 0.11), 0.006, M["brass"])
cup = sb.lathe("PencilCup", [(0, 0), (0.035, 0), (0.037, 0.1), (0.034, 0.1), (0.032, 0.005), (0, 0.005)], segs=32, mat=M["terra"])
cup.location = (-0.3, 0.38, DESK_Z)
for k in range(7):
    a = k * 0.9
    rocket.rod("CupPencil", (-0.3 + 0.015 * math.cos(a), 0.38 + 0.015 * math.sin(a), DESK_Z + 0.01),
               (-0.3 + 0.04 * math.cos(a), 0.38 + 0.04 * math.sin(a), DESK_Z + 0.17), 0.0035,
               sb.mat("PencilPaint%d" % k, rs.choice([(0.7, 0.3, 0.05), (0.1, 0.2, 0.4), (0.15, 0.4, 0.15), (0.8, 0.7, 0.2)]), rough=0.4), verts=6)
for k in range(5):
    fr = rocket.box("Pinned%d" % k, (-1.0 + k * 0.45, WY - 0.105, 1.55 + (k % 2) * 0.12), (0.26, 0.004, 0.19), home.child_drawing_mat(k), bev=0.0)
    fr.location.x = [-1.6, -1.2, 1.05, 1.45, 1.8][k]
# model rocket on the sill (white + graphite + amber band, like the real one), the telescope by the window
mr = sb.lathe("ModelRocket", [(0.0, 0.0), (0.022, 0.0), (0.022, 0.22), (0.018, 0.27), (0.0, 0.31)], segs=32,
              mat=sb.painted("ModelWhite", (0.8, 0.8, 0.78), rough=0.4))
mr.location = (0.55, WY - 0.06, 0.825)
rocket.box("ModelBand", (0.55, WY - 0.06, 0.825 + 0.16), (0.046, 0.046, 0.012), sb.painted("ModelAmber", (0.7, 0.28, 0.03), rough=0.35), bev=0.0)
T = props.telescope(alt_deg=28.0, az_deg=0.0, height=1.02)
T["root"].location = (1.1, 0.1, 0) if isinstance(T, dict) and "root" in T else (1.1, 0.1, 0)
# string lights along the window head (small warm points)
for k in range(14):
    x = -0.75 + k * 1.5 / 13
    z = 2.22 - 0.05 * math.sin(math.pi * k / 13)
    b = sb.prim("sphere", "Fairy", loc=(x, WY - 0.12, z), radius=0.008, segments=8, ring_count=4, mat=sb.emit_mat("FairyE", (1.0, 0.7, 0.4), 30.0))
sb.light('AREA', "WindowCool", loc=(0, WY + 0.2, 1.4), target=(0, -0.5, 0.9), energy=6.0, color=(0.55, 0.65, 1.0), size=1.4)
sb.light('AREA', "RoomBounce", loc=(0, -1.5, 2.4), target=(0, 0, 0.5), energy=4.0, color=(1.0, 0.8, 0.6), size=1.5)

# ------------------------------------------------------------------ sketchbook + the drawing (grows under the pencil)
BOOK_C = V((0.0, 0.14, DESK_Z + 0.004))
paper_m = sb.mat("Sketch", (0.86, 0.84, 0.78), rough=0.85, sheen=0.2)
nb = sb.NB(paper_m)
co = nb.coord('Object')
nb.set('Normal', nb.bump(nb.noise(co, scale=900, detail=4).outputs['Fac'], strength=0.08, distance=0.0002))
rocket.box("BookCover", BOOK_C + V((0, 0, -0.002)), (0.44, 0.3, 0.004), sb.mat("Cover", (0.12, 0.14, 0.2), rough=0.6), bev=0.001)
for s_ in (-1, 1):
    pg = rocket.box("Page", BOOK_C + V((s_ * 0.108, 0, 0.002 + 0.0015 * (s_ > 0))), (0.21, 0.29, 0.003), paper_m, bev=0.0005)
    pg.rotation_euler = (0, math.radians(-2.0 * s_), 0)
rocket.rod("Spine", BOOK_C + V((0, -0.145, 0.004)), BOOK_C + V((0, 0.145, 0.004)), 0.004, sb.mat("Wire", (0.2, 0.2, 0.2), metal=0.9, rough=0.3), verts=12)
graphite = sb.mat("Graphite", (0.08, 0.08, 0.09), metal=0.3, rough=0.35)
PZ = BOOK_C.z + 0.0037               # page surface of the right page


def stroke(name, pts2d, width=0.00045, grow=None):
    """Pencil line on the right page (page-local 2D metres; origin = right page centre). grow=(f_start, f_end)
    animates its bevel_factor_end so the line appears under the pencil."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = width
    cu.bevel_resolution = 1
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts2d) - 1)
    for i, (x, y) in enumerate(pts2d):
        jx, jy = rs.uniform(-0.0002, 0.0002), rs.uniform(-0.0002, 0.0002)
        sp.points[i].co = (BOOK_C.x + 0.108 + x + jx, BOOK_C.y + y + jy, PZ, 1)
        sp.points[i].radius = 0.7 + 0.5 * rs.random()
    o = bpy.data.objects.new(name, cu)
    sb.link_obj(o)
    cu.materials.append(graphite)
    o.scale = (1, 1, 0.25)
    o.location = (0, 0, PZ * 0.75)
    if grow:
        for f in range(F0 - 2, F1 + 2):
            cu.bevel_factor_end = max(0.0, min(1.0, (f - grow[0]) / (grow[1] - grow[0])))
            cu.keyframe_insert("bevel_factor_end", frame=f)
    return o


def circle(c, r, n=48, a0=0.0, a1=2 * math.pi, squash=1.0):
    return [(c[0] + r * math.cos(a0 + (a1 - a0) * i / n), c[1] + r * squash * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


# already drawn: a planet with a ring shadow line, stars, the orbit ellipse, little rocket at the pad
stroke("Planet", circle((0.03, 0.05), 0.035, 60))
stroke("PlanetBand", circle((0.03, 0.05), 0.052, 60, 0.3, 2.9, squash=0.28))
for k in range(9):
    sx, sy = rs.uniform(-0.09, 0.09), rs.uniform(-0.12, 0.13)
    if (V((sx, sy, 0)) - V((0.03, 0.05, 0))).length < 0.07:
        continue
    stroke("StarA%d" % k, [(sx - 0.004, sy), (sx + 0.004, sy)], 0.0003)
    stroke("StarB%d" % k, [(sx, sy - 0.004), (sx, sy + 0.004)], 0.0003)
stroke("Orbit", circle((0.03, 0.05), 0.075, 80, squash=0.45))
stroke("Ground", [(-0.09, -0.11), (0.09, -0.115)])
# the arc being drawn now: from the pad up and around toward the planet (the path of the pencil tip)
ARC = [(-0.05 + 0.1 * t + 0.02 * math.sin(t * 3.1), -0.1 + 0.2 * t ** 0.8 - 0.02 * t * t) for t in [i / 60 for i in range(61)]]
DRAW0, DRAW1 = 1470, 1590
arc_obj = stroke("Arc", ARC, 0.0005, grow=(DRAW0, DRAW1))


def tip_at(f):
    u = max(0.0, min(1.0, (f - DRAW0) / (DRAW1 - DRAW0)))
    i = u * (len(ARC) - 1)
    a, b = ARC[int(i)], ARC[min(len(ARC) - 1, int(i) + 1)]
    fr = i - int(i)
    x = a[0] + (b[0] - a[0]) * fr
    y = a[1] + (b[1] - a[1]) * fr
    lift = 0.0015 * (1 - sb.smoother((f - (DRAW0 - 8)) / 8.0)) + 0.002 * sb.smoother((f - DRAW1) / 8.0)
    return V((BOOK_C.x + 0.108 + x, BOOK_C.y + y, PZ + lift))


# ------------------------------------------------------------------ the child
bm, rig, parts = mhchild.build()
mhchild.fix_eyes(parts)
mhchild.recolor_top(parts)
cuffs = mhchild.add_cuffs(rig, parts)
folks.sit(rig, knee=85, hip=80)
folks.place(rig, (0.04, -0.34, 0.0), 180.0)                 # facing the desk/window (+Y)
bpy.context.view_layer.update()
pel = folks.bone_world(rig, "pelvis.L")
rig.location.z += 0.44 + 0.07 - pel.z
folks.pose(rig, {"spine01": (16, 0, 0), "spine02": (8, 0, 0), "spine03": (6, 0, 0)})
bpy.context.view_layer.update()
folks.amputate(parts, "R", bones=("wrist", "metacarpal", "finger"))
# the SDF drawing hand (tripod grip on the pencil)
hand, hparts = handmesh.load(os.path.join(sb.ROOT, "assets", "hand_pencil.npz"), "DrawHand")
PEN_W = V((-0.026, 0.051, 0.015)); PEN_D = V((0.1203, 0.7020, -0.7020)).normalized(); PEN_LEN = 0.155
TIP_S = 0.083                                             # web point -> graphite tip along the pencil
pencil_m = sb.painted("PencilBody", (0.75, 0.32, 0.04), rough=0.35, coat=0.3)
wood_m = sb.mat("PencilWood", (0.78, 0.6, 0.4), rough=0.6)
pen = sb.empty("Pencil", parent=hand)
pb_ = rocket.rod("PencilShaft", tuple(PEN_W - PEN_D * (PEN_LEN - TIP_S - 0.018)), tuple(PEN_W + PEN_D * (TIP_S - 0.018)), 0.0036,
                 pencil_m, parent=pen, verts=6)
cone = rocket.rod("PencilCone", tuple(PEN_W + PEN_D * (TIP_S - 0.018)), tuple(PEN_W + PEN_D * TIP_S), 0.0022, wood_m, parent=pen, verts=12)
tipc = rocket.rod("PencilTip", tuple(PEN_W + PEN_D * (TIP_S - 0.004)), tuple(PEN_W + PEN_D * TIP_S), 0.0008, graphite, parent=pen, verts=8)
TIP_H = PEN_W + PEN_D * TIP_S                              # tip in hand frame
# writing orientation: palm faces down-left (ulnar edge on the paper), fingers point forward-left
Rw = (Euler((math.radians(-8), math.radians(38), math.radians(28)), 'XYZ').to_matrix())


def hand_mat(f):
    t = tip_at(f)
    wob = Euler((0.0, math.radians(1.5 * math.sin(f * 0.21)), math.radians(2.0 * math.sin(f * 0.13))), 'XYZ').to_matrix()
    R = wob @ Rw
    return Matrix.Translation(t - R @ TIP_H) @ R.to_4x4()


hand.parent = None
for f in range(F0 - 2, F1 + 2):
    hand.matrix_world = hand_mat(f)
    hand.keyframe_insert("location", frame=f)
    hand.keyframe_insert("rotation_euler", frame=f)
    # forearm follows: wrist bone onto the hand's wrist centre
    wr = hand.matrix_world.translation
    folks.reach(rig, "R", wr + V((0.0, 0.0, 0.0)), elbow_hint=wr + V((0.12, -0.25, -0.06)), iters=15)
    for b in ("upperarm01.R", "lowerarm01.R"):
        rig.pose.bones[b].keyframe_insert("rotation_euler", frame=f)
    if f in (F0 - 2, (F0 + F1) // 2, F1 + 1):
        folks.look_at(rig, tip_at(f))
        for b in ("neck02", "head"):
            rig.pose.bones[b].keyframe_insert("rotation_euler", frame=f)
# left hand steadies the sketchbook
folks.reach(rig, "L", BOOK_C + V((-0.16, -0.06, 0.03)), elbow_hint=BOOK_C + V((-0.32, -0.3, -0.04)))

# ------------------------------------------------------------------ camera: one slow push from the room to the cuff
bpy.context.view_layer.update()
cuff_pt = folks.bone_world(rig, "lowerarm02.R", tail=True)
P = [V((1.05, -1.55, 1.25)), V((0.62, -0.9, 1.12)), V((0.25, -0.38, 0.98)), cuff_pt + V((0.1, -0.2, 0.08))]
Tg = [V((0.0, 0.25, 1.0)), V((0.05, 0.15, 0.85)), tip_at(1560) + V((0, 0, 0.01)), cuff_pt + V((0.0, 0.03, 0.0))]


def cam_pos(t):
    return sb.catmull(P, sb.smoother(t) * 0.8 + t * 0.2)


def cam_tgt(t):
    return sb.catmull(Tg, sb.smoother(t) * 0.8 + t * 0.2)


cam = sb.camera("Cam", loc=P[0], target=Tg[0], lens=35, fstop=2.2, clip=(0.005, 3000))
sb.cam_bake(cam, F0, F1, cam_pos, cam_tgt, lens=lambda t: sb.lerp(35.0, 60.0, sb.smoother(t)),
            focus=lambda t: (cam_pos(t) - cam_tgt(t)).length)
dbgcam.apply()
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
