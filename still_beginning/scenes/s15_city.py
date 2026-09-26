"""s15 A FUTURE PEOPLE LIVE IN (1260-1350): golden hour on a tree-lined street of the city the train arrived at.
The camera starts low at a cafe table on the sidewalk (cups, a hand around a coffee, soft) and cranes up through
the street trees as a tram glides in on its grass track; cyclists pass in the bike lane, people walk under the
trees, balconies are full of plants; at the top the street opens toward the timber towers of the city centre in
warm haze. Sun low and behind the city (backlight along the street)."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, street as S, city as C, land as L, dbgcam
import timeline as TL
from mathutils import Vector as V

sid = (sb.argv() or ["s15"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.9")))
rs = random.Random(15)
M = S.materials()
SUN_EL, SUN_AZ = float(os.environ.get("SUNEL", "8")), float(os.environ.get("SUNAZ", "84"))
sb.world_sky(elev=SUN_EL, azim=SUN_AZ, strength=0.16, sun_disc=False, aerosol=2.6)
sb.sun(SUN_EL, SUN_AZ, energy=4.6, color=(1.0, 0.70, 0.42), angle=0.6)

X0, X1 = -60.0, 230.0
# ---- street surfaces
def strip(name, y0, y1, mat, z=0.0, x0=X0, x1=X1):
    o = sb.prim("plane", name, loc=((x0 + x1) / 2, (y0 + y1) / 2, z), scale=((x1 - x0) / 2, abs(y1 - y0) / 2, 1), mat=mat)
    return o


strip("TramBed", -2.0, 2.0, M["grass"], 0.02)
strip("LaneN", 2.0, 4.0, M["pavers"])
strip("LaneS", -4.0, -2.0, M["pavers"])
strip("BikeN", 4.0, 5.2, M["bike"])
strip("BikeS", -5.2, -4.0, M["bike"])
strip("WalkN", 5.35, 11.0, M["flags"], 0.12)
strip("WalkS", -11.0, -5.35, M["flags"], 0.12)
for yy in (5.28, -5.28):
    S.box("Curb", ((X0 + X1) / 2, yy, 0.06), (X1 - X0, 0.16, 0.14), M["curb"], bev=0.02)
for yy in (2.0, -2.0):
    S.box("TrackEdge", ((X0 + X1) / 2, yy, 0.03), (X1 - X0, 0.12, 0.06), M["curb"], bev=0.01)
# rails (two tracks) + grass blades in the track bed
for tc in (-1.0, 1.0):
    for r in (-0.7175, 0.7175):
        S.box("Rail", ((X0 + X1) / 2, tc + r, 0.07), (X1 - X0, 0.07, 0.08), M["rail"], bev=0.004)
gm = L.leaf_mat("TrackGrass", col=(0.07, 0.15, 0.035), var=0.4, autumn=0.15)
verts, faces = [], []
for k in range(90000):
    x = rs.uniform(X0, 80.0); y = rs.uniform(-1.95, 1.95)
    if any(abs(y - (tc + r)) < 0.06 for tc in (-1, 1) for r in (-0.7175, 0.7175)):
        continue
    h = rs.uniform(0.05, 0.16)
    a = rs.uniform(0, 6.28)
    dx, dy = math.cos(a) * 0.006, math.sin(a) * 0.006
    lean = V((rs.uniform(-0.3, 0.3), rs.uniform(-0.3, 0.3), 1.0)).normalized() * h
    b = len(verts)
    verts += [(x - dx, y - dy, 0.02), (x + dx, y + dy, 0.02), (x + lean.x, y + lean.y, 0.02 + lean.z)]
    faces.append((b, b + 1, b + 2))
sb.mesh_obj("TrackGrassBlades", verts, faces, gm, smooth_shade=False)

# ---- buildings, both sides
A = S.Acc()
cols = C.PLASTER
for side in (1, -1):
    x = X0
    i = 0
    while x < X1:
        w = rs.uniform(12, 22)
        fl = rs.randint(4, 6)
        wall = C.plaster_mat("Plaster_%d_%d" % (side, i), cols[(i * 3 + (side > 0)) % len(cols)], streaks=0.3)
        S.building(A, x, w, fl, side, M, rs, wall, awning_i=i + (side > 0))
        x += w + 0.0
        i += 1
A.build("Facades", bevel=0.012)
# ---- far city at the street end, in warm haze
haze_col = (0.95, 0.78, 0.6)


def haze(m, dist=1400.0):
    import wind
    wind.HAZE_COL = haze_col
    wind.aerial(m, dist=dist)


before = set(bpy.data.objects)
C.far_city(center=(820.0, 0.0, 0.0), radius=520.0, n=200, seed=9, haze=haze, tall=9)
# keep the view down the street axis open: clear low blocks inside a widening wedge in front of the street end,
# so the planted timber towers of the centre read at the end of the vista
for o in list(set(bpy.data.objects) - before):
    p_ = o.location
    if o.type == 'MESH' and p_.x > X1 - 5 and abs(p_.y) < 18 + (p_.x - X1) * 0.12 and not o.name.startswith(("TW", "TG", "TB")):
        bpy.data.objects.remove(o)
far = sb.mat("FarGround", (0.25, 0.22, 0.18), rough=0.9)
sb.prim("plane", "FarGround", loc=(0, 0, -0.05), scale=(4000, 4000, 1), mat=far)

# ---- street trees (both sidewalks), lamps, benches, bikes racks
lib = L.tree_library(n=3, seed=150, leaf_count=int(os.environ.get("SB_LEAVES", "60000")), height=10.0, crown_r=3.6)
tree_pts = []
for side in (1, -1):
    for k in range(int((X1 - X0) / 9.0)):
        x = X0 + 4 + k * 9.0 + rs.uniform(-0.4, 0.4)
        if side < 0 and 1.0 < x < 13.0:
            continue                      # the cafe terrace's stretch: no trunk in the opening's line of sight
        tree_pts.append(V((x, side * 7.0, 0.12)))
        if k % 2 == 0:
            S.lamp("Lamp%d_%d" % (side > 0, k), (x + 4.5, side * 5.8), side, M)
        elif k % 4 == 1:
            S.bench("Bench%d_%d" % (side > 0, k), (x + 4.5, side * 8.6, 0.12), math.pi if side > 0 else 0.0, M)
L.place_trees(lib, tree_pts, scale_rng=(0.85, 1.15), seed=5, name="StreetTree")
# tree pits (steel grates) + planted verges
for p in tree_pts:
    S.box("TreeGrate", (p.x, p.y, 0.125), (1.6, 1.6, 0.01), M["metal"], bev=0.002)

# ---- cafe terrace on the south sidewalk (camera side) + umbrella
CAFE_X = 6.0
for k in range(5):
    tx, ty = CAFE_X + k * 2.2, -9.2 + (k % 2) * 0.9
    S.cafe_set("Cafe%d" % k, (tx, ty), M, rs)
um = sb.lathe("Umbrella", [(0.0, 2.4), (1.6, 2.05), (1.62, 2.02)], segs=8, mat=M["umbrella"])
um.location = (CAFE_X + 3.3, -9.3, 0.12)
sb.solidify(um, 0.01)
import rocket
rocket.rod("UmbPole", (CAFE_X + 3.3, -9.3, 0.12), (CAFE_X + 3.3, -9.3, 2.55), 0.02, M["metal"])

# ---- tram (north track, heading -X toward camera)
TR = S.tram("Tram", M)
TRAM_V = 6.5


def tram_x(f):
    return 72.0 - TRAM_V * (f - F0) / 24.0


for f in range(F0 - 10, F1 + 11):
    TR.location = (tram_x(f), 1.0, 0.1)
    TR.rotation_euler = (0, 0, math.pi)             # nose toward -X
    TR.keyframe_insert("location", frame=f)
    TR.keyframe_insert("rotation_euler", frame=f)

# ---- people (MPFB): cafe sitters near camera, walkers, cyclists
if os.environ.get("SB_NOPEOPLE") != "1":
    import folks
    presets = ["researcher", "partner", "grandma", "teen", "wearer", "engineer"]
    # cafe sitters
    for k, (tx, ty) in enumerate([(CAFE_X + 0.4, -9.8), (CAFE_X + 2.6, -8.1), (CAFE_X + 4.8, -9.8)]):
        bm, rig, parts = folks.person(presets[k % len(presets)], name="Sit%d" % k)
        folks.sit(rig, knee=86, hip=82)
        folks.arms_down(rig, drop=26, elbow=40)
        folks.place(rig, (tx, ty, 0.12), 180.0 if ty < -9.0 else 0.0)
        bpy.context.view_layer.update()
        pel = folks.bone_world(rig, "pelvis.L")
        rig.location.z += 0.12 + 0.47 - pel.z
        folks.expression(rig, smile=0.6)
    # walkers on both sidewalks
    walkers = [(-8.0, 9.0, 0.0, 1.35), (30.0, 8.2, 180.0, 1.2), (14.0, -7.6, 180.0, 1.3), (-2.0, -6.4, 0.0, 1.4),
               (45.0, 9.6, 180.0, 1.25), (58.0, -8.8, 180.0, 1.3)]
    walk_src = []
    for k, (x, y, hd, sp) in enumerate(walkers):
        bm, rig, parts = folks.person(presets[(k + 2) % len(presets)], name="Walk%d" % k)
        bpy.context.view_layer.update()
        foot = min(folks.bone_world(rig, "foot.L").z, folks.bone_world(rig, "foot.R").z)
        folks.walk(rig, F0 - 2, F1 + 2, V((x, y, 0.12 + (rig.location.z - foot) + 0.06)), hd, speed=sp, phase=k * 1.3)
        walk_src.append((rig, rig.location.z))
    # the street is alive: a crowd of clones (shared meshes, own walks) along both sidewalks, further out where
    # repetition can't be read; pairs stopped to talk; people at more cafe tables down the street
    rsc = random.Random(15)
    for k in range(26):
        src, z0 = walk_src[k % len(walk_src)]
        r = folks.clone(src, "Crowd%d" % k)
        side = 1 if k % 2 == 0 else -1
        x = rsc.uniform(12.0, 140.0)
        y = side * rsc.uniform(6.6, 9.8)
        hd = 0.0 if rsc.random() < 0.5 else 180.0
        folks.walk(r, F0 - 2, F1 + 2, V((x, y, z0)), hd, speed=rsc.uniform(1.1, 1.45), phase=rsc.uniform(0, 6.28),
                   stride=rsc.uniform(1.3, 1.55))
    for k in range(4):                       # pairs talking, turned to each other, a little weight on one leg
        x = rsc.uniform(20.0, 110.0); y = (1 if k % 2 else -1) * rsc.uniform(7.0, 9.0)
        a, b = folks.clone(walk_src[k % len(walk_src)][0], "TalkA%d" % k), folks.clone(walk_src[(k + 3) % len(walk_src)][0], "TalkB%d" % k)
        for r, dx, rz in ((a, -0.45, 90.0), (b, 0.45, -90.0)):
            folks.arms_down(r, drop=30, elbow=18)
            folks.pose(r, {"upperleg01.L": (0, 0, 0), "upperleg01.R": (-4, 0, 3), "lowerleg01.R": (8, 0, 0),
                           "lowerleg01.L": (0, 0, 0), "foot.L": (0, 0, 0), "foot.R": (0, 0, 0), "spine01": (0, 3, 0)})
            folks.place(r, (x + dx, y, walk_src[0][1]), rz)
        bpy.context.view_layer.update()
        folks.look_at(a, folks.bone_world(b, "head")); folks.look_at(b, folks.bone_world(a, "head"))
        folks.hand(a, "R", "relaxed"); folks.hand(b, "L", "relaxed")

# ---- camera: low at the cafe table -> crane up through the trees to reveal the street and the city
# the crane leaves the cafe table low, swings out over the tram track (clear of the canopies), then rises
PATH = [V((CAFE_X - 1.8, -7.4, 1.1)), V((CAFE_X + 3.5, -4.2, 1.5)), V((CAFE_X + 5.0, -1.4, 3.6)), V((CAFE_X + 6.0, -0.6, 9.0))]
LOOK = [V((40.0, -3.0, 1.4)), V((60.0, -1.0, 2.0)), V((100.0, 0.0, 3.0)), V((150.0, 0.8, 4.5))]


def e(t):
    return sb.smoother(t) * 0.85 + t * 0.15


cam = sb.camera("Cam", loc=PATH[0], target=LOOK[0], lens=28, fstop=4.0, clip=(0.05, 20000))
sb.cam_bake(cam, F0, F1, lambda t: sb.catmull(PATH, e(t)), lambda t: sb.catmull(LOOK, e(t)),
            focus=lambda t: sb.lerp(18.0, 60.0, e(t)))
dbgcam.apply()
import soul
soul.alive_all(sid, calm=0.9)
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
