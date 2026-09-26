"""s14 MOVING FORWARD: a fast electric train crosses coherent farmland toward the bright city of s15.
Start low beside the line (fence posts, grass, a catenary mast whipping past) as the train overtakes; crane up and
look down the line: the train runs on toward the city on the horizon in low golden sun. Track geometry, wheels,
pantograph/contact wire and travel direction (+X) are consistent throughout."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb
import numpy as np
import energy as E
import land as L
import wind as W
import rail as R
import city as C
import timeline as TL
from mathutils import Vector as V

sid = (sb.argv() or ["s14"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=float(os.environ.get("EXPO", "0.0")))
rnd = random.Random(14)
SUN_EL, SUN_AZ = float(os.environ.get("SUNEL", "11")), float(os.environ.get("SUNAZ", "262"))
HAZE = float(os.environ.get("HAZED", "5200"))
CITY_X = 4800.0
VT = 83.0                  # train speed m/s (~300 km/h)

sb.world_sky(elev=SUN_EL, azim=SUN_AZ, strength=float(os.environ.get("SKY", "0.13")), sun_disc=False, aerosol=1.8)
sb.sun(SUN_EL, SUN_AZ, energy=float(os.environ.get("SUNE", "4.6")), color=(1.0, 0.80, 0.56), angle=0.55)
W.HAZE_COL = (0.80, 0.74, 0.66)
hz = lambda m: W.aerial(m, dist=HAZE)


def track_z(x):
    return 1.2 + 2.0 * math.sin(x / 1400.0)


def ground(X, Y):
    h = L.hills(X, Y) * 0.6
    tz = 1.2 + 2.0 * np.sin(X / 1400.0) - 0.6
    w = np.clip((np.abs(Y) - 9.0) / 60.0, 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    g = tz * (1 - w) + h * w
    c = np.clip((X - (CITY_X - 1500.0)) / 900.0, 0.0, 1.0)      # flatten toward the city
    return g * (1 - c) + 1.5 * c


# ---- land
field = L.field_mat("Field", rows_dir=(1.0, 0.25), pitch=0.75, site=False)
field.node_tree.nodes  # (site mask in field_mat is harmless here)
hz(field)
ter = L.terrain("Terrain", ground, -1500, 12000, -4500, 4500, 540, 360, field)
farm = sb.mat("FarLand", (0.12, 0.13, 0.06), rough=0.9); hz(farm)
sb.prim("plane", "FarGround", loc=(0, 0, -20), scale=(300000, 300000, 1), mat=farm)
# near meadow strip along the line (grass blades) where the camera starts
grass_m = L.leaf_mat("Grass", col=(0.07, 0.12, 0.035), var=0.45, autumn=0.25)
verts, faces, uvs, lv = [], [], [], []
for k in range(60000):
    x = rnd.uniform(-60, 240); y = rnd.uniform(-16.5, -8.0)
    z = float(ground(np.array(x), np.array(y)))
    hgt = rnd.uniform(0.25, 0.75) * (1.3 if rnd.random() < 0.1 else 1.0)
    a = rnd.uniform(0, 2 * math.pi); w_ = rnd.uniform(0.012, 0.025)
    lean = V((rnd.uniform(-0.25, 0.25), rnd.uniform(-0.25, 0.25), 1.0)).normalized()
    side = V((math.cos(a), math.sin(a), 0)) * w_
    base = V((x, y, z - 0.03))
    tip = base + lean * hgt
    b = len(verts)
    verts += [tuple(base - side), tuple(base + side), tuple(tip + side * 0.2), tuple(tip - side * 0.2)]
    faces.append((b, b + 1, b + 2, b + 3)); uvs += [(0.3, 0), (0.7, 0), (0.55, 1), (0.45, 1)]; lv.append(rnd.random())
go = E.mesh_np("GrassBlades", verts, np.array(faces), grass_m, np.array(uvs), smooth=False)
at = go.data.attributes.new("lv", 'FLOAT', 'FACE'); at.data.foreach_set("value", np.array(lv, np.float32))
# ---- track + catenary + fence
trk = R.track("Track", -1500, CITY_X - 300, track_z, sleeper_range=(-80, 900))
cat = R.catenary("Cat", -1500, CITY_X - 300, track_z, spacing=60.0, detail_range=(-200, 1500))
for o in [trk, cat] + list(trk.children) + list(cat.children):
    for s in getattr(getattr(o, 'data', None), 'materials', []) or []:
        pass
galv = E.painted_steel("FenceGalv", (0.5, 0.51, 0.52), rough=0.45, wear=0.3)
fp_src = E.box("FPSrc", (0.06, 0.06, 1.9), loc=(0, 0, 0), mat=galv)
fp_src.hide_render = True
for x in np.arange(-60, 400, 3.0):
    z = float(ground(np.array(x), np.array(-9.6)))
    o = fp_src.copy(); sb.link_obj(o); o.hide_render = False
    o.name = f"FP{x:.0f}"; o.location = (x, -9.6, z + 0.9)
for zz in (0.4, 1.1, 1.75):
    pts = [(x, -9.6, float(ground(np.array(x), np.array(-9.6))) + zz) for x in np.arange(-60, 401, 3.0)]
    E.sweep(f"FW{zz}", pts, radius=0.003, segs=4, mat=galv, sub=1, resample=False, caps=False)
# ---- hedgerows / trees / distant solar rows
bark = L.bark_mat("Bark"); leafm = L.leaf_mat("LeafG", col=(0.042, 0.10, 0.022))
hz(bark); hz(leafm)
lib_near = L.tree_library(3, seed=41, height=12.0, crown_r=4.8, bark=bark, leaf=leafm)
lib_far = L.tree_library(3, seed=71, height=12.0, crown_r=5.0, bark=bark, leaf=leafm, lod=True, leaf_count=4500, leaf_size=0.4)
near, far = [], []
for (ya, yb, xa, xb) in ((-60, -95, -500, 3000), (70, 120, -500, 5000), (-400, -380, 0, 6000), (600, 560, -400, 6000)):
    n = int((xb - xa) / 10)
    for k in range(n):
        if rnd.random() < 0.2:
            continue
        x = xa + (xb - xa) * k / n + rnd.uniform(-3, 3); y = ya + (yb - ya) * k / n + rnd.uniform(-3, 3)
        z = float(ground(np.array(x), np.array(y))) - 0.2
        (near if (abs(y) < 150 and -100 < x < 900) else far).append(V((x, y, z)))
for k in range(60):
    cx_, cy_ = rnd.uniform(-1000, 7000), rnd.uniform(-3500, 3500)
    if abs(cy_) < 150:
        continue
    for m_ in range(rnd.randint(5, 18)):
        x, y = cx_ + rnd.gauss(0, 22), cy_ + rnd.gauss(0, 22)
        far.append(V((x, y, float(ground(np.array(x), np.array(y))) - 0.2)))
far = [p for p in far if not (p.x > CITY_X - 700 and abs(p.y) < 2000)]      # no orchards through the city
L.place_trees(lib_near, near, seed=2, name="Tn")
L.place_trees(lib_far, far, seed=3, name="Tf")
# ---- the city on the horizon (same palette/architecture as s15)
city = C.far_city(center=(CITY_X + 700, 0, 1.5), radius=1100, n=320, seed=15, haze=lambda m: W.aerial(m, dist=HAZE * 2.4), tall=16)
# ---- the train
tr, wheels, panto = R.train("Train", ncars=8)
for o in [tr] + list(tr.children_recursive):
    if getattr(o, 'data', None) is not None and hasattr(o.data, 'materials'):
        for m in o.data.materials:
            if m and not m.get("hz"):
                hz(m); m["hz"] = 1
X_N0 = -18.0                               # nose x at F0


def nose_x(f):
    return X_N0 + VT * (f - F0) / 24.0


for f in range(F0 - 1, F1 + 2):
    x = nose_x(f)
    tr.location = (x, R.TRACK_Y[0], track_z(x) + R.RAIL_TOP)
    tr.keyframe_insert("location", frame=f)
    for ws in wheels:
        ws.rotation_euler = (0, VT * (f - F0) / 24.0 / 0.46, 0)
        ws.keyframe_insert("rotation_euler", frame=f)
# ---- camera
cam = E.dof_cam("Cam", 32, 5.6, clip=(0.05, 400000))
sc.eevee.bokeh_max_size = 120


def cam_x(t):
    # 40 m/s decelerating to ~28 m/s (integrated)
    T = 3.75
    return 0.0 + 40.0 * t * T - 6.0 * (t * T) ** 2 / T


def pos(t):
    rise = sb.smoother(sb.remap(t, 0.32, 1.0))
    return V((cam_x(t), sb.lerp(-11.3, -15.0, rise), sb.lerp(1.35, 19.0, rise) + float(ground(np.array(cam_x(t)), np.array(-11.3)))))


def tgt(t):
    p = pos(t)
    k = sb.smoother(sb.remap(t, 0.25, 0.95))
    near_t = V((p.x + 22.0, -1.5, p.z + 0.6 - (p.z - 2.0) * 0.0))
    far_t = V((CITY_X, 60.0, 30.0))
    return near_t.lerp(far_t, k)


def foc(t):
    p = pos(t)
    k = sb.smoother(sb.remap(t, 0.3, 0.8))
    return sb.lerp(abs(R.TRACK_Y[0] - p.y) + 1.0, 400.0, k)


sb.cam_bake(cam, F0, F1, pos, tgt, focus=foc)
sb.frames(F0, F1)
sb.render_shot(sid)
