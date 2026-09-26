"""s23 ANOTHER SHORE (2160-2250): a modest lunar south-pole research settlement at grazing sunlight. A small
pressurised-suit crew rover crosses the foreground (wheels roll without slip, ballistic dust that falls at once),
regolith-shielded inflatable habitats, vertical solar arrays on the ridge, Earth low over the massifs. Airless:
black sky, one hard sun, no haze."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, moon
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler

sid = (sb.argv() or ["s23"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
T0 = time.time()
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
sc.eevee.shadow_resolution_scale = 1.0
STAGE = os.environ.get("S23_STAGE", "full")

SUN_EL, SUN_AZ = 6.0, -118.0
SUN = sb.sun_dir(SUN_EL, SUN_AZ)
CAM0 = V((0.0, 0.0, 0.0))

# settlement pads (graded by the builders)
HAB = [(-34.0, 78.0, 0.0), (-18.0, 92.0, 0.0), (10.0, 104.0, 0.0)]
T = moon.Terrain(seed=11, n_craters=int(os.environ.get('S23_NC', 22000)), rmin=0.18, big=[(55.0, 60.0, 26.0, 4.5), (-70.0, 30.0, 14.0, 2.6), (140.0, 260.0, 60.0, 9.0)],
                 flat=[])
T.flat = []
zpad = float(T.height(np.array([-4.0]), np.array([63.0]))[0])
T.flat = [(-4.0, 63.0, 30.0, zpad)]

reg = moon.regolith_mat("Regolith", SUN)
ground = moon.fan_mesh(T, "Ground", (0.0, -2.0), -75, 75, 1.2, 2600.0, nr=int(os.environ.get("S23_NR", 900)),
                       na=int(os.environ.get("S23_NA", 1300)), mat=reg,
                       cache=os.path.join(sb.ROOT, "assets", "moon_ground_%s_%s_%s.npz" % (os.environ.get("S23_NR", 900), os.environ.get("S23_NA", 1300), os.environ.get("S23_NC", 22000))))
rockm = moon.rock_mat("Rock", SUN)
rng = np.random.default_rng(4)
p1, s1 = moon.scatter(T, rng, 5000, 0, -2, 2.0, 45, -70, 70, 0.03, 0.7)
p2, s2 = moon.scatter(T, rng, 1400, 0, -2, 40, 260, -70, 70, 0.08, 1.8)
p3, s3 = moon.scatter(T, rng, 8000, 0, -2, 1.3, 14, -70, 70, 0.008, 0.06, alpha=2.6)
pts = np.concatenate([p1, p2, p3]); sz = np.concatenate([s1, s2, s3])
# keep the rover lane and settlement pad clear of big rocks
lane = (np.abs(pts[:, 1] - 9.5) < 1.9) & (sz > 0.04)
pad = np.hypot(pts[:, 0] + 4, pts[:, 1] - 63) < 32
keep = ~(lane | (pad & (sz > 0.1)))
rk = moon.rocks(T, "Rocks", pts[keep], sz[keep], rockm, seed=2)
mass_m = moon.regolith_mat("Massif", SUN, albedo=0.115, bump=0.25, attr_fresh=False, coord_scale=0.08)
# south-polar massifs (Malapert / Mouton scale): several km high, 20-60 km away, rising well above the horizon
# (their camera-facing flanks only catch the grazing sun right of centre: the sun is behind-left, so the big ones stand
# there; a low gap under the Earth)
PEAKS = [(-14.0, 44000, 4200, 12000), (-34.0, 64000, 2600, 14000), (4.0, 62000, 3600, 12000), (40.0, 26000, 2400, 7000),
         (58.0, 40000, 4600, 12000), (30.0, 78000, 3600, 16000), (75.0, 52000, 3800, 15000)]
ms = moon.massifs("Massifs", (0, 0), -80, 80, mat=mass_m, peaks=PEAKS)

# ---------------------------------------------------------------- sky: black + Earth (earth.py) + sun lamp
try:
    import earth
    ed = V((0.35, 1.0, 0.0)).normalized()
    el = math.radians(6.0)
    edir = V((ed.x * math.cos(el), ed.y * math.cos(el), math.sin(el)))
    E = earth.build(center_km=tuple(edir * 384400.0), sun_dir=tuple(SUN), clouds=0.55, sun_strength=5.0, sun_disc=False,
                    nadir=(-30.0, 40.0, 0.0), samples=12)
    E.track(None, S0, S1) if False else None
except Exception as ex:
    print("EARTH FAIL", ex)
    sb.world_color((0, 0, 0), 1.0)
sun = sb.sun(SUN_EL, SUN_AZ, energy=float(os.environ.get("S23_SUN", "9.0")), color=(1.0, 0.985, 0.97), angle=0.53)
sun.data.use_shadow = True
try:
    sun.data.shadow_maximum_resolution = 0.001
except Exception:
    pass

# ---------------------------------------------------------------- settlement
import suit
M = suit.materials(dust=0.55, dirt=0.3)
SM = dict(shield=moon.shield_mat("Shield", SUN), white=suit.graphite_mat("HabWhite", color=(0.72, 0.71, 0.68), wear=0.3, dust=0.4),
          graphite=M["graphite"], window=sb.emit_mat("HabWindow", (1.0, 0.68, 0.36), 40.0), lamp=sb.emit_mat("HabLamp", (1.0, 0.93, 0.84), 80.0),
          cells=moon.cells_mat("Cells"), metal=M["metal"], wheel=moon.wheel_mat(), seat=M["fabric_soft"],
          screen=sb.emit_mat("RoverScreen", (0.55, 0.62, 0.7), 0.6))
habs = []
for i, (hx, hy, hd, L) in enumerate(((-24.0, 62.0, -8.0, 11.0), (-3.0, 70.0, 172.0, 12.5), (15.0, 58.0, -38.0, 10.0))):
    habs += moon.habitat("Hab%d" % i, T, hx, hy, hd, length=L, mats=SM, seed=i + 1)
# covered tunnels between modules
for (a_, b_) in (((-17.0, 64.0), (-9.5, 69.0)), ((3.5, 67.0), (9.0, 61.5))):
    za, zb = float(T.height(np.array([a_[0]]), np.array([a_[1]]))[0]), float(T.height(np.array([b_[0]]), np.array([b_[1]]))[0])
    pth = np.linspace((a_[0], a_[1], za + 0.4), (b_[0], b_[1], zb + 0.4), 20)
    tu, _ = suit.tube("Tunnel", pth, lambda i, s_: (1.25, 1.1, 2.3), n=40, up=(0, 0, 1),
                      disp=lambda i, s_, th, p: 0.12 * __import__('mathutils').noise.noise(__import__('mathutils').Vector((p[0] * 0.4, p[1] * 0.4, th))),
                      mats=[SM["shield"]])
# vertical solar arrays on the ridge to the right (face the sun)
saz = math.atan2(SUN.x, -SUN.y)
for i, (ax_, ay_) in enumerate(((38.0, 118.0), (52.0, 132.0), (66.0, 146.0), (-48.0, 128.0), (-62.0, 142.0), (-76.0, 156.0))):
    r_, objs = moon.solar_array("Array%d" % i, T, ax_, ay_, 0.0, SM, height=15.0 + (i % 3))
    for o in objs:
        if o.name.startswith("Array%dYaw" % i) or o.parent and o.parent.name.startswith("Array%dYaw" % i):
            pass
    yaw = bpy.data.objects["Array%dYaw" % i]
    yaw.rotation_euler = (0, 0, saz)
# power cables on the ground from the arrays to the habitats
for (ax_, ay_) in ((38.0, 118.0), (-48.0, 128.0)):
    xs_ = np.linspace(ax_, -3.0, 120); ys_ = np.linspace(ay_, 72.0, 120)
    xs_ += np.sin(np.linspace(0, 9, 120)) * 1.5
    zs_ = T.height(xs_, ys_) + 0.03
    cb = sb.tube_along("Cable", list(zip(xs_, ys_, zs_)), radius=0.025, mat=M["graphite"], bevel_res=2)
# lander (descent stage) far left
land = sb.empty("Lander", loc=(-30.0, 190.0, float(T.height(np.array([-30.0]), np.array([190.0]))[0])))
foil = sb.mat("Foil", (0.85, 0.62, 0.28), metal=1.0, rough=0.35)
lb = sb.prim("cyl", "LanderBody", vertices=8, radius=2.2, depth=1.6, mat=foil, loc=(0, 0, 2.3), parent=land)
lt = sb.prim("cyl", "LanderTop", vertices=8, radius=1.6, depth=1.4, mat=SM["white"], loc=(0, 0, 3.8), parent=land)
for k in range(4):
    a_ = k * math.pi / 2 + math.pi / 4
    ft = V((math.cos(a_) * 4.0, math.sin(a_) * 4.0, 0.1))
    tp = V((math.cos(a_) * 1.8, math.sin(a_) * 1.8, 2.4))
    lg = sb.prim("cyl", "LanderLeg", vertices=10, radius=0.09, depth=(tp - ft).length, mat=M["metal"], loc=(ft + tp) / 2, parent=land)
    lg.rotation_mode = 'QUATERNION'; lg.rotation_quaternion = (tp - ft).to_track_quat('Z', 'Y')
    sb.prim("cyl", "LanderPad", vertices=16, radius=0.45, depth=0.1, mat=M["metal"], loc=ft, parent=land)
# comm dish near the habs
dz = float(T.height(np.array([24.0]), np.array([70.0]))[0])
dish = suit.lathe("CommDish", [(0.0, 0.0), (0.6, 0.06), (1.2, 0.25), (1.25, 0.28)], 64, [SM["white"]], axis='Z')
dish.location = (24.0, 70.0, dz + 3.2); dish.rotation_euler = (math.radians(70), 0, math.radians(200))
sb.prim("cyl", "DishMast", vertices=12, radius=0.08, depth=3.2, mat=M["graphite"], loc=(24.0, 70.0, dz + 1.6))
# standing astronaut by the middle airlock
st = suit.build("Crew2", visor="gold", mats=M, dust=0.55, glove_voxel=2.0, detail=0.6)
sx_, sy_ = 21.0, 55.5
st["root"].location = (sx_, sy_, float(T.height(np.array([sx_]), np.array([sy_]))[0]) + 0.005)
st["root"].rotation_euler = (0, 0, math.radians(200))
suit.pose(st, l_shoulder=(25, 8, 0), l_elbow=55, r_shoulder=(-5, 10, 0), r_elbow=15, torso=(4, 0, 0))

# ---------------------------------------------------------------- rover + driver, rolling without slip
RV = moon.rover("Rover", M, SM)
LANE_Y = 9.5
X0, X1 = -8.6, 3.4                  # body centre x at shot start / end (constant speed, already moving)
VEL = (X1 - X0) / ((S1 - S0) / 24.0)
HF = moon.local_heightfield(T, -20.0, 10.0, LANE_Y - 3.0, LANE_Y + 3.0, 0.04)


def rover_x(f):
    return X0 + VEL * (f - S0) / 24.0


drv = suit.build("Driver", visor="gold", mats=M, dust=0.35, glove_voxel=1.6, detail=0.7)
suit.pose(drv, l_hip=(82, 6, 0), r_hip=(82, 6, 0), l_knee=78, r_knee=78, l_ankle=5, r_ankle=5, torso=(-6, 0, 0),
          l_shoulder=(42, 6, 0), r_shoulder=(42, 6, 0), l_elbow=58, r_elbow=58, l_wrist=(0, 0, 0), r_wrist=(0, 0, 0))
drv["root"].parent = RV["body"]
drv["root"].matrix_parent_inverse = Matrix.Identity(4)
drv["root"].location = RV["seat_driver"].to_translation() + V((0.08, 0.0, -0.905 + 0.02))
drv["root"].rotation_euler = (0, 0, math.pi / 2)

wheel_local = [c for (_, c) in RV["wheels"]]
for fr in range(S0 - 2, S1 + 3):
    x = rover_x(fr)
    RV["root"].location = (x, LANE_Y, 0.0)
    RV["root"].keyframe_insert("location", frame=fr)
    zs = []
    for (spin, c), we in zip(RV["wheels"], RV["wheel_empties"]):
        wx, wy = x + c.x, LANE_Y + c.y
        zg = float(HF(wx, wy)) - 0.025              # tyre sinks ~2.5 cm into the regolith
        we.location = (c.x, c.y, zg + moon.WHEEL_R)
        we.keyframe_insert("location", frame=fr)
        zs.append(zg)
        spin.rotation_euler = (0, (x - X0) / moon.WHEEL_R, 0)
        spin.keyframe_insert("rotation_euler", frame=fr)
    # body: rigid chassis riding on the four contact heights (front/rear, left/right)
    zf_ = (zs[2] + zs[3]) / 2; zr_ = (zs[0] + zs[1]) / 2
    zl_ = (zs[1] + zs[3]) / 2; zR_ = (zs[0] + zs[2]) / 2
    pitch = math.atan2(zf_ - zr_, moon.WHEELBASE)
    roll = math.atan2(zl_ - zR_, moon.TRACK)
    RV["body"].location = (0, 0, sum(zs) / 4)
    RV["body"].rotation_euler = (roll, -pitch, 0)
    RV["body"].keyframe_insert("location", frame=fr)
    RV["body"].keyframe_insert("rotation_euler", frame=fr)

# tracks: this rover's fresh tracks (revealed behind it) + older tracks toward the habitats
moon.add_tracks(reg, [((-40.0, LANE_Y), (6.0, LANE_Y)), ((-22.0, 4.0), (-24.0, 80.0)), ((-30.0, 15.0), (-9.0, 86.0))],
                reveal={0: "track_x"})
tv = reg.node_tree.nodes["track_x"]
for fr in range(S0 - 2, S1 + 3):
    tv.outputs[0].default_value = rover_x(fr) - moon.WHEELBASE / 2 + moon.WHEEL_R * 0.4
    tv.outputs[0].keyframe_insert("default_value", frame=fr)

# ---------------------------------------------------------------- ballistic dust (no air: it just falls)
dust_m = sb.mat("DustGrains", (0.12, 0.117, 0.11), rough=1.0, spec=0.2)
rng2 = np.random.default_rng(7)
T_A, T_B = (S0 - 30) / 24.0, (S1 + 2) / 24.0
RATE, LIFE = 14000, 1.3
N = int(RATE * (T_B - T_A))
birth = np.sort(rng2.uniform(T_A, T_B, N))
wid = rng2.integers(0, 4, N)
phi = rng2.uniform(math.radians(195), math.radians(250), N)
lat = rng2.uniform(-0.1, 0.1, N)
sf = rng2.uniform(0.15, 1.0, N) ** 1.5
spread = rng2.normal(0, 1, (N, 3)) * np.array([0.25, 0.35, 0.2])
size = np.exp(rng2.normal(math.log(0.006), 0.5, N))
wl = np.array([[c.x, c.y] for c in wheel_local])
bx = X0 + VEL * (birth * 24.0 - S0) / 24.0 + wl[wid, 0]
by = LANE_Y + wl[wid, 1]
bz = HF(bx, by) - 0.025 + moon.WHEEL_R
om = VEL / moon.WHEEL_R
# rim point (forward = +X); angle phi from +X toward +Z
px0 = bx + np.cos(phi) * moon.WHEEL_R; pz0 = bz + np.sin(phi) * moon.WHEEL_R; py0 = by + lat
tx, tz = -np.sin(phi), np.cos(phi)
vx = VEL + tx * om * moon.WHEEL_R * sf + spread[:, 0] * VEL
vz = tz * om * moon.WHEEL_R * sf * -1.0
vz = np.abs(vz) * 0.55 + spread[:, 2] * VEL * 0.3
vx = VEL * 0.15 - np.abs(tx * om * moon.WHEEL_R * sf) * 0.9 + spread[:, 0] * VEL * 0.4   # thrown backwards
vy = spread[:, 1] * VEL
pc = moon.point_cloud("Dust", N, dust_m)


def dust_update(scene, depsgraph=None):
    t = (scene.frame_current + scene.frame_subframe) / 24.0
    a = t - birth
    x = px0 + vx * a; y = py0 + vy * a; z = pz0 + vz * a - 0.5 * moon.G_MOON * a * a
    g = HF(x, y)
    alive = (a >= 0) & (z > g) & (a < LIFE)
    P = np.stack([np.where(alive, x, bx), np.where(alive, y, by), np.where(alive, z, -5.0)], 1)
    me = pc.data
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    me.attributes["radius"].data.foreach_set("value", np.where(alive, size, 0.0).astype(np.float32))
    me.update()


bpy.app.handlers.frame_change_pre.append(dust_update)

# ---------------------------------------------------------------- camera: low, follows the rover with a slow pan
zc = float(T.height(np.array([0.0]), np.array([-2.0]))[0])
cam = sb.camera("Cam", loc=(0, -2, zc + 1.1), target=(0, 60, zc), lens=35, clip=(0.1, 200000))
cam.data.dof.use_dof = True
cam.data.dof.aperture_fstop = 8.0


def cam_pos(t):
    return V((0.5 * t - 0.2, -2.0 + 0.4 * t, zc + 1.1))


def cam_tgt(t):
    k = sb.smooth(t) * 0.6 + t * 0.4
    return V((sb.lerp(-5.0, 6.0, k), 60.0, zc - 4.0))


sb.cam_bake(cam, S0 - 2, S1 + 2, lambda t: cam_pos((t * (S1 - S0 + 4) - 2) / (S1 - S0)),
            lambda t: cam_tgt((t * (S1 - S0 + 4) - 2) / (S1 - S0)), focus=11.8)
import dbgcam
dbgcam.apply()
print("BUILD %.1fs" % (time.time() - T0))
sb.frames(S0, S1)
if os.environ.get("SB_SAVE"):
    sb.save("s23")
sb.render_shot(sid)
