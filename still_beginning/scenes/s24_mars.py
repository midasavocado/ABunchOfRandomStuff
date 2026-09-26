"""s24 LIFE TRAVELS WITH US (2250-2340): inside a pressurised greenhouse module on Mars. Afternoon sun through the
window bays catches healthy leaves; a human hand (MPFB, docs/PEOPLE.md) slips under a basil leaf and lifts it to look.
Through the windows: copper terrain, dust-hazed distance, butterscotch sky, coherent habitat modules. Nobody outside
without a suit; the plants are all inside the pressure hull."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, bmesh, sb, plants, suit
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler, noise

sid = (sb.argv() or ["s24"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
T0 = time.time()
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
HAND = os.environ.get("S24_HAND", "1") == "1"

SUN_EL, SUN_AZ = 24.0, 20.0          # sun beyond the windows (+Y), slightly right
SUN = sb.sun_dir(SUN_EL, SUN_AZ)
R_MOD, ZAX = 2.6, 1.35               # module radius / axis height (floor z = 0)
XL = 7.0                             # half length shown

# ---------------------------------------------------------------- Mars sky + light
w = bpy.data.worlds.new("MarsSky"); sc.world = w; w.use_nodes = True
nt = w.node_tree; nt.nodes.clear()
nb = sb.NB.__new__(sb.NB); nb.m = None; nb.nt = nt; nb.n = nt.nodes; nb.l = nt.links
tc = nt.nodes.new('ShaderNodeTexCoord')
d = tc.outputs['Generated']
sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(d, sep.inputs[0])
zz = nb.maprange(sep.outputs[2], -0.05, 1.0)
sky = nb.ramp(zz, [(0.0, (0.16, 0.08, 0.035)), (0.05, (0.52, 0.29, 0.135)), (0.14, (0.44, 0.24, 0.11)), (0.5, (0.28, 0.15, 0.07)), (1.0, (0.18, 0.10, 0.05))])
sdp = nb.vmath('DOT_PRODUCT', d, tuple(SUN))
halo = nb.math('POWER', nb.math('MAXIMUM', sdp, 0.0), 60.0)
sky = nb.mix(nb.math('MULTIPLY', halo, 0.8), sky, (0.55, 0.58, 0.62, 1), blend='ADD', clamp=False)   # bluish-white near the sun
disc = nb.maprange(sdp, math.cos(math.radians(0.36)), math.cos(math.radians(0.3)), 0.0, 1.0)
sky = nb.mix(nb.math('MULTIPLY', disc, 400.0), sky, (1.0, 0.97, 0.92, 1), blend='ADD', clamp=False)
bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs[1].default_value = 1.4
out = nt.nodes.new('ShaderNodeOutputWorld')
nt.links.new(sky, bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs[0])
sun = sb.sun(SUN_EL, SUN_AZ, energy=float(os.environ.get("S24_SUN", "5.5")), color=(1.0, 0.92, 0.80), angle=0.35)

# ---------------------------------------------------------------- Mars outside
def mars_ground_mat(name, haze_col=(0.50, 0.29, 0.14)):
    m = sb.mat(name, (0.34, 0.17, 0.08), rough=0.95, spec=0.3)
    n = sb.NB(m)
    co = n.coord('Object')
    n1 = n.noise(co, scale=0.02, detail=6, rough=0.6)
    n2 = n.noise(co, scale=0.4, detail=5, rough=0.6)
    col = n.mix(n.maprange(n1.outputs['Fac'], 0.3, 0.7), (0.15, 0.065, 0.03, 1), (0.34, 0.16, 0.075, 1))
    # dark basaltic patches + bright dust drifts, sand ripples (wave, ~0.4 m) in the drifts
    n3 = n.noise(co, scale=0.08, detail=4, rough=0.55)
    col = n.mix(n.maprange(n2.outputs['Fac'], 0.52, 0.72, 0.0, 0.75), col, (0.07, 0.04, 0.025, 1))
    drift = n.maprange(n3.outputs['Fac'], 0.55, 0.7)
    col = n.mix(n.math('MULTIPLY', drift, 0.6), col, (0.42, 0.22, 0.11, 1))
    rip = n.wave(n.mapping(co, scale=(1.0, 0.35, 1.0)), scale=2.4, wtype='BANDS', direction='Y', dist=4.0, detail=2)
    # aerial perspective: dust haze by distance from the module (kilometres, not hundreds of metres)
    cam_d = n.vmath('LENGTH', n.vmath('SUBTRACT', n.coord('Object'), (0.0, 0.0, 0.0)))
    hz = n.math('SUBTRACT', 1.0, n.math('EXPONENT', n.math('DIVIDE', cam_d, -2500.0)))
    col = n.mix(n.math('MULTIPLY', hz, 0.55), col, (*haze_col, 1))
    n.set('Base Color', col)
    n.set('Emission Color', (*haze_col, 1))
    n.set('Emission Strength', n.math('MULTIPLY', n.math('MULTIPLY', hz, hz), 0.18))
    bh = n.math('ADD', n.noise(co, scale=3.0, detail=6).outputs['Fac'], n.math('MULTIPLY', n.math('MULTIPLY', rip.outputs['Fac'], drift), 0.8))
    n.set('Normal', n.bump(bh, strength=0.5, distance=0.3))
    return m


gm = mars_ground_mat("MarsGround")
# heightfield: gentle plain with dunes, rim of mesas far away
N = 260
xs = np.linspace(-1800, 1800, N); ys = np.linspace(3, 3000, N)
ys = 3 + (3000 - 3) * np.linspace(0, 1, N) ** 2.2
X, Y = np.meshgrid(xs, ys, indexing='ij')
H = -1.2 + 0.8 * np.sin(X * 0.01) * np.cos(Y * 0.013) + 2.5 * np.sin(X * 0.003 + 1) * np.sin(Y * 0.0021)
H += 45.0 * np.clip((Y - 1600) / 900, 0, 1) ** 2 * (0.6 + 0.4 * np.sin(X * 0.0023 + 0.5))     # distant mesa rise
# flat-topped mesas + a long crater-rim ridge on the horizon (150-300 m, 2-3 km out): steep eroded flanks
def _mesa(cx, cy, r, h):
    d = np.hypot((X - cx) / r[0], (Y - cy) / r[1])
    return h * np.clip((1.25 - d) / 0.35, 0, 1) ** 1.6
H += _mesa(-900, 2500, (520, 260), 120) + _mesa(-250, 2750, (300, 200), 95) + _mesa(700, 2400, (650, 300), 150)
H += _mesa(1350, 2800, (500, 220), 110)
H += 50.0 * np.clip((Y - 2300) / 500, 0, 1) * (0.55 + 0.45 * np.sin(X * 0.0017 + 1.3))
H += 20.0 * np.clip((Y - 250) / 600, 0, 1) * (0.5 + 0.5 * np.sin(X * 0.004 + 2.0)) * np.clip((Y - 250) / 300, 0, 1)
ii = np.arange(N * N).reshape(N, N)
F = np.stack([ii[:-1, :-1], ii[1:, :-1], ii[1:, 1:], ii[:-1, 1:]], -1).reshape(-1, 4)
ground = suit.mesh_np("MarsTerrain", np.stack([X, Y, H], -1).reshape(-1, 3), F, [gm])
sb.recalc_normals(ground)
# rocks outside (a few, near)
rng = np.random.default_rng(3)
rk_m = sb.mat("MarsRock", (0.22, 0.11, 0.06), rough=0.8)
import moon
pts = np.stack([rng.uniform(-60, 60, 260), 6 + rng.uniform(0, 120, 260) ** 1.0], 1)
szs = 0.05 + rng.pareto(2.5, 260) * 0.12
class _T:
    def height(self, x, y):
        return -1.2 + 0.8 * np.sin(np.asarray(x) * 0.01) * np.cos(np.asarray(y) * 0.013) + 2.5 * np.sin(np.asarray(x) * 0.003 + 1) * np.sin(np.asarray(y) * 0.0021)
moon.rocks(_T(), "MarsRocks", pts, np.minimum(szs, 1.2), rk_m, seed=5)
# habitat structures outside: shielded modules + a dome + a tall mast, dusted
M = suit.materials(dust=0.6)
hab_white = suit.graphite_mat("HabWhite", color=(0.66, 0.60, 0.54), wear=0.3, dust=0.5)
SM = dict(shield=mars_ground_mat("MarsShield"), white=hab_white, graphite=M["graphite"],
          window=sb.emit_mat("HabWin", (1.0, 0.75, 0.45), 3.0), lamp=sb.emit_mat("HabLamp", (1.0, 0.95, 0.9), 8.0))
for i, (hx, hy, hd, L) in enumerate(((-26.0, 48.0, 15.0, 12.0), (-9.0, 70.0, -5.0, 11.0), (22.0, 58.0, 160.0, 12.0))):
    for o in moon.habitat("MHab%d" % i, _T(), hx, hy, hd, length=L, mats=SM, seed=i + 7):
        pass
dome = sb.prim("sphere", "Dome", loc=(40.0, 92.0, -1.0), segments=64, ring_count=32, radius=9.0, mat=hab_white)
dome.scale = (1, 1, 0.62)
ring_ = sb.prim("cyl", "DomeRing", loc=(40.0, 92.0, -0.6), vertices=64, radius=9.2, depth=1.2, mat=M["graphite"])
mast = sb.prim("cyl", "Mast", loc=(-40.0, 110.0, 8.0), vertices=12, radius=0.18, depth=18.0, mat=hab_white)
tun = sb.prim("cyl", "Corridor", loc=(0.0, 18.0, 0.2), vertices=32, radius=1.3, depth=30.0, mat=hab_white)
tun.rotation_euler = (math.radians(90), 0, math.radians(-8))

# solar field (tilted rows toward the sun) + a parked pressurised rover between the modules
cells = moon.cells_mat("MarsCells")
dust_film = sb.mat("PanelDust", (0.36, 0.2, 0.1), rough=0.9)
for r in range(5):
    for c in range(8):
        x = 30.0 + c * 4.6; y = 30.0 + r * 5.5
        z = float(_T().height(np.array([x]), np.array([y]))[0])
        pnl = sb.prim("cube", "SolarP", loc=(x, y, z + 1.0), rot=(math.radians(-28), 0, 0), scale=(2.1, 1.1, 0.03), mat=cells)
        sb.prim("cyl", "SolarLeg", loc=(x, y + 0.2, z + 0.45), vertices=8, radius=0.05, depth=0.9, mat=M["graphite"])
rz = float(_T().height(np.array([-6.0]), np.array([30.0]))[0])
rv_body = sb.prim("cube", "PRover", loc=(-6.0, 30.0, rz + 1.55), rot=(0, 0, math.radians(24)), scale=(2.6, 1.3, 0.9), mat=hab_white)
sb.bevel(rv_body, 0.45, 4)
for dx in (-1.7, 0.0, 1.7):
    for dy in (-1.35, 1.35):
        p_ = V((dx, dy, 0))
        p_.rotate(Euler((0, 0, math.radians(24))))
        w_ = sb.prim("cyl", "PRWheel", loc=(-6.0 + p_.x, 30.0 + p_.y, rz + 0.55), rot=(math.radians(90), 0, math.radians(24)),
                     vertices=32, radius=0.55, depth=0.45, mat=M["graphite"])
win_r = sb.prim("cube", "PRWin", loc=(-6.0 + 2.2 * math.cos(math.radians(24)), 30.0 + 2.2 * math.sin(math.radians(24)), rz + 1.9),
                rot=(0, 0, math.radians(24)), scale=(0.5, 1.0, 0.3), mat=sb.emit_mat("PRWinM", (1.0, 0.72, 0.42), 4.0))

# ---------------------------------------------------------------- greenhouse module interior
wall_m = suit.graphite_mat("GHWall", color=(0.70, 0.69, 0.66), wear=0.15)
rib_m = suit.graphite_mat("GHRib", color=(0.06, 0.062, 0.066), wear=0.4)
glass_m = suit.bubble_mat("GHGlass")
floor_m = sb.mat("Grating", (0.08, 0.08, 0.085), metal=0.7, rough=0.45)
# hull: cylinder panels with window bays on the +Y side (theta measured from +Y horizontal toward +Z)
NA, NXs = 96, 140
verts, faces = [], []
xs_ = np.linspace(-XL, XL, NXs)
ths = np.linspace(-math.pi * 0.62, math.pi * 1.62, NA)       # skip under the floor
for i, x in enumerate(xs_):
    for j, th in enumerate(ths):
        verts.append((x, R_MOD * math.cos(th), ZAX + R_MOD * math.sin(th)))
BAY = 1.6
def is_window(x, th):
    k = (x + XL) % BAY
    in_x = 0.12 < k < BAY - 0.12
    in_t = math.radians(-18) < th < math.radians(38)
    return in_x and in_t
for i in range(NXs - 1):
    for j in range(NA - 1):
        xm = (xs_[i] + xs_[i + 1]) / 2; tm = (ths[j] + ths[j + 1]) / 2
        if is_window(xm, tm):
            continue
        a = i * NA + j
        faces.append((a, a + 1, a + NA + 1, a + NA))
hull = sb.mesh_obj("Hull", verts, faces, wall_m)
sb.recalc_normals(hull, inside=True)
sb.solidify(hull, 0.06, offset=1)
glass = sb.mesh_obj("Glass", verts, [f for f in [(i * NA + j, i * NA + j + 1, (i + 1) * NA + j + 1, (i + 1) * NA + j)
                                               for i in range(NXs - 1) for j in range(NA - 1)]
                                     if is_window((xs_[f[0] // NA] + xs_[f[2] // NA]) / 2, (ths[f[0] % NA] + ths[f[1] % NA]) / 2)], glass_m)
glass.visible_shadow = False
# ribs + window frames
for k in range(int(2 * XL / BAY) + 1):
    x = -XL + k * BAY
    rb = suit.lathe("Rib", [(R_MOD - 0.02, -0.05), (R_MOD - 0.10, -0.045), (R_MOD - 0.12, 0.0), (R_MOD - 0.10, 0.045), (R_MOD - 0.02, 0.05)], 96, [rib_m], axis='Y')
    rb.rotation_euler = (0, 0, math.pi / 2)
    rb.location = (x, 0, ZAX)
for th in (math.radians(-18), math.radians(38)):
    fr = sb.prim("cube", "Sill", loc=(0, (R_MOD - 0.08) * math.cos(th), ZAX + (R_MOD - 0.08) * math.sin(th)), scale=(XL, 0.08, 0.05), mat=rib_m)
    fr.rotation_euler = (th, 0, 0)
floor = sb.prim("cube", "Floor", loc=(0, 0, 0.0), scale=(XL, 2.0, 0.03), mat=floor_m)
# troughs (soil) along X: hero trough at y = 0.35, second at y = -1.2
soil_m = sb.mat("Soil", (0.045, 0.030, 0.020), rough=0.85, spec=0.35)
nbs = sb.NB(soil_m)
co = nbs.coord('Object')
clump = nbs.voronoi(co, scale=90.0, feature='F1')
per = nbs.maprange(nbs.voronoi(co, scale=160.0, feature='F1').outputs['Distance'], 0.0, 0.06, 1.0, 0.0)
nbs.set('Base Color', nbs.mix(nbs.math('MULTIPLY', per, 0.8), nbs.mix(nbs.maprange(nbs.noise(co, scale=20).outputs['Fac'], 0.3, 0.7), (0.035, 0.024, 0.016, 1), (0.07, 0.05, 0.034, 1)), (0.75, 0.74, 0.72, 1)))
nbs.set('Roughness', nbs.maprange(nbs.noise(co, scale=8).outputs['Fac'], 0.4, 0.7, 0.45, 0.9))
nbs.set('Normal', nbs.bump(nbs.math('ADD', clump.outputs['Distance'], nbs.math('MULTIPLY', nbs.noise(co, scale=300).outputs['Fac'], 0.3)), strength=0.8, distance=0.004))
trough_m = suit.graphite_mat("Trough", color=(0.80, 0.79, 0.76), wear=0.2)
TROUGHS = [(0.45, 0.82), (1.55, 0.62)]
for ty, tz in TROUGHS:
    tb = suit.rbox("Trough", (2 * XL - 0.4, 0.62, 0.30), r=0.03, mats=[trough_m]); tb.location = (0, ty, tz - 0.165)
    sl = sb.prim("plane", "SoilTop", loc=(0, ty, tz + 0.005), size=1.0, mat=soil_m); sl.scale = (XL - 0.25, 0.28, 1)
    for x in np.arange(-XL + 0.6, XL - 0.4, 1.4):
        for yy in (-0.2, 0.2):
            lg = sb.prim("cyl", "TroughLeg", loc=(x, ty + yy, (tz - 0.28) / 2), vertices=12, radius=0.025, depth=tz - 0.28, mat=rib_m)
    drip = sb.tube_along("Drip", [(-XL, ty - 0.22, tz + 0.02), (XL, ty - 0.22, tz + 0.02)], radius=0.006, mat=sb.mat("DripPipe", (0.02, 0.02, 0.022), rough=0.4))
# grow light bars overhead
led_m = sb.emit_mat("LED", (1.0, 0.88, 0.80), 22.0)
for ty, tz in TROUGHS:
    bar = sb.prim("cube", "GrowBar", loc=(0, ty, tz + 1.05), scale=(XL - 0.3, 0.06, 0.012), mat=rib_m)
    ld = sb.prim("cube", "GrowLED", loc=(0, ty, tz + 1.036), scale=(XL - 0.35, 0.045, 0.003), mat=led_m)
    gl = sb.light('AREA', "GrowL", loc=(0, ty, tz + 1.02), rot=(0, 0, 0), energy=260, color=(1.0, 0.86, 0.80), size=2 * XL - 0.8, size_y=0.12)
    gl.data.shape = 'RECTANGLE'
    gl.visible_glossy = False

# ---------------------------------------------------------------- plants
wm = plants.droplet_mat()
basil_m = plants.leaf_mat("BasilLeaf", (0.030, 0.105, 0.016), gloss=0.35, translucency=0.2)
basil_s = plants.stem_mat("BasilStem", (0.10, 0.20, 0.05))
let_m = plants.leaf_mat("LettuceLeaf", (0.10, 0.24, 0.025), gloss=0.2, translucency=0.28, veins=9.0, var=0.2)
tom_l = plants.leaf_mat("TomLeaf", (0.045, 0.13, 0.025), gloss=0.15, translucency=0.22, veins=6.0, hairy=0.3)
fruit_ms = [sb.mat("TomRed", (0.55, 0.05, 0.02), rough=0.18, coat=0.6, sss=0.2), sb.mat("TomOrange", (0.6, 0.2, 0.03), rough=0.2, coat=0.5),
            sb.mat("TomGreen", (0.2, 0.3, 0.06), rough=0.2, coat=0.5)]
HERO = V((0.05, 0.45, 0.83))
chard_m = plants.chard_leaf_mat("ChardLeaf")
stalk_m = sb.mat("ChardStalk", (0.80, 0.40, 0.06), rough=0.3, coat=0.5, sss=0.3, sss_radius=(1, 0.5, 0.2), sss_scale=0.01)
hero = plants.chard("HeroChard", HERO, seed=12, leaf_m=chard_m, stalk_m=stalk_m, n=9)
rng2 = np.random.default_rng(1)
k = 0
for ty, tz in TROUGHS:
    for x in np.arange(-XL + 0.4, XL - 0.3, 0.26):
        k += 1
        if ty < 1 and abs(x - HERO.x) < 0.22:
            continue
        p = V((x + rng2.uniform(-0.04, 0.04), ty + rng2.uniform(-0.1, 0.1), tz + 0.01))
        pre = "P%d_" % k
        if ty < 1 and k % 4 == 0:
            plants.chard(pre + "C", p, seed=k, leaf_m=chard_m, stalk_m=stalk_m, n=7, size=rng2.uniform(0.75, 0.95))
        elif ty < 1 and k % 4 == 2:
            plants.basil(pre + "B", p, seed=k, height=rng2.uniform(0.18, 0.24), leaf_m=basil_m, stem_m=basil_s, nodes=5)
        else:
            plants.lettuce(pre + "L", p, seed=k, radius=rng2.uniform(0.10, 0.15) * (0.8 if ty > 1 else 1.0), leaf_m=let_m)
        plants.merge(plants.plant_objects(pre), pre + "M")
for k2, x in enumerate(np.arange(-XL + 0.6, -2.0, 0.55)):
    plants.tomato("Tom%d_" % k2, V((x, 0.45, 0.83)), seed=k2, height=1.1, leaf_m=tom_l, fruit_m=fruit_ms)
    plants.merge([o for o in plants.plant_objects("Tom%d_" % k2) if "Fruit" not in o.name], "Tom%dM" % k2)
# droplets on the hero leaves
for lf in hero["leaves"][:14]:
    plants.droplets("Drop", lf, n=2, rmin=0.0010, rmax=0.0024, mat=wm, seed=abs(hash(lf.name)) % 999)

# ---------------------------------------------------------------- the gardener's hand (MPFB, docs/PEOPLE.md)
def f_rel(fr):
    return fr - S0


# hero leaf: front-right leaf of the hero chard (toward camera side, reachable from the right)
cands = sorted(hero["leaves"], key=lambda o: (V(o["stalk_end"]).y - 0.6 * V(o["stalk_end"]).x))
HL = cands[0]
bpy.context.view_layer.update()
HL_M0 = HL.matrix_world.copy()
pivot = V(HL["stalk_end"])
# contact point on the underside of the midrib at 45% of the blade
nu_, nv_ = 22, 34
jv = int(0.45 * nv_)
vi = jv * (nu_ + 1) + nu_ // 2
c_local = HL.data.vertices[vi].co.copy() - V((0, 0, 0.0006))
lift_axis = (HL_M0.to_3x3() @ V((1, 0, 0))).normalized()     # leaf local X = hinge across the stalk


def lift_angle(f):
    k = sb.smoother(sb.remap(f, 47, 76))
    return math.radians(-13.0) * k + math.radians(0.6) * math.sin(max(0, f - 76) * 0.25) * sb.remap(f, 76, 89)


def leaf_matrix(f):
    R = Matrix.Rotation(lift_angle(f), 4, lift_axis)
    return Matrix.Translation(pivot) @ R @ Matrix.Translation(-pivot) @ HL_M0


def contact_world(f):
    return leaf_matrix(f) @ c_local


if HAND:
    import mhchild
    bm_, rig, parts = mhchild.build(name="Gardener", phenotype=dict(age=0.42, gender=0.25, weight=0.52, muscle=0.5,
                                    race=dict(african=0.5, asian=0.3, caucasian=0.2)), hair="bob01",
                                    clothes=["male_casualsuit01"], skin="young_african_female")
    mhchild.recolor_top(parts, color=(0.20, 0.22, 0.16))           # sage work shirt
    GP = V((0.323, -0.068, 0.0))
    rig.location = GP
    rig.rotation_euler = (0, 0, math.radians(160))                   # faces +Y toward the trough, slightly left
    bpy.context.view_layer.update()
    pb = rig.pose.bones
    # IK arm to a wrist target, wrist orientation from a target empty
    tgt = sb.empty("WristTgt"); pole = sb.empty("ElbowPole")
    ik = pb["lowerarm02.R"].constraints.new('IK')
    ik.target = tgt; ik.pole_target = pole; ik.chain_count = 4; ik.pole_angle = math.radians(-90)
    rot = sb.empty("HandRot")
    cr = pb["wrist.R"].constraints.new('COPY_ROTATION')
    cr.target = rot
    B = rig.data.bones
    Rw_rest = (rig.matrix_world.to_3x3() @ B["wrist.R"].matrix_local.to_3x3())
    Yf = (rig.matrix_world.to_3x3() @ B["finger3-1.R"].matrix_local.to_3x3()) @ V((0, 1, 0))
    Zf = (rig.matrix_world.to_3x3() @ B["finger3-1.R"].matrix_local.to_3x3()) @ V((0, 0, 1))

    def frame_rot(y, z):
        y = y.normalized(); z = (z - y * z.dot(y)).normalized(); x = y.cross(z)
        return Matrix((x, y, z)).transposed()
    R_rest = frame_rot(Yf, Zf)
    # finger curls (radians about bone X): relaxed supporting hand; index & middle nearly straight under the leaf
    CURL = {"finger2": (0.10, 0.12, 0.08), "finger3": (0.16, 0.18, 0.1), "finger4": (0.45, 0.55, 0.35), "finger5": (0.6, 0.7, 0.4)}
    for fn, cs in CURL.items():
        for k_, c_ in enumerate(cs):
            p_ = pb.get("%s-%d.R" % (fn, k_ + 1))
            if p_:
                p_.rotation_mode = 'XYZ'; p_.rotation_euler = (c_, 0, 0)
    for k_, c_ in enumerate((0.15, 0.25, 0.2)):
        p_ = pb.get("finger1-%d.R" % (k_ + 1))
        if p_:
            p_.rotation_mode = 'XYZ'; p_.rotation_euler = (c_, 0, -0.1)
    DIRN = V((-0.55, 0.80, 0.10)).normalized()          # fingers point in toward the plant
    PALM = V((0.12, -0.05, 1.0)).normalized()           # palm up (supporting)
    Rdes = frame_rot(DIRN, PALM)
    Rd = Rdes @ R_rest.inverted()
    rot.rotation_mode = 'QUATERNION'
    rot.rotation_quaternion = (Rd @ Rw_rest).to_quaternion()

    def pad_point():
        """world position of the index finger pad (distal phalanx, palm side)."""
        dg = bpy.context.evaluated_depsgraph_get()
        pbd = rig.pose.bones["finger2-3.R"]
        Mb = rig.matrix_world @ pbd.matrix
        head = Mb.translation; tail = rig.matrix_world @ pbd.tail
        zf = (Mb.to_3x3() @ V((0, 0, 1))).normalized()
        return head.lerp(tail, 0.55) + zf * 0.0068

    def solve_target(goal, guess):
        tgt.location = guess
        for _ in range(6):
            bpy.context.view_layer.update()
            err = goal - pad_point()
            tgt.location = tgt.location + err
        bpy.context.view_layer.update()
        return tgt.location.copy(), (goal - pad_point()).length

    pole.location = GP + V((0.45, 0.2, 0.6))
    C_rest = contact_world(0)
    t_contact, e0 = solve_target(C_rest, C_rest + V((0.05, -0.12, -0.05)))
    print("HAND contact err mm", e0 * 1000, "contact", tuple(round(c, 3) for c in C_rest),
          "shoulder", tuple(round(c, 3) for c in rig.matrix_world @ pb["upperarm01.R"].head), "pad", tuple(round(c, 3) for c in pad_point()))
    # approach path: from lower-right-front toward the contact under the leaf, arriving with a gentle ease
    APP0 = t_contact + V((0.16, -0.14, -0.10))

    def wrist_target(f):
        if f < 47:
            k = sb.smoother(sb.remap(f, 6, 47))
            base = APP0.lerp(t_contact, k)
            # slight arc: comes up from below the leaf
            base = base + V((0, 0, -0.03 * math.sin(math.pi * k)))
            return base
        # after contact: follow the leaf contact exactly (solved per frame below)
        return None
    for fr in range(S0 - 1, S1 + 2):
        f = f_rel(fr)
        wt = wrist_target(f)
        if wt is None:
            wt, e = solve_target(contact_world(f), tgt.location.copy())
        tgt.location = wt
        tgt.keyframe_insert("location", frame=fr)
        HL.matrix_world = leaf_matrix(f)
        HL.keyframe_insert("location", frame=fr); HL.keyframe_insert("rotation_euler", frame=fr)
        # fingers: tiny living motion (thumb brushes, ring/pinky relax)
        p_ = pb.get("finger1-2.R")
        if p_:
            p_.rotation_euler = (0.25 + 0.12 * sb.smooth(sb.remap(f, 55, 80)), 0, -0.1); p_.keyframe_insert("rotation_euler", frame=fr)
    # the droplets on the hero leaf ride with it
    for o in bpy.data.objects:
        if o.name.startswith("Drop") and (o.location - HL_M0.translation).length < 0.25:
            pass

# ---------------------------------------------------------------- camera
# aimed so the hand's lift (contact ~(0.23, 0.33)) plays just right of centre, the window bays behind it
cam = sb.camera("Cam", loc=(-0.72, -0.30, 0.95), target=V((0.36, 0.95, 0.97)), lens=50, fstop=5.6, clip=(0.02, 10000))
cam.data.dof.focus_distance = 1.14
FOCUS = os.environ.get("S24_FOCUS")
if FOCUS:
    cam.data.dof.focus_distance = float(FOCUS)
import dbgcam
dbgcam.apply()
print("BUILD %.1fs" % (time.time() - T0))
sb.frames(S0, S1)
if os.environ.get("SB_SAVE"):
    sb.save("s24")
sb.render_shot(sid)
