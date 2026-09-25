"""Shared hero props: the child's refractor telescope, rooftop world."""
import bpy, math, random
from mathutils import Vector as V, Euler, Matrix
import sb

AMBER = (0.95, 0.42, 0.08)


def amber_anodized(name="AmberAnod"):
    return sb.brushed_metal(name, color=(0.90, 0.40, 0.07), rough=0.24, aniso=0.5, scale=300, bump=0.03)


def knurl_mat(name, color=(0.03, 0.03, 0.035)):
    m = sb.mat(name, color, metal=0.6, rough=0.4)
    nb = sb.NB(m)
    co = nb.coord('Object')
    w1 = nb.wave(nb.mapping(co, rot=(0, 0, 0.6)), scale=900, wtype='BANDS', direction='Z')
    w2 = nb.wave(nb.mapping(co, rot=(0, 0, -0.6)), scale=900, wtype='BANDS', direction='Z')
    h = nb.math('MULTIPLY', w1.outputs['Fac'], w2.outputs['Fac'])
    nb.set('Normal', nb.bump(h, strength=0.6, distance=0.0004))
    return m


def telescope(alt_deg=34.0, az_deg=0.0, height=1.02, name="Scope", ep_tilt=3.0):
    """Builds a small refractor on an alt-az tripod. Root empty at the tripod foot centre.
    Returns dict with root, tube (pivot empty), objective centre helper, focus knob, eyepiece."""
    root = sb.empty(name)
    white = sb.painted(name + "White", (0.82, 0.83, 0.84), rough=0.3, coat=0.4, grime=0.12)
    graphite = sb.painted(name + "Graphite", (0.055, 0.058, 0.064), rough=0.45, coat=0.1)
    black = sb.mat(name + "Flock", (0.006, 0.006, 0.007), rough=0.95, spec=0.1)
    alu = sb.brushed_metal(name + "Alu", (0.72, 0.73, 0.75), rough=0.3, scratches=0.3)
    amber = amber_anodized(name + "Amber")
    knurl = knurl_mat(name + "Knurl")
    # tripod
    head = sb.empty(name + "Head", loc=(0, 0, height), parent=root)
    for i in range(3):
        a = math.radians(90 + i * 120 + az_deg)
        foot = V((math.cos(a) * 0.42, math.sin(a) * 0.42, 0))
        top = V((math.cos(a) * 0.05, math.sin(a) * 0.05, height - 0.06))
        mid = (foot + top) / 2
        L = (top - foot).length
        leg = sb.prim("cyl", f"{name}Leg{i}", loc=mid, vertices=24, radius=0.013, depth=L, mat=alu)
        leg.rotation_mode = 'QUATERNION'
        leg.rotation_quaternion = (top - foot).to_track_quat('Z', 'Y')
        leg.parent = root
        low = sb.prim("cyl", f"{name}LegLow{i}", loc=foot.lerp(top, 0.18), vertices=24, radius=0.016, depth=L * 0.36, mat=graphite)
        low.rotation_mode = 'QUATERNION'
        low.rotation_quaternion = leg.rotation_quaternion
        low.parent = root
        rub = sb.prim("sphere", f"{name}Foot{i}", loc=foot + V((0, 0, 0.012)), radius=0.02, mat=sb.rubber(name + "Rub"))
        rub.scale = (1, 1, 0.6)
        rub.parent = root
    tray = sb.prim("cyl", name + "Tray", loc=(0, 0, height * 0.45), vertices=6, radius=0.12, depth=0.012, mat=graphite)
    tray.parent = root
    hub = sb.prim("cyl", name + "Hub", loc=(0, 0, height - 0.04), vertices=48, radius=0.07, depth=0.08, mat=graphite)
    hub.parent = root
    sb.bevel(hub, 0.006)
    # az/alt mount
    az = sb.empty(name + "Az", loc=(0, 0, height + 0.02), parent=root)
    az.rotation_euler = (0, 0, math.radians(az_deg))
    arm = sb.prim("cube", name + "Arm", loc=(0.0, -0.09, 0.12), scale=(0.035, 0.018, 0.14), mat=graphite, parent=az)
    sb.bevel(arm, 0.008)
    base = sb.prim("cyl", name + "AzBase", loc=(0, 0, 0.0), vertices=48, radius=0.06, depth=0.04, mat=white, parent=az)
    sb.bevel(base, 0.005)
    amber_ring0 = sb.prim("cyl", name + "AzRing", loc=(0, 0, 0.022), vertices=64, radius=0.061, depth=0.006, mat=amber, parent=az)
    alt = sb.empty(name + "Alt", loc=(0.0, -0.06, 0.22), parent=az)
    alt.rotation_euler = (math.radians(alt_deg), 0, 0)
    # tube along +Y (local), pivot at alt
    tube = sb.prim("cyl", name + "Tube", loc=(0, 0.08, 0), rot=(math.pi / 2, 0, 0), vertices=96, radius=0.047, depth=0.62, mat=white, parent=alt)
    sb.bevel(tube, 0.003, 2)
    inner = sb.prim("cyl", name + "TubeIn", loc=(0, 0.08, 0), rot=(math.pi / 2, 0, 0), vertices=64, radius=0.044, depth=0.63, mat=black, parent=alt)
    # dew shield (graphite) with amber anodized front rim
    shield = sb.lathe(name + "Shield", [(0.0555, 0.0), (0.0555, 0.20), (0.0530, 0.2045), (0.0505, 0.20), (0.0505, 0.004), (0.0475, 0.0)], segs=128, mat=graphite)
    shield.rotation_euler = (-math.pi / 2, 0, 0)
    shield.location = (0, 0.33, 0)
    shield.parent = alt
    rim = sb.lathe(name + "Rim", [(0.0558, 0.0), (0.0562, 0.012), (0.0548, 0.016), (0.0506, 0.016), (0.0500, 0.012), (0.0503, 0.0)], segs=160, mat=amber)
    rim.rotation_euler = (-math.pi / 2, 0, 0)
    rim.location = (0, 0.33 + 0.198, 0)
    rim.parent = alt
    # objective cell + lens
    cell = sb.lathe(name + "Cell", [(0.0475, 0.0), (0.0475, 0.03), (0.0415, 0.03), (0.0415, 0.026), (0.044, 0.0)], segs=128, mat=graphite)
    cell.rotation_euler = (-math.pi / 2, 0, 0)
    cell.location = (0, 0.36, 0)
    cell.parent = alt
    lens_m = sb.mat(name + "Objective", (0.02, 0.025, 0.03), rough=0.02, spec=1.0, coat=1.0, coat_rough=0.0, thin_film=380)
    lens = sb.prim("sphere", name + "Lens", segments=96, ring_count=48, radius=0.26, mat=lens_m)
    # a shallow cap: scale to a disc-like dome
    lens.scale = (0.041 / 0.26, 0.004 / 0.26, 0.041 / 0.26)
    lens.location = (0, 0.385, 0)
    lens.parent = alt
    # tube rings + dovetail
    for yy in (-0.05, 0.16):
        ring = sb.prim("torus", name + "Ring", loc=(0, yy, 0), rot=(math.pi / 2, 0, 0), major_radius=0.05, minor_radius=0.006, mat=graphite, parent=alt)
        ring.scale = (1, 1, 1.6)
    dove = sb.prim("cube", name + "Dove", loc=(0.0, 0.055, -0.056), scale=(0.018, 0.16, 0.008), mat=alu, parent=alt)
    sb.bevel(dove, 0.003)
    # focuser at the back (-Y end)
    fh = sb.prim("cyl", name + "FocHouse", loc=(0, -0.24, 0), rot=(math.pi / 2, 0, 0), vertices=64, radius=0.05, depth=0.03, mat=graphite, parent=alt)
    sb.bevel(fh, 0.004)
    draw = sb.prim("cyl", name + "Draw", loc=(0, -0.30, 0), rot=(math.pi / 2, 0, 0), vertices=64, radius=0.031, depth=0.1, mat=alu, parent=alt)
    pin = sb.prim("cube", name + "Pinion", loc=(0, -0.27, -0.042), scale=(0.035, 0.02, 0.014), mat=graphite, parent=alt)
    sb.bevel(pin, 0.004)
    knobs = []
    for sx in (-1, 1):
        k = sb.prim("cyl", name + "Knob", loc=(sx * 0.052, -0.27, -0.042), rot=(0, math.pi / 2, 0), vertices=48, radius=0.017, depth=0.018, mat=knurl, parent=alt)
        sb.bevel(k, 0.002)
        kr = sb.prim("cyl", name + "KnobRing", loc=(sx * 0.0625, -0.27, -0.042), rot=(0, math.pi / 2, 0), vertices=48, radius=0.0172, depth=0.003, mat=amber, parent=alt)
        knobs.append(k)
    # star diagonal + eyepiece (points up toward the observer)
    diag = sb.prim("cube", name + "Diag", loc=(0, -0.37, 0), scale=(0.022, 0.022, 0.022), mat=graphite, parent=alt)
    sb.bevel(diag, 0.005)
    ep_dir = sb.empty(name + "EPdir", loc=(0, -0.37, 0.0), parent=alt)
    ep_dir.rotation_euler = (math.radians(ep_tilt), 0, 0)
    ep = sb.prim("cyl", name + "EP", loc=(0, 0, 0.055), vertices=48, radius=0.017, depth=0.07, mat=graphite, parent=ep_dir)
    epc = sb.prim("cyl", name + "EPcup", loc=(0, 0, 0.095), vertices=48, radius=0.019, depth=0.012, mat=sb.rubber(name + "Cup"), parent=ep_dir)
    epr = sb.prim("cyl", name + "EPring", loc=(0, 0, 0.03), vertices=48, radius=0.0175, depth=0.004, mat=amber, parent=ep_dir)
    return dict(root=root, alt=alt, az=az, lens=lens, rim=rim, knobs=knobs, eyepiece=epc, shield=shield, ep_dir=ep_dir)


def objective_world(sc_obj):
    """World-space centre of the objective and tube axis (unit, pointing out to the sky)."""
    bpy.context.view_layer.update()
    lens = sc_obj["lens"]
    alt = sc_obj["alt"]
    c = lens.matrix_world.translation.copy()
    axis = (alt.matrix_world.to_3x3() @ V((0, 1, 0))).normalized()
    return c, axis
