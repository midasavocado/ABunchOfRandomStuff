"""s13 ABUNDANCE: a glancing reflection across solar-module glass (cells, busbars, frame) opens in one continuous
drone move into a wide view of an agrivoltaic installation over crops on rolling farmland at golden hour.
Fixed-tilt rows (25 deg, facing south), all identical orientation; nothing tracks or changes between frames."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb
import numpy as np
import energy as E
import land as L
import wind as W
import timeline as TL
from mathutils import Vector as V, Matrix, Quaternion

sid = (sb.argv() or ["s13"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=float(os.environ.get("EXPO", "0.0")))
rnd = random.Random(13)

SUN_EL, SUN_AZ = float(os.environ.get("SUNEL", "15")), float(os.environ.get("SUNAZ", "248"))
MW, MH = 1.134, 2.278            # module (portrait): width across the row, length up the slope
TILT = math.radians(25.0)
NCOL = 12                         # modules per table along the row
TW = NCOL * (MW + 0.02)           # table width (m)
ROWP = 11.0                       # row pitch (agrivoltaic: tractor lanes between rows)
LOW = 3.1                         # lower-edge clearance above the crop
HAZE = float(os.environ.get("HAZED", "4200"))

# ---- sky / sun
sb.world_sky(elev=SUN_EL, azim=SUN_AZ, strength=float(os.environ.get("SKY", "0.13")), sun_disc=False, aerosol=1.8)
sun = sb.sun(SUN_EL, SUN_AZ, energy=float(os.environ.get("SUNE", "4.6")), color=(1.0, 0.80, 0.56), angle=0.55)
W.HAZE_COL = (0.78, 0.72, 0.64)


def module_mat():
    """Mono-PERC half-cut module under AR glass: 6 x 24 half-cells, 12 wire busbars per cell, white backsheet gaps,
    silver frame border, glass coat with faint prismatic texture and dust gradient toward the lower edge.
    UV: u across (0..1 per module, any integer range for tables), v up the slope."""
    m = sb.mat("Module", (0.012, 0.016, 0.032), metal=0.0, rough=0.35, coat=1.0, coat_rough=0.025)
    nb = sb.NB(m)
    b = nb.bsdf
    b.inputs['Coat IOR'].default_value = 1.52
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    U, Vv = sep.outputs[0], sep.outputs[1]
    fu = nb.math('FRACT', U); fv = nb.math('FRACT', Vv)
    xm = nb.math('MULTIPLY', fu, MW); ym = nb.math('MULTIPLY', fv, MH)            # metres inside the module
    fr = 0.032
    frame = nb.math('MAXIMUM', nb.math('MAXIMUM', nb.math('LESS_THAN', xm, fr), nb.math('GREATER_THAN', xm, MW - fr)),
                    nb.math('MAXIMUM', nb.math('LESS_THAN', ym, fr), nb.math('GREATER_THAN', ym, MH - fr)))
    # cell grid (6 across, 12 half-cells in each of two blocks with a 20 mm centre gap)
    cu = nb.math('DIVIDE', nb.math('SUBTRACT', xm, fr + 0.004), (MW - 2 * fr - 0.008) / 6.0)
    cuf = nb.math('FRACT', cu)
    yb = nb.math('SUBTRACT', ym, fr + 0.004)
    half = (MH - 2 * fr - 0.008 - 0.02) / 2.0
    yb2 = nb.math('SUBTRACT', yb, nb.math('MULTIPLY', nb.math('GREATER_THAN', yb, half), 0.02))
    cv = nb.math('DIVIDE', yb2, half / 12.0)
    cvf = nb.math('FRACT', cv)
    gap_u = nb.math('MAXIMUM', nb.math('LESS_THAN', cuf, 0.011), nb.math('GREATER_THAN', cuf, 0.989))
    gap_v = nb.math('MAXIMUM', nb.math('LESS_THAN', cvf, 0.022), nb.math('GREATER_THAN', cvf, 0.978))
    centre = nb.math('MULTIPLY', nb.math('GREATER_THAN', yb, half - 0.001), nb.math('LESS_THAN', yb, half + 0.021))
    gap = nb.math('MINIMUM', nb.math('MAXIMUM', nb.math('MAXIMUM', gap_u, gap_v), centre), 1.0)
    # busbar wires: 12 per cell, running along v
    bb = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('MULTIPLY', cuf, 12.0)), 0.5))
    bus = nb.maprange(bb, 0.0, 0.045, 1.0, 0.0)
    # per-cell tone variation
    cid = nb.new('ShaderNodeCombineXYZ')
    nb.link(nb.math('FLOOR', nb.math('ADD', cu, nb.math('MULTIPLY', nb.math('FLOOR', U), 7.0))), cid.inputs[0])
    nb.link(nb.math('FLOOR', nb.math('ADD', cv, nb.math('MULTIPLY', nb.math('FLOOR', Vv), 13.0))), cid.inputs[1])
    wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'; nb.link(cid.outputs[0], wn.inputs['Vector'])
    cell_col = nb.mix(nb.maprange(wn.outputs['Value'], 0, 1, 0, 1), (0.010, 0.013, 0.026, 1), (0.018, 0.024, 0.048, 1))
    col = nb.mix(bus, cell_col, (0.55, 0.56, 0.58, 1))
    col = nb.mix(gap, col, (0.62, 0.63, 0.64, 1))
    col = nb.mix(frame, col, (0.60, 0.61, 0.63, 1))
    nb.set('Base Color', col)
    nb.set('Metallic', nb.math('MAXIMUM', nb.math('MULTIPLY', bus, 0.9), nb.math('MULTIPLY', frame, 0.85)))
    nb.set('Roughness', nb.mix(frame, nb.mix(bus, 0.4, 0.25, dtype='FLOAT'), 0.22, dtype='FLOAT'))
    # glass: dust toward the lower edge + noise, prismatic micro texture on the coat normal
    co = nb.coord('Object')
    dust_n = nb.noise(co, scale=3.0, detail=6, rough=0.6)
    dust = nb.math('ADD', nb.maprange(fv, 0.0, 0.25, 0.08, 0.0), nb.maprange(dust_n.outputs['Fac'], 0.45, 0.75, 0.0, 0.05))
    nb.set('Coat Roughness', nb.math('ADD', 0.02, dust))
    nb.set('Coat Weight', nb.math('SUBTRACT', 1.0, frame))
    pr = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 1.0)), scale=260.0, detail=2, rough=0.4)
    cb = nb.new('ShaderNodeBump'); cb.inputs['Strength'].default_value = 0.03; cb.inputs['Distance'].default_value = 0.0005
    nb.link(pr.outputs['Fac'], cb.inputs['Height'])
    nb.link(cb.outputs[0], b.inputs['Coat Normal'])
    # frame lip: slight bevel shading
    fb = nb.new('ShaderNodeBump'); fb.inputs['Strength'].default_value = 0.4; fb.inputs['Distance'].default_value = 0.004
    nb.link(frame, fb.inputs['Height'])
    nb.set('Normal', fb.outputs[0])
    return m


def table_mesh(name, mod_m, steel, lod=False):
    """One fixed-tilt table: NCOL x 2 modules in portrait on purlins, rafters, 2 posts (+ braces).
    Local frame: row along X, modules face -Y (south) tilted TILT; origin at ground under the table centre."""
    parts = []
    slope = 2 * MH + 0.02
    dy = math.cos(TILT) * slope; dz = math.sin(TILT) * slope
    # lower edge at y = -dy/2 (south), z = LOW; upper edge at y = +dy/2, z = LOW + dz
    y0, z0 = -dy / 2, LOW
    up = V((0, math.cos(TILT), math.sin(TILT)))
    nrm = V((0, -math.sin(TILT), math.cos(TILT)))
    verts, faces, uvs = [], [], []
    if lod:
        c0 = V((-TW / 2, y0, z0)); c1 = V((TW / 2, y0, z0)); c2 = c1 + up * slope; c3 = c0 + up * slope
        verts = [tuple(c0), tuple(c1), tuple(c2), tuple(c3)]
        faces = [(0, 1, 2, 3)]
        uvs = [(0, 0), (NCOL, 0), (NCOL, 2), (0, 2)]
        parts.append(E.mesh_np(name + "Mods", verts, np.array(faces), mod_m, np.array(uvs), smooth=False))
    else:
        th = 0.035
        for r in range(2):
            for c in range(NCOL):
                x0 = -TW / 2 + c * (MW + 0.02)
                o = V((x0, y0, z0)) + up * (r * (MH + 0.02))
                a = o; bq = o + V((MW, 0, 0)); cq = bq + up * MH; d = a + up * MH
                base = len(verts)
                for p in (a, bq, cq, d):
                    verts.append(tuple(p))
                for p in (a, bq, cq, d):
                    verts.append(tuple(p - nrm * th))
                faces += [(base, base + 1, base + 2, base + 3), (base + 7, base + 6, base + 5, base + 4),
                          (base, base + 4, base + 5, base + 1), (base + 1, base + 5, base + 6, base + 2),
                          (base + 2, base + 6, base + 7, base + 3), (base + 3, base + 7, base + 4, base)]
                uvs += [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 1), (1, 1), (1, 0)]
                uvs += [(0, 0), (0, 0.01), (0.01, 0.01), (0.01, 0)] * 4
        o = E.mesh_np(name + "Mods", verts, np.array(faces), mod_m, np.array(uvs), smooth=False)
        parts.append(o)
        # purlins (along X, under the modules) and rafters (up the slope) - galvanised C sections
        for fr_ in (0.22, 0.78):
            for r in range(2):
                p = V((0, y0, z0)) + up * (r * (MH + 0.02) + MH * fr_) - nrm * 0.07
                pl = E.box(name + f"Pur{r}{fr_}", (TW + 0.2, 0.05, 0.08), loc=tuple(p), mat=steel)
                pl.rotation_euler = (TILT, 0, 0)
                parts.append(pl)
    # posts + rafters + braces (both LODs; LOD without braces)
    for px in (-TW / 2 + 1.6, TW / 2 - 1.6):
        pz_top = LOW + dz * 0.45
        py = y0 + dy * 0.45
        post = E.box(name + f"Post{px}", (0.14, 0.12, pz_top + 0.8), loc=(px, py, (pz_top - 0.8) / 2), mat=steel)
        parts.append(post)
        if not lod:
            raf = E.box(name + f"Raf{px}", (0.1, slope * 0.95, 0.12), loc=tuple(V((px, 0, 0)) + V((0, y0, z0)) + up * slope / 2 - nrm * 0.16), mat=steel)
            raf.rotation_euler = (TILT, 0, 0)
            parts.append(raf)
            for s_ in (-1, 1):
                br = E.sweep(name + f"Br{px}{s_}", [(px, py, LOW * 0.55), (px, py + s_ * dy * 0.38, LOW + (dz * (0.45 + 0.38 * s_)) - 0.25)],
                             radius=0.035, segs=6, mat=steel, sub=1)
                parts.append(br)
    o = sb.join(parts, name)
    return o


def lod_near(x, y, cam_path_pts, r=220.0):
    return min((V((x, y, 0)) - V((p.x, p.y, 0))).length for p in cam_path_pts) < r


# ---- materials
mod_m = module_mat()
steel = E.painted_steel("Galv", (0.52, 0.53, 0.54), rough=0.42, wear=0.25, scale=0.8)
field = L.field_mat("Field", rows_dir=(1.0, 0.0), pitch=0.75)
for mm in (mod_m, steel, field):
    W.aerial(mm, dist=HAZE, power=1.0)
# ---- terrain
ter = L.terrain("Terrain", L.hills, -4200, 4200, -4200, 4200, 420, 420, field)
farm = sb.mat("FarLand", (0.12, 0.13, 0.06), rough=0.9)
W.aerial(farm, dist=HAZE)
fg = sb.prim("plane", "FarGround", loc=(0, 0, -25.0), scale=(300000, 300000, 1), mat=farm)
bpy.ops.object.select_all(action='DESELECT')
# ---- tables (full detail near the camera path, LOD elsewhere)
src_full = table_mesh("TableFull", mod_m, steel, lod=False)
src_lod = table_mesh("TableLod", mod_m, steel, lod=True)
for o in (src_full, src_lod):
    o.hide_render = True; o.hide_viewport = True

CAM_PTS = [V((6, -2, 6)), V((60, -80, 40)), V((110, -150, 60))]
count = 0
for j in range(-27, 28):
    y = j * ROWP
    for i in range(-27, 27):
        x = i * (TW + 0.5) + (j % 2) * 0.0
        if abs(x) > 385 or abs(y) > 300:
            continue
        # leave a service track every 9th row gap and a field lane
        z = float(L.hills(np.array(x), np.array(y)))
        z1 = float(L.hills(np.array(x + TW / 2), np.array(y))); z0_ = float(L.hills(np.array(x - TW / 2), np.array(y)))
        roll = math.atan2(z1 - z0_, TW)
        src = src_full if lod_near(x, y, CAM_PTS) else src_lod
        o = bpy.data.objects.new(f"Tab{i}_{j}", src.data)
        sb.link_obj(o)
        o.location = (x, y, z)
        o.rotation_euler = (0, -roll, 0)
        count += 1
print("TABLES", count)

# ---- hedgerows, field trees and copses (LOD trees far away)
bark = L.bark_mat("Bark")
leafm = L.leaf_mat("LeafG", col=(0.042, 0.10, 0.022))
W.aerial(bark, dist=HAZE); W.aerial(leafm, dist=HAZE)
lib_near = L.tree_library(3, seed=31, height=13.0, crown_r=5.0, bark=bark, leaf=leafm)
lib_far = L.tree_library(3, seed=61, height=12.0, crown_r=5.0, bark=bark, leaf=leafm, lod=True, leaf_count=4500, leaf_size=0.4)
pts_near, pts_far = [], []
def add_line(xa, ya, xb, yb, step, jitter=3.0):
    n = int(math.hypot(xb - xa, yb - ya) / step)
    for k in range(n):
        u = k / max(1, n - 1)
        x = xa + (xb - xa) * u + rnd.uniform(-jitter, jitter); y = ya + (yb - ya) * u + rnd.uniform(-jitter, jitter)
        if rnd.random() < 0.15:
            continue
        z = float(L.hills(np.array(x), np.array(y))) - 0.2
        (pts_near if math.hypot(x - 60, y + 80) < 420 else pts_far).append(V((x, y, z)))
for yy in (-330.0, 322.0, 900.0, -1200.0):
    add_line(-2600, yy, 2600, yy + rnd.uniform(-60, 60), 11.0)
for xx in (-415.0, 412.0, -1500.0, 1650.0, 2800.0):
    add_line(xx, -2600, xx + rnd.uniform(-80, 80), 2600, 11.0)
for k in range(40):                 # copses on the hills
    cx, cy = rnd.uniform(-3500, 3500), rnd.uniform(-3500, 3500)
    if abs(cx) < 500 and abs(cy) < 420:
        continue
    for m_ in range(rnd.randint(6, 22)):
        x, y = cx + rnd.gauss(0, 25), cy + rnd.gauss(0, 25)
        pts_far.append(V((x, y, float(L.hills(np.array(x), np.array(y))) - 0.2)))
L.place_trees(lib_near, pts_near, scale_rng=(0.8, 1.25), seed=3, name="Hn")
L.place_trees(lib_far, pts_far, scale_rng=(0.8, 1.3), seed=4, name="Hf")
print("TREES", len(pts_near), len(pts_far))

# ---- camera: skim the module glass looking west along the row into the low sun, then rise and pull back
cam = E.dof_cam("Cam", 28, 4.0, clip=(0.02, 400000))
sc.eevee.bokeh_max_size = 150
# the hero module: table (0,0) upper module, centre area
hz = float(L.hills(np.array(0.0), np.array(0.0)))
slope = 2 * MH + 0.02
up = V((0, math.cos(TILT), math.sin(TILT))); nrm = V((0, -math.sin(TILT), math.cos(TILT)))
glass_pt = V((1.2, -math.cos(TILT) * slope / 2, LOW + hz)) + up * (MH * 1.45)
P0 = glass_pt + nrm * 0.26 + V((1.3, 0, 0))
P1 = glass_pt + nrm * 1.6 + V((7.0, -4.0, 2.0))
P2 = V((55.0, -70.0, hz + 32.0))
P3 = V((120.0, -150.0, hz + 60.0))


def timing(t):
    return sb.ease_in(t, 2.2) * 0.75 + sb.smoother(t) * 0.25


def pos(t):
    return sb.catmull([P0, P1, P2, P3], timing(t))


T0 = glass_pt + V((-1.7, 0.0, 0.0)) - nrm * 0.12
T3 = V((-90.0, 150.0, hz - 15.0))


def tgt(t):
    k = sb.smoother(sb.remap(timing(t), 0.0, 0.8))
    return T0.lerp(T3, k)


def foc(t):
    p = pos(t); tg = tgt(t)
    return sb.lerp(1.1, 250.0, sb.smooth(sb.remap(timing(t), 0.03, 0.35)))


sb.cam_bake(cam, F0, F1, pos, tgt, focus=foc, fstop=lambda t: sb.lerp(4.0, 8.0, sb.smooth(sb.remap(timing(t), 0.0, 0.4))))
sb.frames(F0, F1)
sb.render_shot(sid)
