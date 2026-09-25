"""The child's rooftop at blue hour: an ordinary city roof, neighbours, skyline with lit windows."""
import bpy, math, random
from mathutils import Vector as V
import sb


def window_building_mat(name, seed, wall=(0.05, 0.05, 0.055), lit_frac=0.35, warm=(1.0, 0.62, 0.30), cool=(0.62, 0.72, 0.95),
                        floor_h=3.2, bay_w=2.6, strength=3.0):
    """Facade material: procedural window grid, random lit windows (object coords in metres)."""
    m = sb.mat(name, wall, rough=0.8)
    nb = sb.NB(m)
    co = nb.new('ShaderNodeNewGeometry').outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    # horizontal coordinate along facade: use x + y so all four faces get windows
    hcoord = nb.math('ADD', sep.outputs[0], sep.outputs[1])
    fx = nb.math('FRACT', nb.math('DIVIDE', hcoord, bay_w))
    fz = nb.math('FRACT', nb.math('DIVIDE', sep.outputs[2], floor_h))
    win = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fx, 0.5)), 0.15, 0.12),
                  nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fz, 0.55)), 0.17, 0.14))
    # per-window random
    cellx = nb.math('FLOOR', nb.math('DIVIDE', hcoord, bay_w))
    cellz = nb.math('FLOOR', nb.math('DIVIDE', sep.outputs[2], floor_h))
    cmb = nb.new('ShaderNodeCombineXYZ')
    nb.link(cellx, cmb.inputs[0]); nb.link(cellz, cmb.inputs[1]); cmb.inputs[2].default_value = seed
    wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'
    nb.link(cmb.outputs[0], wn.inputs['Vector'])
    r = wn.outputs['Value']
    lit = nb.maprange(r, 1 - lit_frac, 1 - lit_frac + 0.001)
    wc = nb.new('ShaderNodeSeparateColor'); nb.link(wn.outputs['Color'], wc.inputs[0])
    col = nb.mix(nb.maprange(wc.outputs[0], 0.7, 0.72), (*warm, 1), (*cool, 1))
    bright = nb.maprange(wc.outputs[1], 0, 1, 0.35, 1.0)
    e = nb.math('MULTIPLY', nb.math('MULTIPLY', win, lit), bright)
    nb.set('Emission Color', col)
    nb.set('Emission Strength', nb.math('MULTIPLY', e, strength))
    # glass for unlit windows (darker, glossy)
    nb.set('Roughness', nb.mix(win, 0.85, 0.12, dtype='FLOAT'))
    base = nb.mix(win, (*wall, 1), (0.015, 0.018, 0.024, 1))
    nb.set('Base Color', base)
    return m


def skyline(center=(0, 0, 0), rmin=25, rmax=420, count=380, seed=11, ground_z=-14.0, avoid=None, low_cone=None):
    rnd = random.Random(seed)
    mats = [window_building_mat(f"Fac{i}", seed + i, wall=(0.03 + 0.012 * i, 0.032 + 0.012 * i, 0.04 + 0.01 * i),
                                lit_frac=0.07 + 0.035 * i, cool=(0.85, 0.72, 0.55)) for i in range(4)]
    objs = []
    for i in range(count):
        a = rnd.uniform(0, 2 * math.pi)
        d = rmin + (rmax - rmin) * rnd.random() ** 1.6
        if avoid and avoid(a, d):
            continue
        w = rnd.uniform(8, 24)
        dd = rnd.uniform(8, 20)
        h = rnd.uniform(6, 30) * (1.8 if rnd.random() < 0.08 else 1.0)
        if d < 70:
            h = min(h, abs(ground_z) - 1.5)
        if low_cone is not None:
            ca, cw, cd = low_cone
            da = math.atan2(math.sin(a - ca), math.cos(a - ca))
            if abs(da) < cw and d < cd:
                h = rnd.uniform(3, 8) + (d / cd) * 13
        c = V((center[0] + math.cos(a) * d, center[1] + math.sin(a) * d, ground_z + h / 2))
        b = sb.prim("cube", f"bld{i}", loc=c, scale=(w / 2, dd / 2, h / 2), mat=mats[i % 4])
        b.rotation_euler = (0, 0, rnd.choice([0, math.pi / 2]) + rnd.uniform(-0.05, 0.05))
        objs.append(b)
        # roof clutter
        if rnd.random() < 0.35:
            t = sb.prim("cyl", f"tank{i}", loc=c + V((rnd.uniform(-w / 4, w / 4), 0, h / 2 + 1.6)), scale=(1.2, 1.2, 1.3), vertices=16, mat=mats[0])
        if rnd.random() < 0.2:
            sb.prim("cyl", f"mast{i}", loc=c + V((0, 0, h / 2 + 3)), scale=(0.06, 0.06, 3), vertices=6, mat=mats[1])
    return objs


def roof_deck(size=18.0):
    """The child's roof: bitumen membrane with gravel, a timber deck patch, parapet with coping."""
    bit = sb.mat("Bitumen", (0.05, 0.05, 0.052), rough=0.85)
    nb = sb.NB(bit)
    co = nb.coord('Object')
    n = nb.noise(co, scale=40, detail=10, rough=0.7)
    v = nb.voronoi(co, scale=220, feature='F1')
    h = nb.math('ADD', n.outputs['Fac'], nb.math('MULTIPLY', nb.maprange(v.outputs['Distance'], 0, 0.5, 1, 0), 0.6))
    nb.set('Normal', nb.bump(h, strength=0.4, distance=0.004))
    nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.4, 0.7), (0.035, 0.036, 0.04, 1), (0.085, 0.084, 0.082, 1)))
    deck = sb.prim("plane", "Roof", loc=(0, 0, 0), scale=(size / 2, size / 2, 1), mat=bit)
    # timber deck boards under the telescope
    wood = sb.mat("Deck", (0.23, 0.15, 0.09), rough=0.6)
    nbw = sb.NB(wood)
    cw = nbw.coord('Object')
    grain = nbw.wave(nbw.mapping(cw, scale=(1, 25, 1)), scale=3, dist=6, detail=4, wtype='BANDS', direction='X')
    nbw.set('Base Color', nbw.mix(nbw.maprange(grain.outputs['Fac'], 0.2, 0.8), (0.14, 0.085, 0.05, 1), (0.30, 0.20, 0.12, 1)))
    nbw.set('Normal', nbw.bump(grain.outputs['Fac'], strength=0.2, distance=0.002))
    for i in range(22):
        b = sb.prim("cube", f"board{i}", loc=(-1.6 + i * 0.145, 0.4, 0.02), scale=(0.066, 1.9, 0.018), mat=wood)
        sb.bevel(b, 0.004, 2)
    # parapet
    conc = sb.concrete("Parapet", (0.36, 0.35, 0.34), scale=2.0, stains=0.5)
    cope = sb.concrete("Coping", (0.46, 0.45, 0.43), scale=3.0, stains=0.3)
    for (x, y, sx, sy) in ((0, 4.2, 6.0, 0.12), (6.0, -1.0, 0.12, 5.2), (-6.0, -1.0, 0.12, 5.2)):
        w = sb.prim("cube", "Wall", loc=(x, y, 0.5), scale=(sx, sy, 0.5), mat=conc)
        c = sb.prim("cube", "Cope", loc=(x, y, 1.03), scale=(sx + 0.05, sy + 0.05, 0.035), mat=cope)
        sb.bevel(c, 0.01, 2)
    return deck


def string_lights(p0, p1, n=14, sag=0.35, strength=12.0, bulb_w=2.2):
    """Warm festoon bulbs between two points (practical amber light on the roof)."""
    wire = sb.mat("Wire", (0.01, 0.01, 0.01), rough=0.5)
    bulb = sb.emit_mat("Bulb", (1.0, 0.55, 0.2), strength)
    pts = []
    for i in range(41):
        t = i / 40
        p = V(p0).lerp(V(p1), t) - V((0, 0, sag * 4 * t * (1 - t)))
        pts.append(p)
    sb.tube_along("Festoon", pts, radius=0.003, mat=wire)
    lights = []
    for i in range(1, n):
        t = i / n
        p = V(p0).lerp(V(p1), t) - V((0, 0, sag * 4 * t * (1 - t)))
        b = sb.prim("sphere", f"bulb{i}", loc=p - V((0, 0, 0.05)), radius=0.025, segments=16, ring_count=8, mat=bulb)
        l = sb.light('POINT', f"bulbL{i}", loc=p - V((0, 0, 0.08)), energy=bulb_w, color=(1.0, 0.55, 0.22), size=0.03)
        lights.append(l)
    return lights


def roof_props(rnd_seed=4):
    rnd = random.Random(rnd_seed)
    metal = sb.brushed_metal("Duct", (0.55, 0.56, 0.58), rough=0.4, scratches=0.4)
    graphite = sb.painted("AC", (0.42, 0.43, 0.44), rough=0.5, grime=0.3)
    # AC unit + vent pipes
    ac = sb.prim("cube", "ACunit", loc=(4.6, 2.9, 0.45), scale=(0.55, 0.45, 0.45), mat=graphite); sb.bevel(ac, 0.02)
    fan = sb.prim("cyl", "ACfan", loc=(4.6, 2.9, 0.905), scale=(0.33, 0.33, 0.01), vertices=48, mat=metal)
    for i, (x, y) in enumerate(((-4.8, 3.4), (-4.2, 3.6), (3.3, 3.7))):
        p = sb.prim("cyl", f"vent{i}", loc=(x, y, 0.35), scale=(0.07, 0.07, 0.35), vertices=24, mat=metal)
        cap = sb.prim("cone", f"ventcap{i}", loc=(x, y, 0.76), scale=(0.12, 0.12, 0.06), vertices=24, mat=metal)
    # potted plants along the parapet: small-leaved shrubs (herbs/olive-like), many leaves on thin stems
    pot = sb.mat("Terracotta", (0.34, 0.15, 0.08), rough=0.85)
    leaf = sb.mat("Leaf", (0.035, 0.07, 0.03), rough=0.5, sss=0.3, sss_radius=(0.3, 1.0, 0.2), sss_scale=0.01)
    stem = sb.mat("Stem", (0.08, 0.06, 0.04), rough=0.7)
    leaf_src = sb.prim("plane", "LeafSrc", scale=(0.012, 0.028, 1), mat=leaf)
    leaf_src.hide_render = True
    for i in range(5):
        x = -3.3 + i * 1.6 + rnd.uniform(-0.2, 0.2)
        y = 3.8
        sb.prim("cyl", f"pot{i}", loc=(x, y, 0.17), scale=(0.17, 0.17, 0.17), vertices=40, mat=pot)
        for k in range(7):
            a0 = rnd.uniform(0, 2 * math.pi)
            top = V((x + math.cos(a0) * rnd.uniform(0.05, 0.22), y + math.sin(a0) * rnd.uniform(0.05, 0.22), 0.55 + rnd.uniform(0, 0.4)))
            sb.tube_along(f"stem{i}_{k}", [V((x, y, 0.32)), (V((x, y, 0.32)) + top) / 2 + V((0, 0, 0.05)), top], radius=0.004, mat=stem)
            for j in range(26):
                t = rnd.uniform(0.35, 1.0)
                p = V((x, y, 0.32)).lerp(top, t) + V((rnd.gauss(0, 0.03), rnd.gauss(0, 0.03), rnd.gauss(0, 0.02)))
                lf = sb.dup(leaf_src, loc=p, rot=(rnd.uniform(-1.0, 1.0), rnd.uniform(-0.6, 0.6), rnd.uniform(0, 6.28)))
                k_ = rnd.uniform(0.7, 1.3)
                lf.scale = (0.012 * k_, 0.028 * k_, 1)
                lf.hide_render = False
    # folding chair + small table with notebook
    alu = sb.brushed_metal("ChairAlu", (0.6, 0.6, 0.62), rough=0.35)
    fab = sb.fabric("ChairFab", (0.12, 0.16, 0.2), weave=260)
    seat = sb.prim("cube", "Seat", loc=(2.6, 1.6, 0.45), scale=(0.22, 0.22, 0.015), mat=fab)
    for sx in (-1, 1):
        for sy in (-1, 1):
            sb.prim("cyl", "ChairLeg", loc=(2.6 + sx * 0.2, 1.6 + sy * 0.2, 0.22), scale=(0.01, 0.01, 0.23), vertices=12, mat=alu)
    back = sb.prim("cube", "Back", loc=(2.6, 1.82, 0.7), scale=(0.22, 0.01, 0.2), mat=fab)
    table = sb.prim("cyl", "SideTable", loc=(1.0, 0.35, 0.34), scale=(0.2, 0.2, 0.012), vertices=48, mat=sb.painted("TableTop", (0.8, 0.78, 0.74), rough=0.4))
    sb.prim("cyl", "TableLeg", loc=(1.0, 0.35, 0.17), scale=(0.015, 0.015, 0.17), vertices=12, mat=alu)
    nbk = sb.prim("cube", "Notebook", loc=(1.02, 0.33, 0.36), scale=(0.075, 0.105, 0.006), mat=sb.fabric("Cover", (0.55, 0.23, 0.06), weave=500))
    nbk.rotation_euler = (0, 0, 0.3)
    torch = sb.prim("cyl", "Torch", loc=(0.93, 0.44, 0.365), rot=(0, math.pi / 2, 0.8), scale=(0.013, 0.013, 0.07), vertices=16, mat=graphite)
