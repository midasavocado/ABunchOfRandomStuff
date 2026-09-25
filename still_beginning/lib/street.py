"""s15 A FUTURE PEOPLE LIVE IN - a tree-lined city street at golden hour (the city from s14, arrived at).

Street frame (metres): the street runs along +X; centre line y = 0; ground z = 0.
  y in [-2, 2]      grass-bed tram track (two tracks, rails at +-0.72 / 1.435 m gauge each side of y=+-1.0)
  y in [2, 4] / [-4, -2]   traffic-calmed lanes (pavers) - bikes, a delivery cargo-bike, no cars
  y in [4, 5.2] / [-5.2, -4]   bike lanes (red-brown asphalt)
  y in [5.2, 11] / [-11, -5.2]  sidewalks (stone flags) with street trees, benches, lamps, cafe terraces
  facades at |y| = 11, 5-7 storeys, real front geometry; far city beyond the street end (+X) in haze.
No text, no logos, no brand marks."""
import bpy, math, random
from mathutils import Vector as V, Matrix, Euler
import sb, rocket
import city as C

_M = {}
FLOOR = 3.2
GROUND = 4.4
FACADE_Y = 11.0


# --------------------------------------------------------------------------- materials

def flags(name="Flags", col=(0.56, 0.54, 0.50), size=(0.6, 0.6)):
    """Stone flags with joints, per-flag tone, wear and a little moss in the joints."""
    m = sb.mat(name, col, rough=0.75)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    u = nb.math('DIVIDE', sep.outputs[0], size[0]); v = nb.math('DIVIDE', sep.outputs[1], size[1])
    row = nb.math('FLOOR', v)
    u = nb.math('ADD', u, nb.math('MULTIPLY', nb.math('MODULO', row, 2.0), 0.5))
    fu, fv = nb.math('FRACT', u), nb.math('FRACT', v)
    j = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fu, 0.5)), 0.49, 0.47, 0.0, 1.0),
                nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fv, 0.5)), 0.49, 0.47, 0.0, 1.0))
    cm = nb.new('ShaderNodeCombineXYZ'); nb.link(nb.math('FLOOR', u), cm.inputs[0]); nb.link(row, cm.inputs[1])
    wn = nb.new('ShaderNodeTexWhiteNoise'); nb.link(cm.outputs[0], wn.inputs['Vector'])
    grain = nb.noise(co, scale=40, detail=8, rough=0.7)
    stone = nb.mix(nb.maprange(wn.outputs['Value'], 0, 1, 0, 1), (col[0] * 0.85, col[1] * 0.85, col[2] * 0.84, 1),
                   (col[0] * 1.1, col[1] * 1.08, col[2] * 1.06, 1))
    stone = nb.mix(nb.maprange(grain.outputs['Fac'], 0.3, 0.7, 0, 0.3), stone, (col[0] * 0.7, col[1] * 0.7, col[2] * 0.68, 1))
    joint = (0.12, 0.13, 0.08, 1)
    nb.set('Base Color', nb.mix(j, joint, stone))
    nb.set('Roughness', nb.mix(j, 0.95, 0.7, dtype='FLOAT'))
    nb.set('Normal', nb.bump(nb.math('ADD', j, nb.math('MULTIPLY', grain.outputs['Fac'], 0.15)), strength=0.5, distance=0.006))
    return m


def asphalt(name, col=(0.28, 0.15, 0.10)):
    m = sb.mat(name, col, rough=0.85)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=180, detail=6, rough=0.7)
    v = nb.voronoi(co, scale=300)
    nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (col[0] * 0.8, col[1] * 0.8, col[2] * 0.8, 1), (col[0] * 1.15, col[1] * 1.12, col[2] * 1.1, 1)))
    nb.set('Normal', nb.bump(nb.maprange(v.outputs['Distance'], 0.0, 0.5, 1.0, 0.0), strength=0.3, distance=0.002))
    return m


def window_glass(name="WinGlass2"):
    """Street glazing: dark, sharply reflective glass with a faint uniform warm interior glow (no blotches); curtains
    and room depth are implied by a soft vertical falloff."""
    m = sb.mat(name, (0.015, 0.018, 0.022), rough=0.02, spec=0.9)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    fz = nb.math('FRACT', nb.math('DIVIDE', sep.outputs[2], 3.2))
    nb.set('Emission Color', (1.0, 0.74, 0.48, 1))
    nb.set('Emission Strength', nb.maprange(fz, 0.1, 0.9, 0.05, 0.015))
    return m


def materials():
    if _M:
        return _M
    _M["flags"] = flags()
    _M["pavers"] = flags("Pavers", (0.46, 0.42, 0.38), size=(0.2, 0.1))
    _M["bike"] = asphalt("BikeLane")
    _M["curb"] = sb.concrete("Curb", (0.62, 0.61, 0.58), scale=5.0, rough=0.7, stains=0.25)
    _M["rail"] = sb.brushed_metal("RailSteel", (0.55, 0.54, 0.52), rough=0.3, aniso=0.5, scale=40, scratches=0.4)
    _M["grass"] = sb.mat("TramGrass", (0.08, 0.16, 0.04), rough=0.9)
    _M["frame"] = sb.painted("WinFrameWarm", (0.84, 0.82, 0.77), rough=0.5, grime=0.2, wear=0.05)
    _M["frame_dark"] = sb.painted("WinFrameDark", (0.10, 0.10, 0.11), rough=0.45, grime=0.15)
    _M["glass"] = window_glass()
    _M["stone"] = sb.concrete("StoneBase", (0.66, 0.64, 0.60), scale=2.0, rough=0.75, stains=0.3)
    _M["metal"] = sb.painted("StreetMetal", (0.13, 0.14, 0.14), rough=0.45, grime=0.3, wear=0.3, scale=6.0)
    _M["wood"] = sb.mat("BenchWood", (0.42, 0.28, 0.16), rough=0.6)
    _M["awning"] = [sb.fabric("Awning%d" % i, c, weave=600, sheen=0.3) for i, c in
                    enumerate([(0.55, 0.22, 0.08), (0.20, 0.30, 0.26), (0.72, 0.66, 0.55), (0.36, 0.18, 0.12)])]
    _M["pot"] = sb.mat("Planter", (0.36, 0.20, 0.12), rough=0.7)
    _M["leaf"] = sb.mat("BalconyLeaf", (0.06, 0.15, 0.04), rough=0.55, sss=0.15)
    _M["flower"] = [sb.mat("Flower%d" % i, c, rough=0.5, sss=0.2) for i, c in
                    enumerate([(0.7, 0.2, 0.25), (0.85, 0.65, 0.2), (0.85, 0.85, 0.8), (0.5, 0.25, 0.6)])]
    _M["lamp"] = sb.emit_mat("StreetLampGlow", (1.0, 0.78, 0.5), 6.0)
    _M["table"] = sb.brushed_metal("CafeTable", (0.7, 0.7, 0.7), rough=0.3)
    _M["cup"] = sb.mat("CafeCup", (0.9, 0.88, 0.84), rough=0.1, coat=0.5)
    _M["umbrella"] = sb.fabric("Umbrella", (0.86, 0.80, 0.68), weave=500)
    for m in list(_M.values()):
        for x in (m if isinstance(m, list) else [m]):
            x.use_fake_user = True
    return _M


def box(name, c, s, mat, parent=None, bev=0.01, rot=(0, 0, 0)):
    return rocket.box(name, c, s, mat, parent=parent, bev=bev, rot=rot)


# --------------------------------------------------------------------------- street-front building

class Acc:
    """Accumulate many boxes into one mesh per material (fast scenes with thousands of facade parts)."""
    def __init__(self):
        self.d = {}

    def box(self, mat, c, s, rz=0.0):
        v, f = self.d.setdefault(mat.name, ([], [], mat))[:2]
        cx, cy, cz = c
        sx, sy, sz = s[0] / 2, s[1] / 2, s[2] / 2
        cs, sn = math.cos(rz), math.sin(rz)
        b = len(v)
        for dz in (-sz, sz):
            for dx, dy in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)):
                v.append((cx + dx * cs - dy * sn, cy + dx * sn + dy * cs, cz + dz))
        f += [(b, b + 3, b + 2, b + 1), (b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5),
              (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)]

    def build(self, name, bevel=0.015):
        out = []
        for k, (v, f, mat) in self.d.items():
            o = sb.mesh_obj(name + "_" + k, v, f, mat, smooth_shade=False)
            if bevel:
                md = o.modifiers.new("Bev", 'BEVEL'); md.width = bevel; md.segments = 1; md.limit_method = 'ANGLE'
                md.angle_limit = math.radians(30)
            out.append(o)
        return out


def building(A, x0, width, floors, side, M, rs, wall_mat, awning_i=0, shop=True):
    """Street-front building at x in [x0, x0+width] on the given side (+1: y>0 side facing -Y; -1: y<0 facing +Y).
    Front: stone base with shopfront glazing, piers + spandrels with recessed windows (frames, mullion, sill),
    balconies with railings and planters on alternate bays, cornice + parapet. Body behind uses facade_mat."""
    s = side
    yf = FACADE_Y * s                       # facade plane
    H = GROUND + floors * FLOOR
    depth = 14.0
    # body (behind the facade layer), sides get the procedural window grid
    body = sb.prim("cube", "Body", loc=(x0 + width / 2, yf + s * (depth / 2 + 0.25), H / 2), scale=(width / 2 - 0.02, depth / 2, H / 2), mat=wall_mat)
    bay = rs.choice((2.6, 2.8, 3.0))
    n = max(2, int(width / bay))
    bay = width / n
    win_w, win_h = bay * rs.uniform(0.44, 0.52), FLOOR * rs.uniform(0.52, 0.6)
    fy = yf + s * 0.12                     # window recess plane
    # wall layer: piers between windows and spandrels between floors (solid pieces, openings left open)
    for fl in range(floors):
        z0 = GROUND + fl * FLOOR
        sill_z = z0 + 0.9
        top_z = sill_z + win_h
        A.box(wall_mat, (x0 + width / 2, yf, (z0 + sill_z) / 2), (width, 0.3, sill_z - z0))
        A.box(wall_mat, (x0 + width / 2, yf, (top_z + z0 + FLOOR) / 2), (width, 0.3, z0 + FLOOR - top_z))
        for i in range(n + 1):
            px = x0 + i * bay
            w = (bay - win_w) / (1 if 0 < i < n else 2)
            cx = px if 0 < i < n else (px + w / 2 if i == 0 else px - w / 2)
            A.box(wall_mat, (cx, yf, (sill_z + top_z) / 2), (w, 0.3, top_z - sill_z))
        for i in range(n):
            cx = x0 + (i + 0.5) * bay
            # window unit: frame, glass, mullion, sill, lintel
            A.box(M["glass"], (cx, fy + s * 0.02, (sill_z + top_z) / 2), (win_w - 0.02, 0.02, win_h - 0.02))
            for dz in (sill_z + 0.03, top_z - 0.03):
                A.box(M["frame"], (cx, fy - s * 0.01, dz), (win_w, 0.07, 0.06))
            for dx in (-win_w / 2 + 0.03, win_w / 2 - 0.03, 0.0):
                A.box(M["frame"], (cx + dx, fy - s * 0.01, (sill_z + top_z) / 2), (0.06 if dx else 0.045, 0.07, win_h))
            A.box(M["frame"], (cx, fy - s * 0.01, sill_z + win_h * 0.66), (win_w, 0.06, 0.045))
            A.box(M["stone"], (cx, yf - s * 0.2, sill_z - 0.04), (win_w + 0.16, 0.14, 0.07))
            A.box(M["stone"], (cx, yf - s * 0.16, top_z + 0.09), (win_w + 0.2, 0.06, 0.18))
            # balcony on some bays (floors 1..)
            if fl >= 1 and (i + fl) % 2 == 0 and rs.random() < 0.8:
                bz = z0 + 0.02
                A.box(M["stone"], (cx, yf - s * 0.55, bz), (win_w + 0.9, 1.1, 0.14))
                for k in range(9):
                    A.box(M["metal"], (cx - (win_w + 0.8) / 2 + k * (win_w + 0.8) / 8, yf - s * 1.07, bz + 0.52), (0.022, 0.022, 0.96))
                A.box(M["metal"], (cx, yf - s * 1.07, bz + 1.0), (win_w + 0.84, 0.05, 0.04))
                for sgn in (-1, 1):
                    A.box(M["metal"], (cx + sgn * (win_w + 0.82) / 2, yf - s * 0.6, bz + 1.0), (0.04, 1.0, 0.04))
                # planter + plants trailing over the rail
                A.box(M["pot"], (cx, yf - s * 0.95, bz + 0.2), (win_w + 0.6, 0.22, 0.26))
                for k in range(rs.randint(5, 10)):
                    p = V((cx + rs.uniform(-(win_w + 0.5) / 2, (win_w + 0.5) / 2), yf - s * 0.95, bz + 0.38 + rs.uniform(0, 0.25)))
                    lf = sb.prim("ico", "BalPlant", loc=p, subdivisions=2, radius=rs.uniform(0.14, 0.26),
                                 mat=M["leaf"] if rs.random() > 0.25 else rs.choice(M["flower"]))
                    lf.scale = (1.0, 0.8, rs.uniform(0.7, 1.4))
    # ground floor: stone base, shopfront glazing between stone piers, awning
    A.box(M["stone"], (x0 + width / 2, yf - s * 0.05, GROUND / 2), (width, 0.4, GROUND))  # will be hidden behind glass in openings
    for i in range(n):
        cx = x0 + (i + 0.5) * bay
        gw = bay - 0.5
        A.box(M["glass"], (cx, yf - s * 0.27, 1.95), (gw, 0.03, 3.1))
        A.box(M["frame_dark"], (cx, yf - s * 0.29, 0.25), (gw, 0.06, 0.12))
        A.box(M["frame_dark"], (cx, yf - s * 0.29, 3.52), (gw, 0.06, 0.1))
        A.box(M["frame_dark"], (cx, yf - s * 0.29, 1.95), (0.05, 0.06, 3.1))
        A.box(M["stone"], (x0 + i * bay, yf - s * 0.3, GROUND / 2), (0.5, 0.2, GROUND))
    A.box(M["stone"], (x0 + width, yf - s * 0.3, GROUND / 2), (0.5, 0.2, GROUND))
    if shop:
        aw = M["awning"][awning_i % len(M["awning"])]
        a = box("Awning", (x0 + width / 2, yf - s * 1.05, 3.72), (width - 0.6, 1.6, 0.05), aw, bev=0.01, rot=(math.radians(14) * s, 0, 0))
        a2 = box("AwningVal", (x0 + width / 2, yf - s * 1.84, 3.5), (width - 0.6, 0.02, 0.28), aw, bev=0.005)
    # cornice + parapet + roof greenery / solar
    A.box(M["stone"], (x0 + width / 2, yf - s * 0.15, H + 0.12), (width, 0.6, 0.24))
    A.box(wall_mat, (x0 + width / 2, yf + s * 0.2, H + 0.6), (width, 0.3, 0.8))
    if rs.random() < 0.5:
        for k in range(int(width / 2.0)):
            p = V((x0 + 1 + k * 2.0, yf + s * rs.uniform(3, 10), H + 1.4))
            t = sb.prim("ico", "RoofTree", loc=p, subdivisions=2, radius=rs.uniform(0.8, 1.5), mat=M["leaf"])
            t.scale = (1, 1, 1.2)
    return body


def street_tree(lib, loc, rs):
    import land
    return land.place_trees(lib, [loc], scale_rng=(0.85, 1.15), seed=rs.randint(0, 999), name="StreetTree")


def lamp(name, loc, side, M):
    x, y = loc
    rocket.rod(name + "Pole", (x, y, 0), (x, y, 6.5), 0.06, M["metal"], verts=16)
    rocket.rod(name + "Arm", (x, y, 6.4), (x, y - side * 1.6, 6.9), 0.04, M["metal"], verts=12)
    hd = box(name + "Head", (x, y - side * 1.7, 6.85), (0.6, 0.25, 0.1), M["metal"], bev=0.02)
    lens = sb.prim("plane", name + "Lens", loc=(x, y - side * 1.7, 6.79), size=1.0, mat=M["lamp"])
    lens.scale = (0.27, 0.1, 1); lens.rotation_euler = (math.pi, 0, 0)


def bench(name, loc, rz, M):
    g = sb.empty(name, loc=loc)
    g.rotation_euler = (0, 0, rz)
    for k in range(5):
        box(name + "Slat", (0, -0.2 + k * 0.1, 0.45), (1.8, 0.08, 0.04), M["wood"], parent=g, bev=0.008)
    for k in range(3):
        box(name + "Back", (0, 0.26, 0.6 + k * 0.12), (1.8, 0.04, 0.08), M["wood"], parent=g, bev=0.008)
    for sx in (-0.75, 0.75):
        box(name + "Leg", (sx, 0.0, 0.22), (0.06, 0.5, 0.44), M["metal"], parent=g, bev=0.01)
    return g


def cafe_set(name, loc, M, rs):
    x, y = loc
    rocket.rod(name + "TLeg", (x, y, 0), (x, y, 0.74), 0.025, M["table"], verts=12)
    t = sb.prim("cyl", name + "Top", loc=(x, y, 0.75), vertices=40, radius=0.33, depth=0.02, mat=M["table"])
    for k in range(rs.randint(1, 3)):
        a = rs.uniform(0, 6.28)
        c = sb.lathe(name + "Cup", [(0, 0), (0.035, 0), (0.042, 0.08), (0.039, 0.08), (0.032, 0.006), (0, 0.006)], segs=24, mat=M["cup"])
        c.location = (x + 0.15 * math.cos(a), y + 0.15 * math.sin(a), 0.76)
    return t


def tram(name, M, length=32.0, sections=4):
    """Low-floor tram: white/graphite with a restrained amber line (same family as the s14 train), large windows with
    warm interior, articulated sections, pantograph on the roof. Returns root empty (tram runs along +X, origin at
    its front bottom centre, y=0 on its track centre)."""
    g = sb.empty(name)
    white = sb.painted("TramWhite", (0.82, 0.82, 0.80), rough=0.3, coat=0.6, grime=0.1, wear=0.02, scale=0.5)
    dark = sb.painted("TramDark", (0.06, 0.065, 0.07), rough=0.35, coat=0.5)
    amber = sb.painted("TramAmber", (0.75, 0.32, 0.04), rough=0.3, coat=0.6)
    win = sb.mat("TramWin", (0.02, 0.025, 0.03), rough=0.03, spec=0.8)
    nb = sb.NB(win); nb.set('Emission Color', (1.0, 0.82, 0.62, 1)); nb.set('Emission Strength', 0.25)
    L = length / sections
    for i in range(sections):
        x = -(i + 0.5) * L
        box(name + "Car%d" % i, (x, 0, 1.75), (L - 0.35, 2.65, 3.0), white, parent=g, bev=0.12)
        box(name + "Skirt%d" % i, (x, 0, 0.45), (L - 0.35, 2.6, 0.6), dark, parent=g, bev=0.05)
        box(name + "Line%d" % i, (x, 0, 1.08), (L - 0.3, 2.67, 0.06), amber, parent=g, bev=0.01)
        box(name + "Win%d" % i, (x, 0, 2.15), (L - 1.2, 2.68, 1.3), win, parent=g, bev=0.05)
        box(name + "Roof%d" % i, (x, 0, 3.35), (L - 1.0, 1.8, 0.3), white, parent=g, bev=0.08)
        if i < sections - 1:
            box(name + "Bellows%d" % i, (x - L / 2, 0, 1.75), (0.5, 2.4, 2.8), dark, parent=g, bev=0.08)
    # rounded nose with a big windscreen
    nose = sb.prim("sphere", name + "Nose", loc=(-0.6, 0, 1.9), radius=1.35, segments=32, ring_count=16, mat=white, parent=g)
    nose.scale = (0.7, 0.98, 1.2)
    ws = sb.prim("sphere", name + "Screen", loc=(-0.55, 0, 2.2), radius=1.3, segments=32, ring_count=16, mat=win, parent=g)
    ws.scale = (0.68, 0.9, 0.62)
    # pantograph
    p = V((-L * 1.5, 0, 3.55))
    rocket.rod(name + "Panto1", p, p + V((1.0, 0, 0.9)), 0.03, dark, parent=g)
    rocket.rod(name + "Panto2", p + V((1.0, 0, 0.9)), p + V((0.1, 0, 1.6)), 0.025, dark, parent=g)
    box(name + "PantoHead", p + V((0.1, 0, 1.62)), (0.25, 1.6, 0.05), dark, parent=g, bev=0.01)
    return g
