"""The family's home at dusk (s09 table; reusable for s17's drawing corner): a warm, lived-in dining room + open
kitchen on an upper floor, a wide window onto the twilight city.

Layout (metres, floor z = 0):
  table: oak, 1.9 x 0.95, top at TABLE_Z, centred at the origin, long axis X
  window wall: x = -2.6 (floor-to-ceiling glazing with mullions) -> dusk city (skyline2) + interior reflections
  kitchen: counter run along y = +2.1 (base + wall cabinets, tiled backsplash, open shelves, fridge, plants)
  back wall: y = -2.2 with a bookcase, framed drawings, a doorway
  lights: two warm pendants over the table, candles on the table, under-cabinet strips, the cool window sky."""
import bpy, math, random
from mathutils import Vector as V, Matrix, Euler
import sb, rocket

TABLE_Z = 0.75
_M = {}


def oak(name="Oak", base=(0.42, 0.27, 0.15)):
    m = sb.mat(name, base, rough=0.38, coat=0.35, coat_rough=0.25)
    nb = sb.NB(m)
    co = nb.coord('Object')
    rings = nb.wave(nb.mapping(co, scale=(0.25, 1.0, 1.0)), scale=9.0, dist=7.0, detail=6, wtype='RINGS', direction='Y')
    fine = nb.noise(nb.mapping(co, scale=(0.08, 1.0, 1.0)), scale=260, detail=5, rough=0.6)
    g = nb.math('ADD', nb.math('MULTIPLY', rings.outputs['Fac'], 0.7), nb.math('MULTIPLY', fine.outputs['Fac'], 0.3))
    c = nb.mix(nb.maprange(g, 0.2, 0.9), (base[0] * 0.62, base[1] * 0.55, base[2] * 0.5, 1), (base[0] * 1.2, base[1] * 1.15, base[2] * 1.05, 1))
    wear = nb.maprange(nb.noise(co, scale=2.0, detail=4).outputs['Fac'], 0.55, 0.75, 0, 0.3)
    c = nb.mix(wear, c, (base[0] * 1.3, base[1] * 1.25, base[2] * 1.15, 1))
    nb.set('Base Color', c)
    nb.set('Roughness', nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.3, 0.46))
    nb.set('Normal', nb.bump(g, strength=0.06, distance=0.0008))
    return m


def linen(name, color):
    return sb.fabric(name, color, rough=0.9, weave=1400, sheen=0.35, fuzz=0.35)


def ceramic(name, color, rough=0.12):
    m = sb.mat(name, color, rough=rough, coat=0.4, coat_rough=0.05, spec=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=40, detail=4)
    nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.4, 0.7), (*color, 1),
                                (color[0] * 0.9, color[1] * 0.88, color[2] * 0.85, 1)))
    return m


def tile_mat(name="Backsplash"):
    """Hand-made glazed tiles (7.5 x 15 cm, running bond) with grout lines and per-tile tone/wobble."""
    m = sb.mat(name, (0.74, 0.72, 0.66), rough=0.1, coat=0.6, coat_rough=0.05)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    u = nb.math('DIVIDE', sep.outputs[0], 0.15)
    v = nb.math('DIVIDE', sep.outputs[2], 0.075)
    row = nb.math('FLOOR', v)
    u2 = nb.math('ADD', u, nb.math('MULTIPLY', nb.math('MODULO', row, 2.0), 0.5))
    fu, fv = nb.math('FRACT', u2), nb.math('FRACT', v)
    gu = nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', fu, 0.5)), 0.47)
    gv = nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', fv, 0.5)), 0.44)
    tile = nb.math('MULTIPLY', gu, gv)
    cm = nb.new('ShaderNodeCombineXYZ'); nb.link(nb.math('FLOOR', u2), cm.inputs[0]); nb.link(row, cm.inputs[1])
    wn = nb.new('ShaderNodeTexWhiteNoise'); nb.link(cm.outputs[0], wn.inputs['Vector'])
    tc = nb.mix(nb.maprange(wn.outputs['Value'], 0, 1, 0, 1), (0.66, 0.66, 0.60, 1), (0.80, 0.78, 0.71, 1))
    nb.set('Base Color', nb.mix(tile, (0.35, 0.34, 0.32, 1), tc))
    wob = nb.noise(co, scale=40, detail=3)
    nb.set('Normal', nb.bump(nb.math('ADD', tile, nb.math('MULTIPLY', wob.outputs['Fac'], 0.3)), strength=0.25, distance=0.002))
    return m


def materials():
    if _M:
        return _M
    _M["oak"] = oak()
    _M["oak_dark"] = oak("OakDark", (0.22, 0.14, 0.08))
    _M["floor"] = oak("Floorboards", (0.36, 0.25, 0.16))
    _M["wall"] = sb.concrete("LimePlaster", (0.70, 0.66, 0.60), scale=1.2, rough=0.92, stains=0.1)
    _M["cab"] = sb.painted("CabinetPaint", (0.24, 0.28, 0.25), rough=0.35, coat=0.2, grime=0.05, wear=0.1, scale=3.0)
    _M["counter"] = sb.concrete("Terrazzo", (0.62, 0.60, 0.56), scale=30.0, rough=0.3, stains=0.05)
    _M["tile"] = tile_mat()
    _M["steel"] = sb.brushed_metal("HomeSteel", (0.7, 0.7, 0.71), rough=0.2)
    _M["brass"] = sb.brushed_metal("Brass", (0.78, 0.56, 0.28), rough=0.25, scale=60)
    _M["black"] = sb.mat("HomeBlack", (0.02, 0.02, 0.022), rough=0.4)
    _M["white_cer"] = ceramic("PlateWhite", (0.86, 0.84, 0.80))
    _M["blue_cer"] = ceramic("BowlBlue", (0.16, 0.24, 0.34), 0.15)
    _M["terra"] = ceramic("Terracotta", (0.55, 0.28, 0.16), 0.45)
    _M["glass"] = sb.glass("DrinkGlass", (0.98, 0.99, 1.0), rough=0.0, ior=1.5)
    _M["water"] = sb.glass("Water", (0.96, 0.98, 1.0), rough=0.0, ior=1.33)
    _M["cloth"] = linen("Runner", (0.62, 0.55, 0.46))
    _M["napkin"] = linen("Napkin", (0.55, 0.30, 0.16))
    _M["bread"] = bread_mat()
    _M["leaf"] = sb.mat("SaladLeaf", (0.12, 0.30, 0.06), rough=0.45, sss=0.2, sss_radius=(0.4, 1.0, 0.2))
    _M["leaf2"] = sb.mat("SaladLeaf2", (0.22, 0.42, 0.08), rough=0.45, sss=0.25, sss_radius=(0.4, 1.0, 0.2))
    _M["tomato"] = sb.mat("Tomato", (0.55, 0.05, 0.02), rough=0.2, sss=0.3, coat=0.4)
    _M["candle"] = sb.mat("Wax", (0.86, 0.82, 0.72), rough=0.5, sss=0.6, sss_radius=(1.0, 0.8, 0.5))
    _M["flame"] = flame_mat()
    _M["bulb"] = sb.emit_mat("PendantBulb", (1.0, 0.72, 0.42), 30.0)
    _M["shade"] = sb.painted("ShadeCopper", (0.55, 0.32, 0.18), rough=0.3, coat=0.4, scale=10.0)
    _M["book"] = [sb.mat("Book%d" % i, c, rough=0.6) for i, c in enumerate(
        [(0.35, 0.1, 0.06), (0.1, 0.18, 0.25), (0.6, 0.55, 0.45), (0.12, 0.2, 0.12), (0.5, 0.35, 0.15), (0.2, 0.2, 0.22)])]
    _M["leafplant"] = sb.mat("PlantLeaf2", (0.05, 0.13, 0.04), rough=0.5, sss=0.15)
    _M["paper"] = sb.mat("DrawPaper", (0.86, 0.84, 0.78), rough=0.85)
    _M["window_frame"] = sb.painted("WinFrame", (0.08, 0.08, 0.085), rough=0.4, grime=0.1, wear=0.1)
    _M["pane"] = sb.glass("Pane", (0.97, 0.98, 1.0), rough=0.0, thin=True)
    for m in list(_M.values()):
        if isinstance(m, list):
            for x in m:
                x.use_fake_user = True
        else:
            m.use_fake_user = True
    return _M


def grain_mat():
    if "grain" in _M:
        return _M["grain"]
    m = sb.mat("Grains", (0.62, 0.50, 0.30), rough=0.55, sss=0.2)
    nb = sb.NB(m)
    co = nb.coord('Object')
    v = nb.voronoi(co, scale=180, feature='F1')
    nb.set('Normal', nb.bump(nb.maprange(v.outputs['Distance'], 0.0, 0.5, 1.0, 0.0), strength=0.6, distance=0.002))
    nb.set('Base Color', nb.mix(nb.maprange(v.outputs['Distance'], 0.1, 0.5), (0.72, 0.60, 0.38, 1), (0.48, 0.36, 0.2, 1)))
    m.use_fake_user = True
    _M["grain"] = m
    return m


def flame_mat():
    """Candle flame: hot white-yellow core, orange envelope toward the tip (by object Z), slight transparency."""
    m = bpy.data.materials.new("Flame")
    m.use_nodes = True
    nt = m.node_tree
    nb = sb.NB(m)
    nt.nodes.remove(nb.bsdf)
    co = nb.coord('Generated')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    col = nb.ramp(sep.outputs[2], [(0.0, (0.35, 0.45, 1.0)), (0.18, (1.0, 0.92, 0.75)), (0.6, (1.0, 0.62, 0.22)), (1.0, (0.9, 0.3, 0.05))])
    em = nb.new('ShaderNodeEmission'); nb.link(col, em.inputs[0]); em.inputs[1].default_value = 120.0
    tr = nb.new('ShaderNodeBsdfTransparent')
    mx = nb.new('ShaderNodeMixShader'); mx.inputs[0].default_value = 0.85
    nb.link(tr.outputs[0], mx.inputs[1]); nb.link(em.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], nb.out.inputs[0])
    return m


def bread_mat():
    m = sb.mat("Bread", (0.55, 0.32, 0.13), rough=0.7, sss=0.1)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=30, detail=8, rough=0.7)
    nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.35, 0.7), (0.30, 0.14, 0.05, 1), (0.66, 0.42, 0.18, 1)))
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.4, distance=0.004))
    return m


def box(name, c, s, mat, parent=None, bev=0.004, rot=(0, 0, 0)):
    return rocket.box(name, c, s, mat, parent=parent, bev=bev, rot=rot)


def cyl(name, c, r, h, mat, parent=None, rot=(0, 0, 0), verts=32, bev=0.0):
    o = sb.prim("cyl", name, loc=c, rot=rot, vertices=verts, radius=r, depth=h, mat=mat, parent=parent)
    if bev:
        sb.bevel(o, bev, 2)
    return o


# --------------------------------------------------------------------------- tableware

def plate(name, c, M, r=0.13, parent=None):
    p = [(0.0, 0.0), (r * 0.62, 0.0), (r * 0.64, 0.004), (r * 0.72, 0.006), (r * 0.98, 0.016), (r, 0.018), (r * 0.97, 0.019),
         (r * 0.7, 0.009), (0.0, 0.009)]
    o = sb.lathe(name, p, segs=64, mat=M["white_cer"], axis='Z')
    o.location = c
    if parent:
        o.parent = parent
    return o


def bowl(name, c, M, r=0.11, h=0.07, mat="blue_cer", parent=None):
    p = [(0.0, 0.0), (r * 0.45, 0.0), (r * 0.47, 0.006), (r * 0.8, h * 0.4), (r, h), (r * 0.95, h + 0.002), (r * 0.76, h * 0.4),
         (r * 0.42, 0.012), (0.0, 0.012)]
    o = sb.lathe(name, p, segs=64, mat=M[mat], axis='Z')
    o.location = c
    if parent:
        o.parent = parent
    return o


def leaf_mesh(name, L=0.03, W=0.018, ruffle=0.25, seed=0):
    """Curled, ruffled salad leaf (grid bent along its length, wavy edges)."""
    rs = random.Random(seed)
    nu, nv = 9, 12
    verts, faces = [], []
    for j in range(nv + 1):
        v = j / nv
        half = W * 0.5 * math.sin(math.pi * min(1.0, v * 1.15)) ** 0.7
        for i in range(nu + 1):
            u = i / nu * 2 - 1
            x = u * half
            y = v * L
            z = 0.25 * W * u * u + 0.12 * L * math.sin(math.pi * v) + ruffle * W * 0.25 * math.sin(u * 7 + v * 11 + seed) * abs(u)
            verts.append((x, y, z))
    for j in range(nv):
        for i in range(nu):
            a = j * (nu + 1) + i
            faces.append((a, a + 1, a + nu + 2, a + nu + 1))
    return verts, faces


def salad(name, c, M, r=0.09, n=40, seed=3, leaf=0.03):
    rs = random.Random(seed)
    lv, lf_ = leaf_mesh(name + "LeafMesh", L=leaf * 1.6, W=leaf, seed=seed)
    me = None
    for i in range(n):
        a = rs.uniform(0, 2 * math.pi); d = rs.uniform(0, r) ** 0.8 * r ** 0.2
        p = V(c) + V((math.cos(a) * d, math.sin(a) * d, 0.045 + rs.uniform(0, 0.022) + (r - d) * 0.35))
        if rs.random() < 0.14:
            t = sb.prim("sphere", name + "Tomato", loc=p, radius=leaf * 0.5, segments=16, ring_count=8, mat=M["tomato"])
            t.scale = (1.0, 1.0, 0.55)
            t.rotation_euler = (rs.uniform(-0.4, 0.4), rs.uniform(-0.4, 0.4), 0)
            continue
        if me is None:
            o = sb.mesh_obj(name + "Leaf", lv, lf_, M["leaf"] if rs.random() > 0.3 else M.get("leaf2", M["leaf"]))
            sb.solidify(o, 0.0006)
            me = o.data
        else:
            o = bpy.data.objects.new(name + "Leaf", me); sb.link_obj(o)
        o.location = p
        s_ = rs.uniform(0.7, 1.25)
        o.scale = (s_, s_, s_)
        o.rotation_euler = (rs.uniform(-0.6, 0.6), rs.uniform(-0.6, 0.6), a + rs.uniform(-1, 1))


def glass(name, c, M, r=0.036, h=0.11, fill=0.62, parent=None):
    """Tumbler with a thick base + water (separate volume slightly inset, meniscus implied)."""
    t = 0.0022
    p = [(0.0, 0.0), (r * 0.92, 0.0), (r, 0.004), (r * 1.02, h), (r * 1.02 - t, h), (r - t, 0.012), (0.0, 0.012)]
    o = sb.lathe(name, p, segs=96, mat=M["glass"], axis='Z')
    o.location = c
    w = None
    if fill > 0:
        wh = 0.012 + (h - 0.012) * fill
        # the water overlaps the glass wall by 0.15 mm (no air film between them -> no false total internal reflection)
        w = sb.lathe(name + "Water", [(0.0, 0.0119), (r - t + 0.00015, 0.0119), (r - t + 0.00015 + (0.02 * t) * wh / h, wh), (0.0, wh)],
                     segs=96, mat=M["water"], axis='Z')
        w.location = c
    if parent:
        o.parent = parent
        if w:
            w.parent = parent
    return o, w


def candle(name, c, M, h=0.16, r=0.012):
    cyl(name + "Holder", V(c) + V((0, 0, 0.01)), r * 2.2, 0.02, M["brass"], verts=32, bev=0.002)
    cyl(name + "Wax", V(c) + V((0, 0, 0.02 + h / 2)), r, h, M["candle"], verts=24)
    tip = V(c) + V((0, 0, 0.02 + h + 0.014))
    fl = sb.prim("sphere", name + "Flame", loc=tip, radius=0.006, segments=16, ring_count=8, mat=M["flame"])
    fl.scale = (0.6, 0.6, 1.8)
    fl.visible_shadow = False
    L = sb.light('POINT', name + "Light", loc=tip + V((0, 0, 0.005)), energy=2.2, color=(1.0, 0.58, 0.25), size=0.008)
    # the wick + the soft blue base of the flame
    cyl(name + "Wick", V(c) + V((0, 0, 0.02 + h + 0.004)), 0.0008, 0.008, M["black"], verts=6)
    return fl, L


def chair(name, loc, rz, M, parent=None):
    g = sb.empty(name, loc=loc, parent=parent)
    g.rotation_euler = (0, 0, rz)
    box(name + "Seat", (0, 0, 0.45), (0.44, 0.42, 0.03), M["oak_dark"], parent=g, bev=0.008)
    for sx in (-0.19, 0.19):
        for sy in (-0.18, 0.18):
            leg = cyl(name + "Leg", (sx, sy, 0.22), 0.016, 0.44, M["oak_dark"], parent=g, verts=16)
    for sx in (-0.19, 0.19):
        cyl(name + "Back", (sx, 0.19, 0.68), 0.014, 0.46, M["oak_dark"], parent=g, verts=16)
    box(name + "BackRail", (0, 0.19, 0.86), (0.42, 0.025, 0.07), M["oak_dark"], parent=g, bev=0.006)
    return g


# --------------------------------------------------------------------------- the room

def build(city=True):
    M = materials()
    rs = random.Random(5)
    Z = TABLE_Z
    # table
    box("TableTop", (0, 0, Z - 0.02), (1.9, 0.95, 0.04), M["oak"], bev=0.006)
    for sx in (-0.85, 0.85):
        for sy in (-0.4, 0.4):
            box("TableLeg", (sx, sy, (Z - 0.04) / 2), (0.06, 0.06, Z - 0.04), M["oak"], bev=0.004)
    box("Apron", (0, 0.42, Z - 0.08), (1.7, 0.02, 0.08), M["oak"], bev=0.003)
    box("Apron2", (0, -0.42, Z - 0.08), (1.7, 0.02, 0.08), M["oak"], bev=0.003)
    box("Runner", (0, 0, Z + 0.0015), (1.6, 0.34, 0.003), M["cloth"], bev=0.001)
    # room
    sb.prim("plane", "Floor", loc=(0, 0, 0), scale=(6, 6, 1), mat=M["floor"])
    sb.prim("cube", "Ceiling", loc=(0, 0, 2.75), scale=(6, 6, 0.05), mat=M["wall"])
    sb.prim("cube", "BackWall", loc=(0, -2.25, 1.4), scale=(6, 0.05, 1.4), mat=M["wall"])
    sb.prim("cube", "EndWall", loc=(3.2, 0, 1.4), scale=(0.05, 6, 1.4), mat=M["wall"])
    sb.prim("cube", "KitchenWall", loc=(0, 2.45, 1.4), scale=(6, 0.05, 1.4), mat=M["wall"])
    # window wall (x = -2.6): mullions + panes, sill
    wx = -2.6
    for y in (-2.2, -1.1, 0.0, 1.1, 2.2):
        box("Mullion", (wx, y, 1.35), (0.08, 0.06, 2.7), M["window_frame"], bev=0.004)
    for z in (0.06, 2.1, 2.7):
        box("Transom", (wx, 0, z), (0.08, 4.5, 0.06), M["window_frame"], bev=0.004)
    pane = sb.prim("plane", "Pane", loc=(wx + 0.01, 0, 1.35), size=1.0, mat=M["pane"])
    pane.scale = (1, 4.5, 2.7); pane.rotation_euler = (0, math.pi / 2, 0)
    box("Sill", (wx + 0.15, 0, 0.3), (0.3, 4.4, 0.04), M["oak"], bev=0.004)
    # potted plants on the sill + a floor fig
    for i, y in enumerate((-1.6, -0.9, 1.3)):
        cyl("SillPot%d" % i, (wx + 0.15, y, 0.4), 0.07, 0.16, M["terra"], verts=32, bev=0.004)
        for k in range(10):
            a = rs.uniform(0, 6.28)
            lf = sb.prim("sphere", "SillLeaf", loc=(wx + 0.15 + 0.06 * math.cos(a), y + 0.06 * math.sin(a), 0.52 + rs.uniform(0, 0.18)),
                         radius=0.06, segments=10, ring_count=5, mat=M["leafplant"])
            lf.scale = (1.0, 0.4, 0.1); lf.rotation_euler = (rs.uniform(-0.8, 0.8), rs.uniform(-0.8, 0.8), a)
    cyl("FigPot", (wx + 0.5, 1.9, 0.2), 0.2, 0.4, M["terra"], verts=40, bev=0.01)
    for k in range(40):
        a = rs.uniform(0, 6.28); r = rs.uniform(0.1, 0.45)
        lf = sb.prim("sphere", "FigLeaf", loc=(wx + 0.5 + r * math.cos(a), 1.9 + r * math.sin(a), 0.9 + rs.uniform(0, 1.0)),
                     radius=0.09, segments=10, ring_count=5, mat=M["leafplant"])
        lf.scale = (1.0, 0.6, 0.08); lf.rotation_euler = (rs.uniform(-1, 1), rs.uniform(-1, 1), a)
    cyl("FigTrunk", (wx + 0.5, 1.9, 0.9), 0.02, 1.2, M["oak_dark"], verts=10)
    # kitchen run along y = +2.1
    ky = 2.1
    box("BaseCab", (0.5, ky, 0.44), (3.4, 0.6, 0.88), M["cab"], bev=0.006)
    for i in range(7):
        box("CabDoor", (-1.0 + i * 0.5, ky - 0.305, 0.44), (0.47, 0.012, 0.76), M["cab"], bev=0.004)
        box("CabPull", (-1.0 + i * 0.5, ky - 0.32, 0.72), (0.14, 0.015, 0.012), M["brass"], bev=0.003)
    box("Counter", (0.5, ky - 0.02, 0.9), (3.5, 0.66, 0.04), M["counter"], bev=0.004)
    splash = box("Backsplash", (0.5, 2.41, 1.2), (3.5, 0.02, 0.56), M["tile"], bev=0.0)
    box("WallCab", (1.2, 2.24, 2.05), (2.0, 0.36, 0.7), M["cab"], bev=0.006)
    for i in range(4):
        box("WallCabDoor", (0.45 + i * 0.5, 2.055, 2.05), (0.47, 0.012, 0.66), M["cab"], bev=0.003)
    # open shelves with jars/plates/cookbooks
    for z in (1.65, 2.05):
        box("Shelf", (-0.6, 2.3, z), (1.0, 0.24, 0.03), M["oak"], bev=0.003)
        x = -1.05
        while x < -0.15:
            k = rs.random()
            if k < 0.4:
                cyl("Jar", (x, 2.3, z + 0.08), 0.045, 0.14, M["glass"], verts=24)
                cyl("JarLid", (x, 2.3, z + 0.155), 0.047, 0.012, M["brass"], verts=24)
                x += 0.11
            elif k < 0.7:
                bowl("ShelfBowl", (x + 0.05, 2.3, z + 0.015), M, r=0.07, h=0.05, mat=rs.choice(["white_cer", "blue_cer", "terra"]))
                x += 0.16
            else:
                for j in range(rs.randint(3, 6)):
                    box("Cookbook", (x, 2.3, z + 0.12), (0.03, 0.2, 0.24 - rs.random() * 0.05), rs.choice(M["book"]), bev=0.002)
                    x += 0.032
                x += 0.05
    # fridge + range hood + stove top + a kettle and fruit bowl on the counter
    box("Fridge", (2.55, 2.1, 1.0), (0.7, 0.65, 2.0), M["steel"], bev=0.01)
    box("FridgeHandle", (2.25, 1.76, 1.1), (0.02, 0.03, 0.6), M["black"], bev=0.004)
    box("Hood", (-0.2, 2.25, 2.1), (0.7, 0.4, 0.2), M["steel"], bev=0.01)
    box("Cooktop", (-0.2, 2.1, 0.925), (0.6, 0.5, 0.01), M["black"], bev=0.002)
    kt = sb.lathe("Kettle", [(0.0, 0.0), (0.08, 0.0), (0.095, 0.05), (0.09, 0.15), (0.05, 0.2), (0.012, 0.21), (0.0, 0.21)], segs=48, mat=M["steel"])
    kt.location = (-0.3, 2.05, 0.93)
    bowl("FruitBowl", (0.9, 2.0, 0.92), M, r=0.14, h=0.07, mat="white_cer")
    for i in range(6):
        a = i * 1.1
        sb.prim("sphere", "Fruit", loc=(0.9 + 0.06 * math.cos(a), 2.0 + 0.06 * math.sin(a), 0.99 + 0.02 * (i % 2)), radius=0.035,
                segments=16, ring_count=8, mat=rs.choice([M["tomato"], sb.mat("Orange", (0.8, 0.35, 0.05), rough=0.4, sss=0.2)]))
    # under-cabinet warm strip
    sb.light('AREA', "UnderCab", loc=(1.2, 2.15, 1.68), target=(1.2, 2.05, 0.9), energy=25.0, color=(1.0, 0.78, 0.5), size=1.9)
    # back wall: bookcase + framed children's drawings + doorway glow
    box("Bookcase", (1.2, -2.05, 1.0), (1.4, 0.32, 2.0), M["oak_dark"], bev=0.006)
    for zi in range(5):
        z = 0.12 + zi * 0.4
        x = 0.58
        while x < 1.82:
            w = rs.uniform(0.025, 0.05)
            h = rs.uniform(0.22, 0.32)
            b = box("Book", (x, -1.93, z + h / 2 + 0.01), (w, 0.2, h), rs.choice(M["book"]), bev=0.002)
            if rs.random() < 0.1:
                b.rotation_euler = (0, rs.uniform(0.2, 0.35), 0)
            x += w + 0.004
    for i, (x, z, w, h) in enumerate(((-0.9, 1.55, 0.42, 0.32), (-0.35, 1.7, 0.3, 0.4), (-0.35, 1.2, 0.3, 0.24))):
        box("Frame%d" % i, (x, -2.19, z), (w, 0.02, h), M["black"], bev=0.003)
        box("Drawing%d" % i, (x, -2.178, z), (w - 0.05, 0.005, h - 0.05), child_drawing_mat(i), bev=0.0)
    box("Doorway", (2.5, -2.22, 1.05), (0.9, 0.02, 2.1), sb.emit_mat("Hall", (1.0, 0.75, 0.5), 0.6), bev=0.0)
    # pendants over the table (warm)
    for x in (-0.45, 0.45):
        cyl("Cord", (x, 0, 2.35), 0.003, 0.8, M["black"], verts=8)
        sh = sb.lathe("Shade", [(0.0, 0.0), (0.02, 0.0), (0.2, -0.16), (0.205, -0.17)], segs=48, mat=M["shade"])
        sh.location = (x, 0, 1.95)
        sb.solidify(sh, 0.002)
        sb.prim("sphere", "Bulb", loc=(x, 0, 1.83), radius=0.04, segments=16, ring_count=8, mat=M["bulb"])
        L = sb.light('POINT', "Pendant", loc=(x, 0, 1.8), energy=35.0, color=(1.0, 0.72, 0.45), size=0.06)
    # soft bounce so shadows are not black (warm ceiling bounce)
    sb.light('AREA', "Bounce", loc=(0, 0, 2.6), target=(0, 0, 0), energy=30.0, color=(1.0, 0.82, 0.62), size=2.5)
    if city:
        import skyline2
        skyline2.build(center=(0, 0), rmin=26, rmax=900, count=420, ground_z=-14.0, view_dir=math.pi, view_half=math.radians(80))
    # dusk sky: deep blue with a warm band low on the horizon behind the city
    sb.world_bluehour(glow_az=185.0, glow=5.0, glow_col=(1.0, 0.42, 0.16), zenith=(0.03, 0.06, 0.18), horizon=(0.2, 0.2, 0.3),
                      strength=1.0)
    return M


def child_drawing_mat(seed):
    """A child's crayon drawing: a rocket and a big circle planet / stars (abstract strokes, no text)."""
    m = sb.mat("ChildDrawing%d" % seed, (0.88, 0.86, 0.8), rough=0.85)
    nb = sb.NB(m)
    co = nb.coord('Object')
    rings = nb.wave(nb.mapping(co, loc=(0.02 * seed, 0.01, 0)), scale=8 + seed * 3, dist=4, detail=2, wtype='RINGS', direction='Y')
    l = nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', rings.outputs['Fac'], 0.5)), 0.05)
    mask = nb.maprange(nb.noise(co, scale=6, detail=2).outputs['Fac'], 0.45, 0.55)
    ink = nb.math('MULTIPLY', l, mask)
    col = [(0.75, 0.3, 0.05, 1), (0.1, 0.25, 0.55, 1), (0.2, 0.45, 0.15, 1)][seed % 3]
    nb.set('Base Color', nb.mix(ink, (0.88, 0.86, 0.8, 1), col))
    return m


def set_table(M, seats):
    """Place settings in front of each seat (seat = (x, y, facing_rad)), shared dishes down the middle."""
    Z = TABLE_Z
    for i, (x, y, rz) in enumerate(seats):
        d = V((math.cos(rz), math.sin(rz), 0))          # toward the table centre
        p = V((x, y, Z)) + d * 0.28
        plate("Plate%d" % i, p + V((0, 0, 0.002)), M)
        side = V((-d.y, d.x, 0))
        box("Napkin%d" % i, p - side * 0.2 + V((0, 0, 0.004)), (0.1, 0.14, 0.006), M["napkin"], bev=0.002, rot=(0, 0, rz))
        cutl = box("Fork%d" % i, p - side * 0.16 + V((0, 0, 0.009)), (0.018, 0.19, 0.003), M["steel"], bev=0.001, rot=(0, 0, rz + math.pi / 2))
        # food on plates: a small salad (layered leaves, tomato halves), a slice of bread, a spoon of grains
        salad("PlateSalad%d" % i, p + V((0.025, 0.01, -0.035)), M, r=0.045, n=22, seed=20 + i, leaf=0.018)
        prof = [(-0.036, 0.0), (0.036, 0.0), (0.038, 0.03)] + [(0.034 * math.cos(math.radians(a)), 0.03 + 0.024 * math.sin(math.radians(a)))
                                                               for a in range(10, 171, 20)] + [(-0.038, 0.03)]
        sl = rocket.extrude("PlateBread%d" % i, prof, 0.012, p + V((-0.05, -0.02, 0.016)), V((1, 0, 0)), V((0, 1, 0)), M["bread"], None,
                            bevel=0.003)
        sl.rotation_euler = (0.0, 0.0, rz + 0.5)
        grains = sb.prim("ico", "Grains%d" % i, loc=p + V((0.0, -0.05, 0.012)), subdivisions=3, radius=0.03, mat=grain_mat())
        grains.scale = (1.3, 1.0, 0.28)
    # shared: salad bowl, bread board, water jug, candles
    bowl("SaladBowl", (0.25, 0.05, Z + 0.003), M, r=0.14, h=0.09, mat="white_cer")
    salad("Salad", (0.25, 0.05, Z + 0.003), M, r=0.11)
    box("BreadBoard", (-0.35, 0.0, Z + 0.012), (0.36, 0.2, 0.02), M["oak_dark"], bev=0.004)
    loaf = sb.prim("sphere", "Loaf", loc=(-0.38, 0.0, Z + 0.06), radius=0.1, segments=32, ring_count=16, mat=M["bread"])
    loaf.scale = (1.4, 0.75, 0.5)
    for k in range(3):
        sl = box("Slice%d" % k, (-0.2 + k * 0.03, -0.02, Z + 0.05), (0.015, 0.1, 0.07), M["bread"], bev=0.006, rot=(0, -0.3, 0))
    jug = sb.lathe("Jug", [(0.0, 0.0), (0.06, 0.0), (0.065, 0.1), (0.05, 0.2), (0.055, 0.24), (0.052, 0.24), (0.047, 0.2), (0.06, 0.1),
                           (0.055, 0.004), (0.0, 0.004)], segs=64, mat=M["glass"])
    jug.location = (0.62, -0.08, Z)
    candle("Candle1", (-0.05, 0.1, Z), M)
    candle("Candle2", (0.05, 0.13, Z), M, h=0.12)
