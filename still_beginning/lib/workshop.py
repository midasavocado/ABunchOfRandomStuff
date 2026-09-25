"""The prosthetics workshop for s08 (engineer's bench): a real, lived-in making space so every close-up has depth.

Layout (metres, bench top surface at z = BENCH_Z, engineer sits at -Y facing +Y):
  bench:   butcher-block top x in [-1.0, 1.0], y in [-0.40, 0.40]; self-healing cutting mat at the centre
  wall:    pegboard at y = 0.62 with hanging tools, a shelf of parts bins above it, a whiteboard with sketches
  window:  tall steel-framed window on the -X wall (x = -1.35) -> late-afternoon warm sun raking across the bench
  props:   task lamp (warm practical), second monitor with CAD, parts tray with printed links on their build plate,
           calipers, screwdriver set, tweezers, loupe, mug, notebook, cable reel, soldering station, 3D-printed
           prototypes of earlier iterations (v1/v2) lined up.
Everything is procedural (no text, no logos)."""
import bpy, math, os, random
from mathutils import Vector as V, Matrix, Euler
import sb, rocket

BENCH_Z = 0.92
_M = {}


# --------------------------------------------------------------------------- materials

def wood(name="Butcher", base=(0.46, 0.30, 0.17), strip=0.045):
    """Edge-grain butcher block: parallel strips (x) with per-strip tone, fine grain lines, oiled satin, knife
    marks and darker wear toward the front edge."""
    m = sb.mat(name, base, rough=0.45, coat=0.25, coat_rough=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    sid = nb.math('FLOOR', nb.math('DIVIDE', sep.outputs[1], strip))
    wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '1D'; nb.link(sid, wn.inputs['W'])
    tone = wn.outputs['Value']
    grain = nb.wave(nb.mapping(co, scale=(1.0, 9.0, 1.0)), scale=60.0, dist=6.0, detail=4, wtype='BANDS', direction='X')
    fine = nb.noise(nb.mapping(co, scale=(0.15, 6.0, 1.0)), scale=120.0, detail=6, rough=0.6)
    g = nb.math('ADD', nb.math('MULTIPLY', grain.outputs['Fac'], 0.6), nb.math('MULTIPLY', fine.outputs['Fac'], 0.4))
    dark = (base[0] * 0.55, base[1] * 0.5, base[2] * 0.45, 1)
    light = (base[0] * 1.2, base[1] * 1.15, base[2] * 1.1, 1)
    c = nb.mix(nb.maprange(tone, 0, 1, 0.15, 0.85), dark, light)
    c = nb.mix(nb.maprange(g, 0.3, 0.8, 0.0, 0.35), c, dark)
    # knife marks / scratches (thin, random directions) + stains
    sc1 = nb.noise(nb.mapping(co, scale=(1, 40, 1), rot=(0, 0, 0.5)), scale=30, detail=2)
    sc2 = nb.noise(nb.mapping(co, scale=(40, 1, 1), rot=(0, 0, 0.3)), scale=30, detail=2)
    scr = nb.math('MAXIMUM', nb.maprange(sc1.outputs['Fac'], 0.63, 0.66), nb.maprange(sc2.outputs['Fac'], 0.64, 0.67))
    st = nb.maprange(nb.noise(co, scale=3.0, detail=5).outputs['Fac'], 0.55, 0.75, 0, 0.5)
    c = nb.mix(st, c, (base[0] * 0.4, base[1] * 0.33, base[2] * 0.25, 1))
    c = nb.mix(nb.math('MULTIPLY', scr, 0.5), c, light)
    nb.set('Base Color', c)
    nb.set('Roughness', nb.math('ADD', nb.maprange(g, 0, 1, 0.38, 0.55), nb.math('MULTIPLY', scr, 0.2)))
    nb.set('Normal', nb.bump(nb.math('SUBTRACT', g, nb.math('MULTIPLY', scr, 1.5)), strength=0.12, distance=0.0006))
    return m


def cutting_mat(name="CutMat"):
    """Grey self-healing mat: 10 mm grid of thin lighter lines, cut scars, matte."""
    m = sb.mat(name, (0.11, 0.115, 0.12), rough=0.8, spec=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])

    def lines(x, pitch, w):
        fr = nb.math('FRACT', nb.math('DIVIDE', x, pitch))
        return nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', fr, 0.5)), w)
    gx = nb.math('MAXIMUM', lines(sep.outputs[0], 0.01, 0.03), lines(sep.outputs[1], 0.01, 0.03))
    gX = nb.math('MAXIMUM', lines(sep.outputs[0], 0.05, 0.012), lines(sep.outputs[1], 0.05, 0.012))
    g = nb.math('MAXIMUM', nb.math('MULTIPLY', gx, 0.5), gX)
    cuts = nb.maprange(nb.noise(nb.mapping(co, scale=(1, 30, 1), rot=(0, 0, 0.7)), scale=18, detail=2).outputs['Fac'], 0.64, 0.66)
    c = nb.mix(g, (0.11, 0.115, 0.12, 1), (0.30, 0.31, 0.32, 1))
    c = nb.mix(nb.math('MULTIPLY', cuts, 0.6), c, (0.2, 0.2, 0.21, 1))
    nb.set('Base Color', c)
    nb.set('Normal', nb.bump(nb.math('MULTIPLY', cuts, -1.0), strength=0.3, distance=0.0003))
    return m


def pegboard(name="Pegboard"):
    """Painted steel pegboard (warm grey), 25 mm hole pitch; holes are dark (alpha-free, shading only)."""
    m = sb.painted(name, (0.46, 0.45, 0.43), rough=0.5, grime=0.25, wear=0.1, streaks=0.1, scale=2.0)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    fx = nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[0], 0.025)), 0.5)
    fz = nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[2], 0.025)), 0.5)
    r = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', fx, fx), nb.math('MULTIPLY', fz, fz)))
    hole = nb.math('LESS_THAN', r, 0.14)
    bc = nb.bsdf.inputs['Base Color']
    src = bc.links[0].from_socket if bc.links else (0.46, 0.45, 0.43, 1)
    nb.set('Base Color', nb.mix(hole, src, (0.01, 0.01, 0.01, 1)))
    nb.set('Normal', nb.bump(nb.math('MULTIPLY', hole, -1.0), strength=0.6, distance=0.001))
    return m


def screen_image(name, path, strength=1.0):
    """Monitor showing an image (the s06 CAD viewport, for continuity): emission from the image + glossy glass."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nb = sb.NB(m)
    tex = nb.new('ShaderNodeTexImage')
    if os.path.exists(path):
        tex.image = bpy.data.images.load(path, check_existing=True)
    nb.link(nb.coord('UV'), tex.inputs['Vector'])
    b = nb.bsdf
    b.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1)
    b.inputs['Roughness'].default_value = 0.08
    nb.link(tex.outputs['Color'], b.inputs['Emission Color'])
    b.inputs['Emission Strength'].default_value = strength
    return m


def plastic(name, color, rough=0.35):
    m = sb.mat(name, color, rough=rough, spec=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    nb.set('Normal', nb.bump(nb.noise(co, scale=400, detail=3).outputs['Fac'], strength=0.03, distance=0.0002))
    return m


def materials():
    if _M:
        return _M
    _M["wood"] = wood()
    _M["mat"] = cutting_mat()
    _M["frame"] = sb.painted("BenchFrame", (0.06, 0.062, 0.066), rough=0.45, grime=0.3, wear=0.35, scale=6.0)
    _M["peg"] = pegboard()
    _M["wall"] = sb.concrete("WallPlaster", (0.62, 0.60, 0.56), scale=1.5, rough=0.9, stains=0.15)
    _M["floor"] = sb.concrete("ShopFloor", (0.32, 0.31, 0.30), scale=2.5, rough=0.6, stains=0.4)
    _M["steel"] = sb.brushed_metal("ShopSteel", (0.62, 0.62, 0.63), rough=0.25, scratches=0.3)
    _M["chrome"] = sb.mat("ShopChrome", (0.85, 0.85, 0.86), metal=1.0, rough=0.08)
    _M["black"] = plastic("ShopBlack", (0.02, 0.02, 0.022), 0.4)
    _M["grey"] = plastic("ShopGrey", (0.18, 0.18, 0.19), 0.45)
    _M["amber"] = plastic("ShopAmber", (0.55, 0.20, 0.02), 0.35)
    _M["white"] = plastic("ShopWhite", (0.72, 0.72, 0.70), 0.4)
    _M["sage"] = plastic("ShopSage", (0.28, 0.33, 0.28), 0.5)
    _M["blue"] = plastic("ShopBlueGrey", (0.14, 0.19, 0.26), 0.5)
    _M["ceramic"] = sb.mat("Mug", (0.78, 0.76, 0.72), rough=0.15, coat=0.5)
    _M["paper"] = sb.mat("Paper", (0.82, 0.80, 0.75), rough=0.8, sheen=0.3)
    _M["cork"] = sb.mat("Cork", (0.45, 0.33, 0.2), rough=0.9)
    _M["glass"] = sb.glass("WinGlass", (0.95, 0.97, 1.0), rough=0.02, thin=True)
    _M["lamp"] = sb.emit_mat("LampLED", (1.0, 0.82, 0.62), 8.0)
    _M["screen"] = sb.emit_mat("Screen2", (0.55, 0.6, 0.66), 0.8)
    _M["cad"] = screen_image("CadScreen", os.path.join(sb.ROOT, "assets", "screen_s06", "scr_0539.jpg"), 1.1)
    _M["foam"] = sb.mat("Foam", (0.07, 0.07, 0.075), rough=0.95)
    _M["copper"] = sb.brushed_metal("Copper", (0.85, 0.48, 0.3), rough=0.3, scale=50)
    for m in _M.values():
        m.use_fake_user = True          # MPFB purges orphan materials when it builds people
    return _M


# --------------------------------------------------------------------------- small props

def box(name, c, s, mat, parent=None, bev=0.002, rot=(0, 0, 0)):
    return rocket.box(name, c, s, mat, parent=parent, bev=bev, rot=rot)


def cyl(name, c, r, h, mat, parent=None, rot=(0, 0, 0), verts=32, bev=0.0):
    o = sb.prim("cyl", name, loc=c, rot=rot, vertices=verts, radius=r, depth=h, mat=mat, parent=parent)
    if bev:
        sb.bevel(o, bev, 2)
    return o


def screwdriver(name, p, ang, M, handle="black", L=0.19, parent=None):
    """Precision screwdriver lying on the bench (or hanging when ang is a vector)."""
    g = sb.empty(name, loc=p, parent=parent)
    g.rotation_euler = ang
    hl = L * 0.45
    h = sb.lathe(name + "H", [(0.0, 0), (0.0085, 0.002), (0.0095, 0.02), (0.009, hl * 0.8), (0.0065, hl), (0.0035, hl + 0.004)],
                 segs=24, mat=M[handle], axis='Z')
    h.parent = g
    # knurled grip flutes implied by 6 thin grooves
    s = cyl(name + "S", (0, 0, hl + (L - hl) / 2), 0.0018, L - hl, M["chrome"], parent=g, verts=12)
    return g


def pliers(name, p, rz, M, parent=None, open_=0.2):
    g = sb.empty(name, loc=p, parent=parent)
    g.rotation_euler = (0, 0, rz)
    for s in (-1, 1):
        arm = rocket.extrude(name + "Arm", [(0.0, 0.0), (0.012, 0.004), (0.11, 0.012 + 0.01 * s * 0 + 0.0), (0.115, 0.004), (0.0, -0.004)],
                             0.006, (0, 0, 0.003 + 0.0035 * (s + 1) / 2), V((1, 0, 0)), V((0, 1, 0)), M["steel"], g, bevel=0.0008)
        arm.rotation_euler = (0, 0, s * open_ * 0.5)
        grip = rocket.extrude(name + "Grip", [(0.012, 0.004), (0.11, 0.012), (0.114, 0.006), (0.012, -0.0005)], 0.009,
                              (0, 0, 0.003 + 0.0035 * (s + 1) / 2), V((1, 0, 0)), V((0, 1, 0)), M["amber" if s > 0 else "black"], g, bevel=0.0015)
        grip.rotation_euler = (0, 0, s * open_ * 0.5)
        for o in (arm, grip):
            o.scale = (1, s, 1)
    jaw = rocket.extrude(name + "Jaw", [(0.0, -0.005), (-0.05, -0.001), (-0.05, 0.001), (0.0, 0.005)], 0.007, (0, 0, 0.0055),
                         V((1, 0, 0)), V((0, 1, 0)), M["steel"], g, bevel=0.0008)
    cyl(name + "Rivet", (0, 0, 0.0055), 0.004, 0.009, M["chrome"], parent=g, verts=16)
    return g


def calipers(name, p, rz, M, parent=None):
    g = sb.empty(name, loc=p, parent=parent)
    g.rotation_euler = (0, 0, rz)
    box(name + "Beam", (0.07, 0, 0.0017), (0.19, 0.016, 0.0034), M["steel"], parent=g, bev=0.0004)
    box(name + "Slide", (0.02, 0.0, 0.004), (0.05, 0.03, 0.007), M["steel"], parent=g, bev=0.001)
    box(name + "Disp", (0.02, 0.004, 0.0078), (0.03, 0.013, 0.0012), M["black"], parent=g, bev=0.0003)
    for x, ln in ((-0.025, 0.045), (0.0, 0.04)):
        box(name + "Jaw", (x, -0.022, 0.0017), (0.007, ln, 0.0034), M["steel"], parent=g, bev=0.0004)
    return g


def mug(name, p, M, parent=None):
    g = sb.empty(name, loc=p, parent=parent)
    o = sb.lathe(name + "Body", [(0.0, 0.0), (0.036, 0.0), (0.04, 0.004), (0.041, 0.095), (0.037, 0.095), (0.036, 0.008),
                                 (0.0, 0.008)], segs=64, mat=M["ceramic"], axis='Z')
    o.parent = g
    h = sb.prim("torus", name + "Handle", loc=(0.047, 0, 0.05), rot=(math.pi / 2, 0, 0), major_radius=0.022,
                minor_radius=0.0055, mat=M["ceramic"], parent=g)
    h.scale = (0.8, 1.0, 1.0)
    tea = cyl(name + "Tea", (0, 0, 0.07), 0.0365, 0.001, sb.mat("Coffee", (0.05, 0.025, 0.01), rough=0.05, spec=0.6), parent=g, verts=48)
    return g


def notebook(name, p, rz, M, parent=None, sketch=True):
    g = sb.empty(name, loc=p, parent=parent)
    g.rotation_euler = (0, 0, rz)
    box(name + "Cover", (0, 0, 0.004), (0.21, 0.15, 0.008), M["grey"], parent=g, bev=0.0015)
    pg = box(name + "Pages", (0.003, 0, 0.0085), (0.205, 0.145, 0.002), sketch_paper(), parent=g, bev=0.0004)
    return g


def sketch_paper():
    """Notebook page with pencil sketch strokes of linkage geometry (abstract curves, no text)."""
    if "sketch" in _M:
        return _M["sketch"]
    m = sb.mat("SketchPaper", (0.84, 0.82, 0.77), rough=0.85, sheen=0.2)
    nb = sb.NB(m)
    co = nb.coord('Object')
    w1 = nb.wave(nb.mapping(co, rot=(0, 0, 0.4)), scale=35, dist=9, detail=2, wtype='RINGS', direction='Z', profile='SIN')
    w2 = nb.wave(nb.mapping(co, loc=(0.03, 0.02, 0), rot=(0, 0, -0.3)), scale=22, dist=6, detail=2, wtype='RINGS', direction='Z')
    l1 = nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', w1.outputs['Fac'], 0.5)), 0.012)
    l2 = nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', w2.outputs['Fac'], 0.5)), 0.01)
    mask = nb.maprange(nb.noise(co, scale=14, detail=2).outputs['Fac'], 0.45, 0.6)
    ink = nb.math('MULTIPLY', nb.math('MAXIMUM', l1, l2), mask)
    # ruled grid (faint dots)
    nb.set('Base Color', nb.mix(ink, (0.84, 0.82, 0.77, 1), (0.25, 0.25, 0.27, 1)))
    _M["sketch"] = m
    return m


def bins(name, origin, M, n=8, parent=None, seed=3):
    """Stackable open-front parts bins on a shelf (muted colours), each with a little content."""
    rs = random.Random(seed)
    cols = ["grey", "sage", "blue", "grey", "white", "amber", "grey", "blue"]
    for i in range(n):
        x = origin[0] + i * 0.115
        c = V((x, origin[1], origin[2]))
        prof = [(-0.05, 0.0), (0.05, 0.0), (0.05, 0.085), (-0.03, 0.085), (-0.05, 0.055)]
        rocket.extrude(name + "%d" % i, [(p[1], p[0]) for p in prof], 0.1, c + V((0, 0, 0)), V((0, 0, 1)), V((0, 1, 0)),
                       M[cols[i % len(cols)]], parent, bevel=0.002)
        for k in range(rs.randint(4, 9)):
            q = c + V((rs.uniform(-0.035, 0.035), rs.uniform(-0.03, 0.02), 0.05 + rs.uniform(0, 0.02)))
            o = sb.prim("cyl", name + "Part", loc=q, rot=(rs.uniform(0, 3), rs.uniform(0, 3), 0), vertices=6,
                        radius=rs.uniform(0.004, 0.009), depth=rs.uniform(0.004, 0.02), mat=M[rs.choice(["steel", "copper", "black"])],
                        parent=parent)


def task_lamp(name, base, M, head_target, parent=None, head_off=(0.0, -0.05, 0.28)):
    """Articulated LED task lamp: weighted base, two arms with springs, wide head with warm LED panel + real light."""
    base = V(base)
    g = sb.empty(name, loc=(0, 0, 0), parent=parent)
    cyl(name + "Base", base + V((0, 0, 0.012)), 0.075, 0.024, M["frame"], parent=g, verts=48, bev=0.004)
    j1 = base + V((0, 0, 0.03))
    j2 = j1 + V((0.08, 0.10, 0.36))
    head = V(head_target) + V(head_off)
    for a, b in ((j1, j2), (j2, head)):
        for s in (-1, 1):
            d = (b - a)
            side = d.cross(V((0, 0, 1))).normalized() * 0.012 * s
            rocket.rod(name + "Arm", a + side, b + side, 0.005, M["frame"], parent=g, verts=12)
        rocket.rod(name + "Spring", a + (b - a) * 0.15 + V((0, 0, 0.02)), a + (b - a) * 0.6 + V((0, 0, 0.02)), 0.006,
                   M["steel"], parent=g, verts=10)
        sb.prim("sphere", name + "Joint", loc=b, radius=0.014, segments=16, ring_count=8, mat=M["frame"], parent=g)
    q = (V(head_target) - head).normalized().to_track_quat('-Z', 'Y')
    hd = box(name + "Head", head, (0.26, 0.06, 0.03), M["frame"], parent=g, bev=0.006)
    hd.rotation_mode = 'QUATERNION'; hd.rotation_quaternion = q
    led = sb.prim("plane", name + "LED", loc=head + q @ V((0, 0, -0.0155)), size=1.0, mat=M["lamp"], parent=g)
    led.scale = (0.24, 0.045, 1)
    led.rotation_mode = 'QUATERNION'; led.rotation_quaternion = q @ Euler((math.pi, 0, 0)).to_quaternion()
    L = sb.light('AREA', name + "Light", loc=head + q @ V((0, 0, -0.03)), target=head_target, energy=3.0,
                 color=(1.0, 0.84, 0.66), size=0.24)
    L.data.shape = 'RECTANGLE'
    L.data.size_y = 0.045
    L.data.spread = math.radians(110)
    return g, L


# --------------------------------------------------------------------------- the room

def build(detail=1, window_sun=True, sun_elev=16.0, sun_az=200.0):
    """Builds the workshop; returns dict with anchors (bench centre etc.)."""
    M = materials()
    root = sb.empty("Workshop")
    Z = BENCH_Z
    # bench top (butcher block) with a front edge bevel, steel frame, lower shelf, drawers
    top = box("BenchTop", (0, 0, Z - 0.022), (2.0, 0.8, 0.044), M["wood"], parent=root, bev=0.004)
    for sx in (-0.95, 0.95):
        for sy in (-0.35, 0.35):
            box("BenchLeg", (sx, sy, (Z - 0.044) / 2), (0.05, 0.05, Z - 0.044), M["frame"], parent=root, bev=0.003)
    for sy in (-0.35, 0.35):
        box("BenchRail", (0, sy, 0.18), (1.9, 0.04, 0.04), M["frame"], parent=root, bev=0.003)
    box("BenchShelf", (0, 0, 0.2), (1.9, 0.72, 0.02), M["wood"], parent=root, bev=0.002)
    box("DrawerUnit", (0.62, -0.02, Z - 0.2), (0.5, 0.7, 0.3), M["frame"], parent=root, bev=0.004)
    for k in range(3):
        box("Drawer", (0.62, -0.372, Z - 0.1 - k * 0.095), (0.46, 0.012, 0.085), M["grey"], parent=root, bev=0.002)
        box("DrawerPull", (0.62, -0.382, Z - 0.1 - k * 0.095), (0.12, 0.012, 0.012), M["steel"], parent=root, bev=0.002)
    mat = box("CutMat", (0.0, -0.02, Z + 0.0015), (0.6, 0.45, 0.003), M["mat"], parent=root, bev=0.001)
    # floor + walls
    sb.prim("plane", "Floor", loc=(0, 0, 0), scale=(4, 4, 1), mat=M["floor"])
    wall = sb.prim("cube", "BackWall", loc=(0, 0.75, 1.4), scale=(3.0, 0.05, 1.4), mat=M["wall"])
    sb.prim("cube", "SideWallR", loc=(1.9, 0, 1.4), scale=(0.05, 2.5, 1.4), mat=M["wall"])
    sb.prim("cube", "FrontWall", loc=(0, -2.4, 1.4), scale=(3.0, 0.05, 1.4), mat=M["wall"])
    ceil = sb.prim("cube", "Ceiling", loc=(0, 0, 2.85), scale=(3.0, 3.0, 0.05), mat=sb.concrete("CeilM", (0.55, 0.54, 0.52), scale=1.0))
    # pegboard + tools
    pb = box("Pegboard", (0.1, 0.69, Z + 0.45), (1.6, 0.012, 0.75), M["peg"], parent=root, bev=0.002)
    rs = random.Random(7)
    for i in range(7):
        screwdriver("WallSD%d" % i, (-0.55 + i * 0.045, 0.66, Z + 0.62), (math.pi, 0, 0), M,
                    handle=rs.choice(["black", "amber", "black", "grey"]), L=rs.uniform(0.16, 0.23), parent=root)
    for i in range(3):
        pl = pliers("WallPliers%d" % i, (0.05 + i * 0.12, 0.665, Z + 0.62), -math.pi / 2, M, parent=root, open_=0.12)
        pl.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
    for i in range(6):   # wrenches (flat), graduated
        L = 0.12 + i * 0.02
        w = rocket.extrude("Wrench%d" % i, [(-0.008, 0), (0.008, 0), (0.007, L), (0.015, L + 0.012), (0.006, L + 0.026), (-0.006, L + 0.026),
                                              (-0.015, L + 0.012), (-0.007, L)], 0.004, (0.45 + i * 0.04, 0.66, Z + 0.35),
                           V((1, 0, 0)), V((0, 0, 1)), M["chrome"], root, bevel=0.001)
    # coil of cable + tape rolls on pegboard hooks
    tor = sb.prim("torus", "CableCoil", loc=(-0.35, 0.64, Z + 0.4), rot=(math.pi / 2, 0, 0), major_radius=0.07, minor_radius=0.012,
                  mat=M["black"], parent=root)
    for i, c in enumerate(("amber", "blue", "white")):
        sb.prim("torus", "Tape%d" % i, loc=(-0.15 + i * 0.07, 0.655, Z + 0.33), rot=(math.pi / 2, 0, 0), major_radius=0.024,
                minor_radius=0.01, mat=M[c], parent=root)
    # shelf with bins above the pegboard
    box("Shelf", (0.1, 0.62, Z + 0.87), (1.6, 0.22, 0.02), M["wood"], parent=root, bev=0.002)
    for sx in (-0.6, 0.8):
        rocket.extrude("ShelfBracket", [(0, 0), (0.18, 0), (0.0, -0.18)], 0.01, (sx, 0.69, Z + 0.86), V((0, -1, 0)), V((0, 0, 1)),
                       M["frame"], root, bevel=0.001)
    bins("Bin", (-0.55, 0.62, Z + 0.88), M, n=11, parent=root)
    # whiteboard on the right wall section with pinned sketches (abstract)
    box("Board", (1.25, 0.72, Z + 0.55), (0.7, 0.012, 0.55), M["white"], parent=root, bev=0.003)
    for i in range(4):
        pg = box("PinnedSketch%d" % i, (1.05 + (i % 2) * 0.28, 0.712, Z + 0.68 - (i // 2) * 0.26), (0.2, 0.002, 0.15),
                 sketch_paper(), parent=root, bev=0.0003)
        pg.rotation_euler = (0, rs.uniform(-0.05, 0.05), 0)
    # window on the -X wall: frame, mullions, glass; outside: warm sky + far buildings/trees as soft silhouettes
    wx = -1.35
    for (c, s) in (((wx, 0.1, 0.55), (0.1, 1.8, 0.1)), ((wx, 0.1, 2.35), (0.1, 1.8, 0.1)), ((wx, -0.8, 1.45), (0.1, 0.06, 1.9)),
                   ((wx, 1.0, 1.45), (0.1, 0.06, 1.9)), ((wx, 0.1, 1.45), (0.1, 0.04, 1.9)), ((wx, 0.1, 1.45), (0.1, 1.8, 0.04))):
        box("WinFrame", c, s, M["frame"], parent=root, bev=0.004)
    for (c, s) in (((wx - 0.1, -1.5, 1.4), (0.1, 1.4, 2.8)), ((wx - 0.1, 1.7, 1.4), (0.1, 1.4, 2.8)), ((wx - 0.1, 0.1, 0.25), (0.1, 3.0, 0.5)),
                   ((wx - 0.1, 0.1, 2.7), (0.1, 3.0, 0.7))):
        box("WallL", c, s, M["wall"], parent=root, bev=0.0)
    # outside world: a street of warm facades and trees (soft through the window)
    out_m = sb.painted("Facade", (0.62, 0.5, 0.38), rough=0.8, grime=0.3, scale=0.5)
    for i in range(6):
        h = rs.uniform(6, 14)
        box("OutBldg%d" % i, (wx - rs.uniform(14, 30), -6 + i * 3.2, h / 2 - 2), (rs.uniform(4, 8), 2.8, h), out_m, parent=root, bev=0.0)
    for i in range(5):
        sb.prim("ico", "OutTree%d" % i, loc=(wx - rs.uniform(6, 10), -3 + i * 1.6, rs.uniform(2.0, 3.5)), subdivisions=3,
                radius=rs.uniform(1.2, 2.0), mat=sb.mat("TreeGreen", (0.06, 0.1, 0.04), rough=0.9), parent=root)
    # lights: low warm sun through the window, sky fill, practical lamp placed by the scene
    sun = None
    if window_sun:
        sun = sb.sun(sun_elev, sun_az, energy=3.2, color=(1.0, 0.76, 0.52), angle=0.8)
    sb.world_sky(elev=sun_elev, azim=sun_az, strength=0.25)
    sb.light('AREA', "WindowFill", loc=(wx + 0.08, 0.1, 1.45), target=(0, 0, Z), energy=25.0, color=(0.85, 0.9, 1.0), size=1.8)
    sb.light('AREA', "CeilingPanel", loc=(0.2, -0.2, 2.78), target=(0.2, -0.2, 0), energy=35.0, color=(1.0, 0.95, 0.88), size=1.2)
    return dict(root=root, M=M, Z=Z, sun=sun)


def bench_dressing(M, Z, keep_clear=((0.0, 0.0), 0.16)):
    """Props on the bench around (not inside) the working area keep_clear=(centre_xy, radius)."""
    c0, r0 = V((*keep_clear[0], 0)), keep_clear[1]
    items = []
    items.append(calipers("Calipers", (0.2, -0.16, Z + 0.003), 0.25, M))
    items.append(screwdriver("SD1", (-0.22, -0.18, Z + 0.0095), (0, math.pi / 2, 0.35), M, handle="amber"))
    items.append(screwdriver("SD2", (-0.25, -0.12, Z + 0.0095), (0, math.pi / 2, 0.25), M, handle="black", L=0.16))
    items.append(pliers("Tweezers", (0.2, 0.08, Z + 0.001), 2.2, M, open_=0.05))
    items.append(mug("Mug", (0.52, 0.18, Z), M))
    items.append(notebook("Notebook", (-0.45, 0.12, Z), -0.2, M))
    # soldering station + spool
    box("SolderStn", (0.72, 0.28, Z + 0.045), (0.16, 0.13, 0.09), M["black"], bev=0.006)
    cyl("SolderSpool", (0.62, 0.3, Z + 0.02), 0.03, 0.03, M["copper"], rot=(math.pi / 2, 0, 0), verts=32, bev=0.002)
    # the second monitor (angled, CAD glow) far right
    box("Mon2", (0.85, 0.45, Z + 0.3), (0.55, 0.03, 0.33), M["black"], bev=0.006, rot=(0, 0, -0.5))
    scr = sb.prim("plane", "Mon2Scr", loc=(0.845, 0.432, Z + 0.3), size=1.0, mat=M["cad"])
    scr.scale = (0.26, 0.155, 1); scr.rotation_euler = (math.pi / 2, 0, -0.5)
    box("Mon2Stand", (0.87, 0.48, Z + 0.07), (0.05, 0.05, 0.14), M["frame"], bev=0.004)
    return items


def far_side(M, y0=-2.3):
    """The rest of the shop behind the engineer (-Y): back wall with a door and pendant lights, an enclosed 3D
    printer glowing on a second bench, a tall shelving rack with boxes and spools, a plant, a stool, a jacket."""
    rs = random.Random(11)
    root = sb.empty("FarSide")
    sb.prim("cube", "BackWallS", loc=(0, y0 - 0.05, 1.4), scale=(3.0, 0.05, 1.4), mat=M["wall"])
    # door (graphite, with a narrow glazed strip)
    box("Door", (1.1, y0 + 0.01, 1.05), (0.9, 0.04, 2.1), M["frame"], parent=root, bev=0.006)
    box("DoorGlass", (0.8, y0 + 0.035, 1.3), (0.12, 0.01, 1.2), M["screen"], parent=root, bev=0.002)
    # second bench + enclosed printer with a lit chamber
    box("Bench2", (-0.6, y0 + 0.45, 0.9), (1.4, 0.7, 0.04), M["wood"], parent=root, bev=0.003)
    for sx in (-1.25, 0.05):
        for sy in (y0 + 0.15, y0 + 0.75):
            box("Bench2Leg", (sx, sy, 0.44), (0.05, 0.05, 0.88), M["frame"], parent=root, bev=0.003)
    pr = V((-0.85, y0 + 0.45, 0.92))
    box("Printer", pr + V((0, 0, 0.25)), (0.5, 0.5, 0.5), M["white"], parent=root, bev=0.012)
    box("PrinterWin", pr + V((0, 0.252, 0.27)), (0.36, 0.01, 0.34), sb.emit_mat("PrinterGlow", (1.0, 0.72, 0.45), 3.0), parent=root, bev=0.004)
    box("PrinterPanel", pr + V((0.2, 0.253, 0.07)), (0.06, 0.01, 0.04), M["screen"], parent=root, bev=0.002)
    sb.light('POINT', "PrinterLight", loc=pr + V((0, 0.35, 0.3)), energy=6.0, color=(1.0, 0.75, 0.5), size=0.2)
    box("FilBox", (-0.35, y0 + 0.5, 0.97), (0.22, 0.22, 0.1), M["grey"], parent=root, bev=0.005)
    cyl("Spool1", (-0.15, y0 + 0.45, 1.0), 0.1, 0.07, M["amber"], rot=(0, math.pi / 2, 0), parent=root, verts=40, bev=0.004)
    # shelving rack
    for sx in (-2.0, -1.4):
        for sy in (y0 + 0.1, y0 + 0.5):
            box("RackPost", (sx + 0.0, sy, 1.0), (0.035, 0.035, 2.0), M["frame"], parent=root, bev=0.002)
    for k in range(5):
        zz = 0.25 + k * 0.42
        box("RackShelf", (-1.7, y0 + 0.3, zz), (0.64, 0.46, 0.02), M["steel"], parent=root, bev=0.002)
        x = -1.95
        while x < -1.45:
            w = rs.uniform(0.1, 0.22)
            h = rs.uniform(0.12, 0.3)
            box("RackBox", (x + w / 2, y0 + 0.3 + rs.uniform(-0.05, 0.05), zz + h / 2 + 0.01), (w, rs.uniform(0.25, 0.4), h),
                M[rs.choice(["cork", "grey", "white", "blue", "cork"])], parent=root, bev=0.004)
            x += w + 0.02
    # plant in a pot
    cyl("Pot", (0.4, y0 + 0.35, 0.18), 0.14, 0.36, M["ceramic"], parent=root, verts=40, bev=0.01)
    leafm = sb.mat("PlantLeaf", (0.06, 0.14, 0.05), rough=0.5, sss=0.1)
    for i in range(14):
        a = rs.uniform(0, 2 * math.pi)
        lf = sb.prim("sphere", "Leaf", loc=(0.4 + 0.18 * math.cos(a), y0 + 0.35 + 0.18 * math.sin(a), 0.55 + rs.uniform(0, 0.5)),
                     radius=0.12, segments=12, ring_count=6, mat=leafm, parent=root)
        lf.scale = (1.0, 0.35, 0.08)
        lf.rotation_euler = (rs.uniform(-0.6, 0.6), rs.uniform(-0.6, 0.6), a)
    # pendant lamps (warm)
    for x in (-0.8, 0.6):
        cyl("PendantCord", (x, y0 + 1.2, 2.45), 0.003, 0.5, M["black"], parent=root, verts=8)
        sb.lathe("PendantShade", [(0.0, 0.0), (0.03, 0.0), (0.16, -0.14), (0.165, -0.15)], segs=40, mat=M["frame"]).location = (x, y0 + 1.2, 2.2)
        sb.prim("sphere", "PendantBulb", loc=(x, y0 + 1.2, 2.07), radius=0.035, segments=16, ring_count=8,
                mat=sb.emit_mat("Bulb", (1.0, 0.8, 0.55), 40.0))
        sb.light('POINT', "Pendant", loc=(x, y0 + 1.2, 2.02), energy=45.0, color=(1.0, 0.8, 0.58), size=0.05)
    return root


def hand_fixture(name, loc, rz, M, socket_y=-0.13, socket_r=0.034, socket_z=0.0, palm_pad=(0.0, 0.035)):
    """Machined holding fixture for the prosthetic: anodised base plate with a tapped-hole grid, two uprights with
    rubber-lined V-cradles under the carbon socket, clamping knobs, and a silicone pad under the palm.
    loc = wrist centre of the hand (world); rz = yaw (rad). Socket axis runs along local -Y from the wrist."""
    g = sb.empty(name, loc=loc)
    g.rotation_euler = (0, 0, rz)
    Zb = -loc[2] + BENCH_Z                       # bench surface in fixture-local z
    anod = sb.painted(name + "Anod", (0.05, 0.052, 0.058), rough=0.35, coat=0.2, grime=0.1, wear=0.35, scale=20.0,
                      under=(0.7, 0.7, 0.72))
    anod.use_fake_user = True
    base = rocket.extrude(name + "Base", [(-0.06, -0.26), (0.06, -0.26), (0.07, -0.25), (0.07, 0.05), (0.06, 0.06),
                                          (-0.06, 0.06), (-0.07, 0.05), (-0.07, -0.25)], 0.012,
                          (0, 0, Zb + 0.006), V((1, 0, 0)), V((0, 1, 0)), anod, g, bevel=0.0015)
    hole_m = sb.mat(name + "Hole", (0.005, 0.005, 0.006), rough=0.6)
    for i in range(3):
        for j in range(7):
            cyl(name + "Tap", (-0.045 + i * 0.045, -0.24 + j * 0.045, Zb + 0.0121), 0.0028, 0.0004, hole_m, parent=g, verts=12)
    for k, yy in enumerate((socket_y - 0.03, socket_y + 0.06)):
        zc = socket_z
        top = zc - socket_r * 0.35
        prof = [(-0.05, Zb + 0.012), (0.05, Zb + 0.012), (0.05, top), (socket_r * 0.92, top)]
        prof += [(socket_r * 1.0 * math.cos(math.radians(a)), zc + socket_r * 1.0 * math.sin(math.radians(a))) for a in range(-20, -161, -10)]
        prof += [(-socket_r * 0.92, top), (-0.05, top)]
        rocket.extrude(name + "Up%d" % k, prof, 0.014, (0, yy, 0), V((1, 0, 0)), V((0, 0, 1)), anod, g, bevel=0.0012)
        # rubber lining in the cradle
        lin = [(socket_r * 1.02 * math.cos(math.radians(a)), zc + socket_r * 1.02 * math.sin(math.radians(a))) for a in range(-25, -156, -10)]
        lin += [(socket_r * 0.95 * math.cos(math.radians(a)), zc + socket_r * 0.95 * math.sin(math.radians(a))) for a in range(-155, -24, 10)]
        rocket.extrude(name + "Lin%d" % k, lin, 0.016, (0, yy, 0), V((1, 0, 0)), V((0, 0, 1)), M["black"], g, bevel=0.0005)
        for s in (-1, 1):
            kn = cyl(name + "Knob", (s * 0.058, yy, top - 0.01), 0.009, 0.012, M["black"], parent=g, rot=(0, math.pi / 2, 0), verts=12, bev=0.001)
            cyl(name + "KnobStem", (s * 0.051, yy, top - 0.01), 0.003, 0.01, M["steel"], parent=g, rot=(0, math.pi / 2, 0), verts=12)
    # silicone pad under the palm on a short post
    px, py = palm_pad
    pz = -0.018
    cyl(name + "PadPost", (px, py, (Zb + 0.012 + pz - 0.006) / 2), 0.008, (pz - 0.006) - (Zb + 0.012), M["steel"], parent=g, verts=24, bev=0.001)
    pad = rocket.extrude(name + "Pad", chamfer(0.05, 0.05, 0.01), 0.008, (px, py, pz - 0.004), V((1, 0, 0)), V((0, 1, 0)),
                         sb.mat(name + "Sil", (0.06, 0.06, 0.065), rough=0.7), g, bevel=0.002)
    return g


def chamfer(w, h, c):
    x, y = w / 2, h / 2
    return [(-x + c, -y), (x - c, -y), (x, -y + c), (x, y - c), (x - c, y), (-x + c, y), (-x, y - c), (-x, -y + c)]
