"""Manufacturing hall around a robot cell (s07b): steel portal frame, ribbed cladding with a clerestory band of
daylight, roof trusses, extraction ducting, a row of powder-bed machines, pallet racking with totes and cartons,
robot controller + stack light. Everything real-scale (metres); restrained graphite / silver / cool daylight.
Axes: the hall's back wall is at x = X_BACK (the camera looks roughly toward -x)."""
import bpy, math
import numpy as np
from mathutils import Vector as V
import sb, props

X_BACK = -4.2
Y_SIDE = -3.6
ROOF = 5.2


def cladding(name="Cladding", color=(0.62, 0.64, 0.66), pitch=0.2):
    """Trapezoidal steel sheet: vertical ribs (bump from a clipped triangle wave), faint horizontal sheet seams,
    streaky grime from the top, occasional dents."""
    m = sb.mat(name, color, metal=0.35, rough=0.45)
    nb = sb.NB(m)
    geo = nb.new('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(pos, sep.inputs[0])
    # rib coordinate along the wall: use x + y so the same material works on walls facing either axis
    u = nb.math('ADD', sep.outputs[0], sep.outputs[1])
    tri = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', u, pitch)), 0.5))
    rib = nb.maprange(tri, 0.12, 0.2)
    seam = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[2], 1.8)), 0.5)), 0.497, 0.4995)
    dent = nb.noise(pos, scale=1.3, detail=3)
    h = nb.math('ADD', nb.math('SUBTRACT', rib, nb.math('MULTIPLY', seam, 0.5)), nb.math('MULTIPLY', dent.outputs['Fac'], 0.05))
    nb.set('Normal', nb.bump(h, strength=0.6, distance=0.01))
    st = nb.new('ShaderNodeVectorMath'); st.operation = 'MULTIPLY'
    nb.link(pos, st.inputs[0]); st.inputs[1].default_value = (3.0, 3.0, 0.15)
    streak = nb.maprange(nb.noise(st.outputs[0], scale=4, detail=4).outputs['Fac'], 0.45, 0.7)
    top = nb.maprange(sep.outputs[2], 1.0, ROOF)
    dirt = nb.math('MULTIPLY', streak, nb.math('ADD', 0.25, nb.math('MULTIPLY', top, 0.6)))
    nb.set('Base Color', nb.mix(nb.math('MULTIPLY', dirt, 0.45), color + (1,), (0.33, 0.33, 0.33, 1)))
    nb.set('Roughness', nb.mix(dirt, 0.42, 0.62, dtype='FLOAT'))
    return m


def daylight_glass(name="Clerestory", strength=6.0):
    """Translucent polycarbonate daylight panel: emissive sky, slightly mottled, with glazing bars."""
    m = sb.mat(name, (0.9, 0.95, 1.0), rough=0.4)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    u = nb.math('ADD', sep.outputs[0], sep.outputs[1])
    bars = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('MULTIPLY', u, 0.8)), 0.5)), 0.47, 0.485)
    mot = nb.noise(co, scale=3, detail=3)
    k = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, bars), nb.maprange(mot.outputs['Fac'], 0.3, 0.7, 0.8, 1.1))
    nb.set('Emission Color', (0.86, 0.92, 1.0, 1))
    nb.set('Emission Strength', nb.math('MULTIPLY', k, strength))
    nb.set('Base Color', (0.02, 0.02, 0.022, 1))
    return m


def column(name, x, y, h, mat, web_x=True, d=0.36, w=0.18, t=0.014):
    """Vertical I-section column (web along x if web_x else along y)."""
    def box(loc, sc_):
        return sb.prim("cube", name, loc=(x + loc[0], y + loc[1], h / 2), scale=(sc_[0], sc_[1], h / 2), mat=mat)
    if web_x:
        box((0, 0), (d / 2 - t, t / 2)); box((d / 2 - t / 2, 0), (t / 2, w / 2)); box((-(d / 2 - t / 2), 0), (t / 2, w / 2))
    else:
        box((0, 0), (t / 2, d / 2 - t)); box((0, d / 2 - t / 2), (w / 2, t / 2)); box((0, -(d / 2 - t / 2)), (w / 2, t / 2))


def hall(M):
    """Back wall + side wall with cladding, clerestory daylight band, portal columns, roof trusses, ducting."""
    clad = cladding()
    steel = sb.painted("HallSteel", (0.20, 0.22, 0.25), rough=0.45, grime=0.25, wear=0.2)
    M["hall_steel"] = steel
    L = 16.0
    # walls (the dado: a darker painted block-work band to 1.2 m)
    block = sb.painted("Dado", (0.30, 0.31, 0.32), rough=0.7, grime=0.4, scale=2.0)
    for (loc, sc_, rz) in (((X_BACK, 0, ROOF / 2), (0.02, L / 2, ROOF / 2), 0), ((0, Y_SIDE, ROOF / 2), (L / 2, 0.02, ROOF / 2), 0)):
        w = sb.prim("cube", "HallWall", loc=loc, scale=sc_, mat=clad)
    sb.prim("cube", "DadoBack", loc=(X_BACK + 0.06, 0, 0.6), scale=(0.05, L / 2, 0.6), mat=block)
    sb.prim("cube", "DadoSide", loc=(0, Y_SIDE + 0.06, 0.6), scale=(L / 2, 0.05, 0.6), mat=block)
    # clerestory daylight band high on both walls
    day = daylight_glass()
    sb.prim("cube", "ClereBack", loc=(X_BACK + 0.025, 0, 3.9), scale=(0.005, L / 2 - 0.2, 0.55), mat=day)
    sb.prim("cube", "ClereSide", loc=(0, Y_SIDE + 0.025, 3.9), scale=(L / 2 - 0.2, 0.005, 0.55), mat=day)
    for zz in (3.33, 4.47):
        sb.prim("cube", "ClereSill", loc=(X_BACK + 0.08, 0, zz), scale=(0.06, L / 2, 0.03), mat=steel)
        sb.prim("cube", "ClereSillS", loc=(0, Y_SIDE + 0.08, zz), scale=(L / 2, 0.06, 0.03), mat=steel)
    # portal columns along both walls + roof trusses spanning the hall
    for y in np.arange(-6.0, 7.0, 3.0):
        column("ColB", X_BACK + 0.25, y, ROOF, steel, web_x=True)
        _base_plate((X_BACK + 0.25, y), steel)
    for x in np.arange(-3.0, 8.0, 3.0):
        column("ColS", x, Y_SIDE + 0.25, ROOF, steel, web_x=False)
        _base_plate((x, Y_SIDE + 0.25), steel)
    for y in np.arange(-6.0, 7.0, 3.0):
        truss((X_BACK, y), (8.0, y), ROOF - 0.05, steel)
    # roof deck (dark) and purlins
    sb.prim("cube", "RoofDeck", loc=(2, 0, ROOF + 0.9), scale=(8, 8, 0.02), mat=sb.painted("RoofDeck", (0.12, 0.125, 0.13), rough=0.6))
    for x in np.arange(-4.0, 8.0, 1.5):
        sb.prim("cube", "Purlin", loc=(x, 0, ROOF + 0.78), scale=(0.05, 8.0, 0.1), mat=steel)
    # extraction duct along the back wall with drops to the machines (spiral-seam galvanised)
    galv = sb.brushed_metal("Galv", (0.66, 0.67, 0.68), rough=0.42, aniso=0.2, scale=60)
    d = sb.prim("cyl", "Duct", loc=(X_BACK + 0.9, 0, 3.0), rot=(math.pi / 2, 0, 0), vertices=48, radius=0.22, depth=L, mat=galv)
    for y in np.arange(-7.5, 8.0, 1.2):
        sb.prim("torus", "DuctSeam", loc=(X_BACK + 0.9, y, 3.0), rot=(math.pi / 2, 0, 0), major_radius=0.222, minor_radius=0.008, mat=galv)
    # cable ladder along the side wall
    for s in (-1, 1):
        sb.prim("cube", "Ladder", loc=(2.0, Y_SIDE + 0.6 + s * 0.15, 3.4), scale=(6.0, 0.012, 0.04), mat=galv)
    for x in np.arange(-3.9, 8.0, 0.3):
        sb.prim("cube", "Rung", loc=(x, Y_SIDE + 0.6, 3.37), scale=(0.012, 0.15, 0.008), mat=galv)
    # high-bay luminaires (visible practicals) on drop rods
    lum = sb.painted("LumBody", (0.8, 0.8, 0.8), rough=0.4)
    lem = sb.emit_mat("LumLED", (0.95, 0.97, 1.0), 18.0)
    for x in (-2.5, 0.5, 3.5):
        for y in (-2.0, 1.0):
            sb.prim("cube", "Lum", loc=(x, y, 4.2), scale=(0.7, 0.12, 0.03), mat=lum)
            sb.prim("cube", "LumE", loc=(x, y, 4.168), scale=(0.66, 0.09, 0.003), mat=lem)
            for sx in (-1, 1):
                sb.prim("cyl", "Rod", loc=(x + sx * 0.5, y, 4.7), vertices=8, radius=0.006, depth=1.0, mat=galv)


def _base_plate(xy, mat):
    sb.prim("cube", "BasePl", loc=(xy[0], xy[1], 0.012), scale=(0.22, 0.22, 0.012), mat=mat)
    bm = sb.brushed_metal("AnchorNut", (0.5, 0.5, 0.52), rough=0.35)
    for sx in (-1, 1):
        for sy in (-1, 1):
            sb.prim("cyl", "Nut", loc=(xy[0] + sx * 0.16, xy[1] + sy * 0.16, 0.034), vertices=6, radius=0.02, depth=0.022, mat=bm)


def truss(a, b, z, mat, depth=0.9, bay=1.0):
    """Planar Warren roof truss from a to b (xy) at height z (bottom chord)."""
    a, b = V((a[0], a[1], z)), V((b[0], b[1], z))
    L = (b - a).length
    d = (b - a).normalized()
    top = V((0, 0, depth))
    sb.prim("cube", "Chord", loc=(a + b) / 2, rot=(0, 0, math.atan2(d.y, d.x)), scale=(L / 2, 0.06, 0.06), mat=mat)
    sb.prim("cube", "ChordT", loc=(a + b) / 2 + top, rot=(0, 0, math.atan2(d.y, d.x)), scale=(L / 2, 0.06, 0.06), mat=mat)
    n = int(L / bay)
    for i in range(n):
        p0 = a + d * (i * bay)
        p1 = a + d * ((i + 0.5) * bay) + top
        p2 = a + d * ((i + 1) * bay)
        for (u, v) in ((p0, p1), (p1, p2)):
            c = (u + v) / 2
            e = v - u
            sb.prim("cube", "Web", loc=c, rot=(0, -math.atan2(e.z, math.hypot(e.x, e.y)), math.atan2(d.y, d.x)),
                    scale=(e.length / 2, 0.025, 0.025), mat=mat)


def machine(name, loc, rz, M, glow=(0.55, 0.62, 0.75), glow_s=0.8, rs=None):
    """Powder-bed fusion machine: white cabinet on a graphite plinth, graphite door with round viewport (lit build
    chamber), amber viewport ring, glovebox ports with black gloves, touchscreen, status light bar, filter module."""
    rs = rs or np.random.default_rng(len(name))
    root = bpy.data.objects.new(name, None)
    sb.link_obj(root)
    root.location = loc
    root.rotation_euler = (0, 0, rz)
    white = M["white"]; graph = M["graph"]

    def P(*a, **k):
        o = sb.prim(*a, **k)
        o.parent = root
        return o
    cab = P("cube", name + "Cab", loc=(0, 0, 1.05), scale=(0.55, 0.8, 0.97), mat=white); sb.bevel(cab, 0.035, 4)
    P("cube", name + "Plinth", loc=(0, 0, 0.05), scale=(0.53, 0.78, 0.05), mat=graph)
    P("cube", name + "Top", loc=(-0.05, 0, 2.06), scale=(0.45, 0.7, 0.04), mat=graph)
    door = P("cube", name + "Door", loc=(0.555, 0.1, 1.1), scale=(0.012, 0.44, 0.46), mat=graph); sb.bevel(door, 0.02, 3)
    ch = sb.emit_mat(name + "Chamber", glow, glow_s)
    P("cyl", name + "Win", loc=(0.566, 0.1, 1.14), rot=(0, math.pi / 2, 0), vertices=64, radius=0.16, depth=0.006, mat=ch)
    ring = P("torus", name + "WinRing", loc=(0.57, 0.1, 1.14), rot=(0, math.pi / 2, 0), major_radius=0.166, minor_radius=0.009,
             mat=M["amber"])
    for s in (-1, 1):
        # glovebox ports below the viewport: steel ring + black glove sleeve hanging inside (dark disc)
        P("torus", name + "GPort", loc=(0.57, 0.1 + s * 0.2, 0.78), rot=(0, math.pi / 2, 0), major_radius=0.075, minor_radius=0.012, mat=M["rubber"])
        P("cyl", name + "Glove", loc=(0.566, 0.1 + s * 0.2, 0.78), rot=(0, math.pi / 2, 0), vertices=32, radius=0.07, depth=0.004, mat=M["rubber"])
    hdl = P("cube", name + "Handle", loc=(0.59, -0.3, 1.1), scale=(0.015, 0.012, 0.22), mat=M["steel"]); sb.bevel(hdl, 0.006)
    # touchscreen (dim UI glow) on an arm
    P("cube", name + "ScrArm", loc=(0.62, -0.62, 1.35), scale=(0.08, 0.015, 0.015), mat=graph)
    scr = P("cube", name + "Scr", loc=(0.7, -0.62, 1.38), rot=(0, math.radians(-12), 0), scale=(0.01, 0.16, 0.1), mat=graph)
    P("cube", name + "ScrE", loc=(0.712, -0.62, 1.38), rot=(0, math.radians(-12), 0), scale=(0.002, 0.145, 0.085), mat=M["ui"])
    # status light bar along the top edge (cool white = building)
    P("cube", name + "Status", loc=(0.56, 0.0, 1.98), scale=(0.006, 0.7, 0.006), mat=M["status"])
    # filter / powder-recirculation module on the side + flexible duct up to the extraction
    fm = P("cube", name + "Filter", loc=(-0.2, 0.95, 0.75), scale=(0.28, 0.15, 0.7), mat=white); sb.bevel(fm, 0.02, 3)
    P("cyl", name + "Flex", loc=(-0.2, 0.95, 2.35), vertices=24, radius=0.08, depth=0.9, mat=M["flex"])
    for i in range(6):
        P("cube", name + "Vent", loc=(0.085, 0.95 + (i - 2.5) * 0.02, 1.2), scale=(0.001, 0.004, 0.12), mat=graph)
    return root


def rack(x0, y0, M, bays=3, levels=4, rs=None, rz=0.0):
    """Pallet racking along +x from (x0, y0): graphite uprights with perforations, amber-free steel-blue beams,
    wire decking, totes and wrapped cartons with variation."""
    rs = rs or np.random.default_rng(5)
    up = M["rack_up"]; bm = M["rack_beam"]
    bw, dp, lh = 2.7, 1.0, 1.35
    for i in range(bays + 1):
        for s in (0, 1):
            sb.prim("cube", "Upright", loc=(x0 + i * bw, y0 + s * dp, levels * lh / 2 + 0.2), scale=(0.045, 0.035, levels * lh / 2 + 0.2), mat=up)
        for z in np.arange(0.3, levels * lh, 0.6):
            sb.prim("cube", "Brace", loc=(x0 + i * bw, y0 + dp / 2, z + 0.3), rot=(math.radians(30), 0, 0), scale=(0.012, 0.018, 0.35), mat=up)
    for i in range(bays):
        for l in range(levels):
            z = 0.15 + l * lh
            for s in (0, 1):
                sb.prim("cube", "Beam", loc=(x0 + (i + 0.5) * bw, y0 + s * dp, z), scale=(bw / 2 - 0.05, 0.03, 0.06), mat=bm)
            sb.prim("cube", "Deck", loc=(x0 + (i + 0.5) * bw, y0 + dp / 2, z + 0.065), scale=(bw / 2 - 0.05, dp / 2, 0.005), mat=M["deck"])
            x = x0 + i * bw + 0.1
            while x < x0 + (i + 1) * bw - 0.45:
                kind = rs.random()
                if kind < 0.45:
                    w, d, h = 0.6, 0.4, rs.choice([0.22, 0.32])
                    tm = M["tote"][int(rs.integers(0, len(M["tote"])))]
                    t = sb.prim("cube", "Tote", loc=(x + w / 2, y0 + dp / 2 + rs.uniform(-0.1, 0.1), z + 0.07 + h / 2), scale=(w / 2, d / 2, h / 2), mat=tm)
                    sb.bevel(t, 0.02, 2)
                    if rs.random() < 0.6:
                        t2 = sb.prim("cube", "Tote", loc=(x + w / 2, y0 + dp / 2, z + 0.07 + h * 1.5 + 0.005), scale=(w / 2, d / 2, h / 2), mat=tm)
                        sb.bevel(t2, 0.02, 2)
                elif kind < 0.8:
                    w, d, h = rs.uniform(0.35, 0.8), rs.uniform(0.3, 0.7), rs.uniform(0.25, 0.7)
                    c = sb.prim("cube", "Carton", loc=(x + w / 2, y0 + dp / 2 + rs.uniform(-0.1, 0.1), z + 0.07 + h / 2),
                                rot=(0, 0, rs.normal(0, 0.03)), scale=(w / 2, d / 2, h / 2), mat=M["carton"])
                    sb.bevel(c, 0.006, 2)
                else:
                    w = 0.9
                    h = rs.uniform(0.6, 1.0)
                    c = sb.prim("cube", "Wrapped", loc=(x + w / 2, y0 + dp / 2, z + 0.07 + h / 2), scale=(w / 2, 0.45, h / 2), mat=M["wrap"])
                    sb.bevel(c, 0.04, 3)
                x += w + rs.uniform(0.04, 0.15)


def controller(loc, M):
    """Robot controller cabinet + teach pendant on its hook + stack light (green = running)."""
    c = sb.prim("cube", "RCtrl", loc=(loc[0], loc[1], 0.5), scale=(0.3, 0.25, 0.5), mat=M["graph"]); sb.bevel(c, 0.01, 2)
    for i in range(8):
        sb.prim("cube", "RVent", loc=(loc[0] + 0.301, loc[1] - 0.1 + i * 0.025, 0.25), scale=(0.001, 0.006, 0.12), mat=M["rubber"])
    sb.prim("cube", "Pendant", loc=(loc[0] + 0.32, loc[1] + 0.12, 0.8), rot=(0, math.radians(8), 0), scale=(0.02, 0.1, 0.14), mat=M["white"])
    sb.prim("cube", "PendScr", loc=(loc[0] + 0.342, loc[1] + 0.12, 0.83), rot=(0, math.radians(8), 0), scale=(0.001, 0.07, 0.06), mat=M["ui"])
    sb.prim("cyl", "Cable", loc=(loc[0] + 0.3, loc[1] + 0.12, 0.45), vertices=12, radius=0.008, depth=0.6, mat=M["rubber"])
    # stack light on a post
    x, y = loc[0] - 0.2, loc[1] + 0.15
    sb.prim("cyl", "SLPost", loc=(x, y, 1.15), vertices=16, radius=0.012, depth=0.3, mat=M["steel"])
    cols = [(M["sl_off_r"], 1.3), (M["sl_off_a"], 1.36), (M["sl_green"], 1.42), (M["white"], 1.47)]
    for m, z in cols:
        sb.prim("cyl", "SLSeg", loc=(x, y, z), vertices=32, radius=0.03, depth=0.055, mat=m)


def materials():
    return {
        "white": sb.painted("CellWhite2", (0.76, 0.76, 0.75), rough=0.32, coat=0.3, grime=0.12, wear=0.1),
        "graph": sb.painted("CellGraphite2", (0.05, 0.052, 0.058), rough=0.4, grime=0.1),
        "steel": sb.brushed_metal("CellSteel", (0.62, 0.62, 0.64), rough=0.3),
        "rubber": sb.rubber("CellRubber"),
        "amber": props.amber_anodized("CellAmber"),
        "ui": sb.emit_mat("CellUI", (0.35, 0.45, 0.6), 1.2),
        "status": sb.emit_mat("CellStatus", (0.85, 0.92, 1.0), 6.0),
        "flex": sb.mat("FlexDuct", (0.55, 0.56, 0.58), metal=0.7, rough=0.5),
        "rack_up": sb.painted("RackUp", (0.13, 0.15, 0.19), rough=0.45, grime=0.2, wear=0.25),
        "rack_beam": sb.painted("RackBeam", (0.30, 0.34, 0.40), rough=0.4, grime=0.15, wear=0.3),
        "deck": sb.mat("Deck", (0.45, 0.46, 0.47), metal=0.8, rough=0.45),
        "tote": [sb.painted("ToteG", (0.12, 0.13, 0.15), rough=0.55, grime=0.2),
                 sb.painted("ToteL", (0.42, 0.45, 0.48), rough=0.5, grime=0.2)],
        "carton": sb.painted("Carton", (0.42, 0.32, 0.21), rough=0.85, grime=0.25, scale=6.0),
        "wrap": sb.mat("Wrap", (0.72, 0.74, 0.76), rough=0.25, spec=0.6),
        "sl_off_r": sb.mat("SLRed", (0.25, 0.02, 0.02), rough=0.2),
        "sl_off_a": sb.mat("SLAmb", (0.3, 0.14, 0.02), rough=0.2),
        "sl_green": sb.emit_mat("SLGreen", (0.2, 1.0, 0.45), 6.0),
    }


def build():
    M = materials()
    hall(M)
    return M
