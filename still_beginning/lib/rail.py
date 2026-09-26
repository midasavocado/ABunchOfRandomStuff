"""High-speed electric train (original design: white/graphite, restrained amber accent line, no logos), ballasted
double track with concrete sleepers, UIC-like rails, and overhead catenary (masts, cantilevers, messenger + staggered
contact wire, droppers). Track runs along +X. Units: metres."""
import bpy, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion
import sb
import energy as E

GAUGE = 1.435
RAIL_TOP = 0.95            # rail head height above the formation (ballast shoulder at ~0.6)
CW_H = 5.3                 # contact-wire height above the rail head
TRACK_Y = (-2.25, 2.25)    # track centre lines


# =========================================================================== materials
def train_body_mat(name="TrainBody", Lcar=25.0, nose=False):
    """White clear-coated body; graphite glazing band with window frames; amber accent line under the band;
    door seams; graphite lower skirt. Object coords of the car (x along the car, origin at car centre)."""
    m = sb.mat(name, (0.86, 0.87, 0.88), rough=0.22, coat=0.8, coat_rough=0.05)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    x, z = sep.outputs[0], sep.outputs[2]
    band = nb.math('MULTIPLY', nb.maprange(z, 2.02, 2.06, 0.0, 1.0), nb.maprange(z, 2.86, 2.90, 1.0, 0.0))
    amber = nb.math('MULTIPLY', nb.maprange(z, 1.80, 1.815, 0.0, 1.0), nb.maprange(z, 1.865, 1.88, 1.0, 0.0))
    skirt = nb.maprange(z, 0.92, 0.95, 1.0, 0.0)
    # windows inside the band: 1.6 m pitch with 0.18 m pillars
    wx = nb.math('FRACT', nb.math('DIVIDE', nb.math('ADD', x, 50.0), 1.62))
    pillar = nb.math('MAXIMUM', nb.math('LESS_THAN', wx, 0.06), nb.math('GREATER_THAN', wx, 0.94))
    # doors at +-(L/2 - 3.2): seams + door window
    ax_ = nb.math('ABSOLUTE', x)
    door = nb.math('MULTIPLY', nb.maprange(ax_, Lcar / 2 - 4.05, Lcar / 2 - 4.03, 0.0, 1.0), nb.maprange(ax_, Lcar / 2 - 2.73, Lcar / 2 - 2.71, 1.0, 0.0))
    seam = nb.math('MAXIMUM', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', ax_, Lcar / 2 - 4.04)), 0.0, 0.008, 1.0, 0.0),
                   nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', ax_, Lcar / 2 - 2.72)), 0.0, 0.008, 1.0, 0.0))
    seam = nb.math('MULTIPLY', seam, nb.maprange(z, 0.95, 1.0, 0.0, 1.0))
    body = (0.86, 0.87, 0.88, 1)
    col = nb.mix(skirt, body, (0.05, 0.052, 0.058, 1))
    glass = nb.mix(pillar, (0.015, 0.018, 0.022, 1), (0.045, 0.048, 0.055, 1))
    col = nb.mix(nb.math('MULTIPLY', band, nb.math('SUBTRACT', 1.0, door)), col, glass)
    col = nb.mix(nb.math('MULTIPLY', amber, nb.math('SUBTRACT', 1.0, door)), col, (0.80, 0.34, 0.05, 1))
    col = nb.mix(nb.math('MULTIPLY', seam, 0.8), col, (0.12, 0.12, 0.13, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(band, 0.25, 0.04, dtype='FLOAT'))
    nb.set('Metallic', nb.math('MULTIPLY', amber, 0.6))
    grime = nb.noise(nb.mapping(co, scale=(0.4, 1.0, 3.0)), scale=2.0, detail=6)
    low = nb.maprange(z, 0.9, 1.8, 1.0, 0.0)
    nb.set('Coat Roughness', nb.math('ADD', 0.04, nb.math('MULTIPLY', low, nb.maprange(grime.outputs['Fac'], 0.4, 0.7, 0.0, 0.2))))
    return m


def nose_mat(name="NoseBody"):
    """Nose: same livery; dark windshield on the upper nose, amber line sweeping down to the tip, headlamp slits."""
    m = sb.mat(name, (0.86, 0.87, 0.88), rough=0.22, coat=0.8, coat_rough=0.05)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    x, y, z = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    # nose-local: x = 0 at the tip, increasing toward the car body (0..LN)
    ws = nb.math('MULTIPLY', nb.maprange(x, 3.6, 4.2, 0.0, 1.0), nb.maprange(x, 8.2, 8.8, 1.0, 0.0))
    ws = nb.math('MULTIPLY', ws, nb.maprange(z, 2.55, 2.7, 0.0, 1.0))
    ws = nb.math('MULTIPLY', ws, nb.maprange(nb.math('ABSOLUTE', y), 1.05, 1.2, 1.0, 0.0))
    # amber line descends toward the tip
    zl = nb.math('ADD', 1.84, nb.math('MULTIPLY', nb.maprange(x, 0.0, 9.0, -0.55, 0.0), 1.0))
    amber = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', z, zl)), 0.028, 0.036, 1.0, 0.0)
    lamp = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', z, nb.math('ADD', zl, 0.16))), 0.02, 0.03, 1.0, 0.0),
                   nb.math('MULTIPLY', nb.maprange(x, 1.2, 1.4, 0.0, 1.0), nb.maprange(x, 2.6, 2.8, 1.0, 0.0)))
    skirt = nb.maprange(z, 0.92, 0.95, 1.0, 0.0)
    col = nb.mix(skirt, (0.86, 0.87, 0.88, 1), (0.05, 0.052, 0.058, 1))
    col = nb.mix(ws, col, (0.012, 0.014, 0.018, 1))
    col = nb.mix(amber, col, (0.80, 0.34, 0.05, 1))
    col = nb.mix(lamp, col, (0.9, 0.92, 0.95, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(ws, 0.22, 0.02, dtype='FLOAT'))
    nb.set('Emission Color', (1.0, 0.95, 0.88, 1))
    nb.set('Emission Strength', nb.math('MULTIPLY', lamp, 6.0))
    return m


# =========================================================================== train geometry
def _section(w, zb, zt, n=40, p=4.0, tumble=0.06):
    """Rounded car cross-section (superellipse) in the YZ plane, half width w, bottom zb, top zt."""
    zc = (zb + zt) / 2; h = (zt - zb) / 2
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s = math.cos(a), math.sin(a)
        y = w * np.sign(c) * abs(c) ** (2 / p)
        z = h * np.sign(s) * abs(s) ** (2 / 3.2)
        # tumblehome: narrower toward the roof
        y *= 1.0 - tumble * max(0.0, z / h)
        pts.append((y, zc + z))
    return pts


def car_mesh(name, L=25.0, mat=None, nose_len=0.0, flip=False, nx=60):
    """Car body lofted along X. If nose_len > 0 the +X end becomes an aerodynamic nose (tip at +X)."""
    W, ZB, ZT = 1.5, 0.45, 3.95
    verts, faces = [], []
    xs = list(np.linspace(-L / 2, L / 2 - nose_len, nx // 2))
    if nose_len > 0:
        xs += list(L / 2 - nose_len + nose_len * np.linspace(0, 1, nx)[1:])
    rings = []
    ns = 40
    for x in xs:
        if nose_len > 0 and x > L / 2 - nose_len:
            s = 1.0 - (x - (L / 2 - nose_len)) / nose_len        # 1 at the body, 0 at the tip
            s = max(s, 0.012)
            w = W * (1 - (1 - s) ** 2.4) ** 0.5
            zt = 0.95 + (ZT - 0.95) * (1 - (1 - s) ** 1.7) ** 0.62
            zb = ZB + (0.62 - ZB) * (1 - s) ** 2
            sec = _section(w, zb, max(zt, zb + 0.08), ns)
        else:
            sec = _section(W, ZB, ZT, ns)
        rings.append([len(verts) + i for i in range(ns)])
        for (y, z) in sec:
            verts.append((x, y, z))
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(ns):
            i1 = (i + 1) % ns
            faces.append((a[i], a[i1], b[i1], b[i]))
    faces.append(tuple(rings[0][::-1]))
    tip = len(verts); verts.append((xs[-1] + 0.02, 0.0, (verts[rings[-1][0]][2] + verts[rings[-1][ns // 2]][2]) / 2))
    for i in range(ns):
        faces.append((rings[-1][i], rings[-1][(i + 1) % ns], tip))
    o = E.mesh_np(name, verts, [tuple(f) for f in faces], mat)
    sb.recalc_normals(o)
    if flip:
        o.data.transform(Matrix.Rotation(math.pi, 4, 'Z'))
    return o


def bogie(name, mats):
    """Two-axle bogie: frame, wheelsets (disc wheels, 0.92 m), axle boxes, coil springs, dampers.
    Returns (root, [wheel objects]) - wheels rotate about local Y."""
    steel, dark, wheel_m = mats
    root = sb.empty(name)
    parts = []
    parts.append(E.box(name + "Frame", (3.0, 2.0, 0.28), loc=(0, 0, 0.72), mat=dark, bev=0.03))
    for s_ in (-1, 1):
        parts.append(E.box(name + f"Side{s_}", (3.3, 0.22, 0.42), loc=(0, s_ * 1.02, 0.66), mat=dark, bev=0.03))
        for ax in (-1.25, 1.25):
            parts.append(E.box(name + f"AB{s_}{ax}", (0.36, 0.3, 0.3), loc=(ax, s_ * 1.02, 0.46), mat=steel, bev=0.02))
            pts = [(ax + 0.12 * math.cos(t), s_ * 1.02 + 0.12 * math.sin(t), 0.6 + 0.022 * t) for t in np.linspace(0, 2 * math.pi * 5, 90)]
            parts.append(E.sweep(name + f"Spr{s_}{ax}", pts, radius=0.018, segs=6, mat=wheel_m, sub=1, resample=False))
        parts.append(E.cyl(name + f"Damp{s_}", 0.05, 0.6, loc=(0, s_ * 1.1, 0.62), rot=(0, math.radians(60), 0), mat=steel, verts=12))
    for o in parts:
        o.parent = root
    wheels = []
    for ax in (-1.25, 1.25):
        ws = sb.empty(name + f"WS{ax}", loc=(ax, 0, 0.46), parent=root)
        E.cyl(name + f"Axle{ax}", 0.08, 2.0, rot=(math.pi / 2, 0, 0), mat=steel, verts=16).parent = ws
        for s_ in (-1, 1):
            prof = [(0.0, -0.07), (0.44, -0.07), (0.46, -0.04), (0.46, 0.05), (0.50, 0.06), (0.50, 0.075), (0.43, 0.075), (0.30, 0.03), (0.0, 0.03)]
            w = E.lathe_obj(name + f"Wh{ax}{s_}", prof, segs=64, mat=wheel_m)
            w.rotation_euler = (s_ * math.pi / 2, 0, 0)
            w.location = (0, s_ * GAUGE / 2, 0)
            w.parent = ws
            disc = E.lathe_obj(name + f"Disc{ax}{s_}", [(0.18, -0.02), (0.33, -0.02), (0.33, 0.02), (0.18, 0.02)], segs=48, mat=steel)
            disc.rotation_euler = (math.pi / 2, 0, 0); disc.location = (0, s_ * 0.45, 0); disc.parent = ws
            for k in range(6):
                a = 2 * math.pi * k / 6
                h = E.cyl(name + f"Bolt{ax}{s_}{k}", 0.025, 0.05, loc=(0.24 * math.cos(a), s_ * 0.47, 0.24 * math.sin(a)), rot=(math.pi / 2, 0, 0), mat=dark, verts=8)
                h.parent = ws
        wheels.append(ws)
    return root, wheels


def pantograph(name, mats, height=1.35):
    """Single-arm pantograph on the roof: base frame + insulators, lower arm, upper arm, pan head with carbon strips.
    Origin at the roof mounting point; the pan head top sits at +height."""
    steel, dark, ins_m, carbon = mats
    root = sb.empty(name)
    ps = []
    ps.append(E.box(name + "Base", (1.8, 1.2, 0.12), loc=(0, 0, 0.26), mat=dark, bev=0.02))
    for (x, y) in ((-0.8, -0.5), (-0.8, 0.5), (0.8, -0.5), (0.8, 0.5)):
        prof = [(0.0, 0.0), (0.07, 0.0)] + sum(([(0.07, 0.02 + k * 0.04), (0.1, 0.035 + k * 0.04), (0.07, 0.05 + k * 0.04)] for k in range(4)), []) + [(0.07, 0.2), (0.0, 0.2)]
        ps.append(E.lathe_obj(name + f"Ins{x}{y}", prof, segs=24, mat=ins_m, loc=(x, y, 0.0)))
    knee = V((0.95, 0, height * 0.55)); hinge = V((-0.6, 0, 0.36)); head = V((-0.35, 0, height - 0.05))
    for yy in (-0.12, 0.12):
        ps.append(E.sweep(name + f"Low{yy}", [hinge + V((0, yy, 0)), knee + V((0, yy, 0))], radius=0.04, segs=10, mat=steel, sub=1))
    ps.append(E.sweep(name + "Up", [knee, head], radius=0.028, segs=10, mat=steel, sub=1))
    ps.append(E.sweep(name + "Tie", [hinge + V((0.25, 0, 0)), knee + V((-0.05, 0, -0.05))], radius=0.012, segs=6, mat=steel, sub=1))
    for dx in (-0.12, 0.12):
        ps.append(E.box(name + f"Pan{dx}", (0.07, 1.9, 0.06), loc=(head.x + dx, 0, height - 0.02), mat=steel))
        ps.append(E.box(name + f"Carbon{dx}", (0.05, 1.3, 0.03), loc=(head.x + dx, 0, height + 0.02), mat=carbon))
        for s_ in (-1, 1):
            ps.append(E.sweep(name + f"Horn{dx}{s_}", [(head.x + dx, s_ * 0.95, height - 0.02), (head.x + dx, s_ * 1.1, height - 0.12)], radius=0.02, segs=6, mat=steel, sub=1))
    for o in ps:
        o.parent = root
    return root


def train(name="Train", ncars=8, Lcar=25.0, nose_len=11.0):
    """Returns (root, wheelsets, pantograph_root). Train points toward +X; root at the leading nose tip x,
    rail-head level z = 0. Cars coupled with 0.6 m gaps (diaphragms)."""
    root = sb.empty(name)
    steel = E.stainless(name + "Steel", color=(0.45, 0.45, 0.46), rough=0.4, brushed=False, grime=0.4, scale=0.5)
    dark = E.painted_steel(name + "Dark", (0.035, 0.036, 0.04), rough=0.55, wear=0.3, scale=0.5)
    wheel_m = E.stainless(name + "Wheel", color=(0.30, 0.29, 0.28), rough=0.35, brushed=True, grime=0.5, scale=1.0)
    ins_m = sb.mat(name + "Ins", (0.35, 0.12, 0.06), rough=0.3, coat=0.6)
    carbon = sb.mat(name + "Carbon", (0.02, 0.02, 0.02), rough=0.6)
    bell = sb.rubber(name + "Bellows", (0.03, 0.03, 0.033))
    wheels = []
    x = 0.0
    panto = None
    for k in range(ncars):
        lead = (k == 0); tail = (k == ncars - 1)
        body_m = train_body_mat(name + f"Body{k}", Lcar)
        if lead or tail:
            cm = car_mesh(name + f"Car{k}", L=Lcar, mat=None, nose_len=nose_len, flip=tail)
            cm.data.materials.append(nose_mat(name + f"Nose{k}"))
        else:
            cm = car_mesh(name + f"Car{k}", L=Lcar, mat=body_m)
        cx = x - Lcar / 2
        cm.location = (cx, 0, 0.0)
        cm.parent = root
        # nose material uses nose-local x from the tip; car material for the rest is in car coords:
        if lead or tail:
            # split: nose mat for the whole lead car keeps body livery via a 2nd slot on the parallel section
            cm.data.materials.append(body_m)
            for p in cm.data.polygons:
                xc = sum(cm.data.vertices[v].co.x for v in p.vertices) / len(p.vertices)
                in_nose = (xc < -(Lcar / 2 - nose_len - 0.3)) if tail else (xc > Lcar / 2 - nose_len - 0.3)
                p.material_index = 0 if in_nose else 1
            # remap the nose shader coords: object x of the car -> we give the nose its own object via a driver-free
            # trick: nose shader uses (L/2 - x) through a mapping node
            nm = cm.data.materials[0]
            nt = nm.node_tree
            tc = nt.nodes.get("_tc")
            mp = nt.nodes.new('ShaderNodeMapping')
            mp.inputs['Location'].default_value = (Lcar / 2, 0, 0)
            mp.inputs['Scale'].default_value = (1, 1, 1) if tail else (-1, 1, 1)
            for l in list(tc.outputs['Object'].links):
                to = l.to_socket
                nt.links.remove(l)
                nt.links.new(mp.outputs[0], to)
            nt.links.new(tc.outputs['Object'], mp.inputs[0])
            # the mapping computes -(x) + L/2 ... Blender Mapping: out = (in * scale) rotated + location
        # bogies
        for bx in (-Lcar / 2 + 3.2, Lcar / 2 - 3.2):
            if lead and bx > 0:
                bx = Lcar / 2 - nose_len + 1.0
            if tail and bx < 0:
                bx = -Lcar / 2 + nose_len - 1.0
            br, ws = bogie(name + f"Bg{k}{bx:.0f}", (steel, dark, wheel_m))
            br.location = (cx + bx, 0, 0.0)
            br.parent = root
            wheels += ws
        # diaphragm to the next car
        if not tail:
            d = E.box(name + f"Diaph{k}", (0.9, 2.7, 3.2), loc=(x - Lcar - 0.3, 0, 2.2), mat=bell, bev=0.3)
            d.parent = root
        if k == 1:
            panto = pantograph(name + "Panto", (steel, dark, ins_m, carbon), height=CW_H - 3.95 - 0.02)
            panto.location = (cx - 5.0, 0, 3.95 - 0.02)
            panto.parent = root
            # roof equipment fairing
            E.box(name + "RoofBox", (4.0, 1.6, 0.25), loc=(cx + 3.0, 0, 3.98), mat=dark, bev=0.08).parent = root
        x -= Lcar + 0.6
    return root, wheels, panto


# =========================================================================== track & catenary
def ballast_mat(name="Ballast"):
    m = sb.mat(name, (0.30, 0.29, 0.27), rough=0.9)
    nb = sb.NB(m)
    co = nb.coord('Object')
    v = nb.voronoi(co, scale=22.0, feature='F1')
    v2 = nb.voronoi(co, scale=9.0, feature='F1')
    sepc = nb.new('ShaderNodeSeparateColor'); nb.link(v.outputs['Color'], sepc.inputs[0])
    h = nb.math('ADD', nb.maprange(v.outputs['Distance'], 0.0, 0.6, 1.0, 0.0), nb.math('MULTIPLY', nb.maprange(v2.outputs['Distance'], 0, 0.6, 1, 0), 0.5))
    nb.set('Normal', nb.bump(h, strength=0.9, distance=0.03))
    c = nb.mix(sepc.outputs[0], (0.20, 0.19, 0.18, 1), (0.42, 0.40, 0.37, 1))
    rust = nb.noise(co, scale=0.8, detail=4)
    c = nb.mix(nb.maprange(rust.outputs['Fac'], 0.5, 0.7, 0.0, 0.35), c, (0.28, 0.18, 0.10, 1))
    nb.set('Base Color', c)
    return m


def rail_profile():
    # simplified flat-bottom rail (UIC60-like), in (y, z) with z=0 at the rail head top; closed loop
    return [(-0.075, -0.172), (0.075, -0.172), (0.075, -0.16), (0.012, -0.145), (0.009, -0.05), (0.036, -0.035),
            (0.036, -0.004), (0.028, 0.0), (-0.028, 0.0), (-0.036, -0.004), (-0.036, -0.035), (-0.009, -0.05),
            (-0.012, -0.145), (-0.075, -0.16)]


def track(name, x0, x1, zfn, sleeper_range=None):
    """Double track from x0 to x1 following height zfn(x) (formation level). Rails as swept profiles, one merged
    sleeper mesh over sleeper_range, ballast bed prism."""
    root = sb.empty(name)
    rail_m = E.stainless(name + "Rail", color=(0.42, 0.40, 0.38), rough=0.45, brushed=False, grime=0.6, scale=0.3)
    head_m = E.stainless(name + "RailHead", color=(0.72, 0.72, 0.73), rough=0.18, brushed=True, dir_scale=(80, 1, 1))
    conc = sb.concrete(name + "Sleeper", (0.52, 0.51, 0.48), scale=3.0, stains=0.4)
    bal = ballast_mat(name + "Ballast")
    xs = np.linspace(x0, x1, int((x1 - x0) / 4) + 1)
    # ballast prism (both tracks) with shoulders
    prof = [(-7.0, -0.3), (-4.6, 0.62), (4.6, 0.62), (7.0, -0.3)]
    verts, faces = [], []
    for i, x in enumerate(xs):
        z = zfn(x)
        for (y, zz) in prof:
            verts.append((x, y, z + zz))
    m = len(prof)
    for i in range(len(xs) - 1):
        for j in range(m - 1):
            faces.append((i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1))
    bo = E.mesh_np(name + "Bed", verts, np.array(faces), bal); bo.parent = root
    sb.recalc_normals(bo)
    for ty in TRACK_Y:
        for s_ in (-1, 1):
            yy = ty + s_ * (GAUGE / 2 + 0.036)
            P = np.array([(x, yy, zfn(x) + RAIL_TOP) for x in xs])
            r = E.sweep_planar(name + f"Rail{ty}{s_}", P, [(zz, y) for (y, zz) in rail_profile()], normal=(0, 0, 1) if False else (0, 0, 1),
                               closed=False, mat=rail_m)
            # sweep_planar maps profile (a along normal, b along T x normal): we want (z along Z, y across)
            r.parent = root
            hd = E.sweep_planar(name + f"Head{ty}{s_}", P + np.array([0, 0, 0.0005]), [(0.0, -0.028), (0.0, 0.028), (-0.012, 0.036), (-0.012, -0.036)],
                                normal=(0, 0, 1), closed=False, mat=head_m)
            hd.parent = root
    if sleeper_range:
        sx0, sx1 = sleeper_range
        n = int((sx1 - sx0) / 0.6)
        verts, faces = [], []
        for ty in TRACK_Y:
            for k in range(n):
                x = sx0 + k * 0.6
                z = zfn(x) + RAIL_TOP - 0.172
                b = len(verts)
                for (dx, dy, dz) in ((-0.13, -1.3, -0.22), (0.13, -1.3, -0.22), (0.13, 1.3, -0.22), (-0.13, 1.3, -0.22),
                                     (-0.11, -1.28, 0.0), (0.11, -1.28, 0.0), (0.11, 1.28, 0.0), (-0.11, 1.28, 0.0)):
                    verts.append((x + dx, ty + dy, z + dz))
                faces += [(b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5),
                          (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)]
        so = E.mesh_np(name + "Sleepers", verts, np.array(faces), conc, smooth=False); so.parent = root
    return root


def catenary(name, x0, x1, zfn, spacing=60.0, detail_range=None):
    """Masts on both outer sides with cantilevers to each track; messenger + contact wire (staggered +-0.3 m at
    alternate masts) with droppers every ~9 m. Returns root."""
    root = sb.empty(name)
    galv = E.painted_steel(name + "Galv", (0.48, 0.49, 0.5), rough=0.45, wear=0.3, scale=0.5)
    wire_m = E.copper(name + "Wire", rough=0.35, tarnish=0.8)
    wire_m.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.35, 0.22, 0.14, 1)
    ins_m = sb.mat(name + "Ins", (0.3, 0.1, 0.05), rough=0.3, coat=0.5)
    xs = np.arange(x0, x1 + 1, spacing)
    # H-section mast built once (origin at its foot) and placed as linked copies: bpy.ops per mast piece re-evaluates
    # the whole (large) scene each time
    h = CW_H + RAIL_TOP + 2.4
    parts = [E.box(name + "MSrcF%d" % k, (0.26, 0.02, h), loc=(0, dy, h / 2), mat=galv) for k, dy in enumerate((-0.11, 0.11))]
    parts.append(E.box(name + "MSrcW", (0.02, 0.22, h), loc=(0, 0, h / 2), mat=galv))
    for o_ in parts:
        o_.data.transform(o_.matrix_basis); o_.matrix_basis = Matrix.Identity(4)
    mast_src = sb.join(parts, name + "MastSrc")
    mast_src.hide_render = True; mast_src.hide_viewport = True
    for i, x in enumerate(xs):
        z = zfn(x)
        det = detail_range is None or (detail_range[0] <= x <= detail_range[1])
        for side in (-1, 1):
            my = side * 6.2
            mo = mast_src.copy(); sb.link_obj(mo)
            mo.name = name + f"Mast{i}{side}"
            mo.hide_render = False; mo.hide_viewport = False
            mo.location = (x, my, z)
            mo.parent = root
            if not det:
                continue
            ty = TRACK_Y[0] if side < 0 else TRACK_Y[1]
            stag = 0.3 * (1 if i % 2 == 0 else -1)
            # cantilever: upper tube to the messenger support, lower tube to the registration arm
            top = V((x, my, z + RAIL_TOP + CW_H + 1.35)); low = V((x, my, z + RAIL_TOP + CW_H - 0.25))
            msg = V((x, ty, z + RAIL_TOP + CW_H + 1.2)); reg = V((x, ty + stag, z + RAIL_TOP + CW_H))
            E.sweep(name + f"Cu{i}{side}", [top, msg + V((0, 0, 0.05))], radius=0.03, segs=8, mat=galv, sub=1).parent = root
            E.sweep(name + f"Cl{i}{side}", [low, V((x, ty + stag - side * 0.6, reg.z + 0.25))], radius=0.028, segs=8, mat=galv, sub=1).parent = root
            E.sweep(name + f"Rg{i}{side}", [V((x, ty + stag - side * 0.6, reg.z + 0.25)), reg + V((0, 0, 0.02))], radius=0.012, segs=6, mat=galv, sub=1).parent = root
            E.sweep(name + f"St{i}{side}", [V((x, ty + stag - side * 0.6, reg.z + 0.25)), msg], radius=0.012, segs=6, mat=galv, sub=1).parent = root
            for p in (top, low):
                E.lathe_obj(name + f"Is{i}{side}{p.z:.1f}", [(0.0, 0.0), (0.05, 0.0), (0.08, 0.05), (0.05, 0.1), (0.08, 0.15), (0.05, 0.2), (0.08, 0.25), (0.05, 0.3), (0.0, 0.3)],
                            segs=16, mat=ins_m, loc=tuple(p + V((0, -side * 0.35, 0))), rot=(math.pi / 2, 0, 0)).parent = root
    # wires per track: messenger sags between masts; contact wire straight with stagger
    for t_i, ty in enumerate(TRACK_Y):
        side = -1 if t_i == 0 else 1
        mpts, cpts = [], []
        for i in range(len(xs) - 1):
            xa, xb = xs[i], xs[i + 1]
            sa = 0.3 * (1 if i % 2 == 0 else -1); sb_ = -sa
            for k in range(12):
                u = k / 12
                x = xa + (xb - xa) * u
                z = zfn(x) + RAIL_TOP
                mpts.append((x, ty, z + CW_H + 1.2 - 0.9 * 4 * u * (1 - u)))
                cpts.append((x, ty + sa + (sb_ - sa) * u, z + CW_H))
        mpts.append((xs[-1], ty, zfn(xs[-1]) + RAIL_TOP + CW_H + 1.2)); cpts.append((xs[-1], ty + 0.3 * (1 if (len(xs) - 1) % 2 == 0 else -1), zfn(xs[-1]) + RAIL_TOP + CW_H))
        E.sweep(name + f"Msg{t_i}", mpts, radius=0.0065, segs=6, mat=wire_m, sub=1, resample=False, caps=False).parent = root
        E.sweep(name + f"Con{t_i}", cpts, radius=0.0062, segs=6, mat=wire_m, sub=1, resample=False, caps=False).parent = root
        if detail_range:
            for k in range(len(mpts)):
                x = mpts[k][0]
                if detail_range[0] <= x <= detail_range[1] and k % 2 == 1:
                    E.sweep(name + f"Dr{t_i}{k}", [mpts[k], cpts[k]], radius=0.0025, segs=4, mat=wire_m, sub=1, resample=False, caps=False).parent = root
    return root
