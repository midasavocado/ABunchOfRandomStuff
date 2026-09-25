"""s11 NEW ENERGY: speculative fusion research chamber. Tactile port-flange edge -> through the port duct ->
the luminous toroidal plasma. Ends looking straight down the torus axis with the plasma ring centred
(outer edge ~0.40 of frame height) to match-cut to the round turbine hub in s12a."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb
import numpy as np
import energy as E
import timeline as TL
from mathutils import Vector as V, Matrix, Quaternion

sid = (sb.argv() or ["s11"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=float(os.environ.get("EXPO", "0.0")),
                volumes=True, vol_tile=4, vol_samples=48)

RV = 8.5                      # vessel inner radius (sphere centred at origin)
R0, A, KAP, DEL = 3.1, 0.95, 1.6, 0.35     # plasma major radius, minor radius, elongation, triangularity
PORT_EL = math.radians(42.0)
U = V((-math.cos(PORT_EL), 0.0, math.sin(PORT_EL)))    # port axis (outward)
PORT_R = 0.95
H_END = 7.0                   # final camera height above the plasma midplane (on axis)
rnd = random.Random(11)


# =========================================================================== materials
def tile_mat():
    """Tungsten armour tiles: per-tile variation (face attribute 'rnd'), deposition darkening low in the vessel,
    fine machining, slightly bright bevels."""
    m = sb.mat("Tungsten", (0.42, 0.41, 0.40), metal=0.9, rough=0.55)
    nb = sb.NB(m)
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "rnd"; at.attribute_type = 'GEOMETRY'
    r = at.outputs['Fac']
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    n = nb.noise(co, scale=3.0, detail=6, rough=0.6)
    fine = nb.noise(co, scale=180, detail=4)
    low = nb.maprange(sep.outputs[2], -3.0, 1.0, 1.0, 0.0)      # deposition toward the divertor
    dep = nb.math('MULTIPLY', low, nb.maprange(n.outputs['Fac'], 0.35, 0.7, 0.2, 0.9))
    base = nb.mix(nb.maprange(r, 0, 1, 0, 1), (0.26, 0.255, 0.25, 1), (0.50, 0.49, 0.47, 1))
    base = nb.mix(dep, base, (0.10, 0.09, 0.085, 1))
    # rare heat-tinted tiles
    tint = nb.maprange(r, 0.93, 0.95, 0.0, 0.6)
    base = nb.mix(tint, base, (0.42, 0.30, 0.22, 1))
    nb.set('Base Color', base)
    nb.set('Roughness', nb.math('ADD', nb.maprange(r, 0, 1, 0.50, 0.68), nb.math('MULTIPLY', dep, 0.2)))
    nb.set('Normal', nb.bump(fine.outputs['Fac'], strength=0.05, distance=0.0005))
    return m


def plasma_mat(name, strength, core_pow=1.4, halo=False, q=2.4):
    """Additive emissive shell: brightness follows chord length (facing), colour from warm amber edge
    to pale peach-white core. Field-aligned striations twist along the torus (safety factor q)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB); nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    geo = nb.new('ShaderNodeNewGeometry')
    facing = nb.math('ABSOLUTE', nb.vmath('DOT_PRODUCT', geo.outputs['Normal'], geo.outputs['Incoming']))
    c = nb.math('POWER', facing, core_pow)
    col = nb.ramp(c, [(0.0, (1.0, 0.36, 0.07)), (0.35, (1.0, 0.56, 0.24)), (0.7, (1.0, 0.76, 0.56)), (1.0, (1.0, 0.90, 0.80))])
    # field-line striations: noise in (poloidal - q * toroidal) angle
    co = geo.outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    phi = nb.math('ARCTAN2', sep.outputs[1], sep.outputs[0])
    rho = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', sep.outputs[0], sep.outputs[0]), nb.math('MULTIPLY', sep.outputs[1], sep.outputs[1])))
    th = nb.math('ARCTAN2', sep.outputs[2], nb.math('SUBTRACT', rho, R0))
    fl = nb.math('SUBTRACT', th, nb.math('MULTIPLY', phi, q))
    tval = nb.new('ShaderNodeValue'); tval.name = "Time"
    cmb = nb.new('ShaderNodeCombineXYZ')
    nb.link(nb.math('ADD', nb.math('MULTIPLY', nb.math('SINE', fl), 2.0), 0.0), cmb.inputs[0])
    nb.link(nb.math('ADD', nb.math('MULTIPLY', nb.math('COSINE', fl), 2.0), 0.0), cmb.inputs[1])
    nb.link(nb.math('ADD', nb.math('MULTIPLY', phi, 0.6), tval.outputs[0]), cmb.inputs[2])
    st = nb.noise(cmb.outputs[0], scale=2.2, detail=3, rough=0.55)
    mod = nb.maprange(st.outputs['Fac'], 0.3, 0.7, 0.78, 1.22)
    s = nb.math('MULTIPLY', nb.math('ADD', 0.12, c), strength)
    s = nb.math('MULTIPLY', s, mod)
    em = nb.new('ShaderNodeEmission'); nb.link(col, em.inputs[0]); nb.link(s, em.inputs[1])
    tr = nb.new('ShaderNodeBsdfTransparent')
    add = nb.new('ShaderNodeAddShader'); nb.link(em.outputs[0], add.inputs[0]); nb.link(tr.outputs[0], add.inputs[1])
    out = nb.new('ShaderNodeOutputMaterial'); nb.link(add.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = True
    try:
        m.use_transparency_overlap = True
    except Exception:
        pass
    return m, tval


# =========================================================================== geometry
def plasma_torus(name, a_scale, mat, nt=192, npol=64):
    verts, faces = [], []
    for i in range(nt):
        ph = 2 * math.pi * i / nt
        cph, sph = math.cos(ph), math.sin(ph)
        for j in range(npol):
            th = 2 * math.pi * j / npol
            r = R0 + A * a_scale * math.cos(th + DEL * math.sin(th))
            z = KAP * A * a_scale * math.sin(th)
            verts.append((r * cph, r * sph, z))
    for i in range(nt):
        i1 = (i + 1) % nt
        for j in range(npol):
            j1 = (j + 1) % npol
            faces.append((i * npol + j, i1 * npol + j, i1 * npol + j1, i * npol + j1))
    o = E.mesh_np(name, verts, np.array(faces), mat)
    sb.recalc_normals(o)
    o.visible_shadow = False
    return o


def sphere_tiles(name, R, tile=0.24, gap=0.014, th=0.03, lat0=-62, lat1=80, holes=(), mat=None, bevel=0.0035):
    """Armour tiles on the inside of a sphere (radius R), rows by latitude. holes: list of (unit_axis, half_angle)."""
    verts, faces, rnds = [], [], []
    dlat = tile / R
    lat = math.radians(lat0)
    while lat < math.radians(lat1):
        la0, la1 = lat, lat + dlat
        n = max(6, int(round(2 * math.pi * R * math.cos(lat + dlat / 2) / tile)))
        off = rnd.random() * 2 * math.pi / n
        for k in range(n):
            lo0 = off + 2 * math.pi * k / n; lo1 = off + 2 * math.pi * (k + 1) / n
            cmid = V((math.cos((la0 + la1) / 2) * math.cos((lo0 + lo1) / 2), math.cos((la0 + la1) / 2) * math.sin((lo0 + lo1) / 2), math.sin((la0 + la1) / 2)))
            skip = False
            for ax, ang in holes:
                if cmid.angle(ax) < ang:
                    skip = True; break
            if skip:
                continue
            # corners (inset by the gap), base on the sphere, top face inward by th with a chamfer
            g_la = gap / R / 2; g_lo = gap / (R * max(0.2, math.cos((la0 + la1) / 2))) / 2
            cs = []
            for (la, lo) in ((la0 + g_la, lo0 + g_lo), (la0 + g_la, lo1 - g_lo), (la1 - g_la, lo1 - g_lo), (la1 - g_la, lo0 + g_lo)):
                cs.append(V((math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))))
            b = len(verts)
            for c in cs:
                verts.append(tuple(c * R))
            ctr = sum(cs, V()) / 4
            h = th * (0.85 + 0.3 * rnd.random())
            for c in cs:
                ci = c.lerp(ctr, bevel / tile * 2)
                verts.append(tuple(ci.normalized() * (R - h)))
            faces += [(b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5),
                      (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)]
            r = rnd.random()
            rnds += [r] * 5
        lat += dlat
    o = E.mesh_np(name, verts, np.array(faces), mat, smooth=False)
    sb.recalc_normals(o, inside=False)
    # normals must face the sphere centre (inward): flip if the top face points outward
    me = o.data
    p0 = me.polygons[0]
    if p0.normal.dot(V(me.vertices[p0.vertices[0]].co)) > 0:
        sb.recalc_normals(o, inside=True)
    at = me.attributes.new("rnd", 'FLOAT', 'FACE')
    at.data.foreach_set("value", np.array(rnds, np.float32))
    return o


def sphere_shell(name, R, holes=(), mat=None, nlat=96, nlon=192, lat0=-90, lat1=90):
    verts, faces = [], []
    for i in range(nlat + 1):
        la = math.radians(lat0 + (lat1 - lat0) * i / nlat)
        for j in range(nlon):
            lo = 2 * math.pi * j / nlon
            verts.append((R * math.cos(la) * math.cos(lo), R * math.cos(la) * math.sin(lo), R * math.sin(la)))
    for i in range(nlat):
        for j in range(nlon):
            j1 = (j + 1) % nlon
            q = (i * nlon + j, i * nlon + j1, (i + 1) * nlon + j1, (i + 1) * nlon + j)
            c = sum((V(verts[k]) for k in q), V()) / 4
            if any(c.normalized().angle(ax) < ang for ax, ang in holes):
                continue
            faces.append(q)
    o = E.mesh_np(name, verts, np.array(faces), mat)
    sb.recalc_normals(o)
    return o


def ring_obj(name, R, z, w, h, mat, segs=256):
    """Rectangular-section ring (e.g. PF coil case) about the Z axis."""
    prof = E.rounded_rect(w, h, min(w, h) * 0.08, 3)
    P = np.array([(R * math.cos(2 * math.pi * i / segs), R * math.sin(2 * math.pi * i / segs), z) for i in range(segs)])
    return E.sweep_planar(name, P, [(b, a) for a, b in prof], normal=(0, 0, 1), closed=True, mat=mat)


def build():
    tiles_m = tile_mat()
    ss = E.stainless("VesselSS", color=(0.55, 0.55, 0.56), rough=0.3, brushed=False, grime=0.3, scale=0.2)
    ss_m = E.stainless("Machined", color=(0.66, 0.66, 0.67), rough=0.2, brushed=True, dir_scale=(1, 1, 40))
    dark = E.anodized("DarkSteel", (0.05, 0.05, 0.055), rough=0.45)
    amber = __import__("props").amber_anodized("AmberRing")
    cu = E.copper("CuPF", rough=0.3, tarnish=0.3)
    bolt_m = E.zinc("ZincF", rough=0.25)
    glass = sb.mat("Viewport", (0.9, 0.95, 1.0), rough=0.02, transmission=1.0, ior=1.5)
    # ---- ports: the entry port (camera) + diagnostic ports around the sphere
    ports = [(U, PORT_R)]
    diag = []
    for k in range(14):
        el = math.radians(rnd.choice((-18, 8, 25, 38)))
        az = math.radians(20 + k * 360 / 14 + rnd.uniform(-6, 6))
        ax = V((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        if ax.angle(U) < 0.5:
            continue
        r = rnd.choice((0.35, 0.45, 0.6))
        diag.append((ax, r))
    holes = [(ax, (r + 0.08) / RV) for ax, r in ports + diag]
    sphere_tiles("FirstWall", RV, holes=holes, mat=tiles_m)
    shell = sphere_shell("VesselIn", RV + 0.02, holes=[(ax, (r + 0.02) / RV) for ax, r in ports + diag], mat=dark)
    outer = sphere_shell("VesselOut", RV + 0.35, holes=[(ax, (r + 0.02) / (RV + 0.35)) for ax, r in ports + diag], mat=ss, nlat=128, nlon=256)
    # port ducts (tubes through the wall) + flanges
    bolt = E.hex_bolt_mesh("PBolt", d=0.03, mats=[bolt_m]); bolt.hide_render = True
    for n_, (ax, r) in enumerate(ports + diag):
        inner = RV - 0.05
        L = (RV + 1.6 if n_ == 0 else RV + 0.9) - inner
        c = ax * (inner + L / 2)
        duct = sb.prim("cyl", f"Duct{n_}", loc=tuple(c), vertices=96, radius=r + 0.03, depth=L, mat=ss)
        duct.rotation_mode = 'QUATERNION'; duct.rotation_quaternion = ax.to_track_quat('Z', 'Y')
        sb.solidify(duct, 0.03, offset=1)
        # remove caps: use a lathe instead for clean open tube
        bpy.data.objects.remove(duct)
        tube = sb.lathe(f"Duct{n_}", [(r + 0.06, 0), (r + 0.06, L), (r, L), (r, 0)], segs=128, mat=(dark if n_ == 0 else ss))
        if n_ == 0:
            # entry port: stiffening ribs + a bellows section inside the duct (reads as engineered, not a plain pipe)
            for k in range(int(L / 0.22)):
                z = 0.15 + k * 0.22
                rb = sb.lathe(f"Rib{k}", [(r + 0.001, z), (r - 0.035, z + 0.01), (r - 0.035, z + 0.05), (r + 0.001, z + 0.06)], segs=128, mat=ss_m)
                rb.rotation_mode = 'QUATERNION'; rb.rotation_quaternion = ax.to_track_quat('Z', 'Y')
                rb.location = tuple(ax * inner)
        tube.rotation_mode = 'QUATERNION'; tube.rotation_quaternion = ax.to_track_quat('Z', 'Y')
        tube.location = tuple(ax * inner)
        # flange at the outer end: stepped, machined sealing face, bolt circle, amber anodized seal carrier
        fr = r + 0.32 if n_ == 0 else r + 0.18
        ft = 0.14 if n_ == 0 else 0.08
        prof = [(r, 0), (fr, 0), (fr, ft * 0.9), (fr - 0.01, ft), (r + 0.1, ft), (r + 0.1, ft + 0.005), (r + 0.02, ft + 0.005), (r, ft)]
        fl = E.lathe_obj(f"Flange{n_}", prof, segs=160, mat=ss_m)
        fl.rotation_mode = 'QUATERNION'; fl.rotation_quaternion = ax.to_track_quat('Z', 'Y')
        fl.location = tuple(ax * (inner + L - ft))
        if n_ == 0:
            ar = E.lathe_obj("AmberSeal", [(r + 0.035, 0), (r + 0.075, 0), (r + 0.075, 0.012), (r + 0.035, 0.012)], segs=160, mat=amber)
            ar.rotation_mode = 'QUATERNION'; ar.rotation_quaternion = ax.to_track_quat('Z', 'Y')
            ar.location = tuple(ax * (inner + L + 0.001))
        nb_ = 36 if n_ == 0 else 16
        q = ax.to_track_quat('Z', 'Y')
        for k in range(nb_):
            a = 2 * math.pi * k / nb_
            lp = q @ V(((fr - (0.09 if n_ == 0 else 0.05)) * math.cos(a), (fr - (0.09 if n_ == 0 else 0.05)) * math.sin(a), 0))
            E.place_linked(bolt, tuple(ax * (inner + L) + lp), direction=tuple(ax), spin=k * 0.37,
                           scale=1.0 if n_ == 0 else 0.6)
        if n_ > 0:
            # diagnostic viewport glass + instrument head on some ports
            g = sb.prim("cyl", f"Glass{n_}", vertices=64, radius=r, depth=0.03, mat=glass)
            g.rotation_mode = 'QUATERNION'; g.rotation_quaternion = q; g.location = tuple(ax * (inner + L - 0.2))
            if rnd.random() < 0.6:
                ih = sb.prim("cyl", f"Inst{n_}", vertices=48, radius=r * 0.7, depth=0.8, mat=dark)
                ih.rotation_mode = 'QUATERNION'; ih.rotation_quaternion = q; ih.location = tuple(ax * (inner + L + 0.45))
    # ---- floor of the chamber, central column, divertor, PF coils, supports
    floor_z = -4.6
    fz_r = math.sqrt(RV ** 2 - floor_z ** 2)
    fl_tiles = sphere_tiles  # (floor uses a flat tiled disc)
    verts, faces, rr = [], [], []
    ring_r = 1.3
    while ring_r < fz_r:
        n = int(2 * math.pi * ring_r / 0.26)
        for k in range(n):
            a0 = 2 * math.pi * k / n + 0.004; a1 = 2 * math.pi * (k + 1) / n - 0.004
            r0_, r1_ = ring_r + 0.006, ring_r + 0.25
            b = len(verts)
            for (rr_, aa) in ((r0_, a0), (r0_, a1), (r1_, a1), (r1_, a0)):
                verts.append((rr_ * math.cos(aa), rr_ * math.sin(aa), floor_z))
            for (rr_, aa) in ((r0_ + 0.01, a0 + 0.002), (r0_ + 0.01, a1 - 0.002), (r1_ - 0.01, a1 - 0.002), (r1_ - 0.01, a0 + 0.002)):
                verts.append((rr_ * math.cos(aa), rr_ * math.sin(aa), floor_z + 0.03))
            faces += [(b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5), (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)]
            rr += [rnd.random()] * 5
        ring_r += 0.26
    fo = E.mesh_np("FloorTiles", verts, np.array(faces), tiles_m, smooth=False)
    sb.recalc_normals(fo)
    at = fo.data.attributes.new("rnd", 'FLOAT', 'FACE'); at.data.foreach_set("value", np.array(rr, np.float32))
    E.cyl("FloorBase", fz_r + 0.2, 0.1, loc=(0, 0, floor_z - 0.05), mat=dark, verts=128)
    # central column (solenoid) with tiled armour and a machined top cap with a bolt circle
    col_top = 2.3
    E.cyl("Column", 1.12, col_top - floor_z, loc=(0, 0, (col_top + floor_z) / 2), mat=tiles_m, verts=128)
    colm = bpy.data.objects["Column"]
    at = colm.data.attributes.new("rnd", 'FLOAT', 'FACE'); at.data.foreach_set("value", np.random.RandomState(3).rand(len(colm.data.polygons)).astype(np.float32))
    cap = E.lathe_obj("ColCap", [(0, 0.0), (1.18, 0.0), (1.18, 0.10), (1.14, 0.14), (0.9, 0.14), (0.9, 0.18), (0.35, 0.18), (0.30, 0.26), (0.0, 0.26)], segs=192, mat=ss_m, loc=(0, 0, col_top))
    for k in range(48):
        a = 2 * math.pi * k / 48
        E.place_linked(bolt, (1.02 * math.cos(a), 1.02 * math.sin(a), col_top + 0.14), direction=(0, 0, 1), spin=k * 0.5, scale=0.8)
    E.lathe_obj("ColAmber", [(0.9, 0.0), (0.93, 0.0), (0.93, 0.012), (0.9, 0.012)], segs=192, mat=amber, loc=(0, 0, col_top + 0.18))
    # lower divertor: angled cassette ring below the plasma
    dv = []
    for k in range(54):
        a = 2 * math.pi * k / 54
        c = V((3.1 * math.cos(a), 3.1 * math.sin(a), -2.85))
        cas = sb.prim("cube", f"Div{k}", loc=tuple(c), scale=(0.95, 0.17, 0.35), mat=tiles_m)
        cas.rotation_euler = (0, math.radians(18), a)
        dv.append(cas)
        at = cas.data.attributes.new("rnd", 'FLOAT', 'FACE'); at.data.foreach_set("value", np.full(6, rnd.random(), np.float32))
    # PF coils (copper windings in steel cases) + support posts
    for (R, z, w, h) in ((6.4, -3.3, 0.55, 0.7), (6.6, 3.9, 0.5, 0.6), (5.2, -3.9, 0.4, 0.5)):
        ring_obj(f"PF{R}{z}", R, z, w, h, ss)
        ring_obj(f"PFcu{R}{z}", R, z, w * 0.8, h + 0.02, cu, segs=256)
    for k in range(12):
        a = 2 * math.pi * k / 12 + 0.13
        for (R, z0, z1) in ((6.4, floor_z, -3.65),):
            E.box(f"Post{k}", (0.22, 0.22, z1 - z0), loc=(R * math.cos(a), R * math.sin(a), (z0 + z1) / 2), mat=dark, bev=0.02)
    # cable trays / cryo lines hugging the wall at two latitudes
    for el in (-12.0, 30.0):
        la = math.radians(el)
        P = [(math.cos(la) * (RV - 0.35) * math.cos(2 * math.pi * i / 200), math.cos(la) * (RV - 0.35) * math.sin(2 * math.pi * i / 200), math.sin(la) * (RV - 0.35)) for i in range(200)]
        E.sweep_planar(f"Line{el}", np.array(P), [(0.06 * math.cos(t), 0.06 * math.sin(t)) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)],
                       normal=(0, 0, 1), closed=True, mat=ss_m)
    # ---- plasma: three nested additive shells + point lights on the magnetic axis
    mats = []
    ps = float(os.environ.get("PS", "1.0"))
    m_core, t1 = plasma_mat("PlasmaCore", 2.4 * ps, core_pow=3.0)
    m_mid, t2 = plasma_mat("PlasmaMid", 0.7 * ps, core_pow=1.8)
    m_halo, t3 = plasma_mat("PlasmaHalo", 0.22 * ps, core_pow=0.7)
    plasma_torus("PlasmaCore", 0.42, m_core)
    plasma_torus("PlasmaMid", 0.78, m_mid)
    plasma_torus("PlasmaHalo", 1.12, m_halo)
    for tv in (t1, t2, t3):
        sb.bake_socket(None, tv.outputs[0], F0, F1, lambda f: f * 0.035)
    NL = 16
    try:
        sc.eevee.shadow_pool_size = '1024'
        sc.eevee.shadow_resolution_scale = 0.5
    except Exception:
        pass
    for k in range(NL):
        a = 2 * math.pi * k / NL
        l = sb.light('POINT', f"PL{k}", loc=(R0 * math.cos(a), R0 * math.sin(a), 0.0), energy=float(os.environ.get("PLE", "180")),
                     color=(1.0, 0.70, 0.48), size=0.8)
    # outside: dark machine hall, dim warm work light on the flange, cool fill
    sb.world_color((0.004, 0.0045, 0.006), 1.0)
    fl_c = U * (RV + 1.6)
    wl = sb.light('AREA', "Work", loc=tuple(fl_c + V((0.5, -2.5, 1.5))), target=tuple(fl_c), energy=120, color=(1.0, 0.78, 0.55), size=1.5)
    cf = sb.light('AREA', "CoolFill", loc=tuple(fl_c + V((-2.0, 2.5, 2.0))), target=tuple(fl_c), energy=60, color=(0.6, 0.72, 1.0), size=3.0)
    cf.visible_glossy = False
    E.softbox("FlangeSB", tuple(fl_c + V((0.0, 3.0, 0.0)) + U * 1.5), tuple(fl_c), size=(3.0, 0.3), strength=4.0, color=(1.0, 0.85, 0.7))
    E.sphere_probe((0, 0, 1.0), radius=9.0)
    E.sphere_probe(tuple(fl_c * 1.03), radius=2.5, name="ProbePort")
    # faint haze inside the vessel so the plasma light has body (kept very thin)
    hz = sb.prim("ico", "Haze", subdivisions=3, radius=RV - 0.4)
    hz.visible_shadow = False
    hz.data.materials.append(sb.volume_mat("HazeV", density=float(os.environ.get("HAZE", "0.0015")), color=(1, 0.95, 0.9), anisotropy=0.3))


def cam_bake_up(cam, f0, f1, pos_fn, fwd_fn, up_ref=(1, 0, 0), lens_fn=None, focus_fn=None):
    for f in range(f0, f1 + 1):
        t = (f - f0) / max(1, f1 - f0)
        p = V(pos_fn(t)); fw = V(fwd_fn(t)).normalized()
        up = V(up_ref) - fw * fw.dot(V(up_ref)); up.normalize()
        x = fw.cross(up).normalized(); y = x.cross(fw)
        M = Matrix((tuple(x), tuple(y), tuple(-fw))).transposed()
        cam.location = p; cam.rotation_mode = 'QUATERNION'; cam.rotation_quaternion = M.to_quaternion()
        cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_quaternion", frame=f)
        if lens_fn:
            cam.data.lens = lens_fn(t); cam.data.keyframe_insert("lens", frame=f)
        if focus_fn:
            cam.data.dof.focus_distance = focus_fn(t, p, fw); cam.data.dof.keyframe_insert("focus_distance", frame=f)


build()
cam = E.dof_cam("Cam", 20, 2.8, clip=(0.02, 200))
sc.eevee.bokeh_max_size = 200
side = V((0, 1, 0))
upv = U.cross(side).normalized()        # 'up' across the port face
P0 = U * (RV + 2.25) + side * 0.95 + upv * 0.55        # just outside, beside the flange rim
P1 = U * (RV + 1.2) + side * 0.15 + upv * 0.1
P2 = U * (RV - 1.2)
P3 = V((-2.6, 0.0, H_END + 0.25))
P4 = V((0.0, 0.0, H_END))
pts = [P0, P1, P2, P3, P4]


def timing(t):
    # slow tactile creep at the flange, accelerate through the duct, decelerate onto the axis
    return sb.smoother(t) * 0.85 + t * 0.15


def pos(t):
    return sb.catmull(pts, timing(t))


DOWN = V((0, 0, -1))


def fwd(t):
    k = sb.smoother(sb.remap(timing(t), 0.45, 1.0))
    a = -U
    # look slightly across the flange at the start, then inward, then straight down
    a = (a + side * -0.18 * (1 - sb.smooth(sb.remap(t, 0.0, 0.3)))).normalized()
    q = a.rotation_difference(DOWN)
    return Quaternion().slerp(q, k) @ a


def lens(t):
    return sb.lerp(20.0, 14.0, sb.smooth(sb.remap(timing(t), 0.3, 1.0)))


def focus(t, p, fw):
    # near: the flange rim; far: the plasma midplane
    k = sb.smooth(sb.remap(t, 0.08, 0.4))
    near = (U * (RV + 1.6) + side * 0.95 - p).dot(fw)
    far = max(3.0, H_END + 0.0)
    return sb.lerp(max(0.3, near), far, k)


cam_bake_up(cam, F0, F1, pos, fwd, up_ref=(1, 0, 0), lens_fn=lens, focus_fn=focus)
sb.frames(F0, F1)
sb.render_shot(sid)
