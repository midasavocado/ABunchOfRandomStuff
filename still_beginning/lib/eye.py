"""The child's eye: shared by s01 (opening) and s27 (return). Units: meters.
Eye centre at origin, looking toward -Y. +X = lateral (temple), -X = medial (nose), +Z = up."""
import bpy, math, random
from mathutils import Vector as V, noise
import sb

R = 0.0115           # eyeball radius (child)
IRIS_R = 0.0058
PUPIL_R = 0.0021     # base pupil radius (dilated state set via shape key)
CORNEA_RC = 0.0076   # corneal radius of curvature

SKIN_TONE = (0.52, 0.34, 0.24)
HAIR_COL = (0.035, 0.02, 0.012)


def _limbus_y():
    return -math.sqrt(R * R - IRIS_R * IRIS_R)


def build_eyeball(root):
    yl = _limbus_y()
    # --- sclera
    scl = sb.mat("Sclera", (0.86, 0.83, 0.80), rough=0.25, sss=0.6, sss_radius=(1, 0.5, 0.35), sss_scale=0.002, coat=1.0, coat_rough=0.02)
    nb = sb.NB(scl)
    co = nb.coord('Object')
    veins = nb.noise(co, scale=180, detail=5, rough=0.55, dist=1.5)
    vline = nb.math('ABSOLUTE', nb.math('SUBTRACT', veins.outputs['Fac'], 0.5))
    vmask = nb.maprange(vline, 0.0, 0.012, 1.0, 0.0)
    # veins only toward periphery (away from limbus): use |x| from object coords
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    per = nb.maprange(nb.math('ABSOLUTE', sep.outputs[0]), 0.0070, 0.0110, 0.0, 0.5)
    vm = nb.math('MULTIPLY', vmask, per)
    pink = nb.maprange(nb.math('ABSOLUTE', sep.outputs[0]), 0.006, 0.012, 0.0, 0.35)
    base = nb.mix(pink, (0.90, 0.88, 0.86, 1), (0.86, 0.70, 0.64, 1))
    col = nb.mix(vm, base, (0.62, 0.18, 0.14, 1))
    ao = nb.new('ShaderNodeAmbientOcclusion', Distance=0.004)
    ao.samples = 8
    col = nb.mix(nb.maprange(ao.outputs['AO'], 0.2, 1.0, 0.75, 0.0), col, (0.30, 0.20, 0.18, 1), blend='MULTIPLY')
    nb.set('Base Color', col)
    wet = nb.noise(co, scale=900, detail=3)
    nb.set('Normal', nb.bump(wet.outputs['Fac'], strength=0.04, distance=0.00005))
    prof = []
    th_l = math.asin(IRIS_R / R)
    for k in range(64):
        th = th_l + (math.pi - th_l) * k / 63
        prof.append((R * math.sin(th), -R * math.cos(th)))
    ball = sb.lathe("Sclera", [(r, z) for r, z in prof], segs=128, mat=scl)
    # lathe revolves around Z; our axis is Y -> rotate
    ball.rotation_euler = (-math.pi / 2, 0, 0)
    sb.apply_mods(ball)
    ball.parent = root

    # --- iris: annulus with fibre relief, UV = (angle, radius)
    NA, NR = 256, 40
    verts, faces, uvs = [], [], []
    ri, ro = PUPIL_R, IRIS_R + 0.0007
    rnd = random.Random(7)
    for i in range(NA):
        a = 2 * math.pi * i / NA
        for j in range(NR):
            t = j / (NR - 1)
            r = ri + (ro - ri) * t
            # iris is gently conical: pupil margin sits ~0.4mm forward of root
            y = yl + 0.0009 - 0.00045 * (1 - t) ** 1.5
            # collarette ridge + radial trabeculae relief
            rel = 0.00012 * math.exp(-((t - 0.34) / 0.05) ** 2)
            rel += 0.00005 * noise.noise(V((math.cos(a) * 30, math.sin(a) * 30, t * 3)))
            verts.append((r * math.cos(a), y - rel, r * math.sin(a)))
    for i in range(NA):
        k = (i + 1) % NA
        for j in range(NR - 1):
            faces.append((i * NR + j, k * NR + j, k * NR + j + 1, i * NR + j + 1))
    iris_m = _iris_material()
    iris = sb.mesh_obj("Iris", verts, faces, iris_m)
    me = iris.data
    uvl = me.uv_layers.new(name="UV")
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            i, j = divmod(vi, NR)
            uvl.data[li].uv = (i / NA, j / (NR - 1))
    # pupil shape key (value 1 = constricted to 0.75x)
    iris.shape_key_add(name="Basis")
    sk = iris.shape_key_add(name="Constrict")
    for vi, v in enumerate(me.vertices):
        i, j = divmod(vi, NR)
        t = j / (NR - 1)
        a = 2 * math.pi * i / NA
        r0 = ri + (ro - ri) * t
        r1 = ri * 0.72 + (ro - ri * 0.72) * t
        sk.data[vi].co = (r1 * math.cos(a), v.co.y, r1 * math.sin(a))
    iris.parent = root

    # --- pupil depth (dark lens/retina) and faint lens reflection
    dark = sb.mat("PupilDark", (0.003, 0.003, 0.004), rough=0.35, spec=0.2)
    back = sb.prim("sphere", "EyeInterior", segments=48, ring_count=24, radius=R * 0.9, mat=dark)
    back.location = (0, 0.0024, 0)
    back.parent = root
    # --- cornea: spherical cap meeting the limbus + tear film
    yc = yl + math.sqrt(CORNEA_RC ** 2 - IRIS_R ** 2)
    prof = []
    rmax = IRIS_R + 0.0011
    NP = 70
    for k in range(NP):
        r = rmax * (1 - k / (NP - 1))
        ys = -math.sqrt(max(R * R - r * r, 0)) - 0.00003
        yc_r = yc - math.sqrt(max(CORNEA_RC ** 2 - r * r, 0)) if r < CORNEA_RC else ys
        w = sb.smoother((r - (IRIS_R - 0.0006)) / 0.0011)
        y = min(sb.lerp(yc_r, ys, w), ys)
        y += 0.00008 * sb.smooth((r - (IRIS_R + 0.0004)) / 0.0006)   # dive under the sclera: no visible edge
        prof.append((r, y))
    verts, faces = [], []
    NS = 128
    for i in range(NS):
        a = 2 * math.pi * i / NS
        for r, y in prof:
            verts.append((r * math.cos(a), y, r * math.sin(a)))
    n = len(prof)
    for i in range(NS):
        k = (i + 1) % NS
        for j in range(n - 1):
            faces.append((i * n + j, k * n + j, k * n + j + 1, i * n + j + 1))
    cm = sb.mat("Cornea", (1, 1, 1), rough=0.0, transmission=1.0, ior=1.376, spec=0.6)
    cor = sb.mesh_obj("Cornea", verts, faces, cm)
    cor.visible_shadow = False
    cor.parent = root
    return dict(ball=ball, iris=iris, cornea=cor)


def _sphere_y(r):
    return -math.sqrt(max(R * R - r * r, 0))


def _iris_material():
    m = sb.mat("Iris", (0.3, 0.18, 0.08), rough=0.5, spec=0.35)
    nb = sb.NB(m)
    uv = nb.new('ShaderNodeUVMap').outputs[0]
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    ang = nb.math('MULTIPLY', u, 2 * math.pi)
    cx = nb.math('COSINE', ang); sx = nb.math('SINE', ang)

    def polar(A, B, off=0.0, vv=None):
        vv = vv if vv is not None else v
        cmb = nb.new('ShaderNodeCombineXYZ')
        nb.link(nb.math('MULTIPLY', cx, A), cmb.inputs[0])
        nb.link(nb.math('MULTIPLY', sx, A), cmb.inputs[1])
        nb.link(nb.math('ADD', nb.math('MULTIPLY', vv, B), off), cmb.inputs[2])
        return cmb.outputs[0]
    # wobble the radius coordinate so strands wander instead of being ruler-straight
    wob = nb.noise(polar(6, 3, 11.0), scale=2.0, detail=3)
    vw = nb.math('ADD', v, nb.math('MULTIPLY', nb.math('SUBTRACT', wob.outputs['Fac'], 0.5), 0.05))
    # angular warp -> strands curve and braid
    warp = nb.noise(polar(4, 4, 5.0), scale=1.5, detail=2)
    angw = nb.math('ADD', ang, nb.math('MULTIPLY', nb.math('SUBTRACT', warp.outputs['Fac'], 0.5), 0.03))
    cxw = nb.math('COSINE', angw); sxw = nb.math('SINE', angw)

    def polarw(A, B, off=0.0):
        cmb = nb.new('ShaderNodeCombineXYZ')
        nb.link(nb.math('MULTIPLY', cxw, A), cmb.inputs[0])
        nb.link(nb.math('MULTIPLY', sxw, A), cmb.inputs[1])
        nb.link(nb.math('ADD', nb.math('MULTIPLY', vw, B), off), cmb.inputs[2])
        return cmb.outputs[0]

    def ridge(n, sharp=3.0):
        r = nb.math('SUBTRACT', 1.0, nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('MULTIPLY', n, 2.0), 1.0)))
        return nb.math('POWER', r, sharp)
    # three scales of radial structure: thick bundles, medium strands, fine fibres
    bund = ridge(nb.noise(polarw(16, 1.3), scale=3.0, detail=3, rough=0.5, dist=0.5).outputs['Fac'], 3.0)
    med = ridge(nb.noise(polarw(42, 1.6, 7.0), scale=3.0, detail=4, rough=0.55, dist=0.4).outputs['Fac'], 3.0)
    fine = ridge(nb.noise(polarw(110, 0.9, 3.0), scale=3.0, detail=5, rough=0.6).outputs['Fac'], 2.0)
    tone = nb.noise(polar(8, 5, 9.0), scale=2.0, detail=5, rough=0.6)
    tone2 = nb.noise(polar(30, 3, 13.0), scale=2.0, detail=4, rough=0.6)
    # crypts: radially elongated dark lacunae around/outside the collarette
    cry = nb.voronoi(polarw(12, 3.0, 21.0), scale=2.2, feature='F1')
    cr = nb.maprange(cry.outputs['Distance'], 0.04, 0.26, 1.0, 0.0)
    band = nb.math('MULTIPLY', nb.maprange(v, 0.30, 0.40), nb.maprange(v, 0.80, 0.55))
    cr = nb.math('MULTIPLY', nb.math('POWER', cr, 1.6), band)
    # collarette zig-zag
    zz = nb.noise(polar(3.5, 0.5, 2.0), scale=2.0, detail=4)
    cpos = nb.math('ADD', 0.35, nb.math('MULTIPLY', nb.math('SUBTRACT', zz.outputs['Fac'], 0.5), 0.22))
    dcol = nb.math('ABSOLUTE', nb.math('SUBTRACT', v, cpos))
    collar = nb.math('POWER', nb.maprange(dcol, 0.0, 0.05, 1.0, 0.0), 1.5)
    inner = nb.maprange(nb.math('SUBTRACT', v, cpos), 0.04, -0.08)
    # albedo-plausible amber iris
    t1 = nb.maprange(tone.outputs['Fac'], 0.3, 0.7)
    ciliary = nb.mix(t1, (0.14, 0.055, 0.012, 1), (0.28, 0.12, 0.025, 1))
    ciliary = nb.mix(nb.maprange(tone2.outputs['Fac'], 0.3, 0.7, 0.0, 0.5), ciliary, (0.24, 0.12, 0.03, 1))
    pup = nb.mix(t1, (0.26, 0.09, 0.02, 1), (0.46, 0.19, 0.035, 1))
    col = nb.mix(inner, ciliary, pup)
    strands = nb.math('ADD', nb.math('ADD', nb.math('MULTIPLY', bund, 0.75), nb.math('MULTIPLY', med, 0.35)),
                      nb.math('MULTIPLY', fine, 0.12))
    gold = nb.mix(t1, (0.58, 0.28, 0.06, 1), (0.74, 0.42, 0.11, 1))
    col = nb.mix(nb.maprange(strands, 0.25, 0.95, 0.0, 0.85), col, gold)
    col = nb.mix(nb.maprange(strands, 0.18, 0.0, 0.0, 0.7), col, (0.025, 0.012, 0.004, 1))
    col = nb.mix(nb.math('MULTIPLY', collar, 0.32), col, (0.66, 0.38, 0.09, 1))
    col = nb.mix(cr, col, (0.02, 0.01, 0.004, 1))
    # darker periphery and limbal ring, pupil ruff
    per = nb.maprange(v, 0.62, 0.86, 0.0, 0.55)
    col = nb.mix(per, col, (0.06, 0.035, 0.015, 1))
    limb = nb.maprange(v, 0.84, 0.95)
    col = nb.mix(nb.math('MULTIPLY', limb, 0.95), col, (0.025, 0.02, 0.016, 1))
    ruff = nb.maprange(v, 0.045, 0.0)
    col = nb.mix(ruff, col, (0.03, 0.012, 0.005, 1))
    nb.set('Base Color', col)
    h = nb.math('SUBTRACT', nb.math('ADD', strands, nb.math('MULTIPLY', collar, 0.6)), nb.math('MULTIPLY', cr, 1.5))
    nb.set('Normal', nb.bump(h, strength=0.7, distance=0.00015))
    nb.set('Roughness', nb.maprange(strands, 0.0, 1.0, 0.55, 0.35))
    return m


# --------------------------------------------------------------------------- lids + face patch

def _boundary(u, open_up=1.0, open_lo=1.0):
    """Palpebral fissure boundary, u in [0,1): upper arc medial->lateral (0..0.5), lower back."""
    xm, zm = -0.0112, -0.0010
    xl, zl = 0.0106, 0.0014
    if u < 0.5:
        s = u / 0.5
        x = sb.lerp(xm, xl, s)
        # upper lid peak sits medial of centre and covers the top of the iris
        z = sb.lerp(zm, zl, s) + 0.0046 * open_up * math.sin(math.pi * s) ** 0.7 * (1.0 + 0.16 * (0.5 - s))
    else:
        s = (u - 0.5) / 0.5
        x = sb.lerp(xl, xm, s)
        z = sb.lerp(zl, zm, s) - 0.0056 * open_lo * math.sin(math.pi * s) ** 1.05 * (1.0 + 0.10 * (0.5 - s))
    return x, z


def _radial(u, open_up=1.0, open_lo=1.0):
    bx, bz = _boundary(u, open_up, open_lo)
    cx, cz = 0.0, 0.0003
    dx, dz = bx - cx, bz - cz
    L = math.hypot(dx, dz)
    return bx, bz, dx / L, dz / L


def _face_y(x, z, wonder=0.0):
    """Child face surface in front of the eye (flat orbit; brow ~level with the cornea apex)."""
    y = -0.0128
    # soft brow ridge above
    y -= 0.0016 * math.exp(-((z - 0.0150) / 0.0085) ** 2) * (1 - 0.3 * sb.smooth((x - 0.01) / 0.03))
    # nose side (medial) comes forward
    y -= 0.0120 * sb.smooth((-x - 0.016) / 0.022) * math.exp(-((z + 0.002) / 0.03) ** 2)
    # temple recedes gently
    y += 0.0035 * sb.smooth((x - 0.020) / 0.03)
    # full child cheek below (lifts slightly with a smile)
    cheek = 0.0022 * (1 + 0.5 * wonder)
    y -= cheek * math.exp(-(((z + 0.020 - 0.002 * wonder) / 0.012) ** 2 + ((x - 0.004) / 0.022) ** 2))
    # forehead recedes above brow
    y += 0.004 * sb.smooth((z - 0.026) / 0.03)
    return y


# (d offset in projection, shell radius above eyeball) for the lid margin profile
MARGIN = [(0.0, 0.00010), (0.00012, 0.00075), (0.00030, 0.00125), (0.00060, 0.00150), (0.0011, 0.00162)]


def build_lids(root, open_up=1.0, open_lo=1.0, wonder=0.0, NU=260, extent=0.045):
    rings = [m[0] for m in MARGIN] + [0.0018, 0.0028, 0.0040, 0.0052, 0.0062, 0.0074, 0.0090, 0.0110,
                                      0.0135, 0.0165, 0.020, 0.024, 0.029, 0.035, extent]
    NR = len(rings)
    verts = []
    for i in range(NU):
        u = i / NU
        bx, bz, dx, dz = _radial(u, open_up, open_lo)
        upper = u < 0.5
        su = math.sin(math.pi * (u if upper else u - 0.5) / 0.5)
        for k, d in enumerate(rings):
            x, z = bx + dx * d, bz + dz * d
            if not upper:
                z += 0.0006 * wonder * su * math.exp(-d / 0.006)
            rr = x * x + z * z
            if k < len(MARGIN):
                rs = R + MARGIN[k][1]
            else:
                rs = R + 0.00165 + 0.30 * max(0, d - 0.0011) ** 1.08
            q = rs * rs - rr
            c2 = 0.0030 ** 2
            q = 0.5 * (q + c2 + math.sqrt((q - c2) ** 2 + (2.5e-5) ** 2))
            yl = -math.sqrt(q)
            yf = _face_y(x, z, wonder)
            w = sb.smoother((d - 0.0025) / 0.016)
            y = sb.lerp(yl, yf, w)
            if upper:
                # pretarsal bulge, then a real crease under the orbital fold
                y -= 0.00035 * math.exp(-((d - 0.0025) / 0.0014) ** 2) * su
                y += 0.0008 * math.exp(-((d - 0.0058) / 0.0010) ** 2) * su ** 0.6
                y -= 0.0003 * math.exp(-((d - 0.0080) / 0.002) ** 2) * su ** 0.6
            else:
                # lower pretarsal roll (grows with a smile) and soft tear-trough
                y -= (0.00015 + 0.0005 * wonder) * math.exp(-((d - 0.0022) / 0.0015) ** 2) * su
                y += 0.00030 * math.exp(-((d - 0.0080) / 0.0030) ** 2) * su ** 0.8
            verts.append((x, y, z))
    faces, mats = [], []
    for i in range(NU):
        j = (i + 1) % NU
        for k in range(NR - 1):
            faces.append((i * NR + k, i * NR + k + 1, j * NR + k + 1, j * NR + k))
            mats.append(1 if k < 1 else 0)
    sk = eye_skin()
    wl = sb.mat("Waterline", (0.50, 0.30, 0.27), rough=0.08, sss=1.0, sss_radius=(1, 0.3, 0.2), sss_scale=0.001, coat=1.0, coat_rough=0.02, spec=0.6)
    o = sb.mesh_obj("Lids", verts, faces, sk)
    o.data.materials.append(wl)
    for p, mi in zip(o.data.polygons, mats):
        p.material_index = mi
    o.parent = root
    # relax the skewed quads around the canthi (keep the lid margin rings exact)
    vg = o.vertex_groups.new(name="relax")
    for i in range(NU):
        for k in range(NR):
            wgt = sb.smooth((k - 3) / 4.0)
            if wgt > 0:
                vg.add([i * NR + k], wgt, 'REPLACE')
    sm = o.modifiers.new("Relax", 'LAPLACIANSMOOTH')
    sm.iterations = 12
    sm.lambda_factor = 0.6
    sm.lambda_border = 0.0
    sm.vertex_group = "relax"
    sm.use_volume_preserve = True
    sb.subsurf(o, 1, 2)
    return o


def eye_skin():
    """Periocular skin: pores, anisotropic fine lines, anatomical colour zones. Object space = eye space (m)."""
    tone = SKIN_TONE
    m = sb.mat("ChildSkin", tone, rough=0.42, sss=1.0, sss_radius=(1.0, 0.38, 0.2), sss_scale=0.0018, spec=0.5, sheen=0.25)
    b = m.node_tree.nodes["Principled BSDF"]
    b.subsurface_method = 'BURLEY'
    b.inputs['Sheen Roughness'].default_value = 0.4
    b.inputs['Sheen Tint'].default_value = (1.0, 0.85, 0.75, 1)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    x, z = sep.outputs[0], sep.outputs[2]
    rad = nb.vmath('LENGTH', nb.vmath('MULTIPLY', co, (1.0, 0.0, 1.35)))
    # --- colour
    mott = nb.noise(co, scale=260, detail=6, rough=0.6)
    blot = nb.noise(co, scale=70, detail=4, rough=0.5)
    base = nb.mix(nb.maprange(mott.outputs['Fac'], 0.35, 0.65), (tone[0] * 0.93, tone[1] * 0.9, tone[2] * 0.88, 1), (tone[0] * 1.04, tone[1] * 1.03, tone[2] * 1.02, 1))
    # rosier thin lid skin close to the fissure, strongest at the inner corner
    near = nb.maprange(rad, 0.0075, 0.0035)
    inner = nb.maprange(x, -0.006, -0.013)
    rose = nb.math('MULTIPLY', nb.math('MAXIMUM', near, nb.math('MULTIPLY', inner, nb.maprange(rad, 0.012, 0.006))), 0.35)
    base = nb.mix(rose, base, (tone[0] * 1.05, tone[1] * 0.70, tone[2] * 0.68, 1))
    # faint under-eye pigment and darker upper-lid crease
    under = nb.math('MULTIPLY', nb.maprange(z, -0.006, -0.011), nb.maprange(rad, 0.018, 0.010))
    base = nb.mix(nb.math('MULTIPLY', under, 0.35), base, (tone[0] * 0.78, tone[1] * 0.68, tone[2] * 0.72, 1))
    base = nb.mix(nb.maprange(blot.outputs['Fac'], 0.6, 0.8, 0.0, 0.06), base, (tone[0] * 1.06, tone[1] * 0.82, tone[2] * 0.78, 1))
    nb.set('Base Color', base)
    # --- micro relief
    pore = nb.voronoi(co, scale=1500, feature='F1')
    pd = nb.maprange(pore.outputs['Distance'], 0.0, 0.45, -1.0, 0.0)
    pd = nb.math('POWER', nb.math('MULTIPLY', pd, -1.0), 3.0)            # small sharp pits
    fine = nb.noise(co, scale=2600, detail=3, rough=0.5)

    def lines(rot, sx, thr):
        n = nb.noise(nb.mapping(co, rot=rot, scale=(sx, 1.0, 1.0)), scale=2200, detail=2, rough=0.5)
        r = nb.math('ABSOLUTE', nb.math('SUBTRACT', n.outputs['Fac'], 0.5))
        return nb.maprange(r, 0.0, thr, 1.0, 0.0)
    l1 = lines((0.0, 0.0, 0.0), 0.12, 0.018)
    l2 = lines((0.0, 1.1, 0.0), 0.15, 0.015)
    ln = nb.math('MAXIMUM', l1, nb.math('MULTIPLY', l2, 0.7))
    h = nb.math('SUBTRACT', nb.math('MULTIPLY', fine.outputs['Fac'], 0.35), nb.math('ADD', nb.math('MULTIPLY', pd, 0.6), nb.math('MULTIPLY', ln, 0.2)))
    nb.set('Normal', nb.bump(h, strength=0.65, distance=0.00015))
    nb.set('Roughness', nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.34, 0.5))
    return m


def lid_margin_points(side="upper", n=90, open_up=1.0, open_lo=1.0, k=2):
    """Points on the anterior lid margin (lash line)."""
    pts = []
    for i in range(n):
        if side == "upper":
            u = 0.035 + 0.43 * (i + 0.5) / n
        else:
            u = 0.54 + 0.40 * (i + 0.5) / n
        bx, bz, dx, dz = _radial(u, open_up, open_lo)
        d, rsh = MARGIN[k]
        x, z = bx + dx * d, bz + dz * d
        rs = R + rsh
        y = -math.sqrt(max(rs * rs - x * x - z * z, 0.0026 ** 2))
        pts.append((V((x, y, z)), V((dx, 0, dz)), u))
    return pts


def _arc_lash(cu, root, d0, d1, L, rad, segs=7):
    sp = cu.splines.new('POLY')
    sp.points.add(segs - 1)
    pos = root.copy()
    q = d0.rotation_difference(d1)
    from mathutils import Quaternion
    for k in range(segs):
        t = k / (segs - 1)
        sp.points[k].co = (pos.x, pos.y, pos.z, 1)
        sp.points[k].radius = rad * ((1 - t) ** 0.7 * 0.95 + 0.05)
        dirv = Quaternion().slerp(q, min(1.0, (k + 0.5) / (segs - 1))) @ d0
        pos = pos + dirv * (L / (segs - 1))
    return sp


def build_lashes(root, open_up=1.0, open_lo=1.0, seed=3):
    rnd = random.Random(seed)
    cu = bpy.data.curves.new("Lashes", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.000045
    cu.bevel_resolution = 2
    cu.use_fill_caps = False
    fwd = V((0, -1, 0))
    for side, n, length in (("upper", 170, 0.0090), ("lower", 70, 0.0042)):
        pts = lid_margin_points(side, n, open_up, open_lo, k=2)
        clump = None
        for idx, (p, outward, u) in enumerate(pts):
            if side == "lower" and rnd.random() < 0.35:
                continue
            s = (u - 0.035) / 0.43 if side == "upper" else (u - 0.54) / 0.40
            s = min(max(s, 0), 1)
            if side == "upper":
                prof = (math.sin(math.pi * (0.08 + 0.87 * s)) ** 0.6) * (0.45 + 0.55 * s)
            else:
                prof = math.sin(math.pi * (0.1 + 0.8 * s)) ** 0.8 * (0.6 + 0.4 * s)
            L = length * (0.3 + 0.7 * prof) * rnd.uniform(0.72, 1.08)
            up = outward.normalized()
            lat = V((1, 0, 0)) * ((0.5 if side == "upper" else 0.3) * (s - 0.45))
            jit = V((rnd.gauss(0, 0.08), rnd.gauss(0, 0.05), rnd.gauss(0, 0.08)))
            if side == "upper":
                d0 = (fwd - up * 0.15 + lat * 0.4 + jit).normalized()
                d1 = (fwd * 0.35 + up * 1.0 + lat * 0.8 + jit).normalized()
            else:
                d0 = (fwd + up * 0.25 + lat * 0.4 + jit).normalized()
                d1 = (fwd * 0.55 + up * 0.85 + lat * 0.7 + jit).normalized()
            # natural clumping: neighbours lean toward a shared direction
            if idx % 5 == 0 or clump is None:
                clump = (d1 + V((rnd.gauss(0, 0.12), rnd.gauss(0, 0.08), rnd.gauss(0, 0.12)))).normalized()
            d1 = d1.lerp(clump, 0.5).normalized()
            base = p + V((rnd.gauss(0, 0.00005), rnd.gauss(0, 0.00003), rnd.gauss(0, 0.00003)))
            rad = (1.0 if side == "upper" else 0.7) * rnd.uniform(0.75, 1.2)
            _arc_lash(cu, base, d0, d1, L, rad)
    m = sb.mat("Lash", (0.008, 0.005, 0.004), rough=0.45, spec=0.3)
    cu.materials.append(m)
    o = bpy.data.objects.new("Lashes", cu)
    sb.link_obj(o)
    o.parent = root
    return o


def build_caruncle(root):
    m = sb.mat("Caruncle", (0.66, 0.30, 0.28), rough=0.35, sss=1.0, sss_radius=(1, 0.3, 0.2), sss_scale=0.0015, coat=0.6, coat_rough=0.1, spec=0.4)
    o = sb.prim("sphere", "Caruncle", segments=32, ring_count=16, radius=0.0011, mat=m)
    o.scale = (1.0, 0.7, 0.8)
    o.location = (-0.0103, -0.0046, -0.0006)
    o.parent = root
    # tear meniscus along the lower lid: thin wet tube
    pts = [p for p, _, _ in lid_margin_points("lower", 80, k=0)]
    wet = sb.mat("Tear", (1, 1, 1), rough=0.0, transmission=1.0, ior=1.34, spec=0.8)
    t = sb.tube_along("TearMeniscus", [tuple(p + V((0, 0.00005, 0))) for p in pts], radius=0.00007, mat=wet, bevel_res=3)
    t.parent = root
    return o


def build_brow(root, seed=11, n=600):
    """Eyebrow hairs for the wider return shot."""
    rnd = random.Random(seed)
    cu = bpy.data.curves.new("Brow", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.00005
    cu.bevel_resolution = 1
    for i in range(n):
        s = rnd.random()
        x = sb.lerp(-0.017, 0.022, s)
        zc = 0.0145 + 0.004 * math.sin(math.pi * (s * 0.9 + 0.05)) - 0.002 * s
        z = zc + rnd.gauss(0, 0.0017 * (1.2 - 0.6 * s))
        y = _face_y(x, z) - 0.0001
        sp = cu.splines.new('POLY')
        sp.points.add(4)
        d = V((0.55 + 0.4 * s, -0.25, 0.35 - 0.5 * s + rnd.gauss(0, 0.15))).normalized()
        pos = V((x, y, z))
        L = rnd.uniform(0.004, 0.007)
        for k in range(5):
            sp.points[k].co = (*pos, 1)
            sp.points[k].radius = 1 - k / 5
            d = (d + V((0, 0.12, -0.06))).normalized()
            pos = pos + d * (L / 4)
    m = sb.mat("BrowHair", HAIR_COL, rough=0.4, coat=0.2)
    cu.materials.append(m)
    o = bpy.data.objects.new("Brow", cu)
    sb.link_obj(o)
    o.parent = root
    return o


def build(open_up=1.0, open_lo=1.0, wonder=0.0, brow=False):
    root = sb.empty("EyeRoot")
    gaze = sb.empty("Gaze", parent=root)       # rotate this for gaze changes
    parts = build_eyeball(gaze)
    lids = build_lids(root, open_up=open_up, open_lo=open_lo, wonder=wonder)
    lashes = build_lashes(root, open_up, open_lo)
    build_caruncle(root)
    if brow:
        build_brow(root)
    # corneal caustic: light focused by the cornea lands as a bright crescent on the far side of the iris
    coll = bpy.data.collections.new("IrisOnly")
    bpy.context.scene.collection.children.link(coll)
    coll.objects.link(parts["iris"])
    ld = bpy.data.lights.new("Caustic", 'SPOT')
    ld.energy = 0.05
    ld.color = (1.0, 0.78, 0.5)
    ld.spot_size = math.radians(40)
    ld.spot_blend = 1.0
    ld.shadow_soft_size = 0.0005
    cl = bpy.data.objects.new("Caustic", ld)
    sb.link_obj(cl)
    cl.parent = gaze
    cl.location = (-0.0035, -0.0135, 0.0045)
    cl.rotation_mode = 'QUATERNION'
    cl.rotation_quaternion = sb.look_quat(cl.location, (0.0028, -0.0095, -0.0032))
    cl.visible_glossy = False
    try:
        cl.light_linking.receiver_collection = coll
    except Exception as ex:
        print("light linking unavailable", ex)
    parts.update(root=root, gaze=gaze, lids=lids, lashes=lashes, caustic=cl)
    return parts


def build_rooftop_env(seed=5, telescope=True, window_glow=3.0):
    """Environment seen in the cornea: varied skyline silhouettes, lit windows, the child's telescope."""
    roof = sb.mat("RoofSil", (0.004, 0.0045, 0.006), rough=0.9)
    rnd = sb.rng(seed)
    wmat = sb.emit_mat("WinLit", (1.0, 0.62, 0.3), window_glow)
    for i in range(110):
        a = math.radians(-175 + i * 3.2 + rnd.uniform(-1, 1))
        d = rnd.uniform(5, 14)
        kind = rnd.random()
        h = rnd.uniform(0.1, 0.6) if kind < 0.6 else rnd.uniform(0.6, 2.4)
        base_z = -1.4
        c = V((math.sin(a) * d, -math.cos(a) * d, base_z + h / 2))
        wdt = rnd.uniform(0.4, 1.6)
        b = sb.prim("cube", f"roof{i}", loc=c, scale=(wdt, 0.5, h / 2), mat=roof)
        b.rotation_euler = (0, 0, -a)
        top = base_z + h
        r2 = rnd.random()
        if r2 < 0.18:      # water tank
            sb.prim("cyl", f"tank{i}", loc=c + V((0, 0, h / 2 + 0.35)), scale=(0.25, 0.25, 0.35), mat=roof)
        elif r2 < 0.32:    # antenna mast
            sb.prim("cyl", f"ant{i}", loc=c + V((rnd.uniform(-0.3, 0.3), 0, h / 2 + 0.6)), scale=(0.015, 0.015, 0.6), mat=roof)
        elif r2 < 0.45:    # tree crown
            for t in range(3):
                sb.prim("ico", f"tree{i}_{t}", loc=c + V((rnd.uniform(-0.4, 0.4), rnd.uniform(-0.2, 0.2), h / 2 + 0.2 + rnd.uniform(0, 0.4))),
                        scale=(rnd.uniform(0.3, 0.55),) * 3, mat=roof, subdivisions=2)
        nwin = int(h * 3 * rnd.random())
        for k in range(nwin):
            wl = sb.prim("plane", f"win{i}_{k}", loc=c + V((0, 0, rnd.uniform(-h / 2.5, h / 2.5))) - V((math.sin(a), -math.cos(a), 0)) * 0.52
                         + V((math.cos(a), math.sin(a), 0)) * rnd.uniform(-wdt * 0.8, wdt * 0.8),
                         scale=(0.08, 0.1, 1), mat=wmat)
            wl.rotation_euler = (math.pi / 2, 0, -a)
    if telescope:
        # the child's refractor, just beside them, aimed at the sky (a small dark silhouette in the reflection)
        tm = sb.mat("TubeSil", (0.02, 0.02, 0.022), rough=0.5)
        t = sb.prim("cyl", "ScopeSil", loc=(-0.28, -0.32, 0.12), scale=(0.045, 0.045, 0.42), mat=tm)
        t.rotation_euler = (math.radians(-55), 0, math.radians(-30))
