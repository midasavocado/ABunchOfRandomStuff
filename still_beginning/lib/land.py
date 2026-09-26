"""Landscape assets for s13-s16: rolling terrain, crop/field materials, procedural trees with real canopies
(trunk + branch hierarchy + thousands of alpha-cut leaf cards, backlit translucency), hedgerows.
Units: metres. Owner: energy/city artist."""
import bpy, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion
import sb
import energy as E


# =========================================================================== terrain
def hills(x, y, amp=1.0):
    """Rolling farmland height (numpy-friendly)."""
    h = (9.0 * np.sin(x / 210.0 + 0.7) * np.cos(y / 260.0 - 0.3)
         + 6.0 * np.sin((x * 0.6 + y) / 150.0 + 1.9)
         + 3.0 * np.sin((x - 0.8 * y) / 95.0 + 0.4)
         + 18.0 * np.sin(x / 900.0 - 0.5) * np.sin(y / 1100.0 + 0.8))
    return h * amp


def terrain(name, fn, x0, x1, y0, y1, nx, ny, mat=None):
    xs = np.linspace(x0, x1, nx); ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    i = np.arange(nx - 1); j = np.arange(ny - 1)
    I, J = np.meshgrid(i, j)
    a = (J * nx + I).ravel()
    faces = np.stack([a, a + 1, a + nx + 1, a + nx], 1)
    uv = None
    o = E.mesh_np(name, verts, faces, mat)
    return o


# =========================================================================== materials
def field_mat(name, rows_dir=(1.0, 0.0), pitch=0.75, green=(0.05, 0.13, 0.03), soil=(0.10, 0.07, 0.045),
              patch=True, haze=None):
    """Farmland: crop rows (stripes of plants over soil) with growth variation, field patchwork of different crops
    at large scale (green crops, ripening wheat, meadow), tractor wheel lines."""
    m = sb.mat(name, green, rough=0.8)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    # row coordinate (perpendicular to the rows)
    across = nb.math('ADD', nb.math('MULTIPLY', sep.outputs[0], -rows_dir[1]), nb.math('MULTIPLY', sep.outputs[1], rows_dir[0]))
    rowf = nb.math('FRACT', nb.math('DIVIDE', across, pitch))
    plant = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', rowf, 0.5)), 0.18, 0.40, 1.0, 0.0)
    clumps = nb.voronoi(co, scale=4.5, feature='F1')
    lushn = nb.noise(co, scale=0.08, detail=6, rough=0.6)
    fine = nb.noise(co, scale=9.0, detail=6, rough=0.65)
    cov = nb.math('MULTIPLY', plant, nb.maprange(clumps.outputs['Distance'], 0.1, 0.55, 1.0, 0.55))
    cov = nb.math('ADD', cov, nb.maprange(lushn.outputs['Fac'], 0.35, 0.65, -0.1, 0.35))
    cov = nb.math('MINIMUM', nb.math('MAXIMUM', cov, 0.0), 1.0)
    g1 = nb.mix(nb.maprange(fine.outputs['Fac'], 0.3, 0.7), (green[0] * 0.7, green[1] * 0.75, green[2] * 0.6, 1), (green[0] * 1.35, green[1] * 1.25, green[2] * 1.1, 1))
    col = nb.mix(cov, (*soil, 1), g1)
    if patch:
        # field patchwork: cells of ~150-300 m with different crops
        fv = nb.voronoi(nb.mapping(co, scale=(1.0, 1.35, 1.0)), scale=0.0045, feature='F1', rand=0.8)
        fc = nb.new('ShaderNodeSeparateColor'); nb.link(fv.outputs['Color'], fc.inputs[0])
        crop = nb.ramp(fc.outputs[0], [(0.0, (0.055, 0.12, 0.03)), (0.16, (0.09, 0.16, 0.04)), (0.30, (0.30, 0.22, 0.07)),
                                        (0.42, (0.45, 0.35, 0.14)), (0.54, (0.50, 0.42, 0.22)), (0.64, (0.15, 0.10, 0.065)),
                                        (0.76, (0.07, 0.10, 0.03)), (0.88, (0.20, 0.24, 0.09)), (1.0, (0.12, 0.16, 0.05))])
        crop.node.color_ramp.interpolation = 'CONSTANT'
        # tramlines (sprayer wheel tracks every 24 m along the rows) + field margins / hedge lines at cell edges
        tram = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', across, 24.0)), 0.5)), 0.488, 0.496)
        crop = nb.mix(nb.math('MULTIPLY', tram, 0.55), crop, (0.11, 0.08, 0.05, 1))
        edge = nb.new('ShaderNodeTexVoronoi'); edge.feature = 'DISTANCE_TO_EDGE'
        edge.inputs['Scale'].default_value = 0.0045
        edge.inputs['Randomness'].default_value = 0.8
        nb.link(nb.mapping(co, scale=(1.0, 1.35, 1.0)), edge.inputs['Vector'])
        margin = nb.maprange(edge.outputs['Distance'], 0.006, 0.014, 1.0, 0.0)
        hedge = nb.maprange(edge.outputs['Distance'], 0.0, 0.004, 1.0, 0.0)
        crop = nb.mix(margin, crop, (0.10, 0.15, 0.05, 1))
        crop = nb.mix(hedge, crop, (0.03, 0.06, 0.018, 1))
        far = nb.new('ShaderNodeCameraData')
        dfar = nb.maprange(far.outputs['View Distance'], 60.0, 260.0, 0.0, 1.0)
        # the solar site (|x|<380, |y|<300) keeps its green understorey crop
        site = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', sep.outputs[0]), 380.0, 400.0, 1.0, 0.0),
                       nb.maprange(nb.math('ABSOLUTE', sep.outputs[1]), 290.0, 310.0, 1.0, 0.0))
        blend = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, site), 0.85)
        texd = nb.mix(nb.maprange(fine.outputs['Fac'], 0.3, 0.7), (0.8, 0.8, 0.8, 1), (1.15, 1.15, 1.1, 1))
        crop2 = nb.mix(1.0, crop, texd, blend='MULTIPLY')
        col = nb.mix(blend, col, crop2)
        col = nb.mix(nb.math('MULTIPLY', dfar, nb.math('SUBTRACT', 1.0, site)), col, crop2)
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(cov, 0.95, 0.65, dtype='FLOAT'))
    h = nb.math('ADD', nb.math('MULTIPLY', cov, 1.0), nb.math('MULTIPLY', fine.outputs['Fac'], 0.5))
    nb.set('Normal', nb.bump(h, strength=0.35, distance=0.05))
    nb.bsdf.inputs['Specular IOR Level'].default_value = 0.3
    return m


def bark_mat(name="Bark", col=(0.10, 0.085, 0.07)):
    m = sb.mat(name, col, rough=0.85)
    nb = sb.NB(m)
    uv = nb.coord('UV')
    co = nb.coord('Object')
    fis = nb.voronoi(nb.mapping(co, scale=(9.0, 9.0, 1.6)), scale=3.0, feature='DISTANCE_TO_EDGE')
    n = nb.noise(co, scale=14.0, detail=8, rough=0.7)
    h = nb.math('ADD', nb.maprange(fis.outputs['Distance'], 0.0, 0.12, 0.0, 1.0), nb.math('MULTIPLY', n.outputs['Fac'], 0.5))
    nb.set('Normal', nb.bump(h, strength=0.8, distance=0.01))
    nb.set('Base Color', nb.mix(nb.maprange(h, 0.3, 1.2), (col[0] * 0.55, col[1] * 0.55, col[2] * 0.55, 1), (col[0] * 1.5, col[1] * 1.45, col[2] * 1.35, 1)))
    return m


def leaf_mat(name="Leaf", col=(0.045, 0.11, 0.025), var=0.35, autumn=0.0):
    """Alpha-cut leaf card (UV 0..1 -> pointed ellipse with midrib), per-leaf colour variation (face attribute 'lv'),
    translucent when backlit, waxy sheen."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nb = sb.NB(m)
    b = nb.bsdf
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    u = nb.math('SUBTRACT', nb.math('MULTIPLY', sep.outputs[0], 2.0), 1.0)    # -1..1 across
    v = sep.outputs[1]                                                         # 0 base .. 1 tip
    half = nb.math('MULTIPLY', nb.math('POWER', nb.math('SINE', nb.math('MULTIPLY', v, math.pi)), 0.75),
                   nb.maprange(v, 0.0, 0.25, 0.55, 1.0))
    inside = nb.math('LESS_THAN', nb.math('ABSOLUTE', u), half)
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "lv"; at.attribute_type = 'GEOMETRY'
    lv = at.outputs['Fac']
    c0 = (col[0] * (1 - var), col[1] * (1 - var * 0.6), col[2] * (1 - var), 1)
    c1 = (col[0] * (1 + var * 1.4), col[1] * (1 + var), col[2] * (1 + var * 0.3), 1)
    lc = nb.mix(lv, c0, c1)
    if autumn > 0:
        lc = nb.mix(nb.maprange(lv, 1 - autumn, 1.0), lc, (0.35, 0.22, 0.03, 1))
    rib = nb.maprange(nb.math('ABSOLUTE', u), 0.0, 0.06, 1.0, 0.0)
    lc = nb.mix(nb.math('MULTIPLY', rib, 0.35), lc, (col[0] * 1.8, col[1] * 1.6, col[2] * 1.2, 1))
    nb.set('Base Color', lc)
    nb.set('Roughness', 0.42)
    nb.set('Alpha', inside)
    b.inputs['Specular IOR Level'].default_value = 0.45
    # backlit translucency
    tr = nb.new('ShaderNodeBsdfTranslucent')
    nb.link(nb.mix(1.0, lc, (1.6, 1.8, 0.9, 1), blend='MULTIPLY'), tr.inputs[0])
    mix = nb.new('ShaderNodeMixShader'); mix.inputs[0].default_value = 0.35
    out = nb.out
    nb.link(b.outputs[0], mix.inputs[1]); nb.link(tr.outputs[0], mix.inputs[2])
    # keep alpha: wrap with transparent by the same mask
    tp = nb.new('ShaderNodeBsdfTransparent')
    mx2 = nb.new('ShaderNodeMixShader'); nb.link(inside, mx2.inputs[0]); nb.link(tp.outputs[0], mx2.inputs[1]); nb.link(mix.outputs[0], mx2.inputs[2])
    nb.link(mx2.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'DITHERED'
    try:
        m.use_transparent_shadow = True
    except Exception:
        pass
    return m


# =========================================================================== trees
def _branch_pts(p0, d, L, n=6, droop=0.0, wig=0.08, rnd=None):
    pts = [V(p0)]
    p = V(p0); dd = V(d).normalized()
    for i in range(n):
        dd = (dd + V((rnd.uniform(-wig, wig), rnd.uniform(-wig, wig), rnd.uniform(-wig, wig) - droop))).normalized()
        p = p + dd * (L / n)
        pts.append(p.copy())
    return pts


def tree(name, height=11.0, crown_r=4.2, crown_h=None, n_main=3, leaf_count=90000, leaf_size=0.085, seed=1,
         bark=None, leaf=None, lean=0.0, lod=False, clear_trunk=0.3):
    """Deciduous tree. Trunk forks into leaders; each leader carries side branches; foliage lives in lobes around
    the branch ends (dense outer shells -> bumpy, self-shadowing silhouette like a real canopy).
    Returns root empty at the trunk base."""
    rnd = random.Random(seed)
    bark = bark or bark_mat()
    leaf = leaf or leaf_mat()
    crown_h = crown_h or height * 0.66
    root = sb.empty(name)
    parts = []
    tb = height * clear_trunk
    tr0 = 0.020 * height
    trunk_pts = _branch_pts((0, 0, 0), (lean, 0, 1), tb, n=5, wig=0.04, rnd=rnd)
    parts.append(E.sweep(name + "Trunk", trunk_pts, radius=tr0, segs=10 if lod else 18, mat=bark, sub=3,
                         rad_fn=lambda u: 1.3 - 0.35 * u + 0.4 * max(0, 0.06 - u) / 0.06))
    top = V(trunk_pts[-2]).lerp(V(trunk_pts[-1]), 0.6)
    lobes = []            # (centre, radius)
    for k in range(n_main):
        a = 2 * math.pi * k / n_main + rnd.uniform(-0.4, 0.4)
        spread = rnd.uniform(0.35, 0.6)
        d = V((math.cos(a) * spread, math.sin(a) * spread, 1.0)).normalized()
        Ll = (height - tb) * rnd.uniform(0.62, 0.8)
        lp = _branch_pts(top, d, Ll, n=6, droop=-0.01, wig=0.1, rnd=rnd)
        parts.append(E.sweep(f"{name}L{k}", lp, radius=tr0 * 0.72, segs=8 if lod else 12, mat=bark, sub=2, rad_fn=lambda u: 1.0 - 0.8 * u))
        lobes.append((V(lp[-1]) + V((0, 0, -0.3)), crown_r * rnd.uniform(0.42, 0.55)))
        nside = 3 if lod else 5
        for s_ in range(nside):
            u = rnd.uniform(0.2, 0.9)
            i = min(len(lp) - 2, int(u * (len(lp) - 1)))
            p = V(lp[i]).lerp(V(lp[i + 1]), rnd.random())
            sa = a + rnd.uniform(-1.3, 1.3)
            sd = V((math.cos(sa), math.sin(sa), rnd.uniform(-0.05, 0.6))).normalized()
            sL = crown_r * rnd.uniform(0.45, 0.8) * (1.1 - 0.5 * u)
            sp = _branch_pts(p, sd, sL, n=4, droop=0.04, wig=0.15, rnd=rnd)
            parts.append(E.sweep(f"{name}S{k}_{s_}", sp, radius=tr0 * 0.3, segs=5 if lod else 7, mat=bark, sub=2, rad_fn=lambda u: 1.0 - 0.8 * u))
            lobes.append((V(sp[-1]), crown_r * rnd.uniform(0.3, 0.45)))
            if not lod:
                for t_ in range(2):
                    q = V(sp[rnd.randrange(1, len(sp))])
                    td = (sd + V((rnd.uniform(-0.8, 0.8), rnd.uniform(-0.8, 0.8), rnd.uniform(-0.1, 0.7)))).normalized()
                    tw = _branch_pts(q, td, sL * 0.45, n=3, droop=0.05, wig=0.25, rnd=rnd)
                    parts.append(E.sweep(f"{name}T{k}_{s_}_{t_}", tw, radius=tr0 * 0.1, segs=4, mat=bark, sub=1, rad_fn=lambda u: 1.0 - 0.7 * u))
                    lobes.append((V(tw[-1]), crown_r * rnd.uniform(0.22, 0.32)))
    # leaves: sample lobes by volume, dense toward each lobe's outer shell
    vols = np.array([r ** 3 for _, r in lobes]); vols /= vols.sum()
    verts, faces, uvs, lv = [], [], [], []
    n_leaves = leaf_count // (3 if lod else 1)
    choice = np.random.RandomState(seed).choice(len(lobes), size=n_leaves, p=vols)
    for li in choice:
        c, r = lobes[li]
        dirv = V((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1) * 0.85)).normalized()
        rr = r * (rnd.random() ** 0.2)
        p = c + dirv * rr
        if p.z < tb * 0.9:
            p.z = tb * 0.9 + rnd.random() * 0.3
        nrm = (dirv * 0.8 + V((0, 0, 0.7)) + V((rnd.uniform(-0.7, 0.7), rnd.uniform(-0.7, 0.7), rnd.uniform(-0.5, 0.5)))).normalized()
        tang = Quaternion(nrm, rnd.uniform(0, 2 * math.pi)) @ nrm.orthogonal().normalized()
        tang = (tang + V((0, 0, -0.5))).normalized()          # leaves hang a little
        tang = (tang - nrm * tang.dot(nrm)).normalized()
        bit = nrm.cross(tang)
        sz = leaf_size * rnd.uniform(0.7, 1.3) * (2.0 if lod else 1.0)
        base = p - tang * sz * 0.5
        c0 = base - bit * sz * 0.33; c1 = base + bit * sz * 0.33
        c2 = c1 + tang * sz; c3 = c0 + tang * sz
        b = len(verts)
        verts += [tuple(c0), tuple(c1), tuple(c2), tuple(c3)]
        faces.append((b, b + 1, b + 2, b + 3))
        uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
        # inner leaves a touch darker/yellower (less light), outer greener
        lv.append(min(1.0, max(0.0, rnd.random() * 0.7 + 0.3 * (rr / r))))
    lo = E.mesh_np(name + "Leaves", verts, np.array(faces), leaf, np.array(uvs), smooth=False)
    at = lo.data.attributes.new("lv", 'FLOAT', 'FACE'); at.data.foreach_set("value", np.array(lv, np.float32))
    parts.append(lo)
    for p in parts:
        p.parent = root
    return root


def tree_library(n=4, seed=100, **kw):
    """Build n tree variants in a hidden collection; returns list of (collection) for collection instancing."""
    lib = []
    for i in range(n):
        c = bpy.data.collections.new(f"TreeLib{seed}_{i}")
        bpy.context.scene.collection.children.link(c)
        before = set(bpy.data.objects)
        tree(f"TreeV{seed}_{i}", seed=seed + i * 7, **kw)
        for o in bpy.data.objects:
            if o not in before:
                for uc in o.users_collection:
                    uc.objects.unlink(o)
                c.objects.link(o)
        lc = bpy.context.view_layer.layer_collection.children.get(c.name)
        if lc:
            lc.exclude = True
        lib.append(c)
    return lib


def place_trees(lib, points, scale_rng=(0.8, 1.2), seed=5, name="Tree"):
    rnd = random.Random(seed)
    out = []
    for i, p in enumerate(points):
        c = lib[rnd.randrange(len(lib))]
        s = rnd.uniform(*scale_rng)
        o = sb.collection_instance(c, loc=tuple(p), rot=(0, 0, rnd.uniform(0, 2 * math.pi)), scale=s, name=f"{name}{i}")
        out.append(o)
    return out
