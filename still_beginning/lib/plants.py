"""STILL BEGINNING - living plants for the Mars greenhouse (s24) (and reusable elsewhere).

leaf_mesh(...)      parametric leaf blade (UV: u across -1..1, v along 0..1) with serration, cupping, curl, ruffle
leaf_mat(...)       living leaf: vein network from UV (midrib + pinnate secondaries + areoles), translucency (light
                    through the blade shows veins), glossy cuticle top / matte paler underside, subtle chlorosis-free
                    healthy variation, optional water droplets are separate geometry (droplets())
basil(...), lettuce(...), tomato(...)   plants built from leaves + stems; return dict(root, leaves=[obj...])
All units metres."""
import bpy, bmesh, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler, noise
import sb


def leaf_mesh(name, L=0.07, W=0.04, nu=16, nv=30, serr=0.0, nserr=18, cup=0.25, curl=0.15, ruffle=0.0, twist=0.0,
              tip=0.9, base=0.35, seed=0, mat=None, petiole=0.0, widest=0.38, pucker=0.0, arch=0.12, subd=1):
    """blade in local frame: base at origin, midrib +Y, upper surface +Z.
    cup > 0: edges up (lettuce/boat); cup < 0: convex/domed (basil). arch: midrib rises then (curl) droops to the tip.
    pucker: bumps between veins (savoy basil/lettuce). ruffle: smooth margin waves."""
    rng = random.Random(seed)
    off = V((rng.uniform(0, 50), rng.uniform(0, 50), 0))
    p_exp = math.log(0.5) / math.log(widest)
    verts, uvs = [], []
    for j in range(nv + 1):
        v = j / nv
        sh = math.sin(math.pi * min(1.0, v ** p_exp)) ** tip
        sh *= sb.smoother(v / 0.10) ** 0.6 * (1 - base) + base * (1.0 if v > 0.02 else 0.3)
        sh = max(sh, 0.0)
        hw = W / 2 * sh
        if serr > 0:
            saw = (v * nserr + 0.3) % 1.0
            hw *= 1 + serr * (0.5 - abs(saw - 0.5)) * 2 * min(1.0, v * 3) * (1.2 - v)
        for i in range(nu + 1):
            u = -1 + 2 * i / nu
            x = u * hw
            y = v * L
            au = abs(u)
            z = cup * (au ** 1.8) * hw
            z -= 0.0010 * math.exp(-(u * 7) ** 2) * sh                       # midrib groove
            z += arch * L * math.sin(math.pi * min(1.0, v * 1.3)) * 0.35 - curl * (v ** 2.2) * L
            if ruffle > 0:
                z += ruffle * hw * (au ** 2.2) * math.sin(v * 22.0 + off.x) * math.cos(u * 2.0 + off.y * 0.1)
            if pucker > 0:
                z += pucker * hw * 0.08 * (0.5 + 0.5 * math.sin(v * 38 - au * 9 + off.x)) * math.sin(au * 7.0) * sh
            z += 0.0006 * noise.noise(V((x * 50, y * 50, 0)) + off) * sh
            pt = V((x, y, z))
            if twist:
                pt = Matrix.Rotation(twist * v, 3, 'Y') @ pt
            verts.append(pt)
            uvs.append((u, v))
    faces = []
    for j in range(nv):
        for i in range(nu):
            a_ = j * (nu + 1) + i
            faces.append((a_, a_ + 1, a_ + nu + 2, a_ + nu + 1))
    o = sb.mesh_obj(name, verts, faces, mat)
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    fl = []
    for pg in me.polygons:
        for li in pg.loop_indices:
            fl.append(uvs[me.loops[li].vertex_index])
    uvl.uv.foreach_set("vector", np.asarray(fl, np.float32).ravel())
    if subd:
        sb.subsurf(o, 0, subd)
    sb.solidify(o, 0.0003, offset=0)
    return o


def leaf_mat(name="Leaf", color=(0.035, 0.11, 0.018), gloss=0.35, translucency=0.18, vein_contrast=1.0, veins=7.0,
             lighter_under=True, var=0.15, seed=0.0, hairy=0.0):
    """living leaf. Uses UV (u across -1..1, v along 0..1) for veins; object coords for variation."""
    m = sb.mat(name, color, rough=0.38, spec=0.5, coat=gloss, coat_rough=0.18, sheen=hairy)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs['Subsurface Weight'].default_value = 0.0
    nb = sb.NB(m)
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    au = nb.math('ABSOLUTE', u)
    # midrib
    mid = nb.maprange(au, 0.035, 0.0, 0.0, 1.0)
    mid = nb.math('MULTIPLY', mid, nb.maprange(v, 1.0, 0.6, 0.3, 1.0))
    # pinnate secondary veins: lines leaving the midrib at ~50deg toward the tip
    ph = nb.math('SUBTRACT', nb.math('MULTIPLY', v, veins), nb.math('MULTIPLY', au, 0.9))
    ph = nb.math('ADD', ph, nb.math('MULTIPLY', nb.noise(uv, scale=3.0, detail=2).outputs['Fac'], 0.25))
    sec = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', ph), 0.5)), 0.44, 0.5, 0.0, 1.0)
    sec = nb.math('MULTIPLY', sec, nb.maprange(au, 0.95, 0.6, 0.0, 1.0))
    # tertiary network (areoles)
    ar = nb.voronoi(nb.mapping(uv, scale=(1.0, 2.2, 1.0)), scale=22.0 + seed, feature='DISTANCE_TO_EDGE')
    ter = nb.maprange(ar.outputs['Distance'], 0.0, 0.035, 1.0, 0.0)
    ter = nb.math('MULTIPLY', ter, 0.45)
    vein = nb.math('MAXIMUM', nb.math('MAXIMUM', mid, sec), ter)
    vein = nb.math('MULTIPLY', vein, vein_contrast)
    # colour: healthy variation, lighter veins, slightly yellower margins, lighter near base
    co = nb.coord('Object')
    n1 = nb.noise(co, scale=40.0, detail=4, rough=0.6)
    c0 = (color[0] * (1 - var), color[1] * (1 - var * 0.6), color[2] * (1 - var), 1)
    c1 = (color[0] * (1 + var), color[1] * (1 + var * 0.8), color[2] * (1 + var * 0.4), 1)
    col = nb.mix(nb.maprange(n1.outputs['Fac'], 0.3, 0.7), c0, c1)
    col = nb.mix(nb.math('MULTIPLY', vein, 0.45), col, (color[0] * 1.5 + 0.02, color[1] * 1.35 + 0.025, color[2] * 1.3, 1))
    col = nb.mix(nb.maprange(au, 0.75, 1.0, 0.0, 0.25), col, (color[0] * 1.5, color[1] * 1.25, color[2] * 0.6, 1))
    # underside paler & matte (backfacing)
    gn = nb.new('ShaderNodeNewGeometry')
    back = gn.outputs['Backfacing']
    if lighter_under:
        col = nb.mix(nb.math('MULTIPLY', back, 0.6), col, (color[0] * 1.8 + 0.03, color[1] * 1.5 + 0.04, color[2] * 1.7 + 0.02, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(back, nb.maprange(n1.outputs['Fac'], 0.3, 0.7, 0.3, 0.45), 0.6, dtype='FLOAT'))
    nb.set('Coat Weight', nb.mix(back, gloss, 0.0, dtype='FLOAT'))
    # relief: veins sunken on top (raised below), fine cell texture, blistering between veins
    cells = nb.voronoi(co, scale=2600.0, feature='F1')
    blister = nb.maprange(ar.outputs['Distance'], 0.0, 0.08, 0.0, 1.0)
    h = nb.math('ADD', nb.math('MULTIPLY', vein, -1.0), nb.math('MULTIPLY', blister, 0.5))
    h = nb.math('ADD', h, nb.math('MULTIPLY', cells.outputs['Distance'], 0.15))
    nb.set('Normal', nb.bump(h, strength=0.35, distance=0.0004))
    # translucency: light through the blade (veins block a little) -> mix a Translucent BSDF
    out = nb.out
    pb = out.inputs['Surface'].links[0].from_socket
    tr = nb.new('ShaderNodeBsdfTranslucent')
    tcol = nb.mix(nb.math('MULTIPLY', vein, 0.7), (color[0] * 2.4 + 0.02, color[1] * 2.6 + 0.03, color[2] * 1.2, 1), (color[0] * 1.4, color[1] * 1.8, color[2] * 0.8, 1))
    nb.link(tcol, tr.inputs['Color'])
    mx = nb.new('ShaderNodeMixShader')
    mx.inputs[0].default_value = translucency
    nb.link(pb, mx.inputs[1]); nb.link(tr.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs['Surface'])
    return m


def stem_mat(name="Stem", color=(0.16, 0.30, 0.07)):
    m = sb.mat(name, color, rough=0.45, spec=0.5, coat=0.2, sheen=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    st = nb.noise(nb.mapping(co, scale=(1, 1, 30)), scale=80.0, detail=2)
    nb.set('Base Color', nb.mix(nb.maprange(st.outputs['Fac'], 0.3, 0.7), (color[0] * 0.8, color[1] * 0.85, color[2] * 0.8, 1), (color[0] * 1.3, color[1] * 1.15, color[2] * 1.1, 1)))
    return m


def droplet_mat(name="Water"):
    m = sb.mat(name, (1, 1, 1), rough=0.0, transmission=1.0, ior=1.333)
    try:
        m.surface_render_method = 'DITHERED'
        m.use_raytrace_refraction = True
        m.thickness_mode = 'SPHERE'
    except Exception:
        pass
    return m


def place_leaf(lf, base, direction, up, scale=1.0, roll=0.0):
    """orient a leaf: +Y along direction, +Z toward up."""
    d = V(direction).normalized()
    z = V(up) - d * V(up).dot(d); z.normalize()
    x = d.cross(z)
    Mr = Matrix((x, d, z)).transposed()
    if roll:
        Mr = Mr @ Matrix.Rotation(roll, 3, 'Y')
    lf.matrix_world = Matrix.Translation(base) @ Mr.to_4x4() @ Matrix.Scale(scale, 4)
    return lf


def _stem(name, pts, r0, r1, mat):
    return sb.tube_along(name, pts, radius=1.0, mat=mat, bevel_res=3, taper=lambda t: r0 + (r1 - r0) * t)


def basil(name, loc, seed=0, height=0.30, leaf_m=None, stem_m=None, nodes=7, lean=(0.0, 0.0), scale=1.0, branches=None):
    """bushy Genovese basil: several branches, short internodes, opposite leaf pairs rotating 90deg per node,
    glossy convex (downward-cupped) puckered ovate leaves on 1.5-2.5 cm petioles, small leaves at the tips."""
    rng = random.Random(seed)
    root = sb.empty(name, loc=loc)
    leaves = []
    lm = leaf_m or leaf_mat(name + "Leaf", (0.030, 0.105, 0.016), gloss=0.55, translucency=0.18)
    sm = stem_m or stem_mat(name + "Stem", (0.10, 0.20, 0.05))
    stems = []
    nbr = branches or (4 + rng.randint(0, 2))
    for bI in range(nbr):
        a = 2 * math.pi * bI / nbr + rng.uniform(-0.35, 0.35)
        tilt = rng.uniform(0.15, 0.45) if bI else 0.05
        h = height * rng.uniform(0.65, 1.0) * scale * (1.0 if bI else 1.1)
        base = V(loc) + V((math.cos(a), math.sin(a), 0)) * 0.006 * (1 if bI else 0) + V((0, 0, 0.01 * bI * 0.0))
        pts = []
        for k in range(14):
            t = k / 13
            bend = math.sin(tilt) * h * (t ** 1.3)
            p = base + V((math.cos(a) * bend + lean[0] * t * t, math.sin(a) * bend + lean[1] * t * t, math.cos(tilt) * h * t))
            pts.append(p)
        st = _stem(name + "Stem", pts, 0.0038 * scale, 0.0016 * scale, sm)
        stems.append(st)
        for k in range(nodes):
            t = 0.18 + 0.78 * k / (nodes - 1)
            fi = t * 13
            i = min(12, int(fi))
            p = pts[i].lerp(pts[i + 1], fi - i)
            sz = (0.078 - 0.050 * (t - 0.18) / 0.78) * scale * rng.uniform(0.85, 1.12)
            rot0 = k * math.pi / 2 + a + rng.uniform(-0.2, 0.2)
            for side in (0, math.pi):
                ang = rot0 + side
                elev = 0.35 - 0.25 * (1 - t) + rng.uniform(-0.15, 0.15)
                dirv = V((math.cos(ang), math.sin(ang), elev)).normalized()
                pl = 0.012 + 0.012 * (1 - t)
                pet_end = p + dirv * pl + V((0, 0, pl * 0.3))
                sb.tube_along(name + "Petiole", [p, p.lerp(pet_end, 0.5) + V((0, 0, pl * 0.12)), pet_end], radius=0.0012 * scale, mat=sm)
                lf = leaf_mesh(name + "Leaf", L=sz, W=sz * 0.64, cup=-0.55, curl=0.22 + rng.uniform(0, 0.12), arch=0.10,
                               pucker=0.9, serr=0.015, nserr=12, tip=0.85, widest=0.36, base=0.25,
                               seed=rng.randint(0, 9999), mat=lm, nu=14, nv=24, twist=rng.uniform(-0.15, 0.15))
                ld = V((dirv.x, dirv.y, dirv.z - 0.25)).normalized()
                place_leaf(lf, pet_end, ld, V((0, 0, 1)), 1.0, roll=rng.uniform(-0.2, 0.2))
                lf["node_t"] = t
                leaves.append(lf)
        tip_ = pts[-1]
        for k in range(4):
            ang = a + k * math.pi / 2 + 0.4
            dirv = V((math.cos(ang), math.sin(ang), 1.3))
            lf = leaf_mesh(name + "LeafTop", L=0.024 * scale, W=0.016 * scale, cup=0.5, curl=0.05, arch=0.05,
                           seed=rng.randint(0, 9999), mat=lm, nu=8, nv=12)
            place_leaf(lf, tip_, dirv, V((0, 0, 1)) - dirv * 0.3, 1.0)
            leaves.append(lf)
    return dict(root=root, leaves=leaves, stems=stems, mats=(lm, sm))


def lettuce(name, loc, seed=0, radius=0.12, leaf_m=None, n=26, color=(0.10, 0.24, 0.025)):
    """loose-leaf lettuce rosette: ruffled, pale-veined leaves spiralling out (golden angle)."""
    rng = random.Random(seed)
    lm = leaf_m or leaf_mat(name + "Leaf", color, gloss=0.25, translucency=0.25, veins=9.0, var=0.2)
    leaves = []
    for k in range(n):
        t = k / n
        ang = k * 2.39996 + rng.uniform(-0.1, 0.1)
        L = radius * (0.45 + 0.75 * t) * rng.uniform(0.9, 1.1)
        el = math.radians(75 - 55 * t)
        d = V((math.cos(ang) * math.cos(el), math.sin(ang) * math.cos(el), math.sin(el)))
        lf = leaf_mesh(name + "Leaf", L=L, W=L * 0.85, cup=0.45, curl=0.30 * t + 0.05, arch=0.05, ruffle=0.22 + 0.12 * t,
                       serr=0.0, tip=0.5, widest=0.6, base=0.6, pucker=0.5, seed=rng.randint(0, 9999), mat=lm, nu=20, nv=28)
        place_leaf(lf, V(loc) + V((math.cos(ang), math.sin(ang), 0)) * 0.006 * t + V((0, 0, 0.004)), d, V((0, 0, 1)))
        leaves.append(lf)
    return dict(leaves=leaves, mats=(lm,))


def tomato(name, loc, seed=0, height=1.3, leaf_m=None, stem_m=None, fruit_m=None, trusses=4):
    """staked tomato: vine up a string, compound leaves (leaflets), trusses of fruit (green -> red)."""
    rng = random.Random(seed)
    lm = leaf_m or leaf_mat(name + "Leaf", (0.09, 0.24, 0.05), gloss=0.2, translucency=0.35, veins=6.0, hairy=0.3)
    sm = stem_m or stem_mat(name + "Stem", (0.2, 0.33, 0.1))
    fm = fruit_m
    pts = [V(loc) + V((0.01 * math.sin(k * 0.9 + seed), 0.01 * math.cos(k * 0.7 + seed), height * k / 20)) for k in range(21)]
    st = _stem(name + "Vine", pts, 0.006, 0.003, sm)
    leaves, fruit = [], []
    for k in range(3, 20, 2):
        p = pts[k]
        ang = k * 2.4 + seed
        d = V((math.cos(ang), math.sin(ang), -0.15))
        rach = [p + d * 0.02 * i + V((0, 0, -0.003 * i * i)) for i in range(12)]
        sb.tube_along(name + "Rachis", rach, radius=0.0018, mat=sm)
        for i in (3, 6, 9, 11):
            for sgn in ((-1, 1) if i < 11 else (0,)):
                side = d.cross(V((0, 0, 1))).normalized()
                ld = (d * 0.6 + side * sgn).normalized() if sgn else d
                lf = leaf_mesh(name + "Leaflet", L=0.05 + 0.01 * i / 11, W=0.028, cup=0.2, curl=0.25, serr=0.12, nserr=10,
                               ruffle=0.15, seed=rng.randint(0, 9999), mat=lm, nu=10, nv=16)
                place_leaf(lf, rach[i], ld, V((0, 0, 1)), 1.0)
                leaves.append(lf)
    if fm:
        for tI in range(trusses):
            k = 5 + tI * 4
            p = pts[min(k, 20)]
            ang = k * 1.7 + seed
            for j in range(4):
                r = rng.uniform(0.017, 0.026)
                fp = p + V((math.cos(ang) * (0.05 + 0.015 * j), math.sin(ang) * (0.05 + 0.015 * j), -0.04 - 0.02 * j))
                fr = sb.prim("sphere", name + "Fruit", loc=fp, segments=24, ring_count=16, radius=r, mat=fm[(tI + j) % len(fm)])
                fr.scale = (1, 1, 0.85)
                fruit.append(fr)
    return dict(leaves=leaves, stem=st, fruit=fruit)


def droplets(name, leaf, n=6, rmin=0.0012, rmax=0.003, seed=0, mat=None):
    """water beads resting on the upper surface of a leaf (flattened spheres on sampled face centres)."""
    rng = random.Random(seed)
    me = leaf.data
    mw = leaf.matrix_world
    polys = list(me.polygons)
    out = []
    for i in range(n):
        p = polys[rng.randrange(len(polys))]
        c = mw @ p.center
        nrm = (mw.to_3x3() @ p.normal).normalized()
        if nrm.z < 0.2:
            nrm = -nrm
        if nrm.z < 0.2:
            continue
        r = rng.uniform(rmin, rmax)
        d = sb.prim("sphere", name, loc=c + nrm * r * 0.45, segments=24, ring_count=12, radius=r, mat=mat)
        d.rotation_mode = 'QUATERNION'; d.rotation_quaternion = nrm.to_track_quat('Z', 'Y')
        d.scale = (1, 1, 0.62)
        out.append(d)
    return out


def merge(objs, name):
    """apply modifiers and join plant parts into one mesh (fast to render); curves are converted."""
    import suit
    ms = []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        if o.type not in ('MESH', 'CURVE'):
            continue
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
        me.transform(o.matrix_world)
        n = bpy.data.objects.new(o.name + "_m", me)
        sb.link_obj(n)
        ms.append(n)
    for o in objs:
        if o.type in ('MESH', 'CURVE'):
            bpy.data.objects.remove(o)
    if not ms:
        return None
    return sb.join(ms, name)


def plant_objects(prefix):
    return [o for o in bpy.data.objects if o.name.startswith(prefix)]


def chard_leaf_mat(name="ChardLeaf", color=(0.028, 0.085, 0.018), rib=(0.85, 0.42, 0.06), translucency=0.22):
    """savoyed dark-green blade with coloured (amber/ruby) midrib + secondary veins that glow when backlit."""
    m = leaf_mat(name, color, gloss=0.45, translucency=translucency, veins=6.5, vein_contrast=1.0, var=0.12)
    nb = sb.NB(m)
    b = nb.bsdf
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    au = nb.math('ABSOLUTE', u)
    mid = nb.math('MULTIPLY', nb.maprange(au, 0.07, 0.03, 0.0, 1.0), nb.maprange(v, 1.0, 0.5, 0.2, 1.0))
    ph = nb.math('SUBTRACT', nb.math('MULTIPLY', v, 6.5), nb.math('MULTIPLY', au, 0.9))
    ph = nb.math('ADD', ph, nb.math('MULTIPLY', nb.noise(uv, scale=3.0, detail=2).outputs['Fac'], 0.25))
    sec = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', ph), 0.5)), 0.455, 0.5, 0.0, 1.0)
    sec = nb.math('MULTIPLY', sec, nb.maprange(au, 0.85, 0.3, 0.0, 0.8))
    rv = nb.math('MAXIMUM', mid, sec)
    src = b.inputs['Base Color'].links[0].from_socket
    nb.set('Base Color', nb.mix(rv, src, (*rib, 1)))
    # translucent colour of the ribs: warm glow
    for n_ in m.node_tree.nodes:
        if n_.type == 'BSDF_TRANSLUCENT':
            tsrc = n_.inputs['Color'].links[0].from_socket
            nb.link(nb.mix(rv, tsrc, (rib[0] * 1.2, rib[1] * 1.1, rib[2], 1)), n_.inputs['Color'])
    return m


def chard(name, loc, seed=0, leaf_m=None, stalk_m=None, n=9, size=1.0):
    """rainbow/golden chard rosette: thick coloured stalks arching out, big savoyed glossy blades."""
    rng = random.Random(seed)
    lm = leaf_m or chard_leaf_mat(name + "Leaf")
    stm = stalk_m or sb.mat(name + "Stalk", (0.80, 0.40, 0.06), rough=0.3, coat=0.5, sss=0.3, sss_radius=(1, 0.5, 0.2), sss_scale=0.01)
    leaves = []
    for k in range(n):
        t = k / max(1, n - 1)
        ang = k * 2.39996 + rng.uniform(-0.15, 0.15)
        el = math.radians(78 - 40 * t + rng.uniform(-6, 6))
        L = (0.24 - 0.08 * (1 - t) ** 2) * size * rng.uniform(0.85, 1.08)
        sl = (0.14 + 0.08 * t) * size
        d = V((math.cos(ang) * math.cos(el), math.sin(ang) * math.cos(el), math.sin(el)))
        base = V(loc) + V((math.cos(ang), math.sin(ang), 0)) * 0.01
        pts = [base + d * sl * (i / 8) + V((0, 0, -0.35 * sl * (i / 8) ** 2 * math.cos(el))) for i in range(9)]
        sb.tube_along(name + "Stalk", pts, radius=1.0, mat=stm, bevel_res=3, taper=lambda tt: (0.0075 - 0.0035 * tt) * size)
        tipd = (pts[-1] - pts[-2]).normalized()
        lf = leaf_mesh(name + "Leaf", L=L, W=L * 0.62, cup=0.25, curl=0.30 + 0.2 * t, arch=0.06, pucker=1.6, ruffle=0.12,
                       tip=0.7, widest=0.45, base=0.3, seed=rng.randint(0, 9999), mat=lm, nu=22, nv=34,
                       twist=rng.uniform(-0.25, 0.25))
        place_leaf(lf, pts[-1] - tipd * 0.004, tipd, V((0, 0, 1)), 1.0, roll=rng.uniform(-0.15, 0.15))
        lf["stalk_end"] = tuple(pts[-1])
        leaves.append(lf)
    return dict(leaves=leaves, mats=(lm, stm))
