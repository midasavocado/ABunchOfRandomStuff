"""ENERGY -> ABUNDANCE -> CITY sequence helpers (s10-s16). Geometry builders with proper UVs,
industrial hardware (bolts, busbars, hoses, fins), and physically based industrial materials.
Units: metres. Owner: energy/city artist."""
import bpy, bmesh, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion
import sb

COPPER = (0.92, 0.50, 0.33)


# =========================================================================== geometry

def mesh_np(name, verts, faces, mat=None, uvs=None, smooth=True, coll=None, mats=None, mat_idx=None):
    """Fast mesh from numpy arrays. faces: (F,4) or (F,3) int array or list of tuples.
    uvs: per-loop UV array (L,2) in face-corner order (optional)."""
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        n = faces.shape[1]
        me.vertices.add(len(verts))
        me.vertices.foreach_set("co", verts.ravel())
        me.loops.add(faces.size)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(len(faces))
        me.polygons.foreach_set("loop_start", (np.arange(len(faces)) * n).astype(np.int32))
        me.polygons.foreach_set("loop_total", np.full(len(faces), n, np.int32))
        me.update(calc_edges=True)
    else:
        me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
        me.update()
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", np.asarray(uvs, np.float32).ravel())
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o, coll)
    for m in (mats or ([mat] if mat else [])):
        me.materials.append(m)
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    return o


def smooth_path(pts, sub=6):
    """Catmull-Rom resample of control points (list of 3-tuples) -> numpy (N,3)."""
    P = [V(p) for p in pts]
    if len(P) < 3:
        out = [P[0].lerp(P[-1], i / (sub * 2)) for i in range(sub * 2 + 1)]
        return np.array([tuple(p) for p in out])
    out = []
    n = len(P) - 1
    for i in range(n):
        p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, n)]
        for k in range(sub):
            u = k / sub
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-1])
    return np.array([tuple(p) for p in out])


def frames_along(P, up=(0, 0, 1)):
    """Parallel-transport frames along polyline P (N,3). Returns T,N,B arrays."""
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    Nn = np.zeros_like(T); Bn = np.zeros_like(T)
    u = np.array(up, float)
    if abs(np.dot(u, T[0])) > 0.95:
        u = np.array((1.0, 0, 0)) if abs(T[0][0]) < 0.9 else np.array((0, 1.0, 0))
    n0 = u - np.dot(u, T[0]) * T[0]
    n0 /= np.linalg.norm(n0)
    Nn[0] = n0
    for i in range(1, len(P)):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        n = Nn[i - 1]
        if s > 1e-9:
            ax = v / s
            ang = math.atan2(s, np.dot(T[i - 1], T[i]))
            n = (n * math.cos(ang) + np.cross(ax, n) * math.sin(ang) + ax * np.dot(ax, n) * (1 - math.cos(ang)))
        n = n - np.dot(n, T[i]) * T[i]
        Nn[i] = n / (np.linalg.norm(n) + 1e-12)
    Bn = np.cross(T, Nn)
    return T, Nn, Bn


def sweep(name, pts, radius=0.01, segs=24, mat=None, sub=6, caps=True, rad_fn=None, resample=True,
          profile=None, up=(0, 0, 1), coll=None):
    """Tube (or arbitrary closed profile) swept along a smooth path, with UVs: u = arc length (m),
    v = around (0..1). profile: list of (x, y) in the N/B plane (closed loop), overrides radius/segs."""
    P = smooth_path(pts, sub) if resample else np.asarray(pts, float)
    T, Nn, Bn = frames_along(P, up)
    seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seglen)])
    L = s[-1]
    if profile is None:
        ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        prof = np.stack([np.cos(ang), np.sin(ang)], 1)
    else:
        prof = np.asarray(profile, float)
        segs = len(prof)
    verts = []
    for i in range(len(P)):
        r = radius * (rad_fn(s[i] / L) if rad_fn else 1.0)
        ring = P[i] + r * (prof[:, 0:1] * Nn[i] + prof[:, 1:2] * Bn[i])
        verts.append(ring)
    verts = np.concatenate(verts)
    faces, uvs = [], []
    n = len(P)
    for i in range(n - 1):
        for j in range(segs):
            j1 = (j + 1) % segs
            faces.append((i * segs + j, (i + 1) * segs + j, (i + 1) * segs + j1, i * segs + j1))
            uvs += [(s[i], j / segs), (s[i + 1], j / segs), (s[i + 1], (j + 1) / segs), (s[i], (j + 1) / segs)]
    faces = np.array(faces)
    uvs = np.array(uvs)
    if caps:
        c0 = len(verts); verts = np.vstack([verts, P[0], P[-1]])
        extra_f, extra_uv = [], []
        for j in range(segs):
            j1 = (j + 1) % segs
            extra_f.append((c0, j1, j, j))
            extra_f.append((c0 + 1, (n - 1) * segs + j, (n - 1) * segs + j1, (n - 1) * segs + j1))
            extra_uv += [(0, 0)] * 8
        # triangles as degenerate quads are ugly; build as tris via from_pydata path
        tri = [(f[0], f[1], f[2]) for f in extra_f]
        allf = [tuple(f) for f in faces] + tri
        o = mesh_np(name, verts, allf, mat, None, coll=coll)
        uvl = o.data.uv_layers.new(name="UVMap")
        flat = list(map(tuple, uvs)) + [(0.0, 0.0)] * (3 * len(tri))
        uvl.data.foreach_set("uv", np.array(flat, np.float32).ravel())
        # flat-shade caps
        sm = np.ones(len(o.data.polygons), bool); sm[len(faces):] = False
        o.data.polygons.foreach_set("use_smooth", sm)
        return o
    return mesh_np(name, verts, faces, mat, uvs, coll=coll)


def box(name, size, loc=(0, 0, 0), mat=None, bev=0.0, bev_seg=3, rot=(0, 0, 0), parent=None):
    o = sb.prim("cube", name, loc=loc, rot=rot, scale=(size[0] / 2, size[1] / 2, size[2] / 2), mat=mat, parent=parent)
    if bev > 0:
        sb.apply_scale(o) if hasattr(sb, "apply_scale") else apply_scale(o)
        m = sb.bevel(o, bev, bev_seg)
        m.harden_normals = True
    return o


def apply_scale(o):
    me = o.data
    S = Matrix.Diagonal((*o.scale, 1.0))
    me.transform(S)
    o.scale = (1, 1, 1)
    me.update()


def cyl(name, r, h, loc=(0, 0, 0), rot=(0, 0, 0), mat=None, verts=48, bev=0.0, parent=None):
    o = sb.prim("cyl", name, loc=loc, rot=rot, vertices=verts, radius=r, depth=h, mat=mat, parent=parent)
    if bev > 0:
        m = sb.bevel(o, bev, 3)
        m.harden_normals = True
    sb.shade_auto(o, 40)
    return o


def lathe_obj(name, profile, segs=64, mat=None, loc=(0, 0, 0), rot=(0, 0, 0), parent=None, smooth_angle=40):
    o = sb.lathe(name, profile, segs=segs, mat=mat)
    o.location = loc
    o.rotation_euler = rot
    if parent:
        o.parent = parent
    sb.shade_auto(o, smooth_angle)
    return o


def _hex_radius(th, af):
    """Boundary radius of a hexagon (across-flats af, flats facing +-X rotated 30deg) at polar angle th."""
    a = af / 2.0
    k = ((th + math.pi / 6) % (math.pi / 3)) - math.pi / 6
    return a / math.cos(k)


def hex_head_mesh(name, af=0.018, h=0.008, chamfer_top=True, chamfer_bot=False, washer_face=0.0, bore=0.0,
                  nr=10, per_side=12):
    """Hex prism with the classic 30deg conical chamfer on top (and optionally bottom). Axis +Z, base at z=0.
    washer_face: height of a thin round bearing collar under the head. bore: through-hole radius (for nuts)."""
    na = 6 * per_side
    rc = af / 2 * 0.93                  # chamfer circle radius (just inside the flats)
    tan30 = math.tan(math.radians(30))
    verts, faces = [], []
    def zc(r, top):
        d = max(0.0, r - rc)
        d = d if d > 0 else 0.0
        return (h - d * tan30) if top else (d * tan30)
    r0 = bore if bore > 0 else 0.0
    rings_top, rings_bot = [], []
    # top surface rings (from r0 or centre to hex boundary)
    for side, top in (("t", True), ("b", False)):
        ch = chamfer_top if top else chamfer_bot
        rings = []
        # rings: circles up to the chamfer circle rc (flat face), then morph from circle rc to the hex outline
        # so the flat/chamfer crease lies exactly on a ring (clean shading)
        n_in = max(2, nr // 2); n_out = nr - n_in
        for i in range(nr + 1):
            ring = []
            for j in range(na):
                th = 2 * math.pi * j / na
                R = _hex_radius(th, af)
                if r0 > 0:
                    rin = max(r0, 0.0)
                    if i <= n_in:
                        r = rin + (rc - rin) * i / n_in
                    else:
                        r = rc + (R - rc) * (i - n_in) / n_out
                else:
                    if i <= n_in:
                        r = rc * max(i / n_in, 0.02)
                    else:
                        r = rc + (R - rc) * (i - n_in) / n_out
                z = (zc(r, top) if ch else (h if top else 0.0))
                ring.append(len(verts)); verts.append((r * math.cos(th), r * math.sin(th), z + washer_face))
            rings.append(ring)
        if top: rings_top = rings
        else: rings_bot = rings
    def quad_strip(A, B, flip=False):
        for j in range(na):
            j1 = (j + 1) % na
            f = (A[j], A[j1], B[j1], B[j])
            faces.append(f[::-1] if flip else f)
    for i in range(nr):
        quad_strip(rings_top[i], rings_top[i + 1], flip=False)
        quad_strip(rings_bot[i], rings_bot[i + 1], flip=True)
    # side walls between outer rings
    quad_strip(rings_top[-1], rings_bot[-1], flip=False)
    if r0 > 0:   # bore wall
        quad_strip(rings_bot[0], rings_top[0], flip=False)
    else:        # close centre (tiny ring) with fans
        c1 = len(verts); verts.append((0, 0, h + washer_face)); c2 = len(verts); verts.append((0, 0, washer_face))
        for j in range(na):
            j1 = (j + 1) % na
            faces.append((c1, rings_top[0][j], rings_top[0][j1]))
            faces.append((c2, rings_bot[0][j1], rings_bot[0][j]))
    o = mesh_np(name, verts, faces, smooth=False)
    sb.recalc_normals(o)
    if washer_face > 0:
        col = sb.lathe(name + "Col", [(r0 if r0 > 0 else 0.0, 0.0), (af / 2 * 0.98, 0.0), (af / 2 * 0.98, washer_face * 0.7),
                                         (af / 2 * 0.95, washer_face), (r0 if r0 > 0 else 0.0, washer_face)], segs=na)
        o = sb.join([o, col], name)
    sb.shade_auto(o, 25)
    return o


def hex_bolt_mesh(name, d=0.012, head_h=None, shank=0.0, washer=True, mats=None):
    """ISO-style hex bolt head (chamfered, with bearing collar) + flat washer. Axis +Z, washer bottom at z=0."""
    af = d * 1.5
    hh = head_h or d * 0.64
    wt = 0.0022 if washer else 0.0
    head = hex_head_mesh(name + "H", af=af, h=hh, washer_face=0.0006, nr=12)
    head.location = (0, 0, wt)
    parts = [head]
    if washer:
        w = sb.lathe(name + "W", [(d * 0.54, 0.0), (d * 1.04, 0.0), (d * 1.05, 0.0004), (d * 1.05, wt - 0.0004),
                                  (d * 1.04, wt), (d * 0.54, wt), (d * 0.53, wt * 0.5)], segs=72)
        parts.append(w)
    for p in parts:
        bpy.context.view_layer.objects.active = p
    o = sb.join(parts, name)
    with bpy.context.temp_override(object=o, active_object=o, selected_editable_objects=[o], selected_objects=[o]):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    sb.shade_auto(o, 25)
    if mats:
        for m in mats:
            o.data.materials.append(m)
    return o


def nut_mesh(name, d=0.012, h=None):
    af = d * 1.5
    h = h or d * 0.8
    o = hex_head_mesh(name, af=af, h=h, chamfer_top=True, chamfer_bot=True, bore=d * 0.43)
    return o


def place_linked(src, loc, direction=(0, 0, 1), spin=0.0, name=None, parent=None, scale=1.0):
    o = src.copy()
    sb.link_obj(o)
    o.location = loc
    o.rotation_mode = 'QUATERNION'
    q = V(direction).normalized().to_track_quat('Z', 'Y') @ Quaternion((0, 0, 1), spin)
    o.rotation_quaternion = q
    o.scale = (scale,) * 3
    o.hide_render = False
    o.hide_viewport = False
    if parent:
        o.parent = parent
    if name:
        o.name = name
    return o


# =========================================================================== materials

def copper(name, rough=0.2, tarnish=0.35, flycut=False, lines_dir='X', scale=1.0, fingerprint=0.3):
    """Machined/rolled copper: rolling lines (anisotropic), tarnish/oxidation patches, faint heat tint.
    lines_dir: object axis the rolling lines run along."""
    m = sb.mat(name, COPPER, metal=1.0, rough=rough, aniso=0.55)
    nb = sb.NB(m)
    b = nb.bsdf
    try:
        b.inputs['Specular Tint'].default_value = (1.0, 0.82, 0.72, 1)   # F82 tint (edge colour)
    except Exception:
        pass
    co = nb.coord('Object')
    st = {'X': (1, 1, 1), 'Y': (1, 1, 1), 'Z': (1, 1, 1)}
    sc = {'X': (0.02, 1, 1), 'Y': (1, 0.02, 1), 'Z': (1, 1, 0.02)}[lines_dir]
    lines = nb.noise(nb.mapping(co, scale=tuple(v * 1.0 for v in sc)), scale=900 * scale, detail=3, rough=0.6)
    fine = nb.noise(nb.mapping(co, scale=tuple(v * 1.0 for v in sc)), scale=2600 * scale, detail=2, rough=0.5)
    patch = nb.noise(co, scale=14 * scale, detail=6, rough=0.62, dist=0.4)
    patch2 = nb.noise(co, scale=55 * scale, detail=4, rough=0.6)
    tmask = nb.maprange(patch.outputs['Fac'], 0.50, 0.72, 0.0, tarnish)
    tmask = nb.math('ADD', tmask, nb.maprange(patch2.outputs['Fac'], 0.62, 0.75, 0.0, tarnish * 0.5))
    bright = (0.94, 0.54, 0.36, 1)
    mid = (0.90, 0.47, 0.30, 1)
    dark = (0.50, 0.21, 0.11, 1)
    col = nb.mix(nb.maprange(lines.outputs['Fac'], 0.3, 0.7), mid, bright)
    col = nb.mix(tmask, col, dark)
    # faint heat tint (thin oxide film: straw / rose)
    ht = nb.maprange(patch.outputs['Fac'], 0.28, 0.40, 0.25 * tarnish, 0.0)
    col = nb.mix(ht, col, (0.85, 0.40, 0.42, 1))
    nb.set('Base Color', col)
    r = nb.maprange(lines.outputs['Fac'], 0.3, 0.7, rough * 0.75, rough * 1.3)
    r = nb.math('ADD', r, nb.math('MULTIPLY', tmask, 0.25))
    if fingerprint > 0:
        fp = nb.voronoi(co, scale=60 * scale, feature='SMOOTH_F1') if False else nb.noise(co, scale=120 * scale, detail=2, dist=3.0)
        fpm = nb.maprange(fp.outputs['Fac'], 0.62, 0.66, 0.0, fingerprint * 0.12)
        r = nb.math('ADD', r, fpm)
    nb.set('Roughness', r)
    h = nb.math('ADD', lines.outputs['Fac'], nb.math('MULTIPLY', fine.outputs['Fac'], 0.5))
    nb.set('Normal', nb.bump(h, strength=0.08, distance=0.0002))
    tang = nb.new('ShaderNodeTangent'); tang.direction_type = 'RADIAL'; tang.axis = lines_dir if lines_dir != 'X' else 'X'
    # anisotropic direction along rolling lines: radial tangent around a perpendicular axis approximates it
    return m


def flycut_mat(name, base=COPPER, rough=0.14, ring_scale=1.0, center=(0, 0, 0), plate=0.0):
    """Fly-cut / faced contact surface: concentric arc tool marks around `center` (object space).
    plate>0 mixes toward a silver tin plating."""
    col = tuple(base[i] * (1 - plate) + (0.80, 0.80, 0.80)[i] * plate for i in range(3))
    m = sb.mat(name, col, metal=1.0, rough=rough)
    nb = sb.NB(m)
    co = nb.coord('Object')
    rel = nb.vmath('SUBTRACT', co, center)
    d = nb.vmath('LENGTH', rel)
    rings = nb.math('SINE', nb.math('MULTIPLY', d, 6000 * ring_scale))
    n = nb.noise(co, scale=300, detail=4)
    h = nb.math('ADD', nb.math('MULTIPLY', rings, 0.3), n.outputs['Fac'])
    nb.set('Normal', nb.bump(h, strength=0.12, distance=0.0001))
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, rough * 0.7, rough * 1.4))
    return m


def stainless(name, color=(0.62, 0.62, 0.63), rough=0.22, brushed=True, scale=1.0, grime=0.15, dir_scale=(1, 1, 60)):
    m = sb.mat(name, color, metal=1.0, rough=rough, aniso=0.5 if brushed else 0.0)
    nb = sb.NB(m)
    co = nb.coord('Object')
    br = nb.noise(nb.mapping(co, scale=dir_scale), scale=500 * scale, detail=4, rough=0.6)
    big = nb.noise(co, scale=6 * scale, detail=6, rough=0.6)
    r = nb.maprange(br.outputs['Fac'], 0.3, 0.7, rough * 0.7, rough * 1.35)
    r = nb.math('ADD', r, nb.maprange(big.outputs['Fac'], 0.45, 0.75, 0.0, grime * 0.5))
    nb.set('Roughness', r)
    c = nb.mix(nb.maprange(big.outputs['Fac'], 0.4, 0.7), (color[0] * 0.92, color[1] * 0.92, color[2] * 0.93, 1), (*color, 1))
    nb.set('Base Color', c)
    nb.set('Normal', nb.bump(br.outputs['Fac'], strength=0.05, distance=0.0002))
    return m


def zinc(name, rough=0.3):
    """Zinc-plated (clear passivated) steel fastener with faint iridescence."""
    m = sb.mat(name, (0.78, 0.78, 0.77), metal=1.0, rough=rough, thin_film=250)
    b = m.node_tree.nodes["Principled BSDF"]
    try:
        b.inputs['Thin Film IOR'].default_value = 1.6
    except Exception:
        pass
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=900, detail=3)
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, rough * 0.9, rough * 1.12))
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.01, distance=0.0001))
    return m


def g10(name, color=(0.52, 0.46, 0.20)):
    """Glass-epoxy laminate (G10/FR4-like insulator): translucent-ish, woven-cloth texture visible."""
    m = sb.mat(name, color, rough=0.35, sss=0.25, sss_radius=(1.0, 0.8, 0.3), sss_scale=0.004, coat=0.4, coat_rough=0.15)
    nb = sb.NB(m)
    co = nb.coord('Object')
    w1 = nb.wave(co, scale=900, wtype='BANDS', direction='X')
    w2 = nb.wave(co, scale=900, wtype='BANDS', direction='Y')
    h = nb.math('MULTIPLY', w1.outputs['Fac'], w2.outputs['Fac'])
    n = nb.noise(co, scale=40, detail=5)
    c = nb.mix(nb.maprange(n.outputs['Fac'], 0.35, 0.7), (color[0] * 0.8, color[1] * 0.8, color[2] * 0.75, 1), (*color, 1))
    c = nb.mix(nb.math('MULTIPLY', h, 0.15), c, (0.8, 0.75, 0.5, 1))
    nb.set('Base Color', c)
    nb.set('Normal', nb.bump(h, strength=0.05, distance=0.0002))
    return m


def anodized(name, color=(0.04, 0.042, 0.047), rough=0.35, scale=1.0):
    """Bead-blasted anodized aluminium (matte-satin, fine speckle)."""
    m = sb.mat(name, color, metal=0.85, rough=rough)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=3000 * scale, detail=2)
    n2 = nb.noise(co, scale=20 * scale, detail=4)
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.06, distance=0.0001))
    nb.set('Roughness', nb.maprange(n2.outputs['Fac'], 0.3, 0.7, rough * 0.85, rough * 1.2))
    return m


def braided(name, color=(0.68, 0.68, 0.69), rough=0.28, pitch=260.0, around=24.0):
    """Stainless over-braid for hoses. Needs UV from sweep(): u = length (m), v = around (0..1)."""
    m = sb.mat(name, color, metal=1.0, rough=rough, aniso=0.3)
    nb = sb.NB(m)
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    u = nb.math('MULTIPLY', sep.outputs[0], pitch)
    v = nb.math('MULTIPLY', sep.outputs[1], around)
    a = nb.math('FRACT', nb.math('ADD', u, v))          # strand band A (helix +)
    bb = nb.math('FRACT', nb.math('SUBTRACT', u, v))    # strand band B (helix -)
    # over/under: which set is on top alternates every other crossing
    ca = nb.math('FLOOR', nb.math('ADD', u, v))
    cb = nb.math('FLOOR', nb.math('SUBTRACT', u, v))
    top = nb.math('MODULO', nb.math('ABSOLUTE', nb.math('ADD', ca, cb)), 2.0)
    # strand profile (rounded bundles of ~6 wires)
    pa = nb.math('SINE', nb.math('MULTIPLY', a, math.pi))
    pb = nb.math('SINE', nb.math('MULTIPLY', bb, math.pi))
    wires_a = nb.math('ABSOLUTE', nb.math('SINE', nb.math('MULTIPLY', a, math.pi * 6)))
    wires_b = nb.math('ABSOLUTE', nb.math('SINE', nb.math('MULTIPLY', bb, math.pi * 6)))
    ha = nb.math('MULTIPLY', pa, nb.math('ADD', 0.75, nb.math('MULTIPLY', wires_a, 0.25)))
    hb = nb.math('MULTIPLY', pb, nb.math('ADD', 0.75, nb.math('MULTIPLY', wires_b, 0.25)))
    h = nb.mix(top, ha, hb, dtype='FLOAT')
    h = nb.math('MAXIMUM', h, nb.math('MULTIPLY', nb.mix(top, hb, ha, dtype='FLOAT'), 0.55))
    nb.set('Normal', nb.bump(h, strength=0.9, distance=0.0006))
    occl = nb.maprange(h, 0.0, 0.5, 0.25, 1.0)
    nb.set('Base Color', nb.mix(occl, (0.12, 0.12, 0.12, 1), (*color, 1)))
    nb.set('Roughness', nb.maprange(h, 0.0, 1.0, rough * 1.8, rough))
    return m


def frost(name, base=(0.62, 0.62, 0.63), amount_axis=None, coverage=0.8, grad=None, radial=None):
    """Hoarfrost over a metal: white crystalline diffuse, sparkling facets. Uses object coords.
    grad=(axis_index, a, b): frost coverage fades from full at coord a to none at coord b."""
    m = sb.mat(name, (0.9, 0.92, 0.95), rough=0.55, sss=0.4, sss_radius=(0.5, 0.6, 0.8), sss_scale=0.002)
    nb = sb.NB(m)
    co = nb.coord('Object')
    v = nb.voronoi(co, scale=1400, feature='F1')
    n = nb.noise(co, scale=300, detail=6, rough=0.7)
    big = nb.noise(co, scale=25, detail=5, rough=0.6)
    cov = nb.maprange(big.outputs['Fac'], 0.5 - coverage * 0.4, 0.55 - coverage * 0.4 + 0.08, 0.0, 1.0)
    if grad is not None or radial is not None:
        if radial is not None:     # radial=(centre, r_full, r_none): frost bloom around a cold point
            dd = nb.vmath('DISTANCE', co, radial[0])
            g = nb.maprange(dd, radial[1], radial[2], 1.0, 0.0)
        else:
            sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
            g = nb.maprange(sep.outputs[grad[0]], grad[1], grad[2], 1.0, 0.0)
        feath = nb.noise(co, scale=260, detail=6, rough=0.7)
        gn = nb.math('ADD', g, nb.math('MULTIPLY', nb.math('SUBTRACT', big.outputs['Fac'], 0.5), 0.9))
        gn = nb.math('ADD', gn, nb.math('MULTIPLY', nb.math('SUBTRACT', feath.outputs['Fac'], 0.5), 0.7))
        cov = nb.math('MULTIPLY', cov, nb.maprange(gn, 0.25, 0.75, 0.0, 1.0, clamp=True))
        cov = nb.math('POWER', cov, 0.8)
    lump = nb.noise(co, scale=70, detail=4, rough=0.6)
    h = nb.math('ADD', nb.math('MULTIPLY', n.outputs['Fac'], 0.5), nb.math('MULTIPLY', v.outputs['Distance'], 0.6))
    h = nb.math('ADD', h, nb.math('MULTIPLY', lump.outputs['Fac'], 0.9))
    hb = nb.math('MULTIPLY', h, cov)
    nb.set('Normal', nb.bump(hb, strength=1.0, distance=0.0012))
    spk = nb.voronoi(co, scale=2200, feature='F1')
    sp = nb.maprange(spk.outputs['Color'], 0.0, 1.0, 0.7, 0.1)
    sepc = nb.new('ShaderNodeSeparateColor'); nb.link(spk.outputs['Color'], sepc.inputs[0])
    sp = nb.maprange(sepc.outputs[0], 0.80, 0.86, 0.65, 0.08)
    nb.set('Roughness', nb.mix(cov, 0.3, sp, dtype='FLOAT'))
    thick = nb.maprange(nb.math('ADD', lump.outputs['Fac'], nb.math('MULTIPLY', v.outputs['Distance'], 0.4)), 0.35, 0.95)
    col = nb.mix(cov, (*base, 1), nb.mix(thick, (0.36, 0.40, 0.46, 1), (0.74, 0.78, 0.84, 1)))
    nb.set('Base Color', col)
    nb.set('Metallic', nb.maprange(cov, 0.0, 1.0, 1.0, 0.0))
    return m


def painted_steel(name, color, rough=0.45, wear=0.2, scale=1.0):
    """Industrial 2K paint over steel with edge wear driven by pointiness-free noise + AO-ish grime."""
    m = sb.mat(name, color, rough=rough, coat=0.15, coat_rough=0.2)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=4 * scale, detail=8, rough=0.65)
    g = nb.noise(co, scale=20 * scale, detail=6, rough=0.6)
    c = nb.mix(nb.maprange(n.outputs['Fac'], 0.35, 0.7), (color[0] * 0.85, color[1] * 0.85, color[2] * 0.85, 1), (*color, 1))
    c = nb.mix(nb.maprange(g.outputs['Fac'], 0.6, 0.8, 0.0, wear), c, (color[0] * 0.5, color[1] * 0.48, color[2] * 0.45, 1))
    nb.set('Base Color', c)
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, rough * 0.85, rough * 1.2))
    orange = nb.noise(co, scale=260 * scale, detail=2)
    nb.set('Normal', nb.bump(orange.outputs['Fac'], strength=0.03, distance=0.0004))
    return m


def sphere_probe(loc=(0, 0, 0), radius=5.0, clip=0.01, name="Probe"):
    try:
        pd = bpy.data.lightprobes.new(name, 'SPHERE')
    except Exception:
        return None
    try:
        pd.influence_distance = radius
        pd.clip_start = clip
    except Exception:
        pass
    o = bpy.data.objects.new(name, pd)
    sb.link_obj(o)
    o.location = loc
    return o


def volume_probe(loc=(0, 0, 0), size=(5, 5, 3), res=(8, 8, 4), name="IrrVol"):
    try:
        pd = bpy.data.lightprobes.new(name, 'VOLUME')
        pd.resolution_x, pd.resolution_y, pd.resolution_z = res
    except Exception:
        return None
    o = bpy.data.objects.new(name, pd)
    sb.link_obj(o)
    o.location = loc
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    return o


def bake_volume_probes():
    try:
        with bpy.context.temp_override(scene=bpy.context.scene):
            bpy.ops.object.lightprobe_cache_bake(subset='ALL')
    except Exception as ex:
        print("probe bake failed", ex)


def softbox(name, loc, target, size=(1, 1), strength=5.0, color=(1, 1, 1), visible_camera=False):
    """Emissive card (visible in reflections, captured by sphere probes) - studio softbox for metals."""
    o = sb.prim("plane", name, loc=loc, scale=(size[0] / 2, size[1] / 2, 1))
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = (V(target) - V(loc)).to_track_quat('Z', 'Y')
    o.data.materials.append(sb.emit_mat(name + "M", color, strength))
    o.visible_camera = visible_camera
    o.visible_shadow = False
    return o


def dof_cam(name, lens, fstop, sensor=36.0, clip=(0.01, 2000)):
    cam = sb.camera(name, loc=(0, -1, 0), target=(0, 0, 0), lens=lens, fstop=fstop, focus=1.0, sensor=sensor, clip=clip)
    cam.data.dof.aperture_blades = 9
    return cam


# =========================================================================== loops / D coils

def chaikin(pts, it=4, closed=True):
    P = np.asarray(pts, float)
    for _ in range(it):
        Q = []
        n = len(P)
        rng_ = range(n) if closed else range(n - 1)
        if not closed:
            Q.append(P[0])
        for i in rng_:
            a, b = P[i], P[(i + 1) % n]
            Q.append(0.75 * a + 0.25 * b); Q.append(0.25 * a + 0.75 * b)
        if not closed:
            Q.append(P[-1])
        P = np.array(Q)
    return P


def resample(P, n, closed=True):
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]]) if closed else P
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    t = np.linspace(0, s[-1], n, endpoint=not closed)
    out = np.stack([np.interp(t, s, Q[:, k]) for k in range(3)], 1)
    return out


def rounded_rect(w, h, r, nc=4):
    """Closed profile (list of (a, b)), a across w, b across h, corner radius r. CCW."""
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for k in range(nc + 1):
            a = math.radians(a0 + 90 * k / nc)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def sweep_planar(name, P, profile, normal=(0, 1, 0), closed=True, open_profile=False, mat=None, coll=None, uv_scale=1.0):
    """Sweep a profile along a planar path P (N,3). Profile (a, b): a along the fixed plane `normal`,
    b along the in-plane perpendicular (T x normal). UV: u = arc length along path, v = arc length around profile."""
    P = np.asarray(P, float)
    nrm = np.array(normal, float)
    n = len(P)
    if closed:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    else:
        T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Bv = np.cross(T, nrm)
    Bv /= np.linalg.norm(Bv, axis=1, keepdims=True)
    prof = np.asarray(profile, float)
    m = len(prof)
    verts = (P[:, None, :] + prof[None, :, 0:1] * nrm[None, None, :] + prof[None, :, 1:2] * Bv[:, None, :]).reshape(-1, 3)
    Pc = np.vstack([P, P[:1]]) if closed else P
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))])
    pc = np.vstack([prof, prof[:1]]) if not open_profile else prof
    vs = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))])
    faces, uvs = [], []
    ni = n if closed else n - 1
    mj = m if not open_profile else m - 1
    for i in range(ni):
        i1 = (i + 1) % n
        for j in range(mj):
            j1 = (j + 1) % m
            faces.append((i * m + j, i1 * m + j, i1 * m + j1, i * m + j1))
            uvs += [(s[i], vs[j]), (s[i + 1], vs[j]), (s[i + 1], vs[j + 1]), (s[i], vs[j + 1])]
    o = mesh_np(name, verts, np.array(faces), mat, np.array(uvs) * uv_scale, coll=coll)
    sb.recalc_normals(o)
    sb.shade_auto(o, 40)
    return o


def d_centerline(scale=1.0, n=400):
    """D-shaped TF-coil centreline in the XZ plane: straight inboard leg at x=0, height ~12.6 m, width ~7.3 m."""
    ctrl = [(0, 0, -4.6), (0, 0, 4.6), (0.35, 0, 6.3), (2.6, 0, 7.05), (5.5, 0, 6.0), (7.45, 0, 3.3), (7.95, 0, 0.0),
            (7.45, 0, -3.3), (5.5, 0, -6.0), (2.6, 0, -7.05), (0.35, 0, -6.3)]
    P = chaikin(ctrl, 5, True) * scale
    return resample(P, n, True)


def winding_mat(name, pitch=45.0):
    """Exposed winding pack: conductor turns (copper stabiliser in steel jackets) separated by glass-epoxy wrap.
    Stripes follow the path (constant profile coordinate v)."""
    m = sb.mat(name, (0.8, 0.5, 0.3), metal=1.0, rough=0.3)
    nb = sb.NB(m)
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    v = nb.math('MULTIPLY', sep.outputs[1], pitch)
    f = nb.math('FRACT', v)
    ins = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', f, 0.5)), 0.36, 0.40, 0.0, 1.0)   # 1 = insulation gap
    # every 7th turn a thicker ground-insulation layer between pancakes
    grp = nb.math('FRACT', nb.math('DIVIDE', v, 7.0))
    thick_ins = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', grp, 0.5)), 0.43, 0.45, 0.0, 1.0)
    ins = nb.math('MAXIMUM', ins, thick_ins)
    turn = nb.math('FLOOR', v)
    cmb = nb.new('ShaderNodeCombineXYZ'); nb.link(turn, cmb.inputs[0]); cmb.inputs[1].default_value = 3.7
    wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'; nb.link(cmb.outputs[0], wn.inputs['Vector'])
    co = nb.coord('Object')
    tape = nb.wave(nb.mapping(co, rot=(0.4, 0.2, 0.7)), scale=180, wtype='BANDS', direction='X')
    cu_col = nb.mix(nb.maprange(wn.outputs['Value'], 0, 1, 0, 0.35), (0.90, 0.50, 0.33, 1), (0.72, 0.36, 0.22, 1))
    glass_col = nb.mix(nb.maprange(tape.outputs['Fac'], 0.2, 0.8), (0.30, 0.22, 0.10, 1), (0.46, 0.36, 0.17, 1))
    nb.set('Base Color', nb.mix(ins, cu_col, glass_col))
    nb.set('Metallic', nb.mix(ins, 1.0, 0.0, dtype='FLOAT'))
    nb.set('Roughness', nb.mix(ins, 0.28, 0.55, dtype='FLOAT'))
    prof_h = nb.math('SINE', nb.math('MULTIPLY', f, math.pi))
    h = nb.math('SUBTRACT', prof_h, nb.math('MULTIPLY', ins, 0.6))
    nb.set('Normal', nb.bump(nb.math('ADD', h, nb.math('MULTIPLY', tape.outputs['Fac'], nb.math('MULTIPLY', ins, 0.2))), strength=0.5, distance=0.003))
    return m


def weld_mat(name):
    """Weld bead with ripples (UV u along the bead) and heat tint fading out to the sides (UV v around)."""
    m = sb.mat(name, (0.55, 0.55, 0.56), metal=1.0, rough=0.35)
    nb = sb.NB(m)
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(uv, sep.inputs[0])
    rip = nb.math('FRACT', nb.math('MULTIPLY', sep.outputs[0], 160))
    n = nb.noise(nb.coord('Object'), scale=40, detail=4)
    nb.set('Normal', nb.bump(nb.math('ADD', nb.math('POWER', rip, 0.6), nb.math('MULTIPLY', n.outputs['Fac'], 0.5)), strength=0.6, distance=0.002))
    tint = nb.maprange(n.outputs['Fac'], 0.35, 0.65)
    nb.set('Base Color', nb.ramp(tint, [(0.0, (0.35, 0.30, 0.34)), (0.4, (0.55, 0.45, 0.30)), (0.7, (0.58, 0.57, 0.58)), (1.0, (0.50, 0.50, 0.52))]))
    return m


def tf_coil(name="TF", scale=1.0, window=None, detail=True, case_mat=None, pack_mat=None, weld_m=None, lug_mat=None,
            bolt_src=None, depth=0.95, radial=1.15):
    """D-shaped superconducting TF coil: stainless case (rounded rect section) with weld seams, bolted lugs.
    window=(s0, s1) fraction range of the loop where the near (-Y) side plate is absent, exposing the winding pack.
    Returns root empty; coil lies in its local XZ plane, inboard leg at x=0."""
    root = sb.empty(name)
    P = d_centerline(scale, 520)
    n = len(P)
    W, D = radial * scale, depth * scale
    case_mat = case_mat or stainless(name + "Case", rough=0.32, brushed=False, grime=0.25, scale=0.05)
    objs = []
    prof = rounded_rect(D, W, 0.035 * scale, 3)
    if window:
        i0, i1 = int(window[0] * n), int(window[1] * n)
        idx_a = list(range(i1, n)) + list(range(0, i0 + 1))
        A = sweep_planar(name + "CaseA", P[idx_a], prof, closed=False, mat=case_mat)
        objs.append(A)
        # window segment: U-channel (open towards -Y) + winding pack inside
        t = 0.06 * scale
        u_prof = [(-D / 2 + 0.03 * scale, W / 2), (D / 2, W / 2), (D / 2, -W / 2), (-D / 2 + 0.03 * scale, -W / 2),
                  (-D / 2 + 0.03 * scale, -W / 2 + t), (D / 2 - t, -W / 2 + t), (D / 2 - t, W / 2 - t), (-D / 2 + 0.03 * scale, W / 2 - t)]
        Bseg = P[i0:i1 + 1]
        U = sweep_planar(name + "CaseU", Bseg, u_prof, closed=False, open_profile=False, mat=case_mat)
        objs.append(U)
        pk = sweep_planar(name + "Pack", Bseg, rounded_rect(D - 2 * t + 0.02 * scale, W - 2 * t, 0.01 * scale, 2), closed=False,
                          mat=pack_mat or winding_mat(name + "Wind"))
        objs.append(pk)
    else:
        objs.append(sweep_planar(name + "Case", P, prof, closed=True, mat=case_mat))
    if detail:
        weld_m = weld_m or weld_mat(name + "Weld")
        # weld seams where side plates meet the U-shell (both faces), slightly inset from the corners
        Nn = np.array((0, 1.0, 0))
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0); T /= np.linalg.norm(T, axis=1, keepdims=True)
        Bv = np.cross(T, Nn); Bv /= np.linalg.norm(Bv, axis=1, keepdims=True)
        for sa in (-1, 1):
            for sbb in (-1, 1):
                off = sa * (D / 2) * Nn[None, :] + sbb * (W / 2 - 0.09 * scale) * Bv
                Q = P + off + sa * 0.004 * scale * Nn[None, :]
                if window:
                    Q = Q[idx_a] if sa < 0 else Q
                wobj = sweep_planar(name + f"Weld{sa}{sbb}", Q, [(0.012 * scale * math.cos(a), 0.012 * scale * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)],
                                    closed=(not window or sa > 0), mat=weld_m)
                objs.append(wobj)
        # bolted lugs (intercoil structure pads) along the outer curve
        lug_mat = lug_mat or stainless(name + "Lug", rough=0.25, brushed=True, scale=0.2)
        for k, fr in enumerate(np.linspace(0.12, 0.88, 9)):
            i = int(fr * n) % n
            if window and int(window[0] * n) - 8 <= i <= int(window[1] * n) + 8:
                continue
            p = P[i]; b = Bv[i]; tng = T[i]
            if p[0] < 1.0 * scale:
                continue
            c = p + b * (W / 2 + 0.09 * scale)
            lug = sb.prim("cube", f"{name}Lug{k}", loc=tuple(c), scale=(0.5 * D * 1.15, 0.09 * scale, 0.32 * scale), mat=lug_mat)
            M = Matrix((tuple(Nn), tuple(b), tuple(tng))).transposed()
            lug.rotation_mode = 'QUATERNION'; lug.rotation_quaternion = M.to_quaternion()
            m_ = sb.bevel(lug, 0.012 * scale, 2); m_.harden_normals = True
            lug.parent = root
            if bolt_src is not None:
                for bx in (-0.25, 0.0, 0.25):
                    for bz in (-0.18, 0.18):
                        bp = c + Nn * (bx * D) + tng * (bz * scale) + b * (0.09 * scale)
                        place_linked(bolt_src, tuple(bp), direction=tuple(b), spin=k * 0.7 + bx, parent=None).parent = root
    for o in objs:
        o.parent = root
    return root, P
