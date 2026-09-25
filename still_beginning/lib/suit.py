"""STILL BEGINNING - the ONE EVA suit design (used in s18 glove close-up, s22 orbit, s23 moon, ...).

Design language
  * soft outer layer: warm-white ortho-fabric (plain weave + ripstop grid, metric UVs so the weave never swims),
    welted seams with stitching, stitched blank patches/pockets (no text, logos or flags), subtle dirt / scuffs,
    optional regolith dust that climbs from the boots.
  * graphite hard elements: shoulder / upper-arm / waist / hip / ankle bearings, chest module, PLSS side panels,
    wrist-disconnect housings (edge wear shows bare aluminium).
  * LOCK COLLARS ARE AMBER: every rotating lock ring (wrist disconnects, neck ring) carries an amber anodized
    flange (props.amber_anodized palette) + a scratched bare-aluminium collar = the film's amber-highlight motif.
  * gloves: SDF-generated (lib/sdfglove.py): white TMG back, grey rubberised palm, dark silicone grip pads and
    fingertip caps, knuckle bar, seams.  helmet: clear bubble + visor assembly with gold / amber visor.

Coordinates: metres, Z up, the suit faces -Y, its LEFT side is +X ("L" = +X).  Feet on z = 0 in the bind pose.

API (all builders return dicts)
  M = suit.materials(dust=0.0, dirt=0.35)           shared material dict (keys below); tweak M[...] freely
  S = suit.build(name="Astro", visor="gold"|"amber"|"clear", gloves=("relaxed","relaxed"), dust=0.0,
                 glove_voxel=1.2, mats=None, detail=1.0, occupant=True|dict(phenotype=..., skin=...))
        visor="clear" raises the gold visor and shows a REAL MPFB head (docs/PEOPLE.md) in a comm cap inside the
        bubble (occupant dict -> mhchild phenotype / skin name).  "gold" (default) hides the face.
        S["root"]  armature object (move/rotate/keyframe THIS to place the figure; origin between the feet)
        S["parts"] {name: object}, S["mats"], S["bones"] names
  suit.pose(S, frame=None, **joints)                 coarse posing in degrees (keyframes when frame given):
        torso=(fwd, side, twist)          pelvis=(fwd, side, twist)
        l_shoulder / r_shoulder=(fwd, out, twist)   l_elbow / r_elbow=bend    l_wrist / r_wrist=(flex, dev, twist)
        l_hip / r_hip=(fwd, out, twist)            l_knee / r_knee=bend      l_ankle / r_ankle=flex(+toes up)
     fwd = limb swings forward (-Y), out = abduction away from the body, bend = natural flexion (>0).
  suit.hand_socket(S, "L"|"R")  -> (bone name) + suit.attach(S, obj, bone) to parent props to hands etc.
  F = suit.forearm(side="L", glove="relaxed", mats=None, sleeve_len=0.32, glove_voxel=0.6, name="FA")
        close-up forearm + wrist disconnect + glove. F["root"] = empty at the disconnect ring face, local +Y
        points to the hand. F["collar"] = empty that rotates the lock collar+amber flange (rotate about local Y),
        F["pin"] = lock indicator pin object (animate location z 0 -> PIN_POP to show it lock),
        F["glove"] = glove mesh (parented to F["glove_root"]), F["sleeve"], F["parts"].
  suit.glove(pose, side="R", voxel=0.8, mats=None, name="Glove")  -> glove mesh object in glove-local frame
        (wrist at origin, fingers +Y, palm -Z; pose = name in sdfglove.POSES or a dict; cached in assets/suit/)
  suit.glove_npz(pose, voxel) -> path  (generates with system python3 if missing)

Performance: full suit ~ 180-260k tris at detail=1 (gloves at 1.2 mm voxel), all EEVEE-friendly
(no displacement modifiers, no subsurf at render).  forearm() close-up is ~0.6-1M tris.
"""
import bpy, bmesh, math, os, sys, json, hashlib, subprocess, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler, noise
import sb

LIB = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(sb.ROOT, "assets", "suit")
os.makedirs(ASSETS, exist_ok=True)

WHITE = (0.80, 0.785, 0.75)
GRAPHITE = (0.030, 0.031, 0.034)
AMBER = (0.93, 0.47, 0.09)
PIN_POP = 0.0048          # lock indicator pin travel (m)
GLOVE_OFF = 0.074         # glove-local origin sits this far (+Y) from the disconnect ring face

# =========================================================================== small utils

def _attr_float(me, name, arr, domain='POINT'):
    a = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', domain)
    a.data.foreach_set("value", np.asarray(arr, np.float32).ravel())


def _attr_vec(me, name, arr):
    a = me.attributes.get(name) or me.attributes.new(name, 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set("vector", np.asarray(arr, np.float32).ravel())


def _attr_col(me, name, arr):
    a = me.color_attributes.new(name, 'FLOAT_COLOR', 'POINT')
    a.data.foreach_set("color", np.asarray(arr, np.float32).ravel())


def mesh_np(name, Vt, F, mats=(), smooth=True, coll=None, uv=None):
    """fast mesh from numpy arrays; F: (n,3) or (n,4) int; uv: per-loop (nloops,2)."""
    Vt = np.asarray(Vt, np.float32)
    F = np.asarray(F, np.int32)
    k = F.shape[1]
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(Vt))
    me.vertices.foreach_set("co", Vt.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, k, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(F), k, np.int32))
    me.update(calc_edges=True)
    if uv is not None:
        l = me.uv_layers.new(name="UVMap")
        l.uv.foreach_set("vector", np.asarray(uv, np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.full(len(F), smooth, bool))
    for m in mats:
        me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o, coll)
    return o


def _frames(path, up0):
    """parallel-transport frames along a polyline. returns T, N(side), B(up) arrays."""
    P = np.asarray(path, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.zeros_like(P); Nn = np.zeros_like(P)
    u = np.asarray(up0, float)
    u = u - T[0] * (u @ T[0]); u /= np.linalg.norm(u)
    for i in range(n):
        if i > 0:
            u = u - T[i] * (u @ T[i]); u /= np.linalg.norm(u)
        B[i] = u
        Nn[i] = np.cross(T[i], u)
    return T, Nn, B


def tube(name, path, prof, n=64, up=(0, 0, 1), disp=None, mats=(), cap0=False, cap1=False,
         seam_theta=0.0, seams_v=(), coll=None, hfun=None):
    """Lofted soft tube with metric UVs (u = around, v = along; metres) and seam attributes.
    path: (k,3) centreline; prof(i, s) -> (rx, ry, expo)  (rx along side, ry along up; expo 2 = ellipse)
    disp(i, s, th, p) -> radial offset (m).  seams: lengthwise at theta=seam_theta and circumferential at seams_v (s).
    Attributes: seam (m to nearest seam), seam_t (coordinate along that seam), h (0..1 height for dust)."""
    P = np.asarray(path, float)
    T, Nn, B = _frames(P, up)
    k = len(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    th = np.linspace(0, 2 * math.pi, n, endpoint=False) + seam_theta
    Vt = np.zeros((k, n, 3))
    rad = np.zeros((k, n))
    for i in range(k):
        rx, ry, ex = prof(i, s[i])
        c, sn = np.cos(th - seam_theta + seam_theta), np.sin(th)
        c = np.cos(th)
        # superellipse radius along each angle
        e = 2.0 / ex
        x = np.sign(c) * np.abs(c) ** e * rx
        y = np.sign(sn) * np.abs(sn) ** e * ry
        if disp is not None:
            r0 = np.sqrt(x * x + y * y) + 1e-9
            dd = np.array([disp(i, s[i], th[j], P[i]) for j in range(n)])
            x = x * (1 + dd / r0); y = y * (1 + dd / r0)
        Vt[i] = P[i] + np.outer(x, Nn[i]) + np.outer(y, B[i])
        rad[i] = np.sqrt(x * x + y * y)
    # metric UVs: u from ring perimeter, v from mean meridian length
    du = np.linalg.norm(np.roll(Vt, -1, axis=1) - Vt, axis=2)       # (k, n)
    ucum = np.concatenate([np.zeros((k, 1)), np.cumsum(du, axis=1)], 1)   # (k, n+1)
    dv = np.linalg.norm(np.diff(Vt, axis=0), axis=2).mean(1)
    vcum = np.concatenate([[0], np.cumsum(dv)])
    idx = np.arange(k * n).reshape(k, n)
    F, UV = [], []
    for i in range(k - 1):
        for j in range(n):
            j1 = (j + 1) % n
            F.append((idx[i, j], idx[i, j1], idx[i + 1, j1], idx[i + 1, j]))
            UV += [(ucum[i, j], vcum[i]), (ucum[i, j + 1], vcum[i]), (ucum[i + 1, j + 1], vcum[i + 1]), (ucum[i + 1, j], vcum[i + 1])]
    verts = Vt.reshape(-1, 3)
    extra_v = []
    if cap0 or cap1:
        for which, i in ((cap0, 0), (cap1, k - 1)):
            if not which:
                continue
            ci = len(verts) + len(extra_v)
            extra_v.append(Vt[i].mean(0) + (T[i] * (-0.35 if i == 0 else 0.35) * rad[i].mean()))
            for j in range(n):
                j1 = (j + 1) % n
                if i == 0:
                    F.append((idx[i, j1], idx[i, j], ci, ci))
                else:
                    F.append((idx[i, j], idx[i, j1], ci, ci))
                UV += [(ucum[i, j1] if i == 0 else ucum[i, j], vcum[i]), (ucum[i, j] if i == 0 else ucum[i, j + 1], vcum[i]),
                       (ucum[i, j], vcum[i] + 0.05), (ucum[i, j], vcum[i] + 0.05)]
    if extra_v:
        verts = np.concatenate([verts, np.asarray(extra_v)], 0)
    # triangulate degenerate cap quads
    F2, UV2 = [], []
    for f, q in zip(F, np.asarray(UV).reshape(-1, 4, 2)):
        if f[2] == f[3]:
            F2.append((f[0], f[1], f[2], f[2]))
        else:
            F2.append(f)
    o = _mesh_mixed(name, verts, F, UV, mats, coll)
    # seam attributes
    C = ucum[:, -1:]
    uu = ucum[:, :-1]
    d_len = np.minimum(uu, C - uu)                       # to lengthwise seam
    t_len = np.repeat(vcum[:, None], n, 1)
    seam = d_len.copy(); seam_t = t_len.copy()
    for sv in seams_v:
        dvv = np.abs(np.repeat(s[:, None], n, 1) - sv)
        better = dvv < seam
        seam = np.where(better, dvv, seam)
        seam_t = np.where(better, uu, seam_t)
    ext = len(verts) - k * n
    seam = np.concatenate([seam.ravel(), np.full(ext, 1.0)])
    seam_t = np.concatenate([seam_t.ravel(), np.zeros(ext)])
    _attr_float(o.data, "seam", seam)
    _attr_float(o.data, "seam_t", seam_t)
    if hfun is not None:
        _attr_float(o.data, "h", hfun(verts))
    o["s_len"] = float(s[-1])
    return o, dict(P=P, T=T, N=Nn, B=B, s=s, rad=rad, verts=verts)


def _mesh_mixed(name, verts, F, UV, mats, coll=None):
    """quads with possible repeated last index -> tris."""
    faces = []
    uvs = []
    UV = np.asarray(UV, float).reshape(-1, 4, 2)
    for f, q in zip(F, UV):
        if f[2] == f[3]:
            faces.append(f[:3]); uvs.append(q[:3])
        else:
            faces.append(f); uvs.append(q)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(int(x) for x in f) for f in faces])
    me.update()
    l = me.uv_layers.new(name="UVMap")
    flat = np.concatenate([np.asarray(u) for u in uvs], 0)
    l.uv.foreach_set("vector", flat.astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    for m in mats:
        me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o, coll)
    return o


def lathe(name, prof, segs=96, mats=(), rfun=None, axis='Y', coll=None, mat_idx=None):
    """Revolved hard part with metric UVs and an 'edge' attribute.
    prof: list of (r, a, edge) with a = axial coordinate (local +Y by default), edge 0..1 = edge-wear weight.
    rfun(r, a, th) -> r'  for angular features (scallops, lugs). mat_idx(k) -> material index per profile span."""
    prof = _auto_edge(prof)
    m = len(prof)
    th = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    Vt = np.zeros((segs, m, 3)); E = np.zeros((segs, m))
    for j, t in enumerate(th):
        for k, pr in enumerate(prof):
            r, a = pr[0], pr[1]
            e = pr[2] if len(pr) > 2 else 0.0
            if rfun:
                r = rfun(r, a, t)
            if axis == 'Y':
                Vt[j, k] = (r * math.cos(t), a, r * math.sin(t))
            else:
                Vt[j, k] = (r * math.cos(t), r * math.sin(t), a)
            E[j, k] = e
    # metric uv
    ls = [0.0]
    for k in range(1, m):
        ls.append(ls[-1] + math.hypot(prof[k][0] - prof[k - 1][0], prof[k][1] - prof[k - 1][1]))
    F, UV, MI = [], [], []
    for j in range(segs):
        j1 = (j + 1) % segs
        for k in range(m - 1):
            a0, a1 = j * m + k, j1 * m + k
            if axis == 'Y':
                F.append((a0, a0 + 1, a1 + 1, a1))
            else:
                F.append((a0, a1, a1 + 1, a0 + 1))
            rr = 0.5 * (prof[k][0] + prof[k + 1][0])
            u0, u1 = th[j] * rr, (th[j] + 2 * math.pi / segs) * rr
            if axis == 'Y':
                UV.append([(u0, ls[k]), (u0, ls[k + 1]), (u1, ls[k + 1]), (u1, ls[k])])
            else:
                UV.append([(u0, ls[k]), (u1, ls[k]), (u1, ls[k + 1]), (u0, ls[k + 1])])
            MI.append(mat_idx(k) if mat_idx else 0)
    o = mesh_np(name, Vt.reshape(-1, 3), F, mats, uv=np.asarray(UV).reshape(-1, 2), coll=coll)
    if mat_idx:
        o.data.polygons.foreach_set("material_index", np.asarray(MI, np.int32))
    _attr_float(o.data, "edge", E.ravel())
    sb.recalc_normals(o)
    return o


def _auto_edge(prof, fall=0.0012):
    """edge-wear weight from profile turning angle (convex corners only), densified so wear stays near edges."""
    P = [(float(p[0]), float(p[1])) for p in prof]
    n = len(P)
    e = [0.0] * n
    for k in range(1, n - 1):
        a = np.subtract(P[k], P[k - 1]); b = np.subtract(P[k + 1], P[k])
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la < 1e-9 or lb < 1e-9:
            continue
        cr = (a[0] * b[1] - a[1] * b[0]) / (la * lb)
        ang = math.degrees(math.acos(max(-1, min(1, (a @ b) / (la * lb)))))
        e[k] = min(1.0, ang / 50.0)
    out = []
    for k in range(n):
        out.append((P[k][0], P[k][1], e[k]))
        if k < n - 1:
            L = math.hypot(P[k + 1][0] - P[k][0], P[k + 1][1] - P[k][1])
            if L > 3 * fall:
                for t, w in ((fall / L, 0.25), (0.5, 0.0), (1 - fall / L, 0.25)):
                    ee = w * (e[k] if t < 0.5 else e[k + 1])
                    out.append((P[k][0] + t * (P[k + 1][0] - P[k][0]), P[k][1] + t * (P[k + 1][1] - P[k][1]), ee))
    return out


def rbox(name, size, r=0.01, mats=(), seg=3, loc=(0, 0, 0), edge=True, coll=None):
    """rounded box (bevelled cube, applied) with an 'edge' attribute from bevel geometry."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = V((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    bev = bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts)[:0], offset=r, segments=seg, profile=0.5, affect='EDGES')
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o, coll)
    for m in mats:
        me.materials.append(m)
    # edge attribute: distance from the flat faces
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    h = np.asarray(size) / 2
    dd = np.clip((np.abs(co) - (h - r)) / r, 0, 1)
    e = np.clip((dd > 0.01).sum(1) >= 2, 0, 1) * np.clip(dd.sum(1) / 2, 0, 1)
    _attr_float(me, "edge", e)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    try:
        me.set_sharp_from_angle(angle=math.radians(40))
    except Exception:
        pass
    o.location = loc
    l = me.uv_layers.new(name="UVMap")
    # box-projected metric UVs
    uvs = []
    for p in me.polygons:
        n = p.normal
        ax = np.argmax(np.abs(n))
        for li in p.loop_indices:
            c = me.vertices[me.loops[li].vertex_index].co
            uvs.append((c.y, c.z) if ax == 0 else ((c.x, c.z) if ax == 1 else (c.x, c.y)))
    l.uv.foreach_set("vector", np.asarray(uvs, np.float32).ravel())
    return o

# =========================================================================== materials

def _weave(nb, u, v, tpm, rip=9):
    """plain weave + ripstop grid in (u, v) metres. returns (height, thread_tone) sockets."""
    su = nb.math('MULTIPLY', u, tpm)
    sv = nb.math('MULTIPLY', v, tpm)
    fu = nb.math('FRACT', su); fv = nb.math('FRACT', sv)
    iu = nb.math('FLOOR', su); iv = nb.math('FLOOR', sv)
    chk = nb.math('FLOORED_MODULO', nb.math('ADD', iu, iv), 2.0)
    pu = nb.math('SINE', nb.math('MULTIPLY', fu, math.pi))
    pv = nb.math('SINE', nb.math('MULTIPLY', fv, math.pi))
    hw = nb.math('MULTIPLY', pu, nb.math('ADD', 0.45, nb.math('MULTIPLY', pv, 0.55)))
    hf = nb.math('MULTIPLY', pv, nb.math('ADD', 0.45, nb.math('MULTIPLY', pu, 0.55)))
    h = nb.mix(chk, hf, hw, dtype='FLOAT')
    # ripstop: every `rip` threads a doubled thread stands proud
    ru = nb.math('LESS_THAN', nb.math('FLOORED_MODULO', iu, float(rip)), 1.0)
    rv = nb.math('LESS_THAN', nb.math('FLOORED_MODULO', iv, float(rip)), 1.0)
    rp = nb.math('MAXIMUM', nb.math('MULTIPLY', ru, pu), nb.math('MULTIPLY', rv, pv))
    h = nb.math('ADD', h, nb.math('MULTIPLY', rp, 0.55))
    # per-thread tone
    wn_u = nb.new('ShaderNodeTexWhiteNoise', _noise_dimensions='1D'); nb.link(iu, wn_u.inputs['W'])
    wn_v = nb.new('ShaderNodeTexWhiteNoise', _noise_dimensions='1D'); nb.link(iv, wn_v.inputs['W'])
    tone = nb.mix(chk, wn_v.outputs['Value'], wn_u.outputs['Value'], dtype='FLOAT')
    tone = nb.math('ADD', tone, nb.math('MULTIPLY', rp, 0.8))
    return h, tone


def _uv_split(nb):
    sep = nb.new('ShaderNodeSeparateXYZ')
    nb.link(nb.coord('UV'), sep.inputs[0])
    return sep.outputs[0], sep.outputs[1]


def _attr(nb, name, out='Fac'):
    a = nb.new('ShaderNodeAttribute')
    a.attribute_name = name
    return a.outputs[out]


def fabric_mat(name="SuitFabric", color=WHITE, dirt=0.35, dust=0.0, tpm=2600.0, mode="UV", seed=0.0):
    """Ortho-fabric outer layer. mode 'UV' (lofted parts with metric UVs + seam attrs) or 'TRI' (triplanar on the
    'rest' vector attribute, used by gloves)."""
    m = sb.mat(name, color, rough=0.9, sheen=0.3, spec=0.3)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs['Sheen Roughness'].default_value = 0.45
    b.inputs['Sheen Tint'].default_value = (1, 1, 1, 1)
    nb = sb.NB(m)
    if mode == "UV":
        u, v = _uv_split(nb)
        h, tone = _weave(nb, u, v, tpm)
        pos = nb.new('ShaderNodeCombineXYZ'); nb.link(u, pos.inputs[0]); nb.link(v, pos.inputs[1])
        pos.inputs[2].default_value = seed
        pos = pos.outputs[0]
        seam = _attr(nb, "seam"); seam_t = _attr(nb, "seam_t")
        hgt = _attr(nb, "h")
    else:
        rest = _attr(nb, "rest", 'Vector')
        sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(rest, sep.inputs[0])
        x, y, z = sep.outputs[0], sep.outputs[1], sep.outputs[2]
        gn = nb.new('ShaderNodeNewGeometry')
        nsep = nb.new('ShaderNodeSeparateXYZ')
        vt = nb.new('ShaderNodeVectorTransform', _vector_type='NORMAL', _convert_from='WORLD', _convert_to='OBJECT')
        nb.link(gn.outputs['Normal'], vt.inputs[0]); nb.link(vt.outputs[0], nsep.inputs[0])
        ws = [nb.math('POWER', nb.math('ABSOLUTE', nsep.outputs[k]), 4.0) for k in range(3)]
        tot = nb.math('ADD', nb.math('ADD', ws[0], ws[1]), nb.math('ADD', ws[2], 1e-4))
        h = None; tone = None
        for (a_, b_), w in (((y, z), ws[0]), ((x, z), ws[1]), ((x, y), ws[2])):
            hh, tt = _weave(nb, a_, b_, tpm)
            wn = nb.math('DIVIDE', w, tot)
            h = nb.math('MULTIPLY', hh, wn) if h is None else nb.math('ADD', h, nb.math('MULTIPLY', hh, wn))
            tone = nb.math('MULTIPLY', tt, wn) if tone is None else nb.math('ADD', tone, nb.math('MULTIPLY', tt, wn))
        pos = rest
        seam = None; seam_t = None; hgt = None
    # --- colour: thread tone, broad dirt, scuffs, dust
    col = nb.mix(nb.maprange(tone, 0.0, 1.8, 0.0, 1.0), (color[0] * 0.95, color[1] * 0.95, color[2] * 0.94, 1),
                 (color[0] * 1.03, color[1] * 1.03, color[2] * 1.03, 1))
    g1 = nb.noise(pos, scale=7.0, detail=6, rough=0.6)
    g2 = nb.noise(pos, scale=38.0, detail=4, rough=0.65)
    gm = nb.math('MULTIPLY', nb.maprange(g1.outputs['Fac'], 0.45, 0.75, 0.0, 1.0),
                 nb.maprange(g2.outputs['Fac'], 0.3, 0.7, 0.4, 1.0))
    col = nb.mix(nb.math('MULTIPLY', gm, dirt), col, (0.56, 0.54, 0.50, 1))
    # scuffs: thin streaks
    sc_ = nb.noise(nb.mapping(pos, scale=(1.0, 9.0, 1.0), rot=(0.0, 0.0, 0.5)), scale=30.0, detail=3, rough=0.5)
    scm = nb.maprange(sc_.outputs['Fac'], 0.66, 0.72, 0.0, 1.0)
    scm = nb.math('MULTIPLY', scm, nb.maprange(nb.noise(pos, scale=5.0, detail=2).outputs['Fac'], 0.5, 0.65))
    col = nb.mix(nb.math('MULTIPLY', scm, 0.55 * min(1.0, dirt * 2.5)), col, (0.50, 0.49, 0.47, 1))
    # crevice darkening (EEVEE horizon-scan AO)
    ao = nb.new('ShaderNodeAmbientOcclusion', _samples=8)
    ao.inputs['Distance'].default_value = 0.03
    col = nb.mix(nb.maprange(ao.outputs['AO'], 0.2, 1.0, 0.35 * dirt + 0.08, 0.0), col, (0.36, 0.34, 0.31, 1))
    # regolith dust (height gradient + patches)
    dust_amt = nb.new('ShaderNodeValue', name="dust"); dust_amt.name = "dust"; dust_amt.outputs[0].default_value = dust
    if hgt is not None:
        dh = nb.maprange(hgt, 0.0, 0.45, 1.0, 0.0)
    else:
        dh = 0.3
    dn = nb.noise(pos, scale=14.0, detail=6, rough=0.7)
    dmask = nb.math('MULTIPLY', nb.math('ADD', dh if not isinstance(dh, float) else 0.3,
                                        nb.maprange(dn.outputs['Fac'], 0.4, 0.8, 0.0, 0.8)), dust_amt.outputs[0])
    dmask = nb.math('MINIMUM', dmask, 0.92)
    col = nb.mix(dmask, col, (0.33, 0.32, 0.30, 1))
    # --- seams: welted ridge + stitches
    hh = nb.math('MULTIPLY', h, 0.6)
    rough = nb.maprange(tone, 0.0, 1.8, 0.86, 0.76)
    if seam is not None:
        ridge = nb.math('MULTIPLY', nb.maprange(seam, 0.0045, 0.0016, 0.0, 1.0), nb.maprange(seam, 0.0, 0.0007, 0.35, 1.0))
        stitch_band = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', seam, 0.0032)), 0.00055, 0.0002, 0.0, 1.0)
        dash = nb.maprange(nb.math('SINE', nb.math('MULTIPLY', seam_t, 2 * math.pi / 0.0032)), -0.2, 0.3, 0.0, 1.0)
        st = nb.math('MULTIPLY', stitch_band, dash)
        hh = nb.math('ADD', hh, nb.math('ADD', nb.math('MULTIPLY', ridge, 2.2), nb.math('MULTIPLY', st, 1.6)))
        col = nb.mix(nb.math('MULTIPLY', ridge, 0.10), col, (0.62, 0.60, 0.56, 1))
        col = nb.mix(nb.math('MULTIPLY', st, 0.5), col, (0.92, 0.91, 0.88, 1))
        rough = nb.mix(st, rough, 0.6, dtype='FLOAT')
    rough = nb.mix(dmask, rough, 0.95, dtype='FLOAT')
    nb.set('Base Color', col)
    nb.set('Roughness', rough)
    # macro folds on top of geometry: soft low-freq bump
    fold = nb.noise(nb.mapping(pos, scale=(1.0, 3.0, 1.0)), scale=22.0, detail=3, rough=0.5)
    hh = nb.math('ADD', hh, nb.math('MULTIPLY', fold.outputs['Fac'], 8.0))
    nb.set('Normal', nb.bump(hh, strength=0.32, distance=0.00012))
    return m


def graphite_mat(name="SuitGraphite", color=GRAPHITE, wear=0.6, dust=0.0):
    """hard graphite composite / anodized parts: satin, micro scratches, edge wear to bare aluminium."""
    m = sb.mat(name, color, metal=0.0, rough=0.42, spec=0.45, coat=0.2, coat_rough=0.22)
    nb = sb.NB(m)
    co = nb.coord('Object')
    e = _attr(nb, "edge")
    n1 = nb.noise(co, scale=60.0, detail=5, rough=0.6)
    em = nb.math('MULTIPLY', nb.math('POWER', e, 4.0), nb.maprange(n1.outputs['Fac'], 0.55, 0.68, 0.0, 1.0))
    em = nb.math('MULTIPLY', em, wear)
    # scratches: several families of thin lines
    s = None
    for i, (rot, sc) in enumerate((((0.3, 0.1, 0.4), 70.0), ((1.1, 0.7, 2.0), 95.0), ((0.2, 1.4, 2.8), 55.0))):
        w = nb.noise(nb.mapping(co, rot=rot, scale=(1.0, 40.0, 1.0)), scale=sc, detail=2, rough=0.4)
        l = nb.maprange(w.outputs['Fac'], 0.505, 0.515, 0.0, 1.0)
        l = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('MULTIPLY', l, 2.0), 1.0))), 1.0)
        gate = nb.maprange(nb.noise(co, scale=8.0 + i * 3, detail=2).outputs['Fac'], 0.55, 0.62)
        l = nb.math('MULTIPLY', l, gate)
        s = l if s is None else nb.math('MAXIMUM', s, l)
    sc_amt = nb.math('MULTIPLY', s, 0.3)
    col = nb.mix(nb.maprange(n1.outputs['Fac'], 0.3, 0.7), (color[0] * 0.92, color[1] * 0.92, color[2] * 0.92, 1), (color[0] * 1.08, color[1] * 1.08, color[2] * 1.1, 1))
    col = nb.mix(sc_amt, col, (0.09, 0.09, 0.095, 1))
    col = nb.mix(em, col, (0.62, 0.62, 0.64, 1))
    dust_amt = nb.new('ShaderNodeValue'); dust_amt.name = "dust"; dust_amt.outputs[0].default_value = dust
    dn = nb.noise(co, scale=18.0, detail=5, rough=0.7)
    dm = nb.math('MULTIPLY', nb.maprange(dn.outputs['Fac'], 0.35, 0.75, 0.2, 1.0), dust_amt.outputs[0])
    col = nb.mix(dm, col, (0.30, 0.29, 0.27, 1))
    nb.set('Base Color', col)
    nb.set('Metallic', nb.mix(em, 0.0, 1.0, dtype='FLOAT'))
    rgh = nb.maprange(n1.outputs['Fac'], 0.3, 0.7, 0.36, 0.48)
    rgh = nb.mix(sc_amt, rgh, 0.55, dtype='FLOAT')
    rgh = nb.mix(dm, rgh, 0.9, dtype='FLOAT')
    nb.set('Roughness', rgh)
    nb.set('Coat Weight', nb.mix(nb.math('MAXIMUM', em, dm), 0.2, 0.0, dtype='FLOAT'))
    fine = nb.noise(co, scale=900.0, detail=3)
    nb.set('Normal', nb.bump(nb.math('ADD', nb.math('MULTIPLY', fine.outputs['Fac'], 0.2), nb.math('MULTIPLY', s, -0.6)), strength=0.12, distance=0.0004))
    return m


def metal_mat(name="SuitCollarMetal", color=(0.80, 0.80, 0.81), rough=0.27, turn=True, scratches=1.0, grime=0.4):
    """scratched machined aluminium: concentric lathe marks (object radius), random scratches, grime in recesses."""
    m = sb.mat(name, color, metal=1.0, rough=rough, spec=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    rr = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', sep.outputs[0], sep.outputs[0]), nb.math('MULTIPLY', sep.outputs[2], sep.outputs[2])))
    ax = sep.outputs[1]
    # lathe marks: rings in (r, axial)
    lm = nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', rr, ax), 2 * math.pi / 0.00018))
    lmn = nb.noise(co, scale=400.0, detail=2)
    lmark = nb.math('MULTIPLY', lm, nb.maprange(lmn.outputs['Fac'], 0.3, 0.7, 0.3, 1.0))
    s = None
    for i, (rot, sc) in enumerate((((0.3, 0.1, 0.4), 110.0), ((1.1, 0.7, 2.0), 140.0), ((0.2, 1.4, 2.8), 80.0), ((2.2, 0.4, 1.2), 160.0))):
        w = nb.noise(nb.mapping(co, rot=rot, scale=(1.0, 50.0, 1.0)), scale=sc, detail=2, rough=0.4)
        l = nb.maprange(w.outputs['Fac'], 0.503, 0.509, 0.0, 1.0)
        l = nb.math('SUBTRACT', 1.0, nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('MULTIPLY', l, 2.0), 1.0)))
        gate = nb.maprange(nb.noise(co, scale=10.0 + i * 4, detail=2).outputs['Fac'], 0.5, 0.58)
        l = nb.math('MULTIPLY', l, gate)
        s = l if s is None else nb.math('MAXIMUM', s, l)
    s = nb.math('MULTIPLY', s, scratches)
    smudge = nb.noise(co, scale=25.0, detail=5, rough=0.6)
    rgh = nb.maprange(smudge.outputs['Fac'], 0.35, 0.7, rough * 0.85, rough * 1.3)
    rgh = nb.math('ADD', rgh, nb.math('MULTIPLY', lmark, 0.03))
    rgh = nb.mix(s, rgh, 0.42, dtype='FLOAT')
    nb.set('Roughness', rgh)
    ao = nb.new('ShaderNodeAmbientOcclusion', _samples=8)
    ao.inputs['Distance'].default_value = 0.01
    g = nb.maprange(ao.outputs['AO'], 0.3, 1.0, grime, 0.0)
    col = nb.mix(g, (*color, 1), (0.25, 0.24, 0.22, 1))
    col = nb.mix(nb.math('MULTIPLY', s, 0.35), col, (1.0, 1.0, 1.0, 1))
    nb.set('Base Color', col)
    nb.set('Normal', nb.bump(nb.math('ADD', nb.math('MULTIPLY', lmark, 0.15), nb.math('MULTIPLY', s, -0.8)), strength=0.2, distance=0.00015))
    return m


def amber_mat(name="SuitAmber", wear=0.7):
    """amber anodized lock flange (props.amber_anodized palette) + edge wear to bright aluminium + scratches."""
    import props
    m = props.amber_anodized(name)
    nb = sb.NB(m)
    co = nb.coord('Object')
    e = _attr(nb, "edge")
    n1 = nb.noise(co, scale=120.0, detail=5, rough=0.6)
    em = nb.math('MULTIPLY', nb.math('POWER', e, 6.0), nb.maprange(n1.outputs['Fac'], 0.52, 0.64, 0.0, 1.0))
    em = nb.math('MULTIPLY', em, wear)
    s = None
    for i, (rot, sc) in enumerate((((0.3, 0.1, 0.4), 120.0), ((1.1, 0.7, 2.0), 150.0), ((0.2, 1.4, 2.8), 90.0))):
        w = nb.noise(nb.mapping(co, rot=rot, scale=(1.0, 50.0, 1.0)), scale=sc, detail=2, rough=0.4)
        l = nb.maprange(w.outputs['Fac'], 0.5055, 0.5075, 0.0, 1.0)
        l = nb.math('SUBTRACT', 1.0, nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('MULTIPLY', l, 2.0), 1.0)))
        gate = nb.maprange(nb.noise(co, scale=9.0 + i * 5, detail=2).outputs['Fac'], 0.62, 0.66)
        s = nb.math('MULTIPLY', l, gate) if s is None else nb.math('MAXIMUM', s, nb.math('MULTIPLY', l, gate))
    wearm = nb.math('MAXIMUM', em, nb.math('MULTIPLY', s, 0.12))
    tone = nb.noise(co, scale=40.0, detail=3)
    base = nb.mix(nb.maprange(tone.outputs['Fac'], 0.3, 0.7), (AMBER[0] * 0.97, AMBER[1] * 0.95, AMBER[2] * 0.92, 1), (AMBER[0] * 1.02, AMBER[1] * 1.03, AMBER[2] * 1.04, 1))
    nb.set('Base Color', nb.mix(wearm, base, (0.88, 0.87, 0.85, 1)))
    b = nb.bsdf
    # keep the existing roughness chain but lift scratches
    nb.set('Roughness', nb.mix(wearm, nb.maprange(tone.outputs['Fac'], 0.3, 0.7, 0.2, 0.3), 0.32, dtype='FLOAT'))
    nb.set('Normal', nb.bump(nb.math('MULTIPLY', s, -0.4), strength=0.1, distance=0.0001))
    return m


def rubber_mat(name="SuitRubber", color=(0.045, 0.045, 0.048), rough=0.6, dots=True, dust=0.0, coord="Object"):
    m = sb.mat(name, color, rough=rough, spec=0.4, sheen=0.15)
    nb = sb.NB(m)
    co = nb.coord('Object') if coord == "Object" else _attr(nb, "rest", 'Vector')
    h = nb.noise(co, scale=500.0, detail=3).outputs['Fac']
    if dots:
        vd = nb.voronoi(co, scale=700.0, feature='F1')
        dd = nb.maprange(vd.outputs['Distance'], 0.22, 0.32, 1.0, 0.0)
        h = nb.math('ADD', nb.math('MULTIPLY', h, 0.3), dd)
    dn = nb.noise(co, scale=30.0, detail=5, rough=0.7)
    dust_amt = nb.new('ShaderNodeValue'); dust_amt.name = "dust"; dust_amt.outputs[0].default_value = dust
    dm = nb.math('MULTIPLY', nb.maprange(dn.outputs['Fac'], 0.3, 0.7, 0.2, 1.0), dust_amt.outputs[0])
    nb.set('Base Color', nb.mix(dm, (*color, 1), (0.28, 0.27, 0.25, 1)))
    nb.set('Roughness', nb.mix(dm, rough, 0.95, dtype='FLOAT'))
    nb.set('Normal', nb.bump(h, strength=0.25, distance=0.0002))
    return m


def glove_mat(name="SuitGlove", dirt=0.3, dust=0.0, tpm=2600.0):
    """one material for the SDF glove: white TMG back, grey rubberised palm, dark silicone pads + fingertip caps."""
    m = fabric_mat(name, WHITE, dirt=dirt, dust=dust, tpm=tpm, mode="TRI")
    nb = sb.NB(m)
    b = nb.bsdf
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "glove"
    sep = nb.new('ShaderNodeSeparateColor'); nb.link(at.outputs['Color'], sep.inputs[0])
    pad, dors, crease = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    region = at.outputs['Alpha']
    cap = _attr(nb, "cap")
    rest = _attr(nb, "rest", 'Vector')
    rs = nb.new('ShaderNodeSeparateXYZ'); nb.link(rest, rs.inputs[0])
    # current fabric colour/rough/normal feeding the BSDF
    fab_col = b.inputs['Base Color'].links[0].from_socket
    fab_rgh = b.inputs['Roughness'].links[0].from_socket
    fab_nrm = b.inputs['Normal'].links[0].from_socket
    # palm-side mask (smooth) & seam at the dorsal/palmar boundary
    palm = nb.maprange(dors, -0.05, -0.30, 0.0, 1.0)
    palm = nb.math('MULTIPLY', palm, nb.maprange(region, 0.3, 0.0, 0.0, 1.0))
    seam = nb.maprange(nb.math('ABSOLUTE', nb.math('ADD', dors, 0.12)), 0.07, 0.02, 0.0, 1.0)
    seam = nb.math('MULTIPLY', seam, nb.maprange(region, 0.3, 0.0, 0.0, 1.0))
    dash = nb.maprange(nb.math('SINE', nb.math('MULTIPLY', rs.outputs[1], 2 * math.pi / 0.0032)), -0.2, 0.3)
    st = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('ADD', dors, 0.12)), 0.035, 0.015), dash)
    st = nb.math('MULTIPLY', st, nb.maprange(region, 0.3, 0.0, 0.0, 1.0))
    # rubberised palm fabric: grey, finer texture
    fine = nb.noise(rest, scale=900.0, detail=3)
    palm_col = nb.mix(nb.maprange(fine.outputs['Fac'], 0.3, 0.7), (0.30, 0.30, 0.30, 1), (0.36, 0.355, 0.35, 1))
    pad_m = nb.math('MAXIMUM', pad, cap)
    vd = nb.voronoi(rest, scale=900.0, feature='F1')
    dots = nb.maprange(vd.outputs['Distance'], 0.2, 0.3, 1.0, 0.0)
    pad_col = nb.mix(nb.maprange(fine.outputs['Fac'], 0.3, 0.7), (0.075, 0.075, 0.078, 1), (0.095, 0.095, 0.098, 1))
    # grip wear: pads polished lighter at the centre
    pwear = nb.maprange(nb.noise(rest, scale=160.0, detail=4).outputs['Fac'], 0.5, 0.75, 0.0, 0.5)
    pad_col = nb.mix(nb.math('MULTIPLY', pwear, pad), pad_col, (0.09, 0.09, 0.092, 1))
    col = nb.mix(palm, fab_col, palm_col)
    col = nb.mix(pad_m, col, pad_col)
    col = nb.mix(nb.math('MULTIPLY', st, 0.6), col, (0.88, 0.87, 0.84, 1))
    col = nb.mix(nb.math('MULTIPLY', seam, 0.25), col, (0.45, 0.44, 0.42, 1))
    # dust also settles on the glove
    nb.set('Base Color', col)
    rgh = nb.mix(palm, fab_rgh, 0.7, dtype='FLOAT')
    rgh = nb.mix(pad_m, rgh, nb.maprange(pwear, 0.0, 0.5, 0.58, 0.42), dtype='FLOAT')
    nb.set('Roughness', rgh)
    nb.set('Sheen Weight', nb.mix(pad_m, 0.35, 0.05, dtype='FLOAT'))
    # extra relief: knuckle creases, seam groove, pad dots
    wr = nb.math('SINE', nb.math('MULTIPLY', rs.outputs[1], 2 * math.pi / 0.0045))
    wn = nb.noise(rest, scale=300.0, detail=2)
    wrinkle = nb.math('MULTIPLY', nb.math('MULTIPLY', wr, crease), nb.maprange(wn.outputs['Fac'], 0.3, 0.7, 0.2, 1.0))
    wrinkle = nb.math('MULTIPLY', wrinkle, nb.math('SUBTRACT', 1.0, pad_m))
    hx = nb.math('ADD', nb.math('MULTIPLY', wrinkle, 0.8), nb.math('MULTIPLY', nb.math('MULTIPLY', dots, pad_m), 0.5))
    hx = nb.math('SUBTRACT', hx, nb.math('MULTIPLY', seam, 0.6))
    hx = nb.math('ADD', hx, nb.math('MULTIPLY', st, 0.4))
    bump2 = nb.bump(hx, strength=0.5, distance=0.0003, normal=fab_nrm)
    nb.set('Normal', bump2)
    return m


def visor_mat(name="SuitVisorGold", kind="gold"):
    if kind == "gold":
        m = sb.mat(name, (1.0, 0.76, 0.36), metal=1.0, rough=0.04, spec=0.5, thin_film=0.0)
    else:   # amber tint: semi-transparent amber glass with strong reflection
        m = sb.mat(name, (0.95, 0.55, 0.18), metal=0.0, rough=0.03, transmission=1.0, ior=1.6)
        m.use_raytrace_refraction = False
        try:
            m.thickness_mode = 'SLAB'
        except Exception:
            pass
    nb = sb.NB(m)
    co = nb.coord('Object')
    sm = nb.noise(co, scale=30.0, detail=4, rough=0.6)
    nb.set('Roughness', nb.maprange(sm.outputs['Fac'], 0.45, 0.75, 0.035, 0.12))
    return m


def bubble_mat(name="SuitBubble"):
    """clear polycarbonate shell: transparent + fresnel gloss (EEVEE friendly, no refraction smear)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Roughness'].default_value = 0.02
    fr = nt.nodes.new('ShaderNodeFresnel'); fr.inputs[0].default_value = 1.55
    mx = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(fr.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0])
    m.surface_render_method = 'BLENDED'
    try:
        m.use_transparency_overlap = True
    except Exception:
        pass
    return m


def lens_mat(name="SuitLampLens"):
    m = sb.mat(name, (0.9, 0.92, 0.95), rough=0.05, spec=1.0, coat=1.0, emission=(1.0, 0.95, 0.88), emit=0.0)
    return m


def materials(dust=0.0, dirt=0.35):
    M = dict(
        fabric=fabric_mat("SuitFabric", WHITE, dirt=dirt, dust=dust),
        fabric_soft=fabric_mat("SuitFabricSoft", (0.76, 0.75, 0.72), dirt=dirt, dust=dust, tpm=3400.0, seed=3.0),
        graphite=graphite_mat("SuitGraphite", dust=dust),
        metal=metal_mat("SuitMetal"),
        polished=metal_mat("SuitPolished", color=(0.92, 0.92, 0.93), rough=0.1, scratches=0.5, grime=0.2),
        amber=amber_mat("SuitAmber"),
        rubber=rubber_mat("SuitRubber", dust=dust),
        sole=rubber_mat("SuitSole", color=(0.035, 0.035, 0.037), rough=0.75, dots=False, dust=dust * 1.5),
        glove=glove_mat("SuitGlove", dirt=dirt * 0.8, dust=dust * 0.6),
        gold=visor_mat("SuitVisorGold", "gold"),
        amber_visor=visor_mat("SuitVisorAmber", "amber"),
        bubble=bubble_mat(),
        lens=lens_mat(),
        paint=sb.mat("SuitPaintMark", (0.85, 0.84, 0.80), rough=0.5),
        dark=sb.mat("SuitDark", (0.012, 0.012, 0.013), rough=0.6),
    )
    for m in M.values():
        m.use_fake_user = True      # MPFB (occupant) purges orphan data
    return M


def set_dust(M, amount):
    for m in M.values():
        if m and m.node_tree:
            n = m.node_tree.nodes.get("dust")
            if n:
                n.outputs[0].default_value = amount

# =========================================================================== glove

def glove_npz(pose, voxel=0.8):
    """cache + generate a glove mesh with system python3. pose: name or dict."""
    key = pose if isinstance(pose, str) else hashlib.md5(json.dumps(pose, sort_keys=True).encode()).hexdigest()[:10]
    p = os.path.join(ASSETS, "glove_%s_%.2f.npz" % (key, voxel))
    if not os.path.exists(p):
        args = ["python3", os.path.join(LIB, "sdfglove.py")]
        args += [pose] if isinstance(pose, str) else ["--json", json.dumps(pose)]
        args += [p, str(voxel)]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
    return p


def glove(pose="relaxed", side="R", voxel=0.8, mats=None, name="Glove", npz=None, decimate=None):
    """Load a glove as a mesh object in glove-local space (wrist at origin, fingers +Y, palm -Z).
    side 'L' mirrors in X. Material = mats['glove']."""
    M = mats or materials()
    p = npz or glove_npz(pose, voxel)
    D = np.load(p)
    Vt, F, A, R = D["V"].copy(), D["F"].copy(), D["A"], D["R"].copy()
    cap = D["CAP"] if "CAP" in D.files else np.zeros(len(Vt), np.float32)
    if side == "L":
        Vt[:, 0] *= -1
        R[:, 0] *= -1
        F = F[:, ::-1].copy()
    o = mesh_np(name, Vt, F, [M["glove"]])
    _attr_col(o.data, "glove", A)
    _attr_vec(o.data, "rest", R)
    _attr_float(o.data, "cap", cap)
    sb.recalc_normals(o)
    if decimate:
        dm = o.modifiers.new("Dec", 'DECIMATE'); dm.ratio = decimate
    return o

# =========================================================================== wrist disconnect + forearm

def _scallop(r, a, t, n=12, a0=-0.0215, a1=-0.0095, depth=0.0022):
    """grip scallops on the collar band."""
    if a0 <= a <= a1 and r > 0.061:
        ph = (t * n / (2 * math.pi)) % 1.0
        w = max(0.0, 1 - ((ph - 0.5) / 0.28) ** 2)
        fa = min(1.0, (a - a0) / 0.0025, (a1 - a) / 0.0025)
        return r - depth * w * max(0.0, fa)
    return r


def _lugs(r, a, t):
    """three machined lug notches on the amber flange rim (a visible rotation cue)."""
    if r > 0.064:
        for k in range(3):
            dt = ((t - k * 2 * math.pi / 3 + math.pi) % (2 * math.pi)) - math.pi
            if abs(dt) < 0.07:
                return r - 0.0024 * (1 - (dt / 0.07) ** 4)
    return r


WD_SHIFT = 0.013       # collar band widened for a two-finger grip (clamp/housing moved back by this)


def wrist_disconnect(M, name="WD", segs=192, coll=None):
    """Sleeve-side disconnect assembly in local space: axis +Y toward the hand, ring face at y=0.
    Stack (sleeve -> hand): graphite clamp band (sleeve captured, 6 screws) | graphite housing with the lock-indicator
    boss at local +Z | scratched aluminium LOCK COLLAR with 16 grip scallops | AMBER anodized flange (flat annulus
    facing the hand, 3 lug notches) | polished glove-side ring.  The collar + flange rotate together about local Y
    (F['collar']); the painted collar mark (+Z at rotation 0) lines up with the housing mark when locked.
    Returns dict(root, collar(empty), pin, pin_root, parts)."""
    D = WD_SHIFT
    root = sb.empty(name)
    parts = []
    clamp = lathe(name + "Clamp", [(0.0540, -0.064 - D), (0.0556, -0.0634 - D), (0.0563, -0.0620 - D), (0.0563, -0.0480 - D),
                                   (0.0556, -0.0466 - D), (0.0548, -0.0460 - D), (0.0545, -0.0445 - D)], segs, [M["graphite"]])
    for k in range(6):
        t = k * 2 * math.pi / 6 + 0.3
        sc = sb.prim("cyl", name + "Screw", loc=(0.0565 * math.cos(t), -0.055 - D, 0.0565 * math.sin(t)), vertices=16, radius=0.0022, depth=0.0014, mat=M["metal"])
        sc.rotation_mode = 'QUATERNION'
        sc.rotation_quaternion = V((math.cos(t), 0, math.sin(t))).to_track_quat('Z', 'Y')
        sc.parent = root
        parts.append(sc)
    housing = lathe(name + "Housing", [(0.0520, -0.0470 - D), (0.0588, -0.0462 - D), (0.0596, -0.0450 - D), (0.0596, -0.0395 - D),
                                       (0.0582, -0.0390 - D), (0.0582, -0.0370 - D), (0.0596, -0.0365 - D), (0.0596, -0.0290 - D),
                                       (0.0590, -0.0283 - D), (0.0575, -0.0280 - D), (0.0560, -0.0275 - D)], segs, [M["graphite"]])
    c0 = -0.0277 - D
    cprof = [(0.0560, c0), (0.0612, c0 + 0.0003), (0.0624, c0 + 0.0011)]
    for a in np.linspace(c0 + 0.0019, -0.0072, 40):
        cprof.append((0.0626, float(a)))
    cprof += [(0.0624, -0.0064), (0.0610, -0.0058), (0.0590, -0.0056)]
    collar_e = sb.empty(name + "CollarRot", parent=root)
    collar = lathe(name + "Collar", cprof, segs, [M["metal"]], rfun=lambda r, a, t: _scallop(r, a, t, n=16, a0=c0 + 0.0045, a1=-0.0098, depth=0.0024))
    flange = lathe(name + "Flange", [(0.0580, -0.0058), (0.0648, -0.0058), (0.0662, -0.0052), (0.0666, -0.0040),
                                     (0.0664, -0.0012), (0.0655, -0.0004), (0.0630, -0.0002), (0.0500, -0.0002),
                                     (0.0488, -0.0006), (0.0482, -0.0016)], segs, [M["amber"]], rfun=_lugs)
    for o in (collar, flange):
        o.parent = collar_e
    mk = sb.prim("cube", name + "MarkC", loc=(0, (c0 - 0.0072) / 2, 0.0628), scale=(0.0008, (-0.0072 - c0) / 2 - 0.002, 0.00025), mat=M["paint"], parent=collar_e)
    gring = lathe(name + "GloveRing", [(0.0440, -0.0060), (0.0472, -0.0060), (0.0478, -0.0050), (0.0478, 0.0040),
                                       (0.0492, 0.0048), (0.0494, 0.0085), (0.0480, 0.0098), (0.0450, 0.0100),
                                       (0.0440, 0.0090)], segs, [M["polished"]])
    # lock indicator: graphite boss on the housing; spring pin with a graphite cap over an amber band that is only
    # revealed when the pin snaps out (locked).  pin travel = PIN_POP.
    boss = lathe(name + "PinBoss", [(0.0, 0.0120), (0.0046, 0.0120), (0.0066, 0.0114), (0.0072, 0.0100), (0.0072, 0.0),
                                    (0.0086, -0.003)], 64, [M["graphite"]], axis='Z')
    boss.location = (0, -0.0335 - D, 0.0550)
    boss.parent = root
    pin_root = sb.empty(name + "PinRoot", loc=(0, -0.0335 - D, 0.0550 + 0.0120), parent=root)
    pin = lathe(name + "Pin", [(0.0, 0.0022), (0.0030, 0.0022), (0.0040, 0.0014), (0.0042, 0.0), (0.0042, -0.0010),
                               (0.0040, -0.0012)], 48, [M["graphite"]], axis='Z')
    pin.parent = pin_root
    pband = lathe(name + "PinBand", [(0.0040, -0.0012), (0.0041, -0.0016), (0.0041, -0.0056), (0.0038, -0.0060), (0.0030, -0.0062)], 48, [M["amber"]], axis='Z')
    pband.parent = pin_root
    parts.append(pband)
    mh = sb.prim("cube", name + "MarkH", loc=(0, -0.0290 - D - 0.004, 0.0598), scale=(0.0008, 0.0035, 0.00025), mat=M["paint"], parent=root)
    for o in (clamp, housing, gring):
        o.parent = root
    parts += [clamp, housing, collar, flange, gring, boss, pin, mk, mh]
    return dict(root=root, collar=collar_e, pin=pin, pin_root=pin_root, parts=parts, flange=flange, collar_mesh=collar)


def _fold_disp(scale=1.0, amp=0.0022, seed=0.0, along=18.0, around=5.0):
    """soft fabric folds: circumferential creases (high freq along the tube) broken up by noise."""
    def f(i, s, th, p):
        q = V((math.cos(th) * around * 0.1, math.sin(th) * around * 0.1, s * along)) + V((seed, seed * 0.7, 0))
        n1 = noise.noise(q)
        n2 = noise.noise(q * 2.3 + V((3.1, 1.7, 0.4)))
        ridge = 1.0 - abs(n1)
        return amp * scale * (ridge ** 4 - 0.35 + 0.45 * n2)
    return f


def sleeve(M, length=0.32, name="Sleeve", n=128, rings=None, seed=1.0, folds=1.0, coll=None):
    """forearm sleeve in disconnect-local space: from y=-0.046 (under the clamp band) back to y=-length."""
    k = rings or max(40, int(length / 0.0025))
    ys = np.linspace(-0.0455 - WD_SHIFT, -length, k)
    path = np.stack([np.zeros(k), ys, np.zeros(k)], 1)

    def prof(i, s):
        # tucked under the clamp, then flares to the forearm tube
        t = s / length
        r = 0.0536 + 0.0125 * sb.smoother(min(1.0, s / 0.05)) + 0.004 * sb.smooth(min(1.0, max(0.0, (s - 0.12) / 0.15)))
        return r * 1.02, r * 0.98, 2.1

    fd = _fold_disp(amp=0.0055 * folds, seed=seed, along=16.0, around=6.0)

    def disp(i, s, th, p):
        w = sb.smooth(min(1.0, max(0.0, (s - 0.012) / 0.03)))    # no folds under the clamp
        return fd(i, s, th, p) * w

    o, info = tube(name, path, prof, n=n, up=(0, 0, 1), disp=disp, mats=[M["fabric"]], seam_theta=-math.pi / 2,
                   seams_v=(0.018,), coll=coll)
    sb.recalc_normals(o)
    return o


def forearm(side="L", glove_pose="relaxed", mats=None, sleeve_len=0.32, glove_voxel=0.6, name="FA", glove_npz_path=None):
    """Close-up forearm: sleeve + wrist disconnect + glove. Root = disconnect ring face, +Y toward the hand.
    For side 'L' the glove is the left glove (thumb on +X of the glove frame)."""
    M = mats or materials()
    root = sb.empty(name)
    wd = wrist_disconnect(M, name + "WD")
    wd["root"].parent = root
    sl = sleeve(M, sleeve_len, name + "Sleeve")
    sl.parent = root
    groot = sb.empty(name + "GloveRoot", loc=(0, GLOVE_OFF, 0), parent=root)
    g = glove(glove_pose, side, glove_voxel, M, name + "Glove", npz=glove_npz_path)
    g.parent = groot
    return dict(root=root, collar=wd["collar"], pin=wd["pin"], pin_root=wd["pin_root"], glove=g, glove_root=groot,
                sleeve=sl, parts=wd["parts"] + [sl, g], wd=wd, mats=M)

# =========================================================================== full suit

BONES = {
    # name: (head, tail, parent)
    "pelvis": ((0, 0.0, 0.95), (0, 0.0, 1.10), None),
    "spine": ((0, 0.0, 1.10), (0, 0.0, 1.50), "pelvis"),
}
ARM_A = math.radians(16)     # bind-pose abduction of the arms
_SH = (0.268, 0.0, 1.425)
_UA, _FA, _HD = 0.300, 0.275, 0.20
_HIP = (0.112, -0.005, 0.905)
_KNEE_Z, _ANK_Z = 0.525, 0.135


def _limb_points(side):
    sx = 1 if side == "L" else -1
    d = V((sx * math.sin(ARM_A), 0.0, -math.cos(ARM_A)))
    sh = V((sx * _SH[0], _SH[1], _SH[2]))
    el = sh + d * _UA
    wr = el + d * _FA
    hd = wr + d * _HD
    hip = V((sx * _HIP[0], _HIP[1], _HIP[2]))
    knee = V((sx * 0.112, 0.0, _KNEE_Z))
    ank = V((sx * 0.118, 0.012, _ANK_Z))
    toe = V((sx * 0.120, -0.150, 0.045))
    return dict(sh=sh, el=el, wr=wr, hd=hd, hip=hip, knee=knee, ank=ank, toe=toe, d=d)


def _make_rig(name):
    ad = bpy.data.armatures.new(name + "Rig")
    arm = bpy.data.objects.new(name, ad)
    sb.link_obj(arm)
    ad.display_type = 'STICK'
    bpy.context.view_layer.objects.active = arm
    for o in bpy.context.selected_objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = ad.edit_bones

    def add(n, h, t, parent=None, xaxis=(1, 0, 0)):
        b = eb.new(n)
        b.head = V(h); b.tail = V(t)
        b.align_roll(V((0, -1, 0)) if abs((V(t) - V(h)).normalized().y) < 0.9 else V((0, 0, 1)))
        if parent:
            b.parent = eb[parent]
            b.use_connect = False
        return b
    add("pelvis", (0, 0, 0.95), (0, 0, 1.10))
    add("spine", (0, 0, 1.10), (0, 0, 1.50), "pelvis")
    for s in ("L", "R"):
        P = _limb_points(s)
        add("upperarm." + s, P["sh"], P["el"], "spine")
        add("forearm." + s, P["el"], P["wr"], "upperarm." + s)
        add("hand." + s, P["wr"], P["hd"], "forearm." + s)
        add("thigh." + s, P["hip"], P["knee"], "pelvis")
        add("shin." + s, P["knee"], P["ank"], "thigh." + s)
        add("foot." + s, P["ank"], P["toe"], "shin." + s)
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    return arm


def _skin(o, arm, groups):
    """groups: {bone: weights array or 1.0}. parents to the armature with an Armature modifier."""
    nv = len(o.data.vertices)
    for bname, w in groups.items():
        vg = o.vertex_groups.new(name=bname)
        if isinstance(w, (int, float)):
            vg.add(list(range(nv)), float(w), 'REPLACE')
        else:
            w = np.asarray(w)
            for val in np.unique(np.round(w, 3)):
                if val <= 0.0005:
                    continue
                idx = np.nonzero(np.abs(np.round(w, 3) - val) < 1e-6)[0].tolist()
                vg.add(idx, float(val), 'REPLACE')
    mod = o.modifiers.new("Rig", 'ARMATURE')
    mod.object = arm
    mod.use_deform_preserve_volume = True
    mw = o.matrix_world.copy()
    o.parent = arm
    o.matrix_world = mw
    return o


def _rigid(o, arm, bone):
    """rigid part: world-space bind matrix preserved, deformed by one bone (via modifier, so it can be joined)."""
    bpy.context.view_layer.update()
    return _skin(o, arm, {bone: 1.0})


def _apply_xf(o):
    """bake object transform (and parent chain) into mesh data, clear parent."""
    bpy.context.view_layer.update()
    mw = o.matrix_world.copy()
    o.parent = None
    if o.type == 'MESH':
        o.data.transform(mw)
        o.matrix_world = Matrix.Identity(4)
    return o


def _join(objs, name):
    objs = [o for o in objs if o.type == 'MESH']
    for o in objs:
        _apply_xf(o)
    return sb.join(objs, name) if len(objs) > 1 else objs[0]


def _hfun(z0=0.0, z1=1.9):
    return lambda Vt: np.clip((np.asarray(Vt)[:, 2] - z0) / (z1 - z0), 0, 1)


def _smooth_w(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _bez(p0, p1, p2, k):
    t = np.linspace(0, 1, k)[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) + t * t * np.asarray(p2)


def _convolute(s, s0, s1, amp=0.006, lam=0.021):
    """bellows ribs between s0..s1."""
    if s < s0 or s > s1:
        return 0.0
    w = min(1.0, (s - s0) / 0.012, (s1 - s) / 0.012)
    return amp * w * (0.5 + 0.5 * math.cos(2 * math.pi * (s - s0) / lam))


def build(name="Astro", visor="gold", gloves=("relaxed", "relaxed"), dust=0.0, dirt=0.35, glove_voxel=1.2,
          mats=None, detail=1.0, occupant=True):
    """Full posable EVA suit. See module docstring."""
    M = mats or materials(dust=dust, dirt=dirt)
    if mats is not None:
        set_dust(M, dust)
    arm = _make_rig(name)
    parts = {}
    N = lambda x: max(16, int(x * detail))
    hf = _hfun()
    rigid = []   # (obj, bone)

    # ---------------------------------------------------------- torso (HUT) - hard shell under fabric
    zs = np.linspace(1.095, 1.605, N(70))
    path = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], 1)
    # (z, half-width, half-depth, y-centre)
    TK = np.array([(1.095, 0.190, 0.150, 0.005), (1.16, 0.205, 0.158, 0.0), (1.26, 0.238, 0.172, -0.004),
                   (1.36, 0.262, 0.180, -0.004), (1.43, 0.268, 0.176, 0.0), (1.49, 0.245, 0.168, 0.004),
                   (1.54, 0.200, 0.156, 0.006), (1.58, 0.158, 0.142, 0.004), (1.605, 0.140, 0.136, 0.0)])

    def tk(z, c):
        return float(np.interp(z, TK[:, 0], TK[:, c]))
    path[:, 1] = [tk(z, 3) for z in zs]

    def tprof(i, s):
        z = zs[i]
        ex = 2.0 + 1.1 * float(np.clip((1.58 - z) / 0.2, 0, 1)) * float(np.clip((z - 1.10) / 0.08, 0, 1))
        return tk(z, 1), tk(z, 2), ex
    fdt = _fold_disp(amp=0.0012, seed=4.0, along=9.0, around=3.0)

    def tdisp(i, s, th, p):
        return fdt(i, s, th, p) * float(np.clip((1.24 - zs[i]) / 0.1, 0, 1))
    torso, _ = tube(name + "Torso", path, tprof, n=N(96), up=(0, -1, 0), disp=tdisp, mats=[M["fabric"]],
                    seam_theta=math.pi / 2 + 0.001, seams_v=(0.13, 0.33), hfun=hf)
    sb.recalc_normals(torso)
    rigid.append((torso, "spine"))
    # neck ring + amber lock collar (helmet mount)
    nr = lathe(name + "NeckRing", [(0.128, 1.592, 0), (0.1415, 1.594, 0.6), (0.1440, 1.600, 1), (0.1440, 1.618, 0.5), (0.1400, 1.622, 1),
                                   (0.1300, 1.623, 0.2)], N(128), [M["graphite"]], axis='Z')
    nra = lathe(name + "NeckAmber", [(0.138, 1.6225, 0), (0.1465, 1.6225, 0.5), (0.1480, 1.6250, 1), (0.1480, 1.6320, 1),
                                     (0.1465, 1.6340, 1), (0.1380, 1.6345, 0.2)], N(128), [M["amber"]], axis='Z', rfun=_lugs_neck)
    nrm = lathe(name + "NeckMetal", [(0.132, 1.6345, 0), (0.1420, 1.6345, 0.6), (0.1430, 1.6360, 1), (0.1430, 1.6440, 0.4),
                                     (0.1410, 1.6460, 1), (0.1300, 1.6465, 0)], N(128), [M["metal"]], axis='Z')
    rigid += [(nr, "spine"), (nra, "spine"), (nrm, "spine")]
    # chest control module (graphite + white) with knobs, no text
    dcm = rbox(name + "DCM", (0.25, 0.085, 0.105), r=0.014, mats=[M["graphite"]])
    dcm.location = (0, -0.205, 1.30)
    dcm_top = rbox(name + "DCMTop", (0.23, 0.07, 0.03), r=0.01, mats=[M["fabric_soft"]])
    dcm_top.location = (0, -0.207, 1.365)
    rigid += [(dcm, "spine"), (dcm_top, "spine")]
    for k, (x, z, r_) in enumerate(((-0.085, 1.33, 0.013), (-0.03, 1.33, 0.010), (0.03, 1.33, 0.010), (0.085, 1.33, 0.013), (0.0, 1.275, 0.016))):
        kb = lathe(name + "Knob%d" % k, [(0.0, 0.028, 0), (r_ * 0.9, 0.028, 1), (r_, 0.025, 1), (r_, 0.006, 0.3), (r_ * 1.2, 0.0, 0.5)], 48, [M["graphite"]], axis='Z')
        kb.rotation_euler = (math.pi / 2, 0, 0)
        kb.location = (x, -0.2465, z)
        rigid.append((kb, "spine"))
    # umbilical hoses from the chest module around to the PLSS (corrugated)
    for sx in (-1, 1):
        hp = _bez((sx * 0.115, -0.205, 1.27), (sx * 0.30, -0.14, 1.20), (sx * 0.24, 0.17, 1.24), 40)
        ho, _ = tube(name + "Hose", hp, lambda i, s: (0.0115, 0.0115, 2.0), n=20,
                     disp=lambda i, s, th, p: 0.0012 * math.cos(2 * math.pi * s / 0.009), mats=[M["graphite"]], up=(0, 0, 1))
        _attr_float(ho.data, "edge", np.full(len(ho.data.vertices), 0.3))
        rigid.append((ho, "spine"))
    # stitched blank patches (chest, arm) + pocket
    patches = []
    # PLSS backpack
    plss = rbox(name + "PLSS", (0.47, 0.20, 0.62), r=0.045, mats=[M["fabric"]])
    plss.location = (0, 0.285, 1.33)
    _patch_uv_attrs(plss, 0.045)
    plss_side = []
    for sx in (-1, 1):
        sp = rbox(name + "PLSSSide", (0.03, 0.17, 0.56), r=0.012, mats=[M["graphite"]])
        sp.location = (sx * 0.232, 0.285, 1.33)
        plss_side.append(sp)
    ptop = rbox(name + "PLSSTop", (0.40, 0.16, 0.035), r=0.012, mats=[M["graphite"]])
    ptop.location = (0, 0.29, 1.645)
    handle, _ = tube(name + "PLSSHandle", _bez((-0.13, 0.36, 1.645), (0.0, 0.36, 1.72), (0.13, 0.36, 1.645), 30),
                     lambda i, s: (0.009, 0.009, 2.0), n=16, mats=[M["graphite"]])
    _attr_float(handle.data, "edge", np.full(len(handle.data.vertices), 0.6))
    vent = rbox(name + "PLSSVent", (0.16, 0.01, 0.07), r=0.004, mats=[M["graphite"]])
    vent.location = (0.10, 0.386, 1.12)
    rigid += [(plss, "spine"), (ptop, "spine"), (handle, "spine"), (vent, "spine")] + [(p, "spine") for p in plss_side]
    # ---------------------------------------------------------- helmet
    hc = V((0, -0.012, 1.735))
    bub = sb.prim("sphere", name + "Bubble", loc=hc, segments=N(96), ring_count=N(48), radius=0.188, mat=M["bubble"])
    _cut_below(bub, 1.636)
    rigid.append((bub, "spine"))
    evva = _shell(name + "EVVA", hc, 0.201, M["fabric"], front_cut=True)
    rigid.append((evva, "spine"))
    if visor in ("gold", "amber"):
        vs = _visor(name + "Visor", hc, 0.197, M["gold"] if visor == "gold" else M["amber_visor"])
        rigid.append((vs, "spine"))
    else:
        vs = _visor(name + "Visor", hc, 0.197, M["gold"], raised=True)
        rigid.append((vs, "spine"))
        if occupant:
            for h_ in _occupant(name + "Occ", hc, M, **(occupant if isinstance(occupant, dict) else {})):
                rigid.append((h_, "spine"))
    for sx in (-1, 1):
        lamp = rbox(name + "Lamp", (0.045, 0.075, 0.042), r=0.012, mats=[M["graphite"]])
        lamp.location = hc + V((sx * 0.192, -0.02, 0.07))
        lamp.rotation_euler = (0, sx * math.radians(18), 0)
        lens = sb.prim("cyl", name + "LampLens", loc=hc + V((sx * 0.192, -0.058, 0.07)), rot=(math.pi / 2, 0, 0), vertices=32, radius=0.014, depth=0.004, mat=M["lens"])
        rigid += [(lamp, "spine"), (lens, "spine")]
    # ---------------------------------------------------------- lower torso: waist bearing + brief
    wb = lathe(name + "WaistBearing", [(0.170, 1.075, 0), (0.1925, 1.076, 0.7), (0.196, 1.082, 1), (0.196, 1.104, 0.4),
                                       (0.1925, 1.110, 1), (0.170, 1.111, 0)], N(128), [M["graphite"]], axis='Z',
               rfun=lambda r, a, t: r * _ell(t, 1.0, 0.79))
    rigid.append((wb, "pelvis"))
    zs2 = np.linspace(1.085, 0.845, N(40))
    path2 = np.stack([np.zeros_like(zs2), np.zeros_like(zs2), zs2], 1)
    BK = np.array([(1.085, 0.192, 0.152), (1.02, 0.200, 0.148), (0.95, 0.205, 0.140), (0.90, 0.19, 0.125), (0.86, 0.15, 0.10), (0.845, 0.10, 0.07)])

    def bprof(i, s):
        z = zs2[i]
        return float(np.interp(z, BK[::-1, 0], BK[::-1, 1])), float(np.interp(z, BK[::-1, 0], BK[::-1, 2])), 2.4
    fdb = _fold_disp(amp=0.005, seed=7.0, along=12.0, around=4.0)
    brief, _ = tube(name + "Brief", path2, bprof, n=N(96), up=(0, -1, 0), disp=fdb, mats=[M["fabric"]],
                    seam_theta=math.pi / 2, seams_v=(0.06,), cap1=True, hfun=hf)
    sb.recalc_normals(brief)
    rigid.append((brief, "pelvis"))
    # ---------------------------------------------------------- limbs
    for s in ("L", "R"):
        sx = 1 if s == "L" else -1
        P = _limb_points(s)
        d = P["d"]
        # shoulder bearing: ring on the HUT side, normal out & down
        nrm = V((sx * math.cos(math.radians(40)), 0, -math.sin(math.radians(40))))
        bc = P["sh"]
        sbear = lathe(name + "ShBearing" + s, [(0.078, -0.020, 0), (0.0885, -0.019, 0.7), (0.0905, -0.012, 1), (0.0905, 0.006, 0.5),
                                               (0.0885, 0.012, 1), (0.0800, 0.013, 0)], N(96), [M["graphite"]])
        sbear.rotation_mode = 'QUATERNION'
        sbear.rotation_quaternion = nrm.to_track_quat('Y', 'Z')
        sbear.location = bc
        rigid.append((sbear, "spine"))
        # arm tube: from inside the bearing along the normal, curving into the arm direction, to the wrist clamp
        p0 = bc - nrm * 0.005
        p1 = bc + nrm * 0.055
        p2 = bc + d * 0.12 + nrm * 0.02
        seg1 = _bez(p0, p1, p2, 18)
        wr_clamp = P["wr"] - d * (0.064 + WD_SHIFT)
        seg2 = np.linspace(np.asarray(p2), np.asarray(wr_clamp), N(90))[1:]
        apath = np.concatenate([seg1, seg2], 0)
        seglen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(apath, axis=0), axis=1))])
        s_el = float(seglen[np.argmin(np.linalg.norm(apath - np.asarray(P["el"]), axis=1))])
        s_ub = s_el - 0.13     # upper-arm bearing
        L_arm = float(seglen[-1])

        def aprof(i, s_, s_el=s_el, L=L_arm):
            r = 0.080 - 0.012 * sb.smooth(s_ / 0.22) - 0.010 * sb.smooth((s_ - s_el) / 0.18)
            r -= 0.0105 * sb.smoother(max(0.0, (s_ - (L - 0.035)) / 0.035))   # tuck into the clamp
            r += _convolute(s_, s_el - 0.05, s_el + 0.05, 0.0055, 0.0205)
            return r * 1.03, r * 0.97, 2.0
        fda = _fold_disp(amp=0.0055, seed=10.0 + sx, along=14.0, around=5.0)

        def adisp(i, s_, th, p, s_el=s_el, L=L_arm, s_ub=s_ub):
            w = min(1.0, max(0.0, (L - 0.01 - s_) / 0.03))
            w *= 0.35 if abs(s_ - s_el) < 0.05 else 1.0
            w *= min(1.0, abs(s_ - s_ub) / 0.03)
            return fda(i, s_, th, p) * w
        atube, info = tube(name + "Arm" + s, apath, aprof, n=N(80), up=(0, -1, 0), disp=adisp, mats=[M["fabric"]],
                           seam_theta=(math.pi if s == "L" else 0.0), seams_v=(0.10, s_ub - 0.02, s_ub + 0.02), hfun=hf)
        sb.recalc_normals(atube)
        vv = info["verts"]
        sv = np.repeat(info["s"], N(80))
        w_sp = 1 - _smooth_w(sv, 0.0, 0.10)
        w_fa = _smooth_w(sv, s_el - 0.05, s_el + 0.05)
        w_ua = np.clip(1 - w_sp - w_fa, 0, 1)
        parts["arm" + s] = atube
        _skin(atube, arm, {"spine": w_sp, "upperarm." + s: w_ua, "forearm." + s: w_fa})
        # upper-arm bearing (graphite ring around the tube)
        ub_c = V(apath[np.argmin(np.abs(seglen - s_ub))])
        ubr = lathe(name + "UABearing" + s, [(0.068, -0.012, 0), (0.0735, -0.011, 0.7), (0.0755, -0.006, 1), (0.0755, 0.006, 1),
                                             (0.0735, 0.011, 0.7), (0.068, 0.012, 0)], N(80), [M["graphite"]])
        ubr.rotation_mode = 'QUATERNION'
        ubr.rotation_quaternion = d.to_track_quat('Y', 'Z')
        ubr.location = ub_c
        rigid.append((ubr, "upperarm." + s))
        # stitched patch on the upper arm (outer side), blank
        pt = _patch(name + "ArmPatch" + s, ub_c - d * 0.08, d, V((sx, 0, 0)), 0.066, 0.05, 0.0745, M["fabric_soft"])
        rigid.append((pt, "upperarm." + s))
        # wrist disconnect + glove on the forearm/hand
        wd = wrist_disconnect(M, name + "WD" + s, segs=N(96))
        q = d.to_track_quat('Y', 'Z')
        # orient so the lock pin sits on top of the wrist (+Z of the disconnect -> world up-ish/outward)
        wd["root"].rotation_mode = 'QUATERNION'
        wd["root"].rotation_quaternion = _frame_quat(d, V((sx * 0.3, -1.0, 0.0)))
        wd["root"].location = P["wr"]
        bpy.context.view_layer.update()
        for o in wd["parts"]:
            rigid.append((o, "forearm." + s))
        gl = glove(gloves[0 if s == "L" else 1], s, glove_voxel, M, name + "Glove" + s)
        # glove frame: +Y along d, palm (-Z) faces the body (medial), thumb forward
        med = V((-sx, 0, 0))
        gl.matrix_world = Matrix.Translation(P["wr"] + d * GLOVE_OFF) @ _frame_quat(d, -med).to_matrix().to_4x4()
        rigid.append((gl, "hand." + s))
        parts["glove" + s] = gl
        # ---- leg
        hip = P["hip"]
        hn = V((sx * 0.25, -0.05, -1.0)).normalized()
        hbear = lathe(name + "HipBearing" + s, [(0.086, -0.016, 0), (0.0955, -0.015, 0.7), (0.0975, -0.009, 1), (0.0975, 0.008, 0.5),
                                                (0.0955, 0.013, 1), (0.086, 0.014, 0)], N(96), [M["graphite"]])
        hbear.rotation_mode = 'QUATERNION'
        hbear.rotation_quaternion = hn.to_track_quat('Y', 'Z')
        hbear.location = hip + hn * 0.01
        rigid.append((hbear, "pelvis"))
        ld = (P["ank"] - P["knee"]).normalized()
        lpath = np.concatenate([_bez(hip, hip + hn * 0.08, P["knee"] + (hip - P["knee"]).normalized() * 0.22, 16),
                                np.linspace(np.asarray(P["knee"] + (hip - P["knee"]).normalized() * 0.22), np.asarray(P["ank"] + V((0, 0, 0.085))), N(80))[1:]], 0)
        lslen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lpath, axis=0), axis=1))])
        s_kn = float(lslen[np.argmin(np.linalg.norm(lpath - np.asarray(P["knee"]), axis=1))])
        L_leg = float(lslen[-1])

        def lprof(i, s_, s_kn=s_kn, L=L_leg):
            r = 0.112 - 0.012 * sb.smooth(s_ / 0.3) - 0.014 * sb.smooth((s_ - s_kn) / 0.2)
            r += _convolute(s_, s_kn - 0.06, s_kn + 0.06, 0.006, 0.023)
            r -= 0.006 * sb.smoother(max(0.0, (s_ - (L - 0.03)) / 0.03))
            return r * 1.02, r * 0.98, 2.0
        fdl = _fold_disp(amp=0.006, seed=20.0 + sx, along=11.0, around=4.0)

        def ldisp(i, s_, th, p, s_kn=s_kn):
            return fdl(i, s_, th, p) * (0.35 if abs(s_ - s_kn) < 0.06 else 1.0)
        ltube, linfo = tube(name + "Leg" + s, lpath, lprof, n=N(80), up=(0, -1, 0), disp=ldisp, mats=[M["fabric"]],
                            seam_theta=(math.pi if s == "L" else 0.0), seams_v=(0.12, s_kn + 0.12), hfun=hf)
        sb.recalc_normals(ltube)
        sv = np.repeat(linfo["s"], N(80))
        w_pl = 1 - _smooth_w(sv, 0.0, 0.10)
        w_sh = _smooth_w(sv, s_kn - 0.06, s_kn + 0.06)
        w_th = np.clip(1 - w_pl - w_sh, 0, 1)
        parts["leg" + s] = ltube
        _skin(ltube, arm, {"pelvis": w_pl, "thigh." + s: w_th, "shin." + s: w_sh})
        # thigh pocket with flap
        pk_c = V(lpath[np.argmin(np.abs(lslen - (s_kn - 0.17)))])
        pk = _patch(name + "Pocket" + s, pk_c, (P["knee"] - hip).normalized(), V((sx * 0.55, -0.83, 0)).normalized(), 0.10, 0.085, 0.089, M["fabric_soft"], thick=0.012)
        rigid.append((pk, "thigh." + s))
        # ankle bearing + boot
        ab_c = P["ank"] + V((0, 0, 0.085))
        abr = lathe(name + "AnkleBearing" + s, [(0.070, -0.012, 0), (0.0775, -0.011, 0.7), (0.0795, -0.005, 1), (0.0795, 0.008, 1),
                                                (0.0775, 0.012, 0.7), (0.070, 0.013, 0)], N(80), [M["graphite"]])
        abr.rotation_mode = 'QUATERNION'
        abr.rotation_quaternion = V((0, 0, 1)).to_track_quat('Y', 'Z')
        abr.location = ab_c
        rigid.append((abr, "shin." + s))
        for bo in _boot(name + "Boot" + s, P["ank"], sx, M, hf):
            rigid.append((bo, "foot." + s))
    # ---------------------------------------------------------- chest patch
    cp = _patch(name + "ChestPatch", V((0.155, -0.176, 1.405)), V((0, 0, -1)), V((0.35, -0.94, 0)).normalized(), 0.075, 0.045, 0.40, M["fabric_soft"], flat=True)
    rigid.append((cp, "spine"))
    # ---------------------------------------------------------- merge rigid parts per bone (fewer objects)
    by_bone = {}
    for o, b in rigid:
        by_bone.setdefault(b, []).append(o)
    for b, objs in by_bone.items():
        # keep glass / visor separate (different render method), join the rest
        keep = [o for o in objs if o.type == 'MESH' and any(ms and ms.name.startswith(("SuitBubble", "SuitVisor")) for ms in o.data.materials)]
        rest = [o for o in objs if o not in keep and o.type == 'MESH']
        emp = [o for o in objs if o.type != 'MESH']
        if rest:
            j = _join(rest, name + "_" + b.replace(".", "_"))
            _skin(j, arm, {b: 1.0})
            parts[b] = j
        for o in keep:
            _apply_xf(o)
            _skin(o, arm, {b: 1.0})
            parts[o.name] = o
    # clean empties left from assemblies
    for o in list(bpy.data.objects):
        if o.type == 'EMPTY' and o.name.startswith(name) and not o.children:
            bpy.data.objects.remove(o)
    return dict(root=arm, rig=arm, parts=parts, mats=M, bones=[b.name for b in arm.data.bones])


def _ell(t, a, b):
    """radius multiplier for an ellipse a x b (normalised to a) at angle t."""
    return (a * b) / math.sqrt((b * math.cos(t)) ** 2 + (a * math.sin(t)) ** 2) / a


def _lugs_neck(r, a, t):
    if r > 0.146:
        for k in range(4):
            dt = ((t - k * math.pi / 2 - 0.4 + math.pi) % (2 * math.pi)) - math.pi
            if abs(dt) < 0.05:
                return r - 0.003 * (1 - (dt / 0.05) ** 4)
    return r


def _frame_quat(y_dir, z_hint):
    """rotation whose local +Y = y_dir and local +Z as close as possible to z_hint."""
    y = V(y_dir).normalized()
    z = V(z_hint) - y * V(z_hint).dot(y)
    z.normalize()
    x = y.cross(z)
    return Matrix((x, y, z)).transposed().to_quaternion()


def _cut_below(o, zcut):
    bm = bmesh.new(); bm.from_mesh(o.data)
    mw = o.matrix_world.copy()
    kill = [v for v in bm.verts if (mw @ v.co).z < zcut]
    bmesh.ops.delete(bm, geom=kill, context='VERTS')
    bm.to_mesh(o.data); bm.free()


def _shell(name, c, R, mat, front_cut=True):
    """visor-assembly shell: spherical cap over top/back/sides of the bubble with a front opening + thickness."""
    segs, rings = 96, 48
    verts, faces = [], []
    th0, th1 = math.radians(-12), math.radians(88)     # elevation range
    for i in range(rings + 1):
        el = th0 + (th1 - th0) * i / rings
        for j in range(segs + 1):
            az = -math.pi + 2 * math.pi * j / segs      # az 0 = front (-Y)
            # front opening: lower edge rises toward the front
            verts.append(V((R * math.cos(el) * math.sin(az), -R * math.cos(el) * math.cos(az), R * math.sin(el))) + c)
    keep = []
    for i in range(rings):
        for j in range(segs):
            el = th0 + (th1 - th0) * (i + 0.5) / rings
            az = -math.pi + 2 * math.pi * (j + 0.5) / segs
            # opening in front: |az| < 70deg and el < 50deg is open
            open_ = abs(az) < math.radians(78) and el < math.radians(56 - 12 * (abs(az) / math.radians(78)) ** 2)
            side_low = abs(az) > math.radians(78) and el < math.radians(-8 + 10 * max(0, (abs(az) - math.radians(78)) / 1.5))
            if open_ or side_low:
                continue
            a = i * (segs + 1) + j
            faces.append((a, a + 1, a + segs + 2, a + segs + 1))
    o = sb.mesh_obj(name, verts, faces, mat)
    bm = bmesh.new(); bm.from_mesh(o.data)
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(o.data); bm.free()
    sm = sb.solidify(o, 0.008, offset=1)
    sm.use_rim = True
    sb.apply_mods(o)
    sb.recalc_normals(o)
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    uv = []
    for p in me.polygons:
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co - c
            az = math.atan2(co.x, -co.y); el = math.asin(max(-1, min(1, co.z / co.length)))
            uv.append((az * R, el * R))
    uvl.uv.foreach_set("vector", np.asarray(uv, np.float32).ravel())
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    _attr_float(me, "seam", np.full(len(me.vertices), 0.05))
    _attr_float(me, "seam_t", np.zeros(len(me.vertices)))
    _attr_float(me, "h", np.full(len(me.vertices), 1.0))
    return o


def _visor(name, c, R, mat, raised=False):
    segs, rings = 72, 36
    verts, faces = [], []
    e0, e1 = (math.radians(-14), math.radians(52)) if not raised else (math.radians(46), math.radians(80))
    a0 = math.radians(76)
    for i in range(rings + 1):
        el = e0 + (e1 - e0) * i / rings
        for j in range(segs + 1):
            az = -a0 + 2 * a0 * j / segs
            verts.append(V((R * math.cos(el) * math.sin(az), -R * math.cos(el) * math.cos(az), R * math.sin(el))) + c)
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j
            faces.append((a, a + 1, a + segs + 2, a + segs + 1))
    o = sb.mesh_obj(name, verts, faces, mat)
    sb.solidify(o, 0.002, offset=1)
    sb.apply_mods(o)
    sb.recalc_normals(o)
    return o


def _occupant(name, hc, M, phenotype=None, skin="young_caucasian_male", eyes=True):
    """Real MPFB head (docs/PEOPLE.md) inside the helmet, wearing a fitted comm cap built from its own scalp.
    Only built for visor='clear'. ~1 s + MPFB load."""
    import mhchild
    ph = dict(age=0.5, gender=0.65, weight=0.5, muscle=0.55, race=dict(african=0.3, asian=0.3, caucasian=0.4))
    if phenotype:
        ph.update(phenotype)
    bm_, rig, parts = mhchild.build(name=name, phenotype=ph, hair=None, clothes=[], skin=skin)
    if eyes:
        try:
            mhchild.fix_eyes(parts)
        except Exception:
            pass
    out = []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in parts:
        for m_ in list(o.modifiers):
            if m_.type == 'ARMATURE':
                o.modifiers.remove(m_)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        nm = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        o.modifiers.clear()
        if o.data.shape_keys:
            o.shape_key_clear()
        o.data = nm
        mw = o.matrix_world.copy()
        o.parent = None
        o.data.transform(mw)
        o.matrix_world = Matrix.Identity(4)
        for vg in list(o.vertex_groups):
            o.vertex_groups.remove(vg)
        out.append(o)
    bpy.data.objects.remove(rig)
    # anchor on the real eyes: eye centres at helmet-centre height, ~11 cm behind the bubble front
    eyes_o = next(o for o in out if "high-poly" in o.name)
    ec = np.zeros(len(eyes_o.data.vertices) * 3); eyes_o.data.vertices.foreach_get("co", ec); ec = ec.reshape(-1, 3)
    e_c = V(tuple(ec.mean(0)))
    target = V((0.0, hc.y - 0.188 + 0.112, hc.z + 0.026))
    shift = target - e_c
    zcut = e_c.z - 0.165
    print("OCC eyes", e_c, "shift", shift)
    for o in out:
        _cut_below(o, zcut)
        o.data.transform(Matrix.Translation(shift))
    body = next(o for o in out if o.name.endswith(".body"))
    # comm cap: scalp + ear region of the head, pushed out 4.5 mm, solidified
    c = target + V((0, 0.085, 0.0))
    bmx = bmesh.new(); bmx.from_mesh(body.data)
    bmx.verts.ensure_lookup_table()

    def in_cap(v):
        q = v.co - c
        face = q.y < -0.035 and q.z < 0.055 and abs(q.x) < 0.062
        return (q.z > -0.075 and not face) and q.z > -0.1 and not (q.y < -0.02 and q.z < -0.03)
    keep = set(f.index for f in bmx.faces if all(in_cap(v) for v in f.verts))
    bmx.free()
    cap = body.copy(); cap.data = body.data.copy(); cap.name = name + "CommCap"
    sb.link_obj(cap)
    bmc = bmesh.new(); bmc.from_mesh(cap.data)
    bmesh.ops.delete(bmc, geom=[f for f in bmc.faces if f.index not in keep], context='FACES')
    bmesh.ops.delete(bmc, geom=[v for v in bmc.verts if not v.link_faces], context='VERTS')
    for v in bmc.verts:
        v.co += v.normal * 0.0045
    bmc.to_mesh(cap.data); bmc.free()
    cap.data.materials.clear()
    capm = sb.fabric(name + "CapFab", (0.045, 0.043, 0.042), weave=2200, sheen=0.3)
    capw = sb.fabric(name + "CapTop", (0.62, 0.60, 0.56), weave=2200, sheen=0.3)
    cap.data.materials.append(capm); cap.data.materials.append(capw)
    for pg in cap.data.polygons:
        q = pg.center - c
        pg.material_index = 0
    sm = sb.solidify(cap, 0.0025, offset=1)
    sb.subsurf(cap, 1)
    sb.apply_mods(cap)
    out.append(cap)
    for sx in (-1, 1):
        cup = sb.prim("sphere", name + "EarCup", loc=c + V((sx * 0.080, 0.008, -0.028)), segments=32, ring_count=16, radius=0.036, mat=capm)
        cup.scale = (0.42, 0.95, 1.1)
        out.append(cup)
    boom, _ = tube(name + "Mic", _bez(c + V((0.083, -0.01, -0.05)), c + V((0.07, -0.08, -0.08)), c + V((0.022, -0.105, -0.085)), 16),
                   lambda i, s_: (0.0028, 0.0028, 2.0), n=10, mats=[M["graphite"]])
    _attr_float(boom.data, "edge", np.zeros(len(boom.data.vertices)))
    out.append(boom)
    return out


def _cut_front(o, ymax, zmin):
    bm = bmesh.new(); bm.from_mesh(o.data)
    mw = o.matrix_world.copy()
    kill = [v for v in bm.verts if (mw @ v.co).y < ymax and (mw @ v.co).z < zmin + 0.06]
    bmesh.ops.delete(bm, geom=kill, context='VERTS')
    bm.to_mesh(o.data); bm.free()


def _patch_uv_attrs(o, rr):
    me = o.data
    n = len(me.vertices)
    _attr_float(me, "seam", np.full(n, 0.05))
    _attr_float(me, "seam_t", np.zeros(n))
    co = np.zeros(n * 3); me.vertices.foreach_get("co", co)
    _attr_float(me, "h", np.clip((co.reshape(-1, 3)[:, 2] + 1.33) / 1.9, 0, 1))


def _patch(name, c, along, out, w, h, R, mat, thick=0.0035, flat=False):
    """stitched blank patch (or pocket when thick) conforming to a cylinder of radius R about the limb axis.
    c: point on the limb axis (or on the surface when flat), along: limb axis, out: outward direction of the patch."""
    along = V(along).normalized()
    out = V(out) - along * V(out).dot(along); out.normalize()
    side = along.cross(out)
    nu, nv = 24, 18
    verts, faces, uvs = [], [], []
    rr = 0.006
    for i in range(nv + 1):
        for j in range(nu + 1):
            u = (j / nu - 0.5) * w
            v = (i / nv - 0.5) * h
            # rounded-rect puff: thicker at the centre
            eu = 1 - min(1.0, (abs(u) / (w / 2)) ** 6)
            ev = 1 - min(1.0, (abs(v) / (h / 2)) ** 6)
            lift = thick * (0.35 + 0.65 * (eu * ev) ** 0.3) + 0.0008
            if flat:
                p = V(c) + side * u + along * (-v) + out * lift
            else:
                ang = u / R
                p = V(c) + along * (-v) + (out * math.cos(ang) + side * math.sin(ang)) * (R + lift)
            verts.append(p)
            uvs.append((u, v))
    for i in range(nv):
        for j in range(nu):
            a = i * (nu + 1) + j
            faces.append((a, a + 1, a + nu + 2, a + nu + 1))
    o = sb.mesh_obj(name, verts, faces, mat)
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    fl = []
    for p in me.polygons:
        for li in p.loop_indices:
            fl.append(uvs[me.loops[li].vertex_index])
    uvl.uv.foreach_set("vector", np.asarray(fl, np.float32).ravel())
    U = np.asarray(uvs)
    dd = np.minimum(w / 2 - np.abs(U[:, 0]), h / 2 - np.abs(U[:, 1]))
    st = np.where(w / 2 - np.abs(U[:, 0]) < h / 2 - np.abs(U[:, 1]), U[:, 1], U[:, 0])
    _attr_float(me, "seam", np.clip(dd + 0.0002, 0, 1))
    _attr_float(me, "seam_t", st)
    _attr_float(me, "h", np.full(len(verts), 0.5))
    sb.solidify(o, 0.0015, offset=-1)
    sb.apply_mods(o)
    sb.recalc_normals(o)
    return o


def _boot(name, ank, sx, M, hf):
    """EVA boot: lofted upper (white fabric), graphite toe/heel guards, thick lugged sole."""
    out = []
    # foot upper: loft along -Y from heel to toe
    ys = np.linspace(0.085, -0.205, 44)
    path = np.stack([np.full_like(ys, ank.x), ys, np.zeros_like(ys)], 1)
    K = np.array([(0.085, 0.050, 0.050, 0.095), (0.07, 0.060, 0.070, 0.105), (0.03, 0.064, 0.085, 0.115),
                  (-0.03, 0.066, 0.075, 0.100), (-0.09, 0.068, 0.055, 0.082), (-0.15, 0.064, 0.042, 0.072),
                  (-0.185, 0.055, 0.035, 0.068), (-0.205, 0.030, 0.020, 0.066)])
    path[:, 2] = [float(np.interp(-y, -K[:, 0], K[:, 3])) for y in ys]

    def prof(i, s):
        y = ys[i]
        return float(np.interp(-y, -K[:, 0], K[:, 1])), float(np.interp(-y, -K[:, 0], K[:, 2])), 2.6
    up, _ = tube(name + "Upper", path, prof, n=48, up=(0, 0, 1), mats=[M["fabric"]], cap0=True, cap1=True,
                 disp=_fold_disp(amp=0.0012, seed=30 + sx, along=10, around=3), hfun=hf, seams_v=(0.09,))
    sb.recalc_normals(up)
    out.append(up)
    # shaft from the ankle bearing down into the foot
    zs = np.linspace(ank.z + 0.085, ank.z - 0.03, 16)
    sp = np.stack([np.full_like(zs, ank.x), np.full_like(zs, ank.y + 0.01), zs], 1)
    sh, _ = tube(name + "Shaft", sp, lambda i, s: (0.074, 0.078, 2.2), n=48, up=(0, -1, 0), mats=[M["fabric"]],
                 disp=_fold_disp(amp=0.0018, seed=40 + sx, along=12, around=4), hfun=hf)
    sb.recalc_normals(sh)
    out.append(sh)
    # sole: thick rounded slab with lugs (bump in material) and a graphite toe bumper
    sole = rbox(name + "Sole", (0.142, 0.315, 0.036), r=0.014, mats=[M["sole"]])
    sole.location = (ank.x, -0.058, 0.018)
    out.append(sole)
    toe = rbox(name + "Toe", (0.118, 0.06, 0.05), r=0.02, mats=[M["graphite"]])
    toe.location = (ank.x, -0.188, 0.055)
    heel = rbox(name + "Heel", (0.11, 0.035, 0.06), r=0.015, mats=[M["graphite"]])
    heel.location = (ank.x, 0.088, 0.065)
    out += [toe, heel]
    return out

# =========================================================================== posing

def _q_axis(axis, deg):
    return Quaternion(V(axis), math.radians(deg))


def _to_local(arm, bone, q_arm):
    """convert a rotation expressed in armature space (about the bone head) to the pose-bone local rotation."""
    B = arm.data.bones[bone].matrix_local.to_3x3()
    return (B.inverted() @ q_arm.to_matrix() @ B).to_quaternion()


def pose(S, frame=None, **J):
    """see module docstring. Unspecified joints keep their current rotation."""
    arm = S["rig"]
    pb = arm.pose.bones

    def setq(bone, q):
        p = pb[bone]
        p.rotation_quaternion = _to_local(arm, bone, q)
        if frame is not None:
            p.keyframe_insert("rotation_quaternion", frame=frame)
    if "pelvis" in J:
        f, sd, tw = J["pelvis"]
        setq("pelvis", _q_axis((0, 0, 1), tw) @ _q_axis((0, 1, 0), sd) @ _q_axis((1, 0, 0), -f))
    if "torso" in J:
        f, sd, tw = J["torso"]
        setq("spine", _q_axis((0, 0, 1), tw) @ _q_axis((0, 1, 0), sd) @ _q_axis((1, 0, 0), -f))
    for s in ("L", "R"):
        sx = 1 if s == "L" else -1
        k = s.lower()
        if k + "_shoulder" in J:
            f, o, tw = J[k + "_shoulder"]
            d = arm.data.bones["upperarm." + s].matrix_local.to_3x3() @ V((0, 1, 0))
            setq("upperarm." + s, _q_axis((0, 1, 0), -sx * o) @ _q_axis((1, 0, 0), -f) @ _q_axis(d, sx * tw))
        if k + "_elbow" in J:
            b = J[k + "_elbow"]
            setq("forearm." + s, _q_axis((1, 0, 0), -b))
        if k + "_wrist" in J:
            fl, dv, tw = J[k + "_wrist"]
            d = arm.data.bones["hand." + s].matrix_local.to_3x3() @ V((0, 1, 0))
            setq("hand." + s, _q_axis((0, 1, 0), sx * fl) @ _q_axis((1, 0, 0), -dv) @ _q_axis(d, sx * tw))
        if k + "_hip" in J:
            f, o, tw = J[k + "_hip"]
            d = arm.data.bones["thigh." + s].matrix_local.to_3x3() @ V((0, 1, 0))
            setq("thigh." + s, _q_axis((0, 1, 0), -sx * o) @ _q_axis((1, 0, 0), -f) @ _q_axis(d, sx * tw))
        if k + "_knee" in J:
            setq("shin." + s, _q_axis((1, 0, 0), J[k + "_knee"]))
        if k + "_ankle" in J:
            setq("foot." + s, _q_axis((1, 0, 0), -J[k + "_ankle"]))


def hand_socket(S, side="R"):
    return "hand." + side


def attach(S, obj, bone):
    """parent obj to a suit bone keeping its current world transform."""
    arm = S["rig"]
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    obj.parent = arm
    obj.parent_type = 'BONE'
    obj.parent_bone = bone
    bpy.context.view_layer.update()
    obj.matrix_world = mw
    return obj


def ground_offset(S):
    """lowest z of the evaluated suit (for foot contact when posing)."""
    dg = bpy.context.evaluated_depsgraph_get()
    zmin = 1e9
    for o in S["parts"].values():
        if "Boot" in o.name or "foot" in o.name:
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            mw = ev.matrix_world
            for v in me.vertices:
                zmin = min(zmin, (mw @ v.co).z)
            ev.to_mesh_clear()
    return zmin
