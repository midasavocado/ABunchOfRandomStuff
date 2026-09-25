"""s04 (270-360) HIDDEN STRUCTURE: a protein-like alpha/beta structure settles from a loose coil into its fold
while the camera travels through it (foreground occlusion, scale, controlled studio light). Ends face-on to the
parallel beta-sheet (strands horizontal) for the geometric match-cut into s05's parallel interconnects."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, molecule as MOL, dbgcam
from mathutils import Vector as V, Quaternion, Matrix

sid = (sb.argv() or ["s04"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
F0, F1 = 270, 360
S = MOL.SC

# ------------------------------------------------------------------ data
P_fold, SS, SID = MOL.fold()
P_unf = MOL.unfolded(P_fold)
N = len(P_fold)
G_fold = MOL.guides(P_fold, SS)
atoms = MOL.sidechains(P_fold, SS)
# keep the sheet's camera-facing (-Z) side clean for the final composition: drop side chains of strand residues
# that point below the sheet
pos0 = MOL.side_positions(P_fold, SS, atoms) / S
atoms = [a for a, p in zip(atoms, pos0) if not (SS[a[0]] == 'E' and p[2] < P_fold[a[0]][2] * 1.0 - 0.1)]
# segment rank for the staggered settle: helices form first, strands zip up in sheet order
seg = np.zeros(N)
cur, rank = SS[0], 0
for i in range(1, N):
    if SS[i] != cur:
        rank += 1
        cur = SS[i]
    seg[i] = rank
seg /= seg.max()


def progress(f):
    """per-residue fold progress 0..1 at (sub)frame f."""
    t0 = 250 + seg * 32.0           # staggered starts (the settle is under way at the cut)
    return np.array([sb.smoother((f - a) / 34.0) for a in t0])


def chain(f):
    w = progress(f)[:, None]
    # loose coil drifts slowly while unfolded; converges exactly to the fold
    drift = np.sin(np.arange(N)[:, None] * np.array([0.37, 0.53, 0.29]) + f * 0.05) * 1.2
    P = P_unf * (1 - w) + P_fold * w + drift * (1 - w)
    # helices wind up as they form: blend the coil radius in (already in P_fold); gentle breathing after settle
    return P


# ------------------------------------------------------------------ materials (premium molecular vis: satin, soft)
def satin(name, col, rough=0.42, sss=0.15, sheen=0.25):
    m = sb.mat(name, col, rough=rough, spec=0.45, sss=sss, sss_radius=(1, 1, 1), sss_scale=0.004, sheen=sheen, coat=0.15, coat_rough=0.3)
    m.node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
    nb = sb.NB(m)
    n = nb.noise(nb.coord('Object'), scale=180, detail=2)
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.015, distance=0.001))
    return m


M_H = satin("MolHelix", (0.78, 0.75, 0.69))
M_E = satin("MolStrand", (0.46, 0.56, 0.70))
M_L = satin("MolLoop", (0.36, 0.37, 0.40))
EL = {'C': satin("AtC", (0.60, 0.61, 0.64), 0.5), 'N': satin("AtN", (0.34, 0.43, 0.72), 0.5),
      'O': satin("AtO", (0.72, 0.33, 0.28), 0.5), 'S': satin("AtS", (0.86, 0.70, 0.30), 0.5)}
M_LIG = satin("AtLig", (0.95, 0.42, 0.08), 0.32, sss=0.3)

# ------------------------------------------------------------------ ribbon object
RB = MOL.Ribbon(SS)
V0 = RB.verts(chain(F0), MOL.guides(chain(F0), SS))
me = bpy.data.meshes.new("Ribbon")
me.from_pydata([tuple(v) for v in V0], [], RB.faces)
me.update()
rib = bpy.data.objects.new("Ribbon", me)
sb.link_obj(rib)
for m in (M_H, M_E, M_L):
    me.materials.append(m)
me.polygons.foreach_set("material_index", np.array(RB.mats, np.int32))
me.polygons.foreach_set("use_smooth", np.ones(len(RB.faces), bool))
me.update()


# ------------------------------------------------------------------ atoms (GN instanced spheres)
def sphere_gn(name, mat):
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = ng.nodes.new('NodeGroupInput'); go = ng.nodes.new('NodeGroupOutput')
    m2p = ng.nodes.new('GeometryNodeMeshToPoints')
    sph = ng.nodes.new('GeometryNodeMeshUVSphere')
    sph.inputs['Segments'].default_value = 32; sph.inputs['Rings'].default_value = 16
    sph.inputs['Radius'].default_value = 1.0
    ss_ = ng.nodes.new('GeometryNodeSetShadeSmooth')
    inst = ng.nodes.new('GeometryNodeInstanceOnPoints')
    rad = ng.nodes.new('GeometryNodeInputNamedAttribute'); rad.data_type = 'FLOAT'; rad.inputs['Name'].default_value = "rad"
    sm = ng.nodes.new('GeometryNodeSetMaterial'); sm.inputs['Material'].default_value = mat
    L = ng.links
    L.new(gi.outputs[0], m2p.inputs['Mesh'])
    L.new(sph.outputs['Mesh'], ss_.inputs['Geometry'])
    L.new(ss_.outputs[0], sm.inputs['Geometry'])
    L.new(m2p.outputs[0], inst.inputs['Points'])
    L.new(sm.outputs[0], inst.inputs['Instance'])
    L.new(rad.outputs['Attribute'], inst.inputs['Scale'])
    L.new(inst.outputs[0], go.inputs[0])
    return ng


atom_objs = {}
for e, mat in EL.items():
    idx = [k for k, a in enumerate(atoms) if a[5] == e]
    if not idx:
        continue
    pm = bpy.data.meshes.new("Atoms" + e)
    pm.from_pydata([(0, 0, 0)] * len(idx), [], [])
    at = pm.attributes.new("rad", 'FLOAT', 'POINT')
    at.data.foreach_set("value", np.array([atoms[k][4] * S for k in idx], np.float32))
    o = bpy.data.objects.new("Atoms" + e, pm)
    sb.link_obj(o)
    md = o.modifiers.new("Sph", 'NODES'); md.node_group = sphere_gn("Sph" + e, mat)
    atom_objs[e] = (o, np.array(idx))

# ligand (amber): a small bound cofactor-like cluster at the strands' C-terminal edge, appears as the pocket forms
rng = np.random.default_rng(21)
x_edge = P_fold[SS.index('E'):][:, 0].max() if 'E' in SS else 10
lig_c = np.array([x_edge + 2.5, 0.8, 5.0])
lig = [lig_c]
for k in range(22):
    d = rng.normal(0, 1, 3); d[1] *= 2.2; d /= np.linalg.norm(d)
    lig.append(lig[rng.integers(0, len(lig))] + d * 1.45)
lig = np.array(lig)
lm = bpy.data.meshes.new("Ligand")
lm.from_pydata([tuple(p * S) for p in lig], [], [])
la = lm.attributes.new("rad", 'FLOAT', 'POINT')
la.data.foreach_set("value", np.full(len(lig), 1.05 * S, np.float32))
ligo = bpy.data.objects.new("Ligand", lm)
sb.link_obj(ligo)
ligo.modifiers.new("Sph", 'NODES').node_group = sphere_gn("SphLig", M_LIG)


def update(scene, *args):
    f = scene.frame_current + scene.frame_subframe
    P = chain(f)
    Vv = RB.verts(P, MOL.guides(P, SS))
    me.vertices.foreach_set("co", Vv.astype(np.float32).ravel())
    me.update()
    A = MOL.side_positions(P, SS, atoms)
    for e, (o, idx) in atom_objs.items():
        o.data.vertices.foreach_set("co", A[idx].astype(np.float32).ravel())
        o.data.update()
    k = sb.smoother((f - 318) / 18.0)
    ligo.scale = (k, k, k) if k > 0.001 else (0.001,) * 3
    ligo.location = V(tuple(lig_c * S)) * (1 - k)


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(update)

# ------------------------------------------------------------------ context: two more copies far behind (scale, depth)
for i, (loc, rot) in enumerate((((0.9, 1.6, 1.1), (0.6, 0.3, 1.2)), ((-1.3, 2.2, -0.6), (2.1, -0.4, 0.3)))):
    for src in [rib] + [o for o, _ in atom_objs.values()]:
        d = src.copy()
        d.data = src.data.copy()
        if src is rib:
            d.data.vertices.foreach_set("co", RB.verts(P_fold, G_fold).astype(np.float32).ravel())
        else:
            idx = next(ix for o_, ix in atom_objs.values() if o_ is src)
            d.data.vertices.foreach_set("co", MOL.side_positions(P_fold, SS, atoms)[idx].astype(np.float32).ravel())
        sb.link_obj(d)
        d.location = loc
        d.rotation_euler = rot
        d.name = src.name + "_bg%d" % i


def freeze_bg(scene, *a):
    pass


# ------------------------------------------------------------------ light & world
sb.world_gradient(top=(0.018, 0.022, 0.030), horizon=(0.035, 0.040, 0.050), bottom=(0.010, 0.011, 0.014), strength=1.0)
cen = V(tuple(P_fold.mean(0) * S))
sb.light('AREA', "Key", loc=cen + V((-0.9, -0.5, 1.2)), target=cen, energy=28, color=(1.0, 0.97, 0.93), size=1.4)
sb.light('AREA', "Rim", loc=cen + V((0.9, 1.1, 0.4)), target=cen, energy=22, color=(0.75, 0.85, 1.0), size=0.8)
sb.light('AREA', "Under", loc=cen + V((0.2, -0.3, -1.2)), target=cen, energy=8, color=(0.85, 0.9, 1.0), size=1.5)
sc.eevee.fast_gi_distance = 0.12

# ------------------------------------------------------------------ camera: through the settling structure -> face-on sheet
sheet_c = V(tuple(P_fold[np.array(SS) == 'E'].mean(0) * S))
cam = sb.camera("Cam", loc=(0, -1, 0), target=cen, lens=35, fstop=2.8, clip=(0.005, 50))
# establishing: the whole loose coil collapsing -> push in -> slip between the helix layer and the sheet ->
# bank under the sheet and settle face-on with the strands vertical (match-cut to s05's bond fingers)
Pp = [cen + V((-0.26, -0.90, 0.34)), cen + V((-0.14, -0.50, 0.18)), sheet_c + V((-0.06, -0.24, 0.05)),
      sheet_c + V((-0.03, -0.22, -0.16)), sheet_c + V((-0.02, -0.03, -0.46))]
Tp = [cen, cen + V((0.0, 0.0, 0.02)), sheet_c + V((0.03, 0.02, 0.03)), sheet_c + V((0.01, 0.0, 0.0)), sheet_c]
KT = [0.0, 0.33, 0.58, 0.80, 1.0]


def spline_t(pts, t):
    # map t through key times onto the Catmull-Rom parameter
    for i in range(len(KT) - 1):
        if t <= KT[i + 1]:
            u_ = (i + (t - KT[i]) / (KT[i + 1] - KT[i])) / (len(KT) - 1)
            return sb.catmull(pts, u_)
    return sb.catmull(pts, 1.0)


def ease(t):
    return sb.smoother(t) * 0.6 + t * 0.4


def cpos(t):
    return spline_t(Pp, ease(t))


def ctgt(t):
    return spline_t(Tp, ease(t))


def up(t):
    k = sb.smoother(sb.remap(t, 0.55, 0.95))
    return V((0, 0, 1)).slerp(V((1, 0, 0)), k) if k < 1 else V((1, 0, 0))


def look_up(p, tg, u_):
    z = (p - tg).normalized()
    x = u_.cross(z).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed().to_quaternion()


Aall = np.vstack([P_fold * S, MOL.side_positions(P_fold, SS, atoms)])
for f in range(F0, F1 + 1):
    t = (f - F0) / (F1 - F0)
    p, tg = cpos(t), ctgt(t)
    cam.location = p
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = look_up(p, tg, up(t))
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_quaternion", frame=f)
    cam.data.lens = sb.lerp(35, 45, sb.smooth(t)); cam.data.keyframe_insert("lens", frame=f)
    cam.data.dof.focus_distance = (p - tg).length; cam.data.dof.keyframe_insert("focus_distance", frame=f)
    cam.data.dof.aperture_fstop = sb.lerp(4.0, 2.8, sb.smooth(t)); cam.data.dof.keyframe_insert("aperture_fstop", frame=f)
    if os.environ.get("SB_CHECK") and f % 5 == 0:
        P_ = chain(f) * S
        A_ = np.vstack([P_, MOL.side_positions(chain(f), SS, atoms)])
        dmin = np.min(np.linalg.norm(A_ - np.array(p), axis=1))
        print("CLEAR", f, round(dmin * 100, 2), "A")

dbgcam.apply()
sb.frames(F0, F1)
update(sc)
sb.save("s04")
sb.render_shot(sid)
