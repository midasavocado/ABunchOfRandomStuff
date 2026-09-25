"""Load an SDF hand (.npz from sdfhand.py) into Blender with nails and an attribute-driven skin shader."""
import bpy, json, math, os
import numpy as np
from mathutils import Vector as V, Matrix
import sb

TONE = (0.52, 0.34, 0.24)   # same child as the eye


def hand_material(name="ChildHandSkin", tone=TONE):
    m = sb.mat(name, tone, rough=0.45, sss=1.0, sss_radius=(1.0, 0.4, 0.22), sss_scale=0.0025, spec=0.5, sheen=0.2)
    b = m.node_tree.nodes["Principled BSDF"]
    b.subsurface_method = 'BURLEY'
    nb = sb.NB(m)
    co = nb.coord('Object')
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "hand"
    sep = nb.new('ShaderNodeSeparateColor'); nb.link(at.outputs['Color'], sep.inputs[0])
    crease, palm, phase = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    # colour: palms paler/pinker, knuckles slightly darker & rosier, subtle mottling
    mott = nb.noise(co, scale=300, detail=5, rough=0.6)
    base = nb.mix(nb.maprange(mott.outputs['Fac'], 0.35, 0.65), (tone[0] * 0.94, tone[1] * 0.9, tone[2] * 0.88, 1), (tone[0] * 1.04, tone[1] * 1.03, tone[2] * 1.02, 1))
    base = nb.mix(nb.math('MULTIPLY', palm, 0.55), base, (tone[0] * 1.18, tone[1] * 0.98, tone[2] * 0.92, 1))
    base = nb.mix(nb.math('MULTIPLY', crease, 0.45), base, (tone[0] * 0.92, tone[1] * 0.70, tone[2] * 0.66, 1))
    nb.set('Base Color', base)
    # knuckle wrinkles: arcs across the finger at dorsal joints
    wr = nb.math('SINE', nb.math('MULTIPLY', phase, 2 * math.pi / 0.85))
    wn = nb.noise(co, scale=900, detail=2)
    wrm = nb.math('MULTIPLY', nb.math('MULTIPLY', nb.maprange(wr, 0.2, 1.0), crease),
                  nb.maprange(wn.outputs['Fac'], 0.3, 0.7, 0.4, 1.0))
    pore = nb.voronoi(co, scale=1800, feature='F1')
    pd = nb.math('POWER', nb.maprange(pore.outputs['Distance'], 0.0, 0.45, 1.0, 0.0), 3.0)
    fine = nb.noise(co, scale=2600, detail=3)
    # palm: fine ridged skin (dermatoglyphic texture)
    ridge = nb.wave(nb.mapping(co, rot=(0.0, 0.0, 0.3)), scale=2200, dist=4.0, detail=2)
    h = nb.math('ADD', nb.math('MULTIPLY', fine.outputs['Fac'], 0.3),
                nb.math('MULTIPLY', nb.math('MULTIPLY', ridge.outputs['Fac'], palm), 0.25))
    h = nb.math('SUBTRACT', h, nb.math('ADD', nb.math('MULTIPLY', pd, 0.35), nb.math('MULTIPLY', wrm, 0.9)))
    nb.set('Normal', nb.bump(h, strength=0.5, distance=0.0002))
    nb.set('Roughness', nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.38, 0.52))
    return m


def nail_material(name="Nail"):
    m = sb.mat(name, (0.78, 0.55, 0.50), rough=0.18, sss=0.8, sss_radius=(1, 0.5, 0.4), sss_scale=0.001, coat=0.7, coat_rough=0.1)
    nb = sb.NB(m)
    co = nb.coord('Object')
    # lunula (pale base) and free edge
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    t = sep.outputs[1]
    col = nb.ramp(nb.maprange(t, -1.0, 1.0), [(0.0, (0.86, 0.74, 0.70)), (0.18, (0.72, 0.48, 0.44)), (0.85, (0.70, 0.47, 0.42)), (1.0, (0.90, 0.84, 0.78))])
    nb.set('Base Color', col)
    ridges = nb.wave(co, scale=40, wtype='BANDS', direction='X')
    nb.set('Normal', nb.bump(ridges.outputs['Fac'], strength=0.05, distance=0.0001))
    return m


def nail_mesh(name, w, L, mat):
    """Curved rectangular nail plate in local space: across X, along Y (-1..1 normalised in object space), up Z."""
    nx, ny = 12, 10
    verts, faces = [], []
    for j in range(ny + 1):
        v = j / ny
        for i in range(nx + 1):
            u = i / nx
            x = (u - 0.5) * w
            # rounded free edge at the tip, square-ish at the cuticle
            y = (v - 0.5) * L
            curve = -((x / (w * 0.5)) ** 2) * w * 0.22
            if v > 0.8:
                shrink = 1 - ((v - 0.8) / 0.2) ** 2 * 0.25
                x *= shrink
            verts.append((x, y, curve))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    o = sb.mesh_obj(name, verts, faces, mat)
    sol = sb.solidify(o, 0.0004, offset=-1)
    sb.subsurf(o, 1, 2)
    return o


def load(npz_path, name="Hand", skin=None, nails=True):
    D = np.load(npz_path)
    Vv, F, A = D["V"], D["F"], D["A"]
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(Vv))
    me.vertices.foreach_set("co", Vv.ravel())
    me.loops.add(len(F) * 3)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, len(F) * 3, 3))
    me.polygons.foreach_set("loop_total", np.full(len(F), 3))
    # marching cubes faces are wound for outward normals with our axis order? flip if needed below
    me.update(calc_edges=True)
    me.validate()
    attr = me.color_attributes.new("hand", 'FLOAT_COLOR', 'POINT')
    cols = np.concatenate([A, np.ones((len(A), 1), np.float32)], 1)
    attr.data.foreach_set("color", cols.ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    sb.recalc_normals(o)
    o.data.materials.append(skin or hand_material())
    parts = [o]
    if nails:
        nm = nail_material()
        for n in json.loads(str(D["N"])):
            if n["name"].startswith("__"):
                continue
            c, d, up = V(n["c"]), V(n["d"]), V(n["up"])
            side = d.cross(up).normalized()
            up2 = side.cross(d).normalized()
            mw = Matrix((side, d, up2)).transposed().to_4x4()
            mw.translation = c
            nl = nail_mesh(name + "Nail_" + n["name"], n["w"], n["L"], nm)
            nl.matrix_world = mw
            nl.parent = o
            nl.matrix_parent_inverse = Matrix.Identity(4)
            nl.matrix_basis = mw
            parts.append(nl)
    return o, parts
