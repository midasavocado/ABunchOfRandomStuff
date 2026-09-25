"""Load the SDF child (sdfchild.py npz) into Blender with continuity materials."""
import bpy, json, math
import numpy as np
from mathutils import Vector as V, Matrix
import sb

SKIN = (0.52, 0.34, 0.24)


def mesh_from_arrays(name, Vv, F, mat=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(Vv))
    me.vertices.foreach_set("co", Vv.ravel())
    me.loops.add(len(F) * 3)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, len(F) * 3, 3))
    me.polygons.foreach_set("loop_total", np.full(len(F), 3))
    me.update(calc_edges=True)
    me.validate()
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    sb.recalc_normals(o)
    if mat:
        o.data.materials.append(mat)
    return o


def hoodie_mat(name="Hoodie", color=(0.060, 0.064, 0.072)):
    """Graphite cotton fleece: soft sheen, fine knit, pilling, slight colour variation."""
    m = sb.mat(name, color, rough=0.9, sheen=0.6, spec=0.3)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs['Sheen Roughness'].default_value = 0.5
    b.inputs['Sheen Tint'].default_value = (0.7, 0.75, 0.85, 1)
    nb = sb.NB(m)
    co = nb.coord('Object')
    knit = nb.wave(nb.mapping(co, scale=(1, 1, 2.2)), scale=1100, dist=2.0, wtype='BANDS', direction='Z')
    pill = nb.noise(co, scale=600, detail=4, rough=0.7)
    mott = nb.noise(co, scale=30, detail=3)
    nb.set('Base Color', nb.mix(nb.maprange(mott.outputs['Fac'], 0.3, 0.7), (color[0] * 0.85, color[1] * 0.85, color[2] * 0.88, 1), (color[0] * 1.15, color[1] * 1.15, color[2] * 1.12, 1)))
    h = nb.math('ADD', nb.math('MULTIPLY', knit.outputs['Fac'], 0.5), nb.math('MULTIPLY', pill.outputs['Fac'], 0.5))
    nb.set('Normal', nb.bump(h, strength=0.35, distance=0.0006))
    return m


def cuff_mat(name="AmberCuff"):
    """Amber rib-knit cuff (identity marker)."""
    col = (0.62, 0.24, 0.035)
    m = sb.mat(name, col, rough=0.85, sheen=0.7, spec=0.3)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs['Sheen Tint'].default_value = (1.0, 0.7, 0.4, 1)
    nb = sb.NB(m)
    co = nb.coord('Object')
    loops = nb.wave(co, scale=2600, dist=1.5, wtype='RINGS', direction='Z')
    fz = nb.noise(co, scale=900, detail=4)
    nb.set('Normal', nb.bump(nb.math('ADD', loops.outputs['Fac'], nb.math('MULTIPLY', fz.outputs['Fac'], 0.4)), strength=0.4, distance=0.0004))
    nb.set('Base Color', nb.mix(nb.maprange(fz.outputs['Fac'], 0.3, 0.7), (col[0] * 0.8, col[1] * 0.78, col[2] * 0.7, 1), (col[0] * 1.1, col[1] * 1.08, col[2], 1)))
    return m


def pants_mat(name="Pants", color=(0.035, 0.045, 0.075)):
    m = sb.fabric(name, color, rough=0.8, weave=900, sheen=0.3)
    return m


def shoe_mat(name="Canvas"):
    return sb.fabric(name, (0.72, 0.72, 0.70), rough=0.8, weave=700, sheen=0.2)


def child_skin(name="ChildBodySkin"):
    m = sb.skin(name, SKIN, rough=0.42, pores=0.7, scale=0.8)
    m.node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
    return m


def load(npz, prefix="Child", hood_up=True, parts=None):
    D = np.load(npz)
    meta = json.loads(str(D["meta"]))
    mats = dict(skin=child_skin(), hoodie=hoodie_mat(), hood=hoodie_mat("Hood"), cuffs=cuff_mat(),
                pants=pants_mat(), shoes=shoe_mat())
    objs = {}
    for part in ("skin", "hoodie", "hood", "cuffs", "pants", "shoes"):
        if parts and part not in parts:
            continue
        if part == "hood" and not hood_up:
            continue
        k = part + "_V"
        if k not in D.files or len(D[k]) == 0:
            continue
        objs[part] = mesh_from_arrays(prefix + part.capitalize(), D[k], D[part + "_F"], mats[part])
    return objs, meta
