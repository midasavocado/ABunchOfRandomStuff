"""People-collision scan (SB_COLLIDE=1 in sb.render_shot): at sampled frames, every skinned (armature-deformed) body
mesh is tested against every other body and against nearby props with BVH overlap. Ground-like objects (huge,
flat), a person's own parts (eyes, hair, clothes proxies share the rig) and whatever they sit on (seat names) are
ignored. Prints `COLLIDE <sid> f<frame> <a> x <b> <n pairs> depth~<m>` lines; a handful of pairs is skin contact,
dozens is an arm through a table or two walkers inside each other."""
import bpy
from mathutils import Vector as V
from mathutils.bvhtree import BVHTree
import timeline as TL

IGNORE = ("Ground", "Floor", "Terrain", "Road", "Street", "Sidewalk", "Grass", "Verge", "Pad", "Cap", "Haze", "Sky",
          "Seat", "Chair", "Bench", "Stool", "Sofa", "Lawn", "Field", "Plane", "Deck", "Pavement", "Rug", "Carpet")


def _rig_of(o):
    for m in o.modifiers:
        if m.type == 'ARMATURE' and m.object:
            return m.object
    return o.parent if o.parent and o.parent.type == 'ARMATURE' else None


def _bvh(o, dg):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    M = o.matrix_world
    verts = [M @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    if not verts or not polys:
        return None, None
    lo = V((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    hi = V((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    return BVHTree.FromPolygons(verts, polys, epsilon=0.0), (lo, hi)


def _near(a, b, pad=0.05):
    return all(a[0][i] - pad <= b[1][i] and b[0][i] - pad <= a[1][i] for i in range(3))


def scan(sid, step=6):
    sc = bpy.context.scene
    _, a, b, _ = TL.shot(sid)
    bodies = {}
    for o in sc.objects:
        if o.type != 'MESH' or o.hide_render:
            continue
        r = _rig_of(o)
        if r is not None and any(k in o.name.lower() for k in ("body", "base", "proxy", "human", "skin")) or \
                (r is not None and len(o.data.vertices) > 3000):
            bodies.setdefault(r.name, []).append(o)
    props = [o for o in sc.objects if o.type == 'MESH' and not o.hide_render and _rig_of(o) is None
             and not any(k.lower() in o.name.lower() for k in IGNORE)
             and max(o.dimensions) < 12.0 and len(o.data.polygons) > 0]
    out = []
    for f in range(a, b, step):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        B = {}
        for rn, objs in bodies.items():
            main = max(objs, key=lambda o: len(o.data.vertices))
            t, bb = _bvh(main, dg)
            if t:
                B[rn] = (t, bb, main.name)
        names = list(B)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ta, ba, na = B[names[i]]; tb, bb_, nb_ = B[names[j]]
                if _near(ba, bb_):
                    n = len(ta.overlap(tb))
                    if n > 6:
                        out.append((f, na, nb_, n))
        for rn, (t, bb, na) in B.items():
            for p in props:
                pb = [p.matrix_world @ V(c) for c in p.bound_box]
                pbb = (V((min(c.x for c in pb), min(c.y for c in pb), min(c.z for c in pb))),
                       V((max(c.x for c in pb), max(c.y for c in pb), max(c.z for c in pb))))
                if not _near(bb, pbb):
                    continue
                tp, _ = _bvh(p, dg)
                if tp is None:
                    continue
                n = len(t.overlap(tp))
                if n > 6:
                    out.append((f, na, p.name, n))
    for f, x, y, n in out:
        print("COLLIDE %s f%d %s x %s %d" % (sid, f, x, y, n))
    print("COLLIDE %s done: %d hits" % (sid, len(out)))
    return out
