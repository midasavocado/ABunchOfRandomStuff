"""Procedural posable hand (skin-modifier skeleton + subdivision), child proportions.
Local space: wrist at origin, fingers along +Y, palm facing -Z (back of hand +Z), thumb toward +X (right hand)."""
import bpy, math
from mathutils import Vector as V, Matrix, Euler, Quaternion
import sb

# finger definitions: base (metacarpal head position), segment lengths, radii at joints, splay angle
S = 0.78   # child scale relative to adult
FINGERS = {
    #          mcp position (x, y, z)          phalanx lengths            joint radii (mcp, pip, dip, tip)
    "index":  ((0.024, 0.078, 0.004),  (0.040, 0.024, 0.018), (0.0098, 0.0088, 0.0077, 0.0068)),
    "middle": ((0.004, 0.082, 0.006),  (0.045, 0.028, 0.019), (0.0102, 0.0090, 0.0078, 0.0068)),
    "ring":   ((-0.015, 0.078, 0.004), (0.041, 0.026, 0.018), (0.0096, 0.0085, 0.0074, 0.0064)),
    "pinky":  ((-0.031, 0.069, 0.000), (0.032, 0.019, 0.016), (0.0085, 0.0075, 0.0066, 0.0058)),
}
THUMB = ((0.020, 0.012, -0.004), (0.036, 0.030, 0.024), (0.0135, 0.0112, 0.0095, 0.0080))


def _chain(base, direction, up, lengths, curls, splay=0.0, twist=0.0):
    """Returns joint positions for a finger chain. curls in radians per joint (flexion toward palm, -Z)."""
    pts = [V(base)]
    d = V(direction).normalized()
    upv = V(up).normalized()
    side = d.cross(upv).normalized()
    q = Quaternion(upv, splay)
    d = q @ d
    side = q @ side
    for L, c in zip(lengths, curls):
        # flex around the side axis (positive curl bends toward palm = -up)
        r = Quaternion(side, -c)
        d = r @ d
        upv = r @ upv
        pts.append(pts[-1] + d * L * S)
    return pts


def pose_points(pose):
    """pose: dict finger -> (curl_mcp, curl_pip, curl_dip, splay). thumb -> (cmc_abd, curl1, curl2, curl3, opp).
    Returns list of (points, radii) chains plus palm points."""
    chains = []
    for name, (base, lens, rad) in FINGERS.items():
        c = pose.get(name, (0.1, 0.15, 0.1, 0.0))
        b = V(base) * S
        pts = _chain(b, (0, 1, 0), (0, 0, 1), lens, c[:3], splay=c[3])
        chains.append((name, [V((b.x * 0.9, 0.012, b.z * 0.5 - 0.002))] + pts, [rad[0] * 1.15 * S] + [r * S for r in rad]))
    t = pose.get("thumb", (0.6, 0.2, 0.2, 0.2, 0.5))
    tb = V(THUMB[0]) * S
    tdir = Quaternion((0, 1, 0), -t[4]) @ (Quaternion((0, 0, 1), -t[0]) @ V((0.55, 1.0, -0.35)))
    tup = Quaternion((0, 1, 0), -t[4]) @ V((0.3, -0.2, 1)).normalized()
    pts = _chain(tb, tdir, tup, THUMB[1], t[1:4])
    chains.append(("thumb", [V((0.004, 0.0, -0.002))] + pts, [0.014 * S] + [r * S for r in THUMB[2]]))
    return chains


def build(name="Hand", pose=None, skin_mat=None, nail_mat=None, poses_anim=None, f0=0):
    """Builds the hand. If poses_anim (list of pose dicts per frame from f0) is given, the skeleton is animated
    via absolute shape keys so the skin re-generates each frame."""
    pose = pose or {}
    chains = pose_points(pose)
    verts, edges, radii = [], [], []
    wrist = len(verts); verts.append(V((0, -0.022, 0.0))); radii.append((0.024 * S, 0.014 * S))
    palm_c = len(verts); verts.append(V((0.0, 0.03 * S, 0.004))); radii.append((0.030 * S, 0.012 * S))
    edges.append((wrist, palm_c))
    tips = []
    for cname, pts, rads in chains:
        prev = palm_c if cname != "thumb" else wrist
        for i, (p, r) in enumerate(zip(pts, rads)):
            vi = len(verts)
            verts.append(p)
            flat = 0.82 if cname != "thumb" else 0.9
            radii.append((r, r * flat))
            edges.append((prev, vi))
            prev = vi
        tips.append((cname, prev))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], edges, [])
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    sk = o.modifiers.new("Skin", 'SKIN')
    sk.use_smooth_shade = True
    sk.branch_smoothing = 0.6
    for i, r in enumerate(radii):
        o.data.skin_vertices[0].data[i].radius = r
    o.data.skin_vertices[0].data[wrist].use_root = True
    sb.subsurf(o, 2, 3)
    if skin_mat:
        o.data.materials.append(skin_mat)
    # nails: small curved plates at each distal phalanx, parented via vertex-hook-like approach (rebuilt per pose)
    if poses_anim:
        o.shape_key_add(name="Basis")
        key = o.data.shape_keys
        key.use_relative = False
        for k, ps in enumerate(poses_anim):
            chains_k = pose_points(ps)
            sk_ = o.shape_key_add(name=f"p{k}")
            idx = 2
            for cname, pts, rads in chains_k:
                for p in pts:
                    sk_.data[idx].co = p
                    idx += 1
        # eval_time runs through the keys in order (absolute keys are spaced 10 apart)
        for k in range(len(poses_anim)):
            key.eval_time = (k + 1) * 10
            key.keyframe_insert("eval_time", frame=f0 + k)
    return o


def nail_positions(pose):
    """(position, direction, up) at each distal segment, for placing nails."""
    out = []
    for cname, pts, rads in pose_points(pose):
        a, b = pts[-2], pts[-1]
        d = (b - a).normalized()
        out.append((cname, a.lerp(b, 0.62), d, rads[-1]))
    return out


def hand_skin(name="HandSkin", tone=(0.52, 0.34, 0.24)):
    m = sb.skin(name, tone, rough=0.45, pores=0.6, scale=0.6)
    m.node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
    nb = sb.NB(m)
    co = nb.coord('Object')
    # knuckle creases and palm lines (fine) - subtle
    w = nb.wave(nb.mapping(co, scale=(1, 1, 1)), scale=320, dist=3.0, detail=2, wtype='BANDS', direction='Y')
    b = m.node_tree.nodes["Principled BSDF"]
    old = b.inputs['Normal'].links[0].from_node
    b2 = nb.new('ShaderNodeBump', Strength=0.12, Distance=0.0003)
    nb.link(w.outputs['Fac'], b2.inputs['Height'])
    nb.link(old.outputs[0], b2.inputs['Normal'])
    nb.link(b2.outputs[0], b.inputs['Normal'])
    return m
