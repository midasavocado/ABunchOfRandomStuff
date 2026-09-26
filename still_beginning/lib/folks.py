"""People for the making sequence (s06 researcher, s08 engineer, s09 prosthesis wearer + family), built with MPFB
through lib/mhchild.py (docs/PEOPLE.md). Presets + posing helpers (sitting, arm reach) + garment recolouring."""
import bpy, math
from mathutils import Vector as V, Euler, Matrix, Quaternion
import sb, mhchild

PRESETS = {
    # s06: design researcher, seen over the shoulder (back of head, shoulder, arm)
    "researcher": dict(phenotype=dict(age=0.56, gender=0.0, weight=0.45, muscle=0.45, height=0.5,
                                      race=dict(african=0.15, asian=0.6, caucasian=0.25)),
                       hair="ponytail01", clothes=["male_casualsuit01", "shoes02"], skin="young_asian_female",
                       tint={"casualsuit": (0.26, 0.29, 0.33)}),
    # s08: engineer at the bench (hands + forearms mostly, body soft)
    "engineer": dict(phenotype=dict(age=0.62, gender=1.0, weight=0.5, muscle=0.55,
                                    race=dict(african=0.6, asian=0.1, caucasian=0.3)),
                     hair="short02", clothes=["male_worksuit01", "shoes01"], skin="middleage_african_male", iris="darkbrown",
                     hair_color=(0.02, 0.018, 0.016),
                     tint={"worksuit": (0.20, 0.22, 0.25)}),
    # s09: the prosthesis wearer (adult woman, 40s) and family
    "wearer": dict(phenotype=dict(age=0.66, gender=0.0, weight=0.5, muscle=0.5,
                                  race=dict(african=0.3, asian=0.1, caucasian=0.6)),
                   hair="ponytail01", clothes=["male_casualsuit01", "shoes02"], skin="middleage_caucasian_female", iris="hazel",
                   hair_color=(0.16, 0.09, 0.05),
                   tint={"casualsuit": (0.70, 0.64, 0.55)}),
    "partner": dict(phenotype=dict(age=0.68, gender=1.0, weight=0.55, muscle=0.5,
                                   race=dict(african=0.25, asian=0.35, caucasian=0.4)),
                    hair="short04", clothes=["male_casualsuit03", "shoes03"], skin="middleage_asian_male",
                    hair_color=(0.03, 0.025, 0.022),
                    tint={"casualsuit": (0.20, 0.30, 0.38)}),
    "grandma": dict(phenotype=dict(age=0.92, gender=0.0, weight=0.55, muscle=0.35,
                                   race=dict(african=0.2, asian=0.1, caucasian=0.7)),
                    hair="short03", clothes=["male_casualsuit03", "shoes02"], skin="old_caucasian_female", iris="blue",
                    hair_color=(0.62, 0.60, 0.58),
                    tint={"casualsuit": (0.50, 0.40, 0.34)}),
    "teen": dict(phenotype=dict(age=0.27, gender=1.0, weight=0.45, muscle=0.45,
                                race=dict(african=0.35, asian=0.2, caucasian=0.45)),
                 hair="short01", clothes=["male_casualsuit01", "shoes05"], skin="young_caucasian_male", iris="green",
                 hair_color=(0.10, 0.06, 0.035),
                 tint={"casualsuit": (0.55, 0.60, 0.52)}),
}


def tint_clothes(parts, key, color):
    """Desaturate + multiply a garment's albedo (removes any print colour, keeps weave/fold detail)."""
    for o in parts:
        if key not in o.name.lower():
            continue
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes:
                continue
            nt = m.node_tree
            for n in list(nt.nodes):
                if n.type != 'BSDF_PRINCIPLED':
                    continue
                inp = n.inputs['Base Color']
                src = inp.links[0].from_socket if inp.links else None
                hsv = nt.nodes.new('ShaderNodeHueSaturation')
                hsv.inputs['Saturation'].default_value = 0.0
                mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
                mul.inputs[0].default_value = 1.0
                if src:
                    nt.links.new(src, hsv.inputs['Color'])
                else:
                    hsv.inputs['Color'].default_value = inp.default_value
                nt.links.new(hsv.outputs[0], mul.inputs[6])
                mul.inputs[7].default_value = (color[0] * 2.2, color[1] * 2.2, color[2] * 2.2, 1)
                nt.links.new(mul.outputs[2], inp)


def person(preset, name=None, subdiv=1):
    p = PRESETS[preset]
    bm, rig, parts = mhchild.build(name=name or preset.capitalize(), hair=p["hair"], clothes=p["clothes"],
                                   skin=p["skin"], phenotype=p["phenotype"], subdiv=subdiv, iris=p.get("iris", "brown"))
    mhchild.fix_eyes(parts, "brownlight_eye.png") if False else None
    for k, c in p.get("tint", {}).items():
        tint_clothes(parts, k, c)
    if p.get("hair_color"):
        hair_color(parts, p["hair_color"])
    return bm, rig, parts


def pose(rig, d):
    """d: {bone: (x, y, z) degrees}"""
    for b, r in d.items():
        mhchild.pose_bone(rig, b, rot=r)


def sit(rig, knee=90.0, hip=85.0):
    pose(rig, {"upperleg01.L": (-hip, 0, 0), "upperleg01.R": (-hip, 0, 0),
               "lowerleg01.L": (knee, 0, 0), "lowerleg01.R": (knee, 0, 0)})


def place(rig, loc=(0, 0, 0), rot_z=0.0):
    rig.location = loc
    rig.rotation_euler = (0, 0, math.radians(rot_z))


def bone_world(rig, bone, tail=False):
    bpy.context.view_layer.update()
    pb = rig.pose.bones[bone]
    return rig.matrix_world @ (pb.tail if tail else pb.head)


def reach(rig, side, target, elbow_hint=None, iters=40, bones=None, end="wrist", tol=0.004):
    """Damped-least-squares IK on Euler angles of (upperarm01, lowerarm01) [+ shoulder01 slightly] so the head of
    wrist.<side> reaches `target` (world). elbow_hint: world point the elbow should drift toward (soft)."""
    import numpy as np
    bones = bones or [("upperarm01.%s" % side, (0, 1, 2)), ("lowerarm01.%s" % side, (0, 2))]
    vars_ = [(b, ax) for b, axes in bones for ax in axes]
    for b, _ in bones:
        rig.pose.bones[b].rotation_mode = 'XYZ'
    x = np.array([rig.pose.bones[b].rotation_euler[ax] for b, ax in vars_], float)
    tgt = np.array(target, float)
    if elbow_hint is None and end == "wrist":
        # natural elbow: below the shoulder-target line, a little out from the body and back (arms hang, not flare)
        sh = bone_world(rig, "upperarm01.%s" % side)
        spine = bone_world(rig, "spine03")
        out = V((sh.x - spine.x, sh.y - spine.y, 0.0))
        out = out.normalized() if out.length > 1e-6 else V((0, 0, 0))
        mid = (sh + V(target)) / 2
        elbow_hint = mid + V((0, 0, -0.14)) + out * 0.06

    def apply(xv):
        for (b, ax), v in zip(vars_, xv):
            rig.pose.bones[b].rotation_euler[ax] = v

    def f(xv):
        apply(xv)
        p = np.array(bone_world(rig, "%s.%s" % (end, side)))
        r = p - tgt
        if elbow_hint is not None:
            e = np.array(bone_world(rig, "lowerarm01.%s" % side))
            r = np.concatenate([r, 0.15 * (e - np.array(elbow_hint))])
        return r
    lam = 0.02
    for it in range(iters):
        r = f(x)
        if np.linalg.norm(r[:3]) < tol:
            break
        J = np.zeros((len(r), len(x)))
        for i in range(len(x)):
            d = np.zeros(len(x)); d[i] = 0.01
            J[:, i] = (f(x + d) - r) / 0.01
        dx = -np.linalg.solve(J.T @ J + lam * np.eye(len(x)), J.T @ r)
        x = x + np.clip(dx, -0.3, 0.3)
    apply(x)
    err = float(np.linalg.norm(f(x)[:3]))
    if err > 0.03:
        print("REACH %s.%s short by %.0f mm (target out of reach?)" % (rig.name, side, err * 1000))
    return err


def expression(rig, smile=0.0, brows=0.0, eyes=0.0, jaw=0.0, frame=None):
    """Facial expression on MakeHuman's sculpted expression units (lib/soul.py): smile 0..1 is a real Duchenne
    smile (zygomaticus corner pull, cheeks lifting the lower lids, a little nose wrinkle and parted lips as it grows),
    brows -1..1 lowers/raises (inner brows lead), eyes 0..1 narrows, jaw 0..1 opens the mouth. frame -> keyed."""
    import soul
    meshes = [o for o in rig.children_recursive if o.type == 'MESH']
    s = max(0.0, smile)
    w = {"mouth-corner-puller": 0.9 * s,
         "mouth-upward-retraction": 0.35 * max(0.0, s - 0.45),
         "mouth-parling": 0.35 * max(0.0, s - 0.55) + 0.4 * jaw,
         "mouth-open": 0.55 * jaw,
         "eye-*-slit": min(1.0, 0.32 * s + 0.5 * eyes),
         "nose-*-elevation": 0.25 * max(0.0, s - 0.5),
         "eyebrows-*-up": 0.4 * max(0.0, brows),
         "eyebrows-*-inner-up": 0.25 * max(0.0, brows) + 0.08 * s,
         "eyebrows-*-down": 0.4 * max(0.0, -brows)}
    soul.set_units(meshes, w, frame)
    # the old bone-driven face stays neutral (it tore the lip corners at strong smiles)
    for b in ("oris04.L", "oris04.R", "oris03.L", "oris03.R"):
        pb = rig.pose.bones.get(b)
        if pb:
            pb.location = (0, 0, 0)


def hair_color(parts, color, spec=0.25):
    """Re-colour MPFB card hair (keeps the strand texture/alpha): desaturate the diffuse and multiply by a natural
    colour; adds a soft anisotropic sheen so it reads as hair, not plastic."""
    for o in parts:
        if not any(k in o.name.lower() for k in ("afro", "bob", "braid", "long", "ponytail", "short", "eyebrow")):
            continue
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes:
                continue
            nt = m.node_tree
            for n in list(nt.nodes):
                if n.type != 'BSDF_PRINCIPLED':
                    continue
                inp = n.inputs['Base Color']
                src = inp.links[0].from_socket if inp.links else None
                hsv = nt.nodes.new('ShaderNodeHueSaturation')
                hsv.inputs['Saturation'].default_value = 0.0
                mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
                mul.inputs[0].default_value = 1.0
                if src:
                    nt.links.new(src, hsv.inputs['Color'])
                else:
                    hsv.inputs['Color'].default_value = inp.default_value
                nt.links.new(hsv.outputs[0], mul.inputs[6])
                mul.inputs[7].default_value = (color[0] * 1.6, color[1] * 1.6, color[2] * 1.6, 1)
                nt.links.new(mul.outputs[2], inp)
                n.inputs['Roughness'].default_value = 0.55
                n.inputs['Anisotropic'].default_value = 0.3
                n.inputs['Specular IOR Level'].default_value = spec
                n.inputs['Coat Weight'].default_value = 0.0


def amputate(parts, side="R", bones=("lowerarm01", "lowerarm02", "wrist", "metacarpal", "finger")):
    """Hide a forearm+hand (body and clothes) with a Mask modifier on a union vertex group of those bones' weights,
    so a prosthesis + a sleeve built separately can take its place."""
    import numpy as np
    for o in parts:
        if o.type != 'MESH':
            continue
        vgs = [g for g in o.vertex_groups if g.name.endswith("." + side) and any(g.name.startswith(b) for b in bones)]
        if not vgs:
            continue
        idx = {g.index for g in vgs}
        me = o.data
        sel = []
        for v in me.vertices:
            w = sum(ge.weight for ge in v.groups if ge.group in idx)
            if w > 0.35:
                sel.append(v.index)
        if not sel:
            continue
        g = o.vertex_groups.new(name="_amp_" + side)
        g.add(sel, 1.0, 'REPLACE')
        md = o.modifiers.new("Amputate" + side, 'MASK')
        md.vertex_group = g.name
        md.invert_vertex_group = True
        # keep it right after the armature so the cut deforms with the pose
        arm_i = next((i for i, m in enumerate(o.modifiers) if m.type == 'ARMATURE'), 0)
        with bpy.context.temp_override(object=o):
            bpy.ops.object.modifier_move_to_index(modifier=md.name, index=arm_i + 1)


def garment_material(parts, key):
    for o in parts:
        if key in o.name.lower() and o.material_slots:
            return o.material_slots[0].material
    return None


def look_at(rig, target, bones=(("neck02", 0.35), ("head", 0.65)), eyes=True, iters=25, limit=35.0):
    """Turn neck/head (and the eyes) so the gaze points at `target` (world). The gaze direction is the eye bones'
    Y axis (MPFB eye bones point out of the pupils). Small damped Gauss-Newton on (pitch x, yaw y) shared across
    the chain by weight; clamped to +-limit degrees so nobody breaks their neck."""
    import numpy as np
    tgt = V(target)

    def gaze():
        bpy.context.view_layer.update()
        e = [rig.pose.bones.get("eye." + s) for s in ("L", "R")]
        mid = sum((rig.matrix_world @ b.head for b in e), V((0, 0, 0))) / 2
        fwd = sum(((rig.matrix_world.to_3x3() @ b.matrix.to_3x3()).col[1] for b in e), V((0, 0, 0))).normalized()
        return mid, fwd

    for b, _ in bones:
        rig.pose.bones[b].rotation_mode = 'XYZ'
    x = np.zeros(2)
    base = {b: V(rig.pose.bones[b].rotation_euler) for b, _ in bones}

    def apply(xv):
        for b, w in bones:
            r = base[b].copy()
            r.x += math.radians(float(np.clip(xv[0], -limit, limit))) * w
            r.y += math.radians(float(np.clip(xv[1], -limit, limit))) * w
            rig.pose.bones[b].rotation_euler = r

    def err(xv):
        apply(xv)
        mid, fwd = gaze()
        d = (tgt - mid).normalized()
        return np.array(fwd - d)
    for _ in range(iters):
        r = err(x)
        if np.linalg.norm(r) < 1e-3:
            break
        J = np.zeros((3, 2))
        for i in range(2):
            dx = np.zeros(2); dx[i] = 1.0
            J[:, i] = (err(x + dx) - r)
        x = x - np.linalg.solve(J.T @ J + 1e-6 * np.eye(2), J.T @ r)
        x = np.clip(x, -limit, limit)
    apply(x)
    return x


def walk(rig, f0, f1, start, heading_deg, speed=1.3, stride=1.45, phase=0.0, arm=14.0, fps=24.0):
    """Procedural walk cycle keyed on the MPFB default rig: root travels along `heading` at `speed` (m/s) while the
    legs cycle with a stride matched to that speed (hip flex/extend, knee flexion peaking mid-swing, ankle roll),
    opposite arm swing, pelvis bob/rock. Good at a distance (no foot IK)."""
    h = math.radians(heading_deg)
    d = V((math.cos(h), math.sin(h), 0.0))
    rig.rotation_euler = (0, 0, h + math.pi / 2)        # MPFB faces -Y; heading 0 = +X
    z0 = V(start).z
    for f in range(f0, f1 + 1):
        t = (f - f0) / fps
        dist = speed * t
        ph = 2 * math.pi * dist / stride + phase
        rig.location = V(start) + d * dist + V((0, 0, 0.018 * math.cos(2 * ph)))
        rig.keyframe_insert("location", frame=f)
        for side, off in (("L", 0.0), ("R", math.pi)):
            p = ph + off
            hip = 21.0 * math.sin(p)
            knee = 6.0 + 52.0 * max(0.0, math.cos(p)) ** 1.6 + 8.0 * max(0.0, -math.sin(p)) * 0.5
            ank = -8.0 * math.sin(p) + 10.0 * max(0.0, math.cos(p)) ** 2
            pose(rig, {"upperleg01." + side: (-hip, 0, 0), "lowerleg01." + side: (knee, 0, 0), "foot." + side: (-ank, 0, 0)})
            sg = 1.0 if side == "L" else -1.0
            pose(rig, {"upperarm01." + side: (arm * math.sin(p + math.pi), 0, -32.0 * sg),
                       "lowerarm01." + side: (10.0 + 8.0 * max(0.0, math.sin(p)), 0, 0)})
            for b in ("upperleg01", "lowerleg01", "foot", "upperarm01", "lowerarm01"):
                pb = rig.pose.bones.get(b + "." + side)
                if pb:
                    pb.keyframe_insert("rotation_euler", frame=f)
        pose(rig, {"pelvis.L": (0, 0, 0), "spine01": (0, 3.0 * math.sin(ph), 0)})
        rig.pose.bones["spine01"].keyframe_insert("rotation_euler", frame=f)


def arms_down(rig, drop=32.0, elbow=10.0):
    """Relaxed standing arms from MPFB's A-pose rest."""
    for side, sg in (("L", 1.0), ("R", -1.0)):
        pose(rig, {"upperarm01." + side: (4.0, 0, -drop * sg), "lowerarm01." + side: (elbow, 0, 0)})


HANDS = {
    # per finger (thumb=1 .. pinky=5): curl of the three joints in degrees (bone X)
    "relaxed": {1: (8, 12, 10), 2: (12, 18, 12), 3: (16, 22, 14), 4: (20, 26, 16), 5: (24, 30, 18)},
    "flat": {1: (4, 6, 4), 2: (4, 6, 4), 3: (5, 6, 4), 4: (6, 8, 5), 5: (8, 10, 6)},
    "point": {1: (25, 35, 20), 2: (2, 3, 2), 3: (70, 85, 45), 4: (75, 90, 50), 5: (78, 92, 50)},
    "touch": {1: (12, 16, 10), 2: (8, 10, 6), 3: (22, 28, 16), 4: (32, 40, 24), 5: (38, 46, 26)},
    "fist_soft": {1: (30, 30, 20), 2: (55, 65, 40), 3: (60, 70, 45), 4: (62, 72, 45), 5: (64, 74, 45)},
}


def hand(rig, side, shape="relaxed", spread=1.0, seed=0):
    """Finger shape for one hand, with a little per-finger irregularity (no two fingers ever match)."""
    import random as _r
    rs = _r.Random(hash((rig.name, side, seed)) & 0xffff)
    for fi, curls in HANDS[shape].items():
        for j, c in enumerate(curls):
            pb = rig.pose.bones.get("finger%d-%d.%s" % (fi, j + 1, side))
            if pb is None:
                continue
            pb.rotation_mode = 'XYZ'
            splay = (fi - 3) * 3.0 * spread if j == 0 else 0.0
            pb.rotation_euler = (math.radians(c + rs.uniform(-3, 3)), 0.0, math.radians(splay * (1 if side == "L" else -1)))
