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
                     hair="short02", clothes=["male_worksuit01", "shoes01"], skin="middleage_african_male",
                     hair_color=(0.02, 0.018, 0.016),
                     tint={"worksuit": (0.20, 0.22, 0.25)}),
    # s09: the prosthesis wearer (adult woman, 40s) and family
    "wearer": dict(phenotype=dict(age=0.66, gender=0.0, weight=0.5, muscle=0.5,
                                  race=dict(african=0.3, asian=0.1, caucasian=0.6)),
                   hair="bob02", clothes=["male_casualsuit01", "shoes02"], skin="middleage_caucasian_female",
                   hair_color=(0.16, 0.09, 0.05),
                   tint={"casualsuit": (0.70, 0.64, 0.55)}),
    "partner": dict(phenotype=dict(age=0.68, gender=1.0, weight=0.55, muscle=0.5,
                                   race=dict(african=0.25, asian=0.35, caucasian=0.4)),
                    hair="short04", clothes=["male_casualsuit03", "shoes03"], skin="middleage_asian_male",
                    hair_color=(0.03, 0.025, 0.022),
                    tint={"casualsuit": (0.20, 0.30, 0.38)}),
    "grandma": dict(phenotype=dict(age=0.92, gender=0.0, weight=0.55, muscle=0.35,
                                   race=dict(african=0.2, asian=0.1, caucasian=0.7)),
                    hair="bob01", clothes=["male_casualsuit03", "shoes02"], skin="old_caucasian_female",
                    hair_color=(0.62, 0.60, 0.58),
                    tint={"casualsuit": (0.50, 0.40, 0.34)}),
    "teen": dict(phenotype=dict(age=0.27, gender=1.0, weight=0.45, muscle=0.45,
                                race=dict(african=0.35, asian=0.2, caucasian=0.45)),
                 hair="short01", clothes=["male_casualsuit01", "shoes05"], skin="young_caucasian_male",
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
                                   skin=p["skin"], phenotype=p["phenotype"], subdiv=subdiv)
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
    return float(np.linalg.norm(f(x)[:3]))


def expression(rig, smile=0.0, brows=0.0, eyes=0.0, jaw=0.0):
    """Facial expression on the MPFB default rig's face bones (degrees / metres tuned by eye in lookdev):
    smile 0..1 pulls the lip corners up/back and lifts the cheeks (with a slight eye squint - a real Duchenne
    smile), brows -1..1 lowers/raises the inner brows, eyes 0..1 narrows the lids, jaw 0..1 parts the lips."""
    def rot(b, x=0.0, y=0.0, z=0.0):
        pb = rig.pose.bones.get(b)
        if pb is None:
            return
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (math.radians(x), math.radians(y), math.radians(z))

    def mov(b, v):
        pb = rig.pose.bones.get(b)
        if pb is None:
            return
        pb.location = v
    s = smile
    for side, sg in (("L", 1), ("R", -1)):
        # lip corners up/out (bone-local offsets found in lookdev_face sweeps), cheeks lift, lower lids rise
        mov("oris04.%s" % side, (-0.004 * s, 0.005 * s, 0.0))
        mov("oris03.%s" % side, (-0.002 * s, 0.0025 * s, 0.0))
        rot("levator05.%s" % side, 20.0 * s, 0.0, 0.0)
        rot("orbicularis04.%s" % side, -6.0 * (0.6 * s + eyes), 0.0, 0.0)
        rot("oculi01.%s" % side, 8.0 * brows, 0.0, 0.0)
    rot("jaw", 6.0 * jaw + 1.5 * s, 0.0, 0.0)


def hair_color(parts, color, spec=0.35):
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
                n.inputs['Roughness'].default_value = 0.38
                n.inputs['Anisotropic'].default_value = 0.6
                n.inputs['Specular IOR Level'].default_value = spec
                n.inputs['Coat Weight'].default_value = 0.0
