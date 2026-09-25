"""The film's child, built with MPFB (MakeHuman for Blender, CC0 assets): real anatomy, skin textures, eyes,
lashes, brows, hair and fitted clothes, rigged for posing. Our continuity tweaks: graphite top, amber rib-knit cuffs."""
import bpy, importlib, math
import addon_utils
from mathutils import Vector as V, Euler, Quaternion, Matrix
import sb

MP = "bl_ext.user_default.mpfb"
_HS = None


def mpfb():
    global _HS
    if _HS is None:
        addon_utils.enable(MP, default_set=True)
        _HS = importlib.import_module(MP + ".services.humanservice").HumanService
    return _HS


PHENOTYPE = dict(gender=0.3, age=0.155, muscle=0.5, weight=0.48, proportions=0.55, height=0.5, cupsize=0.5, firmness=0.5,
                 race=dict(african=0.55, asian=0.2, caucasian=0.25))


def build(name="Child", hair="afro01", top="male_casualsuit01", shoes="shoes05", rig="default", subdiv=1,
          skin="young_african_female", eyes_mat="brown", phenotype=None, clothes=None, eyebrows="eyebrow010"):
    """Build a rigged MPFB human. phenotype: dict overriding PHENOTYPE (age 0.5 = 25 y, 0.155 ~ 9 y, 0.875 ~ 60 y;
    gender 0 = female, 1 = male; race weights). clothes: list of clothes asset names (overrides top/shoes)."""
    HS = mpfb()
    before = set(bpy.data.objects)
    info = HS._create_default_human_info_dict()
    ph = dict(PHENOTYPE)
    if phenotype:
        ph.update(phenotype)
    info["phenotype"] = ph
    info["name"] = name
    info["rig"] = rig
    info["eyes"] = "high-poly/high-poly.mhclo"
    info["eyebrows"] = f"{eyebrows}/{eyebrows}.mhclo"
    info["eyelashes"] = "eyelashes02/eyelashes02.mhclo"
    info["teeth"] = "teeth_base/teeth_base.mhclo"
    info["hair"] = f"{hair}/{hair}.mhclo" if hair else ""
    cl = clothes if clothes is not None else [top, shoes]
    info["clothes"] = [f"{c}/{c}.mhclo" for c in cl if c]
    info["skin_mhmat"] = f"{skin}/{skin}.mhmat"
    info["skin_material_type"] = "ENHANCED_SSS"
    info["eyes_material_type"] = "MAKESKIN"
    info["alternative_materials"] = {}
    settings = HS.get_default_deserialization_settings()
    settings["subdiv_levels"] = subdiv
    basemesh = HS.deserialize_from_dict(info, settings)
    new = [o for o in bpy.data.objects if o not in before]
    rig_obj = next((o for o in new if o.type == 'ARMATURE'), None)
    parts = [o for o in new if o.type == 'MESH']
    return basemesh, rig_obj, parts


def recolor_top(parts, color=(0.055, 0.058, 0.066)):
    """Re-shade the long-sleeve top to graphite (keep its normal/AO detail)."""
    for o in parts:
        if "casualsuit" not in o.name.lower():
            continue
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes:
                continue
            nt = m.node_tree
            for n in nt.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    inp = n.inputs['Base Color']
                    src = inp.links[0].from_socket if inp.links else None
                    hsv = nt.nodes.new('ShaderNodeHueSaturation')
                    hsv.inputs['Saturation'].default_value = 0.0
                    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
                    mul.inputs[0].default_value = 1.0
                    if src:
                        nt.links.new(src, hsv.inputs['Color'])
                    nt.links.new(hsv.outputs[0], mul.inputs[6])
                    mul.inputs[7].default_value = (color[0] * 3.2, color[1] * 3.2, color[2] * 3.2, 1)
                    nt.links.new(mul.outputs[2], inp)


def pose_bone(rig, name, rot=None, quat=None):
    pb = rig.pose.bones.get(name)
    if pb is None:
        return None
    if quat is not None:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = quat
    elif rot is not None:
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = Euler([math.radians(a) for a in rot], 'XYZ')
    return pb


def fix_eyes(parts, tex="brownlight_eye.png"):
    """Swap the iris texture to the warm light-brown (amber-hazel) iris; wet, glossy cornea."""
    import os
    d = os.path.join(bpy.utils.user_resource('EXTENSIONS'), ".user", "user_default", "mpfb", "data", "eyes", "materials")
    if not os.path.isdir(d):
        d = os.path.expanduser("~/Library/Application Support/Blender/5.2/extensions/.user/user_default/mpfb/data/eyes/materials")
    img = bpy.data.images.load(os.path.join(d, tex), check_existing=True)
    for o in parts:
        if "high-poly" not in o.name:
            continue
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes:
                continue
            for n in m.node_tree.nodes:
                if n.type == 'TEX_IMAGE':
                    n.image = img
                if n.type == 'BSDF_PRINCIPLED':
                    n.inputs['Roughness'].default_value = 0.05
                    n.inputs['Coat Weight'].default_value = 1.0
                    n.inputs['Coat Roughness'].default_value = 0.0


def add_cuffs(rig, parts, length=0.06, color=(0.62, 0.24, 0.035)):
    """Turn the last ~6 cm of each sleeve into an amber rib-knit cuff (identity marker). Works on the shirt mesh
    itself (vertex mask attribute), so it deforms with the sleeve."""
    import numpy as np
    shirt = next((o for o in parts if "casualsuit" in o.name.lower()), None)
    if shirt is None:
        return None
    me = shirt.data
    co = np.array([v.co for v in me.vertices])
    mw = np.array(shirt.matrix_world)
    cow = co @ mw[:3, :3].T + mw[:3, 3]
    mask = np.zeros(len(co))
    angle = np.zeros(len(co))
    for side in ("L", "R"):
        b = rig.data.bones.get(f"lowerarm02.{side}")
        if b is None:
            continue
        rw = np.array(rig.matrix_world)
        tail = rw[:3, :3] @ np.array(b.tail_local) + rw[:3, 3]
        head = rw[:3, :3] @ np.array(b.head_local) + rw[:3, 3]
        ax = (tail - head) / np.linalg.norm(tail - head)
        rel = cow - tail
        along = rel @ ax                       # negative = toward the elbow
        radial = np.linalg.norm(rel - np.outer(along, ax), axis=1)
        m = np.clip((along + length) / 0.006, 0, 1) * (radial < 0.07) * (along < 0.04)
        ref = np.cross(ax, [0, 0, 1.0]); ref /= np.linalg.norm(ref)
        ref2 = np.cross(ax, ref)
        radv = rel - np.outer(along, ax)
        ang = np.arctan2(radv @ ref2, radv @ ref)
        angle = np.where(m > mask, ang, angle)
        mask = np.maximum(mask, m)
    cols = np.stack([mask, (angle + np.pi) / (2 * np.pi), np.zeros_like(mask), np.ones_like(mask)], 1)
    attr = me.color_attributes.new("cuff", 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", cols.ravel().astype(np.float32))
    for slot in shirt.material_slots:
        mt = slot.material
        if not mt or not mt.use_nodes:
            continue
        nt = mt.node_tree
        bs = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bs is None:
            continue
        at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = "cuff"
        sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(at.outputs['Color'], sep.inputs[0])
        inp = bs.inputs['Base Color']
        src = inp.links[0].from_socket if inp.links else None
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
        nt.links.new(sep.outputs[0], mix.inputs[0])
        if src:
            nt.links.new(src, mix.inputs[6])
        # rib-knit amber with subtle variation
        # ribs run along the forearm: sine of the angle around the forearm axis (G channel)
        mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 2 * math.pi * 38
        nt.links.new(sep.outputs[1], mul.inputs[0])
        sn = nt.nodes.new('ShaderNodeMath'); sn.operation = 'SINE'
        nt.links.new(mul.outputs[0], sn.inputs[0])
        rng = nt.nodes.new('ShaderNodeMapRange'); rng.inputs[1].default_value = -1; rng.inputs[2].default_value = 1
        nt.links.new(sn.outputs[0], rng.inputs[0])
        rib = nt.nodes.new('ShaderNodeMix'); rib.data_type = 'RGBA'
        nt.links.new(rng.outputs[0], rib.inputs[0])
        # rib relief in the normal
        bmp = nt.nodes.new('ShaderNodeBump'); bmp.inputs['Strength'].default_value = 0.6; bmp.inputs['Distance'].default_value = 0.0008
        hm = nt.nodes.new('ShaderNodeMath'); hm.operation = 'MULTIPLY'
        nt.links.new(rng.outputs[0], hm.inputs[0]); nt.links.new(sep.outputs[0], hm.inputs[1])
        nt.links.new(hm.outputs[0], bmp.inputs['Height'])
        nin = bs.inputs['Normal']
        if nin.links:
            nt.links.new(nin.links[0].from_socket, bmp.inputs['Normal'])
        nt.links.new(bmp.outputs[0], nin)
        rib.inputs[6].default_value = (color[0] * 0.75, color[1] * 0.7, color[2] * 0.6, 1)
        rib.inputs[7].default_value = (*color, 1)
        nt.links.new(rib.outputs[2], mix.inputs[7])
        nt.links.new(mix.outputs[2], inp)
        # sheen/roughness for knit
        bs.inputs['Sheen Weight'].default_value = 0.4
    return shirt
