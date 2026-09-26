"""Soul for the MPFB people: living skin, wet eyes, faces that feel something, bodies that breathe.

skin(parts)            pores + micro-wrinkles + mottled colour + oily/dry sheen breakup + fuller subsurface
eyes(parts, iris)      a natural iris (the stock 'brown' reads maroon), wet glossy cornea that catches catchlights
mood(rig, name)        a named, readable expression built from the face bones (see MOODS)
life(rig, f0, f1)      additive, never-still motion over whatever the shot animates: blinks (with the lower lid
                       answering), breathing (chest + shoulders), head drift, eye micro-saccades, a slow weight shift.
                       Layered with F-curve modifiers / added keys so it never fights the shot's own poses.
All deterministic (seeded by the rig name)."""
import bpy, math, random, os
from mathutils import Euler, Quaternion

# ----------------------------------------------------------------------------------------------------- skin


def skin(parts, pores=1.0, mottling=1.0, sss=0.35, sheen=1.0):
    for o in parts:
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes or not o.name.lower().endswith(".body"):
                continue
            for g in [n for n in m.node_tree.nodes if n.type == 'GROUP']:
                _skin_group(g, pores, mottling, sss, sheen)


def _skin_group(g, pores, mottling, sss, sheen):
    nt = g.node_tree
    if nt.get("soul"):
        return
    nt["soul"] = 1
    N, L = nt.nodes, nt.links
    tc = next(n for n in N if n.type == 'TEX_COORD')
    bsdf = next(n for n in N if n.type == 'BSDF_PRINCIPLED')
    bump = next(n for n in N if n.type == 'BUMP')
    ramp = next(n for n in N if n.type == 'VALTORGB')
    uv = tc.outputs['UV']

    def node(t, **kw):
        n = N.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def math_(op, a, b):
        n = node('ShaderNodeMath', operation=op)
        for i, x in enumerate((a, b)):
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                L.new(x, n.inputs[i])
        return n.outputs[0]

    def mrange(x, a, b, c=0.0, d=1.0):
        n = node('ShaderNodeMapRange')
        L.new(x, n.inputs[0])
        n.inputs[1].default_value, n.inputs[2].default_value = a, b
        n.inputs[3].default_value, n.inputs[4].default_value = c, d
        return n.outputs[0]

    def noise(vec, scale, detail=4.0, rough=0.5):
        n = node('ShaderNodeTexNoise')
        L.new(vec, n.inputs['Vector'])
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        return n.outputs['Fac']

    def mapping(vec, scale):
        n = node('ShaderNodeMapping')
        L.new(vec, n.inputs[0])
        n.inputs['Scale'].default_value = scale
        return n.outputs[0]
    # pores: sunken dots (Voronoi cells, ~0.3-0.5 mm on the face in MakeHuman UV space) over the existing noise
    vor = node('ShaderNodeTexVoronoi', feature='F1')
    L.new(uv, vor.inputs['Vector'])
    vor.inputs['Scale'].default_value = 3200.0
    vor.inputs['Randomness'].default_value = 0.9
    pit = mrange(vor.outputs['Distance'], 0.0, 0.22, -1.0, 0.0)
    # micro-wrinkles: two crossed, stretched line fields (skin "diamond" relief)
    w1 = noise(mapping(uv, (900.0, 70.0, 1.0)), 1.0, 3.0, 0.6)
    w2 = noise(mapping(uv, (70.0, 900.0, 1.0)), 1.0, 3.0, 0.6)
    wr = math_('MULTIPLY', math_('ADD', mrange(w1, 0.45, 0.55, -1.0, 0.0), mrange(w2, 0.45, 0.55, -1.0, 0.0)), 0.5)
    h = math_('ADD', math_('ADD', ramp.outputs['Color'], math_('MULTIPLY', pit, 0.9 * pores)), math_('MULTIPLY', wr, 0.35 * pores))
    for l in list(bump.inputs['Height'].links):
        L.remove(l)
    L.new(h, bump.inputs['Height'])
    g.inputs['Pore strength'].default_value = 0.28
    bump.inputs['Distance'].default_value = 0.0006
    # colour: soft blotchy mottling (capillaries / melanin), a few darker specks, keeps the texture's own map
    bc = bsdf.inputs['Base Color'].links[0].from_socket
    blot = noise(uv, 45.0, 5.0, 0.6)
    fine = noise(uv, 420.0, 3.0, 0.6)
    spk = node('ShaderNodeTexVoronoi', feature='F1')
    L.new(uv, spk.inputs['Vector'])
    spk.inputs['Scale'].default_value = 260.0
    speck = mrange(spk.outputs['Distance'], 0.0, 0.08, 1.0, 0.0)
    warm = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L.new(mrange(blot, 0.3, 0.7, 0.0, 0.9 * mottling), warm.inputs[0])
    L.new(bc, warm.inputs[6])
    warm.inputs[7].default_value = (1.10, 0.86, 0.84, 1.0)          # a flush of red
    dark = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L.new(math_('MULTIPLY', speck, math_('MULTIPLY', mrange(fine, 0.55, 0.7), 0.35 * mottling)), dark.inputs[0])
    L.new(warm.outputs[2], dark.inputs[6])
    dark.inputs[7].default_value = (0.78, 0.70, 0.64, 1.0)
    # broad tone: sallower / rosier zones at a few-cm scale (no skin is one colour)
    broad = noise(uv, 9.0, 3.0, 0.5)
    tone = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    L.new(mrange(broad, 0.3, 0.7, 0.0, 0.8 * mottling), tone.inputs[0])
    L.new(dark.outputs[2], tone.inputs[6])
    tone.inputs[7].default_value = (0.94, 0.90, 0.80, 1.0)
    L.new(tone.outputs[2], bsdf.inputs['Base Color'])
    # sheen breakup: oily and dry patches; pores read slightly rougher
    r0 = g.inputs['Roughness'].default_value
    for l in list(bsdf.inputs['Roughness'].links):
        L.remove(l)
    rough = math_('ADD', mrange(noise(uv, 30.0, 4.0, 0.55), 0.3, 0.7, r0 - 0.1 * sheen, r0 + 0.08 * sheen),
                  math_('MULTIPLY', math_('MULTIPLY', pit, -1.0), 0.08))
    L.new(rough, bsdf.inputs['Roughness'])
    g.inputs['Clearcoat'].default_value = 0.14 * sheen
    g.inputs['Clearcoat Roughness'].default_value = 0.38
    g.inputs['SSS strength'].default_value = sss

    try:
        bsdf.inputs['Subsurface Scale'].default_value = 0.012
    except KeyError:
        pass


# ----------------------------------------------------------------------------------------------------- eyes
IRIS = {"brown": ("brown_eye.png", 0.55, 0.55), "hazel": ("brownlight_eye.png", 0.9, 0.85), "green": ("green_eye.png", 0.85, 0.9),
        "blue": ("blue_eye.png", 0.8, 0.9), "grey": ("grey_eye.png", 0.8, 0.9), "darkbrown": ("brown_eye.png", 0.4, 0.38)}


def _mpfb_eye_dir():
    d = os.path.join(bpy.utils.user_resource('EXTENSIONS'), ".user", "user_default", "mpfb", "data", "eyes", "materials")
    if os.path.isdir(d):
        return d
    for v in ("5.2", "5.1", "5.0"):
        for base in ("~/Library/Application Support/Blender/%s" % v, "~/.config/blender/%s" % v):
            d = os.path.expanduser(base + "/extensions/.user/user_default/mpfb/data/eyes/materials")
            if os.path.isdir(d):
                return d
    return None


def eyes(parts, iris="brown"):
    tex, sat, val = IRIS.get(iris, IRIS["brown"])
    d = _mpfb_eye_dir()
    img = bpy.data.images.load(os.path.join(d, tex), check_existing=True) if d else None
    for o in parts:
        if "high-poly" not in o.name:
            continue
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes or m.get("soul"):
                continue
            m["soul"] = 1
            nt = m.node_tree
            ti = next((n for n in nt.nodes if n.type == 'TEX_IMAGE'), None)
            b = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if ti is None or b is None:
                continue
            if img:
                ti.image = img
            hs = nt.nodes.new('ShaderNodeHueSaturation')
            hs.inputs['Saturation'].default_value = sat
            hs.inputs['Value'].default_value = val
            src = b.inputs['Base Color'].links[0].from_socket
            nt.links.new(src, hs.inputs['Color'])
            nt.links.new(hs.outputs[0], b.inputs['Base Color'])
            b.inputs['Roughness'].default_value = 0.08
            b.inputs['Coat Weight'].default_value = 1.0          # the tear film: sharp, bright catchlights
            b.inputs['Coat Roughness'].default_value = 0.0
            b.inputs['Coat IOR'].default_value = 1.376
            b.inputs['Subsurface Weight'].default_value = 0.15
            b.inputs['Subsurface Radius'].default_value = (0.6, 0.3, 0.25)


# ----------------------------------------------------------------------------------------------------- moods
# values: smile 0..1, brows -1..1 (raise +), squint 0..1, jaw 0..1, plus asymmetry (-1..1: which side smiles more)
MOODS = {
    "warm": dict(smile=0.35, brows=0.25, eyes=0.1),
    "delight": dict(smile=0.8, brows=0.35, eyes=0.35, jaw=0.25),
    "laugh": dict(smile=1.0, brows=0.3, eyes=0.5, jaw=0.6),
    "focus": dict(smile=0.08, brows=-0.35, eyes=0.25),
    "wonder": dict(smile=0.2, brows=0.6, eyes=0.0, jaw=0.18),
    "tender": dict(smile=0.5, brows=0.45, eyes=0.2),
    "pride": dict(smile=0.55, brows=0.1, eyes=0.25, asym=0.35),
    "listen": dict(smile=0.18, brows=0.3, eyes=0.05),
}


def mood(rig, name="warm", amount=1.0, asym=None):
    import folks
    p = dict(MOODS[name])
    a = p.pop("asym", 0.0) if asym is None else asym
    p.pop("asym", None)
    folks.expression(rig, **{k: v * amount for k, v in p.items()})
    if a:
        # a lopsided, human smile: one corner a little higher
        side = "L" if a > 0 else "R"
        pb = rig.pose.bones.get("oris04." + side)
        if pb:
            pb.location = (pb.location[0] * (1 + abs(a)), pb.location[1] * (1 + abs(a)), pb.location[2])


# ----------------------------------------------------------------------------------------------------- life


def _fcurve(rig, pb, path, idx, frame):
    """F-curve for pose bone channel (creating a constant key at the current value if needed)."""
    dp = 'pose.bones["%s"].%s' % (pb.name, path)
    ad = rig.animation_data or rig.animation_data_create()
    act = ad.action
    fc = None
    if act is not None:
        try:
            fc = act.fcurves.find(dp, index=idx)
        except Exception:
            fc = None
            for c in _all_fcurves(act):
                if c.data_path == dp and c.array_index == idx:
                    fc = c
    if fc is None:
        pb.keyframe_insert(path, index=idx, frame=frame)
        act = rig.animation_data.action
        for c in _all_fcurves(act):
            if c.data_path == dp and c.array_index == idx:
                fc = c
    return fc


def _all_fcurves(act):
    try:
        return list(act.fcurves)
    except Exception:          # layered actions (Blender 4.4+)
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    out += list(cb.fcurves)
        return out


def _euler(pb):
    if pb.rotation_mode == 'QUATERNION':
        e = pb.rotation_quaternion.to_euler('XYZ')
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = e


def _wobble(rig, bone, idx, f0, deg, period, seed, path="rotation_euler"):
    pb = rig.pose.bones.get(bone)
    if pb is None:
        return
    if path == "rotation_euler":
        _euler(pb)
    fc = _fcurve(rig, pb, path, idx, f0)
    if fc is None:
        return
    m = fc.modifiers.new('NOISE')
    m.blend_type = 'ADD'
    m.scale = period
    m.strength = math.radians(deg) * 2.0 if path == "rotation_euler" else deg * 2.0
    m.phase = seed
    m.depth = 1


def _sine(rig, bone, idx, f0, deg, period, phase=0.0):
    pb = rig.pose.bones.get(bone)
    if pb is None:
        return
    _euler(pb)
    fc = _fcurve(rig, pb, "rotation_euler", idx, f0)
    if fc is None:
        return
    m = fc.modifiers.new('FNGENERATOR')
    m.function_type = 'SIN'
    m.use_additive = True
    m.amplitude = math.radians(deg)
    m.phase_multiplier = 2 * math.pi / period
    m.phase_offset = phase


def life(rig, f0, f1, blink=True, breathe=1.0, head=1.0, saccade=1.0, sway=1.0, blink_every=(2.2, 5.0), fps=24.0,
         calm=1.0, seed=None):
    """calm < 1 = more animated (a laughing table), > 1 = stiller (a child concentrating)."""
    rs = random.Random(seed if seed is not None else rig.name)
    ph = rs.uniform(0, 100)
    br = 4.2 * calm                                   # seconds per breath (~14 / min)
    if breathe:
        for b, deg in (("spine03", 0.55), ("spine04", 0.45), ("spine05", 0.3)):
            _sine(rig, b, 0, f0, deg * breathe, br * fps, ph)
        for s in ("L", "R"):
            _sine(rig, "clavicle." + s, 1, f0, 0.9 * breathe * (1 if s == "L" else -1), br * fps, ph)
    if head:
        _wobble(rig, "head", 0, f0, 1.1 * head / calm, 55.0 * calm, ph)
        _wobble(rig, "head", 2, f0, 1.3 * head / calm, 70.0 * calm, ph + 17)
        _wobble(rig, "neck02", 1, f0, 0.7 * head / calm, 90.0 * calm, ph + 31)
    if sway:
        _wobble(rig, "spine01", 1, f0, 0.8 * sway / calm, 120.0 * calm, ph + 5)
        _wobble(rig, "spine02", 0, f0, 0.5 * sway / calm, 100.0 * calm, ph + 9)
    if saccade:
        # both eyes share the same noise (phase) so they move together
        for s in ("L", "R"):
            _wobble(rig, "eye." + s, 0, f0, 1.4 * saccade, 9.0, ph + 50)
            _wobble(rig, "eye." + s, 2, f0, 1.8 * saccade, 11.0, ph + 60)
    if blink:
        up = [rig.pose.bones.get("orbicularis03." + s) for s in ("L", "R")]
        lo = [rig.pose.bones.get("orbicularis04." + s) for s in ("L", "R")]
        if all(up):
            for pb in up + [p for p in lo if p]:
                _euler(pb)
            base_u = [pb.rotation_euler.x for pb in up]
            base_l = [pb.rotation_euler.x for pb in lo if pb]
            t = f0 - rs.uniform(0, blink_every[1] * fps)
            shape = [(0, 0.0), (1, 0.55), (2, 1.0), (3, 0.9), (5, 0.35), (7, 0.08), (8, 0.0)]   # fast close, slower open
            while t < f1 + 10:
                t += rs.uniform(*blink_every) * fps * calm
                if rs.random() < 0.12:                          # the occasional double blink
                    starts = [t, t + 10]
                else:
                    starts = [t]
                for st in starts:
                    st = int(round(st))
                    if st > f1 + 2 or st + 8 < f0 - 2:
                        continue
                    for df, k in shape:
                        for pb, b0 in zip(up, base_u):
                            pb.rotation_euler.x = b0 - math.radians(32.0) * k
                            pb.keyframe_insert("rotation_euler", index=0, frame=st + df)
                        for pb, b0 in zip([p for p in lo if p], base_l):
                            pb.rotation_euler.x = b0 - math.radians(5.0) * k
                            pb.keyframe_insert("rotation_euler", index=0, frame=st + df)
            for pb, b0 in zip(up, base_u):
                pb.rotation_euler.x = b0
            for pb, b0 in zip([p for p in lo if p], base_l):
                pb.rotation_euler.x = b0


def ensoul(parts, iris="brown", **skin_kw):
    if os.environ.get("SB_NOSOULSKIN") != "1":
        skin(parts, **skin_kw)
    eyes(parts, iris)


# ----------------------------------------------------------------------------------------------------- face units
# MakeHuman's sculpted expression units (data/targets/expression/units/<race>/*.target.gz) as shape keys on the
# basemesh, carried to brows/lashes/eyes proxies. Real anatomy (zygomaticus smile, orbicularis blink) instead of
# bone nudges.
UNITS = ["eye-left-closure", "eye-right-closure", "eye-left-slit", "eye-right-slit", "eye-left-opened-up", "eye-right-opened-up",
         "eyebrows-left-inner-up", "eyebrows-right-inner-up", "eyebrows-left-up", "eyebrows-right-up",
         "eyebrows-left-down", "eyebrows-right-down", "mouth-corner-puller", "mouth-upward-retraction", "mouth-open",
         "mouth-parling", "mouth-compression", "mouth-elevation", "nose-left-elevation", "nose-right-elevation"]
KP = "fu-"

# expressions as unit weights (L/R pairs written once with '*' for both sides)
FACES = {
    "warm": {"mouth-corner-puller": 0.45, "eye-*-slit": 0.12, "eyebrows-*-inner-up": 0.15},
    "delight": {"mouth-corner-puller": 0.85, "mouth-upward-retraction": 0.25, "mouth-parling": 0.25, "eye-*-slit": 0.3,
                "eyebrows-*-up": 0.25, "nose-*-elevation": 0.15},
    "laugh": {"mouth-corner-puller": 1.0, "mouth-upward-retraction": 0.45, "mouth-open": 0.45, "eye-*-slit": 0.5,
              "eyebrows-*-up": 0.2, "nose-*-elevation": 0.3},
    "focus": {"eyebrows-*-down": 0.3, "eye-*-slit": 0.18, "mouth-compression": 0.25},
    "wonder": {"eyebrows-*-up": 0.5, "eyebrows-*-inner-up": 0.3, "eye-*-opened-up": 0.3, "mouth-parling": 0.35,
               "mouth-corner-puller": 0.15},
    "tender": {"mouth-corner-puller": 0.5, "eyebrows-*-inner-up": 0.45, "eye-*-slit": 0.22},
    "pride": {"mouth-corner-puller": 0.6, "mouth-compression": 0.15, "eye-*-slit": 0.25, "eyebrows-*-up": 0.1},
    "listen": {"mouth-corner-puller": 0.2, "eyebrows-*-inner-up": 0.3, "eyebrows-*-up": 0.15},
}


def _mpfb_mod(path):
    import importlib
    return importlib.import_module("bl_ext.user_default.mpfb." + path)


def _units_dir(race):
    d = os.path.dirname(_mpfb_mod("services.targetservice").__file__)
    return os.path.join(os.path.dirname(d), "data", "targets", "expression", "units", race)


def load_face_units(basemesh, race="caucasian"):
    TS = _mpfb_mod("services.targetservice").TargetService
    d = _units_dir(race)
    if not os.path.isdir(d):
        return False
    for u in UNITS:
        name = KP + u
        if basemesh.data.shape_keys and name in basemesh.data.shape_keys.key_blocks:
            continue
        p = os.path.join(d, u + ".target.gz")
        if os.path.exists(p):
            TS.load_target(basemesh, p, weight=0.0, name=name)
    try:
        _mpfb_mod("services.faceservice").FaceService.interpolate_targets(basemesh)
    except Exception as ex:
        print("SOUL interpolate failed", ex)
    return True


def _meshes_with(parts, key):
    return [o for o in parts if o.type == 'MESH' and o.data.shape_keys and key in o.data.shape_keys.key_blocks]


def set_units(parts, weights, frame=None, add=False):
    """weights: {unit (with '*' for both sides): value}. frame -> also keyframe."""
    for u, v in weights.items():
        names = [u.replace("*", "left"), u.replace("*", "right")] if "*" in u else [u]
        for n in names:
            for o in _meshes_with(parts, KP + n):
                kb = o.data.shape_keys.key_blocks[KP + n]
                kb.value = min(1.0, (kb.value if add else 0.0) + v)
                if frame is not None:
                    kb.keyframe_insert("value", frame=frame)


def face(parts, name, amount=1.0, frame=None):
    set_units(parts, {k: v * amount for k, v in FACES[name].items()}, frame)


def face_blinks(parts, f0, f1, seed="x", every=(2.2, 5.0), fps=24.0, calm=1.0):
    """Blinks on the sculpted closure units, over whatever the lids currently hold (squint etc.)."""
    rs = random.Random(seed)
    shape = [(0, 0.0), (1, 0.6), (2, 1.0), (3, 0.92), (5, 0.4), (7, 0.1), (8, 0.0)]
    keys = []
    for side in ("left", "right"):
        keys += [o.data.shape_keys.key_blocks[KP + "eye-%s-closure" % side] for o in _meshes_with(parts, KP + "eye-%s-closure" % side)]
    if not keys:
        return
    base = [k.value for k in keys]
    t = f0 - rs.uniform(0, every[1] * fps)
    while t < f1 + 10:
        t += rs.uniform(*every) * fps * calm
        for st in ([t, t + 10] if rs.random() < 0.12 else [t]):
            st = int(round(st))
            if st > f1 + 2 or st + 8 < f0 - 2:
                continue
            for df, k in shape:
                for kb, b0 in zip(keys, base):
                    kb.value = b0 + (1.0 - b0) * k
                    kb.keyframe_insert("value", frame=st + df)
    for kb, b0 in zip(keys, base):
        kb.value = b0


def alive(rig, f0, f1, calm=1.0, blink=True, **kw):
    """Everything that keeps a person from looking like a mannequin, additive over the shot's animation."""
    life(rig, f0, f1, blink=False, calm=calm, **kw)
    if blink:
        meshes = [o for o in rig.children_recursive if o.type == 'MESH']
        face_blinks(meshes, f0, f1, seed=rig.name, calm=calm)


def alive_shot(rig, sid, **kw):
    import sb
    rig["alive"] = 1
    _, f0, f1, _ = sb.TL.shot(sid)
    alive(rig, f0 - 12, f1 + 12, **kw)


def alive_all(sid, per=None, **kw):
    """alive_shot() for every armature in the scene not already animated; per = {rig name: overrides}."""
    per = per or {}
    for o in list(bpy.context.scene.objects):
        if o.type != 'ARMATURE' or o.get("alive"):
            continue
        o["alive"] = 1
        k = dict(kw)
        for key, v in per.items():
            if o.name.startswith(key):
                k.update(v)
        alive_shot(o, sid, **k)
