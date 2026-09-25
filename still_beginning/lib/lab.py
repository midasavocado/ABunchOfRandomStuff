"""Laboratory assets for s03 (and the researcher glove reused in s08): nitrile glove material, an original
upright research microscope (Zeiss/Leica-class proportions, no logos/text), specimen slide, lab bench dressing.
World: microscope faces -Y (the user sits at -Y), Z up, metres. Base centre at the origin, floor of base z=0."""
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion
import sb, props, handmesh

ROOT = sb.ROOT


# ------------------------------------------------------------------ materials

def nitrile(name="Nitrile", color=(0.075, 0.09, 0.36), nails=None):
    """Blue-violet nitrile: satin, faint translucency, crease wrinkles at the joints (SDF hand 'hand' attribute),
    fingernail edges embossed through the thin glove (nails = list from the hand npz 'N'), textured fingertips."""
    m = sb.mat(name, color, rough=0.30, spec=0.55, sheen=0.1, sss=0.10, sss_radius=(0.35, 0.4, 1.0), sss_scale=0.0012)
    b = m.node_tree.nodes["Principled BSDF"]
    b.subsurface_method = 'BURLEY'
    b.inputs['Sheen Tint'].default_value = (0.7, 0.75, 1.0, 1)
    nb = sb.NB(m)
    co = nb.coord('Object')
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "hand"
    sep = nb.new('ShaderNodeSeparateColor'); nb.link(at.outputs['Color'], sep.inputs[0])
    crease, palm, phase = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    # joint wrinkles: arcs across the finger (phase = mm along the joint axis) broken up by noise
    wph = nb.noise(co, scale=180, detail=2)
    wr = nb.math('SINE', nb.math('ADD', nb.math('MULTIPLY', phase, 2 * math.pi / 1.1), nb.math('MULTIPLY', wph.outputs['Fac'], 9.0)))
    wn = nb.noise(co, scale=500, detail=3)
    wr = nb.math('MULTIPLY', nb.maprange(wr, 0.3, 1.0), nb.maprange(wn.outputs['Fac'], 0.4, 0.7, 0.0, 1.0))
    wr = nb.math('MULTIPLY', wr, nb.maprange(crease, 0.02, 0.5))
    # loose glove folds (stretched noise along the fingers), fine grain
    folds = nb.noise(nb.mapping(co, scale=(1, 3.5, 1)), scale=70, detail=4, rough=0.55, dist=0.8)
    fold_h = nb.math('POWER', nb.maprange(folds.outputs['Fac'], 0.45, 0.75), 2.0)
    fine = nb.noise(co, scale=5000, detail=2)
    tex = nb.voronoi(co, scale=2600, feature='F1')
    texm = nb.math('POWER', nb.maprange(tex.outputs['Distance'], 0.0, 0.5, 1.0, 0.0), 2.0)
    h = nb.math('ADD', nb.math('MULTIPLY', wr, -1.0), nb.math('MULTIPLY', fold_h, -0.6))
    h = nb.math('ADD', h, nb.math('MULTIPLY', fine.outputs['Fac'], 0.05))
    h = nb.math('ADD', h, nb.math('MULTIPLY', texm, 0.12))
    if nails:
        for n in nails:
            if n["name"].startswith("__"):
                continue
            c, d, up = V(n["c"]), V(n["d"]), V(n["up"])
            side = d.cross(up).normalized()
            rel = nb.vmath('SUBTRACT', co, tuple(c))
            uu = nb.math('DIVIDE', nb.vmath('DOT_PRODUCT', rel, tuple(side)), n["w"] * 0.52)
            vv = nb.math('DIVIDE', nb.vmath('DOT_PRODUCT', rel, tuple(d)), n["L"] * 0.55)
            nn = nb.vmath('DOT_PRODUCT', rel, tuple(up))
            rr = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', uu, uu), nb.math('MULTIPLY', vv, vv)))
            plate = nb.math('MULTIPLY', nb.maprange(rr, 1.0, 0.86), nb.maprange(nn, -0.003, -0.0015))
            h = nb.math('ADD', h, nb.math('MULTIPLY', plate, 0.9))
    nb.set('Normal', nb.bump(h, strength=0.55, distance=0.0005))
    col = nb.mix(fold_h, (color[0] * 1.08, color[1] * 1.06, color[2] * 1.04, 1), (color[0] * 0.72, color[1] * 0.74, color[2] * 0.8, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.27, 0.36))
    return m


def enamel(name="Enamel", color=(0.80, 0.80, 0.78)):
    """Microscope stand: warm-white powder-coated/enamelled cast body (slight orange peel, very soft wear)."""
    m = sb.mat(name, color, rough=0.32, spec=0.5, coat=0.35, coat_rough=0.15)
    nb = sb.NB(m)
    co = nb.coord('Object')
    op = nb.noise(co, scale=900, detail=3)
    nb.set('Normal', nb.bump(op.outputs['Fac'], strength=0.05, distance=0.0003))
    n2 = nb.noise(co, scale=12, detail=4)
    nb.set('Roughness', nb.maprange(n2.outputs['Fac'], 0.3, 0.7, 0.28, 0.38))
    return m


def black_anod(name="BlackAnod"):
    return sb.brushed_metal(name, (0.028, 0.028, 0.03), rough=0.33, aniso=0.3, scale=200, bump=0.02)


def chrome(name="Chrome"):
    return sb.brushed_metal(name, (0.86, 0.86, 0.87), rough=0.09, aniso=0.2, scale=300, bump=0.005)


def satin_alu(name="SatinAlu"):
    return sb.brushed_metal(name, (0.74, 0.745, 0.75), rough=0.24, aniso=0.55, scale=400, bump=0.02)


def knurl(name, color=(0.05, 0.05, 0.055), pitch=1400):
    """Straight knurl (ribs parallel to the object's local Z axis = lathe/cylinder axis)."""
    m = sb.mat(name, color, metal=0.7, rough=0.35)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    ang = nb.math('ARCTAN2', sep.outputs[1], sep.outputs[0])
    rib = nb.math('SINE', nb.math('MULTIPLY', ang, pitch / 10.0))
    nb.set('Normal', nb.bump(rib, strength=0.6, distance=0.0003))
    return m


# ------------------------------------------------------------------ glove hand

def glove_hand(npz, name="Glove", mat=None):
    import json
    nails = json.loads(str(np.load(npz)["N"]))
    o, parts = handmesh.load(npz, name, skin=mat or nitrile(nails=nails), nails=False)
    o.scale = (1.25, 1.25, 1.25)
    return o


def lab_sleeve(name, parent, length=0.16, r0=0.029, r1=0.034):
    """White cotton lab-coat cuff around the forearm (hand frame: forearm along -Y from the wrist)."""
    prof = [(r0 - 0.002, -0.058), (r0, -0.060), (r0 + 0.001, -0.066), (r0 + 0.0015, -0.09), (r1, -0.06 - length),
            (r1 - 0.004, -0.06 - length)]
    o = sb.lathe(name, prof, segs=64, mat=sb.fabric(name + "M", (0.80, 0.81, 0.82), weave=2500, sheen=0.3, fuzz=0.2), axis='Y')
    o.rotation_euler = (-math.pi / 2, 0, 0)
    o.scale = (1.0, 1.0, 0.82)
    o.parent = parent
    sb.solidify(o, 0.0015)
    return o


# ------------------------------------------------------------------ microscope

def _extrude_profile(name, pts_yz, half_x, bevel, mat, segs=4):
    """Side-silhouette polygon (y,z) extruded along X (+-half_x) with bevelled edges."""
    bm = bmesh.new()
    vf = [bm.verts.new((-half_x, y, z)) for y, z in pts_yz]
    face = bm.faces.new(vf)
    r = bmesh.ops.extrude_face_region(bm, geom=[face])
    for v in r['geom']:
        if isinstance(v, bmesh.types.BMVert):
            v.co.x += 2 * half_x
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    me.materials.append(mat)
    m = o.modifiers.new("Bev", 'BEVEL')
    m.width = bevel
    m.segments = segs
    m.limit_method = 'ANGLE'
    m.angle_limit = math.radians(40)
    sb.shade_auto(o, 50)
    return o


def arc_pts(c, r, a0, a1, n=12):
    return [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def microscope(stage_z=0.200):
    """Builds the microscope. Returns dict of key objects/points (fine knob empty for animation etc.)."""
    M = {}
    white = enamel("MsEnamel")
    graph = sb.painted("MsGraphite", (0.05, 0.052, 0.056), rough=0.42, coat=0.2)
    blk = black_anod("MsBlack")
    chr_ = chrome("MsChrome")
    alu = satin_alu("MsAlu")
    amber = props.amber_anodized("MsAmber")
    # --- base: low cast plinth, rounded front, slight taper
    base_pts = [(-0.150, 0.0), (0.135, 0.0), (0.140, 0.040), (0.130, 0.056), (-0.120, 0.056), (-0.150, 0.030)]
    base = _extrude_profile("MsBase", base_pts, 0.115, 0.012, white, 5)
    # --- pillar + arm: C-shaped cast body at the back, arm reaches forward over the stage
    arm = [(0.060, 0.050), (0.135, 0.050), (0.138, 0.270), (0.128, 0.330), (0.100, 0.350), (-0.050, 0.350),
           (-0.070, 0.340), (-0.072, 0.300), (-0.050, 0.268), (0.030, 0.262), (0.052, 0.245), (0.058, 0.20)]
    body = _extrude_profile("MsArm", arm, 0.042, 0.010, white, 5)
    # graphite side accent panel on the pillar
    pan = _extrude_profile("MsPanel", [(0.075, 0.12), (0.125, 0.12), (0.125, 0.30), (0.075, 0.30)], 0.0435, 0.004, graph, 3)
    # --- illuminator port on the base: field lens under a graphite ring with an amber index ring
    port_y = -0.010
    ring = sb.lathe("MsPort", [(0.020, 0.0561), (0.026, 0.0561), (0.027, 0.064), (0.024, 0.066), (0.019, 0.066)], segs=96, mat=graph)
    ring.location = (0, port_y, 0)
    amb = sb.lathe("MsPortAmber", [(0.0265, 0.0578), (0.0275, 0.0578), (0.0276, 0.0600), (0.0266, 0.0600)], segs=128, mat=amber)
    amb.location = (0, port_y, 0)
    lensm = sb.mat("MsFieldLens", (0.9, 0.95, 1.0), rough=0.0, transmission=1.0, ior=1.52, coat=1.0, thin_film=300)
    fl = sb.prim("cyl", "MsFieldLens", loc=(0, port_y, 0.0645), vertices=64, radius=0.0195, depth=0.002, mat=lensm)
    # --- condenser under the stage (black body, chrome ring, aperture lever)
    cz0 = stage_z - 0.052
    cond = sb.lathe("MsCond", [(0.0, cz0), (0.024, cz0), (0.026, cz0 + 0.004), (0.026, cz0 + 0.03), (0.020, cz0 + 0.038),
                                (0.012, cz0 + 0.046), (0.0, cz0 + 0.046)], segs=96, mat=blk)
    cond.location = (0, port_y, 0)
    cr = sb.lathe("MsCondRing", [(0.0262, cz0 + 0.012), (0.0268, cz0 + 0.013), (0.0268, cz0 + 0.02), (0.0262, cz0 + 0.021)], segs=96, mat=chr_)
    cr.location = (0, port_y, 0)
    lever = sb.prim("cyl", "MsLever", loc=(0.036, port_y - 0.012, cz0 + 0.016), rot=(0, math.pi / 2, 0.4), vertices=16, radius=0.0022, depth=0.028, mat=chr_)
    # condenser carrier arm back to the pillar
    carr = sb.prim("cube", "MsCondCarrier", loc=(0, 0.045, cz0 + 0.016), scale=(0.018, 0.045, 0.008), mat=blk)
    sb.bevel(carr, 0.003)
    # the condenser top lens: emits the transmitted light up through the specimen
    top_lens = sb.prim("cyl", "MsCondLens", loc=(0, port_y, cz0 + 0.0462), vertices=64, radius=0.0115, depth=0.0008,
                       mat=sb.mat("MsCondGlass", (0.02, 0.022, 0.025), rough=0.05, coat=1.0))
    fld = sb.prim("cyl", "MsField", loc=(0, port_y, cz0 + 0.0468), vertices=64, radius=0.0042, depth=0.0002,
                  mat=field_light("MsFieldLight"))
    top_lens = fld
    M["cond_light"] = top_lens
    # --- stage: black anodized plate with a light aperture, mechanical stage, slide holder
    sz = stage_z
    st = _extrude_profile("MsStage", [(-0.085, sz - 0.012), (0.054, sz - 0.012), (0.054, sz), (-0.085, sz)], 0.090, 0.003, blk, 3)
    # stage aperture (dark hole look): a black disc slightly above the plate edge is not needed - we cut with boolean
    hole = sb.prim("cyl", "MsStageHole", loc=(0, port_y, sz - 0.006), vertices=64, radius=0.014, depth=0.03)
    bo = st.modifiers.new("Hole", 'BOOLEAN'); bo.object = hole; bo.operation = 'DIFFERENCE'
    st.modifiers.move(len(st.modifiers) - 1, 0)
    hole.hide_render = True; hole.hide_viewport = True
    # stage support block to the pillar
    sup = sb.prim("cube", "MsStageSup", loc=(0, 0.052, sz - 0.03), scale=(0.035, 0.012, 0.022), mat=blk)
    sb.bevel(sup, 0.005)
    # mechanical stage rails + slide holder arm (chrome) with spring clip
    rail = sb.prim("cube", "MsRail", loc=(0.0, -0.068, sz + 0.003), scale=(0.085, 0.006, 0.003), mat=alu)
    sb.bevel(rail, 0.001)
    holder = sb.prim("cube", "MsHolder", loc=(-0.045, port_y, sz + 0.0035), scale=(0.004, 0.030, 0.0025), mat=alu)
    sb.bevel(holder, 0.0008)
    clip = sb.bezier_curve("MsClip", [(-0.041, port_y - 0.024, sz + 0.004), (-0.030, port_y - 0.020, sz + 0.003),
                                      (-0.024, port_y - 0.016, sz + 0.0021)], radius=0.0011, mat=chr_)
    # stage drive knobs (coaxial, hanging below on the right front)
    for i, (r, zz) in enumerate(((0.009, sz - 0.040), (0.0072, sz - 0.056))):
        k = sb.prim("cyl", "MsStageKnob%d" % i, loc=(0.070, -0.050, zz), vertices=48, radius=r, depth=0.014, mat=knurl("MsSK%d" % i, pitch=900))
        sb.bevel(k, 0.001)
    shaft = sb.prim("cyl", "MsStageShaft", loc=(0.070, -0.050, sz - 0.03), vertices=16, radius=0.003, depth=0.05, mat=chr_)
    # --- glass slide + coverslip + specimen
    slide_z = sz + 0.0005
    # thin slide/coverslip glass: no screen-space refraction (it would hide the specimen layer), just
    # transmission + crisp reflections
    glass = sb.glass("MsSlideGlass", color=(0.96, 0.99, 0.98), rough=0.0, ior=1.52)
    glass.use_raytrace_refraction = False
    sl = sb.prim("cube", "MsSlide", loc=(0.0, port_y, slide_z), scale=(0.038, 0.013, 0.0005), mat=glass)
    cs = sb.prim("cube", "MsCover", loc=(0.0, port_y, slide_z + 0.00058), scale=(0.011, 0.011, 0.00008), mat=glass)
    frost = sb.prim("cube", "MsFrost", loc=(-0.029, port_y, slide_z + 0.00052), scale=(0.009, 0.0129, 0.00003),
                    mat=sb.mat("MsFrostM", (0.85, 0.86, 0.87), rough=0.8))
    spec = specimen_plane("MsSpecimen", (0.0, port_y, slide_z + 0.00068), 0.0095)
    M["specimen"] = spec
    M["slide_top"] = V((0.0, port_y, slide_z + 0.0007))
    # --- nosepiece (revolving turret, axis tilted back 18 deg) + objectives; active one vertical over the slide
    wd, olen, rho, tilt = 0.0085, 0.045, 0.031, math.radians(18)
    A = V((0, math.sin(tilt), math.cos(tilt)))
    S0 = V((0.0, port_y, M["slide_top"].z + wd + olen))
    u = V((0, math.cos(tilt), -math.sin(tilt)))
    T = S0 - u * rho          # active objective at the rear of the turret; the others splay forward
    qA = V((0, 0, 1)).rotation_difference(A)
    turret = sb.empty("MsTurret", loc=T)
    turret.rotation_mode = 'QUATERNION'
    turret.rotation_quaternion = qA
    tdisk = sb.lathe("MsTurretDisk", [(0.0, 0.0), (0.047, 0.0), (0.049, 0.004), (0.045, 0.016), (0.028, 0.024), (0.0, 0.025)],
                     segs=128, mat=chr_)
    tdisk.parent = turret
    tknurl = sb.lathe("MsTurretGrip", [(0.0492, 0.0045), (0.0496, 0.006), (0.0496, 0.0105), (0.0475, 0.0125)], segs=128,
                      mat=knurl("MsTK", (0.6, 0.6, 0.62), pitch=2000))
    tknurl.parent = turret
    # neck from the turret up into the arm overhang
    neck = sb.prim("cyl", "MsNeck", loc=T + A * 0.035, vertices=64, radius=0.022, depth=0.03, mat=chr_)
    neck.rotation_mode = 'QUATERNION'; neck.rotation_quaternion = qA
    objs = []
    codes = [(0.15, 0.45, 0.2), (0.75, 0.6, 0.1), (0.55, 0.12, 0.08), (0.1, 0.25, 0.6), (0.85, 0.85, 0.85)]
    for i in range(5):
        R = Quaternion(A, math.radians(i * 72))
        S = T + R @ (S0 - T)
        d = R @ V((0, 0, -1))
        L_ = olen - (0 if i == 0 else 0.003 * ((i * 3) % 4))
        ob = objective("MsObj%d" % i, L_, codes[i], chr_, blk)
        ob.location = S
        ob.rotation_mode = 'QUATERNION'
        ob.rotation_quaternion = V((0, 0, -1)).rotation_difference(d)
        objs.append(ob)
    M["objectives"] = objs
    M["turret"] = turret
    M["obj_front"] = S0 - V((0, 0, olen))
    # --- head: binocular body + eyepiece tubes toward the user
    head = _extrude_profile("MsHead", [(-0.068, 0.345), (0.030, 0.345), (0.030, 0.385), (-0.02, 0.400), (-0.068, 0.385)], 0.034, 0.008, graph, 4)
    for s in (-1, 1):
        tube_c = V((s * 0.032, -0.085, 0.430))
        tb = sb.prim("cyl", "MsTube", loc=tube_c, rot=(math.radians(-60), 0, 0), vertices=48, radius=0.0135, depth=0.07, mat=graph)
        sb.bevel(tb, 0.002)
        ep = sb.prim("cyl", "MsEP", loc=tube_c + V((0, -0.030, 0.018)), rot=(math.radians(-60), 0, 0), vertices=48, radius=0.0145, depth=0.03, mat=blk)
        sb.bevel(ep, 0.002)
        cup = sb.prim("cyl", "MsEPCup", loc=tube_c + V((0, -0.045, 0.026)), rot=(math.radians(-60), 0, 0), vertices=48, radius=0.0158, depth=0.012, mat=sb.rubber("MsCupR"))
        sb.bevel(cup, 0.003)
    bridge = sb.prim("cube", "MsBino", loc=(0, -0.070, 0.400), scale=(0.048, 0.025, 0.018), mat=graph)
    bridge.rotation_euler = (math.radians(-30), 0, 0)
    sb.bevel(bridge, 0.008)
    # --- focus knobs (coaxial coarse + fine) on both sides of the pillar, axis X
    kz, ky = 0.118, 0.090
    fine_parts = {}
    for s in (-1, 1):
        coarse = sb.lathe("MsCoarse%d" % s, [(0.0, 0.0), (0.022, 0.0), (0.0235, 0.002), (0.0235, 0.012), (0.022, 0.014), (0.012, 0.016), (0.0, 0.016)],
                          segs=128, mat=knurl("MsCK%d" % s, (0.72, 0.72, 0.73), pitch=640), axis='X')
        coarse.location = (s * 0.043, ky, kz)
        if s < 0:
            coarse.rotation_euler = (0, -math.pi / 2, 0)
        amb = sb.lathe("MsKnobAmber%d" % s, [(0.0120, 0.0), (0.0158, 0.0), (0.0158, 0.0022), (0.0120, 0.0022)], segs=128, mat=amber, axis='X')
        amb.location = (s * 0.0592, ky, kz)
        if s < 0:
            amb.rotation_euler = (0, -math.pi / 2, 0)
        # fine knob: pivot empty (animated), knurled rim + graduated face
        shaft = sb.prim("cyl", "MsFineShaft%d" % s, loc=(s * 0.0625, ky, kz), rot=(0, math.pi / 2, 0), vertices=32, radius=0.006, depth=0.008, mat=chr_)
        piv = sb.empty("MsFinePiv%d" % s, loc=(s * 0.0655, ky, kz))
        piv.rotation_mode = 'XYZ'
        fk = sb.lathe("MsFine%d" % s, [(0.0, 0.0), (0.0150, 0.0), (0.0152, 0.001), (0.0152, 0.0115), (0.0145, 0.0128), (0.0105, 0.0135), (0.0, 0.0138)],
                      segs=160, mat=knurl("MsFK%d" % s, (0.06, 0.06, 0.065), pitch=2400), axis='X')
        fk.parent = piv
        if s < 0:
            fk.rotation_euler = (0, -math.pi / 2, 0)
        face = sb.prim("cyl", "MsFineFace%d" % s, loc=(s * 0.0137, 0, 0), rot=(0, math.pi / 2, 0), vertices=96, radius=0.0104, depth=0.0006,
                       mat=graduated_face("MsGrad%d" % s))
        face.parent = piv
        fine_parts[s] = piv
    M["fine"] = fine_parts
    M["fine_axis_pt"] = V((0.0655, ky, kz))
    M["knob_face_x"] = 0.0655 + 0.0138
    return M


_OBJ_BODY = []


def _objective_body():
    if not _OBJ_BODY:
        _OBJ_BODY.append(sb.brushed_metal("MsObjBody", (0.80, 0.80, 0.81), rough=0.2, aniso=0.6, scale=500, bump=0.01))
    return _OBJ_BODY[0]


def objective(name, length, code_col, chr_, blk):
    """Objective along -Z from its shoulder (origin): chrome barrel steps, colour code ring, black front."""
    prof = [(0.0, 0.0), (0.0098, 0.0), (0.0098, -0.005), (0.0093, -0.006), (0.0093, -length * 0.50),
            (0.0080, -length * 0.56), (0.0080, -length * 0.80), (0.0062, -length * 0.90), (0.0044, -length * 0.97),
            (0.0036, -length), (0.0, -length)]
    ob = sb.lathe(name, prof, segs=96, mat=_objective_body())
    ringm = sb.mat(name + "Code", code_col, rough=0.35, coat=0.4)
    rg = sb.lathe(name + "Ring", [(0.00928, -length * 0.30), (0.00940, -length * 0.30), (0.00940, -length * 0.30 - 0.0016), (0.00928, -length * 0.30 - 0.0016)],
                  segs=96, mat=ringm)
    rg.parent = ob
    fr = sb.lathe(name + "Front", [(0.0053, -length * 0.935), (0.0040, -length * 0.995), (0.0022, -length * 1.0005)], segs=96, mat=blk)
    fr.parent = ob
    lens = sb.prim("sphere", name + "Lens", loc=(0, 0, -length * 0.999), segments=48, ring_count=24, radius=0.0021,
                   mat=sb.mat(name + "LensM", (0.02, 0.025, 0.03), rough=0.0, spec=1.0, coat=1.0, thin_film=420))
    lens.scale = (1, 1, 0.25)
    lens.parent = ob
    return ob


def field_light(name, strength=0.6):
    """Illuminated field diaphragm: bright disc with a soft edge (radial falloff) - no hard-edged emitter."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    co = nb.coord('Object')
    r = nb.vmath('LENGTH', nb.vmath('MULTIPLY', co, (1, 1, 0)))
    f = nb.math('POWER', nb.maprange(r, 1.0, 0.25), 1.5)
    e = nt.nodes.new('ShaderNodeEmission')
    e.inputs[0].default_value = (1.0, 0.96, 0.9, 1)
    nb.link(nb.math('MULTIPLY', f, strength), e.inputs[1])
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    ad = nt.nodes.new('ShaderNodeAddShader')
    nb.link(e.outputs[0], ad.inputs[0]); nb.link(tr.outputs[0], ad.inputs[1])
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nb.link(ad.outputs[0], o.inputs[0])
    try:
        m.surface_render_method = 'BLENDED'
    except Exception:
        pass
    return m


def graduated_face(name):
    """Fine-focus knob face: satin graphite with engraved graduation ticks (no numerals)."""
    m = sb.mat(name, (0.05, 0.05, 0.055), metal=0.6, rough=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    ang = nb.math('ARCTAN2', sep.outputs[1], sep.outputs[0])
    r = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', sep.outputs[0], sep.outputs[0]), nb.math('MULTIPLY', sep.outputs[1], sep.outputs[1])))
    ticks = nb.maprange(nb.math('SINE', nb.math('MULTIPLY', ang, 50.0)), 0.93, 0.99)
    band = nb.math('MULTIPLY', nb.maprange(r, 0.0078, 0.0082), nb.maprange(r, 0.0100, 0.0096))
    long_ = nb.maprange(nb.math('SINE', nb.math('MULTIPLY', ang, 10.0)), 0.985, 0.997)
    band2 = nb.math('MULTIPLY', nb.maprange(r, 0.0068, 0.0072), nb.maprange(r, 0.0100, 0.0096))
    t = nb.math('MAXIMUM', nb.math('MULTIPLY', ticks, band), nb.math('MULTIPLY', long_, band2))
    nb.set('Base Color', nb.mix(t, (0.05, 0.05, 0.055, 1), (0.75, 0.75, 0.74, 1)))
    nb.set('Roughness', nb.mix(t, 0.3, 0.5, dtype='FLOAT'))
    # concentric machining
    rings = nb.math('SINE', nb.math('MULTIPLY', r, 2 * math.pi / 0.00006))
    nb.set('Normal', nb.bump(rings, strength=0.08, distance=0.00002))
    return m


def specimen_plane(name, loc, half):
    """Stained cell smear seen in transmitted light: a thin transparent layer whose transmission colour is a
    procedural cell field (violet nuclei, pale lilac cytoplasm, cell walls)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    co = nb.coord('Object')
    warp = nb.noise(co, scale=14.0, detail=3)
    cw = nb.mix(0.08, co, warp.outputs['Color'], dtype='VECTOR', blend='MIX')
    cells = nb.voronoi(cw, scale=26.0, feature='F1', rand=0.9)
    edge = nb.voronoi(cw, scale=26.0, feature='DISTANCE_TO_EDGE', rand=0.9)
    nuc_d = cells.outputs['Distance']
    rnd = cells.outputs['Color']
    rs = nb.new('ShaderNodeSeparateColor'); nb.link(rnd, rs.inputs[0])
    size = nb.maprange(rs.outputs[0], 0.0, 1.0, 0.30, 0.42)            # per-cell radius variation
    body = nb.maprange(nb.math('SUBTRACT', nuc_d, size), 0.02, -0.02)    # rounded cell bodies with gaps
    rim_ = nb.math('MULTIPLY', body, nb.maprange(nb.math('SUBTRACT', nuc_d, size), -0.07, -0.01))  # darker membrane
    nucleus = nb.maprange(nuc_d, 0.15, 0.10)
    chroma = nb.noise(co, scale=300, detail=4)
    cyto = nb.mix(nb.maprange(chroma.outputs['Fac'], 0.3, 0.7), (0.66, 0.52, 0.86, 1), (0.80, 0.66, 0.92, 1))
    cyto = nb.mix(nb.math('MULTIPLY', rs.outputs[1], 0.5), cyto, (0.86, 0.56, 0.78, 1))   # some cells pinker
    nucc = nb.mix(nb.maprange(chroma.outputs['Fac'], 0.35, 0.7), (0.20, 0.08, 0.42, 1), (0.36, 0.18, 0.58, 1))
    bg_ = (0.93, 0.92, 0.97, 1)
    col = nb.mix(body, bg_, cyto)
    col = nb.mix(nb.math('MULTIPLY', rim_, 0.5), col, (0.48, 0.34, 0.70, 1))
    col = nb.mix(nb.math('MULTIPLY', nucleus, body), col, nucc)
    # sparse / dense regions of the smear
    gaps = nb.noise(co, scale=5.0, detail=2)
    col = nb.mix(nb.maprange(gaps.outputs['Fac'], 0.60, 0.70), col, bg_)
    # Koehler illumination: the condenser focuses the field on the specimen plane -> the stained cells glow in a
    # soft-edged disc; outside it the slide is dark (opaque layer: no dithered transparency noise).
    r = nb.vmath('LENGTH', nb.vmath('MULTIPLY', co, (1, 1, 0)))
    disc = nb.math('POWER', nb.maprange(r, 0.78, 0.55), 1.3)
    sat = nb.new('ShaderNodeHueSaturation')
    sat.inputs['Saturation'].default_value = 1.6
    nb.link(col, sat.inputs['Color'])
    em = nt.nodes.new('ShaderNodeEmission')
    nb.link(sat.outputs[0], em.inputs[0])
    nb.link(nb.math('MULTIPLY', disc, 0.55), em.inputs[1])
    gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Roughness'].default_value = 0.08
    gl.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1)
    ad = nt.nodes.new('ShaderNodeAddShader')
    nb.link(em.outputs[0], ad.inputs[0]); nb.link(gl.outputs[0], ad.inputs[1])
    nb.link(ad.outputs[0], out.inputs[0])
    o = sb.prim("plane", name, loc=loc, scale=(half, half, 1), mat=m)
    # object-space pattern scale: plane spans +-1 in object coords
    return o


# ------------------------------------------------------------------ lab room dressing

def lab_bench(z_top=0.0, cool=True):
    """Bench top (dark grey phenolic resin) + distant soft lab background forms (shelves, bottles, window)."""
    top = sb.prim("cube", "Bench", loc=(0, 0.35, z_top - 0.015), scale=(1.4, 0.75, 0.015),
                  mat=sb.painted("BenchTop", (0.05, 0.052, 0.056), rough=0.5, coat=0.15, scale=2.0))
    sb.bevel(top, 0.004)
    objs = [top]
    rng = sb.rng(7)
    glassm = sb.glass("LabGlass", color=(0.95, 0.98, 1.0))
    amberbot = sb.glass("LabAmberBottle", color=(0.55, 0.25, 0.05))
    # reagent bottles, a pipette rack and a monitor far behind
    for i in range(9):
        x = rng.uniform(-0.8, 0.8)
        y = rng.uniform(0.75, 0.95)
        h = rng.uniform(0.10, 0.22)
        r = rng.uniform(0.025, 0.045)
        b = sb.prim("cyl", "Bottle%d" % i, loc=(x, y, z_top + h / 2), vertices=32, radius=r, depth=h,
                    mat=amberbot if i % 3 == 0 else glassm)
        cap = sb.prim("cyl", "Cap%d" % i, loc=(x, y, z_top + h + 0.012), vertices=24, radius=r * 0.55, depth=0.024,
                      mat=sb.mat("Cap", (0.05, 0.12, 0.35) if i % 2 else (0.8, 0.8, 0.8), rough=0.5))
        objs += [b, cap]
    # back wall with a large softly lit window and shelves
    wall = sb.prim("plane", "LabWall", loc=(0, 1.3, 1.0), rot=(math.pi / 2, 0, 0), scale=(3.0, 1.5, 1),
                   mat=sb.mat("LabWallM", (0.18, 0.19, 0.21), rough=0.8))
    win = sb.prim("plane", "LabWindow", loc=(-0.9, 1.28, 0.9), rot=(math.pi / 2, 0, 0), scale=(0.7, 0.45, 1),
                  mat=sb.emit_mat("LabWinM", (0.75, 0.85, 1.0), 3.0))
    for zz in (0.55, 0.85):
        sh = sb.prim("cube", "Shelf", loc=(0.55, 1.15, zz), scale=(0.55, 0.12, 0.01), mat=sb.mat("ShelfM", (0.6, 0.6, 0.62), rough=0.5))
    return objs
