"""Offshore wind farm assets for s12 (and the far horizon of other shots): lofted blades, turbine assembly,
crew-transfer vessel, Ocean-modifier sea with foam, vessel wake and monopile foam rings, aerial haze.
Units: metres. Wind blows toward -Y; rotors face +Y (upwind). Rotation: clockwise seen from upwind."""
import bpy, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion
import sb
import energy as E

BLADE_L = 80.0          # blade length (V164-class rotor, D ~ 164 m)
HUB_H = 112.0           # hub height above mean sea level
SPIN_R = 4.1            # spinner radius (hub cover)
RPM = 12.0
HAZE_COL = (0.62, 0.72, 0.86)


# =========================================================================== haze
def aerial(mat, dist=9000.0, col=None, power=1.0, emit_col=None):
    """Aerial perspective: mix the material's surface with a flat haze emission by camera distance."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
    src = out.inputs['Surface'].links[0].from_socket if out.inputs['Surface'].links else None
    if src is None:
        return mat
    cd = nt.nodes.new('ShaderNodeCameraData')
    div = nt.nodes.new('ShaderNodeMath'); div.operation = 'DIVIDE'; div.inputs[1].default_value = dist
    nt.links.new(cd.outputs['View Distance'], div.inputs[0])
    neg = nt.nodes.new('ShaderNodeMath'); neg.operation = 'MULTIPLY'; neg.inputs[1].default_value = -1.0
    nt.links.new(div.outputs[0], neg.inputs[0])
    ex = nt.nodes.new('ShaderNodeMath'); ex.operation = 'EXPONENT'
    nt.links.new(neg.outputs[0], ex.inputs[0])
    fac = nt.nodes.new('ShaderNodeMath'); fac.operation = 'SUBTRACT'; fac.inputs[0].default_value = 1.0
    nt.links.new(ex.outputs[0], fac.inputs[1])
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = power
    nt.links.new(fac.outputs[0], pw.inputs[0])
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs[0].default_value = (*(col or HAZE_COL), 1); em.inputs[1].default_value = 1.0
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(pw.outputs[0], mix.inputs[0]); nt.links.new(src, mix.inputs[1]); nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    return mat


# =========================================================================== blade
def _airfoil(n=48, t=0.18, camber=0.02):
    """NACA-4-like airfoil, chord 0..1 along x (LE at 0), returns closed loop (upper TE->LE->lower TE)."""
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    p = 0.4
    yc = np.where(x < p, camber / p ** 2 * (2 * p * x - x ** 2), camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    up = np.stack([x, yc + yt], 1)[::-1]
    lo = np.stack([x, yc - yt], 1)[1:]
    return np.vstack([up, lo])


def blade_mesh(name, L=BLADE_L, mat=None, nspan=70, nsec=96):
    """Lofted blade along +Z (root at z=0). Chord in X (LE toward -X = direction of travel), thickness along Y."""
    verts = []
    s = np.linspace(0, 1, nspan) ** 1.15
    for k, u in enumerate(s):
        r = u * L
        # planform
        if u < 0.22:
            w = sb.smooth(u / 0.22)
            chord = sb.lerp(3.4, 5.4, w)
            tc = sb.lerp(1.0, 0.36, w)
        else:
            v = (u - 0.22) / 0.78
            chord = sb.lerp(5.4, 0.9, v ** 0.85) * (1 - 0.35 * max(0, (v - 0.9) / 0.1) ** 2)
            tc = sb.lerp(0.36, 0.15, min(1, v / 0.7) ** 0.8)
        twist = math.radians(sb.lerp(13.0, -1.5, sb.smooth(sb.remap(u, 0.18, 1.0))))
        prebend = 3.2 * u ** 2.2                      # tip bends upwind (+Y)
        af = _airfoil(nsec // 2 + 1, t=min(tc, 0.40), camber=0.025 * sb.smooth(sb.remap(u, 0.15, 0.4)))
        # circle blend at the root (circle centred on the pitch axis at 30% chord)
        N = len(af)
        ph = 2 * np.pi * np.arange(N) / N
        circ = np.stack([0.3 + 0.5 * np.cos(ph), 0.5 * np.sin(ph)], 1)
        wr = sb.smooth(sb.remap(u, 0.03, 0.2))
        prof = circ * (1 - wr) + af * wr
        sx = sy = sb.lerp(3.4, chord, wr) if u < 0.22 else chord
        x = (prof[:, 0] - 0.3) * sx          # pitch axis at 30% chord
        y = prof[:, 1] * sy
        c, sn = math.cos(twist), math.sin(twist)
        xr = x * c - y * sn; yr = x * sn + y * c
        for i in range(len(prof)):
            verts.append((xr[i], yr[i] + prebend, r))
    m = len(verts) // nspan
    faces = []
    for k in range(nspan - 1):
        for i in range(m):
            i1 = (i + 1) % m
            faces.append((k * m + i, k * m + i1, (k + 1) * m + i1, (k + 1) * m + i))
    # tip cap
    tip = len(verts); verts.append((0.0, 3.2, L + 0.15))
    for i in range(m):
        faces.append(((nspan - 1) * m + i, (nspan - 1) * m + (i + 1) % m, tip))
    o = E.mesh_np(name, verts, [tuple(f) for f in faces], mat)
    sb.recalc_normals(o)
    return o


# =========================================================================== materials
def turbine_mats():
    white = sb.mat("TurbWhite", (0.80, 0.81, 0.82), rough=0.32, coat=0.25, coat_rough=0.15)
    nb = sb.NB(white)
    co = nb.coord('Object')
    n = nb.noise(co, scale=0.08, detail=6, rough=0.6)
    streak = nb.noise(nb.mapping(co, scale=(3.0, 3.0, 0.05)), scale=1.2, detail=5, rough=0.6)
    dirt = nb.maprange(streak.outputs['Fac'], 0.55, 0.8, 0.0, 0.22)
    # nacelle GRP panelling (turbine-local metres; only above z > HUB_H - 10 so the rotor, modelled about its own
    # origin, is untouched): transverse panel joints every 2.25 m, the upper/lower cover split line, roof panel
    # joints, rivet rows beside the joints; darker sealant, slight recess, grime collecting under the joints
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    on = nb.math('GREATER_THAN', sep.outputs[2], HUB_H - 10.0)
    fy = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', nb.math('ADD', sep.outputs[1], 0.3), 2.25)), 0.5))
    j_y = nb.maprange(fy, 0.4955, 0.4985)
    j_z = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', sep.outputs[2], HUB_H - 0.6)), 0.016, 0.008)
    j_x = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('ABSOLUTE', sep.outputs[0]), 1.3)), 0.014, 0.007),
                  nb.math('GREATER_THAN', sep.outputs[2], HUB_H + 3.0))
    joint = nb.math('MULTIPLY', nb.math('MAXIMUM', nb.math('MAXIMUM', j_y, j_z), j_x), on)
    # rivets: dots every 0.18 m along a row 0.05 m either side of each transverse joint
    ry = nb.maprange(fy, 0.4745, 0.4775, 0.0, 1.0)
    ry = nb.math('MULTIPLY', ry, nb.maprange(fy, 0.4805, 0.4775, 0.0, 1.0))
    rz = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', nb.math('ADD', sep.outputs[2], sep.outputs[0]), 0.18)), 0.5))
    rivet = nb.math('MULTIPLY', nb.math('MULTIPLY', ry, nb.maprange(rz, 0.06, 0.03)), on)
    under = nb.math('MULTIPLY', nb.maprange(nb.math('SUBTRACT', HUB_H - 0.6, sep.outputs[2]), 0.0, 1.2, 1.0, 0.0),
                    nb.math('LESS_THAN', sep.outputs[2], HUB_H - 0.6))
    dirt = nb.math('MAXIMUM', dirt, nb.math('MULTIPLY', nb.math('MULTIPLY', under, on), nb.math('MULTIPLY', streak.outputs['Fac'], 0.3)))
    col = nb.mix(dirt, nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (0.74, 0.75, 0.76, 1), (0.82, 0.83, 0.84, 1)), (0.55, 0.55, 0.53, 1))
    col = nb.mix(nb.math('MULTIPLY', joint, 0.85), col, (0.30, 0.31, 0.32, 1))
    col = nb.mix(nb.math('MULTIPLY', rivet, 0.5), col, (0.62, 0.63, 0.64, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(joint, nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.28, 0.42), 0.6, dtype='FLOAT'))
    nb.set('Normal', nb.bump(nb.math('ADD', nb.math('MULTIPLY', joint, -1.0), nb.math('MULTIPLY', rivet, 0.6)), strength=0.35, distance=0.01))
    blade = sb.mat("BladeWhite", (0.82, 0.83, 0.84), rough=0.3, coat=0.3, coat_rough=0.12)
    nb = sb.NB(blade)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    # leading-edge erosion/dirt band near the LE (x < -1.2 local) toward the outer span
    le = nb.math('MULTIPLY', nb.maprange(sep.outputs[0], -1.9, -1.0, 1.0, 0.0), nb.maprange(sep.outputs[2], 30, 70, 0.0, 1.0))
    n = nb.noise(co, scale=0.6, detail=6)
    le = nb.math('MULTIPLY', le, nb.maprange(n.outputs['Fac'], 0.35, 0.65, 0.2, 1.0))
    nb.set('Base Color', nb.mix(nb.math('MULTIPLY', le, 0.45), (0.82, 0.83, 0.84, 1), (0.50, 0.49, 0.46, 1)))
    nb.set('Roughness', nb.mix(le, 0.28, 0.6, dtype='FLOAT'))
    yellow = E.painted_steel("TPYellow", (0.80, 0.55, 0.04), rough=0.45, wear=0.35, scale=0.3)
    grey = E.painted_steel("TowerGrey", (0.66, 0.67, 0.68), rough=0.4, wear=0.15, scale=0.05)
    # tower can welds: circumferential bead every 3.2 m (plate width) + a longitudinal seam; faint rust weep below
    nb = sb.NB(grey)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    fz = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[2], 3.2)), 0.5))
    weld = nb.maprange(fz, 0.4955, 0.4985)
    ang = nb.math('ARCTAN2', sep.outputs[1], sep.outputs[0])
    lw = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', ang, 2.2)), 0.004, 0.002),
                 nb.math('GREATER_THAN', sep.outputs[2], 17.6))
    bead = nb.math('MAXIMUM', weld, lw)
    bsdf = [n_ for n_ in grey.node_tree.nodes if n_.type == 'BSDF_PRINCIPLED'][0]
    prev = bsdf.inputs['Normal'].links[0].from_socket if bsdf.inputs['Normal'].is_linked else None
    nb.link(nb.bump(bead, strength=0.5, distance=0.006, normal=prev), bsdf.inputs['Normal'])
    dark = E.painted_steel("DarkSteel", (0.05, 0.05, 0.055), rough=0.5, wear=0.2, scale=0.3)
    galv = E.painted_steel("Galv", (0.45, 0.46, 0.47), rough=0.45, wear=0.2, scale=0.5)
    return dict(white=white, blade=blade, yellow=yellow, grey=grey, dark=dark, galv=galv)


# =========================================================================== turbine
def turbine_parts(M, detail=True):
    """Build master meshes once (hidden). Returns dict of source objects in turbine-local coords:
    tower/TP/nacelle (static), rotor (hub+spinner+3 blades, rotating about local Y at the hub centre)."""
    parts = {}
    coll = bpy.data.collections.new("TurbineSrc")
    bpy.context.scene.collection.children.link(coll)
    def put(o):
        for c in o.users_collection:
            c.objects.unlink(o)
        coll.objects.link(o)
        return o
    static = []
    # monopile + transition piece (yellow) + platform + boat landing
    static.append(E.lathe_obj("MP", [(3.4, -18.0), (3.4, 2.0), (0.0, 2.0)], segs=96, mat=M['dark']))
    static.append(E.lathe_obj("TP", [(3.55, 0.0), (3.55, 17.0), (3.3, 17.6), (0.0, 17.6)], segs=96, mat=M['yellow']))
    static.append(E.lathe_obj("Tower", [(3.2, 17.6), (2.15, HUB_H - 4.0), (0.0, HUB_H - 4.0)], segs=96, mat=M['grey']))
    # tower flanges (bolted section joints)
    for z in (17.6, 45.0, 75.0):
        rr = 3.2 + (2.15 - 3.2) * (z - 17.6) / (HUB_H - 4 - 17.6)
        static.append(E.lathe_obj(f"TFl{z}", [(rr + 0.02, z - 0.08), (rr + 0.06, z - 0.06), (rr + 0.06, z + 0.06), (rr + 0.02, z + 0.08)], segs=96, mat=M['grey']))
    if detail:
        pl = E.lathe_obj("Platform", [(3.5, 16.8), (7.2, 16.8), (7.2, 17.1), (3.5, 17.1)], segs=96, mat=M['galv'])
        static.append(pl)
        for k in range(40):   # railing posts + two rails
            a = 2 * math.pi * k / 40
            static.append(E.cyl(f"RP{k}", 0.03, 1.1, loc=(7.1 * math.cos(a), 7.1 * math.sin(a), 17.65), mat=M['yellow'], verts=8))
        for hz in (17.65, 18.2):
            static.append(E.sweep_planar(f"Rail{hz}", np.array([(7.1 * math.cos(2 * math.pi * i / 120), 7.1 * math.sin(2 * math.pi * i / 120), hz) for i in range(120)]),
                                         [(0.03 * math.cos(t), 0.03 * math.sin(t)) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)],
                                         normal=(0, 0, 1), closed=True, mat=M['yellow']))
        # boat landing on the -X side: two vertical fender tubes + ladder
        for dy in (-1.0, 1.0):
            static.append(E.cyl(f"BL{dy}", 0.3, 16.0, loc=(-4.6, dy, 6.0), mat=M['yellow'], verts=24))
            for z in (0.0, 5.0, 11.0):
                static.append(E.box(f"BLa{dy}{z}", (1.2, 0.25, 0.25), loc=(-4.0, dy, z), mat=M['yellow']))
        for z in np.arange(-1.0, 16.5, 0.35):
            static.append(E.box(f"Rung{z:.2f}", (0.06, 1.5, 0.06), loc=(-4.6, 0, z), mat=M['galv']))
        # access door
        static.append(E.box("Door", (0.25, 1.1, 2.1), loc=(-3.2, 0, 18.7), mat=M['dark']))
    # nacelle: rounded box, hub at +Y end
    nac = sb.prim("cube", "Nacelle", loc=(0, -8.0, HUB_H + 0.2), scale=(3.6, 9.0, 3.6), mat=M['white'])
    E.apply_scale(nac); mb = sb.bevel(nac, 1.1, 5); mb.harden_normals = False
    static.append(nac)
    static.append(E.cyl("YawBase", 2.3, 1.5, loc=(0, -5.0, HUB_H - 3.6), mat=M['white'], verts=64))
    if detail:
        # rear top: cooler (radiator block) + helihoist platform with rails + met mast
        static.append(E.box("Cooler", (6.2, 2.5, 3.2), loc=(0, -15.5, HUB_H + 5.2), mat=M['white'], bev=0.25))
        for k in range(14):
            static.append(E.box(f"Fin{k}", (6.0, 0.05, 2.8), loc=(0, -16.72 + k * 0.001, HUB_H + 5.2), mat=M['galv']))
        static.append(E.box("HeliDeck", (6.8, 7.5, 0.2), loc=(0, -10.5, HUB_H + 3.9), mat=M['galv']))
        for (x, y, sx_, sy_) in ((3.35, -10.5, 0.06, 7.5), (-3.35, -10.5, 0.06, 7.5), (0, -6.8, 6.8, 0.06)):
            for hz in (0.6, 1.2):
                static.append(E.box(f"HR{x}{y}{hz}", (sx_, sy_, 0.06), loc=(x, y, HUB_H + 3.9 + hz), mat=M['yellow']))
        # side louvre banks (rear flanks), service hatches, roof grab rails + tie-off points, aviation lights
        for sx in (-1, 1):
            for k in range(9):
                static.append(E.box(f"Louv{sx}{k}", (0.12, 2.6, 0.05), loc=(sx * 3.62, -13.2, HUB_H + 0.6 + k * 0.2),
                                    rot=(0, sx * math.radians(35), 0), mat=M['galv']))
            static.append(E.box(f"LouvFr{sx}", (0.06, 2.8, 2.0), loc=(sx * 3.585, -13.2, HUB_H + 1.4), mat=M['dark']))
            for (hy, hz, hw, hh) in ((-4.2, HUB_H + 0.7, 1.4, 1.6), (-8.6, HUB_H - 1.6, 1.0, 0.8)):
                static.append(E.box(f"Hatch{sx}{hy}", (0.03, hw, hh), loc=(sx * 3.61, hy, hz), mat=M['white'], bev=0.02))
                static.append(E.box(f"HatchL{sx}{hy}", (0.05, 0.06, 0.18), loc=(sx * 3.64, hy + hw / 2 - 0.12, hz), mat=M['dark']))
            static.append(E.box(f"GrabR{sx}", (0.05, 12.0, 0.05), loc=(sx * 2.9, -7.5, HUB_H + 4.1), mat=M['yellow']))
            for y in np.arange(-13.0, -1.0, 1.5):
                static.append(E.box(f"GrabP{sx}{y:.1f}", (0.05, 0.05, 0.3), loc=(sx * 2.9, y, HUB_H + 3.95), mat=M['yellow']))
        for sx in (-1, 1):
            static.append(E.cyl(f"AvLt{sx}", 0.18, 0.35, loc=(sx * 2.6, -16.2, HUB_H + 7.0), mat=M['dark'], verts=16))
            static.append(E.lathe_obj(f"AvDome{sx}", [(0.0, 0.0), (0.16, 0.0), (0.14, 0.12), (0.0, 0.2)], segs=24,
                                      mat=sb.emit_mat("AvRed", (1.0, 0.08, 0.04), 30.0), loc=(sx * 2.6, -16.2, HUB_H + 7.17)))
        static.append(E.cyl("MetMast", 0.06, 3.0, loc=(1.5, -17.2, HUB_H + 8.2), mat=M['galv'], verts=8))
        static.append(E.cyl("Anemo", 0.25, 0.1, loc=(1.5, -17.2, HUB_H + 9.7), mat=M['dark'], verts=12))
    for o in static:
        put(o)
    st = sb.join(static, "TurbStatic") if len(static) > 1 else static[0]
    parts['static'] = st
    # rotor about local origin (hub centre), axis +Y
    rot = []
    hub = E.lathe_obj("Hub", [(0.0, -2.6), (3.6, -2.6), (4.0, -1.6), (4.1, -0.4), (4.0, 0.9), (3.6, 2.4), (2.7, 3.9), (1.5, 4.9), (0.0, 5.3)], segs=128, mat=M['white'])
    hub.rotation_euler = (-math.pi / 2, 0, 0)       # lathe axis Z -> Y
    rot.append(hub)
    # spinner details: nose ring, three radial panel seams, blade-root rain collars
    seam_m = sb.mat("Seam", (0.30, 0.31, 0.32), rough=0.5)
    nose = E.lathe_obj("NoseRing", [(0.55, 0.0), (0.62, 0.0), (0.62, 0.05), (0.55, 0.05)], segs=64, mat=seam_m)
    nose.rotation_euler = (-math.pi / 2, 0, 0); nose.location = (0, 5.18, 0)
    rot.append(nose)
    prof_h = [(0.0, -2.6), (3.6, -2.6), (4.0, -1.6), (4.1, -0.4), (4.0, 0.9), (3.6, 2.4), (2.7, 3.9), (1.5, 4.9), (0.0, 5.3)]
    pr = np.array(prof_h)
    for k in range(3):
        a = 2 * math.pi * k / 3 + math.pi / 3
        pts = []
        for zz in np.linspace(-2.4, 5.1, 60):
            rr = np.interp(zz, pr[:, 1], pr[:, 0]) + 0.012
            pts.append((rr * math.sin(a), zz, rr * math.cos(a)))
        rot.append(E.sweep(f"SpSeam{k}", pts, radius=0.018, segs=6, mat=seam_m, sub=1, resample=False))
    # fasteners: countersunk bolt heads either side of each radial seam + round the nose hatch
    bolt_m = sb.mat("SpBolt", (0.52, 0.53, 0.54), metal=0.8, rough=0.35)
    bsrc = []
    for k in range(3):
        a0 = 2 * math.pi * k / 3 + math.pi / 3
        for zz in np.arange(-2.2, 4.9, 0.32):
            rr = float(np.interp(zz, pr[:, 1], pr[:, 0])) + 0.004
            for da in (-0.07, 0.07):
                a = a0 + da / max(rr, 0.5)
                bsrc.append(sb.prim("sphere", "SpB", loc=(rr * math.sin(a), zz, rr * math.cos(a)), segments=8, ring_count=4,
                                    radius=0.032, scale=(1, 1, 1), mat=bolt_m))
    for i in range(16):
        a = 2 * math.pi * i / 16
        bsrc.append(sb.prim("sphere", "NoseB", loc=(0.78 * math.sin(a), 5.12 - 0.02, 0.78 * math.cos(a)), segments=8, ring_count=4,
                            radius=0.03, mat=bolt_m))
    for b_ in bsrc:
        b_.scale = (1.0, 0.35, 1.0) if b_.name.startswith("Nose") else (1.0, 1.0, 1.0)
    rot.extend(bsrc)
    hatch = E.lathe_obj("NoseHatch", [(0.0, 0.0), (0.9, 0.0), (0.9, 0.012), (0.0, 0.012)], segs=64, mat=seam_m)
    hatch.rotation_euler = (-math.pi / 2, 0, 0); hatch.location = (0, 5.0, 0)
    for zz in (0.9, 3.2):
        rr = float(np.interp(zz, pr[:, 1], pr[:, 0])) + 0.01
        rg = E.lathe_obj(f"SpRing{zz}", [(rr, -0.02), (rr + 0.02, 0.0), (rr, 0.02)], segs=128, mat=seam_m)
        rg.rotation_euler = (-math.pi / 2, 0, 0); rg.location = (0, zz, 0)
        rot.append(rg)
    for k in range(3):
        a = 2 * math.pi * k / 3
        col = E.lathe_obj(f"Collar{k}", [(1.72, 0.0), (1.95, 0.05), (1.95, 0.25), (1.72, 0.3)], segs=96, mat=M['white'])
        col.rotation_euler = (0, a, 0)
        col.location = V((0, -0.4, 0)) + V((math.sin(a), 0, math.cos(a))) * 3.6
        rot.append(col)
    root_m = E.painted_steel("BladeRoot", (0.42, 0.43, 0.44), rough=0.45, wear=0.2, scale=2.0)
    for k in range(3):
        a = 2 * math.pi * k / 3
        rc = E.lathe_obj(f"RootCyl{k}", [(1.745, 0.0), (1.745, 1.1), (1.70, 1.2), (0.0, 1.2)], segs=96, mat=root_m)
        rc.rotation_euler = (0, a, 0)
        rc.location = V((0, -0.4, 0)) + V((math.sin(a), 0, math.cos(a))) * 3.3
        rot.append(rc)
    blade = blade_mesh("Blade", mat=M['blade'])
    for k in range(3):
        b = blade.copy(); b.data = blade.data
        sb.link_obj(b)
        a = 2 * math.pi * k / 3
        b.rotation_euler = (0, a, 0)
        b.location = (0, 0.4, 0)
        # blade root starts at the hub surface
        b.location = V((0, -0.4, 0)) + V((math.sin(a), 0, math.cos(a))) * 3.2
        rot.append(b)
    bpy.data.objects.remove(blade)
    for o in rot:
        put(o)
    # apply transforms and join into one rotor mesh (origin at hub centre)
    for o in rot:
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
    for o in rot:
        o.data = o.data.copy()
        o.data.transform(o.matrix_basis)
        o.matrix_basis = Matrix.Identity(4)
    rt = sb.join(rot, "Rotor")
    parts['rotor'] = rt
    coll.hide_render = True
    coll.hide_viewport = True
    parts['coll'] = coll
    return parts


def place_turbine(parts, loc, yaw=0.0, phase=0.0, rpm=RPM, f0=0, f1=100, name="T"):
    """Linked instance of a turbine at loc (sea level), rotor spinning clockwise seen from upwind (+Y)."""
    root = sb.empty(name, loc=loc)
    root.rotation_euler = (0, 0, yaw)
    st = bpy.data.objects.new(name + "S", parts['static'].data); sb.link_obj(st); st.parent = root
    ro = bpy.data.objects.new(name + "R", parts['rotor'].data); sb.link_obj(ro); ro.parent = root
    ro.location = (0, 0.9, HUB_H + 0.2)
    ro.rotation_mode = 'XYZ'
    w = rpm * 2 * math.pi / 60.0 / 24.0          # rad per frame
    for f in (f0 - 2, f1 + 2):
        ro.rotation_euler = (0, phase - w * f, 0)     # negative about +Y = clockwise seen from upwind (+Y)
        ro.keyframe_insert("rotation_euler", frame=f)
    try:
        for fc in ro.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
    except Exception:
        pass
    return root, ro


# =========================================================================== vessel
def ctv(M, name="CTV"):
    """~26 m crew-transfer catamaran: two slender hulls, deck, wheelhouse, bow fender. Bow toward +Y."""
    root = sb.empty(name)
    white = sb.mat("HullWhite", (0.80, 0.81, 0.82), rough=0.3, coat=0.4, coat_rough=0.1)
    graph = sb.mat("HullGraphite", (0.05, 0.055, 0.06), rough=0.35, coat=0.3)
    glass = sb.mat("WheelGlass", (0.02, 0.025, 0.03), rough=0.05, spec=0.8, coat=1.0, coat_rough=0.0)
    amber = __import__("props").amber_anodized("CTVAmber")
    rub = sb.rubber("Fender", (0.02, 0.02, 0.022))
    L, Bh = 26.0, 1.9
    for sx in (-3.4, 3.4):
        pts, prof = [], []
        n = 40
        verts, faces = [], []
        for i in range(n + 1):
            u = i / n
            y = -L / 2 + u * L
            wdt = Bh * (1 - (max(0, u - 0.7) / 0.3) ** 2 * 0.95) * (1 - (max(0, 0.08 - u) / 0.08) * 0.3)
            dep = 1.6 * (1 - (max(0, u - 0.75) / 0.25) ** 1.5 * 0.7)
            ring = [(-wdt / 2, 1.2), (-wdt / 2, -0.2), (-wdt * 0.35, -dep * 0.8), (0, -dep), (wdt * 0.35, -dep * 0.8), (wdt / 2, -0.2), (wdt / 2, 1.2)]
            for (x, z) in ring:
                verts.append((sx + x, y, z))
        m = 7
        for i in range(n):
            for j in range(m - 1):
                faces.append((i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j))
        faces.append(tuple(range(0, m))[::-1])
        faces.append(tuple(range(n * m, n * m + m)))
        h = E.mesh_np(name + f"Hull{sx}", verts, [tuple(f) for f in faces], None)
        sb.recalc_normals(h)
        h.data.materials.append(white); h.data.materials.append(graph)
        # graphite below the rubbing strake
        for p in h.data.polygons:
            zc = sum(h.data.vertices[v].co.z for v in p.vertices) / len(p.vertices)
            p.material_index = 1 if zc < 0.25 else 0
        sb.shade_auto(h, 40)
        h.parent = root
        E.box(name + f"Strake{sx}", (0.12, L * 0.85, 0.12), loc=(sx + (1 if sx > 0 else -1) * Bh / 2, -0.8, 0.35), mat=amber).parent = root
    E.box(name + "Deck", (8.8, L * 0.78, 0.35), loc=(0, -1.8, 1.35), mat=white, bev=0.1).parent = root
    E.box(name + "Cabin", (6.6, 8.0, 2.4), loc=(0, 2.2, 2.7), mat=white, bev=0.25).parent = root
    E.box(name + "Wind", (6.2, 7.8, 0.9), loc=(0, 2.25, 3.35), mat=glass, bev=0.05).parent = root
    E.box(name + "Roof", (7.0, 8.6, 0.25), loc=(0, 2.0, 4.0), mat=graph, bev=0.08).parent = root
    E.cyl(name + "Mast", 0.08, 2.5, loc=(0, 0.5, 5.3), mat=graph, verts=8).parent = root
    E.box(name + "Bow", (7.6, 0.8, 0.9), loc=(0, L / 2 - 0.2, 1.3), mat=rub, bev=0.3).parent = root
    for k in range(3):
        E.box(name + f"Cargo{k}", (1.6, 1.6, 1.3), loc=(-2 + k * 2.0, -8.5, 2.2), mat=E.painted_steel(f"Box{k}", (0.12, 0.3, 0.45) if k != 1 else (0.55, 0.40, 0.2), rough=0.5)).parent = root
    return root


# =========================================================================== ocean
def ocean_material(name, wake=None, piles=None, haze_dist=9000.0):
    """Sea surface: deep blue-green body, sky reflection (Fresnel via principled IOR 1.33), capillary normal
    detail, whitecap foam from the Ocean modifier 'foam' attribute, vessel wake + monopile foam masks.
    wake: dict(pos_socket_values...) animated via returned value nodes; piles: (spacing_x, spacing_y, off_x, off_y)."""
    m = sb.mat(name, (0.006, 0.030, 0.042), rough=0.06, spec=0.5, ior=1.333)
    nb = sb.NB(m)
    b = nb.bsdf
    b.inputs['IOR'].default_value = 1.333
    geo = nb.new('ShaderNodeNewGeometry')
    wp = geo.outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(wp, sep.inputs[0])
    tval = nb.new('ShaderNodeValue'); tval.name = "Time"
    # capillary ripples + mid chop (world coords, drifting downwind -Y)
    drift = nb.new('ShaderNodeCombineXYZ'); nb.link(nb.math('MULTIPLY', tval.outputs[0], -1.2), drift.inputs[1])
    wpd = nb.vmath('ADD', wp, drift.outputs[0])
    n1 = nb.noise(nb.mapping(wpd, scale=(1.0, 1.6, 1.0)), scale=0.35, detail=6, rough=0.62, dist=0.3)
    n2 = nb.noise(nb.mapping(wpd, scale=(1.0, 1.4, 1.0)), scale=0.06, detail=4, rough=0.6)
    big = nb.noise(wp, scale=0.0018, detail=3, rough=0.5)             # km-scale variation (wind slicks)
    h = nb.math('ADD', n1.outputs['Fac'], nb.math('MULTIPLY', n2.outputs['Fac'], 1.5))
    slick = nb.maprange(big.outputs['Fac'], 0.42, 0.62, 0.35, 1.0)
    bmp = nb.new('ShaderNodeBump'); bmp.inputs['Distance'].default_value = 0.4
    nb.link(nb.math('MULTIPLY', slick, 0.35), bmp.inputs['Strength']); nb.link(h, bmp.inputs['Height'])
    nb.set('Normal', bmp.outputs[0])
    # body colour: slightly greener in the troughs / lighter where the foam attribute is dense
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "foam"; at.attribute_type = 'GEOMETRY'
    fn = nb.noise(wp, scale=0.9, detail=5, rough=0.65)
    # Ocean-modifier foam (stored inverted in the byte-colour layer; alpha = 0 where the layer is absent)
    fa = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, at.outputs['Fac']), at.outputs['Alpha'])
    fine_f = nb.noise(wp, scale=0.35, detail=8, rough=0.7, dist=0.6)
    foam = nb.math('MULTIPLY', nb.maprange(fa, 0.35, 0.95), nb.maprange(fine_f.outputs['Fac'], 0.50, 0.64))
    foam = nb.math('MULTIPLY', foam, 0.55)
    extra = []
    if piles is not None:
        sx, sy, ox, oy = piles
        fx = nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', nb.math('SUBTRACT', sep.outputs[0], ox - sx / 2), sx)), 0.5)
        fy = nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', nb.math('SUBTRACT', sep.outputs[1], oy - sy / 2), sy)), 0.5)
        d = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', nb.math('MULTIPLY', fx, sx), nb.math('MULTIPLY', fx, sx)),
                                     nb.math('MULTIPLY', nb.math('MULTIPLY', fy, sy), nb.math('MULTIPLY', fy, sy))))
        ring = nb.math('MULTIPLY', nb.maprange(d, 3.4, 4.2, 1.0, 0.0), nb.maprange(d, 3.3, 3.45, 0.0, 1.0))
        halo = nb.maprange(d, 3.5, 14.0, 0.5, 0.0)
        pf = nb.math('MULTIPLY', nb.math('ADD', ring, halo), nb.maprange(fn.outputs['Fac'], 0.35, 0.6, 0.2, 1.0))
        foam = nb.math('MAXIMUM', foam, pf)
    wake_nodes = None
    if wake is not None:
        bx = nb.new('ShaderNodeValue'); bx.name = "BoatX"
        by = nb.new('ShaderNodeValue'); by.name = "BoatY"
        dirx, diry = wake['dir']
        # along = distance behind the boat (m), lat = lateral offset
        rx = nb.math('SUBTRACT', sep.outputs[0], bx.outputs[0]); ry = nb.math('SUBTRACT', sep.outputs[1], by.outputs[0])
        along = nb.math('MULTIPLY', nb.math('ADD', nb.math('MULTIPLY', rx, dirx), nb.math('MULTIPLY', ry, diry)), -1.0)
        lat = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('MULTIPLY', rx, diry), nb.math('MULTIPLY', ry, dirx)))
        behind = nb.maprange(along, -12.0, 2.0, 0.0, 1.0)
        width = nb.math('ADD', 4.5, nb.math('MULTIPLY', along, 0.05))
        core = nb.math('EXPONENT', nb.math('MULTIPLY', -1.0, nb.math('POWER', nb.math('DIVIDE', lat, width), 2.0)))
        decay = nb.math('EXPONENT', nb.math('DIVIDE', nb.math('MAXIMUM', along, 0.0), -260.0))
        wn = nb.noise(nb.mapping(wp, scale=(1.0, 1.0, 1.0)), scale=0.25, detail=6, rough=0.7, dist=0.5)
        wash = nb.math('MULTIPLY', nb.math('MULTIPLY', core, decay), nb.maprange(wn.outputs['Fac'], 0.3, 0.62, 0.15, 1.2))
        # twin hull wakes merge: two lobes near the stern
        # Kelvin arms at ~19.5 deg
        armd = nb.math('ABSOLUTE', nb.math('SUBTRACT', lat, nb.math('MULTIPLY', nb.math('MAXIMUM', along, 0.0), math.tan(math.radians(19.5)))))
        aw = nb.math('ADD', 1.2, nb.math('MULTIPLY', along, 0.03))
        arm = nb.math('EXPONENT', nb.math('MULTIPLY', -1.0, nb.math('POWER', nb.math('DIVIDE', armd, aw), 2.0)))
        arm = nb.math('MULTIPLY', arm, nb.math('EXPONENT', nb.math('DIVIDE', nb.math('MAXIMUM', along, 0.0), -140.0)))
        armf = nb.math('MULTIPLY', arm, nb.maprange(wn.outputs['Fac'], 0.45, 0.62, 0.0, 0.7))
        wk = nb.math('MULTIPLY', nb.math('MAXIMUM', wash, armf), behind)
        foam = nb.math('MAXIMUM', foam, nb.math('MINIMUM', wk, 1.0))
        wake_nodes = (bx, by)
    foam = nb.math('MINIMUM', foam, 1.0)
    nb.set('Base Color', nb.mix(foam, (0.004, 0.022, 0.034, 1), (0.74, 0.77, 0.78, 1)))
    # distance-based roughness: stands in for sub-pixel wave facets so the displaced patch and far plane match
    cd = nb.new('ShaderNodeCameraData')
    dr = nb.maprange(cd.outputs['View Distance'], 300.0, 4500.0, 0.05, 0.16)
    nb.set('Roughness', nb.mix(foam, dr, 0.75, dtype='FLOAT'))
    b.inputs['Subsurface Weight'].default_value = 0.0
    aerial(m, dist=haze_dist)
    return m, tval, wake_nodes


def nb_float(nb, sock, mul):
    return nb.math('MULTIPLY', sock, mul)
