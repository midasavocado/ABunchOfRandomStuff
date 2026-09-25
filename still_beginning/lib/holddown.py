"""Hero hold-down clamp (pad side) for s19c close-up and every pad view (s19a/b, s20).

A fabricated, heavily used piece of launch hardware: welded box pedestal with chamfered corners, gussets and a
bolted base plate; a forked, flame-cut jaw (two thick tines + cross webs) pivoting on a greased pin in clevis
brackets; bronze bearing liners in the hook notches (they wrap the vehicle's lug pin -> the film's warm curved
highlight, arising from real hardware); a tie-rod hydraulic release cylinder with clevises and braided hoses; the
pneumatic lock-bolt housing on its own gusseted stand-off. Paint is graphite with launch soot, deluge streaks,
chipped edges; fasteners are zinc-plated. Local frame = launchpad.clamp(): +X radial outward, Y tangential, Z up.
"""
import bpy, bmesh, math
from mathutils import Vector as V, Matrix
import sb
import rocket

_M = {}


def materials():
    if _M:
        return _M
    _M["paint"] = sb.painted("HD_Paint", (0.075, 0.078, 0.082), rough=0.5, coat=0.05, grime=0.55, wear=0.55,
                             streaks=0.9, scale=5.0, under=(0.42, 0.41, 0.40))
    _M["paint_jaw"] = sb.painted("HD_JawPaint", (0.085, 0.087, 0.09), rough=0.46, coat=0.05, grime=0.45, wear=0.7,
                                 streaks=0.7, scale=6.0, under=(0.48, 0.47, 0.46))
    _M["base"] = sb.painted("HD_BasePlate", (0.12, 0.12, 0.125), rough=0.6, grime=0.8, wear=0.4, streaks=0.4, scale=4.0)
    _M["bronze"] = bronze()
    _M["steel"] = sb.brushed_metal("HD_Steel", (0.60, 0.60, 0.61), rough=0.22, aniso=0.5, scale=60, bump=0.06, scratches=0.4)
    _M["zinc"] = zinc()
    _M["chrome"] = sb.mat("HD_Chrome", (0.86, 0.86, 0.87), metal=1.0, rough=0.06)
    _M["hose"] = braided_hose()
    _M["rubber"] = sb.rubber("HD_Rubber", (0.025, 0.025, 0.027), rough=0.55)
    _M["weld"] = weld_mat()
    _M["grease"] = sb.mat("HD_Grease", (0.09, 0.07, 0.035), rough=0.12, spec=0.6, coat=0.6, coat_rough=0.05)
    _M["amber"] = sb.painted("HD_AmberBand", (0.62, 0.25, 0.03), rough=0.38, coat=0.2, grime=0.3, wear=0.6, scale=8.0,
                             under=(0.4, 0.4, 0.41))
    return _M


def bronze():
    m = sb.brushed_metal("HD_Bronze", (0.80, 0.50, 0.24), rough=0.24, aniso=0.3, scale=90, bump=0.04, scratches=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    # darker oxidised patches away from the rubbing surfaces
    ox = nb.maprange(nb.noise(co, scale=14, detail=6, rough=0.6).outputs['Fac'], 0.5, 0.7)
    nb.set('Base Color', nb.mix(ox, (0.80, 0.50, 0.24, 1), (0.42, 0.26, 0.12, 1)))
    return m


def zinc():
    """Yellow-zinc passivated fasteners: iridescent gold/olive tint, varies per bolt."""
    m = sb.mat("HD_Zinc", (0.72, 0.64, 0.42), metal=1.0, rough=0.3, thin_film=380.0)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=60, detail=4, rough=0.6)
    nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (0.62, 0.58, 0.44, 1), (0.78, 0.66, 0.38, 1)))
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.22, 0.42))
    return m


def braided_hose():
    """Stainless over-braid hydraulic hose: a diamond weave pattern in UV (bevel curve UVs), metallic, dirty."""
    m = sb.mat("HD_Braid", (0.55, 0.55, 0.56), metal=1.0, rough=0.38)
    nb = sb.NB(m)
    uv = nb.coord('UV')
    sep = nb.new('ShaderNodeSeparateXYZ')
    nb.link(uv, sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    a = nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', nb.math('MULTIPLY', u, 900.0), nb.math('MULTIPLY', v, 40.0)), 1.0))
    b = nb.math('SINE', nb.math('MULTIPLY', nb.math('SUBTRACT', nb.math('MULTIPLY', u, 900.0), nb.math('MULTIPLY', v, 40.0)), 1.0))
    h = nb.math('MAXIMUM', a, b)
    nb.set('Normal', nb.bump(h, strength=0.5, distance=0.0015))
    dirt = nb.maprange(nb.noise(nb.coord('Object'), scale=9, detail=6).outputs['Fac'], 0.45, 0.7)
    nb.set('Base Color', nb.mix(dirt, (0.55, 0.55, 0.56, 1), (0.16, 0.15, 0.14, 1)))
    nb.set('Roughness', nb.maprange(dirt, 0, 1, 0.3, 0.7))
    return m


def weld_mat():
    m = sb.mat("HD_Weld", (0.20, 0.19, 0.18), metal=0.6, rough=0.45)
    nb = sb.NB(m)
    co = nb.coord('Object')
    heat = nb.maprange(nb.noise(co, scale=40, detail=4).outputs['Fac'], 0.35, 0.65)
    # heat tint: straw / blue oxide on bead
    nb.set('Base Color', nb.ramp(heat, [(0.0, (0.16, 0.15, 0.14)), (0.5, (0.30, 0.22, 0.12)), (1.0, (0.14, 0.16, 0.22))]))
    return m


# --------------------------------------------------------------------------- geometry helpers

def ext(name, pts, thick, origin, u, v, mat, parent, bevel=0.012, segs=2):
    return rocket.extrude(name, pts, thick, origin, u, v, mat, parent, bevel=bevel, segs=segs)


def chamfer_rect(w, h, c):
    """CCW rectangle (centred) with 45-degree corner chamfers c."""
    x, y = w / 2, h / 2
    return [(-x + c, -y), (x - c, -y), (x, -y + c), (x, y - c), (x - c, y), (-x + c, y), (-x, y - c), (-x, -y + c)]


def hexnut(name, loc, axis, af, h, mat, parent, washer=True, stud=0.0):
    """Hex nut (across-flats af) on axis with washer and optional protruding threaded stud."""
    axis = V(axis).normalized()
    q = axis.to_track_quat('Z', 'Y')
    r = af / math.sqrt(3.0)
    objs = []
    if washer:
        w = sb.prim("cyl", name + "W", vertices=24, radius=af * 0.95, depth=h * 0.18, mat=mat)
        w.rotation_mode = 'QUATERNION'; w.rotation_quaternion = q; w.location = V(loc) + axis * (h * 0.09)
        objs.append(w)
    n = sb.prim("cyl", name, vertices=6, radius=r, depth=h, mat=mat)
    n.rotation_mode = 'QUATERNION'; n.rotation_quaternion = q; n.location = V(loc) + axis * (h * 0.18 + h / 2)
    sb.bevel(n, h * 0.12, 2, angle=30)
    objs.append(n)
    if stud > 0:
        s = sb.prim("cyl", name + "S", vertices=16, radius=af * 0.36, depth=stud, mat=_M.get("steel", mat))
        s.rotation_mode = 'QUATERNION'; s.rotation_quaternion = q; s.location = V(loc) + axis * (h * 1.18 + stud / 2)
        sb.bevel(s, af * 0.05, 1)
        objs.append(s)
    for o in objs:
        o.parent = parent
    return objs


def capscrew(name, loc, axis, d, mat, parent):
    """Socket-head cap screw head (cylinder with hex socket recess implied by a darker inset disc)."""
    axis = V(axis).normalized()
    q = axis.to_track_quat('Z', 'Y')
    h = sb.prim("cyl", name, vertices=20, radius=d * 0.75, depth=d, mat=mat)
    h.rotation_mode = 'QUATERNION'; h.rotation_quaternion = q; h.location = V(loc) + axis * (d / 2)
    sb.bevel(h, d * 0.12, 2, angle=30)
    s = sb.prim("cyl", name + "Sk", vertices=6, radius=d * 0.34, depth=d * 0.3, mat=_M["rubber"])
    s.rotation_mode = 'QUATERNION'; s.rotation_quaternion = q; s.location = V(loc) + axis * (d * 0.9)
    h.parent = s.parent = parent
    return h


def weld_bead(name, pts, r, parent, closed=False, ripple=0.2, pitch=None):
    """Fillet weld bead: tube along a polyline whose radius ripples (weld 'stacked dimes')."""
    pts = [V(p) for p in pts]
    if closed:
        pts = pts + [pts[0]]
    dense = []
    for a, b in zip(pts, pts[1:]):
        L = (b - a).length
        n = max(2, int(L / (r * 0.6)))
        for i in range(n):
            dense.append(a.lerp(b, i / n))
    dense.append(pts[-1])
    p = pitch or r * 1.4
    acc = [0.0]
    for a, b in zip(dense, dense[1:]):
        acc.append(acc[-1] + (b - a).length)
    o = sb.tube_along(name, [tuple(q) for q in dense], radius=r, mat=_M["weld"], bevel_res=2,
                      taper=None)
    sp = o.data.splines[0]
    for i, pt in enumerate(sp.points):
        ph = (acc[i] / p) % 1.0
        pt.radius = 1.0 - ripple * (ph ** 2)
    o.data.use_fill_caps = True
    if parent:
        o.parent = parent
    return o


# --------------------------------------------------------------------------- the clamp

def build(name, M_pad, parent, rs, table_z, zt, lug_pin):
    """Returns the same dict layout as launchpad.clamp()."""
    M = materials()
    X, Y, Z = V((1, 0, 0)), V((0, 1, 0)), V((0, 0, 1))
    g = parent
    col_top = zt - 0.35
    z0 = table_z + 0.12
    # --- base plate (chamfered, 100 mm) with 12 anchor studs
    bx = rs + 0.28
    ext(name + "Base", chamfer_rect(1.36, 1.45, 0.16), 0.1, (bx, 0, table_z + 0.06), X, Y, M["base"], g, bevel=0.01)
    anchors = [(rs + dx, s * 0.62) for dx in (-0.3, 0.0, 0.3, 0.6) for s in (-1, 1)] + [(rs + 0.86, s * 0.34) for s in (-1, 1)]
    for k, (ax, ay) in enumerate(anchors):
        hexnut(name + "Anc%d" % k, V((ax, ay, table_z + 0.11)), Z, 0.075, 0.06, M["zinc"], g, stud=0.035)
    # grout pad under the plate
    ext(name + "Grout", chamfer_rect(1.46, 1.58, 0.2), 0.04, (bx, 0, table_z + 0.0), X, Y, M_pad["concrete"], g, bevel=0.008)
    # --- column: chamfered box section
    col = chamfer_rect(0.78, 0.72, 0.07)
    hc = col_top - z0
    ext(name + "Col", col, hc, (rs, 0, z0 + hc / 2), X, Y, M["paint"], g, bevel=0.008)
    weld_bead(name + "WeldBase", [(rs + x, y, z0 + 0.012) for x, y in [(p[0] * 1.02, p[1] * 1.02) for p in col]], 0.016, g, closed=True)
    # vertical stiffener ribs on the tangential faces and radial faces
    for s in (-1, 1):
        for dx in (-0.2, 0.2):
            ext(name + "RibY", [(-0.0, 0), (0.07, 0), (0.07, hc * 0.72), (0.0, hc * 0.8)], 0.035,
                (rs + dx, s * 0.36, z0), Y * s, Z, M["paint"], g, bevel=0.006)
        # triangular gussets at the base (tangential faces)
        ext(name + "Gus", [(0, 0), (0.28, 0), (0.28, 0.05), (0.04, 0.46), (0, 0.46)], 0.04,
            (rs, s * 0.36, z0), Y * s, Z, M["paint"], g, bevel=0.006)
        weld_bead(name + "WeldGus%d" % (s > 0), [(rs + 0.03, s * 0.365, z0 + 0.46), (rs + 0.03, s * 0.365, z0 + 0.02),
                                                 (rs + 0.03, s * 0.64, z0 + 0.02)], 0.011, g)
    # --- bearing cap (machined) + bronze wear pad under the vehicle shoe
    ext(name + "Cap", chamfer_rect(0.98, 0.84, 0.05), 0.34, (rs - 0.05, 0, zt - 0.23), X, Y, M["paint"], g, bevel=0.012)
    ext(name + "Pad", chamfer_rect(0.86, 0.66, 0.03), 0.06, (rs - 0.05, 0, zt - 0.03), X, Y, M["bronze"], g, bevel=0.006)
    for i in range(4):
        for s in (-1, 1):
            capscrew(name + "PadScr", (rs - 0.42 + i * 0.25, s * 0.375, zt - 0.06), Z, 0.03, M["zinc"], g)
    weld_bead(name + "WeldCap", [(rs + x * 1.0, y * 1.0, col_top - 0.05) for x, y in col], 0.014, g, closed=True)
    # amber anodised index band around the cap (thin, chipped) - the only colour on the clamp
    ext(name + "Band", chamfer_rect(0.995, 0.855, 0.05), 0.035, (rs - 0.05, 0, col_top + 0.0), X, Y, M["amber"], g, bevel=0.004)

    # --- clevis brackets for the jaw pivot
    piv = V((rs + 0.75, 0, zt - 0.95))
    br_pts = [(-0.42, -0.62)] + [(0.0 + 0.2 * math.cos(math.radians(a)), 0.0 + 0.2 * math.sin(math.radians(a)))
                                  for a in range(-60, 181, 20)] + [(-0.42, 0.18)]
    br_pts = [(p[0], p[1]) for p in br_pts]
    for s in (-1, 1):
        ext(name + "Clev", br_pts, 0.1, piv + V((0, s * 0.47, 0)), X, Z, M["paint"], g, bevel=0.01)
        weld_bead(name + "WeldClev%d" % (s > 0), [(rs + 0.395, s * 0.47 + 0.055 * s, piv.z - 0.6), (rs + 0.395, s * 0.47 + 0.055 * s, piv.z + 0.17)],
                  0.012, g)
        # pivot bosses + grease fittings
        b = sb.prim("cyl", name + "Boss", loc=piv + V((0, s * 0.54, 0)), rot=(math.pi / 2, 0, 0), vertices=40, radius=0.15,
                    depth=0.05, mat=M["paint"], parent=g)
        sb.bevel(b, 0.01, 2)
    pin = sb.prim("cyl", name + "HPin", loc=piv, rot=(math.pi / 2, 0, 0), vertices=40, radius=0.085, depth=1.2,
                  mat=M["steel"], parent=g)
    sb.bevel(pin, 0.01, 2)
    for s in (-1, 1):
        ep = sb.prim("cyl", name + "PinPlate", loc=piv + V((0, s * 0.585, 0)), rot=(math.pi / 2, 0, 0), vertices=6,
                     radius=0.14, depth=0.03, mat=M["zinc"], parent=g)
        sb.bevel(ep, 0.006, 1)
        for k in (-1, 1):
            capscrew(name + "PinScr", piv + V((0.07 * k, s * 0.6, 0.07 * -k)), Y * s, 0.028, M["zinc"], g)
        # grease zerk + grease smear
        zk = piv + V((0.0, s * 0.55, 0.16))
        sb.prim("cyl", name + "Zerk", loc=zk + V((0, 0, 0.02)), vertices=6, radius=0.014, depth=0.03, mat=M["zinc"], parent=g)
        sb.prim("sphere", name + "ZerkB", loc=zk + V((0, 0, 0.045)), radius=0.011, segments=12, ring_count=6, mat=M["zinc"], parent=g)
        gr = sb.prim("sphere", name + "Grease", loc=piv + V((0.0, s * 0.515, -0.02)), radius=0.1, segments=24, ring_count=12,
                     mat=M["grease"], parent=g)
        gr.scale = (1.0, 0.12, 1.1)

    # --- the jaw (rotates about the pivot): forked tines with a hook over the lug pin
    jaw = sb.empty(name + "Jaw", loc=piv, parent=g)
    P = V((lug_pin, 0, zt + 0.56)) - piv
    px, pz = P.x, P.z
    rb = 0.2
    prof = [(rb * math.cos(math.radians(a)), rb * math.sin(math.radians(a))) for a in range(-150, 21, 15)]
    prof += [(0.2, 0.3), (0.16, 0.8), (0.1, 1.25), (0.02, pz + 0.12), (px + 0.3, pz + 0.3), (px + 0.02, pz + 0.33),
             (px - 0.16, pz + 0.24), (px - 0.19, pz + 0.12)]
    prof += [(px + 0.118 * math.cos(math.radians(a)), pz + 0.118 * math.sin(math.radians(a))) for a in range(160, 19, -10)]
    prof += [(px + 0.2, pz - 0.02), (-0.14, pz - 0.35), (-0.15, 0.9), (-0.17, 0.3)]
    prof += [(rb * math.cos(math.radians(a)), rb * math.sin(math.radians(a))) for a in range(150, 211, 15)][1:-1]
    for s in (-1, 1):
        ext(name + "Tine", prof, 0.1, V((0, s * 0.35, 0)), X, Z, M["paint_jaw"], jaw, bevel=0.012)
        # bronze liner in the hook notch (half-ring)
        liner = [(px + 0.118 * math.cos(math.radians(a)), pz + 0.118 * math.sin(math.radians(a))) for a in range(15, 166, 10)]
        liner += [(px + 0.1 * math.cos(math.radians(a)), pz + 0.1 * math.sin(math.radians(a))) for a in range(165, 14, -10)]
        ext(name + "Liner", liner, 0.106, V((0, s * 0.35, 0)), X, Z, M["bronze"], jaw, bevel=0.003, segs=1)
        # bushing boss at the pivot
        bb = sb.prim("cyl", name + "JBoss", loc=(0, s * 0.35, 0), rot=(math.pi / 2, 0, 0), vertices=40, radius=0.17,
                     depth=0.13, mat=M["paint_jaw"], parent=jaw)
        sb.bevel(bb, 0.012, 2)
        # flame-cut edge texture: a thin darker band along the tine perimeter is implied by edge wear in the paint
    # cross webs tying the tines (bolted): mid web carries the actuator ear
    ext(name + "WebMid", chamfer_rect(0.22, 0.6, 0.03), 0.7, V((0.02, 0, 0.95)), X, Z, M["paint_jaw"], jaw, bevel=0.01)
    ext(name + "WebTop", chamfer_rect(0.13, 0.12, 0.02), 0.6, V((-0.13, 0, pz + 0.17)), X, Z, M["paint_jaw"], jaw, bevel=0.008)
    for s in (-1, 1):
        for zz in (0.75, 0.95, 1.15):
            capscrew(name + "WebScr", V((0.02, s * 0.4, zz)), Y * s, 0.032, M["zinc"], jaw)
    ear = [(0.1, 0.72), (0.3, 0.8)] + [(0.35 + 0.075 * math.cos(math.radians(a)), 0.9 + 0.075 * math.sin(math.radians(a)))
                                       for a in range(-60, 181, 20)] + [(0.1, 1.1)]
    for s in (-1, 1):
        ext(name + "Ear", ear, 0.04, V((0, s * 0.075, 0)), X, Z, M["paint_jaw"], jaw, bevel=0.006)
    sb.prim("cyl", name + "EarPin", loc=(0.35, 0, 0.9), rot=(math.pi / 2, 0, 0), vertices=24, radius=0.03, depth=0.24,
            mat=M["steel"], parent=jaw)
    # soot on the jaw top from previous flights is handled by paint grime; frost dusting comes from the vehicle frost.

    # --- lock-bolt housing on a gusseted stand-off (camera side, +Y)
    hp = piv + V((0.06, 0.55, 0.75))
    ext(name + "LockPost", [(-0.1, -0.62), (0.14, -0.62), (0.14, -0.1), (0.1, -0.02), (-0.1, -0.02)], 0.1,
        hp + V((0, 0.0, 0)), X, Z, M["paint"], g, bevel=0.008)
    hs = ext(name + "LockHousing", chamfer_rect(0.22, 0.22, 0.03), 0.3, hp + V((0, 0.12, 0)), X, Z, M["paint"], g, bevel=0.01)
    hs.rotation_euler = (0, 0, 0)
    # the housing is extruded along Y (X x Z = -Y): OK, it is centred at hp+0.12 y
    for k in range(4):
        a = math.radians(45 + 90 * k)
        capscrew(name + "LockScr", hp + V((0.078 * math.cos(a), 0.27, 0.078 * math.sin(a))), Y, 0.022, M["zinc"], g)
    sb.prim("cyl", name + "LockAir", loc=hp + V((0, 0.33, 0.0)), rot=(math.pi / 2, 0, 0), vertices=6, radius=0.03,
            depth=0.05, mat=M["zinc"], parent=g)
    rocket.pipe(name + "LockHose", [hp + V((0, 0.36, 0)), hp + V((0.02, 0.5, -0.05)), hp + V((0.12, 0.52, -0.5)),
                                    hp + V((0.2, 0.46, -1.1)), V((rs + 0.5, 0.66, table_z + 0.13))], 0.018, M["rubber"], g)
    lock = sb.empty(name + "Lock", loc=hp, parent=g)
    lb = sb.prim("cyl", name + "LockBolt", loc=(0, -0.08, 0), rot=(math.pi / 2, 0, 0), vertices=32, radius=0.05, depth=0.34,
                 mat=M["chrome"], parent=lock)
    sb.bevel(lb, 0.01, 2)
    # jaw-side receiver bushing (bronze) where the bolt enters the tine
    sb.prim("cyl", name + "LockRecv", loc=hp + V((0, -0.14, 0)), rot=(math.pi / 2, 0, 0), vertices=32, radius=0.07,
            depth=0.02, mat=M["bronze"], parent=jaw).location = (hp - piv) + V((0, -0.14, 0))

    act_base = V((rs + 0.7, 0, table_z + 0.5))
    act_tip_local = V((0.35, 0, 0.9))
    # actuator base clevis bracket on the base plate side
    for s in (-1, 1):
        ext(name + "ActClev", [(-0.12, -0.38), (0.12, -0.38), (0.12, 0.0)] +
            [(0.09 * math.cos(math.radians(a)), 0.09 * math.sin(math.radians(a))) for a in range(0, 181, 20)] + [(-0.12, 0.0)],
            0.04, act_base + V((0, s * 0.11, 0)), X, Z, M["paint"], g, bevel=0.006)
    ext(name + "ActFoot", chamfer_rect(0.34, 0.34, 0.04), 0.04, V((rs + 0.7, 0, table_z + 0.13)), X, Y, M["paint"], g, bevel=0.006)
    return dict(root=g, jaw=jaw, pivot=piv, act_base=act_base, act_tip_local=act_tip_local, cyl_parent=g, open=55.0,
                lock=lock)


def actuator(cl, name):
    """Tie-rod hydraulic cylinder from the base clevis to the jaw ear; Damped Track keeps it aimed."""
    M = materials()
    g = cl["root"]
    base = cl["act_base"]
    tip = cl["pivot"] + cl["act_tip_local"]
    body_e = sb.empty(name + "CylE", loc=base, parent=g)
    tgt = sb.empty(name + "CylT", loc=cl["act_tip_local"], parent=cl["jaw"])
    L0 = (tip - base).length
    bl = L0 * 0.58
    # base clevis eye + cap
    sb.prim("cyl", name + "Eye", loc=(0, 0, 0), rot=(math.pi / 2, 0, 0), vertices=24, radius=0.07, depth=0.18, mat=M["paint"], parent=body_e)
    for zz, nm in ((0.1, "CapB"), (bl, "CapR")):
        c = sb.prim("cube", name + nm, loc=(0, 0, zz), scale=(0.12, 0.12, 0.06), mat=M["paint"], parent=body_e)
        sb.bevel(c, 0.012, 2)
    barrel = sb.prim("cyl", name + "Barrel", loc=(0, 0, (0.1 + bl) / 2), vertices=40, radius=0.095, depth=bl - 0.1,
                     mat=M["paint"], parent=body_e)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = 0.095 * math.cos(a) * 1.25, 0.095 * math.sin(a) * 1.25
        sb.prim("cyl", name + "Tie", loc=(x, y, (0.1 + bl) / 2), vertices=10, radius=0.012, depth=bl - 0.06, mat=M["zinc"], parent=body_e)
        for zz in (0.02, bl + 0.08):
            sb.prim("cyl", name + "TieNut", loc=(x, y, zz), vertices=6, radius=0.022, depth=0.03, mat=M["zinc"], parent=body_e)
    sb.prim("cyl", name + "Gland", loc=(0, 0, bl + 0.09), vertices=32, radius=0.06, depth=0.05, mat=M["steel"], parent=body_e)
    rod = sb.prim("cyl", name + "Rod", loc=(0, 0, L0 * 0.78), vertices=32, radius=0.04, depth=L0 * 0.5, mat=M["chrome"], parent=body_e)
    sb.prim("cube", name + "RodEnd", loc=(0, 0, L0 - 0.05), scale=(0.06, 0.03, 0.08), mat=M["steel"], parent=body_e)
    # ports + braided hoses running down to the deck manifold
    for zz, dx in ((0.16, 0.0), (bl - 0.05, 0.0)):
        sb.prim("cyl", name + "Port", loc=(0.0, 0.12, zz), rot=(math.pi / 2, 0, 0), vertices=6, radius=0.028, depth=0.05,
                mat=M["zinc"], parent=body_e)
    c = body_e.constraints.new('DAMPED_TRACK')
    c.target = tgt
    c.track_axis = 'TRACK_Z'
    # hoses are in pedestal space (they sag from the ports to a manifold block on the deck)
    man = V((cl["act_base"].x + 0.32, 0.3, cl["act_base"].z - 0.32))
    mb = sb.prim("cube", name + "Manifold", loc=man, scale=(0.12, 0.09, 0.07), mat=M["paint"], parent=g)
    sb.bevel(mb, 0.01, 2)
    d = (tip - base).normalized()
    for k, zz in enumerate((0.16, bl - 0.05)):
        p0 = base + d * zz + V((0, 0.17, 0))
        rocket.pipe(name + "Hose%d" % k, [p0, p0 + V((0.05, 0.12, -0.12)), man + V((0.0 + 0.06 * k, 0.12, 0.2)),
                                          man + V((0.06 * k - 0.03, 0.1, 0.07))], 0.016, M["hose"], g, res=3)
        for q in (p0, man + V((0.06 * k - 0.03, 0.1, 0.07))):
            sb.prim("cyl", name + "Ferrule", loc=q, vertices=6, radius=0.024, depth=0.05, mat=M["steel"], parent=g)
    return body_e, rod, tgt
