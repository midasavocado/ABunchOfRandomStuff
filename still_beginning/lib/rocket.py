"""STILL BEGINNING - the launch vehicle. ONE design for every shot (s19a/b/c, s20, s21 ...).

Units are metres. VEHICLE FRAME: +Z up along the axis, origin on the axis at the bottom plane of the first-stage
aft skirt (station 0, where the four hold-down lugs bear on the pad pedestals). The five first-stage engines hang
below to z = EXIT_Z. Local +Y is the "tower side": all umbilical plates face +Y.

Public API
----------
DIM / ST          dimensions and station table (metres, vehicle frame). AMBER band = ST["band"] (z0, z1).
build(name, frost=1.0, detail=1)          full two-stage stack. Returns dict:
        root (empty; move/rotate THIS), stage1, stage2 (empties under root; stage2 can be un-parented for staging),
        engines [5 engine empties], nozzle_exits [(x, y, z) vehicle frame], lugs [4 hold-down lug objs],
        vents {name: (loc, outward_dir)}, umbilicals {name: (loc, dir)}, mats {name: material}
build_upper_stage(name, frost=0.6, fairing=True)   upper stage alone (amber band at its aft end, vacuum engine,
        fairing). Same station numbers as in the full stack (so its base is at z = ST["band"][0]); returns dict
        root, engine, nozzle_exit_z, mats.
plume_material(name, ...), plume_mesh(name, ...), plume_rig(...)  exhaust (see bottom of file).

Design notes (keep consistent everywhere): 5.0 m diameter, ~66 m nozzle-exit-to-tip. Graphite aft skirt,
graphite intertank ring, graphite interstage, white tanks / upper stage / fairing; a single 0.45 m amber anodized
ring (upper-stage aft ring, sits on top of the interstage). 5 sea-level engines, 4 around 1 (outer at 45 deg
azimuths, r = 1.6), hold-down lugs at 0/90/180/270 deg. Raceway at azimuth -150 deg. No markings, no text.
"""
import bpy, bmesh, math, random
from mathutils import Vector as V, Matrix, Quaternion
import sb
import props

R = 2.5                    # body radius
EXIT_Z = -1.6              # first-stage nozzle exit plane
GIMBAL_Z = 1.6             # engine gimbal plane
SHIELD_Z = 1.75            # base heat-shield (blanket) plane
ENGINE_R = 1.6             # outer engine ring radius
EXIT_RAD = 0.68            # nozzle exit radius
ST = dict(
    skirt=(0.0, 3.6), fuel=(3.6, 15.6), intertank=(15.6, 17.0), lox=(17.0, 36.8), fwd=(36.8, 38.0),
    interstage=(38.0, 44.0), band=(44.0, 44.45), upper=(44.45, 53.0), fairing=(53.0, 64.0),
)
TIP_Z = ST["fairing"][1]
LENGTH = TIP_Z - EXIT_Z    # 65.6 m
DIM = dict(radius=R, diameter=2 * R, length=LENGTH, exit_z=EXIT_Z, tip_z=TIP_Z, band=ST["band"],
           engine_ring=ENGINE_R, exit_radius=EXIT_RAD, gimbal_z=GIMBAL_Z)
ENGINE_XY = [(0.0, 0.0)] + [(ENGINE_R * math.cos(math.radians(a)), ENGINE_R * math.sin(math.radians(a)))
                            for a in (45, 135, 225, 315)]
LUG_AZ = (0, 90, 180, 270)
LUG_R = 3.0                # hold-down shoe radial centre
LUG_PIN = 3.3              # hold-down bearing pin radius (z = 0.56)
RACEWAY_AZ = -150.0
SEGS = 192


# =========================================================================== small node helpers

def _ss(nb, x, a, b):
    """smoothstep(a, b, x) via Map Range (SMOOTHSTEP)."""
    nd = nb.n.new('ShaderNodeMapRange')
    nd.interpolation_type = 'SMOOTHSTEP'
    nd.clamp = True
    if isinstance(x, (int, float)):
        nd.inputs[0].default_value = x
    else:
        nb.l.new(x, nd.inputs[0])
    nd.inputs[1].default_value = a
    nd.inputs[2].default_value = b
    return nd.outputs[0]


def _xyz(nb, vec):
    sep = nb.new('ShaderNodeSeparateXYZ')
    nb.link(vec, sep.inputs[0])
    return sep.outputs[0], sep.outputs[1], sep.outputs[2]


def _comb(nb, x, y, z):
    c = nb.new('ShaderNodeCombineXYZ')
    for i, v in enumerate((x, y, z)):
        if isinstance(v, (int, float)):
            c.inputs[i].default_value = v
        else:
            nb.link(v, c.inputs[i])
    return c.outputs[0]


def _wn(nb, vec, w=None):
    n = nb.new('ShaderNodeTexWhiteNoise')
    n.noise_dimensions = '4D' if w is not None else '3D'
    nb.link(vec, n.inputs['Vector'])
    if w is not None:
        n.inputs['W'].default_value = w
    return n.outputs['Value']


def _cyl(nb, co, radius=R):
    """theta (-pi..pi), arc length (m), z from object coords."""
    x, y, z = _xyz(nb, co)
    th = nb.math('ARCTAN2', y, x)
    return th, nb.math('MULTIPLY', th, radius), z


def _seam(nb, coord, pitch, width, offset=0.0):
    """1 on regularly spaced lines (distance in coord units), 0 elsewhere."""
    c = nb.math('ADD', coord, offset) if offset else coord
    f = nb.math('FRACT', nb.math('DIVIDE', c, pitch))
    d = nb.math('MULTIPLY', nb.math('MINIMUM', f, nb.math('SUBTRACT', 1.0, f)), pitch)
    return nb.math('SUBTRACT', 1.0, _ss(nb, d, width * 0.35, width))


# =========================================================================== materials

def paint_white(name, base=(0.78, 0.78, 0.765), rough=0.34, frost_zones=(), pitch=2.4, nseg=8, radius=R,
                soot=0.0, seam_w=0.010, seed=0.0, long_seams=True):
    """Satin white vehicle paint, vehicle-frame object coords. Per-panel tone variation, weld/panel seams,
    orange peel, faint handling streaks, cryogenic frost below each tank's fill line (frost_zones =
    [(z0, z1, fill_z, amount)]) with a thin wet condensation band above it."""
    m = sb.mat(name, base, rough=rough, coat=0.12, coat_rough=0.15, spec=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    th, arc, z = _cyl(nb, co, radius)
    seg = 2 * math.pi / nseg
    seam = _seam(nb, z, pitch, seam_w)
    if long_seams:
        st = _seam(nb, arc, seg * radius, seam_w, offset=0.37)
        seam = nb.math('MAXIMUM', seam, nb.math('MULTIPLY', st, 0.8))
    pid = _comb(nb, nb.math('FLOOR', nb.math('DIVIDE', z, pitch)),
                nb.math('FLOOR', nb.math('DIVIDE', nb.math('ADD', th, 3.2 + 0.37 / radius), seg)), seed)
    tone = nb.maprange(_wn(nb, pid), 0.0, 1.0, 0.955, 1.025)
    col = nb.mix(1.0, (*base, 1), _comb(nb, tone, tone, nb.math('MULTIPLY', tone, 1.004)), blend='MULTIPLY')
    # seam dirt
    col = nb.mix(nb.math('MULTIPLY', seam, 0.35), col, (base[0] * 0.55, base[1] * 0.55, base[2] * 0.55, 1))
    # vertical handling / rain streaks (very faint)
    stk = nb.noise(nb.mapping(co, scale=(5.0, 5.0, 0.18)), scale=3.0, detail=6, rough=0.6)
    stk_m = nb.maprange(stk.outputs['Fac'], 0.58, 0.8, 0.0, 0.035)
    col = nb.mix(stk_m, col, (0.55, 0.53, 0.5, 1))
    if soot > 0:
        sn = nb.noise(nb.mapping(co, scale=(3.0, 3.0, 0.5)), scale=2.0, detail=6, rough=0.65)
        sm = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, _ss(nb, z, ST["fuel"][0], ST["fuel"][0] + 5.0)),
                     nb.maprange(sn.outputs['Fac'], 0.35, 0.75, 0.2, 1.0))
        col = nb.mix(nb.math('MULTIPLY', sm, soot), col, (0.18, 0.16, 0.14, 1))
    # orange peel + seam grooves
    peel = nb.noise(co, scale=160.0, detail=3, rough=0.5)
    h = nb.math('ADD', nb.math('MULTIPLY', peel.outputs['Fac'], 0.15), nb.math('MULTIPLY', seam, -1.0))
    rgh = nb.maprange(_wn(nb, pid, 3.0), 0.0, 1.0, rough * 0.85, rough * 1.2)
    # ---- frost
    frost = None
    wet = None
    for (z0, z1, fill, amt) in frost_zones:
        rag = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 0.6)), scale=0.55, detail=5, rough=0.62)
        zr = nb.math('ADD', z, nb.maprange(rag.outputs['Fac'], 0.3, 0.7, -0.9, 0.9, clamp=False))
        inz = nb.math('MULTIPLY', _ss(nb, z, z0 + 0.05, z0 + 0.45),
                      nb.math('SUBTRACT', 1.0, _ss(nb, zr, fill - 0.35, fill + 0.05)))
        patch = nb.noise(co, scale=0.9, detail=7, rough=0.62)
        streak = nb.noise(nb.mapping(co, scale=(2.2, 2.2, 0.22)), scale=2.0, detail=6, rough=0.6)
        cov = nb.math('MAXIMUM', nb.maprange(patch.outputs['Fac'], 0.40, 0.58, 0.25, 1.0),
                      nb.maprange(streak.outputs['Fac'], 0.50, 0.66, 0.0, 1.0))
        f = nb.math('MULTIPLY', nb.math('MULTIPLY', inz, cov), amt)
        # wet film just above the frost line
        wb = nb.math('MULTIPLY', _ss(nb, zr, fill - 0.2, fill + 0.1),
                     nb.math('SUBTRACT', 1.0, _ss(nb, zr, fill + 0.3, fill + 1.0)))
        wb = nb.math('MULTIPLY', nb.math('MULTIPLY', wb, _ss(nb, z, z0, z0 + 0.5)), amt)
        frost = f if frost is None else nb.math('MAXIMUM', frost, f)
        wet = wb if wet is None else nb.math('MAXIMUM', wet, wb)
    if frost is not None:
        crys = nb.voronoi(co, scale=420.0, feature='F1')
        fine = nb.noise(co, scale=60.0, detail=8, rough=0.7)
        fcol = nb.mix(nb.maprange(fine.outputs['Fac'], 0.3, 0.7), (0.86, 0.89, 0.93, 1), (0.95, 0.965, 0.985, 1))
        col = nb.mix(frost, col, fcol)
        h = nb.math('ADD', h, nb.math('MULTIPLY', frost, nb.math('ADD', nb.math('MULTIPLY', crys.outputs['Distance'], 1.5),
                                                               nb.math('MULTIPLY', fine.outputs['Fac'], 2.5))))
        rgh = nb.mix(frost, rgh, 0.78, dtype='FLOAT')
        rgh = nb.mix(nb.math('MULTIPLY', wet, 0.8), rgh, 0.12, dtype='FLOAT')
    nb.set('Base Color', col)
    nb.set('Roughness', rgh)
    nb.set('Normal', nb.bump(h, strength=0.25, distance=0.004))
    return m


def composite(name, color=(0.030, 0.032, 0.036), rough=0.40, stringers=0, radius=R, seams=(), sheen=0.08, wear=0.15):
    """Graphite composite / satin black paint. Optional external stringers (bump), ring seams at z values,
    mottled sanding marks, fine twill visible only up close, light scuffs."""
    m = sb.mat(name, color, rough=rough, coat=0.25, coat_rough=0.25, sheen=sheen, spec=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    th, arc, z = _cyl(nb, co, radius)
    h = nb.math('MULTIPLY', nb.noise(co, scale=140.0, detail=2).outputs['Fac'], 0.1)
    twill = nb.wave(nb.mapping(co, rot=(0.0, 0.0, 0.785)), scale=900.0, wtype='BANDS', direction='Z', profile='SAW')
    h = nb.math('ADD', h, nb.math('MULTIPLY', twill.outputs['Fac'], 0.06))
    if stringers:
        sp = 2 * math.pi * radius / stringers
        st = _seam(nb, arc, sp, 0.045)
        h = nb.math('ADD', h, nb.math('MULTIPLY', st, 2.0))
    ring = None
    for zz in seams:
        r = nb.math('SUBTRACT', 1.0, _ss(nb, nb.math('ABSOLUTE', nb.math('SUBTRACT', z, zz)), 0.003, 0.009))
        ring = r if ring is None else nb.math('MAXIMUM', ring, r)
    if ring is not None:
        h = nb.math('SUBTRACT', h, nb.math('MULTIPLY', ring, 1.5))
    mot = nb.noise(co, scale=1.3, detail=6, rough=0.6)
    rgh = nb.maprange(mot.outputs['Fac'], 0.3, 0.7, rough * 0.75, rough * 1.3)
    nb.set('Roughness', rgh)
    col = nb.mix(nb.maprange(mot.outputs['Fac'], 0.3, 0.7), (color[0] * 0.85, color[1] * 0.85, color[2] * 0.85, 1),
                 (color[0] * 1.2, color[1] * 1.2, color[2] * 1.22, 1))
    if wear > 0:
        sc = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 4.0)), scale=9.0, detail=4, rough=0.5)
        wm = nb.maprange(sc.outputs['Fac'], 0.68, 0.74, 0.0, wear)
        col = nb.mix(wm, col, (0.16, 0.16, 0.17, 1))
    nb.set('Base Color', col)
    nb.set('Normal', nb.bump(h, strength=0.3, distance=0.004))
    return m


def nozzle_outer_mat(name, z_throat=-1.25, z_exit=-3.2):
    """Regeneratively cooled tube-wall bell: fine meridional tube ridges, heat tint straw->violet->steel,
    soot toward the lip. Engine-local object coords (engine axis = local Z, pointing -Z)."""
    m = sb.mat(name, (0.2, 0.17, 0.14), metal=1.0, rough=0.34, aniso=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    th, arc, z = _cyl(nb, co, 1.0)
    tubes = nb.math('ABSOLUTE', nb.math('SINE', nb.math('MULTIPLY', th, 110.0)))
    tubes = nb.math('POWER', tubes, 0.45)
    u = nb.maprange(z, z_throat, z_exit)          # 0 at throat, 1 at exit
    tint = nb.ramp(u, [(0.0, (0.46, 0.34, 0.20)), (0.22, (0.40, 0.26, 0.20)), (0.45, (0.24, 0.21, 0.27)),
                       (0.7, (0.22, 0.22, 0.23)), (1.0, (0.16, 0.155, 0.15))])
    n = nb.noise(co, scale=5.0, detail=6, rough=0.6)
    tint = nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.0, 0.35), tint, (0.13, 0.12, 0.12, 1))
    soot = nb.math('MULTIPLY', _ss(nb, u, 0.72, 1.0), nb.maprange(n.outputs['Fac'], 0.35, 0.65, 0.3, 0.9))
    tint = nb.mix(soot, tint, (0.03, 0.028, 0.026, 1))
    nb.set('Base Color', tint)
    nb.set('Roughness', nb.mix(soot, nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.26, 0.42), 0.7, dtype='FLOAT'))
    band = _seam(nb, z, 0.42, 0.018, offset=0.1)
    h = nb.math('SUBTRACT', nb.math('MULTIPLY', tubes, 1.0), nb.math('MULTIPLY', band, 0.8))
    nb.set('Normal', nb.bump(h, strength=0.35, distance=0.003))
    return m


def nozzle_inner_mat(name, z_throat=-1.25, z_exit=-3.2):
    m = sb.mat(name, (0.05, 0.045, 0.04), metal=0.7, rough=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    th, arc, z = _cyl(nb, co, 1.0)
    u = nb.maprange(z, z_throat, z_exit)
    n = nb.noise(co, scale=7.0, detail=6, rough=0.6)
    c = nb.ramp(u, [(0.0, (0.10, 0.07, 0.05)), (0.3, (0.07, 0.055, 0.05)), (1.0, (0.025, 0.022, 0.02))])
    c = nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.0, 0.6), c, (0.015, 0.014, 0.013, 1))
    nb.set('Base Color', c)
    rings = nb.wave(co, scale=6.0, wtype='RINGS', direction='Z', dist=2.0, detail=3)
    nb.set('Roughness', nb.maprange(rings.outputs['Fac'], 0.0, 1.0, 0.38, 0.62))
    tubes = nb.math('ABSOLUTE', nb.math('SINE', nb.math('MULTIPLY', th, 110.0)))
    nb.set('Normal', nb.bump(tubes, strength=0.15, distance=0.002))
    return m


def foil_mat(name, color=(0.80, 0.52, 0.18)):
    """Crinkled gold MLI / kapton wrap on engine lines."""
    m = sb.mat(name, color, metal=1.0, rough=0.2)
    nb = sb.NB(m)
    co = nb.coord('Object')
    v = nb.voronoi(co, scale=38.0, feature='F1')
    n = nb.noise(co, scale=25.0, detail=6, rough=0.7, dist=0.6)
    h = nb.math('ADD', v.outputs['Distance'], nb.math('MULTIPLY', n.outputs['Fac'], 0.8))
    nb.set('Normal', nb.bump(h, strength=0.7, distance=0.004))
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.12, 0.35))
    col = nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (color[0] * 0.8, color[1] * 0.75, color[2] * 0.7, 1), (*color, 1))
    nb.set('Base Color', col)
    return m


def blanket_mat(name, color=(0.60, 0.58, 0.54), cell=0.14, soot=0.4):
    """Quilted silica-cloth base heat-shield blanket (horizontal: object x/y)."""
    m = sb.mat(name, color, rough=0.92, sheen=0.35, spec=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = _xyz(nb, co)
    fx = nb.math('SINE', nb.math('MULTIPLY', x, math.pi / cell))
    fy = nb.math('SINE', nb.math('MULTIPLY', y, math.pi / cell))
    pillow = nb.math('POWER', nb.math('ABSOLUTE', nb.math('MULTIPLY', fx, fy)), 0.5)
    weave = nb.wave(co, scale=700.0, wtype='BANDS', direction='X')
    h = nb.math('ADD', pillow, nb.math('MULTIPLY', weave.outputs['Fac'], 0.05))
    nb.set('Normal', nb.bump(h, strength=0.6, distance=0.012))
    n = nb.noise(co, scale=1.6, detail=6, rough=0.65)
    sm = nb.maprange(n.outputs['Fac'], 0.42, 0.72, 0.0, soot)
    stitch = nb.math('SUBTRACT', 1.0, pillow)
    c = nb.mix(nb.math('MULTIPLY', _ss(nb, stitch, 0.8, 0.97), 0.5), (*color, 1), (color[0] * 0.5, color[1] * 0.48, color[2] * 0.45, 1))
    c = nb.mix(sm, c, (0.10, 0.085, 0.07, 1))
    nb.set('Base Color', c)
    return m


def steel_mat(name, color=(0.42, 0.42, 0.43), rough=0.35, wear=0.3):
    return sb.brushed_metal(name, color, rough=rough, aniso=0.4, scale=60, bump=0.03, scratches=wear)


def _materials(frost=1.0, prefix="RK"):
    M = {}
    M["white1"] = paint_white(prefix + "White1", frost_zones=[
        (ST["fuel"][0], ST["fuel"][1], ST["fuel"][1] - 0.9, 0.55 * frost),
        (ST["lox"][0], ST["lox"][1], ST["lox"][1] - 1.1, 1.0 * frost)], soot=0.25, seed=1.0)
    M["white2"] = paint_white(prefix + "White2", frost_zones=[
        (ST["upper"][0] + 0.4, 48.4, 47.9, 0.45 * frost), (48.8, 52.6, 52.0, 0.8 * frost)], seed=2.0)
    M["fairing"] = paint_white(prefix + "Fairing", base=(0.80, 0.80, 0.79), pitch=3.7, nseg=2, seed=3.0, rough=0.3)
    M["skirt"] = composite(prefix + "Skirt", stringers=64, seams=(1.2, 2.4), wear=0.25)
    M["inter"] = composite(prefix + "Inter", stringers=0, seams=(40.0, 42.0), wear=0.08)
    M["intertank"] = composite(prefix + "Intertank", stringers=96, wear=0.1)
    M["intertank_w"] = paint_white(prefix + "IntertankW", base=(0.74, 0.74, 0.73), rough=0.42, pitch=0.7, nseg=96,
                                   seam_w=0.03, seed=4.0)
    M["amber"] = props.amber_anodized(prefix + "Amber")
    M["black"] = sb.mat(prefix + "Black", (0.012, 0.012, 0.013), rough=0.6)
    M["noz_out"] = nozzle_outer_mat(prefix + "NozOut")
    M["noz_in"] = nozzle_inner_mat(prefix + "NozIn")
    M["inconel"] = sb.brushed_metal(prefix + "Inconel", (0.50, 0.46, 0.41), rough=0.32, aniso=0.5, scale=80, bump=0.02, scratches=0.15)
    M["pump"] = sb.brushed_metal(prefix + "Pump", (0.36, 0.36, 0.37), rough=0.38, aniso=0.3, scale=50, bump=0.02)
    M["duct"] = sb.brushed_metal(prefix + "Duct", (0.62, 0.62, 0.63), rough=0.24, aniso=0.6, scale=150, bump=0.02)
    M["foil"] = foil_mat(prefix + "Foil")
    M["chrome"] = sb.mat(prefix + "Chrome", (0.9, 0.9, 0.9), metal=1.0, rough=0.08)
    M["act"] = sb.painted(prefix + "Act", (0.09, 0.09, 0.095), rough=0.4, coat=0.2, grime=0.2)
    M["blanket"] = blanket_mat(prefix + "Blanket")
    M["boot"] = sb.fabric(prefix + "Boot", (0.46, 0.44, 0.41), rough=0.9, weave=500, sheen=0.3)
    M["steel"] = steel_mat(prefix + "Steel")
    M["lug"] = sb.painted(prefix + "Lug", (0.22, 0.22, 0.225), rough=0.38, coat=0.1, grime=0.3)
    M["cable"] = sb.rubber(prefix + "Cable", (0.015, 0.015, 0.016))
    M["grille"] = sb.mat(prefix + "Grille", (0.02, 0.02, 0.022), metal=0.5, rough=0.5)
    return M


# =========================================================================== geometry helpers

def lathe(name, prof, segs=SEGS, mat=None, parent=None, auto=35, cap_top=False):
    """Revolve (r, z) profile around Z. Normals face +r when the profile goes up in z (outer surfaces)."""
    verts, faces = [], []
    n = len(prof)
    for i in range(segs):
        a = 2 * math.pi * i / segs
        c, s = math.cos(a), math.sin(a)
        for r, z in prof:
            verts.append((r * c, r * s, z))
    for i in range(segs):
        j = (i + 1) % segs
        for k in range(n - 1):
            faces.append((i * n + k, j * n + k, j * n + k + 1, i * n + k + 1))
    o = sb.mesh_obj(name, verts, faces, mat, True)
    if auto:
        _auto_smooth(o, auto)
    if parent:
        o.parent = parent
    return o


def _auto_smooth(o, angle=35):
    me = o.data
    bm = bmesh.new()
    bm.from_mesh(me)
    for e in bm.edges:
        if len(e.link_faces) == 2:
            a = e.link_faces[0].normal.angle(e.link_faces[1].normal, 0.0)
            e.smooth = a < math.radians(angle)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True


def _orient(o, a, b):
    """Place a Z-aligned primitive (centred, length along local Z) between points a and b."""
    a, b = V(a), V(b)
    o.location = (a + b) / 2
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = (b - a).to_track_quat('Z', 'Y')
    return o


def rod(name, a, b, r, mat, parent=None, verts=24):
    L = (V(b) - V(a)).length
    o = sb.prim("cyl", name, vertices=verts, radius=r, depth=L, mat=mat)
    _orient(o, a, b)
    if parent:
        o.parent = parent
    return o


def box(name, loc, size, mat, parent=None, rot=(0, 0, 0), bev=0.01):
    o = sb.prim("cube", name, loc=loc, rot=rot, scale=(size[0] / 2, size[1] / 2, size[2] / 2), mat=mat)
    if bev:
        sb.bevel(o, bev, 2)
    if parent:
        o.parent = parent
    return o


def pipe(name, pts, r, mat, parent=None, res=4):
    """Smooth pipe along points (bezier, auto handles) converted to mesh."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = r
    cu.bevel_resolution = res
    cu.resolution_u = 10
    cu.use_fill_caps = True
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        bp = sp.bezier_points[i]
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    o = bpy.data.objects.new(name, cu)
    sb.link_obj(o)
    cu.materials.append(mat)
    if parent:
        o.parent = parent
    return o


def _polar(r, az_deg, z):
    a = math.radians(az_deg)
    return V((r * math.cos(a), r * math.sin(a), z))


def _bell(r_t, z_t, r_e, z_e, a0=32.0, a1=9.0, n=28):
    """Rao-like bell: quadratic Bezier from throat to exit with initial/exit wall angles (deg). Profile (r,z) going
    DOWN in z (z_e < z_t)."""
    L = z_t - z_e
    t0 = V((math.tan(math.radians(a0)), -1.0))
    t1 = V((math.tan(math.radians(a1)), -1.0))
    p0 = V((r_t, z_t))
    p2 = V((r_e, z_e))
    # intersect p0 + s t0 = p2 - u t1
    A = Matrix(((t0.x, t1.x), (t0.y, t1.y)))
    s, u = A.inverted() @ (p2 - p0)
    p1 = p0 + t0 * s
    out = []
    for i in range(n + 1):
        t = i / n
        p = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2
        out.append((p.x, p.y))
    return out


# =========================================================================== engine

def engine(name, M, parent, loc, az_face=0.0, detail=1):
    """Sea-level engine. Local frame: gimbal point at origin, thrust axis -Z. Nozzle exit at local z=-3.2."""
    e = sb.empty(name, loc=loc, parent=parent)
    e.rotation_euler = (0, 0, math.radians(az_face))
    zt, zx = -1.25, -3.2
    rt, rx = 0.16, EXIT_RAD
    bell = _bell(rt, zt, rx, zx, 40.0, 6.0, 40)
    # outer skin: chamber + converging + bell (profile going up so normals face out) -> reverse
    wall = 0.028
    conv = [(0.27, -0.95), (0.262, -1.02), (0.235, -1.10), (0.20, -1.17), (rt + 0.01, zt + 0.02)]
    outer = [(r + wall * (0.4 + 0.6 * i / len(bell)), z) for i, (r, z) in enumerate(bell)]
    prof = list(reversed(outer)) + list(reversed([(r + 0.012, z) for r, z in conv])) + [(0.285, -0.40)]
    lathe(name + "Bell", prof, segs=128, mat=M["noz_out"], parent=e)
    inner = [(0.25, -0.40), (0.25, -0.95)] + conv[1:] + bell[1:]
    ino = lathe(name + "BellIn", inner, segs=128, mat=M["noz_in"], parent=e)
    # rolled exit lip + manifolds / stiffener rings
    lip = sb.prim("torus", name + "Lip", loc=(0, 0, zx), major_radius=rx + wall * 0.5, minor_radius=wall * 0.62,
                  major_segments=128, minor_segments=10, mat=M["noz_out"], parent=e)
    for zz, mr in ((-2.25, 0.013),):
        rr = _interp_r(outer, zz)
        sb.prim("torus", name + "Band", loc=(0, 0, zz), major_radius=rr + mr * 0.3, minor_radius=mr,
                major_segments=128, minor_segments=8, mat=M["noz_out"], parent=e)
    # fuel manifold (fat torus below throat) with feed stubs
    sb.prim("torus", name + "Manifold", loc=(0, 0, -1.30), major_radius=0.235, minor_radius=0.05,
            major_segments=64, minor_segments=14, mat=M["inconel"], parent=e)
    # chamber jacket rings
    for zz in (-0.48, -0.66, -0.84):
        sb.prim("torus", name + "Jacket", loc=(0, 0, zz), major_radius=0.29, minor_radius=0.014,
                major_segments=64, minor_segments=8, mat=M["inconel"], parent=e)
    # actuator collar with two clevis lugs (pitch & yaw)
    sb.prim("torus", name + "Collar", loc=(0, 0, -1.05), major_radius=0.285, minor_radius=0.035,
            major_segments=64, minor_segments=10, mat=M["steel"], parent=e)
    for a in (0, 90):
        box(name + "Clevis", _polar(0.33, a, -1.05), (0.12, 0.09, 0.12), M["steel"], parent=e,
            rot=(0, 0, math.radians(a)), bev=0.01)
    # powerhead / injector dome
    ph = lathe(name + "Head", [(0.0, 0.02), (0.12, 0.0), (0.24, -0.06), (0.31, -0.16), (0.32, -0.30), (0.29, -0.34),
                                (0.29, -0.42)], segs=96, mat=M["inconel"], parent=e)
    sb.prim("torus", name + "HeadFl", loc=(0, 0, -0.33), major_radius=0.315, minor_radius=0.025,
            major_segments=64, minor_segments=8, mat=M["steel"], parent=e)
    # gimbal block
    box(name + "Gimbal", (0, 0, 0.05), (0.22, 0.22, 0.14), M["steel"], parent=e, bev=0.015)
    # turbopumps (ox + fuel) on opposite sides, ducts into the head
    for k, (a, rr, hh, col) in enumerate(((150, 0.19, 0.62, "pump"), (-40, 0.16, 0.52, "pump"))):
        c = _polar(0.58, a, -0.55)
        tp = lathe(name + "TP%d" % k, [(0.0, hh / 2 + 0.05), (rr * 0.6, hh / 2 + 0.04), (rr, hh / 2 - 0.02),
                                         (rr, -hh / 2 + 0.05), (rr * 1.25, -hh / 2 + 0.02), (rr * 1.25, -hh / 2 - 0.06),
                                         (rr * 0.5, -hh / 2 - 0.10), (0.0, -hh / 2 - 0.11)], segs=64, mat=M[col], parent=e)
        tp.location = c
        sb.prim("torus", name + "Vol%d" % k, loc=c + V((0, 0, -hh / 2 + 0.02)), major_radius=rr * 1.1, minor_radius=0.05,
                major_segments=48, minor_segments=10, mat=M["inconel"], parent=e)
        top = c + V((0, 0, hh / 2 + 0.05))
        pipe(name + "Duct%d" % k, [top, top + V((0, 0, 0.12)), _polar(0.2, a, -0.02) + V((0, 0, 0.06))],
             0.065 if k == 0 else 0.055, M["foil"] if k == 0 else M["duct"], parent=e)
        pipe(name + "Feed%d" % k, [c + _polar(rr * 1.2, a + 180, -hh / 2), _polar(0.36, a + (25 if k else -25), -0.9),
                                     _polar(0.25, a + (35 if k else -35), -1.28)], 0.028, M["duct"], parent=e)
        # suction line up to the thrust structure
        pipe(name + "Suction%d" % k, [c + V((0, 0, hh / 2 + 0.02)), c + V((0, 0, 0.35)), c * 0.8 + V((0, 0, 0.55))],
             0.075, M["duct"], parent=e)
    if detail:
        # small instrumentation lines & harness
        rs = random.Random(hash(name) & 0xffff)
        for k in range(5):
            a = rs.uniform(0, 360)
            pipe(name + "Line%d" % k, [_polar(0.30, a, -0.30), _polar(0.36, a + 20, -0.6), _polar(0.30, a + 30, -0.95)],
                 0.009, M["cable"] if k % 2 else M["duct"], parent=e, res=2)
    return e


def _interp_r(prof, z):
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        if (z0 - z) * (z1 - z) <= 0 and z0 != z1:
            t = (z - z0) / (z1 - z0)
            return r0 + (r1 - r0) * t
    return prof[-1][0]


def vac_engine(name, M, parent, loc):
    """Upper-stage vacuum engine: compact powerhead + large radiatively cooled nozzle extension. Local exit at z=-4.4."""
    e = sb.empty(name, loc=loc, parent=parent)
    bell = _bell(0.13, -0.95, 1.18, -4.4, 35.0, 7.0, 44)
    prof = list(reversed([(r + 0.02, z) for r, z in bell])) + [(0.26, -0.8), (0.27, -0.35)]
    lathe(name + "Ext", prof, segs=160, mat=M["vac_ext"], parent=e)
    lathe(name + "ExtIn", [(0.22, -0.35), (0.22, -0.8)] + bell, segs=160, mat=M["noz_in"], parent=e)
    sb.prim("torus", name + "Lip", loc=(0, 0, -4.4), major_radius=1.2, minor_radius=0.022, major_segments=160,
            minor_segments=8, mat=M["inconel"], parent=e)
    sb.prim("torus", name + "Joint", loc=(0, 0, -1.7), major_radius=_interp_r(prof[::-1], -1.7) + 0.02, minor_radius=0.03,
            major_segments=128, minor_segments=8, mat=M["inconel"], parent=e)
    lathe(name + "Head", [(0.0, 0.05), (0.15, 0.02), (0.30, -0.12), (0.32, -0.34), (0.27, -0.38)], segs=96,
          mat=M["inconel"], parent=e)
    for k, a in enumerate((20, 200)):
        c = _polar(0.5, a, -0.4)
        tp = sb.prim("cyl", name + "TP%d" % k, loc=c, vertices=48, radius=0.14, depth=0.45, mat=M["pump"], parent=e)
        pipe(name + "Duct%d" % k, [c + V((0, 0, 0.22)), c + V((0, 0, 0.35)), _polar(0.15, a, 0.1)], 0.05, M["foil"], parent=e)
    return e


def vac_ext_mat(name, glow=0.0):
    """Niobium-alloy nozzle extension: dark bronze-grey with heat bands; optional orange heat glow (s21)."""
    m = sb.mat(name, (0.16, 0.14, 0.13), metal=0.9, rough=0.36)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = _xyz(nb, co)
    u = nb.maprange(z, -0.9, -4.4)
    n = nb.noise(co, scale=4.0, detail=5)
    c = nb.ramp(u, [(0.0, (0.35, 0.25, 0.16)), (0.35, (0.20, 0.17, 0.2)), (1.0, (0.13, 0.12, 0.12))])
    c = nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.0, 0.3), c, (0.08, 0.075, 0.07, 1))
    nb.set('Base Color', c)
    hoop = _seam(nb, z, 0.35, 0.02)
    nb.set('Normal', nb.bump(hoop, strength=0.3, distance=0.003))
    if glow > 0:
        g = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, u), glow)
        nb.set('Emission Color', (1.0, 0.33, 0.06))
        nb.set('Emission Strength', g)
    return m


# =========================================================================== vehicle

def _shell(name, z0, z1, mat, parent, r=R, lip0=0.0, lip1=0.0, segs=SEGS, nz=2):
    """Cylindrical shell z0..z1 with optional rolled edges (slight radius step)."""
    prof = []
    if lip0:
        prof += [(r - lip0, z0), (r - lip0 * 0.25, z0 + lip0 * 0.4)]
    else:
        prof += [(r, z0)]
    for k in range(1, nz):
        prof.append((r, z0 + (z1 - z0) * k / nz))
    if lip1:
        prof += [(r - lip1 * 0.25, z1 - lip1 * 0.4), (r - lip1, z1)]
    else:
        prof += [(r, z1)]
    return lathe(name, prof, segs=segs, mat=mat, parent=parent)


def _ogive(z0, L, rb, tip_r=0.22, n=40):
    rho = (rb * rb + L * L) / (2 * rb)
    pts = []
    # x measured from base upward: r(x) = sqrt(rho^2 - x^2) + rb - rho  (tangent at base, point at x=L)
    for i in range(n + 1):
        x = L * (i / n) ** 0.85
        r = math.sqrt(max(0.0, rho * rho - x * x)) + rb - rho
        pts.append((max(r, 0.0), z0 + x))
    # blunt tip: cut where r = tip_r and add a spherical cap
    out = [p for p in pts if p[0] >= tip_r]
    zc = out[-1][1]
    rc = out[-1][0]
    for k in range(1, 7):
        a = (math.pi / 2) * k / 6
        out.append((rc * math.cos(a), zc + rc * 0.9 * math.sin(a)))
    out[-1] = (0.0, out[-1][1])
    return out


def build_stage1(M, parent, detail=1):
    s1 = sb.empty("Stage1", parent=parent)
    z = ST
    # aft skirt (graphite) with a heavier lower ring and inner wall
    _shell("S1Skirt", 0.25, z["skirt"][1], M["skirt"], s1)
    lathe("S1SkirtRing", [(2.44, 0.0), (2.52, 0.0), (2.525, 0.02), (2.525, 0.23), (2.515, 0.25), (2.5, 0.26)],
          mat=M["lug"], parent=s1)
    lathe("S1SkirtIn", [(2.44, 0.0), (2.44, SHIELD_Z + 0.02)], mat=M["skirt"], parent=s1)
    for zz in (0.62, 1.22):
        sb.prim("torus", "S1Frame", loc=(0, 0, zz), major_radius=2.40, minor_radius=0.045, major_segments=SEGS,
                minor_segments=8, mat=M["lug"], parent=s1)
    # base heat shield with 5 openings + fabric boots
    _heat_shield(M, s1)
    # tanks & intertank
    _shell("S1Fuel", z["fuel"][0], z["fuel"][1], M["white1"], s1, nz=6)
    _shell("S1Intertank", z["intertank"][0], z["intertank"][1], M["intertank_w"], s1)
    _shell("S1Lox", z["lox"][0], z["lox"][1], M["white1"], s1, nz=10)
    _shell("S1Fwd", z["fwd"][0], z["fwd"][1], M["white1"], s1)
    # engines
    engines = []
    for i, (x, y) in enumerate(ENGINE_XY):
        az_face = math.degrees(math.atan2(y, x)) if i else 20.0
        engines.append(engine("Eng%d" % i, M, s1, (x, y, GIMBAL_Z), az_face=az_face, detail=detail))
    # gimbal actuators: anchored on the thrust structure just below the blanket, to each engine's collar lugs
    bpy.context.view_layer.update()
    for i, e in enumerate(engines):
        mw = e.matrix_local
        for k, a in enumerate((0, 90)):
            lugp = mw @ _polar(0.40, a, -1.05)
            axis_dir = (mw.to_3x3() @ _polar(1.0, a, 0.0)).normalized()
            anchor = V((lugp.x, lugp.y, SHIELD_Z - 0.05)) + axis_dir * 0.28
            _actuator("Act%d_%d" % (i, k), anchor, lugp, M, s1)
    # hold-down lugs
    lugs = []
    for a in LUG_AZ:
        lugs.append(_lug("Lug%d" % a, a, M, s1))
    # raceway
    _raceway("S1Race", z["skirt"][1] - 0.4, z["fwd"][1] - 0.2, M["white1"], M["lug"], s1)
    # tail umbilical plates (tower side, +Y)
    umb = {}
    for k, a in enumerate((70, 110)):
        p = _polar(R + 0.005, a, 2.3)
        pl = box("S1Umb%d" % k, p, (0.75, 0.05, 0.95), M["lug"], parent=s1, rot=(0, 0, math.radians(a + 90)), bev=0.012)
        for j in range(3):
            c = _polar(R + 0.035, a + (j - 1) * 5.5, 2.3 + (0.15 if j == 1 else -0.12))
            cc = sb.prim("cyl", "S1UmbC", loc=c, rot=(0, math.pi / 2, math.radians(a)), vertices=24, radius=0.085,
                         depth=0.05, mat=M["steel"], parent=s1)
        umb["tail%d" % k] = (p, _polar(1.0, a, 0.0))
    # vents (flush louvred doors) - positions for venting FX
    vents = {}
    for nm, zz, a in (("lox1", z["fwd"][0] + 0.35, -110.0), ("fuel1", z["fuel"][1] - 0.4, -70.0)):
        p = _polar(R + 0.004, a, zz)
        box("Vent_" + nm, p, (0.55, 0.03, 0.34), M["grille"], parent=s1, rot=(0, 0, math.radians(a + 90)), bev=0.006)
        vents[nm] = (p, _polar(1.0, a, 0.0))
    if detail:
        # access hatches on the skirt
        for a in (-40.0, 150.0, 200.0):
            p = _polar(R + 0.006, a, 2.6)
            box("S1Hatch", p, (0.8, 0.02, 0.6), M["skirt"], parent=s1, rot=(0, 0, math.radians(a + 90)), bev=0.008)
    return s1, engines, lugs, vents, umb


def _heat_shield(M, parent):
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=SEGS, radius=2.44)
    me = bpy.data.meshes.new("S1Shield")
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new("S1Shield", me)
    sb.link_obj(o)
    o.location = (0, 0, SHIELD_Z)
    cutters = []
    for (x, y) in ENGINE_XY:
        c = sb.prim("cyl", "cut", loc=(x, y, SHIELD_Z), vertices=64, radius=0.52, depth=0.4)
        cutters.append(c)
    cut = sb.join(cutters, "cutters")
    md = o.modifiers.new("B", 'BOOLEAN')
    md.object = cut
    md.operation = 'DIFFERENCE'
    md.solver = 'EXACT'
    sol = o.modifiers.new("Sol", 'SOLIDIFY')
    sol.thickness = 0.06
    sb.apply_mods(o)
    bpy.data.objects.remove(cut)
    o.data.materials.append(M["blanket"])
    for p in o.data.polygons:
        p.use_smooth = False
    o.location = (0, 0, 0)
    o.data.transform(Matrix.Translation((0, 0, SHIELD_Z)))
    o.parent = parent
    # flexible bellows boots from the opening to each powerhead
    for i, (x, y) in enumerate(ENGINE_XY):
        prof = []
        n = 9
        for k in range(n + 1):
            t = k / n
            r = sb.lerp(0.33, 0.52, t) + (0.028 if k % 2 else -0.0)
            prof.append((r, sb.lerp(GIMBAL_Z - 0.34, SHIELD_Z + 0.01, t)))
        b = lathe("Boot%d" % i, prof, segs=96, mat=M["boot"], parent=parent)
        b.location = (x, y, 0)
    return o


def _actuator(name, anchor, tip, M, parent):
    anchor, tip = V(anchor), V(tip)
    d = (tip - anchor)
    L = d.length
    u = d.normalized()
    cyl_end = anchor + u * (L * 0.58)
    rod(name + "Body", anchor + u * 0.06, cyl_end, 0.075, M["act"], parent, verts=32)
    rod(name + "Cap", anchor, anchor + u * 0.1, 0.085, M["steel"], parent, verts=32)
    rod(name + "Rod", cyl_end - u * 0.05, tip - u * 0.05, 0.034, M["chrome"], parent, verts=24)
    rod(name + "Gland", cyl_end - u * 0.02, cyl_end + u * 0.05, 0.08, M["steel"], parent, verts=32)
    s = sb.prim("sphere", name + "Eye", loc=tip, radius=0.05, segments=16, ring_count=8, mat=M["steel"], parent=parent)
    # hose loop to the body
    side = u.cross(V((0, 0, 1)))
    if side.length < 1e-3:
        side = V((1, 0, 0))
    side.normalize()
    p0 = anchor + u * 0.25 + side * 0.08
    pipe(name + "Hose", [p0, p0 + side * 0.18 + V((0, 0, 0.1)), anchor + side * 0.2 + V((0, 0, 0.12))], 0.014, M["cable"], parent, res=2)


def _lug(name, az, M, parent):
    """Forged hold-down fitting: tapered bracket blending into the skirt (radial x / z profile, extruded tangentially),
    doubler plate with bolt rows, and a shouldered bearing pin (axis tangential) with retaining nuts. Lug-local frame:
    +X radial outward. Pin centre at (LUG_PIN, 0, 0.56); shoe underside at z=0 bears on the pad pedestal."""
    g = sb.empty(name, parent=parent)
    g.rotation_euler = (0, 0, math.radians(az))
    X, Y, Z = V((1, 0, 0)), V((0, 1, 0)), V((0, 0, 1))
    prof = [(2.47, 0.0), (LUG_R + 0.52, 0.0), (LUG_R + 0.55, 0.05), (LUG_R + 0.55, 0.30), (LUG_PIN + 0.02, 0.36),
            (2.95, 0.62), (2.72, 1.25), (2.56, 1.75), (2.47, 1.9)]
    extrude(name + "Web", prof, 0.2, (0, 0, 0), X, Z, M["lug"], g, bevel=0.012)
    ear = [(2.47, 0.0), (LUG_R + 0.55, 0.0), (LUG_R + 0.55, 0.36)] + arc_pts((LUG_PIN, 0.56), 0.21, -20, 200, 14)[1:] + \
          [(2.9, 0.82), (2.62, 1.55), (2.47, 1.8)]
    for yy in (-0.2, 0.2):
        extrude(name + "Ear", ear, 0.08, (0, yy, 0), X, Z, M["lug"], g, bevel=0.01)
    extrude(name + "Shoe", [(2.47, 0.0), (LUG_R + 0.55, 0.0), (LUG_R + 0.55, 0.12), (2.47, 0.12)], 0.62, (0, 0, 0), X, Z,
            M["steel"], g, bevel=0.01)
    box(name + "Doubler", (2.515, 0, 0.9), (0.03, 0.9, 1.7), M["lug"], parent=g, bev=0.008)
    for i in range(6):
        for s in (-1, 1):
            sb.prim("cyl", name + "Rivet", loc=(2.535, s * 0.36, 0.25 + i * 0.27), rot=(0, math.pi / 2, 0), vertices=8,
                    radius=0.022, depth=0.02, mat=M["steel"], parent=g)
    # pin: shaft + shoulders + nuts (axis along Y)
    sb.prim("cyl", name + "Pin", loc=(LUG_PIN, 0, 0.56), rot=(math.pi / 2, 0, 0), vertices=48, radius=0.10,
            depth=0.96, mat=M["steel"], parent=g)       # long enough for the pad clamp's fork tines (|y| 0.30-0.40)
    for s in (-1, 1):
        sb.prim("cyl", name + "Collar", loc=(LUG_PIN, s * 0.155, 0.56), rot=(math.pi / 2, 0, 0), vertices=32, radius=0.13,
                depth=0.04, mat=M["steel"], parent=g)
        sb.prim("cyl", name + "Nut", loc=(LUG_PIN, s * 0.475, 0.56), rot=(math.pi / 2, 0, 0), vertices=6, radius=0.085,
                depth=0.05, mat=M["steel"], parent=g)
    return g


def _raceway(name, z0, z1, mat, bmat, parent, az=RACEWAY_AZ, w=0.32, d=0.16):
    """Cable raceway: a low rounded fairing with tapered ramp ends, strapped to the skin every ~1.2 m."""
    a = math.radians(az)
    prof_pts = []
    L = z1 - z0
    ramp = 0.6
    # cross-section (tangential t, radial r) rounded-rect; ends taper the radial depth to zero over `ramp`
    nt = 10
    sec = []
    for k in range(nt + 1):
        t = -w / 2 + w * k / nt
        u = abs(t) / (w / 2)
        sec.append((t, d * math.sqrt(max(0.0, 1 - u ** 6))))
    verts, faces = [], []
    zs = [z0 + ramp * (k / 6) for k in range(7)] + [z0 + ramp + (L - 2 * ramp) * k / 8 for k in range(1, 8)] + \
         [z1 - ramp + ramp * (k / 6) for k in range(7)]
    rows = len(zs)
    for zz in zs:
        f = min(1.0, (zz - z0) / ramp, (z1 - zz) / ramp)
        f = sb.smooth(f)
        for t, h in sec:
            ang = a + t / R
            rr = R - 0.004 + h * f
            verts.append((rr * math.cos(ang), rr * math.sin(ang), zz))
    n = len(sec)
    for i in range(rows - 1):
        for k in range(n - 1):
            faces.append((i * n + k, i * n + k + 1, (i + 1) * n + k + 1, (i + 1) * n + k))
    o = sb.mesh_obj(name, verts, faces, mat, True)
    o.parent = parent
    nstr = int(L / 1.2)
    for k in range(1, nstr):
        zz = z0 + k * L / nstr
        lathe_arc = []
        seg = []
        for q in range(9):
            t = -w / 2 - 0.03 + (w + 0.06) * q / 8
            u = min(1.0, abs(t) / (w / 2))
            hh = d * math.sqrt(max(0.0, 1 - u ** 6)) + 0.008
            seg.append((t, hh))
        vv, ff = [], []
        for (t, hh) in seg:
            ang = a + t / R
            for dz in (-0.03, 0.03):
                vv.append(((R - 0.003 + hh) * math.cos(ang), (R - 0.003 + hh) * math.sin(ang), zz + dz))
        for q in range(len(seg) - 1):
            ff.append((2 * q, 2 * q + 2, 2 * q + 3, 2 * q + 1))
        st = sb.mesh_obj(name + "Strap", vv, ff, mat, True)
        sb.solidify(st, 0.006, -1)
        st.parent = parent
    return o


def build_stage2(M, parent, detail=1, fairing=True, vac=False):
    s2 = sb.empty("Stage2", parent=parent)
    z = ST
    # amber anodized aft ring: machined chamfers, sits proud by 12 mm
    b0, b1 = z["band"]
    lathe("AmberBand", [(R - 0.01, b0), (R + 0.006, b0 + 0.006), (R + 0.012, b0 + 0.02), (R + 0.012, b1 - 0.02),
                        (R + 0.006, b1 - 0.006), (R - 0.01, b1)], mat=M["amber"], parent=s2, auto=25)
    # thin dark separation groove below the band is the interstage top lip
    _shell("S2Tank", z["upper"][0], z["upper"][1], M["white2"], s2, nz=4)
    lathe("S2Aft", [(0.0, 43.2), (1.2, 43.35), (2.0, 43.7), (2.42, 44.1), (R - 0.01, b0)], mat=M["white2"], parent=s2)
    if vac:
        vac_engine("VacEng", M, s2, (0, 0, 43.25))
    # payload adapter separation ring (graphite line) + fairing
    f0, f1 = z["fairing"]
    lathe("PLARing", [(R - 0.005, f0 - 0.06), (R + 0.004, f0 - 0.05), (R + 0.004, f0 + 0.05), (R - 0.005, f0 + 0.06)],
          mat=M["inter"], parent=s2)
    if fairing:
        cylL = 4.5
        prof = [(R, f0 + 0.06), (R, f0 + cylL * 0.5), (R, f0 + cylL)] + _ogive(f0 + cylL, f1 - f0 - cylL, R)[1:]
        lathe("Fairing", prof, mat=M["fairing"], parent=s2, auto=40)
        # separation rail (vertical seam fitting) on both sides
        for a in (0.0, 180.0):
            ca = math.radians(a + 0.37 / R * 180 / math.pi * 0 + 21.2)
            p = V((math.cos(ca), math.sin(ca), 0))
            box("FairRail", p * (R + 0.01) + V((0, 0, f0 + cylL / 2)), (0.04, 0.09, cylL - 0.2), M["fairing"],
                parent=s2, rot=(0, 0, ca), bev=0.01)
    # upper stage umbilical plate (+Y) and vent
    umb = {}
    p = _polar(R + 0.006, 90.0, 49.2)
    box("S2Umb", p, (1.0, 0.06, 1.25), M["lug"], parent=s2, rot=(0, 0, math.radians(180)), bev=0.015)
    for j in range(4):
        c = p + V(((j % 2 - 0.5) * 0.45, 0.04, (j // 2 - 0.5) * 0.55))
        sb.prim("cyl", "S2UmbC", loc=c, rot=(math.pi / 2, 0, 0), vertices=24, radius=0.1, depth=0.05, mat=M["steel"], parent=s2)
    umb["upper"] = (p, V((0, 1, 0)))
    vents = {}
    for nm, zz, a in (("lox2", 52.35, -120.0), ("fuel2", 48.2, -40.0)):
        q = _polar(R + 0.004, a, zz)
        box("Vent_" + nm, q, (0.45, 0.03, 0.28), M["grille"], parent=s2, rot=(0, 0, math.radians(a + 90)), bev=0.005)
        vents[nm] = (q, _polar(1.0, a, 0.0))
    if detail:
        for a in (-30.0, 150.0):
            q = _polar(R + 0.07, a, 52.3)
            box("S2Blade", q, (0.14, 0.03, 0.26), M["lug"], parent=s2, rot=(0, 0, math.radians(a + 90)), bev=0.01)
        _raceway("S2Race", 45.0, 52.6, M["white2"], M["lug"], s2)
    return s2, vents, umb


def build_interstage(M, parent):
    z = ST
    i0, i1 = z["interstage"]
    ist = _shell("Interstage", i0, i1, M["inter"], parent, lip1=0.012)
    # inner wall (seen at staging), top lip ring
    lathe("InterstageIn", [(R - 0.06, i1), (R - 0.06, i0 + 0.3)], mat=M["inter"], parent=parent)
    lathe("InterstageLip", [(R - 0.06, i1 - 0.005), (R - 0.004, i1 - 0.005)], mat=M["lug"], parent=parent)
    # camera pods
    for a in (-45.0, 135.0):
        c = _polar(R + 0.05, a, 42.9)
        s = sb.prim("sphere", "CamPod", loc=c, radius=0.14, segments=24, ring_count=12, mat=M["inter"], parent=parent)
        s.scale = (1, 1, 1.8)
        s.rotation_euler = (0, 0, math.radians(a))
    return ist


def build(name="Rocket", frost=1.0, detail=1, vac=False, prefix="RK"):
    """Full stack. Returns dict (see module doc). Move/rotate/animate result['root'] only."""
    M = _materials(frost, prefix)
    M["vac_ext"] = vac_ext_mat(prefix + "VacExt")
    root = sb.empty(name)
    s1, engines, lugs, v1, u1 = build_stage1(M, root, detail)
    build_interstage(M, s1)
    s2, v2, u2 = build_stage2(M, root, detail, fairing=True, vac=vac)
    vents = dict(v1)
    vents.update(v2)
    umb = dict(u1)
    umb.update(u2)
    exits = [(x, y, EXIT_Z) for (x, y) in ENGINE_XY]
    return dict(root=root, stage1=s1, stage2=s2, engines=engines, lugs=lugs, vents=vents, umbilicals=umb,
                nozzle_exits=exits, mats=M, dims=DIM)


def build_upper_stage(name="UpperStage", frost=0.6, fairing=True, glow=0.0, prefix="US"):
    """Upper stage alone (after separation): amber aft ring, tank, vacuum engine exposed, fairing.
    Station numbers are identical to the full stack: its aft ring bottom is at z = ST['band'][0] = 44.0,
    vacuum nozzle exit at z = nozzle_exit_z (~38.85)."""
    M = _materials(frost, prefix)
    M["vac_ext"] = vac_ext_mat(prefix + "VacExt", glow=glow)
    root = sb.empty(name)
    s2, vents, umb = build_stage2(M, root, 1, fairing=fairing, vac=True)
    return dict(root=root, stage2=s2, engine=bpy.data.objects.get("VacEng"), nozzle_exit_z=43.25 - 4.4,
                vents=vents, mats=M, dims=DIM)


# =========================================================================== exhaust

def plume_material(name, intensity=1.0, core_col=(1.0, 0.86, 0.62), mid_col=(1.0, 0.52, 0.16), tail_col=(0.9, 0.28, 0.06),
                   length=40.0, diamonds=0.12, flow_speed=0.0):
    """Emissive additive plume shell. Object coords: plume axis = local -Z starting at z=0 (nozzle exit plane).
    Brightness: view-facing core (thicker through the middle), along-axis colour/decay ramp, subtle shock-diamond
    modulation near the exit, streaky turbulence advected downstream (drive node 'Flow' W socket per frame).
    Drive overall strength via node 'Gain' (Math multiply, input 1)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    try:
        m.surface_render_method = 'BLENDED'
    except Exception:
        pass
    m.use_backface_culling = False
    nt = m.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    co = nb.coord('Object')
    x, y, z = _xyz(nb, co)
    u = nb.math('DIVIDE', nb.math('MULTIPLY', z, -1.0), length)          # 0 at exit -> 1 at tail
    lw = nb.new('ShaderNodeLayerWeight')
    lw.inputs[0].default_value = 0.5
    facing = nb.math('SUBTRACT', 1.0, lw.outputs['Facing'])               # 1 facing camera (centre), 0 at rim
    core = nb.math('POWER', facing, 1.6)
    # streaky turbulence advected along -z
    adv = nb.new('ShaderNodeMapping')
    adv.name = "Adv"
    nb.link(co, adv.inputs[0])
    adv.inputs['Scale'].default_value = (1.0, 1.0, 0.3)
    tn = nb.noise(adv.outputs[0], scale=1.4, detail=6, rough=0.55, dim='4D')
    tn.name = "Flow"
    tfac = nb.maprange(tn.outputs['Fac'], 0.3, 0.72, 0.55, 1.25)
    # shock diamonds near the exit (subtle)
    dia = nb.math('ADD', 1.0, nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('COSINE', nb.math('MULTIPLY', z, 2 * math.pi / 1.5)),
                                                             nb.math('SUBTRACT', 1.0, _ss(nb, u, 0.0, 0.18))), diamonds))
    decay = nb.math('SUBTRACT', 1.0, _ss(nb, u, 0.0, 1.0))
    decay = nb.math('ADD', nb.math('MULTIPLY', nb.math('POWER', decay, 6.0), 0.8), nb.math('MULTIPLY', decay, 0.2))
    col = nb.ramp(u, [(0.0, core_col), (0.10, core_col), (0.3, mid_col), (1.0, tail_col)])
    # ragged, turbulent silhouette: erode the rim with the advected turbulence (no smooth lathe outline)
    en = nb.noise(adv.outputs[0], scale=3.2, detail=4, rough=0.6, dim='4D')
    en.name = "Flow2"
    rim = _ss(nb, nb.math('ADD', facing, nb.math('MULTIPLY', nb.math('SUBTRACT', en.outputs['Fac'], 0.5), 0.9)), 0.12, 0.55)
    amt = nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('MULTIPLY', core, tfac), dia), decay), rim)
    g = nb.new('ShaderNodeMath')
    g.operation = 'MULTIPLY'
    g.name = "Gain"
    nb.link(amt, g.inputs[0])
    g.inputs[1].default_value = intensity * 30.0
    em = nb.new('ShaderNodeEmission')
    nb.link(col, em.inputs[0])
    nb.link(g.outputs[0], em.inputs[1])
    tr = nb.new('ShaderNodeBsdfTransparent')
    add = nb.new('ShaderNodeAddShader')
    nb.link(em.outputs[0], add.inputs[0])
    nb.link(tr.outputs[0], add.inputs[1])
    out = nb.new('ShaderNodeOutputMaterial')
    nb.link(add.outputs[0], out.inputs['Surface'])
    return m


def plume_mesh(name, length=40.0, r0=0.62, r_max=3.2, neck=0.3, segs=64, n=40, mat=None, parent=None, r_mid=None, knee=0.2):
    """Lathe plume shell: starts at exit radius r0 (z=0), optional neck, widens quickly to r_mid by `knee` of the
    length, then slowly to r_max at the tail (z=-length)."""
    r_mid = r_mid if r_mid is not None else r0 + (r_max - r0) * knee
    prof = []
    for i in range(n + 1):
        t = (i / n) ** 1.4
        z = -length * t
        k = sb.smooth(t / knee)
        r = r0 + (r_mid - r0) * k + (r_max - r_mid) * max(0.0, t - knee) / (1 - knee)
        r *= (1 - neck * math.sin(min(1.0, t * 12) * math.pi) * (1 - t))
        prof.append((r, z))
    prof = list(reversed(prof))
    o = lathe(name, prof, segs=segs, mat=mat, parent=parent, auto=0)
    o.visible_shadow = False
    return o


def flame_volume_material(name="FlameVol", L0=60.0, r0=2.3, r_mid=4.6, knee=9.0, spread=0.07, soot=1.0):
    """Volumetric exhaust flame (EEVEE/Cycles). Object coords in metres; flame axis = local -Z from z=0 (just below
    the nozzle exits). Turbulent emissive body (white-yellow -> orange -> deep red downstream) with sooty, absorbing
    wisps on the outer/downstream regions. Nodes to drive per frame: 'Gain' (Math, input 1: emission scale),
    'Adv' (Mapping Location z: downstream advection, metres), 'Flow' (noise W)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    co = nb.coord('Object')
    x, y, z = _xyz(nb, co)
    zd = nb.math('MULTIPLY', z, -1.0)                                   # downstream distance (m)
    u = nb.math('DIVIDE', zd, L0)
    rho = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', x, x), nb.math('MULTIPLY', y, y)))
    R = nb.math('ADD', nb.math('ADD', r0, nb.math('MULTIPLY', _ss(nb, zd, 0.0, knee), r_mid - r0)),
                nb.math('MULTIPLY', nb.math('MAXIMUM', nb.math('SUBTRACT', zd, knee), 0.0), spread))
    adv = nb.new('ShaderNodeMapping')
    adv.name = "Adv"
    nb.link(co, adv.inputs[0])
    adv.inputs['Scale'].default_value = (0.42, 0.42, 0.16)
    t1 = nb.noise(adv.outputs[0], scale=1.0, detail=5, rough=0.58, dist=0.4, dim='4D')
    t1.name = "Flow"
    t2 = nb.noise(nb.vmath('SCALE', adv.outputs[0], scale=3.1), scale=1.0, detail=3, rough=0.5, dim='4D')
    t2.name = "Flow2"
    amp = nb.math('ADD', 0.6, nb.math('MULTIPLY', zd, 0.07))
    disp = nb.math('MULTIPLY', nb.math('ADD', nb.math('SUBTRACT', t1.outputs['Fac'], 0.5),
                                       nb.math('MULTIPLY', nb.math('SUBTRACT', t2.outputs['Fac'], 0.5), 0.45)), nb.math('MULTIPLY', amp, 2.4))
    edge = nb.math('SUBTRACT', nb.math('ADD', R, disp), rho)            # >0 inside
    body = _ss(nb, edge, 0.0, 0.9)
    tail = nb.math('SUBTRACT', 1.0, _ss(nb, u, 0.65, 1.0))
    body = nb.math('MULTIPLY', body, tail)
    # temperature: hot near the exit and on the axis, cooling downstream; turbulent
    rr = nb.math('DIVIDE', rho, nb.math('MAXIMUM', R, 0.5))
    temp = nb.math('MULTIPLY', nb.math('POWER', nb.math('SUBTRACT', 1.0, nb.math('MINIMUM', u, 1.0)), 1.6),
                   nb.math('SUBTRACT', 1.15, nb.math('MULTIPLY', rr, 0.55)))
    temp = nb.math('MULTIPLY', temp, nb.maprange(t2.outputs['Fac'], 0.25, 0.75, 0.6, 1.25))
    col = nb.ramp(temp, [(0.0, (0.30, 0.04, 0.005)), (0.18, (0.80, 0.14, 0.015)), (0.42, (1.0, 0.30, 0.04)),
                         (0.72, (1.0, 0.52, 0.14)), (1.0, (1.0, 0.76, 0.42))])
    g = nb.new('ShaderNodeMath')
    g.operation = 'MULTIPLY'
    g.name = "Gain"
    em = nb.math('MULTIPLY', body, nb.math('ADD', 0.015, nb.math('POWER', temp, 2.2)))
    nb.link(em, g.inputs[0])
    g.inputs[1].default_value = 3.0
    # soot: absorbing wisps where the flame has cooled (outer skin, downstream), gated by noise
    sw = nb.maprange(t1.outputs['Fac'], 0.45, 0.62, 0.0, 1.0)
    sootm = nb.math('MULTIPLY', nb.math('MULTIPLY', sw, _ss(nb, u, 0.12, 0.45)),
                    nb.math('SUBTRACT', 1.0, _ss(nb, temp, 0.15, 0.45)))
    dens = nb.math('MULTIPLY', body, nb.math('ADD', 0.05, nb.math('MULTIPLY', sootm, 0.9 * soot)))
    pv = nb.new('ShaderNodeVolumePrincipled')
    pv.inputs['Color'].default_value = (0.16, 0.13, 0.11, 1)
    pv.inputs['Anisotropy'].default_value = 0.0
    nb.link(dens, pv.inputs['Density'])
    nb.link(col, pv.inputs['Emission Color'])
    nb.link(g.outputs[0], pv.inputs['Emission Strength'])
    out = nb.new('ShaderNodeOutputMaterial')
    nb.link(pv.outputs[0], out.inputs['Volume'])
    return m


def flame_volume(name="FlameVol", L0=60.0, half=13.0, mat=None, parent=None):
    """Box domain (metres, not scaled) for flame_volume_material: z from +1 to -L0, x/y in +/-half. Scale local Z to
    lengthen/shorten the flame without changing its shader coordinates' units near the nozzle."""
    v = [(sx * half, sy * half, zz) for zz in (1.0, -L0) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    o = sb.mesh_obj(name, v, f, mat, smooth_shade=False)
    if parent:
        o.parent = parent
    o.visible_shadow = False
    return o


# =========================================================================== machined-part helper

def extrude(name, pts, thick, origin, ax_u, ax_v, mat=None, parent=None, bevel=0.012, segs=2, arc_res=None):
    """Extrude a 2D polygon (list of (u, v), CCW) by `thick` along ax_u x ax_v, centred on `origin`.
    A bevel modifier (angle-limited) softens every hard edge so machined parts catch highlights like real metal."""
    o_ = V(origin)
    u = V(ax_u).normalized()
    v = V(ax_v).normalized()
    n = u.cross(v).normalized()
    bm = bmesh.new()
    fr = [bm.verts.new(o_ + u * a + v * b + n * (thick / 2)) for a, b in pts]
    bk = [bm.verts.new(o_ + u * a + v * b - n * (thick / 2)) for a, b in pts]
    bm.faces.new(fr)
    bm.faces.new(list(reversed(bk)))
    k = len(pts)
    for i in range(k):
        j = (i + 1) % k
        bm.faces.new((bk[i], bk[j], fr[j], fr[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    if mat:
        me.materials.append(mat)
    if bevel:
        b_ = o.modifiers.new("Bevel", 'BEVEL')
        b_.width = bevel
        b_.segments = segs
        b_.limit_method = 'ANGLE'
        b_.angle_limit = math.radians(30)
        b_.harden_normals = True
        try:
            o.data.use_auto_smooth = True
        except Exception:
            pass
        o.modifiers.new("WN", 'WEIGHTED_NORMAL')
    for p in me.polygons:
        p.use_smooth = True
    if parent:
        o.parent = parent
    return o


def arc_pts(c, r, a0, a1, n=10):
    return [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]
