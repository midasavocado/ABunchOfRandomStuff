"""MARS at every scale, one continuous surface: the colony sits at the scene origin, the planet centre is at
(0, 0, -R). The same albedo function (a unit-vector field on the sphere) drives

  * cap(): a real-scale terrain disc (150 km radius, polar grid, exponentially finer toward the colony) with
    numpy relief near the site (dunes, craters, a mesa escarpment, a far crater rim), vertices exactly on the
    sphere so it meets the planet seamlessly; local detail (rocks, ripples, drifts) on top of the regional albedo;
  * planet(): the whole planet (UV sphere with the cap region removed) for orbit views;
  * atmosphere(): a thin limb shell: butterscotch on the day side, BLUE forward-scattered glow around the sun at the
    terminator (Martian sunrises are blue);
  * world(): Mars daytime sky mixed with deep space (stars + sun disc) by a keyable 'SpaceMix' value, so one camera
    can rise from the ground into orbit.

Geography (relative to the colony, which is at lat -2 deg, lon 0): Valles Marineris ~300 km south running east-west,
the Tharsis volcanoes and Olympus Mons to the west, polar caps along scene +/-Y. Keep the numbers here in sync."""
import bpy, math
import numpy as np
from mathutils import Vector as V, Matrix
import sb, space

R = 3389.5e3
C = V((0.0, 0.0, -R))
LAT0 = -2.0
CAP_R = 150e3                        # terrain disc radius (m)
HAZE = (0.62, 0.40, 0.24)


def _basis():
    lat = math.radians(LAT0)
    P = V((0.0, math.cos(lat), -math.sin(lat))).normalized()          # north pole (colony n = +Z has lat LAT0)
    n0 = V((0, 0, 1))
    Q = (n0 - P * P.dot(n0)).normalized()
    E = P.cross(Q).normalized()
    return P, Q, E


POLE, QREF, EAST = _basis()


def ll(lat, lon):
    """unit vector (planet frame, scene axes) for areographic lat/lon in degrees relative to the colony meridian."""
    la, lo = math.radians(lat), math.radians(lon)
    return (QREF * math.cos(lo) + EAST * math.sin(lo)) * math.cos(la) + POLE * math.sin(la)


VOLCANOES = [(ll(18.0, -95.0), 0.105, 9000.0), (ll(1.0, -52.0), 0.045, 6000.0), (ll(-3.0, -60.0), 0.05, 6500.0),
             (ll(6.0, -45.0), 0.04, 5500.0)]          # (centre, angular radius, height m): Olympus, Tharsis Montes
CANYON = (-7.5, 0.024, ll(-7.5, 12.0), math.cos(math.radians(34.0)))     # lat, half-width (sin units), centre, extent


# ============================================================================== albedo / height on the sphere
def albedo(nb, n, detail=1.0):
    """n: unit-vector socket. Returns (color socket, height socket in metres)."""
    s = nb.vmath('DOT_PRODUCT', n, tuple(POLE))                          # sin(latitude)
    warp = nb.noise(n, scale=1.3, detail=3, rough=0.5)
    nw = nb.vmath('ADD', n, nb.vmath('SCALE', nb.vmath('SUBTRACT', warp.outputs['Color'], (0.5, 0.5, 0.5)), scale=0.35))
    reg = nb.noise(nw, scale=1.7, detail=8 * detail, rough=0.58)
    streak = nb.noise(nb.vmath('ADD', nw, (3.0, 1.0, 7.0)), scale=11.0, detail=6 * detail, rough=0.6)
    fine = nb.noise(n, scale=80.0, detail=6 * detail, rough=0.62)
    # dichotomy: darker, more cratered southern highlands; dark 'mare' regions from the regional noise
    south = nb.maprange(s, 0.15, -0.45)
    dark = nb.math('ADD', nb.maprange(reg.outputs['Fac'], 0.52, 0.575), nb.math('MULTIPLY', south, 0.18))
    dark = nb.math('MULTIPLY', dark, nb.maprange(streak.outputs['Fac'], 0.35, 0.65, 0.7, 1.2), clamp=True)
    # wind streaks: noise stretched along the local east direction (tails behind craters, dust fans)
    wind = nb.noise(nb.vmath('ADD', nb.vmath('SCALE', n, scale=1.0), nb.vmath('SCALE', nb.vmath('CROSS_PRODUCT', n, tuple(POLE)), scale=0.0)), scale=30.0, detail=5, rough=0.6)
    col = nb.ramp(nb.maprange(nb.math('ADD', nb.math('MULTIPLY', fine.outputs['Fac'], 0.4), nb.math('MULTIPLY', wind.outputs['Fac'], 0.6)), 0.3, 0.7),
                  [(0.0, (0.36, 0.16, 0.07)), (0.45, (0.50, 0.23, 0.10)), (1.0, (0.62, 0.32, 0.15))])
    col = nb.mix(dark, col, (0.14, 0.075, 0.045, 1))
    # bright dust plains in the north
    col = nb.mix(nb.math('MULTIPLY', nb.maprange(s, 0.1, 0.5), nb.maprange(streak.outputs['Fac'], 0.45, 0.6)), col,
                 (0.66, 0.40, 0.22, 1))
    # craters at three scales (depth ~ 0.2 x radius, raised rims, bright rims / dark floors)
    h = nb.math('MULTIPLY', nb.math('SUBTRACT', reg.outputs['Fac'], 0.5), 2500.0)
    crat_alb = None
    for sc_, keep, amp in ((90.0, 0.62, 4200.0), (300.0, 0.45, 1600.0), (1100.0, 0.35, 520.0), (4200.0, 0.3, 140.0)):
        if detail < 0.6 and sc_ > 1000:
            continue
        v = nb.voronoi(nb.vmath('ADD', n, (sc_ * 0.001, 0.3, 0.0)), scale=sc_, feature='F1')
        sep = nb.new('ShaderNodeSeparateColor'); nb.link(v.outputs['Color'], sep.inputs[0])
        rad = nb.maprange(sep.outputs[0], 0.0, 1.0, 0.12, 0.42)
        on = nb.math('GREATER_THAN', sep.outputs[1], keep)
        on = nb.math('MULTIPLY', on, nb.math('ADD', 0.35, nb.math('MULTIPLY', south, 0.8)), clamp=True)
        x = nb.math('DIVIDE', v.outputs['Distance'], rad)                # 0 centre .. 1 rim
        bowl = nb.math('MULTIPLY', nb.math('SUBTRACT', nb.math('MULTIPLY', x, x), 1.0), nb.math('LESS_THAN', x, 1.0))
        rim = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('POWER', nb.math('SUBTRACT', x, 1.0), 2.0), -18.0))
        prof = nb.math('ADD', bowl, nb.math('MULTIPLY', rim, 0.35))
        h = nb.math('ADD', h, nb.math('MULTIPLY', nb.math('MULTIPLY', prof, on), amp))
        ca = nb.math('MULTIPLY', nb.math('SUBTRACT', nb.math('MULTIPLY', rim, 0.5), nb.math('MULTIPLY', nb.math('LESS_THAN', x, 0.8), 0.25)), on)
        crat_alb = ca if crat_alb is None else nb.math('ADD', crat_alb, ca)
    col = nb.mix(nb.math('MULTIPLY', crat_alb, 0.6, clamp=True), col, (0.68, 0.44, 0.26, 1)) if crat_alb else col
    col = nb.mix(nb.math('MULTIPLY', crat_alb, -0.8, clamp=True), col, (0.10, 0.06, 0.04, 1)) if crat_alb else col
    # Valles Marineris: ragged east-west chasm system, dark floor, layered bright walls
    lat_c, hw, cc, ext = CANYON
    jag = nb.math('MULTIPLY', nb.math('SUBTRACT', nb.noise(n, scale=35.0, detail=7, rough=0.65).outputs['Fac'], 0.5), 0.05)
    dl = nb.math('ABSOLUTE', nb.math('ADD', nb.math('SUBTRACT', s, math.sin(math.radians(lat_c))), jag))
    along = nb.maprange(nb.vmath('DOT_PRODUCT', n, tuple(cc)), ext, ext + 0.05)
    width = nb.math('MULTIPLY', hw, nb.maprange(nb.noise(n, scale=9.0, detail=3).outputs['Fac'], 0.3, 0.7, 0.4, 1.4))
    can = nb.math('MULTIPLY', nb.math('DIVIDE', nb.math('SUBTRACT', width, dl), nb.math('MULTIPLY', width, 0.65), clamp=True), along)
    wall = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('MULTIPLY', can, 2.0), 1.0))), 1.0)
    col = nb.mix(nb.math('MULTIPLY', can, 0.85), col, (0.11, 0.07, 0.05, 1))
    col = nb.mix(nb.math('MULTIPLY', wall, 0.45), col, (0.64, 0.42, 0.26, 1))
    h = nb.math('SUBTRACT', h, nb.math('MULTIPLY', can, 7000.0))
    # Tharsis shields: broad cones with summit calderas, slightly brighter slopes
    for c_, ar, hm in VOLCANOES:
        dd = nb.math('SUBTRACT', 1.0, nb.vmath('DOT_PRODUCT', n, tuple(c_)))       # ~ ang^2/2
        a = nb.math('DIVIDE', nb.math('SQRT', nb.math('MULTIPLY', nb.math('MAXIMUM', dd, 0.0), 2.0)), ar)
        cone = nb.math('POWER', nb.maprange(a, 1.0, 0.0), 1.6)
        cald = nb.maprange(a, 0.09, 0.06)
        h = nb.math('ADD', h, nb.math('MULTIPLY', nb.math('SUBTRACT', cone, nb.math('MULTIPLY', cald, 0.12)), hm))
        col = nb.mix(nb.math('MULTIPLY', cone, 0.35), col, (0.56, 0.36, 0.22, 1))
        col = nb.mix(nb.math('MULTIPLY', nb.maprange(a, 1.25, 1.0), nb.maprange(a, 0.9, 1.1)), col, (0.18, 0.11, 0.07, 1))
    # polar caps (north larger), ragged edges, dusty spiral troughs hinted by noise
    cedge = nb.math('MULTIPLY', nb.math('SUBTRACT', nb.noise(n, scale=18.0, detail=5).outputs['Fac'], 0.5), 0.05)
    capn = nb.maprange(nb.math('ADD', s, cedge), 0.945, 0.955)
    caps = nb.maprange(nb.math('ADD', nb.math('MULTIPLY', s, -1.0), cedge), 0.97, 0.978)
    icec = nb.mix(nb.maprange(nb.noise(n, scale=40.0, detail=4).outputs['Fac'], 0.4, 0.62), (0.92, 0.88, 0.84, 1), (0.78, 0.66, 0.56, 1))
    col = nb.mix(nb.math('MAXIMUM', capn, caps), col, icec)
    return col, h


def planet_mat(name, offset=(0.0, 0.0, 0.0), detail=1.0, local=False, haze=None, bump_k=1.0):
    """Planet surface. offset: added to Object coordinates to get the planet-centred position (cap: (0,0,R)).
    local: add ground-scale detail from the (precise) object coordinates. haze: name of a keyable value node gain."""
    m = sb.mat(name, (0.4, 0.22, 0.1), rough=0.93, spec=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    pos = nb.vmath('ADD', co, tuple(offset)) if any(offset) else co
    n = nb.vmath('NORMALIZE', pos)
    col, h = albedo(nb, n, detail)
    bumpn = nb.bump(h, strength=1.0, distance=(2.5e-4 if local else 3.0e-3) * bump_k)   # bump_k: radius / R for a scaled globe
    if local:
        # ground detail, faded in near the camera; the regional albedo stays underneath
        n1 = nb.noise(co, scale=0.02, detail=6, rough=0.6)
        n2 = nb.noise(co, scale=0.35, detail=5, rough=0.6)
        drift = nb.maprange(nb.noise(co, scale=0.06, detail=4, rough=0.55).outputs['Fac'], 0.55, 0.7)
        k = nb.maprange(n1.outputs['Fac'], 0.3, 0.7, 0.72, 1.25)
        col = nb.mix(1.0, col, nb.maprange(n2.outputs['Fac'], 0.3, 0.75, 0.75, 1.2), blend='MULTIPLY')
        col = nb.mix(1.0, col, k, blend='MULTIPLY')
        col = nb.mix(nb.math('MULTIPLY', drift, 0.5), col, (0.62, 0.36, 0.19, 1))
        cam = nb.coord('Camera')
        dist = nb.vmath('LENGTH', cam)
        near = nb.maprange(dist, 1500.0, 150.0)
        rip = nb.wave(nb.mapping(co, scale=(1.0, 0.35, 1.0)), scale=2.4, wtype='BANDS', direction='Y', dist=4.0, detail=2)
        pebble = nb.voronoi(co, scale=9.0, feature='F1')
        hl = nb.math('ADD', nb.noise(co, scale=3.0, detail=6).outputs['Fac'], nb.math('MULTIPLY', nb.math('MULTIPLY', rip.outputs['Fac'], drift), 0.8))
        hl = nb.math('ADD', hl, nb.math('MULTIPLY', nb.maprange(pebble.outputs['Distance'], 0.0, 0.25, -0.4, 0.0), 1.0))
        bumpn = nb.bump(nb.math('MULTIPLY', hl, near), strength=0.5, distance=0.25, normal=bumpn)
        if haze:
            gain = nb.new('ShaderNodeValue'); gain.name = haze; gain.outputs[0].default_value = 1.0
            hz = nb.math('SUBTRACT', 1.0, nb.math('EXPONENT', nb.math('DIVIDE', dist, -14000.0)))
            hz = nb.math('MULTIPLY', hz, gain.outputs[0])
            col = nb.mix(nb.math('MULTIPLY', hz, 0.6), col, (*HAZE, 1))
            nb.set('Emission Color', (*HAZE, 1))
            nb.set('Emission Strength', nb.math('MULTIPLY', nb.math('MULTIPLY', hz, hz), 0.25))
    nb.set('Base Color', col)
    nb.set('Normal', bumpn)
    return m


# ============================================================================== terrain near the colony
def fbm(x, y, base, octaves=4, seed=0, gain=0.5, lac=2.03):
    """numpy value-noise fbm in [-1, 1]-ish: base = largest feature size (m)."""
    rs = np.random.default_rng(seed)
    perm = rs.permutation(1024)
    vals = rs.uniform(-1.0, 1.0, 1024)
    out = np.zeros_like(x, dtype=float)
    amp, f, tot = 1.0, 1.0 / base, 0.0
    for o in range(octaves):
        X = x * f + 17.3 * o; Y = y * f - 9.1 * o
        ix = np.floor(X).astype(np.int64); iy = np.floor(Y).astype(np.int64)
        fx = X - ix; fy = Y - iy
        ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10); uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)

        def h(i, j):
            return vals[perm[(perm[i & 1023] + j) & 1023]]
        v00, v10, v01, v11 = h(ix, iy), h(ix + 1, iy), h(ix, iy + 1), h(ix + 1, iy + 1)
        out += amp * ((v00 * (1 - ux) + v10 * ux) * (1 - uy) + (v01 * (1 - ux) + v11 * ux) * uy)
        tot += amp; amp *= gain; f *= lac
    return out / tot


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Terrain:
    """local relief (metres, tangent plane at the colony) -- also used to seat buildings, pads, rocks.
    Plain around the colony (levelled inside LEVEL m), fbm undulation, mesa outliers and a wandering eroded
    escarpment 4-9 km north, craters of all sizes, a dune field NE, one big old crater rim on the horizon."""
    LEVEL = 650.0

    def __init__(self, seed=4):
        rs = np.random.default_rng(seed)
        self.cr = []                                                  # craters: x, y, radius
        for _ in range(700):
            r = 8.0 * rs.pareto(1.4) + 5.0
            d = 800.0 + 40000.0 * rs.uniform(0, 1) ** 1.8
            a = rs.uniform(0, 2 * math.pi)
            self.cr.append((d * math.cos(a), d * math.sin(a), min(r, 3000.0)))
        self.cr.append((-21000.0, 26000.0, 9000.0))                 # the big old crater whose rim closes the horizon
        self.cr = np.array(self.cr)

    def height(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        r = np.hypot(x, y)
        h = 30.0 * fbm(x, y, 9000.0, 3, seed=1) + 6.0 * fbm(x, y, 900.0, 4, seed=2) + 0.9 * fbm(x, y, 70.0, 3, seed=3)
        # dune field NE (transverse ridges, crescent-bent by noise)
        dmask = _ss(0.0, 0.5, 1 - np.hypot((x - 2600) / 2200, (y - 1800) / 1400))
        ph = (x * 0.8 + y * 0.6) * 0.045 + 3.0 * fbm(x, y, 700.0, 2, seed=5)
        h += dmask * 3.4 * np.abs(np.sin(ph)) ** 1.5
        # eroded escarpment + mesa outliers (flat caprock, steep walls, talus)
        yedge = 5400.0 + 1500.0 * fbm(x, np.zeros_like(x), 6000.0, 4, seed=7)
        scarp = _ss(-250.0, 250.0, y - yedge + 700.0 * fbm(x, y, 900.0, 4, seed=11))
        mes = _ss(0.20, 0.27, fbm(x, y, 1800.0, 5, seed=8)) * _ss(-4000.0, -1500.0, y - yedge) * _ss(2500.0, 4000.0, r)
        top = np.maximum(scarp, mes)
        h += 230.0 * top * (0.9 + 0.1 * fbm(x, y, 1500.0, 2, seed=9))
        for cx, cy, cr in self.cr:
            m = (np.abs(x - cx) < 2.2 * cr) & (np.abs(y - cy) < 2.2 * cr)
            if not np.any(m):
                continue
            dd = np.hypot(x[m] - cx, y[m] - cy) / cr
            prof = np.where(dd < 1.0, (dd * dd - 1.0) * 0.2, 0.0) + 0.05 * np.exp(-((dd - 1.0) / 0.25) ** 2)
            h[m] += prof * cr
        # levelled colony site + fade to the smooth sphere at the cap edge
        h *= _ss(self.LEVEL, self.LEVEL + 900.0, r) * 0.96 + 0.04
        h *= np.clip((CAP_R * 0.8 - r) / (CAP_R * 0.3), 0, 1)
        return h


def cap(name="MarsCap", T=None, mat=None, nr=420, na=900, r0=1.2):
    """polar-grid terrain disc on the sphere; returns object (origin at the colony)."""
    import suit
    T = T or Terrain()
    k = (CAP_R / r0) ** (1.0 / (nr - 1))
    rr = np.concatenate([[0.0], r0 * k ** np.arange(nr)])
    ph = np.linspace(0, 2 * math.pi, na, endpoint=False)
    Rg, Pg = np.meshgrid(rr[1:], ph, indexing='ij')
    X, Y = Rg * np.cos(Pg), Rg * np.sin(Pg)
    Hh = T.height(X, Y)
    th = Rg / R
    rad = R + Hh
    Vx = rad * np.sin(th) * np.cos(Pg)
    Vy = rad * np.sin(th) * np.sin(Pg)
    Vz = rad * np.cos(th) - R                                    # float64 before subtraction: precise near origin
    verts = np.concatenate([[[0.0, 0.0, float(T.height(np.array([0.0]), np.array([0.0]))[0])]],
                            np.stack([Vx, Vy, Vz], -1).reshape(-1, 3)])
    faces = []
    for j in range(na):
        faces.append((0, 1 + j, 1 + (j + 1) % na, 1 + (j + 1) % na))
    ii = np.arange(nr * na).reshape(nr, na) + 1
    jn = np.roll(ii, -1, axis=1)
    F = np.stack([ii[:-1], ii[1:], jn[1:], jn[:-1]], -1).reshape(-1, 4)
    o = suit.mesh_np(name, verts, F, [mat] if mat else [])
    # centre fan as triangles
    me = o.data
    bm_faces = [(0, 1 + j, 1 + (j + 1) % na) for j in range(na)]
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    for f in bm_faces:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except ValueError:
            pass
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return o


def planet(name="Mars", mat=None, segs=1024, rings=512, hole=True):
    """the whole planet: UV sphere (poles on the colony axis), faces under the terrain cap removed."""
    import bmesh
    o = sb.prim("sphere", name, loc=tuple(C), segments=segs, ring_count=rings, radius=R, mat=mat)
    if hole:
        me = o.data
        bm = bmesh.new(); bm.from_mesh(me)
        lim = CAP_R / R - 2.5 * math.pi / rings
        dead = [f for f in bm.faces if f.calc_center_median().normalized().z > math.cos(lim)]
        bmesh.ops.delete(bm, geom=dead, context='FACES')
        bm.to_mesh(me); bm.free()
    for p in o.data.polygons:
        p.use_smooth = True
    o.visible_shadow = False              # the terminator is N.L shading; a 3400 km shadow caster only hurts EEVEE
    return o


def atmosphere(name="MarsAtmo", sun=(0, 1, 0), thick=42e3, gain=1.0, obj=None):
    """limb glow shell. Node 'AtmoGain' (Value) scales it (key it to 0 while the camera is inside the air).
    obj: use this mesh (e.g. a limb_patch at R + thick, origin at C) instead of a 512-segment sphere."""
    if obj is None:
        o = sb.prim("sphere", name, loc=tuple(C), segments=512, ring_count=256, radius=R + thick)
        for p in o.data.polygons:
            p.use_smooth = True
    else:
        o = obj
    m = bpy.data.materials.new(name + "M")
    m.use_nodes = True
    try:
        m.surface_render_method = 'BLENDED'
    except Exception:
        pass
    m.use_backface_culling = True
    nt = m.node_tree; nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB); nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    S = tuple(V(sun).normalized())
    Ra = R + thick
    HS = 11.1e3                                                   # Mars scale height
    P = nb.coord('Object')                                        # point on the shell, planet-centred (m)
    geo = nb.new('ShaderNodeNewGeometry')
    D = nb.vmath('SCALE', geo.outputs['Incoming'], scale=-1.0)    # view ray direction
    pd = nb.vmath('DOT_PRODUCT', P, D)
    b = nb.math('SQRT', nb.math('MAXIMUM', nb.math('SUBTRACT', Ra * Ra, nb.math('MULTIPLY', pd, pd)), 0.0))
    # column density (normalised to a grazing ray at the surface): Chapman grazing ~ exp(-(b-R)/H); rays that hit
    # the ground see only the column down to the surface, ~ H / cos(zenith)
    above = nb.math('EXPONENT', nb.math('DIVIDE', nb.math('SUBTRACT', R, nb.math('MAXIMUM', b, R)), HS))
    graze_col = math.sqrt(2 * math.pi * R / HS)
    mu = nb.math('SQRT', nb.math('MAXIMUM', nb.math('SUBTRACT', 1.0, nb.math('POWER', nb.math('DIVIDE', nb.math('MINIMUM', b, R), R), 2.0)), 1e-6))
    hit = nb.math('LESS_THAN', b, R)
    down = nb.math('MINIMUM', nb.math('DIVIDE', 1.0 / graze_col, mu), 0.5)
    col = nb.mix(hit, above, down, dtype='FLOAT')
    # illumination: rays that miss the planet at their tangent point (the limb air), rays that hit it at the ground
    # point they end on (air over the night side stays dark), twilight wrapping past the terminator
    Tp = nb.vmath('SUBTRACT', P, nb.vmath('SCALE', D, scale=pd))
    thit = nb.math('SUBTRACT', nb.math('MULTIPLY', pd, -1.0),
                   nb.math('SQRT', nb.math('MAXIMUM', nb.math('SUBTRACT', R * R, nb.math('MULTIPLY', b, b)), 0.0)))
    G = nb.vmath('ADD', P, nb.vmath('SCALE', D, scale=thit))
    Lp = nb.vmath('NORMALIZE', nb.mix(hit, Tp, G, dtype='VECTOR'))
    ns = nb.vmath('DOT_PRODUCT', Lp, S)
    lit = nb.math('POWER', nb.maprange(ns, -0.18, 0.25), 1.5)
    cosv = nb.vmath('DOT_PRODUCT', D, S)
    fwd = nb.math('ADD', nb.math('MULTIPLY', nb.math('POWER', nb.math('MAXIMUM', cosv, 0.0), 12.0), 3.0),
                  nb.math('MULTIPLY', nb.math('POWER', nb.math('MAXIMUM', cosv, 0.0), 120.0), 12.0))
    fwd = nb.math('MULTIPLY', fwd, nb.maprange(ns, -0.3, 0.05))            # the blue dawn arc hugs the terminator
    fwd = nb.math('MULTIPLY', fwd, nb.math('SUBTRACT', 1.0, hit))          # only in the limb air, not over the disc
    day = nb.math('MULTIPLY', nb.math('MULTIPLY', col, lit), 0.55)
    blue = nb.math('MULTIPLY', col, fwd)
    c1 = nb.new('ShaderNodeEmission'); c1.inputs[0].default_value = (0.86, 0.56, 0.34, 1)
    nb.link(day, c1.inputs[1])
    c2 = nb.new('ShaderNodeEmission'); c2.inputs[0].default_value = (0.42, 0.64, 1.0, 1)
    nb.link(blue, c2.inputs[1])
    a = nb.new('ShaderNodeAddShader'); nb.link(c1.outputs[0], a.inputs[0]); nb.link(c2.outputs[0], a.inputs[1])
    g = nb.new('ShaderNodeValue'); g.name = "AtmoGain"; g.outputs[0].default_value = gain
    tr = nb.new('ShaderNodeBsdfTransparent')
    mx = nb.new('ShaderNodeMixShader')
    nb.link(g.outputs[0], mx.inputs[0]); nb.link(tr.outputs[0], mx.inputs[1])
    ad = nb.new('ShaderNodeAddShader'); nb.link(a.outputs[0], ad.inputs[0]); nb.link(tr.outputs[0], ad.inputs[1])
    nb.link(ad.outputs[0], mx.inputs[2])
    out = nb.new('ShaderNodeOutputMaterial'); nb.link(mx.outputs[0], out.inputs[0])
    o.data.materials.append(m)
    o.visible_shadow = False
    o.visible_diffuse = False
    return o, g


# ============================================================================== sky
def world(sun, space_mix=0.0, sky_strength=1.4, stars=1.0, sun_disc=True):
    """Mars daytime sky (butterscotch, blue-white aureole) mixed into deep space (stars + sun disc).
    Returns (world, SpaceMix value node). Key SpaceMix.outputs[0].default_value 0 (ground) .. 1 (space)."""
    w = space.starfield("MarsWorld", strength=stars)
    nt = w.node_tree
    nb = sb.NB.__new__(sb.NB); nb.m, nb.nt, nb.n, nb.l = w, nt, nt.nodes, nt.links
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    star_sh = out.inputs[0].links[0].from_socket
    S = V(sun).normalized()
    tc = nb.new('ShaderNodeTexCoord')
    d = nb.vmath('NORMALIZE', tc.outputs['Generated'])
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(d, sep.inputs[0])
    zz = nb.maprange(sep.outputs[2], -0.05, 1.0)
    sky = nb.ramp(zz, [(0.0, (0.16, 0.08, 0.035)), (0.05, (0.52, 0.29, 0.135)), (0.14, (0.44, 0.24, 0.11)),
                       (0.5, (0.28, 0.15, 0.07)), (1.0, (0.18, 0.10, 0.05))])
    sdp = nb.vmath('DOT_PRODUCT', d, tuple(S))
    halo = nb.math('POWER', nb.math('MAXIMUM', sdp, 0.0), 60.0)
    sky = nb.mix(nb.math('MULTIPLY', halo, 0.8), sky, (0.55, 0.58, 0.62, 1), blend='ADD', clamp=False)
    # low sun: the whole sky dims and the aureole turns blue (Martian sunset/sunrise)
    low = nb.math('SUBTRACT', 1.0, nb.maprange(S.z, 0.0, 0.3)) if False else None
    bg = nb.new('ShaderNodeBackground'); bg.inputs[1].default_value = sky_strength
    nb.link(sky, bg.inputs[0])
    # sun disc + soft glow in space (camera rays only; the lamp lights the scene)
    disc = nb.maprange(sdp, math.cos(math.radians(0.36)), math.cos(math.radians(0.30)), 0.0, 1.0)
    glow = nb.math('ADD', nb.math('MULTIPLY', nb.math('POWER', nb.math('MAXIMUM', sdp, 0.0), 4000.0), 12.0),
                   nb.math('MULTIPLY', nb.math('POWER', nb.math('MAXIMUM', sdp, 0.0), 300.0), 0.6))
    se = nb.new('ShaderNodeEmission'); se.inputs[0].default_value = (1.0, 0.96, 0.9, 1)
    vis = nb.new('ShaderNodeValue'); vis.name = "SunVis"; vis.outputs[0].default_value = 1.0     # key 0 while occulted
    nb.link(nb.math('ADD', nb.math('MULTIPLY', disc, 3000.0 if sun_disc else 0.0), nb.math('MULTIPLY', glow, vis.outputs[0])), se.inputs[1])
    lp = nb.new('ShaderNodeLightPath')
    blk = nb.new('ShaderNodeBackground'); blk.inputs[0].default_value = (0, 0, 0, 1)
    sm = nb.new('ShaderNodeMixShader'); nb.link(lp.outputs['Is Camera Ray'], sm.inputs[0])
    nb.link(blk.outputs[0], sm.inputs[1]); nb.link(se.outputs[0], sm.inputs[2])
    spc = nb.new('ShaderNodeAddShader'); nb.link(star_sh, spc.inputs[0]); nb.link(sm.outputs[0], spc.inputs[1])
    v = nb.new('ShaderNodeValue'); v.name = "SpaceMix"; v.outputs[0].default_value = space_mix
    mx = nb.new('ShaderNodeMixShader'); nb.link(v.outputs[0], mx.inputs[0])
    nb.link(bg.outputs[0], mx.inputs[1]); nb.link(spc.outputs[0], mx.inputs[2])
    for l in list(out.inputs[0].links):
        nt.links.remove(l)
    nb.link(mx.outputs[0], out.inputs[0])
    try:
        w.sun_threshold = 1.0e9
    except Exception:
        pass
    return w, v


def ground_mats(prefix="MC"):
    """shared colony materials: dusted hab white, graphite, shield regolith (same albedo as the ground), lamps."""
    import suit, moon
    M = suit.materials(dust=0.6)
    return dict(white=suit.graphite_mat(prefix + "White", color=(0.70, 0.64, 0.58), wear=0.3, dust=0.5),
                graphite=M["graphite"], shield=planet_mat(prefix + "Shield", offset=(0, 0, R), local=True),
                window=sb.emit_mat(prefix + "Win", (1.0, 0.75, 0.45), 3.0), lamp=sb.emit_mat(prefix + "Lamp", (1.0, 0.95, 0.9), 8.0),
                cells=moon.cells_mat(prefix + "Cells"), concrete=sb.painted(prefix + "Pad", (0.30, 0.22, 0.17), rough=0.85, grime=0.7),
                glass=sb.glass(prefix + "Glass", color=(0.9, 0.85, 0.8), rough=0.05))


def aim(cam, loc, target, up=None, frame=None):
    """point a camera from loc at target with a chosen world 'up' (default: Mars north), optionally keyed."""
    up = V(up) if up is not None else POLE
    f = (V(target) - V(loc)).normalized()
    r = f.cross(up).normalized()
    u = r.cross(f)
    M = Matrix((r, u, -f)).transposed()
    cam.rotation_mode = 'QUATERNION'
    cam.location = loc
    cam.rotation_quaternion = M.to_quaternion()
    if frame is not None:
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_quaternion", frame=frame)


SUN_DAY = (QREF * 0.55 + EAST * 0.8 + POLE * 0.15).normalized()     # afternoon at the colony (~33 deg up, from the east)


def limb_patch(name, center_dir, ref_dir, th0, th1, th_limb, az_half, n_th=520, n_az=1600, radius=R, mat=None, focus=0.12):
    """Partial sphere (origin at C) around center_dir for close orbital views: polar angle th0..th1 (radians from
    center_dir), azimuth +-az_half around ref_dir, rings packed around th_limb (the camera's limb), so a 4K frame
    never sees a facet on the horizon."""
    import suit
    c = V(center_dir).normalized()
    r0 = (V(ref_dir) - c * c.dot(V(ref_dir))).normalized()
    r1 = c.cross(r0)
    u = np.linspace(-1.0, 1.0, n_th)
    # sinh-spaced around the limb
    k = np.sinh(u * 3.0) / np.sinh(3.0)
    th = np.where(k < 0, th_limb + k * (th_limb - th0), th_limb + k * (th1 - th_limb))
    az = np.linspace(-az_half, az_half, n_az)
    TH, AZ = np.meshgrid(th, az, indexing='ij')
    cvec, r0v, r1v = np.array(c), np.array(r0), np.array(r1)
    d = (np.cos(TH)[..., None] * cvec + np.sin(TH)[..., None] * (np.cos(AZ)[..., None] * r0v + np.sin(AZ)[..., None] * r1v))
    Vt = (d * radius).reshape(-1, 3)
    ii = np.arange(n_th * n_az).reshape(n_th, n_az)
    F = np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]], -1).reshape(-1, 4)
    o = suit.mesh_np(name, Vt, F, [mat] if mat else [])
    o.location = C
    me = o.data
    p0 = me.polygons[len(me.polygons) // 2]
    if p0.normal.dot(p0.center) < 0:                      # outward normals
        import bmesh
        bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return o
