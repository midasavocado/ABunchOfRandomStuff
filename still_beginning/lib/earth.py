"""EARTH — analytic, physically based planet rendered in the WORLD shader (EARTH & ORBIT team).

The planet is ray-traced per pixel inside the world node tree: ray/sphere intersection from a camera
position you pass in (km, Earth-centred), view direction = TexCoord 'Generated'. This gives exact
perspective/parallax at any altitude (40 km … lunar distance), no depth-precision problems, and — because
the same world feeds EEVEE's light probe — correct blue/white EARTHSHINE on the scene for free.

Contents
  * single-scattering atmosphere raymarch (Rayleigh + Mie + ozone), sun transmittance via Schüler's
    Chapman approximation (so the terminator, twilight arc and sunrise limb turn AMBER naturally)
  * procedural continents (domain-warped fbm on the sphere, climate-driven albedo, mountains, ice caps)
  * ocean with Beckmann sun glint (wind-varying roughness), shelf seas
  * cloud shell at R+hc with weather-scale structure, sun-ward relief shading and cast shadows
  * optional sparse city lights on the night side, sun disc (camera rays only)

Usage (scene in metres; Earth centre is given in km in scene axes):
    import earth
    E = earth.build(alt_km=400, sun_dir=(0.3, 0.9, 0.1))       # Earth centre at (0,0,-(R+400)) km
    E.sun_lamp()                                               # matching SUN light (0.53 deg, same irradiance)
    ... animate camera ...
    E.track(cam, f0, f1)                                       # bakes camera position into the shader
    E.bake_rot(f0, f1, lambda f: (0, 0, 0.0001 * f))           # optional slow planet rotation (radians)

Helpers: earth.horizon_dir(cam_km, azim_deg) -> world direction to the horizon (for aiming cameras),
E.cam_km(loc_m) -> Earth-centred km vector for a scene point.
"""
import bpy, math
from mathutils import Vector as V, Euler
import sb
from nodex import Ctx

R = 6371.0            # km
ATM = 90.0            # km top of atmosphere
HR, HM = 8.0, 1.8     # scale heights (km); aerosol layer a bit thicker than textbook for a visible amber base
BR = (5.802e-3, 13.558e-3, 33.1e-3)   # Rayleigh scattering /km (sea level)
BM_S, BM_E = 3.996e-3, 4.44e-3        # Mie scattering / extinction /km (clear-ish)
BO = (0.650e-3, 1.881e-3, 0.085e-3)   # ozone absorption /km at peak
KO = 1.6                              # ozone column coupling to rayleigh column (approx)
G_MIE = 0.78


def _chapman_c(H):
    return math.sqrt(math.pi * (R + H) / (2 * H))


class Earth:
    pass


def rot_for(center_km, lat=30.0, lon=-20.0, heading=0.0):
    """Euler (XYZ) for the surface lookup so that the point directly below the scene origin is at
    (lat, lon) degrees, with texture-north pointing along scene +Y rotated by heading (deg, clockwise)."""
    from mathutils import Matrix
    up_w = (-V(center_km)).normalized()
    ref = V((0, 1, 0)) if abs(up_w.y) < 0.95 else V((1, 0, 0))
    east_w = ref.cross(up_w).normalized()
    north_w = up_w.cross(east_w).normalized()
    h = math.radians(heading)
    e2 = east_w * math.cos(h) - north_w * math.sin(h)
    n2 = north_w * math.cos(h) + east_w * math.sin(h)
    la, lo = math.radians(lat), math.radians(lon)
    up_t = V((math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))
    east_t = V((-math.sin(lo), math.cos(lo), 0.0))
    north_t = up_t.cross(east_t).normalized()
    W = Matrix((e2, n2, up_w)).transposed()        # columns = world basis
    T = Matrix((east_t, north_t, up_t)).transposed()
    M = T @ W.transposed()
    return tuple(M.to_euler('XYZ'))


def build(alt_km=400.0, center_km=None, sun_dir=(0.0, 1.0, 0.2), rot=None, nadir=(30.0, -20.0, 0.0), sun_strength=5.0,
          sun_color=(1.0, 0.975, 0.95), clouds=0.5, lights=0.0, samples=16, ms=1.05, ms_limb=0.9, grey=0.45, aureole=0.15, strat=0.08, wb=(0.93, 0.975, 1.08), land_gain=1.25, seed=0.0,
          sun_disc=True, disc_strength=1.0, detail=1.0, cloud_height=6.0, probe_res='1024',
          land_offset=(0.0, 0.0, 0.0), debug=None, map_mode=False, name="Earth"):
    """Create the Earth world. Returns an Earth handle.

    alt_km       camera altitude used to place Earth centre at (0,0,-(R+alt_km)) km (scene origin above surface)
    center_km    explicit Earth centre in scene km (overrides alt_km)
    sun_dir      unit vector TO the sun (scene axes)
    rot          Euler (radians) applied to surface/cloud lookup (choose which part of the planet faces you)
    sun_strength irradiance (Blender W/m^2 convention, same as the SUN lamp you add with E.sun_lamp())
    clouds       coverage 0..1;  lights = city-light strength (0 = off);  ms = multiple-scatter boost
    samples      atmosphere raymarch samples (16 is plenty for orbit; 20 for in-atmosphere views)
    detail       multiplier on noise octaves (1 = full)
    """
    sc = bpy.context.scene
    w = bpy.data.worlds.new(name)
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    try:
        w.sun_threshold = 1.0e9      # never auto-extract a sun from the world; we add our own lamp
        w.probe_resolution = probe_res
    except Exception:
        pass
    X = Ctx(nt)
    E = Earth()
    E.world, E.nt, E.X = w, nt, X
    E.center_km = V(center_km) if center_km is not None else V((0.0, 0.0, -(R + alt_km)))
    if rot is None:
        rot = rot_for(E.center_km, *nadir)
    E.sun_strength = sun_strength
    E.sun_color = sun_color

    # ---------------------------------------------------------------- inputs
    tc = X.node('ShaderNodeTexCoord')
    D = X.vec(tc.outputs['Generated']).normalize()
    O, E.cam_node = X.vinput(tuple(-E.center_km), "CamKm")
    L, E.sun_node = X.vinput(tuple(V(sun_dir).normalized()), "SunDir")
    Ln = L.normalize()
    rot_in, E.rot_node = X.vinput(tuple(rot), "EarthRot")
    cov_in, E.cloud_node = X.value(clouds, "CloudCover")
    lights_in, E.lights_node = X.value(lights, "Lights")
    Esun = sun_strength
    scol = sun_color

    b = O.dot(D)
    cr = O.cross(D)
    p2 = cr.dot(cr)
    DL = D.dot(Ln)
    OL = O.dot(Ln)

    # ---------------------------------------------------------------- intersections (s = t + b)
    Rg, Ra, Rc = R, R + ATM, R + cloud_height
    disc_g = (Rg * Rg) - p2
    hitG = disc_g.gt(0.0) * b.lt(0.0)
    sg = -(disc_g.max(0.0).sqrt())                    # ground entry in s
    disc_a = (Ra * Ra) - p2
    sqa = disc_a.max(0.0).sqrt()
    s_start = (-sqa).max(b)
    s_end_atm = sqa
    s_end = X.mix(hitG, s_end_atm, sg).max(s_start)
    inatm = disc_a.gt(0.0) * s_end_atm.gt(b)
    disc_c = (Rc * Rc) - p2
    hitC = disc_c.gt(0.0) * b.lt(0.0)
    s_c = -(disc_c.max(0.0).sqrt())

    # ---------------------------------------------------------------- phase functions
    mu = DL
    PR = (mu * mu + 1.0) * (3.0 / (16.0 * math.pi))
    g = G_MIE
    PM = (mu * mu + 1.0) * (3.0 / (8.0 * math.pi) * (1 - g * g) / (2 + g * g)) / ((mu * (-2 * g) + (1 + g * g)) ** 1.5)
    g2 = 0.96                                   # narrow forward aerosol lobe -> bright limb glow near the sun
    PM2 = ((mu * (-2 * g2) + (1 + g2 * g2)) ** 1.5) * (4.0 * math.pi / (1 - g2 * g2))
    PM = PM * (1.0 - aureole) + (1.0 / PM2) * aureole

    cR, cM = _chapman_c(HR), _chapman_c(HM)
    BRe = tuple(BR[i] + KO * BO[i] for i in range(3))    # rayleigh + coupled ozone (extinction)

    def sun_column(r, mus, rhoR, rhoM):
        """Rayleigh & Mie sun-path columns (in km of sea-level density) and ground-shadow factor."""
        amu = mus.abs()
        neg = mus.lt(0.0)
        sinx = (1.0 - mus * mus).max(0.0).sqrt()
        ht = r * sinx - R                               # tangent height of sun ray (if sun below local horizon)
        out = []
        for c, H, rho in ((cR, HR, rhoR), (cM, HM, rhoM)):
            pos = rho * (c / (amu * (c - 1.0) + 1.0))
            term = ((ht * (-1.0 / H)).min(40.0)).exp() * (2.0 * c)
            col = (pos + neg * (term - pos * 2.0)) * H
            out.append(col)
        lit = 1.0 - neg * (1.0 - X.smoothstep(-4.0, 10.0, ht))
        return out[0], out[1], lit

    def sun_trans(colR, colM):
        tau = X.vec(BRe) * colR + X.vec((BM_E, BM_E, BM_E)) * colM
        return X.vmath('POWER', (1 / math.e,) * 3, tau)

    # ---------------------------------------------------------------- surface point & frames
    Pg = O + D * (sg - b)
    n = Pg * (1.0 / R)
    Pc = O + D * (s_c - b)
    nc = Pc.normalize()
    if map_mode:          # debug: sphere point = view direction (render with an equirect camera)
        n = nc = D
    q_g, _ = X.rotate_euler(n, rot_in)
    q_c, rotc = X.rotate_euler(nc, rot_in)
    if land_offset != (0, 0, 0):
        q_g = q_g + land_offset
    mus_g = n.dot(Ln)
    mus_c = nc.dot(Ln)
    muv_g = -(n.dot(D))
    # sun transmittance at ground / cloud tops
    one = X.f(1.0)
    cRg, cMg, litg = sun_column(X.f(R + 0.3), mus_g, X.f(math.exp(-0.3 / HR)), X.f(math.exp(-0.3 / HM)))
    Tsg = sun_trans(cRg, cMg) * litg * X.vec(wb)
    cRc, cMc, litc = sun_column(X.f(Rc), mus_c, X.f(math.exp(-cloud_height / HR)), X.f(math.exp(-cloud_height / HM)))
    Tsc = sun_trans(cRc, cMc) * litc * X.vec(wb)

    E.q_g, E.q_c = q_g, q_c
    DBG = {}
    E.dbg = DBG
    Lrg, _ = X.rotate_euler(Ln, rot_in)
    Ltg = Lrg - q_g * q_g.dot(Lrg)
    albedo, water, elev = _land(X, q_g, seed, detail, DBG, Ltg, mus_g)
    dens, dens_sh, relief = _clouds(X, q_c, n, Ln, mus_g, rot_in, cov_in, seed, detail, cloud_height, DBG)
    DBG.update(albedo=albedo, water=water, elev=elev, dens=dens, dens_sh=dens_sh)

    # ---------------------------------------------------------------- ground radiance
    inv_pi = 1.0 / math.pi
    lam_g = X.smoothstep(-0.02, 0.06, mus_g) * mus_g.max(0.0)
    sky_amb = X.smoothstep(-0.12, 0.35, mus_g)
    sky_col = X.vec((0.05, 0.085, 0.16))
    cshadow = 1.0 - dens_sh
    Eg = Tsg * (lam_g * cshadow) * (Esun * inv_pi) + sky_col * (sky_amb * (Esun * inv_pi))
    Lg = albedo * Eg * land_gain
    # ocean glint (Beckmann, Fresnel) — roughness varies with 'wind'
    Vv = -D
    Hh = (Ln + Vv).normalize()
    ndh = n.dot(Hh).max(1e-3)
    vdh = Vv.dot(Hh).max(0.0)
    wind, _ = X.noise(q_g, scale=9.0, detail=4 * detail, rough=0.6)
    sig2 = X.f(0.012) + wind * wind * 0.05
    ndh2 = ndh * ndh
    tan2 = (1.0 - ndh2) / ndh2
    Dm = (tan2 / sig2 * -1.0).exp() / (sig2 * ndh2 * ndh2 * math.pi)
    fr = (1.0 - vdh) ** 5.0 * 0.98 + 0.02
    spec = Dm * fr / (muv_g.max(0.06) * 4.0) * X.smoothstep(0.0, 0.05, mus_g)
    Lg = Lg + Tsg * (spec * water * cshadow * Esun)

    # ---------------------------------------------------------------- clouds radiance
    wrap = ((mus_c + 0.08) / 1.08).max(0.0)
    lam_c = X.smoothstep(-0.08, 0.04, mus_c) * wrap
    Ec = Tsc * lam_c * (Esun * inv_pi) + sky_col * (X.smoothstep(-0.12, 0.35, mus_c) * (Esun * inv_pi * 0.7))
    rel = (relief * 7.0 / (mus_c.max(0.0) + 0.3) + 1.0).clamp(0.5, 1.4)
    DBG.update(rel=rel)
    Lc = Ec * rel
    alpha_c = dens * hitC

    # ---------------------------------------------------------------- raymarch
    N = samples
    sstar = X.f(0.0).max(s_start).min(s_end)
    A = (sstar - s_start).max(0.0).sqrt()
    B = (s_end - sstar).max(0.0).sqrt()
    K = (A + B) * (A + B)
    ustar = A / (A + B + 1e-6)
    tau = None
    tau_c = None
    Lin = None
    inv_e = (1 / math.e,) * 3
    mR = sum(BRe) / 3.0
    BRv = tuple(BRe[i] * (1 - grey) + mR * grey for i in range(3))   # partially greyed view extinction (MS proxy)
    for k in range(N):
        uk = (k + 0.5) / N
        q = (ustar * -1.0) + uk
        qa = q.abs()
        sk = (q * qa) * K + sstar
        ds = (qa * K) * (2.0 / N)
        r = (sk * sk + p2).sqrt()
        h = r - R
        rhoR = (h * (-1.0 / HR)).exp()
        rhoM0 = (h * (-1.0 / HM)).exp()
        mus = ((sk - b) * DL + OL) / r
        colR, colM, lit = sun_column(r, mus, rhoR, rhoM0)
        hs_ = (h - 21.0) * (1.0 / 6.0)
        rhoM = rhoM0 + (hs_ * hs_ * -1.0).exp() * strat           # stratospheric aerosol (Junge) layer
        a = rhoR * ds
        m = rhoM * ds
        dtau = X.vec(BRv) * a + X.vec((BM_E,) * 3) * m
        taus = X.vec(BRe) * colR + X.vec((BM_E,) * 3) * colM
        tmid = taus + (dtau * 0.5 if tau is None else tau + dtau * 0.5)
        T = X.vmath('POWER', inv_e, tmid) * lit
        front = s_c.gt(sk)                                   # sample in front of the cloud-shell entry
        occ = 1.0 - alpha_c * (1.0 - front)                  # samples behind a cloud are hidden by it
        dL = (X.vec(BR) * (a * PR * occ) + X.vec((BM_S,) * 3) * (m * PM * occ)) * T
        Lin = dL if Lin is None else Lin + dL
        tau = dtau if tau is None else tau + dtau
        dtc = dtau * front
        tau_c = dtc if tau_c is None else tau_c + dtc
    Tview = X.vmath('POWER', inv_e, tau)
    tmean = tau.dot((1 / 3.0, 1 / 3.0, 1 / 3.0))
    ms_thick = (1.0 - (tmean * -1.0).exp()) * ms_limb + 1.0          # optically thick paths gain more MS energy
    Lin = Lin * (Esun * ms) * inatm * ms_thick
    Lin = Lin * X.vec(scol)
    # transmittance for rays that miss the atmosphere entirely = 1
    Tview = X.mix(inatm, (1.0, 1.0, 1.0), Tview)
    Tcl = X.mix(inatm, (1.0, 1.0, 1.0), X.vmath('POWER', inv_e, tau_c))


    # ---------------------------------------------------------------- night lights
    if lights > 0:
        city = _city(X, q_g, water, elev, detail)
        night = 1.0 - X.smoothstep(-0.12, 0.02, mus_g)
        DBG.update(city=city)
        Lg = Lg + city * (night * lights_in)

    # ---------------------------------------------------------------- sun disc (camera rays only)
    lp = X.node('ShaderNodeLightPath')
    iscam = X.f(lp.outputs['Is Camera Ray'])
    cos_sun = DL
    ang = math.radians(0.2665)
    sun_rad = Esun / (math.pi * ang * ang)            # radiance of the disc so its irradiance = Esun
    cth = math.cos(ang)
    x = ((cos_sun - cth) / (1.0 - cth)).max(0.0)       # 0 at edge .. 1 at centre (approx)
    disc = X.smoothstep(0.0, 0.08, x)
    limbd = (x.sqrt() * 0.6 + 0.4)                     # limb darkening
    glow = ((cos_sun - 1.0) * 20000.0).exp() * 0.0015
    disc_k, E.disc_node = X.value(disc_strength if sun_disc else 0.0, "SunDiscK")
    Ls = X.vec(scol) * ((disc * limbd * sun_rad + glow * sun_rad * 0.02) * iscam * (1.0 - hitG) * disc_k)

    # ---------------------------------------------------------------- composite
    base = X.mix(hitG, Ls, Lg)
    surf = Tview * base * (1.0 - alpha_c) + Tcl * Lc * alpha_c
    Lsum = Lin + surf
    DBG.update(Lin=Lin, Tview=Tview, Lg=Lg, Lc=Lc, surface=surf)
    if debug:
        v = DBG[debug]
        Lsum = v if isinstance(v, type(Lsum)) else X.combine(v, v, v)
    bg = X.node('ShaderNodeBackground')
    X._in(bg.inputs['Color'], Lsum)
    bg.inputs['Strength'].default_value = 1.0
    out = X.node('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs[0], out.inputs[0])
    E.bg = bg
    print("[earth] world nodes:", len(nt.nodes))
    import os as _os
    if _os.environ.get("SB_EARTH_PROXY") == "1":
        _proxy(E, nt, out, clouds)

    # ---------------------------------------------------------------- methods
    def cam_km(loc_m):
        return V(loc_m) / 1000.0 - E.center_km

    def set_cam(loc_m):
        v = cam_km(loc_m)
        for i in range(3):
            E.cam_node.inputs[i].default_value = v[i]

    def track(cam, f0, f1):
        scn = bpy.context.scene
        for f in range(f0, f1 + 1):
            scn.frame_set(f)
            v = cam_km(cam.matrix_world.translation)
            for i in range(3):
                E.cam_node.inputs[i].default_value = v[i]
                E.cam_node.inputs[i].keyframe_insert("default_value", frame=f)
        scn.frame_set(f0)

    def bake_rot(f0, f1, fn):
        for f in range(f0, f1 + 1):
            r = fn(f)
            for i in range(3):
                E.rot_node.inputs[i].default_value = r[i]
                E.rot_node.inputs[i].keyframe_insert("default_value", frame=f)

    def set_sun(d):
        d = V(d).normalized()
        for i in range(3):
            E.sun_node.inputs[i].default_value = d[i]
        if getattr(E, "lamp", None):
            E.lamp.rotation_mode = 'QUATERNION'
            E.lamp.rotation_quaternion = (-d).to_track_quat('-Z', 'Y')

    def sun_lamp(strength=None, color=None, angle=0.53, name="Sun", at_m=(0, 0, 0)):
        """SUN lamp matching the shader's sun. Colour includes atmospheric transmittance toward the sun
        from scene point at_m (so a station at sunrise is lit gold, in Earth's shadow -> strength 0)."""
        d = V(tuple(E.sun_node.inputs[i].default_value for i in range(3))).normalized()
        if color is None:
            T = sun_transmittance(cam_km(at_m), d)
            m = max(T) if max(T) > 0 else 1.0
            color = tuple(E.sun_color[i] * T[i] / m for i in range(3))
            if strength is None:
                strength = E.sun_strength * m
        o = sb.light('SUN', name, loc=d * 50, energy=strength if strength is not None else E.sun_strength,
                     color=color or E.sun_color, angle=angle)
        o.rotation_mode = 'QUATERNION'
        o.rotation_quaternion = (-d).to_track_quat('-Z', 'Y')
        o.data.use_shadow = True
        E.lamp = o
        return o

    E.cam_km, E.set_cam, E.track, E.bake_rot, E.set_sun, E.sun_lamp = cam_km, set_cam, track, bake_rot, set_sun, sun_lamp
    set_cam((0, 0, 0))
    return E


# =====================================================================================================
def _land(X, q, seed, detail, DBG, Lt=None, mus=None):
    """Returns (albedo vec, water mask 0..1, elevation 0..1) from unit sphere coords q (z = spin axis).
    Scale layers: continents (1000s km) > biome mosaic (100s km) > mountains/hills with hillshade (5-50 km)
    > drainage valleys > field/vegetation mottling (~1 km)."""
    so = (seed * 13.1, seed * 7.7, seed * 3.3)
    q0 = q + so
    _, wc = X.noise(q0, scale=0.8, detail=3, rough=0.5)
    qw = q0 + (wc - 0.5) * 0.8
    _, wc2 = X.noise(qw, scale=4.0, detail=3, rough=0.5)
    qw = qw + (wc2 - 0.5) * 0.08
    cont, _ = X.noise(qw, scale=1.25, detail=14 * detail, rough=0.58, lac=2.03)
    z = q.z
    lat = z.abs()
    cont = cont + X.smoothstep(-0.86, -0.97, z) * 0.18          # Antarctica
    sea = 0.535
    landm = X.smoothstep(sea - 0.001, sea + 0.001, cont)
    water = 1.0 - landm
    elev = X.lin(sea, sea + 0.14, cont)
    DBG.update(cont=cont)

    # ---- terrain height field (km-ish units, only relative): mountain belts + hills
    belt, _ = X.noise(qw + (2.0, 5.0, 1.0), scale=3.0, detail=4, rough=0.5)
    beltm = X.smoothstep(0.52, 0.62, belt) * X.smoothstep(0.02, 0.25, elev) + X.smoothstep(0.35, 0.8, elev) * 0.5
    def terrain(p):
        rg, _ = X.noise(p, scale=38.0, detail=10 * detail, rough=0.52, ntype='RIDGED_MULTIFRACTAL', offset=1.0, gain=2.2, normalize=False)
        hl, _ = X.noise(p, scale=160.0, detail=8 * detail, rough=0.55)
        return rg * (beltm * 0.6 + 0.12) + hl * 0.15
    T0 = terrain(qw)
    hs = X.f(1.0)
    if Lt is not None:
        T1 = terrain(qw + Lt * 0.00018)
        hs = ((T0 - T1) * 45.0 / (mus.max(0.0) + 0.3) + 1.0).clamp(0.2, 1.7)
        DBG.update(hs=hs)
    mount = X.lin(0.9, 1.9, T0 * 1.6) * beltm

    # ---- climate / biomes
    moist, _ = X.noise(qw + (5.2, 1.3, 7.7), scale=1.8, detail=6, rough=0.6)
    patch, _ = X.noise(qw + (1.0, 2.0, 3.0), scale=22.0, detail=6 * detail, rough=0.6)
    clim = (X.smoothstep(0.30, 0.05, lat) * 0.16 - X.smoothstep(0.18, 0.36, lat) * X.smoothstep(0.62, 0.45, lat) * 0.16
            + X.smoothstep(0.5, 0.7, lat) * 0.04)
    m = moist + 0.03 + clim + elev * -0.05 + (patch - 0.5) * 0.35 + mount * 0.05
    DBG.update(moist=m)
    fine, _ = X.noise(q0, scale=900.0, detail=6 * detail, rough=0.6)
    med, _ = X.noise(q0, scale=90.0, detail=6 * detail, rough=0.6)
    tone = X.lin(0.36, 0.64, fine * 0.25 + med * 0.75)
    c_des = X.mix(tone, (0.30, 0.18, 0.085), (0.50, 0.34, 0.18))
    c_arid = X.mix(tone, (0.17, 0.115, 0.065), (0.30, 0.21, 0.12))
    c_sav = X.mix(tone, (0.075, 0.078, 0.030), (0.16, 0.135, 0.06))
    c_for = X.mix(tone, (0.016, 0.032, 0.012), (0.05, 0.068, 0.026))
    c_bor = X.mix(tone, (0.02, 0.03, 0.02), (0.05, 0.055, 0.038))
    c_tun = X.mix(tone, (0.075, 0.07, 0.058), (0.14, 0.125, 0.10))
    col = X.mix(X.smoothstep(0.47, 0.41, m), c_sav, c_arid)
    col = X.mix(X.smoothstep(0.42, 0.35, m) * X.smoothstep(0.62, 0.5, lat), col, c_des)
    col = X.mix(X.smoothstep(0.53, 0.61, m), col, c_for)
    col = X.mix(X.smoothstep(0.66, 0.74, lat) * X.smoothstep(0.44, 0.52, m), col, c_bor)
    col = X.mix(X.smoothstep(0.84, 0.90, lat), col, c_tun)
    # drainage valleys: darker, greener lines in the landscape
    vr, _ = X.noise(qw + (7.0, 3.0, 5.0), scale=55.0, detail=7 * detail, rough=0.55, ntype='RIDGED_MULTIFRACTAL', offset=1.0, gain=2.0, normalize=False)
    valley = X.smoothstep(1.55, 1.9, vr) * (1.0 - mount) * 0.7
    col = X.mix(valley, col, col * X.vec((0.55, 0.75, 0.55)))
    # bare rock + snow on mountains
    col = X.mix(mount * 0.65, col, X.mix(tone, (0.09, 0.08, 0.07), (0.17, 0.15, 0.13)))
    snow = X.smoothstep(0.55, 0.85, mount + lat * 0.5) * X.smoothstep(1.2, 1.7, T0 * 1.6)
    col = X.mix(snow, col, (0.62, 0.64, 0.68))
    # ice
    icen, _ = X.noise(q0, scale=7.0, detail=8 * detail, rough=0.6)
    ice_s = X.smoothstep(-0.90, -0.93, z + (icen - 0.5) * 0.05)
    ice_n = X.smoothstep(0.935, 0.955, z + (icen - 0.5) * 0.06)
    ice_l = X.smoothstep(0.80, 0.86, lat + (icen - 0.5) * 0.08) * landm
    ice = ice_s.max(ice_n).max(ice_l)
    # ocean: deep / shelf with sediment variation
    shelf = X.smoothstep(sea - 0.010, sea - 0.0008, cont)
    sed, _ = X.noise(q0, scale=60.0, detail=6, rough=0.6)
    shelf = shelf * X.lin(0.35, 0.65, sed)
    ocol = X.mix(shelf, (0.0055, 0.014, 0.030), (0.016, 0.050, 0.056))
    col = col * hs
    alb = X.mix(landm, ocol, col)
    alb = X.mix(ice, alb, X.mix(fine, (0.60, 0.65, 0.72), (0.78, 0.81, 0.85)) * X.mix(ice_l, 1.0, hs))
    water = water * (1.0 - ice)
    return alb, water, elev


VORTICES = [  # (lat deg, lon deg, radius (chord), spin rad) — mid-latitude cyclones + a tropical storm
    (48, 20, 0.17, 3.2), (-52, -40, 0.20, 3.6), (40, 150, 0.14, 2.6), (-45, 100, 0.16, 3.2),
    (58, -80, 0.18, 3.0), (-60, 170, 0.18, 3.0), (18, -60, 0.045, 5.0)]


def _vortex(X, q, lat, lon, rad, spin):
    la, lo = math.radians(lat), math.radians(lon)
    c = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
    d = q - c
    d2 = d.dot(d) * (d.dot(X.vec((c[1], -c[0], 0.3))) * (3.0 / max(rad, 0.05)) + 1.0).max(0.35)
    th = (d2 * (-1.0 / (rad * rad))).exp() * (-spin if lat > 0 else spin)
    ct, st = th.cos(), th.sin()
    cq = X.vec(c).cross(q)
    cd = X.vec(c).dot(q)
    return q * ct + cq * st + X.vec(c) * (cd * (1.0 - ct))


def _cloud_density(X, q, cov, seed, detail, lo=False, DBG=None, ret_q=False):
    """Cloud reflectance-like opacity (0..~0.9) from a two-stream optical-thickness model."""
    so = (seed * 5.3 + 11.0, seed * 2.1 + 3.0, seed * 9.7 + 7.0)
    qv = q
    for v in VORTICES:
        qv = _vortex(X, qv, *v)
    q0 = qv + so
    _, w1 = X.noise(q0, scale=1.7, detail=3, rough=0.5)
    qa = q0 + (w1 - 0.5) * 0.45
    _, w2 = X.noise(qa, scale=7.0, detail=3, rough=0.5)
    qb = qa + (w2 - 0.5) * 0.07
    big, _ = X.noise(qb, scale=2.0, detail=4, rough=0.5)
    z = q.z
    _, wl = X.noise(q0, scale=3.0, detail=2)
    alat = (z + (wl.x - 0.5) * 0.25).abs()
    band = (X.smoothstep(0.14, 0.0, alat) * 0.05 - X.smoothstep(0.12, 0.28, alat) * X.smoothstep(0.55, 0.38, alat) * 0.06
            + X.smoothstep(0.55, 0.8, alat) * 0.03)
    det, _ = X.noise(qb, scale=11.0, detail=(7 if lo else 13) * detail, rough=0.63)
    base = big * 0.55 + det * 0.45 + band + (cov - 0.5) * 0.25
    thr = 0.522
    k = X.lin(thr, thr + 0.07, base)
    tau = k * k * 90.0 + X.smoothstep(thr - 0.004, thr + 0.006, base) * 2.5
    if not lo:
        # cellular structure inside decks (closed-cell stratocumulus / convective cells)
        _, wc = X.noise(qb, scale=60.0, detail=2)
        cd, _, _ = X.voronoi(qb + (wc - 0.5) * 0.004, scale=420.0, feature='F1')
        cells = X.smoothstep(0.95, 0.15, cd)
        tex, _ = X.noise(qb, scale=140.0, detail=8 * detail, rough=0.6)
        tau = tau * (cells * 0.7 + 0.45) * (tex * 1.6 + 0.2)
        # thin veils / cirrus: stretched east-west
        ci, _ = X.noise(qa * (2.5, 2.5, 9.0), scale=5.0, detail=9 * detail, rough=0.62)
        if DBG is not None: DBG.update(c_ci=ci)
        tau = tau + X.smoothstep(0.57, 0.72, ci) * X.smoothstep(0.45, 0.52, base) * 1.8
    tp = tau * 0.15
    d = tp / (tp + 2.0)
    if DBG is not None:
        DBG.update(c_big=big, c_mid=det, c_base=base, c_tau=tau)
    return (d, qb, base) if ret_q else d


def _clouds(X, q_c, n, L, mus_g, rot_in, cov, seed, detail, hc, DBG):
    dens, qb, base = _cloud_density(X, q_c, cov, seed, detail, DBG=DBG, ret_q=True)
    # relief: cloud-top height field, compared one step toward the sun (in the rotated frame)
    Lr, _ = X.rotate_euler(L, rot_in)
    Lt = Lr - q_c * q_c.dot(Lr)
    def top(p):
        h1, _ = X.noise(p, scale=32.0, detail=4, rough=0.5)
        return h1 * 0.6 + X.lin(0.52, 0.62, base) * 0.8
    h0 = top(qb)
    h1 = top(qb + Lt * 0.00035)
    DBG.update(c_relief=(h0 - h1))
    # shadow lookup: from the ground point, walk toward the sun up to the cloud layer
    tang = L - n * mus_g
    off = tang * ((hc / R) / mus_g.max(0.08))
    qs, _ = X.rotate_euler((n + off).normalize(), rot_in)
    dens_sh = _cloud_density(X, qs, cov, seed, detail * 0.6, lo=True) * 0.85
    return dens, dens_sh, (h0 - h1)


def _city(X, q, water, elev, detail, desert=None):
    """Night lights: returns RGB radiance-like (pre lights_in). Cities cluster along coasts and in temperate,
    non-desert lowlands; bright cores with halos, towns, corridors between them, fine speckle."""
    land = 1.0 - water
    lat = q.z.abs()
    pop, _ = X.noise(q + (3.0, 9.0, 1.0), scale=2.2, detail=5, rough=0.6)
    coast = X.smoothstep(0.0, 0.03, elev) * X.smoothstep(0.25, 0.02, elev)          # low, near-coast land
    P = X.smoothstep(0.47, 0.62, pop) * (coast * 0.7 + 0.3) * X.smoothstep(0.78, 0.55, lat) * X.smoothstep(0.35, 0.05, elev) * land
    if desert is not None:
        P = P * (1.0 - desert * 0.85)
    # metro cores (~40-80 km cells)
    d1, c1, _ = X.voronoi(q, scale=140.0, feature='F1', rand=0.9)
    keep1 = X.smoothstep(0.55, 0.9, c1.x + P * 0.9)
    size1 = c1.y * 0.25 + 0.12
    core = (d1 * d1 * -1.0 / (size1 * size1)).exp() * keep1
    halo = (d1 * -1.0 / (size1 * 2.2)).exp() * keep1 * 0.25
    # towns (~5-10 km cells)
    d2, c2, _ = X.voronoi(q, scale=1400.0, feature='F1', rand=1.0)
    keep2 = X.smoothstep(0.75, 0.95, c2.x + P * 0.6)
    towns = (d2 * d2 * -40.0).exp() * keep2 * (c2.y * 0.6 + 0.2)
    # corridors (roads / rivers linking settlements)
    rd, _ = X.noise(q + (1.0, 4.0, 2.0), scale=60.0, detail=5, rough=0.5, ntype='RIDGED_MULTIFRACTAL', offset=1.0, gain=2.0, normalize=False)
    road = X.smoothstep(1.7, 1.95, rd) * 0.25
    sp, _ = X.noise(q, scale=9000.0, detail=2, rough=0.5)
    speck = X.smoothstep(0.35, 0.75, sp)
    L = (core * 1.0 + halo + towns + road * P) * (P + 0.08) * speck
    L = L * X.smoothstep(0.02, 0.2, P + core * 0.2)
    tint = X.mix(c1.z, (1.0, 0.55, 0.22), (1.0, 0.78, 0.55))                    # sodium vs whiter LED districts
    return tint * (L * 0.06)


# =====================================================================================================
def sun_transmittance(cam_km, sun_dir, steps=400):
    """Python-side transmittance (RGB) of sunlight reaching an Earth-centred point (km) — same model as the
    shader (without the grey MS proxy). Use to colour the SUN lamp when the sun sits near the limb."""
    P = V(cam_km); L = V(sun_dir).normalized()
    b = P.dot(L); c = P.dot(P) - R * R
    if b < 0 and b * b - c > 0:
        # sun ray hits the planet: find tangent height; soft cut
        pass
    ra = R + ATM
    disc = b * b - (P.dot(P) - ra * ra)
    if disc <= 0:
        return (1.0, 1.0, 1.0)
    t1 = -b + math.sqrt(disc)
    t0 = max(0.0, -b - math.sqrt(disc))
    tauR = tauM = 0.0
    hmin = 1e9
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = t0 + (i + 0.5) * dt
        h = (P + L * t).length - R
        hmin = min(hmin, h)
        tauR += math.exp(-max(h, -5) / HR) * dt
        tauM += math.exp(-max(h, -5) / HM) * dt
    BRe = [BR[i] + KO * BO[i] for i in range(3)]
    T = [math.exp(-(BRe[i] * tauR + BM_E * tauM)) for i in range(3)]
    lit = min(1.0, max(0.0, (hmin + 4.0) / 14.0))
    lit = lit * lit * (3 - 2 * lit)
    return tuple(x * lit for x in T)


def horizon_dir(cam_km_vec, azim_deg=0.0, above_deg=0.0):
    """World direction from an Earth-centred camera position to the geometric horizon at azimuth
    azim_deg (0 = +Y, 90 = +X, measured in the local tangent plane), raised by above_deg."""
    up = V(cam_km_vec).normalized()
    r = V(cam_km_vec).length
    dip = math.acos(min(1.0, R / r))
    ref = V((0, 1, 0)) if abs(up.y) < 0.9 else V((1, 0, 0))
    east = ref.cross(up).normalized()
    north = up.cross(east).normalized()
    a = math.radians(azim_deg)
    hdir = (north * math.cos(a) + east * math.sin(a) * -1.0).normalized()
    el = -dip + math.radians(above_deg)
    return (hdir * math.cos(el) + up * math.sin(el)).normalized()



def _proxy(E, nt, out, clouds=0.5):
    """LOOKDEV ONLY (SB_EARTH_PROXY=1): Cycles cannot compile the analytic world (SVM stack), so for framing checks
    on CPU we disconnect it and put a real-scale sphere + limb-glow shell in the scene instead. Same centre, radius,
    scale and sun, so horizon position/curvature, terminator and composition match the real render."""
    for l in list(out.inputs[0].links):
        nt.links.remove(l)
    bg2 = nt.nodes.new('ShaderNodeBackground')
    bg2.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1)
    nt.links.new(bg2.outputs[0], out.inputs[0])
    c = V(E.center_km) * 1000.0
    Rm = R * 1000.0
    bpy.ops.mesh.primitive_uv_sphere_add(segments=512, ring_count=256, radius=Rm, location=c)
    g = bpy.context.active_object
    g.name = "EarthProxy"
    for p in g.data.polygons:
        p.use_smooth = True
    m = sb.mat("EarthProxyMat", (0.02, 0.05, 0.12), rough=0.35, spec=0.6)
    nb = sb.NB(m)
    co = nb.coord('Object')
    land = nb.maprange(nb.noise(co, scale=2.2e-6 * 1.0, detail=8, rough=0.6).outputs['Fac'], 0.55, 0.57)
    lc = nb.mix(nb.maprange(nb.noise(co, scale=1e-5, detail=6).outputs['Fac'], 0.3, 0.7), (0.08, 0.12, 0.04, 1), (0.30, 0.24, 0.14, 1))
    base = nb.mix(land, (0.012, 0.035, 0.09, 1), lc)
    cl = nb.maprange(nb.noise(co, scale=6e-6, detail=10, rough=0.62).outputs['Fac'], 0.62 - 0.25 * clouds, 0.72 - 0.25 * clouds)
    base = nb.mix(cl, base, (0.85, 0.86, 0.88, 1))
    nb.set('Base Color', base)
    nb.set('Roughness', nb.mix(land, 0.25, 0.9, dtype='FLOAT'))
    g.data.materials.append(m)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=512, ring_count=256, radius=Rm + 60000.0, location=c)
    a = bpy.context.active_object
    a.name = "AtmoProxy"
    for p in a.data.polygons:
        p.use_smooth = True
    am = bpy.data.materials.new("AtmoProxyMat")
    am.use_nodes = True
    ant = am.node_tree
    ant.nodes.clear()
    lw = ant.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.2
    pw = ant.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 3.0
    ant.links.new(lw.outputs['Facing'], pw.inputs[0])
    em = ant.nodes.new('ShaderNodeEmission'); em.inputs[0].default_value = (0.35, 0.6, 1.0, 1)
    mul = ant.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 3.0
    ant.links.new(pw.outputs[0], mul.inputs[0]); ant.links.new(mul.outputs[0], em.inputs[1])
    tr = ant.nodes.new('ShaderNodeBsdfTransparent')
    ad = ant.nodes.new('ShaderNodeAddShader')
    ant.links.new(em.outputs[0], ad.inputs[0]); ant.links.new(tr.outputs[0], ad.inputs[1])
    o = ant.nodes.new('ShaderNodeOutputMaterial')
    ant.links.new(ad.outputs[0], o.inputs[0])
    a.data.materials.append(am)
    a.visible_shadow = False
    E.proxy = (g, a)
