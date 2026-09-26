"""STILL BEGINNING - launch FX: pre-dawn environment, pad haze, EEVEE volumetric smoke/vapor, exhaust plume rig,
fire lighting. Used by scenes/s19_ready.py and scenes/s20_launch.py (and reusable for s21).

All time functions take GLOBAL frame numbers. Ignition = IGN (1800). Clamp release = REL.
"""
import bpy, math, random
from mathutils import Vector as V, Matrix, Quaternion, noise as mnoise
import sb
import rocket
import launchpad as LP

IGN = 1800
REL = 1830
LOCK_F = 1789          # clamp lock bolts withdraw (s19c insert)
FPS = 24.0
BOUND = 1.7             # puff cube half-size relative to its core radius (billows reach ~1.6)
AMBIENT = 0.045         # multiple-scatter ambient emission (per unit density)
GLOW_AZ = 38.0          # afterglow azimuth (sb convention: 0 = +Y, 90 = +X): behind the pad as seen from s20 camera


# =========================================================================== environment

def environment(sc, haze=0.00032, glow=2.0, strength=1.0):
    """Pre-dawn blue hour: deep blue zenith, cool horizon, a restrained amber band behind the pad. Pad haze box
    (makes flood beams and later the fire glow readable)."""
    w, bg = sb.world_bluehour(glow_az=GLOW_AZ, glow=glow, zenith=(0.006, 0.016, 0.055), horizon=(0.07, 0.095, 0.16),
                              glow_col=(1.0, 0.36, 0.07), glow_width=55, band_height=2.2, strength=strength,
                              clouds=0.85)
    haze_box(haze)
    return w


def haze_box(density=0.0011, size=(1400.0, 1400.0, 260.0), name="PadHaze"):
    if density <= 0:
        return None
    o = sb.prim("cube", name, loc=(0, 0, size[2] / 2 - 1.0), scale=(size[0] / 2, size[1] / 2, size[2] / 2))
    m = bpy.data.materials.new(name + "Mat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    # denser near the ground (marine layer), gently varying
    zf = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('ADD', z, 1.0), size[2] / 2), -1.0 / 45.0))
    n = nb.noise(co, scale=3.0, detail=3, rough=0.5)
    dens = nb.math('MULTIPLY', nb.math('ADD', 0.35, zf), nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.7, 1.3))
    pv = nb.new('ShaderNodeVolumePrincipled')
    pv.inputs['Color'].default_value = (0.85, 0.88, 0.95, 1)
    pv.inputs['Anisotropy'].default_value = 0.55
    g = nb.new('ShaderNodeMath')
    g.operation = 'MULTIPLY'
    g.name = "Density"
    nb.link(dens, g.inputs[0])
    g.inputs[1].default_value = density
    nb.link(g.outputs[0], pv.inputs['Density'])
    out = nb.new('ShaderNodeOutputMaterial')
    nb.link(pv.outputs[0], out.inputs['Volume'])
    o.data.materials.append(m)
    o.visible_shadow = False
    return o


def volume_range(sc, cam, target, near=None, far=None):
    """Concentrate EEVEE froxel depth resolution where the action is."""
    d = (V(target) - V(cam.location)).length
    e = sc.eevee
    try:
        e.use_volume_custom_range = True
    except Exception:
        pass
    if near is None:
        near = 0.1 if d < 80 else d * 0.55
    if far is None:
        far = max(400.0, d * 1.9)
    e.volumetric_start = near
    e.volumetric_end = far


# =========================================================================== timing / physics

def ascent(f, rel=None):
    """Vehicle vertical displacement (m) at global frame f (float ok). Held until release, then net acceleration
    ramps in over 0.5 s to ~0.44 g and grows (propellant burn-off) ~ +0.04 g/s. Continuous velocity."""
    rel = REL if rel is None else rel
    tau = (f - rel) / FPS
    if tau <= 0:
        return 0.0
    dt = 1.0 / 240.0
    h = v = 0.0
    t = 0.0
    while t < tau - 1e-9:
        s = min(dt, tau - t)
        a = 9.81 * (0.44 + 0.04 * t) * sb.smooth(t / 0.55)
        v += a * s
        h += v * s
        t += s
    return h


def lean(f, rel=None):
    """Tower-clearance lean (radians about +X, top moves to -Y, away from the tower). Subtle."""
    rel = REL if rel is None else rel
    tau = (f - rel) / FPS
    return -math.radians(1.3) * sb.smoother((tau - 1.2) / 2.8)


def thrust(f):
    """0..1 engine output. Frame 1800 already clearly lit (ignition), full ~0.7 s later."""
    if f < IGN:
        return 0.0
    t = (f - IGN) / FPS
    return min(1.0, 0.38 + 0.62 * sb.ease_out(t / 0.72, 2.2))


def flicker(f, seed=0.0, amt=0.08, rate=1.0):
    """Smooth, temporally coherent light flicker (no per-frame randomness)."""
    t = f / FPS * rate
    n = mnoise.noise(V((t * 7.3, seed * 13.1, 0.3))) * 0.6 + mnoise.noise(V((t * 17.9, seed * 5.7, 2.1))) * 0.4
    return 1.0 + amt * n


# =========================================================================== volumetric puffs

def smoke_material(name="Smoke", albedo=(0.78, 0.76, 0.73), heat_col=(1.0, 0.42, 0.10), edge=0.10, billow=1.0,
                   noise_scale=1.25, aniso=0.25, fine=1.0):
    """Shared puff volume material. Per-object custom props (Attribute node, OBJECT): sb_seed, sb_dens, sb_heat,
    sb_age (drives 4D noise W so billows evolve slowly), sb_white (0 sooty-grey .. 1 steam-white),
    sb_dx/dy/dz (unit direction from the puff toward the fire, for fire-facing glow).
    Object coords: the puff fills the unit sphere of its cube bounds; the surface is a multi-octave cauliflower
    (three billow octaves) so silhouettes are crisp lobes-on-lobes, not spheres."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links

    def attr(n_):
        a = nb.new('ShaderNodeAttribute')
        a.attribute_type = 'OBJECT'
        a.attribute_name = n_
        return a.outputs['Fac']
    seed, dens, heat, age, white = attr("sb_seed"), attr("sb_dens"), attr("sb_heat"), attr("sb_age"), attr("sb_white")
    fdir = rocket._comb(nb, attr("sb_dx"), attr("sb_dy"), attr("sb_dz"))
    co = nb.vmath('SCALE', nb.coord('Object'), scale=BOUND)      # puff core radius 1 = 1/BOUND of the cube
    r = nb.vmath('LENGTH', co)
    off = nb.vmath('SCALE', (1.0, 1.7, 2.3), scale=nb.math('MULTIPLY', seed, 17.0))
    q = nb.vmath('ADD', co, off)

    def octave(sc_, w_rate, detail):
        n_ = nb.noise(q, scale=sc_, detail=detail, rough=0.5, dim='4D')
        nb.link(nb.math('MULTIPLY', age, w_rate), n_.inputs['W'])
        return n_.outputs['Fac']
    n1 = octave(noise_scale, 0.18, 3.0)
    n2 = octave(noise_scale * 2.9, 0.3, 2.0)
    n3 = octave(noise_scale * 8.0, 0.5, 2.0)
    # billows: rounded lobes (abs of centred noise, inverted) at three scales
    def lobes(n_, amp):
        return nb.math('MULTIPLY', nb.math('SUBTRACT', 0.5, nb.math('ABSOLUTE', nb.math('SUBTRACT', n_, 0.5))), amp)
    disp = nb.math('ADD', nb.math('MULTIPLY', nb.math('SUBTRACT', n1, 0.5), billow), lobes(n2, 0.34))
    disp = nb.math('ADD', disp, lobes(n3, 0.12 * fine))
    shape = nb.math('SUBTRACT', r, disp)
    d = rocket._ss(nb, nb.math('SUBTRACT', 0.92, shape), 0.0, edge)
    inner = nb.maprange(n2, 0.25, 0.75, 0.55, 1.2)
    dd = nb.math('MULTIPLY', nb.math('MULTIPLY', d, inner), dens)
    pv = nb.new('ShaderNodeVolumePrincipled')
    pv.inputs['Anisotropy'].default_value = aniso
    nb.link(dd, pv.inputs['Density'])
    col = nb.mix(white, (*albedo, 1), (0.93, 0.935, 0.95, 1))
    nb.link(col, pv.inputs['Color'])
    # fire-facing glow: the side of each puff that looks at the flame is lit from within; lumpy with the billows
    nco = nb.vmath('NORMALIZE', co)
    face = nb.maprange(nb.vmath('DOT_PRODUCT', nco, fdir), -0.35, 0.9, 0.0, 1.0)
    face = nb.math('ADD', 0.1, nb.math('MULTIPLY', nb.math('POWER', face, 2.0), 0.9))
    depth_w = rocket._ss(nb, nb.math('SUBTRACT', 0.92, shape), 0.0, 0.45)
    em = nb.math('MULTIPLY', nb.math('MULTIPLY', face, heat), nb.maprange(n2, 0.3, 0.7, 0.25, 1.6))
    em = nb.math('MULTIPLY', em, nb.math('ADD', 0.35, depth_w))
    # multiple-scattering stand-in: EEVEE volumes are single-scatter, so dense cloud goes unnaturally black.
    # a faint cool ambient (sky) + the warm heat term, added as emission weighted by density.
    amb = nb.math('MULTIPLY', rocket._ss(nb, nb.math('SUBTRACT', 0.92, shape), 0.0, 0.3), AMBIENT)
    ecol = nb.mix(nb.math('DIVIDE', em, nb.math('ADD', em, nb.math('ADD', amb, 1e-4))), (0.35, 0.45, 0.62, 1), (*heat_col, 1))
    nb.link(ecol, pv.inputs['Emission Color'])
    nb.link(nb.math('MULTIPLY', nb.math('ADD', em, amb), dd), pv.inputs['Emission Strength'])
    out = nb.new('ShaderNodeOutputMaterial')
    nb.link(pv.outputs[0], out.inputs['Volume'])
    return m


_CUBE = None


def _cube_mesh():
    global _CUBE
    if _CUBE is None or _CUBE.name not in bpy.data.meshes:
        o = sb.prim("cube", "PuffProto")
        _CUBE = o.data
        bpy.data.objects.remove(o)
    return _CUBE


class Puffs:
    """Deterministic puff 'particles'. add(...) with callables of age tau (seconds) -> baked per frame.
    pos(tau)->Vector, rad(tau)->float or (rx, ry, rz), dens(tau), heat(tau), white."""

    def __init__(self, name, mat, fire_fn=None, post=None):
        self.name, self.mat, self.items, self.fire_fn, self.post = name, mat, [], fire_fn, post

    def add(self, f_spawn, life, pos, rad, dens, heat=None, white=0.3, seed=None, fade_in=0.25, fade_out=1.0):
        self.items.append(dict(fs=f_spawn, life=life, pos=pos, rad=rad, dens=dens, heat=heat, white=white,
                               seed=random.random() if seed is None else seed, fi=fade_in, fo=fade_out))

    def build(self, f0, f1, coll=None):
        me = _cube_mesh().copy()
        me.name = self.name + "Cube"
        me.materials.append(self.mat)
        objs = []
        for i, it in enumerate(self.items):
            o = bpy.data.objects.new("%s%03d" % (self.name, i), me)
            (coll or bpy.context.scene.collection).objects.link(o)
            o.visible_shadow = True
            o["sb_seed"] = it["seed"]
            o["sb_white"] = it["white"]
            o["sb_dens"] = 0.0
            o["sb_heat"] = 0.0
            o["sb_age"] = 0.0
            o["sb_dx"], o["sb_dy"], o["sb_dz"] = 0.0, 0.0, -1.0
            fa = max(f0, int(math.floor(it["fs"])))
            fb = min(f1, int(math.ceil(it["fs"] + it["life"] * FPS)))
            if fb < f0 or fa > f1:
                o.hide_render = True
                objs.append(o)
                continue
            # hide outside life
            for f, h in ((f0, True), (fa - 1, True), (fa, False), (fb, False), (fb + 1, True)):
                if f0 <= f <= f1 + 1:
                    o.hide_render = h
                    o.keyframe_insert("hide_render", frame=f)
            for f in range(max(f0, fa - 1), min(f1, fb + 1) + 1):
                tau = max(0.0, (f - it["fs"]) / FPS)
                p = it["pos"](tau)
                r = it["rad"](tau)
                if isinstance(r, (int, float)):
                    r = (r, r, r)
                if self.post is not None:
                    p = self.post(f, p, max(r[0], r[1]))
                k = sb.smooth(tau / it["fi"]) * (1 - sb.smooth((tau - (it["life"] - it["fo"])) / it["fo"]))
                o.location = p
                o.scale = (r[0] * BOUND, r[1] * BOUND, r[2] * BOUND)
                o.keyframe_insert("location", frame=f)
                o.keyframe_insert("scale", frame=f)
                o["sb_dens"] = it["dens"](tau) * k
                o["sb_heat"] = (it["heat"](tau, p, f) if it["heat"] else 0.0) * k
                o["sb_age"] = tau + it["seed"] * 10.0
                if self.fire_fn is not None:
                    dv = self.fire_fn(f, p)
                    o["sb_dx"], o["sb_dy"], o["sb_dz"] = dv.x, dv.y, dv.z
                for pr in ("sb_dens", "sb_heat", "sb_age", "sb_dx", "sb_dy", "sb_dz"):
                    o.keyframe_insert('["%s"]' % pr, frame=f)
            objs.append(o)
        return objs


def drift(seed, tau, amp=3.0, rate=0.25):
    """Smooth low-frequency wander (m) so puff paths aren't straight lines."""
    t = tau * rate
    return V((mnoise.noise(V((t, seed * 7.1, 0.5))), mnoise.noise(V((seed * 3.3, t, 1.7))),
              0.5 * mnoise.noise(V((2.9, seed * 5.2, t))))) * amp


def decel(v0, k, tau):
    """Distance travelled with linear drag: v0 (1 - e^-k tau) / k."""
    return v0 * (1 - math.exp(-k * tau)) / k


# =========================================================================== exhaust plume rig

def _gain_socket(m):
    return m.node_tree.nodes["Gain"].inputs[1]


def plume_rig(RK, trench=True):
    """Emissive plume shells parented to the vehicle root (vehicle frame) + trench flame jets (world) + fire lights.
    Returns dict of objects/materials; animate with animate_plume()."""
    root = RK["root"]
    ex = rocket.EXIT_Z
    core_m = rocket.plume_material("PlumeCore", intensity=1.0, core_col=(1.0, 0.94, 0.80), mid_col=(1.0, 0.72, 0.38),
                                   tail_col=(1.0, 0.45, 0.14), length=11.0, diamonds=0.22)
    main_m = rocket.plume_material("PlumeMain", intensity=0.5, core_col=(1.0, 0.74, 0.38), mid_col=(1.0, 0.40, 0.07),
                                   tail_col=(0.85, 0.22, 0.04), length=60.0, diamonds=0.0)
    glow_m = rocket.plume_material("PlumeGlow", intensity=0.06, core_col=(1.0, 0.62, 0.28), mid_col=(1.0, 0.45, 0.12),
                                   tail_col=(0.8, 0.22, 0.05), length=60.0, diamonds=0.0)
    cores = []
    for i, (x, y, z) in enumerate(RK["nozzle_exits"]):
        o = rocket.plume_mesh("PlumeCore%d" % i, length=11.0, r0=0.60, r_max=1.35, neck=0.14, segs=48, mat=core_m, parent=root)
        o.location = (x, y, ex + 0.02)
        cores.append(o)
    main = rocket.plume_mesh("PlumeMain", length=60.0, r0=2.3, r_mid=5.8, r_max=9.0, knee=0.22, neck=0.0, segs=64, mat=main_m, parent=root)
    main.location = (0, 0, ex - 0.6)
    main.hide_render = True          # superseded by the volumetric flame body (kept for reference / s21 option)
    vol_m = rocket.flame_volume_material("FlameVol")
    vol = rocket.flame_volume("FlameVol", mat=vol_m, parent=root)
    vol.location = (0, 0, ex - 0.3)
    glow = rocket.plume_mesh("PlumeGlow", length=60.0, r0=3.6, r_mid=9.0, r_max=15.0, knee=0.25, neck=0.0, segs=48, mat=glow_m, parent=root)
    glow.location = (0, 0, ex - 0.3)
    jets = []
    jet_m = None
    if trench:
        jet_m = rocket.plume_material("TrenchJet", intensity=0.45, core_col=(1.0, 0.80, 0.48), mid_col=(1.0, 0.48, 0.13),
                                      tail_col=(0.8, 0.22, 0.05), length=26.0, diamonds=0.0)
        for s in (-1, 1):
            o = rocket.plume_mesh("TrenchJet%d" % (s > 0), length=26.0, r0=2.6, r_max=6.5, neck=0.0, segs=48, mat=jet_m)
            o.location = (s * 5.5, 0, 2.6)
            o.rotation_euler = (0, -s * math.pi / 2, 0)
            o.scale = (0.55, 1.0, 1.0)
            jets.append(o)
    # lights
    L = {}
    fire = (1.0, 0.58, 0.26)
    L["eng"] = sb.light('POINT', "FireEngine", loc=(0, 0, ex - 3.0), energy=0, color=fire, size=3.0)
    L["eng"].parent = root
    # flame-column lights live in world space (baked along the visible column, never below the deck)
    L["pl1"] = sb.light('POINT', "FirePlume1", loc=(0, 0, 20.0), energy=0, color=(1.0, 0.52, 0.2), size=6.0)
    L["pl2"] = sb.light('POINT', "FirePlume2", loc=(0, 0, 14.0), energy=0, color=(1.0, 0.48, 0.17), size=8.0)
    L["hole"] = sb.light('POINT', "FireHole", loc=(0, 0, LP.TABLE_Z - 3.5), energy=0, color=fire, size=2.5)
    for s in (-1, 1):
        L["tr%d" % (s > 0)] = sb.light('POINT', "FireTrench%d" % (s > 0), loc=(s * 16.0, 0, 3.5), energy=0,
                                       color=(1.0, 0.5, 0.18), size=6.0)
    for l in L.values():
        l.data.use_soft_falloff = True
        l.data.volume_factor = 0.35 if l.name.startswith("FireEngine") else 1.0
    return dict(cores=cores, main=main, glow=glow, jets=jets, core_m=core_m, main_m=main_m, glow_m=glow_m,
                jet_m=jet_m, lights=L, vol=vol, vol_m=vol_m)


def exit_height(f):
    """World z of the first-stage nozzle exit plane."""
    return LP.ROCKET_Z0 + rocket.EXIT_Z + ascent(f)


def animate_plume(P, f0, f1, E=None):
    """Bake plume visibility/length/brightness and fire lights over [f0, f1]."""
    E = E or dict(eng=4.0e5, pl1=3.0e5, pl2=2.0e5, hole=2.5e5, tr=6.0e5)
    objs = P["cores"] + [P["glow"], P["vol"]] + P["jets"]
    P["main"].hide_render = True
    for f in range(f0, f1 + 1):
        T = thrust(f)
        on = T > 0
        for o in objs:
            o.hide_render = not on
            o.keyframe_insert("hide_render", frame=f)
        he = exit_height(f)
        above = he - LP.TABLE_Z             # nozzle height above deck
        # main plume reaches down to the deck/deflector while low; later a free plume ~70 m
        Lp = min(78.0, max(12.0, above + 9.0))
        sc_ = Lp / 60.0
        P["main"].scale = (1.0, 1.0, sc_)
        P["glow"].scale = (1.0, 1.0, sc_)
        P["main"].keyframe_insert("scale", frame=f)
        P["glow"].keyframe_insert("scale", frame=f)
        P["vol"].scale = (1.0, 1.0, sc_)
        P["vol"].keyframe_insert("scale", frame=f)
        vg = P["vol_m"].node_tree.nodes["Gain"].inputs[1]
        vg.default_value = VOL_GAIN * T * (0.95 + 0.05 * flicker(f, 1.0, 0.06))
        vg.keyframe_insert("default_value", frame=f)
        fl = flicker(f, 1.0, 0.06)
        # emission gains (build with thrust; slight flicker)
        for m, base in ((P["core_m"], 20.0), (P["main_m"], 5.5), (P["glow_m"], 0.55)):
            s_ = _gain_socket(m)
            s_.default_value = base * T * (0.97 + 0.03 * fl)
            s_.keyframe_insert("default_value", frame=f)
        # advect turbulence downstream (object z units), evolve W
        for m, sp in ((P["core_m"], 55.0), (P["main_m"], 40.0), (P["glow_m"], 20.0), (P["jet_m"], 30.0), (P["vol_m"], 32.0)):
            if m is None:
                continue
            nd = m.node_tree.nodes["Adv"].inputs['Location']
            nd.default_value = (0.0, 0.0, 0.3 * sp * (f - IGN) / FPS)
            nd.keyframe_insert("default_value", frame=f)
            for nn, rate in (("Flow", 0.9), ("Flow2", 1.6)):
                w = m.node_tree.nodes[nn].inputs['W']
                w.default_value = (f - IGN) / FPS * rate
                w.keyframe_insert("default_value", frame=f)
        # trench jets: strong while the vehicle is low, fade as it climbs
        jt = T * (1 - sb.smooth((above - 25.0) / 45.0))
        if P["jet_m"] is not None:
            s_ = _gain_socket(P["jet_m"])
            s_.default_value = 9.0 * jt * fl
            s_.keyframe_insert("default_value", frame=f)
        L = P["lights"]
        L["pl1"].location = (0.0, -(he - LP.ROCKET_Z0) * 0.012, max(LP.TABLE_Z + 9.0, he - 14.0))
        L["pl2"].location = (0.0, 0.0, max(LP.TABLE_Z + 3.0, he - 38.0))
        L["pl1"].keyframe_insert("location", frame=f)
        L["pl2"].keyframe_insert("location", frame=f)
        vals = dict(eng=E["eng"] * T * fl, pl1=E["pl1"] * T * flicker(f, 2.0, 0.08), pl2=E["pl2"] * T * flicker(f, 3.0, 0.1),
                    hole=E["hole"] * T * (1 - sb.smooth((above - 8.0) / 30.0)) * flicker(f, 4.0, 0.1),
                    tr0=E["tr"] * jt * flicker(f, 5.0, 0.1), tr1=E["tr"] * jt * flicker(f, 6.0, 0.1))
        for k, v in vals.items():
            L[k].data.energy = v
            L[k].data.keyframe_insert("energy", frame=f)


# =========================================================================== vehicle / pad animation

def animate_vehicle(RK, f0, f1):
    root = RK["root"]
    for f in range(f0, f1 + 1):
        root.location = (0.0, 0.0, LP.ROCKET_Z0 + ascent(f))
        root.rotation_euler = (lean(f), 0.0, 0.0)
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)


def animate_pad(CX, f0, f1, rel=None, arm_start=None):
    """Clamp jaws swing open at release (hydraulic, ~0.3 s, eased), tail-service plates retract & drop into their
    hoods at release, upper umbilical arm swings back from T-0."""
    rel = REL if rel is None else rel
    arm_start = IGN if arm_start is None else arm_start
    for f in range(f0, f1 + 1):
        u = sb.smoother((f - (rel - 2)) / 7.0)
        for i, cl in enumerate(CX["clamps"]):
            cl["jaw"].rotation_euler = (0.0, math.radians(cl["open"]) * u * (1.0 + 0.02 * i), 0.0)
            cl["jaw"].keyframe_insert("rotation_euler", frame=f)
        v = sb.smoother((f - (rel - 1)) / 9.0)
        for t in CX["tsm"]:
            pl = t["plate"]
            if "rest" not in t:
                t["rest"] = pl.location.copy()
            pl.location = t["rest"] + t["dir"] * (1.6 * v) + V((0, 0, -0.9 * v * v))
            pl.rotation_euler = (0.0, math.radians(25.0) * v, pl.rotation_euler.z)
            pl.keyframe_insert("location", frame=f)
            pl.keyframe_insert("rotation_euler", frame=f)
        # clamp lock bolts withdraw ~0.4 s before T-0 (fast pneumatic stroke with a tiny rebound)
        tl = (f - LOCK_F) / 3.0
        ul = sb.smoother(tl) + 0.08 * math.sin(min(max(tl - 1.0, 0.0), 1.5) * math.pi * 2) * math.exp(-max(tl - 1.0, 0.0) * 1.5)
        for cl in CX["clamps"]:
            lk = cl["lock"]
            if "lock_rest" not in cl:
                cl["lock_rest"] = lk.location.copy()
            lk.location = cl["lock_rest"] + V((0, LP.LOCK_TRAVEL * ul, 0))
            lk.keyframe_insert("location", frame=f)
        arm = CX["tower"]["upper_arm"]
        a = sb.smoother((f - arm_start) / (2.6 * FPS))
        arm["pivot"].rotation_euler = (0.0, 0.0, arm["rest"] - math.radians(80.0) * a)
        arm["pivot"].keyframe_insert("rotation_euler", frame=f)


# =========================================================================== launch smoke

WIND = V((2.5, 0.8, 0.0))       # m/s, light onshore breeze


def _seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
    return (p - (a + ab * t)).length


def heat_field(f, p):
    """Fire-glow strength at world point p (emission fakes light scattered inside the cloud from the flame):
    trench jets while the vehicle is low, the impingement zone under the plume, and the plume column itself."""
    T = thrust(f)
    if T <= 0:
        return 0.0
    he = exit_height(f)
    above = he - LP.TABLE_Z
    jt = T * (1 - sb.smooth((above - 25.0) / 45.0))
    h = 0.0
    for s_ in (-1, 1):
        d = _seg_dist(p, V((s_ * 6.0, 0, 2.5)), V((s_ * 34.0, 0, 3.5)))
        h += jt * math.exp(-d / 22.0)
    d = (p - V((0, 0, LP.TABLE_Z - 2.0))).length
    h += T * (1 - sb.smooth((above - 10.0) / 50.0)) * (1.2 * math.exp(-d / 20.0) + 0.22 * math.exp(-d / 45.0))
    d = _seg_dist(p, V((0, 0, he)), V((0, 0, max(0.0, he - 55.0))))
    h += T * (0.8 * math.exp(-d / 13.0) + 0.15 * math.exp(-d / 35.0))
    return h


HEAT = 6.0
VOL_GAIN = 3.4


def clear_axis(f, p, r):
    """The exhaust column blows cloud away: puffs below the nozzles are kept out of a cylinder around the plume
    (smoothly), so the flame stays visible through liftoff."""
    T = thrust(f)
    if T <= 0:
        return p
    he = exit_height(f)
    if p.z > he + 4.0:
        return p
    xy = V((p.x, p.y, 0.0))
    d = xy.length
    need = (1.45 * r + 3.0) * T
    if d >= need:
        return p
    if d < 1e-3:
        xy = V((1.0, 0.0, 0.0))
        d = 1e-3
    k = sb.smooth(1.0 - d / need)
    return p + xy.normalized() * (need - d) * (0.6 + 0.4 * k)


def fire_dir(f, p):
    """Unit vector from p toward the nearest flame element (trench jets / impingement / plume column)."""
    he = exit_height(f)
    best, bd = None, 1e9
    segs = [(V((-6.0, 0, 2.5)), V((-30.0, 0, 3.0))), (V((6.0, 0, 2.5)), V((30.0, 0, 3.0))),
            (V((0, 0, he)), V((0, 0, max(LP.TABLE_Z - 3.0, he - 50.0))))]
    for a, b in segs:
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / max(1e-6, ab.length_squared)))
        q = a + ab * t
        d = (p - q).length
        if d < bd:
            bd, best = d, q
    v = best - p
    return v.normalized() if v.length > 1e-4 else V((0, 0, -1))


def launch_smoke(f0, f1, density=1.0, seed=11):
    """Ground cloud from the flame-trench exits, deck spill after liftoff, deluge steam and the exhaust trail.
    Returns (objects, materials)."""
    rs = random.Random(seed)
    m_cloud = smoke_material("LaunchCloud", albedo=(0.86, 0.84, 0.81), edge=0.08, billow=0.9, noise_scale=1.3)
    m_trail = smoke_material("LaunchTrail", albedo=(0.84, 0.83, 0.81), edge=0.3, billow=0.8, noise_scale=1.0, aniso=0.35, fine=0.6)
    P = Puffs("Cloud", m_cloud, fire_dir, clear_axis)
    Tr = Puffs("Trail", m_trail, fire_dir)

    def spawn_frames(a, b, i0, i1):
        out, f = [], float(a)
        while f < b:
            out.append(f)
            f += sb.lerp(i0, i1, (f - a) / max(1.0, b - a))
        return out

    # --- A: trench blast, both sides, from T-0 through the shot (rate decays as the vehicle climbs)
    for side in (-1, 1):
        for fs in spawn_frames(IGN + 1, f1 - 6, 2.0, 7.0):
            sd = rs.random()
            v0 = rs.uniform(26.0, 42.0) * (0.55 + 0.45 * thrust(fs))
            yv = rs.uniform(-4.5, 4.5)
            y0 = rs.uniform(-2.5, 2.5)
            k = rs.uniform(0.45, 0.7)
            r0 = rs.uniform(2.4, 3.6)
            grow = rs.uniform(4.5, 6.5)
            climb = rs.uniform(0.2, 0.9)
            hot = rs.uniform(4.0, 9.0)

            def pos(t, side=side, v0=v0, yv=yv, y0=y0, k=k, sd=sd, climb=climb, r0=r0, grow=grow):
                x = side * (8.0 + decel(v0, k, t))
                y = y0 + decel(yv, k * 0.8, t)
                r = r0 + grow * math.sqrt(t)
                z = max(r * 0.5, 2.0 + climb * t + 0.12 * t * t)
                return V((x, y, z)) + WIND * t + drift(sd, t, 2.0 + 1.2 * t)

            def rad(t, r0=r0, grow=grow):
                r = r0 + grow * math.sqrt(t)
                return (r * 1.15, r, r * 0.78)

            P.add(fs, (f1 - fs) / FPS + 1.0, pos, rad, lambda t: 0.7 * density,
                  heat=lambda t, p, f, hot=hot: HEAT * (heat_field(f, p) + hot * 0.1 * math.exp(-t / 0.4)), white=rs.uniform(0.2, 0.7), seed=sd)
    # --- B: upward spill around the table edges (deluge steam + blast), whiter
    for fs in spawn_frames(IGN + 3, IGN + 70, 3.0, 6.0):
        sd = rs.random()
        a = rs.choice((0.0, math.pi)) + rs.gauss(0.0, 0.45)
        p0 = V((math.cos(a) * 11.0, math.sin(a) * 9.0, LP.TABLE_Z - 1.0))
        vz = rs.uniform(2.5, 5.0)
        vr = rs.uniform(4.0, 10.0)

        def pos(t, p0=p0, a=a, vz=vz, vr=vr, sd=sd):
            return p0 + V((math.cos(a), math.sin(a), 0)) * decel(vr, 0.6, t) + V((0, 0, decel(vz, 0.35, t))) + WIND * t + drift(sd, t, 1.5 + t)

        P.add(fs, (f1 - fs) / FPS + 1.0, pos, lambda t: 3.0 + 4.0 * math.sqrt(t), lambda t: 0.5 * density,
              heat=lambda t, p, f: HEAT * heat_field(f, p), white=rs.uniform(0.6, 0.95), seed=sd)
    # --- C: after release the plume hits the deck directly: radial spill hugging the deck, pushed outward
    for fs in spawn_frames(REL + 2, REL + 95, 2.5, 6.0):
        sd = rs.random()
        a = rs.choice((0.0, math.pi)) + rs.gauss(0.0, 0.6)
        vr = rs.uniform(18.0, 30.0)
        k = rs.uniform(0.5, 0.8)

        def pos(t, a=a, vr=vr, k=k, sd=sd):
            rr = 3.5 + 4.5 * math.sqrt(t)
            return V((math.cos(a), math.sin(a) * 0.8, 0)) * (7.0 + decel(vr, k, t)) + V((0, 0, LP.TABLE_Z + rr * 0.35 + 0.5 * t)) + WIND * t + drift(sd, t, 1.0 + t)

        P.add(fs, (f1 - fs) / FPS + 1.0, pos, lambda t: (3.5 + 4.5 * math.sqrt(t), 3.5 + 4.5 * math.sqrt(t), 0.7 * (3.5 + 4.5 * math.sqrt(t))),
              lambda t: 0.6 * density, heat=lambda t, p, f: HEAT * heat_field(f, p), white=rs.uniform(0.3, 0.8), seed=sd)
    # --- D: exhaust trail behind the rising vehicle (thin, wispy, drifting)
    for fs in spawn_frames(REL + 20, f1 - 2, 2.0, 2.5):
        sd = rs.random()
        he = exit_height(fs)
        if he < LP.TABLE_Z + 12:
            continue
        z0 = he - rs.uniform(26.0, 40.0)
        if z0 < LP.TABLE_Z + 6:
            continue
        p0 = V((rs.uniform(-1.5, 1.5), rs.uniform(-1.5, 1.5) - (he - LP.ROCKET_Z0) * math.sin(-lean(fs)) * 0.3, z0))

        def pos(t, p0=p0, sd=sd):
            return p0 + WIND * t * 1.5 + drift(sd, t, 1.0 + 1.5 * t)

        Tr.add(fs, (f1 - fs) / FPS + 1.0, pos, lambda t: (5.0 + 3.5 * math.sqrt(t), 5.0 + 3.5 * math.sqrt(t), 9.0 + 3.0 * math.sqrt(t)),
               lambda t: 0.05 * density, heat=lambda t, p, f: HEAT * 0.6 * heat_field(f, p), white=0.5, seed=sd, fade_in=0.4)
    objs = P.build(f0, f1) + Tr.build(f0, f1)
    return objs, (m_cloud, m_trail)


# =========================================================================== pre-launch vapor ("the vehicle breathing")

def vapor_material(name="Vapor", density_edge=0.45):
    """Wispy cold condensation: bright, soft-edged, finer billows, forward scattering (glows toward the floods)."""
    return smoke_material(name, albedo=(0.95, 0.96, 0.98), edge=density_edge, billow=0.75, noise_scale=1.5,
                          aniso=0.45, fine=1.2)


def _world(RK, local):
    """Vehicle-frame point -> world at rest (root at ROCKET_Z0, no rotation)."""
    return V(local) + V((0, 0, LP.ROCKET_Z0))


def prelaunch_vapor(RK, f0, f1, stop=None, seed=21, amount=1.0, vents=True, skin=True, chill=True):
    """Restrained venting before T-0 (and blown away/fading after ignition):
      vents  - LOX / fuel vent plumes streaming leeward from the vent doors
      skin   - condensation sheets sliding down the frosted tank walls
      chill  - LOX chill-down vapor spilling from the engine bells, sinking through the flame hole
    `stop`: frame after which no new vapor spawns (default IGN)."""
    rs = random.Random(seed)
    stop = IGN if stop is None else stop
    m = vapor_material("Vapor")
    P = Puffs("Vapor", m)

    def frames(a, b, step):
        f = float(a)
        while f < b:
            yield f
            f += step * rs.uniform(0.8, 1.25)
    life_cap = lambda fs: max(0.5, (f1 - fs) / FPS + 0.5)
    if vents:
        for nm, (loc, d) in RK["vents"].items():
            p0 = _world(RK, loc) + V(d) * 0.3
            strength = 1.0 if nm.startswith("lox") else 0.55
            for fs in frames(f0 - 60, min(stop, f1), 2.0 / strength):
                sd = rs.random()
                v0 = rs.uniform(2.5, 4.5)
                sink = rs.uniform(0.3, 0.9)

                def pos(t, p0=p0, d=V(d), v0=v0, sd=sd, sink=sink):
                    return p0 + d * decel(v0, 1.2, t) + WIND * t * 0.9 + V((0, 0, -sink * t)) + drift(sd, t, 0.4 + 0.5 * t, 0.6)
                P.add(fs, min(3.2, life_cap(fs)), pos, lambda t: 0.25 + 0.9 * math.sqrt(t),
                      lambda t, s=strength: 0.35 * amount * s * math.exp(-t / 1.6), white=1.0, seed=sd, fade_in=0.15, fade_out=1.2)
    if skin:
        # leeward side of the LOX tank (wind is +X): condensation sheets drifting down the wall
        for fs in frames(f0 - 72, min(stop + 8, f1), 3.0):
            sd = rs.random()
            az = math.radians(rs.uniform(-40.0, 40.0))
            zz = rs.uniform(rocket.ST["lox"][0] + 2.0, rocket.ST["lox"][1] - 1.5) if rs.random() < 0.75 else \
                rs.uniform(rocket.ST["fuel"][0] + 1.0, rocket.ST["fuel"][1] - 1.0)
            p0 = _world(RK, (math.cos(az) * (rocket.R + 0.5), math.sin(az) * (rocket.R + 0.5), zz))

            def pos(t, p0=p0, sd=sd, az=az):
                out_ = V((math.cos(az), math.sin(az), 0))
                return p0 + out_ * (0.25 * t) + V((0, 0, -1.3 * t)) + WIND * t * 0.35 + drift(sd, t, 0.3, 0.8)
            P.add(fs, min(3.0, life_cap(fs)), pos, lambda t: (0.5 + 0.35 * t, 0.5 + 0.35 * t, 1.4 + 0.9 * t),
                  lambda t: 0.10 * amount, white=1.0, seed=sd, fade_in=0.5, fade_out=1.2)
    if chill:
        for (x, y, z) in RK["nozzle_exits"]:
            p0 = _world(RK, (x, y, z + 0.4))
            for fs in frames(f0 - 60, min(stop, f1), 3.5):
                sd = rs.random()

                def pos(t, p0=p0, sd=sd):
                    fall = decel(3.5, 0.7, t)
                    return p0 + V((0, 0, -fall)) + WIND * t * 0.25 + drift(sd, t, 0.25 + 0.4 * t, 0.9)
                P.add(fs, min(3.0, life_cap(fs)), pos, lambda t: 0.35 + 0.8 * math.sqrt(t),
                      lambda t: 0.28 * amount * math.exp(-t / 1.8), white=1.0, seed=sd, fade_in=0.2, fade_out=1.0)
    return P.build(f0, f1), m
