"""STILL BEGINNING - launch complex for the ROCKET & LAUNCH sequence (s19a/b/c, s20; reusable for s21 ground views).

WORLD LAYOUT (metres, ground z=0; math polar angles measured from +X, counter-clockwise):
  vehicle axis at (0, 0). Launch-table deck top at TABLE_Z; the vehicle's station 0 (aft-skirt bottom) at ROCKET_Z0.
  Flame deflector under the table splits the exhaust to +/-X along a walled flame trench (|y| < TRENCH_W/2).
  Fixed umbilical tower centred at TOWER_C (front face at y = TOWER_C.y - TOWER_S/2), swing arms reach -Y to the
  vehicle; they retract by rotating about their vertical pivot (animate the returned pivot empties' rotation z).
  Hold-down clamps at vehicle azimuths 0/90/180/270; each jaw pivots about a tangential axis (animate 'jaw' rot y).

Structure is built from real member profiles (H columns, I girders, angle bracing, gussets) accumulated into a few
meshes by `Beams`; every member carries a random per-face 'tone' attribute that drives paint/weathering variation.

build_complex(...) builds everything and returns a dict of animatable handles; see function doc.
"""
import bpy, bmesh, math, random
from mathutils import Vector as V, Matrix, Quaternion
import sb
import rocket

TABLE_Z = 9.0
PED_H = 3.2
ROCKET_Z0 = TABLE_Z + PED_H          # 12.2
HOLE_R = 2.6
TABLE_X, TABLE_Y = 11.0, 10.0        # half sizes
TRENCH_W = 13.0
TRENCH_L = 70.0                      # half length (walls run x in [-L, L])
TOWER_C = V((-4.0, 18.5, 0.0))
TOWER_S = 9.0
TOWER_H = 70.0
LEVEL_H = 6.0
ARM_Z = ROCKET_Z0 + 49.2             # upper-stage umbilical arm height (centre of the S2 plate)
CREW_Z = 64.0
FLOOD_R = 95.0


# =========================================================================== member mesh builder

class Beams:
    """Accumulates structural members into one flat-shaded mesh. Each member gets a random 'tone' (0..1) stored
    as a FACE float attribute (paint batch / weathering variation). Profiles: box, I/H (ibeam), angle (L),
    channel, n-gon tube, plates."""

    def __init__(self, seed=0):
        self.v, self.f, self.t = [], [], []
        self.rs = random.Random(seed)

    @staticmethod
    def frame(a, b, up=None):
        d = (b - a)
        L = d.length
        d = d / L if L > 1e-9 else V((0, 0, 1))
        if up is None:
            up = V((0, 0, 1)) if abs(d.z) < 0.95 else V((1, 0, 0))
        x = d.cross(V(up))
        if x.length < 1e-6:
            x = d.cross(V((1, 0, 0)) if abs(d.x) < 0.9 else V((0, 1, 0)))
        x.normalize()
        y = x.cross(d).normalized()
        return d, x, y, L

    def add(self, a, b, w, h=None, up=None, off=(0.0, 0.0), tone=None):
        """Box section w (along frame x) by h (along frame y, ~up), optionally offset in the section plane."""
        a, b = V(a), V(b)
        h = w if h is None else h
        d, x, y, L = self.frame(a, b, up)
        if L < 1e-6:
            return
        o = x * off[0] + y * off[1]
        base = len(self.v)
        for p in (a, b):
            for cx, cy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                self.v.append(p + o + x * (cx * w / 2) + y * (cy * h / 2))
        for i in range(4):
            j = (i + 1) % 4
            self.f.append((base + i, base + j, base + 4 + j, base + 4 + i))
        self.f.append((base + 3, base + 2, base + 1, base + 0))
        self.f.append((base + 4, base + 5, base + 6, base + 7))
        tn = self.rs.random() if tone is None else tone
        self.t += [tn] * 6

    def ibeam(self, a, b, w, h, tf=None, tw=None, up=None, tone=None):
        """I/H section: flange width w, depth h (depth along frame y)."""
        tf = tf or max(0.012, h * 0.07)
        tw = tw or max(0.008, w * 0.05)
        tn = self.rs.random() if tone is None else tone
        self.add(a, b, w, tf, up, (0, (h - tf) / 2), tn)
        self.add(a, b, w, tf, up, (0, -(h - tf) / 2), tn)
        self.add(a, b, tw, h - 2 * tf, up, (0, 0), tn)

    def angle(self, a, b, w, t=None, up=None, tone=None):
        t = t or max(0.008, w * 0.1)
        tn = self.rs.random() if tone is None else tone
        self.add(a, b, w, t, up, (0, -(w - t) / 2), tn)
        self.add(a, b, t, w, up, (-(w - t) / 2, 0), tn)

    def channel(self, a, b, w, h, t=None, up=None, tone=None):
        t = t or max(0.008, h * 0.08)
        tn = self.rs.random() if tone is None else tone
        self.add(a, b, t, h, up, (-(w - t) / 2, 0), tn)
        self.add(a, b, w, t, up, (0, (h - t) / 2), tn)
        self.add(a, b, w, t, up, (0, -(h - t) / 2), tn)

    def tube(self, a, b, r, n=8, up=None, tone=None):
        a, b = V(a), V(b)
        d, x, y, L = self.frame(a, b, up)
        if L < 1e-6:
            return
        base = len(self.v)
        for p in (a, b):
            for k in range(n):
                ang = 2 * math.pi * (k + 0.5) / n
                self.v.append(p + x * (math.cos(ang) * r) + y * (math.sin(ang) * r))
        for k in range(n):
            j = (k + 1) % n
            self.f.append((base + k, base + j, base + n + j, base + n + k))
        self.f.append(tuple(base + k for k in reversed(range(n))))
        self.f.append(tuple(base + n + k for k in range(n)))
        tn = self.rs.random() if tone is None else tone
        self.t += [tn] * (n + 2)

    def box(self, c, s, tone=None):
        c = V(c)
        self.add(c - V((0, 0, s[2] / 2)), c + V((0, 0, s[2] / 2)), s[0], s[1], up=V((0, 1, 0)), tone=tone)

    def build(self, name, mat, parent=None, smooth=False):
        o = sb.mesh_obj(name, [tuple(p) for p in self.v], self.f, mat, smooth_shade=smooth)
        me = o.data
        at = me.attributes.new("tone", 'FLOAT', 'FACE')
        at.data.foreach_set("value", self.t[:len(me.polygons)] + [0.5] * max(0, len(me.polygons) - len(self.t)))
        if parent:
            o.parent = parent
        return o


# =========================================================================== materials

def _tone(nb):
    a = nb.new('ShaderNodeAttribute')
    a.attribute_type = 'GEOMETRY'
    a.attribute_name = "tone"
    return a.outputs['Fac']


def tower_steel(name="TowerSteel", color=(0.20, 0.205, 0.21), rust=0.6, grime=0.5):
    """Painted structural steel: per-member paint tone (batch/fade), rust bleed streaks running down from joints,
    darker grime toward the ground, chalky faded patches, fine pitting."""
    m = sb.mat(name, color, rough=0.62, metal=0.0, spec=0.4)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    tone = _tone(nb)
    n = nb.noise(co, scale=0.45, detail=6, rough=0.6)
    c = nb.mix(tone, (color[0] * 0.72, color[1] * 0.72, color[2] * 0.74, 1), (color[0] * 1.28, color[1] * 1.26, color[2] * 1.22, 1))
    chalk = nb.maprange(n.outputs['Fac'], 0.55, 0.75, 0.0, 0.35)
    c = nb.mix(chalk, c, (color[0] * 1.6, color[1] * 1.6, color[2] * 1.55, 1))
    # rust streaks: long vertical streaks, gated per member by tone and by a coarse noise
    stk = nb.noise(nb.mapping(co, scale=(4.0, 4.0, 0.10)), scale=3.0, detail=8, rough=0.62)
    gate = nb.maprange(nb.noise(co, scale=0.15, detail=2).outputs['Fac'], 0.45, 0.65, 0.2, 1.0)
    rs_ = nb.math('MULTIPLY', nb.maprange(stk.outputs['Fac'], 0.56, 0.72, 0.0, rust), gate)
    rs_ = nb.math('MULTIPLY', rs_, nb.maprange(tone, 0.0, 1.0, 0.4, 1.2))
    c = nb.mix(rs_, c, (0.20, 0.085, 0.035, 1))
    pit = nb.noise(co, scale=60.0, detail=3)
    c = nb.mix(nb.maprange(pit.outputs['Fac'], 0.62, 0.7, 0.0, 0.5), c, (0.13, 0.07, 0.04, 1))
    low = nb.math('SUBTRACT', 1.0, rocket._ss(nb, z, 0.0, 14.0))
    c = nb.mix(nb.math('MULTIPLY', low, grime), c, (0.06, 0.055, 0.05, 1))
    nb.set('Base Color', c)
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.5, 0.78))
    nb.set('Normal', nb.bump(pit.outputs['Fac'], strength=0.15, distance=0.002))
    return m


def galv_mat(name="Galv", color=(0.46, 0.47, 0.48)):
    """Hot-dip galvanized steel: spangle, per-member tone, dull white corrosion patches."""
    m = sb.mat(name, color, metal=0.85, rough=0.42)
    nb = sb.NB(m)
    co = nb.coord('Object')
    tone = _tone(nb)
    sp = nb.voronoi(co, scale=18.0, feature='F1')
    c = nb.mix(tone, (color[0] * 0.7, color[1] * 0.7, color[2] * 0.72, 1), (color[0] * 1.15, color[1] * 1.15, color[2] * 1.15, 1))
    wr = nb.noise(co, scale=1.5, detail=6)
    c = nb.mix(nb.maprange(wr.outputs['Fac'], 0.6, 0.72, 0.0, 0.5), c, (0.55, 0.55, 0.53, 1))
    nb.set('Base Color', c)
    nb.set('Roughness', nb.maprange(sp.outputs['Distance'], 0.0, 0.6, 0.3, 0.6))
    return m


def grating_mat(name="Grating"):
    m = sb.mat(name, (0.22, 0.22, 0.21), metal=0.8, rough=0.5)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    gx = rocket._seam(nb, x, 0.04, 0.008)
    gy = rocket._seam(nb, y, 0.12, 0.006)
    g = nb.math('MAXIMUM', gx, gy)
    nb.set('Base Color', nb.mix(g, (0.015, 0.015, 0.015, 1), (0.30, 0.30, 0.29, 1)))
    nb.set('Normal', nb.bump(g, strength=0.6, distance=0.01))
    return m


def cladding_mat(name="Cladding", color=(0.36, 0.37, 0.38)):
    """Corrugated metal siding (vertical ribs), streaked."""
    m = sb.mat(name, color, metal=0.55, rough=0.45)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    rib = nb.math('ABSOLUTE', nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', x, y), 2 * math.pi / 0.2)))
    nb.set('Normal', nb.bump(rib, strength=0.5, distance=0.02))
    n = nb.noise(nb.mapping(co, scale=(1.5, 1.5, 0.15)), scale=2.0, detail=6)
    c = nb.mix(nb.maprange(n.outputs['Fac'], 0.45, 0.72, 0, 0.6), (*color, 1), (0.16, 0.13, 0.11, 1))
    nb.set('Base Color', c)
    return m


def formed_concrete(name="Formed", color=(0.42, 0.41, 0.39), soot=0.0):
    """Cast-in-place concrete: form-panel seams (1.2 m lifts, 2.4 m panels), tie holes, run-off stains, soot."""
    m = sb.concrete(name, color, scale=0.6, rough=0.88, stains=0.45)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    bc = nb.bsdf.inputs['Base Color'].links[0].from_socket
    lift = rocket._seam(nb, z, 1.2, 0.012)
    pan = rocket._seam(nb, nb.math('ADD', x, y), 2.4, 0.01)
    seam = nb.math('MAXIMUM', lift, pan)
    tie = nb.voronoi(nb.mapping(co, scale=(1.0, 1.0, 1.0)), scale=1.6, feature='F1', rand=0.0)
    ties = nb.math('SUBTRACT', 1.0, rocket._ss(nb, tie.outputs['Distance'], 0.015, 0.03))
    run = nb.noise(nb.mapping(co, scale=(3.0, 3.0, 0.2)), scale=2.0, detail=6, rough=0.6)
    c = nb.mix(nb.maprange(run.outputs['Fac'], 0.55, 0.72, 0.0, 0.45), bc, (color[0] * 0.5, color[1] * 0.48, color[2] * 0.45, 1))
    c = nb.mix(nb.math('MULTIPLY', nb.math('MAXIMUM', seam, ties), 0.55), c, (0.08, 0.08, 0.08, 1))
    if soot > 0:
        sn = nb.noise(co, scale=0.4, detail=6, rough=0.6)
        c = nb.mix(nb.maprange(sn.outputs['Fac'], 0.35, 0.7, 0.2, soot), c, (0.05, 0.045, 0.04, 1))
    nb.set('Base Color', c)
    old = nb.bsdf.inputs['Normal'].links[0].from_socket
    nb.set('Normal', nb.bump(nb.math('MULTIPLY', seam, -1.0), strength=0.3, distance=0.01, normal=old))
    return m


def deck_plate_mat(name="DeckPlate"):
    """Heavy steel deck plate around the flame hole: weld seams, heat scale (blue/brown), soot, scuffs."""
    m = sb.mat(name, (0.14, 0.135, 0.13), metal=0.7, rough=0.55)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    seam = nb.math('MAXIMUM', rocket._seam(nb, x, 2.4, 0.02), rocket._seam(nb, y, 3.0, 0.02))
    r = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', x, x), nb.math('MULTIPLY', y, y)))
    heat = nb.math('SUBTRACT', 1.0, rocket._ss(nb, r, 2.8, 8.0))
    n = nb.noise(co, scale=0.8, detail=6, rough=0.6)
    c = nb.ramp(nb.math('MULTIPLY', heat, nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.6, 1.2)),
                [(0.0, (0.15, 0.145, 0.14)), (0.4, (0.16, 0.11, 0.08)), (0.7, (0.10, 0.10, 0.13)), (1.0, (0.03, 0.028, 0.026))])
    c = nb.mix(nb.math('MULTIPLY', seam, 0.6), c, (0.05, 0.05, 0.05, 1))
    scuff = nb.noise(nb.mapping(co, scale=(1.0, 6.0, 1.0)), scale=4.0, detail=3)
    c = nb.mix(nb.maprange(scuff.outputs['Fac'], 0.66, 0.72, 0.0, 0.4), c, (0.35, 0.34, 0.33, 1))
    nb.set('Base Color', c)
    nb.set('Normal', nb.bump(seam, strength=0.4, distance=0.01))
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, 0.4, 0.75))
    return m


def pad_concrete(name="PadConc", color=(0.36, 0.35, 0.335), scale=0.08):
    """Large-area concrete: expansion joints (6 m grid), soot/scorch toward the trench, stains."""
    m = sb.concrete(name, color, scale=scale, rough=0.85, stains=0.4)
    nb = sb.NB(m)
    co = nb.coord('Object')
    x, y, z = rocket._xyz(nb, co)
    j = nb.math('MAXIMUM', rocket._seam(nb, x, 6.0, 0.03), rocket._seam(nb, y, 6.0, 0.03))
    bc = nb.bsdf.inputs['Base Color'].links[0].from_socket
    r = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', x, x), nb.math('MULTIPLY', y, y)))
    sootn = nb.noise(co, scale=0.05, detail=6, rough=0.6)
    soot = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, rocket._ss(nb, r, 10.0, 60.0)),
                   nb.maprange(sootn.outputs['Fac'], 0.35, 0.7, 0.1, 0.8))
    trench = nb.math('SUBTRACT', 1.0, rocket._ss(nb, nb.math('ABSOLUTE', y), TRENCH_W / 2, TRENCH_W / 2 + 18.0))
    soot = nb.math('MAXIMUM', soot, nb.math('MULTIPLY', trench, nb.maprange(sootn.outputs['Fac'], 0.3, 0.7, 0.2, 0.7)))
    c = nb.mix(soot, bc, (0.07, 0.065, 0.06, 1))
    c = nb.mix(nb.math('MULTIPLY', j, 0.6), c, (0.05, 0.05, 0.05, 1))
    nb.set('Base Color', c)
    return m


def add_distance_fog(m, color=(0.055, 0.07, 0.11), dist=1800.0, strength=1.0):
    """Aerial perspective for surfaces beyond the pad haze box: emission toward a horizon haze colour with view
    distance (EEVEE volume range stops near the pad)."""
    nb = sb.NB(m)
    cd = nb.new('ShaderNodeCameraData')
    f = nb.math('SUBTRACT', 1.0, nb.math('EXPONENT', nb.math('DIVIDE', cd.outputs['View Distance'], -dist)))
    nb.set('Emission Color', (*color, 1))
    nb.set('Emission Strength', nb.math('MULTIPLY', f, strength))
    return m


def ground_mat(name="Scrub"):
    """Low coastal scrub / grass seen at a distance: dark olive with clumps and sandy patches."""
    m = sb.mat(name, (0.05, 0.055, 0.035), rough=0.9, spec=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n1 = nb.noise(co, scale=0.02, detail=8, rough=0.65)
    n2 = nb.noise(co, scale=0.4, detail=6, rough=0.7)
    c = nb.ramp(n1.outputs['Fac'], [(0.3, (0.035, 0.04, 0.025)), (0.55, (0.07, 0.07, 0.045)), (0.7, (0.16, 0.14, 0.10))])
    c = nb.mix(nb.maprange(n2.outputs['Fac'], 0.3, 0.7, 0.0, 0.5), c, (0.02, 0.025, 0.015, 1))
    nb.set('Base Color', c)
    nb.set('Normal', nb.bump(nb.noise(co, scale=3.0, detail=6).outputs['Fac'], strength=0.6, distance=0.08))
    return m


def grass_mat(name="Grass"):
    m = sb.mat(name, (0.06, 0.07, 0.035), rough=0.7, sss=0.0, spec=0.35, sheen=0.3)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=0.8, detail=4)
    c = nb.ramp(n.outputs['Fac'], [(0.3, (0.05, 0.06, 0.03)), (0.55, (0.10, 0.10, 0.05)), (0.75, (0.20, 0.17, 0.10))])
    nb.set('Base Color', c)
    return m


def asphalt_mat(name="Asphalt"):
    m = sb.mat(name, (0.05, 0.05, 0.052), rough=0.8)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=0.3, detail=6)
    nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (0.035, 0.035, 0.037, 1), (0.08, 0.08, 0.08, 1)))
    return m


def lamp_mat(name, color=(1.0, 0.92, 0.80), strength=40.0):
    """Lamp lens: emissive with a hot centre (camera-facing), dim at grazing angles."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, nt, nt.nodes, nt.links
    lw = nb.new('ShaderNodeLayerWeight')
    lw.inputs[0].default_value = 0.3
    f = nb.math('POWER', nb.math('SUBTRACT', 1.0, lw.outputs['Facing']), 3.0)
    s = nb.math('ADD', 0.08, nb.math('MULTIPLY', f, 1.0))
    g = nb.new('ShaderNodeMath')
    g.operation = 'MULTIPLY'
    g.name = "Gain"
    nb.link(s, g.inputs[0])
    g.inputs[1].default_value = strength
    em = nb.new('ShaderNodeEmission')
    em.inputs[0].default_value = (*color, 1)
    nb.link(g.outputs[0], em.inputs[1])
    out = nb.new('ShaderNodeOutputMaterial')
    nb.link(em.outputs[0], out.inputs['Surface'])
    return m


def window_mat(name="Window", color=(1.0, 0.78, 0.52), strength=3.0):
    """Lit window: emission gated by the per-face 'tone' attribute (tone>0.45 lit), dark glass otherwise."""
    m = sb.mat(name, (0.01, 0.012, 0.014), rough=0.08, spec=0.7)
    nb = sb.NB(m)
    on = rocket._ss(nb, _tone(nb), 0.45, 0.5)
    nb.set('Emission Color', (*color, 1))
    nb.set('Emission Strength', nb.math('MULTIPLY', on, strength))
    return m


def glass_dark(name="DarkGlass"):
    return sb.mat(name, (0.01, 0.012, 0.014), rough=0.06, spec=0.8, coat=0.5)


def materials():
    M = {}
    M["deck"] = deck_plate_mat()
    M["concrete"] = formed_concrete("PadStruct", (0.40, 0.39, 0.37), soot=0.5)
    M["concrete_clean"] = formed_concrete("Formed", (0.44, 0.43, 0.41))
    M["pad_conc"] = pad_concrete()
    M["trench"] = pad_concrete("TrenchConc", (0.22, 0.21, 0.20), scale=0.2)
    M["scorch"] = sb.brushed_metal("Scorch", (0.16, 0.14, 0.12), rough=0.6, aniso=0.0, scale=5, bump=0.1)
    M["steel_dark"] = tower_steel("SteelDark", (0.12, 0.122, 0.125), rust=0.7, grime=0.4)
    M["steel_bright"] = sb.brushed_metal("SteelBright", (0.62, 0.62, 0.63), rough=0.25, scale=80)
    M["clamp"] = sb.painted("ClampPaint", (0.20, 0.205, 0.21), rough=0.45, coat=0.1, grime=0.45, scale=4.0)
    M["clamp_jaw"] = sb.painted("ClampJaw", (0.24, 0.24, 0.245), rough=0.42, coat=0.1, grime=0.4, scale=5.0)
    M["tower"] = tower_steel()
    M["galv"] = galv_mat()
    M["grating"] = grating_mat()
    M["clad"] = cladding_mat()
    M["clad_light"] = cladding_mat("CladLight", (0.55, 0.56, 0.56))
    M["yellow"] = sb.painted("SafetyYellow", (0.55, 0.40, 0.05), rough=0.55, grime=0.45)
    M["pipe_grey"] = sb.painted("PipeGrey", (0.36, 0.37, 0.38), rough=0.4, grime=0.35)
    M["insul"] = sb.painted("Insul", (0.66, 0.66, 0.64), rough=0.6, grime=0.45, scale=2.0)
    M["hose"] = sb.rubber("Hose", (0.03, 0.03, 0.032), rough=0.5)
    M["tsm"] = sb.painted("TSMPaint", (0.45, 0.455, 0.46), rough=0.45, grime=0.5)
    M["room"] = sb.painted("WhiteRoom", (0.55, 0.55, 0.54), rough=0.45, grime=0.4)
    M["cabinet"] = sb.painted("Cabinet", (0.40, 0.42, 0.40), rough=0.4, grime=0.35)
    M["lamp_body"] = sb.mat("LampBody", (0.05, 0.05, 0.05), rough=0.5)
    M["lamp_flood"] = lamp_mat("LampFlood", (1.0, 0.93, 0.82), 60.0)
    M["lamp_sodium"] = lamp_mat("LampSodium", (1.0, 0.58, 0.22), 9.0)
    M["window"] = window_mat()
    M["beacon"] = sb.emit_mat("Beacon", (1.0, 0.06, 0.02), 30.0)
    M["marker"] = sb.emit_mat("Marker", (1.0, 0.45, 0.05), 8.0)
    M["wire"] = sb.mat("Wire", (0.1, 0.1, 0.1), metal=0.8, rough=0.4)
    M["tank_white"] = sb.painted("TankWhite", (0.58, 0.58, 0.57), rough=0.45, grime=0.45, scale=0.3)
    M["frost"] = sb.mat("Frost", (0.85, 0.88, 0.92), rough=0.85, spec=0.3)
    M["truck"] = sb.painted("TruckPaint", (0.52, 0.53, 0.54), rough=0.35, coat=0.5, grime=0.25)
    M["truck_cab"] = sb.painted("TruckCab", (0.62, 0.62, 0.6), rough=0.3, coat=0.6, grime=0.2)
    M["tyre"] = sb.rubber("Tyre")
    M["glass"] = glass_dark()
    M["chrome"] = sb.mat("Chrome", (0.8, 0.8, 0.8), metal=1.0, rough=0.12)
    M["ground"] = add_distance_fog(ground_mat())
    M["grass"] = grass_mat()
    M["asphalt"] = asphalt_mat()
    M["gravel"] = sb.concrete("Gravel", (0.30, 0.29, 0.27), scale=3.0, rough=0.95, stains=0.2)
    M["treeline"] = add_distance_fog(sb.mat("TreeSil", (0.01, 0.012, 0.01), rough=0.95), dist=5200.0)
    M["fence"] = galv_mat("FenceMetal")
    M["chainlink"] = chainlink_mat()
    return M


def chainlink_mat(name="Chainlink"):
    """Diamond wire mesh via alpha (dithered)."""
    m = sb.mat(name, (0.32, 0.32, 0.33), metal=0.9, rough=0.45)
    try:
        m.surface_render_method = 'DITHERED'
    except Exception:
        pass
    nb = sb.NB(m)
    x, y, z = rocket._xyz(nb, nb.coord('Object'))
    p = 0.06
    d1 = rocket._seam(nb, nb.math('ADD', x, y), p, 0.006)
    d2 = rocket._seam(nb, nb.math('SUBTRACT', x, y), p, 0.006)
    nb.set('Alpha', nb.math('MAXIMUM', d1, d2))
    return m


# =========================================================================== reusable detail kits

def stair_flight(B, a, b, width=1.0, tread=0.26, up=V((0, 0, 1))):
    """Steel stair flight from a (bottom) to b (top): channel stringers, grating treads, handrails, posts."""
    a, b = V(a), V(b)
    run = V((b.x - a.x, b.y - a.y, 0))
    side = run.normalized().cross(V((0, 0, 1))).normalized() * (width / 2)
    n = max(2, int((b.z - a.z) / 0.18))
    for s in (-1, 1):
        B.channel(a + side * s, b + side * s, 0.06, 0.25, up=V((0, 0, 1)))
        B.add(a + side * s + V((0, 0, 0.95)), b + side * s + V((0, 0, 0.95)), 0.045, 0.045)
        for k in range(0, n + 1, 4):
            p = a.lerp(b, k / n) + side * s
            B.add(p, p + V((0, 0, 0.95)), 0.04, 0.04, up=V((0, 1, 0)))
    for k in range(1, n):
        p = a.lerp(b, k / n)
        B.add(p - side * 0.98, p + side * 0.98, tread, 0.035, up=V((0, 0, 1)))


def ladder_cage(B, base, top, face_dir):
    """Fixed ladder with safety cage (rails, rungs, hoops, straps)."""
    base, top = V(base), V(top)
    f = V(face_dir).normalized()
    s = f.cross(V((0, 0, 1))).normalized()
    for k in (-1, 1):
        B.add(base + s * 0.22 * k, top + s * 0.22 * k, 0.05, 0.02)
    h = top.z - base.z
    for i in range(int(h / 0.3)):
        p = base + V((0, 0, 0.3 * i + 0.15))
        B.add(p - s * 0.22, p + s * 0.22, 0.025, 0.025)
    for i in range(int(max(0.0, h - 2.2) / 1.1)):
        z = base.z + 2.2 + i * 1.1
        prev = None
        for k in range(9):
            ang = math.pi * k / 8
            q = V((base.x, base.y, z)) + f * (0.38 + 0.36 * math.sin(ang)) + s * (0.38 * math.cos(ang))
            if prev is not None:
                B.add(prev, q, 0.05, 0.012)
            prev = q
    for k in (-1, 0, 1):
        ang = math.pi / 2 + k * 1.1
        off = f * (0.38 + 0.36 * math.sin(ang)) + s * (0.38 * math.cos(ang))
        B.add(base + V((0, 0, 2.2)) + off, top + off, 0.04, 0.008)


def cable_tray(B, a, b, w=0.6, rung=0.3):
    a, b = V(a), V(b)
    d, x, y, L = Beams.frame(a, b, V((0, 1, 0)) if abs((b - a).normalized().z) > 0.9 else V((0, 0, 1)))
    for s in (-1, 1):
        B.channel(a + x * (w / 2 * s), b + x * (w / 2 * s), 0.03, 0.12)
    n = int(L / rung)
    for i in range(1, n):
        p = a.lerp(b, i / n)
        B.add(p - x * w / 2, p + x * w / 2, 0.03, 0.02)


def cabinet(B, c, s):
    """Electrical enclosure: box + door seam + top hood + conduit stubs."""
    c = V(c)
    B.box(c, s)
    B.box(c + V((0, 0, s[2] / 2 + 0.03)), (s[0] + 0.08, s[1] + 0.08, 0.05))
    for k in range(2):
        p = c + V(((k - 0.5) * s[0] * 0.5, 0, -s[2] / 2))
        B.tube(p, p - V((0, 0, 0.4)), 0.03, 6)


def pipe_run(name, pts, r, mat, parent=None, flanges=True, supports=None, fl_mat=None):
    """Round pipe (bezier-smoothed polyline via sb.tube_along w/ corner fillets) + flanges every ~6 m."""
    o = sb.tube_along(name, pts, radius=r, mat=mat, bevel_res=4)
    if parent:
        o.parent = parent
    if flanges:
        F = Beams(hash(name) & 0xffff)
        for a, b in zip(pts, pts[1:]):
            a, b = V(a), V(b)
            L = (b - a).length
            n = int(L / 6.0)
            for k in range(1, n + 1):
                p = a.lerp(b, k / (n + 1))
                u = (b - a).normalized()
                F.tube(p - u * 0.04, p + u * 0.04, r * 1.45, 12)
        if F.v:
            F.build(name + "Fl", fl_mat or mat, parent)
    return o


# =========================================================================== pad structures

def launch_mount(M, parent=None):
    """Steel launch table on four formed-concrete legs, deep plate girders with stiffeners, flame hole with
    heat-scaled coaming, flame deflector, trench walls, deluge headers with nozzles, access stair, cable trays."""
    objs = []
    # deck plate with hole
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(2 * TABLE_X, 2 * TABLE_Y, 0.5), verts=bm.verts)
    me = bpy.data.meshes.new("TableDeck")
    bm.to_mesh(me)
    bm.free()
    deck = bpy.data.objects.new("TableDeck", me)
    sb.link_obj(deck)
    deck.location = (0, 0, TABLE_Z - 0.25)
    cut = sb.prim("cyl", "cutHole", loc=(0, 0, TABLE_Z), vertices=96, radius=HOLE_R, depth=3.0)
    md = deck.modifiers.new("B", 'BOOLEAN')
    md.object = cut
    md.solver = 'EXACT'
    sb.apply_mods(deck)
    bpy.data.objects.remove(cut)
    deck.data.materials.append(M["deck"])
    deck.data.transform(Matrix.Translation((0, 0, TABLE_Z - 0.25)))
    deck.location = (0, 0, 0)
    objs.append(deck)
    rocket.lathe("HoleLiner", [(HOLE_R, TABLE_Z - 2.4), (HOLE_R, TABLE_Z + 0.01)], segs=96, mat=M["scorch"], parent=parent)
    rocket.lathe("HoleCoaming", [(HOLE_R, TABLE_Z), (HOLE_R, TABLE_Z + 0.18), (HOLE_R + 0.35, TABLE_Z + 0.18),
                                 (HOLE_R + 0.45, TABLE_Z)], segs=96, mat=M["scorch"], parent=parent)
    G = Beams(11)
    zg = TABLE_Z - 0.5
    # deep plate girders (I sections, 2.2 m deep) around the perimeter and flanking the hole
    for s in (-1, 1):
        G.ibeam((-TABLE_X, s * (TABLE_Y - 0.5), zg - 1.1), (TABLE_X, s * (TABLE_Y - 0.5), zg - 1.1), 0.9, 2.2, 0.08, 0.05)
        G.ibeam((s * (TABLE_X - 0.5), -TABLE_Y, zg - 1.1), (s * (TABLE_X - 0.5), TABLE_Y, zg - 1.1), 0.9, 2.2, 0.08, 0.05)
        G.ibeam((s * (HOLE_R + 1.0), -TABLE_Y, zg - 1.0), (s * (HOLE_R + 1.0), TABLE_Y, zg - 1.0), 0.7, 2.0, 0.07, 0.045)
        G.ibeam((-TABLE_X, s * (HOLE_R + 1.0), zg - 1.0), (TABLE_X, s * (HOLE_R + 1.0), zg - 1.0), 0.7, 2.0, 0.07, 0.045)
        # secondary floor beams
        for k in range(1, 4):
            y = s * (HOLE_R + 1.0 + k * (TABLE_Y - HOLE_R - 1.5) / 4)
            G.ibeam((-TABLE_X, y, zg - 0.4), (TABLE_X, y, zg - 0.4), 0.3, 0.7)
    # web stiffeners + bolted splice plates on the outer girder faces
    for s in (-1, 1):
        for k in range(15):
            x = -TABLE_X + 0.7 + k * (2 * TABLE_X - 1.4) / 14
            G.add((x, s * (TABLE_Y - 0.5 + 0.06 * s), zg - 2.1), (x, s * (TABLE_Y - 0.5 + 0.06 * s), zg - 0.1), 0.35, 0.025, up=V((1, 0, 0)))
            y = -TABLE_Y + 0.7 + k * (2 * TABLE_Y - 1.4) / 14
            G.add((s * (TABLE_X - 0.5 + 0.06 * s), y, zg - 2.1), (s * (TABLE_X - 0.5 + 0.06 * s), y, zg - 0.1), 0.35, 0.025, up=V((0, 1, 0)))
        for x in (-TABLE_X * 0.33, TABLE_X * 0.33):
            G.box((x, s * (TABLE_Y - 0.5 + 0.1 * s), zg - 1.1), (1.1, 0.04, 1.9))
    objs.append(G.build("TableGirders", M["steel_dark"], parent))
    # bolt heads on splice plates
    Bt = Beams(12)
    for s in (-1, 1):
        for x in (-TABLE_X * 0.33, TABLE_X * 0.33):
            for i in range(4):
                for j in range(8):
                    p = V((x - 0.4 + i * 0.27, s * (TABLE_Y - 0.5 + 0.13 * s), zg - 1.95 + j * 0.24))
                    Bt.tube(p, p + V((0, 0.05 * s, 0)), 0.03, 6)
    objs.append(Bt.build("TableBolts", M["steel_dark"], parent))
    # legs: formed concrete with chamfered plinths and embedded steel base plates
    L = Beams(13)
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = V((sx * (TABLE_X - 2.0), sy * (TABLE_Y - 1.6), 0))
            L.box(c + V((0, 0, (zg - 2.2) / 2)), (3.2, 2.8, zg - 2.2))
            L.box(c + V((0, 0, 0.45)), (4.0, 3.6, 0.9))
            L.box(c + V((0, 0, zg - 2.2 - 0.3)), (3.6, 3.2, 0.6))
    objs.append(L.build("TableLegs", M["concrete"], parent))
    # flame deflector: steel-clad double curved wedge (ridge along Y) with plate seams
    prof = []
    n = 24
    for i in range(n + 1):
        t = i / n
        prof.append((11.0 * t, 5.2 * (1 - t) ** 1.7 + 0.2))
    verts, faces = [], []
    W = TRENCH_W / 2 - 0.2
    side = [(-x, zz) for x, zz in reversed(prof)] + prof[1:]
    for yy in (-W, W):
        for x, zz in side:
            verts.append((x, yy, zz))
    m_ = len(side)
    for i in range(m_ - 1):
        faces.append((i, i + 1, m_ + i + 1, m_ + i))
    defl = sb.mesh_obj("Deflector", verts, faces, M["scorch"], True)
    sb.solidify(defl, 0.6, -1)
    objs.append(defl)
    fl = sb.prim("plane", "TrenchFloor", loc=(0, 0, 0.05), scale=(TRENCH_L, TRENCH_W / 2, 1), mat=M["trench"])
    objs.append(fl)
    W2 = Beams(14)
    for s in (-1, 1):
        W2.add((-TRENCH_L, s * (TRENCH_W / 2 + 0.6), 1.6), (TRENCH_L, s * (TRENCH_W / 2 + 0.6), 1.6), 1.2, 3.2)
        W2.add((-TRENCH_L, s * (TRENCH_W / 2 + 0.6), 3.25), (TRENCH_L, s * (TRENCH_W / 2 + 0.6), 3.25), 1.4, 0.12)
        for k in range(-11, 12):
            W2.box((k * 6.0, s * (TRENCH_W / 2 + 1.35), 1.4), (0.5, 0.3, 2.8))     # buttresses
    objs.append(W2.build("TrenchWalls", M["concrete"], parent))
    # deluge: round headers along both long edges with nozzles; big feed riser with valve
    for s in (-1, 1):
        y = s * (TABLE_Y + 0.45)
        pipe_run("DelugeHdr%d" % (s > 0), [(-TABLE_X + 0.4, y, TABLE_Z + 0.35), (TABLE_X - 0.4, y, TABLE_Z + 0.35)], 0.22,
                 M["pipe_grey"], parent)
        N = Beams(15 + s)
        for k in range(12):
            x = -TABLE_X + 1.2 + k * (2 * TABLE_X - 2.4) / 11
            N.tube((x, y, TABLE_Z + 0.35), (x, y - s * 0.45, TABLE_Z + 0.75), 0.07, 8)
            N.tube((x, y - s * 0.45, TABLE_Z + 0.75), (x, y - s * 0.52, TABLE_Z + 0.8), 0.11, 8)
            N.add((x, y, TABLE_Z - 0.1), (x, y, TABLE_Z + 0.15), 0.12, 0.3, up=V((1, 0, 0)))   # support
        N.build("DelugeNoz%d" % (s > 0), M["pipe_grey"], parent)
    pipe_run("DelugeFeed", [(TABLE_X + 6.0, -TABLE_Y - 0.45, 0.4), (TABLE_X + 1.5, -TABLE_Y - 0.45, 0.4),
                            (TABLE_X + 1.5, -TABLE_Y - 0.45, TABLE_Z + 0.35), (TABLE_X - 0.4, -TABLE_Y - 0.45, TABLE_Z + 0.35)],
             0.35, M["pipe_grey"], parent)
    Vv = Beams(17)
    Vv.box((TABLE_X + 1.5, -TABLE_Y - 0.45, 3.0), (0.9, 0.9, 1.0))
    Vv.tube((TABLE_X + 1.5, -TABLE_Y - 0.45, 3.5), (TABLE_X + 1.5, -TABLE_Y - 0.45, 4.6), 0.06, 8)
    Vv.box((TABLE_X + 1.5, -TABLE_Y - 0.45, 4.7), (0.5, 0.5, 0.35))
    Vv.build("DelugeValve", M["yellow"], parent)
    # railings + kick plates on the deck perimeter
    R_ = Beams(18)
    for s in (-1, 1):
        for zz in (0.55, 1.1):
            R_.add((-TABLE_X, s * TABLE_Y, TABLE_Z + zz), (TABLE_X, s * TABLE_Y, TABLE_Z + zz), 0.045)
            R_.add((s * TABLE_X, -TABLE_Y, TABLE_Z + zz), (s * TABLE_X, TABLE_Y, TABLE_Z + zz), 0.045)
        for k in range(23):
            x = -TABLE_X + k * 2 * TABLE_X / 22
            R_.add((x, s * TABLE_Y, TABLE_Z), (x, s * TABLE_Y, TABLE_Z + 1.1), 0.045)
        for k in range(21):
            y = -TABLE_Y + k * 2 * TABLE_Y / 20
            R_.add((s * TABLE_X, y, TABLE_Z), (s * TABLE_X, y, TABLE_Z + 1.1), 0.045)
    objs.append(R_.build("TableRail", M["yellow"], parent))
    K = Beams(19)
    for s in (-1, 1):
        K.add((-TABLE_X, s * TABLE_Y, TABLE_Z + 0.08), (TABLE_X, s * TABLE_Y, TABLE_Z + 0.08), 0.015, 0.15)
        K.add((s * TABLE_X, -TABLE_Y, TABLE_Z + 0.08), (s * TABLE_X, TABLE_Y, TABLE_Z + 0.08), 0.015, 0.15)
    objs.append(K.build("TableKick", M["steel_dark"], parent))
    # access stair (switchback) at the -X,-Y corner + landing
    S = Beams(20)
    x0, y0 = -TABLE_X - 1.6, -TABLE_Y + 3.0
    stair_flight(S, (x0, y0 + 6.0, 0.0), (x0, y0, TABLE_Z / 2), 1.1)
    stair_flight(S, (x0 - 1.4, y0, TABLE_Z / 2), (x0 - 1.4, y0 + 6.0, TABLE_Z), 1.1)
    S.box((x0 - 0.7, y0 - 0.6, TABLE_Z / 2), (2.9, 1.3, 0.06))
    for dx in (0.0, -1.4):
        for dy in (-1.2, 0.0):
            S.ibeam((x0 + dx + 0.55 * (1 if dx == 0 else -1), y0 + dy, 0), (x0 + dx + 0.55 * (1 if dx == 0 else -1), y0 + dy, TABLE_Z / 2),
                    0.2, 0.2, up=V((0, 1, 0)))
    objs.append(S.build("TableStair", M["galv"], parent))
    # deck equipment: cable trays, cabinets, conduit
    E = Beams(21)
    for s in (-1, 1):
        cable_tray(E, (-TABLE_X + 1.0, s * (TABLE_Y - 1.4), TABLE_Z + 0.15), (TABLE_X - 1.0, s * (TABLE_Y - 1.4), TABLE_Z + 0.15), 0.5)
    for c in ((-8.0, 7.6), (-6.8, 7.6), (7.0, -7.8), (8.6, 6.0)):
        cabinet(E, (c[0], c[1], TABLE_Z + 0.9), (0.9, 0.5, 1.6))
    objs.append(E.build("DeckEquip", M["cabinet"], parent))
    for o in objs:
        if parent and o.parent is None:
            o.parent = parent
    return objs


def clamp(name, az, M, parent=None):
    """Pad-side hold-down clamp under vehicle lug at azimuth az (deg). Returns dict(jaw=<empty>, open=<deg>)
    Animate jaw.rotation_euler.y from 0 (closed, bearing on the lug pin) to +open (released, swung outward)."""
    g = sb.empty(name, loc=(0, 0, 0), parent=parent)
    g.rotation_euler = (0, 0, math.radians(az))
    rs = rocket.LUG_R
    zt = ROCKET_Z0
    B = Beams(hash(name) & 0xfff)
    # pedestal: welded box column with web stiffeners, base plate bolted to deck, bearing cap
    B.add((rs, 0, TABLE_Z), (rs, 0, zt - 0.35), 0.78, 0.7, up=V((0, 1, 0)))
    B.add((rs + 0.2, 0, TABLE_Z), (rs + 0.2, 0, TABLE_Z + 0.12), 1.2, 1.3, up=V((0, 1, 0)))
    for s in (-1, 1):
        B.add((rs + 0.42, s * 0.3, TABLE_Z + 0.1), (rs + 0.42, s * 0.3, zt - 1.2), 0.08, 0.3, up=V((0, 1, 0)))
        B.add((rs, s * 0.4, TABLE_Z + 0.1), (rs, s * 0.4, zt - 1.6), 0.5, 0.06, up=V((0, 1, 0)))
    B.add((rs - 0.05, 0, zt - 0.35), (rs - 0.05, 0, zt - 0.05), 0.95, 0.8, up=V((0, 1, 0)))
    ped = B.build(name + "Ped", M["clamp"], g)
    Bo = Beams(3)
    for k in range(8):
        a = 2 * math.pi * k / 8
        p = V((rs + 0.2 + 0.52 * math.cos(a), 0.56 * math.sin(a), TABLE_Z + 0.12))
        Bo.tube(p, p + V((0, 0, 0.07)), 0.045, 6)
        Bo.tube(p + V((0, 0, 0.07)), p + V((0, 0, 0.14)), 0.022, 6)
    Bo.build(name + "Bolts", M["steel_dark"], g)
    piv = V((rs + 0.75, 0, zt - 0.95))
    for s in (-1, 1):
        rocket.box(name + "Hinge", piv + V((0, s * 0.33, 0)), (0.36, 0.1, 0.5), M["clamp"], parent=g, bev=0.02)
    sb.prim("cyl", name + "HPin", loc=piv, rot=(math.pi / 2, 0, 0), vertices=24, radius=0.08, depth=0.86, mat=M["steel_bright"], parent=g)
    jaw = sb.empty(name + "Jaw", loc=piv, parent=g)
    J = Beams(4)
    pin = V((rocket.LUG_PIN, 0, zt + 0.56)) - piv
    for s in (-1, 1):
        y = s * 0.2
        pts = [V((0, y, 0)), V((0.05, y, 0.7)), V((0.02, y, 1.35)), pin + V((0.2, y, 0.28)), pin + V((-0.02, y, 0.26))]
        for a, b in zip(pts, pts[1:]):
            J.add(a, b, 0.12, 0.3, up=V((1, 0, 0)))
    J.add(pin + V((0.12, -0.26, 0.27)), pin + V((0.12, 0.26, 0.27)), 0.34, 0.2)
    J.add(V((0.02, -0.26, 1.2)), V((0.02, 0.26, 1.2)), 0.22, 0.3)
    J.build(name + "JawBody", M["clamp_jaw"], jaw)
    sb.prim("cyl", name + "Saddle", loc=pin + V((0, 0, 0.14)), rot=(math.pi / 2, 0, 0), vertices=32, radius=0.14,
            depth=0.46, mat=M["steel_bright"], parent=jaw)
    act_base = V((rs + 0.55, 0, TABLE_Z + 0.5))
    act_tip_local = V((0.35, 0, 0.9))
    # pneumatic lock bolt: housing on the hinge bracket, bolt engages the jaw side plate. Animate lock.location.y
    # from 0 (engaged) to +LOCK_TRAVEL (withdrawn) - the last mechanical 'click' before T-0.
    hp = piv + V((0.06, 0.55, 0.75))
    rocket.box(name + "LockHousing", hp + V((0, 0.12, 0)), (0.22, 0.3, 0.22), M["clamp"], parent=g, bev=0.015)
    sb.prim("cyl", name + "LockAir", loc=hp + V((0, 0.34, 0.0)), rot=(math.pi / 2, 0, 0), vertices=16, radius=0.04,
            depth=0.16, mat=M["steel_bright"], parent=g)
    rocket.pipe(name + "LockHose", [hp + V((0, 0.42, 0)), hp + V((0.1, 0.55, -0.3)), hp + V((0.2, 0.45, -1.2)),
                                     V((rs + 0.6, 0.45, TABLE_Z + 0.2))], 0.02, M["hose"], parent=g)
    lock = sb.empty(name + "Lock", loc=hp, parent=g)
    sb.prim("cyl", name + "LockBolt", loc=(0, -0.08, 0), rot=(math.pi / 2, 0, 0), vertices=24, radius=0.05, depth=0.34,
            mat=M["steel_bright"], parent=lock)
    return dict(root=g, jaw=jaw, pivot=piv, act_base=act_base, act_tip_local=act_tip_local, cyl_parent=g, open=55.0,
                lock=lock)


LOCK_TRAVEL = 0.16


def deck_lights(M, parent=None, energy=4500.0):
    """Practical LED floods on the launch-table corners aimed up at the vehicle base (visible fixtures)."""
    out = []
    for i, (x, y) in enumerate(((-TABLE_X + 0.8, -TABLE_Y + 0.8), (TABLE_X - 0.8, -TABLE_Y + 0.8), (TABLE_X - 0.8, TABLE_Y - 0.8))):
        p = V((x, y, TABLE_Z + 1.3))
        tgt = V((0, 0, ROCKET_Z0 + 1.5))
        fwd = (tgt - p).normalized()
        q = fwd.to_track_quat('Z', 'Y')
        B = Beams(700 + i)
        B.tube(p - V((0, 0, 1.3)), p - V((0, 0, 0.15)), 0.06, 8)
        B.add(p - fwd * 0.2, p + fwd * 0.1, 0.55, 0.4, up=V((0, 0, 1)))
        B.build("DeckLampBody%d" % i, M["lamp_body"], parent)
        fc = sb.prim("plane", "DeckLampLens%d" % i, loc=p + fwd * 0.11, size=0.46, mat=M["lamp_flood"])
        fc.scale = (1.0, 0.72, 1.0)
        fc.rotation_mode = 'QUATERNION'
        fc.rotation_quaternion = q
        fc.parent = parent
        sp = sb.light('SPOT', "DeckSpot%d" % i, loc=p + fwd * 0.3, target=tgt, energy=energy, color=(1.0, 0.94, 0.86),
                      size=0.35, spot=70, blend=0.5)
        out.append(sp)
    return out


def clamp_actuator_update(cl, M, name):
    """Hydraulic release cylinder from pedestal to jaw; a Damped Track keeps it aimed at the jaw lug."""
    g = cl["root"]
    base = cl["act_base"]
    tip = cl["pivot"] + cl["act_tip_local"]
    body_e = sb.empty(name + "CylE", loc=base, parent=g)
    tgt = sb.empty(name + "CylT", loc=cl["act_tip_local"], parent=cl["jaw"])
    L0 = (tip - base).length
    sb.prim("cyl", name + "Cyl", loc=(0, 0, L0 * 0.3), vertices=24, radius=0.1, depth=L0 * 0.6, mat=M["clamp"], parent=body_e)
    rodm = sb.prim("cyl", name + "Rod", loc=(0, 0, L0 * 0.75), vertices=16, radius=0.045, depth=L0 * 0.55, mat=M["steel_bright"], parent=body_e)
    c = body_e.constraints.new('DAMPED_TRACK')
    c.target = tgt
    c.track_axis = 'TRACK_Z'
    return body_e, rodm, tgt


def tail_masts(M, parent=None):
    """Two tail-service masts (+Y side) with carrier plates mating the aft-skirt umbilical plates.
    Returns list of dict(plate=<empty>, dir) - animate plate location (retract outward/down) at release."""
    out = []
    for k, a in enumerate((70.0, 110.0)):
        ar = math.radians(a)
        dirv = V((math.cos(ar), math.sin(ar), 0))
        side = dirv.cross(V((0, 0, 1)))
        base = dirv * 5.6 + V((0, 0, TABLE_Z))
        g = sb.empty("TSM%d" % k, parent=parent)
        B = Beams(40 + k)
        # framed mast: 4 H posts, girts, X bracing, clad hood (blast shield) on the vehicle side
        cs = [base + dirv * dx + side * dy for dx in (-0.8, 0.8) for dy in (-0.75, 0.75)]
        for c in cs:
            B.ibeam(c, c + V((0, 0, 6.0)), 0.25, 0.25, up=dirv)
        for zz in (0.2, 2.0, 4.0, 6.0):
            for i, j in ((0, 1), (2, 3), (0, 2), (1, 3)):
                B.angle(cs[i] + V((0, 0, zz)), cs[j] + V((0, 0, zz)), 0.12)
        for zz0, zz1 in ((0.2, 2.0), (2.0, 4.0), (4.0, 6.0)):
            B.angle(cs[2] + V((0, 0, zz0)), cs[3] + V((0, 0, zz1)), 0.1)
            B.angle(cs[0] + V((0, 0, zz0)), cs[1] + V((0, 0, zz1)), 0.1)
        B.box(base + V((0, 0, 6.25)) - dirv * 0.3, (2.4, 2.1, 0.5))
        B.build("TSMFrame%d" % k, M["tsm"], g)
        H = Beams(50 + k)
        hood_c = base - dirv * 0.95 + V((0, 0, 4.4))
        H.add(hood_c - V((0, 0, 1.8)), hood_c + V((0, 0, 1.8)), 0.08, 1.9, up=dirv)
        H.build("TSMHood%d" % k, M["clad_light"], g)
        plate = sb.empty("TSMPlate%d" % k, loc=dirv * (rocket.R + 0.12) + V((0, 0, ROCKET_Z0 + 2.3)), parent=g)
        plate.rotation_euler = (0, 0, ar)
        rocket.box("TSMCarrier%d" % k, (0.1, 0, 0), (0.18, 0.85, 1.05), M["clamp"], parent=plate, bev=0.02)
        for j in range(3):
            sb.prim("cyl", "TSMQD%d" % k, loc=(0.25, (j - 1) * 0.26, 0.1 * (j - 1)), rot=(0, math.pi / 2, 0), vertices=16,
                    radius=0.07, depth=0.16, mat=M["steel_bright"], parent=plate)
        p0 = plate.location
        p1 = base + V((0, 0, 5.2)) - dirv * 0.8
        for j in range(3):
            off = V((0, 0, (j - 1) * 0.28))
            mid = (p0 + p1) / 2 + V((0, 0, -0.9 - 0.2 * j))
            rocket.pipe("TSMHose%d_%d" % (k, j), [p0 + dirv * 0.2 + off, mid, p1 + off], 0.08 + 0.02 * (j == 1), M["hose"], parent=g)
        out.append(dict(root=g, plate=plate, dir=dirv))
    return out


def tower(M, parent=None, detail=1):
    """Fixed umbilical tower with swing arms. Returns dict(upper_arm=<pivot dict>, crew_arm=<pivot dict>, top_z,
    lamp_locs, beacons, lattice, corners)."""
    T = TOWER_C
    S = TOWER_S / 2
    B = Beams(101)      # primary steel
    P = Beams(102)      # platforms (grating)
    Rl = Beams(103)     # handrails (yellow)
    D = Beams(104)      # secondary: stairs, trays, galv
    nlev = int(TOWER_H / LEVEL_H)
    corners = [V((T.x + sx * S, T.y + sy * S, 0)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # H columns (0.6 m), splice plates every other level
    for c in corners:
        B.ibeam(c, c + V((0, 0, TOWER_H)), 0.6, 0.6, 0.045, 0.03, up=V((0, 1, 0)))
        for L in range(1, nlev, 2):
            B.box(c + V((0, 0, L * LEVEL_H + 1.2)), (0.7, 0.7, 0.5))
        B.box(c + V((0, 0, 0.3)), (1.4, 1.4, 0.6))        # base plate / grout pad
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        m_ = (a + b) / 2
        B.ibeam(m_, m_ + V((0, 0, TOWER_H)), 0.36, 0.36, up=(b - a).normalized())
    for L in range(nlev + 1):
        z = L * LEVEL_H
        for i in range(4):
            a, b = corners[i] + V((0, 0, z)), corners[(i + 1) % 4] + V((0, 0, z))
            B.ibeam(a, b, 0.3, 0.55)
            # gusset plates at column joints
            u = (b - a).normalized()
            n_ = u.cross(V((0, 0, 1)))
            for p in (a + u * 0.55, b - u * 0.55, (a + b) / 2):
                B.add(p - V((0, 0, 0.45)), p + V((0, 0, 0.45)), 0.9, 0.025, up=n_)
            if L < nlev:
                m_ = (a + b) / 2
                up = V((0, 0, LEVEL_H))
                for p0, p1 in ((a, m_ + up * 0.5), (b, m_ + up * 0.5), (a + up, m_ + up * 0.5), (b + up, m_ + up * 0.5)):
                    B.angle(p0, p1, 0.2, 0.02)
                B.ibeam(a + up * 0.5, b + up * 0.5, 0.2, 0.3)
        if L > 0:
            # floor framing + grating with a stair opening in the back-right quadrant
            B.ibeam(V((T.x - S, T.y, z)), V((T.x + S, T.y, z)), 0.25, 0.45)
            B.ibeam(V((T.x, T.y - S, z)), V((T.x, T.y + S, z)), 0.25, 0.45)
            for k in (-0.5, 0.5):
                B.ibeam(V((T.x - S, T.y + k * S, z)), V((T.x + S, T.y + k * S, z)), 0.15, 0.25)
            P.box(V((T.x - S * 0.5, T.y, z + 0.26)), (S, 2 * S, 0.06))
            P.box(V((T.x + S * 0.5, T.y - S * 0.5, z + 0.26)), (S, S, 0.06))
            for i in range(4):
                a, b = corners[i] + V((0, 0, z)), corners[(i + 1) % 4] + V((0, 0, z))
                for hh in (0.55, 1.1):
                    Rl.add(a + V((0, 0, hh)), b + V((0, 0, hh)), 0.045)
                for k in range(1, 9):
                    p = a.lerp(b, k / 9)
                    Rl.add(p + V((0, 0, 0.3)), p + V((0, 0, 1.1)), 0.04)
                D.add(a + V((0, 0, 0.36)), b + V((0, 0, 0.36)), 0.012, 0.15)     # toe board
    # internal stairs: two flights per level in the back-right quadrant
    for L in range(nlev):
        z0 = L * LEVEL_H + 0.3
        xs = T.x + S * 0.5
        y_lo, y_hi = T.y + 0.4, T.y + S - 0.5
        if L % 2 == 0:
            stair_flight(D, (xs - 0.7, y_hi, z0), (xs - 0.7, y_lo, z0 + LEVEL_H / 2), 0.95)
            stair_flight(D, (xs + 0.7, y_lo, z0 + LEVEL_H / 2), (xs + 0.7, y_hi, z0 + LEVEL_H), 0.95)
        else:
            stair_flight(D, (xs + 0.7, y_hi, z0), (xs + 0.7, y_lo, z0 + LEVEL_H / 2), 0.95)
            stair_flight(D, (xs - 0.7, y_lo, z0 + LEVEL_H / 2), (xs - 0.7, y_hi, z0 + LEVEL_H), 0.95)
        D.box(V((xs, y_lo - 0.55, z0 + LEVEL_H / 2)), (2.6, 1.1, 0.05))
    # vertical cable trays on the -X face and a caged ladder on the +X face
    for dy in (-1.8, 1.8):
        cable_tray(D, V((T.x - S - 0.55, T.y + dy, 0.5)), V((T.x - S - 0.55, T.y + dy, TOWER_H - 1.0)), 0.6)
    top = TOWER_H
    # roof, jib crane (lattice jib), lightning mast (lattice)
    P.box(V((T.x, T.y, top + 0.35)), (2 * S + 0.6, 2 * S + 0.6, 0.12))
    cp = V((T.x - S * 0.4, T.y + S * 0.4, top))
    B.ibeam(cp, cp + V((0, 0, 6.5)), 0.8, 0.8, up=V((0, 1, 0)))
    jd = V((0.55, -0.83, 0)).normalized()
    js = jd.cross(V((0, 0, 1)))
    j0 = cp + V((0, 0, 6.2)) - jd * 5.0
    j1 = cp + V((0, 0, 6.2)) + jd * 14.0
    for s in (-1, 1):
        B.angle(j0 + js * 0.45 * s, j1 + js * 0.45 * s, 0.14)
    B.angle(j0 + V((0, 0, 0.9)), j1 + V((0, 0, 0.9)) - jd * 1.0, 0.14)
    for k in range(20):
        pa = j0.lerp(j1, k / 20)
        pb = j0.lerp(j1, (k + 1) / 20)
        B.angle(pa + js * 0.45, pb + V((0, 0, 0.9)), 0.08)
        B.angle(pa - js * 0.45, pb + V((0, 0, 0.9)), 0.08)
        B.add(pa - js * 0.45, pa + js * 0.45, 0.06, 0.06)
    B.add(cp + V((0, 0, 8.4)), j1 + V((0, 0, 0.9)), 0.05, 0.05)
    B.add(cp + V((0, 0, 8.4)), j0 + V((0, 0, 0.9)), 0.05, 0.05)
    B.ibeam(cp + V((0, 0, 6.5)), cp + V((0, 0, 8.6)), 0.4, 0.4, up=V((0, 1, 0)))
    B.box(j0 + V((0, 0, -0.5)) + jd * 0.8, (2.0, 2.0, 1.6))
    B.box(j1 - jd * 2.5 + V((0, 0, -0.35)), (0.8, 0.8, 0.5))        # trolley
    mast_base = V((T.x + S * 0.5, T.y - S * 0.5, top))
    for k in range(3):
        a = 2 * math.pi * k / 3
        o = V((math.cos(a), math.sin(a), 0))
        B.add(mast_base + o * 0.5, mast_base + o * 0.12 + V((0, 0, 10.0)), 0.07, 0.07)
    for i in range(10):
        z0, z1 = i * 1.0, (i + 1) * 1.0
        for k in range(3):
            a0, a1 = 2 * math.pi * k / 3, 2 * math.pi * (k + 1) / 3
            f0, f1 = 0.5 - 0.038 * z0, 0.5 - 0.038 * z1
            B.add(mast_base + V((math.cos(a0) * f0, math.sin(a0) * f0, z0)), mast_base + V((math.cos(a1) * f1, math.sin(a1) * f1, z1)), 0.03, 0.03)
    B.add(mast_base + V((0, 0, 9.8)), mast_base + V((0, 0, 11.2)), 0.05, 0.05, up=V((0, 1, 0)))
    lattice = B.build("TowerLattice", M["tower"], parent)
    P.build("TowerDecks", M["grating"], parent)
    Rl.build("TowerRails", M["yellow"], parent)
    D.build("TowerSecondary", M["galv"], parent)
    # elevator shaft (clad) with machine room; enclosed equipment rooms on a few levels (lit windows)
    sh = sb.prim("cube", "ElevShaft", loc=(T.x - S * 0.5, T.y + S * 0.5, top / 2 + 1.0), scale=(1.7, 1.7, top / 2 + 1.0), mat=M["clad"])
    sh.parent = parent
    mr = sb.prim("cube", "ElevMachine", loc=(T.x - S * 0.5, T.y + S * 0.5, top + 3.4), scale=(1.9, 1.9, 1.3), mat=M["clad"])
    mr.parent = parent
    Rm = Beams(105)
    Wn = Beams(106)
    for L, quad in ((1, (-1, -1)), (4, (1, 1)), (7, (-1, -1)), (9, (1, -1))):
        z = L * LEVEL_H + 0.3
        c = V((T.x + quad[0] * S * 0.5, T.y + quad[1] * S * 0.5, z + 1.6))
        Rm.box(c, (S - 0.3, S - 0.3, 3.2))
        # windows / door on the outward faces
        for (dx, dy) in ((quad[0], 0), (0, quad[1])):
            n_ = V((dx, dy, 0))
            fc = c + n_ * ((S - 0.3) / 2 + 0.02)
            t_ = n_.cross(V((0, 0, 1)))
            Wn.box(fc + t_ * 0.9 + V((0, 0, 0.35)), (abs(t_.x) * 1.1 + 0.02, abs(t_.y) * 1.1 + 0.02, 0.7))
    Rm.build("TowerRooms", M["clad_light"], parent)
    Wn.build("TowerWindows", M["window"], parent)
    # vertical propellant / pneumatic lines on the front face (round, insulated) with clamps and flanges
    fy = T.y - S - 0.75
    for k, (dx, r, mk) in enumerate(((0.9, 0.30, "insul"), (1.75, 0.22, "insul"), (2.45, 0.13, "pipe_grey"),
                                     (-1.6, 0.17, "pipe_grey"), (-2.2, 0.10, "pipe_grey"))):
        pipe_run("TowerLine%d" % k, [(T.x + dx, fy, 0.5), (T.x + dx, fy, ARM_Z - 1.6 - 0.4 * k),
                                     (T.x + S - 0.6, fy, ARM_Z - 1.6 - 0.4 * k)], r, M[mk], parent, fl_mat=M["pipe_grey"])
    C = Beams(107)
    for L in range(1, nlev):
        z = L * LEVEL_H + 3.0
        C.ibeam(V((T.x - 2.6, fy + 0.35, z)), V((T.x + 3.0, fy + 0.35, z)), 0.15, 0.2)
        for dx in (0.9, 1.75, 2.45, -1.6, -2.2):
            C.add(V((T.x + dx, fy + 0.35, z)), V((T.x + dx, fy, z)), 0.06, 0.08)
        cabinet(C, V((T.x + S - 1.2, T.y - S + 0.6, L * LEVEL_H + 1.2)), (0.8, 0.45, 1.3))
    C.build("TowerPipeSupports", M["steel_dark"], parent)
    # work lights: bracket-mounted fixtures on the front corners of each level
    lamp_locs = []
    for L in range(1, nlev + 1):
        z = L * LEVEL_H + 2.4
        for c in (corners[0], corners[1]):
            p = c + V((0, -0.55, z))
            rocket.box("WorkLamp", p, (0.38, 0.22, 0.26), M["lamp_body"], parent=parent, bev=0.02)
            f = sb.prim("plane", "WorkLampFace", loc=p + V((0, -0.12, 0)), rot=(math.pi / 2, 0, 0), size=0.3, mat=M["lamp_sodium"])
            f.parent = parent
            lamp_locs.append(p + V((0, -0.4, 0)))
    beacons = []
    for c in corners:
        s_ = sb.prim("sphere", "Beacon", loc=c + V((0, 0, top + 0.8)), radius=0.18, segments=12, ring_count=6, mat=M["beacon"])
        s_.parent = parent
        beacons.append(s_)
    s_ = sb.prim("sphere", "Beacon", loc=mast_base + V((0, 0, 11.3)), radius=0.2, segments=12, ring_count=6, mat=M["beacon"])
    s_.parent = parent
    beacons.append(s_)
    upper = _swing_arm("UpperArm", V((T.x + S, T.y - S, ARM_Z - 1.4)), V((0.0, rocket.R + 0.25, ARM_Z - 1.4)), M,
                       parent, hood=True, height=2.6)
    crew_piv = V((T.x + S, T.y - S, CREW_Z))
    crew = _swing_arm("CrewArm", crew_piv, crew_piv + V((-12.0, 0, 0)), M, parent, hood=False, height=3.0, room=True)
    crew["pivot"].location = crew_piv + V((0, -0.9, 0))
    return dict(upper_arm=upper, crew_arm=crew, top_z=top, mast_top=mast_base.z + 11.2, lamp_locs=lamp_locs,
                beacons=beacons, lattice=lattice, corners=corners)


def _swing_arm(name, piv, tip, M, parent, hood=False, height=2.5, room=False):
    """Box-truss swing arm (angle chords, lacing, walkway, handrails) from pivot to tip. Returns dict(pivot, length, rest)."""
    pv = sb.empty(name + "Pivot", loc=piv, parent=parent)
    d = (tip - piv)
    d.z = 0
    L = d.length
    ang = math.atan2(d.y, d.x)
    pv.rotation_euler = (0, 0, ang)
    B = Beams(hash(name) & 0xfff)
    w = 1.2
    h = height
    for sy in (-1, 1):
        for sz in (0, 1):
            B.ibeam((0.3, sy * w, sz * h), (L, sy * w, sz * h), 0.2, 0.25)
    nb_ = max(2, int(L / 1.6))
    for k in range(nb_ + 1):
        x = 0.3 + (L - 0.3) * k / nb_
        B.angle((x, -w, 0), (x, w, 0), 0.12)
        B.angle((x, -w, h), (x, w, h), 0.12)
        for sy in (-1, 1):
            B.angle((x, sy * w, 0), (x, sy * w, h), 0.12)
            if k < nb_:
                x2 = 0.3 + (L - 0.3) * (k + 1) / nb_
                B.angle((x, sy * w, 0 if k % 2 else h), (x2, sy * w, h if k % 2 else 0), 0.1)
        if k < nb_:
            x2 = 0.3 + (L - 0.3) * (k + 1) / nb_
            B.angle((x, -w, h), (x2, w, h), 0.08)
    B.ibeam((0, 0, -1.0), (0, 0, h + 1.0), 0.6, 0.6, up=V((0, 1, 0)))
    for zz in (-0.6, h + 0.6):
        B.box((0.3, 0, zz), (1.0, 0.9, 0.35))
    B.build(name + "Truss", M["tower"], pv)
    fl = sb.prim("cube", name + "Walk", loc=(L / 2 + 0.15, 0, 0.12), scale=(L / 2 - 0.15, w - 0.1, 0.03), mat=M["grating"])
    fl.parent = pv
    R_ = Beams(7)
    for sy in (-1, 1):
        R_.add((0.3, sy * (w - 0.1), 1.1), (L, sy * (w - 0.1), 1.1), 0.04)
        R_.add((0.3, sy * (w - 0.1), 0.55), (L, sy * (w - 0.1), 0.55), 0.04)
    R_.build(name + "Rail", M["yellow"], pv)
    if hood:
        rocket.box(name + "Hood", (L - 0.4, 0, h * 0.62), (0.9, 1.6, 1.6), M["tsm"], parent=pv, bev=0.04)
        cable_tray(R_ := Beams(8), V((0.8, 0.6, h - 0.2)), V((L - 1.0, 0.6, h - 0.2)), 0.45)
        R_.build(name + "Tray", M["galv"], pv)
        for j in range(3):
            y = (j - 1) * 0.45
            rocket.pipe(name + "Hose%d" % j, [V((0.8, y, -0.2)), V((L * 0.35, y, -0.6 - 0.1 * j)), V((L * 0.7, y, -0.45)),
                                             V((L - 0.5, y * 0.6, h * 0.35))], 0.09, M["hose"], parent=pv)
    if room:
        rocket.box(name + "Room", (L - 1.5, 0, h / 2 + 0.1), (3.0, 2.8, h + 0.4), M["room"], parent=pv, bev=0.04)
        rocket.box(name + "Skin", (L * 0.45, 0, h / 2 + 0.1), (L * 0.75, 2.5, h + 0.2), M["clad"], parent=pv, bev=0.03)
    return dict(pivot=pv, length=L, rest=ang)


def lightning_towers(M, parent=None, n=3, radius=150.0, height=108.0, az0=85.0):
    """Freestanding triangular lattice lightning masts (angle legs, lacing), guy wires to ground anchors."""
    tops = []
    B = Beams(201)
    for i in range(n):
        a = math.radians(az0 + i * 360.0 / n)
        c = V((math.cos(a) * radius, math.sin(a) * radius, 0))
        s = 1.2
        pts = [c + V((math.cos(t) * s, math.sin(t) * s, 0)) for t in (0, 2.094, 4.188)]
        taper = 0.35
        for p in pts:
            B.angle(p, c + (p - c) * taper + V((0, 0, height)), 0.2, 0.02)
            B.box(p + V((0, 0, 0.3)), (0.9, 0.9, 0.6))
        nl = int(height / 2.4)
        for k in range(nl):
            z0 = k * height / nl
            z1 = (k + 1) * height / nl
            f0 = 1 - (1 - taper) * z0 / height
            f1 = 1 - (1 - taper) * z1 / height
            for j in range(3):
                a0 = c + (pts[j] - c) * f0 + V((0, 0, z0))
                b1 = c + (pts[(j + 1) % 3] - c) * f1 + V((0, 0, z1))
                B.angle(a0, b1, 0.07)
                B.angle(a0, c + (pts[(j + 1) % 3] - c) * f0 + V((0, 0, z0)), 0.07)
        B.tube(c + V((0, 0, height)), c + V((0, 0, height + 6.0)), 0.09, 8)
        tops.append(c + V((0, 0, height + 5.0)))
    o = B.build("LightningTowers", M["galv"], parent)
    for i in range(n):
        a = tops[i]
        out_ = V((a.x, a.y, 0)).normalized()
        b = V((a.x, a.y, 0)) + out_ * 70.0 + V((0, 0, 0.5))
        pts = []
        for k in range(25):
            t = k / 24
            p = a.lerp(b, t)
            p.z -= 6.0 * 4 * t * (1 - t)
            pts.append(p)
        sb.tube_along("GuyWire%d" % i, pts, radius=0.04, mat=M["wire"], bevel_res=2)
    return o, tops


def flood_masts(M, parent=None, azs=(-128.0, -62.0, 28.0, 152.0), radius=FLOOD_R, height=28.0, target_z=None,
                energy=1.0e5, color=(1.0, 0.92, 0.80)):
    """Lighting towers: tapered octagonal galvanized pole on a concrete pier, caged ladder, head platform with a
    framed array of visored lamps aimed at the vehicle; one SPOT light per mast. Returns list of dict(spot, head)."""
    out = []
    tz = target_z if target_z is not None else ROCKET_Z0 + 36.0
    for i, a in enumerate(azs):
        ar = math.radians(a)
        c = V((math.cos(ar) * radius, math.sin(ar) * radius, 0))
        pole = rocket.lathe("FloodPole%d" % i, [(0.62, 0.0), (0.62, 1.2), (0.46, 1.25), (0.42, 1.3), (0.24, height), (0.2, height + 0.05)],
                            segs=8, mat=M["galv"], parent=parent, auto=10)
        pole.location = c
        pier = sb.prim("cyl", "FloodPier%d" % i, loc=c + V((0, 0, 0.25)), vertices=24, radius=1.1, depth=0.5, mat=M["concrete_clean"])
        pier.parent = parent
        head_c = c + V((0, 0, height + 1.2))
        tgt = V((0, 0, tz))
        fwd = (tgt - head_c).normalized()
        fh = V((fwd.x, fwd.y, 0)).normalized()
        right = fwd.cross(V((0, 0, 1))).normalized()
        upv = right.cross(fwd)
        B = Beams(300 + i)
        ladder_cage(B, c - fh * 0.5, c - fh * 0.5 + V((0, 0, height - 0.4)), -fh)
        # head platform
        B.box(c + V((0, 0, height)), (3.2, 3.2, 0.06))
        for s in (-1, 1):
            B.add(c + V((0, 0, height)) + right * 3.6 * s - fh * 1.2, c + V((0, 0, height)) + right * 3.6 * s + fh * 1.2, 0.12, 0.2)
        B.ibeam(c + V((0, 0, height)) - right * 3.8, c + V((0, 0, height)) + right * 3.8, 0.2, 0.25)
        for k in range(-3, 4):
            B.add(c + V((0, 0, height)) + right * k * 1.16, head_c + right * k * 1.16 + upv * 1.3, 0.08, 0.08)
        B.add(head_c - right * 3.5, head_c + right * 3.5, 0.14, 0.14)
        B.add(head_c - right * 3.5 + upv * 1.3, head_c + right * 3.5 + upv * 1.3, 0.14, 0.14)
        B.add(head_c - right * 3.5 + upv * 2.6, head_c + right * 3.5 + upv * 2.6, 0.14, 0.14)
        # rails
        for s in (-1, 1):
            B.add(c + V((0, 0, height + 1.1)) - right * 3.8 + fh * 1.3 * s, c + V((0, 0, height + 1.1)) + right * 3.8 + fh * 1.3 * s, 0.04, 0.04)
        cabinet(B, c - fh * 0.9 + V((0, 0, 2.2)), (0.7, 0.45, 1.0))
        B.build("FloodFrame%d" % i, M["galv"], parent)
        H = Beams(320 + i)
        lenses = []
        q = fwd.to_track_quat('Z', 'Y')
        for r_ in range(2):
            for k in range(6):
                p = head_c + right * (-2.9 + k * 1.16) + upv * (r_ * 1.3 + 0.65)
                H.add(p - fwd * 0.3, p + fwd * 0.12, 0.72, 0.5, up=upv)                   # housing
                H.add(p + fwd * 0.12 + upv * 0.3, p + fwd * 0.42 + upv * 0.36, 0.76, 0.02, up=upv)  # visor
                for s in (-1, 1):
                    H.add(p + fwd * 0.12 + right * 0.37 * s, p + fwd * 0.35 + right * 0.37 * s + upv * 0.05, 0.02, 0.5, up=upv)
                H.add(p - fwd * 0.3, p - fwd * 0.4, 0.35, 0.2, up=upv)                   # ballast/driver
                lenses.append(p + fwd * 0.125)
        H.build("FloodHeads%d" % i, M["lamp_body"], parent)
        for p in lenses:
            fc = sb.prim("plane", "FloodLens", loc=p, size=0.64, mat=M["lamp_flood"])
            fc.scale = (1.0, 0.7, 1.0)
            fc.rotation_mode = 'QUATERNION'
            fc.rotation_quaternion = q
            fc.parent = parent
        sp = sb.light('SPOT', "FloodSpot%d" % i, loc=head_c + fwd * 1.0, target=tgt, energy=energy, color=color,
                      size=2.0, spot=64, blend=0.55)
        out.append(dict(spot=sp, head=head_c, mast=pole))
    return out


def lox_farm(M, parent=None, center=(95.0, -40.0)):
    """Propellant storage: insulated sphere on H-legs with sway bracing and equator girder, stair tower with flights,
    top platform + vent stack, piping, a frosted ambient-vaporizer bank, horizontal tanks on saddles."""
    c = V((center[0], center[1], 0))
    rs = 9.5
    zc = rs + 5.0
    sph = sb.prim("sphere", "LoxSphere", loc=c + V((0, 0, zc)), radius=rs, segments=96, ring_count=48, mat=M["tank_white"])
    sph.parent = parent
    B = Beams(401)
    legs = []
    for k in range(10):
        a = 2 * math.pi * k / 10
        p = c + V((math.cos(a) * rs * 0.97, math.sin(a) * rs * 0.97, 0))
        B.ibeam(p, p + V((0, 0, zc)), 0.45, 0.45, up=V((math.cos(a), math.sin(a), 0)))
        B.box(p + V((0, 0, 0.35)), (1.2, 1.2, 0.7))
        legs.append(p)
    for k in range(10):
        a, b = legs[k], legs[(k + 1) % 10]
        B.tube(a + V((0, 0, 1.0)), b + V((0, 0, zc - 2.0)), 0.05, 6)
        B.tube(b + V((0, 0, 1.0)), a + V((0, 0, zc - 2.0)), 0.05, 6)
    for k in range(40):
        a0, a1 = 2 * math.pi * k / 40, 2 * math.pi * (k + 1) / 40
        B.ibeam(c + V((math.cos(a0) * rs * 1.005, math.sin(a0) * rs * 1.005, zc)), c + V((math.cos(a1) * rs * 1.005, math.sin(a1) * rs * 1.005, zc)), 0.2, 0.5)
    st = c + V((rs + 3.2, 0, 0))
    scs = [st + V((dx, dy, 0)) for dx in (-1.3, 1.3) for dy in (-1.8, 1.8)]
    for p in scs:
        B.ibeam(p, p + V((0, 0, 2 * rs + 6.0)), 0.2, 0.2, up=V((0, 1, 0)))
    for lv in range(1, 8):
        z = lv * (2 * rs + 6.0) / 7
        for i, j in ((0, 1), (2, 3), (0, 2), (1, 3)):
            B.angle(scs[i] + V((0, 0, z)), scs[j] + V((0, 0, z)), 0.1)
        B.box(st + V((0, 0, z)), (2.6, 3.6, 0.05))
    B.build("LoxSteel", M["galv"], parent)
    S = Beams(402)
    hh = (2 * rs + 6.0) / 7
    for lv in range(7):
        z0 = lv * hh
        if lv % 2 == 0:
            stair_flight(S, st + V((-0.6, 1.6, z0)), st + V((-0.6, -1.6, z0 + hh)), 0.9)
        else:
            stair_flight(S, st + V((0.6, -1.6, z0)), st + V((0.6, 1.6, z0 + hh)), 0.9)
    S.box(c + V((0, 0, 2 * rs + 5.3)), (6.0, 6.0, 0.08))
    S.ibeam(st + V((0, 0, 2 * rs + 5.5)), c + V((1.5, 0, 2 * rs + 5.4)), 1.0, 0.25)
    S.tube(c + V((0, 0, 2 * rs + 4.8)), c + V((0, 0, 2 * rs + 9.0)), 0.25, 12)
    S.build("LoxStair", M["galv"], parent)
    # ambient vaporizer bank (finned tubes, frosted)
    Vb = Beams(403)
    v0 = c + V((-6.0, -rs - 7.0, 0))
    for i in range(6):
        for j in range(3):
            p = v0 + V((i * 1.3, j * 1.3, 0))
            for f in range(4):
                a = math.pi / 4 * f
                Vb.add(p + V((0, 0, 0.4)), p + V((0, 0, 7.5)), 0.9, 0.012, up=V((math.cos(a), math.sin(a), 0)))
            Vb.tube(p + V((0, 0, 0.4)), p + V((0, 0, 7.6)), 0.06, 8)
    Vb.build("Vaporizers", M["frost"], parent)
    Fr = Beams(404)
    for i in range(6):
        for j in range(3):
            p = v0 + V((i * 1.3, j * 1.3, 0))
            Fr.box(p + V((0, 0, 0.2)), (0.2, 0.2, 0.4))
    Fr.box(v0 + V((3.25, 1.3, 7.9)), (8.2, 3.2, 0.12))
    Fr.build("VapFrame", M["galv"], parent)
    for k in range(3):
        y = -12.0 + k * 7.0
        t = sb.prim("cyl", "HTank%d" % k, loc=c + V((-26.0, y, 3.6)), rot=(0, math.pi / 2, 0), vertices=48,
                    radius=2.6, depth=22.0, mat=M["tank_white"])
        t.parent = parent
        for s in (-1, 1):
            e = sb.prim("sphere", "HTankEnd", loc=c + V((-26.0 + s * 11.0, y, 3.6)), radius=2.6, segments=32,
                        ring_count=16, mat=M["tank_white"])
            e.scale = (0.45, 1, 1)
            e.parent = parent
        Sd = Beams(410 + k)
        for x in (-7.0, 0.0, 7.0):
            Sd.box(c + V((-26.0 + x, y, 0.6)), (0.6, 4.2, 1.2))
        Sd.build("Saddles%d" % k, M["concrete_clean"], parent)
    # pipe rack to the pad (round pipes on steel bents)
    P = Beams(420)
    a = c + V((-8.0, 0, 0))
    b = V((14.0, -9.0, 0))
    n = int((a - b).length / 7.0)
    for k in range(n + 1):
        p = a.lerp(b, k / n)
        side = (b - a).normalized().cross(V((0, 0, 1)))
        for s in (-1, 1):
            P.ibeam(p + side * 0.9 * s, p + side * 0.9 * s + V((0, 0, 3.0)), 0.2, 0.2)
        P.ibeam(p - side * 1.1 + V((0, 0, 3.0)), p + side * 1.1 + V((0, 0, 3.0)), 0.2, 0.3)
    P.build("PipeRack", M["steel_dark"], parent)
    side = (b - a).normalized().cross(V((0, 0, 1)))
    for k, (off, r, mk) in enumerate(((-0.6, 0.35, "insul"), (0.0, 0.25, "insul"), (0.55, 0.15, "pipe_grey"))):
        pipe_run("RackPipe%d" % k, [a + side * off + V((0, 0, 3.15 + r)), b + side * off + V((0, 0, 3.15 + r))], r, M[mk], parent,
                 fl_mat=M["pipe_grey"])
    return dict(sphere=sph, vent=c + V((0, 0, 2 * rs + 9.0)), vaporizers=v0)


def water_tower(M, parent=None, loc=(-60.0, 260.0), height=62.0, rt=9.0):
    c = V((loc[0], loc[1], 0))
    B = Beams(501)
    for k in range(6):
        a = 2 * math.pi * k / 6
        p = c + V((math.cos(a) * rt * 1.1, math.sin(a) * rt * 1.1, 0))
        top = c + V((math.cos(a) * rt * 0.75, math.sin(a) * rt * 0.75, height))
        B.tube(p, top, 0.45, 12)
    for zz in (15.0, 32.0, 48.0):
        for k in range(6):
            a0, a1 = 2 * math.pi * k / 6, 2 * math.pi * (k + 1) / 6
            f = 1.1 - 0.35 * zz / height
            B.ibeam(c + V((math.cos(a0) * rt * f, math.sin(a0) * rt * f, zz)), c + V((math.cos(a1) * rt * f, math.sin(a1) * rt * f, zz)), 0.3, 0.4)
    B.tube(c, c + V((0, 0, height)), 1.2, 16)
    o = B.build("WaterTowerLegs", M["tower"], parent)
    t = sb.prim("sphere", "WaterTank", loc=c + V((0, 0, height + rt * 0.8)), radius=rt, segments=64, ring_count=32, mat=M["tank_white"])
    t.scale = (1, 1, 0.8)
    t.parent = parent
    return o


def truck(name, loc, heading, M, parent=None, kind="tanker"):
    """Conventional-cab semi with a cryogenic tanker (banded, rear cabinet) or box trailer (~18 m)."""
    g = sb.empty(name, loc=loc, parent=parent)
    g.rotation_euler = (0, 0, math.radians(heading))
    # tractor
    rocket.box(name + "Cab", (6.6, 0, 2.05), (2.2, 2.45, 2.3), M["truck_cab"], parent=g, bev=0.14)
    rocket.box(name + "Sleeper", (5.2, 0, 2.3), (1.2, 2.4, 2.8), M["truck_cab"], parent=g, bev=0.12)
    rocket.box(name + "Hood", (8.25, 0, 1.45), (1.7, 2.1, 1.3), M["truck_cab"], parent=g, bev=0.2)
    rocket.box(name + "Grille", (9.12, 0, 1.4), (0.05, 1.1, 1.0), M["chrome"], parent=g, bev=0.01)
    rocket.box(name + "Bumper", (9.2, 0, 0.72), (0.25, 2.45, 0.35), M["chrome"], parent=g, bev=0.05)
    rocket.box(name + "Wind", (7.72, 0, 2.55), (0.04, 2.1, 0.9), M["glass"], parent=g, bev=0.0)
    for s in (-1, 1):
        rocket.box(name + "SideWin", (6.9, s * 1.23, 2.6), (1.1, 0.02, 0.7), M["glass"], parent=g, bev=0.0)
        rocket.box(name + "Mirror", (7.8, s * 1.45, 2.6), (0.08, 0.2, 0.45), M["lamp_body"], parent=g, bev=0.01)
        t = sb.prim("cyl", name + "Fuel", loc=(6.3, s * 1.05, 1.0), rot=(0, math.pi / 2, 0), vertices=24, radius=0.33, depth=1.4, mat=M["chrome"])
        t.parent = g
        st = sb.prim("cyl", name + "Stack", loc=(5.9, s * 1.1, 3.4), vertices=12, radius=0.08, depth=2.2, mat=M["chrome"])
        st.parent = g
        hl = sb.prim("cube", name + "Head", loc=(9.1, s * 0.85, 1.25), scale=(0.03, 0.18, 0.1), mat=M["marker"])
        hl.parent = g
    rocket.box(name + "Chassis", (3.0, 0, 0.8), (12.0, 1.0, 0.3), M["steel_dark"], parent=g, bev=0.02)
    if kind == "tanker":
        t = sb.prim("cyl", name + "Tank", loc=(-1.6, 0, 2.2), rot=(0, math.pi / 2, 0), vertices=64, radius=1.25, depth=11.4, mat=M["steel_bright"])
        t.parent = g
        for s in (-1, 1):
            e = sb.prim("sphere", name + "TankEnd", loc=(-1.6 + s * 5.7, 0, 2.2), radius=1.25, segments=32, ring_count=16, mat=M["steel_bright"])
            e.scale = (0.35, 1, 1)
            e.parent = g
        for k in range(6):
            b = sb.prim("torus", name + "Band", loc=(-6.5 + k * 1.95, 0, 2.2), rot=(0, math.pi / 2, 0), major_radius=1.27,
                        minor_radius=0.03, major_segments=48, minor_segments=6, mat=M["steel_dark"])
            b.parent = g
        rocket.box(name + "Cab2", (-7.6, 0, 1.7), (0.7, 2.2, 1.6), M["truck"], parent=g, bev=0.05)
        rocket.box(name + "Walk", (-1.6, 0, 3.52), (9.0, 0.6, 0.05), M["grating"], parent=g, bev=0.0)
    else:
        rocket.box(name + "Box", (-1.4, 0, 2.4), (12.6, 2.5, 3.0), M["tank_white"], parent=g, bev=0.05)
    for x in (8.1, 4.1, 2.9, -4.8, -6.1):
        for s in (-1, 1):
            w = sb.prim("cyl", name + "Wheel", loc=(x, s * 1.02, 0.52), rot=(math.pi / 2, 0, 0), vertices=24, radius=0.52,
                        depth=0.55, mat=M["tyre"])
            w.parent = g
            h_ = sb.prim("cyl", name + "Hub", loc=(x, s * 1.3, 0.52), rot=(math.pi / 2, 0, 0), vertices=16, radius=0.3,
                         depth=0.02, mat=M["chrome"])
            h_.parent = g
    for x in (-7.2, -4.0, -0.8, 2.4):
        for s in (-1, 1):
            m_ = sb.prim("sphere", name + "Marker", loc=(x, s * 1.3, 0.95), radius=0.05, segments=8, ring_count=4, mat=M["marker"])
            m_.parent = g
    return g


def building(name, c, size, heading, M, parent=None, lit=0.3, seed=0):
    """Low service building: formed walls, parapet, rooftop HVAC units, roll-up door, wall packs, lit windows."""
    rs = random.Random(seed)
    g = sb.empty(name, loc=c, parent=parent)
    g.rotation_euler = (0, 0, math.radians(heading))
    sx, sy, sz = size
    b = rocket.box(name + "Walls", (0, 0, sz / 2), (sx, sy, sz), M["concrete_clean"], parent=g, bev=0.03)
    rocket.box(name + "Parapet", (0, 0, sz + 0.25), (sx + 0.2, sy + 0.2, 0.5), M["concrete_clean"], parent=g, bev=0.02)
    for k in range(rs.randint(2, 4)):
        rocket.box(name + "HVAC", (rs.uniform(-sx / 3, sx / 3), rs.uniform(-sy / 3, sy / 3), sz + 0.9), (2.0, 1.4, 1.1), M["cabinet"], parent=g, bev=0.04)
    rocket.box(name + "Door", (sx / 2 + 0.02, 0, 2.2), (0.04, 3.6, 4.2), M["clad"], parent=g, bev=0.0)
    W = Beams(600 + seed)
    for k in range(int(sx / 3)):
        if rs.random() < 0.6:
            W.box((-sx / 2 + 1.5 + k * 3.0, -sy / 2 - 0.02, sz * 0.6), (1.2, 0.03, 0.8), tone=1.0 if rs.random() < lit else 0.0)
    if W.v:
        W.build(name + "Win", M["window"], g)
    lp = sb.prim("cube", name + "WallPack", loc=(sx / 2 + 0.15, 2.4, sz - 0.6), scale=(0.1, 0.2, 0.12), mat=M["lamp_sodium"])
    lp.parent = g
    return g


def fence(name, a, b, M, parent=None, h=2.4, post=3.0):
    """Chain-link fence: galvanized posts + rails + barbed-wire outriggers + dithered wire mesh."""
    a, b = V(a), V(b)
    B = Beams(hash(name) & 0xfff)
    n = max(1, int((b - a).length / post))
    for k in range(n + 1):
        p = a.lerp(b, k / n)
        B.tube(p, p + V((0, 0, h)), 0.04, 8)
        B.tube(p + V((0, 0, h)), p + V((0, 0, h + 0.45)), 0.02, 6)
    B.tube(a + V((0, 0, h)), b + V((0, 0, h)), 0.025, 6)
    B.tube(a + V((0, 0, 0.1)), b + V((0, 0, 0.1)), 0.015, 6)
    for dz in (0.15, 0.3, 0.45):
        B.tube(a + V((0, 0, h + dz)), b + V((0, 0, h + dz)), 0.006, 4)
    o = B.build(name, M["fence"], parent)
    d = (b - a)
    pn = sb.prim("plane", name + "Mesh", loc=(a + b) / 2 + V((0, 0, h / 2)), size=1.0, mat=M["chainlink"])
    pn.scale = (d.length / 2, h / 2, 1)
    pn.rotation_euler = (math.pi / 2, 0, math.atan2(d.y, d.x))
    pn.parent = parent
    return o


def grass_patch(name, center, size, heading, M, height_fn=None, n=9000, parent=None, seed=5, blade=(0.25, 0.9)):
    """Clumped grass/scrub blades (thin tapered triangles) scattered over a rectangle, following height_fn(x, y)."""
    rs = random.Random(seed)
    c = V(center)
    hr = math.radians(heading)
    ax = V((math.cos(hr), math.sin(hr), 0))
    ay = V((-math.sin(hr), math.cos(hr), 0))
    verts, faces = [], []
    clumps = [(rs.uniform(-size[0] / 2, size[0] / 2), rs.uniform(-size[1] / 2, size[1] / 2)) for _ in range(n // 12)]
    for i in range(n):
        cx, cy = clumps[rs.randrange(len(clumps))]
        u = cx + rs.gauss(0, 0.35)
        v = cy + rs.gauss(0, 0.35)
        p = c + ax * u + ay * v
        z = height_fn(p.x, p.y) if height_fn else 0.0
        h = rs.uniform(*blade)
        w = rs.uniform(0.012, 0.03)
        a = rs.uniform(0, 2 * math.pi)
        lean = V((math.cos(a), math.sin(a), 0)) * rs.uniform(0.05, 0.4) * h
        side = V((-math.sin(a + 1.3), math.cos(a + 1.3), 0)) * w
        base = V((p.x, p.y, z - 0.02))
        tip = base + V((0, 0, h)) + lean
        k = len(verts)
        verts += [tuple(base - side), tuple(base + side), tuple(tip)]
        faces.append((k, k + 1, k + 2))
    o = sb.mesh_obj(name, verts, faces, M["grass"], smooth_shade=False)
    if parent:
        o.parent = parent
    return o


def berm(name, center, length, heading, M, parent=None, h=3.2, w=9.0, grass=True):
    """Earth blast berm: long rounded mound, scrubby grass tufts on top (foreground scale cue)."""
    c = V(center)
    verts, faces = [], []
    nx, ny = 80, 20
    hr = math.radians(heading)
    ax = V((math.cos(hr), math.sin(hr), 0))
    ay = V((-math.sin(hr), math.cos(hr), 0))
    rs = random.Random(7)

    def hfun(u, v):
        endf = max(0.0, min(1.0, (length / 2 - abs(u)) / (w * 0.6)))
        hh = h * endf * (max(0.0, math.cos(math.pi * v / w)) ** 1.4 if abs(v) < w / 2 else 0)
        return hh + 0.15 * math.sin(u * 1.7 + v * 0.9) * endf - 0.1
    for i in range(nx + 1):
        u = -length / 2 + length * i / nx
        for j in range(ny + 1):
            v = -w / 2 + w * j / ny
            p = c + ax * u + ay * v + V((0, 0, hfun(u, v) + rs.uniform(-0.04, 0.04)))
            verts.append(tuple(p))
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            faces.append((a, a + ny + 1, a + ny + 2, a + 1))
    o = sb.mesh_obj(name, verts, faces, M["ground"], True)
    if parent:
        o.parent = parent
    if grass:
        def hxy(x, y):
            q = V((x, y, 0)) - V((c.x, c.y, 0))
            return c.z + hfun(q.dot(ax), q.dot(ay))
        grass_patch(name + "Grass", c, (length * 0.9, w * 0.9), heading, M, hxy, n=int(length * w * 22), parent=parent,
                    blade=(0.1, 0.42))
    return o


def treeline(M, parent=None, r=2600.0, seed=3):
    """Distant low tree/scrub silhouette ring on the horizon (6-16 m)."""
    rs = random.Random(seed)
    verts, faces = [], []
    n = 720
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        h = 7 + 4 * math.sin(a * 7 + 1) + 3 * math.sin(a * 23) + rs.uniform(-1.5, 2.5)
        c, s = math.cos(a), math.sin(a)
        verts += [(c * r, s * r, -2.0), (c * r, s * r, h)]
    for i in range(n):
        faces.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
    o = sb.mesh_obj("Treeline", verts, faces, M["treeline"], True)
    if parent:
        o.parent = parent
    return o


def ground(M, parent=None):
    g = sb.prim("plane", "Ground", size=12000, mat=M["ground"])
    hs = sb.prim("cyl", "Hardstand", loc=(0, 0, 0.0), vertices=128, radius=78.0, depth=0.12, mat=M["pad_conc"])
    gr = sb.prim("cyl", "GravelRing", loc=(0, 0, -0.02), vertices=128, radius=92.0, depth=0.1, mat=M["gravel"])
    rd = sb.prim("plane", "RoadA", loc=(40, -260, 0.03), scale=(5, 260, 1), mat=M["asphalt"])
    rd.rotation_euler = (0, 0, math.radians(-8))
    rd2 = sb.prim("plane", "RoadB", loc=(60, -40, 0.03), scale=(140, 4, 1), mat=M["asphalt"])
    for o in (g, hs, gr, rd, rd2):
        o.parent = parent
    return g


def build_complex(detail=1, far=True, trucks=True):
    """Builds the whole launch complex (without the vehicle). Returns dict:
    M, clamps[4] (dict jaw/open), tsm[2] (dict plate/dir), tower (dict upper_arm/crew_arm pivots, lamp_locs, beacons),
    floods [dict spot/head], lox (dict sphere/vent)."""
    M = materials()
    root = sb.empty("Complex")
    ground(M, root)
    launch_mount(M, root)
    clamps = []
    for az in rocket.LUG_AZ:
        cl = clamp("Clamp%d" % az, az, M, root)
        clamp_actuator_update(cl, M, "Clamp%d" % az)
        clamps.append(cl)
    tsm = tail_masts(M, root)
    dl = deck_lights(M, root)
    tw = tower(M, root, detail)
    floods = flood_masts(M, root)
    lt, tops = lightning_towers(M, root)
    lox = lox_farm(M, root)
    # perimeter fence on the hardstand edge (camera side arc) + service buildings
    for k in range(10):
        a0 = math.radians(-170 + k * 16)
        a1 = math.radians(-170 + (k + 1) * 16)
        fence("PadFence%d" % k, (math.cos(a0) * 86, math.sin(a0) * 86, 0), (math.cos(a1) * 86, math.sin(a1) * 86, 0), M, root)
    building("BldgA", (-70.0, 40.0, 0), (18.0, 10.0, 5.5), 20.0, M, root, seed=1)
    building("BldgB", (-95.0, -30.0, 0), (12.0, 8.0, 4.5), -35.0, M, root, seed=2)
    if far:
        water_tower(M, root)
        treeline(M, root)
    if trucks:
        truck("Truck0", (70.0, -58.0, 0), 172.0, M, root, "tanker")
        truck("Truck1", (74.0, -66.0, 0), 176.0, M, root, "tanker")
        truck("Truck2", (-58.0, -70.0, 0), 20.0, M, root, "box")
    return dict(M=M, root=root, clamps=clamps, tsm=tsm, tower=tw, floods=floods, lox=lox, lightning_tops=tops, deck_lights=dl)
