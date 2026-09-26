"""The Mars colony (s24c, s25z share it): a real settlement, on the GLOBAL timeline so the ship that lands in s24c
stands on its pad when s25z pulls away, and another lifts off under the zoom-out. Colony centre at the origin on the
lib/mars.py terrain; sun from the east (+X); north (+Y) is the spaceport.

  hub + ~22 shielded habitats on rings, joined by corridors; three avenues (north to the spaceport, east to industry,
  south to the solar farm) with streetlights; four glowing greenhouse vaults and three geodesic domes; an industrial
  quarter (tank farm, spheres, ISRU plant, pipe racks, a radiator tower); a spaceport of five pads with blast berms,
  a lattice launch tower, standing ships, one landing (A), one leaving far off (C), one lifting in s25z (B); rovers.

build(F0, F1) -> dict(T, gz, A, B, C, PAD_*, posA, posB, posC, smix, ground, SUN, ...)."""
import math, os
import numpy as np
from mathutils import Vector as V
import bpy, sb, mars, ship, moon, suit

EV_TD = 2447            # ship A touches down on pad A
EV_LO_C = 2345          # ship C lifts off (far pad)
EV_LO_B = 2512          # ship B lifts off (pad B) - seen from above in s25z

PAD_A, PAD_B, PAD_D, PAD_E, PAD_C = V((-130.0, 560.0)), V((90.0, 610.0)), V((300.0, 700.0)), V((-330.0, 760.0)), V((-760.0, 1750.0))
AVENUE_N = [V((0, 30)), V((0, 250)), V((-20, 420)), V((-40, 520))]
AVENUE_E = [V((30, 0)), V((150, -5)), V((250, 20)), V((330, 60))]
AVENUE_S = [V((0, -30)), V((5, -120)), V((0, -200))]


def cp(src, **kw):
    """linked copy of a hidden prototype (copies inherit the prototype's hide flags)."""
    o = sb.dup(src, **kw)
    o.hide_render = False; o.hide_viewport = False
    return o


def _road_mat():
    m = mars.planet_mat("Road", offset=(0, 0, mars.R), local=True, haze="HazeR")
    m.node_tree.nodes["HazeR"].outputs[0].default_value = 1.0
    nb = sb.NB(m)
    bsdf = m.node_tree.nodes["Principled BSDF"]
    co = nb.coord('Object')
    trk = nb.wave(co, scale=0.9, wtype='BANDS', direction='X', dist=0.5, detail=1)
    lk = bsdf.inputs['Base Color'].links[0].from_socket
    nb.set('Base Color', nb.mix(nb.math('MULTIPLY', trk.outputs['Fac'], 0.25), nb.mix(0.4, lk, (0.15, 0.085, 0.05, 1)), (0.09, 0.055, 0.035, 1)))
    return m


def build(F0, F1, detail=1.0):
    SUN = mars.SUN_DAY
    T = mars.Terrain()

    def gz(x, y):
        return float(T.height(np.array([x]), np.array([y]))[0])

    # ------------------------------------------------------------ ground, sky, sun
    gmat = mars.planet_mat("MarsGround", offset=(0, 0, mars.R), local=True, haze="Haze")
    ground = mars.cap("MarsCap", T=T, mat=gmat)
    w, smix = mars.world(SUN, space_mix=0.0, sky_strength=float(os.environ.get("SB_MARS_SKY", "1.6")))
    sun = sb.light('SUN', "Sun", energy=float(os.environ.get("S24C_SUN", "4.4")), color=(1.0, 0.91, 0.80), angle=0.35)
    sun.rotation_mode = 'QUATERNION'; sun.rotation_quaternion = SUN.to_track_quat('Z', 'Y')
    sky_fill = sb.light('SUN', "SkyFill", energy=0.3, color=(0.95, 0.62, 0.38), angle=20.0)
    sky_fill.visible_glossy = False

    MT = mars.ground_mats()
    MT["glass"] = suit.bubble_mat("DomeGlass")
    road_m = _road_mat()
    tank_m = suit.graphite_mat("TankWhite", color=(0.78, 0.74, 0.69), wear=0.25, dust=0.45)
    rad_m = sb.mat("Radiator", (0.72, 0.72, 0.74), metal=0.6, rough=0.3)
    amber = sb.emit_mat("Amber", (1.0, 0.55, 0.18), 12.0)
    lampw = sb.emit_mat("StreetLamp", (1.0, 0.86, 0.66), 25.0)
    keep_out = []                                         # (x, y, r): no rocks / habs here

    # ------------------------------------------------------------ roads
    def road(name, pts, wdt=8.0, n=None, lights=True):
        pts = [V((p.x, p.y, 0)) for p in pts]
        L = sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1))
        n = n or max(20, int(L / 3.0))
        P = [sb.catmull(pts, i / (n - 1)) for i in range(n)]
        verts, faces = [], []
        for i, p in enumerate(P):
            t = (P[min(i + 1, n - 1)] - P[max(i - 1, 0)]); t.z = 0; t.normalize()
            nrm = V((-t.y, t.x, 0))
            for s_ in (-1, 1):
                q = p + nrm * s_ * wdt / 2
                verts.append((q.x, q.y, gz(q.x, q.y) + 0.07))
            if i:
                faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
        o = sb.mesh_obj(name, verts, faces, road_m)
        o.visible_shadow = False
        if lights:
            acc = 0.0
            for i in range(1, n):
                acc += (P[i] - P[i - 1]).length
                if acc > 28.0:
                    acc = 0.0
                    t = (P[min(i + 1, n - 1)] - P[i - 1]); t.z = 0; t.normalize()
                    q = P[i] + V((-t.y, t.x, 0)) * (wdt / 2 + 1.2) * (1 if (i // 3) % 2 else -1)
                    z = gz(q.x, q.y)
                    cp(_pole, loc=(q.x, q.y, z + 3.0))
                    cp(_head, loc=(q.x, q.y, z + 6.05))
        for p in P[::4]:
            keep_out.append((p.x, p.y, wdt))
        return o, P

    _pole = sb.prim("cyl", "LampPole", vertices=10, radius=0.09, depth=6.0, mat=MT["graphite"])
    _head = sb.prim("cube", "LampHead", scale=(0.35, 0.18, 0.06), mat=lampw)
    for o in (_pole, _head):
        o.hide_render = True; o.hide_viewport = True
    road("AvenueN", AVENUE_N + [PAD_A + V((40, -30))], wdt=10.0)
    road("AvenueE", AVENUE_E, wdt=9.0)
    road("AvenueS", AVENUE_S, wdt=8.0)
    road("PadRoadB", [V((-20, 420)), V((30, 520)), PAD_B + V((-30, -25))])
    road("PadRoadD", [PAD_B + V((30, -25)), V((200, 640)), PAD_D + V((-32, -12))])
    road("PadRoadE", [V((-20, 420)), V((-180, 560)), PAD_E + V((30, -25))])
    _, RPATH_E = road("ToIndustry", [V((330, 60)), V((360, 130)), V((330, 200))], wdt=7.0)
    road("FarRoad", [PAD_E + V((-20, 30)), V((-450, 1100)), V((-600, 1450)), PAD_C + V((20, -50))], wdt=6.0, lights=False)

    # ------------------------------------------------------------ hub
    hz = gz(0, 0)
    hub = sb.prim("sphere", "Hub", loc=(0, 0, hz - 3.0), segments=128, ring_count=64, radius=34.0, mat=MT["shield"])
    hub.scale = (1, 1, 0.5)
    sb.prim("cyl", "HubRing", loc=(0, 0, hz + 7.5), vertices=128, radius=31.0, depth=1.8, mat=MT["window"])
    rf = sb.prim("cyl", "HubRingF", loc=(0, 0, hz + 7.5), vertices=128, radius=31.1, depth=2.2, mat=MT["graphite"])
    sb.solidify(rf, 0.25)
    for k in range(4):                                   # entrance portals on the avenues
        a = k * math.pi / 2
        p = V((math.cos(a), math.sin(a), 0)) * 33.0
        pt = sb.prim("cube", "Portal", loc=(p.x, p.y, hz + 3.0), rot=(0, 0, a), scale=(3.0, 5.0, 3.0), mat=MT["white"])
        sb.bevel(pt, 0.4, 3)
        sb.prim("cube", "PortalWin", loc=(p.x + math.cos(a) * 3.02, p.y + math.sin(a) * 3.02, hz + 3.6), rot=(0, 0, a),
                scale=(0.02, 3.6, 1.2), mat=MT["window"])
    tower = sb.prim("cyl", "Tower", loc=(0, 0, hz + 24.0), vertices=24, radius=1.2, depth=40.0, mat=MT["white"])
    for zt in (30.0, 38.0):
        sb.prim("cyl", "TowerDeck", loc=(0, 0, hz + zt), vertices=32, radius=3.0, depth=0.6, mat=MT["graphite"])
    dish = sb.prim("sphere", "Dish", loc=(2.0, 0.0, hz + 41.0), segments=48, ring_count=24, radius=3.2, mat=MT["white"])
    dish.scale = (1, 1, 0.3); dish.rotation_euler = (math.radians(50), 0, math.radians(-40))
    sb.prim("sphere", "Beacon", loc=(0, 0, hz + 44.3), segments=16, ring_count=8, radius=0.45, mat=sb.emit_mat("BeaconM", (1.0, 0.18, 0.08), 60.0))
    keep_out.append((0, 0, 42))

    # ------------------------------------------------------------ habitats on two rings (avenues left clear)
    HABS = []
    for ring, (r0, n) in enumerate(((70.0, 10), (122.0, 14))):
        for k in range(n):
            a = 2 * math.pi * (k + 0.5 * ring) / n
            x, y = math.cos(a) * r0, math.sin(a) * r0
            if abs(x) < 16 and y > 0 or abs(y) < 16 and x > 0 or abs(x) < 14 and y < 0:
                continue                                  # the avenues
            if -150 < x < -40 and -60 < y < 70:
                continue                                  # the greenhouse vaults
            HABS.append((x, y, math.degrees(a)))
    for i, (hx, hy, hd) in enumerate(HABS):
        moon.habitat("Hab%d" % i, T, hx, hy, hd + 90.0, length=12.0 + (i % 3) * 1.5, mats=MT, seed=i + 3)
        keep_out.append((hx, hy, 14))
    # corridors: inner ring to hub (radial)
    for i, (hx, hy, hd) in enumerate(HABS):
        r = math.hypot(hx, hy)
        if r > 100:
            continue
        d = V((hx, hy)).normalized()
        a_, b_ = d * 33.0, d * (r - 5.0)
        c = (a_ + b_) / 2
        cor = sb.prim("cyl", "Corridor%d" % i, loc=(c.x, c.y, gz(c.x, c.y) + 1.4), vertices=32, radius=1.45, depth=(b_ - a_).length, mat=MT["white"])
        cor.rotation_euler = (math.radians(90), 0, math.atan2(d.y, d.x) + math.pi / 2)

    # ------------------------------------------------------------ greenhouses: glass vaults + domes, lit green
    green_m = sb.mat("Crops", (0.05, 0.16, 0.03), rough=0.6)
    ng = sb.NB(green_m)
    cog = ng.coord('Object')
    rows = ng.wave(cog, scale=0.35, wtype='BANDS', direction='X', dist=0.0, detail=0)
    leafy = ng.noise(cog, scale=4.0, detail=6)
    gcol = ng.mix(ng.maprange(leafy.outputs['Fac'], 0.35, 0.7), (0.03, 0.10, 0.02, 1), (0.14, 0.32, 0.05, 1))
    gcol = ng.mix(ng.maprange(rows.outputs['Fac'], 0.75, 0.9), gcol, (0.18, 0.12, 0.08, 1))
    ng.set('Base Color', gcol)
    ng.set('Emission Color', (0.35, 0.9, 0.25, 1)); ng.set('Emission Strength', 0.5)
    ng.set('Normal', ng.bump(leafy.outputs['Fac'], strength=0.6, distance=0.2))
    glow_m = sb.emit_mat("GrowGlow", (1.0, 0.72, 0.85), 4.0)
    for i, x in enumerate((-135.0, -112.0, -89.0, -66.0)):
        y0, L = 5.0, 110.0
        z = gz(x, y0)
        v = sb.prim("cyl", "Vault%d" % i, loc=(x, y0, z), rot=(math.radians(90), 0, 0), vertices=64, radius=8.0, depth=L, mat=MT["glass"])
        v.scale = (1, 0.85, 1)
        vf = sb.prim("cyl", "VaultFrame%d" % i, loc=(x, y0, z), rot=(math.radians(90), 0, 0), vertices=24, radius=8.05, depth=L, mat=MT["graphite"])
        vf.scale = (1, 0.85, 1)
        wf = vf.modifiers.new("W", 'WIREFRAME'); wf.thickness = 0.14
        sb.prim("cube", "VaultCrop%d" % i, loc=(x, y0, z + 0.5), scale=(7.2, L / 2 - 0.5, 0.1), mat=green_m)
        sb.prim("cube", "VaultLight%d" % i, loc=(x, y0, z + 5.2), scale=(0.25, L / 2 - 2, 0.05), mat=glow_m)
        for yy in np.linspace(y0 - L / 2 + 6, y0 + L / 2 - 6, 5):
            gl = sb.light('POINT', "VaultL", loc=(x, yy, z + 4.5), energy=6000.0, color=(1.0, 0.8, 0.9), size=3.0)
            gl.data.use_shadow = False
        for e in (-1, 1):
            cap = sb.prim("cyl", "VaultEnd", loc=(x, y0 + e * L / 2, z), rot=(math.radians(90), 0, 0), vertices=64, radius=8.1, depth=0.8, mat=MT["white"])
            cap.scale = (1, 0.85, 1)
        keep_out.append((x, y0, 60))
    DOMES = [(-60.0, -110.0, 20.0), (-115.0, -90.0, 14.0), (80.0, -95.0, 16.0)]
    for i, (dx, dy, dr) in enumerate(DOMES):
        z = gz(dx, dy)
        g = sb.prim("ico", "Dome%d" % i, loc=(dx, dy, z - dr * 0.12), subdivisions=3, radius=dr, mat=MT["glass"])
        g.scale = (1, 1, 0.78)
        fr = sb.prim("ico", "DomeFrame%d" % i, loc=(dx, dy, z - dr * 0.12), subdivisions=3, radius=dr * 1.004, mat=MT["graphite"])
        fr.scale = (1, 1, 0.78)
        wf = fr.modifiers.new("W", 'WIREFRAME'); wf.thickness = 0.16; wf.use_even_offset = True
        sb.prim("cyl", "DomeBase%d" % i, loc=(dx, dy, z + 0.3), vertices=96, radius=dr * 1.02, depth=1.4, mat=MT["white"])
        sb.prim("cyl", "Crop%d" % i, loc=(dx, dy, z + 1.05), vertices=64, radius=dr * 0.94, depth=0.1, mat=green_m)
        for k in range(-int(dr * 0.6), int(dr * 0.6) + 1, 3):
            sb.prim("cube", "GrowBar%d" % i, loc=(dx + k, dy, z + 4.2), scale=(0.08, dr * 0.7 * math.sqrt(max(0.05, 1 - (k / dr) ** 2)), 0.04), mat=glow_m)
        gl = sb.light('POINT', "DomeL%d" % i, loc=(dx, dy, z + 5.0), energy=2.2e4 * (dr / 15.0) ** 2, color=(1.0, 0.8, 0.9), size=dr * 0.5)
        gl.data.use_shadow = False
        keep_out.append((dx, dy, dr + 6))

    # ------------------------------------------------------------ industry: tank farm, spheres, ISRU plant, radiators
    tank = sb.prim("cyl", "TankProto", vertices=48, radius=4.5, depth=18.0, mat=tank_m)
    band = sb.prim("cyl", "TankBandProto", vertices=48, radius=4.56, depth=0.8, mat=MT["graphite"])
    for o in (tank, band):
        o.hide_render = True; o.hide_viewport = True
    for r in range(3):
        for c in range(4):
            x, y = 380.0 + c * 12.0, 40.0 + r * 12.0
            z = gz(x, y)
            cp(tank, loc=(x, y, z + 9.0))
            cp(band, loc=(x, y, z + 4.0)); cp(band, loc=(x, y, z + 14.0))
    for k in range(4):
        x, y = 395.0 + k * 17.0, 110.0
        z = gz(x, y)
        s = sb.prim("sphere", "Sphere%d" % k, loc=(x, y, z + 8.5), segments=48, ring_count=24, radius=7.0, mat=tank_m)
        for dx, dy in ((-4, -4), (4, -4), (-4, 4), (4, 4)):
            sb.prim("cyl", "SphLeg", loc=(x + dx, y + dy, z + 2.5), vertices=12, radius=0.35, depth=5.0, mat=MT["graphite"])
    z = gz(330, 150)
    plant = sb.prim("cube", "ISRU", loc=(330.0, 150.0, z + 7.0), scale=(22.0, 13.0, 7.0), mat=MT["white"])
    sb.bevel(plant, 0.6, 3)
    for k in range(6):
        sb.prim("cube", "ISRUWin", loc=(330.0 - 18 + k * 7.2, 136.95, z + 10.0), scale=(2.2, 0.05, 0.6), mat=MT["window"])
    for k in range(3):
        sb.prim("cyl", "Stack", loc=(318.0 + k * 10, 155.0, z + 20.0), vertices=24, radius=1.2, depth=12.0, mat=MT["graphite"])
    for yy in (95.0, 97.5):
        pr = sb.prim("cyl", "PipeRack", loc=(360.0, yy, z + 6.0), rot=(0, math.radians(90), 0), vertices=16, radius=0.5, depth=70.0, mat=rad_m)
    for k in range(8):
        sb.prim("cube", "RackLeg", loc=(328.0 + k * 9, 96.25, z + 3.0), scale=(0.3, 2.0, 3.0), mat=MT["graphite"])
    zr = gz(470, 180)
    sb.prim("cube", "RadTower", loc=(470.0, 180.0, zr + 20.0), scale=(2.0, 2.0, 20.0), mat=MT["graphite"])
    for k in range(7):
        fin = sb.prim("cube", "RadFin", loc=(470.0, 180.0, zr + 8.0 + k * 4.6), scale=(16.0, 0.1, 1.9), mat=rad_m)
    keep_out += [(410, 70, 45), (330, 150, 30), (420, 110, 40), (470, 180, 20)]

    # ------------------------------------------------------------ solar farm (south)
    cells = MT["cells"]
    proto = sb.prim("cube", "SolarProto", scale=(2.2, 1.05, 0.03), mat=cells)
    leg = sb.prim("cyl", "SolarLegProto", vertices=8, radius=0.06, depth=1.3, mat=MT["graphite"])
    for o in (proto, leg):
        o.hide_render = True; o.hide_viewport = True
    step = 2 if detail < 1 else 1
    for r in range(0, 22, step):
        for c in range(0, 50, step):
            x = -130.0 + c * 5.0 + (r % 2) * 0.4
            y = -440.0 + r * 7.5
            if abs(x) < 12.0:
                continue                                  # the avenue's service lane
            z = gz(x, y)
            cp(proto, loc=(x, y, z + 1.35), rot=(0, math.radians(24), 0))       # tilted toward the east sun
            cp(leg, loc=(x - 0.2, y, z + 0.65))
    keep_out.append((0, -360, 150))

    # ------------------------------------------------------------ spaceport: pads, berms, launch tower
    pad_m = MT["concrete"]
    lamp_m = sb.emit_mat("PadLamp", (1.0, 0.85, 0.6), 30.0)
    paint = sb.mat("PadPaint", (0.55, 0.50, 0.44), rough=0.7)
    for i, P in enumerate((PAD_A, PAD_B, PAD_D, PAD_E, PAD_C)):
        rp = 32.0
        z = gz(P.x, P.y)
        sb.prim("cyl", "Pad%d" % i, loc=(P.x, P.y, z + 0.25), vertices=128, radius=rp, depth=0.6, mat=pad_m)
        sb.prim("cyl", "PadMark%d" % i, loc=(P.x, P.y, z + 0.56), vertices=128, radius=rp * 0.62, depth=0.02, mat=paint)
        sb.prim("cyl", "PadMarkIn%d" % i, loc=(P.x, P.y, z + 0.575), vertices=128, radius=rp * 0.58, depth=0.02, mat=pad_m)
        for k in range(24):
            a = 2 * math.pi * k / 24
            sb.prim("cube", "PadLamp%d" % i, loc=(P.x + math.cos(a) * (rp - 0.8), P.y + math.sin(a) * (rp - 0.8), z + 0.62),
                    scale=(0.25, 0.25, 0.06), mat=lamp_m)
        bm = sb.prim("torus", "Berm%d" % i, loc=(P.x, P.y, z - 1.4), major_radius=rp + 16.0, minor_radius=4.5, major_segments=128,
                     minor_segments=16, mat=MT["shield"])
        bm.scale = (1, 1, 0.8)
        keep_out.append((P.x, P.y, rp + 24))
    # lattice launch tower beside pad D, with two catch arms
    tx, ty = PAD_D.x + 44.0, PAD_D.y + 10.0
    tz = gz(tx, ty)
    Htw, Wd = 92.0, 4.5
    rodp = sb.prim("cyl", "TwRodProto", vertices=8, radius=0.22, depth=1.0, mat=MT["graphite"])
    rodp.hide_render = True; rodp.hide_viewport = True
    corners = [V((tx + sx * Wd, ty + sy * Wd, tz)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for c in corners:
        o = sb.prim("cube", "TwLeg", loc=(c.x, c.y, tz + Htw / 2), scale=(0.45, 0.45, Htw / 2), mat=MT["graphite"])

    def strut(a, b):
        o = cp(rodp, loc=tuple((a + b) / 2))
        d = b - a
        o.scale = (1, 1, d.length)
        o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = d.to_track_quat('Z', 'Y')
    for lv in range(0, int(Htw / 6.0)):
        z0, z1 = tz + lv * 6.0, tz + (lv + 1) * 6.0
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            strut(V((a.x, a.y, z1)), V((b.x, b.y, z1)))
            strut(V((a.x, a.y, z0)), V((b.x, b.y, z1)) if lv % 2 else V((b.x, b.y, z0)))
            if lv % 2 == 0:
                strut(V((b.x, b.y, z0)), V((a.x, a.y, z1)))
    for k, ang in enumerate((-18.0, 18.0)):
        arm = sb.prim("cube", "CatchArm", loc=(tx - 16.0, ty + (k * 2 - 1) * 6.0, tz + 66.0), scale=(16.0, 0.8, 1.6), mat=MT["graphite"])
        arm.rotation_euler = (0, 0, math.radians(ang))
    sb.prim("sphere", "TwBeacon", loc=(tx, ty, tz + Htw + 0.6), segments=16, ring_count=8, radius=0.5, mat=sb.emit_mat("TwBeaconM", (1.0, 0.2, 0.1), 60.0))
    keep_out.append((tx, ty, 20))

    # ------------------------------------------------------------ ships
    SM = ship.materials("SH")
    # A: lands on pad A (the s24c hero)
    A = ship.build("ShipA", M=SM, legs_deployed=0.0)
    A["root"].rotation_euler = (0, 0, math.radians(35))
    zA = gz(PAD_A.x, PAD_A.y) + 0.55 - ship.GROUND_Z
    FTD = EV_TD

    def posA(f):
        if f >= FTD:
            return V((PAD_A.x, PAD_A.y, zA))
        u = min(1.0, (FTD - f) / 115.0)
        return V((PAD_A.x + 16.0 * u ** 2, PAD_A.y + 10.0 * u ** 2, zA + 330.0 * u ** 1.8))

    def thrA(f):
        return 1.0 if f < FTD + 2 else max(0.0, 1.0 - (f - FTD - 2) / 8.0)

    def tiltA(f):
        u = max(0.0, min(1.0, (FTD - f) / 115.0))
        return (math.radians(2.5) * u, math.radians(3.0) * u * math.sin(f * 0.05))

    ship.fly(A, F0, F1, posA, thrA, tilt_fn=tiltA, legs_fn=lambda f: sb.smoother(sb.remap(f, FTD - 87, FTD - 37)), light_max=4e6)
    if F0 < FTD + 40:
        ship.dust_ring("DustA", (PAD_A.x, PAD_A.y, gz(PAD_A.x, PAD_A.y)), FTD - 40, FTD + 6, radius_max=95.0, n=34,
                       color=(0.62, 0.40, 0.24), density=0.18, seed=3, height=8.0)
    # B: standing on pad B (lifts off in s25z)
    B = ship.build("ShipB", M=SM, legs_deployed=1.0)
    zB = gz(PAD_B.x, PAD_B.y) + 0.55 - ship.GROUND_Z
    B["root"].location = (PAD_B.x, PAD_B.y, zB)
    B["root"].rotation_euler = (0, 0, math.radians(-60))

    def posB(f):
        t = max(0.0, (f - EV_LO_B) / 24.0)
        return V((PAD_B.x + 0.8 * t * t, PAD_B.y, zB + 6.0 * t * t))
    if F1 > EV_LO_B - 30:
        ship.fly(B, F0, F1, posB, lambda f: sb.smoother(sb.remap(f, EV_LO_B - 18, EV_LO_B)),
                 legs_fn=lambda f: 1.0 - sb.smoother(sb.remap(f, EV_LO_B + 40, EV_LO_B + 80)), light_max=6e6)
        for k in range(F0, F1 + 1):
            th = sb.smoother(sb.remap(k, EV_LO_B - 18, EV_LO_B))
            B["plume"].scale = (1.5, 1.5, 0.6 + 2.2 * th); B["plume"].keyframe_insert("scale", frame=k)
        ship.dust_ring("DustB", (PAD_B.x, PAD_B.y, gz(PAD_B.x, PAD_B.y)), EV_LO_B - 10, F1, radius_max=180.0, n=30,
                       color=(0.60, 0.40, 0.25), density=0.14, seed=9, height=14.0, rise=1.2)
    # D, E: standing ships
    for nm, P, rz in (("ShipD", PAD_D, 20.0), ("ShipE", PAD_E, 140.0)):
        S_ = ship.build(nm, M=SM, legs_deployed=1.0)
        S_["root"].location = (P.x, P.y, gz(P.x, P.y) + 0.55 - ship.GROUND_Z)
        S_["root"].rotation_euler = (0, 0, math.radians(rz))
    # C: leaving from the far pad
    Cs = ship.build("ShipC", M=SM, legs_deployed=1.0)
    zC = gz(PAD_C.x, PAD_C.y) + 0.55 - ship.GROUND_Z
    FLO = EV_LO_C

    def posC(f):
        t = max(0.0, (f - FLO) / 24.0)
        return V((PAD_C.x - 1.2 * t * t, PAD_C.y, zC + 5.5 * t * t))

    ship.fly(Cs, F0, F1, posC, lambda f: sb.smoother(sb.remap(f, FLO - 20, FLO)),
             legs_fn=lambda f: 1.0 - sb.smoother(sb.remap(f, FLO + 40, FLO + 80)), light_max=6e6)
    for k in range(F0, F1 + 1):
        th = sb.smoother(sb.remap(k, FLO - 20, FLO))
        Cs["plume"].scale = (1.5, 1.5, 0.6 + 2.2 * th); Cs["plume"].keyframe_insert("scale", frame=k)
    ship.dust_ring("DustC", (PAD_C.x, PAD_C.y, gz(PAD_C.x, PAD_C.y)), FLO - 12, F1, radius_max=210.0, n=30,
                   color=(0.60, 0.40, 0.25), density=0.12, seed=5, height=16.0, rise=1.2)

    # ------------------------------------------------------------ rovers on the avenues (global-frame timing)
    def rover(name):
        root = sb.empty(name)
        body = sb.prim("cube", name + "Body", loc=(0, 0, 1.55), scale=(2.6, 1.3, 0.9), mat=MT["white"], parent=root)
        sb.bevel(body, 0.45, 4)
        sb.prim("cube", name + "Win", loc=(2.25, 0, 1.9), scale=(0.35, 1.0, 0.3), mat=MT["window"], parent=root)
        sb.prim("cube", name + "Rack", loc=(-0.6, 0, 2.55), scale=(1.4, 1.0, 0.08), mat=MT["graphite"], parent=root)
        for dx in (-1.7, 0.0, 1.7):
            for dy in (-1.35, 1.35):
                sb.prim("cyl", name + "Wheel", loc=(dx, dy, 0.55), rot=(math.radians(90), 0, 0), vertices=32, radius=0.55, depth=0.45,
                        mat=MT["graphite"], parent=root)
        return root

    def dense(pts, n=200):
        return [sb.catmull([V((p.x, p.y, 0)) for p in pts], i / (n - 1)) for i in range(n)]
    routes = [dense(AVENUE_N + [PAD_A + V((40, -30))]), dense(AVENUE_E), dense(AVENUE_S)]
    for j, (ri, spd, off, side) in enumerate(((0, 5.5, 0.1, 1), (0, 4.8, 0.6, -1), (1, 5.0, 0.3, 1), (1, 4.2, 0.8, -1),
                                             (2, 4.5, 0.5, 1), (0, 6.0, 0.35, 1))):
        path = routes[ri] if side > 0 else list(reversed(routes[ri]))
        rv = rover("Rover%d" % j)
        L = sum((path[i + 1] - path[i]).length for i in range(len(path) - 1))
        for f in range(F0, F1 + 1, 2):
            s_ = (off * L + spd * f / 24.0) % L
            acc = 0.0
            for i in range(len(path) - 1):
                seg = (path[i + 1] - path[i]).length
                if acc + seg >= s_:
                    p = path[i].lerp(path[i + 1], (s_ - acc) / max(seg, 1e-6))
                    d = path[i + 1] - path[i]
                    break
                acc += seg
            sv = V((-d.y, d.x, 0)).normalized() * 2.2
            rv.location = (p.x + sv.x, p.y + sv.y, gz(p.x, p.y) + 0.07)
            rv.rotation_euler = (0, 0, math.atan2(d.y, d.x))
            rv.keyframe_insert("location", frame=f); rv.keyframe_insert("rotation_euler", frame=f)

    # ------------------------------------------------------------ boulders (kept off everything built)
    rng = np.random.default_rng(7)
    ko = np.array(keep_out)
    pts, szs = [], []
    for _ in range(int(4000 * detail)):
        r_ = 30.0 + 900.0 * rng.uniform() ** 1.5
        a_ = rng.uniform(0, 2 * math.pi)
        x, y = r_ * math.cos(a_), r_ * math.sin(a_)
        if np.any(np.hypot(ko[:, 0] - x, ko[:, 1] - y) < ko[:, 2]):
            continue
        pts.append((x, y)); szs.append(0.08 + rng.pareto(2.2) * 0.22)
    rk_m = sb.mat("MarsRock", (0.20, 0.10, 0.055), rough=0.8)
    moon.rocks(T, "Rocks", np.array(pts), np.minimum(np.array(szs), 1.8), rk_m, seed=5)

    return dict(T=T, gz=gz, A=A, B=B, C=Cs, PAD_A=PAD_A, PAD_B=PAD_B, PAD_C=PAD_C, PAD_D=PAD_D, PAD_E=PAD_E, posA=posA,
                posB=posB, posC=posC, smix=smix, world=w, ground=ground, SUN=SUN, HABS=HABS, DOMES=DOMES)
