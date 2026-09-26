"""Colony ship (Moon / Mars): the film's vehicle family scaled up into a reusable lander. White body with a graphite
heat-shield belly, four graphite flaps, the amber band, a band of warm crew windows, six sea-level engines, six
landing legs that deploy. Origin at the base centre (bottom of the engine skirt); +Z up. ~9 m wide, ~48 m tall.

build(name) -> dict(root, legs, plume, gain, gain0, light, M). Landed: root.z = ground - GROUND_Z.
fly(sh, f0, f1, path_fn, thrust_fn, legs_fn) keys position/attitude/plume per frame.
dust_ring(...) radial ground dust (volume puffs) for touchdowns and liftoffs."""
import bpy, math, random
from mathutils import Vector as V, Euler, Matrix
import sb, rocket, props

R = 4.5
H_BODY = 40.0
NOSE_L = 8.0
LEG_Z, LEG_L = 6.0, 7.6
GROUND_Z = LEG_Z - (LEG_L + 0.19) * math.cos(math.radians(34.0))   # footpad bottom (root z) when deployed: ~-0.46


def materials(prefix="SH"):
    M = {}
    M["white"] = rocket.paint_white(prefix + "White", base=(0.80, 0.80, 0.79), pitch=3.0, nseg=12, radius=R, seed=7.0)
    M["tile"] = rocket.composite(prefix + "Tile", color=(0.03, 0.032, 0.036), stringers=0, radius=R, sheen=0.1, wear=0.3)
    M["graph"] = sb.painted(prefix + "Graph", (0.05, 0.052, 0.058), rough=0.4, grime=0.35)
    M["amber"] = props.amber_anodized(prefix + "Amber")
    M["steel"] = rocket.steel_mat(prefix + "Steel")
    M["win"] = sb.emit_mat(prefix + "Win", (1.0, 0.74, 0.44), 6.0)
    M["pad"] = sb.painted(prefix + "Pad", (0.18, 0.18, 0.19), rough=0.6, grime=0.6)
    E = rocket._materials(frost=0.0, prefix=prefix + "E")
    M["noz_out"], M["noz_in"], M["inconel"], M["pump"], M["duct"] = E["noz_out"], E["noz_in"], E["inconel"], E["pump"], E["duct"]
    M["_E"] = E
    return M


def _split_belly(o, tile):
    """heat-shield tiles on the -Y half (the entry side): second material slot on faces facing -Y."""
    me = o.data
    me.materials.append(tile)
    for p in me.polygons:
        if p.center.y < -0.35 * R:
            p.material_index = 1


def build(name="Ship", M=None, legs_deployed=1.0, belly=True):
    M = M or materials()
    root = sb.empty(name)
    z0 = 3.4                                           # engine skirt height
    # aft skirt (graphite), body, amber band, nose
    rocket.lathe(name + "Skirt", [(R * 0.93, 0.0), (R, 0.6), (R, z0)], mat=M["graph"], parent=root)
    body = rocket.lathe(name + "Body", [(R, z0 + (H_BODY - z0) * i / 12) for i in range(13)], mat=M["white"], parent=root)
    rocket.lathe(name + "Band", [(R + 0.01, H_BODY - 5.2), (R + 0.03, H_BODY - 5.15), (R + 0.03, H_BODY - 4.65),
                                 (R + 0.01, H_BODY - 4.6)], mat=M["amber"], parent=root)
    prof = []
    for i in range(25):
        t = i / 24
        prof.append((R * math.cos(t * math.pi / 2) ** 0.85 + 0.0001, H_BODY + NOSE_L * math.sin(t * math.pi / 2)))
    nose = rocket.lathe(name + "Nose", prof, mat=M["white"], parent=root)
    if belly:
        _split_belly(body, M["tile"]); _split_belly(nose, M["tile"])
    # crew window band (+Y side, just under the nose)
    for k in range(9):
        a = math.radians(60 + k * 7)
        p = V((math.cos(a) * (R + 0.02), math.sin(a) * (R + 0.02), H_BODY - 2.3))
        w = sb.prim("cube", name + "Win", loc=p, rot=(0, 0, a), scale=(0.05, 0.28, 0.42), mat=M["win"], parent=root)
    # four flaps (two fore near the nose, two aft), graphite, hinged on the ±X sides
    for sx in (-1, 1):
        for (zc, L, W) in ((H_BODY + 2.0, 6.0, 2.2), (9.0, 10.0, 3.4)):
            f = sb.prim("cube", name + "Flap", loc=(sx * (R + W / 2 - 0.3), 0, zc), scale=(W / 2, 0.18, L / 2), mat=M["graph"], parent=root)
            sb.bevel(f, 0.12, 3)
    # engines: ring of six under the skirt
    E = M["_E"]
    for k in range(6):
        a = math.radians(30 + 60 * k)
        rocket.engine(name + "Eng%d" % k, E, root, (math.cos(a) * 2.4, math.sin(a) * 2.4, 3.3), az_face=math.degrees(a))
    # six landing legs: hinge just outside the skirt; fold up flush along the hull, swing out and down to deploy
    legs = []
    for k in range(6):
        a = math.radians(60 * k)
        piv = sb.empty(name + "LegPiv%d" % k, loc=(math.cos(a) * (R + 0.4), math.sin(a) * (R + 0.4), LEG_Z), parent=root)
        piv.rotation_euler = (0, 0, a)
        rocket.rod(name + "Leg%d" % k, (0, 0, 0), (0, 0, -LEG_L), 0.28, M["graph"], parent=piv)
        rocket.rod(name + "LegStrut%d" % k, (0, 0, -LEG_L * 0.55), (0, 0, -LEG_L * 0.2), 0.16, M["steel"], parent=piv)
        fp = sb.empty(name + "FootPiv%d" % k, loc=(0, 0, -LEG_L), parent=piv)
        sb.prim("cyl", name + "Foot%d" % k, loc=(0, 0, -0.06), vertices=24, radius=0.9, depth=0.25, mat=M["pad"], parent=fp)
        legs.append(piv)
    set_legs(legs, legs_deployed)
    # plume: engine cluster exhaust (additive shell) + warm light
    pm = rocket.plume_material(name + "PlumeM", intensity=1.0)
    plume = rocket.plume_mesh(name + "Plume", length=26.0, r0=2.6, r_max=9.0, mat=pm, parent=root)
    plume.location = (0, 0, 0.1)
    gain = pm.node_tree.nodes.get("Gain")
    root["gain0"] = gain.inputs[1].default_value
    gain.inputs[1].default_value = 0.0
    light = sb.light('POINT', name + "PlumeLight", loc=(0, 0, -3.0), energy=0.0, color=(1.0, 0.6, 0.3), size=3.0)
    light.parent = root
    return dict(root=root, legs=legs, plume=plume, gain=gain, gain0=root["gain0"], light=light, M=M)


def set_legs(legs, k, frame=None):
    """k: 0 stowed (folded up flush along the hull) .. 1 deployed (splayed 34 deg); feet stay level."""
    k = max(0.0, min(1.0, k))
    th = math.radians(-172.0 + (172.0 - 34.0) * sb.smooth(k))
    for piv in legs:
        piv.rotation_mode = 'XYZ'
        piv.rotation_euler = (0.0, th, piv.rotation_euler.z)
        fp = next((c for c in piv.children if "FootPiv" in c.name), None)
        if fp:
            fp.rotation_euler = (0.0, -th, 0.0)
        if frame is not None:
            piv.keyframe_insert("rotation_euler", frame=frame)
            if fp:
                fp.keyframe_insert("rotation_euler", frame=frame)


def fly(sh, f0, f1, pos_fn, thrust_fn, tilt_fn=None, legs_fn=None, light_max=6e5):
    """pos_fn(f) -> Vector base position; thrust_fn(f) -> 0..1 plume; tilt_fn(f) -> (rx, ry) radians."""
    root = sh["root"]
    root.rotation_mode = 'XYZ'
    for f in range(f0, f1 + 1):
        root.location = pos_fn(f)
        rx, ry = tilt_fn(f) if tilt_fn else (0.0, 0.0)
        root.rotation_euler = (rx, ry, root.rotation_euler.z)
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)
        th = max(0.0, min(1.0, thrust_fn(f)))
        fl = th * (0.92 + 0.08 * math.sin(f * 2.7) * math.sin(f * 1.3))
        if sh["gain"] is not None:
            sh["gain"].inputs[1].default_value = fl * sh["gain0"]
            sh["gain"].inputs[1].keyframe_insert("default_value", frame=f)
        sh["plume"].scale = (1.0, 1.0, 0.35 + 0.65 * th)
        sh["plume"].keyframe_insert("scale", frame=f)
        sh["light"].data.energy = light_max * fl
        sh["light"].data.keyframe_insert("energy", frame=f)
        if legs_fn:
            set_legs(sh["legs"], legs_fn(f), frame=f)


def dust_ring(name, center, f_on, f_off, radius_max=60.0, n=26, color=(0.55, 0.38, 0.24), density=0.25, seed=1,
              height=6.0, rise=0.3):
    """Radial ground dust thrown out by the plume: soft volume puffs racing outward and settling. Returns objects."""
    rs = random.Random(seed)
    vm = sb.volume_mat(name + "M", density=density, color=color, anisotropy=0.3)
    objs = []
    for i in range(n):
        a = 2 * math.pi * i / n + rs.uniform(-0.1, 0.1)
        o = sb.prim("sphere", name + "%d" % i, loc=center, segments=16, ring_count=8, radius=1.0, mat=vm)
        o.visible_shadow = False
        spd = rs.uniform(0.7, 1.2)
        for f in range(f_on - 1, f_off + 30):
            t = max(0.0, (f - f_on) / 24.0)
            act = 1.0 if f < f_off else max(0.0, 1.0 - (f - f_off) / 30.0)
            r = radius_max * (1 - math.exp(-t * 1.4 * spd))
            s = (2.0 + r * 0.22) * (0.2 if f < f_on else 1.0) * act + 0.001
            o.location = V(center) + V((math.cos(a) * r, math.sin(a) * r, height * 0.4 + r * rise * 0.1))
            o.scale = (s, s, s * 0.45)
            o.keyframe_insert("location", frame=f); o.keyframe_insert("scale", frame=f)
        objs.append(o)
    return objs
