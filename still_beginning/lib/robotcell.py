"""s07b robot cell: pedestal-mounted arm, build-plate station (raw components lying as built), fixture with amber
locating pins; pick (grip across the MCP boss) -> lift -> transfer -> seat the bores onto the pins -> release."""
import bpy, math, os
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler
import sb, robot as RB, prosthesis as P, props

F0, F1 = 574, 630
TABLE_Z = 0.78
PLATE_C = V((0.64, 0.40, TABLE_Z + 0.022))
FIX_C = V((0.60, -0.44, TABLE_Z))
PARTS = [(-32, 28), (0, 28), (32, 28), (-32, -28), (0, -28), (32, -28)]
HERO = 4
R_PART = Matrix(((0, 0, -1), (0, 1, 0), (1, 0, 0)))    # columns: part x -> up, part y -> +Y, part z -> -X


def part_world(center_xy_mm, base):
    cx, cy = center_xy_mm
    M = R_PART.to_4x4()
    M.translation = base + V((cx * 0.001, (cy - 20) * 0.001, 0.007))
    return M


def tooling_plate(name="ToolPlate", pitch=0.05):
    """black anodized aluminium tooling plate: blanchard-ground swirl + tapped hole grid."""
    m = sb.mat(name, (0.035, 0.036, 0.04), metal=0.9, rough=0.32)
    nb = sb.NB(m)
    co = nb.coord('Object')
    geo = nb.new('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(pos, sep.inputs[0])
    fx = nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[0], pitch)), 0.5)
    fy = nb.math('SUBTRACT', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[1], pitch)), 0.5)
    r = nb.math('SQRT', nb.math('ADD', nb.math('MULTIPLY', fx, fx), nb.math('MULTIPLY', fy, fy)))
    hole = nb.maprange(r, 0.075, 0.065)
    ring = nb.math('MULTIPLY', nb.maprange(r, 0.09, 0.08), nb.maprange(r, 0.075, 0.085))
    # swirl (overlapping arcs of a fly cutter)
    sw = nb.wave(nb.mapping(pos, scale=(1.0, 1.0, 1.0)), scale=60, wtype='RINGS', direction='Z', dist=3.0, detail=3)
    scr = nb.noise(nb.mapping(pos, scale=(1, 30, 1)), scale=60, detail=3)
    h = nb.math('ADD', nb.math('MULTIPLY', sw.outputs['Fac'], 0.3), nb.math('MULTIPLY', scr.outputs['Fac'], 0.2))
    h = nb.math('SUBTRACT', h, nb.math('MULTIPLY', hole, 2.0))
    nb.set('Normal', nb.bump(h, strength=0.25, distance=0.0008))
    col = nb.mix(hole, (0.035, 0.036, 0.04, 1), (0.003, 0.003, 0.004, 1))
    col = nb.mix(nb.math('MULTIPLY', ring, 0.8), col, (0.22, 0.22, 0.23, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(ring, nb.maprange(sw.outputs['Fac'], 0.2, 0.8, 0.26, 0.4), 0.2, dtype='FLOAT'))
    return m


def extrusion_table(c, w=0.52, d=0.46, top_z=TABLE_Z, name="Table"):
    alu = sb.brushed_metal(name + "Ext", (0.66, 0.67, 0.69), rough=0.3, aniso=0.6, scale=300)
    tp = sb.prim("cube", name + "Top", loc=(c.x, c.y, top_z - 0.012), scale=(w / 2, d / 2, 0.012), mat=tooling_plate(name + "TP"))
    sb.bevel(tp, 0.0015, 2)
    groove = sb.mat(name + "Groove", (0.02, 0.02, 0.022), rough=0.6)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lg = sb.prim("cube", name + "Leg", loc=(c.x + sx * (w / 2 - 0.03), c.y + sy * (d / 2 - 0.03), (top_z - 0.024) / 2),
                         scale=(0.02, 0.02, (top_z - 0.024) / 2), mat=alu)
            sb.bevel(lg, 0.002)
            for gx, gy in ((0.0202, 0), (-0.0202, 0), (0, 0.0202), (0, -0.0202)):
                sb.prim("cube", name + "Slot", loc=(lg.location.x + gx, lg.location.y + gy, lg.location.z), scale=(0.003 if gx == 0 else 0.0006, 0.003 if gy == 0 else 0.0006, (top_z - 0.06) / 2), mat=groove)
            foot = sb.prim("cyl", name + "Foot", loc=(lg.location.x, lg.location.y, 0.012), vertices=24, radius=0.028, depth=0.024, mat=sb.rubber(name + "FootR"))
    for z in (0.12, top_z - 0.05):
        for sy in (-1, 1):
            sb.prim("cube", name + "Rail", loc=(c.x, c.y + sy * (d / 2 - 0.03), z), scale=(w / 2 - 0.05, 0.02, 0.02), mat=alu)
        for sx in (-1, 1):
            sb.prim("cube", name + "RailS", loc=(c.x + sx * (w / 2 - 0.03), c.y, z), scale=(0.02, d / 2 - 0.05, 0.02), mat=alu)


def env():
    floor = sb.prim("plane", "Floor", scale=(8, 8, 1), mat=sb.concrete("Epoxy", (0.24, 0.25, 0.26), scale=0.8, rough=0.3, stains=0.2))
    lm = sb.mat("FloorLine", (0.70, 0.42, 0.06), rough=0.55)
    for (loc, sc_) in (((0.3, 1.35, 0.001), (1.6, 0.03, 1)), ((0.3, -1.35, 0.001), (1.6, 0.03, 1)), ((-1.3, 0.0, 0.001), (0.03, 1.35, 1))):
        sb.prim("plane", "Line", loc=loc, scale=sc_, mat=lm)
    graph = sb.painted("CellGraphite", (0.05, 0.052, 0.058), rough=0.4)
    white = sb.painted("CellWhite", (0.74, 0.74, 0.73), rough=0.35, coat=0.3, grime=0.1)
    for c in (PLATE_C, FIX_C):
        extrusion_table(c, name="Tbl%d" % int(c.y > 0))
    ped = sb.prim("cube", "Pedestal", loc=(0, 0, 0.15), scale=(0.24, 0.24, 0.15), mat=graph)
    sb.bevel(ped, 0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            sb.prim("cyl", "Anchor", loc=(sx * 0.2, sy * 0.2, 0.302), vertices=6, radius=0.012, depth=0.008, mat=sb.brushed_metal("Bolt", (0.6, 0.6, 0.62)))
    # safety fence: black steel mesh panels with posts (alpha mesh)
    mesh_m = sb.mat("FenceMesh", (0.03, 0.03, 0.035), metal=0.5, rough=0.5)
    nb = sb.NB(mesh_m)
    co = nb.coord('Object')
    sepx = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sepx.inputs[0])
    fx = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('MULTIPLY', sepx.outputs[0], 22.0)), 0.5))
    fz = nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('FRACT', nb.math('MULTIPLY', sepx.outputs[1], 22.0)), 0.5))
    wire = nb.math('MAXIMUM', nb.maprange(fx, 0.44, 0.47), nb.maprange(fz, 0.44, 0.47))
    nb.set('Alpha', wire)
    mesh_m.surface_render_method = 'DITHERED'
    for i in range(5):
        x = -1.25 + i * 0.9
        fr = sb.prim("plane", "Fence", loc=(x, 1.9, 1.05), rot=(math.pi / 2, 0, 0), scale=(0.43, 0.95, 1), mat=mesh_m)
        post = sb.prim("cube", "Post", loc=(x - 0.45, 1.9, 1.0), scale=(0.025, 0.025, 1.0), mat=graph)
        for zz in (0.1, 2.0):
            sb.prim("cube", "FenceRail", loc=(x, 1.9, zz), scale=(0.43, 0.012, 0.012), mat=graph)
    # the SLM machine: white/graphite cabinet, recessed window with the build chamber glow, control panel
    cab = sb.prim("cube", "SLMCabinet", loc=(-1.55, 0.25, 1.05), scale=(0.55, 0.8, 1.05), mat=white)
    sb.bevel(cab, 0.04, 4)
    base = sb.prim("cube", "SLMBase", loc=(-1.55, 0.25, 0.08), scale=(0.56, 0.81, 0.08), mat=graph)
    door = sb.prim("cube", "SLMDoor", loc=(-0.995, 0.25, 1.1), scale=(0.01, 0.42, 0.42), mat=graph)
    sb.bevel(door, 0.02, 3)
    win = sb.prim("cyl", "SLMWindow", loc=(-0.982, 0.25, 1.12), rot=(0, math.pi / 2, 0), vertices=64, radius=0.16, depth=0.01,
                  mat=sb.emit_mat("SLMWin", (0.55, 0.62, 0.75), 0.8))
    wr = sb.lathe("SLMWinRing", [(0.160, 0.0), (0.172, 0.0), (0.172, 0.01), (0.160, 0.01)], segs=96, mat=props.amber_anodized("SLMAmber"), axis='X')
    wr.location = (-0.99, 0.25, 1.12)
    panel = sb.prim("cube", "SLMPanel", loc=(-0.99, -0.35, 1.3), scale=(0.012, 0.12, 0.09), mat=sb.emit_mat("SLMPanelM", (0.18, 0.2, 0.24), 0.6))
    # overhead: cable tray + high-bay linear lights (visible, soft)
    tray = sb.prim("cube", "CableTray", loc=(0.2, 1.4, 2.6), scale=(2.5, 0.12, 0.03), mat=sb.brushed_metal("Tray", (0.55, 0.56, 0.58), rough=0.4))
    for x in (-1.2, 0.4, 2.0):
        sb.prim("cube", "HighBay", loc=(x, 0.2, 3.3), scale=(0.6, 0.08, 0.02), mat=sb.emit_mat("HB", (0.95, 0.97, 1.0), 12.0))
    # parts bins + a rolling cart in the far background
    for i in range(4):
        b = sb.prim("cube", "Bin", loc=(1.3 + i * 0.28, 1.6, 0.9), scale=(0.12, 0.18, 0.08), mat=sb.painted("BinM", (0.08, 0.085, 0.095), rough=0.6))
        sb.bevel(b, 0.01)
    sb.prim("cube", "Shelf", loc=(1.72, 1.6, 0.81), scale=(0.6, 0.22, 0.01), mat=sb.brushed_metal("ShelfM", (0.6, 0.6, 0.62)))
    # build plate with the raw parts (depowdered)
    plate = sb.prim("cube", "BuildPlate", loc=(PLATE_C.x, PLATE_C.y, PLATE_C.z - 0.011), scale=(0.06, 0.06, 0.011), mat=sb.brushed_metal("PlateM", (0.58, 0.57, 0.55), rough=0.38, scale=800, scratches=0.3))
    sb.bevel(plate, 0.0015)
    raw = P.ti_sintered("TiRaw", build=True)
    parts = []
    for i, c in enumerate(PARTS):
        r, o = P.component("v3", "Raw%d" % i, bushings=False, mat=raw)
        r.matrix_world = part_world(c, PLATE_C)
        parts.append(r)
    # fixture: machined aluminium block (black anodized) on a riser, chamfered pocket, amber locating pins
    fxm = sb.brushed_metal("FixM", (0.06, 0.062, 0.068), rough=0.3, scale=500)
    fx = sb.prim("cube", "Fixture", loc=(FIX_C.x, FIX_C.y + 0.02, FIX_C.z + 0.018), scale=(0.04, 0.05, 0.018), mat=fxm)
    sb.bevel(fx, 0.002, 2)
    for sx in (-1, 1):
        for sy in (-1, 1):
            sb.prim("cyl", "FixBolt", loc=(FIX_C.x + sx * 0.03, FIX_C.y + 0.02 + sy * 0.04, FIX_C.z + 0.0362), vertices=24, radius=0.004, depth=0.001,
                    mat=sb.mat("CBore", (0.01, 0.01, 0.012), rough=0.6))
    top = FIX_C.z + 0.036
    fix_part = part_world((0, 20), V((FIX_C.x, FIX_C.y, top - 0.0)))
    amber = props.amber_anodized("PinAmber")
    for yy in (0.0, 0.040):
        pw = fix_part @ V((0, yy, 0))
        pin = sb.lathe("Pin", [(0.0, 0.0), (0.00195, 0.0), (0.00195, 0.017), (0.0016, 0.0185), (0.0, 0.019)], segs=48, mat=amber)
        pin.location = (pw.x, pw.y, top - 0.004)
        coll = sb.lathe("PinColl", [(0.0, 0.0), (0.0042, 0.0), (0.0042, 0.0015), (0.0, 0.0015)], segs=48, mat=amber)
        coll.location = (pw.x, pw.y, top)
    return parts, fix_part


def build(sc):
    parts, fix_part = env()
    hero = parts[HERO]
    grip_part = part_world(PARTS[HERO], PLATE_C)
    rob = RB.Robot("Rob", loc=(0, 0, 0.30))
    # TCP at the MCP boss centre, 3 mm above mid-height; approach straight down; fingers close along part z (world X)
    def tcp_at(part_M, lift=0.0):
        c = part_M @ V((0.0, 0.0, 0.0))
        return RB.tcp_matrix(c + V((0, 0, 0.003 + lift)), approach=(0, 0, -1), close_dir=(1, 0, 0))
    G = tcp_at(grip_part)
    Fx = tcp_at(fix_part)
    keys = [(566, tcp_at(grip_part, 0.16)), (578, tcp_at(grip_part, 0.028)), (583, G), (587, G),
            (596, tcp_at(grip_part, 0.11)), (613, tcp_at(fix_part, 0.075)), (619, Fx), (625, Fx), (630, tcp_at(fix_part, 0.07))]

    def lerpM(A, B, t):
        pa, pb = A.translation, B.translation
        qa, qb = A.to_quaternion(), B.to_quaternion()
        M = qa.slerp(qb, t).to_matrix().to_4x4()
        M.translation = pa.lerp(pb, t)
        return M

    def tcp(f):
        if f <= keys[0][0]:
            return keys[0][1]
        for (fa, A), (fb, B) in zip(keys[:-1], keys[1:]):
            if f <= fb:
                t = (f - fa) / (fb - fa)
                if fa == 596:        # transfer: arc over, eased
                    t = sb.smoother(t)
                    M = lerpM(A, B, t)
                    M.translation += V((0.0, 0.0, 0.06 * math.sin(math.pi * t)))
                    return M
                return lerpM(A, B, sb.smooth(t))
        return keys[-1][1]

    q = np.array([0.6, 0.3, 0.9, 0.0, 1.2, 0.0])
    # solve a good seed at the first key
    for _ in range(3):
        q, e = rob.ik(tcp(566), q, iters=120)
    errs = []
    Q = {}
    for f in range(566, F1 + 3):
        q, e = rob.ik(tcp(f), q, iters=40)
        Q[f] = q.copy()
        errs.append(e)
        rob.set(q, frame=f)
    print("IK max err mm", round(max(errs) * 1000, 3))
    # gripper: open 16 mm half gap, closes to contact on the boss (r = 5.2 mm) 583-586, opens 621-625
    def half(f):
        if f < 583:
            return 0.016
        if f < 586:
            return sb.lerp(0.016, 0.0052, sb.smooth((f - 583) / 3.0))
        if f < 621:
            return 0.0052
        return sb.lerp(0.0052, 0.016, sb.smooth((f - 621) / 4.0))
    for f in range(566, F1 + 3):
        rob.grip(half(f), frame=f)
    # carried part follows the TCP rigidly between contact (586) and release (621)
    offs = G.inverted() @ grip_part
    hero.rotation_mode = 'QUATERNION'
    for f in range(566, F1 + 3):
        if f < 586:
            M = grip_part
        elif f < 621:
            M = rob.fk(Q[f]) @ offs
        else:
            M = fix_part
        hero.matrix_world = M
        hero.keyframe_insert("location", frame=f)
        hero.keyframe_insert("rotation_quaternion", frame=f)
    # lighting: high industrial LED panels (cool), a warm practical from the SLM side, soft bounce
    sb.world_gradient(top=(0.05, 0.055, 0.065), horizon=(0.08, 0.085, 0.09), bottom=(0.03, 0.03, 0.03), strength=0.6)
    for x in (-0.6, 0.6):
        sb.light('AREA', "Panel", loc=(x, 0.0, 3.2), target=(x, 0, 0), energy=80, color=(0.9, 0.95, 1.0), size=1.2, size_y=0.4)
    sb.light('AREA', "Key", loc=(1.8, -1.6, 2.2), target=(0.4, 0, 0.9), energy=55, color=(1.0, 0.96, 0.9), size=1.0)
    sb.light('AREA', "WarmSide", loc=(-1.0, 1.2, 1.2), target=(0.3, 0, 0.9), energy=25, color=(1.0, 0.72, 0.42), size=0.8)
    # camera: opens medium on the plate (grip legible), then pans/dollies with the carried part to the fixture,
    # revealing the whole arm; focus rides on the part
    cam = sb.camera("Cam", loc=(1.3, 0.95, 1.1), target=(0.6, 0.4, 0.83), lens=35, fstop=4.0, clip=(0.02, 60))

    def part_pos(f):
        return hero.matrix_world.translation.copy() if False else (rob.fk(Q[int(f)]) @ offs).translation

    def pos(t):
        k = sb.smoother(sb.remap(t, 0.25, 0.95))
        return V((sb.lerp(1.28, 1.52, k), sb.lerp(0.98, 0.10, k), sb.lerp(1.08, 1.30, k)))

    def tg(t):
        f = F0 + t * (F1 - F0)
        a = V((0.62, 0.40, 0.86))
        b = V((0.52, -0.30, 0.90))
        k = sb.smoother(sb.remap(t, 0.28, 0.9))
        return a.lerp(b, k) + V((0, 0, 0.05 * math.sin(math.pi * k)))

    def foc(t):
        f = int(round(F0 + t * (F1 - F0)))
        return (pos(t) - part_pos(f)).length

    sb.cam_bake(cam, F0, F1, pos, tg, focus=foc, fstop=4.0)
    return F0, F1
