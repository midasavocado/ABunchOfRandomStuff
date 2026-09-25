"""s05 processor package at extreme macro (real dimensions, metres): 10 mm die with an iridescent passivated
surface, 4 x N gold wire bonds (25 um) from die pads to gold bond fingers, fan-out traces under a dark solder
mask (with a restrained pulse visualisation), 0402 decoupling capacitors, organic substrate."""
import bpy, math
import numpy as np
from mathutils import Vector as V, Matrix
import sb

DIE = 0.010
DIE_T = 0.00035
DIE_Z = 0.00004 + DIE_T              # die top (substrate top = 0)
NPS = 64                             # pads / fingers per side
PAD_PITCH = 0.000135
FIN_PITCH = 0.000150
FIN_GAP = 0.0012                     # die edge -> finger inner end
FIN_L, FIN_W, FIN_H = 0.00050, 0.000085, 0.000018
PKG = 0.034


def rotz(P, k):
    a = k * math.pi / 2
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return P @ R.T


def _mesh(name, V_, F, mat, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V_], [], [tuple(f) for f in F])
    me.update()
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    if mat:
        me.materials.append(mat)
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    return o


def _boxes(centers, half, rot_k=None):
    """axis-aligned boxes: centers [n,3], half [n,3] -> verts, faces"""
    Vs, Fs = [], []
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    fz = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for i, (c, h) in enumerate(zip(centers, half)):
        b = len(Vs)
        for cc in corners:
            Vs.append(np.asarray(c) + cc * np.asarray(h))
        for f in fz:
            Fs.append(tuple(b + x for x in f))
    return np.array(Vs), Fs


# ------------------------------------------------------------------ materials

def gold(name="Gold", rough=0.14):
    m = sb.mat(name, (1.0, 0.74, 0.34), metal=1.0, rough=rough)
    nb = sb.NB(m)
    n = nb.noise(nb.coord('Object'), scale=40000, detail=2)
    nb.set('Roughness', nb.maprange(n.outputs['Fac'], 0.3, 0.7, rough * 0.8, rough * 1.4))
    return m


def solder_mask(name="Mask"):
    """Dark green-black solder mask over the laminate: glossy, faintly translucent, slight orange peel."""
    m = sb.mat(name, (0.012, 0.028, 0.022), rough=0.28, spec=0.5, coat=0.7, coat_rough=0.06)
    nb = sb.NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=9000, detail=3)
    glass_weave = nb.wave(nb.mapping(co, rot=(0, 0, 0.0)), scale=2200, wtype='BANDS', direction='X')
    glass_weave2 = nb.wave(co, scale=2200, wtype='BANDS', direction='Y')
    wv = nb.math('MULTIPLY', nb.math('ADD', glass_weave.outputs['Fac'], glass_weave2.outputs['Fac']), 0.5)
    h = nb.math('ADD', n.outputs['Fac'], nb.math('MULTIPLY', wv, 0.3))
    nb.set('Normal', nb.bump(h, strength=0.08, distance=0.00001))
    nb.set('Base Color', nb.mix(nb.math('MULTIPLY', wv, 0.4), (0.012, 0.028, 0.022, 1), (0.02, 0.04, 0.032, 1)))
    return m


def trace_mat(name, t_socket_holder):
    """Copper trace seen through the mask (slightly lighter/greener-bronze), with a travelling pulse emission
    driven by per-vertex attributes u (0..1 along the trace) and act (phase or -1 = inactive)."""
    m = sb.mat(name, (0.11, 0.10, 0.06), metal=0.55, rough=0.22, coat=0.7, coat_rough=0.06)
    nb = sb.NB(m)
    au = nb.new('ShaderNodeAttribute'); au.attribute_name = "u"
    aa = nb.new('ShaderNodeAttribute'); aa.attribute_name = "act"
    tv = nb.new('ShaderNodeValue'); tv.name = "PulseTime"
    t_socket_holder.append(tv.outputs[0])
    # pulse position p = (time - act) * speed ; e = exp(-((u - p)/w)^2) for act >= 0 and 0 <= p <= 1.2
    p = nb.math('MULTIPLY', nb.math('SUBTRACT', tv.outputs[0], aa.outputs['Fac']), 0.9)
    d = nb.math('DIVIDE', nb.math('SUBTRACT', au.outputs['Fac'], p), 0.035)
    e = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('MULTIPLY', d, d), -1.0))
    # tail: a softer trailing glow behind the head
    dt = nb.math('DIVIDE', nb.math('SUBTRACT', p, au.outputs['Fac']), 0.12)
    tail = nb.math('MULTIPLY', nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('MAXIMUM', dt, 0.0), -1.0)),
                   nb.math('GREATER_THAN', dt, 0.0))
    e = nb.math('ADD', e, nb.math('MULTIPLY', tail, 0.25))
    live = nb.math('MULTIPLY', nb.math('GREATER_THAN', aa.outputs['Fac'], -0.5), nb.math('LESS_THAN', p, 1.15))
    e = nb.math('MULTIPLY', e, live)
    nb.set('Emission Color', (1.0, 0.62, 0.28, 1))
    nb.set('Emission Strength', nb.math('MULTIPLY', e, 6.0))
    return m


def die_mat(name="Die"):
    """Passivated silicon die: mirror-dark with a mosaic of functional blocks (brick texture), fine cell-row
    striations inside blocks, per-block thin-film thickness -> iridescent colour shifts at glancing light."""
    m = sb.mat(name, (0.10, 0.11, 0.13), metal=0.85, rough=0.1, coat=1.0, coat_rough=0.02, thin_film=500)
    b = m.node_tree.nodes["Principled BSDF"]
    nb = sb.NB(m)
    co = nb.coord('Object')
    # floorplan: two overlaid rectangle tilings (different pitch/offset) -> irregular functional blocks
    def tiling(scale, bw, rh, off, rot):
        br = nb.new('ShaderNodeTexBrick')
        nb.link(nb.mapping(co, rot=(0, 0, rot), loc=(0.37, 0.11, 0)), br.inputs['Vector'])
        br.inputs['Scale'].default_value = scale
        br.inputs['Mortar Size'].default_value = 0.006
        br.inputs['Brick Width'].default_value = bw
        br.inputs['Row Height'].default_value = rh
        br.offset = off
        br.inputs['Color1'].default_value = (0.0, 0.0, 0.0, 1)
        br.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1)
        br.inputs['Mortar'].default_value = (0.5, 0.5, 0.5, 1)
        return br
    b1 = tiling(2.6, 0.9, 0.45, 0.31, 0.0)
    b2 = tiling(5.3, 0.55, 0.3, 0.63, math.pi / 2)
    s1_ = nb.new('ShaderNodeSeparateColor'); nb.link(b1.outputs['Color'], s1_.inputs[0])
    s2_ = nb.new('ShaderNodeSeparateColor'); nb.link(b2.outputs['Color'], s2_.inputs[0])
    pick = nb.math('GREATER_THAN', s1_.outputs[0], 0.45)
    blockv = nb.mix(pick, s1_.outputs[0], s2_.outputs[0], dtype='FLOAT')
    mortar = nb.math('MAXIMUM', b1.outputs['Fac'], nb.math('MULTIPLY', b2.outputs['Fac'], pick))
    # striations: fine rows (direction swaps by block)
    s1 = nb.wave(co, scale=180, wtype='BANDS', direction='X')
    s2 = nb.wave(co, scale=150, wtype='BANDS', direction='Y')
    stri = nb.mix(nb.math('GREATER_THAN', blockv, 0.5), s1.outputs['Fac'], s2.outputs['Fac'], dtype='FLOAT')
    # memory-array-like dense grid in some blocks
    grid = nb.math('MULTIPLY', s1.outputs['Fac'], s2.outputs['Fac'])
    stri = nb.mix(nb.math('GREATER_THAN', blockv, 0.8), stri, grid, dtype='FLOAT')
    h = nb.math('ADD', nb.math('MULTIPLY', stri, 0.06), nb.math('MULTIPLY', mortar, 0.8))
    nb.set('Normal', nb.bump(h, strength=0.25, distance=0.02))
    col = nb.ramp(blockv, [(0.0, (0.07, 0.075, 0.09)), (0.35, (0.11, 0.10, 0.12)), (0.6, (0.09, 0.10, 0.10)),
                           (0.85, (0.13, 0.12, 0.10)), (1.0, (0.08, 0.085, 0.10))])
    col = nb.mix(nb.math('MULTIPLY', mortar, 0.6), col, (0.35, 0.36, 0.38, 1))
    nb.set('Base Color', col)
    nb.set('Thin Film Thickness', nb.maprange(blockv, 0.0, 1.0, 320, 760))
    nb.set('Roughness', nb.maprange(stri, 0.0, 1.0, 0.06, 0.16))
    return m


def alu_pad(name="AlPad"):
    return sb.mat(name, (0.80, 0.80, 0.82), metal=1.0, rough=0.35)


def ceramic(name="Ceramic"):
    m = sb.mat(name, (0.34, 0.30, 0.25), rough=0.6, spec=0.4)
    nb = sb.NB(m)
    n = nb.noise(nb.coord('Object'), scale=30000, detail=3)
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.2, distance=0.000005))
    return m


def tin(name="Tin"):
    m = sb.mat(name, (0.78, 0.78, 0.76), metal=1.0, rough=0.22)
    nb = sb.NB(m)
    n = nb.noise(nb.coord('Object'), scale=20000, detail=3)
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.15, distance=0.000004))
    return m


# ------------------------------------------------------------------ build

def build(seed=4):
    rng = np.random.default_rng(seed)
    out = {}
    M_gold = gold("Gold")
    M_wire = gold("WireGold", 0.12)
    # substrate
    sub = sb.prim("cube", "Substrate", loc=(0, 0, -0.0005), scale=(PKG / 2, PKG / 2, 0.0005), mat=solder_mask())
    sb.bevel(sub, 0.0003, 3)
    # die attach fillet + die
    da = sb.prim("cube", "DieAttach", loc=(0, 0, 0.00002), scale=(DIE / 2 + 0.00008, DIE / 2 + 0.00008, 0.00002),
                 mat=sb.mat("Epoxy", (0.6, 0.6, 0.58), rough=0.35))
    sb.bevel(da, 0.00005, 2)
    die = sb.prim("cube", "Die", loc=(0, 0, DIE_Z - DIE_T / 2), scale=(DIE / 2, DIE / 2, DIE_T / 2), mat=die_mat())
    sb.bevel(die, 0.00001, 1)
    # seal ring (bright metal frame just inside the die edge)
    ring_m = sb.mat("SealRing", (0.75, 0.75, 0.77), metal=1.0, rough=0.25)
    rv, rf = [], []
    cen, half = [], []
    for k in range(4):
        c = rotz(np.array([[0, -DIE / 2 + 0.00006, DIE_Z + 0.0000015]]), k)[0]
        hx = (DIE / 2 - 0.00004, 0.000018, 0.0000015) if k % 2 == 0 else (0.000018, DIE / 2 - 0.00004, 0.0000015)
        cen.append(c); half.append(hx)
    Vr, Fr = _boxes(cen, half)
    _mesh("SealRing", Vr, Fr, ring_m)
    # pads, fingers, wires, fan-out traces (one side, rotated x4)
    pads_c, pads_h, fin_c, fin_h = [], [], [], []
    wires = []
    traces = []
    for k in range(4):
        for i in range(NPS):
            xo = (i - (NPS - 1) / 2)
            pad = np.array([xo * PAD_PITCH, -DIE / 2 + 0.00016, DIE_Z + 0.000002])
            fin_in = np.array([xo * FIN_PITCH, -DIE / 2 - FIN_GAP, FIN_H])
            fin_cen = fin_in + np.array([0, -FIN_L / 2, 0])
            pads_c.append(rotz(pad[None], k)[0]); pads_h.append((0.00004, 0.00004, 0.000002) if k % 2 == 0 else (0.00004, 0.00004, 0.000002))
            fc = rotz(fin_cen[None], k)[0]
            fin_c.append(fc)
            fin_h.append((FIN_W / 2, FIN_L / 2, FIN_H / 2) if k % 2 == 0 else (FIN_L / 2, FIN_W / 2, FIN_H / 2))
            # wire: ball on pad -> vertical rise -> loop apex -> stitch on the finger
            loop_h = 0.00022 + 0.00006 * (i % 3) + rng.normal(0, 0.000006)
            stitch = fin_in + np.array([0, -0.00013, FIN_H / 2])
            p0 = pad + np.array([0, 0, 0.000018])
            p1 = p0 + np.array([0, 0.0000, loop_h * 0.75])
            apex = p0 + (stitch - p0) * 0.28 + np.array([0, 0, loop_h])
            p3 = stitch + (apex - stitch) * 0.25 + np.array([0, 0, 0.00004])
            wires.append(rotz(np.array([p0, p1, apex, p3, stitch]), k))
            # fan-out trace from the finger outward to a via ring
            start = fin_in + np.array([0, -FIN_L, 0])
            out_x = xo * FIN_PITCH * 1.9
            mid = np.array([xo * FIN_PITCH, -DIE / 2 - FIN_GAP - FIN_L - 0.0006 - abs(xo) * 0.00001, 0])
            end = np.array([out_x, -DIE / 2 - 0.0062, 0])
            end2 = np.array([out_x * 1.05, -PKG / 2 + 0.0015, 0])
            traces.append((rotz(np.array([start, mid, end, end2]), k), i, k))
    Vp, Fp = _boxes(pads_c, pads_h)
    out["pads"] = _mesh("DiePads", Vp, Fp, alu_pad())
    Vf, Ff = _boxes(fin_c, fin_h)
    out["fingers"] = _mesh("BondFingers", Vf, Ff, M_gold)
    sb.bevel(out["fingers"], 0.000008, 2)
    # finger ring opening in the mask: a slightly recessed darker laminate band + nickel edge is implied by bevel
    # wires as one curve object
    cu = bpy.data.curves.new("Wires", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.0000125
    cu.bevel_resolution = 3
    cu.resolution_u = 10
    for w in wires:
        sp = cu.splines.new('BEZIER')
        sp.bezier_points.add(len(w) - 1)
        for j, p in enumerate(w):
            bp = sp.bezier_points[j]
            bp.co = tuple(p)
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
    wo = bpy.data.objects.new("Wires", cu)
    sb.link_obj(wo)
    cu.materials.append(M_wire)
    out["wires"] = wo
    # ball bonds on the pads + stitch wedges on the fingers
    bc, bh = [], []
    for w in wires:
        bc.append(w[0] - np.array([0, 0, 0.000008])); bh.append((0.000032, 0.000032, 0.000012))
    Vb, Fb = _boxes(bc, bh)
    balls = _mesh("Balls", Vb, Fb, M_wire)
    sb.subsurf(balls, 2)
    # traces (strips) with u (along) and act (pulse phase / -1)
    holder = []
    tm = trace_mat("Trace", holder)
    out["pulse_time"] = holder[0]
    TV, TF, TU, TA = [], [], [], []
    for pts, i, k in traces:
        # resample polyline
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = seg.sum()
        n = 40
        s = np.linspace(0, L, n)
        cum = np.concatenate([[0], np.cumsum(seg)])
        P = np.array([np.interp(s, cum, pts[:, d]) for d in range(3)]).T
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        Nn = np.cross(np.array([0, 0, 1.0]), T)
        w = 0.000032
        active = rng.random() < 0.13
        phase = (rng.integers(0, 8) * 11.25 + rng.uniform(0, 3)) if active else -1.0
        b = len(TV)
        for j in range(n):
            for sgn in (-1, 1):
                TV.append(P[j] + Nn[j] * w * sgn + np.array([0, 0, 0.000009]))
                TU.append(s[j] / L)
                TA.append(phase)
        for j in range(n - 1):
            a0 = b + j * 2
            TF.append((a0, a0 + 1, a0 + 3, a0 + 2))
    tr = _mesh("Traces", np.array(TV), TF, tm)
    au = tr.data.attributes.new("u", 'FLOAT', 'POINT'); au.data.foreach_set("value", np.array(TU, np.float32))
    aa = tr.data.attributes.new("act", 'FLOAT', 'POINT'); aa.data.foreach_set("value", np.array(TA, np.float32))
    out["traces"] = tr
    # vias at trace ends (small gold-ish rings under mask) - skip; 0402 decoupling caps on two sides
    cer, tn = ceramic(), tin()
    caps = []
    for k in (0, 1):
        for j in range(9):
            x = (j - 4) * 0.0017
            c = rotz(np.array([[x, -DIE / 2 - 0.0042, 0.00025]]), k)[0]
            body = sb.prim("cube", "Cap", loc=tuple(c), scale=(0.00025, 0.00050, 0.00025) if k == 0 else (0.0005, 0.00025, 0.00025), mat=cer)
            sb.bevel(body, 0.00004, 2)
            for sg in (-1, 1):
                off = np.array([0, sg * 0.00043, 0]) if k == 0 else np.array([sg * 0.00043, 0, 0])
                t_ = sb.prim("cube", "CapTerm", loc=tuple(c + off), scale=(0.00026, 0.0001, 0.000262) if k == 0 else (0.0001, 0.00026, 0.000262), mat=tn)
                sb.bevel(t_, 0.00005, 3)
                fil = sb.prim("cube", "Fillet", loc=tuple(c + off * 1.25 + np.array([0, 0, -0.00017])),
                              scale=(0.0003, 0.00012, 0.00008) if k == 0 else (0.00012, 0.0003, 0.00008), mat=tn)
                sb.bevel(fil, 0.00007, 3)
            caps.append(body)
    out["caps"] = caps
    return out
