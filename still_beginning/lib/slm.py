"""s07a laser powder-bed fusion (selective laser melting) close insert: titanium powder bed with six copies of
the component (lying on their sides; the current slice shows the Warren-truss silhouette), fused hatch tracks,
the laser tracing the hero part's contour: white-amber melt pool, cooling trail, spatter. Metres, plate at z=0."""
import bpy, math, os, json
import numpy as np
from mathutils import Vector as V
import sb

ROOT = sb.ROOT
D = os.path.join(ROOT, "assets", "slm")


def _nb(m):
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = m, m.node_tree, m.node_tree.nodes, m.node_tree.links
    return nb


def bed_material(fused_ids=(0, 1, 2, 3), plate=0.12):
    """Powder (matte, granular) + fused regions (dark metallic with hatch ripples) from assets/slm/slice.png."""
    m = bpy.data.materials.new("PowderBed")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = _nb(m)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(pos, sep.inputs[0])
    u = nb.math('ADD', nb.math('DIVIDE', sep.outputs[0], plate), 0.5)
    v = nb.math('ADD', nb.math('DIVIDE', sep.outputs[1], plate), 0.5)
    cmb = nb.new('ShaderNodeCombineXYZ'); nb.link(u, cmb.inputs[0]); nb.link(v, cmb.inputs[1])
    img = bpy.data.images.load(os.path.join(D, "slice.png"))
    img.colorspace_settings.name = 'Non-Color'
    tex = nb.new('ShaderNodeTexImage'); tex.image = img; tex.interpolation = 'Linear'; tex.extension = 'CLIP'
    nb.link(cmb.outputs[0], tex.inputs['Vector'])
    sc_ = nb.new('ShaderNodeSeparateColor'); nb.link(tex.outputs['Color'], sc_.inputs[0])
    inside = nb.maprange(sc_.outputs[0], 0.45, 0.55)
    pid = nb.math('SUBTRACT', nb.math('MULTIPLY', sc_.outputs[2], 8.0), 0.5)
    fused = 0.0
    fsum = None
    for i in fused_ids:
        k = nb.math('LESS_THAN', nb.math('ABSOLUTE', nb.math('SUBTRACT', pid, float(i))), 0.3)
        fsum = k if fsum is None else nb.math('MAXIMUM', fsum, k)
    fmask = nb.math('MULTIPLY', inside, fsum)
    # powder: fine grains (~40 um) with sparkle
    g1 = nb.voronoi(pos, scale=22000, feature='F1')
    g2 = nb.noise(pos, scale=3000, detail=4, rough=0.6)
    grain = nb.math('POWER', nb.maprange(g1.outputs['Distance'], 0.0, 0.55, 1.0, 0.0), 1.5)
    # fused: parallel hatch tracks along 67 deg, 100 um pitch, with slight waviness
    h = math.radians(67.0)
    proj = nb.vmath('DOT_PRODUCT', pos, (math.cos(h + math.pi / 2), math.sin(h + math.pi / 2), 0.0))
    wob = nb.noise(pos, scale=900, detail=2)
    trk = nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', proj, nb.math('MULTIPLY', wob.outputs['Fac'], 0.00004)), 2 * math.pi / 0.0001))
    ripple = nb.math('SINE', nb.math('MULTIPLY', nb.vmath('DOT_PRODUCT', pos, (math.cos(h), math.sin(h), 0.0)), 2 * math.pi / 0.00005))
    hfused = nb.math('ADD', nb.math('MULTIPLY', trk, 0.5), nb.math('MULTIPLY', ripple, 0.12))
    hpow = nb.math('ADD', grain, nb.math('MULTIPLY', g2.outputs['Fac'], 0.3))
    height = nb.mix(fmask, hpow, hfused, dtype='FLOAT')
    bump = nb.bump(height, strength=0.45, distance=0.00002)
    powder = nt.nodes.new('ShaderNodeBsdfPrincipled')
    powder.inputs['Base Color'].default_value = (0.30, 0.30, 0.31, 1)
    nb.link(nb.mix(grain, (0.13, 0.13, 0.135, 1), (0.30, 0.30, 0.31, 1)), powder.inputs['Base Color'])
    powder.inputs['Metallic'].default_value = 0.35
    powder.inputs['Roughness'].default_value = 0.62
    nb.link(bump, powder.inputs['Normal'])
    fz = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tint = nb.noise(pos, scale=400, detail=3)
    fcol = nb.mix(nb.maprange(tint.outputs['Fac'], 0.35, 0.65), (0.30, 0.29, 0.28, 1), (0.42, 0.40, 0.36, 1))
    nb.link(fcol, fz.inputs['Base Color'])
    fz.inputs['Metallic'].default_value = 1.0
    fz.inputs['Roughness'].default_value = 0.28
    nb.link(bump, fz.inputs['Normal'])
    mx = nt.nodes.new('ShaderNodeMixShader')
    nb.link(fmask, mx.inputs[0]); nb.link(powder.outputs[0], mx.inputs[1]); nb.link(fz.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs[0])
    return m


def contour_ribbon(loops, width=0.00013, lift=0.000012):
    """Hero contour as a thin raised bead ribbon; attribute 'arc' = metres along the outer loop (inner loops get
    arc beyond the outer perimeter so they read as already traced)."""
    Vs, Fs, arc = [], [], []
    total = 0.0
    outer_len = None
    for li, L in enumerate(loops):
        P = np.array(L) * 0.001
        P = P[::3]
        if len(P) < 4:
            continue
        seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)[:-1]])
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        Nn = np.stack([-T[:, 1], T[:, 0]], 1)
        b = len(Vs)
        n = len(P)
        for j in range(n):
            for s in (-1, 0, 1):
                off = Nn[j] * width * 0.5 * s
                z = lift * (1.0 if s == 0 else 0.15)
                Vs.append((P[j, 0] + off[0], P[j, 1] + off[1], z))
                arc.append((cum[j] if li == 0 else -1.0))
        for j in range(n):
            k = (j + 1) % n
            for c in range(2):
                a0 = b + j * 3 + c
                a1 = b + k * 3 + c
                Fs.append((a0, a1, a1 + 1, a0 + 1))
        if li == 0:
            outer_len = seg.sum()
    me = bpy.data.meshes.new("Contour")
    me.from_pydata(Vs, [], Fs)
    me.update()
    at = me.attributes.new("arc", 'FLOAT', 'POINT')
    at.data.foreach_set("value", np.array(arc, np.float32))
    me.polygons.foreach_set("use_smooth", np.ones(len(Fs), bool))
    o = bpy.data.objects.new("Contour", me)
    sb.link_obj(o)
    return o, outer_len


def contour_material(holder):
    """Traced contour: solidified bead (bright metal) where arc < current; glowing trail just behind the pool;
    invisible ahead of the laser. Inner loops (arc = -1) are drawn as already traced (cold)."""
    m = bpy.data.materials.new("ContourMat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = _nb(m)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    at = nb.new('ShaderNodeAttribute'); at.attribute_name = "arc"
    cur = nb.new('ShaderNodeValue'); cur.name = "ArcNow"
    holder.append(cur.outputs[0])
    arc = at.outputs['Fac']
    behind = nb.math('SUBTRACT', cur.outputs[0], arc)             # metres behind the pool
    done = nb.math('MAXIMUM', nb.math('GREATER_THAN', behind, 0.0), nb.math('LESS_THAN', arc, -0.5))
    age = nb.math('MULTIPLY', nb.math('MAXIMUM', behind, 0.0), nb.math('GREATER_THAN', arc, -0.5))
    glow = nb.math('MULTIPLY', nb.math('EXPONENT', nb.math('MULTIPLY', age, -1.0 / 0.0022)), nb.math('GREATER_THAN', behind, 0.0))
    glow = nb.math('MULTIPLY', glow, nb.math('GREATER_THAN', arc, -0.5))
    bead = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bead.inputs['Base Color'].default_value = (0.62, 0.60, 0.57, 1)
    bead.inputs['Metallic'].default_value = 1.0
    bead.inputs['Roughness'].default_value = 0.18
    # incandescent colour: white-yellow at the pool -> amber -> deep red as it cools
    col = nb.ramp(glow, [(0.0, (0.5, 0.05, 0.0)), (0.35, (1.0, 0.25, 0.02)), (0.75, (1.0, 0.55, 0.15)), (1.0, (1.0, 0.85, 0.6))])
    nb.link(col, bead.inputs['Emission Color'])
    nb.link(nb.math('MULTIPLY', nb.math('POWER', glow, 1.3), 60.0), bead.inputs['Emission Strength'])
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader')
    nb.link(done, mx.inputs[0]); nb.link(tr.outputs[0], mx.inputs[1]); nb.link(bead.outputs[0], mx.inputs[2])
    nb.link(mx.outputs[0], out.inputs[0])
    m.surface_render_method = 'DITHERED'
    return m


def point_on_loop(P, cum, total, s):
    s = s % total
    i = int(np.searchsorted(cum, s) - 1)
    i = max(0, min(i, len(P) - 2))
    t = (s - cum[i]) / max(1e-12, cum[i + 1] - cum[i])
    return P[i] * (1 - t) + P[i + 1] * t, (P[i + 1] - P[i]) / (np.linalg.norm(P[i + 1] - P[i]) + 1e-12)


def chamber(plate=0.12):
    """Build plate edge, chamber floor (powder), recoater blade on its rail, dark chamber walls, gas nozzle bar."""
    steel = sb.brushed_metal("SLMSteel", (0.55, 0.55, 0.56), rough=0.3, aniso=0.4)
    dark = sb.painted("SLMWall", (0.05, 0.052, 0.056), rough=0.5)
    bed = sb.prim("plane", "Bed", loc=(0, 0, 0), scale=(0.16, 0.16, 1))
    for zz in (-0.001,):
        rim = sb.prim("cube", "PlateRim", loc=(0, 0, -0.004), scale=(plate / 2 + 0.004, plate / 2 + 0.004, 0.0035), mat=steel)
    # recoater: blade carrier resting at the -Y side
    rc = sb.prim("cube", "Recoater", loc=(0, -0.105, 0.012), scale=(0.15, 0.012, 0.012), mat=sb.painted("RecoaterM", (0.75, 0.76, 0.77), rough=0.35))
    sb.bevel(rc, 0.002)
    blade = sb.prim("cube", "Blade", loc=(0, -0.094, 0.0015), scale=(0.148, 0.0015, 0.0015), mat=sb.rubber("BladeM", (0.12, 0.12, 0.13)))
    for sx in (-1, 1):
        sb.prim("cube", "Rail", loc=(sx * 0.17, 0, 0.01), scale=(0.008, 0.2, 0.01), mat=steel)
    # chamber walls + inert-gas flow nozzle bar at the back
    for (loc, sc_) in (((0, 0.2, 0.15), (0.25, 0.005, 0.15)), ((-0.25, 0, 0.15), (0.005, 0.25, 0.15)), ((0.25, 0, 0.15), (0.005, 0.25, 0.15))):
        sb.prim("cube", "Wall", loc=loc, scale=sc_, mat=dark)
    noz = sb.prim("cube", "GasNozzle", loc=(0, 0.17, 0.02), scale=(0.16, 0.012, 0.018), mat=steel)
    sb.bevel(noz, 0.002)
    slots_m = sb.mat("Slots", (0.01, 0.01, 0.01), rough=0.8)
    for i in range(24):
        sb.prim("cube", "Slot", loc=(-0.14 + i * 0.0122, 0.1575, 0.02), scale=(0.004, 0.0006, 0.009), mat=slots_m)
    return bed
