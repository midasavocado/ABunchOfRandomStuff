"""Deep-space world for shots away from Earth (s25): a physically restrained star field (many faint stars, few
bright ones, slight colour temperature spread, no twinkle) + a soft Milky Way band with dark dust lanes.
No nebula neon: brightness and colour are kept natural (the brief: space is not a nightclub)."""
import bpy, math
import sb


def starfield(name="DeepSpace", strength=1.0, band_normal=(0.25, -0.35, 0.9), band=0.55, density=1.0):
    w = bpy.data.worlds.new(name)
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    nb = sb.NB.__new__(sb.NB)
    nb.m, nb.nt, nb.n, nb.l = w, nt, nt.nodes, nt.links
    tc = nb.new('ShaderNodeTexCoord')
    d = tc.outputs['Generated']
    dn = nb.vmath('NORMALIZE', d)

    def stars(scale, thr, gain, seed):
        v = nb.new('ShaderNodeTexVoronoi'); v.feature = 'F1'
        v.inputs['Scale'].default_value = scale
        v.inputs['Randomness'].default_value = 1.0
        nb.link(nb.vmath('ADD', dn, (seed, seed * 1.7, seed * 2.3)), v.inputs['Vector'])
        core = nb.math('POWER', nb.maprange(v.outputs['Distance'], 0.0, 0.06 * 300.0 / scale, 1.0, 0.0), 6.0)
        wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'
        nb.link(v.outputs['Position'], wn.inputs['Vector'])
        keep = nb.math('GREATER_THAN', wn.outputs['Value'], thr)
        mag = nb.math('POWER', wn.outputs['Value'], 18.0)
        b = nb.math('MULTIPLY', nb.math('MULTIPLY', core, keep), nb.math('ADD', 0.15, nb.math('MULTIPLY', mag, gain)))
        # colour temperature: blue-white .. warm
        sepc = nb.new('ShaderNodeSeparateColor'); nb.link(wn.outputs['Color'], sepc.inputs[0])
        col = nb.ramp(sepc.outputs[0], [(0.0, (0.72, 0.82, 1.0)), (0.55, (1.0, 0.98, 0.95)), (1.0, (1.0, 0.78, 0.55))])
        return b, col

    b1, c1 = stars(420.0 * density, 0.35, 40.0, 0.0)
    b2, c2 = stars(1400.0 * density, 0.6, 8.0, 3.1)
    # milky way band: exp falloff from a great circle + dust lanes + clumpy star clouds
    dotn = nb.vmath('DOT_PRODUCT', dn, band_normal)
    bandk = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('MULTIPLY', dotn, dotn), -38.0))
    clump = nb.noise(dn, scale=6.0, detail=10, rough=0.62)
    dust = nb.noise(nb.vmath('ADD', dn, (4.0, 1.0, 2.0)), scale=9.0, detail=8, rough=0.6)
    dustm = nb.maprange(dust.outputs['Fac'], 0.45, 0.62, 1.0, 0.15)
    mw = nb.math('MULTIPLY', nb.math('MULTIPLY', bandk, nb.maprange(clump.outputs['Fac'], 0.35, 0.75, 0.2, 1.0)), dustm)
    mwc = nb.mix(nb.maprange(clump.outputs['Fac'], 0.3, 0.7), (0.55, 0.62, 0.8, 1), (1.0, 0.86, 0.68, 1))
    e1 = nb.new('ShaderNodeEmission'); nb.link(c1, e1.inputs[0]); nb.link(nb.math('MULTIPLY', b1, 2.0 * strength), e1.inputs[1])
    e2 = nb.new('ShaderNodeEmission'); nb.link(c2, e2.inputs[0]); nb.link(nb.math('MULTIPLY', b2, 0.7 * strength), e2.inputs[1])
    e3 = nb.new('ShaderNodeEmission'); nb.link(mwc, e3.inputs[0]); nb.link(nb.math('MULTIPLY', mw, 0.035 * band * strength), e3.inputs[1])
    a1 = nb.new('ShaderNodeAddShader'); nb.link(e1.outputs[0], a1.inputs[0]); nb.link(e2.outputs[0], a1.inputs[1])
    a2 = nb.new('ShaderNodeAddShader'); nb.link(a1.outputs[0], a2.inputs[0]); nb.link(e3.outputs[0], a2.inputs[1])
    # stars must not light the scene (only visible to camera + glossy)
    lp = nb.new('ShaderNodeLightPath')
    vis = nb.math('MAXIMUM', lp.outputs['Is Camera Ray'], lp.outputs['Is Glossy Ray'])
    black = nb.new('ShaderNodeBackground'); black.inputs['Color'].default_value = (0, 0, 0, 1)
    mx = nb.new('ShaderNodeMixShader'); nb.link(vis, mx.inputs[0]); nb.link(black.outputs[0], mx.inputs[1]); nb.link(a2.outputs[0], mx.inputs[2])
    out = nb.new('ShaderNodeOutputWorld'); nb.link(mx.outputs[0], out.inputs[0])
    return w
