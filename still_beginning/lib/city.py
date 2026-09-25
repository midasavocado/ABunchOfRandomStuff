"""The film's city (s14 horizon, s15 street, s16): a warm, lived-in European/Asian mid-rise city.
Palette: lime plaster (cream, ochre, terracotta, sand, sage), brick, pale stone bases, timber-hybrid towers with
planted terraces, solar and green roofs, many street trees. No signage, no logos.
- facade_mat(): procedural window grid facade for secondary/far buildings (frames, reveals, glass reflections,
  a few lit interiors, blinds, weathering streaks), object-space metres.
- far_city(): the skyline cluster seen from the train (s14).
- building(): detailed street-front building with real geometry (piers, spandrels, recessed window units with
  frames/mullions/sills, balconies with railings and plants, cornices, shopfronts with awnings) for s15/s16."""
import bpy, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion
import sb
import energy as E

PLASTER = [(0.80, 0.72, 0.57), (0.74, 0.55, 0.33), (0.64, 0.37, 0.24), (0.82, 0.76, 0.64), (0.56, 0.58, 0.48),
           (0.70, 0.66, 0.60), (0.78, 0.62, 0.46)]
FLOOR_H = 3.2
GROUND_H = 4.4


# =========================================================================== materials
def plaster_mat(name, col, streaks=0.35, scale=1.0):
    """Lime plaster/render: soft mottling, darker weathering streaks running down, dirt band at the base."""
    m = sb.mat(name, col, rough=0.85)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    mot = nb.noise(co, scale=0.6 * scale, detail=8, rough=0.65)
    fine = nb.noise(co, scale=25 * scale, detail=6, rough=0.7)
    st = nb.noise(nb.mapping(co, scale=(3.0, 3.0, 0.12)), scale=1.2 * scale, detail=6, rough=0.6)
    stm = nb.maprange(st.outputs['Fac'], 0.52, 0.75, 0.0, streaks)
    base = nb.maprange(sep.outputs[2], 0.0, 1.2, 0.35, 0.0)
    c = nb.mix(nb.maprange(mot.outputs['Fac'], 0.3, 0.7), (col[0] * 0.9, col[1] * 0.88, col[2] * 0.86, 1), (col[0] * 1.06, col[1] * 1.05, col[2] * 1.04, 1))
    c = nb.mix(nb.math('ADD', stm, base), c, (col[0] * 0.55, col[1] * 0.52, col[2] * 0.5, 1))
    nb.set('Base Color', c)
    nb.set('Normal', nb.bump(nb.math('ADD', fine.outputs['Fac'], nb.math('MULTIPLY', mot.outputs['Fac'], 0.5)), strength=0.25, distance=0.004))
    return m


def facade_mat(name, wall=(0.78, 0.70, 0.56), seed=0, bay=2.7, floor_h=FLOOR_H, ground=GROUND_H, lit=0.08, win_w=0.52,
               win_h=0.58, glass_dark=(0.03, 0.04, 0.05), frame=(0.85, 0.84, 0.80), ground_glass=True, streaks=0.3):
    """Window-grid facade in object space (horizontal coordinate = x + y so all four sides get windows)."""
    m = sb.mat(name, wall, rough=0.85)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    hc = nb.math('ADD', sep.outputs[0], sep.outputs[1])
    z = sep.outputs[2]
    zu = nb.math('SUBTRACT', z, ground)
    fx = nb.math('FRACT', nb.math('DIVIDE', hc, bay))
    fz = nb.math('FRACT', nb.math('DIVIDE', zu, floor_h))
    above = nb.math('GREATER_THAN', zu, 0.0)
    dx = nb.math('ABSOLUTE', nb.math('SUBTRACT', fx, 0.5)); dz = nb.math('ABSOLUTE', nb.math('SUBTRACT', fz, 0.52))
    win = nb.math('MULTIPLY', nb.maprange(dx, win_w / 2, win_w / 2 - 0.01, 0.0, 1.0), nb.maprange(dz, win_h / 2, win_h / 2 - 0.01, 0.0, 1.0))
    win = nb.math('MULTIPLY', win, above)
    fr_ = nb.math('MULTIPLY', nb.maprange(dx, win_w / 2 + 0.035, win_w / 2 + 0.03, 0.0, 1.0), nb.maprange(dz, win_h / 2 + 0.035, win_h / 2 + 0.03, 0.0, 1.0))
    fr_ = nb.math('MULTIPLY', nb.math('SUBTRACT', fr_, win), above)
    mull = nb.math('MULTIPLY', win, nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fx, 0.5)), 0.0, 0.012, 1.0, 0.0))
    # ground floor: tall shop glazing
    gz = nb.math('MULTIPLY', nb.maprange(z, 0.35, 0.4, 0.0, 1.0), nb.maprange(z, ground - 0.9, ground - 0.85, 1.0, 0.0))
    gwin = nb.math('MULTIPLY', gz, nb.maprange(dx, 0.42, 0.40, 0.0, 1.0)) if ground_glass else nb.math('MULTIPLY', gz, 0.0)
    # per-window random: lit interiors, blinds
    cmb = nb.new('ShaderNodeCombineXYZ')
    nb.link(nb.math('FLOOR', nb.math('DIVIDE', hc, bay)), cmb.inputs[0])
    nb.link(nb.math('FLOOR', nb.math('DIVIDE', zu, floor_h)), cmb.inputs[1]); cmb.inputs[2].default_value = seed
    wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'; nb.link(cmb.outputs[0], wn.inputs['Vector'])
    wsep = nb.new('ShaderNodeSeparateColor'); nb.link(wn.outputs['Color'], wsep.inputs[0])
    is_lit = nb.maprange(wn.outputs['Value'], 1 - lit, 1 - lit + 0.001, 0.0, 1.0)
    blind = nb.math('MULTIPLY', nb.maprange(wsep.outputs[1], 0.6, 0.61, 0.0, 1.0), nb.maprange(fz, 0.64, 0.66, 0.0, 1.0))
    glass = nb.mix(nb.maprange(wsep.outputs[0], 0, 1, 0, 1), (*glass_dark, 1), (glass_dark[0] * 2.2, glass_dark[1] * 2.1, glass_dark[2] * 2.0, 1))
    glass = nb.mix(blind, glass, (0.62, 0.58, 0.5, 1))
    anyw = nb.math('MINIMUM', nb.math('ADD', win, gwin), 1.0)
    # wall with weathering
    mot = nb.noise(co, scale=0.5, detail=6)
    st = nb.noise(nb.mapping(co, scale=(3.0, 3.0, 0.1)), scale=1.0, detail=5)
    stm = nb.maprange(st.outputs['Fac'], 0.55, 0.8, 0.0, streaks)
    wallc = nb.mix(nb.maprange(mot.outputs['Fac'], 0.3, 0.7), (wall[0] * 0.9, wall[1] * 0.88, wall[2] * 0.86, 1), (wall[0] * 1.05, wall[1] * 1.04, wall[2] * 1.03, 1))
    wallc = nb.mix(stm, wallc, (wall[0] * 0.6, wall[1] * 0.57, wall[2] * 0.55, 1))
    # string course / base
    base = nb.maprange(z, ground - 0.35, ground - 0.3, 1.0, 0.0)
    wallc = nb.mix(nb.math('MULTIPLY', base, 0.35), wallc, (0.62, 0.60, 0.56, 1))
    col = nb.mix(anyw, wallc, glass)
    col = nb.mix(fr_, col, (*frame, 1))
    col = nb.mix(mull, col, (*frame, 1))
    nb.set('Base Color', col)
    nb.set('Roughness', nb.mix(anyw, 0.85, 0.05, dtype='FLOAT'))
    nb.set('Specular IOR Level', nb.mix(anyw, 0.35, 0.6, dtype='FLOAT'))
    em = nb.math('MULTIPLY', nb.math('MULTIPLY', win, is_lit), 2.5)
    nb.set('Emission Color', (1.0, 0.72, 0.42, 1))
    nb.set('Emission Strength', em)
    # reveal depth: darken a thin band inside the window top/left (fake ambient occlusion of the recess)
    nb.set('Normal', nb.bump(nb.math('ADD', fr_, nb.math('MULTIPLY', anyw, -0.6)), strength=0.6, distance=0.03))
    return m


def roof_mat(name, kind="tile"):
    if kind == "tile":
        m = sb.mat(name, (0.46, 0.22, 0.13), rough=0.7)
        nb = sb.NB(m)
        co = nb.coord('Object')
        w = nb.wave(co, scale=4.0, wtype='BANDS', direction='X')
        n = nb.noise(co, scale=2.0, detail=6)
        nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (0.36, 0.16, 0.09, 1), (0.55, 0.28, 0.16, 1)))
        nb.set('Normal', nb.bump(w.outputs['Fac'], strength=0.5, distance=0.05))
        return m
    if kind == "green":
        m = sb.mat(name, (0.08, 0.14, 0.05), rough=0.9)
        nb = sb.NB(m)
        n = nb.noise(nb.coord('Object'), scale=1.5, detail=8, rough=0.7)
        nb.set('Base Color', nb.mix(nb.maprange(n.outputs['Fac'], 0.3, 0.7), (0.05, 0.10, 0.03, 1), (0.16, 0.20, 0.07, 1)))
        return m
    if kind == "solar":
        m = sb.mat(name, (0.02, 0.03, 0.05), rough=0.1, coat=1.0, coat_rough=0.03)
        nb = sb.NB(m)
        co = nb.coord('Object')
        sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
        gx = nb.math('LESS_THAN', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[0], 1.15)), 0.03)
        gy = nb.math('LESS_THAN', nb.math('FRACT', nb.math('DIVIDE', sep.outputs[1], 2.3)), 0.02)
        nb.set('Base Color', nb.mix(nb.math('MAXIMUM', gx, gy), (0.02, 0.03, 0.05, 1), (0.5, 0.5, 0.5, 1)))
        return m
    m = sb.concrete(name, (0.5, 0.49, 0.47), scale=0.5)
    return m


# =========================================================================== far city
def far_city(center=(0, 0, 0), radius=900.0, n=260, seed=7, haze=None, tall=8, facing=(-1, 0)):
    """Skyline cluster: mid-rise blocks (5-9 floors) in a loose street grid with tree-lined gaps, pitched/green/solar
    roofs, and a handful of slender timber-hybrid towers with planted terraces near the centre."""
    rnd = random.Random(seed)
    mats = [facade_mat(f"FarFac{i}", wall=PLASTER[i % len(PLASTER)], seed=seed + i, lit=0.05) for i in range(7)]
    towers = [facade_mat(f"TowerFac{i}", wall=(0.55, 0.42, 0.30) if i == 0 else (0.72, 0.70, 0.66), seed=seed + 40 + i, bay=1.8,
                         win_w=0.7, win_h=0.72, lit=0.04, frame=(0.35, 0.30, 0.25)) for i in range(2)]
    roofs = [roof_mat("RoofTile", "tile"), roof_mat("RoofGreen", "green"), roof_mat("RoofSolar", "solar"), roof_mat("RoofFlat", "flat")]
    green = sb.mat("TerraceGreen", (0.05, 0.11, 0.03), rough=0.9)
    if haze:
        for mm in mats + towers + roofs + [green]:
            haze(mm)
    objs = []
    cx, cy, cz = center
    block = 60.0
    for i in range(n):
        # grid-aligned blocks with jitter, denser toward the centre
        r = radius * rnd.random() ** 0.8
        a = rnd.uniform(0, 2 * math.pi)
        gx = round((cx + math.cos(a) * r) / block) * block + rnd.uniform(-12, 12)
        gy = round((cy + math.sin(a) * r) / block) * block + rnd.uniform(-12, 12)
        w = rnd.uniform(18, 42); d = rnd.uniform(14, 30)
        fl = rnd.randint(4, 8)
        h = GROUND_H + fl * FLOOR_H
        b = sb.prim("cube", f"FB{i}", loc=(gx, gy, cz + h / 2), scale=(w / 2, d / 2, h / 2), mat=mats[i % len(mats)])
        E.apply_scale(b)
        b.location = (gx, gy, cz + h / 2)
        objs.append(b)
        kind = rnd.random()
        if kind < 0.45:        # pitched tile roof
            rf = sb.prim("cube", f"FR{i}", loc=(gx, gy, cz + h + d * 0.18), scale=(w / 2 + 0.3, d / 2 + 0.3, d * 0.18), mat=roofs[0])
            E.apply_scale(rf)
            me = rf.data
            for v in me.vertices:
                if v.co.z > 0:
                    v.co.y *= 0.05
            rf.location = (gx, gy, cz + h + d * 0.18)
        elif kind < 0.7:       # green roof with parapet
            sb.prim("cube", f"FG{i}", loc=(gx, gy, cz + h + 0.15), scale=(w / 2 - 0.4, d / 2 - 0.4, 0.2), mat=roofs[1])
        else:                  # solar roof: tilted rows
            for k in range(int(d / 3.2)):
                pr = sb.prim("cube", f"FS{i}_{k}", loc=(gx, gy - d / 2 + 1.8 + k * 3.2, cz + h + 0.6), scale=(w / 2 - 1.2, 1.1, 0.03), mat=roofs[2])
                pr.rotation_euler = (math.radians(-25), 0, 0)
    # timber-hybrid towers with setbacks + planted terraces
    for k in range(tall):
        a = rnd.uniform(0, 2 * math.pi); r = radius * 0.35 * rnd.random()
        tx, ty = cx + math.cos(a) * r, cy + math.sin(a) * r
        fl = rnd.randint(16, 30)
        w = rnd.uniform(20, 28)
        z = cz
        tiers = rnd.randint(2, 3)
        for t in range(tiers):
            nfl = fl // tiers
            h = nfl * FLOOR_H
            b = sb.prim("cube", f"TW{k}_{t}", loc=(tx, ty, z + h / 2), scale=(w / 2, w / 2 * 0.85, h / 2), mat=towers[k % 2])
            E.apply_scale(b); b.location = (tx, ty, z + h / 2)
            # planted terrace ring at each setback + balcony planting bands
            sb.prim("cube", f"TG{k}_{t}", loc=(tx, ty, z + h + 0.4), scale=(w / 2 + 0.2, w / 2 * 0.85 + 0.2, 0.45), mat=green)
            for f_ in range(2, nfl, 3):
                sb.prim("cube", f"TB{k}_{t}_{f_}", loc=(tx, ty, z + f_ * FLOOR_H + 0.5), scale=(w / 2 + 0.6, w / 2 * 0.85 + 0.6, 0.35), mat=green)
            z += h
            w *= 0.8
    return objs
