"""The Moon as a globe (s23e approach, s23m departure): near-side maria (dark basalt plains, only on the Earth-facing
hemisphere, as on the real Moon), bright cratered highlands, craters at four scales with raised rims, and one young
rayed crater in the southern highlands (a Tycho). Object at the Moon's centre; the material works at any scale (unit
vector field). NEAR = direction from the Moon to the Earth; POLE = lunar north."""
import bpy, math
from mathutils import Vector as V
import sb

R = 1737.4e3


def frame(near, pole):
    near = V(near).normalized()
    pole = (V(pole) - near * near.dot(V(pole))).normalized()
    east = pole.cross(near).normalized()
    return near, pole, east


def ll(near, pole, lat, lon):
    n, p, e = frame(near, pole)
    la, lo = math.radians(lat), math.radians(lon)
    return (n * math.cos(lo) + e * math.sin(lo)) * math.cos(la) + p * math.sin(la)


def globe_mat(name, near, pole, bump_k=1.0):
    m = sb.mat(name, (0.2, 0.2, 0.2), rough=0.95, spec=0.2)
    nb = sb.NB(m)
    n = nb.vmath('NORMALIZE', nb.coord('Object'))
    NEAR, POLE, EAST = frame(near, pole)
    side = nb.vmath('DOT_PRODUCT', n, tuple(NEAR))
    warp = nb.noise(n, scale=1.6, detail=3)
    nw = nb.vmath('ADD', n, nb.vmath('SCALE', nb.vmath('SUBTRACT', warp.outputs['Color'], (0.5, 0.5, 0.5)), scale=0.5))
    mar = nb.noise(nw, scale=2.1, detail=6, rough=0.55)
    maria = nb.math('MULTIPLY', nb.maprange(mar.outputs['Fac'], 0.47, 0.53), nb.maprange(side, -0.15, 0.35))
    fine = nb.noise(n, scale=60.0, detail=6, rough=0.6)
    high = nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.20, 0.30)
    alb = nb.mix(maria, high, nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.065, 0.10), dtype='FLOAT')
    h = nb.math('MULTIPLY', nb.math('SUBTRACT', mar.outputs['Fac'], 0.5), -1500.0)
    rim_b = None
    for sc_, keep, amp in ((40.0, 0.35, 5000.0), (140.0, 0.3, 2500.0), (520.0, 0.2, 900.0), (2000.0, 0.15, 250.0)):
        v = nb.voronoi(nb.vmath('ADD', n, (0.1, sc_ * 0.0007, 0.0)), scale=sc_, feature='F1')
        sep = nb.new('ShaderNodeSeparateColor'); nb.link(v.outputs['Color'], sep.inputs[0])
        rad = nb.maprange(sep.outputs[0], 0.0, 1.0, 0.12, 0.45)
        on = nb.math('GREATER_THAN', sep.outputs[1], keep)
        on = nb.math('MULTIPLY', on, nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', maria, 0.6)))     # maria less cratered
        x = nb.math('DIVIDE', v.outputs['Distance'], rad)
        bowl = nb.math('MULTIPLY', nb.math('SUBTRACT', nb.math('MULTIPLY', x, x), 1.0), nb.math('LESS_THAN', x, 1.0))
        rim = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('POWER', nb.math('SUBTRACT', x, 1.0), 2.0), -14.0))
        h = nb.math('ADD', h, nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('ADD', bowl, nb.math('MULTIPLY', rim, 0.3)), on), amp))
        rb = nb.math('MULTIPLY', rim, on)
        rim_b = rb if rim_b is None else nb.math('ADD', rim_b, rb)
    alb = nb.math('ADD', alb, nb.math('MULTIPLY', rim_b, 0.05))
    # a young rayed crater (Tycho-like): bright halo + radial rays across the southern highlands
    ty = ll(NEAR, POLE, -43.0, -11.0)
    d = nb.math('SUBTRACT', 1.0, nb.vmath('DOT_PRODUCT', n, tuple(ty)))
    ang = nb.math('SQRT', nb.math('MULTIPLY', nb.math('MAXIMUM', d, 0.0), 2.0))
    rays = nb.noise(nb.vmath('CROSS_PRODUCT', n, tuple(ty)), scale=9.0, detail=2, rough=0.5)
    raym = nb.math('MULTIPLY', nb.maprange(rays.outputs['Fac'], 0.56, 0.66), nb.maprange(ang, 0.9, 0.05))
    halo = nb.maprange(ang, 0.09, 0.02)
    alb = nb.math('ADD', alb, nb.math('MULTIPLY', nb.math('MAXIMUM', raym, halo), 0.12))
    nb.set('Base Color', _grey(nb, alb))
    nb.set('Normal', nb.bump(h, strength=1.0, distance=2.0e-3 * bump_k))
    return m


def _grey(nb, a):
    c = nb.new('ShaderNodeCombineColor')
    for i, k in enumerate((1.0, 0.985, 0.955)):          # faintly warm grey
        nb.link(nb.math('MULTIPLY', a, k), c.inputs[i])
    return c.outputs[0]


def globe(name, loc, near, pole, radius=R, segs=1024, rings=512, bump_k=1.0):
    o = sb.prim("sphere", name, loc=tuple(loc), segments=segs, ring_count=rings, radius=radius,
                mat=globe_mat(name + "M", near, pole, bump_k))
    o.visible_shadow = False
    for p in o.data.polygons:
        p.use_smooth = True
    return o
