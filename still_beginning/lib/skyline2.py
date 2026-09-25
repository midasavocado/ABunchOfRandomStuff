"""A believable twilight city skyline: varied massing with setbacks and cornices, rooftop life (water tanks on legs,
stair bulkheads, HVAC, masts), construction cranes, tree canopies, small warm windows, atmospheric fade by distance."""
import bpy, math, random
from mathutils import Vector as V
import sb


def facade_mat(name, seed, wall, lit=0.12, horizon=(0.30, 0.22, 0.22), fade_d=900.0, strength=4.0):
    m = sb.mat(name, wall, rough=0.85)
    nb = sb.NB(m)
    pos = nb.new('ShaderNodeNewGeometry').outputs['Position']
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(pos, sep.inputs[0])
    h = nb.math('ADD', sep.outputs[0], sep.outputs[1])
    bay, flr = 3.1, 3.3
    fx = nb.math('FRACT', nb.math('DIVIDE', h, bay))
    fz = nb.math('FRACT', nb.math('DIVIDE', sep.outputs[2], flr))
    win = nb.math('MULTIPLY', nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fx, 0.5)), 0.2, 0.17),
                  nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', fz, 0.55)), 0.2, 0.17))
    cmb = nb.new('ShaderNodeCombineXYZ')
    nb.link(nb.math('FLOOR', nb.math('DIVIDE', h, bay)), cmb.inputs[0])
    nb.link(nb.math('FLOOR', nb.math('DIVIDE', sep.outputs[2], flr)), cmb.inputs[1])
    cmb.inputs[2].default_value = seed * 7.13
    wn = nb.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'
    nb.link(cmb.outputs[0], wn.inputs['Vector'])
    on = nb.maprange(wn.outputs['Value'], 1 - lit, 1 - lit + 0.001)
    wc = nb.new('ShaderNodeSeparateColor'); nb.link(wn.outputs['Color'], wc.inputs[0])
    col = nb.mix(nb.maprange(wc.outputs[0], 0.75, 0.77), (1.0, 0.58, 0.26, 1), (0.95, 0.80, 0.62, 1))
    bright = nb.maprange(wc.outputs[1], 0, 1, 0.3, 1.0)
    # atmospheric fade: far buildings sink toward the horizon colour
    cam = nb.new('ShaderNodeCameraData')
    fade = nb.maprange(cam.outputs['View Distance'], 60.0, fade_d, 0.0, 0.85)
    em = nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('MULTIPLY', win, on), bright), strength)
    em = nb.math('MULTIPLY', em, nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', fade, 0.6)))
    nb.set('Emission Color', col)
    nb.set('Emission Strength', em)
    base = nb.mix(win, (*wall, 1), (0.012, 0.014, 0.02, 1))
    base = nb.mix(fade, base, (*horizon, 1))
    nb.set('Base Color', base)
    # distant surfaces also *emit* a little of the sky colour (aerial perspective in-scatter)
    return m, fade


class Batch:
    """Accumulates geometry per material; flush() creates one mesh object per material (fast)."""
    def __init__(self):
        self.d = {}

    def _g(self, mat):
        return self.d.setdefault(mat.name, (mat, [], []))

    def box(self, c, sx, sy, sz, mat, rot=0.0):
        m, V_, F_ = self._g(mat)
        ca, sa = math.cos(rot), math.sin(rot)
        b = len(V_)
        for dz in (-0.5, 0.5):
            for dx, dy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                x, y = dx * sx, dy * sy
                V_.append((c[0] + x * ca - y * sa, c[1] + x * sa + y * ca, c[2] + dz * sz))
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            F_.append(tuple(b + i for i in f))

    def cyl(self, c, r, h, mat, n=12, r2=None):
        m, V_, F_ = self._g(mat)
        r2 = r if r2 is None else r2
        b = len(V_)
        for k, (rr, z) in enumerate(((r, -h / 2), (r2, h / 2))):
            for i in range(n):
                a = 2 * math.pi * i / n
                V_.append((c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr, c[2] + z))
        for i in range(n):
            j = (i + 1) % n
            F_.append((b + i, b + j, b + n + j, b + n + i))
        F_.append(tuple(b + i for i in range(n))[::-1])
        if r2 > 1e-4:
            F_.append(tuple(b + n + i for i in range(n)))

    def blob(self, c, r, mat, rnd, n=10, m_=6):
        """lumpy canopy blob (UV sphere with noise displacement)."""
        from mathutils import noise
        mm, V_, F_ = self._g(mat)
        b = len(V_)
        for j in range(1, m_):
            th = math.pi * j / m_
            for i in range(n):
                ph = 2 * math.pi * i / n
                d = V((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
                k = 1 + 0.35 * noise.noise(d * 2.2 + V((c[0], c[1], 0)) * 0.1)
                V_.append(tuple(V(c) + d * r * k))
        top = len(V_); V_.append((c[0], c[1], c[2] + r))
        bot = len(V_); V_.append((c[0], c[1], c[2] - r * 0.8))
        for j in range(m_ - 2):
            for i in range(n):
                i2 = (i + 1) % n
                F_.append((b + j * n + i, b + j * n + i2, b + (j + 1) * n + i2, b + (j + 1) * n + i))
        for i in range(n):
            F_.append((top, b + (i + 1) % n, b + i))
            F_.append((bot, b + (m_ - 2) * n + i, b + (m_ - 2) * n + (i + 1) % n))

    def flush(self, prefix="City"):
        objs = []
        for name, (m, V_, F_) in self.d.items():
            o = sb.mesh_obj(prefix + "_" + name, V_, F_, m, smooth_shade=False)
            objs.append(o)
        self.d = {}
        return objs


B = Batch()


def _box(name, c, sx, sy, sz, mat, rot=0.0):
    B.box(c, sx, sy, sz, mat, rot)


def building(i, c, w, d, h, rot, mats, rnd, ground_z, detail=True):
    mat = mats[i % len(mats)]
    parts = []
    tiers = 1 if h < 18 else rnd.choice([1, 2, 2, 3])
    z = ground_z
    cw, cd = w, d
    for t in range(tiers):
        th = h * (0.62 if t == 0 and tiers > 1 else (0.25 if t == 1 else 0.13)) if tiers > 1 else h
        parts.append(_box(f"b{i}_{t}", V((c.x, c.y, z + th / 2)), cw, cd, th, mat, rot))
        # cornice / parapet cap
        if detail:
            parts.append(_box(f"b{i}_{t}c", V((c.x, c.y, z + th + 0.3)), cw + 0.5, cd + 0.5, 0.6, mat, rot))
        z += th
        cw *= rnd.uniform(0.6, 0.8)
        cd *= rnd.uniform(0.6, 0.8)
    top = z
    if not detail:
        return top
    # rooftop life
    k = rnd.random()
    ca, sa = math.cos(rot), math.sin(rot)
    def off(dx, dy):
        return V((c.x + dx * ca - dy * sa, c.y + dx * sa + dy * ca, 0))
    if k < 0.35:   # timber water tank on steel legs
        p = off(rnd.uniform(-cw / 3, cw / 3), rnd.uniform(-cd / 3, cd / 3))
        B.cyl((p.x, p.y, top + 4.2), 1.6, 3.4, mat, n=18)
        B.cyl((p.x, p.y, top + 6.2), 1.75, 1.2, mat, n=18, r2=0.0)
        for a in range(4):
            aa = a * math.pi / 2 + 0.4
            B.cyl((p.x + math.cos(aa) * 1.2, p.y + math.sin(aa) * 1.2, top + 1.25), 0.08, 2.5, mat, n=5)
    if rnd.random() < 0.5:   # stair bulkhead
        p = off(rnd.uniform(-cw / 3, cw / 3), rnd.uniform(-cd / 3, cd / 3))
        _box(f"bh{i}", V((p.x, p.y, top + 1.4)), 3.0, 2.4, 2.8, mat, rot)
    for j in range(rnd.randint(0, 3)):   # HVAC units
        p = off(rnd.uniform(-cw / 2.5, cw / 2.5), rnd.uniform(-cd / 2.5, cd / 2.5))
        _box(f"hv{i}_{j}", V((p.x, p.y, top + 0.6)), rnd.uniform(1.2, 2.5), rnd.uniform(1.0, 2.0), 1.2, mat, rot)
    if rnd.random() < 0.25:  # mast with cross-arms
        p = off(rnd.uniform(-cw / 3, cw / 3), rnd.uniform(-cd / 3, cd / 3))
        mh = rnd.uniform(4, 9)
        B.cyl((p.x, p.y, top + mh / 2), 0.07, mh, mat, n=5)
        for q in range(2):
            _box(f"msx{i}_{q}", V((p.x, p.y, top + mh * (0.6 + 0.25 * q))), 1.6, 0.06, 0.06, mat, rot + q)
    return top


def crane(i, base, height, jib, rot, mat):
    """Tower crane silhouette: lattice-suggesting mast, jib, counter-jib, counterweight, hook line."""
    parts = []
    parts.append(_box(f"cr{i}m", V((base.x, base.y, base.z + height / 2)), 1.6, 1.6, height, mat))
    top = base.z + height
    ca, sa = math.cos(rot), math.sin(rot)
    L = jib
    parts.append(_box(f"cr{i}j", V((base.x + ca * L * 0.4, base.y + sa * L * 0.4, top + 0.6)), L * 1.3, 1.0, 1.0, mat, rot))
    parts.append(_box(f"cr{i}cw", V((base.x - ca * L * 0.22, base.y - sa * L * 0.22, top + 0.2)), 3.0, 2.2, 2.2, mat, rot))
    parts.append(_box(f"cr{i}t", V((base.x, base.y, top + 3.5)), 0.6, 0.6, 6.0, mat))
    hx, hy = base.x + ca * L * 0.7, base.y + sa * L * 0.7
    B.cyl((hx, hy, top - height * 0.22), 0.05, height * 0.45, mat, n=4)
    # red aviation light
    sb.prim("sphere", f"cr{i}av", loc=(base.x, base.y, top + 6.6), radius=0.35, segments=12, ring_count=6,
            mat=sb.emit_mat(f"av{i}", (1.0, 0.08, 0.03), 30.0))
    return parts


def tree(name, loc, h, rnd, mat, trunk_mat):
    B.cyl((loc[0], loc[1], loc[2] + h * 0.25), 0.18, h * 0.5, trunk_mat, n=6)
    for k in range(7):
        B.blob((loc[0] + rnd.gauss(0, h * 0.14), loc[1] + rnd.gauss(0, h * 0.14), loc[2] + h * rnd.uniform(0.55, 0.85)),
               h * rnd.uniform(0.2, 0.3), mat, rnd)


def build(center=(0, 0), rmin=30, rmax=900, count=520, seed=11, ground_z=-14.0, view_dir=math.pi / 2, view_half=math.radians(85),
          near_cap=60.0, horizon=(0.30, 0.22, 0.22)):
    rnd = random.Random(seed)
    walls = [(0.035, 0.034, 0.036), (0.05, 0.04, 0.035), (0.03, 0.033, 0.04), (0.045, 0.045, 0.045), (0.06, 0.045, 0.038)]
    mats = [facade_mat(f"Fac2_{i}", seed + i, walls[i], lit=0.08 + 0.03 * i, horizon=horizon)[0] for i in range(5)]
    for i in range(count):
        a = rnd.uniform(0, 2 * math.pi)
        d = rmin + (rmax - rmin) * rnd.random() ** 1.7
        da = math.atan2(math.sin(a - view_dir), math.cos(a - view_dir))
        c = V((center[0] + math.cos(a) * d, center[1] + math.sin(a) * d, 0))
        w, dd = rnd.uniform(10, 26), rnd.uniform(10, 22)
        h = rnd.uniform(6, 34) * (2.2 if rnd.random() < 0.06 else 1.0)
        if d < near_cap:
            h = min(h, abs(ground_z) - 1.2)          # nearest neighbours stay below our roof line
        elif abs(da) < view_half and d < 260:
            h = min(h, abs(ground_z) - 1.0 + (d - near_cap) * 0.06)
        building(i, c, w, dd, h, rnd.choice([0, math.pi / 2]) + rnd.uniform(-0.04, 0.04), mats, rnd, ground_z, detail=d < 500)
    # cranes: the city is still being built
    for i in range(4):
        a = view_dir + rnd.uniform(-0.8, 0.8)
        d = rnd.uniform(220, 600)
        crane(i, V((math.cos(a) * d, math.sin(a) * d, ground_z)), rnd.uniform(45, 70), rnd.uniform(35, 55), rnd.uniform(0, 6.28), mats[3])
    # street trees between blocks (silhouettes)
    leaf = sb.mat("TreeLeaf", (0.02, 0.03, 0.02), rough=0.8)
    trunk = sb.mat("TreeTrunk", (0.02, 0.018, 0.015), rough=0.9)
    for i in range(70):
        a = rnd.uniform(0, 2 * math.pi)
        d = rnd.uniform(35, 180)
        tree(f"t{i}", (math.cos(a) * d, math.sin(a) * d, ground_z), rnd.uniform(9, 15), rnd, leaf, trunk)
    B.flush()
