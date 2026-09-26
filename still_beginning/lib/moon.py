"""STILL BEGINNING - lunar surface kit for s23 (lunar south-pole research settlement at grazing sun).

  terrain(cam_xy, az0, az1, r0, r1, ...)  camera-centred fan mesh (log-radial rings -> ~constant screen density),
                                          power-law crater population, fresh craters with bright ejecta ('fresh' attr)
  height(x, y)                             analytic height lookup (same function) for placing props / contact
  regolith_mat(sun_dir)                    Lommel-Seeliger (lunar-Lambert) + opposition surge + mm-scale grain bump
  rocks(...)                               boulder field (one mesh), dust-capped rock material
  massifs(...)                             distant south-pole mountains on the horizon (curvature-aware)
  habitat(...), solar_array(...), rover(...), dust pool helpers
Units metres, Z up. Moon: g = 1.62 m/s^2, R = 1737.4 km, no atmosphere (no haze, no drifting dust)."""
import bpy, bmesh, math, random
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler, noise
import sb

G_MOON = 1.62
R_MOON = 1737400.0

# =========================================================================== heightfield

class Terrain:
    def __init__(self, seed=3, extent=700.0, n_craters=9000, rmin=0.25, rmax=45.0, big=None, flat=None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # power-law crater radii: N(>r) ~ r^-2  -> inverse CDF
        u = self.rng.random(n_craters)
        a = 2.0
        r = rmin * (1 - u * (1 - (rmin / rmax) ** a)) ** (-1 / a)
        # denser population near the camera where small ones matter
        pos = (self.rng.random((n_craters, 2)) - 0.5) * 2 * extent
        small = r < 3.0
        near = self.rng.random(n_craters) < 0.75
        pos[small & near] *= 0.18
        vnear = small & (self.rng.random(n_craters) < 0.45)
        pos[vnear] *= 0.45
        self.cx, self.cy, self.cr = pos[:, 0], pos[:, 1], r
        self.depth = r * self.rng.uniform(0.08, 0.2, n_craters) * np.where(r < 3, 0.8, 1.0)
        self.fresh = (self.rng.random(n_craters) < 0.06) & (r < 6)
        self.degr = self.rng.uniform(0.5, 1.0, n_craters)
        if big:
            for (x, y, rr, d) in big:
                self.cx = np.append(self.cx, x); self.cy = np.append(self.cy, y); self.cr = np.append(self.cr, rr)
                self.depth = np.append(self.depth, d); self.fresh = np.append(self.fresh, False); self.degr = np.append(self.degr, 1.0)
        self.flat = flat or []     # (x, y, radius, z) graded pads (settlement)

    def _crater(self, d, R, D, degr):
        """radial profile at distance d (array)."""
        p = d / R
        bowl = D * (p * p - 1.0) * degr + (1 - degr) * D * 0.5 * (np.cos(np.minimum(p, 1) * math.pi) * -1 - 0.0) * -1 * 0.0
        inside = D * (np.power(np.clip(p, 0, 1), 2.2) - 1.0)
        rim = 0.22 * D * np.exp(-((p - 1.0) / (0.18 + 0.12 * (1 - degr))) ** 2)
        ejecta = 0.22 * D * np.where(p > 1, np.power(np.maximum(p, 1.0), -3.0), 1.0)
        h = np.where(p < 1, inside + rim + 0.22 * D * (1 - (1 - p) ** 0.0) * 0.0, rim * 0.0 + ejecta)
        # smooth blend across the rim
        s = 1 / (1 + np.exp(-(p - 1.0) / 0.05))
        h = (1 - s) * (inside + 0.22 * D * np.exp(-((p - 1.0) / 0.2) ** 2) * 1.0) + s * ejecta
        return h

    def height(self, X, Y, want_fresh=False):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        H = np.zeros_like(X)
        F = np.zeros_like(X)
        # broad undulation
        H += 1.6 * np.sin(X * 0.011 + 0.7) * np.cos(Y * 0.008 + 0.3) + 0.9 * np.sin(X * 0.023 - Y * 0.017 + 1.9)
        H += 0.35 * np.sin(X * 0.07 + Y * 0.05) * np.cos(Y * 0.09 - 0.4)
        rs = np.random.default_rng(self.seed + 99)
        for k in range(48):
            lam = 0.7 * (40.0 / 0.7) ** (k / 47.0)
            th = rs.uniform(0, 2 * math.pi); ph = rs.uniform(0, 2 * math.pi)
            amp = 0.0045 * lam ** 0.9 * rs.uniform(0.3, 1.0)
            H += amp * np.sin((X * math.cos(th) + Y * math.sin(th)) * 2 * math.pi / lam + ph)
        # craters: bucketed
        cell = 8.0
        flatX, flatY = X.ravel(), Y.ravel()
        Hf, Ff = H.ravel(), F.ravel()
        ix = np.floor(flatX / cell).astype(np.int64); iy = np.floor(flatY / cell).astype(np.int64)
        key = ix * 100003 + iy
        order = np.argsort(key, kind='stable')
        ks = key[order]
        for k in range(len(self.cr)):
            R = self.cr[k]; ext = R * 2.6
            x0, x1 = int(math.floor((self.cx[k] - ext) / cell)), int(math.floor((self.cx[k] + ext) / cell))
            y0, y1 = int(math.floor((self.cy[k] - ext) / cell)), int(math.floor((self.cy[k] + ext) / cell))
            idx = []
            for i in range(x0, x1 + 1):
                a = np.searchsorted(ks, i * 100003 + y0, 'left'); b = np.searchsorted(ks, i * 100003 + y1, 'right')
                if b > a:
                    idx.append(order[a:b])
            if not idx:
                continue
            idx = np.concatenate(idx)
            d = np.hypot(flatX[idx] - self.cx[k], flatY[idx] - self.cy[k])
            m = d < ext
            if not m.any():
                continue
            ii = idx[m]
            Hf[ii] += self._crater(d[m], R, self.depth[k], self.degr[k])
            if self.fresh[k]:
                p = d[m] / R
                Ff[ii] = np.maximum(Ff[ii], np.clip(1.6 - p * 0.55, 0, 1) * (0.6 + 0.4 * np.cos(np.arctan2(flatY[ii] - self.cy[k], flatX[ii] - self.cx[k]) * 7 + k) ** 2))
        H = Hf.reshape(X.shape); F = Ff.reshape(X.shape)
        # graded pads for the settlement (smoothly flattened)
        for (x, y, rad, z) in self.flat:
            d = np.hypot(X - x, Y - y)
            w = 1 / (1 + np.exp((d - rad) / (rad * 0.12)))
            H = H * (1 - w) + (z + 0.08 * (H - z)) * w
        # lunar curvature (drop with distance from origin)
        H -= (X * X + Y * Y) / (2 * R_MOON)
        return (H, F) if want_fresh else H


def fan_mesh(T, name, origin, az0, az1, r0, r1, nr=900, na=1300, mat=None, cache=None):
    """log-radial fan around `origin` (x, y) from azimuth az0..az1 (deg, 0 = +Y, clockwise to +X).
    cache: npz path to reuse the (slow) crater evaluation."""
    import os
    ox, oy = origin
    rs = r0 * (r1 / r0) ** (np.linspace(0, 1, nr))
    az = np.radians(np.linspace(az0, az1, na))
    RR, AA = np.meshgrid(rs, az, indexing='ij')
    X = ox + RR * np.sin(AA); Y = oy + RR * np.cos(AA)
    if cache and os.path.exists(cache):
        D = np.load(cache); H, F = D["H"], D["F"]
    else:
        H, F = T.height(X, Y, want_fresh=True)
        if cache:
            np.savez(cache, H=H.astype(np.float32), F=F.astype(np.float16))
    Vt = np.stack([X, Y, H], -1).reshape(-1, 3)
    ii = np.arange(nr * na).reshape(nr, na)
    Fq = np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]], -1).reshape(-1, 4)
    import suit
    o = suit.mesh_np(name, Vt, Fq, [mat] if mat else [])
    suit._attr_float(o.data, "fresh", F.ravel())
    return o

# =========================================================================== materials

def regolith_mat(name="Regolith", sun_dir=(0, 1, 0.1), albedo=0.13, tint=(1.0, 0.97, 0.92), bump=1.0, grain=1.0,
                 opposition=0.45, attr_fresh=True, coord_scale=1.0):
    """Lunar regolith: Lommel-Seeliger limb behaviour (no Lambert darkening toward the terminator), opposition surge
    near the antisolar point, mm-scale grain + cm pebbles + tiny craterlets in the bump, darker/lighter maria patches."""
    col = tuple(albedo * t for t in tint)
    m = sb.mat(name, col, rough=1.0, spec=0.25)
    b = m.node_tree.nodes["Principled BSDF"]
    try:
        b.inputs['Diffuse Roughness'].default_value = 1.0
    except Exception:
        pass
    nb = sb.NB(m)
    co = nb.coord('Object')
    if coord_scale != 1.0:
        co = nb.vmath('SCALE', co, scale=coord_scale)
    sd = V(sun_dir).normalized()
    gn = nb.new('ShaderNodeNewGeometry')
    N = gn.outputs['Normal']; I = gn.outputs['Incoming']          # Incoming points toward the viewer
    mu0 = nb.math('MAXIMUM', nb.vmath('DOT_PRODUCT', N, tuple(sd)), 0.0)
    mu = nb.math('MAXIMUM', nb.vmath('DOT_PRODUCT', N, I), 0.02)
    ls = nb.math('DIVIDE', 2.0, nb.math('ADD', nb.math('ADD', mu0, mu), 0.06))         # (2 mu0/(mu0+mu)) / mu0
    ls = nb.math('MINIMUM', ls, 6.0)
    cosg = nb.vmath('DOT_PRODUCT', I, tuple(sd))
    g = nb.math('ARCCOSINE', nb.math('MINIMUM', nb.math('MAXIMUM', cosg, -1.0), 1.0))
    opp = nb.math('ADD', 1.0, nb.math('MULTIPLY', nb.math('EXPONENT', nb.math('DIVIDE', g, -0.10)), opposition))
    # large-scale albedo variation + fresh ejecta
    n1 = nb.noise(co, scale=0.012, detail=6, rough=0.6)
    n2 = nb.noise(co, scale=0.35, detail=5, rough=0.6)
    av = nb.math('ADD', nb.maprange(n1.outputs['Fac'], 0.3, 0.7, 0.82, 1.15), nb.maprange(n2.outputs['Fac'], 0.3, 0.7, -0.06, 0.06))
    if attr_fresh:
        fr = nb.new('ShaderNodeAttribute'); fr.attribute_name = "fresh"
        av = nb.math('ADD', av, nb.math('MULTIPLY', fr.outputs['Fac'], 0.55))
    k = nb.math('MULTIPLY', nb.math('MULTIPLY', av, ls), opp)
    base = nb.mix(k, (0, 0, 0, 1), (*col, 1), blend='MIX', clamp=False)
    base = nb.mix(1.0, (0, 0, 0, 1), (0, 0, 0, 1)) if False else base
    # scale colour by k (mix trick: factor = k from black to col, unclamped)
    nb.set('Base Color', base)
    # micro relief: grain (1-3 mm), clods (1-3 cm), craterlets (5-30 cm), bumpy undulation (0.5 m)
    grn = nb.noise(co, scale=420.0, detail=3, rough=0.7)
    clod = nb.voronoi(co, scale=45.0, feature='F1')
    cl = nb.maprange(clod.outputs['Distance'], 0.0, 0.5, 1.0, 0.0)
    cl = nb.math('MULTIPLY', nb.math('POWER', cl, 3.0), nb.maprange(nb.noise(co, scale=6.0, detail=2).outputs['Fac'], 0.45, 0.65))
    crl = nb.voronoi(co, scale=4.0, feature='F1', rand=1.0)
    cd = crl.outputs['Distance']
    craterlet = nb.math('ADD', nb.math('MULTIPLY', nb.maprange(cd, 0.0, 0.32, -1.0, 0.0), 1.0),
                        nb.math('MULTIPLY', nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('POWER', nb.math('SUBTRACT', cd, 0.34), 2.0), -600.0)), 0.35))
    craterlet = nb.math('MULTIPLY', craterlet, nb.maprange(nb.noise(co, scale=1.3, detail=2).outputs['Fac'], 0.5, 0.62))
    und = nb.noise(co, scale=1.6, detail=4, rough=0.55)
    h = nb.math('ADD', nb.math('MULTIPLY', grn.outputs['Fac'], 0.0035 * grain), nb.math('MULTIPLY', cl, 0.012 * grain))
    h = nb.math('ADD', h, nb.math('MULTIPLY', craterlet, 0.0))
    h = nb.math('ADD', h, nb.math('MULTIPLY', und.outputs['Fac'], 0.10))
    nb.set('Normal', nb.bump(h, strength=1.0 * bump, distance=1.0))
    return m


def rock_mat(name="MoonRock", sun_dir=(0, 1, 0.1)):
    """basaltic/anorthositic boulders, dust-capped tops, same lunar photometry."""
    m = regolith_mat(name, sun_dir, albedo=0.15, tint=(1.0, 0.98, 0.95), bump=0.6, grain=0.6, attr_fresh=False)
    nb = sb.NB(m)
    b = nb.bsdf
    src = b.inputs['Base Color'].links[0].from_socket
    co = nb.coord('Object')
    gn = nb.new('ShaderNodeNewGeometry')
    top = nb.maprange(nb.new('ShaderNodeSeparateXYZ').outputs[2], 0, 1) if False else None
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(gn.outputs['Normal'], sep.inputs[0])
    dustcap = nb.maprange(sep.outputs[2], 0.35, 0.8, 0.0, 1.0)
    rk = nb.noise(co, scale=9.0, detail=8, rough=0.7)
    rockc = nb.mix(nb.maprange(rk.outputs['Fac'], 0.3, 0.7), (0.55, 0.55, 0.56, 1), (1.25, 1.22, 1.18, 1))
    c = nb.mix(dustcap, nb.mix(1.0, src, rockc, blend='MULTIPLY'), src)
    nb.set('Base Color', c)
    nb.set('Roughness', nb.mix(dustcap, 0.75, 1.0, dtype='FLOAT'))
    frac = nb.voronoi(co, scale=6.0, feature='DISTANCE_TO_EDGE')
    fr = nb.maprange(frac.outputs['Distance'], 0.0, 0.05, -1.0, 0.0)
    old = b.inputs['Normal'].links[0].from_socket
    nb.set('Normal', nb.bump(nb.math('ADD', nb.math('MULTIPLY', rk.outputs['Fac'], 0.06), nb.math('MULTIPLY', fr, 0.02)), strength=1.0, distance=1.0, normal=old))
    return m

# =========================================================================== rocks

def _rock_proto(seed, sub=3):
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    off = V((rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0, 100)))
    sx, sy, sz = rng.uniform(0.8, 1.3), rng.uniform(0.7, 1.2), rng.uniform(0.45, 0.8)
    # angular facets: planes cut
    planes = [(V((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized(), rng.uniform(0.62, 0.9)) for _ in range(9)]
    for v in bm.verts:
        p = v.co.copy()
        d = 1.0 + 0.22 * noise.noise(p * 1.3 + off) + 0.07 * noise.noise(p * 4.0 + off)
        p = p * d
        for n, k in planes:
            t = p.dot(n)
            if t > k:
                p -= n * (t - k) * 0.85
        v.co = V((p.x * sx, p.y * sy, p.z * sz))
    Vt = np.array([v.co[:] for v in bm.verts]); Fc = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    return Vt, np.array(Fc)


def rocks(T, name, pts, sizes, mat, seed=0, sub=(3, 2)):
    """pts: (n,2) xy; sizes: radii. Rocks buried ~30%. Returns one joined mesh object."""
    rng = np.random.default_rng(seed)
    protos = [_rock_proto(seed * 31 + k, sub[0]) for k in range(6)] + [_rock_proto(seed * 31 + 50 + k, sub[1]) for k in range(6)]
    Vs, Fs, off = [], [], 0
    Z = T.height(pts[:, 0], pts[:, 1])
    for i, ((x, y), s, z) in enumerate(zip(pts, sizes, Z)):
        pv, pf = protos[(i % 6) + (0 if s > 0.15 else 6)]
        R = Matrix.Rotation(rng.uniform(0, 6.28), 3, 'Z') @ Matrix.Rotation(rng.uniform(-0.3, 0.3), 3, 'X')
        Rn = np.array(R)
        v = (pv @ Rn.T) * s + np.array([x, y, z - s * rng.uniform(0.2, 0.45)])
        Vs.append(v); Fs.append(pf + off); off += len(pv)
    import suit
    o = suit.mesh_np(name, np.concatenate(Vs), np.concatenate(Fs), [mat])
    return o


def scatter(T, rng, n, cx, cy, r_in, r_out, az0, az1, smin, smax, alpha=2.3):
    """points in an annular sector around (cx, cy) with power-law sizes."""
    rr = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))
    aa = np.radians(rng.uniform(az0, az1, n))
    pts = np.stack([cx + rr * np.sin(aa), cy + rr * np.cos(aa)], 1)
    u = rng.random(n)
    s = smin * (1 - u * (1 - (smin / smax) ** alpha)) ** (-1 / alpha)
    return pts, s

# =========================================================================== horizon massifs

def massifs(name, cam, az0, az1, dist=(18000, 90000), n=700, nd=90, mat=None, seed=5, peaks=None):
    """smooth, dust-mantled lunar massifs beyond the horizon: sum of broad domes + soft fbm, standing on the curved
    surface (their bases drop below the horizon naturally). peaks: list of (az_deg, dist_m, height_m, radius_m)."""
    rng = np.random.default_rng(seed)
    if peaks is None:
        peaks = [(rng.uniform(az0, az1), rng.uniform(dist[0] * 1.2, dist[1] * 0.8), rng.uniform(700, 2600),
                  rng.uniform(5000, 16000)) for _ in range(14)]
    az = np.radians(np.linspace(az0, az1, n))
    d = dist[0] * (dist[1] / dist[0]) ** np.linspace(0, 1, nd)
    AA, DD = np.meshgrid(az, d, indexing='ij')
    X = cam[0] + DD * np.sin(AA); Y = cam[1] + DD * np.cos(AA)
    H = np.zeros_like(X)
    for (pa, pd, ph, pr) in peaks:
        px, py = cam[0] + pd * math.sin(math.radians(pa)), cam[1] + pd * math.cos(math.radians(pa))
        r2 = ((X - px) ** 2 + (Y - py) ** 2) / (pr * pr)
        H += ph * np.exp(-r2 * 1.3) * (1 + 0.15 * np.sin(X / pr * 5 + Y / pr * 3))
    for k in range(40):
        lam = 1500 * (20000 / 1500) ** (k / 39)
        th = rng.uniform(0, 2 * math.pi)
        H += 0.018 * lam * rng.uniform(0.4, 1) * np.sin((X * math.cos(th) + Y * math.sin(th)) * 2 * math.pi / lam + rng.uniform(0, 6.3)) * np.clip(H / 800, 0, 1)
    H -= (DD ** 2) / (2 * R_MOON)
    Vt = np.stack([X, Y, H], -1).reshape(-1, 3)
    ii = np.arange(n * nd).reshape(n, nd)
    F = np.stack([ii[:-1, :-1], ii[1:, :-1], ii[1:, 1:], ii[:-1, 1:]], -1).reshape(-1, 4)
    import suit
    o = suit.mesh_np(name, Vt, F, [mat] if mat else [])
    sb.recalc_normals(o)
    return o

# =========================================================================== settlement

def shield_mat(name="RegShield", sun_dir=(0, 1, 0.1)):
    """3D-printed / bagged regolith shielding: same photometry as the ground, printed layer lines, bag seams."""
    m = regolith_mat(name, sun_dir, albedo=0.125, tint=(1.0, 0.965, 0.91), bump=0.0, attr_fresh=False)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    lay = nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', sep.outputs[2], nb.math('MULTIPLY', nb.noise(co, scale=0.8, detail=2).outputs['Fac'], 0.03)), 2 * math.pi / 0.055))
    # printed courses (0.24 m) read at a distance under the grazing sun, over the fine 5.5 cm layers
    course = nb.math('SINE', nb.math('MULTIPLY', sep.outputs[2], 2 * math.pi / 0.24))
    lay = nb.math('ADD', lay, nb.math('MULTIPLY', course, 3.0))
    bag = nb.voronoi(nb.mapping(co, scale=(1.0, 1.0, 1.8)), scale=1.5, feature='DISTANCE_TO_EDGE')
    bg = nb.maprange(bag.outputs['Distance'], 0.0, 0.08, -3.0, 0.0)
    grn = nb.noise(co, scale=300.0, detail=3)
    lump = nb.noise(co, scale=3.0, detail=5, rough=0.6)
    h = nb.math('ADD', nb.math('MULTIPLY', lay, 0.004), nb.math('MULTIPLY', bg, 0.012))
    h = nb.math('ADD', h, nb.math('ADD', nb.math('MULTIPLY', grn.outputs['Fac'], 0.003), nb.math('MULTIPLY', lump.outputs['Fac'], 0.05)))
    nb.set('Normal', nb.bump(h, strength=1.0, distance=1.0))
    return m


def habitat(name, T, x, y, heading, length=11.0, width=6.2, height=3.6, M=None, mats=None, airlock=True, seed=0):
    """regolith-shielded inflatable module (capsule half-buried under a printed shield) + rigid airlock at +local X.
    Returns list of objects. mats: dict(shield, white, graphite, glass, lamp)."""
    import suit
    out = []
    rng = random.Random(seed)
    zc = float(T.height(np.array([x]), np.array([y]))[0])
    Rz = Matrix.Rotation(math.radians(heading), 4, 'Z')
    base = Matrix.Translation((x, y, zc - 0.25)) @ Rz
    # shield: lofted along local X, cross-section = squashed superellipse arch + noise lumps, domed ends
    xs = np.linspace(-length / 2, length / 2, 70)
    path = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)

    def prof(i, s_):
        t = abs(xs[i]) / (length / 2)
        end = math.sqrt(max(0.0, 1 - max(0.0, (t - 0.62) / 0.38) ** 2))
        return width / 2 * max(0.05, end) * 1.05, height * max(0.05, end), 2.4

    def disp(i, s_, th, p):
        q = V((p[0] * 0.4, math.cos(th) * 1.3, math.sin(th) * 1.3)) + V((seed, seed, 0))
        return 0.18 * noise.noise(q) + 0.06 * noise.noise(q * 3.1)
    o, info = suit.tube(name + "Shield", path, prof, n=96, up=(0, 0, 1), disp=disp, mats=[mats["shield"]])
    # keep only the upper half (arch) - delete verts below local z < -0.3
    bmx = bmesh.new(); bmx.from_mesh(o.data)
    bmesh.ops.delete(bmx, geom=[v for v in bmx.verts if v.co.z < -0.6], context='VERTS')
    bmx.to_mesh(o.data); bmx.free()
    sb.recalc_normals(o)
    o.matrix_world = base
    out.append(o)
    # viewports set into the shield on both flanks (deep graphite reveal, warm cabin light behind the glass) and a
    # floodlight mast: the signs of life that tell the mound is a home
    zw = height * 0.52
    yw = width / 2 * 1.05 * (1 - 0.52 ** 2.4) ** (1 / 2.4) - 0.05
    for side in (-1, 1):
        for xw in (-length * 0.22, length * 0.06):
            fr = sb.prim("cube", name + "Port", scale=(0.62, 0.35, 0.42), mat=mats["graphite"])
            fr.location = (xw, side * yw, zw); sb.bevel(fr, 0.06, 3)
            gl = sb.prim("cube", name + "PortGlass", scale=(0.48, 0.02, 0.3), mat=mats["window"])
            gl.location = (xw, side * (yw + 0.35), zw)
            out += [fr, gl]
            fr.matrix_world = base @ fr.matrix_world
            gl.matrix_world = base @ gl.matrix_world
    pole = sb.prim("cyl", name + "FloodPole", vertices=12, radius=0.06, depth=5.2, mat=mats["graphite"])
    pole.location = (-length * 0.35, -(width / 2 + 1.6), 2.6)
    arm = sb.prim("cube", name + "FloodArm", scale=(0.5, 0.05, 0.05), mat=mats["graphite"])
    arm.location = (-length * 0.35, -(width / 2 + 1.6), 5.15)
    heads = []
    for dx in (-0.45, 0.45):
        hd = sb.prim("cube", name + "FloodHead", scale=(0.16, 0.12, 0.08), mat=mats["graphite"])
        hd.location = (-length * 0.35 + dx, -(width / 2 + 1.6), 5.05)
        em = sb.prim("cube", name + "FloodLED", scale=(0.13, 0.09, 0.01), mat=mats["lamp"])
        em.location = (-length * 0.35 + dx, -(width / 2 + 1.6), 4.965)
        heads += [hd, em]
    for ob in [pole, arm] + heads:
        ob.matrix_world = base @ ob.matrix_world
        out.append(ob)
    if airlock:
        L = 3.4
        al = sb.prim("cyl", name + "Airlock", vertices=64, radius=1.25, depth=L, mat=mats["white"])
        al.rotation_euler = (0, math.pi / 2, 0)
        al.location = (length / 2 + L / 2 - 1.2, 0, 1.35)
        sb.bevel(al, 0.06, 3)
        ring = sb.prim("cyl", name + "AirlockRing", vertices=64, radius=1.31, depth=0.14, mat=mats["graphite"])
        ring.rotation_euler = (0, math.pi / 2, 0); ring.location = (length / 2 + L - 1.25, 0, 1.35)
        ring2 = sb.dup(ring, loc=(length / 2 + 0.2, 0, 1.35), linked=True)
        cap = sb.prim("sphere", name + "AirlockCap", segments=64, ring_count=32, radius=1.25, mat=mats["white"])
        cap.scale = (0.28, 1, 1); cap.location = (length / 2 + L - 1.2, 0, 1.35)
        hatch = sb.prim("cube", name + "Hatch", scale=(0.05, 0.48, 0.75), mat=mats["graphite"])
        hatch.location = (length / 2 + L - 1.2 + 0.36, 0, 1.25)
        sb.bevel(hatch, 0.08, 4)
        win = sb.prim("cyl", name + "Window", vertices=32, radius=0.13, depth=0.06, mat=mats["window"])
        win.rotation_euler = (0, math.pi / 2, 0); win.location = (length / 2 + L - 1.2 + 0.39, 0, 1.68)
        lamp = sb.prim("cube", name + "Lamp", scale=(0.06, 0.18, 0.05), mat=mats["lamp"])
        lamp.location = (length / 2 + L - 1.2 + 0.30, 0, 2.28)
        step = sb.prim("cube", name + "Step", scale=(0.5, 0.6, 0.06), mat=mats["graphite"])
        step.location = (length / 2 + L - 0.6, 0, 0.35)
        for leg in (-0.4, 0.4):
            st = sb.prim("cyl", name + "StepLeg", vertices=12, radius=0.03, depth=0.35, mat=mats["graphite"])
            st.location = (length / 2 + L - 0.3, leg, 0.17)
        # feet / skirt under the airlock
        for px in (length / 2 + 0.6, length / 2 + L - 1.6):
            for py in (-0.8, 0.8):
                f = sb.prim("cyl", name + "Foot", vertices=16, radius=0.07, depth=0.6, mat=mats["graphite"])
                f.location = (px, py, 0.2)
        mast = sb.prim("cyl", name + "Mast", vertices=12, radius=0.035, depth=2.6, mat=mats["graphite"])
        mast.location = (length * 0.1, 0.8, height + 1.0)
        for ob in list(bpy.context.scene.objects):
            pass
        grp = [al, ring, ring2, cap, hatch, win, lamp, step, mast] + [ob for ob in bpy.context.scene.objects if ob.name.startswith(name + "StepLeg") or ob.name.startswith(name + "Foot")]
        for ob in grp:
            ob.matrix_world = base @ ob.matrix_world
            out.append(ob)
    return out


def solar_array(name, T, x, y, sun_az, mats, height=16.0, wing=(2.6, 11.0)):
    """vertical solar array tower (tracks sun azimuth): tripod base, mast, vertical wing facing the sun."""
    zc = float(T.height(np.array([x]), np.array([y]))[0])
    out = []
    root = sb.empty(name, loc=(x, y, zc))
    for k in range(3):
        a = math.radians(k * 120 + 15)
        foot = V((math.cos(a) * 2.2, math.sin(a) * 2.2, 0.0))
        top = V((0, 0, 3.2))
        mid = (foot + top) / 2
        leg = sb.prim("cyl", name + "Leg", vertices=12, radius=0.07, depth=(top - foot).length, mat=mats["graphite"], loc=mid)
        leg.rotation_mode = 'QUATERNION'; leg.rotation_quaternion = (top - foot).to_track_quat('Z', 'Y')
        leg.parent = root; out.append(leg)
        pad = sb.prim("cyl", name + "Pad", vertices=16, radius=0.28, depth=0.08, mat=mats["graphite"], loc=foot + V((0, 0, 0.0)))
        pad.parent = root; out.append(pad)
    mast = sb.prim("cyl", name + "Mast", vertices=16, radius=0.11, depth=height, mat=mats["white"], loc=(0, 0, height / 2))
    mast.parent = root; out.append(mast)
    yaw = sb.empty(name + "Yaw", loc=(0, 0, height - wing[1] / 2 - 0.3), parent=root)
    # face the sun: panel normal toward the sun azimuth
    yaw.rotation_euler = (0, 0, -math.radians(sun_az))
    w = sb.prim("cube", name + "Wing", scale=(wing[0] / 2, 0.025, wing[1] / 2), mat=mats["cells"], loc=(0, -0.2, 0))
    w.parent = yaw; out.append(w)
    fr = sb.prim("cube", name + "WingFrame", scale=(wing[0] / 2 + 0.04, 0.02, wing[1] / 2 + 0.04), mat=mats["graphite"], loc=(0, -0.17, 0))
    fr.parent = yaw; out.append(fr)
    boom = sb.prim("cube", name + "Boom", scale=(0.05, 0.12, wing[1] / 2 + 0.2), mat=mats["graphite"], loc=(0, -0.08, 0))
    boom.parent = yaw; out.append(boom)
    return root, out


def cells_mat(name="Cells"):
    """solar cell array: deep blue-black cells, silver grid lines, glassy coverslips."""
    m = sb.mat(name, (0.02, 0.025, 0.045), rough=0.12, spec=0.6, coat=0.6, coat_rough=0.05)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    gx = nb.math('FRACT', nb.math('MULTIPLY', sep.outputs[0], 1 / 0.08 * 1.3))
    gz = nb.math('FRACT', nb.math('MULTIPLY', sep.outputs[2], 1 / 0.08 * 0.11))
    lx = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', gx, 0.5)), 0.44, 0.48)
    lz = nb.maprange(nb.math('ABSOLUTE', nb.math('SUBTRACT', gz, 0.5)), 0.44, 0.48)
    grid = nb.math('MAXIMUM', lx, lz)
    tone = nb.noise(co, scale=3.0, detail=2)
    base = nb.mix(nb.maprange(tone.outputs['Fac'], 0.3, 0.7), (0.015, 0.02, 0.04, 1), (0.03, 0.035, 0.06, 1))
    nb.set('Base Color', nb.mix(grid, base, (0.55, 0.55, 0.57, 1)))
    nb.set('Metallic', nb.mix(grid, 0.0, 0.9, dtype='FLOAT'))
    return m

# =========================================================================== rover

WHEEL_R = 0.41
WHEELBASE = 2.25
TRACK = 1.85


def wheel_mat(name="WheelMesh"):
    """woven wire-mesh tyre (dark titanium) - the chevrons are geometry."""
    m = sb.mat(name, (0.30, 0.30, 0.31), metal=0.9, rough=0.42)
    nb = sb.NB(m)
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    ang = nb.math('ARCTAN2', sep.outputs[2], sep.outputs[0])
    w1 = nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', nb.math('MULTIPLY', ang, 60.0), nb.math('MULTIPLY', sep.outputs[1], 150.0)), 1.0))
    w2 = nb.math('SINE', nb.math('MULTIPLY', nb.math('SUBTRACT', nb.math('MULTIPLY', ang, 60.0), nb.math('MULTIPLY', sep.outputs[1], 150.0)), 1.0))
    mesh = nb.math('MAXIMUM', w1, w2)
    holes = nb.maprange(mesh, 0.2, 0.7, 1.0, 0.0)
    nb.set('Alpha', nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', holes, 0.0)))
    nb.set('Base Color', nb.mix(holes, (0.36, 0.36, 0.37, 1), (0.06, 0.06, 0.065, 1)))
    nb.set('Normal', nb.bump(mesh, strength=0.6, distance=0.002))
    return m


def rover(name, M, mats):
    """compact unpressurised crew rover. Returns dict(root, body, wheels=[(empty, local_pos)], seat_driver=matrix)."""
    import suit
    root = sb.empty(name)
    body = sb.empty(name + "Body", parent=root)
    parts = []
    G, W = mats["graphite"], mats["white"]

    def tubep(nm, a, b, r=0.028, m=G):
        a, b = V(a), V(b)
        o = sb.prim("cyl", nm, vertices=14, radius=r, depth=(b - a).length, mat=m, loc=(a + b) / 2)
        o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = (b - a).to_track_quat('Z', 'Y')
        o.parent = body; parts.append(o); return o
    zf = WHEEL_R + 0.12            # chassis floor height above ground
    hx, hy = WHEELBASE / 2, TRACK / 2
    # chassis: floor pan + perimeter frame
    pan = suit.rbox(name + "Pan", (2.7, 1.3, 0.07), r=0.02, mats=[G]); pan.location = (0, 0, zf); pan.parent = body; parts.append(pan)
    for sy in (-1, 1):
        tubep(name + "Rail", (-1.4, sy * 0.66, zf + 0.04), (1.4, sy * 0.66, zf + 0.04), 0.035)
    for x in (-1.35, -0.3, 0.55, 1.35):
        tubep(name + "Cross", (x, -0.66, zf + 0.04), (x, 0.66, zf + 0.04), 0.03)
    # suspension arms + fenders + wheel empties
    wheels = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = V((sx * hx, sy * hy, WHEEL_R))
            tubep(name + "Arm", (sx * (hx - 0.35), sy * 0.66, zf + 0.02), (sx * hx, sy * (hy - 0.16), WHEEL_R + 0.02), 0.03)
            tubep(name + "Arm2", (sx * (hx - 0.1), sy * 0.66, zf + 0.06), (sx * hx, sy * (hy - 0.16), WHEEL_R + 0.1), 0.022)
            fend = suit.lathe(name + "Fender", [(WHEEL_R + 0.05, -0.15), (WHEEL_R + 0.09, -0.14), (WHEEL_R + 0.10, 0.0),
                                               (WHEEL_R + 0.09, 0.14), (WHEEL_R + 0.05, 0.15)], 48, [W],
                              rfun=None)
            # keep the upper third (arc over the wheel)
            bmx = bmesh.new(); bmx.from_mesh(fend.data)
            bmesh.ops.delete(bmx, geom=[v for v in bmx.verts if v.co.z < WHEEL_R * 0.35], context='VERTS')
            bmx.to_mesh(fend.data); bmx.free()
            fend.location = c; fend.parent = body; parts.append(fend)
            we = sb.empty(name + "Wheel", loc=c, parent=root)
            wheels.append((we, c.copy(), sy))
    # wheel geometry (shared mesh): tyre + chevrons + hub
    tire = suit.lathe(name + "Tyre", [(WHEEL_R - 0.06, -0.115), (WHEEL_R - 0.01, -0.115), (WHEEL_R, -0.1), (WHEEL_R, 0.1),
                                      (WHEEL_R - 0.01, 0.115), (WHEEL_R - 0.06, 0.115)], 96, [mats["wheel"]])
    chev = []
    for k in range(24):
        a = 2 * math.pi * k / 24
        for side in (-1, 1):
            cb = sb.prim("cube", name + "Chev", scale=(0.012, 0.065, 0.006), mat=mats["metal"])
            cb.rotation_euler = (0, 0, 0)
            pos = V((math.cos(a) * (WHEEL_R + 0.004), side * 0.055, math.sin(a) * (WHEEL_R + 0.004)))
            q = Quaternion((0, 1, 0), -a) @ Quaternion((0, 0, 1), side * 0.5)
            cb.rotation_mode = 'QUATERNION'; cb.rotation_quaternion = q
            cb.location = pos
            chev.append(cb)
    hub = suit.lathe(name + "Hub", [(0.0, 0.13), (0.09, 0.13), (0.11, 0.1), (0.11, -0.05), (WHEEL_R - 0.06, -0.08), (WHEEL_R - 0.06, -0.05)], 48, [G])
    spokes = []
    for k in range(6):
        a = 2 * math.pi * k / 6
        sp = sb.prim("cube", name + "Spoke", scale=(0.018, 0.02, (WHEEL_R - 0.12) / 2), mat=G)
        sp.rotation_euler = (0, -a, 0)
        sp.location = (math.sin(a) * (WHEEL_R / 2), 0.03, math.cos(a) * (WHEEL_R / 2))
        spokes.append(sp)
    wh = suit._join([tire, hub] + chev + spokes, name + "WheelMesh")
    # wheel axis = local Y (lathe axis) -> good (rolling about Y)
    wobjs = []
    for i, (we, c, sy) in enumerate(wheels):
        o = wh if i == 0 else sb.dup(wh, linked=True)
        o.parent = we; o.matrix_parent_inverse = Matrix.Identity(4); o.location = (0, 0, 0)
        o.rotation_euler = (0, 0, 0) if sy > 0 else (0, 0, math.pi)
        spin = sb.empty(name + "Spin", parent=we)
        o.parent = spin
        wobjs.append(spin)
    # seats (2), console, antenna, cargo box, headlights
    for sy in (-0.33, 0.33):
        seat = suit.rbox(name + "Seat", (0.48, 0.5, 0.08), r=0.03, mats=[mats["seat"]]); seat.location = (-0.15, sy, zf + 0.42); seat.parent = body; parts.append(seat)
        back = suit.rbox(name + "SeatBack", (0.08, 0.5, 0.55), r=0.03, mats=[mats["seat"]]); back.location = (-0.45, sy, zf + 0.72); back.rotation_euler = (0, math.radians(-12), 0); back.parent = body; parts.append(back)
        for x in (-0.35, 0.05):
            tubep(name + "SeatLeg", (x, sy, zf + 0.04), (x, sy, zf + 0.38), 0.02)
    con = suit.rbox(name + "Console", (0.22, 0.6, 0.42), r=0.03, mats=[W]); con.location = (0.55, 0, zf + 0.42); con.rotation_euler = (0, math.radians(-20), 0); con.parent = body; parts.append(con)
    scr = sb.prim("cube", name + "Screen", scale=(0.005, 0.16, 0.09), mat=mats["screen"]); scr.location = (0.435, 0.0, zf + 0.55); scr.rotation_euler = (0, math.radians(-20), 0); scr.parent = body; parts.append(scr)
    tubep(name + "Handle", (0.62, -0.3, zf + 0.62), (0.62, 0.3, zf + 0.62), 0.016)
    box = suit.rbox(name + "Cargo", (0.75, 1.2, 0.42), r=0.04, mats=[W]); box.location = (-1.05, 0, zf + 0.26); box.parent = body; parts.append(box)
    for sy in (-0.4, 0.4):
        tubep(name + "Rack", (-1.4, sy, zf + 0.5), (-0.72, sy, zf + 0.5), 0.015)
    tubep(name + "AntMast", (1.1, 0.5, zf + 0.04), (1.1, 0.5, zf + 1.35), 0.018)
    dish = suit.lathe(name + "Dish", [(0.0, 0.0), (0.12, 0.012), (0.24, 0.05), (0.245, 0.056)], 48, [W], axis='Z')
    dish.rotation_euler = (0, math.radians(-70), 0); dish.location = (1.1, 0.5, zf + 1.38); dish.parent = body; parts.append(dish)
    for sy in (-0.45, 0.45):
        hl = sb.prim("cyl", name + "Headlight", vertices=24, radius=0.05, depth=0.06, mat=mats["lamp"], loc=(1.42, sy, zf + 0.12))
        hl.rotation_euler = (0, math.pi / 2, 0); hl.parent = body; parts.append(hl)
    seat_driver = Matrix.Translation((-0.15, 0.33, zf + 0.46))
    return dict(root=root, body=body, wheels=[(s_, w[1]) for s_, w in zip(wobjs, wheels)], wheel_empties=[w[0] for w in wheels],
                parts=parts, seat_driver=seat_driver, zf=zf)


# =========================================================================== ballistic dust (fixed pool, time-continuous)

class Dust:
    """Deterministic ballistic regolith spray from the rear-lower quadrant of each wheel.
    positions(t) is a pure function of time -> correct motion blur, no drifting (no air: pure parabolas)."""
    def __init__(self, T, wheel_track, t0, t1, rate=900, life=1.4, seed=1, speed_frac=(0.25, 1.1)):
        """wheel_track(t) -> list of (centre xyz, forward unit, angular vel rad/s, ground speed)."""
        self.T, self.wt = T, wheel_track
        rng = np.random.default_rng(seed)
        n_w = len(wheel_track(t0))
        self.n = int(rate * (t1 - t0 + life))
        self.birth = np.sort(rng.uniform(t0 - life, t1, self.n))
        self.wheel = rng.integers(0, n_w, self.n)
        self.phi = rng.uniform(math.radians(200), math.radians(260), self.n)    # rim angle behind/below axle
        self.lat = rng.uniform(-0.10, 0.10, self.n)
        self.sf = rng.uniform(*speed_frac, self.n)
        self.spread = rng.normal(0, 0.18, (self.n, 3))
        self.size = np.exp(rng.normal(math.log(0.0035), 0.5, self.n))
        self.cache = {}

    def state(self, t):
        pos = np.zeros((self.n, 3)); rad = np.zeros(self.n)
        tb = self.birth
        for w in range(len(self.wt(t))):
            sel = np.nonzero(self.wheel == w)[0]
            if not len(sel):
                continue
            # evaluate each particle's birth conditions at its own birth time (vectorised by unique times)
            for k in sel:
                pass
        return pos, rad

    def positions(self, t):
        key = round(t, 4)
        if key in self.cache:
            return self.cache[key]
        age = t - self.birth
        alive = age >= 0
        P = np.zeros((self.n, 3)); Rr = np.zeros(self.n)
        idx = np.nonzero(alive)[0]
        # birth kinematics
        for k in idx:
            c, fwd, om, v = self.wt_cached(self.birth[k])[self.wheel[k]]
            ph = self.phi[k]
            # rim point: angle measured in the wheel plane from +forward, CCW toward up
            up = V((0, 0, 1)); side = fwd.cross(up)
            rp = c + (fwd * math.cos(ph) + up * math.sin(ph)) * WHEEL_R + side * self.lat[k]
            # rim velocity = ground velocity + omega x r  (rolling forward: top moves forward)
            tang = (fwd * -math.sin(ph) + up * math.cos(ph))
            vel = fwd * v + tang * (om * WHEEL_R) * self.sf[k]
            vel = vel + V(tuple(self.spread[k])) * v * 0.35
            a = age[k]
            p = rp + vel * a + V((0, 0, -0.5 * G_MOON * a * a))
            g = float(self.T.height(np.array([p.x]), np.array([p.y]))[0])
            if p.z < g or (vel.z < 0 and a > 0.05 and rp.z < g + 0.02):
                continue
            P[k] = p; Rr[k] = self.size[k]
        self.cache[key] = (P, Rr)
        return P, Rr

    def wt_cached(self, t):
        key = round(t, 3)
        if not hasattr(self, "_wc"):
            self._wc = {}
        if key not in self._wc:
            self._wc[key] = self.wt(t)
        return self._wc[key]


def add_tracks(m, segs, halfw=0.12, gap=None, reveal=None, darken=0.78):
    """Rover tracks in a regolith material. segs: list of ((x0,y0),(x1,y1)) centre-lines of a wheel PAIR
    (two bands at +-gap/2 across the line). reveal: dict(seg index -> Value node name) = the x beyond which
    the track does not exist yet (for tracks being laid during the shot; segment must run along +X)."""
    nb = sb.NB(m)
    b = nb.bsdf
    co = nb.coord('Object')
    sep = nb.new('ShaderNodeSeparateXYZ'); nb.link(co, sep.inputs[0])
    X, Y = sep.outputs[0], sep.outputs[1]
    gap = gap if gap is not None else TRACK
    tot = None
    along_any = None
    for i, (a, bb) in enumerate(segs):
        a, bb = V((a[0], a[1], 0)), V((bb[0], bb[1], 0))
        d = bb - a; L = d.length; t_ = d / L; n_ = V((-t_.y, t_.x, 0))
        rx = nb.math('SUBTRACT', X, a.x); ry = nb.math('SUBTRACT', Y, a.y)
        along = nb.math('ADD', nb.math('MULTIPLY', rx, t_.x), nb.math('MULTIPLY', ry, t_.y))
        acr = nb.math('ADD', nb.math('MULTIPLY', rx, n_.x), nb.math('MULTIPLY', ry, n_.y))
        inside = nb.math('MULTIPLY', nb.maprange(along, -0.3, 0.3), nb.maprange(along, L + 0.3, L - 0.3))
        if reveal and i in reveal:
            rv = nb.new('ShaderNodeValue'); rv.name = reveal[i]; rv.label = reveal[i]
            inside = nb.math('MULTIPLY', inside, nb.maprange(nb.math('SUBTRACT', X, rv.outputs[0]), 0.05, -0.05))
        band = None
        for sgn in (-1, 1):
            dd = nb.math('ABSOLUTE', nb.math('SUBTRACT', acr, sgn * gap / 2))
            bnd = nb.maprange(dd, halfw, halfw * 0.7)
            band = bnd if band is None else nb.math('MAXIMUM', band, bnd)
        w = nb.math('MULTIPLY', band, inside)
        # chevron tread pattern along the track
        chev = nb.math('SINE', nb.math('MULTIPLY', nb.math('ADD', along, nb.math('MULTIPLY', nb.math('ABSOLUTE', nb.math('SUBTRACT', nb.math('ABSOLUTE', nb.math('SUBTRACT', acr, 0)), gap / 2)), 0.6)), 2 * math.pi / 0.07))
        cw = nb.math('MULTIPLY', w, nb.math('ADD', -0.6, nb.math('MULTIPLY', chev, 0.25)))
        tot = w if tot is None else nb.math('MAXIMUM', tot, w)
        along_any = cw if along_any is None else nb.math('ADD', along_any, cw)
    src = b.inputs['Base Color'].links[0].from_socket
    nb.set('Base Color', nb.mix(nb.math('MULTIPLY', tot, 1 - darken), src, (0, 0, 0, 1)))
    nrm = b.inputs['Normal'].links[0].from_socket
    nb.set('Normal', nb.bump(along_any, strength=0.8, distance=0.02, normal=nrm))
    return m


def local_heightfield(T, x0, x1, y0, y1, res=0.05):
    xs = np.arange(x0, x1 + res, res); ys = np.arange(y0, y1 + res, res)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    H = T.height(X, Y)

    def f(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        fx = np.clip((x - x0) / res, 0, len(xs) - 1.001); fy = np.clip((y - y0) / res, 0, len(ys) - 1.001)
        i = fx.astype(int); j = fy.astype(int); u = fx - i; v = fy - j
        return (H[i, j] * (1 - u) * (1 - v) + H[i + 1, j] * u * (1 - v) + H[i, j + 1] * (1 - u) * v + H[i + 1, j + 1] * u * v)
    return f


def point_cloud(name, n, mat):
    """mesh with n verts + GN Mesh-to-Points using the 'radius' attribute (renders as spheres in EEVEE/Cycles)."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(n)
    me.attributes.new("radius", 'FLOAT', 'POINT')
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    ng = bpy.data.node_groups.new(name + "GN", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = ng.nodes.new('NodeGroupInput'); go = ng.nodes.new('NodeGroupOutput')
    m2p = ng.nodes.new('GeometryNodeMeshToPoints')
    na = ng.nodes.new('GeometryNodeInputNamedAttribute'); na.data_type = 'FLOAT'; na.inputs[0].default_value = "radius"
    sm = ng.nodes.new('GeometryNodeSetMaterial'); sm.inputs['Material'].default_value = mat
    ng.links.new(gi.outputs[0], m2p.inputs['Mesh'])
    ng.links.new(na.outputs[0], m2p.inputs['Radius'])
    ng.links.new(m2p.outputs[0], sm.inputs[0])
    ng.links.new(sm.outputs[0], go.inputs[0])
    mod = o.modifiers.new("GN", 'NODES'); mod.node_group = ng
    return o
