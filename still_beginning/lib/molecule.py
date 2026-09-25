"""s04 HIDDEN STRUCTURE: procedural alpha/beta protein (Rossmann-like parallel beta-sheet with crossover
helices on one face, a bound ligand in amber at the strands' C-terminal edge), ribbon + side-chain atoms.
Units: 1 Angstrom = SC metres (see SC). Geometry is rebuilt every frame (constant topology) so the chain can
settle from a loose coil into the fold. numpy only (runs inside Blender)."""
import math
import numpy as np

SC = 0.01          # 1 A = 1 cm in the scene
RISE_H, RAD_H, PER_H = 1.5, 2.3, 3.6
RISE_E = 3.3
SHEET_DY = 4.8


def _helix(start, direction, n, phase=0.0, up=(0, 0, 1)):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    u = np.asarray(up, float); u = u - d * (u @ d); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    out = []
    for i in range(n):
        a = phase + 2 * math.pi * i / PER_H
        out.append(np.asarray(start) + d * RISE_H * i + (u * math.cos(a) + v * math.sin(a)) * RAD_H)
    return out


def _strand(start, direction, n):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    out = []
    for i in range(n):
        pleat = 0.18 * (1 if i % 2 else -1)
        out.append(np.asarray(start) + d * RISE_E * i + np.array([0, 0, pleat]))
    return out


def _loop(a, b, n, lift=(0, 0, 0), a_dir=None, b_dir=None):
    """Smooth cubic Bezier loop of n residues between points a and b (exclusive)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ad = np.asarray(a_dir if a_dir is not None else (b - a), float)
    bd = np.asarray(b_dir if b_dir is not None else (b - a), float)
    L = np.linalg.norm(b - a)
    c1 = a + ad / (np.linalg.norm(ad) + 1e-9) * L * 0.45 + np.asarray(lift)
    c2 = b - bd / (np.linalg.norm(bd) + 1e-9) * L * 0.45 + np.asarray(lift)
    out = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        out.append((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t * t * c2 + t ** 3 * b)
    return out


def fold(n_strands=6, strand_len=10, helix_len=13, seed=3):
    """Returns (P [N,3] in A, ss list 'E'/'H'/'L', strand index per residue or -1)."""
    rng = np.random.default_rng(seed)
    P, ss, sid = [], [], []

    def add(pts, kind, k=-1):
        for p in pts:
            P.append(np.asarray(p, float)); ss.append(kind); sid.append(k)
    y0 = -SHEET_DY * (n_strands - 1) / 2
    x0 = -RISE_E * (strand_len - 1) / 2
    # N-terminal lead-in
    lead = _loop(np.array([x0 - 14, y0 - 6, 6]), np.array([x0, y0, 0]), 4, lift=(0, 0, 3), b_dir=(1, 0, 0))
    add(lead, 'L')
    order = list(range(n_strands))
    for k, j in enumerate(order):
        y = y0 + SHEET_DY * j + rng.normal(0, 0.15)
        xs = x0 + rng.normal(0, 0.6) + (0.9 if j % 2 else -0.4)
        st = _strand((xs, y, rng.normal(0, 0.2) + 0.25 * math.sin(j * 0.9)), (1, 0.035 * (j - 2.5), 0.02), strand_len)
        if k > 0:
            add(_loop(P[-1], st[0], 3, lift=(-2.5, 0, 2.0), b_dir=(1, 0, 0)), 'L')
        add(st, 'E', j)
        if k == len(order) - 1:
            break
        # crossover helix above the sheet (+Z), running back antiparallel, between strands j and j+1
        yh = y + SHEET_DY * 0.5 + rng.normal(0, 0.4)
        zh = 9.0 + rng.normal(0, 0.8) + (1.5 if k % 2 else 0.0)
        hs = np.array([st[-1][0] + 4.0, yh, zh])
        hel = _helix(hs, (-1, 0.05 * rng.normal(), 0.08 * rng.normal()), helix_len, phase=rng.uniform(0, 6.28))
        add(_loop(st[-1], hel[0], 3, lift=(3.0, 0, 2.5), a_dir=(1, 0, 0), b_dir=(-1, 0, 0)), 'L')
        add(hel, 'H')
    # C-terminal helix packing across the top
    last = P[-1]
    ht = _helix(np.array([last[0] + 5, last[1] + 3, 14.0]), (-0.35, -1, 0.05), 16, phase=1.0)
    add(_loop(last, ht[0], 4, lift=(3, 2, 4), a_dir=(1, 0, 0)), 'L')
    add(ht, 'H')
    P = np.array(P)
    return P, ss, np.array(sid)


def unfolded(P, seed=11):
    """A loose, extended coil through a larger volume (same residue count) - the pre-fold state."""
    rng = np.random.default_rng(seed)
    n = len(P)
    pts = [np.zeros(3)]
    d = np.array([1.0, 0.0, 0.0])
    for i in range(1, n):
        d = d + rng.normal(0, 0.35, 3)
        d /= np.linalg.norm(d)
        pts.append(pts[-1] + d * 3.4)
    U = np.array(pts)
    U -= U.mean(0)
    # keep it roughly spherical-ish and larger than the fold
    U *= 0.85
    return U + P.mean(0)


# ------------------------------------------------------------------ ribbon mesh (constant topology)

def _catmull(P, s):
    """P [N,3]; s samples per segment -> [(N-1)*s+1, 3] plus residue parameter."""
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, par = [], []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for k in range(s):
            u = k / s
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
            par.append(i - 1 + u)
    out.append(P[-1]); par.append(len(P) - 1)
    return np.array(out), np.array(par)


class Ribbon:
    S = 8          # samples per residue
    K = 14         # cross-section points

    def __init__(self, ss):
        self.ss = list(ss)
        n = len(ss)
        self.n = n
        m = (n - 1) * self.S + 1
        self.m = m
        # per-sample width/thickness/kind, smoothed at SS boundaries; arrowheads at strand ends
        code = np.array([{'H': 0, 'E': 1, 'L': 2}[c] for c in ss])
        par = np.linspace(0, n - 1, m)
        W = np.zeros(m); Hh = np.zeros(m); kind = np.zeros(m, int)
        for j, p in enumerate(par):
            i = int(round(p))
            c = code[i]
            kind[j] = c
            if c == 0:
                W[j], Hh[j] = 2.3, 0.55
            elif c == 1:
                W[j], Hh[j] = 2.1, 0.6
            else:
                W[j], Hh[j] = 0.75, 0.75
        # arrowheads: last residue of each strand: widen to 1.75 then taper to loop width
        for i in range(n - 1):
            if code[i] == 1 and code[i + 1] != 1:
                a = (i - 1) * self.S
                b = (i + 1) * self.S
                for j in range(max(0, a), min(m, b)):
                    u = (j - a) / (b - a)
                    W[j] = 3.4 * (1 - u) + 0.75 * u if u > 0.02 else 3.4
                    Hh[j] = 0.6
                    kind[j] = 1
        # smooth transitions helix/loop etc (moving average)
        ker = np.ones(5) / 5
        self.W = np.convolve(np.pad(W, 2, mode='edge'), ker, 'valid')
        self.H = np.convolve(np.pad(Hh, 2, mode='edge'), ker, 'valid')
        for i in range(n - 1):
            if code[i] == 1 and code[i + 1] != 1:
                a = (i - 1) * self.S; b = (i + 1) * self.S
                self.W[a:b] = W[a:b]      # keep the arrow crisp
        self.kind = kind
        # faces
        K = self.K
        faces, mats = [], []
        for j in range(m - 1):
            for k in range(K):
                a = j * K + k; b = j * K + (k + 1) % K
                faces.append((a, b, b + K, a + K))
                mats.append(kind[j])
        # caps
        self.cap0 = m * K; self.cap1 = m * K + 1
        for k in range(K):
            faces.append((self.cap0, (k + 1) % K, k)); mats.append(kind[0])
            a = (m - 1) * K
            faces.append((self.cap1, a + k, a + (k + 1) % K)); mats.append(kind[-1])
        self.faces = faces
        self.mats = mats
        ang = np.linspace(0, 2 * math.pi, K, endpoint=False)
        # superellipse cross-section (flat faces, rounded edges)
        e = 0.35
        self.cx = np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** e
        self.cy = np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** e

    def verts(self, P, guide):
        """P [n,3] CA positions (A); guide [n,3] per-residue width direction hint."""
        C, par = _catmull(P, self.S)
        G, _ = _catmull(guide, self.S)
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        B = G - T * np.sum(G * T, 1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
        # consistent orientation (avoid flips)
        for j in range(1, len(B)):
            if B[j] @ B[j - 1] < 0:
                B[j] = -B[j]
        N = np.cross(T, B)
        W = self.W[:, None]; H = self.H[:, None]
        V = C[:, None, :] + B[:, None, :] * (self.cx[None, :, None] * W[:, :, None] * 0.5 * 2) * 0.5 \
            + N[:, None, :] * (self.cy[None, :, None] * H[:, :, None] * 0.5)
        V = V.reshape(-1, 3)
        V = np.vstack([V, C[0][None], C[-1][None]])
        return V * SC


def guides(P, ss):
    """Width direction per residue: helices -> radial (outward), strands -> in-sheet lateral, loops -> smooth."""
    n = len(P)
    G = np.zeros((n, 3))
    for i in range(n):
        a = P[max(i - 1, 0)]; b = P[min(i + 1, n - 1)]
        c = P[i]
        curv = (a + b) / 2 - c
        t = b - a
        t /= np.linalg.norm(t) + 1e-9
        if ss[i] == 'E':
            # lateral in the sheet: perpendicular to the strand in the local sheet plane
            g = np.cross(t, np.array([0, 0, 1.0]))
        elif ss[i] == 'H':
            g = np.cross(t, curv)
        else:
            g = np.cross(t, curv) if np.linalg.norm(curv) > 1e-3 else np.array([0, 0, 1.0])
        G[i] = g / (np.linalg.norm(g) + 1e-9)
    # smooth guides along the chain
    for _ in range(2):
        G[1:-1] = 0.5 * G[1:-1] + 0.25 * (G[:-2] + G[2:])
    return G


def sidechains(P, ss, seed=5):
    """Pseudo-atoms per residue (element, offset direction scheme). Returns (resid, local frame weights, radius,
    element) arrays; positions computed per frame by side_positions()."""
    rng = np.random.default_rng(seed)
    atoms = []
    for i in range(len(P)):
        if rng.random() > 0.5:
            continue
        k = rng.integers(1, 4)
        elems = rng.choice(['C', 'C', 'C', 'N', 'O', 'S'], size=k, p=[0.3, 0.25, 0.15, 0.13, 0.14, 0.03])
        for j, e in enumerate(elems):
            dist = 1.7 + j * 1.35 + rng.normal(0, 0.15)
            lat = rng.normal(0, 0.45)
            r = {'C': 1.15, 'N': 1.08, 'O': 1.05, 'S': 1.35}[e]
            atoms.append((i, dist, lat, rng.normal(0, 0.4), r * rng.uniform(0.9, 1.05), e))
    return atoms


def side_positions(P, ss, atoms):
    n = len(P)
    out = np.zeros((len(atoms), 3))
    for idx, (i, dist, lat, lz, r, e) in enumerate(atoms):
        a = P[max(i - 1, 0)]; b = P[min(i + 1, n - 1)]; c = P[i]
        t = b - a; t /= np.linalg.norm(t) + 1e-9
        out_dir = c - (a + b) / 2
        if ss[i] == 'E':
            out_dir = np.array([0, 0, 1.0 if i % 2 else -1.0])
        if np.linalg.norm(out_dir) < 1e-3:
            out_dir = np.array([0, 0, 1.0])
        out_dir -= t * (out_dir @ t)
        out_dir /= np.linalg.norm(out_dir) + 1e-9
        side = np.cross(t, out_dir)
        out[idx] = c + out_dir * dist + side * lat + t * lz
    return out * SC
