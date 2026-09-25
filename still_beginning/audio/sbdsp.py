"""STILL BEGINNING score - DSP helpers (numpy + ctypes wrappers around libdsp.dylib)."""
import ctypes, os, sys
import numpy as np
from scipy import signal

SR = 48000
BPM = 128.0
BEAT = 22500            # samples per beat @48k
BAR = 4 * BEAT          # 90000
TOTAL = 64 * BAR        # 5,760,000

_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, "libdsp.dylib" if sys.platform == "darwin" else "libdsp.so"))
_fp = ctypes.POINTER(ctypes.c_float)
_f, _i, _d = ctypes.c_float, ctypes.c_int, ctypes.c_double

_lib.svf.argtypes = [_fp, _fp, _i, _fp, _f, _i, _f]
_lib.ladder.argtypes = [_fp, _fp, _i, _fp, _f, _f, _f]
_lib.envfollow.argtypes = [_fp, _fp, _i, _f, _f, _f]
_lib.gainsmooth.argtypes = [_fp, _fp, _i, _f, _f, _f]
_lib.fdn_reverb.argtypes = [_fp, _fp, _fp, _fp, _i, _f, _f, _f, _f, _f, _f]
_lib.osc_blep.argtypes = [_fp, _i, _fp, _d, _i, _f, _f]
_lib.osc_blep.restype = _d
_lib.delay_pp.argtypes = [_fp, _fp, _fp, _fp, _i, _f, _f, _f, _f, _f, _f, _f]
_lib.chorus.argtypes = [_fp, _fp, _fp, _fp, _i, _f, _f, _f, _f]
_lib.comp_gain.argtypes = [_fp, _fp, _fp, _i, _f, _f, _f, _f, _f, _f]
_lib.karplus.argtypes = [_fp, _fp, _i, _f, _f, _f, _f]


def _c(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(_fp)


def _arr(x, n):
    if np.isscalar(x):
        return np.full(n, x, dtype=np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


# ------------------------------------------------------------------ oscillators
def osc(freq, n=None, shape="saw", phase=0.0, pw=0.5):
    """band-limited oscillator; freq scalar or per-sample array"""
    if n is None:
        n = len(freq)
    f, fp = _c(_arr(freq, n))
    out = np.zeros(n, np.float32)
    sh = {"saw": 0, "pulse": 1, "square": 1, "tri": 2}[shape]
    _lib.osc_blep(out.ctypes.data_as(_fp), n, fp, float(phase % 1.0), sh, float(pw), float(SR))
    if sh == 2:
        out -= np.mean(out[: min(n, 4800)]) * 0  # triangle already centred
    return out


def sine(freq, n=None, phase=0.0):
    if n is None:
        n = len(freq)
    f = _arr(freq, n).astype(np.float64)
    ph = 2 * np.pi * (np.cumsum(f) / SR) + phase
    return np.sin(ph).astype(np.float32)


# ------------------------------------------------------------------ filters
def svf(x, fc, q=0.707, mode="lp"):
    n = len(x)
    xi, xp = _c(x)
    fi, fp = _c(_arr(fc, n))
    out = np.zeros(n, np.float32)
    m = {"lp": 0, "bp": 1, "hp": 2, "notch": 3}[mode]
    _lib.svf(xp, out.ctypes.data_as(_fp), n, fp, float(q), m, float(SR))
    return out


def ladder(x, fc, res=0.2, drive=1.0):
    n = len(x)
    xi, xp = _c(x)
    fi, fp = _c(_arr(fc, n))
    out = np.zeros(n, np.float32)
    _lib.ladder(xp, out.ctypes.data_as(_fp), n, fp, float(res), float(drive), float(SR))
    return out


def envfollow(x, att_ms, rel_ms):
    xi, xp = _c(x)
    out = np.zeros(len(x), np.float32)
    _lib.envfollow(xp, out.ctypes.data_as(_fp), len(x), att_ms, rel_ms, float(SR))
    return out


def gainsmooth(t, att_ms, rel_ms, sr=SR):
    ti, tp = _c(t)
    out = np.zeros(len(t), np.float32)
    _lib.gainsmooth(tp, out.ctypes.data_as(_fp), len(t), att_ms, rel_ms, float(sr))
    return out


def rbj(kind, f0, q=0.707, gain_db=0.0):
    """RBJ biquad -> sos"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "bp":
        b = [al, 0, -al]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]; a = np.array(a) / a[0]
    return np.hstack([b, a])[None, :]


def eq(x, *bands):
    """bands: tuples (kind, f0, q, gain_db). x mono or (n,2)"""
    sos = np.vstack([rbj(*b) for b in bands])
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def hp(x, f, order=2):
    sos = signal.butter(order, f, "hp", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def lp(x, f, order=2):
    sos = signal.butter(order, f, "lp", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], "bp", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def lr_split(x, f):
    """Linkwitz-Riley 4th order crossover (zero-phase-sum up to allpass) -> low, high"""
    sl = signal.butter(2, f, "lp", fs=SR, output="sos")
    sh = signal.butter(2, f, "hp", fs=SR, output="sos")
    lo = signal.sosfilt(sl, signal.sosfilt(sl, x, axis=0), axis=0)
    hi = signal.sosfilt(sh, signal.sosfilt(sh, x, axis=0), axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


# ------------------------------------------------------------------ nonlinear
def saturate(x, drive=1.0, os=4, asym=0.0):
    """oversampled tanh saturation, level-compensated for small signals"""
    y = signal.resample_poly(x, os, 1, axis=0)
    y = np.tanh(drive * (y + asym)) - np.tanh(drive * asym)
    y = signal.resample_poly(y, 1, os, axis=0)
    return (y[: len(x)] / max(drive, 1e-6)).astype(np.float32)


# ------------------------------------------------------------------ time-based fx
def reverb(xL, xR, size=1.0, decay=2.5, damp=6000, predelay=20, mod=0.3, hp_in=180, lp_in=12000):
    n = len(xL)
    a = hp(np.stack([xL, xR], 1), hp_in)
    a = lp(a, lp_in)
    L, Lp = _c(a[:, 0]); R, Rp = _c(a[:, 1])
    oL = np.zeros(n, np.float32); oR = np.zeros(n, np.float32)
    _lib.fdn_reverb(Lp, Rp, oL.ctypes.data_as(_fp), oR.ctypes.data_as(_fp), n,
                    float(size), float(decay), float(damp), float(predelay), float(mod), float(SR))
    return np.stack([oL, oR], 1)


def delay(xL, xR, dl_beats, dr_beats, fb=0.35, lp_hz=5000, hp_hz=300, cross=0.0):
    n = len(xL)
    L, Lp = _c(xL); R, Rp = _c(xR)
    oL = np.zeros(n, np.float32); oR = np.zeros(n, np.float32)
    _lib.delay_pp(Lp, Rp, oL.ctypes.data_as(_fp), oR.ctypes.data_as(_fp), n,
                  float(dl_beats * BEAT), float(dr_beats * BEAT), float(fb), float(lp_hz), float(hp_hz),
                  float(cross), float(SR))
    return np.stack([oL, oR], 1)


def chorus(x, base_ms=12, depth_ms=3, rate=0.35):
    n = len(x)
    L, Lp = _c(x[:, 0]); R, Rp = _c(x[:, 1])
    oL = np.zeros(n, np.float32); oR = np.zeros(n, np.float32)
    _lib.chorus(Lp, Rp, oL.ctypes.data_as(_fp), oR.ctypes.data_as(_fp), n, base_ms, depth_ms, rate, float(SR))
    return np.stack([oL, oR], 1)


def comp_gain(det, thr_db, ratio, att_ms, rel_ms, knee_db=6.0):
    L, Lp = _c(det[:, 0]); R, Rp = _c(det[:, 1])
    g = np.zeros(len(det), np.float32)
    _lib.comp_gain(Lp, Rp, g.ctypes.data_as(_fp), len(det), thr_db, ratio, att_ms, rel_ms, knee_db, float(SR))
    return g


def karplus(exc, f0, t60=2.0, bright=0.5):
    e, ep = _c(exc)
    out = np.zeros(len(exc), np.float32)
    _lib.karplus(ep, out.ctypes.data_as(_fp), len(exc), float(f0), float(t60), float(bright), float(SR))
    return out


# ------------------------------------------------------------------ utilities
def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nm(s):
    """'F#5' -> midi"""
    s = s.strip()
    base = NOTE[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in "#b":
        base += 1 if s[i] == "#" else -1
        i += 1
    return base + 12 * (int(s[i:]) + 1)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(d):
    return 10 ** (np.asarray(d) / 20)


def pan_gains(p):
    """constant power pan, p in [-1,1]"""
    a = (p + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


def adsr(n, a, d, s, r, gate, curve=4.0):
    """sample-count ADSR (a,d,r in seconds). gate = samples the note is held. returns length n."""
    t = np.arange(n) / SR
    A = max(a, 1e-4)
    env = np.empty(n, np.float32)
    att = t < A
    env[att] = (t[att] / A) ** 1.0
    dec = ~att
    env[dec] = s + (1 - s) * np.exp(-(t[dec] - A) * curve / max(d, 1e-4))
    g = gate / SR
    if g < n / SR:
        lvl = s + (1 - s) * np.exp(-(max(g - A, 0)) * curve / max(d, 1e-4)) if g >= A else g / A
        rel = t >= g
        env[rel] = lvl * np.exp(-(t[rel] - g) * 6.9 / max(r, 1e-3))
    return env


def fade_edges(x, fin=0.002, fout=0.004):
    n = len(x)
    a = min(int(fin * SR), n // 2); b = min(int(fout * SR), n // 2)
    if a > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        x[:a] *= w if x.ndim == 1 else w[:, None]
    if b > 0:
        w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b))
        x[-b:] *= w if x.ndim == 1 else w[:, None]
    return x


def add(buf, x, pos):
    """add x (mono or stereo) into buf at sample pos, clipping to buffer"""
    pos = int(pos)
    if pos >= len(buf):
        return
    s0 = max(0, -pos)
    e = min(len(x), len(buf) - pos)
    if e <= s0:
        return
    if buf.ndim == 2 and x.ndim == 1:
        buf[pos + s0: pos + e] += x[s0:e, None]
    else:
        buf[pos + s0: pos + e] += x[s0:e]


def stereo(x, p=0.0):
    gl, gr = pan_gains(p)
    return np.stack([x * gl, x * gr], 1).astype(np.float32)


def curve(points, n=TOTAL, kind="lin"):
    """piecewise automation. points = [(sample, value), ...]"""
    xs = np.array([p[0] for p in points], float)
    ys = np.array([p[1] for p in points], float)
    t = np.arange(n)
    if kind == "exp":
        return np.exp(np.interp(t, xs, np.log(np.maximum(ys, 1e-9)))).astype(np.float32)
    return np.interp(t, xs, ys).astype(np.float32)


def b2s(bar, beat=0.0):
    """1-based bar + 0-based beat offset -> sample index"""
    return int(round((bar - 1) * BAR + beat * BEAT))


# ------------------------------------------------------------------ metering
def kweight(x):
    # BS.1770 K-weighting (coefficients defined for 48k)
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b1, a1, x, axis=0)
    return signal.lfilter(b2, a2, y, axis=0)


def lufs_integrated(x):
    y = kweight(x.astype(np.float64))
    blk = int(0.4 * SR); hop = int(0.1 * SR)
    ms = []
    for s in range(0, len(y) - blk + 1, hop):
        seg = y[s:s + blk]
        ms.append(np.sum(np.mean(seg ** 2, axis=0)))
    ms = np.array(ms)
    l = -0.691 + 10 * np.log10(ms + 1e-20)
    ms = ms[l > -70]
    if len(ms) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(ms)) - 10
    l = -0.691 + 10 * np.log10(ms + 1e-20)
    ms2 = ms[l > rel]
    return float(-0.691 + 10 * np.log10(np.mean(ms2)))


def lufs_short(x, win=3.0, hop=0.5):
    y = kweight(x.astype(np.float64))
    blk = int(win * SR); h = int(hop * SR)
    t, v = [], []
    for s in range(0, len(y) - blk + 1, h):
        t.append((s + blk) / SR)
        v.append(-0.691 + 10 * np.log10(np.sum(np.mean(y[s:s + blk] ** 2, axis=0)) + 1e-20))
    return np.array(t), np.array(v)


def true_peak(x, os=4):
    y = signal.resample_poly(x, os, 1, axis=0)
    return float(db(np.max(np.abs(y))))
