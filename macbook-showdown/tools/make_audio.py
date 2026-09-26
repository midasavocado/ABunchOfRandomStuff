"""Build the soundtrack: synthesized music, synthesized sound effects, and the voiceover, mixed.

Usage: python3 make_audio.py --build BUILD_DIR [--stems]
Reads BUILD_DIR/cues.json (voice timing), BUILD_DIR/events.json (sound effects and music cues exported
from the page with `node render.js events`), and BUILD_DIR/vo/*.wav. Writes BUILD_DIR/soundtrack.wav.

Everything is generated from code: no samples. The music is 120 BPM in F major, so every scene cut
(snapped to the beat by make_vo.py) lands on the groove.
"""
import argparse, json, os
import numpy as np, soundfile as sf
from scipy import signal
import pyloudnorm

SR = 48000
BPM = 120
BEAT = 60 / BPM
STEP = BEAT / 4  # sixteenth note
rng_master = np.random.default_rng(7)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def t_(n):
    return np.arange(int(n)) / SR


def env_adsr(n, a=0.005, d=0.1, s=0.0, r=0.05, sus_len=None):
    n = int(n)
    a_, d_, r_ = int(a * SR), int(d * SR), int(r * SR)
    sus = max(0, n - a_ - d_ - r_) if sus_len is None else int(sus_len * SR)
    e = np.concatenate([np.linspace(0, 1, max(a_, 1), endpoint=False), np.linspace(1, s, max(d_, 1), endpoint=False),
                        np.full(sus, s), np.linspace(s, 0, max(r_, 1))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def expdec(n, tau):
    return np.exp(-t_(n) / tau)


def saw(f, n, phase=0.0):
    ph = (np.cumsum(np.broadcast_to(f, (int(n),)) / SR) + phase) % 1.0
    return 2 * ph - 1


def sq(f, n, duty=0.5):
    ph = np.cumsum(np.broadcast_to(f, (int(n),)) / SR) % 1.0
    return np.where(ph < duty, 1.0, -1.0)


def sine(f, n, phase=0.0):
    return np.sin(2 * np.pi * (np.cumsum(np.broadcast_to(f, (int(n),)) / SR) + phase))


def noise(n, rng=None):
    return (rng or rng_master).standard_normal(int(n))


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, min(hi, SR / 2 - 100)], 'bandpass', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, min(f, SR / 2 - 100), 'lowpass', fs=SR, output='sos'), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, 'highpass', fs=SR, output='sos'), x)


def sweep_lp(x, f0, f1, curve='exp'):
    """Time-varying one-pole-ish lowpass by processing in blocks (cheap and good enough for synths)."""
    out = np.zeros_like(x)
    blk = 256
    nb = int(np.ceil(len(x) / blk))
    zi = None
    for i in range(nb):
        k = i / max(1, nb - 1)
        f = f0 * (f1 / f0) ** k if curve == 'exp' else f0 + (f1 - f0) * k
        b, a = signal.butter(2, min(max(f, 30), SR / 2 - 200), 'lowpass', fs=SR)
        seg = x[i * blk:(i + 1) * blk]
        if zi is None:
            zi = signal.lfilter_zi(b, a) * 0
        y, zi = signal.lfilter(b, a, seg, zi=zi)
        out[i * blk:i * blk + len(seg)] = y
    return out


def norm(x, peak=1.0):
    m = np.max(np.abs(x)) or 1
    return x / m * peak


class Bus:
    def __init__(self, dur):
        self.x = np.zeros((int(dur * SR) + SR * 4, 2))

    def add(self, t, y, gain=1.0, pan=0.0):
        if y.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            y = np.stack([y * l * 1.414, y * r * 1.414], 1)
        i = int(round(t * SR))
        if i < 0:
            y, i = y[-i:], 0
        n = min(len(y), len(self.x) - i)
        if n > 0:
            self.x[i:i + n] += y[:n] * gain


def reverb_ir(dur=1.6, decay=0.45, seed=3, pre=0.012, bright=6000):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    ir = np.zeros((n, 2))
    for c in range(2):
        e = r.standard_normal(n) * np.exp(-t_(n) / decay)
        e = lp(e, bright)
        ir[:, c] = e
    ir[:int(pre * SR)] = 0
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def convolve_st(x, ir):
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)


# ---------------------------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------------------------
def kick(vel=1.0, dur=0.32):
    n = int(dur * SR)
    f = 45 + 110 * np.exp(-t_(n) / 0.03)
    body = sine(f, n) * expdec(n, 0.12)
    click = lp(noise(n), 3000) * expdec(n, 0.004) * 0.3
    return np.tanh((body + click) * 1.6) * vel


def clap(vel=1.0):
    n = int(0.25 * SR)
    x = np.zeros(n)
    nz = bp(noise(n), 900, 5200)
    for k, d in enumerate([0, 0.011, 0.022]):
        i = int(d * SR)
        x[i:] += nz[:n - i] * expdec(n - i, 0.008 if k < 2 else 0.07)
    tone = sine(190, n) * expdec(n, 0.04) * 0.4
    return (x * 0.8 + tone) * vel


def hat(vel=1.0, open_=False):
    n = int((0.25 if open_ else 0.05) * SR)
    x = hp(noise(n), 7000, 4) * expdec(n, 0.09 if open_ else 0.012)
    return x * vel


def bass_note(m, dur, vel=1.0):
    n = int(dur * SR)
    f = mtof(m)
    x = saw(f, n) * 0.6 + sq(f / 2, n) * 0.25 + sine(f, n) * 0.6
    x = sweep_lp(x, 1400, 280)
    return np.tanh(x * 1.3) * env_adsr(n, 0.003, 0.08, 0.7, 0.03) * vel


def brass(ms, dur, vel=1.0, bright=4200):
    n = int(dur * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in ms:
        f = mtof(m)
        vib = 1 + 0.003 * np.sin(2 * np.pi * 5.5 * t_(n)) * np.clip(t_(n) / 0.2, 0, 1)
        L += saw(f * 1.004 * vib, n) + saw(f * 0.997 * vib, n, 0.3)
        R += saw(f * 0.996 * vib, n, 0.5) + saw(f * 1.003 * vib, n, 0.7)
    e = env_adsr(n, 0.012, 0.12, 0.6, 0.06)
    L = sweep_lp(L, bright, bright * 0.35) * e
    R = sweep_lp(R, bright, bright * 0.35) * e
    return np.stack([L, R], 1) / (2 * len(ms)) * vel


def pad(ms, dur, vel=1.0):
    n = int(dur * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in ms:
        f = mtof(m)
        L += saw(f * 1.006, n) + saw(f * 0.995, n, 0.2)
        R += saw(f * 0.994, n, 0.4) + saw(f * 1.005, n, 0.6)
    e = env_adsr(n, 0.25, 0.3, 0.8, 0.4)
    return np.stack([lp(L, 1400) * e, lp(R, 1400) * e], 1) / (2 * len(ms)) * vel


def pluck(m, dur=0.25, vel=1.0):
    n = int(dur * SR)
    f = mtof(m)
    x = (sq(f, n, 0.3) * 0.5 + sine(f * 2, n) * 0.3) * expdec(n, 0.09)
    return lp(x, 5000) * vel


def lead(m, dur, vel=1.0):
    n = int(dur * SR)
    f = mtof(m) * (1 + 0.004 * np.sin(2 * np.pi * 6 * t_(n)) * np.clip(t_(n) / 0.15, 0, 1))
    x = sq(f, n, 0.25) * 0.5 + saw(f * 1.005, n) * 0.4
    return lp(x, 3800) * env_adsr(n, 0.01, 0.1, 0.75, 0.08) * vel


def crash(vel=1.0, dur=2.2):
    n = int(dur * SR)
    x = hp(noise(n), 3000, 2) * expdec(n, 0.6)
    ring = sum(sine(f, n) for f in [3120, 4310, 5870, 7020]) * expdec(n, 0.4) * 0.05
    return (x * 0.5 + ring) * vel


# ---------------------------------------------------------------------------------------------
# Sound effects (one function per name used by the page)
# ---------------------------------------------------------------------------------------------
def fx_pop(ev, r):
    n = int(0.09 * SR)
    f0 = 380 * (1 + 0.25 * r.random())
    f = f0 + f0 * 1.4 * (1 - np.exp(-t_(n) / 0.02))
    return sine(f, n) * expdec(n, 0.03) * 0.8


def fx_pop2(ev, r):
    n = int(0.12 * SR)
    f0 = 700 * (1 + 0.2 * r.random())
    f = f0 * (1 + 1.2 * (1 - np.exp(-t_(n) / 0.015)))
    return sine(f, n) * expdec(n, 0.035) * 0.7 + hp(noise(n, r), 4000) * expdec(n, 0.004) * 0.3


def fx_slam(ev, r):
    n = int(0.9 * SR)
    boom = sine(40 + 70 * np.exp(-t_(n) / 0.05), n) * expdec(n, 0.25)
    crack = lp(noise(n, r), 2500) * expdec(n, 0.03) * 0.8
    return np.tanh((boom * 1.2 + crack) * 1.5) * 0.9


def fx_impact(ev, r):
    return np.concatenate([fx_slam(ev, r), np.zeros(SR)]) * 0.9 + np.pad(crash(0.6, 1.9), (0, int(0.9 * SR) + SR - int(1.9 * SR)))[:int(1.9 * SR)]


def whoosh_core(r, dur, f0, f1, width=1.2):
    n = int(dur * SR)
    x = noise(n, r)
    out = np.zeros(n)
    blk = 512
    for i in range(0, n, blk):
        k = i / n
        fc = f0 * (f1 / f0) ** k
        seg = x[i:i + blk]
        out[i:i + blk] = seg
    # swept band via block filtering
    y = np.zeros(n)
    zi = None
    for i in range(0, n, blk):
        k = i / n
        fc = f0 * (f1 / f0) ** k
        b, a = signal.butter(2, [fc / (1 + width), min(fc * (1 + width), SR / 2 - 200)], 'bandpass', fs=SR)
        if zi is None:
            zi = np.zeros(max(len(a), len(b)) - 1)
        seg, zi = signal.lfilter(b, a, x[i:i + blk], zi=zi)
        y[i:i + blk] = seg
    e = np.sin(np.pi * np.clip(t_(n) / dur, 0, 1)) ** 2
    return y * e


def fx_whoosh(ev, r):
    return whoosh_core(r, 0.38, 500, 2600) * 1.4


def fx_whooshBig(ev, r):
    a = whoosh_core(r, 0.6, 300, 3000, 1.5)
    b = whoosh_core(np.random.default_rng(r.integers(1e6)), 0.6, 320, 2800, 1.5)
    return np.stack([a, b], 1) * 1.5


def fx_whooshUp(ev, r):
    return whoosh_core(r, 0.5, 300, 5000) * 1.4


def fx_swish(ev, r):
    return whoosh_core(r, 0.16, 1500, 5000, 0.8) * 1.6


def fx_thud(ev, r):
    n = int(0.35 * SR)
    return np.tanh((sine(38 + 45 * np.exp(-t_(n) / 0.04), n) * expdec(n, 0.1) + lp(noise(n, r), 800) * expdec(n, 0.02) * 0.6) * 1.5) * 0.9


def bell_tone(f0, dur, partials=((1, 1, 1.2), (2.76, 0.5, 0.6), (5.4, 0.3, 0.35), (8.93, 0.18, 0.2)), r=None):
    n = int(dur * SR)
    x = sum(a * sine(f0 * p, n) * expdec(n, tau) for p, a, tau in partials)
    strike = hp(noise(n, r), 2000) * expdec(n, 0.005) * 0.4
    return x + strike


def fx_bell(ev, r):
    return bell_tone(740, 1.6, r=r) * 0.45


def fx_bell3(ev, r):
    one = bell_tone(740, 1.6, r=r) * 0.45
    out = np.zeros(int(2.3 * SR))
    for k, d in enumerate([0, 0.26, 0.52]):
        i = int(d * SR)
        out[i:i + len(one)] += one[:len(out) - i] * (1 if k < 2 else 1.1)
    return out


def fx_cashier(ev, r):
    n = int(1.0 * SR)
    rattle = np.zeros(n)
    for d in [0, 0.03, 0.055, 0.085]:
        i = int(d * SR)
        m = int(0.02 * SR)
        rattle[i:i + m] += bp(noise(m, r), 1500, 6000) * expdec(m, 0.004)
    ding = np.zeros(n)
    i = int(0.1 * SR)
    b = bell_tone(2093, 0.9, ((1, 1, 0.5), (1.5, 0.5, 0.35), (2.0, 0.35, 0.25), (3.01, 0.2, 0.15)), r)
    ding[i:i + len(b)] += b[:n - i]
    i2 = int(0.19 * SR)
    b2 = bell_tone(2637, 0.8, ((1, 1, 0.45), (1.5, 0.4, 0.3), (2.0, 0.3, 0.2)), r)
    ding[i2:i2 + len(b2)] += b2[:n - i2] * 0.8
    return rattle * 0.6 + ding * 0.35


def fx_stretch(ev, r):
    n = int(0.45 * SR)
    f = 180 * (2.6 ** (t_(n) / 0.45)) * (1 + 0.03 * np.sin(2 * np.pi * 18 * t_(n)))
    return lp(saw(f, n), 2200) * env_adsr(n, 0.02, 0.1, 0.8, 0.12) * 0.35


def fx_boing(ev, r):
    n = int(0.6 * SR)
    tt = t_(n)
    f = 170 + 260 * (1 - np.exp(-tt / 0.03)) + 60 * np.sin(2 * np.pi * 11 * tt) * np.exp(-tt / 0.25)
    return (sine(f, n) + 0.3 * sine(2 * f, n)) * expdec(n, 0.22) * 0.55


def fx_drumroll(ev, r):
    dur = ev.get('dur', 2.0)
    n = int((dur + 0.6) * SR)
    out = np.zeros(n)
    rate = 26
    hits = int(dur * rate)
    for k in range(hits):
        i = max(0, int((k / rate + r.normal(0, 0.002)) * SR))
        v = 0.25 + 0.75 * (k / hits) ** 1.5
        h = bp(noise(int(0.08 * SR), r), 1200, 7000) * expdec(int(0.08 * SR), 0.02)
        tone = sine(200, len(h)) * expdec(len(h), 0.03) * 0.5
        out[i:i + len(h)] += (h + tone)[:max(0, n - i)] * v
    return out * 0.5


def fx_riser(ev, r):
    dur = ev.get('dur', 2.0)
    n = int(dur * SR)
    tt = t_(n)
    f = 180 * (12 ** (tt / dur))
    tone = (saw(f, n) + saw(f * 1.01, n)) * 0.2
    nz = hp(noise(n, r), 2000) * 0.5
    e = (tt / dur) ** 2
    return lp(tone + nz, 9000) * e * 0.7


def fx_crowd(ev, r):
    dur = ev.get('dur', 2.0)
    n = int((dur + 0.8) * SR)
    tt = t_(n)
    e = np.clip(tt / 0.25, 0, 1) * np.clip((dur + 0.8 - tt) / 0.8, 0, 1)
    out = np.zeros((n, 2))
    for c in range(2):
        rr = np.random.default_rng(r.integers(1e9))
        roar = bp(rr.standard_normal(n), 350, 2600) * (0.6 + 0.4 * lp(rr.standard_normal(n), 4) * 3)
        claps = np.zeros(n)
        k = int(dur * 180)
        pos = rr.integers(0, n - 2000, k)
        cl = bp(rr.standard_normal(int(0.012 * SR)), 1000, 5000) * expdec(int(0.012 * SR), 0.003)
        for p in pos:
            claps[p:p + len(cl)] += cl * rr.uniform(0.3, 1)
        whistle = np.zeros(n)
        if dur > 1.5:
            i = int(rr.uniform(0.2, 0.6) * SR)
            m = int(0.5 * SR)
            fw = 2400 + 500 * np.sin(np.linspace(0, np.pi, m))
            whistle[i:i + m] = sine(fw, m) * np.sin(np.linspace(0, np.pi, m)) * 0.15
        out[:, c] = (roar * 0.5 + claps * 0.6 + whistle) * e
    return out * 0.5


def fx_sparkle(ev, r):
    n = int(0.6 * SR)
    out = np.zeros(n)
    notes = [84, 88, 91, 93, 96, 100]
    for k in range(6):
        i = int(k * 0.05 * SR)
        m = n - i
        out[i:] += sine(mtof(notes[k]), m) * expdec(m, 0.08) * 0.25
    return out


def fx_stamp(ev, r):
    n = int(0.2 * SR)
    return np.tanh((sine(110 * np.exp(-t_(n) / 0.2), n) * expdec(n, 0.04) + lp(noise(n, r), 1500) * expdec(n, 0.015)) * 2) * 0.8


def fx_punch(ev, r):
    n = int(0.3 * SR)
    body = sine(60 + 90 * np.exp(-t_(n) / 0.02), n) * expdec(n, 0.07)
    smack = bp(noise(n, r), 800, 5000) * expdec(n, 0.012)
    return np.tanh((body + smack * 1.2) * 2.2) * 0.75


def fx_punchBig(ev, r):
    p = fx_punch(ev, r)
    s = fx_slam(ev, r)
    out = np.zeros(max(len(p), len(s)))
    out[:len(p)] += p
    out[:len(s)] += s * 0.8
    return out


def fx_ticks(ev, r):
    k, gap = ev.get('n', 8), ev.get('gap', 0.035)
    m = int(0.02 * SR)
    one = (sine(2400, m) * 0.5 + bp(noise(m, r), 2000, 8000) * 0.5) * expdec(m, 0.004)
    out = np.zeros(int((k * gap + 0.05) * SR))
    for j in range(k):
        i = int(j * gap * SR)
        out[i:i + m] += one * (0.7 + 0.3 * (j % 2))
    return out * 0.6


def fx_rise(ev, r):
    dur, f0, f1 = ev.get('dur', 1.0), ev.get('f0', 300), ev.get('f1', 900)
    n = int(dur * SR)
    tt = t_(n)
    f = f0 * (f1 / f0) ** (1 - (1 - tt / dur) ** 2)
    x = (sine(f, n) + 0.3 * sq(f, n)) * (1 + 0.2 * np.sin(2 * np.pi * 25 * tt))
    return lp(x, 4000) * env_adsr(n, 0.02, 0.1, 0.8, 0.1) * 0.3


def fx_clank(ev, r):
    return bell_tone(520, 0.5, ((1, 1, 0.12), (2.3, 0.6, 0.08), (3.9, 0.4, 0.05), (5.7, 0.3, 0.04)), r) * 0.4


def fx_rocket(ev, r):
    dur = ev.get('dur', 1.5)
    n = int(dur * SR)
    tt = t_(n)
    roar = lp(noise(n, r), 900) * 1.5 + bp(noise(n, r), 1500, 6000) * 0.5
    whistle = sine(600 * (3 ** (tt / dur)), n) * 0.08
    e = np.clip(tt / 0.15, 0, 1) * np.clip((dur - tt) / 0.5, 0, 1)
    return np.tanh((roar + whistle) * 1.2) * e * 0.6


def fx_sizzle(ev, r):
    dur = ev.get('dur', 2.0)
    n = int(dur * SR)
    base = hp(noise(n, r), 3500) * 0.25
    pops = np.zeros(n)
    for p in r.integers(0, n - 500, int(dur * 70)):
        pops[p:p + 300] += noise(300, r) * expdec(300, 0.001) * r.uniform(0.2, 1)
    e = np.clip(t_(n) / 0.1, 0, 1) * np.clip((dur - t_(n)) / 0.3, 0, 1)
    return (base * (0.7 + 0.3 * lp(noise(n, r), 8) * 4) + hp(pops, 2000)) * e * 0.6


def fx_click(ev, r):
    m = int(0.04 * SR)
    return (sine(3200, m) * 0.4 + hp(noise(m, r), 3000) * 0.6) * expdec(m, 0.005) * 0.7


def fx_magsnap(ev, r):
    c = fx_click(ev, r) * 1.2
    b = bell_tone(1800, 0.25, ((1, 1, 0.05), (2.4, 0.5, 0.03)), r) * 0.2
    w = fx_swish(ev, r) * 0.4
    out = np.zeros(int(0.3 * SR))
    for y in (c, b, w):
        out[:len(y)] += y[:len(out)]
    return out


def fx_flip3(ev, r):
    out = np.zeros(int(1.3 * SR))
    for k in range(3):
        w = whoosh_core(r, 0.3, 700 + k * 200, 2600 + k * 300) * 1.4
        i = int(k * 0.35 * SR)
        out[i:i + len(w)] += w
    return out


def fx_crash(ev, r):
    n = int(1.4 * SR)
    glass = hp(noise(n, r), 2500) * expdec(n, 0.25) * 0.6
    ring = sum(sine(f, n) * expdec(n, 0.2 + 0.2 * r.random()) for f in r.uniform(2000, 7000, 8)) * 0.06
    thud = fx_thud(ev, r)
    out = glass + ring
    out[:len(thud)] += thud
    return out * 0.8


def fx_glitch(ev, r):
    n = int(0.4 * SR)
    out = np.zeros(n)
    i = 0
    while i < n:
        m = int(r.uniform(0.01, 0.05) * SR)
        out[i:i + m] = sq(r.uniform(200, 1800), m)[:max(0, n - i)] * r.uniform(0.2, 0.6)
        i += m + int(r.uniform(0, 0.02) * SR)
    return np.round(out * 6) / 6 * 0.35


def fx_censor(ev, r):
    n = int(0.4 * SR)
    return sine(1000, n) * env_adsr(n, 0.005, 0.01, 1, 0.01) * 0.3


def fx_scratch(ev, r):
    n = int(0.45 * SR)
    tt = t_(n)
    speed = np.sin(2 * np.pi * 7 * tt) * np.exp(-tt / 0.25)  # back-and-forth
    f = 300 + 900 * np.abs(speed)
    tone = saw(f, n) * 0.3 + bp(noise(n, r), 400, 4000) * np.abs(speed)
    return lp(tone, 5000) * env_adsr(n, 0.005, 0.05, 0.9, 0.1) * 0.8


def fx_clunk(ev, r):
    n = int(0.3 * SR)
    return np.tanh((sine(150 * np.exp(-t_(n) / 0.1), n) * expdec(n, 0.06) + bp(noise(n, r), 300, 2000) * expdec(n, 0.02)) * 2) * 0.7


def fx_tada(ev, r):
    a = brass([65, 69, 72, 77], 0.14, 1.0, 5000)
    b = brass([65, 69, 72, 77], 0.9, 1.0, 5000)
    out = np.zeros((int(1.2 * SR), 2))
    out[:len(a)] += a
    i = int(0.18 * SR)
    out[i:i + len(b)] += b
    return out * 1.3


def fx_crickets(ev, r):
    dur = ev.get('dur', 1.5)
    n = int(dur * SR)
    out = np.zeros((n, 2))
    for c, (f, off) in enumerate([(4700, 0.0), (5100, 0.23)]):
        t0 = off
        while t0 < dur - 0.1:
            for k in range(3):
                i = int((t0 + k * 0.045) * SR)
                m = int(0.03 * SR)
                if i + m < n:
                    out[i:i + m, c] += sine(f, m) * np.sin(np.linspace(0, np.pi, m)) * 0.25
            t0 += 0.5
    return out


def fx_printer(ev, r):
    dur = ev.get('dur', 0.3)
    n = int(dur * SR)
    tt = t_(n)
    gate = (np.sin(2 * np.pi * 42 * tt) > 0).astype(float)
    x = (sq(118, n) * 0.4 + bp(noise(n, r), 1500, 5000) * 0.6) * gate
    return lp(x, 4500) * env_adsr(n, 0.005, 0.02, 1, 0.02) * 0.3


def fx_point(ev, r):
    n = int(0.6 * SR)
    a = sine(mtof(84), n) * expdec(n, 0.2)
    i = int(0.07 * SR)
    b = np.zeros(n)
    b[i:] = sine(mtof(91), n - i) * expdec(n - i, 0.25)
    return (a + b) * 0.25


def fx_ding(ev, r):
    n = int(1.2 * SR)
    return (sine(1760, n) * expdec(n, 0.45) + sine(2640, n) * expdec(n, 0.3) * 0.5) * 0.3


def fx_fanfare(ev, r):
    notes = [(0, 65, 0.12), (0.13, 69, 0.12), (0.26, 72, 0.12), (0.39, 77, 1.3)]
    out = np.zeros((int(2.0 * SR), 2))
    for d, m, L in notes:
        y = brass([m, m - 12 + 7 if m == 77 else m - 5], L, 1.0, 5500)
        i = int(d * SR)
        out[i:i + len(y)] += y
    y = brass([65, 69, 72], 1.3, 0.8, 4000)
    out[int(0.39 * SR):int(0.39 * SR) + len(y)] += y
    return out * 1.4


def fx_sadtrombone(ev, r):
    notes = [(0.0, 58, 0.36), (0.42, 57, 0.36), (0.84, 56, 0.36), (1.26, 55, 1.2)]
    out = np.zeros(int(2.8 * SR))
    for d, m, L in notes:
        n = int(L * SR)
        tt = t_(n)
        vib = 1 + (0.012 * np.sin(2 * np.pi * 5 * tt) * np.clip((tt - 0.3) / 0.3, 0, 1) if L > 1 else 0)
        f = mtof(m) * vib * (1 - 0.01 * np.exp(-tt / 0.05))
        x = saw(f, n) * 0.6 + sq(f, n, 0.4) * 0.2
        # "wah": lowpass opens then closes like a plunger mute
        wah = 500 + 1600 * np.sin(np.pi * np.clip(tt / min(L, 0.36), 0, 1)) ** 2
        y = np.zeros(n)
        blk = 256
        zi = None
        for i in range(0, n, blk):
            b, a = signal.butter(2, wah[i], 'lowpass', fs=SR)
            if zi is None:
                zi = np.zeros(2)
            seg, zi = signal.lfilter(b, a, x[i:i + blk], zi=zi)
            y[i:i + blk] = seg
        y *= env_adsr(n, 0.03, 0.05, 0.9, 0.08 if L < 1 else 0.4)
        i = int(d * SR)
        out[i:i + n] += y
    return out * 0.7


def fx_powerdown(ev, r):
    n = int(0.7 * SR)
    tt = t_(n)
    f = 900 * (0.06 ** (tt / 0.7))
    return (sine(f, n) * 0.5 + sq(f, n) * 0.1) * np.clip((0.7 - tt) / 0.7, 0, 1) * 0.5


FX = {k[3:]: v for k, v in globals().items() if k.startswith('fx_')}


# ---------------------------------------------------------------------------------------------
# Music arrangement
# ---------------------------------------------------------------------------------------------
PROG = [  # (bass root, stab voicing, pad voicing)
    (41, [65, 69, 72], [53, 57, 60, 65]),   # F
    (38, [62, 65, 69], [50, 57, 62, 65]),   # Dm
    (46, [62, 65, 70], [50, 58, 62, 65]),   # Bb
    (36, [64, 67, 72], [52, 55, 60, 64]),   # C
]
BASS_PAT = [0, None, 12, 0, None, 0, 12, 7]    # eighth notes
STAB_STEPS = [2, 7, 10, 14]                    # sixteenth steps


def music(cues, events, total):
    bus = {k: Bus(total) for k in ['drums', 'bass', 'keys', 'lead']}
    sc = cues['scenes']
    L = lambda i: cues['lines'][i]['start']
    t_title, t_n15 = sc['title']['start'], L('n15')
    t_fight_back = next(m['until'] for m in events['music'] if m['type'] == 'stop' and abs(m['t'] - (t_n15 - 0.08)) < 0.2)
    t_verdict, t_win, t_end = sc['verdict']['start'], L('a08') - 0.1, sc['end']['start']

    def section(t):
        if t < t_title:
            return 'intro'
        if t < t_n15 - 0.08:
            return 'groove'
        if t < t_fight_back:
            return 'stop'
        if t < t_verdict:
            return 'groove2'
        if t < t_win:
            return 'tension'
        if t < t_end:
            return 'fanfare'
        return 'end'

    nsteps = int(total / STEP) + 1
    for s in range(nsteps):
        t = s * STEP
        sec = section(t)
        bar = int(t // (4 * BEAT))
        st = s % 16
        root, stab, padv = PROG[bar % 4]
        phraseB = (bar // 8) % 2 == 1
        if sec == 'intro':
            # tension: soft pulse on F, ticking hats, rising pad; builds over the intro
            k = t / t_title
            if st in (0, 8):
                bus['drums'].add(t, kick(0.35 + 0.4 * k))
            if st % 2 == 0:
                bus['drums'].add(t, hat(0.12 + 0.2 * k), pan=0.3)
            if st % 4 == 0:
                bus['bass'].add(t, bass_note(29, STEP * 3.5, 0.5 + 0.3 * k))
            if st == 0 and bar % 2 == 0:
                bus['keys'].add(t, pad([53, 57, 60, 62] if bar % 4 == 0 else [50, 53, 57, 62], BEAT * 8, 0.35 + 0.3 * k))
            if st in (0, 3, 6, 10, 12) and t > 2:
                bus['keys'].add(t, pluck([77, 72, 74, 69, 72][[0, 3, 6, 10, 12].index(st)], 0.3, 0.12 + 0.12 * k), pan=-0.3 + 0.6 * (st % 2))
        elif sec in ('groove', 'groove2', 'fanfare'):
            hot = sec != 'groove' or phraseB
            if st % 4 == 0:
                bus['drums'].add(t, kick(0.95))
            if sec == 'groove2' and st == 14:
                bus['drums'].add(t, kick(0.5))
            if st in (4, 12):
                bus['drums'].add(t, clap(0.75), pan=0.05)
            if hot or st % 2 == 0:
                v = 0.32 if st % 4 == 2 else 0.18
                bus['drums'].add(t, hat(v, open_=(st == 14 and hot)), pan=0.35)
            if st % 2 == 0:
                off = BASS_PAT[st // 2]
                if off is not None:
                    bus['bass'].add(t, bass_note(root + off, STEP * 1.8, 0.9 if st == 0 else 0.75))
            if st in STAB_STEPS:
                bus['keys'].add(t, brass(stab, STEP * 1.6, 0.55))
            if hot and st == 0:
                bus['keys'].add(t, pad(padv, BEAT * 4, 0.35))
            if hot and sec != 'fanfare':
                arp = [stab[0] + 12, stab[1] + 12, stab[2] + 12, stab[1] + 12]
                bus['keys'].add(t, pluck(arp[st % 4], 0.2, 0.1), pan=0.5 if st % 2 else -0.5)
            if st == 0 and bar % 8 == 0:
                bus['drums'].add(t, crash(0.3))
        elif sec == 'tension':
            if st in (0, 3):
                bus['drums'].add(t, kick(0.8))
            if st % 4 == 0:
                bus['bass'].add(t, bass_note(38, STEP * 3, 0.7))
            if st == 0:
                bus['keys'].add(t, pad([50, 57, 62, 65], BEAT * 4, 0.45))
    # title: crash + brass fanfare on the hit
    bus['drums'].add(t_title, crash(0.9, 2.5))
    bus['drums'].add(t_title, kick(1.0))
    for d, m, Ln in [(0, 65, 0.2), (0.25, 69, 0.2), (0.5, 72, 0.2), (0.75, 77, 1.2)]:
        bus['lead'].add(t_title + d, brass([m, m + 4 if m != 77 else m - 5], Ln, 1.0, 5500), 0.9)
    # fighters wipe: groove kicks in with a crash
    bus['drums'].add(sc['fighters']['start'], crash(0.6))
    # fanfare lead melody over the winner
    mel = [(0, 77, 1), (1, 81, 1), (2, 84, 2), (4, 82, 1), (5, 81, 1), (6, 79, 2), (8, 77, 1), (9, 79, 1), (10, 81, 1), (11, 84, 1), (12, 82, 2), (14, 81, 1), (15, 79, 1)]
    t0 = t_win + 0.35
    for b, m, d in mel:
        tt = t0 + b * BEAT
        if tt < t_end:
            bus['lead'].add(tt, lead(m, d * BEAT * 0.95, 0.5), pan=0.0)
    bus['drums'].add(t_win, crash(0.9, 2.5))
    # ending: big final hit that rings out
    bus['drums'].add(t_end, crash(1.0, 3.2))
    bus['drums'].add(t_end, kick(1.0))
    bus['bass'].add(t_end, bass_note(41, 2.4, 1.0))
    bus['keys'].add(t_end, brass([65, 69, 72, 77], 2.6, 1.0, 5000) * env_adsr(int(2.6 * SR), 0.01, 0.2, 0.7, 1.6)[:, None], 1.3)
    bus['keys'].add(t_end, pad([53, 57, 60, 65, 69], 3.3, 0.6))
    return bus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--stems', action='store_true')
    a = ap.parse_args()
    cues = json.load(open(os.path.join(a.build, 'cues.json')))
    events = json.load(open(os.path.join(a.build, 'events.json')))
    total = cues['total']
    n = int(total * SR)

    # --- voice
    vo = Bus(total)
    vo_env = np.zeros(len(vo.x))
    arena_ir = reverb_ir(2.2, 0.55, seed=11, pre=0.03)
    for k, ln in cues['lines'].items():
        x, sr = sf.read(os.path.join(a.build, 'vo', k + '.wav'), dtype='float64')
        x = signal.resample_poly(x, SR, sr)
        x = hp(x, 90)
        x = norm(x, 0.5)
        if ln['v'] == 'A':  # ring announcer: warmer, a little driven, slapback + arena
            x = np.tanh(lp(x, 7000) * 2.2) * 0.45
            st = np.stack([x, x], 1)
            slap = np.zeros_like(st)
            d = int(0.11 * SR)
            slap[d:] = st[:-d] * 0.22
            wet = convolve_st(np.pad(st, ((0, len(arena_ir)), (0, 0))), arena_ir)[:len(st) + len(arena_ir)]
            y = np.pad(st + slap, ((0, len(arena_ir)), (0, 0))) + wet * 0.12
            vo.add(ln['start'], y, 1.0)
        else:
            vo.add(ln['start'], x, 1.0)
        i = int(ln['start'] * SR)
        vo_env[i:i + len(x)] = np.maximum(vo_env[i:i + len(x)], 1.0)
    room = reverb_ir(0.7, 0.18, seed=5, pre=0.008)
    vo.x = vo.x + convolve_st(vo.x, room) * 0.06

    # --- music, with ducking under the voice and the page's stop/duck cues
    mb = music(cues, events, total)
    mus = mb['drums'].x * 0.9 + mb['bass'].x * 0.55 + mb['keys'].x * 0.5 + mb['lead'].x * 0.45
    mus = mus + convolve_st(mus, reverb_ir(1.4, 0.35, seed=9)) * 0.1
    duck = signal.lfilter([1 - np.exp(-1 / (0.06 * SR))], [1, -np.exp(-1 / (0.06 * SR))], vo_env)
    duck = np.maximum(duck, signal.lfilter([1 - np.exp(-1 / (0.25 * SR))], [1, -np.exp(-1 / (0.25 * SR))], vo_env[::-1])[::-1] * 0.0)
    gain = 1 - 0.72 * np.clip(duck, 0, 1)
    for m in events['music']:
        i0, i1 = int(m['t'] * SR), int(m['until'] * SR)
        g = 0.0 if m['type'] == 'stop' else m.get('gain', 0.4)
        f = int(0.015 * SR)
        gain[i0:i1] *= g
        gain[max(0, i0 - f):i0] *= np.linspace(1, g, min(f, i0))
        gain[i1:i1 + f] *= np.linspace(g, 1, f)
    mus *= gain[:, None]

    # --- sound effects
    fxb = Bus(total)
    missing = set()
    for ev in events['sfx']:
        fn = FX.get(ev['name'])
        if fn is None:
            missing.add(ev['name'])
            continue
        r = np.random.default_rng(int(ev['t'] * 1000) + len(ev['name']))
        y = fn(ev, r)
        pan = float(np.clip((r.random() - 0.5) * 0.3, -1, 1))
        fxb.add(ev['t'], y, ev.get('gain', 1.0) * 0.55, pan)
    if missing:
        print('missing sfx:', sorted(missing))
    fxb.x = fxb.x + convolve_st(fxb.x, reverb_ir(1.0, 0.25, seed=13)) * 0.12
    fxb.x *= (1 - 0.4 * np.clip(duck, 0, 1))[:, None]  # effects step back a little while someone talks

    mix = vo.x * 1.0 + mus * 0.3 + fxb.x * 0.65
    mix = mix[:n + int(0.05 * SR)]
    # master: gentle glue compression + limiter, then loudness to -14 LUFS
    meter = pyloudnorm.Meter(SR)
    lufs = meter.integrated_loudness(mix)
    mix *= 10 ** ((-14.0 - lufs) / 20)
    # look-ahead peak limiter at -1 dBFS
    ceiling = 10 ** (-1.2 / 20)
    peak = np.max(np.abs(mix), 1)
    need = np.minimum(1, ceiling / np.maximum(peak, 1e-9))
    la = int(0.003 * SR)
    need = np.minimum.reduce([np.roll(need, -k) for k in range(0, la, 16)])
    rel = np.exp(-1 / (0.08 * SR))
    g = signal.lfilter([1 - rel], [1, -rel], need)
    g = np.minimum(g, need)
    mix *= g[:, None]
    lufs2 = meter.integrated_loudness(mix)
    print(f'loudness {lufs2:.1f} LUFS, peak {20 * np.log10(np.max(np.abs(mix))):.1f} dBFS, length {len(mix) / SR:.2f}s')
    fade = int(0.8 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    sf.write(os.path.join(a.build, 'soundtrack.wav'), mix.astype(np.float32), SR, subtype='PCM_24')
    if a.stems:
        for k, x in [('vo', vo.x), ('music', mus * 0.3), ('sfx', fxb.x * 0.65)]:
            sf.write(os.path.join(a.build, f'stem_{k}.wav'), (x[:n] * 0.5).astype(np.float32), SR)


if __name__ == '__main__':
    main()
