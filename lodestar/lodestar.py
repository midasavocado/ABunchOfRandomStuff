#!/usr/bin/env python3
"""
LODESTAR — an original instrumental theme, composed and synthesized from scratch.

Every sound here is generated from raw math (saw/sine oscillators, noise, filters)
with numpy/scipy. No samples, no soundfonts, no MIDI libraries.

The story (in seven movements, one continuous piece):
  I.   Home          (bars 1-8)   A clock ticks. An organ breathes. A piano remembers a tune.
  II.  Departure     (bars 9-16)  A lone horn takes up the theme. The journey begins.
  III. Ascent        (bars 17-24) Strings start running, drums wake, the theme climbs.
  IV.  The Storm     (bars 25-36) Everything turns dark and driving. Time speeds up.
  V.   Lost          (bars 37-40) An impact. Silence. A heartbeat. A broken fragment of the tune.
  VI.  Lodestar      (bars 41-56) The theme returns in MAJOR — the question finally answered.
  VII. Homecoming    (bars 57-64) The piano again, now at peace. The clock stops.

Usage: python3 lodestar.py   ->  writes lodestar.wav and lodestar.mp3 next to this file
"""
import os
import time
import wave

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, lfilter
from scipy.ndimage import maximum_filter1d

SR = 44100
BPM = 76.0
SPB = 60.0 / BPM
TWO_PI = 2 * np.pi
rng = np.random.default_rng(2026)

TOTAL_BARS = 64
TAIL = 9.0


def bt(bar, beat=0.0):
    """Absolute beat index for (1-indexed bar, beat within bar)."""
    return (bar - 1) * 4 + beat


def sec(beat_abs):
    return beat_abs * SPB


N = int((sec(bt(TOTAL_BARS + 1)) + TAIL) * SR)

# ----------------------------------------------------------------------------------------------
# pitch helpers
# ----------------------------------------------------------------------------------------------
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def midi(name):
    letter, rest, acc = name[0], name[1:], 0
    while rest and rest[0] in '#b':
        acc += 1 if rest[0] == '#' else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + PC[letter] + acc


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def chord_info(name):
    i, acc = 1, 0
    if len(name) > 1 and name[1] in '#b':
        acc = 1 if name[1] == '#' else -1
        i = 2
    q = name[i:]
    pc = (PC[name[0]] + acc) % 12
    third = 3 if q.startswith('m') else (5 if q == 'sus4' else 4)
    return pc, third


def above(pc, lo):
    return lo + ((pc - lo) % 12)


def pad_voicing(ch):
    pc, th = chord_info(ch)
    r = above(pc, 45)
    return [r, r + 7, r + 12, r + 12 + th, r + 19]


def bass_root(ch):
    return above(chord_info(ch)[0], 33)


def ost_notes(ch, lo=55):
    pc, th = chord_info(ch)
    r = above(pc, lo)
    return [r, r + 7, r + 12, r + 12 + th]


def arp_notes(ch):
    pc, _ = chord_info(ch)
    r = above(pc, 55)
    return [r, r + 7, r + 14, r + 7]


def choir_voicing(ch, high=False):
    pc, th = chord_info(ch)
    r = above(pc, 55)
    v = [r - 12, r, r + th, r + 7]
    if high:
        v += [r + 12, r + 12 + th]
    return v


# ----------------------------------------------------------------------------------------------
# buses
# ----------------------------------------------------------------------------------------------
def pan_gains(p):
    a = (p + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)


class Bus:
    def __init__(self, name, gain=1.0, send=0.3):
        self.name, self.gain, self.send = name, gain, send
        self.buf = np.zeros((2, N), dtype=np.float32)

    def add(self, t_sec, sig, pan=0.0, jitter=0.004):
        t_sec += rng.normal(0, jitter) if jitter else 0.0
        i = max(0, int(round(t_sec * SR)))
        gl, gr = pan_gains(pan)
        if sig.ndim == 1:
            sig = np.vstack([sig * gl, sig * gr])
        else:
            sig = sig * np.array([[gl], [gr]])
        n = min(sig.shape[1], N - i)
        if n > 0:
            self.buf[:, i:i + n] += sig[:, :n].astype(np.float32)


BUSES = {}


def bus(name, gain=1.0, send=0.3):
    if name not in BUSES:
        BUSES[name] = Bus(name, gain, send)
    return BUSES[name]


# ----------------------------------------------------------------------------------------------
# DSP building blocks
# ----------------------------------------------------------------------------------------------
def tarr(n):
    return np.arange(n) / SR


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos'), x, axis=-1)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x, axis=-1)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos'), x, axis=-1)


def env_ar(n, dur, a, r):
    t = tarr(n)
    if a > 0:
        att = np.clip(t / a, 0, 1)
        att = 0.5 - 0.5 * np.cos(np.pi * att)
    else:
        att = np.ones(n)
    rel = np.ones(n)
    m = t > dur
    rel[m] = np.exp(-(t[m] - dur) * 6.9 / max(r, 1e-3))
    return att * rel


def dyn_curve(n, dur, vel, cresc):
    if cresc is None:
        return vel
    t = tarr(n)
    v0, v1 = cresc
    x = np.clip(t / max(dur, 1e-3), 0, 1)
    return v0 + (v1 - v0) * x ** 1.6


def saw_blep(freq, n, ph0=0.0):
    """Band-limited sawtooth (polyBLEP). freq may be scalar or per-sample array."""
    dt = np.broadcast_to(np.asarray(freq, dtype=np.float64) / SR, (n,))
    ph = (ph0 + np.cumsum(dt)) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = ph > 1.0 - dt
    x = (ph[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


def saw_ensemble(f, n, dur, voices, detune, vib, glide=None):
    t = tarr(n)
    out = np.zeros((2, n))
    vib_env = np.clip((t - 0.25) / 0.9, 0, 1)
    gl = 1.0 if glide is None else glide ** np.clip(t / max(dur, 1e-3), 0, 1)
    for ch in range(2):
        for v in range(voices):
            c = (detune * ((v / (voices - 1)) * 2 - 1) if voices > 1 else 0.0) + rng.normal(0, 1.5)
            vr = rng.uniform(4.6, 5.9)
            fr = f * 2 ** (c / 1200) * gl * (1 + vib * vib_env * np.sin(TWO_PI * vr * t + rng.uniform(0, TWO_PI)))
            out[ch] += saw_blep(fr, n, rng.random())
    return out / np.sqrt(voices)


# ----------------------------------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------------------------------
def strings(f, dur, vel, a=0.35, r=0.9, voices=5, detune=9.0, vib=0.0035, cutoff=None,
            glide=None, cresc=None):
    n = int((dur + r) * SR)
    out = saw_ensemble(f, n, dur, voices, detune, vib, glide)
    peak_v = vel if cresc is None else max(cresc)
    if cutoff is None:
        cutoff = min(9000, 700 + 2.2 * f + 3800 * peak_v)
    out = lp(out, cutoff)
    out = lp(out, cutoff * 1.6)
    out = hp(out, 70)
    return out * env_ar(n, dur, a, r) * dyn_curve(n, dur, vel, cresc)


def spic(f, vel, dur=0.14):
    """Short spiccato / staccato string note."""
    n = int((dur + 0.22) * SR)
    t = tarr(n)
    out = saw_ensemble(f, n, dur, 3, 7.0, 0.0)
    out = lp(out, min(10000, 900 + 2.0 * f + 5500 * vel))
    out = hp(out, 60)
    e = env_ar(n, dur, 0.005, 0.12) * (0.55 + 0.45 * np.exp(-t / 0.05))
    return out * e * vel


def brass(f, dur, vel, a=0.08, r=0.45, bright=1.0, voices=3, spread=0.5, cresc=None):
    n = int((dur + r) * SR)
    t = tarr(n)
    env = env_ar(n, dur, a, r)
    dyn = dyn_curve(n, dur, vel, cresc)
    dyn_arr = np.broadcast_to(dyn, (n,))
    B = (1.1 + 6.5 * bright * dyn_arr) * (0.3 + 0.7 * env ** 0.7)
    K = int(max(1, min(40, 14000 / f, 1 + 6 * B.max())))
    out = np.zeros((2, n))
    scoop = 1 - 0.018 * np.exp(-t / 0.045)
    for v in range(voices):
        c = rng.normal(0, 4.0)
        vib = 1 + 0.0025 * np.sin(TWO_PI * rng.uniform(4.6, 5.6) * t + rng.uniform(0, TWO_PI)) \
            * np.clip((t - 0.5) / 0.6, 0, 1)
        ph = TWO_PI * np.cumsum(f * 2 ** (c / 1200) * scoop * vib) / SR
        y = np.zeros(n)
        for k in range(1, K + 1):
            y += np.sin(k * ph + rng.uniform(0, TWO_PI)) * np.exp(-(k - 1) / B) * k ** -0.35
        p = spread * ((v / (voices - 1)) * 2 - 1) if voices > 1 else 0.0
        gl, gr = pan_gains(p)
        out[0] += y * gl
        out[1] += y * gr
    out = np.tanh(out * 0.6) / 0.6  # a little brassy saturation
    return out * env * dyn / voices


ORGAN_FULL = [(0.5, 0.45), (1, 1.0), (2, 0.55), (3, 0.22), (4, 0.3), (6, 0.1), (8, 0.12)]
ORGAN_SOFT = [(0.5, 0.3), (1, 1.0), (2, 0.35), (4, 0.08)]


def organ(f, dur, vel, a=0.18, r=0.7, ranks=ORGAN_FULL, cresc=None):
    n = int((dur + r) * SR)
    t = tarr(n)
    out = np.zeros((2, n))
    for mult, amp in ranks:
        fr = f * mult
        if fr > 11000:
            continue
        for ch in range(2):
            det = 1 + rng.normal(0, 0.0007)
            out[ch] += amp * np.sin(TWO_PI * fr * det * t + rng.uniform(0, TWO_PI))
    chiff = lp(rng.normal(size=(2, n)) * np.exp(-t / 0.04), 3000) * 0.08
    out += chiff
    return out * env_ar(n, dur, a, r) * dyn_curve(n, dur, vel, cresc) * 0.35


def piano(f, dur, vel, bright=1.0, ring=None):
    ring = dur if ring is None else ring
    n = int((ring + 0.6) * SR)
    t = tarr(n)
    K = int(max(1, min(16, 9000 / f)))
    tau = float(np.clip(4.2 * (262.0 / f) ** 0.6, 0.8, 9.0))
    Bh = 2.5 + 6.0 * vel * bright
    y = np.zeros(n)
    for k in range(1, K + 1):
        fk = k * f * np.sqrt(1 + 0.00035 * k * k)
        ak = k ** -1.0 * np.exp(-(k - 1) / Bh)
        tk = tau / (1 + 0.35 * (k - 1))
        e = 0.6 * np.exp(-t / (0.18 * tk + 0.02)) + 0.4 * np.exp(-t / tk)
        y += ak * e * (np.sin(TWO_PI * fk * t + rng.uniform(0, TWO_PI))
                       + 0.7 * np.sin(TWO_PI * fk * 1.0006 * t + rng.uniform(0, TWO_PI)))
    ham = lp(rng.normal(size=n) * np.exp(-t / 0.003), 2500) * 0.12
    y += ham
    y *= np.clip(t / 0.002, 0, 1)
    y *= np.where(t > ring, np.exp(-(t - ring) / 0.15), 1.0)
    y = lp(y, 2500 + 4000 * vel * bright)
    return y * vel * 0.5


FORMANTS_AH = [(760, 1.0, 110), (1150, 0.55, 130), (2800, 0.22, 200), (3500, 0.1, 250)]


FORMANTS_OO = [(320, 1.0, 70), (870, 0.3, 100), (2240, 0.07, 160)]


def choir(f, dur, vel, a=0.7, r=1.4, cresc=None, formants=FORMANTS_AH):
    n = int((dur + r) * SR)
    raw = saw_ensemble(f, n, dur, 6, 14.0, 0.006)
    raw += rng.normal(size=(2, n)) * 0.08  # breath
    out = np.zeros_like(raw)
    for fc, amp, bw in formants:
        out += bp(raw, fc - bw, fc + bw) * amp
    out = lp(out, 5000)
    return out * env_ar(n, dur, a, r) * dyn_curve(n, dur, vel, cresc) * 1.6


def sub(f, dur, vel, r=0.4):
    n = int((dur + r) * SR)
    t = tarr(n)
    y = np.sin(TWO_PI * f * t) + 0.25 * np.sin(2 * TWO_PI * f * t) + 0.08 * np.sin(3 * TWO_PI * f * t)
    return y * env_ar(n, dur, 0.03, r) * vel


# ---- percussion ------------------------------------------------------------------------------
def taiko(vel, size=1.0):
    n = int(2.2 * SR)
    t = tarr(n)
    f = 48 / size + 75 * np.exp(-t / 0.035)
    ph = TWO_PI * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / (0.45 * size))
    skin = lp(rng.normal(size=n), 900) * np.exp(-t / 0.03) * 0.9
    slap = hp(rng.normal(size=n), 1500) * np.exp(-t / 0.006) * 0.25 * vel
    y = np.tanh(1.8 * (body + skin)) + slap
    return y * vel


def tom(freq, vel):
    n = int(1.0 * SR)
    t = tarr(n)
    f = freq * (1 + 0.5 * np.exp(-t / 0.03))
    body = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    skin = bp(rng.normal(size=n), 150, 2500) * np.exp(-t / 0.02) * 0.6
    return np.tanh(1.5 * (body + skin)) * vel


def timpani(freq, vel):
    n = int(2.5 * SR)
    t = tarr(n)
    y = np.zeros(n)
    for ratio, amp, dec in [(1, 1.0, 1.3), (1.5, 0.5, 0.9), (1.98, 0.3, 0.6), (2.44, 0.2, 0.45), (2.9, 0.1, 0.3)]:
        y += amp * np.sin(TWO_PI * freq * ratio * t + rng.uniform(0, TWO_PI)) * np.exp(-t / dec)
    y += lp(rng.normal(size=n), 1500) * np.exp(-t / 0.015) * 0.5
    y *= np.clip(t / 0.003, 0, 1)
    return y * vel * 0.6


def snare(vel):
    """Orchestral field-drum ensemble hit (three players, slightly apart)."""
    n = int(0.5 * SR)
    t = tarr(n)
    out = np.zeros(n)
    for _ in range(3):
        off = int(abs(rng.normal(0, 0.006)) * SR)
        noise = bp(rng.normal(size=n), 250, 7000) * np.exp(-t / 0.09)
        tone = np.sin(TWO_PI * 185 * t) * np.exp(-t / 0.04) * 0.6
        hit = noise + tone
        out[off:] += hit[:n - off] * rng.uniform(0.8, 1.0)
    return out * vel * 0.35


def crash(vel, length=3.5):
    n = int(length * SR)
    t = tarr(n)
    out = np.zeros((2, n))
    for ch in range(2):
        noise = hp(rng.normal(size=n), 3500) * np.exp(-t / (length / 4))
        ring = np.zeros(n)
        for _ in range(14):
            fr = rng.uniform(3000, 9500)
            ring += np.sin(TWO_PI * fr * t + rng.uniform(0, TWO_PI)) * np.exp(-t / rng.uniform(0.3, 1.2))
        out[ch] = noise + ring * 0.06
    out *= np.clip(t / 0.002, 0, 1)
    return lp(out, 12000) * vel * 0.35


def riser(length, vel):
    """Reverse cymbal / air swell that peaks at its end."""
    n = int(length * SR)
    x = np.linspace(0, 1, n)
    out = np.zeros((2, n))
    for ch in range(2):
        noise = hp(rng.normal(size=n), 2500)
        out[ch] = noise * x ** 3.5
    return out * vel * 0.3


def impact(vel):
    """Huge cinematic hit: taiko + sub drop + air burst."""
    n = int(6.0 * SR)
    t = tarr(n)
    f = 55 * np.exp(-t / 1.2) + 26
    subd = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / 1.6)
    air = lp(rng.normal(size=n), 1200) * np.exp(-t / 0.25) * 0.5
    y = np.zeros(n)
    tk = taiko(1.0, size=1.4)
    y[:len(tk)] += tk
    y += np.tanh(1.3 * subd) * 0.9 + air
    return y * vel


# A mechanical escapement, not a metronome: every beat is three tiny impacts
# (unlock, impulse, drop) ringing through the small metal modes of the movement
# and the wooden modes of the case. Tick and tock strike different parts.
TICK_MODES = {False: [(1870, 1.0, .0035), (3130, .8, .0025), (4410, .6, .002), (6120, .45, .0015), (7980, .3, .001)],
              True: [(1580, 1.0, .004), (2690, .8, .003), (3950, .55, .0022), (5380, .4, .0016), (7210, .25, .0011)]}
CASE_MODES = [(410, .5, .018), (960, .35, .012), (1330, .2, .009)]


def tick(vel, tock=False):
    n = int(0.07 * SR)
    t = tarr(n)
    y = np.zeros(n)
    for off, a in [(0.0, 0.45), (0.0026, 1.0), (0.0085, 0.3)]:
        i0 = max(0, int((off + rng.normal(0, 0.0002)) * SR))
        tt = t[:n - i0]
        click = np.zeros(n - i0)
        for f, amp, dec in TICK_MODES[tock]:
            f *= rng.uniform(0.985, 1.015)
            click += amp * np.sin(TWO_PI * f * tt + rng.uniform(0, TWO_PI)) * np.exp(-tt / dec)
        for f, amp, dec in CASE_MODES:
            click += 0.6 * amp * np.sin(TWO_PI * f * tt) * np.exp(-tt / dec)
        click += hp(rng.normal(size=n - i0) * np.exp(-tt / 0.0004), 2000) * 0.9
        y[i0:] += a * click * rng.uniform(0.85, 1.0)
    return hp(y, 250) * vel * (0.26 if tock else 0.3)


def shepard(dur, rate, fade_in):
    """An endlessly rising Shepard-Risset glissando."""
    n = int(dur * SR)
    t = tarr(n)
    out = np.zeros((2, n))
    for ch in range(2):
        for k in range(7):
            pos = (k + rate * t) % 7.0
            f = 55.0 * 2 ** pos * (1.0 + 0.002 * ch)
            amp = np.exp(-0.5 * ((pos - 3.3) / 1.25) ** 2)
            ph = TWO_PI * np.cumsum(f) / SR
            out[ch] += amp * (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph))
    env = np.clip(t / fade_in, 0, 1) ** 2
    env *= np.clip((dur - t) / 0.05, 0, 1)
    return out * env * 0.2


def space_wind(dur, vel):
    n = int(dur * SR)
    t = tarr(n)
    out = np.zeros((2, n))
    for ch in range(2):
        nz = rng.normal(size=n)
        a = bp(nz, 120, 500) * (0.6 + 0.4 * np.sin(TWO_PI * 0.11 * t + ch))
        b = bp(nz, 700, 1600) * (0.5 + 0.5 * np.sin(TWO_PI * 0.07 * t + 2 + ch)) * 0.35
        out[ch] = a + b
    env = np.clip(t / 2.5, 0, 1) * np.clip((dur - t) / 2.0, 0, 1)
    return out * env * vel


def heartbeat(vel):
    n = int(0.9 * SR)
    t = tarr(n)
    def thump(t0, v):
        tt = np.clip(t - t0, 0, None)
        f = 42 + 35 * np.exp(-tt / 0.03)
        s = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-tt / 0.11) * (t >= t0)
        return s * v
    return lp(thump(0.0, 1.0) + thump(0.3, 0.7), 400) * vel


def lead_phrase(notes, vel=0.8, glide=0.075, bright=1.0, reso=1.8, vib=0.0045, attack=0.05,
                drive=0.0, detune=6.0, sub_amt=0.22):
    """The Lodestar lead: a Vangelis-style analog voice. One continuous oscillator per phrase that
    glides between notes, with a resonant filter that blooms on every new note.
    notes: list of (start_sec, dur_sec, midi). Returns (stereo signal, start_sec)."""
    notes = sorted(notes)
    t0 = notes[0][0]
    t1 = max(s + d for s, d, m in notes) + 0.6
    n = int((t1 - t0) * SR)
    t = tarr(n)
    target = np.full(n, float(notes[0][2]))
    gate = np.zeros(n)
    onset = np.zeros(n)
    for s, d, m in notes:
        i0 = int((s - t0) * SR)
        i1 = min(n, int((s + d - t0) * SR))
        target[i0:] = m
        gate[i0:i1] = 1.0
        onset[i0:] = np.exp(-t[:n - i0] / 0.22)
    a = np.exp(-1.0 / (glide * SR))
    pitch = lfilter([1 - a], [1, -a], target - target[0]) + target[0]
    ga = np.exp(-1.0 / (attack * SR))
    env = lfilter([1 - ga], [1, -ga], gate)
    f = 440.0 * 2 ** ((pitch - 69) / 12) * (1 + vib * (1 - onset) * np.sin(TWO_PI * 5.2 * t))
    fc = f * (1.6 + 7.0 * bright * (0.35 + 0.65 * onset)) * (0.45 + 0.55 * env)
    out = np.zeros((2, n))
    K = int(max(4, min(44, 15000 / f.max())))
    for ch, det in ((0, -detune), (1, detune)):
        for dc in (det, -det * 0.35):
            ph = TWO_PI * np.cumsum(f * 2 ** (dc / 1200)) / SR + rng.uniform(0, TWO_PI)
            for k in range(1, K + 1):
                fk = k * f
                g = 1.0 / np.sqrt(1.0 + (fk / fc) ** 4) * (1.0 + reso * np.exp(-((fk - fc) / (0.3 * fc)) ** 2))
                out[ch] += np.sin(k * ph) * g * (fk < 16000) / k
        phs = TWO_PI * np.cumsum(f * 0.5) / SR
        for k in (1, 3, 5):
            out[ch] += sub_amt * np.sin(k * phs) / k
    out *= 0.45
    # the Lodestar voice: every note blooms from "oo" to "ah", like a wordless singer inside the synth
    oo = sum(bp(out, fc0 - bw, fc0 + bw) * amp for fc0, amp, bw in FORMANTS_OO)
    ah = sum(bp(out, fc0 - bw, fc0 + bw) * amp for fc0, amp, bw in FORMANTS_AH)
    w = np.clip(1.0 - onset * 1.25, 0.0, 1.0)
    out = 0.4 * out + 1.5 * (oo * (1.0 - w) + ah * w)
    if drive > 0:
        out = np.tanh(out * (1 + drive * 3)) / (1 + drive * 0.8)
    out = hp(out, 90)
    return out * env * (0.85 + 0.15 * onset) * vel, t0


def ratchet_click(vel):
    """one click of the crown's ratchet while the watch is wound: small, dry, bright"""
    n = int(0.03 * SR)
    t = tarr(n)
    y = np.zeros(n)
    for f, a, d in [(2100, 0.4, 0.002), (3300, 1.0, 0.0015), (5200, 0.7, 0.001), (7700, 0.45, 0.0008)]:
        y += a * np.sin(TWO_PI * f * rng.uniform(0.97, 1.03) * t + rng.uniform(0, TWO_PI)) * np.exp(-t / d)
    y += hp(rng.normal(size=n) * np.exp(-t / 0.0005), 3000) * 0.8
    return hp(y, 1200) * vel * 0.3


def pluck(f, vel, decay=0.7):
    """The clock's musical twin: a bright, glassy plucked synth."""
    n = int(decay * 4 * SR)
    t = tarr(n)
    K = int(max(3, min(24, 12000 / f)))
    y = np.zeros(n)
    for k in range(1, K + 1):
        y += np.sin(TWO_PI * k * f * t * (1 + 0.0004 * k * k) + rng.uniform(0, TWO_PI)) / k ** 1.3 \
            * np.exp(-t * (1.0 + 0.9 * k) / decay)
    y += 0.5 * np.sin(TWO_PI * f * 1.0015 * t) * np.exp(-t / decay)
    y *= np.clip(t / 0.0015, 0, 1)
    return y * vel * 0.4


def tamtam(vel, length=9.0):
    """A tam-tam: an initial bong, then a shimmering bloom where the highs swell in late."""
    n = int(length * SR)
    t = tarr(n)
    y = np.zeros(n)
    freqs = np.exp(rng.uniform(np.log(70), np.log(3500), 70))
    for f in freqs:
        bloom = 0.05 + 0.6 * (f / 3500) ** 0.7
        env = (1 - np.exp(-t / bloom)) * np.exp(-t / rng.uniform(2.5, 6.0))
        y += np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI)) * env * (f / 200) ** -0.4
    y += (2.0 * np.sin(TWO_PI * 92 * t) * np.exp(-t / 3.0) + 1.2 * np.sin(TWO_PI * 141 * t) * np.exp(-t / 2.2)) \
        * np.clip(t / 0.01, 0, 1)
    y += lp(rng.normal(size=n) * np.exp(-t / 0.05), 3000) * 0.8
    y /= np.max(np.abs(y))
    return y * vel


def pingpong(x, d_sec, fb=0.35, taps=6, lpf=3500):
    d = int(d_sec * SR)
    mono = lp((x[0] + x[1]) * 0.5, lpf)
    y = x.copy()
    for k in range(1, taps + 1):
        if k * d >= x.shape[1]:
            break
        y[k % 2, k * d:] += mono[:-k * d] * fb ** k
    return y


# ----------------------------------------------------------------------------------------------
# THE SCORE
# ----------------------------------------------------------------------------------------------
CHORDS = {}
for i, c in enumerate(['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'F', 'C']):             # I   Home
    CHORDS[1 + i] = c
for i, c in enumerate(['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'Gm', 'A']):            # II  Departure
    CHORDS[9 + i] = c
    CHORDS[17 + i] = c                                                            # III Ascent
for i, c in enumerate(['Dm', 'Dm', 'Bb', 'Bb', 'Gm', 'Gm', 'Eb', 'A',
                       'Dm', 'Bb', 'Eb', 'A']):                                   # IV  Storm
    CHORDS[25 + i] = c
for i, c in enumerate(['D', 'Bb', 'G', 'A', 'D', 'Bb', 'C', 'D']):               # VI  Lodestar
    CHORDS[41 + i] = c
    CHORDS[49 + i] = c
for i, c in enumerate(['D', 'Bb', 'G', 'A', 'D', 'Gm', 'D', 'D']):               # VII Homecoming
    CHORDS[57 + i] = c

# The theme. Its hook is the rising call — long, short, LONG: D ... A — D' — and its signature
# colour is the raised fourth (E over Bb) in bar 6. The minor version asks a question
# (it ends unresolved on C#); the major version answers it (bVI - bVII - I, melody C -> D).
THEME_MINOR = [
    (0, 0, 1.5, 'D4'), (0, 1.5, 0.5, 'A4'), (0, 2, 2, 'D5'),
    (1, 0, 1.5, 'C5'), (1, 1.5, 0.5, 'Bb4'), (1, 2, 2, 'A4'),
    (2, 0, 1.5, 'F4'), (2, 1.5, 0.5, 'A4'), (2, 2, 2, 'C5'),
    (3, 0, 2, 'E5'), (3, 2, 1, 'D5'), (3, 3, 1, 'C5'),
    (4, 0, 1.5, 'D4'), (4, 1.5, 0.5, 'A4'), (4, 2, 2, 'D5'),
    (5, 0, 1.5, 'F5'), (5, 1.5, 0.5, 'E5'), (5, 2, 2, 'D5'),
    (6, 0, 1.5, 'G4'), (6, 1.5, 0.5, 'Bb4'), (6, 2, 2, 'D5'),
    (7, 0, 3, 'C#5'), (7, 3, 1, 'A4'),
]
THEME_MAJOR = [
    (0, 0, 1.5, 'D4'), (0, 1.5, 0.5, 'A4'), (0, 2, 2, 'D5'),
    (1, 0, 1.5, 'C5'), (1, 1.5, 0.5, 'Bb4'), (1, 2, 2, 'A4'),
    (2, 0, 1.5, 'G4'), (2, 1.5, 0.5, 'B4'), (2, 2, 2, 'D5'),
    (3, 0, 2, 'E5'), (3, 2, 1, 'D5'), (3, 3, 1, 'C#5'),
    (4, 0, 1.5, 'D4'), (4, 1.5, 0.5, 'A4'), (4, 2, 2, 'D5'),
    (5, 0, 1.5, 'F5'), (5, 1.5, 0.5, 'E5'), (5, 2, 2, 'D5'),
    (6, 0, 1.5, 'E5'), (6, 1.5, 0.5, 'G5'), (6, 2, 2, 'C6'),
    (7, 0, 4, 'A5'),
]
THEME_MAJOR_FINAL = THEME_MAJOR[:-1] + [(7, 0, 4, 'D6')]
STORM_LINE = [
    (0, 0, 1.5, 'D3'), (0, 1.5, 0.5, 'A3'), (0, 2, 2, 'D4'),
    (1, 0, 1.5, 'C4'), (1, 1.5, 0.5, 'Bb3'), (1, 2, 2, 'A3'),
    (2, 0, 1.5, 'Bb2'), (2, 1.5, 0.5, 'F3'), (2, 2, 2, 'Bb3'),
    (3, 0, 1.5, 'A3'), (3, 1.5, 0.5, 'G3'), (3, 2, 2, 'F3'),
    (4, 0, 1.5, 'G3'), (4, 1.5, 0.5, 'D4'), (4, 2, 2, 'G4'),
    (5, 0, 1.5, 'F4'), (5, 1.5, 0.5, 'Eb4'), (5, 2, 2, 'D4'),
    (6, 0, 1.5, 'Eb4'), (6, 1.5, 0.5, 'G4'), (6, 2, 2, 'Bb4'),
    (7, 0, 2, 'C#4'), (7, 2, 2, 'E4'),
    (8, 0, 1.5, 'D4'), (8, 1.5, 0.5, 'A4'), (8, 2, 2, 'D5'),
    (9, 0, 1.5, 'F5'), (9, 1.5, 0.5, 'E5'), (9, 2, 2, 'D5'),
    (10, 0, 1.5, 'Eb5'), (10, 1.5, 0.5, 'D5'), (10, 2, 2, 'Bb4'),
    (11, 0, 4, 'C#5'),
]


def clock_notes(ch):
    """tick = the chord's fifth, tock = its root: the clock turned into a melody"""
    pc, _ = chord_info(ch)
    return above((pc + 7) % 12, 76), above(pc, 69)


def phrase(notes_list, start_bar, shift=0, bars=None, legato=1.0):
    return [(sec(b), sec(d) * legato, m) for b, d, m in line(notes_list, start_bar, shift, bars)]


def line(notes, start_bar, shift=0, bars=None):
    """Yield (abs_beat, dur_beats, midi) for a melody list starting at start_bar."""
    for b, beat, d, nm in notes:
        if bars is not None and not (bars[0] <= b < bars[1]):
            continue
        yield bt(start_bar + b, beat), d, midi(nm) + shift


def lerp(a, b, x):
    return a + (b - a) * x


def progress(bar, b0, b1):
    return np.clip((bar - b0) / max(1, (b1 - b0)), 0, 1)


def render_score():
    S = bus('strings', 0.9, 0.45)
    V = bus('violins', 0.9, 0.45)
    SP = bus('spic', 0.75, 0.3)
    BR = bus('brass', 1.0, 0.4)
    OR = bus('organ', 0.8, 0.6)
    PN = bus('piano', 1.0, 0.45)
    CH = bus('choir', 0.7, 0.6)
    DR = bus('drums', 1.0, 0.22)
    IM = bus('impacts', 1.0, 0.3)
    TK = bus('ticks', 1.8, 0.25)
    SB = bus('sub', 0.4, 0.0)
    HB = bus('heart', 0.9, 0.1)
    SH = bus('shepard', 0.5, 0.5)
    AM = bus('ambience', 0.8, 0.6)
    LD = bus('lead', 1.35, 0.45)
    PL = bus('pluck', 0.55, 0.4)
    GG = bus('gong', 0.7, 0.6)

    def add_lead(notes, pan=0.0, **kw):
        sig, t0 = lead_phrase(notes, **kw)
        LD.add(t0, sig, pan=pan, jitter=0)

    # the clock's twin: a pluck on every tick (tick = fifth, tock = root of the current chord)
    def clock_pluck(t_beat, bar, k, vel):
        tk, tc = clock_notes(CHORDS.get(bar, 'Dm'))
        m = tk if k % 2 == 0 else tc
        PL.add(sec(t_beat), pluck(hz(m), vel, decay=0.55 if vel < 0.3 else 0.8), pan=0.35 if k % 2 == 0 else -0.35, jitter=0)

    # gongs: the opening, the storm, the climax, the landing
    # bar 1: someone winds the watch — three turns of the crown — then time begins at bar 2
    tw = 0.35
    for turn in range(3):
        for c in range(6):
            TK.add(tw + c * 0.048 + rng.normal(0, 0.003), ratchet_click(0.5 * rng.uniform(0.8, 1.0)), pan=0.15, jitter=0)
        n_fr = int(0.3 * SR)
        fr = bp(rng.normal(size=n_fr), 1500, 6000) * np.sin(np.linspace(0, np.pi, n_fr)) * 0.012
        TK.add(tw, fr, pan=0.15, jitter=0)
        tw += 0.6
    GG.add(sec(bt(25)), tamtam(0.55), jitter=0)
    GG.add(sec(bt(41)), tamtam(1.0), jitter=0)
    GG.add(sec(bt(57)), tamtam(0.5), jitter=0)

    def pad(bar, ch, vel, beats=4, a=0.5, r=1.0, **kw):
        for m in pad_voicing(ch):
            S.add(sec(bt(bar)), strings(hz(m), sec(beats), vel * 0.55, a=a, r=r, **kw),
                  pan=np.clip((m - 60) / 30, -0.6, 0.6))

    def organ_chord(bar, ch, vel, beats=4, ranks=ORGAN_FULL, a=0.25, r=0.9, beat=0.0, cresc=None, bass=True):
        notes = pad_voicing(ch)
        if bass:
            notes = [bass_root(ch) + 12] + notes
        for m in notes:
            OR.add(sec(bt(bar, beat)), organ(hz(m), sec(beats), vel, a=a, r=r, ranks=ranks, cresc=cresc),
                   pan=np.clip((m - 57) / 36, -0.5, 0.5))

    # ---------------------------------------------------------------- the clock (time itself)
    for bar in range(2, 37):
        if bar <= 28:
            step = 1.0
        elif bar <= 32:
            step = 0.5
        else:
            step = 0.25
        if bar <= 8:
            v = 0.55
        elif bar <= 16:
            v = lerp(0.5, 0.35, progress(bar, 9, 16))
        elif bar <= 24:
            v = 0.25
        else:
            v = lerp(0.25, 0.5, progress(bar, 25, 36))
        b = 0.0
        k = 0
        pv = 0.5 if bar <= 8 else (0.3 if bar <= 16 else (0.18 if bar <= 24 else lerp(0.2, 0.32, progress(bar, 25, 36))))
        while b < 4:
            TK.add(sec(bt(bar, b)), tick(v * rng.uniform(0.95, 1.02), tock=(k % 2 == 1)), pan=0.2, jitter=0)
            clock_pluck(bt(bar, b), bar, k, pv * (1.0 if step == 1.0 else 0.75))
            b += step
            k += 1
    # slowed-down time while lost
    for bar in (38, 39, 40):
        for b in (0, 2):
            if bar == 40 and b == 2:
                continue
            TK.add(sec(bt(bar, b)), tick(0.4, tock=(b == 2)), pan=0.35, jitter=0)
            clock_pluck(bt(bar, b), 37, b // 2, 0.22)
    # homecoming: the clock ticks gently... then stops
    for bar in range(57, 63):
        for b in range(4):
            v = lerp(0.4, 0.15, progress(bar, 57, 62))
            TK.add(sec(bt(bar, b)), tick(v, tock=(b % 2 == 1)), pan=0.35, jitter=0)
            clock_pluck(bt(bar, b), bar, b, lerp(0.4, 0.15, progress(bar, 57, 62)))

    # ---------------------------------------------------------------- I. HOME (1-8)
    for bar in range(1, 17):
        ch = CHORDS[bar]
        v = lerp(0.2, 0.36, progress(bar, 1, 8)) if bar <= 8 else lerp(0.3, 0.2, progress(bar, 9, 16))
        organ_chord(bar, ch, v, ranks=ORGAN_SOFT if bar < 5 else ORGAN_FULL, a=0.6 if bar == 1 else 0.25)
    # piano 8th-note arpeggio: the "memory" figure
    for bar in list(range(3, 17)):
        ch = CHORDS[bar]
        arp = arp_notes(ch)
        for i in range(8):
            v = (0.32 if bar < 9 else 0.28) * (1.15 if i % 4 == 0 else 1.0) * rng.uniform(0.9, 1.08)
            m = arp[i % 4]
            PN.add(sec(bt(bar, i * 0.5)), piano(hz(m), sec(0.5), v, ring=sec(1.5)), pan=(m - 62) / 40)
    # piano plays the first half of the theme, high and fragile
    for b, d, m in line(THEME_MINOR, 5, shift=12, bars=(0, 4)):
        PN.add(sec(b), piano(hz(m), sec(d), 0.62, bright=0.9), pan=0.1)

    # ---------------------------------------------------------------- II. DEPARTURE (9-16)
    for bar in range(9, 17):
        pad(bar, CHORDS[bar], lerp(0.22, 0.4, progress(bar, 9, 16)), a=0.7)
        root = bass_root(CHORDS[bar]) + 12
        for i in range(8):
            v = 0.32 * (1.2 if i % 2 == 0 else 0.85)
            SP.add(sec(bt(bar, i * 0.5)), spic(hz(root), v), pan=-0.3)
    add_lead(phrase(THEME_MINOR, 9, shift=12), vel=0.62, bright=0.7, glide=0.09, pan=0.1)
    for b, d, m in line(THEME_MINOR, 9):
        BR.add(sec(b), brass(hz(m), sec(d) * 0.97, 0.42, bright=0.6, a=0.12), pan=-0.25)
    for bar in range(13, 17):
        DR.add(sec(bt(bar)), timpani(hz(bass_root(CHORDS[bar]) + 12), 0.35), pan=0)

    # ---------------------------------------------------------------- III. ASCENT (17-24)
    for bar in range(17, 25):
        ch = CHORDS[bar]
        x = progress(bar, 17, 24)
        pad(bar, ch, lerp(0.42, 0.62, x), a=0.3)
        # running 16th ostinato
        ost = ost_notes(ch)
        for i in range(16):
            m = ost[[0, 1, 2, 1, 3, 1, 2, 1][i % 8]]
            v = lerp(0.3, 0.6, x) * (1.3 if i % 4 == 0 else 1.0) * rng.uniform(0.92, 1.05)
            SP.add(sec(bt(bar, i * 0.25)), spic(hz(m), v, dur=0.09), pan=0.35)
        # low pulse
        root = bass_root(ch) + 12
        for i in range(8):
            SP.add(sec(bt(bar, i * 0.5)), spic(hz(root), 0.45 * (1.2 if i % 2 == 0 else 0.8)), pan=-0.35)
            SP.add(sec(bt(bar, i * 0.5)), spic(hz(root - 12), 0.35 * (1.2 if i % 2 == 0 else 0.8)), pan=-0.2)
        SB.add(sec(bt(bar)), sub(hz(bass_root(ch)), sec(4) * 0.95, lerp(0.35, 0.6, x)))
        # drums
        hits = [0, 2] if bar < 21 else [0, 1.5, 2, 3.5]
        for h in hits:
            v = lerp(0.45, 0.8, x) * (1.0 if h in (0, 2) else 0.6)
            DR.add(sec(bt(bar, h)), taiko(v), pan=rng.uniform(-0.2, 0.2))
        if bar >= 21:
            for b, d in [(0, 2), (2, 2)]:
                m = bass_root(ch) + 12
                BR.add(sec(bt(bar, b)), brass(hz(m), sec(d) * 0.9, lerp(0.5, 0.7, x), bright=0.9, voices=2),
                       pan=-0.3)
    for b, d, m in line(THEME_MINOR, 17):
        x = progress((b / 4) + 1, 17, 24)
        vel = lerp(0.6, 0.85, x)
        V.add(sec(b), strings(hz(m + 12), sec(d), vel * 0.7, a=0.08, r=0.6, voices=6, vib=0.005), pan=0.25)
        BR.add(sec(b), brass(hz(m), sec(d) * 0.97, vel, bright=1.0), pan=-0.1)
    # build into the storm
    for i in range(32):
        v = lerp(0.1, 0.8, i / 31)
        DR.add(sec(bt(24, i * 0.125)), snare(v), pan=0.1, jitter=0.002)
    for i in range(24):
        DR.add(sec(bt(24, i / 6)), timpani(hz(midi('A1')), lerp(0.2, 0.7, i / 23)), pan=0, jitter=0.002)
    r = riser(sec(4), 0.8)
    IM.add(sec(bt(25)) - r.shape[1] / SR, r, jitter=0)

    # ---------------------------------------------------------------- IV. THE STORM (25-36)
    for bar in (25, 29, 33):
        IM.add(sec(bt(bar)), impact(0.9), jitter=0)
        IM.add(sec(bt(bar)), crash(0.8), jitter=0)
        for m in ['D2', 'A2', 'D3', 'F3', 'A3']:   # the "braam"
            BR.add(sec(bt(bar)), brass(hz(midi(m)), sec(6), 0.8, a=0.05, r=1.2, bright=1.8,
                                       cresc=(0.9, 0.35)), pan=rng.uniform(-0.3, 0.3))
    for bar in range(25, 37):
        ch = CHORDS[bar]
        x = progress(bar, 25, 36)
        pc, th = chord_info(ch)
        r0 = above(pc, 62)
        pat = [12, 7, th, 12, 7, th, 12, 8]
        accents = [1.35, 0.9, 0.9, 1.25, 0.9, 0.9, 1.25, 1.0]
        for i in range(16):
            m = r0 + pat[i % 8]
            v = lerp(0.48, 0.78, x) * accents[i % 8] * rng.uniform(0.95, 1.03)
            SP.add(sec(bt(bar, i * 0.25)), spic(hz(m), v, dur=0.08), pan=0.4)
            if bar >= 29:
                SP.add(sec(bt(bar, i * 0.25)), spic(hz(m - 12), v * 0.8, dur=0.08), pan=-0.4)
            if bar >= 33:
                SP.add(sec(bt(bar, i * 0.25)), spic(hz(m + 12), v * 0.6, dur=0.07), pan=0.6)
        # low strings hammering 3+3+2
        root = bass_root(ch) + 12
        for i, pos in enumerate([0, 0.75, 1.5, 2, 2.75, 3.5]):
            v = 0.6 * (1.2 if i in (0, 3) else 0.9)
            SP.add(sec(bt(bar, pos)), spic(hz(root), v, dur=0.2), pan=-0.3)
            SP.add(sec(bt(bar, pos)), spic(hz(root - 12), v, dur=0.2), pan=-0.1)
        pad(bar, ch, lerp(0.4, 0.7, x), a=0.2, r=0.4)
        SB.add(sec(bt(bar)), sub(hz(bass_root(ch)), sec(4) * 0.96, 0.7, r=0.2))
        # percussion: taiko ensemble in 3+3+2
        for i, pos16 in enumerate([0, 3, 6, 8, 11, 14]):
            v = lerp(0.5, 0.82, x) * [1.0, 0.7, 0.8, 0.9, 0.7, 0.8][i]
            DR.add(sec(bt(bar, pos16 * 0.25)), taiko(v, size=1.0 if i in (0, 3) else 0.8), pan=rng.uniform(-0.3, 0.3))
        if bar >= 29:
            for i in range(16):
                acc = i in (0, 3, 6, 8, 11, 14)
                DR.add(sec(bt(bar, i * 0.25)), snare((0.55 if acc else 0.22) * lerp(0.8, 1.1, x)), pan=0.15)
        if bar in (32, 35):
            for i in range(8):
                DR.add(sec(bt(bar, 2 + i * 0.25)), tom([110, 110, 95, 95, 80, 80, 68, 68][i], 0.7), pan=lerp(0.4, -0.4, i / 7))
        if bar >= 33:
            for m in choir_voicing(ch, high=True):
                CH.add(sec(bt(bar)), choir(hz(m), sec(4) * 0.98, lerp(0.35, 0.6, progress(bar, 33, 36)), a=0.3, r=0.3),
                       pan=np.clip((m - 60) / 30, -0.5, 0.5))
    for b, d, m in line(STORM_LINE, 25):
        bar_no = int(b // 4) + 1
        x = progress(bar_no, 25, 36)
        if bar_no <= 32:
            BR.add(sec(b), brass(hz(m), sec(d) * 0.95, lerp(0.75, 0.9, x), bright=1.3), pan=-0.3)
            BR.add(sec(b), brass(hz(m + 12), sec(d) * 0.95, lerp(0.7, 0.85, x), bright=1.2), pan=0.2)
        else:
            BR.add(sec(b), brass(hz(m - 12), sec(d) * 0.95, 0.9, bright=1.4), pan=-0.3)
            BR.add(sec(b), brass(hz(m), sec(d) * 0.95, 0.9, bright=1.4), pan=0.2)
            V.add(sec(b), strings(hz(m + 12), sec(d), 0.55, a=0.05, r=0.3, voices=6), pan=0.3)
    add_lead(phrase(STORM_LINE, 25, shift=12, bars=(8, 12)), vel=0.55, bright=1.3, drive=0.55, glide=0.05, pan=-0.1)
    # the siren: violins glissando up, time running out
    S.add(sec(bt(35)), strings(hz(midi('A4')), sec(8), 0.4, a=1.0, r=0.2, glide=4.0, voices=7, cresc=(0.1, 0.6)), pan=0.5)
    S.add(sec(bt(35)), strings(hz(midi('E5')), sec(8), 0.4, a=1.0, r=0.2, glide=4.0, voices=7, cresc=(0.1, 0.5)), pan=-0.5)
    for i in range(32):
        DR.add(sec(bt(36, i * 0.125)), snare(lerp(0.3, 1.0, i / 31)), pan=0.1, jitter=0.002)
        DR.add(sec(bt(36, i * 0.125)), timpani(hz(midi('A1')), lerp(0.3, 0.9, i / 31)), jitter=0.002)
    r = riser(sec(4), 1.0)
    IM.add(sec(bt(37)) - r.shape[1] / SR, r, jitter=0)

    SH.add(sec(bt(29)), shepard(sec(bt(37) - bt(29)), 0.16, 10.0), jitter=0)

    # ---------------------------------------------------------------- V. LOST (37-40)
    AM.add(sec(bt(37, 1)), space_wind(sec(bt(40, 3.5) - bt(37, 1)) + 1.5, 0.35), jitter=0)
    IM.add(sec(bt(37)), impact(1.0) * 1.1, jitter=0)
    IM.add(sec(bt(37)), crash(1.0, length=5.0), jitter=0)
    # the void: faint high strings, like distant stars
    S.add(sec(bt(37, 2)), strings(hz(midi('A5')), sec(13.5), 0.07, a=3.0, r=0.3, voices=4, cutoff=3000), pan=0.4)
    S.add(sec(bt(37, 3)), strings(hz(midi('E6')), sec(12.5), 0.05, a=3.0, r=0.3, voices=4, cutoff=4000), pan=-0.4)
    for bar in (38, 39, 40):
        for b in (0, 2):
            if bar == 40 and b == 2:
                continue
            HB.add(sec(bt(bar, b)), heartbeat(0.8), jitter=0)
    # the broken memory of the theme
    for b, beat, d, nm, v in [(38, 0, 1.5, 'D4', 0.45), (38, 1.5, 0.5, 'A4', 0.4), (38, 2, 2, 'D5', 0.42),
                              (39, 0, 1.5, 'C5', 0.36), (39, 1.5, 2.5, 'Bb4', 0.32)]:
        PN.add(sec(bt(b, beat)), piano(hz(midi(nm)), sec(d), v, bright=0.7), pan=0.05)
    # the swell — something appears in the dark (Asus4 -> A), then a breath of silence
    cut = 3.5
    for m in ['A1', 'A2', 'E3', 'A3', 'D4', 'E4']:
        OR.add(sec(bt(40)), organ(hz(midi(m)), sec(2), 0.5, a=0.3, r=0.15, cresc=(0.05, 0.4)), pan=0)
    for m in ['A1', 'A2', 'E3', 'A3', 'C#4', 'E4', 'A4']:
        OR.add(sec(bt(40, 2)), organ(hz(midi(m)), sec(cut - 2), 0.8, a=0.02, r=0.08, cresc=(0.4, 0.8)), pan=0)
    for m in pad_voicing('A'):
        S.add(sec(bt(40)), strings(hz(m), sec(cut), 0.6, a=0.4, r=0.08, cresc=(0.05, 0.75)), pan=np.clip((m - 60) / 30, -0.6, 0.6))
    for m in choir_voicing('A', high=True):
        CH.add(sec(bt(40)), choir(hz(m), sec(cut), 0.6, a=0.5, r=0.08, cresc=(0.05, 0.7)), pan=np.clip((m - 60) / 30, -0.5, 0.5))
    for i in range(int(cut * 6)):
        DR.add(sec(bt(40, i / 6)), timpani(hz(midi('A1')), lerp(0.1, 0.9, i / (cut * 6))), jitter=0.002)
        DR.add(sec(bt(40, i / 6)), snare(lerp(0.05, 0.7, i / (cut * 6))), pan=0.1, jitter=0.002)
    r = riser(sec(cut + 4), 1.0)
    IM.add(sec(bt(40, cut)) - r.shape[1] / SR, r, jitter=0)

    # ---------------------------------------------------------------- VI. LODESTAR (41-56)
    for bar in (41, 49):
        IM.add(sec(bt(bar)), impact(1.0), jitter=0)
        IM.add(sec(bt(bar)), crash(1.0, length=5.0), jitter=0)
    for bar in (45, 53):
        IM.add(sec(bt(bar)), crash(0.6), jitter=0)
    for bar in range(41, 57):
        ch = CHORDS[bar]
        second = bar >= 49
        x = progress(bar, 41, 56)
        pad(bar, ch, lerp(0.6, 0.8, x), a=0.15, r=0.8)
        # choir
        for m in choir_voicing(ch, high=second):
            CH.add(sec(bt(bar)), choir(hz(m), sec(4) * 0.98, 0.45 if not second else 0.62, a=0.25, r=0.8),
                   pan=np.clip((m - 60) / 30, -0.5, 0.5))
        # organ joins for the second pass — the Interstellar-sized sound
        if second:
            organ_chord(bar, ch, 0.55, a=0.12, r=1.0)
        # rolling major ostinato
        ost = ost_notes(ch)
        for i in range(16):
            m = ost[[0, 1, 2, 1, 3, 1, 2, 1][i % 8]] + 12
            v = 0.62 * (1.3 if i % 4 == 0 else 1.0) * rng.uniform(0.93, 1.05)
            SP.add(sec(bt(bar, i * 0.25)), spic(hz(m), v, dur=0.08), pan=0.4)
            SP.add(sec(bt(bar, i * 0.25)), spic(hz(m - 12), v * 0.8, dur=0.08), pan=-0.4)
        root = bass_root(ch)
        for i in range(8):
            v = 0.62 * (1.2 if i % 2 == 0 else 0.85)
            SP.add(sec(bt(bar, i * 0.5)), spic(hz(root + 12), v, dur=0.2), pan=-0.3)
            SP.add(sec(bt(bar, i * 0.5)), spic(hz(root), v, dur=0.2), pan=-0.1)
        SB.add(sec(bt(bar)), sub(hz(root), sec(4) * 0.96, 0.75, r=0.2))
        for b in (0, 2):
            BR.add(sec(bt(bar, b)), brass(hz(root + 12), sec(2) * 0.92, 0.75, bright=1.2, voices=2), pan=-0.35)
        # drums: taiko heartbeat of a hero + marching field drums
        for h, v in [(0, 1.0), (1.75, 0.55), (2, 0.9), (3.5, 0.5), (3.75, 0.6)]:
            DR.add(sec(bt(bar, h)), taiko(0.85 * v), pan=rng.uniform(-0.2, 0.2))
        for beat in range(4):
            for off, v in [(0, 0.5), (0.5, 0.3), (0.75, 0.38)]:
                DR.add(sec(bt(bar, beat + off)), snare(v * (1.25 if (beat == 0 and off == 0) else 1.0)), pan=0.15)
        if bar == 48:
            for i in range(8):
                DR.add(sec(bt(bar, 2 + i * 0.25)), tom([120, 120, 105, 105, 90, 90, 75, 75][i], lerp(0.6, 1.0, i / 7)),
                       pan=lerp(0.4, -0.4, i / 7))
    # the theme, answered at last
    for b, d, m in line(THEME_MAJOR, 41):
        BR.add(sec(b), brass(hz(m), sec(d) * 0.97, 0.9, bright=1.25), pan=-0.15)     # horns
        BR.add(sec(b), brass(hz(m), sec(d) * 0.97, 0.75, bright=1.6, voices=2), pan=0.2)  # trumpets
        BR.add(sec(b), brass(hz(m - 12), sec(d) * 0.97, 0.7, bright=1.1), pan=-0.35)   # trombones
        V.add(sec(b), strings(hz(m + 12), sec(d), 0.45, a=0.06, r=0.5, voices=6, vib=0.005), pan=0.3)
    for b, d, m in line(THEME_MAJOR_FINAL, 49):
        BR.add(sec(b), brass(hz(m), sec(d) * 0.97, 1.0, bright=1.45), pan=-0.15)
        BR.add(sec(b), brass(hz(m), sec(d) * 0.97, 0.8, bright=1.7, voices=2), pan=0.2)
        BR.add(sec(b), brass(hz(m - 12), sec(d) * 0.97, 0.75, bright=1.2), pan=-0.35)
        V.add(sec(b), strings(hz(m + 12), sec(d), 0.7, a=0.05, r=0.6, voices=7, vib=0.006), pan=0.3)
        V.add(sec(b), strings(hz(m + 24), sec(d), 0.3, a=0.05, r=0.6, voices=5, vib=0.006), pan=0.45)
    add_lead(phrase(THEME_MAJOR_FINAL, 49), vel=0.6, bright=1.2, drive=0.6, glide=0.06, pan=0.05)
    counter = [(0, 0, 4, 'F#3'), (1, 0, 2, 'D4'), (1, 2, 2, 'F4'), (2, 0, 2, 'D4'), (2, 2, 2, 'B3'),
               (3, 0, 2, 'C#4'), (3, 2, 2, 'E4'), (4, 0, 2, 'F#4'), (4, 2, 2, 'A4'), (5, 0, 2, 'F4'),
               (5, 2, 2, 'D4'), (6, 0, 2, 'E4'), (6, 2, 2, 'G4'), (7, 0, 4, 'F#4')]
    for b, d, m in line(counter, 49):
        BR.add(sec(b), brass(hz(m), sec(d) * 0.96, 0.6, bright=1.0, a=0.12), pan=0.45)
    # final bar of the climax: timpani roll into the landing
    for i in range(24):
        DR.add(sec(bt(56, i / 6)), timpani(hz(midi('D2')), lerp(0.3, 0.85, i / 23)), jitter=0.002)
    IM.add(sec(bt(57)), impact(0.75), jitter=0)
    IM.add(sec(bt(57)), crash(0.7, length=5.0), jitter=0)

    # ---------------------------------------------------------------- VII. HOMECOMING (57-64)
    for bar in range(57, 65):
        ch = CHORDS[bar]
        x = progress(bar, 57, 64)
        beats = 4 if bar < 64 else 8
        organ_chord(bar, ch, lerp(0.5, 0.3, x), beats=beats, ranks=ORGAN_SOFT if bar > 57 else ORGAN_FULL,
                    a=0.3, r=3.0 if bar == 64 else 0.8)
        for m in pad_voicing(ch):
            S.add(sec(bt(bar)), strings(hz(m), sec(beats), lerp(0.26, 0.13, x), a=0.4,
                                        r=4.0 if bar == 64 else 1.0), pan=np.clip((m - 60) / 30, -0.6, 0.6))
        if bar <= 62:
            arp = arp_notes(ch)
            for i in range(8):
                v = lerp(0.3, 0.18, x) * (1.15 if i % 4 == 0 else 1.0) * rng.uniform(0.9, 1.08)
                m = arp[i % 4]
                PN.add(sec(bt(bar, i * 0.5)), piano(hz(m), sec(0.5), v, ring=sec(1.5)), pan=(m - 62) / 40)
    SB.add(sec(bt(57)), sub(hz(midi('D1')), sec(4), 0.5, r=2.0))
    home = [(57, 0, 1.5, 'D5'), (57, 1.5, 0.5, 'A5'), (57, 2, 2, 'D6'),
            (58, 0, 1.5, 'C6'), (58, 1.5, 0.5, 'Bb5'), (58, 2, 2, 'A5'),
            (59, 0, 1.5, 'G5'), (59, 1.5, 0.5, 'B5'), (59, 2, 2, 'D6'),
            (60, 0, 2, 'E6'), (60, 2, 1, 'D6'), (60, 3, 1, 'C#6'),
            (61, 0, 1.5, 'D5'), (61, 1.5, 0.5, 'A5'), (61, 2, 2, 'D6'),
            (62, 0, 1.5, 'Bb5'), (62, 1.5, 0.5, 'A5'), (62, 2, 2, 'G5'),
            (63, 0, 4, 'F#5')]
    PL.add(sec(bt(64, 2)), pluck(hz(midi('A6')), 0.3, decay=2.2), pan=0.2, jitter=0)
    for b, beat, d, nm in home:
        PN.add(sec(bt(b, beat)), piano(hz(midi(nm)), sec(d), 0.68, bright=0.85), pan=0.1)
    for m in ['D3', 'A3', 'D4', 'F#4', 'A4']:
        CH.add(sec(bt(63)), choir(hz(midi(m)), sec(8), 0.3, a=2.5, r=4.5, formants=FORMANTS_OO),
               pan=np.clip((midi(m) - 60) / 30, -0.5, 0.5))
    # last chord: a rolled D(add9), left to ring into the silence
    for i, nm in enumerate(['D2', 'D3', 'A3', 'E4', 'F#4', 'A4', 'D5']):
        PN.add(sec(bt(64)) + i * 0.09, piano(hz(midi(nm)), 7.5, 0.42 if i else 0.5, bright=0.8), pan=(midi(nm) - 60) / 40,
               jitter=0)


# ----------------------------------------------------------------------------------------------
# mix & master
# ----------------------------------------------------------------------------------------------
def make_ir(rt60=3.6, length=5.5, predelay=0.028):
    n = int(length * SR)
    t = tarr(n)
    ir = np.zeros((2, n))
    for ch in range(2):
        noise = rng.normal(size=n)
        low = lp(noise, 400)
        high = hp(noise, 4000)
        mid = noise - low - high
        ir[ch] = (low * np.exp(-6.9 * t / (rt60 * 1.15)) + mid * np.exp(-6.9 * t / rt60)
                  + high * np.exp(-6.9 * t / (rt60 * 0.4)))
        # early reflections
        for _ in range(18):
            d = int(rng.uniform(0.005, 0.09) * SR)
            ir[ch, d] += rng.uniform(-1, 1) * 3.0
    ir *= np.clip(t / 0.012, 0, 1)
    pd = int(predelay * SR)
    ir = np.pad(ir, ((0, 0), (pd, 0)))[:, :n]
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir


def master():
    dry = np.zeros((2, N))
    send = np.zeros((2, N))
    stats = {}
    for b in BUSES.values():
        x = b.buf.astype(np.float64) * b.gain
        if b.name in ('strings', 'violins', 'spic'):
            x = x + 0.3 * bp(x, 220, 380) + 0.2 * bp(x, 2600, 3800)
        elif b.name == 'brass':
            x = x + 0.35 * bp(x, 900, 1500)
        elif b.name == 'lead':
            x = pingpong(x, 0.75 * SPB, fb=0.33)
        elif b.name == 'pluck':
            x = pingpong(x, 0.75 * SPB, fb=0.3, lpf=5000)
        stats[b.name] = x
        dry += x
        send += x * b.send
    print('  convolving reverb...')
    ir = make_ir()
    wet = np.zeros((2, N))
    for ch in range(2):
        wet[ch] = fftconvolve(send[ch], ir[ch])[:N]
    wet = hp(wet, 120)
    mix = dry + wet * 0.42
    mix = hp(mix, 28)
    # gentle "air" lift and low-mid cleanup
    mix = mix + 0.12 * hp(mix, 8000) - 0.08 * bp(mix, 250, 450)

    # glue compressor (slow, 2:1 above -18 dB relative to peak)
    peak = np.max(np.abs(mix))
    mix /= peak
    lvl = np.max(np.abs(mix), axis=0)
    lvl = maximum_filter1d(lvl, int(0.01 * SR))
    lvl = sosfilt(butter(1, 3.0, 'low', fs=SR, output='sos'), lvl)
    thr = 10 ** (-10 / 20)
    gain = np.where(lvl > thr, (lvl / thr) ** (1 / 1.25 - 1), 1.0)
    mix *= gain
    # lookahead peak limiter
    look = int(0.005 * SR)
    pk = maximum_filter1d(np.max(np.abs(mix), axis=0), 2 * look + 1)
    pk = sosfilt(butter(1, 40.0, 'low', fs=SR, output='sos'), pk)
    pk = maximum_filter1d(pk, 2 * look + 1)
    ceiling = 0.5
    g = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    mix *= g
    mix = np.tanh(mix / ceiling * 0.95) * ceiling / np.tanh(0.95)  # soft safety clip
    mix *= 10 ** (-0.8 / 20) / np.max(np.abs(mix))
    # fades
    fade = int(0.02 * SR)
    mix[:, :fade] *= np.linspace(0, 1, fade)
    end = int((sec(bt(TOTAL_BARS + 1)) + TAIL) * SR)
    mix = mix[:, :end]
    tail = int(3.0 * SR)
    mix[:, -tail:] *= np.linspace(1, 0, tail) ** 2
    return mix, stats


def write_outputs(mix, base):
    pcm = np.clip(mix.T * 32767, -32768, 32767).astype(np.int16)
    with wave.open(base + '.wav', 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    import lameenc
    enc = lameenc.Encoder()
    enc.set_bit_rate(320)
    enc.set_in_sample_rate(SR)
    enc.set_channels(2)
    enc.set_quality(2)
    mp3 = enc.encode(pcm.tobytes()) + enc.flush()
    with open(base + '.mp3', 'wb') as f:
        f.write(mp3)


def report(mix):
    print('\n  section loudness (RMS dBFS):')
    sections = [('I   Home', 1, 9), ('II  Departure', 9, 17), ('III Ascent', 17, 25), ('IV  Storm', 25, 37),
                ('V   Lost', 37, 41), ('VI  Lodestar', 41, 57), ('VII Homecoming', 57, 65)]
    for name, b0, b1 in sections:
        a, b = int(sec(bt(b0)) * SR), int(sec(bt(b1)) * SR)
        seg = mix[:, a:b]
        rms = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        print(f'    {name:<16} {rms:6.1f} dB')


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    cache = os.path.join(here, '.stems.npz')
    if os.environ.get('LODESTAR_CACHE') and os.path.exists(cache):
        d = np.load(cache)
        for k in d.files:
            name, gain, send = k.split('|')
            bus(name, float(gain), float(send)).buf = d[k]
    else:
        print('Rendering score...')
        render_score()
        print(f'  done in {time.time() - t0:.0f}s')
        if os.environ.get('LODESTAR_CACHE'):
            np.savez(cache, **{f'{b.name}|{b.gain}|{b.send}': b.buf for b in BUSES.values()})
    mix, stats = master()
    report(mix)
    write_outputs(mix, os.path.join(here, 'lodestar'))
    print(f'Wrote lodestar.mp3 ({mix.shape[1] / SR:.1f}s) in {time.time() - t0:.0f}s')
