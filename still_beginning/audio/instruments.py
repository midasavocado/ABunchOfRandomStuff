"""STILL BEGINNING score - synthesized instruments. All functions return float32 arrays at 48 kHz
(mono (n,) or stereo (n,2)). Every note is rendered with short fades so nothing clicks."""
import numpy as np
from scipy import signal
import sbdsp as D
from sbdsp import SR, osc, sine, svf, ladder, mtof

JP_OFFS = np.array([-0.11002313, -0.06288439, -0.01952356, 0.0, 0.01991221, 0.06216538, 0.10745242])


def _t(n):
    return np.arange(n, dtype=np.float64) / SR


def _rise(n, ms):
    k = max(1, min(n, int(ms * 0.001 * SR)))
    w = np.ones(n, np.float32)
    w[:k] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    return w


# =============================================================================== FELT PIANO
def piano(midi, vel, hold_s, rng, bright=1.0, tail_s=None, width=0.35):
    """Additive felt/upright piano. Inharmonic stretched partials, 1-3 detuned strings per note (beating),
    two-stage decay, frequency-dependent partial damping, strike-position comb, felt hammer thump and
    damper. hold_s = time until damper (pedal up). Returns stereo."""
    f0 = float(mtof(midi))
    T1 = float(np.clip(13.0 * 2 ** (-(midi - 36) / 17.0), 1.6, 16.0))   # T60 of fundamental
    B = 2.6e-4 * 2 ** ((midi - 60) / 16.0)                               # inharmonicity
    damp_t = 0.38
    ring = min(hold_s, T1 * 1.1) if tail_s is None else tail_s
    n = int((ring + damp_t + 0.05) * SR)
    t = _t(n)
    fc_felt = (420 + 2300 * vel ** 1.6) * bright + 1.3 * f0
    xh = 1 / 7.7
    K = int(min(36, 15000 / f0))
    ns = 3 if midi >= 50 else (2 if midi >= 34 else 1)
    cents = np.array([-0.95, 0.1, 1.05])[:ns] * (0.7 + 0.3 * rng.random(ns))
    spans = np.array([-width, 0.0, width])[:ns] if ns == 3 else (np.array([-width, width]) if ns == 2 else np.zeros(1))
    L = np.zeros(n); R = np.zeros(n)
    for k in range(1, K + 1):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 16000:
            break
        a = k ** -0.85 * np.exp(-fk / fc_felt) * (0.18 + abs(np.sin(np.pi * k * xh)))
        if a < 1e-4:
            continue
        T = T1 / (1 + (fk / 1300.0) ** 1.25)
        env = 0.62 * np.exp(-6.91 * t / (0.28 * T)) + 0.38 * np.exp(-6.91 * t / T)
        # felt: upper partials bloom slightly later
        ta = 0.0012 + 0.00035 * k
        env *= 1 - np.exp(-t / ta)
        nstr = ns if k <= 9 else 1
        ph0 = rng.random() * 2 * np.pi          # strings share the hammer strike -> start in phase
        for s in range(nstr):
            c = cents[s] if nstr > 1 else 0.0
            ph = ph0 + (rng.random() - 0.5) * 0.4
            y = a * env * np.sin(2 * np.pi * fk * (1 + c / 1731.0) * t + ph) / nstr
            p = spans[s] if nstr > 1 else 0.0
            L += y * (0.5 - 0.5 * p)
            R += y * (0.5 + 0.5 * p)
    # hammer / felt thump + faint key knock
    nt = min(n, int(0.08 * SR))
    th = rng.standard_normal(nt)
    th = D.lp(th, 260 + 300 * vel) * np.exp(-_t(nt) / 0.011) * 0.9
    kn = D.bp(rng.standard_normal(nt), 700, 2400) * np.exp(-_t(nt) / 0.004) * 0.05
    thump = (th + kn) * vel * 0.9 * float(np.clip((86 - midi) / 30.0, 0.12, 1.0))
    L[:nt] += thump; R[:nt] += thump
    # damper at pedal up
    dstart = int(ring * SR)
    if dstart < n:
        d = np.ones(n)
        d[dstart:] = np.exp(-6.91 * (t[dstart:] - ring) / damp_t)
        L *= d; R *= d
    reg = 10 ** (0.26 * np.clip(midi - 62, -30, 30) / 20)
    out = np.stack([L, R], 1).astype(np.float32) * (0.25 + 0.75 * vel) * reg
    # key-position panning (player's perspective, subtle)
    pp = np.clip((midi - 62) / 36.0, -0.4, 0.4)
    gl, gr = D.pan_gains(pp)
    out[:, 0] *= gl * 1.414; out[:, 1] *= gr * 1.414
    return D.fade_edges(out, 0.0005, 0.02)


def soundboard(x, rng, mix=0.3):
    """Synthetic soundboard/body IR convolution (adds woody resonance to piano bus)."""
    n = int(0.09 * SR)
    t = _t(n)
    ir = rng.standard_normal((n, 2))
    lo = D.lp(ir, 900) * np.exp(-t / 0.018)[:, None]
    hi = D.hp(ir, 900) * np.exp(-t / 0.006)[:, None] * 0.4
    ir = (lo + hi)
    ir = D.eq(ir, ("peak", 220, 1.2, 4), ("peak", 520, 1.5, 3), ("peak", 1800, 2, -2))
    ir /= np.sqrt(np.sum(ir ** 2, 0, keepdims=True))
    wet = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    return (x * (1 - mix) + wet * mix * 0.9).astype(np.float32)


# =============================================================================== PLUCKS
def pluck(midi, vel, gate_s, rng, bright=1.0, decay=0.22, detune=7.0, sub=0.0):
    """Synth pluck: detuned band-limited saws + pulse through an enveloped SVF lowpass."""
    f0 = float(mtof(midi))
    n = int((gate_s + 0.5) * SR)
    t = _t(n)
    y = np.zeros(n, np.float32)
    for c in (-detune, detune):
        y += osc(np.full(n, f0 * 2 ** (c / 1200)), n, "saw", 0.4 + 0.2 * rng.random()) * 0.5
    y += osc(np.full(n, f0), n, "pulse", 0.1 * rng.random(), 0.28) * 0.35
    if sub:
        y += sine(np.full(n, f0 / 2), n) * sub
    fc = (180 + f0 * 1.2 + (2500 + 5500 * vel) * bright * np.exp(-t / 0.075)).astype(np.float32)
    y = svf(y, fc, 0.9, "lp")
    env = np.exp(-t / decay) * _rise(n, 2.5)
    g = int(gate_s * SR)
    if g < n:
        env[g:] *= np.exp(-(t[g:] - gate_s) / 0.045)
    return (y * env * vel).astype(np.float32)


def kpluck(midi, vel, dur_s, rng, bright=0.45, t60=2.2):
    """Karplus-Strong 'harp / plucked' texture (warm acoustic-ish)."""
    f0 = float(mtof(midi))
    n = int((dur_s + 0.3) * SR)
    exc = np.zeros(n, np.float32)
    ne = int(SR / f0 * 1.0)
    e = rng.standard_normal(ne).astype(np.float32)
    e = D.lp(e, 1500 + 4000 * vel)
    exc[:ne] = e * np.hanning(ne)
    y = D.karplus(exc, f0, t60, bright)
    y = D.hp(y, 60)
    env = np.ones(n, np.float32)
    g = int(dur_s * SR)
    env[g:] = np.exp(-_t(n - g) / 0.08)
    return (y * env * vel * 0.8).astype(np.float32)


# =============================================================================== SUPERSAW
def supersaw_freq(freq, rng, detune=0.2, width=0.9, voices=7, side=0.75, center=1.0):
    """Supersaw on a per-sample frequency curve (Hz). Returns stereo (n,2). Voices get random phases
    and a slow independent drift so the stack 'breathes'."""
    n = len(freq)
    out = np.zeros((n, 2), np.float32)
    offs = JP_OFFS if voices == 7 else np.linspace(-0.11, 0.11, voices)
    pans = np.linspace(-1, 1, voices) * width
    order = np.argsort(np.abs(offs))
    for j, o in enumerate(offs):
        drift = 1 + 0.0009 * np.sin(2 * np.pi * (0.13 + 0.07 * j) * _t(n) + rng.random() * 6.28)
        f = (freq * (1 + o * detune) * drift).astype(np.float32)
        y = osc(f, n, "saw", rng.random())
        g = center if o == 0 else side
        # alternate pans by detune rank so L/R get equal spread
        p = pans[j]
        gl, gr = D.pan_gains(p)
        out[:, 0] += y * g * gl
        out[:, 1] += y * g * gr
    return out / np.sqrt(voices)


def line_curves(notes, n, glide_ms=14, rel_ms=70, att_ms=8, gap_ms=22, vib=(5.3, 14, 0.28), sr=SR,
                accent_tau=0.16):
    """Monophonic line: notes = [(start_samp, len_samp, midi, vel)], relative to block.
    Returns freq (Hz), amp envelope, accent envelope (for filter)."""
    tgt = np.full(n, np.nan)
    gate = np.zeros(n, np.float32)
    acc = np.zeros(n, np.float32)
    vibe = np.zeros(n, np.float32)
    notes = sorted(notes)
    gapn = int(gap_ms * 0.001 * sr)
    for i, (s, l, m, v) in enumerate(notes):
        e = min(n, s + l)
        tgt[s:e] = m
        nxt = notes[i + 1][0] if i + 1 < len(notes) else None
        ge = e - gapn if (nxt is not None and nxt <= e + 2) else e
        gate[s:max(s + 1, ge)] = v
        if ge < e:
            gate[ge:e] = v * 0.55          # legato re-articulation: shallow dip, never a hard retrigger
        k = np.arange(n - s)
        acc[s:] = np.maximum(acc[s:], np.exp(-k / (accent_tau * sr)).astype(np.float32) * v)
        # delayed vibrato on long notes
        if l > 0.35 * sr:
            kk = np.arange(e - s) / sr
            ramp = np.clip((kk - vib[2]) / 0.35, 0, 1)
            vibe[s:e] = np.maximum(vibe[s:e], ramp)
    # hold pitch through rests (fill forward, then back)
    idx = np.where(~np.isnan(tgt), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    tgt = tgt[idx]
    first = np.argmax(~np.isnan(tgt)) if np.any(~np.isnan(tgt)) else 0
    tgt[:first] = tgt[first]
    tgt = np.nan_to_num(tgt, nan=60.0)
    # exponential glide in pitch domain
    a = np.exp(-1.0 / (glide_ms * 0.001 * sr))
    mp = signal.lfilter([1 - a], [1, -a], tgt, zi=[tgt[0] * a])[0]
    t = np.arange(n) / sr
    mp = mp + vibe * (vib[1] / 100.0) * np.sin(2 * np.pi * vib[0] * t)
    freq = mtof(mp).astype(np.float32)
    amp = D.gainsmooth(gate, rel_ms, att_ms)   # (gainsmooth: falling uses 1st arg)
    return freq, amp, acc


# =============================================================================== LEAD
def lead_line(notes, n, rng, bright=1.0, detune=0.19, octave_layer=0.0, focus=0.55, cut=5200.0, width=0.85):
    """Anthemic supersaw lead: 7-voice supersaw + focused pulse (pitch definition) + optional octave layer,
    accent-driven lowpass, delayed vibrato."""
    freq, amp, acc = line_curves(notes, n)
    ss = supersaw_freq(freq, rng, detune=detune, width=width)
    pl = osc(freq * 1.0015, n, "pulse", rng.random(), 0.42) * 0.45 + osc(freq * 0.9985, n, "saw", rng.random()) * 0.35
    body = ss + focus * np.stack([pl, pl], 1)
    if octave_layer:
        o = osc(freq * 2, n, "saw", rng.random()) * 0.5 + osc(freq * 2.003, n, "saw", rng.random()) * 0.5
        body[:, 0] += octave_layer * o * 0.8
        body[:, 1] += octave_layer * o * 0.8
    fc = (cut * bright * (0.55 + 0.75 * acc)).astype(np.float32)
    out = np.stack([svf(body[:, 0], fc, 0.8), svf(body[:, 1], fc, 0.8)], 1)
    out = D.hp(out, 220)
    return (out * amp[:, None]).astype(np.float32)


def soft_lead(notes, n, rng, cut=3200.0):
    """Warm, rounded lead (triangle + filtered saw) used for the tender/breakdown doubling."""
    freq, amp, acc = line_curves(notes, n, glide_ms=22, rel_ms=160, att_ms=25)
    y = osc(freq, n, "tri", rng.random()) * 0.8 + osc(freq * 1.002, n, "saw", rng.random()) * 0.3
    y = svf(y, (cut * (0.7 + 0.5 * acc)).astype(np.float32), 0.7)
    return D.stereo(y * amp, 0.0)


def counter_line(notes, n, rng, cut=2400.0):
    """Tenor counter-melody voice ('synth horn'): 3 detuned saws + pulse, warm ladder, slow attack."""
    freq, amp, acc = line_curves(notes, n, glide_ms=25, rel_ms=140, att_ms=18, vib=(5.0, 10, 0.35))
    y = np.zeros((n, 2), np.float32)
    for j, (d, p) in enumerate([(-9, -0.6), (0, 0), (9, 0.6)]):
        s = osc(freq * 2 ** (d / 1200), n, "saw", rng.random())
        gl, gr = D.pan_gains(p)
        y[:, 0] += s * gl; y[:, 1] += s * gr
    fc = (cut * (0.6 + 0.6 * acc)).astype(np.float32)
    y = np.stack([ladder(y[:, 0], fc, 0.15, 1.3), ladder(y[:, 1], fc, 0.15, 1.3)], 1)
    y = D.hp(y, 160)
    return (y * amp[:, None]).astype(np.float32)


# =============================================================================== CHORDS / PADS
def saw_chord_note(midi, gate_s, rng, a=0.01, d=0.3, s=0.8, r=0.35, detune=0.16, cut=4000, cut_env=0.0,
                   width=0.9, voices=7, tail=None):
    """one supersaw chord voice with ADSR and lowpass"""
    rr = r if tail is None else tail
    n = int((gate_s + rr + 0.05) * SR)
    f0 = float(mtof(midi))
    y = supersaw_freq(np.full(n, f0, np.float32), rng, detune=detune, width=width, voices=voices)
    t = _t(n)
    fc = (cut * (1 + cut_env * np.exp(-t / 0.12))).astype(np.float32)
    y = np.stack([svf(y[:, 0], fc, 0.75), svf(y[:, 1], fc, 0.75)], 1)
    env = D.adsr(n, a, d, s, r, int(gate_s * SR))
    return D.fade_edges(y * env[:, None], 0.001, 0.01)


def warm_pad_note(midi, gate_s, rng, a=0.8, r=1.6, cut=1600, bright_lfo=0.25):
    """Analog-style warm pad: 2 detuned saws + triangle, gently moving lowpass."""
    n = int((gate_s + r + 0.1) * SR)
    f0 = float(mtof(midi))
    t = _t(n)
    y = np.zeros((n, 2), np.float32)
    for c, p in ((-6, -0.7), (6, 0.7), (0.0, 0.0)):
        drift = 1 + 0.0012 * np.sin(2 * np.pi * (0.2 + rng.random() * 0.2) * t + rng.random() * 6)
        s = osc((f0 * 2 ** (c / 1200) * drift).astype(np.float32), n, "saw" if c else "tri", rng.random())
        gl, gr = D.pan_gains(p)
        y[:, 0] += s * gl; y[:, 1] += s * gr
    fc = (cut * (1 + bright_lfo * np.sin(2 * np.pi * 0.11 * t + rng.random() * 6))).astype(np.float32)
    y = np.stack([svf(y[:, 0], fc, 0.6), svf(y[:, 1], fc, 0.6)], 1)
    env = D.adsr(n, a, 0.5, 0.9, r, int(gate_s * SR), curve=3)
    return (y * env[:, None]).astype(np.float32)


def string_pad_note(midi, gate_s, rng, a=0.45, r=1.4, cut=3800):
    """Soft string-ensemble pad: 5 saws with independent vibrato/drift, body resonances, bowed attack."""
    n = int((gate_s + r + 0.1) * SR)
    f0 = float(mtof(midi))
    t = _t(n)
    y = np.zeros((n, 2), np.float32)
    for j in range(5):
        vr = 4.6 + rng.random() * 1.2
        vd = 0.0022 * np.clip((t - 0.3) / 0.6, 0, 1)
        fm = 1 + vd * np.sin(2 * np.pi * vr * t + rng.random() * 6) + (rng.random() - 0.5) * 0.004
        s = osc((f0 * fm).astype(np.float32), n, "saw", rng.random())
        gl, gr = D.pan_gains(-0.8 + 0.4 * j)
        y[:, 0] += s * gl; y[:, 1] += s * gr
    y = D.eq(y, ("peak", 450, 1.4, 3), ("peak", 1150, 2.0, 2.5), ("peak", 2900, 1.8, 2.0), ("highshelf", 6000, 0.7, -8))
    fc = np.full(n, cut, np.float32) * (0.55 + 0.45 * np.clip(t / (a * 1.5 + 1e-3), 0, 1)).astype(np.float32)
    y = np.stack([svf(y[:, 0], fc, 0.7), svf(y[:, 1], fc, 0.7)], 1)
    env = D.adsr(n, a, 0.6, 0.92, r, int(gate_s * SR), curve=3)
    return (y * env[:, None] / 2.2).astype(np.float32)


VOWELS = {  # formant (f, gain, q)
    "ah": [(730, 1.0, 7), (1090, 0.55, 9), (2440, 0.25, 12), (3400, 0.12, 14)],
    "oh": [(520, 1.0, 7), (860, 0.6, 9), (2400, 0.18, 12), (3300, 0.08, 14)],
    "oo": [(330, 1.0, 6), (800, 0.35, 9), (2300, 0.10, 12), (3200, 0.05, 14)],
}


def choir_pad_note(midi, gate_s, rng, vowel="oh", a=0.6, r=1.8, voices=4):
    """Synth choir: glottal-ish pulse sources with vibrato + jitter through a parallel formant bank,
    plus breath noise. Several singers per note for ensemble."""
    n = int((gate_s + r + 0.1) * SR)
    f0 = float(mtof(midi))
    t = _t(n)
    src = np.zeros((n, 2), np.float32)
    for j in range(voices):
        vr = 4.8 + rng.random() * 1.0
        vd = 0.004 * np.clip((t - 0.25) / 0.8, 0, 1)
        jit = D.lp(rng.standard_normal(n).astype(np.float32), 6) * 0.02
        fm = 1 + vd * np.sin(2 * np.pi * vr * t + rng.random() * 6) + jit + (rng.random() - 0.5) * 0.006
        s = osc((f0 * fm).astype(np.float32), n, "pulse", rng.random(), 0.18 + 0.1 * rng.random())
        s = D.lp(s, 3000)
        gl, gr = D.pan_gains(-0.75 + 1.5 * j / max(1, voices - 1))
        src[:, 0] += s * gl; src[:, 1] += s * gr
    br = D.hp(rng.standard_normal((n, 2)).astype(np.float32), 800) * 0.05
    src = src + br
    out = np.zeros_like(src)
    for (f, g, q) in VOWELS[vowel]:
        for c in range(2):
            out[:, c] += svf(src[:, c], np.full(n, f, np.float32), q, "bp") * g
    env = D.adsr(n, a, 0.8, 0.9, r, int(gate_s * SR), curve=3)
    return (out * env[:, None] / voices * 1.6).astype(np.float32)


# =============================================================================== BASS
def sub_line(notes, n, att_ms=4, rel_ms=25):
    """clean sine sub (mono). notes = [(start, len, midi, vel)]; pitch held smoothly"""
    freq, amp, _ = line_curves(notes, n, glide_ms=6, rel_ms=rel_ms, att_ms=att_ms, gap_ms=0, vib=(5, 0, 9))
    y = sine(freq, n) + 0.12 * sine(freq * 2, n)
    return (y * amp).astype(np.float32)


def midbass_note(midi, vel, gate_s, rng, cut=900, env_amt=1.0, decay=0.11, res=0.25, drive=2.2, reese=0.0):
    """mid bass: saw+square through saturating ladder with per-note filter env. HP'd later above sub."""
    n = int((gate_s + 0.12) * SR)
    f0 = float(mtof(midi))
    t = _t(n)
    y = osc(np.full(n, f0 * 1.003, np.float32), n, "saw", rng.random()) * 0.6
    y += osc(np.full(n, f0 * 0.997, np.float32), n, "pulse", rng.random(), 0.5) * 0.45
    if reese:
        y += osc(np.full(n, f0 * 2 ** (14 / 1200), np.float32), n, "saw", rng.random()) * reese
        y += osc(np.full(n, f0 * 2 ** (-14 / 1200), np.float32), n, "saw", rng.random()) * reese
    fc = (cut * (0.45 + env_amt * vel * np.exp(-t / decay)) + f0).astype(np.float32)
    y = ladder(y, fc, res, drive)
    env = D.adsr(n, 0.002, 0.2, 0.85, 0.035, int(gate_s * SR))
    return (y * env * vel).astype(np.float32)


# =============================================================================== DRUMS
def kick(rng, vel=1.0, tone=52.0, decay=0.16, click=1.0, punch=1.0):
    n = int(0.55 * SR)
    t = _t(n)
    f = tone + 105 * punch * np.exp(-t / 0.026) + 330 * np.exp(-t / 0.0032)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.exp(-np.maximum(t - 0.045, 0) / decay) * _rise(n, 0.4)
    body = np.sin(ph) * env
    cl = D.bp(rng.standard_normal(n), 1500, 9000) * np.exp(-t / 0.0022) * 0.35 * click
    y = body * 1.3 + cl
    y = D.saturate(y.astype(np.float32), 1.8)
    y = D.fade_edges(y, 0.0002, 0.02)
    return (y * vel).astype(np.float32)


def clap(rng, vel=1.0, tail=0.11):
    n = int(0.45 * SR)
    t = _t(n)
    out = np.zeros((n, 2))
    for c in range(2):
        nz = rng.standard_normal(n)
        e = np.zeros(n)
        for k, off in enumerate([0.0, 0.0085, 0.0175, 0.028]):
            m = t >= off
            e[m] += np.exp(-(t[m] - off) / 0.0045) * (0.8 if k < 3 else 1.0)
        m = t >= 0.028
        e[m] += np.exp(-(t[m] - 0.028) / tail) * 0.55
        y = D.bp(nz * e, 900, 5200)
        y = D.eq(y, ("peak", 1250, 1.2, 5), ("peak", 3500, 1.5, 2))
        out[:, c] = y
    return (out * vel * 0.9).astype(np.float32)


def snare(rng, vel=1.0, tone=195.0, snap=0.12, body=1.0):
    n = int(0.4 * SR)
    t = _t(n)
    f = tone * (1 + 0.25 * np.exp(-t / 0.012))
    b = (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.055)
         + 0.5 * np.sin(2 * np.pi * np.cumsum(f * 1.72) / SR) * np.exp(-t / 0.035)) * body
    nz = D.bp(rng.standard_normal((n, 2)), 1600, 11000) * np.exp(-t / snap)[:, None]
    y = b[:, None] * 0.7 + nz * 0.75
    return D.fade_edges((y * vel).astype(np.float32), 0.0002, 0.01)


_HAT_F = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * 1.63


def hat(rng, vel=1.0, decay=0.035, tone=1.0, open_=False):
    n = int((0.6 if open_ else 0.18) * SR)
    t = _t(n)
    m = np.zeros(n, np.float32)
    for f in _HAT_F * tone:
        m += osc(np.full(n, f, np.float32), n, "pulse", rng.random(), 0.5)
    m = D.bp(m, 7000, 15000, 2)
    nz = D.hp(rng.standard_normal(n).astype(np.float32), 8000) * 0.8
    y = (m * 0.35 + nz) * np.exp(-t / decay) * _rise(n, 0.3)
    y = D.eq(y, ("peak", 10500, 1.0, 3))
    return D.fade_edges((y * vel * 0.5).astype(np.float32), 0.0002, 0.005)


def shaker(rng, vel=1.0):
    n = int(0.12 * SR)
    t = _t(n)
    e = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.03)
    y = D.bp(rng.standard_normal(n), 4500, 12000) * e
    return (y * vel * 0.6).astype(np.float32)


def ride(rng, vel=1.0):
    n = int(1.6 * SR)
    t = _t(n)
    m = np.zeros(n)
    fr = [3120, 3980, 5210, 6530, 7440, 8900, 10150]
    for i, f in enumerate(fr):
        m += np.sin(2 * np.pi * f * t + rng.random() * 6) * np.exp(-t / (0.35 + 0.15 * rng.random())) / (1 + i * 0.3)
    nz = D.bp(rng.standard_normal(n), 5000, 14000) * np.exp(-t / 0.28) * 0.8
    tk = D.hp(rng.standard_normal(n), 6000) * np.exp(-t / 0.004) * 1.5
    y = (m * 0.18 + nz * 0.5 + tk * 0.4)
    return D.fade_edges((y * vel * 0.4).astype(np.float32), 0.0002, 0.05)


def crash(rng, dur=3.2, vel=1.0, bright=1.0):
    n = int(dur * SR)
    t = _t(n)
    out = np.zeros((n, 2))
    for c in range(2):
        freqs = np.exp(rng.uniform(np.log(420), np.log(13500), 140))
        acc = np.zeros(n)
        for f in freqs:
            tau = 0.5 + 1.4 * rng.random() * (4000 / f) ** 0.25
            acc += np.sin(2 * np.pi * f * t * (1 + 0.004 * np.exp(-t / 0.3)) + rng.random() * 6) * np.exp(-t / tau)
        acc /= np.sqrt(len(freqs))
        nz = D.hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.9)
        nz = D.lp(nz, 9000 * bright + 3000 * np.mean(bright))
        att = D.hp(rng.standard_normal(n), 1500) * np.exp(-t / 0.012) * 1.4
        y = acc * 0.5 + nz * 0.6 + att * 0.7
        out[:, c] = y
    out = D.hp(out, 350)
    out *= _rise(n, 0.5)[:, None]
    return D.fade_edges((out * vel * 0.45).astype(np.float32), 0.0003, 0.4)


def tom(rng, midi=45, vel=1.0):
    n = int(0.7 * SR)
    t = _t(n)
    f0 = float(mtof(midi))
    f = f0 * (1 + 0.7 * np.exp(-t / 0.035))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    y += D.bp(rng.standard_normal(n), 200, 3000) * np.exp(-t / 0.02) * 0.4
    y = D.saturate(y.astype(np.float32), 1.5)
    return D.fade_edges((y * vel * 0.8).astype(np.float32), 0.0002, 0.02)


# =============================================================================== MECHANICAL / FX
def modal(freqs, taus, amps, n, rng, exc_ms=0.25):
    t = _t(n)
    y = np.zeros(n)
    for f, ta, a in zip(freqs, taus, amps):
        y += a * np.sin(2 * np.pi * f * t + rng.random() * 0.5) * np.exp(-t / ta)
    ex = D.hp(rng.standard_normal(n), 2000) * np.exp(-t / (exc_ms * 0.001)) * 0.6
    return (y * _rise(n, 0.08) + ex).astype(np.float32)


def lens_click(rng, vel=1.0):
    """Delicate lens-ring / ratchet detent: three micro-clicks + a tiny settle thunk."""
    n = int(0.12 * SR)
    out = np.zeros(n, np.float32)
    for k, (off, g) in enumerate([(0.0, 0.55), (0.0105, 0.75), (0.0195, 1.0)]):
        m = modal([2230 * (1 + 0.03 * k), 3890, 6150, 9020, 12400], [0.006, 0.005, 0.004, 0.003, 0.002],
                  [1.0, 0.7, 0.5, 0.35, 0.2], n - int(off * SR), rng)
        out[int(off * SR):] += m * g
    th = np.sin(2 * np.pi * 170 * _t(n)) * np.exp(-_t(n) / 0.008) * 0.3
    s = int(0.0195 * SR)
    out[s:] += th[: n - s]
    return D.fade_edges(out * vel * 0.5, 0.0001, 0.02)


def tick(rng, vel=1.0, tock=False):
    n = int(0.06 * SR)
    fr = [1850, 2960, 5150, 7600] if tock else [2650, 4120, 6880, 9900]
    y = modal(fr, [0.007, 0.005, 0.004, 0.002], [1.0, 0.6, 0.4, 0.25], n, rng, exc_ms=0.18)
    return D.fade_edges(y * vel * 0.35, 0.0001, 0.01)


def metal_click(rng, base=1700.0, vel=1.0, tau=0.035):
    n = int(0.25 * SR)
    ratios = [1.0, 2.756, 5.404, 8.933]
    y = modal([base * r for r in ratios], [tau, tau * 0.6, tau * 0.4, tau * 0.25], [1, 0.5, 0.3, 0.15], n, rng, 0.2)
    return D.fade_edges(y * vel * 0.3, 0.0001, 0.03)


def servo(rng, midi=86, vel=1.0, dur=0.11, glide=2.0):
    """tonal servo 'whirr' tick: short pitched glide with motor-AM + band noise"""
    n = int((dur + 0.05) * SR)
    t = _t(n)
    f0 = float(mtof(midi))
    f = f0 * 2 ** ((glide / 12) * np.clip(t / dur, 0, 1) ** 0.6)
    ph = 2 * np.pi * np.cumsum(f) / SR
    am = 1 + 0.35 * np.sin(2 * np.pi * 140 * t)
    y = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * am
    y += D.bp(rng.standard_normal(n), f0 * 0.8, min(f0 * 3, 20000)) * 0.25
    env = _rise(n, 3) * np.exp(-np.maximum(t - dur * 0.4, 0) / (dur * 0.35))
    return D.fade_edges((y * env * vel * 0.25).astype(np.float32), 0.0005, 0.01)


def breath(rng, dur=1.3, vel=1.0, inhale=True):
    n = int((dur + 0.1) * SR)
    t = _t(n)
    nz = rng.standard_normal((n, 2))
    y = np.zeros((n, 2))
    for f, g, q in [(650, 0.6, 2.0), (1500, 0.9, 2.5), (2700, 0.7, 3.0), (4300, 0.35, 2.0)]:
        for c in range(2):
            y[:, c] += svf(nz[:, c].astype(np.float32), np.full(n, f * (1 + 0.08 * c), np.float32), q, "bp") * g
    y += D.hp(nz, 6000) * 0.08
    x = t / dur
    if inhale:
        env = np.clip(x, 0, 1) ** 1.6 * np.clip((1.08 - x) / 0.12, 0, 1)
    else:
        env = np.clip(x / 0.08, 0, 1) * np.exp(-np.maximum(x - 0.08, 0) * 3.5)
    y *= env[:, None] ** 0.9
    return D.fade_edges((y * vel * 0.18).astype(np.float32), 0.005, 0.02)


def noise_riser(rng, dur, f0=350, f1=9000, q=1.4, curve_p=2.2, vel=1.0):
    n = int(dur * SR)
    t = _t(n) / dur
    fc = (f0 * (f1 / f0) ** (t ** 1.3)).astype(np.float32)
    nz = rng.standard_normal((n, 2)).astype(np.float32)
    y = np.stack([svf(nz[:, 0], fc, q, "bp"), svf(nz[:, 1], fc * 1.03, q, "bp")], 1)
    y += D.hp(nz, 5000) * 0.06 * t[:, None] ** 3
    env = (t ** curve_p)
    return D.fade_edges((y * env[:, None] * vel * 0.5).astype(np.float32), 0.01, 0.003)


def downlifter(rng, dur=2.5, f0=7000, f1=200, vel=1.0):
    n = int(dur * SR)
    t = _t(n) / dur
    fc = (f0 * (f1 / f0) ** (t ** 0.7)).astype(np.float32)
    nz = rng.standard_normal((n, 2)).astype(np.float32)
    y = np.stack([svf(nz[:, 0], fc, 1.2, "bp"), svf(nz[:, 1], fc * 0.97, 1.2, "bp")], 1)
    env = (1 - t) ** 2 * _rise(n, 3)
    return D.fade_edges((y * env[:, None] * vel * 0.5).astype(np.float32), 0.001, 0.05)


def sub_boom(rng, dur=1.6, f0=62, f1=32, vel=1.0, tau=0.5):
    n = int(dur * SR)
    t = _t(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.25) + 120 * np.exp(-t / 0.006)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau) * _rise(n, 1)
    y = D.saturate(y.astype(np.float32) * 1.2, 1.4)
    return D.fade_edges((y * vel).astype(np.float32), 0.0003, 0.2)


def ignition(rng, dur=4.5, vel=1.0):
    """Rocket-ignition impact: clean sub drop + low roar with rumble AM + crackle, controlled lows."""
    n = int(dur * SR)
    t = _t(n)
    boom = sub_boom(rng, min(dur, 2.6), 64, 30, 1.0, tau=0.75)
    out = np.zeros((n, 2), np.float32)
    out[: len(boom)] += boom[:, None] * 0.55
    brown = np.cumsum(rng.standard_normal((n, 2)), 0)
    brown = D.hp(brown, 35, 2)
    brown /= np.max(np.abs(brown)) + 1e-9
    roar = D.lp(brown, 900) + D.bp(rng.standard_normal((n, 2)), 150, 2500) * 0.08
    rum = 1 + 0.45 * D.lp(rng.standard_normal(n), 14)[:, None] * 6
    env = (1 - np.exp(-t / 0.004)) * (0.55 * np.exp(-t / 0.35) + 0.45 * np.exp(-t / 1.6))
    roar = roar * rum * env[:, None]
    roar /= np.max(np.abs(roar)) + 1e-9
    out += roar.astype(np.float32) * 0.7
    # crackle
    cr = np.zeros((n, 2))
    k = int(dur * 90)
    pos = rng.integers(0, int(1.8 * SR), k)
    for p, c in zip(pos, rng.integers(0, 2, k)):
        L = min(200, n - p)
        cr[p:p + L, c] += rng.standard_normal(L) * np.exp(-np.arange(L) / 25) * rng.random() * np.exp(-p / SR / 0.6)
    out += D.hp(cr, 2500).astype(np.float32) * 0.25
    # blast: saturated low-mid noise burst (what makes the hit read as loud on any speaker)
    bl = D.bp(rng.standard_normal((n, 2)), 90, 2600, 2)
    benv = (1 - np.exp(-t / 0.0015)) * (0.7 * np.exp(-t / 0.09) + 0.3 * np.exp(-t / 0.7))
    bl = bl * benv[:, None]
    bl = D.eq(bl, ("peak", 180, 1.0, 5), ("peak", 900, 1.0, 2))
    bl /= np.max(np.abs(bl)) + 1e-9
    bl = D.saturate(bl.astype(np.float32) * 1.4, 2.0)
    out += bl * 0.9
    # thump body (tonal, ~2 octaves above the sub so it survives small speakers)
    th = np.sin(2 * np.pi * np.cumsum(95 + 80 * np.exp(-t / 0.02)) / SR) * np.exp(-t / 0.18)
    out += (th * 0.6)[:, None].astype(np.float32)
    # crack
    tr = D.hp(rng.standard_normal((n, 2)), 1800) * np.exp(-t / 0.018)[:, None] * 0.9
    out += tr.astype(np.float32)
    return D.fade_edges(out * vel, 0.0002, 0.6)
