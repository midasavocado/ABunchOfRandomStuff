#!/usr/bin/env python3
"""STILL BEGINNING - original score, 128 BPM, 64 bars, exactly 120.000 s.

Deterministically renders:
  stems/{drums,bass,music,leads,pads,fx}.wav   (pre-master, sum == score_premaster.wav)
  score_premaster.wav                          (summed stems, no limiter, peak ~ -6 dBFS)
  score_master.wav                             (glue comp -> EQ -> 4x oversampled TP limiter, -12 LUFS)
  cues.json, levels.json
Usage: python3 compose.py
"""
import os, sys, json, time
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sbdsp as D
import instruments as I
import score as S
from sbdsp import SR, BEAT, BAR, TOTAL, b2s, nm, undb, add
from score import E8, CH

TAIL = 4 * SR
N = TOTAL + TAIL
S16 = BEAT // 4
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def R(seed):
    return np.random.default_rng(seed)


STEMS = ["drums", "bass", "music", "leads", "pads", "fx"]
stems = {k: np.zeros((N, 2), np.float32) for k in STEMS}
sends = {k: {"plate": np.zeros((N, 2), np.float32), "hall": np.zeros((N, 2), np.float32),
             "delay": np.zeros((N, 2), np.float32)} for k in STEMS}
CUES = {"downbeats": [], "drop_hits": [], "pre_drop_silences": [], "final_chord": None, "transients": []}


def cue(kind, t, label):
    e = {"t": round(t, 6), "frame": round(t * 24, 3), "sample": int(round(t * SR)), "label": label}
    if kind in ("drop_hits", "transients", "downbeats", "pre_drop_silences"):
        CUES[kind].append(e)
    else:
        CUES[kind] = e


# ============================================================================ global masks / sidechain
def window_mask(windows, fade_out=0.02, fade_in=0.001, floor=0.0):
    m = np.ones(N, np.float32)
    for (a, b) in windows:
        a = int(a * SR); b = int(b * SR)
        fo = int(fade_out * SR); fi = max(1, int(fade_in * SR))
        m[a:b] = floor          # fade completes exactly at the cut point (nothing sounds after it)
        m[a - fo:a] = np.minimum(m[a - fo:a], floor + (1 - floor) * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo))))
        m[b - fi:b] = floor
        m[b:b + fi] = floor + (1 - floor) * np.linspace(0, 1, fi)
    return m


CUT1 = (b2s(16, 3) / SR, b2s(17) / SR)    # 29.531 - 30.000 (last beat of bar 16)
CUT2 = (b2s(40) / SR, b2s(41) / SR)       # 73.125 - 75.000 (all of bar 40: held breath)
MASK = window_mask([CUT1, CUT2])
MASK_WET = window_mask([CUT1], fade_out=0.2, floor=0.06) * window_mask([CUT2], fade_out=0.5, floor=0.05)

KICKS = []      # (pos, weight) main sidechain triggers

# pre-drop "drain": rising high-pass on the musical tracks so the low end empties before each drop
DRAIN_SEGS = [(b2s(14), b2s(17), [(b2s(14), 10), (b2s(15), 10), (b2s(16), 90), (b2s(16, 3), 380), (b2s(17), 380)]),
              (b2s(38), b2s(41), [(b2s(38), 10), (b2s(39), 10), (b2s(39, 2), 70), (b2s(40), 320), (b2s(41), 320)])]


def drain(x):
    x = x.copy()
    for a, b, pts in DRAIN_SEGS:
        fc = D.curve([(p - a, v) for p, v in pts], b - a, "exp")
        for c in range(2):
            x[a:b, c] = D.svf(x[a:b, c], fc, 0.707, "hp")
    return x


def duck_env(depth, rel=0.26, hold=0.010, att=0.0025, power=1.6, weight_min=0.0):
    L = int(0.5 * SR)
    tau = np.arange(L) / SR
    shape = np.where(tau < att, tau / att, np.where(tau < att + hold, 1.0,
                     np.clip(1 - (tau - att - hold) / rel, 0, 1) ** power)).astype(np.float32)
    env = np.zeros(N, np.float32)
    for p, w in KICKS:
        if w < weight_min:
            continue
        e = min(N, p + L)
        env[p:e] = np.maximum(env[p:e], shape[: e - p] * w)
    return (1 - depth * env).astype(np.float32)


TRACK_LEVELS = {}
WINDOWS = {"intro": (0, 15), "build": (15, 28.1), "drop1": (30, 60), "brk": (60, 73.1), "drop2": (75, 105),
           "res": (105, 114.3), "end": (114.4, 119)}


def _track_meter(name, stem, x):
    if not os.environ.get("SB_METER"):
        return
    row = {}
    for w, (a, b) in WINDOWS.items():
        seg = x[int(a * SR):int(b * SR)]
        row[w] = round(D.lufs_integrated(seg), 1) if np.abs(seg).max() > 1e-5 else None
    TRACK_LEVELS[f"{stem}/{name}"] = row


def mix(stem, x, gain_db=0.0, plate=0.0, hall=0.0, dly=0.0, duck=None, mask=True, hp=None, lp=None, name="?",
        drained=False):
    x = x.astype(np.float32)
    if drained:
        x = drain(x)
    if hp:
        x = D.hp(x, hp)
    if lp:
        x = D.lp(x, lp)
    x *= undb(gain_db)
    if duck is not None:
        x *= duck[:, None]
    if mask:
        x *= MASK[:, None]
    stems[stem] += x
    _track_meter(name, stem, x)
    if plate:
        sends[stem]["plate"] += x * plate
    if hall:
        sends[stem]["hall"] += x * hall
    if dly:
        sends[stem]["delay"] += x * dly


def buf():
    return np.zeros((N, 2), np.float32)


def span_notes(b0, b1, which=0, octave=0):
    """chord spans -> note list for sub/bass lines: (start, len, midi, 1.0)"""
    out = []
    for s, e, c in S.chord_spans(b0, b1):
        out.append((s, e - s, nm(CH[c][which]) + 12 * octave, 1.0))
    return out


def chord_at(pos):
    bar = pos // BAR + 1
    beat = (pos - (bar - 1) * BAR) / BEAT
    cur = S.HARM[bar][0][1]
    for b, c in S.HARM[bar]:
        if beat >= b - 1e-9:
            cur = c
    return cur


# ============================================================================ PIANO (music stem)
def render_piano():
    rng = R(11)
    ev = []   # (pos, midi, vel, hold_s)

    def pedal_hold(pos, bar_end_bar):
        return (b2s(bar_end_bar) - pos) / SR + 0.12

    def theme(bars, start_bar, vel, transpose=0, jitter=0.03):
        for (s, l, m, v) in S.phrase(bars, start_bar, transpose):
            bar = s // BAR + 1
            ev.append((s, m, vel * (0.88 + 0.12 * v) * (1 + jitter * (rng.random() - 0.5)), pedal_hold(s, bar + 1)))

    def lh_broken(bar, chord, pattern, vel, step=2):
        # pattern of midi notes in eighths
        for i, m in enumerate(pattern):
            s = b2s(bar) + i * (step * E8 // 2)          # step in 16ths
            ev.append((s, m, vel * (1.0 if i == 0 else 0.82) * (1 + 0.06 * (rng.random() - 0.5)), pedal_hold(s, bar + 1)))

    # --- intro seed (bars 2-4)
    theme([S.THEME_A[0]], 2, 0.48)
    theme([S.THEME_A[1]], 3, 0.46)
    ev.append((b2s(2), nm("B2"), 0.40, pedal_hold(b2s(2), 3))); ev.append((b2s(2, 0.02), nm("F#3"), 0.30, pedal_hold(b2s(2), 3)))
    ev.append((b2s(3), nm("G2"), 0.40, pedal_hold(b2s(3), 4))); ev.append((b2s(3, 0.02), nm("D3"), 0.30, pedal_hold(b2s(3), 4)))
    ev.append((b2s(4), nm("A2"), 0.40, pedal_hold(b2s(4), 5))); ev.append((b2s(4, 0.02), nm("E3"), 0.30, pedal_hold(b2s(4), 5)))
    ev.append((b2s(4), nm("E5"), 0.40, pedal_hold(b2s(4), 5)))
    ev.append((b2s(4), nm("D4"), 0.30, 0.95)); ev.append((b2s(4, 2), nm("C#4"), 0.32, pedal_hold(b2s(4, 2), 5)))
    # --- bars 5-8: full theme, restrained, LH broken chords in quarters
    theme(S.THEME_A, 5, 0.56)
    LHQ = {"Bm": ["B2", "F#3", "B3", "D4"], "G": ["G2", "D3", "G3", "B3"], "D": ["D2", "A2", "D3", "F#3"],
           "A": ["A2", "E3", "A3", "C#4"]}
    for i, c in enumerate(["Bm", "G", "D", "A"]):
        lh_broken(5 + i, c, [nm(x) for x in LHQ[c]], 0.36, step=4)
    # --- bars 9-12: answer phrase, softer, block LH
    theme(S.THEME_B, 9, 0.44)
    for i, c in enumerate(["Bm", "G", "D", "A"]):
        lh_broken(9 + i, c, [nm(x) for x in LHQ[c][:2]], 0.30, step=8)
    # --- first drop sparkle doubling (octave up) in phrases 2 and 4
    theme(S.THEME_B, 21, 0.42, 12)
    theme(S.THEME_B, 29, 0.48, 12)
    # --- breakdown: exposed theme + flowing LH eighths
    theme(S.THEME_A_BRK, 33, 0.62, jitter=0.08)
    theme(S.THEME_B, 37, 0.64, jitter=0.08)
    LH8 = {"G": ["G2", "D3", "A3", "B3", "D4", "B3", "A3", "D3"],
           "D/F#": ["F#2", "D3", "A3", "D4", "F#4", "D4", "A3", "D3"],
           "Em7": ["E2", "B2", "G3", "D4", "G4", "D4", "B3", "G3"],
           "A": ["A2", "E3", "A3", "C#4", "E4", "C#4", "A3", "E3"],
           "Asus4": ["A2", "E3", "A3", "D4", "E4", "C#4", "A3", "E3"],
           "Bm7": ["B2", "F#3", "A3", "D4", "F#4", "D4", "A3", "F#3"]}
    for bar, c in zip(range(33, 41), ["G", "D/F#", "Em7", "Asus4", "G", "D/F#", "Bm7", "Asus4"]):
        if bar == 40:
            continue                                   # bar 40 = held breath before ignition
        lh_broken(bar, c, [nm(x) for x in LH8[c]], 0.34 if bar < 37 else 0.37, step=2)
    # --- final drop sparkle doubling (49-56)
    theme(S.THEME_A, 49, 0.46, 12)
    theme(S.THEME_B, 53, 0.52, 12)
    # --- resolution: big chords
    RES = {57: ["G1", "G2", "D3", "B3", "D4", "F#4"], 58: ["F#1", "F#2", "A3", "D4", "F#4"],
           59: ["E1", "E2", "B2", "G3", "D4", "E4"], 60: ["A1", "A2", "E3", "A3", "D4", "E4"]}
    for bar, notes in RES.items():
        for j, x in enumerate(notes):
            ev.append((b2s(bar) + j * 180, nm(x), 0.55 if j < 2 else 0.47, pedal_hold(b2s(bar), bar + 1)))
    ev.append((b2s(60, 2), nm("C#4"), 0.45, pedal_hold(b2s(60, 2), 61)))
    # cadence bar 61 (lead an octave below) + LH
    for (s, l, m, v) in S.phrase(S.CADENCE, 61, -12):
        ev.append((s, m, 0.55, (l + E8) / SR))
    for x, bt in (("G2", 0), ("D3", 0.03), ("A2", 2), ("E3", 2.03)):
        ev.append((b2s(61, bt), nm(x), 0.42, 0.98))
    # final arrival: rolled D major, long ring
    fin = ["D1", "D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"]
    ring = (TOTAL - b2s(62)) / SR + 0.5
    for j, x in enumerate(fin):
        ev.append((b2s(62) + j * 420, nm(x), 0.62 - 0.012 * j, ring))
    # render
    x = buf()
    cuts = [(int(a * SR), int(b * SR)) for a, b in (CUT1, CUT2)]
    ev = [e for e in ev if not any(a <= e[0] < b for a, b in cuts)]
    # humanise: exposed sections (intro, breakdown, ending) get a little timing/velocity life;
    # low notes lead slightly, melody sits a hair behind (a pianist's natural spread)
    hr = R(12)
    hum = []
    for (s, m, v, h) in ev:
        exposed = s < b2s(17) or b2s(33) <= s < b2s(41) or s >= b2s(57)
        if exposed and s > 0:
            dt = int((hr.normal(0, 0.006) + (0.004 if m >= 72 else 0.0)) * SR)
            v = v * (1 + hr.normal(0, 0.05))
            s = max(0, s + dt)
        hum.append((s, m, v, h))
    ev = hum
    for (s, m, v, h) in ev:
        add(x, I.piano(m, float(np.clip(v, 0.05, 1.0)), h, rng), s)
    x = I.soundboard(x, rng, 0.22)
    x = D.eq(x, ("hp", 45, 0.7, 0), ("peak", 250, 1.0, -1.5), ("highshelf", 5000, 0.7, 2.0))
    # piano level automation: tuck under the lead in drops
    g = D.curve([(0, 1.3), (b2s(9), 1.3), (b2s(13), 1.0), (b2s(17), 1.0), (b2s(17) + 1, 0.8), (b2s(33) - 1, 0.8), (b2s(33), 1.0),
                 (b2s(41), 1.0), (b2s(41) + 1, 0.8), (b2s(57) - 1, 0.8), (b2s(57), 1.0), (N, 1.0)], N)
    x *= g[:, None]
    log("piano", len(ev), "notes")
    return x


# ============================================================================ DRUMS
def variants(fn, k, seed, **kw):
    r = R(seed)
    return [fn(r, **kw) for _ in range(k)]


def place(x, hits, vars_, gain=1.0, pan=0.0, jitter_ms=0.0, rng=None):
    for i, (p, v) in enumerate(hits):
        h = vars_[i % len(vars_)] * v * gain
        if h.ndim == 1:
            h = D.stereo(h, pan)
        j = int((rng.random() - 0.5) * 2 * jitter_ms * 0.001 * SR) if (rng is not None and jitter_ms) else 0
        add(x, h, p + j)


def render_drums():
    rng = R(21)
    out = {}
    K = variants(I.kick, 2, 1, decay=0.11)
    KB = variants(I.kick, 1, 2, decay=0.13, tone=51.0, punch=1.15)       # bigger kick for final drop
    CL = variants(I.clap, 4, 3)
    SN = variants(I.snare, 4, 4)
    SNR = variants(I.snare, 4, 5, tone=210.0, snap=0.07, body=0.6)      # roll snare (tighter)
    HT = variants(I.hat, 6, 6)
    OH = variants(I.hat, 3, 7, decay=0.2, open_=True)
    SH = variants(I.shaker, 6, 8)
    RD = variants(I.ride, 3, 9)
    TM = {m: variants(I.tom, 2, 10 + m, midi=m) for m in (40, 45, 50, 55)}

    kick = buf(); kickb = buf(); clap = buf(); snare = buf(); hats = buf(); perc = buf(); cym = buf()
    toms = buf(); roll = buf()

    def beats(b0, b1, pos=(0, 1, 2, 3)):
        return [b2s(b, p) for b in range(b0, b1 + 1) for p in pos]

    # ---- build kicks (filtered) bars 9-16 (except pull-out beat) and 39-40
    bk = [(p, 0.9) for p in beats(9, 15)]
    place(kickb, bk, K)
    bk2 = [(p, 0.85) for p in beats(38, 39) if p < b2s(39, 2)]
    place(kickb, bk2, K)
    for p, _ in bk:
        KICKS.append((p, 0.45))
    for p, _ in bk2:
        KICKS.append((p, 0.35))
    fc = D.curve([(0, 120), (b2s(9), 110), (b2s(13), 200), (b2s(16), 700), (b2s(17), 700),
                  (b2s(38), 150), (b2s(40), 900), (N, 900)], N, "exp")
    kickb = np.stack([D.svf(kickb[:, c], fc, 0.9) for c in range(2)], 1)
    # ---- main drop kicks
    dk = [(p, 1.0) for p in beats(17, 32)]
    dk = [(p, v) for p, v in dk if not (b2s(32, 3) <= p < b2s(33))]        # bar 32 beat 4 = fill
    place(kick, dk, K)
    fk = [(p, 1.0) for p in beats(41, 56)]
    # extra drive: 16th pickup kick before each 4-bar phrase
    fk += [(b2s(b, 3.75), 0.55) for b in (44, 48, 52)]
    place(kick, fk, KB)
    # resolution: half-time then away
    rk = [(b2s(57, 0), 1.0), (b2s(57, 2), 0.9), (b2s(58, 0), 0.95), (b2s(58, 2), 0.85), (b2s(59, 0), 0.8)]
    place(kick, rk, KB)
    for p, v in dk + fk + rk:
        KICKS.append((p, v))
    # ---- claps / snares
    cl = [(b2s(b, x), 1.0) for b in range(13, 17) for x in (1, 3) if b2s(b, x) < b2s(16, 3)]
    cl = [(p, 0.55 + 0.45 * (p - b2s(13)) / (3 * BAR)) for p, _ in cl]
    cl += [(b2s(b, x), 1.0) for b in range(17, 33) for x in (1, 3) if not (b == 32 and x == 3)]
    cl += [(b2s(b, x), 1.0) for b in range(41, 57) for x in (1, 3)]
    place(clap, cl, CL)
    sn = [(b2s(b, x), 0.8) for b in range(41, 57) for x in (1, 3)]
    place(snare, sn, SN)
    # ---- hats
    hh = []
    for b in range(11, 17):          # build 16ths, growing
        for k in range(16):
            p = b2s(b) + k * S16
            if p >= b2s(16, 3):
                continue
            prog = (b - 11) / 6
            v = (0.35 + 0.5 * prog) * (1.0 if k % 4 == 2 else 0.55)
            hh.append((p, v))
    for b in range(17, 33):
        ph = (b - 17) // 4
        for k in range(16):
            p = b2s(b) + k * S16
            if b == 32 and k >= 12:
                continue
            if k % 4 == 2:
                if ph < 2:
                    hh.append((p, 0.95))
            elif ph >= 1 and k % 2 == 1:
                hh.append((p, 0.32))
            elif ph >= 1 and k % 4 == 0:
                hh.append((p, 0.22))
    for b in range(41, 57):
        for k in range(16):
            p = b2s(b) + k * S16
            v = [0.5, 0.3, 0.75, 0.3][k % 4]
            hh.append((p + (int(0.028 * S16) if k % 2 else 0), v))
    place(hats, hh, HT, pan=0.18, jitter_ms=1.5, rng=rng)
    oh = [(b2s(b, x + 0.5), 0.8) for b in range(15, 17) for x in range(4) if b2s(b, x + 0.5) < b2s(16, 3)]
    oh += [(b2s(b, x + 0.5), 0.9) for b in range(25, 33) for x in range(4) if not (b == 32 and x >= 3)]
    oh += [(b2s(b, x + 0.5), 0.85) for b in range(41, 57) for x in range(4)]
    place(hats, oh, OH, pan=-0.12)
    # shaker 16ths with swing
    sh = []
    for b in list(range(21, 33)) + list(range(41, 57)) + [57, 58]:
        for k in range(16):
            if b == 32 and k >= 12:
                continue
            p = b2s(b) + k * S16 + (int(0.05 * S16) if k % 2 else 0)
            sh.append((p, [0.5, 0.35, 0.8, 0.4][k % 4]))
    place(perc, sh, SH, pan=-0.35, jitter_ms=2, rng=rng)
    # ride
    rd = [(b2s(b, x), 0.8 if x % 2 == 0 else 0.6) for b in range(29, 33) for x in range(4) if not (b == 32 and x >= 3)]
    rd += [(b2s(b, x), 0.8 if x % 2 == 0 else 0.6) for b in range(49, 57) for x in range(4)]
    rd += [(b2s(b, x), 0.55) for b in (57, 58, 59) for x in (0, 1, 2, 3)]
    place(cym, rd, RD, pan=0.3)
    # ---- crashes (rendered once per hit)
    crg = R(30)
    for bar, v in [(17, 1.0), (21, 0.6), (25, 0.8), (29, 0.7), (33, 1.0), (41, 1.0), (45, 0.7), (49, 0.9),
                   (53, 0.8), (57, 1.0), (62, 0.55)]:
        add(cym, I.crash(crg, 3.6 if bar not in (33, 57, 62) else 5.0, v), b2s(bar))
    # ---- fills
    def sn_run(bar, beat0, beat1, div, v0, v1, arr=roll):
        k = 0
        pts = np.arange(beat0, beat1 - 1e-9, 1.0 / div)
        for i, bt in enumerate(pts):
            v = v0 + (v1 - v0) * i / max(1, len(pts) - 1)
            add(arr, SNR[k % 4] * v, b2s(bar, bt)); k += 1

    sn_run(20, 3, 4, 4, 0.5, 0.9)
    for i, (bt, m) in enumerate([(2.0, 55), (2.5, 55), (3.0, 50), (3.25, 50), (3.5, 45), (3.75, 40)]):
        add(toms, D.stereo(TM[m][i % 2][:, 0] if TM[m][i % 2].ndim == 2 else TM[m][i % 2], 0.4 - 0.2 * i), b2s(24, bt))
    sn_run(28, 3, 4, 4, 0.55, 1.0)
    sn_run(32, 2, 3, 4, 0.5, 0.8)
    for i, (bt, m) in enumerate([(3.0, 55), (3.25, 50), (3.5, 45), (3.75, 40)]):
        add(toms, D.stereo(TM[m][i % 2], 0.35 - 0.23 * i) * 1.1, b2s(32, bt))
    # build rolls: bar 15 8ths, bar 16 16ths -> 32nds (stop at pull-out)
    sn_run(15, 0, 4, 2, 0.25, 0.45)
    sn_run(16, 0, 2, 4, 0.45, 0.7)
    sn_run(16, 2, 3, 8, 0.7, 1.0)
    sn_run(38, 0, 4, 2, 0.2, 0.4)
    sn_run(39, 0, 2, 4, 0.4, 0.65)
    sn_run(39, 2, 4, 8, 0.65, 1.0)
    # final drop fills
    sn_run(44, 3, 4, 4, 0.5, 0.9)
    for i, (bt, m) in enumerate([(2.5, 55), (3.0, 55), (3.25, 50), (3.5, 45), (3.75, 40)]):
        add(toms, D.stereo(TM[m][i % 2], 0.4 - 0.2 * i), b2s(48, bt))
    sn_run(52, 3, 4, 8, 0.5, 1.0)
    for i, (bt, m) in enumerate([(2.0, 55), (2.25, 55), (2.5, 50), (2.75, 50), (3.0, 45), (3.25, 45), (3.5, 40), (3.75, 40)]):
        add(toms, D.stereo(TM[m][i % 2], 0.45 - 0.13 * i) * (0.8 + 0.05 * i), b2s(56, bt))
    sn_run(56, 3, 4, 4, 0.4, 0.8)
    # tom groove under the last final-drop phrase (tribal lift)
    for b in range(53, 56):
        for bt, m, v in ((1.75, 45, 0.45), (3.5, 50, 0.4), (3.75, 45, 0.5)):
            add(toms, D.stereo(TM[m][b % 2], -0.3) * v, b2s(b, bt))
    # resolution toms on downbeats (timpani-like)
    for b in (59, 60):
        add(toms, D.stereo(TM[40][0], 0.0) * 0.7, b2s(b))

    # ---- drum bus processing
    kick_all = kick + kickb
    kick_all = D.eq(kick_all, ("peak", 60, 1.0, 1.5), ("peak", 350, 1.2, -3), ("peak", 3500, 1.0, 1.5))
    mix("drums", kick_all, 0.0, plate=0.0, name="kick_all")
    mix("drums", clap, -7.5, plate=0.22, hall=0.05, hp=250, name="clap")
    mix("drums", snare, -12.0, plate=0.18, hp=150, name="snare")
    mix("drums", roll, -9.0, plate=0.25, hall=0.08, hp=150, name="roll")
    hats_p = D.eq(hats, ("hp", 5000, 0.7, 0))
    mix("drums", hats_p, -13.0, plate=0.06, name="hats_p")
    mix("drums", perc, -10.0, plate=0.08, hp=3000, name="perc")
    mix("drums", cym, -14.0, hall=0.12, hp=400, name="cym")
    mix("drums", toms, -8.0, plate=0.2, hall=0.08, hp=60, name="toms")
    log("drums")


# ============================================================================ BASS
def render_bass():
    rng = R(31)
    mid = buf()
    # --- intro pulse (bars 1-12): muted 8ths on the root, bar 1 fades in from beat 2
    pulse = np.zeros(N, np.float32)
    for b in range(1, 13):
        for k in range(8):
            if b == 1 and k < 2:
                continue
            p = b2s(b) + k * E8
            c = chord_at(p)
            m = nm(CH[c][1])
            v = (1.0 if k % 2 == 0 else 0.62) * (min(1.0, 0.35 + k * 0.1) if b == 1 else 1.0)
            add(pulse, I.midbass_note(m, v, 0.13, rng, cut=260, env_amt=0.9, decay=0.05, drive=1.3, res=0.1), p)
    pulse = D.lp(pulse, 1100)
    pulse *= D.curve([(0, 1), (b2s(9), 1), (b2s(13), 0.3), (N, 0.3)], N)
    mix("bass", D.stereo(pulse), -11.0, plate=0.05, name="pulse", drained=True)

    # --- bass anticipation: filtered off-beat 8ths (bars 9-16), opening
    ant = np.zeros(N, np.float32)
    for b in range(9, 17):
        for k in range(4):
            p = b2s(b, k + 0.5)
            if p >= b2s(16, 3):
                continue
            prog = (p - b2s(9)) / (8 * BAR)
            m = nm(CH[chord_at(p)][1])
            add(ant, I.midbass_note(m, 0.9, 0.17, rng, cut=180 + 1300 * prog ** 1.5, env_amt=1.2, decay=0.07,
                                    drive=1.6 + prog), p)
    # --- breakdown pulse 8ths (bars 37-40) filtered, opening
    for b in range(37, 41):
        for k in range(8):
            p = b2s(b) + k * E8
            if p >= b2s(40):
                continue
            prog = (p - b2s(37)) / (3 * BAR)
            m = nm(CH[chord_at(p)][1])
            add(ant, I.midbass_note(m, 0.85 if k % 2 else 1.0, 0.16, rng, cut=150 + 900 * prog ** 1.4, env_amt=1.0,
                                    decay=0.06, drive=1.5), p)
    mix("bass", D.stereo(D.hp(ant, 45)), -5.0, name="anticip", drained=True)

    # --- drop rolling bass: 16ths skipping the kick
    def rolling(b0, b1, cut, reese=0.0, octave_jump=False, drive=2.2, skip_last_beat=None):
        for b in range(b0, b1 + 1):
            for bt in range(4):
                if skip_last_beat and b == skip_last_beat and bt == 3:
                    continue
                for k in (1, 2, 3):
                    p = b2s(b, bt) + k * S16
                    c = chord_at(p)
                    m = nm(CH[c][1])
                    if octave_jump and k == 3 and bt % 2 == 1:
                        m += 12
                    v = [0, 0.8, 1.0, 0.85][k]
                    add(mid, D.stereo(I.midbass_note(m, v, 0.095, rng, cut=cut, env_amt=1.1, decay=0.07,
                                                     drive=drive, reese=reese)), p)

    rolling(17, 32, 950, skip_last_beat=32)
    rolling(41, 56, 1250, reese=0.3, octave_jump=True, drive=2.6)
    # --- resolution: sustained mid bass
    for s, e, c in S.chord_spans(57, 61):
        add(mid, D.stereo(I.midbass_note(nm(CH[c][1]), 0.8, (e - s) / SR - 0.02, rng, cut=420, env_amt=0.8,
                                          decay=0.4, drive=1.8)), s)
    ln = (TOTAL - b2s(62)) / SR - 0.3
    fin = I.midbass_note(nm("D2"), 0.8, ln, rng, cut=360, env_amt=0.7, decay=0.5, drive=1.6)
    fin *= np.exp(-np.arange(len(fin)) / SR / 2.2)
    add(mid, D.stereo(fin), b2s(62))
    # mid bass: keep only content above the sub, slight width only above 300 Hz
    lo, hi = D.lr_split(mid, 95)
    hi = D.eq(hi, ("peak", 180, 1.0, 1.0), ("peak", 700, 1.2, 1.5), ("highshelf", 3000, 0.7, -4))
    mix("bass", hi, 0.0, duck=duck_env(0.25), name="midbass")

    # --- sub (mono sine) following the harmony
    notes = []
    for s, l, m, v in span_notes(13, 16):
        notes.append((s, l, m, 0.0))   # placeholder (sub for build uses offbeats below)
    sub = np.zeros(N, np.float32)
    # build: off-beat sub pulses (bars 13-16)
    bnotes = []
    for b in range(13, 17):
        for k in range(4):
            p = b2s(b, k + 0.5)
            if p < b2s(16):
                bnotes.append((p, int(0.2 * SR), nm(CH[chord_at(p)][0]), 0.6 + 0.4 * (b - 13) / 3))
    seg = I.sub_line([(s - b2s(13), l, m, v) for s, l, m, v in bnotes], 4 * BAR, att_ms=3, rel_ms=40)
    add(sub, seg, b2s(13))
    for (b0, b1) in ((17, 32), (41, 56)):
        nts = span_notes(b0, b1)
        if b1 == 32:
            nts[-1] = (nts[-1][0], nts[-1][1] - BEAT, nts[-1][2], 1.0)   # fill beat
        seg = I.sub_line([(s - b2s(b0), l, m, v) for s, l, m, v in nts], (b1 - b0 + 1) * BAR + SR // 2)
        add(sub, seg, b2s(b0))
    # breakdown build sub 39-40 (8ths)
    bn = []
    for b in (38, 39):
        for k in range(8):
            p = b2s(b) + k * E8
            bn.append((p - b2s(38), int(0.17 * SR), nm(CH[chord_at(p)][0]), 0.55 + 0.25 * (b - 38)))
    add(sub, I.sub_line(bn, 2 * BAR + SR // 2, att_ms=3, rel_ms=35), b2s(38))
    # resolution + final
    nts = span_notes(57, 61)
    seg = I.sub_line([(s - b2s(57), l, m, v) for s, l, m, v in nts], 5 * BAR + SR // 2, att_ms=8, rel_ms=60)
    add(sub, seg, b2s(57))
    fl = int((TOTAL - b2s(62)) / 1.0)
    fs = I.sub_line([(0, fl - SR, nm("D2"), 1.0)], fl, att_ms=15, rel_ms=900)
    fs *= np.exp(-np.arange(fl) / SR / 1.8).astype(np.float32)
    add(sub, fs, b2s(62))
    sub = D.lp(sub, 140)
    mix("bass", D.stereo(sub), -9.0, duck=duck_env(0.82, rel=0.24), name="sub", drained=True)
    log("bass")


# ============================================================================ LEADS
def block(notes, pad=2 * SR):
    s0 = min(n[0] for n in notes)
    e0 = max(n[0] + n[1] for n in notes) + pad
    return s0, e0 - s0, [(s - s0, l, m, v) for s, l, m, v in notes]


def render_leads():
    rng = R(41)
    lead = buf(); soft = buf(); ctr = buf(); build = buf()
    # build lead (bars 13-16), heavily filtered, opening
    nts = S.phrase(S.THEME_A, 13, legato=0.96)
    nts = [(s, min(l, b2s(16, 3) - s - 400), m, v) for s, l, m, v in nts if s < b2s(16, 3)]
    s0, n, rel = block(nts)
    add(build, I.lead_line(rel, n, rng, detune=0.16, cut=9000), s0)
    fc = D.curve([(0, 350), (b2s(13), 350), (b2s(16, 3), 3200), (N, 3200)], N, "exp")
    build = np.stack([D.svf(build[:, c], fc, 1.1) for c in range(2)], 1)
    # first drop: A, B, A (+new colour), B with octave lift
    for bar0, th, octv, cut in ((17, S.THEME_A, 0.0, 5000), (21, S.THEME_B, 0.0, 5200),
                                (25, S.THEME_A, 0.25, 5600), (29, S.THEME_B, 0.65, 6200)):
        nts = S.phrase(th, bar0, legato=0.94)
        s0, n, rel = block(nts)
        add(lead, I.lead_line(rel, n, rng, octave_layer=octv, cut=cut) * 0.84, s0)
    # final drop: stronger - wider stack, octave layers, counter-melody
    for bar0, th, octv, cut, cth in ((41, S.THEME_A, 0.35, 6000, S.COUNTER_A), (45, S.THEME_B, 0.45, 6200, S.COUNTER_B),
                                     (49, S.THEME_A, 0.7, 6800, S.COUNTER_A), (53, S.THEME_B, 0.85, 7200, S.COUNTER_B)):
        nts = S.phrase(th, bar0, legato=0.95)
        s0, n, rel = block(nts)
        add(lead, I.lead_line(rel, n, rng, octave_layer=octv, cut=cut, detune=0.22, width=1.0), s0)
        cn = S.phrase(cth, bar0, legato=0.97)
        s0, n, rel = block(cn)
        add(ctr, I.counter_line(rel, n, rng, cut=2600 if bar0 < 49 else 3200), s0)
    # resolution: long-note theme + cadence + arrival
    nts = S.phrase(S.THEME_RES, 57, legato=0.98) + S.phrase(S.CADENCE, 61, legato=0.97)
    arr_len = int(2.2 * BAR)
    nts.append((b2s(62), arr_len, nm(S.ARRIVAL), 1.0))
    s0, n, rel = block(nts, pad=3 * SR)
    ln = I.lead_line(rel, n, rng, octave_layer=0.3, cut=5200, detune=0.2)
    # arrival note: gentle fade over its length
    a0 = b2s(62) - s0
    fade = np.ones(n, np.float32)
    fade[a0:] = np.exp(-np.arange(n - a0) / SR / 1.6)
    add(lead, ln * fade[:, None], s0)
    # soft lead doubles the piano cadence in breakdown bar 40 answer tail (very subtle warmth)
    nts = [x for x in S.phrase(S.THEME_B, 37) if x[0] < b2s(40)]
    s0, n, rel = block(nts)
    add(soft, I.soft_lead(rel, n, rng, cut=2200), s0)

    duck = duck_env(0.12)
    lead_p = D.eq(lead, ("peak", 450, 1.0, -2.0), ("peak", 2800, 1.2, 1.5), ("highshelf", 9000, 0.7, 1.0))
    lead_p = D.saturate(lead_p, 1.3)
    ch = D.chorus(lead_p, 11, 2.5, 0.3)
    lead_p = lead_p + 0.35 * ch
    mix("leads", lead_p, -7.0, plate=0.12, hall=0.22, dly=0.2, duck=duck, name="lead_p")
    mix("leads", build, -9.0, hall=0.25, dly=0.25, name="build", drained=True)
    mix("leads", ctr, -9.5, plate=0.15, hall=0.25, dly=0.08, duck=duck, name="ctr")
    mix("leads", soft, -21.0, hall=0.4, name="soft")
    log("leads")


# ============================================================================ MUSIC: chords + arps
def render_music(piano):
    rng = R(51)
    mix("music", piano, -1.0, plate=0.1, hall=0.32, dly=0.06, name="piano", drained=True)
    chords = buf()

    def chord_block(b0, b1, cut, detune=0.16, a=0.004, s=0.72, r=0.25, extra_top=False, vel=1.0, voices=7):
        for st, en, c in S.chord_spans(b0, b1):
            notes = [nm(x) for x in CH[c][2]]
            if extra_top:
                notes.append(notes[-2] + 12)
            for m in notes:
                x = I.saw_chord_note(m, (en - st) / SR - 0.01, rng, a=a, d=0.35, s=s, r=r, detune=detune, cut=cut,
                                     cut_env=0.5)
                add(chords, x * vel / np.sqrt(len(notes)), st)

    chord_block(17, 28, 3000, vel=0.8)
    chord_block(29, 32, 3800, vel=0.9)
    chord_block(41, 48, 4200, detune=0.2, extra_top=True)
    chord_block(49, 56, 5200, detune=0.22, extra_top=True)
    chord_block(57, 61, 3600, detune=0.18, a=0.03, s=0.9, r=0.6, extra_top=True)
    # final chord: soft supersaw bed with long release
    for m in ["F#3", "A3", "D4", "F#4", "A4"]:
        x = I.saw_chord_note(nm(m), 2.0, rng, a=0.02, d=1.5, s=0.6, r=3.0, detune=0.16, cut=2600)
        add(chords, x / np.sqrt(5) * 0.8, b2s(62))
    chords *= MASK[:, None]
    chords = D.hp(chords, 170)
    chords = D.eq(chords, ("peak", 380, 1.0, -2.5), ("highshelf", 8000, 0.7, -1))
    mix("music", chords, -7.0, plate=0.1, hall=0.25, duck=duck_env(0.55), name="chords")

    # arps
    arp = buf(); hook = buf()
    PAT = [0, 2, 1, 3, 2, 1, 3, 2, 0, 2, 1, 3, 2, 3, 1, 2]

    def arp_run(b0, b1, vel=0.8, oct_=0, arr=arp, stop=None):
        for b in range(b0, b1 + 1):
            for k in range(16):
                p = b2s(b) + k * S16
                if stop and p >= stop:
                    continue
                tones = CH[chord_at(p)][3]
                m = nm(tones[PAT[k]]) + 12 * oct_
                v = vel * (1.0 if k % 4 == 0 else (0.8 if k % 2 == 0 else 0.66))
                x = I.pluck(m, v, 0.085, rng, decay=0.16)
                add(arr, D.stereo(x, 0.35 if k % 2 else -0.35), p)

    def hook_run(b0, b1, vel=0.75):
        # syncopated 3+3+2 counter-arp in the upper register
        pos = [0, 3, 6, 8, 11, 14]
        idx = [2, 1, 0, 2, 1, 3]
        for b in range(b0, b1 + 1):
            for j, k in enumerate(pos):
                p = b2s(b) + k * S16
                tones = CH[chord_at(p)][3]
                m = nm(tones[idx[j]]) + 12
                x = I.pluck(m, vel * (1.0 if j in (0, 3) else 0.8), 0.13, rng, decay=0.28, detune=5, bright=1.2)
                kp = I.kpluck(m + 12, vel * 0.6, 0.3, rng, bright=0.5, t60=1.2)
                nn = min(len(x), len(kp)); x = x.copy(); x[:nn] += 0.5 * kp[:nn]
                add(hook, D.stereo(x, [-0.5, 0.5, 0.0, 0.5, -0.5, 0.0][j]), p)

    arp_run(9, 16, stop=b2s(16, 3))
    arp_run(17, 32, vel=0.7, stop=b2s(32, 3))
    arp_run(41, 56, vel=0.72)
    hook_run(25, 32)
    hook_run(45, 56)
    fc = D.curve([(0, 700), (b2s(9), 700), (b2s(16, 3), 6500), (b2s(17), 3800), (b2s(29), 5200), (b2s(33), 5200),
                  (b2s(41), 4800), (b2s(49), 6000), (N, 6000)], N, "exp")
    arp = np.stack([D.svf(arp[:, c], fc, 0.8) for c in range(2)], 1)
    arp = D.hp(arp, 200)
    arp_g = D.curve([(0, 0.7), (b2s(16, 3), 1.0), (b2s(17), 0.58), (b2s(33), 0.58), (b2s(41), 0.7), (N, 0.7)], N)
    mix("music", arp * arp_g[:, None], -7.0, plate=0.08, hall=0.1, dly=0.35, duck=duck_env(0.35), name="arp", drained=True)
    mix("music", D.hp(hook, 350), -10.5, plate=0.1, hall=0.2, dly=0.3, duck=duck_env(0.2), name="hook")
    log("music")


# ============================================================================ PADS
def render_pads():
    rng = R(61)
    warm = buf(); strings = buf(); choir = buf()

    def pad_block(arr, fn, b0, b1, oct_=0, vel=1.0, extra=None, **kw):
        for st, en, c in S.chord_spans(b0, b1):
            notes = [nm(x) + 12 * oct_ for x in CH[c][2]]
            if extra == "top":
                notes.append(notes[-1] + 12)
            for m in notes:
                x = fn(m, (en - st) / SR, rng, **kw)
                add(arr, x * vel / np.sqrt(len(notes)), st)

    pad_block(warm, I.warm_pad_note, 3, 16, a=1.2, r=1.5, cut=1300)
    pad_block(warm, I.warm_pad_note, 17, 32, a=0.3, r=0.8, cut=1800, vel=0.6)
    pad_block(warm, I.warm_pad_note, 33, 40, a=0.9, r=1.4, cut=1500)
    pad_block(warm, I.warm_pad_note, 41, 56, a=0.3, r=0.8, cut=2000, vel=0.8)
    pad_block(warm, I.warm_pad_note, 57, 61, a=0.5, r=1.5, cut=1800)
    for m in ["D3", "A3", "D4", "F#4"]:
        add(warm, I.warm_pad_note(nm(m), 2.0, rng, a=0.3, r=2.6, cut=1500) / 2, b2s(62))
    pad_block(strings, I.string_pad_note, 13, 16, a=1.5, r=0.6)
    pad_block(strings, I.string_pad_note, 33, 40, a=0.7, r=1.2, oct_=1, vel=0.9)
    pad_block(strings, I.string_pad_note, 41, 56, a=0.35, r=0.8, oct_=1, vel=0.8)
    pad_block(strings, I.string_pad_note, 57, 61, a=0.4, r=1.4, oct_=1, extra="top")
    for m in ["D4", "F#4", "A4", "D5"]:
        add(strings, I.string_pad_note(nm(m), 2.2, rng, a=0.4, r=2.4) / 2, b2s(62))
    pad_block(choir, I.choir_pad_note, 37, 40, oct_=1, vowel="oo", a=1.0, r=1.5, vel=1.6)
    pad_block(choir, I.choir_pad_note, 49, 56, oct_=1, vowel="ah", a=0.4, r=1.2)
    pad_block(choir, I.choir_pad_note, 57, 61, oct_=1, vowel="ah", a=0.5, r=1.6, extra="top")
    for m in ["D4", "A4", "D5", "F#5"]:
        add(choir, I.choir_pad_note(nm(m), 1.8, rng, vowel="oh", a=0.4, r=2.6) / 2, b2s(62))
    # pre-drop build: filter the build strings
    fcw = D.curve([(0, 20000), (b2s(13), 900), (b2s(16, 3), 6000), (b2s(17), 20000), (N, 20000)], N, "exp")
    strings = np.stack([D.svf(strings[:, c], fcw, 0.7) for c in range(2)], 1)
    duck = duck_env(0.5)
    wg = D.curve([(0, 0.7), (b2s(16), 0.7), (b2s(17), 1.0), (b2s(33), 1.0), (b2s(33) + SR, 0.7), (b2s(41), 0.7),
                  (b2s(41) + 1, 1.0), (N, 1.0)], N)
    mix("pads", warm * wg[:, None], -19.0, hall=0.3, duck=duck, hp=120, name="warm", drained=True)
    sc = D.chorus(strings, 14, 3.0, 0.25)
    mix("pads", strings + 0.4 * sc, -17.0, hall=0.35, plate=0.05, duck=duck, hp=150, name="strings", drained=True)
    cc = D.chorus(choir, 18, 4.0, 0.2)
    mix("pads", choir + 0.5 * cc, -24.5, drained=True, hall=0.5, duck=duck_env(0.35), hp=180, name="choir")
    log("pads")


# ============================================================================ FX
def render_fx():
    rng = R(71)
    fx = buf(); mech = buf()
    t = lambda s: int(round(s * SR))
    # opening click + breath
    add(mech, D.stereo(I.lens_click(rng, 0.3), 0.15), t(0.05))
    add(fx, I.breath(rng, 1.25, 0.9), t(0.12))
    cue("transients", 0.05, "opening lens/ratchet click")
    # clock ticks: bars 3-16 (quiet -> present), and breakdown 33-36 (faint)
    for b in range(3, 17):
        for k in range(8):
            p = b2s(b) + k * E8
            if p >= b2s(16, 3):
                continue
            if b < 5 and k % 2:
                continue
            v = (0.35 if b < 5 else 0.55) * (1.0 if k % 2 == 0 else 0.6)
            add(mech, D.stereo(I.tick(rng, v, tock=bool(k % 2)), 0.3 if k % 2 == 0 else -0.3), p)
    for b in range(33, 37):
        for k in range(0, 8, 2):
            add(mech, D.stereo(I.tick(rng, 0.28), 0.35), b2s(b) + k * E8)
    # servo ticks + metal clicks - increasingly connected machine rhythm (bars 9-16)
    SERVO_POS = {9: [3, 10], 10: [3, 10, 14], 11: [3, 6, 10, 14], 12: [1, 3, 6, 10, 14], 13: [1, 3, 6, 9, 10, 14],
                 14: [1, 3, 5, 6, 9, 10, 13, 14], 15: list(range(1, 16, 2)) + [6, 10], 16: list(range(0, 12))}
    for b, poss in SERVO_POS.items():
        for k in sorted(set(poss)):
            p = b2s(b) + k * S16
            if p >= b2s(16, 3):
                continue
            tones = CH[chord_at(p)][3]
            m = nm(tones[k % 4]) + 24
            add(mech, D.stereo(I.servo(rng, m, 0.5 + 0.03 * (b - 9), 0.07 if b >= 15 else 0.1), -0.45 if k % 2 else 0.45), p)
    for b in range(11, 17):
        for k in (2, 7, 12):
            p = b2s(b) + k * S16
            if p < b2s(16, 3):
                add(mech, D.stereo(I.metal_click(rng, 1650 + 300 * (k % 3), 0.6), 0.5 - 0.5 * (k % 3)), p)
    # first-drop phrase 3 percussion change: metallic clicks on the "e"s
    for b in range(25, 29):
        for k in (1, 5, 9, 13):
            add(mech, D.stereo(I.metal_click(rng, 2100, 0.45, tau=0.02), 0.55 if k % 8 == 1 else -0.55), b2s(b) + k * S16)
    # --- build 1: noise riser + tonal riser, pull-out, reverse swell
    rz = I.noise_riser(rng, (b2s(16, 3) - b2s(13)) / SR, 300, 9000)
    add(fx, rz * 0.9, b2s(13))
    # reverse swell into the drop: reversed crash+reverb (ends exactly at the drop)
    cr = I.crash(R(72), 3.0, 1.0)
    wet = D.reverb(cr[:, 0], cr[:, 1], size=1.3, decay=3.0, predelay=10)
    rv = (cr * 0.4 + wet)[::-1]
    L = int(1.75 * SR)
    rv = rv[-L - int(0.01 * SR):-int(0.01 * SR)] if len(rv) > L else rv
    rv = rv[-L:].copy()
    rv *= (np.linspace(0, 1, L) ** 2)[:, None]
    rv = D.fade_edges(rv, 0.02, 0.012)
    add(fx, rv * 0.7, b2s(17) - L - int(0.004 * SR))
    add(fx, I.breath(rng, 1.0, 0.6), b2s(17) - int(1.08 * SR))
    CUES["pre_drop_silences"].append({"t0": CUT1[0], "t1": CUT1[1], "frame0": CUT1[0] * 24, "frame1": CUT1[1] * 24,
                                      "label": "pull-out before first drop: last beat of bar 16 (low end drained over bar 16, kick out for bar 16); reverse swell + inhale"})
    # drop 1 impact
    add(fx, D.stereo(I.sub_boom(rng, 1.4, 62, 36, 0.7, tau=0.4)), b2s(17))
    ig1 = I.ignition(R(75), 2.0, 1.0)
    ig1 = D.saturate(ig1 / np.abs(ig1).max() * 3.0, 1.0) / 3.0 * np.abs(ig1).max()
    add(fx, ig1 * 0.9, b2s(17))
    add(fx, I.downlifter(rng, 2.2, 6000, 300, 0.8), b2s(17))
    cue("drop_hits", b2s(17) / SR, "FIRST DROP - bar 17 downbeat")
    # bar 32 -> 33: short riser + crash/impact tail into the breakdown
    add(fx, I.noise_riser(rng, 1.875, 800, 10000, curve_p=2.5) * 0.7, b2s(32))
    add(fx, D.stereo(I.sub_boom(rng, 1.6, 55, 34, 0.45, tau=0.5)), b2s(33))
    add(fx, I.downlifter(rng, 3.6, 5000, 150, 0.9), b2s(33))
    cue("transients", b2s(33) / SR, "crash/impact into breakdown (bar 33)")
    # build 2: riser (37-40), pull-out click + breath, subtle reverse swell, ignition
    rz = I.noise_riser(rng, (b2s(40) - b2s(37)) / SR, 250, 11000, curve_p=2.6)
    add(fx, rz * 1.0, b2s(37))
    add(mech, D.stereo(I.lens_click(rng, 0.3), -0.1), t(73.30))
    add(fx, I.breath(rng, 1.15, 0.7), t(73.78))
    cue("transients", 73.30, "small mechanical click in the pre-ignition held breath (bar 40)")
    cr2 = I.crash(R(73), 2.0, 1.0)
    wet = D.reverb(cr2[:, 0], cr2[:, 1], size=1.2, decay=2.0, predelay=5)
    rv = (cr2 * 0.3 + wet)[::-1]
    L = int(0.9 * SR)
    rv = rv[-L:].copy() * (np.linspace(0, 1, L) ** 3)[:, None]
    rv = D.fade_edges(rv, 0.02, 0.008)
    add(fx, rv * 0.8, b2s(41) - L - int(0.003 * SR))
    CUES["pre_drop_silences"].append({"t0": CUT2[0], "t1": CUT2[1], "frame0": CUT2[0] * 24, "frame1": CUT2[1] * 24,
                                      "label": "pull-out before final drop: ALL of bar 40 (73.125-75.0, per brief times) - kick/bass/music out; small click @73.30 + inhale to the ignition"})
    ign = I.ignition(rng, 5.0, 1.0)
    ign = D.saturate(ign / np.abs(ign).max() * 3.0, 1.0) / 3.0 * np.abs(ign).max()   # dense, low-crest impact
    add(fx, ign * 1.5, b2s(41))
    add(fx, D.stereo(I.sub_boom(rng, 2.2, 70, 38, 0.6, tau=0.6)), b2s(41))
    cue("drop_hits", b2s(41) / SR, "FINAL DROP / ROCKET IGNITION - bar 41 downbeat (sample 3,600,000)")
    # phrase-change impacts in final drop / resolution
    add(fx, D.stereo(I.sub_boom(rng, 1.2, 58, 36, 0.45, tau=0.35)), b2s(49))
    cue("transients", b2s(49) / SR, "final drop answer phrase lift (bar 49) - crash + boom")
    add(fx, I.noise_riser(rng, 1.875, 600, 9000, curve_p=2.5) * 0.6, b2s(56))
    add(fx, D.stereo(I.sub_boom(rng, 2.4, 52, 30, 0.6, tau=0.9)), b2s(57))
    add(fx, I.downlifter(rng, 4.0, 4000, 180, 0.6), b2s(57))
    cue("transients", b2s(57) / SR, "resolution opens (bar 57) - crash + boom, kick half-time")
    # final arrival swell + soft hit
    cr3 = I.crash(R(74), 2.0, 0.7)
    wet = D.reverb(cr3[:, 0], cr3[:, 1], size=1.3, decay=2.5, predelay=5)
    rv = (cr3 * 0.3 + wet)[::-1]
    L = int(1.2 * SR)
    rv = rv[-L:].copy() * (np.linspace(0, 1, L) ** 2.5)[:, None]
    rv = D.fade_edges(rv, 0.02, 0.01)
    add(fx, rv * 0.6, b2s(62) - L - int(0.003 * SR))
    add(fx, D.stereo(I.sub_boom(rng, 3.0, 50, 36, 0.4, tau=1.2)), b2s(62))
    cue("final_chord", b2s(62) / SR, "FINAL D MAJOR ARRIVAL - bar 62 downbeat")
    # --- picture-sync foley for the scenes of this pass (frames -> seconds at 24 fps)
    fs = lambda f: int(round(f / 24.0 * SR))
    add(mech, D.stereo(I.metal_click(rng, 2600, 0.7, tau=0.02), -0.2), fs(652))            # s08a: pin seats home
    add(mech, D.stereo(I.lens_click(rng, 0.35), -0.15), fs(652) + int(0.03 * SR))
    for k, f in enumerate((680, 683, 686, 689, 692)):                                          # s08b: fingers close
        add(mech, D.stereo(I.servo(rng, 88 + k, 0.35, 0.09), 0.3 - 0.15 * k), fs(f))
    for k, f in enumerate((702, 704, 706, 708, 710)):                                          # ... and open
        add(mech, D.stereo(I.servo(rng, 84 - k, 0.28, 0.08, glide=-2.0), -0.3 + 0.15 * k), fs(f))
    add(mech, D.stereo(I.pencil_scratch(rng, 5.0, 0.55, strokes=[(0.0, 5.0)]), 0.25), fs(1470))   # s17: pencil
    add(mech, D.stereo(I.pencil_scratch(rng, 0.6, 0.35), 0.25), fs(1452))
    add(mech, D.stereo(I.pneumatic(rng, 0.9), 0.1), fs(1789))                                   # s19c: lock bolt
    add(fx, D.stereo(I.latch(rng, 0.8, weight=0.7)), fs(2000))                                 # s21: staging
    add(fx, D.stereo(I.sub_boom(rng, 1.2, 48, 30, 0.35, tau=0.4)), fs(2024))                   # s21: upper stage lights
    add(mech, D.stereo(I.latch(rng, 0.7), -0.25), fs(2136))                                   # s22: bay latches
    add(mech, D.stereo(I.latch(rng, 0.55, weight=1.4), 0.3), fs(2396))                        # s25: mirror wing latch
    cue("transients", 652 / 24.0, "s08a pin seats (metal click)")
    cue("transients", 1789 / 24.0, "s19c lock bolt withdraws (pneumatic clack) in the held breath")
    cue("transients", 2396 / 24.0, "s25 mirror wing latches")
    # bookend: last tiny lens click as the chord decays
    add(mech, D.stereo(I.lens_click(rng, 0.2), 0.2), t(118.2))
    cue("transients", 118.2, "closing lens click (bookend to the opening)")

    mix("fx", mech, -1.0, plate=0.12, hall=0.06, mask=False, hp=150, name="mech")
    fx = D.hp(fx, 28)
    mix("fx", fx, -8.0, hall=0.12, mask=False, name="fx")
    log("fx")


# ============================================================================ returns / stems
def process_returns():
    duck = duck_env(0.3)
    for k in STEMS:
        sd = sends[k]
        pl = D.reverb(sd["plate"][:, 0], sd["plate"][:, 1], size=0.75, decay=1.5, damp=7500, predelay=12, mod=0.25,
                      hp_in=250, lp_in=12000)
        dl = D.delay(sd["delay"][:, 0], sd["delay"][:, 1], 0.75, 0.75, fb=0.38, lp_hz=4500, hp_hz=350, cross=1.0)
        dl_in = sd["delay"][:, 0] * 0  # (placeholder for clarity)
        hin = sd["hall"] + 0.3 * dl
        hl = D.reverb(hin[:, 0], hin[:, 1], size=1.35, decay=3.8, damp=5200, predelay=28, mod=0.45, hp_in=220,
                      lp_in=10000)
        wet = pl * 0.9 + hl * 1.0 + dl * 0.8
        wet = D.eq(wet, ("peak", 400, 1.0, -2.0))
        wet *= (MASK_WET * duck)[:, None]
        stems[k] += wet.astype(np.float32)
        sends[k] = None
    log("returns")


def finalize_stems():
    fade_len = int(1.0 * SR)
    env = np.ones(TOTAL, np.float32)
    env[-fade_len:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade_len))
    env[:48] *= np.linspace(0, 1, 48)   # sample-0 safety ramp (1 ms)
    # impact carve: the band dips briefly under the drop hits so the impacts read as the loudest moments
    carve = np.ones(TOTAL, np.float32)
    for pos, depth, tau in ((b2s(17), 0.1, 0.35), (b2s(41), 0.16, 0.45)):
        tt = np.arange(TOTAL - pos) / SR
        carve[pos:] *= (1 - depth * np.exp(-tt / tau) * np.clip(tt / 0.004, 0, 1)).astype(np.float32)
    # section trim: first drop sits ~1.5 LU under the final drop (ramps hidden in the pull-out / crash)
    trim = D.curve([(0, 1.0), (b2s(16, 3), 1.0), (b2s(17) - 1, 0.92), (b2s(33) - 2400, 0.92), (b2s(33), 1.0),
                    (TOTAL, 1.0)], TOTAL)
    out = {}
    for k in STEMS:
        x = stems[k][:TOTAL] * env[:, None]
        x = D.hp(x, {"leads": 100, "pads": 70, "music": 28, "fx": 24, "drums": 24, "bass": 24}[k], 2)
        if k != "fx":
            x = x * (carve * trim)[:, None]
        # mono below 120 Hz (sub/kick fold): remove side-channel lows
        M = (x[:, 0] + x[:, 1]) / 2
        Sd = (x[:, 0] - x[:, 1]) / 2
        Sd = D.hp(Sd, 120, 4)
        out[k] = np.stack([M + Sd, M - Sd], 1).astype(np.float32)
    return out


def main():
    t_all = time.time()
    render_drums()            # (populates KICKS for sidechain first)
    piano = render_piano()
    render_music(piano); del piano
    render_bass()
    render_leads()
    render_pads()
    render_fx()
    process_returns()
    if TRACK_LEVELS:
        print("track LUFS per window (pre-master, relative):")
        print(f"{'track':22s}" + "".join(f"{w:>8s}" for w in WINDOWS))
        for k, row in TRACK_LEVELS.items():
            print(f"{k:22s}" + "".join(f"{(v if v is not None else float('nan')):8.1f}" for v in row.values()))
    st = finalize_stems()
    import master
    master.deliver(st, CUES, HERE)
    log("done", round(time.time() - t_all, 1), "s")


if __name__ == "__main__":
    main()
