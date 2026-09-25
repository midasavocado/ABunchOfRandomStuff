#!/usr/bin/env python3
"""Sing the Spare Parts lyrics with a neural voice.

Kokoro (a StyleTTS2-style TTS model) normally predicts how long each phoneme lasts, a pitch curve and an
energy curve, then renders audio from those. make_singing_model() rewires the ONNX graph so all three can
be supplied from outside. Then, for every phrase:

1. Words become phonemes and are cut into the lyric's syllables around their vowels.
2. Durations are planned so each vowel starts on its note. Consonants before a vowel keep their natural
   length and sit just ahead of the beat, the way singers place them; vowels fill the rest of the note.
3. The melody is fed in as the pitch curve, and vowels hold their energy across long notes.
4. Kokoro follows the pitch curve only to within tens of cents, so every voiced stretch is tuned onto the
   exact melody with TD-PSOLA, which keeps the model's own waveform.
5. Each voice's small output offset (measured on this song by calibrate_latency.py, stored in
   latency.json) is removed when the phrase is placed on the song timeline.

    pip install kokoro-onnx onnx pyworld soundfile scipy numpy
    python3 sing.py --models DIR --lines lines.json --out vocals.wav --stems DIR
"""
import argparse
import json
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
import pyworld as pw
import soundfile as sf
from kokoro_onnx import Kokoro
from onnx import TensorProto, helper
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample_poly, sosfilt

SR = 24000
UNIT = 0.025                  # one Kokoro duration unit = 600 samples
F0_STEP = UNIT / 2            # the pitch curve runs at two frames per unit
FP = 5.0                      # WORLD frame period, ms
SONG_LEN = 40.0
LOWPASS = butter(4, 700, 'lp', fs=SR, output='sos')

VOWELS = set('aeiouæɑɐəɛɜɪʊʌɔɚɒɝᵻᵊɨ')
LENGTH = set('ː')
STRESS = set('ˈˌ')
VOICELESS = set('ptkfθsʃhʧxʔ')
SPOKEN = {'sort', 'of'}
LEAD = 'am_fenrir'
COMFORT_HZ = 350.0            # the model tracks pitch well below this; higher notes are capped, then tuned up
# Kokoro's audio does not sit exactly on its own alignment. calibrate_latency.py sings the real song with
# each voice at zero offset and measures where its vowels land with two independent instruments (vowel
# energy onset, voicing onset); the value is the mean of the two. audio vowel time = plan + LATENCY.
LATENCY = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latency.json')))
VOICELESS_ENERGY = 3.0       # model energy units (≈ -10 silence … 8 full voice)
VOWEL_START_LEAD = 0.018      # phrase-initial vowels (In, Out, Ev-, It, Is): energy says ~35 ms late, voicing says ~0; split the difference
ONSET_EMPHASIS = 1.25         # sung consonants are a little longer than spoken ones
PHON_OVERRIDE = {'a': 'ɐ', 'the': 'ðə'}
SUNG_VOWEL = {'ɐ': 'ʌ', 'ə': 'ʌ', 'ᵻ': 'ɪ'}   # singers open up reduced vowels


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def make_singing_model(src, dst):
    m = onnx.load(src)
    g = m.graph
    swap = {'/encoder/Clip_output_0': 'durations', '/encoder/Squeeze_output_0': 'f0', '/encoder/Squeeze_1_output_0': 'energy'}
    for old in swap:
        g.output.append(helper.make_tensor_value_info(old, TensorProto.FLOAT, None))
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in swap:
                n.input[i] = swap[x]
    g.input.append(helper.make_tensor_value_info('durations', TensorProto.FLOAT, [1, 'T']))
    g.input.append(helper.make_tensor_value_info('f0', TensorProto.FLOAT, [1, 'F']))
    g.input.append(helper.make_tensor_value_info('energy', TensorProto.FLOAT, [1, 'F']))
    onnx.save(m, dst)


class Singer:
    def __init__(self, models):
        base = f'{models}/kokoro-v1.0.onnx'
        path = f'{models}/kokoro-sing2.onnx'
        if not os.path.exists(path):
            make_singing_model(base, path)
        self.k = Kokoro(base, f'{models}/voices-v1.0.bin')
        self.sess = ort.InferenceSession(path)
        self.vocab = set(self.k.tokenizer.vocab.keys()) if hasattr(self.k.tokenizer, 'vocab') else None

    def phonemes(self, word):
        ph = PHON_OVERRIDE.get(word) or self.k.tokenizer.phonemize(word, 'en-us').strip()
        return [c for c in ph if c != ' ' and len(self.k.tokenizer.tokenize(c)) == 1]

    def run(self, chars, voice, durations, f0, energy=None, speed=1.0):
        """Returns audio, the model's own durations, pitch and energy curves (the last two for these durations)."""
        toks = self.k.tokenizer.tokenize(''.join(chars))
        assert len(toks) == len(chars)
        style = np.asarray(self.k._style_for(self.k.get_voice_style(voice), len(toks)), dtype=np.float32)
        out = self.sess.run(None, {
            'tokens': np.array([[0, *toks, 0]], dtype=np.int64), 'style': style,
            'speed': np.array([speed], dtype=np.float32),
            'durations': np.asarray(durations, dtype=np.float32)[None], 'f0': np.asarray(f0, dtype=np.float32)[None],
            'energy': np.asarray(energy if energy is not None else np.zeros(len(f0)), dtype=np.float32)[None]})
        return out[0].ravel().astype(np.float64), out[1].ravel(), out[2].ravel(), out[3].ravel()

    def natural(self, chars, voice, speed=1.1):
        dummy = np.full(len(chars) + 2, 2.0)
        _, d, _, _ = self.run(chars, voice, dummy, np.zeros(int(2 * dummy.sum())), speed=speed)
        return d


def cls(c):
    if c in VOWELS or c in LENGTH:
        return 'V'
    if c in STRESS:
        return 'S'
    return 'U' if c in VOICELESS else 'C'


def syllabify(chars, nsyl):
    """Split one word's phonemes into nsyl syllables → list of (onset, nucleus, coda) index lists."""
    c = [cls(x) for x in chars]
    runs, i = [], 0
    while i < len(c):
        if c[i] == 'V':
            j = i
            while j < len(c) and c[j] == 'V':
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    if not runs:
        runs = [[0, len(c)]]
    while len(runs) > nsyl:          # merge the pair with the fewest phonemes between them
        gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
        k = int(np.argmin(gaps))
        runs[k] = [runs[k][0], runs[k + 1][1]]
        del runs[k + 1]
    while len(runs) < nsyl:          # rare: split the longest vowel run
        k = int(np.argmax([b - a for a, b in runs]))
        a, b = runs[k]
        if b - a >= 2:
            mid = (a + b) // 2
            runs[k:k + 1] = [[a, mid], [mid, b]]
        else:
            runs.insert(k + 1, [b, b])
    # a stress mark right before a vowel belongs to the nucleus
    for r in runs:
        if r[0] > 0 and c[r[0] - 1] == 'S':
            r[0] -= 1
    bounds = [0]
    for k in range(1, len(runs)):
        between = list(range(runs[k - 1][1], runs[k][0]))
        cons = [i for i in between if c[i] != 'S']
        if len(cons) <= 1:
            bounds.append(runs[k - 1][1])
            continue
        a, b = chars[cons[-2]], chars[cons[-1]]
        affricate = (a, b) in (('t', 'ʃ'), ('d', 'ʒ'))
        cluster = (b in 'lɹwj' and a in 'pbtdkɡfθʃv') or (a == 's' and b in 'ptk')
        bounds.append(cons[-2] if affricate or cluster else cons[-1])
    bounds.append(len(chars))
    out = []
    for k, r in enumerate(runs):
        s0, s1 = bounds[k], bounds[k + 1]
        out.append((list(range(s0, r[0])), list(range(r[0], r[1])), list(range(r[1], s1))))
    return out


def plan_phrase(singer, toks, voice, sung=True):
    """Phonemes, syllables, and per-token natural durations for one phrase."""
    words, cur = [], []
    for tk in toks:
        cur.append(tk)
        if not tk['join']:
            words.append(cur)
            cur = []
    if cur:
        words.append(cur)
    chars, syl = [], []          # syl[i] = (onset idx, nucleus idx, coda idx) into chars
    texts = []
    for w in words:
        text = re.sub(r"[^a-z'\- ]", '', ''.join(t['text'] for t in w).lower().replace('’', "'")).strip('-')
        texts.append(text)
        ph = singer.phonemes(text)
        if sung:
            ph = [SUNG_VOWEL.get(c, c) for c in ph]
        base = len(chars)
        if chars:
            chars.append(' ')    # word gap token, given zero length when sung legato
            base += 1
        chars.extend(ph)
        for on, nu, co in syllabify(ph, len(w)):
            syl.append(([base + i for i in on], [base + i for i in nu], [base + i for i in co]))
    nat = singer.natural(chars, voice) * UNIT
    return chars, syl, nat, texts


def sung_durations(toks, chars, syl, nat):
    """Seconds for every token so each nucleus starts on its note. Returns (dur_s, lead-in before first note)."""
    n = len(toks)
    ons = [ONSET_EMPHASIS * sum(nat[1 + i] for i in s[0]) for s in syl]
    cod = [sum(nat[1 + i] for i in s[2]) for s in syl]
    ons[0] = min(ons[0], 0.16)
    for i in range(n):
        gap = toks[i]['d']
        nxt = ons[i + 1] if i + 1 < n and abs(toks[i + 1]['t'] - (toks[i]['t'] + toks[i]['d'])) < 1e-6 else 0.0
        budget = 0.5 * gap
        if cod[i] + nxt > budget:
            f = budget / (cod[i] + nxt)
            cod[i] *= f
            if i + 1 < n and nxt:
                ons[i + 1] *= f
    dur = np.zeros(len(chars) + 2)
    for i, (on, nu, co) in enumerate(syl):
        def spread(idx, total):
            w = np.array([max(nat[1 + j], 1e-3) for j in idx])
            for j, v in zip(idx, total * w / w.sum()):
                dur[1 + j] = v
        if on:
            spread(on, ons[i])
        if co:
            spread(co, cod[i])
        nxt_on = ons[i + 1] if i + 1 < n and abs(toks[i + 1]['t'] - (toks[i]['t'] + toks[i]['d'])) < 1e-6 else 0.0
        nuc = toks[i]['d'] - cod[i] - nxt_on
        st = [j for j in nu if chars[j] in STRESS]
        vw = [j for j in nu if chars[j] not in STRESS]
        for j in st:
            dur[1 + j] = UNIT        # fixed, as in calibrate_latency.py: the model places the vowel relative to it
        spread(vw, nuc - sum(dur[1 + j] for j in st))
    return dur, ons[0]


def to_units(dur, anchors, lead_pad=0.4, tail_pad=0.15):
    """Round token durations to whole units. The rounding phase is chosen so the note starts
    (token boundaries in `anchors`) stay as evenly spaced as the exact plan."""
    dur = dur.copy()
    dur[0], dur[-1] = lead_pad, tail_pad
    pos = np.concatenate([[0], np.cumsum(dur)]) / UNIT
    best = None
    for c in np.linspace(0, 1, 41)[:-1]:
        r = np.round(pos + c)
        e = r[anchors] - (pos[anchors] + c)
        spread = e.max() - e.min()
        if best is None or spread < best[0]:
            best = (spread, r)
    d = np.diff(best[1])
    d[0] = max(d[0], 1)
    return d


def design_f0(chars, syl, d_units, midis, toks, spoken=False):
    """Target pitch at 12.5 ms steps (0 = voiceless) plus each token's frame span."""
    F = int(2 * d_units.sum())
    note_of = {}
    for i, (on, nu, co) in enumerate(syl):
        joined = i > 0 and abs(toks[i]['t'] - (toks[i - 1]['t'] + toks[i - 1]['d'])) < 1e-6
        for j in on:
            note_of[j] = i - 1 if joined else i
        for j in nu + co:
            note_of[j] = i
    edges = np.concatenate([[0], np.cumsum(d_units)]) * 2
    logf = np.full(F, np.nan)
    voiced = np.zeros(F, bool)
    nuc_start = {}
    for j, ch in enumerate(chars):
        a, b = int(edges[j + 1]), int(edges[j + 2])
        if ch == ' ' or j not in note_of:
            continue
        i = note_of[j]
        logf[a:b] = np.log(midi_hz(midis[i]))
        voiced[a:b] = cls(ch) != 'U'
        if j == syl[i][1][0]:
            nuc_start[i] = a
    # fill gaps, then glide between notes (≈ 25 ms) and add scoops and vibrato
    idx = np.arange(F)
    ok = ~np.isnan(logf)
    logf = np.interp(idx, idx[ok], logf[ok])
    logf = uniform_filter1d(logf, 3)
    for i, a in nuc_start.items():
        k = (idx - a) * F0_STEP
        logf += np.where((k >= 0) & (k < 0.045), -0.35 * (1 - k / 0.045), 0) * np.log(2) / 12
        if toks[i]['d'] >= 0.28:
            end = a + toks[i]['d'] / F0_STEP
            win = (k >= 0.14) & (idx < end)
            depth = 0.3 * np.clip((k - 0.14) / 0.15, 0, 1)
            logf += np.where(win, depth * np.sin(2 * np.pi * 5.6 * (k - 0.14)), 0) * np.log(2) / 12
    return np.where(voiced, np.exp(logf), 0.0), nuc_start


def shape_energy(en, chars, syl, d_units):
    """Sung notes sustain: hold each vowel at its opening level (1 unit of droop across the note),
    keep voiced codas near the vowel, and keep voiceless consonants audible at phrase ends."""
    en = en.copy()
    edges = (np.concatenate([[0], np.cumsum(d_units)]) * 2).astype(int)
    span = lambda j: (edges[j + 1], edges[j + 2])
    for on, nu, co in syl:
        a, b = span(nu[0])[0], span(nu[-1])[1]
        if b <= a:
            continue
        peak = en[a:a + max(2, (b - a) // 3)].max()
        k = np.arange(a, b)
        en[a:b] = np.maximum(en[a:b], peak - 1.0 * (k - a) / max(b - a, 1))
        for j in on + co:
            x, y = span(j)
            floor = VOICELESS_ENERGY if cls(chars[j]) == 'U' else peak - 2.5
            en[x:y] = np.maximum(en[x:y], floor)
    return en


def vowel_onset(y, guess, plateau_len):
    """Where a vowel's low-frequency energy rises through half its plateau, searched around `guess` (s)."""
    e = np.sqrt(sosfilt(LOWPASS, y) ** 2)
    e = np.convolve(e, np.ones(96) / 96, 'same')
    plateau = e[int((guess + 0.03) * SR):int((guess + max(plateau_len, 0.06)) * SR)]
    if len(plateau) < 10:
        return None
    th = 0.5 * np.percentile(plateau, 80)
    lo = max(0, int((guess - 0.12) * SR))
    seg = e[lo:int((guess + 0.1) * SR)]
    below = np.nonzero(seg[:int(0.2 * SR)] < 0.15 * th)[0]
    if not len(below):
        return None
    idx = np.nonzero(seg[below[-1]:] > th)[0]
    return (lo + below[-1] + idx[0]) / SR if len(idx) else None


def joined(toks, i):
    return i + 1 < len(toks) and abs(toks[i + 1]['t'] - (toks[i]['t'] + toks[i]['d'])) < 1e-6


def ideal_f0(toks, midis, start, n):
    """The melody on the true beat grid, sampled at WORLD frames of audio that begins at song time `start`:
    each note begins exactly on its beat (≈ 25 ms glide centred there), with a scoop and vibrato."""
    s = start + np.arange(n) * FP / 1000
    idx = np.clip(np.searchsorted([tk['t'] for tk in toks], s, side='right') - 1, 0, len(toks) - 1)
    logf = np.log(midi_hz(np.asarray(midis, float)[idx]))
    logf = uniform_filter1d(logf, 5)
    for i, tk in enumerate(toks):
        k = s - tk['t']
        logf += np.where((k >= 0) & (k < 0.045), -0.35 * (1 - k / 0.045), 0) * np.log(2) / 12
        if tk['d'] >= 0.28:
            win = (k >= 0.14) & (k < tk['d'])
            depth = 0.3 * np.clip((k - 0.14) / 0.15, 0, 1)
            logf += np.where(win, depth * np.sin(2 * np.pi * 5.6 * (k - 0.14)), 0) * np.log(2) / 12
    return np.exp(logf)


def pitch_correct(y, target, passes=2):
    """Two PSOLA passes: the second one fixes whatever the first left over."""
    for _ in range(passes):
        y = _psola_pass(y, target)
    return y


def _psola_pass(y, target):
    """Tune every voiced stretch to `target` (Hz per WORLD frame) with TD-PSOLA: the model's own waveform is
    cut into two-period grains around each glottal pulse and laid back down at the target spacing. Timing is
    untouched and, for the small corrections used here, the voice keeps its detail. Voiceless audio is copied."""
    f0, t = pw.harvest(y, SR, f0_floor=60.0, f0_ceil=900.0, frame_period=FP)
    f0 = pw.stonemask(y, f0, t, SR)
    ap = pw.d4c(y, f0, t, SR)
    low_ap = ap[:, :int(ap.shape[1] * 3000 / (SR / 2))].mean(1)
    hop = int(SR * FP / 1000)
    tgt = np.pad(target, (0, max(0, len(f0) - len(target))), mode='edge')[:len(f0)]
    # the tracker often locks onto a subharmonic at note onsets: take the octave reading nearest the target
    cand = np.stack([f0 * 2.0 ** k for k in (-1, 0, 1, 2)])
    k = np.argmin(np.abs(np.log2(np.maximum(cand, 1) / np.maximum(tgt, 1))), axis=0)
    f0 = np.where(f0 > 0, cand[k, np.arange(len(f0))], 0.0)
    ratio = np.where(f0 > 0, tgt / np.maximum(f0, 1), 1.0)
    voiced = (f0 > 0) & (low_ap < 0.6) & (np.abs(np.log2(ratio)) < 7 / 12)   # still beyond a fifth: leave it
    src_hz = lambda n: f0[min(n // hop, len(f0) - 1)]
    tgt_hz = lambda n: tgt[min(n // hop, len(tgt) - 1)]
    lp = sosfilt(butter(2, 900, 'lp', fs=SR, output='sos'), y)
    out = np.zeros(len(y))
    wsum = np.zeros(len(y))
    frames = np.nonzero(voiced)[0]
    if len(frames):
        runs = np.split(frames, np.nonzero(np.diff(frames) > 1)[0] + 1)
        for r in runs:
            a, b = r[0] * hop, min(len(y), (r[-1] + 1) * hop)
            if b - a < 3 * SR / 150:
                voiced[r] = False
                continue
            marks, n = [], a                     # analysis pulses, snapped to low-passed peaks
            while n < b:
                P = int(SR / src_hz(n))
                lo, hi = max(a, n - P // 4), min(b, n + P // 4 + 1)
                n = lo + int(np.argmax(lp[lo:hi])) if hi > lo else n
                if marks and n <= marks[-1]:
                    n = marks[-1] + P
                marks.append(n)
                n += P
            marks = np.array(marks)
            s = float(a)                        # synthesis pulses, spaced at the exact (fractional) target period
            while s < b:
                si = int(round(s))
                m = marks[np.argmin(np.abs(marks - si))]
                P = int(SR / src_hz(m))
                w = np.hanning(2 * P + 1)
                g0, g1 = m - P, m + P + 1
                o0 = si - P
                lo, hi = max(0, g0, g0 + (0 - o0)), min(len(y), g1, g1 + (len(y) - (o0 + 2 * P + 1)))
                if hi > lo:
                    seg = slice(o0 + (lo - g0), o0 + (hi - g0))
                    out[seg] += y[lo:hi] * w[lo - g0:hi - g0]
                    wsum[seg] += w[lo - g0:hi - g0]
                s += max(SR / tgt_hz(si), 20.0)
    tuned = np.where(wsum > 1e-3, out / np.maximum(wsum, 1e-3), y)
    mask = np.repeat(voiced.astype(float), hop)[:len(y)]
    mask = np.pad(mask, (0, len(y) - len(mask)))
    mask = np.convolve(mask, np.hanning(241) / np.hanning(241).sum(), 'same')
    return mask * tuned + (1 - mask) * y


def measurable(chars, syl, i):
    on = syl[i][0]
    return (bool(on) and all(cls(chars[j]) == 'U' for j in on)) or (not on and i == 0)


def render(singer, voice, chars, syl, nat, toks, midis):
    """Plan durations and pitch, render with Kokoro, and work out where the audio sits on the song timeline."""
    plan = [dict(tk) for tk in toks]
    if not syl[0][0]:        # a vowel straight out of silence takes the model a moment to voice
        plan[0]['t'] -= VOWEL_START_LEAD
        plan[0]['d'] += VOWEL_START_LEAD
    dur, _ = sung_durations(plan, chars, syl, nat)
    d_units = to_units(dur, [1 + s[1][0] for s in syl])
    f0t, nuc = design_f0(chars, syl, d_units, midis, plan)
    fed = np.minimum(f0t, COMFORT_HZ)
    _, _, _, en = singer.run(chars, voice, d_units, fed)
    y, _, _, _ = singer.run(chars, voice, d_units, fed, shape_energy(en, chars, syl, d_units))
    origin = float(np.mean([tk['t'] - nuc[i] * F0_STEP for i, tk in enumerate(plan)]))
    return y, origin - LATENCY[voice]


def sing_phrase(singer, voice, toks, midis, report, spoken):
    chars, syl, nat, texts = plan_phrase(singer, toks, voice, sung=not spoken)
    if spoken:
        d_units = np.maximum(np.round(nat / UNIT), 0)
        d_units[0], d_units[-1] = 16, 6
        _, _, f0p, en = singer.run(chars, voice, d_units, np.zeros(int(2 * d_units.sum())))
        y, _, _, _ = singer.run(chars, voice, d_units, f0p, en)
        edges = np.concatenate([[0], np.cumsum(d_units)]) * UNIT
        start = toks[0]['t'] - edges[1 + syl[0][1][0]] - LATENCY[voice]
        for tk in toks:
            report.append(dict(t=tk['t'], d=tk['d'], midi=None, text=tk['text'], spoken=True))
        return start, y
    y, start = render(singer, voice, chars, syl, nat, toks, midis)
    y = pitch_correct(y, ideal_f0(toks, midis, start, len(y) // int(SR * FP / 1000) + 2))
    for i, tk in enumerate(toks):
        report.append(dict(t=tk['t'], d=tk['d'], midi=midis[i], text=tk['text'], spoken=False,
                           design_nucleus=tk['t'], clean_onset=measurable(chars, syl, i)))
    return start, y


def phrases_of(line):
    toks = line['toks']
    out = [[toks[0]]]
    for a, b in zip(toks, toks[1:]):
        if abs(b['t'] - (a['t'] + a['d'])) < 1e-6:
            out[-1].append(b)
        else:
            out.append([b])
    return out


def sing_track(singer, voice, lines, transpose, gain, pick=None, harm_index=None, offset=0.0, cents=0.0, report=None):
    out = np.zeros(int(SONG_LEN * SR))
    rep = [] if report is None else report
    for line in lines:
        if pick and not pick(line):
            continue
        if harm_index is not None and not line.get('harm'):
            continue
        midis = line['harm'][harm_index] if harm_index is not None else [tk['midi'] for tk in line['toks']]
        mm = {id(tk): m + transpose + cents / 100 for tk, m in zip(line['toks'], midis)}
        for toks in phrases_of(line):
            texts = {re.sub(r'[^a-z]', '', tk['text'].lower()) for tk in toks}
            spoken = texts <= SPOKEN
            if spoken and report is None:
                continue                  # only the lead speaks the aside
            start, y = sing_phrase(singer, voice, toks, [mm[id(tk)] for tk in toks], rep, spoken)
            # silence the model's warm-up pad and tail so phrases never bleed into each other
            gate = np.zeros(len(y))
            a, b = int((toks[0]['t'] - 0.22 - start) * SR), int((toks[-1]['t'] + toks[-1]['d'] + 0.2 - start) * SR)
            gate[max(0, a):max(0, b)] = 1
            gate = np.convolve(gate, np.hanning(481) / np.hanning(481).sum(), 'same')
            y = y * gate
            v = y[np.abs(y) > 0.02 * np.abs(y).max()]
            y = y / (np.sqrt(np.mean(v ** 2)) + 1e-9) * gain * (1.2 if line.get('shout') else 1.0) * (0.8 if spoken else 1.0)
            fi, fo = int(0.004 * SR), int(0.03 * SR)
            y[:fi] *= np.linspace(0, 1, fi)
            y[-fo:] *= np.linspace(1, 0, fo)
            s = int(round((start + offset) * SR))
            a, b = max(0, s), min(len(out), s + len(y))
            out[a:b] += y[a - s:b - s]
    print(f'  sang {voice} ({transpose:+d})')
    return out


def pan(x, p):
    th = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], axis=1)


def to44(x):
    return resample_poly(x, 147, 80, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--lines', default='lines.json')
    ap.add_argument('--out', default='vocals.wav')
    ap.add_argument('--stems', default=None)
    args = ap.parse_args()
    lines = json.load(open(args.lines))
    singer = Singer(args.models)

    chorus = lambda l: 18 <= l['bar'] < 24
    final = lambda l: l['bar'] == 24
    report = []
    lead = sing_track(singer, LEAD, lines, -12, 0.10, report=report)      # an octave below the written line
    # the last chorus and the title shout are a crowd: different voices, each a hair off in time and tuning.
    # (voice, transpose, gain, which lines, harmony part, time offset s, detune cents, pan)
    crowd = [
        ('am_puck', -12, 0.050, chorus, None, 0.004, +3, -0.45),
        ('am_liam', -12, 0.045, chorus, None, -0.003, -4, 0.45),
        ('af_heart', -12, 0.040, chorus, None, 0.002, +2, -0.25),
        ('af_bella', -12, 0.035, chorus, None, -0.004, -3, 0.3),
        ('am_puck', -12, 0.055, final, 0, 0.003, +2, -0.5),       # D4
        ('am_liam', -12, 0.055, final, 1, -0.002, -3, 0.5),       # B3
        ('af_heart', -12, 0.045, final, None, 0.004, +3, -0.2),   # G4 with the lead
        ('af_nicole', -12, 0.040, final, None, -0.003, -2, 0.25),
        ('af_bella', 0, 0.040, final, 1, 0.002, +2, 0.35),        # B4
    ]
    tracks = [(lead, 0.0)]
    for voice, tr, gain, pick, hi, off, cents, p in crowd:
        tracks.append((sing_track(singer, voice, lines, tr, gain, pick=pick, harm_index=hi, offset=off, cents=cents), p))
    mix = to44(sum(pan(t, p) for t, p in tracks))
    peak = np.abs(mix).max()
    scale = 0.95 / peak if peak > 0.95 else 1.0
    sf.write(args.out, (mix * scale).astype(np.float32), 44100)
    print('wrote', args.out, f'peak {peak:.2f}')
    if args.stems:
        sf.write(f'{args.stems}/lead_dry.wav', (to44(lead) * scale).astype(np.float32), 44100)
        json.dump(report, open(f'{args.stems}/lead_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
