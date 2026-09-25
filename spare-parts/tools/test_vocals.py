#!/usr/bin/env python3
"""Measure the sung lead against the score.

    python3 test_vocals.py --lead lead_dry.wav --report lead_report.json --lines lines.json [--asr WHISPER_DIR] [--mix song.wav]

Pitch   median f0 over each sung syllable's vowel, in cents from the target note.
Timing  where voicing (or the pitch move into a new note) actually starts, against the beat.
Words   Whisper transcript of each line against the lyric, as word error rate.
"""
import argparse
import json
import re

import os
import sys

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FP = 5.0
NUMS = {'4': 'four', '2': 'two', '1': 'one'}


def load16(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 44100
    return resample_poly(x, 160, 441)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def words(s):
    s = s.lower().replace('’', "'").replace('-', ' ')
    return [NUMS.get(w, w) for w in re.findall(r"[a-z0-9']+", s)]


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = range(len(r) + 1), range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1], len(r)


def pitch_and_timing(lead, report):
    """Pitch: median f0 across each vowel. Timing: (1) planned vowel start vs the beat,
    (2) measured moment the voice crosses halfway into each new note (≥ 2 semitones away) vs the beat."""
    x = load16(lead).astype(np.float64)
    f0, t = pw.harvest(x, 16000, f0_floor=70.0, f0_ceil=900.0, frame_period=FP)
    ap = pw.d4c(x, f0, t, 16000)
    clear = ap[:, :int(ap.shape[1] * 3000 / 8000)].mean(1) < 0.35
    f0 = np.where(clear, f0, 0.0)          # only count frames that are clearly sung vowel
    fr = lambda s: int(round(s * 1000 / FP))
    cents, bad, plan, moves = [], [], [], []
    prev = None
    for r in report:
        if r['spoken']:
            prev = None
            continue
        end = r['t'] + r['d']
        plan.append(r['design_nucleus'] - r['t'])
        a, b = r['design_nucleus'] + 0.03, end - 0.04
        if b - a < 0.02:
            a, b = r['design_nucleus'] + 0.015, end - 0.02
        seg = f0[fr(a):fr(b)]
        seg = seg[seg > 0]
        if len(seg) == 0:
            bad.append((r['text'], r['t'], 'no clear vowel'))
        else:
            c = 1200 * np.log2(np.median(seg) / midi_hz(r['midi']))
            cents.append(c)
            if abs(c) > 20:
                bad.append((r['text'], r['t'], f'{c:+.0f} cents'))
        if prev is not None and abs(prev['midi'] - r['midi']) >= 2 and abs(prev['t'] + prev['d'] - r['t']) < 1e-6:
            lo, hi = fr(r['t'] - 0.08), fr(r['t'] + 0.08)
            seg = f0[lo:hi]
            mid = np.log(midi_hz((prev['midi'] + r['midi']) / 2))
            up = r['midi'] > prev['midi']
            ok = seg > 0
            hit = np.nonzero(ok & ((np.log(np.where(ok, seg, 1)) >= mid) if up else (np.log(np.where(ok, seg, 1e9)) <= mid)))[0]
            pre = np.nonzero(ok & ((np.log(np.where(ok, seg, 1e9)) < mid) if up else (np.log(np.where(ok, seg, 1)) > mid)))[0]
            if len(hit) and len(pre) and pre[0] < hit[-1]:
                k = hit[hit > pre[0]]
                if len(k):
                    moves.append(((lo + k[0]) * FP / 1000 - r['t'], r['text']))
        prev = r
    return np.array(cents), bad, np.array(plan), moves


def vowel_onsets(lead, report):
    """Independent beat check on the final audio: where each vowel's low-frequency energy rises after a
    voiceless consonant or silence, against the beat (same instrument sing.py uses while tuning)."""
    import sing
    x, sr = sf.read(lead)
    if x.ndim > 1:
        x = x.mean(1)
    y = resample_poly(x, 80, 147)
    out = []
    for r in report:
        if r['spoken'] or not r.get('clean_onset'):
            continue
        m = sing.vowel_onset(y, r['t'], min(r['d'], 0.15))
        if m is not None:
            out.append((m - r['t'], r['text']))
    return out


def voicing_onsets(lead, report):
    """Second, independent beat check: the first clearly voiced frame (WORLD pitch track with low
    aperiodicity) after a voiceless consonant or silence, against the beat."""
    x = load16(lead).astype(np.float64)
    f0, t = pw.harvest(x, 16000, f0_floor=70.0, f0_ceil=900.0, frame_period=FP)
    ap = pw.d4c(x, f0, t, 16000)
    v = (f0 > 0) & (ap[:, :int(ap.shape[1] * 3000 / 8000)].mean(1) < 0.35)
    out = []
    for r in report:
        if r['spoken'] or not r.get('clean_onset'):
            continue
        lo, hi = int((r['t'] - 0.12) * 1000 / FP), int((r['t'] + 0.1) * 1000 / FP)
        seg = v[lo:hi]
        gaps = np.nonzero(~seg[:int(0.12 * 1000 / FP) + 4])[0]
        if not len(gaps):
            continue
        on = np.nonzero(seg[gaps[-1]:])[0]
        if len(on):
            out.append(((lo + gaps[-1] + on[0]) * FP / 1000 - r['t'], r['text']))
    return out


def asr(model_dir, path, lines):
    import sherpa_onnx
    import glob
    d = model_dir.rstrip('/') + '/'
    pick = lambda pat: sorted(glob.glob(d + pat), key=len)[0]
    enc = [f for f in glob.glob(d + '*-encoder*.onnx') if 'int8' in f] or glob.glob(d + '*-encoder*.onnx')
    dec = [f for f in glob.glob(d + '*-decoder*.onnx') if 'int8' in f] or glob.glob(d + '*-decoder*.onnx')
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=enc[0], decoder=dec[0], tokens=pick('*tokens.txt'),
                                                     language='en', task='transcribe')
    x = load16(path).astype(np.float32)
    BAR = 1.5
    errs = tot = 0
    rows = []
    for i, l in enumerate(lines):
        a = l['bar'] * BAR - 0.09
        b = (lines[i + 1]['bar'] * BAR - 0.09) if i + 1 < len(lines) else a + 2.2
        s = rec.create_stream()
        s.accept_waveform(16000, x[int(a * 16000):int(b * 16000)])
        rec.decode_stream(s)
        ref = ''.join(tk['text'] + ('' if tk['join'] else ' ') for tk in l['toks'])
        e, n = wer(ref, s.result.text)
        errs += e
        tot += n
        rows.append((l['bar'], e, n, s.result.text.strip()))
    return errs / tot, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lead', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--lines', required=True)
    ap.add_argument('--asr')
    ap.add_argument('--mix', nargs='*', default=[])
    args = ap.parse_args()
    report = json.load(open(args.report))
    lines = json.load(open(args.lines))

    cents, bad, plan, moves = pitch_and_timing(args.lead, report)
    print(f'PITCH   {len(cents)} sung syllables measured: median |error| {np.median(np.abs(cents)):.1f} cents, '
          f'95th pct {np.percentile(np.abs(cents), 95):.1f}, worst {np.abs(cents).max():.1f}')
    for b in bad:
        print('   off:', b)
    print(f'PLAN    vowel starts vs beat: worst {np.abs(plan).max() * 1000:.1f} ms, mean |error| {np.abs(plan).mean() * 1000:.1f} ms')
    me = np.array([m for m, _ in moves]) * 1000
    print(f'MOVES   {len(me)} note changes measured in the audio: mean {me.mean():+.1f} ms, median |error| {np.median(np.abs(me)):.1f} ms, worst {np.abs(me).max():.1f} ms')
    for m, w in moves:
        if abs(m) > 0.025:
            print(f'   {w!r} reaches its note {m * 1000:+.0f} ms from the beat')
    on = vowel_onsets(args.lead, report)
    oe = np.array([o for o, _ in on]) * 1000
    print(f'ONSETS  {len(oe)} vowel onsets measured in the audio: mean {oe.mean():+.1f} ms, median |error| {np.median(np.abs(oe)):.1f} ms, '
          f'90th pct {np.percentile(np.abs(oe), 90):.1f} ms, worst {np.abs(oe).max():.1f} ms')
    for o, w in on:
        if abs(o) > 0.03:
            print(f'   {w!r} vowel starts {o * 1000:+.0f} ms from the beat')
    vo = voicing_onsets(args.lead, report)
    ve = np.array([o for o, _ in vo]) * 1000
    print(f'VOICING {len(ve)} voicing onsets (independent check): mean {ve.mean():+.1f} ms, median |error| {np.median(np.abs(ve)):.1f} ms, '
          f'90th pct {np.percentile(np.abs(ve), 90):.1f} ms')
    if args.asr:
        for path in [args.lead] + args.mix:
            rate, rows = asr(args.asr, path, lines)
            print(f'WORDS   {path}: word error rate {rate * 100:.0f}%')
            for bar, e, n, hyp in rows:
                print(f'   bar {bar + 1:2d}  {e}/{n} wrong  heard: {hyp}')


if __name__ == '__main__':
    main()
