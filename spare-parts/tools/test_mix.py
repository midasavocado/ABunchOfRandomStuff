#!/usr/bin/env python3
"""Check the song as the page plays it (renders from render_page.js).

    python3 test_mix.py RENDER_DIR vocals.wav

Vocal sync   cross-correlation of the page's vocal against the sung track: must be 0 ms.
Tuning       every guitar string's pitch against equal temperament (A = 440 Hz).
Drums        kick-drum onsets against the 160 BPM grid.
Balance      vocal level against the band while singing.
"""
import json
import sys

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import butter, correlate, sosfiltfilt

BPM, SR = 160, 44100
BAR, E8 = 240 / BPM, 30 / BPM


def load(p):
    x, sr = sf.read(p)
    assert sr == SR
    return x.mean(1) if x.ndim > 1 else x


def main():
    d, ref_path = sys.argv[1], sys.argv[2]
    vox = json.load(open(f'{d}/vox.json'))
    print(f"VOCAL   decoded: {vox['state']}, song zero found at {vox['offset'] * 1000:.2f} ms into the file (0.500 s written)")

    page, ref = load(f'{d}/vocal.wav'), load(ref_path)
    n = min(len(page), len(ref))
    # compare in 1-5 kHz with zero-phase filtering, so the page's vocal EQ cannot fake a delay
    bp = butter(4, [1000, 5000], 'bandpass', fs=SR, output='sos')
    a, b = sosfiltfilt(bp, page[:n]), sosfiltfilt(bp, ref[:n])
    lags = []
    for w0 in np.arange(3, 37, 4.0):
        seg = a[int(w0 * SR):int((w0 + 4) * SR)]
        c = correlate(seg, b[int(w0 * SR) - 2000:int((w0 + 4) * SR) + 2000], mode='valid', method='fft')
        lags.append((2000 - np.argmax(c)) / SR * 1000)
    print(f'SYNC    page vocal vs sung track, 9 windows: {" ".join(f"{-x:+.2f}" for x in lags)} ms late (target 0)')

    plucks = json.load(open(f'{d}/plucks.json'))
    errs = []
    for m, xs in plucks.items():
        x = np.array(xs, dtype=np.float64)
        f0, t = pw.harvest(x, SR, f0_floor=60.0, f0_ceil=600.0, frame_period=5.0)
        f0 = pw.stonemask(x, f0, t, SR)
        v = f0[(t > 0.1) & (t < 0.6) & (f0 > 0)]
        c = 1200 * np.log2(np.median(v) / (440 * 2 ** ((int(m) - 69) / 12)))
        errs.append((int(m), c))
    worst = max(abs(c) for _, c in errs)
    print(f'TUNING  {len(errs)} guitar strings: worst {worst:.1f} cents  ' + ' '.join(f'{m}:{c:+.1f}' for m, c in errs))

    dr = load(f'{d}/drums.wav')          # kick drum alone
    # the first kick (bar 2) comes out of silence: time it absolutely, then use it as a template for the rest
    t1 = 1 * BAR
    w = np.abs(dr[int((t1 - 0.05) * SR):int((t1 + 0.05) * SR)])
    first = int((t1 - 0.05) * SR) + int(np.argmax(w > 1e-3 * w.max()))
    e0 = first / SR - t1
    tpl = dr[first:first + int(0.06 * SR)]
    kicks = [e0 * 1000]
    for bar in range(2, 23):
        for e in ([0, 2, 4, 6] if bar >= 18 else [0, 3, 4]):
            t0 = bar * BAR + e * E8
            a = int((t0 - 0.02) * SR)
            c = correlate(dr[a:a + int(0.1 * SR)], tpl, mode='valid')
            kicks.append((a + int(np.argmax(c))) / SR * 1000 - t0 * 1000)
    kicks = np.array(kicks)
    print(f'DRUMS   {len(kicks)} kicks vs the 160 BPM grid: first {e0 * 1000:+.2f} ms, all: mean {kicks.mean():+.2f} ms, worst {np.abs(kicks).max():.2f} ms')

    band = load(f'{d}/band.wav')
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    sung = np.zeros(len(page), bool)
    fr = int(0.05 * SR)
    for i in range(0, len(page) - fr, fr):
        sung[i:i + fr] = np.sqrt(np.mean(page[i:i + fr] ** 2)) > 0.02
    print(f'BALANCE vocal {rms(page[sung]):.1f} dBFS vs band {rms(band[sung]):.1f} dBFS while singing '
          f'(vocal {rms(page[sung]) - rms(band[sung]):+.1f} dB over the band)')
    full = load(f'{d}/mix.wav')
    print(f'MIX     peak {20 * np.log10(np.abs(full).max()):.1f} dBFS, loudness (RMS) {rms(full[int(3 * SR):int(38 * SR)]):.1f} dBFS')


if __name__ == '__main__':
    main()
