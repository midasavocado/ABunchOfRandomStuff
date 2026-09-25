#!/usr/bin/env python3
"""Measure where Kokoro's sung vowels land for each voice, and store the offset in latency.json.

Sings every lyric line with the voice at zero offset, then measures each vowel that follows a voiceless
consonant or silence with two independent instruments (test_vocals.vowel_onsets: low-frequency energy
rise; test_vocals.voicing_onsets: first clearly voiced frame). The stored offset is the mean of the two
instruments' means.

    python3 calibrate_latency.py --models DIR --lines lines.json VOICE[:TRANSPOSE] ...
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sing  # noqa: E402
import test_vocals as tv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--lines', default='lines.json')
    ap.add_argument('voices', nargs='+')
    args = ap.parse_args()
    lines = json.load(open(args.lines))
    singer = sing.Singer(args.models)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latency.json')
    table = json.load(open(path)) if os.path.exists(path) else {}
    for spec in args.voices:
        voice, _, tr = spec.partition(':')
        sing.LATENCY[voice] = 0.0
        rep = []
        y = sing.sing_track(singer, voice, lines, int(tr or -12), 0.1, report=rep)
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            sf.write(f.name, sing.to44(y).astype(np.float32), 44100)
            a = np.array([o for o, _ in tv.vowel_onsets(f.name, rep)])
            b = np.array([o for o, _ in tv.voicing_onsets(f.name, rep)])
        table[voice] = round(float((a.mean() + b.mean()) / 2), 4)
        print(f'{voice}: vowel energy {a.mean() * 1000:+.1f} ms (n={len(a)}), voicing {b.mean() * 1000:+.1f} ms (n={len(b)}) '
              f'-> {table[voice] * 1000:+.1f} ms')
        json.dump(table, open(path, 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
