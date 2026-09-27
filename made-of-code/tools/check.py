"""Checks the rendered video against the program that made it.

    python3 tools/check.py [made-of-code.mp4]

- glyphs: non-space characters in the page's <script id="src">, which is what the video counts
- loop: how different the last frame is from the first, next to how much one normal frame changes;
  and the audio jump across the loop point
- loudness per bar, and kick onsets against the 120 BPM grid
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = 'ffmpeg'

HERE = Path(__file__).resolve().parent.parent
MP4 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'made-of-code.mp4'
W, H, FPS, SR, BEAT = 1080, 1920, 30, 48000, 0.5


def ff(*args):
    return subprocess.run([FFMPEG, '-v', 'error', *args], capture_output=True, check=True).stdout


def frames(indices):
    raw = ff('-i', str(MP4), '-vf', 'select=' + '+'.join(f'eq(n\\,{i})' for i in indices),
             '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'gray', '-')
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(float)


src = re.search(r'<script id="src">(.*?)</script>', (HERE / 'index.html').read_text(), re.S).group(1)
src = src[1:] if src.startswith('\n') else src
glyphs = sum(c not in ' \n' for c in src)
print(f'glyphs in the source: {glyphs:,}')

audio = np.frombuffer(ff('-i', str(MP4), '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'), np.int16) / 32768
n = len(audio) // SR
total = round(len(audio) / SR * FPS)
print(f'length: {len(audio) / SR:.3f} s, {total} frames')

f = frames([0, 1, total - 1])
step, seam = np.abs(f[1] - f[0]).mean(), np.abs(f[0] - f[2]).mean()
print(f'loop, video: last->first frame differs by {seam:.2f} (one normal frame step: {step:.2f})')
print(f'loop, audio: jump {abs(audio[0] - audio[-1]):.4f}; '
      f'rms last 20 ms {np.sqrt((audio[-960:] ** 2).mean()):.3f}, first 20 ms {np.sqrt((audio[:960] ** 2).mean()):.3f}')
print(f'peak {20 * np.log10(np.abs(audio).max()):.1f} dBFS')

bars = [np.sqrt((audio[int(b * 2 * SR):int((b + 1) * 2 * SR)] ** 2).mean()) for b in range(n // 2)]
print('rms per bar:', ' '.join(f'{v:.2f}' for v in bars))

# kick onsets: energy under 150 Hz, rising edges, compared with the nearest beat
lp = np.convolve(audio, np.ones(160) / 160, 'same')            # crude low-pass
env = np.convolve(lp ** 2, np.ones(240) / 240, 'same')
hop = 48
e = env[::hop]
rise = np.diff(e)
peaks = [i for i in range(1, len(rise) - 1) if rise[i] > rise[i - 1] and rise[i] >= rise[i + 1]
         and rise[i] > 0.25 * rise.max()]
onsets = np.array(peaks) * hop / SR
err = [(t - round(t / BEAT) * BEAT) * 1000 for t in onsets]
if err:
    print(f'kick-like onsets: {len(err)}, offset from the beat grid: median {np.median(err):+.1f} ms, '
          f'worst {max(err, key=abs):+.1f} ms')
