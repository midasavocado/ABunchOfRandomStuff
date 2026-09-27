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

f = frames([0, total - 2, total - 1])
step, seam = np.abs(f[2] - f[1]).mean(), np.abs(f[0] - f[2]).mean()
print(f'loop, video: last->first frame changes by {seam:.2f}; the frame before that changed by {step:.2f}')
print(f'loop, audio: jump {abs(audio[0] - audio[-1]):.4f}; '
      f'rms last 20 ms {np.sqrt((audio[-960:] ** 2).mean()):.3f}, first 20 ms {np.sqrt((audio[:960] ** 2).mean()):.3f}')
print(f'peak {20 * np.log10(np.abs(audio).max()):.1f} dBFS')

bars = [np.sqrt((audio[int(b * 2 * SR):int((b + 1) * 2 * SR)] ** 2).mean()) for b in range(n // 2)]
print('rms per bar:', ' '.join(f'{v:.2f}' for v in bars))

# low-band onsets (kick and sub bass, 30-100 Hz) against the grid of eighth notes
spec = np.fft.rfft(audio)
hz = np.fft.rfftfreq(len(audio), 1 / SR)
low = np.fft.irfft(spec * ((hz > 30) & (hz < 100)), len(audio))
env = np.convolve(np.abs(low), np.ones(240) / 240, 'same')      # 5 ms envelope
thresh, onsets, i = 0.3 * env.max(), [], 0
while i < len(env):
    if env[i] > thresh:
        onsets.append(i / SR)
        i += int(0.2 * SR)                                         # one per hit
    else:
        i += 1
err = np.array([(t - round(t / (BEAT / 2)) * BEAT / 2) * 1000 for t in onsets])
print(f'low-band onsets: {len(err)}, offset from the eighth-note grid: '
      f'median {np.median(err):+.1f} ms, 95% within {np.percentile(np.abs(err), 95):.1f} ms')
