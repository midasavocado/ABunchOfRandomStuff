"""Analysis helpers: spectrogram PNGs + level stats (used during composition QC)."""
import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
import sbdsp as D


def spectrogram(x, path, title="", t0=0.0, fmax=16000, fmin=20, nfft=4096, hop=512, vmin=-90, height=6,
                width=16, marks=()):
    if x.ndim == 2:
        mono = x.mean(1)
    else:
        mono = x
    f, t, S = signal.stft(mono, D.SR, nperseg=nfft, noverlap=nfft - hop, window="hann")
    S = 20 * np.log10(np.abs(S) + 1e-9)
    S -= S.max()
    fig, axs = plt.subplots(2, 1, figsize=(width, height), gridspec_kw={"height_ratios": [4, 1]}, sharex=True)
    ax = axs[0]
    m = (f >= fmin) & (f <= fmax)
    ax.pcolormesh(t + t0, f[m], S[m], vmin=vmin, vmax=0, cmap="magma", shading="auto")
    ax.set_yscale("log")
    ax.set_ylim(fmin, fmax)
    ax.set_yticks([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_yticklabels(["30", "60", "120", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax.set_title(title)
    for mk in marks:
        ax.axvline(mk, color="cyan", lw=0.6, alpha=0.6)
    ax2 = axs[1]
    blk = 480
    n = len(mono) // blk
    if x.ndim == 2:
        pk = np.abs(x[: n * blk]).max(1).reshape(n, blk).max(1)
    else:
        pk = np.abs(mono[: n * blk]).reshape(n, blk).max(1)
    rms = np.sqrt((mono[: n * blk] ** 2).reshape(n, blk).mean(1))
    tt = np.arange(n) * blk / D.SR + t0
    ax2.plot(tt, D.db(pk), lw=0.5, label="peak")
    ax2.plot(tt, D.db(rms), lw=0.5, label="rms")
    ax2.set_ylim(-70, 3)
    ax2.grid(alpha=0.3)
    for mk in marks:
        ax2.axvline(mk, color="c", lw=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def stats(x, name=""):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    pk = D.db(np.abs(x).max())
    rms = D.db(np.sqrt(np.mean(x ** 2)))
    tp = D.true_peak(x)
    try:
        lu = D.lufs_integrated(x)
    except Exception:
        lu = float("nan")
    M = x.mean(1); S = (x[:, 0] - x[:, 1]) / 2
    corr = np.sum(x[:, 0] * x[:, 1]) / (np.sqrt(np.sum(x[:, 0] ** 2) * np.sum(x[:, 1] ** 2)) + 1e-12)
    print(f"{name:20s} peak {pk:6.1f} dBFS  TP {tp:6.1f}  rms {rms:6.1f}  LUFS {lu:6.1f}  crest {pk - rms:5.1f}  corr {corr:5.2f}")


def band_energy(x, name=""):
    mono = x.mean(1) if x.ndim == 2 else x
    f, P = signal.welch(mono, D.SR, nperseg=8192)
    bands = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
             (8000, 16000), (16000, 24000)]
    tot = P.sum()
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(P[m].sum() / tot + 1e-12))
    print(f"{name:14s} " + " ".join(f"{lo}-{hi}:{v:5.1f}" for (lo, hi), v in zip(bands, out)))
    return out


if __name__ == "__main__":
    p = sys.argv[1]
    x, sr = sf.read(p, dtype="float32")
    t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    t1 = float(sys.argv[4]) if len(sys.argv) > 4 else len(x) / sr
    seg = x[int(t0 * sr): int(t1 * sr)]
    spectrogram(seg, sys.argv[2], title=p, t0=t0)
    stats(seg, p.split("/")[-1])
