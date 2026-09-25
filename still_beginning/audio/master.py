"""STILL BEGINNING score - premaster sum, mastering chain and deliverables.
Chain: bus glue compression (RMS detector, 2:1) -> EQ -> 4x oversampled look-ahead true-peak limiter ->
loudness normalised to -12 LUFS integrated (BS.1770-4), TP <= -1.0 dBTP."""
import os, json, re, subprocess
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
import sbdsp as D
from sbdsp import SR, TOTAL

TARGET_LUFS = -12.0
TP_CEIL = -1.25          # internal target, leaves margin under -1.0 dBTP after 24-bit quantisation
OS = 4


def rms_detector(x, tau_ms=12.0):
    a = np.exp(-1.0 / (tau_ms * 0.001 * SR))
    p = signal.lfilter([1 - a], [1, -a], x.astype(np.float64) ** 2, axis=0)
    return np.sqrt(np.maximum(p, 0)).astype(np.float32) * 1.414


def glue(x):
    det = rms_detector(D.hp(x, 90))
    ddb = D.db(np.max(det, 1))
    loud = ddb[int(75 * SR):int(105 * SR)]
    thr = float(np.percentile(loud, 90) - 2.5)
    g = D.comp_gain(det, thr, 2.0, 15.0, 180.0, 8.0)
    return (x * g[:, None]).astype(np.float32), thr, float(D.db(g.min()))


def master_eq(x):
    y = D.hp(x, 22, 2)
    y = D.eq(y, ("peak", 280, 0.9, -0.8), ("peak", 3200, 0.8, 0.5), ("highshelf", 10000, 0.7, 1.2))
    return y


def limiter(x, ceil_db, look_ms=1.5, rel_ms=110.0):
    up = signal.resample_poly(x, OS, 1, axis=0).astype(np.float32)
    a = np.max(np.abs(up), 1)
    thr = 10 ** (ceil_db / 20)
    greq = np.minimum(1.0, thr / (a + 1e-12)).astype(np.float32)
    L = int(look_ms * 1e-3 * SR * OS) | 1
    # forward-looking min over [n, n+L-1]
    gmin = minimum_filter1d(greq, L, mode="nearest", origin=-(L // 2))
    # instant attack, smooth release
    g2 = D.gainsmooth(gmin, 0.0005, rel_ms, sr=SR * OS)
    g2 = np.minimum(g2, gmin)
    # backward moving average over L -> smooth attack that reaches the min before the peak
    g3 = uniform_filter1d(g2, L, mode="nearest", origin=(L // 2))
    y = up * g3[:, None]
    out = signal.resample_poly(y, 1, OS, axis=0).astype(np.float32)[: len(x)]
    gr = D.db(g3.min())
    return out, float(gr)


def ebur128(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = r.stderr
    summ = txt[txt.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
    LRA = float(re.search(r"LRA:\s+(-?[\d.]+) LU", summ).group(1))
    TP = float(re.search(r"Peak:\s+(-?[\d.inf]+) dBFS", summ).group(1))
    return {"I": I, "LRA": LRA, "TP": TP}


def seg_lufs(x, t0, t1):
    return D.lufs_integrated(x[int(t0 * SR):int(t1 * SR)])


def deliver(stems, cues, here):
    os.makedirs(os.path.join(here, "stems"), exist_ok=True)
    names = list(stems.keys())
    pre = sum(stems[k] for k in names)
    sc = 10 ** (-6.0 / 20) / (np.max(np.abs(pre)) + 1e-12)
    for k in names:
        stems[k] = (stems[k] * sc).astype(np.float32)
    pre = sum(stems[k] for k in names).astype(np.float32)
    assert len(pre) == TOTAL
    for k in names:
        sf.write(os.path.join(here, "stems", f"{k}.wav"), stems[k], SR, subtype="PCM_24")
    sf.write(os.path.join(here, "score_premaster.wav"), pre, SR, subtype="PCM_24")
    print("premaster peak", round(float(D.db(np.abs(pre).max())), 2), "dBFS  LUFS", round(D.lufs_integrated(pre), 2))

    g, thr, grmax = glue(pre)
    e = master_eq(g)
    # loudness search: gain into the limiter
    gain_db = TARGET_LUFS - D.lufs_integrated(e) + 1.0
    ceil = TP_CEIL
    for it in range(8):
        y, gr = limiter(e * 10 ** (gain_db / 20), ceil)
        lu = D.lufs_integrated(y)
        tp = D.true_peak(y, 4)
        print(f"  master it{it}: gain {gain_db:+.2f} dB  LUFS {lu:.2f}  TP {tp:.2f}  maxGR {gr:.1f} dB")
        ok_l = abs(lu - TARGET_LUFS) < 0.05
        ok_t = tp <= TP_CEIL + 0.05
        if ok_l and ok_t:
            break
        if not ok_t:
            ceil -= (tp - TP_CEIL) + 0.02
        gain_db += (TARGET_LUFS - lu) * 1.05
    # final clean end (the chain adds no tail but keep the last samples exactly zero-bound) + safety
    y[-8:] *= np.linspace(1, 0, 8)[:, None]
    mpath = os.path.join(here, "score_master.wav")
    sf.write(mpath, y, SR, subtype="PCM_24")
    y24, _ = sf.read(mpath, dtype="float32")
    meas = {
        "samples": int(len(y24)),
        "own_integrated_lufs": round(D.lufs_integrated(y24), 2),
        "own_true_peak_dbtp_4x": round(D.true_peak(y24, 4), 2),
        "own_true_peak_dbtp_8x": round(D.true_peak(y24, 8), 2),
        "sample_peak_dbfs": round(float(D.db(np.abs(y24).max())), 2),
        "glue_threshold_db": round(thr, 1), "glue_max_gr_db": round(-grmax, 2) if grmax < 0 else round(grmax, 2),
        "limiter_max_gr_db": round(-gr, 2),
        "first_drop_lufs_30_60": round(seg_lufs(y24, 30, 60), 2),
        "final_drop_lufs_75_105": round(seg_lufs(y24, 75, 105), 2),
        "intro_lufs_0_15": round(seg_lufs(y24, 0, 15), 2),
        "build_lufs_15_28": round(seg_lufs(y24, 15, 28.1), 2),
        "breakdown_lufs_60_73": round(seg_lufs(y24, 60, 73.1), 2),
        "resolution_lufs_105_120": round(seg_lufs(y24, 105, 120), 2),
        "premaster_peak_dbfs": round(float(D.db(np.abs(pre).max())), 2),
    }
    meas["drop_contrast_lu"] = round(meas["final_drop_lufs_75_105"] - meas["first_drop_lufs_30_60"], 2)
    try:
        meas["ffmpeg_ebur128"] = ebur128(mpath)
    except Exception as ex:
        meas["ffmpeg_ebur128"] = str(ex)
    for k, v in meas.items():
        print(f"  {k}: {v}")
    with open(os.path.join(here, "levels.json"), "w") as f:
        json.dump(meas, f, indent=2)
    # cues
    phrase_bars = {1: "intro (click+breath, pulse)", 2: "seed motif enters (felt piano)", 5: "full theme, restrained",
                   9: "ACCELERATION - filtered kick, 16th arps, off-beat bass", 13: "build: claps, filtered lead theme",
                   17: "FIRST DROP", 21: "drop 1 phrase 2 (answer)", 25: "drop 1 phrase 3 (counter-arp, open hats)",
                   29: "drop 1 phrase 4 (octave lift, ride)", 33: "HUMAN BREATH - breakdown, exposed piano",
                   37: "breakdown answer + controlled build", 41: "FINAL DROP (ignition)", 45: "final drop answer",
                   49: "final drop - choir, octave-doubled lead", 53: "final drop - peak phrase, toms",
                   57: "RESOLUTION - half-time, sustained chords", 61: "FINAL STATEMENT - cadence",
                   62: "final D major arrival"}
    cues["downbeats"] = [{"t": round((b - 1) * 1.875, 6), "frame": round((b - 1) * 45.0, 3), "sample": (b - 1) * 90000,
                          "bar": b, "label": l}
                         for b, l in sorted(phrase_bars.items())]
    cues["meta"] = {"bpm": 128, "bars": 64, "sr": SR, "samples": TOTAL, "fps": 24,
                    "bar_seconds": 1.875, "bar_frames": 45, "beat_frames": 11.25}
    with open(os.path.join(here, "cues.json"), "w") as f:
        json.dump(cues, f, indent=2)
    return meas
