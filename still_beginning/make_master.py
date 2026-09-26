"""Final film master: edit/<sid>.mov (all shots, timeline order) + audio/score_master.wav -> STILL_BEGINNING.mp4

  3840x2160, 24 fps CFR, exactly 2880 frames, H.264 High yuv420p Rec.709 SDR, faststart,
  stereo AAC 48 kHz 320 kbps, trimmed to exactly 120.000 s. Fine film grain is added here (and only here).

usage: python3 make_master.py [--preview]   (--preview uses edit/<sid>_prev.mov, 960x540 -> scaled, for timing checks)
Verifies the decoded result (frame count, decoded audio length, loudness, true peak) and exits non-zero on failure.
"""
import os, sys, json, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import timeline as TL

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "STILL_BEGINNING.mp4")
AUDIO = os.path.join(ROOT, "audio", "score_master.wav")
PREVIEW = "--preview" in sys.argv


def run(cmd, **kw):
    print("+", " ".join(cmd[:6]), "...")
    return subprocess.run(cmd, check=True, **kw)


def probe(path, *args):
    return subprocess.run(["ffprobe", "-v", "error", *args, path], capture_output=True, text=True).stdout.strip()


class ShotReader:
    """Sequential decoder of one graded shot (edit/<sid>.mov incl. handles) addressed by GLOBAL frame number."""

    def __init__(self, sid, suffix):
        _, a, b, _ = TL.shot(sid)
        hin, hout = TL.handles(sid)
        self.first = a - hin
        self.last = b + hout - 1
        self.path = os.path.join(ROOT, "edit", f"{sid}{suffix}.mov")
        w, h = probe(self.path, "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0").split(",")[:2]
        self.w, self.h = int(w), int(h)
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.pos = self.first
        self.cur = None

    def get(self, g):
        import numpy as np
        assert self.first <= g <= self.last, (self.path, g)
        n = self.w * self.h * 3
        while self.pos <= g:
            buf = self.p.stdout.read(n)
            if len(buf) < n:
                raise RuntimeError(f"{self.path}: short read at frame {self.pos}")
            self.cur = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self.pos += 1
        return self.cur

    def close(self):
        try:
            self.p.stdout.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def blend_weight(x):
    """0..1 across the transition window: a soft S-curve (the cut should barely register)."""
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (x * 6 - 15) + 10)


def main():
    import numpy as np
    TL.check()
    suffix = "_prev" if PREVIEW else ""
    for sid, a, b, _ in TL.SHOTS:
        p = os.path.join(ROOT, "edit", f"{sid}{suffix}.mov")
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        hin, hout = TL.handles(sid)
        n = probe(p, "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0")
        if n != str(b - a + hin + hout):
            sys.exit(f"{sid}: {n} frames, expected {b - a + hin + hout} (shot + handles); re-render it")
    ids = [s[0] for s in TL.SHOTS]
    starts = {s[0]: s[1] for s in TL.SHOTS}
    readers = {}

    def reader(sid):
        if sid not in readers:
            readers[sid] = ShotReader(sid, suffix)
        return readers[sid]

    r0 = reader(ids[0])
    W, H = r0.w, r0.h
    vf = ["scale=3840:2160:flags=lanczos" if PREVIEW else "null",
          # fine luma grain (temporal), breaks up banding in dark gradients before 8-bit quantisation
          "noise=c0s=5:c0f=t+u:c1s=0:c2s=0",
          "format=yuv420p",
          # tag the frames themselves: newer ffmpeg takes colour metadata from the filter graph, not the output flags
          "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-i", AUDIO,
           "-map", "0:v:0", "-map", "1:a:0",
           "-vf", ",".join(vf), "-fps_mode", "cfr", "-r", "24", "-frames:v", "2880",
           "-c:v", "libx264", "-profile:v", "high", "-preset", os.environ.get("SB_MASTER_PRESET", "slow"), "-tune", "film",
           "-crf", "12", "-maxrate", "120M", "-bufsize", "240M",
           "-x264-params", "aq-mode=3:aq-strength=0.9:deblock=-1,-1:colorprim=bt709:transfer=bt709:colormatrix=bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
           "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
           "-t", "120", "-movflags", "+faststart", OUT + ".part.mp4"]
    print("+ conform (soft motion-continuous blends at every cut) -> ffmpeg ...")
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    k = 0                                            # index of the shot that owns frame g
    for g in range(TL.TOTAL):
        while k + 1 < len(ids) and g >= starts[ids[k + 1]]:
            k += 1
            if k >= 2 and ids[k - 2] in readers:          # two shots back: no longer needed
                readers.pop(ids[k - 2]).close()
        sid = ids[k]
        x = reader(sid).get(g).astype(np.float32)
        # blending with the next shot (its incoming window straddles our end)
        if k + 1 < len(ids):
            nxt = ids[k + 1]; c = starts[nxt]; L = TL.trans(nxt)
            if L and g >= c - L // 2:
                w = blend_weight((g - (c - L // 2) + 0.5) / L)
                y = reader(nxt).get(g).astype(np.float32)
                x = _mix(x, y, w)
        # blending with the previous shot (we are the incoming one)
        if k > 0:
            c = starts[sid]; L = TL.trans(sid)
            if L and g < c + (L - L // 2):
                w = blend_weight((g - (c - L // 2) + 0.5) / L)
                y = reader(ids[k - 1]).get(g).astype(np.float32)
                x = _mix(y, x, w)
        enc.stdin.write(np.clip(x + 0.5, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("encode failed")
    for r in readers.values():
        r.close()
    os.replace(OUT + ".part.mp4", OUT)       # never leave a half-written master under the delivery name
    verify()


def _mix(a, b, w):
    """dissolve a -> b with weight w; the incoming shot's highlights lead slightly (a lab-dissolve feel, no dip)."""
    import numpy as np
    lb = b.max(axis=2, keepdims=True) / 255.0
    wl = np.clip(w + 0.15 * (lb - 0.5) * 4 * w * (1 - w), 0, 1)
    return a * (1 - wl) + b * wl


def verify():
    ok = True
    v = json.loads(probe(OUT, "-count_frames", "-show_streams", "-show_format", "-of", "json"))
    vs = [s for s in v["streams"] if s["codec_type"] == "video"][0]
    as_ = [s for s in v["streams"] if s["codec_type"] == "audio"][0]
    checks = {
        "video codec h264": vs["codec_name"] == "h264",
        "profile High": vs.get("profile") == "High",
        "3840x2160": (vs["width"], vs["height"]) == (3840, 2160),
        "yuv420p": vs["pix_fmt"] == "yuv420p",
        "bt709": vs.get("color_primaries") == "bt709" and vs.get("color_transfer") == "bt709",
        "24 fps": vs["r_frame_rate"] == "24/1" and vs["avg_frame_rate"] == "24/1",
        "2880 decoded frames": vs.get("nb_read_frames") == "2880",
        "aac 48k stereo": as_["codec_name"] == "aac" and as_["sample_rate"] == "48000" and as_["channels"] == 2,
    }
    # decoded audio length (after priming/padding removal by the edit list)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT, "-map", "0:a", "-f", "s16le", "-ac", "2", "-ar", "48000", "-"],
                         capture_output=True).stdout
    ns = len(pcm) // 4
    checks[f"decoded audio {ns} samples ~ 5760000"] = abs(ns - 5760000) <= 1024
    # decoded video duration from last packet pts
    last = probe(OUT, "-select_streams", "v:0", "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0").splitlines()
    end = max(float(a) + float(b) for a, b in (l.split(",")[:2] for l in last if l))
    checks[f"video ends at {end:.3f}s"] = abs(end - 120.0) < 1e-3
    # faststart: moov before mdat
    with open(OUT, "rb") as f:
        head = f.read(1 << 20)
    checks["faststart (moov first)"] = head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))
    # loudness + true peak of the delivered AAC
    lo = subprocess.run(["ffmpeg", "-hide_banner", "-i", OUT, "-map", "0:a", "-af", "ebur128=peak=true", "-f", "null", "-"],
                        capture_output=True, text=True).stderr
    I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", lo)[-1])
    TP = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", lo)[-1])
    checks[f"loudness {I} LUFS (-12 +/-1)"] = abs(I + 12) <= 1.0
    checks[f"true peak {TP} dBTP (<= -1)"] = TP <= -1.0
    br = int(v["format"]["bit_rate"]) / 1e6
    print(f"overall bitrate {br:.1f} Mb/s, audio {int(as_.get('bit_rate', 0)) / 1000:.0f} kb/s")
    for k, val in checks.items():
        print(("PASS " if val else "FAIL ") + k)
        ok &= val
    if not ok:
        sys.exit("MASTER FAILED CHECKS")
    print("MASTER OK:", OUT)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        main()
