"""Voice every line of script.json with Kokoro, trim the silence, and lay the lines out in time.

Usage: python3 make_vo.py --models DIR --out BUILD_DIR
Writes BUILD_DIR/vo/<id>.wav (24 kHz mono) and cues.json; also writes ../cues.js for the page.
Scenes start on the beat of the music (script.json "bpm"), so cuts land on the groove.
"""
import argparse, json, math, os
import numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def trim(x, sr, thresh_db=-42, pad=0.03):
    env = np.convolve(np.abs(x), np.ones(int(sr * 0.01)) / int(sr * 0.01), mode="same")
    on = np.where(env > 10 ** (thresh_db / 20) * np.max(env))[0]
    if not len(on):
        return x
    a = max(0, on[0] - int(pad * sr)); b = min(len(x), on[-1] + int(pad * sr))
    y = x[a:b].copy()
    f = int(0.008 * sr)
    y[:f] *= np.linspace(0, 1, f); y[-f:] *= np.linspace(1, 0, f)
    return y


def speech_segments(x, sr, min_gap=0.12, thresh_db=-34):
    """Start/end (seconds) of each stretch of speech, split at pauses of min_gap or more."""
    hop = int(sr * 0.005)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop * 4) / (hop * 4), mode="same"))[::hop]
    on = rms > 10 ** (thresh_db / 20) * rms.max()
    out, start, quiet = [], None, 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * hop / sr >= min_gap:
                out.append([round(start * hop / sr, 3), round((i - quiet) * hop / sr, 3)]); start = None
    if start is not None:
        out.append([round(start * hop / sr, 3), round(len(on) * hop / sr, 3)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    script = json.load(open(os.path.join(ROOT, "script.json")))
    os.makedirs(os.path.join(a.out, "vo"), exist_ok=True)
    tts = Kokoro(os.path.join(a.models, "kokoro-v1.0.onnx"), os.path.join(a.models, "voices-v1.0.bin"))
    durs, segs = {}, {}
    for sc in script["scenes"]:
        for ln in sc["lines"]:
            path = os.path.join(a.out, "vo", ln["id"] + ".wav")
            if a.only is None or ln["id"] in a.only or not os.path.exists(path):
                v = script["voices"][ln["v"]]
                x, sr = tts.create(ln["text"], voice=v["voice"], speed=ln.get("speed", v["speed"]), lang="en-us")
                x = trim(np.asarray(x, dtype=np.float32), sr)
                sf.write(path, x, sr)
            x, sr = sf.read(path)
            durs[ln["id"]] = len(x) / sr
            segs[ln["id"]] = speech_segments(x, sr)

    beat = 60.0 / script["bpm"]
    snap = lambda t: math.ceil(t / beat - 1e-6) * beat
    t, scenes, lines = 0.0, {}, {}
    for sc in script["scenes"]:
        start = snap(t)
        lt = start + sc.get("lead", 0)
        for i, ln in enumerate(sc["lines"]):
            if i:
                lt += ln.get("gap", 0.25)
            lines[ln["id"]] = {"start": round(lt, 4), "dur": round(durs[ln["id"]], 4), "v": ln["v"], "segs": segs[ln["id"]], "scene": sc["id"], "text": ln["text"]}
            lt += durs[ln["id"]]
        end = lt + sc.get("tail", 0) if sc["lines"] else start + sc.get("hold", 2)
        scenes[sc["id"]] = {"start": round(start, 4), "end": round(end, 4)}
        t = end
    total = snap(t)
    cues = {"bpm": script["bpm"], "total": total, "scenes": scenes, "lines": lines}
    json.dump(cues, open(os.path.join(a.out, "cues.json"), "w"), indent=1)
    open(os.path.join(ROOT, "cues.js"), "w").write("window.CUES = " + json.dumps(cues) + ";\n")
    for k, s in scenes.items():
        print(f"{k:10s} {s['start']:7.2f} → {s['end']:7.2f}")
    print("total", total)


if __name__ == "__main__":
    main()
