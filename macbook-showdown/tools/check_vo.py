"""Transcribe every voiceover line with Whisper (sherpa-onnx) to catch mispronunciations.
Usage: python3 check_vo.py --whisper DIR --build BUILD_DIR"""
import argparse, glob, json, os
import numpy as np, soundfile as sf, sherpa_onnx

ap = argparse.ArgumentParser(); ap.add_argument("--whisper", required=True); ap.add_argument("--build", required=True)
a = ap.parse_args()
w = lambda n: glob.glob(os.path.join(a.whisper, "*" + n))[0]
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=w("encoder.int8.onnx"), decoder=w("decoder.int8.onnx"), tokens=w("tokens.txt"), num_threads=4)
cues = json.load(open(os.path.join(a.build, "cues.json")))
for k, ln in cues["lines"].items():
    x, sr = sf.read(os.path.join(a.build, "vo", k + ".wav"), dtype="float32")
    s = rec.create_stream(); s.accept_waveform(sr, np.concatenate([x, np.zeros(sr // 2, np.float32)])); rec.decode_stream(s)
    print(f"{k:5s} {s.result.text.strip()}")
