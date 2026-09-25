# How the vocal is made, and how it's tested

## Singing with a speech model

`sing.py` makes Kokoro, a neural text-to-speech model, sing. Kokoro is StyleTTS2-style: internally it predicts how long each phoneme lasts, a pitch curve, and an energy curve, and then it renders audio from those. `make_singing_model()` rewires the ONNX graph so all three can be supplied from outside. Each phrase then goes through five steps:

1. **Syllables.** The words become phonemes, and the phonemes are split into the lyric's syllables around their vowels. Onset clusters (pr, bl, st…) and affricates (ch, j) stay together, and reduced vowels are opened up the way singers do it ("a" → "uh").
2. **Durations.** Each vowel starts on its note. A consonant before a vowel keeps its natural length and sits just ahead of the beat. The vowel fills the rest of the note.
3. **Pitch and energy.** The melody is fed in as the pitch curve. Vowels hold their volume across long notes instead of fading the way speech does.
4. **Tuning.** Kokoro follows the pitch curve only to within tens of cents, so each voiced stretch gets TD-PSOLA pitch correction onto the exact melody. The model's own waveform is regrained at the target pitch, which keeps the voice's detail.
5. **Timing.** Each voice's small output offset was measured on this song (`calibrate_latency.py`, stored in `latency.json`) and is removed.

The lead is `am_fenrir`, sung an octave below the written melody. The last chorus and the title shout layer six voices (a G-major chord on "SPARE PARTS!").

## Tests

| Script | What it checks |
|---|---|
| `test_vocals.py` | Pitch of every sung syllable in cents; vowel onsets against the beat, measured with two independent instruments; Whisper transcript against the lyrics |
| `render_page.js` | Renders the page's own audio in headless Chromium: the full mix plus vocal, band, and kick-only stems, and single guitar plucks |
| `test_mix.py` | Vocal sync inside the page, guitar tuning, kick drum timing against 160 BPM, and vocal-to-band balance |

Results for the committed version:

| Check | Result |
|---|---|
| Sung pitch, 133 syllables | median error 2.4 cents; 95% within 30 cents |
| Note changes vs beat | mean +0.1 ms, median error 7.5 ms |
| Vowel onsets vs beat | mean −4.8 ms, median error 12.9 ms |
| Vocal sync in the page | 0.02 ms (under one sample) |
| Guitar tuning, 16 strings | worst 0.1 cents |
| Kick drum vs grid, 69 hits | worst 1.6 ms |
| MP4 audio vs rendered mix | 0.0 ms |
| Vocal over band while singing | +3.3 dB |
| Whisper (small.en), vocal alone | 30% word error rate |
| Whisper (small.en), full mix | 37% word error rate |

Whisper is a strict judge of sung words. Plain speech from the same voice scores 12%, and many of the "errors" above are near-misses like "sorts of" for "swords of".

Two things were tried and dropped because the measurements didn't support them. **Per-syllable timing correction** improved the onset instrument it optimized against but not the independent voicing instrument, which means it was fitting measurement noise. **WORLD re-synthesis for tuning** was precise but cost intelligibility, so PSOLA replaced it.

## Rebuilding

```sh
pip install kokoro-onnx onnx pyworld soundfile scipy numpy librosa sherpa-onnx
# model files (from the kokoro-onnx GitHub release "model-files-v1.0"): kokoro-v1.0.onnx, voices-v1.0.bin
node export_lines.js lines.json
python3 sing.py --models MODELS --lines lines.json --out vocals.wav --stems stems
python3 test_vocals.py --lead stems/lead_dry.wav --report stems/lead_report.json --lines lines.json --asr WHISPER_DIR
```

`render_page.js` and `test_mix.py` check the result inside the page. The page embeds the vocal as MP3, with a 10 ms sync chirp at 0.1 s so it lines up exactly in every browser.
