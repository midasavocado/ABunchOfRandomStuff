# MacBook Showdown

A short, funny fight-night video that settles one question: should your buddy get the **13″ MacBook Pro (M1, Touch Bar)** or pay **$125 more** for the **14″ MacBook Pro (M1 Pro, 10-core CPU, 16-core GPU)**? Both have 16 GB of RAM and a 512 GB SSD.

**The verdict: get the 14″.** Final score: 4 to 2½.

- `macbook-showdown.mp4` is the video: 1920×1080, 60 fps, about 2:14, with voiceover, music and sound effects.
- `index.html` is the animation itself. Open it in a browser and click to play it live with the soundtrack.
- `script.json` has the narration, and `tools/` rebuilds everything (see below).

## The rounds

| Round | 13″ M1 | 14″ M1 Pro | Point |
|---|---|---|---|
| Brains | 8-core CPU (4P + 4E), 8-core GPU, ~68 GB/s memory bandwidth | 10-core CPU (8P + 2E), 16-core GPU, 200 GB/s | 14″ |
| Screen | 13.3″ LCD, 60 Hz, 500 nits | 14.2″ mini-LED XDR, up to 120 Hz, 1,600 nits peak (HDR) | 14″ |
| Ports | 2 × USB-C (Thunderbolt), headphone jack | 3 × Thunderbolt 4, HDMI, SD card slot, MagSafe 3, headphone jack | 14″ |
| Video calls | 720p camera, 2 speakers | 1080p camera, 6 speakers | 14″ |
| Weight | 3.0 lb (1.4 kg) | 3.5 lb (1.6 kg) | 13″ |
| Battery | up to 20 hours* | up to 17 hours* | 13″ |
| Touch Bar | yes (it scrolls emoji) | function keys | 13″ gets ½ |

\*Apple's "up to" figures for movie playback in the Apple TV app. The M1 Pro also drives two external displays (the M1 drives one).

The $125 buys nine upgrades, about $13.89 each. That's less than one Apple USB-C multiport adapter ($69).

The 13″ still wins on weight and battery. If every ounce and every hour matters more than everything else in the table, it's a fine pick.

## How it's made

Everything is generated from code. There are no stock clips, samples or recordings.

1. **Voice** (`tools/make_vo.py`). The narrator (`af_heart`) and the ring announcer (`am_fenrir`) are Kokoro, a neural text-to-speech model. The script finds the pauses in each line so the animation can hit individual words. Scene starts snap to the beat of the music. Every line was checked by transcribing it back with Whisper (`tools/check_vo.py`).
2. **Animation** (`index.html`). One paused GSAP timeline holds every move. `renderAt(t)` draws any moment exactly. Particles, camera shake and scrolling are computed from `t` alone, so any frame renders the same way every time. The page also lists every sound effect it wants and when.
3. **Sound** (`tools/make_audio.py`). The music is a synthesized 120 BPM track in F major. It stops for the record scratch and the crickets, and swells for the winner. About 40 sound effects are synthesized from scratch: the boxing bell, cash register, rocket, sad trombone and more. Music ducks about 10 LU under the voice. The master is normalized to −14 LUFS with a −1 dBFS ceiling.
4. **Frames** (`tools/render.js`). Headless Chromium steps through the timeline one frame at a time. Four workers pipe lossless PNGs into x264.

## Rebuilding

```sh
pip install kokoro-onnx soundfile scipy numpy pyloudnorm imageio-ffmpeg
npm install playwright
# Kokoro model files from the kokoro-onnx GitHub release "model-files-v1.0":
#   kokoro-v1.0.onnx, voices-v1.0.bin
tools/build.sh MODELS_DIR BUILD_DIR 60 4
```

Assets: [GSAP](https://gsap.com) (standard license), [Twemoji](https://github.com/jdecked/twemoji) graphics (CC-BY 4.0), and the fonts Anton, Inter, Permanent Marker and JetBrains Mono (SIL Open Font License).
