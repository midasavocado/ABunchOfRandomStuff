# MacBook Showdown: Dylan's $125 question

A funny fight-night video that settles one question for Dylan: get the **13″ MacBook Pro (M1, Touch Bar)**, or pay **$125 more** for the **14″ MacBook Pro (M1 Pro, 10-core CPU, 16-core GPU)**? Both have 16 GB of RAM and a 512 GB SSD.

**The verdict: get the 14″.** Final score: 4 to 2.

- `macbook-showdown.mp4` is the video: 1920×1080, 60 fps, 2:19, with voiceover, music and sound effects.
- `index.html` is the animation itself. Serve the folder (for example `npx serve .`) and click to play it live with the soundtrack.
- `script.json` is the narration. `src/stage.js` builds the 3D laptops. `tools/` rebuilds everything.

## The rounds (every number is from Apple's tech specs)

| Round | 13″ M1 | 14″ M1 Pro | Point |
|---|---|---|---|
| Brains | 8-core CPU (4 performance + 4 efficiency), 8-core GPU, ~68 GB/s memory bandwidth | 10-core CPU (8 + 2), 16-core GPU, 200 GB/s | 14″ |
| Screen | 13.3″ LCD, 60 Hz, 500 nits, thicker bezels | 14.2″ mini-LED Liquid Retina XDR, up to 120 Hz ProMotion, 1,600 nits peak (HDR) | 14″ |
| Ports | 2 × USB-C (Thunderbolt / USB 4), headphone jack | 3 × Thunderbolt 4, HDMI, SDXC slot, MagSafe 3, headphone jack | 14″ |
| Video calls | 720p camera, stereo speakers | 1080p camera, six-speaker system | 14″ |
| Weight | 3.0 lb (1.4 kg) | 3.5 lb (1.6 kg) | 13″ |
| Battery | up to 20 hours* | up to 17 hours* | 13″ |
| Touch Bar | yes (Apple discontinued it in 2023) | full-height function keys | 0 |

\*Apple's "up to" figures for movie playback in the Apple TV app. The M1 Pro also drives two external displays; the M1 drives one. The 13″ uses the Touch Bar-era design Apple introduced in 2016.

The $125 buys nine upgrades, about $13.89 each. That's less than one Apple USB-C Digital AV Multiport Adapter ($69), which the 13″ would need anyway.

## How it's made

Everything is generated from code: voice, music, sound effects and animation.

1. **Voice** (`tools/make_vo.py`). The narrator (`af_heart`) and the ring announcer (`am_fenrir`) are Kokoro, a neural text-to-speech model. The pauses in each line are measured so the animation can hit individual words. Scene starts snap to the music's beat. Every line was checked by transcribing it back with Whisper (`tools/check_vo.py`).
2. **3D laptops** (`src/stage.js`, bundled to `assets/js/stage.js`). Both laptops are three.js models built to Apple's dimensions. Each has its own bezels, notch or camera, keyboard with key legends, Touch Bar or function row, speaker grilles, trackpad and every port on its correct side. The screens and the Touch Bar are live canvases. Materials are matcaps (studio lighting baked into a texture), and the canvas is transparent. Only the laptops are 3D, which keeps each frame cheap without a GPU.
3. **Everything else is 2D** (`index.html`). One paused GSAP timeline drives the camera, the laptops, the 2D backgrounds and glows, the typography, and callout lines pinned to 3D features. Microsoft's Fluent 3D emoji (MIT) supply the faces and props. `renderAt(t)` draws any moment exactly. The page also lists every sound effect and when it plays.
4. **Sound** (`tools/make_audio.py`). The music is a 120 BPM funk and sports track in F major, sequenced from sampled instruments: brass section, slap bass, Rhodes, muted guitar, strings, timpani, orchestra hits and an acoustic drum kit. It stops for the record scratch and the crickets, and swells for the winner. About 50 sound effects are synthesized or sample-based. The music sits about 11 LU under the voice. The master is at −14 LUFS with a −1 dBFS ceiling.
5. **Frames** (`tools/render.js`). Headless Chromium steps through the timeline and pipes lossless frames into x264 across four workers.

## Rebuilding

```sh
pip install kokoro-onnx soundfile scipy numpy pyloudnorm imageio-ffmpeg
tools/build.sh MODELS_DIR SAMPLES_DIR DRUMS_DIR BUILD_DIR 60 4
```

- `MODELS_DIR`: `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the kokoro-onnx release "model-files-v1.0".
- `SAMPLES_DIR`: `MusyngKite/` from [gleitz/midi-js-soundfonts](https://github.com/gleitz/midi-js-soundfonts).
- `DRUMS_DIR`: `drum-samples/` from [Tonejs/audio](https://github.com/Tonejs/audio) (from web-audio-samples, Apache 2.0).

Credits: [three.js](https://threejs.org) (MIT), [GSAP](https://gsap.com), [Fluent Emoji](https://github.com/microsoft/fluentui-emoji) (MIT), MusyngKite soundfont (CC BY-SA 3.0), and the fonts Anton, Inter, Permanent Marker and JetBrains Mono (SIL OFL).
