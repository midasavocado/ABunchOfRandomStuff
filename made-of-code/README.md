# Made of Code

A 40-second vertical video (1080×1920, 30 fps) where **every glyph on screen is one character of the program that draws it.** That's 27,670 characters: a camera flies through them, they form shapes that point back at their own source, and they settle back into readable code. The ending dives into a phone showing the first frame, so the video loops forever.

- `made-of-code.mp4` is the video, ready to post (H.264 + AAC).
- `poster.jpg` is the first frame, for the thumbnail.
- `index.html` is the whole thing. Open it and tap to play. The page reads its own `<script>` text, and those characters become the particles. The soundtrack is synthesized by the same script with the Web Audio API.
- `tools/render.js` renders the page to MP4 in headless Chromium. `tools/check.py` checks the result.

## What happens

| Time | Picture | What it proves |
|---|---|---|
| 0:00 | The source, readable, under the caption "this video is made of its own source code" | The premise, on the thumbnail |
| 0:04 | First drop. The code comes loose and the camera flies through a tunnel of it | Scale and motion |
| 0:08 | A sphere. "See the bright glyphs? That is the function drawing this sphere." The characters of `SPHERE()` light up inside the sphere | The self-reference, shown rather than claimed |
| 0:12 | The code in reading order, flowing round a Möbius strip. "A loop. Because this video is one." | |
| 0:16 | THE MUSIC? / ALSO CODE. | |
| 0:18 | Second drop. The camera moves over the instrument code, and each line lights up as it plays: `kick` on every kick, `clap` on every clap, the melody on `lead` | The music is code too, visibly |
| 0:22 | The whole song as a 3D landscape. Each ridge is a line of code shaped by the song's spectrum, and the white ridge is now | |
| 0:26 | A ring visualizer | |
| 0:28 | 0 IMAGES / 0 AUDIO FILES / 0 VIDEO MODELS, then 1 FILE / 27,670 GLYPHS (counted from the file as it runs) | |
| 0:32 | Back to readable code. "That's all of me. Pause. Read me." | Pause bait |
| 0:35 | The code shrinks into a phone showing frame 0. "OK. Again." The camera dives in and it becomes frame 0 | A seamless loop |

The heads-up display names the shape function drawing each moment (`→ SPHERE(p, t)`), so you can find it in the source.

## How it works

- **One particle per character.** The script reads its own text, colors it like an editor (syntax colors in the code views, one color per numbered section in the shapes), and gives every non-space character a home position in the code layout.
- **Shapes are functions** `(p, t) → [x, y, z, size, alpha, color, spin]`. A scene list says when each shape takes over. Each glyph flies to its new position on its own delay, curving toward the camera as it goes.
- **A 3D camera** with dolly, pan, tilt, roll and handheld drift, blended between scenes. Depth of field uses pre-blurred sprites at three levels, and each frame gets a bloom pass and a vignette.
- **The music** is a 20-bar song in A minor (Am–F–C–G, 120 BPM): kick, clap, hats, bass, arpeggio, pad and a five-voice supersaw lead, plus reverb, echo, sidechain ducking and risers. The arrangement is a small table of strings, one letter per bar. It renders in four parts at once (one per CPU core), and whatever rings past the end is folded onto the start, so the audio loops cleanly.
- **The picture listens to the sound.** Every instrument writes down when it plays. The code-lights scene and the kick pulses read those times directly, and the landscape and ring read an FFT of the rendered audio.
- **Deterministic.** Randomness is seeded and the audio is rendered offline, so every render is identical. Change the script and the video changes to match, including the glyph count it shows.

## Why it's built this way

Opus 5.5 launched on 22 September 2026, and within days people had posted more than 1,100 videos made with it ([index](https://github.com/athemeroy/awesome-opus-5-5-videos)). The ones that spread lead with "no assets, just code". Motion-graphics showreels and launch videos are crowded, and "one prompt" claims are getting pushback. So this video:

- takes the flex literally (it isn't made *with* code, it's made *of* code), and every beat proves the premise instead of just stating it;
- is checkable: open the page, count the characters, read the source on a paused frame;
- puts the hook on frame 0, places a drop every few seconds, and loops seamlessly so rewatches add up.

## Checks

Run `python3 tools/check.py` after rendering. Results for the committed video:

| Check | Result |
|---|---|
| Glyph count shown in the video vs non-space characters in the script | 27,670 = 27,670 |
| Loop, picture: change from the last frame to the first | 14.4, against 14.7 for the step before it, so the zoom carries straight through |
| Loop, audio: jump across the loop point | 0.0014 (the impact on frame 0 is intended) |
| Loudness per 2-second bar (RMS) | intro 0.20/0.12, drops 0.31–0.34, breakdown 0.08, outro 0.12, build 0.22/0.28 |
| Peak | −1.2 dBFS |
| Kick and bass onsets vs the eighth-note grid | median +7 ms (mostly detector lag), 95% within 54 ms |
| Sprite atlas use, sampled every 0.25 s across the video | 3,477 of 3,840 |
| Music synthesis time in the page | 6.7 s on this 4-core container (was 25 s before the parallel render) |
| Live page | draws frame 0 immediately, accepts a tap while the music is being built, no console errors |

## Suggested post

> Everyone's posting videos Opus 5.5 made with code.
>
> This one is made *of* code. Every character you see is a character of the program drawing it, even the music. The bright glyphs in the sphere are the code drawing the sphere.
>
> Pause at the end. 🔁

Put the link to the live page in a reply.

## Rebuilding

```sh
npm i playwright                   # uses the preinstalled Chromium
pip install imageio-ffmpeg numpy   # or any ffmpeg on PATH
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") \
  node tools/render.js made-of-code.mp4          # about 10 minutes on 4 slow cores
node tools/render.js out.mp4 --stills 0,9.5,19   # PNG stills of chosen times instead
python3 tools/check.py
```
