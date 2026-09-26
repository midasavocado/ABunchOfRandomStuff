# Made of Code

A 32-second vertical video (1080×1920, 30 fps) where **every glyph on screen is one character of the program that draws it.** The code on screen breaks apart into a spinning sphere, a spiral galaxy, words, and a music visualizer that follows its own soundtrack, then settles back into readable code. The last frame matches the first, so it loops seamlessly.

- `made-of-code.mp4` is the video, ready to post (H.264 + AAC, 38 MB).
- `poster.jpg` is the first frame, for use as a thumbnail.
- `index.html` is the whole thing. Open it in a browser and tap to play. The page reads its own `<script>` text, and those characters become the 16,507 particles. The music is synthesized by the same script with the Web Audio API.
- `tools/render.js` renders the page to the MP4 in headless Chromium.

## The trend it answers

Opus 5.5 launched on 22 September 2026, and the videos people made with it took over X within a day: launch videos, motion-graphics showreels, one-prompt music videos, Three.js games. A community index ([awesome-opus-5-5-videos](https://github.com/athemeroy/awesome-opus-5-5-videos)) had catalogued more than 1,100 of them by the 26th. Three patterns stood out:

1. **"Everything is code" is the flex.** The posts that spread were the ones that said *no assets, no video model, just JavaScript*. The claim is the hook.
2. **The market is saturated.** Motion graphics (350 videos) and 3D (324) dominate. Another showreel or another launch video is noise.
3. **Pushback on "one-shot."** Critics point out that many "one prompt" videos took several passes, which makes an unverifiable claim a liability.

This video takes the flex literally: the video isn't just *made with* code, it is *made of* code. It's also checkable. The glyph count shown on screen is computed from the file at render time, anyone can open `index.html` and see the same thing, and pausing on the code frames shows the actual source.

## Why it's built the way it is

| Choice | Reason |
|---|---|
| Hook caption on frame 0 | The premise lands before anyone scrolls, and the first frame doubles as the thumbnail |
| Readable source on the first and last frames | Pause bait: comments in the code talk to people who stop to read it ("If you paused the video to read this: hi.") |
| Drops at 0:04 and 0:16 | A payoff every few seconds. The intro sits about 10 dB below the drops so the first one hits |
| Particles spell the words | "THE MUSIC? / ALSO CODE." and "0 IMAGES / 0 AUDIO FILES / 0 VIDEO MODELS" make the claims with the thing being claimed |
| Seamless loop | The last frame and the audio tail wrap onto the start, so loops don't pop. Loops count as rewatches |
| 9:16, text kept away from the edges | Works on X, Reels, Shorts and TikTok without cropping under the platform UI |

## Checks

| Check | Result |
|---|---|
| On-screen glyph count vs non-space characters in the script (counted separately in Python) | 16,507 = 16,507 |
| Kick onsets in the MP4 audio vs the beat grid the visuals pulse to | within ~10–20 ms (the error is mostly from the envelope measurement) |
| Loop seam, last sample to first sample | 0.0009 jump (inaudible) |
| Loudness, intro vs drop (RMS) | 0.064 vs 0.201, about 10 dB |
| Peak | −1.1 dBFS |
| Live page in Chromium | no errors, plays from a tap, and shows a frame immediately while the music is built |

## Suggested post

> Everyone's posting videos Opus 5.5 made with code.
>
> This one is made *of* code. Every character you see is a character of the program drawing it, including the music.
>
> Pause on the first frame. It's the real source. 🔁

Put a link to `index.html` in a reply rather than in the post itself, since links in the post tend to reduce reach.

## Rebuilding

```sh
npm i playwright            # Chromium comes from the preinstalled/bundled browser
pip install imageio-ffmpeg  # or any ffmpeg on PATH
FFMPEG=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") \
  node tools/render.js made-of-code.mp4
node tools/render.js out.mp4 --stills 0,7,18   # PNG stills of chosen times instead
```

Every render is identical: randomness is seeded and the audio is rendered offline. Edit the script and the video changes to match, including the glyph count it prints.
