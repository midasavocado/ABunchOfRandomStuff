# STILL BEGINNING — rendering the film (on the Mac)

Everything is in this folder: scene scripts (`scenes/`), shared libraries (`lib/`), the finished score
(`audio/score_master.wav`), grading/titles (`post.py`) and the verified master assembly (`make_master.py`).

## 0. Once
- Blender 5.2 with the MPFB extension + MakeHuman system assets (already installed on this Mac; see
  `docs/PEOPLE.md`). `blender` on PATH, or `export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender`.
- `python3` with `numpy opencv-python pillow` and `ffmpeg`/`ffprobe` (Homebrew).
- `git lfs pull` (assets: parts, gloves, robot links, screen images, suit caches).

## 1. Watch the whole film small first (recommended, about 1-2 h)
```
python3 render_all.py --preview
open STILL_BEGINNING.mp4
```
Every shot at 960x540, graded and cut to the score, scaled to 4K for the verification pass. This is the cheapest
way to judge timing, continuity and the cut before committing the GPU for a day.

## 2. The final
```
rm -f edit/*_prev.mov STILL_BEGINNING.mp4
python3 render_all.py
```
Renders each shot at 3840x2160 (16-bit PNG) -> grades it (`post.py`: bloom, halation, vignette, the s28 statement)
-> `edit/<sid>.mov`, deletes the PNGs, then assembles and **verifies** `STILL_BEGINNING.mp4`:
H.264 High yuv420p Rec.709, 24 fps CFR, exactly 2880 decoded frames, AAC 48 kHz 320 kbps, decoded audio
5,760,000 samples (120.000 s), -12 LUFS +/-1, true peak <= -1 dBTP, faststart. It exits non-zero if any check fails.

It is resumable: a finished shot (`edit/<sid>.mov`) is skipped; delete it to re-render that shot.
Single shots: `python3 render_all.py --shots s09a,s17`. Re-assemble only: `python3 render_all.py --master-only`.

## Engines and rough timings (M5 Pro GPU)
| engine | shots | ~per 4K frame |
|---|---|---|
| Cycles (glass/water, eyes, gold mirror) | s01, s09a, s09b, s16a, s25 | 60-120 s |
| EEVEE (everything else, incl. the analytic Earth) | all others | 5-20 s |
Budget roughly a day for the whole film at 4K; the preview pass is 1-2 hours.

## The edit: handles + blended cuts (this pass)
Every shot renders 8 extra frames past each cut (`lib/timeline.py` HANDLE; none before the first frame or after the
last), and the camera operator (`lib/cinema.py`) rebuilds each camera so it is always moving at a cut, keeps moving
through the handles, and floats like a real operator (steadicam / handheld / drone / macro / weightless, per shot).
`make_master.py` then blends every cut over 4-16 frames (quintic S-curve, highlights lead) so cuts barely read as
transitions. Old `edit/*.mov` files without handles are rejected -- a full re-render is needed once.
`SB_CLEARCHECK=1` scans a shot for the camera inside / grazing geometry; `SB_NOCINEMA=1` shows the raw authored move.

## The final act (this pass)
s22 orbit build -> s23e Earth to Moon (one flight across 384,000 km) -> s23 the lunar settlement with a spaceport
(ships landing and lifting off) -> s23m Moon to Mars -> s25 the telescope -> s24c the Mars colony (drone flight:
hub, habitats, greenhouses, industry, spaceport; a ship lands in a ring of dust) -> s24 the greenhouse hand ->
s25z the pull-back from the colony to the whole planet (140 m to 22,000 km, one move) -> s28 sunrise over the limb
of Mars, the title. New libraries: `lib/ship.py` (colony ship), `lib/mars.py` (Mars at every scale, limb
atmosphere), `lib/colony.py` (the colony on the global timeline), `lib/luna.py` (the Moon as a globe).

## Earlier passes (cloud session)
- New scenes: s08 (workshop pin press + finger test), s09 (family table: prosthetic grasp + reveal, first drop),
  s15 (golden-hour city street crane), s16a/b/c (water, greenhouse, maker-library), s17 (hero one: the child
  drawing at night, ends on the amber cuff), s21 (staging at the edge of space), s22 (orbital truss assembly),
  s25 (space telescope wing latching), s26 (hero three: orbital sunrise -> gold visor), s27 (the eye at dawn).
- s19c hold-down clamp rebuilt as real fabricated hardware (`lib/holddown.py`), used by every pad view.
- Film-wide: worn/chipped paint with cavity grime (`sb.painted`), lathe shading fix (glass refraction),
  gaze/expression/walk helpers for MPFB people (`lib/folks.py`), the s28 statement typography (Inter, OFL).
- Detail/background pass: s07a real SLM chamber (powder streaks, LED strip catching the fused metal), s07b full
  manufacturing hall (`lib/factory.py`: portal frame, clerestory daylight, machines, racking, controller), s10b plant
  room + earlier focus pull, s12a nacelle/spinner/tower detail, s14 farmland patchwork (tramlines, margins, hedges)
  and a visible city, s16b NFT hydroponic channels, s19b/s20 far spaceport + cirrus dawn sky, s23 south-pole massifs
  and lit habitats, s24 Mars exterior (mesas, dunes, solar field, rover) + hand reach fix, s06b framing, s17 city
  visible in EEVEE.
- Score: first drop trimmed with a swell so the ignition drop lands ~2 LU bigger (rebuilt automatically).
- Scene builds are much faster (`sb.prim` builds meshes directly; pixel-identical to the old operator path).
- Lookdev switches (never set them for finals): `SB_NOVOL=1`, `SB_SAMPLES`, `SB_TAA`, `SB_PCT`,
  `SB_EARTH_PROXY=1` (Cycles-renderable Earth stand-in; the real Earth is EEVEE-only).
