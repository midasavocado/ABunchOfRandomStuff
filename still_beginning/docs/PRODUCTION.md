# STILL BEGINNING — Production Bible (read fully before working)

Project root: `/Users/midas/Desktop/still_beginning/`. We are making a finished, original **120.000 s, 3840x2160, 24 fps (2880 frames)**
cinematic music film about an optimistic human future. Everything is procedural: Blender 5.2 (Python, run headless),
Python/numpy/scipy/cv2/skimage (system `python3`), ffmpeg. No image/video generation models, no downloads.
The client's bar is **maximum quality — “really, really good.”** They explicitly asked us to work **piece by piece**:
lookdev each component (materials, lighting, geometry, motion) at useful resolution against a mental photographic
reference, fix what looks CG/toy-like, and only then move on. Never accept “it renders” as “it’s good.”

## Emotional thesis & rules (from the brief — obey)
- Curiosity → intelligence → making → human benefit → abundance → exploration → renewed curiosity.
- Exhilarating, tactile, intelligent, human. NOT a corporate ad, product demo, or futuristic screensaver.
- No apocalypse, no sinister AI, no national/corporate triumphalism, no logos, no readable brand names, no flags,
  no editorial text/title cards (in-world UI text only if minimal and meaningful), no dates/statistics.
- Speculative tech is imagined future, not documentary.

## Visual language
- Premium, physically based, cohesive. Deliberate stylization OK; low-detail toy rendering NOT OK.
- Palette arc: opens graphite / silver / restrained blue-hour → warm amber grows as we move to human benefit →
  Earth: real greens, ocean blues, natural sunlight → space is NOT a neon nightclub (black, hard sunlight, restrained).
- **Recurring motif: a curved amber highlight** arising naturally (eye reflection of the afterglow horizon, an
  anodized amber rim, machined edge, reactor arc, horizon). Never a drawn glowing ribbon. Amber anodized accents:
  `props.amber_anodized()` (lib/props.py).
- Intentional focal lengths. Macro = intimate/optical (tiny DOF). Human scenes observed, not advertised.
  Engineering scale needs foreground / midground / background, occlusion, recognizable scale cues.
- Camera moves must reveal something; have mass; no endless orbiting, no constant roll, no random speed ramps.
- Brightness from motivated light. Use DOF, motion blur (180° shutter), bloom sparingly (bloom/halation/vignette/grain
  are added in post by `post.py` — do NOT add your own compositor glare).
- Continuity assets: ONE child (brown amber-hazel eyes, warm medium skin tone (0.52,0.34,0.24), dark brown hair,
  graphite hoodie with **amber knit sleeve cuffs**), ONE telescope (lib/props.py `telescope()`), ONE rocket (white +
  graphite with a restrained amber band; consistent proportions/engine layout across shots), ONE astronaut suit design.

## Timeline (lib/timeline.py is the source of truth)
128 BPM 4/4, 1 bar = 45 frames = 1.875 s, 1 beat = 11.25 frames. Shots are defined as (id, start, end_exclusive, script)
using GLOBAL frame numbers. A scene script renders frames [start, end) of its shot id and writes
`preview/<sid>/NNNN.png` (SB_RES=preview, 960x540) / `half/` (1920x1080) / `renders/<sid>/NNNN.png` (SB_RES=final, 4K 16-bit).
Hero shots (uninterrupted, real animation): s17 (1440-1620), s20 launch (1800-1980), s26 orbital reveal (2430-2610).
Rocket ignition lands exactly on frame 1800. First drop at 720. Breakdown 1440-1800. Final statement 2700-2880.

## Code conventions
- Scene script: `scenes/<scene>.py`, run as `blender -b -P scenes/<scene>.py -- <sid>`. Start with the boilerplate:
  ```python
  import sys, os, math
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
  import bpy, sb
  sid = (sb.argv() or ["sXX"])[0]
  sc = sb.reset(); sb.setup_render("EEVEE", ...)   # or "CYCLES"
  ... build ...; animate with GLOBAL frame numbers of the shot(s) ...
  sb.frames(start, end); sb.render_shot(sid)
  ```
  One script may serve several shot ids (branch on `sid` for camera/animation).
- `lib/sb.py` = helpers: reset, setup_render, camera/cam_bake (per-frame baked camera from functions of t),
  bake(obj,f0,f1,fn), materials (mat, brushed_metal, painted, glass, skin, fabric, rubber, concrete, volume_mat,
  emit_mat), NB node-builder class (noise/voronoi/wave/mix/ramp/maprange/bump...), world_sky (Blender physical sky:
  sky_type 'MULTIPLE_SCATTERING'), world_gradient, world_bluehour, light(), sun(), lathe, tube_along, etc. Read it.
- Useful env vars: `SB_RES=preview|half|final`, `SB_FRAMES=first,mid,last` or `SB_FRAMES=0:10` (offsets within shot),
  `SB_BORDER=x0,y0,x1,y1` (crop render for fast 4K detail checks), `SB_ENGINE=EEVEE|CYCLES`.
- Contact sheet: `python3 lib/contact.py preview/<sid> /tmp/cs.png <step> <cols>`.
- Look at your renders with the Read tool (it displays PNG/JPG). Downscale/crop 4K images with PIL first.

## Blender 5.2 gotchas we hit
- Engines: 'BLENDER_EEVEE' (EEVEE Next; ~3-15 s/frame at 4K) and 'CYCLES' (Metal GPU; 60-120 s/frame at 4K).
  Default to EEVEE unless a shot truly needs path tracing (refraction-heavy macro, etc.). Budget matters:
  the whole film must render on one M5 Pro GPU.
- Mix node: clamp_factor defaults True — `NB.mix(clamp=False)` now also unclamps the factor.
- World shaders: TexCoord 'Generated' = view direction. EEVEE world volume available for haze.
- Cycles: glass casts opaque shadows (no caustics) — set `obj.visible_shadow=False` on thin glass/cornea-like shells
  when light must reach what's behind. `obj.visible_glossy=False` on lights to avoid rectangle catchlights.
- Random-walk SSS on open (non-closed) meshes leaks and looks waxy; use `subsurface_method='BURLEY'` there.
- Don't use hard `max()` clamps in procedural geometry (creates ripples); use smooth functions.
- Light linking works in Cycles: `light.light_linking.receiver_collection = coll`.
- Keyframes: `sb.reset()` sets new keyframe interpolation LINEAR; bake per-frame for full control.
- Motion blur: `scene.render.use_motion_blur` (EEVEE+Cycles).

## Quality control (do this for every shot)
1. Lookdev style frame(s) at `SB_RES=half` and 4K crops (`SB_BORDER`) of the most detailed regions.
2. Compare mentally to real photography/cinematography of the subject. List what reads CG and fix it
   (flat lighting, uniform albedo, missing wear/edge highlights, too-clean materials, toy proportions, wrong scale,
   floating contact, intersecting meshes, noisy volumes, flicker, texture swimming).
3. Preview the whole shot at 960x540 (`SB_RES=preview`, all frames) and check motion via contact sheets (every
   ~6 frames) — velocity continuity, no pops, correct contact.
4. Check first/last frames against neighbouring shots for match cuts (see brief/timeline notes).
5. When genuinely good, append the shot id to `queue.txt` (one per line) for the 4K final render queue
   (the queue renders, runs `post.py`, and deletes PNGs). Do not render full 4K sequences yourself outside the queue.
Disk is tight (~19 GB free): never leave large 4K PNG sequences around; preview/half renders are fine but clean up
large temporary files you create.

## Things to avoid (from the brief)
Changing faces/vehicle geometry, extra fingers, broken grip/contact, sliding feet, intersecting meshes, flat planets,
impossible shadows, texture popping, flicker, noisy dark gradients, smoke that looks like a translucent card,
abrupt velocity changes, stutter, glows over everything, RGB split, glitches, white flashes, zoom tunnels.
