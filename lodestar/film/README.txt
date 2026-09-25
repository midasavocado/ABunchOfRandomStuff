LODESTAR — the film
===================

A 3½-minute short film, scored and rendered entirely from code. Every frame is ray-traced
on your GPU from GLSL shaders (a planet with a real atmosphere, a ringed gas giant, a
volumetric storm with lightning, a pocket-watch movement, a small ship called the Wren),
then graded like film and cut to the soundtrack (lodestar.mp3).

HOW TO RENDER (Mac)
-------------------
1. Double-click   "Quick test (low-res preview).command"   first (a few minutes) to make sure
   everything works. It opens LODESTAR_preview.mp4 when done.
2. Then double-click   "Render LODESTAR (1080p).command"   for the real thing.
   (Or "Render LODESTAR (4K).command" if you have time — it's slower.)

The first run downloads a private copy of Python and a few libraries (~100 MB) into
~/.local/share/uv — nothing else on your Mac is touched.

The Terminal window shows a progress bar and a time estimate. On an Apple Silicon Mac
expect roughly 30–120 minutes for 1080p depending on the chip. You can quit any time:
finished shots are kept in render_cache/ and the next run picks up where it stopped.

IF MACOS BLOCKS THE FILE
------------------------
Files downloaded from the internet that aren't from the App Store get quarantined:
  • macOS 15 (Sequoia) and later: double-click once, click "Done", then open
    System Settings → Privacy & Security, scroll down and click "Open Anyway".
  • Older macOS: right-click (Control-click) the .command file → Open → Open.
  • Or, in Terminal, un-quarantine the whole folder once:
        xattr -dr com.apple.quarantine /path/to/LODESTAR

OTHER COMPUTERS
---------------
Any machine with Python 3.10+ and an OpenGL 3.3 GPU works:
    pip install numpy moderngl pillow imageio-ffmpeg
    python3 render.py --preset 1080p        (or 4k / 720p / preview)
Single frames:  python3 render.py --still 95.0 130.0   → stills/

WHAT'S IN HERE
--------------
  render.py       the render engine (GPU passes, motion blur, depth of field, bloom, grading, encoding)
  shots.py        the 21-shot timeline, camera moves and music sync (every clock tick, drum hit, heartbeat)
  shaders/        the worlds: watch, planet, rings, storm, void, post-processing
  fonts/          Cinzel & Cormorant Garamond (SIL Open Font License) for the title card
  lodestar.mp3    the soundtrack
  STORY.md        the story and shot list
