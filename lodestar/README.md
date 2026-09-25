# LODESTAR

An original instrumental theme and a short film, both made entirely from code.

- **The music** (`lodestar.py`) is composed and synthesized from scratch in Python: no samples,
  soundfonts or loops. Strings, brass, pipe organ, piano, choir, drums, the pocket watch and
  the lead voice are all built from oscillators, noise and filters.
- **The film** (`film/`) is ray-traced on the GPU from GLSL shaders, graded like film and cut to
  the score. See [`film/STORY.md`](film/STORY.md) for the story and shot list, and
  [`film/README.txt`](film/README.txt) for how to render it (double-click on a Mac).

**Listen:** [`lodestar.mp3`](lodestar.mp3) · 3:31 · D minor → D major · 76 BPM

## What makes it LODESTAR

- **The watch is the rhythm section.** The piece opens with someone winding a pocket watch.
  From bar 2 every tick is also a plucked note (tick = the chord's fifth, tock = its root), so
  time itself plays the harmony. It races in the storm, stumbles when the ship is lost, and
  stops when the ship is home.
- **The hook** is a rising call, long–short–LONG (D … A — high D). Its signature colour is the
  raised fourth: an E sung over a B♭ chord in bar 6 of the theme.
- **The Lodestar voice** is a synthesized lead that glides between notes, and every note blooms
  from "oo" to "ah" like a wordless singer. It carries the Departure solo, cuts through the end
  of the storm, and soars over the final chorus.
- **The answer.** In minor, the theme ends on an unresolved C♯ (a question). In major, it lands
  on the ♭VI–♭VII–I cadence, with the melody rising C → D.

## The story

A small ship leaves home carrying an old pocket watch. It fights through a storm, is nearly
lost in the dark, finds the one fixed star to steer by, and makes it back.

| Time | Movement | What you hear |
|------|----------|---------------|
| 0:00 | **I. Home** | The watch is wound, then starts to tick. Its pluck, a soft pipe organ, and a piano playing the first half of the theme. |
| 0:25 | **II. Departure** | The Lodestar voice sings the full theme alone, shadowed by a horn. It ends on the unresolved C♯. |
| 0:51 | **III. Ascent** | Strings run in 16ths, taiko drums wake, and the theme climbs an octave on violins and horns. |
| 1:16 | **IV. The Storm** | Low-brass hits, taiko in 3+3+2, grinding violins, and an endlessly rising Shepard tone. The watch doubles, then quadruples, its speed. The voice screams the last four bars. |
| 1:54 | **V. Lost** | An impact, then space wind, a heartbeat and a slowed clock. The piano tries the hook and can't finish it. |
| 2:06 | **VI. Lodestar** | The theme in **major**: full brass, choir, drums and running strings. At 2:32 it comes around again with the organ, a horn countermelody and the Lodestar voice on top. |
| 2:57 | **VII. Homecoming** | The piano sings the major theme softly. The watch ticks a few more times, then stops. One last high note rings out. |

## Rebuild it

```bash
pip install numpy scipy lameenc
python3 lodestar.py     # renders lodestar.wav + lodestar.mp3 (~3 min)
```
