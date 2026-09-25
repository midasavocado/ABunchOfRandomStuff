# LODESTAR

An original instrumental theme, composed and synthesized entirely from scratch in Python.
No samples, soundfonts, or loops. Every string section, brass choir, pipe organ, piano, choir,
taiko, and clock tick is built from raw oscillators, noise, and filters (`lodestar.py`).

**Listen:** [`lodestar.mp3`](lodestar.mp3) · 3:31 · D minor → D major · 76 BPM

## The story

A voyager leaves home, fights through a storm, is nearly lost in the dark, finds the one
fixed star to steer by, and makes it back.

| Time | Movement | What you hear |
|------|----------|---------------|
| 0:00 | **I. Home** | A clock ticks. A soft pipe organ. A piano plays a rippling figure, then the first half of the theme. |
| 0:25 | **II. Departure** | A lone French horn plays the full theme. It ends on an unresolved note, a question. |
| 0:51 | **III. Ascent** | Strings start running in 16th notes, taiko drums wake up, and the theme climbs an octave. A snare roll launches into… |
| 1:16 | **IV. The Storm** | Dark and driving. Low-brass "braams", taiko in 3+3+2, and grinding violins. The clock speeds up to double, then quadruple time. The strings scream upward. |
| 1:54 | **V. Lost** | An impact, then almost nothing. Distant stars, a heartbeat, the clock slowed to half speed. The piano tries to remember the theme and can't finish it. Then something rises in the dark. |
| 2:06 | **VI. Lodestar** | The theme returns in **major**, played by full brass over choir, drums, and running strings. The question from the horn solo finally gets its answer (♭VI–♭VII–I). At 2:32 it repeats even bigger, with the organ and soaring violins. |
| 2:57 | **VII. Homecoming** | The piano from the opening plays the theme again, now at peace. The clock ticks once more, then stops. A final chord rings out. |

## Rebuild it

```bash
pip install numpy scipy lameenc
python3 lodestar.py     # renders lodestar.wav + lodestar.mp3 (~2.5 min)
```
