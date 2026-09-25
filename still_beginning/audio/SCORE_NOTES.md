# STILL BEGINNING: Score Notes

Original score, fully synthesized (no samples, loops, or recordings). 128 BPM, 4/4, 64 bars, D major / B minor.
**Length: exactly 5,760,000 samples at 48 kHz = 120.000 s.** 1 bar = 1.875 s = 45 frames = 90,000 samples.
1 beat = 11.25 frames = 22,500 samples. Bar *n* starts at (n-1) x 1.875 s, which is frame (n-1) x 45.

Rebuild: `cd audio && python3 compose.py` (about 1-2 min, deterministic with fixed seeds). `SB_METER=1` also prints per-track
loudness per section.
Code: `compose.py` (arrangement/mix), `score.py` (theme, counter-melody, harmony), `instruments.py` (synths),
`sbdsp.py` + `dsp.c`/`libdsp.dylib` (DSP), `master.py` (mastering and delivery), `analyze.py` (QC plots).

## Deliverables
| file | contents |
|---|---|
| `score_master.wav` | 48 kHz / 24-bit stereo, mastered: glue comp, EQ, 4x-oversampled look-ahead TP limiter, -12 LUFS |
| `score_premaster.wav` | sum of the stems, no bus processing, peak -6.0 dBFS |
| `stems/drums.wav` | kick, clap, snare, rolls and fills, hats, shaker, ride, crashes, toms (+ own reverb) |
| `stems/bass.wav` | mono sine sub (below 140 Hz), saturated mid-bass (100 Hz-3 kHz harmonics), intro pulse, build bass |
| `stems/music.wav` | felt piano, supersaw chords, 16th pluck arp, syncopated counter-arp "hook" |
| `stems/leads.wav` | supersaw lead (theme), final-drop counter-melody, filtered build lead, soft doubling |
| `stems/pads.wav` | warm analog pad, string-ensemble pad, formant choir pad |
| `stems/fx.wav` | lens/ratchet clicks, clock ticks, servo ticks, metal clicks, breaths, risers, reverse swells, impacts, ignition |
| `cues.json` | emphasized downbeats, drop hits, pre-drop silences, final chord, key transients (seconds, frames, samples) |
| `levels.json` | measurement dump from the last render |

The stems sum to the premaster (verified error below -120 dB). Each stem carries its own reverb and delay returns.
Content under 120 Hz is mono in every stem.

## Measured (final render)
| measure | value |
|---|---|
| Integrated loudness (ffmpeg `ebur128`) | **-12.0 LUFS** (own BS.1770 meter: -11.98) |
| True peak (ffmpeg `ebur128=peak=true`) | **-1.3 dBTP** (own 4x and 8x oversampled: -1.27 dBTP; sample peak -1.28 dBFS) |
| LRA | 8.5 LU |
| First drop, 30-60 s (integrated) | -11.72 LUFS |
| Final drop, 75-105 s (integrated) | -10.15 LUFS, so the final drop is **+1.57 LU** over the first |
| Intro 0-15 / build 15-28 / breakdown 60-73 / resolution 105-120 | -19.4 / -13.6 / -15.4 / -11.1 LUFS |
| Bus glue GR (max) / limiter GR (max) | 3.1 dB / 3.9 dB (typically 1-2 dB in the drops) |
| Mono fold-down loss | 0.3-1.0 dB depending on section; L/R correlation under 120 Hz is 0.99+ |

## The theme (eighth-note units; the chord is in brackets)
**Statement A.**
[Bm] F#5(3) E5(1) F#5(2) A5(2) | [G] B5(3) A5(1) F#5(2) E5(2) | [D] F#5(3) E5(1) D5(2) A4(2) | [A] C#5(2) D5(2) E5(4)

**Answer B.**
[Bm] F#5(3) E5(1) F#5(2) A5(2) | [G] B5(3) D6(1) C#6(2) A5(2) | [D] A5(3) F#5(1) E5(2) D5(2) | [A] E5(6) rest(2)

**Breakdown variant of A, bar 4.** [Asus4 to A] D5(2) C#5(2) E5(4). The D is a suspension that resolves to C# as the chord resolves.

**Final-drop counter-melody** (warm three-saw "horn"; moves in contrary motion and fills the theme's gaps):
A: D5(3) C#5(1) B4(2) A4(2) | G4(3) A4(1) B4(4) | A4(3) B4(1) A4(2) F#4(2) | A4(4) C#5(4)
B: D5(3) C#5(1) B4(2) D5(2) | G4(3) A4(1) B4(2) D5(2) | F#4(3) G4(1) A4(2) F#4(2) | A4(2) B4(2) C#5(2) E5(2). The last four notes answer the theme's rest.

**Resolution (augmented theme).** [Gmaj7] F#5(6) E5(2) | [D/F#] F#5(4) A5(4) | [Em7] B5(6) A5(2) | [Asus4 to A] E5(8)
**Cadence, bar 61.** [G(add9)] D6(3) B5(1) | [A] C#6(4), then **bar 62: D6** over D major (piano an octave lower).

**Theme journey.** Felt-piano seed in bars 2-3. Full theme on restrained piano in bars 5-8, then the answer phrase in bars 9-12.
A filtered supersaw hints at it in the build (13-16). It is then stated on a big supersaw stack across four varied phrases (17-32).
The breakdown exposes it on intimate piano (33-40). The final drop is the strongest version, with a counter-melody, richer
harmony, octave layers, and choir (41-56). The resolution plays it in long notes (57-60), and it resolves to D (61-62).

## Arrangement map
| bars | time | frames | section and content |
|---|---|---|---|
| 1 | 0.000 | 0 | Lens/ratchet click at **0.050 s** (frame 1.2), soft inhale 0.12-1.4 s, muted 8th-note bass pulse fades in from beat 2 |
| 2-4 | 1.875 | 45 | Felt-piano **seed**: bar 2 = A1, bar 3 = A2 over a low piano B/G; bar 4 Asus4 to A (D4 to C#4 inner voice). Clock ticks from bar 3; warm pad enters |
| 5-8 | 7.5 | 180 | Full statement A on piano with quarter-note broken-chord LH; tick-tock 8ths; warm pad (Bm G D A) |
| 9-12 | 15.0 | 360 | **ACCELERATION.** Filtered kick (LP about 110-200 Hz), 16th pluck arp opening (LP 700 Hz up to 6.5 kHz), filtered off-beat bass, servo ticks (tuned to chord tones), metal clicks from bar 11, 16th hats from bar 11, piano answer phrase B |
| 13-16 | 22.5 | 540 | Build: claps on 2 and 4 (crescendo), off-beat sub, string pad swelling, filtered supersaw lead plays A (LP 350 Hz to 3.2 kHz), noise riser, snare roll (bar 15 8ths, bar 16 16ths to 32nds), open hats. **Kick out for bar 16**; rising HP "drain" empties the low end over bars 15-16 |
| 16 b4 | **29.531-30.000** | 708.75-720 | **Pull-out**: near-silence (about -21 LUFS momentary), reversed crash/reverb swell and inhale ending exactly at the drop |
| 17-20 | **30.000** | **720** | **FIRST DROP.** Impact (sub boom + saturated blast + crash + downlifter). Theme A on supersaw lead; supersaw chords, rolling 16th off-kick bass + ducked sub, 4-on-floor, claps, off-beat hats |
| 21-24 | 37.5 | 900 | Answer B; shaker 16ths; piano sparkle doubling an octave up; 16th hat ghosts; tom fill at the end of bar 24 |
| 25-28 | 45.0 | 1080 | A again with new colour: **syncopated 3+3+2 counter-arp** (pluck + KS harp), open hats, metallic clicks on the "e"s |
| 29-32 | 52.5 | 1260 | Answer B with **octave-lift layer**, ride, brighter chords; bar 32 snare/tom fill + riser |
| 33-36 | **60.0** | **1440** | **HUMAN BREATH.** Crash + soft boom tail into the breakdown; heavy drums gone. Exposed felt piano: statement A (breakdown variant) over **G, D/F#, Em7, Asus4 to A**; flowing 8th-note LH; string pad (octave up); faint clock ticks |
| 37-39 | 67.5 | 1620 | Answer B over **G, D/F#, Bm7**; soft-lead doubling, "oo" choir; **controlled build**: filtered 8th bass pulse opening, filtered kick and 8th sub from bar 38, snare roll 38-39, riser, low end drained in bar 39. The phrase cadences on D5 |
| 40 | **73.125-75.000** | **1755-1800** | **Held breath**: kick, bass and music out; small mechanical click at **73.30** (frame 1759.2), inhale 73.78 to 74.93, faint reverse swell |
| 41-44 | **75.000** | **1800** | **FINAL DROP / ROCKET IGNITION** at sample 3,600,000: saturated blast + low roar with rumble + crackle + sub boom + crash. Theme A, wider 7-voice stack with octave layer; **counter-melody**; harmony **Bm(add9), Gmaj7, D/F#, A-Asus4-A**; bigger kick, clap+snare layered, 16th hats, open hats, rolling bass with octave jumps + reese layer |
| 45-48 | 82.5 | 1980 | Answer B over **Bm7, G(add9), D, Asus4 to A**; counter B; counter-arp hook |
| 49-52 | **90.0** | **2160** | Statement A, stronger octave layer, **"ah" choir pad**, ride, piano sparkle; crash + boom |
| 53-56 | 97.5 | 2340 | Peak phrase: answer B over **Bm7, Gmaj9, D/F#, Asus4 to A**; tom groove; bar 56 tom/snare fill + riser |
| 57-60 | **105.0** | **2520** | **RESOLUTION.** Crash + deep boom; half-time kick (57-58), a single kick in 59, then none; timpani-like toms on 59/60. Huge sustained **Gmaj7, D/F#, Em7, Asus4 to A** (supersaw, strings, choir, piano); theme in long notes |
| 61 | 112.5 | 2700 | **FINAL STATEMENT.** Cadence G(add9) to A: D6 B5 then C#6 |
| 62 | **114.375** | **2745** | **Final D major arrival**: rolled piano D1 to D5, D6 lead, strings, choir, soft supersaw bed, gentle sub + boom; reverse swell into it |
| 62-64 | 114.4-120.0 | 2745-2880 | Natural decay (about -30 dB by 118 s); closing lens click at **118.2 s** (bookend); raised-cosine fade 119.0 to 120.0; the file ends at exact zero |

## Key hit times
| event | seconds | frame (24 fps) | sample |
|---|---|---|---|
| Opening click | 0.050 | 1.2 | 2,400 |
| Acceleration starts (filtered kick) | 15.000 | 360 | 720,000 |
| Pre-drop pull-out 1 | 29.531-30.000 | 708.75-720 | 1,417,500-1,440,000 |
| **First drop** | **30.000** | **720** | **1,440,000** |
| Breakdown crash/impact | 60.000 | 1440 | 2,880,000 |
| Pre-ignition click | 73.300 | 1759.2 | 3,518,400 |
| Pre-drop pull-out 2 (held breath) | 73.125-75.000 | 1755-1800 | 3,510,000-3,600,000 |
| **Final drop / ignition** | **75.000** | **1800** | **3,600,000** |
| Final-drop lift (bar 49) | 90.000 | 2160 | 4,320,000 |
| Resolution (bar 57) | 105.000 | 2520 | 5,040,000 |
| Cadence (bar 61) | 112.500 | 2700 | 5,400,000 |
| **Final D major arrival** | **114.375** | **2745** | **5,490,000** |
| Closing click | 118.200 | 2836.8 | 5,673,600 |
| End | 120.000 | 2880 | 5,760,000 |

## Sound design notes
- **Felt piano**: additive synthesis with stretched inharmonic partials; 1-3 strings per note detuned about ±1 cent and struck
  in phase (natural beating and two-stage decay); frequency-dependent damping; strike-position comb; felt hammer thump
  scaled by register; damper; synthetic soundboard-IR convolution; register-balanced; humanized timing and velocity in the exposed sections.
- **Supersaw lead**: 7 band-limited (polyBLEP) saws with JP-style detune and slow drift, a focused pulse layer for pitch
  definition, an optional octave layer, accent-driven lowpass, delayed vibrato, legato glide, 4x-oversampled saturation, chorus.
- **Bass**: a pure mono sine sub (sidechained about 80% to the kick) plus a separate mid-bass (saw + square through a
  saturating 4-pole ladder) high-passed at 95 Hz, so the harmonics carry the line on small speakers.
- **Choir**: pulse sources with vibrato and jitter through a parallel formant bank ("oo", "ah", "oh"), plus breath noise and ensemble.
- **Spaces**: per-stem 8-line FDN plate and hall, plus a dotted-8th ping-pong delay feeding the hall; returns are
  lightly sidechained and gated in the pull-outs.
- **Master**: RMS-detector glue at 2:1, EQ (HP 22 Hz, -0.8 dB at 280 Hz, +0.5 dB at 3.2 kHz, +1.2 dB shelf at 10 kHz),
  4x-oversampled look-ahead limiter (1.5 ms look-ahead, 110 ms release), gain solved iteratively to -12.0 LUFS.

## Known weaknesses (honest)
- I composed and mixed without listening. Every decision was checked by spectrograms, loudness per section and track,
  chroma-versus-harmony checks, mono fold-down, true-peak and discontinuity scans. Timbre judgement (how "real" the felt piano
  and choir sound, how pleasant the supersaw is) is therefore reasoned, not heard. A human listening pass is recommended,
  especially on the choir pad and the piano in the exposed breakdown.
- The drop impacts land at or slightly above the steady-state momentary loudness (+0 to +1.6 LU for the first 200 ms), not far
  above it. Their weight comes from the preceding near-silence, broadband transient, sub bloom and crash. Making them
  louder cost more than 6 dB of limiter gain reduction, so I backed off. The director's ignition SFX can add more on top.
- The low end is generous (sub + kick dominate the energy under 60 Hz). It is controlled and mono, but on large systems the
  director may want 1-2 dB less sub in the bass stem.
- The resolution (bars 57-60) is still fairly dense (about 1 LU under the final drop), so the "opening up" is mostly in texture
  (half-time, sustained chords) rather than level.
- The clock/servo ticks are deliberately sharp transients; they show up as the largest discontinuities in the file and are intended.
