"""STILL BEGINNING score - musical data: theme, countermelody, harmony, voicings.
Units: note lengths in eighth notes (8 per bar). Bars are 1-based."""
from sbdsp import nm, BAR, BEAT, b2s

E8 = BEAT // 2  # 11250 samples

# --------------------------------------------------------------------------- theme
# Statement (A) and answer (B), one list per bar: (eighths, note or None)
THEME_A = [
    [(3, "F#5"), (1, "E5"), (2, "F#5"), (2, "A5")],     # Bm
    [(3, "B5"), (1, "A5"), (2, "F#5"), (2, "E5")],      # G
    [(3, "F#5"), (1, "E5"), (2, "D5"), (2, "A4")],      # D
    [(2, "C#5"), (2, "D5"), (4, "E5")],                 # A
]
THEME_B = [
    [(3, "F#5"), (1, "E5"), (2, "F#5"), (2, "A5")],     # Bm
    [(3, "B5"), (1, "D6"), (2, "C#6"), (2, "A5")],      # G
    [(3, "A5"), (1, "F#5"), (2, "E5"), (2, "D5")],      # D
    [(6, "E5"), (2, None)],                             # A
]
# Breakdown version of A: bar 4 becomes a suspension (D over Asus4 resolving to C#)
THEME_A_BRK = THEME_A[:3] + [[(2, "D5"), (2, "C#5"), (4, "E5")]]

# Countermelody for the final drop (tenor/alto register, contrary motion, fills the theme's gaps)
COUNTER_A = [
    [(3, "D5"), (1, "C#5"), (2, "B4"), (2, "A4")],
    [(3, "G4"), (1, "A4"), (4, "B4")],
    [(3, "A4"), (1, "B4"), (2, "A4"), (2, "F#4")],
    [(4, "A4"), (4, "C#5")],
]
COUNTER_B = [
    [(3, "D5"), (1, "C#5"), (2, "B4"), (2, "D5")],
    [(3, "G4"), (1, "A4"), (2, "B4"), (2, "D5")],
    [(3, "F#4"), (1, "G4"), (2, "A4"), (2, "F#4")],
    [(2, "A4"), (2, "B4"), (2, "C#5"), (2, "E5")],
]

# Resolution (bars 57-60): theme in long notes, then final cadence (61) and arrival (62)
THEME_RES = [
    [(6, "F#5"), (2, "E5")],        # Gmaj7
    [(4, "F#5"), (4, "A5")],        # D/F#
    [(6, "B5"), (2, "A5")],         # Em7
    [(8, "E5")],                    # Asus4 -> A
]
CADENCE = [[(3, "D6"), (1, "B5"), (4, "C#6")]]   # bar 61: G(add9) -> A
ARRIVAL = "D6"                                   # bar 62 downbeat

# --------------------------------------------------------------------------- harmony
CH = {
    #          bass(sub)  mid-bass     pad voicing                         arp tones
    "Bm":    ("B1", "B2", ["F#3", "B3", "D4", "F#4"], ["B3", "D4", "F#4", "B4"]),
    "Bm9":   ("B1", "B2", ["F#3", "B3", "D4", "C#5"], ["B3", "D4", "F#4", "C#5"]),
    "Bm7":   ("B1", "B2", ["A3", "B3", "D4", "F#4"], ["B3", "D4", "F#4", "A4"]),
    "G":     ("G1", "G2", ["G3", "B3", "D4", "G4"], ["G3", "B3", "D4", "G4"]),
    "Gmaj7": ("G1", "G2", ["G3", "B3", "D4", "F#4"], ["G3", "B3", "D4", "F#4"]),
    "Gadd9": ("G1", "G2", ["G3", "B3", "D4", "A4"], ["G3", "B3", "D4", "A4"]),
    "Gmaj9": ("G1", "G2", ["B3", "D4", "F#4", "A4"], ["G3", "B3", "F#4", "A4"]),
    "D":     ("D2", "D2", ["F#3", "A3", "D4", "F#4"], ["A3", "D4", "F#4", "A4"]),
    "D/F#":  ("F#1", "F#2", ["A3", "D4", "F#4", "A4"], ["A3", "D4", "F#4", "A4"]),
    "Em7":   ("E2", "E2", ["G3", "B3", "D4", "E4"], ["G3", "B3", "D4", "E4"]),
    "A":     ("A1", "A2", ["E3", "A3", "C#4", "E4"], ["A3", "C#4", "E4", "A4"]),
    "Asus4": ("A1", "A2", ["E3", "A3", "D4", "E4"], ["A3", "D4", "E4", "A4"]),
}

# per bar: list of (beat, chord)
HARM = {}
def _set(bar, *chs):
    HARM[bar] = [(b, c) for b, c in chs] if isinstance(chs[0], tuple) else [(0, chs[0])]

_set(1, "Bm"); _set(2, "Bm"); _set(3, "G"); _set(4, (0, "Asus4"), (2, "A"))
for i, c in enumerate(["Bm", "G", "D", "A"] * 7):          # bars 5-32
    _set(5 + i, c)
for i, c in enumerate(["G", "D/F#", "Em7"]):
    _set(33 + i, c)
_set(36, (0, "Asus4"), (1, "A"))
for i, c in enumerate(["G", "D/F#", "Bm7"]):
    _set(37 + i, c)
_set(40, (0, "Asus4"), (2, "A"))
NEIGH = ((0, "A"), (1, "Asus4"), (2, "A"))
SUS = ((0, "Asus4"), (2, "A"))
FINAL = [("Bm9", "Gmaj7", "D/F#", NEIGH), ("Bm7", "Gadd9", "D", SUS),
         ("Bm9", "Gmaj7", "D/F#", NEIGH), ("Bm7", "Gmaj9", "D/F#", SUS)]
for p, ph in enumerate(FINAL):
    for i, c in enumerate(ph):
        bar = 41 + p * 4 + i
        if isinstance(c, tuple):
            _set(bar, *c)
        else:
            _set(bar, c)
_set(57, "Gmaj7"); _set(58, "D/F#"); _set(59, "Em7"); _set(60, *SUS)
_set(61, (0, "Gadd9"), (2, "A"))
_set(62, "D"); _set(63, "D"); _set(64, "D")


def chord_spans(bar0, bar1):
    """[(start_sample, end_sample, chord)] for bars bar0..bar1 inclusive"""
    out = []
    for b in range(bar0, bar1 + 1):
        segs = HARM[b]
        for i, (beat, c) in enumerate(segs):
            s = b2s(b, beat)
            e = b2s(b, segs[i + 1][0]) if i + 1 < len(segs) else b2s(b + 1)
            if out and out[-1][2] == c and out[-1][1] == s:
                out[-1] = (out[-1][0], e, c)
            else:
                out.append((s, e, c))
    return out


def phrase(bars, start_bar, transpose=0, vel_down=1.0, vel_other=0.84, legato=1.0):
    """theme bars -> [(abs_start, len, midi, vel)]"""
    out = []
    for i, bar in enumerate(bars):
        pos = 0
        for (l, nn) in bar:
            if nn is not None:
                s = b2s(start_bar + i) + pos * E8
                v = vel_down if pos == 0 else (0.93 if pos % 2 == 0 else vel_other)
                out.append((s, int(l * E8 * legato), nm(nn) + transpose, v))
            pos += l
    return out
