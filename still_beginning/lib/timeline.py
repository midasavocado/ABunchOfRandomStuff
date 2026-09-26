"""Master edit timeline. 24 fps, 128 BPM, 1 bar = 45 frames, 1 beat = 11.25 frames.
Every shot: (id, start_frame, end_frame_exclusive, scene_script, description)."""
FPS = 24
BPM = 128
BAR = 45          # frames
BEAT = 11.25      # frames
TOTAL = 2880


def bar(n, beat=0.0):
    """Frame of bar n (1-based) + beat offset, rounded to nearest frame."""
    return int(round((n - 1) * BAR + beat * BEAT))


SHOTS = [
    # id,    start, end,  scene
    ("s01",     0,   90, "s01_eye"),              # THE EYE
    ("s02",    90,  180, "s02_rooftop"),          # LOOKING: aperture -> child
    ("s03a",  180,  214, "s03_microscope"),       # fingertip on focus knob
    ("s03b",  214,  270, "s03_microscope"),       # track along optics to specimen
    ("s04",   270,  360, "s04_molecule"),         # HIDDEN STRUCTURE
    ("s05",   360,  450, "s05_chip"),             # INTELLIGENCE AS A TOOL
    ("s06a",  450,  506, "s06_design"),           # workstation over-shoulder
    ("s06b",  506,  540, "s06_design"),           # screen close: final choice
    ("s07a",  540,  574, "s07_robot"),            # laser fabrication insert
    ("s07b",  574,  630, "s07_robot"),            # robot wide: grip, transfer, release
    ("s08a",  630,  675, "s08_finger"),           # fit component into finger
    ("s08b",  675,  720, "s08_finger"),           # test movement
    ("s09a",  720,  765, "s09_table"),            # grasp close
    ("s09b",  765,  810, "s09_table"),            # reveal family table
    ("s10a",  810,  833, "s10_power"),            # busbar (2 beats)
    ("s10b",  833,  855, "s10_power"),            # cooling (2 beats)
    ("s10c",  855,  900, "s10_power"),            # coils (4 beats)
    ("s11",   900,  990, "s11_fusion"),           # NEW ENERGY
    ("s12a",  990, 1024, "s12_wind"),             # hub match
    ("s12b", 1024, 1080, "s12_wind"),             # ocean wide
    ("s13",  1080, 1170, "s13_solar"),            # ABUNDANCE
    ("s14",  1170, 1260, "s14_train"),            # MOVING FORWARD
    ("s15",  1260, 1350, "s15_city"),             # A FUTURE PEOPLE LIVE IN
    ("s16a", 1350, 1373, "s16_montage"),          # water into glass
    ("s16b", 1373, 1395, "s16_montage"),          # greenhouse leaves
    ("s16c", 1395, 1440, "s16_montage"),          # learning space
    ("s17",  1440, 1620, "s17_drawing"),          # HERO ONE
    ("s18",  1620, 1710, "s18_glove"),            # WE TAKE OUR CURIOSITY WITH US
    ("s19a", 1710, 1744, "s19_ready"),            # engines
    ("s19b", 1744, 1778, "s19_ready"),            # tower + venting
    ("s19c", 1778, 1800, "s19_ready"),            # hold-down clamp
    ("s20",  1800, 1980, "s20_launch"),           # HERO TWO
    ("s21",  1980, 2070, "s21_edge"),             # THE EDGE
    ("s22",  2070, 2130, "s22_orbit_build"),      # BUILDING BEYOND EARTH
    ("s23e", 2130, 2190, "s23_earth_moon"),       # Earth orbit -> the Moon grows -> dive to the south pole
    ("s23",  2190, 2265, "s23_moon"),             # ANOTHER SHORE: the lunar colony, landers down and up
    ("s23m", 2265, 2295, "s23_moon_mars"),        # leave the Moon into deep space
    ("s25",  2295, 2355, "s25_telescope"),        # LOOKING FARTHER: the telescope unfolds on the way out
    ("s24c", 2355, 2475, "s24_colony"),           # THE MARS COLONY: descend in; pads, starships landing + launching
    ("s24",  2475, 2515, "s24_mars"),             # LIFE TRAVELS WITH US (greenhouse insert)
    ("s25z", 2515, 2610, "s25_mars_zoom"),        # pull out: colony -> orbit -> the whole planet
    ("s28",  2610, 2880, "s28_mars_sunrise"),     # sunrise over the Mars limb -> THE STATEMENT
]


# ---- fluid edit: every shot renders HANDLE extra frames past each cut (not before the first / after the last) and
# each cut is a soft motion-continuous blend of TRANS[incoming sid] frames centred on the cut (default 10 = 0.42 s).
HANDLE = 8
TRANS_DEFAULT = 10
TRANS = {
    "s09a": 4, "s20": 4,                     # the first drop and ignition land on the beat
    "s02": 16, "s05": 16, "s12a": 16, "s18": 16, "s23": 16, "s23m": 14, "s24c": 16, "s25z": 14, "s28": 16,
}


def trans(sid):
    """blend length (frames) of the cut INTO sid (0 for the first shot)."""
    i = [s[0] for s in SHOTS].index(sid)
    return 0 if i == 0 else min(2 * HANDLE, TRANS.get(sid, TRANS_DEFAULT))


def handles(sid):
    """(frames rendered before start, frames after end)."""
    ids = [s[0] for s in SHOTS]
    i = ids.index(sid)
    return (0 if i == 0 else HANDLE), (0 if i == len(ids) - 1 else HANDLE)


def shot(sid):
    for s in SHOTS:
        if s[0] == sid:
            return s
    raise KeyError(sid)


def check():
    t = 0
    for s in SHOTS:
        assert s[1] == t, (s, t)
        assert s[2] > s[1]
        t = s[2]
    assert t == TOTAL
    return len(SHOTS)


if __name__ == "__main__":
    print(check(), "shots")
