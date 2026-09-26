"""s24c A CITY ON MARS (2355-2475): a drone flies in low over the solar field toward a living colony: regolith-shielded
habitats ringed around a lit hub, glass greenhouse domes glowing green, roads and rovers, three landing pads. A colony
ship settles onto the near pad on its engines (legs swinging down, a ring of dust racing outward), another stands
serviced on its pad, and far off a third lifts away on a long plume. Afternoon sun, butterscotch sky, dust haze.
Colony at the origin; lib/mars.py terrain (the same surface s25z pulls back from)."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, mars, colony
import numpy as np
from mathutils import Vector as V, Euler, Matrix

sid = (sb.argv() or ["s24c"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
F0, F1 = S0 - 24, S1 + 24                       # animation span incl. edit handles
T0 = time.time()
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
CO = colony.build(F0, F1)
posA, PAD_A = CO["posA"], CO["PAD_A"]
T0b = time.time()

# ---------------------------------------------------------------- camera: drone flies north up the main avenue - over
# the solar farm, past the hub tower (25 m to its side, above its beacon), the greenhouse vaults glowing to the left,
# industry to the right - and settles on the ship touching down at the spaceport
CP = [V((38.0, -340.0, 24.0)), V((32.0, -170.0, 32.0)), V((26.0, 40.0, 50.0)), V((2.0, 250.0, 52.0)), V((-26.0, 350.0, 46.0))]
LEAD = [V((20.0, -60.0, 8.0)), V((0.0, 180.0, 14.0)), V((-60.0, 420.0, 26.0))]


def cpos(t):
    return sb.catmull(CP, t)


def ctgt(t):
    f = F0 + (F1 - F0) * t
    shipA = posA(f) + V((0, 0, 24.0))
    ahead = sb.catmull(LEAD, t)
    return ahead.lerp(shipA, sb.smoother(sb.remap(t, 0.25, 0.8)))


cam = sb.camera("Cam", loc=CP[0], target=(0, 0, 0), lens=28, fstop=11.0, clip=(0.2, 200000))
sb.cam_bake(cam, F0, F1, lambda t: cpos(t), lambda t: ctgt(t), lens=lambda t: 26.0 + 9.0 * sb.smoother(t), focus="target",
            roll=lambda t: math.radians(-2.5) * math.sin(t * math.pi * 1.3))
print("s24c built in %.0fs" % (time.time() - T0))
sb.render_shot(sid)
