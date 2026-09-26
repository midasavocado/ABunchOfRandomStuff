"""s25z THE WHOLE WORLD (2515-2610): from low over the colony (a ship lifting off its pad on a bright plume, the
landed ship on its pad, domes glowing) the camera pulls straight back and up - one exponential move, 140 m to
22,000 km - through the dust haze, the sky darkening to space, until all of Mars hangs in the frame: Valles
Marineris, the Tharsis volcanoes, a polar cap, the thin limb of air. Same surface at every scale (lib/mars.py);
the colony is lib/colony.py on the global timeline."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, mars, colony
import numpy as np
from mathutils import Vector as V

sid = (sb.argv() or ["s25z"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
F0, F1 = S0 - 24, S1 + 24
T0 = time.time()
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.35, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
CO = colony.build(F0, F1)
SUN = CO["SUN"]
planet = mars.planet("Mars", mat=mars.planet_mat("MarsPlanet"))
atm, ag = mars.atmosphere("MarsAtmo", sun=SUN)
smix = CO["smix"]
haze = CO["ground"].data.materials[0].node_tree.nodes["Haze"]
ALT0, ALT1 = 140.0, 2.2e7


def ease(t):
    return sb.smoother(t) * 0.75 + t * 0.25


def alt(t):
    return ALT0 * (ALT1 / ALT0) ** ease(t)


def lg(a):
    return math.log10(max(a, 1.0))


def pos(t):
    a = alt(t)
    k = 0.85 * (1.0 - sb.smoother(sb.remap(lg(a), 3.2, 5.6)))
    v = V((0.0, -k * a, mars.R + a))
    return mars.C + v.normalized() * (mars.R + a)


def tgt(t):
    a = alt(t)
    w = sb.smoother(sb.remap(lg(a), 3.8, 6.4))
    near = V((-30.0, 170.0, 20.0))
    return near.lerp(mars.C, w)


cam = sb.camera("Cam", loc=pos(0), target=(0, 0, 0), lens=28, fstop=16.0, clip=(0.2, 3e5))
cam.data.dof.use_dof = False
for f in range(F0, F1 + 1):
    t = (f - S0) / (S1 - 1 - S0)
    tc = max(0.0, min(1.0, t))
    p, g = pos(tc), tgt(tc)
    mars.aim(cam, p, g, up=mars.POLE, frame=f)
    a = alt(tc)
    cam.data.lens = 28.0 + 22.0 * sb.smoother(sb.remap(lg(a), 5.0, 7.3))
    cam.data.keyframe_insert("lens", frame=f)
    cam.data.dof.focus_distance = (g - p).length
    cam.data.dof.keyframe_insert("focus_distance", frame=f)
    cam.data.clip_start = max(0.2, a * 0.004)
    cam.data.clip_end = max(3e5, a * 4.0 + 5e6)
    cam.data.keyframe_insert("clip_start", frame=f); cam.data.keyframe_insert("clip_end", frame=f)
    # sky -> space, ground haze off, the limb of air on once we're above it
    smix.outputs[0].default_value = sb.smoother(sb.remap(lg(a), 4.0, 5.0))
    smix.outputs[0].keyframe_insert("default_value", frame=f)
    haze.outputs[0].default_value = 1.0 - sb.smoother(sb.remap(lg(a), 3.6, 4.6))
    haze.outputs[0].keyframe_insert("default_value", frame=f)
    ag.outputs[0].default_value = sb.smoother(sb.remap(lg(a), 4.7, 5.3))
    ag.outputs[0].keyframe_insert("default_value", frame=f)
print("s25z built in %.0fs" % (time.time() - T0))
sb.render_shot(sid)
