"""s05 (360-450) INTELLIGENCE AS A TOOL: match from the beta-strands into the parallel gold bond fingers of a
processor package; glide along them, rise over the arcs of 25 um gold wire bonds onto the iridescent die, then pull
back to reveal the package. Restrained activity: sparse warm pulses travel outward along a few fan-out traces on
the beat (the beat begins to assemble)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, chip, dbgcam
from mathutils import Vector as V, Quaternion

sid = (sb.argv() or ["s05"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
F0, F1 = 360, 450
mm = 0.001
C = chip.build()

# pulse clock: pulses launch on beats from ~frame 378 on (phase values are in frames after PULSE_T0)
PULSE_T0 = 378.0
tv = C["pulse_time"]
mat = bpy.data.materials["Trace"]
for f in range(F0 - 2, F1 + 2):
    # time value in "frames since T0" / 11.25 per unit pulse travel (speed 0.9 -> ~1.1 beats to cross)
    tv.default_value = (f - PULSE_T0) / 11.25
    tv.keyframe_insert("default_value", frame=f)
for n in mat.node_tree.nodes:
    if n.name == "PulseTime":
        pass
# act phases were written in frames: convert to the same unit
tr = C["traces"]
a = np.empty(len(tr.data.vertices), np.float32)
tr.data.attributes["act"].data.foreach_get("value", a)
a = np.where(a >= 0, a / 11.25, -1.0).astype(np.float32)
tr.data.attributes["act"].data.foreach_set("value", a)

# ------------------------------------------------------------------ light: macro studio
sb.world_gradient(top=(0.020, 0.024, 0.032), horizon=(0.035, 0.038, 0.045), bottom=(0.01, 0.01, 0.012), strength=0.8)
# large soft panel to the north, high: the die mirrors it (camera looks north)
sb.light('AREA', "Panel", loc=(0.0, 0.030, 0.030), target=(0, 0, 0), energy=0.008, color=(0.88, 0.93, 1.0), size=0.05, size_y=0.03)
# raking light from the east: catches the wire arcs and finger edges
sb.light('AREA', "Rake", loc=(0.040, -0.010, 0.006), target=(0, -0.004, 0.0003), energy=0.006, color=(1.0, 0.95, 0.88), size=0.012)
sb.light('AREA', "Fill", loc=(-0.03, -0.04, 0.03), target=(0, -0.004, 0), energy=0.002, color=(0.8, 0.86, 1.0), size=0.05)
sb.light('AREA', "Top", loc=(0.0, -0.004, 0.05), target=(0, -0.004, 0), energy=0.003, color=(0.95, 0.96, 1.0), size=0.04)

# ------------------------------------------------------------------ camera
cam = sb.camera("Cam", loc=(0, -0.01, 0.002), target=(0, -0.006, 0), lens=50, fstop=22, clip=(0.00005, 1.0))


def dirv(t):
    a = V((0.0, -0.965, 0.26)).normalized()
    b = V((0.05, -0.80, 0.60)).normalized()
    c = V((-0.32, -0.62, 0.72)).normalized()
    if t < 0.5:
        return a.slerp(b, sb.smooth(t / 0.5))
    return b.slerp(c, sb.smooth((t - 0.5) / 0.5))


def dist(t):
    k = sb.smoother(sb.remap(t, 0.28, 1.0))
    return math.exp(sb.lerp(math.log(1.25 * mm), math.log(19.0 * mm), k))


def tgt(t):
    y = sb.lerp(-6.05, -5.2, sb.smooth(sb.remap(t, 0.0, 0.32)))
    y = sb.lerp(y, -0.8, sb.smoother(sb.remap(t, 0.3, 1.0)))
    z = sb.lerp(0.02, 0.36, sb.smooth(sb.remap(t, 0.12, 0.45)))
    x = 0.12 * math.sin(t * 2.0) * sb.remap(t, 0.0, 0.5)
    return V((x * mm, y * mm, z * mm))


def pos(t):
    return tgt(t) + dirv(t) * dist(t)


sb.cam_bake(cam, F0, F1, pos, tgt, lens=lambda t: sb.lerp(50, 42, sb.smooth(t)),
            focus=lambda t: (pos(t) - tgt(t)).length, fstop=lambda t: math.exp(sb.lerp(math.log(320), math.log(40), sb.smooth(sb.remap(t, 0.2, 1.0)))))
cam.data.clip_start = 0.00003

dbgcam.apply()
sb.frames(F0, F1)
sb.save("s05")
sb.render_shot(sid)
