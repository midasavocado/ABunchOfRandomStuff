"""s26 HERO THREE (2430-2610, final drop): orbital sunrise. An astronaut rests at a handrail on the station's end
(the s22 spine, finished now), facing the night-side Earth where cities glitter; the limb is a thin blue line
with the amber sunrise arc gathering. The camera moves in slowly past the truss; the sun breaks the horizon,
warm light sweeps across the suit and the station, and we end close on the gold visor, the sunrise arc curving
across it (-> s27, the same arc in the child's eye).
Earth is the analytic world (lib/earth.py, EEVEE). CPU lookdev: SB_EARTH_PROXY=1 with Cycles."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, earth, suit, rocket
import launchpad as LP
import timeline as TL
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s26"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render(os.environ.get("SB_ENGINE", "EEVEE"), cycles_samples=96, samples=48, mblur=True, shutter=0.5,
                look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.3")))
ALT = 410.0
dip = math.degrees(math.acos(earth.R / (earth.R + ALT)))
SUN_AZ = 0.0                               # sunrise straight ahead along +Y
RISE_F = F0 + 90                           # the sun breaks the limb on the resolution downbeat (bar 57, 105.0 s)


def sun_elev(f):
    """apparent elevation (vs local horizontal) rising ~0.065 deg/frame (orbital sunrise is fast)"""
    return -(dip + 0.9) + 0.065 * (f - RISE_F) + 0.9


E = earth.build(alt_km=ALT, sun_dir=tuple(sb.sun_dir(sun_elev(F0), SUN_AZ)), clouds=0.45, lights=float(os.environ.get("S26_LIGHTS", "14.0")), samples=16,
                nadir=(20.0, 10.0, 0.0), detail=0.9)
lamp = E.sun_lamp(strength=0.0)
for f in range(F0 - 2, F1 + 3):
    d = sb.sun_dir(sun_elev(f), SUN_AZ)
    for i in range(3):
        E.sun_node.inputs[i].default_value = d[i]
        E.sun_node.inputs[i].keyframe_insert("default_value", frame=f)
    lamp.rotation_quaternion = (-d).to_track_quat('-Z', 'Y')
    lamp.keyframe_insert("rotation_quaternion", frame=f)
    T = earth.sun_transmittance(E.cam_km((0, 0, 0)), d)
    m_ = max(T)
    lamp.data.energy = E.sun_strength * m_
    lamp.data.color = tuple(E.sun_color[i] * T[i] / m_ for i in range(3)) if m_ > 0 else (1, 0.5, 0.2)
    lamp.data.keyframe_insert("energy", frame=f)
    lamp.data.keyframe_insert("color", frame=f)
# faint earthshine / city glow from below before sunrise
sb.light('AREA', "NightFill", loc=(0, 0, -40), target=(0, 0, 0), energy=1.2e4, color=(0.5, 0.6, 1.0), size=80.0)

# ---- station end: truss spine (from s22, now complete) running toward -Y behind the astronaut
alu = sb.brushed_metal("TrussAlu", (0.78, 0.78, 0.8), rough=0.28, aniso=0.4, scale=30, scratches=0.3)
graph = sb.painted("StationGraphite", (0.04, 0.042, 0.047), rough=0.4, wear=0.3)
amber = sb.painted("StationAmber", (0.75, 0.32, 0.04), rough=0.3, coat=0.4, wear=0.2)
cells = sb.mat("Cells", (0.02, 0.03, 0.07), rough=0.15, coat=1.0, coat_rough=0.02)
B = LP.Beams(26)
BAY, W = 2.4, 1.8
for i in range(12):
    y = -i * BAY
    c = [V((-W / 2, y, -W / 2)), V((W / 2, y, -W / 2)), V((W / 2, y, W / 2)), V((-W / 2, y, W / 2))]
    for k in range(4):
        B.tube(c[k], c[(k + 1) % 4], 0.035, 10)
        cn = [p + V((0, -BAY, 0)) for p in c]
        B.tube(c[k], cn[k], 0.05, 12)
        B.tube(c[k], cn[(k + 1) % 4], 0.03, 8)
B.build("Spine", alu)
for i in range(0, 12, 3):
    for (x, z) in ((-W / 2, -W / 2), (W / 2, -W / 2), (W / 2, W / 2), (-W / 2, W / 2)):
        rocket.box("NodeBand", (x, -i * BAY, z), (0.2, 0.2, 0.05), amber, bev=0.005)
for side in (-1, 1):
    for k in range(2):
        y = -6.0 - k * 12.0
        rocket.rod("ArrayMast", (side * W / 2, y, 0), (side * 24.0, y, 0), 0.08, graph, verts=12)
        for j in range(7):
            rocket.box("Array", (side * (3.5 + j * 2.8), y, 0), (2.6, 4.2, 0.03), cells, bev=0.004, rot=(0, math.radians(30 * side), 0))
# handrail on the end face + foot restraint
rail = [V((-0.8, 0.15, W / 2 + 0.25)), V((0.8, 0.15, W / 2 + 0.25))]
rocket.rod("Handrail", rail[0], rail[1], 0.02, amber, verts=12)
for p in rail:
    rocket.rod("RailPost", p, p - V((0, 0, 0.25)), 0.015, amber, verts=8)
# ---- the astronaut: seated on the truss end-face edge, facing +Y (the sunrise), one hand on the rail
S = suit.build("Astro", visor="gold", gloves=("relaxed", "relaxed"), dust=0.0, dirt=0.2)
astro = S["root"]
astro.location = (0.25, 0.35, W / 2 - 0.2)
astro.rotation_euler = (0, 0, math.radians(180))          # suit faces -Y by default -> face +Y
for f in range(F0 - 2, F1 + 3):
    t = (f - F0) / (F1 - F0)
    suit.pose(S, frame=f, torso=(8, 0, 0), l_hip=(70, 8, 0), r_hip=(75, 10, 0), l_knee=70, r_knee=78,
              l_shoulder=(20, 12, 0), r_shoulder=(35 + 3 * math.sin(f * 0.04), 18, 0), l_elbow=30, r_elbow=55,
              r_wrist=(10, 0, 0))
# ---- camera: a slow 180-degree arc around the astronaut - from behind (their silhouette against the gathering
# arc) around to the front, ending close on the gold visor where the sunrise arc curves across it
bpy.context.scene.frame_set(F1 - 1)
bpy.context.view_layer.update()
vis = S["parts"].get("AstroVisor")
dg = bpy.context.evaluated_depsgraph_get()
ev = vis.evaluated_get(dg)
pts = [ev.matrix_world @ v.co for v in ev.data.vertices]
C = sum(pts, V((0, 0, 0))) / len(pts)
HZ = V((0.0, 400.0, -45.0))                        # a point on the horizon ahead (+Y, below local horizontal)


def cp(t):
    # first half: behind the astronaut, drifting in - the silhouette against the gathering arc as the sun breaks the
    # limb (mid-shot); second half: swing round to the front, ending close on the gold visor
    if t < 0.5:
        k = sb.smooth(t / 0.5)
        th = math.radians(sb.lerp(200.0, 182.0, k)); r = sb.lerp(4.4, 3.2, k); h = sb.lerp(0.95, 0.75, k)
    else:
        k = sb.smoother((t - 0.5) / 0.5)
        th = math.radians(sb.lerp(182.0, 20.0, k)); r = sb.lerp(3.2, 0.85, k ** 1.2); h = sb.lerp(0.75, 0.42, k)
    return C + V((math.sin(th) * r, math.cos(th) * r, h))


def ct(t):
    # the astronaut sits low-left of centre with the horizon arc across the frame; then the target settles on the visor
    early = C + V((0.25, 6.0, -1.1))
    k = sb.smoother(sb.remap(t, 0.5, 0.95))
    return early.lerp(C, k)


cam = sb.camera("Cam", loc=cp(0), target=ct(0), lens=28, fstop=4.0, clip=(0.02, 20000.0))
sb.cam_bake(cam, F0, F1, cp, ct, lens=lambda t: sb.lerp(28.0, 55.0, sb.smoother(t)),
            focus=lambda t: max(0.4, (cp(t) - C).length))
# auto-exposure ride: the camera is wide open in the dark (night side, city lights, airglow), then settles as the
# sun breaks the limb (like a real camera adjusting) - the sunrise still blooms, it does not flash white
vs = bpy.context.scene.view_settings
base_exp = vs.exposure
for f in range(F0 - 2, F1 + 3):
    k = sb.smoother((f - (RISE_F - 20)) / 40.0)
    vs.exposure = base_exp + sb.lerp(2.6, 0.0, k)
    vs.keyframe_insert("exposure", frame=f)
E.track(cam, F0 - 2, F1 + 2)
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
