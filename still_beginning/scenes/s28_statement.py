"""s28 THE STATEMENT (2700-2880): clean, slowly evolving Earth-limb plate. Title is added in post, so the
upper/left area stays empty black space. The camera flies a real orbit (local-vertical frame): over 7.5 s the
sun climbs ~0.5 deg toward the limb, so the amber sunrise arc slowly brightens — the sun never breaks the
limb (no flash). Pure world render (no geometry)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, earth
from mathutils import Vector as V, Quaternion, Euler
import timeline as TL

sid = (sb.argv() or ["s28"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", samples=24 if sb.RES == "final" else 12, mblur=False,
                look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.6")))

ALT = 410.0
RO = earth.R + ALT
V_ORB = 7.66            # km/s
LENS = 45.0
PITCH = -15.2           # camera pitch vs local horizontal (limb dip at 410 km ~ 19.9 deg)
YAW0 = 4.0             # small yaw right so the sun glow sits right of centre
SUN_BELOW = 0.95        # sun centre this far below the limb at F0; rises ~0.49 deg -> ends 0.46 (disc radius 0.27: never breaks)
SUN_AZ = 13.0

dip = math.degrees(math.acos(earth.R / RO))
sd = sb.sun_dir(-(dip + SUN_BELOW), SUN_AZ)
E = earth.build(alt_km=ALT, sun_dir=tuple(sd), clouds=0.45, lights=float(os.environ.get("SB_LIGHTS", "1.0")),
                samples=16, nadir=(18.0, 40.0, -10.0), detail=0.85)

cam = sb.camera(loc=(0, 0, 0), target=(0, 1, 0), lens=LENS, clip=(0.1, 1000))


def orbit_theta(f):
    return (f - F0) / 24.0 * V_ORB / RO           # radians travelled along the orbit


def cam_pos(f):
    th = orbit_theta(f)
    c = V((0, 0, -RO))                            # Earth centre (km)
    p = c + V((0, math.sin(th), math.cos(th))) * RO
    return p * 1000.0                             # metres


def cam_rot(f):
    th = orbit_theta(f)
    t = (f - F0) / (F1 - 1 - F0)
    k = sb.smoother(t)
    yaw = math.radians(YAW0 + 0.9 * k)            # very slow drift
    pit = math.radians(PITCH - 0.35 * k)
    base = Euler((math.pi / 2 + pit, 0, -yaw), 'XYZ').to_quaternion()
    lvlh = Quaternion((1, 0, 0), -th)             # rotate with the local vertical as we fly toward +Y
    return lvlh @ base


for f in range(F0, F1):
    cam.location = cam_pos(f)
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = cam_rot(f)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
E.track(cam, F0, F1 - 1)
sb.frames(F0, F1)
sb.render_shot(sid)
