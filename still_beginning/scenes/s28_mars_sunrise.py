"""s28 WE'RE JUST GETTING STARTED (2610-2880): the last shot. High over the night side of Mars, the planet's limb
curves across the lower frame under the stars; our orbit carries us toward the dawn. A blue arc of Martian twilight
brightens on the horizon, then the sun breaks over the limb, the thin atmosphere flares blue-white around it and a
crescent of red daylight spreads along the edge of the world. The title (post.py, frame 2745) arrives in that light."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, mars
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion

sid = (sb.argv() or ["s28"])[0]
_, S0, S1, _ = sb.TL.shot(sid)
F0, F1 = S0 - 24, S1 + 2
T0 = time.time()
sc = sb.reset()
sb.setup_render("EEVEE", mblur=False, look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.0")))

H = 1800e3                                            # orbit altitude
SUN = (mars.EAST * -0.2 + mars.POLE * 0.25 + mars.QREF * -1.0).normalized()
W = SUN.cross(mars.POLE).normalized()                 # orbit plane contains SUN and W
DIP = math.acos(mars.R / (mars.R + H))                # horizon dip
PHI_RISE = math.pi / 2 + DIP                          # sun on the limb
SUNRISE = float(os.environ.get("S28_RISE", str(S0 + 70)))
RATE = math.radians(float(os.environ.get("S28_ARC", "26"))) / (S1 - S0)   # orbital angle per frame (cinematic)


def phi(f):
    return PHI_RISE + (SUNRISE - f) * RATE


def u_of(f):
    p = phi(f)
    return (SUN * math.cos(p) + W * math.sin(p)).normalized()


# planet + atmosphere as limb patches around the mid-shot sub-camera point
um = u_of((S0 + S1) / 2)
refd = SUN
pm = mars.planet_mat("MarsSurf", detail=1.0)
planet = mars.limb_patch("MarsLimb", um, refd, 0.0, DIP + math.radians(32.0), DIP, math.radians(80.0), n_th=720, n_az=1800,
                         mat=pm)
planet.visible_shadow = False
shell = mars.limb_patch("AtmoLimb", um, refd, 0.0, DIP + math.radians(34.0), DIP + math.radians(1.2), math.radians(80.0),
                        n_th=420, n_az=1800, radius=mars.R + 42e3)
atm, ag = mars.atmosphere("MarsAtmo", sun=SUN, obj=shell, gain=float(os.environ.get("S28_ATMO", "1.0")))
w, smix = mars.world(SUN, space_mix=1.0, stars=1.0)
sun = sb.light('SUN', "Sun", energy=4.6, color=(1.0, 0.95, 0.9), angle=0.35)
sun.rotation_mode = 'QUATERNION'; sun.rotation_quaternion = SUN.to_track_quat('Z', 'Y')

# the colony's lights on the night side, a pinprick cluster (with the ships' pad lamps) near the terminator
lights_m = sb.emit_mat("NightLights", (1.0, 0.72, 0.42), 400.0)
hd_m = (SUN - um * um.dot(SUN)).normalized()
cl = mars.C + (um * math.cos(math.radians(30)) + hd_m * math.sin(math.radians(30))).normalized() * (mars.R + 50.0)
rng = np.random.default_rng(3)
nrm = (cl - mars.C).normalized()
tx = nrm.cross(mars.POLE).normalized(); ty = nrm.cross(tx)
for k in range(40):
    off = tx * rng.normal(0, 5000) + ty * rng.normal(0, 3500)
    sb.prim("ico", "Light", loc=tuple(cl + off), subdivisions=1, radius=160.0 + rng.uniform(0, 160), mat=lights_m)

# camera: rides the orbit, looking along the horizon toward the dawn; tilts up a touch as the sun clears the limb
cam = sb.camera("Cam", loc=(0, 0, 0), target=(0, 0, 1), lens=float(os.environ.get("S28_LENS", "32")), clip=(2000.0, 1.2e8))
for f in range(F0, F1 + 1):
    u = u_of(f)
    p = mars.C + u * (mars.R + H)
    # direction to the horizon point under the sun, lifted so the limb sits in the lower third
    hdir = (SUN - u * u.dot(SUN)).normalized()
    look = (hdir * math.cos(DIP) - u * math.sin(DIP)).normalized()
    k = sb.smoother(sb.remap(f, SUNRISE - 30, SUNRISE + 120))
    lift = math.radians(9.0 + 11.0 * k)            # tilt with the rising sun as the dawn spreads below
    axis = look.cross(u).normalized()
    look = (Quaternion(axis, -lift) @ look).normalized()
    mars.aim(cam, p, p + look * 1e6, up=u, frame=f)
    # sun visibility: angular clearance of the camera->sun ray over the limb (+ a little refraction-free margin)
    pc = p - mars.C
    ps = pc.dot(SUN)
    bb = math.sqrt(max(pc.length ** 2 - ps * ps, 0.0))
    marg = 1.0 if ps > 0 else (bb - mars.R - 4e3) / math.sqrt(max(pc.length ** 2 - mars.R ** 2, 1.0))
    sv = w.node_tree.nodes["SunVis"]
    sv.outputs[0].default_value = sb.smoother(sb.remap(math.degrees(marg), -0.15, 0.35))
    sv.outputs[0].keyframe_insert("default_value", frame=f)
    cam.data.dof.focus_distance = 3.9e6
    cam.data.dof.keyframe_insert("focus_distance", frame=f)
print("s28 built in %.0fs; sunrise at f%.0f" % (time.time() - T0, SUNRISE))
sb.render_shot(sid)
