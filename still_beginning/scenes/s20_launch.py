"""s20 LAUNCH - HERO SHOT TWO (1800-1980). Ignition exactly on 1800, hold-downs release ~1830, heavy accelerating
ascent, vehicle clears the tower. Remote long-lens camera ~525 m out behind a blast berm, tilting with the climb."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, rocket, launchpad as LP, launchfx as FX
from mathutils import Vector as V

sid = (sb.argv() or ["s20"])[0]
_, F0, F1, _ = sb.TL.shot(sid)          # 1800, 1980
sc = sb.reset()
tile = {"preview": 4, "half": 2, "final": 4}[sb.RES]
sb.setup_render("EEVEE", samples=int(os.environ.get("SB_SAMPLES", "0")) or None, mblur=True, shutter=0.5,
                volumes=True, vol_tile=tile, vol_samples=96)
sc.eevee.volumetric_shadow_samples = 12
sc.eevee.volumetric_light_clamp = 0.0

# ---------------------------------------------------------------- world, complex, vehicle
FX.environment(sc)
CX = LP.build_complex()
RK = rocket.build(frost=1.0)
RK["root"].location = (0, 0, LP.ROCKET_Z0)
CAM_P = V((-240.0, -465.0, 4.0))
# blast berm between camera and pad, low so the pad base stays visible
VD = V((-CAM_P.x, -CAM_P.y, 0)).normalized()
LP.berm("Berm", V((CAM_P.x, CAM_P.y, 0)) + VD * 42.0, 90.0, math.degrees(math.atan2(-VD.x, VD.y)), CX["M"], h=3.55, w=10.0)

# ---------------------------------------------------------------- animation
FA, FB = F0 - 2, F1 + 1
FX.animate_vehicle(RK, FA, FB)
FX.animate_pad(CX, FA, FB)
P = FX.plume_rig(RK)
FX.animate_plume(P, FA, FB)
smoke, _ = FX.launch_smoke(FA, FB)

# exposure: operator rides the iris down as the fire builds (~1 stop over 0.8 s)
for f in range(FA, FB + 1):
    sc.view_settings.exposure = 0.35 - 1.5 * sb.smoother((f - F0) / 20.0)
    sc.view_settings.keyframe_insert("exposure", frame=f)

# ---------------------------------------------------------------- camera: tilt with mass (critically damped follow)
cam = sb.camera("Cam", loc=CAM_P, target=(0, 0, 40), lens=100, fstop=5.6, clip=(1.0, 20000))
cam.data.dof.focus_distance = (V((0, 0, 40)) - CAM_P).length
FX.volume_range(sc, cam, (0, 0, 40), near=300.0, far=900.0)
sc.eevee.volumetric_sample_distribution = 0.2


def desired(f):
    h = FX.ascent(f)
    return 40.0 + 1.0 * h + 9.0 * sb.smoother(h / 60.0)


zs, vz = 40.0, 0.0
omega = 2 * math.pi / 0.9          # operator response ~1 s
dt = 1.0 / 24.0
track = {}
for f in range(FA - 48, FB + 1):
    d = desired(f)
    a = omega * omega * (d - zs) - 2 * omega * vz
    vz += a * dt
    zs += vz * dt
    track[f] = zs
for f in range(FA, FB + 1):
    tgt = V((0.4, 0.0, track[f]))
    cam.location = CAM_P
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = sb.look_quat(CAM_P, tgt)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)

sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save("s20")
sb.render_shot(sid)
