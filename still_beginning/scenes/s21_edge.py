"""s21 THE EDGE (1980-2070): staging at the edge of space, dawn. The stack coasts high over the curved Earth, the
amber sunrise arc lying along the limb; the first stage separates and falls back, turning slowly; the upper
stage's vacuum engine lights - a wide, faint vacuum plume blooms and the nozzle extension begins to glow.
Chase camera flying with the upper stage, 3/4 behind and below, so Earth fills the lower frame."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import sb, rocket, earth
import timeline as TL
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s21"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
ENGINE = os.environ.get("SB_ENGINE", "EEVEE")          # the analytic Earth is EEVEE-only (Cycles: SVM stack)
sb.setup_render(ENGINE, cycles_samples=96, samples=32, mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.2")))
ALT = 78.0
# sun a hair below the horizon ahead-left: the limb glows amber, the vehicle is lit by the first grazing light
SUN_EL, SUN_AZ = -4.0, 90.0      # below local horizontal, above the geometric horizon (dip ~9 deg): the vehicle
                                   # is lit gold through the thin upper air while the ground is still in night
sd = sb.sun_dir(SUN_EL + 0.0, SUN_AZ)
E = earth.build(alt_km=ALT, sun_dir=tuple(sd), clouds=0.45, lights=0.35, samples=16, nadir=(28.0, -60.0, 20.0), detail=0.9)
E.sun_lamp()

# ---- the stack, pitched ~18 deg above the local horizontal, flying toward +Y
PITCH = math.radians(90 - 18)
Rveh = Euler((-PITCH, 0.0, 0.0), 'XYZ').to_matrix()          # vehicle +Z (nose) -> mostly +Y, a little up
RK = rocket.build(frost=0.0, vac=True)
root = RK["root"]
s1, s2 = RK["stage1"], RK["stage2"]
SEP = F0 + 20                     # separation
IGN = F0 + 44                     # upper-stage ignition
A_S1 = 11.0                       # m/s^2 apparent separation acceleration (upper stage accelerating away)

root.rotation_mode = 'QUATERNION'
root.rotation_quaternion = Rveh.to_quaternion()
root.location = (0, 0, 0)
# stage 1 detaches: parented to root, we animate its local offset along -Z with a slow tumble after SEP
s1.rotation_mode = 'XYZ'
for f in range(F0 - 2, F1 + 3):
    t = max(0.0, (f - SEP) / 24.0)
    push = 0.35 * min(t, 0.5)                                  # spring pushers (first 0.5 s)
    run = 0.5 * 1.2 * t * t + (0.5 * A_S1 * max(0.0, (f - IGN) / 24.0) ** 2)
    s1.location = (0.0, 0.0, -(push + run))
    s1.rotation_euler = (math.radians(1.8) * t * t, math.radians(0.9) * t, 0.0)
    s1.keyframe_insert("location", frame=f)
    s1.keyframe_insert("rotation_euler", frame=f)
# nozzle extension heat glow after ignition
vm = RK["mats"]["vac_ext"]
vg = rocket.vac_ext_mat("VacExtGlow", glow=1.0)
for o in bpy.data.objects:
    if o.type == 'MESH' and o.data.materials and o.data.materials[0] == vm:
        o.data.materials[0] = vg
gain = vg.node_tree.nodes["Glow"]
for f in range(F0 - 2, F1 + 3):
    gain.inputs[1].default_value = 3.0 * sb.smoother((f - IGN - 6) / 40.0)
    gain.inputs[1].keyframe_insert("default_value", frame=f)
# vacuum plume: very wide, faint, warm core; grows from ignition
eng = bpy.data.objects.get("VacEng")
exit_z = RK["dims"]["band"][0] - 5.15 if eng is None else None
pm = rocket.plume_material("VacPlume", intensity=0.6, core_col=(1.0, 0.9, 0.75), mid_col=(1.0, 0.55, 0.22),
                           tail_col=(0.7, 0.35, 0.2), length=90.0, diamonds=0.0)
plume = rocket.plume_mesh("VacPlumeMesh", length=70.0, r0=1.15, r_max=14.0, neck=0.05, segs=96, n=48, mat=pm, parent=None, knee=0.08)
plume.parent = eng if eng is not None else root
plume.location = (0, 0, -4.4) if eng is not None else (0, 0, 38.85)
g_node = pm.node_tree.nodes.get("Gain")
for f in range(F0 - 2, F1 + 3):
    k = sb.smoother((f - IGN) / 10.0)
    plume.scale = (0.2 + 0.8 * k, 0.2 + 0.8 * k, max(0.01, k))
    plume.keyframe_insert("scale", frame=f)
    if g_node:
        g_node.inputs[1].default_value = 0.35 * k
        g_node.inputs[1].keyframe_insert("default_value", frame=f)
    plume.hide_render = k < 0.01
    plume.keyframe_insert("hide_render", frame=f)
# ignition flash light inside the nozzle (warm, short)
fl = sb.light('POINT', "IgnLight", loc=(0, 0, 0), energy=0.0, color=(1.0, 0.6, 0.3), size=0.8)
fl.parent = eng if eng is not None else root
fl.location = (0, 0, -3.0)
for f in range(F0 - 2, F1 + 3):
    fl.data.energy = 60000.0 * sb.smoother((f - IGN) / 6.0) * (0.8 + 0.2 * math.sin(f * 1.7))
    fl.data.keyframe_insert("energy", frame=f)

# ---- chase camera: side-on, flying with the upper stage, looking slightly down so the curved horizon (and the
# amber dawn band on it) crosses the frame beneath the vehicle; drifts back as the stage falls away
band_z = RK["dims"]["band"][0]
axis = Rveh @ V((0, 0, 1))
inter = Rveh @ V((0, 0, band_z - 4.0))
side = axis.cross(V((0, 0, 1))).normalized()


def cam_world(t):
    k = sb.smoother(t)
    return inter + side * sb.lerp(-78.0, -88.0, k) - axis * sb.lerp(8.0, 22.0, k) + V((0, 0, sb.lerp(20.0, 24.0, k)))


def tgt_world(t):
    k = sb.smoother(t)
    return inter - axis * sb.lerp(6.0, 16.0, k) + V((0, 0, 2.0))


# the vehicle is backlit by the grazing sun: without fill it is a flat black cutout. A faint cool fill from the
# camera side (light scattered off the limb and the upper air) lets the white body and its detail read, and a warm
# rim from the sun side traces the silhouette.
def _dir_light(name, direction, energy, color):
    ld = bpy.data.lights.new(name, 'SUN'); ld.energy = energy; ld.color = color; ld.angle = math.radians(8.0)
    o = bpy.data.objects.new(name, ld); sb.link_obj(o)
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = V(direction).normalized().to_track_quat('-Z', 'Y')
    return o
_view = (inter - cam_world(0.5)).normalized()
_dir_light("LimbFill", _view + V((0, 0, 0.35)), 0.35, (0.62, 0.72, 1.0))
_dir_light("SunRim", -V(sd) + V((0, 0, -0.05)), 2.2, (1.0, 0.62, 0.32))
cam = sb.camera("Cam", loc=cam_world(0), target=tgt_world(0), lens=35, fstop=11.0, clip=(0.5, 20000.0))
for f in range(F0 - 2, F1 + 3):
    t = (f - F0) / (F1 - 1 - F0)
    p = cam_world(t)
    cam.location = p
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = sb.look_quat(p, tgt_world(t), roll=math.radians(4.0))
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
import dbgcam; dbgcam.apply(); cam = bpy.context.scene.camera
E.track(cam, F0 - 2, F1 + 2)
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
exec(open(os.environ['SB_PROBE']).read()) if os.environ.get('SB_PROBE') else sb.render_shot(sid)
