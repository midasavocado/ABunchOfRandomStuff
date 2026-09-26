"""s12 ENERGY AT SCALE. s12a: match-cut from the plasma ring to the spinner of a turbine hub (same centred circle,
radius ~0.41 frame height), blades sweeping; the camera pulls back and out into open air. s12b: helicopter
aerial over the ocean, turbines to the horizon, a crew-transfer vessel with a wake."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb
import numpy as np
import energy as E
import wind as W
import timeline as TL
from mathutils import Vector as V, Quaternion, Matrix

sid = (sb.argv() or ["s12a"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=float(os.environ.get("EXPO", "0.0")))
sc.eevee.use_shadows = True
rnd = random.Random(12)

SX, SY = 900.0, 1100.0           # turbine grid spacing (crosswind X, downwind Y)
SUN_EL, SUN_AZ = float(os.environ.get("SUNEL", "24")), float(os.environ.get("SUNAZ", "62"))

# ---- sky + sun (warm afternoon)
w, sky, bg = sb.world_sky(elev=SUN_EL, azim=SUN_AZ, strength=float(os.environ.get("SKY", "0.12")), sun_disc=False,
                          air=1.0, aerosol=1.6, ozone=1.0)
sun = sb.sun(SUN_EL, SUN_AZ, energy=float(os.environ.get("SUNE", "5.0")), color=(1.0, 0.90, 0.76), angle=0.55)
W.HAZE_COL = tuple(float(x) for x in os.environ.get("HAZECOL", "0.55,0.66,0.82").split(","))

# ---- turbines
M = W.turbine_mats()
for k in ('white', 'blade', 'yellow', 'grey', 'dark', 'galv'):
    W.aerial(M[k], dist=14000.0)
parts = W.turbine_parts(M, detail=True)
turbs = []
for i in range(-4, 9):
    for j in range(-2, 12):
        x, y = i * SX + (j % 2) * SX * 0.5 * 0, j * SY
        name = f"T{i}_{j}"
        phase = rnd.uniform(0, 2 * math.pi)
        rpm = W.RPM * rnd.uniform(0.95, 1.05)
        r, ro = W.place_turbine(parts, (x, y, 0.0), yaw=0.0, phase=phase, rpm=rpm, f0=F0, f1=F1, name=name)
        turbs.append((i, j, r, ro))

# ---- ocean: Ocean modifier (displacement + foam) over a large patch, far plane beyond
ocm, tval, wake_nodes = W.ocean_material("Sea", wake=dict(dir=(0.0, 1.0)), piles=(SX, SY, 0.0, 0.0), haze_dist=11000.0)
bpy.ops.mesh.primitive_plane_add(location=(-3000, -1000, 0))     # tile origin; repeats extend +X/+Y
sea = bpy.context.active_object; sea.name = "Sea"
om = sea.modifiers.new("Ocean", 'OCEAN')
om.geometry_mode = 'GENERATE'
om.spatial_size = int(os.environ.get("OCSIZE", "3000"))
om.resolution = int(os.environ.get("OCRES", "32"))
om.viewport_resolution = 4
om.repeat_x = 3; om.repeat_y = 3
om.wind_velocity = 9.0
om.wave_scale = 1.4
om.choppiness = 1.1
om.wave_alignment = 0.35
om.wave_direction = math.radians(-90)      # toward -Y (downwind)
om.use_normals = True
om.use_foam = True
om.foam_layer_name = "foam"
om.foam_coverage = float(os.environ.get("FOAMCOV", "0.35"))
om.random_seed = 7
sea.data.materials.append(ocm)
for f in (F0 - 2, F1 + 2):
    om.time = f / 24.0
    om.keyframe_insert("time", frame=f)
    tval.outputs[0].default_value = f / 24.0
    tval.outputs[0].keyframe_insert("default_value", frame=f)
for dat in (sea.animation_data, ocm.node_tree.animation_data):
    try:
        for fc in dat.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
    except Exception:
        pass
far = sb.prim("plane", "FarSea", loc=(0, 0, -1.2), scale=(400000, 400000, 1))
far.data.materials.append(ocm)
# the far plane must not show through inside the patch: cut a hole the size of the patch
bpy.context.view_layer.update()

# ---- crew transfer vessel heading upwind (+Y) toward the hero row, wake behind
boat = W.ctv(M)
BOAT0 = V((260.0, -240.0, 0.0))
VB = 10.5                                  # m/s (~20 kn)


def boat_pos(f):
    return BOAT0 + V((0, 1, 0)) * VB * (f - 1024) / 24.0


bx, by = wake_nodes
for f in range(F0 - 10, F1 + 11):
    p = boat_pos(f)
    boat.location = p + V((0, 0, 0.15 * math.sin(f * 0.21)))
    boat.rotation_euler = (math.radians(1.2 * math.sin(f * 0.17)), math.radians(1.5 * math.sin(f * 0.13 + 1)), 0)
    boat.keyframe_insert("location", frame=f); boat.keyframe_insert("rotation_euler", frame=f)
    bx.outputs[0].default_value = p.x; bx.outputs[0].keyframe_insert("default_value", frame=f)
    by.outputs[0].default_value = p.y; by.outputs[0].keyframe_insert("default_value", frame=f)


# =========================================================================== cameras
cam = E.dof_cam("Cam", 35, 8.0, clip=(0.5, 500000))
sc.eevee.bokeh_max_size = 100
HUBC = V((0.0, 0.9, W.HUB_H + 0.2))         # rotor centre (T0_0)
SPIN_PLANE_Y = HUBC.y - 0.4


def cam_bake_up(pos_fn, tgt_fn, lens_fn=None, focus_fn=None, roll_fn=None):
    for f in range(F0 - 10, F1 + 11):
        t = (f - F0) / max(1, F1 - F0)
        p = V(pos_fn(t)); tg = V(tgt_fn(t))
        cam.location = p
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = sb.look_quat(p, tg, roll_fn(t) if roll_fn else 0.0)
        cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_quaternion", frame=f)
        if lens_fn:
            cam.data.lens = lens_fn(t); cam.data.keyframe_insert("lens", frame=f)
        if focus_fn:
            cam.data.dof.focus_distance = focus_fn(t, p, tg); cam.data.dof.keyframe_insert("focus_distance", frame=f)


if sid == "s12a":
    # frame 0: spinner centred, silhouette radius = 0.41 * frame height (matches the plasma ring of s11's last frame)
    LENS = 35.0
    Hw = W.SPIN_R / 0.41                           # world height of the frame at the spinner plane
    d0 = Hw * LENS / 20.25
    P0 = V((0.0, SPIN_PLANE_Y + d0, HUBC.z))
    P1 = V((-16.0, SPIN_PLANE_Y + 34.0, HUBC.z + 6.0))
    P2 = V((-70.0, 150.0, HUBC.z + 26.0))
    T0 = V((0.0, SPIN_PLANE_Y, HUBC.z))
    T2 = V((6.0, -14.0, HUBC.z - 22.0))

    def pos(t):
        k = sb.ease_in(t, 1.7) * 0.8 + t * 0.2
        return sb.catmull([P0, P1, P2], k)

    def tgt(t):
        k = sb.smooth(sb.remap(t, 0.05, 1.0))
        return T0.lerp(T2, k)

    cam_bake_up(pos, tgt, focus_fn=lambda t, p, tg: (HUBC - p).length)
elif sid == "s12b":
    # helicopter: high over the sea on the flank of the farm, moving forward along the rows, easing into a bank
    # open on the vessel and its wake (tracking alongside, low), then crane up and tilt to the farm on the horizon
    def pos(t):
        k = sb.smoother(t) * 0.7 + t * 0.3
        b = boat_pos(F0 + t * (F1 - F0))
        return b + V((-95.0, -150.0, 0.0)).lerp(V((-135.0, -185.0, 0.0)), k) + V((0, 0, sb.lerp(48.0, 96.0, k)))

    def tgt(t):
        k = sb.smooth(sb.remap(t, 0.0, 1.0)) * 0.8
        b = boat_pos(F0 + t * (F1 - F0))
        return (b + V((0, 25.0, 0))).lerp(V((-120.0, 2600.0, 0.0)), k)

    cam_bake_up(pos, tgt, lens_fn=lambda t: 32.0, focus_fn=lambda t, p, tg: 600.0,
                roll_fn=lambda t: math.radians(-2.5 * sb.smooth(t)))
    cam.data.dof.aperture_fstop = 16.0

sb.frames(F0, F1)
sb.render_shot(sid)
