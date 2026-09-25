"""Workstation display content for s06 (global frames 450-539): ONE engineering viewport of the component as
the AI-assisted design system refines it (conventional block -> lattice-infilled -> final generative link), rendered
per frame to assets/screen_s06/vp_NNNN.png, then composited with a minimal, text-free UI by lib/screenui.py.
blender -b -P scenes/s06_screen.py -- [thumbs]   (SCREEN_W env for resolution; default 2048)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, prosthesis as P
from mathutils import Vector as V

mode = (sb.argv() or ["frames"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=False, look="AgX - Base Contrast")
W = int(os.environ.get("SCREEN_W", "2048"))
sc.render.resolution_x, sc.render.resolution_y = W, int(W * 0.5625)
sc.eevee.taa_render_samples = 32
sc.view_settings.look = "AgX - Base Contrast"
OUT = os.path.join(sb.ROOT, "assets", "screen_s06")
os.makedirs(OUT, exist_ok=True)

# CAD viewport look: light neutral gradient backdrop, soft contact shadow, faint ground grid
sb.world_gradient(top=(0.018, 0.020, 0.024), horizon=(0.055, 0.058, 0.064), bottom=(0.03, 0.032, 0.036), strength=1.0)
ground = sb.prim("plane", "Ground", loc=(0, 0.02, -0.0076), scale=(0.2, 0.2, 1))
gm = sb.mat("GroundM", (0.045, 0.047, 0.052), rough=0.7, spec=0.2)
nb = sb.NB(gm)
co = nb.coord('Object')
gx = nb.wave(co, scale=40, wtype='BANDS', direction='X', profile='SAW')
gy = nb.wave(co, scale=40, wtype='BANDS', direction='Y', profile='SAW')
lines = nb.math('MAXIMUM', nb.maprange(gx.outputs['Fac'], 0.97, 1.0), nb.maprange(gy.outputs['Fac'], 0.97, 1.0))
fade = nb.maprange(nb.vmath('LENGTH', co), 0.25, 0.9, 1.0, 0.0)
nb.set('Base Color', nb.mix(nb.math('MULTIPLY', lines, nb.math('MULTIPLY', fade, 0.6)), (0.045, 0.047, 0.052, 1), (0.16, 0.17, 0.19, 1)))
ground.data.materials.append(gm)
sb.light('AREA', "Key", loc=(-0.15, -0.12, 0.25), target=(0, 0.02, 0), energy=4.0, color=(1, 0.98, 0.95), size=0.3)
sb.light('AREA', "Rim", loc=(0.2, 0.25, 0.12), target=(0, 0.02, 0), energy=5.0, color=(0.9, 0.93, 1.0), size=0.12)
sb.light('AREA', "Fill", loc=(0.2, -0.2, 0.05), target=(0, 0.02, 0), energy=0.8, size=0.4)

# clay material for design iterations: light warm grey, soft, with a subtle cavity/edge darkening (AO look)
clay = sb.mat("Clay", (0.70, 0.69, 0.67), rough=0.55, spec=0.35)
cnb = sb.NB(clay)
ao = cnb.new('ShaderNodeAmbientOcclusion'); ao.inputs['Distance'].default_value = 0.002
cnb.set('Base Color', cnb.mix(cnb.maprange(ao.outputs['AO'], 0.3, 1.0), (0.30, 0.30, 0.30, 1), (0.72, 0.71, 0.69, 1)))
# final: the real titanium + amber
ti = P.ti_sintered()

pivot = sb.empty("Pivot", loc=(0, 0.0, 0))
pivot.rotation_mode = 'XYZ'
morphs = sorted(os.listdir(os.path.join(sb.ROOT, "assets", "part", "morph")))
mesh_objs = []
for i, fn in enumerate(morphs):
    o = P.load_npz(os.path.join(sb.ROOT, "assets", "part", "morph", fn), "M%02d" % i, clay)
    o.parent = pivot
    o.location = (0, -0.0195, 0)
    o.hide_render = True
    mesh_objs.append(o)
final_root, final_mesh = P.component("v3", "Final", parent=pivot)
final_root.location = (0, -0.0195, 0)
for ch in [final_mesh] + list(final_root.children):
    ch.hide_render = True
finals = [final_mesh] + [c for c in final_root.children if c is not final_mesh]


def mesh_index(f):
    """frame -> morph index (0..40) or -1 for the final hero part."""
    if f < 466:
        return 0
    if f < 490:
        return int(round(sb.smooth((f - 466) / 24.0) * 20))
    if f < 512:
        return 20 + int(round(sb.smooth((f - 490) / 22.0) * 20))
    return -1


cam = sb.camera("VP", loc=(0.06, -0.075, 0.05), target=(0, 0.0, 0.0), lens=55, clip=(0.005, 5))


def set_frame(f):
    idx = mesh_index(f)
    for i, o in enumerate(mesh_objs):
        o.hide_render = (i != idx)
    for o in finals:
        o.hide_render = (idx != -1)
    # slow turntable + an eased turn at the final choice to reveal the amber bushing faces
    # turntable that eases to rest on the chosen design (3/4 view, amber bushing face toward the camera)
    a = math.radians(-40 + 34 * sb.ease_out((f - 450) / 86.0, 1.6))
    pivot.rotation_euler = (0, 0, a)
    t = (f - 450) / 90.0
    d = sb.lerp(1.0, 0.86, sb.smooth(t))
    cam.location = V((0.058, -0.074, 0.046)) * d
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = sb.look_quat(cam.location, (0, 0.0, -0.001))


if mode == "thumbs":
    sc.render.resolution_x, sc.render.resolution_y = 640, 400
    for name, f in (("thumb1", 452), ("thumb2", 489), ("thumb3", 530)):
        set_frame(f)
        pivot.rotation_euler = (0, 0, math.radians(-20))
        sc.render.filepath = os.path.join(OUT, name + ".png")
        bpy.ops.render.render(write_still=True)
else:
    fr = os.environ.get("SCREEN_FRAMES")
    rng = range(450, 540) if not fr else [int(x) for x in fr.split(",")]
    for f in rng:
        sc.frame_set(f)
        set_frame(f)
        sc.render.filepath = os.path.join(OUT, "vp_%04d.png" % f)
        bpy.ops.render.render(write_still=True)
