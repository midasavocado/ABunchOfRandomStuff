"""Equirect world map of Earth procedural layers (Cycles): blender -b -P ld_map.py -- dbg1,dbg2 [seed=..]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lib"))
import bpy, sb, earth
args = sb.argv()
names = args[0].split(",")
kv = dict(a.split("=") for a in args[1:] if "=" in a)
here = os.path.dirname(os.path.abspath(__file__))
for nm in names:
    sc = sb.reset()
    sb.setup_render("CYCLES", cycles_samples=4, mblur=False)
    sc.cycles.samples = 1; sc.cycles.use_denoising = False
    sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'
    sc.render.resolution_x, sc.render.resolution_y = int(kv.get("w", 2048)), int(kv.get("w", 2048)) // 2
    E = earth.build(alt_km=400, sun_dir=sb.sun_dir(30, 90), samples=2, debug=nm, map_mode=True,
                    seed=float(kv.get("seed", 0)), clouds=float(kv.get("clouds", 0.5)))
    cam = sb.camera(loc=(0, 0, 0), target=(0, 1, 0))
    cam.rotation_mode = 'XYZ'; cam.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
    cam.data.type = 'PANO'
    cam.data.panorama_type = 'EQUIRECTANGULAR'
    sc.render.image_settings.file_format = 'OPEN_EXR'; sc.render.image_settings.color_depth = '32'
    sc.render.filepath = os.path.join(here, "map_%s.exr" % nm)
    bpy.ops.render.render(write_still=True)
