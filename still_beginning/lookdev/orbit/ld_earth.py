"""Earth lookdev: blender -b -P lookdev/orbit/ld_earth.py -- <view> [key=val ...]"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lib"))
import bpy, sb, earth
from mathutils import Vector as V

args = sb.argv()
view = args[0] if args else "orbit_day"
kv = dict(a.split("=") for a in args[1:] if "=" in a)
sc = sb.reset()
sb.setup_render("EEVEE", samples=int(kv.get("samples", 16)), mblur=False, look=kv.get("look", "AgX - Medium High Contrast"),
                exposure=float(kv.get("expo", 0.0)))
if "res" in kv:
    w, h = kv["res"].split("x"); sc.render.resolution_x, sc.render.resolution_y = int(w), int(h)

presets = {
    # alt km, look azimuth, look pitch (deg, relative to local horizontal), sun elev, sun azim, lens
    "orbit_day":     (400, 0, -24, 35, 140, 24),
    "orbit_limb":    (400, 0, -17, 25, 100, 35),
    "orbit_sunrise": (400, 0, -17, -21.5, 8, 35),
    "orbit_term":    (400, 0, -30, -10, 95, 24),
    "orbit_glint":   (400, 0, -30, 25, 0, 24),
    "alt40":         (40, 0, -4, 25, 120, 24),
    "alt40_low":     (40, 0, -4, 6, 60, 24),
    "nadir":         (400, 0, -70, 45, 130, 24),
    "far":           (30000, 0, -90, 10, 90, 50),
}
alt, az, pitch, se, sa, lens = presets[view]
alt = float(kv.get("alt", alt)); pitch = float(kv.get("pitch", pitch)); se = float(kv.get("se", se)); sa = float(kv.get("sa", sa))
lens = float(kv.get("lens", lens)); az = float(kv.get("az", az))
sd = sb.sun_dir(se, sa)
t0 = time.time()
E = earth.build(alt_km=alt, sun_dir=tuple(sd), clouds=float(kv.get("clouds", 0.4)), samples=int(kv.get("N", 16)),
                lights=float(kv.get("lights", 0.0)), seed=float(kv.get("seed", 0.0)),
                nadir=tuple(float(x) for x in kv.get("nadir", "30,-20,0").split(",")), ms=float(kv.get("ms", 1.05)), debug=kv.get("dbg"))
if kv.get("dbg"):
    sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'
print("build", time.time() - t0)
E.sun_lamp()
d = sb.sun_dir(pitch, az)
cam = sb.camera(loc=(0, 0, 0), target=tuple(d * 10), lens=lens, clip=(0.1, 1000))
# a small white reference sphere + gold foil sample near camera to check lighting integration
if kv.get("probe", "1") == "1":
    m = sb.painted("White", (0.8, 0.8, 0.8), rough=0.35, coat=0.2)
    s = sb.prim("sphere", "Ref", loc=tuple(d * 6 + V((1.6, 0, -0.9)) * (lens / 24) ** -1), radius=0.25, mat=m)
    g = sb.mat("Gold", (1.0, 0.72, 0.3), metal=1.0, rough=0.12)
    s2 = sb.prim("sphere", "Ref2", loc=tuple(d * 6 + V((2.3, 0, -0.9)) * (lens / 24) ** -1), radius=0.25, mat=g)
sb.frames(0, 1)
if kv.get("layered") == "1":
    import orbit
    t1 = time.time()
    orbit.layered(E, cam, earth_samples=int(kv.get("es", 12)), env_res=int(kv.get("envres", 4096)))
    print("LAYERED_SETUP", time.time() - t1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), kv.get("out", f"earth_{view}.png"))
sc.render.filepath = out
t0 = time.time()
for i in range(int(kv.get("nrep", 1))):
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print("RENDER_TIME", time.time() - t0)
