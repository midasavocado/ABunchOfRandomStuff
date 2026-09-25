"""STILL BEGINNING - shared Blender helpers (Blender 5.2)."""
import bpy, bmesh, math, os, sys, random
from mathutils import Vector, Matrix, Euler, Quaternion, noise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))
import timeline as TL

RES = os.environ.get("SB_RES", "preview")      # preview | final | half
V = Vector

# --------------------------------------------------------------------------- args

def argv():
    a = sys.argv
    return a[a.index("--") + 1:] if "--" in a else []

# --------------------------------------------------------------------------- scene

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    prefs = bpy.context.preferences
    prefs.edit.keyframe_new_interpolation_type = 'LINEAR'
    sc = bpy.context.scene
    sc.render.fps = 24
    sc.render.fps_base = 1.0
    return sc


def setup_render(engine="EEVEE", samples=None, mblur=True, shutter=0.5,
                 look="AgX - Medium High Contrast", exposure=0.0, gamma=1.0,
                 volumes=False, vol_tile=8, vol_samples=64, raytrace=True,
                 transparent=False, cycles_samples=96):
    sc = bpy.context.scene
    r = sc.render
    if RES == "final":
        r.resolution_x, r.resolution_y = 3840, 2160
    elif RES == "half":
        r.resolution_x, r.resolution_y = 1920, 1080
    else:
        r.resolution_x, r.resolution_y = 960, 540
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1
    r.film_transparent = transparent
    r.use_motion_blur = mblur
    r.motion_blur_shutter = shutter
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.color_depth = '16' if RES == "final" else '8'
    r.image_settings.compression = 15
    r.use_persistent_data = True
    b = os.environ.get("SB_BORDER")
    if b:
        x0, y0, x1, y1 = [float(t) for t in b.split(",")]
        r.use_border = True
        r.use_crop_to_border = True
        r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = x0, y0, x1, y1
    engine = os.environ.get("SB_ENGINE", engine)
    sc.view_settings.view_transform = 'AgX'
    try:
        sc.view_settings.look = look
    except Exception:
        pass
    sc.view_settings.exposure = exposure
    sc.view_settings.gamma = gamma
    sc.display_settings.display_device = 'sRGB'
    if engine == "CYCLES":
        r.engine = 'CYCLES'
        p = bpy.context.preferences.addons['cycles'].preferences
        # Metal GPU on the Mac; fall back to CPU (e.g. cloud Linux boxes) when no GPU exists.
        gpu = False
        for kind in ('METAL', 'OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
            try:
                p.compute_device_type = kind
            except TypeError:
                continue
            p.get_devices()
            devs = [d for d in p.devices if d.type == kind]
            if devs:
                for d in p.devices:
                    d.use = (d.type == kind)
                gpu = True
                break
        if not gpu:
            p.compute_device_type = 'NONE'
        sc.cycles.device = 'GPU' if gpu else 'CPU'
        sc.cycles.samples = cycles_samples if RES == "final" else max(16, cycles_samples // 4)
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.max_bounces = 8
        sc.cycles.transmission_bounces = 8
        sc.cycles.glossy_bounces = 4
        sc.cycles.volume_bounces = 1
        sc.cycles.caustics_reflective = False
        sc.cycles.caustics_refractive = False
        sc.cycles.blur_glossy = 1.0
        sc.cycles.sample_clamp_indirect = 8.0
    else:
        r.engine = 'BLENDER_EEVEE'
        e = sc.eevee
        if samples is None:
            samples = 64 if RES == "final" else 16
        e.taa_render_samples = samples
        e.use_shadows = True
        e.shadow_ray_count = 2
        e.shadow_step_count = 8
        e.use_raytracing = raytrace
        e.ray_tracing_method = 'SCREEN'
        try:
            e.ray_tracing_options.resolution_scale = '1' if RES == "final" else '2'
            e.ray_tracing_options.use_denoise = True
        except Exception:
            pass
        e.use_fast_gi = True
        e.fast_gi_method = 'GLOBAL_ILLUMINATION'
        e.fast_gi_resolution = '2'
        e.fast_gi_ray_count = 4
        e.fast_gi_distance = 0.0
        e.volumetric_tile_size = str(vol_tile)
        e.volumetric_samples = vol_samples
        e.use_volumetric_shadows = volumes
        e.volumetric_shadow_samples = 16
        e.motion_blur_steps = 1
        e.use_overscan = True
        e.overscan_size = 3.0
        e.shadow_resolution_scale = 1.0
    return sc


def frames(start, end):
    sc = bpy.context.scene
    sc.frame_start = start
    sc.frame_end = end - 1
    sc.frame_set(start)


def out_path(sid):
    d = os.path.join(ROOT, "renders" if RES == "final" else ("half" if RES == "half" else "preview"), sid)
    os.makedirs(d, exist_ok=True)
    return d


def lookdev_overrides(sc):
    """Fast-iteration switches for CPU lookdev (never set for finals):
    SB_NOVOL=1 strips volume shaders (world + objects); SB_SAMPLES=n overrides Cycles samples."""
    if os.environ.get("SB_NOVOL") == "1":
        mats = list(bpy.data.materials) + ([sc.world] if sc.world else [])
        for m in mats:
            if not m or not m.use_nodes:
                continue
            for n in m.node_tree.nodes:
                if n.type in ('OUTPUT_MATERIAL', 'OUTPUT_WORLD'):
                    for l in list(n.inputs['Volume'].links):
                        m.node_tree.links.remove(l)
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.data.materials and all(
                    m and m.use_nodes and not any(n.type in ('BSDF_PRINCIPLED', 'EMISSION', 'BSDF_DIFFUSE', 'BSDF_TRANSPARENT', 'MIX_SHADER',
                                                             'BSDF_GLOSSY', 'BSDF_METALLIC', 'BSDF_TRANSLUCENT', 'HOLDOUT', 'BSDF_REFRACTION', 'BSDF_GLASS')
                                                  for n in m.node_tree.nodes) for m in o.data.materials):
                o.hide_render = True          # pure volume containers
    if os.environ.get("SB_SAMPLES") and sc.render.engine == 'CYCLES':
        sc.cycles.samples = int(os.environ["SB_SAMPLES"])


def render_shot(sid, start=None, end=None, still=None):
    """Render the frame range of shot `sid` to renders/<sid>/NNNN.png (global frame numbers)."""
    sc = bpy.context.scene
    lookdev_overrides(sc)
    _, s0, s1, _ = TL.shot(sid)
    s0 = s0 if start is None else start
    s1 = s1 if end is None else end
    d = out_path(sid)
    env = os.environ.get("SB_FRAMES")
    if env:            # e.g. "first,mid,last" or "0:10"
        pick = []
        for tok in env.split(","):
            if tok == "first": pick.append(s0)
            elif tok == "mid": pick.append((s0 + s1) // 2)
            elif tok == "last": pick.append(s1 - 1)
            elif ":" in tok:
                a, b = tok.split(":"); pick += list(range(s0 + int(a), s0 + int(b)))
            else: pick.append(s0 + int(tok))
        for f in pick:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(d, "%04d.png" % f)
            bpy.ops.render.render(write_still=True)
        return
    skip = os.environ.get("SB_SKIP_EXISTING", "1") == "1"
    for f in range(s0, s1):
        fp = os.path.join(d, "%04d.png" % f)
        if skip and os.path.exists(fp) and os.path.getsize(fp) > 0:
            continue
        sc.frame_set(f)
        sc.render.filepath = fp
        bpy.ops.render.render(write_still=True)


# --------------------------------------------------------------------------- math / anim

def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def smoother(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out(t, p=2.0):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** p


def ease_in(t, p=2.0):
    t = max(0.0, min(1.0, t))
    return t ** p


def lerp(a, b, t):
    return a + (b - a) * t


def vlerp(a, b, t):
    return V(a).lerp(V(b), t)


def remap(x, a, b):
    return max(0.0, min(1.0, (x - a) / (b - a))) if b != a else 0.0


def look_quat(loc, target, roll=0.0):
    d = V(target) - V(loc)
    q = d.to_track_quat('-Z', 'Y')
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q


def bake(obj, f0, f1, fn):
    """fn(frame) -> dict(loc=, rot=(euler or quat), scale=). Inserts per-frame keys (linear)."""
    for f in range(f0, f1 + 1):
        d = fn(f)
        if "loc" in d:
            obj.location = d["loc"]
            obj.keyframe_insert("location", frame=f)
        if "quat" in d:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = d["quat"]
            obj.keyframe_insert("rotation_quaternion", frame=f)
        if "rot" in d:
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = d["rot"]
            obj.keyframe_insert("rotation_euler", frame=f)
        if "scale" in d:
            s = d["scale"]
            obj.scale = (s, s, s) if isinstance(s, (int, float)) else s
            obj.keyframe_insert("scale", frame=f)


def bake_prop(idblock, path, f0, f1, fn, index=-1):
    for f in range(f0, f1 + 1):
        v = fn(f)
        exec_set(idblock, path, v)
        idblock.keyframe_insert(path, frame=f, index=index)


def exec_set(idblock, path, v):
    # supports "a.b.c" and 'nodes["X"].inputs[1].default_value'
    obj = idblock
    parts = path.rsplit(".", 1)
    if len(parts) == 2:
        obj = idblock.path_resolve(parts[0])
        setattr(obj, parts[1], v)
    else:
        setattr(idblock, path, v)


def bake_socket(ntree_owner, socket, f0, f1, fn):
    """Animate a node socket default_value per frame. ntree_owner = material/world."""
    for f in range(f0, f1 + 1):
        socket.default_value = fn(f)
        socket.keyframe_insert("default_value", frame=f)


# --------------------------------------------------------------------------- objects

def link_obj(o, coll=None):
    (coll or bpy.context.scene.collection).objects.link(o)
    return o


def mesh_obj(name, verts, faces, mat=None, smooth_shade=True, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    o = bpy.data.objects.new(name, me)
    link_obj(o, coll)
    if smooth_shade:
        for p in me.polygons:
            p.use_smooth = True
    if mat:
        me.materials.append(mat)
    return o


def recalc_normals(o, inside=False, weld=False):
    bm = bmesh.new()
    bm.from_mesh(o.data)
    if weld:     # e.g. lathe profiles that touch the axis: fuse the pole vertices so the solid is manifold
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        bmesh.ops.dissolve_degenerate(bm, dist=1e-8, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if inside:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(o.data)
    bm.free()


def empty(name="E", loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, None)
    o.location = loc
    link_obj(o)
    if parent:
        o.parent = parent
    return o


def prim(kind, name=None, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), mat=None,
         smooth_shade=True, parent=None, **kw):
    ops = {
        "cube": bpy.ops.mesh.primitive_cube_add,
        "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
        "ico": bpy.ops.mesh.primitive_ico_sphere_add,
        "cyl": bpy.ops.mesh.primitive_cylinder_add,
        "cone": bpy.ops.mesh.primitive_cone_add,
        "torus": bpy.ops.mesh.primitive_torus_add,
        "plane": bpy.ops.mesh.primitive_plane_add,
        "circle": bpy.ops.mesh.primitive_circle_add,
        "grid": bpy.ops.mesh.primitive_grid_add,
    }
    ops[kind](location=loc, rotation=rot, **kw)
    o = bpy.context.active_object
    o.scale = scale
    if name:
        o.name = name
    if smooth_shade and kind not in ("cube", "plane", "grid"):
        for p in o.data.polygons:
            p.use_smooth = True
    if mat:
        o.data.materials.append(mat)
    if parent:
        o.parent = parent
    return o


def bevel(o, width=0.01, segments=3, limit='ANGLE', angle=30):
    m = o.modifiers.new("Bevel", 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = limit
    if limit == 'ANGLE':
        m.angle_limit = math.radians(angle)
    m.harden_normals = False
    return m


def subsurf(o, levels=2, render=None):
    m = o.modifiers.new("Subsurf", 'SUBSURF')
    m.levels = levels
    m.render_levels = render if render is not None else levels
    return m


def solidify(o, t=0.01, offset=-1):
    m = o.modifiers.new("Solid", 'SOLIDIFY')
    m.thickness = t
    m.offset = offset
    return m


def array(o, count, offset=(1, 0, 0), relative=True):
    m = o.modifiers.new("Array", 'ARRAY')
    m.count = count
    if relative:
        m.relative_offset_displace = offset
    else:
        m.use_relative_offset = False
        m.use_constant_offset = True
        m.constant_offset_displace = offset
    return m


def apply_mods(o):
    bpy.context.view_layer.objects.active = o
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def shade_auto(o, angle=35):
    try:
        with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o], selected_editable_objects=[o]):
            bpy.ops.object.shade_auto_smooth(angle=math.radians(angle))
    except Exception:
        for p in o.data.polygons:
            p.use_smooth = True


def lathe(name, profile, segs=64, mat=None, axis='Z', close=False, smooth_shade=True, auto=40):
    """profile: list of (r, z). Returns a revolved mesh object. Smooth shading is split at profile corners sharper
    than `auto` degrees (otherwise interpolated normals across e.g. a glass rim flip refraction and look black)."""
    verts, faces = [], []
    n = len(profile)
    for i in range(segs):
        a = 2 * math.pi * i / segs
        c, s = math.cos(a), math.sin(a)
        for r, z in profile:
            verts.append((r * c, r * s, z))
    for i in range(segs):
        j = (i + 1) % segs
        for k in range(n - 1):
            faces.append((i * n + k, j * n + k, j * n + k + 1, i * n + k + 1))
    o = mesh_obj(name, verts, faces, mat, smooth_shade)
    recalc_normals(o, weld=True)
    if smooth_shade and auto:
        shade_auto(o, auto)
    if axis == 'X':
        o.rotation_euler = (0, math.pi / 2, 0)
    elif axis == 'Y':
        o.rotation_euler = (math.pi / 2, 0, 0)
    return o


def tube_along(name, pts, radius=0.01, mat=None, bevel_res=6, taper=None, closed=False):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = bevel_res
    cu.use_fill_caps = True
    sp = cu.splines.new('POLY' if len(pts) > 3 else 'POLY')
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1)
        if taper:
            sp.points[i].radius = taper(i / max(1, len(pts) - 1))
    sp.use_cyclic_u = closed
    o = bpy.data.objects.new(name, cu)
    link_obj(o)
    if mat:
        cu.materials.append(mat)
    return o


def bezier_curve(name, pts, radius=0.0, mat=None, closed=False, res=24, handle='AUTO', extrude=0.0):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = res
    cu.extrude = extrude
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        bp = sp.bezier_points[i]
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = handle
    sp.use_cyclic_u = closed
    o = bpy.data.objects.new(name, cu)
    link_obj(o)
    if mat:
        cu.materials.append(mat)
    return o


def to_mesh(o):
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.selected_objects:
        x.select_set(False)
    o.select_set(True)
    bpy.ops.object.convert(target='MESH')
    return bpy.context.active_object


def join(objs, name=None):
    for x in bpy.context.selected_objects:
        x.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.active_object
    if name:
        o.name = name
    return o


def parent_keep(child, parent):
    mw = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    return child


def instance(src, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, parent=None, name=None):
    o = bpy.data.objects.new(name or (src.name + "_i"), src.data)
    o.location = loc
    o.rotation_euler = rot
    o.scale = (scale,) * 3 if isinstance(scale, (int, float)) else scale
    for m in src.modifiers:
        pass
    link_obj(o)
    if parent:
        o.parent = parent
    return o


def dup(src, loc=None, rot=None, scale=None, parent=None, linked=True):
    o = src.copy()
    if not linked and src.data:
        o.data = src.data.copy()
    link_obj(o)
    if loc is not None: o.location = loc
    if rot is not None: o.rotation_euler = rot
    if scale is not None: o.scale = (scale,) * 3 if isinstance(scale, (int, float)) else scale
    if parent is not None: o.parent = parent
    return o


def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def collection_instance(coll, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, name=None):
    o = bpy.data.objects.new(name or coll.name + "_inst", None)
    o.instance_type = 'COLLECTION'
    o.instance_collection = coll
    o.location = loc
    o.rotation_euler = rot
    o.scale = (scale,) * 3 if isinstance(scale, (int, float)) else scale
    link_obj(o)
    return o


def hide_coll_from_render(coll):
    lc = bpy.context.view_layer.layer_collection.children.get(coll.name)
    if lc:
        lc.exclude = True

# --------------------------------------------------------------------------- camera

def camera(name="Cam", loc=(0, -5, 1), target=(0, 0, 0), lens=50, fstop=None, focus=None,
           sensor=36.0, clip=(0.01, 1000)):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.sensor_fit = 'HORIZONTAL'
    cd.clip_start, cd.clip_end = clip
    o = bpy.data.objects.new(name, cd)
    link_obj(o)
    o.location = loc
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = look_quat(loc, target)
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.aperture_blades = 7
        cd.dof.aperture_rotation = math.radians(10)
        cd.dof.focus_distance = focus if focus else (V(target) - V(loc)).length
    bpy.context.scene.camera = o
    return o


def cam_bake(cam, f0, f1, pos, target, lens=None, focus=None, roll=None, fstop=None):
    """pos/target/lens/focus: callables of normalized t in [0,1] (t=(f-f0)/(f1-f0)) or constants."""
    def ev(x, t, f):
        return x(t) if callable(x) else x
    for f in range(f0, f1 + 1):
        t = (f - f0) / max(1, (f1 - f0))
        p = V(ev(pos, t, f))
        tg = V(ev(target, t, f))
        rl = ev(roll, t, f) if roll is not None else 0.0
        cam.location = p
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = look_quat(p, tg, rl)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        if lens is not None:
            cam.data.lens = ev(lens, t, f)
            cam.data.keyframe_insert("lens", frame=f)
        if focus is not None:
            fd = ev(focus, t, f)
            if fd == "target":
                fd = (tg - p).length
            cam.data.dof.focus_distance = fd
            cam.data.dof.keyframe_insert("focus_distance", frame=f)
        if fstop is not None:
            cam.data.dof.aperture_fstop = ev(fstop, t, f)
            cam.data.dof.keyframe_insert("aperture_fstop", frame=f)


def catmull(points, t):
    """Catmull-Rom through list of Vectors, t in [0,1]."""
    pts = [V(p) for p in points]
    n = len(pts) - 1
    x = max(0.0, min(1.0, t)) * n
    i = min(int(x), n - 1)
    u = x - i
    p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)

# --------------------------------------------------------------------------- lights

def light(kind, name="L", loc=(0, 0, 5), target=None, energy=100, color=(1, 1, 1), size=1.0,
          size_y=None, angle=None, spot=None, blend=0.3, shadow_soft=None, rot=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == 'AREA':
        ld.size = size
        if size_y:
            ld.shape = 'RECTANGLE'
            ld.size_y = size_y
    elif kind in ('POINT', 'SPOT'):
        ld.shadow_soft_size = size
        if kind == 'SPOT':
            ld.spot_size = math.radians(spot or 45)
            ld.spot_blend = blend
    elif kind == 'SUN':
        ld.angle = math.radians(angle if angle is not None else 0.5)
    o = bpy.data.objects.new(name, ld)
    link_obj(o)
    o.location = loc
    if target is not None:
        o.rotation_mode = 'QUATERNION'
        o.rotation_quaternion = look_quat(loc, target)
    elif rot is not None:
        o.rotation_euler = rot
    return o


def sun_dir(elev_deg, azim_deg):
    """Direction TO the sun (unit) from elevation/azimuth (azimuth 0 = +Y, 90 = +X)."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return V((math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)))


def sun(elev, azim, energy=4.0, color=(1, 0.95, 0.88), angle=0.53, name="Sun"):
    d = sun_dir(elev, azim)
    o = light('SUN', name, loc=d * 10, target=(0, 0, 0), energy=energy, color=color, angle=angle)
    return o

# --------------------------------------------------------------------------- world

def world_sky(elev=10, azim=0, strength=1.0, sky_type='MULTIPLE_SCATTERING', altitude=0.0,
              air=1.0, aerosol=1.0, ozone=1.0, sun_disc=True, sun_intensity=1.0, tint=None, sun_size=0.545):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("Sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new('ShaderNodeTexSky')
    sky.sky_type = sky_type
    try:
        sky.sun_elevation = math.radians(elev)
        sky.sun_rotation = math.radians(azim)
        sky.sun_disc = sun_disc
        sky.sun_size = math.radians(sun_size)
        sky.sun_intensity = sun_intensity
        sky.altitude = altitude
        sky.air_density = air
        sky.aerosol_density = aerosol
        sky.ozone_density = ozone
    except Exception as ex:
        for k, v in dict(altitude=altitude, air_density=air, dust_density=aerosol, ozone_density=ozone).items():
            try: setattr(sky, k, v)
            except Exception: pass
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs[1].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    if tint:
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = 1.0
        nt.links.new(sky.outputs[0], mix.inputs[6])
        mix.inputs[7].default_value = (*tint, 1)
        nt.links.new(mix.outputs[2], bg.inputs[0])
    else:
        nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w, sky, bg


def world_color(color=(0.05, 0.06, 0.08), strength=1.0):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*color, 1)
    bg.inputs[1].default_value = strength
    return w, bg


def world_gradient(top=(0.02, 0.03, 0.06), horizon=(0.4, 0.35, 0.3), bottom=(0.05, 0.05, 0.05),
                   strength=1.0, sharp=2.0, warm_dir=None, warm_col=(1.0, 0.55, 0.2), warm_amt=0.0):
    """Studio/sky-like gradient env with optional warm glow toward a horizontal direction."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new("Grad")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['Generated'], sep.inputs[0])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    mapr = nt.nodes.new('ShaderNodeMapRange')
    mapr.inputs[1].default_value = -1; mapr.inputs[2].default_value = 1
    nt.links.new(sep.outputs[2], mapr.inputs[0])
    nt.links.new(mapr.outputs[0], ramp.inputs[0])
    cr = ramp.color_ramp
    cr.elements[0].position = 0.35; cr.elements[0].color = (*bottom, 1)
    cr.elements[1].position = 0.5; cr.elements[1].color = (*horizon, 1)
    e = cr.elements.new(0.5 + 0.5 / sharp); e.color = (*top, 1)
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs[1].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    col = ramp.outputs[0]
    if warm_dir is not None and warm_amt > 0:
        dp = nt.nodes.new('ShaderNodeVectorMath'); dp.operation = 'DOT_PRODUCT'
        nt.links.new(tc.outputs['Generated'], dp.inputs[0])
        dp.inputs[1].default_value = V(warm_dir).normalized()
        pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.use_clamp = True
        cl = nt.nodes.new('ShaderNodeMath'); cl.operation = 'MAXIMUM'; cl.inputs[1].default_value = 0
        nt.links.new(dp.outputs['Value'], cl.inputs[0])
        nt.links.new(cl.outputs[0], pw.inputs[0]); pw.inputs[1].default_value = 6.0
        mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = warm_amt
        nt.links.new(pw.outputs[0], mul.inputs[0])
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'ADD'
        nt.links.new(mul.outputs[0], mix.inputs[0])
        nt.links.new(col, mix.inputs[6])
        mix.inputs[7].default_value = (*warm_col, 1)
        col = mix.outputs[2]
    nt.links.new(col, bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w, bg

# --------------------------------------------------------------------------- materials

class NB:
    """Tiny node-building helper."""
    def __init__(self, mat):
        self.m = mat
        self.nt = mat.node_tree
        self.n = self.nt.nodes
        self.l = self.nt.links

    def new(self, t, **kw):
        nd = self.n.new(t)
        for k, v in kw.items():
            if k.startswith("_"):
                setattr(nd, k[1:], v)
            else:
                try:
                    nd.inputs[k].default_value = v
                except (KeyError, TypeError):
                    setattr(nd, k, v)
        return nd

    def link(self, a, b):
        self.l.new(a, b)
        return b

    def math(self, op, a, b=None, clamp=False):
        nd = self.n.new('ShaderNodeMath')
        nd.operation = op
        nd.use_clamp = clamp
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                nd.inputs[i].default_value = x
            else:
                self.l.new(x, nd.inputs[i])
        return nd.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        nd = self.n.new('ShaderNodeVectorMath')
        nd.operation = op
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (tuple, list, Vector)):
                nd.inputs[i].default_value = x
            else:
                self.l.new(x, nd.inputs[i])
        if scale is not None:
            if isinstance(scale, (int, float)):
                nd.inputs[3].default_value = scale
            else:
                self.l.new(scale, nd.inputs[3])
        return nd.outputs[1] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else nd.outputs[0]

    def mix(self, fac, a, b, blend='MIX', dtype='RGBA', clamp=True):
        nd = self.n.new('ShaderNodeMix')
        nd.data_type = dtype
        nd.blend_type = blend
        nd.clamp_result = clamp
        nd.clamp_factor = clamp
        if dtype == 'RGBA':
            ia, ib, io = 6, 7, 2
        elif dtype == 'FLOAT':
            ia, ib, io = 2, 3, 0
        else:
            ia, ib, io = 4, 5, 1
        for i, x in ((0, fac), (ia, a), (ib, b)):
            if isinstance(x, (int, float)):
                nd.inputs[i].default_value = x
            elif isinstance(x, (tuple, list)):
                nd.inputs[i].default_value = x if len(x) == 4 or dtype != 'RGBA' else (*x, 1)
            else:
                self.l.new(x, nd.inputs[i])
        return nd.outputs[io]

    def ramp(self, fac, stops):
        nd = self.n.new('ShaderNodeValToRGB')
        cr = nd.color_ramp
        if isinstance(fac, (int, float)):
            nd.inputs[0].default_value = fac
        else:
            self.l.new(fac, nd.inputs[0])
        for i, (pos, col) in enumerate(stops):
            c = col if len(col) == 4 else (*col, 1)
            if i < 2:
                cr.elements[i].position = pos
                cr.elements[i].color = c
            else:
                e = cr.elements.new(pos)
                e.color = c
        return nd.outputs[0]

    def maprange(self, x, a, b, c=0.0, d=1.0, clamp=True):
        nd = self.n.new('ShaderNodeMapRange')
        nd.clamp = clamp
        self.l.new(x, nd.inputs[0]) if not isinstance(x, (int, float)) else None
        nd.inputs[1].default_value = a; nd.inputs[2].default_value = b
        nd.inputs[3].default_value = c; nd.inputs[4].default_value = d
        return nd.outputs[0]

    def noise(self, vec=None, scale=5.0, detail=4.0, rough=0.5, dist=0.0, dim='3D', lac=2.0, ntype='FBM'):
        nd = self.n.new('ShaderNodeTexNoise')
        nd.noise_dimensions = dim
        try:
            nd.noise_type = ntype
        except Exception:
            pass
        nd.inputs['Scale'].default_value = scale
        nd.inputs['Detail'].default_value = detail
        nd.inputs['Roughness'].default_value = rough
        nd.inputs['Distortion'].default_value = dist
        nd.inputs['Lacunarity'].default_value = lac
        if vec is not None:
            self.l.new(vec, nd.inputs['Vector'])
        return nd

    def voronoi(self, vec=None, scale=5.0, feature='F1', metric='EUCLIDEAN', rand=1.0, dim='3D'):
        nd = self.n.new('ShaderNodeTexVoronoi')
        nd.voronoi_dimensions = dim
        nd.feature = feature
        nd.distance = metric
        nd.inputs['Scale'].default_value = scale
        nd.inputs['Randomness'].default_value = rand
        if vec is not None:
            self.l.new(vec, nd.inputs['Vector'])
        return nd

    def wave(self, vec=None, scale=5.0, dist=0.0, detail=2.0, wtype='BANDS', direction='X', profile='SIN'):
        nd = self.n.new('ShaderNodeTexWave')
        nd.wave_type = wtype
        if wtype == 'BANDS':
            nd.bands_direction = direction
        else:
            nd.rings_direction = direction
        nd.wave_profile = profile
        nd.inputs['Scale'].default_value = scale
        nd.inputs['Distortion'].default_value = dist
        nd.inputs['Detail'].default_value = detail
        if vec is not None:
            self.l.new(vec, nd.inputs['Vector'])
        return nd

    def coord(self, kind='Object'):
        tc = self.n.get("_tc") or self.n.new('ShaderNodeTexCoord')
        tc.name = "_tc"
        return tc.outputs[kind]

    def mapping(self, vec, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        nd = self.n.new('ShaderNodeMapping')
        self.l.new(vec, nd.inputs[0])
        nd.inputs['Location'].default_value = loc
        nd.inputs['Rotation'].default_value = rot
        nd.inputs['Scale'].default_value = scale
        return nd.outputs[0]

    def bump(self, height, strength=0.2, distance=0.01, normal=None):
        nd = self.n.new('ShaderNodeBump')
        nd.inputs['Strength'].default_value = strength
        nd.inputs['Distance'].default_value = distance
        self.l.new(height, nd.inputs['Height'])
        if normal is not None:
            self.l.new(normal, nd.inputs['Normal'])
        return nd.outputs[0]

    @property
    def bsdf(self):
        return self.n.get("Principled BSDF")

    @property
    def out(self):
        return self.n.get("Material Output")

    def set(self, sock, val):
        """Set a BSDF input to a value or link."""
        if isinstance(val, (int, float)):
            self.bsdf.inputs[sock].default_value = val
        elif isinstance(val, (tuple, list)):
            self.bsdf.inputs[sock].default_value = val if len(val) == 4 or sock in ('Normal', 'Tangent', 'Subsurface Radius') else (*val, 1)
        else:
            self.l.new(val, self.bsdf.inputs[sock])


def mat(name, color=(0.8, 0.8, 0.8), metal=0.0, rough=0.5, spec=0.5, coat=0.0, coat_rough=0.05,
        sss=0.0, sss_radius=(1.0, 0.4, 0.2), sss_scale=0.05, transmission=0.0, ior=1.45, alpha=1.0,
        emission=None, emit=0.0, aniso=0.0, sheen=0.0, thin_film=0.0, normal=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs['Base Color'].default_value = (*color, 1) if len(color) == 3 else color
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    b.inputs['Specular IOR Level'].default_value = spec
    b.inputs['IOR'].default_value = ior
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Coat Roughness'].default_value = coat_rough
    b.inputs['Subsurface Weight'].default_value = sss
    b.inputs['Subsurface Radius'].default_value = sss_radius
    b.inputs['Subsurface Scale'].default_value = sss_scale
    b.inputs['Transmission Weight'].default_value = transmission
    b.inputs['Alpha'].default_value = alpha
    b.inputs['Anisotropic'].default_value = aniso
    b.inputs['Sheen Weight'].default_value = sheen
    b.inputs['Thin Film Thickness'].default_value = thin_film
    if emission is not None:
        b.inputs['Emission Color'].default_value = (*emission, 1)
        b.inputs['Emission Strength'].default_value = emit
    if transmission > 0 or alpha < 1:
        try:
            m.surface_render_method = 'DITHERED' if transmission > 0 else 'BLENDED'
            m.use_raytrace_refraction = transmission > 0
        except Exception:
            pass
    return m


def emit_mat(name, color, strength=5.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new('ShaderNodeEmission')
    e.inputs[0].default_value = (*color, 1)
    e.inputs[1].default_value = strength
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def brushed_metal(name, color=(0.75, 0.75, 0.76), rough=0.28, aniso=0.6, scale=400, bump=0.05, scratches=0.0):
    m = mat(name, color, metal=1.0, rough=rough, aniso=aniso)
    nb = NB(m)
    co = nb.coord('Object')
    stretched = nb.mapping(co, scale=(1, 1, scale))
    n = nb.noise(stretched, scale=6, detail=8, rough=0.6)
    r = nb.maprange(n.outputs['Fac'], 0.3, 0.7, rough * 0.8, rough * 1.25)
    if scratches > 0:
        sc = nb.noise(nb.mapping(co, scale=(1, 60, 1), rot=(0, 0, 0.6)), scale=40, detail=2, rough=0.3)
        s2 = nb.maprange(sc.outputs['Fac'], 0.62, 0.66, 0.0, scratches)
        r = nb.math('ADD', r, s2)
    nb.set('Roughness', r)
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=bump, distance=0.001))
    return m


def wear_masks(nb, co, edge_px=0.012, scale=3.0):
    """Edge and cavity masks that work in EEVEE and Cycles (AO node, local only).
    Returns (edge 0..1 on convex edges, cavity 0..1 in creases/corners), both broken up by noise."""
    ao_in = nb.new('ShaderNodeAmbientOcclusion', samples=8, inside=True, only_local=True)
    ao_in.inputs['Distance'].default_value = edge_px
    ao_out = nb.new('ShaderNodeAmbientOcclusion', samples=8, only_local=True)
    ao_out.inputs['Distance'].default_value = edge_px * 8
    brk = nb.noise(co, scale=scale * 14, detail=10, rough=0.72)
    edge = nb.maprange(nb.math('ADD', nb.math('SUBTRACT', 1.0, ao_in.outputs['AO']),
                               nb.math('MULTIPLY', brk.outputs['Fac'], 0.55)), 0.62, 0.8)
    cav = nb.maprange(nb.math('ADD', nb.math('SUBTRACT', 1.0, ao_out.outputs['AO']),
                              nb.math('MULTIPLY', brk.outputs['Fac'], 0.3)), 0.25, 0.7)
    return edge, cav


def painted(name, color, rough=0.35, coat=0.3, grime=0.0, scale=3.0, speck=0.0, wear=None, streaks=None,
            under=(0.33, 0.33, 0.34)):
    """Industrial / product paint: macro value variation, orange-peel micro-normal, rain/grime streaks, cavity dirt and
    chipped convex edges revealing bare metal (wear defaults to follow `grime`). Works in EEVEE and Cycles."""
    m = mat(name, color, rough=rough, coat=coat, coat_rough=0.08)
    nb = NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=scale, detail=6, rough=0.6)
    wear = grime * 0.8 if wear is None else wear
    streaks = grime if streaks is None else streaks
    base = (*color, 1)
    # large, soft paint-batch/sun-fade variation (+-6 %)
    mv = nb.maprange(nb.noise(co, scale=scale * 0.35, detail=3, rough=0.5).outputs['Fac'], 0.3, 0.7)
    base = nb.mix(mv, (color[0] * 0.94, color[1] * 0.94, color[2] * 0.95, 1), (color[0] * 1.05, color[1] * 1.05, color[2] * 1.04, 1))
    rgh = nb.maprange(n.outputs['Fac'], 0.35, 0.65, rough * 0.8, rough * 1.3)
    dirty = (color[0] * 0.5, color[1] * 0.48, color[2] * 0.45, 1)
    if grime > 0:
        g = nb.noise(co, scale=scale * 4, detail=8, rough=0.7)
        gm = nb.maprange(g.outputs['Fac'], 0.5, 0.75, 0, grime)
        base = nb.mix(gm, base, dirty)
    if streaks > 0:
        # vertical run-off streaks (object Z), thin and irregular
        st = nb.noise(nb.mapping(co, scale=(scale * 9, scale * 9, scale * 0.6)), scale=4.0, detail=6, rough=0.6)
        sm = nb.maprange(st.outputs['Fac'], 0.56, 0.72, 0, 0.45 * streaks)
        base = nb.mix(sm, base, dirty)
        rgh = nb.math('ADD', rgh, nb.math('MULTIPLY', sm, 0.25))
    if wear > 0 or grime > 0:
        edge, cav = wear_masks(nb, co, scale=scale)
        base = nb.mix(nb.math('MULTIPLY', cav, min(1.0, 0.35 + grime * 0.6)), base, dirty)
        if wear > 0:
            chip = nb.math('MULTIPLY', edge, min(1.0, wear * 1.4))
            base = nb.mix(chip, base, (*under, 1))
            nb.set('Metallic', nb.math('MULTIPLY', chip, 0.9))
            rgh = nb.mix(chip, rgh, rough * 0.9, dtype='FLOAT')
    nb.set('Base Color', base)
    nb.set('Roughness', rgh)
    op = nb.noise(co, scale=120, detail=3).outputs['Fac']
    nb.set('Normal', nb.bump(op, strength=0.03, distance=0.0005))
    return m


def auto_bevel(objs, frac=0.035, max_w=0.012, min_w=0.0006, segments=2, angle=35):
    """Soft machined edges on hard-surface meshes (bevel modifier, angle limited, harden normals) so edges catch
    light instead of reading as CG boxes. Width scales with each object's smallest dimension."""
    for o in objs:
        if o.type != 'MESH' or any(md.type == 'BEVEL' for md in o.modifiers) or len(o.data.polygons) > 200000:
            continue
        dims = sorted(o.dimensions)
        w = max(min_w, min(max_w, frac * (dims[0] if dims[0] > 1e-4 else dims[1])))
        md = o.modifiers.new("AutoBevel", 'BEVEL')
        md.width = w
        md.segments = segments
        md.limit_method = 'ANGLE'
        md.angle_limit = math.radians(angle)
        md.harden_normals = False
        md.use_clamp_overlap = True


def glass(name, color=(1, 1, 1), rough=0.0, ior=1.5, thin=False):
    m = mat(name, color, rough=rough, transmission=1.0, ior=ior)
    try:
        m.surface_render_method = 'DITHERED'
        m.use_raytrace_refraction = True
        m.thickness_mode = 'SLAB' if thin else 'SPHERE'
    except Exception:
        pass
    if thin and 'Thin Wall' in m.node_tree.nodes["Principled BSDF"].inputs:      # Blender 5.2+
        m.node_tree.nodes["Principled BSDF"].inputs['Thin Wall'].default_value = 0.0
    return m


def skin(name, tone=(0.62, 0.40, 0.30), rough=0.42, pores=1.0, scale=1.0, flush=0.0):
    m = mat(name, tone, rough=rough, sss=1.0, sss_radius=(1.0, 0.35, 0.18), sss_scale=0.004 * scale, spec=0.45)
    b = m.node_tree.nodes["Principled BSDF"]
    try:
        b.subsurface_method = 'RANDOM_WALK_SKIN'
    except Exception:
        pass
    nb = NB(m)
    co = nb.coord('Object')
    mottled = nb.noise(co, scale=18 / scale, detail=6, rough=0.55)
    red = nb.noise(co, scale=6 / scale, detail=3, rough=0.5)
    base = nb.mix(nb.maprange(mottled.outputs['Fac'], 0.35, 0.65, 0.0, 1.0),
                  (tone[0] * 0.92, tone[1] * 0.86, tone[2] * 0.84, 1), (tone[0] * 1.05, tone[1] * 1.03, tone[2] * 1.0, 1))
    base = nb.mix(nb.maprange(red.outputs['Fac'], 0.5, 0.8, 0.0, 0.25 + flush), base,
                  (tone[0] * 1.08, tone[1] * 0.78, tone[2] * 0.75, 1))
    nb.set('Base Color', base)
    pore = nb.voronoi(co, scale=900 / scale, feature='F1')
    pd = nb.maprange(pore.outputs['Distance'], 0.0, 0.35, 0.0, 1.0)
    fine = nb.noise(co, scale=300 / scale, detail=4, rough=0.6)
    h = nb.math('ADD', nb.math('MULTIPLY', pd, 0.6), nb.math('MULTIPLY', fine.outputs['Fac'], 0.4))
    # fine crossing lines (skin microrelief)
    l1 = nb.wave(co, scale=160 / scale, dist=6.0, detail=3, wtype='BANDS', direction='X')
    l2 = nb.wave(nb.mapping(co, rot=(0.3, 0.9, 0.5)), scale=210 / scale, dist=7.0, detail=3, wtype='BANDS', direction='Y')
    lines = nb.math('MINIMUM', l1.outputs['Fac'], l2.outputs['Fac'])
    h = nb.math('ADD', h, nb.math('MULTIPLY', nb.maprange(lines, 0.0, 0.25, -0.5, 0.0), 1.0))
    nb.set('Normal', nb.bump(h, strength=0.22 * pores, distance=0.0004 * scale))
    nb.set('Roughness', nb.maprange(fine.outputs['Fac'], 0.3, 0.7, rough - 0.08, rough + 0.1))
    return m


def fabric(name, color, rough=0.85, weave=300.0, sheen=0.4, fuzz=0.3, knit=False):
    m = mat(name, color, rough=rough, sheen=sheen)
    nb = NB(m)
    co = nb.coord('UV') if False else nb.coord('Object')
    if knit:
        w1 = nb.wave(nb.mapping(co, scale=(1, 3.0, 1)), scale=weave, dist=2.0, wtype='BANDS', direction='Z')
        w2 = nb.wave(co, scale=weave * 0.5, dist=1.0, wtype='BANDS', direction='X', profile='SAW')
        h = nb.math('MULTIPLY', w1.outputs['Fac'], w2.outputs['Fac'])
    else:
        w1 = nb.wave(co, scale=weave, wtype='BANDS', direction='X')
        w2 = nb.wave(co, scale=weave, wtype='BANDS', direction='Z')
        h = nb.math('MAXIMUM', w1.outputs['Fac'], w2.outputs['Fac'])
    fz = nb.noise(co, scale=weave * 0.3, detail=6, rough=0.7)
    hh = nb.math('ADD', h, nb.math('MULTIPLY', fz.outputs['Fac'], fuzz))
    col = nb.mix(nb.maprange(fz.outputs['Fac'], 0.3, 0.7), (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, 1), (*color, 1))
    nb.set('Base Color', col)
    nb.set('Normal', nb.bump(hh, strength=0.35, distance=0.002))
    return m


def rubber(name, color=(0.02, 0.02, 0.022), rough=0.6):
    m = mat(name, color, rough=rough, spec=0.35)
    nb = NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=200, detail=3)
    nb.set('Normal', nb.bump(n.outputs['Fac'], strength=0.05, distance=0.001))
    return m


def concrete(name, color=(0.5, 0.49, 0.47), scale=4.0, rough=0.85, stains=0.3):
    m = mat(name, color, rough=rough)
    nb = NB(m)
    co = nb.coord('Object')
    n = nb.noise(co, scale=scale, detail=10, rough=0.65)
    st = nb.noise(co, scale=scale * 0.3, detail=4)
    cfac = nb.maprange(n.outputs['Fac'], 0.3, 0.7)
    c = nb.mix(cfac, (color[0] * 0.8, color[1] * 0.8, color[2] * 0.78, 1), (color[0] * 1.08, color[1] * 1.08, color[2] * 1.06, 1))
    c = nb.mix(nb.maprange(st.outputs['Fac'], 0.55, 0.75, 0, stains), c, (color[0] * 0.55, color[1] * 0.53, color[2] * 0.5, 1))
    nb.set('Base Color', c)
    fine = nb.noise(co, scale=scale * 60, detail=4)
    nb.set('Normal', nb.bump(nb.math('ADD', n.outputs['Fac'], nb.math('MULTIPLY', fine.outputs['Fac'], 0.5)), strength=0.15, distance=0.01))
    return m


def volume_mat(name, density=0.1, color=(1, 1, 1), anisotropy=0.2, noise_scale=None, emission=None, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.inputs['Color'].default_value = (*color, 1)
    pv.inputs['Density'].default_value = density
    pv.inputs['Anisotropy'].default_value = anisotropy
    if emission is not None:
        pv.inputs['Emission Color'].default_value = (*emission, 1)
        pv.inputs['Emission Strength'].default_value = emit
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(pv.outputs[0], o.inputs['Volume'])
    return m


def ensure_uv(o):
    if o.type == 'MESH' and not o.data.uv_layers:
        bpy.context.view_layer.objects.active = o
        with bpy.context.temp_override(active_object=o, object=o, selected_objects=[o]):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')


def rng(seed):
    return random.Random(seed)


def save(name):
    p = os.path.join(ROOT, "scenes", "blend")
    os.makedirs(p, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(p, name + ".blend"), compress=True)


def world_bluehour(glow_az=30.0, glow=6.0, zenith=(0.012, 0.03, 0.10), horizon=(0.12, 0.15, 0.25),
                   ground=(0.004, 0.005, 0.007), glow_col=(1.0, 0.38, 0.08), glow_width=40.0,
                   band_height=5.0, strength=1.0, sun=None):
    """Twilight sky: elevation gradient + a warm afterglow band hugging the horizon around azimuth glow_az
    (degrees, 0 = +Y, 90 = +X). Optional sun = (elev, az, size_deg, strength) adds a sun disc (sunrise)."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new("BlueHour")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    m = type("M", (), {})()
    m.node_tree = nt
    nb = NB.__new__(NB)
    nb.m, nb.nt, nb.n, nb.l = None, nt, nt.nodes, nt.links
    tc = nt.nodes.new('ShaderNodeTexCoord')
    d = tc.outputs['Generated']
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(d, sep.inputs[0])
    z = sep.outputs[2]
    # sky gradient on elevation
    zz = nb.maprange(z, 0.0, 1.0)
    grad = nb.ramp(zz, [(0.0, horizon), (0.12, tuple(sb_lerp3(horizon, zenith, 0.45))), (0.5, zenith), (1.0, tuple(c * 0.7 for c in zenith))])
    below = nb.maprange(z, -0.02, 0.0, 0.0, 1.0)
    col = nb.mix(below, (*ground, 1), grad)
    # afterglow band
    az = math.radians(glow_az)
    gdir = (math.sin(az), math.cos(az), 0.0)
    dp = nb.vmath('DOT_PRODUCT', d, gdir)
    horiz_len = nb.math('SQRT', nb.math('MAXIMUM', nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', z, z)), 1e-4))
    cosang = nb.math('DIVIDE', dp, horiz_len)
    k = 1.0 / max(1e-3, (1 - math.cos(math.radians(glow_width))))
    lobe = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('SUBTRACT', cosang, 1.0), k))
    bh = math.sin(math.radians(band_height))
    band = nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('ABSOLUTE', z), -1.0 / bh))
    band = nb.math('MULTIPLY', band, below)
    g = nb.math('MULTIPLY', nb.math('MULTIPLY', lobe, band), glow)
    col = nb.mix(g, col, (*glow_col, 1), blend='ADD', clamp=False)
    # soft higher rose/violet belt above the glow (twilight wedge)
    wedge = nb.math('MULTIPLY', nb.math('MULTIPLY', lobe, nb.math('EXPONENT', nb.math('MULTIPLY', nb.math('ABSOLUTE', nb.math('SUBTRACT', z, 0.18)), -8.0))), glow * 0.05)
    col = nb.mix(nb.math('MULTIPLY', wedge, below), col, (0.9, 0.45, 0.5, 1), blend='ADD', clamp=False)
    if sun is not None:
        se, sa, ssize, sstr = sun
        sd = sun_dir(se, sa)
        sdp = nb.vmath('DOT_PRODUCT', d, tuple(sd))
        disc = nb.maprange(sdp, math.cos(math.radians(ssize)), math.cos(math.radians(ssize * 0.8)), 0.0, 1.0)
        halo = nb.math('POWER', nb.math('MAXIMUM', sdp, 0.0), 400.0)
        s_amt = nb.math('ADD', nb.math('MULTIPLY', disc, sstr), nb.math('MULTIPLY', halo, sstr * 0.02))
        col = nb.mix(nb.math('MULTIPLY', s_amt, below), col, (1.0, 0.72, 0.4, 1), blend='ADD', clamp=False)
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs[1].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(col, bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w, bg


def sb_lerp3(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]
