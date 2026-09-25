"""ORBIT render helpers (EARTH & ORBIT team): layered Earth/station rendering for 4K budgets.

The analytic Earth world (lib/earth.py) is expensive per TAA sample. `layered()` renders it in a separate
low-sample scene ("EarthPass") sharing the camera, while the main scene renders geometry at full samples
with a transparent film, lit by a cheap baked equirect of the same Earth (earthshine + reflections).
The compositor alpha-overs both in scene-linear, so the main scene's view transform applies to the result.

    E = earth.build(...); E.sun_lamp(); ... camera anim ...; E.track(cam, f0, f1)
    orbit.layered(E, cam, earth_samples=12)     # call after the camera is animated
    sb.render_shot(sid)
"""
import bpy, os, math
import sb

ROOT = sb.ROOT


def bake_env(E, loc_m=(0, 0, 0), res=4096, path=None, samples=4):
    """Render the Earth world as an equirect from scene point loc_m (Cycles, sun disc off). Returns path."""
    main = bpy.context.scene
    path = path or os.path.join(ROOT, "lookdev", "orbit", "cache", "env_%s.exr" % E.world.name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = bpy.data.scenes.new("EnvBake")
    tmp.world = E.world
    tmp.render.engine = 'CYCLES'
    try:
        p = bpy.context.preferences.addons['cycles'].preferences
        p.compute_device_type = 'METAL'; p.get_devices()
        for d in p.devices: d.use = (d.type == 'METAL')
        tmp.cycles.device = 'GPU'
    except Exception:
        pass
    tmp.cycles.samples = samples
    tmp.cycles.use_denoising = False
    tmp.render.resolution_x, tmp.render.resolution_y = res, res // 2
    tmp.render.resolution_percentage = 100
    tmp.view_settings.view_transform = 'Standard'
    tmp.render.image_settings.file_format = 'OPEN_EXR'
    tmp.render.image_settings.color_depth = '16'
    cd = bpy.data.cameras.new("EnvCam"); cd.type = 'PANO'; cd.panorama_type = 'EQUIRECTANGULAR'
    co = bpy.data.objects.new("EnvCam", cd); tmp.collection.objects.link(co)
    co.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
    tmp.camera = co
    # freeze camera km at loc_m, disc off
    old_disc = E.disc_node.outputs[0].default_value
    E.disc_node.outputs[0].default_value = 0.0
    fcurves_backup = _mute_cam_anim(E, True)
    E.set_cam(loc_m)
    tmp.render.filepath = path
    with bpy.context.temp_override(scene=tmp):
        bpy.ops.render.render(write_still=True, scene=tmp.name)
    E.disc_node.outputs[0].default_value = old_disc
    _mute_cam_anim(E, False)
    bpy.data.objects.remove(co); bpy.data.scenes.remove(tmp)
    bpy.context.window_manager  # noqa
    return path


def _mute_cam_anim(E, mute):
    ad = E.world.node_tree.animation_data
    if ad and ad.action:
        for fc in _fcurves(ad.action):
            if 'CamKm' in fc.data_path:
                fc.mute = mute


def _fcurves(action):
    try:
        return list(action.fcurves)
    except Exception:
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    out += list(cb.fcurves)
        return out


def env_world(path, strength=1.0, name="EarthEnv"):
    w = bpy.data.worlds.new(name)
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord')
    et = nt.nodes.new('ShaderNodeTexEnvironment')
    et.image = bpy.data.images.load(path, check_existing=True)
    et.interpolation = 'Cubic'
    nt.links.new(tc.outputs['Generated'], et.inputs[0])
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs[1].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(et.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs[0])
    try:
        w.sun_threshold = 1e9
        w.probe_resolution = '2048'
    except Exception:
        pass
    return w


def layered(E, cam, earth_samples=12, env_res=4096, env_loc=None, env_path=None, dof=True):
    """Split rendering into EarthPass (analytic world, few samples) + main (geometry, env-lit)."""
    main = bpy.context.scene
    # 1) env for lighting/reflections in the main scene
    if env_loc is None:
        main.frame_set(main.frame_start)
        env_loc = tuple(cam.matrix_world.translation)
    p = env_path or bake_env(E, env_loc, env_res)
    main.world = env_world(p)
    # 2) Earth pass scene
    ep = bpy.data.scenes.new("EarthPass")
    ep.world = E.world
    ep.collection.objects.link(cam)
    ep.camera = cam
    r, m = ep.render, main.render
    r.engine = 'BLENDER_EEVEE'
    r.resolution_x, r.resolution_y, r.resolution_percentage = m.resolution_x, m.resolution_y, 100
    r.use_border, r.use_crop_to_border = m.use_border, m.use_crop_to_border
    r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = m.border_min_x, m.border_min_y, m.border_max_x, m.border_max_y
    r.use_motion_blur = m.use_motion_blur
    r.motion_blur_shutter = m.motion_blur_shutter
    r.fps = m.fps
    ep.frame_start, ep.frame_end = main.frame_start, main.frame_end
    ep.eevee.taa_render_samples = earth_samples
    ep.eevee.use_raytracing = False
    ep.view_settings.view_transform = main.view_settings.view_transform
    r.film_transparent = False
    # 3) main: transparent film
    m.film_transparent = True
    # 4) compositor
    ng = bpy.data.node_groups.new("OrbitComp", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    rl_e = ng.nodes.new("CompositorNodeRLayers"); rl_e.scene = ep
    rl_m = ng.nodes.new("CompositorNodeRLayers"); rl_m.scene = main
    ao = ng.nodes.new("CompositorNodeAlphaOver")
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl_e.outputs['Image'], ao.inputs['Background'])
    ng.links.new(rl_m.outputs['Image'], ao.inputs['Foreground'])
    ng.links.new(ao.outputs['Image'], out.inputs[0])
    main.compositing_node_group = ng
    try:
        main.render.use_compositing = True
    except Exception:
        pass
    return ep
