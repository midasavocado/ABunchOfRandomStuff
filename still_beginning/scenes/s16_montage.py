"""s16 ABUNDANCE -> daily life montage (fast cuts, 1-2 s each):
s16a (1350-1373): morning sun through a kitchen window, a tap fills a glass with clean water - the stream braids and
sparkles, bubbles swirl, the level rises. Macro, backlit.
s16b (1373-1395): a sunlit greenhouse - tiers of lettuce and basil beaded with water; the camera slides past the
leaves, a grower soft beyond, the glass roof structure overhead.
s16c (1395-1440): a learning space - a bright school maker-library; kids around a table building a small rover,
a teacher leaning in; books, plants, big windows onto trees."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, dbgcam
import timeline as TL
from mathutils import Vector as V, Matrix, Euler

sid = (sb.argv() or ["s16a"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
rs = random.Random(16)


def water_stream(name, top, bottom, r0=0.006, r1=0.0042, f0=0, f1=0, mat=None):
    """Laminar tap stream that frays slightly as it falls: a tube along a gently wavering line, radius tapering;
    animated by a travelling noise displacement (ripples run down the stream)."""
    n = 40
    pts = [V(top).lerp(V(bottom), i / (n - 1)) for i in range(n)]
    o = sb.tube_along(name, [tuple(p) for p in pts], radius=r0, mat=mat, bevel_res=6,
                      taper=lambda t: 1.0 - (1.0 - r1 / r0) * t)
    o.data.resolution_u = 2
    me = sb.to_mesh(o)
    tex = bpy.data.textures.new(name + "Ripple", 'CLOUDS')
    tex.noise_scale = 0.004
    tex.noise_depth = 2
    d = me.modifiers.new("Ripple", 'DISPLACE')
    d.texture = tex
    d.strength = 0.0016
    d.texture_coords = 'OBJECT'
    em = sb.empty(name + "RippleCo")
    d.texture_coords_object = em
    for f in range(f0 - 2, f1 + 3):
        em.location = (0, 0, 0.02 * (f - f0))           # pattern slides downward (in object coords upward = flow)
        em.keyframe_insert("location", frame=f)
    for p in me.data.polygons:
        p.use_smooth = True
    return me


if sid == "s16a":
    import home
    sb.setup_render("CYCLES", cycles_samples=128, mblur=True, shutter=0.5, look="AgX - Medium High Contrast")
    sc.cycles.transmission_bounces = 16
    sc.cycles.max_bounces = 16
    M = home.materials()
    home.build(city=False)
    # morning: sun low through the window wall (-X), cool sky + warm sun, plants on the sill
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT' and o.name.startswith(("Pendant", "UnderCab")):
            o.data.energy *= 0.05
    sb.world_sky(elev=12, azim=268, strength=0.35, aerosol=1.5)
    sb.sun(12, 268, energy=5.0, color=(1.0, 0.82, 0.62), angle=0.8)
    # outside: garden trees (soft bokeh through the glazing)
    import land
    lib = land.tree_library(n=2, seed=160, leaf_count=40000, height=8.0, crown_r=3.2)
    land.place_trees(lib, [V((-9, -3, -0.2)), V((-12, 2, -0.2)), V((-8, 5.5, -0.2)), V((-15, -6, -0.2))], name="GardenTree")
    sb.prim("plane", "Garden", loc=(-20, 0, -0.25), scale=(18, 18, 1), mat=sb.mat("Lawn", (0.08, 0.17, 0.04), rough=0.9))
    # the sink + tap on the counter near the window end
    CX, CY, CZ = -1.0, 2.05, 0.92
    sink = sb.lathe("Sink", [(0.0, -0.18), (0.2, -0.18), (0.22, -0.02), (0.235, 0.0), (0.24, 0.0)], segs=64, mat=M["steel"])
    sink.location = (CX, CY, CZ)
    sink.scale = (1.6, 1.0, 1.0)
    tap_top = V((CX + 0.02, CY + 0.12, CZ + 0.36))
    spout = V((CX - 0.02, CY - 0.02, CZ + 0.31))
    import rocket
    rocket.rod("TapBody", (CX + 0.02, CY + 0.2, CZ), (CX + 0.02, CY + 0.2, CZ + 0.3), 0.018, M["brass"], verts=32)
    sb.tube_along("TapNeck", [(CX + 0.02, CY + 0.2, CZ + 0.3), (CX + 0.02, CY + 0.16, CZ + 0.38), (CX + 0.0, CY + 0.06, CZ + 0.38),
                              tuple(spout + V((0, 0, 0.03))), tuple(spout)], radius=0.011, mat=M["brass"], bevel_res=6)
    rocket.rod("TapLever", (CX + 0.02, CY + 0.2, CZ + 0.28), (CX + 0.1, CY + 0.2, CZ + 0.33), 0.006, M["brass"], verts=16)
    # the glass under the stream, held in place on the sink edge (the level rises)
    GB = V((spout.x, spout.y, CZ - 0.02))
    g, _ = home.glass("FillGlass", GB, M, fill=0.0)
    wm = home.materials()["water"]
    water = sb.lathe("FillWater", [(0.0, 0.0119), (0.036 - 0.0022 + 0.00015, 0.0119), (0.036 - 0.0022 + 0.0003, 0.02), (0.0, 0.02)],
                     segs=96, mat=wm)
    water.location = GB
    for f in range(F0 - 2, F1 + 3):
        t = (f - (F0 - 30)) / 60.0
        water.scale = (1.0, 1.0, 1.0 + 3.8 * t)
        water.keyframe_insert("scale", frame=f)
    stream = water_stream("Stream", spout - V((0, 0, 0.002)), GB + V((0, 0, 0.04)), f0=F0, f1=F1, mat=wm)
    # bubbles churning under the impact
    bm_ = sb.mat("Bubble", (1, 1, 1), rough=0.0, transmission=1.0, ior=1.0 / 1.33)
    for k in range(60):
        b = sb.prim("ico", "Bubble", loc=GB + V((rs.uniform(-0.02, 0.02), rs.uniform(-0.02, 0.02), 0.015 + rs.uniform(0, 0.03))),
                    subdivisions=2, radius=rs.uniform(0.0004, 0.0014), mat=bm_)
        for f in range(F0 - 2, F1 + 3):
            ph = rs.random() * 6.28
            b.location.z = GB.z + 0.015 + ((f * 0.0012 + k * 0.0007) % 0.04)
            b.keyframe_insert("location", frame=f)
    tgt = GB + V((0, 0, 0.12))
    CO = V((0.44, 0.1, 0.06))                        # looking back toward the window: the sun backlights the water
    cam = sb.camera("Cam", loc=tgt + CO, target=tgt, lens=85, fstop=3.5, clip=(0.01, 200))
    sb.cam_bake(cam, F0, F1, lambda t: tgt + CO * sb.lerp(1.0, 0.95, t) + V((0.0, 0.0, 0.006 * t)), lambda t: tgt, focus="target")

elif sid == "s16b":
    import plants, rocket
    sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=-0.6)
    sb.world_sky(elev=35, azim=120, strength=0.3, aerosol=1.2)
    sb.sun(35, 120, energy=4.0, color=(1.0, 0.9, 0.76), angle=1.2)
    let_m = plants.leaf_mat("LettuceLeaf", (0.10, 0.24, 0.025), gloss=0.2, translucency=0.28, veins=9.0, var=0.2)
    basil_m = plants.leaf_mat("BasilLeaf", (0.030, 0.105, 0.016), gloss=0.35, translucency=0.2)
    basil_s = plants.stem_mat("BasilStem", (0.10, 0.20, 0.05))
    wm = plants.droplet_mat()
    steel = sb.brushed_metal("RackSteel", (0.7, 0.7, 0.71), rough=0.3)
    # NFT channel plastic: white food-grade PVC with longitudinal ribs, mineral/algae staining near the cups
    tray = sb.mat("Tray", (0.80, 0.80, 0.78), rough=0.35, coat=0.2)
    nbt = sb.NB(tray)
    cot = nbt.coord('Object')
    sept = nbt.new('ShaderNodeSeparateXYZ'); nbt.link(cot, sept.inputs[0])
    rib = nbt.math('SINE', nbt.math('MULTIPLY', sept.outputs[1], 2 * math.pi / 0.012))
    stn = nbt.noise(nbt.mapping(cot, scale=(0.3, 3.0, 3.0)), scale=4.0, detail=6, rough=0.6)
    stain = nbt.maprange(stn.outputs['Fac'], 0.5, 0.75, 0.0, 0.7)
    col = nbt.mix(stain, (0.80, 0.80, 0.78, 1), (0.42, 0.44, 0.30, 1))
    nbt.set('Base Color', col)
    nbt.set('Roughness', nbt.mix(stain, 0.32, 0.6, dtype='FLOAT'))
    nbt.set('Normal', nbt.bump(nbt.math('ADD', nbt.math('MULTIPLY', rib, 0.3), stn.outputs['Fac']), strength=0.25, distance=0.001))
    cup_m = sb.mat("NetCup", (0.02, 0.02, 0.022), rough=0.45)
    pipe_m = sb.mat("FeedPipe", (0.18, 0.19, 0.2), rough=0.4)
    drip_m = sb.mat("DripLine", (0.03, 0.03, 0.03), rough=0.5)
    # a small library of unique plants (6 lettuce, 4 basil) in hidden collections, instanced along the trays
    libs = []
    for k in range(10):
        c = bpy.data.collections.new("PlantLib%d" % k)
        bpy.context.scene.collection.children.link(c)
        before = set(bpy.data.objects)
        if k < 6:
            plants.lettuce("LibL%d" % k, V((0, 0, 0)), seed=k * 5 + 1, radius=0.12, leaf_m=let_m)
            pre = "LibL%d" % k
        else:
            plants.basil("LibB%d" % k, V((0, 0, 0)), seed=k * 7, height=0.2, leaf_m=basil_m, stem_m=basil_s, nodes=5)
            pre = "LibB%d" % k
        new = [o for o in bpy.data.objects if o not in before]
        if len(new) > 1:
            m_ = plants.merge(new, pre + "M")
            new = [m_] if m_ else new
        for o in new:
            for uc in o.users_collection:
                uc.objects.unlink(o)
            c.objects.link(o)
        lc = bpy.context.view_layer.layer_collection.children.get(c.name)
        if lc:
            lc.exclude = True
        libs.append(c)
    for row, y in enumerate((0.0, 1.4)):
        for tier, z in enumerate((0.45, 1.05, 1.65)):
            # two NFT channels per tier on steel cross-supports; feed manifold + drip lines at the head end
            for j in (-0.16, 0.16):
                rocket.box("Chan%d%d" % (row, tier), (2.5, y + j, z - 0.005), (9.0, 0.12, 0.07), tray, bev=0.008)
            for x in np.arange(-1.9, 7.0, 0.8):
                rocket.box("XSup", (x, y, z - 0.055), (0.03, 0.62, 0.03), steel, bev=0.003)
            rocket.rod("Feed", (-2.05, y - 0.3, z + 0.06), (-2.05, y + 0.3, z + 0.06), 0.016, pipe_m, verts=16)
            for j in (-0.16, 0.16):
                rocket.rod("Drip", (-2.05, y + j, z + 0.06), (-1.95, y + j, z + 0.04), 0.004, drip_m, verts=8)
            for k in range(20):
                x = -1.6 + k * 0.42 + rs.uniform(-0.03, 0.03)
                for j in (-0.16, 0.16):
                    lib_i = (6 + (k + tier) % 4) if (k + tier) % 3 == 0 else ((k * 3 + tier + row + (j > 0)) % 6)
                    sb.collection_instance(libs[lib_i], loc=(x, y + j, z + 0.04), rot=(0, 0, rs.uniform(0, 6.28)),
                                           scale=rs.uniform(0.85, 1.1), name="Plant")
                    sb.prim("cyl", "Cup", loc=(x, y + j, z + 0.032), vertices=24, radius=0.036, depth=0.012, mat=cup_m)
            for x in (-2.0, 2.5, 7.0):
                for dy in (-0.33, 0.33):
                    rocket.rod("Post", (x, y + dy, 0), (x, y + dy, 1.9), 0.02, steel, verts=12)
    # droplets on the nearest leaves
    near = [o for o in bpy.data.objects if o.name.startswith("LettuceRow0")]
    # glass roof: frames overhead + a grower soft in the aisle
    frame = sb.painted("GHFrame", (0.85, 0.85, 0.84), rough=0.4)
    for k in range(12):
        rocket.box("RoofBar", (-3 + k * 1.2, 0.7, 3.4), (0.05, 8.0, 0.1), frame, bev=0.005, rot=(0.25, 0, 0))
    for k in range(4):
        rocket.box("Purlin", (2.5, -2.5 + k * 2.2, 3.1 + k * 0.3), (16.0, 0.05, 0.08), frame, bev=0.005)
    sb.prim("plane", "Floor", scale=(20, 20, 1), mat=sb.concrete("GHFloor", (0.5, 0.49, 0.46), scale=3.0, stains=0.4))
    import folks
    bm, rig, parts = folks.person("teen", name="Grower")
    folks.arms_down(rig)
    folks.place(rig, (5.5, 0.72, 0.0), -90.0)
    def cp(t):
        return V((sb.lerp(0.05, 0.4, t), -0.62, 1.24))

    def ctg(t):
        return V((sb.lerp(0.55, 0.9, t), -0.16, 1.12))

    cam = sb.camera("Cam", loc=cp(0), target=ctg(0), lens=50, fstop=2.8, clip=(0.01, 200))
    sb.cam_bake(cam, F0, F1, cp, ctg, focus=lambda t: (cp(t) - ctg(t)).length)

else:  # s16c learning space
    import home, folks, mhchild, rocket
    sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast")
    sb.world_sky(elev=30, azim=250, strength=0.35, aerosol=1.2)
    sb.sun(30, 250, energy=3.5, color=(1.0, 0.9, 0.78), angle=1.0)
    M = home.materials()
    wood = M["oak"]
    # room: big windows (-X), bookshelves (+Y), a long work table with parts
    sb.prim("plane", "Floor", loc=(0.75, 0, 0), scale=(4.25, 4.0, 1), mat=M["floor"])
    sb.prim("cube", "Ceil", loc=(0.75, 0, 3.3), scale=(4.25, 4.0, 0.05), mat=M["wall"])
    for y in (-4.0, 4.0):
        sb.prim("cube", "Wall", loc=(0.75, y, 1.6), scale=(4.25, 0.05, 1.7), mat=M["wall"])
    sb.prim("cube", "BackWallE", loc=(5.0, 0, 1.6), scale=(0.05, 4.0, 1.7), mat=M["wall"])
    for k in range(6):
        rocket.box("Mullion", (-3.5, -3.5 + k * 1.4, 1.6), (0.08, 0.06, 3.2), M["window_frame"], bev=0.004)
    import land
    lib = land.tree_library(n=2, seed=170, leaf_count=40000, height=9.0, crown_r=3.5)
    land.place_trees(lib, [V((-9, -3, 0)), V((-11, 2.5, 0)), V((-8, 6, 0))], name="SchoolTree")
    sb.prim("plane", "Yard", loc=(-120, 0, -0.02), scale=(118, 200, 1), mat=sb.mat("Yard", (0.12, 0.2, 0.06), rough=0.9))
    land.place_trees(lib, [V((-rs.uniform(25, 90), rs.uniform(-40, 40), 0)) for _ in range(18)], scale_rng=(0.8, 1.4), name="FarTree")
    # bookshelves along +Y
    for k in range(5):
        rocket.box("Shelf", (-1.5 + k * 1.1, 3.8, 1.2), (1.0, 0.3, 2.4), M["oak_dark"], bev=0.01)
        for z in range(5):
            x = -1.95 + k * 1.1
            while x < -1.05 + k * 1.1:
                w = rs.uniform(0.025, 0.05)
                rocket.box("Book", (x, 3.72, 0.2 + z * 0.46 + 0.15), (w, 0.2, rs.uniform(0.22, 0.3)), rs.choice(M["book"]), bev=0.002)
                x += w + 0.004
    # the table + rover kit
    T = V((0.0, 0.0, 0.74))
    rocket.box("WorkTable", (0, 0, 0.72), (2.4, 1.0, 0.04), wood, bev=0.006)
    for sx in (-1.1, 1.1):
        for sy in (-0.42, 0.42):
            rocket.box("TLeg", (sx, sy, 0.35), (0.05, 0.05, 0.7), M["black"], bev=0.003)
    alu = sb.brushed_metal("Alu", (0.8, 0.8, 0.82), rough=0.3)
    tread = sb.rubber("WheelR", (0.05, 0.05, 0.055))
    nbw = sb.NB(tread)
    nbw.set('Normal', nbw.bump(nbw.wave(nbw.coord('Object'), scale=180, wtype='RINGS', direction='Z').outputs['Fac'], strength=0.4, distance=0.001))
    rocket.box("RoverBody", (0.0, 0.0, 0.84), (0.26, 0.17, 0.06), sb.painted("RoverWhite", (0.82, 0.82, 0.8), rough=0.35), bev=0.008)
    rocket.box("RoverSolar", (0.0, 0.0, 0.875), (0.3, 0.2, 0.006), sb.mat("RovCells", (0.02, 0.03, 0.08), rough=0.15, coat=1.0), bev=0.001)
    for sy in (-1, 1):
        rocket.rod("Rocker", (0.12, sy * 0.1, 0.8), (-0.02, sy * 0.1, 0.83), 0.006, alu, verts=8)
        rocket.rod("Bogie", (-0.02, sy * 0.1, 0.83), (-0.14, sy * 0.1, 0.8), 0.006, alu, verts=8)
        for x in (0.12, 0.0, -0.13):
            w = sb.prim("cyl", "Wheel", loc=(x, sy * 0.115, 0.78), rot=(math.pi / 2, 0, 0), vertices=32, radius=0.035, depth=0.03, mat=tread)
            sb.bevel(w, 0.004, 2)
            sb.prim("cyl", "Hub", loc=(x, sy * 0.132, 0.78), rot=(math.pi / 2, 0, 0), vertices=16, radius=0.012, depth=0.006, mat=alu)
    rocket.rod("Mast", (0.09, 0.0, 0.87), (0.09, 0.0, 0.99), 0.006, alu, verts=8)
    rocket.box("CamHead", (0.1, 0.0, 1.0), (0.04, 0.06, 0.03), sb.painted("RovGraph", (0.05, 0.05, 0.055), rough=0.4), bev=0.004)
    sb.prim("torus", "LensRing", loc=(0.121, 0.012, 1.0), rot=(0, math.pi / 2, 0), major_radius=0.008, minor_radius=0.0025,
            mat=sb.painted("RovAmber", (0.75, 0.3, 0.04), rough=0.3, coat=0.4))
    for k in range(20):
        p = V((rs.uniform(-0.9, 0.9), rs.uniform(-0.35, 0.35), 0.745))
        if p.length < 0.3:
            continue
        o = sb.prim("cyl", "Part", loc=p, vertices=6, radius=rs.uniform(0.005, 0.02), depth=rs.uniform(0.005, 0.03),
                    mat=rs.choice([M["steel"], M["brass"], M["black"], M["blue_cer"]]))
    # kids + teacher
    kids = [((-0.35, -0.72), 90.0), ((0.4, -0.7), 100.0), ((0.9, 0.2), 180.0), ((-0.9, 0.5), 0.0)]
    for k, ((x, y), rz) in enumerate(kids):
        bm, rig, parts = mhchild.build(name="Kid%d" % k, hair=["short01", "ponytail01", "afro01", "bob01"][k],
                                       top=["male_casualsuit01", "female_casualsuit01", "male_casualsuit03", "male_casualsuit05"][k],
                                       skin=["young_asian_female", "young_caucasian_female", "young_african_male", "young_caucasian_male"][k],
                                       phenotype=dict(age=0.14 + 0.01 * k, gender=[1.0, 0.0, 1.0, 0.0][k],
                                                      race=[dict(asian=0.7, caucasian=0.3, african=0.0), dict(caucasian=0.8, asian=0.1, african=0.1),
                                                            dict(african=0.8, caucasian=0.2, asian=0.0), dict(caucasian=0.5, asian=0.3, african=0.2)][k]))
        folks.arms_down(rig)
        folks.place(rig, (x, y, 0.0), rz + 90.0)
        folks.arms_down(rig, drop=30, elbow=20)
        tgt_r = V((0.0, 0.0, 0.8)) + (V((x, y, 0.8)) - V((0.0, 0.0, 0.8))).normalized() * 0.2
        folks.reach(rig, "R", tgt_r + V((0.0, 0.0, 0.03)), iters=25)
        if k % 2 == 0:
            folks.reach(rig, "L", tgt_r + V((0.06, 0.06, 0.02)), iters=25)
        folks.look_at(rig, V((0.0, 0.0, 0.8)))
        folks.expression(rig, smile=0.6)
    bm, rig, parts = folks.person("researcher", name="Teacher")
    folks.arms_down(rig)
    folks.place(rig, (0.2, 0.85, 0.0), 0.0)
    folks.pose(rig, {"spine01": (18, 0, 0), "spine02": (10, 0, 0)})
    folks.look_at(rig, V((0.0, 0.0, 0.8)))
    folks.expression(rig, smile=0.7)
    tgt = V((0.0, 0.0, 0.95))

    def cp(t):
        return V((sb.lerp(1.9, 1.6, t), sb.lerp(-1.9, -1.6, t), 1.45))

    cam = sb.camera("Cam", loc=cp(0), target=tgt, lens=35, fstop=2.8, clip=(0.01, 200))
    sb.cam_bake(cam, F0, F1, cp, lambda t: tgt, focus=lambda t: (cp(t) - tgt).length)

dbgcam.apply()
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
