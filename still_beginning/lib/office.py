"""s06 design studio: desk, monitor (image-sequence screen), keyboard, 3D mouse, printed prototypes, chair,
window wall and soft background dressing. Metres; desk front edge along +X at y=0, user sits at -Y."""
import bpy, math, os
import numpy as np
from mathutils import Vector as V
import sb, props


def oak(name="Oak"):
    m = sb.mat(name, (0.46, 0.33, 0.21), rough=0.42, coat=0.35, coat_rough=0.2)
    nb = sb.NB(m)
    co = nb.coord('Object')
    grain = nb.wave(nb.mapping(co, scale=(1, 12, 1)), scale=3.0, dist=9.0, detail=4, wtype='BANDS', direction='X')
    fine = nb.noise(nb.mapping(co, scale=(1, 60, 1)), scale=40, detail=5)
    g = nb.maprange(grain.outputs['Fac'], 0.2, 0.8)
    col = nb.mix(g, (0.36, 0.24, 0.14, 1), (0.52, 0.38, 0.25, 1))
    col = nb.mix(nb.maprange(fine.outputs['Fac'], 0.4, 0.7, 0, 0.35), col, (0.30, 0.20, 0.12, 1))
    nb.set('Base Color', col)
    nb.set('Normal', nb.bump(fine.outputs['Fac'], strength=0.08, distance=0.0005))
    nb.set('Roughness', nb.maprange(fine.outputs['Fac'], 0.3, 0.7, 0.36, 0.5))
    return m


def screen_mat(name, first_path, f0, n, strength=1.6):
    img = bpy.data.images.load(first_path)
    img.source = 'SEQUENCE'
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.interpolation = 'Cubic'
    iu = tex.image_user
    iu.frame_start = f0
    iu.frame_offset = f0 - 1
    iu.frame_duration = n
    iu.use_auto_refresh = True
    uv = nt.nodes.new('ShaderNodeTexCoord')
    nt.links.new(uv.outputs['UV'], tex.inputs['Vector'])
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = strength
    nt.links.new(tex.outputs['Color'], em.inputs['Color'])
    # glossy anti-glare coat over the panel
    gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Roughness'].default_value = 0.18
    gl.inputs['Color'].default_value = (0.04, 0.04, 0.04, 1)
    ad = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(em.outputs[0], ad.inputs[0]); nt.links.new(gl.outputs[0], ad.inputs[1])
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(ad.outputs[0], out.inputs[0])
    return m, em


def monitor(screen_m, loc=(0, 0.42, 0.0), w=0.708, h=0.398):
    """32-inch class display: 6 mm bezel, graphite back shell, aluminium stand. loc = desk-surface centre under it."""
    x, y, z = loc
    graph = sb.painted("MonGraphite", (0.035, 0.036, 0.04), rough=0.4, coat=0.1)
    alu = sb.brushed_metal("MonAlu", (0.62, 0.63, 0.65), rough=0.3, aniso=0.5)
    cz = z + 0.155 + h / 2
    body = sb.prim("cube", "MonBody", loc=(x, y + 0.012, cz), scale=(w / 2 + 0.008, 0.012, h / 2 + 0.009), mat=graph)
    sb.bevel(body, 0.004, 3)
    scr = sb.prim("plane", "Screen", loc=(x, y - 0.0005, cz), rot=(math.pi / 2, 0, 0), scale=(w / 2, h / 2, 1), mat=screen_m)
    # UV already 0..1 on the default plane
    neck = sb.prim("cube", "MonNeck", loc=(x, y + 0.06, z + 0.16), scale=(0.035, 0.012, 0.16), mat=alu)
    neck.rotation_euler = (math.radians(-12), 0, 0)
    sb.bevel(neck, 0.004)
    foot = sb.prim("cube", "MonFoot", loc=(x, y + 0.05, z + 0.006), scale=(0.13, 0.10, 0.006), mat=alu)
    sb.bevel(foot, 0.004, 3)
    return scr, cz


def desk(width=1.6, depth=0.8, top_z=0.74):
    top = sb.prim("cube", "DeskTop", loc=(0, depth / 2 - 0.06, top_z - 0.013), scale=(width / 2, depth / 2, 0.013), mat=oak())
    sb.bevel(top, 0.003, 3)
    steel = sb.painted("DeskLeg", (0.05, 0.05, 0.055), rough=0.4)
    for sx in (-1, 1):
        leg = sb.prim("cube", "DeskLeg", loc=(sx * (width / 2 - 0.08), depth / 2 - 0.06, top_z / 2), scale=(0.03, depth / 2 - 0.05, top_z / 2 - 0.013), mat=steel)
    return top_z


def space_mouse(loc):
    graph = sb.painted("SMBase", (0.05, 0.05, 0.055), rough=0.5)
    cap = sb.rubber("SMCap", (0.09, 0.09, 0.1), rough=0.55)
    b = sb.lathe("SpaceMouseBase", [(0.0, 0.0), (0.042, 0.0), (0.045, 0.006), (0.040, 0.022), (0.030, 0.028), (0.0, 0.029)], segs=64, mat=graph)
    b.location = loc
    c = sb.lathe("SpaceMouseCap", [(0.0, 0.028), (0.028, 0.028), (0.030, 0.034), (0.030, 0.050), (0.027, 0.056), (0.0, 0.058)], segs=64, mat=cap)
    c.location = loc
    ring = sb.lathe("SpaceMouseRing", [(0.0301, 0.030), (0.0306, 0.030), (0.0306, 0.032), (0.0301, 0.032)], segs=96, mat=props.amber_anodized("SMAmber"))
    ring.location = loc
    return c


def keyboard(loc):
    alu = sb.brushed_metal("KbAlu", (0.66, 0.67, 0.69), rough=0.3)
    keym = sb.mat("KbKey", (0.08, 0.08, 0.085), rough=0.5)
    kb = sb.prim("cube", "Keyboard", loc=loc, scale=(0.21, 0.065, 0.006), mat=alu)
    sb.bevel(kb, 0.003)
    cen, half = [], []
    for r in range(5):
        for c in range(15):
            cen.append((loc[0] - 0.195 + c * 0.0279, loc[1] - 0.052 + r * 0.026, loc[2] + 0.0075))
            half.append((0.0118, 0.0112, 0.002))
    import chip
    Vv, F = chip._boxes(cen, half)
    o = chip._mesh("Keys", Vv, F, keym)
    sb.bevel(o, 0.0015, 2)
    return kb


def room(top_z):
    """Window wall behind the monitor (soft overcast daylight), side wall with shelves, floor."""
    fl = sb.prim("plane", "Floor", loc=(0, 0.5, 0), scale=(5, 5, 1), mat=sb.concrete("Floor", (0.32, 0.31, 0.30), scale=1.5, rough=0.6, stains=0.1))
    wall = sb.prim("plane", "BackWall", loc=(0, 1.65, 1.5), rot=(math.pi / 2, 0, 0), scale=(4, 1.5, 1), mat=sb.mat("Wall", (0.58, 0.58, 0.57), rough=0.85))
    # window: bright but not blown; mullions
    win = sb.prim("plane", "Window", loc=(-1.05, 1.64, 1.45), rot=(math.pi / 2, 0, 0), scale=(0.75, 0.85, 1),
                  mat=sb.emit_mat("WinM", (0.78, 0.86, 1.0), 1.2))
    mull = sb.painted("Mullion", (0.1, 0.1, 0.11), rough=0.4)
    for xx in (-1.8, -1.05, -0.3):
        sb.prim("cube", "Mull", loc=(xx, 1.63, 1.45), scale=(0.02, 0.02, 0.86), mat=mull)
    sb.prim("cube", "MullH", loc=(-1.05, 1.63, 1.2), scale=(0.76, 0.02, 0.02), mat=mull)
    # shelves with boxes / binders / a plant silhouette on the right, soft background
    shelf = sb.painted("Shelf", (0.62, 0.62, 0.6), rough=0.5)
    rng = sb.rng(3)
    for zz in (1.05, 1.45, 1.85):
        sb.prim("cube", "Shelf", loc=(0.95, 1.52, zz), scale=(0.6, 0.12, 0.012), mat=shelf)
        x = 0.4
        while x < 1.5:
            wdt = rng.uniform(0.03, 0.09); hgt = rng.uniform(0.14, 0.3)
            col = rng.choice([(0.12, 0.13, 0.15), (0.55, 0.52, 0.48), (0.22, 0.28, 0.35), (0.7, 0.68, 0.64)])
            sb.prim("cube", "Book", loc=(x, 1.52, zz + 0.012 + hgt / 2), scale=(wdt / 2, 0.1, hgt / 2), mat=sb.mat("Bk", col, rough=0.6))
            x += wdt + rng.uniform(0.002, 0.03)
    return win
