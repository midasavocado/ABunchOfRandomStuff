"""s25 LOOKING FARTHER (2340-2430): far from Earth, a great segmented space telescope finishes unfolding: the last
hinged wing of gold-coated mirror segments swings up and latches beside the others; the tiered sunshield glows
softly below, the secondary mirror hangs on its struts, the star field and the Milky Way reflect in the gold.
The camera glides across the mirror face (the amber curve of each segment edge catches the sun).
Original design (hex segments, three-strut secondary, five-layer shield), no markings."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, bmesh
import sb, space, rocket
import timeline as TL
from mathutils import Vector as V, Matrix, Euler, Quaternion

sid = (sb.argv() or ["s25"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()
sb.setup_render(os.environ.get("SB_ENGINE", "CYCLES"), cycles_samples=128, samples=48, mblur=True, shutter=0.5,
                look="AgX - Medium High Contrast", exposure=float(os.environ.get("SB_EXPO", "0.5")))
rs = random.Random(25)
space.starfield(strength=6.0, density=0.7)
# deployment attitude: the sun grazes across the mirror face from the left (the gold segments and their amber
# rings catch it); a dim cool fill from the opposite side stands in for scattered light off the sunshield
# the sun sits where the mirror reflects it toward the lens: as the camera glides, a broad gold sheen sweeps
# across the segments (the mirror otherwise reflects black space and reads as nothing)
sun = sb.sun(float(os.environ.get("S25_SUNEL", "58")), float(os.environ.get("S25_SUNAZ", "10")), energy=0.6,
             color=(1.0, 0.96, 0.9), angle=0.53)
sb.light('AREA', "ShieldBounce", loc=(6, 8, -3), target=(0, 0, 0.5), energy=900.0, color=(0.95, 0.8, 0.75), size=10.0)

gold = sb.mat("MirrorGold", (1.0, 0.62, 0.2), metal=1.0, rough=0.16)
nb = sb.NB(gold)
co = nb.coord('Object')
nb.set('Roughness', nb.maprange(nb.noise(co, scale=60, detail=4).outputs['Fac'], 0.3, 0.7, 0.2, 0.28))
backing = sb.painted("MirrorBack", (0.06, 0.065, 0.07), rough=0.4, wear=0.1)
struts = sb.brushed_metal("Strut", (0.1, 0.1, 0.11), rough=0.35)
kapton = sb.mat("Shield", (0.86, 0.80, 0.86), metal=1.0, rough=0.18, thin_film=520.0)
nb2 = sb.NB(kapton)
c2 = nb2.coord('Object')
cr = nb2.noise(c2, scale=0.8, detail=10, rough=0.7)
nb2.set('Normal', nb2.bump(cr.outputs['Fac'], strength=0.35, distance=0.05))
bus = rocket.foil_mat("BusMLI", (0.80, 0.56, 0.2))
graph = sb.painted("ScopeGraphite", (0.04, 0.042, 0.047), rough=0.4, wear=0.3)
amber = sb.painted("ScopeAmber", (0.75, 0.32, 0.04), rough=0.3, coat=0.3)

# ---- primary mirror: 18 hex segments (flat-to-flat 1.32 m) in two rings around a central hole, on a slightly
# concave surface; the +X and -X side wings (3 segments each) fold
SEG = 1.32
Rh = SEG / math.sqrt(3.0)
FOCAL = 16.0


def hex_seg(name, c, mat):
    bm = bmesh.new()
    pts = [bm.verts.new((c.x + Rh * 0.985 * math.cos(math.radians(60 * k)), c.y + Rh * 0.985 * math.sin(math.radians(60 * k)), 0))
           for k in range(6)]
    f = bm.faces.new(pts)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=6, use_grid_fill=True)
    for v in bm.verts:
        r2 = v.co.x ** 2 + v.co.y ** 2
        v.co.z = r2 / (4 * FOCAL)                      # paraboloid sag
    r = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    for v in r['geom']:
        if isinstance(v, bmesh.types.BMVert):
            v.co.z -= 0.08
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me)
    sb.link_obj(o)
    me.materials.append(mat)
    me.materials.append(backing)
    for p in me.polygons:
        p.material_index = 0 if p.normal.z > 0.9 and p.center.z > -0.01 else 1
    sb.bevel(o, 0.01, 2)
    return o


axial = []
for q in range(-2, 3):
    for r_ in range(-2, 3):
        s_ = -q - r_
        if max(abs(q), abs(r_), abs(s_)) in (1, 2):
            axial.append((q, r_))
root = sb.empty("Telescope")
HX = Rh * 2.25                                   # hinge lines between the inner and outer segment columns
wings = {1: sb.empty("WingR", loc=(HX, 0, 0), parent=root), -1: sb.empty("WingL", loc=(-HX, 0, 0), parent=root)}
for (q, r_) in axial:
    x = Rh * 1.5 * q
    y = SEG * (r_ + q / 2.0)
    c = V((x, y, 0))
    o = hex_seg("Seg_%d_%d" % (q, r_), c, gold)
    if x > Rh * 2.2:
        o.parent = wings[1]; o.location = (-HX, 0, 0)
    elif x < -Rh * 2.2:
        o.parent = wings[-1]; o.location = (HX, 0, 0)
    else:
        o.parent = root
    # amber actuator ring behind each segment (the edges catch the sun from behind)
    ring = sb.prim("torus", "SegRing", loc=c - V((0, 0, 0.12)), major_radius=0.18, minor_radius=0.02, mat=amber)
    ring.parent = o.parent
    if o.parent != root:
        ring.location = (c.x + (-HX if x > 0 else HX), c.y, -0.12)
# backplane truss
for k in range(8):
    a = k * math.pi / 4
    rocket.rod("Back", (0, 0, -0.4), (3.2 * math.cos(a), 3.2 * math.sin(a), -0.25), 0.05, graph, parent=root, verts=10)
# secondary mirror on three struts
SEC = V((0, 0, FOCAL * 0.45))
for k in range(3):
    a = math.radians(90 + 120 * k)
    rocket.rod("SecStrut", (3.1 * math.cos(a), 3.1 * math.sin(a), 0.1), SEC, 0.045, struts, parent=root, verts=10)
sm = sb.prim("cyl", "Secondary", loc=SEC, vertices=6, radius=0.4, depth=0.1, mat=gold, parent=root)
# the wings: rotate from folded (back, 100 deg) to deployed; the -X wing is already open, the +X one moves now
wings[-1].rotation_euler = (0, 0, 0)
LATCH = F0 + 56
for f in range(F0 - 2, F1 + 3):
    k = sb.smoother((f - (F0 - 30)) / (LATCH - (F0 - 30)))
    settle = 0.6 * math.sin(max(0.0, f - LATCH) * 0.8) * math.exp(-max(0.0, f - LATCH) * 0.3)
    wings[1].rotation_euler = (0, math.radians(-(1 - k) * 100.0 + settle), 0)
    wings[1].keyframe_insert("rotation_euler", frame=f)
# ---- sunshield: five tensioned layers (diamond planform) below, spacecraft bus + solar panel under it
for L in range(5):
    z = -2.2 - L * 0.28
    s = 1.0 - L * 0.04
    bm = bmesh.new()
    P = [(-9.0 * s, 0), (0, -5.0 * s), (9.0 * s, 0), (0, 5.0 * s)]
    vs = [bm.verts.new((x, y, z + 0.08 * math.sin(x * 0.7 + L))) for x, y in P]
    bm.faces.new(vs)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=20, use_grid_fill=True)
    for v in bm.verts:
        v.co.z += 0.05 * math.sin(v.co.x * 2.1 + L) * math.cos(v.co.y * 1.7)
    me = bpy.data.meshes.new("Shield%d" % L); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("Shield%d" % L, me); sb.link_obj(o); me.materials.append(kapton); o.parent = root
    for p in me.polygons:
        p.use_smooth = True
rocket.box("Bus", (0, 0, -4.2), (2.2, 2.2, 1.6), bus, parent=root, bev=0.04)
rocket.box("Panel", (0, -3.0, -4.6), (2.0, 3.5, 0.05), sb.mat("PanelCells", (0.02, 0.03, 0.07), rough=0.15, coat=1.0), parent=root, bev=0.01,
           rot=(math.radians(-20), 0, 0))
for k, a in enumerate((0, 120, 240)):
    rocket.rod("Boom", (0, 0, -1.2), (6.0 * math.cos(math.radians(a)), 3.4 * math.sin(math.radians(a)), -2.3), 0.04, graph, parent=root, verts=8)
root.rotation_euler = (math.radians(12), math.radians(-8), 0)
# ---- camera: glide across the mirror face, tilting to find the latching wing, stars reflected in the gold
P0, P1 = V((-6.5, -8.5, 6.8)), V((3.0, -9.5, 5.8))
T0, T1 = V((0.5, 0.0, 0.5)), V((2.5, 0.2, 0.3))
cam = sb.camera("Cam", loc=P0, target=T0, lens=30, fstop=8.0, clip=(0.1, 1e6))
sb.cam_bake(cam, F0, F1, lambda t: P0.lerp(P1, sb.smoother(t)), lambda t: T0.lerp(T1, sb.smoother(t)),
            focus=lambda t: (P0.lerp(P1, sb.smoother(t)) - T0.lerp(T1, sb.smoother(t))).length)
sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
