"""s10 POWER inserts (first drop): s10a machined copper busbars + bolted joint (2 beats),
s10b cryogenic cooling assembly (2 beats), s10c D-shaped superconducting coil reveal (4 beats)."""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb
import energy as E
import timeline as TL
import numpy as np
from mathutils import Vector as V, Quaternion, Matrix

sid = (sb.argv() or ["s10a"])[0]
_, F0, F1, _ = TL.shot(sid)
sc = sb.reset()


# =========================================================================== s10a  BUSBARS
def build_busbars():
    sb.setup_render("EEVEE", samples=None, mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=0.0)
    cu = E.copper("Cu", rough=0.2, tarnish=0.16, lines_dir='X')
    cuv = E.copper("CuV", rough=0.22, tarnish=0.2, lines_dir='Z')
    tin = E.copper("CuSp", rough=0.26, tarnish=0.5, lines_dir='Z')
    bolt_m = E.zinc("Zinc", rough=0.2)
    ins = E.g10("G10", color=(0.45, 0.36, 0.14))
    sleeve = sb.mat("Sleeve", (0.025, 0.026, 0.03), rough=0.5, coat=0.2, coat_rough=0.3)
    cab = E.painted_steel("Cabinet", (0.06, 0.062, 0.066), rough=0.6, wear=0.2)
    T, W = 0.010, 0.100            # bar thickness (Y), height (Z)
    GAP = 0.010
    # ---- main horizontal laminated pack (3 bars along X), front face at y=0
    L = 1.6
    for k in range(3):
        y = T / 2 + k * (T + GAP)
        b = E.box(f"Main{k}", (L, T, W), loc=(-0.25, y, 0), mat=cu, bev=0.0012, bev_seg=3)
    # spacer blocks between laminations at the joint (tin-plated copper)
    for k in range(2):
        y = T + GAP / 2 + k * (T + GAP)
        E.box(f"Spacer{k}", (0.11, GAP, W), loc=(0.35, y, 0), mat=tin, bev=0.0008)
    # ---- vertical riser pack crossing in front of the joint (bolted), bars run along Z
    RZ0, RZ1 = -0.05, 0.9
    for k in range(2):
        y = -T / 2 - k * (T + GAP) - 0.0002
        E.box(f"Riser{k}", (W, T, RZ1 - RZ0), loc=(0.35, y, (RZ0 + RZ1) / 2), mat=cuv, bev=0.0012)
    E.box("RSpacer", (W, GAP, 0.11), loc=(0.35, -T - GAP / 2, 0), mat=tin, bev=0.0008)
    # ---- 2x2 bolts through the joint (heads on the front -Y face)
    bolt = E.hex_bolt_mesh("BoltSrc", d=0.012, mats=[bolt_m])
    yf = -2 * T - GAP - 0.0008
    for i, (dx, dz) in enumerate(((-0.025, -0.025), (0.025, -0.025), (-0.025, 0.025), (0.025, 0.025))):
        E.place_linked(bolt, (0.35 + dx, yf, dz), direction=(0, -1, 0), spin=math.radians(7 + 11 * i), name=f"Bolt{i}")
    bolt.hide_render = True; bolt.hide_viewport = True
    # thread stubs + nuts on the back side
    nut = E.nut_mesh("NutSrc", d=0.012)
    nut.data.materials.append(bolt_m)
    yb = 2 * T + 2 * GAP + T + 0.0025
    for i, (dx, dz) in enumerate(((-0.025, -0.025), (0.025, -0.025), (-0.025, 0.025), (0.025, 0.025))):
        E.place_linked(nut, (0.35 + dx, yb, dz), direction=(0, 1, 0), spin=math.radians(3 + 17 * i))
        E.cyl(f"Thread{i}", 0.0058, 0.012, loc=(0.35 + dx, yb + 0.012, dz), rot=(math.pi / 2, 0, 0), mat=bolt_m, verts=24)
    nut.hide_render = True
    # a second joint further left (bolted splice between two main-bar lengths)
    for i, (dx, dz) in enumerate(((-0.03, -0.025), (0.03, -0.025), (-0.03, 0.025), (0.03, 0.025))):
        E.place_linked(bolt, (-0.55 + dx, -0.0008 - 0.003, dz), direction=(0, -1, 0), spin=math.radians(23 * i), name=f"SBolt{i}")
    E.box("SplicePlate", (0.14, 0.003, W), loc=(-0.55, -0.0015, 0), mat=cuv, bev=0.0006)
    # ---- flexible laminated link (stacked foils in an S-bend) from the riser to a terminal on the right
    foils = 14
    for k in range(foils):
        off = (k - foils / 2) * 0.0009
        pts = [(0.35 + W / 2 - 0.01, -0.05 + off, 0.30), (0.45, -0.05 + off, 0.30), (0.52, -0.05 + off + 0.02, 0.24),
               (0.58, -0.05 + off + 0.04, 0.18), (0.66, -0.05 + off + 0.04, 0.18)]
        E.sweep(f"Foil{k}", pts, radius=1.0, profile=[(-0.00035, -0.04), (0.00035, -0.04), (0.00035, 0.04), (-0.00035, 0.04)],
                mat=cu, sub=10, caps=True, up=(0, 1, 0))
    # ---- epoxy standoff insulators holding the main pack (slotted clamp blocks)
    for x in (-1.0, -0.2, 0.9):
        E.box(f"InsBlk{x}", (0.06, 0.09, 0.04), loc=(x, 0.025, -W / 2 - 0.02), mat=ins, bev=0.003)
        E.box(f"InsTop{x}", (0.06, 0.09, 0.03), loc=(x, 0.025, W / 2 + 0.015), mat=ins, bev=0.003)
        for sx in (-1, 1):
            E.cyl(f"InsRod{x}{sx}", 0.004, 0.2, loc=(x + sx * 0.022, 0.025, 0.0), mat=bolt_m, verts=16)
        E.cyl(f"Post{x}", 0.03, 0.3, loc=(x, 0.025, -W / 2 - 0.19), mat=ins, verts=32, bev=0.002)
    # ---- the other two phases behind/above, sleeved in graphite heat-shrink with bare joints
    for ph, (dy, dz) in enumerate(((0.18, 0.16), (0.36, 0.32))):
        for k in range(3):
            y = dy + T / 2 + k * (T + GAP)
            E.box(f"Ph{ph}Bar{k}", (L, T, W), loc=(-0.25, y, dz), mat=cu, bev=0.0012)
            for seg in ((-1.05, -0.62), (-0.48, 0.25), (0.45, 0.55)):
                cx = (seg[0] + seg[1]) / 2
                E.box(f"Ph{ph}Sl{k}{seg[0]}", (seg[1] - seg[0], T + 0.0012, W + 0.0012), loc=(cx, y, dz), mat=sleeve, bev=0.001)
        for i, (dx, dz2) in enumerate(((-0.025, -0.025), (0.025, -0.025), (-0.025, 0.025), (0.025, 0.025))):
            E.place_linked(bolt, (0.35 + dx, dy - 0.001, dz + dz2), direction=(0, -1, 0), spin=math.radians(40 * i + ph * 13))
    # ---- cabinet: back panel with perforations + side, floor
    back = E.box("Back", (3.0, 0.02, 2.0), loc=(0, 0.75, 0.3), mat=cab)
    E.box("Floor", (3.0, 1.5, 0.02), loc=(0, 0.2, -0.45), mat=E.painted_steel("FloorP", (0.02, 0.02, 0.022), rough=0.7))
    E.box("Side", (0.02, 1.5, 2.0), loc=(1.25, 0.2, 0.3), mat=cab)
    # support rail (C-channel) holding the insulator posts
    E.box("Rail", (2.6, 0.06, 0.03), loc=(0, 0.025, -0.36), mat=E.painted_steel("RailP", (0.08, 0.08, 0.085), rough=0.5))
    # ---- lighting: dark room, a warm overhead strip (reads as a crisp streak on the flat faces), a low warm
    # card for the bolt heads, a faint cool fill. Metals only show what they reflect -> keep most of the env dark.
    sb.world_color((0.004, 0.004, 0.005), 1.0)
    key = sb.light('AREA', "Key", loc=(0.1, -0.35, 0.65), target=(0.3, 0.0, 0.0), energy=40, color=(1.0, 0.80, 0.6), size=1.2, size_y=0.05)
    key.visible_glossy = False
    rim = sb.light('AREA', "Rim", loc=(1.0, 0.3, 0.25), target=(0.35, 0, 0), energy=12, color=(1.0, 0.72, 0.5), size=0.3)
    fill = sb.light('AREA', "Fill", loc=(-0.6, -0.8, -0.1), target=(0.0, 0, 0), energy=2, color=(0.65, 0.75, 1.0), size=1.2)
    fill.visible_glossy = False
    E.softbox("SB1", (0.1, -0.55, 0.45), (0.2, 0, 0), size=(2.2, 0.07), strength=14.0, color=(1.0, 0.82, 0.62))
    E.softbox("SB2", (0.4, -0.7, -0.05), (0.3, 0, 0), size=(1.4, 0.04), strength=6.0, color=(1.0, 0.7, 0.45))
    E.softbox("SB4", (0.05, -0.9, 0.30), (0.3, 0, 0), size=(0.5, 0.25), strength=1.2, color=(1.0, 0.9, 0.8))
    E.softbox("SB3", (-0.5, -0.4, 0.3), (0.3, 0, 0), size=(0.5, 0.5), strength=0.35, color=(0.6, 0.7, 0.9))
    E.sphere_probe((0.25, -0.12, 0.05), radius=1.5)
    # ---- camera: 3/4 from above-front; fast lateral slide along the laminated pack, easing onto the bolted joint
    cam = E.dof_cam("Cam", 60, 5.6)
    sc.eevee.bokeh_max_size = 300
    def pos(t):
        k = sb.ease_out(t, 2.4)
        return V((sb.lerp(-0.26, 0.24, k), sb.lerp(-0.30, -0.29, k), sb.lerp(0.19, 0.105, k)))
    def tgt(t):
        k = sb.ease_out(t, 2.4)
        return V((sb.lerp(0.02, 0.37, k), sb.lerp(0.02, -0.03, k), sb.lerp(0.0, 0.002, k)))
    def foc(t):
        # focus pulls from the near laminations onto the bolt heads (distance along the view axis)
        k = sb.smooth(sb.remap(t, 0.2, 0.8))
        p = pos(t); fwd = (tgt(t) - p).normalized()
        fp = V((0.10, -0.01, 0.05)).lerp(V((0.35, -0.040, 0.0)), k)
        return (fp - p).dot(fwd)
    sb.cam_bake(cam, F0, F1, pos, tgt, focus=foc)
    return cam


# =========================================================================== s10b  COOLING ASSEMBLY
def fin_pack(name, x0, nf, pitch, th, y0, y1, z0, z1, mat):
    verts, faces = [], []
    for k in range(nf):
        x = x0 + k * pitch
        b = len(verts)
        for dx in (-th / 2, th / 2):
            for y in (y0, y1):
                for z in (z0, z1):
                    verts.append((x + dx, y, z))
        faces += [(b, b + 1, b + 3, b + 2), (b + 4, b + 6, b + 7, b + 5), (b, b + 4, b + 5, b + 1),
                  (b + 2, b + 3, b + 7, b + 6), (b, b + 2, b + 6, b + 4), (b + 1, b + 5, b + 7, b + 3)]
    return E.mesh_np(name, verts, faces, mat=mat, smooth=False)


def build_cooling():
    sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=0.0)
    ss = E.stainless("SS", rough=0.3, brushed=False, grime=0.1)
    ssb = E.stainless("SSbright", rough=0.16, brushed=True, dir_scale=(1, 1, 80))
    man_m = E.frost("ManFrost", base=(0.60, 0.60, 0.61), coverage=0.9, radial=((-0.09, 0.0, 0.0), 0.012, 0.075))
    frost_m = E.frost("Frost", base=(0.8, 0.82, 0.85), coverage=1.3)
    cu = E.copper("CuT", rough=0.25, tarnish=0.2, lines_dir='X')
    alu = E.stainless("AluFin", color=(0.78, 0.79, 0.80), rough=0.22, brushed=True, dir_scale=(1, 80, 1))
    hose_m = E.braided("Braid", color=(0.56, 0.56, 0.57), rough=0.22, pitch=380, around=24)
    amber = __import__("props").amber_anodized("AmberF")
    water = sb.mat("Water", (1, 1, 1), rough=0.02, transmission=1.0, ior=1.333)
    black = E.anodized("BlackAnod", (0.03, 0.032, 0.036), rough=0.4)
    rnd = random.Random(5)
    # ---- manifold block (bead-blasted stainless), frost bloom spreading from the cold inlet (-X)
    E.box("Manifold", (0.18, 0.06, 0.06), loc=(0, 0, 0), mat=man_m, bev=0.002)
    E.box("Plate", (0.34, 0.12, 0.008), loc=(0.02, 0.01, -0.034), mat=black, bev=0.0015)
    bolt_m = E.zinc("ZincB", rough=0.22)
    bolt = E.hex_bolt_mesh("BoltSrc", d=0.008, mats=[bolt_m]); bolt.hide_render = True
    for x in (-0.13, 0.155):
        for y in (-0.03, 0.05):
            E.place_linked(bolt, (x, y, -0.030), direction=(0, 0, 1), spin=x * 40 + y * 90)
    # ---- 4 top ports: fitting nut + ferrule + crimped collar + braided hose sweeping up and away
    nut = E.nut_mesh("FitNut", d=0.016); nut.data.materials.append(ssb); nut.hide_render = True
    for i, x in enumerate((-0.06, -0.02, 0.02, 0.06)):
        E.cyl(f"Boss{i}", 0.012, 0.004, loc=(x, 0, 0.032), mat=ss, verts=48)
        E.place_linked(nut, (x, 0, 0.034), direction=(0, 0, 1), spin=math.radians(9 * i + 4))
        E.cyl(f"Ferrule{i}", 0.0105, 0.004, loc=(x, 0, 0.0495), mat=(amber if i == 1 else ssb), verts=48, bev=0.0006)
        prof = [(0.0, 0.0), (0.0112, 0.0)]
        for k in range(6):
            z0 = 0.004 + k * 0.0048
            prof += [(0.0112, z0), (0.0106, z0 + 0.0012), (0.0106, z0 + 0.0024), (0.0112, z0 + 0.0036)]
        prof += [(0.0112, 0.034), (0.0104, 0.0355), (0.0096, 0.0355)]
        E.lathe_obj(f"Collar{i}", prof, segs=64, mat=ssb, loc=(x, 0, 0.052))
        z0 = 0.086
        sway = (i - 1.5)
        pts = [(x, 0, z0), (x, 0, z0 + 0.05), (x + sway * 0.012, 0.03, z0 + 0.12), (x + sway * 0.03, 0.12, z0 + 0.18),
               (x + sway * 0.05, 0.28, z0 + 0.21), (x + sway * 0.07, 0.5, z0 + 0.22)]
        E.sweep(f"Hose{i}", pts, radius=0.0095, segs=40, mat=hose_m, sub=14)
    # ---- cold inlet: copper line leaving -X, curving down toward the camera; thick displaced hoarfrost
    pts = [(-0.09, 0, 0), (-0.13, 0, 0), (-0.17, -0.02, -0.01), (-0.20, -0.08, -0.02), (-0.22, -0.18, -0.02), (-0.24, -0.35, -0.02)]
    E.sweep("ColdLine", pts, radius=0.0065, segs=32, mat=cu, sub=10)
    fr = E.sweep("FrostShell", pts, radius=0.0098, segs=72, mat=frost_m, sub=36, rad_fn=lambda u: 0.85 + 0.2 * math.sin(u * 23) ** 2)
    tex = bpy.data.textures.new("FrostTex", 'CLOUDS'); tex.noise_scale = 0.002; tex.noise_depth = 5
    d = fr.modifiers.new("Disp", 'DISPLACE'); d.texture = tex; d.strength = 0.0035; d.mid_level = 0.5; d.texture_coords = 'LOCAL'
    E.cyl("InletNut", 0.013, 0.012, loc=(-0.096, 0, 0), rot=(0, math.pi / 2, 0), mat=man_m, verts=6)
    # ---- condensation droplets on the warmer end of the manifold's front face
    drop = sb.prim("sphere", "DropSrc", segments=24, ring_count=12, radius=1.0, mat=water)
    drop.hide_render = True
    for n in range(160):
        x = -0.035 + 0.12 * rnd.random() ** 1.4; z = rnd.uniform(-0.028, 0.028)
        r = 0.0006 + 0.0017 * rnd.random() ** 1.6
        o = drop.copy(); sb.link_obj(o); o.hide_render = False
        o.location = (x, -0.030 - r * 0.15, z)
        o.scale = (r, r * 0.45, r * (1.0 + 0.15 * rnd.random()))
    # ---- finned heat exchanger in the left foreground (camera pushes past it): fin edges face the lens
    # horizontal fin plates stacked in Z, vertical copper tubes piercing them
    fverts, ffaces = [], []
    FX0, FX1, FY0, FY1 = -0.60, -0.29, -0.53, -0.41
    NFZ, fp_, th = 90, 0.0032, 0.0003
    for k in range(NFZ):
        z = -0.05 + k * fp_
        b = len(fverts)
        for x in (FX0, FX1):
            for y in (FY0, FY1):
                for dz in (-th / 2, th / 2):
                    fverts.append((x, y, z + dz))
        ffaces += [(b, b + 2, b + 3, b + 1), (b + 4, b + 5, b + 7, b + 6), (b, b + 1, b + 5, b + 4),
                   (b + 2, b + 6, b + 7, b + 3), (b, b + 4, b + 6, b + 2), (b + 1, b + 3, b + 7, b + 5)]
    E.mesh_np("Fins", fverts, ffaces, mat=alu, smooth=False)
    for i in range(8):
        for col in range(2):
            x = FX0 + 0.02 + i * 0.038 + col * 0.019; y = FY0 + 0.03 + col * 0.06
            E.cyl(f"Tube{i}{col}", 0.0048, NFZ * fp_ + 0.04, loc=(x, y, -0.05 + NFZ * fp_ / 2), mat=cu, verts=24)
    galv = E.painted_steel("Galv", (0.30, 0.31, 0.32), rough=0.5, wear=0.2)
    E.box("CoilSideR", (0.002, FY1 - FY0 + 0.01, 0.33), loc=(FX1 + 0.003, (FY0 + FY1) / 2, 0.095), mat=galv)
    # ---- background: dark plant room, bench, a second manifold, distant warm practicals (bokeh)
    E.box("Wall", (4, 0.05, 3), loc=(0, 1.2, 0.5), mat=E.painted_steel("WallP", (0.02, 0.02, 0.022), rough=0.8))
    E.box("Bench", (1.6, 1.2, 0.03), loc=(0.1, 0.3, -0.055), mat=E.painted_steel("BenchP", (0.03, 0.031, 0.034), rough=0.55))
    E.box("Manifold2", (0.18, 0.06, 0.06), loc=(0.35, 0.55, 0.0), mat=black, bev=0.002)
    # plant-room pipe rack behind the bench: lagged cryo lines (aluminium cladding with band clamps), bare
    # stainless runs with flanges, valve handwheels, a pressure gauge, cable tray; reads as shapes + highlights
    clad = E.stainless("Lagging", color=(0.70, 0.71, 0.72), rough=0.35, brushed=False, grime=0.3, scale=4.0)
    band = E.stainless("Band", rough=0.2)
    wheel_m = E.painted_steel("Wheel", (0.05, 0.05, 0.055), rough=0.5, wear=0.3)
    for j, (y, z, r, m_) in enumerate(((0.95, 0.22, 0.045, clad), (0.95, 0.36, 0.03, ss), (1.02, 0.50, 0.055, clad),
                                         (0.88, 0.62, 0.022, ss), (1.05, 0.78, 0.04, clad))):
        E.cyl(f"Pipe{j}", r, 3.4, loc=(0.0, y, z), rot=(0, math.pi / 2, 0), mat=m_, verts=40)
        for x in np.arange(-1.6, 1.7, 0.45 if m_ is clad else 0.9):
            E.cyl(f"PBand{j}", r * 1.03, 0.012 if m_ is clad else 0.02, loc=(x + 0.07 * j, y, z), rot=(0, math.pi / 2, 0),
                  mat=band if m_ is clad else ss, verts=40)
    for j, x in enumerate((-0.75, 0.2, 0.95)):
        E.cyl(f"Riser{j}", 0.028, 1.2, loc=(x, 0.82, 0.4), mat=clad, verts=32)
        E.cyl(f"VStem{j}", 0.006, 0.12, loc=(x, 0.78, 0.36), rot=(math.pi / 2, 0, 0), mat=ss, verts=12)
        E.lathe_obj(f"Wheel{j}", [(0.030, -0.004), (0.036, -0.004), (0.036, 0.004), (0.030, 0.004)], segs=48, mat=wheel_m,
                    loc=(x, 0.715, 0.36), rot=(math.pi / 2, 0, 0))
    E.cyl("Gauge", 0.05, 0.03, loc=(0.55, 0.78, 0.42), rot=(math.pi / 2, 0, 0), mat=ss, verts=48)
    E.cyl("GaugeFace", 0.044, 0.002, loc=(0.55, 0.764, 0.42), rot=(math.pi / 2, 0, 0), mat=sb.mat("GaugeF", (0.85, 0.85, 0.83), rough=0.3), verts=48)
    for x in np.arange(-1.6, 1.7, 0.6):
        E.box("Strut", (0.04, 0.04, 1.2), loc=(x, 1.12, 0.4), mat=E.painted_steel("StrutP", (0.35, 0.36, 0.38), rough=0.5, wear=0.2))
    E.box("Tray", (3.4, 0.25, 0.02), loc=(0, 0.95, 0.95), mat=band)
    for i in range(9):
        sb.prim("sphere", f"Prac{i}", loc=(-1.0 + i * 0.3 + rnd.uniform(-0.1, 0.1), 1.15, 0.2 + rnd.uniform(0, 0.6)), radius=0.01,
                mat=sb.emit_mat(f"PracM{i}", (1.0, 0.6, 0.28) if i % 3 else (0.7, 0.8, 1.0), 40))
    # ---- lighting: raking cool key from the left, warm amber rim from behind-right, dark room
    sb.world_color((0.003, 0.0035, 0.0045), 1.0)
    sb.light('AREA', "Key", loc=(-0.6, -0.05, 0.35), target=(-0.05, 0, 0.03), energy=11, color=(0.78, 0.87, 1.0), size=0.3)
    sb.light('AREA', "Rim", loc=(0.40, 0.45, 0.30), target=(-0.05, 0, 0.06), energy=30, color=(1.0, 0.60, 0.28), size=0.2)
    sb.light('AREA', "PlantRoom", loc=(0.3, 0.45, 1.1), target=(0.0, 1.0, 0.4), energy=9, color=(1.0, 0.78, 0.55), size=1.2)
    sb.light('AREA', "FinGlint", loc=(-0.45, -0.6, 0.45), target=(-0.45, -0.47, 0.08), energy=1.5, color=(0.8, 0.88, 1.0), size=0.25)
    fill = sb.light('AREA', "Fill", loc=(-0.1, -0.7, 0.1), target=(0, 0, 0.03), energy=1.2, color=(0.8, 0.85, 1.0), size=1.0)
    fill.visible_glossy = False
    E.softbox("SB1", (-0.25, -0.45, 0.40), (0, 0, 0.05), size=(0.6, 0.05), strength=10.0, color=(0.85, 0.9, 1.0))
    E.softbox("SB2", (0.45, -0.25, 0.25), (0, 0, 0.05), size=(0.05, 0.6), strength=8.0, color=(1.0, 0.62, 0.3))
    E.sphere_probe((-0.05, -0.08, 0.06), radius=1.2)
    # ---- camera: push forward past the fin pack (wipes out frame-left), onto the frosted inlet + fittings
    cam = E.dof_cam("Cam", 65, 4.5)
    sc.eevee.bokeh_max_size = 300
    def pos(t):
        k = t * 0.6 + sb.ease_out(t, 2.0) * 0.4
        return V((-0.36, -0.74, 0.12)).lerp(V((-0.15, -0.34, 0.105)), k)
    def tgt(t):
        k = t * 0.6 + sb.ease_out(t, 2.0) * 0.4
        return V((-0.10, 0.0, 0.03)).lerp(V((-0.03, 0.0, 0.035)), k)
    def foc(t):
        p = pos(t); fwd = (tgt(t) - p).normalized()
        fp = V((-0.30, -0.46, 0.08)).lerp(V((-0.05, -0.012, 0.04)), sb.smooth(sb.remap(t, 0.0, 0.35)))
        return (fp - p).dot(fwd)
    sb.cam_bake(cam, F0, F1, pos, tgt, focus=foc)


# =========================================================================== s10c  D-SHAPED SUPERCONDUCTING COIL
def build_coil():
    sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast", exposure=0.0,
                    volumes=True, vol_tile=4, vol_samples=64)
    sc.eevee.volumetric_end = 120
    rnd = random.Random(3)
    bolt_m = E.zinc("ZincC", rough=0.25)
    bolt = E.hex_bolt_mesh("BoltSrcC", d=0.036, mats=[bolt_m]); bolt.hide_render = True
    case = E.stainless("Case", color=(0.42, 0.42, 0.43), rough=0.38, brushed=False, grime=0.4, scale=0.08)
    # hero coil: stands upright on its pedestal, inboard leg toward -X; exposed winding pack on the upper outer arc
    root, P = E.tf_coil("Hero", window=(0.20, 0.30), case_mat=case, bolt_src=bolt)
    zmin = float(P[:, 2].min()); ZC = 0.95 - zmin + 1.15 / 2
    root.location = (-3.6, 0, ZC)
    # two more coils further down the hall (assembly line), and one lying on a transport frame
    for k, (x, y, rz) in enumerate(((-3.6, 9.0, 0.0), (-3.6, 18.0, 0.0))):
        r2, _ = E.tf_coil(f"Coil{k}", case_mat=case, bolt_src=bolt, detail=(k == 0))
        r2.location = (x, y, ZC)
    # pedestal / gravity support under each coil
    paint = E.painted_steel("Paint", (0.07, 0.075, 0.08), rough=0.45, wear=0.25, scale=0.3)
    amber = __import__("props").amber_anodized("AmberC")
    for y in (0.0, 9.0, 18.0):
        E.box(f"PedA{y}", (2.2, 1.6, 0.55), loc=(-3.2, y, 0.275), mat=paint, bev=0.02)
        E.box(f"PedB{y}", (1.2, 1.1, 0.35), loc=(-3.2, y, 0.72), mat=paint, bev=0.02)
        E.box(f"PedC{y}", (2.0, 1.4, 0.55), loc=(2.6, y, 0.275), mat=paint, bev=0.02)
        E.box(f"PedD{y}", (0.9, 1.0, 0.35), loc=(2.6, y, 0.72), mat=paint, bev=0.02)
    # ---- hall: epoxy-coated concrete floor with painted bay lines, steel columns, crane girder, clerestory
    floor = sb.concrete("Floor", (0.30, 0.30, 0.29), scale=0.6, rough=0.45, stains=0.35)
    fb = sb.NB(floor); fb.bsdf.inputs['Coat Weight'].default_value = 0.35; fb.bsdf.inputs['Coat Roughness'].default_value = 0.12
    E.box("FloorB", (80, 120, 0.2), loc=(0, 20, -0.1), mat=floor)
    line_m = sb.mat("Lines", (0.55, 0.36, 0.12), rough=0.5)
    for x in (-8.0, 8.0):
        E.box(f"Line{x}", (0.12, 60, 0.004), loc=(x, 15, 0.002), mat=line_m)
    col_m = E.painted_steel("Col", (0.20, 0.21, 0.22), rough=0.5, wear=0.2, scale=0.2)
    for y in [-12 + 8 * i for i in range(8)]:
        for x in (-14.0, 14.0):
            E.box(f"ColW{x}{y}", (0.45, 0.02, 24), loc=(x, y, 12), mat=col_m)
            E.box(f"ColF1{x}{y}", (0.02, 0.45, 24), loc=(x - 0.22, y, 12), mat=col_m)
            E.box(f"ColF2{x}{y}", (0.02, 0.45, 24), loc=(x + 0.22, y, 12), mat=col_m)
        # roof truss
        E.box(f"Truss{y}", (28.5, 0.3, 1.4), loc=(0, y, 24.5), mat=col_m)
    # crane runway beams + bridge girder with hook block
    for x in (-13.2, 13.2):
        E.box(f"Runway{x}", (0.5, 70, 1.0), loc=(x, 15, 18.5), mat=col_m)
    E.box("Bridge", (27, 1.2, 1.6), loc=(0, 4.5, 19.3), mat=E.painted_steel("Bridge", (0.62, 0.42, 0.14), rough=0.45, wear=0.3, scale=0.2))
    E.box("Trolley", (2.2, 2.4, 1.2), loc=(-1.0, 4.5, 18.2), mat=paint, bev=0.05)
    for dx in (-0.3, 0.3):
        E.cyl(f"Rope{dx}", 0.025, 9.0, loc=(-1.0 + dx, 4.5, 13.1), mat=bolt_m, verts=12)
    E.box("Hook", (0.8, 0.5, 1.0), loc=(-1.0, 4.5, 8.3), mat=paint, bev=0.05)
    # walls: left wall with tall windows (sun comes through), right wall solid, back wall far
    wall_m = E.painted_steel("Wall", (0.36, 0.37, 0.37), rough=0.6, wear=0.35, scale=0.15)
    wb = sb.NB(wall_m)
    corr = wb.wave(wb.coord('Object'), scale=6.0, wtype='BANDS', direction='Y', profile='SIN')
    wb.set('Normal', wb.bump(corr.outputs['Fac'], strength=0.6, distance=0.03))
    for i in range(-2, 11):
        y = -12 + 8 * i
        E.box(f"WallLow{i}", (0.3, 8, 4), loc=(-14.4, y + 4, 2.0), mat=wall_m)
        E.box(f"WallTop{i}", (0.3, 8, 4), loc=(-14.4, y + 4, 22.5), mat=wall_m)
        for k in range(3):   # window mullions
            E.box(f"Mull{i}{k}", (0.35, 0.25, 16.5), loc=(-14.4, y + 1 + k * 2.7, 12.25), mat=col_m)
        for tz in (9.0, 14.5):
            E.box(f"Trans{i}{tz}", (0.35, 8, 0.2), loc=(-14.4, y + 4, tz), mat=col_m)
    E.box("WallR", (0.3, 120, 26), loc=(14.4, 20, 13), mat=wall_m)
    E.box("WallBack", (30, 0.3, 26), loc=(0, 60, 13), mat=wall_m)
    E.box("WallFront", (30, 0.3, 26), loc=(0, -30, 13), mat=wall_m)
    E.box("Roof", (30, 100, 0.3), loc=(0, 15, 25.5), mat=wall_m)
    # scale cues: mobile scissor lift beside the hero coil, crates, gas cylinders, high-bay lights
    rail_m = E.painted_steel("Rail", (0.62, 0.40, 0.10), rough=0.45, wear=0.3, scale=0.5)
    lx, ly, lz = 5.4, -1.8, 5.2
    E.box("LiftBase", (2.4, 1.1, 0.35), loc=(lx, ly, 0.35), mat=rail_m, bev=0.03)
    for wx in (-0.9, 0.9):
        for wy in (-0.5, 0.5):
            E.cyl(f"Wheel{wx}{wy}", 0.18, 0.12, loc=(lx + wx, ly + wy, 0.18), rot=(math.pi / 2, 0, 0), mat=sb.rubber("Tyre"), verts=24)
    nx = 6
    for k in range(nx):
        z0 = 0.55 + k * (lz - 0.6) / nx; z1 = z0 + (lz - 0.6) / nx
        for wy in (-0.48, 0.48):
            for sgn in (1, -1):
                pts = [(lx - 1.0 * sgn, ly + wy, z0), (lx + 1.0 * sgn, ly + wy, z1)]
                E.sweep(f"Sc{k}{wy}{sgn}", pts, radius=1, profile=[(-0.035, -0.06), (0.035, -0.06), (0.035, 0.06), (-0.035, 0.06)], mat=paint, sub=1, up=(0, 1, 0))
    E.box("LiftDeck", (2.6, 1.2, 0.12), loc=(lx, ly, lz), mat=paint, bev=0.02)
    for hz in (0.55, 1.1):
        for (ax, ay, sx_, sy_) in ((0, -0.6, 2.6, 0.05), (0, 0.6, 2.6, 0.05), (1.3, 0, 0.05, 1.2), (-1.3, 0, 0.05, 1.2)):
            E.box(f"LR{hz}{ax}{ay}", (sx_, sy_, 0.05), loc=(lx + ax, ly + ay, lz + hz), mat=rail_m)
    for (ax, ay) in ((-1.3, -0.6), (1.3, -0.6), (-1.3, 0.6), (1.3, 0.6), (0, -0.6), (0, 0.6)):
        E.cyl(f"LP{ax}{ay}", 0.025, 1.1, loc=(lx + ax, ly + ay, lz + 0.6), mat=rail_m, verts=10)
    crate_m = sb.mat("Ply", (0.42, 0.32, 0.20), rough=0.8)
    for i, (cx, cy, s3) in enumerate(((7.5, 6.0, 1.2), (8.6, 6.4, 0.9), (7.8, 7.4, 1.0), (-9.5, -4, 1.4), (-10.6, -3.2, 1.0))):
        E.box(f"Crate{i}", (s3 * 1.3, s3, s3 * 0.8), loc=(cx, cy, s3 * 0.4), mat=crate_m, bev=0.01)
    lamp_m = sb.emit_mat("HighBay", (1.0, 0.86, 0.7), 25.0)
    for y in [-8 + 8 * i for i in range(7)]:
        for x in (-6.0, 6.0):
            sb.prim("cyl", f"Lamp{x}{y}", loc=(x, y, 23.2), scale=(0.35, 0.35, 0.05), vertices=24, mat=lamp_m)
    E.box("Cart", (1.0, 0.6, 0.9), loc=(1.8, -2.4, 0.45), mat=E.painted_steel("CartP", (0.12, 0.13, 0.15), rough=0.5), bev=0.03)
    E.box("Tray", (0.6, 60, 0.1), loc=(12.5, 15, 8.0), mat=bolt_m)
    # ---- light: warm low-afternoon sun through the clerestory (shafts in faint haze), cool sky fill
    w, sky, bg = sb.world_sky(elev=28, azim=-70, strength=float(os.environ.get("SKY", "0.025")), sun_disc=False)
    sun = sb.sun(28, 0, energy=7.0, color=(1.0, 0.80, 0.58), angle=0.8)
    # sun direction: from -X side (window wall) toward +X, going down
    d = sb.sun_dir(float(os.environ.get("SUNEL", "30")), 270 + float(os.environ.get("SUNAZ", "-30")))
    sun.rotation_mode = 'QUATERNION'; sun.rotation_quaternion = (-d).to_track_quat('-Z', 'Y')
    haze = sb.prim("cube", "Haze", loc=(0, 15, 12.5), scale=(14.2, 45, 12.6))
    haze.visible_shadow = False
    haze.data.materials.append(sb.volume_mat("HazeM", density=float(os.environ.get("HAZE", "0.009")), color=(0.9, 0.88, 0.85), anisotropy=0.6))
    fill = sb.light('AREA', "Bounce", loc=(4, -6, 10), target=(-3, 0, 5), energy=150, color=(0.85, 0.9, 1.0), size=12)
    fill.visible_glossy = False
    work = sb.light('AREA', "WorkLight", loc=(-1.0, -3.0, 3.0), target=(-0.5, 0, 5), energy=150, color=(1.0, 0.8, 0.6), size=1.5)
    E.sphere_probe((0, -3, 5), radius=25)
    E.volume_probe((0, 5, 12), size=(28, 50, 25), res=(10, 16, 8))
    # ---- camera: start inches from the exposed winding pack, pull back + crane up to reveal the whole D
    cam = E.dof_cam("Cam", 28, 2.8)
    cam.data.clip_end = 400
    sc.eevee.bokeh_max_size = 200
    bpy.context.view_layer.update()
    # winding window centre in world space (window 0.20-0.30 of the loop, near side -Y)
    i_w = int(0.25 * len(P))
    wc = V(tuple(P[i_w])) + V((-3.6, -0.40, ZC))
    def pos(t):
        k = sb.smoother(t) * 0.8 + t * 0.2
        p0 = wc + V((0.75, -1.05, -0.55))
        p1 = V((4.6, -20.5, 3.0))
        mid = p0.lerp(p1, 0.5) + V((1.2, 0, -0.8))
        return sb.catmull([p0, mid, p1], k) if False else (p0.lerp(p1, k) + V((1.2, 0, -0.8)) * 4 * k * (1 - k))
    def tgt(t):
        k = sb.smoother(t)
        return wc.lerp(V((0.3, 0, ZC - 1.3)), k)
    def foc(t):
        return (tgt(t) - pos(t)).length
    sb.cam_bake(cam, F0, F1, pos, tgt, focus=foc, fstop=lambda t: sb.lerp(2.8, 5.6, sb.smooth(t)))

# =========================================================================== dispatch
if sid == "s10a":
    build_busbars()
elif sid == "s10b":
    build_cooling()
elif sid == "s10c":
    build_coil()
sb.frames(F0, F1)
sb.render_shot(sid)
