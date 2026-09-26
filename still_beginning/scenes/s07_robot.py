"""s07a (540-574): laser powder-bed fusion insert - the laser traces the component's slice contour (Warren truss
silhouette) in the titanium powder bed: white-amber melt pool, cooling trail, spatter (high-speed-camera slow motion).
s07b (574-630): the robot cell - a white/graphite 6-axis arm with amber joint rings picks a finished component from
the depowdered build plate, transfers it and places it on a fixture; real finger contact, clean release."""
import sys, os, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy
import numpy as np
import sb, dbgcam
from mathutils import Vector as V, Quaternion, Matrix, Euler

sid = (sb.argv() or ["s07a"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))


def build_slm():
    import slm
    F0, F1 = 540, 574
    info = json.load(open(os.path.join(slm.D, "contour.json")))
    bed = slm.chamber()
    bed.data.materials.append(slm.bed_material(fused_ids=(0, 1, 2, 3)))
    rib, outer_len = slm.contour_ribbon(info["loops"])
    holder = []
    rib.data.materials.append(slm.contour_material(holder))
    arc_now = holder[0]
    # the outer loop as a polyline for the laser path
    P = np.array(info["loops"][0])[::3] * 0.001
    P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    SPEED = 0.00062          # m per frame (high-speed-camera slow motion)
    # start on the straightest 24 mm run of the outer contour (a rail edge) so the pool travels cleanly
    T_ = np.diff(P, axis=0); T_ /= np.linalg.norm(T_, axis=1, keepdims=True) + 1e-12
    best = (1e9, 0)
    for i in range(0, len(P) - 2, 4):
        j = int(np.searchsorted(cum, cum[i] + 0.024))
        if j >= len(T_):
            break
        dev = np.max(np.abs(np.arccos(np.clip(T_[i:j] @ T_[i], -1, 1))))
        if dev < best[0]:
            best = (dev, i)
    A0 = cum[best[1]] + 0.002 - SPEED * 0

    def arc_at(f):
        return A0 + SPEED * (f - F0)

    for f in range(F0 - 10, F1 + 11):
        arc_now.default_value = arc_at(f) % total
        arc_now.keyframe_insert("default_value", frame=f)

    def pool(f):
        p, d = slm.point_on_loop(P, cum, total, arc_at(f))
        return V((p[0], p[1], 0.00003)), V((d[0], d[1], 0))

    # melt pool: elongated incandescent droplet + keyhole glow light
    pm = sb.emit_mat("MeltPool", (1.0, 0.90, 0.70), 400.0)
    mp = sb.prim("sphere", "MeltPool", segments=24, ring_count=12, radius=0.00011, mat=pm)
    glowm = sb.emit_mat("PoolHalo", (1.0, 0.45, 0.08), 2.5)
    halo = sb.prim("sphere", "PoolHalo", segments=24, ring_count=12, radius=0.0004, mat=glowm)
    halo.visible_shadow = False
    pl = sb.light('POINT', "PoolLight", loc=(0, 0, 0.0008), energy=0.0025, color=(1.0, 0.62, 0.3), size=0.0004)
    for f in range(F0 - 10, F1 + 11):
        p, d = pool(f)
        ang = math.atan2(d.y, d.x)
        for o, sc_ in ((mp, (2.2, 1.0, 0.6)), (halo, (1.6, 1.0, 0.2))):
            o.location = p
            o.rotation_euler = (0, 0, ang)
            o.scale = sc_
            o.keyframe_insert("location", frame=f); o.keyframe_insert("rotation_euler", frame=f); o.keyframe_insert("scale", frame=f)
        pl.location = p + V((0, 0, 0.0007))
        pl.keyframe_insert("location", frame=f)
        pl.data.energy = 0.0025 * (1 + 0.2 * math.sin(f * 2.3))
        pl.data.keyframe_insert("energy", frame=f)
    # spatter: ballistic incandescent droplets (slow motion); each lives 5-14 frames and cools
    rng = np.random.default_rng(12)
    sparks = []
    for f in range(F0 - 14, F1 + 1):
        for k in range(rng.poisson(5.0)):
            p, d = pool(f)
            back = -d * rng.uniform(0.2, 1.0)
            vdir = V((back.x + rng.normal(0, 0.5), back.y + rng.normal(0, 0.5), rng.uniform(0.6, 1.6))).normalized()
            spd = rng.uniform(0.0012, 0.0032)
            sparks.append((f, p.copy(), vdir * spd, int(rng.integers(5, 15)), rng.uniform(0.6, 1.0)))
    sm_hot = sb.emit_mat("Spark", (1.0, 0.66, 0.3), 120.0)
    base = sb.prim("ico", "SparkSrc", subdivisions=2, radius=0.000028, mat=sm_hot)
    base.hide_render = True
    g = V((0, 0, -0.0000025))
    for i, (fs, p0, v0, life, br) in enumerate(sparks):
        o = base.copy(); sb.link_obj(o); o.hide_render = False
        o.name = "Spark%03d" % i
        for f in range(max(F0 - 3, fs - 1), min(F1 + 3, fs + life + 2)):
            t = f - fs
            if t < 0 or t > life:
                o.scale = (0.001,) * 3
                o.location = p0
            else:
                o.location = p0 + v0 * t + g * t * t * 0.5
                s = br * (1.0 - (t / life) ** 2) + 0.05
                o.scale = (s, s, s)
            o.keyframe_insert("location", frame=f); o.keyframe_insert("scale", frame=f)
    # plume: very faint warm haze just above the pool (volume sphere)
    plume = None and sb.prim("sphere", "Plume", segments=24, ring_count=12, radius=0.004,
                    mat=sb.volume_mat("PlumeM", density=4.0, color=(0.8, 0.8, 0.85), anisotropy=0.4))
    for f in ([] if plume is None else range(F0 - 10, F1 + 11)):
        p, d = pool(f)
        plume.location = p + V((0, 0, 0.0028)) - d * 0.001
        plume.keyframe_insert("location", frame=f)
    # chamber light: cool LED strip above/behind + soft fill; the pool is the warm key
    sb.world_color((0.006, 0.007, 0.009), 1.0)
    sb.light('AREA', "ChamberLED", loc=(0.0, 0.10, 0.22), target=(0, 0.02, 0), energy=0.6, color=(0.75, 0.84, 1.0), size=0.3, size_y=0.04)
    sb.light('AREA', "Window", loc=(-0.10, -0.25, 0.20), target=(0, 0.02, 0), energy=0.12, color=(0.8, 0.86, 1.0), size=0.25)
    # camera: low, across the bed, following the pool with mass (lagged), 100 mm macro
    cam = sb.camera("Cam", loc=(0, -0.06, 0.02), target=(0, 0.028, 0), lens=100, fstop=8, clip=(0.002, 5))

    def lag(f):
        acc = V((0, 0, 0)); wsum = 0
        for k in range(12):
            w = math.exp(-k / 5.0)
            acc += pool(f - k)[0] * w; wsum += w
        return acc / wsum

    mid_f = (F0 + F1) / 2

    def cpos(t):
        f = F0 + t * (F1 - F0)
        c = lag(mid_f).lerp(lag(f), 0.35)
        return c + V((sb.lerp(-0.046, -0.038, t), sb.lerp(-0.062, -0.056, t), sb.lerp(0.024, 0.021, t)))

    def ctgt(t):
        f = F0 + t * (F1 - F0)
        return lag(mid_f).lerp(lag(f), 0.55)

    sb.cam_bake(cam, F0, F1, cpos, ctgt, focus=lambda t: (cpos(t) - pool(F0 + t * (F1 - F0))[0]).length, fstop=22)
    sc.eevee.use_volumetric_shadows = False
    sc.eevee.volumetric_start = 0.002
    sc.eevee.volumetric_end = 0.3
    return F0, F1


if sid == "s07a":
    s0, s1 = build_slm()
else:
    import robotcell
    s0, s1 = robotcell.build(sc)

dbgcam.apply()
sb.frames(s0, s1)
sb.save(sid)
sb.render_shot(sid)
