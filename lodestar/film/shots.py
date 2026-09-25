"""
LODESTAR — the shot list.

Every shot is a function f(t, lt, U) that fills uniforms for time t (global seconds) and
lt (seconds since the shot started). Timings come straight from the score (76 BPM).
"""
import math

import numpy as np

BPM = 76.0
SPB = 60.0 / BPM


def bar(b, beat=0.0):
    return ((b - 1) * 4 + beat) * SPB


END = bar(65) + 9.0 - 0.02   # the soundtrack runs 211.1 s

# ------------------------------------------------------------------------------------------
# music sync data (mirrors lodestar.py)
# ------------------------------------------------------------------------------------------
TICKS = []
for b in range(2, 37):          # bar 1: the watch is wound; it starts ticking at bar 2
    step = 1.0 if b <= 28 else (0.5 if b <= 32 else 0.25)
    x = 0.0
    while x < 4.0 - 1e-9:
        TICKS.append(bar(b, x))
        x += step
for b in (38, 39, 40):
    for x in (0, 2):
        if not (b == 40 and x == 2):
            TICKS.append(bar(b, x))
for b in range(57, 63):
    for x in range(4):
        TICKS.append(bar(b, x))
TICKS.sort()
TICKS_NP = np.array(TICKS)

HITS = []   # (time, strength) — drums that shake the camera
for b in range(17, 25):
    for h in ([0, 2] if b < 21 else [0, 1.5, 2, 3.5]):
        HITS.append((bar(b, h), 0.25 if b < 21 else (0.35 if h in (0, 2) else 0.2)))
for b in range(25, 37):
    for i, p16 in enumerate([0, 3, 6, 8, 11, 14]):
        HITS.append((bar(b, p16 * 0.25), [0.6, 0.35, 0.4, 0.5, 0.35, 0.4][i]))
for b in range(41, 57):
    for h, v in [(0, 0.5), (1.75, 0.25), (2, 0.45), (3.5, 0.2), (3.75, 0.25)]:
        HITS.append((bar(b, h), v))
IMPACTS = [(bar(25), 1.0), (bar(29), 1.0), (bar(33), 1.0), (bar(37), 1.3), (bar(41), 1.2), (bar(49), 1.1), (bar(57), 0.8)]
HITS += IMPACTS
HEARTBEATS = [bar(b, x) for b in (38, 39, 40) for x in (0, 2) if not (b == 40 and x == 2)]


# ------------------------------------------------------------------------------------------
# math helpers
# ------------------------------------------------------------------------------------------
def v3(*a):
    return np.array(a if len(a) == 3 else a[0], dtype=np.float64)


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, x):
    return a + (b - a) * x


def lerpv(a, b, x):
    return np.asarray(a, dtype=np.float64) * (1 - x) + np.asarray(b, dtype=np.float64) * x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def ease_out(x, p=3.0):
    return 1 - (1 - clamp(x)) ** p


def ease_in(x, p=3.0):
    return clamp(x) ** p


def ramp(t, a, b):
    return clamp((t - a) / (b - a)) if b != a else float(t >= a)


def rot_axis(axis, ang):
    axis = norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def basis(forward, up=(0, 1, 0), roll=0.0):
    """matrix with columns (right, up, forward) for a right-handed y-up world."""
    f = norm(forward)
    r = norm(np.cross(f, up))
    u = np.cross(r, f)
    if roll:
        R = rot_axis(f, roll)
        r, u = R @ r, R @ u
    return np.stack([r, u, f], axis=1)


def ship_basis(forward, up=(0, 1, 0), bank=0.0):
    """local->world rotation for the Wren (local: +x right wing... nose +z, up +y)."""
    f = norm(forward)
    r = norm(np.cross(up, f))      # ship's +x
    u = np.cross(f, r)
    if bank:
        R = rot_axis(f, bank)
        r, u = R @ r, R @ u
    return np.stack([r, u, f], axis=1)


def vnoise(t, seed=0.0):
    """smooth 1D value noise in [-1, 1]"""
    i = math.floor(t)
    f = t - i
    def h(n):
        return (math.sin((n + seed * 57.0) * 127.1 + seed * 311.7) * 43758.5453) % 1.0
    a, b = h(i), h(i + 1)
    f = f * f * (3 - 2 * f)
    return (a + (b - a) * f) * 2 - 1


def handheld(t, amt, speed=0.35):
    return (sum(vnoise(t * speed * k, 1 + k) / k for k in (1, 2.3, 5.1)) * amt,
            sum(vnoise(t * speed * k, 7 + k) / k for k in (1, 2.1, 4.7)) * amt)


def shake(t, amt=1.0):
    """camera shake from drum hits: decaying oscillations."""
    yaw = pitch = roll = 0.0
    for th, s in HITS:
        dt = t - th
        if 0.0 <= dt < 0.9:
            e = s * math.exp(-dt / 0.12) * amt
            yaw += e * 0.004 * math.sin(dt * 55 + th * 3.1)
            pitch += e * 0.006 * math.sin(dt * 47 + th * 1.7)
            roll += e * 0.003 * math.sin(dt * 38 + th)
    return yaw, pitch, roll


def recent(events, t, window):
    """time since the most recent event (or None)."""
    best = None
    for e in events:
        if e <= t and t - e < window:
            best = t - e
    return best


def watch_state(t):
    """second-hand step count (with overshoot) and balance-wheel angle."""
    n = int(np.searchsorted(TICKS_NP, t, side='right'))
    if n == 0:
        return 0.0, 0.0, 0.0
    tl = TICKS[n - 1]
    dt = t - tl
    stepped = (n - 1) + (1 - math.exp(-dt / 0.018) * math.cos(2 * math.pi * 14 * dt))
    if n < len(TICKS) and TICKS[n] - tl < 2.2:
        ph = (n - 1) + dt / (TICKS[n] - tl)
        amp = 1.9
    else:
        ph = (n - 1) + dt / 0.8
        amp = 1.9 * math.exp(-dt / 0.35)
    bal = amp * math.sin(math.pi * ph)
    return stepped, bal, float(n)


# ------------------------------------------------------------------------------------------
# uniforms container
# ------------------------------------------------------------------------------------------
POST_DEFAULTS = dict(
    exposure=0.0, bloom=0.22, bloom_threshold=1.2, streak=0.0, streak_tint=(0.45, 0.65, 1.0),
    rays=0.0, light=(0.5, 0.5), ray_decay=0.965, flare=0.0, flare_tint=(1.0, 1.0, 1.0),
    tint=(1.0, 1.0, 1.0), sat=1.0, contrast=1.05, shadow_tint=(1.0, 1.0, 1.0), high_tint=(1.0, 1.0, 1.0),
    vignette=0.55, ca=1.0, grain=0.022, fade=0.0, flash=0.0, flash_col=(1.0, 1.0, 1.0), text=[],
)


class Uniforms:
    def __init__(self):
        self.scene = {'uShipOn': 0.0, 'uShipScale': 1.0, 'uShipPos': (0, 0, 0), 'uShipRot': np.eye(3),
                      'uEngine': 0.0, 'uShipLights': 0.0, 'uHeat': 0.0}
        self.post = dict(POST_DEFAULTS)
        self.post['text'] = []
        self.pos = v3(0, 0, 0)
        self.rot = np.eye(3)
        self.fov = 40.0
        self.focus = 1.0
        self.aperture = 0.0

    def camera(self, pos, target=None, forward=None, up=(0, 1, 0), fov=40.0, roll=0.0, focus=None,
               aperture=0.0, jolt=(0.0, 0.0, 0.0)):
        pos = v3(pos)
        f = norm(v3(target) - pos) if target is not None else norm(forward)
        R = basis(f, up, roll)
        yaw, pitch, rl = jolt
        if yaw or pitch or rl:
            J = rot_axis(R[:, 1], yaw) @ rot_axis(R[:, 0], pitch) @ rot_axis(R[:, 2], rl)
            R = J @ R
        self.pos, self.rot, self.fov = pos, R, fov
        self.focus = focus if focus is not None else (np.linalg.norm(v3(target) - pos) if target is not None else 1.0)
        self.aperture = aperture

    def camera_uniforms(self):
        return {'uCamPos': tuple(self.pos), 'uCamRot': self.rot, 'uFov': math.tan(math.radians(self.fov) / 2),
                'uFocus': self.focus, 'uAperture': self.aperture}

    def project(self, p, aspect=2.39):
        """world point -> (uv, visible)"""
        v = self.rot.T @ (v3(p) - self.pos)
        if v[2] <= 1e-6:
            return (0.5, 0.5), False
        f = math.tan(math.radians(self.fov) / 2)
        x, y = v[0] / v[2] / f, v[1] / v[2] / f
        uv = (0.5 + x / (2 * aspect), 0.5 + y / 2)
        return uv, (-0.2 < uv[0] < 1.2 and -0.2 < uv[1] < 1.2)

    def project_dir(self, d, aspect=2.39):
        return self.project(self.pos + norm(d) * 1e4, aspect)

    def ship(self, pos, rot, scale, engine=1.0, lights=1.0, heat=0.0):
        self.scene.update({'uShipOn': 1.0, 'uShipPos': tuple(v3(pos)), 'uShipRot': rot, 'uShipScale': scale,
                           'uEngine': engine, 'uShipLights': lights, 'uHeat': heat})

    def grade(self, name, **kw):
        g = GRADES[name]
        self.post.update(g)
        self.post.update(kw)


GRADES = {
    'warm': dict(tint=(1.05, 1.0, 0.92), sat=1.05, contrast=1.08, shadow_tint=(0.86, 0.95, 1.08), high_tint=(1.06, 1.0, 0.9)),
    'space': dict(tint=(1.0, 1.0, 1.02), sat=1.05, contrast=1.08, shadow_tint=(0.9, 0.97, 1.1), high_tint=(1.04, 1.0, 0.94)),
    'dawn': dict(tint=(1.04, 1.0, 0.95), sat=1.1, contrast=1.1, shadow_tint=(0.85, 0.95, 1.12), high_tint=(1.08, 1.0, 0.86)),
    'storm': dict(tint=(1.08, 0.96, 0.98), sat=1.15, contrast=1.14, shadow_tint=(0.9, 0.85, 1.1), high_tint=(1.08, 0.98, 0.9)),
    'lost': dict(tint=(0.9, 0.97, 1.08), sat=0.55, contrast=1.1, shadow_tint=(0.85, 0.95, 1.15), high_tint=(1.0, 1.0, 1.0)),
    'lode': dict(tint=(0.95, 1.0, 1.08), sat=1.05, contrast=1.1, shadow_tint=(0.85, 0.95, 1.15), high_tint=(1.0, 1.02, 1.08)),
}


# ------------------------------------------------------------------------------------------
# shots
# ------------------------------------------------------------------------------------------
class Shot:
    def __init__(self, name, start, end, scene, fn, samples=8):
        self.name, self.start, self.end, self.scene, self.fn, self.samples = name, start, end, scene, fn, samples

    def frame_range(self, fps=24):
        return int(math.ceil(self.start * fps - 1e-9)), int(math.ceil(self.end * fps - 1e-9))


HOME_ROT = rot_axis((0, 1, 0), 2.2) @ rot_axis((1, 0, 0), 0.35)
MILKY = tuple(norm((0.3, 0.8, -0.5)))


def planet_common(U, t, sun_dir, sun_col=(1.0, 0.95, 0.88), intensity=22.0, rot_speed=0.004, city=1.0):
    U.scene.update({'uSunDir': tuple(norm(sun_dir)), 'uSunCol': tuple(np.array(sun_col) * intensity),
                    'uSunSize': 0.0075, 'uPlanetPos': (0.0, 0.0, 0.0),
                    'uPlanetRot': rot_axis((0, 1, 0), t * rot_speed) @ HOME_ROT,
                    'uCloudTime': t, 'uCity': city, 'uTrail': 0.0, 'uTrailA': (0, 0, 0), 'uTrailB': (0, 0, 1),
                    'uStarBright': 0.6, 'uMilkyAxis': MILKY})


def sun_flare(U, sun_dir, strength=1.0):
    uv, vis = U.project_dir(sun_dir)
    U.post['light'] = uv
    out = max(0.0, -uv[0], uv[0] - 1.0, -uv[1], uv[1] - 1.0)
    strength *= clamp(1.0 - out * 6.0)
    if vis and strength > 0:
        U.post['flare'] = strength
        U.post['rays'] = 0.18 * strength
        U.post['streak'] = 0.12 * strength


# ---- 2. HOME: sunrise over the planet
def s_home(t, lt, U):
    d = 1.2
    cam = v3(0, 0.0, d)
    limb = math.degrees(math.asin(1.0 / d))
    el = math.radians(limb + 7.0 - 1.5 * lt / 12.6)
    yaw = math.radians(-6 + 3 * lt / 12.6)
    fwd = rot_axis((0, 1, 0), yaw) @ v3(0, math.sin(el), -math.cos(el))
    U.camera(cam, forward=fwd, fov=38)
    a = math.radians(limb - 1.3 + 2.4 * smooth(lt / 12.6))
    sun = rot_axis((0, 1, 0), math.radians(-2)) @ v3(0, math.sin(a), -math.cos(a))
    planet_common(U, t, sun)
    U.grade('dawn', exposure=-0.4 + 0.2 * smooth(lt / 12.6), fade=1 - ramp(lt, 0, 1.2))
    sun_flare(U, sun, smooth(ramp(lt, 7.5, 11.0)))


def relative(U, keys=('uPlanetPos', 'uShipPos', 'uTrailA', 'uTrailB')):
    """camera-relative rendering: move the world so the camera sits at the origin (float precision)."""
    c = U.pos.copy()
    for k in keys:
        if k in U.scene:
            U.scene[k] = tuple(v3(U.scene[k]) - c)
    U.pos = v3(0, 0, 0)


SHIP_S = 0.0004
HOME_CAM = v3(0, 0, 1.2)


def home_sun(extra_deg=1.1, yaw_deg=-2.0):
    limb = math.degrees(math.asin(1.0 / 1.2))
    a = math.radians(limb + extra_deg)
    return rot_axis((0, 1, 0), math.radians(yaw_deg)) @ v3(0, math.sin(a), -math.cos(a))


# ---- 3. THE WREN: glides through frame, backlit by the sunrise
def s_wren(t, lt, U):
    x = lt / 12.63
    sun = home_sun(1.1 + 0.8 * x)
    limb = math.degrees(math.asin(1.0 / 1.2))
    el = math.radians(limb + 4.0)
    base_f = v3(0, math.sin(el), -math.cos(el))
    cam = HOME_CAM + v3(0.0006 * x, 0.0, 0.0)
    right = norm(np.cross(base_f, (0, 1, 0)))
    ship_pos = HOME_CAM + base_f * 0.0062 + right * lerp(0.0042, -0.0012, smooth(x) * 0.5 + x * 0.5) \
        + v3(0, 1, 0) * 0.0002 * math.sin(lt * 0.4)
    cam_up = np.cross(right, base_f)
    heading = norm(-right + base_f * 0.08 + cam_up * 0.03)
    rot = ship_basis(heading, up=cam_up, bank=math.radians(-10 + 3 * math.sin(lt * 0.3)))
    hh = handheld(t, 0.0015)
    look = lerpv(HOME_CAM + base_f * 0.0062 + right * 0.0012, ship_pos, 0.35)
    U.camera(cam, target=look, fov=34, jolt=(hh[0], hh[1], 0.0))
    planet_common(U, t, sun)
    U.ship(ship_pos, rot, SHIP_S, engine=0.12, lights=1.0)
    U.grade('dawn', exposure=-0.2)
    sun_flare(U, sun, 0.8)
    relative(U)


# ---- 4. IGNITION: engines light on the timpani, the Wren leaves
def s_ignite(t, lt, U):
    ti = 0.0                      # the shot starts on the first timpani hit
    sun = home_sun(2.2 + 0.5 * lt / 12.6, -1.0)
    heading = norm(sun + v3(0.08, -0.02, 0.0))
    right = norm(np.cross(heading, (0, 1, 0)))
    up = np.cross(right, heading)
    start = HOME_CAM + v3(0.0, 0.0012, -0.0015)
    acc = 0.0011
    dist = 0.0003 * lt + 0.5 * acc * max(lt - 0.3, 0.0) ** 2
    ship_pos = start + heading * dist
    cam = start - heading * 0.0052 - up * 0.0011 + right * 0.0014 - heading * 0.00015 * lt
    hh = handheld(t, 0.002)
    kick = shake(t, 1.0)
    target = lerpv(ship_pos + up * 0.0003, start + heading * 0.03, smooth(ramp(lt, 2.0, 10.0)))
    U.camera(cam, target=target, up=up, fov=lerp(38, 30, smooth(ramp(lt, 0, 10))), jolt=(hh[0] + kick[0], hh[1] + kick[1], kick[2]))
    planet_common(U, t, sun)
    burn = ramp(lt, 0.0, 0.18)
    eng = lerp(0.12, 1.0, burn) + 2.2 * math.exp(-max(lt - 0.1, 0.0) / 0.8) * burn
    U.ship(ship_pos, ship_basis(heading, up, math.radians(-8)), SHIP_S, engine=eng, lights=1.0)
    U.grade('dawn', exposure=-0.25)
    U.post['bloom'] = 0.25 + 0.15 * math.exp(-lt / 0.8)
    sun_flare(U, sun, 0.7)
    relative(U)


def watch_common(U, t, lamp=(1.0, 0.72, 0.42), lamp_i=7.0, env=(1.0, 0.84, 0.62), env_i=0.8,
                 rim=(0.45, 0.65, 1.0), rim_i=0.4, crack=0.0, dust=1.0, spin=0.0):
    sec, bal, _ = watch_state(t)
    U.scene.update({'uSec': sec, 'uBal': bal, 'uCrack': crack, 'uLampPos': (-24.0, 34.0, -8.0),
                    'uLampCol': tuple(np.array(lamp) * lamp_i), 'uEnvCol': tuple(np.array(env) * env_i),
                    'uRimCol': tuple(np.array(rim) * rim_i), 'uDust': dust, 'uSpin': spin})


# ---- 1. THE WATCH: fade up on the movement, the second hand ticking
def s_watch_open(t, lt, U):
    x = lt / bar(5)
    cam = lerpv((-11.5, 11.5, -17.0), (-8.5, 8.6, -14.5), smoother(x))
    target = lerpv((-6.2, 1.6, -9.6), (-4.6, 1.6, -9.0), smoother(x))
    hh = handheld(t, 0.002, 0.2)
    U.camera(cam, target=target, fov=30, aperture=0.3, jolt=(hh[0], hh[1], 0.0))
    watch_common(U, t)
    U.grade('warm', exposure=-0.3, fade=1.0 - smooth(ramp(lt, 0.2, 3.2)))
    U.post['bloom'] = 0.3


# ---- 8. TIME SLIPS: the escapement racing, storm light flickering
def s_watch_slip(t, lt, U):
    x = lt / (bar(30) - bar(29))
    cam = lerpv((17.5, 6.0, -10.0), (16.0, 5.0, -8.5), smooth(x))
    target = v3(9.0, 0.9, -2.6)
    hh = handheld(t, 0.006, 0.9)
    k = shake(t, 0.8)
    U.camera(cam, target=target, fov=28, roll=math.radians(-12), aperture=0.25,
             jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    flick = 0.55 + 0.45 * math.sin(t * 23.0) * math.sin(t * 7.3)
    watch_common(U, t, lamp=(1.0, 0.28, 0.12), lamp_i=6.0 * flick, env=(0.8, 0.35, 0.3), env_i=0.5,
                 rim=(0.4, 0.5, 1.0), rim_i=0.6, dust=0.6, spin=lt * 5.0)
    U.grade('storm', exposure=-0.2)


# ---- 10. TIME BREAKS: the dial from above, hands spinning, the crystal cracks
def s_watch_break(t, lt, U):
    x = lt / (bar(34) - bar(33))
    cam = v3(0.0, 34.0 - 6.0 * x, -6.0)
    ang = math.radians(8 + 25 * x)
    up = v3(math.sin(ang), 0.0, math.cos(ang))
    k = shake(t, 1.2)
    hh = handheld(t, 0.006, 0.8)
    U.camera(cam, target=(0.8, 3.0, -1.0), up=up, fov=42, aperture=0.15, jolt=(k[0] + hh[0], k[1] + hh[1], k[2]))
    crack = ease_out(ramp(lt, 0.0, 0.35), 2) * 0.55 + 0.45 * smooth(ramp(lt, 0.8, 3.0))
    flash = math.exp(-lt / 0.25)
    watch_common(U, t, lamp=(1.0, 0.35, 0.2), lamp_i=6.0 + 10 * flash, env=(0.75, 0.4, 0.4), env_i=0.5,
                 rim=(0.5, 0.6, 1.0), rim_i=0.8, crack=crack, dust=0.4, spin=lt * 14.0)
    U.scene['uSec'] = U.scene['uSec'] + lt * 22.0
    U.grade('storm', exposure=-0.1, flash=0.6 * math.exp(-lt / 0.12))


# ---- 13. HEARTBEAT: the cracked watch in the dark
def s_watch_heart(t, lt, U):
    x = lt / (bar(40) - bar(39))
    cam = lerpv((-3.0, 16.0, -17.0), (-2.4, 14.0, -15.0), x)
    U.camera(cam, target=(0.5, 3.4, -2.0), fov=34, aperture=0.2)
    hb = recent(HEARTBEATS, t, 1.0)
    pulse = 0.0 if hb is None else math.exp(-hb / 0.12) + 0.6 * math.exp(-max(hb - 0.3, 0.0) / 0.12) * (hb > 0.3)
    watch_common(U, t, lamp=(0.55, 0.7, 1.0), lamp_i=1.6 + 2.5 * pulse, env=(0.4, 0.55, 0.9), env_i=0.22,
                 rim=(0.6, 0.75, 1.0), rim_i=0.25, crack=1.0, dust=0.3)
    U.grade('lost', exposure=-0.3)


# ---- 20. STILL: warm light, the last ticks, then the watch stops
def s_watch_still(t, lt, U):
    x = lt / (bar(64) - bar(61))
    cam = lerpv((9.0, 22.0, -24.0), (6.0, 17.0, -18.0), smoother(x))
    U.camera(cam, target=(0.5, 3.0, -1.5), fov=36, aperture=0.25)
    watch_common(U, t, lamp=(1.0, 0.78, 0.5), lamp_i=8.0, env=(1.0, 0.86, 0.66), env_i=0.9,
                 rim=(0.7, 0.8, 1.0), rim_i=0.3, crack=1.0, dust=1.0)
    U.grade('warm', exposure=-0.15, fade=smooth(ramp(t, bar(63, 3.0), bar(64) - 0.05)))


# ---- 5. THE RINGS: skimming the ring plane of a gas giant
GIANT_SUN = norm((0.75, 0.2, -0.62))


def rings_common(U, t, cam_local, rot=np.eye(3), rocks=True, sun=GIANT_SUN, sun_i=6.0):
    """cam_local: camera position in planet-local space (planet at origin, rings in XZ)."""
    cell = 0.0006
    base = np.floor(np.array([cam_local[0], cam_local[2]]) / cell)
    frac = np.array([cam_local[0], cam_local[2]]) - base * cell
    U.scene.update({'uSunDir': tuple(norm(sun)), 'uSunCol': tuple(np.array((1.0, 0.95, 0.88)) * sun_i),
                    'uGiantPos': tuple(-rot.T @ v3(cam_local)), 'uGiantRot': rot, 'uRockOn': 1.0 if rocks else 0.0,
                    'uRockBase': tuple(base), 'uRockFrac': tuple(frac), 'uCamLocalY': float(cam_local[1]),
                    'uStarBright': 0.5})


def s_rings(t, lt, U):
    x = lt / (bar(21) - bar(17))
    R = 2.08
    v = 0.0045
    phi = 0.25 + v * lt / R
    h = lerp(0.010, 0.014, smooth(x))
    cam_l = v3(R * math.cos(phi), h, R * math.sin(phi))
    T = v3(-math.sin(phi), 0.0, math.cos(phi))
    inward = v3(-math.cos(phi), 0.0, -math.sin(phi))
    look = norm(inward * 0.85 + T * lerp(0.62, 0.45, smooth(x)) + v3(0, -0.2, 0))
    k = shake(t, 0.35)
    hh = handheld(t, 0.0015, 0.5)
    U.camera((0, 0, 0), forward=look, fov=44, jolt=(hh[0] + k[0], hh[1] + k[1], k[2] + math.radians(-6)))
    rings_common(U, t, cam_l, rocks=False, sun_i=4.0)
    right = norm(np.cross(look, (0, 1, 0)))
    ship_rel = look * 0.011 + v3(0, -0.0016, 0) + right * lerp(-0.0022, 0.0008, smooth(x)) \
        + v3(0, 0.0002 * math.sin(lt * 0.6), 0)
    U.ship(ship_rel, ship_basis(norm(T + inward * 0.12), (0, 1, 0), math.radians(-18 + 6 * math.sin(lt * 0.4))),
           0.0004, engine=1.3, lights=1.0)
    U.grade('space', exposure=-0.35)
    uv, vis = U.project_dir(GIANT_SUN)
    U.post['light'] = uv


# ---- storm helpers
STORM_HOT = dict(uColA=(1.0, 0.36, 0.2), uColB=(0.5, 0.06, 0.32), uDustCol=(0.7, 0.5, 0.62))
FLASH_EVENTS = []
for _b in range(21, 37):
    for _x in (0, 2):
        FLASH_EVENTS.append((bar(_b, _x), 0.35 if _b < 25 else 0.6))
for _te, _s in IMPACTS[:3]:
    FLASH_EVENTS.append((_te, 2.5))
FLASH_EVENTS.append((bar(36, 2), 1.2))
FLASH_EVENTS.append((bar(36, 3), 1.5))


def flash_env(dt):
    if dt < 0:
        return 0.0
    e = math.exp(-dt / 0.13)
    e += 0.6 * math.exp(-max(dt - 0.09, 0) / 0.05) * (dt > 0.09)
    e += 0.4 * math.exp(-max(dt - 0.22, 0) / 0.08) * (dt > 0.22)
    return e


def flashes(U, t, cam, fwd, t_from=0.0, gain=1.0):
    active = []
    for te, s in FLASH_EVENTS:
        if te < t_from - 0.01:
            continue
        dt = t - te
        if 0 <= dt < 1.2:
            active.append((s * flash_env(dt) * gain, te, s))
    active.sort(reverse=True)
    right = norm(np.cross(fwd, (0, 1, 0)))
    up = np.cross(right, fwd)
    pos, inten = [], []
    for k in range(3):
        if k < len(active):
            I, te, s = active[k]
            h = [((math.sin(te * (12.9898 + 7 * j)) * 43758.5453) % 1.0) for j in range(3)]
            p = cam + fwd * (4 + 9 * h[0]) + right * (h[1] - 0.5) * 12 + up * (h[2] - 0.5) * 6
            pos.append(tuple(p))
            inten.append(I * 22.0)
        else:
            pos.append((0.0, 0.0, 0.0))
            inten.append(0.0)
    U.scene['uFlashPos'] = pos
    U.scene['uFlashI'] = inten
    U.scene['uFlashCol'] = (0.85, 0.8, 1.0)
    # a visible bolt for the strongest recent strike
    bolt = [(0.0, 0.0, 0.0)] * 8
    bi = 0.0
    if active and active[0][2] >= 0.6:
        I, te, s = active[0]
        rs = np.random.default_rng(int(te * 1000))
        a = v3(pos[0]) + up * 3.5 + right * rs.uniform(-1, 1)
        pts = [a]
        for i in range(7):
            step = -up * rs.uniform(0.7, 1.2) + right * rs.normal(0, 0.45) + fwd * rs.normal(0, 0.3)
            pts.append(pts[-1] + step)
        bolt = [tuple(p) for p in pts]
        bi = min(I, 3.0) * (1.0 if active[0][2] > 1.0 else 0.5)
    U.scene['uBolt'] = bolt
    U.scene['uBoltI'] = bi
    return inten[0] / 22.0 if inten else 0.0


def storm_common(U, t, look=None, **kw):
    d = dict(STORM_HOT)
    d.update({'uDensity': 1.25, 'uEmission': 0.55, 'uWallZ': -1e4, 'uClearZ': 1e4, 'uVortex': 0.0,
              'uVortexPos': (0.0, 0.0, 0.0), 'uVortexAxis': (0.0, 0.0, -1.0), 'uSpin': 0.0,
              'uStarDir': (0.0, 0.0, 1.0), 'uStarCol': (0.8, 0.9, 1.0), 'uStarI': 0.0, 'uStarBright': 0.6,
              'uSteps': 150.0, 'uBlob': (0.0, 0.0, 0.0, 0.0), 'uEyeCol': (1.0, 0.75, 0.85), 'uNoiseScale': 1.0})
    d.update(kw)
    U.scene.update(d)


def buffet(t, amt):
    return v3(vnoise(t * 1.3, 11) * amt, vnoise(t * 1.1, 12) * amt * 0.7, 0.0), vnoise(t * 0.9, 13) * 0.35 * amt / 0.1


# ---- 6. THE WALL: chase cam toward the storm
def s_wall(t, lt, U):
    z = -34.0 + 1.6 * lt + 0.1 * lt * lt
    off, bank = buffet(t, 0.03 + 0.04 * ramp(lt, 6, 12.6))
    ship = v3(0.0, 0.0, z + 1.7) + off
    k = shake(t, 0.9)
    hh = handheld(t, 0.003, 0.6)
    cam = v3(0.35, 0.3, z)
    U.camera(cam, target=ship + v3(0.25, 0.05, 2.5), fov=48, jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    storm_common(U, t, uBlob=(26.0, -14.0, 62.0, 62.0), uDensity=1.3, uEmission=0.7, uStarBright=0.45,
                 uNoiseScale=lerp(0.3, 0.8, smooth(ramp(lt, 0, 12.6))))
    flashes(U, t, cam, v3(0, 0, 1), gain=0.8)
    U.ship(ship, ship_basis((0, 0, 1), (0, 1, 0), math.radians(bank * 25)), 0.06, engine=1.4, lights=1.0)
    U.grade('storm', exposure=-0.2, flash=0.25 * smooth(ramp(lt, 12.2, 12.63)))


# ---- 7. THE STORM: inside, buffeted, lightning on every brass hit
def s_storm(t, lt, U):
    z = 3.0 + 3.2 * lt
    off, bank = buffet(t, 0.08)
    ship = v3(0.0, 0.0, z + 1.5) + off
    k = shake(t, 1.4)
    hh = handheld(t, 0.006, 0.9)
    cam = v3(-0.3, 0.22, z) + off * 0.4
    U.camera(cam, target=ship + v3(0, 0.1, 3.0), fov=50, roll=math.radians(bank * 8), jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    storm_common(U, t)
    fl = flashes(U, t, cam, v3(0, 0, 1))
    U.ship(ship, ship_basis(norm((off[0] * 0.3, off[1] * 0.3, 1)), (0, 1, 0), math.radians(bank * 40)), 0.06, engine=1.6, lights=1.0)
    U.grade('storm', exposure=-0.25, flash=0.3 * math.exp(-max(lt, 0) / 0.08))


# ---- 9. THE EYE: a vortex opens ahead
def s_eye(t, lt, U):
    z = 60.0 + 1.6 * lt
    vpos = v3(9.0, 4.0, 60.0 + 34.0)
    off, bank = buffet(t, 0.07)
    ship = v3(0.2, 0.0, z + 1.6) + off
    k = shake(t, 1.2)
    hh = handheld(t, 0.005, 0.8)
    cam = v3(-0.45, 0.35, z) + off * 0.3
    U.camera(cam, target=lerpv(ship + v3(0, 0.1, 3.0), vpos, 0.25), fov=52, jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    storm_common(U, t, uVortex=1.0, uVortexPos=tuple(vpos), uVortexAxis=tuple(norm((-0.45, -0.25, -1.0))), uSpin=t * 0.35,
                 uDensity=1.1, uEmission=0.45)
    flashes(U, t, cam, v3(0, 0, 1))
    U.ship(ship, ship_basis(norm(vpos - ship), (0, 1, 0), math.radians(bank * 40)), 0.06, engine=1.6, lights=1.0)
    U.grade('storm', exposure=-0.2)


# ---- 11. THE MAELSTROM: dragged into the spinning eye, into white
def s_maelstrom(t, lt, U):
    x = lt / (bar(37) - bar(34))
    vpos = v3(3.0, 1.5, 140.0)
    z = 140.0 - 26.0 + 3.0 * lt + 0.45 * lt * lt
    spin_cam = 0.25 * lt * lt
    off, bank = buffet(t, 0.09)
    ship = v3(0.0, 0.0, z + 1.5) + off
    k = shake(t, 1.6)
    hh = handheld(t, 0.008, 1.1)
    cam = v3(0.0, 0.3, z) + off * 0.3
    U.camera(cam, target=ship + v3(0, 0.05, 4.0), fov=lerp(52, 64, x), roll=spin_cam, jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    storm_common(U, t, uVortex=1.0, uVortexPos=tuple(vpos), uVortexAxis=tuple(norm((-0.2, -0.1, -1.0))), uSpin=t * 0.35 + lt * lt * 0.05,
                 uDensity=0.9, uEmission=0.35 + 0.3 * x)
    flashes(U, t, cam, v3(0, 0, 1))
    roll = spin_cam * 1.3 + bank
    U.ship(ship, ship_basis((0, 0, 1), (math.sin(roll), math.cos(roll), 0), 0.0), 0.06, engine=2.0, lights=1.0)
    U.grade('storm', exposure=-0.2 + 1.2 * ease_in(ramp(lt, 8.4, 9.47), 2),
            flash=ease_in(ramp(lt, 8.9, 9.47), 2.5), flash_col=(1.0, 0.95, 0.92))


# ---- 15. LODESTAR: the engines relight, the Wren turns toward the light
LODE_DIR = norm((0.35, 0.12, 1.0))


def s_lode(t, lt, U):
    z = 300.0 + 0.6 * lt + 0.06 * lt * lt
    turn = smoother(ramp(lt, 0.3, 4.5))
    heading = norm(lerpv(norm((-0.6, -0.2, 0.75)), LODE_DIR, turn))
    ship = v3(0.0, 0.0, z + 1.4)
    cam = v3(-0.55, 0.28, z) - heading * 0.1
    k = shake(t, 1.0)
    hh = handheld(t, 0.003, 0.5)
    U.camera(cam, target=lerpv(ship, ship + LODE_DIR * 6.0, smooth(ramp(lt, 0.5, 6))), fov=50, jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    star_i = lerp(2.5, 3.2, smooth(ramp(lt, 0, 12)))
    storm_common(U, t, uColA=(0.5, 0.45, 0.8), uColB=(0.18, 0.1, 0.3), uDustCol=(0.55, 0.62, 0.75), uDensity=0.45,
                 uEmission=0.08, uStarDir=tuple(LODE_DIR), uStarCol=(0.75, 0.88, 1.0), uStarI=star_i)
    flashes(U, t, cam, v3(0, 0, 1), t_from=1e9)
    burn = math.exp(-lt / 0.9) * 2.5
    U.ship(ship, ship_basis(heading, (0, 1, 0), math.radians(-20 * (1 - turn) + 5)), 0.06, engine=ramp(lt, 0, 0.1) * (1.2 + burn), lights=1.0)
    U.grade('lode', exposure=-0.9)
    uv, vis = U.project_dir(LODE_DIR)
    U.post.update(light=uv, rays=0.55 if vis else 0.0, ray_decay=0.975, flare=0.8 if vis else 0.0, streak=0.35,
                  flare_tint=(0.7, 0.85, 1.0), bloom=0.3)


# ---- 16. BREAKING THROUGH: out of the last wall into open space
def s_breakthrough(t, lt, U):
    x = lt / (bar(49) - bar(45))
    z = 400.0 + 3.2 * lt + 0.12 * lt * lt
    off, bank = buffet(t, 0.03)
    ship = v3(0.0, 0.0, z + 1.6) + off
    k = shake(t, 0.8)
    hh = handheld(t, 0.003, 0.5)
    cam = v3(0.5, -0.25, z) + off * 0.3
    U.camera(cam, target=lerpv(ship + v3(0, 0, 3), ship + LODE_DIR * 10.0, 0.5), fov=lerp(50, 40, x), jolt=(hh[0] + k[0], hh[1] + k[1], k[2]))
    storm_common(U, t, uColA=(0.55, 0.5, 0.85), uColB=(0.2, 0.12, 0.32), uDustCol=(0.92, 0.96, 1.0), uDensity=0.8,
                 uEmission=0.1, uClearZ=400.0 + 3.2 * 6.5 + 0.12 * 6.5 * 6.5 + 6.0, uStarDir=tuple(LODE_DIR),
                 uStarCol=(0.75, 0.88, 1.0), uStarI=3.2, uStarBright=0.8)
    flashes(U, t, cam, v3(0, 0, 1), t_from=1e9)
    U.ship(ship, ship_basis(norm(LODE_DIR * 0.4 + v3(0, 0, 0.6)), (0, 1, 0), math.radians(bank * 30 - 8)), 0.06, engine=1.8, lights=1.0)
    U.grade('lode', exposure=-0.6)
    uv, vis = U.project_dir(LODE_DIR)
    U.post.update(light=uv, rays=0.6 if vis else 0.0, ray_decay=0.975, flare=0.8 if vis else 0.0, streak=0.4,
                  flare_tint=(0.7, 0.85, 1.0), bloom=0.3)


# ---- void helpers
def void_common(U, star_dir=(0, 0, 1), star_i=0.0, star_bright=0.5, rim=(0.3, 0.4, 0.6), rim_i=0.08, debris=0.0):
    U.scene.update({'uStarDir': tuple(norm(star_dir)), 'uStarCol': (0.78, 0.9, 1.0), 'uStarI': star_i,
                    'uStarBright': star_bright, 'uRimDir': tuple(norm((-0.4, 0.6, -0.7))),
                    'uRimCol': tuple(np.array(rim) * rim_i), 'uDebris': debris})


# ---- 12. LOST: out of the white, a dead ship tumbling in the dark
def s_lost(t, lt, U):
    x = lt / (bar(39) - bar(37))
    cam = v3(0.0, 0.0, 0.0) + v3(0.02, 0.01, 0.0) * lt
    ship = v3(0.1, -0.05, 2.2)
    rot = rot_axis((0.3, 1.0, 0.4), 0.9 + lt * 0.16) @ rot_axis((1, 0, 0), 0.5 + lt * 0.07)
    U.camera(cam, target=ship + v3(0.0, 0.02, 0.0), fov=lerp(34, 30, x), aperture=0.0)
    void_common(U, star_bright=0.35, rim_i=1.4, debris=1.0)
    U.ship(ship, rot, 0.25, engine=0.0, lights=0.0)
    U.grade('lost', exposure=0.1, flash=math.exp(-lt / 0.45), flash_col=(1.0, 0.96, 0.94))
    U.post['grain'] = 0.03


# ---- 14. A LIGHT: one star appears and grows until it fills everything
LIGHT_DIR = norm((0.08, 0.06, 1.0))


def s_light(t, lt, U):
    cut = bar(40, 3.5) - bar(40)
    x = ramp(lt, 0.0, cut)
    cam = v3(0.0, 0.0, 0.0)
    ship = v3(-0.25, -0.18, 1.6)
    rot = rot_axis((0.3, 1.0, 0.4), 0.9 + (lt + 6.3) * 0.16) @ rot_axis((1, 0, 0), 0.5 + (lt + 6.3) * 0.07)
    U.camera(cam, target=LIGHT_DIR * 10.0, fov=lerp(32, 28, x))
    si = 0.08 + 5.0 * ease_in(x, 3.2)
    void_common(U, star_dir=LIGHT_DIR, star_i=si, star_bright=0.35, rim_i=1.0, debris=0.6)
    U.ship(ship, rot, 0.2, engine=0.0, lights=0.0)
    uv, vis = U.project_dir(LIGHT_DIR)
    U.grade('lode', exposure=-0.3)
    U.post.update(light=uv, rays=0.8 * ease_in(x, 2), flare=1.5 * ease_in(x, 2), streak=0.6 * ease_in(x, 2),
                  flare_tint=(0.7, 0.85, 1.0), bloom=0.25 + 0.4 * x, ray_decay=0.98)
    if lt >= cut:
        U.post['fade'] = 1.0


# ---- 21. TITLE
def s_title(t, lt, U):
    U.camera(v3(0, 0, 0), forward=rot_axis((0, 1, 0), 0.004 * lt) @ norm((0.3, 0.1, 1.0)), fov=40)
    void_common(U, star_bright=0.3 * smooth(ramp(lt, 0.5, 4.0)))
    a1 = smooth(ramp(lt, 0.6, 3.2)) * (1 - smooth(ramp(lt, 9.3, 11.6)))
    a2 = smooth(ramp(lt, 2.4, 4.6)) * (1 - smooth(ramp(lt, 9.0, 11.2)))
    a3 = smooth(ramp(lt, 4.4, 6.4)) * (1 - smooth(ramp(lt, 8.8, 11.0)))
    track = 0.55 + 0.1 * smooth(ramp(lt, 0.0, 12.0))
    U.post['text'] = [('LODESTAR', 'Cinzel-Regular.ttf', 118, 0.45, track, round(a1, 3)),
                      ('a voyage in seven movements', 'CormorantGaramond-LightItalic.ttf', 40, 0.555, 0.06, round(a2, 3)),
                      ('composed, synthesized & rendered entirely from code', 'CormorantGaramond-Regular.ttf', 24, 0.8, 0.18, round(a3 * 0.7, 3))]
    U.grade('space', exposure=0.0)
    U.post['fade'] = smooth(ramp(lt, 10.8, 12.1))


# ---- 17. HOME, AGAIN: the lodestar rises over home; the Wren streaks toward it
LODE_SUN_COL = (0.9, 0.95, 1.0)


def s_home_again(t, lt, U):
    x = lt / (bar(53) - bar(49))
    d = 1.9 - 0.25 * smooth(x)
    cam = v3(0.0, 0.0, d)
    limb = math.degrees(math.asin(1.0 / d))
    el = math.radians(limb + 5.0)
    fwd = rot_axis((0, 1, 0), math.radians(8 - 6 * x)) @ v3(0, math.sin(el), -math.cos(el))
    k = shake(t, 0.5)
    U.camera(cam, forward=fwd, fov=40, jolt=k)
    a = math.radians(limb - 0.6 + 3.2 * smooth(ramp(lt, 0.0, 9.0)))
    sun = rot_axis((0, 1, 0), math.radians(3)) @ v3(0, math.sin(a), -math.cos(a))
    planet_common(U, t, sun, sun_col=LODE_SUN_COL, intensity=24.0)
    right = norm(np.cross(fwd, (0, 1, 0)))
    up = np.cross(right, fwd)
    start = cam + fwd * 0.004 - right * 0.004 - up * 0.0012
    goal = cam + fwd * 0.6 + up * 0.02
    s = ease_in(ramp(lt, 0.3, 12.6), 1.7)
    ship_pos = lerpv(start, goal, s * 0.12)
    heading = norm(goal - start)
    U.ship(ship_pos, ship_basis(heading, up, math.radians(-25 + 15 * x)), SHIP_S, engine=2.2, lights=1.0)
    U.grade('dawn', exposure=-0.3)
    sun_flare(U, sun, 1.0)
    relative(U)


# ---- 18. DESCENT: atmospheric entry — the Wren becomes a comet over the daylit world
DESC_SUN = norm((0.55, 0.42, -0.72))


def s_descent(t, lt, U):
    x = lt / (bar(57) - bar(53))
    ang = 1.25 + 0.012 * lt
    frame = rot_axis((1, 0, 0), -ang)
    alt_ship = lerp(0.06, 0.022, smooth(x))
    ship_p = frame @ v3(0, 1.0 + alt_ship, 0)
    upv = norm(ship_p)
    tang = norm(frame @ v3(0, -0.28 - 0.1 * x, -1.0))
    right = norm(np.cross(tang, upv))
    cam = ship_p - tang * lerp(0.014, 0.02, x) + upv * 0.004 + right * 0.008
    k = shake(t, 0.9)
    hh = handheld(t, 0.002, 0.8)
    U.camera(cam, target=ship_p + tang * 0.012 - upv * 0.003, up=upv, fov=lerp(42, 50, x),
             jolt=(k[0] + hh[0], k[1] + hh[1], k[2]))
    planet_common(U, t, DESC_SUN, sun_col=(1.0, 0.95, 0.88), intensity=22.0, city=0.0, rot_speed=0.0)
    heat = smooth(ramp(alt_ship, 0.045, 0.03)) * (0.85 + 0.15 * math.sin(lt * 11.0))
    U.ship(ship_p, ship_basis(tang, upv, math.radians(6 * math.sin(lt * 0.5))), SHIP_S * 4.0, engine=0.5, lights=1.0,
           heat=heat)
    U.scene.update({'uTrail': 9.0 * heat + 0.3, 'uTrailA': tuple(ship_p + tang * 0.0015),
                    'uTrailB': tuple(ship_p - tang * lerp(0.03, 0.12, smooth(x)))})
    U.grade('dawn', exposure=-1.9, flash=0.7 * ease_in(ramp(lt, 11.9, 12.63), 3), flash_col=(1.0, 0.92, 0.8))
    relative(U)


# ---- 19. HOMECOMING: dawn over the world, the trail fading into the clouds
def s_homecoming(t, lt, U):
    x = lt / (bar(61) - bar(57))
    ang = 1.02
    frame = rot_axis((1, 0, 0), -ang)
    base = frame @ v3(0, 1.0, 0)
    tang = norm(frame @ v3(0, 0, -1.0))
    upv = norm(base)
    side = norm(np.cross(tang, upv))
    cam = base * (1.0 + lerp(0.13, 0.115, smooth(x))) + tang * 0.002 * lt
    look = norm(tang + upv * lerp(-0.5, -0.42, smooth(x)) + side * 0.2)
    U.camera(cam, forward=look, up=upv, fov=44)
    sun = norm(tang + upv * 0.12 + side * 0.35)
    planet_common(U, t, sun, sun_col=(1.0, 0.9, 0.78), intensity=22.0, city=0.5, rot_speed=0.0)
    tr_end = base * 1.012 + tang * 0.26 - side * 0.02
    U.scene.update({'uTrail': 1.6 * (1 - smooth(ramp(lt, 0.0, 11.0))), 'uTrailA': tuple(tr_end),
                    'uTrailB': tuple(tr_end - tang * 0.14 + upv * 0.04)})
    U.grade('dawn', exposure=-1.1, flash=0.7 * math.exp(-lt / 0.35), flash_col=(1.0, 0.92, 0.8))
    sun_flare(U, sun, 0.9)
    relative(U)


def s_black(t, lt, U):
    U.camera((0, 0, 0), forward=(0, 0, -1))
    U.post['fade'] = 1.0


def _pending(t, lt, U):
    s_black(t, lt, U)


def fn(name, default=_pending):
    return globals().get(name, default)


SHOTS = [
    Shot('watch', 0.0, bar(5), 'watch', s_watch_open, 16),
    Shot('home', bar(5), bar(9), 'planet', s_home, 10),
    Shot('wren', bar(9), bar(13), 'planet', s_wren, 10),
    Shot('ignition', bar(13), bar(17), 'planet', s_ignite, 10),
    Shot('rings', bar(17), bar(21), 'rings', None, 8),
    Shot('wall', bar(21), bar(25), 'storm', None, 8),
    Shot('storm', bar(25), bar(29), 'storm', None, 8),
    Shot('slip', bar(29), bar(30), 'watch', s_watch_slip, 12),
    Shot('eye', bar(30), bar(33), 'storm', None, 8),
    Shot('break', bar(33), bar(34), 'watch', s_watch_break, 12),
    Shot('maelstrom', bar(34), bar(37), 'storm', None, 8),
    Shot('lost', bar(37), bar(39), 'void', None, 10),
    Shot('heartbeat', bar(39), bar(40), 'watch', s_watch_heart, 12),
    Shot('light', bar(40), bar(41), 'void', None, 10),
    Shot('lodestar', bar(41), bar(45), 'storm', None, 8),
    Shot('breakthrough', bar(45), bar(49), 'storm', None, 8),
    Shot('homeagain', bar(49), bar(53), 'planet', None, 10),
    Shot('descent', bar(53), bar(57), 'planet', None, 10),
    Shot('homecoming', bar(57), bar(61), 'planet', None, 10),
    Shot('still', bar(61), bar(64), 'watch', s_watch_still, 16),
    Shot('title', bar(64), END, 'void', None, 6),
]
SHOT_FNS = {'rings': 's_rings', 'wall': 's_wall', 'storm': 's_storm', 'eye': 's_eye', 'maelstrom': 's_maelstrom',
            'lost': 's_lost', 'light': 's_light', 'lodestar': 's_lode', 'breakthrough': 's_breakthrough',
            'homeagain': 's_home_again', 'descent': 's_descent', 'homecoming': 's_homecoming', 'title': 's_title'}
for _s in SHOTS:
    if _s.fn is None:
        _f = globals().get(SHOT_FNS[_s.name])
        if _f is None:
            _s.fn, _s.scene = _pending, 'planet'
        else:
            _s.fn = _f


def shot_at(t):
    for s in SHOTS:
        if s.start <= t < s.end:
            return s
    return SHOTS[-1]
