"""s19a/b/c READY (1710-1800): close low engine hardware with chill-down vapor; tower + vehicle scale with restrained
venting; hold-down clamp insert where the lock bolt withdraws. No ignition (s20 ignites on 1800)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, sb, rocket, launchpad as LP, launchfx as FX
from mathutils import Vector as V

sid = (sb.argv() or ["s19a"])[0]
_, F0, F1, _ = sb.TL.shot(sid)
sc = sb.reset()
tile = {"preview": 4, "half": 2, "final": 4}[sb.RES]
sb.setup_render("EEVEE", samples=int(os.environ.get("SB_SAMPLES", "0")) or None, mblur=True, shutter=0.5,
                volumes=True, vol_tile=tile, vol_samples=96)
sc.eevee.volumetric_shadow_samples = 12
FX.environment(sc)
CX = LP.build_complex(far=(sid == "s19b"), trucks=True)
RK = rocket.build(frost=1.0)
RK["root"].location = (0, 0, LP.ROCKET_Z0)
FA, FB = F0 - 1, F1 + 1
FX.animate_pad(CX, FA, FB)
sc.view_settings.exposure = 0.35


def cam_move(cam, p0, p1, t0, t1, ease=True):
    def pos(t):
        k = sb.smoother(t) * 0.6 + t * 0.4 if ease else t
        return V(p0).lerp(V(p1), k)

    def tgt(t):
        k = sb.smoother(t) * 0.6 + t * 0.4 if ease else t
        return V(t0).lerp(V(t1), k)
    sb.cam_bake(cam, FA, FB, lambda t: pos((t * (FB - FA) - 1) / (F1 - 1 - F0)),
                lambda t: tgt((t * (FB - FA) - 1) / (F1 - 1 - F0)), focus="target")


if sid == "s19a":
    # close, low: under the aft skirt, engines + actuators; chill-down vapor spilling from the bells
    FX.prelaunch_vapor(RK, FA, FB, vents=False, skin=False, chill=True)
    Z = LP.ROCKET_Z0
    cam = sb.camera("Cam", loc=(3.0, -5.2, LP.TABLE_Z + 0.55), target=(0, 0, Z), lens=24, fstop=2.8, clip=(0.05, 5000))
    cam_move(cam, (3.15, -5.35, LP.TABLE_Z + 0.5), (2.75, -4.75, LP.TABLE_Z + 0.72), (0.35, 0.1, Z - 0.9), (0.2, 0.1, Z - 0.55))
    FX.volume_range(sc, cam, (0, 0, Z), near=0.1, far=200.0)
elif sid == "s19b":
    FX.prelaunch_vapor(RK, FA, FB, vents=True, skin=True, chill=True, amount=1.0)
    cam = sb.camera("Cam", loc=(58.0, -86.0, 1.7), target=(-2, 0, 38), lens=32, fstop=8.0, clip=(0.1, 20000))
    cam_move(cam, (60.5, -85.0, 1.7), (56.5, -87.0, 1.75), (-1.5, 0, 37.5), (-2.5, 0, 38.5), ease=False)
    FX.volume_range(sc, cam, (0, 0, 30), near=20.0, far=600.0)
else:  # s19c
    FX.prelaunch_vapor(RK, FA, FB, vents=False, skin=False, chill=True, amount=0.8)
    Z = LP.ROCKET_Z0
    pin = V((0.0, -rocket.LUG_PIN, Z + 0.56))
    cam = sb.camera("Cam", loc=(1.6, -5.6, Z + 0.2), target=pin, lens=55, fstop=2.2, clip=(0.05, 5000))
    cam_move(cam, (1.75, -5.75, Z + 0.15), (1.5, -5.35, Z + 0.22), pin + V((0.1, -0.2, -0.05)), pin + V((0.05, -0.15, 0.0)))
    FX.volume_range(sc, cam, pin, near=0.1, far=120.0)

sb.frames(F0, F1)
if os.environ.get("SB_SAVE"):
    sb.save(sid)
sb.render_shot(sid)
