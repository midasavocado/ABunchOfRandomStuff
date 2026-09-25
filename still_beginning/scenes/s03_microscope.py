"""s03a (180-214): gloved fingertips turning the fine-focus knob - cut on the focusing movement.
s03b (214-270): track down along the optics (turret, objective) to the sharply focused, transmitted-light specimen."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import bpy, json
import numpy as np
import sb, lab, dbgcam
from mathutils import Vector as V, Quaternion, Matrix, Euler

sid = (sb.argv() or ["s03a"])[0]
sc = sb.reset()
sb.setup_render("EEVEE", mblur=True, shutter=0.5, look="AgX - Medium High Contrast",
                exposure=float(os.environ.get("SB_EXPO", "0.0")))
sc.eevee.use_shadows = True

# ------------------------------------------------------------------ set
M = lab.microscope()
lab.lab_bench(0.0)
sb.world_gradient(top=(0.030, 0.036, 0.046), horizon=(0.055, 0.060, 0.068), bottom=(0.02, 0.02, 0.022), strength=0.45)

# lab light: big cool overhead panel (slightly behind -> rims on the enamel), window light from back-left,
# soft front fill; everything restrained graphite/silver
def nog(o):
    return o
sb.light('AREA', "Overhead", loc=(0.1, 0.35, 1.3), target=(0, 0.05, 0.2), energy=35, color=(0.86, 0.92, 1.0), size=1.2, size_y=0.5)
sb.light('AREA', "Window", loc=(-1.0, 0.9, 0.8), target=(0, 0, 0.2), energy=50, color=(0.78, 0.86, 1.0), size=1.0)
sb.light('AREA', "Fill", loc=(0.6, -0.9, 0.45), target=(0.05, 0.05, 0.2), energy=8, color=(0.95, 0.96, 1.0), size=1.0)
rim = sb.light('AREA', "RimStrip", loc=(-0.05, 0.45, 0.30), target=(0.07, 0.09, 0.13), energy=9, color=(0.85, 0.92, 1.0), size=0.3, size_y=0.03)
kick = sb.light('AREA', "Kick", loc=(0.35, 0.45, 0.25), target=(0.07, 0.09, 0.12), energy=6, color=(0.9, 0.95, 1.0), size=0.25)
# the condenser throws a small pool of light up through the slide onto the objective front
sb.light('SPOT', "CondSpot", loc=(0.0, -0.010, 0.195), target=(0.0, -0.010, 0.4), energy=0.08, color=(1.0, 0.97, 0.92), size=0.01, spot=40, blend=1.0)

# ------------------------------------------------------------------ gloved hand on the fine knob
npz = os.path.join(sb.ROOT, "assets", "glove_tap.npz")
meta = json.loads(str(np.load(npz)["META"]))
K_h = V(meta["tap_k"])                    # knob centre in hand units (already x1.25); knob axis = hand X
hand = lab.glove_hand(npz, "Glove")
sleeve = lab.lab_sleeve("Sleeve", hand)
axis_pt = M["fine_axis_pt"]
knob_mid = axis_pt + V((0.0069, 0, 0))
Xh = V((1.0, 0.0, 0.0))
Yh0 = None
Yh = V((-0.10, 0.93, 0.33)).normalized()
Zh = Xh.cross(Yh).normalized()
Xh = Yh.cross(Zh).normalized()
Rh = Matrix((Xh, Yh, Zh)).transposed()
hand_pivot = sb.empty("HandPivot", loc=axis_pt)
hand_pivot.rotation_mode = 'XYZ'
hand.parent = hand_pivot
hand.matrix_parent_inverse = Matrix.Identity(4)
hand_rot = Rh.to_4x4()
hand_rot.translation = (knob_mid - Rh @ (K_h * 1.0)) - axis_pt
hand.matrix_basis = hand_rot @ Matrix.Diagonal((1.25, 1.25, 1.25, 1.0))
hand.data.update()

# knob + hand turn together (rolling contact is preserved because both rotate about the knob axis)
F0, F1 = 180, 270


def knob_angle(f):
    # the focusing movement is already under way at the cut (non-zero velocity at 180), settles by ~205
    t = (f - 172) / (206 - 172)
    return -math.radians(34.0) * sb.smoother(t)


def hand_angle(f):
    return knob_angle(f) * 0.35          # fingertip stays on the rim (arc about the knob axis); knurl rolls under the pad


piv = M["fine"][1]
sb.bake(piv, F0 - 2, F1, lambda f: dict(rot=(knob_angle(f), 0, 0)))
sb.bake(hand_pivot, F0 - 2, F1, lambda f: dict(rot=(hand_angle(f), 0, 0)))
# hand contact is exact at the knob-angle = hand-angle pose; the rolling difference is kept small (< 8 deg)
# and hidden by the knurl.

# ------------------------------------------------------------------ cameras
cam = sb.camera("Cam", loc=(0.2, -0.1, 0.15), target=axis_pt, lens=100, fstop=4.0, clip=(0.002, 20))
cam.data.dof.aperture_blades = 9
if sid == "s03a":
    s0, s1 = 180, 214
    # macro from high behind-right: index fingertip pad on the knurled rim, graduated face and amber ring,
    # the coarse knob and the enamel pillar soft beyond. Slow drift in the direction of the turn.
    tgt = knob_mid + V((-0.004, -0.004, 0.012))

    def pos(t):
        k = sb.smooth(t)
        return tgt + V((sb.lerp(0.150, 0.142, k), sb.lerp(0.120, 0.108, k), sb.lerp(0.070, 0.062, k)))

    contact = knob_mid + V((0.0, 0.0, 0.0155))
    sb.cam_bake(cam, s0, s1, pos, lambda t: tgt + V((0, 0, -0.001 * t)), lens=lambda t: sb.lerp(85, 90, t),
                focus=lambda t: (pos(t) - contact).length, fstop=5.6)
else:
    s0, s1 = 214, 270
    obj_front = M["obj_front"]
    spec = M["slide_top"]
    # starts beside the turret (objective barrels, colour rings soft), travels down the optical axis to the
    # stage and ends low and close on the specimen glowing under the objective front lens.
    # objectives (turret, colour rings) -> down past the active objective -> looking down at the specimen,
    # front lens hovering at the top of frame, stained cells glowing in the field-diaphragm light.
    P = [V((-0.125, -0.150, 0.292)), V((-0.075, -0.095, 0.252)), V((-0.028, -0.052, 0.236)), V((-0.0080, -0.0255, 0.2275))]
    T = [V((0.0, -0.040, 0.262)), V((0.0, -0.022, 0.232)), V((0.0, -0.013, 0.207)), spec + V((0.0, -0.0005, 0.0012))]

    def u(t):
        return sb.smoother(t) * 0.8 + t * 0.2

    def pos(t):
        return sb.catmull(P, u(t))

    def tg(t):
        return sb.catmull(T, u(t))

    def foc(t):
        # rack focus: objectives early, the specimen in the last third
        k = sb.smooth(sb.remap(t, 0.45, 0.85))
        return sb.lerp((pos(t) - tg(t)).length, (pos(t) - (spec + V((0, -0.002, 0)))).length, k)

    sb.cam_bake(cam, s0, s1, pos, tg, lens=lambda t: sb.lerp(60, 92, sb.smooth(t)), focus=foc,
                fstop=lambda t: sb.lerp(5.6, 16.0, t))

dbgcam.apply()
sb.frames(s0, s1)
sb.save("s03")
sb.render_shot(sid)
