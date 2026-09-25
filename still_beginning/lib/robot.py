"""Original 6-axis industrial arm (white shells, graphite joint housings, amber anodized joint rings) with a
parallel gripper. FK is a chain of empties; IK is damped least squares on our own FK (mathutils only).
Joint layout (metres): J1 yaw at base (Z), J2 shoulder pitch at (0, 0, H1) offset A1 forward, upper arm L2,
J3 elbow pitch, forearm L3 to the spherical wrist (J4 roll, J5 pitch, J6 roll), flange -> tool."""
import bpy, math
import numpy as np
from mathutils import Vector as V, Matrix, Quaternion, Euler
import sb, props

H1, A1, L2, A3, L3, L6 = 0.42, 0.10, 0.58, 0.08, 0.56, 0.10
TOOL = 0.155          # flange -> fingertip centre (gripper)
LIMITS = [(-3.0, 3.0), (-1.9, 1.3), (-1.2, 2.6), (-3.2, 3.2), (-2.2, 2.2), (-6.3, 6.3)]


def mats():
    white = sb.painted("RobWhite", (0.72, 0.72, 0.71), rough=0.30, coat=0.5, grime=0.06, scale=4.0)
    graph = sb.painted("RobGraphite", (0.045, 0.047, 0.052), rough=0.38, coat=0.25)
    amber = props.amber_anodized("RobAmber")
    alu = sb.brushed_metal("RobAlu", (0.70, 0.71, 0.72), rough=0.26, aniso=0.5)
    rub = sb.rubber("RobRubber", (0.03, 0.03, 0.033))
    return white, graph, amber, alu, rub


def _cap(name, r, L, mat, axis='Z', r2=None, loc=(0, 0, 0), parent=None, bevel=0.012):
    """rounded cylinder-ish link segment."""
    r2 = r if r2 is None else r2
    prof = [(0.0, 0.0), (r - bevel, 0.0), (r, bevel), (r2, L - bevel), (r2 - bevel, L), (0.0, L)]
    o = sb.lathe(name, prof, segs=64, mat=mat)
    if axis == 'X':
        o.rotation_euler = (0, math.pi / 2, 0)
    elif axis == 'Y':
        o.rotation_euler = (-math.pi / 2, 0, 0)
    elif axis == '-Y':
        o.rotation_euler = (math.pi / 2, 0, 0)
    o.location = loc
    if parent:
        o.parent = parent
    return o


def _joint(name, r, w, graph, amber, parent, axis='Y', loc=(0, 0, 0)):
    """joint housing (graphite drum) with two amber anodized rings at its faces."""
    ax = {'Y': (-math.pi / 2, 0, 0), 'X': (0, math.pi / 2, 0), 'Z': (0, 0, 0)}[axis]
    d = sb.lathe(name, [(0.0, -w / 2), (r - 0.01, -w / 2), (r, -w / 2 + 0.01), (r, w / 2 - 0.01), (r - 0.01, w / 2), (0.0, w / 2)], segs=64, mat=graph)
    d.rotation_euler = ax; d.location = loc; d.parent = parent
    for s in (-1, 1):
        rg = sb.lathe(name + "Ring", [(r * 0.62, 0.0), (r * 0.70, 0.0), (r * 0.70, 0.004), (r * 0.62, 0.004)], segs=96, mat=amber)
        rg.rotation_euler = ax
        off = V((0, 0, s * (w / 2 + 0.0005) - (0.002 if s > 0 else -0.002)))
        rg.location = V(loc) + (Euler(ax).to_matrix() @ off)
        rg.parent = parent
    return d


class Robot:
    def __init__(self, name="Rob", loc=(0, 0, 0), yaw=0.0):
        white, graph, amber, alu, rub = mats()
        self.root = sb.empty(name, loc=loc)
        self.root.rotation_euler = (0, 0, yaw)
        import prosthesis as P, os as _os
        RD = _os.path.join(sb.ROOT, "assets", "robot")

        def shell(link, parent, mat):
            o = P.load_npz(_os.path.join(RD, link + ".npz"), name + link.capitalize(), mat)
            o.parent = parent
            return o

        def cap(nm, parent, c, axis, r, y_face, sgn):
            """graphite cover disc on a yoke face + amber anodized ring (the curved amber highlight)."""
            rot = (-sgn * math.pi / 2, 0, 0) if axis == 'Y' else (0, sgn * math.pi / 2, 0)
            d = sb.lathe(nm, [(0.0, 0.0), (r - 0.006, 0.0), (r, 0.004), (r, 0.009), (r - 0.004, 0.012), (0.0, 0.012)], segs=96, mat=graph)
            d.rotation_euler = rot
            off = V((0, sgn * (y_face - 0.006), 0)) if axis == 'Y' else V((sgn * (y_face - 0.006), 0, 0))
            d.location = V(c) + off
            d.parent = parent
            rg = sb.lathe(nm + "Amb", [(r * 0.70, 0.0110), (r * 0.78, 0.0110), (r * 0.78, 0.0135), (r * 0.70, 0.0135)], segs=128, mat=amber)
            rg.rotation_euler = rot; rg.location = d.location; rg.parent = parent
            hub = sb.lathe(nm + "Hub", [(0.0, 0.012), (r * 0.25, 0.012), (r * 0.25, 0.016), (0.0, 0.017)], segs=64, mat=alu)
            hub.rotation_euler = rot; hub.location = d.location; hub.parent = parent

        # base: graphite plinth with a white collar
        _cap(name + "Base", 0.19, 0.16, graph, parent=self.root, bevel=0.02)
        _cap(name + "BaseTop", 0.168, 0.05, white, loc=(0, 0, 0.16), parent=self.root, bevel=0.015)
        rg0 = sb.lathe(name + "J1Ring", [(0.1535, 0.205), (0.1575, 0.205), (0.1575, 0.211), (0.1535, 0.211)], segs=128, mat=amber)
        rg0.parent = self.root
        j1 = sb.empty(name + "J1", loc=(0, 0, 0.21), parent=self.root); j1.rotation_mode = 'XYZ'
        shell("turret", j1, white)
        for sg in (-1, 1):
            cap(name + "J2Cap%d" % sg, j1, (A1, 0, 0.21), 'Y', 0.118, 0.150, sg)
        j2 = sb.empty(name + "J2", loc=(A1, 0, H1 - 0.21), parent=j1); j2.rotation_mode = 'XYZ'
        shell("upper", j2, white)
        cond = sb.tube_along(name + "Conduit", [(-0.085, 0.0, 0.10), (-0.11, 0.0, L2 * 0.5), (-0.09, 0.0, L2 - 0.12)], radius=0.016, mat=graph)
        cond.parent = j2
        for sg in (-1, 1):
            cap(name + "J3Cap%d" % sg, j2, (0, 0, L2), 'Y', 0.085, 0.097, sg)
        j3 = sb.empty(name + "J3", loc=(0, 0, L2), parent=j2); j3.rotation_mode = 'XYZ'
        shell("fore", j3, white)
        j4 = sb.empty(name + "J4", loc=(L3 * 0.55, 0, A3), parent=j3); j4.rotation_mode = 'XYZ'      # forearm roll (X)
        r4 = sb.lathe(name + "J4Ring", [(0.0605, -0.003), (0.0640, -0.003), (0.0640, 0.003), (0.0605, 0.003)], segs=128, mat=amber)
        r4.rotation_euler = (0, math.pi / 2, 0); r4.parent = j4
        shell("wrist", j4, white)
        for sg in (-1, 1):
            cap(name + "J5Cap%d" % sg, j4, (L3 * 0.45, 0, 0), 'Y', 0.050, 0.066, sg)
        j5 = sb.empty(name + "J5", loc=(L3 * 0.45, 0, 0), parent=j4); j5.rotation_mode = 'XYZ'      # wrist pitch (Y)
        shell("hand", j5, graph)
        j6 = sb.empty(name + "J6", loc=(L6, 0, 0), parent=j5); j6.rotation_mode = 'XYZ'             # flange roll (X)
        fl = _cap(name + "Flange", 0.042, 0.012, alu, axis='X', parent=j6, bevel=0.002)
        flr = sb.lathe(name + "FlangeRing", [(0.043, 0.0), (0.047, 0.0), (0.047, 0.004), (0.043, 0.004)], segs=96, mat=amber)
        flr.rotation_euler = (0, math.pi / 2, 0); flr.location = (0.004, 0, 0); flr.parent = j6
        self.joints = [j1, j2, j3, j4, j5, j6]
        self.axes = ['Z', 'Y', 'Y', 'X', 'Y', 'X']
        # gripper: body along +X of j6; fingers open/close along Y
        gb = sb.prim("cube", name + "GripBody", loc=(0.05, 0, 0), scale=(0.035, 0.045, 0.028), mat=graph, parent=j6)
        sb.bevel(gb, 0.008, 3)
        rail = sb.prim("cube", name + "GripRail", loc=(0.087, 0, 0), scale=(0.004, 0.05, 0.012), mat=alu, parent=j6)
        sb.bevel(rail, 0.002)
        self.fingers = []
        for s in (-1, 1):
            fe = sb.empty(name + "Finger%d" % s, loc=(0.09, s * 0.02, 0), parent=j6)
            # finger: vertical plate down to the tip; inner pad (rubber) with a V-groove face
            pl = sb.prim("cube", name + "FingerPlate", loc=(0.034, s * 0.004, 0), scale=(0.036, 0.004, 0.011), mat=alu, parent=fe)
            sb.bevel(pl, 0.0015, 2)
            tip = sb.prim("cube", name + "FingerTip", loc=(0.061, -s * 0.0005, 0), scale=(0.011, 0.0035, 0.009), mat=white, parent=fe)
            sb.bevel(tip, 0.002, 2)
            pad = sb.prim("cube", name + "FingerPad", loc=(0.061, -s * 0.0038, 0), scale=(0.009, 0.0008, 0.0075), mat=rub, parent=fe)
            sb.bevel(pad, 0.0005, 2)
            self.fingers.append((fe, s))
        self.name = name

    # ---------------------------------------------------------------- kinematics (pure math, no depsgraph)
    def fk(self, q):
        """returns world matrix of the tool centre point (between fingertip pads)."""
        M = self.root.matrix_world.copy()
        locs = [V((0, 0, 0.21)), V((A1, 0, H1 - 0.21)), V((0, 0, L2)), V((L3 * 0.55, 0, A3)), V((L3 * 0.45, 0, 0)), V((L6, 0, 0))]
        for loc, ax, a in zip(locs, self.axes, q):
            M = M @ Matrix.Translation(loc) @ Matrix.Rotation(a, 4, ax)
        return M @ Matrix.Translation((TOOL, 0, 0))

    def set(self, q, frame=None):
        for j, ax, a in zip(self.joints, self.axes, q):
            e = [0.0, 0.0, 0.0]
            e['XYZ'.index(ax)] = a
            j.rotation_euler = e
            if frame is not None:
                j.keyframe_insert("rotation_euler", frame=frame)

    def grip(self, half_open, frame=None):
        """half_open = distance from the TCP centre to each pad face (m)."""
        for fe, s in self.fingers:
            fe.location = (0.09, s * (half_open + 0.0046), 0)
            if frame is not None:
                fe.keyframe_insert("location", frame=frame)

    def ik(self, target, q0, iters=60):
        """target: 4x4 world matrix for the TCP (x = approach axis, y = finger closing axis)."""
        q = np.array(q0, float)
        tp = np.array(target.translation)
        tR = target.to_3x3()

        def err(qv):
            M = self.fk(qv)
            dp = np.array(M.translation) - tp
            R = M.to_3x3()
            # orientation error: compare approach (x) and closing (y) axes
            ex = np.array(R.col[0] - tR.col[0]); ey = np.array(R.col[1] - tR.col[1])
            return np.concatenate([dp * 10.0, ex * 0.6, ey * 0.6])
        lam = 0.01
        for it in range(iters):
            e = err(q)
            if np.linalg.norm(e) < 1e-5:
                break
            J = np.zeros((len(e), 6))
            for i in range(6):
                d = np.zeros(6); d[i] = 1e-4
                J[:, i] = (err(q + d) - e) / 1e-4
            dq = -np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ e)
            q = q + np.clip(dq, -0.2, 0.2)
            q = np.array([min(max(a, lo), hi) for a, (lo, hi) in zip(q, LIMITS)])
        return q, float(np.linalg.norm(err(q)[:3]) / 10.0)


def tcp_matrix(pos, approach=(0, 0, -1), close_dir=(1, 0, 0)):
    x = V(approach).normalized()
    y = V(close_dir); y = (y - x * y.dot(x)).normalized()
    z = x.cross(y)
    M = Matrix((x, y, z)).transposed().to_4x4()
    M.translation = V(pos)
    return M
