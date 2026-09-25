"""Adult gloved researcher/engineer hands for the making sequence, built on lib/sdfhand.py (child skeleton,
scaled x1.25 in Blender). Poses solved with sdfhand's IK so fingertip pads land on real contact targets.
usage: python3 glovehand.py <pose> <out.npz>      poses: knob, pinch, rest
All targets below are in sdfhand (child) units: divide real metres by SCALE."""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdfhand as H

SCALE = 1.25

# fine-focus knob: axis points into the palm (-Z), rim radius 15 mm real
KNOB_C = np.array([-0.012, 0.086, -0.030])
KNOB_R = 0.0150 / SCALE


def rim(ang_deg, dz=0.0):
    a = math.radians(ang_deg)
    return KNOB_C + np.array([math.cos(a) * KNOB_R, math.sin(a) * KNOB_R, dz])


# component pinch (s08): part long axis along hand X-ish between thumb and index pads, width 14 mm real
PINCH_C = np.array([-0.022, 0.080, -0.030])
PINCH_HALF = 0.0072 / SCALE


# s03a: index fingertip pad rolling the top of the fine-focus knurl (knob axis = hand X, palm down)
TAP_K = np.array([-0.024, 0.120, -0.035])
TAP_R = 0.0150 / SCALE


def poses():
    return {
        "tap": dict(index=("pad", TAP_K + np.array([0.0, 0.0, TAP_R]), 0.0), middle=((0.55, 0.75, 0.45), 0.03),
                    ring=((0.75, 0.95, 0.55), -0.02), pinky=((0.85, 1.05, 0.6), -0.08),
                    thumb=(0.35, 0.45, (0.15, 0.3, 0.3), 0.8)),
        # index + middle on the far rim, thumb on the near rim (turning the knob like a jar lid seen edge-on)
        "knob": dict(index=("pad", rim(80.0, -0.001), 0.0), middle=("pad", rim(40.0, -0.001), 0.05),
                     ring=((0.9, 1.05, 0.6), -0.04), pinky=((1.0, 1.15, 0.65), -0.08),
                     thumb=("pad", rim(215.0, -0.001))),
        # precision pinch of the component's MCP boss flats between thumb and index
        "pinch": dict(index=("pad", PINCH_C + np.array([0.0, 0.0, 0.0]) + np.array([PINCH_HALF * 0.3, PINCH_HALF, 0.0]), 0.0),
                      middle=((0.75, 0.95, 0.5), 0.02), ring=((0.95, 1.1, 0.6), -0.04), pinky=((1.05, 1.2, 0.65), -0.1),
                      thumb=("pad", PINCH_C - np.array([PINCH_HALF * 0.3, PINCH_HALF, 0.0]))),
        "rest": dict(index=((0.3, 0.4, 0.25), 0.02), middle=((0.35, 0.45, 0.3), 0.0), ring=((0.4, 0.5, 0.3), -0.03),
                     pinky=((0.45, 0.55, 0.35), -0.08), thumb=(0.15, 0.35, (0.1, 0.25, 0.2), 0.5)),
    }


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    V, F, A, N = H.make(poses()[name], 0.00035)
    meta = dict(tap_k=(TAP_K * SCALE).tolist(), tap_r=TAP_R * SCALE, knob_c=(KNOB_C * SCALE).tolist(), knob_r=KNOB_R * SCALE, pinch_c=(PINCH_C * SCALE).tolist())
    np.savez_compressed(out, V=V, F=F, A=A, N=json.dumps(N), META=json.dumps(meta))
    print("verts", len(V))
