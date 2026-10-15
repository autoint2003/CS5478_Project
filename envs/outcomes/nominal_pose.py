"""Hand-pose error against a captured nominal grasp.

errors() is the pose meter used by the handoff check. It does not decide SUCCESS_12S.
"""

from __future__ import annotations

import numpy as np

from controllers.recovery.geometry_switch import hand_pose, relative_speed, rot_angle
from envs.outcomes.physical_recovery import physical_pack


def rotvec(R):
    c = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    ang = float(np.arccos(c))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    if ang < 1e-8:
        return 0.5 * vee
    return vee * (ang / (2.0 * np.sin(ang)))


def hand_R(sim):
    return np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)


def capture_ref(sim, switch, live=False):
    p, rel = hand_pose(sim)
    return {
        "R_hand": hand_R(sim),
        "r_des": np.array(sim.fsm.r_des, float).copy(),
        "p_nom": np.array(p if live else switch.p0, float).copy(),
        "R_rel": np.array(rel if live else switch.r0, float).copy(),
        "t": float(sim.data.time),
    }


def errors(sim, ref):
    p, rel = hand_pose(sim)
    pack = physical_pack(sim)
    return {
        "t": round(float(sim.data.time), 4),
        "eR": float(rot_angle(ref["R_hand"], hand_R(sim))),
        "eR_rel": float(rot_angle(ref["R_rel"], rel)),
        "eR_des": float(rot_angle(ref["r_des"], np.array(sim.fsm.r_des, float))),
        "ep": float(np.linalg.norm(p - ref["p_nom"])),
        "vrel": float(relative_speed(sim)),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "z": float(pack["obj_z"]),
        "w_cmd": [round(float(x), 4) for x in np.asarray(sim.fsm.w_cmd, float).reshape(-1)[:3]],
    }


def closed_grip(sim):
    return bool(sim.data.ctrl.size > 7 and float(sim.data.ctrl[7]) < -8.0)
