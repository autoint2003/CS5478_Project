"""Contact-preserving slide and privileged chase commands.

controlled sliding keeps contact. The privileged command is the historical
regrasp and chase law. They are not the same controller.
"""

from __future__ import annotations

from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec

import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3, grip_tau, normalize_action
from data_generation.ball_impact_dataset import grasp_sample
from envs.outcomes.physical_recovery import physical_pack


STRATEGIES = (
    {"name": "lift_squeeze", "kp": 4.0, "ff": 1.0, "close": 1.0, "partial": 1.0, "open_m": -1.0, "wrist": 0.8},
    {"name": "lift_slide", "kp": 12.0, "ff": 1.0, "close": 1.0, "partial": 1.0, "open_m": -1.0, "wrist": 0.8},
    {"name": "lift_soft", "kp": 12.0, "ff": 0.4, "close": 1.0, "partial": 0.2, "open_m": 0.03, "wrist": 0.4},
    {"name": "lift_follow_spin", "kp": 8.0, "ff": 1.0, "close": 1.0, "partial": 1.0, "open_m": -1.0, "wrist": 0.4, "spin_follow": True},
    {"name": "unilateral_keep", "kp": 8.0, "ff": 1.0, "close": 0.9, "partial": 0.9, "open_m": -1.0, "wrist": 0.6},
    {"name": "release_short", "kp": 14.0, "ff": 1.2, "close": 1.0, "partial": -0.2, "open_m": 0.02, "wrist": 0.3, "force_open": 3},
    {"name": "release_long", "kp": 16.0, "ff": 1.2, "close": 1.0, "partial": -1.0, "open_m": 0.03, "wrist": 0.2, "force_open": 8},
    {"name": "chase_lost", "kp": 18.0, "ff": 1.5, "close": 1.0, "partial": -1.0, "open_m": 0.012, "wrist": 0.2},
    {"name": "slow_return", "kp": 6.0, "ff": 0.8, "close": 1.0, "partial": 0.5, "open_m": 0.025, "wrist": 2.0},
)


def hand_pose(sim):
    p = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    R = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    return p, R




def legal_action_audit():
    eye = np.eye(3)
    rh = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], float)
    down = map_action([0, 0, 1, 0, 0, 0, 1], eye, rh, AIRBORNE_V3)
    up = map_action([0, 0, -1, 0, 0, 0, -1], eye, rh, AIRBORNE_V3)
    spin = map_action([0, 0, 0, 1, 0, 0, 0], eye, rh, AIRBORNE_V3)
    return {
        "version": AIRBORNE_V3,
        "vx_vy_mps": 1.0,
        "vz_down_mps": float(-down["v_world"][2]),
        "vz_up_mps": float(up["v_world"][2]),
        "wrist_rad_s": float(W_MAX),
        "wrist_command_norm": float(np.linalg.norm(spin["w_world"])),
        "grip_open_nm": float(RECOVERY4D_TAU_OPEN),
        "grip_secure_nm": float(RECOVERY4D_TAU_SECURE),
        "ball_parked": False,
        "qpos_written": False,
    }


def _axis_angle(rotation):
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or angle < 1e-5:
        return 0.0, np.zeros(3)
    return angle, axis / norm


def wrist_return(sim, ref):
    current = hand_R(sim)
    target = np.asarray(ref["R_hand"], float).reshape(3, 3)
    angle, axis = _axis_angle(current.T @ target)
    desired = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    return desired.T @ (current @ (axis * angle))


def command(sim, ref, spec, index):
    pinch, _relative = hand_pose(sim)
    error = np.asarray(pinch, float) - np.asarray(ref["p_nom"], float)
    pack = physical_pack(sim)
    relative = np.asarray(pack["v_rel_h"], float)
    both = int(pack["nL"]) + int(pack["nR"])
    distance = float(np.linalg.norm(error))
    settling = both > 0 and distance <= 0.0032
    loose = float(spec["partial"]) < 0.5
    if int(spec.get("force_open", 0)) > index and distance > 0.0032:
        grip = -1.0
        vel = float(spec["kp"]) * error + float(spec.get("ff", 1.0)) * relative
    elif settling:
        grip = float(spec["close"])

def slide_action(sim, switch, ref, spec, memory):
    pinch, _relative = hand_pose(sim)
    error = np.asarray(pinch, float) - np.asarray(ref["p_nom"], float)
    distance = float(np.linalg.norm(error))
    sample, anchor = grasp_sample(sim, switch, memory.get("anchor"))
    memory["anchor"] = anchor
    contacts = int(sample["nL"]) + int(sample["nR"])
    normal = float(sample["fnL"]) + float(sample["fnR"])
    relative = np.asarray(physical_pack(sim)["v_rel_h"], float)
    grip = float(spec["grip"])
    scale = 1.0
    if contacts == 1:
        grip = max(grip, -0.40)
        scale = 0.35
    elif contacts == 0:
        grip = max(grip, -0.30)
        scale = 0.0
    elif spec.get("brake") and normal < 0.8:
        grip = min(0.0, grip + 0.10)
        scale = 0.45
    settling = distance <= 0.0032 and int(sample["nL"]) > 0 and int(sample["nR"]) > 0
    if settling:
        vel = 3.0 * error - relative
        height = float(physical_pack(sim)["obj_z"])
        if height < 0.49:
            lift = np.array([0.0, 0.0, float(np.clip(10.0 * (0.49 - height), 0.0, 0.85))], float)
            vel = vel + hand_R(sim).T @ lift
        grip = 1.0
    else:
        vel = scale * (float(spec["kp"]) * error + float(spec.get("ff", 0.0)) * relative)
        speed = float(np.linalg.norm(vel))
        cap = float(spec["cap"])
        if speed > cap:
            vel *= cap / max(speed, 1e-9)
    ang = 0.30 * wrist_return(sim, ref)
    return normalize_action(vel, ang, grip_tau(float(grip)), AIRBORNE_V3)
