"""Observable policy observation. Robot proprioception + tactile estimate only.

No object pose/twist, no GT lateral error or relative object motion, no object height.
"""

from __future__ import annotations

import numpy as np

from controllers.gripper_controller import finger_opening
from envs.deterioration import body_twist
from envs.physical_recovery import (
    SCALE_AP,
    SCALE_EX,
    SCALE_F,
    SCALE_G,
    SCALE_TAU,
    SCALE_V,
    SCALE_W,
    SCALE_Z,
    TAU_SECURE,
)
from training.replay_core import TABLE_TOP

# Pad-height half-extent from panda_torque fingertip_pad_collision_1 (8.5 mm).
SCALE_PAD_V = 0.0085
# Finger-slide rate. Same order as recovery v scale; PROVISIONAL.
SCALE_APDOT = 0.080
# Hand-z reference is TABLE_TOP (scene), not object height.
HAND_Z_REF = TABLE_TOP  # 0.40 m
SCALE_TRACK_Z = 0.080  # m, same as v_z command range

OBS_NAMES = [
    "e_hat_x",
    "estimate_valid",
    "contact_present_L",
    "contact_present_R",
    "valid_L",
    "valid_R",
    "u_L",
    "v_L",
    "u_R",
    "v_R",
    "fn_L",
    "fn_R",
    "aperture",
    "ap_dot",
    "tau_from_secure",
    "g_hx",
    "g_hy",
    "g_hz",
    "v_hx",
    "v_hy",
    "v_z_world",
    "w_hx",
    "w_hy",
    "w_hz",
    "hand_z_off_table",
    "track_z",
    "e_hat_dot",
]
OBS_DIM_OBSERVABLE = len(OBS_NAMES)  # 27

OBS_SOURCES = {
    "e_hat_x": "ESTIMATED geometric tactile CoP; 0 placeholder if invalid",
    "estimate_valid": "ESTIMATED mask; 0 => e_hat_x/e_hat_dot placeholders",
    "contact_present_L": "SENSOR finger-object geom contact",
    "contact_present_R": "SENSOR",
    "valid_L": "SENSOR CoP defined (Fn sum >= 0.01 N)",
    "valid_R": "SENSOR",
    "u_L": "SENSOR pad-frame X; 0 if invalid_L",
    "v_L": "SENSOR pad-frame Z; 0 if invalid_L",
    "u_R": "SENSOR",
    "v_R": "SENSOR",
    "fn_L": "SENSOR mj_contactForce abs wrench[0] sum",
    "fn_R": "SENSOR",
    "aperture": "DIRECT finger joint mean",
    "ap_dot": "DIRECT mean finger qvel",
    "tau_from_secure": "DIRECT gripper ctrl[7]",
    "g_hx": "DIRECT R_hand^T g_world (FK + known gravity)",
    "g_hy": "DIRECT",
    "g_hz": "DIRECT",
    "v_hx": "DIRECT hand linear vel in hand frame",
    "v_hy": "DIRECT",
    "v_z_world": "DIRECT hand world-z velocity",
    "w_hx": "DIRECT hand angular vel in hand frame",
    "w_hy": "DIRECT",
    "w_hz": "DIRECT",
    "hand_z_off_table": "DIRECT hand_z - TABLE_TOP (scene, not object)",
    "track_z": "DIRECT p_des_z - hand_z",
    "e_hat_dot": "ESTIMATED (e_hat - e_hat_prev)/dt_policy if both valid else 0",
}


class ObservableObsState:
    def __init__(self, dt_policy: float):
        self.dt = float(dt_policy)
        self.prev_e = float("nan")
        self.prev_valid = False

    def reset(self) -> None:
        self.prev_e = float("nan")
        self.prev_valid = False


def observe_observable(
    model, data, ids, fsm, tactile: dict, hist: ObservableObsState
) -> np.ndarray:
    """Build clipped obs from tactile dict + robot FK/proprio. No object body reads."""
    dt = max(hist.dt, 1e-9)
    valid = bool(tactile["estimate_valid"])
    e_hat = float(tactile["e_hat_x"]) if valid else 0.0
    if valid and hist.prev_valid:
        e_dot = (float(tactile["e_hat_x"]) - float(hist.prev_e)) / dt
    else:
        e_dot = 0.0
    if valid:
        hist.prev_e = float(tactile["e_hat_x"])
        hist.prev_valid = True
    else:
        hist.prev_e = float("nan")
        hist.prev_valid = False

    vL = bool(tactile["valid_L"])
    vR = bool(tactile["valid_R"])
    u_L = float(tactile["u_L"]) if vL else 0.0
    v_Lp = float(tactile["v_L"]) if vL else 0.0
    u_R = float(tactile["u_R"]) if vR else 0.0
    v_Rp = float(tactile["v_R"]) if vR else 0.0

    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(data.xpos[ids.hand_body], float)
    g_w = np.array(model.opt.gravity, float)
    g_h = Rh.T @ g_w
    vh, wh = body_twist(model, data, ids.hand_body)
    v_h = Rh.T @ vh
    w_h = Rh.T @ wh
    ap = finger_opening(data, ids)
    ap_dot = float(np.mean(data.qvel[ids.finger_dof]))
    tau = float(data.ctrl[7]) if data.ctrl.size > 7 else 0.0
    tau_n = (tau - TAU_SECURE) / SCALE_TAU
    p_des = np.asarray(fsm.p_des, float)
    track_z = float(p_des[2] - ph[2])
    hand_z_off = float(ph[2] - HAND_Z_REF)

    raw = np.array(
        [
            e_hat / SCALE_EX,
            1.0 if valid else 0.0,
            1.0 if tactile["contact_present_L"] else 0.0,
            1.0 if tactile["contact_present_R"] else 0.0,
            1.0 if vL else 0.0,
            1.0 if vR else 0.0,
            u_L / SCALE_EX,
            v_Lp / SCALE_PAD_V,
            u_R / SCALE_EX,
            v_Rp / SCALE_PAD_V,
            float(tactile["fn_L"]) / SCALE_F,
            float(tactile["fn_R"]) / SCALE_F,
            ap / SCALE_AP,
            ap_dot / SCALE_APDOT,
            tau_n,
            g_h[0] / SCALE_G,
            g_h[1] / SCALE_G,
            g_h[2] / SCALE_G,
            v_h[0] / SCALE_V,
            v_h[1] / SCALE_V,
            vh[2] / SCALE_V,
            w_h[0] / SCALE_W,
            w_h[1] / SCALE_W,
            w_h[2] / SCALE_W,
            hand_z_off / SCALE_Z,
            track_z / SCALE_TRACK_Z,
            e_dot / SCALE_V,
        ],
        dtype=np.float32,
    )
    if raw.shape[0] != OBS_DIM_OBSERVABLE:
        raise RuntimeError("obs dim mismatch")
    return np.clip(raw, -1.0, 1.0)
