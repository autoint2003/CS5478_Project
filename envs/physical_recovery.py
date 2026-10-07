"""Physical recovery observables, gates, and success (no D_t authority)."""

from __future__ import annotations

import numpy as np

from controllers.gripper_controller import finger_opening
from envs.deterioration import body_twist
from training.replay_core import TABLE_TOP, obs_from_sim


# Persistence in seconds (policy-rate counted via dt_policy).
SUCCESS_HOLD = 0.20
CONTACT_LOSS_HOLD = 0.15

E_TOL = 0.003
V_REL_TOL = 0.02  # tightened vs RULE yaml 0.08; matches RULE success finals ~0.017
# Provisional: existing frozen-RULE CSV logs have v_rel but not omega_rel, so
# 0.5 rad/s is not empirically calibrated from those trajectories.
W_REL_TOL = 0.50
Z_AIR = 0.48
CLEAR_MIN = 0.010
TABLE_DROP = TABLE_TOP + 0.03 + 0.01  # half-height + margin

# Per-component observation scales (physical, not a shared |r_h| scale).
SCALE_EX = 0.0075  # r_h.x = e_x; recovery offset band (~7.5 mm)
SCALE_EY = 0.0075  # r_h.y; same lateral resolution
# Centered pinch: object origin along the gripper axis in the hand frame.
# Absolute r_h.z ~ 0.099 m must not be divided by SCALE_EX (that clips at +1).
RH_Z_NOM = 0.099  # m, nominal centered grasp geometry
SCALE_RH_Z = 0.020  # m, axial deviation about RH_Z_NOM
SCALE_V = 0.080  # m/s, recovery4d v_hx/v_z command range
SCALE_W = 5.0  # rad/s; simulated recovery |omega_y|<=4 plus modest headroom. Not hardware.
SCALE_G = 9.81  # m/s^2
SCALE_AP = 0.040  # m
TAU_SECURE = -18.0
TAU_OPEN = -1.0
SCALE_TAU = TAU_OPEN - TAU_SECURE  # 17 N; obs 0 at secure, +1 at open
SCALE_F = 20.0  # N
SCALE_Z = 0.25  # m above table
SCALE_CLEAR = 0.15  # m
E_SCALE = 0.0075  # potential |e_x| scale (training offset band)


def mat6(r: np.ndarray) -> np.ndarray:
    return np.asarray(r, float).reshape(9)[:6]


def physical_pack(sim) -> dict:
    o = obs_from_sim(sim)
    ids = sim.ids
    d = sim.data
    Rh = o["Rh"]
    vo, wo = body_twist(sim.model, d, ids.object_body)
    vh, wh = body_twist(sim.model, d, ids.hand_body)
    v_rel_w = vo - vh
    w_rel_w = wo - wh
    v_rel_h = Rh.T @ v_rel_w
    w_rel_h = Rh.T @ w_rel_w
    R_obj = np.array(d.xmat[ids.object_body].reshape(3, 3), float)
    R_rel = Rh.T @ R_obj
    g_w = np.array(sim.model.opt.gravity, float)
    g_h = Rh.T @ g_w
    tau = float(d.ctrl[7]) if d.ctrl.size > 7 else 0.0
    o.update(
        {
            "v_rel_h": v_rel_h,
            "w_rel_h": w_rel_h,
            "R_rel": R_rel,
            "g_h": g_h,
            "aperture": finger_opening(d, ids),
            "tau": tau,
            "w_rel": float(np.linalg.norm(w_rel_w)),
        }
    )
    return o


def recovered(o: dict) -> bool:
    return (
        abs(float(o["e_x"])) <= E_TOL
        and int(o["nL"]) > 0
        and int(o["nR"]) > 0
        and float(o["v_rel"]) < V_REL_TOL
        and float(o["w_rel"]) < W_REL_TOL
        and int(o["scene"]) == 0
        and float(o["obj_z"]) >= Z_AIR
        and float(o["clear"]) >= CLEAR_MIN
    )


def fail_kind(o: dict, lost_s: float) -> str | None:
    if int(o["scene"]) > 0:
        return "scene"
    if float(o["obj_z"]) < TABLE_DROP or float(o["clear"]) < -0.005:
        return "drop"
    if int(o["nL"]) == 0 and int(o["nR"]) == 0 and lost_s >= CONTACT_LOSS_HOLD:
        return "contact_loss"
    return None


def observe_recovery4d(sim) -> np.ndarray:
    """Compact normalized observation. No mass/friction. No D_t (see module note)."""
    o = physical_pack(sim)
    rh = np.asarray(o["rh"], float)
    tau_n = (float(o["tau"]) - TAU_SECURE) / SCALE_TAU
    raw = np.concatenate(
        [
            [rh[0] / SCALE_EX, rh[1] / SCALE_EY, (rh[2] - RH_Z_NOM) / SCALE_RH_Z],
            np.asarray(o["v_rel_h"], float) / SCALE_V,
            mat6(o["R_rel"]),
            np.asarray(o["w_rel_h"], float) / SCALE_W,
            np.asarray(o["g_h"], float) / SCALE_G,
            [o["aperture"] / SCALE_AP],
            [tau_n],
            [float(o["Fn_L"]) / SCALE_F, float(o["Fn_R"]) / SCALE_F],
            [(float(o["obj_z"]) - TABLE_TOP) / SCALE_Z],
            [float(o["clear"]) / SCALE_CLEAR],
        ]
    ).astype(np.float32)
    return np.clip(raw, -1.0, 1.0)


# D_t omitted: r_h, v_rel, R_rel, contacts already cover detector ingredients.
OBS_DIM = 3 + 3 + 6 + 3 + 3 + 1 + 1 + 2 + 1 + 1  # 24
