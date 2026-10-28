"""Canonical recovery success.

A rollout succeeds only after the measured nominal basin persists, control
hands back to the nominal controller, and that controller holds the grasp.
The timer starts at the handoff. Absolute t = 12 does not end the test.
RETAINED_TO_12 is recorded only when the simulation clock actually passes
that instant.
"""

from __future__ import annotations

from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec
from controllers.cartesian.tracking import TRACK

from pathlib import Path

import numpy as np

from envs.simulation.yaml_config import load_yaml
from envs.outcomes.physical_recovery import physical_pack
from training.recovery_episode import Z_ABORT
from envs.ballistic_impact import TAU_SEC, z_tgt_of


from training.recovery_execution import TABLE_DROP, _escaped
from controllers.cartesian.velocity_step import tick_vw
from controllers.recovery.geometry_switch import rot_angle

_CFG = load_yaml(Path(__file__).resolve().parents[2] / "config" / "recovery" / "recovery_success.yaml")

ORI_MAX = float(_CFG["orientation_rad"])
POS_MAX = float(_CFG["pinch_position_m"])
SPEED_MAX = float(_CFG["relative_speed_mps"])
PERSIST_S = float(_CFG["persistence_s"])
CONTROL_PERIOD_S = float(_CFG["control_period_s"])
PERSIST_STEPS = int(round(PERSIST_S / CONTROL_PERIOD_S))
PRIMARY_S = float(_CFG["primary_horizon_s"])
EXTENDED_S = float(_CFG["extended_horizon_s"])


def basin_ok(row, grip_closed):
    return bool(
        row["nL"] > 0 and row["nR"] > 0
        and row["eR"] <= ORI_MAX and row["ep"] <= POS_MAX
        and row["vrel"] <= SPEED_MAX and grip_closed
    )


# Max |ω| (rad/s) while smoothly returning r_des → saved grasp after HANDOFF.
_HANDOFF_R_RETURN_OMEGA = 0.8


def _omega_toward_R(R_cur: np.ndarray, R_tgt: np.ndarray, omega_max: float) -> np.ndarray:
    """World angular rate rotating R_cur toward R_tgt, clipped to omega_max."""
    R_err = np.asarray(R_tgt, float).reshape(3, 3) @ np.asarray(R_cur, float).reshape(3, 3).T
    tr = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(tr))
    if theta < 1e-6:
        return np.zeros(3)
    skew = 0.5 * (R_err - R_err.T)
    rotvec = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], float)
    ang = float(np.linalg.norm(rotvec))
    if ang < 1e-12:
        axis = np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]], float)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3)
        axis = axis / n
    else:
        axis = rotvec / ang
    return axis * min(float(omega_max), theta / 0.02)


def _handoff_grip_tau(sim) -> float:
    """Closed-gripper torque: keep continuity, never open; prefer at least |TAU_SEC|."""
    tau = float(TAU_SEC)
    if sim.data.ctrl.size > 7:
        live = float(sim.data.ctrl[7])
        # More negative = stronger close. Keep the stronger of live vs TAU_SEC.
        if live <= -1.0:
            tau = float(min(live, float(TAU_SEC)))
        else:
            tau = float(TAU_SEC)
        sim.data.ctrl[7] = tau
    return tau


def apply_handoff(sim, ref):
    """Hand control back to nominal without yanking a nonstandard recovery pose.

    Contract (validated P36, expert-terminal + IC paired audit):
    - Initialize p_des and r_des from the *actual* hand pose (no orientation snap).
    - Store r_des_target = saved grasp orientation; nominal_step slews toward it.
    - Force phase=lift / lift_started so the FSM cannot re-open the gripper.
    - Preserve/strengthen gripper closure (ctrl[7] continuous, ≤ TAU_SEC).
    qpos and qvel are not written.
    """
    qpos_before = np.array(sim.data.qpos, float).copy()
    qvel_before = np.array(sim.data.qvel, float).copy()
    time_before = float(sim.data.time)
    hand_before = hand_R(sim).copy()

    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    sim.fsm.r_des = hand_before.copy()
    sim.fsm.w_cmd = np.zeros(3)
    sim.fsm.w_des = np.zeros(3)
    sim.fsm.v_cmd = np.zeros(3)
    sim.fsm.v_des = np.zeros(3)

    sim.fsm.phase = "lift"
    sim.fsm.lift_started = True

    tau = _handoff_grip_tau(sim)
    sim.fsm.fg_cmd = float(abs(tau))

    target = np.array(ref["r_des"], float).reshape(3, 3).copy()
    setattr(sim.fsm, "r_des_target", target)
    setattr(sim.fsm, "r_return_omega", float(_HANDOFF_R_RETURN_OMEGA))
    setattr(sim.fsm, "handoff_active", True)

    qpos_after = np.array(sim.data.qpos, float)
    qvel_after = np.array(sim.data.qvel, float)
    grip = None if sim.data.ctrl.size <= 7 else float(sim.data.ctrl[7])
    return {
        "t": time_before,
        "phase": str(sim.fsm.phase),
        "qpos_max_abs": float(np.max(np.abs(qpos_after - qpos_before))),
        "qvel_max_abs": float(np.max(np.abs(qvel_after - qvel_before))),
        "time_delta": float(sim.data.time) - time_before,
        "r_des_to_saved_rad": float(rot_angle(ref["r_des"], np.array(sim.fsm.r_des, float))),
        "r_des_to_hand_rad": float(rot_angle(np.array(sim.fsm.r_des, float), hand_before)),
        "v_cmd_norm": float(np.linalg.norm(np.asarray(sim.fsm.v_cmd, float))),
        "w_cmd_norm": float(np.linalg.norm(np.asarray(sim.fsm.w_cmd, float))),
        "p_des": [round(float(x), 5) for x in np.asarray(sim.fsm.p_des, float).reshape(-1)[:3]],
        "grip_ctrl": grip,
        "physics_edited": False,
        "contract": "init_from_hand_smooth_return",
    }


def nominal_step(sim):
    from envs.outcomes.support_drop import object_vz, observe

    vz_before = object_vz(sim)
    z_now = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
    post_handoff = bool(getattr(sim.fsm, "handoff_active", False))

    target = getattr(sim.fsm, "r_des_target", None)
    omega_max = float(getattr(sim.fsm, "r_return_omega", _HANDOFF_R_RETURN_OMEGA) or _HANDOFF_R_RETURN_OMEGA)
    w_ret = np.zeros(3)
    if target is not None:
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        Rt = np.asarray(target, float).reshape(3, 3)
        if float(rot_angle(Rt, Rd)) < np.deg2rad(1.0):
            sim.fsm.r_des = Rt.copy()
            setattr(sim.fsm, "r_des_target", None)
        else:
            w_ret = _omega_toward_R(Rd, Rt, omega_max)

    if post_handoff:
        tau = _handoff_grip_tau(sim)
        sim.fsm.fg_cmd = float(abs(tau))
        if lift_hold:
            tick_vw(sim, np.zeros(3), w_ret, tau, TRACK)
        else:
            # Climb with grip continuity + smooth orientation return (avoid FSM open path).
            v_lift = np.array([0.0, 0.0, float(sim.fsm.cfg["fsm"].get("lift_speed", 0.08))])
            tick_vw(sim, v_lift, w_ret, tau, TRACK)
    elif lift_hold:
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, TRACK)
    else:
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
    observe(sim, vz_before)


def nominal_validate(env, ref, seconds=None):
    """Nominal control only. Recovery is not called. Clock is not reset."""
    sim = env.sim
    horizon = EXTENDED_S if seconds is None else float(seconds)
    t0 = float(sim.data.time)
    lost = 0
    drop_t = None
    reason = None
    legacy = None
    saw_t12 = float(sim.data.time) >= 12.0 - 1e-9
    while float(sim.data.time) < t0 + horizon - 1e-9:
        nominal_step(sim)
        row = errors(sim, ref)
        if legacy is None and float(sim.data.time) >= 12.0 - 1e-9:
            saw_t12 = True
            legacy = bool(row["z"] >= Z_ABORT and row["z"] >= TABLE_DROP and not _escaped(sim))
        both_off = row["nL"] == 0 and row["nR"] == 0
        lost = lost + 1 if both_off else 0
        if row["z"] < TABLE_DROP or row["z"] < Z_ABORT:
            drop_t = row["t"]
            reason = "drop"
            break
        if _escaped(sim):
            drop_t = row["t"]
            reason = "escape"
            break
        if lost >= 75:
            drop_t = row["t"]
            reason = "bilateral_lost"
            break
    t_end = float(sim.data.time)
    if legacy is None and saw_t12:
        pack = physical_pack(sim)
        legacy = float(pack["obj_z"]) >= Z_ABORT
    return {
        "t_handoff": t0,
        "t_end": t_end,
        "drop_t": drop_t,
        "reason": reason,
        "held_10": bool(drop_t is None and t_end >= t0 + PRIMARY_S - 1e-6),
        "held_12": bool(drop_t is None and t_end >= t0 + EXTENDED_S - 1e-6),
        "legacy_retained_to_12": legacy,
        "recovery_called_during_hold": False,
        "end": errors(sim, ref),
    }


def label_of(fell, basin, handed, hold):
    if fell and not handed:
        return "DROP_DURING_RECOVERY"
    if handed and hold is not None and hold["drop_t"] is not None:
        delay = float(hold["drop_t"]) - float(hold["t_handoff"])
        if delay >= 5.0:
            return "DELAYED_DROP_AFTER_HANDOFF"
        return "DROP_AFTER_HANDOFF"
    if basin and not handed:
        return "NO_HANDOFF"
    if not basin:
        return "BASIN_NOT_REACHED"
    if hold is None:
        return "NO_HANDOFF"
    if hold["held_12"]:
        return "SUCCESS_12S"
    if hold["held_10"]:
        return "SUCCESS_10S"
    return "NOMINAL_HOLD_INCOMPLETE"


def is_success(label):
    return label in ("SUCCESS_10S", "SUCCESS_12S")
