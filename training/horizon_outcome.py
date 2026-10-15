"""Independent recovery limit and 12-second nominal verification.

The 7-D recovery action has no handoff channel. The existing persistent
bilateral-basin check remains an optional early exit. It is not a success
requirement. At the recovery limit, nominal handoff is forced unless the
object has already dropped or escaped.

Nominal time is ``sim.data.time - handoff_sim_time``. The simulation clock
is not reset. ``label_of`` in ``recovery_success`` is the historical basin
label and is not used here.
"""

from __future__ import annotations

import numpy as np

from training.autonomous_episode import Z_ABORT
from training.demo_ballistic_impact import z_tgt_of
from training.nominal_pose_audit import errors
from training.privileged_ball_impact_search import step_legal
from training.recovery_runtime import _escaped
from training.table_drop import latched, terminal_reason
from training.recovery_success import (
    CONTROL_PERIOD_S,
    PERSIST_STEPS,
    apply_handoff,
    basin_ok,
    nominal_step,
)
from envs.physical_recovery import TABLE_DROP
from training.reward_v2 import NominalHoldTrace

T_RECOVERY_MAX = 120 * CONTROL_PERIOD_S
NOMINAL_S = 12.0
# One success still beats any failure inside the 2.4 s recovery limit:
# 1 - lambda * 2.4 > 0. The penalty is 0.12 at the limit, below the
# success bonus, and it does not charge the 12 s nominal window.
LAMBDA_TIME = 0.05
LOST_STEPS = 75
REASONS = (
    "SUCCESS_12S",
    "DROP_DURING_RECOVERY",
    "DROP_AFTER_HANDOFF",
    "LOST_GRASP",
    "NOMINAL_LIFT_NOT_COMPLETED",
    "PHYSICAL_VERIFICATION_FAILED",
    "TABLE_DROP",
    "INVALID",
)


def recovery_failed(sim, ref, lost_low):
    """Same fall rule the legal search used. A brief dip below 0.44 m is not a drop."""
    row = _row(sim, ref)
    contacts = int(row["nL"]) + int(row["nR"])
    if contacts == 0 and float(row["z"]) < Z_ABORT:
        lost_low += 1
    else:
        lost_low = 0
    failed = bool(latched(sim) or _escaped(sim) or float(row["z"]) < 0.30 or lost_low >= 50)
    return failed, lost_low


def recovery_steps(limit_s=T_RECOVERY_MAX):
    return int(round(float(limit_s) / CONTROL_PERIOD_S))


def physical_reward(reason, recovery_elapsed_s):
    terminal = 1.0 if reason == "SUCCESS_12S" else 0.0
    elapsed = max(0.0, float(recovery_elapsed_s))
    return float(terminal - LAMBDA_TIME * elapsed)


def _contacts(sim):
    model, data, ids = sim.model, sim.data, sim.ids
    obj = int(ids.object_body)
    left = int(ids.left_body)
    right = int(ids.right_body)
    n_l = n_r = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        body_1 = int(model.geom_bodyid[int(contact.geom1)])
        body_2 = int(model.geom_bodyid[int(contact.geom2)])
        if obj not in (body_1, body_2):
            continue
        other = body_2 if body_1 == obj else body_1
        if other == left:
            n_l += 1
        elif other == right:
            n_r += 1
    return n_l, n_r, float(data.xpos[ids.object_body][2])


def _row(sim, ref):
    return errors(sim, ref)


def _monitor(sim):
    n_l, n_r, z = _contacts(sim)
    return {"nL": n_l, "nR": n_r, "z": z}


def _dropped(row):
    return float(row["z"]) < TABLE_DROP or float(row["z"]) < Z_ABORT


def _bilateral(row):
    return int(row["nL"]) > 0 and int(row["nR"]) > 0


def nominal_verify(env, ref, seconds=NOMINAL_S):
    """Run the unchanged nominal lift from the current state for ``seconds``."""
    sim = env.sim
    handoff = apply_handoff(sim, ref)
    t0 = float(sim.data.time)
    if abs(float(handoff["time_delta"])) > 1e-9 or abs(float(handoff["qpos_max_abs"])) > 1e-12:
        raise RuntimeError("handoff edited the physical state")
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    hold_trace = NominalHoldTrace(seconds)
    if latched(sim):
        reason = "TABLE_DROP"
    while reason is None and float(sim.data.time) - t0 < float(seconds) - 1e-9:
        t_prev = float(sim.data.time)
        nominal_step(sim)
        row = _monitor(sim)
        end_row = row
        target = float(z_tgt_of(sim))
        bilateral = _bilateral(row)
        command_reached = float(sim.fsm.p_des[2]) >= target - 1e-9
        dropped = _dropped(row) or _escaped(sim)
        hold_trace.observe(float(sim.data.time) - t_prev, bilateral, dropped, command_reached)
        if bilateral and command_reached:
            lift_done = True
        if latched(sim):
            reason = "TABLE_DROP"
            break
        if dropped:
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            hold_trace.mark_lost_grasp()
            break
    t_end = float(sim.data.time)
    nominal_elapsed = t_end - t0
    if reason is None:
        if nominal_elapsed < float(seconds) - 1e-6:
            reason = "PHYSICAL_VERIFICATION_FAILED"
        elif end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    if reason == "LOST_GRASP":
        hold_trace.mark_lost_grasp()
    end_full = errors(sim, ref)
    return {
        "reason": reason,
        "success": reason == "SUCCESS_12S",
        "t_handoff": t0,
        "t_end": t_end,
        "nominal_elapsed": nominal_elapsed,
        "lift_completed": bool(lift_done),
        "held_after_lift_s": float(hold_trace.held_s if lift_done else 0.0),
        "q_nominal": float(hold_trace.q_nominal() if lift_done else 0.0),
        "t_lift": hold_trace.t_lift,
        "t_grasp_failure": hold_trace.t_grasp_failure,
        "handoff_state": {
            "qpos_edited": float(handoff["qpos_max_abs"]),
            "p_des": handoff["p_des"],
            "grip_ctrl": handoff["grip_ctrl"],
        },
        "end": {
            "ep": round(float(end_full["ep"]), 5),
            "eR": round(float(end_full["eR"]), 5),
            "vrel": round(float(end_full["vrel"]), 5),
            "nL": int(end_full["nL"]),
            "nR": int(end_full["nR"]),
            "z": round(float(end_full["z"]), 4),
        },
    }


def _fail_recovery(sim, ref, t0):
    row = _row(sim, ref)
    return {
        "reason": terminal_reason(sim),
        "success": False,
        "handoff": None,
        "recovery_elapsed": float(sim.data.time) - t0,
        "t_recovery_start": t0,
        "t_handoff": None,
        "nominal_elapsed": 0.0,
        "lift_completed": False,
        "held_after_lift_s": 0.0,
        "q_nominal": 0.0,
        "t_lift": None,
        "t_grasp_failure": None,
        "end": {
            "ep": round(float(row["ep"]), 5),
            "eR": round(float(row["eR"]), 5),
            "vrel": round(float(row["vrel"]), 5),
            "nL": int(row["nL"]),
            "nR": int(row["nR"]),
            "z": round(float(row["z"]), 4),
        },
        "reward": physical_reward(terminal_reason(sim), float(sim.data.time) - t0),
    }


def finish_recovery(env, ref, t0, how):
    """Handoff, then the independent nominal window. ``how`` is early or forced."""
    try:
        hold = nominal_verify(env, ref)
    except Exception as exc:
        return {
            "reason": "INVALID",
            "success": False,
            "handoff": how,
            "error": str(exc),
            "recovery_elapsed": float(env.sim.data.time) - t0,
            "t_recovery_start": t0,
            "reward": None,
        }
    elapsed = float(hold["t_handoff"]) - t0
    hold.update({
        "handoff": how,
        "recovery_elapsed": elapsed,
        "t_recovery_start": t0,
        "reward": physical_reward(hold["reason"], elapsed),
    })
    return hold


def replay_commands(env, blob, actions, limit_s=None, early_basin=False):
    """Replay a saved recovery sequence, then verify nominally.

    Historical sequences are followed to their recorded end (``early_basin``
    false) so a multi-stage episode is not cut at an intermediate contact.
    ``limit_s`` forces handoff when the recovery clock reaches it first.
    """
    from training.history_aware_critic_ablation import restore_capture

    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    basin_run = 0
    lost_low = 0
    how = None
    executed = 0
    limit = None if limit_s is None else float(limit_s)
    for action in np.asarray(actions, np.float32):
        if limit is not None and float(env.sim.data.time) - t0 >= limit - 1e-9:
            how = "forced"
            break
        executed_action = step_legal(env.sim, action)
        executed += 1
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            out = _fail_recovery(env.sim, blob["ref"], t0)
            out["recovery_steps"] = executed
            return out
        if early_basin:
            inside = basin_ok(_row(env.sim, blob["ref"]), float(executed_action[6]) > 0.5)
            basin_run = basin_run + 1 if inside else 0
            if basin_run >= PERSIST_STEPS:
                how = "early"
                break
    if how is None:
        how = "forced" if limit is not None and float(env.sim.data.time) - t0 >= limit - 1e-9 else "recorded_end"
    out = finish_recovery(env, blob["ref"], t0, how)
    out["recovery_steps"] = executed
    return out


def roll_policy(env, model, book, blob, device, cap_steps=None, limit_s=None):
    """Autonomous recovery for at most the recovery limit, then nominal verification."""
    from training.history_aware_critic_ablation import restore_capture
    from training.unified_k47 import CAP, decode_action
    import torch

    limit = T_RECOVERY_MAX if limit_s is None else float(limit_s)
    cap = recovery_steps(limit) if cap_steps is None else int(cap_steps)
    handoff_index = getattr(model, "handoff_index", None)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    basin_run = 0
    lost_low = 0
    how = None
    steps = 0
    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= limit - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        if handoff_index is not None and token == int(handoff_index):
            how = "early"
            break
        if int(token) >= len(book):
            raise RuntimeError(f"token {token} is outside the motion codebook")
        _residual, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        steps += 1
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            out = _fail_recovery(env.sim, blob["ref"], t0)
            out["recovery_steps"] = steps
            return out
        inside = basin_ok(_row(env.sim, blob["ref"]), float(executed[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if handoff_index is None and basin_run >= PERSIST_STEPS:
            how = "early"
            break
    if how is None:
        how = "forced"
    out = finish_recovery(env, blob["ref"], t0, how)
    out["recovery_steps"] = steps
    return out
