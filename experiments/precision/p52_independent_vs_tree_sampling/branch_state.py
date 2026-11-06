"""Full branch restore: MuJoCo snapshot + R3F dwell timers + policy history."""

from __future__ import annotations

import copy
from typing import Any

from envs.simulation.capture import restore_snapshot, take_snapshot
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline
from experiments.precision.p43_r3_working_hold_v2.r3_rules import DwellTracker


def export_r3f(mon: R3FOnline) -> dict[str, Any]:
    t = mon.tracker
    assert t is not None
    return {
        "t_max": int(mon.t_max),
        "reason": mon.reason,
        "handoff_step": mon.handoff_step,
        "ok_s": float(t.ok_s),
        "first_ok_step": t.first_ok_step,
        "n_resets": int(t.n_resets),
        "ever_partial": bool(t.ever_partial),
        "_active": bool(t._active),
        "rule_id": t.rule_id,
        "dt": float(t.dt),
        "dwell_s": float(t.dwell_s),
        "require_bilateral": bool(t.require_bilateral),
        "min_pad": int(t.min_pad),
        "theta_max": float(t.theta_max),
        "pos_max": float(t.pos_max),
    }


def import_r3f(blob: dict[str, Any]) -> R3FOnline:
    tracker = DwellTracker(
        rule_id=blob["rule_id"],
        dt=float(blob["dt"]),
        dwell_s=float(blob["dwell_s"]),
        require_bilateral=bool(blob["require_bilateral"]),
        min_pad=int(blob["min_pad"]),
        theta_max=float(blob["theta_max"]),
        pos_max=float(blob["pos_max"]),
        ok_s=float(blob["ok_s"]),
        first_ok_step=blob["first_ok_step"],
        n_resets=int(blob["n_resets"]),
        ever_partial=bool(blob["ever_partial"]),
        _active=bool(blob["_active"]),
    )
    mon = R3FOnline(t_max=int(blob["t_max"]), tracker=tracker)
    mon.reason = blob["reason"]
    mon.handoff_step = blob["handoff_step"]
    return mon


def take_branch_state(env, mon: R3FOnline, *, step: int, entropy: float, token: int) -> dict[str, Any]:
    """Pre-action branch point: physical+controller snapshot and R3F timers."""
    return {
        "step": int(step),
        "entropy": float(entropy),
        "token": int(token),
        "snap": take_snapshot(env),
        "r3f": export_r3f(mon),
        "policy_step": int(step),  # equals recovery policy steps so far
    }


def restore_branch_state(env, state: dict[str, Any]) -> R3FOnline:
    restore_snapshot(env, state["snap"])
    return import_r3f(copy.deepcopy(state["r3f"]))
