"""Outcome-aware GRPO return. The original ``physical_reward`` is unchanged.

SUCCESS_12S:
    R = 2.0 - 0.05 * recovery_seconds
DROP_DURING_RECOVERY:
    R = -1.0 + 0.10 * q_recovery
Physical failure after handoff:
    R = -0.5 + 0.20 * q_nominal - 0.05 * recovery_seconds

``q_nominal`` is the fraction of the 12 s nominal window spent in bilateral
contact after the nominal lift command has reached its target. Time before
that event, time without both finger contacts, and time after the first
terminal grasp failure are not counted. A recovery-stage drop has no nominal
hold, so its ``q_nominal`` is zero.
"""

from __future__ import annotations

import numpy as np

SUCCESS_BONUS = 2.0
TIME_PENALTY = 0.05
DROP_BASE = -1.0
DROP_PROGRESS = 0.10
HANDOFF_FAIL_BASE = -0.5
NOMINAL_PROGRESS = 0.20

AFTER_HANDOFF_FAILURES = (
    "DROP_AFTER_HANDOFF",
    "LOST_GRASP",
    "NOMINAL_LIFT_NOT_COMPLETED",
    "PHYSICAL_VERIFICATION_FAILED",
)


def clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


class NominalHoldTrace:
    """Integrate genuine post-lift grasp time from one nominal physics step."""

    def __init__(self, nominal_s):
        self.nominal_s = float(nominal_s)
        self.held_s = 0.0
        self.lift_completed = False
        self.t_lift = None
        self.t_grasp_failure = None
        self.clock = 0.0

    def observe(self, dt, bilateral, dropped, lift_command_reached):
        """Record one controller step. ``dropped`` already includes escape."""
        step = max(0.0, float(dt))
        self.clock += step
        valid_lift = bool(bilateral) and bool(lift_command_reached) and not bool(dropped)
        if valid_lift and self.t_lift is None:
            self.t_lift = self.clock
        if valid_lift:
            self.lift_completed = True
        if self.t_grasp_failure is None and self.lift_completed and bool(bilateral) and not bool(dropped):
            self.held_s += step
        if bool(dropped) and self.t_grasp_failure is None:
            self.t_grasp_failure = self.clock

    def mark_lost_grasp(self):
        if self.t_grasp_failure is None:
            self.t_grasp_failure = self.clock

    def q_nominal(self):
        if not self.lift_completed:
            return 0.0
        return clip01(self.held_s / self.nominal_s)


def reward_components(reason, recovery_elapsed_s, recovery_limit_s, q_nominal=0.0, lift_completed=False):
    """Return the V2 breakdown. Invalid rollouts are not scored."""
    if reason == "INVALID":
        return None
    elapsed = max(0.0, float(recovery_elapsed_s))
    limit = float(recovery_limit_s)
    if limit <= 0.0:
        raise RuntimeError("recovery limit must be positive")
    q_rec = clip01(elapsed / limit)
    q_nom = 0.0 if not bool(lift_completed) else clip01(q_nominal)
    if reason == "SUCCESS_12S":
        terminal = SUCCESS_BONUS
        time_penalty = TIME_PENALTY * elapsed
        recovery_progress = 0.0
        nominal_progress = 0.0
        q_nom_used = q_nom
    elif reason == "DROP_DURING_RECOVERY":
        terminal = DROP_BASE
        time_penalty = 0.0
        recovery_progress = DROP_PROGRESS * q_rec
        nominal_progress = 0.0
        q_nom_used = 0.0
    elif reason in AFTER_HANDOFF_FAILURES:
        terminal = HANDOFF_FAIL_BASE
        time_penalty = TIME_PENALTY * elapsed
        recovery_progress = 0.0
        nominal_progress = NOMINAL_PROGRESS * q_nom
        q_nom_used = q_nom
    else:
        raise RuntimeError(f"no reward v2 for outcome {reason}")
    total = float(terminal + recovery_progress + nominal_progress - time_penalty)
    return {
        "terminal": terminal,
        "time_penalty": time_penalty,
        "recovery_progress": recovery_progress,
        "nominal_progress": nominal_progress,
        "q_recovery": q_rec if reason == "DROP_DURING_RECOVERY" else None,
        "q_nominal": q_nom_used,
        "recovery_elapsed": elapsed,
        "recovery_limit_s": limit,
        "return": total,
    }


def grpo_return(outcome, recovery_limit_s, version, historical_reward):
    """Select the rollout return. Version v1 keeps the historical scalar."""
    reason = outcome.get("reason")
    if reason == "INVALID":
        return {
            "version": version,
            "reason": reason,
            "return": None,
            "q_recovery": None,
            "q_nominal": None,
            "recovery_elapsed": outcome.get("recovery_elapsed"),
            "components": None,
            "historical_reward": None,
        }
    components = reward_components(
        reason,
        outcome.get("recovery_elapsed") or 0.0,
        recovery_limit_s,
        q_nominal=outcome.get("q_nominal") or 0.0,
        lift_completed=bool(outcome.get("lift_completed")),
    )
    selected = "v2" if str(version) == "v2" else "v1"
    if selected not in ("v1", "v2"):
        raise RuntimeError(f"unknown reward version {version}")
    used = components["return"] if selected == "v2" else float(historical_reward)
    return {
        "version": selected,
        "reason": reason,
        "handoff": outcome.get("handoff"),
        "return": used,
        "q_recovery": components["q_recovery"],
        "q_nominal": components["q_nominal"],
        "recovery_elapsed": components["recovery_elapsed"],
        "components": components,
        "historical_reward": None if historical_reward is None else float(historical_reward),
        "reward_v2": components["return"],
    }


def ordering_bounds(recovery_limit_s):
    """Worst success, best after-handoff failure, and best recovery drop."""
    limit = float(recovery_limit_s)
    success_worst = reward_components("SUCCESS_12S", limit, limit, q_nominal=1.0, lift_completed=True)["return"]
    handoff_best = reward_components("DROP_AFTER_HANDOFF", 0.0, limit, q_nominal=1.0, lift_completed=True)["return"]
    handoff_worst = reward_components("LOST_GRASP", limit, limit, q_nominal=0.0, lift_completed=False)["return"]
    drop_best = reward_components("DROP_DURING_RECOVERY", limit, limit)["return"]
    drop_worst = reward_components("DROP_DURING_RECOVERY", 0.0, limit)["return"]
    return {
        "success_worst": success_worst,
        "handoff_best": handoff_best,
        "handoff_worst": handoff_worst,
        "drop_best": drop_best,
        "drop_worst": drop_worst,
    }


def assert_reward_order(recovery_limit_s=4.0):
    bounds = ordering_bounds(recovery_limit_s)
    if not (bounds["success_worst"] > bounds["handoff_best"] >= bounds["handoff_worst"] > bounds["drop_best"] >= bounds["drop_worst"]):
        raise RuntimeError(f"reward v2 order failed {bounds}")
    return bounds
