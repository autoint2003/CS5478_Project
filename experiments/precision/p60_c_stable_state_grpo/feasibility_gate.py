"""Conservative feasibility gate for stable-state continuation (P60 audit fix).

Dense reward variance is diagnostic only — never an automatic pass.
Require task-level evidence: actual HANDOFF or SUCCESS_RELAXED among probe rolls.
"""

from __future__ import annotations

from typing import Any


# Predeclared conservative defaults (override only with explicit justification).
MIN_HANDOFF_COUNT = 1  # at least one actual HANDOFF in probe set
MIN_SUCCESS_COUNT = 0  # SUCCESS optional if HANDOFF observed; both-zero → STOP
REQUIRE_HANDOFF_OR_SUCCESS = True


def evaluate_state_feasibility(
    *,
    n_success: int,
    n_handoff: int,
    n_probe: int,
    return_std: float,
    mean_epos_delta: float | None = None,
) -> dict[str, Any]:
    """Return retain/viable flags for one state's probe summary."""
    diagnostics = {
        "return_std": float(return_std),
        "mean_epos_delta": None if mean_epos_delta is None else float(mean_epos_delta),
        "dense_reward_variance_signal": bool(return_std > 1e-3),
        "pose_improvement_signal": bool(
            mean_epos_delta is not None and mean_epos_delta < -1.0
        ),
    }
    task_ok = (int(n_handoff) >= MIN_HANDOFF_COUNT) or (int(n_success) >= max(MIN_SUCCESS_COUNT, 1))
    if REQUIRE_HANDOFF_OR_SUCCESS:
        retain = bool(task_ok)
    else:
        retain = bool(task_ok or diagnostics["dense_reward_variance_signal"])
    return {
        "retain_for_train": int(retain),
        "task_level_feasible": int(task_ok),
        "n_success": int(n_success),
        "n_handoff": int(n_handoff),
        "n_probe": int(n_probe),
        "diagnostics": diagnostics,
        "gate": {
            "MIN_HANDOFF_COUNT": MIN_HANDOFF_COUNT,
            "MIN_SUCCESS_COUNT": MIN_SUCCESS_COUNT,
            "REQUIRE_HANDOFF_OR_SUCCESS": REQUIRE_HANDOFF_OR_SUCCESS,
            "note": "return_std is diagnostic only; not an automatic pass",
        },
    }


def evaluate_probe_viability(per_state: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate gate: stop if zero HANDOFF and zero SUCCESS across all states."""
    n = len(per_state)
    total_succ = sum(int(r.get("n_success") or 0) for r in per_state)
    total_hand = sum(int(r.get("n_handoff") or 0) for r in per_state)
    n_retain = sum(int(r.get("retain_for_train") or 0) for r in per_state)
    n_zero = sum(1 for r in per_state if int(r.get("n_success") or 0) == 0)
    # Hard stop: no task-level evidence anywhere
    hard_stop = total_succ == 0 and total_hand == 0
    viable = (not hard_stop) and n_retain >= max(1, min(8, n // 2)) if n else False
    stop_reason = None
    if hard_stop:
        stop_reason = (
            "Probe has zero HANDOFF and zero SUCCESS_RELAXED across all states; "
            "stable-state continuation is not a viable training target under the "
            "predeclared conservative gate. Dense reward variance is diagnostic only."
        )
    elif not viable:
        stop_reason = (
            f"Insufficient task-feasible states retained ({n_retain}/{n}); "
            "continuation training should not proceed."
        )
    return {
        "viable_for_training": bool(viable),
        "hard_stop_zero_handoff_and_success": bool(hard_stop),
        "n_states": n,
        "n_retain": n_retain,
        "n_zero_success_states": n_zero,
        "total_success_count": total_succ,
        "total_handoff_count": total_hand,
        "stop_reason": stop_reason,
    }
