"""Reward v2 cases, advantage groups, and a read-only comparison of saved traces."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, physical_reward
from training.grpo_math import advantages
from training.reward_v2 import (
    NominalHoldTrace,
    assert_reward_order,
    grpo_return,
    reward_components,
)
from training.k47_codebook import group_advantage

LIMIT = 4.0
PHYSICS_DT = 0.002


def _trace(samples):
    meter = NominalHoldTrace(NOMINAL_S)
    lost = 0
    for sample in samples:
        meter.observe(sample["dt"], sample["bilateral"], sample["dropped"], sample["lift_command_reached"])
        if sample["dropped"]:
            break
        lost = lost + 1 if not sample["bilateral"] else 0
        if lost >= LOST_STEPS:
            meter.mark_lost_grasp()
            break
    return meter


def _hold(n_steps, bilateral, dropped=False, lift=False):
    return [
        {"dt": PHYSICS_DT, "bilateral": bilateral, "dropped": dropped, "lift_command_reached": lift}
        for _ in range(n_steps)
    ]


def test_cases():
    early = reward_components("DROP_DURING_RECOVERY", 0.2, LIMIT)
    late = reward_components("DROP_DURING_RECOVERY", LIMIT, LIMIT)
    immediate = reward_components("DROP_AFTER_HANDOFF", 0.0, LIMIT, q_nominal=0.0, lift_completed=False)
    delayed_q = 3.0 / NOMINAL_S
    delayed = reward_components("LOST_GRASP", 1.0, LIMIT, q_nominal=delayed_q, lift_completed=True)
    early_success = reward_components("SUCCESS_12S", 0.4, LIMIT, q_nominal=0.5, lift_completed=True)
    forced_success = reward_components("SUCCESS_12S", LIMIT, LIMIT, q_nominal=1.0, lift_completed=True)
    invalid = grpo_return({"reason": "INVALID", "recovery_elapsed": 0.2, "reward": None}, LIMIT, "v2", None)
    assert abs(early["return"] - (-1.0 + 0.10 * (0.2 / LIMIT))) < 1e-12
    assert abs(late["return"] - (-0.90)) < 1e-12
    assert immediate["q_nominal"] == 0.0 and abs(immediate["return"] - (-0.5)) < 1e-12
    assert abs(delayed["return"] - (-0.5 + 0.20 * delayed_q - 0.05 * 1.0)) < 1e-12
    assert abs(early_success["return"] - (2.0 - 0.05 * 0.4)) < 1e-12
    assert early_success["nominal_progress"] == 0.0
    assert abs(forced_success["return"] - (2.0 - 0.05 * LIMIT)) < 1e-12
    assert invalid["return"] is None and invalid["components"] is None
    premature = reward_components("DROP_AFTER_HANDOFF", 0.04, LIMIT, q_nominal=0.0, lift_completed=False)
    historical = physical_reward("SUCCESS_12S", 0.4)
    kept = grpo_return(
        {"reason": "SUCCESS_12S", "recovery_elapsed": 0.4, "lift_completed": True, "q_nominal": 0.5, "reward": historical},
        LIMIT,
        "v1",
        historical,
    )
    assert kept["return"] == historical and kept["reward_v2"] == early_success["return"]
    assert premature["return"] < 0.0
    assert premature["return"] < early_success["return"]
    no_lift = reward_components("NOMINAL_LIFT_NOT_COMPLETED", 2.0, LIMIT, q_nominal=0.8, lift_completed=False)
    assert no_lift["q_nominal"] == 0.0 and no_lift["nominal_progress"] == 0.0
    return {
        "early_drop": early["return"],
        "late_drop": late["return"],
        "immediate_handoff_failure": immediate["return"],
        "delayed_nominal_failure": delayed["return"],
        "successful_early_handoff": early_success["return"],
        "successful_forced_handoff": forced_success["return"],
        "invalid": invalid["return"],
        "premature_handoff_failure": premature["return"],
    }


def test_hold_trace():
    before_lift = _trace(_hold(100, bilateral=True, lift=False))
    assert before_lift.q_nominal() == 0.0 and not before_lift.lift_completed
    held_steps = 1000
    lost = _trace(_hold(held_steps, bilateral=True, lift=True) + _hold(LOST_STEPS, bilateral=False, lift=True))
    assert abs(lost.held_s - held_steps * PHYSICS_DT) < 1e-9
    assert abs(lost.q_nominal() - (held_steps * PHYSICS_DT) / NOMINAL_S) < 1e-9
    assert lost.t_grasp_failure is not None
    gap = _trace(
        _hold(500, bilateral=True, lift=True)
        + _hold(10, bilateral=False, lift=True)
        + _hold(500, bilateral=True, lift=True)
    )
    assert abs(gap.held_s - 1000 * PHYSICS_DT) < 1e-9
    assert gap.t_grasp_failure is None
    dropped = _trace(_hold(1, bilateral=True, dropped=True, lift=True))
    assert dropped.q_nominal() == 0.0
    return {"lost_q": lost.q_nominal(), "gap_held_s": gap.held_s, "dropped_on_lift_q": dropped.q_nominal()}


def test_advantages():
    early = reward_components("DROP_DURING_RECOVERY", 0.2, LIMIT)["return"]
    late = reward_components("DROP_DURING_RECOVERY", LIMIT, LIMIT)["return"]
    immediate = reward_components("DROP_AFTER_HANDOFF", 0.0, LIMIT, lift_completed=False)["return"]
    delayed = reward_components("LOST_GRASP", 1.0, LIMIT, q_nominal=3.0 / NOMINAL_S, lift_completed=True)["return"]
    success = reward_components("SUCCESS_12S", 0.4, LIMIT, lift_completed=True)["return"]
    mixed, skipped = group_advantage([early, late, immediate, delayed], [False, False, False, False])
    if skipped or abs(float(np.sum(mixed))) > 1e-8:
        raise RuntimeError("all-failure group did not produce advantages")
    if mixed[1] <= mixed[0] or mixed[2] <= mixed[1]:
        raise RuntimeError("later physical progress did not outrank an earlier drop")
    identical, skipped_same = group_advantage([late, late, late, late], [False, False, False, False])
    if not skipped_same or np.any(identical):
        raise RuntimeError("identical failure returns should skip the surrogate")
    with_invalid, skipped_invalid = group_advantage([success, early, None, late], [False, False, True, False])
    if skipped_invalid or with_invalid[2] != 0.0 or with_invalid[0] <= 0.0:
        raise RuntimeError("invalid rollout was scored")
    _adv, _std, skipped_pair = advantages([success, early])
    if skipped_pair or success <= immediate:
        raise RuntimeError("success did not outrank failure")
    return {"all_fail_skipped": False, "identical_fail_skipped": True, "invalid_excluded": True}


def compare_saved_traces():
    """Score stored outcome rows. Do not write back into those files."""
    baseline_path = ROOT / "results/archive/demo_results/diagnostics/raw/horizon_migration/checkpoint_eval.json"
    pilot_dir = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_pilot"
    rows = json.loads(baseline_path.read_text(encoding="utf-8"))
    scored = []
    for row in rows:
        reason = row["reason"]
        elapsed = float(row["recovery_elapsed"])
        historical = physical_reward(reason, elapsed)
        if reason == "SUCCESS_12S" or reason == "DROP_DURING_RECOVERY":
            components = reward_components(
                reason, elapsed, LIMIT, q_nominal=0.0, lift_completed=bool(row.get("lift_completed"))
            )
            v2 = components["return"]
            v2_low = v2_high = v2
            q_known = True
        elif reason == "INVALID":
            v2 = v2_low = v2_high = None
            q_known = False
        else:
            low = reward_components(reason, elapsed, LIMIT, q_nominal=0.0, lift_completed=False)["return"]
            high = reward_components(reason, elapsed, LIMIT, q_nominal=1.0, lift_completed=True)["return"]
            v2 = None
            v2_low, v2_high = low, high
            q_known = False
        scored.append({
            "id": row["id"],
            "reason": reason,
            "recovery_elapsed": elapsed,
            "lift_completed": row.get("lift_completed"),
            "nominal_elapsed": row.get("nominal_elapsed"),
            "historical_reward": historical,
            "reward_v2": v2,
            "reward_v2_low": v2_low,
            "reward_v2_high": v2_high,
            "q_nominal_known": q_known,
        })
    import torch
    pilot_files = sorted(pilot_dir.glob("checkpoint_u*.pt")) if pilot_dir.exists() else []
    pilot_keys = []
    if pilot_files:
        saved = torch.load(pilot_files[-1], map_location="cpu", weights_only=False)
        pilot_keys = sorted(saved.keys())
    summary = {
        "source": "results/archive/demo_results/diagnostics/raw/horizon_migration/checkpoint_eval.json",
        "episodes": len(scored),
        "reasons": dict(Counter(row["reason"] for row in scored)),
        "historical_reward": _stats([row["historical_reward"] for row in scored]),
        "reward_v2_exact": _stats([row["reward_v2"] for row in scored if row["reward_v2"] is not None]),
        "exact_v2_episodes": sum(1 for row in scored if row["q_nominal_known"]),
        "q_nominal_unknown": sum(1 for row in scored if not row["q_nominal_known"]),
        "pilot_checkpoint_files": [path.name for path in pilot_files],
        "pilot_checkpoint_keys": pilot_keys,
        "pilot_trajectories_saved": False,
        "missing_q_nominal_fields": sorted(
            {"held_after_lift_s", "q_nominal", "t_lift", "t_grasp_failure"} - set(rows[0])
        ),
        "saved_eval_fields": sorted(rows[0]),
    }
    import os
    import tempfile
    out = Path(os.environ["REWARD_TRACE_OUT"]) if os.environ.get("REWARD_TRACE_OUT") else Path(tempfile.mkdtemp(prefix="reward_v2_")) / "saved_trace_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": scored}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def _stats(values):
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def main():
    cases = test_cases()
    hold = test_hold_trace()
    advantages_ok = test_advantages()
    bounds = assert_reward_order(LIMIT)
    summary = compare_saved_traces()
    print(json.dumps({
        "cases": cases,
        "hold": hold,
        "advantages": advantages_ok,
        "bounds": bounds,
        "saved_traces": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
