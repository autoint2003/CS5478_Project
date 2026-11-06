"""P60 classification + feasibility gate regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.precision.p60_c_stable_state_grpo.feasibility_gate import (  # noqa: E402
    evaluate_probe_viability,
    evaluate_state_feasibility,
)
from experiments.precision.p60_c_stable_state_grpo.state_labels import (  # noqa: E402
    classify_capture_and_future,
    capture_geometry_flags,
)


def test_capture_not_future_timeout():
    """Timeout future must not erase angular capture label."""
    m = {"e_pos_mm": 10.5, "theta": 85.6, "min_pad": 2, "bilateral": 1, "vrel": 0.1, "latched": 0}
    out = classify_capture_and_future(
        m, lost_after=False, truncated=True, handoff=False, success=False
    )
    assert out["angular_error"] is True
    assert "angular_error" in out["capture_labels"]
    assert out["primary_capture_label"] == "angular_error"
    assert "timeout_before_handoff" in out["future_labels"]
    assert out["primary_future_label"] == "timeout_before_handoff"
    print("PASS capture_not_future_timeout")


def test_multi_label_position_and_contact():
    m = {"e_pos_mm": 30.0, "theta": 3.0, "min_pad": 1, "bilateral": 1, "vrel": 0.2, "latched": 0}
    g = capture_geometry_flags(m)
    assert g["position_error"] and g["contact_quality"]
    assert "position_error" in g["capture_labels"]
    assert "contact_quality" in g["capture_labels"]
    print("PASS multi_label_position_and_contact")


def test_r3_geometry_ok():
    m = {"e_pos_mm": 5.0, "theta": 4.0, "min_pad": 4, "bilateral": 1, "vrel": 0.05, "latched": 0}
    g = capture_geometry_flags(m)
    assert g["r3_geometry_ok"] is True
    assert g["primary_capture_label"] == "r3_geometry_ok_dwell_incomplete"
    print("PASS r3_geometry_ok")


def test_feasibility_rejects_return_std_alone():
    g = evaluate_state_feasibility(
        n_success=0, n_handoff=0, n_probe=16, return_std=5.0, mean_epos_delta=-2.0
    )
    assert g["retain_for_train"] == 0
    assert g["task_level_feasible"] == 0
    assert g["diagnostics"]["dense_reward_variance_signal"] is True
    print("PASS feasibility_rejects_return_std_alone")


def test_feasibility_accepts_handoff():
    g = evaluate_state_feasibility(
        n_success=0, n_handoff=1, n_probe=16, return_std=0.0
    )
    assert g["retain_for_train"] == 1
    print("PASS feasibility_accepts_handoff")


def test_probe_hard_stop_zero_task():
    per = [
        {"n_success": 0, "n_handoff": 0, "retain_for_train": 0},
        {"n_success": 0, "n_handoff": 0, "retain_for_train": 0},
    ]
    # Even if someone wrongly retained via old logic:
    per[0]["retain_for_train"] = 1
    agg = evaluate_probe_viability(per)
    assert agg["hard_stop_zero_handoff_and_success"] is True
    assert agg["viable_for_training"] is False
    assert agg["stop_reason"]
    print("PASS probe_hard_stop_zero_task")


def test_historical_p60_would_stop():
    """Recompute gate on actual FEASIBILITY_PROBE.json if present."""
    import json

    path = ROOT / "results/experiments/p60_c_stable_state_grpo/FEASIBILITY_PROBE.json"
    if not path.is_file():
        print("SKIP historical_p60_would_stop (no probe json)")
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = []
    for r in blob["per_state"]:
        g = evaluate_state_feasibility(
            n_success=int(round(r["p_success_relaxed"] * blob["n_continuations"])),
            n_handoff=int(round(r["p_handoff"] * blob["n_continuations"])),
            n_probe=blob["n_continuations"],
            return_std=float(r["return_std"]),
        )
        rebuilt.append(
            {
                "n_success": g["n_success"],
                "n_handoff": g["n_handoff"],
                "retain_for_train": g["retain_for_train"],
            }
        )
    agg = evaluate_probe_viability(rebuilt)
    assert agg["viable_for_training"] is False
    assert agg["hard_stop_zero_handoff_and_success"] is True
    print("PASS historical_p60_would_stop", agg["stop_reason"][:80])


def main():
    test_capture_not_future_timeout()
    test_multi_label_position_and_contact()
    test_r3_geometry_ok()
    test_feasibility_rejects_return_std_alone()
    test_feasibility_accepts_handoff()
    test_probe_hard_stop_zero_task()
    test_historical_p60_would_stop()
    print("ALL P60 LABEL/GATE TESTS PASSED")


if __name__ == "__main__":
    main()
