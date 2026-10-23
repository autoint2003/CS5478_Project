#!/usr/bin/env python
"""One-shot mechanical split of unified_precision_oracle.py into v2.3 modules.

Preserves function bodies. Run only after baseline freeze. No controller tuning.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_generation/oracles/precision/unified_precision_oracle.py"
PKG = ROOT / "data_generation/oracles/precision"
BACKUP = (
    ROOT
    / "results/diagnostics/unified_precision_oracle_v23/baseline_v22/unified_precision_oracle_v22_source.py"
)

COMMON_IMPORTS = """
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.precision.axis_aware import (
    decide_precision,
    lateral_desire_direction,
)
from data_generation.oracles.precision.endpoint_only_axial_roll import (
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
)
from data_generation.oracles.precision.finger_surface_support import support_force_decomposition
from data_generation.oracles.precision.finger_supported_vertical import U_OPEN_HAND
from data_generation.oracles.precision.force_slip_control import (
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    contact_forces,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import (
    _height_hold_v_hand,
    execute_orientation_plan,
    plan_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.primitives import (
    _action_vw,
    _wrist,
    audit_grasp,
    pivot_segment_metrics,
    primitive_contact_pivot,
)
from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip
from data_generation.oracles.precision.sign_aware_axial import (
    fingertip_overlap_margin_mm,
    select_axial_mode,
)
from data_generation.oracles.precision.stable_grasp import (
    STABLE_THRESHOLDS,
    hold_and_confirm_stable,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import (
    AXIS_V1_EX_MM,
    AXIS_V1_EY_MM,
    AXIS_V1_EZ_MM,
    AXIS_V1_THETA_DEG,
    evaluate_handoff_with_axis,
)
from evaluation.grasp_geometry import hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG
"""


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def sl(ls: list[str], start: int, end: int) -> str:
    """1-indexed inclusive start, exclusive end."""
    return "".join(ls[start - 1 : end - 1])


def main() -> None:
    src_path = BACKUP if BACKUP.exists() else SRC
    text = src_path.read_text(encoding="utf-8")
    ls = lines(text)
    print(f"Splitting from {src_path} ({len(ls)} lines)")

    # Backup live file if not already using frozen source as input
    live_bak = PKG / "unified_precision_oracle.py.bak_pre_v23"
    if not live_bak.exists():
        live_bak.write_bytes(SRC.read_bytes())

    types_src = (
        '"""Shared Precision Oracle types, constants, and result containers (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\n"
        + sl(ls, 80, 137)
        + "\n"
        + sl(ls, 138, 158)
        + "\n"
        + sl(ls, 178, 252)
    )
    (PKG / "precision_types.py").write_text(types_src, encoding="utf-8")

    goals_src = (
        '"""Error measurement, need assessment, and goal-region stop tests (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\nfrom data_generation.oracles.precision.precision_types import (\n"
        "    PrecisionConfig,\n"
        "    STOP_GOAL_REACHED,\n"
        "    STOP_HANDOFF_READY,\n"
        "    SlipGoal,\n"
        ")\n\n"
        + sl(ls, 254, 348)
        + "\n"
        + sl(ls, 1088, 1144)
    )
    (PKG / "precision_goals.py").write_text(goals_src, encoding="utf-8")

    audit_src = (
        '"""Action/capture replay audits and failure snapshots (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\nfrom data_generation.oracles.precision.precision_goals import measure_errors\n\n"
        + sl(ls, 159, 177)
        + "\n"
        + sl(ls, 1427, 1506)
    )
    (PKG / "precision_audit.py").write_text(audit_src, encoding="utf-8")

    slip_src = (
        '"""Goal-aware DualLoop sliding, staged slip, brake/regrip helpers (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\nfrom data_generation.oracles.precision.precision_types import (\n"
        "    A0_SLIP,\n"
        "    FAIL_BRAKE,\n"
        "    FAIL_REGRIP,\n"
        "    FAIL_SLIP_CONTACT,\n"
        "    FAIL_SLIP_TARGET,\n"
        "    FINGERTIP_MARGIN_MIN_MM,\n"
        "    FN_BRAKE,\n"
        "    FN_HOLD,\n"
        "    FN_INIT,\n"
        "    FN_MAX,\n"
        "    FN_MIN,\n"
        "    FORCE_DA,\n"
        "    FORCE_KP,\n"
        "    HOLD_STABLE_S,\n"
        "    LATERAL_ABS_ABORT_MM,\n"
        "    LATERAL_DRIFT_ABORT_MM,\n"
        "    PrecisionConfig,\n"
        "    SLIP_KN,\n"
        "    STOP_BRAKE_AND_REASSESS,\n"
        "    STOP_CONTACT_SAFETY,\n"
        "    STOP_GOAL_REACHED,\n"
        "    STOP_HANDOFF_READY,\n"
        "    STOP_MAX_SAFE_TRAVEL,\n"
        "    STOP_NO_PROGRESS,\n"
        "    STOP_TIME_BUDGET,\n"
        "    STOP_WRONG_DIRECTION,\n"
        "    SlipGoal,\n"
        "    V_SLIP,\n"
        "    Z_ABORT,\n"
        ")\n"
        "from data_generation.oracles.precision.precision_goals import evaluate_slip_goal, measure_errors\n"
        "from data_generation.oracles.precision.precision_audit import _failure_snapshot\n\n"
        + sl(ls, 377, 547)
        + "\n"
        + sl(ls, 1145, 1426)
        + "\n"
        + sl(ls, 1507, 1758)
        + "\n"
        + sl(ls, 1759, 1843)
    )
    (PKG / "precision_slip.py").write_text(slip_src, encoding="utf-8")

    dx_src = (
        '"""Lateral (verified) and horizontal P0–P9 dx correction (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\nfrom data_generation.oracles.precision.precision_types import (\n"
        "    FAIL_FINAL_UNROLL,\n"
        "    FAIL_HORIZONTAL_TILT,\n"
        "    FAIL_PREPARATION,\n"
        "    FAIL_REVERSE_ROLL,\n"
        "    FAIL_SLIDING_ROLL,\n"
        "    FAIL_SLIP_CONTACT,\n"
        "    FAIL_SLIP_TARGET,\n"
        "    FAIL_SLIP_WRONG_DIR,\n"
        "    FAIL_SUPPORT_ROLL,\n"
        "    FN_HOLD,\n"
        "    G_LATERAL_MIN,\n"
        "    HORIZONTAL_SLIDE_ROLL_CANDIDATES,\n"
        "    PrecisionConfig,\n"
        "    STOP_GOAL_REACHED,\n"
        "    STOP_HANDOFF_READY,\n"
        "    SlipGoal,\n"
        "    Z_ABORT,\n"
        ")\n"
        "from data_generation.oracles.precision.precision_goals import (\n"
        "    handoff_geometry_ready,\n"
        "    measure_errors,\n"
        ")\n"
        "from data_generation.oracles.precision.precision_audit import _failure_snapshot\n"
        "from data_generation.oracles.precision.precision_slip import (\n"
        "    run_goal_aware_axial_slip,\n"
        "    run_slip_dual_loop,\n"
        "    run_staged_axial_slip,\n"
        "    settle_hold,\n"
        ")\n\n"
        + sl(ls, 617, 1087)
        + "\n"
        + sl(ls, 1844, 2245)
    )
    (PKG / "precision_dx.py").write_text(dx_src, encoding="utf-8")

    dz_src = (
        '"""Inward / outward axial dz correction (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\nfrom data_generation.oracles.precision.precision_types import (\n"
        "    FAIL_PREPARATION,\n"
        "    FAIL_SLIP_CONTACT,\n"
        "    FAIL_SLIP_WRONG_DIR,\n"
        "    FN_HOLD,\n"
        "    PrecisionConfig,\n"
        "    ROLL_ANGLE_DEG,\n"
        ")\n"
        "from data_generation.oracles.precision.precision_goals import measure_errors\n"
        "from data_generation.oracles.precision.precision_slip import (\n"
        "    _inward_slip_staged,\n"
        "    _outward_slip_staged,\n"
        "    settle_hold,\n"
        ")\n\n"
        + sl(ls, 2246, 2501)
    )
    (PKG / "precision_dz.py").write_text(dz_src, encoding="utf-8")

    handoff_helper = r'''
def evaluate_horizontal_recoverability(env, rh_capture, ref, snap, cfg: PrecisionConfig) -> dict[str, Any]:
    """Counterfactual recoverability for post-horizontal state.

    Restores ``snap`` before return. Probe actions are never returned to the caller.
    """
    restore_capture(env, rh_capture)
    err_h_live = measure_errors(env.sim)
    need_h = assess_need(err_h_live, cfg)
    recoverable = False
    recover_note = None
    detail: dict[str, Any] = {}
    if need_h.get("ready_for_direct_handoff_assessment") or handoff_geometry_ready(err_h_live, cfg):
        pr = probe_direct_handoff(env, rh_capture, ref, stabilize_n=cfg.stabilize_handoff_n)
        restore_capture(env, rh_capture)
        recoverable = bool(pr.get("SUCCESS_NEW"))
        recover_note = "handoff_probe"
        detail = {
            "mode": recover_note,
            "SUCCESS_NEW": pr.get("SUCCESS_NEW"),
            "SUCCESS_12S": pr.get("SUCCESS_12S"),
        }
    if not recoverable and need_h.get("dz_needed"):
        mode_h = need_h.get("dz_mode") or "none"
        if mode_h == "B_inward":
            rz_cf = correct_dz_inward(env, {"capture": rh_capture, "ref": ref}, cfg)
        elif mode_h == "A_outward":
            rz_cf = correct_dz_outward(env, {"capture": rh_capture, "ref": ref}, cfg)
        else:
            rz_cf = {"ok": False, "reason": "no_dz_mode", "actions": []}
        restore_capture(env, snap)
        recoverable = bool(rz_cf.get("ok"))
        recover_note = f"counterfactual_{mode_h}"
        detail = {
            "mode": recover_note,
            "ok": rz_cf.get("ok"),
            "reason": rz_cf.get("reason") or rz_cf.get("stop_reason"),
            "n_cf_actions_discarded": len(rz_cf.get("actions") or []),
        }
    elif not recoverable:
        restore_capture(env, snap)
        detail = {"mode": "no_followup_needed_but_handoff_failed", "ok": False}
    else:
        restore_capture(env, snap)
    return {"recoverable": recoverable, "recover_note": recover_note, "detail": detail}
'''

    handoff_src = (
        '"""Counterfactual HANDOFF probes and stable preparation (v2.3)."""\n'
        + COMMON_IMPORTS
        + "\nfrom data_generation.oracles.precision.precision_types import PrecisionConfig\n"
        "from data_generation.oracles.precision.precision_goals import (\n"
        "    assess_need,\n"
        "    handoff_geometry_ready,\n"
        "    measure_errors,\n"
        ")\n"
        "from data_generation.oracles.precision.precision_slip import settle_hold\n"
        "from data_generation.oracles.precision.precision_dz import (\n"
        "    correct_dz_inward,\n"
        "    correct_dz_outward,\n"
        ")\n\n"
        + sl(ls, 548, 564)
        + "\n"
        + sl(ls, 2502, 2518)
        + "\n"
        + handoff_helper
    )
    (PKG / "precision_handoff.py").write_text(handoff_src, encoding="utf-8")

    # Orchestration module: re-exports + pivot + run_precision (exact body)
    orch = '''\
"""Shared conditional Precision Oracle (Stage-2) for Plans A, B, and C.

v2.3 modular orchestration. Physical primitives live in sibling modules.
Behavior-preserving refactor of the frozen v2.2 baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.precision.force_slip_control import DT, sample_state
from data_generation.oracles.precision.primitives import (
    _action_vw,
    _wrist,
    pivot_segment_metrics,
    primitive_contact_pivot,
)
from data_generation.oracles.precision.precision_audit import (  # noqa: F401
    _failure_snapshot,
    audit_action_capture_replay,
)
from data_generation.oracles.precision.precision_dx import (  # noqa: F401
    _horizontal_open_target,
    _physical_finger_support,
    _score_horizontal_slide_config,
    classify_tilt_physical_reason,
    correct_dx_horizontal_attempt,
    correct_dx_verified,
    search_p1_p2_horizontal,
    select_horizontal_sliding_roll,
)
from data_generation.oracles.precision.precision_dz import (  # noqa: F401
    correct_dz_inward,
    correct_dz_outward,
)
from data_generation.oracles.precision.precision_goals import (  # noqa: F401
    assess_need,
    entry_gate,
    evaluate_slip_goal,
    handoff_geometry_ready,
    measure_errors,
)
from data_generation.oracles.precision.precision_handoff import (  # noqa: F401
    evaluate_horizontal_recoverability,
    prepare_stable_handoff,
    probe_direct_handoff,
)
from data_generation.oracles.precision.precision_slip import (  # noqa: F401
    _inward_slip_staged,
    _outward_slip_staged,
    run_goal_aware_axial_slip,
    run_slip_dual_loop,
    run_staged_axial_slip,
    settle_hold,
)
from data_generation.oracles.precision.precision_types import (  # noqa: F401
    A0_SLIP,
    FAIL_BRAKE,
    FAIL_FINAL_UNROLL,
    FAIL_HANDOFF,
    FAIL_HORIZONTAL_TILT,
    FAIL_PREPARATION,
    FAIL_REGRIP,
    FAIL_RETURN_TILT,
    FAIL_REVERSE_ROLL,
    FAIL_SLIDING_ROLL,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_TARGET,
    FAIL_SLIP_WRONG_DIR,
    FAIL_SUPPORT_ROLL,
    FINGERTIP_MARGIN_MIN_MM,
    FN_BRAKE,
    FN_HOLD,
    FN_INIT,
    FN_MAX,
    FN_MIN,
    FORCE_DA,
    FORCE_KP,
    G_LATERAL_MIN,
    HOLD_STABLE_S,
    HORIZONTAL_SLIDE_ROLL_CANDIDATES,
    LATERAL_ABS_ABORT_MM,
    LATERAL_DRIFT_ABORT_MM,
    PIVOT_SEARCH,
    PrecisionConfig,
    PrecisionResult,
    ROLL_ANGLE_DEG,
    ROOT,
    SLIP_KN,
    STOP_BRAKE_AND_REASSESS,
    STOP_CONTACT_SAFETY,
    STOP_GOAL_REACHED,
    STOP_HANDOFF_READY,
    STOP_MAX_SAFE_TRAVEL,
    STOP_NO_PROGRESS,
    STOP_TIME_BUDGET,
    STOP_WRONG_DIRECTION,
    SlipGoal,
    V_SLIP,
    Z_ABORT,
)
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis

'''
    orch += sl(ls, 349, 376)
    orch += "\n"
    orch += sl(ls, 565, 616)
    orch += "\n"
    orch += sl(ls, 2519, 2995)
    (PKG / "unified_precision_oracle.py").write_text(orch, encoding="utf-8")

    t = (PKG / "precision_types.py").read_text(encoding="utf-8")
    assert "FORCE_KP" in t and "class SlipGoal" in t and "class PrecisionConfig" in t
    assert "class PrecisionResult" in t
    print("Wrote modules:")
    for name in (
        "precision_types.py",
        "precision_goals.py",
        "precision_audit.py",
        "precision_slip.py",
        "precision_dx.py",
        "precision_dz.py",
        "precision_handoff.py",
        "unified_precision_oracle.py",
    ):
        p = PKG / name
        print(f"  {name}: {len(p.read_text(encoding='utf-8').splitlines())} lines")


if __name__ == "__main__":
    main()
