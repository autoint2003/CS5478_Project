"""Counterfactual HANDOFF probes and stable preparation (v2.3)."""

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

from data_generation.oracles.precision.precision_types import (
    FAIL_POSTURE_DOWN_RESTORE,
    FAIL_POSTURE_DOWN_VERIFY,
    FAIL_STABLE_GRASP_PRE_HANDOFF,
    PrecisionConfig,
)
from data_generation.oracles.precision.precision_goals import (
    assess_need,
    handoff_geometry_ready,
    measure_errors,
)
from data_generation.oracles.precision.precision_slip import settle_hold
from data_generation.oracles.precision.precision_dz import (
    correct_dz_inward,
    correct_dz_outward,
)


def posture_is_fingertips_down(sim, *, alpha_near_deg: float = 160.0) -> bool:
    """True when fingertips are vertically DOWN or accepted near-DOWN."""
    g = gates(sim)
    return bool(g["fingertips_down"] or float(g["alpha_up_deg"]) >= float(alpha_near_deg))


def run_final_down_handoff_gate(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    actions: list,
    *,
    rate_scale: float | None = None,
) -> dict[str, Any]:
    """Mandatory pre-HANDOFF gate: restore DOWN → verify → stable grasp → reassess.

    No final HANDOFF evaluation may proceed without this gate succeeding.
    Does not itself run the frozen nominal continuation.
    """
    from data_generation.oracles.precision.precision_dx_support_transfer import (
        restore_fingertips_down,
    )

    restore_capture(env, blob["capture"])
    g0 = gates(env.sim)
    alpha0 = float(g0["alpha_up_deg"])
    already = posture_is_fingertips_down(env.sim)
    restore_log: dict[str, Any] | None = None
    cur = {"capture": blob["capture"], "ref": blob["ref"]}

    if not already:
        rd = restore_fingertips_down(
            env,
            cur,
            cfg,
            rate_scale=float(
                rate_scale
                if rate_scale is not None
                else getattr(cfg, "rate_restore_down", 0.18) or 0.18
            ),
        )
        restore_log = {k: v for k, v in rd.items() if k not in ("actions", "capture")}
        actions.extend(rd.get("actions") or [])
        if not rd.get("ok"):
            return {
                "ok": False,
                "failure_code": FAIL_POSTURE_DOWN_RESTORE,
                "reason": rd.get("reason") or "restore_DOWN_failed",
                "phase": "FINAL_DOWN_RESTORE",
                "alpha_up_deg_entry": alpha0,
                "restore_down": restore_log,
                "actions_appended": True,
                "capture": rd.get("capture") or cur["capture"],
                "errors": measure_errors(env.sim),
            }
        cur = {"capture": rd["capture"], "ref": blob["ref"]}
        restore_capture(env, cur["capture"])

    if not posture_is_fingertips_down(env.sim):
        g1 = gates(env.sim)
        return {
            "ok": False,
            "failure_code": FAIL_POSTURE_DOWN_VERIFY,
            "reason": "fingertips_not_DOWN_after_restore",
            "phase": "FINAL_DOWN_VERIFY",
            "alpha_up_deg_entry": alpha0,
            "alpha_up_deg": float(g1["alpha_up_deg"]),
            "restore_down": restore_log,
            "skipped_restore": already,
            "capture": take_capture(env),
            "errors": measure_errors(env.sim),
        }

    # Re-establish stable grasp / contact and reassess residuals / v_rel
    prep = prepare_stable_handoff(env, cur, actions)
    restore_capture(env, prep["capture"])
    err = prep.get("errors") or measure_errors(env.sim)
    g_exit = gates(env.sim)
    if not prep.get("ok"):
        return {
            "ok": False,
            "failure_code": FAIL_STABLE_GRASP_PRE_HANDOFF,
            "reason": "stable_grasp_pre_handoff_failed",
            "phase": "FINAL_STABLE_GRASP",
            "alpha_up_deg_entry": alpha0,
            "alpha_up_deg": float(g_exit["alpha_up_deg"]),
            "fingertips_down": True,
            "restore_down": restore_log,
            "skipped_restore": already,
            "handoff_prep": {k: v for k, v in prep.items() if k != "capture"},
            "capture": prep["capture"],
            "errors": err,
        }

    return {
        "ok": True,
        "failure_code": None,
        "reason": None,
        "phase": "FINAL_DOWN_GATE_OK",
        "alpha_up_deg_entry": alpha0,
        "alpha_up_deg": float(g_exit["alpha_up_deg"]),
        "fingertips_down": True,
        "skipped_restore": already,
        "restore_down": restore_log,
        "handoff_prep": {k: v for k, v in prep.items() if k != "capture"},
        "capture": prep["capture"],
        "errors": err,
        "need_reassess": assess_need(err, cfg),
    }

def probe_direct_handoff(env, capture, ref, *, stabilize_n: int = 8) -> dict[str, Any]:
    """Counterfactual HANDOFF probe. Always restores ``capture`` before return.

    Does not contribute actions to the caller tape. Callers may still restore
    defensively; restoration here makes the probe safe to invoke in isolation.
    """
    restore_capture(env, capture)
    for _ in range(stabilize_n):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    pre = measure_errors(env.sim)
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    out = {
        "pre": pre,
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
        "reason": h.get("std_reason") or h.get("reason"),
        "handoff": {k: v for k, v in h.items() if k not in ("series", "samples", "trace")},
    }
    restore_capture(env, capture)
    return out


def prepare_stable_handoff(env, blob, actions: list) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    settle_hold(env, actions, hold_s=0.5)
    cap_pre = take_capture(env)
    err = measure_errors(env.sim)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    # Diagnostic hold must not define the live handoff state
    restore_capture(env, cap_pre)
    return {
        "ok": bool(stab["stable"] and err["bilateral"] and err["v_rel_norm"] < 0.08),
        "stability": {k: stab[k] for k in ("stable", "reasons", "consecutive_bil_s_at_end")},
        "errors": err,
        "capture": cap_pre,
        "ref_r_des_to_hand_note": "IC orientation not required; HANDOFF sets r_des←ref['r_des']",
    }



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
