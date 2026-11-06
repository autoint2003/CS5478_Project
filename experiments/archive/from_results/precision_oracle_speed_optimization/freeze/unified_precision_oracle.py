"""Shared conditional Precision Oracle (Stage-2) for Plans A, B, and C.

v2.3 modular orchestration. Physical primitives live in sibling modules.
Behavior-preserving refactor of the frozen v2.2 baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
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
from data_generation.oracles.precision.precision_dx_support_transfer import (  # noqa: F401
    correct_dx_support_transfer,
    restore_fingertips_down,
)
from data_generation.oracles.precision.precision_dz import (  # noqa: F401
    correct_dz_inward,
    correct_dz_outward,
)
from data_generation.oracles.precision.sign_aware_axial import select_axial_mode
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
from data_generation.oracles.precision.stable_grasp import evaluate_stable_grasp
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

def parse_ppivot_tag(tag: str) -> dict:
    rest = tag[len("ppiv_v") :]
    pre_w, after_w = rest.split("_w", 1)
    vx_s, vz_s = pre_w.split("_", 1)
    pre_g, after_g = after_w.split("_g", 1)
    wx_s, wy_s = pre_g.split("_", 1)
    grip_s, dur_s = after_g.split("_d", 1)
    return {
        k: float(v)
        for k, v in zip(("vx", "vz", "wx", "wy", "grip", "dur"), (vx_s, vz_s, wx_s, wy_s, grip_s, dur_s))
    }


def pivot_actions_from_tag(tag: str) -> list:
    p = parse_ppivot_tag(tag)
    a_p = _action_vw([p["vx"], 0.0, p["vz"]], [p["wx"], p["wy"], 0.0], p["grip"])
    return [a_p] * max(1, int(round(p["dur"] / DT))) + [_wrist(0.0, 0.0, 1.0)] * 8


def default_pivot_tag(plan_id: str) -> str | None:
    if not PIVOT_SEARCH.exists():
        return None
    search = json.loads(PIVOT_SEARCH.read_text(encoding="utf-8"))
    key = {"A": "A_raw", "B": "B_raw", "C": "A_raw"}.get(str(plan_id).upper()[:1], "A_raw")
    best = (search.get(key) or {}).get("best") or {}
    return best.get("tag")


def correct_dtheta(env, blob, cfg: PrecisionConfig, plan_id: str) -> dict[str, Any]:
    """Relative pivot → brake/regrip → mandatory fingertips-DOWN restoration."""
    from data_generation.oracles.precision.endpoint_only_axial_roll import gates
    from data_generation.oracles.precision.precision_dx_support_transfer import restore_fingertips_down

    tag = cfg.pivot_tag or default_pivot_tag(plan_id)
    actions: list = []
    pivot_ok = False
    mechanism = None
    meta: dict[str, Any] = {}

    if tag:
        acts = pivot_actions_from_tag(tag)
        met = pivot_segment_metrics(env, blob, acts, record_series=False)
        restore_capture(env, blob["capture"])
        for a in acts:
            step_legal(env.sim, np.asarray(a, np.float32))
            actions.append(np.asarray(a, np.float32).copy())
        th1 = float(measure_errors(env.sim)["dtheta_deg"])
        improved = th1 <= cfg.theta_tol_deg or (met.get("theta0", 99) - th1) >= 2.0
        settle_hold(env, actions, hold_s=0.3)
        bil_after = bool(sample_state(env.sim)["bilateral"])
        pivot_ok = bool(met.get("ok") and improved and bil_after)
        mechanism = "verified_pivot_tag"
        meta = {
            "tag": tag,
            "metrics": {k: met[k] for k in met if k not in ("series", "a0", "a1", "p0", "p1")},
            "bilateral_after_settle": bil_after,
        }
    else:
        if cfg.book is None:
            return {"ok": False, "reason": "no_pivot_tag_or_book", "actions": [], "capture": blob["capture"]}
        piv = primitive_contact_pivot(env, blob, cfg.book, budget=24)
        if not piv.get("physically_demonstrated") or not piv.get("actions"):
            return {
                "ok": False,
                "reason": "pivot_search_failed",
                "failure": piv.get("failure"),
                "actions": [],
                "capture": blob["capture"],
            }
        restore_capture(env, blob["capture"])
        for a in piv["actions"]:
            step_legal(env.sim, np.asarray(a, np.float32))
            actions.append(np.asarray(a, np.float32).copy())
        settle_hold(env, actions, hold_s=0.3)
        pivot_ok = True
        mechanism = "primitive_contact_pivot"
        meta = {"best": piv.get("best")}

    if not pivot_ok:
        return {
            "ok": False,
            "mechanism": mechanism,
            **meta,
            "reason": "pivot_not_improved",
            "actions": actions,
            "capture": take_capture(env),
            "errors": measure_errors(env.sim),
        }

    restore_log = None
    post_pivot_cap = take_capture(env)
    if bool(getattr(cfg, "restore_down_after_dtheta", True)):
        cur = {"capture": post_pivot_cap, "ref": blob["ref"]}
        rd = restore_fingertips_down(
            env, cur, cfg, rate_scale=float(getattr(cfg, "rate_restore_down", 0.16) or 0.16)
        )
        restore_log = {k: v for k, v in rd.items() if k not in ("actions", "capture")}
        if rd.get("ok"):
            actions += rd.get("actions") or []
        else:
            # Keep post-pivot state if restore would drop the object; near-DOWN skip is preferred.
            restore_capture(env, post_pivot_cap)
            g_now = gates(env.sim)
            if float(g_now["alpha_up_deg"]) >= 160.0 and sample_state(env.sim).get("bilateral"):
                restore_log["accepted_near_down"] = True
                restore_log["alpha_up_deg"] = g_now["alpha_up_deg"]
            else:
                return {
                    "ok": False,
                    "mechanism": mechanism,
                    **meta,
                    "reason": rd.get("reason") or "dtheta_restore_DOWN_failed",
                    "restore_down": restore_log,
                    "actions": actions,
                    "capture": post_pivot_cap,
                    "errors": measure_errors(env.sim),
                }

    err = measure_errors(env.sim)
    g_exit = gates(env.sim)
    return {
        "ok": True,
        "mechanism": mechanism,
        **meta,
        "restore_down": restore_log,
        "fingertips_down": bool(g_exit["fingertips_down"] or float(g_exit["alpha_up_deg"]) >= 160.0),
        "actions": actions,
        "capture": take_capture(env),
        "errors": err,
    }


def _build_task_outcomes(
    logs: dict,
    *,
    errors_exit: dict,
    handoff: dict | None,
    done_dx: bool,
    done_dz: bool,
) -> dict[str, Any]:
    """Separate primitive flip metrics from frozen HANDOFF / SUCCESS_*."""
    dz_out = dict(logs.get("dz_outcomes") or {})
    return {
        "FLIP_COMPLETED": bool(dz_out.get("FLIP_COMPLETED")),
        "DZ_SLIDE_COMPLETED": bool(dz_out.get("DZ_SLIDE_COMPLETED")),
        "GRASP_RETAINED": bool(
            errors_exit.get("bilateral") and float(errors_exit.get("obj_z") or 0.0) >= 0.44
        ),
        "RETURN_DOWN_COMPLETED": bool(dz_out.get("RETURN_DOWN_COMPLETED")),
        "HANDOFF_COMPLETED": bool(handoff is not None),
        "SUCCESS_12S": bool(handoff.get("SUCCESS_12S")) if handoff else None,
        "SUCCESS_NEW": bool(handoff.get("SUCCESS_NEW")) if handoff else None,
        "dx_executed": bool(done_dx),
        "dz_executed": bool(done_dz or bool(dz_out)),
        "dx_mechanism": logs.get("dx_mechanism"),
        "dz_s1_reused": logs.get("dz_s1_reused"),
    }


def run_precision(
    env,
    stable_grasp_state: dict,
    plan_id: str,
    precision_config: PrecisionConfig | dict | None = None,
) -> PrecisionResult:
    """Unified Stage-2 entry: stable grasp → assess → optional corrections → HANDOFF.

    `stable_grasp_state` must provide at least `{capture, ref}`. Extra provenance
    keys (prefix actions, ic_id) are ignored for control but preserved in logs.
    """
    cfg = precision_config if isinstance(precision_config, PrecisionConfig) else PrecisionConfig(**(precision_config or {}))
    blob0 = {"capture": stable_grasp_state["capture"], "ref": stable_grasp_state["ref"]}
    plan_id = str(plan_id)
    actions: list = []
    phases: list[str] = []
    logs: dict[str, Any] = {}
    failures: list[str] = []

    restore_capture(env, blob0["capture"])
    gate = entry_gate(env)
    # Restore to pre-hold capture for subsequent work
    restore_capture(env, blob0["capture"])
    if not gate["ok"]:
        return PrecisionResult(
            status="STABLE_GATE_FAILED",
            plan_id=plan_id,
            precision_skipped=True,
            precision_executed=False,
            gate_ok=False,
            phases=["entry_gate"],
            errors_entry=gate["errors"],
            failures=list(gate["stability"].get("reasons") or []),
            note="Return control to Stage-1 recovery; do not start precision from unstable grasp.",
            capture=blob0["capture"],
            ref=blob0["ref"],
            primitive_logs={"gate": gate},
        )

    # Gate hold can settle contact, force, or relative motion that the restored
    # pre-hold capture still violates. Record the same secure-grip settle into
    # the teacher tape so primitives and replay start from the state the gate proved.
    err_pre = measure_errors(env.sim)
    instant_pre = evaluate_stable_grasp(env.sim)
    min_fn = float((gate.get("thresholds") or {}).get("min_fn_n") or 1.0)
    if not instant_pre["stable_instant"]:
        phases.append("entry_grip_settle")
        settle_n = max(1, int(round(0.20 / 0.02)))
        for _ in range(settle_n):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            step_legal(env.sim, a)
            actions.append(a.copy())
        err_settle = measure_errors(env.sim)
        settle_ok = bool(
            err_settle.get("bilateral")
            and float(err_settle.get("fn_L") or 0.0) >= min_fn
            and float(err_settle.get("fn_R") or 0.0) >= min_fn
            and float(err_settle.get("obj_z") or 0.0) >= 0.20
        )
        logs["entry_grip_settle"] = {
            "n_actions": settle_n,
            "ok": settle_ok,
            "errors": err_settle,
            "pre_errors": err_pre,
            "pre_reasons": list(instant_pre.get("reasons") or []),
        }
        if not settle_ok:
            failures.append("entry_grip_settle_failed")
            return PrecisionResult(
                status="STABLE_GATE_FAILED",
                plan_id=plan_id,
                precision_skipped=False,
                precision_executed=True,
                gate_ok=False,
                phases=phases,
                errors_entry=err_pre,
                errors_exit=err_settle,
                actions=actions,
                capture=take_capture(env),
                ref=blob0["ref"],
                failures=failures,
                primitive_logs={"gate": gate, **logs},
                note="Gate hold proved transient stability, but recorded settle could not restore load-bearing bilateral grasp.",
            )
        blob0 = {"capture": take_capture(env), "ref": blob0["ref"]}

    errors_entry = measure_errors(env.sim)
    decision = assess_need(errors_entry, cfg)
    logs["gate"] = {k: gate[k] for k in gate if k != "stability"}
    logs["decision"] = decision
    cur = dict(blob0)

    direct_probe = None
    if cfg.try_direct_handoff and decision["ready_for_direct_handoff_assessment"]:
        phases.append("direct_handoff_probe")
        direct_probe = probe_direct_handoff(env, cur["capture"], cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
        restore_capture(env, cur["capture"])  # discard counterfactual
        logs["direct_handoff_probe"] = {k: v for k, v in direct_probe.items() if k != "handoff"}
        if direct_probe.get("SUCCESS_NEW") and cfg.evaluate_handoff:
            phases.append("PRECISION_SKIPPED_DIRECT_SUCCESS")
            return PrecisionResult(
                status="PRECISION_SKIPPED_DIRECT_SUCCESS",
                plan_id=plan_id,
                precision_skipped=True,
                precision_executed=False,
                gate_ok=True,
                phases=phases,
                decision=decision,
                errors_entry=errors_entry,
                errors_exit=direct_probe.get("pre") or errors_entry,
                actions=[],
                capture=cur["capture"],
                ref=cur["ref"],
                handoff=direct_probe.get("handoff"),
                direct_handoff_probe=logs["direct_handoff_probe"],
                primitive_logs=logs,
                note="Grasp already suitable; no pivot/lateral/axial executed.",
            )
        if decision["ready_for_direct_handoff_assessment"] and not direct_probe.get("SUCCESS_NEW"):
            failures.append(f"direct_handoff_failed:{direct_probe.get('reason')}")
            # Do not assume geometric correction is required; still try if residuals exist
            phases.append("direct_failed_reassess")

    # ---- Bounded closed-loop correction (reassess after each primitive) ----
    cycle_log: list[dict] = []
    last_signature: tuple | None = None
    stagnant = 0
    # Preserve explicit 0 (do not coerce via `or` — 0 means no correction cycles).
    _mc = getattr(cfg, "max_correction_cycles", 4)
    max_cycles = 4 if _mc is None else int(_mc)
    _ab = getattr(cfg, "global_action_budget", 2500)
    action_budget = 2500 if _ab is None else int(_ab)
    done_dtheta = False
    done_dx = False
    done_dz = False
    theta_after_pivot: float | None = None
    s1_state: dict | None = None  # verified support handoff dx → dz

    def _sig(err: dict) -> tuple:
        e = np.asarray(err["e_signed_mm"], float)
        return (round(float(e[0]), 1), round(float(e[2]), 1), round(float(err["dtheta_deg"]), 1), round(float(err["e_pos_mm"]), 1))

    for cycle in range(max_cycles):
        if len(actions) >= action_budget:
            failures.append(f"action_budget:{action_budget}")
            phases.append("ACTION_BUDGET")
            break
        restore_capture(env, cur["capture"])
        err_now = measure_errors(env.sim)
        need = assess_need(err_now, cfg)
        cycle_row = {"cycle": cycle, "errors": err_now, "need": {k: need[k] for k in need if k != "axis_decision"}}
        phases.append(f"reassess_c{cycle}")

        # Level-2 geometry → counterfactual HANDOFF probe (not every 20 ms)
        if need.get("ready_for_direct_handoff_assessment") or handoff_geometry_ready(err_now, cfg):
            phases.append(f"handoff_probe_c{cycle}")
            probe = probe_direct_handoff(env, cur["capture"], cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
            restore_capture(env, cur["capture"])
            cycle_row["handoff_probe"] = {
                "SUCCESS_NEW": probe.get("SUCCESS_NEW"),
                "SUCCESS_12S": probe.get("SUCCESS_12S"),
                "reason": probe.get("reason"),
            }
            if probe.get("SUCCESS_NEW") and cfg.evaluate_handoff:
                # If still in non-standard posture after dx-only, restore DOWN under S1.
                if s1_state and not err_now.get("fingertips_down"):
                    phases.append("restore_DOWN_before_handoff")
                    rd = restore_fingertips_down(env, cur, cfg, rate_scale=float(cfg.rate_restore_down))
                    actions.extend(rd.get("actions") or [])
                    if rd.get("ok") and rd.get("capture"):
                        cur = {"capture": rd["capture"], "ref": cur["ref"]}
                    logs["restore_down_pre_handoff"] = {k: v for k, v in rd.items() if k not in ("actions", "capture")}
                cycle_log.append(cycle_row)
                logs["cycles"] = cycle_log
                logs["s1_state"] = s1_state
                phases.append("PRECISION_EARLY_HANDOFF")
                err_exit_early = measure_errors(env.sim)
                handoff_early = probe.get("handoff")
                logs["task_outcomes"] = _build_task_outcomes(
                    logs,
                    errors_exit=err_exit_early,
                    handoff=handoff_early,
                    done_dx=done_dx,
                    done_dz=done_dz,
                )
                return PrecisionResult(
                    status="PRECISION_EXECUTED_SUCCESS" if actions else "PRECISION_SKIPPED_DIRECT_SUCCESS",
                    plan_id=plan_id,
                    precision_skipped=len(actions) == 0,
                    precision_executed=len(actions) > 0,
                    gate_ok=True,
                    phases=phases,
                    decision=decision,
                    errors_entry=errors_entry,
                    errors_exit=err_exit_early,
                    actions=actions,
                    capture=cur["capture"],
                    ref=cur["ref"],
                    handoff=handoff_early,
                    direct_handoff_probe=direct_probe,
                    primitive_logs=logs,
                    failures=failures,
                    note="Closed-loop: geometry ready; frozen HANDOFF probe SUCCESS_NEW.",
                )

        sig = _sig(err_now)
        if last_signature is not None and sig == last_signature:
            stagnant += 1
        else:
            stagnant = 0
        last_signature = sig
        if stagnant >= 2:
            failures.append("no_progress_oscillation")
            phases.append("NO_PROGRESS")
            cycle_log.append(cycle_row)
            break

        # Select one necessary correction from current state (dtheta → dx → dz).
        if need["dtheta_needed"] and not done_dtheta:
            phases.append("correct_dtheta")
            r = correct_dtheta(env, cur, cfg, plan_id)
            logs["dtheta"] = {k: v for k, v in r.items() if k not in ("actions", "capture")}
            actions.extend(r.get("actions") or [])
            cycle_row["primitive"] = "dtheta"
            cycle_row["ok"] = r.get("ok")
            s1_state = None  # posture restored; prior S1 invalidated
            if not r.get("ok"):
                failures.append(f"dtheta:{r.get('reason') or 'failed'}")
                cycle_log.append(cycle_row)
                logs["cycles"] = cycle_log
                return PrecisionResult(
                    status="PRECISION_PRIMITIVE_FAILED",
                    plan_id=plan_id,
                    precision_skipped=False,
                    precision_executed=True,
                    gate_ok=True,
                    phases=phases,
                    decision=decision,
                    errors_entry=errors_entry,
                    errors_exit=measure_errors(env.sim),
                    actions=actions,
                    capture=r.get("capture") or cur["capture"],
                    ref=cur["ref"],
                    direct_handoff_probe=direct_probe,
                    primitive_logs=logs,
                    failures=failures,
                )
            cur = {"capture": r["capture"], "ref": cur["ref"]}
            done_dtheta = True
            theta_after_pivot = float((r.get("errors") or measure_errors(env.sim))["dtheta_deg"])
            cycle_log.append(cycle_row)
            continue

        if need["dx_needed"] and not done_dx:
            phases.append("correct_dx")
            dx_ok = False
            err_before = err_now
            # --- Flowchart primary: support-transfer gravity-assisted dx ---
            if bool(getattr(cfg, "use_support_transfer_dx", True)):
                phases.append("dx_support_transfer")
                snap = take_capture(env)
                rst = correct_dx_support_transfer(env, cur, cfg)
                logs["dx_support_transfer"] = {k: v for k, v in rst.items() if k not in ("actions", "capture")}
                cycle_row["primitive"] = "dx_support_transfer"
                cycle_row["ok"] = rst.get("ok")
                cycle_row["timing_s"] = rst.get("timing_s")
                if rst.get("ok") and not rst.get("skipped"):
                    actions.extend(rst.get("actions") or [])
                    cur = {"capture": rst["capture"], "ref": cur["ref"]}
                    s1_state = rst.get("s1_state")
                    dx_ok = True
                    logs["dx_mechanism"] = "support_transfer_dx"
                    # DX-7: always keep S1 for the next reassess cycle. Restore DOWN
                    # only when that cycle finds no remaining dz (see restore_DOWN_exit).
                    # Inline restore here previously cleared S1 before a latent dz need
                    # reappeared after posture change.
                    logs["dx_keep_s1_for_dz"] = True
                    logs["dx_post_errors"] = rst.get("errors")
                elif rst.get("skipped"):
                    dx_ok = True  # near-zero; nothing to do
                    logs["dx_mechanism"] = "support_transfer_skipped"
                else:
                    failures.append(
                        f"support_transfer_dx:{rst.get('reason') or rst.get('failure_code')}:phase={rst.get('phase')}"
                    )
                    restore_capture(env, snap)
                    cur = {"capture": snap, "ref": cur["ref"]}

            # Legacy horizontal path (optional baseline; not silent substitute when ST succeeds)
            if not dx_ok and cfg.allow_horizontal_dx and not bool(getattr(cfg, "use_support_transfer_dx", True)):
                phases.append("dx_horizontal_attempt")
                snap = take_capture(env)
                rh = correct_dx_horizontal_attempt(env, cur, cfg)
                logs["dx_horizontal"] = {k: v for k, v in rh.items() if k not in ("actions", "capture")}
                err_h = rh.get("errors") or {}
                dx_mm_h = float(err_h.get("dx_mm", 1e9)) if rh.get("ok") else 1e9
                accept_partial = bool(getattr(cfg, "accept_partial_horizontal_dx", True))
                horizontal_accept = bool(
                    rh.get("ok")
                    and not rh.get("rejected_branch", False)
                    and not rh.get("skipped")
                    and (
                        abs(dx_mm_h) <= float(cfg.ex_tol_mm)
                        or (
                            accept_partial
                            and rh.get("partial_progress")
                            and abs(dx_mm_h) < abs(float(err_before["dx_mm"])) - 0.3
                        )
                    )
                )
                if horizontal_accept:
                    actions.extend(rh.get("actions") or [])
                    cur = {"capture": rh["capture"], "ref": cur["ref"]}
                    dx_ok = True
                    s1_state = None
                    logs["dx_mechanism"] = "horizontal_dx"
                    cycle_row["primitive"] = "dx_horizontal"
                else:
                    restore_capture(env, snap)
                    cur = {"capture": snap, "ref": cur["ref"]}

            if not dx_ok and cfg.use_verified_lateral_fallback:
                phases.append("dx_verified_lateral")
                rv = correct_dx_verified(env, cur, cfg)
                logs["dx_verified"] = {k: v for k, v in rv.items() if k not in ("actions", "capture")}
                actions.extend(rv.get("actions") or [])
                cycle_row["primitive"] = "dx_verified_lateral"
                cycle_row["ok"] = rv.get("ok")
                if not rv.get("ok"):
                    failures.append(f"verified_dx:{rv.get('reason') or rv.get('phase') or 'failed'}")
                    cycle_log.append(cycle_row)
                    logs["cycles"] = cycle_log
                    return PrecisionResult(
                        status="PRECISION_PRIMITIVE_FAILED",
                        plan_id=plan_id,
                        precision_skipped=False,
                        precision_executed=True,
                        gate_ok=True,
                        phases=phases,
                        decision=decision,
                        errors_entry=errors_entry,
                        errors_exit=measure_errors(env.sim),
                        actions=actions,
                        capture=rv.get("capture") or cur["capture"],
                        ref=cur["ref"],
                        direct_handoff_probe=direct_probe,
                        primitive_logs=logs,
                        failures=failures,
                        note="Support-transfer dx failed; verified lateral also failed.",
                    )
                cur = {"capture": rv["capture"], "ref": cur["ref"]}
                dx_ok = True
                s1_state = None
                logs["dx_mechanism"] = "verified_lateral"
            if not dx_ok:
                cycle_log.append(cycle_row)
                logs["cycles"] = cycle_log
                return PrecisionResult(
                    status="PRECISION_PRIMITIVE_FAILED",
                    plan_id=plan_id,
                    precision_skipped=False,
                    precision_executed=True,
                    gate_ok=True,
                    phases=phases,
                    decision=decision,
                    errors_entry=errors_entry,
                    errors_exit=measure_errors(env.sim),
                    actions=actions,
                    capture=cur["capture"],
                    ref=cur["ref"],
                    direct_handoff_probe=direct_probe,
                    primitive_logs=logs,
                    failures=failures,
                )
            cycle_row["ok"] = True
            done_dx = True
            cycle_log.append(cycle_row)
            continue

        if need["dz_needed"] and not done_dz:
            mode = need.get("dz_mode") or select_axial_mode(
                float(err_now["dz_mm"]), axial_tol_mm=cfg.ez_tol_mm
            )["mode"]
            dz_s1 = s1_state
            if mode == "B_inward":
                phases.append("correct_dz_inward")
                rz = correct_dz_inward(env, cur, cfg, s1_state=dz_s1)
                if not rz.get("ok") and bool(getattr(cfg, "use_goal_driven_slip", True)):
                    phases.append("correct_dz_inward_legacy_fallback")
                    cfg_leg = PrecisionConfig(**{**asdict(cfg), "use_goal_driven_slip": False, "inward_slip_mm": 7.0})
                    rz2 = correct_dz_inward(env, cur, cfg_leg, s1_state=dz_s1)
                    if rz2.get("ok"):
                        rz = rz2
                        logs["dz_legacy_fallback"] = True
            elif mode == "A_outward":
                phases.append("correct_dz_outward")
                rz = correct_dz_outward(env, cur, cfg, s1_state=dz_s1)
            else:
                rz = {"ok": True, "skipped": True, "actions": [], "capture": cur["capture"]}
                phases.append("dz_skip")
            logs["dz"] = {k: v for k, v in rz.items() if k not in ("actions", "capture")}
            logs["dz_s1_reused"] = bool(rz.get("reused_s1"))
            logs["dz_outcomes"] = rz.get("outcomes")
            logs["dz_entry_check"] = rz.get("entry_check")
            actions.extend(rz.get("actions") or [])
            cycle_row["primitive"] = f"dz:{mode}"
            cycle_row["ok"] = rz.get("ok")
            cycle_row["branch"] = rz.get("branch")
            cycle_row["stop_reason"] = (rz.get("slip") or {}).get("stop_reason") or rz.get("stop_reason")
            cycle_row["outcomes"] = rz.get("outcomes")
            if not rz.get("ok"):
                code = rz.get("failure_code") or ""
                if code == "DZ_UNSAFE_ENTRY" or (rz.get("phase") or "").endswith("precondition"):
                    failures.append(
                        f"dz_precondition:{rz.get('reason')}:hint=run_dx_before_inward_dz"
                    )
                else:
                    failures.append(f"dz:{rz.get('reason') or rz.get('phase') or 'failed'}")
                # Flip completion without grasp retention is not task success.
                if (rz.get("outcomes") or {}).get("FLIP_COMPLETED") and not (rz.get("outcomes") or {}).get("GRASP_RETAINED"):
                    failures.append("flip_completed_but_grasp_not_retained")
                cycle_log.append(cycle_row)
                logs["cycles"] = cycle_log
                return PrecisionResult(
                    status="PRECISION_PRIMITIVE_FAILED",
                    plan_id=plan_id,
                    precision_skipped=False,
                    precision_executed=True,
                    gate_ok=True,
                    phases=phases,
                    decision=decision,
                    errors_entry=errors_entry,
                    errors_exit=measure_errors(env.sim),
                    actions=actions,
                    capture=rz.get("capture") or cur["capture"],
                    ref=cur["ref"],
                    direct_handoff_probe=direct_probe,
                    primitive_logs=logs,
                    failures=failures,
                    note="Dz rejected or failed; see dz_outcomes (FLIP≠recovery).",
                )
            cur = {"capture": rz["capture"], "ref": cur["ref"]}
            s1_state = None  # dz completes with standard posture
            done_dz = True
            cycle_log.append(cycle_row)
            continue

        # Allow one more dx pass if dz coupling reintroduced lateral error
        if need["dx_needed"] and done_dx and done_dz:
            done_dx = False  # one coupling retry
            cycle_row["primitive"] = "dx_coupling_retry_armed"
            cycle_log.append(cycle_row)
            continue

        # Nothing AXIS-flagged — but post-dx S1 may still need axial for SUCCESS_NEW e_pos,
        # or a protected return to DOWN. Prefer axial under S1 before clearing support.
        if s1_state and not done_dz:
            e_pos_now = float(err_now.get("e_pos_mm") or 0.0)
            ez_now = float(err_now.get("dz_mm") or 0.0)
            if e_pos_now > float(cfg.working_pos_mm) or abs(ez_now) > 0.25 * float(cfg.ez_tol_mm):
                mode = select_axial_mode(ez_now, axial_tol_mm=min(float(cfg.ez_tol_mm), max(abs(ez_now) - 0.1, 1.0)))["mode"]
                if mode in ("B_inward", "A_outward"):
                    phases.append(f"correct_dz_{mode}_under_s1")
                    if mode == "B_inward":
                        rz = correct_dz_inward(env, cur, cfg, s1_state=s1_state)
                    else:
                        rz = correct_dz_outward(env, cur, cfg, s1_state=s1_state)
                    logs["dz"] = {k: v for k, v in rz.items() if k not in ("actions", "capture")}
                    logs["dz_s1_reused"] = bool(rz.get("reused_s1"))
                    logs["dz_outcomes"] = rz.get("outcomes")
                    logs["dz_entry_check"] = rz.get("entry_check")
                    actions.extend(rz.get("actions") or [])
                    cycle_row["primitive"] = f"dz_under_s1:{mode}"
                    cycle_row["ok"] = rz.get("ok")
                    cycle_row["branch"] = rz.get("branch")
                    cycle_row["outcomes"] = rz.get("outcomes")
                    if not rz.get("ok"):
                        failures.append(f"dz:{rz.get('reason') or rz.get('phase') or 'failed'}")
                        cycle_log.append(cycle_row)
                        logs["cycles"] = cycle_log
                        return PrecisionResult(
                            status="PRECISION_PRIMITIVE_FAILED",
                            plan_id=plan_id,
                            precision_skipped=False,
                            precision_executed=True,
                            gate_ok=True,
                            phases=phases,
                            decision=decision,
                            errors_entry=errors_entry,
                            errors_exit=measure_errors(env.sim),
                            actions=actions,
                            capture=rz.get("capture") or cur["capture"],
                            ref=cur["ref"],
                            direct_handoff_probe=direct_probe,
                            primitive_logs=logs,
                            failures=failures,
                        )
                    cur = {"capture": rz["capture"], "ref": cur["ref"]}
                    s1_state = None
                    done_dz = True
                    cycle_log.append(cycle_row)
                    continue

        # Restore standard posture if lingering non-DOWN after dx and no dz path remains
        if s1_state and not err_now.get("fingertips_down"):
            phases.append("restore_DOWN_exit")
            rd = restore_fingertips_down(env, cur, cfg, rate_scale=float(cfg.rate_restore_down))
            actions.extend(rd.get("actions") or [])
            if rd.get("ok") and rd.get("capture"):
                cur = {"capture": rd["capture"], "ref": cur["ref"]}
            logs["restore_down_exit"] = {k: v for k, v in rd.items() if k not in ("actions", "capture")}
            s1_state = None
            cycle_row["primitive"] = "restore_DOWN"
            cycle_log.append(cycle_row)
            continue

        cycle_row["primitive"] = "none"
        cycle_log.append(cycle_row)
        phases.append("NO_REMAINING_CORRECTION")
        break

    logs["cycles"] = cycle_log
    logs["s1_state"] = s1_state
    logs["theta_after_pivot"] = theta_after_pivot

    # ---- HANDOFF prep + evaluate ----
    phases.append("prepare_stable_handoff")
    prep = prepare_stable_handoff(env, cur, actions)
    logs["handoff_prep"] = {k: v for k, v in prep.items() if k != "capture"}
    cur = {"capture": prep["capture"], "ref": cur["ref"]}
    errors_exit = prep["errors"]

    handoff = None
    if cfg.evaluate_handoff:
        phases.append("handoff_nominal_12s")
        restore_capture(env, cur["capture"])
        handoff = evaluate_handoff_with_axis(env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
        restore_capture(env, cur["capture"])

    success = bool(handoff and handoff.get("SUCCESS_NEW")) if cfg.evaluate_handoff else bool(prep.get("ok"))
    if cfg.evaluate_handoff and handoff and not handoff.get("SUCCESS_NEW"):
        failures.append(
            f"handoff_NEW_fail:12S={handoff.get('SUCCESS_12S')}:reason={handoff.get('std_reason') or handoff.get('reason')}"
        )

    status = "PRECISION_EXECUTED_SUCCESS" if success else "PRECISION_EXECUTED_HANDOFF_FAIL"
    if not actions and success:
        status = "PRECISION_SKIPPED_DIRECT_SUCCESS"

    logs["task_outcomes"] = _build_task_outcomes(
        logs,
        errors_exit=errors_exit,
        handoff=handoff,
        done_dx=done_dx,
        done_dz=done_dz,
    )

    return PrecisionResult(
        status=status,
        plan_id=plan_id,
        precision_skipped=len(actions) == 0,
        precision_executed=len(actions) > 0,
        gate_ok=True,
        phases=phases,
        decision=decision,
        errors_entry=errors_entry,
        errors_exit=errors_exit,
        actions=actions,
        capture=cur["capture"],
        ref=cur["ref"],
        handoff={k: v for k, v in (handoff or {}).items() if k not in ("series", "samples", "trace")} if handoff else None,
        direct_handoff_probe=direct_probe,
        primitive_logs=logs,
        failures=failures,
        note="Unified flowchart Stage-2 (dtheta→dx→dz→HANDOFF). FLIP_COMPLETED≠SUCCESS_NEW.",
    )


# Conceptual alias matching the task brief signature (sim-style).
def run_precision_from_sim(sim_env, stable_grasp_state, plan_id, precision_config=None) -> PrecisionResult:
    return run_precision(sim_env, stable_grasp_state, plan_id, precision_config)
