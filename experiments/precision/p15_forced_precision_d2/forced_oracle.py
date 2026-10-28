"""Forced Precision entry for dataset generation (P15).

Routes:
  force_skill=\"dx\"      → production ``run_precision`` with ``force_precision=True``
  force_skill=\"coupled\" → experimental ``coupled_dx_dz`` then STRICT handoff gate

Does not change production defaults. Forced skill is metadata only (not policy obs).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.handoff import (
    posture_is_fingertips_down,
    probe_direct_handoff,
    run_final_down_handoff_gate,
)
from data_generation.oracles.precision.mechanics.open_axis import DOWN, run_tilt_phase
from data_generation.oracles.precision.mechanics.support_transfer import restore_fingertips_down
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import FN_HOLD, PrecisionConfig
from data_generation.oracles.precision.state_machine import run_precision
from envs.simulation.capture import restore_capture, take_capture
from experiments.precision.coupled_feasibility.coupled_dx_dz import coupled_dx_dz


def forced_precision_config(
    *,
    force_skill: str,
    force_ex_tol_mm: float = 1.0,
    force_ez_tol_mm: float = 1.0,
    force_lateral_axis_tol_mm: float = 1.0,
) -> PrecisionConfig:
    """Experimental config: production AXIS bands unchanged; force_* only."""
    return PrecisionConfig(
        use_verified_lateral_fallback=False,
        use_explicit_fsm=True,
        try_direct_handoff=False,
        force_precision=True,
        force_skill=str(force_skill).lower(),
        force_ex_tol_mm=float(force_ex_tol_mm),
        force_ez_tol_mm=float(force_ez_tol_mm),
        force_lateral_axis_tol_mm=float(force_lateral_axis_tol_mm),
        # Tighter stop so small-error DX does not treat |dx|≲3 mm as already done.
        lateral_axis_tol_mm=float(force_lateral_axis_tol_mm),
    )


def _classify_coupled(
    dx0: float,
    dz0: float,
    dx1: float,
    dz1: float,
    *,
    min_mm: float,
    aborted: bool,
    n_actions: int,
) -> str:
    """Require |residual| reduction on an axis — motion that worsens error does not count."""
    if aborted:
        return "COUPLED_ABORTED"
    improved_dx = (abs(dx0) - abs(dx1)) >= min_mm
    improved_dz = (abs(dz0) - abs(dz1)) >= min_mm
    if n_actions <= 0 or (not improved_dx and not improved_dz):
        return "NO_MEANINGFUL_ACTION"
    if improved_dx and improved_dz:
        return "COUPLED_EXECUTED_BOTH_AXES"
    if improved_dx:
        return "COUPLED_EXECUTED_DX_ONLY"
    return "COUPLED_EXECUTED_DZ_ONLY"


def _classify_dx(
    dx0: float,
    dx1: float,
    *,
    min_mm: float,
    n_actions: int,
    skipped: bool,
    no_action_flag: bool,
) -> str:
    if no_action_flag or skipped:
        return "NO_ACTION_NEEDED"
    if n_actions <= 0:
        return "NO_MEANINGFUL_ACTION"
    if abs(dx0 - dx1) >= min_mm or (abs(dx0) - abs(dx1)) >= min_mm:
        return "DX_EXECUTED"
    return "NO_MEANINGFUL_ACTION"


def _classify_dz(
    dz0: float,
    dz1: float,
    *,
    min_mm: float,
    n_actions: int,
    skipped: bool,
    no_action_flag: bool,
    mode: str | None,
) -> str:
    if no_action_flag or skipped:
        return "NO_ACTION_NEEDED"
    if n_actions <= 0:
        return "NO_MEANINGFUL_ACTION"
    if (abs(dz0) - abs(dz1)) >= min_mm or abs(dz0 - dz1) >= min_mm:
        if mode and "inward" in str(mode).lower():
            return "DZ_INWARD_EXECUTED"
        if mode and "outward" in str(mode).lower():
            return "DZ_OUTWARD_EXECUTED"
        return "DZ_EXECUTED"
    return "NO_MEANINGFUL_ACTION"


def run_forced_dz(env, blob: dict, cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    pcfg = cfg or forced_precision_config(force_skill="dz")
    pcfg.force_precision = True
    pcfg.force_skill = "dz"
    pcfg.try_direct_handoff = False

    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    dx0 = float(err0["dx_mm"])
    dz0 = float(err0["dz_mm"])
    epos0 = float(err0.get("e_pos_mm") or 0.0)

    res = run_precision(env, blob, "D", pcfg)
    err1 = res.errors_exit or measure_errors(env.sim)
    dx1 = float(err1.get("dx_mm") or 0.0)
    dz1 = float(err1.get("dz_mm") or 0.0)
    epos1 = float(err1.get("e_pos_mm") or 0.0)
    acts = list(res.actions or [])
    logs = res.primitive_logs or {}
    dz_log = logs.get("dz") or {}
    mode = None
    for ph in res.phases or []:
        if "inward" in str(ph):
            mode = "B_inward"
        elif "outward" in str(ph):
            mode = "A_outward"
    if dz_log.get("branch"):
        mode = str(dz_log.get("branch"))
    skipped = bool(dz_log.get("skipped")) or ("FORCE_DZ_NO_ACTION_NEEDED" in (res.phases or []))
    min_mm = float(getattr(pcfg, "force_min_meaningful_mm", 0.5) or 0.5)
    demo_class = _classify_dz(
        dz0,
        dz1,
        min_mm=min_mm,
        n_actions=len(acts),
        skipped=skipped,
        no_action_flag="FORCE_DZ_NO_ACTION_NEEDED" in (res.phases or []),
        mode=mode,
    )
    hand = res.handoff or {}
    fail_msgs = list(res.failures or [])
    reason = fail_msgs[0] if fail_msgs else (dz_log.get("reason") or res.note)
    strict_ok = bool(
        res.status in ("PRECISION_EXECUTED_SUCCESS",)
        and hand.get("SUCCESS_NEW")
        and demo_class.startswith("DZ_")
        and "EXECUTED" in demo_class
    )
    return {
        "ok": strict_ok,
        "forced_precision": True,
        "requested_skill": "dz",
        "executed_skill": demo_class.lower() if strict_ok else demo_class,
        "demo_class": demo_class,
        "dz_mode": mode,
        "status": res.status,
        "dx0_mm": dx0,
        "dz0_mm": dz0,
        "dx1_mm": dx1,
        "dz1_mm": dz1,
        "e_pos0_mm": epos0,
        "e_pos1_mm": epos1,
        "dz_reduction_mm": abs(dz0) - abs(dz1),
        "n_actions": len(acts),
        "actions": acts,
        "phases": list(res.phases or []),
        "handoff": hand,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S_final": bool(hand.get("SUCCESS_12S")),
        "bilateral_exit": bool(err1.get("bilateral")),
        "primitive_logs": {k: v for k, v in logs.items() if k != "cycles"},
        "precision_result": res.to_dict(),
        "capture": res.capture,
        "ref": res.ref or blob.get("ref"),
        "note": res.note,
        "reason": reason,
        "failures": fail_msgs,
    }


def run_forced_dx(env, blob: dict, cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    pcfg = cfg or forced_precision_config(force_skill="dx")
    pcfg.force_precision = True
    pcfg.force_skill = "dx"
    pcfg.try_direct_handoff = False

    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    dx0 = float(err0["dx_mm"])
    dz0 = float(err0["dz_mm"])
    epos0 = float(err0.get("e_pos_mm") or 0.0)

    res = run_precision(env, blob, "D", pcfg)
    err1 = res.errors_exit or measure_errors(env.sim)
    dx1 = float(err1.get("dx_mm") or 0.0)
    dz1 = float(err1.get("dz_mm") or 0.0)
    epos1 = float(err1.get("e_pos_mm") or 0.0)
    acts = list(res.actions or [])
    logs = res.primitive_logs or {}
    st = logs.get("dx_support_transfer") or {}
    skipped = bool(st.get("skipped")) or ("FORCE_DX_NO_ACTION_NEEDED" in (res.phases or []))
    min_mm = float(getattr(pcfg, "force_min_meaningful_mm", 0.5) or 0.5)
    demo_class = _classify_dx(
        dx0,
        dx1,
        min_mm=min_mm,
        n_actions=len(acts),
        skipped=skipped,
        no_action_flag="FORCE_DX_NO_ACTION_NEEDED" in (res.phases or []),
    )
    hand = res.handoff or {}
    fail_msgs = list(res.failures or [])
    reason = None
    if fail_msgs:
        reason = fail_msgs[0]
    elif st.get("reason"):
        reason = f"support_transfer:{st.get('phase')}:{st.get('reason')}"
    strict_ok = bool(
        res.status in ("PRECISION_EXECUTED_SUCCESS",)
        and hand.get("SUCCESS_NEW")
        and demo_class == "DX_EXECUTED"
    )
    return {
        "ok": strict_ok,
        "forced_precision": True,
        "requested_skill": "dx",
        "executed_skill": "dx" if demo_class == "DX_EXECUTED" else demo_class.lower(),
        "demo_class": demo_class,
        "status": res.status,
        "dx0_mm": dx0,
        "dz0_mm": dz0,
        "dx1_mm": dx1,
        "dz1_mm": dz1,
        "e_pos0_mm": epos0,
        "e_pos1_mm": epos1,
        "dx_reduction_mm": abs(dx0) - abs(dx1),
        "n_actions": len(acts),
        "actions": acts,
        "phases": list(res.phases or []),
        "handoff": hand,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S_final": bool(hand.get("SUCCESS_12S")),
        "bilateral_exit": bool(err1.get("bilateral")),
        "primitive_logs": {k: v for k, v in logs.items() if k != "cycles"},
        "precision_result": res.to_dict(),
        "capture": res.capture,
        "ref": res.ref or blob.get("ref"),
        "note": res.note,
        "reason": reason,
        "failures": fail_msgs,
    }


def run_forced_coupled(env, blob: dict, cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    """Force experimental Coupled DX+DZ controller (not production FSM)."""
    pcfg = cfg or forced_precision_config(force_skill="coupled")
    pcfg.force_precision = True
    pcfg.force_skill = "coupled"
    min_mm = float(getattr(pcfg, "force_min_meaningful_mm", 0.5) or 0.5)

    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    dx0 = float(err0["dx_mm"])
    dz0 = float(err0["dz_mm"])

    # Feasibility precondition: both axes must have measurable residual for a coupled demo.
    if abs(dx0) < float(pcfg.force_ex_tol_mm or 1.0) or abs(dz0) < float(pcfg.force_ez_tol_mm or 1.0):
        return {
            "ok": False,
            "forced_precision": True,
            "requested_skill": "coupled",
            "executed_skill": "none",
            "demo_class": "NO_MEANINGFUL_ACTION",
            "status": "COUPLED_PRECONDITION_SMALL_RESIDUAL",
            "dx0_mm": dx0,
            "dz0_mm": dz0,
            "dx1_mm": dx0,
            "dz1_mm": dz0,
            "n_actions": 0,
            "actions": [],
            "reason": "one_or_both_axis_below_force_tol",
            "SUCCESS_NEW": False,
            "capture": blob["capture"],
            "ref": blob.get("ref"),
        }

    epos0 = float(err0.get("e_pos_mm") or 0.0)
    coup = coupled_dx_dz(env, blob, cfg=pcfg)
    acts = list(coup.get("actions") or [])
    err1 = coup.get("errors1") or measure_errors(env.sim)
    dx1 = float(err1.get("dx_mm") or 0.0)
    dz1 = float(err1.get("dz_mm") or 0.0)
    epos1 = float(err1.get("e_pos_mm") or 0.0)
    aborted = (not coup.get("ok")) and bool(coup.get("infeasible") or not acts)
    demo_class = _classify_coupled(
        dx0, dz0, dx1, dz1, min_mm=min_mm, aborted=bool(aborted and not acts), n_actions=len(acts)
    )
    if coup.get("infeasible") and not acts:
        demo_class = "COUPLED_ABORTED"

    both_axis_improved = (abs(dx0) - abs(dx1)) >= min_mm and (abs(dz0) - abs(dz1)) >= min_mm
    e_pos_improved = (epos0 - epos1) >= min_mm
    primary = "none"
    if (abs(dx0) - abs(dx1)) >= (abs(dz0) - abs(dz1)) and (abs(dx0) - abs(dx1)) >= min_mm:
        primary = "dx"
    elif (abs(dz0) - abs(dz1)) >= min_mm:
        primary = "dz"
    if both_axis_improved:
        primary = "both"

    # Restore fingertips DOWN then STRICT handoff whenever physical actions ran.
    cur = {"capture": coup.get("capture") or take_capture(env), "ref": blob.get("ref")}
    restore_capture(env, cur["capture"])
    handoff = None
    status = "COUPLED_PRIMITIVE_FAILED"
    try_handoff = len(acts) > 0 and not (coup.get("infeasible") and not acts)
    if try_handoff:
        if not posture_is_fingertips_down(env.sim):
            rd = restore_fingertips_down(env, cur, pcfg)
            acts += rd.get("actions") or []
            if rd.get("capture"):
                cur = {"capture": rd["capture"], "ref": cur["ref"]}
            if not rd.get("ok"):
                restore_capture(env, cur["capture"])
                p = run_tilt_phase(
                    env,
                    cur,
                    target_dir_w=DOWN,
                    phase="p15_coupled_restore_down",
                    rate_scale=0.16,
                    fn_hold=FN_HOLD,
                    ang_tol_deg=5.0,
                    allow_roll=False,
                    contact_aware=True,
                    max_time_s=8.0,
                )
                acts += p.get("actions") or []
                if p.get("capture"):
                    cur = {"capture": p["capture"], "ref": cur["ref"]}
        gate = run_final_down_handoff_gate(env, cur, pcfg, acts)
        if gate.get("ok"):
            cur = {"capture": gate["capture"], "ref": cur["ref"]}
            probe = probe_direct_handoff(
                env, cur["capture"], cur["ref"], stabilize_n=pcfg.stabilize_handoff_n
            )
            restore_capture(env, cur["capture"])
            handoff = probe.get("handoff")
            if probe.get("SUCCESS_NEW"):
                status = "PRECISION_EXECUTED_SUCCESS"
            else:
                status = "PRECISION_EXECUTED_HANDOFF_FAIL"
        else:
            status = "COUPLED_DOWN_GATE_FAILED"
    else:
        status = coup.get("reason") or demo_class

    # P16 accept: meaningful physical actions + official STRICT — not both-axis-only.
    meaningful = demo_class not in ("NO_MEANINGFUL_ACTION", "COUPLED_ABORTED") or (
        len(acts) > 0 and (e_pos_improved or both_axis_improved)
    )
    if len(acts) > 0 and status == "PRECISION_EXECUTED_SUCCESS" and not meaningful:
        # Actions ran and STRICT passed but residuals did not improve — unsuccessful trade-off.
        meaningful = False
    strict_ok = bool(
        status == "PRECISION_EXECUTED_SUCCESS"
        and handoff
        and handoff.get("SUCCESS_NEW")
        and len(acts) > 0
        and (e_pos_improved or both_axis_improved or demo_class.startswith("COUPLED_EXECUTED"))
    )
    # Reject pure unsuccessful trade-offs: STRICT but e_pos worse and no axis improved.
    if strict_ok and (not e_pos_improved) and (not both_axis_improved) and demo_class == "NO_MEANINGFUL_ACTION":
        strict_ok = False
    return {
        "ok": strict_ok,
        "forced_precision": True,
        "requested_skill": "coupled",
        "executed_skill": "coupled" if strict_ok else demo_class,
        "demo_class": demo_class,
        "status": status,
        "dx0_mm": dx0,
        "dz0_mm": dz0,
        "dx1_mm": dx1,
        "dz1_mm": dz1,
        "e_pos0_mm": epos0,
        "e_pos1_mm": epos1,
        "both_axis_improved": both_axis_improved,
        "e_pos_improved": e_pos_improved,
        "primary_improvement": primary,
        "dx_reduction_mm": abs(dx0) - abs(dx1),
        "dz_reduction_mm": abs(dz0) - abs(dz1),
        "n_actions": len(acts),
        "actions": acts,
        "phases": ["FORCE_PRECISION", "FORCE_COUPLED", demo_class, status],
        "handoff": handoff,
        "SUCCESS_NEW": bool((handoff or {}).get("SUCCESS_NEW")),
        "SUCCESS_12S_final": bool((handoff or {}).get("SUCCESS_12S")),
        "bilateral_exit": bool(err1.get("bilateral")),
        "coupled_result": {k: v for k, v in coup.items() if k not in ("actions", "capture")},
        "capture": cur.get("capture"),
        "ref": cur.get("ref"),
        "reason": coup.get("reason"),
        "failures": [] if strict_ok else [status, demo_class],
    }


def run_forced_precision(
    env,
    blob: dict,
    *,
    force_skill: str,
    cfg: PrecisionConfig | None = None,
) -> dict[str, Any]:
    skill = str(force_skill).strip().lower()
    if skill == "dx":
        return run_forced_dx(env, blob, cfg=cfg)
    if skill == "dz":
        return run_forced_dz(env, blob, cfg=cfg)
    if skill == "coupled":
        return run_forced_coupled(env, blob, cfg=cfg)
    raise ValueError(f"Unsupported force_skill={force_skill!r}; use 'dx', 'dz', or 'coupled'")
