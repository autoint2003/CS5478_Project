"""Shared conditional Precision Oracle (Stage-2) for Plans A, B, and C.

Stage-1 recovery remains plan-specific. This module only runs after a verified
stable grasp and may skip entirely when direct HANDOFF already succeeds.

Offline Oracle: may use privileged geometry/forces for planning. Runtime policy
must continue to use only the frozen 39D observation/history interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.precision.axis_aware import (
    axial_desire_direction,
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

ROOT = Path(__file__).resolve().parents[3]
PIVOT_SEARCH = ROOT / "results/diagnostics/position_preserving_pivot/search.json"

# Verified DualLoop gains (geometry / axis-aware diagnostics)
FORCE_KP, FORCE_DA = 0.10, 0.12
SLIP_KN = 900.0
V_SLIP = -0.001
FN_MIN, FN_MAX = 3.8, 11.0
FN_INIT, FN_BRAKE = 5.2, 14.0
FN_HOLD = 13.0
HOLD_STABLE_S = 0.15
Z_ABORT = 0.44
LATERAL_DRIFT_ABORT_MM = 4.0
LATERAL_ABS_ABORT_MM = 8.0
ROLL_ANGLE_DEG = 90.0  # verified endpoint-only / handoff_state_optimization
A0_SLIP = 0.55
# P3 sliding-roll candidates at horizontal (sign, angle_deg) — offline diagnostic
HORIZONTAL_SLIDE_ROLL_CANDIDATES = (
    (-1.0, 30.0),
    (-1.0, 45.0),
    (-1.0, 60.0),
    (-1.0, 75.0),
    (-1.0, 90.0),
    (1.0, 30.0),
    (1.0, 45.0),
    (1.0, 60.0),
    (1.0, 75.0),
    (1.0, 90.0),
)
G_LATERAL_MIN = 4.0
FINGERTIP_MARGIN_MIN_MM = 2.0

# Phase-specific failure classifications
FAIL_PREPARATION = "PREPARATION_FAILED"
FAIL_SUPPORT_ROLL = "SUPPORT_ROLL_FAILED"
FAIL_HORIZONTAL_TILT = "HORIZONTAL_TILT_FAILED"
FAIL_SLIDING_ROLL = "SLIDING_ROLL_FAILED"
FAIL_SLIP_CONTACT = "SLIP_CONTACT_LOSS"
FAIL_SLIP_WRONG_DIR = "SLIP_WRONG_DIRECTION"
FAIL_SLIP_TARGET = "SLIP_TARGET_NOT_REACHED"
FAIL_BRAKE = "BRAKE_FAILED"
FAIL_REGRIP = "REGRIP_FAILED"
FAIL_REVERSE_ROLL = "REVERSE_ROLL_FAILED"
FAIL_RETURN_TILT = "RETURN_TILT_FAILED"
FAIL_FINAL_UNROLL = "FINAL_UNROLL_FAILED"
FAIL_HANDOFF = "HANDOFF_FAILED"

# Goal-driven slip / correction stop reasons (v2.2)
STOP_GOAL_REACHED = "GOAL_REACHED"
STOP_BRAKE_AND_REASSESS = "BRAKE_AND_REASSESS"
STOP_CONTACT_SAFETY = "CONTACT_SAFETY_ABORT"
STOP_WRONG_DIRECTION = "WRONG_DIRECTION"
STOP_NO_PROGRESS = "NO_PROGRESS"
STOP_MAX_SAFE_TRAVEL = "MAX_SAFE_TRAVEL"
STOP_TIME_BUDGET = "TIME_BUDGET"
STOP_HANDOFF_READY = "HANDOFF_READY"


@dataclass
class SlipGoal:
    """Goal-driven axial/lateral slip objective. Travel limits are safety bounds only."""

    axis: str  # "dx" | "dz"
    desired_direction: np.ndarray | list[float]  # hand-frame unit desire for COM motion
    max_safe_travel_mm: float = 12.0
    max_duration_s: float = 12.0
    # Local axis band (AXIS_V1 / working); Euclidean SUCCESS_NEW uses working_pos_mm
    axis_tol_mm: float = 12.0
    working_pos_mm: float = WORKING_POS_MM
    working_theta_deg: float = WORKING_THETA_DEG
    ey_tol_mm: float = AXIS_V1_EY_MM
    # Optional signed target for the controlled component (None = drive toward 0)
    target_signed_mm: float | None = 0.0
    min_progress_mm: float = 0.15
    brake_margin_mm: float = 0.8  # start braking slightly before axis band
    # When False (e.g. fingertips-up inward slip), do not trust live e_pos / HANDOFF_READY.
    use_euclidean_stop: bool = True


def _failure_snapshot(sim, *, phase: str, code: str, last_action=None) -> dict[str, Any]:
    s = sample_state(sim)
    err = measure_errors(sim)
    return {
        "failure_code": code,
        "phase": phase,
        "sim_time": float(sim.data.time),
        "last_action": None if last_action is None else np.asarray(last_action, float).tolist(),
        "fn_L": s["fn_L"],
        "fn_R": s["fn_R"],
        "fn_tot": s["fn_tot"],
        "e_signed_mm": err["e_signed_mm"],
        "v_rel_norm": s["v_rel_norm"],
        "alpha_up_deg": err["alpha_up_deg"],
        "obj_z": err["obj_z"],
        "bilateral": s["bilateral"],
    }


@dataclass
class PrecisionConfig:
    """Tunable offline Oracle options. Does not change SUCCESS_NEW definitions.

    Slip ``*_mm`` fields are **max safe travel** bounds (diagnostics / safety), not
    desired displacements. Goal-driven DualLoop stops on geometric target regions.
    """

    try_direct_handoff: bool = True
    evaluate_handoff: bool = True
    allow_horizontal_dx: bool = True
    use_verified_lateral_fallback: bool = True
    accept_partial_horizontal_dx: bool = True  # v2.2: keep useful partial +dx for dz follow-up
    # Commit horizontal only if HANDOFF or counterfactual axial from post-state succeeds.
    require_horizontal_recoverability: bool = True
    max_retries_per_primitive: int = 1
    max_correction_cycles: int = 4
    global_action_budget: int = 2500  # ~50 s @ 20 ms
    pivot_tag: str | None = None  # override; else load verified tag / search
    ex_tol_mm: float = AXIS_V1_EX_MM
    ey_tol_mm: float = AXIS_V1_EY_MM
    ez_tol_mm: float = AXIS_V1_EZ_MM
    theta_tol_deg: float = AXIS_V1_THETA_DEG
    # Also require SUCCESS_NEW Euclidean readiness for skip decision
    require_success_new_pose: bool = True
    working_pos_mm: float = WORKING_POS_MM
    working_theta_deg: float = WORKING_THETA_DEG
    # Safety / diagnostic max travel (NOT desired slide distances)
    inward_slip_mm: float = 12.0
    outward_slip_mm: float = 12.0
    lateral_slip_mm: float = 14.0
    horizontal_slip_mm: float = 14.0
    # Verified lateral: stop in a tight |dx| band so follow-up axial remains recoverable.
    # Empirically |dx|≲3 mm after gravity-assist lateral preserves Plan-B inward contact;
    # larger overshoot (|dx|~4–5) often causes CONTACT_SAFETY_ABORT on inward DualLoop.
    lateral_axis_tol_mm: float = 3.0
    use_p1_p2_search: bool = False  # joint P1/P2 search for +dx horizontal prep
    p1_p2_winner: dict | None = None  # optional precomputed winner_params to skip re-search
    # Deprecated v2.1: ignored when goal-driven slip is active (kept for API compat).
    horizontal_legacy_total: bool = False
    use_goal_driven_slip: bool = True
    stabilize_handoff_n: int = 8
    book: Any = None  # optional codebook for pivot search fallback


@dataclass
class PrecisionResult:
    status: str
    plan_id: str
    precision_skipped: bool
    precision_executed: bool
    gate_ok: bool
    phases: list[str] = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    errors_entry: dict = field(default_factory=dict)
    errors_exit: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    capture: Any = None
    ref: Any = None
    handoff: dict | None = None
    direct_handoff_probe: dict | None = None
    primitive_logs: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("capture", None)
        d.pop("ref", None)
        # actions may be large; keep count + optional thin summary elsewhere
        acts = d.get("actions") or []
        d["n_actions"] = len(acts)
        d["actions"] = None
        return d


def measure_errors(sim) -> dict[str, Any]:
    s = sample_state(sim)
    m = measure(sim, U_TGT, P_TGT)
    e = np.asarray(s["e_signed_mm"], float)
    return {
        "e_signed_mm": e.tolist(),
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "dtheta_deg": float(s["e_theta_deg"]),
        "e_pos_mm": float(s["e_pos_mm"]),
        "bilateral": bool(s["bilateral"]),
        "v_rel_norm": float(s["v_rel_norm"]),
        "w_rel_norm": float(s.get("w_rel_norm") or 0.0),
        "fn_L": float(s["fn_L"]),
        "fn_R": float(s["fn_R"]),
        "fn_tot": float(s["fn_tot"]),
        "obj_z": float(m["obj_z"]),
        "alpha_up_deg": float(gates(sim)["alpha_up_deg"]),
        "fingertips_down": bool(gates(sim)["fingertips_down"]),
        "fingertips_up": bool(gates(sim)["fingertips_up"]),
    }


def entry_gate(env, *, thresholds: dict | None = None) -> dict[str, Any]:
    """150 ms stable-grasp gate before any precision work."""
    stab = hold_and_confirm_stable(env, hold_s=0.20, thresholds=thresholds)
    err = measure_errors(env.sim)
    return {
        "ok": bool(stab["stable"]),
        "stability": {k: stab[k] for k in ("stable", "reasons", "consecutive_bil_s_at_end", "e_theta_deg", "e_pos_mm", "e_signed_mm")},
        "errors": err,
        "thresholds": {**(STABLE_THRESHOLDS), **(thresholds or {})},
    }


def assess_need(errors: dict, cfg: PrecisionConfig) -> dict[str, Any]:
    """Signed-error decision: dtheta / dx / dz needed?"""
    e = np.asarray(errors["e_signed_mm"], float)
    theta = float(errors["dtheta_deg"])
    axis = decide_precision(
        e,
        theta_deg=theta,
        bilateral=bool(errors["bilateral"]),
        ex_tol_mm=cfg.ex_tol_mm,
        ey_tol_mm=cfg.ey_tol_mm,
        ez_tol_mm=cfg.ez_tol_mm,
        theta_tol_deg=cfg.theta_tol_deg,
    )
    dtheta_needed = theta > cfg.theta_tol_deg
    # Prefer SUCCESS_NEW Euclidean for "already ready" when configured
    pose_ready = True
    if cfg.require_success_new_pose:
        pose_ready = (
            float(errors["e_pos_mm"]) <= cfg.working_pos_mm
            and theta <= cfg.working_theta_deg
            and abs(float(e[1])) <= cfg.ey_tol_mm
        )
    axial = select_axial_mode(float(e[2]), axial_tol_mm=cfg.ez_tol_mm)
    dx_needed = abs(float(e[0])) > cfg.ex_tol_mm
    dz_needed = abs(float(e[2])) > cfg.ez_tol_mm
    # SUCCESS_NEW uses Euclidean e_pos≤12 mm. AXIS_V1 may leave |ez|≤20 with e_pos>12;
    # request correction along the dominant residual so Direct/HANDOFF readiness matches SUCCESS_NEW.
    e_pos = float(errors.get("e_pos_mm") or 0.0)
    residual_note = None
    if cfg.require_success_new_pose and e_pos > cfg.working_pos_mm and not dtheta_needed:
        if not dx_needed and not dz_needed:
            if abs(float(e[2])) >= abs(float(e[0])):
                dz_needed = True
                residual_note = "e_pos_over_working_dominant_ez"
            else:
                dx_needed = True
                residual_note = "e_pos_over_working_dominant_ex"
        elif not dz_needed and abs(float(e[2])) >= abs(float(e[0])) and abs(float(e[2])) > 0.5 * cfg.ez_tol_mm:
            dz_needed = True
            residual_note = "e_pos_over_working_add_ez"
    # Recompute axial mode if we newly requested dz
    if dz_needed and axial.get("mode") == "none":
        axial = select_axial_mode(float(e[2]), axial_tol_mm=min(cfg.ez_tol_mm, max(abs(float(e[2])) - 0.1, 1.0)))
    ready = bool(pose_ready and not dtheta_needed and not dx_needed and not dz_needed and axis["mode"] != "abnormal")
    return {
        "ready_for_direct_handoff_assessment": ready,
        "dtheta_needed": dtheta_needed,
        "dx_needed": dx_needed,
        "dz_needed": dz_needed,
        "dz_mode": axial.get("mode"),
        "axis_decision": axis,
        "pose_ready_success_new": pose_ready,
        "e_signed_mm": e.tolist(),
        "dtheta_deg": theta,
        "e_pos_mm": e_pos,
        "residual_note": residual_note,
    }


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


def run_slip_dual_loop(
    env,
    blob,
    d_axis,
    *,
    milestone_mm: float,
    max_s: float = 12.0,
    z_ref: float | None = None,
    guard_lateral: bool = True,
    e_x0: float | None = None,
    vx_target: float = V_SLIP,
) -> dict[str, Any]:
    """Verified DualLoop slip along desire direction d (cylinder COM relative)."""
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    e0 = np.asarray(s0["e_signed_mm"], float)
    if e_x0 is None:
        e_x0 = float(e0[0])
    d = np.asarray(d_axis, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {
            "ok": False,
            "failure_code": FAIL_SLIP_TARGET,
            "reason": "zero_desire",
            "actions": [],
            "series": [],
            "capture": take_capture(env),
            "reached": False,
            "brake_ok": False,
            "regrip_ok": False,
        }
    d = d / n
    # SlipOuterLoop: axis = −d with vx_target < 0 ⇒ unload when not sliding along d
    axis = -d
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=vx_target, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=A0_SLIP, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple(axis.tolist()))
    actions, series = [], []
    lost = 0
    reached = False
    abort_reason = None
    failure_code = None
    for i in range(max(1, int(round(max_s / DT)))):
        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        e2 = np.asarray(s2["e_signed_mm"], float)
        disp = float(np.dot(dp, d)) * 1e3
        series.append(
            {
                "t": (i + 1) * DT,
                "fn": s2["fn_tot"],
                "disp_along_mm": disp,
                "e_signed": s2["e_signed_mm"],
                "theta": s2["e_theta_deg"],
                "bilateral": s2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "slip",
            }
        )
        if failed or latched(env.sim) or not s2["bilateral"] or series[-1]["obj_z"] < Z_ABORT:
            code = FAIL_SLIP_CONTACT if not s2["bilateral"] else "WORLD_HEIGHT"
            return {
                "ok": False,
                "failure_code": code,
                "reason": "contact_loss" if not s2["bilateral"] else "world_height",
                "actions": actions,
                "series": series,
                "reached": False,
                "brake_ok": False,
                "regrip_ok": False,
                "capture": take_capture(env),
                "failure": _failure_snapshot(env.sim, phase="slip", code=code, last_action=a),
            }
        if guard_lateral:
            if abs(float(e2[0]) - e_x0) > LATERAL_DRIFT_ABORT_MM or abs(float(e2[0])) > LATERAL_ABS_ABORT_MM:
                abort_reason = "lateral_drift"
                failure_code = FAIL_SLIP_WRONG_DIR
                break
        if disp < -0.5 and i > 10:
            abort_reason = "wrong_direction"
            failure_code = FAIL_SLIP_WRONG_DIR
            break
        if disp >= milestone_mm:
            reached = True
            break

    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(0.5, series[-1]["fn"] if series else 10.0)
    n_brake = max(1, int(round(0.40 / DT)))
    n_hold = max(1, int(round(HOLD_STABLE_S / DT)))
    for j in range(n_brake + n_hold):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        series.append(
            {
                "t": (series[-1]["t"] if series else 0.0) + DT,
                "fn": s2["fn_tot"],
                "disp_along_mm": float(np.dot(dp, d) * 1e3),
                "e_signed": s2["e_signed_mm"],
                "theta": s2["e_theta_deg"],
                "bilateral": s2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "brake" if j < n_brake else "hold",
            }
        )
        if not s2["bilateral"]:
            code = FAIL_BRAKE if j < n_brake else FAIL_REGRIP
            return {
                "ok": False,
                "failure_code": code,
                "reason": "loss_on_brake" if j < n_brake else "loss_on_regrip",
                "actions": actions,
                "series": series,
                "reached": reached,
                "brake_ok": False,
                "regrip_ok": False,
                "capture": take_capture(env),
                "failure": _failure_snapshot(env.sim, phase="brake" if j < n_brake else "hold", code=code, last_action=a),
            }

    disp = float(series[-1]["disp_along_mm"])
    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    ok = bool(hold_ok and (reached or disp >= 0.5 * milestone_mm) and abort_reason is None)
    if not ok and failure_code is None:
        failure_code = FAIL_SLIP_TARGET if not reached else FAIL_REGRIP
    return {
        "ok": ok,
        "reached": reached,
        "abort_reason": abort_reason,
        "failure_code": failure_code,
        "actions": actions,
        "series": series,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "theta": series[-1]["theta"],
        "brake_ok": True,
        "regrip_ok": hold_ok,
        "capture": take_capture(env),
    }


def settle_hold(env, actions: list, *, fn_hold: float = FN_HOLD, hold_s: float = 0.4) -> None:
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(max(1, int(round(hold_s / DT)))):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())


def probe_direct_handoff(env, capture, ref, *, stabilize_n: int = 8) -> dict[str, Any]:
    """Counterfactual: clone state, HANDOFF, evaluate; restore caller via capture return."""
    restore_capture(env, capture)
    for _ in range(stabilize_n):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    pre = measure_errors(env.sim)
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "pre": pre,
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
        "reason": h.get("std_reason") or h.get("reason"),
        "handoff": {k: v for k, v in h.items() if k not in ("series", "samples", "trace")},
    }


def correct_dtheta(env, blob, cfg: PrecisionConfig, plan_id: str) -> dict[str, Any]:
    tag = cfg.pivot_tag or default_pivot_tag(plan_id)
    actions: list = []
    if tag:
        acts = pivot_actions_from_tag(tag)
        met = pivot_segment_metrics(env, blob, acts, record_series=False)
        restore_capture(env, blob["capture"])
        for a in acts:
            step_legal(env.sim, np.asarray(a, np.float32))
            actions.append(np.asarray(a, np.float32).copy())
        ok = bool(met.get("ok") and sample_state(env.sim)["bilateral"])
        # Prefer θ improvement; tolerate verified class labels
        th1 = float(measure_errors(env.sim)["dtheta_deg"])
        improved = th1 <= cfg.theta_tol_deg or (met.get("theta0", 99) - th1) >= 2.0
        settle_hold(env, actions, hold_s=0.3)
        return {
            "ok": bool(ok and improved),
            "mechanism": "verified_pivot_tag",
            "tag": tag,
            "metrics": {k: met[k] for k in met if k not in ("series", "a0", "a1", "p0", "p1")},
            "actions": actions,
            "capture": take_capture(env),
            "errors": measure_errors(env.sim),
        }

    # Fallback: bounded primitive search (requires codebook)
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
    return {
        "ok": True,
        "mechanism": "primitive_contact_pivot",
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
        "best": piv.get("best"),
    }


def correct_dx_verified(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """Verified gravity-assisted lateral slip (orientation plan → DualLoop).

    Goal-driven by default: stop when |dx| enters ``lateral_axis_tol_mm`` (safety
    travel cap only). Fixed-displacement DualLoop is retained when
    ``use_goal_driven_slip`` is False.
    """
    actions: list = []
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    d_lat = lateral_desire_direction(s0["e_signed_mm"])
    if float(np.linalg.norm(d_lat)) < 0.5:
        return {"ok": True, "skipped": True, "reason": "dx_near_zero", "actions": [], "capture": blob["capture"]}

    plan = plan_orientation(env, blob, d=d_lat, simulate=True, g_useful_min=8.5)
    if not (plan.get("ok") or plan.get("already_sufficient")):
        return {
            "ok": False,
            "mechanism": "verified_lateral",
            "phase": "plan",
            "reason": "lateral_orientation_unreachable",
            "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
            "actions": [],
            "capture": blob["capture"],
        }
    exe = execute_orientation_plan(env, blob, plan, adaptive=True)
    actions.extend(exe.get("actions") or [])
    if not exe.get("ok") and not plan.get("already_sufficient"):
        return {
            "ok": False,
            "mechanism": "verified_lateral",
            "phase": "reori",
            "exe": {k: v for k, v in exe.items() if k not in ("actions", "series", "capture")},
            "actions": actions,
            "capture": exe.get("capture") or blob["capture"],
        }
    tilt = {"capture": exe.get("capture") or take_capture(env), "ref": blob["ref"]}
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    max_safe = float(cfg.lateral_slip_mm)

    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        # Axis-only stop: tipped gravity-assist e_pos is not a reliable HANDOFF proxy,
        # and overshooting past ~3 mm often breaks subsequent inward axial contact.
        axis_tol = float(getattr(cfg, "lateral_axis_tol_mm", 3.0) or 3.0)
        goal = SlipGoal(
            axis="dx",
            desired_direction=d_lat,
            max_safe_travel_mm=max_safe,
            max_duration_s=20.0,
            axis_tol_mm=axis_tol,
            working_pos_mm=float(cfg.working_pos_mm),
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
            use_euclidean_stop=False,
            brake_margin_mm=0.8,
        )
        slip = run_goal_aware_axial_slip(
            env, tilt, goal, z_ref=z_ref, guard_lateral=False,
        )
        actions.extend(slip.get("actions") or [])
        reached = bool(slip.get("ok") and (
            slip.get("goal_reached")
            or slip.get("stop_reason") in (STOP_GOAL_REACHED, STOP_HANDOFF_READY, STOP_MAX_SAFE_TRAVEL)
            or float(slip.get("disp_total_mm") or 0.0) >= 0.5 * min(max_safe, abs(float(s0["e_signed_mm"][0])))
        ))
        if not reached:
            return {
                "ok": False,
                "mechanism": "verified_lateral",
                "phase": "slip",
                "stop_reason": slip.get("stop_reason"),
                "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
                "actions": actions,
                "capture": slip.get("capture") or tilt["capture"],
            }
        restore_capture(env, slip["capture"])
        settle_hold(env, actions, hold_s=0.35)
        return {
            "ok": True,
            "mechanism": "verified_lateral",
            "contract": "goal_driven",
            "stop_reason": slip.get("stop_reason"),
            "goal_reached": slip.get("goal_reached"),
            "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
            "disp_along_mm": slip.get("disp_along_mm"),
            "disp_total_mm": slip.get("disp_total_mm"),
            "actions": actions,
            "capture": take_capture(env),
            "errors": measure_errors(env.sim),
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        }

    milestone = min(max_safe, abs(float(s0["e_signed_mm"][0])) + 1.0)
    slip = run_slip_dual_loop(
        env, tilt, d_lat, milestone_mm=milestone, max_s=20.0, z_ref=z_ref, guard_lateral=False
    )
    actions.extend(slip.get("actions") or [])
    if not slip.get("ok") and not (slip.get("disp_along_mm") or 0) >= 0.5 * milestone:
        return {
            "ok": False,
            "mechanism": "verified_lateral",
            "phase": "slip",
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "actions": actions,
            "capture": slip.get("capture") or tilt["capture"],
        }
    restore_capture(env, slip["capture"])
    # Restore toward fingertips-down if needed (plan_restore optional — settle is enough for many states)
    settle_hold(env, actions, hold_s=0.35)
    return {
        "ok": True,
        "mechanism": "verified_lateral",
        "plan": {k: v for k, v in plan.items() if k not in ("top", "candidates_kin")},
        "disp_along_mm": slip.get("disp_along_mm"),
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
    }


def _horizontal_open_target(sim) -> np.ndarray:
    """World direction for u_open at α≈90° (fingertips horizontal)."""
    u = hand_R(sim) @ np.asarray(U_OPEN_HAND, float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    axis = np.cross(u, UP)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        n = float(np.linalg.norm(axis))
    axis = axis / max(n, 1e-12)
    a0 = float(gates(sim)["alpha_up_deg"])
    delta = np.radians(a0 - 90.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], float)
    R = np.eye(3) + np.sin(delta) * K + (1 - np.cos(delta)) * (K @ K)
    u_tgt = R @ u
    return u_tgt / max(float(np.linalg.norm(u_tgt)), 1e-12)


def _score_horizontal_slide_config(sim, d_lat: np.ndarray) -> dict[str, Any]:
    """Score contact/gravity readiness for lateral DualLoop at current pose."""
    s = sample_state(sim)
    g_sc = score_gravity_alignment(np.asarray(s["g_h"], float), d_lat)
    cf = contact_forces(sim)
    sup = support_force_decomposition(sim)
    bil = bool(s["bilateral"] and cf["fn_L"] > 1.0 and cf["fn_R"] > 1.0)
    useful = float(g_sc["useful"])
    score = useful + (5.0 if bil else -20.0) + 0.01 * float(cf["fn_tot"])
    return {
        "ok": bool(bil and useful >= G_LATERAL_MIN),
        "score": score,
        "g_score": g_sc,
        "bilateral": bil,
        "fn_L": cf["fn_L"],
        "fn_R": cf["fn_R"],
        "support": {k: sup.get(k) for k in ("proximal_fraction", "palm_or_base_support_detected") if k in sup},
        "alpha_up_deg": gates(sim)["alpha_up_deg"],
        "e_signed_mm": s["e_signed_mm"],
    }


def select_horizontal_sliding_roll(
    env, blob, d_lat: np.ndarray, y0, e_ref
) -> dict[str, Any]:
    """Bounded counterfactual search for P3 axial-roll angle/sign at horizontal."""
    trials = []
    best = None
    for sign, ang in HORIZONTAL_SLIDE_ROLL_CANDIDATES:
        restore_capture(env, blob["capture"])
        r = run_roll_phase(
            env, blob, sign=sign, angle_deg=ang, phase=f"dxH_P3_probe_s{int(sign)}_{int(ang)}",
            rate_scale=0.12, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
        )
        # Rejected branch — do not keep actions; restore probe state only for scoring
        if not r.get("ok"):
            trials.append({"sign": sign, "angle_deg": ang, "ok": False, "reason": r.get("reason")})
            continue
        sc = _score_horizontal_slide_config(env.sim, d_lat)
        row = {"sign": sign, "angle_deg": ang, "ok": sc["ok"], "score": sc["score"], "diag": sc}
        trials.append(row)
        if sc["ok"] and (best is None or sc["score"] > best["score"]):
            best = {**row, "capture": r["capture"], "actions": r["actions"]}
    restore_capture(env, blob["capture"])
    if best is None:
        return {"ok": False, "failure_code": FAIL_SLIDING_ROLL, "trials": trials, "reason": "no_viable_P3_roll"}
    return {
        "ok": True,
        "sign": best["sign"],
        "angle_deg": best["angle_deg"],
        "score": best["score"],
        "diag": best["diag"],
        "trials": [{k: v for k, v in t.items() if k not in ("capture", "actions", "diag")} for t in trials],
    }


def _physical_finger_support(sim) -> tuple[bool, dict[str, Any]]:
    """Bilateral contact + force decomposition (not orientation-inferred support)."""
    s = sample_state(sim)
    cf = contact_forces(sim)
    sup = support_force_decomposition(sim)
    sides = sup.get("sides") or {}
    bil = bool(s["bilateral"] and cf["fn_L"] > 1.0 and cf["fn_R"] > 1.0)
    slim_sides = {}
    for k, v in sides.items():
        slim_sides[k] = {
            "n_contacts": v.get("n_contacts"),
            "Fn_z_normal": v.get("Fn_z_normal"),
            "Ft_z_friction": v.get("Ft_z_friction"),
            "normal_support_meaningful": v.get("normal_support_meaningful"),
        }
    return bil, {
        "fn_L": float(cf["fn_L"]),
        "fn_R": float(cf["fn_R"]),
        "fn_tot": float(cf["fn_tot"]),
        "sides": slim_sides,
        "alpha_up_deg": float(gates(sim)["alpha_up_deg"]),
        "u_open_world": (hand_R(sim) @ np.asarray(U_OPEN_HAND, float)).tolist(),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "v_rel_norm": float(s["v_rel_norm"]),
        "w_rel_norm": float(s.get("w_rel_norm") or 0.0),
        "e_signed_mm": s["e_signed_mm"],
    }


def classify_tilt_physical_reason(tilt_result: dict | None, sim=None) -> str:
    """Map tilt failure to a physical event (not a blanket HORIZONTAL_TILT_FAILED)."""
    if not tilt_result:
        return "unknown"
    r = str(tilt_result.get("reason") or "")
    if r == "contact_loss":
        return "true_contact_loss"
    if r == "world_height":
        return "world_height_abort"
    if r == "timeout":
        return "controller_timeout"
    if r == "roll_leak_during_tilt":
        return "roll_leak"
    if r == "ok" and sim is not None:
        g = gates(sim)
        if abs(float(g["alpha_up_deg"]) - 90.0) > 12.0:
            return "orientation_target_not_reached"
    return r or "unknown"


def search_p1_p2_horizontal(env, blob, *, candidates: list[dict] | None = None) -> dict[str, Any]:
    """Bounded joint search over P1 support-roll and P2 tilt-to-horizontal.

    Each trial restores the same pre-P1 capture, executes legal actions, and scores
    contact retention via force decomposition — not hand orientation alone.
    Rejected trials are counterfactual; only the winner's parameters are returned
    for live re-execution on the accepted trajectory.
    """
    restore_capture(env, blob["capture"])
    th0 = float(sample_state(env.sim)["e_theta_deg"])
    e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    auto_sign = float(choose_roll_sign_for_support(env.sim))
    y0 = reference_pad_vector(env.sim)
    cur0 = {"capture": blob["capture"], "ref": blob["ref"]}

    if not gates(env.sim)["fingertips_down"]:
        p0 = run_tilt_phase(
            env, cur0, target_dir_w=DOWN, phase="p12_search_P0", rate_scale=0.20, fn_hold=FN_HOLD,
            ang_tol_deg=5.0, y0_w=y0, e_ref=e0, allow_roll=False, contact_aware=False, max_time_s=8.0,
        )
        if not p0.get("ok"):
            return {
                "ok": False,
                "limiting": "P0_preparation_failed",
                "physical_reason": classify_tilt_physical_reason(p0),
                "n_trials": 0,
                "trials": [],
                "best": None,
                "counterfactual": True,
            }
        cur0 = {"capture": p0["capture"], "ref": blob["ref"]}

    if candidates is None:
        candidates = []
        for ang in (0.0, 25.0, 40.0, 55.0, 70.0):
            signs = (auto_sign,) if ang < 0.5 else (auto_sign, -auto_sign)
            for sgn in signs:
                for rate in (0.06, 0.10):
                    for zb in (0.05, 0.08):
                        candidates.append(
                            {
                                "support_roll_sign": float(sgn),
                                "support_roll_deg": float(ang),
                                "p2_rate": float(rate),
                                "z_boost": float(zb),
                                "ang_tol": 12.0,
                            }
                        )

    trials: list[dict] = []
    best: dict | None = None
    # Prefer low support-roll first (Plan B +dx succeeds with P1=0° in offline search).
    candidates = sorted(
        candidates,
        key=lambda c: (float(c.get("support_roll_deg") or 0.0), float(c.get("p2_rate") or 0.0)),
    )

    for c in candidates:
        restore_capture(env, cur0["capture"])
        e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
        y0_i = reference_pad_vector(env.sim)
        p1_actions: list = []
        if float(c["support_roll_deg"]) > 0.5:
            p1 = run_roll_phase(
                env, cur0, sign=float(c["support_roll_sign"]), angle_deg=float(c["support_roll_deg"]),
                phase="p12_search_P1", rate_scale=0.12, fn_hold=FN_HOLD, require_down=True,
                y0_w=y0_i, e_ref=e_ref,
            )
            if not p1.get("ok"):
                trials.append({
                    **c, "ok": False, "fail_phase": "P1", "reason": p1.get("reason"),
                    "limiting": "support_roll_failed", "counterfactual": True,
                })
                continue
            cur = {"capture": p1["capture"], "ref": blob["ref"]}
            p1_actions = list(p1.get("actions") or [])
        else:
            cur = cur0

        restore_capture(env, cur["capture"])
        bil1, sup1 = _physical_finger_support(env.sim)
        if not bil1:
            trials.append({
                **c, "ok": False, "fail_phase": "P1_support", "support": sup1,
                "limiting": "loss_of_bilateral_after_P1", "counterfactual": True,
            })
            continue

        u_tgt = _horizontal_open_target(env.sim)
        p2 = run_tilt_phase(
            env, cur, target_dir_w=u_tgt, phase="p12_search_P2",
            rate_scale=float(c["p2_rate"]), fn_hold=FN_HOLD,
            ang_tol_deg=float(c["ang_tol"]), y0_w=y0_i, e_ref=e_ref,
            allow_roll=False, contact_aware=True, max_time_s=16.0,
            z_ref_boost=float(c["z_boost"]), height_kp=4.5,
        )
        restore_capture(env, p2.get("capture") or cur["capture"])
        err = measure_errors(env.sim)
        bil2, sup2 = _physical_finger_support(env.sim)
        alpha = float(err["alpha_up_deg"])
        horiz = abs(alpha - 90.0) <= float(c["ang_tol"]) + 2.0
        theta_ok = float(err["dtheta_deg"]) <= max(th0 + 5.0, 12.0)
        z_ok = float(err["obj_z"]) >= Z_ABORT
        try:
            margin = float(fingertip_overlap_margin_mm(env.sim, U_OPEN_HAND))
        except Exception:
            margin = 0.0
        phys = classify_tilt_physical_reason(p2, env.sim)
        # P3 viability (counterfactual): margin alone is unreliable at horizontal;
        # require at least one sliding-roll candidate that keeps bilateral contact.
        p3_ok = False
        p3_best = None
        if bool(p2.get("ok") and horiz and bil2 and z_ok and theta_ok):
            d_lat = lateral_desire_direction(np.asarray(err["e_signed_mm"], float))
            if float(np.linalg.norm(d_lat)) >= 0.5:
                p3_probe = select_horizontal_sliding_roll(
                    env,
                    {"capture": p2["capture"], "ref": blob["ref"]},
                    d_lat,
                    y0_i,
                    np.asarray(err["e_signed_mm"], float),
                )
                p3_ok = bool(p3_probe.get("ok"))
                if p3_ok:
                    p3_best = {
                        "sign": p3_probe.get("sign"),
                        "angle_deg": p3_probe.get("angle_deg"),
                        "score": p3_probe.get("score"),
                    }
            restore_capture(env, p2["capture"])
        ok = bool(p2.get("ok") and horiz and bil2 and z_ok and theta_ok and p3_ok)
        score = (
            -abs(alpha - 90.0)
            + 0.05 * margin
            + (8.0 if bil2 else -30.0)
            + (3.0 if theta_ok else -10.0)
            + (12.0 if p3_ok else -20.0)
            + 0.02 * float(sup2.get("fn_tot") or 0.0)
            + float((p3_best or {}).get("score") or 0.0) * 0.01
        )
        limiting = None
        if not ok:
            if not p2.get("ok"):
                limiting = phys
            elif not bil2:
                limiting = "true_contact_loss"
            elif not z_ok:
                limiting = "world_height_abort"
            elif not horiz:
                limiting = "orientation_target_not_reached"
            elif not theta_ok:
                limiting = "dtheta_not_preserved"
            elif not p3_ok:
                limiting = "no_viable_P3_after_horizontal"
            else:
                limiting = "unknown"
        row = {
            **c,
            "ok": ok,
            "score": score,
            "p1_ok": True,
            "p2_ok": bool(p2.get("ok")),
            "p2_reason": p2.get("reason"),
            "p3_viable": p3_ok,
            "p3_best": p3_best,
            "physical_reason": phys if not ok else "ok",
            "limiting": limiting,
            "alpha_up_deg": alpha,
            "bilateral": bil2,
            "support_after_P1": {k: sup1[k] for k in ("fn_L", "fn_R", "fn_tot", "sides") if k in sup1},
            "support_after_P2": {k: sup2[k] for k in ("fn_L", "fn_R", "fn_tot", "sides") if k in sup2},
            "margin_mm": margin,
            "dtheta0": th0,
            "dtheta": float(err["dtheta_deg"]),
            "obj_z": float(err["obj_z"]),
            "e_signed_mm": err["e_signed_mm"],
            "counterfactual": True,
        }
        trials.append(row)
        if ok and (best is None or score > float(best["score"])):
            best = {
                **row,
                "capture": p2["capture"],
                "actions_p1": p1_actions,
                "actions_p2": list(p2.get("actions") or []),
                "pre_p1_capture": cur0["capture"],
            }
            # Early exit: first viable candidate is enough for live re-execution.
            break

    restore_capture(env, blob["capture"])
    if best is None:
        lim_counts: dict[str, int] = {}
        for t in trials:
            key = str(t.get("limiting") or t.get("fail_phase") or "unknown")
            lim_counts[key] = lim_counts.get(key, 0) + 1
        limiting = max(lim_counts, key=lim_counts.get) if lim_counts else "no_candidates"
        return {
            "ok": False,
            "best": None,
            "trials": trials,
            "n_trials": len(trials),
            "limiting": limiting,
            "limiting_counts": lim_counts,
            "counterfactual": True,
            "physical_reason": limiting,
        }
    return {
        "ok": True,
        "best": {k: v for k, v in best.items() if k not in ("capture", "actions_p1", "actions_p2", "pre_p1_capture")},
        "trials": trials,
        "n_trials": len(trials),
        "limiting": None,
        "counterfactual": True,
        "pre_p1_capture": best["pre_p1_capture"],
        "p1_p2_capture": best["capture"],
        "actions_p1": best["actions_p1"],
        "actions_p2": best["actions_p2"],
        "winner_params": {
            "support_roll_sign": best["support_roll_sign"],
            "support_roll_deg": best["support_roll_deg"],
            "p2_rate": best["p2_rate"],
            "z_boost": best["z_boost"],
            "ang_tol": best["ang_tol"],
        },
    }


def handoff_geometry_ready(errors: dict, cfg: PrecisionConfig) -> bool:
    """Level-2 geometry: Euclidean + theta compatible with SUCCESS_NEW working region."""
    e = np.asarray(errors["e_signed_mm"], float)
    return bool(
        errors.get("bilateral")
        and float(errors["dtheta_deg"]) <= float(cfg.working_theta_deg)
        and float(errors["e_pos_mm"]) <= float(cfg.working_pos_mm)
        and abs(float(e[1])) <= float(cfg.ey_tol_mm)
        and float(errors.get("v_rel_norm") or 0.0) < 0.08
    )


def evaluate_slip_goal(errors: dict, goal: SlipGoal) -> tuple[bool, str | None]:
    """High-frequency local stop test (not a 12 s HANDOFF probe).

    SUCCESS_NEW uses Euclidean e_pos ≤ working_pos_mm. AXIS_V1 axis tolerances
    (especially ez≤20) must not stop sliding while e_pos remains excessive.
    When ``use_euclidean_stop`` is False (tipped-up inward), only the signed axis
    band is used — live e_pos is not predictive of the post-restore grasp.
    """
    e = np.asarray(errors["e_signed_mm"], float)
    theta = float(errors["dtheta_deg"])
    e_pos = float(errors["e_pos_mm"])
    wp = float(goal.working_pos_mm)
    idx = 0 if goal.axis == "dx" else 2
    other = 2 if idx == 0 else 0
    tgt = 0.0 if goal.target_signed_mm is None else float(goal.target_signed_mm)
    axis_err = abs(float(e[idx]) - tgt)
    margin = max(0.0, float(goal.brake_margin_mm))

    if not bool(goal.use_euclidean_stop):
        if axis_err <= float(goal.axis_tol_mm) + margin:
            if axis_err <= float(goal.axis_tol_mm):
                return True, STOP_GOAL_REACHED
        return False, None

    if (
        theta <= float(goal.working_theta_deg)
        and e_pos <= wp
        and abs(float(e[1])) <= float(goal.ey_tol_mm)
    ):
        return True, STOP_HANDOFF_READY

    # Euclidean budget: stop this axis when the controlled component alone cannot
    # be the reason e_pos exceeds the working threshold.
    budget_sq = wp * wp - float(e[0]) ** 2 - float(e[1]) ** 2 - float(e[2]) ** 2 + float(e[idx]) ** 2
    axis_budget = float(np.sqrt(budget_sq)) if budget_sq > 0.0 else 0.0
    if e_pos <= wp:
        return True, STOP_GOAL_REACHED
    if axis_err <= axis_budget + margin:
        if abs(float(e[other])) > abs(float(e[idx])) + 1.0 and abs(float(e[other])) > axis_budget:
            return True, STOP_GOAL_REACHED
        if axis_err <= axis_budget:
            return True, STOP_GOAL_REACHED
    return False, None


def run_goal_aware_axial_slip(
    env,
    blob,
    goal: SlipGoal,
    *,
    z_ref: float | None = None,
    guard_lateral: bool = False,
    guard_fingertip_margin: bool = False,
    min_margin_mm: float = FINGERTIP_MARGIN_MIN_MM,
    fn_min: float = FN_MIN,
    fn_max: float = FN_MAX,
    kn: float = SLIP_KN,
    vx_target: float = V_SLIP,
    fn_init: float = FN_INIT,
    a0_slip: float = A0_SLIP,
    allow_post_brake_nudge: bool = True,
) -> dict[str, Any]:
    """DualLoop slip that stops on geometric goal / safety — not a fixed displacement.

    Displacement is logged diagnostically. ``max_safe_travel_mm`` is a hard safety cap.
    """
    d = np.asarray(goal.desired_direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {
            "ok": False,
            "failure_code": FAIL_SLIP_TARGET,
            "stop_reason": STOP_WRONG_DIRECTION,
            "reason": "zero_direction",
            "actions": [],
            "capture": take_capture(env),
        }
    d = d / n
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(err0["e_signed_mm"][0])
    idx = 0 if goal.axis == "dx" else 2
    e_axis0 = float(err0["e_signed_mm"][idx])

    already, why = evaluate_slip_goal(err0, goal)
    if already:
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": why or STOP_GOAL_REACHED,
            "disp_along_mm": 0.0,
            "disp_total_mm": 0.0,
            "e_signed": err0["e_signed_mm"],
            "errors_entry": err0,
            "errors": err0,
            "actions": [],
            "capture": take_capture(env),
            "reached": True,
            "brake_ok": True,
            "regrip_ok": True,
            "contract": "goal_driven",
            "goal": {"axis": goal.axis, "max_safe_travel_mm": goal.max_safe_travel_mm},
        }

    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=vx_target, fn_min=fn_min, fn_max=fn_max, kn=kn),
        use_slip=True,
    )
    ctl.reset(a0=a0_slip, fn0=s0["fn_tot"], fn_target=fn_init, slip_axis=tuple((-d).tolist()))
    actions: list = []
    series: list[dict] = []
    lost = 0
    last_disp = 0.0
    stop_reason = STOP_TIME_BUDGET
    best_axis_err = abs(e_axis0)
    stall_steps = 0
    goal_hit_pre_brake = False

    def _disp() -> float:
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3

    max_steps = int(float(goal.max_duration_s) / DT)
    for i in range(max_steps):
        err = measure_errors(env.sim)
        hit, why = evaluate_slip_goal(err, goal)
        if hit:
            stop_reason = why or STOP_GOAL_REACHED
            goal_hit_pre_brake = True
            break
        last_disp = _disp()
        if last_disp >= float(goal.max_safe_travel_mm):
            stop_reason = STOP_MAX_SAFE_TRAVEL
            break
        if last_disp < -0.5:
            stop_reason = STOP_WRONG_DIRECTION
            break
        axis_err = abs(float(err["e_signed_mm"][idx]) - (0.0 if goal.target_signed_mm is None else float(goal.target_signed_mm)))
        if axis_err + 1e-6 < best_axis_err - float(goal.min_progress_mm) * 0.25:
            best_axis_err = axis_err
            stall_steps = 0
        else:
            stall_steps += 1
        if stall_steps > int(2.5 / DT) and last_disp > 0.8:
            stop_reason = STOP_NO_PROGRESS
            break

        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        last_disp = _disp()
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {
                "ok": False,
                "failure_code": FAIL_SLIP_CONTACT,
                "stop_reason": STOP_CONTACT_SAFETY,
                "reason": "contact_loss",
                "actions": actions,
                "capture": take_capture(env),
                "failure": _failure_snapshot(env.sim, phase="slip", code=FAIL_SLIP_CONTACT, last_action=a),
                "disp_along_mm": last_disp,
                "disp_total_mm": last_disp,
                "errors_entry": err0,
                "contract": "goal_driven",
            }
        if guard_fingertip_margin:
            marg = fingertip_overlap_margin_mm(env.sim, U_OPEN_HAND)
            if marg < min_margin_mm:
                stop_reason = STOP_CONTACT_SAFETY
                break
        if guard_lateral and abs(float(s2["e_signed_mm"][0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            stop_reason = STOP_WRONG_DIRECTION
            break
        if (i + 1) % 5 == 0:
            series.append(
                {
                    "t": (i + 1) * DT,
                    "disp_mm": last_disp,
                    "e_signed_mm": s2["e_signed_mm"],
                    "e_pos_mm": float(measure_errors(env.sim)["e_pos_mm"]),
                    "fn_tot": s2["fn_tot"],
                    "v_rel_norm": s2["v_rel_norm"],
                }
            )
    else:
        stop_reason = STOP_TIME_BUDGET

    disp_at_stop = _disp()
    # Brake + regrip
    ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
    ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
    for j in range(int(0.45 / DT) + int(0.25 / DT)):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        if not sample_state(env.sim)["bilateral"]:
            for _ in range(8):
                a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                step_legal(env.sim, a_em)
                actions.append(a_em.copy())
                if sample_state(env.sim)["bilateral"]:
                    break
            if not sample_state(env.sim)["bilateral"]:
                code = FAIL_BRAKE if j < int(0.45 / DT) else FAIL_REGRIP
                return {
                    "ok": False,
                    "failure_code": code,
                    "stop_reason": STOP_CONTACT_SAFETY,
                    "reason": "loss_on_brake",
                    "actions": actions,
                    "capture": take_capture(env),
                    "disp_along_mm": disp_at_stop,
                    "disp_total_mm": _disp(),
                    "disp_before_brake_mm": disp_at_stop,
                    "errors_entry": err0,
                    "contract": "goal_driven",
                }

    disp_after = _disp()
    err_after = measure_errors(env.sim)
    hit_after, why_after = evaluate_slip_goal(err_after, goal)

    # Bounded post-brake nudge if braking pushed us out of the target region
    nudged = False
    if (
        allow_post_brake_nudge
        and goal_hit_pre_brake
        and not hit_after
        and err_after["bilateral"]
        and stop_reason in (STOP_GOAL_REACHED, STOP_HANDOFF_READY)
        and disp_after < float(goal.max_safe_travel_mm) - 0.5
    ):
        nudged = True
        ctl2 = DualLoopController(
            force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
            slip=SlipOuterLoop(vx_target=vx_target * 0.7, fn_min=fn_min, fn_max=fn_max, kn=kn),
            use_slip=True,
        )
        ctl2.reset(a0=0.65, fn0=sample_state(env.sim)["fn_tot"], fn_target=fn_init, slip_axis=tuple((-d).tolist()))
        for _ in range(int(1.2 / DT)):
            err = measure_errors(env.sim)
            hit, why = evaluate_slip_goal(err, goal)
            if hit:
                stop_reason = why or STOP_BRAKE_AND_REASSESS
                break
            if _disp() >= float(goal.max_safe_travel_mm):
                break
            s = sample_state(env.sim)
            _, meta = ctl2.step(s)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
            step_legal(env.sim, a)
            actions.append(a.copy())
            if not sample_state(env.sim)["bilateral"]:
                break
        # Short re-brake
        for _ in range(int(0.25 / DT)):
            st = sample_state(env.sim)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 0.95)
            step_legal(env.sim, a)
            actions.append(a.copy())
        disp_after = _disp()
        err_after = measure_errors(env.sim)
        hit_after, why_after = evaluate_slip_goal(err_after, goal)
        if hit_after:
            stop_reason = why_after or STOP_BRAKE_AND_REASSESS

    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    err_final = measure_errors(env.sim)
    hit_final, why_final = evaluate_slip_goal(err_final, goal)
    disp_final = _disp()

    # Success = contact retained and (goal after brake OR meaningful axis improvement without safety abort)
    axis_final = float(err_final["e_signed_mm"][idx])
    improved = abs(axis_final) < abs(e_axis0) - 0.3
    safe_stop = stop_reason not in (STOP_CONTACT_SAFETY, STOP_WRONG_DIRECTION)
    ok = bool(err_final["bilateral"] and safe_stop and (hit_final or improved or stop_reason == STOP_MAX_SAFE_TRAVEL and improved))
    if hit_final:
        stop_reason = why_final or stop_reason

    return {
        "ok": ok,
        "stop_reason": stop_reason,
        "goal_reached": bool(hit_final),
        "goal_reached_pre_brake": bool(goal_hit_pre_brake),
        "post_brake_nudge": nudged,
        "failure_code": None if ok else (FAIL_SLIP_TARGET if safe_stop else FAIL_SLIP_CONTACT),
        "disp_along_mm": disp_final,
        "disp_total_mm": disp_final,
        "disp_before_brake_mm": disp_at_stop,
        "disp_net_after_brake_mm": disp_after,
        "e_signed": err_final["e_signed_mm"],
        "errors_entry": err0,
        "errors": err_final,
        "actions": actions,
        "capture": take_capture(env),
        "reached": bool(hit_final or goal_hit_pre_brake),
        "brake_ok": True,
        "regrip_ok": True,
        "direction": d.tolist(),
        "series": series,
        "contract": "goal_driven",
        "goal": {
            "axis": goal.axis,
            "axis_tol_mm": goal.axis_tol_mm,
            "max_safe_travel_mm": goal.max_safe_travel_mm,
            "working_pos_mm": goal.working_pos_mm,
        },
    }


def audit_action_capture_replay(
    env,
    start_capture,
    actions: list | None,
    terminal_capture,
    *,
    atol_qpos: float = 1e-3,
    atol_qvel: float = 2e-2,
    atol_pose: float = 1e-3,
    atol_time: float = 1e-6,
    atol_err_mm: float = 0.25,
    atol_theta_deg: float = 0.5,
) -> dict[str, Any]:
    """Replay accepted action prefix from start_capture; compare to terminal_capture.

    Tolerances match deterministic MuJoCo replay with minor contact drift — not
    SUCCESS_* label equality.
    """
    if terminal_capture is None:
        return {"ok": False, "reason": "no_terminal_capture", "n_actions": len(actions or [])}
    acts = list(actions or [])
    restore_capture(env, start_capture)
    t0 = float(env.sim.data.time)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    ph_exp = terminal_capture["physics"]
    qpos_err = float(np.max(np.abs(np.asarray(env.sim.data.qpos, float) - np.asarray(ph_exp["qpos"], float))))
    qvel_err = float(np.max(np.abs(np.asarray(env.sim.data.qvel, float) - np.asarray(ph_exp["qvel"], float))))
    # Object free-joint pose: last 7 qpos of object body — use xpos/xquat for robustness
    oid = env.sim.ids.object_body
    pos_got = np.asarray(env.sim.data.xpos[oid], float)
    # Terminal pose from restored capture comparison
    restore_capture(env, terminal_capture)
    pos_exp = np.asarray(env.sim.data.xpos[oid], float)
    err_exp = measure_errors(env.sim)
    # Re-replay to get terminal errors without clobbering comparison state
    restore_capture(env, start_capture)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    err_got = measure_errors(env.sim)
    pos_err = float(np.linalg.norm(pos_got - pos_exp))
    time_err = abs(float(env.sim.data.time) - float(ph_exp["time"]))
    e_got = np.asarray(err_got["e_signed_mm"], float)
    e_exp = np.asarray(err_exp["e_signed_mm"], float)
    e_err = float(np.max(np.abs(e_got - e_exp)))
    th_err = abs(float(err_got["dtheta_deg"]) - float(err_exp["dtheta_deg"]))
    mismatches = []
    if qpos_err > atol_qpos:
        mismatches.append({"field": "qpos", "max_abs": qpos_err, "atol": atol_qpos})
    if qvel_err > atol_qvel:
        mismatches.append({"field": "qvel", "max_abs": qvel_err, "atol": atol_qvel})
    if pos_err > atol_pose:
        mismatches.append({"field": "object_xpos", "err": pos_err, "atol": atol_pose})
    if time_err > atol_time + DT * 0.5:
        # Allow half-step slack only if action count implies time; else flag
        expected_dt = len(acts) * DT
        if abs((float(env.sim.data.time) - t0) - expected_dt) > atol_time + 1e-9:
            mismatches.append({"field": "sim_time", "err": time_err, "atol": atol_time})
    if e_err > atol_err_mm:
        mismatches.append({"field": "e_signed_mm", "max_abs": e_err, "atol": atol_err_mm})
    if th_err > atol_theta_deg:
        mismatches.append({"field": "dtheta_deg", "err": th_err, "atol": atol_theta_deg})
    if bool(err_got["bilateral"]) != bool(err_exp["bilateral"]):
        mismatches.append({"field": "bilateral", "got": err_got["bilateral"], "exp": err_exp["bilateral"]})
    return {
        "ok": len(mismatches) == 0,
        "n_actions": len(acts),
        "qpos_max_abs": qpos_err,
        "qvel_max_abs": qvel_err,
        "object_xpos_err": pos_err,
        "sim_time_err": time_err,
        "e_signed_replay": e_got.tolist(),
        "e_signed_terminal": e_exp.tolist(),
        "dtheta_replay": float(err_got["dtheta_deg"]),
        "dtheta_terminal": float(err_exp["dtheta_deg"]),
        "bilateral_replay": bool(err_got["bilateral"]),
        "mismatches": mismatches,
    }


def run_staged_axial_slip(
    env,
    blob,
    direction: np.ndarray | list[float],
    *,
    milestone_mm: float = 5.0,
    z_ref: float | None = None,
    e_x0: float | None = None,
    guard_lateral: bool = True,
    guard_fingertip_margin: bool = False,
    min_margin_mm: float = FINGERTIP_MARGIN_MIN_MM,
    fn_min: float = FN_MIN,
    fn_max: float = FN_MAX,
    kn: float = SLIP_KN,
    vx_target: float = V_SLIP,
    fn_init: float = FN_INIT,
    a0_slip: float = A0_SLIP,
) -> dict[str, Any]:
    """Shared staged DualLoop axial slip with explicit total-displacement contract.

    `milestone_mm` is interpreted as ``target_total_mm``: net slip along +d from the
    original stage-entry relative pose (not an additional stage-local request).
    Stage 1 calibrates at ``min(2, target)``; stage 2 commands only the remaining
    distance after measured net displacement (including brake drift).

    `direction` is desired cylinder COM motion in the hand frame (unitized).
    Inward: −u_open; outward: +u_open. Same controller architecture; optional
    firmer bands for outward contact safety.
    """
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {
            "ok": False,
            "failure_code": FAIL_SLIP_TARGET,
            "reason": "zero_direction",
            "actions": [],
            "capture": take_capture(env),
        }
    d = d / n
    restore_capture(env, blob["capture"])
    if z_ref is None:
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    if e_x0 is None:
        e_x0 = float(sample_state(env.sim)["e_signed_mm"][0])

    def _one(blob_i, milestone, e_x):
        restore_capture(env, blob_i["capture"])
        s0 = sample_state(env.sim)
        p0 = np.asarray(s0["p_c_h"], float)
        ctl = DualLoopController(
            force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
            slip=SlipOuterLoop(vx_target=vx_target, fn_min=fn_min, fn_max=fn_max, kn=kn),
            use_slip=True,
        )
        ctl.reset(a0=a0_slip, fn0=s0["fn_tot"], fn_target=fn_init, slip_axis=tuple((-d).tolist()))
        actions = []
        lost = 0
        reached = False
        abort = None
        fail_code = None
        last_disp = 0.0
        for _ in range(int(12.0 / DT)):
            s = sample_state(env.sim)
            _, meta = ctl.step(s)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
            step_legal(env.sim, a)
            actions.append(a.copy())
            failed, lost = recovery_failed(env.sim, blob_i["ref"], lost)
            s2 = sample_state(env.sim)
            disp = float(np.dot(np.asarray(s2["p_c_h"], float) - p0, d)) * 1e3
            last_disp = disp
            if failed or latched(env.sim) or not s2["bilateral"]:
                return {
                    "ok": False,
                    "failure_code": FAIL_SLIP_CONTACT,
                    "reason": "contact_loss",
                    "actions": actions,
                    "capture": take_capture(env),
                    "failure": _failure_snapshot(env.sim, phase="slip", code=FAIL_SLIP_CONTACT, last_action=a),
                    "reached": False,
                    "brake_ok": False,
                    "regrip_ok": False,
                }
            if guard_fingertip_margin:
                marg = fingertip_overlap_margin_mm(env.sim, U_OPEN_HAND)
                if marg < min_margin_mm:
                    abort = "fingertip_margin"
                    fail_code = FAIL_SLIP_TARGET
                    break
            if guard_lateral and abs(float(s2["e_signed_mm"][0]) - e_x) > LATERAL_DRIFT_ABORT_MM:
                abort = "lateral_drift"
                fail_code = FAIL_SLIP_WRONG_DIR
                break
            if disp < -0.5:
                abort = "wrong_direction"
                fail_code = FAIL_SLIP_WRONG_DIR
                break
            if disp >= milestone:
                reached = True
                break
        # Firm brake: raise grip quickly (da_max higher) to avoid unloading into loss
        ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
        ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
        for j in range(int(0.45 / DT) + int(0.25 / DT)):
            s = sample_state(env.sim)
            a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
            step_legal(env.sim, a)
            actions.append(a.copy())
            if not sample_state(env.sim)["bilateral"]:
                # One emergency firm-up burst
                for _ in range(8):
                    a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                    step_legal(env.sim, a_em)
                    actions.append(a_em.copy())
                    if sample_state(env.sim)["bilateral"]:
                        break
                if not sample_state(env.sim)["bilateral"]:
                    code = FAIL_BRAKE if j < int(0.45 / DT) else FAIL_REGRIP
                    return {
                        "ok": False,
                        "failure_code": code,
                        "reason": "loss_on_brake",
                        "actions": actions,
                        "capture": take_capture(env),
                        "reached": reached,
                        "brake_ok": False,
                        "regrip_ok": False,
                        "disp": last_disp,
                        "failure": _failure_snapshot(env.sim, phase="brake", code=code, last_action=a),
                    }
        # Stage-local displacement from this stage's p0.
        # Caller interprets cumulative totals via target_total_mm contract.
        ok = bool(abort is None and (reached or last_disp >= 0.5 * milestone))
        net_after = float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3
        return {
            "ok": ok,
            "failure_code": None if ok else (fail_code or FAIL_SLIP_TARGET),
            "disp": last_disp,
            "disp_before_brake_mm": last_disp,
            "disp_net_after_brake_mm": net_after,
            "e_signed": sample_state(env.sim)["e_signed_mm"],
            "actions": actions,
            "capture": take_capture(env),
            "abort_reason": abort,
            "reached": reached,
            "brake_ok": True,
            "regrip_ok": True,
        }

    # Explicit total-displacement contract:
    #   target_total_mm = intended net slip along +d from the original stage-entry pose.
    # Stage 1 calibrates at min(2 mm, target). Stage 2 requests only the remaining
    # distance after accounting for net displacement achieved (incl. brake drift).
    target_total = float(milestone_mm)
    p_entry = np.asarray(sample_state(env.sim)["p_c_h"], float).copy()
    stage_log: list[dict] = []

    def _cum_from_entry() -> float:
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p_entry, d)) * 1e3

    cal_mm = min(2.0, target_total)
    s_cal = _one(blob, cal_mm, e_x0)
    stage_log.append({
        "stage": 1,
        "commanded_mm": cal_mm,
        "disp_before_brake_mm": s_cal.get("disp_before_brake_mm", s_cal.get("disp")),
        "disp_net_after_brake_mm": s_cal.get("disp_net_after_brake_mm", s_cal.get("disp")),
        "ok": bool(s_cal.get("ok")),
        "abort_reason": s_cal.get("abort_reason"),
    })
    if not s_cal.get("ok"):
        return {
            **s_cal,
            "target_total_mm": target_total,
            "disp_total_mm": s_cal.get("disp"),
            "remaining_mm": target_total - float(s_cal.get("disp") or 0.0),
            "staged_log": stage_log,
            "contract": "target_total_mm",
        }

    restore_capture(env, s_cal["capture"])
    cum_after_cal = _cum_from_entry()
    remaining = max(0.0, target_total - cum_after_cal)
    all_actions = list(s_cal["actions"])
    last = s_cal

    if remaining > 0.15:
        s_rem = _one(
            {"capture": s_cal["capture"], "ref": blob["ref"]},
            remaining,
            float(s_cal["e_signed"][0]),
        )
        stage_log.append({
            "stage": 2,
            "commanded_mm": remaining,
            "disp_before_brake_mm": s_rem.get("disp_before_brake_mm", s_rem.get("disp")),
            "disp_net_after_brake_mm": s_rem.get("disp_net_after_brake_mm", s_rem.get("disp")),
            "ok": bool(s_rem.get("ok")),
            "abort_reason": s_rem.get("abort_reason"),
            "cum_before_stage_mm": cum_after_cal,
        })
        all_actions = all_actions + list(s_rem.get("actions") or [])
        if not s_rem.get("ok"):
            restore_capture(env, s_rem.get("capture") or s_cal["capture"])
            cum_fail = _cum_from_entry()
            return {
                **s_rem,
                "actions": all_actions,
                "disp_2": s_cal.get("disp"),
                "disp_total_mm": cum_fail,
                "target_total_mm": target_total,
                "remaining_mm": target_total - cum_fail,
                "staged_log": stage_log,
                "contract": "target_total_mm",
                "overshoot_mm": max(0.0, cum_fail - target_total),
            }
        last = s_rem

    restore_capture(env, last["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    extra = []
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        extra.append(a.copy())
    cum_final = _cum_from_entry()
    return {
        "ok": True,
        "disp_along_mm": cum_final,
        "disp_total_mm": cum_final,
        "disp_2": s_cal.get("disp"),
        "target_total_mm": target_total,
        "remaining_mm": target_total - cum_final,
        "overshoot_mm": max(0.0, cum_final - target_total),
        "backward_drift_mm": max(0.0, -cum_final) if cum_final < 0 else 0.0,
        "e_signed": sample_state(env.sim)["e_signed_mm"],
        "actions": all_actions + extra,
        "capture": take_capture(env),
        "reached": True,
        "brake_ok": True,
        "regrip_ok": True,
        "direction": d.tolist(),
        "staged_log": stage_log,
        "contract": "target_total_mm",
    }


def _inward_slip_staged(
    env, blob, z_ref: float, e_x0: float, *, milestone_mm: float = 12.0, cfg: PrecisionConfig | None = None
) -> dict[str, Any]:
    """Inward (−u_open) DualLoop — goal-driven when cfg.use_goal_driven_slip (default)."""
    cfg = cfg or PrecisionConfig()
    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        # Tipped-up e_pos is not predictive of post-restore SUCCESS_NEW; drive |dz|
        # into a Euclidean-compatible band estimated from the pre-slip lateral errors.
        restore_capture(env, blob["capture"])
        e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
        wp = float(cfg.working_pos_mm)
        budget_sq = wp * wp - float(e0[0]) ** 2 - float(e0[1]) ** 2
        axis_tol = float(np.sqrt(max(budget_sq, 0.0))) if budget_sq > 0 else 3.0
        axis_tol = float(np.clip(axis_tol, 3.0, 8.0))
        goal = SlipGoal(
            axis="dz",
            desired_direction=-np.asarray(U_OPEN_HAND, float),
            max_safe_travel_mm=float(milestone_mm),
            max_duration_s=12.0,
            axis_tol_mm=axis_tol,
            working_pos_mm=wp,
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
            use_euclidean_stop=False,
            brake_margin_mm=1.0,
        )
        return run_goal_aware_axial_slip(
            env, blob, goal, z_ref=z_ref, guard_lateral=True, guard_fingertip_margin=False,
        )
    return run_staged_axial_slip(
        env, blob, -np.asarray(U_OPEN_HAND, float),
        milestone_mm=milestone_mm, z_ref=z_ref, e_x0=e_x0, guard_lateral=True, guard_fingertip_margin=False,
    )


def _outward_slip_staged(
    env, blob, *, z_ref: float, e_x0: float, milestone_mm: float = 12.0, cfg: PrecisionConfig | None = None
) -> dict[str, Any]:
    """Outward (+u_open) DualLoop — goal-driven when cfg.use_goal_driven_slip (default)."""
    cfg = cfg or PrecisionConfig()
    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        goal = SlipGoal(
            axis="dz",
            desired_direction=np.asarray(U_OPEN_HAND, float),
            max_safe_travel_mm=float(milestone_mm),
            max_duration_s=12.0,
            axis_tol_mm=float(cfg.ez_tol_mm),
            working_pos_mm=float(cfg.working_pos_mm),
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
        )
        return run_goal_aware_axial_slip(
            env,
            blob,
            goal,
            z_ref=z_ref,
            guard_lateral=True,
            guard_fingertip_margin=True,
            min_margin_mm=FINGERTIP_MARGIN_MIN_MM,
            fn_min=5.0,
            fn_max=12.0,
            kn=500.0,
            vx_target=-0.0007,
            fn_init=7.0,
            a0_slip=0.70,
        )
    return run_staged_axial_slip(
        env,
        blob,
        np.asarray(U_OPEN_HAND, float),
        milestone_mm=milestone_mm,
        z_ref=z_ref,
        e_x0=e_x0,
        guard_lateral=True,
        guard_fingertip_margin=True,
        min_margin_mm=FINGERTIP_MARGIN_MIN_MM,
        fn_min=5.0,
        fn_max=12.0,
        kn=500.0,
        vx_target=-0.0007,
        fn_init=7.0,
        a0_slip=0.70,
    )


def correct_dx_horizontal_attempt(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """Full horizontal-gripper dx: P0–P9 including P3/P6 sliding rolls."""
    phase_log: list[dict] = []
    actions: list = []
    restore_capture(env, blob["capture"])
    y0 = reference_pad_vector(env.sim)
    err0 = measure_errors(env.sim)
    e0 = np.asarray(err0["e_signed_mm"], float)
    th0 = float(err0["dtheta_deg"])
    # Skip unnecessary manipulation when geometry is already SUCCESS_NEW-compatible.
    if handoff_geometry_ready(err0, cfg):
        return {
            "ok": True,
            "skipped": True,
            "reason": STOP_HANDOFF_READY,
            "stop_reason": STOP_HANDOFF_READY,
            "goal_reached": True,
            "partial_progress": False,
            "dx_solved": abs(float(e0[0])) <= float(cfg.ex_tol_mm),
            "actions": [],
            "capture": blob["capture"],
            "errors": err0,
            "errors_entry": {"e_signed_mm": e0.tolist(), "dtheta_deg": th0, "e_pos_mm": float(err0["e_pos_mm"])},
            "disp_total_mm": 0.0,
            "contract": "goal_driven",
        }
    d_lat = lateral_desire_direction(e0)
    if float(np.linalg.norm(d_lat)) < 0.5:
        return {"ok": True, "skipped": True, "reason": "dx_near_zero", "actions": [], "capture": blob["capture"]}

    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    e_ref = e0.copy()

    def _fail(code, phase, cur_cap, extra=None):
        restore_capture(env, cur_cap)
        extra = dict(extra or {})
        phys = extra.get("physical_reason")
        out = {
            "ok": False,
            "mechanism": "horizontal_dx",
            "failure_code": code,
            "phase": phase,
            "reason": phys or code,
            "physical_reason": phys,
            "actions": list(actions),
            "capture": take_capture(env),
            "phase_log": phase_log,
            "failure": _failure_snapshot(env.sim, phase=phase, code=code, last_action=actions[-1] if actions else None),
            "n_accepted_actions": len(actions),
            "rejected_branch": bool(extra.get("rejected_branch", False)),
            "in_selected_trajectory": bool(extra.get("in_selected_trajectory", True)),
        }
        out.update(extra)
        return out

    # P0: fingertips-down
    if not gates(env.sim)["fingertips_down"]:
        p0 = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dxH_P0", rate_scale=0.20, fn_hold=FN_HOLD,
            ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=False, max_time_s=8.0,
        )
        actions += p0.get("actions") or []
        phase_log.append({"phase": "P0", "ok": p0.get("ok"), "errors": measure_errors(env.sim)})
        if not p0.get("ok"):
            return _fail(
                FAIL_PREPARATION, "P0", p0.get("capture") or cur["capture"],
                {"physical_reason": classify_tilt_physical_reason(p0)},
            )
        cur = {"capture": p0["capture"], "ref": blob["ref"]}
    else:
        phase_log.append({"phase": "P0", "ok": True, "skipped": True, "errors": measure_errors(env.sim)})

    # P1 + P2: support roll then tilt to horizontal
    roll_sign = float(choose_roll_sign_for_support(env.sim))
    support_roll_deg = 55.0
    p2_rate, p2_boost, p2_tol = 0.12, 0.05, 10.0

    if bool(getattr(cfg, "use_p1_p2_search", False)):
        wp = getattr(cfg, "p1_p2_winner", None)
        search = None
        if not isinstance(wp, dict) or not wp:
            search = search_p1_p2_horizontal(env, cur)
            phase_log.append(
                {
                    "phase": "P1_P2_search",
                    "ok": search.get("ok"),
                    "best": search.get("best"),
                    "n_trials": search.get("n_trials"),
                    "limiting": search.get("limiting"),
                    "counterfactual": True,
                    "in_selected_trajectory": False,
                }
            )
            if not search.get("ok"):
                return _fail(
                    FAIL_HORIZONTAL_TILT,
                    "P2",
                    cur["capture"],
                    {
                        "physical_reason": search.get("physical_reason") or search.get("limiting"),
                        "P1_P2_search": {
                            k: search[k]
                            for k in ("ok", "best", "n_trials", "limiting", "limiting_counts", "trials")
                            if k in search
                        },
                    },
                )
            wp = search["winner_params"]
        else:
            phase_log.append(
                {
                    "phase": "P1_P2_search",
                    "ok": True,
                    "best": wp,
                    "reused_winner": True,
                    "counterfactual": True,
                    "in_selected_trajectory": False,
                }
            )
        roll_sign = float(wp["support_roll_sign"])
        support_roll_deg = float(wp["support_roll_deg"])
        p2_rate = float(wp["p2_rate"])
        p2_boost = float(wp["z_boost"])
        p2_tol = float(wp.get("ang_tol", 12.0))
        restore_capture(env, cur["capture"])
        e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
        y0 = reference_pad_vector(env.sim)

    if support_roll_deg > 0.5:
        p1 = run_roll_phase(
            env, cur, sign=roll_sign, angle_deg=support_roll_deg, phase="dxH_P1",
            rate_scale=0.12, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
        )
        actions += p1.get("actions") or []
        phase_log.append({
            "phase": "P1", "ok": p1.get("ok"), "angle_deg": support_roll_deg,
            "sign": roll_sign, "errors": measure_errors(env.sim),
            "in_selected_trajectory": True,
        })
        if not p1.get("ok"):
            return _fail(
                FAIL_SUPPORT_ROLL, "P1", p1.get("capture") or cur["capture"],
                {"physical_reason": p1.get("reason")},
            )
        cur = {"capture": p1["capture"], "ref": blob["ref"]}
        bil1, sup1 = _physical_finger_support(env.sim)
        phase_log.append({"phase": "P1_support", "ok": bil1, "support": sup1, "in_selected_trajectory": True})
        if not bil1:
            return _fail(
                FAIL_SUPPORT_ROLL, "P1_support", cur["capture"],
                {"physical_reason": "loss_of_bilateral_after_P1", "support": sup1},
            )
    else:
        phase_log.append({"phase": "P1", "ok": True, "skipped": True, "angle_deg": 0.0, "in_selected_trajectory": True})

    restore_capture(env, cur["capture"])
    u_tgt = _horizontal_open_target(env.sim)
    p2 = run_tilt_phase(
        env, cur, target_dir_w=u_tgt, phase="dxH_P2", rate_scale=p2_rate, fn_hold=FN_HOLD,
        ang_tol_deg=p2_tol, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
        z_ref_boost=p2_boost, height_kp=4.5,
    )
    if not p2.get("ok") and not bool(getattr(cfg, "use_p1_p2_search", False)):
        rejected_p2 = {
            "reason": p2.get("reason"),
            "physical_reason": classify_tilt_physical_reason(p2),
            "counterfactual": True,
            "in_selected_trajectory": False,
            "n_actions_rejected": len(p2.get("actions") or []),
        }
        phase_log.append({"phase": "P2_attempt", "ok": False, **rejected_p2})
        restore_capture(env, cur["capture"])
        p2b = run_tilt_phase(
            env, cur, target_dir_w=u_tgt, phase="dxH_P2_retry", rate_scale=0.08, fn_hold=FN_HOLD,
            ang_tol_deg=12.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=18.0,
            z_ref_boost=0.06, height_kp=5.0,
        )
        if p2b.get("ok"):
            p2 = p2b
        else:
            phase_log.append({
                "phase": "P2", "ok": False, "alpha": gates(env.sim)["alpha_up_deg"],
                "retry_failed": True, "physical_reason": classify_tilt_physical_reason(p2b),
                "in_selected_trajectory": True,
            })
            return _fail(
                FAIL_HORIZONTAL_TILT, "P2", p2b.get("capture") or p2.get("capture") or cur["capture"],
                {"physical_reason": classify_tilt_physical_reason(p2b), "first_attempt": rejected_p2},
            )
    elif not p2.get("ok"):
        actions += p2.get("actions") or []
        phase_log.append({
            "phase": "P2", "ok": False, "alpha": gates(env.sim)["alpha_up_deg"],
            "physical_reason": classify_tilt_physical_reason(p2),
            "in_selected_trajectory": True,
        })
        return _fail(
            FAIL_HORIZONTAL_TILT, "P2", p2.get("capture") or cur["capture"],
            {"physical_reason": classify_tilt_physical_reason(p2)},
        )
    actions += p2.get("actions") or []
    bil2, sup2 = _physical_finger_support(env.sim)
    phase_log.append({
        "phase": "P2", "ok": p2.get("ok"), "alpha": gates(env.sim)["alpha_up_deg"],
        "errors": measure_errors(env.sim), "support": sup2, "in_selected_trajectory": True,
    })
    if not p2.get("ok") or not bil2:
        return _fail(
            FAIL_HORIZONTAL_TILT, "P2", p2.get("capture") or cur["capture"],
            {
                "physical_reason": "true_contact_loss" if not bil2 else classify_tilt_physical_reason(p2),
                "support": sup2,
            },
        )
    cur = {"capture": p2["capture"], "ref": blob["ref"]}

    # P3: select + execute sliding roll (counterfactual search; only winner enters tape)
    sel = select_horizontal_sliding_roll(env, cur, d_lat, y0, e_ref)
    if not sel.get("ok"):
        return _fail(FAIL_SLIDING_ROLL, "P3_select", cur["capture"], {"P3_search": sel})
    p3 = run_roll_phase(
        env, cur, sign=float(sel["sign"]), angle_deg=float(sel["angle_deg"]), phase="dxH_P3",
        rate_scale=0.12, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
    )
    actions += p3.get("actions") or []
    phase_log.append({"phase": "P3", "ok": p3.get("ok"), "sign": sel["sign"], "angle_deg": sel["angle_deg"], "errors": measure_errors(env.sim)})
    if not p3.get("ok"):
        return _fail(FAIL_SLIDING_ROLL, "P3", p3.get("capture") or cur["capture"], {"P3_search": sel})
    cur = {"capture": p3["capture"], "ref": blob["ref"]}
    restore_capture(env, cur["capture"])
    sc = _score_horizontal_slide_config(env.sim, d_lat)
    if not sc["ok"]:
        return _fail(FAIL_SLIDING_ROLL, "P3_geometry", cur["capture"], {"geometry": sc, "P3_search": sel})

    # P4–P5: goal-driven DualLoop lateral (max_safe_travel only; stop on dx / e_pos region)
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    max_travel = float(getattr(cfg, "horizontal_slip_mm", 14.0) or 14.0)
    if bool(getattr(cfg, "use_goal_driven_slip", True)):
        slip_goal = SlipGoal(
            axis="dx",
            desired_direction=d_lat,
            max_safe_travel_mm=max_travel,
            max_duration_s=12.0,
            axis_tol_mm=float(cfg.ex_tol_mm),
            working_pos_mm=float(cfg.working_pos_mm),
            working_theta_deg=float(cfg.working_theta_deg),
            ey_tol_mm=float(cfg.ey_tol_mm),
        )
        slip = run_goal_aware_axial_slip(
            env, cur, slip_goal, z_ref=z_ref, guard_lateral=False, guard_fingertip_margin=False,
        )
    else:
        milestone = max_travel
        if bool(getattr(cfg, "horizontal_legacy_total", False)):
            milestone = 2.0 + max_travel
        slip = run_staged_axial_slip(
            env, cur, d_lat, milestone_mm=milestone, z_ref=z_ref,
            e_x0=None, guard_lateral=False, guard_fingertip_margin=False,
        )
    actions += slip.get("actions") or []
    phase_log.append(
        {
            "phase": "P4_P5",
            "ok": slip.get("ok"),
            "disp_along_mm": slip.get("disp_along_mm"),
            "disp_total_mm": slip.get("disp_total_mm"),
            "stop_reason": slip.get("stop_reason"),
            "goal_reached": slip.get("goal_reached"),
            "contract": slip.get("contract"),
            "errors": measure_errors(env.sim) if slip.get("capture") else None,
            "failure_code": slip.get("failure_code"),
            "in_selected_trajectory": True,
        }
    )
    if not slip.get("ok"):
        return _fail(
            slip.get("failure_code") or FAIL_SLIP_CONTACT,
            "P4_P5",
            slip.get("capture") or cur["capture"],
            {
                "physical_reason": slip.get("reason") or slip.get("abort_reason") or slip.get("failure_code"),
                "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            },
        )
    # Coupling check — warn, but only abort on severe loss of grasp geometry
    e_post = np.asarray(slip["e_signed"], float)
    th_post = float(measure_errors(env.sim)["dtheta_deg"])
    if abs(float(e_post[2]) - float(e0[2])) > 8.0 and abs(float(e_post[2])) > abs(float(e0[2])) + 4.0:
        return _fail(
            FAIL_SLIP_WRONG_DIR, "P4_P5_dz_regression", slip["capture"],
            {"e0": e0.tolist(), "e_post": e_post.tolist()},
        )
    if th_post > 15.0 and (th_post - th0) > 10.0:
        return _fail(FAIL_SLIP_WRONG_DIR, "P4_P5_theta_regression", slip["capture"], {"th0": th0, "th_post": th_post})
    if th_post > 10.0:
        phase_log.append({"phase": "P4_P5_theta_warn", "th0": th0, "th_post": th_post})
    cur = {"capture": slip["capture"], "ref": blob["ref"]}
    e_ref = e_post.copy()

    # P6: reverse P3 sliding roll (still near-horizontal)
    p6 = run_roll_phase(
        env, cur, sign=-float(sel["sign"]), angle_deg=float(sel["angle_deg"]), phase="dxH_P6",
        rate_scale=0.12, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
    )
    actions += p6.get("actions") or []
    phase_log.append({"phase": "P6", "ok": p6.get("ok"), "errors": measure_errors(env.sim)})
    if not p6.get("ok"):
        return _fail(FAIL_REVERSE_ROLL, "P6", p6.get("capture") or cur["capture"])
    cur = {"capture": p6["capture"], "ref": blob["ref"]}

    # P7: tilt back to fingertips-down (counterfactual retry; rejected attempt not concatenated)
    p7 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="dxH_P7", rate_scale=0.16, fn_hold=FN_HOLD,
        ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=14.0,
        z_ref_boost=0.03, height_kp=4.0,
    )
    if not p7.get("ok"):
        phase_log.append({
            "phase": "P7_attempt",
            "ok": False,
            "physical_reason": classify_tilt_physical_reason(p7),
            "counterfactual": True,
            "in_selected_trajectory": False,
            "n_actions_rejected": len(p7.get("actions") or []),
        })
        restore_capture(env, cur["capture"])
        p7b = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dxH_P7_retry", rate_scale=0.10, fn_hold=FN_HOLD,
            ang_tol_deg=7.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
            z_ref_boost=0.06, height_kp=5.0,
        )
        if p7b.get("ok"):
            p7 = p7b
        else:
            # Live failure after retry — keep retry actions as the executed failure prefix
            actions += p7b.get("actions") or []
            phase_log.append({
                "phase": "P7", "ok": False, "errors": measure_errors(env.sim),
                "physical_reason": classify_tilt_physical_reason(p7b),
                "in_selected_trajectory": True,
            })
            return _fail(
                FAIL_RETURN_TILT, "P7", p7b.get("capture") or cur["capture"],
                {"physical_reason": classify_tilt_physical_reason(p7b)},
            )
    actions += p7.get("actions") or []
    phase_log.append({"phase": "P7", "ok": p7.get("ok"), "errors": measure_errors(env.sim), "in_selected_trajectory": True})
    if not p7.get("ok"):
        return _fail(
            FAIL_RETURN_TILT, "P7", p7.get("capture") or cur["capture"],
            {"physical_reason": classify_tilt_physical_reason(p7)},
        )
    cur = {"capture": p7["capture"], "ref": blob["ref"]}

    # P8: reverse P1 support roll (require down)
    p8 = run_roll_phase(
        env, cur, sign=-roll_sign, angle_deg=support_roll_deg, phase="dxH_P8",
        rate_scale=0.12, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    actions += p8.get("actions") or []
    phase_log.append({"phase": "P8", "ok": p8.get("ok"), "errors": measure_errors(env.sim)})
    if not p8.get("ok"):
        return _fail(FAIL_FINAL_UNROLL, "P8", p8.get("capture") or cur["capture"])
    restore_capture(env, p8["capture"])
    settle_hold(env, actions, hold_s=0.4)
    err_f = measure_errors(env.sim)
    phase_log.append({"phase": "P9", "ok": True, "errors": err_f})
    # Safe full P0–P9 with meaningful progress (need not solve entire SUCCESS_NEW alone).
    dx_improved = abs(float(err_f["dx_mm"])) < abs(float(e0[0])) - 0.3
    e_pos0 = float(np.linalg.norm(e0))
    e_pos_f = float(err_f["e_pos_mm"])
    pos_improved = e_pos_f < e_pos0 - 0.3
    theta_ok = float(err_f["dtheta_deg"]) <= max(th0 + 8.0, 15.0)
    dz_ok = abs(float(err_f["dz_mm"])) <= abs(float(e0[2])) + 8.0
    partial_ok = bool(err_f["bilateral"] and theta_ok and dz_ok and (dx_improved or pos_improved))
    ok_h = partial_ok
    return {
        "ok": ok_h,
        "mechanism": "horizontal_dx",
        "failure_code": None if ok_h else FAIL_SLIP_TARGET,
        "partial_progress": bool(dx_improved or pos_improved),
        "dx_solved": abs(float(err_f["dx_mm"])) <= float(cfg.ex_tol_mm),
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
        "P3_search": {k: sel[k] for k in sel if k != "trials"},
        "P3_trials": sel.get("trials"),
        "geometry": sc,
        "disp_along_mm": slip.get("disp_along_mm"),
        "disp_total_mm": slip.get("disp_total_mm"),
        "contract": slip.get("contract"),
        "support_roll_sign": roll_sign,
        "support_roll_deg": support_roll_deg,
        "phase_log": phase_log,
        "actions": actions,
        "capture": take_capture(env),
        "errors": err_f,
        "errors_entry": {"e_signed_mm": e0.tolist(), "dtheta_deg": th0, "e_pos_mm": e_pos0},
        "rejected_branch": False,
        "n_accepted_actions": len(actions),
    }


def correct_dz_inward(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """Endpoint-only supported path: down → roll → up → DualLoop inward → reverse.

    Matches verified `handoff_state_optimization.rebuild_through_p5` (no IC P6).
    """
    actions: list = []
    restore_capture(env, blob["capture"])
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    roll_sign = choose_roll_sign_for_support(env.sim)

    p0 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="dz_P0", rate_scale=0.20, fn_hold=FN_HOLD,
        ang_tol_deg=4.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=False, max_time_s=8.0,
    )
    actions += p0.get("actions") or []
    if not (p0.get("ok") and gates(env.sim)["fingertips_down"]):
        return {"ok": False, "phase": "P0", "reason": p0.get("reason"), "actions": actions, "capture": blob["capture"]}
    cur = {"capture": p0["capture"], "ref": blob["ref"]}

    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=ROLL_ANGLE_DEG, phase="dz_P1",
        rate_scale=0.15, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    actions += p1.get("actions") or []
    if not p1.get("ok"):
        return {"ok": False, "phase": "P1", "reason": p1.get("reason"), "actions": actions, "capture": cur["capture"]}
    cur = {"capture": p1["capture"], "ref": blob["ref"]}

    p2 = run_tilt_phase(
        env, cur, target_dir_w=UP, phase="dz_P2", rate_scale=0.22, fn_hold=FN_HOLD,
        ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
    )
    actions += p2.get("actions") or []
    if not (p2.get("ok") and gates(env.sim)["fingertips_up"]):
        return {"ok": False, "phase": "P2", "reason": p2.get("reason"), "actions": actions, "capture": p2.get("capture") or cur["capture"]}
    cur = {"capture": p2["capture"], "ref": blob["ref"]}

    restore_capture(env, cur["capture"])
    hold = HoldModeGrip(fn_hold=14.0)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(int(0.4 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    cur = {"capture": take_capture(env), "ref": blob["ref"]}
    e_x0 = float(sample_state(env.sim)["e_signed_mm"][0])
    slip = _inward_slip_staged(
        env, cur, z_ref, e_x0, milestone_mm=float(cfg.inward_slip_mm), cfg=cfg,
    )
    actions += slip.get("actions") or []
    if not slip.get("ok"):
        return {
            "ok": False,
            "phase": "P3_slip",
            "reason": slip.get("reason") or slip.get("abort_reason") or slip.get("stop_reason") or "inward_slip_fail",
            "stop_reason": slip.get("stop_reason"),
            "actions": actions,
            "capture": slip.get("capture") or cur["capture"],
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        }
    cur = {"capture": slip["capture"], "ref": blob["ref"]}
    e_post = np.asarray(slip["e_signed"], float)

    p4 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="dz_P4", rate_scale=0.18, fn_hold=FN_HOLD,
        ang_tol_deg=4.0, y0_w=y0, e_ref=e_post, allow_roll=False, contact_aware=True, max_time_s=16.0,
        z_ref_boost=0.04, height_kp=4.0,
    )
    actions += p4.get("actions") or []
    if not p4.get("ok"):
        return {"ok": False, "phase": "P4", "reason": p4.get("reason"), "actions": actions, "capture": p4.get("capture") or cur["capture"]}
    cur = {"capture": p4["capture"], "ref": blob["ref"]}

    p5 = run_roll_phase(
        env, cur, sign=-roll_sign, angle_deg=ROLL_ANGLE_DEG, phase="dz_P5",
        rate_scale=0.15, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_post,
    )
    actions += p5.get("actions") or []
    if not p5.get("ok"):
        return {"ok": False, "phase": "P5", "reason": p5.get("reason"), "actions": actions, "capture": p5.get("capture") or cur["capture"]}
    restore_capture(env, p5["capture"])
    settle_hold(env, actions, hold_s=0.5)
    return {
        "ok": True,
        "mechanism": "endpoint_only_inward",
        "disp_along_mm": slip.get("disp_along_mm"),
        "disp_total_mm": slip.get("disp_total_mm"),
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
        "contract": slip.get("contract"),
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
        "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
    }


def correct_dz_outward(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """Outward axial slip: fingertips-down + shared staged DualLoop along +u_open.

    Does not tip to fingertips-up. Preparation failure ≠ slip failure.
    """
    actions: list = []
    phase_log: list[dict] = []
    restore_capture(env, blob["capture"])
    y0 = reference_pad_vector(env.sim)
    e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    d_out = np.asarray(U_OPEN_HAND, float)  # +u_open toward fingertips

    restore_capture(env, cur["capture"])
    s0 = sample_state(env.sim)
    g_sc = score_gravity_alignment(np.asarray(s0["g_h"], float), d_out)
    alpha0 = float(gates(env.sim)["alpha_up_deg"])
    # Prefer true fingertips-down for outward support. Skip only if already at the down gate.
    already_good = bool(gates(env.sim)["fingertips_down"])
    if already_good:
        phase_log.append(
            {
                "phase": "prep_down",
                "ok": True,
                "skipped": True,
                "reason": "already_fingertips_down",
                "alpha": alpha0,
                "g_score": g_sc,
            }
        )
    else:
        # Gentle contact-aware tip toward down (prep failure ≠ slip failure)
        p0 = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dzOut_P0", rate_scale=0.12, fn_hold=FN_HOLD,
            ang_tol_deg=6.0, y0_w=y0, e_ref=e0, allow_roll=False, contact_aware=True, max_time_s=14.0,
            z_ref_boost=0.05, height_kp=4.5,
        )
        if not p0.get("ok"):
            restore_capture(env, cur["capture"])
            p0 = run_tilt_phase(
                env, cur, target_dir_w=DOWN, phase="dzOut_P0_retry", rate_scale=0.10, fn_hold=FN_HOLD,
                ang_tol_deg=8.0, y0_w=y0, e_ref=e0, allow_roll=False, contact_aware=True, max_time_s=16.0,
                z_ref_boost=0.06, height_kp=5.0,
            )
        if p0.get("ok"):
            actions += p0.get("actions") or []
            cur = {"capture": p0["capture"], "ref": blob["ref"]}
            phase_log.append({"phase": "prep_down", "ok": True, "errors": measure_errors(env.sim)})
        else:
            phase_log.append({"phase": "prep_down", "ok": False, "errors": measure_errors(env.sim)})
            return {
                "ok": False,
                "mechanism": "dual_loop_outward",
                "failure_code": FAIL_PREPARATION,
                "phase": "prep_down",
                "reason": p0.get("reason") or FAIL_PREPARATION,
                "actions": actions,  # empty or prior only — failed prep actions not kept if retry from cur
                "capture": take_capture(env) if not p0.get("capture") else p0["capture"],
                "phase_log": phase_log,
                "failure": _failure_snapshot(env.sim, phase="prep_down", code=FAIL_PREPARATION),
            }

    restore_capture(env, cur["capture"])
    s0 = sample_state(env.sim)
    g_sc = score_gravity_alignment(np.asarray(s0["g_h"], float), d_out)
    phase_log.append({"phase": "gravity_check", "g_score": g_sc, "alpha": gates(env.sim)["alpha_up_deg"]})
    if float(g_sc["useful"]) < 2.5:
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "failure_code": FAIL_PREPARATION,
            "phase": "gravity_projection",
            "reason": "insufficient_outward_gravity",
            "g_score": g_sc,
            "actions": actions,
            "capture": cur["capture"],
            "phase_log": phase_log,
        }

    e_x0 = float(s0["e_signed_mm"][0])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    # max_safe_travel only — goal-driven DualLoop stops on geometric region
    milestone = float(cfg.outward_slip_mm)
    slip = _outward_slip_staged(
        env, cur, z_ref=z_ref, e_x0=e_x0, milestone_mm=milestone, cfg=cfg,
    )
    actions += slip.get("actions") or []
    phase_log.append(
        {
            "phase": "slip",
            "ok": slip.get("ok"),
            "disp_along_mm": slip.get("disp_along_mm"),
            "disp_total_mm": slip.get("disp_total_mm"),
            "stop_reason": slip.get("stop_reason"),
            "goal_reached": slip.get("goal_reached"),
            "contract": slip.get("contract"),
            "failure_code": slip.get("failure_code"),
        }
    )
    if not slip.get("ok"):
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "failure_code": slip.get("failure_code") or FAIL_SLIP_CONTACT,
            "phase": "slip",
            "reason": slip.get("reason") or slip.get("abort_reason"),
            "physical_reason": slip.get("reason") or slip.get("abort_reason"),
            "actions": actions,
            "capture": slip.get("capture") or take_capture(env),
            "phase_log": phase_log,
            "g_score": g_sc,
            "failure": slip.get("failure"),
            "disp_total_mm": slip.get("disp_total_mm"),
            "staged_log": slip.get("staged_log"),
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        }

    restore_capture(env, slip["capture"])
    # Stay fingertips-down — no IC restore, no fingertips-up
    settle_hold(env, actions, hold_s=0.4)
    err = measure_errors(env.sim)
    # Coupling: lateral / theta should not explode
    if abs(float(err["dx_mm"]) - float(e0[0])) > 5.0 and abs(float(err["dx_mm"])) > 12.0:
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "failure_code": FAIL_SLIP_WRONG_DIR,
            "phase": "coupling_dx",
            "reason": "lateral_regression_after_outward",
            "actions": actions,
            "capture": take_capture(env),
            "phase_log": phase_log,
            "errors": err,
            "errors_entry": e0.tolist(),
        }
    phase_log.append({"phase": "settle", "ok": True, "errors": err})
    return {
        "ok": True,
        "mechanism": "dual_loop_outward",
        "failure_code": None,
        "disp_along_mm": slip.get("disp_along_mm"),
        "disp_total_mm": slip.get("disp_total_mm"),
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
        "contract": slip.get("contract"),
        "g_score": g_sc,
        "actions": actions,
        "capture": take_capture(env),
        "errors": err,
        "phase_log": phase_log,
        "direction": d_out.tolist(),
    }


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
    max_cycles = int(getattr(cfg, "max_correction_cycles", 4) or 4)
    action_budget = int(getattr(cfg, "global_action_budget", 2500) or 2500)
    done_dtheta = False
    done_dx = False
    done_dz = False
    theta_after_pivot: float | None = None

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
                cycle_log.append(cycle_row)
                logs["cycles"] = cycle_log
                phases.append("PRECISION_EARLY_HANDOFF")
                return PrecisionResult(
                    status="PRECISION_EXECUTED_SUCCESS" if actions else "PRECISION_SKIPPED_DIRECT_SUCCESS",
                    plan_id=plan_id,
                    precision_skipped=len(actions) == 0,
                    precision_executed=len(actions) > 0,
                    gate_ok=True,
                    phases=phases,
                    decision=decision,
                    errors_entry=errors_entry,
                    errors_exit=err_now,
                    actions=actions,
                    capture=cur["capture"],
                    ref=cur["ref"],
                    handoff=probe.get("handoff"),
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
        # Pivot at most once — lateral/axial coupling must not re-trigger a failed second pivot.
        if need["dtheta_needed"] and not done_dtheta:
            phases.append("correct_dtheta")
            r = correct_dtheta(env, cur, cfg, plan_id)
            logs["dtheta"] = {k: v for k, v in r.items() if k not in ("actions", "capture")}
            actions.extend(r.get("actions") or [])
            cycle_row["primitive"] = "dtheta"
            cycle_row["ok"] = r.get("ok")
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
            if cfg.allow_horizontal_dx:
                phases.append("dx_horizontal_attempt")
                snap = take_capture(env)
                rh = correct_dx_horizontal_attempt(env, cur, cfg)
                logs["dx_horizontal"] = {k: v for k, v in rh.items() if k not in ("actions", "capture")}
                err_h = rh.get("errors") or {}
                dx_mm_h = float(err_h.get("dx_mm", 1e9)) if rh.get("ok") else 1e9
                e_pos_h = float(err_h.get("e_pos_mm", 1e9)) if rh.get("ok") else 1e9
                # Accept full or partial horizontal if safe progress and recoverable
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
                            and (
                                abs(dx_mm_h) < abs(float(err_before["dx_mm"])) - 0.3
                                or e_pos_h < float(err_before["e_pos_mm"]) - 0.3
                            )
                        )
                    )
                )
                # Do not commit horizontal to the live tape unless the post-state is
                # HANDOFF-viable or a counterfactual inward/outward axial remains recoverable.
                if horizontal_accept and bool(getattr(cfg, "require_horizontal_recoverability", True)):
                    restore_capture(env, rh["capture"])
                    err_h_live = measure_errors(env.sim)
                    need_h = assess_need(err_h_live, cfg)
                    recoverable = False
                    recover_note = None
                    if need_h.get("ready_for_direct_handoff_assessment") or handoff_geometry_ready(err_h_live, cfg):
                        pr = probe_direct_handoff(
                            env, rh["capture"], cur["ref"], stabilize_n=cfg.stabilize_handoff_n
                        )
                        restore_capture(env, rh["capture"])
                        recoverable = bool(pr.get("SUCCESS_NEW"))
                        recover_note = "handoff_probe"
                        logs["dx_horizontal_recoverability"] = {
                            "mode": recover_note,
                            "SUCCESS_NEW": pr.get("SUCCESS_NEW"),
                            "SUCCESS_12S": pr.get("SUCCESS_12S"),
                        }
                    if not recoverable and need_h.get("dz_needed"):
                        mode_h = need_h.get("dz_mode") or "none"
                        # Counterfactual axial — discard actions; do not append to tape.
                        if mode_h == "B_inward":
                            rz_cf = correct_dz_inward(
                                env, {"capture": rh["capture"], "ref": cur["ref"]}, cfg
                            )
                        elif mode_h == "A_outward":
                            rz_cf = correct_dz_outward(
                                env, {"capture": rh["capture"], "ref": cur["ref"]}, cfg
                            )
                        else:
                            rz_cf = {"ok": False, "reason": "no_dz_mode"}
                        restore_capture(env, snap)
                        recoverable = bool(rz_cf.get("ok"))
                        recover_note = f"counterfactual_{mode_h}"
                        logs["dx_horizontal_recoverability"] = {
                            "mode": recover_note,
                            "ok": rz_cf.get("ok"),
                            "reason": rz_cf.get("reason") or rz_cf.get("stop_reason"),
                            "n_cf_actions_discarded": len(rz_cf.get("actions") or []),
                        }
                    elif not recoverable:
                        restore_capture(env, snap)
                        logs["dx_horizontal_recoverability"] = {
                            "mode": "no_followup_needed_but_handoff_failed",
                            "ok": False,
                        }
                    if not recoverable:
                        horizontal_accept = False
                        failures.append(
                            f"horizontal_dx:unrecoverable_post_state:note={recover_note}"
                        )
                        logs["dx_horizontal_rejected"] = {
                            "failure_code": "UNRECOVERABLE_AFTER_HORIZONTAL",
                            "phase": "recoverability",
                            "ok_but_rejected": True,
                            "dx_mm_after": dx_mm_h,
                            "e_pos_mm_after": e_pos_h,
                            "n_rejected_actions": len(rh.get("actions") or []),
                            "note": "Horizontal omitted; restored pre-attempt; verified lateral may follow.",
                        }
                        restore_capture(env, snap)
                        cur = {"capture": snap, "ref": cur["ref"]}
                if horizontal_accept:
                    actions.extend(rh.get("actions") or [])
                    cur = {"capture": rh["capture"], "ref": cur["ref"]}
                    dx_ok = True
                    logs["dx_mechanism"] = "horizontal_dx"
                    logs["dx_partial"] = bool(abs(dx_mm_h) > float(cfg.ex_tol_mm))
                    cycle_row["primitive"] = "dx_horizontal"
                    cycle_row["partial"] = logs["dx_partial"]
                    cycle_row["stop_reason"] = rh.get("stop_reason")
                elif "dx_horizontal_rejected" not in logs:
                    code = rh.get("failure_code") or rh.get("reason") or "failed"
                    failures.append(f"horizontal_dx:{code}:phase={rh.get('phase')}")
                    logs["dx_horizontal_rejected"] = {
                        "failure_code": rh.get("failure_code"),
                        "phase": rh.get("phase"),
                        "physical_reason": rh.get("physical_reason"),
                        "ok_but_rejected": bool(rh.get("ok")),
                        "dx_mm_after": dx_mm_h if rh.get("ok") else None,
                        "n_rejected_actions": len(rh.get("actions") or []),
                        "note": "Rejected branch omitted; restored to pre-attempt capture.",
                    }
                    restore_capture(env, snap)
                    cur = {"capture": snap, "ref": cur["ref"]}
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
                        note="Horizontal dx failed or skipped; verified lateral also failed.",
                    )
                cur = {"capture": rv["capture"], "ref": cur["ref"]}
                dx_ok = True
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
            if mode == "B_inward":
                phases.append("correct_dz_inward")
                rz = correct_dz_inward(env, cur, cfg)
                # Preserve verified inward baseline if goal-driven slip leaves unrestorable state.
                if not rz.get("ok") and bool(getattr(cfg, "use_goal_driven_slip", True)):
                    phases.append("correct_dz_inward_legacy_fallback")
                    cfg_leg = PrecisionConfig(**{**asdict(cfg), "use_goal_driven_slip": False, "inward_slip_mm": 7.0})
                    rz2 = correct_dz_inward(env, cur, cfg_leg)
                    if rz2.get("ok"):
                        rz = rz2
                        logs["dz_legacy_fallback"] = True
            elif mode == "A_outward":
                phases.append("correct_dz_outward")
                rz = correct_dz_outward(env, cur, cfg)
            else:
                rz = {"ok": True, "skipped": True, "actions": [], "capture": cur["capture"]}
                phases.append("dz_skip")
            logs["dz"] = {k: v for k, v in rz.items() if k not in ("actions", "capture")}
            actions.extend(rz.get("actions") or [])
            cycle_row["primitive"] = f"dz:{mode}"
            cycle_row["ok"] = rz.get("ok")
            cycle_row["stop_reason"] = (rz.get("slip") or {}).get("stop_reason") or rz.get("stop_reason")
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
            done_dz = True
            cycle_log.append(cycle_row)
            continue

        # Allow one more dx pass if dz coupling reintroduced lateral error
        if need["dx_needed"] and done_dx and done_dz:
            done_dx = False  # one coupling retry
            cycle_row["primitive"] = "dx_coupling_retry_armed"
            cycle_log.append(cycle_row)
            continue

        # Nothing needed by assess_need but HANDOFF probe already failed this cycle
        cycle_row["primitive"] = "none"
        cycle_log.append(cycle_row)
        phases.append("NO_REMAINING_CORRECTION")
        break

    logs["cycles"] = cycle_log

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
        note="Goal-driven closed-loop Stage-2; criterion is frozen nominal SUCCESS_NEW.",
    )


# Conceptual alias matching the task brief signature (sim-style).
def run_precision_from_sim(sim_env, stable_grasp_state, plan_id, precision_config=None) -> PrecisionResult:
    return run_precision(sim_env, stable_grasp_state, plan_id, precision_config)
