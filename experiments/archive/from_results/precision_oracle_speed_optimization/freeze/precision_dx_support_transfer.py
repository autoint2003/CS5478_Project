"""Support-transfer gravity-assisted dx (flowchart DX-1..DX-7).

Distinct from ``correct_dx_verified`` (orientation-plan DualLoop) and from the
legacy horizontal path that always restores fingertips-DOWN before exit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.axis_aware import lateral_desire_direction
from data_generation.oracles.precision.contact_modes import (
    MODE_BILATERAL_FRICTION,
    MODE_ONE_SIDED_SUPPORT,
    classify_contact_mode,
    make_s1_state,
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
from data_generation.oracles.precision.force_slip_control import DT, sample_state
from data_generation.oracles.precision.precision_dx import (
    _horizontal_open_target,
    _physical_finger_support,
    _score_horizontal_slide_config,
    select_horizontal_sliding_roll,
)
from data_generation.oracles.precision.precision_goals import measure_errors
from data_generation.oracles.precision.precision_slip import run_goal_aware_axial_slip, settle_hold
from data_generation.oracles.precision.precision_types import (
    FAIL_BRAKE,
    FAIL_HORIZONTAL_TILT,
    FAIL_SLIDING_ROLL,
    FAIL_SLIP_CONTACT,
    FAIL_SUPPORT_ROLL,
    FN_HOLD,
    G_LATERAL_MIN,
    PrecisionConfig,
    SlipGoal,
    STOP_GOAL_REACHED,
    STOP_HANDOFF_READY,
    STOP_MAX_SAFE_TRAVEL,
)
from envs.simulation.capture import restore_capture, take_capture


def restore_fingertips_down(
    env,
    blob,
    cfg: PrecisionConfig,
    *,
    rate_scale: float = 0.16,
    z_ref_boost: float = 0.05,
    height_kp: float = 5.0,
    pre_lift_s: float = 0.0,
    pre_lift_z: float | None = None,
) -> dict[str, Any]:
    """Protected return to fingertips-DOWN under current grasp (post-dx if no dz)."""
    from data_generation.oracles.precision.force_slip_control import make_action
    from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand
    from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip
    from envs.step_legal import step_legal

    actions: list = []
    restore_capture(env, blob["capture"])
    g0 = gates(env.sim)
    # Near-DOWN is treated as already restored (gate is typically ≥165°).
    if g0["fingertips_down"] or float(g0["alpha_up_deg"]) >= 160.0:
        return {
            "ok": True,
            "skipped": True,
            "actions": [],
            "capture": blob["capture"],
            "errors": measure_errors(env.sim),
            "contact_mode": classify_contact_mode(env.sim),
            "alpha_up_deg": g0["alpha_up_deg"],
        }
    if not sample_state(env.sim).get("bilateral"):
        return {
            "ok": False,
            "reason": "cannot_restore_DOWN_without_bilateral",
            "actions": [],
            "capture": blob["capture"],
        }
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    z_lift = float(pre_lift_z) if pre_lift_z is not None else max(z0 + 0.10, 0.54)
    for _ in range(int(max(0.0, pre_lift_s) / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_lift, kp=max(height_kp, 8.0)), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        if not sample_state(env.sim).get("bilateral"):
            restore_capture(env, blob["capture"])
            return {
                "ok": False,
                "reason": "contact_loss_during_pre_lift",
                "actions": [],
                "capture": blob["capture"],
            }
    cur_blob = {"capture": take_capture(env), "ref": blob.get("ref")}
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    # Conservative first attempt
    p = run_tilt_phase(
        env,
        cur_blob,
        target_dir_w=DOWN,
        phase="restore_DOWN",
        rate_scale=min(float(rate_scale), 0.18),
        fn_hold=FN_HOLD,
        ang_tol_deg=6.0,
        y0_w=y0,
        e_ref=e_ref,
        allow_roll=False,
        contact_aware=True,
        max_time_s=14.0,
        z_ref_boost=z_ref_boost,
        height_kp=height_kp,
    )
    if not p.get("ok"):
        restore_capture(env, cur_blob["capture"])
        p = run_tilt_phase(
            env,
            cur_blob,
            target_dir_w=DOWN,
            phase="restore_DOWN_retry",
            rate_scale=min(0.08, float(rate_scale)),
            fn_hold=FN_HOLD,
            ang_tol_deg=8.0,
            y0_w=y0,
            e_ref=e_ref,
            allow_roll=False,
            contact_aware=True,
            max_time_s=18.0,
            z_ref_boost=max(z_ref_boost, 0.12),
            height_kp=max(height_kp, 8.0),
        )
    actions += p.get("actions") or []
    if not p.get("ok"):
        # Do not keep destabilizing actions; return to entry capture.
        restore_capture(env, blob["capture"])
        return {
            "ok": False,
            "reason": p.get("reason") or "restore_DOWN_failed",
            "actions": [],
            "capture": blob["capture"],
        }
    settle_hold(env, actions, hold_s=0.25)
    return {
        "ok": True,
        "actions": actions,
        "capture": take_capture(env),
        "errors": measure_errors(env.sim),
        "contact_mode": classify_contact_mode(env.sim),
    }


def correct_dx_support_transfer(env, blob, cfg: PrecisionConfig) -> dict[str, Any]:
    """DX-1..DX-6: S1 → horizontal → S2 → gravity lateral slide → brake → S1 (keep).

    Does **not** restore fingertips-DOWN. Caller restores only if dz is not needed.
    """
    import time

    t_phases: dict[str, float] = {}
    actions: list = []
    phase_log: list[dict] = []
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    d_lat = lateral_desire_direction(s0["e_signed_mm"])
    if float(np.linalg.norm(d_lat)) < 0.5:
        return {
            "ok": True,
            "skipped": True,
            "mechanism": "support_transfer_dx",
            "reason": "dx_near_zero",
            "actions": [],
            "capture": blob["capture"],
            "s1_state": make_s1_state(env.sim),
            "timing_s": {},
        }

    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(s0["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    roll_sign = float(choose_roll_sign_for_support(env.sim))

    # Support-aware rates. Horizontal tilt on rolled supports is contact-fragile:
    # modest DX-1 roll, then protected DX-2 at verified ≤0.08 (faster caps leave |ex| unsafe).
    rate_s1 = float(getattr(cfg, "rate_establish_s1", 0.14) or 0.14)
    rate_protected = float(getattr(cfg, "rate_protected_s1", 0.08) or 0.08)
    rate_horiz_cap = float(getattr(cfg, "dx_horizontal_rate_cap", 0.08) or 0.08)
    rate_friction = float(getattr(cfg, "rate_friction_support", 0.14) or 0.14)
    s1_roll_deg = float(getattr(cfg, "dx_s1_roll_deg", 12.0) or 12.0)

    # --- DX-1: establish preparatory one-sided support bias (modest roll at DOWN) ---
    t0 = time.time()
    if not gates(env.sim)["fingertips_down"]:
        p0 = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dxST_P0", rate_scale=rate_s1, fn_hold=FN_HOLD,
            ang_tol_deg=6.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=10.0,
        )
        actions += p0.get("actions") or []
        if not (p0.get("ok") and gates(env.sim)["fingertips_down"]):
            return {
                "ok": False,
                "mechanism": "support_transfer_dx",
                "phase": "DX1_prep_down",
                "failure_code": FAIL_SUPPORT_ROLL,
                "reason": p0.get("reason"),
                "actions": actions,
                "capture": p0.get("capture") or cur["capture"],
            }
        cur = {"capture": p0["capture"], "ref": blob["ref"]}

    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=s1_roll_deg, phase="dxST_DX1_S1",
        rate_scale=rate_s1, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    actions += p1.get("actions") or []
    t_phases["DX1_S1"] = time.time() - t0
    bil1, sup1 = _physical_finger_support(env.sim)
    mode1 = classify_contact_mode(env.sim)
    phase_log.append({
        "phase": "DX1_S1",
        "ok": p1.get("ok") and bil1,
        "roll_deg": s1_roll_deg,
        "mode": mode1,
        "support": {k: sup1.get(k) for k in ("fn_L", "fn_R", "fn_tot")},
        "note": "Modest support-bias roll; full one-sided load appears under DX-2 gravity.",
    })
    if not p1.get("ok") or not bil1:
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX1_S1",
            "failure_code": FAIL_SUPPORT_ROLL,
            "reason": "mechanical_support_not_established",
            "actions": actions,
            "capture": p1.get("capture") or cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    cur = {"capture": p1["capture"], "ref": blob["ref"]}

    # --- DX-2: protected slow rotation to horizontal; verify S1 under gravity ---
    t0 = time.time()
    restore_capture(env, cur["capture"])
    u_tgt = _horizontal_open_target(env.sim)
    p2 = run_tilt_phase(
        env, cur, target_dir_w=u_tgt, phase="dxST_DX2_horizontal",
        rate_scale=min(rate_protected, rate_horiz_cap),
        fn_hold=FN_HOLD, ang_tol_deg=12.0, y0_w=y0, e_ref=e_ref, allow_roll=False,
        contact_aware=True, max_time_s=22.0, z_ref_boost=0.05, height_kp=5.0,
    )
    if not p2.get("ok"):
        restore_capture(env, cur["capture"])
        p2 = run_tilt_phase(
            env, cur, target_dir_w=u_tgt, phase="dxST_DX2_retry",
            rate_scale=min(0.05, rate_horiz_cap),
            fn_hold=FN_HOLD, ang_tol_deg=14.0, y0_w=y0, e_ref=e_ref, allow_roll=False,
            contact_aware=True, max_time_s=24.0, z_ref_boost=0.07, height_kp=5.5,
        )
    actions += p2.get("actions") or []
    t_phases["DX2_horizontal"] = time.time() - t0
    bil2, sup2 = _physical_finger_support(env.sim)
    mode2 = classify_contact_mode(env.sim)
    alpha2 = float(gates(env.sim)["alpha_up_deg"])
    near_horizontal = 70.0 <= alpha2 <= 120.0
    s1_at_horiz = bool(mode2.get("one_sided_mechanical") or mode2.get("bilateral"))
    phase_log.append({
        "phase": "DX2_horizontal",
        "ok": bool(p2.get("ok") and bil2 and near_horizontal),
        "alpha": alpha2,
        "near_horizontal": near_horizontal,
        "s1_under_gravity": mode2.get("one_sided_mechanical"),
        "support_side": mode2.get("support_side"),
        "mode": mode2.get("mode"),
        "support": {k: sup2.get(k) for k in ("fn_L", "fn_R", "fn_tot")},
    })
    if not bil2:
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX2_horizontal",
            "failure_code": FAIL_HORIZONTAL_TILT,
            "reason": "support_lost_during_rotation",
            "actions": actions,
            "capture": p2.get("capture") or cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    if not (p2.get("ok") or near_horizontal) or not s1_at_horiz:
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX2_horizontal",
            "failure_code": FAIL_HORIZONTAL_TILT,
            "reason": p2.get("reason") or "horizontal_unreachable",
            "actions": actions,
            "capture": p2.get("capture") or cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    cur = {"capture": p2.get("capture") or take_capture(env), "ref": blob["ref"]}

    # --- DX-3: bilateral friction support (sliding roll) ---
    t0 = time.time()
    sel = select_horizontal_sliding_roll(env, cur, d_lat, y0, e_ref)
    if not sel.get("ok"):
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX3_S2_select",
            "failure_code": FAIL_SLIDING_ROLL,
            "reason": "bilateral_friction_support_unreachable",
            "actions": actions,
            "capture": cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    p3 = run_roll_phase(
        env, cur, sign=float(sel["sign"]), angle_deg=float(sel["angle_deg"]), phase="dxST_DX3_S2",
        rate_scale=rate_friction, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
    )
    actions += p3.get("actions") or []
    t_phases["DX3_S2"] = time.time() - t0
    restore_capture(env, p3.get("capture") or cur["capture"])
    sc = _score_horizontal_slide_config(env.sim, d_lat)
    mode3 = classify_contact_mode(env.sim)
    phase_log.append({"phase": "DX3_S2", "ok": p3.get("ok") and sc["ok"], "mode": mode3, "geometry": sc})
    if not p3.get("ok") or not sc["ok"]:
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX3_S2",
            "failure_code": FAIL_SLIDING_ROLL,
            "reason": "bilateral_friction_support_failed",
            "actions": actions,
            "capture": p3.get("capture") or cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    cur = {"capture": p3["capture"], "ref": blob["ref"]}

    # --- DX-4: gravity-driven lateral sliding ---
    t0 = time.time()
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    ex0 = float(sample_state(env.sim)["e_signed_mm"][0])
    goal = SlipGoal(
        axis="dx",
        desired_direction=d_lat,
        max_safe_travel_mm=float(cfg.horizontal_slip_mm),
        max_duration_s=12.0,
        axis_tol_mm=float(getattr(cfg, "lateral_axis_tol_mm", 3.0) or 3.0),
        working_pos_mm=float(cfg.working_pos_mm),
        working_theta_deg=float(cfg.working_theta_deg),
        ey_tol_mm=float(cfg.ey_tol_mm),
        use_euclidean_stop=False,
    )
    slip = run_goal_aware_axial_slip(env, cur, goal, z_ref=z_ref, guard_lateral=False)
    actions += slip.get("actions") or []
    t_phases["DX4_slide"] = time.time() - t0
    disp = float(slip.get("disp_along_mm") or slip.get("disp_total_mm") or 0.0)
    ex1 = float((slip.get("e_signed") or sample_state(env.sim)["e_signed_mm"])[0])
    phase_log.append({
        "phase": "DX4_slide",
        "ok": slip.get("ok"),
        "disp_along_mm": slip.get("disp_along_mm"),
        "ex0": ex0,
        "ex1": ex1,
        "stop_reason": slip.get("stop_reason"),
        "mode": MODE_BILATERAL_FRICTION,
    })
    if not slip.get("ok") and abs(disp) < 0.5:
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX4_slide",
            "failure_code": slip.get("failure_code") or FAIL_SLIP_CONTACT,
            "reason": slip.get("reason") or "sliding_did_not_occur",
            "actions": actions,
            "capture": slip.get("capture") or cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    # Wrong-direction check
    if abs(ex1) > abs(ex0) + 2.0 and np.sign(ex1 - ex0) == np.sign(ex0):
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX4_slide",
            "failure_code": FAIL_SLIP_CONTACT,
            "reason": "sliding_wrong_direction",
            "actions": actions,
            "capture": slip.get("capture") or cur["capture"],
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    cur = {"capture": slip["capture"], "ref": blob["ref"]}

    # --- DX-5: brake and regrip ---
    t0 = time.time()
    restore_capture(env, cur["capture"])
    settle_hold(env, actions, hold_s=0.35)
    st = sample_state(env.sim)
    t_phases["DX5_brake"] = time.time() - t0
    if float(st.get("v_rel_norm") or 0) > 0.08 or not st.get("bilateral"):
        return {
            "ok": False,
            "mechanism": "support_transfer_dx",
            "phase": "DX5_brake",
            "failure_code": FAIL_BRAKE,
            "reason": "brake_could_not_arrest_slip" if st.get("bilateral") else "regrip_failed",
            "actions": actions,
            "capture": take_capture(env),
            "phase_log": phase_log,
            "timing_s": t_phases,
        }
    cur = {"capture": take_capture(env), "ref": blob["ref"]}
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)

    # --- DX-6: reverse S2 roll → re-establish S1; do NOT restore DOWN ---
    t0 = time.time()
    p6 = run_roll_phase(
        env, cur, sign=-float(sel["sign"]), angle_deg=float(sel["angle_deg"]), phase="dxST_DX6_S1",
        rate_scale=rate_friction, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
    )
    actions += p6.get("actions") or []
    # Small reinforce support roll toward S1
    if p6.get("ok"):
        cur = {"capture": p6["capture"], "ref": blob["ref"]}
        p_s1 = run_roll_phase(
            env, cur, sign=roll_sign, angle_deg=25.0, phase="dxST_DX6_reinforce_S1",
            rate_scale=rate_s1, fn_hold=FN_HOLD, require_down=False, y0_w=y0, e_ref=e_ref,
        )
        if p_s1.get("ok"):
            actions += p_s1.get("actions") or []
            cur = {"capture": p_s1["capture"], "ref": blob["ref"]}
    t_phases["DX6_S1"] = time.time() - t0
    restore_capture(env, cur["capture"])
    settle_hold(env, actions, hold_s=0.2)
    bil_exit = bool(sample_state(env.sim).get("bilateral"))
    s1 = make_s1_state(env.sim, roll_sign=roll_sign, force_valid=False)
    mode_exit = classify_contact_mode(env.sim)
    phase_log.append({"phase": "DX6_S1", "ok": bool(s1.get("valid") or mode_exit["bilateral"]), "s1": s1, "mode": mode_exit})

    err = measure_errors(env.sim)
    ok = bool(
        bil_exit
        and (
            abs(float(err["dx_mm"])) <= float(cfg.ex_tol_mm) + 2.0
            or abs(float(err["dx_mm"])) < abs(float(s0["e_signed_mm"][0])) - 1.0
            or abs(float(slip.get("disp_along_mm") or 0.0)) >= 0.8
        )
    )
    return {
        "ok": ok,
        "mechanism": "support_transfer_dx",
        "actions": actions,
        "capture": take_capture(env),
        "errors": err,
        "s1_state": s1,
        "keep_s1_for_dz": True,
        "phase_log": phase_log,
        "timing_s": t_phases,
        "disp_along_mm": slip.get("disp_along_mm"),
        "ex0": ex0,
        "ex1": float(err["e_signed_mm"][0]),
        "contact_mode_exit": mode_exit,
        "note": "Exit in S1-capable state; do not restore DOWN here.",
    }
