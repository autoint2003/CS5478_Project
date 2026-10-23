"""Inward / outward axial dz with support-aware flips and Case A–D shortcuts."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode, make_s1_state
from data_generation.oracles.precision.mechanics.open_axis import (
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    project_out_roll,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
    tilt_omega_toward,
    u_open_world,
    world_omega_to_body,
)
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND
from data_generation.oracles.precision.mechanics.dual_loop import DT, make_action, sample_state
from data_generation.oracles.precision.corrections.orientation import (
    _height_hold_v_hand,
    score_gravity_alignment,
)
from data_generation.oracles.precision.replay_audit import _failure_snapshot
from data_generation.oracles.precision.corrections.lateral import _horizontal_open_target
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.mechanics.axial_sliding import (
    _inward_slip_staged,
    _outward_slip_staged,
    settle_hold,
)
from data_generation.oracles.precision.state import (
    FAIL_DZ_PRECONDITION,
    FAIL_PREPARATION,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_WRONG_DIR,
    FAIL_SUPPORT_ROLL,
    FN_HOLD,
    PrecisionConfig,
    Z_ABORT,
)
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from data_generation.oracles.precision.mechanics.protected_rotation import (
    plan_and_execute_support_tilt,
    run_fixed_axis_tilt,
    select_support_aware_tilt_axis,
    verify_mechanical_s1,
)
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal

# Lazy import helper to avoid circular import with support-transfer dx.
from data_generation.oracles.precision.corrections.axial_common import (
    _dx_support_transfer,
    _empty_dz_outcomes,
    _near_down,
    _prep_support_for_flip,
    _rate,
    _s1_valid,
    _support_tilt,
    _try_reverse_up_actions,
)

def correct_dz_outward(
    env, blob, cfg: PrecisionConfig, *, s1_state: dict | None = None
) -> dict[str, Any]:
    """Case A/C/D outward: direct slide when DOWN; else support-aware rotate to DOWN."""
    actions: list = []
    phase_log: list[dict] = []
    rotation_diag: list[dict] = []
    restore_capture(env, blob["capture"])
    y0 = reference_pad_vector(env.sim)
    e0 = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    d_out = np.asarray(U_OPEN_HAND, float)
    roll_sign = float(
        (s1_state or {}).get("roll_sign")
        if (s1_state or {}).get("roll_sign") is not None
        else choose_roll_sign_for_support(env.sim)
    )
    branch = "A_direct_down"
    reused_s1 = False
    skipped_rotation = False
    side_hint = (s1_state or {}).get("support_side")

    g0 = gates(env.sim)
    if _near_down(env.sim):
        skipped_rotation = True
        branch = "A_direct_down" if not s1_state else "D_already_down"
        phase_log.append({
            "phase": "prep",
            "ok": True,
            "skipped": True,
            "reason": "already_fingertips_down_direct_outward",
            "alpha": g0["alpha_up_deg"],
        })
    else:
        if _s1_valid(s1_state, env.sim):
            reused_s1 = True
            branch = "C_reuse_s1_to_down"
            phase_log.append({"phase": "S1", "skipped": True, "reused": True})
        else:
            live0 = verify_mechanical_s1(env.sim)
            if live0.get("ok"):
                phase_log.append({"phase": "S1", "skipped": True, "reused_live": True})
                side_hint = side_hint or live0["live"].get("support_side")
            else:
                est = _prep_support_for_flip(env, cur, cfg, y0, e0, roll_sign, actions)
                phase_log.append({"phase": "S1_establish", "ok": est.get("ok")})
                if not est.get("ok"):
                    return {
                        "ok": False,
                        "mechanism": "dual_loop_outward",
                        "branch": branch,
                        "failure_code": est.get("failure_code") or FAIL_PREPARATION,
                        "phase": "S1",
                        "reason": est.get("reason"),
                        "actions": actions,
                        "capture": est.get("capture") or cur["capture"],
                        "phase_log": phase_log,
                    }
                cur = {"capture": est["capture"], "ref": blob["ref"]}
                side_hint = side_hint or (est.get("s1") or {}).get("support_side")
        p_down = _support_tilt(
            env, cur, DOWN, cfg, actions, "dzOut_to_DOWN",
            side_hint=side_hint, y0=y0, e_ref=e0,
        )
        rotation_diag.append({"phase": "to_DOWN", **{k: p_down.get(k) for k in (
            "ok", "reason", "axis_w", "sign", "selection", "path_length_u", "sim_s",
        )}})
        phase_log.append({"phase": "to_DOWN", "ok": p_down.get("ok"), "selection": (p_down.get("selection") or {}).get("chosen_name")})
        if not (p_down.get("ok") and _near_down(env.sim)):
            return {
                "ok": False,
                "mechanism": "dual_loop_outward",
                "branch": branch,
                "failure_code": FAIL_PREPARATION,
                "phase": "to_DOWN",
                "reason": p_down.get("reason") or "cannot_align_down_for_outward",
                "actions": actions,
                "capture": p_down.get("capture") or cur["capture"],
                "phase_log": phase_log,
                "rotation_diag": rotation_diag,
                "reused_s1": reused_s1,
            }
        cur = {"capture": p_down["capture"], "ref": blob["ref"]}

    restore_capture(env, cur["capture"])
    s0 = sample_state(env.sim)
    g_sc = score_gravity_alignment(np.asarray(s0["g_h"], float), d_out)
    phase_log.append({"phase": "gravity_check", "g_score": g_sc, "alpha": gates(env.sim)["alpha_up_deg"]})
    if float(g_sc["useful"]) < 2.5:
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "branch": branch,
            "failure_code": FAIL_PREPARATION,
            "phase": "gravity_projection",
            "reason": "insufficient_outward_gravity",
            "g_score": g_sc,
            "actions": actions,
            "capture": cur["capture"],
            "phase_log": phase_log,
            "skipped_rotation": skipped_rotation,
            "reused_s1": reused_s1,
        }

    if not s0.get("bilateral") or float(s0.get("fn_tot") or 0) < 4.0:
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "branch": branch,
            "failure_code": FAIL_PREPARATION,
            "phase": "grasp_margin",
            "reason": "unstable_or_insufficient_contact_margin",
            "actions": actions,
            "capture": cur["capture"],
            "phase_log": phase_log,
        }

    e_x0 = float(s0["e_signed_mm"][0])
    ez0 = float(s0["e_signed_mm"][2])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    slip = _outward_slip_staged(
        env, cur, z_ref=z_ref, e_x0=e_x0, milestone_mm=float(cfg.outward_slip_mm), cfg=cfg,
    )
    actions += slip.get("actions") or []
    phase_log.append({
        "phase": "slip",
        "ok": slip.get("ok"),
        "ez0": ez0,
        "disp_along_mm": slip.get("disp_along_mm"),
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
    })
    if not slip.get("ok"):
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "branch": branch,
            "failure_code": slip.get("failure_code") or FAIL_SLIP_CONTACT,
            "phase": "slip",
            "reason": slip.get("reason") or slip.get("abort_reason"),
            "actions": actions,
            "capture": slip.get("capture") or take_capture(env),
            "phase_log": phase_log,
            "skipped_rotation": skipped_rotation,
            "reused_s1": reused_s1,
            "failure": slip.get("failure") or _failure_snapshot(env.sim, phase="slip", code=FAIL_SLIP_CONTACT),
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        }

    restore_capture(env, slip["capture"])
    settle_hold(env, actions, hold_s=0.4)
    err = measure_errors(env.sim)
    if abs(float(err["dx_mm"]) - float(e0[0])) > 5.0 and abs(float(err["dx_mm"])) > 12.0:
        return {
            "ok": False,
            "mechanism": "dual_loop_outward",
            "branch": branch,
            "failure_code": FAIL_SLIP_WRONG_DIR,
            "phase": "coupling_dx",
            "reason": "lateral_regression_after_outward",
            "actions": actions,
            "capture": take_capture(env),
            "phase_log": phase_log,
            "errors": err,
            "skipped_rotation": skipped_rotation,
            "reused_s1": reused_s1,
        }
    phase_log.append({"phase": "settle", "ok": True, "errors": err})
    return {
        "ok": True,
        "mechanism": "dual_loop_outward",
        "branch": branch,
        "skipped_rotation": skipped_rotation,
        "reused_s1": reused_s1,
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
        "rotation_diag": rotation_diag,
        "direction": d_out.tolist(),
        "fingertips_down": bool(gates(env.sim)["fingertips_down"] or _near_down(env.sim)),
    }

