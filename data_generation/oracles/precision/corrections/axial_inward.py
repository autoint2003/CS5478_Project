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

def check_dz_inward_entry(sim, cfg: PrecisionConfig) -> dict[str, Any]:
    """Physical preconditions for inward flip/slip — not SUCCESS_NEW geometry alone.

    Large |ex| must be corrected by the scheduler's dx stage first. Dz must not
    silently run support-transfer seating; that hides ordering bugs in probes.
    """
    err = measure_errors(sim)
    st = sample_state(sim)
    m = classify_contact_mode(sim)
    e = np.asarray(err["e_signed_mm"], float)
    ex = abs(float(e[0]))
    ey = abs(float(e[1]))
    vrel = float(st.get("v_rel_norm") or 0.0)
    max_ex = float(getattr(cfg, "dz_safe_ex_mm", 6.0) or 6.0)
    max_vrel = float(getattr(cfg, "dz_safe_vrel", 0.025) or 0.025)
    reasons: list[str] = []
    if ex > max_ex:
        reasons.append(f"lateral_ex_too_large:{ex:.2f}>{max_ex:.2f}")
    if ey > float(cfg.ey_tol_mm):
        reasons.append(f"lateral_ey_too_large:{ey:.2f}>{float(cfg.ey_tol_mm):.2f}")
    if not st.get("bilateral"):
        reasons.append("not_bilateral")
    if latched(sim):
        reasons.append("support_drop_latched")
    if vrel > max_vrel:
        reasons.append(f"relative_velocity_high:{vrel:.4f}>{max_vrel:.4f}")
    if float(err.get("fn_tot") or 0.0) < 4.0:
        reasons.append("insufficient_grasp_force")
    if float(err.get("obj_z") or 0.0) < Z_ABORT:
        reasons.append("world_height_unsafe")
    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "errors": err,
        "contact_mode": m.get("mode"),
        "support_side": m.get("support_side"),
        "Fn_z_L": m.get("Fn_z_L"),
        "Fn_z_R": m.get("Fn_z_R"),
        "one_sided_mechanical": m.get("one_sided_mechanical"),
        "ex_mm": float(e[0]),
        "ez_mm": float(e[2]),
        "v_rel_norm": vrel,
        "dz_safe_ex_mm": max_ex,
        "note": (
            "Scheduler must run dx first when lateral_ex_too_large; "
            "do not treat AXIS_V1/SUCCESS_NEW 12 mm as flip-safe."
        ),
    }


def correct_dz_inward(
    env, blob, cfg: PrecisionConfig, *, s1_state: dict | None = None
) -> dict[str, Any]:
    """Case B/C/D inward with support-aware UP/DOWN flips (no blind reverse replay).

    Does **not** run dx/support-transfer seating. Unsafe lateral offset → structured
    ``DZ_UNSAFE_ENTRY`` so the unified scheduler can correct dx first.
    """
    actions: list = []
    phase_log: list[dict] = []
    rotation_diag: list[dict] = []
    outcomes = _empty_dz_outcomes()
    restore_capture(env, blob["capture"])
    entry0 = check_dz_inward_entry(env.sim, cfg)
    phase_log.append({"phase": "entry_precondition", **{k: entry0[k] for k in entry0 if k != "errors"},
                      "errors": {kk: entry0["errors"].get(kk) for kk in (
                          "dx_mm", "dz_mm", "dtheta_deg", "e_pos_mm", "bilateral", "fn_L", "fn_R", "obj_z"
                      )}})
    if not entry0.get("ok"):
        return {
            "ok": False,
            "mechanism": "endpoint_only_inward",
            "branch": "precondition_reject",
            "failure_code": FAIL_DZ_PRECONDITION,
            "phase": "entry_precondition",
            "reason": ";".join(entry0.get("reasons") or ["unsafe_dz_entry"]),
            "entry_check": entry0,
            "actions": [],
            "capture": blob["capture"],
            "phase_log": phase_log,
            "outcomes": outcomes,
            "note": "Run dx (support-transfer) before inward dz; standalone probe must not bypass scheduler.",
        }

    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    roll_sign = float(
        (s1_state or {}).get("roll_sign")
        if (s1_state or {}).get("roll_sign") is not None
        else choose_roll_sign_for_support(env.sim)
    )
    branch = "B_full"
    reused_s1 = False
    side_hint = (s1_state or {}).get("support_side")
    up_tilt_actions: list = []
    up_axis_w = None
    up_sign = None

    if gates(env.sim)["fingertips_up"]:
        branch = "D_already_up"
        outcomes["FLIP_COMPLETED"] = True
        phase_log.append({"phase": "orientation", "skipped": True, "reason": "already_fingertips_up"})
    else:
        if _s1_valid(s1_state, env.sim):
            reused_s1 = True
            branch = "C_reuse_s1"
            side_hint = side_hint or (s1_state or {}).get("support_side")
            phase_log.append({
                "phase": "S1",
                "skipped": True,
                "reused": True,
                "s1": verify_mechanical_s1(env.sim, s1_state)["live"],
            })
        else:
            # Already have live mechanical S1 (e.g. gbi_063 at α≈159) — do not rebuild.
            live0 = verify_mechanical_s1(env.sim)
            if live0.get("ok"):
                side_hint = side_hint or live0["live"].get("support_side")
                phase_log.append({"phase": "S1", "skipped": True, "reused_live": True, "s1": live0["live"]})
            else:
                est = _prep_support_for_flip(env, cur, cfg, y0, e_ref, roll_sign, actions)
                phase_log.append({
                    "phase": "S1_establish",
                    "ok": est.get("ok"),
                    "reason": est.get("reason"),
                    "via": est.get("via"),
                })
                if not est.get("ok"):
                    return {
                        "ok": False,
                        "mechanism": "endpoint_only_inward",
                        "branch": branch,
                        "failure_code": est.get("failure_code") or FAIL_SUPPORT_ROLL,
                        "phase": "S1",
                        "reason": est.get("reason"),
                        "actions": actions,
                        "capture": est.get("capture") or cur["capture"],
                        "phase_log": phase_log,
                        "outcomes": outcomes,
                    }
                cur = {"capture": est["capture"], "ref": blob["ref"]}
                side_hint = side_hint or (est.get("s1") or {}).get("support_side")

        # Re-check after S1 prep (prep must not inflate |ex| into the unsafe band).
        restore_capture(env, cur["capture"])
        entry1 = check_dz_inward_entry(env.sim, cfg)
        phase_log.append({"phase": "pre_flip_precondition", "ok": entry1.get("ok"), "reasons": entry1.get("reasons"),
                          "ex_mm": entry1.get("ex_mm"), "ez_mm": entry1.get("ez_mm")})
        if not entry1.get("ok"):
            return {
                "ok": False,
                "mechanism": "endpoint_only_inward",
                "branch": branch,
                "failure_code": FAIL_DZ_PRECONDITION,
                "phase": "pre_flip_precondition",
                "reason": ";".join(entry1.get("reasons") or ["unsafe_dz_entry_before_flip"]),
                "entry_check": entry1,
                "actions": actions,
                "capture": cur["capture"],
                "phase_log": phase_log,
                "outcomes": outcomes,
                "reused_s1": reused_s1,
            }

        # Single-axis support-aware flip to UP from the *current* pose (no DOWN restore).
        p2 = _support_tilt(
            env, cur, UP, cfg, actions, "dzIn_to_UP",
            side_hint=side_hint, y0=y0, e_ref=e_ref,
        )
        rotation_diag.append({"phase": "to_UP", **{k: p2.get(k) for k in (
            "ok", "reason", "axis_w", "sign", "ang0_deg", "ang1_deg", "path_length_u",
            "sim_s", "selection", "planner",
        )}})
        phase_log.append({
            "phase": "to_UP",
            "ok": p2.get("ok"),
            "alpha": gates(env.sim)["alpha_up_deg"],
            "axis": p2.get("axis_w"),
            "sign": p2.get("sign"),
            "selection": (p2.get("selection") or {}).get("chosen_name"),
        })
        up_tol = float(getattr(cfg, "dz_up_tol_deg", 12.0) or 12.0)
        alpha_up = float(gates(env.sim)["alpha_up_deg"])
        g_sc = score_gravity_alignment(
            np.asarray(sample_state(env.sim)["g_h"], float), np.asarray(U_OPEN_HAND, float)
        )
        g_min = float(getattr(cfg, "dz_up_g_useful_min", 6.0) or 6.0)
        up_ok = bool(
            p2.get("ok")
            and (
                gates(env.sim)["fingertips_up"]
                or alpha_up <= up_tol
                or (alpha_up <= max(up_tol, 18.0) and float(g_sc.get("useful") or 0.0) >= g_min)
            )
        )
        phase_log[-1]["alpha"] = alpha_up
        phase_log[-1]["up_tol_deg"] = up_tol
        phase_log[-1]["g_useful"] = float(g_sc.get("useful") or 0.0)
        if not up_ok:
            outcomes["GRASP_RETAINED"] = bool(sample_state(env.sim).get("bilateral") and not latched(env.sim))
            return {
                "ok": False,
                "mechanism": "endpoint_only_inward",
                "branch": branch,
                "failure_code": FAIL_PREPARATION,
                "phase": "to_UP",
                "reason": p2.get("reason") or "support_lost_or_up_unreachable",
                "actions": actions,
                "capture": p2.get("capture") or cur["capture"],
                "phase_log": phase_log,
                "rotation_diag": rotation_diag,
                "reused_s1": reused_s1,
                "outcomes": outcomes,
                "alpha_up_deg": alpha_up,
            }
        outcomes["FLIP_COMPLETED"] = True
        cur = {"capture": p2["capture"], "ref": blob["ref"]}
        up_tilt_actions = list(p2.get("actions") or [])
        up_axis_w = p2.get("axis_w")
        up_sign = p2.get("sign")

    restore_capture(env, cur["capture"])
    hold = HoldModeGrip(fn_hold=14.0)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for _ in range(int(0.20 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    cur = {"capture": take_capture(env), "ref": blob["ref"]}
    e_x0 = float(sample_state(env.sim)["e_signed_mm"][0])
    ez0 = float(sample_state(env.sim)["e_signed_mm"][2])
    slip = _inward_slip_staged(
        env, cur, z_ref, e_x0, milestone_mm=float(cfg.inward_slip_mm), cfg=cfg,
    )
    actions += slip.get("actions") or []
    phase_log.append({
        "phase": "slip_inward",
        "ok": slip.get("ok"),
        "ez0": ez0,
        "disp_along_mm": slip.get("disp_along_mm"),
        "stop_reason": slip.get("stop_reason"),
        "mode": classify_contact_mode(env.sim, sliding="dz")["mode"],
    })
    if not slip.get("ok"):
        outcomes["GRASP_RETAINED"] = bool(sample_state(env.sim).get("bilateral") and not latched(env.sim))
        return {
            "ok": False,
            "mechanism": "endpoint_only_inward",
            "branch": branch,
            "phase": "slip",
            "reason": slip.get("reason") or slip.get("abort_reason") or slip.get("stop_reason") or "inward_slip_fail",
            "stop_reason": slip.get("stop_reason"),
            "actions": actions,
            "capture": slip.get("capture") or cur["capture"],
            "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
            "phase_log": phase_log,
            "rotation_diag": rotation_diag,
            "reused_s1": reused_s1,
            "outcomes": outcomes,
        }
    outcomes["DZ_SLIDE_COMPLETED"] = True
    cur = {"capture": slip["capture"], "ref": blob["ref"]}

    # Return to DOWN from the *actual* post-slip state.
    # Candidates (each verified from an identical capture): reverse-UP (if safe now),
    # support-aware fixed axis, then height-boosted restore_fingertips_down.
    restore_capture(env, cur["capture"])
    settle_hold(env, actions, hold_s=0.25)
    cur = {"capture": take_capture(env), "ref": blob["ref"]}
    if not _near_down(env.sim):
        v_ret = verify_mechanical_s1(env.sim)
        phase_log.append({"phase": "return_S1_check", "ok": v_ret.get("ok"), "reason": v_ret.get("reason")})
        side_hint = side_hint or (v_ret.get("live") or {}).get("support_side")
        p4: dict[str, Any] = {"ok": False, "reason": "no_return_candidate"}

        # 1) Preferred: same world axis as UP with opposite sign (single-axis return).
        # Empirically safe after axial slip; re-selection often picks a worse axis.
        if up_axis_w is not None and up_sign is not None:
            restore_capture(env, cur["capture"])
            cruise = float(getattr(cfg, "rate_tilt_cruise", 0.14) or 0.14) * 0.85
            peak = float(getattr(cfg, "rate_tilt_peak", 0.20) or 0.20) * 0.85
            floor = float(getattr(cfg, "rate_tilt_floor", 0.06) or 0.06)
            p_axis = run_fixed_axis_tilt(
                env,
                cur,
                target_dir_w=DOWN,
                axis_w=np.asarray(up_axis_w, float),
                sign=-float(up_sign),
                phase="dzIn_return_DOWN_neg_up_axis",
                rate_scale=cruise,
                fn_hold=16.0,
                z_ref_boost=0.12,
                height_kp=10.0,
                max_time_s=18.0,
                ang_tol_deg=8.0,
                pre_lift_s=float(getattr(cfg, "dz_return_pre_lift_s", 0.25) or 0.25),
                support_side_hint=side_hint,
                cruise=cruise,
                peak=peak,
                floor=floor,
                accel_s=float(getattr(cfg, "tilt_accel_s", 0.35) or 0.35),
                use_speed_profile=True,
            )
            actions += p_axis.get("actions") or []
            phase_log.append({
                "phase": "return_neg_up_axis",
                "ok": p_axis.get("ok"),
                "reason": p_axis.get("reason"),
                "axis": p_axis.get("axis_w"),
                "sign": p_axis.get("sign"),
            })
            if p_axis.get("ok") and _near_down(env.sim):
                p4 = {
                    **p_axis,
                    "planner": "negated_up_fixed_axis",
                    "selection": {
                        "chosen_name": "negated_up_axis",
                        "chosen_sign": -float(up_sign),
                        "chosen_axis": list(np.asarray(up_axis_w, float)),
                    },
                }
            else:
                # Discard failed attempt state.
                restore_capture(env, cur["capture"])
                # Drop failed actions from the tape.
                n_fail = len(p_axis.get("actions") or [])
                if n_fail > 0:
                    del actions[-n_fail:]

        # 2) Verified reverse of body-frame UP actions (only if still safe now).
        if not p4.get("ok"):
            rev = _try_reverse_up_actions(env, cur["capture"], up_tilt_actions)
            phase_log.append({
                "phase": "return_reverse_candidate",
                "ok": rev.get("ok"),
                "reason": rev.get("reason"),
                "n_up_actions": len(up_tilt_actions),
            })
            if rev.get("ok"):
                actions += rev.get("actions") or []
                p4 = {
                    "ok": True,
                    "reason": "ok",
                    "capture": rev["capture"],
                    "planner": rev.get("planner"),
                    "selection": {"chosen_name": "reverse_up_actions", "chosen_sign": -1.0},
                    "sim_s": len(rev.get("actions") or []) * DT,
                }

        # 3) Support-aware re-selection / restore fallbacks.
        if not p4.get("ok"):
            sel_down = select_support_aware_tilt_axis(
                env, cur["capture"], DOWN, support_side_hint=side_hint, horizon_s=1.6, rate_scale=0.045
            )
            if sel_down.get("ok"):
                p4 = _support_tilt(
                    env, cur, DOWN, cfg, actions, "dzIn_return_DOWN",
                    side_hint=side_hint, y0=y0, e_ref=e_ref,
                )
            else:
                p4 = {
                    "ok": False,
                    "reason": "no_safe_down_axis",
                    "selection": {
                        "ok": False,
                        "top_trials": sorted(sel_down.get("trials") or [], key=lambda t: t["cost"])[:6],
                        "chosen_name": (sel_down.get("selected") or {}).get("name"),
                    },
                }
            if not p4.get("ok"):
                _, restore_fingertips_down = _dx_support_transfer()
                restore_capture(env, cur["capture"])
                p4b = restore_fingertips_down(
                    env, cur, cfg, rate_scale=0.05,
                    z_ref_boost=0.18, height_kp=11.0, pre_lift_s=0.8, pre_lift_z=0.58,
                )
                actions += p4b.get("actions") or []
                p4 = {
                    "ok": p4b.get("ok"),
                    "reason": p4b.get("reason") or ("ok" if p4b.get("ok") else "restore_DOWN_failed"),
                    "capture": p4b.get("capture") or cur["capture"],
                    "planner": "restore_fingertips_down",
                    "selection": p4.get("selection"),
                    "fallback_of": p4.get("reason"),
                    "axis_w": None,
                    "sign": None,
                    "path_length_u": None,
                    "sim_s": len(p4b.get("actions") or []) * DT,
                }
        rotation_diag.append({"phase": "return_DOWN", **{k: p4.get(k) for k in (
            "ok", "reason", "axis_w", "sign", "ang0_deg", "ang1_deg", "path_length_u",
            "sim_s", "selection", "planner",
        )}})
        phase_log.append({
            "phase": "return_DOWN",
            "ok": p4.get("ok"),
            "reason": p4.get("reason"),
            "axis": p4.get("axis_w"),
            "sign": p4.get("sign"),
            "selection": (p4.get("selection") or {}).get("chosen_name"),
            "planner": p4.get("planner"),
        })
        if not p4.get("ok"):
            outcomes["GRASP_RETAINED"] = bool(sample_state(env.sim).get("bilateral") and not latched(env.sim))
            return {
                "ok": False,
                "mechanism": "endpoint_only_inward",
                "branch": branch,
                "phase": "return_DOWN",
                "reason": p4.get("reason") or "return_rotation_unsafe",
                "actions": actions,
                "capture": p4.get("capture") or cur["capture"],
                "phase_log": phase_log,
                "rotation_diag": rotation_diag,
                "reused_s1": reused_s1,
                "outcomes": outcomes,
            }
        outcomes["RETURN_DOWN_COMPLETED"] = True
        cur = {"capture": p4["capture"], "ref": blob["ref"]}
    else:
        outcomes["RETURN_DOWN_COMPLETED"] = True
        phase_log.append({"phase": "return_DOWN", "ok": True, "skipped": True, "reason": "already_near_down"})

    restore_capture(env, cur["capture"])
    settle_hold(env, actions, hold_s=0.35)
    err_exit = measure_errors(env.sim)
    outcomes["GRASP_RETAINED"] = bool(
        err_exit.get("bilateral") and not latched(env.sim) and float(err_exit.get("obj_z") or 0) >= Z_ABORT
    )
    # Primitive ok requires grasp retained — flip-alone is not recovery success.
    primitive_ok = bool(
        outcomes["FLIP_COMPLETED"]
        and outcomes["DZ_SLIDE_COMPLETED"]
        and outcomes["RETURN_DOWN_COMPLETED"]
        and outcomes["GRASP_RETAINED"]
    )
    return {
        "ok": primitive_ok,
        "mechanism": "endpoint_only_inward",
        "branch": branch,
        "reused_s1": reused_s1,
        "disp_along_mm": slip.get("disp_along_mm"),
        "disp_total_mm": slip.get("disp_total_mm"),
        "stop_reason": slip.get("stop_reason"),
        "goal_reached": slip.get("goal_reached"),
        "contract": slip.get("contract"),
        "actions": actions,
        "capture": take_capture(env),
        "errors": err_exit,
        "phase_log": phase_log,
        "rotation_diag": rotation_diag,
        "slip": {k: v for k, v in slip.items() if k not in ("actions", "series", "capture")},
        "fingertips_down": bool(gates(env.sim)["fingertips_down"] or _near_down(env.sim)),
        "outcomes": outcomes,
        "entry_check": entry0,
    }

