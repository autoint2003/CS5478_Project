"""Inward / outward axial dz with support-aware flips and Case A–D shortcuts."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.contact_modes import classify_contact_mode, make_s1_state
from data_generation.oracles.precision.endpoint_only_axial_roll import (
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
from data_generation.oracles.precision.finger_supported_vertical import U_OPEN_HAND
from data_generation.oracles.precision.force_slip_control import DT, make_action, sample_state
from data_generation.oracles.precision.orientation_planner import (
    _height_hold_v_hand,
    score_gravity_alignment,
)
from data_generation.oracles.precision.precision_audit import _failure_snapshot
from data_generation.oracles.precision.precision_dx import _horizontal_open_target
from data_generation.oracles.precision.precision_goals import measure_errors
from data_generation.oracles.precision.precision_slip import (
    _inward_slip_staged,
    _outward_slip_staged,
    settle_hold,
)
from data_generation.oracles.precision.precision_types import (
    FAIL_DZ_PRECONDITION,
    FAIL_PREPARATION,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_WRONG_DIR,
    FAIL_SUPPORT_ROLL,
    FN_HOLD,
    PrecisionConfig,
    Z_ABORT,
)
from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip
from data_generation.oracles.precision.support_aware_rotation import (
    plan_and_execute_support_tilt,
    run_fixed_axis_tilt,
    select_support_aware_tilt_axis,
    verify_mechanical_s1,
)
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal

# Lazy import helper to avoid circular import with support-transfer dx.
def _dx_support_transfer():
    from data_generation.oracles.precision.precision_dx_support_transfer import (
        correct_dx_support_transfer,
        restore_fingertips_down,
    )
    return correct_dx_support_transfer, restore_fingertips_down


def _rate(cfg: PrecisionConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default) or default)


def _try_reverse_up_actions(env, capture, up_actions: list, *, max_frac: float = 1.0) -> dict[str, Any]:
    """Closed-loop reverse of a recorded UP tilt — selected only if currently safe."""
    if not up_actions:
        return {"ok": False, "reason": "no_up_actions", "actions": [], "capture": capture}
    restore_capture(env, capture)
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    actions: list = []
    n = max(1, int(len(up_actions) * float(max_frac)))
    rev = list(reversed(up_actions[:n]))
    for a_prev in rev:
        a = np.asarray(a_prev, float).copy()
        # Negate body ω; keep grip from hold; boost world-z hold.
        if a.shape[0] >= 6:
            a[3:6] = -a[3:6]
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        v_h = _height_hold_v_hand(env.sim, max(z0, Z_ABORT + 0.12), kp=10.0, vmax=1.2)
        a[0:3] = v_h
        a[6] = a_g
        step_legal(env.sim, a.astype(np.float32))
        actions.append(a.astype(np.float32).copy())
        st2 = sample_state(env.sim)
        if not st2.get("bilateral") or latched(env.sim):
            restore_capture(env, capture)
            return {"ok": False, "reason": "contact_loss", "actions": [], "capture": capture}
        if float(env.sim.data.xpos[env.sim.ids.object_body][2]) < Z_ABORT:
            restore_capture(env, capture)
            return {"ok": False, "reason": "world_height", "actions": [], "capture": capture}
        if _near_down(env.sim):
            break
    ok = _near_down(env.sim) and bool(sample_state(env.sim).get("bilateral"))
    if not ok:
        restore_capture(env, capture)
        return {
            "ok": False,
            "reason": "reverse_did_not_reach_down",
            "actions": [],
            "capture": capture,
            "alpha": gates(env.sim)["alpha_up_deg"],
        }
    return {
        "ok": True,
        "reason": "ok",
        "actions": actions,
        "capture": take_capture(env),
        "planner": "reverse_up_actions_verified",
        "alpha": gates(env.sim)["alpha_up_deg"],
    }


def _near_down(sim) -> bool:
    g = gates(sim)
    return bool(g["fingertips_down"] or float(g["alpha_up_deg"]) >= 160.0)


def _s1_valid(s1_state: dict | None, sim) -> bool:
    """Live mechanical S1 only — stale flags / bilateral-only overrides rejected."""
    if not s1_state:
        return False
    v = verify_mechanical_s1(sim, s1_state)
    return bool(v.get("ok"))


def _prep_support_for_flip(env, cur, cfg, y0, e_ref, roll_sign: float, actions: list) -> dict[str, Any]:
    """Ensure a live mechanical S1 before a large flip.

    Prefer a shallow tip to α≈155–160° (gbi-like gravity-loaded S1) over a full
    horizontal tip. Full horizontal is used only if shallow S1 cannot be verified.
    """
    v0 = verify_mechanical_s1(env.sim)
    if v0.get("ok"):
        return {"ok": True, "capture": cur["capture"], "skipped": True, "s1": v0["live"]}

    rate_s1 = _rate(cfg, "rate_establish_s1", 0.14)
    if not _near_down(env.sim) and float(gates(env.sim)["alpha_up_deg"]) > 120.0:
        p0 = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="dzIn_prep_down", rate_scale=rate_s1,
            fn_hold=FN_HOLD, ang_tol_deg=8.0, y0_w=y0, e_ref=e_ref, allow_roll=False,
            contact_aware=True, max_time_s=12.0, z_ref_boost=0.05, height_kp=5.0,
        )
        actions += p0.get("actions") or []
        if not p0.get("ok"):
            return {
                "ok": False,
                "reason": p0.get("reason") or "cannot_reach_down_for_S1",
                "capture": p0.get("capture") or cur["capture"],
                "failure_code": FAIL_PREPARATION,
            }
        cur = {"capture": p0["capture"], "ref": cur["ref"]}

    # Modest bias roll (DX-1 style) — not a mandatory 90° antiparallel roll.
    s1_deg = float(getattr(cfg, "dx_s1_roll_deg", 12.0) or 12.0)
    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=s1_deg, phase="dzIn_S1_bias",
        rate_scale=rate_s1, fn_hold=FN_HOLD,
        require_down=bool(gates(env.sim)["fingertips_down"]),
        y0_w=y0, e_ref=e_ref,
    )
    if p1.get("ok"):
        actions += p1.get("actions") or []
        cur = {"capture": p1["capture"], "ref": cur["ref"]}

    v1 = verify_mechanical_s1(env.sim)
    if v1.get("ok"):
        return {"ok": True, "capture": cur["capture"], "s1": v1["live"], "via": "bias_roll"}

    # Shallow tip toward UP until gravity loads one finger (gbi-like α≈155–160).
    restore_capture(env, cur["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    shallow_actions: list = []
    shallow_fail = None
    for _ in range(int(16.0 / DT)):
        alpha = float(gates(env.sim)["alpha_up_deg"])
        if alpha <= 158.0 and verify_mechanical_s1(env.sim).get("ok"):
            break
        if alpha <= 150.0:
            break
        w_body, _, _ = tilt_omega_toward(env.sim, UP, 0.055)
        u = u_open_world(env.sim)
        ow = project_out_roll(np.array(env.sim.fsm.r_des, float).reshape(3, 3) @ w_body, u)
        w_body = world_omega_to_body(env.sim, ow)
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z0 + 0.06, kp=6.0), w_body, a_g)
        step_legal(env.sim, a)
        shallow_actions.append(a.copy())
        if not sample_state(env.sim).get("bilateral"):
            shallow_fail = "contact_loss"
            break
    actions += shallow_actions
    cur = {"capture": take_capture(env), "ref": cur["ref"]}
    v_shallow = verify_mechanical_s1(env.sim)
    if v_shallow.get("ok") and shallow_fail is None:
        return {
            "ok": True,
            "capture": cur["capture"],
            "s1": v_shallow["live"],
            "via": "shallow_s1_tip",
            "alpha_up_deg": gates(env.sim)["alpha_up_deg"],
        }

    # Fallback: tip to horizontal (DX-2) if shallow path did not yield S1.
    restore_capture(env, cur["capture"])
    alpha = float(gates(env.sim)["alpha_up_deg"])
    if alpha > 110.0 or alpha < 70.0:
        u_tgt = _horizontal_open_target(env.sim)
        p2 = run_tilt_phase(
            env, cur, target_dir_w=u_tgt, phase="dzIn_to_horizontal_S1",
            rate_scale=min(_rate(cfg, "rate_protected_s1", 0.08), 0.08),
            fn_hold=FN_HOLD, ang_tol_deg=12.0, y0_w=y0, e_ref=e_ref, allow_roll=False,
            contact_aware=True, max_time_s=18.0, z_ref_boost=0.05, height_kp=5.0,
        )
        if not p2.get("ok"):
            restore_capture(env, cur["capture"])
            p2 = run_tilt_phase(
                env, cur, target_dir_w=u_tgt, phase="dzIn_to_horizontal_S1_retry",
                rate_scale=0.05, fn_hold=FN_HOLD, ang_tol_deg=14.0, y0_w=y0, e_ref=e_ref,
                allow_roll=False, contact_aware=True, max_time_s=20.0, z_ref_boost=0.07, height_kp=5.5,
            )
        actions += p2.get("actions") or []
        if not p2.get("ok"):
            return {
                "ok": False,
                "reason": p2.get("reason") or "horizontal_s1_unreachable",
                "capture": p2.get("capture") or cur["capture"],
                "failure_code": FAIL_SUPPORT_ROLL,
            }
        cur = {"capture": p2["capture"], "ref": cur["ref"]}

    v2 = verify_mechanical_s1(env.sim)
    if not v2.get("ok"):
        return {
            "ok": False,
            "reason": v2.get("reason") or "mechanical_support_not_established",
            "capture": cur["capture"],
            "failure_code": FAIL_SUPPORT_ROLL,
            "mode": v2.get("mode"),
        }
    return {"ok": True, "capture": cur["capture"], "s1": v2["live"], "via": "horizontal_s1"}


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


def _empty_dz_outcomes() -> dict[str, bool | None]:
    return {
        "FLIP_COMPLETED": False,
        "DZ_SLIDE_COMPLETED": False,
        "GRASP_RETAINED": False,
        "RETURN_DOWN_COMPLETED": False,
        "HANDOFF_COMPLETED": None,
        "SUCCESS_12S": None,
        "SUCCESS_NEW": None,
    }


def _support_tilt(env, cur, target, cfg, actions, phase: str, *, side_hint=None, y0=None, e_ref=None) -> dict[str, Any]:
    """Fixed-axis support-aware tilt; adaptive contact-aware fallback if no safe axis."""
    p = plan_and_execute_support_tilt(
        env, cur, target, phase=phase, cfg=cfg,
        support_side_hint=side_hint,
        rate_scale=_rate(cfg, "rate_tilt_cruise", _rate(cfg, "rate_protected_s1", 0.14)),
        require_mechanical_s1=False,
    )
    if p.get("ok"):
        actions += p.get("actions") or []
        return p

    # Justified multi-segment fallback: adaptive ω toward target under contact monitoring.
    # Used when no single fixed axis remains support-safe in short rollouts.
    restore_capture(env, cur["capture"])
    e0 = np.asarray(e_ref if e_ref is not None else sample_state(env.sim)["e_signed_mm"], float)
    returning = float(np.dot(np.asarray(target, float).reshape(3), DOWN)) > 0.9
    fb_rate = _rate(cfg, "rate_tilt_cruise", _rate(cfg, "rate_protected_s1", 0.12))
    fb_rate = min(fb_rate, 0.14) * (0.75 if returning else 1.0)
    p2 = run_tilt_phase(
        env, cur, target_dir_w=target, phase=phase + "_adaptive_fallback",
        rate_scale=fb_rate,
        fn_hold=FN_HOLD, ang_tol_deg=7.0, y0_w=y0, e_ref=e0, allow_roll=False,
        contact_aware=True, max_time_s=18.0,
        z_ref_boost=0.12 if returning else 0.08,
        height_kp=8.5 if returning else 6.5,
    )
    actions += p2.get("actions") or []
    p2["selection"] = p.get("selection")
    p2["planner"] = "adaptive_tilt_fallback"
    p2["fallback_of"] = p.get("reason")
    # Synthesize axis diagnostics from net u motion when available.
    if p2.get("ok"):
        p2["axis_w"] = (p.get("selection") or {}).get("chosen_axis")
        p2["sign"] = (p.get("selection") or {}).get("chosen_sign")
    return p2


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
