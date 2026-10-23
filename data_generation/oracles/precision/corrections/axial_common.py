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
def _dx_support_transfer():
    from data_generation.oracles.precision.mechanics.support_transfer import (
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

