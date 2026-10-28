"""Experimental DZ with UP staging + direction-consistency (not production).

- Target dz→0; no MAX_SAFE_TRAVEL; no success-threshold skip.
- Abort if |error| grows persistently during slip; brake; do not reverse mid-motion.
- Multi-attempt within action budget on NO_PROGRESS / WRONG_DIRECTION only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.axial_common import _prep_support_for_flip, _support_tilt
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand
from data_generation.oracles.precision.mechanics.dual_loop import (
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    DT,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND
from data_generation.oracles.precision.mechanics.open_axis import (
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
)
from data_generation.oracles.precision.mechanics.protected_rotation import (
    run_fixed_axis_tilt,
    verify_mechanical_s1,
)
from data_generation.oracles.precision.mechanics.support_transfer import restore_fingertips_down
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import (
    A0_SLIP,
    FN_BRAKE,
    FN_INIT,
    FN_MAX,
    FN_MIN,
    FORCE_DA,
    FORCE_KP,
    LATERAL_DRIFT_ABORT_MM,
    PrecisionConfig,
    SLIP_KN,
    STOP_CONTACT_SAFETY,
    STOP_GOAL_REACHED,
    STOP_NO_PROGRESS,
    STOP_TIME_BUDGET,
    STOP_WRONG_DIRECTION,
    V_SLIP,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal

# Numerical resolution — not a SUCCESS_NEW / AXIS tolerance
EPS_DZ_MM = 1e-3
GROW_WINDOW_S = 0.40
BRAKE_S = 0.45
ATTEMPT_TIME_S = 10.0
MAX_ATTEMPTS = 4
STOP_ERROR_GROWTH = "ERROR_GROWTH_ABORT"


def _v_axial(st, d) -> float:
    v = np.asarray(st.get("v_rel_h") or [0, 0, 0], float)
    return float(np.dot(v, d)) * 1e3


def prepare_up_for_dz(env, blob: dict, cfg: PrecisionConfig, *, bypass_ex_gate: bool) -> dict[str, Any]:
    """Production-like UP staging. Optional bypass of |ex| entry gate for gate audit."""
    from data_generation.oracles.precision.corrections.axial_inward import check_dz_inward_entry

    actions: list = []
    restore_capture(env, blob["capture"])
    entry = check_dz_inward_entry(env.sim, cfg)
    if not entry.get("ok"):
        reasons = list(entry.get("reasons") or [])
        hard = [r for r in reasons if "lateral_ex_too_large" not in r]
        ex_gated = any("lateral_ex_too_large" in r for r in reasons)
        # Strict audit: optionally bypass only the |ex| gate; keep real contact/actuator reasons.
        if hard or (ex_gated and not bypass_ex_gate):
            return {
                "ok": False,
                "blocked": True,
                "block_reason": (
                    "DZ_BLOCKED_BY_EX_GATE"
                    if ex_gated and not hard
                    else "DZ_ENTRY_REJECT"
                ),
                "entry_reasons": reasons,
                "actions": [],
                "capture": blob["capture"],
            }

    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob.get("ref")}
    roll_sign = float(choose_roll_sign_for_support(env.sim))
    up_axis_w = None
    up_sign = None

    if gates(env.sim)["fingertips_up"]:
        return {
            "ok": True,
            "blocked": False,
            "already_up": True,
            "actions": [],
            "capture": blob["capture"],
            "up_axis_w": None,
            "up_sign": None,
            "entry_bypassed_ex": bool(not entry.get("ok")),
        }

    live0 = verify_mechanical_s1(env.sim)
    side_hint = (live0.get("live") or {}).get("support_side") if live0.get("ok") else None
    if not live0.get("ok"):
        est = _prep_support_for_flip(env, cur, cfg, y0, e_ref, roll_sign, actions)
        if not est.get("ok"):
            return {
                "ok": False,
                "blocked": True,
                "block_reason": "DZ_S1_PREP_FAILED",
                "reason": est.get("reason"),
                "actions": actions,
                "capture": take_capture(env),
            }
        cur = {"capture": est["capture"], "ref": blob.get("ref")}
        side_hint = side_hint or (est.get("s1") or {}).get("support_side")

    p2 = _support_tilt(env, cur, UP, cfg, actions, "strict_dz_to_UP", side_hint=side_hint, y0=y0, e_ref=e_ref)
    if not p2.get("ok"):
        return {
            "ok": False,
            "blocked": True,
            "block_reason": "DZ_UP_FAILED",
            "reason": p2.get("reason"),
            "actions": actions,
            "capture": p2.get("capture") or take_capture(env),
        }
    # `_support_tilt` already appended actions
    return {
        "ok": True,
        "blocked": False,
        "already_up": False,
        "actions": actions,
        "capture": p2["capture"],
        "up_axis_w": p2.get("axis_w"),
        "up_sign": p2.get("sign"),
        "side_hint": side_hint,
        "entry_bypassed_ex": bool(not entry.get("ok")),
        "entry_reasons": list(entry.get("reasons") or []),
    }


def _brake_regrip(env, z_ref, actions, timeline, t) -> tuple[bool, float]:
    ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
    ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
    n_b = int(BRAKE_S / DT)
    for j in range(n_b + int(0.25 / DT)):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
        if timeline is not None and j % 3 == 0:
            err = measure_errors(env.sim)
            timeline.append(
                {
                    "t": t,
                    "phase": "brake" if j < n_b else "regrip",
                    "dz_mm": float(err["dz_mm"]),
                    "abs_dz": abs(float(err["dz_mm"])),
                    "bilateral": bool(err.get("bilateral")),
                    "grip_cmd": float(a[6]),
                }
            )
        if not sample_state(env.sim)["bilateral"]:
            for _ in range(8):
                a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                step_legal(env.sim, a_em)
                actions.append(a_em.copy())
                t += DT
                if sample_state(env.sim)["bilateral"]:
                    break
            if not sample_state(env.sim)["bilateral"]:
                return False, t
    return True, t


def slip_toward_zero(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    timeline: list,
    t0: float = 0.0,
    max_time_s: float = ATTEMPT_TIME_S,
) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if not err0.get("bilateral"):
        return {"ok": False, "stop_reason": STOP_CONTACT_SAFETY, "actions": [], "capture": blob["capture"], "errors0": err0, "errors1": err0, "t_end": t0}

    ez0 = float(err0["dz_mm"])
    tgt = 0.0
    u = np.asarray(U_OPEN_HAND, float)
    u = u / float(np.linalg.norm(u))
    # After UP, production inward uses -U_OPEN for positive ez
    d = -u if ez0 >= tgt else u
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(err0["dx_mm"])
    p0 = np.asarray(sample_state(env.sim)["p_c_h"], float)
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=A0_SLIP, fn0=sample_state(env.sim)["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))

    actions: list = []
    lost = 0
    stop_reason = STOP_TIME_BUDGET
    t = float(t0)
    best_abs = abs(ez0 - tgt)
    grow_steps = 0
    stall_steps = 0
    markers = {
        "dz_start": ez0,
        "d": d.tolist(),
        "slip_axis": (-d).tolist(),
        "dz_brake": None,
        "disp_end": 0.0,
    }

    def disp():
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3

    def log(phase, **kw):
        err = measure_errors(env.sim)
        st = sample_state(env.sim)
        a_grip = float(actions[-1][6]) if actions else None
        timeline.append(
            {
                "t": t,
                "phase": phase,
                "dz_mm": float(err["dz_mm"]),
                "abs_dz": abs(float(err["dz_mm"])),
                "dx_mm": float(err["dx_mm"]),
                "e_pos_mm": float(err["e_pos_mm"]),
                "dtheta_deg": float(err["dtheta_deg"]),
                "v_axial_mm_s": _v_axial(st, d),
                "desired_d": d.tolist(),
                "disp_along_d_mm": disp(),
                "bilateral": bool(err.get("bilateral")),
                "grip_cmd": a_grip,
                "fn_tot": float(err.get("fn_tot") or 0),
                **kw,
            }
        )

    # No SUCCESS-tolerance skip: only skip if numerically ~0
    if abs(ez0 - tgt) <= EPS_DZ_MM:
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": "NUMERICALLY_ZERO",
            "actions": [],
            "capture": take_capture(env),
            "errors0": err0,
            "errors1": err0,
            "markers": markers,
            "t_end": t,
            "disp_along_mm": 0.0,
        }

    log("slip_entry", stop_reason=None)
    for i in range(int(max_time_s / DT)):
        err = measure_errors(env.sim)
        ez = float(err["dz_mm"])
        abs_err = abs(ez - tgt)
        st = sample_state(env.sim)
        v_ax = _v_axial(st, d)

        # Convergence to numerical/physical resolution
        if abs_err <= EPS_DZ_MM:
            stop_reason = STOP_GOAL_REACHED
            markers["dz_brake"] = ez
            log("brake_trigger", stop_reason=stop_reason)
            break

        # Velocity-aware approach (still require measurable progress toward 0)
        stop_dist = abs(v_ax) * BRAKE_S
        if abs_err <= max(EPS_DZ_MM, stop_dist) and disp() > 0.2 and (ez - tgt) * v_ax < 0:
            stop_reason = STOP_GOAL_REACHED
            markers["dz_brake"] = ez
            log("brake_trigger_vaware", stop_reason=stop_reason)
            break

        # Direction consistency: absolute error growing
        if abs_err > best_abs + 0.05:
            grow_steps += 1
        else:
            grow_steps = 0
        if abs_err + 1e-9 < best_abs - 0.02:
            best_abs = abs_err
            stall_steps = 0
        else:
            stall_steps += 1

        if grow_steps > int(GROW_WINDOW_S / DT):
            stop_reason = STOP_ERROR_GROWTH
            markers["dz_brake"] = ez
            log("error_growth_abort", stop_reason=stop_reason)
            break
        if stall_steps > int(2.5 / DT) and abs(disp()) > 0.3:
            stop_reason = STOP_NO_PROGRESS
            markers["dz_brake"] = ez
            break

        _, meta = ctl.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        if (i + 1) % 2 == 0:
            log("slip")
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {
                "ok": False,
                "stop_reason": STOP_CONTACT_SAFETY,
                "reason": "contact_loss",
                "actions": actions,
                "capture": take_capture(env),
                "errors0": err0,
                "errors1": measure_errors(env.sim),
                "markers": markers,
                "t_end": t,
                "disp_along_mm": disp(),
            }
        if abs(float(s2["e_signed_mm"][0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            stop_reason = STOP_WRONG_DIRECTION
            break
    else:
        markers["dz_brake"] = float(measure_errors(env.sim)["dz_mm"])

    ok_brake, t = _brake_regrip(env, z_ref, actions, timeline, t)
    if not ok_brake:
        return {
            "ok": False,
            "stop_reason": STOP_CONTACT_SAFETY,
            "reason": "loss_on_brake",
            "actions": actions,
            "capture": take_capture(env),
            "errors0": err0,
            "errors1": measure_errors(env.sim),
            "markers": markers,
            "t_end": t,
            "disp_along_mm": disp(),
        }

    err1 = measure_errors(env.sim)
    markers["dz_end"] = float(err1["dz_mm"])
    markers["disp_end"] = disp()
    improved = abs(float(err1["dz_mm"])) < abs(ez0) - 0.2
    safe = stop_reason not in (STOP_CONTACT_SAFETY,)
    # Do not claim success merely for deadband; require abs improvement or near-zero
    near0 = abs(float(err1["dz_mm"])) <= max(EPS_DZ_MM, 0.5)
    ok = bool(err1.get("bilateral") and safe and (near0 or improved) and stop_reason != STOP_ERROR_GROWTH)
    log("slip_done", stop_reason=stop_reason)
    return {
        "ok": ok,
        "stop_reason": stop_reason,
        "goal_reached": near0,
        "actions": actions,
        "capture": take_capture(env),
        "errors0": err0,
        "errors1": err1,
        "markers": markers,
        "t_end": t,
        "disp_along_mm": disp(),
        "improved_abs": improved,
    }


def run_strict_dz(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    bypass_ex_gate: bool = False,
    do_return_down: bool = True,
    action_budget_left: int = 3000,
    timeline: list | None = None,
    label: str = "dz",
) -> dict[str, Any]:
    tl = timeline if timeline is not None else []
    actions: list = []
    restore_capture(env, blob["capture"])
    ez0 = float(measure_errors(env.sim)["dz_mm"])
    if abs(ez0) <= EPS_DZ_MM:
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": "NUMERICALLY_ZERO",
            "actions": [],
            "capture": blob["capture"],
            "dz0": ez0,
            "dz1": ez0,
            "n_attempts": 0,
            "blocked": False,
            "axial_correction_occurred": False,
            "label": label,
        }

    prep = prepare_up_for_dz(env, blob, cfg, bypass_ex_gate=bypass_ex_gate)
    actions += list(prep.get("actions") or [])
    if prep.get("blocked") or not prep.get("ok"):
        return {
            "ok": False,
            "blocked": True,
            "block_reason": prep.get("block_reason") or "DZ_PREP_BLOCKED",
            "entry_reasons": prep.get("entry_reasons"),
            "actions": actions,
            "capture": prep.get("capture") or blob["capture"],
            "dz0": ez0,
            "dz1": ez0,
            "n_attempts": 0,
            "axial_correction_occurred": False,
            "label": label,
        }

    cur = {"capture": prep["capture"], "ref": blob.get("ref")}
    attempts = []
    total_disp = 0.0
    t = 0.0
    last = None
    for k in range(MAX_ATTEMPTS):
        if len(actions) >= action_budget_left:
            break
        rem_t = min(ATTEMPT_TIME_S, 0.02 * max(1, action_budget_left - len(actions)))
        r = slip_toward_zero(env, cur, cfg, timeline=tl, t0=t, max_time_s=rem_t)
        acts = list(r.get("actions") or [])
        actions += acts
        total_disp += abs(float(r.get("disp_along_mm") or 0))
        attempts.append({kk: vv for kk, vv in r.items() if kk not in ("actions", "capture")})
        t = float(r.get("t_end") or t)
        last = r
        cur = {"capture": r.get("capture") or take_capture(env), "ref": blob.get("ref")}
        if r.get("stop_reason") == STOP_CONTACT_SAFETY:
            break
        if r.get("goal_reached") or r.get("skipped"):
            break
        # Retry only after safe stop with wrong-way / no progress — reassess from current capture
        if r.get("stop_reason") not in (STOP_NO_PROGRESS, STOP_ERROR_GROWTH, STOP_WRONG_DIRECTION, STOP_TIME_BUDGET):
            break
        # settle briefly before reassessment (no direction reverse while moving)
        for _ in range(int(0.15 / DT)):
            a = make_action(_height_hold_v_hand(env.sim, float(env.sim.data.xpos[env.sim.ids.object_body][2])), [0, 0, 0], 0.9)
            step_legal(env.sim, a)
            actions.append(a.copy())

    dz1 = float(measure_errors(env.sim)["dz_mm"])
    rd = {"ok": True, "skipped": True}
    if do_return_down and sample_state(env.sim).get("bilateral") and not gates(env.sim).get("fingertips_down"):
        # Production-style return: negate the UP fixed axis, then boosted restore fallback.
        up_axis = prep.get("up_axis_w")
        up_sign = prep.get("up_sign")
        rd = {"ok": False}
        if up_axis is not None and up_sign is not None:
            cruise = float(getattr(cfg, "rate_tilt_cruise", 0.14) or 0.14) * 0.85
            peak = float(getattr(cfg, "rate_tilt_peak", 0.20) or 0.20) * 0.85
            floor = float(getattr(cfg, "rate_tilt_floor", 0.06) or 0.06)
            p_axis = run_fixed_axis_tilt(
                env,
                cur,
                target_dir_w=DOWN,
                axis_w=np.asarray(up_axis, float),
                sign=-float(up_sign),
                phase="strict_dz_return_DOWN_neg_up_axis",
                rate_scale=cruise,
                fn_hold=16.0,
                z_ref_boost=0.12,
                height_kp=10.0,
                max_time_s=18.0,
                ang_tol_deg=8.0,
                pre_lift_s=float(getattr(cfg, "dz_return_pre_lift_s", 0.25) or 0.25),
                support_side_hint=prep.get("side_hint"),
                cruise=cruise,
                peak=peak,
                floor=floor,
                accel_s=float(getattr(cfg, "tilt_accel_s", 0.35) or 0.35),
                use_speed_profile=True,
            )
            actions += list(p_axis.get("actions") or [])
            if p_axis.get("ok") and (
                gates(env.sim).get("fingertips_down") or float(gates(env.sim).get("alpha_up_deg") or 0) >= 160.0
            ):
                rd = p_axis
                cur = {"capture": p_axis["capture"], "ref": blob.get("ref")}
            else:
                restore_capture(env, cur["capture"])
                n_fail = len(p_axis.get("actions") or [])
                if n_fail > 0:
                    del actions[-n_fail:]
        if not rd.get("ok"):
            rd = restore_fingertips_down(
                env,
                cur,
                cfg,
                rate_scale=0.05,
                z_ref_boost=0.18,
                height_kp=11.0,
                pre_lift_s=0.8,
                pre_lift_z=0.58,
            )
            actions += list(rd.get("actions") or [])
            if rd.get("ok") and rd.get("capture") is not None:
                cur = {"capture": rd["capture"], "ref": blob.get("ref")}
        dz_after_return = float(measure_errors(env.sim)["dz_mm"])
    else:
        dz_after_return = dz1

    err_f = measure_errors(env.sim)
    axial = abs(total_disp) > 0.5 or abs(dz1 - ez0) > 0.5
    improved_final = abs(dz_after_return) < abs(ez0) - 0.3
    # DZ success = abs improvement (or near-zero) + bilateral + safe stop.
    # Return-DOWN is reported separately; ordering runner re-attempts DOWN before DX.
    last_stop = (last or {}).get("stop_reason")
    ok = bool(
        err_f.get("bilateral")
        and last_stop != STOP_CONTACT_SAFETY
        and last_stop != STOP_ERROR_GROWTH
        and (improved_final or abs(dz_after_return) <= 0.5)
    )
    return {
        "ok": ok,
        "blocked": False,
        "actions": actions,
        "capture": cur["capture"],
        "dz0": ez0,
        "dz1_pre_return": dz1,
        "dz1": dz_after_return,
        "n_attempts": len(attempts),
        "attempts": attempts,
        "total_disp_mm": total_disp,
        "stop_reason": last_stop,
        "axial_correction_occurred": axial,
        "improved_abs_after_return": improved_final,
        "entry_bypassed_ex": prep.get("entry_bypassed_ex"),
        "return_down_ok": bool(rd.get("ok")),
        "timeline": tl,
        "label": label,
        "errors_exit": err_f,
        "used_max_safe_travel": False,
    }
