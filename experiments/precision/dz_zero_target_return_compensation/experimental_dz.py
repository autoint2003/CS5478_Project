"""Experimental inward DZ slip: zero-target (V1) and return-compensated (V2).

Does not modify production ``_inward_slip_staged`` / ``correct_dz_inward``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from data_generation.oracles.precision.mechanics.support_transfer import restore_fingertips_down
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import (
    A0_SLIP,
    FAIL_BRAKE,
    FAIL_REGRIP,
    FAIL_SLIP_CONTACT,
    FAIL_SLIP_TARGET,
    FN_BRAKE,
    FN_HOLD,
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
    STOP_MAX_SAFE_TRAVEL,
    STOP_NO_PROGRESS,
    STOP_TIME_BUDGET,
    STOP_WRONG_DIRECTION,
    V_SLIP,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal

# Experimental defaults (reported vs production inward_slip_mm=12)
EXP_MAX_SAFE_TRAVEL_MM = 20.0
EXP_DEADBAND_MM = 1.5
EXP_BRAKE_S = 0.45
EXP_REGRIP_S = 0.25


def _axial_speed_mm_s(st: dict, d_hand: np.ndarray) -> float:
    """Signed relative COM speed along desire direction (mm/s), hand frame."""
    v = np.asarray(st.get("v_rel_h") or [0, 0, 0], float)
    return float(np.dot(v, d_hand)) * 1e3


def experimental_inward_slip(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    dz_target_mm: float = 0.0,
    deadband_mm: float = EXP_DEADBAND_MM,
    max_safe_travel_mm: float = EXP_MAX_SAFE_TRAVEL_MM,
    max_duration_s: float = 12.0,
    velocity_aware_brake: bool = True,
    timeline: list | None = None,
) -> dict[str, Any]:
    """Drive signed ez toward ``dz_target_mm`` with small deadband; optional v-aware brake.

    Desire direction is chosen to reduce (ez - target): if ez > target, slip along -u_open
    (inward production direction); if ez < target, slip along +u_open.
    """
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if not err0.get("bilateral"):
        return {
            "ok": False,
            "reason": "no_bilateral",
            "stop_reason": STOP_CONTACT_SAFETY,
            "actions": [],
            "capture": blob["capture"],
            "timeline": timeline or [],
            "errors0": err0,
            "errors1": err0,
        }

    e0 = np.asarray(err0["e_signed_mm"], float)
    ez0 = float(e0[2])
    tgt = float(dz_target_mm)
    # Unit desire for COM motion that reduces (ez - tgt)
    # Production inward uses -U_OPEN to reduce positive ez (object proud along +open).
    u_open = np.asarray(U_OPEN_HAND, float)
    u_open = u_open / float(np.linalg.norm(u_open))
    if ez0 >= tgt:
        d = -u_open  # inward
        slip_sense = "inward"
    else:
        d = u_open  # outward toward target
        slip_sense = "outward"

    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(e0[0])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=A0_SLIP, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))

    actions: list = []
    series = timeline if timeline is not None else []
    lost = 0
    stop_reason = STOP_TIME_BUDGET
    best_err = abs(ez0 - tgt)
    stall = 0
    t = 0.0
    markers: dict[str, Any] = {
        "dz_target_mm": tgt,
        "deadband_mm": float(deadband_mm),
        "slip_sense": slip_sense,
        "dz_at_slip_start": ez0,
        "dz_at_brake_start": None,
        "dz_after_slip_stop": None,
        "brake_mode": None,
        "max_safe_travel_mm": float(max_safe_travel_mm),
    }

    def _disp() -> float:
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3

    def _log(phase: str, **extra):
        err = measure_errors(env.sim)
        st = sample_state(env.sim)
        row = {
            "t": t,
            "phase": phase,
            "dx_mm": float(err["dx_mm"]),
            "dy_mm": float(err["dy_mm"]),
            "dz_mm": float(err["dz_mm"]),
            "e_pos_mm": float(err["e_pos_mm"]),
            "dtheta_deg": float(err["dtheta_deg"]),
            "dz_target_mm": tgt,
            "axis_err_mm": abs(float(err["dz_mm"]) - tgt),
            "v_axial_mm_s": _axial_speed_mm_s(st, d),
            "bilateral": bool(err.get("bilateral")),
            "fn_tot": float(err.get("fn_tot") or 0),
            "disp_along_mm": _disp(),
            **extra,
        }
        series.append(row)

    _log("slip_entry")
    if abs(ez0 - tgt) <= float(deadband_mm):
        markers["dz_after_slip_stop"] = ez0
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": STOP_GOAL_REACHED,
            "actions": [],
            "capture": take_capture(env),
            "errors0": err0,
            "errors1": err0,
            "markers": markers,
            "timeline": series,
            "goal_reached": True,
        }

    max_steps = int(float(max_duration_s) / DT)
    brake_started = False
    for i in range(max_steps):
        err = measure_errors(env.sim)
        ez = float(err["dz_mm"])
        axis_err = abs(ez - tgt)
        st = sample_state(env.sim)
        v_ax = _axial_speed_mm_s(st, d)
        # Stopping distance estimate (mm): |v| * brake_duration (crude, legal)
        stop_dist = abs(v_ax) * EXP_BRAKE_S
        # Convergence: within deadband
        if axis_err <= float(deadband_mm):
            stop_reason = STOP_GOAL_REACHED
            brake_started = True
            markers["brake_mode"] = "deadband_reached"
            markers["dz_at_brake_start"] = ez
            _log("brake_trigger", stop_reason=stop_reason, brake_trigger=True)
            break
        # Velocity-aware: start brake when remaining error ≈ stopping distance
        if velocity_aware_brake and axis_err <= stop_dist + float(deadband_mm) and axis_err <= max(3.0, stop_dist + 1.0):
            # Only trigger if we're moving toward the target
            toward = (ez - tgt) * v_ax < 0 or abs(v_ax) < 5.0
            if toward and _disp() > 0.5:
                stop_reason = STOP_GOAL_REACHED
                brake_started = True
                markers["brake_mode"] = "velocity_aware"
                markers["dz_at_brake_start"] = ez
                markers["predicted_stop_dist_mm"] = stop_dist
                _log("brake_trigger", stop_reason=stop_reason, brake_trigger=True, predicted_stop_dist_mm=stop_dist)
                break

        if _disp() >= float(max_safe_travel_mm):
            stop_reason = STOP_MAX_SAFE_TRAVEL
            markers["dz_at_brake_start"] = ez
            markers["brake_mode"] = "max_safe_travel"
            break
        if _disp() < -0.5:
            stop_reason = STOP_WRONG_DIRECTION
            break
        if axis_err + 1e-6 < best_err - 0.05:
            best_err = axis_err
            stall = 0
        else:
            stall += 1
        if stall > int(2.5 / DT) and _disp() > 0.8:
            stop_reason = STOP_NO_PROGRESS
            markers["dz_at_brake_start"] = ez
            markers["brake_mode"] = "no_progress"
            break

        _, meta = ctl.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        if (i + 1) % 2 == 0:
            _log("slip", grip_cmd=float(a[6]))
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
                "timeline": series,
            }
        if abs(float(s2["e_signed_mm"][0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            stop_reason = STOP_WRONG_DIRECTION
            break
    else:
        markers["dz_at_brake_start"] = float(measure_errors(env.sim)["dz_mm"])
        markers["brake_mode"] = "time_budget"

    if markers["dz_at_brake_start"] is None:
        markers["dz_at_brake_start"] = float(measure_errors(env.sim)["dz_mm"])

    # Brake + regrip (same legal structure as production)
    ctl_b = ForcePController(kp=0.22, da_max=0.25, a_min=-0.5, a_max=1.0)
    ctl_b.reset(0.95, sample_state(env.sim)["fn_tot"])
    n_brake = int(EXP_BRAKE_S / DT)
    n_hold = int(EXP_REGRIP_S / DT)
    for j in range(n_brake + n_hold):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], max(FN_BRAKE, 14.0))
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT
        if j % 3 == 0:
            _log("brake" if j < n_brake else "regrip", brake_trigger=True)
        if not sample_state(env.sim)["bilateral"]:
            for _ in range(8):
                a_em = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], 1.0)
                step_legal(env.sim, a_em)
                actions.append(a_em.copy())
                t += DT
                if sample_state(env.sim)["bilateral"]:
                    break
            if not sample_state(env.sim)["bilateral"]:
                return {
                    "ok": False,
                    "stop_reason": STOP_CONTACT_SAFETY,
                    "reason": "loss_on_brake",
                    "failure_code": FAIL_BRAKE if j < n_brake else FAIL_REGRIP,
                    "actions": actions,
                    "capture": take_capture(env),
                    "errors0": err0,
                    "errors1": measure_errors(env.sim),
                    "markers": markers,
                    "timeline": series,
                }

    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT

    err1 = measure_errors(env.sim)
    markers["dz_after_slip_stop"] = float(err1["dz_mm"])
    _log("slip_done", stop_reason=stop_reason)
    hit = abs(float(err1["dz_mm"]) - tgt) <= float(deadband_mm) + 0.5
    improved = abs(float(err1["dz_mm"]) - tgt) < abs(ez0 - tgt) - 0.3
    safe = stop_reason not in (STOP_CONTACT_SAFETY, STOP_WRONG_DIRECTION)
    ok = bool(err1.get("bilateral") and safe and (hit or improved or stop_reason == STOP_MAX_SAFE_TRAVEL and improved))
    return {
        "ok": ok,
        "stop_reason": stop_reason,
        "goal_reached": hit,
        "actions": actions,
        "capture": take_capture(env),
        "errors0": err0,
        "errors1": err1,
        "markers": markers,
        "timeline": series,
        "disp_along_mm": _disp(),
        "failure_code": None if ok else FAIL_SLIP_TARGET,
        "contract": "experimental_zero_target",
    }


def probe_return_dz_drift(env, blob: dict, cfg: PrecisionConfig) -> dict[str, Any]:
    """Counterfactual return probe from current capture; restores afterward.

    Uses only information available at decision time (simulate then restore).
    Does not append probe actions to the caller tape.
    """
    pre = take_capture(env)
    restore_capture(env, pre)
    err_pre = measure_errors(env.sim)
    dz_pre = float(err_pre["dz_mm"])
    rd = restore_fingertips_down(
        env, {"capture": pre, "ref": blob.get("ref")}, cfg, rate_scale=float(cfg.rate_restore_down)
    )
    if not rd.get("ok") or not measure_errors(env.sim).get("bilateral"):
        restore_capture(env, pre)
        return {
            "ok": False,
            "reliable": False,
            "dz_pre": dz_pre,
            "dz_post": None,
            "drift_mm": None,
            "reason": rd.get("reason") or "probe_return_failed",
        }
    err_post = measure_errors(env.sim)
    dz_post = float(err_post["dz_mm"])
    drift = dz_post - dz_pre
    restore_capture(env, pre)
    # Reliability: bilateral retained and |drift| not pathological
    reliable = bool(err_post.get("bilateral") and abs(drift) < 25.0)
    return {
        "ok": True,
        "reliable": reliable,
        "dz_pre": dz_pre,
        "dz_post": dz_post,
        "drift_mm": float(drift),
        "reason": None if reliable else "unreliable_drift_magnitude",
    }


def experimental_inward_slip_v2(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    deadband_mm: float = EXP_DEADBAND_MM,
    max_safe_travel_mm: float = EXP_MAX_SAFE_TRAVEL_MM,
    timeline: list | None = None,
) -> dict[str, Any]:
    """Zero-target first, then decision-time return probe → signed pre-return target."""
    tl = timeline if timeline is not None else []
    # Pass A: toward 0
    r0 = experimental_inward_slip(
        env,
        blob,
        cfg,
        dz_target_mm=0.0,
        deadband_mm=deadband_mm,
        max_safe_travel_mm=max_safe_travel_mm,
        velocity_aware_brake=True,
        timeline=tl,
    )
    actions = list(r0.get("actions") or [])
    if not r0.get("ok") or not r0.get("capture"):
        r0["variant"] = "V2"
        r0["compensation_applied"] = False
        return r0

    restore_capture(env, r0["capture"])
    probe = probe_return_dz_drift(env, {"capture": r0["capture"], "ref": blob.get("ref")}, cfg)
    markers = dict(r0.get("markers") or {})
    markers["return_probe"] = {k: v for k, v in probe.items()}
    predicted = probe.get("drift_mm")
    compensation_applied = False
    pre_return_target = 0.0

    if probe.get("reliable") and predicted is not None and abs(float(predicted)) >= 0.5:
        # Desired: ez_pre + predicted ≈ 0  =>  ez_pre_target = -predicted
        pre_return_target = float(-predicted)
        # Bound compensation to ±8 mm (contact/slip safety)
        pre_return_target = float(np.clip(pre_return_target, -8.0, 8.0))
        ez_now = float(measure_errors(env.sim)["dz_mm"])
        if abs(ez_now - pre_return_target) > deadband_mm:
            travel_left = max(2.0, float(max_safe_travel_mm) - abs(float(r0.get("disp_along_mm") or 0.0)))
            r1 = experimental_inward_slip(
                env,
                {"capture": take_capture(env), "ref": blob.get("ref")},
                cfg,
                dz_target_mm=pre_return_target,
                deadband_mm=deadband_mm,
                max_safe_travel_mm=travel_left,
                velocity_aware_brake=True,
                timeline=tl,
            )
            actions += list(r1.get("actions") or [])
            compensation_applied = True
            markers["compensation_slip"] = {k: v for k, v in (r1.get("markers") or {}).items()}
            if not r1.get("ok"):
                # Reject compensation that loses grasp — restore to post-r0
                restore_capture(env, r0["capture"])
                compensation_applied = False
                markers["compensation_rejected"] = r1.get("reason") or r1.get("stop_reason")
                err1 = measure_errors(env.sim)
                return {
                    **{k: v for k, v in r0.items() if k not in ("actions", "capture", "timeline")},
                    "ok": bool(r0.get("ok")),
                    "actions": list(r0.get("actions") or []),
                    "capture": r0["capture"],
                    "timeline": tl,
                    "markers": markers,
                    "compensation_applied": False,
                    "predicted_return_drift_mm": float(predicted),
                    "pre_return_target_mm": pre_return_target,
                    "variant": "V2",
                    "errors1": err1,
                    "errors": err1,
                }
            final_cap = r1.get("capture") or take_capture(env)
            err1 = r1.get("errors1") or measure_errors(env.sim)
            markers["dz_after_slip_stop"] = float(err1["dz_mm"])
            return {
                "ok": bool(r1.get("ok")),
                "stop_reason": r1.get("stop_reason"),
                "goal_reached": bool(r1.get("goal_reached")),
                "actions": actions,
                "capture": final_cap,
                "errors0": r0.get("errors0"),
                "errors1": err1,
                "errors": err1,
                "markers": markers,
                "timeline": tl,
                "disp_along_mm": float(r0.get("disp_along_mm") or 0) + float(r1.get("disp_along_mm") or 0),
                "disp_total_mm": float(r0.get("disp_along_mm") or 0) + float(r1.get("disp_along_mm") or 0),
                "compensation_applied": True,
                "predicted_return_drift_mm": float(predicted),
                "pre_return_target_mm": pre_return_target,
                "variant": "V2",
                "contract": "experimental_return_compensated",
                "reached": bool(r1.get("goal_reached")),
                "brake_ok": True,
                "regrip_ok": True,
            }
    elif not probe.get("reliable"):
        markers["compensation_fallback"] = "unreliable_probe_use_zero_target"

    err1 = measure_errors(env.sim)
    return {
        **{k: v for k, v in r0.items() if k not in ("actions", "timeline", "markers")},
        "actions": actions,
        "capture": take_capture(env),
        "timeline": tl,
        "markers": markers,
        "compensation_applied": compensation_applied,
        "predicted_return_drift_mm": None if predicted is None else float(predicted),
        "pre_return_target_mm": pre_return_target,
        "variant": "V2",
        "errors1": err1,
        "errors": err1,
        "disp_total_mm": r0.get("disp_along_mm"),
        "reached": bool(r0.get("goal_reached")),
        "brake_ok": True,
        "regrip_ok": True,
    }


def patched_inward_slip_factory(variant: str, ctx: dict):
    """Return a drop-in replacement for ``_inward_slip_staged``."""

    def _patched(env, blob, z_ref, e_x0, *, milestone_mm: float = 12.0, cfg=None):
        cfg = cfg or PrecisionConfig()
        tl: list = ctx.setdefault("timeline", [])
        if variant == "V2":
            res = experimental_inward_slip_v2(
                env,
                blob,
                cfg,
                deadband_mm=float(ctx.get("deadband_mm", EXP_DEADBAND_MM)),
                max_safe_travel_mm=float(ctx.get("max_safe_travel_mm", EXP_MAX_SAFE_TRAVEL_MM)),
                timeline=tl,
            )
        else:
            # V1
            res = experimental_inward_slip(
                env,
                blob,
                cfg,
                dz_target_mm=float(ctx.get("dz_target_mm", 0.0)),
                deadband_mm=float(ctx.get("deadband_mm", EXP_DEADBAND_MM)),
                max_safe_travel_mm=float(ctx.get("max_safe_travel_mm", EXP_MAX_SAFE_TRAVEL_MM)),
                velocity_aware_brake=True,
                timeline=tl,
            )
            res["compensation_applied"] = False
            res["variant"] = "V1"

        res["disp_total_mm"] = res.get("disp_total_mm", res.get("disp_along_mm"))
        res["reached"] = bool(res.get("goal_reached") or res.get("reached"))
        res["brake_ok"] = True
        res["regrip_ok"] = True
        res["errors"] = res.get("errors1") or res.get("errors")
        ctx["last_slip"] = {k: v for k, v in res.items() if k not in ("actions", "timeline", "capture")}
        return res

    return _patched
