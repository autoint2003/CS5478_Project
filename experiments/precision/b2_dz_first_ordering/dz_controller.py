"""Ordering-experiment DZ: zero-target, no arbitrary MAX_SAFE_TRAVEL.

Stops on: deadband / velocity-aware brake / contact / lateral abort / no-progress / time budget.
Multiple bounded attempts allowed within the caller's action budget.
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
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import (
    A0_SLIP,
    FAIL_BRAKE,
    FAIL_REGRIP,
    FAIL_SLIP_CONTACT,
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
    STOP_NO_PROGRESS,
    STOP_TIME_BUDGET,
    STOP_WRONG_DIRECTION,
    V_SLIP,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal

DEADBAND_MM = 1.5
BRAKE_S = 0.45
ATTEMPT_TIME_S = 10.0
MAX_ATTEMPTS = 3


def _v_axial(st, d) -> float:
    v = np.asarray(st.get("v_rel_h") or [0, 0, 0], float)
    return float(np.dot(v, d)) * 1e3


def _one_slip(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    dz_target_mm: float,
    deadband_mm: float,
    max_time_s: float,
    timeline: list,
    t0: float,
) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if not err0.get("bilateral"):
        return {"ok": False, "stop_reason": STOP_CONTACT_SAFETY, "actions": [], "capture": blob["capture"], "errors0": err0, "errors1": err0}

    e0 = np.asarray(err0["e_signed_mm"], float)
    ez0 = float(e0[2])
    tgt = float(dz_target_mm)
    u = np.asarray(U_OPEN_HAND, float)
    u = u / float(np.linalg.norm(u))
    d = -u if ez0 >= tgt else u
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(e0[0])
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
    best = abs(ez0 - tgt)
    stall = 0
    t = float(t0)
    markers = {"dz_start": ez0, "dz_brake": None, "disp_mm": 0.0, "used_max_safe_travel": False}

    def disp():
        return float(np.dot(np.asarray(sample_state(env.sim)["p_c_h"], float) - p0, d)) * 1e3

    def log(phase, **kw):
        err = measure_errors(env.sim)
        st = sample_state(env.sim)
        timeline.append(
            {
                "t": t,
                "phase": phase,
                "dx_mm": float(err["dx_mm"]),
                "dy_mm": float(err["dy_mm"]),
                "dz_mm": float(err["dz_mm"]),
                "e_pos_mm": float(err["e_pos_mm"]),
                "dtheta_deg": float(err["dtheta_deg"]),
                "dz_target_mm": tgt,
                "v_axial_mm_s": _v_axial(st, d),
                "bilateral": bool(err.get("bilateral")),
                "fn_tot": float(err.get("fn_tot") or 0),
                "disp_along_mm": disp(),
                **kw,
            }
        )

    if abs(ez0 - tgt) <= deadband_mm:
        markers["dz_end"] = ez0
        return {
            "ok": True,
            "skipped": True,
            "stop_reason": STOP_GOAL_REACHED,
            "actions": [],
            "capture": take_capture(env),
            "errors0": err0,
            "errors1": err0,
            "markers": markers,
            "disp_along_mm": 0.0,
        }

    log("slip_entry")
    for i in range(int(max_time_s / DT)):
        err = measure_errors(env.sim)
        ez = float(err["dz_mm"])
        axis_err = abs(ez - tgt)
        st = sample_state(env.sim)
        v_ax = _v_axial(st, d)
        stop_dist = abs(v_ax) * BRAKE_S
        if axis_err <= deadband_mm:
            stop_reason = STOP_GOAL_REACHED
            markers["dz_brake"] = ez
            log("brake_trigger", brake_trigger=True)
            break
        if axis_err <= stop_dist + deadband_mm and disp() > 0.3 and ((ez - tgt) * v_ax < 0 or abs(v_ax) < 5):
            stop_reason = STOP_GOAL_REACHED
            markers["dz_brake"] = ez
            log("brake_trigger_vaware", brake_trigger=True, predicted_stop_dist_mm=stop_dist)
            break
        # NOTE: no MAX_SAFE_TRAVEL break — travel is diagnostic only
        if axis_err + 1e-6 < best - 0.05:
            best = axis_err
            stall = 0
        else:
            stall += 1
        if stall > int(2.5 / DT) and disp() > 0.5:
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
                "disp_along_mm": disp(),
            }
        if abs(float(s2["e_signed_mm"][0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            stop_reason = STOP_WRONG_DIRECTION
            break
    else:
        markers["dz_brake"] = float(measure_errors(env.sim)["dz_mm"])

    # brake/regrip
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
                    "failure_code": FAIL_BRAKE if j < n_b else FAIL_REGRIP,
                    "actions": actions,
                    "capture": take_capture(env),
                    "errors0": err0,
                    "errors1": measure_errors(env.sim),
                    "markers": markers,
                    "disp_along_mm": disp(),
                }

    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    for _ in range(int(0.2 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        t += DT

    err1 = measure_errors(env.sim)
    markers["dz_end"] = float(err1["dz_mm"])
    markers["disp_mm"] = disp()
    hit = abs(float(err1["dz_mm"]) - tgt) <= deadband_mm + 0.5
    improved = abs(float(err1["dz_mm"]) - tgt) < abs(ez0 - tgt) - 0.3
    safe = stop_reason not in (STOP_CONTACT_SAFETY, STOP_WRONG_DIRECTION)
    ok = bool(err1.get("bilateral") and safe and (hit or improved))
    log("slip_done", stop_reason=stop_reason)
    return {
        "ok": ok,
        "stop_reason": stop_reason,
        "goal_reached": hit,
        "actions": actions,
        "capture": take_capture(env),
        "errors0": err0,
        "errors1": err1,
        "markers": markers,
        "disp_along_mm": disp(),
        "t_end": t,
    }


def run_zero_target_dz(
    env,
    blob: dict,
    cfg: PrecisionConfig,
    *,
    dz_target_mm: float = 0.0,
    deadband_mm: float = DEADBAND_MM,
    max_attempts: int = MAX_ATTEMPTS,
    action_budget_left: int = 2000,
    timeline: list | None = None,
    label: str = "dz",
) -> dict[str, Any]:
    """Multi-attempt zero-target DZ without arbitrary cumulative travel limit."""
    tl = timeline if timeline is not None else []
    actions: list = []
    attempts = []
    cur = {"capture": blob["capture"], "ref": blob.get("ref")}
    t = 0.0
    total_disp = 0.0
    for k in range(int(max_attempts)):
        if len(actions) >= action_budget_left:
            break
        restore_capture(env, cur["capture"])
        err = measure_errors(env.sim)
        if abs(float(err["dz_mm"]) - dz_target_mm) <= deadband_mm:
            break
        if not err.get("bilateral"):
            return {
                "ok": False,
                "reason": "contact_lost_between_attempts",
                "actions": actions,
                "capture": cur["capture"],
                "attempts": attempts,
                "n_attempts": len(attempts),
                "total_disp_mm": total_disp,
                "timeline": tl,
                "label": label,
            }
        rem_time = min(ATTEMPT_TIME_S, 0.02 * max(1, action_budget_left - len(actions)))
        r = _one_slip(
            env,
            cur,
            cfg,
            dz_target_mm=dz_target_mm,
            deadband_mm=deadband_mm,
            max_time_s=rem_time,
            timeline=tl,
            t0=t,
        )
        acts = list(r.get("actions") or [])
        actions += acts
        total_disp += abs(float(r.get("disp_along_mm") or 0))
        attempts.append({k2: v for k2, v in r.items() if k2 not in ("actions", "timeline", "capture")})
        t = float(r.get("t_end") or t)
        cur = {"capture": r.get("capture") or take_capture(env), "ref": blob.get("ref")}
        if not r.get("ok"):
            return {
                "ok": False,
                "reason": r.get("reason") or r.get("stop_reason"),
                "stop_reason": r.get("stop_reason"),
                "actions": actions,
                "capture": cur["capture"],
                "attempts": attempts,
                "n_attempts": len(attempts),
                "total_disp_mm": total_disp,
                "timeline": tl,
                "errors1": r.get("errors1"),
                "label": label,
                "used_max_safe_travel": False,
            }
        if r.get("goal_reached") or r.get("skipped"):
            break
        # continue if no_progress but still far and contact ok
        if r.get("stop_reason") != STOP_NO_PROGRESS:
            break

    err1 = measure_errors(env.sim)
    hit = abs(float(err1["dz_mm"]) - dz_target_mm) <= deadband_mm + 0.5
    return {
        "ok": bool(err1.get("bilateral") and (hit or len(attempts) > 0)),
        "goal_reached": hit,
        "actions": actions,
        "capture": take_capture(env),
        "errors1": err1,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "total_disp_mm": total_disp,
        "timeline": tl,
        "label": label,
        "dz_target_mm": dz_target_mm,
        "used_max_safe_travel": False,
        "stop_reason": (attempts[-1].get("stop_reason") if attempts else STOP_GOAL_REACHED),
    }
