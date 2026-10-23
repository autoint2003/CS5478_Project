"""Diagnostic SUCCESS_AXIS_V1 — does NOT modify SUCCESS_NEW / SUCCESS_12S.

Offline / reporting only. Same 12 s nominal + final-window 0.5 s dwell as SUCCESS_NEW,
but position limits are axis-separated (lateral vs axial).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import (
    LOST_STEPS,
    NOMINAL_S,
    _bilateral,
    _dropped,
    _escaped,
    _monitor,
)
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.support_drop import latched
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import (
    FINAL_WINDOW_S,
    HOLD_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)

# Provisional candidate limits (experimental — not a replacement for SUCCESS_NEW)
AXIS_V1_EX_MM = 12.0
AXIS_V1_EY_MM = 5.0
AXIS_V1_EZ_MM = 20.0  # provisional; sensitivity over {12,15,20,25}
AXIS_V1_THETA_DEG = 10.0
Z_LIMIT_CANDIDATES_MM = (12.0, 15.0, 20.0, 25.0)


def axis_limits_ok(
    e_x: float,
    e_y: float,
    e_z: float,
    theta: float,
    *,
    ex_mm: float = AXIS_V1_EX_MM,
    ey_mm: float = AXIS_V1_EY_MM,
    ez_mm: float = AXIS_V1_EZ_MM,
    theta_deg: float = AXIS_V1_THETA_DEG,
) -> bool:
    return (
        abs(float(e_x)) <= ex_mm + 1e-9
        and abs(float(e_y)) <= ey_mm + 1e-9
        and abs(float(e_z)) <= ez_mm + 1e-9
        and float(theta) <= theta_deg + 1e-9
    )


def longest_axis_ok_s(
    samples: list[dict],
    *,
    window_start: float,
    ez_mm: float,
    ex_mm: float = AXIS_V1_EX_MM,
    ey_mm: float = AXIS_V1_EY_MM,
    theta_deg: float = AXIS_V1_THETA_DEG,
    hold_s: float = HOLD_S,
) -> dict[str, Any]:
    """Longest contiguous dwell in final window under axis-separated limits."""
    win = [s for s in samples if s["t_rel"] + 1e-9 >= window_start]
    empty = {
        "n": 0,
        "longest_ok_s": 0.0,
        "SUCCESS_AXIS_V1_hold": False,
        "frac_ok": 0.0,
        "ez_mm": ez_mm,
        "ex_mm": ex_mm,
        "ey_mm": ey_mm,
        "theta_deg": theta_deg,
    }
    if not win:
        return empty
    dt = 0.02
    if len(win) > 1:
        dts = np.diff([s["t_rel"] for s in win])
        if len(dts):
            dt = float(np.median(dts))
    longest = cur = 0.0
    n_ok = 0
    for s in win:
        e = s.get("e_signed_mm")
        if e is None:
            e = [s.get("e_x_mm", 1e3), s.get("e_y_mm", 1e3), s.get("e_z_mm", 1e3)]
        ok = axis_limits_ok(e[0], e[1], e[2], s["e_theta_deg"], ex_mm=ex_mm, ey_mm=ey_mm, ez_mm=ez_mm, theta_deg=theta_deg)
        if ok:
            cur += dt
            longest = max(longest, cur)
            n_ok += 1
        else:
            cur = 0.0
    return {
        "n": len(win),
        "longest_ok_s": float(longest),
        "SUCCESS_AXIS_V1_hold": bool(longest + 1e-9 >= hold_s),
        "frac_ok": float(n_ok / max(len(win), 1)),
        "ez_mm": ez_mm,
        "ex_mm": ex_mm,
        "ey_mm": ey_mm,
        "theta_deg": theta_deg,
        "final_e_signed_mm": list(win[-1].get("e_signed_mm") or []),
        "final_theta_deg": float(win[-1]["e_theta_deg"]),
        "final_e_pos_mm": float(win[-1]["e_pos_mm"]),
        "frac_bilateral": float(np.mean([1.0 if s.get("bilateral") else 0.0 for s in win])),
    }


def sensitivity_over_z_limits(samples: list[dict], *, s12: bool, window_start: float | None = None) -> dict[str, Any]:
    if window_start is None:
        window_start = float(NOMINAL_S) - FINAL_WINDOW_S
    out = {}
    for ez in Z_LIMIT_CANDIDATES_MM:
        st = longest_axis_ok_s(samples, window_start=window_start, ez_mm=ez)
        out[f"ez_{int(ez)}"] = {
            **st,
            "SUCCESS_AXIS_V1": bool(s12 and st["SUCCESS_AXIS_V1_hold"]),
        }
    return out


def evaluate_handoff_with_axis(
    sim,
    ref,
    *,
    stabilize_n: int = 0,
    hold_action_fn=None,
) -> dict[str, Any]:
    """Legal HANDOFF + 12 s nominal; report SUCCESS_NEW and SUCCESS_AXIS_V1 (provisional).

    Does not alter SUCCESS_NEW computation: Euclidean WORKING_POS_MM still used for that flag.
    """
    from data_generation.oracles.precision.corrections.contact_primitives import _wrist

    if hold_action_fn is None:
        hold_action_fn = lambda g=1.0: _wrist(0.0, 0.0, g)

    for _ in range(int(stabilize_n)):
        step_legal(sim, hold_action_fn(1.0))
        if latched(sim):
            break

    m0 = measure(sim, U_TGT, P_TGT)
    e0 = (np.asarray(m0["p_c_h"], float) - np.asarray(P_TGT, float)) * 1e3
    pb0 = contact_summary(sim)
    apply_handoff(sim, ref)
    t_h = float(sim.data.time)
    reason = None
    samples: list[dict] = []
    next_sample = 0.0
    lost = 0
    lift_done = False
    min_obj_z = float(m0.get("obj_z") or sim.data.xpos[sim.ids.object_body][2])
    slip_events = 0
    delayed_drop = False

    def _row():
        m = measure(sim, U_TGT, P_TGT)
        e = (np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)) * 1e3
        cs = contact_summary(sim)
        return {
            "t_rel": float(sim.data.time) - t_h,
            "e_theta_deg": float(m["e_theta_deg"]),
            "e_pos_mm": float(m["e_pos_mm"]),
            "e_signed_mm": e.tolist(),
            "e_x_mm": float(e[0]),
            "e_y_mm": float(e[1]),
            "e_z_mm": float(e[2]),
            "bilateral": bool(cs["bilateral"]),
            "obj_z": float(m["obj_z"]),
            "nL": cs["nL"],
            "nR": cs["nR"],
        }

    while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(sim)
        t_rel = float(sim.data.time) - t_h
        mon = _monitor(sim)
        bil = _bilateral(mon)
        z = float(sim.data.xpos[sim.ids.object_body][2])
        min_obj_z = min(min_obj_z, z)
        if bil and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            delayed_drop = True
            break
        if _dropped(mon, sim) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            delayed_drop = True
            break
        lost = lost + 1 if not bil else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            slip_events += 1
            break
        if t_rel + 1e-12 >= next_sample:
            samples.append(_row())
            if not samples[-1]["bilateral"]:
                slip_events += 1
            next_sample += 0.02

    if reason is None:
        mon = _monitor(sim)
        if not _bilateral(mon) or _dropped(mon, sim) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"

    # Ensure final sample
    if not samples or samples[-1]["t_rel"] < float(sim.data.time) - t_h - 1e-6:
        samples.append(_row())

    win = window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
    s12 = reason == "SUCCESS_12S"
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))
    sens = sensitivity_over_z_limits(samples, s12=s12)
    axis_default = longest_axis_ok_s(
        samples, window_start=float(NOMINAL_S) - FINAL_WINDOW_S, ez_mm=AXIS_V1_EZ_MM
    )
    success_axis = bool(s12 and axis_default["SUCCESS_AXIS_V1_hold"])

    return {
        "SUCCESS_12S": s12,
        "SUCCESS_NEW": success_new,
        "SUCCESS_AXIS_V1": success_axis,
        "SUCCESS_AXIS_V1_provisional": True,
        "reason": reason,
        "pre_theta": float(m0["e_theta_deg"]),
        "pre_pos_mm": float(m0["e_pos_mm"]),
        "pre_e_signed_mm": e0.tolist(),
        "pre_bilateral": bool(pb0["bilateral"]),
        "pre_in_band_NEW": int(
            m0["e_theta_deg"] <= WORKING_THETA_DEG and m0["e_pos_mm"] <= WORKING_POS_MM and pb0["bilateral"]
        ),
        "pre_in_band_AXIS_V1": int(
            axis_limits_ok(e0[0], e0[1], e0[2], m0["e_theta_deg"]) and pb0["bilateral"]
        ),
        "window_SUCCESS_NEW": {
            k: win[k]
            for k in win
            if k
            in (
                "longest_both_ok_s",
                "working_SUCCESS_NEW_hold",
                "mean_theta",
                "mean_pos",
                "max_theta",
                "max_pos",
                "frac_bilateral",
            )
        },
        "window_AXIS_V1": axis_default,
        "z_limit_sensitivity": sens,
        "min_obj_z": min_obj_z,
        "z_abort_margin": float(min_obj_z - 0.44),
        "slip_events": slip_events,
        "delayed_drop": delayed_drop,
        "samples_tail": samples[-40:],
        "n_samples": len(samples),
        # full samples kept for offline re-thresholding
        "samples": samples,
    }
