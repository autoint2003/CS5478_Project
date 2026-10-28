"""Experimental inertial stick-slip dθ cycles (P8). Production Oracle untouched."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3, W_MAX, grip_tau, normalize_action
from data_generation.oracles.precision.mechanics.dual_loop import sample_state
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_THETA_DEG
from experiments.precision.p7_oscillatory_dtheta.geometry import (
    body_axis_for_correction,
    signed_theta_to_target,
)
from experiments.precision.p8_inertial_dtheta.authority import (
    DT,
    hand_omega_along_body,
    obj_omega_along_body,
    rel_omega_along_axis_h,
)

Mode = Literal["p7_baseline", "fast_hold_grip", "fast_reduced_grip", "phase_grip"]

MIN_IMPROVE = 0.5
MAX_CYCLES = 5
ABORT_THETA_GROWTH = 2.5
ABORT_DX = 8.0
ABORT_EPOS = 14.0


def _action(w_body: np.ndarray, grip: float) -> np.ndarray:
    return np.asarray(
        normalize_action(np.zeros(3), np.asarray(w_body, float), grip_tau(float(grip)), AIRBORNE_V3),
        np.float32,
    )


def run_stick_slip_cycle(
    env,
    blob: dict,
    *,
    mode: Mode,
    w_cruise: float,
    grip_hold: float,
    grip_relax: float,
    n_accel: int = 4,
    n_cruise: int = 6,
    n_brake: int = 4,
    n_settle: int = 12,
    direction: float = 1.0,
) -> dict[str, Any]:
    """One accel → (optional relax) → brake/reclamp → settle cycle."""
    restore_capture(env, blob["capture"])
    th0 = signed_theta_to_target(env.sim)
    m0 = measure(env.sim, U_TGT, P_TGT)
    u0_h = np.asarray(m0["u_c_h"], float)
    axis_h = np.asarray(th0["axis_h"], float)
    body_ax, ax_name = body_axis_for_correction(env.sim)
    body_cmd = body_ax * float(np.sign(direction) or 1.0)
    R0 = hand_R(env.sim).copy()

    actions: list = []
    series: list = []
    dropped = lost = False
    w_prev = hand_omega_along_body(env.sim, body_cmd)
    peak_alpha = 0.0
    peak_w = 0.0
    peak_wrel = 0.0
    peak_rel_deg = 0.0

    from experiments.precision.oscillation_assisted_reorientation.metrics import (
        relative_cyl_angle,
        snapshot_state,
    )

    def step(phase: str, w_scale: float, grip: float) -> bool:
        nonlocal dropped, lost, w_prev, peak_alpha, peak_w, peak_wrel, peak_rel_deg
        a = _action(body_cmd * float(w_scale), grip)
        step_legal(env.sim, a)
        actions.append(a.copy())
        w_now = hand_omega_along_body(env.sim, body_cmd)
        alpha = (w_now - w_prev) / DT
        w_prev = w_now
        th = signed_theta_to_target(env.sim)
        st = sample_state(env.sim)
        rel = relative_cyl_angle(env.sim, u0_h, axis_h)
        wrel = rel_omega_along_axis_h(env.sim, axis_h)
        peak_alpha = max(peak_alpha, abs(alpha))
        peak_w = max(peak_w, abs(w_now))
        peak_wrel = max(peak_wrel, abs(wrel))
        peak_rel_deg = max(peak_rel_deg, abs(rel))
        series.append(
            {
                "phase": phase,
                "t": len(actions) * DT,
                "w_cmd": float(w_scale) * float(np.sign(direction)),
                "w_hand": w_now,
                "w_obj": obj_omega_along_body(env.sim, body_cmd),
                "alpha_hand": alpha,
                "w_rel": wrel,
                "signed_rel_deg": rel,
                "theta_abs": th["abs_theta_deg"],
                "signed_theta": th["signed_theta_deg"],
                "dx": th["dx_mm"],
                "dz": th["dz_mm"],
                "e_pos": th["e_pos_mm"],
                "fn_tot": float(st.get("fn_tot") or 0),
                "fn_L": float(st.get("fn_L") or 0),
                "fn_R": float(st.get("fn_R") or 0),
                "bilateral": bool(st.get("bilateral")),
                "grip_cmd": float(grip),
            }
        )
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if latched(env.sim) or failed:
            dropped = True
            return False
        if not series[-1]["bilateral"] and series[-1]["fn_tot"] < 0.4:
            lost = True
            return False
        return True

    # Phase-dependent grips
    if mode == "p7_baseline":
        # Slow cruise like P7 (~0.35 rad/s), constant mid grip
        w_cruise_use = min(0.4, abs(w_cruise))
        g_acc = g_cru = g_brk = grip_hold
    elif mode == "fast_hold_grip":
        w_cruise_use = min(float(W_MAX), abs(w_cruise))
        g_acc = g_cru = g_brk = grip_hold
    elif mode == "fast_reduced_grip":
        w_cruise_use = min(float(W_MAX), abs(w_cruise))
        g_acc = g_cru = g_brk = grip_relax
    else:  # phase_grip: hold during accel, relax at cruise/reversal onset, reclamp on brake
        w_cruise_use = min(float(W_MAX), abs(w_cruise))
        g_acc = grip_hold
        g_cru = grip_relax
        g_brk = grip_hold

    # Accel ramp to cruise
    for i in range(n_accel):
        scale = w_cruise_use * ((i + 1) / max(1, n_accel))
        if not step("accel", scale, g_acc):
            break
    # Cruise
    if not dropped and not lost:
        for _ in range(n_cruise):
            if not step("cruise", w_cruise_use, g_cru):
                break
    # Brake / reverse pulse (inertial unload) then zero
    if not dropped and not lost:
        for i in range(n_brake):
            # decelerate toward 0; last half slight reverse impulse for stick-slip
            if i < n_brake // 2:
                scale = w_cruise_use * (1.0 - (i + 1) / max(1, n_brake // 2))
            else:
                scale = -0.35 * w_cruise_use * ((i - n_brake // 2 + 1) / max(1, n_brake - n_brake // 2))
            grip_b = g_cru if (mode == "phase_grip" and i < n_brake // 2) else g_brk
            if not step("brake", scale, grip_b):
                break
    # Reclamp settle
    if not dropped:
        for _ in range(n_settle):
            if not step("settle", 0.0, grip_hold if mode != "fast_reduced_grip" else max(grip_relax, 0.7)):
                break

    th1 = signed_theta_to_target(env.sim)
    net_rel = float(series[-1]["signed_rel_deg"]) if series else 0.0
    improve = float(th0["abs_theta_deg"] - th1["abs_theta_deg"])
    co_rot = peak_w >= 0.5 and peak_rel_deg < 0.35 * max(peak_w * 57.3 * 0.05, 1.0)
    # clearer: if peak relative deg tiny vs hand motion estimate
    hand_tilt_proxy = peak_w * (n_accel + n_cruise) * DT * 57.3  # rough deg if sustained
    if peak_rel_deg < 0.5 and hand_tilt_proxy > 3.0:
        cls = "CO_ROTATION"
    elif lost:
        cls = "CONTACT_LOSS"
    elif dropped:
        cls = "DROP"
    elif abs(th1["dx_mm"] - th0["dx_mm"]) > 5.0 or (th1["e_pos_mm"] - th0["e_pos_mm"]) > 6.0:
        cls = "TRANSLATIONAL_SLIP"
    elif peak_rel_deg >= 1.0 and improve >= MIN_IMPROVE:
        cls = "USEFUL_ROTATIONAL_SLIP"
    elif peak_rel_deg >= 1.0 and improve < MIN_IMPROVE:
        cls = "RELATIVE_BUT_NO_ABS_IMPROVE"
    else:
        cls = "NO_NET_RELATIVE_ROTATION"

    return {
        "ok": bool(cls == "USEFUL_ROTATIONAL_SLIP" and th1["bilateral"]),
        "mode": mode,
        "classification": cls,
        "direction": float(direction),
        "w_cruise": float(w_cruise_use),
        "grip_hold": float(grip_hold),
        "grip_relax": float(grip_relax),
        "body_axis_name": ax_name,
        "theta0_abs": float(th0["abs_theta_deg"]),
        "theta1_abs": float(th1["abs_theta_deg"]),
        "signed0": float(th0["signed_theta_deg"]),
        "signed1": float(th1["signed_theta_deg"]),
        "net_improve_abs": improve,
        "net_relative_deg": net_rel,
        "peak_relative_deg": peak_rel_deg,
        "peak_w_hand": peak_w,
        "peak_alpha_hand": peak_alpha,
        "peak_w_rel": peak_wrel,
        "dx0": float(th0["dx_mm"]),
        "dx1": float(th1["dx_mm"]),
        "dz0": float(th0["dz_mm"]),
        "dz1": float(th1["dz_mm"]),
        "e_pos0": float(th0["e_pos_mm"]),
        "e_pos1": float(th1["e_pos_mm"]),
        "fn_min": min((s["fn_tot"] for s in series), default=None),
        "fn_max": max((s["fn_tot"] for s in series), default=None),
        "bilateral_end": bool(th1["bilateral"]),
        "dropped": dropped,
        "contact_loss": lost,
        "n_actions": len(actions),
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
    }


def run_controller(
    env,
    blob: dict,
    *,
    mode: Mode,
    w_cruise: float = 4.0,
    grip_hold: float = 0.90,
    grip_relax: float = 0.05,
    max_cycles: int = MAX_CYCLES,
) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    th_start = signed_theta_to_target(env.sim)
    direction = 1.0
    cycles: list = []
    all_actions: list = []
    all_series: list = []
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    stagnant = 0
    abort = None
    reached = False

    for k in range(max_cycles):
        th_now = signed_theta_to_target(env.sim)
        if th_now["abs_theta_deg"] <= WORKING_THETA_DEG:
            reached = True
            break
        r = run_stick_slip_cycle(
            env,
            cur,
            mode=mode,
            w_cruise=w_cruise,
            grip_hold=grip_hold,
            grip_relax=grip_relax,
            direction=direction,
        )
        if k == 0 and r["net_improve_abs"] < -0.4:
            restore_capture(env, blob["capture"])
            direction = -1.0
            r = run_stick_slip_cycle(
                env,
                blob,
                mode=mode,
                w_cruise=w_cruise,
                grip_hold=grip_hold,
                grip_relax=grip_relax,
                direction=direction,
            )
        row = {kk: vv for kk, vv in r.items() if kk not in ("actions", "series", "capture")}
        row["cycle"] = k
        cycles.append(row)
        all_actions += r.get("actions") or []
        for s in r.get("series") or []:
            s2 = dict(s)
            s2["cycle"] = k
            all_series.append(s2)

        if r.get("dropped") or r.get("contact_loss") or not r.get("bilateral_end"):
            abort = r.get("classification") or "CONTACT_ABORT"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break
        if r["theta1_abs"] > th_start["abs_theta_deg"] + ABORT_THETA_GROWTH:
            abort = "THETA_GROWTH"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break
        if abs(r["dx1"] - th_start["dx_mm"]) > ABORT_DX or r["e_pos1"] > ABORT_EPOS:
            abort = "POSITION_DRIFT"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break
        if r["net_improve_abs"] < MIN_IMPROVE:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= 2:
            abort = "TWO_CYCLES_NO_IMPROVE"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break
        cur = {"capture": r["capture"], "ref": blob["ref"]}
        restore_capture(env, cur["capture"])
        if signed_theta_to_target(env.sim)["abs_theta_deg"] <= WORKING_THETA_DEG:
            reached = True
            break

    restore_capture(env, cur["capture"])
    for _ in range(int(0.4 / DT)):
        a = _action(np.zeros(3), grip_hold)
        step_legal(env.sim, a)
        all_actions.append(a.copy())
    final_cap = take_capture(env)
    th_f = signed_theta_to_target(env.sim)
    return {
        "mode": mode,
        "direction": direction,
        "n_cycles": len(cycles),
        "cycles": cycles,
        "abort_reason": abort,
        "reached_theta_tol": bool(reached or th_f["abs_theta_deg"] <= WORKING_THETA_DEG),
        "theta0_abs": float(th_start["abs_theta_deg"]),
        "theta_final_abs": float(th_f["abs_theta_deg"]),
        "signed0": float(th_start["signed_theta_deg"]),
        "signed_final": float(th_f["signed_theta_deg"]),
        "net_improve_total": float(th_start["abs_theta_deg"] - th_f["abs_theta_deg"]),
        "dx_final": float(th_f["dx_mm"]),
        "dz_final": float(th_f["dz_mm"]),
        "e_pos_final": float(th_f["e_pos_mm"]),
        "bilateral_final": bool(th_f["bilateral"]),
        "peak_alpha_all": max((c.get("peak_alpha_hand") or 0) for c in cycles) if cycles else 0.0,
        "peak_w_rel_all": max((c.get("peak_w_rel") or 0) for c in cycles) if cycles else 0.0,
        "actions": all_actions,
        "series": all_series,
        "capture": final_cap,
        "start": th_start,
        "final": th_f,
    }
