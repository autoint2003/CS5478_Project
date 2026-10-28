"""Experimental oscillatory dθ controller (P7). Does not modify production Oracle."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.corrections.contact_primitives import _action_vw
from data_generation.oracles.precision.mechanics.dual_loop import DT
from data_generation.oracles.precision.residuals import measure_errors
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_THETA_DEG
from experiments.precision.p7_oscillatory_dtheta.geometry import (
    body_axis_for_correction,
    full_orientation_snapshot,
    signed_theta_to_target,
)

Mode = Literal["symmetric", "asymmetric", "feedback"]

MIN_IMPROVE_DEG = 0.5
MAX_CYCLES = 6
ABORT_THETA_GROWTH = 2.0
ABORT_DX_DRIFT_MM = 8.0
ABORT_E_POS_MM = 14.0


def _envelope(i: int, n: int) -> float:
    ramp = max(1, n // 5)
    if i < ramp:
        return (i + 1) / ramp
    if i >= n - ramp:
        return max(0.0, (n - i) / ramp)
    return 1.0


def run_one_cycle(
    env,
    blob: dict,
    *,
    mode: Mode,
    amplitude_fwd_deg: float,
    amplitude_rev_deg: float,
    half_period_s: float = 0.45,
    settle_s: float = 0.35,
    grip_fwd: float = 0.05,
    grip_rev: float = 0.92,
    grip_settle: float = 0.90,
    direction: float = 1.0,
    w_cap: float = 0.85,
) -> dict[str, Any]:
    """One forward→reverse→settle cycle. Reverse amp may be smaller (asymmetric)."""
    restore_capture(env, blob["capture"])
    th0 = signed_theta_to_target(env.sim)
    m0 = measure(env.sim, U_TGT, P_TGT)
    u0_h = np.asarray(m0["u_c_h"], float)
    axis_h = np.asarray(th0["axis_h"], float)
    body_ax, ax_name = body_axis_for_correction(env.sim)
    # Align body command with signed-error sense: +direction should reduce |signed_theta|
    # signed_theta > 0 means need +rotation of cylinder about axis_h.
    # Forward stroke moves hand about body_ax * direction; with loose grip, relative slip
    # should accumulate along axis_h.
    body_cmd = body_ax * float(np.sign(direction) or 1.0)
    R0 = hand_R(env.sim).copy()

    amp_f = float(np.radians(amplitude_fwd_deg))
    amp_r = float(np.radians(amplitude_rev_deg))
    n_half = max(1, int(round(half_period_s / DT)))
    n_settle = max(1, int(round(settle_s / DT)))
    w_fwd = float(np.clip(amp_f / max(half_period_s, 1e-3), 0.12, min(w_cap, W_MAX)))
    w_rev = float(np.clip(amp_r / max(half_period_s, 1e-3), 0.08, min(w_cap, W_MAX)))

    actions: list = []
    series: list = []
    dropped = False
    lost = False
    peak_rel = 0.0
    peak_hand = 0.0

    def step(phase: str, w_body: np.ndarray, grip: float) -> bool:
        nonlocal dropped, lost, peak_rel, peak_hand
        a = _action_vw([0.0, 0.0, 0.0], w_body.tolist(), grip)
        step_legal(env.sim, a)
        actions.append(a.copy())
        snap = full_orientation_snapshot(env.sim, u0_h=u0_h, axis_h=axis_h, R0=R0)
        snap["phase"] = phase
        snap["grip_cmd"] = float(grip)
        snap["w_cmd"] = w_body.tolist()
        snap["t"] = len(actions) * DT
        series.append(snap)
        peak_rel = max(peak_rel, abs(float(snap["signed_rel_deg"])))
        peak_hand = max(peak_hand, abs(float(snap.get("hand_tilt_deg") or 0)))
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if latched(env.sim) or failed:
            dropped = True
            return False
        if not snap["bilateral"] and float(snap.get("fn_tot") or 0) < 0.4:
            lost = True
            return False
        return True

    # Forward
    for i in range(n_half):
        w = body_cmd * (w_fwd * _envelope(i, n_half))
        if not step("forward", w, grip_fwd):
            break
    rel_after_fwd = float(series[-1]["signed_rel_deg"]) if series else 0.0
    th_after_fwd = float(series[-1]["abs_theta_deg"]) if series else th0["abs_theta_deg"]
    signed_after_fwd = float(series[-1]["signed_theta_deg"]) if series else th0["signed_theta_deg"]

    # Reverse (smaller for asymmetric / feedback) — unload / return hand, not full undo
    if not dropped and not lost:
        for i in range(n_half):
            w = -body_cmd * (w_rev * _envelope(i, n_half))
            if not step("reverse", w, grip_rev):
                break

    if not dropped:
        for _ in range(n_settle):
            if not step("settle", np.zeros(3), grip_settle):
                break

    th1 = signed_theta_to_target(env.sim)
    snap_f = series[-1] if series else {}
    net_rel = float(snap_f.get("signed_rel_deg") or 0)
    undid = abs(rel_after_fwd) >= 1.0 and abs(net_rel) < 0.35 * abs(rel_after_fwd)
    improve = float(th0["abs_theta_deg"]) - float(th1["abs_theta_deg"])
    co_rotate = peak_hand >= 2.0 and peak_rel < 0.35 * peak_hand
    irreversible = (not undid) and abs(net_rel) >= 1.0 and improve >= MIN_IMPROVE_DEG

    cls = "unknown"
    if dropped:
        cls = "DROP"
    elif lost:
        cls = "CONTACT_LOSS"
    elif co_rotate and abs(net_rel) < 1.0:
        cls = "CO_ROTATION"
    elif undid:
        cls = "REVERSIBLE_ELASTIC"
    elif irreversible:
        cls = "IRREVERSIBLE_SLIP"
    elif abs(net_rel) >= 1.0 and improve < MIN_IMPROVE_DEG:
        cls = "WRONG_DIRECTION_OR_NO_ABS_IMPROVE"
    else:
        cls = "NO_NET_RELATIVE_ROTATION"

    return {
        "ok": bool(irreversible and not dropped and not lost and th1["bilateral"]),
        "mode": mode,
        "classification": cls,
        "amplitude_fwd_deg": amplitude_fwd_deg,
        "amplitude_rev_deg": amplitude_rev_deg,
        "direction": float(direction),
        "body_axis_name": ax_name,
        "body_axis": body_cmd.tolist(),
        "axis_h": axis_h.tolist(),
        "theta0_abs": float(th0["abs_theta_deg"]),
        "theta1_abs": float(th1["abs_theta_deg"]),
        "signed0": float(th0["signed_theta_deg"]),
        "signed1": float(th1["signed_theta_deg"]),
        "signed_after_forward": signed_after_fwd,
        "theta_after_forward": th_after_fwd,
        "net_improve_abs": improve,
        "net_relative_deg": net_rel,
        "rel_after_forward": rel_after_fwd,
        "peak_relative_deg": peak_rel,
        "peak_hand_tilt_deg": peak_hand,
        "undid_on_reverse": undid,
        "co_rotation_dominant": co_rotate,
        "dx0": float(th0["dx_mm"]),
        "dx1": float(th1["dx_mm"]),
        "dz0": float(th0["dz_mm"]),
        "dz1": float(th1["dz_mm"]),
        "e_pos0": float(th0["e_pos_mm"]),
        "e_pos1": float(th1["e_pos_mm"]),
        "bilateral_end": bool(th1["bilateral"]),
        "dropped": dropped,
        "contact_loss": lost,
        "n_actions": len(actions),
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
        "errors_end": measure_errors(env.sim),
    }


def run_oscillation_controller(
    env,
    blob: dict,
    *,
    mode: Mode,
    amp_fwd0: float = 10.0,
    amp_rev_frac: float = 1.0,
    max_cycles: int = MAX_CYCLES,
    half_period_s: float = 0.45,
) -> dict[str, Any]:
    """Multi-cycle oscillatory correction with stop / abort rules."""
    restore_capture(env, blob["capture"])
    th_start = signed_theta_to_target(env.sim)
    # Choose direction: first try +1, if worsens try -1 on cycle 0 only
    direction = 1.0
    cycles: list[dict] = []
    all_actions: list = []
    all_series: list = []
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    stagnant = 0
    abort_reason = None
    reached_tol = False

    for k in range(max_cycles):
        th_now = signed_theta_to_target(env.sim)
        if float(th_now["abs_theta_deg"]) <= WORKING_THETA_DEG:
            reached_tol = True
            break

        if mode == "symmetric":
            amp_f, amp_r = amp_fwd0, amp_fwd0
        elif mode == "asymmetric":
            amp_f, amp_r = amp_fwd0, amp_fwd0 * float(amp_rev_frac)
        else:  # feedback: shrink amplitude as θ approaches 10°
            excess = max(0.0, float(th_now["abs_theta_deg"]) - WORKING_THETA_DEG)
            scale = float(np.clip(excess / 6.0, 0.35, 1.0))
            amp_f = amp_fwd0 * scale
            amp_r = amp_f * float(amp_rev_frac)

        r = run_one_cycle(
            env,
            cur,
            mode=mode,
            amplitude_fwd_deg=amp_f,
            amplitude_rev_deg=amp_r,
            half_period_s=half_period_s,
            direction=direction,
        )
        # On first cycle, flip direction if abs θ worsened and relative slip wrong way
        if k == 0 and r["net_improve_abs"] < -0.3:
            restore_capture(env, blob["capture"])
            direction = -1.0
            r = run_one_cycle(
                env,
                blob,
                mode=mode,
                amplitude_fwd_deg=amp_f,
                amplitude_rev_deg=amp_r,
                half_period_s=half_period_s,
                direction=direction,
            )

        row = {kk: vv for kk, vv in r.items() if kk not in ("actions", "series", "capture", "errors_end")}
        row["cycle"] = k
        cycles.append(row)
        all_actions += r.get("actions") or []
        for s in r.get("series") or []:
            s2 = dict(s)
            s2["cycle"] = k
            all_series.append(s2)

        if r.get("dropped") or r.get("contact_loss") or not r.get("bilateral_end"):
            abort_reason = r.get("classification") or "CONTACT_ABORT"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break
        if r["theta1_abs"] > th_start["abs_theta_deg"] + ABORT_THETA_GROWTH:
            abort_reason = "THETA_GROWTH"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break
        if abs(r["dx1"] - th_start["dx_mm"]) > ABORT_DX_DRIFT_MM or r["e_pos1"] > ABORT_E_POS_MM:
            abort_reason = "POSITION_DRIFT"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break

        if r["net_improve_abs"] < MIN_IMPROVE_DEG:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= 2:
            abort_reason = "TWO_CYCLES_NO_IMPROVE"
            cur = {"capture": r["capture"], "ref": blob["ref"]}
            break

        cur = {"capture": r["capture"], "ref": blob["ref"]}
        restore_capture(env, cur["capture"])
        if float(signed_theta_to_target(env.sim)["abs_theta_deg"]) <= WORKING_THETA_DEG:
            reached_tol = True
            break

    restore_capture(env, cur["capture"])
    # Settle before validation snapshot
    for _ in range(int(0.4 / DT)):
        a = _action_vw([0, 0, 0], [0, 0, 0], 0.92)
        step_legal(env.sim, a)
        all_actions.append(a.copy())
    final_cap = take_capture(env)
    th_f = signed_theta_to_target(env.sim)
    return {
        "mode": mode,
        "direction": direction,
        "n_cycles": len(cycles),
        "cycles": cycles,
        "abort_reason": abort_reason,
        "reached_theta_tol": bool(reached_tol or th_f["abs_theta_deg"] <= WORKING_THETA_DEG),
        "theta0_abs": float(th_start["abs_theta_deg"]),
        "theta_final_abs": float(th_f["abs_theta_deg"]),
        "signed0": float(th_start["signed_theta_deg"]),
        "signed_final": float(th_f["signed_theta_deg"]),
        "net_improve_total": float(th_start["abs_theta_deg"]) - float(th_f["abs_theta_deg"]),
        "dx_final": float(th_f["dx_mm"]),
        "dz_final": float(th_f["dz_mm"]),
        "e_pos_final": float(th_f["e_pos_mm"]),
        "bilateral_final": bool(th_f["bilateral"]),
        "actions": all_actions,
        "series": all_series,
        "capture": final_cap,
        "start": th_start,
        "final": th_f,
    }
