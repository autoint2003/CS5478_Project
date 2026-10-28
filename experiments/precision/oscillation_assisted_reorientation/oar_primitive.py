"""Isolated OAR single-cycle controller (experimental, not production).

Conditions:
  A — Symmetric oscillation (constant grip)
  B — Forward-release / reverse-lock
  C — Forward-lock / reverse-release
  D — Grip-only (no angular oscillation)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from data_generation.oracles.precision.corrections.contact_primitives import _action_vw, _wrist
from data_generation.oracles.precision.mechanics.dual_loop import DT
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT
from experiments.precision.oscillation_assisted_reorientation.metrics import (
    desired_correction_axis_hand,
    ic_log,
    relative_cyl_angle,
    snapshot_state,
)

Condition = Literal["A", "B", "C", "D"]

# Diagnostic COM translation reference (not SUCCESS threshold).
DIAG_MAX_DP_MM = 3.0
MIN_NET_REL_DEG = 1.5  # persistent net relative rotation after settle


def _unit(v) -> np.ndarray:
    a = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.zeros(3)


def _body_axis_for_correction(sim, osc_axis: str = "auto") -> np.ndarray:
    """Body-frame (r_des) angular command axis."""
    if osc_axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if osc_axis == "z":
        return np.array([0.0, 0.0, 1.0])
    if osc_axis == "y":
        return np.array([0.0, 1.0, 0.0])
    # auto: map desired hand-frame correction axis into approximate body y/x
    ax_h = desired_correction_axis_hand(sim)
    # Prefer the body axis with largest |projection| onto ax_h (hand ≈ body for precision grasp)
    cands = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    best = max(cands.items(), key=lambda kv: abs(float(np.dot(ax_h, kv[1]))))
    return cands[best[0]] * np.sign(float(np.dot(ax_h, cands[best[0]])) or 1.0)


def _grip_profile(condition: Condition, phase: str, *, grip_hold: float, grip_loose: float, grip_lock: float) -> float:
    """phase in {forward, reverse, settle, idle}."""
    if condition == "A":
        return grip_hold
    if condition == "D":
        # same modulation timeline without ω
        if phase == "forward":
            return grip_loose
        if phase == "reverse":
            return grip_lock
        return grip_hold
    if condition == "B":
        # forward-release / reverse-lock
        if phase == "forward":
            return grip_loose
        if phase == "reverse":
            return grip_lock
        return grip_lock if phase == "settle" else grip_hold
    if condition == "C":
        # forward-lock / reverse-release
        if phase == "forward":
            return grip_lock
        if phase == "reverse":
            return grip_loose
        return grip_lock if phase == "settle" else grip_hold
    return grip_hold


def _classify(
    *,
    net_rel: float,
    peak_rel: float,
    dtheta_improve: float,
    delta_p_mm: float,
    dropped: bool,
    contact_loss: bool,
    tracked: bool,
    undid: bool,
    grip_only_effect: float | None,
) -> tuple[bool, str | None]:
    if dropped:
        return False, "DROP"
    if contact_loss:
        return False, "CONTACT_LOSS"
    if not tracked:
        return False, "ACTUATION_NOT_TRACKED"
    if delta_p_mm > 12.0:
        return False, "EXCESSIVE_COM_TRANSLATION"
    if abs(net_rel) < MIN_NET_REL_DEG:
        if abs(peak_rel) >= MIN_NET_REL_DEG and undid:
            return False, "REVERSE_STROKE_UNDOES_CORRECTION"
        if abs(peak_rel) < 0.5:
            return False, "RIGID_CO_ROTATION" if abs(dtheta_improve) < 0.5 else "NO_NET_RELATIVE_ROTATION"
        return False, "NO_NET_RELATIVE_ROTATION"
    if net_rel * dtheta_improve < 0 and abs(dtheta_improve) >= 1.0 and abs(net_rel) >= MIN_NET_REL_DEG:
        # relative rotation opposite to dθ improvement sense — still may be wrong corrective dir
        pass
    if dtheta_improve < 0.5 and abs(net_rel) >= MIN_NET_REL_DEG:
        # moved relatively but world dθ not improved
        return False, "WRONG_DIRECTION_ROTATION"
    if undid and abs(net_rel) < abs(peak_rel) * 0.35:
        return False, "REVERSE_STROKE_UNDOES_CORRECTION"
    if abs(net_rel) >= MIN_NET_REL_DEG and dtheta_improve >= 1.0 and delta_p_mm <= 12.0:
        return True, None
    if abs(net_rel) >= MIN_NET_REL_DEG and dtheta_improve < 1.0:
        return False, "WRONG_DIRECTION_ROTATION"
    return False, "NO_NET_RELATIVE_ROTATION"


def run_oar_cycle(
    env,
    blob: dict,
    *,
    condition: Condition,
    direction: float = 1.0,
    amplitude_deg: float = 12.0,
    half_period_s: float = 0.6,
    settle_s: float = 0.4,
    grip_hold: float = 0.85,
    grip_loose: float = 0.05,
    grip_lock: float = 0.95,
    osc_axis: str = "auto",
    w_scale: float = 0.7,
) -> dict[str, Any]:
    """One complete oscillation cycle from identical IC snapshot."""
    restore_capture(env, blob["capture"])
    ic = ic_log(env.sim)
    if not ic.get("bilateral"):
        return {
            "ok": False,
            "failure": "CONTACT_LOSS",
            "reason": "no_bilateral_at_ic",
            "condition": condition,
            "actions": [],
            "capture": blob["capture"],
            "ic": ic,
        }

    m0 = measure(env.sim, U_TGT, P_TGT)
    u0_h = _unit(m0["u_c_h"])
    axis_h = desired_correction_axis_hand(env.sim)
    # Sign for relative angle: positive = toward target about axis_h
    R0 = hand_R(env.sim).copy()
    body_ax = _body_axis_for_correction(env.sim, osc_axis) * float(np.sign(direction) or 1.0)
    amp = float(np.radians(amplitude_deg))
    # Constant |ω| such that amp ≈ |ω| * half_period (trapezoid approx)
    w_mag = float(np.clip(amp / max(half_period_s, 1e-3), 0.15, w_scale))
    n_half = max(1, int(round(half_period_s / DT)))
    n_settle = max(1, int(round(settle_s / DT)))

    actions: list = []
    series: list = []
    lost = False
    dropped = False
    peak_rel = 0.0
    peak_forward = 0.0
    hand_peak = 0.0
    cmd_peak = 0.0

    def step(phase: str, w_body: np.ndarray, grip: float):
        nonlocal lost, dropped, peak_rel, hand_peak, cmd_peak
        a = _action_vw([0.0, 0.0, 0.0], w_body.tolist(), grip)
        step_legal(env.sim, a)
        actions.append(a.copy())
        cmd_peak = max(cmd_peak, float(np.linalg.norm(w_body)))
        snap = snapshot_state(env.sim, u0_h=u0_h, axis_h=axis_h, R0=R0)
        snap["phase"] = phase
        snap["grip_cmd"] = float(grip)
        snap["w_cmd_body"] = w_body.tolist()
        snap["t"] = len(actions) * DT
        series.append(snap)
        peak_rel = max(peak_rel, abs(float(snap["signed_rel_deg"])))
        hand_peak = max(hand_peak, float(snap["w_hand_norm"]))
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if latched(env.sim) or failed:
            dropped = True
            return False
        if not snap["bilateral"] and float(snap["fn_tot"]) < 0.4:
            lost = True
            return False
        return True

    # --- Forward half ---
    w_fwd = body_ax * w_mag
    g_fwd = _grip_profile(condition, "forward", grip_hold=grip_hold, grip_loose=grip_loose, grip_lock=grip_lock)
    for i in range(n_half):
        # Soft accel/decel envelope
        env_frac = 1.0
        if i < max(1, n_half // 5):
            env_frac = (i + 1) / max(1, n_half // 5)
        elif i > n_half - max(1, n_half // 5):
            env_frac = (n_half - i) / max(1, n_half // 5)
        w = w_fwd * env_frac if condition != "D" else np.zeros(3)
        if not step("forward", w, g_fwd):
            break
    peak_forward = max((abs(s["signed_rel_deg"]) for s in series if s["phase"] == "forward"), default=0.0)
    rel_after_forward = float(series[-1]["signed_rel_deg"]) if series else 0.0

    # --- Reverse half (return hand) ---
    if not dropped and not lost:
        w_rev = -body_ax * w_mag
        g_rev = _grip_profile(condition, "reverse", grip_hold=grip_hold, grip_loose=grip_loose, grip_lock=grip_lock)
        for i in range(n_half):
            env_frac = 1.0
            if i < max(1, n_half // 5):
                env_frac = (i + 1) / max(1, n_half // 5)
            elif i > n_half - max(1, n_half // 5):
                env_frac = (n_half - i) / max(1, n_half // 5)
            w = w_rev * env_frac if condition != "D" else np.zeros(3)
            if not step("reverse", w, g_rev):
                break

    # --- Settle ---
    g_set = _grip_profile(condition, "settle", grip_hold=grip_hold, grip_loose=grip_loose, grip_lock=grip_lock)
    if not dropped:
        for _ in range(n_settle):
            if not step("settle", np.zeros(3), g_set):
                break

    # Final metrics
    restore_cap = take_capture(env)
    snap_f = snapshot_state(env.sim, u0_h=u0_h, axis_h=axis_h, R0=R0)
    net_rel = float(snap_f["signed_rel_deg"])
    dtheta0 = float(ic["dtheta_deg"])
    dtheta1 = float(snap_f["dtheta_deg"])
    d_imp = dtheta0 - dtheta1
    p0 = np.asarray(ic["p_c_h"], float)
    p1 = np.asarray(snap_f["p_c_h"], float)
    delta_p = float(1e3 * np.linalg.norm(p1 - p0))

    # Tracking: did hand actually move during forward?
    hand_tilts = [abs(float(s.get("hand_tilt_deg") or 0)) for s in series if s["phase"] == "forward"]
    peak_hand_tilt = max(hand_tilts) if hand_tilts else 0.0
    tracked = True
    if condition != "D":
        # expect some hand motion; if peak hand ω and tilt tiny vs command → not tracked
        if hand_peak < 0.05 * max(w_mag, 0.1) and peak_hand_tilt < 0.25 * amplitude_deg:
            tracked = False
    undid = abs(rel_after_forward) >= MIN_NET_REL_DEG and abs(net_rel) < 0.35 * abs(rel_after_forward)

    # Corrective sense: positive net_rel along axis_h should reduce dθ if axis chosen correctly
    # Prefer improvement in dθ as success criterion together with net_rel
    ok, failure = _classify(
        net_rel=net_rel,
        peak_rel=peak_rel,
        dtheta_improve=d_imp,
        delta_p_mm=delta_p,
        dropped=dropped,
        contact_loss=lost and not dropped,
        tracked=tracked,
        undid=undid,
        grip_only_effect=None,
    )
    # Settling rebound: net small but was large at end of reverse before settle
    if series:
        pre_settle = [s for s in series if s["phase"] == "reverse"]
        if pre_settle and abs(float(pre_settle[-1]["signed_rel_deg"])) >= MIN_NET_REL_DEG:
            if abs(net_rel) < 0.35 * abs(float(pre_settle[-1]["signed_rel_deg"])) and d_imp < 0.5:
                if failure in (None, "NO_NET_RELATIVE_ROTATION", "REVERSE_STROKE_UNDOES_CORRECTION"):
                    failure = "SETTLING_REBOUND"
                    ok = False

    return {
        "ok": bool(ok),
        "failure": failure,
        "condition": condition,
        "direction": float(direction),
        "amplitude_deg": amplitude_deg,
        "half_period_s": half_period_s,
        "body_axis": body_ax.tolist(),
        "axis_h": axis_h.tolist(),
        "ic": ic,
        "net_relative_rotation_deg": net_rel,
        "peak_relative_rotation_deg": peak_rel,
        "rel_after_forward_deg": rel_after_forward,
        "dtheta0_deg": dtheta0,
        "dtheta1_deg": dtheta1,
        "d_theta_improve_deg": d_imp,
        "delta_p_rel_mm": delta_p,
        "diag_3mm_ok": delta_p <= DIAG_MAX_DP_MM,
        "peak_hand_tilt_deg": peak_hand_tilt,
        "peak_hand_w": hand_peak,
        "peak_w_cmd": cmd_peak,
        "tracked": tracked,
        "dropped": dropped,
        "contact_loss": lost,
        "undid_on_reverse": undid,
        "bilateral_end": bool(snap_f["bilateral"]),
        "final": snap_f,
        "series": series,
        "actions": actions,
        "capture": restore_cap,
        "verified_lateral_fallback": False,
    }


def run_multi_cycle(env, blob: dict, *, condition: Condition, n_cycles: int = 3, **kwargs) -> dict[str, Any]:
    """Repeat best single-cycle strategy; only call if single-cycle showed net correction."""
    restore_capture(env, blob["capture"])
    cycles = []
    all_actions: list = []
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    d0 = None
    for k in range(n_cycles):
        r = run_oar_cycle(env, cur, condition=condition, **kwargs)
        all_actions += r.get("actions") or []
        cycles.append({k: v for k, v in r.items() if k not in ("actions", "series", "capture")})
        if d0 is None:
            d0 = r.get("dtheta0_deg")
        if not r.get("bilateral_end") or r.get("dropped") or r.get("failure") in ("DROP", "CONTACT_LOSS"):
            break
        cur = {"capture": r["capture"], "ref": blob["ref"]}
    last = cycles[-1] if cycles else {}
    return {
        "ok": bool(last.get("ok")) and len(cycles) == n_cycles,
        "n_cycles_run": len(cycles),
        "cycles": cycles,
        "net_relative_rotation_deg": last.get("net_relative_rotation_deg"),
        "d_theta_improve_total": (d0 - last.get("dtheta1_deg")) if d0 is not None and last.get("dtheta1_deg") is not None else None,
        "delta_p_rel_mm": last.get("delta_p_rel_mm"),
        "actions": all_actions,
        "capture": take_capture(env),
        "failure": last.get("failure"),
    }
