"""Replay action tapes and record DX/DZ/dθ/contact time series for causal analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.open_axis import gates
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
from data_generation.oracles.precision.residuals import measure_errors
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R, object_R

DT = 0.02


def _sample(env) -> dict[str, Any]:
    err = measure_errors(env.sim)
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    g = gates(env.sim)
    mode = classify_contact_mode(env.sim)
    Ro = object_R(env.sim)
    Rh = hand_R(env.sim)
    # Relative orientation: angle between object long axis and hand open (approx)
    try:
        # cylinder axis ≈ object body z in local; use world z column
        u_obj = Ro[:, 2]
        u_hand = Rh[:, 2]
        cos = float(np.clip(np.dot(u_obj, u_hand), -1.0, 1.0))
        rel_deg = float(np.degrees(np.arccos(cos)))
    except Exception:
        rel_deg = float("nan")
    return {
        "t": None,
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "dtheta_deg": float(err.get("dtheta_deg") or 0.0),
        "e_pos_mm": float(err.get("e_pos_mm") or 0.0),
        "bilateral": bool(err.get("bilateral")),
        "fn_L": float(err.get("fn_L") or 0.0),
        "fn_R": float(err.get("fn_R") or 0.0),
        "fn_asym": float(err.get("fn_L") or 0.0) - float(err.get("fn_R") or 0.0),
        "v_rel_norm": float(err.get("v_rel_norm") or 0.0),
        "w_rel_norm": float(err.get("w_rel_norm") or 0.0),
        "support_side": mode.get("support_side"),
        "one_sided_mechanical": bool(mode.get("one_sided_mechanical")),
        "Fn_z_L": float(mode.get("Fn_z_L") or 0.0),
        "Fn_z_R": float(mode.get("Fn_z_R") or 0.0),
        "alpha_up_deg": float(g.get("alpha_up_deg") or 0.0),
        "fingertips_down": bool(g.get("fingertips_down")),
        "obj_hand_rel_deg": rel_deg,
        "grip_cmd": None,
    }


def replay_trace(
    env,
    blob: dict,
    actions: list,
    *,
    stride: int = 5,
    max_steps: int | None = None,
) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    series: list[dict] = []
    s0 = _sample(env)
    s0["t"] = 0.0
    s0["grip_cmd"] = 0.0
    series.append(s0)

    acts = list(actions or [])
    if max_steps is not None:
        acts = acts[: int(max_steps)]

    dtheta0 = float(s0["dtheta_deg"])
    dtheta_max = dtheta0
    dtheta_min = dtheta0
    dz_at_dx_start = None
    dz_at_dx_end = None
    # Heuristic: DX phase ≈ first contiguous stretch where |ddx| decreases while posture not DOWN
    # We mark "during large |cmd| lateral" via grip+wrist magnitude; keep simple extrema.
    contact_loss_events = 0
    prev_bil = True
    for i, a in enumerate(acts):
        aa = np.asarray(a, np.float32).reshape(7)
        step_legal(env.sim, aa)
        bil = bool(measure_errors(env.sim).get("bilateral"))
        if prev_bil and not bil:
            contact_loss_events += 1
        prev_bil = bil
        if i % stride == 0 or i == len(acts) - 1:
            s = _sample(env)
            s["t"] = float((i + 1) * DT)
            s["grip_cmd"] = float(aa[6])
            series.append(s)
            dtheta_max = max(dtheta_max, float(s["dtheta_deg"]))
            dtheta_min = min(dtheta_min, float(s["dtheta_deg"]))

    # Phase markers from extrema of dx and dz
    if series:
        dx_abs = [abs(s["dx_mm"]) for s in series]
        # Approximate DX window: while dx_abs is decreasing from start until local min before rebound
        i_dx_end = int(np.argmin(dx_abs))
        dz_at_dx_start = float(series[0]["dz_mm"])
        dz_at_dx_end = float(series[i_dx_end]["dz_mm"])

    # Return/regrip window: last 20% of tape where alpha moves toward DOWN
    n = len(series)
    i_ret0 = max(0, int(0.75 * n))
    dz_ret_start = float(series[i_ret0]["dz_mm"]) if n else None
    dz_ret_end = float(series[-1]["dz_mm"]) if n else None

    # θ deterioration classification: compare obj_hand_rel vs dtheta
    # If dθ rises but obj_hand_rel stays flat → rigid co-rotation; if rel rises → relative.
    if len(series) >= 3:
        dth = np.array([s["dtheta_deg"] for s in series], float)
        rel = np.array([s["obj_hand_rel_deg"] for s in series], float)
        i_peak = int(np.argmax(dth))
        dth_rise = float(dth[i_peak] - dth[0])
        rel_rise = float(rel[i_peak] - rel[0]) if np.isfinite(rel).all() else float("nan")
        if dth_rise > 2.0 and (not np.isfinite(rel_rise) or abs(rel_rise) < 1.0):
            theta_mode = "rigid_co_rotation"
        elif dth_rise > 2.0 and rel_rise > 1.5:
            theta_mode = "relative_rotation"
        elif dth_rise <= 2.0:
            theta_mode = "stable_or_improved"
        else:
            theta_mode = "mixed_or_unclear"
    else:
        theta_mode = "insufficient_samples"
        dth_rise = 0.0
        rel_rise = float("nan")

    return {
        "series": series,
        "n_steps": len(acts),
        "stride": stride,
        "dtheta0": dtheta0,
        "dtheta_max": dtheta_max,
        "dtheta_min": dtheta_min,
        "max_dtheta_deterioration": float(dtheta_max - dtheta0),
        "dz_at_dx_start": dz_at_dx_start,
        "dz_at_dx_end": dz_at_dx_end,
        "dz_return_start": dz_ret_start,
        "dz_return_end": dz_ret_end,
        "dz_return_drift": (None if dz_ret_start is None else float(dz_ret_end - dz_ret_start)),
        "contact_loss_events": contact_loss_events,
        "theta_mode": theta_mode,
        "dtheta_rise_to_peak": dth_rise if series else 0.0,
        "rel_orient_rise_to_peak": rel_rise if series else float("nan"),
        "final": series[-1] if series else None,
    }
