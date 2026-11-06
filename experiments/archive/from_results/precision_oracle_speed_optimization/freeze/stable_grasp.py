"""Shared stable-grasp boundary (offline diagnostic).

Stabilization ≠ SUCCESS_NEW. A grasp may be stable with θ>10° or e_pos>12 mm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.support_drop import latched
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT

DT = 0.02

# Justified from Plan C catch hold (60–150 ms bilateral) and Plan B recatch
# persistence (first_bilateral_after persist=3 → 60 ms). We require a longer
# continuous bilateral window for a "stable grasp" gate before precision work.
STABLE_THRESHOLDS = {
    "min_bilateral_hold_s": 0.15,  # ≥150 ms continuous bilateral at end of hold
    "max_v_rel_m_s": 0.08,  # controlled linear relative motion
    "max_w_rel_rad_s": 1.5,  # controlled angular relative motion
    "min_obj_z_m": 0.20,  # clearly airborne (GROUND_DROP_Z=0.04 is far below)
    "min_fn_n": 1.0,  # nontrivial normal load each side (load-bearing)
    "require_opposing_sidewall": False,  # preferred but not mandatory for all plans
    "hold_s_default": 0.20,  # diagnostic hold length (not written into expert tape)
}


def _geom(sim) -> dict[str, Any]:
    m = measure(sim, U_TGT, P_TGT)
    pack = physical_pack(sim)
    pb = pad_bilateral(sim)
    surf = classify_contacts(sim)
    v = np.asarray(pack["v_rel_h"], float)
    w = np.asarray(pack["w_rel_h"], float)
    e = np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)
    return {
        "bilateral": bool(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "Fn_L": float(pb["fnL"]),
        "Fn_R": float(pb["fnR"]),
        "obj_z": float(m["obj_z"]),
        "ground_drop": bool(latched(sim)),
        "v_rel_h": v.tolist(),
        "v_rel_norm": float(np.linalg.norm(v)),
        "w_rel_h": w.tolist(),
        "w_rel_norm": float(np.linalg.norm(w)),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_signed_mm": (e * 1e3).tolist(),
        "grasp_kind": surf.get("grasp_kind"),
        "opposing_sidewall": int(surf.get("opposing_sidewall") or 0),
        "contact_s_mean": surf.get("contact_s_mean"),
        "n_sidewall": surf.get("n_sidewall"),
        "n_end_face": surf.get("n_end_face"),
    }


def evaluate_stable_grasp(sim, *, thresholds: dict | None = None) -> dict[str, Any]:
    """Single-instant gate (still requires load-bearing bilateral + velocity bounds).

    Prefer `hold_and_confirm_stable` for acceptance — a single frame is insufficient.
    """
    th = {**STABLE_THRESHOLDS, **(thresholds or {})}
    g = _geom(sim)
    reasons = []
    if g["ground_drop"]:
        reasons.append("GROUND_DROP")
    if g["obj_z"] < float(th["min_obj_z_m"]):
        reasons.append("OBJ_Z_LOW")
    if not g["bilateral"]:
        reasons.append("NOT_BILATERAL")
    if g["Fn_L"] < float(th["min_fn_n"]) or g["Fn_R"] < float(th["min_fn_n"]):
        reasons.append("LOW_CONTACT_FORCE")
    if g["v_rel_norm"] > float(th["max_v_rel_m_s"]):
        reasons.append("V_REL_HIGH")
    if g["w_rel_norm"] > float(th["max_w_rel_rad_s"]):
        reasons.append("W_REL_HIGH")
    if th.get("require_opposing_sidewall") and not g["opposing_sidewall"]:
        reasons.append("NO_OPPOSING_SIDEWALL")
    ok = len(reasons) == 0
    return {
        "stable_instant": ok,
        "reasons": reasons,
        "geom": g,
        "thresholds": th,
        # Explicitly not SUCCESS_NEW
        "SUCCESS_NEW_relevant": False,
        "note": "Instant gate only; use hold_and_confirm_stable for verified stability",
    }


def hold_and_confirm_stable(
    env,
    *,
    hold_s: float | None = None,
    thresholds: dict | None = None,
    grip_action_7d: np.ndarray | None = None,
) -> dict[str, Any]:
    """Diagnostic short physical hold. Does not mutate caller intent for expert tapes.

    Steps secure-hold actions for `hold_s` and requires continuous bilateral for
    `min_bilateral_hold_s` at the end. Returns geometry time series evidence.
    """
    th = {**STABLE_THRESHOLDS, **(thresholds or {})}
    hold_s = float(th["hold_s_default"] if hold_s is None else hold_s)
    n = max(1, int(round(hold_s / DT)))
    if grip_action_7d is None:
        # Secure brake / hold: zeros + grip channel from current ctrl if available
        a = np.zeros(7, np.float32)
        a[6] = 1.0
    else:
        a = np.asarray(grip_action_7d, np.float32).reshape(7)

    series = []
    consec = 0.0
    max_consec = 0.0
    for i in range(n):
        step_legal(env.sim, a)
        g = _geom(env.sim)
        series.append({"step": i, "t_rel": (i + 1) * DT, **{k: g[k] for k in g if k not in ("v_rel_h", "w_rel_h")}})
        if g["bilateral"] and not g["ground_drop"]:
            consec += DT
            max_consec = max(max_consec, consec)
        else:
            consec = 0.0

    end = evaluate_stable_grasp(env.sim, thresholds=th)
    continuous_ok = consec + 1e-12 >= float(th["min_bilateral_hold_s"])
    stable = bool(end["stable_instant"] and continuous_ok)
    reasons = list(end["reasons"])
    if not continuous_ok:
        reasons.append(f"BILATERAL_HOLD_SHORT({consec:.3f}<{th['min_bilateral_hold_s']})")

    return {
        "stable": stable,
        "reasons": reasons,
        "consecutive_bil_s_at_end": float(consec),
        "max_consecutive_bil_s": float(max_consec),
        "hold_s": hold_s,
        "n_steps": n,
        "end": end,
        "series_tail": series[-8:],
        "geom": end["geom"],
        # Final pose errors are recorded but do NOT gate stability
        "e_theta_deg": end["geom"]["e_theta_deg"],
        "e_pos_mm": end["geom"]["e_pos_mm"],
        "e_signed_mm": end["geom"]["e_signed_mm"],
    }


# Alias matching the conceptual interface name in the task brief.
def is_stable_grasp(env, *, short_rollout: bool = True, **kwargs) -> dict[str, Any]:
    if short_rollout:
        return hold_and_confirm_stable(env, **kwargs)
    return evaluate_stable_grasp(env.sim, **kwargs)
