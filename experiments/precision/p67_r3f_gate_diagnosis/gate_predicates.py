"""Independent R3F gate predicates — mirrors DwellTracker / R3FOnline (source of truth).

Does NOT invent conditions. Force balance is NOT part of the gate.
WORKING here means working_geometry pose (θ≤10°, e_pos≤12 mm), not an FSM phase flag.
"""

from __future__ import annotations

from typing import Any

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.support_drop import latched
from evaluation.success_criteria import HOLD_S, WORKING_POS_MM, WORKING_THETA_DEG
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry
from experiments.precision.p43_r3_working_hold_v2.r3_rules import (
    MARGIN_POS_MM,
    MARGIN_THETA_DEG,
    MIN_PAD_CONTACTS,
)
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from models.transformer_policy import DT_POLICY


PREDICATE_NAMES = (
    "not_latched",
    "working_theta_ok",  # θ ≤ WORKING_THETA_DEG (10°)
    "working_pos_ok",  # e_pos ≤ WORKING_POS_MM (12 mm)
    "working_ok",  # both WORKING pose predicates
    "bilateral",
    "min_pad_ok",  # min(nL,nR) ≥ 4
    "r3_theta_ok",  # θ ≤ 8°
    "r3_pos_ok",  # e_pos ≤ 9 mm
    "margin_ok",  # r3_theta_ok ∧ r3_pos_ok
    "frame_ok",  # all instantaneous R3 frame conditions
    "dwell_complete",  # continuous ok_s ≥ HOLD_S (from tracker)
)


def gate_spec() -> dict[str, Any]:
    return {
        "source": "experiments/precision/p43_r3_working_hold_v2/r3_rules.py::DwellTracker + p44 R3FOnline",
        "HOLD_S": float(HOLD_S),
        "DT_POLICY": float(DT_POLICY),
        "dwell_steps": int(round(float(HOLD_S) / float(DT_POLICY))),
        "WORKING_THETA_DEG": float(WORKING_THETA_DEG),
        "WORKING_POS_MM": float(WORKING_POS_MM),
        "R3_MARGIN_THETA_DEG": float(MARGIN_THETA_DEG),
        "R3_MARGIN_POS_MM": float(MARGIN_POS_MM),
        "MIN_PAD_CONTACTS": int(MIN_PAD_CONTACTS),
        "require_bilateral": True,
        "force_balance_enforced": False,
        "fsm_phase_enforced": False,
        "note_WORKING": (
            "R3F docstring says WORKING; implementation uses working_geometry "
            "(θ/e_pos only). No FSM WORKING-phase check in DwellTracker."
        ),
        "note_force": "Contact-force balance is NOT an R3F predicate.",
        "fallbacks": ["TIMEOUT_FALLBACK at t_max", "TAPE_END_FALLBACK (dataset path)"],
        "predicates": list(PREDICATE_NAMES),
    }


def evaluate_predicates(sim, *, ok_s: float, dwell_s: float = float(HOLD_S)) -> dict[str, Any]:
    """Evaluate every independent R3 frame predicate at the current sim state."""
    is_latched = bool(latched(sim))
    if is_latched:
        return {
            "not_latched": False,
            "working_theta_ok": False,
            "working_pos_ok": False,
            "working_ok": False,
            "bilateral": False,
            "min_pad_ok": False,
            "r3_theta_ok": False,
            "r3_pos_ok": False,
            "margin_ok": False,
            "frame_ok": False,
            "dwell_complete": False,
            "latched": True,
            "theta": None,
            "e_pos": None,
            "nL": 0,
            "nR": 0,
            "ok_s": float(ok_s),
            "correct_sidewall": False,
            "grasp_kind": "latched",
        }

    wk = working_geometry(sim)
    theta = float(wk["dtheta_deg"])
    e_pos = float(wk["e_pos_mm"])
    working_theta_ok = bool(theta <= float(WORKING_THETA_DEG) + 1e-9)
    working_pos_ok = bool(e_pos <= float(WORKING_POS_MM) + 1e-9)
    working_ok = bool(wk["working_ok"])  # same as both
    pb = pad_bilateral(sim)
    nL, nR = int(pb["nL"]), int(pb["nR"])
    bilateral = bool(pb["bilateral"])
    min_pad_ok = min(nL, nR) >= int(MIN_PAD_CONTACTS)
    r3_theta_ok = bool(theta <= float(MARGIN_THETA_DEG) + 1e-9)
    r3_pos_ok = bool(e_pos <= float(MARGIN_POS_MM) + 1e-9)
    margin_ok = r3_theta_ok and r3_pos_ok
    frame_ok = bool(working_ok and bilateral and min_pad_ok and margin_ok)
    dwell_complete = bool(float(ok_s) + 1e-9 >= float(dwell_s))

    audit = classify_contacts(sim)
    return {
        "not_latched": True,
        "working_theta_ok": working_theta_ok,
        "working_pos_ok": working_pos_ok,
        "working_ok": working_ok,
        "bilateral": bilateral,
        "min_pad_ok": min_pad_ok,
        "r3_theta_ok": r3_theta_ok,
        "r3_pos_ok": r3_pos_ok,
        "margin_ok": margin_ok,
        "frame_ok": frame_ok,
        "dwell_complete": dwell_complete,
        "latched": False,
        "theta": theta,
        "e_pos": e_pos,
        "dx_mm": float(wk.get("dx_mm") or 0),
        "dy_mm": float(wk.get("dy_mm") or 0),
        "dz_mm": float(wk.get("dz_mm") or 0),
        "nL": nL,
        "nR": nR,
        "ok_s": float(ok_s),
        "correct_sidewall": bool(is_correct_sidewall(audit)),
        "grasp_kind": str(audit.get("grasp_kind") or ""),
        "fn_asym": None,  # filled by caller from state_metrics if available
    }


def first_false_after_near(preds: dict[str, bool]) -> list[str]:
    """Predicates false when frame_ok is false (multi-label; not exclusive)."""
    if preds.get("frame_ok"):
        return []
    keys = (
        "not_latched",
        "working_ok",
        "working_theta_ok",
        "working_pos_ok",
        "bilateral",
        "min_pad_ok",
        "margin_ok",
        "r3_theta_ok",
        "r3_pos_ok",
    )
    return [k for k in keys if not preds.get(k)]
