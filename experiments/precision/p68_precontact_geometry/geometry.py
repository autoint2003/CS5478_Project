"""Geometry probes for pre-contact / first-contact analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall


def relative_geometry(sim) -> dict[str, Any]:
    """Object–gripper relative pose + working errors."""
    wk = working_geometry(sim)
    m = state_metrics(sim)
    ids = sim.ids
    po = np.asarray(sim.data.xpos[ids.object_body], float)
    ph = np.asarray(sim.data.xpos[ids.hand_body], float)
    # approach: hand linear velocity in world
    try:
        vh = np.asarray(sim.data.cvel[ids.hand_body][3:], float)  # spatial vel; use xpos diff proxy
    except Exception:  # noqa: BLE001
        vh = np.zeros(3)
    # Prefer ctrl-ish: object relative vector from hand
    rel = po - ph
    rel_n = float(np.linalg.norm(rel))
    approach = rel / max(rel_n, 1e-9)
    pb = pad_bilateral(sim)
    audit = classify_contacts(sim)
    n_contacts = int(audit.get("n_contacts") or 0)
    return {
        "e_pos_mm": float(wk["e_pos_mm"]),
        "theta_deg": float(wk["dtheta_deg"]),
        "dx_mm": float(wk.get("dx_mm") or 0),
        "dy_mm": float(wk.get("dy_mm") or 0),
        "dz_mm": float(wk.get("dz_mm") or 0),
        "hand_pos": ph.tolist(),
        "obj_pos": po.tolist(),
        "rel_hand_to_obj": rel.tolist(),
        "rel_dist_m": rel_n,
        "approach_unit": approach.tolist(),
        "bilateral": bool(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "min_pad": int(min(pb["nL"], pb["nR"])),
        "n_contacts": n_contacts,
        "correct_sidewall": bool(is_correct_sidewall(audit)),
        "grasp_kind": str(audit.get("grasp_kind") or ""),
        "maj_L": audit.get("maj_L"),
        "maj_R": audit.get("maj_R"),
        "opposing_sidewall": int(audit.get("opposing_sidewall") or 0),
        "vrel": float(m.get("vrel") or 0),
        "has_pad_contact": bool(pb["nL"] > 0 or pb["nR"] > 0),
        "hand_vel_proxy": vh.tolist(),
    }


def contact_points_summary(sim) -> dict[str, Any]:
    audit = classify_contacts(sim)
    pts = []
    for c in audit.get("contacts") or []:
        pts.append(
            {
                "side": c.get("side"),
                "surface": c.get("surface"),
                "pos_w": c.get("pos_w"),
                "rho": c.get("rho"),
                "z": c.get("z"),
                "Fn": c.get("Fn"),
            }
        )
    return {
        "points": pts,
        "n": len(pts),
        "grasp_kind": audit.get("grasp_kind"),
        "correct_sidewall": bool(is_correct_sidewall(audit)),
    }
