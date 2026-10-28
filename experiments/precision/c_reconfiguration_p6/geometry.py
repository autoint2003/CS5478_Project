"""Contact / cylinder-frame geometry helpers for P6 audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.plan_c.catch_stabilization import grasp_geometry
from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.precision.residuals import measure_errors
from envs.simulation.ground_scene import CYL_HALF


def snapshot_geometry(sim) -> dict[str, Any]:
    e = measure_errors(sim)
    surf = classify_contacts(sim)
    geo = grasp_geometry(sim)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    Ro = np.asarray(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    u = Ro[:, 2]
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    s_mean = surf.get("contact_s_mean")
    if s_mean is None or not np.isfinite(float(s_mean)):
        s_mean = geo.get("contact_s_mean")
    s_mean = float(s_mean) if s_mean is not None and np.isfinite(float(s_mean)) else float("nan")
    near_end = bool(np.isfinite(s_mean) and abs(s_mean) >= (CYL_HALF - 0.008))
    central = bool(np.isfinite(s_mean) and abs(s_mean) <= 0.012)
    dist_end = float(geo.get("dist_nearest_end_m") or float("nan"))
    if not np.isfinite(dist_end) and np.isfinite(s_mean):
        dist_end = float(CYL_HALF - abs(s_mean))
    kind = str(surf.get("grasp_kind") or "")
    placement_ok = bool(
        central
        and int(surf.get("n_end_face") or 0) == 0
        and (
            int(surf.get("opposing_sidewall") or 0) == 1
            or kind.startswith("opposing")
            or (kind.startswith("sidewall") and dist_end >= 0.015)
        )
    )
    return {
        "dx": float(e["dx_mm"]),
        "dy": float(e["dy_mm"]),
        "dz": float(e["dz_mm"]),
        "dtheta": float(e["dtheta_deg"]),
        "e_pos": float(e["e_pos_mm"]),
        "bilateral": bool(e.get("bilateral")),
        "fn_tot": float(e.get("fn_tot") or 0),
        "obj_z": float(e.get("obj_z") or po[2]),
        "p_obj": po.tolist(),
        "u_cyl_world": u.tolist(),
        "grasp_kind": surf.get("grasp_kind"),
        "contact_s_mean": s_mean,
        "n_end_face": int(surf.get("n_end_face") or 0),
        "n_sidewall": int(surf.get("n_sidewall") or 0),
        "opposing_sidewall": int(surf.get("opposing_sidewall") or 0),
        "dist_nearest_end_m": dist_end,
        "near_end": near_end,
        "CONTACT_PLACEMENT_VALID": placement_ok,
        "cyl_half": float(CYL_HALF),
    }
