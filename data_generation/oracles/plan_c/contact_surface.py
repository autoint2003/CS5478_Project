"""Offline contact-surface classification for Plan C cylinder grasps.

Privileged geometry for oracle/diagnostics only. Not part of the 39D runtime obs.
Classifies finger–cylinder contacts as sidewall / end_face / edge using MuJoCo
contact points and normals in the cylinder body frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.contacts.frames import extract_contacts
from envs.simulation.ground_scene import CYL_HALF, CYL_R

# Numerical margins (meters / dimensionless). Contacts penetrate slightly.
Z_FACE_TOL = 0.003  # |z| within this of half-height → face-eligible
R_WALL_TOL = 0.004  # rho within this of R → wall-eligible
EDGE_BAND = 0.004  # near both boundaries → edge
N_FACE_DOT = 0.75  # |n·ez| above → normal points along axis (face-like)
N_WALL_DOT = 0.45  # |n·ez| below → normal mostly radial (wall-like)


def cylinder_frame_contact(sim, pos_w: np.ndarray, n_w: np.ndarray) -> dict[str, float]:
    """Map world contact point/normal into cylinder body frame."""
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    Ro = np.asarray(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    p_loc = Ro.T @ (np.asarray(pos_w, float) - po)
    n_loc = Ro.T @ np.asarray(n_w, float)
    n_loc = n_loc / max(float(np.linalg.norm(n_loc)), 1e-12)
    rho = float(np.hypot(p_loc[0], p_loc[1]))
    z = float(p_loc[2])
    n_axial = float(abs(n_loc[2]))
    n_radial = float(np.hypot(n_loc[0], n_loc[1]))
    return {
        "rho": rho,
        "z": z,
        "abs_z": abs(z),
        "n_axial": n_axial,
        "n_radial": n_radial,
        "p_loc": p_loc,
        "n_loc": n_loc,
    }


def classify_point(rho: float, abs_z: float, n_axial: float) -> str:
    """Label one contact from cylinder-frame geometry + normal."""
    near_face_z = abs_z >= (CYL_HALF - Z_FACE_TOL)
    near_wall_r = rho >= (CYL_R - R_WALL_TOL)
    deep_face_r = rho <= (CYL_R - EDGE_BAND)
    deep_wall_z = abs_z <= (CYL_HALF - Z_FACE_TOL)

    # Prefer normal evidence when position is ambiguous.
    if n_axial >= N_FACE_DOT and near_face_z:
        return "end_face"
    if n_axial <= N_WALL_DOT and near_wall_r and deep_wall_z:
        return "sidewall"
    if near_face_z and near_wall_r:
        return "edge"
    if near_face_z and deep_face_r:
        return "end_face"
    if near_wall_r and deep_wall_z:
        return "sidewall"
    if near_face_z:
        return "edge"
    if near_wall_r:
        return "sidewall"
    return "unknown"


def classify_contacts(sim) -> dict[str, Any]:
    """Full finger–cylinder contact-surface audit for the current sim state."""
    rows, summary = extract_contacts(sim)
    classified = []
    for r in rows:
        if r.get("side") not in ("L", "R"):
            continue
        geo = cylinder_frame_contact(
            sim,
            np.asarray(r["pos_w"], float),
            np.asarray(r["n_w"], float),
        )
        label = classify_point(geo["rho"], geo["abs_z"], geo["n_axial"])
        classified.append(
            {
                "side": r["side"],
                "geom1": int(r["geom1"]),
                "geom2": int(r["geom2"]),
                "geom1_name": r["geom1_name"],
                "geom2_name": r["geom2_name"],
                "pos_w": [float(x) for x in r["pos_w"]],
                "n_w": [float(x) for x in r["n_w"]],
                "Fn": float(r["Fn"]),
                "rho": geo["rho"],
                "z": geo["z"],
                "abs_z": geo["abs_z"],
                "n_axial": geo["n_axial"],
                "n_radial": geo["n_radial"],
                "surface": label,
                # Axial COM coordinate used by contact_s (same definition).
                "contact_s": float(geo["z"]),
            }
        )

    def _side(side: str) -> list[dict]:
        return [c for c in classified if c["side"] == side]

    left, right = _side("L"), _side("R")

    def _majority(side_rows: list[dict]) -> str:
        if not side_rows:
            return "none"
        counts: dict[str, int] = {}
        for c in side_rows:
            counts[c["surface"]] = counts.get(c["surface"], 0) + 1
        return max(counts, key=counts.get)

    maj_L, maj_R = _majority(left), _majority(right)
    n_sidewall = sum(1 for c in classified if c["surface"] == "sidewall")
    n_face = sum(1 for c in classified if c["surface"] == "end_face")
    n_edge = sum(1 for c in classified if c["surface"] == "edge")

    # Opposing sidewall: both sides present, majority sidewall, normals anti-aligned.
    oppose = False
    n_dot = None
    lat_sep = None
    if left and right:
        # Force-weighted mean normals / positions
        def _wmean(side_rows, key):
            w = np.array([max(abs(c["Fn"]), 1e-6) for c in side_rows], float)
            v = np.stack([np.asarray(c[key], float) for c in side_rows], axis=0)
            return (w[:, None] * v).sum(axis=0) / w.sum()

        nL = _wmean(left, "n_w")
        nR = _wmean(right, "n_w")
        pL = _wmean(left, "pos_w")
        pR = _wmean(right, "pos_w")
        nL = nL / max(float(np.linalg.norm(nL)), 1e-12)
        nR = nR / max(float(np.linalg.norm(nR)), 1e-12)
        n_dot = float(np.dot(nL, nR))
        po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
        Ro = np.asarray(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
        # Lateral separation in cylinder xy (perp to axis)
        dL = Ro.T @ (pL - po)
        dR = Ro.T @ (pR - po)
        lat_L = dL[:2]
        lat_R = dR[:2]
        lat_sep = float(np.dot(lat_L, lat_R))
        oppose = bool(
            maj_L == "sidewall"
            and maj_R == "sidewall"
            and n_dot < -0.2
            and lat_sep < 0.0
        )

    # contact_s_mean as used elsewhere (axial only — no surface identity)
    s_vals = [c["contact_s"] for c in classified]
    contact_s_mean = float(np.mean(s_vals)) if s_vals else float("nan")

    if oppose and n_face == 0:
        grasp_kind = "opposing_sidewall"
    elif n_face > 0 and n_sidewall == 0:
        grasp_kind = "end_face_pinch"
    elif n_face > 0 and n_sidewall > 0:
        grasp_kind = "mixed_face_wall"
    elif n_edge > 0 and n_sidewall == 0 and n_face == 0:
        grasp_kind = "edge_only"
    elif n_sidewall > 0:
        grasp_kind = "sidewall_partial"
    elif not classified:
        grasp_kind = "no_pad_contact"
    else:
        grasp_kind = "unknown"

    return {
        "contacts": classified,
        "n_contacts": len(classified),
        "nL": len(left),
        "nR": len(right),
        "bilateral": bool(left and right),
        "maj_L": maj_L,
        "maj_R": maj_R,
        "n_sidewall": n_sidewall,
        "n_end_face": n_face,
        "n_edge": n_edge,
        "opposing_sidewall": int(oppose),
        "normal_dot_LR": n_dot,
        "lateral_sep_dot": lat_sep,
        "grasp_kind": grasp_kind,
        "contact_s_mean": contact_s_mean,
        "contact_s_only_axial": True,
        "cyl_r": float(CYL_R),
        "cyl_half": float(CYL_HALF),
        "t": float(summary.get("t", sim.data.time)),
    }


def surface_score(audit: dict[str, Any]) -> float:
    """Offline ranking bonus favoring correct opposing sidewall grasps."""
    kind = audit.get("grasp_kind")
    score = 0.0
    if kind == "opposing_sidewall":
        score += 40.0
    elif kind == "sidewall_partial":
        score += 15.0
    elif kind == "mixed_face_wall":
        score -= 10.0
    elif kind == "end_face_pinch":
        score -= 35.0
    elif kind == "edge_only":
        score -= 20.0
    if int(audit.get("opposing_sidewall") or 0):
        score += 10.0
    # Prefer mid-body contacts (|s| small) when sidewall
    s = audit.get("contact_s_mean")
    if s is not None and np.isfinite(float(s)) and kind in ("opposing_sidewall", "sidewall_partial"):
        score += 8.0 * max(0.0, 1.0 - abs(float(s)) / max(CYL_HALF, 1e-6))
    return float(score)


def score_catch_out(out: dict[str, Any]) -> float:
    """Offline Plan C catch ranking: surface + stability + orientation, not bil-duration alone.

    Keeps unusual but physically valid sidewall grasps; heavily penalizes end-face pinches.
    Does not require SUCCESS_NEW / exact final pose at first contact.
    """
    from data_generation.oracles.plan_c.interception import (
        FULL_DETACH_ACTIVE_RECATCH,
        PARTIAL_SLIP_RECOVERY,
        POST_IMPACT_GRASP_PRESERVATION,
    )

    score = (
        100.0 * int(out.get("SUCCESS_NEW") or 0)
        + 45.0 * int(out.get("mechanism") == FULL_DETACH_ACTIVE_RECATCH)
        + 30.0 * int(out.get("mechanism") == PARTIAL_SLIP_RECOVERY)
        + 25.0 * int(out.get("mechanism") == POST_IMPACT_GRASP_PRESERVATION)
        + 12.0 * int(out.get("continuous_150") or 0)
        + 6.0 * int(out.get("continuous_60") or 0)
        + 8.0 * int(out.get("jaw_entered") or 0)
        + float(out.get("max_consec_bil") or 0)
        - 5.0 * int(out.get("ground_drop") or 0)
    )
    # Contact-surface geometry (privileged offline)
    audit = out.get("surface_end") or {}
    score += surface_score(audit)
    kind = audit.get("grasp_kind") or out.get("grasp_kind")
    if kind == "end_face_pinch":
        score -= 50.0  # additional: do not prefer long bil on wrong surface
    elif kind == "opposing_sidewall":
        score += 25.0

    # Orientation compatibility with desired hand-frame cylinder pose
    theta = out.get("e_theta_end")
    if theta is not None and np.isfinite(float(theta)):
        th = float(theta)
        if th <= 15.0:
            score += 20.0 * (1.0 - th / 15.0)
        elif th >= 60.0:
            score -= 25.0 * min(1.0, (th - 60.0) / 30.0)

    # Low relative velocity at first contact
    vrel = out.get("vrel_at_first_contact")
    if vrel is not None and np.isfinite(float(vrel)):
        score += 8.0 * max(0.0, 1.0 - float(vrel) / 0.80)

    # Lift / brake feasibility proxies already in continuous_* and end_bilateral
    if int(out.get("end_bilateral") or 0) and kind in ("opposing_sidewall", "sidewall_partial"):
        score += 10.0
    return float(score)
