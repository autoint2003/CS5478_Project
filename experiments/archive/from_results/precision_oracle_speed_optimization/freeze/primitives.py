"""Shared precision-manipulation primitives (offline privileged search, legal 7D exec).

Each primitive returns actions + physical evidence. Callers must not treat wrist
co-rotation as relative pivot, or rigid co-motion as axial sliding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3, grip_action, grip_tau, normalize_action
from data_generation.oracles.plan_c.catch_stabilization import grasp_geometry
from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable
from data_generation.oracles.representability import representability
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.ground_scene import CYL_HALF
from envs.step_legal import step_legal
from evaluation.grasp_geometry import cylinder_axis_hand, cylinder_axis_world, measure
from evaluation.success_criteria import P_TGT, U_TGT
from scripts.generate_plan_c_experts import evaluate_handoff_from_state

DT = 0.02
TAIL_S_MIN = 0.018
TAIL_S_MAX = float(CYL_HALF) - 0.004
DS_TAIL_MIN = 0.003
DS_REL_PIVOT_MIN_DEG = 8.0
# Position-preserving pivot: keep object COM fixed in the hand frame.
# SUCCESS_NEW allows e_pos ≤ 12 mm; a precision pivot must not spend that budget.
# 3 mm ≈ 1/4 of the success band and well below the 10–15 mm coupled-slip regime.
MAX_DELTA_P_REL_MM = 3.0
MIN_THETA_IMPROVE_DEG = 3.0


def _wrist(wx, wy, grip) -> np.ndarray:
    return np.asarray(
        normalize_action(np.zeros(3), np.array([wx, wy, 0.0], float), grip_tau(float(grip)), AIRBORNE_V3),
        np.float32,
    )


def _action_vw(v_hand, w_body, grip) -> np.ndarray:
    return np.asarray(
        normalize_action(
            np.asarray(v_hand, float),
            np.asarray(w_body, float),
            grip_tau(float(grip)),
            AIRBORNE_V3,
        ),
        np.float32,
    )


def _legal(book, a) -> bool:
    packed = representability(np.asarray(a, np.float32).reshape(1, 7), book)
    return float(packed["max_linf"]) <= 0.25 + 1e-8


def relative_pose(sim) -> dict[str, Any]:
    """Cylinder COM and axis in the hand frame (pivot diagnostics)."""
    m = measure(sim, U_TGT, P_TGT)
    p = np.asarray(m["p_c_h"], float)
    u = np.asarray(m["u_c_h"], float)
    return {
        "p_rel": p.tolist(),
        "u_hand": u.tolist(),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "v_rel_norm": float(np.linalg.norm(m["v_rel_h"])),
        "w_rel_norm": float(np.linalg.norm(m["w_rel_h"])),
        "bilateral": bool(m["bilateral"]),
        "nL": int(m["nL"]),
        "nR": int(m["nR"]),
    }


def classify_pivot_motion(
    *,
    delta_p_rel_mm: float,
    d_axis_deg: float,
    d_theta: float,
    bilateral_end: bool,
    max_delta_p_mm: float = MAX_DELTA_P_REL_MM,
) -> str:
    """Classify a candidate pivot segment (mutually exclusive labels)."""
    if not bilateral_end:
        return "uncontrolled_slip_or_loss_of_contact"
    if d_axis_deg < 1.0 and abs(d_theta) < 1.0:
        # Little relative rotation: either near-static / rigid co-motion, or pure slip.
        if delta_p_rel_mm <= max_delta_p_mm:
            return "rigid_hand_object_rotation"
        return "coupled_rotation_translation"
    if delta_p_rel_mm <= max_delta_p_mm and d_axis_deg >= MIN_THETA_IMPROVE_DEG:
        return "position_preserving_relative_pivot"
    if d_axis_deg >= 1.0 and delta_p_rel_mm > max_delta_p_mm:
        return "coupled_rotation_translation"
    if d_axis_deg < MIN_THETA_IMPROVE_DEG:
        return "rigid_hand_object_rotation"
    return "coupled_rotation_translation"


def pivot_segment_metrics(env, blob, actions, *, record_series: bool = True) -> dict[str, Any]:
    """Execute actions from blob capture; measure relative COM / orientation change."""
    restore_capture(env, blob["capture"])
    p0 = relative_pose(env.sim)
    a0 = audit_grasp(env.sim)
    series = []
    lost = 0
    ok = True
    reason = None
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        if failed or latched(env.sim):
            ok = False
            reason = "DROP"
            break
        if record_series and (i % 2 == 0 or i == len(actions) - 1):
            pr = relative_pose(env.sim)
            p_now = np.asarray(pr["p_rel"], float)
            p_start = np.asarray(p0["p_rel"], float)
            series.append(
                {
                    "step": i,
                    "t_rel": (i + 1) * DT,
                    "p_rel": pr["p_rel"],
                    "delta_p_from_start_mm": (1e3 * (p_now - p_start)).tolist(),
                    "delta_p_norm_mm": float(1e3 * np.linalg.norm(p_now - p_start)),
                    "e_theta_deg": pr["e_theta_deg"],
                    "bilateral": pr["bilateral"],
                    "v_rel_norm": pr["v_rel_norm"],
                    "w_rel_norm": pr["w_rel_norm"],
                }
            )
    p1 = relative_pose(env.sim)
    a1 = audit_grasp(env.sim)
    dp = np.asarray(p1["p_rel"], float) - np.asarray(p0["p_rel"], float)
    u0 = np.asarray(p0["u_hand"], float)
    u1 = np.asarray(p1["u_hand"], float)
    c = float(np.clip(np.dot(u0 / (np.linalg.norm(u0) + 1e-12), u1 / (np.linalg.norm(u1) + 1e-12)), -1, 1))
    d_axis_deg = float(np.degrees(np.arccos(c)))
    d_theta = float(p1["e_theta_deg"] - p0["e_theta_deg"])
    delta_p_mm = float(1e3 * np.linalg.norm(dp))
    label = classify_pivot_motion(
        delta_p_rel_mm=delta_p_mm,
        d_axis_deg=d_axis_deg,
        d_theta=d_theta,
        bilateral_end=bool(p1["bilateral"] and ok),
    )
    return {
        "ok": ok,
        "reason": reason,
        "p0": p0,
        "p1": p1,
        "a0": a0,
        "a1": a1,
        "delta_p_rel_mm": delta_p_mm,
        "delta_p_rel_components_mm": (1e3 * dp).tolist(),
        "d_axis_deg": d_axis_deg,
        "d_theta": d_theta,
        "theta0": p0["e_theta_deg"],
        "theta1": p1["e_theta_deg"],
        "classification": label,
        "series": series,
        "max_delta_p_rel_mm_allowed": MAX_DELTA_P_REL_MM,
    }


def audit_grasp(sim) -> dict[str, Any]:
    surf = classify_contacts(sim)
    geo = grasp_geometry(sim)
    m = measure(sim, U_TGT, P_TGT)
    pack = physical_pack(sim)
    u_w = cylinder_axis_world(sim)
    u_h = cylinder_axis_hand(sim)
    g_h = np.asarray(pack["g_h"], float)
    e_h = np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)
    s = surf.get("contact_s_mean")
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        s = geo.get("contact_s_mean")
    sL = [c["contact_s"] for c in (surf.get("contacts") or []) if c.get("side") == "L"]
    sR = [c["contact_s"] for c in (surf.get("contacts") or []) if c.get("side") == "R"]
    return {
        "contact_s_mean": None if s is None else float(s),
        "contact_s_L": None if not sL else float(np.mean(sL)),
        "contact_s_R": None if not sR else float(np.mean(sR)),
        "dist_nearest_end_m": geo.get("dist_nearest_end_m"),
        "near_end": geo.get("near_end"),
        "grasp_kind": surf.get("grasp_kind"),
        "opposing_sidewall": int(surf.get("opposing_sidewall") or 0),
        "n_sidewall": surf.get("n_sidewall"),
        "n_end_face": surf.get("n_end_face"),
        "u_world": u_w.tolist(),
        "u_hand": u_h.tolist(),
        "g_h": g_h.tolist(),
        "g_dot_u_hand": float(np.dot(g_h, u_h)),
        "g_axial_world": float(np.dot(np.array([0.0, 0.0, -9.81]), u_w)),
        "e_signed_mm": (e_h * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "bilateral": bool(surf.get("bilateral")),
        "Fn_L": float(pack["Fn_L"]),
        "Fn_R": float(pack["Fn_R"]),
        "v_rel_norm": float(np.linalg.norm(pack["v_rel_h"])),
        "w_rel_norm": float(np.linalg.norm(pack["w_rel_h"])),
        "w_rel_h": np.asarray(pack["w_rel_h"], float).tolist(),
        "obj_z": float(m["obj_z"]),
        "ground_drop": bool(latched(sim)),
        "cyl_half": float(CYL_HALF),
    }


def _run(env, blob, actions) -> tuple[bool, str | None]:
    lost = 0
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        if failed or latched(env.sim):
            return False, "DROP"
    return True, None


def primitive_axial_tail_slide(env, blob, book, *, budget: int = 48) -> dict[str, Any]:
    """Primitive 1: migrate contact_s toward nearer cylinder end (true sliding)."""
    restore_capture(env, blob["capture"])
    a0 = audit_grasp(env.sim)
    near = a0["near_end"] or "plus"
    tail_sign = +1.0 if near == "plus" else -1.0
    s0 = a0["contact_s_mean"]
    cands = []
    unload_opts = (-0.48, float(grip_action(-5.0)))
    menu = []
    for wx, wy in ((0.0, 3.0), (0.0, -1.0), (1.12, 2.0), (-1.12, 2.0), (1.12, 0.0), (-1.12, 0.0)):
        for tilt_s in (0.35, 0.55):
            for slip_s in (0.45, 0.70):
                for unload in unload_opts:
                    menu.append((wx, wy, tilt_s, slip_s, unload))
    for wx, wy, tilt_s, slip_s, unload in menu[:budget]:
        a_tilt, a_slip, a_brake = _wrist(wx, wy, 1.0), _wrist(0.0, 0.0, unload), _wrist(0.0, 0.0, 1.0)
        if not all(_legal(book, a) for a in (a_tilt, a_slip, a_brake)):
            continue
        acts = (
            [a_tilt] * max(1, int(round(tilt_s / DT)))
            + [a_slip] * max(1, int(round(slip_s / DT)))
            + [a_brake] * max(1, int(round(0.20 / DT)))
        )
        restore_capture(env, blob["capture"])
        ok_run, _ = _run(env, blob, acts)
        a1 = audit_grasp(env.sim)
        if not ok_run or not a1["bilateral"] or s0 is None or a1["contact_s_mean"] is None:
            continue
        ds = float(a1["contact_s_mean"] - s0)
        toward = (ds * tail_sign) > 0
        d_near = float((a0["dist_nearest_end_m"] or 0) - (a1["dist_nearest_end_m"] or 0))
        abs_s1 = abs(a1["contact_s_mean"])
        tail_band = TAIL_S_MIN <= abs_s1 <= TAIL_S_MAX
        not_face = a1["grasp_kind"] != "end_face_pinch" and int(a1.get("n_end_face") or 0) <= 2
        # True sliding evidence: axial contact migration + nearer-end approach
        demonstrated = toward and abs(ds) >= DS_TAIL_MIN and d_near >= 0.002 and not_face
        score = (1000 * abs(ds) * (1 if toward else -1)) + 500 * d_near + (40 if demonstrated else 0)
        cands.append(
            {
                "tag": f"wx{wx}_wy{wy}_t{tilt_s}_s{slip_s}_g{unload}",
                "actions": acts,
                "s0": s0,
                "s1": a1["contact_s_mean"],
                "ds": ds,
                "d_dist_nearest": d_near,
                "toward_tail": toward,
                "tail_band": tail_band,
                "demonstrated": demonstrated,
                "a0": a0,
                "a1": a1,
                "score": score,
            }
        )
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = next((c for c in cands if c["demonstrated"]), None)
    return {
        "primitive": "axial_tail_slide",
        "implemented": True,
        "physically_demonstrated": bool(best is not None),
        "intended_tail": near,
        "tail_sign": tail_sign,
        "n_screened": len(cands),
        "best": None
        if best is None
        else {k: v for k, v in best.items() if k != "actions"},
        "actions": None if best is None else best["actions"],
        "failure": None
        if best is not None
        else {
            "mechanism": "no_legal_schedule_produced_measurable_axial_migration_toward_tail",
            "best_attempt": None if not cands else {k: cands[0][k] for k in ("tag", "ds", "toward_tail", "a1") if k in cands[0]},
        },
    }


def primitive_contact_pivot(env, blob, book, *, budget: int = 48) -> dict[str, Any]:
    """Primitive 2: position-preserving relative pivot (rotate without COM slip in hand frame).

    Hand Cartesian motion may compensate; object p_rel must stay within MAX_DELTA_P_REL_MM.
    Orientation improvement (θ decrease) is required — not mere Δu_hand from coupled slip.
    """
    restore_capture(env, blob["capture"])
    a0 = audit_grasp(env.sim)
    menu = []
    # Prefer compensating translation early: object COM must stay fixed in hand frame.
    for vx in (-0.20, -0.12, -0.06, 0.0, 0.06, 0.12, 0.20):
        for vz in (-0.15, -0.08, 0.0, 0.08, 0.15, 0.25):
            for wy in (3.0, -3.0, 1.0, -1.0, 2.0):
                for grip in (1.0, -0.48, float(grip_action(-5.0))):
                    for dur in (0.25, 0.40):
                        menu.append((vx, 0.0, vz, 0.0, wy, grip, dur))
    for wx, wy in ((0.0, 1.0), (0.0, -1.0), (0.0, 3.0), (1.12, 2.0), (-1.12, 2.0)):
        for grip in (1.0, -0.48):
            for dur in (0.30, 0.45):
                menu.append((0.0, 0.0, 0.0, wx, wy, grip, dur))
    # Deterministic subsample when budget < full menu (spread across menu)
    if budget < len(menu):
        idx = np.linspace(0, len(menu) - 1, num=budget, dtype=int)
        menu = [menu[i] for i in idx]
    cands = []
    for vx, vy, vz, wx, wy, grip, dur in menu:
        a_p = _action_vw([vx, vy, vz], [wx, wy, 0.0], grip)
        a_b = _wrist(0.0, 0.0, 1.0)
        if not _legal(book, a_p):
            continue
        acts = [a_p] * max(1, int(round(dur / DT))) + [a_b] * 8
        met = pivot_segment_metrics(env, blob, acts, record_series=False)
        if not met["ok"] or not met["p1"]["bilateral"]:
            continue
        theta_improve = float(met["theta0"] - met["theta1"])  # positive = better
        pos_ok = met["delta_p_rel_mm"] <= MAX_DELTA_P_REL_MM
        ori_ok = theta_improve >= MIN_THETA_IMPROVE_DEG and met["d_axis_deg"] >= MIN_THETA_IMPROVE_DEG
        face_ok = int(met["a1"].get("n_end_face") or 0) <= 4 or bool(met["a1"].get("opposing_sidewall"))
        demonstrated = bool(pos_ok and ori_ok and face_ok and met["classification"] == "position_preserving_relative_pivot")
        score = (
            50.0 * theta_improve
            - 8.0 * met["delta_p_rel_mm"]
            + (40.0 if demonstrated else 0.0)
            - 0.2 * abs(met["a1"]["e_pos_mm"] - met["a0"]["e_pos_mm"])
        )
        cands.append(
            {
                "tag": f"ppiv_v{vx}_{vz}_w{wx}_{wy}_g{grip}_d{dur}",
                "actions": acts,
                "d_axis_deg": met["d_axis_deg"],
                "theta0": met["theta0"],
                "theta1": met["theta1"],
                "d_theta": met["d_theta"],
                "theta_improve_deg": theta_improve,
                "delta_p_rel_mm": met["delta_p_rel_mm"],
                "delta_p_rel_components_mm": met["delta_p_rel_components_mm"],
                "classification": met["classification"],
                "a0": met["a0"],
                "a1": met["a1"],
                "demonstrated": demonstrated,
                "score": score,
            }
        )
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = next((c for c in cands if c["demonstrated"]), None)
    # Also surface best-by-score even if not demonstrated (for diagnostics)
    best_any = cands[0] if cands else None
    return {
        "primitive": "position_preserving_relative_pivot",
        "implemented": True,
        "physically_demonstrated": bool(best is not None),
        "max_delta_p_rel_mm": MAX_DELTA_P_REL_MM,
        "min_theta_improve_deg": MIN_THETA_IMPROVE_DEG,
        "n_screened": len(cands),
        "best": None if best is None else {k: v for k, v in best.items() if k != "actions"},
        "best_any": None if best_any is None else {k: v for k, v in best_any.items() if k != "actions"},
        "actions": None if best is None else best["actions"],
        "failure": None
        if best is not None
        else {
            "mechanism": (
                f"no_legal_schedule_improved_θ_by_≥{MIN_THETA_IMPROVE_DEG}° "
                f"with Δp_rel≤{MAX_DELTA_P_REL_MM} mm and stable bilateral"
            ),
            "best_attempt": None
            if best_any is None
            else {
                k: best_any[k]
                for k in (
                    "tag",
                    "d_axis_deg",
                    "d_theta",
                    "theta_improve_deg",
                    "delta_p_rel_mm",
                    "delta_p_rel_components_mm",
                    "classification",
                    "demonstrated",
                )
                if k in best_any
            },
        },
    }


def primitive_gravity_reorient(env, blob, book, *, g_axial_target: float = 5.0, budget: int = 20) -> dict[str, Any]:
    """Primitive 3: establish |g·u_hand| ≥ target via legal wrist motion (or confirm already sufficient)."""
    restore_capture(env, blob["capture"])
    a0 = audit_grasp(env.sim)
    g0 = abs(float(a0["g_dot_u_hand"]))
    if g0 >= g_axial_target and a0["bilateral"] and not a0["ground_drop"]:
        brake = [_wrist(0.0, 0.0, 1.0)] * 5
        return {
            "primitive": "gravity_reorient",
            "implemented": True,
            "physically_demonstrated": True,
            "already_sufficient": True,
            "g_axial_target": g_axial_target,
            "n_screened": 0,
            "best": {
                "tag": "already_sufficient",
                "g_dot_u0": a0["g_dot_u_hand"],
                "g_dot_u1": a0["g_dot_u_hand"],
                "g_h1": a0["g_h"],
                "a1": a0,
                "demonstrated": True,
            },
            "actions": brake,
            "failure": None,
        }
    cands = []
    for wx, wy, s in (
        (0.0, 3.0, 0.50),
        (0.0, -1.0, 0.80),
        (0.0, 3.0, 0.70),
        (0.0, 1.0, 0.60),
        (1.12, 2.0, 0.50),
        (-1.12, 2.0, 0.50),
        (0.0, 3.0, 1.00),
        (0.0, -1.0, 1.20),
    )[:budget]:
        a_t = _wrist(wx, wy, 1.0)
        a_b = _wrist(0.0, 0.0, 1.0)
        if not _legal(book, a_t):
            continue
        acts = [a_t] * max(1, int(round(s / DT))) + [a_b] * 5
        restore_capture(env, blob["capture"])
        ok_run, _ = _run(env, blob, acts)
        a1 = audit_grasp(env.sim)
        if not ok_run or not a1["bilateral"]:
            continue
        g1 = abs(float(a1["g_dot_u_hand"]))
        demonstrated = g1 >= g_axial_target and g1 > g0 + 1.0
        cands.append(
            {
                "tag": f"reori_wx{wx}_wy{wy}_s{s}",
                "actions": acts,
                "g_dot_u0": a0["g_dot_u_hand"],
                "g_dot_u1": a1["g_dot_u_hand"],
                "g_h1": a1["g_h"],
                "a1": a1,
                "demonstrated": demonstrated,
                "score": g1 + (30 if demonstrated else 0),
            }
        )
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = next((c for c in cands if c["demonstrated"]), None)
    return {
        "primitive": "gravity_reorient",
        "implemented": True,
        "physically_demonstrated": bool(best is not None),
        "already_sufficient": False,
        "g_axial_target": g_axial_target,
        "n_screened": len(cands),
        "best": None if best is None else {k: v for k, v in best.items() if k != "actions"},
        "actions": None if best is None else best["actions"],
        "failure": None
        if best is not None
        else {
            "mechanism": f"could_not_reach_|g·u_hand|≥{g_axial_target} while retaining bilateral",
            "best_attempt": None if not cands else {k: cands[0][k] for k in ("tag", "g_dot_u1", "demonstrated")},
        },
    }


def primitive_final_axial_slide(env, blob, book, *, budget: int = 24) -> dict[str, Any]:
    """Primitive 4: improve e_pos while preserving θ as much as possible."""
    restore_capture(env, blob["capture"])
    a0 = audit_grasp(env.sim)
    cands = []
    for wx, wy, tilt_s, slip_s, unload in (
        (0.0, 3.0, 0.40, 0.50, float(grip_action(-5.0))),
        (0.0, -1.0, 0.50, 0.50, -0.48),
        (0.0, 1.0, 0.40, 0.70, float(grip_action(-5.0))),
        (1.12, 2.0, 0.35, 0.50, -0.48),
        (0.0, 3.0, 0.55, 0.70, -0.48),
        (0.0, -1.0, 0.70, 0.70, float(grip_action(-5.0))),
    )[:budget]:
        a_tilt, a_slip, a_brake = _wrist(wx, wy, 1.0), _wrist(0.0, 0.0, unload), _wrist(0.0, 0.0, 1.0)
        if not all(_legal(book, a) for a in (a_tilt, a_slip, a_brake)):
            continue
        acts = (
            [a_tilt] * max(1, int(round(tilt_s / DT)))
            + [a_slip] * max(1, int(round(slip_s / DT)))
            + [a_brake] * 10
        )
        restore_capture(env, blob["capture"])
        ok_run, _ = _run(env, blob, acts)
        a1 = audit_grasp(env.sim)
        if not ok_run or not a1["bilateral"]:
            continue
        d_pos = float(a0["e_pos_mm"] - a1["e_pos_mm"])
        d_th = float(abs(a1["e_theta_deg"] - a0["e_theta_deg"]))
        demonstrated = d_pos >= 1.0 and a1["e_theta_deg"] <= a0["e_theta_deg"] + 5.0
        cands.append(
            {
                "tag": f"final_wx{wx}_wy{wy}",
                "actions": acts,
                "e_pos0": a0["e_pos_mm"],
                "e_pos1": a1["e_pos_mm"],
                "e_signed0": a0["e_signed_mm"],
                "e_signed1": a1["e_signed_mm"],
                "theta0": a0["e_theta_deg"],
                "theta1": a1["e_theta_deg"],
                "d_pos_mm": d_pos,
                "d_theta_abs": d_th,
                "demonstrated": demonstrated,
                "score": d_pos - 0.3 * d_th + (20 if demonstrated else 0),
                "a1": a1,
            }
        )
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = next((c for c in cands if c["demonstrated"]), None)
    return {
        "primitive": "final_controlled_slide",
        "implemented": True,
        "physically_demonstrated": bool(best is not None),
        "n_screened": len(cands),
        "best": None if best is None else {k: v for k, v in best.items() if k != "actions"},
        "actions": None if best is None else best["actions"],
        "failure": None
        if best is not None
        else {
            "mechanism": "no_slide_improved_e_pos_by_≥1mm_without_θ_blowup",
            "best_attempt": None if not cands else {k: cands[0][k] for k in ("tag", "d_pos_mm", "theta1", "demonstrated")},
        },
    }


def primitive_stabilize_handoff(env, blob, actions_prefix: list | None = None) -> dict[str, Any]:
    """Primitive 5: brake hold + HANDOFF + SUCCESS_NEW evaluation (full 12 s)."""
    restore_capture(env, blob["capture"])
    acts = list(actions_prefix or [])
    brake = _wrist(0.0, 0.0, 1.0)
    acts = acts + [brake] * 8
    ok_run, reason = _run(env, blob, acts)
    if not ok_run:
        return {
            "primitive": "brake_handoff",
            "implemented": True,
            "physically_demonstrated": False,
            "SUCCESS_NEW": False,
            "SUCCESS_12S": False,
            "reason": reason,
            "actions": acts,
        }
    stab = hold_and_confirm_stable(env, hold_s=0.12)
    # Note: hold steps are diagnostic only — for SUCCESS_NEW we re-exec prefix+brake without extra hold
    restore_capture(env, blob["capture"])
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    meta = evaluate_handoff_from_state(env.sim, blob["ref"], stabilize_n=0)
    win = meta.get("window") or {}
    return {
        "primitive": "brake_handoff",
        "implemented": True,
        "physically_demonstrated": bool(meta.get("SUCCESS_12S") or meta.get("reason") == "SUCCESS_12S"),
        "stable_before_handoff": stab.get("stable"),
        "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
        "reason": meta.get("reason"),
        "pre_theta": meta.get("pre_theta"),
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "longest_both_ok_s": win.get("longest_both_ok_s"),
        "actions": acts,
    }


def classify_stable_needs(geom: dict) -> str:
    """Route label for the shared Oracle (not a hard-coded fixed sequence)."""
    th, pos = float(geom.get("e_theta_deg") or 99), float(geom.get("e_pos_mm") or 99)
    s = geom.get("contact_s_mean")
    abs_s = abs(float(s)) if s is not None and np.isfinite(float(s)) else None
    if th <= 10.0 and pos <= 12.0:
        return "already_in_band"
    if th <= 10.0 and pos > 12.0:
        return "position_only"
    if th > 10.0 and pos <= 12.0:
        return "orientation_only"
    if abs_s is not None and abs_s < TAIL_S_MIN:
        return "needs_tail_before_pivot"
    return "needs_both"
