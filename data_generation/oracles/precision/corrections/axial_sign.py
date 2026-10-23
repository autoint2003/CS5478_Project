"""Sign-aware axial precision modes (offline Oracle).

Physical opening axis from MuJoCo geometry — not a hardcoded hand-z sign guess.
Mode A (outward): cylinder too far toward palm → slide toward fingertips.
Mode B (inward): cylinder too far toward fingertips → slide toward palm,
preferring palm/finger-base support when contacts provide it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.mechanics.dual_loop import (
    DT,
    ForcePController,
    contact_forces,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import (
    _height_hold_v_hand,
    gravity_hand,
    score_gravity_alignment,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import AXIS_V1_EZ_MM
from evaluation.grasp_geometry import hand_R
from evaluation.success_criteria import P_TGT

G_WORLD = np.array([0.0, 0.0, -9.81], float)
# Engineering targets for reorientation-only validation (not SUCCESS_* criteria)
REORI_DX_MM = 3.0
REORI_DTHETA_DEG = 3.0
G_AXIAL_USEFUL_MIN = 6.0


def opening_axis_hand(sim) -> dict[str, Any]:
    """Verify palm→fingertip opening direction from finger body placement."""
    Rh = hand_R(sim)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    pl = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    pr = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    fl_h = Rh.T @ (pl - ph)
    fr_h = Rh.T @ (pr - ph)
    mid = 0.5 * (fl_h + fr_h)
    # Primary: mean finger position in hand frame (Panda MJCF: fingers at +z)
    u = mid / max(float(np.linalg.norm(mid)), 1e-12)
    # Align to nearest hand basis axis for a clean unit opening vector
    basis = np.eye(3)
    align = basis @ u
    axis_idx = int(np.argmax(np.abs(align)))
    u_open = basis[axis_idx] * float(np.sign(align[axis_idx]) or 1.0)
    return {
        "u_open": u_open.tolist(),
        "u_open_raw_from_fingers": u.tolist(),
        "finger_L_h": fl_h.tolist(),
        "finger_R_h": fr_h.tolist(),
        "nearest_hand_axis": ["x", "y", "z"][axis_idx],
        "toward_fingertips": u_open.tolist(),
        "toward_palm": (-u_open).tolist(),
        "hand_z_equals_u_open": bool(axis_idx == 2 and u_open[2] > 0),
        "note": "Panda finger bodies attach at hand local +z; u_open = palm→fingertips.",
    }


def signed_opening_error(sim, u_open: np.ndarray | None = None) -> dict[str, Any]:
    s = sample_state(sim)
    e = np.asarray(s["e_signed_mm"], float) * 1e-3
    if u_open is None:
        u_open = np.asarray(opening_axis_hand(sim)["u_open"], float)
    else:
        u_open = np.asarray(u_open, float)
    u_open = u_open / max(float(np.linalg.norm(u_open)), 1e-12)
    e_open = float(np.dot(e, u_open))
    return {
        "e_signed_mm": (e * 1e3).tolist(),
        "e_open_mm": e_open * 1e3,
        "u_open": u_open.tolist(),
        "p_c_h": s["p_c_h"],
        "P_TGT": np.asarray(P_TGT, float).tolist(),
        "theta_deg": s["e_theta_deg"],
        "e_pos_mm": s["e_pos_mm"],
        "bilateral": s["bilateral"],
        "g_h": s["g_h"],
        "interpretation": (
            "e_open>0: too far toward fingertips (need inward / toward palm); "
            "e_open<0: too far toward palm (need outward / toward fingertips)"
        ),
    }


def select_axial_mode(
    e_open_mm: float,
    *,
    axial_tol_mm: float = AXIS_V1_EZ_MM,
) -> dict[str, Any]:
    """Return mode A/B/none from physical opening-axis error."""
    if e_open_mm > axial_tol_mm:
        d = np.array([0.0, 0.0, 0.0])  # filled below with -u_open by caller
        return {
            "mode": "B_inward",
            "needed": True,
            "e_open_mm": e_open_mm,
            "desire_sign_along_u_open": -1.0,
            "gravity_goal": "g·u_open < 0 (opening upward / gravity toward palm)",
            "support_strategy": "prefer palm or finger-base upward world support",
        }
    if e_open_mm < -axial_tol_mm:
        return {
            "mode": "A_outward",
            "needed": True,
            "e_open_mm": e_open_mm,
            "desire_sign_along_u_open": +1.0,
            "gravity_goal": "g·u_open > 0 (opening downward / gravity toward fingertips)",
            "support_strategy": "finger friction + fingertip overlap margin",
        }
    return {
        "mode": "none",
        "needed": False,
        "e_open_mm": e_open_mm,
        "desire_sign_along_u_open": 0.0,
        "note": "within provisional axial tolerance — prefer direct HANDOFF",
    }


def desire_direction(mode: str, u_open: np.ndarray) -> np.ndarray:
    u = np.asarray(u_open, float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    if mode == "A_outward":
        return u.copy()
    if mode == "B_inward":
        return -u
    return np.zeros(3)


def support_force_analysis(sim) -> dict[str, Any]:
    """Classify palm/finger-base vs finger-pad support from contact normals."""
    Rh = hand_R(sim)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    cf = contact_forces(sim)
    up_w = -G_WORLD / float(np.linalg.norm(G_WORLD))
    rows = []
    F_up_palm = 0.0
    F_up_finger = 0.0
    F_up_total = 0.0
    for c in cf["contacts"]:
        nw = np.asarray(c["normal_world"], float)
        # mj_contactForce normal: from geom2→geom1 convention varies; use force magnitude
        # along world-up of the contact normal * fn (support if n·up > 0 on object)
        # Approximate: contact normal in world from force frame first column
        n = nw / max(float(np.linalg.norm(nw)), 1e-12)
        # Force on object ≈ −n * fn or +n*fn depending on which body; take max upward component
        f_up = float(c["fn"] * abs(np.dot(n, up_w)))  # magnitude contribution proxy
        pos_w = np.asarray(c["pos"], float)
        pos_h = Rh.T @ (pos_w - ph)
        # Finger-base / near-palm: small |z| or z closer to palm (z < ~0.04)
        # Pad / distal: larger z along opening
        kind = "finger_distal" if pos_h[2] > 0.04 else "finger_base_or_proximal"
        if abs(pos_h[2]) < 0.02 and abs(pos_h[0]) < 0.02:
            kind = "near_palm_region"
        if kind in ("near_palm_region", "finger_base_or_proximal"):
            F_up_palm += f_up
        else:
            F_up_finger += f_up
        F_up_total += f_up
        rows.append(
            {
                "side": c["side"],
                "fn": c["fn"],
                "pos_h": pos_h.tolist(),
                "n_hand": (Rh.T @ n).tolist(),
                "kind": kind,
                "f_up_proxy": f_up,
            }
        )
    return {
        "bilateral": cf["bilateral"],
        "fn_L": cf["fn_L"],
        "fn_R": cf["fn_R"],
        "F_up_proximal_proxy": F_up_palm,
        "F_up_distal_proxy": F_up_finger,
        "F_up_total_proxy": F_up_total,
        "proximal_fraction": F_up_palm / max(F_up_total, 1e-9),
        "palm_or_base_support_detected": bool(F_up_palm > 0.5 and any(r["kind"] != "finger_distal" for r in rows)),
        "contacts": rows,
        "note": (
            "Proxy from contact normals × Fn; not a free-body solve. "
            "palm_or_base_support_detected requires proximal contact locations, not opening direction alone."
        ),
    }


def fingertip_overlap_margin_mm(sim, u_open: np.ndarray) -> float:
    """Distal contact margin: max contact z_open − COM z_open (mm).

    Positive ⇒ contacts still distal of COM along opening (overlap remaining).
    Uses actual contact positions; does not assume a fixed tip length.
    """
    Rh = hand_R(sim)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    u = np.asarray(u_open, float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    s = sample_state(sim)
    com_along = float(np.dot(np.asarray(s["p_c_h"], float), u))
    cf = contact_forces(sim)
    if not cf["contacts"]:
        return -1e3
    zs = []
    for c in cf["contacts"]:
        pos_h = Rh.T @ (np.asarray(c["pos"], float) - ph)
        zs.append(float(np.dot(pos_h, u)))
    return (max(zs) - com_along) * 1e3


def waypoint_reorient(
    env,
    blob: dict,
    *,
    d: np.ndarray,
    axis_name: str,
    angle_deg: float,
    rate_scale: float,
    fn_target: float = 12.0,
    waypoint_deg: float = 10.0,
    dx_abort_mm: float = REORI_DX_MM,
    dtheta_abort_deg: float = REORI_DTHETA_DEG,
    g_useful_min: float = G_AXIAL_USEFUL_MIN,
) -> dict[str, Any]:
    """Slow signed wrist rotation with intermediate stability checks."""
    axis_map = {
        "+wy": np.array([0.0, 1.0, 0.0]),
        "-wy": np.array([0.0, -1.0, 0.0]),
        "+wx": np.array([1.0, 0.0, 0.0]),
        "-wx": np.array([-1.0, 0.0, 0.0]),
    }
    if axis_name not in axis_map:
        return {"ok": False, "reason": f"unknown_axis_{axis_name}"}
    axis = axis_map[axis_name]
    d = np.asarray(d, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    e0 = np.asarray(s0["e_signed_mm"], float)
    th0 = float(s0["e_theta_deg"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    omega = float(W_MAX) * float(rate_scale)
    n_total = max(1, int(round(np.radians(angle_deg) / max(omega, 1e-6) / DT)))
    n_wp = max(1, int(round(np.radians(waypoint_deg) / max(omega, 1e-6) / DT)))

    ctl = ForcePController(kp=0.12, da_max=0.10)
    ctl.reset(1.0, s0["fn_tot"])
    actions = []
    waypoints = []
    lost = 0
    support_series = []

    for i in range(n_total):
        st = sample_state(env.sim)
        a_g, _ = ctl.step(st["fn_tot"], fn_target)
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, axis * omega, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        st2 = sample_state(env.sim)
        e2 = np.asarray(st2["e_signed_mm"], float)
        dx = abs(float(e2[0]) - float(e0[0]))
        dth = abs(float(st2["e_theta_deg"]) - th0)
        if failed or latched(env.sim) or not st2["bilateral"]:
            return {
                "ok": False,
                "reason": "contact_loss",
                "actions": actions,
                "waypoints": waypoints,
                "n_steps": i + 1,
                "axis_name": axis_name,
                "angle_deg": angle_deg,
                "rate_scale": rate_scale,
            }
        if dx > dx_abort_mm or dth > dtheta_abort_deg:
            return {
                "ok": False,
                "reason": "reori_drift_abort",
                "actions": actions,
                "waypoints": waypoints,
                "n_steps": i + 1,
                "axis_name": axis_name,
                "angle_deg": angle_deg,
                "rate_scale": rate_scale,
                "e0": e0.tolist(),
                "e_now": e2.tolist(),
                "theta0": th0,
                "theta_now": st2["e_theta_deg"],
                "delta_ex_mm": float(e2[0] - e0[0]),
                "delta_theta_deg": float(st2["e_theta_deg"] - th0),
                "capture": take_capture(env),
            }
        if (i + 1) % n_wp == 0 or i == n_total - 1:
            g_sc = score_gravity_alignment(np.asarray(st2["g_h"], float), d)
            sup = support_force_analysis(env.sim)
            wp = {
                "step": i + 1,
                "angle_approx_deg": (i + 1) * float(np.degrees(omega * DT)),
                "e_signed_mm": e2.tolist(),
                "theta": st2["e_theta_deg"],
                "g_h": st2["g_h"],
                "useful": g_sc["useful"],
                "unwanted": g_sc["unwanted"],
                "delta_ex_mm": float(e2[0] - e0[0]),
                "delta_theta_deg": float(st2["e_theta_deg"] - th0),
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "support": {
                    "proximal_fraction": sup["proximal_fraction"],
                    "palm_or_base_support_detected": sup["palm_or_base_support_detected"],
                    "F_up_proximal_proxy": sup["F_up_proximal_proxy"],
                },
            }
            waypoints.append(wp)
            support_series.append(sup)

    sf = sample_state(env.sim)
    g_sc = score_gravity_alignment(np.asarray(sf["g_h"], float), d)
    e_f = np.asarray(sf["e_signed_mm"], float)
    ok = (
        sf["bilateral"]
        and abs(float(e_f[0]) - float(e0[0])) <= dx_abort_mm
        and abs(float(sf["e_theta_deg"]) - th0) <= dtheta_abort_deg
    )
    return {
        "ok": ok,
        "clears_gravity": bool(g_sc["useful"] >= g_useful_min),
        "reason": "ok" if ok else "end_state_out_of_reori_budget",
        "actions": actions,
        "waypoints": waypoints,
        "n_steps": n_total,
        "axis_name": axis_name,
        "angle_deg": angle_deg,
        "rate_scale": rate_scale,
        "fn_target": fn_target,
        "e0": e0.tolist(),
        "e_final": e_f.tolist(),
        "theta0": th0,
        "theta_final": sf["e_theta_deg"],
        "delta_ex_mm": float(e_f[0] - e0[0]),
        "delta_theta_deg": float(sf["e_theta_deg"] - th0),
        "g_h": sf["g_h"],
        "useful": g_sc["useful"],
        "unwanted": g_sc["unwanted"],
        "support_final": support_force_analysis(env.sim),
        "min_obj_z": float(min(w["obj_z"] for w in waypoints)) if waypoints else float(env.sim.data.xpos[env.sim.ids.object_body][2]),
        "capture": take_capture(env),
        "d": d.tolist(),
    }


def search_stable_reorientation(
    env,
    blob: dict,
    *,
    mode: str,
    u_open: np.ndarray,
    angles: tuple[float, ...] = (25.0, 35.0, 45.0, 55.0, 70.0),
    rates: tuple[float, ...] = (0.15, 0.25, 0.35),
    axes: tuple[str, ...] = ("+wy", "-wy"),
    g_useful_min: float = G_AXIAL_USEFUL_MIN,
) -> dict[str, Any]:
    """Bounded search: prefer lateral/theta preservation, then gravity useful."""
    d = desire_direction(mode, u_open)
    trials = []
    best = None
    for axis_name in axes:
        for ang in angles:
            for rate in rates:
                tr = waypoint_reorient(
                    env,
                    blob,
                    d=d,
                    axis_name=axis_name,
                    angle_deg=ang,
                    rate_scale=rate,
                    g_useful_min=g_useful_min,
                )
                summary = {k: v for k, v in tr.items() if k not in ("actions", "waypoints", "capture", "support_final")}
                summary["n_waypoints"] = len(tr.get("waypoints") or [])
                if tr.get("support_final"):
                    summary["palm_support"] = tr["support_final"].get("palm_or_base_support_detected")
                    summary["proximal_fraction"] = tr["support_final"].get("proximal_fraction")
                trials.append(summary)
                score = None
                if tr.get("ok"):
                    score = (
                        (50.0 if tr.get("clears_gravity") else 0.0)
                        + 3.0 * float(tr.get("useful") or 0.0)
                        - 2.0 * abs(float(tr.get("delta_ex_mm") or 0.0))
                        - 2.0 * abs(float(tr.get("delta_theta_deg") or 0.0))
                        - 0.1 * ang
                        - 5.0 * rate
                    )
                    summary["score"] = score
                    if best is None or score > best["score"]:
                        best = {**tr, "score": score}
                print(
                    f"  reori {mode} {axis_name}@{ang}° r={rate}: ok={tr.get('ok')} "
                    f"useful={tr.get('useful')} Δex={tr.get('delta_ex_mm')} Δθ={tr.get('delta_theta_deg')} "
                    f"reason={tr.get('reason')}",
                    flush=True,
                )
    return {
        "mode": mode,
        "d": d.tolist(),
        "n_trials": len(trials),
        "trials": trials,
        "best": None
        if best is None
        else {k: v for k, v in best.items() if k not in ("actions", "waypoints")},
        "best_full": best,
        "any_ok": best is not None,
        "any_clears_gravity": bool(best and best.get("clears_gravity")),
    }
