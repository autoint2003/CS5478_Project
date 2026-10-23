"""Geometry-inferred wrist orientation for gravity-assisted precision slip.

Offline Oracle only. Infers tilt axis / sign / magnitude from current cylinder–hand
geometry. Does not memorize a fixed 60° wy sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import W_MAX
from data_generation.oracles.precision.force_slip_control import (
    DT,
    ForcePController,
    make_action,
    sample_state,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import hand_R
from evaluation.success_criteria import P_TGT

G_WORLD = np.array([0.0, 0.0, -9.81], float)


def _height_hold_v_hand(sim, z_ref: float, kp: float = 2.5, vmax: float = 0.8) -> np.ndarray:
    """Hand-frame velocity to keep object world-z near z_ref during wrist motion."""
    z = float(sim.data.xpos[sim.ids.object_body][2])
    v_w = np.array([0.0, 0.0, float(np.clip(kp * (z_ref - z), -vmax, vmax))])
    Rh = hand_R(sim)
    return Rh.T @ v_w
# Minimum useful gravity projection along desired correction (m/s²).
# Calibrated closed-loop sliding used ~9.7 m/s²; weaker tilts lose contact mid-slip.
G_USEFUL_MIN = 9.0
# Prefer smallest feasible angle among candidates that clear the threshold.
# Angles >90° are needed when d has a large −hand-z component (g must tip past horizontal).
ANGLE_DEGS = (25.0, 35.0, 45.0, 55.0, 60.0, 70.0, 85.0, 100.0, 115.0)
RATE_SCALES = (0.45, 0.60)


def signed_position_error_m(sim) -> np.ndarray:
    s = sample_state(sim)
    return np.asarray(s["e_signed_mm"], float) * 1e-3


def desired_correction_direction(e_m: np.ndarray) -> np.ndarray:
    """d = -e / ||e|| — desired cylinder relative COM motion in the hand frame."""
    e = np.asarray(e_m, float).reshape(3)
    n = float(np.linalg.norm(e))
    if n < 1e-6:
        return np.zeros(3)
    return -e / n


def gravity_hand(Rh: np.ndarray, g_w: np.ndarray = G_WORLD) -> np.ndarray:
    return np.asarray(Rh, float).T @ np.asarray(g_w, float)


def score_gravity_alignment(g_h: np.ndarray, d: np.ndarray) -> dict[str, float]:
    d = np.asarray(d, float)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        return {"useful": 0.0, "unwanted": float(np.linalg.norm(g_h)), "align": 0.0}
    d = d / dn
    useful = float(np.dot(g_h, d))
    unwanted = float(np.linalg.norm(g_h - useful * d))
    return {"useful": useful, "unwanted": unwanted, "align": useful / max(float(np.linalg.norm(g_h)), 1e-9)}


def joint_limit_margin(sim) -> float:
    """Fraction of unused joint range (min over hinges); higher is safer."""
    m = sim.model
    q = np.asarray(sim.data.qpos, float)
    margins = []
    for i in range(int(m.njnt)):
        if int(m.jnt_type[i]) not in (2, 3):  # hinge / slide
            continue
        adr = int(m.jnt_qposadr[i])
        lo, hi = float(m.jnt_range[i, 0]), float(m.jnt_range[i, 1])
        if hi <= lo + 1e-9:
            continue
        x = float(q[adr])
        margins.append(min(x - lo, hi - x) / (hi - lo))
    return float(min(margins)) if margins else 1.0


def _candidate_axes() -> list[tuple[str, np.ndarray]]:
    """Wrist ω directions in the action body frame (same as make_action w_body)."""
    axes = []
    for name, v in (
        ("+wy", np.array([0.0, 1.0, 0.0])),
        ("-wy", np.array([0.0, -1.0, 0.0])),
        ("+wx", np.array([1.0, 0.0, 0.0])),
        ("-wx", np.array([-1.0, 0.0, 0.0])),
        ("+wz", np.array([0.0, 0.0, 1.0])),
        ("-wz", np.array([0.0, 0.0, -1.0])),
        ("+wy+wx", np.array([0.7, 0.7, 0.0])),
        ("-wy+wx", np.array([0.7, -0.7, 0.0])),
        ("+wy-wx", np.array([-0.7, 0.7, 0.0])),
        ("-wy-wx", np.array([-0.7, -0.7, 0.0])),
    ):
        axes.append((name, v / np.linalg.norm(v)))
    return axes


def kinematic_predict_g_h(Rh0: np.ndarray, axis_body: np.ndarray, angle_rad: float) -> np.ndarray:
    """Predict g_h after rotating the hand by angle about body axis (Rodrigues)."""
    k = np.asarray(axis_body, float)
    k = k / max(float(np.linalg.norm(k)), 1e-12)
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], float)
    R_delta = np.eye(3) + s * K + (1 - c) * (K @ K)
    # Body ω rotates the hand: Rh_new ≈ Rh0 @ R_delta (right-multiply in body).
    Rh1 = np.asarray(Rh0, float) @ R_delta
    return gravity_hand(Rh1)


def plan_orientation(
    env,
    blob: dict,
    *,
    d: np.ndarray | None = None,
    g_useful_min: float = G_USEFUL_MIN,
    max_candidates: int = 48,
    fn_target: float = 10.0,
    rate_scales: tuple[float, ...] = RATE_SCALES,
    angle_degs: tuple[float, ...] = ANGLE_DEGS,
    simulate: bool = True,
) -> dict[str, Any]:
    """Infer wrist rotation axis/sign/magnitude from current geometry.

    Returns a plan dict with omega_body, n_steps, rate_scale, scores, and rationale.
    """
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    e = np.asarray(s0["e_signed_mm"], float) * 1e-3
    if d is None:
        d = desired_correction_direction(e)
    d = np.asarray(d, float)
    dn = float(np.linalg.norm(d))
    if dn < 1e-8:
        return {
            "ok": False,
            "reason": "error_already_near_zero",
            "d": d.tolist(),
            "e_signed_mm": s0["e_signed_mm"],
        }

    Rh0 = hand_R(env.sim)
    g0 = gravity_hand(Rh0)
    base_score = score_gravity_alignment(g0, d)
    if base_score["useful"] >= g_useful_min:
        return {
            "ok": True,
            "already_sufficient": True,
            "d": d.tolist(),
            "e_signed_mm": s0["e_signed_mm"],
            "g_h0": g0.tolist(),
            "useful0": base_score["useful"],
            "axis_name": "none",
            "omega_body": [0.0, 0.0, 0.0],
            "angle_deg": 0.0,
            "n_steps": 0,
            "rate_scale": 1.0,
            "fn_target": fn_target,
            "rationale": "current gravity already aligns with desired correction",
        }

    # Rank kinematic candidates first (cheap), then simulate top-K.
    kin = []
    for axis_name, axis in _candidate_axes():
        for ang_deg in angle_degs:
            ang = float(np.radians(ang_deg))
            g_pred = kinematic_predict_g_h(Rh0, axis, ang)
            sc = score_gravity_alignment(g_pred, d)
            # Prefer useful ≥ threshold, then smaller angle, then less unwanted.
            clears = sc["useful"] >= g_useful_min
            kin.append(
                {
                    "axis_name": axis_name,
                    "axis": axis.tolist(),
                    "angle_deg": ang_deg,
                    "useful_pred": sc["useful"],
                    "unwanted_pred": sc["unwanted"],
                    "clears": clears,
                    "kin_rank": (
                        0 if clears else 1,
                        -sc["useful"],
                        ang_deg,
                        sc["unwanted"],
                    ),
                }
            )
    kin.sort(key=lambda r: r["kin_rank"])
    kin = kin[:max_candidates]

    if not simulate:
        best = kin[0]
        omega = W_MAX * 0.45
        n = max(1, int(round(np.radians(best["angle_deg"]) / max(omega, 1e-6) / DT)))
        return {
            "ok": best["clears"],
            "already_sufficient": False,
            "d": d.tolist(),
            "e_signed_mm": s0["e_signed_mm"],
            "g_h0": g0.tolist(),
            "useful0": base_score["useful"],
            "axis_name": best["axis_name"],
            "omega_body": (np.asarray(best["axis"]) * omega).tolist(),
            "angle_deg": best["angle_deg"],
            "n_steps": n,
            "rate_scale": 0.45,
            "fn_target": fn_target,
            "kinematic_only": True,
            "candidates_kin": kin[:8],
        }

    # Physical evaluation of top kinematic candidates × rate scales.
    evaluated = []
    p0 = np.asarray(s0["p_c_h"], float)
    for cand in kin[:16]:
        for rate in rate_scales:
            restore_capture(env, blob["capture"])
            omega = float(W_MAX) * float(rate)
            ang = float(np.radians(cand["angle_deg"]))
            n = max(1, int(round(ang / max(omega, 1e-6) / DT)))
            axis = np.asarray(cand["axis"], float)
            w_body = axis * omega
            ctl = ForcePController(kp=0.10, da_max=0.10)
            ctl.reset(1.0, s0["fn_tot"])
            lost = 0
            ok = True
            drift_max = 0.0
            z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
            for _ in range(n):
                st = sample_state(env.sim)
                a_g, _ = ctl.step(st["fn_tot"], fn_target)
                v_h = _height_hold_v_hand(env.sim, z_ref)
                a = make_action(v_h, w_body, a_g)
                step_legal(env.sim, a)
                failed, lost = recovery_failed(env.sim, blob["ref"], lost)
                st2 = sample_state(env.sim)
                dp = np.linalg.norm(np.asarray(st2["p_c_h"], float) - p0) * 1e3
                drift_max = max(drift_max, dp)
                if failed or latched(env.sim) or not st2["bilateral"]:
                    ok = False
                    break
                # Adaptive slowdown signal recorded only; rate fixed per trial here.
            stf = sample_state(env.sim)
            g_h = np.asarray(stf["g_h"], float)
            sc = score_gravity_alignment(g_h, d)
            jl = joint_limit_margin(env.sim)
            dp_end = (np.asarray(stf["p_c_h"], float) - p0) * 1e3
            clears = ok and sc["useful"] >= g_useful_min and stf["bilateral"]
            # Objective: clear threshold with small angle, low drift, low unwanted.
            score = (
                (100.0 if clears else 0.0)
                + 4.0 * sc["useful"]
                - 0.8 * sc["unwanted"]
                - 1.5 * float(np.linalg.norm(dp_end))
                - 0.15 * cand["angle_deg"]
                + 5.0 * jl
                - (0.0 if ok else 50.0)
            )
            evaluated.append(
                {
                    "axis_name": cand["axis_name"],
                    "axis": cand["axis"],
                    "angle_deg": cand["angle_deg"],
                    "rate_scale": rate,
                    "n_steps": n,
                    "omega_body": w_body.tolist(),
                    "ok": ok,
                    "clears": clears,
                    "useful": sc["useful"],
                    "unwanted": sc["unwanted"],
                    "g_h": g_h.tolist(),
                    "delta_p_rel_mm": dp_end.tolist(),
                    "drift_max_mm": drift_max,
                    "theta": stf["e_theta_deg"],
                    "bilateral": stf["bilateral"],
                    "joint_margin": jl,
                    "score": score,
                    "fn_end": stf["fn_tot"],
                }
            )

    evaluated.sort(key=lambda r: r["score"], reverse=True)
    # Prefer smallest angle among clearing candidates, but not if useful is far
    # below the best clearing useful (need enough drive for controllable slip).
    clearing = [r for r in evaluated if r["clears"]]
    if clearing:
        best_u = max(r["useful"] for r in clearing)
        clearing = [r for r in clearing if r["useful"] >= best_u - 1.5] or clearing
        clearing.sort(key=lambda r: (r["angle_deg"], -r["useful"], r["drift_max_mm"]))
        best = clearing[0]
    else:
        best = evaluated[0] if evaluated else None

    if best is None:
        return {"ok": False, "reason": "no_candidates", "d": d.tolist(), "e_signed_mm": s0["e_signed_mm"]}

    return {
        "ok": bool(best["clears"]),
        "already_sufficient": False,
        "d": d.tolist(),
        "e_signed_mm": s0["e_signed_mm"],
        "g_h0": g0.tolist(),
        "useful0": base_score["useful"],
        "axis_name": best["axis_name"],
        "omega_body": best["omega_body"],
        "angle_deg": best["angle_deg"],
        "n_steps": best["n_steps"],
        "rate_scale": best["rate_scale"],
        "fn_target": fn_target,
        "useful": best["useful"],
        "unwanted": best["unwanted"],
        "g_h": best["g_h"],
        "delta_p_rel_mm": best["delta_p_rel_mm"],
        "drift_max_mm": best["drift_max_mm"],
        "score": best["score"],
        "top": evaluated[:6],
        "rationale": (
            f"minimize angle subject to g·d≥{g_useful_min}; "
            f"chose {best['axis_name']} @ {best['angle_deg']}° rate={best['rate_scale']} "
            f"(useful={best['useful']:.2f}, drift={best['drift_max_mm']:.2f} mm)"
        ),
    }


def execute_orientation_plan(
    env,
    blob: dict,
    plan: dict,
    *,
    adaptive: bool = True,
    drift_slow_mm: float = 1.5,
) -> dict[str, Any]:
    """Execute inferred reorientation with force regulation; optionally slow on drift."""
    restore_capture(env, blob["capture"])
    if plan.get("already_sufficient") or int(plan.get("n_steps") or 0) == 0:
        s = sample_state(env.sim)
        return {
            "ok": True,
            "n_steps": 0,
            "actions": [],
            "g_h": s["g_h"],
            "e_signed_mm": s["e_signed_mm"],
            "delta_p_rel_mm": [0.0, 0.0, 0.0],
            "series": [],
        }

    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    d = np.asarray(plan["d"], float)
    fn_tgt = float(plan.get("fn_target", 10.0))
    w_body = np.asarray(plan["omega_body"], float)
    n = int(plan["n_steps"])
    ctl = ForcePController(kp=0.10, da_max=0.10)
    ctl.reset(1.0, s0["fn_tot"])
    actions = []
    series = []
    lost = 0
    rate_scale = 1.0
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    for i in range(n):
        st = sample_state(env.sim)
        a_g, _ = ctl.step(st["fn_tot"], fn_tgt)
        w = w_body * rate_scale
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, w, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        st2 = sample_state(env.sim)
        dp = (np.asarray(st2["p_c_h"], float) - p0) * 1e3
        if adaptive and float(np.linalg.norm(dp)) > drift_slow_mm:
            rate_scale = max(0.4, rate_scale * 0.85)
        if i % 2 == 0 or i == n - 1:
            series.append(
                {
                    "t": (i + 1) * DT,
                    "g_h": st2["g_h"],
                    "useful": float(np.dot(st2["g_h"], d)),
                    "dpx": float(dp[0]),
                    "dpy": float(dp[1]),
                    "dpz": float(dp[2]),
                    "fn": st2["fn_tot"],
                    "a6": float(a[6]),
                    "theta": st2["e_theta_deg"],
                    "bilateral": st2["bilateral"],
                    "rate_scale": rate_scale,
                    "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                }
            )
        if failed or latched(env.sim) or not st2["bilateral"]:
            return {
                "ok": False,
                "reason": "contact_loss",
                "n_steps": i + 1,
                "actions": actions,
                "series": series,
            }
    # short settle with height hold
    for _ in range(6):
        st = sample_state(env.sim)
        a_g, _ = ctl.step(st["fn_tot"], fn_tgt)
        v_h = _height_hold_v_hand(env.sim, z_ref)
        a = make_action(v_h, [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    sf = sample_state(env.sim)
    dp = (np.asarray(sf["p_c_h"], float) - p0) * 1e3
    return {
        "ok": bool(sf["bilateral"]),
        "n_steps": len(actions),
        "actions": actions,
        "g_h": sf["g_h"],
        "useful": float(np.dot(sf["g_h"], d)),
        "e_signed_mm": sf["e_signed_mm"],
        "e_pos_mm": sf["e_pos_mm"],
        "theta": sf["e_theta_deg"],
        "delta_p_rel_mm": dp.tolist(),
        "series": series,
        "capture": take_capture(env),
    }


def plan_restore_orientation(
    env,
    blob: dict,
    *,
    g_lateral_max: float = 3.5,
    fn_target: float = 12.0,
) -> dict[str, Any]:
    """Infer restore rotation from current state toward upright gravity (hand +z).

    Target: |g_h.x|,|g_h.y| small and g_h.z ≈ +|g| (nominal horizontal grasp).
    """
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    g0 = np.asarray(s0["g_h"], float)
    lateral0 = float(np.linalg.norm(g0[:2]))
    if lateral0 <= g_lateral_max and g0[2] > 7.0:
        return {
            "ok": True,
            "already_upright": True,
            "axis_name": "none",
            "n_steps": 0,
            "omega_body": [0.0, 0.0, 0.0],
            "angle_deg": 0.0,
            "rate_scale": 1.0,
            "fn_target": fn_target,
            "g_h0": g0.tolist(),
        }

    # Desire: g_h → [0,0,+9.81]
    d_up = np.array([0.0, 0.0, 1.0])
    Rh0 = hand_R(env.sim)
    kin = []
    for axis_name, axis in _candidate_axes():
        for ang_deg in ANGLE_DEGS:
            g_pred = kinematic_predict_g_h(Rh0, axis, float(np.radians(ang_deg)))
            lat = float(np.linalg.norm(g_pred[:2]))
            score = (
                (50.0 if lat <= g_lateral_max and g_pred[2] > 7.0 else 0.0)
                + 3.0 * float(g_pred[2])
                - 2.0 * lat
                - 0.1 * ang_deg
            )
            kin.append(
                {
                    "axis_name": axis_name,
                    "axis": axis.tolist(),
                    "angle_deg": ang_deg,
                    "g_pred": g_pred.tolist(),
                    "lateral_pred": lat,
                    "score": score,
                }
            )
    kin.sort(key=lambda r: r["score"], reverse=True)

    evaluated = []
    p0 = np.asarray(s0["p_c_h"], float)
    for cand in kin[:12]:
        for rate in (0.45, 0.60):
            restore_capture(env, blob["capture"])
            omega = float(W_MAX) * rate
            n = max(1, int(round(np.radians(cand["angle_deg"]) / max(omega, 1e-6) / DT)))
            w_body = np.asarray(cand["axis"], float) * omega
            ctl = ForcePController(kp=0.12, da_max=0.12)
            ctl.reset(1.0, s0["fn_tot"])
            lost = 0
            ok = True
            z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
            for _ in range(n):
                st = sample_state(env.sim)
                a_g, _ = ctl.step(st["fn_tot"], fn_target)
                v_h = _height_hold_v_hand(env.sim, z_ref)
                step_legal(env.sim, make_action(v_h, w_body, a_g))
                failed, lost = recovery_failed(env.sim, blob["ref"], lost)
                st2 = sample_state(env.sim)
                if failed or latched(env.sim) or not st2["bilateral"]:
                    ok = False
                    break
            sf = sample_state(env.sim)
            g = np.asarray(sf["g_h"], float)
            lat = float(np.linalg.norm(g[:2]))
            dp = (np.asarray(sf["p_c_h"], float) - p0) * 1e3
            z_end = float(env.sim.data.xpos[env.sim.ids.object_body][2])
            clears = ok and lat <= g_lateral_max and g[2] > 7.0 and sf["bilateral"] and z_end >= 0.44
            score = (
                (100.0 if clears else 0.0)
                + 3.0 * float(g[2])
                - 2.5 * lat
                - 1.2 * float(np.linalg.norm(dp))
                - 0.1 * cand["angle_deg"]
                + 20.0 * min(z_end, 0.55)
                - (0.0 if ok else 40.0)
            )
            evaluated.append(
                {
                    "axis_name": cand["axis_name"],
                    "axis": cand["axis"],
                    "angle_deg": cand["angle_deg"],
                    "rate_scale": rate,
                    "n_steps": n,
                    "omega_body": w_body.tolist(),
                    "ok": ok,
                    "clears": clears,
                    "g_h": g.tolist(),
                    "lateral": lat,
                    "delta_p_rel_mm": dp.tolist(),
                    "theta": sf["e_theta_deg"],
                    "e_signed_mm": sf["e_signed_mm"],
                    "score": score,
                }
            )
    evaluated.sort(key=lambda r: r["score"], reverse=True)
    clearing = [r for r in evaluated if r["clears"]]
    best = (sorted(clearing, key=lambda r: (r["angle_deg"], r["lateral"]))[0] if clearing else evaluated[0])
    return {
        "ok": bool(best["clears"]),
        "already_upright": False,
        "axis_name": best["axis_name"],
        "omega_body": best["omega_body"],
        "angle_deg": best["angle_deg"],
        "n_steps": best["n_steps"],
        "rate_scale": best["rate_scale"],
        "fn_target": fn_target,
        "g_h0": g0.tolist(),
        "g_h": best["g_h"],
        "delta_p_rel_mm": best["delta_p_rel_mm"],
        "e_signed_mm": best["e_signed_mm"],
        "top": evaluated[:5],
        "rationale": f"restore via {best['axis_name']} @ {best['angle_deg']}° (lateral={best.get('lateral', float('nan')):.2f})",
        "d": d_up.tolist(),
    }


def execute_restore_plan(env, blob: dict, plan: dict) -> dict[str, Any]:
    """Execute restore plan (same machinery as tilt execution)."""
    # Reuse execute_orientation_plan structure
    plan_exec = {
        **plan,
        "d": plan.get("d") or [0.0, 0.0, 1.0],
        "already_sufficient": plan.get("already_upright", False),
    }
    return execute_orientation_plan(env, blob, plan_exec, adaptive=True, drift_slow_mm=2.0)
