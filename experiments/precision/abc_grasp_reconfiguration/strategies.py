"""Experimental A/B/C reconfiguration from a stable Stage-1 exit (not production)."""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3, TAU_OPEN, TAU_SECURE, normalize_action
from data_generation.oracles.plan_b.release_recatch import _action_from
from data_generation.oracles.plan_b.vertical_recatch import rollout as b_rollout
from data_generation.oracles.plan_c.dynamic_catch import jaw_state
from data_generation.oracles.plan_c.future_trajectory import grasp_point, pad_bilateral, sample_kin
from data_generation.oracles.plan_c.interception import FULL_DETACH_ACTIVE_RECATCH
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.representability import representability
from envs.outcomes.ground_drop import latched as ground_latched
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.table_drop import latched as table_latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis

DT = 0.02
PINCH_Z = 0.0


def err_pack(sim) -> dict:
    e = measure_errors(sim)
    return {
        "dx": float(e["dx_mm"]),
        "dy": float(e["dy_mm"]),
        "dz": float(e["dz_mm"]),
        "dtheta": float(e["dtheta_deg"]),
        "e_pos": float(e["e_pos_mm"]),
        "bilateral": bool(e.get("bilateral")),
        "fn_tot": float(e.get("fn_tot") or 0),
        "alpha_up_deg": float(e.get("alpha_up_deg") or 0),
        "v_rel_norm": float(e.get("v_rel_norm") or 0),
    }


def verify_12s(env, ref) -> dict:
    """Authoritative 12 s: apply_handoff + nominal, then restore capture."""
    from envs.outcomes.horizon_outcome import NOMINAL_S, nominal_verify

    cap = take_capture(env)
    restore_capture(env, cap)
    try:
        out = nominal_verify(env, ref, seconds=NOMINAL_S)
        ok12 = bool(out.get("success"))
        reason = out.get("reason")
    except Exception as exc:
        ok12 = False
        reason = f"verify_exception:{exc}"
        out = {}
    # Also probe SUCCESS_NEW semantics without consuming state twice
    restore_capture(env, cap)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    restore_capture(env, cap)
    return {
        "SUCCESS_12S": ok12 or bool(hand.get("SUCCESS_12S")),
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "nominal_reason": reason,
        "nominal_success": bool(out.get("success")) if out else False,
        "handoff": {
            k: hand.get(k)
            for k in ("SUCCESS_12S", "SUCCESS_NEW", "e_pos_mm", "e_theta_deg", "hold_ok")
        },
    }


def reconfig_improved(e0: dict, e1: dict, *, t_dx: float, t_dz: float) -> dict:
    ax0, az0 = abs(float(e0["dx"])), abs(float(e0["dz"]))
    ax1, az1 = abs(float(e1["dx"])), abs(float(e1["dz"]))
    dth0, dth1 = abs(float(e0["dtheta"])), abs(float(e1["dtheta"]))
    dx_ok = ax1 <= t_dx
    dz_ok = az1 <= t_dz
    other_blown = False
    if dx_ok and not dz_ok:
        other_blown = az1 > az0 + 8.0 or dth1 > dth0 + 15.0
    if dz_ok and not dx_ok:
        other_blown = ax1 > ax0 + 8.0 or dth1 > dth0 + 15.0
    if dx_ok and dz_ok:
        other_blown = dth1 > max(25.0, dth0 + 15.0)
    useful = (dx_ok or dz_ok) and (not other_blown) and bool(e1.get("bilateral"))
    return {
        "dx_in_band": dx_ok,
        "dz_in_band": dz_ok,
        "other_deteriorated": other_blown,
        "RECONFIG_IMPROVED": useful,
        "delta_abs_dx": ax1 - ax0,
        "delta_abs_dz": az1 - az0,
        "delta_abs_dtheta": dth1 - dth0,
        "delta_e_pos": float(e1["e_pos"]) - float(e0["e_pos"]),
    }


def _quantize(raw, book):
    packed = representability(np.asarray(raw, np.float32).reshape(1, 7), book)
    return np.asarray(packed["quantized"][0], np.float32)


# ---------------------------------------------------------------------------
# A — inward sliding
# ---------------------------------------------------------------------------


def run_strategy_A(env, blob: dict, book) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    e0 = err_pack(env.sim)
    actions: list = []
    timeline: list = []
    lost = 0
    t0 = float(env.sim.data.time)
    ex = float(e0["dx"]) * 1e-3
    wy_sign = -1.0 if ex >= 0 else 1.0
    phases = [
        ("reorient", 0.35, 0.9, 1.6 * wy_sign, 1.0),
        ("relax", 0.55, 0.0, 0.0, -0.48),
        ("retighten", 0.25, 0.0, 0.0, 1.0),
        ("hold", 0.30, 0.0, 0.0, 1.0),
    ]
    reason = "completed"
    genuine_slide = False
    disp_mm = 0.0
    p0 = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float).copy()

    for tag, dur, wx, wy, grip in phases:
        for i in range(int(dur / DT)):
            raw = normalize_action(np.zeros(3), np.array([wx, wy, 0.0], float), grip, AIRBORNE_V3)
            a = _quantize(raw, book)
            step_legal(env.sim, a)
            actions.append(a.copy())
            failed, lost = recovery_failed(env.sim, blob["ref"], lost)
            e = err_pack(env.sim)
            if i % 2 == 0:
                timeline.append({"t": float(env.sim.data.time) - t0, "phase": f"A_{tag}", **e})
            if failed or table_latched(env.sim) or ground_latched(env.sim) or not e["bilateral"]:
                reason = "contact_loss" if not e["bilateral"] else "drop"
                break
        p1 = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        step_disp = float(np.linalg.norm(p1 - p0)) * 1e3
        if tag == "relax" and step_disp > 1.0:
            genuine_slide = True
        disp_mm += step_disp
        p0 = p1
        if reason != "completed":
            break

    e1 = err_pack(env.sim)
    return {
        "strategy": "A",
        "ok_maneuver": reason == "completed" and bool(e1.get("bilateral")),
        "reason": reason,
        "genuine_slide": genuine_slide,
        "disp_mm": disp_mm,
        "released": False,
        "reacquired": False,
        "actions": actions,
        "timeline": timeline,
        "errors0": e0,
        "errors1": e1,
        "capture": take_capture(env),
        "n_actions": len(actions),
    }


# ---------------------------------------------------------------------------
# B — deliberate recatch
# ---------------------------------------------------------------------------


def run_strategy_B(env, blob: dict, book) -> dict[str, Any]:
    restore_capture(env, blob["capture"])
    e0 = err_pack(env.sim)
    spec = {
        "open_min": 0.08,
        "chase_max": 1.20,
        "kp": 4.0,
        "kz": 3.0,
        "z": 0.02,
        "down": 0.8,
        "axial": 0.012,
        "x_tol": 0.028,
        "y_tol": 0.028,
        "z_tol": 0.040,
        "gap": 0.030,
        "pad_steps": 3,
        "brake": 2.5,
        "lift": 0.35,
        "rise": 0.04,
    }
    result = b_rollout(env, blob, book, spec)
    end_cap = take_capture(env)
    restore_capture(env, end_cap)
    e1 = err_pack(env.sim)
    rows = result.get("rows") or []
    released = False
    reacquired = False
    t_off = t_re = None
    seen_off = False
    for r in rows:
        if int(r.get("nL", 1)) == 0 and int(r.get("nR", 1)) == 0:
            seen_off = True
            released = True
            if t_off is None:
                t_off = float(r.get("t") or 0)
        elif seen_off and int(r.get("nL", 0)) > 0 and int(r.get("nR", 0)) > 0:
            reacquired = True
            if t_re is None:
                t_re = float(r.get("t") or 0)
            break
    mode = result.get("mode")
    genuine = bool(released and reacquired and mode in ("handoff", "settle", "lift", "brake"))
    acts = result.get("actions")
    act_list = list(acts) if acts is not None and len(acts) else []
    return {
        "strategy": "B",
        "ok_maneuver": genuine and bool(e1.get("bilateral")),
        "reason": mode,
        "released": released,
        "reacquired": reacquired,
        "genuine_recatch": genuine,
        "t_release": t_off,
        "t_reacquire": t_re,
        "contact_free_s": None if t_off is None or t_re is None else float(t_re - t_off),
        "actions": act_list,
        "timeline": [
            {"t": float(r.get("t") or 0), "phase": f"B_{r.get('mode')}", "nL": r.get("nL"), "nR": r.get("nR")}
            for r in rows[::2]
        ],
        "errors0": e0,
        "errors1": e1,
        "capture": end_cap,
        "n_actions": len(act_list),
        "rollout_mode": mode,
    }


# ---------------------------------------------------------------------------
# C — controlled-release large recatch (experimental; no ball impact)
# ---------------------------------------------------------------------------


def _preview_free_motion(env, blob, open_steps: int = 5, horizon_s: float = 0.40) -> dict:
    """Open then coast (legal); restore capture afterward. Planning only."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    traj = []
    for _ in range(open_steps):
        a = _action_from(env.sim, np.zeros(3), np.zeros(3), TAU_OPEN)
        a[6] = -1.0
        step_legal(env.sim, a)
    for _ in range(int(horizon_s / DT)):
        k = sample_kin(env.sim)
        pb = pad_bilateral(env.sim)
        traj.append(
            {
                "t": float(env.sim.data.time) - t0,
                "p_obj": np.asarray(k["p_obj"], float).tolist(),
                "v_obj": np.asarray(k["v_obj"], float).tolist(),
                "nL": int(pb.get("nL") or 0),
                "nR": int(pb.get("nR") or 0),
            }
        )
        a = _action_from(env.sim, np.zeros(3), np.zeros(3), TAU_OPEN)
        a[6] = -1.0
        step_legal(env.sim, a)
        if ground_latched(env.sim) or table_latched(env.sim):
            break
    restore_capture(env, blob["capture"])
    t_c = 0.14
    for row in traj:
        if row["nL"] == 0 and row["nR"] == 0 and float(row["t"]) >= 0.08:
            t_c = float(row["t"])
            break
    # Prefer axial offset that counters lateral drift in free motion (shrink DX)
    s_axial = 0.0
    if len(traj) >= 5:
        p0 = np.asarray(traj[0]["p_obj"], float)
        p1 = np.asarray(traj[min(len(traj) - 1, 10)]["p_obj"], float)
        s_axial = float(np.clip(-(p1[0] - p0[0]), -0.02, 0.02))
    return {"traj": traj, "t_c": t_c, "s_axial": s_axial}


def run_strategy_C(env, blob: dict, book) -> dict[str, Any]:
    """OPEN → intercept → close → brake from a stable grasp (MuJoCo physics only)."""
    restore_capture(env, blob["capture"])
    e0 = err_pack(env.sim)
    plan = _preview_free_motion(env, blob)
    restore_capture(env, blob["capture"])

    open_steps = 5
    t_c = float(plan["t_c"])
    s = float(plan["s_axial"])
    match = 1.05
    approach = 4.5
    force_close_s = 0.34
    brake_delay = 0.04
    brake_dur = 0.28
    horizon_s = 0.85

    actions: list = []
    timeline: list = []
    t0 = float(env.sim.data.time)
    opened = False
    closing = False
    first_bil_t = None
    saw_detach = False
    max_free = 0.0
    free_run = 0.0
    t_release = t_reacq = None
    release_pose = None

    k0 = sample_kin(env.sim)
    p_tgt = grasp_point(k0["p_obj"], k0["u_obj"], s)
    v_tgt = np.asarray(k0["v_obj"], float).copy()

    for i in range(int(horizon_s / DT)):
        t_rel = float(env.sim.data.time) - t0
        k = sample_kin(env.sim)
        js = jaw_state(env.sim)
        pb = pad_bilateral(env.sim)

        if t_rel >= t_c - 0.02:
            p_tgt = grasp_point(k["p_obj"], k["u_obj"], s)
            v_tgt = np.asarray(k["v_obj"], float).copy()

        if i < open_steps:
            tau, grip_a = TAU_OPEN, -1.0
            opened = True
        else:
            opened = True
            opening = float(js.get("opening_m") or 0)
            if (not closing) and (
                (opening >= 0.026 and bool(js.get("inside_jaw")))
                or t_rel >= force_close_s
            ):
                closing = True
            if closing:
                tau, grip_a = TAU_SECURE, 1.0
            else:
                tau, grip_a = TAU_OPEN, -1.0

        if pb.get("bilateral") and first_bil_t is None and saw_detach:
            first_bil_t = t_rel
            t_reacq = t_rel
        if pb.get("bilateral") and first_bil_t is None and opened and i >= open_steps:
            # reacquire after open even if detach was brief
            if saw_detach:
                first_bil_t = t_rel
                t_reacq = t_rel

        ph, Rh = k["p_hand"], k["R_hand"]
        pinch_w = ph + Rh @ np.array([0.0, 0.0, PINCH_Z], float)
        err = p_tgt - pinch_w
        remain = max(t_c - t_rel, 0.02)

        brake_active = first_bil_t is not None and t_rel >= first_bil_t + brake_delay
        if brake_active and pb.get("bilateral"):
            v_cmd = 0.5 * match * v_tgt + 0.3 * err
            if t_rel >= first_bil_t + brake_delay + brake_dur:
                v_cmd = 0.1 * v_tgt + 0.2 * err
        else:
            v_cmd = approach * (err / remain) + match * v_tgt

        a = _action_from(env.sim, v_cmd, np.zeros(3), tau)
        a[6] = grip_a
        a = _quantize(a, book)
        step_legal(env.sim, a)
        actions.append(a.copy())

        pb2 = pad_bilateral(env.sim)
        nL, nR = int(pb2.get("nL") or 0), int(pb2.get("nR") or 0)
        if nL == 0 and nR == 0:
            if not saw_detach:
                t_release = t_rel
                release_pose = {
                    "t": t_rel,
                    "p_obj": np.asarray(k["p_obj"], float).tolist(),
                    "v_obj": np.asarray(k["v_obj"], float).tolist(),
                }
            saw_detach = True
            free_run += DT
            max_free = max(max_free, free_run)
        else:
            free_run = 0.0
            if saw_detach and nL > 0 and nR > 0 and t_reacq is None:
                t_reacq = t_rel
                first_bil_t = first_bil_t or t_rel

        if i % 2 == 0:
            timeline.append(
                {
                    "t": t_rel,
                    "phase": "C_open" if not closing else ("C_brake" if brake_active else "C_close"),
                    "nL": nL,
                    "nR": nR,
                    "bilateral": bool(pb2.get("bilateral")),
                    **err_pack(env.sim),
                }
            )
        if ground_latched(env.sim) or table_latched(env.sim):
            break
        if first_bil_t is not None and t_rel >= first_bil_t + brake_delay + brake_dur + 0.25:
            if pb2.get("bilateral"):
                break

    # Short secure hold (avoid aggressive world-lift — it dropped several recatches)
    if pad_bilateral(env.sim).get("bilateral"):
        for _ in range(20):
            if not pad_bilateral(env.sim).get("bilateral") or ground_latched(env.sim):
                break
            a = _action_from(env.sim, np.zeros(3), np.zeros(3), TAU_SECURE)
            a[6] = 1.0
            a = _quantize(a, book)
            step_legal(env.sim, a)
            actions.append(a.copy())

    e1 = err_pack(env.sim)
    genuine = bool(saw_detach and max_free >= DT - 1e-9 and e1.get("bilateral") and t_reacq is not None)
    mech = FULL_DETACH_ACTIVE_RECATCH if genuine else ("FAILED_OR_NO_DETACH" if not saw_detach else "DETACH_NO_STABLE_REACQUIRE")

    return {
        "strategy": "C",
        "ok_maneuver": genuine,
        "reason": mech,
        "released": saw_detach,
        "reacquired": t_reacq is not None and bool(e1.get("bilateral")),
        "genuine_controlled_release": genuine,
        "t_release": t_release,
        "t_reacquire": t_reacq,
        "contact_free_s": max_free,
        "release_state": release_pose,
        "plan_t_c": t_c,
        "plan_s_axial": s,
        "free_motion_preview": plan["traj"][::2],
        "actions": actions,
        "timeline": timeline,
        "errors0": e0,
        "errors1": e1,
        "capture": take_capture(env),
        "n_actions": len(actions),
        "mechanism": mech,
    }
