"""Future-informed dynamic airborne catch (GROUND_SCENE).

Sequence: OPEN → INTERCEPT → VELOCITY MATCH → CLOSE WHILE FOLLOWING → BRAKE.

Privileged future inspection allowed for offline search only.
Final experts are uninterrupted legal 7D tapes. No physics / SUCCESS_NEW / K49 edits.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import mujoco
import numpy as np

from controllers.recovery.action_map import (
    AIRBORNE_V3,
    TAU_OPEN,
    TAU_SECURE,
    V_AIRBORNE,
    V_Z_DOWN,
    V_Z_UP,
    W_MAX,
    map_action,
)
from data_generation.oracles.plan_c.interception import (
    FULL_DETACH_ACTIVE_RECATCH,
    PARTIAL_SLIP_RECOVERY,
    POST_IMPACT_GRASP_PRESERVATION,
    finger_opening_m,
    finger_opening_vel,
    geometry_pack,
)
from data_generation.oracles.plan_c.future_trajectory import (
    PINCH_Z,
    empty_bilateral_stats,
    grasp_point,
    grasp_point_vel,
    hold_action,
    pad_bilateral,
    pose_correct_and_handoff,
    record_no_intervention_future,
    sample_kin,
    topology,
    update_bilateral_stats,
)
from data_generation.oracles.plan_b.release_recatch import _action_from
from envs.contacts.frames import extract_contacts
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.simulation.ground_scene import CYL_HALF, CYL_R
from envs.step_legal import step_legal
from evaluation.success_criteria import contact_summary
from training.recovery_execution import restore_ballistic

DT_POL = 0.02


@dataclass
class CatchParams:
    """Joint search variables for the full catch sequence (no fixed 80 ms open)."""

    open_steps: int = 4  # max legal open at start
    match: float = 1.0  # grasp-point velocity match
    approach_gain: float = 4.0  # position-to-intercept gain
    close_clearance_m: float = 0.002  # close when jaw_clearance > this
    close_lat_max: float = 0.022
    close_axial_max: float = 0.045
    close_vrel_max: float = 0.55
    min_open_before_close: float = 0.028
    force_close_after_s: float = 0.28  # fallback if geometry never triggers
    brake_onset_delay_s: float = 0.04  # after first bilateral
    brake: float = 0.55  # blend toward slow follow after bilateral
    brake_duration_s: float = 0.25
    wrist_gain: float = 4.0
    s_axial: float = 0.0  # grasp point offset along cylinder axis
    t_c: float = 0.12  # predicted interception time (rel)


def jaw_state(sim) -> dict[str, Any]:
    """Geometry for jaw-entry / close-onset decisions (measured, not commanded)."""
    g = geometry_pack(sim)
    k = sample_kin(sim)
    Rh, ph, po = k["R_hand"], k["p_hand"], k["p_obj"]
    prel = Rh.T @ (po - ph)
    pinch = np.array([0.0, 0.0, PINCH_Z], float)
    err_pinch = prel - pinch
    opening = float(g["opening_m"])
    need = float(g["need_clear"])
    # Cylinder inside usable jaw volume (diagnostic).
    inside = (
        opening + 1e-6 >= need * 0.85
        and abs(float(prel[1])) < 0.028
        and abs(float(prel[2] - PINCH_Z)) < 0.050
        and abs(float(prel[0])) < 0.035
    )
    return {
        **{kk: g[kk] for kk in (
            "opening_m", "opening_vel", "jaw_clearance", "need_clear",
            "axial_from_pinch", "lateral_offset", "v_rel_norm", "bilateral",
            "any_pad", "housing", "obj_z",
        )},
        "prel": prel.copy(),
        "err_pinch": err_pinch.copy(),
        "inside_jaw": bool(inside),
        "topo": topology(sim),
        "cs_nL": int(contact_summary(sim)["nL"]),
        "cs_nR": int(contact_summary(sim)["nR"]),
    }


def measure_actuator_response(sim, snap, *, max_s: float = 0.40) -> dict[str, Any]:
    """Controlled legal-action tests from a Recovery snapshot. Measured, not assumed."""
    out: dict[str, Any] = {
        "tau_open": TAU_OPEN,
        "tau_secure": TAU_SECURE,
        "v_airborne_cmd": V_AIRBORNE,
        "v_z_down_cmd": V_Z_DOWN,
        "v_z_up_cmd": V_Z_UP,
        "w_max_cmd": W_MAX,
        "dt_policy": DT_POL,
    }

    def _series(cmd_builder, label: str, horizon: float) -> dict:
        restore_ballistic(sim, snap)
        restore_drop(sim, None)
        arm(sim)
        t0 = float(sim.data.time)
        open0 = finger_opening_m(sim)
        k0 = sample_kin(sim)
        rows = []
        while float(sim.data.time) - t0 < horizon - 1e-9:
            a = cmd_builder(sim)
            step_legal(sim, a)
            k = sample_kin(sim)
            tr = float(sim.data.time) - t0
            rows.append({
                "t_rel": tr,
                "opening_m": finger_opening_m(sim),
                "opening_vel": finger_opening_vel(sim),
                "v_hand_norm": float(np.linalg.norm(k["v_hand"])),
                "v_hand": k["v_hand"].tolist(),
                "w_hand_norm": float(np.linalg.norm(k["w_hand"])),
                "obj_contact": int(pad_bilateral(sim)["any_pad"] or contact_summary(sim)["nL"] + contact_summary(sim)["nR"] > 0),
                "ground_drop": int(latched(sim)),
            })
            if latched(sim):
                break
        opens = [r["opening_m"] for r in rows]
        t_hit_03 = next((r["t_rel"] for r in rows if r["opening_m"] >= 0.030), None)
        t_hit_04 = next((r["t_rel"] for r in rows if r["opening_m"] >= 0.040), None)
        # Finite-diff opening speed from first 100 ms (or available)
        early = [r for r in rows if r["t_rel"] <= 0.10 + 1e-9]
        open_rate = None
        if len(early) >= 2:
            open_rate = (early[-1]["opening_m"] - early[0]["opening_m"]) / max(
                early[-1]["t_rel"] - early[0]["t_rel"], 1e-6
            )
        v_norms = [r["v_hand_norm"] for r in rows]
        return {
            "label": label,
            "open0": open0,
            "open_end": opens[-1] if opens else open0,
            "t_to_0p03": t_hit_03,
            "t_to_0p04": t_hit_04,
            "open_rate_m_s": open_rate,
            "max_v_hand": max(v_norms) if v_norms else 0.0,
            "mean_v_hand": float(np.mean(v_norms)) if v_norms else 0.0,
            "n_steps": len(rows),
            "ground_drop": int(latched(sim)),
            "had_object_contact": int(any(r["obj_contact"] for r in rows)),
            "v0_hand": float(np.linalg.norm(k0["v_hand"])),
            "series": rows,
        }

    # Max legal open, hand hold
    def open_hold(sim):
        a = hold_action(-1.0)
        a[6] = -1.0
        return a

    # Max legal close, hand hold
    def close_hold(sim):
        return hold_action(1.0)

    # Max legal downward chase (hand +z / world -z on horizontal grasp)
    def max_down(sim):
        a = np.zeros(7, np.float32)
        a[2] = 1.0  # maps to +3.8 m/s hand-z (world down)
        a[6] = -1.0
        return a

    # Max lateral (hand y)
    def max_lat(sim):
        a = np.zeros(7, np.float32)
        a[1] = 1.0
        a[6] = -1.0
        return a

    # Max wrist pitch
    def max_wrist(sim):
        a = np.zeros(7, np.float32)
        a[4] = 1.0
        a[6] = -1.0
        return a

    out["open_hold"] = _series(open_hold, "open_hold", max_s)
    out["close_hold"] = _series(close_hold, "close_hold", min(0.25, max_s))
    out["max_down"] = _series(max_down, "max_down", min(0.20, max_s))
    out["max_lat"] = _series(max_lat, "max_lat", min(0.20, max_s))
    out["max_wrist"] = _series(max_wrist, "max_wrist", min(0.16, max_s))

    # Opening under contact (if still touching)
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    pb0 = pad_bilateral(sim)
    out["start_bilateral"] = int(pb0["bilateral"])
    out["start_any_pad"] = int(pb0["any_pad"])
    out["open_under_contact"] = _series(open_hold, "open_under_contact", min(0.30, max_s))
    return out


def propose_catch_interceptions(future: dict, *, n_times: int = 8, n_axial: int = 5) -> list[dict]:
    """Future-informed (t_c, s) candidates aiming for jaw entry, not COM proximity alone."""
    rows = future.get("rows") or []
    if not rows:
        return []
    t_free = future.get("first_contact_free_t")
    t_ground = future.get("t_ground_drop")
    usable = []
    for r in rows:
        if t_ground is not None and r["t_rel"] + 0.04 >= float(t_ground):
            continue
        if r["obj_z"] < 0.05:
            continue
        if t_free is not None and r["t_rel"] + 1e-12 < float(t_free) - 0.02:
            # allow slightly before full free-flight for gradual slip
            pass
        usable.append(r)
    if not usable:
        usable = rows
    idxs = np.linspace(0, len(usable) - 1, num=min(n_times, len(usable)))
    idxs = sorted({int(round(i)) for i in idxs})
    s_vals = np.linspace(-(CYL_HALF - 0.004), (CYL_HALF - 0.004), n_axial)
    cands = []
    for i in idxs:
        r = usable[i]
        for s in s_vals:
            p = grasp_point(r["p_obj"], r["u_obj"], float(s))
            v = grasp_point_vel(r["v_obj"], r["w_obj"], r["u_obj"], float(s))
            cands.append({
                "t_c": float(r["t_rel"]),
                "s": float(s),
                "p_grasp": p.copy(),
                "v_grasp": v.copy(),
                "u": np.asarray(r["u_obj"], float).copy(),
                "obj_z": float(r["obj_z"]),
                "opening_need": float(max(2.2 * CYL_R, 0.040)),
                "v_grasp_norm": float(np.linalg.norm(v)),
            })
    return cands


def _should_close(js: dict, params: CatchParams, t_rel: float, opened: bool) -> bool:
    if not opened:
        return False
    if t_rel + 1e-12 >= float(params.force_close_after_s):
        return True
    if float(js["opening_m"]) < float(params.min_open_before_close):
        return False
    if float(js["jaw_clearance"]) < float(params.close_clearance_m):
        # still allow if already deeply inside with residual clearance near zero
        if not js["inside_jaw"]:
            return False
    if abs(float(js["lateral_offset"])) > float(params.close_lat_max):
        return False
    if abs(float(js["axial_from_pinch"])) > float(params.close_axial_max):
        return False
    if float(js["v_rel_norm"]) > float(params.close_vrel_max) and not js["inside_jaw"]:
        return False
    return bool(js["inside_jaw"] or float(js["jaw_clearance"]) >= float(params.close_clearance_m))


def dynamic_catch_sequence(
    sim,
    snap,
    ref,
    params: CatchParams,
    cand: dict | None = None,
    *,
    horizon_s: float = 0.70,
) -> dict:
    """One uninterrupted legal rollout: OPEN→INTERCEPT→MATCH→CLOSE→BRAKE."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    g0 = geometry_pack(sim)
    start_bil = bool(g0["bilateral"] or (g0["cs_nL"] > 0 and g0["cs_nR"] > 0))
    start_any = bool(g0["any_pad"] or g0["cs_nL"] + g0["cs_nR"] > 0)
    saw_full_detach = not start_any
    max_contact_free_s = 0.0
    contact_free_run = 0.0

    t_c = float(params.t_c if cand is None else cand.get("t_c", params.t_c))
    s = float(params.s_axial if cand is None else cand.get("s", params.s_axial))
    if cand is not None:
        p_tgt = np.asarray(cand["p_grasp"], float).copy()
        v_tgt = np.asarray(cand["v_grasp"], float).copy()
        u = np.asarray(cand["u"], float).copy()
    else:
        k = sample_kin(sim)
        p_tgt = grasp_point(k["p_obj"], k["u_obj"], s)
        v_tgt = grasp_point_vel(k["v_obj"], k["w_obj"], k["u_obj"], s)
        u = k["u_obj"].copy()

    actions: list[np.ndarray] = []
    bil = empty_bilateral_stats()
    t0 = float(sim.data.time)
    n_steps = int(round(horizon_s / DT_POL))
    opened = False
    closing = False
    first_bil_t = None
    first_pad_t = None
    jaw_entered = False
    open_at_intercept = None
    vrel_at_first_contact = None
    inside_at_close = False
    brake_active = False

    for i in range(n_steps):
        t_rel = float(sim.data.time) - t0
        k = sample_kin(sim)
        js = jaw_state(sim)
        if js["inside_jaw"]:
            jaw_entered = True
        # Live update of grasp target after intercept time (follow cylinder)
        if t_rel + 1e-12 >= t_c - 0.02:
            p_tgt = grasp_point(k["p_obj"], k["u_obj"], s)
            v_tgt = grasp_point_vel(k["v_obj"], k["w_obj"], k["u_obj"], s)
            u = k["u_obj"].copy()
        if open_at_intercept is None and t_rel + 1e-12 >= t_c:
            open_at_intercept = float(js["opening_m"])

        ph, Rh = k["p_hand"], k["R_hand"]
        pinch_w = ph + Rh @ np.array([0.0, 0.0, PINCH_Z], float)
        err = p_tgt - pinch_w
        remain = max(t_c - t_rel, 0.02)

        # OPEN phase
        if i < int(params.open_steps):
            tau = TAU_OPEN
            grip_a = -1.0
            opened = True
        else:
            opened = True
            if not closing and _should_close(js, params, t_rel, opened):
                closing = True
                inside_at_close = bool(js["inside_jaw"])
            if closing:
                tau = TAU_SECURE
                grip_a = 1.0
            else:
                tau = TAU_OPEN
                grip_a = -1.0

        # Velocity: intercept approach + grasp-point match; after bil, synchronized brake
        pb_now = pad_bilateral(sim)
        if pb_now["bilateral"] and first_bil_t is None:
            first_bil_t = t_rel
        if first_bil_t is not None and t_rel >= first_bil_t + float(params.brake_onset_delay_s):
            brake_active = True

        if brake_active and pb_now["bilateral"]:
            # Follow slowly then blend toward object velocity fraction
            v_follow = float(params.match) * (1.0 - float(params.brake)) * v_tgt
            v_cmd = v_follow + (1.0 - float(params.brake)) * (0.5 * err / max(remain, 0.05))
            # After brake_duration, deepen brake
            if t_rel >= first_bil_t + float(params.brake_onset_delay_s) + float(params.brake_duration_s):
                v_cmd = 0.08 * v_tgt + 0.15 * err
        else:
            v_cmd = float(params.approach_gain) * (err / remain) + float(params.match) * v_tgt

        # Wrist: align hand -z toward -u (cylinder axis into jaw)
        z_h = Rh[:, 2]
        axis = np.cross(z_h, -u)
        ang = float(np.linalg.norm(axis))
        if ang > 1e-8:
            axis = axis / ang
            w_world = axis * min(ang * float(params.wrist_gain), 2.5)
            w_body = Rh.T @ w_world
        else:
            w_body = np.zeros(3)
        # Match object spin lightly while open/closing
        if not brake_active:
            w_body = w_body + 0.25 * (Rh.T @ k["w_obj"])

        a = _action_from(sim, v_cmd, w_body, tau)
        a[6] = grip_a
        actions.append(a.copy())
        step_legal(sim, a)

        pb = pad_bilateral(sim)
        cs = contact_summary(sim)
        if first_pad_t is None and pb["any_pad"]:
            first_pad_t = float(sim.data.time) - t0
            vrel_at_first_contact = float(np.linalg.norm(sample_kin(sim)["v_obj"] - sample_kin(sim)["v_hand"]))
        if not pb["any_pad"] and not (cs["nL"] + cs["nR"] > 0):
            saw_full_detach = True
            contact_free_run += DT_POL
            max_contact_free_s = max(max_contact_free_s, contact_free_run)
        else:
            contact_free_run = 0.0
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        if latched(sim):
            break

    # Short secure hold if bilateral
    if pad_bilateral(sim)["bilateral"]:
        for _ in range(10):
            actions.append(hold_action(1.0))
            step_legal(sim, hold_action(1.0))
            pb = pad_bilateral(sim)
            update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
            if latched(sim) or not pb["bilateral"]:
                break

    end_bil = bool(pad_bilateral(sim)["bilateral"])
    c60 = bool(bil.get("continuous_ge_60ms_at_end"))
    c150 = bool(bil.get("continuous_ge_150ms_at_end"))
    max_consec = float(bil.get("max_consecutive_bil_s") or 0)
    airborne = (not latched(sim)) and float(sample_kin(sim)["obj_z"]) > 0.05

    # Full-detach requires a measured contact-free gap (>=1 legal step) before reacquisition.
    genuine_detach_gap = bool(saw_full_detach and max_contact_free_s + 1e-12 >= DT_POL)
    if start_bil and end_bil and not genuine_detach_gap and (c60 or max_consec >= 0.06):
        mech = POST_IMPACT_GRASP_PRESERVATION
    elif start_any and end_bil and not genuine_detach_gap and (c60 or end_bil):
        mech = PARTIAL_SLIP_RECOVERY
    elif genuine_detach_gap and end_bil and c60:
        mech = FULL_DETACH_ACTIVE_RECATCH
    else:
        mech = "FAILED_OR_TRANSIENT"

    n_capture = len(actions)
    success_meta = {"SUCCESS_NEW": False, "reason": "no_stable_capture"}
    if end_bil and c150 and not latched(sim):
        for _ in range(8):
            actions.append(hold_action(1.0))
            step_legal(sim, hold_action(1.0))
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                break
        n_capture = len(actions)
        if pad_bilateral(sim)["bilateral"] and not latched(sim):
            pose_acts, success_meta = pose_correct_and_handoff(sim, ref, max_s=3.5)
            actions = actions + pose_acts
        else:
            success_meta = {"SUCCESS_NEW": False, "reason": "lost_before_pose_correct"}

    js_end = jaw_state(sim)
    return {
        "actions": actions,
        "capture_actions": actions[:n_capture],
        "mechanism": mech,
        "SUCCESS_NEW": bool(success_meta.get("SUCCESS_NEW", False)),
        "reason": success_meta.get("reason"),
        "continuous_60": c60,
        "continuous_150": c150,
        "max_consec_bil": max_consec,
        "end_bilateral": end_bil,
        "saw_full_detach": saw_full_detach,
        "max_contact_free_s": float(max_contact_free_s),
        "genuine_detach_gap": int(genuine_detach_gap),
        "start_bilateral": start_bil,
        "start_any_contact": start_any,
        "ground_drop": int(latched(sim)),
        "airborne_end": int(airborne),
        "jaw_entered": int(jaw_entered),
        "inside_at_close": int(inside_at_close),
        "closing_triggered": int(closing),
        "open_at_intercept": open_at_intercept,
        "open_end": float(js_end["opening_m"]),
        "vrel_at_first_contact": vrel_at_first_contact,
        "first_pad_t": first_pad_t,
        "first_bil_t": first_bil_t,
        "brake_used": int(brake_active),
        "t_c": t_c,
        "s": s,
        "params": asdict(params),
        "g0_opening": float(g0["opening_m"]),
        "g0_axial": float(g0["axial_from_pinch"]),
        "g0_vrel": float(g0["v_rel_norm"]),
    }


def catch_param_grid(rng: np.random.Generator, cands: list[dict]) -> list[tuple[CatchParams, dict]]:
    """Joint (params, interception candidate) grid; shuffled."""
    pairs: list[tuple[CatchParams, dict]] = []
    open_opts = (2, 3, 4, 6, 8, 10)
    match_opts = (0.7, 0.9, 1.05, 1.2)
    gain_opts = (2.5, 4.0, 6.0, 9.0)
    brake_opts = (0.3, 0.55, 0.75)
    close_v_opts = (0.35, 0.55, 0.80)
    force_close_opts = (0.16, 0.22, 0.30, 0.40)
    if not cands:
        cands = [{"t_c": 0.12, "s": 0.0, "p_grasp": np.zeros(3), "v_grasp": np.zeros(3), "u": np.array([1.0, 0, 0])}]
    # Prefer mid-horizon intercepts
    cands_sorted = sorted(cands, key=lambda c: abs(float(c["t_c"]) - 0.12))
    use_cands = cands_sorted[: max(12, len(cands_sorted) // 2)]
    for cand in use_cands:
        for open_steps in open_opts:
            for match in match_opts:
                for gain in gain_opts:
                    for brake in brake_opts:
                        for cv in close_v_opts:
                            for fc in force_close_opts:
                                p = CatchParams(
                                    open_steps=int(open_steps),
                                    match=float(match),
                                    approach_gain=float(gain),
                                    brake=float(brake),
                                    close_vrel_max=float(cv),
                                    force_close_after_s=float(fc),
                                    t_c=float(cand["t_c"]),
                                    s_axial=float(cand["s"]),
                                )
                                pairs.append((p, cand))
    rng.shuffle(pairs)
    return pairs


def classify_limiting_factor(best: dict | None, act: dict | None) -> str:
    """Quantitative limiting-constraint label when SUCCESS_NEW is absent."""
    if best is None:
        return "search_failure_no_candidate"
    if best.get("SUCCESS_NEW"):
        return "none"
    if int(best.get("jaw_entered") or 0) == 0:
        open_ok = True
        if act and act.get("open_hold", {}).get("t_to_0p04") is None:
            open_ok = False
        if not open_ok:
            return "opening_time"
        if float(best.get("open_at_intercept") or 0) < 0.035:
            return "opening_time"
        return "jaw_entry_geometry"
    if best.get("vrel_at_first_contact") is not None and float(best["vrel_at_first_contact"]) > 0.7:
        return "contact_impulse"
    if float(best.get("max_consec_bil") or 0) < 0.06:
        if int(best.get("closing_triggered") or 0):
            return "closing_ejection_or_miss"
        return "hand_speed_or_intercept_miss"
    if float(best.get("max_consec_bil") or 0) >= 0.15 and not best.get("SUCCESS_NEW"):
        if best.get("reason") in ("DROP_AFTER_HANDOFF", "LOST_GRASP", "lost_before_pose_correct"):
            return "braking_or_handoff_slip"
        return "braking_induced_slip"
    if int(best.get("ground_drop") or 0):
        return "ground_too_soon"
    return "search_failure"
