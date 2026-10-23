"""Post-catch end-grasp recovery: slow lift vs gravity-assisted axial sliding.

Preserves dynamic-catch interception/braking tapes. Legal 7D only.
Targets remain locked U_TGT / P_TGT from evaluation.success_criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from controllers.recovery.action_map import TAU_OPEN, TAU_SECURE
from data_generation.oracles.plan_c.interception import finger_opening_m
from data_generation.oracles.plan_c.future_trajectory import (
    PINCH_Z,
    empty_bilateral_stats,
    hold_action,
    pad_bilateral,
    pose_correct_and_handoff,
    sample_kin,
    update_bilateral_stats,
)
from data_generation.oracles.plan_b.release_recatch import _action_from
from envs.contacts.frames import extract_contacts
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.simulation.ground_scene import CYL_HALF, CYL_REST_Z, GROUND_DROP_Z
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import measure
from experiments.audits.plan_a_mechanism_audit import sample_kinematics
from training.recovery_execution import make_snap, restore_ballistic

DT_POL = 0.02
G_WORLD = np.array([0.0, 0.0, -9.81], float)


@dataclass
class LiftParams:
    vz_up: float = 0.04  # world-up m/s
    grip: float = 1.0
    hold_orient: bool = True
    max_s: float = 4.0


@dataclass
class SlideParams:
    wrist_axis: int = 0  # 0=wx, 1=wy, 2=wz in body/r_des frame
    wrist_sign: float = 1.0
    wrist_speed: float = 0.8  # rad/s commanded
    grip: float = 0.35  # lighter for sliding
    retighten_grip: float = 1.0
    slide_s: float = 0.40
    retighten_s: float = 0.20
    compensate_com: float = 0.6  # track object COM laterally
    vz_hold: float = 0.0  # world-up hold during slide


def replay_to_brake_complete(sim, snap, capture_actions: list | np.ndarray) -> dict:
    """Replay catch capture tape; return brake-complete snapshot + audit."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    acts = [np.asarray(a, np.float32) for a in capture_actions]
    t0 = float(sim.data.time)
    saw_none = False
    none_run = 0.0
    max_none = 0.0
    bil = empty_bilateral_stats()
    first_bil_after = None
    for a in acts:
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        cs = contact_summary(sim)
        anyc = pb["any_pad"] or (cs["nL"] + cs["nR"] > 0)
        if not anyc:
            saw_none = True
            none_run += DT_POL
            max_none = max(max_none, none_run)
        else:
            none_run = 0.0
            if saw_none and pb["bilateral"] and first_bil_after is None:
                first_bil_after = float(sim.data.time) - t0
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        if latched(sim):
            break
    brake_snap = make_snap(sim)
    geo = grasp_geometry(sim)
    return {
        "brake_snap": brake_snap,
        "saw_full_detach": saw_none,
        "max_contact_free_s": max_none,
        "first_bil_after_detach": first_bil_after,
        "continuous_150": bool(bil.get("continuous_ge_150ms_at_end")),
        "max_consec_bil": float(bil.get("max_consecutive_bil_s") or 0),
        "end_bilateral": bool(pad_bilateral(sim)["bilateral"]),
        "airborne": bool(not latched(sim) and geo["obj_z"] > GROUND_DROP_Z),
        "ground_drop": int(latched(sim)),
        "geometry": geo,
        "n_actions": len(acts),
        "t_elapsed": float(sim.data.time) - t0,
    }


def grasp_geometry(sim) -> dict[str, Any]:
    """End-grasp diagnostics from MuJoCo contacts + kinematics."""
    k = sample_kinematics(sim)
    kin = sample_kin(sim)
    po = k["p_obj"]
    u = kin["u_obj"]
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    s_mean = k["contact_s_mean"]
    if not np.isfinite(s_mean):
        # fallback: COM axial from pinch along cylinder
        Rh, ph = kin["R_hand"], kin["p_hand"]
        prel = Rh.T @ (po - ph)
        pinch_w = ph + Rh @ np.array([0.0, 0.0, PINCH_Z], float)
        s_mean = float(np.dot(po - pinch_w, u))

    dist_end_plus = float(CYL_HALF - s_mean)
    dist_end_minus = float(CYL_HALF + s_mean)
    near_end = "plus" if abs(dist_end_plus) < abs(dist_end_minus) else "minus"
    dist_nearest_end = min(abs(dist_end_plus), abs(dist_end_minus))
    # Interior direction: reduce |s_mean| → slide object along -sign(s)*u relative to fingers
    desired_obj_slide_sign = float(-np.sign(s_mean)) if abs(s_mean) > 1e-4 else 0.0
    g_axial = float(np.dot(G_WORLD, u))
    # Gravity helps if it pulls object along desired slide direction
    g_helps = bool(desired_obj_slide_sign != 0.0 and g_axial * desired_obj_slide_sign > 0.0)

    rows, _ = extract_contacts(sim)
    pad_rows = [r for r in rows if r.get("side") in ("L", "R")]
    clearance = float(po[2] - CYL_REST_Z)

    Rh = kin["R_hand"]
    prel = Rh.T @ (po - ph) if (ph := kin["p_hand"]) is not None else np.zeros(3)
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)

    return {
        "obj_z": float(po[2]),
        "hand_z": float(kin["p_hand"][2]),
        "clearance_m": clearance,
        "u_world": u.tolist(),
        "u_hand": (Rh.T @ u).tolist(),
        "contact_s_mean": float(s_mean) if np.isfinite(s_mean) else None,
        "dist_end_plus_m": dist_end_plus,
        "dist_end_minus_m": dist_end_minus,
        "dist_nearest_end_m": dist_nearest_end,
        "near_end": near_end,
        "near_end_fraction": float(1.0 - dist_nearest_end / max(CYL_HALF, 1e-6)),
        "com_to_pinch_axial": float(np.dot(po - (kin["p_hand"] + Rh @ np.array([0.0, 0.0, PINCH_Z])), u)),
        "g_axial": g_axial,
        "desired_obj_slide_sign": desired_obj_slide_sign,
        "gravity_helps_now": int(g_helps),
        "n_pad_contacts": len(pad_rows),
        "fnL": float(pb["fnL"]),
        "fnR": float(pb["fnR"]),
        "bilateral": int(pb["bilateral"]),
        "opening_m": finger_opening_m(sim),
        "v_axial": float(k["v_axial"]),
        "v_obj_norm": float(np.linalg.norm(kin["v_obj"])),
        "v_hand_norm": float(np.linalg.norm(kin["v_hand"])),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "p_rel": prel.tolist(),
        "cyl_half": float(CYL_HALF),
    }


def _world_up_to_action(sim, vz_up: float, w_body=None, grip: float = 1.0) -> np.ndarray:
    """Command world +z velocity vz_up via legal AIRBORNE_V3 (hand -z is world up on horiz grasp)."""
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    v_world = np.array([0.0, 0.0, float(vz_up)], float)
    # Optional small xy hold / compensate later
    w = np.zeros(3) if w_body is None else np.asarray(w_body, float)
    tau = TAU_SECURE if grip >= 0.5 else (TAU_OPEN + 0.5 * (grip + 1.0) * (TAU_SECURE - TAU_OPEN))
    # grip in [-1,1]: map via action[6]
    a = _action_from(sim, v_world, w, tau)
    a[6] = float(np.clip(grip if grip <= 1.0 else 1.0, -1.0, 1.0))
    if grip < 0:
        a[6] = float(grip)
    return a


def fast_world_lift_sequence(
    sim,
    snap,
    *,
    vz_up: float = 0.12,
    target_z: float = 0.55,
    max_s: float = 8.0,
    grip: float = 1.0,
) -> dict:
    """World-+z lift via legal 7D (verified path for low-altitude sidewall catches).

    Prefer this when ``slow_lift_sequence`` stalls near the ground. Does not change
    physics; commands world-up velocity through ``_action_from``.
    """
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    g0 = grasp_geometry(sim)
    actions: list[np.ndarray] = []
    bil = empty_bilateral_stats()
    t0 = float(sim.data.time)
    reached = False
    while float(sim.data.time) - t0 < max_s - 1e-9:
        geo = grasp_geometry(sim)
        if geo["obj_z"] >= target_z - 0.02 and pad_bilateral(sim)["bilateral"]:
            reached = True
            break
        a = _action_from(sim, np.array([0.0, 0.0, float(vz_up)], float), np.zeros(3), TAU_SECURE)
        a[6] = float(np.clip(grip, -1.0, 1.0))
        actions.append(a.copy())
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        if latched(sim) or not pb["bilateral"]:
            break
    for _ in range(12):
        actions.append(hold_action(1.0))
        step_legal(sim, hold_action(1.0))
        pb = pad_bilateral(sim)
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        if latched(sim) or not pb["bilateral"]:
            break
    g1 = grasp_geometry(sim)
    return {
        "actions": actions,
        "end_snap": make_snap(sim),
        "reached_target_z": int(reached),
        "obj_z_end": float(g1["obj_z"]),
        "hand_z_end": float(g1["hand_z"]),
        "end_bilateral": int(g1["bilateral"]),
        "continuous_150": bool(bil.get("continuous_ge_150ms_at_end")),
        "max_consec_bil": float(bil.get("max_consecutive_bil_s") or 0),
        "ground_drop": int(latched(sim)),
        "g0": g0,
        "g1": g1,
        "vz_up": float(vz_up),
        "target_z": float(target_z),
    }


def slow_lift_sequence(sim, snap, params: LiftParams, *, target_z: float) -> dict:
    """Branch A: maintain grip, slow world-up lift toward reference height."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    g0 = grasp_geometry(sim)
    z0_obj = g0["obj_z"]
    z0_hand = g0["hand_z"]
    actions = []
    bil = empty_bilateral_stats()
    t0 = float(sim.data.time)
    min_clear = g0["clearance_m"]
    max_dz_obj = 0.0
    reached = False
    s0 = g0["contact_s_mean"]
    traces = []
    n_steps = int(round(params.max_s / DT_POL))
    for i in range(n_steps):
        geo = grasp_geometry(sim)
        min_clear = min(min_clear, geo["clearance_m"])
        max_dz_obj = max(max_dz_obj, geo["obj_z"] - z0_obj)
        if geo["obj_z"] >= target_z - 0.01:
            reached = True
            break
        w = np.zeros(3)
        a = _world_up_to_action(sim, params.vz_up, w_body=w, grip=params.grip)
        actions.append(a.copy())
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        if i % 5 == 0:
            traces.append({
                "t_rel": float(sim.data.time) - t0,
                "obj_z": geo["obj_z"],
                "hand_z": geo["hand_z"],
                "clearance": geo["clearance_m"],
                "bilateral": geo["bilateral"],
                "contact_s": geo["contact_s_mean"],
                "v_hand": float(np.linalg.norm(sample_kin(sim)["v_hand"])),
            })
        if latched(sim) or not pb["bilateral"]:
            break
    # measure actual lift speed over first 0.2 s of traces
    actual_vz = None
    if len(traces) >= 2:
        dt = traces[-1]["t_rel"] - traces[0]["t_rel"]
        if dt > 1e-6:
            actual_vz = (traces[-1]["obj_z"] - traces[0]["obj_z"]) / dt
    g1 = grasp_geometry(sim)
    ds = None
    if s0 is not None and g1["contact_s_mean"] is not None:
        ds = float(g1["contact_s_mean"] - s0)
    return {
        "actions": actions,
        "end_snap": make_snap(sim),
        "vz_cmd": params.vz_up,
        "actual_obj_vz_est": actual_vz,
        "dz_obj": float(g1["obj_z"] - z0_obj),
        "dz_hand": float(g1["hand_z"] - z0_hand),
        "obj_z_end": g1["obj_z"],
        "hand_z_end": g1["hand_z"],
        "min_clearance": min_clear,
        "max_dz_obj": max_dz_obj,
        "reached_target_z": int(reached),
        "target_z": target_z,
        "end_bilateral": int(g1["bilateral"]),
        "continuous_60": bool(bil.get("continuous_ge_60ms_at_end")),
        "continuous_150": bool(bil.get("continuous_ge_150ms_at_end")),
        "max_consec_bil": float(bil.get("max_consecutive_bil_s") or 0),
        "ground_drop": int(latched(sim)),
        "ds_contact_axial": ds,
        "cylinder_rose_with_hand": int(g1["obj_z"] - z0_obj > 0.5 * max(1e-6, g1["hand_z"] - z0_hand)),
        "g0": g0,
        "g1": g1,
        "params": asdict(params),
    }


def gravity_slide_sequence(sim, snap, params: SlideParams) -> dict:
    """Branch B: wrist rotation + grip modulation for axial contact migration."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    g0 = grasp_geometry(sim)
    s0 = g0["contact_s_mean"]
    desired = float(g0["desired_obj_slide_sign"])
    actions = []
    bil = empty_bilateral_stats()
    t0 = float(sim.data.time)
    min_clear = g0["clearance_m"]
    max_abs_s = abs(s0) if s0 is not None else None
    n_slide = int(round(params.slide_s / DT_POL))
    n_tight = int(round(params.retighten_s / DT_POL))

    def _step(grip, wrist_on: bool):
        kin = sample_kin(sim)
        Rh, ph, po = kin["R_hand"], kin["p_hand"], kin["p_obj"]
        # Compensate COM drift in world xy; optional small up hold
        err_xy = (po - ph)
        err_xy[2] = 0.0
        v_world = params.compensate_com * err_xy + np.array([0.0, 0.0, params.vz_hold])
        w_body = np.zeros(3)
        if wrist_on:
            w_body[int(params.wrist_axis)] = float(params.wrist_sign) * float(params.wrist_speed)
        a = _action_from(sim, v_world, w_body, TAU_SECURE if grip > 0.5 else TAU_OPEN)
        a[6] = float(np.clip(grip, -1.0, 1.0))
        actions.append(a.copy())
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        return grasp_geometry(sim), pb

    for _ in range(n_slide):
        geo, pb = _step(params.grip, True)
        min_clear = min(min_clear, geo["clearance_m"])
        if geo["contact_s_mean"] is not None and max_abs_s is not None:
            max_abs_s = min(max_abs_s, abs(geo["contact_s_mean"]))  # track best (closest to 0)
        if latched(sim) or not pb["bilateral"]:
            break
    for _ in range(n_tight):
        geo, pb = _step(params.retighten_grip, False)
        min_clear = min(min_clear, geo["clearance_m"])
        if latched(sim) or not pb["bilateral"]:
            break

    g1 = grasp_geometry(sim)
    ds = None
    toward_interior = 0
    if s0 is not None and g1["contact_s_mean"] is not None:
        ds = float(g1["contact_s_mean"] - s0)
        # Interior means |s| decreased
        toward_interior = int(abs(g1["contact_s_mean"]) + 1e-6 < abs(s0))
        # Also check sign matches desired object slide (ds_object relative):
        # if object slides +u relative to fingers and s0>0, s decreases
    g_axial_end = g1["g_axial"]
    return {
        "actions": actions,
        "end_snap": make_snap(sim),
        "s0": s0,
        "s1": g1["contact_s_mean"],
        "ds_contact_axial": ds,
        "toward_interior": toward_interior,
        "abs_s_reduced_m": None if s0 is None or g1["contact_s_mean"] is None else float(abs(s0) - abs(g1["contact_s_mean"])),
        "desired_slide_sign": desired,
        "g_axial_start": g0["g_axial"],
        "g_axial_end": g_axial_end,
        "gravity_helps_end": g1["gravity_helps_now"],
        "min_clearance": min_clear,
        "end_bilateral": int(g1["bilateral"]),
        "continuous_60": bool(bil.get("continuous_ge_60ms_at_end")),
        "continuous_150": bool(bil.get("continuous_ge_150ms_at_end")),
        "max_consec_bil": float(bil.get("max_consecutive_bil_s") or 0),
        "ground_drop": int(latched(sim)),
        "obj_z_end": g1["obj_z"],
        "dist_nearest_end_0": g0["dist_nearest_end_m"],
        "dist_nearest_end_1": g1["dist_nearest_end_m"],
        "genuine_axial_shift": int(
            toward_interior and ds is not None and abs(ds) >= 0.003  # >=3 mm Plan A threshold
        ),
        "g0": g0,
        "g1": g1,
        "params": asdict(params),
    }


def run_sequence(
    sim,
    brake_snap,
    ref,
    *,
    order: str,
    target_z: float,
    lift: LiftParams,
    slide: SlideParams,
    clearance_lift_z: float = 0.08,
) -> dict:
    """A1 / B1 / B2 composite sequences from the same brake-complete snapshot."""
    actions_all: list[np.ndarray] = []
    stages = []
    snap = brake_snap

    def _do_lift(p: LiftParams, tag: str, z_tgt: float):
        nonlocal snap
        out = slow_lift_sequence(sim, snap, p, target_z=z_tgt)
        actions_all.extend(out["actions"])
        snap = out["end_snap"]
        stages.append({"stage": tag, **{k: v for k, v in out.items() if k not in ("actions", "end_snap", "g0", "g1")}})
        return out

    def _do_slide(p: SlideParams, tag: str):
        nonlocal snap
        out = gravity_slide_sequence(sim, snap, p)
        actions_all.extend(out["actions"])
        snap = out["end_snap"]
        stages.append({"stage": tag, **{k: v for k, v in out.items() if k not in ("actions", "end_snap", "g0", "g1")}})
        return out

    lift_out = slide_out = None
    if order == "A1":
        # Brake → Slow Lift → Slide
        lift_out = _do_lift(lift, "slow_lift", target_z)
        if lift_out["end_bilateral"] and not lift_out["ground_drop"]:
            slide_out = _do_slide(slide, "slide_after_lift")
    elif order == "B1":
        # Brake → Slide → Lift
        slide_out = _do_slide(slide, "slide_first")
        if slide_out["end_bilateral"] and not slide_out["ground_drop"]:
            lift_out = _do_lift(lift, "lift_after_slide", target_z)
    elif order == "B2":
        # Brake → small clearance lift → slide → remaining lift
        restore_ballistic(sim, snap)
        restore_drop(sim, None)
        arm(sim)
        g = grasp_geometry(sim)
        z_partial = min(target_z, float(g["obj_z"]) + float(clearance_lift_z))
        lift_partial = LiftParams(vz_up=lift.vz_up, grip=lift.grip, max_s=min(2.0, lift.max_s))
        lift_out = _do_lift(lift_partial, "clearance_lift", z_partial)
        if lift_out["end_bilateral"] and not lift_out["ground_drop"]:
            slide_out = _do_slide(slide, "slide_mid")
            if slide_out["end_bilateral"] and not slide_out["ground_drop"]:
                lift_out = _do_lift(lift, "remaining_lift", target_z)
    else:
        raise ValueError(order)

    # Pose recovery + handoff if elevated and bilateral
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    geo = grasp_geometry(sim)
    pose_meta = {"SUCCESS_NEW": False, "reason": "not_attempted"}
    pose_ok = False
    if geo["bilateral"] and geo["obj_z"] >= min(target_z - 0.05, 0.35) and not latched(sim):
        # Brief secure hold
        for _ in range(5):
            actions_all.append(hold_action(1.0))
            step_legal(sim, hold_action(1.0))
            if not pad_bilateral(sim)["bilateral"] or latched(sim):
                break
        if pad_bilateral(sim)["bilateral"] and not latched(sim):
            pose_acts, pose_meta = pose_correct_and_handoff(sim, ref, max_s=3.5)
            actions_all.extend(pose_acts)
            pose_ok = bool(pose_meta.get("SUCCESS_NEW"))
            m = measure(sim, U_TGT, P_TGT)
            pose_meta = {**pose_meta, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                         "obj_z": float(sample_kin(sim)["obj_z"])}

    return {
        "order": order,
        "actions": actions_all,
        "stages": stages,
        "lift": lift_out,
        "slide": slide_out,
        "end_geometry": geo,
        "pose_meta": pose_meta,
        "SUCCESS_NEW": bool(pose_meta.get("SUCCESS_NEW", False)),
        "reached_height": int(
            (lift_out or {}).get("reached_target_z", 0)
            or geo["obj_z"] >= target_z - 0.02
        ),
        "genuine_slide": int((slide_out or {}).get("genuine_axial_shift", 0)),
        "end_bilateral": int(geo["bilateral"]),
        "ground_drop": int(latched(sim)),
        "final_snap": make_snap(sim),
    }


def score_sequence(out: dict) -> float:
    return (
        100 * int(out.get("SUCCESS_NEW") or 0)
        + 40 * int(out.get("reached_height") or 0)
        + 25 * int(out.get("genuine_slide") or 0)
        + 15 * int(out.get("end_bilateral") or 0)
        + 10 * float(((out.get("lift") or {}) or {}).get("dz_obj") or 0)
        + 5 * float(((out.get("slide") or {}) or {}).get("abs_s_reduced_m") or 0 or 0)
        - 20 * int(out.get("ground_drop") or 0)
    )
