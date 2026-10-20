"""Impact-aligned early Recovery and jaw-entry feasibility (GROUND_SCENE).

Reconstructs impact→loss timelines, branch snapshots A–E, jaw-entry diagnostics,
and early post-impact legal 7D search. No pre-impact Recovery. No physics edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, TAU_OPEN, TAU_SECURE, map_action
from data_generation.strategies.ground_future_oracle import (
    PINCH_Z,
    STABLE_BIL_150_S,
    STABLE_BIL_S,
    empty_bilateral_stats,
    exact_replay_actions,
    hold_action,
    install_near_end_grasp,
    launch_ball,
    pad_bilateral,
    pose_correct_and_handoff,
    sample_kin,
    settle_grasp,
    topology,
    update_bilateral_stats,
)
from data_generation.strategies.recatch_oracle import _action_from
from envs.contacts.frames import extract_contacts
from envs.outcomes.ground_drop import arm, disarm, latched, restore_state as restore_drop
from envs.outcomes.support_drop import object_vz, observe
from envs.simulation.ground_scene import CYL_HALF, CYL_R
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.full_objective import contact_summary
from training.recovery_execution import make_snap, restore_ballistic

DT = 0.002
DT_POL = 0.02
FN_MIN = 0.05

# Mechanism labels (exact strings required)
POST_IMPACT_GRASP_PRESERVATION = "POST_IMPACT_GRASP_PRESERVATION"
PARTIAL_SLIP_RECOVERY = "PARTIAL_SLIP_RECOVERY"
FULL_DETACH_ACTIVE_RECATCH = "FULL_DETACH_ACTIVE_RECATCH"


def finger_opening_m(sim) -> float:
    """Mean finger joint qpos (m). Uses qposadr — not joint-id indexing."""
    vals = []
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            vals.append(float(sim.data.qpos[int(sim.model.jnt_qposadr[jid])]))
    return float(np.mean(vals)) if vals else float("nan")


def finger_opening_vel(sim) -> float:
    vals = []
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            vals.append(float(sim.data.qvel[int(sim.model.jnt_dofadr[jid])]))
    return float(np.mean(vals)) if vals else float("nan")


def geometry_pack(sim) -> dict[str, Any]:
    k = sample_kin(sim)
    Rh, ph, po = k["R_hand"], k["p_hand"], k["p_obj"]
    prel = Rh.T @ (po - ph)
    open_m = finger_opening_m(sim)
    pb = pad_bilateral(sim)
    rows, _ = extract_contacts(sim)
    housing = sum(1 for r in rows if r.get("side") == "hand")
    # Required clearance: cylinder radius projected into jaw (hand y)
    u = k["u_obj"]
    u_h = Rh.T @ u
    # radial extent in jaw opening direction (hand y)
    rad_in_jaw = float(CYL_R * np.sqrt(max(0.0, 1.0 - u_h[1] ** 2)))
    need_clear = 2.0 * rad_in_jaw + 0.002
    jaw_sep = float(open_m)  # approximate pad separation proxy
    axial_from_pinch = float(prel[2] - PINCH_Z)
    return {
        "t": float(sim.data.time),
        "topo": topology(sim),
        "bilateral": bool(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "fnL": float(pb["fnL"]),
        "fnR": float(pb["fnR"]),
        "any_pad": bool(pb["any_pad"]),
        "housing": int(housing),
        "opening_m": open_m,
        "opening_vel": finger_opening_vel(sim),
        "p_rel": prel.copy(),
        "lateral_offset": float(prel[1]),
        "axial_from_pinch": axial_from_pinch,
        "abs_lateral": abs(float(prel[1])),
        "rad_in_jaw": rad_in_jaw,
        "need_clear": need_clear,
        "jaw_sep": jaw_sep,
        "jaw_clearance": jaw_sep - need_clear,
        "obj_z": float(po[2]),
        "v_obj": k["v_obj"].copy(),
        "v_hand": k["v_hand"].copy(),
        "v_rel": (k["v_obj"] - k["v_hand"]).copy(),
        "v_rel_norm": float(np.linalg.norm(k["v_obj"] - k["v_hand"])),
        "u_obj": k["u_obj"].copy(),
        "ground_drop": int(latched(sim)),
        "cs_nL": int(contact_summary(sim)["nL"]),
        "cs_nR": int(contact_summary(sim)["nR"]),
    }


def ball_hits(sim) -> bool:
    from envs.ballistic_impact import ball_force_on_ball

    _F, rows = ball_force_on_ball(sim)
    return any(r.get("kind") == "ball-cylinder" for r in rows)


def _policy_hold_substeps(sim, act: np.ndarray, rows: list, t0: float) -> None:
    """One 20 ms legal step with per-mj logging into rows."""
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(np.asarray(act, np.float32), rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    for _ in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, DT)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz)
        g = geometry_pack(sim)
        g["t_rel"] = float(sim.data.time) - t0
        g["ball_hit"] = int(ball_hits(sim))
        rows.append(g)


@dataclass
class TimelineEvents:
    first_ball_contact_t: float | None = None
    first_bilateral_depart_t: float | None = None  # physical onset: not bilateral
    first_unilateral_t: float | None = None
    first_no_finger_t: float | None = None
    first_full_separation_t: float | None = None  # no pad + no housing
    first_sustained_loss_t: float | None = None  # 20 ms confirmed bilateral loss
    first_ground_drop_t: float | None = None


def reconstruct_impact_timeline(
    sim,
    *,
    cfg: dict,
    ball: dict,
    max_s: float = 1.0,
) -> dict:
    """From settled grasp: launch ball, hold closed (no Recovery chase), log mj steps."""
    # Install/settle expected already done OR we rebuild:
    restore_drop(sim, None)
    disarm(sim)
    # Ensure ball launch from current object pose using saved launch if provided
    if ball and "p_launch" in ball:
        from envs.simulation.impact_ball import apply_ball_free_state

        sim.set_guide_weld(False)
        apply_ball_free_state(sim, np.asarray(ball["p_launch"], float), np.asarray(ball["v0"], float))
        mujoco.mj_forward(sim.model, sim.data)
    else:
        launch_ball(
            sim,
            speed=float(cfg["speed"]),
            approach=str(cfg["approach"]),
            offset=float(cfg["offset"]),
            z_off=float(cfg.get("z_off", 0.0)),
        )

    pre_launch_bilateral = bool(pad_bilateral(sim)["bilateral"] or contact_summary(sim)["bilateral"])
    act = hold_action(1.0)  # no Recovery chase before/during impact hold
    t0 = float(sim.data.time)
    rows: list[dict] = []
    ev = TimelineEvents()
    uni_run = 0.0
    none_bil_run = 0.0
    was_bilateral = True
    armed = False
    snaps: dict[str, Any] = {}
    # A–E filled as events occur; E = late sustained loss (prior style)

    # initial sample
    g0 = geometry_pack(sim)
    g0["t_rel"] = 0.0
    g0["ball_hit"] = int(ball_hits(sim))
    rows.append(g0)

    while float(sim.data.time) - t0 < max_s - 1e-9:
        if not armed and ball_hits(sim):
            arm(sim)
            armed = True
        _policy_hold_substeps(sim, act, rows, t0)
        # scan new rows for events (last N_SUB)
        for g in rows[-N_SUB:]:
            tr = float(g["t_rel"])
            if g["ball_hit"] and ev.first_ball_contact_t is None:
                ev.first_ball_contact_t = tr
                if "A" not in snaps:
                    snaps["A"] = make_snap(sim)
                    snaps["A_t"] = tr
                    snaps["A_note"] = "first_ball_cylinder_contact_policy_boundary"
            bil = bool(g["bilateral"] or (g["cs_nL"] > 0 and g["cs_nR"] > 0))
            any_finger = bool(g["any_pad"] or g["cs_nL"] + g["cs_nR"] > 0)
            if was_bilateral and not bil and ev.first_bilateral_depart_t is None:
                ev.first_bilateral_depart_t = tr
                if "B" not in snaps:
                    snaps["B"] = make_snap(sim)
                    snaps["B_t"] = tr
                    snaps["B_note"] = "first_observable_contact_degradation"
                if "C" not in snaps:
                    snaps["C"] = snaps["B"]
                    snaps["C_t"] = tr
                    snaps["C_note"] = "first_complete_loss_of_bilateral"
            nL = int(g["cs_nL"])
            nR = int(g["cs_nR"])
            if not bil and ((nL > 0) ^ (nR > 0)) and ev.first_unilateral_t is None:
                ev.first_unilateral_t = tr
            if not bil:
                none_bil_run += DT
            else:
                none_bil_run = 0.0
            if not any_finger:
                if ev.first_no_finger_t is None:
                    ev.first_no_finger_t = tr
                if int(g["housing"]) == 0 and ev.first_full_separation_t is None:
                    ev.first_full_separation_t = tr
            if none_bil_run + 1e-12 >= 0.020 and ev.first_sustained_loss_t is None:
                ev.first_sustained_loss_t = tr
                if "D" not in snaps:
                    snaps["D"] = make_snap(sim)
                    snaps["D_t"] = tr
                    snaps["D_note"] = "first_confirmed_20ms_bilateral_loss"
            if g["ground_drop"] and ev.first_ground_drop_t is None:
                ev.first_ground_drop_t = tr
            was_bilateral = bil
        if latched(sim):
            break
        # Keep rolling after sustained loss until GROUND_DROP (or max_s) so remain_s is real.
        # Still record E once shortly after D for the late post-loss branch.
        if ev.first_sustained_loss_t is not None and "E" not in snaps:
            if float(sim.data.time) - t0 > float(ev.first_sustained_loss_t) + 0.04:
                snaps["E"] = make_snap(sim)
                snaps["E_t"] = float(sim.data.time) - t0
                snaps["E_note"] = "late_post_loss_after_sustained"

    if "E" not in snaps:
        snaps["E"] = make_snap(sim)
        snaps["E_t"] = float(sim.data.time) - t0
        snaps["E_note"] = "late_post_loss_end_of_hold"

    return {
        "events": ev.__dict__,
        "rows": rows,
        "snaps": snaps,
        "pre_launch_bilateral": pre_launch_bilateral,
        "t0": t0,
    }


def classify_jaw_entry(g: dict, *, t_ground: float | None, open_time_s: float) -> str:
    """Diagnostic jaw-entry class from geometry pack + measured open time.

    Nominal grasp sits near axial_from_pinch ≈ 0.025 m — that is *not* past the pads.
    Axial escape past usable finger pads is ~0.05 m+ distal of pinch.
    """
    axial = float(g.get("axial_from_pinch", 0.0))
    lat = abs(float(g.get("lateral_offset", 0.0)))
    remain = None if t_ground is None else max(0.0, float(t_ground) - float(g.get("t_rel", 0)))
    # Still load-bearing bilateral → object is already in the jaw (entry not required).
    if g.get("bilateral") and abs(axial) < 0.055 and lat < 0.03:
        return "ALREADY_INSIDE_JAW"
    # Any residual pad/housing contact with COM still in pad span.
    if (g.get("any_pad") or int(g.get("cs_nL", 0)) + int(g.get("cs_nR", 0)) > 0) and abs(axial) < 0.045:
        return "ALREADY_INSIDE_JAW"
    # Distal of usable pads (COM well beyond pinch + pad half-length).
    if axial > 0.055:
        return "AXIALLY_PAST_FINGERS"
    if lat > 0.040:
        return "LATERALLY_UNREACHABLE"
    if float(g.get("housing", 0)) > 0 and not g.get("any_pad") and not g.get("bilateral"):
        return "CONTACT_GEOMETRY_BLOCKED"
    need_open = max(0.0, float(g.get("need_clear", 0.04)) - float(g.get("opening_m", 0)))
    if remain is not None and remain < 0.06:
        return "GROUND_TOO_SOON"
    if need_open > 0.01 and open_time_s > 0 and remain is not None and open_time_s > remain - 0.04:
        return "OPENING_TOO_SLOW"
    if abs(axial) < 0.050 and lat < 0.030:
        # Opening may still be short of clearance, but geometry is within reach.
        return "JAW_ENTRY_POSSIBLE"
    if remain is not None and open_time_s > 0 and remain < open_time_s + 0.06:
        return "GROUND_TOO_SOON"
    return "CONTACT_GEOMETRY_BLOCKED"


def measure_opening_response(sim, snap, *, max_s: float = 0.40) -> dict:
    """Legal open (grip=-1) + velocity-matched hand; measure time to target opening."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    t0 = float(sim.data.time)
    open0 = finger_opening_m(sim)
    target = 0.030
    t_hit = None
    series = []
    while float(sim.data.time) - t0 < max_s - 1e-9:
        k = sample_kin(sim)
        # match object fall, open fingers
        a = _action_from(sim, 0.85 * k["v_obj"], np.zeros(3), TAU_OPEN)
        a[6] = -1.0
        step_legal(sim, a)
        o = finger_opening_m(sim)
        series.append({"t_rel": float(sim.data.time) - t0, "opening_m": o})
        if t_hit is None and o >= target:
            t_hit = float(sim.data.time) - t0
            break
        if latched(sim):
            break
    return {
        "open0": open0,
        "target": target,
        "t_to_target": t_hit,
        "open_end": finger_opening_m(sim),
        "series": series,
        "ground_drop": int(latched(sim)),
    }


@dataclass
class EarlyParams:
    open_steps: int = 0  # 0 = grasp-preserve / close-first (no deliberate open)
    match: float = 0.85
    gain: float = 6.0
    close_delay: int = 2
    brake: float = 0.5
    approach_axial: bool = True
    mode: str = "preserve"  # preserve | open_recatch


def early_recovery_sequence(sim, snap, ref, params: EarlyParams, *, horizon_s: float = 0.55) -> dict:
    """Legal 7D early Recovery from a post-impact branch snapshot."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    g0 = geometry_pack(sim)
    start_bil = bool(g0["bilateral"] or (g0["cs_nL"] > 0 and g0["cs_nR"] > 0))
    start_any = bool(g0["any_pad"] or g0["cs_nL"] + g0["cs_nR"] > 0)
    saw_full_detach = not start_any
    actions = []
    bil = empty_bilateral_stats()
    t0 = float(sim.data.time)
    n_steps = int(round(horizon_s / DT_POL))
    first_pad = None
    # Preserve: never open. Soft pose gain to avoid jerking the grasp open.
    preserve = (params.mode == "preserve") or (params.open_steps <= 0 and start_bil)
    gain = float(params.gain)
    if preserve:
        gain = min(gain, 4.0)
    for i in range(n_steps):
        k = sample_kin(sim)
        Rh, ph, po = k["R_hand"], k["p_hand"], k["p_obj"]
        prel = Rh.T @ (po - ph)
        target = np.array([0.0, 0.0, PINCH_Z], float)
        err = target - prel
        v_h = -err * gain
        v_w = Rh @ np.clip(v_h, -0.85, 0.85)
        v_w = v_w + params.match * k["v_obj"]
        if preserve or params.open_steps <= 0:
            # Close / secure immediately; match + brake relative velocity.
            v_w = (1.0 - params.brake) * v_w + params.brake * (0.10 * k["v_obj"])
            tau = TAU_SECURE
            grip_a = 1.0
        elif i < params.open_steps:
            tau = TAU_OPEN
            grip_a = -1.0
        elif i < params.open_steps + params.close_delay:
            tau = 0.5 * (TAU_OPEN + TAU_SECURE)
            grip_a = 0.0
        else:
            v_w = (1.0 - params.brake) * v_w + params.brake * (0.15 * k["v_obj"])
            tau = TAU_SECURE
            grip_a = 1.0
        a = _action_from(sim, v_w, np.zeros(3), tau)
        a[6] = grip_a
        actions.append(a.copy())
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        if first_pad is None and pb["any_pad"]:
            first_pad = float(sim.data.time) - t0
        update_bilateral_stats(bil, bool(pb["bilateral"]), float(sim.data.time) - t0, DT_POL)
        cs = contact_summary(sim)
        if not pb["any_pad"] and not (cs["nL"] + cs["nR"] > 0):
            saw_full_detach = True
        if latched(sim):
            break

    # Extend hold while bilateral to accumulate consecutive capture time.
    if pad_bilateral(sim)["bilateral"]:
        for _ in range(12):
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
    # Mechanism labels (distinct; preservation ≠ detached recatch).
    # Continuity is measured on the Recovery segment before pose/handoff.
    if start_bil and end_bil and not saw_full_detach and (c60 or max_consec >= 0.06):
        mech = POST_IMPACT_GRASP_PRESERVATION
    elif start_any and end_bil and not saw_full_detach and (c60 or end_bil):
        mech = PARTIAL_SLIP_RECOVERY
    elif saw_full_detach and end_bil and c60:
        mech = FULL_DETACH_ACTIVE_RECATCH
    else:
        mech = "FAILED_OR_TRANSIENT"

    success_meta = {"SUCCESS_NEW": False, "reason": "no_stable_capture"}
    n_capture = len(actions)
    if end_bil and c150 and not latched(sim):
        # Extra secure hold to rebuild pad load before pose/handoff.
        for _ in range(10):
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

    return {
        "actions": actions,
        "capture_actions": actions[:n_capture],
        "n_capture": n_capture,
        "bil": bil,
        "mechanism": mech,
        "first_pad_t": first_pad,
        "start_bilateral": start_bil,
        "start_any_contact": start_any,
        "saw_full_detach": saw_full_detach,
        "end_bilateral": end_bil,
        "ground_drop": int(latched(sim)),
        "SUCCESS_NEW": bool(success_meta.get("SUCCESS_NEW", False)),
        "reason": success_meta.get("reason"),
        "continuous_60": c60,
        "continuous_150": c150,
        "max_consec_bil": max_consec,
        "mode": "preserve" if preserve else "open_recatch",
        "g0": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g0.items() if k != "u_obj"},
    }


def early_param_grid(rng: np.random.Generator, *, prefer_preserve: bool = True) -> list[EarlyParams]:
    """Mixed preserve (open_steps=0) and open-recatch schedules. No fixed 80 ms open."""
    grid: list[EarlyParams] = []
    # Grasp preservation / partial-slip close-first
    for match in (0.4, 0.7, 0.9, 1.0, 1.1):
        for gain in (1.0, 2.0, 3.5, 5.0):
            for brake in (0.0, 0.3, 0.6, 0.9):
                grid.append(EarlyParams(0, match, gain, 0, brake, mode="preserve"))
    # Open-then-close recatch (variable onset, not fixed 80 ms)
    for open_steps in (1, 2, 3, 4, 6, 8):
        for match in (0.5, 0.85, 1.05):
            for gain in (3.0, 6.0, 10.0):
                for close_delay in (0, 1, 2, 4):
                    for brake in (0.0, 0.4, 0.8):
                        grid.append(EarlyParams(open_steps, match, gain, close_delay, brake, mode="open_recatch"))
    rng.shuffle(grid)
    if prefer_preserve:
        grid.sort(key=lambda p: 0 if p.mode == "preserve" else 1)
    return grid


def rebuild_settled_root(sim, cfg: dict) -> dict:
    """Reinstall near-end grasp + settle (pre-impact)."""
    install_near_end_grasp(
        sim,
        z=float(cfg["z"]),
        axial_offset=float(cfg["axial_offset"]),
        tilt_deg=float(cfg.get("tilt_deg", 0.0)),
    )
    pre = settle_grasp(sim, 0.45)
    return pre
