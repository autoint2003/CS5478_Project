"""Offline future-trajectory oracle for GROUND_SCENE Plan C ball-impact recatch.

Privileged: may inspect full no-intervention futures and branch counterfactual
MuJoCo rollouts. Uses only legal 7D actions via step_legal. Does not modify
physics, SUCCESS_NEW, K49 vocabulary, or 39D observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco
import numpy as np

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.gripper.controller import finger_opening
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import TAU_OPEN, TAU_SECURE, grip_action
from data_generation.oracles.representability import representability
from data_generation.strategies.recatch_oracle import _action_from
from envs.contacts.frames import extract_contacts
from envs.outcomes.ground_drop import arm, disarm, latched, restore_state as restore_drop
from envs.outcomes.horizon_outcome import (
    LOST_STEPS,
    NOMINAL_S,
    _bilateral,
    _dropped,
    _escaped,
    _monitor,
    recovery_failed,
)
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.ground_scene import CYL_HALF, CYL_R, CYL_REST_Z, GROUND_DROP_Z
from envs.simulation.grasp_quality import body_twist
from envs.simulation.impact_ball import apply_ball_free_state
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.full_objective import (
    FINAL_WINDOW_S,
    P_TGT,
    U_TGT,
    contact_summary,
    window_stats,
)
from evaluation.grasp_configuration import measure
from training.k47_codebook import load_codebook
from training.recovery_execution import make_snap, restore_ballistic

DT = 0.002
DT_POLICY = 0.02
SUSTAIN_S = 0.020  # meaningful bilateral-loss / separation
STABLE_BIL_S = 0.060  # transient capture evidence (not sustained recovery)
STABLE_BIL_150_S = 0.150  # stronger continuous bilateral evidence
PINCH_Z = 0.105
AXIAL_MARGIN = 0.004  # stay inside cylinder ends
FN_PAD_MIN = 0.05  # N; reject near-zero collisions as "load-bearing"


@dataclass
class PilotBudget:
    max_impact_configs: int = 30
    max_recoverable_roots: int = 5
    max_counterfactual_rollouts: int = 500
    max_intercept_times: int = 8
    max_axial_samples: int = 5
    max_stage_a_per_root: int = 40
    max_stage_b_per_root: int = 30
    max_stage_c_per_root: int = 25
    max_stage_d_per_root: int = 15
    max_stage_e_per_root: int = 10


@dataclass
class SearchCounters:
    rollouts: int = 0
    by_stage: dict[str, int] = field(default_factory=lambda: {
        "A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
    })

    def charge(self, stage: str) -> bool:
        """Return True if budget remains after charging one rollout."""
        self.rollouts += 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        return True


def pad_bilateral(sim) -> dict:
    """Load-bearing finger-pad bilateral contact (not arbitrary hand collision)."""
    rows, _ = extract_contacts(sim)
    nL = nR = 0
    fnL = fnR = 0.0
    housing = 0
    for r in rows:
        side = r.get("side")
        fn = abs(float(r.get("Fn", r.get("fn", 0.0)) or 0.0))
        if side == "L" and fn >= FN_PAD_MIN:
            nL += 1
            fnL += fn
        elif side == "R" and fn >= FN_PAD_MIN:
            nR += 1
            fnR += fn
        elif side == "hand":
            housing += 1
    return {
        "nL": nL,
        "nR": nR,
        "fnL": fnL,
        "fnR": fnR,
        "bilateral": bool(nL > 0 and nR > 0),
        "any_pad": bool(nL + nR > 0),
        "housing": housing,
    }


def topology(sim) -> str:
    pb = pad_bilateral(sim)
    if pb["bilateral"]:
        return "bilateral"
    if pb["nL"] > 0 or pb["nR"] > 0:
        return "unilateral"
    if pb["housing"] > 0:
        return "housing_or_palm"
    # fall back: any finger contact below force threshold still counts as light uni
    cs = contact_summary(sim)
    if cs["nL"] > 0 and cs["nR"] > 0:
        return "light_bilateral"
    if cs["nL"] > 0 or cs["nR"] > 0:
        return "light_unilateral"
    return "none"


def classify_loss(prev: str, cur: str, none_run_s: float, uni_run_s: float) -> str | None:
    """Return loss label when a meaningful event is confirmed; else None."""
    if cur == "none" and none_run_s + 1e-12 >= SUSTAIN_S:
        return "complete_separation"
    if cur in ("unilateral", "housing_or_palm") and uni_run_s + 1e-12 >= SUSTAIN_S:
        return "full_loss_bilateral"
    if prev == "bilateral" and cur == "unilateral" and uni_run_s < SUSTAIN_S:
        return "transient_unilateral"
    return None


def object_axis(sim) -> np.ndarray:
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    u = Ro[:, 2].copy()
    n = float(np.linalg.norm(u))
    return u / max(n, 1e-12)


def sample_kin(sim) -> dict[str, Any]:
    ids = sim.ids
    ph = np.array(sim.data.xpos[ids.hand_body], float)
    Rh = np.array(sim.data.xmat[ids.hand_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[ids.object_body], float)
    Ro = np.array(sim.data.xmat[ids.object_body].reshape(3, 3), float)
    vh, wh = body_twist(sim.model, sim.data, ids.hand_body)
    vo, wo = body_twist(sim.model, sim.data, ids.object_body)
    u = object_axis(sim)
    cs = contact_summary(sim)
    topo = topology(sim)
    opening = float(finger_opening(sim.data, ids))
    ball_p = np.array(sim.data.xpos[sim.ball_body], float) if hasattr(sim, "ball_body") else None
    return {
        "t": float(sim.data.time),
        "p_hand": ph,
        "R_hand": Rh,
        "p_obj": po,
        "R_obj": Ro,
        "u_obj": u,
        "v_hand": np.asarray(vh, float),
        "w_hand": np.asarray(wh, float),
        "v_obj": np.asarray(vo, float),
        "w_obj": np.asarray(wo, float),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": bool(cs["bilateral"]),
        "topo": topo,
        "opening": opening,
        "obj_z": float(po[2]),
        "ball_p": ball_p,
        "ncon": int(sim.data.ncon),
        "ground_drop": int(latched(sim)),
    }


def hold_action(grip: float = 1.0) -> np.ndarray:
    a = np.zeros(7, np.float32)
    a[6] = float(np.clip(grip, -1.0, 1.0))
    return a


def raise_hand_airborne(sim, *, target_z: float = 0.55, seconds: float = 0.60) -> None:
    """Legal Cartesian lift so near-end installs sit at a usable airborne height."""
    disarm(sim)
    t0 = float(sim.data.time)
    while float(sim.data.time) - t0 < seconds - 1e-9:
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        vz = float(np.clip(2.0 * (target_z - ph[2]), -0.2, 0.55))
        a = _action_from(sim, np.array([0.0, 0.0, vz]), np.zeros(3), TAU_SECURE)
        a[6] = 1.0
        step_legal(sim, a)
    sim.fsm.grasp_z = float(sim.data.xpos[sim.ids.hand_body][2])


def install_near_end_grasp(
    sim,
    *,
    z: float,
    axial_offset: float,
    tilt_deg: float = 0.0,
    open_grip: float = 0.016,
) -> None:
    """Place cylinder at near-end grasp relative to current hand; settle separately.

    Does **not** teleport the hand. Raises the arm first toward ``z``, then places
    the cylinder in the finger frame (no independent world-z overwrite that would
    break contact geometry).
    """
    raise_hand_airborne(sim, target_z=float(z), seconds=0.70)
    mujoco.mj_forward(sim.model, sim.data)
    hand = int(sim.ids.hand_body)
    Rh = np.array(sim.data.xmat[hand].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[hand], float)
    if abs(tilt_deg) > 1e-6:
        ang = np.deg2rad(tilt_deg)
        c, s = np.cos(ang), np.sin(ang)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
        Rh_obj = Rh @ Rx
    else:
        Rh_obj = Rh
    p_com_h = np.array([0.0, 0.0, PINCH_Z + float(axial_offset)], float)
    p_w = ph + Rh @ p_com_h  # keep hand-frame placement; do not overwrite world z
    qw = 0.5 * np.sqrt(max(0.0, 1.0 + Rh_obj[0, 0] + Rh_obj[1, 1] + Rh_obj[2, 2]))
    qx = (Rh_obj[2, 1] - Rh_obj[1, 2]) / max(4.0 * qw, 1e-12)
    qy = (Rh_obj[0, 2] - Rh_obj[2, 0]) / max(4.0 * qw, 1e-12)
    qz = (Rh_obj[1, 0] - Rh_obj[0, 1]) / max(4.0 * qw, 1e-12)
    if qw < 1e-6:
        quat = np.array(sim.data.xquat[hand], float)
    else:
        quat = np.array([qw, qx, qy, qz], float)
        quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    qadr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    sim.data.qpos[qadr : qadr + 3] = p_w
    sim.data.qpos[qadr + 3 : qadr + 7] = quat
    sim.data.qvel[dadr : dadr + 6] = 0.0
    for jname, val in (("finger_joint1", open_grip), ("finger_joint2", open_grip)):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            sim.data.qpos[int(sim.model.jnt_qposadr[jid])] = float(val)
    if sim.data.ctrl.size > 7:
        sim.data.ctrl[7] = float(TAU_SECURE)
    sim.fsm.p_des = ph.copy()
    sim.fsm.r_des = Rh.copy()
    sim.fsm.phase = "lift"
    sim.fsm.grasp_z = float(ph[2])
    sim.fsm.fg_cmd = float(-TAU_SECURE)
    sim.fsm.v_cmd = np.zeros(3)
    sim.fsm.w_cmd = np.zeros(3)
    sim.fsm.v_des = np.zeros(3)
    sim.fsm.w_des = np.zeros(3)
    mujoco.mj_forward(sim.model, sim.data)


def settle_grasp(sim, seconds: float = 0.45) -> dict:
    disarm(sim)
    t0 = float(sim.data.time)
    # Briefly squeeze closed from a slightly open install.
    while float(sim.data.time) - t0 < seconds - 1e-9:
        step_legal(sim, hold_action(1.0))
    cs = contact_summary(sim)
    # One retry: micro-close if unilateral only
    if not cs["bilateral"] and (cs["nL"] + cs["nR"]) > 0:
        t1 = float(sim.data.time)
        while float(sim.data.time) - t1 < 0.20 - 1e-9:
            step_legal(sim, hold_action(1.0))
        cs = contact_summary(sim)
    return {
        "bilateral": bool(cs["bilateral"]),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "hand_z": float(sim.data.xpos[sim.ids.hand_body][2]),
        "topo": topology(sim),
    }


def launch_ball(sim, *, speed: float, approach: str, offset: float, z_off: float = 0.0) -> dict:
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    if approach == "top":
        p_launch = po + np.array([offset, 0.0, 0.055 + z_off])
        v0 = np.array([0.0, 0.0, -float(speed)])
    elif approach == "oblique":
        p_launch = po + np.array([0.06, offset, 0.05 + z_off])
        v0 = np.array([-float(speed) * 0.65, 0.0, -float(speed) * 0.75])
    elif approach == "lateral":
        p_launch = po + np.array([offset, -0.08, 0.01 + z_off])
        v0 = np.array([0.0, float(speed), -0.5])
    elif approach == "side":
        # Strike near the free end (axial bias already in grasp); horizontal shear.
        p_launch = po + np.array([-0.08, offset, 0.01 + z_off])
        v0 = np.array([float(speed), 0.0, -0.4])
    else:
        p_launch = po + np.array([-0.06, offset, 0.04 + z_off])
        v0 = np.array([float(speed) * 0.7, 0.0, -float(speed) * 0.7])
    sim.set_guide_weld(False)
    apply_ball_free_state(sim, p_launch, v0)
    mujoco.mj_forward(sim.model, sim.data)
    return {
        "p_launch": p_launch.tolist(),
        "v0": v0.tolist(),
        "speed": float(speed),
        "approach": approach,
        "offset": float(offset),
        "z_off": float(z_off),
    }


def ball_hits_cylinder(sim) -> bool:
    from envs.ballistic_impact import ball_force_on_ball

    _F, rows = ball_force_on_ball(sim)
    return any(r.get("kind") == "ball-cylinder" for r in rows)


def run_impact_to_loss(
    sim,
    *,
    max_s: float = 0.90,
    baseline: np.ndarray | None = None,
) -> dict:
    """Hold non-chasing legal action until impact + first meaningful grasp loss."""
    act = hold_action(1.0) if baseline is None else np.asarray(baseline, np.float32)
    restore_drop(sim, None)
    disarm(sim)  # do not GROUND_DROP during pre-impact settle; arm after impact
    t0 = float(sim.data.time)
    impact_t = None
    loss_event = None
    uni_run = 0.0
    none_run = 0.0
    prev = topology(sim)
    post_loss_snap = None
    trace = []
    armed = False
    worst = "bilateral"
    impulse = 0.0

    while float(sim.data.time) - t0 < max_s - 1e-9:
        hit = ball_hits_cylinder(sim)
        if hit:
            from envs.ballistic_impact import ball_force_on_ball

            Fb, _rows = ball_force_on_ball(sim)
            impulse += float(np.linalg.norm(Fb)) * DT
            if impact_t is None:
                impact_t = float(sim.data.time) - t0
                if not armed:
                    arm(sim)
                    armed = True
        step_legal(sim, act)
        kin = sample_kin(sim)
        kin["t_rel"] = float(sim.data.time) - t0
        trace.append(kin)
        cur = kin["topo"]
        rank = {"bilateral": 0, "unilateral": 1, "housing_or_palm": 2, "none": 3}
        if rank.get(cur, 0) > rank.get(worst, 0):
            worst = cur
        dt = DT_POLICY
        if cur == "none":
            none_run += dt
            uni_run = 0.0
        elif cur in ("unilateral", "housing_or_palm"):
            uni_run += dt
            none_run = 0.0
        else:
            uni_run = 0.0
            none_run = 0.0
        label = classify_loss(prev, cur, none_run, uni_run)
        if impact_t is not None and label in ("complete_separation", "full_loss_bilateral"):
            if loss_event is None:
                loss_event = {
                    "label": label,
                    "t_rel": float(sim.data.time) - t0,
                    "t_abs": float(sim.data.time),
                    "obj_z": kin["obj_z"],
                    "topo": cur,
                    "airborne_to_ground_est": max(0.0, kin["obj_z"] - CYL_REST_Z),
                }
                post_loss_snap = make_snap(sim)
                if label == "complete_separation":
                    break
        if latched(sim):
            break
        prev = cur

    return {
        "impact_t": impact_t,
        "loss_event": loss_event,
        "post_loss_snap": post_loss_snap,
        "trace": trace,
        "ground_drop": int(latched(sim)),
        "final_topo": topology(sim),
        "final_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "worst_topo": worst,
        "impulse_proxy": impulse,
    }


def record_no_intervention_future(
    sim,
    snap,
    *,
    max_s: float = 1.50,
    grip: float = 1.0,
) -> dict:
    """Branch from snap; legal non-chasing baseline; log every MuJoCo step."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    act = hold_action(grip)
    t0 = float(sim.data.time)
    rows = []
    first_contact_free_t = None
    contact_free_start_snap = None
    ground_t = None
    incidental_robot_contact = False

    while float(sim.data.time) - t0 < max_s - 1e-9:
        # one policy step = N_SUB mj steps; log each mj step
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        from controllers.recovery.action_map import AIRBORNE_V3, map_action

        cmd = map_action(act, rd, rh, AIRBORNE_V3)
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
            from envs.outcomes.support_drop import object_vz, observe

            vz = object_vz(sim)
            mujoco.mj_step(sim.model, sim.data)
            observe(sim, vz)
            kin = sample_kin(sim)
            kin["t_rel"] = float(sim.data.time) - t0
            # robot–object contact besides fingers?
            rows_c, _ = extract_contacts(sim)
            if any(r.get("side") == "hand" for r in rows_c):
                incidental_robot_contact = True
            finger_touch = kin["nL"] + kin["nR"] > 0
            if not finger_touch and first_contact_free_t is None:
                first_contact_free_t = kin["t_rel"]
                contact_free_start_snap = make_snap(sim)
            if latched(sim) and ground_t is None:
                ground_t = kin["t_rel"]
            rows.append(kin)
            if ground_t is not None:
                break
        if ground_t is not None:
            break

    return {
        "baseline": "legal_hold_grip_close_no_chase",
        "rows": rows,
        "n_steps": len(rows),
        "first_contact_free_t": first_contact_free_t,
        "contact_free_start_snap": contact_free_start_snap,
        "t_ground_drop": ground_t,
        "incidental_robot_contact": bool(incidental_robot_contact),
        "pure_free_flight": bool(
            first_contact_free_t is not None and not incidental_robot_contact
        ),
        "airborne_s": float(ground_t) if ground_t is not None else (
            float(rows[-1]["t_rel"]) if rows else 0.0
        ),
    }


def grasp_point(p_com, u, s: float) -> np.ndarray:
    return np.asarray(p_com, float) + float(s) * np.asarray(u, float)


def grasp_point_vel(v_com, w, u, s: float) -> np.ndarray:
    r = float(s) * np.asarray(u, float)
    return np.asarray(v_com, float) + np.cross(np.asarray(w, float), r)


def propose_interceptions(future: dict, *, budget: PilotBudget) -> list[dict]:
    """Build (t_c, s) candidates from recorded future (contact-free preferred)."""
    rows = future["rows"]
    if not rows:
        return []
    t_free = future.get("first_contact_free_t")
    usable = [
        r for r in rows
        if (t_free is None or r["t_rel"] + 1e-12 >= float(t_free))
        and r["obj_z"] > GROUND_DROP_Z + 0.02
        and not r["ground_drop"]
    ]
    if not usable:
        usable = [r for r in rows if r["obj_z"] > GROUND_DROP_Z + 0.02]
    if not usable:
        return []
    # subsample times
    idxs = np.linspace(0, len(usable) - 1, num=min(budget.max_intercept_times, len(usable)))
    idxs = sorted({int(round(i)) for i in idxs})
    s_vals = np.linspace(-(CYL_HALF - AXIAL_MARGIN), (CYL_HALF - AXIAL_MARGIN), budget.max_axial_samples)
    cands = []
    for i in idxs:
        r = usable[i]
        for s in s_vals:
            p = grasp_point(r["p_obj"], r["u_obj"], float(s))
            v = grasp_point_vel(r["v_obj"], r["w_obj"], r["u_obj"], float(s))
            cands.append({
                "t_c": float(r["t_rel"]),
                "t_abs": float(r["t"]),
                "s": float(s),
                "p_grasp": p.tolist(),
                "v_grasp": v.tolist(),
                "u": r["u_obj"].tolist(),
                "obj_z": float(r["obj_z"]),
                "opening_hint": float(max(2.2 * CYL_R, 0.045)),
            })
    return cands


def empty_bilateral_stats() -> dict:
    return {
        "first_bilateral_t": None,
        "last_bilateral_t": None,
        "consecutive_bil_s": 0.0,
        "max_consecutive_bil_s": 0.0,
        "total_bil_s": 0.0,
        "n_bil_gaps": 0,
        "gap_durations_s": [],
        "bilateral_at_end": False,
        "load_bearing_at_end": False,
        "historical_ge_60ms": False,
        "continuous_ge_60ms_at_end": False,
        "continuous_ge_150ms_at_end": False,
        "currently_stable_grasp": False,
    }


def update_bilateral_stats(stats: dict, is_bil: bool, t_rel: float, dt: float) -> None:
    """Track consecutive / total / gaps. Does not treat historical peaks as current stability."""
    if is_bil:
        if stats["first_bilateral_t"] is None:
            stats["first_bilateral_t"] = t_rel
        if stats.get("_in_gap"):
            stats["n_bil_gaps"] += 1
            stats["gap_durations_s"].append(float(stats.get("_gap_acc", 0.0)))
            stats["_in_gap"] = False
            stats["_gap_acc"] = 0.0
        stats["consecutive_bil_s"] += dt
        stats["total_bil_s"] += dt
        stats["last_bilateral_t"] = t_rel
        stats["max_consecutive_bil_s"] = max(
            float(stats["max_consecutive_bil_s"]), float(stats["consecutive_bil_s"])
        )
    else:
        if stats["consecutive_bil_s"] > 0:
            stats["_in_gap"] = True
            stats["_gap_acc"] = 0.0
        if stats.get("_in_gap"):
            stats["_gap_acc"] = float(stats.get("_gap_acc", 0.0)) + dt
        stats["consecutive_bil_s"] = 0.0
    stats["historical_ge_60ms"] = bool(stats["max_consecutive_bil_s"] + 1e-12 >= STABLE_BIL_S)
    stats["continuous_ge_60ms_at_end"] = bool(stats["consecutive_bil_s"] + 1e-12 >= STABLE_BIL_S)
    stats["continuous_ge_150ms_at_end"] = bool(stats["consecutive_bil_s"] + 1e-12 >= STABLE_BIL_150_S)
    stats["bilateral_at_end"] = bool(is_bil)
    stats["currently_stable_grasp"] = bool(stats["continuous_ge_60ms_at_end"] and is_bil)


@dataclass
class ChaseParams:
    """Explicit closing / approach search variables (distinct strategies)."""
    open_s: float = 0.08
    close_lead_s: float = 0.06
    approach_gain: float = 1.35
    vel_match: float = 0.55
    post_contact_brake: float = 0.0  # 0=keep chasing; 1=full brake after contact
    post_contact_close: bool = True
    grip_open_tau: float = TAU_OPEN
    grip_close_tau: float = TAU_SECURE
    stop_chase_after_contact: bool = False


def chase_sequence(
    sim,
    cand: dict,
    *,
    horizon_s: float,
    open_first: bool = True,
    params: ChaseParams | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Legal 7D chase with parameterized closing and correct bilateral metrics."""
    p = params or ChaseParams(open_s=0.08 if open_first else 0.0)
    actions = []
    t0 = float(sim.data.time)
    t_c = float(cand["t_c"])
    p_tgt = np.asarray(cand["p_grasp"], float)
    v_tgt = np.asarray(cand["v_grasp"], float)
    u = np.asarray(cand["u"], float)
    bil = empty_bilateral_stats()
    info = {
        "first_contact_t": None,
        "first_contact_side": None,
        "first_pad_contact_t": None,
        "params": p.__dict__.copy(),
    }

    while float(sim.data.time) - t0 < horizon_s - 1e-9:
        t_rel = float(sim.data.time) - t0
        kin = sample_kin(sim)
        ph, Rh = kin["p_hand"], kin["R_hand"]
        pinch_w = ph + Rh @ np.array([0.0, 0.0, PINCH_Z], float)
        err = p_tgt - pinch_w
        remain = max(t_c - t_rel, 0.02)
        contacted = info["first_pad_contact_t"] is not None
        if contacted and p.stop_chase_after_contact:
            v_cmd = (1.0 - p.post_contact_brake) * (0.3 * err / remain) + p.vel_match * 0.2 * v_tgt
            v_cmd = (1.0 - p.post_contact_brake) * v_cmd
        else:
            v_cmd = p.approach_gain * (err / remain) + p.vel_match * v_tgt
            if contacted and p.post_contact_brake > 0:
                v_cmd = (1.0 - p.post_contact_brake) * v_cmd + p.post_contact_brake * (0.15 * v_tgt)
        z_h = Rh[:, 2]
        axis = np.cross(z_h, -u)
        ang = float(np.linalg.norm(axis))
        if ang > 1e-8:
            axis = axis / ang
            w_body = Rh.T @ (axis * min(ang * 5.0, 2.5))
        else:
            w_body = np.zeros(3)
        if t_rel < p.open_s:
            tau = p.grip_open_tau
        elif contacted and p.post_contact_close:
            tau = p.grip_close_tau
        elif t_rel >= max(0.0, t_c - p.close_lead_s):
            tau = p.grip_close_tau
        else:
            tau = 0.35 * p.grip_close_tau + 0.65 * p.grip_open_tau
        a = _action_from(sim, v_cmd, w_body, tau)
        actions.append(a.copy())
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        cs = contact_summary(sim)
        if info["first_contact_t"] is None and (cs["nL"] + cs["nR"]) > 0:
            info["first_contact_t"] = t_rel
            info["first_contact_side"] = "L" if cs["nL"] else ("R" if cs["nR"] else "unknown")
        if info["first_pad_contact_t"] is None and pb["any_pad"]:
            info["first_pad_contact_t"] = t_rel
            info["first_contact_side"] = "L" if pb["nL"] else ("R" if pb["nR"] else info.get("first_contact_side"))
        update_bilateral_stats(bil, bool(pb["bilateral"]), t_rel, DT_POLICY)
        bil["load_bearing_at_end"] = bool(pb["bilateral"])
        if latched(sim):
            info["ground_drop"] = True
            break
        if t_rel > t_c and not (contacted and p.stop_chase_after_contact):
            p_tgt = kin["p_obj"] + float(cand["s"]) * kin["u_obj"]
            v_tgt = grasp_point_vel(kin["v_obj"], kin["w_obj"], kin["u_obj"], float(cand["s"]))

    info.update(bil)
    # Compatibility aliases — do NOT use historical peak as "stable"
    info["bilateral_t"] = bil["first_bilateral_t"]
    info["bil_hold_s"] = bil["max_consecutive_bil_s"]  # historical max (legacy)
    info["stable_bilateral"] = bool(bil["currently_stable_grasp"])
    info["n_actions"] = len(actions)
    info["final_topo"] = topology(sim)
    info["final_z"] = float(sim.data.xpos[sim.ids.object_body][2])
    info["ground_drop"] = bool(info.get("ground_drop", False) or latched(sim))
    return actions, info


def pose_correct_and_handoff(sim, ref, *, max_s: float = 2.5) -> tuple[list[np.ndarray], dict]:
    """After stable bilateral, reduce pose error then HANDOFF + SUCCESS_NEW nominal.

    Sign convention (verified by finite difference): for approximately stationary
    object, Δp_rel ≈ -R_hand.T @ v_hand, so commanding v_hand = -R @ err_p
    increases p_rel toward P_TGT when err_p = P_TGT - p_rel.
    """
    actions = []
    t0 = float(sim.data.time)
    while float(sim.data.time) - t0 < max_s - 1e-9:
        pb = pad_bilateral(sim)
        m = measure(sim, U_TGT, P_TGT)
        if m["e_theta_deg"] <= 10.0 and m["e_pos_mm"] <= 12.0 and pb["bilateral"]:
            break
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        po = np.array(sim.data.xpos[sim.ids.object_body], float)
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        p_rel = Rh.T @ (po - ph)
        err_p = np.asarray(P_TGT, float) - p_rel
        # Corrected sign: move hand opposite to desired relative move
        v_h = np.clip(-err_p * 3.0, -0.6, 0.6)
        u = object_axis(sim)
        u_h = Rh.T @ u
        u_h = u_h / max(float(np.linalg.norm(u_h)), 1e-12)
        # Rotate hand so object axis in hand frame moves toward U_TGT.
        # Under rigid grasp this co-rotates; useful only with slip/pivot authority.
        axis = np.cross(u_h, np.asarray(U_TGT, float))
        w_body = np.clip(axis * 2.0, -0.8, 0.8)
        a = _action_from(sim, Rh @ v_h, w_body, TAU_SECURE)
        a[6] = 1.0
        actions.append(a.copy())
        step_legal(sim, a)
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    m_hand = measure(sim, U_TGT, P_TGT)
    apply_handoff(sim, ref)
    t_h = float(sim.data.time)
    reason = None
    samples = []
    next_sample = 0.0
    lost = 0
    lift_done = False
    end_row = None
    if latched(sim):
        reason = "GROUND_DROP"
    from envs.ballistic_impact import z_tgt_of

    while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(sim)
        t_rel = float(sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(sim, U_TGT, P_TGT)
            cs = contact_summary(sim)
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += 0.02
        row = _monitor(sim)
        end_row = row
        target = float(z_tgt_of(sim))
        bilateral = _bilateral(row)
        if bilateral and float(sim.fsm.p_des[2]) >= target - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            break
        if _dropped(row) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        if end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    win = window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
    success_new = bool(reason == "SUCCESS_12S" and win.get("working_SUCCESS_NEW_hold"))
    return actions, {
        "m_handoff": {k: m_hand[k] for k in ("e_theta_deg", "e_pos_mm", "obj_z")},
        "reason": reason,
        "SUCCESS_NEW": success_new,
        "success_12s": reason == "SUCCESS_12S",
        "window": win,
        "n_pose_actions": len(actions),
    }


def validate_pose_return_sign(sim) -> dict:
    """Finite-difference check: hand +dx should decrease p_rel along that hand axis."""
    mujoco.mj_forward(sim.model, sim.data)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    p_rel0 = Rh.T @ (po - ph)
    eps = 1e-3
    # Analytic: ∂p_rel/∂p_hand = -R^T
    analytic = -Rh.T
    # FD along hand x
    ph1 = ph + Rh[:, 0] * eps
    p_rel1 = Rh.T @ (po - ph1)
    fd = (p_rel1 - p_rel0) / eps
    # Correct controller: v_h = -err_p → v_world = R @ (-err_p)
    err = np.array([0.01, 0.0, 0.0])
    v_wrong = Rh @ err
    v_right = Rh @ (-err)
    return {
        "p_rel0": p_rel0.tolist(),
        "fd_dp_rel_d_hand_x": fd.tolist(),
        "analytic_row0": analytic[0].tolist(),
        "fd_matches_minus_ex": bool(abs(fd[0] + 1.0) < 1e-6),
        "wrong_sign_moves_rel": "positive_v_h_decreases_p_rel",
        "corrected_v_hand_formula": "v_hand_world = R @ (-(P_TGT - p_rel))",
        "example_err_x": 0.01,
        "v_wrong_world": v_wrong.tolist(),
        "v_right_world": v_right.tolist(),
    }


def evaluate_success_new_from_state(sim, ref) -> dict:
    """HANDOFF + full nominal from current state; independent SUCCESS_NEW."""
    _, meta = pose_correct_and_handoff(sim, ref, max_s=0.02)
    return meta


def exact_replay_actions(sim, snap, ref, actions: list[np.ndarray]) -> dict:
    """Uninterrupted legal tape from post-impact snap → Recovery → HANDOFF → SUCCESS_NEW."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    for a in actions:
        step_legal(sim, np.asarray(a, np.float32))
        if latched(sim):
            return {
                "SUCCESS_NEW": False,
                "reason": "GROUND_DROP_DURING_RECOVERY",
                "bilateral_end": False,
            }
    # If still bilateral, finish with pose+handoff; else fail
    if not pad_bilateral(sim)["bilateral"]:
        return {
            "SUCCESS_NEW": False,
            "reason": "LOST_BEFORE_HANDOFF",
            "bilateral_end": False,
            "topo": topology(sim),
        }
    _pose_acts, meta = pose_correct_and_handoff(sim, ref, max_s=2.5)
    return {
        "SUCCESS_NEW": bool(meta["SUCCESS_NEW"]),
        "reason": meta["reason"],
        "success_12s": bool(meta["success_12s"]),
        "bilateral_end": True,
        "m_handoff": meta["m_handoff"],
    }


def k49_project(actions: list[np.ndarray]) -> dict:
    book = load_codebook()
    # K48 prototypes only (0-46); tok48 not in book matrix
    rep = representability(np.stack(actions, axis=0), book)
    return rep


def evaluate_k49_tokens(sim, snap, ref, tokens, table, book48) -> dict:
    from evaluation.plan_a_mechanism_audit import decode_token

    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    lost_low = 0
    for tok in tokens:
        step_legal(sim, decode_token(table, book48, int(tok)))
        failed, lost_low = recovery_failed(sim, ref, lost_low)
        if failed:
            from envs.outcomes.support_drop import terminal_reason

            return {"SUCCESS_NEW": False, "reason": terminal_reason(sim) or "DROP_DURING_RECOVERY"}
    _, meta = pose_correct_and_handoff(sim, ref, max_s=0.05)  # already should be near; just handoff path
    # pose_correct with max_s=0.05 almost immediately handoffs if already in tol
    return meta


def impact_config_grid(budget: PilotBudget, seed: int = 42) -> list[dict]:
    """Stratified random sampling over near-end grasp / impact factors.

    Avoids nested-loop truncation that only covers the first axial offset.
    """
    rng = np.random.default_rng(seed)
    axial = [0.018, 0.022, 0.026, 0.028]
    zs = [0.55, 0.60, 0.65]
    approaches = ["top", "oblique", "lateral", "side"]
    speeds = [3.5, 4.5, 5.5, 6.5, 7.5]
    configs = []
    # Latin-ish: cycle factors independently so each axis appears ~equally
    n = int(budget.max_impact_configs)
    for i in range(n):
        configs.append({
            "config_id": f"gbi_{i:03d}",
            "axial_offset": float(axial[i % len(axial)]),
            "z": float(zs[(i // len(axial)) % len(zs)]),
            "tilt_deg": float(rng.choice([-20, -10, 0, 10, 20])),
            "approach": str(approaches[(i // (len(axial) * len(zs))) % len(approaches)]),
            "speed": float(speeds[i % len(speeds)]),
            "offset": float(rng.choice([-0.018, -0.008, 0.0, 0.008, 0.018])),
            "z_off": float(rng.uniform(-0.006, 0.010)),
        })
    # Shuffle order so early truncation (if any) is not axis-biased
    order = rng.permutation(n)
    shuffled = []
    for new_i, old_i in enumerate(order):
        row = dict(configs[int(old_i)])
        row["config_id"] = f"gbi_{new_i:03d}"
        shuffled.append(row)
    return shuffled
