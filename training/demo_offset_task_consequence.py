"""Visualize task-level consequence of offset ICs (diagnostic, not a new eval set).

    python training/demo_offset_task_consequence.py --mode zero --offset 0.0075
    python training/demo_offset_task_consequence.py --mode heuristic --offset 0.0075
    python training/demo_offset_task_consequence.py --mode zero --offset -0.0075
    python training/demo_offset_task_consequence.py --mode heuristic --offset -0.0075
    python training/demo_offset_task_consequence.py --mode zero --case original_failure
    python training/demo_offset_task_consequence.py --mode zero --case nominal
    python training/demo_offset_task_consequence.py --mode heuristic --case original_failure
    python training/demo_offset_task_consequence.py --mode heuristic --case nominal

Does not train SAC, retune RULE, or load held-out eval npz.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from controllers.nominal import grasp_orientation
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.grasp_sim import GraspSim
from envs.physical_recovery import (
    SUCCESS_HOLD,
    Z_AIR,
    fail_kind,
    physical_pack,
    recovered,
)
from training.impact_demo_core import RENDER_HZ, RealtimePacer, _tilt_deg
from training.impact_visualization_utils import (
    HoldClocks,
    apply_viewer_camera,
    enable_viewer_flags,
)
from training.offset_construct import apply_rel_pose, ic_ok, lift_to_airborne, settle_hold
from training.replay_core import (
    G_HOLD,
    MASS,
    MU,
    disable_object_table,
    freeze,
    pack_replay,
    params_from_frozen,
    restore_replay,
    sha256_file,
    tick_vw,
    CTRL_PY,
    FROZEN_YAML,
)

LOG_DIR = ROOT / "results" / "logs" / "offset_task_consequence"
HARSH_Y = (0.24, 0.40, np.array([0.0, 0.010, 0.0]))  # calibrate_detector.py
HARSH_SET = [
    (0.22, 0.42, np.array([0.010, 0.0, 0.0])),
    (0.24, 0.40, np.array([0.0, 0.010, 0.0])),
    (0.20, 0.45, np.array([0.008, 0.008, 0.0])),
    (0.25, 0.40, np.array([0.012, 0.0, 0.003])),
]


def _snapshot(sim) -> dict:
    s = pack_replay(sim)
    s["mass"] = float(sim.mass)
    s["friction"] = float(sim.friction)
    s["grasp_offset"] = np.asarray(sim.fsm.grasp_offset, float).copy()
    return s


def _ic_fields(sim) -> dict:
    o = physical_pack(sim)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return {
        "e_x": float(o["e_x"]),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": float(o["obj_z"]),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "clear": float(o["clear"]),
        "scene": int(o["scene"]),
        "aperture": float(o["aperture"]),
        "tau": float(o["tau"]),
        "p_hand": ph.tolist(),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "r_hand": rh.tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).tolist(),
        "po": po.tolist(),
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "table_disabled_protocol": True,
    }


def construct_recovery_offset_ic(cfg, e_x: float) -> tuple:
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    if not lift_to_airborne(sim, cfg):
        raise RuntimeError("lift_to_airborne failed")
    disable_object_table(sim)
    freeze(sim)
    apply_rel_pose(sim, float(e_x))
    settle_hold(sim, cfg, 0.25, G_HOLD)
    o = physical_pack(sim)
    ok = ic_ok(o, 0.002, 0.0085)
    fields = _ic_fields(sim)
    fields["target_e_x"] = float(e_x)
    fields["ic_ok"] = bool(ok)
    fields["mass"] = MASS
    fields["friction"] = MU
    fields["story"] = "B_recovery_hand_x"
    snap = _snapshot(sim)
    return sim, snap, fields


def _row(sim, o, live_fk, phase: str) -> dict:
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    rh = np.asarray(o["rh"], float)
    Robj = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    Rhand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return {
        "t": float(sim.data.time),
        "phase": phase,
        "hand": ph.tolist(),
        "object": po.tolist(),
        "R_obj": Robj.tolist(),
        "R_hand": Rhand.tolist(),
        "e_x": float(rh[0]),
        "e_y": float(rh[1]),
        "e_z": float(rh[2]),
        "v_rel_hand": np.asarray(o["v_rel_h"], float).tolist(),
        "omega_rel_hand": np.asarray(o["w_rel_h"], float).tolist(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(o["Fn_L"]),
        "Fn_R": float(o["Fn_R"]),
        "aperture": float(o["aperture"]),
        "tau": float(o["tau"]),
        "obj_z": float(po[2]),
        "hand_z": float(ph[2]),
        "clear": float(o["clear"]),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "tilt_deg": _tilt_deg(sim),
        "recovered": bool(recovered(o)),
        "fail_kind": live_fk or "",
    }


def _overlay(viewer, mode, offset_label, phase, t, e_x) -> None:
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        viewer.add_overlay(pos, "MODE", str(mode).upper())
        if offset_label:
            viewer.add_overlay(pos, "OFFSET", str(offset_label))
        viewer.add_overlay(pos, "PHASE", str(phase).upper())
        viewer.add_overlay(pos, "t", f"{t:.3f} s")
        if e_x is not None:
            viewer.add_overlay(pos, "e_x", f"{1e3 * float(e_x):+.1f} mm")
    except Exception:
        return


def _cam(sim) -> dict:
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return {
        "lookat": po + np.array([0.0, 0.0, 0.02]),
        "distance": 0.72,
        "azimuth": 125.0,
        "elevation": -22.0,
    }


def _tac_dummy(o: dict) -> dict:
    return {
        "estimate_valid": False,
        "e_hat_x": 0.0,
        "bilateral_valid": False,
        "contact_present_L": int(o["nL"]) > 0,
        "contact_present_R": int(o["nR"]) > 0,
    }


def _maxabs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def compare_ics(a: dict, b: dict) -> dict:
    return {
        "dqpos": _maxabs(a["qpos"], b["qpos"]),
        "dqvel": _maxabs(a["qvel"], b["qvel"]),
        "d_pdes": _maxabs(a["p_des"], b["p_des"]),
        "d_rdes": _maxabs(a["r_des"], b["r_des"]),
        "d_vcmd": _maxabs(a["v_cmd"], b["v_cmd"]),
        "d_ctrl": _maxabs(a["ctrl"], b["ctrl"]),
        "d_phand": _maxabs(a["p_hand"], b["p_hand"]),
        "d_po": _maxabs(a["po"], b["po"]),
        "d_e_x": abs(float(a["e_x"]) - float(b["e_x"])),
        "nL": (int(a["nL"]), int(b["nL"])),
        "nR": (int(a["nR"]), int(b["nR"])),
    }


def run_recovery_offset_task(
    mode: str,
    e_x_target: float,
    *,
    viewer=None,
    interactive: bool = True,
    playback: float = 0.5,
    seed: int = 0,
) -> dict:
    np.random.seed(int(seed))
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    params, _ = params_from_frozen()
    v_lift = float(cfg["fsm"]["lift_speed"])
    lift_dz = float(cfg["fsm"]["lift_offset_z"])
    hold_s = 2.0
    sim, snap, ic0 = construct_recovery_offset_ic(cfg, e_x_target)
    if not ic0["ic_ok"]:
        return {
            "ok": False,
            "reason": "constructed IC failed ic_ok (not a valid settled offset)",
            "ic": {k: v for k, v in ic0.items() if k not in ("qpos", "qvel", "ctrl", "r_hand", "r_des")},
        }
    restore_replay(sim, snap)
    ic_check = _ic_fields(sim)
    sim_twin = GraspSim(cfg)
    sim_twin.reset(MASS, MU, np.zeros(3))
    disable_object_table(sim_twin)
    restore_replay(sim_twin, snap)
    ic_twin = _ic_fields(sim_twin)
    ic_match_modes = compare_ics(ic_check, ic_twin)
    dt = float(sim.model.opt.timestep)
    clocks = HoldClocks(dt=dt)
    ctrl = RuleBasedRecovery(params)
    ctrl.reset(grasp_orientation())
    phase = "recovery" if mode == "heuristic" else "continue_lift"
    if mode == "zero":
        ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
        sim.fsm.p_des = ph.copy()
        sim.fsm.v_cmd = np.array([0.0, 0.0, v_lift])
        sim.fsm.w_cmd = np.zeros(3)
    start_lift = {
        "t": float(sim.data.time),
        "p_hand": np.array(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "r_hand": np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "tau": float(G_HOLD),
        "z_hand": float(sim.data.xpos[sim.ids.hand_body][2]),
    }
    t_recov_ok = None
    t_continue = None if mode == "heuristic" else float(sim.data.time)
    t_lift_done = None
    t_hold_start = None
    t_drop = None
    t_uni = None
    t_both = None
    p_des_z0 = float(sim.fsm.p_des[2]) if mode == "zero" else None
    log = []
    viewer_cm = None
    if interactive and viewer is None:
        import mujoco.viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(sim.model, sim.data)
        viewer = viewer_cm.__enter__()
    pacer = RealtimePacer(playback, RENDER_HZ, dt) if viewer is not None else None
    if viewer is not None:
        apply_viewer_camera(viewer, _cam(sim))
        enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
        if hasattr(viewer, "sync"):
            viewer.sync()
        if pacer is not None:
            pacer.start(float(sim.data.time))
            pacer.note_sync(float(sim.data.time))

    n_max = int(round((4.0 + lift_dz / v_lift + hold_s + 2.0) / dt))
    lift_v = np.array([0.0, 0.0, v_lift])
    for _ in range(n_max):
        t = float(sim.data.time)
        if phase == "recovery":
            o_rule = physical_pack(sim)
            cmd = ctrl.step(o_rule, dt)
            if cmd.get("freeze_p"):
                freeze(sim)
            if cmd.get("r_des") is not None:
                sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
            tick_vw(sim, cmd["v_world"], np.zeros(3), cmd["tau"], gains)
        elif phase == "continue_lift":
            tick_vw(sim, lift_v, np.zeros(3), G_HOLD, gains)
        else:
            tick_vw(sim, np.zeros(3), np.zeros(3), G_HOLD, gains)

        o = physical_pack(sim)
        clock = clocks.update(o, _tac_dummy(o))
        live_fk = fail_kind(o, clocks.lost_gt)
        nL, nR = int(o["nL"]), int(o["nR"])
        if t_uni is None and ((nL == 0) ^ (nR == 0)):
            t_uni = t
        if t_both is None and nL == 0 and nR == 0:
            t_both = t
        if t_drop is None and live_fk == "drop":
            t_drop = t
        if (not log) or t - log[-1]["t"] >= 0.0095:
            log.append(_row(sim, o, live_fk, phase))

        if phase == "recovery" and clock.get("success_GT"):
            t_recov_ok = t
            ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
            sim.fsm.p_des = ph.copy()
            sim.fsm.v_cmd = lift_v.copy()
            p_des_z0 = float(sim.fsm.p_des[2])
            t_continue = t
            start_lift = {
                "t": t,
                "p_hand": ph.tolist(),
                "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
                "r_hand": np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).tolist(),
                "r_des": np.asarray(sim.fsm.r_des, float).tolist(),
                "v_cmd": lift_v.tolist(),
                "tau": float(G_HOLD),
                "z_hand": float(ph[2]),
            }
            phase = "continue_lift"
        if phase == "continue_lift" and p_des_z0 is not None:
            if t_lift_done is None and float(sim.fsm.p_des[2]) - p_des_z0 >= lift_dz - 1e-9:
                t_lift_done = t
                t_hold_start = t
                phase = "hold"
        t = float(sim.data.time)
        if viewer is not None and pacer is not None and pacer.should_render(t):
            _overlay(viewer, mode, f"{1e3 * e_x_target:+.1f} mm hand-x", phase, t, o["e_x"])
            viewer.sync()
            pacer.note_sync(t)
            pacer.wait_if_ahead(t)
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
        elif pacer is not None:
            pacer.wait_if_ahead(t)

        if t_drop is not None and t >= t_drop + 1.5:
            break
        if t_hold_start is not None and t >= t_hold_start + hold_s:
            break
        if mode == "heuristic" and t_continue is None and t - float(snap["time"]) > 6.0:
            break

    oF = physical_pack(sim)
    last = log[-1] if log else {}
    captured = int(oF["nL"]) > 0 and int(oF["nR"]) > 0
    out = {
        "ok": True,
        "story": "B_recovery_hand_x",
        "mode": mode,
        "target_e_x": e_x_target,
        "ic": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ic0.items() if k not in ("qpos", "qvel", "ctrl")},
        "ic_check": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ic_check.items() if k not in ("qpos", "qvel", "ctrl")},
        "ic_match_restore": compare_ics(ic0, ic_check),
        "ic_match_zero_vs_heuristic_before_divergence": ic_match_modes,
        "nominal_lift": {
            "v": [0.0, 0.0, v_lift],
            "lift_offset_z": lift_dz,
            "semantics": "constant world-z velocity then 2.0 s hold; p_des:=p_hand once at resume",
        },
        "start_lift": start_lift,
        "t_continue": t_continue,
        "t_recovered": t_recov_ok,
        "t_lift_done": t_lift_done,
        "t_uni": t_uni,
        "t_both": t_both,
        "t_drop": t_drop,
        "lift_completed": bool(t_lift_done is not None and captured and t_drop is None),
        "drop": t_drop is not None,
        "unilateral": t_uni is not None,
        "both_lost": t_both is not None,
        "max_tilt_deg": max((row["tilt_deg"] for row in log), default=0.0),
        "max_v_rel": max((float(np.linalg.norm(row["v_rel_hand"])) for row in log), default=0.0),
        "final_e_x": float(oF["e_x"]),
        "final_obj_z": float(oF["obj_z"]),
        "final_nL": int(oF["nL"]),
        "final_nR": int(oF["nR"]),
        "recovered_end": bool(recovered(oF)),
        "fail_kind_end": fail_kind(oF, clocks.lost_gt) or "",
        "hashes": {"yaml": sha256_file(FROZEN_YAML), "py": sha256_file(CTRL_PY)},
        "used_final_eval_data": False,
        "log": log,
    }
    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    return out


def run_from_t0(
    mode: str,
    mass: float,
    mu: float,
    grasp_offset,
    *,
    label: str,
    viewer=None,
    interactive: bool = True,
    playback: float = 0.5,
) -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    params, _ = params_from_frozen()
    sim = GraspSim(cfg)
    off = np.asarray(grasp_offset, float)
    sim.reset(mass, mu, off)
    dt = float(sim.model.opt.timestep)
    clocks = HoldClocks(dt=dt)
    ctrl = RuleBasedRecovery(params)
    switched = False
    phase = "grasp"
    t_uni = t_both = t_drop = t_instab = None
    t_air = None
    t_recov_ok = t_continue = t_lift_done = t_hold_start = None
    p_des_z0 = None
    lift_goal_z = None
    if np.linalg.norm(off) > 1e-9:
        off_label = f"grasp-offset [{off[0]:.3f},{off[1]:.3f},{off[2]:.3f}] m"
    else:
        off_label = "centered 0 mm"
    v_lift = float(cfg["fsm"]["lift_speed"])
    lift_dz = float(cfg["fsm"]["lift_offset_z"])
    lift_v = np.array([0.0, 0.0, v_lift])
    log = []
    viewer_cm = None
    if interactive and viewer is None:
        import mujoco.viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(sim.model, sim.data)
        viewer = viewer_cm.__enter__()
    pacer = RealtimePacer(playback, RENDER_HZ, dt) if viewer is not None else None
    if viewer is not None:
        apply_viewer_camera(viewer, _cam(sim))
        enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
        if hasattr(viewer, "sync"):
            viewer.sync()
        if pacer is not None:
            pacer.start(0.0)
            pacer.note_sync(0.0)

    n_max = int(round(14.0 / dt))
    for _ in range(n_max):
        t = float(sim.data.time)
        if phase == "grasp":
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            if sim.fsm.phase == "lift" and sim.captured:
                if lift_goal_z is None and sim.fsm.grasp_z is not None:
                    lift_goal_z = float(sim.fsm.grasp_z) + lift_dz
                z = float(sim.data.xpos[sim.ids.object_body][2])
                if z >= Z_AIR and t_air is None:
                    t_air = t
                if mode == "heuristic" and not switched and t_air is not None:
                    freeze(sim)
                    ctrl.reset(grasp_orientation())
                    switched = True
                    phase = "recovery"
        elif phase == "recovery":
            o_rule = physical_pack(sim)
            cmd = ctrl.step(o_rule, dt)
            if cmd.get("freeze_p"):
                freeze(sim)
            if cmd.get("r_des") is not None:
                sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
            tick_vw(sim, cmd["v_world"], np.zeros(3), cmd["tau"], gains)
        elif phase == "continue_lift":
            tick_vw(sim, lift_v, np.zeros(3), G_HOLD, gains)
        else:
            tick_vw(sim, np.zeros(3), np.zeros(3), G_HOLD, gains)

        o = physical_pack(sim)
        clock = clocks.update(o, _tac_dummy(o))
        live_fk = fail_kind(o, clocks.lost_gt)
        nL, nR = int(o["nL"]), int(o["nR"])
        # fail_kind() drop/scene is an airborne-recovery criterion. On the table
        # during approach the cylinder is supposed to contact the table.
        taskish = phase in ("recovery", "continue_lift", "hold") or str(sim.fsm.phase) in (
            "close",
            "lift",
        )
        if t_uni is None and taskish and ((nL == 0) ^ (nR == 0)):
            t_uni = t
        if t_both is None and taskish and nL == 0 and nR == 0:
            t_both = t
        airborne_drop = False
        if phase in ("recovery", "continue_lift", "hold"):
            airborne_drop = live_fk == "drop"
        elif str(sim.fsm.phase) == "lift":
            airborne_drop = bool(sim.dropped())
        if t_drop is None and airborne_drop:
            t_drop = t
        if t_instab is None and sim.fsm.phase == "lift" and (
            abs(float(o["e_x"])) > 0.005 or _tilt_deg(sim) > 25.0 or float(o["v_rel"]) > 0.05
        ):
            t_instab = t
        if (not log) or t - log[-1]["t"] >= 0.0095:
            log.append(_row(sim, o, live_fk, phase if phase != "grasp" else str(sim.fsm.phase)))

        if phase == "recovery" and clock.get("success_GT"):
            t_recov_ok = t
            ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
            sim.fsm.p_des = ph.copy()
            p_des_z0 = float(ph[2])
            t_continue = t
            phase = "continue_lift"
        if phase == "continue_lift":
            goal = lift_goal_z if lift_goal_z is not None else (
                (p_des_z0 + lift_dz) if p_des_z0 is not None else None
            )
            if t_lift_done is None and goal is not None and float(sim.fsm.p_des[2]) >= goal - 1e-9:
                t_lift_done = t
                t_hold_start = t
                phase = "hold"
        if mode == "zero" and sim.fsm.success and t_hold_start is None:
            t_lift_done = t
            t_hold_start = t
            phase = "hold"

        t = float(sim.data.time)
        if viewer is not None and pacer is not None and pacer.should_render(t):
            _overlay(
                viewer,
                mode,
                off_label,
                phase if phase != "grasp" else sim.fsm.phase,
                t,
                o["e_x"],
            )
            viewer.sync()
            pacer.note_sync(t)
            pacer.wait_if_ahead(t)
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
        elif pacer is not None:
            pacer.wait_if_ahead(t)

        if t_drop is not None and t >= t_drop + 1.5:
            break
        if t_hold_start is not None and t >= t_hold_start + 2.0:
            break
        if t >= 12.0:
            break

    oF = physical_pack(sim)
    captured = int(oF["nL"]) > 0 and int(oF["nR"]) > 0
    out = {
        "ok": True,
        "story": "A_task_grasp_offset" if np.linalg.norm(off) > 1e-9 else "nominal_centered",
        "label": label,
        "mode": mode,
        "mass": mass,
        "friction": mu,
        "grasp_offset": off.tolist(),
        "lift_goal_z": lift_goal_z,
        "t_air": t_air,
        "t_instab": t_instab,
        "t_uni": t_uni,
        "t_both": t_both,
        "t_drop": t_drop,
        "t_recovered": t_recov_ok,
        "t_continue": t_continue,
        "t_lift_done": t_lift_done,
        "fsm_success": bool(sim.fsm.success),
        "lift_completed": bool((sim.fsm.success or t_lift_done is not None) and captured and t_drop is None),
        "drop": t_drop is not None,
        "unilateral": t_uni is not None,
        "both_lost": t_both is not None,
        "max_tilt_deg": max((row["tilt_deg"] for row in log), default=0.0),
        "max_v_rel": max((float(np.linalg.norm(row["v_rel_hand"])) for row in log), default=0.0),
        "final_e_x": float(oF["e_x"]),
        "final_obj_z": float(oF["obj_z"]),
        "final_nL": int(oF["nL"]),
        "final_nR": int(oF["nR"]),
        "recovered_end": bool(recovered(oF)),
        "fail_kind_end": fail_kind(oF, clocks.lost_gt) or "",
        "hashes": {"yaml": sha256_file(FROZEN_YAML), "py": sha256_file(CTRL_PY)},
        "used_final_eval_data": False,
        "log": log,
    }
    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    return out


def probe_harsh(cfg) -> list[dict]:
    rows = []
    for m, mu, off in HARSH_SET:
        r = run_from_t0("zero", m, mu, off, label="probe", interactive=False, playback=1.0)
        rows.append({k: r[k] for k in r if k != "log"})
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Offset task-consequence viewer")
    p.add_argument("--mode", required=True, choices=("zero", "heuristic"))
    p.add_argument("--offset", type=float, default=None, help="hand-x e_x in metres, e.g. 0.0075")
    p.add_argument("--case", choices=("recovery_offset", "original_failure", "nominal"), default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--playback", type=float, default=0.5)
    p.add_argument("--probe-harsh", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    if args.probe_harsh:
        rows = probe_harsh(cfg)
        path = LOG_DIR / "harsh_probe.json"
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(json.dumps(rows, indent=2, default=str))
        print("wrote", path)
        return 0
    case = args.case
    if case is None:
        case = "recovery_offset" if args.offset is not None else None
    if case is None:
        print("provide --offset or --case")
        return 2
    if case == "recovery_offset":
        if args.offset is None:
            print("--offset required for recovery_offset")
            return 2
        r = run_recovery_offset_task(
            args.mode,
            float(args.offset),
            interactive=not args.headless,
            playback=float(args.playback),
        )
        tag = f"{args.mode}_ex_{args.offset:+.4f}".replace("+", "p").replace("-", "m")
    elif case == "original_failure":
        m, mu, off = HARSH_Y
        r = run_from_t0(
            args.mode,
            m,
            mu,
            off,
            label="original_failure",
            interactive=not args.headless,
            playback=float(args.playback),
        )
        tag = f"{args.mode}_original_failure"
    else:
        m, mu, _ = HARSH_Y
        r = run_from_t0(
            args.mode,
            m,
            mu,
            np.zeros(3),
            label="nominal",
            interactive=not args.headless,
            playback=float(args.playback),
        )
        tag = f"{args.mode}_nominal"
    slim = {k: v for k, v in r.items() if k != "log"}
    out = LOG_DIR / f"{tag}.json"
    out.write_text(json.dumps({"summary": slim, "log": r.get("log", [])}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(slim, indent=2, default=str))
    print("log", out)
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
