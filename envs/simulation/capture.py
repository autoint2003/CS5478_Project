"""Snapshot and restore one closed-loop recovery environment.

The blob stores MuJoCo state, the nominal FSM, the geometry switch, the
table-drop latch, and the policy observation window. Restoring it does not
change the controller.
"""

from __future__ import annotations

import mujoco
import numpy as np

from envs.simulation.support_contacts import restore_supports, supports_of

STATE_SPEC = None


def _state_spec():
    names = (
        "mjSTATE_TIME",
        "mjSTATE_QPOS",
        "mjSTATE_QVEL",
        "mjSTATE_ACT",
        "mjSTATE_WARMSTART",
        "mjSTATE_CTRL",
        "mjSTATE_MOCAP_POS",
        "mjSTATE_MOCAP_QUAT",
        "mjSTATE_USERDATA",
        "mjSTATE_PLUGIN",
    )
    spec = 0
    for name in names:
        if hasattr(mujoco.mjtState, name):
            spec |= int(getattr(mujoco.mjtState, name))
    if spec == 0:
        spec = int(mujoco.mjtState.mjSTATE_PHYSICS)
    return spec


def _physics_blob(sim):
    global STATE_SPEC
    if STATE_SPEC is None:
        STATE_SPEC = _state_spec()
    n = int(mujoco.mj_stateSize(sim.model, STATE_SPEC))
    state = np.empty(n, np.float64)
    mujoco.mj_getState(sim.model, sim.data, state, STATE_SPEC)
    blob = {
        "state": state,
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "time": float(sim.data.time),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, float).copy(),
        "captured": bool(sim.captured),
    }
    if int(sim.model.na):
        blob["act"] = np.array(sim.data.act, float).copy()
    if hasattr(sim, "guide_eq"):
        blob["eq_data"] = int(sim.data.eq_active[sim.guide_eq]) if hasattr(sim.data, "eq_active") else None
        blob["eq_model"] = int(sim.model.eq_active0[sim.guide_eq])
        blob["mocap_pos"] = np.array(sim.data.mocap_pos[sim.guide_mocap], float).copy()
        blob["mocap_quat"] = np.array(sim.data.mocap_quat[sim.guide_mocap], float).copy()
    return blob


def _fsm_blob(fsm):
    arrays = {}
    for name in ("p_des", "r_des", "v_des", "w_des", "v_cmd", "w_cmd", "grasp_xy", "grasp_offset"):
        value = getattr(fsm, name)
        arrays[name] = None if value is None else np.array(value, float).copy()
    return {
        "phase": str(fsm.phase),
        "t_phase": float(fsm.t_phase),
        "t_stable": float(fsm.t_stable),
        "t_held": float(fsm.t_held),
        "lift_started": bool(fsm.lift_started),
        "success": bool(fsm.success),
        "grasp_z": float(fsm.grasp_z),
        "fg_cmd": float(fsm.fg_cmd),
        "arrays": arrays,
    }


def _switch_blob(switch):
    return {
        "p0": None if switch.p0 is None else np.array(switch.p0, float).copy(),
        "r0": None if switch.r0 is None else np.array(switch.r0, float).copy(),
        "pos_run": int(switch.pos_run),
        "exit_run": int(switch.exit_run),
        "entered": bool(switch.entered),
        "exited": bool(switch.exited),
        "t_entry": None if switch.t_entry is None else float(switch.t_entry),
        "t_exit": None if switch.t_exit is None else float(switch.t_exit),
        "entry_reason": switch.entry_reason,
    }


def take_capture(env):
    from envs.outcomes.table_drop import export_state as export_table_drop

    hist = env._hist
    return {
        "physics": _physics_blob(env.sim),
        "fsm": _fsm_blob(env.sim.fsm),
        "switch": _switch_blob(env.switch),
        "supports": supports_of(env.sim),
        "hist_prev_e": None if hist is None else float(hist.prev_e),
        "hist_prev_valid": False if hist is None else bool(hist.prev_valid),
        "obs_seq": [np.asarray(x, np.float32).copy() for x in env._obs_seq],
        "prev_seq": [np.asarray(x, np.float32).copy() for x in env._prev_seq],
        "prev": np.asarray(env._prev, np.float32).copy(),
        "table_drop": export_table_drop(env.sim),
    }


def restore_capture(env, cap):
    global STATE_SPEC
    from envs.observations.observable_obs import ObservableObsState
    from models.airborne_v3 import DT_POLICY

    if STATE_SPEC is None:
        STATE_SPEC = _state_spec()
    sim = env.sim
    physics = cap["physics"]
    mujoco.mj_setState(sim.model, sim.data, physics["state"], STATE_SPEC)
    sim.data.qpos[:] = physics["qpos"]
    sim.data.qvel[:] = physics["qvel"]
    sim.data.ctrl[:] = physics["ctrl"]
    sim.data.time = physics["time"]
    sim.data.qacc_warmstart[:] = physics["qacc_warmstart"]
    if "act" in physics and int(sim.model.na):
        sim.data.act[:] = physics["act"]
    if "eq_model" in physics:
        sim.model.eq_active0[sim.guide_eq] = physics["eq_model"]
        if physics["eq_data"] is not None and hasattr(sim.data, "eq_active"):
            sim.data.eq_active[sim.guide_eq] = physics["eq_data"]
        sim.data.mocap_pos[sim.guide_mocap] = physics["mocap_pos"]
        sim.data.mocap_quat[sim.guide_mocap] = physics["mocap_quat"]
    restore_supports(sim, cap["supports"])
    fsm = sim.fsm
    blob = cap["fsm"]
    fsm.phase = blob["phase"]
    fsm.t_phase = blob["t_phase"]
    fsm.t_stable = blob["t_stable"]
    fsm.t_held = blob["t_held"]
    fsm.lift_started = blob["lift_started"]
    fsm.success = blob["success"]
    fsm.grasp_z = blob["grasp_z"]
    fsm.fg_cmd = blob["fg_cmd"]
    for name, value in blob["arrays"].items():
        setattr(fsm, name, None if value is None else np.array(value, float).copy())
    sw = env.switch
    sb = cap["switch"]
    sw.p0 = None if sb["p0"] is None else np.array(sb["p0"], float).copy()
    sw.r0 = None if sb["r0"] is None else np.array(sb["r0"], float).copy()
    sw.pos_run = sb["pos_run"]
    sw.exit_run = sb["exit_run"]
    sw.entered = sb["entered"]
    sw.exited = sb["exited"]
    sw.t_entry = sb["t_entry"]
    sw.t_exit = sb["t_exit"]
    sw.entry_reason = sb["entry_reason"]
    env._hist = ObservableObsState(DT_POLICY)
    if cap["hist_prev_e"] is None or not cap["hist_prev_valid"]:
        env._hist.prev_e = float("nan")
        env._hist.prev_valid = False
    else:
        env._hist.prev_e = float(cap["hist_prev_e"])
        env._hist.prev_valid = True
    env._obs_seq = [x.copy() for x in cap["obs_seq"]]
    env._prev_seq = [x.copy() for x in cap["prev_seq"]]
    env._prev = cap["prev"].copy()
    sim.captured = physics["captured"]
    mujoco.mj_forward(sim.model, sim.data)
    from envs.outcomes.table_drop import restore_state
    restore_state(sim, cap.get("table_drop"))
    err = float(np.max(np.abs(np.array(sim.data.qpos, float) - physics["qpos"])))
    if err > 1e-6:
        raise RuntimeError(f"qpos restore error {err}")
