"""Gripper-channel test and table-free free-fall capture diagnostic.

The table geom is disabled only in a separately loaded model, before the
rollout, and only for trials tagged TABLE_FREE_DIAGNOSTIC. The official
scene, TABLE_DROP rule, and SUCCESS_12S rule are unchanged. Nothing here
is written to an SFT manifest. Cylinder and arm poses are written only as
the initial condition of a trial, never during a rollout.
"""

from __future__ import annotations


import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from controllers.recovery.action_map import AIRBORNE_V3, grip_tau, map_action
from experiments.legacy.demo.demonstrations.demo_ballistic_recovery_comparison import caption
from experiments.legacy.demo.demonstrations.demo_grav_reposition_comparison import write_frames_mp4
from envs.simulation.grasp_quality import body_twist
BALLISTIC = ROOT / "assets" / "scene_ballistic_impact.xml"
from envs.simulation.mujoco_ids import resolve_ids
from data_generation.strategies.capture_geometry import ENTRY_GAP, PINCH_HAND, capture_geometry, lead_action

from visualization.impact_view import mjv_camera_from_preset
from experiments.ball_impact.privileged_ball_impact_search import representability, step_legal
from envs.outcomes.table_drop import latched
from training.unified_k48 import load_trained_k48

OUT = ROOT / "results" / "diagnostics" / "freefall_capture"
SNAP = ROOT / "data" / "snapshots" / "train" / "ball_train_v2_0051.pkl"
CODEBOOK = ROOT / "results/archive/demo_results" / "diagnostics" / "raw" / "legal_recovery_v2" / "codebook_selected_k47.npy"


def _hold(grip: float) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip], np.float32)


def _write_object(sim, pos, vel, quat=None):
    adr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    dof = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    sim.data.qpos[adr : adr + 3] = np.asarray(pos, float)
    if quat is not None:
        sim.data.qpos[adr + 3 : adr + 7] = np.asarray(quat, float)
    sim.data.qvel[dof : dof + 6] = 0.0
    sim.data.qvel[dof : dof + 3] = np.asarray(vel, float)
    mujoco.mj_forward(sim.model, sim.data)


def _row(sim, action):
    geo = capture_geometry(sim)
    hand_v, _ = body_twist(sim.model, sim.data, sim.ids.hand_body)
    obj_v, _ = body_twist(sim.model, sim.data, sim.ids.object_body)
    mapped = map_action(action, sim.fsm.r_des, sim.data.xmat[sim.ids.hand_body], AIRBORNE_V3)
    return {
        "t": round(float(sim.data.time), 4),
        "grip_action": round(float(action[6]), 4),
        "tau_command_nm": round(float(mapped["tau"]), 4),
        "actuator_ctrl": round(float(sim.data.ctrl[7]), 4),
        "finger_qpos_m": [round(v, 5) for v in geo["finger_qpos_m"]],
        "inner_gap_m": round(float(geo["inner_gap_m"]), 5),
        "gap_fits_36mm": bool(geo["gap_fits_36mm"]),
        "between": bool(geo["between"]),
        "ready_to_close": bool(geo["ready_to_close"]),
        "would_eject_if_closed": bool(geo["would_eject_if_closed"]),
        "center_hand_m": [round(v, 5) for v in geo["center_hand_m"]],
        "clearance_m": [round(float(geo["clearance_positive_m"]), 5), round(float(geo["clearance_negative_m"]), 5)],
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "hand_v_mps": [round(float(v), 4) for v in hand_v],
        "obj_v_mps": [round(float(v), 4) for v in obj_v],
        "obj_z": round(float(geo["obj_z"]), 4),
        "table_drop": bool(latched(sim)),
    }


def _camera(sim):
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    return mjv_camera_from_preset({
        "lookat": hand,
        "distance": 0.32,
        "azimuth": 180.0,
        "elevation": -8.0,
    })


class Film:
    def __init__(self, sim, renderer, cam):
        self.sim = sim
        self.renderer = renderer
        self.cam = cam
        self.frames = []
        self.next = float(sim.data.time)

    def snap(self, text: str, force: bool = False):
        t = float(self.sim.data.time)
        if not force and t + 1e-9 < self.next:
            return
        hand = np.array(self.sim.data.xpos[self.sim.ids.hand_body], float)
        self.cam.lookat[:] = hand
        self.renderer.update_scene(self.sim.data, self.cam)
        self.frames.append(caption(self.renderer.render().copy(), text))
        self.next = t + 0.04


def _steps(sim, film, action, count, label, rows):
    for i in range(count):
        step_legal(sim, action)
        row = _row(sim, action)
        row["phase"] = label
        row["step"] = i
        rows.append(row)
        gap = row["inner_gap_m"]
        film.snap(
            f"{label} a={action[6]:+.1f} gap={gap*1000:.1f}mm n={row['nL']}/{row['nR']} t={row['t']:.2f}"
        )


def gripper_probe(env, blob):
    sim = env.sim
    restore_capture(env, blob["capture"])
    renderer = mujoco.Renderer(sim.model, 480, 640)
    film = Film(sim, renderer, _camera(sim))
    rows = []
    grasped = _row(sim, _hold(0.0))
    grasped["phase"] = "grasped_before_command"
    rows.append(grasped)
    film.snap("grasped before command", force=True)
    t_grasp_open = float(sim.data.time)
    _steps(sim, film, _hold(-1.0), 20, "open_while_grasped", rows)

    restore_capture(env, blob["capture"])
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    _write_object(sim, hand + np.array([0.0, 0.0, 0.75]), np.zeros(3))
    for _ in range(12):
        step_legal(sim, _hold(-1.0))
    parked = _row(sim, _hold(0.0))
    parked["phase"] = "empty_parked"
    rows.append(parked)
    t_mod = float(sim.data.time)
    for i in range(30):
        qmin = min(capture_geometry(sim)["finger_qpos_m"])
        grip = -1.0 if qmin < 0.0 else (-0.8 if qmin < 0.025 else -0.72)
        step_legal(sim, _hold(grip))
        row = _row(sim, _hold(grip))
        row["phase"] = "empty_moderate_close"
        row["step"] = i
        rows.append(row)
        film.snap(f"moderate close a={grip:+.2f} gap={row['inner_gap_m']*1000:.1f}mm q={qmin:.3f}")
    t_reopen = float(sim.data.time)
    _steps(sim, film, _hold(-1.0), 15, "empty_reopen", rows)
    _steps(sim, film, _hold(1.0), 8, "empty_secure_close", rows)
    _steps(sim, film, _hold(-1.0), 12, "open_after_secure_close", rows)
    video = write_frames_mp4(film.frames, OUT / "videos" / "gripper_open_close.mp4")
    renderer.close()

    def phase_rows(name):
        return [r for r in rows if r["phase"] == name]

    grasp_rows = phase_rows("open_while_grasped")
    moderate = phase_rows("empty_moderate_close")
    reopen = phase_rows("empty_reopen")
    locked = phase_rows("open_after_secure_close")

    def elapsed(group, origin, predicate):
        for row in group:
            if predicate(row):
                return round(row["t"] - origin, 4)
        return None

    summary = {
        "open_action": -1.0,
        "open_tau_nm": grip_tau(-1.0),
        "close_action": 1.0,
        "close_tau_nm": grip_tau(1.0),
        "moderate_close_action": -0.62,
        "moderate_close_tau_nm": grip_tau(-0.62),
        "zero_torque_action": -0.8,
        "mapping": "action -1 -> +2 Nm opens; action +1 -> -18 Nm closes; zero torque is action -0.8",
        "grasped_initial_gap_m": grasped["inner_gap_m"],
        "grasped_initial_qpos_m": grasped["finger_qpos_m"],
        "grasped_initial_contacts": [grasped["nL"], grasped["nR"]],
        "grasped_open_final_gap_m": grasp_rows[-1]["inner_gap_m"],
        "grasped_open_final_qpos_m": grasp_rows[-1]["finger_qpos_m"],
        "grasped_open_final_contacts": [grasp_rows[-1]["nL"], grasp_rows[-1]["nR"]],
        "seconds_grasped_to_36mm": elapsed(grasp_rows, t_grasp_open, lambda r: r["inner_gap_m"] >= 0.036),
        "seconds_grasped_to_entry_gap": elapsed(grasp_rows, t_grasp_open, lambda r: r["inner_gap_m"] >= ENTRY_GAP),
        "moderate_close_final_qpos_m": moderate[-1]["finger_qpos_m"],
        "moderate_close_final_gap_m": moderate[-1]["inner_gap_m"],
        "seconds_moderate_close_under_20mm": elapsed(moderate, t_mod, lambda r: r["inner_gap_m"] <= 0.020),
        "reopen_final_gap_m": reopen[-1]["inner_gap_m"],
        "seconds_reopen_to_36mm": elapsed(reopen, t_reopen, lambda r: r["inner_gap_m"] >= 0.036),
        "secure_close_final_qpos_m": phase_rows("empty_secure_close")[-1]["finger_qpos_m"],
        "open_after_secure_final_qpos_m": locked[-1]["finger_qpos_m"],
        "open_after_secure_final_gap_m": locked[-1]["inner_gap_m"],
        "legal_open_recovers_after_secure_close": bool(locked[-1]["inner_gap_m"] >= 0.036 and min(locked[-1]["finger_qpos_m"]) > 0.0),
        "video": video,
        "actuator_matches_open_command": bool(
            grasp_rows[-1]["actuator_ctrl"] > 0.0 and grasp_rows[-1]["inner_gap_m"] > 0.036
        ),
    }
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / "gripper_steps.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return summary, rows


def _table_free_model(src):
    model = mujoco.MjModel.from_xml_path(str(BALLISTIC))
    if model.nq != src.nq or model.nv != src.nv or model.nu != src.nu:
        raise RuntimeError(f"diagnostic model size {model.nq} does not match the training model {src.nq}")
    for name in ("body_mass", "body_inertia", "geom_friction", "geom_solref", "geom_solimp"):
        getattr(model, name)[:] = np.array(getattr(src, name))
    model.opt.timestep = src.opt.timestep
    model.opt.gravity[:] = np.array(src.opt.gravity, float)
    model.opt.noslip_iterations = src.opt.noslip_iterations
    model.opt.noslip_tolerance = src.opt.noslip_tolerance
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table < 0:
        raise RuntimeError("diagnostic scene has no table geom to disable")
    model.geom_contype[table] = 0
    model.geom_conaffinity[table] = 0
    data = mujoco.MjData(model)
    return model, data


class _Shim:
    def __init__(self, model, data, ids, fsm):
        self.model = model
        self.data = data
        self.ids = ids
        self.fsm = fsm


def _load_shim(env):
    model, data = _table_free_model(env.sim.model)
    data.qpos[:] = env.sim.data.qpos
    data.qvel[:] = env.sim.data.qvel
    data.ctrl[:] = env.sim.data.ctrl
    data.time = env.sim.data.time
    mujoco.mj_forward(model, data)
    return _Shim(model, data, resolve_ids(model), env.sim.fsm)


def _pinch(sim):
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    rotation = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return hand + rotation @ PINCH_HAND, rotation


def _object_contacts(sim):
    gid = int(sim.ids.object_geom)
    return sum(
        1
        for i in range(sim.data.ncon)
        if int(sim.data.contact[i].geom1) == gid or int(sim.data.contact[i].geom2) == gid
    )


def _point_fingers_up(sim):
    from controllers.recovery.action_map import TAU_OPEN, normalize_action

    dots = []
    for _ in range(80):
        rotation = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        finger_axis = rotation[:, 2]
        dot = float(np.dot(finger_axis, np.array([0.0, 0.0, 1.0])))
        dots.append(round(dot, 4))
        if dot > 0.92:
            break
        world_rate = np.cross(finger_axis, np.array([0.0, 0.0, 1.0]))
        desired = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        rate = desired.T @ np.clip(world_rate * 6.0, -3.5, 3.5)
        step_legal(sim, normalize_action(np.zeros(3), rate, TAU_OPEN, AIRBORNE_V3))
    rotation = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return dots, round(float(np.dot(rotation[:, 2], np.array([0.0, 0.0, 1.0]))), 4)


def _prepare(sim, spec, film=None):
    """Legal setup before the drop. The cylinder is written only at the end."""
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    _write_object(sim, hand + np.array([0.0, 0.0, 0.9]), np.zeros(3))
    dots, finger_up = _point_fingers_up(sim)
    if film is not None:
        film.snap(f"fingers up {finger_up:.2f}", force=True)
    for _ in range(10):
        step_legal(sim, _hold(-1.0))
        if film is not None:
            film.snap(f"preopen gap={capture_geometry(sim)['inner_gap_m']*1000:.0f}mm")
    closed_ok = True
    if spec.get("start_closed"):
        for _ in range(30):
            geo = capture_geometry(sim)
            qmin = min(geo["finger_qpos_m"])
            if spec.get("half") and qmin < 0.028:
                break
            if qmin < 0.0:
                grip = -1.0
            elif qmin < 0.025:
                grip = -0.8
            else:
                grip = -0.72
            step_legal(sim, _hold(grip))
            if film is not None:
                film.snap(f"gentle close q={qmin:.3f}")
        qmin = min(capture_geometry(sim)["finger_qpos_m"])
        closed_ok = qmin > 0.0
    pinch, _rotation = _pinch(sim)
    offset = np.asarray(spec["world_offset"], float)
    velocity = np.asarray(spec["world_velocity"], float)
    quat = np.asarray(spec["quat"], float)
    lifts = 0
    for lifts in range(6):
        _write_object(sim, pinch + offset + np.array([0.0, 0.0, 0.015 * lifts]), velocity, quat)
        if _object_contacts(sim) == 0:
            break
    geo = capture_geometry(sim)
    obj_v, _ = body_twist(sim.model, sim.data, sim.ids.object_body)
    valid = bool(
        finger_up > 0.9
        and closed_ok
        and _object_contacts(sim) == 0
        and abs(float(obj_v[2])) < 0.05
        and min(geo["finger_qpos_m"]) > 0.0
    )
    info = {
        "finger_up_dot": finger_up,
        "finger_up_trace": dots[::5],
        "clearance_lifts": int(lifts if _object_contacts(sim) == 0 else -1),
        "initial_contacts": int(_object_contacts(sim)),
        "initial_obj_v": [round(float(v), 4) for v in obj_v],
        "initial_gap_m": round(float(geo["inner_gap_m"]), 5),
        "initial_qpos_m": [round(float(v), 5) for v in geo["finger_qpos_m"]],
        "initial_obj_z": round(float(geo["obj_z"]), 4),
        "valid_initial": valid,
    }
    return info


def _grasped(row):
    qmin = min(row["finger_qpos_m"])
    return (
        row["nL"] > 0
        and row["nR"] > 0
        and row["between"]
        and qmin > 0.008
        and 0.028 <= row["inner_gap_m"] <= 0.050
    )


def run_trial(env, blob, spec, proto):
    restore_capture(env, blob["capture"])
    sim = _load_shim(env)
    info = _prepare(sim, spec)
    rows = []
    actions = []
    held = 0
    reason = "INVALID_INITIAL" if not info["valid_initial"] else "TIMEOUT"
    if info["valid_initial"]:
        for index in range(int(spec["steps"])):
            before = capture_geometry(sim)
            action, _geo = lead_action(sim, spec["mode"], float(spec["lead_m"]))
            actions.append(np.array(action, np.float32))
            step_legal(sim, action)
            row = _row(sim, action)
            row["step"] = index
            row["ready_before"] = bool(before["ready_to_close"])
            row["gap_before"] = round(float(before["inner_gap_m"]), 5)
            rows.append(row)
            if min(row["finger_qpos_m"]) < -0.003:
                reason = "FINGER_INVERSION"
                break
            if row["table_drop"] or row["obj_z"] < 0.06:
                reason = "FLOOR_BOUND" if row["obj_z"] < 0.06 else "TABLE_DROP_FLOOR"
                break
            if _grasped(row):
                held += 1
            else:
                held = 0
            if held >= 10:
                reason = "DIAGNOSTIC_CAPTURE"
                break
    rep = None
    if proto is not None and actions:
        rep = representability(actions, proto)
        rep = {k: v for k, v in rep.items() if k != "quantized"}
    return {
        "label": "TABLE_FREE_DIAGNOSTIC",
        "sft": False,
        "success_12s": False,
        "name": spec["name"],
        "mode": spec["mode"],
        "lead_m": spec["lead_m"],
        "world_offset": spec["world_offset"],
        "world_velocity": spec["world_velocity"],
        "setup": info,
        "reason": reason,
        "held_steps": held,
        "steps": len(rows),
        "final": rows[-1] if rows else None,
        "min_hand_vz": min((r["hand_v_mps"][2] for r in rows), default=None),
        "representability": rep,
        "trace": rows,
        "actions": [a.tolist() for a in actions],
    }


def film_spec(env, blob, spec):
    restore_capture(env, blob["capture"])
    sim = _load_shim(env)
    renderer = mujoco.Renderer(sim.model, 480, 640)
    film = Film(sim, renderer, _camera(sim))
    info = _prepare(sim, spec, film)
    film.snap(spec["name"] + " release", force=True)
    held = 0
    reason = "INVALID_INITIAL" if not info["valid_initial"] else "TIMEOUT"
    if info["valid_initial"]:
        for _ in range(int(spec["steps"])):
            action, _geo = lead_action(sim, spec["mode"], float(spec["lead_m"]))
            step_legal(sim, action)
            after = capture_geometry(sim)
            film.snap(
                f"{spec['name']} gap={after['inner_gap_m']*1000:.0f}mm "
                f"z={after['obj_z']:.2f} n={after['nL']}/{after['nR']} a6={action[6]:+.1f}"
            )
            row = _row(sim, action)
            if min(row["finger_qpos_m"]) < -0.003:
                reason = "FINGER_INVERSION"
                break
            if latched(sim) or after["obj_z"] < 0.06:
                reason = "FLOOR_BOUND"
                break
            if _grasped(row):
                held += 1
            else:
                held = 0
            if held >= 10:
                reason = "DIAGNOSTIC_CAPTURE"
                break
    film.snap(f"{spec['name']} {reason}", force=True)
    path = write_frames_mp4(film.frames, OUT / "videos" / f"{spec['name']}.mp4")
    renderer.close()
    return path, reason


def _spec(name, mode, lead, offset, velocity=None, quat=None, start_closed=False, half=False, steps=80):
    return {
        "name": name,
        "mode": mode,
        "lead_m": lead,
        "world_offset": offset,
        "world_velocity": velocity or [0.0, 0.0, 0.0],
        "quat": quat or [1.0, 0.0, 0.0, 0.0],
        "start_closed": start_closed,
        "half": half,
        "steps": steps,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", "cpu")
    from training.recovery_episode import make_training_env

    env = make_training_env(model=model)
    with SNAP.open("rb") as handle:
        blob = pickle.load(handle)
    official_table = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    official_contype = int(env.sim.model.geom_contype[official_table])
    gripper, _rows = gripper_probe(env, blob)
    proto = np.load(CODEBOOK) if CODEBOOK.is_file() else None
    specs = [
        # world_offset is metres above the upward fingertips. z is world up.
        _spec("A_lead_h05", "intercept", 0.02, [0.0, 0.0, 0.05]),
        _spec("A_lead_h05_L01", "intercept", 0.01, [0.0, 0.0, 0.05]),
        _spec("A_lead_h05_L04", "intercept", 0.04, [0.0, 0.0, 0.05]),
        _spec("A_lead_h10", "intercept", 0.03, [0.0, 0.0, 0.10]),
        _spec("A_lead_h16", "intercept", 0.04, [0.0, 0.0, 0.16]),
        _spec("B_sync_h05", "sync", 0.015, [0.0, 0.0, 0.05]),
        _spec("B_sync_h10", "sync", 0.02, [0.0, 0.0, 0.10]),
        _spec("C_closed_h10", "open_first", 0.03, [0.0, 0.0, 0.10], start_closed=True),
        _spec("C_closed_h16", "open_first", 0.04, [0.0, 0.0, 0.16], start_closed=True),
        _spec("A_offset_x015", "intercept", 0.03, [0.015, 0.0, 0.08]),
        _spec("A_offset_x030", "intercept", 0.03, [0.030, 0.0, 0.10]),
        _spec("A_lateral_010", "intercept", 0.03, [0.0, 0.0, 0.10], velocity=[0.10, 0.0, 0.0]),
        _spec("A_lateral_020", "intercept", 0.03, [0.0, 0.0, 0.12], velocity=[0.20, 0.0, 0.0]),
        _spec("A_tilt30", "intercept", 0.03, [0.0, 0.0, 0.10], quat=[0.9659258, 0.258819, 0.0, 0.0]),
        _spec("A_half_open", "intercept", 0.03, [0.0, 0.0, 0.10], start_closed=True, half=True),
    ]
    results = []
    for spec in specs:
        trial = run_trial(env, blob, spec, proto)
        results.append(trial)
        gap = None if not trial["final"] else trial["final"]["inner_gap_m"]
        print(spec["name"], trial["reason"], "steps", trial["steps"], "gap", gap, "up", trial["setup"]["finger_up_dot"], "z0", trial["setup"]["initial_obj_z"], flush=True)
    captured = [r for r in results if r["reason"] == "DIAGNOSTIC_CAPTURE"]
    video_specs = [specs[0]]
    if captured:
        video_specs.append(next(s for s in specs if s["name"] == captured[0]["name"]))
    videos = {}
    for spec in video_specs:
        path, reason = film_spec(env, blob, spec)
        videos[spec["name"]] = {"path": path, "reason": reason}
    summary = {
        "scene": "TABLE_FREE_DIAGNOSTIC",
        "sft_written": False,
        "official_table_contype_unchanged": int(env.sim.model.geom_contype[official_table]) == official_contype,
        "object_mass_kg": float(env.sim.model.body_mass[env.sim.ids.object_body]),
        "gripper": gripper,
        "trials": [{k: v for k, v in r.items() if k != "trace"} for r in results],
        "captures": [r["name"] for r in captured],
        "videos": videos,
    }
    (OUT / "raw" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "raw" / "traces.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "actions"} for r in results], indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"gripper_opens": gripper["actuator_matches_open_command"], "captures": summary["captures"]}, indent=2))


if __name__ == "__main__":
    main()
