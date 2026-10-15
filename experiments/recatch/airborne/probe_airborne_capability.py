"""Measure recovery tracking, gripper motion, and post-impact cylinder state."""

from __future__ import annotations


import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from controllers.gripper.controller import finger_opening
from controllers.recovery.action_map import AIRBORNE_V3, map_action
from envs.simulation.grasp_quality import body_twist
from envs.outcomes.physical_recovery import physical_pack
from training.recovery_episode import make_training_env

from experiments.ball_impact.privileged_ball_impact_search import step_legal
from envs.outcomes.table_drop import latched
from training.unified_k48 import load_trained_k48

OUT = ROOT / "results" / "diagnostics" / "airborne_recatch_search" / "raw"


def _load(path):
    with (ROOT / path).open("rb") as handle:
        return pickle.load(handle)


def _hand(sim):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    return rh, ph, np.asarray(vh, float), np.asarray(wh, float)


def _object(sim):
    ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    return ro, po, np.asarray(vo, float), np.asarray(wo, float)


def _fingers(sim):
    left = np.array(sim.data.xpos[sim.ids.left_body], float)
    right = np.array(sim.data.xpos[sim.ids.right_body], float)
    return left, right, finger_opening(sim.data, sim.ids)


def _state(sim):
    rh, ph, vh, wh = _hand(sim)
    ro, po, vo, wo = _object(sim)
    left, right, opening = _fingers(sim)
    pack = physical_pack(sim)
    return {
        "t": float(sim.data.time),
        "ph": ph,
        "vh": vh,
        "wh": wh,
        "rh": rh,
        "po": po,
        "vo": vo,
        "wo": wo,
        "axis": ro[:, 2],
        "local": rh.T @ (po - ph),
        "left": left,
        "right": right,
        "opening": opening,
        "gap": float(np.linalg.norm(left - right)),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "z": float(pack["obj_z"]),
        "latched": bool(latched(sim)),
        "ctrl_grip": float(sim.data.ctrl[7]),
    }


def _run(sim, action, steps):
    rows = []
    for _ in range(steps):
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(action, rd, rh, AIRBORNE_V3)
        before = _state(sim)
        step_legal(sim, np.asarray(action, np.float32))
        after = _state(sim)
        rows.append({
            "command_v_world": cmd["v_world"].tolist(),
            "actual_v": after["vh"].tolist(),
            "speed": float(np.linalg.norm(after["vh"])),
            "opening": after["opening"],
            "gap": after["gap"],
            "dz": float(after["ph"][2] - before["ph"][2]),
            "dhand": float(np.linalg.norm(after["ph"] - before["ph"])),
        })
    return rows


def main():
    import torch
    device = torch.device("cpu")
    model, _saved = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = _load("data/snapshots/train/ball_train_v2_0051.pkl")
    restore_capture(env, blob["capture"])
    sim = env.sim
    initial = _state(sim)
    summary = {
        "dt_physics": float(sim.model.opt.timestep),
        "finger_range": [float(sim.model.jnt_range[sim.ids.finger_jnt[0], 0]), float(sim.model.jnt_range[sim.ids.finger_jnt[0], 1])],
        "grip_ctrl_range": [float(sim.ids.ctrl_low[7]), float(sim.ids.ctrl_high[7])],
        "object_mass": float(sim.model.body_mass[sim.ids.object_body]),
        "gravity": np.array(sim.model.opt.gravity, float).tolist(),
        "initial": {
            "t": initial["t"],
            "local": initial["local"].tolist(),
            "gap": initial["gap"],
            "opening": initial["opening"],
            "nL": initial["nL"],
            "nR": initial["nR"],
            "z": initial["z"],
            "vo": initial["vo"].tolist(),
            "wo": initial["wo"].tolist(),
            "vh": initial["vh"].tolist(),
            "axis": initial["axis"].tolist(),
            "hand_x": initial["rh"][:, 0].tolist(),
            "hand_y": initial["rh"][:, 1].tolist(),
            "hand_z": initial["rh"][:, 2].tolist(),
            "left_local": (initial["rh"].T @ (initial["left"] - initial["ph"])).tolist(),
            "right_local": (initial["rh"].T @ (initial["right"] - initial["ph"])).tolist(),
        },
    }
    axes = {
        "vx+": np.array([1, 0, 0, 0, 0, 0, 1], np.float32),
        "vy+": np.array([0, 1, 0, 0, 0, 0, 1], np.float32),
        "vz+": np.array([0, 0, 1, 0, 0, 0, 1], np.float32),
        "vz-": np.array([0, 0, -1, 0, 0, 0, 1], np.float32),
        "wx+": np.array([0, 0, 0, 1, 0, 0, 1], np.float32),
    }
    tracking = {}
    for name, action in axes.items():
        restore_capture(env, blob["capture"])
        rows = _run(sim, action, 25)
        speeds = [row["speed"] for row in rows]
        cmd = np.linalg.norm(rows[-1]["command_v_world"])
        tracking[name] = {
            "command_speed": float(cmd),
            "actual_speed_p50": float(np.median(speeds[5:])),
            "actual_speed_last": float(speeds[-1]),
            "actual_v_last": rows[-1]["actual_v"],
            "command_v_last": rows[-1]["command_v_world"],
            "travel_20steps_m": float(sum(row["dhand"] for row in rows)),
        }
    restore_capture(env, blob["capture"])
    open_rows = _run(sim, np.array([0, 0, 0, 0, 0, 0, -1], np.float32), 30)
    open_t = None
    for index, row in enumerate(open_rows):
        if row["opening"] > 0.03 and open_t is None:
            open_t = (index + 1) * 0.02
    close_rows = _run(sim, np.array([0, 0, 0, 0, 0, 0, 1], np.float32), 30)
    close_t = None
    for index, row in enumerate(close_rows):
        if row["opening"] < 0.005 and close_t is None:
            close_t = (index + 1) * 0.02
    summary["tracking"] = tracking
    summary["gripper"] = {
        "open_to_0.03_s": open_t,
        "close_to_0.005_s": close_t,
        "opening_trace": [round(row["opening"], 4) for row in open_rows[:15]],
        "closing_trace": [round(row["opening"], 4) for row in close_rows[:15]],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capability.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
