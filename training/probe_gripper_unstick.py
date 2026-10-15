"""Why a legal open torque stops moving the fingers after a full close."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.autonomous_episode import make_training_env
from training.capture_geometry import capture_geometry
from training.history_aware_critic_ablation import restore_capture
from training.privileged_ball_impact_search import step_legal
from training.unified_k48 import load_trained_k48


def qpos(sim):
    adr = [int(sim.model.jnt_qposadr[j]) for j in sim.ids.finger_jnt]
    dof = [int(sim.model.jnt_dofadr[j]) for j in sim.ids.finger_jnt]
    return (
        [round(float(sim.data.qpos[a]), 5) for a in adr],
        [round(float(sim.data.qvel[d]), 5) for d in dof],
        [round(float(sim.data.qfrc_actuator[d]), 4) for d in dof],
        [round(float(sim.data.qfrc_constraint[d]), 4) for d in dof],
    )


def run(sim, action, n, label):
    print(label)
    for i in range(n):
        step_legal(sim, np.array([0, 0, 0, 0, 0, 0, action], np.float32))
        if i % 2 == 0 or i == n - 1:
            q, v, fa, fc = qpos(sim)
            gap = capture_geometry(sim)["inner_gap_m"]
            print(f"  {i:02d} a={action:+.2f} ctrl={sim.data.ctrl[7]:+.2f} q={q} v={v} fact={fa} fcon={fc} gap={gap:.5f}")


def sweep():
    import mujoco
    model, _ = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", "cpu")
    env = make_training_env(model=model)
    blob = pickle.load(open(ROOT / "data/snapshots/train/ball_train_v2_0051.pkl", "rb"))
    restore_capture(env, blob["capture"])
    sim = env.sim
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    adr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    sim.data.qpos[adr : adr + 3] = hand + np.array([0.0, 0.0, 0.8])
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(15):
        step_legal(sim, np.array([0, 0, 0, 0, 0, 0, -1.0], np.float32))
    print("opened", qpos(sim)[0], "gap", round(capture_geometry(sim)["inner_gap_m"], 5))
    for action in (-0.85, -0.8, -0.7, -0.6, -0.4, 0.0, 0.4, 1.0):
        # reopen fully between trials
        for _ in range(12):
            step_legal(sim, np.array([0, 0, 0, 0, 0, 0, -1.0], np.float32))
        q0 = qpos(sim)[0][0]
        if q0 < 0.03:
            print("cannot reopen, stop sweep at", action, q0)
            break
        min_q = q0
        for i in range(20):
            step_legal(sim, np.array([0, 0, 0, 0, 0, 0, action], np.float32))
            q = qpos(sim)[0][0]
            min_q = min(min_q, q)
            if i in (0, 4, 19):
                print(f"  a={action:+.2f} i={i} q={q:.5f} gap={capture_geometry(sim)['inner_gap_m']:.5f}")
        print(f"  a={action:+.2f} min_q={min_q:.5f}")


def main():
    model, _ = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", "cpu")
    env = make_training_env(model=model)
    blob = pickle.load(open(ROOT / "data/snapshots/train/ball_train_v2_0051.pkl", "rb"))
    restore_capture(env, blob["capture"])
    sim = env.sim
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    adr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    sim.data.qpos[adr : adr + 3] = hand + np.array([0.0, 0.0, 0.8])
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    print("limits", sim.model.jnt_range[sim.ids.finger_jnt].tolist())
    print("start", qpos(sim), "gap", round(capture_geometry(sim)["inner_gap_m"], 5))
    run(sim, 1.0, 1, "one close step")
    run(sim, -1.0, 15, "open after one close")
    run(sim, 1.0, 8, "deep close")
    run(sim, -1.0, 10, "open after deep close")
    run(sim, -0.2, 8, "milder open action -0.2")
    sweep()


if __name__ == "__main__":
    main()
