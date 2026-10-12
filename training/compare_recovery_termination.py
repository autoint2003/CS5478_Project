"""Compare old and new recovery stop rules on the 13 reclassified episodes.

Does not train and does not change physics, the codebook, or the task rule.
The old closed-loop diagnosis stopped only for object height below 0.30 m or
for the basin early-exit. The migrated evaluator also stops after 50
contact-free control steps below 0.44 m, or on escape.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.autonomous_episode import make_training_env
from training.diagnose_k47_closed_loop import CKPT, load_model
from training.history_aware_critic_ablation import restore_capture
from training.horizon_outcome import recovery_failed
from training.nominal_pose_audit import errors
from training.privileged_ball_impact_search import step_legal
from training.recovery_runtime import _escaped
from training.recovery_success import PERSIST_STEPS, basin_ok
from training.unified_k47 import decode_action, load_codebook, read_jsonl

IDS = (
    "ball_train_v2_0015",
    "ball_train_v2_0035",
    "ball_train_v2_0050",
    "ball_train_v2_0060",
    "ball_train_v2_0065",
    "ball_train_v2_0110",
    "ball_train_v2_0140",
    "ball_train_v2_0155",
    "ball_train_v2_0170",
    "ball_train_v2_0180",
    "ball_train_v2_0245",
    "ball_train_v2_0265",
    "ball_train_v2_0270",
)
OUT = ROOT / "results/diagnostics/raw/horizon_migration/termination_comparison.json"
GRPO = ROOT / "results/diagnostics/raw/legal_recovery_v2/grpo_initial_conditions.jsonl"


def snapshot(env):
    sim = env.sim
    fsm = sim.fsm
    return {
        "time": float(sim.data.time),
        "qpos": np.array(sim.data.qpos, np.float64).copy(),
        "qvel": np.array(sim.data.qvel, np.float64).copy(),
        "ctrl": np.array(sim.data.ctrl, np.float64).copy(),
        "phase": str(fsm.phase),
        "grasp_z": float(fsm.grasp_z),
        "hist": int(len(env._obs_seq)),
        "prev": np.array(env._prev, np.float32).copy(),
    }


def rollout(env, model, book, blob, device, stop_on_search_rule):
    restore_capture(env, blob["capture"])
    opened = snapshot(env)
    lost_low = 0
    basin_run = 0
    steps = []
    stop = "cap"
    for _ in range(120):
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        _residual, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = executed
        measured = errors(env.sim, blob["ref"])
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        inside = basin_ok(measured, float(executed[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        steps.append({
            "token": token,
            "action": [round(float(x), 5) for x in executed],
            "nL": int(measured["nL"]),
            "nR": int(measured["nR"]),
            "z": round(float(measured["z"]), 4),
            "ep": round(float(measured["ep"]), 5),
            "lost_low": int(lost_low),
            "escaped": bool(_escaped(env.sim)),
            "basin": bool(inside),
        })
        if float(measured["z"]) < 0.30:
            stop = "z_below_0.30"
            break
        if stop_on_search_rule and failed:
            stop = "search_fall_rule"
            break
        if basin_run >= PERSIST_STEPS:
            stop = "basin_early_exit"
            break
    return opened, steps, stop


def prefix_gap(left, right):
    n = min(len(left), len(right))
    for index in range(n):
        gap = float(np.max(np.abs(np.asarray(left[index]["action"]) - np.asarray(right[index]["action"]))))
        if gap > 1e-5:
            return index, gap
    return None, 0.0


def main():
    book = load_codebook()
    by_id = {row["id"]: row for row in read_jsonl(GRPO)}
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    rows = []
    old_actions = {}
    for device in devices:
        model = load_model(book, device)
        if model.training:
            raise RuntimeError("checkpoint is not in eval mode")
        env = make_training_env(device, model=model)
        for episode_id in IDS:
            with (ROOT / by_id[episode_id]["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            old_state, old_steps, old_stop = rollout(env, model, book, blob, device, False)
            new_state, new_steps, new_stop = rollout(env, model, book, blob, device, True)
            state_gap = float(np.max(np.abs(old_state["qpos"] - new_state["qpos"])))
            ctrl_gap = float(np.max(np.abs(old_state["ctrl"] - new_state["ctrl"])))
            diverge, gap = prefix_gap(old_steps, new_steps)
            search_at = next((i + 1 for i, step in enumerate(old_steps) if step["lost_low"] >= 50 or step["escaped"] or step["z"] < 0.30), None)
            old_actions[(str(device), episode_id)] = [step["action"] for step in old_steps]
            rows.append({
                "id": episode_id,
                "device": str(device),
                "inference": "eval_argmax",
                "checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
                "training_mode": bool(model.training),
                "init_time": old_state["time"],
                "init_phase": old_state["phase"],
                "init_hist": old_state["hist"],
                "init_prev_linf": float(np.max(np.abs(old_state["prev"]))),
                "restore_qpos_gap": state_gap,
                "restore_ctrl_gap": ctrl_gap,
                "old_steps": len(old_steps),
                "old_stop": old_stop,
                "old_min_z": min(step["z"] for step in old_steps),
                "old_end_z": old_steps[-1]["z"],
                "old_end_contacts": [old_steps[-1]["nL"], old_steps[-1]["nR"]],
                "search_rule_step": search_at,
                "new_steps": len(new_steps),
                "new_stop": new_stop,
                "new_end_z": new_steps[-1]["z"],
                "new_end_contacts": [new_steps[-1]["nL"], new_steps[-1]["nR"]],
                "new_end_lost_low": new_steps[-1]["lost_low"],
                "new_end_escaped": new_steps[-1]["escaped"],
                "action_diverge_step": diverge,
                "action_gap": gap,
            })
            print(episode_id, device, old_stop, len(old_steps), new_stop, len(new_steps), "diverge", diverge, flush=True)
    if len(devices) == 2:
        cross = {}
        for episode_id in IDS:
            left = old_actions[("cpu", episode_id)]
            right = old_actions[("cuda", episode_id)]
            index, gap = prefix_gap(
                [{"action": action} for action in left],
                [{"action": action} for action in right],
            )
            cross[episode_id] = {"diverge_step": index, "gap": gap, "cpu_steps": len(left), "cuda_steps": len(right)}
        rows.append({"cpu_vs_cuda_old_rule": cross})
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
