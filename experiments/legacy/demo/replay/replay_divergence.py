"""Step-by-step comparison of continuous and K=47 decoded replay.

Does not change physics, the success rule, or the active codebook.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAJ = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/divergence.json"
CASES = [
    ("ball_train_v2_0016", "chase_lost"),
    ("ball_train_v2_0020", "chase_lost"),
    ("ball_train_v2_0035", "chase_lost"),
]


def decode(actions, proto):
    gap = np.max(np.abs(proto[None] - actions[:, None]), axis=2)
    nearest = gap.argmin(1)
    residual = np.clip(actions - proto[nearest], -0.25, 0.25)
    recon = np.clip(proto[nearest] + residual, -1.0, 1.0).astype(np.float32)
    return recon, gap.min(1)


def fingerprint(env):
    sim = env.sim
    fsm = sim.fsm
    return {
        "qpos": np.array(sim.data.qpos, np.float64).copy(),
        "qvel": np.array(sim.data.qvel, np.float64).copy(),
        "ctrl": np.array(sim.data.ctrl, np.float64).copy(),
        "time": float(sim.data.time),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, np.float64).copy(),
        "p_des": np.array(fsm.p_des, np.float64).copy(),
        "r_des": np.array(fsm.r_des, np.float64).copy(),
        "v_cmd": np.array(fsm.v_cmd, np.float64).copy(),
        "w_cmd": np.array(fsm.w_cmd, np.float64).copy(),
        "fg_cmd": float(fsm.fg_cmd),
        "phase": str(fsm.phase),
    }


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def trace(env, blob, actions):
    from envs.simulation.capture import restore_capture
    from experiments.ball_impact.privileged_ball_impact_search import step_legal
    from envs.outcomes.physical_recovery import physical_pack

    restore_capture(env, blob["capture"])
    start = fingerprint(env)
    rows = []
    sim = env.sim
    for index, action in enumerate(actions):
        sent = np.asarray(action, np.float32).copy()
        step_legal(sim, sent)
        pack = physical_pack(sim)
        rows.append({
            "step": index,
            "action": sent.astype(np.float64),
            "qpos": np.array(sim.data.qpos, np.float64).copy(),
            "qvel": np.array(sim.data.qvel, np.float64).copy(),
            "hand": np.array(sim.data.xpos[sim.ids.hand_body], np.float64).copy(),
            "obj": np.array(sim.data.xpos[sim.ids.object_body], np.float64).copy(),
            "hand_v": np.array(sim.data.cvel[sim.ids.hand_body], np.float64).copy() if hasattr(sim.data, "cvel") else np.zeros(6),
            "obj_v": np.array(sim.data.cvel[sim.ids.object_body], np.float64).copy() if hasattr(sim.data, "cvel") else np.zeros(6),
            "nL": int(pack["nL"]),
            "nR": int(pack["nR"]),
            "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else 0.0,
            "v_cmd": np.array(sim.fsm.v_cmd, np.float64).copy(),
            "fg_cmd": float(sim.fsm.fg_cmd),
        })
    return start, rows


def first_divergence(left, right):
    found = []
    limit = min(len(left), len(right))
    for index in range(limit):
        a = left[index]
        b = right[index]
        checks = {
            "action": max_abs(a["action"], b["action"]),
            "v_cmd": max_abs(a["v_cmd"], b["v_cmd"]),
            "fg_cmd": abs(a["fg_cmd"] - b["fg_cmd"]),
            "ctrl7": abs(a["ctrl7"] - b["ctrl7"]),
            "hand": max_abs(a["hand"], b["hand"]),
            "obj": max_abs(a["obj"], b["obj"]),
            "hand_v": max_abs(a["hand_v"], b["hand_v"]),
            "obj_v": max_abs(a["obj_v"], b["obj_v"]),
            "qpos": max_abs(a["qpos"], b["qpos"]),
            "qvel": max_abs(a["qvel"], b["qvel"]),
            "contact": 0.0 if (a["nL"], a["nR"]) == (b["nL"], b["nR"]) else 1.0,
        }
        interesting = {key: value for key, value in checks.items() if value > 0.0}
        if interesting and not found:
            found.append({"step": index, "nonzero": {k: v for k, v in interesting.items()}})
        if checks["qpos"] > 1e-6 or checks["contact"] > 0:
            return {
                "first_nonzero": found[0] if found else None,
                "first_state_or_contact": {"step": index, **{k: checks[k] for k in checks}},
                "contact_at_divergence": {"left": [a["nL"], a["nR"]], "right": [b["nL"], b["nR"]]},
            }
    return {"first_nonzero": found[0] if found else None, "first_state_or_contact": None}


def main():
    import torch
    from training.recovery_episode import make_training_env
    from models.historical.k36_checkpoint import load_models
    from envs.simulation.capture import restore_capture
    from experiments.sliding.sliding_vs_regrasp import _replay_actions

    proto = np.asarray(torch.load(ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra = np.load(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    candidate = np.concatenate([proto, extra], axis=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt",
        ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    blobs = {}
    for path in V2.glob("*.pkl"):
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        blobs[blob["spec"]["id"]] = blob
    report = []
    for episode_id, name in CASES:
        actions = np.load(TRAJ / f"{episode_id}__{name}.npz")["actions"].astype(np.float32)
        recon, linf = decode(actions, candidate)
        blob = blobs[episode_id]
        restore_capture(env, blob["capture"])
        start_a = fingerprint(env)
        restore_capture(env, blob["capture"])
        start_b = fingerprint(env)
        restore_gap = {key: max_abs(start_a[key], start_b[key]) for key in ("qpos", "qvel", "ctrl", "qacc_warmstart", "p_des", "r_des", "v_cmd", "w_cmd")}
        restore_gap["fg_cmd"] = abs(start_a["fg_cmd"] - start_b["fg_cmd"])
        restore_gap["time"] = abs(start_a["time"] - start_b["time"])
        restore_gap["phase_match"] = start_a["phase"] == start_b["phase"]
        _start_c, trace_c = trace(env, blob, actions)
        _start_d, trace_d = trace(env, blob, recon)
        _start_e, trace_e = trace(env, blob, actions.copy())
        split = first_divergence(trace_c, trace_d)
        repeat = first_divergence(trace_c, trace_e)
        raw = _replay_actions(env, blob, actions.copy(), "regrasp", None)
        again = _replay_actions(env, blob, actions.copy(), "regrasp", None)
        decoded = _replay_actions(env, blob, recon.copy(), "regrasp", None)
        decoded_again = _replay_actions(env, blob, recon.copy(), "regrasp", None)
        item = {
            "id": episode_id,
            "strategy": name,
            "steps": int(len(actions)),
            "max_action_delta": float(np.max(np.abs(actions - recon))),
            "max_linf_to_codebook": float(linf.max()),
            "n_steps_action_differs": int(np.sum(np.max(np.abs(actions - recon), axis=1) > 0)),
            "restore_repeat_gap": restore_gap,
            "continuous_vs_decoded": split,
            "continuous_vs_continuous_trace": repeat,
            "outcomes": {
                "continuous": raw.get("existing_label"),
                "continuous_repeat": again.get("existing_label"),
                "decoded": decoded.get("existing_label"),
                "decoded_repeat": decoded_again.get("existing_label"),
            },
        }
        report.append(item)
        print(episode_id, item["outcomes"], "delta", item["max_action_delta"], "split", split.get("first_state_or_contact", {}).get("step") if split.get("first_state_or_contact") else None, flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
