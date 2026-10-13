"""Legal recovery search on v2 TRAIN snapshots.

Uses handoff_z=None. A second replay with 0.455 m is stored beside it and does
not replace the label. Does not read FINAL_TEST. Does not train.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import v1_normalized_to_v3
from envs.physical_recovery import Z_AIR
from training.privileged_ball_impact_search import command as release_command
from training.recovery_policy_pretraining_preparation import schedules
from training.sliding_vs_regrasp import (
    SLIDE_SPECS,
    _replay_actions,
    _run_policy,
    representability,
    slide_action,
)

RAW = ROOT / "results" / "diagnostics" / "raw" / "legal_recovery_v2"
LOG = RAW / "search.jsonl"
V2 = ROOT / "results" / "diagnostics" / "raw" / "ball_impact_dataset_v2" / "accepted" / "train"
PROTO_PATH = ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_B_init.pt"


def slide_policy(sim, switch, ref, spec, memory):
    return slide_action(sim, switch, ref, spec, memory)


def release_policy(sim, switch, ref, spec, memory):
    return release_command(sim, ref, spec, int(memory["index"]))


def load_done():
    if not LOG.is_file():
        return set()
    done = set()
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add(row["key"])
    return done


def append(row):
    RAW.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def episodes():
    if not V2.is_dir():
        return []
    out = []
    for path in sorted(V2.glob("*.pkl")):
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        out.append(blob)
    return out


def brief(result, gate_name):
    hold = result.get("hold") or {}
    return {
        "gate": gate_name,
        "reached": bool(result.get("reached_gate")),
        "handed": bool(result.get("handed")),
        "primary_10s": bool(result.get("primary_10s")),
        "held_12": None if not hold else bool(hold.get("held_12")),
        "legacy_t12": None if not hold else hold.get("legacy_retained_to_12"),
        "label": result.get("existing_label"),
        "min_ep_m": result.get("metrics", {}).get("min_ep_m"),
        "zero_steps": result.get("metrics", {}).get("zero_steps"),
        "fell": bool(result.get("fell")),
        "z_end": None if not hold else (hold.get("end") or {}).get("z"),
    }


def coverage(actions, proto):
    if actions is None or len(actions) == 0:
        return None
    info = representability(actions, proto)
    return {
        "fraction_within_0.25": info["fraction_within_0.25"],
        "max_linf": info["max_linf"],
        "representable": bool(info["fraction_within_0.25"] == 1.0),
    }


def main():
    import argparse
    import torch
    from training.autonomous_episode import make_training_env
    from training.formal_grpo import load_models

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ics", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(PROTO_PATH, ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_A_reference.pt", device)
    proto = np.asarray(torch.load(PROTO_PATH, map_location="cpu", weights_only=False)["proto"], np.float32)
    env = make_training_env(device, model=model)
    done = load_done()
    wrist = np.stack([v1_normalized_to_v3(np.asarray(a, np.float32)) for a in schedules()["wrist_reseat"]])
    slide = np.stack([v1_normalized_to_v3(np.asarray(a, np.float32)) for a in schedules()["gravity_inward"]])
    specs = [
        ("sliding", "slide_50", slide_policy, SLIDE_SPECS[3]),
        ("sliding", "slide_45", slide_policy, SLIDE_SPECS[1]),
        ("regrasp", "chase_lost", release_policy, {"name": "chase_lost", "kp": 18.0, "ff": 1.5, "close": 1.0, "partial": -1.0, "open_m": 0.012, "wrist": 0.2}),
        ("regrasp", "release_long", release_policy, {"name": "release_long", "kp": 16.0, "ff": 1.2, "close": 1.0, "partial": -1.0, "open_m": 0.03, "wrist": 0.2, "force_open": 8}),
    ]
    blobs = episodes()[: args.max_ics]
    print("ICS", [blob["spec"]["id"] for blob in blobs], flush=True)
    for blob in blobs:
        episode_id = blob["spec"]["id"]
        for family, name, policy, spec in specs:
            key = f"{episode_id}|{family}|{name}"
            if key in done:
                continue
            result = _run_policy(env, blob, policy, spec, family, None, 0, None, True)
            row = {"key": key, "id": episode_id, "family": family, "strategy": name, "z_air": Z_AIR}
            row.update(brief(result, "none"))
            row["k36"] = coverage(result["actions"], proto)
            if result.get("primary_10s"):
                gated = _replay_actions(env, blob, result["actions"], family, 0.455)
                row["gate_0.455"] = brief(gated, "0.455")
                info = representability(result["actions"], proto)
                if info["fraction_within_0.25"] == 1.0:
                    replay = _replay_actions(env, blob, info["quantized"], family, None)
                    row["k36_replay_10s"] = bool(replay.get("primary_10s"))
                else:
                    row["k36_replay_10s"] = False
                folder = RAW / "trajectories"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f"{episode_id}__{name}.npz", actions=result["actions"], observations=result["observations"])
            append(row)
            done.add(key)
            print("TRY", key, row["label"], "10s", row["primary_10s"], "k36", None if not row["k36"] else row["k36"]["representable"], flush=True)
        for name, actions in (("hist_wrist", wrist), ("hist_gravity_inward", slide)):
            key = f"{episode_id}|historical|{name}"
            if key in done:
                continue
            result = _replay_actions(env, blob, actions, "slide", None)
            row = {"key": key, "id": episode_id, "family": "historical", "strategy": name}
            row.update(brief(result, "none"))
            row["k36"] = coverage(actions, proto)
            append(row)
            done.add(key)
            print("TRY", key, row["label"], "10s", row["primary_10s"], flush=True)
    print("SEARCH_DONE", len(blobs), flush=True)


if __name__ == "__main__":
    main()
