"""Physical replay of representable holds under K=47 and the revised K=56 candidate.

Resumable. Does not train and does not edit the continuous trajectories.
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

LOG = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
TOKENS = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_tokens.jsonl"
OUT47 = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/replay_k47_all.jsonl"
OUT56 = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/replay_k56.jsonl"
PRIOR = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/replay_v3.json"


def load_done(path):
    done = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[f"{row['id']}|{row['strategy']}"] = row
    return done


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main():
    import torch
    from training.recovery_episode import make_training_env
    from models.historical.k36_checkpoint import load_models
    from experiments.sliding.sliding_vs_regrasp import _replay_actions, representability

    proto = np.asarray(torch.load(ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"], np.float32)
    extra47 = np.load(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    extra56 = np.load(ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/revised_extra.npy")
    book47 = np.concatenate([proto, extra47], axis=0)
    book56 = np.concatenate([proto, extra56], axis=0)
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("primary_10s")]
    gaps = {}
    if TOKENS.is_file():
        for line in TOKENS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                gaps[f"{item['id']}|{item['strategy']}"] = int(item.get("none_intervals") or 0)
    prior = {}
    if PRIOR.is_file():
        for item in json.loads(PRIOR.read_text(encoding="utf-8")):
            prior[f"{item['id']}|{item['strategy']}"] = item
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
    done47 = load_done(OUT47)
    done56 = load_done(OUT56)

    def play(book, episode_id, strategy, family, actions):
        info = representability(actions, book)
        if info["fraction_within_0.25"] < 1.0 - 1e-9:
            return {"status": "NOT_REPRESENTABLE", "max_linf": round(float(info["max_linf"]), 4), "primary_10s": None, "label": None, "held_12": None, "legacy_t12": None}
        result = _replay_actions(env, blobs[episode_id], info["quantized"], family, None)
        hold = result.get("hold") or {}
        return {
            "status": "REPLAYED",
            "max_linf": round(float(info["max_linf"]), 4),
            "primary_10s": bool(result.get("primary_10s")),
            "label": result.get("existing_label"),
            "held_12": None if not hold else hold.get("held_12"),
            "legacy_t12": None if not hold else hold.get("legacy_retained_to_12"),
        }

    for row in rows:
        key = f"{row['id']}|{row['strategy']}"
        if key in done47:
            continue
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        info36 = representability(actions, proto)
        info47 = representability(actions, book47)
        same = info36["fraction_within_0.25"] == 1.0 and np.allclose(info36["quantized"], info47["quantized"], atol=1e-5)
        if info47["fraction_within_0.25"] < 1.0 - 1e-9:
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"], "status": "NOT_REPRESENTABLE", "max_linf": round(float(info47["max_linf"]), 4), "primary_10s": None, "label": None}
        elif key in prior:
            old = prior[key]
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"], "status": "PRIOR_REPLAY", "max_linf": old.get("max_linf"), "primary_10s": old.get("primary_10s"), "label": old.get("label"), "held_12": old.get("held_12"), "legacy_t12": old.get("legacy_t12")}
        elif same and row.get("k36_replay_10s"):
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"], "status": "INHERITED_K36", "max_linf": round(float(info47["max_linf"]), 4), "primary_10s": True, "label": "SUCCESS_12S", "held_12": row.get("held_12"), "legacy_t12": row.get("legacy_t12")}
        else:
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"]}
            item.update(play(book47, row["id"], row["strategy"], row["family"], actions))
        append(OUT47, item)
        done47[key] = item
        print("K47", row["id"], row["strategy"], item["status"], item.get("label"), flush=True)

    for row in rows:
        key = f"{row['id']}|{row['strategy']}"
        number = int(row["id"].rsplit("_", 1)[-1])
        multi = gaps.get(key, 0) >= 2
        named = row["id"] in {"ball_train_v2_0016", "ball_train_v2_0020", "ball_train_v2_0024", "ball_train_v2_0025", "ball_train_v2_0035", "ball_train_v2_0021"}
        original = number < 40
        if not (number % 5 == 0 or multi or named or original):
            continue
        if key in done56:
            continue
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        info36 = representability(actions, proto)
        info56 = representability(actions, book56)
        same = info36["fraction_within_0.25"] == 1.0 and np.allclose(info36["quantized"], info56["quantized"], atol=1e-5)
        if info56["fraction_within_0.25"] < 1.0 - 1e-9:
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"], "status": "NOT_REPRESENTABLE", "max_linf": round(float(info56["max_linf"]), 4), "primary_10s": None, "label": None}
        elif same and row.get("k36_replay_10s"):
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"], "status": "INHERITED_K36", "max_linf": round(float(info56["max_linf"]), 4), "primary_10s": True, "label": "SUCCESS_12S", "held_12": row.get("held_12"), "legacy_t12": row.get("legacy_t12")}
        else:
            item = {"id": row["id"], "strategy": row["strategy"], "family": row["family"]}
            item.update(play(book56, row["id"], row["strategy"], row["family"], actions))
        append(OUT56, item)
        done56[key] = item
        print("K56", row["id"], row["strategy"], item["status"], item.get("label"), flush=True)
    print("REPLAY_DONE", len(done47), len(done56), flush=True)


if __name__ == "__main__":
    main()
