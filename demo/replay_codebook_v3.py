"""Replay release holds under the fit-only appended codebook. Does not train."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
TRAJ = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/trajectories"
V2 = ROOT / "demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
OUT = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/replay_v3.json"


def main():
    import torch
    from training.autonomous_episode import make_training_env
    from training.formal_grpo import load_models
    from training.sliding_vs_regrasp import _replay_actions

    proto = np.asarray(
        torch.load(ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt", map_location="cpu", weights_only=False)["proto"],
        np.float32,
    )
    extra = np.load(ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/candidate_extra.npy")
    candidate = np.concatenate([proto, extra], axis=0)
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("primary_10s") and json.loads(line).get("family") == "regrasp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt",
        ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
        device,
    )
    env = make_training_env(device, model=model)
    blobs = {}
    for path in V2.glob("*.pkl"):
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        blobs[blob["spec"]["id"]] = blob
    results = []
    for row in rows:
        actions = np.load(TRAJ / f"{row['id']}__{row['strategy']}.npz")["actions"].astype(np.float32)
        gap = np.max(np.abs(candidate[None] - actions[:, None]), axis=2)
        nearest = gap.argmin(1)
        linf = gap.min(1)
        if float(linf.max()) > 0.25 + 1e-6:
            item = {"id": row["id"], "strategy": row["strategy"], "max_linf": round(float(linf.max()), 3), "label": "OUT_OF_RANGE", "primary_10s": None, "held_12": None, "legacy_t12": None}
            results.append(item)
            print("OUT", item["id"], item["strategy"], item["max_linf"], flush=True)
            continue
        resid = np.clip(actions - candidate[nearest], -0.25, 0.25)
        recon = np.clip(candidate[nearest] + resid, -1.0, 1.0).astype(np.float32)
        replay = _replay_actions(env, blobs[row["id"]], recon, "regrasp", None)
        hold = replay.get("hold") or {}
        item = {
            "id": row["id"],
            "strategy": row["strategy"],
            "max_linf": round(float(linf.max()), 3),
            "label": replay.get("existing_label"),
            "primary_10s": bool(replay.get("primary_10s")),
            "held_12": hold.get("held_12"),
            "legacy_t12": hold.get("legacy_retained_to_12"),
        }
        results.append(item)
        print("REPLAY", item["id"], item["strategy"], item["label"], flush=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for item in results if item.get("primary_10s"))
    print("HOLDS", ok, "of", len(results), flush=True)


if __name__ == "__main__":
    main()
