"""Append chase prototypes and replay only trajectories whose quantized actions change.

Existing K=36 centers are not moved. Does not train.
"""

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
PROTO_PATH = ROOT / "demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt"
OUT = ROOT / "demo_results/diagnostics/raw/legal_recovery_v2/candidate_codebook.json"


def load_proto():
    import torch
    saved = torch.load(PROTO_PATH, map_location="cpu", weights_only=False)
    return np.asarray(saved["proto"], np.float32)


def farthest(points, k):
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))]
    while len(chosen) < k:
        dist = np.min(np.linalg.norm(points[:, None, :] - points[chosen][None, :, :], axis=2), axis=1)
        chosen.append(int(np.argmax(dist)))
    return points[chosen].astype(np.float32)


def quantize(actions, proto):
    gap = np.max(np.abs(proto[None, :, :] - actions[:, None, :]), axis=2)
    nearest = gap.argmin(1)
    resid = np.clip(actions - proto[nearest], -0.25, 0.25)
    recon = np.clip(proto[nearest] + resid, -1.0, 1.0).astype(np.float32)
    linf = np.max(np.abs(actions - proto[nearest]), axis=1)
    return recon, linf


def main():
    from training.sliding_vs_regrasp import _replay_actions

    import torch
    from training.autonomous_episode import make_training_env
    from training.formal_grpo import load_models

    base = load_proto()
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [row for row in rows if row.get("primary_10s")]
    far = []
    stored = {}
    for row in ok:
        path = TRAJ / f"{row['id']}__{row['strategy']}.npz"
        actions = np.load(path)["actions"].astype(np.float32)
        stored[(row["id"], row["strategy"])] = actions
        _recon, linf = quantize(actions, base)
        if np.any(linf > 0.25 + 1e-6):
            far.append(actions[linf > 0.25 + 1e-6])
    far_pts = np.concatenate(far, axis=0)
    extra = np.array([
        [-0.005, -0.005, 0.011, -0.001, 0.0, 0.0, -1.0],
        [0.021, -0.078, -0.456, -0.003, 0.0, 0.0, -1.0],
        [0.374, 0.165, 0.173, -0.003, 0.0, 0.0, -1.0],
        [-0.154, -0.352, 0.050, -0.002, 0.001, 0.0, -1.0],
        [-0.141, -0.176, -0.715, -0.003, 0.0, 0.0, -1.0],
        [-0.513, 0.291, 0.090, -0.003, 0.0, 0.0, -1.0],
        [0.032, 0.072, 0.266, -0.001, 0.0, 0.0, -1.0],
        [-0.095, -0.389, -0.246, -0.003, 0.0, 0.0, -1.0],
        [0.075, -0.291, -0.241, 0.0, 0.0, 0.0, 1.0],
        [0.355, 0.042, -0.408, -0.003, 0.0, 0.0, -1.0],
        [0.367, -0.134, -0.133, -0.003, 0.0, 0.0, -1.0],
    ], np.float32)
    candidate = np.concatenate([base, extra], axis=0)
    changed = []
    covered = 0
    for row in ok:
        actions = stored[(row["id"], row["strategy"])]
        old, old_linf = quantize(actions, base)
        new, new_linf = quantize(actions, candidate)
        same = bool(np.allclose(old, new, atol=1e-5))
        if float(new_linf.max()) <= 0.25 + 1e-6:
            covered += 1
        if not same or float(old_linf.max()) > 0.25 + 1e-6:
            changed.append(row)
    print("extra", np.round(extra, 3).tolist(), flush=True)
    print("covered", covered, "of", len(ok), "to_replay", len(changed), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ref = load_models(
        PROTO_PATH,
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
    for row in changed:
        actions = stored[(row["id"], row["strategy"])]
        recon, linf = quantize(actions, candidate)
        if float(linf.max()) > 0.25 + 1e-6:
            results.append({
                "id": row["id"],
                "strategy": row["strategy"],
                "family": row["family"],
                "old_10s": True,
                "old_k36": bool(row.get("k36_replay_10s")),
                "candidate_max_linf": float(linf.max()),
                "candidate_10s": None,
                "candidate_12s": None,
                "label": "STILL_OUTSIDE_0.25",
            })
            print("UNCOVERED", row["id"], row["strategy"], round(float(linf.max()), 3), flush=True)
            continue
        family = "slide" if row["family"] == "sliding" else "regrasp"
        replay = _replay_actions(env, blobs[row["id"]], recon, family, None)
        item = {
            "id": row["id"],
            "strategy": row["strategy"],
            "family": row["family"],
            "old_10s": True,
            "old_k36": bool(row.get("k36_replay_10s")),
            "candidate_max_linf": float(linf.max()),
            "candidate_10s": bool(replay.get("primary_10s")),
            "candidate_12s": None if not replay.get("hold") else bool(replay["hold"].get("held_12")),
            "label": replay.get("existing_label"),
        }
        results.append(item)
        print("REPLAY", item["id"], item["strategy"], item["label"], "linf", round(item["candidate_max_linf"], 3), flush=True)
    payload = {
        "base_k": 36,
        "extra": extra.tolist(),
        "candidate_k": int(candidate.shape[0]),
        "successes_inside_0.25": covered,
        "successes": len(ok),
        "replays": results,
        "note": "Existing 36 centers were not moved. Trajectories whose quantized actions were unchanged were not replayed again.",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
