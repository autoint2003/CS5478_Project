"""Write the selected K=47 book, the two dataset views, and check the decoder.

Does not train.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2"
MANIFEST = RAW / "train_manifest.jsonl"
PROTO_PATH = ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt"


def mechanism(traj):
    multi = int(traj.get("none_intervals") or 0) >= 2
    if traj.get("family") == "sliding" and not multi:
        return "contact_preserving_slide"
    if traj.get("family") == "sliding":
        return "sliding"
    if multi:
        return "repeated_recatch"
    if traj.get("strategy") == "chase_lost":
        return "airborne_chase"
    return "release_regrasp"


def main():
    saved = torch.load(PROTO_PATH, map_location="cpu", weights_only=False)
    proto = np.asarray(saved["proto"], np.float32)
    extra = np.load(RAW / "candidate_extra.npy").astype(np.float32)
    book = np.concatenate([proto, extra], axis=0)
    if book.shape != (47, 7):
        raise RuntimeError(f"selected book shape {book.shape}")
    if not np.array_equal(book[:36], proto):
        raise RuntimeError("tokens 0-35 differ from the K=36 checkpoint")
    np.save(RAW / "codebook_selected_k47.npy", book)
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    grpo_path = RAW / "grpo_initial_conditions.jsonl"
    sup_path = RAW / "supervised_targets.jsonl"
    fragile_path = RAW / "fragile_trajectories.jsonl"
    supervised = []
    fragile = []
    with grpo_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "id": row["id"],
                "source": row["source"],
                "snapshot": row["snapshot"],
                "n_success": row.get("n_success", 0),
                "verified_k47": bool(row.get("verified_k47")),
                "unsolved": row.get("source") == "ball_impact_v2" and row.get("n_success", 0) == 0,
            }) + "\n")
            for traj in row.get("trajectories") or []:
                if not traj.get("primary_10s"):
                    continue
                record = {
                    "id": row["id"],
                    "snapshot": row["snapshot"],
                    "strategy": traj["strategy"],
                    "family": traj["family"],
                    "mechanism": mechanism(traj),
                    "continuous_npz": traj.get("continuous_npz"),
                    "token_residual_npz": traj.get("candidate_k47_tokens"),
                    "none_intervals": traj.get("none_intervals"),
                    "steps_are_one_episode": True,
                    "original_label": traj.get("label"),
                    "held_12": traj.get("held_12"),
                    "legacy_t12": traj.get("legacy_t12"),
                    "k47_replay_label": traj.get("k47_replay_label"),
                    "k47_primary_10s": traj.get("k47_primary_10s"),
                }
                if traj.get("k47_primary_10s") is True:
                    record["supervisory_target"] = True
                    supervised.append(record)
                elif traj.get("k47_inside") and traj.get("k47_primary_10s") is False:
                    record["supervisory_target"] = False
                    record["sensitivity"] = "decoded_replay_failed_inside_cap"
                    fragile.append(record)
    counts = Counter(item["mechanism"] for item in supervised)
    for item in supervised:
        item["sample_weight"] = round(1.0 / counts[item["mechanism"]], 6)
    with sup_path.open("w", encoding="utf-8") as handle:
        for item in supervised:
            handle.write(json.dumps(item) + "\n")
    with fragile_path.open("w", encoding="utf-8") as handle:
        for item in fragile:
            handle.write(json.dumps(item) + "\n")
    from models.k48_policy import build_policy
    from experiments.legacy.continual_skill_addition import ContinualPolicy
    from training.k36_checkpoint import load_models

    model = build_policy(book, torch.device("cpu"), 0.25)
    logits, residual, action = model(torch.zeros(4, 39), torch.zeros(4, 7))
    if tuple(logits.shape) != (4, 47):
        raise RuntimeError(f"logit shape {tuple(logits.shape)}")
    if tuple(residual.shape) != (4, 7) or tuple(action.shape) != (4, 7):
        raise RuntimeError("residual or action shape")
    if float(residual.abs().max()) > 0.25 + 1e-6:
        raise RuntimeError("residual left the cap")
    legacy = ContinualPolicy(book)
    if legacy.logits_old.out_features != 35 or legacy.proto.shape != (47, 7):
        raise RuntimeError("legacy head check failed")
    fake = RAW / "_reject_k47_init.pt"
    torch.save({"proto": book, "res_cap": 0.25, "model": {}}, fake)
    rejected = False
    try:
        load_models(
            fake,
            ROOT / "results/archive/demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt",
            torch.device("cpu"),
        )
    except RuntimeError as exc:
        rejected = "expected K=36" in str(exc)
    finally:
        fake.unlink(missing_ok=True)
    if not rejected:
        raise RuntimeError("formal_grpo accepted a K=47 initialization")
    case = [item for item in supervised if item["id"] == "ball_train_v2_0021" and item["strategy"] == "release_long"]
    print("grpo", sum(1 for _ in grpo_path.open(encoding="utf-8")))
    print("supervised", len(supervised), dict(counts))
    print("fragile", len(fragile))
    print("0021", len(case), case[0]["mechanism"] if case else None, case[0]["none_intervals"] if case else None)
    print("decoder_ok", True)


if __name__ == "__main__":
    main()
