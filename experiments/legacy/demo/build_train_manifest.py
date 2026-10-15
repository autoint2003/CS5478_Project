"""Manifest of TRAIN ICs, legal trajectories, and replay status. Does not train."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset/accepted/train"
V2 = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset_v2/accepted/train"
SEARCH = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl"
COV = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/coverage_all_holds.json"
K47 = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/replay_k47_all.jsonl"
K56 = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/replay_k56.jsonl"
TOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/candidate_tokens.jsonl"
OUT = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/train_manifest.jsonl"


def load_jsonl(path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    coverage = {f"{row['id']}|{row['strategy']}": row for row in json.loads(COV.read_text(encoding="utf-8"))["records"]}
    replay47 = {f"{row['id']}|{row['strategy']}": row for row in load_jsonl(K47)}
    replay56 = {f"{row['id']}|{row['strategy']}": row for row in load_jsonl(K56)}
    tokens = {f"{row['id']}|{row['strategy']}": row for row in load_jsonl(TOK)}
    by_ic = defaultdict(list)
    for row in load_jsonl(SEARCH):
        by_ic[row["id"]].append(row)
    lines = []
    verified_ics = []
    only_unrep = []
    fragile_ics = []
    both = []
    for path in sorted(V2.glob("*.pkl")):
        # id is in the filename ball_train_v2_NNNN.pkl
        episode_id = path.stem
        trials = by_ic.get(episode_id, [])
        trajectories = []
        verified = False
        fragile = False
        any_success = False
        any_unrep_success = False
        for row in trials:
            key = f"{episode_id}|{row['strategy']}"
            cov = coverage.get(key, {})
            play = replay47.get(key, {})
            play56 = replay56.get(key, {})
            token = tokens.get(key, {})
            success = bool(row.get("primary_10s"))
            verified_row = success and play.get("primary_10s") is True
            fragile_row = success and play.get("status") in {"REPLAYED", "PRIOR_REPLAY"} and play.get("primary_10s") is False
            unrep_row = success and play.get("status") == "NOT_REPRESENTABLE"
            if success:
                any_success = True
            if verified_row:
                verified = True
            if fragile_row:
                fragile = True
            if unrep_row:
                any_unrep_success = True
            trajectories.append({
                "strategy": row["strategy"],
                "family": row["family"],
                "label": row.get("label"),
                "primary_10s": success,
                "held_12": row.get("held_12"),
                "legacy_t12": row.get("legacy_t12"),
                "continuous_npz": f"results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories/{episode_id}__{row['strategy']}.npz" if success else None,
                "k36_inside": cov.get("k36_inside"),
                "k36_max_linf": cov.get("k36_max_linf"),
                "k47_inside": cov.get("k47_inside"),
                "k47_max_linf": cov.get("k47_max_linf"),
                "k47_replay_status": play.get("status"),
                "k47_replay_label": play.get("label"),
                "k47_primary_10s": play.get("primary_10s"),
                "k47_held_12": play.get("held_12"),
                "k47_legacy_t12": play.get("legacy_t12"),
                "k56_replay_status": play56.get("status"),
                "k56_replay_label": play56.get("label"),
                "k56_primary_10s": play56.get("primary_10s"),
                "none_intervals": token.get("none_intervals"),
                "candidate_k47_tokens": f"results/archive/demo_results/diagnostics/raw/legal_recovery_v2/trajectories/{episode_id}__{row['strategy']}__candidate_k47.npz" if success else None,
            })
        if verified:
            verified_ics.append(episode_id)
        if fragile and not verified:
            fragile_ics.append(episode_id)
        if fragile and verified:
            both.append(episode_id)
        if any_success and not verified and not fragile and any_unrep_success:
            only_unrep.append(episode_id)
        lines.append({
            "id": episode_id,
            "source": "ball_impact_v2",
            "snapshot": str(path.relative_to(ROOT)).replace("\\", "/"),
            "searched": bool(trials),
            "n_trials": len(trials),
            "n_success": sum(1 for item in trajectories if item["primary_10s"]),
            "verified_k47": verified,
            "fragile_k47": fragile,
            "trajectories": trajectories,
        })
    for path in sorted(V1.glob("*.pkl")):
        lines.append({
            "id": path.stem,
            "source": "ball_impact_v1",
            "snapshot": str(path.relative_to(ROOT)).replace("\\", "/"),
            "searched_in_v2_menu": False,
            "trajectories": [],
        })
    with OUT.open("w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row) + "\n")
    print("manifest", len(lines))
    print("v2", sum(1 for row in lines if row["source"] == "ball_impact_v2"))
    print("verified_ics", len(verified_ics))
    print("fragile_without_verified", len(fragile_ics), fragile_ics)
    print("verified_and_fragile", len(both), both)
    print("only_unrepresentable", len(only_unrep))
    no_success = [row["id"] for row in lines if row["source"] == "ball_impact_v2" and row["n_success"] == 0]
    print("no_success", len(no_success))


if __name__ == "__main__":
    main()
