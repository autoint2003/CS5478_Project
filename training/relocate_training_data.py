"""Copy recatch and controlled-sliding training rows into data/.

The diagnostic tree is left in place. data/supervised_targets.jsonl is the
training entry and also includes the inward rows already stored under data/.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OLD = ROOT / "demo_results/diagnostics/raw/k48_recovery_v3"
DATA = ROOT / "data"
RECATCH = {"chase_lost", "release_long", "release_short"}
SLIDE = {"controlled_slide"}


def _rows(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _copy(src: Path, dst: Path):
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _category(strategy: str) -> str:
    if strategy in RECATCH:
        return "recatch"
    if strategy in SLIDE:
        return "controlled_sliding"
    if strategy == "immediate_handoff":
        return "immediate_handoff"
    raise KeyError(strategy)


def _place(row: dict) -> dict:
    strategy = row.get("strategy")
    if strategy == "inward_reseat":
        return row
    split = "valid" if row.get("split") == "valid" else "train"
    category = _category(strategy)
    folder = DATA / category / split
    stem = Path(row["continuous_npz"]).name
    token_name = Path(row["token_residual_npz"]).name
    continuous = folder / stem
    tokens = folder / token_name
    _copy(ROOT / row["continuous_npz"], continuous)
    if row["token_residual_npz"] != row["continuous_npz"]:
        _copy(ROOT / row["token_residual_npz"], tokens)
    else:
        tokens = continuous
    snap_src = ROOT / row["snapshot"]
    snap_dst = DATA / "snapshots" / split / f"{row['id']}.pkl"
    _copy(snap_src, snap_dst)
    out = dict(row)
    out["split"] = split
    out["continuous_npz"] = _rel(continuous)
    out["token_residual_npz"] = _rel(tokens)
    out["snapshot"] = _rel(snap_dst)
    out["source_snapshot"] = row["snapshot"]
    return out


def _weights(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[row.get("mechanism") or "unknown"][row["id"]].append(index)
    weights = {}
    for _mechanism, episodes in grouped.items():
        share = 1.0 / max(len(episodes), 1)
        for episode_id, indexes in episodes.items():
            piece = share / len(indexes)
            for index in indexes:
                row = rows[index]
                weights[(row["id"], row.get("continuous_npz"))] = piece
    return weights


def _inward_rows():
    rows = []
    for row in _rows(DATA / "inward_sliding" / "manifest.jsonl"):
        packed = __import__("numpy").load(ROOT / row["trajectory"])
        label = "early_handoff" if "handoff_obs" in packed.files else "recorded_end"
        rows.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "strategy": "inward_reseat",
            "family": "sliding",
            "mechanism": "gravity_inward",
            "continuous_npz": row["trajectory"],
            "token_residual_npz": row["tokens"],
            "handoff_label": label,
            "source_kind": "inward_sliding",
            "nominal_class": "RECOVERY_CANDIDATE",
            "verified_success": True,
            "success": True,
            "reason": "SUCCESS_12S",
            "split": row["split"],
            "provenance": "data/inward_sliding",
        })
    return rows


def _dump(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    train_src = _rows(OLD / "supervised_targets.jsonl")
    valid_src = _rows(OLD / "valid_demonstrations.jsonl")
    placed = [_place(row) for row in train_src + valid_src]
    inward = _inward_rows()
    recovery = [row for row in placed + inward if row.get("nominal_class") == "RECOVERY_CANDIDATE" and row.get("split") != "valid"]
    weights = _weights(recovery)
    train = []
    valid = []
    for row in placed + inward:
        if row.get("nominal_class") == "NOMINAL_SOLVABLE":
            row["sample_weight"] = 0.0
            row["split"] = "train"
            train.append(row)
            continue
        if row.get("split") == "valid":
            valid.append(row)
            continue
        row["sample_weight"] = weights[(row["id"], row["continuous_npz"])]
        row["verified_success"] = True
        row["nominal_class"] = "RECOVERY_CANDIDATE"
        row["split"] = "train"
        train.append(row)
    _dump(DATA / "supervised_targets.jsonl", train)
    _dump(DATA / "valid_demonstrations.jsonl", valid)
    for category in ("recatch", "controlled_sliding", "immediate_handoff"):
        subset = [row for row in placed if _category(row.get("strategy")) == category]
        _dump(DATA / category / "manifest.jsonl", subset)
    snap_for = {}
    for row in train + valid:
        snap_for[(row["split"], row["id"])] = row["snapshot"]
    grpo = []
    for row in _rows(OLD / "grpo_initial_conditions.jsonl"):
        item = dict(row)
        split = "valid" if row.get("split") == "valid" else "train"
        key = (split, row["id"])
        if key not in snap_for:
            dst = DATA / "snapshots" / split / f"{row['id']}.pkl"
            _copy(ROOT / row["snapshot"], dst)
            snap_for[key] = _rel(dst)
        item["snapshot"] = snap_for[key]
        grpo.append(item)
    _dump(DATA / "grpo_initial_conditions.jsonl", grpo)
    have_valid = {row["id"] for row in _rows(OLD / "valid_recovery.jsonl")}
    valid_recovery = []
    for row in _rows(OLD / "valid_recovery.jsonl"):
        item = dict(row)
        dst = DATA / "snapshots" / "valid" / f"{row['id']}.pkl"
        if not dst.exists():
            _copy(ROOT / row["snapshot"], dst)
        item["snapshot"] = _rel(dst)
        valid_recovery.append(item)
    for row in valid:
        if row["id"] in have_valid:
            continue
        have_valid.add(row["id"])
        valid_recovery.append({
            "id": row["id"],
            "snapshot": row["snapshot"],
            "pool": "recovery",
            "nominal_class": "RECOVERY_CANDIDATE",
            "split": "valid",
            "horizon": "recovery_limit_4s_then_nominal_12s",
            "recovery_limit_s": 4.0,
            "success": False,
            "provenance": row.get("provenance"),
            "strategy": row.get("strategy"),
        })
    _dump(DATA / "valid_recovery.jsonl", valid_recovery)
    summary = {
        "supervised_train": len(train),
        "valid_demonstrations": len(valid),
        "grpo": len(grpo),
        "valid_recovery": len(valid_recovery),
        "by_strategy_train": {},
    }
    from collections import Counter
    summary["by_strategy_train"] = dict(Counter(row.get("strategy") for row in train))
    summary["by_strategy_valid"] = dict(Counter(row.get("strategy") for row in valid))
    (DATA / "entry.json").write_text(json.dumps({
        "supervised_targets": "data/supervised_targets.jsonl",
        "valid_demonstrations": "data/valid_demonstrations.jsonl",
        "grpo_initial_conditions": "data/grpo_initial_conditions.jsonl",
        "eval_manifest": "data/valid_recovery.jsonl",
        "categories": {
            "regrasp": "data/regrasp/manifest.jsonl",
            "recatch": "data/recatch/manifest.jsonl",
            "controlled_sliding": "data/controlled_sliding/manifest.jsonl",
            "inward_sliding": "data/inward_sliding/manifest.jsonl",
            "immediate_handoff": "data/immediate_handoff/manifest.jsonl",
        },
        "counts": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
