"""Merge Heuristic GRPO ICs with Hard Mining train/dev/test manifests.

Writes under results/grpo/adaptive_g/datasets/. Does not modify source data.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.datasets.skill_quotas import _fingerprint
from training.adaptive_hybrid import HORIZONTAL, SLIDING, VERTICAL, load_train_snapshots, primary_category
from training.adaptive_independent_g import assert_no_train_valid_overlap

OUT = ROOT / "results" / "grpo" / "adaptive_g" / "datasets"
HARD_DIR = ROOT / "results" / "diagnostics" / "hard_ic_mining"
VALID_PATH = ROOT / "data" / "valid_skill_quotas.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _physics_params(blob: dict) -> dict | None:
    spec = blob.get("spec")
    if not isinstance(spec, dict):
        return None
    keys = ("azimuth_deg", "speed_mps", "phi_deg", "z_off_m", "delay_s", "horizon", "ball_mass")
    out = {key: spec[key] for key in keys if key in spec}
    return out or dict(spec)


def _mechanism_evidence(row: dict, source: str) -> dict:
    if source == "hard":
        return {
            "oracle_mechanism": row.get("oracle_mechanism"),
            "oracle_strategy": row.get("oracle_strategy"),
            "oracle_found": row.get("oracle_found"),
            "sft_det_mechanism": row.get("sft_det_mechanism"),
            "sft_det_reason": row.get("sft_det_reason"),
            "sft_det_success": row.get("sft_det_success"),
            "sft_stoch_rate": row.get("sft_stoch_rate"),
            "sft_stoch_successes": row.get("sft_stoch_successes"),
            "note": "mechanism labels are trajectory evidence, not a skill selector",
        }
    return {
        "categories": list(row.get("categories") or []),
        "note": "heuristic skill quotas; execution mechanism classified at rollout time",
    }


def _enrich_heuristic(row: dict, split: str) -> dict:
    snap = str(row["snapshot"]).replace("\\", "/")
    blob = pickle.loads((ROOT / snap).read_bytes())
    fp = _fingerprint(blob)
    cats = sorted(set(row.get("categories") or []))
    return {
        "id": row["id"],
        "ic_id": row["id"],
        "snapshot": snap,
        "snapshot_path": snap,
        "source_dataset": "heuristic",
        "source_episode": row["id"],
        "original_grasp_configuration": _grasp_from_categories(cats),
        "mechanism_evidence": _mechanism_evidence(row, "heuristic"),
        "recoverability_status": "heuristic_verified",
        "difficulty": "heuristic",
        "split": split,
        "physics_parameters": _physics_params(blob),
        "fingerprint": fp,
        "categories": cats,
        "primary_category": primary_category(cats),
        "nominal_class": row.get("nominal_class") or "RECOVERY_CANDIDATE",
        "pool": row.get("pool") or "recovery",
        "tier": "heuristic",
    }


def _grasp_from_categories(cats: list[str]) -> list[str]:
    """Horizontal/Vertical = original grasp configuration; Sliding is a skill label."""
    grasp = []
    if HORIZONTAL in cats:
        grasp.append("horizontal")
    if VERTICAL in cats:
        grasp.append("vertical")
    if SLIDING in cats and not grasp:
        grasp.append("vertical_or_unspecified_sliding")
    if SLIDING in cats and VERTICAL in cats:
        pass  # vertical grasp with sliding skill evidence
    if not grasp and SLIDING in cats:
        grasp.append("sliding_labeled")
    return grasp or ["unspecified"]


def _enrich_hard(row: dict, split: str) -> dict:
    snap = str(row["snapshot"]).replace("\\", "/")
    blob = pickle.loads((ROOT / snap).read_bytes())
    fp = row.get("fingerprint") or _fingerprint(blob)
    recomputed = _fingerprint(blob)
    if row.get("fingerprint") and row["fingerprint"] != recomputed:
        raise RuntimeError(f"fingerprint mismatch for {row['id']}")
    cats = sorted(set(row.get("categories") or []))
    tier = str(row.get("tier") or "")
    if tier == "retention_solved":
        difficulty = "retention"
        recoverability = "retention_sft_success"
    elif tier == "hard_verified":
        difficulty = "hard_verified"
        recoverability = "hard_verified"
    else:
        difficulty = tier or "hard"
        recoverability = tier or "hard"
    return {
        "id": row["id"],
        "ic_id": row["id"],
        "snapshot": snap,
        "snapshot_path": snap,
        "source_dataset": "hard",
        "source_episode": row.get("id"),
        "original_grasp_configuration": _grasp_from_categories(cats),
        "mechanism_evidence": _mechanism_evidence(row, "hard"),
        "recoverability_status": recoverability,
        "difficulty": difficulty,
        "split": split,
        "physics_parameters": {
            "azimuth_deg": row.get("azimuth_deg"),
            "speed_mps": row.get("speed_mps"),
            "phi_deg": row.get("phi_deg"),
            "z_off_m": row.get("z_off_m"),
            "delay_s": row.get("delay_s"),
            "horizon": row.get("horizon"),
            **(_physics_params(blob) or {}),
        },
        "fingerprint": fp,
        "categories": cats,
        "primary_category": primary_category(cats),
        "nominal_class": row.get("nominal_class") or "RECOVERY_CANDIDATE",
        "pool": row.get("pool") or "recovery",
        "tier": tier,
        "assigned_split": row.get("assigned_split") or row.get("split"),
        "hard_by_sft": row.get("hard_by_sft"),
        "provenance": row.get("provenance"),
    }


def _merge_unique(primary: list[dict], secondary: list[dict], prefer: str, audit: list[str]) -> list[dict]:
    """Union by fingerprint then snapshot. Prefer rows from ``prefer`` source on conflict."""
    by_fp: dict[str, dict] = {}
    by_snap: dict[str, dict] = {}
    order: list[str] = []

    def add(row: dict, origin: str) -> None:
        fp = row["fingerprint"]
        snap = row["snapshot"]
        if fp in by_fp:
            existing = by_fp[fp]
            if existing["snapshot"] != snap:
                audit.append(
                    f"fingerprint collision {fp[:12]}…: {existing['id']} ({existing['snapshot']}) "
                    f"vs {row['id']} ({snap}); kept {prefer}-preferred / first"
                )
            if existing.get("source_dataset") == prefer or origin != prefer:
                # keep existing unless new is preferred and old is not
                if origin == prefer and existing.get("source_dataset") != prefer:
                    audit.append(f"replaced {existing['id']} with preferred {row['id']} (fp)")
                    by_fp[fp] = row
                    by_snap[snap] = row
                    if existing["snapshot"] in by_snap and by_snap[existing["snapshot"]] is existing:
                        del by_snap[existing["snapshot"]]
                else:
                    audit.append(f"dedupe drop {row['id']} (fp match {existing['id']})")
                return
        if snap in by_snap and by_snap[snap]["fingerprint"] != fp:
            audit.append(f"snapshot collision {snap}: {by_snap[snap]['id']} vs {row['id']}")
            if origin != prefer and by_snap[snap].get("source_dataset") == prefer:
                audit.append(f"dedupe drop {row['id']} (snap, prefer {prefer})")
                return
        if fp not in by_fp:
            order.append(fp)
        by_fp[fp] = row
        by_snap[snap] = row

    for row in primary:
        add(row, row.get("source_dataset") or "primary")
    for row in secondary:
        add(row, row.get("source_dataset") or "secondary")
    return [by_fp[fp] for fp in order if fp in by_fp]


def _stats(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "by_source": dict(Counter(r["source_dataset"] for r in rows)),
        "by_difficulty": dict(Counter(r.get("difficulty") for r in rows)),
        "by_recoverability": dict(Counter(r.get("recoverability_status") for r in rows)),
        "by_primary": dict(Counter(r["primary_category"] for r in rows)),
        "by_category_coverage": {
            name: sum(1 for r in rows if name in r["categories"])
            for name in (HORIZONTAL, VERTICAL, SLIDING)
        },
        "ids": [r["id"] for r in rows],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: list[str] = []

    heuristic_train_raw = load_train_snapshots()
    heuristic_valid_raw = _read_jsonl(VALID_PATH)
    hard_train_raw = _read_jsonl(HARD_DIR / "selected_train_manifest.jsonl")
    hard_dev_raw = _read_jsonl(HARD_DIR / "selected_dev_manifest.jsonl")
    hard_test_raw = _read_jsonl(HARD_DIR / "selected_test_manifest.jsonl")

    audit.append(f"heuristic train unique snapshots (adaptive load_train_snapshots): {len(heuristic_train_raw)}")
    audit.append(f"heuristic validation (valid_skill_quotas): {len(heuristic_valid_raw)}")
    audit.append(
        f"hard selected train/dev/test: {len(hard_train_raw)}/{len(hard_dev_raw)}/{len(hard_test_raw)}"
    )
    audit.append(
        "note: classic data/grpo_initial_conditions.jsonl has 87 unique ICs; "
        "adaptive training historically used skill-quota unique snapshots (~99). "
        "This merge uses the adaptive heuristic set for continuity with Tree80/adaptive loaders."
    )

    heuristic_train = [_enrich_heuristic(row, "train") for row in heuristic_train_raw]
    heuristic_valid = []
    for row in heuristic_valid_raw:
        enriched = _enrich_heuristic(row, "validation")
        enriched["split"] = "validation"
        heuristic_valid.append(enriched)

    hard_train = [_enrich_hard(row, "train") for row in hard_train_raw]
    hard_dev = [_enrich_hard(row, "dev") for row in hard_dev_raw]
    for row in hard_dev:
        row["split"] = "validation"
    hard_test = [_enrich_hard(row, "test") for row in hard_test_raw]
    for row in hard_test:
        row["split"] = "test"

    # Retention hard ICs are newly mined snapshots (SFT-solvable), not copies of heuristic tapes.
    # Still dedupe by fingerprint/snapshot against heuristic.
    merged_train = _merge_unique(heuristic_train, hard_train, prefer="heuristic", audit=audit)
    merged_validation = _merge_unique(heuristic_valid, hard_dev, prefer="heuristic", audit=audit)

    # Cross-split: validation wins; strip from train.
    valid_fp = {r["fingerprint"] for r in merged_validation}
    valid_snap = {r["snapshot"] for r in merged_validation}
    kept_train = []
    for row in merged_train:
        if row["fingerprint"] in valid_fp or row["snapshot"] in valid_snap:
            audit.append(
                f"CONFLICT train∩validation: removed train {row['id']} "
                f"(prefer validation ownership)"
            )
            continue
        kept_train.append(row)
    merged_train = kept_train

    # Test must not leak into train/val.
    test_fp = {r["fingerprint"] for r in hard_test}
    test_snap = {r["snapshot"] for r in hard_test}
    for label, rows in (("train", merged_train), ("validation", merged_validation)):
        leaked = [r["id"] for r in rows if r["fingerprint"] in test_fp or r["snapshot"] in test_snap]
        if leaked:
            raise RuntimeError(f"hard test leaked into {label}: {leaked}")

    assert_no_train_valid_overlap(merged_train, merged_validation)

    # Near-duplicate note: identical fingerprints already collapsed. Soft near-dup
    # (similar but non-identical physics) is not claimed as resolved.
    audit.append(
        "near-duplicate: exact fingerprint (qpos/qvel/ctrl/p_des rounded) and snapshot path "
        "used for identity. Soft geometric near-duplicates without shared fingerprint are "
        "NOT deterministically judged; see hard mining REPORT known-fingerprint exclusion."
    )

    _write_jsonl(OUT / "merged_train.jsonl", merged_train)
    _write_jsonl(OUT / "merged_validation.jsonl", merged_validation)
    _write_jsonl(OUT / "hard_test.jsonl", hard_test)

    stats = {
        "heuristic_train": _stats(heuristic_train),
        "heuristic_validation": _stats(heuristic_valid),
        "hard_train": _stats(hard_train),
        "hard_dev": _stats(hard_dev),
        "hard_test": _stats(hard_test),
        "merged_train": _stats(merged_train),
        "merged_validation": _stats(merged_validation),
        "original_17_ids": [r["id"] for r in heuristic_valid],
        "merge_rules": {
            "merged_train": "heuristic_train ∪ hard_train (deduped)",
            "merged_validation": "heuristic_validation ∪ hard_dev (deduped)",
            "hard_test": "held out; not used for train or checkpoint selection",
            "conflict_policy": "validation ownership preferred; strip from train",
            "identity": "fingerprint + snapshot path",
        },
    }
    (OUT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    lines = [
        "# GRPO IC merge audit",
        "",
        "## Phase 0 interface notes (implementation)",
        "",
        "- Heuristic train loader: `training.adaptive_hybrid.load_train_snapshots` (skill-quota unique snapshots).",
        "- Heuristic validation: `data/valid_skill_quotas.jsonl` (17 ICs).",
        "- Hard mining: `results/diagnostics/hard_ic_mining/selected_{train,dev,test}_manifest.jsonl`.",
        "- GRPO reuse: Transformer policy, K48 sampling, `independent_rollout`, `reward_v2`, "
        "`apply_group_update` (equal-trajectory mean), checkpoints.",
        "- Formal training entry switches to Adaptive Independent-G (`exploration: independent_g`); "
        "Tree80 code retained but not used by the new config.",
        "- Unchanged: MuJoCo physics, nominal controller, 39D obs, K48 codebook, `step_legal`, "
        "`reward_v2`, HANDOFF / SUCCESS_12S, protected SFT checkpoint bytes.",
        "",
        "## Counts",
        "",
        f"- Heuristic train unique: **{len(heuristic_train)}**",
        f"- Heuristic validation unique: **{len(heuristic_valid)}**",
        f"- Hard train / dev / test: **{len(hard_train)}** / **{len(hard_dev)}** / **{len(hard_test)}**",
        f"- Merged train unique: **{len(merged_train)}**",
        f"- Merged validation unique: **{len(merged_validation)}**",
        f"- Hard test (holdout): **{len(hard_test)}**",
        "",
        "## Source coverage (merged train)",
        "",
        f"```json\n{json.dumps(stats['merged_train']['by_source'], indent=2)}\n```",
        "",
        f"Primary categories: `{stats['merged_train']['by_primary']}`",
        f"Category coverage: `{stats['merged_train']['by_category_coverage']}`",
        "",
        "## Audit log",
        "",
    ]
    lines.extend(f"- {item}" for item in audit)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{OUT / 'merged_train.jsonl'}`",
            f"- `{OUT / 'merged_validation.jsonl'}`",
            f"- `{OUT / 'hard_test.jsonl'}`",
            f"- `{OUT / 'dataset_statistics.json'}`",
            "",
        ]
    )
    (OUT / "merge_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "merged_train": len(merged_train),
        "merged_validation": len(merged_validation),
        "hard_test": len(hard_test),
        "out": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
