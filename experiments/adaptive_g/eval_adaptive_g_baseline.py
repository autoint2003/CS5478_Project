"""Frozen Balanced-SFT baseline on merged Adaptive-G validation.

Uses the same deterministic roll_policy_full path as GRPO epoch validation.
Writes results/grpo/adaptive_g/full_run_v1/baseline_sft/.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.k48_checkpoint import load_trained_k48
from training.adaptive_grpo import _summarize_validation, _prepare_blob
from training.adaptive_independent_g import load_merged_rows
from training.k47_codebook import load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from evaluation.classify_validation_skills import classify_mechanism, prepare_blob
from evaluation import skill_traces as audit

SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
VALID = ROOT / "results/grpo/adaptive_g/datasets/merged_validation.jsonl"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
OUT = ROOT / "results/grpo/adaptive_g/full_run_v1/baseline_sft"


def eval_rows(env, model, book, device, rows, limit_s, seated_ref):
    per_ic = []
    for row in rows:
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            if seated_ref is not None:
                try:
                    blob = prepare_blob(blob, seated_ref)
                except Exception:
                    blob = _prepare_blob(blob, seated_ref)
            else:
                blob = _prepare_blob(blob, seated_ref)
        print("BASELINE_IC", row["id"], row.get("source_dataset"), flush=True)
        if "capture" in blob:
            samples, t0, traced, handoff_t = audit.roll_policy_full(
                env, model, book, blob, device, limit_s,
            )
        else:
            from evaluation.classify_validation_skills import roll_policy_full_ballistic

            samples, t0, traced, handoff_t = roll_policy_full_ballistic(
                env, model, book, blob, device, limit_s,
            )
        skill = classify_mechanism(samples, t0, handoff_t)
        per_ic.append({
            "id": row["id"],
            "source_dataset": row.get("source_dataset"),
            "difficulty": row.get("difficulty"),
            "categories": list(row.get("categories") or []),
            "primary_category": row.get("primary_category"),
            "reason": traced.get("reason"),
            "success": bool(traced.get("success")),
            "mechanism": skill.get("mechanism"),
            "mechanism_confidence": skill.get("confidence"),
            "mechanism_why": "; ".join(skill.get("why") or []),
            "handoff": handoff_t,
        })
    return per_ic


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_merged_rows(VALID)
    book = load_codebook(CODEBOOK)
    model, saved = load_trained_k48(SFT, device)
    if saved.shape != book.shape:
        raise RuntimeError("codebook mismatch")
    model.eval()
    env = make_training_env(device, model=model)
    seated_ref = None
    from data_generation.strategies.recatch_oracle import _setup

    _sim, _base, seated_ref, _target, _code = _setup()
    started = time.perf_counter()
    per_ic = eval_rows(env, model, book, device, rows, SELECTED_LIMIT_S, seated_ref)
    summary = _summarize_validation(per_ic)
    payload = {
        "checkpoint": str(SFT),
        "wall_s": time.perf_counter() - started,
        "summary": summary,
        "per_ic": per_ic,
    }
    (OUT / "baseline_sft.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Compact table
    lines = [
        "# Balanced SFT baseline (merged validation)",
        "",
        f"- Checkpoint: `{SFT}`",
        f"- Heuristic 17: **{summary['heuristic']['success']}/{summary['heuristic']['n']}**",
        f"- Hard dev 6: **{summary['hard_dev']['success']}/{summary['hard_dev']['n']}**",
        f"- Merged 23: **{summary['all']['success']}/{summary['all']['n']}**",
        "",
        "| id | source | SUCCESS_12S | reason | mechanism | handoff_t |",
        "|---|---|---|---|---|---|",
    ]
    for row in per_ic:
        lines.append(
            f"| {row['id']} | {row.get('source_dataset')} | {int(row['success'])} | "
            f"{row.get('reason')} | {row.get('mechanism')} | {row.get('handoff')} |"
        )
    (OUT / "baseline_sft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "heuristic": f"{summary['heuristic']['success']}/{summary['heuristic']['n']}",
        "hard_dev": f"{summary['hard_dev']['success']}/{summary['hard_dev']['n']}",
        "merged": f"{summary['all']['success']}/{summary['all']['n']}",
        "out": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
