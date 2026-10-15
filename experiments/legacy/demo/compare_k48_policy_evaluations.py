"""Evaluate the K=48 supervised checkpoint and the V1/V2 GRPO checkpoints.

Uses the existing deterministic closed-loop evaluator. Does not train and
does not write into the checkpoint files. FINAL snapshots are not opened.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = "config/training/k48/unified_k48_long_reward_v1.yaml"
COMPARE = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/eval_compare"
VALID_DIR = ROOT / "results/archive/demo_results/diagnostics/raw/ball_impact_dataset/accepted/validation"
JOBS = (
    ("supervised", "results/archive/demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt"),
    ("v1_u0312", "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v1/checkpoint_u0312.pt"),
    ("v1_u0624", "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v1/checkpoint_u0624.pt"),
    ("v1_u0936", "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v1/checkpoint_u0936.pt"),
    ("v2_u0312", "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2/checkpoint_u0312.pt"),
    ("v2_u0624", "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2/checkpoint_u0624.pt"),
    ("v2_u0936", "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2/checkpoint_u0936.pt"),
)


def command_for(name, checkpoint):
    return [
        sys.executable, "-u", "training/unified_k48.py",
        "--eval",
        "--config", CONFIG,
        "--checkpoint", checkpoint,
        "--eval-output", f"results/archive/demo_results/diagnostics/raw/unified_k48/eval_compare/{name}",
    ]


def _duration_stats(values):
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _rate(rows, handoff):
    group = [row for row in rows if row.get("handoff") == handoff]
    if not group:
        return {"n": 0, "success": 0, "rate": None}
    success = sum(1 for row in group if row.get("success"))
    return {"n": len(group), "success": success, "rate": success / len(group)}


def summarize_rows(rows):
    elapsed = [float(row["recovery_elapsed"]) for row in rows if row.get("recovery_elapsed") is not None]
    success_elapsed = [float(row["recovery_elapsed"]) for row in rows if row.get("success") and row.get("recovery_elapsed") is not None]
    return {
        "episodes": len(rows),
        "success": sum(1 for row in rows if row.get("success")),
        "reasons": dict(Counter(row["label"] for row in rows)),
        "handoff": dict(Counter(row.get("handoff") for row in rows)),
        "handoff_success": {
            "early": _rate(rows, "early"),
            "forced": _rate(rows, "forced"),
        },
        "recovery_elapsed_s": _duration_stats(elapsed),
        "success_recovery_elapsed_s": _duration_stats(success_elapsed),
    }


def load_eval(name):
    path = COMPARE / name / "heldout_train.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = summarize_rows(payload["rows"])
    stats["name"] = name
    stats["file"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return stats, payload["rows"]


def run_train_evals():
    COMPARE.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in JOBS:
        destination = COMPARE / name / "heldout_train.json"
        if destination.exists():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if len(payload.get("rows", [])) == 56:
                print("SKIP", name, flush=True)
                continue
        cmd = command_for(name, checkpoint)
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def evaluate_validation(name, checkpoint):
    from envs.simulation.yaml_config import load_yaml
    import torch
    from training.recovery_episode import make_training_env
    from envs.outcomes.horizon_outcome import NOMINAL_S, roll_policy
    from training.unified_k48 import SELECTED_LIMIT_S, load_trained_k48

    out_dir = COMPARE / f"valid_{name}"
    destination = out_dir / "heldout_valid.json"
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("rows"):
            print("SKIP_VALID", name, flush=True)
            return payload
    checkpoint_path = ROOT / checkpoint
    if "final" in checkpoint_path.name:
        raise RuntimeError("refusing a final checkpoint name")
    before = (checkpoint_path.stat().st_mtime, checkpoint_path.stat().st_size)
    cfg = load_yaml(ROOT / CONFIG)
    device_name = str(cfg.get("eval_device") or "cpu")
    device = torch.device(device_name)
    limit_s = float(cfg["recovery_limit_s"])
    if abs(limit_s - SELECTED_LIMIT_S) > 1e-9 or abs(float(cfg["nominal_seconds"]) - NOMINAL_S) > 1e-9:
        raise RuntimeError("validation eval settings differ from the 4 s / 12 s evaluator")
    paths = sorted(VALID_DIR.glob("ball_validation_*.pkl"))
    if not paths:
        raise RuntimeError("no validation snapshots")
    if any("final" in path.name for path in paths):
        raise RuntimeError("validation directory contains a final snapshot")
    model, book = load_trained_k48(checkpoint_path, device)
    env = make_training_env(device, model=model)
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            blob = pickle.load(handle)
        if blob["spec"].get("split") != "validation":
            raise RuntimeError(f"{path.name} is not a validation snapshot")
        out = roll_policy(
            env, model, book, blob, device,
            cap_steps=int(cfg.get("eval_cap_steps") or 200),
            limit_s=limit_s,
        )
        rows.append({
            "id": blob["spec"].get("id") or path.stem,
            "label": out["reason"],
            "success": bool(out.get("success")),
            "handoff": out.get("handoff"),
            "recovery_elapsed": out.get("recovery_elapsed"),
            "recovery_steps": out.get("recovery_steps"),
            "nominal_elapsed": out.get("nominal_elapsed"),
            "lift_completed": out.get("lift_completed"),
            "recovery_limit_s": limit_s,
            "nominal_seconds": NOMINAL_S,
            "device": device_name,
            "truncated": out["reason"] == "INVALID",
        })
        print("VALID", name, rows[-1]["id"], out["reason"], out.get("handoff"), flush=True)
    if checkpoint_path.stat().st_mtime != before[0] or checkpoint_path.stat().st_size != before[1]:
        raise RuntimeError("validation evaluation changed the checkpoint")
    payload = {"summary": summarize_rows(rows), "rows": rows, "checkpoint": checkpoint}
    out_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_report(train_stats, valid_payloads, winners):
    lines = [
        "# K=48 GRPO physical evaluation",
        "",
        "All seven checkpoints were evaluated with argmax decoding on CPU, a 4-second recovery limit, and the unchanged 12-second physical SUCCESS_12S check. The 56 training initial conditions are the fixed v2 cases whose numeric id is divisible by 5.",
        "",
        "| Checkpoint | Success | Early handoff | Forced handoff | Early success | Forced success | Recovery median (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stats in train_stats:
        early = stats["handoff_success"]["early"]
        forced = stats["handoff_success"]["forced"]
        handoff = stats["handoff"]
        median = stats["recovery_elapsed_s"]["median"]
        lines.append(
            f"| {stats['name']} | {stats['success']}/{stats['episodes']} | "
            f"{handoff.get('early', 0)} | {handoff.get('forced', 0)} | "
            f"{early['success']}/{early['n']} | {forced['success']}/{forced['n']} | {median:.3f} |"
        )
    lines.extend(["", "## Failure reasons", ""])
    for stats in train_stats:
        failures = {key: value for key, value in stats["reasons"].items() if key != "SUCCESS_12S"}
        lines.append(f"- {stats['name']}: {failures or 'none'}")
    lines.extend(["", f"Best training success count: {max(item['success'] for item in train_stats)}/{train_stats[0]['episodes']}.",
                  f"Selected by physical success: {', '.join(winners)}.", ""])
    if valid_payloads:
        lines.append("## Validation")
        lines.append("")
        lines.append("Twenty-six v1 validation snapshots were evaluated with the same controller and success rule. Final-split files were not opened.")
        lines.append("")
        for name, payload in valid_payloads.items():
            summary = payload["summary"]
            lines.append(f"- {name}: {summary['success']}/{summary['episodes']} {summary['reasons']}")
    report = ROOT / "results/archive/demo_results/diagnostics/K48_GRPO_EVAL_COMPARISON.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (COMPARE / "comparison.json").write_text(json.dumps({
        "train": train_stats,
        "winners": winners,
        "validation": {name: payload["summary"] for name, payload in valid_payloads.items()},
        "commands": [" ".join(command_for(name, checkpoint)) for name, checkpoint in JOBS],
    }, indent=2), encoding="utf-8")
    print("REPORT", report, flush=True)


def main():
    run_train_evals()
    train_stats = []
    for name, _checkpoint in JOBS:
        stats, _rows = load_eval(name)
        train_stats.append(stats)
        print("TRAIN", name, f"{stats['success']}/{stats['episodes']}", stats["reasons"], flush=True)
    best = max(item["success"] for item in train_stats)
    winners = [item["name"] for item in train_stats if item["success"] == best]
    selected = list(winners)
    if "supervised" not in selected:
        selected.append("supervised")
    valid_payloads = {}
    by_checkpoint = dict(JOBS)
    for name in selected:
        valid_payloads[name] = evaluate_validation(name, by_checkpoint[name])
    write_report(train_stats, valid_payloads, winners)


if __name__ == "__main__":
    main()
