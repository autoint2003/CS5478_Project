"""Closed-loop K48 evaluation. Physics success still comes from roll_policy."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from training.autonomous_episode import make_training_env
from envs.horizon_outcome import roll_policy
from training.unified_k47 import read_jsonl
from training.unified_k48 import (
    HANDOFF,
    LOGIT_K,
    NOMINAL_S,
    ROOT,
    SELECTED_LIMIT_S,
    assert_recovery_v3_paths,
    load_trained_k48,
)


def evaluate(cfg, checkpoint, eval_data=""):
    checkpoint_path = Path(checkpoint)
    before = checkpoint_path.stat().st_mtime
    size_before = checkpoint_path.stat().st_size
    device_name = str(cfg.get("eval_device") or "cpu")
    device = torch.device(device_name)
    model, book = load_trained_k48(checkpoint_path, device)
    if model.handoff_index != HANDOFF or model.logits.out_features != LOGIT_K:
        raise RuntimeError("loaded policy is not the 48-token handoff head")
    limit_s = float(cfg.get("recovery_limit_s") or SELECTED_LIMIT_S)
    if abs(limit_s - SELECTED_LIMIT_S) > 1e-9 or abs(float(cfg.get("nominal_seconds") or NOMINAL_S) - NOMINAL_S) > 1e-9:
        raise RuntimeError("evaluation must use a 4 s recovery limit and a 12 s nominal window")
    env = make_training_env(device, model=model)
    rebuilt = assert_recovery_v3_paths(cfg)
    if rebuilt:
        default_manifest = str(cfg["eval_manifest"]).replace("\\", "/")
        requested = str(eval_data).replace("\\", "/") if eval_data else default_manifest
        explicit = requested != default_manifest
        baseline_path = ROOT / requested
        baseline = read_jsonl(baseline_path)
        if not baseline:
            raise RuntimeError("evaluation manifest is empty")
        if any("final" in str(row.get("snapshot", "")) for row in baseline):
            raise RuntimeError("evaluation manifest points at FINAL")
        if explicit:
            bad = [
                row["id"] for row in baseline
                if row.get("nominal_class") != "RECOVERY_CANDIDATE" or row.get("split") not in ("train", "valid")
            ]
            if bad:
                raise RuntimeError(f"evaluation list is not a recovery pool: {bad[:5]}")
        elif any(row.get("split") != "valid" or row.get("nominal_class") != "RECOVERY_CANDIDATE" for row in baseline):
            raise RuntimeError("evaluation manifest is not the valid recovery pool")
        baseline_ids = [row["id"] for row in baseline]
        by_id = {row["id"]: row for row in baseline}
        allowed_split = {"valid"}
    else:
        baseline_path = ROOT / cfg["baseline_eval"]
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_ids = [row["id"] for row in baseline]
        if len(baseline_ids) != 56 or len(set(baseline_ids)) != 56:
            raise RuntimeError("K=47 baseline is not the 56-IC evaluation")
        by_id = {
            row["id"]: row
            for row in read_jsonl(ROOT / cfg["grpo_initial_conditions"])
            if row.get("source") == "ball_impact_v2" and int(row["id"].rsplit("_", 1)[-1]) % 5 == 0
        }
        if set(by_id) != set(baseline_ids):
            raise RuntimeError("K=48 evaluation ICs differ from the 56 K=47 baseline ICs")
        allowed_split = {None, "train"}
    rows = []
    for episode_id in baseline_ids:
        row = by_id[episode_id]
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        allowed = {row.get("split")} if rebuilt and explicit else allowed_split
        if blob["spec"].get("split") not in allowed:
            raise RuntimeError(f"evaluation refused snapshot {episode_id}")
        out = roll_policy(
            env, model, book, blob, device,
            cap_steps=int(cfg.get("eval_cap_steps") or 200),
            limit_s=limit_s,
        )
        rows.append({
            "id": episode_id,
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
        print("EVAL", episode_id, out["reason"], out.get("handoff"), out.get("recovery_elapsed"), flush=True)
    elapsed = [float(row["recovery_elapsed"]) for row in rows if row.get("recovery_elapsed") is not None]
    base_success = sum(1 for row in baseline if row.get("success") or row.get("reason") == "SUCCESS_12S")
    summary = {
        "episodes": len(rows),
        "success": sum(1 for row in rows if row["success"]),
        "baseline_success": base_success,
        "baseline_episodes": len(baseline),
        "baseline_file": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "device": device_name,
        "recovery_limit_s": limit_s,
        "nominal_seconds": NOMINAL_S,
        "reasons": dict(Counter(row["label"] for row in rows)),
        "handoff": dict(Counter(row["handoff"] for row in rows)),
        "recovery_elapsed_s": {
            "min": min(elapsed) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
    }
    out_dir = ROOT / cfg["eval_output"]
    historical = {
        (ROOT / "results/archive/demo_results/diagnostics/raw/unified_k47/eval").resolve(),
        (ROOT / "results/archive/demo_results/diagnostics/raw/horizon_migration").resolve(),
        (ROOT / "results/archive/demo_results/diagnostics/raw/unified_k47/supervised").resolve(),
    }
    if out_dir.resolve() in historical:
        raise RuntimeError(f"refusing to write evaluation into {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    result_name = "closed_loop.json" if rebuilt and explicit else "heldout_train.json"
    (out_dir / result_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if checkpoint_path.stat().st_mtime != before or checkpoint_path.stat().st_size != size_before:
        raise RuntimeError("evaluation changed the checkpoint")
    print("EVAL_DONE", json.dumps(summary), flush=True)
