"""Run Adaptive Independent-G epochs (natural unique-IC shuffle by default).

Usage:
  python -u scripts/run_adaptive_g_full.py --epochs 1
  python -u scripts/run_adaptive_g_full.py --epochs 1 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.yaml_config import load_yaml
from training.adaptive_grpo import adaptive_epoch_path, train_adaptive_independent_g

CFG = "config/training/k48/unified_k48_recovery_v3_adaptive_g.yaml"
SFT = "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"


def latest_epoch(out: Path) -> int:
    epochs = sorted(out.glob("checkpoint_epoch_*.pt"))
    if not epochs:
        return 0
    return max(int(p.stem.split("_")[-1]) for p in epochs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=CFG)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg["ic_epochs"] = int(args.epochs)
    out = ROOT / cfg["grpo_output"]
    done = latest_epoch(out)
    if args.resume:
        if done < 1:
            raise SystemExit("nothing to resume; start without --resume")
        ckpt = str(adaptive_epoch_path(out, done).relative_to(ROOT)).replace("\\", "/")
        print("RESUME_FROM", ckpt, "add_epochs", args.epochs, flush=True)
        summary = train_adaptive_independent_g(cfg, ckpt, limit_ics=None, epochs=args.epochs)
    else:
        if done > 0:
            raise SystemExit(f"{out} already has epoch {done}; use --resume or a new output dir")
        print("FRESH_START", SFT, "epochs", args.epochs, "out", cfg["grpo_output"], flush=True)
        summary = train_adaptive_independent_g(cfg, SFT, limit_ics=None, epochs=args.epochs)
    console = {
        "epoch": summary.get("epoch"),
        "sampling": summary.get("schedule", {}).get("sampling_mode"),
        "slots": summary.get("ics"),
        "source_exposure_ratio": summary.get("source_exposure_ratio"),
        "stop_buckets": summary.get("stop_buckets"),
        "kl_mean": summary.get("kl_mean"),
        "grad_norm_max": summary.get("grad_norm_max"),
        "by_source": summary.get("by_source"),
        "validation_heuristic": summary.get("validation", {}).get("heuristic"),
        "validation_hard_dev": summary.get("validation", {}).get("hard_dev"),
        "validation_all": summary.get("validation", {}).get("all"),
        "checkpoint": summary.get("checkpoint"),
    }
    print("FULL_RUN_EPOCH_DONE", json.dumps(console), flush=True)


if __name__ == "__main__":
    main()
