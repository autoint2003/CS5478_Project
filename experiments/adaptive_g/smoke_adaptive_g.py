"""Small smoke test for Adaptive Independent-G GRPO.

Loads equal-weight SFT, merged manifests, runs a tiny Independent-G step,
saves/reloads a checkpoint, and evaluates merged validation.
Writes under results/grpo/adaptive_g/smoke/ only.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.yaml_config import load_yaml
from training.adaptive_grpo import train_adaptive_independent_g

CFG = "experiments/config/training/k48/unified_k48_recovery_v3_adaptive_g.yaml"
SFT = "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
SMOKE_OUT = ROOT / "results" / "grpo" / "adaptive_g" / "smoke"


def main() -> None:
    cfg = load_yaml(CFG)
    cfg["grpo_output"] = "results/grpo/adaptive_g/smoke"
    cfg["ic_epochs"] = 1
    cfg["device"] = cfg.get("device") or "cuda"
    if SMOKE_OUT.exists():
        shutil.rmtree(SMOKE_OUT)
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("SMOKE_START", flush=True)
    summary = train_adaptive_independent_g(cfg, SFT, limit_ics=2, epochs=1)
    ckpt = SMOKE_OUT / "checkpoint_epoch_0001.pt"
    if not ckpt.is_file():
        raise RuntimeError("smoke did not write checkpoint")
    import torch
    from models.k48_checkpoint import load_trained_k48

    model, book = load_trained_k48(ckpt, "cpu")
    assert model is not None and book is not None
    report = {
        "ok": True,
        "wall_s": time.perf_counter() - started,
        "checkpoint": str(ckpt),
        "summary_keys": sorted(summary.keys()),
        "validation_all": summary.get("validation", {}).get("all"),
        "stop_buckets": summary.get("stop_buckets"),
        "ics": summary.get("ics"),
    }
    (SMOKE_OUT / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SMOKE_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
