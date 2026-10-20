"""Short Adaptive Independent-G pilot (2 epochs).

Uses source-aware sampling over a capped schedule for wall-clock feasibility,
while validating the full merged validation set every epoch.
Writes under results/grpo/adaptive_g/ (not smoke/).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.yaml_config import load_yaml
from training.adaptive_grpo import train_adaptive_independent_g

CFG = "config/training/k48/unified_k48_recovery_v3_adaptive_g.yaml"
SFT = "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
OUT = ROOT / "results" / "grpo" / "adaptive_g"


def main() -> None:
    cfg = load_yaml(CFG)
    cfg["ic_epochs"] = 2
    # ~20 heuristic + ~20 hard draws from the 1:1 schedule; unique hard still cycles.
    limit_ics = 40
    started = time.perf_counter()
    print("PILOT_START", "limit_ics", limit_ics, "epochs", 2, flush=True)
    summary = train_adaptive_independent_g(cfg, SFT, limit_ics=limit_ics, epochs=2)
    report = {
        "wall_s": time.perf_counter() - started,
        "limit_ics": limit_ics,
        "epochs": 2,
        "final_checkpoint": str(OUT / "checkpoint_epoch_0002.pt"),
        "last_summary": {
            key: summary.get(key)
            for key in (
                "epoch", "ics", "updates", "stop_buckets", "found_success_ics",
                "env_steps_total", "env_steps_per_update", "group_reward_std_mean",
                "schedule", "by_source",
            )
        },
        "validation_epoch_2": summary.get("validation"),
    }
    # Attach epoch-1 validation if present.
    e1 = OUT / "validation_epoch_0001.json"
    if e1.is_file():
        report["validation_epoch_1"] = json.loads(e1.read_text(encoding="utf-8"))
    (OUT / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PILOT_DONE", json.dumps({
        "wall_s": report["wall_s"],
        "final_checkpoint": report["final_checkpoint"],
        "stop_buckets": summary.get("stop_buckets"),
        "val_all": summary.get("validation", {}).get("all"),
        "val_heuristic": summary.get("validation", {}).get("heuristic"),
        "val_hard_dev": summary.get("validation", {}).get("hard_dev"),
    }), flush=True)


if __name__ == "__main__":
    main()
