"""Copy / summarize reusable upstream P63 artifacts into this experiment tree."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from experiments.precision.p63_transformer_ppo_c_grasp.dataset import full_pool_counts, sync_c_manifest
from experiments.precision.p63_transformer_ppo_c_grasp.paths import OUT, P63_CORE_OUT


def sync() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (
        "THROUGHPUT_BENCHMARK.json",
        "THROUGHPUT_BENCHMARK.md",
        "CRITIC_PRETRAIN_REPORT.json",
        "CRITIC_PRETRAIN_REPORT.md",
        "PPO_CORRECTNESS_TESTS.json",
        "PPO_CORRECTNESS_TESTS.md",
        "ARCHITECTURE.md",
    ):
        src = P63_CORE_OUT / name
        if src.is_file():
            dest_name = name
            if name.startswith("CRITIC_PRETRAIN_REPORT"):
                dest_name = name.replace("CRITIC_PRETRAIN_REPORT", "CRITIC_WARMSTART")
            shutil.copy2(src, OUT / dest_name)
            copied.append(dest_name)
            # also keep REPORT alias for warmstart
            if dest_name.startswith("CRITIC_WARMSTART.") and dest_name != name:
                pass

    man = sync_c_manifest()
    pools = full_pool_counts()
    (OUT / "TRAIN_POOL_COUNTS.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")
    return {"copied": copied, "c_manifest": str(man), "pools": pools}


if __name__ == "__main__":
    print(json.dumps(sync(), indent=2))
