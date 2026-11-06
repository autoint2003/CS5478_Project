"""Orchestrate P76 audit (no neural training, no Actor PPO)."""

from __future__ import annotations

import argparse
import json
import time

from experiments.precision.p76_value_observability_audit.advantage import run_advantage
from experiments.precision.p76_value_observability_audit.aliasing import run_aliasing
from experiments.precision.p76_value_observability_audit.bottleneck import run_bottleneck
from experiments.precision.p76_value_observability_audit.calibrate import run_calibrate
from experiments.precision.p76_value_observability_audit.decide import decide
from experiments.precision.p76_value_observability_audit.paths import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {}

    print("=== P76 Phase 1–2: Affine calibration ===", flush=True)
    t1 = time.time()
    cal = run_calibrate(device=args.device)
    log["calibrate_sec"] = time.time() - t1

    print("=== P76 Phase 3: Advantage ===", flush=True)
    t2 = time.time()
    run_advantage()
    log["advantage_sec"] = time.time() - t2

    print("=== P76 Phase 4: Aliasing ===", flush=True)
    t3 = time.time()
    run_aliasing()
    log["aliasing_sec"] = time.time() - t3

    print("=== P76 Phase 5: Bottleneck ===", flush=True)
    t4 = time.time()
    run_bottleneck()
    log["bottleneck_sec"] = time.time() - t4

    print("=== P76 Phase 6: Decide ===", flush=True)
    decision = decide()
    log["decision"] = {"case": decision["case"], "recommend_ppo": decision["recommend_actor_ppo_pilot"]}
    log["total_sec"] = time.time() - t0
    log["selected"] = cal.get("selected")
    (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
