"""Orchestrate P75 Critic-only calibration end-to-end."""

from __future__ import annotations

import argparse
import json
import time

from experiments.precision.p75_critic_calibration.audit import run_audit
from experiments.precision.p75_critic_calibration.collect import collect_dataset
from experiments.precision.p75_critic_calibration.decide import decide
from experiments.precision.p75_critic_calibration.evaluate import run_evaluate
from experiments.precision.p75_critic_calibration.paths import OUT
from experiments.precision.p75_critic_calibration.train_critic import run_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--skip-collect",
        action="store_true",
        help="Reuse existing train/eval episode pickles",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {"phases": []}

    if not args.skip_collect:
        print("=== P75 Phase 1: Collect ===", flush=True)
        meta = collect_dataset(device=args.device)
        log["phases"].append({"phase": 1, "meta": meta, "sec": time.time() - t0})
    else:
        print("=== P75 Phase 1: SKIP collect ===", flush=True)

    print("=== P75 Phase 2: Audit ===", flush=True)
    t1 = time.time()
    audit = run_audit(device=args.device)
    log["phases"].append({"phase": 2, "checks": audit.get("checks"), "sec": time.time() - t1})

    if not args.skip_train:
        print("=== P75 Phase 3: Critic-only train ===", flush=True)
        t2 = time.time()
        train = run_train(device=args.device)
        log["phases"].append(
            {
                "phase": 3,
                "actor_digest_unchanged": train.get("actor_digest_unchanged"),
                "sec": time.time() - t2,
            }
        )
    else:
        print("=== P75 Phase 3: SKIP train ===", flush=True)

    if not args.skip_eval:
        print("=== P75 Phase 4–5: Eval + Advantage ===", flush=True)
        t3 = time.time()
        run_evaluate(device=args.device)
        log["phases"].append({"phase": "4_5", "sec": time.time() - t3})
    else:
        print("=== P75 Phase 4–5: SKIP eval ===", flush=True)

    print("=== P75 Phase 6–7: Decide ===", flush=True)
    decision = decide()
    log["decision"] = decision
    log["total_sec"] = time.time() - t0
    (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"case": decision["case"], "recommend_ppo": decision["recommend_matched_actor_ppo_p76"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
