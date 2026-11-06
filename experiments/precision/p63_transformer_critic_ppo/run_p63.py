"""P63 orchestrator: tests → benchmark → critic warm → expand C → short pilots → report.

Does **not** launch a long PPO run; that decision follows the pilot.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.precision.p63_transformer_critic_ppo.paths import OUT
from experiments.precision.p63_transformer_critic_ppo.write_final import write_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--skip-expand-c", action="store_true")
    ap.add_argument("--skip-pilot", action="store_true")
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--warm-ics", type=int, default=None)
    ap.add_argument("--expand-n", type=int, default=24)
    ap.add_argument("--pilot-updates", type=int, default=4)
    ap.add_argument("--pilot-max-ics", type=int, default=48)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # Publish architecture doc into results
    src_arch = Path(__file__).with_name("ARCHITECTURE.md")
    if src_arch.is_file():
        shutil.copy(src_arch, OUT / "ARCHITECTURE.md")

    if not args.skip_tests:
        from experiments.precision.p63_transformer_critic_ppo.tests_correctness import run_all

        print("=== Phase 0/2: correctness tests ===", flush=True)
        tests = run_all(device=args.device)
        if not tests.get("all_ok"):
            write_final()
            raise SystemExit("Correctness tests failed — aborting before expensive work.")

    if args.tests_only:
        write_final()
        return

    if not args.skip_bench:
        from experiments.precision.p63_transformer_critic_ppo.throughput import run_benchmark

        print("=== Throughput benchmark ===", flush=True)
        run_benchmark(device=args.device)

    if not args.skip_warm:
        from experiments.precision.p63_transformer_critic_ppo.critic_pretrain import (
            run_critic_pretrain,
        )

        print("=== Phase 3: critic warm-start ===", flush=True)
        warm = run_critic_pretrain(device=args.device, n_ics=args.warm_ics)
        if not (warm.get("gates") or {}).get("pass"):
            write_final()
            raise SystemExit("Critic warm-start gates failed — no Actor PPO yet.")

    if not args.skip_expand_c:
        from experiments.precision.p63_transformer_critic_ppo.expand_c import select_expanded_c

        print("=== Phase 4: expand C ICs ===", flush=True)
        select_expanded_c(n=args.expand_n)

    if not args.skip_pilot:
        from experiments.precision.p63_transformer_critic_ppo.pilot import run_pilots

        print("=== Phase 5: short PPO pilots ===", flush=True)
        run_pilots(
            device=args.device,
            n_updates=args.pilot_updates,
            max_ics=args.pilot_max_ics,
        )

    path = write_final()
    print(f"Wrote {path}", flush=True)
    print(json.dumps({"out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
