"""P70 orchestrator — diagnosis + no-train counterfactuals only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p70_geometry_to_r3f.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=("all", "12", "3", "45", "reports"),
        default="all",
        help="all = 1–2 funnel → 3 transitions → 4–5 variants → reports",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        print(f"\n######## {name} (t+{time.perf_counter()-t0:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": time.perf_counter() - t0})

    if args.phase in ("all", "12"):
        mark("Phase1_2_funnel")
        from experiments.precision.p70_geometry_to_r3f.funnel_rollout import run_phase12

        run_phase12(device=args.device)

    if args.phase in ("all", "3"):
        mark("Phase3_transitions")
        from experiments.precision.p70_geometry_to_r3f.transition_analysis import run_phase3

        run_phase3(device=args.device)

    if args.phase in ("all", "45"):
        mark("Phase4_5_variants")
        from experiments.precision.p70_geometry_to_r3f.variants_eval import run_phase45

        run_phase45(device=args.device)

    if args.phase in ("all", "12", "3", "45", "reports"):
        mark("reports")
        from experiments.precision.p70_geometry_to_r3f.write_reports import write_all

        decision = write_all()
        print(f"P70 decision={decision}", flush=True)

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2),
        encoding="utf-8",
    )
    print(f"\nP70 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)
    print("NO TRAINING LAUNCHED.", flush=True)


if __name__ == "__main__":
    main()
