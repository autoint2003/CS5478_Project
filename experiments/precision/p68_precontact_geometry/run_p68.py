"""P68 orchestrator — diagnosis only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p68_precontact_geometry.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("all", "12", "cf", "reports"), default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        print(f"\n######## {name} (t+{time.perf_counter()-t0:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": time.perf_counter() - t0})

    if args.phase in ("all", "12"):
        mark("Phase1_2_trace")
        from experiments.precision.p68_precontact_geometry.trace import run_phase12

        run_phase12(device=args.device)

    if args.phase in ("all", "cf"):
        mark("Phase3_5_counterfactual")
        from experiments.precision.p68_precontact_geometry.counterfactual import run_counterfactuals

        run_counterfactuals(device=args.device)

    if args.phase in ("all", "cf", "reports", "12"):
        mark("reports")
        from experiments.precision.p68_precontact_geometry.write_reports import write_all

        case = write_all()
        print(f"P68 case={case}", flush=True)

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2),
        encoding="utf-8",
    )
    print(f"\nP68 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)
    print("NO TRAINING LAUNCHED.", flush=True)


if __name__ == "__main__":
    main()
