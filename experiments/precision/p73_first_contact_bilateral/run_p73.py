"""P73 orchestrator — first-contact → opposing bilateral; no training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p73_first_contact_bilateral.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("all", "1", "23", "45", "reports"), default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        print(f"\n######## {name} (t+{time.perf_counter()-t0:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": time.perf_counter() - t0})

    if args.phase in ("all", "1"):
        mark("Phase1_attribution")
        from experiments.precision.p73_first_contact_bilateral.phase1_attr import run_phase1

        run_phase1(device=args.device)

    if args.phase in ("all", "23"):
        mark("Phase2_3_cf_window")
        from experiments.precision.p73_first_contact_bilateral.phase23_cf import run_phase23

        run_phase23(device=args.device)

    if args.phase in ("all", "45"):
        mark("Phase4_5_validate")
        from experiments.precision.p73_first_contact_bilateral.phase45_validate import run_phase45

        run_phase45(device=args.device)

    if args.phase in ("all", "1", "23", "45", "reports"):
        mark("reports")
        from experiments.precision.p73_first_contact_bilateral.write_reports import write_all

        decision = write_all()
        print(f"P73 decision={decision}", flush=True)

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2), encoding="utf-8"
    )
    print(f"\nP73 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)
    print("NO TRAINING. Pre-contact V2 unchanged. P72 not promoted.", flush=True)


if __name__ == "__main__":
    main()
