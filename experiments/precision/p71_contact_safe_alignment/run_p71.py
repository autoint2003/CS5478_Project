"""P71 orchestrator — contact-safe alignment refinement; no training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p71_contact_safe_alignment.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("all", "1", "2", "34", "5", "reports"), default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-v5", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        print(f"\n######## {name} (t+{time.perf_counter()-t0:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": time.perf_counter() - t0})

    if args.phase in ("all", "1"):
        mark("Phase1_V2_success_geometry")
        from experiments.precision.p71_contact_safe_alignment.phase1_success_geometry import run_phase1

        run_phase1(device=args.device)

    if args.phase in ("all", "2"):
        mark("Phase2_token56")
        from experiments.precision.p71_contact_safe_alignment.phase2_token56 import run_phase2

        run_phase2(device=args.device)

    if args.phase in ("all", "34"):
        mark("Phase3_4_variants")
        from experiments.precision.p71_contact_safe_alignment.variants_eval import run_phase34

        run_phase34(device=args.device, include_v5=not args.no_v5)

    if args.phase in ("all", "5"):
        mark("Phase5_panel")
        from experiments.precision.p71_contact_safe_alignment.panel_eval import run_phase5

        run_phase5(device=args.device)

    if args.phase in ("all", "1", "2", "34", "5", "reports"):
        mark("reports")
        from experiments.precision.p71_contact_safe_alignment.write_reports import write_all

        case = write_all()
        print(f"P71 case={case}", flush=True)

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2), encoding="utf-8"
    )
    print(f"\nP71 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)
    print("NO TRAINING LAUNCHED.", flush=True)


if __name__ == "__main__":
    main()
