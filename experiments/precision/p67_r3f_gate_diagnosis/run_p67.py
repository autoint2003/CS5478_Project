"""P67 orchestrator — diagnosis only (Phases 1–5). No training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p67_r3f_gate_diagnosis.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=("all", "1", "2", "3", "4", "5"),
        default="all",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        dt = time.perf_counter() - t0
        print(f"\n######## {name} (t+{dt:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": dt})

    if args.phase in ("all", "1"):
        mark("Phase1_instrument")
        from experiments.precision.p67_r3f_gate_diagnosis.instrument import run_phase1
        from experiments.precision.p67_r3f_gate_diagnosis.write_reports import write_gate_audit

        rep = run_phase1(device=args.device)
        write_gate_audit(rep)

    if args.phase in ("all", "2"):
        mark("Phase2_attribution")
        from experiments.precision.p67_r3f_gate_diagnosis.attribution import run_attribution

        run_attribution()

    if args.phase in ("all", "3"):
        mark("Phase3_mine")
        from experiments.precision.p67_r3f_gate_diagnosis.counterfactual import (
            mine_post_stable_snapshots,
            run_counterfactuals,
        )

        mine_post_stable_snapshots(device=args.device)
        mark("Phase3_counterfactual")
        run_counterfactuals(device=args.device)

    if args.phase in ("all", "4"):
        mark("Phase4_skill_critic")
        from experiments.precision.p67_r3f_gate_diagnosis.skill_critic import run_skill_critic

        run_skill_critic(device=args.device)

    if args.phase in ("all", "5"):
        mark("Phase5_decision")
        from experiments.precision.p67_r3f_gate_diagnosis.write_reports import write_final_report

        write_final_report()

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2),
        encoding="utf-8",
    )
    print(f"\nP67 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)
    print("NO TRAINING LAUNCHED.", flush=True)


if __name__ == "__main__":
    main()
