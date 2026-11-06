"""P74 orchestrator — contact learning readiness audit; no training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p74_contact_learning_readiness.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("all", "12", "34", "design", "reports"), default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        print(f"\n######## {name} (t+{time.perf_counter()-t0:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": time.perf_counter() - t0})

    if args.phase in ("all", "12"):
        mark("Phase1_2_ownership_obs")
        from experiments.precision.p74_contact_learning_readiness.phase12 import run_phase12

        run_phase12(device=args.device)

    if args.phase in ("all", "34"):
        mark("Phase3_4_learn_critic")
        from experiments.precision.p74_contact_learning_readiness.phase34 import run_phase34

        run_phase34()

    if args.phase in ("all", "34", "design"):
        mark("Phase5_7_designs")
        from experiments.precision.p74_contact_learning_readiness.designs import write_designs

        own = json.loads((OUT / "ACTOR_ACTION_OWNERSHIP.json").read_text(encoding="utf-8"))
        obs = json.loads((OUT / "CONTACT_OBSERVABILITY.json").read_text(encoding="utf-8"))
        learn = json.loads((OUT / "LEARNABLE_DECISIONS.json").read_text(encoding="utf-8"))
        crit = json.loads((OUT / "CONTACT_STAGE_CRITIC.json").read_text(encoding="utf-8"))
        decision = write_designs(ownership=own, observability=obs, learnable=learn, critic=crit)
        print(f"P74 case={decision['case']}", flush=True)

    if args.phase in ("all", "34", "design", "reports"):
        mark("reports")
        from experiments.precision.p74_contact_learning_readiness.write_reports import write_final

        print(f"FINAL case={write_final()}", flush=True)

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2), encoding="utf-8"
    )
    print(f"\nP74 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)
    print("NO TRAINING. No new M111 overlays. V2 unchanged.", flush=True)


if __name__ == "__main__":
    main()
