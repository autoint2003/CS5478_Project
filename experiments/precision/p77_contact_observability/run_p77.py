"""Orchestrate P77 observability & sensing feasibility (no NN / PPO)."""

from __future__ import annotations

import argparse
import json
import time

from experiments.precision.p77_contact_observability.action_relevance import run_action_relevance
from experiments.precision.p77_contact_observability.aliasing import run_aliasing
from experiments.precision.p77_contact_observability.decide import decide
from experiments.precision.p77_contact_observability.existing_obs import run_existing_obs
from experiments.precision.p77_contact_observability.label_audit import run_label_audit
from experiments.precision.p77_contact_observability.paths import OUT
from experiments.precision.p77_contact_observability.sensing import run_sensing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {}

    print("=== P77 Phase 1: Label audit ===", flush=True)
    run_label_audit()
    print("=== P77 Phase 2: Existing observation info ===", flush=True)
    run_existing_obs()
    print("=== P77 Phase 3: Rare-positive aliasing ===", flush=True)
    run_aliasing()
    print("=== P77 Phase 4–5: Sensing + info gain ===", flush=True)
    sense = run_sensing(device=args.device)
    log["n_diag_eps"] = sense.get("n_diag_eps")
    print("=== P77 Phase 6: Action relevance ===", flush=True)
    run_action_relevance()
    print("=== P77 Phase 7: Decide ===", flush=True)
    decision = decide()
    log["case"] = decision["case"]
    log["p78"] = decision["p78_direction"]
    log["total_sec"] = time.time() - t0
    (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2), flush=True)


if __name__ == "__main__":
    main()
