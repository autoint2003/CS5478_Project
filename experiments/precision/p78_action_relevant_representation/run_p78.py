"""Orchestrate P78 diagnostic representation study."""

from __future__ import annotations

import argparse
import json
import time

from experiments.precision.p78_action_relevant_representation.aliasing import run_aliasing
from experiments.precision.p78_action_relevant_representation.cf_action import run_cf_pipeline
from experiments.precision.p78_action_relevant_representation.compare import run_compare
from experiments.precision.p78_action_relevant_representation.decide import decide
from experiments.precision.p78_action_relevant_representation.features import write_feature_defs
from experiments.precision.p78_action_relevant_representation.paths import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== P78 Phase 1: Feature definitions ===", flush=True)
    write_feature_defs()
    print("=== P78 Phase 2: Representation comparison ===", flush=True)
    run_compare()
    print("=== P78 Phase 3–4: CF + action discrimination ===", flush=True)
    cf = run_cf_pipeline(device=args.device)
    print("=== P78 Phase 5: Action-relevant aliasing ===", flush=True)
    run_aliasing(cf.get("rows"))
    print("=== P78 Phase 6–7: Integration + decide ===", flush=True)
    decision = decide()
    log = {
        "case": decision["case"],
        "recommend_actor_exp": decision["recommend_actor_representation_experiment"],
        "p79": decision["p79_direction"],
        "n_cf_rows": len(cf.get("rows") or []),
        "total_sec": time.time() - t0,
    }
    (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2), flush=True)


if __name__ == "__main__":
    main()
