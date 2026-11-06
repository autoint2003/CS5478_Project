"""Orchestrate P60 end-to-end: select → collect → restore test → probe → train → val → heldout."""

from __future__ import annotations

import json
import sys
import time

from experiments.precision.p60_c_stable_state_grpo.collect_states import collect_states
from experiments.precision.p60_c_stable_state_grpo.feasibility_probe import run_feasibility_probe
from experiments.precision.p60_c_stable_state_grpo.heldout_eval import main as heldout_main
from experiments.precision.p60_c_stable_state_grpo.paths import N_STATES_MIN, OUT
from experiments.precision.p60_c_stable_state_grpo.restoration_test import run_restoration_test
from experiments.precision.p60_c_stable_state_grpo.select_checkpoint import select_init_checkpoint
from experiments.precision.p60_c_stable_state_grpo.train_continuation import run_train
from experiments.precision.p60_c_stable_state_grpo.validate import run_validation
from experiments.precision.p60_c_stable_state_grpo.write_final import write_final


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-heldout", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("=== P60 Phase 1: checkpoint selection ===", flush=True)
    sel = select_init_checkpoint()
    print(json.dumps(sel, indent=2), flush=True)

    if not args.skip_collect:
        print("=== P60 Phase 2: collect stable-but-failed states ===", flush=True)
        man = collect_states(device=args.device)
        if man["n_states"] < N_STATES_MIN:
            print(f"ERROR: only {man['n_states']} states (<{N_STATES_MIN})", flush=True)
            write_final()
            sys.exit(2)

    print("=== P60 Phase 3: restoration test ===", flush=True)
    rest = run_restoration_test(device=args.device)
    if not rest["all_ok"]:
        print("ERROR: restoration failed", flush=True)
        write_final()
        sys.exit(3)

    print("=== P60 Phase 4: feasibility probe ===", flush=True)
    probe = run_feasibility_probe(device=args.device)
    if not probe["viable_for_training"]:
        print("STOP: not viable for training", flush=True)
        print(probe.get("stop_reason"), flush=True)
        write_final()
        sys.exit(0)

    if not args.skip_train:
        print("=== P60 Phase 5: GRPO continuation training ===", flush=True)
        # train already validates after each pass when skip_val=False
        run_train(device=args.device, skip_val=False)
    else:
        print("=== P60 Phase 6: validation only ===", flush=True)
        run_validation(device=args.device, include_baselines=True)

    if not args.skip_heldout:
        print("=== P60 Phase 7: held-out eval ===", flush=True)
        # heldout_main uses argparse from sys.argv — temporarily strip
        argv = sys.argv
        sys.argv = ["heldout_eval.py", "--device", args.device]
        try:
            heldout_main()
        finally:
            sys.argv = argv

    write_final()
    print(f"=== P60 done in {(time.perf_counter()-t0)/3600:.2f} h ===", flush=True)
    print(f"Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
