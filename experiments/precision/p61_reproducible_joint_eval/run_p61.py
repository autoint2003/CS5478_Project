"""P61 orchestrator: residual audit → repro test → joint eval → final report. No training."""

from __future__ import annotations

import json
import time

from experiments.precision.p61_reproducible_joint_eval.joint_eval import run_joint_eval
from experiments.precision.p61_reproducible_joint_eval.paths import OUT
from experiments.precision.p61_reproducible_joint_eval.reproducibility_test import (
    run_reproducibility_test,
)
from experiments.precision.p61_reproducible_joint_eval.residual_kl_audit import (
    run_residual_kl_audit,
)
from experiments.precision.p61_reproducible_joint_eval.write_final import write_final


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", help="Joint eval device")
    ap.add_argument("--repro-device", default="cpu", help="Reproducibility test device")
    ap.add_argument("--repro-rollouts", type=int, default=4)
    ap.add_argument("--skip-repro", action="store_true")
    ap.add_argument("--skip-joint", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("=== P61 residual/KL audit ===", flush=True)
    run_residual_kl_audit(device="cpu")

    if not args.skip_repro:
        print("=== P61 reproducibility test (canonical) ===", flush=True)
        run_reproducibility_test(device=args.repro_device, n_rollouts=args.repro_rollouts)
        if args.repro_device == "cpu" and str(args.device).startswith("cuda"):
            print("=== P61 CUDA repro probe (optional) ===", flush=True)
            import shutil

            # Preserve CPU artifacts
            for name in ("REPRODUCIBILITY_TEST.md", "REPRODUCIBILITY_TEST.json"):
                p = OUT / name
                if p.is_file():
                    shutil.copy(p, OUT / name.replace("REPRODUCIBILITY_TEST", "REPRODUCIBILITY_TEST_cpu"))
            try:
                cuda_rep = run_reproducibility_test(device="cuda", n_rollouts=2)
                (OUT / "REPRODUCIBILITY_TEST_cuda_probe.json").write_text(
                    json.dumps(cuda_rep, indent=2), encoding="utf-8"
                )
            except Exception as e:
                (OUT / "REPRODUCIBILITY_TEST_cuda_probe.md").write_text(
                    f"# CUDA repro probe failed\n\n{e}\n", encoding="utf-8"
                )
            # Restore CPU as canonical deliverable
            for name in ("REPRODUCIBILITY_TEST.md", "REPRODUCIBILITY_TEST.json"):
                src = OUT / name.replace("REPRODUCIBILITY_TEST", "REPRODUCIBILITY_TEST_cpu")
                if src.is_file():
                    shutil.copy(src, OUT / name)

    if not args.skip_joint:
        print("=== P61 joint evaluation ===", flush=True)
        run_joint_eval(device=args.device)

    write_final()
    print(f"=== P61 done in {(time.perf_counter()-t0)/3600:.2f} h ===", flush=True)
    print(f"Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
