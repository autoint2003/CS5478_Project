"""P62 orchestrator: diagnose → decision → multi-epoch continuation → select → final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.precision.p62_grpo_optimization.diagnose_logs import run_diagnose
from experiments.precision.p62_grpo_optimization.eval_dev import eval_checkpoint_final
from experiments.precision.p62_grpo_optimization.paths import CKPT_DIR, OUT, P57A_CKPT
from experiments.precision.p62_grpo_optimization.policy_drift import run_policy_drift
from experiments.precision.p62_grpo_optimization.train_epochs import run_continuation
from experiments.precision.p62_grpo_optimization.write_final import write_reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-diagnose", action="store_true")
    ap.add_argument("--skip-drift", action="store_true")
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--max-extra-epochs", type=int, default=3)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_diagnose:
        print("=== Phase 1: GRPO / reward / sampling diagnose ===", flush=True)
        run_diagnose()

    if not args.skip_drift:
        print("=== Phase 1: policy drift P57 vs P59 ===", flush=True)
        run_policy_drift(device=args.device)

    if args.diagnose_only:
        write_reports()
        print("diagnose_only done", flush=True)
        return

    decision = {}
    if (OUT / "PHASE2_DECISION.json").is_file():
        decision = json.loads((OUT / "PHASE2_DECISION.json").read_text(encoding="utf-8"))
    print(f"Phase2: {decision.get('decision', 'n/a')}", flush=True)

    if not args.skip_train:
        print("=== Phase 3: multi-epoch continuation ===", flush=True)
        run_continuation(device=args.device, max_extra_epochs=args.max_extra_epochs)

    # Select + optional final test
    write_reports()
    sel = {}
    if (OUT / "SELECTED_CHECKPOINT.json").is_file():
        sel = json.loads((OUT / "SELECTED_CHECKPOINT.json").read_text(encoding="utf-8"))
    name = sel.get("selected", "P57_A")
    if name == "P57_A":
        ckpt = P57A_CKPT
    else:
        # P62_epoch2 → P62_epoch2.pt
        ep = name.replace("P62_", "")
        ckpt = CKPT_DIR / f"P62_{ep}.pt"
        if not ckpt.is_file():
            ckpt = CKPT_DIR / f"{name}.pt"

    final = None
    if not args.skip_final and ckpt.is_file():
        print(f"=== Phase 5: fresh final test on {name} ===", flush=True)
        final = eval_checkpoint_final(f"{name}_final", ckpt, device=args.device)
        (OUT / "FINAL_TEST_RESULTS.json").write_text(
            json.dumps({k: v for k, v in final.items() if k != "per_ic"}, indent=2),
            encoding="utf-8",
        )
        write_reports(final_eval=final)

    print(json.dumps({"selected": name, "ckpt": str(ckpt), "final_msp": (final or {}).get("MSP_RELAXED")}, indent=2))


if __name__ == "__main__":
    main()
