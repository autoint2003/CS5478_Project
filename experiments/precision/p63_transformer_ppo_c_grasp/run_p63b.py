"""Orchestrator: sync → correctness → reward audit → Branch A (then B/C if stable)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.precision.p63_transformer_ppo_c_grasp.paths import OUT
from experiments.precision.p63_transformer_ppo_c_grasp.sync_artifacts import sync
from experiments.precision.p63_transformer_ppo_c_grasp.tests_correctness import run_all
from experiments.precision.p63_transformer_ppo_c_grasp.reward_audit import audit_c_returns
from experiments.precision.p63_transformer_ppo_c_grasp.pilot_branches import run_pilots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-pilot", action="store_true")
    ap.add_argument("--only", choices=["A", "B", "C"], default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--max-ics", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # Copy audit doc
    src_audit = Path(__file__).with_name("IMPLEMENTATION_AUDIT.md")
    if src_audit.is_file():
        shutil.copy2(src_audit, OUT / "IMPLEMENTATION_AUDIT.md")

    print("=== sync artifacts ===", flush=True)
    sync_info = sync()
    print(json.dumps(sync_info, indent=2), flush=True)

    if not args.skip_tests:
        print("=== correctness tests ===", flush=True)
        tests = run_all(device=args.device)
        if not tests.get("all_ok"):
            print("CORE/CONTACT TESTS FAILED — refusing to start pilot.", flush=True)
            (OUT / "ABORT.json").write_text(
                json.dumps({"reason": "tests_failed", "tests": tests}, indent=2),
                encoding="utf-8",
            )
            return
    else:
        tests = {"skipped": True}

    if not args.skip_audit:
        print("=== reward audit ===", flush=True)
        audit = audit_c_returns(device=args.device)
        if audit.get("exploitation_risk"):
            print("REWARD EXPLOITATION RISK — Branch C should not be trusted.", flush=True)
    else:
        audit = {"skipped": True}

    if args.skip_pilot:
        print("=== pilot skipped ===", flush=True)
        return

    kw = {"device": args.device, "only": args.only}
    if args.updates is not None:
        kw["n_updates"] = args.updates
    if args.max_ics is not None:
        kw["max_ics"] = args.max_ics
    print("=== pilots ===", flush=True)
    pilots = run_pilots(**kw)
    print(json.dumps({"summary": pilots["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
