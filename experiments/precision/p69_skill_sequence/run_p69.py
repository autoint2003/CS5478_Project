"""P69 orchestrator — Phases 0–3 first; Phase 4 only if Phase 3 clears R3F gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p69_skill_sequence.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=("all", "0", "12", "3", "4", "reports"),
        default="all",
        help="all = 0→12→3→(4 if gated)→reports",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-ppo", action="store_true", help="Override Phase-3 gate (debug only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def mark(name):
        print(f"\n######## {name} (t+{time.perf_counter()-t0:.0f}s) ########\n", flush=True)
        log.append({"phase": name, "t": time.perf_counter() - t0})

    proceed_ppo = False
    p3_path = OUT / "NO_TRAIN_INTERVENTION_RESULTS.json"
    if p3_path.is_file():
        try:
            p3blob = json.loads(p3_path.read_text(encoding="utf-8"))
            proceed_ppo = bool(p3blob.get("proceed_to_ppo"))
            gate = p3blob.get("gate") or {}
            # Refuse stale soft-gate artifacts (pre-tightening) that lack succ_up
            if proceed_ppo and "succ_up" not in gate:
                print(
                    "WARNING: refusing stale P69 Phase-3 gate without succ_up; "
                    "re-run --phase 3 (or use --force_ppo only for override).",
                    flush=True,
                )
                proceed_ppo = False
            elif proceed_ppo and not gate.get("succ_up"):
                proceed_ppo = False
        except Exception:  # noqa: BLE001
            proceed_ppo = False

    if args.phase in ("all", "0"):
        mark("Phase0_precontact_audit")
        from experiments.precision.p69_skill_sequence.phase0_audit import run_phase0

        run_phase0(device=args.device)

    if args.phase in ("all", "12"):
        mark("Phase1_2_sequence_search")
        from experiments.precision.p69_skill_sequence.sequence_search import run_phase12

        run_phase12(device=args.device)

    if args.phase in ("all", "3"):
        mark("Phase3_fixed_intervention")
        from experiments.precision.p69_skill_sequence.fixed_intervention import run_phase3

        p3 = run_phase3(device=args.device)
        proceed_ppo = bool(p3.get("proceed_to_ppo"))

    if args.phase == "4" or (args.phase == "all" and (proceed_ppo or args.force_ppo)):
        if not proceed_ppo and not args.force_ppo:
            mark("Phase4_skipped_gate")
        elif not proceed_ppo and args.force_ppo:
            mark("Phase4_forced_override")
            print("WARNING: --force-ppo set; still refusing long train. Implement pilot only if coded.", flush=True)
            # Intentionally no training module until Phase 3 evidence exists in-repo.
            (OUT / "PPO_SKIPPED.json").write_text(
                json.dumps({"reason": "force_ppo requested but pilot trainer not enabled without gate"}, indent=2),
                encoding="utf-8",
            )
        else:
            mark("Phase4_ppo_pilot")
            from experiments.precision.p69_skill_sequence.train_pilot import run_matched_pilots

            run_matched_pilots(device=args.device)
            mark("Phase5_eval")
            from experiments.precision.p69_skill_sequence.eval_compare import run_evaluations

            run_evaluations(device=args.device)
    elif args.phase == "all":
        mark("Phase4_gated_off")
        print("Phase 4 not launched (fixed intervention did not clear R3F gate).", flush=True)

    if args.phase in ("all", "0", "12", "3", "4", "reports"):
        mark("reports")
        from experiments.precision.p69_skill_sequence.write_reports import write_all

        decision = write_all()
        print(f"P69 decision={decision}", flush=True)

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0, "proceed_ppo": proceed_ppo}, indent=2),
        encoding="utf-8",
    )
    print(f"\nP69 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
