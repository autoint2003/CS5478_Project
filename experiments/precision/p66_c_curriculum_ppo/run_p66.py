"""P66 orchestrator — Phase 1→4 then eval; no long train until pilots pass."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p66_c_curriculum_ppo.paths import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=("all", "1", "2", "3", "4", "6", "reports"),
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
        mark("Phase1_geometry")
        from experiments.precision.p66_c_curriculum_ppo.geometry_validate import (
            run_geometry_validation,
        )

        run_geometry_validation()
        mark("Phase1_stages")
        from experiments.precision.p66_c_curriculum_ppo.stage_transitions import (
            run_stage_transitions,
        )

        run_stage_transitions()

    if args.phase in ("all", "2"):
        mark("Phase2_mine")
        from experiments.precision.p66_c_curriculum_ppo.mine_snapshots import (
            mine_pre_recatch_snapshots,
        )

        mine_pre_recatch_snapshots(n_experts=40)
        mark("Phase2_restore")
        from experiments.precision.p66_c_curriculum_ppo.restore_validate import (
            run_expert_feasibility,
            run_restoration_tests,
        )

        run_restoration_tests()
        run_expert_feasibility()

    if args.phase in ("all", "3"):
        mark("Phase3_critic")
        from experiments.precision.p66_c_curriculum_ppo.critic_diag import (
            run_critic_diagnostics,
        )

        run_critic_diagnostics()

    if args.phase in ("all", "4"):
        mark("Phase4_pilots")
        from experiments.precision.p66_c_curriculum_ppo.train_pilot import run_matched_pilots

        run_matched_pilots(device=args.device)

    if args.phase in ("all", "6"):
        mark("Phase6_eval")
        from experiments.precision.p66_c_curriculum_ppo.eval_compare import run_evaluations

        run_evaluations(device=args.device)

    if args.phase in ("all", "6", "reports"):
        mark("reports")
        from experiments.precision.p66_c_curriculum_ppo.write_reports import write_final_report

        write_final_report()

    (OUT / "RUN_LOG.json").write_text(
        json.dumps({"log": log, "total_s": time.perf_counter() - t0}, indent=2),
        encoding="utf-8",
    )
    print(f"\nP66 done in {time.perf_counter()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
