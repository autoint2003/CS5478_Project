"""CLI for listing branches and generating experts_v2 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_generation.expert_pipeline.branches import BRANCHES
from data_generation.expert_pipeline.generate import generate_all, generate_branch
from data_generation.expert_pipeline.sources import discover_branch_ics, load_generation_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expert generation (experts_v2). Default mode consumes pre-captured Stage-1 ICs. "
            "Live Stage-1→Precision generation lives in "
            "experiments/precision/precision_required_pilot_v2/run_pilot.py "
            "(data_generation.expert_pipeline.live_stage1)."
        )
    )
    parser.add_argument("--list-branches", action="store_true")
    parser.add_argument("--list-ics", action="store_true", help="Print IC counts per branch")
    parser.add_argument("--branch", choices=sorted(BRANCHES), help="Generate one branch")
    parser.add_argument("--all-branches", action="store_true", help="Generate all six branches")
    parser.add_argument("--limit", type=int, default=None, help="Max ICs per branch (smoke tests)")
    parser.add_argument("--output", type=Path, default=None, help="Output root (default data/experts_v2)")
    parser.add_argument("--config", type=Path, default=None, help="Generation YAML")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear prior attempts for the selected branch(es) before regenerating",
    )
    args = parser.parse_args(argv)

    if args.list_branches:
        rows = [
            {
                "branch_id": b.branch_id,
                "stage1": b.stage1,
                "continuation": b.continuation,
                "description": b.description,
            }
            for b in BRANCHES.values()
        ]
        print(json.dumps(rows, indent=2))
        return 0

    if args.list_ics:
        cfg = load_generation_config(args.config)
        counts = {bid: len(discover_branch_ics(bid, cfg)) for bid in BRANCHES}
        print(json.dumps({"config_id": cfg.get("config_id"), "counts": counts, "total_attempts": sum(counts.values())}, indent=2))
        return 0

    if args.all_branches:
        summary = generate_all(
            output_root=args.output,
            config_path=args.config,
            limit_per_branch=args.limit,
            force=args.force,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.branch:
        stats = generate_branch(
            args.branch,
            output_root=args.output,
            config_path=args.config,
            limit=args.limit,
            force=args.force,
        )
        print(json.dumps(stats, indent=2, default=str))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
