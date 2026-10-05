"""Canonical end-to-end impact recovery demo.

    python training/demo_impact_recovery.py --mode zero
    python training/demo_impact_recovery.py --mode heuristic
    python training/demo_impact_recovery.py --mode sac
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.impact_demo_core import LOG_DIR, frozen_hashes, run_episode  # noqa: E402

SAC_MSG = "No valid observable SAC checkpoint available."


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Canonical impact recovery demo")
    p.add_argument("--mode", required=True, choices=("zero", "heuristic", "sac"))
    p.add_argument(
        "--headless",
        action="store_true",
        help="Optional diagnostic: no interactive viewer (same physics).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "sac":
        print(SAC_MSG)
        return 2

    hashes = frozen_hashes()
    result = run_episode(
        args.mode,
        viewer=None,
        interactive=not args.headless,
        screenshots=True,
        shot_prefix=f"{args.mode}_",
        seed=0,
    )
    out = LOG_DIR / f"demo_{args.mode}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "mode",
                    "t_first_contact",
                    "t_last_contact",
                    "t_park",
                    "pre_contact_speed",
                    "momentum",
                    "kinetic_energy",
                    "final",
                )
            },
            indent=2,
            default=str,
        )
    )
    print("hashes", hashes)
    print("log", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
