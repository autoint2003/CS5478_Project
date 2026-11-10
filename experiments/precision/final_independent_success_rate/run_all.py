"""Orchestrate small-scale independent final evaluation."""

from __future__ import annotations

import argparse
import json
import sys

from experiments.precision.final_independent_success_rate.analyze import write_reports
from experiments.precision.final_independent_success_rate.manifest import build_and_freeze_manifest
from experiments.precision.final_independent_success_rate.run_eval import run_evaluation
from experiments.precision.final_independent_success_rate.write_protocol import (
    write_leakage_audit,
    write_protocol,
)


def main(*, device: str = "cuda", force: bool = False, force_manifest: bool = False) -> dict:
    print("=== Freeze manifest ===", flush=True)
    manifest = build_and_freeze_manifest(force=force_manifest)
    write_protocol(manifest)
    write_leakage_audit(manifest)

    shortfalls = manifest.get("shortfalls") or {}
    print("Shortfalls:", json.dumps(shortfalls), flush=True)
    if not manifest.get("can_meet_10_per_class"):
        print(
            "WARNING: cannot meet 10/class without recycling. "
            "Proceeding with available ICs only (no train padding).",
            flush=True,
        )

    print("=== Run evaluation ===", flush=True)
    out = run_evaluation(device=device, force=force)
    print("=== Write reports ===", flush=True)
    rep = write_reports(out["rows"], out["manifest"])
    summary = {
        "n_episodes": len(out["rows"]),
        "shortfalls": shortfalls,
        "overall_sr": (rep.get("overall") or {}).get("equal_weight_success_rate"),
        "per_class_sr": {
            sk: (rep.get("per_class") or {}).get(sk, {}).get("success_rate") for sk in ("A1", "B1", "C", "D")
        },
        "v2_activation": f"{rep.get('n_v2')}/{rep.get('n_valid')}",
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="Re-run rollouts")
    ap.add_argument("--force-manifest", action="store_true")
    args = ap.parse_args()
    main(device=args.device, force=args.force, force_manifest=args.force_manifest)
