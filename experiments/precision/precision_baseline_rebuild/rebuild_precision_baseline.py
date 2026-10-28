#!/usr/bin/env python
"""Rebuild Precision baseline on Stage-1–eligible ICs only (no controller changes)."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.entry_conditions import assess_need, entry_gate  # noqa: E402
from data_generation.oracles.precision.handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.run_precision_candidate_audit import discover_ics, first_failure_primitive  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_baseline_rebuild"
HIST = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"


def jsafe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg = PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)
    ics = discover_ics()
    if args.limit:
        ics = ics[: args.limit]

    hist = {}
    if HIST.is_file():
        for line in HIST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                hist[r["sample_id"]] = r

    rows = []
    excl = []
    print(f"baseline rebuild n={len(ics)} @ {datetime.now(timezone.utc).isoformat()}", flush=True)
    for i, meta in enumerate(ics):
        sid = meta["sample_id"]
        print(f"[{i+1}/{len(ics)}] {sid}", flush=True)
        path = ROOT / meta["ic_path"]
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception as exc:
            excl.append({"sample_id": sid, "reason": f"load:{exc}"})
            continue
        elig = check_stage1_eligible(
            env,
            {"capture": blob["capture"], "ref": blob["ref"], "sample_id": sid},
            allow_live_verify=True,
            record_live=False,
        )
        if not elig.eligible:
            excl.append(
                {
                    "sample_id": sid,
                    "ic_path": meta["ic_path"],
                    "reason": f"STAGE1_INELIGIBLE:{elig.reason}",
                    "historical_class": (hist.get(sid) or {}).get("class"),
                }
            )
            continue

        restore_capture(env, blob["capture"])
        gate = entry_gate(env)
        restore_capture(env, blob["capture"])
        err0 = measure_errors(env.sim)
        need = assess_need(err0, cfg)
        e = np.asarray(err0.get("e_signed_mm") or [0, 0, 0], float)
        direct = probe_direct_handoff(env, blob["capture"], blob["ref"])
        row = {
            "sample_id": sid,
            "ic_path": meta["ic_path"],
            "plan_family": meta.get("plan_family"),
            "source_pool": meta.get("source_pool"),
            "stage1_eligible": True,
            "stage1_eligibility": elig.to_dict(),
            "entry_gate_ok": bool(gate.get("ok")),
            "dx_mm": float(e[0]),
            "dy_mm": float(e[1]),
            "dz_mm": float(e[2]),
            "dtheta_deg": float(err0.get("dtheta_deg") or 0),
            "direct_SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
            "direct_SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
            "direct_reason": direct.get("reason"),
            "assess_need": jsafe(need),
        }
        if direct.get("SUCCESS_NEW"):
            row["class"] = "DIRECT_ELIGIBLE"
            row["precision_required"] = False
        elif not (
            need.get("dtheta_needed") or need.get("dx_needed") or need.get("dz_needed")
        ):
            row["class"] = "OUT_OF_SCOPE_OR_INVALID_IC"
            row["precision_required"] = False
        else:
            row["precision_required"] = True
            restore_capture(env, blob["capture"])
            res = run_precision(
                env,
                {"capture": blob["capture"], "ref": blob["ref"], "sample_id": sid},
                str(meta.get("plan_family") or "A"),
                cfg,
            )
            hand = res.handoff or {}
            row["precision_status"] = res.status
            row["precision_SUCCESS_NEW"] = bool(hand.get("SUCCESS_NEW"))
            row["precision_SUCCESS_12S"] = bool(hand.get("SUCCESS_12S"))
            row["precision_failures"] = list(res.failures or [])
            row["first_failed_primitive"] = first_failure_primitive(res)
            row["n_actions"] = len(res.actions or [])
            row["phases"] = list(res.phases or [])
            row["dropped"] = any("DROP" in str(f) or "drop" in str(f).lower() for f in (res.failures or []))
            if res.status == "STAGE1_INELIGIBLE":
                row["class"] = "STAGE1_INELIGIBLE"
                row["precision_required"] = False
            elif bool(hand.get("SUCCESS_NEW")):
                row["class"] = "PRECISION_REQUIRED_AND_SUCCESSFUL"
            else:
                row["class"] = "PRECISION_REQUIRED_BUT_FAILED"
        rows.append(jsafe(row))
        with (OUT / "logs" / "baseline.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsafe(row)) + "\n")

    (OUT / "PRECISION_BASELINE_RESULTS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        # flatten for CSV
        flat = []
        for r in rows:
            flat.append(
                {
                    "sample_id": r.get("sample_id"),
                    "class": r.get("class"),
                    "plan_family": r.get("plan_family"),
                    "dx_mm": r.get("dx_mm"),
                    "dy_mm": r.get("dy_mm"),
                    "dz_mm": r.get("dz_mm"),
                    "dtheta_deg": r.get("dtheta_deg"),
                    "direct_SUCCESS_NEW": r.get("direct_SUCCESS_NEW"),
                    "direct_SUCCESS_12S": r.get("direct_SUCCESS_12S"),
                    "precision_required": r.get("precision_required"),
                    "precision_status": r.get("precision_status"),
                    "precision_SUCCESS_NEW": r.get("precision_SUCCESS_NEW"),
                    "precision_SUCCESS_12S": r.get("precision_SUCCESS_12S"),
                    "first_failed_primitive": r.get("first_failed_primitive"),
                    "n_actions": r.get("n_actions"),
                    "dropped": r.get("dropped"),
                }
            )
        with (OUT / "PRECISION_BASELINE_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)

    with (OUT / "EXCLUSION_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        if excl:
            w = csv.DictWriter(f, fieldnames=list(excl[0].keys()))
            w.writeheader()
            w.writerows(excl)

    summary = {
        "n_discovered": len(ics),
        "n_eligible_run": len(rows),
        "n_excluded_stage1": len(excl),
        "class_counts": dict(Counter(r.get("class") for r in rows)),
        "precision_fail_primitives": dict(
            Counter(
                r.get("first_failed_primitive")
                for r in rows
                if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"
            )
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "BASELINE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
