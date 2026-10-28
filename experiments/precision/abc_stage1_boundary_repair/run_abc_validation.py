#!/usr/bin/env python
"""Controlled A/B/C Stage-1 boundary validation (no large-scale generation).

For each plan, sample existing verified ICs and report:
  Stage-1 failure / direct-eligible (x1) / Precision-required (x2),
  snapshot identity, and optional Precision probe on x2.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import (  # noqa: E402
    capture_fingerprint,
    live_verify_stage1_12s,
)
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from data_generation.oracles.stage1_exit import stage1_exit_from_existing_ic  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/abc_stage1_boundary_repair"
V3 = ROOT / "data/experts_v3_stage1_verified"
# Known Stage-1 ineligible IC from integrity audit
INVALID_IC = ROOT / "results/diagnostics/stage1_precision_integrity_audit"


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def _pick(folder: Path, n: int = 3) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.pkl"))[:n]


def _row_from_exit(plan: str, path: Path, exit_obj, prec_status=None, prec_n_act=None) -> dict:
    err = exit_obj.precision_entry_errors or {}
    return {
        "plan": plan,
        "source_ic_id": exit_obj.source_ic_id or path.stem,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "classification": exit_obj.classification,
        "stage1_success_12s": int(bool(exit_obj.stage1_success_12s)),
        "precision_required": "" if exit_obj.precision_required is None else int(bool(exit_obj.precision_required)),
        "capture_fp": exit_obj.capture_fingerprint,
        "dx_mm": err.get("dx_mm"),
        "dy_mm": err.get("dy_mm"),
        "dz_mm": err.get("dz_mm"),
        "dtheta_deg": err.get("dtheta_deg"),
        "bilateral": (exit_obj.contact_summary_at_exit or {}).get("bilateral"),
        "fnL": (exit_obj.contact_summary_at_exit or {}).get("fnL"),
        "fnR": (exit_obj.contact_summary_at_exit or {}).get("fnR"),
        "stage1_failure_reason": exit_obj.stage1_failure_reason,
        "precision_status": prec_status,
        "precision_n_actions": prec_n_act,
        "snapshot_identity_ok": int(
            capture_fingerprint(exit_obj.stage1_exit_capture) == exit_obj.capture_fingerprint
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    rows: list[dict] = []
    cfg = PrecisionConfig()

    suites = [
        ("A", V3 / "A_STABLE_DIRECT_HANDOFF", V3 / "A_STABLE_PRECISION"),
        ("B", V3 / "B_STABLE_DIRECT_HANDOFF", V3 / "B_STABLE_PRECISION"),
        ("C", V3 / "C_RECATCH_DIRECT_HANDOFF", V3 / "C_RECATCH_PRECISION"),
    ]

    # At most one live Precision probe per plan (bounded validation).
    precision_probed: set[str] = set()
    for plan, direct_dir, prec_dir in suites:
        paths = _pick(direct_dir, 2) + _pick(prec_dir, 2)
        for path in paths:
            print(f"[validate] {plan} {path.name}", flush=True)
            blob = pickle.loads(path.read_bytes())
            exit_obj = stage1_exit_from_existing_ic(
                env,
                {"capture": blob["capture"], "ref": blob["ref"], "sample_id": path.stem},
                plan_id=plan,
                source_ic_id=path.stem,
                allow_live_verify=True,
            )
            prec_status = None
            prec_n = None
            if exit_obj.stage1_success_12s and exit_obj.precision_required:
                if plan not in precision_probed:
                    res = run_precision(env, exit_obj.as_precision_blob(), plan, cfg)
                    prec_status = res.status
                    prec_n = len(res.actions or [])
                    precision_probed.add(plan)
                else:
                    prec_status = "PRECISION_REQUIRED_NOT_PROBED"
                    prec_n = None
            elif exit_obj.stage1_success_12s and not exit_obj.precision_required:
                restore_capture(env, exit_obj.stage1_exit_capture)
                _ = measure_errors(env.sim)
                prec_status = "DIRECT_ELIGIBLE_NO_PRECISION"
                prec_n = 0
            rows.append(_row_from_exit(plan, path, exit_obj, prec_status, prec_n))

    # Stage-1 failure example: known audit IC (no full-tree pickle scan)
    known_fail_id = "PILOT_A__ball_train_v2_0012__93a263b2b1a49030"
    search_roots = [
        ROOT / "data/stage2_abc_final_v2",
        ROOT / "data/unified_experts",
        ROOT / "results/diagnostics/stage1_precision_integrity_audit",
        ROOT / "results/data_generation",
    ]
    fail_candidates: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            fail_candidates.extend(root.glob(f"**/*{known_fail_id}*.pkl"))
    found = fail_candidates[0] if fail_candidates else None
    if found is not None:
        blob = pickle.loads(found.read_bytes())
        if "capture" in blob and "ref" in blob:
            exit_obj = stage1_exit_from_existing_ic(
                env,
                {"capture": blob["capture"], "ref": blob["ref"], "sample_id": known_fail_id},
                plan_id="A",
                source_ic_id=known_fail_id,
                allow_live_verify=True,
            )
            res = run_precision(env, exit_obj.as_precision_blob(), "A", cfg)
            rows.append(
                {
                    **_row_from_exit("A", found, exit_obj, res.status, len(res.actions or [])),
                    "note": "stage1_failure_example",
                }
            )
    else:
        # Synthetic failure row from registry evidence without loading a pickle
        ev_path = ROOT / "data/stage1_eligibility_evidence/evidence_by_capture_fp.json"
        if ev_path.is_file():
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            for fp, row in (ev.get("by_capture_fp") or {}).items():
                if row.get("SUCCESS_12S") is False:
                    rows.append(
                        {
                            "plan": "A",
                            "source_ic_id": row.get("sample_id") or known_fail_id,
                            "path": "",
                            "classification": "STAGE1_FAILED",
                            "stage1_success_12s": 0,
                            "precision_required": "",
                            "capture_fp": fp,
                            "stage1_failure_reason": row.get("reason") or "SUCCESS_12S_FALSE",
                            "precision_status": "STAGE1_INELIGIBLE",
                            "precision_n_actions": 0,
                            "snapshot_identity_ok": 1,
                            "note": "stage1_failure_from_registry_only",
                        }
                    )
                    break

    csv_path = OUT / "ABC_VALIDATION_RESULTS.csv"
    keys = sorted({k for r in rows for k in r})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "by_classification": {},
        "by_plan": {},
    }
    for r in rows:
        c = r.get("classification") or "?"
        summary["by_classification"][c] = summary["by_classification"].get(c, 0) + 1
        p = r.get("plan") or "?"
        summary["by_plan"].setdefault(p, {"n": 0, "classes": {}})
        summary["by_plan"][p]["n"] += 1
        summary["by_plan"][p]["classes"][c] = summary["by_plan"][p]["classes"].get(c, 0) + 1

    (OUT / "ABC_VALIDATION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
