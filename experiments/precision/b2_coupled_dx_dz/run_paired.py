#!/usr/bin/env python
"""P3: Coupled DX+DZ paired feasibility on seven verified B2 Stage-1 exits.

Writes results/diagnostics/b2_coupled_dx_dz/. Does not modify production Oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.b2_coupled_dx_dz.cohort import (  # noqa: E402
    EXPECTED_STATUS,
    rebuild_exits,
)
from experiments.precision.b2_coupled_dx_dz.coupled_variants import (  # noqa: E402
    coupled_v1,
    coupled_v2,
    should_attempt_coupled,
)
from experiments.precision.b2_coupled_dx_dz.experimental_sm import run_variant  # noqa: E402
from experiments.precision.b2_coupled_dx_dz.trace_replay import replay_trace  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/b2_coupled_dx_dz"


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


def cfg() -> PrecisionConfig:
    return PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)


def strip_acts(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("actions", "capture_exit", "series")}


def row_metrics(hard_id: str, variant: str, packed: dict, trace: dict | None) -> dict:
    ee = packed.get("errors_exit") or {}
    tr = trace or {}
    cr = packed.get("coupled_result") or {}
    return {
        "hard_id": hard_id,
        "variant": variant,
        "SUCCESS_NEW": packed.get("SUCCESS_NEW"),
        "SUCCESS_12S": packed.get("SUCCESS_12S"),
        "status": packed.get("status"),
        "dropped": packed.get("dropped"),
        "final_dx_mm": ee.get("dx_mm"),
        "final_dy_mm": ee.get("dy_mm"),
        "final_dz_mm": ee.get("dz_mm"),
        "final_dtheta_deg": ee.get("dtheta_deg"),
        "final_e_pos_mm": ee.get("e_pos_mm"),
        "max_dtheta_deterioration": tr.get("max_dtheta_deterioration"),
        "dz_at_dx_start": tr.get("dz_at_dx_start"),
        "dz_at_dx_end": tr.get("dz_at_dx_end"),
        "dz_return_start": tr.get("dz_return_start"),
        "dz_return_end": tr.get("dz_return_end"),
        "dz_return_drift": tr.get("dz_return_drift"),
        "contact_loss_events": tr.get("contact_loss_events"),
        "theta_mode": tr.get("theta_mode"),
        "coupled_attempted": packed.get("coupled_attempted"),
        "coupled_done": packed.get("coupled_done"),
        "coupled_ok": cr.get("ok"),
        "coupled_reason": cr.get("reason") or cr.get("stop_reason"),
        "n_actions": packed.get("n_actions"),
        "correction_duration_s": packed.get("correction_duration_s"),
        "expected_status": EXPECTED_STATUS.get(hard_id),
        "status_matches_expected": packed.get("status") == EXPECTED_STATUS.get(hard_id) if variant == "V0" else None,
    }


def run_case(env, meta: dict, pcfg: PrecisionConfig) -> dict:
    hid = meta["hard_id"]
    blob = meta["blob"]
    blob = {**blob, "sample_id": meta["raw_id"], "hard_id": hid}
    print(f"\n=== {hid} ===", flush=True)
    case: dict[str, Any] = {
        "hard_id": hid,
        "raw_id": meta["raw_id"],
        "capture_fp": meta.get("capture_fp"),
        "stage1_eligibility": meta.get("stage1_eligibility"),
        "entry_errors": meta.get("entry_errors"),
        "expected_oracle_status": meta.get("expected_oracle_status"),
        "pilot_skills": meta.get("pilot_skills"),
        "variants": {},
        "traces": {},
        "primitive_only": {},
    }

    restore_capture(env, blob["capture"])
    from data_generation.oracles.precision.residuals import measure_errors

    err0 = measure_errors(env.sim)
    case["coupled_relevant_at_entry"] = should_attempt_coupled(err0, pcfg)

    for variant in ("V0", "V1", "V2"):
        print(f"  [{hid}] {variant} ...", flush=True)
        try:
            packed = run_variant(env, blob, plan_id="B", variant=variant, cfg=pcfg)
            acts = packed.get("actions") or []
            # Persist actions for video/replay
            act_path = OUT / "raw" / f"{hid}__{variant}__actions.npz"
            act_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(act_path, actions=np.asarray(acts, np.float32))
            (OUT / "raw" / f"{hid}__{variant}__exit.pkl").write_bytes(
                pickle.dumps({"capture": packed.get("capture_exit"), "ref": blob["ref"]})
            )
            trace = replay_trace(env, blob, acts, stride=5)
            # Drop heavy series from main JSON; save separately
            series = trace.pop("series")
            (OUT / "raw" / f"{hid}__{variant}__trace.json").write_text(
                json.dumps(jsafe({"meta": strip_acts(trace), "series": series}), indent=2),
                encoding="utf-8",
            )
            case["variants"][variant] = strip_acts(packed)
            case["traces"][variant] = strip_acts(trace)
            case["metrics_rows"] = case.get("metrics_rows") or []
            # filled later
            print(
                f"    status={packed.get('status')} NEW={packed.get('SUCCESS_NEW')} "
                f"12S={packed.get('SUCCESS_12S')} n={packed.get('n_actions')} "
                f"coupled={packed.get('coupled_attempted')}/{packed.get('coupled_done')}",
                flush=True,
            )
        except Exception as exc:
            case["variants"][variant] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"    ERROR {exc}", flush=True)

    # Primitive-only coupled vs sequential (diagnostic, not full SM)
    print(f"  [{hid}] primitive-only V1/V2 ...", flush=True)
    try:
        restore_capture(env, blob["capture"])
        p1 = coupled_v1(env, blob, cfg=pcfg)
        case["primitive_only"]["V1"] = strip_acts({k: v for k, v in p1.items() if k != "capture"})
        restore_capture(env, blob["capture"])
        p2 = coupled_v2(env, blob, cfg=pcfg)
        case["primitive_only"]["V2"] = strip_acts({k: v for k, v in p2.items() if k != "capture"})
    except Exception as exc:
        case["primitive_only"]["error"] = str(exc)

    return case


def run_ablations(env, metas: list[dict], pcfg: PrecisionConfig) -> list[dict]:
    """Diagnostic ablations on the same seven ICs."""
    rows = []
    for meta in metas:
        hid = meta["hard_id"]
        blob = {**meta["blob"], "sample_id": meta["raw_id"], "hard_id": hid}
        print(f"\n--- ablation {hid} ---", flush=True)

        # A1: coupled without return compensation == full-traj V1 (already in paired)
        # A2: return compensation without coupled — tighten working_pos on production DZ only
        try:
            cfg_tight = PrecisionConfig(
                evaluate_handoff=True,
                use_verified_lateral_fallback=False,
                working_pos_mm=max(4.0, float(pcfg.working_pos_mm) - 4.0),
            )
            # Also tighten ez for stop probing via working_pos used in inward slip budget
            r = run_variant(env, blob, plan_id="B", variant="V0", cfg=cfg_tight)
            rows.append(
                {
                    "hard_id": hid,
                    "ablation": "A2_tighter_working_pos_no_coupled",
                    "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                    "SUCCESS_12S": r.get("SUCCESS_12S"),
                    "status": r.get("status"),
                    "dropped": r.get("dropped"),
                    "final_dz_mm": (r.get("errors_exit") or {}).get("dz_mm"),
                    "n_actions": r.get("n_actions"),
                    "note": "production SM with working_pos_mm-=4; no coupled",
                }
            )
            print(f"  A2 NEW={r.get('SUCCESS_NEW')} status={r.get('status')}", flush=True)
        except Exception as exc:
            rows.append({"hard_id": hid, "ablation": "A2_tighter_working_pos_no_coupled", "error": str(exc)})

        # A3: smaller coupled travel
        try:
            r = run_variant(env, blob, plan_id="B", variant="V1", cfg=pcfg, max_safe_travel_mm=10.0)
            rows.append(
                {
                    "hard_id": hid,
                    "ablation": "A3_coupled_smaller_travel_10mm",
                    "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                    "SUCCESS_12S": r.get("SUCCESS_12S"),
                    "status": r.get("status"),
                    "dropped": r.get("dropped"),
                    "coupled_attempted": r.get("coupled_attempted"),
                    "coupled_done": r.get("coupled_done"),
                    "n_actions": r.get("n_actions"),
                }
            )
            print(f"  A3 NEW={r.get('SUCCESS_NEW')} status={r.get('status')}", flush=True)
        except Exception as exc:
            rows.append({"hard_id": hid, "ablation": "A3_coupled_smaller_travel_10mm", "error": str(exc)})

        # A4: force coupled even if only one axis large (timing / applicability)
        try:
            r = run_variant(env, blob, plan_id="B", variant="V1", cfg=pcfg, force_coupled=True)
            rows.append(
                {
                    "hard_id": hid,
                    "ablation": "A4_force_coupled_when_dx_called",
                    "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                    "SUCCESS_12S": r.get("SUCCESS_12S"),
                    "status": r.get("status"),
                    "dropped": r.get("dropped"),
                    "coupled_attempted": r.get("coupled_attempted"),
                    "coupled_done": r.get("coupled_done"),
                    "coupled_reason": (r.get("coupled_result") or {}).get("reason"),
                    "n_actions": r.get("n_actions"),
                }
            )
            print(f"  A4 NEW={r.get('SUCCESS_NEW')} status={r.get('status')}", flush=True)
        except Exception as exc:
            rows.append({"hard_id": hid, "ablation": "A4_force_coupled_when_dx_called", "error": str(exc)})

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rebuild-exits", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--only", type=str, default="", help="comma-separated hard_ids")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    env = make_env()
    pcfg = cfg()

    print("Rebuilding / loading Stage-1 exits...", flush=True)
    metas = rebuild_exits(env, force=args.force_rebuild_exits)
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        metas = [m for m in metas if m["hard_id"] in want]
    print(f"cohort={len(metas)} → {[m['hard_id'] for m in metas]}", flush=True)

    cases = []
    paired_rows = []
    for meta in metas:
        case = run_case(env, meta, pcfg)
        cases.append(jsafe(case))
        for variant in ("V0", "V1", "V2"):
            packed = case.get("variants", {}).get(variant) or {}
            if "error" in packed:
                paired_rows.append({"hard_id": meta["hard_id"], "variant": variant, "error": packed["error"]})
                continue
            tr = case.get("traces", {}).get(variant) or {}
            paired_rows.append(row_metrics(meta["hard_id"], variant, packed, tr))

    (OUT / "PAIRED_CASES.json").write_text(json.dumps(jsafe(cases), indent=2), encoding="utf-8")
    write_csv(OUT / "PAIRED_RESULTS.csv", paired_rows)

    abl_rows = []
    if not args.skip_ablations:
        abl_rows = run_ablations(env, metas, pcfg)
        write_csv(OUT / "ABLATION_RESULTS.csv", abl_rows)
        (OUT / "ABLATION_RESULTS.json").write_text(json.dumps(jsafe(abl_rows), indent=2), encoding="utf-8")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(metas),
        "hard_ids": [m["hard_id"] for m in metas],
        "v0_success_new": sum(1 for r in paired_rows if r.get("variant") == "V0" and r.get("SUCCESS_NEW")),
        "v1_success_new": sum(1 for r in paired_rows if r.get("variant") == "V1" and r.get("SUCCESS_NEW")),
        "v2_success_new": sum(1 for r in paired_rows if r.get("variant") == "V2" and r.get("SUCCESS_NEW")),
        "v0_match_expected": sum(
            1 for r in paired_rows if r.get("variant") == "V0" and r.get("status_matches_expected")
        ),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
