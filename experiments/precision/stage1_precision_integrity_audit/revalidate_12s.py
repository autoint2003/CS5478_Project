#!/usr/bin/env python
"""Independent Stage-1 SUCCESS_12S revalidation for all Precision-related ICs.

Does not modify production code or IC files. Writes under
results/diagnostics/stage1_precision_integrity_audit/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.entry_conditions import entry_gate  # noqa: E402
from data_generation.oracles.precision.handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import sample_state  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/stage1_precision_integrity_audit"
AUDIT = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"
FINAL_MAN = ROOT / "results/data_generation/stage2_abc_final_v2/manifest.jsonl"
FINAL_ICS = ROOT / "results/data_generation/stage2_abc_final_v2/ics"
A_MAN = ROOT / "results/data_generation/plan_a_stage1_terminals_v1/manifest.jsonl"
A_ICS = ROOT / "results/data_generation/plan_a_stage1_terminals_v1/ics"
OFFICIAL = ROOT / "data/experts_v2_official"


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


def capture_fp(capture) -> str:
    raw = pickle.dumps(capture, protocol=4)
    return hashlib.sha256(raw).hexdigest()[:16]


def inventory() -> list[dict]:
    """Deduplicated IC inventory across production + diagnostic consumers."""
    seen: dict[str, dict] = {}

    def add(sid: str, path: Path, *, pool: str, plan: str | None = None, consumers: list[str] | None = None):
        if not path.is_file():
            return
        if sid not in seen:
            seen[sid] = {
                "sample_id": sid,
                "ic_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_pools": set(),
                "plan_families": set(),
                "consumers": set(),
            }
        seen[sid]["source_pools"].add(pool)
        if plan:
            seen[sid]["plan_families"].add(plan)
        for c in consumers or []:
            seen[sid]["consumers"].add(c)

    if FINAL_MAN.is_file():
        for line in FINAL_MAN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            path = FINAL_ICS / f"{sid}.pkl"
            add(sid, path, pool="stage2_abc_final_v2", plan=str(row.get("plan_provenance") or "?"), consumers=["experts_v2_official", "precision_candidate_audit"])

    if A_MAN.is_file():
        for line in A_MAN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            add(sid, A_ICS / f"{sid}.pkl", pool="plan_a_stage1_terminals_v1", plan="A_STAGE1", consumers=["experts_v2_official", "precision_candidate_audit"])

    if AUDIT.is_file():
        for line in AUDIT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            path = ROOT / str(row["ic_path"])
            add(sid, path, pool=str(row.get("source_pool") or "audit"), plan=str(row.get("plan_family") or "?"), consumers=["precision_candidate_audit"])

    # Diagnostic experiment cohorts (by reading paired/result JSON if present)
    for label, rel in [
        ("coupled_feasibility", "results/diagnostics/precision_coupled_feasibility"),
        ("lap", "results/diagnostics/lever_arm_assisted_pivot"),
        ("oar", "results/diagnostics/oscillation_assisted_reorientation"),
        ("gravity_pivot_via_coupled", "results/diagnostics/precision_coupled_feasibility"),
    ]:
        base = ROOT / rel
        for name in ("PAIRED_RESULTS.json", "PAIRED_SINGLE_CYCLE.json", "ORIENT_RESULTS.json", "POSITION_RESULTS.json"):
            p = base / name
            if not p.is_file():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows = data if isinstance(data, list) else data.get("rows") or data.get("results") or []
            if isinstance(rows, dict):
                rows = list(rows.values())
            for r in rows:
                if not isinstance(r, dict):
                    continue
                sid = r.get("sample_id")
                icp = r.get("ic_path")
                if not sid:
                    continue
                path = ROOT / icp if icp else None
                if path is None or not path.is_file():
                    # resolve via audit
                    continue
                add(str(sid), path, pool="diagnostic_experiment", consumers=[label])

    # Official experts: map accepted trajectories back to source IC if present in record
    if OFFICIAL.is_dir():
        for pkl in sorted(OFFICIAL.rglob("*.pkl")):
            try:
                blob = pickle.loads(pkl.read_bytes())
            except Exception:
                continue
            sid = str(blob.get("source_ic_id") or blob.get("sample_id") or blob.get("ic_id") or "")
            ic_path = blob.get("source_ic_path") or blob.get("ic_path")
            if ic_path:
                path = ROOT / str(ic_path)
                if path.is_file() and sid:
                    add(sid, path, pool="official_expert_source", consumers=["experts_v2_official"])

    out = []
    for sid, row in sorted(seen.items()):
        out.append(
            {
                "sample_id": sid,
                "ic_path": row["ic_path"],
                "source_pools": sorted(row["source_pools"]),
                "plan_families": sorted(row["plan_families"]),
                "consumers": sorted(row["consumers"]),
            }
        )
    return out


def original_flags(sample_id: str) -> dict:
    """Recover historical SUCCESS_12S / provenance flags if present."""
    out: dict[str, Any] = {"ORIGINALLY_VERIFIED": False, "original_direct_SUCCESS_12S": None, "original_reason": None, "source_kind": None, "final_category": None}
    if FINAL_MAN.is_file():
        for line in FINAL_MAN.read_text(encoding="utf-8").splitlines():
            if sample_id not in line:
                continue
            row = json.loads(line)
            if row.get("sample_id") != sample_id:
                continue
            d = row.get("direct") or {}
            out["original_direct_SUCCESS_12S"] = d.get("SUCCESS_12S")
            out["original_reason"] = d.get("reason")
            out["source_kind"] = row.get("source_kind")
            out["final_category"] = row.get("final_category")
            out["ORIGINALLY_VERIFIED"] = bool(d.get("SUCCESS_12S"))
            break
    if A_MAN.is_file() and out["original_direct_SUCCESS_12S"] is None:
        for line in A_MAN.read_text(encoding="utf-8").splitlines():
            if sample_id not in line:
                continue
            row = json.loads(line)
            if row.get("sample_id") != sample_id:
                continue
            d = row.get("direct") or {}
            out["original_direct_SUCCESS_12S"] = d.get("SUCCESS_12S")
            out["original_reason"] = d.get("reason")
            out["source_kind"] = "plan_a_stage1_terminal"
            out["final_category"] = row.get("final_category")
            out["ORIGINALLY_VERIFIED"] = bool(d.get("SUCCESS_12S"))
            break
    if AUDIT.is_file():
        for line in AUDIT.read_text(encoding="utf-8").splitlines():
            if sample_id not in line:
                continue
            row = json.loads(line)
            if row.get("sample_id") != sample_id:
                continue
            out["audit_class"] = row.get("class")
            out["audit_grasp_stable"] = row.get("grasp_stable")
            out["audit_entry_gate_ok"] = row.get("entry_gate_ok")
            out["audit_direct_SUCCESS_12S"] = (row.get("direct") or {}).get("SUCCESS_12S")
            out["audit_precision_status"] = (row.get("precision") or {}).get("status")
            out["audit_first_failed_primitive"] = (row.get("precision") or {}).get("first_failed_primitive")
            fails = (row.get("precision") or {}).get("failures") or []
            out["audit_failures"] = fails[:5]
            break
    return out


def revalidate_one(env, meta: dict) -> dict:
    path = ROOT / meta["ic_path"]
    flags = original_flags(meta["sample_id"])
    row = {
        **meta,
        **flags,
        "assessed_utc": datetime.now(timezone.utc).isoformat(),
        "NOMINAL_S": float(NOMINAL_S),
        "LOST_STEPS": int(LOST_STEPS),
        "revalidate_stabilize_n": 0,
    }
    try:
        blob = pickle.loads(path.read_bytes())
    except Exception as exc:
        row["eligibility"] = "UNRESOLVED_PROVENANCE"
        row["error"] = f"load:{exc}"
        return row

    if "capture" not in blob or "ref" not in blob:
        row["eligibility"] = "UNRESOLVED_PROVENANCE"
        row["error"] = "missing_capture_or_ref"
        return row

    row["has_stabilization_prefix"] = bool(blob.get("stabilization_prefix_actions"))
    row["n_stabilization_prefix"] = len(blob.get("stabilization_prefix_actions") or [])
    row["capture_fp"] = capture_fp(blob["capture"])
    row["blob_keys"] = sorted(blob.keys())

    # Entry physics + short gate (historical Precision admission proxy)
    restore_capture(env, blob["capture"])
    phys = {
        "bilateral": bool(sample_state(env.sim).get("bilateral")),
        "fn_tot": float(sample_state(env.sim).get("fn_tot") or 0),
        "dtheta_deg": float(measure_errors(env.sim).get("dtheta_deg") or 0),
        "e_signed_mm": measure_errors(env.sim).get("e_signed_mm"),
        "contact_mode": classify_contact_mode(env.sim).get("mode"),
        "fingertips_down": bool(gates(env.sim).get("fingertips_down")),
        "alpha_up_deg": float(gates(env.sim).get("alpha_up_deg") or 0),
    }
    row["physics_entry"] = jsafe(phys)
    restore_capture(env, blob["capture"])
    gate = entry_gate(env)
    row["entry_gate_ok"] = bool(gate.get("ok"))
    row["entry_gate_hold_s"] = (gate.get("stability") or {}).get("consecutive_bil_s_at_end")
    row["grasp_stable_short"] = bool(gate.get("ok") and phys["bilateral"])

    # Independent 12S from exact snapshot (no stabilize_n, no prefix replay)
    restore_capture(env, blob["capture"])
    t0 = time.time()
    try:
        h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=0)
    except Exception as exc:
        row["eligibility"] = "UNRESOLVED_PROVENANCE"
        row["error"] = f"handoff:{type(exc).__name__}:{exc}"
        row["traceback"] = traceback.format_exc(limit=4)
        return row
    wall = time.time() - t0
    row["revalidate_wall_s"] = wall
    row["revalidate_SUCCESS_12S"] = bool(h.get("SUCCESS_12S"))
    row["revalidate_SUCCESS_NEW"] = bool(h.get("SUCCESS_NEW"))
    row["revalidate_reason"] = h.get("reason") or h.get("std_reason")
    row["revalidate_pre_theta"] = h.get("pre_theta")
    row["revalidate_pre_pos_mm"] = h.get("pre_pos_mm")
    row["revalidate_pre_bilateral"] = h.get("pre_bilateral")
    # failure timing if samples present
    samples = h.get("samples") or []
    fail_t = None
    if not h.get("SUCCESS_12S") and samples:
        for s in samples:
            if not s.get("bilateral"):
                fail_t = s.get("t_rel")
                break
        if fail_t is None and samples:
            fail_t = samples[-1].get("t_rel")
    row["revalidate_fail_t_rel"] = fail_t

    # Classification under established Stage-1 contract = SUCCESS_12S
    if row["revalidate_SUCCESS_12S"]:
        if flags.get("ORIGINALLY_VERIFIED"):
            row["eligibility"] = "ORIGINALLY_VERIFIED"
        else:
            row["eligibility"] = "REVALIDATED_ONLY"
    else:
        row["eligibility"] = "FAILED_12S"

    # Compare short gate vs 12S
    row["short_gate_admits_but_12s_fails"] = bool(row["grasp_stable_short"] and not row["revalidate_SUCCESS_12S"])
    return row


def classify_precision_validity(audit_row: dict | None, elig: str) -> str:
    if audit_row is None:
        return "NO_AUDIT_ROW"
    cls = audit_row.get("class")
    if elig == "FAILED_12S":
        return "INVALID_STAGE1_NOT_PRECISION_FAILURE"
    if elig in ("UNRESOLVED_PROVENANCE",):
        return "UNRESOLVED"
    if cls == "PRECISION_REQUIRED_BUT_FAILED":
        return "GENUINE_PRECISION_FAILURE_AFTER_STAGE1_OK"
    if cls == "PRECISION_REQUIRED_AND_SUCCESSFUL":
        return "GENUINE_PRECISION_SUCCESS_AFTER_STAGE1_OK"
    if cls == "DIRECT_ELIGIBLE":
        return "DIRECT_ELIGIBLE_STAGE1_OK"
    return str(cls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    inv = inventory()
    if args.only:
        inv = [r for r in inv if r["sample_id"] == args.only]
    if args.limit:
        inv = inv[: args.limit]

    # index audit
    audit_by = {}
    if AUDIT.is_file():
        for line in AUDIT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                audit_by[r["sample_id"]] = r

    env = make_env()
    rows = []
    print(f"revalidate {len(inv)} ICs NOMINAL_S={NOMINAL_S} @ {datetime.now(timezone.utc).isoformat()}", flush=True)
    for i, meta in enumerate(inv):
        print(f"[{i+1}/{len(inv)}] {meta['sample_id']}", flush=True)
        try:
            row = revalidate_one(env, meta)
        except Exception as exc:
            row = {**meta, "eligibility": "UNRESOLVED_PROVENANCE", "error": str(exc), "traceback": traceback.format_exc(limit=4)}
        aud = audit_by.get(meta["sample_id"])
        row["precision_validity"] = classify_precision_validity(aud, row.get("eligibility") or "")
        rows.append(row)
        # incremental log
        with (OUT / "logs" / "revalidate.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsafe(row)) + "\n")

    (OUT / "IC_ELIGIBILITY_AUDIT.json").write_text(json.dumps(jsafe(rows), indent=2), encoding="utf-8")

    # CSV flat
    flat_keys = [
        "sample_id", "ic_path", "eligibility", "precision_validity",
        "ORIGINALLY_VERIFIED", "original_direct_SUCCESS_12S", "original_reason", "source_kind",
        "revalidate_SUCCESS_12S", "revalidate_SUCCESS_NEW", "revalidate_reason", "revalidate_fail_t_rel",
        "grasp_stable_short", "entry_gate_ok", "short_gate_admits_but_12s_fails",
        "audit_class", "audit_first_failed_primitive",
        "n_stabilization_prefix", "capture_fp", "source_pools", "consumers",
    ]
    flat = []
    for r in rows:
        flat.append({k: (json.dumps(r[k]) if isinstance(r.get(k), (list, dict)) else r.get(k)) for k in flat_keys})
    with (OUT / "IC_ELIGIBILITY_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys)
        w.writeheader()
        w.writerows(flat)

    # summary
    elig = Counter(r.get("eligibility") for r in rows)
    short_fail = sum(1 for r in rows if r.get("short_gate_admits_but_12s_fails"))
    summary = {
        "n_unique_ics": len(rows),
        "eligibility_counts": dict(elig),
        "short_gate_admits_but_12s_fails": short_fail,
        "precision_validity": dict(Counter(r.get("precision_validity") for r in rows)),
        "NOMINAL_S": NOMINAL_S,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "REVALIDATION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
