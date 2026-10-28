#!/usr/bin/env python
"""Precision candidate mining and physical failure classification.

Non-mutating eligibility assessment on identical restored snapshots, then
official Oracle execution for Precision-required ICs only.

Does not relax SUCCESS_NEW / SUCCESS_12S. Does not enable verified_lateral fallback.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.entry_conditions import assess_need, entry_gate  # noqa: E402
from data_generation.oracles.precision.handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_candidate_audit"
FINAL_MANIFEST = ROOT / "results/data_generation/stage2_abc_final_v2/manifest.jsonl"
FINAL_ICS = ROOT / "results/data_generation/stage2_abc_final_v2/ics"
A_TERM_MANIFEST = ROOT / "results/data_generation/plan_a_stage1_terminals_v1/manifest.jsonl"
A_TERM_ICS = ROOT / "results/data_generation/plan_a_stage1_terminals_v1/ics"


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


def official_cfg() -> PrecisionConfig:
    return PrecisionConfig(
        evaluate_handoff=True,
        use_support_transfer_dx=True,
        use_verified_lateral_fallback=False,
    )


def physics_snap(env) -> dict:
    err = measure_errors(env.sim)
    g = gates(env.sim)
    mode = classify_contact_mode(env.sim)
    return {
        "e_signed_mm": err.get("e_signed_mm"),
        "e_pos_mm": err.get("e_pos_mm"),
        "dtheta_deg": err.get("dtheta_deg"),
        "bilateral": bool(err.get("bilateral")),
        "fn_L": err.get("fn_L"),
        "fn_R": err.get("fn_R"),
        "fn_tot": err.get("fn_tot"),
        "alpha_up_deg": float(g.get("alpha_up_deg") or 0.0),
        "fingertips_down": bool(g.get("fingertips_down")),
        "fingertips_up": bool(g.get("fingertips_up")),
        "contact_mode": mode.get("mode"),
        "support_side": mode.get("support_side"),
        "Fn_z_L": mode.get("Fn_z_L"),
        "Fn_z_R": mode.get("Fn_z_R"),
        "one_sided_mechanical": bool(mode.get("one_sided_mechanical")),
    }


def first_failure_primitive(res) -> str | None:
    fails = list(res.failures or [])
    phases = list(res.phases or [])
    joined = " | ".join(fails + phases)
    keys = (
        ("dtheta", "dtheta"),
        ("DX1", "DX1"),
        ("support_transfer", "support_transfer"),
        ("DX2", "protected_rotation"),
        ("verified_lateral", "verified_lateral"),
        ("dz", "dz"),
        ("FINAL_DOWN", "FINAL_DOWN_GATE"),
        ("LOST_GRASP", "HANDOFF"),
        ("handoff", "HANDOFF"),
        ("entry", "entry_gate"),
    )
    low = joined.lower()
    for needle, label in keys:
        if needle.lower() in low:
            return label
    if res.status == "PRECISION_PRIMITIVE_FAILED":
        return "primitive"
    if "HANDOFF" in str(res.status):
        return "HANDOFF"
    return None


def executed_correction_skills(phases: list[str], logs: dict) -> list[str]:
    """True correction skills only — not NO_REMAINING_CORRECTION / skip / gates."""
    if any("PRECISION_SKIPPED" in p for p in phases):
        return []
    skills: list[str] = []
    for p in phases:
        if "correct_dtheta" in p and "dtheta" not in skills:
            skills.append("dtheta")
        if p in ("correct_dx", "dx_support_transfer") or str(p).startswith("dx_support"):
            if "dx" not in skills:
                skills.append("dx")
        if "correct_dz" in p:
            if "dz" not in skills:
                skills.append("dz")
    mech = logs.get("dx_mechanism")
    if mech == "support_transfer_dx" and "dx" not in skills:
        skills.append("dx")
    if "NO_REMAINING_CORRECTION" in phases and not skills:
        return []
    return skills


def discover_ics() -> list[dict]:
    """Unique post-Stage-1 ICs from final_v2 + A Stage-1 terminals."""
    seen: set[str] = set()
    out: list[dict] = []

    def add(sample_id: str, plan: str, path: Path, pool: str):
        if sample_id in seen or not path.is_file():
            return
        seen.add(sample_id)
        out.append(
            {
                "sample_id": sample_id,
                "plan_family": plan,
                "ic_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_pool": pool,
            }
        )

    if FINAL_MANIFEST.is_file():
        for line in FINAL_MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            plan = str(row.get("plan_provenance") or "?")
            path = FINAL_ICS / f"{sid}.pkl"
            if not path.is_file():
                hits = sorted(FINAL_ICS.glob(f"{sid}*.pkl"))
                path = hits[0] if hits else path
            add(sid, plan, path, "stage2_abc_final_v2")

    if A_TERM_MANIFEST.is_file():
        for line in A_TERM_MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            path = A_TERM_ICS / f"{sid}.pkl"
            add(sid, "A_STAGE1", path, "plan_a_stage1_terminals_v1")

    out.sort(key=lambda r: (r["plan_family"], r["sample_id"]))
    return out


def load_blob(path: Path) -> dict:
    blob = pickle.loads(path.read_bytes())
    if "capture" not in blob or "ref" not in blob:
        raise KeyError("IC missing capture/ref")
    return blob


def classify_one(env, meta: dict, cfg: PrecisionConfig) -> dict:
    path = ROOT / meta["ic_path"]
    row: dict[str, Any] = {
        **meta,
        "assessed_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        blob = load_blob(path)
    except Exception as exc:
        row["class"] = "OUT_OF_SCOPE_OR_INVALID_IC"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    # --- non-mutating assessment on restored snapshot ---
    restore_capture(env, blob["capture"])
    gate = entry_gate(env)
    restore_capture(env, blob["capture"])
    phys0 = physics_snap(env)
    need = assess_need(measure_errors(env.sim), cfg)
    correction_needed = bool(
        need.get("dtheta_needed") or need.get("dx_needed") or need.get("dz_needed")
    )
    row["entry_gate_ok"] = bool(gate.get("ok"))
    row["grasp_stable"] = bool(gate.get("ok") and phys0.get("bilateral"))
    row["physics_entry"] = jsafe(phys0)
    row["assess_need"] = jsafe(need)
    row["correction_needed"] = correction_needed
    row["target_pose_compliant"] = bool(need.get("ready_for_direct_handoff_assessment"))

    if not row["grasp_stable"]:
        row["class"] = "UNSTABLE_AT_ENTRY"
        row["direct"] = None
        row["precision"] = None
        return row

    # Authoritative Stage-1 SUCCESS_12S (short grasp_stable is not enough)
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible

    blob_ad = {"capture": blob["capture"], "ref": blob["ref"], "sample_id": meta.get("sample_id")}
    elig = check_stage1_eligible(env, blob_ad, allow_live_verify=True, record_live=False)
    row["stage1_eligibility"] = jsafe(elig.to_dict())
    row["stage1_SUCCESS_12S"] = bool(elig.SUCCESS_12S)
    if not elig.eligible:
        row["class"] = "STAGE1_INELIGIBLE"
        row["precision_required"] = False
        row["direct"] = None
        row["precision"] = None
        row["note"] = f"STAGE1_INELIGIBLE:{elig.reason}"
        return row

    # Direct probe from identical restored capture (mutates then restores inside probe)
    restore_capture(env, blob["capture"])
    direct = probe_direct_handoff(env, blob["capture"], blob["ref"])
    row["direct"] = {
        "SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
        "reason": direct.get("reason"),
    }

    if direct.get("SUCCESS_NEW"):
        row["class"] = "DIRECT_ELIGIBLE"
        row["precision_required"] = False
        row["precision"] = None
        # Note residual mismatch if assess asked for correction but Direct still OK
        row["note"] = (
            "direct_SUCCESS_NEW; assess residual flags recorded but IC is Direct-eligible"
            if correction_needed
            else "direct_SUCCESS_NEW and target-ready"
        )
        return row

    if not correction_needed:
        # Direct failed but assess says no residual correction — posture / gate / other
        row["precision_required"] = False
        row["class"] = "OUT_OF_SCOPE_OR_INVALID_IC"
        row["note"] = "direct_failed_without_assessed_dtheta_dx_dz"
        row["precision"] = None
        return row

    # Precision genuinely required (only on Stage-1 eligible ICs)
    row["precision_required"] = True
    restore_capture(env, blob["capture"])
    state = {"capture": blob["capture"], "ref": blob["ref"], "sample_id": meta.get("sample_id")}
    res = run_precision(env, state, str(meta["plan_family"]), cfg)
    if res.status == "STAGE1_INELIGIBLE":
        row["class"] = "STAGE1_INELIGIBLE"
        row["precision_required"] = False
        row["precision"] = {"status": res.status, "failures": list(res.failures or []), "note": res.note}
        row["note"] = "run_precision_rejected_stage1"
        return row
    logs = res.primitive_logs or {}
    phases = list(res.phases or [])
    skills = executed_correction_skills(phases, logs)
    handoff = res.handoff or {}
    prec = {
        "run_precision_called": True,
        "status": res.status,
        "phases": phases,
        "failures": list(res.failures or []),
        "dx_mechanism": logs.get("dx_mechanism"),
        "legacy_fallback_explicit": bool(logs.get("legacy_fallback_explicit")),
        "executed_correction_skills": skills,
        "nontrivial_correction": bool(skills),
        "first_failed_primitive": first_failure_primitive(res),
        "SUCCESS_NEW": bool(handoff.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(handoff.get("SUCCESS_12S")),
        "n_actions": len(res.actions or []),
        "precision_skipped": bool(res.precision_skipped),
        "precision_executed": bool(res.precision_executed),
        "physics_exit": jsafe(physics_snap(env)),
        "note": res.note,
    }
    # Persist actions for video replay
    act_path = OUT / "precision_actions" / f"{meta['sample_id']}.npz"
    act_path.parent.mkdir(parents=True, exist_ok=True)
    if res.actions:
        np.savez_compressed(act_path, actions=np.asarray(res.actions, np.float32))
        prec["actions_path"] = str(act_path.relative_to(ROOT)).replace("\\", "/")
    row["precision"] = prec

    if prec["SUCCESS_NEW"] and prec["nontrivial_correction"]:
        row["class"] = "PRECISION_REQUIRED_AND_SUCCESSFUL"
    elif prec["SUCCESS_NEW"] and not prec["nontrivial_correction"]:
        # Oracle succeeded without true correction skills — not counted as Precision skill demo
        row["class"] = "PRECISION_REQUIRED_BUT_FAILED"
        row["note"] = "SUCCESS_NEW without nontrivial correction skills (skip/NO_REMAINING); not a Precision skill demo"
        row["subclass"] = "success_without_correction_skill"
    else:
        row["class"] = "PRECISION_REQUIRED_BUT_FAILED"
    return row


def paired_matrix(rows: list[dict]) -> dict:
    """Direct-vs-Precision outcome matrix for Precision-required and Direct-eligible."""
    cells = Counter()
    for r in rows:
        d = (r.get("direct") or {})
        p = (r.get("precision") or {})
        d_ok = bool(d.get("SUCCESS_NEW"))
        d_12 = bool(d.get("SUCCESS_12S"))
        if r.get("class") == "DIRECT_ELIGIBLE":
            cells["direct_ok_precision_not_run"] += 1
            continue
        if not r.get("precision_required"):
            continue
        p_ok = bool(p.get("SUCCESS_NEW"))
        p_skill = bool(p.get("nontrivial_correction"))
        if d_ok and not p_ok:
            cells["direct_ok_precision_fail"] += 1
        elif (not d_ok) and p_ok and p_skill:
            cells["direct_fail_precision_ok_skill"] += 1
        elif (not d_ok) and p_ok and not p_skill:
            cells["direct_fail_precision_ok_no_skill"] += 1
        elif (not d_ok) and not p_ok:
            cells["both_fail"] += 1
        elif d_ok and p_ok:
            cells["both_ok"] += 1
        # Direct survives 12s but misses SUCCESS_NEW pose
        if d_12 and not d_ok and not p_ok:
            cells["direct_12s_miss_target_precision_fail"] += 1
        if d_12 and not d_ok and p_ok:
            cells["direct_12s_miss_target_precision_ok"] += 1
    return dict(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-plan", type=str, default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ics = discover_ics()
    if args.only_plan:
        ics = [x for x in ics if x["plan_family"] == args.only_plan]
    if args.limit and args.limit > 0:
        ics = ics[: args.limit]

    cfg = official_cfg()
    assert cfg.use_verified_lateral_fallback is False
    env = make_env()
    rows: list[dict] = []
    class_path = OUT / "CLASSIFICATION.jsonl"
    if class_path.is_file():
        class_path.unlink()

    print(f"auditing {len(ics)} ICs...", flush=True)
    for i, meta in enumerate(ics):
        print(f"[{i+1}/{len(ics)}] {meta['plan_family']} {meta['sample_id']}", flush=True)
        try:
            row = classify_one(env, meta, cfg)
        except Exception as exc:
            row = {
                **meta,
                "class": "OUT_OF_SCOPE_OR_INVALID_IC",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        rows.append(row)
        with class_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsafe(row), ensure_ascii=False) + "\n")
        print(f"  -> {row.get('class')} direct={((row.get('direct') or {}).get('SUCCESS_NEW'))}", flush=True)

    by_class = dict(Counter(r.get("class") for r in rows))
    prec_fail = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_BUT_FAILED"]
    prec_ok = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL"]
    prim_fail = Counter(
        (r.get("precision") or {}).get("first_failed_primitive") or "none" for r in prec_fail
    )
    skills_ok = Counter()
    for r in prec_ok:
        sk = (r.get("precision") or {}).get("executed_correction_skills") or []
        skills_ok["|".join(sk) if sk else "none"] += 1

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_ics": len(rows),
        "by_class": by_class,
        "paired_matrix": paired_matrix(rows),
        "primitive_failure_counts": dict(prim_fail),
        "successful_nontrivial_skills": dict(skills_ok),
        "n_precision_required_failed": len(prec_fail),
        "n_precision_required_successful": len(prec_ok),
        "fallback_enabled": False,
        "precision_required_failed_ids": [r["sample_id"] for r in prec_fail],
        "precision_required_successful_ids": [r["sample_id"] for r in prec_ok],
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "PAIRED_MATRIX.json").write_text(
        json.dumps(jsafe(summary["paired_matrix"]), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["by_class"], indent=2), flush=True)
    print("paired", summary["paired_matrix"], flush=True)
    print(f"wrote {class_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
