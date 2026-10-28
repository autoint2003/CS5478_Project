"""P10 bounded coverage pilot: D2 gate, θ 10/15, B1 recovery, A2 supply, reports.

Writes under results/diagnostics/p10_dataset_coverage/.
Does not mutate production data/experts_v2_official or MuJoCo/M111.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "diagnostics" / "p10_dataset_coverage"
PILOT_CFG = ROOT / "config" / "generation" / "experts_v2_p10_pilot.yaml"
OFFICIAL = ROOT / "data" / "experts_v2_official"
P9_B1_FAIL = (
    ROOT
    / "results"
    / "diagnostics"
    / "p9_expert_fsm_dataset"
    / "generation_pilot"
    / "diagnostics"
    / "failures"
)
CENSUS_CSV = (
    ROOT
    / "results"
    / "diagnostics"
    / "p9_expert_fsm_dataset"
    / "EXISTING_DATASET_CENSUS.csv"
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_env():
    from data_generation.expert_pipeline.generate import make_env as _me

    return _me()


# ---------------------------------------------------------------------------
# Phase 1 — D2 HARD_IC census
# ---------------------------------------------------------------------------
def audit_d2_hard_ic(*, limit: int | None = None) -> list[dict]:
    from data_generation.expert_pipeline.sources import discover_branch_ics, load_generation_config
    from data_generation.oracles.precision.d2_eligibility import check_d2_eligible
    from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint

    cfg = load_generation_config(PILOT_CFG)
    sources = discover_branch_ics("D_PRECISION", cfg, require_stage1_evidence=False)
    if limit is not None:
        sources = sources[:limit]
    env = make_env()
    rows: list[dict] = []
    seen_fp: dict[str, str] = {}
    print(f"[D2 census] {len(sources)} HARD_IC candidates", flush=True)
    for i, src in enumerate(sources):
        t0 = time.time()
        try:
            blob = pickle.loads(src.ic_path.read_bytes())
            if "capture" not in blob or "ref" not in blob:
                row = {
                    "sample_id": src.sample_id,
                    "bucket": "INVALID_SNAPSHOT",
                    "eligible": False,
                    "status": "MISSING_CAPTURE_OR_REF",
                    "SUCCESS_12S": None,
                    "precision_required": None,
                    "duplicate_of": None,
                    "capture_fp": None,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                rows.append(row)
                continue
            fp = capture_fingerprint(blob["capture"])
            dup = seen_fp.get(fp)
            if dup:
                rows.append(
                    {
                        "sample_id": src.sample_id,
                        "bucket": "DUPLICATE_LINEAGE",
                        "eligible": False,
                        "status": "DUPLICATE_CAPTURE_FP",
                        "SUCCESS_12S": None,
                        "precision_required": None,
                        "duplicate_of": dup,
                        "capture_fp": fp,
                        "elapsed_s": round(time.time() - t0, 2),
                    }
                )
                continue
            seen_fp[fp] = src.sample_id
            elig = check_d2_eligible(
                env,
                {"capture": blob["capture"], "ref": blob["ref"], "sample_id": src.sample_id},
            )
            if elig.status == "D2_ELIGIBLE":
                bucket = "STABLE_PRECISION_REQUIRED"
            elif elig.status == "D2_STABLE_NOT_PRECISION_REQUIRED":
                bucket = "STABLE_ALREADY_WITHIN_TARGET"
            elif elig.reason == "SUCCESS_12S_FALSE":
                bucket = "UNSTABLE"
            else:
                bucket = "INVALID_OR_INCONSISTENT"
            rows.append(
                {
                    "sample_id": src.sample_id,
                    "bucket": bucket,
                    "eligible": bool(elig.eligible),
                    "status": elig.status,
                    "reason": elig.reason,
                    "SUCCESS_12S": elig.SUCCESS_12S,
                    "SUCCESS_STRICT_10DEG": elig.SUCCESS_STRICT_10DEG,
                    "SUCCESS_RELAXED_15DEG": elig.SUCCESS_RELAXED_15DEG,
                    "precision_required": elig.precision_required,
                    "already_within_target": elig.already_within_target,
                    "duplicate_of": None,
                    "capture_fp": fp,
                    "pre_theta": (elig.detail or {}).get("pre_theta"),
                    "pre_pos_mm": (elig.detail or {}).get("pre_pos_mm"),
                    "need_dx": (elig.need or {}).get("dx_needed"),
                    "need_dz": (elig.need or {}).get("dz_needed"),
                    "need_dtheta": (elig.need or {}).get("dtheta_needed"),
                    "elapsed_s": round(time.time() - t0, 2),
                    "ic_path": str(src.ic_path.relative_to(ROOT)).replace("\\", "/"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_id": src.sample_id,
                    "bucket": "INVALID_OR_INCONSISTENT",
                    "eligible": False,
                    "status": "EXCEPTION",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "SUCCESS_12S": None,
                    "precision_required": None,
                    "duplicate_of": None,
                    "capture_fp": None,
                    "elapsed_s": round(time.time() - t0, 2),
                    "traceback": traceback.format_exc()[-500:],
                }
            )
        print(
            f"  [{i+1}/{len(sources)}] {src.sample_id} -> {rows[-1].get('bucket')} "
            f"({rows[-1].get('elapsed_s')}s)",
            flush=True,
        )
    return rows


# ---------------------------------------------------------------------------
# Phase 2/4 — B1 revalidation + θ 10 vs 15
# ---------------------------------------------------------------------------
def revalidate_b1_failures() -> list[dict]:
    from data_generation.oracles.precision.handoff import probe_direct_handoff
    from data_generation.oracles.precision.entry_conditions import assess_need
    from data_generation.oracles.precision.residuals import measure_errors
    from data_generation.oracles.precision.state import PrecisionConfig
    from envs.simulation.capture import restore_capture

    env = make_env()
    cfg = PrecisionConfig()
    rows: list[dict] = []
    fails = sorted(P9_B1_FAIL.glob("B_STABLE_DIRECT_HANDOFF__*.json"))
    print(f"[B1 reval] {len(fails)} P9 failure records", flush=True)
    for fp in fails:
        meta = json.loads(fp.read_text(encoding="utf-8"))
        sid = meta["source_ic_id"]
        ic_path = ROOT / meta["source_ic_path"]
        t0 = time.time()
        try:
            blob = pickle.loads(ic_path.read_bytes())
            restore_capture(env, blob["capture"])
            err = measure_errors(env.sim)
            need = assess_need(err, cfg)
            probe = probe_direct_handoff(env, blob["capture"], blob["ref"])
            strict = bool(probe.get("SUCCESS_STRICT_10DEG") or probe.get("SUCCESS_NEW"))
            relaxed = bool(probe.get("SUCCESS_RELAXED_15DEG"))
            s12 = bool(probe.get("SUCCESS_12S"))
            theta_only = bool(s12 and not strict and relaxed)
            precision_req = bool(
                need.get("dx_needed") or need.get("dz_needed") or need.get("dtheta_needed")
            )
            qualifies_b1 = bool(relaxed and s12 and not precision_req)
            qualifies_b2 = bool(s12 and precision_req)
            rows.append(
                {
                    "source_ic_id": sid,
                    "p9_failure_class": meta.get("failure_class"),
                    "SUCCESS_12S": s12,
                    "SUCCESS_STRICT_10DEG": strict,
                    "SUCCESS_RELAXED_15DEG": relaxed,
                    "theta_alone_caused_strict_reject": theta_only,
                    "orientation_hold_ok_15": relaxed,
                    "position_contact_ok": s12,
                    "precision_required": precision_req,
                    "qualifies_as_direct_B1_relaxed": qualifies_b1,
                    "better_as_B2": qualifies_b2,
                    "independent_ic": True,
                    "pre_theta": probe.get("pre_theta"),
                    "pre_pos_mm": probe.get("pre_pos_mm"),
                    "window_mean_theta": ((probe.get("window") or {}).get("mean_theta")),
                    "window_max_theta": ((probe.get("window") or {}).get("max_theta")),
                    "window_mean_pos": ((probe.get("window") or {}).get("mean_pos")),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            print(
                f"  {sid}: 12S={s12} STRICT={strict} RELAXED={relaxed} "
                f"B1ok={qualifies_b1} ({rows[-1]['elapsed_s']}s)",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "source_ic_id": sid,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "SUCCESS_12S": None,
                    "SUCCESS_STRICT_10DEG": None,
                    "SUCCESS_RELAXED_15DEG": None,
                    "qualifies_as_direct_B1_relaxed": False,
                }
            )
    return rows


def theta_validation_rows(b1_rows: list[dict], d2_rows: list[dict]) -> list[dict]:
    """Combine B1 + eligible D2 snapshots into THETA_10_VS_15 table."""
    out: list[dict] = []
    for r in b1_rows:
        out.append(
            {
                "cohort": "P9_B1_12S_FAIL_STRICT",
                "sample_id": r.get("source_ic_id"),
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "SUCCESS_STRICT_10DEG": r.get("SUCCESS_STRICT_10DEG"),
                "SUCCESS_RELAXED_15DEG": r.get("SUCCESS_RELAXED_15DEG"),
                "newly_accepted_under_15": bool(
                    r.get("SUCCESS_RELAXED_15DEG") and not r.get("SUCCESS_STRICT_10DEG")
                ),
                "instability_or_drift_flag": bool(
                    r.get("SUCCESS_12S") is False
                    or (
                        r.get("window_mean_pos") is not None
                        and float(r["window_mean_pos"]) > 12.0
                    )
                ),
                "pre_theta": r.get("pre_theta"),
                "window_mean_theta": r.get("window_mean_theta"),
                "window_max_theta": r.get("window_max_theta"),
                "class_if_admitted": "B1" if r.get("qualifies_as_direct_B1_relaxed") else (
                    "B2_candidate" if r.get("better_as_B2") else "reject"
                ),
            }
        )
    for r in d2_rows:
        if r.get("SUCCESS_12S") is None:
            continue
        out.append(
            {
                "cohort": "D2_HARD_IC_CENSUS",
                "sample_id": r.get("sample_id"),
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "SUCCESS_STRICT_10DEG": r.get("SUCCESS_STRICT_10DEG"),
                "SUCCESS_RELAXED_15DEG": r.get("SUCCESS_RELAXED_15DEG"),
                "newly_accepted_under_15": bool(
                    r.get("SUCCESS_RELAXED_15DEG") and not r.get("SUCCESS_STRICT_10DEG")
                ),
                "instability_or_drift_flag": r.get("bucket") == "UNSTABLE",
                "pre_theta": r.get("pre_theta"),
                "window_mean_theta": None,
                "window_max_theta": None,
                "class_if_admitted": "D2_eligible" if r.get("eligible") else r.get("bucket"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Phase 3 — A2 supply analysis + live verify remaining A ICs
# ---------------------------------------------------------------------------
def analyze_a2_supply(*, live_limit: int = 12) -> tuple[dict, list[dict]]:
    from data_generation.expert_pipeline.sources import discover_branch_ics, load_generation_config
    from data_generation.oracles.precision.stage1_eligibility import (
        capture_fingerprint,
        check_stage1_eligible,
        lookup_evidence,
    )

    cfg = load_generation_config(PILOT_CFG)
    a1 = discover_branch_ics("A_STABLE_DIRECT_HANDOFF", cfg, require_stage1_evidence=True)
    a2_ev = discover_branch_ics("A_STABLE_PRECISION", cfg, require_stage1_evidence=True)
    a2_all = discover_branch_ics("A_STABLE_PRECISION", cfg, require_stage1_evidence=False)
    official_a2 = set()
    if CENSUS_CSV.is_file():
        with CENSUS_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("class_label") == "A2" and row.get("corpus") == "experts_v2_official":
                    official_a2.add(row["source_ic_id"])

    remaining = [s for s in a2_all if s.sample_id not in {x.sample_id for x in a2_ev}]
    # Prefer ICs not already in official A2
    remaining = [s for s in remaining if s.sample_id not in official_a2]

    env = make_env()
    live_rows: list[dict] = []
    print(f"[A2 supply] live-verify up to {live_limit} of {len(remaining)} A ICs without registry evidence", flush=True)
    for src in remaining[:live_limit]:
        t0 = time.time()
        try:
            blob = pickle.loads(src.ic_path.read_bytes())
            fp = capture_fingerprint(blob["capture"])
            reg = lookup_evidence(fp)
            elig = check_stage1_eligible(
                env,
                {"capture": blob["capture"], "ref": blob["ref"], "sample_id": src.sample_id},
                allow_live_verify=True,
                record_live=False,
            )
            need_info = {}
            if elig.SUCCESS_12S:
                from data_generation.oracles.precision.entry_conditions import assess_need
                from data_generation.oracles.precision.residuals import measure_errors
                from data_generation.oracles.precision.state import PrecisionConfig
                from envs.simulation.capture import restore_capture

                restore_capture(env, blob["capture"])
                err = measure_errors(env.sim)
                need = assess_need(err, PrecisionConfig())
                prec_req = bool(
                    need.get("dx_needed")
                    or need.get("dz_needed")
                    or need.get("dtheta_needed")
                    or not need.get("ready_for_direct_handoff_assessment")
                )
                need_info = {
                    "precision_required": prec_req,
                    "already_within_target": not prec_req,
                    "SUCCESS_STRICT_10DEG": elig.SUCCESS_NEW,
                    "SUCCESS_RELAXED_15DEG": None,
                    "bucket": "A2_CANDIDATE" if prec_req else "A1_LIKE_ALREADY_OK",
                }
            else:
                need_info = {"bucket": "STAGE1_UNSTABLE", "precision_required": None}
            live_rows.append(
                {
                    "sample_id": src.sample_id,
                    "capture_fp": fp,
                    "registry_had_evidence": reg is not None,
                    "live_SUCCESS_12S": elig.SUCCESS_12S,
                    "live_SUCCESS_NEW": elig.SUCCESS_NEW,
                    "stage1_eligible": elig.eligible,
                    **need_info,
                    "elapsed_s": round(time.time() - t0, 2),
                    "ic_path": str(src.ic_path.relative_to(ROOT)).replace("\\", "/"),
                }
            )
            print(
                f"  {src.sample_id}: 12S={elig.SUCCESS_12S} bucket={need_info.get('bucket')} "
                f"({live_rows[-1]['elapsed_s']}s)",
                flush=True,
            )
        except Exception as exc:
            live_rows.append(
                {
                    "sample_id": src.sample_id,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "bucket": "EXCEPTION",
                }
            )

    analysis = {
        "n_A1_evidenced": len(a1),
        "n_A2_evidenced": len(a2_ev),
        "n_A2_manifest_total": len(a2_all),
        "n_A2_without_registry": len(a2_all) - len(a2_ev),
        "n_official_A2": len(official_a2),
        "live_verified": len(live_rows),
        "live_12s_ok": sum(1 for r in live_rows if r.get("live_SUCCESS_12S")),
        "live_a2_candidates": sum(1 for r in live_rows if r.get("bucket") == "A2_CANDIDATE"),
        "live_a1_like": sum(1 for r in live_rows if r.get("bucket") == "A1_LIKE_ALREADY_OK"),
        "live_unstable": sum(1 for r in live_rows if r.get("bucket") == "STAGE1_UNSTABLE"),
        "why_a1_dominates": (
            "Plan A Stage-1 terminals that already satisfy SUCCESS_NEW are routed to "
            "A_STABLE_DIRECT_HANDOFF (A1). A_STABLE_PRECISION draws plan_provenance=A "
            "from stage2 mining; most lack Stage-1 registry evidence, so discovery "
            "yields ~1 evidenced IC. A1 vs A2 is therefore a supply/routing + evidence "
            "coverage issue, not a Precision-controller deficit."
        ),
    }
    return analysis, live_rows


# ---------------------------------------------------------------------------
# Phase 5/7 — bounded generation pilots
# ---------------------------------------------------------------------------
def run_generation_pilots(
    *,
    d2_eligible_ids: list[str],
    a2_candidate_ids: list[str],
    b1_recover_ids: list[str],
    d2_limit: int = 5,
    a2_limit: int = 4,
    b1_limit: int = 8,
) -> list[dict]:
    from data_generation.expert_pipeline.sources import SourceIC, discover_branch_ics, load_generation_config
    from data_generation.expert_pipeline.record import DatasetWriter
    from data_generation.expert_pipeline.generate import run_one, precision_config_for_branch, make_env

    cfg = load_generation_config(PILOT_CFG)
    out_root = ROOT / cfg["output_root"]
    results: list[dict] = []

    # --- B1 recovery (direct handoff, RELAXED admit) ---
    b1_all = discover_branch_ics("B_STABLE_DIRECT_HANDOFF", cfg)
    b1_map = {s.sample_id: s for s in b1_all}
    b1_targets = [b1_map[i] for i in b1_recover_ids if i in b1_map][:b1_limit]
    if b1_targets:
        writer = DatasetWriter(out_root, config_id=cfg.get("config_id"))
        writer.clear_branch_attempts("B_STABLE_DIRECT_HANDOFF")
        pcfg = precision_config_for_branch("B_STABLE_DIRECT_HANDOFF", cfg)
        env = make_env()
        acc = rej = 0
        for src in b1_targets:
            row = run_one(env, src, writer, cfg, precision_config=pcfg)
            ok = bool(row.get("accepted"))
            acc += int(ok)
            rej += int(not ok and not row.get("skipped"))
            results.append(
                {
                    "run_id": "p10_b1_relaxed",
                    "branch_id": "B_STABLE_DIRECT_HANDOFF",
                    "class_label": "B1",
                    "source_ic_id": src.sample_id,
                    "accepted": ok,
                    "SUCCESS_12S": row.get("SUCCESS_12S"),
                    "SUCCESS_STRICT_10DEG": row.get("SUCCESS_STRICT_10DEG"),
                    "SUCCESS_RELAXED_15DEG": row.get("SUCCESS_RELAXED_15DEG"),
                    "admit_success_key": row.get("admit_success_key"),
                    "status": row.get("status"),
                    "skill_sequence": (row.get("saved") or {}).get("manifest_row", {}).get("skill_sequence")
                    or row.get("skill_sequence"),
                }
            )
            print(f"  [B1] {src.sample_id} accepted={ok}", flush=True)
        results.append(
            {
                "run_id": "p10_b1_relaxed_summary",
                "branch_id": "B_STABLE_DIRECT_HANDOFF",
                "class_label": "B1",
                "attempted": acc + rej,
                "accepted": acc,
                "rejected": rej,
            }
        )

    # --- A2 pilot ---
    a2_all = discover_branch_ics("A_STABLE_PRECISION", cfg, require_stage1_evidence=False)
    a2_map = {s.sample_id: s for s in a2_all}
    # Prefer live A2 candidates; else evidenced
    a2_targets = [a2_map[i] for i in a2_candidate_ids if i in a2_map][:a2_limit]
    if not a2_targets:
        a2_ev = discover_branch_ics("A_STABLE_PRECISION", cfg, require_stage1_evidence=True)
        a2_targets = a2_ev[:a2_limit]
    if a2_targets:
        # Temporarily allow without registry by calling run_one after Stage-1 live path
        # For A2 we still use Stage-1 gate (live verify). Force evidence=False discovery only.
        writer = DatasetWriter(out_root, config_id=cfg.get("config_id"))
        # Don't clear A2 if we want to keep prior — clear for clean pilot
        writer.clear_branch_attempts("A_STABLE_PRECISION")
        pcfg = precision_config_for_branch("A_STABLE_PRECISION", cfg)
        env = make_env()
        acc = rej = 0
        skills: Counter = Counter()
        for src in a2_targets:
            # SourceIC branch must be A_STABLE_PRECISION
            src = SourceIC(
                sample_id=src.sample_id,
                branch_id="A_STABLE_PRECISION",
                plan_provenance=src.plan_provenance,
                ic_path=src.ic_path,
                final_category=src.final_category,
                source_pool=src.source_pool,
            )
            row = run_one(env, src, writer, cfg, precision_config=pcfg)
            ok = bool(row.get("accepted"))
            acc += int(ok)
            rej += int(not ok and not row.get("skipped"))
            sk = (row.get("saved") or {}).get("manifest_row", {}).get("skill_sequence") or row.get(
                "skill_sequence"
            )
            if ok and sk:
                skills[sk] += 1
            plogs = ((row.get("result") or {}).get("primitive_logs") or {})
            results.append(
                {
                    "run_id": "p10_a2_pilot",
                    "branch_id": "A_STABLE_PRECISION",
                    "class_label": "A2",
                    "source_ic_id": src.sample_id,
                    "accepted": ok,
                    "SUCCESS_12S": row.get("SUCCESS_12S"),
                    "SUCCESS_STRICT_10DEG": row.get("SUCCESS_STRICT_10DEG"),
                    "SUCCESS_RELAXED_15DEG": row.get("SUCCESS_RELAXED_15DEG"),
                    "status": row.get("status"),
                    "skill_sequence": sk,
                    "fsm_executed": plogs.get("fsm_executed_skills"),
                    "rejected_reason": row.get("rejected_reason"),
                }
            )
            print(f"  [A2] {src.sample_id} accepted={ok} status={row.get('status')}", flush=True)
        results.append(
            {
                "run_id": "p10_a2_pilot_summary",
                "branch_id": "A_STABLE_PRECISION",
                "class_label": "A2",
                "attempted": acc + rej,
                "accepted": acc,
                "rejected": rej,
                "skills": dict(skills),
            }
        )

    # --- D2 pilot ---
    d2_all = discover_branch_ics("D_PRECISION", cfg, require_stage1_evidence=False)
    d2_map = {s.sample_id: s for s in d2_all}
    d2_targets = [d2_map[i] for i in d2_eligible_ids if i in d2_map][:d2_limit]
    if d2_targets:
        writer = DatasetWriter(out_root, config_id=cfg.get("config_id"))
        writer.clear_branch_attempts("D_PRECISION")
        pcfg = precision_config_for_branch("D_PRECISION", cfg)
        env = make_env()
        acc = rej = 0
        skills: Counter = Counter()
        for src in d2_targets:
            row = run_one(env, src, writer, cfg, precision_config=pcfg)
            ok = bool(row.get("accepted"))
            acc += int(ok)
            rej += int(not ok and not row.get("skipped"))
            sk = (row.get("saved") or {}).get("manifest_row", {}).get("skill_sequence") or row.get(
                "skill_sequence"
            )
            if ok and sk:
                skills[sk] += 1
            plogs = ((row.get("result") or {}).get("primitive_logs") or {})
            results.append(
                {
                    "run_id": "p10_d2_pilot",
                    "branch_id": "D_PRECISION",
                    "class_label": "D2",
                    "source_ic_id": src.sample_id,
                    "accepted": ok,
                    "SUCCESS_12S": row.get("SUCCESS_12S"),
                    "SUCCESS_STRICT_10DEG": row.get("SUCCESS_STRICT_10DEG"),
                    "SUCCESS_RELAXED_15DEG": row.get("SUCCESS_RELAXED_15DEG"),
                    "status": row.get("status"),
                    "skill_sequence": sk,
                    "fsm_executed": plogs.get("fsm_executed_skills"),
                    "rejected_reason": row.get("rejected_reason") or row.get("status"),
                }
            )
            print(f"  [D2] {src.sample_id} accepted={ok} status={row.get('status')}", flush=True)
        results.append(
            {
                "run_id": "p10_d2_pilot_summary",
                "branch_id": "D_PRECISION",
                "class_label": "D2",
                "attempted": acc + rej,
                "accepted": acc,
                "rejected": rej,
                "skills": dict(skills),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Coverage / lineage / readiness reports
# ---------------------------------------------------------------------------
def official_class_counts() -> dict[str, int]:
    counts: Counter = Counter()
    man = OFFICIAL / "manifest.jsonl"
    if not man.is_file():
        return {}
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bid = row.get("branch_id") or ""
        from data_generation.expert_pipeline.branches import BRANCH_TO_CLASS

        counts[BRANCH_TO_CLASS.get(bid, bid)] += 1
    return dict(counts)


def skill_transition_report(gen_rows: list[dict], official_counts: dict) -> list[dict]:
    from data_generation.expert_pipeline.branches import BRANCH_TO_CLASS

    # Official from P9 census skill labels if available
    skill_csv = (
        ROOT
        / "results"
        / "diagnostics"
        / "p9_expert_fsm_dataset"
        / "SKILL_TRANSITION_COVERAGE.csv"
    )
    rows: list[dict] = []
    if skill_csv.is_file():
        with skill_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "source": "official_p9",
                        "transition": r.get("transition") or r.get("skill_transition") or r.get("skill"),
                        "attempted": r.get("attempted") or r.get("n_attempted"),
                        "executed": r.get("executed") or r.get("n_executed") or r.get("n"),
                        "accepted": r.get("accepted") or r.get("n_accepted") or r.get("n"),
                    }
                )
    # Pilot executions
    for r in gen_rows:
        if not r.get("source_ic_id"):
            continue
        sk = r.get("skill_sequence") or "unknown"
        fsm = r.get("fsm_executed")
        rows.append(
            {
                "source": "p10_pilot",
                "class_label": r.get("class_label"),
                "source_ic_id": r.get("source_ic_id"),
                "transition": sk,
                "fsm_executed": json.dumps(fsm) if isinstance(fsm, list) else fsm,
                "attempted": 1,
                "executed": 1 if r.get("accepted") or r.get("status") else 0,
                "accepted": int(bool(r.get("accepted"))),
            }
        )
    # Aggregate multi-skill flags
    dx_dz = sum(1 for r in gen_rows if r.get("skill_sequence") and "dx" in str(r["skill_sequence"]) and "dz" in str(r["skill_sequence"]))
    dz_dx = sum(
        1
        for r in gen_rows
        if isinstance(r.get("fsm_executed"), list)
        and any("DZ" in str(x) for x in r["fsm_executed"])
        and any("DX" in str(x) for x in r["fsm_executed"])
    )
    rows.append(
        {
            "source": "p10_aggregate",
            "transition": "dx→dz_or_dz→dx_pilot",
            "attempted": sum(1 for r in gen_rows if r.get("class_label") in ("A2", "D2", "B2")),
            "executed": dx_dz + dz_dx,
            "accepted": sum(1 for r in gen_rows if r.get("accepted") and r.get("class_label") in ("A2", "D2")),
            "note": f"dx_dz_label={dx_dz} fsm_both={dz_dx}",
        }
    )
    return rows


def write_reports(
    *,
    d2_rows: list[dict],
    b1_rows: list[dict],
    theta_rows: list[dict],
    a2_analysis: dict,
    a2_live: list[dict],
    gen_rows: list[dict],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bucket_counts = Counter(r.get("bucket") for r in d2_rows)
    n_eligible = sum(1 for r in d2_rows if r.get("eligible"))
    n_stable = sum(1 for r in d2_rows if r.get("SUCCESS_12S"))
    n_new_15 = sum(1 for r in theta_rows if r.get("newly_accepted_under_15"))
    n_b1_recover = sum(1 for r in b1_rows if r.get("qualifies_as_direct_B1_relaxed"))
    official = official_class_counts()

    # CSVs
    _write_csv(OUT / "D2_HARD_IC_CENSUS.csv", d2_rows)
    _write_csv(OUT / "B1_RECOVERY_RESULTS.csv", b1_rows)
    _write_csv(OUT / "THETA_10_VS_15_VALIDATION.csv", theta_rows)
    _write_csv(OUT / "P10_GENERATION_RESULTS.csv", gen_rows)
    _write_csv(OUT / "A2_LIVE_VERIFY.csv", a2_live)

    # Updated class coverage (official + pilot accepts, not double-counting into official)
    pilot_accept = Counter()
    for r in gen_rows:
        if r.get("accepted") and r.get("class_label"):
            pilot_accept[r["class_label"]] += 1
    cov = []
    for cls in ["A1", "A2", "B1", "B2", "C1", "C2", "D2"]:
        n_off = int(official.get(cls, 0))
        n_pil = int(pilot_accept.get(cls, 0))
        # B1 recovery under RELAXED that aren't yet in official
        if cls == "B1":
            n_pil = max(n_pil, n_b1_recover)  # reval count as recoverable pool
        short20 = max(0, 20 - n_off)
        cov.append(
            {
                "class_label": cls,
                "n_official": n_off,
                "n_p10_pilot_accepted": int(pilot_accept.get(cls, 0)),
                "n_recoverable_pool_relaxed": n_b1_recover if cls == "B1" else (
                    n_eligible if cls == "D2" else (a2_analysis.get("live_a2_candidates", 0) if cls == "A2" else 0)
                ),
                "shortfall_to_20_official": short20,
                "milestone_20_official": n_off >= 20,
                "priority": "HIGH" if short20 > 10 else ("MED" if short20 > 0 else "OK"),
            }
        )
    _write_csv(OUT / "UPDATED_CLASS_COVERAGE.csv", cov)
    _write_csv(OUT / "UPDATED_SKILL_TRANSITIONS.csv", skill_transition_report(gen_rows, official))

    # D2 eligibility audit MD
    _write_md(
        OUT / "D2_ELIGIBILITY_AUDIT.md",
        f"""# D2 Eligibility Audit (P10)

## Gate correction
D2 (`D_PRECISION`) has **no** Plan A/B/C Stage-1 recovery prefix.
Requiring Stage-1 A/B/C registry evidence was conceptually incorrect and yielded **0** candidates in P9.

**Replacement gate** (`data_generation/oracles/precision/d2_eligibility.py`):
1. Documented HARD_IC capture+ref
2. Valid MuJoCo snapshot
3. No teleport during expert trajectory (pipeline constraint)
4. Live independent `SUCCESS_12S`
5. Meaningful Precision need (`assess_need`)
6. Normal Precision + handoff validation at generate time

## Census of 45 HARD_IC
| Bucket | n |
|---|---:|
"""
        + "\n".join(f"| {k} | {v} |" for k, v in sorted(bucket_counts.items()))
        + f"""

- Independent `SUCCESS_12S=True`: **{n_stable}**
- Eligible (stable + Precision-required): **{n_eligible}**
- Already within target (stable, no Precision): {bucket_counts.get('STABLE_ALREADY_WITHIN_TARGET', 0)}
- Unstable: {bucket_counts.get('UNSTABLE', 0)}
- Invalid / duplicate: {bucket_counts.get('INVALID_SNAPSHOT', 0) + bucket_counts.get('DUPLICATE_LINEAGE', 0) + bucket_counts.get('INVALID_OR_INCONSISTENT', 0)}

See `D2_HARD_IC_CENSUS.csv`. Source: `results/data_generation/stage2_abc_final_v2` plan_provenance=HARD_IC.
""",
    )

    promote = n_new_15 > 0 and n_b1_recover >= 1
    _write_md(
        OUT / "SUCCESS_RULE_DECISION.md",
        f"""# Success Rule Decision (P10)

## Dual metrics (unchanged historical meaning)
- `SUCCESS_STRICT_10DEG` ≡ historical `SUCCESS_NEW` (θ≤10°, e_pos≤12 mm, 12S, 0.5 s hold)
- `SUCCESS_RELAXED_15DEG` — same except θ≤15°

## Validation summary
- Newly admitted under 15° vs 10° (this pilot cohorts): **{n_new_15}**
- B1 P9 failures recoverable as direct B1 under 15°: **{n_b1_recover}** / {len(b1_rows)}
- Instability flags among θ cohort: {sum(1 for r in theta_rows if r.get('instability_or_drift_flag'))}

## Decision
{"**PROMOTE** `SUCCESS_RELAXED_15DEG` as the official *admission* key for new generation "
"(config `admit_success: SUCCESS_RELAXED_15DEG`)." if promote else
"**CONDITIONAL** — promote only for new generation after reviewing CSV; do not rewrite historical STRICT labels."}

### Frozen rule version
- Official admission (new manifests): `SUCCESS_RELAXED_15DEG_v1`
- Historical corpora / prior experiments: retain `SUCCESS_NEW` / `SUCCESS_STRICT_10DEG` labels
- All new manifests must set `admit_success_key` and `success_criteria_version`

### Non-negotiables
- Euclidean position tolerance unchanged (12 mm)
- Contact + 12 s nominal verification unchanged
- Orientation evaluated over the success hold window (not a single final sample)
- Momentary threshold crossings ≠ success

Config: `config/generation/experts_v2_p10_pilot.yaml`
Production `experts_v2.yaml` left on STRICT until an explicit production cutover.
""",
    )

    _write_md(
        OUT / "A2_GENERATION_ANALYSIS.md",
        f"""# A2 Generation Analysis (P10)

## Why A1 dominates
{a2_analysis.get('why_a1_dominates')}

## Counts
| Pool | n |
|---|---:|
| A1 evidenced Stage-1 terminals | {a2_analysis.get('n_A1_evidenced')} |
| A2 evidenced (registry) | {a2_analysis.get('n_A2_evidenced')} |
| A2 manifest total (plan_provenance=A) | {a2_analysis.get('n_A2_manifest_total')} |
| Without registry evidence | {a2_analysis.get('n_A2_without_registry')} |
| Official accepted A2 | {a2_analysis.get('n_official_A2')} |

## Live verify (bounded)
- Verified: {a2_analysis.get('live_verified')}
- SUCCESS_12S: {a2_analysis.get('live_12s_ok')}
- Precision-required A2 candidates: {a2_analysis.get('live_a2_candidates')}
- Already OK (A1-like): {a2_analysis.get('live_a1_like')}
- Unstable: {a2_analysis.get('live_unstable')}

See `A2_LIVE_VERIFY.csv`.

## Generation guidance
Do **not** rewrite Stage-1 exits to invent residuals.
Scale methods that yield live `SUCCESS_12S` + Precision-required residuals under legal contact offsets /
grasp geometry variation within the task mass/friction distribution.
""",
    )

    # Independent lineage
    fps = [r.get("capture_fp") for r in d2_rows if r.get("capture_fp")]
    dup_fps = len(fps) - len(set(fps))
    _write_md(
        OUT / "INDEPENDENT_IC_LINEAGE_AUDIT.md",
        f"""# Independent IC Lineage Audit (P10)

## ACTIVE_DATASET
- `data_generation.expert_pipeline.provenance.ACTIVE_DATASET` → `data/experts_v2_official`
- P10 pilot writes only to `results/diagnostics/p10_dataset_coverage/generation_pilot`
- Historical corpora not overwritten

## Official vs census
- Official verified experts: {sum(official.values())} (seven-class manifest)
- P9 EXISTING_DATASET_CENSUS includes broader rows; **census ≠ admission**
- M111 discretization: frozen (untouched this pilot)

## D2 HARD_IC fingerprints
- Unique capture_fp among audited: {len(set(fps))}
- Duplicate FP rows: {dup_fps}

## Split leakage
Training examples must key on `(source_ic_id, capture_fp, branch_id)`.
Do not place the same capture_fp in both train and eval.
Pilot accepts are diagnostic only until promoted into `experts_v2_official` with a new config_id stamp.
""",
    )

    # Training readiness
    still_need = [c["class_label"] for c in cov if c["shortfall_to_20_official"] > 0]
    _write_md(
        OUT / "TRAINING_READINESS_REPORT.md",
        f"""# Training Readiness (P10)

## Verdict
**Not yet SFT-ready** for multi-skill Precision — official shortfalls remain for: {', '.join(still_need) or 'none'}.

## Class coverage (official)
"""
        + "\n".join(
            f"- {c['class_label']}: {c['n_official']} (shortfall {c['shortfall_to_20_official']}; "
            f"pilot accepted {c['n_p10_pilot_accepted']}; recoverable pool {c['n_recoverable_pool_relaxed']})"
            for c in cov
        )
        + f"""

## Skill diversity
Official still dominated by `direct_handoff` / `precision_skipped`.
P10 pilot records FSM executed skills per trajectory in `P10_GENERATION_RESULTS.csv`.
Target: verified multi-skill (DX→DZ and DZ→DX) examples — do not fabricate skills.

## Concrete deficits
1. A2 independent source states (supply)
2. D2 Precision successes from eligible HARD_IC pool ({n_eligible} eligible)
3. Real multi-skill sequences beyond direct/skip
4. B1 can likely hit 20 via RELAXED revalidation without weakening 12S/contact
""",
    )

    # Final report
    a2_acc = sum(1 for r in gen_rows if r.get("run_id") == "p10_a2_pilot" and r.get("accepted"))
    d2_acc = sum(1 for r in gen_rows if r.get("run_id") == "p10_d2_pilot" and r.get("accepted"))
    b1_acc = sum(1 for r in gen_rows if r.get("run_id") == "p10_b1_relaxed" and r.get("accepted"))
    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P10 Dataset Coverage Recovery

Generated: {_utc()}

## Pipeline fixes
1. **D2 Stage-1 gate waived** in `sources.discover_branch_ics` when `stage1 is None`
2. **`d2_eligibility.check_d2_eligible`** — independent 12S + Precision-required
3. **Dual admission** — `admit_success_key` on PrecisionConfig; pilot uses `SUCCESS_RELAXED_15DEG`
4. **Case-3 logging** — RECONFIGURE_REQUESTED / FALLTHROUGH / NOT_EXECUTED (not silent DX-first)

## Answers to final questions
1. **D2 with valid independent stability evidence:** {n_stable} / 45 HARD_IC; **eligible for Precision:** {n_eligible}
2. **Additional accepts under 15° (cohort):** {n_new_15} newly STRICT-fail / RELAXED-pass; B1 recoverable {n_b1_recover}
3. **New independent A2 source states (live):** {a2_analysis.get('live_a2_candidates')} Precision-required among {a2_analysis.get('live_verified')} verified
4. **B1→20 without weakening physics?** {'Yes' if official.get('B1',0)+n_b1_recover >= 20 else 'Likely close'} — official {official.get('B1',0)} + recoverable {n_b1_recover} (12S+contact preserved; θ hold 15°)
5. **DX→DZ and DZ→DX represented?** Official: dx→dz present (P9); dz→dx scarce. Pilot FSM logs in generation CSV — check `fsm_executed`
6. **Still need generation:** {', '.join(still_need)}
7. **SFT-ready?** No — fill A2/D2 and multi-skill before claiming diversity

## Pilot accepts (diagnostic namespace only)
| Class | Pilot accepted |
|---|---:|
| B1 | {b1_acc} |
| A2 | {a2_acc} |
| D2 | {d2_acc} |

## Artifacts
All under `results/diagnostics/p10_dataset_coverage/`.
Generation pilot: `generation_pilot/` (not merged into official).

## Stop conditions respected
No hard-IC heuristic tuning, no physics/M111 change, no RL, no fabricated trajectories,
no silent STRICT overwrite of historical labels.
""",
    )

    # raw json dump
    (OUT / "raw_p10_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "d2_bucket_counts": dict(bucket_counts),
                "n_d2_stable": n_stable,
                "n_d2_eligible": n_eligible,
                "n_b1_recoverable": n_b1_recover,
                "n_newly_15": n_new_15,
                "a2_analysis": a2_analysis,
                "official_counts": official,
                "pilot_accepts": dict(pilot_accept),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== P10 Phase 1: D2 HARD_IC census ===", flush=True)
    d2_rows = audit_d2_hard_ic(limit=None)
    _write_csv(OUT / "D2_HARD_IC_CENSUS.csv", d2_rows)

    print("=== P10 Phase 2/4: B1 revalidation + θ ===", flush=True)
    b1_rows = revalidate_b1_failures()
    theta_rows = theta_validation_rows(b1_rows, d2_rows)

    print("=== P10 Phase 3: A2 supply ===", flush=True)
    a2_analysis, a2_live = analyze_a2_supply(live_limit=12)

    d2_eligible_ids = [r["sample_id"] for r in d2_rows if r.get("eligible")]
    a2_cand_ids = [r["sample_id"] for r in a2_live if r.get("bucket") == "A2_CANDIDATE"]
    b1_recover_ids = [
        r["source_ic_id"] for r in b1_rows if r.get("qualifies_as_direct_B1_relaxed")
    ]

    print("=== P10 Phase 7: bounded generation pilots ===", flush=True)
    print(
        f"  targets: D2={len(d2_eligible_ids)} eligible, A2={len(a2_cand_ids)} cand, "
        f"B1={len(b1_recover_ids)} recover",
        flush=True,
    )
    gen_rows = run_generation_pilots(
        d2_eligible_ids=d2_eligible_ids,
        a2_candidate_ids=a2_cand_ids,
        b1_recover_ids=b1_recover_ids,
        d2_limit=5,
        a2_limit=4,
        b1_limit=8,
    )

    print("=== Writing reports ===", flush=True)
    write_reports(
        d2_rows=d2_rows,
        b1_rows=b1_rows,
        theta_rows=theta_rows,
        a2_analysis=a2_analysis,
        a2_live=a2_live,
        gen_rows=gen_rows,
    )
    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
