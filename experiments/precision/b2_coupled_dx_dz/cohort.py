"""Rebuild and cache the seven verified B2 Stage-1 exit snapshots."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "results/data_generation/precision_required_pilot_v2"
HARD = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
B1_OUT = ROOT / "results/data_generation/stage2_abc_mining/v2/outcomes"
CACHE = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"

# Historical Oracle outcomes from the B Precision-required video cohort.
EXPECTED_STATUS = {
    "hard_910026": "PRECISION_EXECUTED_SUCCESS",
    "hard_910132": "PRECISION_PRIMITIVE_FAILED",
    "hard_910024": "PRECISION_PRIMITIVE_FAILED",
    "hard_910263": "PRECISION_PRIMITIVE_FAILED",
    "hard_910277": "PRECISION_PRIMITIVE_FAILED",
    "hard_910084": "PRECISION_EXECUTED_HANDOFF_FAIL",
    "hard_910305": "PRECISION_EXECUTED_HANDOFF_FAIL",
}

HARD_IDS = list(EXPECTED_STATUS.keys())


def load_funnel_rows() -> list[dict[str, str]]:
    fun = list(csv.DictReader((PILOT / "FULL_FUNNEL.csv").open(encoding="utf-8")))
    rows = [
        r
        for r in fun
        if r.get("branch") == "B"
        and str(r.get("stage1_12s")) in ("1", "True", "true")
        and str(r.get("precision_required")) in ("1", "True", "true")
        and any(h in str(r.get("raw_id") or "") for h in HARD_IDS)
    ]
    by: dict[str, dict] = {}
    for r in rows:
        by[r["raw_id"]] = r
    return list(by.values())


def hard_id_from_raw(raw_id: str) -> str:
    for part in raw_id.split("__"):
        if part.startswith("hard_"):
            return part
    raise KeyError(raw_id)


def b1_tokens(hard_id: str) -> list[int]:
    hits = sorted(B1_OUT.glob(f"B__{hard_id}__post_b1_v2__*.json"))
    if not hits:
        raise FileNotFoundError(hard_id)
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    toks = list(d.get("b1_tokens") or [])
    if not toks:
        raise ValueError(f"no b1_tokens for {hard_id}")
    return toks


def rebuild_exits(env, *, force: bool = False) -> list[dict[str, Any]]:
    """Replay Plan-B known tokens → Stage-1 exit; cache capture+ref pickles."""
    from data_generation.oracles.plan_b.branch_compare import load_vocab
    from data_generation.oracles.plan_b.stage1_runner import replay_plan_b_tokens_to_exit
    from data_generation.oracles.precision.stage1_eligibility import (
        capture_fingerprint,
        check_stage1_eligible,
    )
    from data_generation.oracles.precision.state import PrecisionConfig

    CACHE.mkdir(parents=True, exist_ok=True)
    table, book48 = load_vocab()
    pcfg = PrecisionConfig(use_verified_lateral_fallback=False)
    rows = load_funnel_rows()
    out: list[dict[str, Any]] = []

    for row in rows:
        raw_id = row["raw_id"]
        hid = hard_id_from_raw(raw_id)
        dest = CACHE / f"{hid}.pkl"
        meta_path = CACHE / f"{hid}__meta.json"
        if dest.is_file() and meta_path.is_file() and not force:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            blob = pickle.loads(dest.read_bytes())
            out.append({**meta, "blob": blob, "exit_path": str(dest.relative_to(ROOT)).replace("\\", "/")})
            continue

        snap = pickle.loads((HARD / f"{hid}.pkl").read_bytes())
        toks = b1_tokens(hid)
        pref = min(3, len(toks))
        exit_obj = replay_plan_b_tokens_to_exit(
            env,
            {"capture": snap["capture"], "ref": snap["ref"], "sample_id": raw_id},
            toks,
            plan_id="B",
            source_ic_id=raw_id,
            require_active_recatch=True,
            prefix_len=pref,
            precision_config=pcfg,
            table=table,
            book48=book48,
        )
        blob = {"capture": exit_obj.stage1_exit_capture, "ref": exit_obj.stage1_exit_ref}
        fp = capture_fingerprint(blob["capture"])
        elig = check_stage1_eligible(
            env,
            {"capture": blob["capture"], "ref": blob["ref"], "sample_id": raw_id},
            allow_live_verify=True,
            record_live=False,
        )
        meta = {
            "raw_id": raw_id,
            "hard_id": hid,
            "capture_fp": fp,
            "b1_tokens": toks,
            "prefix_len": pref,
            "stage1_success_12s": bool(exit_obj.stage1_success_12s),
            "precision_required": bool(exit_obj.precision_required),
            "classification": exit_obj.classification,
            "expected_oracle_status": EXPECTED_STATUS.get(hid),
            "pilot_funnel_status": row.get("precision_status"),
            "pilot_skills": row.get("skills"),
            "stage1_eligibility": elig.to_dict(),
            "entry_errors": exit_obj.precision_entry_errors,
            "exit_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        }
        dest.write_bytes(pickle.dumps(blob))
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        out.append({**meta, "blob": blob})
    return out
