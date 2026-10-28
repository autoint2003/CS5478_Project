"""Common Stage-1 exit contract for Plans A / B / C → Precision.

A2/B2/C2 classification is determined from the Stage-1 exit state *before*
any Precision action executes. SUCCESS_12S uses the authoritative
``stage1_eligibility`` implementation — this module does not redefine it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.stage1_eligibility import (
    capture_fingerprint,
    check_stage1_eligible,
    live_verify_stage1_12s,
)
from data_generation.oracles.precision.state import PrecisionConfig
from envs.simulation.capture import restore_capture, take_capture
from evaluation.success_criteria import contact_summary


@dataclass
class Stage1Exit:
    plan_id: str
    source_ic_id: str | None
    raw_ic: dict
    stage1_actions: list
    stage1_exit_capture: Any
    stage1_exit_ref: Any
    capture_fingerprint: str
    stage1_success_12s: bool
    stage1_verification_evidence: dict
    stage1_failure_reason: str | None
    stage1_action_count: int
    stage1_duration: float | None
    contact_summary_at_exit: dict
    precision_required: bool | None
    precision_entry_errors: dict | None
    classification: str  # STAGE1_FAILED | {A,B,C}1 | {A,B,C}2
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # capture/ref are not JSON-serializable; callers keep them on the object
        d.pop("stage1_exit_capture", None)
        d.pop("stage1_exit_ref", None)
        d.pop("raw_ic", None)
        d["stage1_actions"] = [
            np.asarray(a, np.float32).tolist() if hasattr(a, "__len__") else a for a in (self.stage1_actions or [])
        ]
        return d

    def as_precision_blob(self) -> dict[str, Any]:
        return {
            "capture": self.stage1_exit_capture,
            "ref": self.stage1_exit_ref,
            "sample_id": self.source_ic_id,
        }


def _plan_letter(plan_id: str) -> str:
    p = str(plan_id or "").strip().upper()
    if p.startswith("A"):
        return "A"
    if p.startswith("B"):
        return "B"
    if p.startswith("C"):
        return "C"
    return p[:1] or "?"


def classify_stage1_continuation(
    *,
    stage1_success_12s: bool,
    precision_required: bool | None,
    plan_id: str,
) -> str:
    letter = _plan_letter(plan_id)
    if not stage1_success_12s:
        return "STAGE1_FAILED"
    if precision_required is False:
        return f"{letter}1"
    if precision_required is True:
        return f"{letter}2"
    return f"{letter}_STAGE1_OK_UNCLASSIFIED"


def finalize_stage1_exit(
    env,
    *,
    plan_id: str,
    source_ic_id: str | None,
    raw_ic: dict,
    stage1_actions: list,
    stage1_exit_capture: Any,
    stage1_exit_ref: Any,
    stage1_duration: float | None = None,
    precision_config: PrecisionConfig | None = None,
    allow_live_verify: bool = True,
    record_live: bool = False,
    extra: dict | None = None,
) -> Stage1Exit:
    """Bind exit capture to SUCCESS_12S evidence and direct-vs-Precision classification.

    Does not execute Precision. Restores ``stage1_exit_capture`` into ``env`` on return
    so subsequent Precision starts from the verified snapshot.
    """
    blob = {"capture": stage1_exit_capture, "ref": stage1_exit_ref, "sample_id": source_ic_id}
    fp = capture_fingerprint(stage1_exit_capture)
    elig = check_stage1_eligible(
        env,
        blob,
        allow_live_verify=allow_live_verify,
        record_live=record_live,
    )
    # Ensure env sits on the verified exit (eligibility may have mutated sim during live verify)
    restore_capture(env, stage1_exit_capture)
    cs = contact_summary(env.sim)
    errors = None
    precision_required = None
    if elig.eligible:
        errors = measure_errors(env.sim)
        cfg = precision_config or PrecisionConfig()
        need = assess_need(errors, cfg)
        precision_required = not bool(need.get("ready_for_direct_handoff_assessment"))
        restore_capture(env, stage1_exit_capture)
    classification = classify_stage1_continuation(
        stage1_success_12s=bool(elig.eligible),
        precision_required=precision_required,
        plan_id=plan_id,
    )
    acts = list(stage1_actions or [])
    return Stage1Exit(
        plan_id=str(plan_id),
        source_ic_id=source_ic_id,
        raw_ic=dict(raw_ic or {}),
        stage1_actions=acts,
        stage1_exit_capture=stage1_exit_capture,
        stage1_exit_ref=stage1_exit_ref,
        capture_fingerprint=fp,
        stage1_success_12s=bool(elig.eligible),
        stage1_verification_evidence=elig.to_dict(),
        stage1_failure_reason=None if elig.eligible else (elig.reason or "STAGE1_INELIGIBLE"),
        stage1_action_count=len(acts),
        stage1_duration=stage1_duration,
        contact_summary_at_exit={
            "nL": int(cs.get("nL") or 0),
            "nR": int(cs.get("nR") or 0),
            "bilateral": bool(cs.get("bilateral")),
            "fnL": float(cs.get("fnL") or 0.0),
            "fnR": float(cs.get("fnR") or 0.0),
        },
        precision_required=precision_required,
        precision_entry_errors=errors,
        classification=classification,
        extra=dict(extra or {}),
    )


def stage1_exit_from_existing_ic(
    env,
    blob: dict,
    *,
    plan_id: str,
    source_ic_id: str | None = None,
    stage1_actions: list | None = None,
    precision_config: PrecisionConfig | None = None,
    allow_live_verify: bool = True,
) -> Stage1Exit:
    """Wrap a pre-captured Stage-1 IC (expert-pipeline style) in the common contract."""
    sid = source_ic_id or blob.get("sample_id") or blob.get("ic_id")
    return finalize_stage1_exit(
        env,
        plan_id=plan_id,
        source_ic_id=sid,
        raw_ic=blob,
        stage1_actions=list(stage1_actions or []),
        stage1_exit_capture=blob["capture"],
        stage1_exit_ref=blob["ref"],
        precision_config=precision_config,
        allow_live_verify=allow_live_verify,
        record_live=False,
    )


def capture_current_as_stage1_exit(
    env,
    *,
    plan_id: str,
    ref,
    source_ic_id: str | None = None,
    raw_ic: dict | None = None,
    stage1_actions: list | None = None,
    stage1_duration: float | None = None,
    precision_config: PrecisionConfig | None = None,
    extra: dict | None = None,
) -> Stage1Exit:
    """Take capture at current sim state and finalize (for live Stage-1 runners)."""
    cap = take_capture(env)
    return finalize_stage1_exit(
        env,
        plan_id=plan_id,
        source_ic_id=source_ic_id,
        raw_ic=dict(raw_ic or {}),
        stage1_actions=list(stage1_actions or []),
        stage1_exit_capture=cap,
        stage1_exit_ref=ref,
        stage1_duration=stage1_duration,
        precision_config=precision_config,
        allow_live_verify=True,
        record_live=False,
        extra=extra,
    )


def verify_exit_matches_precision_restore(env, exit_obj: Stage1Exit) -> dict[str, Any]:
    """Regression helper: live 12S verify then confirm env restored to same fingerprint."""
    blob = exit_obj.as_precision_blob()
    fp0 = exit_obj.capture_fingerprint
    res = live_verify_stage1_12s(env, blob)
    fp1 = capture_fingerprint(take_capture(env))
    restore_capture(env, exit_obj.stage1_exit_capture)
    fp2 = capture_fingerprint(take_capture(env))
    return {
        "exit_fp": fp0,
        "after_live_verify_restore_fp": res.capture_fp,
        "env_fp_after_live_verify_api": fp1,
        "env_fp_after_explicit_restore": fp2,
        "SUCCESS_12S": res.SUCCESS_12S,
        "snapshot_identity_ok": fp0 == fp2 == res.capture_fp,
    }
