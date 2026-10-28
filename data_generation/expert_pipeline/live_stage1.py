"""Live Stage-1 → Precision generation mode.

Legacy ``generate.run_one`` consumes pre-captured Stage-1 ICs.
This module executes genuine Plan A/B/(optional C) Stage-1 trajectories from
raw scenarios, finalizes via ``stage1_exit``, then runs Precision only on
A2/B2/C2 candidates.

Does not change Stage-1 or Precision control laws.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.plan_a.stage1_runner import replay_plan_a_actions_to_exit
from data_generation.oracles.plan_b.stage1_runner import replay_plan_b_tokens_to_exit
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.state_machine import run_precision
from data_generation.oracles.stage1_exit import Stage1Exit, finalize_stage1_exit
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal


@dataclass
class LiveGenerationResult:
    branch_target: str  # A2 / B2 / C2 / D2 / A1 / B1 / C1 / REJECT
    source_type: str  # stage1_rollout | synthetic_ic
    plan_id: str
    raw_scenario_id: str
    accepted: bool
    reject_reason: str | None = None
    classification: str | None = None
    stage1_success_12s: bool | None = None
    precision_required: bool | None = None
    SUCCESS_NEW: bool | None = None
    SUCCESS_12S_final: bool | None = None
    precision_status: str | None = None
    precision_phases: list[str] = field(default_factory=list)
    precision_skills: list[str] = field(default_factory=list)
    n_stage1_actions: int = 0
    n_precision_actions: int = 0
    wall_s: float = 0.0
    capture_fp: str | None = None
    entry_errors: dict | None = None
    exit_errors: dict | None = None
    stage1_actions: list = field(default_factory=list)
    precision_actions: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    synthetic_initial_condition: bool = False

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("stage1_actions", None)
        d.pop("precision_actions", None)
        d["precision_phases"] = "|".join(self.precision_phases or [])
        d["precision_skills"] = "|".join(self.precision_skills or [])
        return d


def extract_precision_skills(phases: list[str] | None) -> list[str]:
    """Map state_machine phase tags to coarse Precision skills."""
    skills: list[str] = []
    for p in phases or []:
        pl = str(p).lower()
        if "dtheta" in pl or "correct_dtheta" in pl:
            skills.append("dtheta")
        elif "dx_support_transfer" in pl or "support_transfer" in pl:
            skills.append("support_transfer")
        elif "correct_dx" in pl or "dx_horizontal" in pl or "dx_verified" in pl:
            skills.append("dx")
        elif "dz_inward" in pl or "correct_dz_inward" in pl:
            skills.append("dz_inward")
        elif "dz_outward" in pl or "correct_dz_outward" in pl:
            skills.append("dz_outward")
        elif "correct_dz" in pl or "dz_" in pl:
            skills.append("dz")
    # preserve order, unique consecutive
    out: list[str] = []
    for s in skills:
        if not out or out[-1] != s:
            out.append(s)
    return out


def _accept_precision(env, exit_obj: Stage1Exit, *, plan_id: str, cfg: PrecisionConfig) -> tuple[Any, LiveGenerationResult]:
    """Run Precision only when exit is *2 (precision-required)."""
    t0 = time.perf_counter()
    letter = str(plan_id).upper()[:1]
    if not exit_obj.stage1_success_12s:
        return None, LiveGenerationResult(
            branch_target="REJECT",
            source_type="stage1_rollout",
            plan_id=letter,
            raw_scenario_id=str(exit_obj.source_ic_id or ""),
            accepted=False,
            reject_reason=f"STAGE1_FAILED:{exit_obj.stage1_failure_reason}",
            classification=exit_obj.classification,
            stage1_success_12s=False,
            capture_fp=exit_obj.capture_fingerprint,
            n_stage1_actions=exit_obj.stage1_action_count,
            wall_s=time.perf_counter() - t0,
            stage1_actions=list(exit_obj.stage1_actions or []),
            provenance={"stage1_evidence": exit_obj.stage1_verification_evidence},
        )
    if not exit_obj.precision_required:
        return None, LiveGenerationResult(
            branch_target=f"{letter}1",
            source_type="stage1_rollout",
            plan_id=letter,
            raw_scenario_id=str(exit_obj.source_ic_id or ""),
            accepted=False,
            reject_reason="DIRECT_ELIGIBLE_NOT_PRECISION_REQUIRED",
            classification=exit_obj.classification,
            stage1_success_12s=True,
            precision_required=False,
            capture_fp=exit_obj.capture_fingerprint,
            entry_errors=exit_obj.precision_entry_errors,
            n_stage1_actions=exit_obj.stage1_action_count,
            wall_s=time.perf_counter() - t0,
            stage1_actions=list(exit_obj.stage1_actions or []),
            provenance={"stage1_evidence": exit_obj.stage1_verification_evidence},
        )

    restore_capture(env, exit_obj.stage1_exit_capture)
    fp_before = exit_obj.capture_fingerprint
    res = run_precision(env, exit_obj.as_precision_blob(), letter, cfg)
    skills = extract_precision_skills(list(res.phases or []))
    handoff = res.handoff or {}
    success_new = bool((handoff or {}).get("SUCCESS_NEW")) or res.status == "PRECISION_EXECUTED_SUCCESS"
    accepted = bool(success_new) and res.status not in ("STAGE1_INELIGIBLE", "STABLE_GATE_FAILED")
    row = LiveGenerationResult(
        branch_target=f"{letter}2",
        source_type="stage1_rollout",
        plan_id=letter,
        raw_scenario_id=str(exit_obj.source_ic_id or ""),
        accepted=accepted,
        reject_reason=None if accepted else f"PRECISION_FAIL:{res.status}",
        classification=exit_obj.classification,
        stage1_success_12s=True,
        precision_required=True,
        SUCCESS_NEW=success_new,
        SUCCESS_12S_final=bool(handoff.get("SUCCESS_12S")) if handoff else None,
        precision_status=res.status,
        precision_phases=list(res.phases or []),
        precision_skills=skills,
        n_stage1_actions=exit_obj.stage1_action_count,
        n_precision_actions=len(res.actions or []),
        wall_s=time.perf_counter() - t0,
        capture_fp=fp_before,
        entry_errors=exit_obj.precision_entry_errors,
        exit_errors=res.errors_exit if hasattr(res, "errors_exit") else None,
        stage1_actions=list(exit_obj.stage1_actions or []),
        precision_actions=list(res.actions or []),
        provenance={
            "stage1_evidence": exit_obj.stage1_verification_evidence,
            "precision_failures": list(res.failures or []),
            "snapshot_identity_ok": capture_fingerprint(exit_obj.stage1_exit_capture) == fp_before,
        },
    )
    return res, row


def generate_a2_from_raw(
    env,
    raw_blob: dict,
    stage1_actions: list | np.ndarray,
    *,
    raw_scenario_id: str,
    precision_config: PrecisionConfig | None = None,
    provenance: dict | None = None,
) -> LiveGenerationResult:
    """Raw capture → Plan A Stage-1 actions → classify → Precision if A2."""
    cfg = precision_config or PrecisionConfig(use_verified_lateral_fallback=False)
    exit_obj = replay_plan_a_actions_to_exit(
        env,
        {"capture": raw_blob["capture"], "ref": raw_blob["ref"], "sample_id": raw_scenario_id},
        stage1_actions,
        plan_id="A",
        source_ic_id=raw_scenario_id,
        precision_config=cfg,
    )
    _, row = _accept_precision(env, exit_obj, plan_id="A", cfg=cfg)
    row.provenance = {**(row.provenance or {}), **(provenance or {}), "runner": "live_stage1.generate_a2_from_raw"}
    return row


def generate_b2_from_raw(
    env,
    raw_blob: dict,
    tokens: list[int],
    *,
    raw_scenario_id: str,
    prefix_len: int | None = None,
    precision_config: PrecisionConfig | None = None,
    provenance: dict | None = None,
    table=None,
    book48=None,
) -> LiveGenerationResult:
    """Raw capture → Plan B Recovery tokens → classify → Precision if B2."""
    cfg = precision_config or PrecisionConfig(use_verified_lateral_fallback=False)
    exit_obj = replay_plan_b_tokens_to_exit(
        env,
        {"capture": raw_blob["capture"], "ref": raw_blob["ref"], "sample_id": raw_scenario_id},
        tokens,
        plan_id="B",
        source_ic_id=raw_scenario_id,
        require_active_recatch=True,
        prefix_len=prefix_len,
        precision_config=cfg,
        table=table,
        book48=book48,
    )
    _, row = _accept_precision(env, exit_obj, plan_id="B", cfg=cfg)
    row.provenance = {
        **(row.provenance or {}),
        **(provenance or {}),
        "runner": "live_stage1.generate_b2_from_raw",
        "tokens": list(tokens),
        "prefix_len": prefix_len,
    }
    return row


def generate_c2_from_stage1_exit_sim(
    env,
    *,
    stage1_actions: list,
    exit_capture,
    ref,
    raw_scenario_id: str,
    precision_config: PrecisionConfig | None = None,
    provenance: dict | None = None,
    pose_correct_called: bool = False,
) -> LiveGenerationResult:
    """Finalize a Plan C Stage-1-only exit (must not have run pose_correct)."""
    if pose_correct_called:
        return LiveGenerationResult(
            branch_target="REJECT",
            source_type="stage1_rollout",
            plan_id="C",
            raw_scenario_id=raw_scenario_id,
            accepted=False,
            reject_reason="POSE_CORRECT_BEFORE_STAGE1_EXIT_FORBIDDEN",
            classification="STAGE1_FAILED",
        )
    cfg = precision_config or PrecisionConfig(use_verified_lateral_fallback=False)
    exit_obj = finalize_stage1_exit(
        env,
        plan_id="C",
        source_ic_id=raw_scenario_id,
        raw_ic={"sample_id": raw_scenario_id},
        stage1_actions=stage1_actions,
        stage1_exit_capture=exit_capture,
        stage1_exit_ref=ref,
        precision_config=cfg,
        allow_live_verify=True,
        extra={"pose_correct_called": False, **(provenance or {})},
    )
    _, row = _accept_precision(env, exit_obj, plan_id="C", cfg=cfg)
    row.provenance = {**(row.provenance or {}), **(provenance or {}), "runner": "live_stage1.generate_c2_from_stage1_exit_sim"}
    return row


def generate_d2_from_synthetic_ic(
    env,
    ic_blob: dict,
    *,
    raw_scenario_id: str,
    construction_meta: dict,
    precision_config: PrecisionConfig | None = None,
) -> LiveGenerationResult:
    """Synthetic grasp IC → SUCCESS_12S → Precision-required → Oracle (no Stage-1 A/B/C)."""
    cfg = precision_config or PrecisionConfig(use_verified_lateral_fallback=False)
    t0 = time.perf_counter()
    exit_obj = finalize_stage1_exit(
        env,
        plan_id="D",
        source_ic_id=raw_scenario_id,
        raw_ic=ic_blob,
        stage1_actions=[],  # D2 has no A/B/C Stage-1 actions
        stage1_exit_capture=ic_blob["capture"],
        stage1_exit_ref=ic_blob["ref"],
        precision_config=cfg,
        allow_live_verify=True,
        extra={"synthetic_initial_condition": True, "construction": construction_meta},
    )
    if not exit_obj.stage1_success_12s:
        return LiveGenerationResult(
            branch_target="REJECT",
            source_type="synthetic_ic",
            plan_id="D",
            raw_scenario_id=raw_scenario_id,
            accepted=False,
            reject_reason=f"D2_STAGE1_12S_FAIL:{exit_obj.stage1_failure_reason}",
            classification="STAGE1_FAILED",
            stage1_success_12s=False,
            capture_fp=exit_obj.capture_fingerprint,
            wall_s=time.perf_counter() - t0,
            synthetic_initial_condition=True,
            provenance={"construction": construction_meta},
        )
    if not exit_obj.precision_required:
        return LiveGenerationResult(
            branch_target="D1_NOT_USED",
            source_type="synthetic_ic",
            plan_id="D",
            raw_scenario_id=raw_scenario_id,
            accepted=False,
            reject_reason="D2_NOT_PRECISION_REQUIRED",
            classification="D1",
            stage1_success_12s=True,
            precision_required=False,
            capture_fp=exit_obj.capture_fingerprint,
            entry_errors=exit_obj.precision_entry_errors,
            wall_s=time.perf_counter() - t0,
            synthetic_initial_condition=True,
            provenance={"construction": construction_meta},
        )
    restore_capture(env, exit_obj.stage1_exit_capture)
    res = run_precision(env, exit_obj.as_precision_blob(), "D", cfg)
    skills = extract_precision_skills(list(res.phases or []))
    handoff = res.handoff or {}
    success_new = bool(handoff.get("SUCCESS_NEW")) or res.status == "PRECISION_EXECUTED_SUCCESS"
    accepted = bool(success_new)
    return LiveGenerationResult(
        branch_target="D2",
        source_type="synthetic_ic",
        plan_id="D",
        raw_scenario_id=raw_scenario_id,
        accepted=accepted,
        reject_reason=None if accepted else f"PRECISION_FAIL:{res.status}",
        classification="D2",
        stage1_success_12s=True,
        precision_required=True,
        SUCCESS_NEW=success_new,
        SUCCESS_12S_final=bool(handoff.get("SUCCESS_12S")) if handoff else None,
        precision_status=res.status,
        precision_phases=list(res.phases or []),
        precision_skills=skills,
        n_stage1_actions=0,
        n_precision_actions=len(res.actions or []),
        wall_s=time.perf_counter() - t0,
        capture_fp=exit_obj.capture_fingerprint,
        entry_errors=exit_obj.precision_entry_errors,
        precision_actions=list(res.actions or []),
        synthetic_initial_condition=True,
        provenance={
            "construction": construction_meta,
            "stage1_evidence": exit_obj.stage1_verification_evidence,
            "precision_failures": list(res.failures or []),
        },
    )


def settle_grip(env, n: int = 10) -> list[np.ndarray]:
    acts = []
    for _ in range(int(n)):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(env.sim, a)
        acts.append(a.copy())
    return acts
