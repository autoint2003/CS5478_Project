"""Dispatch a selected branch onto the current Oracle or direct HANDOFF.

Stage-1 search stays in ``data_generation.oracles.plan_{a,b,c}``. This module
only applies the continuation once a capture and reference are available.
Source ICs for ``experts_v2`` are post-Stage-1 stable captures.
"""

from __future__ import annotations

from typing import Any

from data_generation.expert_pipeline.branches import STAGE1_MODULES, select_branch
from data_generation.expert_pipeline.provenance import stamp_record
from data_generation.oracles.precision.handoff import probe_direct_handoff
from data_generation.oracles.precision.state import PrecisionConfig, PrecisionResult
from data_generation.oracles.precision.state_machine import run_precision

_PRECISION_SUCCESS = {"PRECISION_EXECUTED_SUCCESS", "PRECISION_SKIPPED_DIRECT_SUCCESS"}


def _admit_key_from_cfg(precision_config: PrecisionConfig | dict | None) -> str:
    if precision_config is None:
        return "SUCCESS_NEW"
    if isinstance(precision_config, dict):
        return str(precision_config.get("admit_success_key") or "SUCCESS_NEW")
    return str(getattr(precision_config, "admit_success_key", None) or "SUCCESS_NEW")


def _result_payload(result: PrecisionResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload["status"] = result.status
    payload["phases"] = list(result.phases or [])
    payload["precision_skipped"] = bool(result.precision_skipped)
    payload["precision_executed"] = bool(result.precision_executed)
    payload["failures"] = list(result.failures or [])
    payload["decision"] = result.decision
    payload["errors_entry"] = result.errors_entry
    payload["errors_exit"] = result.errors_exit
    payload["primitive_logs"] = result.primitive_logs
    payload["handoff"] = result.handoff
    if result.handoff:
        payload["SUCCESS_NEW"] = bool(result.handoff.get("SUCCESS_NEW"))
        payload["SUCCESS_12S"] = bool(result.handoff.get("SUCCESS_12S"))
        payload["SUCCESS_STRICT_10DEG"] = bool(
            result.handoff.get("SUCCESS_STRICT_10DEG") or result.handoff.get("SUCCESS_NEW")
        )
        payload["SUCCESS_RELAXED_15DEG"] = bool(result.handoff.get("SUCCESS_RELAXED_15DEG"))
    return payload


def run_continuation(
    env,
    state: dict,
    branch_id: str,
    precision_config: PrecisionConfig | dict | None = None,
) -> dict[str, Any]:
    """Run Precision or direct HANDOFF for ``branch_id``.

    ``state`` must contain ``capture`` and ``ref``. A failed continuation is
    returned with ``expert`` set to None and is never stamped as successful.
    Admission follows ``admit_success_key`` on PrecisionConfig (default
    SUCCESS_NEW ≡ STRICT 10°). Dual STRICT/RELAXED flags are always recorded.
    """
    branch = select_branch(branch_id)
    if "capture" not in state or "ref" not in state:
        raise KeyError("continuation state requires capture and ref")
    admit_key = _admit_key_from_cfg(precision_config)
    actions: list = []
    if branch.continuation == "precision":
        result = run_precision(env, state, branch.branch_id, precision_config)
        payload = _result_payload(result)
        hand = result.handoff or {}
        if admit_key == "SUCCESS_NEW":
            ok = result.status in _PRECISION_SUCCESS
        else:
            ok = bool(hand.get(admit_key) and hand.get("SUCCESS_12S"))
            # Also accept classical Precision success under STRICT
            ok = ok or result.status in _PRECISION_SUCCESS
        actions = list(result.actions or [])
    elif branch.continuation == "direct_handoff":
        payload = probe_direct_handoff(env, state["capture"], state["ref"])
        if admit_key == "SUCCESS_NEW":
            ok = bool(payload.get("SUCCESS_NEW"))
        else:
            ok = bool(payload.get(admit_key) and payload.get("SUCCESS_12S"))
            ok = ok or bool(payload.get("SUCCESS_NEW"))
        actions = []
    else:
        raise RuntimeError(f"unsupported continuation {branch.continuation}")
    record = {
        "branch_id": branch.branch_id,
        "stage1": branch.stage1,
        "stage1_module": STAGE1_MODULES.get(branch.stage1 or "", None),
        "continuation": branch.continuation,
        "physical_success": bool(ok),
        "n_actions": len(actions),
        "actions": actions,
        "result": payload,
    }
    if not ok:
        record["expert"] = None
        record["rejected_reason"] = "physical_success_false"
        return record
    record["expert"] = stamp_record(record)
    return record
