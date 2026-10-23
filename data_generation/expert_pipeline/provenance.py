"""Dataset version isolation for regenerated experts.

``data/unified_experts/`` remains the historical manifest. It is not a training
source for the new Precision Oracle. New experts are written only under
``data/experts_v2/`` and only when physical success is true.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "experts_v2"
ORACLE_IMPLEMENTATION = "precision_state_machine_v23"
SUCCESS_CRITERIA_VERSION = "SUCCESS_12S_and_SUCCESS_NEW_frozen"
ACTIVE_DATASET = ROOT / "data" / "experts_v2"
LEGACY_MANIFEST = ROOT / "data" / "unified_experts" / "manifest.jsonl"
LEGACY_DIRECT_HANDOFF_BRANCHES = {
    "B_STABLE_DIRECT_HANDOFF",
    "C_RECATCH_DIRECT_HANDOFF",
}


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def stamp_record(partial: dict[str, Any], *, ic_id: str | None = None, config_id: str | None = None) -> dict[str, Any]:
    """Provenance block stored beside a successful expert. Does not copy actions."""
    if not partial.get("physical_success"):
        raise ValueError("refusing to stamp an unsuccessful trajectory as an expert")
    continuation = partial.get("continuation")
    return {
        "dataset_version": DATASET_VERSION,
        "oracle_implementation": ORACLE_IMPLEMENTATION if continuation == "precision" else None,
        "git_commit": git_commit(),
        "source_ic_id": ic_id or partial.get("source_ic_id"),
        "branch_id": partial.get("branch_id"),
        "continuation": continuation,
        "stage1": partial.get("stage1"),
        "physical_success": True,
        "success_criteria_version": SUCCESS_CRITERIA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": config_id,
        "output_namespace": "data/experts_v2",
    }


def _uses_precision_oracle(row: dict[str, Any]) -> bool:
    continuation = row.get("continuation")
    branch = str(row.get("branch_id") or "")
    if continuation == "precision" or branch.endswith("_PRECISION") or branch == "D_PRECISION":
        return True
    oracle = row.get("oracle_implementation")
    if oracle and "precision" in str(oracle):
        return True
    return False


def row_allowed_for_training(row: dict[str, Any]) -> bool:
    """New-training admission.

    Legacy Precision rows are rejected. Legacy direct-HANDOFF rows are admitted
    only when they carry an explicit non-precision branch and physical success.
    Failures are never admitted.
    """
    if not row.get("physical_success") and not (
        row.get("SUCCESS_NEW") is True or row.get("SUCCESS_12S") is True
    ):
        return False
    version = row.get("dataset_version")
    if version == DATASET_VERSION:
        if not row.get("physical_success"):
            return False
        if _uses_precision_oracle(row):
            return row.get("oracle_implementation") == ORACLE_IMPLEMENTATION and bool(row.get("git_commit"))
        return row.get("branch_id") in LEGACY_DIRECT_HANDOFF_BRANCHES or row.get("continuation") == "direct_handoff"
    # Historical unified_experts rows have no experts_v2 stamp.
    if _uses_precision_oracle(row):
        return False
    branch = row.get("branch_id")
    if branch in LEGACY_DIRECT_HANDOFF_BRANCHES and row.get("physical_success") is True:
        return True
    return False
