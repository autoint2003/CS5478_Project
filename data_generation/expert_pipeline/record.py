"""Serialize accepted experts and diagnostic failures under experts_v2."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.expert_pipeline.provenance import (
    ACTIVE_DATASET,
    DATASET_VERSION,
    ORACLE_IMPLEMENTATION,
    SUCCESS_CRITERIA_VERSION,
    git_commit,
)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj).replace("\\", "/")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def attempt_id(branch_id: str, source_ic_id: str) -> str:
    return f"{branch_id}__{source_ic_id}"


def skill_sequence_label(phases: list[str] | None, *, precision_skipped: bool, continuation: str | None = None) -> str:
    """Collapse executed phases into a skill-path label for reporting."""
    phases = [str(p) for p in (phases or [])]
    if continuation == "direct_handoff" or phases == ["direct_handoff"]:
        return "direct_handoff"
    if precision_skipped:
        return "precision_skipped_direct_handoff"
    has_dth = any("dtheta" in p for p in phases)
    has_dx = any(("dx" in p) and ("dz" not in p) for p in phases)
    has_dz = any("dz" in p for p in phases)
    parts = []
    if has_dth:
        parts.append("dtheta")
    if has_dx:
        parts.append("dx")
    if has_dz:
        parts.append("dz")
    if not parts:
        if any("handoff" in p or "final_down" in p for p in phases):
            return "precision_skipped_direct_handoff"
        return "other:" + ("|".join(phases[:8]) if phases else "empty")
    return "→".join(parts)


def classify_failure(record: dict[str, Any]) -> str:
    if record.get("exception"):
        return "simulation_or_serialization_error"
    result = record.get("result") or {}
    failures = result.get("failures") or record.get("failures") or []
    status = str(result.get("status") or "")
    if status == "STABLE_GATE_FAILED":
        return "invalid_entry_conditions"
    text = " ".join(str(f) for f in failures).lower()
    if "final_down_gate" in text:
        return "final_down_gate_rejection"
    if "support" in text and ("lost" in text or "fail" in text):
        return "support_loss_or_s1_failure"
    if "s1" in text:
        return "s1_establishment_failure"
    if "rotation" in text or "tilt" in text or "to_up" in text:
        return "protected_rotation_failure"
    if "slip" in text or "slide" in text:
        return "sliding_failure"
    if "brake" in text or "regrip" in text:
        return "brake_regrip_failure"
    if "restore" in text and "down" in text:
        return "down_restoration_failure"
    if "handoff" in text or status.endswith("HANDOFF_FAIL"):
        if result.get("SUCCESS_12S") and not result.get("SUCCESS_NEW"):
            return "success_12s_without_success_new"
        return "handoff_physical_failure"
    if record.get("rejected_reason") == "physical_success_false":
        if result.get("SUCCESS_12S") and not (
            result.get("SUCCESS_NEW") or (result.get("handoff") or {}).get("SUCCESS_NEW")
        ):
            return "success_12s_without_success_new"
        return "physical_success_false"
    return "unclassified_failure"


class DatasetWriter:
    """Crash-safe writers for experts_v2 branch trees."""

    def __init__(self, root: Path | None = None, *, config_id: str | None = None):
        self.root = Path(root) if root else ACTIVE_DATASET
        self.config_id = config_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "diagnostics").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        self.index_path = self.root / "index.json"
        self.progress_path = self.root / "diagnostics" / "progress.json"
        self._completed = self._load_completed()

    def _load_completed(self) -> set[str]:
        done: set[str] = set()
        if self.progress_path.is_file():
            try:
                blob = json.loads(self.progress_path.read_text(encoding="utf-8"))
                done.update(str(x) for x in blob.get("completed_attempts") or [])
            except json.JSONDecodeError:
                pass
        if self.manifest_path.is_file():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                aid = row.get("attempt_id")
                if aid:
                    done.add(str(aid))
        fail_dir = self.root / "diagnostics" / "failures"
        if fail_dir.is_dir():
            for p in fail_dir.glob("*.json"):
                done.add(p.stem)
        return done

    def already_done(self, branch_id: str, source_ic_id: str) -> bool:
        return attempt_id(branch_id, source_ic_id) in self._completed

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _atomic_write_text(self, path: Path, text: str) -> None:
        self._atomic_write_bytes(path, text.encode("utf-8"))

    def _append_manifest(self, row: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_jsonable(row), ensure_ascii=False) + "\n"
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def _mark_done(self, aid: str) -> None:
        self._completed.add(aid)
        payload = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_attempts": sorted(self._completed),
            "n_completed": len(self._completed),
        }
        self._atomic_write_text(self.progress_path, json.dumps(payload, indent=2))

    def write_success(self, record: dict[str, Any]) -> dict[str, Any]:
        branch_id = record["branch_id"]
        source_ic_id = record["source_ic_id"]
        aid = attempt_id(branch_id, source_ic_id)
        if aid in self._completed and (self.root / branch_id / f"{aid}.pkl").is_file():
            return {"attempt_id": aid, "skipped": True, "reason": "already_present"}

        actions = record.get("actions") or []
        act_arr = np.asarray(actions, np.float32) if len(actions) else np.zeros((0, 7), np.float32)
        handoff = record.get("handoff") or {}
        result = record.get("result") or {}
        phases = list(record.get("phases") or result.get("phases") or [])
        precision_skipped = bool(record.get("precision_skipped") or result.get("precision_skipped"))
        skill = skill_sequence_label(
            phases,
            precision_skipped=precision_skipped,
            continuation=record.get("continuation"),
        )

        heavy = {
            "dataset_version": DATASET_VERSION,
            "attempt_id": aid,
            "expert_id": aid,
            "branch_id": branch_id,
            "source_ic_id": source_ic_id,
            "source_ic_path": str(record.get("source_ic_path") or "").replace("\\", "/"),
            "plan_provenance": record.get("plan_provenance"),
            "stage1": record.get("stage1"),
            "continuation": record.get("continuation"),
            "git_commit": git_commit(),
            "oracle_implementation": (
                ORACLE_IMPLEMENTATION if record.get("continuation") == "precision" else None
            ),
            "config_id": self.config_id or record.get("config_id"),
            "seed": record.get("seed"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "success_criteria_version": SUCCESS_CRITERIA_VERSION,
            "physical_success": True,
            "SUCCESS_NEW": True,
            "SUCCESS_12S": bool(record.get("SUCCESS_12S") or handoff.get("SUCCESS_12S")),
            "status": record.get("status") or result.get("status"),
            "precision_skipped": precision_skipped,
            "precision_executed": bool(record.get("precision_executed") or result.get("precision_executed")),
            "phases": phases,
            "skill_sequence": skill,
            "n_actions": int(act_arr.shape[0]),
            "actions_7d": act_arr,
            "control_period_s": 0.02,
            "action_interface": "legal_7d",
            "handoff": _jsonable({k: v for k, v in handoff.items() if k not in ("series", "samples", "trace")}),
            "decision": _jsonable(result.get("decision") or record.get("decision")),
            "errors_entry": _jsonable(result.get("errors_entry")),
            "errors_exit": _jsonable(result.get("errors_exit")),
            "primitive_logs_summary": _jsonable(
                {
                    k: v
                    for k, v in (result.get("primitive_logs") or {}).items()
                    if k in ("cycles", "task_outcomes", "dx_mechanism", "dz_s1_reused", "entry_grip_settle", "final_down_gate")
                }
            ),
            "capture": record.get("capture"),
            "ref": record.get("ref"),
        }
        raw = pickle.dumps(heavy, protocol=pickle.HIGHEST_PROTOCOL)
        checksum = hashlib.sha256(raw).hexdigest()
        heavy["data_integrity_checksum"] = checksum
        raw = pickle.dumps(heavy, protocol=pickle.HIGHEST_PROTOCOL)

        out_pkl = self.root / branch_id / f"{aid}.pkl"
        self._atomic_write_bytes(out_pkl, raw)

        manifest_row = {
            "dataset_version": DATASET_VERSION,
            "attempt_id": aid,
            "expert_id": aid,
            "branch_id": branch_id,
            "source_ic_id": source_ic_id,
            "source_ic_path": heavy["source_ic_path"],
            "continuation": heavy["continuation"],
            "stage1": heavy["stage1"],
            "physical_success": True,
            "SUCCESS_NEW": True,
            "SUCCESS_12S": heavy["SUCCESS_12S"],
            "oracle_implementation": heavy["oracle_implementation"],
            "git_commit": heavy["git_commit"],
            "config_id": heavy["config_id"],
            "generated_utc": heavy["generated_utc"],
            "success_criteria_version": SUCCESS_CRITERIA_VERSION,
            "status": heavy["status"],
            "precision_skipped": precision_skipped,
            "skill_sequence": skill,
            "n_actions": heavy["n_actions"],
            "record_path": str(out_pkl.relative_to(self.root.parent.parent) if False else out_pkl).replace("\\", "/"),
            "data_integrity_checksum": checksum,
        }
        # Prefer repo-relative path
        try:
            manifest_row["record_path"] = str(out_pkl.resolve().relative_to(ACTIVE_DATASET.parent.parent)).replace("\\", "/")
        except ValueError:
            manifest_row["record_path"] = str(out_pkl).replace("\\", "/")

        self._append_manifest(manifest_row)
        self._mark_done(aid)
        self.rebuild_index()
        return {"attempt_id": aid, "skipped": False, "manifest_row": manifest_row}

    def write_failure(self, record: dict[str, Any]) -> dict[str, Any]:
        branch_id = record["branch_id"]
        source_ic_id = record["source_ic_id"]
        aid = attempt_id(branch_id, source_ic_id)
        reason = classify_failure(record)
        result = record.get("result") or {}
        phases = list(record.get("phases") or result.get("phases") or [])
        payload = {
            "dataset_version": DATASET_VERSION,
            "attempt_id": aid,
            "branch_id": branch_id,
            "source_ic_id": source_ic_id,
            "source_ic_path": str(record.get("source_ic_path") or "").replace("\\", "/"),
            "continuation": record.get("continuation"),
            "stage1": record.get("stage1"),
            "physical_success": False,
            "SUCCESS_NEW": bool(record.get("SUCCESS_NEW") or (result.get("handoff") or {}).get("SUCCESS_NEW")),
            "SUCCESS_12S": bool(record.get("SUCCESS_12S") or (result.get("handoff") or {}).get("SUCCESS_12S") or result.get("SUCCESS_12S")),
            "status": record.get("status") or result.get("status"),
            "failure_class": reason,
            "failures": _jsonable(result.get("failures") or record.get("failures") or []),
            "phases": phases,
            "skill_sequence": skill_sequence_label(
                phases,
                precision_skipped=bool(result.get("precision_skipped")),
                continuation=record.get("continuation"),
            ),
            "n_actions": int(record.get("n_actions") or len(record.get("actions") or [])),
            "exception": record.get("exception"),
            "git_commit": git_commit(),
            "oracle_implementation": (
                ORACLE_IMPLEMENTATION if record.get("continuation") == "precision" else None
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "config_id": self.config_id or record.get("config_id"),
        }
        # Keep compact action summary for diagnostics (not training).
        actions = record.get("actions")
        if actions:
            act = np.asarray(actions, np.float32)
            payload["actions_7d_digest"] = {
                "n": int(act.shape[0]),
                "sha256": hashlib.sha256(act.tobytes()).hexdigest(),
            }
            fail_pkl = self.root / "diagnostics" / "failures" / f"{aid}.pkl"
            self._atomic_write_bytes(
                fail_pkl,
                pickle.dumps({**payload, "actions_7d": act}, protocol=pickle.HIGHEST_PROTOCOL),
            )
        out_json = self.root / "diagnostics" / "failures" / f"{aid}.json"
        self._atomic_write_text(out_json, json.dumps(_jsonable(payload), indent=2))
        self._mark_done(aid)
        return {"attempt_id": aid, "failure_class": reason}

    def rebuild_index(self) -> dict[str, Any]:
        by_branch: dict[str, int] = {}
        n = 0
        if self.manifest_path.is_file():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                bid = str(row.get("branch_id"))
                by_branch[bid] = by_branch.get(bid, 0) + 1
                n += 1
        index = {
            "dataset_version": DATASET_VERSION,
            "n_experts": n,
            "by_branch": by_branch,
            "manifest": str(self.manifest_path).replace("\\", "/"),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        }
        self._atomic_write_text(self.index_path, json.dumps(index, indent=2))
        return index
