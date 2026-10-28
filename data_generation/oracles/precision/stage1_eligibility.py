"""Authoritative Stage-1 eligibility: SUCCESS_12S on the exact exit snapshot.

Short bilateral ``entry_gate`` (~150–200 ms) is a contact safety check only and
must never satisfy this contract.

Evidence records are bound to ``capture_fp`` + evaluation configuration so a
12 s simulation need not re-run on every Precision invocation when trusted
evidence already matches the restored physical state.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S
from envs.simulation.capture import restore_capture
from evaluation.axis_aware_success import evaluate_handoff_with_axis

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE_PATH = ROOT / "data" / "stage1_eligibility_evidence" / "evidence_by_capture_fp.json"
EVAL_CONFIG_ID = "SUCCESS_12S_nominal_v1"
EVAL_CONFIG = {
    "eval_config_id": EVAL_CONFIG_ID,
    "NOMINAL_S": float(NOMINAL_S),
    "LOST_STEPS": int(LOST_STEPS),
    "stabilize_n": 0,
    "evaluator": "evaluation.axis_aware_success.evaluate_handoff_with_axis",
    "replay_stabilization_prefix": False,
    "note": (
        "Precision restores blob['capture'] as Stage-1 exit. "
        "stabilization_prefix_actions are NOT replayed (would alter qpos)."
    ),
}

STATUS_ELIGIBLE = "STAGE1_ELIGIBLE"
STATUS_INELIGIBLE = "STAGE1_INELIGIBLE"
REASON_SHORT_HOLD_NOT_SUFFICIENT = "SHORT_HOLD_NOT_STAGE1_SUCCESS"
REASON_MISSING_EVIDENCE = "MISSING_OR_MISMATCHED_EVIDENCE"
REASON_FAILED_12S = "SUCCESS_12S_FALSE"
REASON_MISSING_CAPTURE = "MISSING_CAPTURE_OR_REF"


@dataclass
class Stage1EligibilityResult:
    eligible: bool
    status: str
    reason: str | None
    capture_fp: str | None = None
    SUCCESS_12S: bool | None = None
    SUCCESS_NEW: bool | None = None
    evidence_source: str | None = None  # "registry" | "live_verify" | None
    eval_config: dict = field(default_factory=lambda: dict(EVAL_CONFIG))
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Stage1NotEligible(RuntimeError):
    """Raised when callers request hard-fail admission semantics."""

    def __init__(self, result: Stage1EligibilityResult):
        self.result = result
        super().__init__(f"{result.status}:{result.reason}")


def capture_fingerprint(capture: Any) -> str:
    raw = pickle.dumps(capture, protocol=4)
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_EVIDENCE_PATH
    if not p.is_file():
        return {"eval_config": EVAL_CONFIG, "by_capture_fp": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_evidence_record(
    *,
    capture_fp: str,
    sample_id: str | None,
    SUCCESS_12S: bool,
    SUCCESS_NEW: bool | None = None,
    reason: str | None = None,
    path: Path | None = None,
    extra: dict | None = None,
) -> None:
    """Append/update a registry row (does not mutate IC pickles)."""
    p = path or DEFAULT_EVIDENCE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    reg = _load_registry(p)
    by = reg.setdefault("by_capture_fp", {})
    by[capture_fp] = {
        "capture_fp": capture_fp,
        "sample_id": sample_id,
        "SUCCESS_12S": bool(SUCCESS_12S),
        "SUCCESS_NEW": None if SUCCESS_NEW is None else bool(SUCCESS_NEW),
        "reason": reason,
        "eval_config": dict(EVAL_CONFIG),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    reg["eval_config"] = dict(EVAL_CONFIG)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def lookup_evidence(capture_fp: str, *, path: Path | None = None) -> dict[str, Any] | None:
    reg = _load_registry(path)
    row = (reg.get("by_capture_fp") or {}).get(capture_fp)
    if not row:
        return None
    cfg = row.get("eval_config") or {}
    # Reject evidence recorded under a different evaluation contract
    if float(cfg.get("NOMINAL_S") or 0) != float(NOMINAL_S):
        return None
    if int(cfg.get("LOST_STEPS") or -1) != int(LOST_STEPS):
        return None
    if int(cfg.get("stabilize_n") if cfg.get("stabilize_n") is not None else -1) != 0:
        return None
    if cfg.get("replay_stabilization_prefix") is True:
        return None
    return row


def live_verify_stage1_12s(env, blob: dict) -> Stage1EligibilityResult:
    """Run SUCCESS_12S from exact capture; does not mutate stored blob."""
    if "capture" not in blob or "ref" not in blob:
        return Stage1EligibilityResult(
            eligible=False,
            status=STATUS_INELIGIBLE,
            reason=REASON_MISSING_CAPTURE,
        )
    fp = capture_fingerprint(blob["capture"])
    restore_capture(env, blob["capture"])
    h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=0)
    # Restore original snapshot after verification (verification mutates sim only)
    restore_capture(env, blob["capture"])
    ok = bool(h.get("SUCCESS_12S"))
    return Stage1EligibilityResult(
        eligible=ok,
        status=STATUS_ELIGIBLE if ok else STATUS_INELIGIBLE,
        reason=None if ok else (str(h.get("reason") or REASON_FAILED_12S)),
        capture_fp=fp,
        SUCCESS_12S=ok,
        SUCCESS_NEW=bool(h.get("SUCCESS_NEW")) if ok else bool(h.get("SUCCESS_NEW")),
        evidence_source="live_verify",
        detail={
            "handoff_reason": h.get("reason"),
            "pre_theta": h.get("pre_theta"),
            "pre_pos_mm": h.get("pre_pos_mm"),
            "pre_bilateral": h.get("pre_bilateral"),
        },
    )


def check_stage1_eligible(
    env,
    blob: dict,
    *,
    allow_live_verify: bool = True,
    evidence_path: Path | None = None,
    record_live: bool = False,
) -> Stage1EligibilityResult:
    """Authoritative eligibility check.

    Prefer registry evidence bound to ``capture_fp``; optionally live-verify.
    Never treats short ``entry_gate`` as Stage-1 success.
    """
    if not isinstance(blob, dict) or "capture" not in blob or "ref" not in blob:
        return Stage1EligibilityResult(
            eligible=False,
            status=STATUS_INELIGIBLE,
            reason=REASON_MISSING_CAPTURE,
            detail={"hint": REASON_SHORT_HOLD_NOT_SUFFICIENT},
        )
    fp = capture_fingerprint(blob["capture"])
    ev = lookup_evidence(fp, path=evidence_path)
    if ev is not None:
        ok = bool(ev.get("SUCCESS_12S"))
        return Stage1EligibilityResult(
            eligible=ok,
            status=STATUS_ELIGIBLE if ok else STATUS_INELIGIBLE,
            reason=None if ok else (str(ev.get("reason") or REASON_FAILED_12S)),
            capture_fp=fp,
            SUCCESS_12S=ok,
            SUCCESS_NEW=ev.get("SUCCESS_NEW"),
            evidence_source="registry",
            detail={"sample_id": ev.get("sample_id"), "registry_reason": ev.get("reason")},
        )
    if not allow_live_verify:
        return Stage1EligibilityResult(
            eligible=False,
            status=STATUS_INELIGIBLE,
            reason=REASON_MISSING_EVIDENCE,
            capture_fp=fp,
            detail={"hint": REASON_SHORT_HOLD_NOT_SUFFICIENT},
        )
    res = live_verify_stage1_12s(env, blob)
    if record_live and res.capture_fp:
        save_evidence_record(
            capture_fp=res.capture_fp,
            sample_id=(blob.get("sample_id") or blob.get("ic_id")),
            SUCCESS_12S=bool(res.SUCCESS_12S),
            SUCCESS_NEW=res.SUCCESS_NEW,
            reason=res.reason,
            path=evidence_path,
            extra={"source": "live_verify"},
        )
    return res


def require_stage1_eligible(env, blob: dict, **kwargs) -> Stage1EligibilityResult:
    """Hard-fail wrapper for pipelines that must not continue on ineligible ICs."""
    res = check_stage1_eligible(env, blob, **kwargs)
    if not res.eligible:
        raise Stage1NotEligible(res)
    return res


def stage1_ineligible_precision_result(plan_id: str, elig: Stage1EligibilityResult, blob: dict):
    """Build a PrecisionResult that records STAGE1_INELIGIBLE without executing corrections."""
    from data_generation.oracles.precision.state import PrecisionResult

    return PrecisionResult(
        status="STAGE1_INELIGIBLE",
        plan_id=str(plan_id),
        precision_skipped=True,
        precision_executed=False,
        gate_ok=False,
        phases=["stage1_12s_eligibility"],
        failures=[f"STAGE1_INELIGIBLE:{elig.reason}"],
        note=(
            "Stage-1 SUCCESS_12S required before Precision. "
            "Short entry_gate / bilateral hold is not sufficient."
        ),
        capture=blob.get("capture"),
        ref=blob.get("ref"),
        primitive_logs={"stage1_eligibility": elig.to_dict()},
    )
