"""D2 (independent stable IC) eligibility — not Stage-1 A/B/C provenance.

D2 has no Plan A/B/C recovery prefix. Eligibility is:
  1. Documented reproducible IC (HARD_IC pool)
  2. Valid capture+ref
  3. Live SUCCESS_12S on the snapshot (independent stability)
  4. Meaningful Precision need (assess_need)
  5. No teleport during subsequent expert trajectory (enforced by pipeline)

Does not bypass contact/safety checks during Precision or handoff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint
from envs.simulation.capture import restore_capture
from evaluation.axis_aware_success import evaluate_handoff_with_axis


@dataclass
class D2EligibilityResult:
    eligible: bool
    status: str
    reason: str | None
    capture_fp: str | None = None
    SUCCESS_12S: bool | None = None
    SUCCESS_STRICT_10DEG: bool | None = None
    SUCCESS_RELAXED_15DEG: bool | None = None
    precision_required: bool | None = None
    already_within_target: bool | None = None
    need: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_d2_eligible(
    env,
    blob: dict,
    *,
    cfg: PrecisionConfig | None = None,
    stabilize_n: int = 0,
) -> D2EligibilityResult:
    """Live independent stability + Precision-need gate for D2 ICs."""
    pcfg = cfg or PrecisionConfig(evaluate_handoff=True)
    if "capture" not in blob or "ref" not in blob:
        return D2EligibilityResult(
            eligible=False,
            status="D2_INELIGIBLE",
            reason="MISSING_CAPTURE_OR_REF",
        )
    fp = capture_fingerprint(blob["capture"])
    restore_capture(env, blob["capture"])
    hand = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=stabilize_n)
    restore_capture(env, blob["capture"])
    s12 = bool(hand.get("SUCCESS_12S"))
    strict = bool(hand.get("SUCCESS_STRICT_10DEG") or hand.get("SUCCESS_NEW"))
    relaxed = bool(hand.get("SUCCESS_RELAXED_15DEG"))
    if not s12:
        return D2EligibilityResult(
            eligible=False,
            status="D2_INELIGIBLE",
            reason="SUCCESS_12S_FALSE",
            capture_fp=fp,
            SUCCESS_12S=False,
            SUCCESS_STRICT_10DEG=strict,
            SUCCESS_RELAXED_15DEG=relaxed,
            detail={"handoff_reason": hand.get("reason")},
        )
    err = measure_errors(env.sim)
    need = assess_need(err, pcfg)
    precision_required = bool(
        need.get("dx_needed") or need.get("dz_needed") or need.get("dtheta_needed")
        or not need.get("ready_for_direct_handoff_assessment")
    )
    already_ok = bool(need.get("ready_for_direct_handoff_assessment") and not precision_required)
    if already_ok or not precision_required:
        return D2EligibilityResult(
            eligible=False,
            status="D2_STABLE_NOT_PRECISION_REQUIRED",
            reason="ALREADY_WITHIN_TARGET",
            capture_fp=fp,
            SUCCESS_12S=True,
            SUCCESS_STRICT_10DEG=strict,
            SUCCESS_RELAXED_15DEG=relaxed,
            precision_required=False,
            already_within_target=True,
            need={k: need[k] for k in need if k != "axis_decision"},
        )
    return D2EligibilityResult(
        eligible=True,
        status="D2_ELIGIBLE",
        reason=None,
        capture_fp=fp,
        SUCCESS_12S=True,
        SUCCESS_STRICT_10DEG=strict,
        SUCCESS_RELAXED_15DEG=relaxed,
        precision_required=True,
        already_within_target=False,
        need={k: need[k] for k in need if k != "axis_decision"},
        detail={"pre_theta": hand.get("pre_theta"), "pre_pos_mm": hand.get("pre_pos_mm")},
    )
