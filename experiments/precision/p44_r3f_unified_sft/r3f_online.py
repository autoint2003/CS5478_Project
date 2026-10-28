"""Unified online R3F HANDOFF rule (same for A/B/C/D).

Early (R3): continuous 0.5s of
  WORKING + bilateral + min(nL,nR)>=4 + theta<=8 + e_pos<=9
Fallback: at configurable T_max if R3 never ready.

Does not modify SUCCESS_12S / SUCCESS_NEW thresholds or nominal controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from experiments.precision.p43_r3_working_hold_v2.r3_rules import (
    MARGIN_POS_MM,
    MARGIN_THETA_DEG,
    MIN_PAD_CONTACTS,
    DwellTracker,
)
from experiments.precision.p44_r3f_unified_sft.paths import T_MAX_DEFAULT
from evaluation.success_criteria import HOLD_S
from models.transformer_policy import DT_POLICY

HandoffReason = Literal["R3_EARLY", "TIMEOUT_FALLBACK", "TAPE_END_FALLBACK", "NONE"]


@dataclass
class R3FOnline:
    """Step-wise online R3F monitor."""

    t_max: int = T_MAX_DEFAULT
    dt: float = float(DT_POLICY)
    tracker: DwellTracker | None = None
    reason: HandoffReason = "NONE"
    handoff_step: int | None = None

    def __post_init__(self) -> None:
        if self.tracker is None:
            self.tracker = DwellTracker.from_rule("R3", dt=self.dt)

    @classmethod
    def create(cls, t_max: int = T_MAX_DEFAULT) -> "R3FOnline":
        return cls(t_max=int(t_max), tracker=DwellTracker.from_rule("R3", dt=float(DT_POLICY)))

    def observe(self, sim, step: int, *, tape_ended: bool = False) -> dict[str, Any]:
        """Return status; ready⇒caller should HANDOFF now."""
        assert self.tracker is not None
        st = self.tracker.observe(sim, step)
        ready = False
        reason: HandoffReason = "NONE"

        if st["ready"]:
            ready = True
            reason = "R3_EARLY"
        elif int(step) >= int(self.t_max):
            ready = True
            reason = "TIMEOUT_FALLBACK"
        elif tape_ended:
            # Expert/dataset path: tape finished before T_max without R3
            ready = True
            reason = "TAPE_END_FALLBACK"

        if ready and self.handoff_step is None:
            self.handoff_step = int(step)
            self.reason = reason

        return {
            **st,
            "ready": ready,
            "reason": reason if ready else "NONE",
            "t_max": int(self.t_max),
            "step": int(step),
        }

    def spec(self) -> dict[str, Any]:
        return {
            "rule": "R3F",
            "dwell_s": float(HOLD_S),
            "min_pad": int(MIN_PAD_CONTACTS),
            "theta_max_deg": float(MARGIN_THETA_DEG),
            "pos_max_mm": float(MARGIN_POS_MM),
            "t_max": int(self.t_max),
            "dt": float(self.dt),
            "unified": True,
            "skill_specific": False,
        }
