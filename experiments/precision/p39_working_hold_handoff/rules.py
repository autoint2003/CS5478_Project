"""Shared HANDOFF rules R0 / R1 / R2.

R0: single-frame WORKING (P38 baseline).
R1: continuous WORKING for HOLD_S (0.5 s); any violation resets dwell.
R2: continuous WORKING + bilateral pad contact + no latched drop for HOLD_S.

Reuse evaluation.success_criteria WORKING_* and pad_bilateral contact definition.
Does not modify physics or STRICT criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.support_drop import latched
from evaluation.success_criteria import HOLD_S, WORKING_POS_MM, WORKING_THETA_DEG
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry
from models.transformer_policy import DT_POLICY

RULE_IDS = ("R0", "R1", "R2")


def rule_spec(rule_id: str) -> dict[str, Any]:
    return {
        "R0": {
            "id": "R0",
            "name": "single_frame_WORKING",
            "dwell_s": 0.0,
            "require_bilateral": False,
            "require_no_latch": True,  # drop still aborts before handoff
            "desc": "HANDOFF on first frame with WORKING (θ≤10°, e_pos≤12 mm).",
        },
        "R1": {
            "id": "R1",
            "name": "WORKING_hold_0.5s",
            "dwell_s": float(HOLD_S),
            "require_bilateral": False,
            "require_no_latch": True,
            "desc": f"Continuous WORKING for {HOLD_S}s; violation resets dwell.",
        },
        "R2": {
            "id": "R2",
            "name": "WORKING_hold_bilateral_0.5s",
            "dwell_s": float(HOLD_S),
            "require_bilateral": True,
            "require_no_latch": True,
            "desc": (
                f"Continuous WORKING + bilateral pad contact + no latched drop "
                f"for {HOLD_S}s; any violation resets dwell."
            ),
        },
    }[rule_id]


@dataclass
class DwellTracker:
    rule_id: str
    dt: float = float(DT_POLICY)
    dwell_s: float = 0.0
    require_bilateral: bool = False
    ok_s: float = 0.0
    first_ok_step: int | None = None
    n_resets: int = 0
    ever_partial: bool = False
    _active: bool = False

    @classmethod
    def from_rule(cls, rule_id: str, dt: float = float(DT_POLICY)) -> "DwellTracker":
        spec = rule_spec(rule_id)
        return cls(
            rule_id=rule_id,
            dt=float(dt),
            dwell_s=float(spec["dwell_s"]),
            require_bilateral=bool(spec["require_bilateral"]),
        )

    def observe(self, sim, step: int) -> dict[str, Any]:
        """Update dwell from current state. Returns status dict; ready⇒HANDOFF now."""
        if latched(sim):
            if self._active:
                self.n_resets += 1
            self._active = False
            self.ok_s = 0.0
            return {
                "ready": False,
                "working_ok": False,
                "bilateral": False,
                "latched": True,
                "ok_s": self.ok_s,
                "frame_ok": False,
            }

        wk = working_geometry(sim)
        working_ok = bool(wk["working_ok"])
        pb = pad_bilateral(sim)
        bilateral = bool(pb["bilateral"])

        if self.rule_id == "R0":
            frame_ok = working_ok
            ready = frame_ok
            if ready and self.first_ok_step is None:
                self.first_ok_step = step
            return {
                "ready": ready,
                "working_ok": working_ok,
                "bilateral": bilateral,
                "latched": False,
                "ok_s": float(self.dt) if ready else 0.0,
                "frame_ok": frame_ok,
                "theta": wk["dtheta_deg"],
                "e_pos": wk["e_pos_mm"],
            }

        frame_ok = working_ok
        if self.require_bilateral:
            frame_ok = frame_ok and bilateral

        if frame_ok:
            if not self._active:
                self._active = True
                if self.first_ok_step is None:
                    self.first_ok_step = step
            self.ok_s += self.dt
            self.ever_partial = True
        else:
            if self._active:
                self.n_resets += 1
            self._active = False
            self.ok_s = 0.0

        ready = bool(self.ok_s + 1e-9 >= self.dwell_s)
        return {
            "ready": ready,
            "working_ok": working_ok,
            "bilateral": bilateral,
            "latched": False,
            "ok_s": self.ok_s,
            "frame_ok": frame_ok,
            "theta": wk["dtheta_deg"],
            "e_pos": wk["e_pos_mm"],
        }


def metric_meta() -> dict[str, Any]:
    return {
        "WORKING_THETA_DEG": float(WORKING_THETA_DEG),
        "WORKING_POS_MM": float(WORKING_POS_MM),
        "HOLD_S": float(HOLD_S),
        "DT_POLICY": float(DT_POLICY),
        "dwell_steps_R1_R2": int(round(float(HOLD_S) / float(DT_POLICY))),
        "bilateral": "pad_bilateral (finger-pad Fn≥FN_PAD_MIN, L and R)",
        "rules": {rid: rule_spec(rid) for rid in RULE_IDS},
    }
