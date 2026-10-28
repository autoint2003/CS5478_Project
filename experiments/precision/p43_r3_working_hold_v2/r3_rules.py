"""R3 = R2 + contact richness + geometric interior margin.

R2 baseline: continuous WORKING + bilateral + no latch for HOLD_S.
R3 extras (every frame of the dwell):
  - min(nL, nR) >= MIN_PAD_CONTACTS
  - theta <= MARGIN_THETA_DEG
  - e_pos <= MARGIN_POS_MM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.support_drop import latched
from evaluation.success_criteria import HOLD_S, WORKING_POS_MM, WORKING_THETA_DEG
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry
from models.transformer_policy import DT_POLICY

# Proposed R3 constants (not SUCCESS_NEW thresholds — trigger only).
MIN_PAD_CONTACTS = 4
MARGIN_THETA_DEG = 8.0
MARGIN_POS_MM = 9.0

RULE_IDS = ("R2", "R3")


def rule_spec(rule_id: str) -> dict[str, Any]:
    if rule_id == "R2":
        return {
            "id": "R2",
            "name": "WORKING_hold_bilateral_0.5s",
            "dwell_s": float(HOLD_S),
            "require_bilateral": True,
            "min_pad": 0,
            "theta_max": float(WORKING_THETA_DEG),
            "pos_max": float(WORKING_POS_MM),
            "desc": "R2 baseline: WORKING + bilateral + no latch for 0.5s.",
        }
    if rule_id == "R3":
        return {
            "id": "R3",
            "name": "WORKING_HOLD_V2",
            "dwell_s": float(HOLD_S),
            "require_bilateral": True,
            "min_pad": int(MIN_PAD_CONTACTS),
            "theta_max": float(MARGIN_THETA_DEG),
            "pos_max": float(MARGIN_POS_MM),
            "desc": (
                f"R2 + min(nL,nR)>={MIN_PAD_CONTACTS} + "
                f"θ≤{MARGIN_THETA_DEG}° + e_pos≤{MARGIN_POS_MM} mm for {HOLD_S}s."
            ),
        }
    raise KeyError(rule_id)


@dataclass
class DwellTracker:
    rule_id: str
    dt: float = float(DT_POLICY)
    dwell_s: float = 0.0
    require_bilateral: bool = True
    min_pad: int = 0
    theta_max: float = float(WORKING_THETA_DEG)
    pos_max: float = float(WORKING_POS_MM)
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
            min_pad=int(spec["min_pad"]),
            theta_max=float(spec["theta_max"]),
            pos_max=float(spec["pos_max"]),
        )

    def observe(self, sim, step: int) -> dict[str, Any]:
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
                "nL": 0,
                "nR": 0,
                "min_pad_ok": False,
                "margin_ok": False,
            }

        wk = working_geometry(sim)
        theta = float(wk["dtheta_deg"])
        e_pos = float(wk["e_pos_mm"])
        working_ok = bool(wk["working_ok"])
        pb = pad_bilateral(sim)
        nL, nR = int(pb["nL"]), int(pb["nR"])
        bilateral = bool(pb["bilateral"])
        min_pad_ok = min(nL, nR) >= self.min_pad if self.min_pad > 0 else True
        margin_ok = theta <= self.theta_max + 1e-9 and e_pos <= self.pos_max + 1e-9

        frame_ok = working_ok
        if self.require_bilateral:
            frame_ok = frame_ok and bilateral
        frame_ok = frame_ok and min_pad_ok and margin_ok

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
            "theta": theta,
            "e_pos": e_pos,
            "nL": nL,
            "nR": nR,
            "min_pad_ok": min_pad_ok,
            "margin_ok": margin_ok,
        }


def metric_meta() -> dict[str, Any]:
    return {
        "WORKING_THETA_DEG": float(WORKING_THETA_DEG),
        "WORKING_POS_MM": float(WORKING_POS_MM),
        "HOLD_S": float(HOLD_S),
        "DT_POLICY": float(DT_POLICY),
        "R3_MIN_PAD_CONTACTS": MIN_PAD_CONTACTS,
        "R3_MARGIN_THETA_DEG": MARGIN_THETA_DEG,
        "R3_MARGIN_POS_MM": MARGIN_POS_MM,
        "rules": {rid: rule_spec(rid) for rid in RULE_IDS},
    }
