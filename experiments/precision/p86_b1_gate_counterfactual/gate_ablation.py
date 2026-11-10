"""Diagnostic HANDOFF monitors for P86 — does not modify official R3FOnline / R3 rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.support_drop import latched
from evaluation.success_criteria import HOLD_S, WORKING_POS_MM, WORKING_THETA_DEG
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry
from experiments.precision.p43_r3_working_hold_v2.r3_rules import MIN_PAD_CONTACTS
from models.transformer_policy import DT_POLICY

HandoffReason = Literal["ABLATION_EARLY", "TIMEOUT_FALLBACK", "NONE"]

# Official R3 margin (G0). Ablations only change pos_max / dwell_s.
R3_THETA_DEG = 8.0
R3_POS_MM = 9.0


@dataclass
class AblationDwellTracker:
    """Same structure as DwellTracker.observe, parameterized for P86 only."""

    pos_max_mm: float
    theta_max_deg: float = R3_THETA_DEG
    dwell_s: float = float(HOLD_S)
    dt: float = float(DT_POLICY)
    require_bilateral: bool = True
    min_pad: int = int(MIN_PAD_CONTACTS)
    ok_s: float = 0.0
    first_ok_step: int | None = None
    n_resets: int = 0
    _active: bool = False

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
                "theta": None,
                "e_pos": None,
            }

        wk = working_geometry(sim)
        theta = float(wk["dtheta_deg"])
        e_pos = float(wk["e_pos_mm"])
        # Preserve official WORKING predicate (not silently weakened).
        working_ok = bool(wk["working_ok"])
        pb = pad_bilateral(sim)
        nL, nR = int(pb["nL"]), int(pb["nR"])
        bilateral = bool(pb["bilateral"])
        min_pad_ok = min(nL, nR) >= self.min_pad if self.min_pad > 0 else True
        margin_ok = theta <= self.theta_max_deg + 1e-9 and e_pos <= self.pos_max_mm + 1e-9

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
        else:
            if self._active:
                self.n_resets += 1
            self._active = False
            self.ok_s = 0.0

        # dwell_s==0 means instantaneous HANDOFF on the first frame_ok only.
        # Do NOT use ok_s>=0 (always true) — that would fire HANDOFF every step.
        if float(self.dwell_s) <= 0.0:
            ready = bool(frame_ok)
        else:
            ready = bool(self.ok_s + 1e-9 >= float(self.dwell_s))
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


@dataclass
class AblationHandoffMonitor:
    """Drop-in replacement for env.mon.observe API used by StabilizeGrpoEnv."""

    tracker: AblationDwellTracker
    t_max: int
    variant: str
    reason: HandoffReason = "NONE"
    handoff_step: int | None = None

    def observe(self, sim, step: int, *, tape_ended: bool = False) -> dict[str, Any]:
        st = self.tracker.observe(sim, step)
        ready = False
        reason: HandoffReason = "NONE"
        if st["ready"]:
            ready = True
            reason = "ABLATION_EARLY"
        elif int(step) >= int(self.t_max):
            ready = True
            reason = "TIMEOUT_FALLBACK"
        if ready and self.handoff_step is None:
            self.handoff_step = int(step)
            self.reason = reason
        return {
            **st,
            "ready": ready,
            "reason": reason if ready else "NONE",
            "t_max": int(self.t_max),
            "step": int(step),
            "variant": self.variant,
        }


GATE_SPECS: dict[str, dict[str, Any]] = {
    "G0": {
        "label": "original_R3F",
        "pos_max_mm": R3_POS_MM,
        "dwell_s": float(HOLD_S),
        "use_official_r3f": True,
        "note": "Official R3FOnline (9 mm margin + HOLD_S dwell). Unchanged implementation.",
    },
    "G1": {
        "label": "pos12_dwell_on",
        "pos_max_mm": float(WORKING_POS_MM),  # 12 mm
        "dwell_s": float(HOLD_S),
        "use_official_r3f": False,
        "note": "Margin pos ≤12 mm; dwell unchanged. WORKING/bilateral/min_pad/θ≤8 preserved.",
    },
    "G2": {
        "label": "pos9_dwell_off",
        "pos_max_mm": R3_POS_MM,
        "dwell_s": 0.0,
        "use_official_r3f": False,
        "note": "9 mm margin; dwell removed (ready on first frame_ok).",
    },
    "G3": {
        "label": "pos12_dwell_off",
        "pos_max_mm": float(WORKING_POS_MM),
        "dwell_s": 0.0,
        "use_official_r3f": False,
        "note": "12 mm margin; dwell removed. Still requires WORKING+bilateral+min_pad+θ≤8.",
    },
}


def make_monitor(variant: str, *, t_max: int) -> Any:
    """Return official R3FOnline for G0, else AblationHandoffMonitor."""
    from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline

    spec = GATE_SPECS[variant]
    if spec["use_official_r3f"]:
        return R3FOnline.create(t_max=int(t_max))
    tracker = AblationDwellTracker(
        pos_max_mm=float(spec["pos_max_mm"]),
        theta_max_deg=R3_THETA_DEG,
        dwell_s=float(spec["dwell_s"]),
        min_pad=int(MIN_PAD_CONTACTS),
    )
    return AblationHandoffMonitor(tracker=tracker, t_max=int(t_max), variant=variant)


def gate_constants() -> dict[str, Any]:
    return {
        "WORKING_THETA_DEG": float(WORKING_THETA_DEG),
        "WORKING_POS_MM": float(WORKING_POS_MM),
        "HOLD_S": float(HOLD_S),
        "R3_THETA_DEG": R3_THETA_DEG,
        "R3_POS_MM": R3_POS_MM,
        "MIN_PAD_CONTACTS": int(MIN_PAD_CONTACTS),
        "DT_POLICY": float(DT_POLICY),
        "variants": GATE_SPECS,
    }
