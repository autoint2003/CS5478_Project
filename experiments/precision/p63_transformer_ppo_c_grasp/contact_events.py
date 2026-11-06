"""Privileged contact-geometry events for C grasp reward (training/audit only).

Uses Plan-C ``classify_contacts`` — not written into Actor/Critic observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from experiments.precision.p63_transformer_ppo_c_grasp.paths import (
    DT,
    R_BAD_CONTACT,
    R_CORRECT_RECATCH,
    R_LOST_AFTER,
    R_R3F_READY,
    R_SUSTAINED,
    SUSTAIN_S,
)


def is_correct_sidewall(audit: dict) -> bool:
    """True iff opposing sidewall bilateral grasp (not mere bilateral pad count)."""
    if not audit:
        return False
    if str(audit.get("grasp_kind") or "") != "opposing_sidewall":
        return False
    if not int(audit.get("opposing_sidewall") or 0):
        return False
    if int(audit.get("n_end_face") or 0) > 0:
        return False
    # Require both sides present with sidewall majority (already in oppose)
    return bool(audit.get("bilateral"))


def is_bad_contact(audit: dict) -> bool:
    kind = str(audit.get("grasp_kind") or "")
    return kind in ("end_face_pinch", "mixed_face_wall", "edge_only")


@dataclass
class CGraspEventTracker:
    """Per-episode once-only event rewards for C."""

    awarded_recatch: bool = False
    awarded_sustain: bool = False
    awarded_r3f: bool = False
    awarded_bad: bool = False
    awarded_lost: bool = False
    had_correct: bool = False
    sustain_steps: int = 0
    steps_since_recatch: int = 0
    pos_total: float = 0.0
    neg_total: float = 0.0
    log: list = field(default_factory=list)

    def reset(self):
        self.__init__()

    def step(
        self,
        sim,
        *,
        metrics: dict,
        r3f_early_ready: bool,
        dropped: bool,
    ) -> tuple[float, dict]:
        """Return (delta_reward, event_flags)."""
        audit = classify_contacts(sim)
        correct = is_correct_sidewall(audit)
        bad = is_bad_contact(audit)
        bilateral = bool(audit.get("bilateral"))
        vrel = float(metrics.get("vrel") or 0.0)
        delta = 0.0
        flags = {
            "correct_sidewall": int(correct),
            "grasp_kind": audit.get("grasp_kind"),
            "opposing_sidewall": int(audit.get("opposing_sidewall") or 0),
            "event_recatch": 0,
            "event_sustain": 0,
            "event_r3f": 0,
            "event_bad": 0,
            "event_lost": 0,
            "r_c_grasp": 0.0,
        }

        # 3.1 first correct sidewall recatch
        if correct and not self.awarded_recatch:
            delta += R_CORRECT_RECATCH
            self.pos_total += R_CORRECT_RECATCH
            self.awarded_recatch = True
            self.had_correct = True
            self.sustain_steps = 1
            self.steps_since_recatch = 0
            flags["event_recatch"] = 1
            self.log.append("correct_recatch")
        elif correct and self.had_correct and not self.awarded_sustain:
            self.sustain_steps += 1
            need = int(round(SUSTAIN_S / DT))
            if self.sustain_steps >= need:
                delta += R_SUSTAINED
                self.pos_total += R_SUSTAINED
                self.awarded_sustain = True
                flags["event_sustain"] = 1
                self.log.append("sustained_bilateral")
        else:
            # break continuous sustain streak
            if self.had_correct and not self.awarded_sustain and not correct:
                self.sustain_steps = 0

        if self.had_correct:
            self.steps_since_recatch += 1

        # 3.3 R3F-ready (early dwell only)
        if r3f_early_ready and not self.awarded_r3f:
            delta += R_R3F_READY
            self.pos_total += R_R3F_READY
            self.awarded_r3f = True
            flags["event_r3f"] = 1
            self.log.append("r3f_ready")

        # 3.4 bad contact then pop / high slip
        if (
            bad
            and not self.awarded_bad
            and (dropped or (not bilateral and vrel > 0.35))
        ):
            delta += R_BAD_CONTACT
            self.neg_total += abs(R_BAD_CONTACT)
            self.awarded_bad = True
            flags["event_bad"] = 1
            self.log.append("bad_contact_pop")

        # lost after correct recatch within 0.5s
        if (
            self.had_correct
            and not self.awarded_lost
            and self.steps_since_recatch <= int(round(SUSTAIN_S / DT))
            and (dropped or not bilateral)
            and not correct
        ):
            delta += R_LOST_AFTER
            self.neg_total += abs(R_LOST_AFTER)
            self.awarded_lost = True
            flags["event_lost"] = 1
            self.log.append("lost_after_recatch")

        flags["r_c_grasp"] = float(delta)
        flags["c_pos_total"] = float(self.pos_total)
        flags["c_neg_total"] = float(self.neg_total)
        return float(delta), flags


def offline_event_summary(tracker: CGraspEventTracker) -> dict:
    return {
        "events": list(tracker.log),
        "pos_total": tracker.pos_total,
        "neg_total": tracker.neg_total,
        "awarded_recatch": tracker.awarded_recatch,
        "awarded_sustain": tracker.awarded_sustain,
        "awarded_r3f": tracker.awarded_r3f,
    }
