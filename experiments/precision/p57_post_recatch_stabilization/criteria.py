"""Shared recatch / stable-hold detectors (identical for both P57 branches)."""

from __future__ import annotations

from experiments.precision.p57_post_recatch_stabilization.paths import (
    RECATCH_MIN_PAD,
    RECATCH_MIN_STEPS,
    STABLE_HOLD_STEPS,
    STABLE_MIN_PAD,
    STABLE_VREL_MAX,
)


def is_bilateral_ok(m: dict, *, min_pad: int) -> bool:
    return int(m.get("bilateral") or 0) == 1 and int(m.get("min_pad") or 0) >= int(min_pad)


class RecatchTracker:
    """Online detector for first valid recatch and subsequent stable hold / loss."""

    def __init__(self):
        self.bil_run = 0
        self.recatch_step: int | None = None
        self.post_recatch = False
        self.stable_hold = False
        self.lost_after_recatch = False
        self.hold_run = 0
        self.hold_vrels: list[float] = []
        self.steps_since_recatch = 0

    def update(self, step: int, m: dict) -> dict:
        ok_catch = is_bilateral_ok(m, min_pad=RECATCH_MIN_PAD)
        if ok_catch:
            self.bil_run += 1
        else:
            self.bil_run = 0

        if self.recatch_step is None and self.bil_run >= RECATCH_MIN_STEPS:
            self.recatch_step = int(step)
            self.post_recatch = True

        out = {
            "recatch": self.post_recatch,
            "recatch_step": self.recatch_step,
            "just_recaught": False,
            "lost_after_recatch": False,
            "stable_hold": self.stable_hold,
        }
        if self.post_recatch:
            self.steps_since_recatch += 1
            if self.recatch_step == step:
                out["just_recaught"] = True
            ok_hold = is_bilateral_ok(m, min_pad=STABLE_MIN_PAD)
            if ok_hold:
                self.hold_run += 1
                self.hold_vrels.append(float(m.get("vrel") or 0.0))
                if (
                    not self.stable_hold
                    and self.hold_run >= STABLE_HOLD_STEPS
                    and (sum(self.hold_vrels[-STABLE_HOLD_STEPS:]) / STABLE_HOLD_STEPS) <= STABLE_VREL_MAX
                ):
                    self.stable_hold = True
            else:
                if self.hold_run > 0 or ok_catch:
                    # lost sustained contact after having been in post-recatch
                    if not ok_catch:
                        self.lost_after_recatch = True
                        out["lost_after_recatch"] = True
                self.hold_run = 0
                self.hold_vrels = []
            out["stable_hold"] = self.stable_hold
            out["hold_run"] = self.hold_run
            out["lost_after_recatch_latched"] = self.lost_after_recatch
        return out
