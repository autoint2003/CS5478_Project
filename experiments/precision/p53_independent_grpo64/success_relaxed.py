"""SUCCESS_RELAXED_12S — experimental terminal metric (does not redefine SUCCESS_NEW/12S).

S_relaxed = SUCCESS_12S ∧ (e_p ≤ ε_p) ∧ (e_θ ≤ ε_θ) with HOLD_S dwell in final window.

ε chosen strictly looser than R3F HANDOFF (8° / 9 mm) and SUCCESS_NEW (10° / 12 mm).
Not equivalent to SUCCESS_RELAXED_15DEG (which keeps WORKING_POS_MM=12).
"""

from __future__ import annotations

from typing import Any

from envs.outcomes.horizon_outcome import NOMINAL_S
from evaluation.success_criteria import FINAL_WINDOW_S, HOLD_S, WORKING_POS_MM, WORKING_THETA_DEG
from experiments.precision.p43_r3_working_hold_v2.r3_rules import MARGIN_POS_MM, MARGIN_THETA_DEG

# Terminal pose (less restrictive than HANDOFF geometry)
RELAXED_THETA_DEG = 15.0
RELAXED_POS_MM = 15.0

SUCCESS_RELAXED_VERSION = "SUCCESS_RELAXED_12S_v1"

assert RELAXED_THETA_DEG > float(MARGIN_THETA_DEG)  # > HANDOFF 8°
assert RELAXED_POS_MM > float(MARGIN_POS_MM)  # > HANDOFF 9 mm
assert RELAXED_THETA_DEG >= float(WORKING_THETA_DEG)  # ≥ SUCCESS_NEW θ
assert RELAXED_POS_MM > float(WORKING_POS_MM)  # > SUCCESS_NEW pos


def _hold_ok(samples: list[dict], *, window_start: float, theta_deg: float, pos_mm: float) -> bool:
    """Longest continuous in-band bilateral-agnostic pose hold ≥ HOLD_S (matches success_versions)."""
    import numpy as np

    win = [s for s in samples if s["t_rel"] + 1e-9 >= window_start]
    if not win:
        return False
    thetas = np.array([s["e_theta_deg"] for s in win], float)
    poss = np.array([s["e_pos_mm"] for s in win], float)
    dt = 0.02
    if len(win) > 1:
        dts = np.diff([s["t_rel"] for s in win])
        if len(dts):
            dt = float(np.median(dts))
    ok = (thetas <= theta_deg) & (poss <= pos_mm)
    longest = cur = 0.0
    for v in ok:
        if v:
            cur += dt
            longest = max(longest, cur)
        else:
            cur = 0.0
    return bool(longest + 1e-9 >= HOLD_S)


def evaluate_success_relaxed_12s(handoff_ev: dict[str, Any] | None) -> dict[str, Any]:
    """Attach SUCCESS_RELAXED_12S without mutating SUCCESS_12S / SUCCESS_NEW."""
    if not handoff_ev:
        return {
            "SUCCESS_RELAXED_12S": False,
            "SUCCESS_12S": False,
            "SUCCESS_NEW": False,
            "relaxed_theta_deg": RELAXED_THETA_DEG,
            "relaxed_pos_mm": RELAXED_POS_MM,
            "version": SUCCESS_RELAXED_VERSION,
        }
    s12 = bool(handoff_ev.get("SUCCESS_12S"))
    snew = bool(handoff_ev.get("SUCCESS_NEW"))
    samples = handoff_ev.get("samples") or []
    wstart = float(NOMINAL_S) - float(FINAL_WINDOW_S)
    pose_ok = bool(
        s12
        and samples
        and _hold_ok(
            samples,
            window_start=wstart,
            theta_deg=RELAXED_THETA_DEG,
            pos_mm=RELAXED_POS_MM,
        )
    )
    # Without samples: cannot claim relaxed pose; do not invent success from NEW alone
    if not samples:
        pose_ok = False
    return {
        "SUCCESS_RELAXED_12S": bool(pose_ok),
        "SUCCESS_12S": s12,
        "SUCCESS_NEW": snew,
        "relaxed_theta_deg": RELAXED_THETA_DEG,
        "relaxed_pos_mm": RELAXED_POS_MM,
        "hold_s": float(HOLD_S),
        "final_window_s": float(FINAL_WINDOW_S),
        "window_start_s": wstart,
        "version": SUCCESS_RELAXED_VERSION,
        "note": (
            "Requires SUCCESS_12S (12s nominal stability + lift) AND "
            f"θ≤{RELAXED_THETA_DEG}° and e_pos≤{RELAXED_POS_MM}mm for ≥{HOLD_S}s "
            f"in the final {FINAL_WINDOW_S}s of nominal. "
            "Not equivalent to SUCCESS_RELAXED_15DEG (pos still 12mm there)."
        ),
    }
