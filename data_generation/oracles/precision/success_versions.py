"""Dual orientation success metrics (P9). Does not silently overwrite SUCCESS_NEW.

Official binary SUCCESS_NEW remains the historical 10° hold unless generation
config explicitly selects the relaxed criterion after census validation.

Always report:
  SUCCESS_STRICT_10DEG  — θ≤10°, e_pos≤12 mm, 12S, 0.5 s hold
  SUCCESS_RELAXED_15DEG — θ≤15°, e_pos≤12 mm, 12S, 0.5 s hold
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.success_criteria import HOLD_S, WORKING_POS_MM, WORKING_THETA_DEG, window_stats

STRICT_THETA_DEG = float(WORKING_THETA_DEG)  # 10.0
RELAXED_THETA_DEG = 15.0
SUCCESS_CRITERIA_VERSION_STRICT = "SUCCESS_STRICT_10DEG_v1"
SUCCESS_CRITERIA_VERSION_RELAXED = "SUCCESS_RELAXED_15DEG_v1"
# Candidate official after census — activated only via generation config
# ``admit_success: SUCCESS_RELAXED_15DEG`` (see P10 SUCCESS_RULE_DECISION).
CANDIDATE_OFFICIAL_THETA_DEG = RELAXED_THETA_DEG
OFFICIAL_ADMIT_CANDIDATE = "SUCCESS_RELAXED_15DEG"
# Historical labels are never rewritten; new manifests record which key admitted.


def _hold_ok(samples: list[dict], *, window_start: float, theta_deg: float, pos_mm: float) -> bool:
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


def annotate_dual_theta_success(handoff: dict[str, Any] | None) -> dict[str, Any]:
    """Attach STRICT/RELAXED flags to an existing handoff dict without mutating SUCCESS_NEW meaning."""
    if not handoff:
        return {
            "SUCCESS_STRICT_10DEG": False,
            "SUCCESS_RELAXED_15DEG": False,
            "strict_theta_deg": STRICT_THETA_DEG,
            "relaxed_theta_deg": RELAXED_THETA_DEG,
            "working_pos_mm": WORKING_POS_MM,
        }
    out = dict(handoff)
    s12 = bool(handoff.get("SUCCESS_12S") or handoff.get("success_12s"))
    win = handoff.get("window") or handoff.get("window_SUCCESS_NEW") or {}
    samples = handoff.get("samples")  # may be absent
    # Prefer recomputing from samples when present; else derive from existing SUCCESS_NEW + mean θ
    if samples:
        from envs.outcomes.horizon_outcome import NOMINAL_S
        from evaluation.success_criteria import FINAL_WINDOW_S

        wstart = float(NOMINAL_S) - float(FINAL_WINDOW_S)
        strict = bool(s12 and _hold_ok(samples, window_start=wstart, theta_deg=STRICT_THETA_DEG, pos_mm=WORKING_POS_MM))
        relaxed = bool(s12 and _hold_ok(samples, window_start=wstart, theta_deg=RELAXED_THETA_DEG, pos_mm=WORKING_POS_MM))
    else:
        # Fallback: SUCCESS_NEW ≡ strict; relaxed if 12S and mean/max θ ≤15 when available
        strict = bool(handoff.get("SUCCESS_NEW"))
        mean_th = win.get("mean_theta")
        max_th = win.get("max_theta")
        if mean_th is None:
            relaxed = strict  # cannot prove 15° without window
        else:
            th_ok = float(mean_th) <= RELAXED_THETA_DEG and (
                max_th is None or float(max_th) <= RELAXED_THETA_DEG * 1.25
            )
            pos_ok = True
            if win.get("mean_pos") is not None:
                pos_ok = float(win["mean_pos"]) <= WORKING_POS_MM
            # Prefer hold matrix key if present
            mat = win.get("matrix") or {}
            hold15 = mat.get(f"th{RELAXED_THETA_DEG}_pos{WORKING_POS_MM}") or mat.get("th15.0_pos12.0")
            if hold15 is not None and "SUCCESS_NEW" in hold15:
                relaxed = bool(s12 and hold15.get("SUCCESS_NEW"))
            else:
                # longest_both_ok at 10° is stricter; if 12S and mean≤15, count relaxed candidate
                relaxed = bool(s12 and th_ok and pos_ok)

    out["SUCCESS_STRICT_10DEG"] = bool(strict)
    out["SUCCESS_RELAXED_15DEG"] = bool(relaxed)
    out["strict_theta_deg"] = STRICT_THETA_DEG
    out["relaxed_theta_deg"] = RELAXED_THETA_DEG
    out["success_criteria_versions"] = {
        "strict": SUCCESS_CRITERIA_VERSION_STRICT,
        "relaxed_candidate": SUCCESS_CRITERIA_VERSION_RELAXED,
        "official_unchanged": "SUCCESS_NEW==STRICT_10DEG until census promotes relaxed",
    }
    return out


def extend_window_matrix_for_15deg(win: dict) -> dict:
    """Ensure th15_pos12 appears in window matrix when raw samples unavailable downstream."""
    # window_stats already grids THETA_GRID including 15.0 — no-op helper for docs.
    return win
