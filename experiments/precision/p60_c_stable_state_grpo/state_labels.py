"""Capture-time geometry/contact labels vs future trajectory outcome (P60 audit fix).

Capture labels use R3 HANDOFF gate quantities (θ, e_pos, bilateral pad), not only
a generic position-error metric. Multiple geometry violations may be active at once.
Future outcomes are stored separately and never used as the sole capture label.
"""

from __future__ import annotations

from typing import Any

from experiments.precision.p43_r3_working_hold_v2.r3_rules import (
    MARGIN_POS_MM,
    MARGIN_THETA_DEG,
    MIN_PAD_CONTACTS,
)

# R3 gate thresholds (HANDOFF trigger interior margin)
R3_THETA_MAX = float(MARGIN_THETA_DEG)  # 8.0
R3_POS_MAX = float(MARGIN_POS_MM)  # 9.0
R3_MIN_PAD = int(MIN_PAD_CONTACTS)  # 4


def capture_geometry_flags(metrics: dict) -> dict[str, Any]:
    """Label the *capture* state using R3 gate quantities. Multi-label allowed."""
    e_pos = float(metrics.get("e_pos_mm") or 0.0)
    theta = float(metrics.get("theta") or 0.0)
    min_pad = int(metrics.get("min_pad") or 0)
    bilateral = int(metrics.get("bilateral") or 0) == 1
    vrel = float(metrics.get("vrel") or 0.0)
    latched = bool(metrics.get("latched"))

    angular_error = theta > R3_THETA_MAX + 1e-9
    position_error = e_pos > R3_POS_MAX + 1e-9
    contact_quality = (not bilateral) or (min_pad < R3_MIN_PAD)
    # Within R3 margins on all axes (geometry+contact); dwell may still be incomplete
    r3_geometry_ok = (
        bilateral
        and min_pad >= R3_MIN_PAD
        and theta <= R3_THETA_MAX + 1e-9
        and e_pos <= R3_POS_MAX + 1e-9
        and not latched
    )

    labels: list[str] = []
    if angular_error:
        labels.append("angular_error")
    if position_error:
        labels.append("position_error")
    if contact_quality:
        labels.append("contact_quality")
    if not labels and not r3_geometry_ok:
        labels.append("other_gate_fail")
    if not labels and r3_geometry_ok:
        labels.append("r3_geometry_ok_dwell_incomplete")

    # Primary tag: worst violation (angular > position > contact), for bucketing only
    if angular_error and (not position_error or theta >= e_pos):
        primary = "angular_error"
    elif position_error:
        primary = "position_error"
    elif contact_quality:
        primary = "contact_quality"
    elif r3_geometry_ok:
        primary = "r3_geometry_ok_dwell_incomplete"
    else:
        primary = "other_gate_fail"

    return {
        "e_pos_mm": e_pos,
        "theta_deg": theta,
        "min_pad": min_pad,
        "bilateral": int(bilateral),
        "vrel": vrel,
        "latched": int(latched),
        "r3_theta_max": R3_THETA_MAX,
        "r3_pos_max": R3_POS_MAX,
        "r3_min_pad": R3_MIN_PAD,
        "angular_error": bool(angular_error),
        "position_error": bool(position_error),
        "contact_quality": bool(contact_quality),
        "r3_geometry_ok": bool(r3_geometry_ok),
        "capture_labels": labels,
        "primary_capture_label": primary,
    }


def future_outcome_labels(
    *,
    lost_after: bool,
    truncated: bool,
    handoff: bool,
    success: bool,
    drop: bool,
) -> dict[str, Any]:
    """Labels for what happened *after* capture in the original trajectory."""
    tags: list[str] = []
    if success:
        tags.append("eventually_success")
    if handoff and not success:
        tags.append("handoff_then_fail")
    if lost_after:
        tags.append("loses_contact")
    if truncated and not handoff:
        tags.append("timeout_before_handoff")
    if drop:
        tags.append("drop")
    if not tags:
        tags.append("other_fail")
    return {
        "future_labels": tags,
        "primary_future_label": tags[0],
        "lost_after": bool(lost_after),
        "truncated": bool(truncated),
        "handoff": bool(handoff),
        "success": bool(success),
        "drop": bool(drop),
    }


def classify_capture_and_future(
    metrics: dict,
    *,
    lost_after: bool,
    truncated: bool,
    handoff: bool,
    success: bool,
    drop: bool = False,
) -> dict[str, Any]:
    cap = capture_geometry_flags(metrics)
    fut = future_outcome_labels(
        lost_after=lost_after,
        truncated=truncated,
        handoff=handoff,
        success=success,
        drop=drop,
    )
    return {**cap, **fut}


# Backward-compatible wrapper used by older call sites (deprecated)
def classify_failure_pattern(
    *,
    metrics: dict,
    lost_after: bool,
    truncated: bool,
    handoff: bool,
    success: bool,
) -> str:
    """Deprecated: returns primary *capture* label (not future outcome)."""
    return str(capture_geometry_flags(metrics)["primary_capture_label"])
