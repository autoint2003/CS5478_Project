"""Task-success and scene-category rules for recovery data.

Task success is nominal failure, a legal K48 action, and SUCCESS_12S.
Slip size and post-impact tilt are diagnostics. They do not accept or reject
a demonstration.
"""
from __future__ import annotations

DIAGNOSTIC_REJECTS = frozenset({
    "no_in_hand_slip",
    "weak_net_correction",
    "release_then_regrasp",
    "gravity_inconsistent",
})


def task_success(nominal_succeeded: bool, k48_legal: bool, success_12s: bool) -> bool:
    """The only acceptance rule. Mechanism evidence is not an input."""
    return (not nominal_succeeded) and bool(k48_legal) and bool(success_12s)


def observed_mechanism(diagnostic_status: str, release_then_regrasp: bool = False) -> str:
    """Label what the trace showed. This does not decide task success."""
    if release_then_regrasp or diagnostic_status == "release_then_regrasp":
        return "undetermined"
    if diagnostic_status == "confirmed":
        return "in_hand_slip"
    if diagnostic_status in ("no_in_hand_slip", "weak_net_correction", "gravity_inconsistent"):
        return "stabilization_without_slip"
    return "undetermined"


def blocks_task_success(diagnostic_status: str) -> bool:
    """Historical mistake: these labels were used as rejection reasons."""
    return False


def scene_category(grasp_scene: str, post_impact_tilt_deg: float | None = None) -> str | None:
    """Category follows the original grasp, not a later tilt."""
    del post_impact_tilt_deg
    if grasp_scene == "horizontal_cylinder":
        return "horizontal_recatch"
    if grasp_scene == "vertical_cylinder":
        return "vertical_recatch"
    return None


def admit_horizontal_search(grasp_scene: str, post_impact_tilt_deg: float) -> bool:
    """A tipped cylinder stays in the horizontal-grasp search."""
    del post_impact_tilt_deg
    return grasp_scene == "horizontal_cylinder"


def release_and_regrip(n_left, n_right, min_off: int = 2, min_on: int = 2) -> bool:
    """Require a sustained both-off interval followed by sustained bilateral contact.

    One sample of lost contact is not a recatch.
    """
    off = [int(left) == 0 and int(right) == 0 for left, right in zip(n_left, n_right)]
    both = [int(left) > 0 and int(right) > 0 for left, right in zip(n_left, n_right)]
    index = 0
    count = len(off)
    while index < count:
        if not off[index]:
            index += 1
            continue
        end = index
        while end < count and off[end]:
            end += 1
        if end - index >= min_off:
            run = 0
            for cursor in range(end, count):
                if both[cursor]:
                    run += 1
                    if run >= min_on:
                        return True
                else:
                    run = 0
        index = end
    return False
