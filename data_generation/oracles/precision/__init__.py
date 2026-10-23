"""Shared post-stabilization precision-manipulation Oracle (offline).

Plan A and Plan B only establish a stable grasp. Precision stages live here and
are independent of which stabilizer produced the grasp.
"""

from data_generation.oracles.precision.stable_grasp import (
    STABLE_THRESHOLDS,
    evaluate_stable_grasp,
    hold_and_confirm_stable,
    is_stable_grasp,
)
from data_generation.oracles.precision.primitives import (
    MAX_DELTA_P_REL_MM,
    audit_grasp,
    classify_pivot_motion,
    classify_stable_needs,
    pivot_segment_metrics,
    primitive_axial_tail_slide,
    primitive_contact_pivot,
    primitive_final_axial_slide,
    primitive_gravity_reorient,
    primitive_stabilize_handoff,
    relative_pose,
)
from data_generation.oracles.precision.force_slip_control import (
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    contact_forces,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import (
    desired_correction_direction,
    plan_orientation,
    plan_restore_orientation,
)
from data_generation.oracles.precision.axis_aware import (
    axial_desire_direction,
    decide_precision,
    lateral_desire_direction,
)
from data_generation.oracles.precision.sign_aware_axial import (
    opening_axis_hand,
    select_axial_mode,
    signed_opening_error,
)
from data_generation.oracles.precision.unified_precision_oracle import (
    PrecisionConfig,
    PrecisionResult,
    SlipGoal,
    assess_need,
    entry_gate,
    run_goal_aware_axial_slip,
    run_precision,
    run_staged_axial_slip,
)

__all__ = [
    "STABLE_THRESHOLDS",
    "MAX_DELTA_P_REL_MM",
    "evaluate_stable_grasp",
    "hold_and_confirm_stable",
    "is_stable_grasp",
    "audit_grasp",
    "relative_pose",
    "classify_pivot_motion",
    "pivot_segment_metrics",
    "classify_stable_needs",
    "primitive_axial_tail_slide",
    "primitive_contact_pivot",
    "primitive_gravity_reorient",
    "primitive_final_axial_slide",
    "primitive_stabilize_handoff",
    "DualLoopController",
    "ForcePController",
    "SlipOuterLoop",
    "contact_forces",
    "sample_state",
    "desired_correction_direction",
    "plan_orientation",
    "plan_restore_orientation",
    "axial_desire_direction",
    "decide_precision",
    "lateral_desire_direction",
    "opening_axis_hand",
    "select_axial_mode",
    "signed_opening_error",
    "PrecisionConfig",
    "PrecisionResult",
    "SlipGoal",
    "assess_need",
    "entry_gate",
    "run_goal_aware_axial_slip",
    "run_precision",
    "run_staged_axial_slip",
]
