"""Shared Precision Oracle types, constants, and result containers (v2.3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.residual_directions import (
    decide_precision,
    lateral_desire_direction,
)
from data_generation.oracles.precision.mechanics.open_axis import (
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
)
from data_generation.oracles.precision.mechanics.finger_surface import support_force_decomposition
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND
from data_generation.oracles.precision.mechanics.dual_loop import (
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    contact_forces,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import (
    _height_hold_v_hand,
    execute_orientation_plan,
    plan_orientation,
    score_gravity_alignment,
)
from data_generation.oracles.precision.corrections.contact_primitives import (
    _action_vw,
    _wrist,
    audit_grasp,
    pivot_segment_metrics,
    primitive_contact_pivot,
)
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from data_generation.oracles.precision.corrections.axial_sign import (
    fingertip_overlap_margin_mm,
    select_axial_mode,
)
from data_generation.oracles.precision.mechanics.grasp_stabilization import (
    STABLE_THRESHOLDS,
    hold_and_confirm_stable,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import (
    AXIS_V1_EX_MM,
    AXIS_V1_EY_MM,
    AXIS_V1_EZ_MM,
    AXIS_V1_THETA_DEG,
    evaluate_handoff_with_axis,
)
from evaluation.grasp_geometry import hand_R, measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG

ROOT = Path(__file__).resolve().parents[3]
PIVOT_SEARCH = ROOT / "results/diagnostics/position_preserving_pivot/search.json"

# Verified DualLoop gains (geometry / axis-aware diagnostics)
FORCE_KP, FORCE_DA = 0.10, 0.12
SLIP_KN = 900.0
V_SLIP = -0.001
FN_MIN, FN_MAX = 3.8, 11.0
FN_INIT, FN_BRAKE = 5.2, 14.0
FN_HOLD = 13.0
HOLD_STABLE_S = 0.15
Z_ABORT = 0.44
LATERAL_DRIFT_ABORT_MM = 4.0
LATERAL_ABS_ABORT_MM = 8.0
ROLL_ANGLE_DEG = 90.0  # verified endpoint-only / handoff_state_optimization
A0_SLIP = 0.55
# P3 sliding-roll candidates at horizontal (sign, angle_deg) — offline diagnostic
HORIZONTAL_SLIDE_ROLL_CANDIDATES = (
    (-1.0, 30.0),
    (-1.0, 45.0),
    (-1.0, 60.0),
    (-1.0, 75.0),
    (-1.0, 90.0),
    (1.0, 30.0),
    (1.0, 45.0),
    (1.0, 60.0),
    (1.0, 75.0),
    (1.0, 90.0),
)
G_LATERAL_MIN = 4.0
FINGERTIP_MARGIN_MIN_MM = 2.0

# Phase-specific failure classifications
FAIL_PREPARATION = "PREPARATION_FAILED"
FAIL_DZ_PRECONDITION = "DZ_UNSAFE_ENTRY"
FAIL_SUPPORT_ROLL = "SUPPORT_ROLL_FAILED"
FAIL_HORIZONTAL_TILT = "HORIZONTAL_TILT_FAILED"
FAIL_SLIDING_ROLL = "SLIDING_ROLL_FAILED"
FAIL_SLIP_CONTACT = "SLIP_CONTACT_LOSS"
FAIL_SLIP_WRONG_DIR = "SLIP_WRONG_DIRECTION"
FAIL_SLIP_TARGET = "SLIP_TARGET_NOT_REACHED"
FAIL_BRAKE = "BRAKE_FAILED"
FAIL_REGRIP = "REGRIP_FAILED"
FAIL_REVERSE_ROLL = "REVERSE_ROLL_FAILED"
FAIL_RETURN_TILT = "RETURN_TILT_FAILED"
FAIL_FINAL_UNROLL = "FINAL_UNROLL_FAILED"
FAIL_HANDOFF = "HANDOFF_FAILED"
# Mandatory pre-HANDOFF posture / grasp gate (fingertips vertically DOWN)
FAIL_POSTURE_DOWN_RESTORE = "POSTURE_DOWN_RESTORE_FAILED"
FAIL_POSTURE_DOWN_VERIFY = "POSTURE_DOWN_VERIFY_FAILED"
FAIL_STABLE_GRASP_PRE_HANDOFF = "STABLE_GRASP_PRE_HANDOFF_FAILED"

# Goal-driven slip / correction stop reasons (v2.2)
STOP_GOAL_REACHED = "GOAL_REACHED"
STOP_BRAKE_AND_REASSESS = "BRAKE_AND_REASSESS"
STOP_CONTACT_SAFETY = "CONTACT_SAFETY_ABORT"
STOP_WRONG_DIRECTION = "WRONG_DIRECTION"
STOP_NO_PROGRESS = "NO_PROGRESS"
STOP_MAX_SAFE_TRAVEL = "MAX_SAFE_TRAVEL"
STOP_TIME_BUDGET = "TIME_BUDGET"
STOP_HANDOFF_READY = "HANDOFF_READY"


@dataclass
class SlipGoal:
    """Goal-driven axial/lateral slip objective. Travel limits are safety bounds only."""

    axis: str  # "dx" | "dz"
    desired_direction: np.ndarray | list[float]  # hand-frame unit desire for COM motion
    max_safe_travel_mm: float = 12.0
    max_duration_s: float = 12.0
    # Local axis band (AXIS_V1 / working); Euclidean SUCCESS_NEW uses working_pos_mm
    axis_tol_mm: float = 12.0
    working_pos_mm: float = WORKING_POS_MM
    working_theta_deg: float = WORKING_THETA_DEG
    ey_tol_mm: float = AXIS_V1_EY_MM
    # Optional signed target for the controlled component (None = drive toward 0)
    target_signed_mm: float | None = 0.0
    min_progress_mm: float = 0.15
    brake_margin_mm: float = 0.8  # start braking slightly before axis band
    # When False (e.g. fingertips-up inward slip), do not trust live e_pos / HANDOFF_READY.
    use_euclidean_stop: bool = True


@dataclass
class PrecisionConfig:
    """Tunable offline Oracle options. Does not change SUCCESS_NEW definitions.

    Slip ``*_mm`` fields are **max safe travel** bounds (diagnostics / safety), not
    desired displacements. Goal-driven DualLoop stops on geometric target regions.
    """

    try_direct_handoff: bool = True
    evaluate_handoff: bool = True
    allow_horizontal_dx: bool = True
    # Flowchart default: support-transfer gravity-assisted dx (ends at S1).
    use_support_transfer_dx: bool = True
    # Legacy baselines: OFF by default. Must never silently substitute after ST failure.
    # Enable only for explicit A/B baseline experiments (logs legacy_fallback_explicit).
    use_verified_lateral_fallback: bool = False
    accept_partial_horizontal_dx: bool = True  # v2.2: keep useful partial +dx for dz follow-up
    # Commit horizontal only if HANDOFF or counterfactual axial from post-state succeeds.
    require_horizontal_recoverability: bool = True
    max_retries_per_primitive: int = 1
    max_correction_cycles: int = 4
    global_action_budget: int = 4000  # ~80 s @ 20 ms (support-transfer + reverse-UP return)
    pivot_tag: str | None = None  # override; else load verified tag / search
    ex_tol_mm: float = AXIS_V1_EX_MM
    ey_tol_mm: float = AXIS_V1_EY_MM
    ez_tol_mm: float = AXIS_V1_EZ_MM
    theta_tol_deg: float = AXIS_V1_THETA_DEG
    # Also require SUCCESS_NEW Euclidean readiness for skip decision
    require_success_new_pose: bool = True
    working_pos_mm: float = WORKING_POS_MM
    working_theta_deg: float = WORKING_THETA_DEG
    # Safety / diagnostic max travel (NOT desired slide distances)
    inward_slip_mm: float = 12.0
    outward_slip_mm: float = 12.0
    lateral_slip_mm: float = 14.0
    horizontal_slip_mm: float = 14.0
    # Verified lateral: stop in a tight |dx| band so follow-up axial remains recoverable.
    # Empirically |dx|≲3 mm after gravity-assist lateral preserves Plan-B inward contact;
    # larger overshoot (|dx|~4–5) often causes CONTACT_SAFETY_ABORT on inward DualLoop.
    lateral_axis_tol_mm: float = 3.0
    use_p1_p2_search: bool = False  # joint P1/P2 search for +dx horizontal prep
    p1_p2_winner: dict | None = None  # optional precomputed winner_params to skip re-search
    # Deprecated v2.1: ignored when goal-driven slip is active (kept for API compat).
    horizontal_legacy_total: bool = False
    use_goal_driven_slip: bool = True
    stabilize_handoff_n: int = 8
    book: Any = None  # optional codebook for pivot search fallback
    # Support-aware rotational rates (fraction of max wrist rate in tilt/roll helpers)
    rate_establish_s1: float = 0.16
    rate_protected_s1: float = 0.14  # S1-protected cruise (contact-aware profile may go lower)
    rate_friction_support: float = 0.16
    rate_restore_down: float = 0.18
    rate_tilt_cruise: float = 0.18  # fixed-axis UP/DOWN cruise under verified S1
    rate_tilt_peak: float = 0.24  # brief peak if margins healthy
    rate_tilt_floor: float = 0.06  # never go below this while rotating
    tilt_accel_s: float = 0.35  # accel / decel ramp duration
    use_tilt_speed_profile: bool = True  # accel–cruise–decel + contact guard
    dx_s1_roll_deg: float = 12.0  # modest DX-1 bias; large rolls break Plan-B horizontal
    dx_horizontal_rate_cap: float = 0.08  # DX-2 protected tilt cap (Plan-B verified; >0.10 leaves |ex| unsafe)
    restore_down_after_dtheta: bool = True
    # Physical safety for inward dz flip/slip — tighter than SUCCESS_NEW / AXIS_V1 ex=12 mm.
    # Large |ex| before UP unloads support mid-flip even when S1 looks valid.
    dz_safe_ex_mm: float = 6.0
    dz_safe_vrel: float = 0.025
    # Inward UP need not be perfectly vertical if gravity still drives axial slip.
    dz_up_tol_deg: float = 12.0
    dz_up_g_useful_min: float = 6.0  # m/s² along inward desire
    dz_return_pre_lift_s: float = 0.25


@dataclass
class PrecisionResult:
    status: str
    plan_id: str
    precision_skipped: bool
    precision_executed: bool
    gate_ok: bool
    phases: list[str] = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    errors_entry: dict = field(default_factory=dict)
    errors_exit: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    capture: Any = None
    ref: Any = None
    handoff: dict | None = None
    direct_handoff_probe: dict | None = None
    primitive_logs: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("capture", None)
        d.pop("ref", None)
        # actions may be large; keep count + optional thin summary elsewhere
        acts = d.get("actions") or []
        d["n_actions"] = len(acts)
        d["actions"] = None
        return d
