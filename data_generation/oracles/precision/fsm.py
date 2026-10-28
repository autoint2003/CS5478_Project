"""Explicit Precision finite-state machine (P9).

Replaces the rigid dθ→dx→dz priority with case-based ordering derived from
verified AXIS_V1 operating ranges (ex_tol / ez_tol), not ad-hoc constants.

States: ASSESS, CORRECT_DX, CORRECT_DZ, CORRECT_DTHETA,
RECONFIGURE_A/B/C, SETTLE, FINAL_VERIFY, HANDOFF, FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.state import PrecisionConfig


class PrecisionState(str, Enum):
    ASSESS = "ASSESS"
    CORRECT_DX = "CORRECT_DX"
    CORRECT_DZ = "CORRECT_DZ"
    CORRECT_DTHETA = "CORRECT_DTHETA"
    RECONFIGURE_A = "RECONFIGURE_A"
    RECONFIGURE_B = "RECONFIGURE_B"
    RECONFIGURE_C = "RECONFIGURE_C"
    SETTLE = "SETTLE"
    FINAL_VERIFY = "FINAL_VERIFY"
    HANDOFF = "HANDOFF"
    FAIL = "FAIL"


@dataclass
class FSMPlan:
    """Planned skill sequence from one ASSESS (may differ from executed)."""

    case: str
    planned_skills: list[str]
    dx_large: bool
    dz_large: bool
    dx_mm: float
    dz_mm: float
    dtheta_deg: float
    e_pos_mm: float
    ex_tol_mm: float
    ez_tol_mm: float
    theta_tol_deg: float
    note: str = ""


@dataclass
class FSMTrace:
    state: PrecisionState = PrecisionState.ASSESS
    planned: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    assess_history: list[dict] = field(default_factory=list)
    reconfigure_attempts: int = 0
    dtheta_attempts: int = 0
    stagnant: int = 0
    last_signature: tuple | None = None
    fail_reason: str | None = None


def _sig(errors: dict) -> tuple:
    e = np.asarray(errors.get("e_signed_mm") or [0, 0, 0], float)
    return (
        round(float(e[0]), 1),
        round(float(e[2]), 1),
        round(float(errors.get("dtheta_deg") or 0), 1),
        round(float(errors.get("e_pos_mm") or 0), 1),
    )


def classify_case(errors: dict, cfg: PrecisionConfig) -> FSMPlan:
    """Case 1–4 branching using verified AXIS_V1 / PrecisionConfig tolerances."""
    e = np.asarray(errors.get("e_signed_mm") or [0, 0, 0], float)
    dx = float(e[0])
    dz = float(e[2])
    th = float(errors.get("dtheta_deg") or 0)
    e_pos = float(errors.get("e_pos_mm") or 0)
    ex = float(cfg.ex_tol_mm)
    ez = float(cfg.ez_tol_mm)
    th_tol = float(cfg.theta_tol_deg)

    # Large = outside verified primitive operating band (AXIS_V1).
    dx_large = abs(dx) > ex
    dz_large = abs(dz) > ez
    # Residual Euclidean over working band also counts as needing position work
    # without inventing new thresholds: use existing working_pos_mm.
    if cfg.require_success_new_pose and e_pos > float(cfg.working_pos_mm):
        if abs(dz) >= abs(dx) and not dz_large and abs(dz) > 0.0:
            # Meaningful residual DZ (not a coarse 0.25*ez skip): any |dz| that
            # dominates e_pos overage while within AXIS still schedules correction.
            if abs(dz) >= abs(dx):
                dz_large = abs(dz) > 1.0  # numerical floor only (mm), not a new band
        if abs(dx) > abs(dz) and not dx_large and abs(dx) > 1.0:
            dx_large = True

    dtheta_optional = th > th_tol

    if dx_large and not dz_large:
        case = "CASE1_DX_LARGE_DZ_SMALL"
        planned = [PrecisionState.CORRECT_DX.value, PrecisionState.CORRECT_DZ.value]
        if dtheta_optional:
            planned.append(PrecisionState.CORRECT_DTHETA.value)
        planned.append(PrecisionState.FINAL_VERIFY.value)
        note = "DX-first then residual DZ check; optional dθ"
    elif dz_large and not dx_large:
        case = "CASE2_DZ_LARGE_DX_SMALL"
        planned = [PrecisionState.CORRECT_DZ.value, PrecisionState.CORRECT_DX.value]
        if dtheta_optional:
            planned.append(PrecisionState.CORRECT_DTHETA.value)
        planned.append(PrecisionState.FINAL_VERIFY.value)
        note = "DZ-first then residual DX check; optional dθ"
    elif dx_large and dz_large:
        case = "CASE3_BOTH_LARGE"
        # Prefer reconfiguration; concrete A/B/C chosen by caller preconditions.
        planned = [
            PrecisionState.RECONFIGURE_C.value,
            PrecisionState.RECONFIGURE_B.value,
            PrecisionState.RECONFIGURE_A.value,
            PrecisionState.ASSESS.value,
        ]
        note = "Both residuals outside AXIS bands → Stage-1 reconfigure then re-ASSESS"
    else:
        case = "CASE4_BOTH_SMALL"
        planned = []
        if dtheta_optional:
            planned.append(PrecisionState.CORRECT_DTHETA.value)
        planned.append(PrecisionState.FINAL_VERIFY.value)
        note = "Position within AXIS bands; optional bounded dθ then verify"

    return FSMPlan(
        case=case,
        planned_skills=planned,
        dx_large=dx_large,
        dz_large=dz_large,
        dx_mm=dx,
        dz_mm=dz,
        dtheta_deg=th,
        e_pos_mm=e_pos,
        ex_tol_mm=ex,
        ez_tol_mm=ez,
        theta_tol_deg=th_tol,
        note=note,
    )


def next_skill_from_plan(
    plan: FSMPlan,
    *,
    done_dx: bool,
    done_dz: bool,
    done_dtheta: bool,
    reconfigure_done: bool,
    need: dict,
    max_dtheta_attempts: int,
    dtheta_attempts: int,
) -> PrecisionState | None:
    """Pick the next executable skill respecting FSM case order and need flags.

    Does not silently substitute DX-first when DZ-first was selected.
    """
    if plan.case == "CASE3_BOTH_LARGE" and not reconfigure_done:
        # Caller tries applicable reconfigure; if none applicable, fall through.
        return PrecisionState.RECONFIGURE_C  # sentinel; executor picks A/B/C by precondition

    # Walk planned position skills in order
    for sk in plan.planned_skills:
        if sk == PrecisionState.CORRECT_DX.value:
            if need.get("dx_needed") and not done_dx:
                return PrecisionState.CORRECT_DX
        elif sk == PrecisionState.CORRECT_DZ.value:
            if need.get("dz_needed") and not done_dz:
                return PrecisionState.CORRECT_DZ
        elif sk == PrecisionState.CORRECT_DTHETA.value:
            if (
                need.get("dtheta_needed")
                and not done_dtheta
                and dtheta_attempts < max_dtheta_attempts
            ):
                return PrecisionState.CORRECT_DTHETA
        elif sk == PrecisionState.FINAL_VERIFY.value:
            return PrecisionState.FINAL_VERIFY
        elif sk == PrecisionState.ASSESS.value:
            return PrecisionState.ASSESS

    # Coupling: after both position skills, residual may reappear
    if need.get("dx_needed") and done_dx and done_dz:
        return PrecisionState.CORRECT_DX  # one coupling retry armed by caller clearing done_dx
    if need.get("dz_needed") and done_dx and done_dz:
        return PrecisionState.CORRECT_DZ

    if need.get("ready_for_direct_handoff_assessment") or (
        not need.get("dx_needed") and not need.get("dz_needed") and not need.get("dtheta_needed")
    ):
        return PrecisionState.FINAL_VERIFY

    return None


def assess_and_plan(errors: dict, cfg: PrecisionConfig) -> tuple[dict, FSMPlan]:
    need = assess_need(errors, cfg)
    plan = classify_case(errors, cfg)
    return need, plan


def select_reconfigure_family(plan_id: str, contact_hint: dict | None = None) -> list[PrecisionState]:
    """Order reconfigure attempts by plan family and optional contact geometry."""
    pid = str(plan_id or "B").upper()[:1]
    order = {
        "A": [PrecisionState.RECONFIGURE_A, PrecisionState.RECONFIGURE_B, PrecisionState.RECONFIGURE_C],
        "B": [PrecisionState.RECONFIGURE_B, PrecisionState.RECONFIGURE_C, PrecisionState.RECONFIGURE_A],
        "C": [PrecisionState.RECONFIGURE_C, PrecisionState.RECONFIGURE_B, PrecisionState.RECONFIGURE_A],
        "D": [PrecisionState.RECONFIGURE_C, PrecisionState.RECONFIGURE_B, PrecisionState.RECONFIGURE_A],
    }.get(pid, [PrecisionState.RECONFIGURE_C, PrecisionState.RECONFIGURE_B, PrecisionState.RECONFIGURE_A])
    hint = contact_hint or {}
    # Prefer C when opposing sidewall mid-body is available for release/recatch.
    if hint.get("prefer_c"):
        order = [PrecisionState.RECONFIGURE_C] + [s for s in order if s != PrecisionState.RECONFIGURE_C]
    return order


CLASS_ALIASES = {
    "A1": "A_STABLE_DIRECT_HANDOFF",
    "A2": "A_STABLE_PRECISION",
    "B1": "B_STABLE_DIRECT_HANDOFF",
    "B2": "B_STABLE_PRECISION",
    "C1": "C_RECATCH_DIRECT_HANDOFF",
    "C2": "C_RECATCH_PRECISION",
    "D2": "D_PRECISION",
}

BRANCH_TO_CLASS = {v: k for k, v in CLASS_ALIASES.items()}
