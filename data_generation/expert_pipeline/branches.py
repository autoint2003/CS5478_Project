"""The six expert-generation branches. Identifiers are stable dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Branch:
    branch_id: str
    stage1: str | None
    continuation: str
    description: str


BRANCHES: dict[str, Branch] = {
    "A_STABLE_PRECISION": Branch(
        "A_STABLE_PRECISION",
        "plan_a_stable_grasp",
        "precision",
        "Plan A stable grasp, then Precision Oracle.",
    ),
    "B_STABLE_DIRECT_HANDOFF": Branch(
        "B_STABLE_DIRECT_HANDOFF",
        "plan_b_stable_grasp",
        "direct_handoff",
        "Plan B stable grasp, then direct nominal HANDOFF.",
    ),
    "B_STABLE_PRECISION": Branch(
        "B_STABLE_PRECISION",
        "plan_b_stable_grasp",
        "precision",
        "Plan B stable grasp, then Precision Oracle.",
    ),
    "C_RECATCH_DIRECT_HANDOFF": Branch(
        "C_RECATCH_DIRECT_HANDOFF",
        "plan_c_recatch",
        "direct_handoff",
        "Plan C recatch, then direct nominal HANDOFF.",
    ),
    "C_RECATCH_PRECISION": Branch(
        "C_RECATCH_PRECISION",
        "plan_c_recatch",
        "precision",
        "Plan C recatch, then Precision Oracle.",
    ),
    "D_PRECISION": Branch(
        "D_PRECISION",
        None,
        "precision",
        "Precision Oracle from a supplied stable state, with no Stage-1 plan.",
    ),
}

STAGE1_MODULES = {
    "plan_a_stable_grasp": "data_generation.oracles.plan_a",
    "plan_b_stable_grasp": "data_generation.oracles.plan_b",
    "plan_c_recatch": "data_generation.oracles.plan_c",
}


def select_branch(branch_id: str) -> Branch:
    key = str(branch_id).strip()
    if key not in BRANCHES:
        known = ", ".join(BRANCHES)
        raise KeyError(f"unknown expert branch {branch_id!r}; expected one of: {known}")
    return BRANCHES[key]
