"""Branch selection and dataset-admission rules. No trajectory generation."""

from data_generation.expert_pipeline.branches import BRANCHES, select_branch
from data_generation.expert_pipeline.provenance import (
    DATASET_VERSION,
    ORACLE_IMPLEMENTATION,
    row_allowed_for_training,
    stamp_record,
)


def test_six_branch_ids():
    assert set(BRANCHES) == {
        "A_STABLE_PRECISION",
        "B_STABLE_DIRECT_HANDOFF",
        "B_STABLE_PRECISION",
        "C_RECATCH_DIRECT_HANDOFF",
        "C_RECATCH_PRECISION",
        "D_PRECISION",
    }
    assert select_branch("D_PRECISION").stage1 is None
    assert select_branch("B_STABLE_DIRECT_HANDOFF").continuation == "direct_handoff"
    assert select_branch("C_RECATCH_PRECISION").continuation == "precision"


def test_unknown_branch_rejected():
    try:
        select_branch("K49_RECOVERY")
    except KeyError:
        return
    raise AssertionError("unknown branch was accepted")


def test_legacy_precision_row_excluded():
    legacy = {
        "expert_id": "plan_c__old_precision",
        "branch_id": "C_RECATCH_PRECISION",
        "continuation": "precision",
        "physical_success": True,
        "SUCCESS_NEW": True,
        "oracle_implementation": "unified_precision_oracle_v22",
    }
    assert row_allowed_for_training(legacy) is False


def test_failed_v2_row_excluded():
    row = {
        "dataset_version": DATASET_VERSION,
        "branch_id": "D_PRECISION",
        "continuation": "precision",
        "physical_success": False,
        "oracle_implementation": ORACLE_IMPLEMENTATION,
        "git_commit": "abc",
    }
    assert row_allowed_for_training(row) is False


def test_stamped_precision_row_admitted():
    stamp = stamp_record(
        {
            "branch_id": "A_STABLE_PRECISION",
            "continuation": "precision",
            "stage1": "plan_a_stable_grasp",
            "physical_success": True,
            "source_ic_id": "ic-1",
        }
    )
    assert stamp["dataset_version"] == DATASET_VERSION
    assert stamp["output_namespace"] == "data/experts_v2"
    assert row_allowed_for_training(stamp) is True


def test_direct_handoff_stamp_does_not_require_precision_oracle():
    stamp = stamp_record(
        {
            "branch_id": "C_RECATCH_DIRECT_HANDOFF",
            "continuation": "direct_handoff",
            "stage1": "plan_c_recatch",
            "physical_success": True,
        }
    )
    assert stamp["oracle_implementation"] is None
    assert row_allowed_for_training(stamp) is True


def test_untagged_historical_success_is_not_mixed_in():
    assert row_allowed_for_training({"SUCCESS_NEW": True, "plan_family": "C"}) is False
