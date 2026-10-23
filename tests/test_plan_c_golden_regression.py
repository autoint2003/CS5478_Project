"""Golden Plan C experts must remain 3/3 SUCCESS_NEW after consolidation."""

from __future__ import annotations

import pytest

from evaluation.plan_c_golden_replay import replay_plan_c


@pytest.mark.slow
@pytest.mark.parametrize("root_id", ["gbi_010", "gbi_022"])
def test_plan_c_golden_exact_replay(root_id):
    payload, _, _ = replay_plan_c(root_id, n_replay=3)
    assert payload["exact_replay"]["n_pass"] == 3
    assert payload["SUCCESS_NEW"] is True
