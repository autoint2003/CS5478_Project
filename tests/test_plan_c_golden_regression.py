"""Golden Plan C experts must remain 3/3 SUCCESS_NEW after consolidation."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "scripts" / "consolidation" / "freeze_unified_experts.py"


@pytest.mark.slow
@pytest.mark.parametrize("root_id", ["gbi_010", "gbi_022"])
def test_plan_c_golden_exact_replay(root_id):
    mod = runpy.run_path(str(FREEZE))
    payload, _, _ = mod["replay_plan_c"](root_id, n_replay=3)
    assert payload["exact_replay"]["n_pass"] == 3
    assert payload["SUCCESS_NEW"] is True
