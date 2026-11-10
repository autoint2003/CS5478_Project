"""Fail-closed IC resolution tests for P107 (no physics / controller changes)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.precision.p107_action_aware_multiseed_validation.run_p107 as p107  # noqa: E402


def _write_ep(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"ep": {"traj_id": marker, "marker": marker}}))


def test_two_ids_sharing_prefix_resolve_exactly(tmp_path, monkeypatch):
    """IDs sharing a gbi_XXX prefix must not cross-resolve."""
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    ics = tmp_path / "results/data_generation/stage2_abc_final/ics"
    a = "plan_c__gbi_058__aaaaaaaaaaaaaaaa"
    b = "plan_c__gbi_058__bbbbbbbbbbbbbbbb"
    _write_ep(p107.P97_NEW / f"{a}.pkl", "A")
    _write_ep(p107.P97_NEW / f"{b}.pkl", "B")
    # distractor that would win under old prefix-first glob
    _write_ep(ics / "C__gbi_058__v2__cccccccccccccccc.pkl", "DISTRACTOR")

    starts: dict = {}
    st_a = p107.resolve_c_start_robust(a, p107.phys_root(a), starts)
    st_b = p107.resolve_c_start_robust(b, p107.phys_root(b), starts)
    assert st_a is not None and st_a.ep["marker"] == "A"
    assert st_b is not None and st_b.ep["marker"] == "B"


def test_missing_exact_id_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    ics = tmp_path / "results/data_generation/stage2_abc_final/ics"
    _write_ep(ics / "C__gbi_092__v2__deadbeefdeadbeef.pkl", "WRONG")
    tid = "plan_c__gbi_092_78b462d67f"
    st = p107.resolve_c_start_robust(tid, p107.phys_root(tid), {})
    assert st is None


def test_multiple_exact_candidate_snapshots_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    tid = "plan_c__gbi_114__5c9c4e95f83b"
    _write_ep(p107.P97_NEW / f"{tid}.pkl", "P97")
    stage = tmp_path / "results/data_generation/stage2_abc_final/ics" / f"{tid}.pkl"
    _write_ep(stage, "STAGE2")
    with pytest.raises(p107.AmbiguousICResolutionError):
        p107.resolve_c_start_robust(tid, p107.phys_root(tid), {})


def test_manifest_ordering_does_not_change_exact_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    tid = "plan_c__gbi_085__d5ca460d970b"
    # Lexicographically earlier distractor must be ignored (no prefix match).
    ics = tmp_path / "results/data_generation/stage2_abc_final/ics"
    _write_ep(ics / "C__gbi_085__aaa_first.pkl", "FIRST")
    _write_ep(p107.P97_NEW / f"{tid}.pkl", "EXACT")
    st = p107.resolve_c_start_robust(tid, p107.phys_root(tid), {})
    assert st is not None and st.ep["marker"] == "EXACT"


def test_duplicate_canonical_ids_reuse_same_start(tmp_path, monkeypatch):
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    tid = "plan_c__gbi_064__c4cbda2b93ba"
    _write_ep(p107.P97_NEW / f"{tid}.pkl", "ONE")
    starts: dict = {}
    st1 = p107.resolve_c_start_robust(tid, p107.phys_root(tid), starts)
    st2 = p107.resolve_c_start_robust(tid, p107.phys_root(tid), starts)
    assert st1 is st2
    assert st1.ep["marker"] == "ONE"


def test_train_eval_truncated_collision_still_exact_only(tmp_path, monkeypatch):
    """Train alias gbi_058__834e5244e9 vs eval plan_c__gbi_058__834e5244e937.

    Truncated phys_root collides, but resolution keys on full traj_id.
    """
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    train_alias = "gbi_058__834e5244e9"
    eval_tid = "plan_c__gbi_058__834e5244e937"
    assert p107.phys_root(train_alias) == p107.phys_root(eval_tid) == "gbi_058"
    _write_ep(p107.P97_NEW / f"{eval_tid}.pkl", "EVAL")
    # Train-style file name must not satisfy eval traj_id exact match.
    _write_ep(p107.P97_NEW / f"{train_alias}.pkl", "TRAIN")
    st_eval = p107.resolve_c_start_robust(eval_tid, p107.phys_root(eval_tid), {})
    st_train = p107.resolve_c_start_robust(train_alias, p107.phys_root(train_alias), {})
    assert st_eval is not None and st_eval.ep["marker"] == "EVAL"
    assert st_train is not None and st_train.ep["marker"] == "TRAIN"


def test_never_silently_selects_first_prefix_pkl(tmp_path, monkeypatch):
    monkeypatch.setattr(p107, "P97_NEW", tmp_path / "new_ics")
    monkeypatch.setattr(p107, "ROOT", tmp_path)
    ics = tmp_path / "results/data_generation/stage2_abc_final/ics"
    _write_ep(ics / "C__gbi_092__v2__1111111111111111.pkl", "V2")
    _write_ep(ics / "PILOT_C__gbi_092__2222222222222222.pkl", "PILOT")
    tid = "plan_c__gbi_092_78b462d67f"
    assert p107.resolve_c_start_robust(tid, p107.phys_root(tid), {}) is None
