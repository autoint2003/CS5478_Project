"""Unit and small integration tests for Adaptive Independent-G GRPO."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from training.adaptive_independent_g import (
    assert_no_train_valid_overlap,
    independent_g_decision,
    load_merged_rows,
    schedule_source_counts,
    source_aware_schedule,
    stop_bucket,
    unique_ic_shuffle_schedule,
)
from training.adaptive_grpo import apply_group_update, explore_ic_independent_g
from training.k47_codebook import group_advantage
from training.k48_config import ROOT


def _state(done, success=0):
    return {"rollouts_done": done, "success_count": success}


class IndependentGDecisionTests(unittest.TestCase):
    def test_g8_success_stops(self):
        d = independent_g_decision(_state(8, success=1))
        self.assertEqual(d["action"], "update")
        self.assertEqual(d["final_g"], 8)
        self.assertEqual(stop_bucket(8, True), "G8_stop")

    def test_g8_fail_expands_to_16(self):
        d = independent_g_decision(_state(8, success=0))
        self.assertEqual(d, {"action": "sample", "stage": "B", "add": 8, "target_g": 16})

    def test_g16_success_stops_at_16(self):
        d = independent_g_decision(_state(16, success=2))
        self.assertEqual(d["action"], "update")
        self.assertEqual(d["final_g"], 16)
        self.assertEqual(stop_bucket(16, True), "G16_stop")

    def test_g16_fail_expands_to_32(self):
        d = independent_g_decision(_state(16, success=0))
        self.assertEqual(d["add"], 16)
        self.assertEqual(d["target_g"], 32)

    def test_g32_success_stops_at_32(self):
        d = independent_g_decision(_state(32, success=1))
        self.assertEqual(d["final_g"], 32)
        self.assertEqual(stop_bucket(32, True), "G32_stop")

    def test_g32_fail_expands_to_64(self):
        d = independent_g_decision(_state(32, success=0))
        self.assertEqual(d["add"], 32)
        self.assertEqual(d["target_g"], 64)

    def test_g64_updates_without_success(self):
        d = independent_g_decision(_state(64, success=0))
        self.assertEqual(d["action"], "update")
        self.assertEqual(d["final_g"], 64)
        self.assertEqual(stop_bucket(64, False), "G64_no_success")

    def test_g64_success_bucket(self):
        self.assertEqual(stop_bucket(64, True), "G64_success")


class AdvantageAndLossTests(unittest.TestCase):
    def test_zero_variance_no_nan(self):
        adv, skipped = group_advantage([1.0, 1.0, 1.0], [False, False, False])
        self.assertTrue(skipped)
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertTrue(np.allclose(adv, 0.0))

    def test_g64_variance_finite_advantage(self):
        returns = [float(i % 5) for i in range(64)]
        adv, skipped = group_advantage(returns, [False] * 64)
        self.assertFalse(skipped)
        self.assertEqual(len(adv), 64)
        self.assertTrue(np.all(np.isfinite(adv)))

    def test_loss_equal_trajectory_weight_across_g(self):
        """G8 and G64 groups both mean-reduce over trajectories (not tokens×G)."""

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def forward(self, *args, **kwargs):
                raise RuntimeError("unused")

        def fake_step_logprob(model, obs, prev, token, pre):
            return model.w * 0.0 + (-1.0)

        def fake_kl(model, ref, obs, prev):
            return model.w * 0.0 + 0.01

        def member(ret, n_actions=3):
            return {
                "origin": "independent",
                "return": ret,
                "truncated": False,
                "label": "TABLE_DROP",
                "records": [
                    {
                        "obs": np.zeros((1, 39), np.float32),
                        "prev": np.zeros((1, 8), np.float32),
                        "token": 0,
                        "pre": np.zeros(7, np.float32),
                        "logp": -1.0,
                    }
                    for _ in range(n_actions)
                ],
            }

        cfg = {"clip": 0.2, "kl_coef": 0.05}
        device = torch.device("cpu")
        with patch("training.adaptive_grpo.step_logprob", fake_step_logprob), patch(
            "training.adaptive_grpo.categorical_kl", fake_kl
        ):
            for g, returns in ((8, list(range(8))), (64, list(range(64)))):
                model = Tiny()
                ref = Tiny()
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                members = [member(float(r)) for r in returns]
                stats = apply_group_update(model, ref, members, cfg, opt, device)
                self.assertTrue(stats["finite"])
                self.assertFalse(stats["skipped"])
                self.assertEqual(stats["group"], g)
                self.assertTrue(np.isfinite(stats["loss"]))
                self.assertTrue(np.isfinite(stats["grad_norm"]))


class ExploreIntegrationTests(unittest.TestCase):
    def test_explore_stops_at_g8_and_g16_counts(self):
        calls = {"n": 0}

        def fake_sample(env, model, book, blob, device, limit_s, reward_version, seed, offset, count, keep_captures):
            out = []
            for i in range(count):
                calls["n"] += 1
                success = False
                # First wave (offset 0): all fail. Second wave: first of eight succeeds.
                if offset >= 100 and i == 0:
                    success = True
                out.append({
                    "origin": "independent",
                    "records": [],
                    "return": 1.0 if success else 0.0,
                    "truncated": False,
                    "label": "SUCCESS_12S" if success else "TABLE_DROP",
                    "success": success,
                    "handoff_t": 0.1 if success else None,
                    "recovery_physics": 10,
                    "nominal_elapsed": 0.0,
                })
            return out

        with patch("training.adaptive_grpo._sample_group", fake_sample):
            log, group = explore_ic_independent_g(
                None, None, None, {}, torch.device("cpu"), 4.0, "v2", 1, "ic0",
            )
        self.assertEqual(log["final_g"], 16)
        self.assertEqual(log["stop_bucket"], "G16_stop")
        self.assertEqual(len(group), 16)
        self.assertEqual(calls["n"], 16)

    def test_explore_g8_stop(self):
        def fake_sample(env, model, book, blob, device, limit_s, reward_version, seed, offset, count, keep_captures):
            return [{
                "origin": "independent",
                "records": [],
                "return": 1.0 if i == 0 else 0.0,
                "truncated": False,
                "label": "SUCCESS_12S" if i == 0 else "LOST_GRASP",
                "success": i == 0,
                "handoff_t": 0.2 if i == 0 else None,
                "recovery_physics": 5,
                "nominal_elapsed": 1.0 if i == 0 else 0.0,
            } for i in range(count)]

        with patch("training.adaptive_grpo._sample_group", fake_sample):
            log, group = explore_ic_independent_g(
                None, None, None, {}, torch.device("cpu"), 4.0, "v2", 1, "ic1",
            )
        self.assertEqual(len(group), 8)
        self.assertEqual(log["stop_bucket"], "G8_stop")

    def test_explore_g64_no_success_still_returns_group(self):
        def fake_sample(env, model, book, blob, device, limit_s, reward_version, seed, offset, count, keep_captures):
            return [{
                "origin": "independent",
                "records": [{"obs": 0, "prev": 0, "token": 0, "pre": 0, "logp": 0.0}],
                "return": float(i % 3),
                "truncated": False,
                "label": "TABLE_DROP",
                "success": False,
                "handoff_t": None,
                "recovery_physics": 3,
                "nominal_elapsed": 0.0,
            } for i in range(count)]

        with patch("training.adaptive_grpo._sample_group", fake_sample):
            log, group = explore_ic_independent_g(
                None, None, None, {}, torch.device("cpu"), 4.0, "v2", 1, "ic2",
            )
        self.assertEqual(len(group), 64)
        self.assertEqual(log["stop_bucket"], "G64_no_success")
        self.assertFalse(log["found_success"])


class DatasetMergeTests(unittest.TestCase):
    def test_merged_manifests_exist_and_disjoint(self):
        base = ROOT / "results" / "grpo" / "adaptive_g" / "datasets"
        train_path = base / "merged_train.jsonl"
        valid_path = base / "merged_validation.jsonl"
        if not train_path.is_file():
            self.skipTest("run scripts/merge_grpo_ic_datasets.py first")
        train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid = [json.loads(line) for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert_no_train_valid_overlap(train, valid)
        self.assertGreater(len(train), 100)
        self.assertGreaterEqual(len(valid), 17)
        # Original 17 retained
        valid_ids = {row["id"] for row in valid if row.get("source_dataset") == "heuristic"}
        self.assertEqual(len(valid_ids), 17)

    def test_source_aware_schedule_roughly_balanced(self):
        rows = []
        for i in range(10):
            rows.append({
                "id": f"h{i}", "snapshot": f"h{i}.pkl",
                "categories": ["gravity_assisted_in_hand_sliding"],
                "source_dataset": "heuristic",
            })
        for i in range(3):
            rows.append({
                "id": f"d{i}", "snapshot": f"d{i}.pkl",
                "categories": ["vertical_recatch"],
                "source_dataset": "hard",
            })
        schedule = source_aware_schedule(rows, hard_fraction=0.5)
        hard = sum(1 for r in schedule if r["source_dataset"] == "hard")
        self.assertGreaterEqual(hard / len(schedule), 0.4)
        self.assertLessEqual(hard / len(schedule), 0.6)
        # Truncated prefixes must not collapse to hard-only (pilot failure mode).
        prefix = schedule[:10]
        hard_prefix = sum(1 for r in prefix if r["source_dataset"] == "hard")
        self.assertGreaterEqual(hard_prefix, 3)
        self.assertLessEqual(hard_prefix, 7)


class UniqueShuffleScheduleTests(unittest.TestCase):
    def _toy_rows(self):
        rows = []
        for i in range(10):
            rows.append({
                "id": f"h{i}", "snapshot": f"h{i}.pkl",
                "categories": ["gravity_assisted_in_hand_sliding"],
                "source_dataset": "heuristic",
            })
        for i in range(3):
            rows.append({
                "id": f"d{i}", "snapshot": f"d{i}.pkl",
                "categories": ["vertical_recatch"],
                "source_dataset": "hard",
            })
        return rows

    def test_covers_each_unique_ic_once(self):
        rows = self._toy_rows()
        schedule = unique_ic_shuffle_schedule(rows, np.random.default_rng(1))
        self.assertEqual(len(schedule), 13)
        ids = [row["id"] for row in schedule]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {row["id"] for row in rows})
        hard = [row for row in schedule if row["source_dataset"] == "hard"]
        self.assertEqual(len(hard), 3)

    def test_natural_source_ratio(self):
        rows = self._toy_rows()
        schedule = unique_ic_shuffle_schedule(rows, np.random.default_rng(2))
        counts = schedule_source_counts(schedule)
        self.assertAlmostEqual(counts["heuristic_fraction"], 10 / 13)
        self.assertAlmostEqual(counts["hard_fraction"], 3 / 13)

    def test_reproducible_with_fixed_seed(self):
        rows = self._toy_rows()
        a = [row["id"] for row in unique_ic_shuffle_schedule(rows, np.random.default_rng(42))]
        b = [row["id"] for row in unique_ic_shuffle_schedule(rows, np.random.default_rng(42))]
        c = [row["id"] for row in unique_ic_shuffle_schedule(rows, np.random.default_rng(43))]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_different_epochs_can_differ(self):
        rows = self._toy_rows()
        e1 = [row["id"] for row in unique_ic_shuffle_schedule(rows, np.random.default_rng(1 + 7919 * 1))]
        e2 = [row["id"] for row in unique_ic_shuffle_schedule(rows, np.random.default_rng(1 + 7919 * 2))]
        self.assertEqual(sorted(e1), sorted(e2))
        self.assertNotEqual(e1, e2)

    def test_merged_train_natural_coverage(self):
        path = ROOT / "results/grpo/adaptive_g/datasets/merged_train.jsonl"
        if not path.is_file():
            self.skipTest("merged train missing")
        rows = load_merged_rows(path)
        schedule = unique_ic_shuffle_schedule(rows, np.random.default_rng(1))
        self.assertEqual(len(schedule), len(rows))
        counts = schedule_source_counts(schedule)
        self.assertEqual(counts["unique_heuristic"], counts["heuristic"])
        self.assertEqual(counts["unique_hard"], counts["hard"])
        self.assertAlmostEqual(counts["hard_fraction"], counts["hard"] / len(schedule))
        # Natural ratio near 28/127 if pool matches expectations.
        if len(rows) == 127:
            self.assertEqual(counts["heuristic"], 99)
            self.assertEqual(counts["hard"], 28)

    def test_train_valid_disjoint_unchanged(self):
        train_path = ROOT / "results/grpo/adaptive_g/datasets/merged_train.jsonl"
        valid_path = ROOT / "results/grpo/adaptive_g/datasets/merged_validation.jsonl"
        if not train_path.is_file():
            self.skipTest("merged manifests missing")
        train = load_merged_rows(train_path)
        valid = load_merged_rows(valid_path)
        assert_no_train_valid_overlap(train, valid)
        schedule = unique_ic_shuffle_schedule(train, np.random.default_rng(7))
        valid_snaps = {row["snapshot"] for row in valid}
        self.assertTrue(all(row["snapshot"] not in valid_snaps for row in schedule))

    def test_adaptive_g_ladder_unaffected(self):
        d = independent_g_decision({"rollouts_done": 8, "success_count": 0})
        self.assertEqual(d["target_g"], 16)
        d = independent_g_decision({"rollouts_done": 64, "success_count": 0})
        self.assertEqual(d["stop_reason"], "G64_no_success")


class HandoffLossContractTests(unittest.TestCase):
    def test_handoff_uses_token_only_logprob_path(self):
        """HANDOFF must go through step_logprob (token-only); no residual decode."""
        from models.k48_checkpoint import HANDOFF

        seen = {"handoff": False}

        def fake_step_logprob(model, obs, prev, token, pre):
            if int(token) == HANDOFF:
                seen["handoff"] = True
            return model.w * 0.0 + (-0.5)

        def fake_kl(model, ref, obs, prev):
            return model.w * 0.0 + 0.01

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

        members = [{
            "origin": "independent",
            "return": 1.0,
            "truncated": False,
            "label": "SUCCESS_12S",
            "records": [{
                "obs": np.zeros((1, 39), np.float32),
                "prev": np.zeros((1, 8), np.float32),
                "token": HANDOFF,
                "pre": np.zeros(7, np.float32),
                "logp": -0.5,
            }],
        }, {
            "origin": "independent",
            "return": 0.0,
            "truncated": False,
            "label": "TABLE_DROP",
            "records": [{
                "obs": np.zeros((1, 39), np.float32),
                "prev": np.zeros((1, 8), np.float32),
                "token": 0,
                "pre": np.zeros(7, np.float32),
                "logp": -1.0,
            }],
        }]
        with patch("training.adaptive_grpo.step_logprob", fake_step_logprob), patch(
            "training.adaptive_grpo.categorical_kl", fake_kl
        ):
            model = Tiny()
            ref = Tiny()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            stats = apply_group_update(model, ref, members, {"clip": 0.2, "kl_coef": 0.05}, opt, torch.device("cpu"))
        self.assertTrue(seen["handoff"])
        self.assertTrue(stats["finite"])


if __name__ == "__main__":
    unittest.main()
