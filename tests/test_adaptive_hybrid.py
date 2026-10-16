"""Adaptive hybrid gates. Physics replay is the last test."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from training.adaptive_hybrid import (
    HORIZONTAL,
    SLIDING,
    VERTICAL,
    assert_train_valid_disjoint,
    balanced_schedule,
    exploration_decision,
    load_train_snapshots,
    loss_members,
    primary_category,
    sampling_frequency,
    select_branch_points,
)
from training.k48_config import ROOT


def _state(**kwargs):
    base = {
        "stage_a_success": 0,
        "stage_b_success": 0,
        "stage_b_done": False,
        "tree_uniform_success": 0,
        "tree_entropy_success": 0,
        "tree_done": False,
        "resample_success": 0,
        "resample_done": False,
    }
    base.update(kwargs)
    return base


class AdaptiveHybridTests(unittest.TestCase):
    def test_stage_a_success_stops(self):
        decision = exploration_decision(_state(stage_a_success=1))
        self.assertEqual(decision["action"], "update")
        self.assertEqual(decision["source"], "independent_stage_a")
        self.assertNotIn("tree", decision["source"])

    def test_stage_a_failure_requests_eight_more(self):
        decision = exploration_decision(_state())
        self.assertEqual(decision, {"action": "sample", "stage": "B"})

    def test_sixteen_failures_request_tree(self):
        decision = exploration_decision(_state(stage_b_done=True))
        self.assertEqual(decision, {"action": "sample", "stage": "C"})

    def test_completed_tree_updates_without_resample(self):
        failed = exploration_decision(_state(stage_b_done=True, tree_done=True))
        self.assertEqual(failed["action"], "update")
        self.assertEqual(failed["source"], "tree_augmented")
        self.assertEqual(failed["group"], "stage_c")
        succeeded = exploration_decision(_state(
            stage_b_done=True, tree_done=True, tree_uniform_success=1,
        ))
        self.assertEqual(succeeded["action"], "update")
        self.assertNotIn("resample", succeeded["source"])

    def test_uniform_and_entropy_differ(self):
        candidates = [{"entropy": 0.1 * index, "step": index} for index in range(12)]
        rng = np.random.default_rng(1)
        uniform = [item["step"] for item in select_branch_points(candidates, rng, 8, "uniform")]
        entropy = [item["step"] for item in select_branch_points(candidates, np.random.default_rng(1), 8, "entropy")]
        self.assertEqual(entropy, [11, 10, 9, 8, 7, 6, 5, 4])
        self.assertNotEqual(uniform, entropy)

    def test_loss_keeps_tree_failures_and_successes(self):
        members = [
            {"origin": "independent", "label": "TABLE_DROP"},
            {"origin": "tree", "label": "SUCCESS_12S"},
            {"origin": "tree", "label": "TABLE_DROP"},
        ]
        self.assertEqual(len(loss_members(members)), 3)
        with self.assertRaises(RuntimeError):
            loss_members([])
        with self.assertRaises(RuntimeError):
            loss_members([{"origin": "replay", "label": "SUCCESS_12S"}])

    def test_multi_oracle_snapshot_is_one_schedule_slot(self):
        rows = [
            {"snapshot": "a", "categories": [SLIDING, VERTICAL]},
            {"snapshot": "a", "categories": [SLIDING]},
        ]
        with self.assertRaises(RuntimeError):
            balanced_schedule(rows)
        unique = [{"snapshot": "a", "categories": [SLIDING, VERTICAL]}, {"snapshot": "b", "categories": [HORIZONTAL]}]
        schedule = balanced_schedule(unique)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(primary_category([SLIDING, VERTICAL]), VERTICAL)
        freq = sampling_frequency(unique)
        self.assertAlmostEqual(freq["primary_probability"][VERTICAL], 0.5)
        self.assertEqual(freq["coverage_count"][SLIDING], 1)
        self.assertEqual(freq["coverage_count"][VERTICAL], 1)

    def test_train_pool_excludes_valid(self):
        train = load_train_snapshots()
        ids = [row["id"] for row in train]
        self.assertEqual(len(ids), len(set(ids)))
        snaps = [row["snapshot"] for row in train]
        self.assertEqual(len(snaps), len(set(snaps)))
        self.assertTrue(all(row["split"] == "train" for row in train))
        assert_train_valid_disjoint(train, ROOT / "data/valid_skill_quotas.jsonl")
        self.assertGreater(len(train), 17)

    def test_ballistic_rollout_clears_a_previous_table_latch(self):
        from unittest.mock import patch

        from envs.outcomes.table_drop import latched, restore_state
        from training.adaptive_grpo import independent_rollout

        class _Data:
            time = 0.0

        class _Sim:
            def __init__(self):
                self.data = _Data()

        class _Env:
            def __init__(self):
                self.sim = _Sim()
                self.cleared = False

            def observation(self):
                self.cleared = not latched(self.sim)
                raise RuntimeError("stop-after-reset")

        env = _Env()
        restore_state(env.sim, {"latched": True, "t": 0.2, "kind": "bounce"})
        with patch("training.recovery_execution.restore_ballistic", lambda sim, blob: None):
            with self.assertRaisesRegex(RuntimeError, "stop-after-reset"):
                independent_rollout(
                    env, None, None, {"ref": {}}, np.random.default_rng(0), "cpu", 4.0, "v2", False,
                )
        self.assertTrue(env.cleared)

    def test_sft_checkpoint_is_the_required_init(self):
        from training.adaptive_grpo import SFT_CHECKPOINT

        path = SFT_CHECKPOINT
        self.assertTrue(path.is_file())
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(tuple(saved["model"]["logits.weight"].shape), (48, 64))
        self.assertEqual(int(saved["epoch"]), 40)


class PrefixRestoreTest(unittest.TestCase):
    def test_saved_prefix_replays(self):
        import pickle

        from training.prefix_rollout import restore_probe
        from models.k48_checkpoint import load_trained_k48
        from training.k47_codebook import load_codebook
        from training.recovery_episode import make_training_env

        rows = load_train_snapshots()
        blob = None
        for row in rows:
            candidate = pickle.loads((ROOT / row["snapshot"]).read_bytes())
            if "capture" in candidate and "ref" in candidate:
                blob = candidate
                break
        if blob is None:
            self.skipTest("no capture snapshot in the train pool")
        device = __import__("torch").device("cpu")
        book = load_codebook()
        from training.adaptive_grpo import SFT_CHECKPOINT

        model, _book = load_trained_k48(SFT_CHECKPOINT, device)
        env = make_training_env(device, model=model)
        report = restore_probe(env, model, book, blob, device, 1)
        self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
