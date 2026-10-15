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

    def test_all_methods_fail_skips_update(self):
        decision = exploration_decision(_state(stage_b_done=True, tree_done=True))
        self.assertEqual(decision["action"], "skip")
        self.assertFalse(decision["tree_success_excluded"])

    def test_tree_success_is_not_an_independent_update(self):
        decision = exploration_decision(_state(
            stage_b_done=True, tree_done=True, tree_uniform_success=2,
        ))
        self.assertEqual(decision["action"], "sample")
        self.assertEqual(decision["stage"], "resample")
        failed_resample = exploration_decision(_state(
            stage_b_done=True, tree_done=True, tree_entropy_success=1, resample_done=True, resample_success=0,
        ))
        self.assertEqual(failed_resample["action"], "skip")
        self.assertTrue(failed_resample["tree_success_excluded"])

    def test_uniform_and_entropy_differ(self):
        candidates = [{"entropy": 0.1 * index, "step": index} for index in range(12)]
        rng = np.random.default_rng(1)
        uniform = [item["step"] for item in select_branch_points(candidates, rng, 8, "uniform")]
        entropy = [item["step"] for item in select_branch_points(candidates, np.random.default_rng(1), 8, "entropy")]
        self.assertEqual(entropy, [11, 10, 9, 8, 7, 6, 5, 4])
        self.assertNotEqual(uniform, entropy)

    def test_loss_rejects_tree_and_keeps_failures(self):
        members = [
            {"origin": "independent", "label": "SUCCESS_12S"},
            {"origin": "independent", "label": "TABLE_DROP"},
        ]
        self.assertEqual(len(loss_members(members)), 2)
        with self.assertRaises(RuntimeError):
            loss_members(members + [{"origin": "tree", "label": "SUCCESS_12S"}])
        with self.assertRaises(RuntimeError):
            loss_members([{"origin": "independent", "label": "TABLE_DROP"}])

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

    def test_sft_checkpoint_is_the_required_init(self):
        path = ROOT / "results/sft/current/checkpoint_epoch_0040.pt"
        self.assertTrue(path.is_file())
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(tuple(saved["model"]["logits.weight"].shape), (48, 64))
        self.assertEqual(int(saved["epoch"]), 40)


class PrefixRestoreTest(unittest.TestCase):
    def test_saved_prefix_replays(self):
        import pickle

        from experiments.handoff.benchmark_k48_prefix_branching import restore_probe
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
        model, _book = load_trained_k48(ROOT / "results/sft/current/checkpoint_epoch_0040.pt", device)
        env = make_training_env(device, model=model)
        report = restore_probe(env, model, book, blob, device, 1)
        self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
