"""Tree-augmented GRPO: fixed 64 branches and a G=80 update."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from models.k48_checkpoint import HANDOFF, build_k48
from training.adaptive_grpo import (
    SFT_CHECKPOINT,
    adaptive_epoch_path,
    adaptive_resume_plan,
    adaptive_round_count,
    apply_group_update,
    cpu_adam_state,
)
from training.adaptive_hybrid import (
    BRANCHES_PER_METHOD,
    STAGE_A,
    STAGE_B,
    allocate_tree_branches,
    execute_allocated_branches,
    exploration_decision,
    prefix_multiplicity,
)
from training.grpo_math import advantages
from training.k47_codebook import group_advantage, load_codebook
from training.k48_config import ROOT
from training.reward_v2 import reward_components
from training.train_grpo import step_logprob


def _trunk(index, steps, entropies=None):
    rows = []
    captures = {}
    for step in range(steps):
        rows.append({
            "step": step,
            "entropy": float(step + 1 if entropies is None else entropies[step]),
            "lost_low": 0,
            "t": step * 0.02,
            "token": 3,
        })
        captures[step] = {"trunk": index, "step": step}
    return {"steps": rows, "captures": captures}


def _record(model, length, token):
    obs = np.zeros((length, 39), np.float32)
    prev = np.zeros((length, 7), np.float32)
    pre = np.zeros(7, np.float32)
    with torch.no_grad():
        logp = step_logprob(model, torch.tensor(obs), torch.tensor(prev), token, pre)
    return {"obs": obs, "prev": prev, "token": int(token), "pre": pre, "logp": float(logp)}


class TreeAugmentedTests(unittest.TestCase):
    def test_each_trunk_gets_two_uniform_and_two_entropy_branches(self):
        trunks = [_trunk(index, 5) for index in range(16)]
        uniform, entropy = allocate_tree_branches(
            trunks, np.random.default_rng(0), np.random.default_rng(1), BRANCHES_PER_METHOD,
        )
        self.assertEqual(len(uniform), 32)
        self.assertEqual(len(entropy), 32)
        for index in range(16):
            self.assertEqual(sum(point["traj"] == index for point in uniform), 2)
            self.assertEqual(sum(point["traj"] == index for point in entropy), 2)
            self.assertEqual(len({point["step"] for point in uniform if point["traj"] == index}), 2)
        top = [point["step"] for point in entropy if point["traj"] == 0]
        self.assertEqual(top, [4, 3])

    def test_short_trunk_reuses_positions(self):
        trunks = [_trunk(0, 1)]
        uniform, entropy = allocate_tree_branches(
            trunks, np.random.default_rng(2), np.random.default_rng(3),
        )
        self.assertEqual([point["step"] for point in uniform], [0, 0])
        self.assertEqual([point["step"] for point in entropy], [0, 0])

    def test_run_tree_builds_a_real_group_of_eighty(self):
        from unittest.mock import patch

        from training import adaptive_grpo as runner

        calls = {"n": 0}

        def fake_rollout(*_args, **_kwargs):
            calls["n"] += 1
            row = {
                "token": 4,
                "t": 0.02,
                "pre": np.zeros(7, np.float32),
                "obs": np.zeros((1, 39), np.float32),
                "prev": np.zeros((1, 7), np.float32),
                "logp": -1.0,
            }
            success = calls["n"] == 1
            return {
                "reason": "SUCCESS_12S" if success else "TABLE_DROP",
                "success": success,
                "recovery_elapsed": 0.2,
                "recovery_physics": 10,
                "nominal_elapsed": 0.0,
                "q_nominal": 0.0,
                "lift_completed": False,
                "reward": 1.99 if success else -0.995,
                "handoff": "early" if success else None,
            }, [row], {}, 0.0

        trunks = []
        for index in range(16):
            record = {
                "obs": np.zeros((1, 39), np.float32),
                "prev": np.zeros((1, 7), np.float32),
                "token": 0,
                "pre": np.zeros(7, np.float32),
                "logp": -0.5,
            }
            trunks.append({
                "origin": "independent",
                "steps": [{"step": 0, "entropy": 0.2 + index, "lost_low": 0, "t": 0.0, "token": 0}],
                "captures": {0: {"trunk": index}},
                "records": [record],
                "t0": 0.0,
                "return": -0.995,
                "truncated": False,
                "label": "TABLE_DROP",
                "success": False,
                "recovery_physics": 10,
                "nominal_elapsed": 0.0,
                "handoff_t": None,
            })
        with patch("training.adaptive_grpo.tree_rollout", fake_rollout):
            uniform, entropy = runner._run_tree(None, None, None, {"ref": {}}, "cpu", trunks, 11, "v2", 4.0)
        self.assertEqual(calls["n"], 64)
        group = trunks + uniform + entropy
        self.assertEqual(len(group), 80)
        self.assertEqual(sum(int(item["success"]) for item in uniform + entropy), 1)
        stitched = next(item for item in uniform + entropy if item["success"])
        self.assertEqual(len(stitched["records"]), 1)
        self.assertEqual(stitched["prefix_len"], 0)
        self.assertEqual(stitched["label"], "SUCCESS_12S")
        self.assertGreater(stitched["return"], max(item["return"] for item in trunks))

    def test_success_does_not_stop_remaining_branches(self):
        trunks = [_trunk(index, 4) for index in range(16)]
        uniform, entropy = allocate_tree_branches(
            trunks, np.random.default_rng(4), np.random.default_rng(5),
        )
        calls = []

        def rollout_one(point):
            calls.append(point)
            return {"success": len(calls) == 1, "label": "SUCCESS_12S" if len(calls) == 1 else "TABLE_DROP"}

        leaves = execute_allocated_branches(uniform + entropy, rollout_one)
        self.assertEqual(len(calls), 64)
        self.assertEqual(len(leaves), 64)
        self.assertEqual(sum(int(leaf["success"]) for leaf in leaves), 1)

    def test_stage_group_sizes(self):
        self.assertEqual(STAGE_A, 8)
        self.assertEqual(STAGE_A + STAGE_B, 16)
        self.assertEqual((STAGE_A + STAGE_B) * (1 + 2 * BRANCHES_PER_METHOD), 80)
        early = exploration_decision({"stage_a_success": 1, "stage_b_done": False, "tree_done": False})
        self.assertEqual(early["stage"], "A")
        middle = exploration_decision({
            "stage_a_success": 0, "stage_b_success": 1, "stage_b_done": True, "tree_done": False,
        })
        self.assertEqual(middle["stage"], "B")
        late = exploration_decision({
            "stage_a_success": 0, "stage_b_success": 0, "stage_b_done": True, "tree_done": True,
            "tree_uniform_success": 1,
        })
        self.assertEqual(late["group"], "stage_c")

    def test_population_advantage_for_one_high_return(self):
        returns = [0.0] * 79 + [1.0]
        values, std, skipped = advantages(returns)
        self.assertFalse(skipped)
        self.assertAlmostEqual(std, math.sqrt(79) / 80, places=9)
        self.assertAlmostEqual(float(values[-1]), math.sqrt(79), places=6)
        self.assertAlmostEqual(float(values[0]), -1.0 / math.sqrt(79), places=6)
        grouped, grouped_skip = group_advantage(returns, [False] * 80)
        self.assertFalse(grouped_skip)
        self.assertAlmostEqual(float(grouped[-1]), math.sqrt(79), places=6)

    def test_zero_variance_skips_without_a_fake_signal(self):
        _values, _std, skipped = advantages([1.0, 1.0, 1.0])
        self.assertTrue(skipped)

    def test_tree_success_enters_the_update(self):
        book = load_codebook()
        model = build_k48(book, torch.device("cpu"))
        ref = build_k48(book, torch.device("cpu"))
        ref.load_state_dict(model.state_dict())
        model.train()
        success = {
            "origin": "tree",
            "label": "SUCCESS_12S",
            "truncated": False,
            "return": 1.0,
            "records": [_record(model, 2, 0)],
            "prefix_source": 0,
            "prefix_len": 1,
        }
        members = []
        for _ in range(79):
            members.append({
                "origin": "independent",
                "label": "TABLE_DROP",
                "truncated": False,
                "return": 0.0,
                "records": [],
            })
        members.append(success)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        stats = apply_group_update(model, ref, members, {"clip": 0.2, "kl_coef": 0.0}, opt, torch.device("cpu"))
        self.assertFalse(stats["skipped"])
        self.assertEqual(stats["group"], 80)
        self.assertEqual(stats["success"], 1)
        self.assertAlmostEqual(stats["advantages"][-1], math.sqrt(79), places=5)
        self.assertGreater(stats["grad_norm"], 0.0)
        self.assertEqual(stats["clip_fraction"], 0.0)
        self.assertEqual(prefix_multiplicity(members)["max"], 1)

    def test_handoff_logprob_ignores_residual_and_lengths_vary(self):
        book = load_codebook()
        model = build_k48(book, torch.device("cpu"))
        obs = torch.zeros(3, 39)
        prev = torch.zeros(3, 7)
        short = step_logprob(model, obs[:1], prev[:1], HANDOFF, np.zeros(7, np.float32))
        other = step_logprob(model, obs[:1], prev[:1], HANDOFF, np.ones(7, np.float32))
        self.assertEqual(float(short.detach()), float(other.detach()))
        ref = build_k48(book, torch.device("cpu"))
        ref.load_state_dict(model.state_dict())
        model.train()
        members = [
            {
                "origin": "tree", "label": "SUCCESS_12S", "truncated": False, "return": 1.0,
                "records": [_record(model, 1, HANDOFF)], "prefix_source": 0, "prefix_len": 0,
            },
            {
                "origin": "independent", "label": "TABLE_DROP", "truncated": False, "return": 0.0,
                "records": [_record(model, 4, 1), _record(model, 2, 2), _record(model, 3, 3)],
            },
        ]
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        stats = apply_group_update(model, ref, members, {"clip": 0.2, "kl_coef": 0.05}, opt, torch.device("cpu"))
        self.assertTrue(math.isfinite(stats["loss"]))
        self.assertTrue(math.isfinite(stats["kl"]))
        self.assertGreater(stats["grad_norm"], 0.0)

    def test_resume_counts_completed_epochs_and_refuses_clobber(self):
        import tempfile

        self.assertEqual(adaptive_epoch_path("results/grpo/tree80", 1).name, "checkpoint_epoch_0001.pt")
        self.assertEqual(adaptive_round_count({"ic_epochs": 1}, None), 1)
        self.assertEqual(adaptive_round_count({"ic_epochs": 1}, 3), 3)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            fresh = out / "fresh"
            fresh.mkdir()
            started = adaptive_resume_plan(SFT_CHECKPOINT, fresh)
            self.assertEqual(started["epoch"], 0)
            (fresh / "ic_log.jsonl").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                adaptive_resume_plan(SFT_CHECKPOINT, fresh)
            with self.assertRaises(RuntimeError):
                adaptive_resume_plan(out / "missing.pt", out)
            bare = out / "checkpoint_adaptive.pt"
            torch.save({"model": {}, "init_checkpoint": str(SFT_CHECKPOINT)}, bare)
            with self.assertRaisesRegex(RuntimeError, "no epoch"):
                adaptive_resume_plan(bare, out)
            weights_only = out / "checkpoint_epoch_0001.pt"
            torch.save({"epoch": 1, "updates": 99, "init_checkpoint": "results/sft/checkpoint_epoch_0040.pt"}, weights_only)
            with self.assertRaisesRegex(RuntimeError, "no Adam moments"):
                adaptive_resume_plan(weights_only, out)
            moment = torch.zeros(2)
            moment[0] = 0.25
            second = torch.ones(2)
            optimizer = {
                "state": {0: {"step": torch.tensor(3), "exp_avg": moment, "exp_avg_sq": second}},
                "param_groups": [{"lr": 3e-5, "betas": (0.9, 0.999), "eps": 1e-8, "params": [0]}],
            }
            torch.save({
                "epoch": 1,
                "updates": 99,
                "init_checkpoint": "results/sft/checkpoint_epoch_0040.pt",
                "optimizer": optimizer,
            }, weights_only)
            plan = adaptive_resume_plan(weights_only, out)
            self.assertEqual(plan["epoch"], 1)
            self.assertEqual(plan["updates"], 99)
            self.assertTrue(torch.equal(plan["optimizer"]["state"][0]["exp_avg"], moment))
            self.assertEqual(plan["init_checkpoint"].resolve(), SFT_CHECKPOINT.resolve())
            layer = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([[0.5, -0.25]]))
            opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
            loss = layer(torch.ones(2)).sum()
            loss.backward()
            opt.step()
            packed = cpu_adam_state(opt)
            self.assertIn("exp_avg", packed["state"][0])
            self.assertIn("exp_avg_sq", packed["state"][0])
            restored = torch.optim.Adam(layer.parameters(), lr=1e-3)
            restored.load_state_dict(packed)
            self.assertTrue(torch.equal(restored.state[layer.weight]["exp_avg"], opt.state[layer.weight]["exp_avg"]))
            elsewhere = out / "other"
            elsewhere.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must live"):
                adaptive_resume_plan(weights_only, elsewhere)

    def test_reward_v2_success_and_drop_are_unchanged(self):
        success = reward_components("SUCCESS_12S", 1.0, 4.0, q_nominal=1.0, lift_completed=True)
        drop = reward_components("TABLE_DROP", 1.0, 4.0, lift_completed=False)
        self.assertAlmostEqual(success["return"], 1.95)
        self.assertAlmostEqual(drop["return"], -0.975)

    def test_branch_restore_matches_one_recorded_observation(self):
        checkpoint = ROOT / "results/sft/checkpoint_epoch_0040.pt"
        if not checkpoint.is_file():
            checkpoint = ROOT / "results/sft/current/checkpoint_epoch_0040.pt"
        if not checkpoint.is_file():
            self.skipTest("no K48 checkpoint is available")
        from models.k48_checkpoint import load_trained_k48
        from training.adaptive_grpo import independent_rollout
        from training.recovery_episode import make_training_env
        from envs.simulation.capture import restore_snapshot

        device = torch.device("cpu")
        model, _book = load_trained_k48(checkpoint, device)
        model.eval()
        book = load_codebook()
        env = make_training_env(device, model=model)
        snapshot = next(Path(ROOT / "data/snapshots/train").glob("*.pkl"))
        import pickle
        with snapshot.open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            from data_generation.strategies.recatch_oracle import _setup

            _sim, _base, ref, _target, _code = _setup()
            blob = dict(blob)
            blob["ref"] = ref
        rollout = independent_rollout(
            env, model, book, blob, np.random.default_rng(7), device, 4.0, "v2", True,
        )
        self.assertGreater(len(rollout["records"]), 0)
        step = min(2, len(rollout["records"]) - 1)
        restore_snapshot(env, rollout["captures"][step])
        obs = np.asarray(env.observation(), np.float32)
        recorded = np.asarray(rollout["records"][step]["obs"][-1], np.float32)
        self.assertLess(float(np.max(np.abs(obs - recorded))), 1e-5)
        prefix = rollout["records"][:step]
        self.assertEqual(len(prefix), step)
        self.assertNotEqual(id(prefix), id(rollout["records"]))
