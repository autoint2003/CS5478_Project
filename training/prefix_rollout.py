"""Stochastic K48 rollout used by prefix branching.

The sampling order matches GRPO: a token is drawn from the 48-way softmax,
then the residual is ``mean + N(0, SIGMA)``, including when the token is
HANDOFF. Experiments import this module. This module does not import them.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch

from envs.outcomes.horizon_outcome import _contacts, finish_recovery, recovery_failed
from envs.simulation.capture import restore_capture, restore_snapshot, take_snapshot
from envs.step_legal import N_SUB, step_legal
from models.k48_checkpoint import HANDOFF
from training.grpo_math import SIGMA
from training.k47_codebook import decode_action

LIMIT_S = 4.0
CAP_STEPS = 200


def token_entropy(probs):
    """Categorical entropy of the 48-way token distribution, in nats."""
    clipped = np.clip(np.asarray(probs, np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return 0.0
    clipped = clipped / total
    positive = clipped[clipped > 0.0]
    return float(-(positive * np.log(positive)).sum())


def sample_action(model, window_obs, window_prev, rng):
    with torch.no_grad():
        logits, mean, _act = model(window_obs, window_prev)
    probs = torch.softmax(logits[-1].float(), dim=-1).cpu().numpy()
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(float(probs.sum()), 1e-12)
    token = int(rng.choice(len(probs), p=probs))
    pre = mean[-1].detach().float().cpu().numpy() + rng.normal(0.0, SIGMA, size=7).astype(np.float32)
    return token, pre.astype(np.float32), probs, token_entropy(probs)


def _windows(env, device):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    window_obs = torch.tensor(np.stack(env._obs_seq[-160:]), device=device)
    window_prev = torch.tensor(np.stack(env._prev_seq[-160:]), device=device)
    return obs, window_obs, window_prev


def _nominal_steps(outcome, physics_dt):
    if outcome.get("handoff") is None:
        return 0
    elapsed = float(outcome.get("nominal_elapsed") or 0.0)
    counted = int(round(elapsed / float(physics_dt)))
    if abs(elapsed - counted * float(physics_dt)) > 1e-6:
        raise RuntimeError(f"nominal elapsed {elapsed} is not an integer number of physics steps")
    return counted


def _score_view(outcome, recovery_physics, physics_dt):
    return {
        "reason": outcome.get("reason"),
        "success": bool(outcome.get("success")),
        "handoff": outcome.get("handoff"),
        "recovery_elapsed": None if outcome.get("recovery_elapsed") is None else float(outcome["recovery_elapsed"]),
        "recovery_physics": int(recovery_physics),
        "nominal_steps": _nominal_steps(outcome, physics_dt),
        "nominal_elapsed": float(outcome.get("nominal_elapsed") or 0.0),
        "q_nominal": float(outcome.get("q_nominal") or 0.0),
        "lift_completed": bool(outcome.get("lift_completed")),
        "reward": outcome.get("reward"),
    }


def rollout(env, model, book, blob, rng, device, capture=None, lost_low=0, t0=None, keep_captures=False, keep_records=False, expected_obs=None):
    """One stochastic K48 rollout. Matches train_grpo sampling, including noise before HANDOFF."""
    if capture is None:
        if "capture" in blob:
            restore_capture(env, blob["capture"])
        else:
            from models.airborne_v3 import DT_POLICY
            from envs.observations.observable_obs import ObservableObsState
            from envs.outcomes.table_drop import restore_state
            from training.recovery_execution import restore_ballistic

            if "ref" not in blob:
                raise RuntimeError("ballistic snapshot has no nominal grasp reference")
            restore_ballistic(env.sim, blob)
            restore_state(env.sim, None)
            env._hist = ObservableObsState(DT_POLICY)
            env._obs_seq = []
            env._prev_seq = []
            env._prev = np.zeros(7, np.float32)
        lost_low = 0
        t0 = float(env.sim.data.time)
    else:
        restore_snapshot(env, capture)
        if t0 is None:
            raise RuntimeError("a restored rollout needs the original recovery start time")
    how = None
    physics = 0
    physics_dt = float(env.sim.model.opt.timestep)
    rows = []
    captures = {}
    checked_obs = False
    for _ in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        if keep_captures:
            captures[len(rows)] = take_snapshot(env)
        obs, window_obs, window_prev = _windows(env, device)
        if expected_obs is not None and not checked_obs:
            mismatch = float(np.max(np.abs(np.asarray(obs, np.float32) - np.asarray(expected_obs, np.float32))))
            if mismatch > 1e-4:
                raise RuntimeError(f"restored branch observation differs from the trunk record by {mismatch}")
            checked_obs = True
        token, pre, probs, entropy = sample_action(model, window_obs, window_prev, rng)
        n_l, n_r, z = _contacts(env.sim)
        row = {
            "step": len(rows),
            "t": float(env.sim.data.time) - t0,
            "token": int(token),
            "pre": np.asarray(pre, np.float32),
            "entropy": float(entropy),
            "p47": float(probs[HANDOFF]),
            "lost_low": int(lost_low),
            "nL": int(n_l),
            "nR": int(n_r),
            "z": float(z),
            "obj": np.asarray(env.sim.data.xpos[env.sim.ids.object_body], np.float32).copy(),
            "hand": np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], np.float32).copy(),
            "hist_len": int(len(env._obs_seq)),
        }
        if keep_records:
            from training.train_grpo import step_logprob

            logp = step_logprob(model, window_obs, window_prev, token, pre)
            row["obs"] = window_obs.detach().float().cpu().numpy()
            row["prev"] = window_prev.detach().float().cpu().numpy()
            row["logp"] = float(logp.detach())
        rows.append(row)
        if token == HANDOFF:
            how = "early"
            break
        if int(token) < 0 or int(token) >= HANDOFF:
            raise RuntimeError(f"illegal motion token {token}")
        _residual, action = decode_action(book, token, pre)
        before = float(env.sim.data.time)
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        advanced = float(env.sim.data.time) - before
        expected = N_SUB * physics_dt
        if abs(advanced - expected) > 1e-8:
            raise RuntimeError(f"recovery step advanced {advanced}, expected {expected}")
        env._prev = executed
        row["residual"] = np.asarray(_residual, np.float32)
        row["grip"] = float(executed[6])
        physics += int(N_SUB)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            from envs.outcomes.table_drop import terminal_reason
            from envs.outcomes.horizon_outcome import physical_reward

            outcome = {
                "reason": terminal_reason(env.sim),
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "nominal_elapsed": 0.0,
                "lift_completed": False,
                "q_nominal": 0.0,
                "reward": physical_reward(terminal_reason(env.sim), float(env.sim.data.time) - t0),
            }
            return _score_view(outcome, physics, physics_dt), rows, captures, t0
    if how is None:
        how = "forced"
    outcome = finish_recovery(env, blob["ref"], t0, how)
    return _score_view(outcome, physics, physics_dt), rows, captures, t0


def restore_probe(env, model, book, blob, device, seed):
    """Check physics, controller, contacts, observation, and one-step continuation."""
    rng = np.random.default_rng(int(seed) + 17)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    saved = None
    target = None
    for step in range(8):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            break
        capture = take_snapshot(env)
        obs = np.asarray(env.observation(), np.float32).copy()
        n_l, n_r, _z = _contacts(env.sim)
        phase = str(env.sim.fsm.phase)
        qvel = np.asarray(env.sim.data.qvel, np.float64).copy()
        ctrl = np.asarray(env.sim.data.ctrl, np.float64).copy()
        clock = float(env.sim.data.time)
        hist = len(env._obs_seq)
        if step == min(3, step) and saved is None and step == 3:
            saved = {
                "capture": capture,
                "obs": obs,
                "nL": int(n_l),
                "nR": int(n_r),
                "phase": phase,
                "qvel": qvel,
                "ctrl": ctrl,
                "time": clock,
                "hist": hist,
                "lost_low": int(lost_low),
                "rng": pickle.loads(pickle.dumps(rng.bit_generator.state)),
            }
            target = step
        _obs, window_obs, window_prev = _windows(env, device)
        token, pre, _probs, _entropy = sample_action(model, window_obs, window_prev, rng)
        if token == HANDOFF:
            break
        _residual, action = decode_action(book, token, pre)
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = executed
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            break
    if saved is None:
        raise RuntimeError("restore probe did not reach a branchable step")
    restore_snapshot(env, saved["capture"])
    obs2 = np.asarray(env.observation(), np.float32)
    n_l, n_r, _z = _contacts(env.sim)
    report = {
        "step": int(target),
        "obs_max_abs": float(np.max(np.abs(obs2 - saved["obs"]))),
        "qvel_max_abs": float(np.max(np.abs(np.asarray(env.sim.data.qvel, np.float64) - saved["qvel"]))),
        "ctrl_max_abs": float(np.max(np.abs(np.asarray(env.sim.data.ctrl, np.float64) - saved["ctrl"]))),
        "time_abs": abs(float(env.sim.data.time) - saved["time"]),
        "phase_match": str(env.sim.fsm.phase) == saved["phase"],
        "contact_match": (int(n_l), int(n_r)) == (saved["nL"], saved["nR"]),
        "hist_match": len(env._obs_seq) == saved["hist"],
        "lost_low_stored": saved["lost_low"],
    }
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    rng_a.bit_generator.state = pickle.loads(pickle.dumps(saved["rng"]))
    rng_b.bit_generator.state = pickle.loads(pickle.dumps(saved["rng"]))
    _obs, window_obs, window_prev = _windows(env, device)
    token_a, pre_a, _probs, _entropy = sample_action(model, window_obs, window_prev, rng_a)
    qpos_a = None
    if token_a != HANDOFF:
        _residual, action = decode_action(book, token_a, pre_a)
        step_legal(env.sim, action)
        qpos_a = np.asarray(env.sim.data.qpos, np.float64).copy()
    restore_snapshot(env, saved["capture"])
    _obs, window_obs, window_prev = _windows(env, device)
    token_b, pre_b, _probs, _entropy = sample_action(model, window_obs, window_prev, rng_b)
    qpos_b = None
    if token_b != HANDOFF:
        _residual, action = decode_action(book, token_b, pre_b)
        step_legal(env.sim, action)
        qpos_b = np.asarray(env.sim.data.qpos, np.float64).copy()
    report["continuation_token_match"] = int(token_a) == int(token_b)
    report["continuation_pre_max_abs"] = float(np.max(np.abs(pre_a - pre_b)))
    report["continuation_qpos_max_abs"] = None if qpos_a is None or qpos_b is None else float(np.max(np.abs(qpos_a - qpos_b)))
    report["passed"] = bool(
        report["obs_max_abs"] <= 1e-6
        and report["qvel_max_abs"] <= 1e-6
        and report["ctrl_max_abs"] <= 1e-6
        and report["time_abs"] <= 1e-9
        and report["phase_match"]
        and report["contact_match"]
        and report["hist_match"]
        and report["continuation_token_match"]
        and report["continuation_pre_max_abs"] <= 1e-6
        and (report["continuation_qpos_max_abs"] is None or report["continuation_qpos_max_abs"] <= 1e-6)
    )
    return report
