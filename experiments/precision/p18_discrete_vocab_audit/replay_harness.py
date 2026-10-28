"""Shared continuous / discrete replay harness for P18."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.corrections.contact_primitives import _wrist
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from models.transformer_policy import DT_POLICY


def make_env():
    from controllers.recovery.geometry_switch import GeometrySwitch
    from envs.simulation.ground_sim import make_ground_sim

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def replay_actions(
    env,
    capture: Any,
    ref: Any,
    actions: np.ndarray | list,
    *,
    stabilize_n: int = 8,
    hold_grip_n: int = 0,
) -> dict[str, Any]:
    """Restore capture, play actions, STRICT handoff evaluate."""
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    for _ in range(int(hold_grip_n)):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=int(stabilize_n))
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "pre_pos_mm": h.get("pre_pos_mm"),
        "pre_theta": h.get("pre_theta"),
    }


def quantize_with_book(
    book: np.ndarray,
    actions: np.ndarray,
    rho,
    w: np.ndarray | None = None,
) -> dict[str, Any]:
    from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory

    return quantize_trajectory(book, actions, rho, w=w if w is not None else DEFAULT_WEIGHTS)
