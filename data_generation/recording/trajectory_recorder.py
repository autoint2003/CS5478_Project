"""Replay a saved action tape through the official recovery step."""

from __future__ import annotations

import numpy as np

from envs.step_legal import step_legal


def replay_actions(sim, actions, observe):
    """Apply each 7-D action with step_legal and record one observation per step."""
    observations = []
    applied = []
    for action in np.asarray(actions, np.float32):
        applied.append(np.asarray(step_legal(sim, action), np.float32).copy())
        observations.append(np.asarray(observe(sim), np.float32).reshape(39).copy())
    return np.stack(applied, axis=0), np.stack(observations, axis=0)
