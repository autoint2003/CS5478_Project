"""Replay a saved action tape through the official recovery step."""

from __future__ import annotations

import numpy as np

from envs.step_legal import step_legal


def save_pair(folder, stem, executed, observations, contacts, tokens, residuals, handoff_obs):
    """Write one continuous tape and its K48 token/residual tape."""
    folder.mkdir(parents=True, exist_ok=True)
    continuous = folder / f"{stem}.npz"
    tokens_path = folder / f"{stem}__tokens.npz"
    payload = {
        "actions": np.asarray(executed, np.float32),
        "observations": np.asarray(observations, np.float32),
        "timestamps": np.asarray([row["t"] for row in contacts], np.float64),
        "nL": np.asarray([row["nL"] for row in contacts], np.int32),
        "nR": np.asarray([row["nR"] for row in contacts], np.int32),
        "obj_z": np.asarray([row["z"] for row in contacts], np.float32),
    }
    if handoff_obs is not None:
        payload["handoff_obs"] = np.asarray(handoff_obs, np.float32)
    np.savez_compressed(continuous, **payload)
    np.savez_compressed(
        tokens_path,
        tokens=np.asarray(tokens, np.int64),
        residuals=np.asarray(residuals, np.float32),
    )
    return continuous, tokens_path


def replay_actions(sim, actions, observe):
    """Apply each 7-D action with step_legal and record one observation per step."""
    observations = []
    applied = []
    for action in np.asarray(actions, np.float32):
        applied.append(np.asarray(step_legal(sim, action), np.float32).copy())
        observations.append(np.asarray(observe(sim), np.float32).reshape(39).copy())
    return np.stack(applied, axis=0), np.stack(observations, axis=0)
