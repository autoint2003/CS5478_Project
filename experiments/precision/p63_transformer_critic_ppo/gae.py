"""GAE with true-terminal vs artificial-truncation bootstrap rules."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard GAE.

    ``values`` length is ``len(rewards) + 1`` (bootstrap slot at the end).
    ``dones[t] == 1`` means a *true* episode terminal after reward t
    (drop / handoff / finite-horizon TIMEOUT). Artificial mid-buffer cuts
    leave ``dones[-1] == 0`` and put V(s_T) into ``values[-1]``.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    n = len(rewards)
    assert len(values) == n + 1
    assert len(dones) == n
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in reversed(range(n)):
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    ret = adv + values[:n]
    return adv.astype(np.float32), ret.astype(np.float32)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)
