"""GRPO residual likelihood and group-relative advantages.

SIGMA is the frozen residual sampling scale. The formulas are unchanged.
"""

from __future__ import annotations

import numpy as np

SIGMA = 0.05
LOG2PI = float(np.log(2.0 * np.pi))


def gaussian_logprob(residual, mean):
    var = SIGMA ** 2
    err = (residual - mean) / SIGMA
    return -0.5 * (err ** 2).sum() - 0.5 * 7.0 * (LOG2PI + float(np.log(var)))


def advantages(returns):
    values = np.asarray(returns, np.float64)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values), std, True
    return (values - values.mean()) / std, std, False
