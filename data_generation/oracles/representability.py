"""K48 nearest-codeword projection shared by the recovery searches."""

from __future__ import annotations

import numpy as np


def representability(actions, proto):
    distances = []
    tokens = []
    quantized = []
    for action in actions:
        gap = np.max(np.abs(proto - action.reshape(1, 7)), axis=1)
        index = int(np.argmin(gap))
        distances.append(float(gap[index]))
        tokens.append(index)
        residual = np.clip(action - proto[index], -0.25, 0.25)
        quantized.append(np.clip(proto[index] + residual, -1.0, 1.0).astype(np.float32))
    distances = np.asarray(distances, float)
    return {
        "fraction_within_0.25": float(np.mean(distances <= 0.25 + 1e-8)),
        "max_linf": float(np.max(distances)) if len(distances) else None,
        "median_linf": float(np.median(distances)) if len(distances) else None,
        "tokens": tokens,
        "quantized": np.stack(quantized, axis=0) if quantized else np.zeros((0, 7), np.float32),
    }
