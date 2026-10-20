"""Numpy k-means++ used to build action tokens."""

from __future__ import annotations

import numpy as np

def kmeans(x, k, seed, restarts=8, iters=40):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = np.empty((k, x.shape[1]), np.float32)
        choice = rng.integers(0, len(x))
        centers[0] = x[choice]
        for j in range(1, k):
            d = ((x[:, None, :] - centers[None, :j, :]) ** 2).sum(-1).min(1)
            probs = d / max(float(d.sum()), 1e-8)
            centers[j] = x[rng.choice(len(x), p=probs)]
        labels = np.zeros(len(x), np.int64)
        for _it in range(iters):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = dist.argmin(1)
            nxt = centers.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    nxt[j] = x[mask].mean(0)
            if np.allclose(nxt, centers, atol=1e-6):
                centers = nxt
                break
            centers = nxt
        inertia = float(((x - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers.astype(np.float32), labels.copy())
    return best
