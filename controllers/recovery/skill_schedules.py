"""Fixed 7D recovery action tapes used by the historical skill demonstrations.

These are physical command sequences, not optimizer or learning-rate schedules.
"""

from __future__ import annotations

import numpy as np

def embed4(a4) -> list:
    """4D [wy, vhx, vz_world, grip] -> 7D. Refuses a nonzero world-z channel."""
    a = np.asarray(a4, float).reshape(4)
    if abs(float(a[2])) > 1e-8:
        raise ValueError("4D world-z is not a hand-frame vz; this skill does not use it")
    return [float(a[1]), 0.0, 0.0, 0.0, float(a[0]), 0.0, float(a[3])]


def rep(a, n):
    return [list(a) for _ in range(int(n))]


def schedules() -> dict:
    return {
        "wrist_reseat": rep(embed4([0.8, 0.6, 0.0, 1.0]), 12) + rep(embed4([0, 0, 0, 1]), 6),
        "wrist_opposite": rep(embed4([-0.8, -0.6, 0.0, 1.0]), 12) + rep(embed4([0, 0, 0, 1]), 6),
        "recatch_open20": rep(embed4([1, 0, 0, 1]), 19) + rep(embed4([1, 0, 0, -1]), 1) + rep(embed4([0, 0, 0, 1]), 12),
        "recatch_open60": rep(embed4([1, 0, 0, 1]), 19) + rep(embed4([1, 0, 0, -1]), 3) + rep(embed4([0, 0, 0, 1]), 12),
        "gravity_inward": rep([0, 0, 0, 0.28, 0.50, 0, 1.0], 20) + rep([0, 0, 0, 0, 0, 0, -0.48], 24) + rep([0, 0, 0, 0, 0, 0, 1.0], 8),
        "gravity_outward": rep([0, 0, 0, 0.28, -0.50, 0, 1.0], 20) + rep([0, 0, 0, 0, 0, 0, -0.48], 24) + rep([0, 0, 0, 0, 0, 0, 1.0], 8),
        "gravity_secure": rep([0, 0, 0, 0.28, 0.50, 0, 1.0], 20) + rep([0, 0, 0, 0, 0, 0, 1.0], 24) + rep([0, 0, 0, 0, 0, 0, 1.0], 8),
    }
