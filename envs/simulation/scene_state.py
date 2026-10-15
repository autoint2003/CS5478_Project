"""Table height and the parked impact ball.

TABLE_TOP is 0.40 m. Parking the ball does not change the cylinder or the hand.
"""

from __future__ import annotations

import mujoco
import numpy as np

TABLE_TOP = 0.40
BALL_R = 0.012
BALL_PARK_POS = np.array([1.00, 0.80, 1.20])


def park_impact_ball(sim) -> None:
    """Park/remove the diagnostic ball. No-op on scenes without a ball."""
    if not hasattr(sim, "park_ball"):
        return
    sim.park_ball()
    # Hold the parked ball (diagnostic only; cylinder/hand physics unchanged).
    if hasattr(sim.model, "body_gravcomp") and hasattr(sim, "ball_body"):
        sim.model.body_gravcomp[int(sim.ball_body)] = 1.0
    mujoco.mj_forward(sim.model, sim.data)


def ball_is_parked(sim) -> bool:
    if not hasattr(sim, "ball_qadr"):
        return True
    q = int(sim.ball_qadr)
    d = int(sim.ball_dadr)
    pos = np.array(sim.data.qpos[q : q + 3], float)
    vel = np.array(sim.data.qvel[d : d + 6], float)
    return bool(np.allclose(pos, BALL_PARK_POS, atol=1e-6) and np.linalg.norm(vel) < 1e-8)

