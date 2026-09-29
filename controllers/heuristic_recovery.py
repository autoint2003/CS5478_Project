"""Hand-designed reactive recovery: same bounds as SAC."""

from __future__ import annotations

import numpy as np

from controllers.residual import ResidualCommand, ResidualLimiter


class HeuristicRecovery:
    def __init__(self, cfg: dict, limiter: ResidualLimiter):
        h = cfg.get("heuristic", {})
        self.kp_F = float(h.get("kp_F", 8.0))
        self.kd_F = float(h.get("kd_F", 1.5))
        self.kp_v = float(h.get("kp_v", 0.10))
        self.kd_v = float(h.get("kd_v", 0.02))
        self.limiter = limiter

    def command(self, D: float, Ddot: float) -> ResidualCommand:
        dfg = self.kp_F * D + self.kd_F * Ddot
        dvz = -(self.kp_v * D + self.kd_v * Ddot)
        dvz = float(np.clip(dvz, -self.limiter.dv_max[2], 0.0))
        cmd = ResidualCommand(
            dv=np.array([0.0, 0.0, dvz]),
            dw=np.zeros(3),
            dfg=float(dfg),
        )
        return self.limiter.clip(cmd)
