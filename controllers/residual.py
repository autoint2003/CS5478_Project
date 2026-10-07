"""Shared residual clipping and rate limits for SAC and the heuristic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResidualCommand:
    dv: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dw: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dfg: float = 0.0


class ResidualLimiter:
    def __init__(self, cfg: dict):
        rec = cfg.get("recovery", {})
        self.dv_max = np.asarray(rec.get("dv_max", [0.06, 0.06, 0.12]), dtype=float)
        self.dw_max = np.asarray(rec.get("dw_max", [0.6, 0.6, 0.6]), dtype=float)
        self.dfg_max = float(rec.get("dfg_max", 16.0))
        self.dv_rate = float(rec.get("dv_rate", 0.8))
        self.dw_rate = float(rec.get("dw_rate", 4.0))
        self.dfg_rate = float(rec.get("dfg_rate", 80.0))
        self.prev = ResidualCommand()

    def reset(self) -> None:
        self.prev = ResidualCommand()

    def clip(self, cmd: ResidualCommand) -> ResidualCommand:
        return ResidualCommand(
            dv=np.clip(cmd.dv, -self.dv_max, self.dv_max),
            dw=np.clip(cmd.dw, -self.dw_max, self.dw_max),
            dfg=float(np.clip(cmd.dfg, -self.dfg_max, self.dfg_max)),
        )

    def apply(self, cmd: ResidualCommand, dt: float) -> ResidualCommand:
        cmd = self.clip(cmd)
        dv = _rate(self.prev.dv, cmd.dv, self.dv_rate * dt)
        dw = _rate(self.prev.dw, cmd.dw, self.dw_rate * dt)
        dfg = float(
            np.clip(cmd.dfg - self.prev.dfg, -self.dfg_rate * dt, self.dfg_rate * dt)
            + self.prev.dfg
        )
        out = ResidualCommand(dv=dv, dw=dw, dfg=dfg)
        self.prev = ResidualCommand(dv=out.dv.copy(), dw=out.dw.copy(), dfg=out.dfg)
        return out

    def from_action(self, action: np.ndarray, kind: str) -> ResidualCommand:
        a = np.asarray(action, dtype=float).reshape(-1)
        cmd = ResidualCommand()
        if kind == "2d":
            cmd.dv = np.array([0.0, 0.0, float(a[0]) * self.dv_max[2]])
            cmd.dfg = float(a[1]) * self.dfg_max
        elif kind == "3d":
            # Diagnostic [dv_y, dv_z, dfg]; official 2d/7d mappings unchanged.
            cmd.dv = np.array(
                [
                    0.0,
                    float(a[0]) * self.dv_max[1],
                    float(a[1]) * self.dv_max[2],
                ]
            )
            cmd.dfg = float(a[2]) * self.dfg_max
        else:
            cmd.dv = a[0:3] * self.dv_max
            cmd.dw = a[3:6] * self.dw_max
            cmd.dfg = float(a[6]) * self.dfg_max
        return cmd

    def to_action(self, cmd: ResidualCommand, kind: str) -> np.ndarray:
        if kind == "2d":
            return np.array(
                [
                    cmd.dv[2] / (self.dv_max[2] + 1e-8),
                    cmd.dfg / (self.dfg_max + 1e-8),
                ],
                dtype=np.float32,
            )
        if kind == "3d":
            return np.array(
                [
                    cmd.dv[1] / (self.dv_max[1] + 1e-8),
                    cmd.dv[2] / (self.dv_max[2] + 1e-8),
                    cmd.dfg / (self.dfg_max + 1e-8),
                ],
                dtype=np.float32,
            )
        return np.concatenate(
            [
                cmd.dv / (self.dv_max + 1e-8),
                cmd.dw / (self.dw_max + 1e-8),
                [cmd.dfg / (self.dfg_max + 1e-8)],
            ]
        ).astype(np.float32)


def _rate(prev: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = np.clip(target - prev, -max_delta, max_delta)
    return prev + delta


# recovery4d: independent of 7d dw_max=0.6.
# SIMULATION recovery-policy bound: a[0]=±1 maps to omega_y=±4 rad/s.
# This is NOT established as hardware-safe for a real Panda.
# Must match config recovery.w_hy_max (RecoveryEnv asserts equality).
RECOVERY4D_W_HY_MAX = 4.0  # rad/s, body y of r_des; simulation policy only
RECOVERY4D_V_HX_MAX = 0.08  # m/s, hand-frame x
RECOVERY4D_V_Z_MAX = 0.08  # m/s, world z, symmetric about 0
RECOVERY4D_TAU_SECURE = -18.0
RECOVERY4D_TAU_OPEN = 2.0  # a3=-1 active open; a3=+1 secure -18


def map_recovery4d(action: np.ndarray, r_des: np.ndarray, r_hand: np.ndarray) -> dict:
    """a in [-1,1]^4 -> world v, world w, tendon tau. No RULE FSM."""
    a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
    if a.size < 4:
        raise ValueError("recovery4d action must have 4 components")
    w_hy = float(a[0]) * RECOVERY4D_W_HY_MAX
    v_hx = float(a[1]) * RECOVERY4D_V_HX_MAX
    v_z = float(a[2]) * RECOVERY4D_V_Z_MAX
    tau = RECOVERY4D_TAU_OPEN + 0.5 * (float(a[3]) + 1.0) * (
        RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN
    )
    r_h = np.asarray(r_hand, float).reshape(3, 3)
    r_d = np.asarray(r_des, float).reshape(3, 3)
    v_world = r_h @ np.array([v_hx, 0.0, 0.0])
    v_world = v_world + np.array([0.0, 0.0, v_z])
    w_world = r_d @ np.array([0.0, w_hy, 0.0])
    return {
        "a": a[:4].copy(),
        "v_hx": v_hx,
        "v_z": v_z,
        "w_hy": w_hy,
        "v_world": v_world,
        "w_world": w_world,
        "tau": float(tau),
    }
