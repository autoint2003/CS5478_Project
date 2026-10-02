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
