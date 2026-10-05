"""e_x estimators from SpatialTactileReading only. No object GT inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensors.spatial_tactile import FingerTactile, SpatialTactileReading


@dataclass(frozen=True)
class ExEstimate:
    e_hat_x: float
    estimate_valid: bool
    bilateral_valid: bool
    valid_L: bool
    valid_R: bool
    method: str


def _nan() -> float:
    return float("nan")


def geometric_from_reading(reading: SpatialTactileReading) -> ExEstimate:
    """Direct mapping using finger-frame convention (VERIFIED FROM CODE).

    Left finger X aligns with hand X → u_L ≈ contact hand-x.
    Right finger X = −hand X (180 deg about Z) → −u_R ≈ contact hand-x.
    For this cylinder grasp, contact hand-x ≈ object-origin e_x.
    """
    vL = bool(reading.left.valid)
    vR = bool(reading.right.valid)
    parts = []
    if vL:
        parts.append(float(reading.left.u))
    if vR:
        parts.append(-float(reading.right.u))
    if not parts:
        return ExEstimate(_nan(), False, False, vL, vR, "geometric")
    return ExEstimate(
        e_hat_x=float(np.mean(parts)),
        estimate_valid=True,
        bilateral_valid=vL and vR,
        valid_L=vL,
        valid_R=vR,
        method="geometric",
    )


def features_uv(reading: SpatialTactileReading) -> np.ndarray | None:
    if not (reading.left.valid and reading.right.valid):
        return None
    return np.array(
        [float(reading.left.u), float(reading.right.u)], dtype=float
    )


def fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with intercept. Returns [b0, b1, ...]."""
    A = np.column_stack([np.ones(len(X)), X])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    n = A.shape[1]
    I = np.eye(n)
    I[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + lam * I, A.T @ y)
    return w


def apply_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    return A @ w


class LinearExEstimator:
    def __init__(self, w: np.ndarray, method: str):
        self.w = np.asarray(w, float)
        self.method = method

    def from_uv(self, u_L: float, u_R: float, valid_L: bool, valid_R: bool) -> ExEstimate:
        if not (valid_L and valid_R):
            parts = []
            if valid_L:
                parts.append(float(u_L))
            if valid_R:
                parts.append(-float(u_R))
            if not parts:
                return ExEstimate(_nan(), False, False, valid_L, valid_R, self.method)
            return ExEstimate(float(np.mean(parts)), True, False, valid_L, valid_R, self.method)
        y = float(self.w[0] + self.w[1] * u_L + self.w[2] * u_R)
        return ExEstimate(y, True, True, True, True, self.method)
