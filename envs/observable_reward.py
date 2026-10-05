"""Observable-tactile recovery reward. No object GT in dense or terminal training terms.

ORACLE reward stays in RecoveryEnv._reward. This module is OBSERVABLE_TACTILE only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.physical_recovery import CONTACT_LOSS_HOLD, E_SCALE, E_TOL, SUCCESS_HOLD
from sensors.ex_estimator import geometric_from_reading
from sensors.spatial_tactile import SpatialTactileReading, measure_spatial_tactile

# Engineering choice from TACTILE_EX_ESTIMATOR_AUDIT.md: 100 ms Δ|e_hat|
# had 0 false-progress events on that diagnostic set. NOT a physical constant.
T_PROGRESS_S = 0.100
T_PROGRESS_PROVENANCE = (
    "results/diagnostics/tactile_ex_estimator: 100 ms false_progress=0 on RULE/all; "
    "50 ms false_progress~0.013. Chosen T_progress=0.100 s (option A: emit once per interval)."
)

# PROVISIONAL: same numeric value as GT V_REL_TOL, but applied to |Δ e_hat|/dt_policy
# (contact-location migration), not object-hand twist.
E_HAT_DOT_TOL = 0.02


def phi_hat(e_hat: float, e_scale: float = E_SCALE) -> float:
    return -abs(float(e_hat)) / max(float(e_scale), 1e-9)


def read_tactile(model, data, ids) -> tuple[SpatialTactileReading, dict]:
    reading = measure_spatial_tactile(model, data, ids)
    est = geometric_from_reading(reading)
    d = reading.to_array()
    d.update(
        {
            "e_hat_x": float(est.e_hat_x),
            "estimate_valid": bool(est.estimate_valid),
            "bilateral_valid": bool(est.bilateral_valid),
            "valid_L": bool(est.valid_L),
            "valid_R": bool(est.valid_R),
            "contact_present_L": bool(reading.left.contact_present),
            "contact_present_R": bool(reading.right.contact_present),
        }
    )
    return reading, d


@dataclass
class ObservableTactileReward:
    """Policy-step accumulator. Progress credited only when a T_progress window completes."""

    dt_policy: float
    e_scale: float = E_SCALE
    lambda_cap: float = 0.02
    lambda_t: float = 0.01
    lambda_a: float = 0.01
    recovered_bonus: float = 8.0
    drop_penalty: float = 8.0
    t_progress: float = T_PROGRESS_S
    e_hat_dot_tol: float = E_HAT_DOT_TOL
    success_hold: float = SUCCESS_HOLD
    contact_loss_hold: float = CONTACT_LOSS_HOLD

    def __post_init__(self) -> None:
        self.n_prog = max(1, int(round(self.t_progress / self.dt_policy)))
        self.reset_state()

    def reset_state(self) -> None:
        self.ref_e = float("nan")
        self.ref_valid = False
        self.steps_in_window = 0
        self.prev_e = float("nan")
        self.prev_valid = False
        self.ok_s = 0.0
        self.lost_s = 0.0

    def seed_from_reading(self, tactile: dict) -> None:
        self.reset_state()
        if tactile["estimate_valid"]:
            self.ref_e = float(tactile["e_hat_x"])
            self.ref_valid = True
            self.prev_e = float(tactile["e_hat_x"])
            self.prev_valid = True
        self.steps_in_window = 0

    def _progress(self, tactile: dict) -> tuple[float, bool]:
        """Option A: r_progress only when a new T_progress interval completes.

        Validity gap: discard reference; do not compare across invalid samples.
        """
        valid = bool(tactile["estimate_valid"])
        e = float(tactile["e_hat_x"]) if valid else float("nan")
        if not valid:
            self.ref_e = float("nan")
            self.ref_valid = False
            self.steps_in_window = 0
            return 0.0, False
        if not self.ref_valid:
            self.ref_e = e
            self.ref_valid = True
            self.steps_in_window = 0
            return 0.0, False
        self.steps_in_window += 1
        if self.steps_in_window < self.n_prog:
            return 0.0, False
        r = phi_hat(e, self.e_scale) - phi_hat(self.ref_e, self.e_scale)
        self.ref_e = e
        self.ref_valid = True
        self.steps_in_window = 0
        return float(r), True

    def _contact(self, tactile: dict) -> float:
        # Same scale as oracle: lambda_cap * (1 if bilat else -0.5) → +0.02 / -0.01
        bilat = bool(tactile["bilateral_valid"])
        return float(self.lambda_cap) * (1.0 if bilat else -0.5)

    def _action(self, a_used: np.ndarray) -> float:
        a = np.asarray(a_used, float).reshape(-1)
        if a.size >= 3:
            return -float(self.lambda_a) * float(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        return -float(self.lambda_a) * float(np.dot(a, a))

    def _success_step(self, tactile: dict) -> bool:
        if not (
            tactile["estimate_valid"]
            and tactile["bilateral_valid"]
            and abs(float(tactile["e_hat_x"])) <= E_TOL
        ):
            return False
        if self.prev_valid:
            edot = abs(float(tactile["e_hat_x"]) - float(self.prev_e)) / max(self.dt_policy, 1e-9)
            if edot > self.e_hat_dot_tol:
                return False
        return True

    def terms(self, tactile: dict, a_used: np.ndarray) -> dict:
        """Dense + hold clocks. Call once per policy step. No GT fields in `tactile`."""
        r_prog, prog_valid = self._progress(tactile)
        r_cap = self._contact(tactile)
        r_t = -float(self.lambda_t)
        r_a = self._action(a_used)
        both_lost = (not tactile["contact_present_L"]) and (not tactile["contact_present_R"])
        if both_lost:
            self.lost_s += self.dt_policy
        else:
            self.lost_s = 0.0
        if self._success_step(tactile):
            self.ok_s += self.dt_policy
        else:
            self.ok_s = 0.0
        success_obs = self.ok_s >= self.success_hold
        failure_obs = self.lost_s >= self.contact_loss_hold
        if tactile["estimate_valid"]:
            self.prev_e = float(tactile["e_hat_x"])
            self.prev_valid = True
        else:
            self.prev_e = float("nan")
            self.prev_valid = False
        r_term = 0.0
        kind = None
        if success_obs:
            r_term = float(self.recovered_bonus)
            kind = "success_obs"
        elif failure_obs:
            r_term = -float(self.drop_penalty)
            kind = "failure_obs"
        r = r_prog + r_cap + r_t + r_a + r_term
        return {
            "r": float(r),
            "r_progress": float(r_prog),
            "r_contact": float(r_cap),
            "r_time": float(r_t),
            "r_action": float(r_a),
            "r_term": float(r_term),
            "progress_valid": bool(prog_valid),
            "success_obs": bool(success_obs),
            "failure_obs": bool(failure_obs),
            "term_obs": kind,
            "ok_s": float(self.ok_s),
            "lost_s": float(self.lost_s),
            "n_prog_steps": int(self.n_prog),
        }


def from_cfg(cfg: dict, dt_policy: float) -> ObservableTactileReward:
    rw = cfg.get("reward", {})
    rec = cfg.get("recovery", {})
    return ObservableTactileReward(
        dt_policy=float(dt_policy),
        e_scale=float(rw.get("e_scale", E_SCALE)),
        lambda_cap=float(rw.get("lambda_cap", 0.02)),
        lambda_t=float(rw.get("lambda_t", 0.01)),
        lambda_a=float(rw.get("lambda_a", 0.01)),
        recovered_bonus=float(rw.get("recovered", 8.0)),
        drop_penalty=float(rw.get("drop", 8.0)),
        t_progress=float(rw.get("t_progress", T_PROGRESS_S)),
        e_hat_dot_tol=float(rw.get("e_hat_dot_tol", E_HAT_DOT_TOL)),
        success_hold=float(rec.get("success_hold", SUCCESS_HOLD)),
        contact_loss_hold=float(rec.get("contact_loss_hold", CONTACT_LOSS_HOLD)),
    )
