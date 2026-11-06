"""P53 dense pose potential reward + retained P50 contact/slip/time terms.

Φ(s) = w_p exp(-e_p²/(2σ_p²)) + w_θ exp(-e_θ²/(2σ_θ²))
r_pose = γ Φ(s_{t+1}) − Φ(s_t)

Terminal primarily favors SUCCESS_RELAXED_12S. Does not redefine SUCCESS_NEW/12S.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PoseRewardWeights:
    # dense pose (Gaussian RBF on mm / deg)
    w_p: float = 1.0
    w_theta: float = 1.0
    sigma_p_mm: float = 15.0  # matches RELAXED_POS scale
    sigma_theta_deg: float = 15.0
    gamma_phi: float = 0.99
    # retained P50-style shaping
    w_contact: float = 0.15
    w_slip: float = 0.20
    w_imbalance: float = 0.10
    w_time: float = 0.01
    w_switch: float = 0.02
    # terminals (applied once)
    r_success_relaxed: float = 25.0
    r_success_12s_partial: float = 8.0  # 12S without relaxed pose
    r_success_new_bonus: float = 3.0  # optional NEW on top of relaxed/12S
    r_drop: float = -25.0
    r_lost_after_hand: float = -20.0
    r_timeout_fail: float = -10.0
    r_premature_hint: float = 0.0  # logged only; no extra penalty beyond timeout/lost


@dataclass
class RewardBreakdown:
    total: float
    pose: float
    contact: float
    slip: float
    imbalance: float
    time: float
    switch: float
    terminal: float


def pose_potential(m: dict, w: PoseRewardWeights | None = None) -> float:
    """Higher is better; bounded in [0, w_p+w_θ]. Stationary ⇒ Φ constant ⇒ r_pose≈0."""
    ww = w or PoseRewardWeights()
    e = float(m.get("e_pos_mm") or 0.0)
    th = float(m.get("theta") or 0.0)
    phi_p = ww.w_p * float(np.exp(-(e * e) / (2.0 * ww.sigma_p_mm * ww.sigma_p_mm)))
    phi_th = ww.w_theta * float(np.exp(-(th * th) / (2.0 * ww.sigma_theta_deg * ww.sigma_theta_deg)))
    return float(phi_p + phi_th)


def step_reward(
    m0: dict,
    m1: dict,
    *,
    token_switched: bool,
    weights: PoseRewardWeights | None = None,
) -> tuple[float, RewardBreakdown]:
    w = weights or PoseRewardWeights()
    pose = w.gamma_phi * pose_potential(m1, w) - pose_potential(m0, w)

    cq0 = int(m0.get("bilateral") or 0) * min(int(m0.get("min_pad") or 0), 4) / 4.0
    cq1 = int(m1.get("bilateral") or 0) * min(int(m1.get("min_pad") or 0), 4) / 4.0
    contact = w.w_contact * (cq1 - cq0)
    if int(m1.get("bilateral") or 0) and cq1 >= 0.75:
        contact += 0.02

    # Uncontrolled slip only: punish *increase* in relative speed.
    # Controlled C sliding with decreasing/stable vrel is not systematically punished.
    dv = float(m1.get("vrel") or 0.0) - float(m0.get("vrel") or 0.0)
    slip = 0.0
    if int(m1.get("bilateral") or 0) or int(m0.get("bilateral") or 0):
        slip = -w.w_slip * float(np.clip(dv, 0.0, 0.5))

    imb = 0.0
    fl, fr = float(m1.get("fn_L") or 0.0), float(m1.get("fn_R") or 0.0)
    if (fl + fr) > 1e-3 and int(m1.get("bilateral") or 0):
        imb = -w.w_imbalance * abs(fl - fr) / (fl + fr + 1e-6)

    time_p = -w.w_time
    switch_p = -w.w_switch if token_switched else 0.0
    total = pose + contact + slip + imb + time_p + switch_p
    bd = RewardBreakdown(
        total=float(total),
        pose=float(pose),
        contact=float(contact),
        slip=float(slip),
        imbalance=float(imb),
        time=float(time_p),
        switch=float(switch_p),
        terminal=0.0,
    )
    return float(total), bd


def terminal_reward(
    *,
    drop: bool,
    success_relaxed: bool,
    success_12s: bool,
    success_new: bool,
    lost_after_hand: bool,
    timeout_fail: bool,
    weights: PoseRewardWeights | None = None,
) -> tuple[float, RewardBreakdown]:
    w = weights or PoseRewardWeights()
    term = 0.0
    if drop:
        term += w.r_drop
    elif lost_after_hand:
        term += w.r_lost_after_hand
    elif success_relaxed:
        term += w.r_success_relaxed
        if success_new:
            term += w.r_success_new_bonus
    elif success_12s:
        term += w.r_success_12s_partial
        if success_new:
            term += w.r_success_new_bonus
    elif timeout_fail:
        term += w.r_timeout_fail
    bd = RewardBreakdown(
        total=float(term),
        pose=0.0,
        contact=0.0,
        slip=0.0,
        imbalance=0.0,
        time=0.0,
        switch=0.0,
        terminal=float(term),
    )
    return float(term), bd


def trajectory_return(rewards: list[float], *, gamma: float = 1.0) -> float:
    """Undiscounted (gamma=1) sum by default — consistent length scaling via mean logp in GRPO."""
    g = float(gamma)
    if abs(g - 1.0) < 1e-12:
        return float(sum(rewards))
    acc = 0.0
    disc = 1.0
    for r in rewards:
        acc += disc * float(r)
        disc *= g
    return float(acc)


def weights_dict(w: PoseRewardWeights) -> dict:
    return asdict(w)
