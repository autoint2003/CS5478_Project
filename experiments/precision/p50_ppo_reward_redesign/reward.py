"""P50 potential-based recovery reward (stable grasp → R3F, not WORKING spam)."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class RewardWeights:
    w_progress: float = 1.0
    w_contact: float = 0.15
    w_slip: float = 0.20
    w_imbalance: float = 0.10
    w_time: float = 0.01
    w_switch: float = 0.02
    # terminals (applied once)
    r_success_12s: float = 20.0
    r_success_new_bonus: float = 5.0
    r_drop: float = -20.0
    r_lost_after_hand: float = -20.0
    r_timeout_fail: float = -10.0
    gamma_phi: float = 0.99


@dataclass
class RewardBreakdown:
    total: float
    progress: float
    contact: float
    slip: float
    imbalance: float
    time: float
    switch: float
    terminal: float


def potential(m: dict) -> float:
    """Higher is better. Uses mm/deg/contact quality; no flat WORKING bonus."""
    e = float(m.get("e_pos_mm") or 0.0)
    th = float(m.get("theta") or 0.0)
    bil = int(m.get("bilateral") or 0)
    min_pad = int(m.get("min_pad") or 0)
    vrel = float(m.get("vrel") or 0.0)
    # contact quality 0..1
    cq = bil * min(max(min_pad, 0), 4) / 4.0
    phi = -0.05 * e - 0.03 * th - 0.5 * vrel + 1.5 * cq
    return float(phi)


def step_reward(
    m0: dict,
    m1: dict,
    *,
    token_switched: bool,
    weights: RewardWeights | None = None,
) -> tuple[float, RewardBreakdown]:
    w = weights or RewardWeights()
    # potential-based progress
    prog = w.w_progress * (w.gamma_phi * potential(m1) - potential(m0))
    # contact: reward sustained bilateral quality change, not raw WORKING
    cq0 = int(m0.get("bilateral") or 0) * min(int(m0.get("min_pad") or 0), 4) / 4.0
    cq1 = int(m1.get("bilateral") or 0) * min(int(m1.get("min_pad") or 0), 4) / 4.0
    contact = w.w_contact * (cq1 - cq0)
    if int(m1.get("bilateral") or 0) and cq1 >= 0.75:
        contact += 0.02  # tiny sustain, capped via time penalty
    # slip: punish increase in relative speed when in contact
    dv = float(m1.get("vrel") or 0.0) - float(m0.get("vrel") or 0.0)
    slip = 0.0
    if int(m1.get("bilateral") or 0) or int(m0.get("bilateral") or 0):
        slip = -w.w_slip * float(np.clip(dv, 0.0, 0.5))
    # force imbalance if available
    imb = 0.0
    fl, fr = float(m1.get("fn_L") or 0.0), float(m1.get("fn_R") or 0.0)
    if (fl + fr) > 1e-3 and int(m1.get("bilateral") or 0):
        imb = -w.w_imbalance * abs(fl - fr) / (fl + fr + 1e-6)
    time_p = -w.w_time
    switch_p = -w.w_switch if token_switched else 0.0
    total = prog + contact + slip + imb + time_p + switch_p
    bd = RewardBreakdown(
        total=float(total),
        progress=float(prog),
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
    success_12s: bool,
    success_new: bool,
    lost_after_hand: bool,
    timeout_fail: bool,
    weights: RewardWeights | None = None,
) -> tuple[float, RewardBreakdown]:
    w = weights or RewardWeights()
    term = 0.0
    if drop:
        term += w.r_drop
    elif lost_after_hand:
        term += w.r_lost_after_hand
    elif success_12s:
        term += w.r_success_12s
        if success_new:
            term += w.r_success_new_bonus
    elif timeout_fail:
        term += w.r_timeout_fail
    bd = RewardBreakdown(
        total=float(term),
        progress=0.0,
        contact=0.0,
        slip=0.0,
        imbalance=0.0,
        time=0.0,
        switch=0.0,
        terminal=float(term),
    )
    return float(term), bd


def weights_dict(w: RewardWeights) -> dict:
    return asdict(w)
