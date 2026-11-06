"""Post-recatch stabilization shaping (additive; preserves P53 base reward)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from experiments.precision.p53_independent_grpo64.reward import (
    PoseRewardWeights,
    RewardBreakdown,
    step_reward as base_step_reward,
    terminal_reward as base_terminal_reward,
)


@dataclass
class StabilizationWeights:
    """Activated only after a valid recatch event (see criteria.RecatchTracker)."""

    # A — sustained bilateral potential Φ_hold ∈ [0, w_hold]
    # gamma=1 ⇒ maintaining high Φ gives r≈0 (no drain / no delay-farming from sitting).
    w_hold: float = 0.35
    gamma_hold: float = 1.0
    hold_pad_cap: int = 4
    # B — relative-motion reduction (potential on calmness)
    w_calm: float = 0.25
    gamma_calm: float = 1.0
    vrel_scale: float = 0.5  # m/s scale for exp(-v²/2σ²)
    # C — uncontrolled slip after recatch (increase in vrel only)
    w_post_slip: float = 0.15
    slip_dv_cap: float = 0.4
    # D — bilateral loss after recatch (one-shot, bounded; not stacked with full DROP)
    r_lost_contact: float = -1.5
    # E — over-grip: soft penalty when total normal force exceeds soft cap while bilateral
    w_overgrip: float = 0.08
    fn_soft_cap: float = 25.0  # Newtons-ish sim units; soft, not hard crush ban
    # Bound per-step post-recatch extras
    step_clip: float = 0.75


@dataclass
class StabBreakdown:
    hold: float
    calm: float
    post_slip: float
    lost_contact: float
    overgrip: float
    total: float


def _hold_potential(m: dict, w: StabilizationWeights) -> float:
    if int(m.get("bilateral") or 0) != 1:
        return 0.0
    pad = min(int(m.get("min_pad") or 0), w.hold_pad_cap) / float(w.hold_pad_cap)
    return float(w.w_hold * pad)


def _calm_potential(m: dict, w: StabilizationWeights) -> float:
    """Higher when relative speed is low (post-recatch only)."""
    v = float(m.get("vrel") or 0.0)
    sig = max(w.vrel_scale, 1e-6)
    return float(w.w_calm * np.exp(-(v * v) / (2.0 * sig * sig)))


def post_recatch_step_reward(
    m0: dict,
    m1: dict,
    *,
    post_recatch: bool,
    just_lost_bilateral: bool,
    weights: StabilizationWeights | None = None,
) -> tuple[float, StabBreakdown]:
    w = weights or StabilizationWeights()
    z = StabBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if not post_recatch:
        return 0.0, z

    hold = w.gamma_hold * _hold_potential(m1, w) - _hold_potential(m0, w)
    calm = w.gamma_calm * _calm_potential(m1, w) - _calm_potential(m0, w)

    dv = float(m1.get("vrel") or 0.0) - float(m0.get("vrel") or 0.0)
    post_slip = 0.0
    if int(m1.get("bilateral") or 0) or int(m0.get("bilateral") or 0):
        post_slip = -w.w_post_slip * float(np.clip(dv, 0.0, w.slip_dv_cap))

    lost = w.r_lost_contact if just_lost_bilateral else 0.0

    over = 0.0
    if int(m1.get("bilateral") or 0):
        fn = float(m1.get("fn_L") or 0.0) + float(m1.get("fn_R") or 0.0)
        excess = max(0.0, fn - w.fn_soft_cap) / max(w.fn_soft_cap, 1e-6)
        over = -w.w_overgrip * float(np.clip(excess, 0.0, 2.0))

    total = float(np.clip(hold + calm + post_slip + lost + over, -w.step_clip, w.step_clip))
    bd = StabBreakdown(
        hold=float(hold),
        calm=float(calm),
        post_slip=float(post_slip),
        lost_contact=float(lost),
        overgrip=float(over),
        total=total,
    )
    return total, bd


def combined_step_reward(
    m0: dict,
    m1: dict,
    *,
    token_switched: bool,
    post_recatch: bool,
    just_lost_bilateral: bool,
    base_weights: PoseRewardWeights | None = None,
    stab_weights: StabilizationWeights | None = None,
    use_stabilization: bool = True,
) -> tuple[float, RewardBreakdown, StabBreakdown]:
    r0, bd0 = base_step_reward(m0, m1, token_switched=token_switched, weights=base_weights)
    if use_stabilization:
        rs, bds = post_recatch_step_reward(
            m0,
            m1,
            post_recatch=post_recatch,
            just_lost_bilateral=just_lost_bilateral,
            weights=stab_weights,
        )
    else:
        rs, bds = 0.0, StabBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bd = RewardBreakdown(
        total=float(bd0.total + rs),
        pose=bd0.pose,
        contact=bd0.contact,
        slip=bd0.slip,
        imbalance=bd0.imbalance,
        time=bd0.time,
        switch=bd0.switch,
        terminal=0.0,
    )
    return float(r0 + rs), bd, bds


def stab_weights_dict(w: StabilizationWeights | None = None) -> dict:
    return asdict(w or StabilizationWeights())


# Re-export terminal unchanged
terminal_reward = base_terminal_reward
