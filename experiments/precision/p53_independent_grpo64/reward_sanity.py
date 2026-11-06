"""Reward sanity: synthetic metric trajectories (no MuJoCo required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.paths import OUT
from experiments.precision.p53_independent_grpo64.reward import (
    PoseRewardWeights,
    pose_potential,
    step_reward,
    terminal_reward,
    trajectory_return,
)


def _m(e_pos, theta, *, bil=1, min_pad=4, vrel=0.01, fn_L=1.0, fn_R=1.0, latched=0):
    return {
        "e_pos_mm": float(e_pos),
        "theta": float(theta),
        "bilateral": int(bil),
        "min_pad": int(min_pad),
        "vrel": float(vrel),
        "fn_L": float(fn_L),
        "fn_R": float(fn_R),
        "latched": int(latched),
    }


def accumulate(seq: list[dict], *, terminal_kwargs: dict, n_switch: int = 0) -> dict:
    w = PoseRewardWeights()
    rewards = []
    pose_sum = 0.0
    for i in range(len(seq) - 1):
        r, bd = step_reward(seq[i], seq[i + 1], token_switched=(i < n_switch), weights=w)
        rewards.append(r)
        pose_sum += bd.pose
    rt, bd_t = terminal_reward(**terminal_kwargs, weights=w)
    rewards.append(rt)
    return {
        "return": trajectory_return(rewards, gamma=1.0),
        "pose_sum": pose_sum,
        "terminal": bd_t.terminal,
        "n_steps": len(rewards),
    }


def run_sanity() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # Successful: pose improves to target then SUCCESS_RELAXED
    succ = [_m(40 - 4 * t, 25 - 2 * t, vrel=0.02) for t in range(10)]
    succ.append(_m(5, 4, vrel=0.01))
    r_succ = accumulate(
        succ,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=True,
            success_12s=True,
            success_new=True,
            lost_after_hand=False,
            timeout_fail=False,
        ),
    )
    # Stochastic success-like but slightly worse pose path
    stoch = [_m(45 - 3.5 * t, 28 - 1.8 * t, vrel=0.03) for t in range(10)]
    stoch.append(_m(8, 6, vrel=0.015))
    r_stoch = accumulate(
        stoch,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=True,
            success_12s=True,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=False,
        ),
    )
    # DROP_BEFORE mid-way
    drop_seq = [_m(40 - 2 * t, 20, vrel=0.05) for t in range(5)]
    drop_seq.append(_m(30, 20, latched=1, bil=0, min_pad=0))
    r_drop = accumulate(
        drop_seq,
        terminal_kwargs=dict(
            drop=True,
            success_relaxed=False,
            success_12s=False,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=False,
        ),
    )
    # Stable wrong position (bilateral good, e_pos large)
    wrong_pos = [_m(30, 5, vrel=0.01) for _ in range(12)]
    r_wp = accumulate(
        wrong_pos,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=False,
            success_12s=True,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=False,
        ),
    )
    # Stable wrong angle
    wrong_ang = [_m(5, 35, vrel=0.01) for _ in range(12)]
    r_wa = accumulate(
        wrong_ang,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=False,
            success_12s=True,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=False,
        ),
    )
    # Controlled C sliding: vrel decreases while pose slowly improves
    slide = []
    for t in range(12):
        slide.append(_m(25 - t, 18 - 0.5 * t, vrel=0.08 - 0.005 * t, bil=1, min_pad=3))
    r_slide = accumulate(
        slide,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=True,
            success_12s=True,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=False,
        ),
    )
    # Premature HANDOFF: still far, TIMEOUT-like fail after short path
    prem = [_m(35, 22, bil=1, min_pad=2, vrel=0.04) for _ in range(4)]
    r_prem = accumulate(
        prem,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=False,
            success_12s=False,
            success_new=False,
            lost_after_hand=True,
            timeout_fail=False,
        ),
    )
    # Timeout without success
    to = [_m(20, 15, vrel=0.02) for _ in range(15)]
    r_to = accumulate(
        to,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=False,
            success_12s=False,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=True,
        ),
    )
    # Stationary: Φ constant ⇒ pose reward ~0; time cost accumulates
    stat = [_m(20, 15, vrel=0.01) for _ in range(40)]
    r_stat = accumulate(
        stat,
        terminal_kwargs=dict(
            drop=False,
            success_relaxed=False,
            success_12s=False,
            success_new=False,
            lost_after_hand=False,
            timeout_fail=True,
        ),
    )
    # Max pose improvement alone cannot beat drop magnitude
    w = PoseRewardWeights()
    phi_lo = pose_potential(_m(100, 90), w)
    phi_hi = pose_potential(_m(0, 0), w)
    max_pose_delta = w.gamma_phi * phi_hi - phi_lo
    drop_mag = abs(w.r_drop)

    checks = {
        "succ_gt_drop": r_succ["return"] > r_drop["return"],
        "succ_gt_timeout": r_succ["return"] > r_to["return"],
        "succ_gt_wrong_pos": r_succ["return"] > r_wp["return"],
        "succ_gt_wrong_ang": r_succ["return"] > r_wa["return"],
        "succ_gt_premature": r_succ["return"] > r_prem["return"],
        "stoch_succ_gt_drop": r_stoch["return"] > r_drop["return"],
        "slide_succ_gt_drop": r_slide["return"] > r_drop["return"],
        "pose_cannot_outweigh_drop": max_pose_delta < drop_mag,
        "stationary_not_unlimited": r_stat["return"] < 0.0 and r_stat["pose_sum"] < 0.5,
        "terminal_dominates": r_succ["terminal"] > abs(r_succ["pose_sum"]),
        "controlled_slide_not_heavily_punished": r_slide["return"] > r_to["return"],
    }
    report = {
        "returns": {
            "expert_like_success": r_succ,
            "stochastic_success": r_stoch,
            "drop_before": r_drop,
            "stable_wrong_pos": r_wp,
            "stable_wrong_ang": r_wa,
            "controlled_c_slide": r_slide,
            "premature_handoff": r_prem,
            "timeout": r_to,
            "stationary_timeout": r_stat,
        },
        "max_single_step_pose_delta": float(max_pose_delta),
        "drop_penalty_mag": float(drop_mag),
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    path = OUT / "reward_sanity.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_sanity()
    print("all_pass", rep["all_pass"])
    print(json.dumps(rep["checks"], indent=2))
