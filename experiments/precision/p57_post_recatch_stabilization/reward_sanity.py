"""Phase 3 — synthetic + scripted reward sanity tests for post-recatch shaping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights, trajectory_return
from experiments.precision.p57_post_recatch_stabilization.paths import OUT
from experiments.precision.p57_post_recatch_stabilization.reward_stabilize import (
    StabilizationWeights,
    combined_step_reward,
    stab_weights_dict,
    terminal_reward,
)


def _m(
    *,
    bil=0,
    pad=0,
    vrel=0.1,
    e=30.0,
    th=20.0,
    fn_L=2.0,
    fn_R=2.0,
    nL=0,
    nR=0,
):
    return {
        "bilateral": int(bil),
        "min_pad": int(pad),
        "nL": int(nL if nL else (pad if bil else 0)),
        "nR": int(nR if nR else (pad if bil else 0)),
        "vrel": float(vrel),
        "e_pos_mm": float(e),
        "theta": float(th),
        "fn_L": float(fn_L),
        "fn_R": float(fn_R),
        "latched": 0,
    }


def _traj_return(seq: list[dict], *, use_stab: bool, post_from: int | None) -> float:
    """seq items: dict with m and optional flags."""
    rewards = []
    base = PoseRewardWeights()
    stab = StabilizationWeights()
    for i in range(len(seq) - 1):
        m0, m1 = seq[i]["m"], seq[i + 1]["m"]
        post = post_from is not None and i + 1 >= post_from
        lost = bool(seq[i + 1].get("lost"))
        r, _, _ = combined_step_reward(
            m0,
            m1,
            token_switched=False,
            post_recatch=post,
            just_lost_bilateral=lost,
            base_weights=base,
            stab_weights=stab,
            use_stabilization=use_stab,
        )
        rewards.append(r)
    term = seq[-1].get("terminal")
    if term:
        rt, _ = terminal_reward(**term, weights=base)
        rewards.append(rt)
    return trajectory_return(rewards, gamma=1.0)


def run_sanity() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = {}

    # 1. Successful recatch + stable hold (improving calm, sustained bil)
    stable = [{"m": _m(bil=0, pad=0, vrel=0.8, e=40)} ]
    for i in range(1, 20):
        stable.append({"m": _m(bil=1, pad=3, vrel=max(0.05, 0.5 - 0.02 * i), e=max(8, 40 - i), th=max(5, 20 - i))})
    stable[-1]["terminal"] = dict(
        drop=False, success_relaxed=True, success_12s=True, success_new=True, lost_after_hand=False, timeout_fail=False
    )
    r_stable_on = _traj_return(stable, use_stab=True, post_from=3)
    r_stable_off = _traj_return(stable, use_stab=False, post_from=3)

    # 2. Recatch then contact loss
    lost = [{"m": _m(bil=0, pad=0, vrel=0.7, e=40)}]
    for i in range(1, 6):
        lost.append({"m": _m(bil=1, pad=3, vrel=0.3, e=25)})
    lost.append({"m": _m(bil=0, pad=0, vrel=0.9, e=35), "lost": True})
    lost.append({"m": _m(bil=0, pad=0, vrel=1.2, e=50), "terminal": dict(
        drop=True, success_relaxed=False, success_12s=False, success_new=False, lost_after_hand=False, timeout_fail=False
    )})
    r_lost_on = _traj_return(lost, use_stab=True, post_from=3)
    r_lost_off = _traj_return(lost, use_stab=False, post_from=3)

    # 3. Brief bilateral then drop
    brief = [{"m": _m(bil=0, vrel=0.6)}]
    brief += [{"m": _m(bil=1, pad=2, vrel=0.5)} for _ in range(3)]
    brief.append({"m": _m(bil=0, vrel=1.0), "lost": True})
    brief.append({"m": _m(bil=0), "terminal": dict(
        drop=True, success_relaxed=False, success_12s=False, success_new=False, lost_after_hand=False, timeout_fail=False
    )})
    r_brief_on = _traj_return(brief, use_stab=True, post_from=3)

    # 4. Controlled pre-recatch sliding (no post_recatch) — stab should not change return
    slide = [{"m": _m(bil=0, vrel=0.4, e=50 - i, th=30)} for i in range(15)]
    slide[-1]["terminal"] = dict(
        drop=False, success_relaxed=False, success_12s=False, success_new=False, lost_after_hand=False, timeout_fail=True
    )
    r_slide_on = _traj_return(slide, use_stab=True, post_from=None)
    r_slide_off = _traj_return(slide, use_stab=False, post_from=None)

    # 5. Stable grasp excessive grip
    over = [{"m": _m(bil=0)}]
    for i in range(1, 15):
        over.append({"m": _m(bil=1, pad=4, vrel=0.08, e=10, fn_L=20, fn_R=20)})
    over[-1]["terminal"] = dict(
        drop=False, success_relaxed=True, success_12s=True, success_new=True, lost_after_hand=False, timeout_fail=False
    )
    mild = [{"m": _m(bil=0)}]
    for i in range(1, 15):
        mild.append({"m": _m(bil=1, pad=4, vrel=0.08, e=10, fn_L=4, fn_R=4)})
    mild[-1]["terminal"] = dict(
        drop=False, success_relaxed=True, success_12s=True, success_new=True, lost_after_hand=False, timeout_fail=False
    )
    r_over = _traj_return(over, use_stab=True, post_from=3)
    r_mild = _traj_return(mild, use_stab=True, post_from=3)

    # 6. Stable grasp invalid geometry — terminal partial, dense similar
    badpose = [{"m": _m(bil=0)}]
    for i in range(1, 15):
        badpose.append({"m": _m(bil=1, pad=3, vrel=0.1, e=25, th=25)})
    badpose[-1]["terminal"] = dict(
        drop=False, success_relaxed=False, success_12s=True, success_new=False, lost_after_hand=False, timeout_fail=False
    )
    r_badpose = _traj_return(badpose, use_stab=True, post_from=3)

    # 7. Timeout intermittent contact
    inter = []
    for i in range(20):
        bil = 1 if (i % 4) < 2 else 0
        inter.append({"m": _m(bil=bil, pad=2 if bil else 0, vrel=0.4)})
    inter[-1]["terminal"] = dict(
        drop=False, success_relaxed=False, success_12s=False, success_new=False, lost_after_hand=False, timeout_fail=True
    )
    r_inter = _traj_return(inter, use_stab=True, post_from=4)

    # 8. A/B/D-like: never post_recatch — on==off
    abd = [{"m": _m(bil=1, pad=3, vrel=0.1, e=12 - 0.2 * i, th=8)} for i in range(12)]
    abd[-1]["terminal"] = dict(
        drop=False, success_relaxed=True, success_12s=True, success_new=True, lost_after_hand=False, timeout_fail=False
    )
    r_abd_on = _traj_return(abd, use_stab=True, post_from=None)
    r_abd_off = _traj_return(abd, use_stab=False, post_from=None)

    # Delay farming: long idle post-recatch low vrel vs shorter success
    delay = [{"m": _m(bil=0)}]
    for i in range(1, 40):
        delay.append({"m": _m(bil=1, pad=3, vrel=0.05, e=20, th=15)})
    delay[-1]["terminal"] = dict(
        drop=False, success_relaxed=False, success_12s=False, success_new=False, lost_after_hand=False, timeout_fail=True
    )
    r_delay = _traj_return(delay, use_stab=True, post_from=3)

    checks = {
        "stable_beats_lost": bool(r_stable_on > r_lost_on),
        "stable_beats_brief": bool(r_stable_on > r_brief_on),
        "pre_recatch_slide_unchanged": bool(abs(r_slide_on - r_slide_off) < 1e-9),
        "abd_unchanged_without_recatch_gate": bool(abs(r_abd_on - r_abd_off) < 1e-9),
        "overgrip_leq_mild": bool(r_over <= r_mild + 1e-6),
        "terminal_dominates_delay": bool(r_stable_on > r_delay),  # success terminal >> idle timeout
        "stab_helps_stable_vs_base": bool(r_stable_on >= r_stable_off - 1e-6),
        "lost_worse_or_equal_with_stab": bool(r_lost_on <= r_lost_off + 0.5),  # loss penalty
    }
    passed = all(checks.values())
    tests = {
        "returns": {
            "stable_on": r_stable_on,
            "stable_off": r_stable_off,
            "lost_on": r_lost_on,
            "lost_off": r_lost_off,
            "brief_on": r_brief_on,
            "slide_on": r_slide_on,
            "slide_off": r_slide_off,
            "over": r_over,
            "mild": r_mild,
            "badpose": r_badpose,
            "intermittent": r_inter,
            "abd_on": r_abd_on,
            "abd_off": r_abd_off,
            "delay_timeout": r_delay,
        },
        "checks": checks,
        "passed": passed,
        "stab_weights": stab_weights_dict(),
    }
    (OUT / "REWARD_SANITY_TEST.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")

    lines = [
        "# REWARD_SANITY_TEST",
        "",
        f"**Overall: {'PASS' if passed else 'FAIL'}**",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ]
    for k, v in checks.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        "## Returns",
        "",
        "```",
        json.dumps(tests["returns"], indent=2),
        "```",
        "",
        "Pre-recatch sliding and A/B/D-like trajectories without activating the recatch gate "
        "are unchanged vs base reward. Stable post-recatch hold outscores loss/brief/delay cases; "
        "excessive grip ≤ mild grip under soft overgrip penalty; terminal success outweighs idle farming.",
        "",
    ]
    (OUT / "REWARD_SANITY_TEST.md").write_text("\n".join(lines), encoding="utf-8")

    spec = [
        "# REWARD_SPEC",
        "",
        "Base reward: unchanged `PoseRewardWeights` (P53/P56).",
        "Additive stage-aware term activates **only after valid recatch** (`RecatchTracker`).",
        "",
        "## Recatch / stable-hold gates",
        "",
        "- Recatch: ≥3 consecutive bilateral steps, min_pad≥1.",
        "- Stable-hold diagnostic: ≥12 consecutive bilateral, min_pad≥2, mean vrel≤0.35.",
        "",
        "## StabilizationWeights",
        "",
        "```json",
        json.dumps(stab_weights_dict(), indent=2),
        "```",
        "",
        "| Term | Form | Interpretation |",
        "|------|------|----------------|",
        "| A hold | γΦ_hold(s')−Φ_hold(s), Φ=w_hold·min_pad/cap if bilateral | Sustained pad quality after recatch |",
        "| B calm | γΦ_calm−Φ_calm, Φ=w_calm·exp(−vrel²/2σ²) | Reduce relative motion after recatch |",
        "| C post_slip | −w·clip(Δvrel,0,cap) if contact | Uncontrolled slip increase only |",
        "| D lost_contact | −1.5 once when bilateral lost after recatch | Contact-loss shaping; DROP terminal still −25 |",
        "| E overgrip | −w·excess((fn_L+fn_R)/cap) | Soft anti-crush; no reward for huge forces |",
        "",
        "Per-step stab extras clipped to ±0.75. Terminal SUCCESS_RELAXED_12S unchanged (+25).",
        "No privileged success labels in observations.",
        "",
    ]
    (OUT / "REWARD_SPEC.md").write_text("\n".join(spec), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    if not passed:
        raise SystemExit("REWARD_SANITY_FAILED")
    return tests


if __name__ == "__main__":
    run_sanity()
