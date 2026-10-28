"""Phase 4 — Reward sanity on synthetic + fixed MuJoCo trajectories."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.precision.p50_ppo_reward_redesign.reward import (
    RewardWeights,
    potential,
    step_reward,
    terminal_reward,
    weights_dict,
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def synth_m(**kw) -> dict:
    base = {
        "e_pos_mm": 20.0,
        "theta": 15.0,
        "bilateral": 0,
        "min_pad": 0,
        "vrel": 0.2,
        "fn_L": 0.0,
        "fn_R": 0.0,
    }
    base.update(kw)
    return base


def accumulate(traj: list[dict], *, terminals: dict | None = None, switches: list[bool] | None = None) -> dict:
    w = RewardWeights()
    total = 0.0
    comps = {k: 0.0 for k in ("progress", "contact", "slip", "imbalance", "time", "switch", "terminal")}
    for i in range(len(traj) - 1):
        sw = bool(switches[i]) if switches else False
        r, bd = step_reward(traj[i], traj[i + 1], token_switched=sw, weights=w)
        total += r
        for k in ("progress", "contact", "slip", "imbalance", "time", "switch"):
            comps[k] += getattr(bd, k)
    if terminals:
        rt, bdt = terminal_reward(weights=w, **terminals)
        total += rt
        comps["terminal"] += bdt.terminal
    return {"return": float(total), "components": comps, "length": len(traj) - 1, "phi0": potential(traj[0]), "phiT": potential(traj[-1])}


def build_cases() -> list[dict]:
    cases = []
    # Successful A/B-like: error shrinks, bilateral strengthens → SUCCESS_12S
    good = [synth_m(e_pos_mm=25 - 2 * i, theta=12 - 0.8 * i, bilateral=int(i > 2), min_pad=min(i, 4), vrel=0.15 - 0.01 * i, fn_L=2.0 if i > 2 else 0, fn_R=2.0 if i > 2 else 0) for i in range(12)]
    cases.append({"name": "success_stable_bilateral", "kind": "success", **accumulate(good, terminals={"drop": False, "success_12s": True, "success_new": True, "lost_after_hand": False, "timeout_fail": False})})

    # DROP_BEFORE: brief bilateral then loss
    drop_t = [synth_m(e_pos_mm=18, theta=10, bilateral=1, min_pad=2, vrel=0.1 + 0.05 * i, fn_L=1.0, fn_R=0.5) for i in range(6)]
    drop_t.append(synth_m(e_pos_mm=40, theta=25, bilateral=0, min_pad=0, vrel=0.8))
    cases.append({"name": "drop_before", "kind": "drop", **accumulate(drop_t, terminals={"drop": True, "success_12s": False, "success_new": False, "lost_after_hand": False, "timeout_fail": False})})

    # Early unstable bilateral (shallow, asymmetric)
    shallow = [synth_m(e_pos_mm=15, theta=8, bilateral=1, min_pad=1, vrel=0.25, fn_L=3.0, fn_R=0.2) for _ in range(10)]
    cases.append({"name": "unstable_shallow_bilateral", "kind": "fail_shallow", **accumulate(shallow, terminals={"drop": False, "success_12s": False, "success_new": False, "lost_after_hand": False, "timeout_fail": True})})

    # Stable bilateral contact
    stable = [synth_m(e_pos_mm=6, theta=4, bilateral=1, min_pad=4, vrel=0.03, fn_L=2.5, fn_R=2.4) for _ in range(10)]
    cases.append({"name": "stable_bilateral_hold", "kind": "stable", **accumulate(stable, terminals={"drop": False, "success_12s": True, "success_new": False, "lost_after_hand": False, "timeout_fail": False})})

    # C controlled sliding: relative motion present but progress improves
    c_slide = []
    for i in range(12):
        c_slide.append(synth_m(e_pos_mm=18 - i, theta=10 - 0.5 * i, bilateral=1, min_pad=3, vrel=0.12, fn_L=2.0, fn_R=1.8))
    cases.append({"name": "c_controlled_sliding", "kind": "c_slide", **accumulate(c_slide, terminals={"drop": False, "success_12s": True, "success_new": True, "lost_after_hand": False, "timeout_fail": False})})

    # Working linger hack: stay in WORKING without improving → should not beat success
    linger = [synth_m(e_pos_mm=12, theta=7, bilateral=1, min_pad=3, vrel=0.08, fn_L=2, fn_R=2) for _ in range(40)]
    cases.append({"name": "working_linger_no_handoff", "kind": "hack_linger", **accumulate(linger, terminals={"drop": False, "success_12s": False, "success_new": False, "lost_after_hand": False, "timeout_fail": True})})

    # Token switch spam
    spam = [synth_m(e_pos_mm=20 - 0.2 * i, theta=10, bilateral=1, min_pad=2, vrel=0.1) for i in range(20)]
    cases.append({"name": "token_switch_spam", "kind": "hack_switch", **accumulate(spam, switches=[True] * 19, terminals={"drop": False, "success_12s": False, "success_new": False, "lost_after_hand": False, "timeout_fail": True})})

    # Timeout fail
    to = [synth_m(e_pos_mm=22, theta=14, bilateral=0, min_pad=0, vrel=0.3) for _ in range(15)]
    cases.append({"name": "timeout_fail", "kind": "timeout", **accumulate(to, terminals={"drop": False, "success_12s": False, "success_new": False, "lost_after_hand": False, "timeout_fail": True})})

    # Lost after handoff
    lost = [synth_m(e_pos_mm=8, theta=5, bilateral=1, min_pad=4, vrel=0.05) for _ in range(8)]
    cases.append({"name": "lost_after_handoff", "kind": "lost", **accumulate(lost, terminals={"drop": False, "success_12s": False, "success_new": False, "lost_after_hand": True, "timeout_fail": False})})

    return cases


def checks(cases: list[dict]) -> list[dict]:
    by = {c["name"]: c for c in cases}
    out = []

    def ok(name, cond, detail):
        out.append({"check": name, "pass": bool(cond), "detail": detail})

    ok(
        "success_gt_drop",
        by["success_stable_bilateral"]["return"] > by["drop_before"]["return"],
        f"success={by['success_stable_bilateral']['return']:.2f} drop={by['drop_before']['return']:.2f}",
    )
    ok(
        "stable_gt_shallow",
        by["stable_bilateral_hold"]["return"] > by["unstable_shallow_bilateral"]["return"],
        f"stable={by['stable_bilateral_hold']['return']:.2f} shallow={by['unstable_shallow_bilateral']['return']:.2f}",
    )
    ok(
        "c_slide_not_overpenalized",
        by["c_controlled_sliding"]["return"] > by["drop_before"]["return"],
        f"c={by['c_controlled_sliding']['return']:.2f} drop={by['drop_before']['return']:.2f}",
    )
    ok(
        "linger_lt_success",
        by["working_linger_no_handoff"]["return"] < by["success_stable_bilateral"]["return"],
        f"linger={by['working_linger_no_handoff']['return']:.2f} success={by['success_stable_bilateral']['return']:.2f}",
    )
    ok(
        "drop_strongly_negative",
        by["drop_before"]["return"] < -10,
        f"drop_return={by['drop_before']['return']:.2f}",
    )
    ok(
        "timeout_lt_success",
        by["timeout_fail"]["return"] < by["success_stable_bilateral"]["return"],
        f"timeout={by['timeout_fail']['return']:.2f}",
    )
    # unbounded length: linger 40 steps should not explode via progress (potential-based ~0)
    linger_prog = by["working_linger_no_handoff"]["components"]["progress"]
    ok("linger_progress_bounded", abs(linger_prog) < 5.0, f"linger_progress={linger_prog:.3f}")
    return out


def run_synthetic(out_dir: Path) -> dict:
    cases = build_cases()
    chk = checks(cases)
    report = {
        "generated": utc(),
        "weights": weights_dict(RewardWeights()),
        "cases": cases,
        "checks": chk,
        "all_pass": all(c["pass"] for c in chk),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REWARD_SANITY_CASES.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# REWARD_SANITY_TEST",
        "",
        f"Generated: {utc()}",
        "",
        "## Weights",
        "",
        "```json",
        json.dumps(report["weights"], indent=2),
        "```",
        "",
        "## Cases (return)",
        "",
        "| Case | Kind | Return | Terminal | Progress | Contact | Slip |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        co = c["components"]
        lines.append(
            f"| {c['name']} | {c['kind']} | {c['return']:.2f} | {co['terminal']:.1f} | "
            f"{co['progress']:.3f} | {co['contact']:.3f} | {co['slip']:.3f} |"
        )
    lines += ["", "## Checks", ""]
    for c in chk:
        lines.append(f"- [{'PASS' if c['pass'] else 'FAIL'}] **{c['check']}**: {c['detail']}")
    lines += ["", f"## Verdict: {'PASS' if report['all_pass'] else 'FAIL — fix reward before PPO'}", ""]
    (out_dir / "REWARD_SANITY_TEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
