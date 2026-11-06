"""Offline reward audit: success vs failure return ordering under C grasp events."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    init_critic_from_actor,
    load_actor,
)
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p63_transformer_ppo_c_grasp.paths import OUT, P62_EPOCH3, SEED


def audit_c_returns(*, device: str = "cpu", n_ics: int = 8, n_traj: int = 4) -> dict:
    """Roll short C episodes with grasp reward; check exploitation risk."""
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P62_EPOCH3, device_t)
    critic = init_critic_from_actor(actor).to(device_t)
    ds = build_p54_dataset(seed=SEED)
    c_starts = [s for s in ds["schedule"] if s.skill == "C"][:n_ics]
    env = make_env(book, rho, enable_c_grasp_reward=True)

    rows = []
    for st in c_starts:
        for _ in range(n_traj):
            env.reset(st)
            done = False
            info = {}
            while not done:
                tok, resid, _, _ = act_with_value(
                    actor, critic, env.obs_seq, env.prev_seq, device_t
                )
                _, done, info = env.step(tok, resid)
            bd = info.get("reward_breakdown") or {}
            rows.append(
                {
                    "traj_id": st.traj_id,
                    "success": int(info.get("SUCCESS_RELAXED_12S") or 0),
                    "r_raw": float(bd.get("r_raw_sum") or 0.0),
                    "r_c": float(bd.get("r_c_sum") or 0.0),
                    "r_total": float(bd.get("r_raw_sum") or 0.0) + float(bd.get("r_c_sum") or 0.0),
                    "events": list(bd.get("c_events") or []),
                    "handoff_reason": info.get("handoff_reason"),
                }
            )

    succ = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    mean_succ = float(np.mean([r["r_total"] for r in succ])) if succ else None
    mean_fail = float(np.mean([r["r_total"] for r in fail])) if fail else None
    # Exploitation heuristic: top-quartile fails beat mean success
    exploit = False
    detail = ""
    if succ and fail:
        thr = float(np.percentile([r["r_total"] for r in fail], 75))
        exploit = thr > mean_succ
        detail = f"fail_p75={thr:.3f} vs mean_succ={mean_succ:.3f}"
    report = {
        "n_rows": len(rows),
        "n_success": len(succ),
        "n_fail": len(fail),
        "mean_total_success": mean_succ,
        "mean_total_fail": mean_fail,
        "mean_r_c_success": float(np.mean([r["r_c"] for r in succ])) if succ else None,
        "mean_r_c_fail": float(np.mean([r["r_c"] for r in fail])) if fail else None,
        "exploitation_risk": exploit,
        "detail": detail,
        "rows_sample": rows[:12],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REWARD_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# REWARD_AUDIT",
        "",
        "C-only grasp event reward on frozen P62_epoch3 rollouts.",
        "",
        f"- n={report['n_rows']} (succ={report['n_success']}, fail={report['n_fail']})",
        f"- mean total return success={mean_succ} fail={mean_fail}",
        f"- mean r_c success={report['mean_r_c_success']} fail={report['mean_r_c_fail']}",
        f"- exploitation_risk={exploit} ({detail})",
        "",
        "Positive event budget capped at +5; events once-only.",
        "Privileged geometry used for reward only — not in Actor/Critic obs.",
        "",
    ]
    if exploit:
        lines.append("**WARNING:** high-reward failures exceed mean success — stop this reward branch.")
    (OUT / "REWARD_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(audit_c_returns(device=args.device), indent=2)[:2500])
