"""Cross-class token-distribution KL: P57-A vs P59_pass1 on fixed starts (no training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import (
    load_policy_actor,
    mask_handoff_logits,
)
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p62_grpo_optimization.paths import (
    MAX_EP_STEPS,
    OUT,
    P57A_CKPT,
    P59_PASS1,
    SEED,
    T_MAX,
)


def _dist(actor, env, device):
    obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = actor(obs, prev)
        logp = torch.log_softmax(mask_handoff_logits(logits[-1].float()), dim=-1)
    return logp


def _kl(p_log, q_log) -> float:
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().cpu())


def run_policy_drift(*, device: str = "cpu", per_class: int = 8) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    a57, book, rho, _ = load_policy_actor(P57A_CKPT, device_t)
    a59, _, _, _ = load_policy_actor(P59_PASS1, device_t)
    a57.eval()
    a59.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    ds = build_p54_dataset(seed=SEED)
    by = {"A1": [], "B1": [], "C": [], "D": []}
    for st in ds["schedule"]:
        if len(by[st.skill]) < per_class:
            by[st.skill].append(st)

    rows = []
    for sk, starts in by.items():
        kls = []
        ents57 = []
        ents59 = []
        for st in starts:
            env.reset(st)
            # initial state (pre-action) — acquisition-relevant
            log57 = _dist(a57, env, device_t)
            log59 = _dist(a59, env, device_t)
            kls.append(_kl(log59, log57))  # KL(P59 || P57)
            ents57.append(float(-(log57.exp() * log57).sum().cpu()))
            ents59.append(float(-(log59.exp() * log59).sum().cpu()))
        rows.append(
            {
                "skill": sk,
                "class": {"A1": "A", "B1": "B", "C": "C", "D": "D"}[sk],
                "n_states": len(starts),
                "mean_kl_p59_vs_p57": float(np.mean(kls)),
                "mean_entropy_p57": float(np.mean(ents57)),
                "mean_entropy_p59": float(np.mean(ents59)),
                "delta_entropy": float(np.mean(ents59) - np.mean(ents57)),
            }
        )

    out = {
        "ckpt_a": str(P57A_CKPT),
        "ckpt_b": str(P59_PASS1),
        "stage": "IC_reset_initial_obs",
        "note": (
            "KL(P59||P57) on train-schedule ICs at reset. Does not prove ABD held-out regression "
            "cause, but measures policy shift magnitude per class."
        ),
        "per_class": rows,
    }
    (OUT / "POLICY_DRIFT_ANALYSIS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# POLICY_DRIFT_ANALYSIS",
        "",
        "Token-distribution KL(P59_pass1 ∥ P57_A) at IC reset (pre-action).",
        "",
        "| Class | n | mean KL | H(P57) | H(P59) | ΔH |",
        "|-------|--:|--------:|-------:|-------:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['class']} | {r['n_states']} | {r['mean_kl_p59_vs_p57']:.4f} | "
            f"{r['mean_entropy_p57']:.3f} | {r['mean_entropy_p59']:.3f} | {r['delta_entropy']:.3f} |"
        )
    lines += [
        "",
        "If A/B KL is large relative to C, C-only training likely moved A/B behavior without "
        "adequate KL coverage on those states (consistent with P61 residual/KL audit).",
        "No KL implementation change in P62.",
        "",
    ]
    (OUT / "POLICY_DRIFT_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run_policy_drift(), indent=2))
