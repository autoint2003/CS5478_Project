"""Cross-class policy drift from P62_epoch3 on fixed train-derived states."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p63_transformer_critic_ppo.paths import OUT, P62_EPOCH3, SEED
from experiments.precision.p63_transformer_critic_ppo.rollout import make_env


def _logp(actor, env, device):
    obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = actor(obs, prev)
        return torch.log_softmax(mask_handoff_logits(logits[-1].float()), dim=-1)


def _kl(p_log, q_log) -> float:
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().cpu())


def measure_drift(actor_path, *, device: str = "cuda", per_class: int = 8, tag: str = "pilot") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    a0, book, rho, _ = load_actor(P62_EPOCH3, device_t)
    a1, _, _, _ = load_actor(actor_path, device_t)
    a0.eval()
    a1.eval()
    env = make_env(book, rho)
    ds = build_p54_dataset(seed=SEED)
    by = {"A1": [], "B1": [], "C": [], "D": []}
    for st in ds["schedule"]:
        if len(by[st.skill]) < per_class:
            by[st.skill].append(st)

    rows = []
    for sk, starts in by.items():
        kls = []
        for st in starts:
            env.reset(st)
            log0 = _logp(a0, env, device_t)
            log1 = _logp(a1, env, device_t)
            kls.append(_kl(log1, log0))
        rows.append(
            {
                "class": {"A1": "A", "B1": "B", "C": "C", "D": "D"}[sk],
                "n": len(starts),
                "mean_kl_vs_p62": float(np.mean(kls)),
                "max_kl_vs_p62": float(np.max(kls)),
            }
        )
    out = {"tag": tag, "ckpt": str(actor_path), "per_class": rows}
    path = OUT / f"POLICY_DRIFT_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
