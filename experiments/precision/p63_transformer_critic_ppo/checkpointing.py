"""Save / load Actor + Critic without mutating GRPO checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p63_transformer_critic_ppo.critic import TransformerCritic
from models.m111_checkpoint import save_m111


def load_actor(ckpt: Path, device):
    actor, book, rho, saved = load_policy_actor(ckpt, device)
    return actor, book, rho, saved


def init_critic_from_actor(actor) -> TransformerCritic:
    return TransformerCritic.from_actor(actor)


def save_pair(
    actor_path: Path,
    critic_path: Path,
    actor,
    critic: TransformerCritic,
    book,
    rho,
    *,
    epoch: int = 0,
    extra: dict | None = None,
):
    actor_path = Path(actor_path)
    critic_path = Path(critic_path)
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    save_m111(
        actor_path,
        actor,
        book,
        epoch=epoch,
        rho=rho,
        extra={"p63": True, **(extra or {})},
    )
    torch.save(
        {
            "critic": critic.state_dict(),
            "epoch": epoch,
            "extra": extra or {},
        },
        critic_path,
    )


def load_critic(path: Path, actor, device) -> TransformerCritic:
    crit = TransformerCritic.from_actor(actor).to(device)
    blob = torch.load(path, map_location=device, weights_only=False)
    crit.load_state_dict(blob["critic"])
    return crit
