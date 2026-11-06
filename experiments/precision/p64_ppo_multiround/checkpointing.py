"""Actor/Critic + optimizer checkpointing for P64 multi-round resume."""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    load_actor,
    load_critic,
)
from experiments.precision.p63_transformer_critic_ppo.ppo import PPOHyper, make_optimizers
from models.m111_checkpoint import save_m111


def save_train_state(
    actor_path: Path,
    critic_path: Path,
    opt_path: Path,
    actor,
    critic,
    opt_a,
    opt_c,
    book,
    rho,
    *,
    step: int,
    update: int,
    cumulative_traj: int,
    extra: dict | None = None,
):
    actor_path = Path(actor_path)
    critic_path = Path(critic_path)
    opt_path = Path(opt_path)
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "p64": True,
        "step": step,
        "update": update,
        "cumulative_traj": cumulative_traj,
        **(extra or {}),
    }
    save_m111(actor_path, actor, book, epoch=step, rho=rho, extra=meta)
    torch.save({"critic": critic.state_dict(), "step": step, "extra": meta}, critic_path)
    torch.save(
        {
            "opt_actor": opt_a.state_dict(),
            "opt_critic": opt_c.state_dict(),
            "step": step,
            "update": update,
            "cumulative_traj": cumulative_traj,
            "hyp": {
                "actor_lr": opt_a.param_groups[0]["lr"],
                "critic_lr": opt_c.param_groups[0]["lr"],
            },
            "extra": meta,
        },
        opt_path,
    )


def load_train_state(actor_ckpt: Path, critic_ckpt: Path, opt_ckpt: Path | None, device):
    """Load Actor/Critic; restore Adam if opt_ckpt exists else fresh optimizers."""
    actor, book, rho, saved = load_actor(actor_ckpt, device)
    critic = load_critic(critic_ckpt, actor, device)
    opt_a, opt_c = make_optimizers(actor, critic, PPOHyper())
    opt_note = "fresh_adam"
    if opt_ckpt is not None and Path(opt_ckpt).is_file():
        blob = torch.load(opt_ckpt, map_location=device, weights_only=False)
        opt_a.load_state_dict(blob["opt_actor"])
        opt_c.load_state_dict(blob["opt_critic"])
        opt_note = f"restored:{opt_ckpt}"
    return actor, critic, book, rho, opt_a, opt_c, opt_note, saved
