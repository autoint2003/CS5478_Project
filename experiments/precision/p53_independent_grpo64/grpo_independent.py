"""Independent GRPO-G: G full stochastic rollouts from identical IC (no Tree Branching)."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import build_pools, sample_balanced
from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.paths import (
    CKPT_DIR,
    CLIP,
    GAMMA_RETURN,
    GROUP_SIZE,
    KL_COEF,
    LR,
    MAX_EP_STEPS,
    OUT,
    P47T_CKPT,
    SEED,
    SKILLS,
    T_MAX,
    TEMPERATURE,
)
from experiments.precision.p53_independent_grpo64.policy_sample import (
    categorical_kl_masked,
    load_policy_actor,
    sample_step,
    token_logprob,
)
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights, trajectory_return
from models.m111_checkpoint import save_m111
from training.grpo_math import advantages


@dataclass
class GrpoConfig:
    group: int = GROUP_SIZE
    max_ep_steps: int = MAX_EP_STEPS
    t_max: int = T_MAX
    kl_coef: float = KL_COEF
    clip: float = CLIP
    lr: float = LR
    temperature: float = TEMPERATURE
    gamma_return: float = GAMMA_RETURN
    updates: int = 8
    seed: int = SEED
    device: str = "cpu"
    per_class_train: int = 8
    per_class_val: int = 2


def _group_advantage(returns: list[float | None]) -> tuple[np.ndarray, float, bool, int]:
    """Normalize within IC group; None/invalid → excluded. Zero-variance → zeros, skip update."""
    keep_idx = [i for i, r in enumerate(returns) if r is not None]
    out = np.zeros(len(returns), np.float64)
    if len(keep_idx) < 2:
        return out, 0.0, True, 0
    vals = np.asarray([returns[i] for i in keep_idx], np.float64)
    adv, std, skipped = advantages(vals)
    for j, i in enumerate(keep_idx):
        out[i] = adv[j]
    nonzero = int(np.sum(np.abs(out) > 1e-8))
    return out, float(std), bool(skipped), nonzero


def rollout_one(
    env: GrpoRecoveryEnv,
    actor,
    start,
    device,
    *,
    temperature: float,
    gamma_return: float,
) -> dict[str, Any]:
    """One independent stochastic trajectory from restored IC. No branching."""
    env.reset(start)
    records: list[dict] = []
    rewards: list[float] = []
    done = False
    info: dict = {}
    while not done:
        tok, resid, logp_beh, meta = sample_step(
            actor, env.obs_seq, env.prev_seq, device, temperature=temperature
        )
        obs_w = np.stack(env.obs_seq[-160:]).astype(np.float32)
        prev_w = np.stack(env.prev_seq[-160:]).astype(np.float32)
        r, done, info = env.step(tok, resid)
        records.append(
            {
                "obs": obs_w,
                "prev": prev_w,
                "token": int(tok),
                "resid": resid.astype(np.float32),
                "logp_behavior": float(logp_beh),
                "entropy": float(meta["entropy"]),
            }
        )
        rewards.append(float(r))
    # Mask: no records during nominal continuation (env stops at HANDOFF/DROP/TIMEOUT)
    ret = trajectory_return(rewards, gamma=gamma_return)
    return {
        "records": records,
        "rewards": rewards,
        "return": float(ret),
        "truncated": False,
        "info": info,
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "SUCCESS_NEW": int(info.get("SUCCESS_NEW") or 0),
        "drop": int(info.get("drop") or 0),
        "handoff": int(info.get("handoff") or 0),
        "n_steps": len(records),
    }


def independent_group(
    env: GrpoRecoveryEnv,
    actor,
    start,
    device,
    *,
    group: int,
    temperature: float,
    gamma_return: float,
) -> list[dict[str, Any]]:
    """Exactly `group` complete independent trajectories from the same IC."""
    members = []
    for _ in range(int(group)):
        members.append(
            rollout_one(
                env, actor, start, device, temperature=temperature, gamma_return=gamma_return
            )
        )
    assert len(members) == int(group)
    return members


def _mean_token_logprob_batched(actor, records: list[dict], device, *, mb: int = 32) -> torch.Tensor:
    """Length-normalized mean logp; batched windows for speed (same math as per-step)."""
    from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits

    logps = []
    for start in range(0, len(records), mb):
        chunk = records[start : start + mb]
        lengths = [int(rec["obs"].shape[0]) for rec in chunk]
        tmax = max(lengths)
        b = len(chunk)
        o = torch.zeros(b, tmax, 39, device=device, dtype=torch.float32)
        p = torch.zeros(b, tmax, 7, device=device, dtype=torch.float32)
        toks = torch.zeros(b, device=device, dtype=torch.long)
        for j, rec in enumerate(chunk):
            ti = lengths[j]
            o[j, :ti] = torch.as_tensor(rec["obs"], device=device, dtype=torch.float32)
            p[j, :ti] = torch.as_tensor(rec["prev"], device=device, dtype=torch.float32)
            toks[j] = int(rec["token"])
        logits, _res, _act = actor(o, p)
        # last valid step per row
        last = torch.stack([logits[j, lengths[j] - 1] for j in range(b)])
        last = mask_handoff_logits(last.float())
        lp = torch.log_softmax(last, dim=-1).gather(1, toks.view(-1, 1)).squeeze(1)
        logps.append(lp)
    return torch.cat(logps, dim=0).mean()


def grpo_loss_from_group(
    actor,
    ref_actor,
    members: list[dict],
    adv: np.ndarray,
    device,
    *,
    clip: float,
    kl_coef: float,
) -> tuple[torch.Tensor | None, dict]:
    terms = []
    ents = []
    for member, a in zip(members, adv):
        if abs(float(a)) < 1e-8 or not member["records"]:
            continue
        # Length-normalized mean logp — avoids overweighting long trajectories
        logp = _mean_token_logprob_batched(actor, member["records"], device)
        behavior = float(np.mean([rec["logp_behavior"] for rec in member["records"]]))
        ratio = torch.exp(torch.clamp(logp - behavior, -20.0, 20.0))
        unclipped = ratio * float(a)
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * float(a)
        terms.append(-torch.minimum(unclipped, clipped))
        ents.append(float(np.mean([rec["entropy"] for rec in member["records"]])))
    if not terms:
        return None, {"skipped": True, "n_terms": 0}
    surrogate = torch.stack(terms).mean()
    # KL on first available record (reference frozen P47T)
    rec0 = next(m["records"][0] for m in members if m["records"])
    kl = categorical_kl_masked(actor, ref_actor, rec0["obs"], rec0["prev"], device)
    loss = surrogate + float(kl_coef) * kl
    return loss, {
        "skipped": False,
        "n_terms": len(terms),
        "surrogate": float(surrogate.detach()),
        "kl": float(kl.detach()),
        "entropy_mean": float(np.mean(ents)) if ents else 0.0,
    }


def train_independent_grpo(cfg: GrpoConfig | None = None, *, tag: str = "pilot") -> dict:
    cfg = cfg or GrpoConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    actor, book, rho, _saved = load_policy_actor(P47T_CKPT, device)
    ref_actor, _, _, _ = load_policy_actor(P47T_CKPT, device)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    weights = PoseRewardWeights()
    env = GrpoRecoveryEnv(book, rho, max_steps=cfg.max_ep_steps, t_max=cfg.t_max, weights=weights)

    pools = build_pools(per_class_train=cfg.per_class_train, per_class_val=cfg.per_class_val)
    train_pools = pools["train"]
    rng = np.random.default_rng(cfg.seed)

    log_rows: list[dict] = []
    best_val = -1.0
    best_path = CKPT_DIR / f"P53_{tag}_best.pt"
    # Round-robin class allocation for balanced A/B/C/D coverage
    skill_cycle = [sk for sk in SKILLS if train_pools.get(sk)]

    for update in range(1, int(cfg.updates) + 1):
        t0 = time.perf_counter()
        if skill_cycle:
            sk = skill_cycle[(update - 1) % len(skill_cycle)]
            cand = train_pools[sk]
            start = cand[int(rng.integers(0, len(cand)))]
        else:
            start = sample_balanced(train_pools, rng)
        members = independent_group(
            env,
            actor,
            start,
            device,
            group=cfg.group,
            temperature=cfg.temperature,
            gamma_return=cfg.gamma_return,
        )
        returns = [m["return"] for m in members]
        adv, std, skipped, n_nz = _group_advantage(returns)
        n_succ = int(sum(m["SUCCESS_RELAXED_12S"] for m in members))
        n_drop = int(sum(m["drop"] for m in members))
        n_hand = int(sum(m["handoff"] for m in members))
        token_hist = np.zeros(112, np.int64)
        for m in members:
            for rec in m["records"]:
                token_hist[int(rec["token"])] += 1

        opt.zero_grad(set_to_none=True)
        loss_v = None
        kl_v = None
        ent_v = None
        if not skipped:
            loss, meta = grpo_loss_from_group(
                actor, ref_actor, members, adv, device, clip=cfg.clip, kl_coef=cfg.kl_coef
            )
            if loss is not None:
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
                if np.isfinite(gn):
                    opt.step()
                loss_v = float(loss.detach())
                kl_v = float(meta["kl"])
                ent_v = float(meta["entropy_mean"])
            else:
                skipped = True

        row = {
            "update": update,
            "skill": start.skill,
            "traj_id": start.traj_id,
            "group": cfg.group,
            "n_members": len(members),
            "n_success_relaxed": n_succ,
            "n_success_12s": int(sum(m["SUCCESS_12S"] for m in members)),
            "n_drop": n_drop,
            "n_handoff": n_hand,
            "return_mean": float(np.mean(returns)),
            "return_std": float(std),
            "nonzero_adv": n_nz,
            "skipped": int(skipped),
            "loss": loss_v,
            "kl": kl_v,
            "entropy": ent_v,
            "top_tokens": [int(i) for i in np.argsort(-token_hist)[:8]],
            "elapsed_s": time.perf_counter() - t0,
        }
        log_rows.append(row)
        print(json.dumps({k: row[k] for k in row if k != "top_tokens"}), flush=True)

        # periodic checkpoint
        if update % max(1, cfg.updates // 4) == 0 or update == cfg.updates:
            path = CKPT_DIR / f"P53_{tag}_u{update:04d}.pt"
            save_m111(
                path,
                actor,
                book,
                epoch=update,
                rho=rho,
                extra={
                    "p53": True,
                    "tag": tag,
                    "init": str(P47T_CKPT),
                    "group": cfg.group,
                    "sampling": "independent",
                },
            )
            # crude val: 1 rollout × val IC per class → fraction SUCCESS_RELAXED
            val_hits = 0
            val_n = 0
            for sk in SKILLS:
                for st in pools["val"].get(sk) or []:
                    m = rollout_one(
                        env,
                        actor,
                        st,
                        device,
                        temperature=cfg.temperature,
                        gamma_return=cfg.gamma_return,
                    )
                    val_hits += int(m["SUCCESS_RELAXED_12S"])
                    val_n += 1
            val_rate = val_hits / max(val_n, 1)
            if val_rate >= best_val:
                best_val = val_rate
                save_m111(
                    best_path,
                    actor,
                    book,
                    epoch=update,
                    rho=rho,
                    extra={"p53": True, "tag": tag, "val_relaxed": best_val, "best": True},
                )

    report = {
        "cfg": asdict(cfg),
        "init": str(P47T_CKPT),
        "log": log_rows,
        "best_val_relaxed_1roll": best_val,
        "best_ckpt": str(best_path) if best_path.exists() else None,
        "note": "Checkpoint selection uses independent val ICs (not held-out). Primary metric is MSP.",
    }
    (OUT / f"train_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
