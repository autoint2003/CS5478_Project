"""P51 mixed A/B/C/D PPO — same hyperparameters as P50; balanced episode sampling."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p26_c_curriculum_rl.ppo_m111 import (
    WINDOW,
    ActorCriticM111,
    compute_gae,
    load_actor_critic,
    save_actor_critic,
)
from experiments.precision.p50_ppo_reward_redesign.ppo_train import (
    P50PPOConfig,
    act_masked,
    load_sft_pools,
    ppo_update_p50,
    sft_loss,
)
from experiments.precision.p50_ppo_reward_redesign.reward import RewardWeights
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedRecoveryEnv, MixedStart
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import sample_balanced
from experiments.precision.p51_four_class_mixed_ppo.paths import OUT, P47T_CKPT, SEED, SKILLS
from models.m111_checkpoint import MOTION_K, load_m111
from training.m111_dataset import collate_m111


def sft_loss_per_skill(ac, pools, n_per, device, rng) -> dict[str, float]:
    out = {}
    for sk, chs in pools.items():
        if not chs:
            out[sk] = float("nan")
            continue
        samples = [chs[rng.randrange(len(chs))] for _ in range(n_per)]
        loss = sft_loss(ac, {sk: samples}, n_per, device, rng)
        out[sk] = float(loss.detach())
    return out


def token_acc_per_skill(ac, pools, device, rng, n_per=16) -> dict[str, float | None]:
    from models.m111_checkpoint import HANDOFF

    ac.eval()
    out = {}
    for sk, chs in pools.items():
        if not chs:
            out[sk] = None
            continue
        samples = [chs[rng.randrange(len(chs))] for _ in range(min(n_per, len(chs)))]
        obs, prev, tokens, _r, mask, _w = collate_m111(samples, device)
        mask = mask & (tokens != HANDOFF)
        with torch.no_grad():
            logits, _, _ = ac.forward(obs, prev)
        correct = total = 0
        for i in range(logits.shape[0]):
            active = mask[i]
            if not bool(active.any()):
                continue
            correct += int((logits[i][active].argmax(-1) == tokens[i][active]).sum().item())
            total += int(active.sum().item())
        out[sk] = (correct / total) if total else None
    ac.train()
    return out


def collect_mixed_rollout(
    ac: ActorCriticM111,
    env: MixedRecoveryEnv,
    pools: dict[str, list[MixedStart]],
    cfg: P50PPOConfig,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    device = next(ac.parameters()).device
    obs_buf, prev_buf, tok_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
    ep_returns, ep_infos = [], []
    ep_counts = {sk: 0 for sk in SKILLS}
    tr_counts = {sk: 0 for sk in SKILLS}
    steps = 0
    bootstrap = 0.0
    while steps < cfg.rollout_steps:
        st = sample_balanced(pools, rng)
        env.reset(st)
        ep_r = 0.0
        done = False
        info: dict = {}
        n_tr = 0
        while not done and steps < cfg.rollout_steps:
            tok, resid, logp, val = act_masked(ac, env.obs_seq, env.prev_seq, device)
            assert 0 <= tok < MOTION_K
            obs_buf.append(np.stack(env.obs_seq[-WINDOW:]).astype(np.float32))
            prev_buf.append(np.stack(env.prev_seq[-WINDOW:]).astype(np.float32))
            tok_buf.append(tok)
            logp_buf.append(logp)
            val_buf.append(val)
            r, done, info = env.step(tok, resid)
            rew_buf.append(r)
            done_buf.append(float(done))
            ep_r += r
            steps += 1
            n_tr += 1
        if not done:
            _, _, _, bootstrap = act_masked(ac, env.obs_seq, env.prev_seq, device)
        else:
            bootstrap = 0.0
        sk = info.get("skill") or st.skill
        ep_counts[sk] = ep_counts.get(sk, 0) + 1
        tr_counts[sk] = tr_counts.get(sk, 0) + n_tr
        ep_returns.append(ep_r)
        ep_infos.append(info)

    values_ext = list(val_buf) + [bootstrap if (done_buf and done_buf[-1] < 0.5) else 0.0]
    adv, ret = compute_gae(
        np.asarray(rew_buf, np.float32),
        np.asarray(values_ext, np.float32),
        np.asarray(done_buf, np.float32),
        cfg.gamma,
        cfg.gae_lambda,
    )
    y_pred = np.asarray(val_buf, np.float32)
    y_true = ret
    var_y = float(np.var(y_true))
    explained = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0

    for i, info in enumerate(ep_infos):
        info["ep_return"] = ep_returns[i]
    per = {}
    for sk in SKILLS:
        sub = [i for i in ep_infos if i.get("skill") == sk]
        if not sub:
            per[sk] = {"n": 0, "drop": 0.0, "s12": 0.0, "return": 0.0, "components": {}}
            continue
        per[sk] = {
            "n": len(sub),
            "drop": float(np.mean([i.get("drop", 0) for i in sub])),
            "s12": float(np.mean([i.get("SUCCESS_12S", 0) for i in sub])),
            "return": float(np.mean([i["ep_return"] for i in sub])),
            "components": {
                k: float(
                    np.mean(
                        [i.get("reward_breakdown", {}).get("components", {}).get(k, 0.0) for i in sub]
                    )
                )
                for k in ("progress", "contact", "slip", "imbalance", "time", "switch", "terminal")
            },
        }

    n_ep = sum(ep_counts.values()) or 1
    n_tr = sum(tr_counts.values()) or 1
    batch = {
        "obs": obs_buf,
        "prev": prev_buf,
        "tokens": np.asarray(tok_buf, np.int64),
        "logp": np.asarray(logp_buf, np.float32),
        "adv": adv,
        "ret": ret,
    }
    stats = {
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "n_episodes": len(ep_returns),
        "drop_rate": float(np.mean([i.get("drop", 0) for i in ep_infos])) if ep_infos else 0.0,
        "success_12s_rate": float(np.mean([i.get("SUCCESS_12S", 0) for i in ep_infos])) if ep_infos else 0.0,
        "explained_var": explained,
        "ep_frac": {sk: ep_counts[sk] / n_ep for sk in SKILLS},
        "tr_frac": {sk: tr_counts[sk] / n_tr for sk in SKILLS},
        "ep_counts": ep_counts,
        "tr_counts": tr_counts,
        "per_class": per,
    }
    return batch, stats


def eval_val_pools(ac, book, rho, val_pools, device, *, max_steps: int = 80) -> dict:
    env = MixedRecoveryEnv(book, rho, max_steps=max_steps)
    out = {}
    for sk, starts in val_pools.items():
        returns, drops, s12s = [], [], []
        for st in starts:
            env.reset(st)
            ep_r = 0.0
            done = False
            info = {}
            while not done:
                tok, resid, _, _ = act_masked(ac, env.obs_seq, env.prev_seq, device, deterministic=True)
                r, done, info = env.step(tok, resid)
                ep_r += r
            returns.append(ep_r)
            drops.append(int(info.get("drop") or 0))
            s12s.append(int(info.get("SUCCESS_12S") or 0))
        out[sk] = {
            "n": len(starts),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "drop_rate": float(np.mean(drops)) if drops else 0.0,
            "success_12s_rate": float(np.mean(s12s)) if s12s else 0.0,
        }
    return out


def checkpoint_score(val: dict, sft_acc: dict) -> tuple[float, dict]:
    """Pareto-ish: protect A/C (low drop, high SFT), reward B/D low drop; no A/C collapse."""
    detail = {}
    score = 0.0
    for sk in SKILLS:
        d = float((val.get(sk) or {}).get("drop_rate") or 1.0)
        acc = sft_acc.get(sk)
        acc_f = float(acc) if acc is not None else 0.0
        detail[sk] = {"drop": d, "acc": acc_f}
        # prefer lower drop; A/C weighted higher for preservation
        w = 1.5 if sk in ("A1", "C") else 1.0
        score += w * (1.0 - d) + 0.5 * acc_f
    # penalty if A or C SFT collapses
    for sk in ("A1", "C"):
        acc = sft_acc.get(sk)
        if acc is not None and acc < 0.85:
            score -= 2.0 * (0.85 - acc)
    return float(score), detail


def train_mixed_ppo(
    *,
    train_pools: dict[str, list[MixedStart]],
    val_pools: dict[str, list[MixedStart]],
    cfg: P50PPOConfig | None = None,
    tag: str = "P51",
) -> dict[str, Any]:
    cfg = cfg or P50PPOConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    _actor, book, saved = load_m111(P47T_CKPT, device)
    rho = saved.get("residual_rho", saved.get("res_cap"))
    ac = load_actor_critic(P47T_CKPT, book, device, rho)
    ref = ActorCriticM111(copy.deepcopy(ac.actor)).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    sft_pools = load_sft_pools("P47T")
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    env = MixedRecoveryEnv(book, rho, max_steps=cfg.max_ep_steps, weights=RewardWeights())
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hist = []
    best = {"score": -1e9, "path": None, "steps": 0}
    candidates = []
    steps = 0
    while steps < cfg.total_steps:
        batch, stats = collect_mixed_rollout(ac, env, train_pools, cfg, rng)
        metrics = ppo_update_p50(ac, ref, batch, cfg, opt, sft_pools, py_rng)
        steps += cfg.rollout_steps
        row: dict[str, Any] = {"tag": tag, "steps": steps, **stats, **metrics}

        if steps % (cfg.rollout_steps * 2) == 0 or steps >= cfg.total_steps:
            val = eval_val_pools(ac, book, rho, val_pools, device, max_steps=cfg.max_ep_steps)
            acc = token_acc_per_skill(ac, sft_pools, device, py_rng)
            sft_l = sft_loss_per_skill(ac, sft_pools, 4, device, py_rng)
            score, detail = checkpoint_score(val, acc)
            row["val"] = val
            row["sft_acc"] = acc
            row["sft_loss"] = sft_l
            row["ckpt_score"] = score
            row["ckpt_detail"] = detail
            path = ckpt_dir / f"{tag}_s{steps:05d}.pt"
            save_actor_critic(path, ac, book, rho, extra={"tag": tag, "steps": steps, "p51": True})
            cand = {
                "path": str(path),
                "steps": steps,
                "score": score,
                "val": val,
                "sft_acc": acc,
                "ep_frac": stats["ep_frac"],
                "tr_frac": stats["tr_frac"],
            }
            candidates.append(cand)
            if score >= best["score"]:
                best = {**cand, "path": str(ckpt_dir / f"{tag}_best.pt")}
                save_actor_critic(
                    Path(best["path"]), ac, book, rho, extra={"tag": tag, "steps": steps, "best": True}
                )

        hist.append(row)
        print(
            f"  {tag} steps={steps}/{cfg.total_steps} ret={stats['mean_return']:.2f} "
            f"drop={stats['drop_rate']:.2f} kl={metrics.get('kl_ref', float('nan')):.4f} "
            f"ep={stats['ep_frac']} score={row.get('ckpt_score', float('nan'))}",
            flush=True,
        )
        # always save step ckpt
        save_actor_critic(
            ckpt_dir / f"{tag}_s{steps:05d}.pt",
            ac,
            book,
            rho,
            extra={"tag": tag, "steps": steps},
        )

    return {
        "history": hist,
        "best": best,
        "candidates": candidates,
        "cfg": asdict(cfg),
        "init": str(P47T_CKPT),
    }
