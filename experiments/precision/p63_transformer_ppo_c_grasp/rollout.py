"""On-policy rollouts for C-grasp PPO (reuses GAE/act from critic-ppo core)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.critic import TransformerCritic
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import CGraspEnv, make_env
from experiments.precision.p63_transformer_ppo_c_grasp.paths import (
    GROUP,
    MAX_EP_STEPS,  # noqa: F401 — documented contract
    WINDOW,
)
from experiments.precision.p63_transformer_critic_ppo.paths import GAE_LAMBDA, GAMMA


def collect_ic_group(
    env: CGraspEnv,
    actor: torch.nn.Module,
    critic: TransformerCritic,
    start,
    device,
    *,
    n_traj: int = GROUP,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[dict, dict]:
    obs_buf, prev_buf, tok_buf, logp_buf, val_buf, rew_buf, done_buf = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    ep_returns = []
    ep_raw = []
    ep_c = []
    ep_infos = []
    n_steps = 0
    t0 = time.perf_counter()

    for _ in range(n_traj):
        env.reset(start)
        ep_r = 0.0
        done = False
        info: dict[str, Any] = {}
        while not done:
            tok, resid, logp, val = act_with_value(
                actor, critic, env.obs_seq, env.prev_seq, device
            )
            obs_buf.append(np.stack(env.obs_seq[-WINDOW:]).astype(np.float32))
            prev_buf.append(np.stack(env.prev_seq[-WINDOW:]).astype(np.float32))
            tok_buf.append(tok)
            logp_buf.append(logp)
            val_buf.append(val)
            r, done, info = env.step(tok, resid)
            rew_buf.append(float(r))
            done_buf.append(1.0 if done else 0.0)
            ep_r += float(r)
            n_steps += 1
        ep_returns.append(ep_r)
        ep_raw.append(float(info.get("reward_breakdown", {}).get("r_raw_sum") or ep_r))
        ep_c.append(float(info.get("reward_breakdown", {}).get("r_c_sum") or 0.0))
        ep_infos.append(info)

    values_flat = list(val_buf) + [0.0]
    adv_all, ret_all = compute_gae(
        np.asarray(rew_buf, np.float32),
        np.asarray(values_flat, np.float32),
        np.asarray(done_buf, np.float32),
        gamma,
        gae_lambda,
    )
    y_pred = np.asarray(val_buf, np.float32)
    succ = sum(int(i.get("SUCCESS_RELAXED_12S") or 0) for i in ep_infos)
    batch = {
        "obs": obs_buf,
        "prev": prev_buf,
        "tokens": np.asarray(tok_buf, np.int64),
        "logp": np.asarray(logp_buf, np.float32),
        "adv": np.asarray(adv_all, np.float32),
        "ret": np.asarray(ret_all, np.float32),
        "values": y_pred,
    }
    stats = {
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "mean_r_raw": float(np.mean(ep_raw)) if ep_raw else 0.0,
        "mean_r_c": float(np.mean(ep_c)) if ep_c else 0.0,
        "n_episodes": len(ep_returns),
        "n_steps": n_steps,
        "p_relaxed": succ / max(1, len(ep_infos)),
        "explained_var": explained_variance(y_pred, ret_all),
        "wall_s": time.perf_counter() - t0,
        "skill": start.skill,
        "traj_id": start.traj_id,
    }
    return batch, stats


def merge_batches(batches: list[dict]) -> dict:
    if not batches:
        return {}
    out = {
        "obs": [],
        "prev": [],
        "tokens": [],
        "logp": [],
        "adv": [],
        "ret": [],
        "values": [],
    }
    for b in batches:
        out["obs"].extend(b["obs"])
        out["prev"].extend(b["prev"])
        out["tokens"].append(b["tokens"])
        out["logp"].append(b["logp"])
        out["adv"].append(b["adv"])
        out["ret"].append(b["ret"])
        out["values"].append(b["values"])
    out["tokens"] = np.concatenate(out["tokens"])
    out["logp"] = np.concatenate(out["logp"])
    out["adv"] = np.concatenate(out["adv"])
    out["ret"] = np.concatenate(out["ret"])
    out["values"] = np.concatenate(out["values"])
    return out


def collect_schedule(
    env,
    actor,
    critic,
    schedule: list,
    device,
    *,
    n_traj: int = GROUP,
) -> tuple[dict, dict]:
    batches = []
    stats_list = []
    for i, st in enumerate(schedule):
        b, s = collect_ic_group(env, actor, critic, st, device, n_traj=n_traj)
        batches.append(b)
        stats_list.append(s)
        if (i + 1) % 8 == 0 or i == 0:
            print(
                f"  rollout {i+1}/{len(schedule)} {s['skill']} "
                f"ret={s['mean_return']:.2f} raw={s['mean_r_raw']:.2f} "
                f"c={s['mean_r_c']:.2f} p_rel={s['p_relaxed']:.3f} "
                f"steps={s['n_steps']} ({s['wall_s']:.1f}s)",
                flush=True,
            )
    batch = merge_batches(batches)
    by_class = {"A": [], "B": [], "C": [], "D": []}
    cmap = {"A1": "A", "B1": "B", "C": "C", "D": "D"}
    for s in stats_list:
        by_class[cmap[s["skill"]]].append(s["p_relaxed"])
    summary = {
        "n_ics": len(schedule),
        "n_episodes": int(sum(s["n_episodes"] for s in stats_list)),
        "n_steps": int(sum(s["n_steps"] for s in stats_list)),
        "wall_s": float(sum(s["wall_s"] for s in stats_list)),
        "mean_return": float(np.mean([s["mean_return"] for s in stats_list])),
        "mean_r_raw": float(np.mean([s["mean_r_raw"] for s in stats_list])),
        "mean_r_c": float(np.mean([s["mean_r_c"] for s in stats_list])),
        "explained_var": float(explained_variance(batch["values"], batch["ret"])),
        "p_relaxed_by_class": {
            c: float(np.mean(v)) if v else None for c, v in by_class.items()
        },
        "overall_p_relaxed": float(np.mean([s["p_relaxed"] for s in stats_list])),
    }
    return batch, summary
