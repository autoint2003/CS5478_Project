"""Phase 4 — Matched-budget Branch A (control) vs Branch B (C curriculum) PPO pilots."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.ppo import ppo_update
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p63_transformer_ppo_c_grasp.paths import WINDOW
from experiments.precision.p63_transformer_ppo_c_grasp.rollout import collect_schedule, merge_batches
from experiments.precision.p63_transformer_critic_ppo.paths import GAE_LAMBDA, GAMMA
from experiments.precision.p64_ppo_multiround.checkpointing import load_train_state, save_train_state
from experiments.precision.p64_ppo_multiround.schedule import (
    blocked_eval_ids,
    full_train_pool,
    sample_stratified,
)
from experiments.precision.p66_c_curriculum_ppo.paths import (
    CKPT_DIR,
    CURRICULUM_C_FRAC_INITIAL,
    GROUP,
    ICS_PER_UPDATE,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P64_STEP4_OPTIM,
    SEED,
    TRAJ_PER_STEP,
    UPDATES_PER_STEP,
)


def _load_curriculum_index() -> dict[str, list[dict]]:
    man_path = OUT / "CURRICULUM_SNAPSHOT_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    by: dict[str, list[dict]] = {}
    for s in man["states"]:
        # Prefer pre-recatch stages (exclude offset 0 if we want, but keep all feasible)
        by.setdefault(s["traj_id"], []).append(s)
    return by


def collect_ic_group_curriculum(
    env,
    actor,
    critic,
    start,
    device,
    *,
    n_traj: int,
    curriculum_entries: list[dict] | None,
    curriculum_frac: float,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """Like collect_ic_group but optionally restore C curriculum states for a fraction of trajs."""
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
    ep_infos = []
    n_steps = 0
    n_curr = 0
    t0 = time.perf_counter()
    lookup_start = start

    for _ in range(n_traj):
        use_curr = (
            curriculum_entries
            and str(start.skill) == "C"
            and float(rng.random()) < float(curriculum_frac)
        )
        if use_curr:
            ent = curriculum_entries[int(rng.integers(0, len(curriculum_entries)))]
            state = pickle.loads(Path(ent["path"]).read_bytes())
            restore_full_state(env, state, lookup_start)
            n_curr += 1
        else:
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
        ep_infos.append(info)

    values_flat = list(val_buf) + [0.0]
    adv_all, ret_all = compute_gae(
        np.asarray(rew_buf, np.float32),
        np.asarray(values_flat, np.float32),
        np.asarray(done_buf, np.float32),
        GAMMA,
        GAE_LAMBDA,
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
        "n_episodes": len(ep_returns),
        "n_steps": n_steps,
        "n_curriculum": n_curr,
        "p_relaxed": succ / max(1, len(ep_infos)),
        "explained_var": explained_variance(y_pred, ret_all),
        "wall_s": time.perf_counter() - t0,
        "skill": start.skill,
        "traj_id": start.traj_id,
    }
    return batch, stats


def collect_schedule_curriculum(
    env,
    actor,
    critic,
    schedule: list,
    device,
    *,
    n_traj: int,
    curriculum_by_tid: dict[str, list[dict]],
    curriculum_frac: float,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    batches = []
    stats_list = []
    for i, st in enumerate(schedule):
        entries = curriculum_by_tid.get(st.traj_id) if st.skill == "C" else None
        # If this IC has no snapshots, fall back to all C snapshots
        if st.skill == "C" and not entries:
            flat = [e for ents in curriculum_by_tid.values() for e in ents]
            entries = flat if flat else None
        b, s = collect_ic_group_curriculum(
            env,
            actor,
            critic,
            st,
            device,
            n_traj=n_traj,
            curriculum_entries=entries,
            curriculum_frac=curriculum_frac if st.skill == "C" else 0.0,
            rng=rng,
        )
        batches.append(b)
        stats_list.append(s)
        if (i + 1) % 8 == 0 or i == 0:
            print(
                f"  rollout {i+1}/{len(schedule)} {s['skill']} "
                f"ret={s['mean_return']:.2f} p_rel={s['p_relaxed']:.3f} "
                f"curr={s['n_curriculum']} steps={s['n_steps']} ({s['wall_s']:.1f}s)",
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
        "n_curriculum_episodes": int(sum(s.get("n_curriculum") or 0 for s in stats_list)),
        "wall_s": float(sum(s["wall_s"] for s in stats_list)),
        "mean_return": float(np.mean([s["mean_return"] for s in stats_list])),
        "explained_var": float(explained_variance(batch["values"], batch["ret"])),
        "p_relaxed_by_class": {
            c: float(np.mean(v)) if v else None for c, v in by_class.items()
        },
        "overall_p_relaxed": float(np.mean([s["p_relaxed"] for s in stats_list])),
    }
    return batch, summary


def run_branch(
    *,
    branch: str,
    use_curriculum: bool,
    device: str = "cuda",
    target_sim_steps: int | None = None,
) -> dict:
    """One matched pilot ≈ 2048 traj opportunities (or matched sim steps if provided)."""
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, critic, book, rho, opt_a, opt_c, opt_note, _ = load_train_state(
        P64_STEP4_ACTOR, P64_STEP4_CRITIC, P64_STEP4_OPTIM, device_t
    )
    env = make_env(book, rho, enable_c_grasp_reward=False)
    pool = full_train_pool(SEED)
    blocked = blocked_eval_ids()
    curr_by = _load_curriculum_index() if use_curriculum else {}

    history = []
    total_steps = 0
    total_eps = 0
    total_curr = 0
    t0 = time.perf_counter()
    updates_done = 0
    # Curriculum continuations are shorter → allow extra updates to match sim steps
    max_updates = UPDATES_PER_STEP if target_sim_steps is None else max(UPDATES_PER_STEP, 12)

    for u in range(max_updates):
        if target_sim_steps is not None and total_steps >= target_sim_steps:
            break
        if target_sim_steps is None and u >= UPDATES_PER_STEP:
            break
        seed_u = SEED + 66000 + (1 if use_curriculum else 0) * 10000 + u * 17
        schedule = sample_stratified(
            pool, n_ics=ICS_PER_UPDATE, seed=seed_u, blocked=blocked
        )
        print(
            f"=== P66 {branch} update {u+1}/{max_updates} "
            f"curriculum={use_curriculum} seed={seed_u} "
            f"steps_so_far={total_steps} ===",
            flush=True,
        )
        if use_curriculum:
            batch, summary = collect_schedule_curriculum(
                env,
                actor,
                critic,
                schedule,
                device_t,
                n_traj=GROUP,
                curriculum_by_tid=curr_by,
                curriculum_frac=CURRICULUM_C_FRAC_INITIAL,
                seed=seed_u + 3,
            )
        else:
            batch, summary = collect_schedule(
                env, actor, critic, schedule, device_t, n_traj=GROUP
            )
            summary["n_curriculum_episodes"] = 0

        metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
        total_steps += summary["n_steps"]
        total_eps += summary["n_episodes"]
        total_curr += int(summary.get("n_curriculum_episodes") or 0)
        updates_done += 1
        row = {
            "branch": branch,
            "update": u + 1,
            "rollout": summary,
            "ppo": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "branch": branch,
                    "update": u + 1,
                    "p_rel": summary["overall_p_relaxed"],
                    "steps": summary["n_steps"],
                    "curr_eps": summary.get("n_curriculum_episodes"),
                    "ev": summary["explained_var"],
                    "approx_kl": metrics.get("approx_kl"),
                }
            ),
            flush=True,
        )

    wall = time.perf_counter() - t0
    tag = f"P66_{branch}"
    actor_path = CKPT_DIR / f"{tag}_actor.pt"
    critic_path = CKPT_DIR / f"{tag}_critic.pt"
    opt_path = CKPT_DIR / f"{tag}_optim.pt"
    save_train_state(
        actor_path,
        critic_path,
        opt_path,
        actor,
        critic,
        opt_a,
        opt_c,
        book,
        rho,
        step=0,
        update=updates_done,
        cumulative_traj=total_eps,
        extra={
            "branch": branch,
            "use_curriculum": use_curriculum,
            "curriculum_frac": CURRICULUM_C_FRAC_INITIAL if use_curriculum else 0.0,
            "opt_note": opt_note,
            "n_sim_steps": total_steps,
            "wall_s": wall,
        },
    )
    out = {
        "branch": branch,
        "use_curriculum": use_curriculum,
        "opt_note": opt_note,
        "n_episodes": total_eps,
        "n_sim_steps": total_steps,
        "n_curriculum_episodes": total_curr,
        "n_actor_updates": updates_done,
        "wall_s": wall,
        "gpu_hours": wall / 3600.0,
        "target_traj_opportunities": TRAJ_PER_STEP,
        "actor": str(actor_path),
        "critic": str(critic_path),
        "optim": str(opt_path),
        "history": history,
    }
    (OUT / f"PILOT_{branch}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_matched_pilots(*, device: str = "cuda") -> dict:
    """Branch A first; Branch B matched to A's simulator step count."""
    print("=== P66 Branch A — Standard PPO Control ===", flush=True)
    a = run_branch(branch="A_control", use_curriculum=False, device=device)
    print(
        f"=== P66 Branch B — Curriculum (match sim steps ≈ {a['n_sim_steps']}) ===",
        flush=True,
    )
    b = run_branch(
        branch="B_curriculum",
        use_curriculum=True,
        device=device,
        target_sim_steps=int(a["n_sim_steps"]),
    )
    pilot = {
        "A_control": {k: v for k, v in a.items() if k != "history"},
        "B_curriculum": {k: v for k, v in b.items() if k != "history"},
        "A_history": a["history"],
        "B_history": b["history"],
        "matched_sim_steps_target": a["n_sim_steps"],
        "step_match_ratio": (b["n_sim_steps"] / max(1, a["n_sim_steps"])),
    }
    (OUT / "PILOT_RESULTS.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")
    return pilot


if __name__ == "__main__":
    run_matched_pilots()
