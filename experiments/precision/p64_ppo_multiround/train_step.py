"""One P64 step: +2048 fresh on-policy trajectories + PPO updates + eval."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p62_grpo_optimization.eval_dev import paired_delta
from experiments.precision.p63_transformer_critic_ppo.ppo import ppo_update
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p63_transformer_ppo_c_grasp.rollout import collect_schedule
from experiments.precision.p64_ppo_multiround.checkpointing import (
    load_train_state,
    save_train_state,
)
from experiments.precision.p64_ppo_multiround.eval_p64 import eval_tag
from experiments.precision.p64_ppo_multiround.paths import (
    CKPT_DIR,
    GROUP,
    ICS_PER_UPDATE,
    KL_ABORT,
    OUT,
    P63_A_ACTOR,
    P63_A_CRITIC,
    P63_A_CUMULATIVE,
    SEED,
    TRAJ_PER_STEP,
    UPDATES_PER_STEP,
)
from experiments.precision.p64_ppo_multiround.schedule import (
    append_exposure,
    blocked_eval_ids,
    exposure_row,
    full_train_pool,
    sample_stratified,
)


def _slim_eval(e: dict) -> dict:
    return {k: v for k, v in e.items() if k != "per_ic"}


def run_step(
    *,
    step: int,
    device: str = "cuda",
    actor_in: Path | None = None,
    critic_in: Path | None = None,
    opt_in: Path | None = None,
    prev_eval: dict | None = None,
    baseline_eval: dict | None = None,
) -> dict:
    """Continue PPO for one matched +2048-trajectory step."""
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )

    actor_in = Path(actor_in or P63_A_ACTOR)
    critic_in = Path(critic_in or P63_A_CRITIC)
    actor, critic, book, rho, opt_a, opt_c, opt_note, _ = load_train_state(
        actor_in, critic_in, opt_in, device_t
    )
    env = make_env(book, rho, enable_c_grasp_reward=False)
    pool = full_train_pool(SEED)
    blocked = blocked_eval_ids()

    history = []
    total_steps = 0
    total_eps = 0
    t0 = time.perf_counter()
    aborted = None

    for u in range(UPDATES_PER_STEP):
        seed_u = SEED + 64000 + step * 1000 + u * 17
        schedule = sample_stratified(
            pool, n_ics=ICS_PER_UPDATE, seed=seed_u, blocked=blocked
        )
        exp = exposure_row(schedule, step=step, update=u + 1, n_traj=GROUP)
        append_exposure(exp)
        print(
            f"=== P64 step{step} update {u+1}/{UPDATES_PER_STEP} "
            f"({len(schedule)} ICs × {GROUP}, seed={seed_u}) "
            f"skills={exp['by_skill']} ===",
            flush=True,
        )
        batch, summary = collect_schedule(
            env, actor, critic, schedule, device_t, n_traj=GROUP
        )
        adv = batch["adv"]
        metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
        metrics["adv_mean"] = float(np.mean(adv)) if len(adv) else 0.0
        metrics["adv_std"] = float(np.std(adv)) if len(adv) else 0.0
        total_steps += summary["n_steps"]
        total_eps += summary["n_episodes"]
        row = {
            "step": step,
            "update": u + 1,
            "exposure": exp,
            "rollout": summary,
            "ppo": metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "step": step,
                    "update": u + 1,
                    "p_rel": summary["overall_p_relaxed"],
                    "ev": summary["explained_var"],
                    "approx_kl": metrics.get("approx_kl"),
                    "clipfrac": metrics.get("clipfrac"),
                    "ent": metrics.get("ent"),
                    "pg": metrics.get("pg"),
                    "vf": metrics.get("vf"),
                },
                indent=2,
            ),
            flush=True,
        )
        if not np.isfinite(metrics.get("pg", 0)) or not np.isfinite(metrics.get("vf", 0)):
            aborted = "non_finite_loss"
            break
        if metrics.get("approx_kl", 0) > KL_ABORT:
            aborted = "kl_explosion"
            break

    wall = time.perf_counter() - t0
    cum = P63_A_CUMULATIVE + step * TRAJ_PER_STEP
    tag = f"P64_step{step}"
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
        step=step,
        update=len(history),
        cumulative_traj=cum,
        extra={"opt_init": opt_note, "aborted": aborted},
    )

    print(f"=== Dev eval {tag} ===", flush=True)
    cur = eval_tag(tag, actor_path, device=device)
    result = {
        "tag": tag,
        "step": step,
        "opt_note": opt_note,
        "aborted": aborted,
        "total_episodes": total_eps,
        "total_steps": total_steps,
        "wall_s": wall,
        "gpu_hours_approx": wall / 3600.0,
        "cumulative_traj_after_p63a": cum,
        "actor": str(actor_path),
        "critic": str(critic_path),
        "optim": str(opt_path),
        "eval": _slim_eval(cur),
        "eval_full_per_ic": cur.get("per_ic"),
        "history": history,
    }
    if baseline_eval is not None:
        result["paired_vs_p62"] = paired_delta(baseline_eval, cur)
    if prev_eval is not None:
        result["paired_vs_prev"] = paired_delta(prev_eval, cur)

    (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _append_training_log(result)
    return result


def _append_training_log(result: dict) -> None:
    path = OUT / "TRAINING_LOGS.json"
    if path.is_file():
        blob = json.loads(path.read_text(encoding="utf-8"))
    else:
        blob = {"steps": []}
    slim = {k: v for k, v in result.items() if k not in ("history", "eval_full_per_ic")}
    slim["update_rows"] = [
        {
            "update": h["update"],
            "rollout": h["rollout"],
            "ppo": h["ppo"],
            "exposure_by_skill": h["exposure"]["by_skill"],
        }
        for h in result.get("history") or []
    ]
    # replace same step if re-run
    blob["steps"] = [s for s in blob["steps"] if s.get("step") != result["step"]]
    blob["steps"].append(slim)
    blob["steps"].sort(key=lambda s: s["step"])
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
