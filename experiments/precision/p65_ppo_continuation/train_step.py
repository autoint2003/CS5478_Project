"""P65 PPO increment (+2048 traj) continuing from a checkpoint."""

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
from experiments.precision.p64_ppo_multiround.schedule import (
    blocked_eval_ids,
    exposure_row,
    full_train_pool,
    sample_stratified,
)
from experiments.precision.p65_ppo_continuation.paths import (
    CKPT_DIR,
    GROUP,
    ICS_PER_UPDATE,
    OUT,
    P64_CUMULATIVE,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P64_STEP4_OPTIM,
    SEED,
    TRAJ_PER_STEP,
    UPDATES_PER_STEP,
)


def _append_exposure(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "IC_EXPOSURE.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_p65_step(
    *,
    step_name: str,
    step_index: int,
    device: str = "cuda",
    actor_in: Path | None = None,
    critic_in: Path | None = None,
    opt_in: Path | None = None,
    cumulative_before: int = P64_CUMULATIVE,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor_in = Path(actor_in or P64_STEP4_ACTOR)
    critic_in = Path(critic_in or P64_STEP4_CRITIC)
    opt_in = Path(opt_in or P64_STEP4_OPTIM)

    actor, critic, book, rho, opt_a, opt_c, opt_note, _ = load_train_state(
        actor_in, critic_in, opt_in, device_t
    )
    assert "restored" in opt_note or opt_in.is_file(), opt_note
    env = make_env(book, rho, enable_c_grasp_reward=False)
    pool = full_train_pool(SEED)
    blocked = blocked_eval_ids()

    history = []
    total_steps = total_eps = 0
    aborted = None
    t0 = time.perf_counter()

    for u in range(UPDATES_PER_STEP):
        seed_u = SEED + 65000 + step_index * 1000 + u * 17
        schedule = sample_stratified(
            pool, n_ics=ICS_PER_UPDATE, seed=seed_u, blocked=blocked
        )
        exp = exposure_row(schedule, step=step_index, update=u + 1, n_traj=GROUP)
        exp["experiment"] = "p65"
        _append_exposure(exp)
        print(
            f"=== {step_name} update {u+1}/{UPDATES_PER_STEP} "
            f"({len(schedule)}×{GROUP}, seed={seed_u}) {exp['by_skill']} ===",
            flush=True,
        )
        batch, summary = collect_schedule(
            env, actor, critic, schedule, device_t, n_traj=GROUP
        )
        metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
        metrics["adv_mean"] = float(np.mean(batch["adv"])) if len(batch["adv"]) else 0.0
        metrics["adv_std"] = float(np.std(batch["adv"])) if len(batch["adv"]) else 0.0
        total_steps += summary["n_steps"]
        total_eps += summary["n_episodes"]
        history.append({"update": u + 1, "exposure": exp, "rollout": summary, "ppo": metrics})
        print(
            json.dumps(
                {
                    "step": step_name,
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
        if metrics.get("approx_kl", 0) > 0.5:
            aborted = "kl_explosion"
            break

    wall = time.perf_counter() - t0
    cum = cumulative_before + TRAJ_PER_STEP
    actor_path = CKPT_DIR / f"{step_name}_actor.pt"
    critic_path = CKPT_DIR / f"{step_name}_critic.pt"
    opt_path = CKPT_DIR / f"{step_name}_optim.pt"
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
        step=step_index,
        update=len(history),
        cumulative_traj=cum,
        extra={
            "p65": True,
            "step_name": step_name,
            "opt_init": opt_note,
            "aborted": aborted,
            "c_grasp_reward": False,
        },
    )

    print(f"=== Dev eval {step_name} ===", flush=True)
    cur = eval_tag(step_name, actor_path, device=device)
    result = {
        "tag": step_name,
        "step_index": step_index,
        "opt_note": opt_note,
        "aborted": aborted,
        "total_episodes": total_eps,
        "total_steps": total_steps,
        "wall_s": wall,
        "gpu_hours_approx": wall / 3600.0,
        "cumulative_traj": cum,
        "actor": str(actor_path),
        "critic": str(critic_path),
        "optim": str(opt_path),
        "eval": {k: v for k, v in cur.items() if k != "per_ic"},
        "eval_full_per_ic": cur.get("per_ic"),
        "history": history,
    }
    (OUT / f"{step_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _append_training_log(result)
    return result


def _append_training_log(result: dict) -> None:
    path = OUT / "TRAINING_LOGS.json"
    blob = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"steps": []}
    slim = {k: v for k, v in result.items() if k not in ("history", "eval_full_per_ic")}
    slim["update_rows"] = [
        {"update": h["update"], "rollout": h["rollout"], "ppo": h["ppo"]}
        for h in result.get("history") or []
    ]
    blob["steps"] = [s for s in blob["steps"] if s.get("tag") != result["tag"]]
    blob["steps"].append(slim)
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
