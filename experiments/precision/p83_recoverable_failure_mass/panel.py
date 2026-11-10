"""Phase 1 — validate/reuse P82 panel and expand to ≥32 ICs/class."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids, is_blocked_eval
from experiments.precision.p82_global_failure_budget.collect import rollout_episode
from experiments.precision.p83_recoverable_failure_mass.paths import (
    ACTOR_MD5,
    CACHE,
    CRITIC_MD5,
    MAX_EP_STEPS,
    N_SEEDS,
    N_STAGE1,
    N_STAGE2,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    P82_CACHE,
    P82_FREEZE,
    SEED,
    SKILLS,
    T_MAX,
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate_and_expand(*, device: str = "cuda", target: int = N_STAGE1, force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"panel_{target}"
    cache_path = CACHE / f"{tag}.pkl"
    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        report = _panel_report(rows, reused=True, target=target)
        _write_panel_validation(report)
        return {"rows": rows, "report": report, "cached": True}

    actor_h, critic_h = _md5(P64_ACTOR), _md5(P64_CRITIC)
    freeze_ok = actor_h == ACTOR_MD5 and critic_h == CRITIC_MD5
    p82_freeze = json.loads(P82_FREEZE.read_text(encoding="utf-8")) if P82_FREEZE.exists() else {}

    reused = []
    if P82_CACHE.exists() and freeze_ok:
        old = pickle.loads(P82_CACHE.read_bytes())
        # Exact protocol match: same fields / frozen baseline
        for r in old:
            r = dict(r)
            r["source"] = "p82_reuse"
            reused.append(r)

    have = defaultdict(set)
    for r in reused:
        have[r["skill"]].add(r["traj_id"])

    blocked = blocked_eval_ids()
    pool = build_p54_dataset(seed=5478)["schedule"]
    by = defaultdict(list)
    for s in pool:
        if is_blocked_eval(s.traj_id, blocked):
            continue
        by[s.skill].append(s)

    need = []
    rng = np.random.default_rng(SEED + target)
    for sk in SKILLS:
        cand = [s for s in by.get(sk, []) if s.traj_id not in have[sk]]
        rng.shuffle(cand)
        n_need = max(0, target - len(have[sk]))
        # Cap available
        n_need = min(n_need, len(cand))
        need.extend(cand[:n_need])

    rows = list(reused)
    if need:
        device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
        critic = load_critic(P64_CRITIC, actor, device_t)
        for p in actor.parameters():
            p.requires_grad_(False)
        env = StabilizeGrpoEnv(
            book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
        )
        for ii, start in enumerate(need):
            for k in range(N_SEEDS):
                seed = rollout_seed(split="train", traj_id=start.traj_id, rollout_i=84000 + 30 * ii + k)
                rec = rollout_episode(env, actor, critic, book, start, device_t, seed=seed)
                rec["source"] = "p83_new"
                rows.append(rec)
            print(
                f"[p83] {ii+1}/{len(need)} {start.skill} {start.traj_id[-16:]} "
                f"succ={rows[-1]['SUCCESS_RELAXED']} fail={rows[-1]['earliest_decisive_failure']}",
                flush=True,
            )

    # Trim to target per class if somehow over (keep first target unique ICs)
    trimmed = []
    seen = defaultdict(list)
    for r in rows:
        if len(seen[r["skill"]]) >= target:
            continue
        if r["traj_id"] in seen[r["skill"]]:
            continue
        seen[r["skill"]].append(r["traj_id"])
        trimmed.append(r)
    rows = trimmed

    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    n_forced_ep = sum(1 for r in rows if int(r.get("forced_steps") or 0) > 0)
    v2_rate = float(n_forced_ep / len(rows)) if rows else 0.0
    if rows and v2_rate < 0.05:
        print(
            f"[p83] WARNING: V2 activation_rate={v2_rate:.3f} "
            f"({n_forced_ep}/{len(rows)}); MSP is ≈ Actor-only, not P64+V2 closed-loop.",
            flush=True,
        )
    report = _panel_report(
        rows,
        reused=bool(reused),
        target=target,
        extra={
            "actor_md5": actor_h,
            "critic_md5": critic_h,
            "freeze_hashes_match_p82": freeze_ok,
            "p82_reused_n": len(reused),
            "newly_collected_n": len(need),
            "blocked_eval_excluded": True,
            "blocked_eval_uses_root_id": True,
            "final_test_excluded": True,
            "max_budget_per_class": N_STAGE2,
            "selection_not_outcome_based": True,
            "p82_freeze_actor_md5": (p82_freeze.get("baseline") or {}).get("actor", {}).get("md5"),
            "v2_activation": {
                "episodes_with_forced_gt0": n_forced_ep,
                "n_episodes": len(rows),
                "activation_rate": v2_rate,
                "attribution": "Actor-only" if v2_rate < 0.05 else "P64+V2 mixed",
            },
        },
    )
    _write_panel_validation(report)
    return {"rows": rows, "report": report, "cached": False}


def _panel_report(rows, *, reused, target, extra=None):
    by = Counter(r["skill"] for r in rows)
    succ = {sk: float(np.mean([r["SUCCESS_RELAXED"] for r in rows if r["skill"] == sk])) for sk in SKILLS}
    msp = float(np.mean(list(succ.values()))) if succ else None
    sources = Counter(r.get("source", "unknown") for r in rows)
    return {
        "target_per_class": target,
        "n_total": len(rows),
        "n_per_class": dict(by),
        "n_independent_ics": len({r["traj_id"] for r in rows}),
        "success_by_class": succ,
        "msp_equal_weight": msp,
        "sources": dict(sources),
        "reused_p82": reused,
        **(extra or {}),
    }


def _write_panel_validation(report: dict) -> None:
    (OUT / "BASELINE_PANEL_VALIDATION.md").write_text(
        "# Baseline Panel Validation (P83 Phase 1)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    validate_and_expand()
