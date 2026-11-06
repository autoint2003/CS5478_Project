"""Seeded Independent GRPO-style rollouts for reproducible joint evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights, trajectory_return
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.paths import (
    MAX_EP_STEPS,
    T_MAX,
    TEMPERATURE,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all


def make_eval_env(book, rho) -> StabilizeGrpoEnv:
    # use_stabilization=False ⇒ original P57-A / P53 reward path
    return StabilizeGrpoEnv(
        book,
        rho,
        max_steps=MAX_EP_STEPS,
        t_max=T_MAX,
        weights=PoseRewardWeights(),
        use_stabilization=False,
    )


def load_actor(ckpt, device):
    actor, book, rho, saved = load_policy_actor(ckpt, device)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor, book, rho, saved


def rollout_seeded(
    env: StabilizeGrpoEnv,
    actor,
    start: MixedStart,
    device,
    *,
    seed: int,
    temperature: float = TEMPERATURE,
    record_trace: bool = False,
    deterministic_torch: bool = True,
) -> dict[str, Any]:
    """One full episode from original IC with freshly applied seed (no RNG carry-over)."""
    seed_meta = seed_all(seed, deterministic_torch=deterministic_torch)
    env.reset(start)
    obs0 = np.asarray(env.obs_seq[-1], np.float32).copy()
    qpos0 = np.asarray(env.env.sim.data.qpos, float).copy()
    records: list[dict] = []
    rewards: list[float] = []
    actions: list[dict] = []
    qpos_trace: list[np.ndarray] = []
    done = False
    info: dict = {}
    while not done:
        tok, resid, logp_beh, meta = sample_step(
            actor, env.obs_seq, env.prev_seq, device, temperature=temperature
        )
        if record_trace:
            actions.append(
                {
                    "token": int(tok),
                    "resid": resid.astype(np.float32).tolist(),
                    "logp": float(logp_beh),
                    "entropy": float(meta["entropy"]),
                }
            )
        r, done, info = env.step(int(tok), resid)
        rewards.append(float(r))
        if record_trace:
            qpos_trace.append(np.asarray(env.env.sim.data.qpos, float).copy())
        records.append({"token": int(tok), "logp_behavior": float(logp_beh)})

    cd = (info or {}).get("c_diag") or {}
    out = {
        "seed": int(seed),
        "seed_meta": seed_meta,
        "traj_id": start.traj_id,
        "skill": start.skill,
        "SUCCESS_RELAXED_12S": int((info or {}).get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int((info or {}).get("SUCCESS_12S") or 0),
        "SUCCESS_NEW": int((info or {}).get("SUCCESS_NEW") or 0),
        "drop": int((info or {}).get("drop") or 0),
        "handoff": int((info or {}).get("handoff") or 0),
        "truncated": int((info or {}).get("truncated") or 0),
        "fail_reason": (info or {}).get("fail_reason"),
        "handoff_reason": (info or {}).get("handoff_reason"),
        "n_steps": len(records),
        "return": float(trajectory_return(rewards, gamma=1.0)),
        "recatch": int(cd.get("recatch") or 0),
        "stable_hold": int(cd.get("stable_hold") or 0),
        "obs0_sha16": _sha16(obs0),
        "qpos0_sha16": _sha16(qpos0),
        "obs0": obs0 if record_trace else None,
        "qpos0": qpos0 if record_trace else None,
        "actions": actions if record_trace else None,
        "qpos_trace_sha16": [_sha16(q) for q in qpos_trace] if record_trace else None,
        "tokens": [r["token"] for r in records],
    }
    return out


def _sha16(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def eval_starts(
    tag: str,
    ckpt,
    starts: list[MixedStart],
    *,
    split: str,
    n_rollouts: int,
    device: str,
    record_first_trace: bool = False,
) -> dict[str, Any]:
    from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_actor(ckpt, device_t)
    env = make_eval_env(book, rho)
    per_ic = []
    for st in starts:
        succ_r = succ12 = succ_new = drop = hand = rec = stab = 0
        tokens_lens = []
        for i in range(int(n_rollouts)):
            seed = rollout_seed(split=split, traj_id=st.traj_id, rollout_i=i)
            m = rollout_seeded(
                env,
                actor,
                st,
                device_t,
                seed=seed,
                record_trace=bool(record_first_trace and i == 0),
            )
            succ_r += int(m["SUCCESS_RELAXED_12S"])
            succ12 += int(m["SUCCESS_12S"])
            succ_new += int(m["SUCCESS_NEW"])
            drop += int(m["drop"])
            hand += int(m["handoff"])
            rec += int(m["recatch"])
            stab += int(m["stable_hold"])
            tokens_lens.append(int(m["n_steps"]))
        n = int(n_rollouts)
        row = {
            "tag": tag,
            "skill": st.skill,
            "traj_id": st.traj_id,
            "n_rollouts": n,
            "n_success_relaxed": succ_r,
            "n_success_12s": succ12,
            "n_success_new": succ_new,
            "p_relaxed": succ_r / n,
            "p_12s": succ12 / n,
            "p_new": succ_new / n,
            "p_drop": drop / n,
            "p_handoff": hand / n,
            "p_recatch": rec / n,
            "p_stable": stab / n,
            "zero_success": int(succ_r == 0),
            "mean_n_steps": float(np.mean(tokens_lens)),
        }
        per_ic.append(row)
        print(
            f"[{tag}/{split}] {st.skill} {st.traj_id}: "
            f"p_rel={row['p_relaxed']:.4f} ({succ_r}/{n})",
            flush=True,
        )
    return {
        "tag": tag,
        "ckpt": str(ckpt),
        "split": split,
        "device": str(device_t),
        "n_rollouts": n_rollouts,
        "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in per_ic])),
        "MSP_12S": float(np.mean([r["p_12s"] for r in per_ic])),
        "MSP_NEW": float(np.mean([r["p_new"] for r in per_ic])),
        "zero_success_ics": int(sum(r["zero_success"] for r in per_ic)),
        "per_ic": per_ic,
        "by_skill": {
            sk: float(np.mean([r["p_relaxed"] for r in per_ic if r["skill"] == sk]))
            for sk in sorted({r["skill"] for r in per_ic})
        },
    }
