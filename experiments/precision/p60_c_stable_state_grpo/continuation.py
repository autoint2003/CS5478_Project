"""Continue Independent GRPO rollouts from restored full intermediate states."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p53_independent_grpo64.policy_sample import sample_step
from experiments.precision.p53_independent_grpo64.reward import trajectory_return
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state


def load_state_pkl(path: Path) -> dict[str, Any]:
    return pickle.loads(path.read_bytes())


def rollout_continuation(
    env: StabilizeGrpoEnv,
    actor,
    start: MixedStart,
    state: dict[str, Any],
    device,
    *,
    temperature: float,
    gamma_return: float,
    fixed_actions: list[tuple[int, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Continue from restored intermediate state; returns only continuation records/rewards.

    Remaining horizon is preserved via restored env.t / max_steps / R3F / sim time.
    HANDOFF-ready geometry is NOT treated as terminal success — only env.step terminal path.
    """
    restore_full_state(env, state, start)
    records: list[dict] = []
    rewards: list[float] = []
    done = False
    info: dict = {}
    ai = 0
    while not done:
        obs_w = np.stack(env.obs_seq[-160:]).astype(np.float32)
        prev_w = np.stack(env.prev_seq[-160:]).astype(np.float32)
        if fixed_actions is not None:
            if ai >= len(fixed_actions):
                break
            tok, resid = fixed_actions[ai]
            resid = np.asarray(resid, np.float32)
            logp_beh, entropy = 0.0, 0.0
            ai += 1
        else:
            tok, resid, logp_beh, meta = sample_step(
                actor, env.obs_seq, env.prev_seq, device, temperature=temperature
            )
            entropy = float(meta["entropy"])
        r, done, info = env.step(int(tok), resid)
        records.append(
            {
                "obs": obs_w,
                "prev": prev_w,
                "token": int(tok),
                "resid": np.asarray(resid, np.float32),
                "logp_behavior": float(logp_beh),
                "entropy": float(entropy),
            }
        )
        rewards.append(float(r))

    ret = trajectory_return(rewards, gamma=gamma_return) if rewards else 0.0
    cd = (info or {}).get("c_diag") or {}
    return {
        "records": records,
        "rewards": rewards,
        "return": float(ret),
        "info": info,
        "SUCCESS_RELAXED_12S": int((info or {}).get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int((info or {}).get("SUCCESS_12S") or 0),
        "SUCCESS_NEW": int((info or {}).get("SUCCESS_NEW") or 0),
        "drop": int((info or {}).get("drop") or 0),
        "handoff": int((info or {}).get("handoff") or 0),
        "truncated": int((info or {}).get("truncated") or 0),
        "n_steps": len(records),
        "start_policy_step": int(state["policy_step"]),
        "end_policy_step": int((info or {}).get("t") or state["policy_step"]),
        "c_diag": dict(cd),
        "handoff_reason": (info or {}).get("handoff_reason"),
        "fail_reason": (info or {}).get("fail_reason"),
        "lost_contact": int(cd.get("lost_after_recatch") or 0),
    }


def continuation_group(
    env: StabilizeGrpoEnv,
    actor,
    start: MixedStart,
    state: dict[str, Any],
    device,
    *,
    group: int,
    temperature: float,
    gamma_return: float,
) -> list[dict[str, Any]]:
    members = []
    for _ in range(int(group)):
        members.append(
            rollout_continuation(
                env,
                actor,
                start,
                state,
                device,
                temperature=temperature,
                gamma_return=gamma_return,
            )
        )
    return members
