"""C-focused denser candidate harvest: prioritize partial contact / lower e_pos before drop."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import take_branch_state
from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p55_expert_takeover.audit_zero import load_zero_ics
from experiments.precision.p55_expert_takeover.ic_resolve import resolve_starts
from experiments.precision.p55_expert_takeover.paths import (
    CAPTURE_DIR,
    MAX_EP_STEPS,
    OUT,
    P54_CKPT,
    SEED,
    T_MAX,
    TEMPERATURE,
)

N_ROLLOUTS = 16
MAX_CAND = 8
STRIDE = 2


def _score(m: dict) -> float:
    """Higher = better C takeover candidate (partial contact, closer, not latched)."""
    if m.get("latched"):
        return -1e9
    e = float(m.get("e_pos_mm") or 999)
    th = float(m.get("theta") or 180)
    nL = int(m.get("nL") or 0)
    nR = int(m.get("nR") or 0)
    bil = int(m.get("bilateral") or 0)
    vrel = float(m.get("vrel") or 0)
    contact = float(nL + nR)
    # Prefer unilateral/bilateral contact, lower error, lower speed
    return (
        100.0 * bil
        + 10.0 * min(contact, 12)
        - 0.05 * e
        - 0.02 * th
        - 2.0 * vrel
    )


def collect_c(device: str = "cuda") -> dict:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    zeros = [r for r in load_zero_ics() if r["skill"] == "C"]
    starts = resolve_starts([r["ic_id"] for r in zeros])
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P54_CKPT, device_t)
    actor.eval()
    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )
    rng = np.random.default_rng(SEED + 55)

    candidates = []
    for i, row in enumerate(zeros):
        tid = row["ic_id"]
        if tid not in starts:
            continue
        start = starts[tid]
        print(f"[{i+1}/{len(zeros)}] C-recollect {tid}", flush=True)
        pool = []
        for ri in range(N_ROLLOUTS):
            env.reset(start)
            done = False
            step = 0
            while not done:
                if step > 0 and step % STRIDE == 0 and env.mon is not None:
                    m = state_metrics(env.env.sim)
                    if not m.get("latched"):
                        st = take_branch_state(env.env, env.mon, step=step, entropy=0.0, token=-1)
                        pool.append(
                            {
                                "rollout_i": ri,
                                "step": step,
                                "metrics": m,
                                "branch": st,
                                "score": _score(m),
                            }
                        )
                tok, resid, _, _ = sample_step(
                    actor, env.obs_seq, env.prev_seq, device_t, temperature=TEMPERATURE
                )
                _r, done, info = env.step(tok, resid)
                step += 1
                if info.get("drop"):
                    break
        pool.sort(key=lambda h: h["score"], reverse=True)
        # Dedupe by coarse fingerprint, keep top MAX_CAND
        seen = set()
        picked = []
        for h in pool:
            key = (
                int(h["metrics"].get("bilateral") or 0),
                int(round(float(h["metrics"].get("e_pos_mm") or 0) / 8.0)),
                int(h["metrics"].get("nL") or 0) > 0,
                int(h["metrics"].get("nR") or 0) > 0,
                int(h["step"] // 10),
            )
            if key in seen:
                continue
            seen.add(key)
            picked.append(h)
            if len(picked) >= MAX_CAND:
                break

        for j, h in enumerate(picked):
            cap_path = CAPTURE_DIR / f"C_dense__{tid.replace('/', '_')}__c{j}.pkl"
            pickle.dump(h["branch"], cap_path.open("wb"))
            m = h["metrics"]
            candidates.append(
                {
                    "candidate_id": f"{tid}::dense_c{j}",
                    "ic_id": tid,
                    "skill": "C",
                    "class": "C",
                    "rollout_i": h["rollout_i"],
                    "step": h["step"],
                    "phase": "dense",
                    "score": float(h["score"]),
                    "metrics": {
                        k: (
                            float(v)
                            if isinstance(v, (float, np.floating))
                            else int(v)
                            if isinstance(v, (int, np.integer, bool, np.bool_))
                            else v
                        )
                        for k, v in m.items()
                    },
                    "capture_path": str(cap_path),
                }
            )
            print(
                f"  pick c{j} step={h['step']} score={h['score']:.1f} "
                f"e={m.get('e_pos_mm'):.1f} bil={m.get('bilateral')} "
                f"nL={m.get('nL')} nR={m.get('nR')}",
                flush=True,
            )

    out = {
        "n_zero_ics": len(zeros),
        "n_candidates": len(candidates),
        "n_probe_rollouts": N_ROLLOUTS,
        "max_per_ic": MAX_CAND,
        "stride": STRIDE,
        "mode": "C_dense_partial_contact",
        "candidates": candidates,
    }
    (OUT / "TAKEOVER_CANDIDATES_C_DENSE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("C dense candidates", len(candidates), flush=True)
    return out


if __name__ == "__main__":
    collect_c()
