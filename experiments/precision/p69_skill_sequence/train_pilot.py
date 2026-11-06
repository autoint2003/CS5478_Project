"""Phase 4 — Matched PPO: Branch A control vs Branch B sequence-preference aux.

Mechanism (Phase 2 = B_ordered_transition):
  Separate supervised preference loss on the frozen corrective token sequence.
  Never injects forced actions into the PPO buffer / importance ratios.
  Optional KL anchor to frozen P64_step4 Actor.
"""

from __future__ import annotations

import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import (
    categorical_kl_masked,
    mask_handoff_logits,
)
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p63_transformer_critic_ppo.paths import WINDOW
from experiments.precision.p63_transformer_critic_ppo.ppo import ppo_update
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value, pad_batch
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p63_transformer_ppo_c_grasp.rollout import collect_schedule
from experiments.precision.p64_ppo_multiround.checkpointing import load_train_state, save_train_state
from experiments.precision.p64_ppo_multiround.schedule import (
    blocked_eval_ids,
    full_train_pool,
    sample_stratified,
)
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p69_skill_sequence.paths import (
    CKPT_DIR,
    GROUP,
    ICS_PER_UPDATE,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P64_STEP4_OPTIM,
    SEED,
    STATE_DIR,
    TRAJ_PER_BRANCH,
    UPDATES,
)

AUX_COEF = 0.15
ANCHOR_KL_COEF = 0.05
AUX_MINIBATCH = 32
AUX_EPOCHS = 1


def _load_rule_and_pref_pairs(actor, critic, book, rho, device) -> tuple[list[int], list[dict]]:
    """Build offline (obs, prev, target_token) pairs from discovery pre-contact states."""
    from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset

    hyp = json.loads((OUT / "FIXED_SEQUENCE_HYPOTHESIS.json").read_text(encoding="utf-8"))
    seq = [int(x) for x in hyp["sequence"]]
    blocked = blocked_eval_ids()
    lookup = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    env = make_env(book, rho, enable_c_grasp_reward=False)
    pairs: list[dict] = []
    for path in sorted(STATE_DIR.glob("disc_*.pkl"))[:24]:
        state = pickle.loads(path.read_bytes())
        meta = state.get("meta") or {}
        tid = meta.get("traj_id")
        if tid not in lookup:
            continue
        start = lookup[tid]
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        for tok in seq:
            pairs.append(
                {
                    "obs": np.stack(env.obs_seq[-WINDOW:]).astype(np.float32),
                    "prev": np.stack(env.prev_seq[-WINDOW:]).astype(np.float32),
                    "token": int(tok),
                }
            )
            _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            env.step(int(tok), resid)
            if env.t >= env.max_steps:
                break
    return seq, pairs


def aux_sequence_update(actor, ref_actor, pairs: list[dict], opt_actor, device) -> dict:
    """Supervised NLL on frozen sequence tokens + KL(π||π_ref). Separate from PPO ratios."""
    if not pairs:
        return {"aux_nll": None, "anchor_kl": None, "n": 0}
    metrics = []
    idxs = np.arange(len(pairs))
    for _ in range(AUX_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, len(pairs), AUX_MINIBATCH):
            mb = idxs[start : start + AUX_MINIBATCH]
            obs = [pairs[i]["obs"] for i in mb]
            prev = [pairs[i]["prev"] for i in mb]
            toks = torch.as_tensor([pairs[i]["token"] for i in mb], device=device, dtype=torch.long)
            o, p, lengths, pad_mask = pad_batch(obs, prev, device)
            logits, _, _ = actor(o, p, pad_mask=pad_mask)
            last = torch.stack(
                [mask_handoff_logits(logits[j, lengths[j] - 1].float()) for j in range(len(mb))]
            )
            dist = Categorical(logits=last)
            nll = -dist.log_prob(toks).mean()
            # Anchor KL on first sample of minibatch (cheap)
            kl = categorical_kl_masked(actor, ref_actor, obs[0], prev[0], device)
            loss = AUX_COEF * nll + ANCHOR_KL_COEF * kl
            opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
            opt_actor.step()
            metrics.append({"aux_nll": float(nll.detach()), "anchor_kl": float(kl.detach()), "grad": gn})
    return {
        "aux_nll": float(np.mean([m["aux_nll"] for m in metrics])) if metrics else None,
        "anchor_kl": float(np.mean([m["anchor_kl"] for m in metrics])) if metrics else None,
        "n": len(pairs),
        "n_minibatches": len(metrics),
    }


def run_branch(
    *,
    branch: str,
    use_seq_aux: bool,
    device: str = "cuda",
    target_sim_steps: int | None = None,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, critic, book, rho, opt_a, opt_c, opt_note, _ = load_train_state(
        P64_STEP4_ACTOR, P64_STEP4_CRITIC, P64_STEP4_OPTIM, device_t
    )
    # Frozen reference for KL anchor (Branch B)
    ref_actor = None
    pairs: list[dict] = []
    if use_seq_aux:
        ref_actor = copy.deepcopy(actor).eval()
        for p in ref_actor.parameters():
            p.requires_grad_(False)
        _seq, pairs = _load_rule_and_pref_pairs(actor, critic, book, rho, device_t)
        print(f"[p69-{branch}] pref pairs={len(pairs)} seq={_seq}", flush=True)

    env = make_env(book, rho, enable_c_grasp_reward=False)
    pool = full_train_pool(SEED)
    blocked = blocked_eval_ids()

    history = []
    total_steps = 0
    total_eps = 0
    t0 = time.perf_counter()
    updates_done = 0
    max_updates = UPDATES if target_sim_steps is None else max(UPDATES, 12)

    for u in range(max_updates):
        if target_sim_steps is not None and total_steps >= target_sim_steps:
            break
        if target_sim_steps is None and u >= UPDATES:
            break
        seed_u = SEED + 69000 + (1 if use_seq_aux else 0) * 10000 + u * 17
        schedule = sample_stratified(pool, n_ics=ICS_PER_UPDATE, seed=seed_u, blocked=blocked)
        print(
            f"=== P69 {branch} update {u+1}/{max_updates} seq_aux={use_seq_aux} "
            f"steps_so_far={total_steps} ===",
            flush=True,
        )
        batch, summary = collect_schedule(env, actor, critic, schedule, device_t, n_traj=GROUP)
        # PPO on on-policy Actor data only (correct log-probs)
        metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
        aux_m = {}
        if use_seq_aux and pairs and ref_actor is not None:
            aux_m = aux_sequence_update(actor, ref_actor, pairs, opt_a, device_t)
        total_steps += summary["n_steps"]
        total_eps += summary["n_episodes"]
        updates_done += 1
        row = {
            "branch": branch,
            "update": u + 1,
            "rollout": {k: summary[k] for k in summary if k != "p_relaxed_by_class"},
            "ppo": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
            "aux": aux_m,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "branch": branch,
                    "update": u + 1,
                    "p_rel": summary["overall_p_relaxed"],
                    "steps": summary["n_steps"],
                    "approx_kl": metrics.get("approx_kl"),
                    "aux_nll": aux_m.get("aux_nll"),
                    "anchor_kl": aux_m.get("anchor_kl"),
                }
            ),
            flush=True,
        )

    wall = time.perf_counter() - t0
    tag = f"P69_{branch}"
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
            "use_seq_aux": use_seq_aux,
            "aux_coef": AUX_COEF if use_seq_aux else 0.0,
            "anchor_kl_coef": ANCHOR_KL_COEF if use_seq_aux else 0.0,
            "opt_note": opt_note,
            "n_sim_steps": total_steps,
            "wall_s": wall,
            "mechanism": "B_sequence_preference_aux" if use_seq_aux else "none",
        },
    )
    out = {
        "branch": branch,
        "use_seq_aux": use_seq_aux,
        "opt_note": opt_note,
        "n_episodes": total_eps,
        "n_sim_steps": total_steps,
        "n_actor_updates": updates_done,
        "wall_s": wall,
        "gpu_hours": wall / 3600.0,
        "target_traj_opportunities": TRAJ_PER_BRANCH,
        "actor": str(actor_path),
        "critic": str(critic_path),
        "optim": str(opt_path),
        "n_pref_pairs": len(pairs),
        "history": history,
    }
    (OUT / f"PILOT_{branch}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_matched_pilots(*, device: str = "cuda") -> dict:
    print("=== P69 Branch A — Standard PPO Control ===", flush=True)
    a = run_branch(branch="A_control", use_seq_aux=False, device=device)
    print(f"=== P69 Branch B — Sequence preference (match steps ≈ {a['n_sim_steps']}) ===", flush=True)
    b = run_branch(
        branch="B_seq_pref",
        use_seq_aux=True,
        device=device,
        target_sim_steps=int(a["n_sim_steps"]),
    )
    pilot = {
        "A_control": {k: v for k, v in a.items() if k != "history"},
        "B_seq_pref": {k: v for k, v in b.items() if k != "history"},
        "A_history": a["history"],
        "B_history": b["history"],
        "matched_sim_steps_target": a["n_sim_steps"],
        "step_match_ratio": b["n_sim_steps"] / max(1, a["n_sim_steps"]),
        "ppo_correctness": {
            "forced_actions_in_ppo_buffer": False,
            "aux_separate_from_importance_ratios": True,
            "reward": "P57-A unchanged",
            "anchor": "KL to frozen P64_step4 on Branch B",
            "mechanism": "B_sequence_preference_aux",
        },
    }
    (OUT / "PILOT_RESULTS.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")
    (OUT / "PPO_CORRECTNESS_AUDIT.json").write_text(
        json.dumps(pilot["ppo_correctness"], indent=2), encoding="utf-8"
    )
    return pilot


if __name__ == "__main__":
    run_matched_pilots()
