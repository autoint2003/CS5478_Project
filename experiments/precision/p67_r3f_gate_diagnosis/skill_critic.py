"""Phase 4 — Skill/token transitions + Critic bias on post-stable / Expert-recoverable states."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.paths import (
    MAX_EP_STEPS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


def _lookup():
    return {s.traj_id: s for s in build_p54_dataset(seed=SEED)["schedule"] if s.skill == "C"}


def run_skill_critic(*, device: str = "cuda", n_cont: int = 6) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cf = json.loads((OUT / "EXPERT_COUNTERFACTUAL.json").read_text(encoding="utf-8"))
    man = json.loads((OUT / "POST_STABLE_SNAPSHOTS.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in man["states"]}
    # Focus Expert-recoverable / Actor-failing; also include neither for contrast
    focus = [r for r in cf["rows"] if r["bucket"] in ("expert_ok_actor_fail", "neither", "both_ok")]
    lookup = _lookup()
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    token_trans = Counter()
    entropies = []
    vals_all, rets_all = [], []
    v0_by_bucket = {"expert_ok_actor_fail": [], "neither": [], "both_ok": []}
    mc_by_bucket = {"expert_ok_actor_fail": [], "neither": [], "both_ok": []}
    switch_before_loss = []
    rows = []

    for row in focus:
        ent = by_id.get(row["id"])
        if not ent:
            continue
        start = lookup[row["traj_id"]]
        state = pickle.loads(Path(ent["path"]).read_bytes())
        bucket = row["bucket"]
        # One detailed rollout for token transitions + several for critic MC
        seed_all(SEED + hash(row["id"]) % 9000, deterministic_torch=True)
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        prev_tok = None
        lost_step = None
        tok_seq = []
        done = False
        info = {}
        ep_v, ep_r, ep_done = [], [], []
        while not done:
            # entropy via act meta if available
            tok, resid, logp, val = act_with_value(
                actor, critic, env.obs_seq, env.prev_seq, device_t
            )
            # approximate entropy from logp magnitude (higher |uncertainty| not exact)
            entropies.append(float(-logp))  # proxy: neg logprob of chosen token
            if prev_tok is not None and int(tok) != int(prev_tok):
                token_trans[(int(prev_tok), int(tok))] += 1
                if lost_step is None:
                    switch_before_loss.append(1)
            prev_tok = int(tok)
            tok_seq.append(int(tok))
            ep_v.append(float(val))
            r, done, info = env.step(tok, resid)
            ep_r.append(float(r))
            ep_done.append(1.0 if done else 0.0)
            bil = int((info.get("metrics") or {}).get("bilateral") or 0)
            if lost_step is None and not bil and env.t > int(state["policy_step"]) + 1:
                lost_step = int(env.t)

        _, ret = compute_gae(
            np.asarray(ep_r, np.float32),
            np.asarray(ep_v + [0.0], np.float32),
            np.asarray(ep_done, np.float32),
            1.0,
            0.95,
        )
        vals_all.extend(ep_v)
        rets_all.extend(ret.tolist())
        v0 = float(ep_v[0]) if ep_v else 0.0
        mc = float(np.sum(ep_r))
        if bucket in v0_by_bucket:
            v0_by_bucket[bucket].append(v0)
            mc_by_bucket[bucket].append(mc)

        # Extra continuations for bias estimate
        biases = []
        for i in range(n_cont - 1):
            seed_all(SEED + 1000 + i + hash(row["id"]) % 5000, deterministic_torch=True)
            restore_full_state(env, state, start)
            ep_v2, ep_r2, ep_d2 = [], [], []
            done = False
            while not done:
                tok, resid, _, val = act_with_value(
                    actor, critic, env.obs_seq, env.prev_seq, device_t
                )
                ep_v2.append(float(val))
                r, done, info = env.step(tok, resid)
                ep_r2.append(float(r))
                ep_d2.append(1.0 if done else 0.0)
            _, ret2 = compute_gae(
                np.asarray(ep_r2, np.float32),
                np.asarray(ep_v2 + [0.0], np.float32),
                np.asarray(ep_d2, np.float32),
                1.0,
                0.95,
            )
            vals_all.extend(ep_v2)
            rets_all.extend(ret2.tolist())
            biases.append(float(ep_v2[0]) - float(np.sum(ep_r2)))

        rows.append(
            {
                "id": row["id"],
                "bucket": bucket,
                "kind": row["kind"],
                "v0": v0,
                "mc_return": mc,
                "bias_v0_minus_mc": v0 - mc,
                "extra_bias_mean": float(np.mean(biases)) if biases else None,
                "lost_contact_step": lost_step,
                "n_token_switches": sum(
                    1 for a, b in zip(tok_seq, tok_seq[1:]) if a != b
                ),
                "tokens_head": tok_seq[:12],
                "actor_p_succ": row["actor"]["p_succ"],
                "expert_succ": row["expert"].get("SUCCESS_RELAXED_12S"),
            }
        )
        print(
            f"[skill] {bucket} V0={v0:.2f} MC={mc:.2f} bias={v0-mc:.2f} "
            f"switches={rows[-1]['n_token_switches']} lost@{lost_step}",
            flush=True,
        )

    y_pred = np.asarray(vals_all, np.float32)
    y_true = np.asarray(rets_all, np.float32)
    ev = float(explained_variance(y_pred, y_true)) if len(y_pred) else float("nan")
    bias = float(np.mean(y_pred - y_true)) if len(y_pred) else float("nan")
    # V0-level bias (matches P66 near-recatch ≈ −15 claim better)
    v0_biases = [r["bias_v0_minus_mc"] for r in rows]
    v0_bias_mean = float(np.mean(v0_biases)) if v0_biases else None

    top_trans = [{"from": a, "to": b, "n": n} for (a, b), n in token_trans.most_common(15)]

    skill_report = {
        "n_states": len(rows),
        "top_token_transitions": top_trans,
        "mean_neglogp_proxy": float(np.mean(entropies)) if entropies else None,
        "rows": rows,
        "notes": [
            "M111 tokens are unlabeled prototypes — no token→skill map; switches = codebook ID changes.",
            "Expert actions are NOT treated as PPO on-policy samples.",
            "Entropy proxy uses -logπ(a); not full categorical entropy.",
        ],
    }
    critic_report = {
        "explained_variance": ev,
        "step_value_bias_mean": bias,
        "v0_minus_mc_mean": v0_bias_mean,
        "n_value_samples": int(len(y_pred)),
        "v0_by_bucket": {k: float(np.mean(v)) if v else None for k, v in v0_by_bucket.items()},
        "mc_by_bucket": {k: float(np.mean(v)) if v else None for k, v in mc_by_bucket.items()},
        "p66_near_recatch_bias_claim": -15.0,
        "validation": (
            f"Post-stable V0−MC mean={v0_bias_mean}; "
            "compare to P66 near-recatch bias≈−15 on before_closing/before_bilateral stages."
        ),
    }
    (OUT / "SKILL_TRANSITION_ANALYSIS.json").write_text(
        json.dumps(skill_report, indent=2), encoding="utf-8"
    )
    (OUT / "CRITIC_STAGE_ANALYSIS.json").write_text(
        json.dumps(critic_report, indent=2), encoding="utf-8"
    )

    md_s = ["# Skill Transition Analysis (P67 Phase 4)\n"]
    md_s.append(f"States analyzed: **{len(rows)}** (post-stable counterfactual set)\n")
    md_s.append("## Top token transitions (prev→next)\n")
    for t in top_trans[:10]:
        md_s.append(f"- `{t['from']} → {t['to']}`: {t['n']}")
    md_s.append("\n## Per-state summary\n")
    for r in rows[:20]:
        md_s.append(
            f"- {r['bucket']}/{r['kind']}: switches={r['n_token_switches']} "
            f"lost@{r['lost_contact_step']} V0={r['v0']:.1f} MC={r['mc_return']:.1f}"
        )
    for n in skill_report["notes"]:
        md_s.append(f"\n_{n}_")
    (OUT / "SKILL_TRANSITION_ANALYSIS.md").write_text("\n".join(md_s) + "\n", encoding="utf-8")

    md_c = ["# Critic Stage Analysis (P67 Phase 4)\n"]
    md_c.append(f"- Step-level EV: **{ev:.4f}**")
    md_c.append(f"- Step-level bias (V−GAE-ret): **{bias:.4f}**")
    md_c.append(f"- Snapshot V0−MC mean: **{v0_bias_mean}**")
    md_c.append(f"- P66 claim (near-recatch): **−15**")
    md_c.append(f"- Samples: {len(y_pred)}\n")
    md_c.append("## By counterfactual bucket\n")
    for k in v0_by_bucket:
        md_c.append(
            f"- **{k}**: V0={critic_report['v0_by_bucket'][k]}, MC={critic_report['mc_by_bucket'][k]}"
        )
    md_c.append(f"\n_{critic_report['validation']}_\n")
    (OUT / "CRITIC_STAGE_ANALYSIS.md").write_text("\n".join(md_c) + "\n", encoding="utf-8")
    print("\n".join(md_c), flush=True)
    return {"skill": skill_report, "critic": critic_report}


if __name__ == "__main__":
    run_skill_critic()
