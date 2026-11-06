"""Phase 1 — Per-IC C stage transition rates under P64_step4."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.geometry_validate import _r3f_ready
from experiments.precision.p66_c_curriculum_ppo.paths import (
    GEOM_STAGE_ROLLOUTS,
    MAX_EP_STEPS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


STAGES = (
    "recatch",
    "correct_sidewall",
    "sustained_bilateral",
    "r3f_ready",
    "handoff",
    "SUCCESS_RELAXED",
)


def _flags_from_episode(env, info, ever_sw, ever_bil, max_bil_run, ever_r3f) -> dict[str, int]:
    cd = info.get("c_diag") or {}
    return {
        "recatch": int(cd.get("recatch") or 0),
        "correct_sidewall": int(ever_sw),
        "sustained_bilateral": int(max_bil_run >= 5 or int(cd.get("stable_hold") or 0)),
        "r3f_ready": int(ever_r3f),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
    }


def run_stage_transitions(*, n_ics: int = 16, n_per: int = GEOM_STAGE_ROLLOUTS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    critic = load_critic(P64_STEP4_CRITIC, actor, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    blocked = blocked_eval_ids()
    starts = [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ][:n_ics]

    all_flags: list[dict[str, int]] = []
    per_ic: list[dict] = []

    for st in starts:
        ic_flags = []
        for i in range(n_per):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=2000 + i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ever_sw = False
            ever_bil = False
            ever_r3f = False
            max_bil_run = cur = 0
            done = False
            info: dict[str, Any] = {}
            while not done:
                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                _r, done, info = env.step(tok, resid)
                audit = classify_contacts(env.env.sim)
                if is_correct_sidewall(audit):
                    ever_sw = True
                if audit.get("bilateral"):
                    ever_bil = True
                    cur += 1
                    max_bil_run = max(max_bil_run, cur)
                else:
                    cur = 0
                if _r3f_ready(env):
                    ever_r3f = True
            flags = _flags_from_episode(env, info, ever_sw, ever_bil, max_bil_run, ever_r3f)
            # also mark bil ever into sustained if needed — already handled
            _ = ever_bil
            ic_flags.append(flags)
            all_flags.append(flags)
        rates = {k: float(np.mean([f[k] for f in ic_flags])) for k in STAGES}
        per_ic.append({"traj_id": st.traj_id, "n": n_per, "rates": rates})
        print(f"[stage] {st.traj_id[-36:]} {rates}", flush=True)

    n = max(1, len(all_flags))
    overall = {k: float(np.mean([f[k] for f in all_flags])) for k in STAGES}
    # Conditional transitions with denominators
    def cond(a: str, b: str) -> dict:
        num = sum(1 for f in all_flags if f[a] and f[b])
        den = sum(1 for f in all_flags if f[a])
        return {"p": (num / den) if den else None, "num": num, "den": den}

    chain = [
        ("recatch", "correct_sidewall"),
        ("correct_sidewall", "sustained_bilateral"),
        ("sustained_bilateral", "r3f_ready"),
        ("r3f_ready", "handoff"),
        ("handoff", "SUCCESS_RELAXED"),
        ("correct_sidewall", "handoff"),
        ("correct_sidewall", "SUCCESS_RELAXED"),
        ("recatch", "SUCCESS_RELAXED"),
    ]
    conditionals = {f"P({b}|{a})": cond(a, b) for a, b in chain}

    # Dominant bottleneck: largest drop along primary chain
    primary = [
        ("start", "recatch", overall["recatch"]),
        ("recatch", "correct_sidewall", conditionals["P(correct_sidewall|recatch)"]["p"]),
        ("correct_sidewall", "sustained_bilateral", conditionals["P(sustained_bilateral|correct_sidewall)"]["p"]),
        ("sustained_bilateral", "r3f_ready", conditionals["P(r3f_ready|sustained_bilateral)"]["p"]),
        ("r3f_ready", "handoff", conditionals["P(handoff|r3f_ready)"]["p"]),
        ("handoff", "SUCCESS_RELAXED", conditionals["P(SUCCESS_RELAXED|handoff)"]["p"]),
    ]
    drops = []
    for a, b, p in primary:
        if p is None:
            continue
        drops.append({"from": a, "to": b, "conditional_p": p, "drop": 1.0 - float(p)})
    bottleneck = max(drops, key=lambda d: d["drop"]) if drops else None

    report = {
        "policy": str(P64_STEP4_ACTOR),
        "n_episodes": n,
        "n_ics": len(starts),
        "rollouts_per_ic": n_per,
        "overall_stage_rates": overall,
        "conditionals": conditionals,
        "bottleneck": bottleneck,
        "primary_chain": primary,
        "per_ic": per_ic,
        "caution": (
            "Do not overinterpret P(SUCCESS|HANDOFF)=1 when handoff denominator is tiny."
        ),
    }
    (OUT / "C_STAGE_TRANSITIONS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# C Stage Transitions (P66 Phase 1)\n"]
    md.append(f"Policy: `{P64_STEP4_ACTOR.name}`  ")
    md.append(f"Episodes: **{n}** ({len(starts)} ICs × {n_per})\n")
    md.append("## Overall stage rates\n")
    for k in STAGES:
        md.append(f"- {k}: **{100*overall[k]:.1f}%** (n={n})")
    md.append("\n## Conditional probabilities\n")
    for name, c in conditionals.items():
        p = c["p"]
        p_s = "n/a" if p is None else f"{p:.3f}"
        md.append(f"- {name} = {p_s}  (num={c['num']}, den={c['den']})")
    md.append("\n## Dominant bottleneck\n")
    if bottleneck:
        md.append(
            f"- Largest drop: **{bottleneck['from']} → {bottleneck['to']}** "
            f"(conditional P={bottleneck['conditional_p']:.3f}, drop={bottleneck['drop']:.3f})"
        )
    md.append(f"\n_{report['caution']}_\n")
    (OUT / "C_STAGE_TRANSITIONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return report


if __name__ == "__main__":
    run_stage_transitions()
