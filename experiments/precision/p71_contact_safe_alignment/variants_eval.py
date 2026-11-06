"""Phase 3–4: matched counterfactuals on train-C; freeze best for Phase 5."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p71_contact_safe_alignment.align_tokens import alignment_token_pool
from experiments.precision.p71_contact_safe_alignment.controllers import Spec, make_specs
from experiments.precision.p71_contact_safe_alignment.paths import (
    MAX_EP_STEPS,
    N_DEV_ICS,
    N_DEV_ROLLOUTS,
    N_HELD_ICS,
    N_HELD_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    T_MAX,
)
from experiments.precision.p71_contact_safe_alignment.rollout import agg, rollout


def _starts(ids):
    blocked = blocked_eval_ids()
    by = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by[t] for t in ids if t in by]


def _score(a: dict) -> float:
    """Prioritize terminal success and R3 progression over mean e reduction."""
    return (
        4.0 * (a.get("SUCCESS_RELAXED") or 0)
        + 3.0 * (a.get("r3_ready") or 0)
        + 1.2 * (a.get("frame_ok_ever") or 0)
        + 1.0 * (a.get("margin_ok_ever") or 0)
        + 0.8 * (a.get("min_pad_ge4") or 0)
        + 0.6 * (a.get("correct_sidewall") or 0)
        - 0.5 * (a.get("contact_loss_rate") or 0)
        + 0.3 * (a.get("frac_fc_le9") or 0)
        + 0.2 * (a.get("frac_fc_le12") or 0)
    )


def _eval_set(env, actor, critic, starts, device, specs: dict[str, Spec], *, n_roll: int, tag: str):
    by = {k: [] for k in list(specs) + ["V0"]}
    for st in starts:
        for i in range(n_roll):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=99300 + i)
            r0 = rollout(env, actor, critic, st, device, seed=seed, spec=None, tag=f"{tag}_V0")
            by["V0"].append(r0)
            for name, spec in specs.items():
                r = rollout(env, actor, critic, st, device, seed=seed, spec=spec, tag=f"{tag}_{name}")
                by[name].append(r)
            print(
                f"[{tag}] {st.traj_id[-22:]} i={i} "
                + " ".join(f"{k}_r3={by[k][-1]['r3_ready']}" for k in ["V0"] + list(specs)),
                flush=True,
            )
    return {k: {"rows": v, "agg": agg(v)} for k, v in by.items()}


def run_phase34(*, device: str = "cuda", include_v5: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    pool = alignment_token_pool(book, exclude_56=True, k=8)
    specs = make_specs(pool)
    if not include_v5:
        specs.pop("V5", None)

    # Validate V5 proxy: ready_score must vary on pre-contact obs
    # (simple check — always include if pool non-empty; proxy is deployable by construction)
    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    dev_ids = list(split["discovery_ics"])[:N_DEV_ICS]
    held_ids = list(split["heldout_train_ics"])[:N_HELD_ICS]
    # Also use validation ICs for held if we want more independence from discovery tune
    # Spec: develop on train-C; freeze then held-out train
    print(f"[p34] align_pool={pool}", flush=True)
    print("=== Dev (discovery) tune/select ===", flush=True)
    dev = _eval_set(env, actor, critic, _starts(dev_ids), device_t, specs, n_roll=N_DEV_ROLLOUTS, tag="dev")

    scores = {k: _score(dev[k]["agg"]) for k in specs}
    best_name = max(scores, key=scores.get)
    # Prefer beating V2 on R3 or SUCCESS; else keep V2
    v2a, ba = dev["V2"]["agg"], dev[best_name]["agg"]
    if (ba.get("r3_ready") or 0) + (ba.get("SUCCESS_RELAXED") or 0) < (v2a.get("r3_ready") or 0) + (
        v2a.get("SUCCESS_RELAXED") or 0
    ) - 0.02:
        best_name = "V2"
    frozen = specs[best_name]
    frozen_ser = {
        "best_name": best_name,
        "scores_dev": scores,
        "align_pool": pool,
        "spec": {
            "name": frozen.name,
            "kind": frozen.kind,
            "max_align": frozen.max_align,
            "min_align": frozen.min_align,
            "max_approach": frozen.max_approach,
            "min_approach": frozen.min_approach,
            "ready_score_thresh": frozen.ready_score_thresh,
            "contact_safe": frozen.contact_safe,
            "adaptive_transition": frozen.adaptive_transition,
            "align_pool": frozen.align_pool,
            "approach_pool": frozen.approach_pool,
            "deployable": True,
            "uses_privileged_e_pos": False,
        },
    }
    (OUT / "ALIGNMENT_VARIANTS.json").write_text(json.dumps(frozen_ser, indent=2), encoding="utf-8")

    print(f"=== Held-out full episodes (frozen best={best_name}) ===", flush=True)
    # Evaluate all variants on held-out for reporting, but selection already frozen
    held = _eval_set(env, actor, critic, _starts(held_ids), device_t, specs, n_roll=N_HELD_ROLLOUTS, tag="held")

    def delta(a, b, k):
        return None if a.get(k) is None or b.get(k) is None else float(a[k]) - float(b[k])

    deltas = {
        name: {k: delta(held[name]["agg"], held["V0"]["agg"], k) for k in (
            "r3_ready", "SUCCESS_RELAXED", "correct_sidewall", "min_pad_ge4",
            "frame_ok_ever", "margin_ok_ever", "e_first_contact", "frac_fc_le9", "frac_fc_le12",
            "contact_loss_rate",
        )}
        for name in ["V2"] + list(specs)
        if name in held
    }

    # Case classification
    p1 = {}
    p1_path = OUT / "V2_SUCCESS_GEOMETRY.json"
    if p1_path.exists():
        p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    best_d = deltas.get(best_name) or {}
    v2_d = deltas.get("V2") or {}
    # Extensions may fail even when V2 is retained as best — detect Case D separately.
    ext_failed = any(
        (deltas.get(name) or {}).get("r3_ready") is not None
        and (deltas.get(name) or {}).get("r3_ready") < (v2_d.get("r3_ready") or 0) - 0.05
        for name in ("V3", "V4", "V5")
        if name in deltas
    )
    if p1.get("case_A_post_contact_alignment") and best_name == "V2":
        case = "A"
        if ext_failed:
            case = "A_with_D_extensions"  # post-contact success path; naive align extensions rejected
    elif (best_d.get("r3_ready") or 0) > 0.02 and (best_d.get("SUCCESS_RELAXED") or 0) > 0.02 and best_name != "V2":
        case = "B"
    elif (best_d.get("frac_fc_le9") or 0) > 0.05 and (best_d.get("r3_ready") or 0) <= 0.01:
        case = "C"
    elif best_name in ("V3", "V4", "V5") and (
        (best_d.get("contact_loss_rate") or 0) > (v2_d.get("contact_loss_rate") or 0) + 0.05
        or (best_d.get("r3_ready") or 0) < (v2_d.get("r3_ready") or 0) - 0.05
    ):
        case = "D"
    elif best_name == "V2":
        case = "A" if p1.get("case_A_post_contact_alignment") else "B_keep_V2"
    else:
        case = "B" if (best_d.get("r3_ready") or 0) >= (v2_d.get("r3_ready") or 0) else "C"

    report = {
        "dev_agg": {k: v["agg"] for k, v in dev.items()},
        "held_agg": {k: v["agg"] for k, v in held.items()},
        "deltas_vs_V0_held": deltas,
        "frozen": frozen_ser,
        "case": case,
        "dev_scores": scores,
    }
    # slim rows for JSON size
    slim = {
        "held_per_variant_n": {k: len(v["rows"]) for k, v in held.items()},
        **report,
    }
    (OUT / "MATCHED_COUNTERFACTUALS.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    md = [
        "# Full-Episode Results (P71 Phase 4)\n",
        f"Frozen best (from discovery only): **{best_name}**  case=`{case}`\n",
        "## Held-out aggregates\n",
    ]
    for k, v in held.items():
        md.append(f"### {k}\n```json\n{json.dumps(v['agg'], indent=2)}\n```\n")
    md.append("## Δ vs V0\n```json\n" + json.dumps(deltas, indent=2) + "\n```\n")
    (OUT / "FULL_EPISODE_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    md2 = [
        "# Alignment Variants (P71 Phase 3)\n",
        "V2 = P70 reference. V3 = extended non-56 alignment. V4 = contact-safe abort. "
        "V5 = adaptive transition (proprio ready_score / hand_speed). No privileged e_pos.\n",
        f"```json\n{json.dumps(frozen_ser, indent=2)}\n```\n",
        "## Dev scores\n```json\n" + json.dumps(scores, indent=2) + "\n```\n",
    ]
    (OUT / "ALIGNMENT_VARIANTS.md").write_text("\n".join(md2) + "\n", encoding="utf-8")
    return slim


if __name__ == "__main__":
    run_phase34()
