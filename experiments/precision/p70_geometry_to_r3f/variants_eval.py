"""Phase 4–5: closed-loop variants — tune on train ICs, freeze, validate full episodes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p70_geometry_to_r3f.controllers import VARIANTS, VariantSpec
from experiments.precision.p70_geometry_to_r3f.funnel_rollout import (
    _agg,
    _c_starts,
    absolute_threshold_analysis,
    rollout_funnel,
)
from experiments.precision.p70_geometry_to_r3f.paths import (
    MAX_EP_STEPS,
    N_TUNE_ICS,
    N_TUNE_ROLLOUTS,
    N_VAL_ICS,
    N_VAL_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    T_MAX,
)


def _split_ids() -> dict:
    return json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))


def _eval_variant(env, actor, critic, starts, device, spec: VariantSpec, *, n_rollouts: int, tag: str) -> list:
    rows = []
    for st in starts:
        for i in range(n_rollouts):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=99000 + i)
            row = rollout_funnel(env, actor, critic, st, device, seed=seed, spec=spec, tag=tag)
            rows.append(row)
            print(
                f"[var] {tag}/{spec.name} {st.traj_id[-22:]} i={i} "
                f"act={row['activated']} e_fc={None if not row['first_contact'] else round(row['first_contact']['e_pos_mm'],1)} "
                f"r3={row['r3_ready']} sw={row['correct_sidewall']} pad4={row['min_pad_ge4']}",
                flush=True,
            )
    return rows


def _score(agg: dict) -> float:
    """Tune score: prioritize R3 / frame / margin, then contact geometry."""
    return (
        3.0 * (agg.get("r3_ready") or 0)
        + 1.5 * (agg.get("frame_ok_ever") or 0)
        + 1.5 * (agg.get("margin_ok_ever") or 0)
        + 1.0 * (agg.get("min_pad_ge4") or 0)
        + 0.8 * (agg.get("correct_sidewall") or 0)
        - 0.002 * (agg.get("e_first_contact") or 0)
    )


def run_phase45(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    split = _split_ids()
    # Tune on discovery ICs only; validate on held-out train ICs (independent of discovery)
    tune_ids = list(split["discovery_ics"])[:N_TUNE_ICS]
    val_ids = list(split["heldout_train_ics"])[:N_VAL_ICS]
    tune_starts = _c_starts(tune_ids)
    val_starts = _c_starts(val_ids)

    # Small hyper grid for V2 only (ready threshold / max_approach) — freeze before val
    v2_grid = [
        VariantSpec(
            name=f"V2_a{a}_r{r}",
            kind="adaptive_approach",
            max_approach=a,
            min_approach=2,
            grip_steps=1,
            post_approach=2,
            ready_score_thresh=r,
        )
        for a, r in ((6, -0.05), (8, -0.05), (8, 0.0), (10, -0.1))
    ]
    v3_grid = [
        VariantSpec(
            name=f"V3_a{a}",
            kind="contact_aware",
            max_approach=a,
            min_approach=1,
            grip_steps=1,
            post_approach=2,
        )
        for a in (4, 6, 8)
    ]

    tune_results = {}
    print("=== Tune V0/V1 baselines on discovery ===", flush=True)
    for key in ("V0_baseline", "V1_fixed"):
        rows = _eval_variant(
            env, actor, critic, tune_starts, device_t, VARIANTS[key], n_rollouts=N_TUNE_ROLLOUTS, tag="tune"
        )
        tune_results[key] = {"agg": _agg(rows), "abs": absolute_threshold_analysis(rows), "n": len(rows)}

    print("=== Tune V2 grid ===", flush=True)
    best_v2, best_v2_score = None, -1e9
    for spec in v2_grid:
        rows = _eval_variant(env, actor, critic, tune_starts, device_t, spec, n_rollouts=N_TUNE_ROLLOUTS, tag="tune")
        agg = _agg(rows)
        sc = _score(agg)
        tune_results[spec.name] = {"agg": agg, "abs": absolute_threshold_analysis(rows), "score": sc, "n": len(rows)}
        print(f"  score {spec.name}={sc:.3f} r3={agg.get('r3_ready')}", flush=True)
        if sc > best_v2_score:
            best_v2_score, best_v2 = sc, spec

    print("=== Tune V3 grid ===", flush=True)
    best_v3, best_v3_score = None, -1e9
    for spec in v3_grid:
        rows = _eval_variant(env, actor, critic, tune_starts, device_t, spec, n_rollouts=N_TUNE_ROLLOUTS, tag="tune")
        agg = _agg(rows)
        sc = _score(agg)
        tune_results[spec.name] = {"agg": agg, "abs": absolute_threshold_analysis(rows), "score": sc, "n": len(rows)}
        print(f"  score {spec.name}={sc:.3f} r3={agg.get('r3_ready')}", flush=True)
        if sc > best_v3_score:
            best_v3_score, best_v3 = sc, spec

    frozen = {
        "V0": VARIANTS["V0_baseline"],
        "V1": VARIANTS["V1_fixed"],
        "V2": best_v2 or VARIANTS["V2_adaptive_approach"],
        "V3": best_v3 or VARIANTS["V3_contact_aware"],
    }
    frozen_ser = {
        k: {
            "name": v.name,
            "kind": v.kind,
            "max_activate_step": v.max_activate_step,
            "max_approach": v.max_approach,
            "min_approach": v.min_approach,
            "grip_steps": v.grip_steps,
            "post_approach": v.post_approach,
            "ready_score_thresh": v.ready_score_thresh,
            "sequence": v.sequence,
            "deployable_note": (
                "V2 uses proprioceptive ready_score (hand_speed/aperture/track_z), NOT e_pos. "
                "V3 uses contact_present_L/R from Actor obs."
            ),
        }
        for k, v in frozen.items()
    }
    (OUT / "CLOSED_LOOP_VARIANTS.json").write_text(json.dumps(frozen_ser, indent=2), encoding="utf-8")

    print("=== Full-episode validation on held-out train ICs (frozen rules) ===", flush=True)
    val_rows = {}
    matched = {}
    for key, spec in frozen.items():
        rows = _eval_variant(
            env, actor, critic, val_starts, device_t, spec, n_rollouts=N_VAL_ROLLOUTS, tag="val"
        )
        val_rows[key] = rows
        matched[key] = {
            "agg": _agg(rows),
            "abs": absolute_threshold_analysis(rows),
            "n": len(rows),
            "spec": frozen_ser[key],
        }

    # Per-IC paired vs V0
    per_ic = []
    by_v = {k: defaultdict(list) for k in frozen}
    for k, rows in val_rows.items():
        for r in rows:
            by_v[k][r["traj_id"]].append(r)
    for tid in val_ids:
        row = {"traj_id": tid}
        for k in frozen:
            row[k] = _agg(by_v[k].get(tid, []))
        if row["V0"].get("r3_ready") is not None and row["V1"].get("r3_ready") is not None:
            row["delta_r3_V1"] = row["V1"]["r3_ready"] - row["V0"]["r3_ready"]
            row["delta_r3_V2"] = (row["V2"].get("r3_ready") or 0) - (row["V0"].get("r3_ready") or 0)
            row["delta_r3_V3"] = (row["V3"].get("r3_ready") or 0) - (row["V0"].get("r3_ready") or 0)
        per_ic.append(row)

    # Promotion gate (Phase 6) on validation full episodes
    v0, v_best_name, v_best = "V0", "V0", matched["V0"]["agg"]
    for name in ("V1", "V2", "V3"):
        if _score(matched[name]["agg"]) > _score(v_best):
            v_best_name, v_best = name, matched[name]["agg"]

    def delta(k):
        return (v_best.get(k) or 0) - (matched["V0"]["agg"].get(k) or 0)

    gate = {
        "best_variant": v_best_name,
        "improve_abs_alignment": delta("e_first_contact") < -5.0 or (
            (matched[v_best_name]["abs"].get("first_contact") or {}).get("frac_le_25mm") or 0
        ) > ((matched["V0"]["abs"].get("first_contact") or {}).get("frac_le_25mm") or 0) + 0.05,
        "improve_contact_geometry": delta("correct_sidewall") > 0.05 or delta("min_pad_ge4") > 0.03,
        "improve_frame_or_margin": delta("frame_ok_ever") > 0.02 or delta("margin_ok_ever") > 0.02,
        "improve_r3_full_episode": delta("r3_ready") > 0.02,
        "e_only_without_r3": delta("e_first_contact") < -5 and delta("r3_ready") <= 0.01 and delta("frame_ok_ever") <= 0.01,
    }
    gate["recommend_train"] = bool(
        gate["improve_abs_alignment"]
        and gate["improve_contact_geometry"]
        and gate["improve_frame_or_margin"]
        and gate["improve_r3_full_episode"]
        and not gate["e_only_without_r3"]
    )

    report = {
        "tune_discovery": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in tune_results.items()},
        "frozen_variants": frozen_ser,
        "validation_heldout": matched,
        "per_ic": per_ic,
        "promotion_gate": gate,
        "deltas_best_vs_V0": {k: delta(k) for k in (
            "e_first_contact", "correct_sidewall", "min_pad_ge4",
            "frame_ok_ever", "margin_ok_ever", "r3_ready", "handoff", "SUCCESS_RELAXED",
        )},
    }
    (OUT / "MATCHED_COUNTERFACTUALS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_phase45()
