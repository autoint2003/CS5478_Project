"""Phase 5 — Independent validation on P61 + P65 development C ICs (frozen controller)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p62_grpo_optimization.paths import PER_CLASS_DEV
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p65_ppo_continuation.dev_panel import load_panel_starts
from experiments.precision.p71_contact_safe_alignment.align_tokens import alignment_token_pool
from experiments.precision.p71_contact_safe_alignment.controllers import Spec, make_specs
from experiments.precision.p71_contact_safe_alignment.paths import (
    MAX_EP_STEPS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    PANEL_C_ROLLOUTS,
    T_MAX,
)
from experiments.precision.p71_contact_safe_alignment.rollout import agg, count_r3, rollout, wilson_ci


def _load_frozen_spec(book) -> tuple[str, Spec]:
    path = OUT / "ALIGNMENT_VARIANTS.json"
    pool = alignment_token_pool(book)
    specs = make_specs(pool)
    if not path.exists():
        return "V2", specs["V2"]
    man = json.loads(path.read_text(encoding="utf-8"))
    name = man.get("best_name") or "V2"
    # Rebuild from saved fields if present
    sp = man.get("spec") or {}
    if name in specs:
        base = specs[name]
    else:
        base = specs["V2"]
    # Apply saved pools
    if sp.get("align_pool"):
        base.align_pool = list(sp["align_pool"])
    if sp.get("approach_pool"):
        base.approach_pool = list(sp["approach_pool"])
    return name, base


def _eval_starts(env, actor, critic, starts, device, specs: dict[str, Spec | None], *, n_roll: int, split: str):
    out = {k: [] for k in specs}
    for st in starts:
        for i in range(n_roll):
            seed = rollout_seed(split=split, traj_id=st.traj_id, rollout_i=1000 + i)
            for name, spec in specs.items():
                r = rollout(env, actor, critic, st, device, seed=seed, spec=spec, tag=f"{split}_{name}")
                out[name].append(r)
            print(
                f"[panel:{split}] {st.traj_id[-24:]} i={i} "
                + " ".join(f"{k}={out[k][-1]['r3_ready']}/{out[k][-1]['SUCCESS_RELAXED']}" for k in specs),
                flush=True,
            )
    return {k: {"agg": agg(v), "n": len(v), "rows": v} for k, v in out.items()}


def _per_ic(rows_by_name: dict) -> list:
    from collections import defaultdict

    tmp = {name: defaultdict(list) for name in rows_by_name}
    for name, blob in rows_by_name.items():
        for r in blob["rows"]:
            tmp[name][r["traj_id"]].append(r)
    tids = sorted(set().union(*[set(tmp[n]) for n in tmp])) if tmp else []
    per = []
    for tid in tids:
        row = {"traj_id": tid}
        for name in rows_by_name:
            rs = tmp[name].get(tid, [])
            row[name] = {
                "r3": float(np.mean([r["r3_ready"] for r in rs])) if rs else None,
                "succ": float(np.mean([r["SUCCESS_RELAXED"] for r in rs])) if rs else None,
                "n": len(rs),
            }
        if "V0" in row and "V2" in row:
            row["delta_r3_V2"] = (row["V2"]["r3"] or 0) - (row["V0"]["r3"] or 0)
        extras = [k for k in rows_by_name if k not in ("V0", "V2")]
        if extras and "V0" in row:
            bn = extras[0]
            row["delta_r3_best"] = (row[bn]["r3"] or 0) - (row["V0"]["r3"] or 0)
        per.append(row)
    return per


def run_phase5(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    best_name, best_spec = _load_frozen_spec(book)
    v2 = make_specs(alignment_token_pool(book))["V2"]
    label_best = best_name
    specs: dict[str, Spec | None] = {"V0": None, "V2": v2}
    if best_name not in ("V2", "V0"):
        specs[best_name] = best_spec

    # P61/P62 development C ICs
    held = load_heldout_starts(per_class=PER_CLASS_DEV)
    p61_c = list(held["C"])
    print(f"=== P61/P62 dev C ICs n={len(p61_c)} rolls={PANEL_C_ROLLOUTS} best={label_best} ===", flush=True)
    p61 = _eval_starts(
        env, actor, critic, p61_c, device_t, specs, n_roll=PANEL_C_ROLLOUTS, split="heldout"
    )

    # P65 extra panel C
    try:
        p65_map = load_panel_starts()
        p65_c = list(p65_map.get("C") or [])
    except Exception as e:  # noqa: BLE001
        print(f"[panel] P65 load failed: {e}", flush=True)
        p65_c = []
    p65 = {}
    if p65_c:
        print(f"=== P65 extra-dev C ICs n={len(p65_c)} ===", flush=True)
        p65 = _eval_starts(
            env, actor, critic, p65_c, device_t, specs, n_roll=PANEL_C_ROLLOUTS, split="p65_extra_dev"
        )

    def pack(blob):
        out = {}
        for k, v in blob.items():
            a = v["agg"]
            k_r3, n_r3 = count_r3(v["rows"])
            out[k] = {
                "agg": a,
                "n": v["n"],
                "r3_k": k_r3,
                "r3_wilson": wilson_ci(k_r3, n_r3),
            }
        return out

    report = {
        "best_name": label_best,
        "P61_dev_C": pack(p61),
        "P65_extra_C": pack(p65) if p65 else None,
        "per_ic_P61": _per_ic(p61),
        "per_ic_P65": _per_ic(p65) if p65 else [],
        "note": "Controller overlays apply on C early steps only; A/B/D unchanged vs P64 (not re-rolled here).",
    }
    (OUT / "PANEL_VALIDATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_phase5()
