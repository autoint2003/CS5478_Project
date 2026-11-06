"""Phase 1 — Reconcile V2 alignment vs R3 success (post-contact correction?)."""

from __future__ import annotations

import json
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
from experiments.precision.p71_contact_safe_alignment.controllers import make_specs
from experiments.precision.p71_contact_safe_alignment.paths import (
    MAX_EP_STEPS,
    N_P1_ICS,
    N_P1_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    T_MAX,
)
from experiments.precision.p71_contact_safe_alignment.rollout import agg, rollout, wilson_ci


def _starts(ids: list[str]):
    blocked = blocked_eval_ids()
    by = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by[t] for t in ids if t in by]


def run_phase1(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    pool = alignment_token_pool(book)
    specs = make_specs(pool)
    v2 = specs["V2"]
    # Same held-out ICs as P70; denser rollouts for uncertainty
    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    ids = list(split["heldout_train_ics"])[:N_P1_ICS]
    starts = _starts(ids)

    rows_v0, rows_v2 = [], []
    for st in starts:
        for i in range(N_P1_ROLLOUTS):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=99100 + i)
            r0 = rollout(env, actor, critic, st, device_t, seed=seed, spec=None, tag="V0")
            r2 = rollout(env, actor, critic, st, device_t, seed=seed, spec=v2, tag="V2")
            rows_v0.append(r0)
            rows_v2.append(r2)
            print(
                f"[p1] {st.traj_id[-24:]} i={i} V0_r3={r0['r3_ready']} V2_r3={r2['r3_ready']} "
                f"V2_fc_e={None if not r2['first_contact'] else round(r2['first_contact']['e_pos_mm'],1)} "
                f"V2_succ={r2['SUCCESS_RELAXED']}",
                flush=True,
            )

    succ = [r for r in rows_v2 if r["r3_ready"] or r["SUCCESS_RELAXED"]]
    fail = [r for r in rows_v2 if not (r["r3_ready"] or r["SUCCESS_RELAXED"])]

    def stage_table(rows):
        out = []
        for r in rows:
            out.append(
                {
                    "traj_id": r["traj_id"],
                    "seed": r["seed"],
                    "e_before_fc": (r.get("e_before_first_contact") or {}).get("e_pos_mm"),
                    "e_fc": (r.get("first_contact") or {}).get("e_pos_mm"),
                    "theta_fc": (r.get("first_contact") or {}).get("theta_deg"),
                    "pad_fc": (r.get("first_contact") or {}).get("min_pad"),
                    "e_bil": (r.get("first_bilateral") or {}).get("e_pos_mm"),
                    "min_e_pre_r3": (r.get("min_e_before_r3") or {}).get("e_pos_mm"),
                    "e_r3": (r.get("at_r3_ready") or {}).get("e_pos_mm"),
                    "theta_r3": (r.get("at_r3_ready") or {}).get("theta_deg"),
                    "pad_r3": (r.get("at_r3_ready") or {}).get("min_pad"),
                    "e_handoff": (r.get("at_handoff") or {}).get("e_pos_mm"),
                    "r3": r["r3_ready"],
                    "success": r["SUCCESS_RELAXED"],
                }
            )
        return out

    succ_tbl = stage_table(succ)
    fail_tbl = stage_table(fail)

    n_succ = len(succ)
    fc_gt9 = [s for s in succ_tbl if s["e_fc"] is not None and s["e_fc"] > 9]
    corrected = [
        s
        for s in succ_tbl
        if s["e_fc"] is not None
        and s["e_fc"] > 9
        and (
            (s["e_r3"] is not None and s["e_r3"] <= 9)
            or (s["min_e_pre_r3"] is not None and s["min_e_pre_r3"] <= 9)
            or (s["e_bil"] is not None and s["e_bil"] <= 9)
        )
    ]

    n0 = len(rows_v0)
    k0 = int(sum(r["r3_ready"] for r in rows_v0))
    k2 = int(sum(r["r3_ready"] for r in rows_v2))
    # paired: same seed episodes
    paired_delta = [rows_v2[i]["r3_ready"] - rows_v0[i]["r3_ready"] for i in range(len(rows_v0))]
    # bootstrap-ish SE of paired mean
    pd = np.asarray(paired_delta, float)
    se = float(pd.std(ddof=1) / np.sqrt(len(pd))) if len(pd) > 1 else 0.0

    case_a = bool(n_succ > 0 and len(fc_gt9) / max(n_succ, 1) >= 0.5 and len(corrected) / max(len(fc_gt9), 1) >= 0.5)

    report = {
        "n_episodes_per_arm": n0,
        "n_ics": len(starts),
        "n_rollouts_per_ic": N_P1_ROLLOUTS,
        "V0": agg(rows_v0),
        "V2": agg(rows_v2),
        "r3_V0": {"k": k0, "n": n0, "rate": k0 / n0, "wilson95": wilson_ci(k0, n0)},
        "r3_V2": {"k": k2, "n": n0, "rate": k2 / n0, "wilson95": wilson_ci(k2, n0)},
        "paired_delta_r3_mean": float(pd.mean()) if len(pd) else None,
        "paired_delta_r3_se": se,
        "paired_delta_r3_ci95": [float(pd.mean() - 1.96 * se), float(pd.mean() + 1.96 * se)] if len(pd) else None,
        "n_success_V2": n_succ,
        "frac_succ_fc_gt9": float(len(fc_gt9) / n_succ) if n_succ else None,
        "frac_succ_fc_gt9_then_correct_le9": float(len(corrected) / len(fc_gt9)) if fc_gt9 else None,
        "n_succ_fc_gt9": len(fc_gt9),
        "n_corrected_after_contact": len(corrected),
        "case_A_post_contact_alignment": case_a,
        "first_contact_9mm_gate_too_strict": case_a,
        "success_stages": succ_tbl,
        "failure_stages_slim": fail_tbl[:40],
        "align_pool": pool,
    }
    (OUT / "V2_SUCCESS_GEOMETRY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# V2 Success Geometry (P71 Phase 1)\n",
        f"Denominators: **n={n0}** matched episodes ({len(starts)} ICs × {N_P1_ROLLOUTS}).\n",
        f"- V0 R3: {k0}/{n0} = {k0/n0:.3f} Wilson95={wilson_ci(k0,n0)}",
        f"- V2 R3: {k2}/{n0} = {k2/n0:.3f} Wilson95={wilson_ci(k2,n0)}",
        f"- Paired ΔR3 mean={report['paired_delta_r3_mean']:.3f} SE={se:.3f} CI95={report['paired_delta_r3_ci95']}",
        f"- V0 SUCCESS={report['V0']['SUCCESS_RELAXED']:.3f} V2 SUCCESS={report['V2']['SUCCESS_RELAXED']:.3f}\n",
        "## Post-contact correction among V2 successes\n",
        f"- n_success={n_succ}",
        f"- frac first-contact e>9 mm: {report['frac_succ_fc_gt9']}",
        f"- among those, frac later ≤9 mm (bil/min/r3): {report['frac_succ_fc_gt9_then_correct_le9']}",
        f"- **Case A (post-contact alignment)**: `{case_a}`",
        f"- First-contact 9 mm gate too restrictive?: `{case_a}`\n",
        "## Aggregate\n```json\n"
        + json.dumps({"V0": report["V0"], "V2": report["V2"]}, indent=2)
        + "\n```\n",
    ]
    (OUT / "V2_SUCCESS_GEOMETRY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:20]), flush=True)
    return report


if __name__ == "__main__":
    run_phase1()
