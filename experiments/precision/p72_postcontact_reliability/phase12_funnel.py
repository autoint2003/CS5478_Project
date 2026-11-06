"""Phase 1–2: post-contact funnel + success/failure timelines."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p72_postcontact_reliability.instrumented import STAGES, rollout_instrumented
from experiments.precision.p72_postcontact_reliability.paths import (
    MAX_EP_STEPS,
    N_FUNNEL_ICS,
    N_FUNNEL_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    T_MAX,
)


def _starts(ids):
    blocked = blocked_eval_ids()
    by = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by[t] for t in ids if t in by]


def _funnel_stats(rows: list[dict]) -> dict:
    n = len(rows)
    ever = {s: float(np.mean([r["ever"].get(s, 0) for r in rows])) for s in STAGES}
    # conditional: P(stage_i | stage_{i-1} ever)
    cond = {}
    for a, b in zip(STAGES, STAGES[1:]):
        base = [r for r in rows if r["ever"].get(a)]
        if not base:
            cond[f"{b}|{a}"] = None
        else:
            cond[f"{b}|{a}"] = float(np.mean([r["ever"].get(b, 0) for r in base]))
    # residence among those who ever reached
    res = {}
    for k in ("contact", "bilateral", "min_pad_ge4", "frame_ok", "margin_ok"):
        xs = [r["residence_steps"].get(k, 0) for r in rows if r["residence_steps"].get(k, 0) > 0]
        res[k] = {"n": len(xs), "mean": float(np.mean(xs)) if xs else None, "median": float(np.median(xs)) if xs else None}
    lost = {k: float(np.mean([r["lost_after_counts"].get(k, 0) > 0 for r in rows])) for k in (
        "first_contact", "opposing_bilateral", "sustained_bilateral", "min_pad_ge4", "frame_ok", "margin_ok"
    )}
    return {"n": n, "first_passage_ever": ever, "conditional": cond, "residence": res, "frac_ever_lost_after": lost}


def run_phase12(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    # discovery for analysis; held-out reserved for later validation
    ids = list(split["discovery_ics"])[:N_FUNNEL_ICS]
    starts = _starts(ids)

    rows = []
    for st in starts:
        for i in range(N_FUNNEL_ROLLOUTS):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=99400 + i)
            r = rollout_instrumented(
                env, actor, critic, book, st, device_t, seed=seed, post=None, tag="V2", record_trace=True
            )
            rows.append(r)
            print(
                f"[funnel] {st.traj_id[-24:]} i={i} r3={r['r3_ready']} e9={r['reached_e9']} "
                f"pad4={r['ever']['min_pad_ge4']} sw={r['ever']['opposing_bilateral']} "
                f"near={r['near_ready_seen']} switches={r['n_family_switches_post']}",
                flush=True,
            )

    succ = [r for r in rows if r["r3_ready"] or r["SUCCESS_RELAXED"]]
    fail = [r for r in rows if not (r["r3_ready"] or r["SUCCESS_RELAXED"])]
    funnel_all = _funnel_stats(rows)
    funnel_succ = _funnel_stats(succ) if succ else None
    funnel_fail = _funnel_stats(fail) if fail else None

    def geom_cmp(group):
        return {
            "n": len(group),
            "mean_e_fc": float(np.mean([r["e_at_fc"] for r in group if r["e_at_fc"] is not None])) if group else None,
            "mean_min_e": float(np.mean([r["min_e_after_fc"] for r in group if r["min_e_after_fc"] is not None])) if group else None,
            "frac_reach_e12": float(np.mean([r["reached_e12"] for r in group])) if group else None,
            "frac_reach_e9": float(np.mean([r["reached_e9"] for r in group])) if group else None,
            "mean_t_to_12": float(np.mean([r["t_to_12"] for r in group if r["t_to_12"] is not None])) if group else None,
            "mean_t_to_9": float(np.mean([r["t_to_9"] for r in group if r["t_to_9"] is not None])) if group else None,
            "frac_pad4": float(np.mean([r["ever"]["min_pad_ge4"] for r in group])) if group else None,
            "frac_opp_bil": float(np.mean([r["ever"]["opposing_bilateral"] for r in group])) if group else None,
            "frac_lost_contact": float(np.mean([r["lost_after_counts"]["first_contact"] > 0 for r in group])) if group else None,
            "mean_post_switches": float(np.mean([r["n_family_switches_post"] for r in group])) if group else None,
            "mean_max_ok_s": float(np.mean([r["max_ok_s"] for r in group])) if group else None,
            "frac_near_ready": float(np.mean([r["near_ready_seen"] for r in group])) if group else None,
        }

    # near-R3 failures: near_ready but not r3
    near_fail = [r for r in fail if r["near_ready_seen"]]
    viol_counter = Counter()
    for r in near_fail:
        fv = r.get("first_violated_near") or {}
        for v in fv.get("violations") or ["unknown"]:
            viol_counter[v] += 1

    # post-contact skill patterns among successes
    succ_seqs = Counter()
    for r in succ:
        seq = tuple(r["post_tokens"][:8])
        if seq:
            succ_seqs[seq] += 1
    fail_seqs = Counter()
    for r in fail:
        seq = tuple(r["post_tokens"][:8])
        if seq:
            fail_seqs[seq] += 1

    fam_succ = Counter()
    for r in succ:
        fam_succ[tuple(r["post_families"][:8])] += 1

    report = {
        "n": len(rows),
        "n_succ": len(succ),
        "n_fail": len(fail),
        "r3_rate": float(np.mean([r["r3_ready"] for r in rows])),
        "success_rate": float(np.mean([r["SUCCESS_RELAXED"] for r in rows])),
        "funnel_all": funnel_all,
        "funnel_succ": funnel_succ,
        "funnel_fail": funnel_fail,
        "geom_succ": geom_cmp(succ),
        "geom_fail": geom_cmp(fail),
        "near_fail_n": len(near_fail),
        "near_fail_first_violation": dict(viol_counter),
        "top_succ_token_prefix": [{"seq": list(k), "n": v} for k, v in succ_seqs.most_common(8)],
        "top_succ_family_prefix": [{"families": list(k), "n": v} for k, v in fam_succ.most_common(8)],
        "top_fail_token_prefix": [{"seq": list(k), "n": v} for k, v in fail_seqs.most_common(5)],
        "discovery_ics": ids,
    }
    # slim rows for later CF mining
    slim_rows = []
    for r in rows:
        slim_rows.append({
            "traj_id": r["traj_id"],
            "seed": r["seed"],
            "r3_ready": r["r3_ready"],
            "SUCCESS_RELAXED": r["SUCCESS_RELAXED"],
            "ever": r["ever"],
            "e_at_fc": r["e_at_fc"],
            "min_e_after_fc": r["min_e_after_fc"],
            "reached_e9": r["reached_e9"],
            "t_to_9": r["t_to_9"],
            "post_tokens": r["post_tokens"],
            "post_families": r["post_families"],
            "n_family_switches_post": r["n_family_switches_post"],
            "near_ready_seen": r["near_ready_seen"],
            "first_violated_near": r["first_violated_near"],
            "max_ok_s": r["max_ok_s"],
            "first_pass": r["first_pass"],
            "lost_after_counts": r["lost_after_counts"],
            "trace": r["trace"],
        })
    (OUT / "FUNNEL_RAW.json").write_text(json.dumps({"rows": slim_rows, **report}, indent=2), encoding="utf-8")
    (OUT / "POSTCONTACT_FUNNEL.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Post-Contact Funnel (P72 Phase 1)\n",
        f"n={report['n']}  R3={report['r3_rate']:.3f}  SUCCESS={report['success_rate']:.3f}  "
        f"succ/fail={report['n_succ']}/{report['n_fail']}\n",
        "## First-passage (ever)\n```json\n" + json.dumps(funnel_all["first_passage_ever"], indent=2) + "\n```\n",
        "## Conditional P(next|prev ever)\n```json\n" + json.dumps(funnel_all["conditional"], indent=2) + "\n```\n",
        "## Residence / loss\n```json\n"
        + json.dumps({"residence": funnel_all["residence"], "lost": funnel_all["frac_ever_lost_after"]}, indent=2)
        + "\n```\n",
    ]
    (OUT / "POSTCONTACT_FUNNEL.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    md2 = [
        "# Success vs Failure Timelines (P72 Phase 2)\n",
        "## Geometry\n```json\n"
        + json.dumps({"succ": report["geom_succ"], "fail": report["geom_fail"]}, indent=2)
        + "\n```\n",
        "## Near-R3 failures\n",
        f"n={report['near_fail_n']} first violations=`{report['near_fail_first_violation']}`\n",
        "## Skills\n```json\n"
        + json.dumps(
            {
                "top_succ_tokens": report["top_succ_token_prefix"],
                "top_succ_families": report["top_succ_family_prefix"],
            },
            indent=2,
        )
        + "\n```\n",
    ]
    (OUT / "SUCCESS_FAILURE_TIMELINES.md").write_text("\n".join(md2) + "\n", encoding="utf-8")
    (OUT / "M111_POSTCONTACT_SKILLS.md").write_text(
        "# M111 Post-Contact Skills\n\n```json\n"
        + json.dumps(
            {
                "top_succ_token_prefix": report["top_succ_token_prefix"],
                "top_succ_family_prefix": report["top_succ_family_prefix"],
                "top_fail_token_prefix": report["top_fail_token_prefix"],
                "note": "Prefixes are first ≤8 post-contact tokens under frozen V2→Actor continuation.",
            },
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "CONTACT_SUPPORT_ANALYSIS.md").write_text(
        "# Contact Support Analysis\n\n```json\n"
        + json.dumps(
            {
                "succ_frac_pad4": report["geom_succ"]["frac_pad4"],
                "fail_frac_pad4": report["geom_fail"]["frac_pad4"],
                "succ_frac_opp_bil": report["geom_succ"]["frac_opp_bil"],
                "fail_frac_opp_bil": report["geom_fail"]["frac_opp_bil"],
                "succ_frac_lost_contact": report["geom_succ"]["frac_lost_contact"],
                "fail_frac_lost_contact": report["geom_fail"]["frac_lost_contact"],
                "funnel_pad4_ever": funnel_all["first_passage_ever"].get("min_pad_ge4"),
                "cond_pad4_given_sustained_bil": funnel_all["conditional"].get("min_pad_ge4|sustained_bilateral"),
            },
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"r3": report["r3_rate"], "funnel": funnel_all["first_passage_ever"], "geom_succ": report["geom_succ"], "geom_fail": report["geom_fail"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_phase12()
