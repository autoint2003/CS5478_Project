"""Phase 1 — first-contact failure attribution under frozen V2."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p73_first_contact_bilateral.paths import (
    MAX_EP_STEPS,
    N_ATTR_ICS,
    N_ATTR_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    T_MAX,
)
from experiments.precision.p73_first_contact_bilateral.rollout import rollout


def _starts(ids):
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
    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    ids = list(split["discovery_ics"])[:N_ATTR_ICS]
    starts = _starts(ids)

    rows = []
    for st in starts:
        for i in range(N_ATTR_ROLLOUTS):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=99600 + i)
            r = rollout(env, actor, critic, book, st, device_t, seed=seed, rule=None, tag="V2")
            rows.append(r)
            print(
                f"[p1] {st.traj_id[-22:]} i={i} opp={r['reached_opposing']} "
                f"bil={r['reached_bilateral']} e9={r['reached_e9']} r3={r['r3_ready']} "
                f"labels={r['failure_labels']} sw_rate={r['switch_rate_before_bil']:.2f}",
                flush=True,
            )

    n = len(rows)
    with_fc = [r for r in rows if r["fc_has_contact"]]
    fail_opp = [r for r in with_fc if not r["reached_opposing"]]
    succ_opp = [r for r in with_fc if r["reached_opposing"]]

    label_counts = Counter()
    for r in fail_opp:
        for lab in r["failure_labels"]:
            label_counts[lab] += 1

    # Overlaps
    pair_counts = Counter()
    for r in fail_opp:
        labs = sorted(r["failure_labels"])
        for i, a in enumerate(labs):
            for b in labs[i + 1 :]:
                pair_counts[f"{a}+{b}"] += 1

    per_ic = {}
    for tid in ids:
        rs = [r for r in rows if r["traj_id"] == tid]
        per_ic[tid] = {
            "n": len(rs),
            "opp_rate": float(np.mean([r["reached_opposing"] for r in rs])) if rs else None,
            "r3_rate": float(np.mean([r["r3_ready"] for r in rs])) if rs else None,
            "label_mode": Counter(
                lab for r in rs if not r["reached_opposing"] for lab in r["failure_labels"]
            ).most_common(3),
        }

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    report = {
        "n": n,
        "n_with_fc": len(with_fc),
        "opp_rate": mean([r["reached_opposing"] for r in with_fc]),
        "bil_rate": mean([r["reached_bilateral"] for r in with_fc]),
        "pad4_rate": mean([r["reached_pad4"] for r in with_fc]),
        "e9_rate": mean([r["reached_e9"] for r in with_fc]),
        "frame_ok_rate": mean([r["reached_frame_ok"] for r in with_fc]),
        "r3_rate": mean([r["r3_ready"] for r in rows]),
        "success_rate": mean([r["SUCCESS_RELAXED"] for r in rows]),
        "n_fail_opp": len(fail_opp),
        "n_succ_opp": len(succ_opp),
        "failure_label_counts": dict(label_counts),
        "failure_label_rates": {k: v / max(len(fail_opp), 1) for k, v in label_counts.items()},
        "failure_label_overlaps": dict(pair_counts.most_common(12)),
        "geom_succ_opp": {
            "mean_e_fc": mean([r["e_at_fc"] for r in succ_opp if r["e_at_fc"] is not None]),
            "mean_min_e": mean([r["min_e_in_window"] for r in succ_opp if r["min_e_in_window"] is not None]),
            "mean_t_to_opp": mean([r["t_to_opp"] for r in succ_opp if r["t_to_opp"] is not None]),
            "mean_switch_rate": mean([r["switch_rate_before_bil"] for r in succ_opp]),
            "mean_max_disp_from_fc": mean([r["max_obj_disp_from_fc"] for r in succ_opp]),
            "frac_ever_sw_window": mean([r["ever_correct_sw_in_window"] for r in succ_opp]),
            "frac_fc_sw_frame0": mean([r["fc_correct_sidewall"] for r in succ_opp]),
        },
        "geom_fail_opp": {
            "mean_e_fc": mean([r["e_at_fc"] for r in fail_opp if r["e_at_fc"] is not None]),
            "mean_min_e": mean([r["min_e_in_window"] for r in fail_opp if r["min_e_in_window"] is not None]),
            "mean_switch_rate": mean([r["switch_rate_before_bil"] for r in fail_opp]),
            "mean_max_disp_from_fc": mean([r["max_obj_disp_from_fc"] for r in fail_opp]),
            "frac_ever_sw_window": mean([r["ever_correct_sw_in_window"] for r in fail_opp]),
            "frac_fc_sw_frame0": mean([r["fc_correct_sidewall"] for r in fail_opp]),
            "frac_lost_contact": mean([r["lost_contact_before_opposing"] for r in fail_opp]),
        },
        "per_ic": per_ic,
        "discovery_ics": ids,
        "top_succ_token_prefix": [
            {"seq": list(k), "n": v}
            for k, v in Counter(
                tuple(r["post_tokens"][:5]) for r in succ_opp if r["post_tokens"]
            ).most_common(6)
        ],
        "top_fail_token_prefix": [
            {"seq": list(k), "n": v}
            for k, v in Counter(
                tuple(r["post_tokens"][:5]) for r in fail_opp if r["post_tokens"]
            ).most_common(6)
        ],
    }

    slim = []
    for r in rows:
        slim.append({k: v for k, v in r.items() if k != "fc_window_trace" or True})
        # keep short traces
        if len(slim[-1].get("fc_window_trace") or []) > 16:
            slim[-1]["fc_window_trace"] = slim[-1]["fc_window_trace"][:16]

    (OUT / "ATTR_RAW.json").write_text(json.dumps({"rows": slim, **report}, indent=2), encoding="utf-8")
    (OUT / "FIRST_CONTACT_FAILURES.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# First-Contact Failures (P73 Phase 1)\n",
        f"n={n} with_fc={len(with_fc)}  opp={report['opp_rate']:.3f}  "
        f"r3={report['r3_rate']:.3f}  fail_opp={len(fail_opp)}\n",
        "## Failure labels (overlapping; denom=fail_opp)\n```json\n"
        + json.dumps(report["failure_label_rates"], indent=2)
        + "\n```\n",
        "## Overlaps\n```json\n" + json.dumps(report["failure_label_overlaps"], indent=2) + "\n```\n",
        "## Geometry succ vs fail opposing\n```json\n"
        + json.dumps({"succ": report["geom_succ_opp"], "fail": report["geom_fail_opp"]}, indent=2)
        + "\n```\n",
        "Switch rates are **normalized** by decisions before bilateral (or window end).\n",
    ]
    (OUT / "FIRST_CONTACT_FAILURES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "BILATERAL_FORMATION_ANALYSIS.md").write_text(
        "# Bilateral Formation Analysis\n\n```json\n"
        + json.dumps(
            {
                "opp_rate": report["opp_rate"],
                "bil_rate": report["bil_rate"],
                "pad4_rate": report["pad4_rate"],
                "mean_t_to_opp_succ": report["geom_succ_opp"]["mean_t_to_opp"],
                "frac_ever_sw_window_succ": report["geom_succ_opp"]["frac_ever_sw_window"],
                "frac_ever_sw_window_fail": report["geom_fail_opp"]["frac_ever_sw_window"],
                "frac_lost_before_opp_fail": report["geom_fail_opp"]["frac_lost_contact"],
                "switch_rate_succ": report["geom_succ_opp"]["mean_switch_rate"],
                "switch_rate_fail": report["geom_fail_opp"]["mean_switch_rate"],
                "mean_disp_from_fc_succ": report["geom_succ_opp"]["mean_max_disp_from_fc"],
                "mean_disp_from_fc_fail": report["geom_fail_opp"]["mean_max_disp_from_fc"],
                "top_succ_tokens": report["top_succ_token_prefix"],
                "top_fail_tokens": report["top_fail_token_prefix"],
            },
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("opp_rate", "r3_rate", "failure_label_rates", "geom_fail_opp")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_phase1()
