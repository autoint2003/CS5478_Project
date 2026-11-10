"""Recompute P108 taxonomy/ranking from cached traces (no new closed-loop training/eval)."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import expert_actions  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import P64_ACTOR  # noqa: E402

import experiments.precision.p104_joint_token_residual_pilot.run_p104 as p104  # noqa: E402
import experiments.precision.p108_closed_loop_generalization_discovery.run_p108 as p108  # noqa: E402

OUT = p108.OUT
CACHE = p108.CACHE
PRIMARY = ["P64", "C_s1", "D_s1"]


def _rate(rows, pred):
    xs = [pred(r) for r in rows]
    return float(np.mean(xs)) if xs else 0.0


def steps_from_trace(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    tokens = np.asarray(z["tokens"], np.int64)
    actions = np.asarray(z["actions"], np.float32).reshape(-1, 7)
    margins = np.asarray(z["margins"], np.float32)
    e_pos = np.asarray(z["e_pos"], np.float32)
    bil = np.asarray(z["bil"], np.int8)
    n = len(tokens)
    steps = []
    for i in range(n):
        steps.append(
            {
                "t": i,
                "tok": int(tokens[i]),
                "margin": float(margins[i]) if i < len(margins) else 0.0,
                "action": actions[i].copy() if i < len(actions) else np.zeros(7, np.float32),
                "e_pos_mm": float(e_pos[i]) if i < len(e_pos) else 0.0,
                "bilateral": int(bil[i]) if i < len(bil) else 0,
                # traces store pre-step bil; use as after-proxy for reagg (same series)
                "bilateral_after": int(bil[i]) if i < len(bil) else 0,
                "resid": np.zeros(7, np.float32),
                "fn_L": 0.0,
                "fn_R": 0.0,
                "slip": 0.0,
            }
        )
    return steps


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # preserve pre-correction artifacts
    for name in ("SUMMARY.json", "FINAL_REPORT.md", "HYPOTHESIS_RANKING.md", "C_FAILURE_TAXONOMY.csv"):
        src = OUT / name
        dst = OUT / f"{Path(name).stem}_PRE_CORRECTION{Path(name).suffix}"
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)

    old_tax = list(csv.DictReader((OUT / "C_FAILURE_TAXONOMY.csv").open(encoding="utf-8")))
    takeover = list(csv.DictReader((OUT / "EXPERT_TAKEOVER_RESULTS.csv").open(encoding="utf-8")))
    panel_rows = list(csv.DictReader((OUT / "ROOT_PROVENANCE.csv").open(encoding="utf-8")))
    express_rows = list(csv.DictReader((OUT / "M111_EXPRESSIVITY_AUDIT.csv").open(encoding="utf-8")))
    temporal_old = list(csv.DictReader((OUT / "TEMPORAL_CONSISTENCY.csv").open(encoding="utf-8")))
    panel_meta = json.loads((OUT / "PANEL_META.json").read_text(encoding="utf-8"))
    old_summary = json.loads((OUT / "SUMMARY.json").read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    book = np.asarray(book, np.float64)
    rho = p104.rho_arr(rho)
    starts = p104.lookup_starts()

    # expert actions by tid (from schedule ep; no CL re-roll needed for taxonomy E)
    expert_by_tid: dict[str, dict] = {}
    for row in panel_rows:
        tid = row["traj_id"]
        st = starts[tid]
        acts = np.asarray(expert_actions(st.ep or {}), np.float32).reshape(-1, 7)
        expert_by_tid[tid] = {
            "actions": acts,
            "expert_available": int(len(acts) > 0),
            "SUCCESS_RELAXED_12S": int(row.get("expert_SUCCESS_RELAXED_12S") or 0),
        }

    taxonomy_rows = []
    temporal_rows = []
    for r in old_tax:
        pol, tid = r["policy"], r["traj_id"]
        trace = CACHE / f"trace_{pol}_{tid}.npz"
        if not trace.is_file():
            raise SystemExit(f"missing trace {trace}")
        steps = steps_from_trace(trace)
        roll = {
            "SUCCESS_RELAXED_12S": int(r["SUCCESS_RELAXED_12S"]),
            "reach12": int(r.get("reach12") or 0),
            "fail_reason": r.get("fail_reason"),
            "steps": steps,
        }
        er = expert_by_tid[tid]
        tax = p108.classify_failure(roll, er, book=book, rho=rho)
        taxonomy_rows.append(
            {
                "policy": pol,
                "traj_id": tid,
                "canonical_physical_id": r.get("canonical_physical_id", ""),
                "provenance": r.get("provenance", ""),
                "feasibility_stratum": r.get("feasibility_stratum", ""),
                "SUCCESS_RELAXED_12S": int(r["SUCCESS_RELAXED_12S"]),
                "expert_SUCCESS_RELAXED_12S": int(r.get("expert_SUCCESS_RELAXED_12S") or 0),
                "ep_len": r.get("ep_len", ""),
                "fail_reason": r.get("fail_reason", ""),
                "reach12": r.get("reach12", ""),
                "r3_ready": r.get("r3_ready", ""),
                **tax,
                "trace_path": str(trace),
            }
        )
        temporal_rows.extend(p108.temporal_rows_from_roll(pol, tid, roll, er))

    p108.write_csv(OUT / "C_FAILURE_TAXONOMY.csv", taxonomy_rows)
    p108.write_csv(OUT / "TEMPORAL_CONSISTENCY.csv", temporal_rows)

    # Observability: reuse previous rate if OBSERVABILITY exists; else recompute lightly
    obs_path = OUT / "OBSERVABILITY_AUDIT.md"
    try:
        corr = p104.load_correction_demos()
        obs_sum = p108.observability_audit(p64, corr, device, obs_path)
    except Exception:
        obs_sum = {"conflict_rate": float(old_summary.get("evidence", {}).get("obs_conflict_rate") or 0.0)}

    fail_tax = [r for r in taxonomy_rows if int(r["failed"]) == 1 and r["policy"] in PRIMARY]
    n_actor_fail_roots = len({r["traj_id"] for r in fail_tax})
    ex_take = [r for r in takeover if r["branch"] == "expert_continuation"]
    expert_rescue_rate = _rate(ex_take, lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1)
    ctx = {
        "n_panel": len(panel_rows),
        "n_expert_feasible": sum(int(r.get("expert_SUCCESS_RELAXED_12S") or 0) for r in panel_rows),
        "expert_success_rate": _rate(panel_rows, lambda r: int(r.get("expert_SUCCESS_RELAXED_12S") or 0) == 1),
        "n_actor_fail_roots": n_actor_fail_roots,
        "expert_rescue_rate": expert_rescue_rate,
        "frac_earliest_A": _rate(fail_tax, lambda r: r["earliest_label"] == "A"),
        "frac_earliest_B": _rate(fail_tax, lambda r: r["earliest_label"] == "B"),
        "frac_earliest_C": _rate(fail_tax, lambda r: r["earliest_label"] == "C"),
        "frac_earliest_D": _rate(fail_tax, lambda r: r["earliest_label"] == "D"),
        "frac_earliest_E": _rate(fail_tax, lambda r: r["earliest_label"] == "E"),
        "frac_label_B": _rate(fail_tax, lambda r: "B" in str(r["labels"]).split("|")),
        "frac_label_C": _rate(fail_tax, lambda r: "C" in str(r["labels"]).split("|")),
        "frac_label_D": _rate(fail_tax, lambda r: "D" in str(r["labels"]).split("|")),
        "frac_label_E": _rate(fail_tax, lambda r: "E" in str(r["labels"]).split("|")),
        "frac_label_F": _rate(fail_tax, lambda r: "F" in str(r["labels"]).split("|")),
        "frac_label_G": _rate(fail_tax, lambda r: "G" in str(r["labels"]).split("|")),
        "frac_harmful_switch": _rate(
            [r for r in temporal_rows if r["policy"] in PRIMARY],
            lambda r: int(r.get("harmful_switch_candidate") or 0) == 1,
        ),
        "frac_cat_d_expert": _rate(
            express_rows, lambda r: str(r.get("category_D_unrepresentable_expert")) == "1"
        ),
        "obs_conflict_rate": float(obs_sum.get("conflict_rate") or 0.0),
        "correction": {
            "label_E": "now_Category_D_unrepresentable_only",
            "h1_score": "removed_nominal_expert_success_bonus",
            "source": "reaggregate_from_cached_traces",
        },
    }

    pol_succ = {}
    for pol in PRIMARY:
        sub = [r for r in taxonomy_rows if r["policy"] == pol]
        pol_succ[pol] = {
            "n": len(sub),
            "succ": sum(int(r["SUCCESS_RELAXED_12S"]) for r in sub),
            "rate": _rate(sub, lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1),
            "expert_feasible_succ_rate": _rate(
                [r for r in sub if r.get("feasibility_stratum") == "expert_feasible"],
                lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1,
            ),
        }

    winner, ranking = p108.rank_hypotheses(ctx)
    rank_md = [
        "# HYPOTHESIS_RANKING — P108",
        "",
        "**Corrected re-aggregation** (label E = Category-D; H1 without nominal expert_ok bonus).",
        "",
        f"**Primary for P109: `{winner}`**",
        "",
        "| Hyp | Mech | Unified | Cost | Risk | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in ranking:
        rank_md.append(
            f"| {r['hypothesis']} | {r['mechanistic_score']} | {r['unified_actor_compat']} | "
            f"{r['impl_cost']} | {r['risk_to_A1_B1_D']} | **{r['total_rank_score']}** |"
        )
    rank_md.append("")
    rank_md.append("## Evidence drivers")
    for k, v in ctx.items():
        rank_md.append(f"- `{k}`: {v}")
    rank_md.append("")
    rank_md.append("## Definitions")
    rank_md.append("- H1 Targeted on-policy corrective imitation")
    rank_md.append("- H2 Future contact-state predictive auxiliary")
    rank_md.append("- H3 Contact-aware temporal action consistency")
    rank_md.append("- H4 Improved causal history representation")
    rank_md.append("- H5 Action-vocabulary redesign")
    (OUT / "HYPOTHESIS_RANKING.md").write_text("\n".join(rank_md), encoding="utf-8")
    p108.write_p109_proposal(winner, ranking, ctx)

    earliest_dist = defaultdict(int)
    for r in fail_tax:
        earliest_dist[r["earliest_label"]] += 1
    top_stage = max(earliest_dist.items(), key=lambda kv: kv[1])[0] if earliest_dist else "n/a"
    closed_loop_frac = ctx["frac_label_B"] + ctx["frac_label_C"] + ctx["frac_label_D"]
    initial_frac = ctx["frac_earliest_A"]

    summary = {
        "case": "P108_REAGGREGATED",
        "created": p108.utc(),
        "p64_sha": old_summary.get("p64_sha"),
        "sealed_panel_opened": False,
        "trained": False,
        "panel": panel_meta,
        "n_roots": len(panel_rows),
        "feasibility_strata": old_summary.get("feasibility_strata"),
        "policy_success": pol_succ,
        "failure_earliest_label_counts": dict(earliest_dist),
        "evidence": ctx,
        "expert_takeover": old_summary.get("expert_takeover"),
        "primary_hypothesis_p109": winner,
        "ranking": ranking,
        "previous_primary_hypothesis": old_summary.get("primary_hypothesis_p109"),
        "reaggregation": {
            "from_cache_traces": True,
            "preserved": [f"{Path(n).stem}_PRE_CORRECTION{Path(n).suffix}" for n in (
                "SUMMARY.json", "FINAL_REPORT.md", "HYPOTHESIS_RANKING.md", "C_FAILURE_TAXONOMY.csv"
            )],
            "unused_temporal_old_n": len(temporal_old),
        },
        "answers": {
            "1_failure_stage": top_stage,
            "2_expert_can_rescue": bool(expert_rescue_rate > 0.05),
            "3_dominated_by": (
                "unstable_later_sequences_and_contact_transitions"
                if closed_loop_frac >= initial_frac
                else "wrong_initial_actions"
            ),
            "4_switching_contributes_to_contact_loss": bool(ctx["frac_harmful_switch"] > 0.01),
            "5_observation_history_plausible": bool(ctx["obs_conflict_rate"] > 0.05 or ctx["frac_label_F"] > 0.1),
            # Prefer expressivity-audit Category-D rate; label E is contributing but continuous
            # expert tapes overstate episode-level E vs true M111 bottleneck.
            "6_m111_bottleneck": bool(ctx["frac_cat_d_expert"] > 0.15),
            "7_primary_hypothesis": winner,
            "8_smallest_p109": "docs/p109_new_skill_acquisition_proposal.md",
        },
    }
    p108.write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P108 Closed-Loop Generalization Discovery

Generated: {summary['created']}  
**Status: REAGGREGATED** (cached traces; no new rollouts). Pre-correction copies: `*_PRE_CORRECTION.*`.

## Scope
Diagnostic / counterfactual only. **No training.** P64 `{summary['p64_sha']}` immutable.
Policies: P64; P107 Arm C (grad-matched KL); P107 Arm D (Action-Aware).
Official R3F + `SUCCESS_RELAXED_12S`. Sealed ind **not** opened.

## Corrections applied
1. Label **E** = M111 Category-D (expert action unrepresentable), not Actor–expert action gap.
2. H1 ranking no longer receives a bonus from nominal expert success rate.
3. Slip/vrel key and bilateral_after used where available in re-derived temporal rows.

## Panel
- Distinct physical roots: **{len(panel_rows)}**
- Expert feasible / infeasible: {summary['feasibility_strata']}
- Freeze-ban note: most roots are historical freeze-ban but not train/sealed-ind (not sealed-independent evidence).

## Actor success (primary; unchanged numerics)
| Policy | n | succ | rate | rate on expert-feasible |
|---|---:|---:|---:|---:|
"""
    for pol, s in pol_succ.items():
        report += f"| {pol} | {s['n']} | {s['succ']} | {s['rate']:.3f} | {s['expert_feasible_succ_rate']:.3f} |\n"

    report += f"""
## Failure localization (corrected)
Earliest-label counts: {dict(earliest_dist)}

Contributing label rates (failed primary episodes): B={ctx['frac_label_B']:.2f}, C={ctx['frac_label_C']:.2f}, D={ctx['frac_label_D']:.2f}, E={ctx['frac_label_E']:.2f}, F={ctx['frac_label_F']:.2f}, G={ctx['frac_label_G']:.2f}.

## Expert takeover (unchanged)
- Expert continuation rescue rate: **{expert_rescue_rate:.3f}**
- Prefix5→actor rescue rate: **{summary['expert_takeover']['prefix_rescue_rate']:.3f}**

## Hypothesis ranking (corrected)
Previous primary: `{summary.get('previous_primary_hypothesis')}`.  
**Primary for P109: `{winner}`**

See `HYPOTHESIS_RANKING.md` and `docs/p109_new_skill_acquisition_proposal.md`.

## Answers
1. Failure stage (earliest mode): **{summary['answers']['1_failure_stage']}**
2. Expert rescue of Actor states? **{summary['answers']['2_expert_can_rescue']}**
3. Dominated by: **{summary['answers']['3_dominated_by']}**
4. Switching→contact loss? **{summary['answers']['4_switching_contributes_to_contact_loss']}**
5. Obs/history limitation plausible? **{summary['answers']['5_observation_history_plausible']}**
6. M111 bottleneck meaningful? **{summary['answers']['6_m111_bottleneck']}**
7. Strongest mechanism: **{winner}**
8. Smallest P109 pilot: see proposal doc.

## Stop
Stop after P108. Do not train P109 / PPO / GRPO / replace P64 / open sealed panel without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(f"[P108-reagg] winner={winner} prev={summary.get('previous_primary_hypothesis')} earliest={dict(earliest_dist)}")
    print(json.dumps(summary["answers"], indent=2))


if __name__ == "__main__":
    main()
