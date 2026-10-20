"""Finalize unified expansion reports with accurate recounts from CSVs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "expert_generation" / "k49_unified_dataset_expansion"


def main():
    audit = json.loads((OUT / "RAW" / "baseline_audit_summary.json").read_text(encoding="utf-8"))
    canon = list(csv.DictReader((OUT / "CANONICAL_MANIFEST.csv").open(encoding="utf-8")))
    exp = list(csv.DictReader((OUT / "VERIFIED_FULL_EXPERTS.csv").open(encoding="utf-8")))
    strata = list(csv.DictReader((OUT / "IC_DIFFICULTY_STRATA.csv").open(encoding="utf-8"))) if (OUT / "IC_DIFFICULTY_STRATA.csv").exists() else []
    new_roots = list(csv.DictReader((OUT / "NEW_PHYSICAL_ROOTS.csv").open(encoding="utf-8"))) if (OUT / "NEW_PHYSICAL_ROOTS.csv").exists() else []
    splits = list(csv.DictReader((OUT / "TRAIN_VAL_TEST_SPLITS.csv").open(encoding="utf-8"))) if (OUT / "TRAIN_VAL_TEST_SPLITS.csv").exists() else []

    roots = {r["root_episode"] for r in canon}
    ics = {r["ic_id"] for r in canon}
    prior_roots = set(audit["unique_roots"])
    new_indep = roots - prior_roots

    def fnum(r, k):
        try:
            return int(float(r.get(k) or 0))
        except Exception:
            return 0

    newly = [r for r in exp if fnum(r, "newly_vs_simple") == 1]
    stable = [r for r in exp if r.get("stability_class") == "STABLE_POSE_ERROR"]
    recatch_manip = [r for r in exp if fnum(r, "recatch_plus_manip") == 1]
    ball = {r["root_episode"] for r in canon
            if "ball" in str(r.get("source") or r.get("source_dataset") or "").lower()
            or str(r["root_episode"]).startswith(("bimp_", "ubimp"))}
    near_h = [r for r in exp if "NEAR_HORIZONTAL" in str(r.get("regimes", ""))]
    high_mot_ics = {s["ic_id"] for s in strata if float(s.get("v0") or 0) >= 0.2 or float(s.get("w0") or 0) >= 4}
    high_mot_full = [r for r in exp if r["root_episode"] in high_mot_ics or r["ic_id"] in high_mot_ics]

    lengths = []
    for r in canon:
        if r.get("n_tokens"):
            try:
                lengths.append(int(float(r["n_tokens"])))
            except Exception:
                pass
    handoff_s = [0.02 * x for x in lengths]

    src = Counter((r.get("source_dataset") or r.get("source") or r.get("provenance") or "?") for r in canon)
    klass = Counter(r.get("stability_class") or "?" for r in exp)
    diff = Counter(r.get("difficulty") or "?" for r in exp)
    split_ctr = Counter(r.get("split") for r in splits)

    n_accepted_gen = sum(1 for r in new_roots if str(r.get("accepted")) == "1")

    under = []
    if len(roots) < 75:
        under.append(f"stage2_75_shortfall_have_{len(roots)}")
    if len(roots) < 100:
        under.append(f"stage3_100_shortfall_have_{len(roots)}")
    if len(ball) < 15:
        under.append("ball_impact_roots")
    if len(recatch_manip) < 8:
        under.append("recatch_plus_manipulation")
    if len(high_mot_full) < 5:
        under.append("high_motion_FULL")
    if len(near_h) < 8:
        under.append("near_horizontal_FULL")

    answers = {
        "Q1_n_verified_traj_before": audit["n_source_SUCCESS_NEW_rows"],
        "Q2_n_unique_ics_before": audit["n_unique_ics"],
        "Q2_n_unique_roots_before": audit["n_unique_roots"],
        "Q3_n_new_physical_roots_generated": n_accepted_gen,
        "Q3_n_ball_attempts": len(new_roots),
        "Q4_n_new_independent_roots_SUCCESS_NEW": len(new_indep),
        "Q5_n_unique_successful_roots_after": len(roots),
        "Q5_n_unique_ics_after": len(ics),
        "Q5_n_trajectories_after": len(canon),
        "Q6_n_newly_vs_simple": len(newly),
        "Q7_n_recatch_plus_slide_pivot_expansion": len(recatch_manip),
        "Q8_n_ball_impact_roots": len(ball),
        "Q9_n_near_horizontal_expansion": len(near_h),
        "Q9_n_high_motion_full_expansion": len(high_mot_full),
        "Q10_seq_len_mean": float(np.mean(lengths)) if lengths else None,
        "Q10_seq_len_median": float(np.median(lengths)) if lengths else None,
        "Q10_handoff_s_mean": float(np.mean(handoff_s)) if handoff_s else None,
        "Q10_handoff_s_median": float(np.median(handoff_s)) if handoff_s else None,
        "Q11_stability_class_expansion": dict(klass),
        "Q11_difficulty_expansion": dict(diff),
        "Q11_source_distribution_all": dict(src),
        "Q11_stable_pose_error_full": len(stable),
        "Q11_split_counts": dict(split_ctr),
        "Q12_underrepresented": under,
        "Q13_suitable_for_SFT": len(roots) >= 50,
        "stage1_50": len(roots) >= 50,
        "stage2_75": len(roots) >= 75,
        "stage3_100": len(roots) >= 100,
        "limiting_factors": [
            "STABLE_POSE_ERROR transfer often fails when residual axial/lateral error needs longer Slide+Return",
            "High-motion / LOST_GRASP ball-impact ICs remain mostly unsolved within bounded budgets",
            "Easy-cap no longer drops roots; remaining gap is genuine search yield, not counting",
            "Ball-impact FULL yield low vs generation attempts",
        ],
        "unique_roots_list": sorted(roots),
        "new_independent_roots": sorted(new_indep),
    }
    answers["Q13_note"] = (
        f"Stage1(50)={'YES' if answers['stage1_50'] else 'NO'}; "
        f"Stage2(75)={'YES' if answers['stage2_75'] else 'NO'}; "
        f"Stage3(100)={'YES' if answers['stage3_100'] else 'NO'}. "
        f"Have {len(roots)} unique SUCCESS_NEW roots "
        f"(prior {audit['n_unique_roots']} + new {len(new_indep)}). "
        + ("Suitable for initial unified SFT with root-grouped splits; "
           "continue mining dynamic/high-motion regimes before claiming coverage completeness."
           if answers["Q13_suitable_for_SFT"] else
           "Below 50-root SFT floor.")
    )
    (OUT / "RAW" / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

    report = f"""# Unified K49 Grasp-Recovery Dataset Expansion

**Config:** `K49_tok48_wy0p25` — SUCCESS_NEW locked. No SFT/GRPO. No physics changes.  
**Amendment applied:** historical STABLE_POSE_ERROR ICs are first-class.

## Stage status

| Stage | Target | Achieved |
|-------|-------:|----------|
| 1 | 50 | **{'YES' if answers['stage1_50'] else 'NO'}** ({len(roots)}) |
| 2 | 75 | **{'YES' if answers['stage2_75'] else 'NO'}** |
| 3 | 100 | **{'YES' if answers['stage3_100'] else 'NO'}** |

## Baseline (before)

| Metric | Value |
|--------|------:|
| SUCCESS_NEW traj rows | {answers['Q1_n_verified_traj_before']} |
| Unique ICs | {answers['Q2_n_unique_ics_before']} |
| Unique roots | **{answers['Q2_n_unique_roots_before']}** |

## After expansion

| Metric | Value |
|--------|------:|
| Trajectories | {answers['Q5_n_trajectories_after']} |
| Unique ICs | {answers['Q5_n_unique_ics_after']} |
| **Unique SUCCESS_NEW roots** | **{answers['Q5_n_unique_successful_roots_after']}** |
| New independent roots | {answers['Q4_n_new_independent_roots_SUCCESS_NEW']} |
| Accepted new ball roots generated | {answers['Q3_n_new_physical_roots_generated']} / {answers['Q3_n_ball_attempts']} attempts |
| Newly vs simple baselines | {answers['Q6_n_newly_vs_simple']} |
| STABLE_POSE_ERROR expansion FULL | {answers['Q11_stable_pose_error_full']} |
| Ball-impact successful roots | {answers['Q8_n_ball_impact_roots']} |
| Recatch+Slide/Pivot (expansion flags) | {answers['Q7_n_recatch_plus_slide_pivot_expansion']} |

## Separate coverage (amendment)

- **Stable-grasp pose correction:** {len(stable)} expansion FULL with `STABLE_POSE_ERROR`
- **Dynamic / ball-impact roots:** {len(ball)} — {sorted(ball)[:25]}{'...' if len(ball)>25 else ''}

## Q1–Q13

1. Verified traj before: **{answers['Q1_n_verified_traj_before']}**
2. Unique ICs / roots before: **{answers['Q2_n_unique_ics_before']}** / **{answers['Q2_n_unique_roots_before']}**
3. New physical roots generated (accepted): **{answers['Q3_n_new_physical_roots_generated']}**
4. New independent SUCCESS_NEW roots: **{answers['Q4_n_new_independent_roots_SUCCESS_NEW']}**
5. Unique successful roots after dedup: **{answers['Q5_n_unique_successful_roots_after']}**
6. Newly vs simple: **{answers['Q6_n_newly_vs_simple']}**
7. Recatch + Sliding/Pivoting: **{answers['Q7_n_recatch_plus_slide_pivot_expansion']}**
8. Real ball-impact roots: **{answers['Q8_n_ball_impact_roots']}**
9. Near-horizontal / high-motion expansion FULL: **{answers['Q9_n_near_horizontal_expansion']}** / **{answers['Q9_n_high_motion_full_expansion']}**
10. Seq length mean/median: **{answers['Q10_seq_len_mean']:.1f}** / **{answers['Q10_seq_len_median']:.1f}**; HANDOFF time mean/median: **{answers['Q10_handoff_s_mean']:.2f}s** / **{answers['Q10_handoff_s_median']:.2f}s**
11. Balance — stability {answers['Q11_stability_class_expansion']}; difficulty {answers['Q11_difficulty_expansion']}; sources {answers['Q11_source_distribution_all']}; splits {answers['Q11_split_counts']}
12. Underrepresented / limiting: {answers['Q12_underrepresented']} — {answers['limiting_factors']}
13. Suitable for unified SFT? **{answers['Q13_suitable_for_SFT']}** — {answers['Q13_note']}

## Artifacts

`BASELINE_DATASET_AUDIT.csv`, `CANONICAL_MANIFEST.csv`, `NEW_PHYSICAL_ROOTS.csv`,  
`IC_DIFFICULTY_STRATA.csv`, `MATCHED_BASELINES.csv`, `EXPERT_TRANSFER_RESULTS.csv`,  
`SEARCH_RESULTS.csv`, `VERIFIED_FULL_EXPERTS.csv`, `ROOT_COVERAGE.csv`,  
`DIVERSITY_REPORT.md`, `TRAIN_VAL_TEST_SPLITS.csv`, `DATASET_READINESS.md`,  
`trajectories/`, `snapshots/`, `raw_contact_traces/`

Scripts: `scripts/expand_k49_unified_dataset.py`, `experiments/sliding/_tmp_expand_pass2.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Unified K49 Expansion

## Verdict: {"YES (Stage-1 complete)" if answers["Q13_suitable_for_SFT"] else "NOT YET"}

- Unique SUCCESS_NEW roots: **{len(roots)}**
- Stage 1 (≥50: **{answers['stage1_50']}**
- Stage 2 ≥75: **{answers['stage2_75']}**
- Stage 3 ≥100: **{answers['stage3_100']}**
- Prior roots preserved: {audit['n_unique_roots']}
- New independent roots: {len(new_indep)}
- Stable pose-error vs ball-impact: {len(stable)} expansion STABLE_POSE_ERROR FULL / {len(ball)} ball roots

### Recommendation
Start **initial** unified Transformer SFT on `CANONICAL_MANIFEST.csv` with `TRAIN_VAL_TEST_SPLITS.csv` (group by `root_episode`).  
Continue expert search for high-motion / lost-grasp / near-horizontal before claiming full regime coverage.

Do **not** start SFT/GRPO inside this expert-generation task.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")

    div = f"""# Diversity Report — Unified K49 Expansion

## Roots
- Total unique SUCCESS_NEW roots: **{len(roots)}**
- New independent: {len(new_indep)}
- Ball-impact: {len(ball)}
- Expansion STABLE_POSE_ERROR: {len(stable)}

## Sources
{dict(src)}

## Expansion stability / difficulty
{dict(klass)}
{dict(diff)}

## Splits
{dict(split_ctr)}

## Sequence lengths
mean={answers['Q10_seq_len_mean']}, median={answers['Q10_seq_len_median']}
HANDOFF time mean/median: {answers['Q10_handoff_s_mean']} / {answers['Q10_handoff_s_median']}

## Gaps
{under}
"""
    (OUT / "DIVERSITY_REPORT.md").write_text(div, encoding="utf-8")
    print(json.dumps({k: v for k, v in answers.items() if not str(k).endswith("_list") and k != "new_independent_roots"}, indent=2))


if __name__ == "__main__":
    main()
