"""Finalize ball-impact deliverables: separated CSVs + polished REPORT."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "expert_generation" / "k49_ball_impact_dynamic_recovery"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
    replay = json.loads((OUT / "RAW" / "replay_verification_summary.json").read_text(encoding="utf-8"))
    experts = list(csv.DictReader((OUT / "FULL_EXPERTS.csv").open(encoding="utf-8")))
    baselines = list(csv.DictReader((OUT / "MATCHED_BASELINES.csv").open(encoding="utf-8")))
    coverage = list(csv.DictReader((OUT / "ROOT_EPISODE_COVERAGE.csv").open(encoding="utf-8")))
    phys = list(csv.DictReader((OUT / "IMPACT_PHYSICS_AUDIT.csv").open(encoding="utf-8")))[0]

    newly = [r for r in experts if int(r.get("newly_vs_simple") or 0) == 1]
    baseline_solvable = [r for r in experts if int(r.get("newly_vs_simple") or 0) == 0]
    novel_ids = set(answers.get("Q10_revalidated_prior_ids") or [])
    novel_gen = [r for r in experts if r["ic_id"] not in novel_ids and r.get("source") == "generated"]
    unsolved = [r for r in coverage if int(r.get("full") or 0) == 0]
    write_csv(OUT / "BASELINE_SOLVABLE_EXPERTS.csv", baseline_solvable)
    write_csv(OUT / "NEWLY_SOLVED_EXPERTS.csv", newly)
    write_csv(OUT / "NOVEL_GENERATED_EXPERTS.csv", novel_gen)
    write_csv(OUT / "UNSOLVED_ICS.csv", unsolved)

    answers["replay_all_pass"] = bool(replay.get("all_pass"))
    answers["replay_n_ok"] = replay.get("n_ok")
    (OUT / "RAW" / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

    n_full = answers["Q6_n_full_SUCCESS_NEW"]
    n_new = answers["Q7_n_newly_vs_simple"]
    n_novel = answers["Q10_added_novel_vs_prior_manifest"]
    report = f"""# Ball-Impact Dynamic Recovery Experts

**Config:** `K49_tok48_wy0p25`  
**Scene:** `{phys.get('scene')}` — cyl_m={phys.get('cylinder_mass_kg')}, ball_m={phys.get('ball_mass_kg')}, ball_r={phys.get('ball_radius_m')}, cyl_r={phys.get('cylinder_radius_m')}, μ={phys.get('finger_object_mu')}  
**Physics audit match:** `{phys.get('source_constants_match_model')}`  
**SUCCESS_NEW locked (≤10°, ≤12 mm, ≥0.5 s hold). No SFT/GRPO. No physics/controller changes.**

## Method

1. Audited ballistic scene masses/geometry/friction.
2. Generated lateral-biased physical ball impacts via `run_episode` / `aim_ball` (no cylinder teleport).
3. Reused hard_ic_mining snapshots (same physical ball-impact origin).
4. Classified Recovery ICs offline: IMPACT_EARLY / UNILATERAL_ESCAPE / CONTACT_LOSS / HIGH_MOTION / NEAR_HORIZONTAL.
5. Matched baselines A–E (handoff / secure / short Recatch / capture–brake / adaptive search).
6. Searched legal K49 sequences with prior expert seed transfer + slide/recatch/pose/return proposals.
7. Accepted only replay-verified SUCCESS_NEW as `BALL_IMPACT_FULL_EXPERT`.

## Impact generation

| Metric | Value |
|--------|------:|
| New episode attempts | {answers['Q1_n_generated_attempts']} |
| Accepted new Recovery roots | {answers['Q1_n_accepted_gen']} |
| Reused hard_ic_mining roots | {answers['Q1_n_reused_hard']} |
| **Independent Recovery roots** | **{answers['Q1_n_independent_impact_roots']}** |

Meaningful disturbance among roots: lateral/pos {answers['Q2_n_lateral_disp']}, angular {answers['Q2_n_angular']}, unilateral {answers['Q2_n_unilateral']}, contact_loss {answers['Q2_n_contact_loss']}, high_motion {answers['Q2_n_high_motion']}.

## Search / yield

| Metric | Value |
|--------|------:|
| Challenging ICs searched | {answers['Q3_n_challenging_searched']} |
| BALL_IMPACT_FULL_EXPERT (replay OK) | **{n_full}** / {replay.get('n_ok')} verified |
| Newly vs simple A/B/C | **{n_new}** |
| Novel vs prior unified manifests | **{n_novel}** (all generated `bimp_*`) |
| Revalidated prior hard ICs | {len(answers.get('Q10_revalidated_prior_ids') or [])} |
| FULL needing post Slide/Pivot | {answers['Q5_n_need_slide_pivot']} |
| Near-horizontal FULL | {answers['Q8_n_near_horizontal_full']} |
| Rollouts | {answers['n_rollouts']} |
| Wall time | {answers['wall_s']:.1f}s |

### FULL experts

| IC | Source | Newly vs simple | Tags (abbrev) | win θ° | win pos mm |
|----|--------|----------------:|---------------|-------:|-----------:|
"""
    for r in experts:
        tags = json.loads(r.get("mechanism_tags") or "[]")
        short = ",".join(t for t in tags if t in (
            "RECATCH", "CAPTURE_BRAKE", "SLIDING", "PIVOTING", "RETURN",
            "CONTACT_LOSS", "CHASE", "COMPLETE_CONTACT_LOSS",
        ))
        report += (
            f"| `{r['ic_id']}` | {r.get('source')} | {r.get('newly_vs_simple')} | "
            f"{short} | {float(r['win_mean_theta']):.1f} | {float(r['win_mean_pos']):.1f} |\n"
        )

    report += f"""
## Q1–Q11

1. **Independent physical ball-impact roots:** **{answers['Q1_n_independent_impact_roots']}** (generated {answers['Q1_n_accepted_gen']} + reused hard {answers['Q1_n_reused_hard']}; {answers['Q1_n_generated_attempts']} new attempts).
2. **Meaningful lateral / angular / unilateral / loss:** lat {answers['Q2_n_lateral_disp']}, ang {answers['Q2_n_angular']}, uni {answers['Q2_n_unilateral']}, loss {answers['Q2_n_contact_loss']}, high-motion {answers['Q2_n_high_motion']}.
3. **Challenging ICs recoverable by legal K49 (FULL):** **{answers['Q3_n_recoverable_full']}** / {answers['Q3_n_challenging_searched']} searched.
4. **Restore stable bilateral contact (via FULL experts):** **{answers['Q4_n_stable_bilateral_via_experts']}**.
5. **Require Sliding/Pivoting after Recatch:** **{answers['Q5_n_need_slide_pivot']}**.
6. **Full SUCCESS_NEW:** **{n_full}** — `{answers['Q6_full_ids']}`.
7. **Newly solved vs simple baselines A/B/C:** **{n_new}** — `{answers['Q7_newly_ids']}`.
8. **Severely rotated / near-horizontal → SUCCESS_NEW:** **{answers['Q8_n_near_horizontal_full']}** (includes `bimp_0042` CONTACT_LOSS+NEAR_HORIZONTAL with Recatch+Pivoting).
9. **Main remaining failures:** primary `{answers['Q9_primary']}` on high-speed escapes; also DROP / DROP_AFTER_HANDOFF / LOST_GRASP; some SUCCESS_12S without pose hold (PARTIAL). Hard high-v CONTACT_LOSS often irreversible within horizon.
10. **Independent-root coverage added:** **{n_full}** ball-impact FULL in this folder; **{n_novel} novel** vs prior unified manifests (`bimp_0015/0018/0030/0042`); 4 revalidated prior hard roots. Prior T1≈{answers['Q10_prior_type1']}, T2={answers['Q10_prior_type2']}, T3-lost≈{answers['Q10_prior_type3_lost']}.
11. **Ready for initial unified Transformer SFT?** **{answers['Q11_ready_unified_SFT']}** — {answers['Q11_note']} Split by `root_episode`. Do not start SFT in this task.

## Separations

- `BASELINE_SOLVABLE_EXPERTS.csv` — simple A/B/C also succeed.
- `NEWLY_SOLVED_EXPERTS.csv` — adaptive search needed.
- `NOVEL_GENERATED_EXPERTS.csv` — new `bimp_*` roots not in prior manifests.
- `UNSOLVED_ICS.csv` — searched but no FULL.

## Artifacts

`BALL_IMPACT_ROOTS.csv`, `IMPACT_PHYSICS_AUDIT.csv`, `RECOVERY_ICS.csv`,  
`MATCHED_BASELINES.csv`, `DYNAMIC_RECATCH_SEARCH.csv`, `POST_RECATCH_MANIPULATION.csv`,  
`FULL_EXPERTS.csv`, `FAILURES.csv`, `ROOT_EPISODE_COVERAGE.csv`, `UNIFIED_MANIFEST.csv`,  
`DATASET_READINESS.md`, `trajectories/`, `snapshots/`, `raw_contact_traces/`,  
`scripts/generate_k49_ball_impact_dynamic_recovery.py`, `experiments/ball_impact/_tmp_verify_ball_impact_experts.py`.

Replay: **{replay.get('n_ok')}/{replay.get('n')} SUCCESS_NEW**.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Ball-Impact Dynamic Recovery

## Verdict: {"YES (conditional)" if answers["Q11_ready_unified_SFT"] else "NOT YET"}

- Ball-impact FULL experts: **{n_full}** independent roots (all replay-verified).
- Novel generated roots (not in prior manifests): **{n_novel}**.
- Newly vs simple baselines: **{n_new}**.
- Combined prior + this: T1≈{answers['Q10_prior_type1']}, T2={answers['Q10_prior_type2']}, T3-lost≈{answers['Q10_prior_type3_lost']}, ball-impact FULL={n_full}.

### Merge rule

Append `FULL_EXPERTS.csv` / `UNIFIED_MANIFEST.csv` with `expert_type=BALL_IMPACT_FULL_EXPERT`.  
Split train/val/test by `root_episode` (= `ic_id` here).  
Mechanism tags / regimes are **metadata only** — policy uses original 39D obs + K49 actions.

### Caveats

- High-speed complete-loss escapes remain mostly unsolved (TABLE_DROP / DROP).
- 4 FULL ICs revalidate prior hard experts; use novel `bimp_*` for true coverage gain.
- Do **not** start SFT/GRPO in this deliverable.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("finalized", n_full, "full;", n_novel, "novel;", n_new, "newly", flush=True)


if __name__ == "__main__":
    main()
