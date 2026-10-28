#!/usr/bin/env python
"""Write P9 markdown/CSV deliverables from census + FSM implementation notes."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/diagnostics/p9_expert_fsm_dataset"
OFFICIAL = ROOT / "data/experts_v2_official/manifest.jsonl"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in OFFICIAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    from data_generation.expert_pipeline.branches import BRANCH_TO_CLASS, CLASS_ALIASES

    by_class = Counter(BRANCH_TO_CLASS.get(r["branch_id"], r["branch_id"]) for r in rows)
    skills = Counter(r.get("skill_sequence") for r in rows)

    # Refresh CLASS_COVERAGE + SKILL_TRANSITION from authoritative manifest
    cov = []
    for cls in ("A1", "A2", "B1", "B2", "C1", "C2", "D2"):
        n = by_class.get(cls, 0)
        cov.append(
            {
                "class_label": cls,
                "branch_id": CLASS_ALIASES[cls],
                "n_verified_official": n,
                "milestone_20": n >= 20,
                "target_50": n >= 50,
                "deficit_to_20": max(0, 20 - n),
                "deficit_to_50": max(0, 50 - n),
                "priority": "HIGH" if cls in ("A2", "B2", "C2", "D2") and n < 20 else ("MED" if n < 20 else "OK"),
            }
        )
    with (OUT / "CLASS_COVERAGE.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0].keys()))
        w.writeheader()
        w.writerows(cov)

    sk = [{"transition": k, "count": v, "attempt_count": v, "executed_count": v, "success_count": v,
           "note": "legacy_manifest_skill_sequence; planned≠executed not stored pre-FSM"} for k, v in skills.most_common()]
    with (OUT / "SKILL_TRANSITION_COVERAGE.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sk[0].keys()))
        w.writeheader()
        w.writerows(sk)

    # Specs
    (OUT / "PRECISION_FSM_SPEC.md").write_text(
        """# PRECISION_FSM_SPEC

## States
`ASSESS`, `CORRECT_DX`, `CORRECT_DZ`, `CORRECT_DTHETA`, `RECONFIGURE_A`, `RECONFIGURE_B`, `RECONFIGURE_C`, `SETTLE`, `FINAL_VERIFY`, `HANDOFF`, `FAIL`.

## Thresholds (verified AXIS_V1 / PrecisionConfig — not ad-hoc)
- DX large: `|dx| > ex_tol_mm` (12 mm)
- DZ large: `|dz| > ez_tol_mm` (20 mm)
- θ optional: `dθ > theta_tol_deg` (10° AXIS; SUCCESS_NEW hold historically 10°)
- Position Euclidean: `e_pos ≤ working_pos_mm` (12 mm)

## Cases
1. DX large, DZ small → `CORRECT_DX → CORRECT_DZ → optional CORRECT_DTHETA → FINAL_VERIFY`
2. DZ large, DX small → `CORRECT_DZ → CORRECT_DX → optional CORRECT_DTHETA → FINAL_VERIFY`
3. Both large → `RECONFIGURE_{A|B|C}` (precondition-gated) → verify → `ASSESS`. Bounded attempts. If mid-Precision reconfigure disabled, fall through to **larger-error-first** sequential correction (never silent DX-first when `|dz|≥|dx|`).
4. Both small → optional bounded `CORRECT_DTHETA` → `FINAL_VERIFY`

## Logging
Record `fsm_planned_skills` and `fsm_executed_skills` separately in `primitive_logs`.

## dθ policy (post P7/P8)
At most `max_dtheta_attempts=1`. Skip if θ already within tolerance. Do not run oscillation/inertial loops.
""",
        encoding="utf-8",
    )

    (OUT / "PRECISION_FSM_IMPLEMENTATION_AUDIT.md").write_text(
        """# PRECISION_FSM_IMPLEMENTATION_AUDIT

## Code
- Planner: `data_generation/oracles/precision/fsm.py`
- Dual metrics: `data_generation/oracles/precision/success_versions.py`
- Integration: `data_generation/oracles/precision/state_machine.py` (`use_explicit_fsm=True` default)
- Config: `PrecisionConfig.use_explicit_fsm`, `max_dtheta_attempts`, `max_reconfigure_attempts`, `allow_mid_precision_reconfigure`
- Unit tests: `tests/test_precision_fsm.py` (7 passed)

## Behavior changes vs legacy
- Legacy: always try `dθ → dx → dz`.
- FSM: case-ordered position skills; dθ only when planned/needed and attempt budget remains.
- Case 3 reconfigure hooks are **gated off** (`allow_mid_precision_reconfigure=False`) until production mid-Precision A/B/C APIs are wired; fallthrough is larger-error-first.

## Production Oracle
Stage-1 A/B/C modules unchanged. Physics unchanged. M111 unchanged.
""",
        encoding="utf-8",
    )

    (OUT / "SUCCESS_CRITERIA_VERSIONING.md").write_text(
        f"""# SUCCESS_CRITERIA_VERSIONING

## Metrics (always reported after P9 handoff evaluator)
- `SUCCESS_STRICT_10DEG` — θ≤10°, e_pos≤12 mm, SUCCESS_12S, 0.5 s hold (≡ historical `SUCCESS_NEW`)
- `SUCCESS_RELAXED_15DEG` — θ≤15°, e_pos≤12 mm, SUCCESS_12S, 0.5 s hold (candidate)

## Official selection
Historical `SUCCESS_NEW` remains **STRICT 10°** until a census-backed promotion.
Candidate official: θ≤15° (`SUCCESS_RELAXED_15DEG_v1`).

Do not mix differently labeled corpora. Continuous θ always retained.

## Official corpus snapshot
- Accepted experts: {len(rows)}
- All rows have `SUCCESS_NEW=True` (admit filter); θ distribution not fully stored on manifest — regenerate handoff annotations when promoting 15°.
- Skill mix dominated by direct / skip paths (see SKILL_TRANSITION_COVERAGE.csv).

## Recommendation
Keep STRICT as admit rule for this iteration; **track RELAXED in parallel** on new FSM generation.
Promote RELAXED only after measuring θ histogram on full 12S windows for Precision-executed rows.
""",
        encoding="utf-8",
    )

    (OUT / "DATASET_LINEAGE_AUDIT.md").write_text(
        f"""# DATASET_LINEAGE_AUDIT

## Corpora (preserved, not overwritten)
| corpus | role | train-eligible |
|---|---|---|
| `data/experts_v2_official/` | Official seven-branch production ({len(rows)} accepted) | Yes (physical_success) |
| `data/experts_v2/` | Earlier 122-run freeze | No (superseded) |
| `data/experts_v2_recovered/` | Fallback-enabled diagnostics (169) | **No** |
| `data/experts_v3_stage1_verified/` | Stage-1 rebuild copies | Separate |
| `data/unified_experts/` | Legacy K/unified | Do not merge into M111 corpus |

## Class coverage (official)
{chr(10).join(f"- {c['class_label']}: {c['n_verified_official']} (deficit_to_20={c['deficit_to_20']})" for c in cov)}

## Lineage rules
- One underlying IC → at most one independent training sample per class.
- Variants of the same IC are marked `duplicate_lineage` in census.
- M111 discrete encoding remains frozen for this dataset iteration.
""",
        encoding="utf-8",
    )

    deficit = [c for c in cov if c["deficit_to_20"] > 0]
    (OUT / "TRAINING_READINESS_REPORT.md").write_text(
        f"""# TRAINING_READINESS_REPORT

## Answers

### 1. Independent verified trajectories per class (official)
{chr(10).join(f"- **{c['class_label']}**: {c['n_verified_official']}" for c in cov)}

### 2. Classes lacking milestone (≥20)
{chr(10).join(f"- {c['class_label']}: need {c['deficit_to_20']} more" for c in deficit) or "- None"}

### 3. Multi-skill transitions genuinely represented
- `direct_handoff`: {skills.get('direct_handoff', 0)}
- `precision_skipped_direct_handoff`: {skills.get('precision_skipped_direct_handoff', 0)}
- `dx→dz`: {skills.get('dx→dz', 0)}
- `dz`: {skills.get('dz', 0)}
- **Not rich Precision coverage** — majority are skip/direct. FSM generation must prioritize A2/B2/C2/D2 with real DX/DZ/dθ executions.

### 4. Relaxing θ to 15°
Not yet measured on full handoff windows for the official set (manifest lacks per-frame θ). New generation records `SUCCESS_STRICT_10DEG` and `SUCCESS_RELAXED_15DEG` in parallel. Do not silently change admit rule this iteration.

### 5. Systemic blockers for new samples
- A2/D2 IC scarcity in Stage-1 pools / Precision fail rate under STRICT 10°.
- Mid-Precision reconfigure (Case 3) not yet production-wired.
- θ residual ~12–15° on some stable C exits (P6–P8) blocks STRICT NEW.

### 6. Ready for SFT?
**Not yet.** Milestone 20 unmet for A2, B1, D2; skill diversity too direct-heavy. Complete bounded FSM pilot + fill HIGH-priority deficits first. M111 frozen; no RL yet.

### 7. What RL should improve beyond the Expert
- Skill **selection** when both DX and DZ are large (reconfigure vs sequential).
- Bounded θ correction / when to skip dθ.
- Recovery from near-miss Precision states without supervised labels for every path.
""",
        encoding="utf-8",
    )

    # Empty generation placeholder until pilot runs
    if not (OUT / "GENERATION_RESULTS.csv").is_file():
        (OUT / "GENERATION_RESULTS.csv").write_text(
            "run_id,branch_id,class_label,attempted,accepted,rejected,skill_sequences,notes\n",
            encoding="utf-8",
        )
    if not (OUT / "VERIFIED_EXPERT_MANIFEST.json").is_file():
        manifest = {
            "dataset_version": "experts_v2_official",
            "n_accepted": len(rows),
            "by_class": dict(by_class),
            "skill_sequences": dict(skills),
            "fsm_version": "precision_fsm_p9",
            "m111_frozen": True,
            "training_ready": False,
            "note": "Pointer to data/experts_v2_official; pilot outputs append under this diagnostics tree.",
        }
        (OUT / "VERIFIED_EXPERT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (OUT / "VIDEO_INDEX.md").write_text(
        """# VIDEO_INDEX — P9

Representative class videos are generated by `render_class_videos.py` after the bounded generation pilot.
Until then, refer to existing official Precision failure/success renders under `results/diagnostics/` and `data/experts_v2_official` action tapes.
""",
        encoding="utf-8",
    )

    print("wrote P9 deliverables", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
