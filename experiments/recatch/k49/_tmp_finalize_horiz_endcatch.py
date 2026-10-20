"""Finalize horizontal endcatch reports with corrected stats + probe findings."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path("results/expert_generation/k49_horizontal_endcatch_recatch")

answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
cands = list(csv.DictReader((OUT / "ENDCATCH_CANDIDATES.csv").open(encoding="utf-8")))
matched = list(csv.DictReader((OUT / "MATCHED_BASELINES.csv").open(encoding="utf-8")))
coverage = list(csv.DictReader((OUT / "HORIZONTAL_IC_COVERAGE.csv").open(encoding="utf-8")))
partials = list(csv.DictReader((OUT / "PARTIAL_MECHANISM.csv").open(encoding="utf-8")))
phases = list(csv.DictReader((OUT / "CAPTURE_ROTATION_BRAKING.csv").open(encoding="utf-8")))

def truthy(x):
    return x in ("1", "True", True, 1)


mech = [c for c in cands if truthy(c["eccentric"]) and truthy(c["downward"]) and truthy(c["beneficial_rot"])]
mech_sup = [c for c in mech if truthy(c["supported"])]
mech_bil = [c for c in mech if truthy(c["bilateral"])]
# honest: ecc+ben+bil WITHOUT requiring downward (often no-fall)
ecc_ben_bil = [
    c for c in cands
    if truthy(c["eccentric"]) and truthy(c["beneficial_rot"]) and truthy(c["bilateral"])
]

dth = [float(c["delta_theta_deg"]) for c in mech if c.get("delta_theta_deg") not in (None, "")]
probe = {}
pp = OUT / "RAW" / "focused_brake_probe.json"
if pp.exists():
    probe_rows = json.loads(pp.read_text(encoding="utf-8"))
    good = [r for r in probe_rows if r.get("ecc") and r.get("down") and r.get("ben")]
    probe = {
        "n_mech": len(good),
        "n_bilateral": sum(1 for r in good if r.get("bil")),
        "n_NEW": sum(1 for r in good if r.get("NEW")),
        "max_dth": None if not good else max(r.get("dth") or -1 for r in good),
        "note": "Focused brake refine around Δθ≈22° seeds never obtained bilateral+NEW.",
    }

# limit among mechanism partials
lim_mech = Counter((c.get("limit_factor") or "?") for c in mech)

answers.update({
    "Q1_horizontal_endcatch_possible": len(mech) > 0,
    "Q1_n_eccentric_beneficial_downward": len(mech),
    "Q2_orientation_correction_deg": {
        "n": len(dth),
        "max": None if not dth else float(max(dth)),
        "mean": None if not dth else float(sum(dth) / len(dth)),
        "p50": None if not dth else float(sorted(dth)[len(dth) // 2]),
    },
    "Q3_supported_while_rotating": len(mech_sup) > 0,
    "Q3_n_supported_eccentric_beneficial": len(mech_sup),
    # CORRECTED: require downward for Type3 bilateral claim
    "Q4_brake_to_bilateral": len(mech_bil) > 0,
    "Q4_n_bilateral_after_eccentric_downward_beneficial": len(mech_bil),
    "Q4_n_bilateral_eccentric_beneficial_any_fall": len(ecc_ben_bil),
    "Q4_note": (
        "No candidate simultaneously had downward fall + eccentric catch + beneficial rotation "
        "+ stable bilateral. Bilateral+eccentric+beneficial without downward are non-Type3 "
        "(no genuine fall)."
    ),
    "Q5_full_SUCCESS_NEW_type3": False,
    "Q5_n_type3_ics": 0,
    "Q5_type3_ids": [],
    "Q6_limiting_factor_on_mech_candidates": dict(lim_mech.most_common()),
    "Q6_primary_limit": (
        lim_mech.most_common(1)[0][0] if lim_mech else answers.get("Q6_primary_limit")
    ),
    "Q6_interpretation": (
        "Primary physical bottleneck: after eccentric end-catch produces useful Δθ, "
        "contact is not sustained long enough to brake into stable bilateral grasp "
        "(PARTIAL_DROP_AFTER_MECH). When bilateral is restored early, inertial rotation "
        "collapses (Δθ≈0). Tradeoff between rotation freedom and capture stability."
    ),
    "Q7_matched_D_wins": sum(1 for m in matched if m["best"] == "D_TYPE3"),
    "Q7_matched_C_wins": sum(1 for m in matched if m["best"] == "C_TYPE2"),
    "Q7_n_matched": len(matched),
    "focused_brake_probe": probe,
    "n_partial_drop_after_mech": sum(1 for p in partials if p.get("status") == "PARTIAL_DROP_AFTER_MECH"),
})

(OUT / "RAW" / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

checksum = "see ACTION_MAPPING.json"
amap = OUT / "ACTION_MAPPING.json"
if amap.exists():
    checksum = json.loads(amap.read_text(encoding="utf-8")).get("checksum_sha256_16", checksum)

report = f"""# Horizontal-Cylinder End-Catch Dynamic Recatch

**Config:** `K49_tok48_wy0p25` checksum `{checksum}`  
**SUCCESS_NEW:** locked 10° / 12 mm / ≥0.5 s hold  
**No SFT/GRPO. No K49 / SUCCESS_NEW changes.**

## IC coverage

| Metric | Value |
|--------|------:|
| Selected ICs | {len(coverage)} |
| Approx-horizontal (θ or horiz≥30°) | {sum(1 for r in coverage if r.get('approx_horizontal')=='1')} |
| Center-grasp flagged | {sum(1 for r in coverage if r.get('center_grasp')=='1')} |
| Near-miss seeds revisited | {sum(1 for r in coverage if r.get('near_miss_seed')=='1')} |

See `HORIZONTAL_IC_COVERAGE.csv` for hand-frame axis, grasp axial offset, velocities, source episode.

Legal pivot-to-horizontal from mid-tilt center grasps did **not** yield additional high-θ snapshots within contact; search used real disturbance ICs with large orientation error.

## Yield

| Outcome | Count |
|---------|------:|
| VERIFIED_FULL Type3 experts | **0** |
| Eccentric ∧ downward ∧ beneficial Δθ candidates | **{len(mech)}** |
| …and contact-supported while rotating | {len(mech_sup)} |
| …and stable bilateral after catch | **{len(mech_bil)}** |
| PARTIAL_DROP_AFTER_MECH | {answers['n_partial_drop_after_mech']} |

Focused brake probe (318/031): max Δθ ≈ {probe.get('max_dth')}, bilateral among mech = {probe.get('n_bilateral')}, NEW = {probe.get('n_NEW')}.

## Q1–Q7

1. **Real horizontal-cylinder end-catch under locked K49?** **{answers['Q1_horizontal_endcatch_possible']}**  
   Measured eccentric end interception after downward relative motion on multiple ICs (esp. `hard_910318`, `hrv1_920031`).

2. **Orientation correction from eccentric impact + constrained rotation:**  
   n={answers['Q2_orientation_correction_deg']['n']},  
   max={answers['Q2_orientation_correction_deg']['max']:.2f}°,  
   mean={answers['Q2_orientation_correction_deg']['mean']:.2f}°,  
   p50={answers['Q2_orientation_correction_deg']['p50']:.2f}°  
   Focused probe recovered Δθ up to ≈22° on `hard_910318` (still no bilateral).

3. **Supported while rotating?** **{answers['Q3_supported_while_rotating']}** ({len(mech_sup)} / {len(mech)} mech candidates with contact retention during rotation window).

4. **Legal grip timing → stable bilateral after true Type3 mech?** **{answers['Q4_brake_to_bilateral']}**  
   Count with downward∧eccentric∧beneficial∧bilateral = **{len(mech_bil)}**.  
   {answers['Q4_note']}

5. **Any FULL SUCCESS_NEW Type3 without post-regrasp Slide/Pivot?** **False** (0 ICs).

6. **Limiting factor:** **{answers['Q6_primary_limit']}**  
   Among mech candidates: {dict(lim_mech.most_common())}.  
   {answers['Q6_interpretation']}

7. **Type3 vs Type2 on same ICs:** D wins {answers['Q7_matched_D_wins']}/{len(matched)};  
   C (Type2) wins {answers['Q7_matched_C_wins']}/{len(matched)} (`hrv1_920031`, `hrv1_920109`).  
   All approx-horizontal hard ICs (117/185/318/144/198/063): **NONE** under A/B/C/D.

## Artifacts

`HORIZONTAL_IC_COVERAGE.csv`, `ENDCATCH_CANDIDATES.csv`, `CAPTURE_ROTATION_BRAKING.csv`,  
`MATCHED_BASELINES.csv`, `TYPE3_FULL_EXPERTS.csv`, `FAILURES.csv`, `PARTIAL_MECHANISM.csv`,  
`DATASET_READINESS.md`, `trajectories/`, `snapshots/`, `raw_contact_traces/`,  
`scripts/generate_k49_horizontal_endcatch_recatch.py`
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

ready = f"""# Dataset Readiness — Horizontal End-Catch Type3

## Verdict: NOT YET

- FULL Type3 experts: **0**
- Mechanism evidence (eccentric+downward+beneficial): **{len(mech)}** candidates across episodes
- Stable bilateral after that mechanism: **0**
- Type2 still wins on milder near-miss ICs (`hrv1_920031`, `hrv1_920109`)

### Merge rule
Do **not** merge any Type3 rows into the unified Transformer expert pool from this directory.
Keep `PARTIAL_MECHANISM.csv` / contact traces for research; they are not VERIFIED_FULL experts.

### Blocker
Eccentric end-catch can produce ~8–22° beneficial rotation during fall/recontact, but legal K49 grip timing examined here cannot brake that motion into a stable bilateral grasp that also passes SUCCESS_NEW without a separate post-regrasp correction stage.
"""
(OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
print("FINALIZED", answers["Q1_horizontal_endcatch_possible"], answers["Q5_full_SUCCESS_NEW_type3"],
      answers["Q6_primary_limit"], answers["Q4_n_bilateral_after_eccentric_downward_beneficial"])
