"""Finalize capture-brake deliverables with honest Type3 vs capture-success labeling."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

OUT = Path("results/expert_generation/k49_type3_capture_brake")

answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
full_rows = list(csv.DictReader((OUT / "FULL_EXPERTS.csv").open(encoding="utf-8")))
hyb_rows = list(csv.DictReader((OUT / "HYBRID_EXPERTS.csv").open(encoding="utf-8"))) if (OUT / "HYBRID_EXPERTS.csv").exists() else []
cands = list(csv.DictReader((OUT / "CAPTURE_BRAKE_CANDIDATES.csv").open(encoding="utf-8")))
branches = list(csv.DictReader((OUT / "BRANCH_SNAPSHOTS.csv").open(encoding="utf-8")))
coverage = list(csv.DictReader((OUT / "ROOT_EPISODE_COVERAGE.csv").open(encoding="utf-8")))
diag = list(csv.DictReader((OUT / "RAW" / "geometry_diagnosis.csv").open(encoding="utf-8")))
beyond = list(csv.DictReader((OUT / "BEYOND_RECOVERY.csv").open(encoding="utf-8"))) if (OUT / "BEYOND_RECOVERY.csv").exists() else []

# Honest reclassification of the SUCCESS_NEW trajectory:
# eccentric+downward+bilateral+SUCCESS_NEW, but rot_window≈2.15° < 3° beneficial threshold.
# Keep as CAPTURE_BRAKE_SUCCESS_NEW; strict FULL Type3 count = 0.
capture_success = []
strict_full = []
for r in full_rows:
    # known measured values from verification
    r2 = dict(r)
    r2["rot_window_deg"] = 2.1493859774813107
    r2["m3_delta_theta_deg"] = 1.743744410329013
    r2["m3_beneficial_ge3"] = 0
    r2["eccentric"] = 1
    r2["downward"] = 1
    r2["post_corr"] = 0
    r2["label"] = "CAPTURE_BRAKE_SUCCESS_NEW_MODEST_ROTATION"
    capture_success.append(r2)
    # also record as hybrid-like (Type3 catch lineage → success without large inertial Δθ)
    hyb_rows.append({
        "ic_id": r["ic_id"],
        "tag": r["tag"].replace("full_", "hybrid_modest_rot_"),
        "expert_type": "TYPE3_CATCH_CAPTURE_BRAKE_MODEST_ROT",
        "SUCCESS_NEW": r["SUCCESS_NEW"],
        "win_mean_pos": r["win_mean_pos"],
        "win_mean_theta": r["win_mean_theta"],
        "hold_both_s": r.get("hold_both_s"),
        "rot_window_deg": 2.1493859774813107,
        "tokens": r["tokens"],
        "branch": r.get("branch"),
        "note": "Passes SUCCESS_NEW with eccentric catch + legal capture/brake; recapture Δθ<3°.",
    })

# clear strict FULL
strict_full_path = OUT / "FULL_EXPERTS.csv"
with strict_full_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[
        "ic_id", "tag", "expert_type", "SUCCESS_NEW", "win_mean_pos", "win_mean_theta",
        "hold_both_s", "tokens", "branch", "mapping_checksum", "rot_window_deg", "note",
    ])
    w.writeheader()
    # empty: no strict FULL with beneficial≥3°

with (OUT / "CAPTURE_BRAKE_SUCCESS_NEW.csv").open("w", newline="", encoding="utf-8") as f:
    if capture_success:
        w = csv.DictWriter(f, fieldnames=list(capture_success[0].keys()))
        w.writeheader()
        w.writerows(capture_success)

with (OUT / "HYBRID_EXPERTS.csv").open("w", newline="", encoding="utf-8") as f:
    if hyb_rows:
        # unify keys
        keys = []
        for r in hyb_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(hyb_rows)

n_C = sum(1 for r in cands if r.get("C") == "1")
n_D = sum(1 for r in cands if r.get("D") == "1")
n_B = sum(1 for r in cands if r.get("B") == "1")

mode_ctr = Counter()
for d in diag:
    try:
        for m in json.loads(d["failure_modes"]):
            mode_ctr[m] += 1
    except Exception:
        pass

# helpful tokens already in answers
answers.update({
    "Q1_can_convert_to_bilateral": n_C > 0,
    "Q1_n_C_milestones": n_C,
    "Q1_n_D_stable": n_D,
    "Q4_n_full_type3_strict_beneficial_ge3": 0,
    "Q4_n_capture_brake_SUCCESS_NEW": len(capture_success),
    "Q4_capture_brake_ids": sorted({r["ic_id"] for r in capture_success}),
    "Q4_note": (
        "Strict Type3 FULL (eccentric∧downward∧Δθ≥3°∧bilateral∧no post∧SUCCESS_NEW)=0. "
        "One end-to-end SUCCESS_NEW on hrv1_920031 with eccentric catch + capture/brake, "
        "but recapture orientation improvement only ≈2.15° (standard beneficial threshold 3°). "
        "See CAPTURE_BRAKE_SUCCESS_NEW.csv."
    ),
    "Q5_n_hybrid_ics": len({r["ic_id"] for r in hyb_rows}),
    "Q5_hybrid_ids": sorted({r["ic_id"] for r in hyb_rows}),
    "Q5_note": (
        "Modest-rotation SUCCESS_NEW capture-brake counted as Type3-catch→stabilize hybrid lineage; "
        "no separate post Slide/Pivot tokens in the winning sequence."
    ),
    "Q6_n_beyond": len(beyond),
    "Q6_note": (
        "hard_910318: many continuations restore bilateral (C/D) from branch states but none "
        "pass SUCCESS_NEW within budget — residual pose/hold after capture. "
        "Contact-retaining 031 branches are recoverable to bilateral."
    ),
    "Q3_assessment": (
        "For hrv1_920031: failure was primarily timing/search — early secure grip (token 0) and "
        "soft brace (12) from intermediate states restore bilateral and can pass SUCCESS_NEW. "
        "For hard_910318: after bilateral restore, SUCCESS_NEW still fails (pose/hold), suggesting "
        "limited control authority or insufficient time-resolution to finish braking into the "
        "SUCCESS_NEW pose window — not merely missing the bilateral milestone."
    ),
    "measured_winning_tokens": [30, 12, 12, 8, 0, 0, 0, 0, 0, 0, 0],
    "measured_rot_window_deg": 2.1493859774813107,
})
(OUT / "RAW" / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

checksum = "33d4820d1ec09c8b"
amap = OUT / "ACTION_MAPPING.json"
if amap.exists():
    checksum = json.loads(amap.read_text(encoding="utf-8")).get("checksum_sha256_16", checksum)

diag_md = f"""# Contact Geometry Diagnosis

## Branch-point failure modes (pre-continuation)

| Mode | Count |
|------|------:|
"""
for m, c in mode_ctr.most_common():
    diag_md += f"| `{m}` | {c} |\n"

diag_md += f"""

## What the geometry shows

Typical contact-retaining Type3 intermediates (especially `hrv1_920031`) have:

- Unilateral eccentric contact near a cylinder end (axial offset ≈20 mm).
- Non-zero relative angular velocity (often flagged `excessive_relative_angular_velocity`).
- Free cylinder end sometimes outside comfortable opposing-finger reach.
- Soft/partial grip opening rather than fully closed secure grip.

## What legal continuations achieve

- **Token 0 (secure hold)** and **token 12 (soft/mid unload brace)** dominate successful bilateral restorations.
- Wrist tokens **4 / 6 / 10** and chase **17 / 33** appear in B/C successes, helping reduce |ω| while tracking.
- From `hrv1_920031` max-useful-orientation branches, a short continuation of secure zeros after a brief soft+chase prefix restores bilateral and passes SUCCESS_NEW.

## `hard_910318`

Branch continuations often reach milestones **C** and **D** (bilateral restore / short stable hold) but do not produce SUCCESS_NEW in the allocated budget. Geometry still shows high |ω| and reach issues at the branch point; capture is possible, final pose/hold is not.

See `RAW/geometry_diagnosis.csv`.
"""
(OUT / "CONTACT_GEOMETRY_DIAGNOSIS.md").write_text(diag_md, encoding="utf-8")

report = f"""# Type 3 Capture–Brake from Intermediate States

**Config:** `K49_tok48_wy0p25` checksum `{checksum}`  
**Source mechanism candidates:** `k49_horizontal_endcatch_recatch` (45 contact-retaining on `hrv1_920031`; plus `hard_910318` high-Δθ)  
**No SFT/GRPO. No K49 / SUCCESS_NEW changes.**

## Search summary

| Item | Value |
|------|------:|
| Branch snapshots | {len(branches)} |
| Continuations scored | {len(cands)} |
| Rollouts | {answers.get('n_rollouts')} |
| Wall time (main batch) | {answers.get('wall_s')} s |

## Milestones

| Milestone | Count |
|-----------|------:|
| B — angular velocity reduced | {n_B} |
| C — bilateral restored | {n_C} |
| D — stable bilateral run | {n_D} |
| Strict Type3 FULL (Δθ≥3° + SUCCESS_NEW) | **0** |
| Capture–brake SUCCESS_NEW (modest Δθ≈2.1°) | **{len(capture_success)}** |

Winning sequence (reproducible from root `hrv1_920031`):  
`{answers.get('measured_winning_tokens')}`  
rot_window≈{answers.get('measured_rot_window_deg'):.2f}°, eccentric axial≈20.4 mm, hold≈2.0 s.

## Q1–Q6

1. **Convert contact-retaining intermediate → stable bilateral with legal K49?** **Yes.**  
   C={n_C}, D={n_D} continuation milestones. End-to-end bilateral+SUCCESS_NEW achieved on `hrv1_920031`.

2. **Helpful actions:** secure **0**, soft brace **12**, wrists **4/6/10**, chase **17/33**  
   (see `RAW/answers.json` Q2 token counts).

3. **Remaining failure class:** {answers['Q3_assessment']}

4. **Full Type3 SUCCESS_NEW (strict beneficial ≥3°):** **0**  
   Capture–brake SUCCESS_NEW with eccentric catch (modest rotation ≈2.1°): **{len(capture_success)}** (`hrv1_920031`).  
   {answers['Q4_note']}

5. **Type3→Type2 / modest-rot hybrids:** **{answers['Q5_n_hybrid_ics']}** IC(s) {answers['Q5_hybrid_ids']}  
   {answers['Q5_note']}

6. **Beyond recovery:** {answers['Q6_note']}  
   Beyond-list size={len(beyond)} (`BEYOND_RECOVERY.csv`).

## Artifacts

`BRANCH_SNAPSHOTS.csv`, `CONTACT_GEOMETRY_DIAGNOSIS.md`, `CAPTURE_BRAKE_CANDIDATES.csv`,  
`FULL_EXPERTS.csv` (strict; empty), `CAPTURE_BRAKE_SUCCESS_NEW.csv`, `HYBRID_EXPERTS.csv`,  
`FAILURES.csv`, `ROOT_EPISODE_COVERAGE.csv`, `DATASET_READINESS.md`,  
`branch_snapshots/`, `trajectories/`, `raw_contact_traces/`,  
`scripts/generate_k49_type3_capture_brake.py`
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

ready = f"""# Dataset Readiness — Type3 Capture–Brake

## Verdict: CONDITIONAL

- Strict Type3 FULL (Δθ≥3° during recapture + SUCCESS_NEW): **0**
- Capture–brake SUCCESS_NEW with verified eccentric end-catch on `hrv1_920031`: **{len(capture_success)}**
- Rotation on the successful traj is modest (≈2.1°); do **not** claim large inertial-pivot Type3 without the ≥3° evidence.

### Merge rule
- Safe to merge `CAPTURE_BRAKE_SUCCESS_NEW.csv` / trajectory as a recovery expert with mechanism tag  
  `TYPE3_CATCH_CAPTURE_BRAKE_MODEST_ROT` (or SIMPLE_RECATCH+eccentric evidence), **not** as a classic large-Δθ Type3 dynamic recatch.
- Keep diagnostic branch snapshots out of the 39D policy stream.
"""
(OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
print("FINAL", answers["Q1_can_convert_to_bilateral"], answers["Q4_n_full_type3_strict_beneficial_ge3"],
      answers["Q4_n_capture_brake_SUCCESS_NEW"], answers["Q5_n_hybrid_ics"])
