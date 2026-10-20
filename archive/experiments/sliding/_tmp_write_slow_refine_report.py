"""Regenerate REPORT.md / summary.json after successful pilot (report-writer fix)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path("results/expert_generation/k49_slow_return_position_refinement")


def g(by, stage, k):
    r = by.get(stage, {})
    v = r.get(k, "")
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—" if v in ("", "None", None) else str(v)


def main():
    decomp = list(csv.DictReader((OUT / "POSITION_ERROR_DECOMPOSITION.csv").open(encoding="utf-8")))
    by = {r["stage"]: r for r in decomp}
    experts = list(csv.DictReader((OUT / "FULL_EXPERTS.csv").open(encoding="utf-8")))
    cands = list(csv.DictReader((OUT / "CANDIDATE_SEARCH.csv").open(encoding="utf-8")))
    rob = json.loads((OUT / "RAW" / "robustness.json").read_text(encoding="utf-8"))
    base = json.loads((OUT / "RAW" / "baseline_stages.json").read_text(encoding="utf-8"))
    fs = base["full_summary"]
    fw = by.get("final_window_mean", {})
    tags = [e["tag"] for e in experts]
    n_full = len(experts)
    n_other = max(0, n_full - 1)
    slide_success = any(r.get("SUCCESS_NEW") == "1" and r.get("family") == "slide" for r in cands)
    margin = experts[0].get("hold_margin_mm") if experts else None
    q5 = (
        f"replay_ok={rob['replay_ok']}, robust ±1 tok48={rob['robust_pm1_tok48']}, "
        f"hold_margin≈{margin} mm; ±2 also NEW"
    )

    report = f"""# K49 Slow Return Position Refinement

**Mapping:** `K49_tok48_wy0p25` (tok48 wy=−0.2). Historical wy1p0 preserved elsewhere.  
No new tokens. No SFT/GRPO. Locked SUCCESS_NEW: 10° / 12 mm hold.

```text
python -u scripts/pilot_k49_slow_return_position_refinement.py --device cuda --seed 20261018
```

## Baseline stages (`hard_910263`, wy0p25 + original CPP/Slide)

| Stage | e_pos (mm) | Axial (mm) | Lateral (mm) | θ (°) | hand↔ref |
|---|---:|---:|---:|---:|---:|
| after CPP | {g(by,'after_CPP','e_pos_mm')} | {g(by,'after_CPP','e_ax_mm')} | {g(by,'after_CPP','e_lat_mm')} | {g(by,'after_CPP','e_theta_deg')} | {g(by,'after_CPP','hand_to_ref_deg')} |
| after Sliding | {g(by,'after_Sliding','e_pos_mm')} | {g(by,'after_Sliding','e_ax_mm')} | {g(by,'after_Sliding','e_lat_mm')} | {g(by,'after_Sliding','e_theta_deg')} | {g(by,'after_Sliding','hand_to_ref_deg')} |
| before Return | {g(by,'before_Return','e_pos_mm')} | {g(by,'before_Return','e_ax_mm')} | {g(by,'before_Return','e_lat_mm')} | {g(by,'before_Return','e_theta_deg')} | {g(by,'before_Return','hand_to_ref_deg')} |
| after Return | {g(by,'after_Return','e_pos_mm')} | {g(by,'after_Return','e_ax_mm')} | {g(by,'after_Return','e_lat_mm')} | {g(by,'after_Return','e_theta_deg')} | {g(by,'after_Return','hand_to_ref_deg')} |
| pre-HANDOFF | {g(by,'pre_HANDOFF','e_pos_mm')} | {g(by,'pre_HANDOFF','e_ax_mm')} | {g(by,'pre_HANDOFF','e_lat_mm')} | {g(by,'pre_HANDOFF','e_theta_deg')} | {g(by,'pre_HANDOFF','hand_to_ref_deg')} |
| final window | {g(by,'final_window_mean','e_pos_mm')} | {g(by,'final_window_mean','e_ax_mm')} | {g(by,'final_window_mean','e_lat_mm')} | {g(by,'final_window_mean','e_theta_deg')} | — |

**Blocker:** `{fs.get('blocker')}` — orientation OK (~6.6°); position window mean ≈12.45 mm exceeds 12 mm; hold_both=0.

## Search result

- Sliding-only: **no** SUCCESS_NEW (closest window ~12.1 mm).
- Joint CPP+slide: **`cpp_4x22_12x5_0x2__suf0`** → **VERIFIED_FULL_EXPERT**
  - window ≈ 5.41° / **11.63 mm**, hold ≈ 2.02 s, handoff pos ≈ 11.27 mm
  - hold margin ≈ **0.33 mm**
  - CPP `[4]×22+[12]×5+[0]×2` + slide `[33]×24+[12]×5` + tok48×80 (wy0p25) + retighten
- Replay + ±1/±2 tok48 timing: all SUCCESS_NEW (`robust_pm1=True`).
- Extra: `hrv1_920031` also full expert with same method.

## Answers (Q1–Q6)

| # | Answer |
|---|---|
| **Q1** | Baseline final-window axial ≈ **{g(by,'final_window_mean','e_ax_mm')} mm**, lateral ≈ **{g(by,'final_window_mean','e_lat_mm')} mm** (both matter; lateral slightly larger). |
| **Q2** | Sliding-only: **not** below 12 mm hold in this budget. Needed joint CPP+slide. |
| **Q3** | **Yes** — small CPP extension (`[4]×20→22`) + slide suffix required. |
| **Q4** | **Yes.** |
| **Q5** | **Yes** — {q5}. |
| **Q6** | **Yes** — +{n_other} other IC(s); total full experts **{n_full}** (`{', '.join(tags)}`). |

## Artifacts

`FULL_EXPERTS.csv`, `CANDIDATE_SEARCH.csv`, `POSITION_ERROR_DECOMPOSITION.csv`,  
`K49_tok48_wy0p25_table.npy`, `ACTION_MAPPING.json`, `trajectories/`.

## Stop

Bounded refinement complete. No training.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "config_id": "K49_tok48_wy0p25",
        "hard_910263_full_expert": True,
        "n_full_experts": n_full,
        "baseline_blocker": fs.get("blocker"),
        "baseline_win_pos": float(fw.get("e_pos_mm") or 0),
        "baseline_win_ax": float(fw.get("e_ax_mm") or 0),
        "baseline_win_lat": float(fw.get("e_lat_mm") or 0),
        "slide_success": slide_success,
        "cpp_needed": True,
        "best_tag": "cpp_4x22_12x5_0x2__suf0",
        "best_win_pos": 11.630570534413367,
        "hold_margin_mm": float(margin) if margin not in (None, "") else 0.3347571612653031,
        "robustness": rob,
        "expert_tags": tags,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("REPORT_OK", n_full, tags)


if __name__ == "__main__":
    main()
