import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, ".")
from scripts.generate_k49_lost_grasp_recovery import OUT, classify_ic, root_episode_id
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
    write_json,
)

rows = list(csv.DictReader((OUT / "TYPE3_FULL_EXPERTS.csv").open(encoding="utf-8")))
print("n_full_rows", len(rows))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book = load_codebook()
table = build_k49_table(book, config_id=CONFIG_ID)
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)

verified = []
for r in rows:
    toks = json.loads(r["tokens"])
    blob = pickle.loads((OUT / "snapshots" / f"{r['ic_id']}.pkl").read_bytes())
    info = classify_ic(env, table, book, blob)
    res = evaluate_full(env, table, book, blob, toks)
    ok = bool(info["is_type3_onset"] and res["SUCCESS_NEW"])
    print(
        r["ic_id"], "onset", info["lost_class"], "type3", info["is_type3_onset"],
        "NEW", res["SUCCESS_NEW"], "hold", res["window"].get("longest_both_ok_s"),
        "th", res["window"].get("mean_theta"), "pos", res["window"].get("mean_pos"),
        "simple_new_flag", r.get("newly_vs_simple"),
    )
    if ok:
        verified.append({**r, "repro_ok": 1, "onset_recheck": info["lost_class"]})

ics = list(csv.DictReader((OUT / "LOST_GRASP_ICS.csv").open(encoding="utf-8")))
roots_all = sorted({
    r.get("root_episode") or root_episode_id(r["ic_id"])
    for r in ics if str(r.get("is_type3_onset")) in ("1", "True")
})
full_roots = sorted({r.get("root_episode") or root_episode_id(r["ic_id"]) for r in verified})
# baselines: how many full also solvable by simple?
base = {}
bp = OUT / "RAW" / "pass2_baselines.csv"
if bp.exists():
    for b in csv.DictReader(bp.open(encoding="utf-8")):
        base[b["ic_id"]] = b

newly = []
for r in verified:
    b = base.get(r["ic_id"])
    if b and b.get("simple") == "1":
        continue
    # also check A/B/C now
    blob = pickle.loads((OUT / "snapshots" / f"{r['ic_id']}.pkl").read_bytes())
    a = evaluate_full(env, table, book, blob, [], record_obs=False)
    bres = evaluate_full(env, table, book, blob, [0] * 16, record_obs=False)
    c = evaluate_full(env, table, book, blob, [30] + [0] * 8, record_obs=False)
    simple = a["SUCCESS_NEW"] or bres["SUCCESS_NEW"] or c["SUCCESS_NEW"]
    print(" baseline", r["ic_id"], "A", a["SUCCESS_NEW"], "B", bres["SUCCESS_NEW"], "C", c["SUCCESS_NEW"])
    if not simple:
        newly.append(r["ic_id"])

answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
answers.update({
    "Q1_n_independent_lost_grasp_roots": len(roots_all),
    "Q1_root_ids": roots_all,
    "Q4_n_type3_full_ics": len({r["ic_id"] for r in verified}),
    "Q4_n_type3_full_roots": len(full_roots),
    "Q4_full_ids": sorted({r["ic_id"] for r in verified}),
    "Q4_full_roots": full_roots,
    "Q4_sublabels": dict(Counter(r["sublabel"] for r in verified)),
    "Q5_n_newly_solved_vs_simple": len(set(newly)),
    "Q5_newly_ids": sorted(set(newly)),
    "Q7_counts": {
        "type1_ics_approx": answers.get("Q7_counts", {}).get("type1_ics_approx", 8),
        "type2_ics": answers.get("Q7_counts", {}).get("type2_ics", 5),
        "type3_lost_grasp_ics": len({r["ic_id"] for r in verified}),
        "type3_lost_grasp_roots": len(full_roots),
    },
})
# readiness: want >=3 independent ROOTS with type3
n1 = answers["Q7_counts"]["type1_ics_approx"]
n2 = answers["Q7_counts"]["type2_ics"]
n3r = len(full_roots)
answers["Q7_ready_unified_SFT"] = bool(n1 >= 3 and n2 >= 2 and n3r >= 3)
answers["Q7_note"] = (
    f"Type1≈{n1}, Type2={n2}, Type3 lost-grasp FULL ICs={len(verified)} "
    f"across {n3r} independent roots. "
    + ("Enough mechanism diversity to begin unified SFT with episode-grouped splits."
       if answers["Q7_ready_unified_SFT"] else "Need more Type3 root coverage.")
    + " Note: several Type3 experts are legal-unload-derived loss onsets from hard poses."
)
# Q2/Q3 from files
prefs = list(csv.DictReader((OUT / "RECATCH_PREFIXES.csv").open(encoding="utf-8"))) if (OUT / "RECATCH_PREFIXES.csv").exists() else []
answers["Q2_n_ics_bilateral_restored"] = len({p["ic_id"] for p in prefs} | {r["ic_id"] for r in verified})
need = [p for p in prefs if float(p.get("theta_at_bil") or 0) > 10 or float(p.get("pos_at_bil") or 0) > 12]
answers["Q3_n_stable_recatch_need_slide_pivot"] = len({p["ic_id"] for p in need})
# among verified, how many needed post manip sublabel
need_post = [r for r in verified if r["sublabel"] != "TYPE3_RECATCH_ONLY"]
answers["Q3_n_full_with_slide_or_pivot"] = len(need_post)
answers["Q3_note"] = (
    f"Pass1 prefixes needing pose fix at bil: {answers['Q3_n_stable_recatch_need_slide_pivot']}. "
    f"Among FULL experts, non-RECATCH_ONLY sublabels: {len(need_post)}."
)
write_json(OUT / "RAW" / "answers.json", answers)
write_json(OUT / "RAW" / "verified_full.json", verified)

# update REPORT / readiness briefly
checksum = "33d4820d1ec09c8b"
report = f"""# Type 3 Lost-Grasp Recovery Experts

**Config:** `K49_tok48_wy0p25` checksum `{checksum}`  
**Type 3 = lost/unreliable grasp at Recovery onset** (eccentric impact / min Δθ not required).  
**No SFT/GRPO.**

## Yield (replay-verified)

| Metric | Value |
|--------|------:|
| Independent lost-grasp roots (mined/selected) | {len(roots_all)} |
| TYPE3_LOST_GRASP_FULL_EXPERT ICs | **{len(verified)}** |
| Independent roots among FULL experts | **{n3r}** |
| Newly solved vs simple A/B/C | **{len(set(newly))}** |
| Sublabels | {answers['Q4_sublabels']} |

FULL ICs: {answers['Q4_full_ids']}  
FULL roots: {full_roots}

### `hrv1_920031` natural onset
Classifies as **STABLE_GRASP** — the prior `[30,12,12,8,0×7]` success is **not** Type3 by onset.
Derived `hrv1_920031_loss*` snapshots can be Type3 if onset is lost-grasp (none became FULL in this batch).

## Q1–Q7

1. **Independent lost-grasp roots:** **{len(roots_all)}**
2. **ICs restored to bilateral (prefixes ∪ FULL):** **{answers['Q2_n_ics_bilateral_restored']}**
3. **Need Sliding/Pivoting:** prefixes with θ>10 or pos>12 at bil: **{answers['Q3_n_stable_recatch_need_slide_pivot']}**; FULL with slide/pivot sublabel: **{len(need_post)}**
4. **Complete SUCCESS_NEW Type3:** **{len(verified)}** ICs / **{n3r}** roots
5. **Newly vs simple baselines:** **{len(set(newly))}** {sorted(set(newly))}
6. **Remaining failures:** primarily DROP / DROP_AFTER_HANDOFF / SUCCESS_12S without hold (`RAW/pass2_failures.csv`, `FAILURES.csv`)
7. **Ready unified SFT?** **{answers['Q7_ready_unified_SFT']}** — {answers['Q7_note']}

## Artifacts

`LOST_GRASP_ICS.csv`, `BASELINE_COMPARISON.csv`, `RECATCH_PREFIXES.csv`,  
`POST_RECATCH_CORRECTIONS.csv`, `TYPE3_FULL_EXPERTS.csv`, `FAILURES.csv`,  
`ROOT_EPISODE_COVERAGE.csv`, `UNIFIED_MANIFEST.csv`, `DATASET_READINESS.md`,  
`scripts/generate_k49_lost_grasp_recovery.py`
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")
ready = f"""# Dataset Readiness — Lost-Grasp Type3

## Verdict: {"YES — begin unified SFT with care" if answers['Q7_ready_unified_SFT'] else "NOT YET"}

- Type1 ≈ {n1}, Type2 = {n2}, Type3 FULL ICs = {len(verified)} ({n3r} roots)
- Several Type3 experts are **legal-unload-derived** loss onsets (same parent root episode for train/val/test grouping)

### Merge rule
Merge verified `TYPE3_FULL_EXPERTS.csv` rows. Split by `root_episode` (derived ICs inherit parent).
Keep `lost_class` as metadata only — not in 39D observations.
"""
(OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
print("FINAL", answers["Q4_n_type3_full_ics"], answers["Q4_n_type3_full_roots"],
      answers["Q5_n_newly_solved_vs_simple"], answers["Q7_ready_unified_SFT"])
