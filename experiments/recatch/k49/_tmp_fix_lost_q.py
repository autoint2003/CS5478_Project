import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.generate_k49_sliding_recatch_dataset import write_csv

OUT = Path("results/expert_generation/k49_lost_grasp_recovery")
ans = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
base = {b["ic_id"]: b for b in csv.DictReader((OUT / "RAW" / "pass2_baselines.csv").open(encoding="utf-8"))}
full = list(csv.DictReader((OUT / "TYPE3_FULL_EXPERTS.csv").open(encoding="utf-8")))
for r in full:
    b = base.get(r["ic_id"], {})
    if b.get("simple") == "1":
        r["newly_vs_simple"] = 0
write_csv(OUT / "TYPE3_FULL_EXPERTS.csv", full)
write_csv(OUT / "UNIFIED_MANIFEST.csv", full)
newly = [r["ic_id"] for r in full if int(float(r.get("newly_vs_simple") or 0)) == 1]
ans["Q5_n_newly_solved_vs_simple"] = len(set(newly))
ans["Q5_newly_ids"] = sorted(set(newly))
n1 = ans["Q7_counts"]["type1_ics_approx"]
n2 = ans["Q7_counts"]["type2_ics"]
n3 = ans["Q4_n_type3_full_ics"]
n3r = ans["Q4_n_type3_full_roots"]
ans["Q7_note"] = (
    f"Type1~{n1}, Type2={n2}, Type3 lost-grasp FULL={n3} ICs / {n3r} roots. "
    f"Only {len(set(newly))} newly harder than simple A/B/C; "
    "several FULL are legal-unload complete-loss onsets where secure/recatch is easy. "
    "Conditional: enough Type3 onset coverage to begin unified SFT; expand nontrivial lost-grasp recoveries."
)
ans["Q7_ready_unified_SFT"] = True
(OUT / "RAW" / "answers.json").write_text(json.dumps(ans, indent=2), encoding="utf-8")
# patch REPORT Q5/Q7 lines if present
rep = (OUT / "REPORT.md").read_text(encoding="utf-8")
# rewrite readiness
ready = f"""# Dataset Readiness — Lost-Grasp Type3

## Verdict: CONDITIONAL YES

- Type1 ≈ {n1}, Type2 = {n2}, Type3 FULL = {n3} ICs ({n3r} roots)
- Newly vs simple A/B/C: {len(set(newly))} (`{sorted(set(newly))}`)
- Several experts are legal-unload-derived complete-loss onsets (group by parent `root_episode`)

### Merge rule
Merge verified `TYPE3_FULL_EXPERTS.csv`. Split by `root_episode`.
Do not put `lost_class` into 39D observations.
"""
(OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
print("newly", newly)
print(ans["Q4_full_ids"], ans["Q4_full_roots"])
