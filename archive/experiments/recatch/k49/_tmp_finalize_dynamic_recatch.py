import csv
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.generate_k49_dynamic_recatch_experts import OUT
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

ecc = list(csv.DictReader((OUT / "ECCENTRIC_CATCH_MECHANISM.csv").open(encoding="utf-8")))


def truthy(x):
    return x in ("True", "1", True, 1)


hits = [
    e for e in ecc
    if truthy(e.get("eccentric")) and truthy(e.get("beneficial_rotation")) and truthy(e.get("downward_motion"))
]
print("hits", len(hits))
for e in hits:
    print(
        e["ic_id"], e["tag"],
        "ax", e.get("axial_max_mm"),
        "dth", e.get("delta_theta_deg"),
        "vdown", e.get("v_down_hand_z"),
        "dz", e.get("obj_dz_m"),
        "why", e.get("reason"),
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book48 = load_codebook()
table = build_k49_table(book48, config_id=CONFIG_ID)
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)

rows = list(csv.DictReader((OUT / "TYPE2_EXPERTS.csv").open(encoding="utf-8")))
repro = []
for r in rows:
    toks = json.loads(r["tokens"])
    blob = pickle.loads((OUT / "snapshots" / f"{r['ic_id']}.pkl").read_bytes())
    res = evaluate_full(env, table, book48, blob, toks)
    repro.append({
        "ic_id": r["ic_id"],
        "ok": bool(res["SUCCESS_NEW"]),
        "win_pos": res["window"].get("mean_pos"),
        "win_th": res["window"].get("mean_theta"),
    })
    print("REPRO", r["ic_id"], res["SUCCESS_NEW"])

write_json(OUT / "RAW" / "reproducibility_type2.json", repro)

answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
answers["Q4_type3_near_miss_mechanism"] = [
    {
        "ic_id": e["ic_id"],
        "tag": e["tag"],
        "axial_max_mm": e.get("axial_max_mm"),
        "delta_theta_deg": e.get("delta_theta_deg"),
        "v_down_hand_z": e.get("v_down_hand_z"),
        "obj_dz_m": e.get("obj_dz_m"),
        "reason_not_full_expert": e.get("reason"),
    }
    for e in hits
]
answers["n_type2_repro"] = sum(1 for x in repro if x["ok"])
answers["Q7_note"] = (
    "Type1~6 prior, Type2 ICs=5 (all reproducible), Type3 FULL=0 "
    f"({len(hits)} eccentric+beneficial+downward near-misses without SUCCESS_NEW). "
    "Not ready for Type3-inclusive unified SFT."
)
write_json(OUT / "RAW" / "answers.json", answers)

# refresh REPORT answers section clarity
report = (OUT / "REPORT.md").read_text(encoding="utf-8")
# rewrite readiness
ready = f"""# Dataset Readiness — Dynamic Recatch Expansion

## Verdict: NOT YET for Type3-inclusive unified SFT

- Type2 FULL experts (independent ICs): **{len(rows)}** (repro {answers['n_type2_repro']}/{len(rows)})
- Type3 FULL experts: **0**
- Type3 physical near-misses (eccentric + downward + beneficial rotation): **{len(hits)}**
- Unsolved hard ICs: {answers.get('Q5_unsolved_hard_ics')}

### Merge rule
- Safe to merge **TYPE2** rows from `UNIFIED_MANIFEST.csv` / `TYPE2_EXPERTS.csv` into the unified pool.
- Do **not** claim Type3 coverage until at least one VERIFIED_FULL_EXPERT has measured eccentric end-catch stats and no post-regrasp Slide/Pivot.

### Type3 blocker
Candidates can show eccentric contact and some orientation change during recapture, but none simultaneously satisfy SUCCESS_NEW without a separate post-regrasp correction phase (or they fail hold / drop).
"""
(OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")

# Update REPORT Q4 with near-miss table
q4_lines = ["\n### Q4 Type3 near-miss mechanism (not FULL experts)\n",
            "| IC | tag | axial_max_mm | delta_theta_deg | v_down | obj_dz_m | why_not_full |\n",
            "|----|-----|-------------:|----------------:|-------:|---------:|-------------|\n"]
for e in hits:
    q4_lines.append(
        f"| {e['ic_id']} | {e['tag']} | {e.get('axial_max_mm')} | {e.get('delta_theta_deg')} | "
        f"{e.get('v_down_hand_z')} | {e.get('obj_dz_m')} | {e.get('reason')} |\n"
    )
# append if not already
if "Q4 Type3 near-miss" not in report:
    report = report.replace(
        "## Artifacts",
        "".join(q4_lines) + "\n## Artifacts",
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

print("FINALIZE_OK", answers["Q2_type3_full_expert_exists"], answers["Q3_n_type2_ics"])
