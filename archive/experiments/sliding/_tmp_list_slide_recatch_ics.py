import json
from pathlib import Path

rows = [
    json.loads(l)
    for l in open("results/diagnostics/hard_ic_mining/candidate_manifest.jsonl", encoding="utf-8")
    if l.strip()
]
bilat, loss, uni = [], [], []
for r in rows:
    st = set(r.get("states") or [])
    snap = Path("results/diagnostics/hard_ic_mining/snapshots") / f"{r['id']}.pkl"
    entry = {"id": r["id"], "states": list(st), "snapshot": str(snap), "split": r.get("assigned_split"), "exists": snap.exists()}
    if "complete_grasp_loss" in st:
        loss.append(entry)
    elif "asymmetric_or_unilateral" in st:
        uni.append(entry)
    if ("contact_migration" in st or "tangential_slip" in st or "recaught_displaced" in st) and "complete_grasp_loss" not in st:
        bilat.append(entry)
print("bilat-ish", len(bilat), [b["id"] for b in bilat[:20]])
print("loss", len(loss), [b["id"] for b in loss[:20]])
print("uni", len(uni), [b["id"] for b in uni[:20]])

vr = [
    json.loads(l)
    for l in open("results/benchmarks/hard_recovery_v1/verified_manifest.jsonl", encoding="utf-8")
    if l.strip()
]
print("hrv1 verified", len(vr), list(vr[0].keys()) if vr else None)
print([r.get("id") or r.get("ic_id") for r in vr[:25]])

lt = list(Path("results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots").glob("*.pkl"))
print("large tilt", [p.stem for p in lt])

# dump pools for main script
out = Path("results/expert_generation/k49_sliding_recatch_dataset/RAW")
out.mkdir(parents=True, exist_ok=True)
json.dump({"sliding_candidates": bilat, "complete_loss": loss, "unilateral": uni}, open(out / "ic_pools_raw.json", "w"), indent=2)
print("wrote ic_pools_raw.json")
