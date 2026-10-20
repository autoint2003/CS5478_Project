import csv
from pathlib import Path

B = Path("results/expert_generation/k49_ball_impact_dynamic_recovery")
U = Path("results/expert_generation/k49_unified_dataset_expansion")
solved = {r["root_episode"] for r in csv.DictReader((U / "CANONICAL_MANIFEST.csv").open(encoding="utf-8"))}
unsolved = []
for r in csv.DictReader((B / "ROOT_EPISODE_COVERAGE.csv").open(encoding="utf-8")):
    if str(r.get("full")) != "1" and r["root_id"] not in solved:
        unsolved.append(r)
print("unsolved ball searched", len(unsolved))
for r in unsolved:
    p = B / "snapshots" / f"{r['root_id']}.pkl"
    print(r.get("root_id"), "snap", p.exists(), r.get("regimes"), "simple", r.get("simple_solves"))
nr = list(csv.DictReader((U / "NEW_PHYSICAL_ROOTS.csv").open(encoding="utf-8")))
acc = [r for r in nr if str(r.get("accepted")) == "1" and r["root_id"] not in solved]
print("accepted unsolved ubimp", len(acc))
for r in acc[:20]:
    print(r["root_id"], r.get("e_theta0"), r.get("e_pos0"), r.get("v0"))
