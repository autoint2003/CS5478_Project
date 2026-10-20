import ast
import csv
from pathlib import Path

rows = list(csv.DictReader(open(
    "results/diagnostics/nontrivial_recovery_ic_mining/PREFIX_HANDOFF_AUDIT.csv",
    encoding="utf-8")))
succ = [r for r in rows if r["success"] == "1"]
print("n_succ", len(succ))
for r in succ:
    ic = r["ic_id"]
    toks = ast.literal_eval(r["tokens"])
    cands = [
        Path(f"results/benchmarks/hard_recovery_v1/snapshots/{ic}.pkl"),
        Path(f"results/diagnostics/hard_ic_mining/snapshots/{ic}.pkl"),
    ]
    if "leftover_" in ic:
        hard = ic.split("leftover_")[-1]
        cands.append(Path(f"results/diagnostics/hard_ic_mining/snapshots/{hard}.pkl"))
        cands.append(Path(f"results/benchmarks/hard_recovery_v1/snapshots/{hard}.pkl"))
    found = next((str(p) for p in cands if p.exists()), None)
    print(f"{ic}\t{toks}\tsnap={found}")
