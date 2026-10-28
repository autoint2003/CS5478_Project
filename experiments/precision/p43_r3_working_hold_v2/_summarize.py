import csv
from collections import Counter
from pathlib import Path

rows = list(csv.DictReader(Path("results/diagnostics/p43_r3_working_hold_v2/EXPERT_ROLLOUTS.csv").open(encoding="utf-8")))
print("n", len(rows), dict(Counter((r["skill"], r["rule"]) for r in rows)))
for skill in ("A1", "B1", "C", "D"):
    for rule in ("R2", "R3"):
        sub = [r for r in rows if r["skill"] == skill and r["rule"] == rule]
        if not sub:
            continue
        new = sum(int(r["SUCCESS_NEW"]) for r in sub)
        hand = sum(int(r["handoff"]) for r in sub)
        never = sum(int(r["never_satisfied"]) for r in sub)
        print(f"{skill} {rule}: NEW {new}/{len(sub)} hand={hand} never={never} cats={dict(Counter(r['fail_category'] for r in sub))}")
