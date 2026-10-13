"""Mechanism and coverage counts that do not require the replay job."""

import json
from collections import Counter
from pathlib import Path

rows = json.loads(Path("demo_results/diagnostics/raw/legal_recovery_v2/coverage_all_holds.json").read_text(encoding="utf-8"))["records"]
tokens = {}
for line in Path("demo_results/diagnostics/raw/legal_recovery_v2/candidate_tokens.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        item = json.loads(line)
        tokens[f"{item['id']}|{item['strategy']}"] = item
search = [json.loads(line) for line in Path("demo_results/diagnostics/raw/legal_recovery_v2/search.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
ics = sorted({row["id"] for row in search})
hold_ics = sorted({row["id"] for row in rows})
print("searched_ics", len(ics), "hold_ics", len(hold_ics), "holds", len(rows))
print("no_hold", len(set(ics) - set(hold_ics)))
fam = Counter(row["family"] for row in rows)
print("family", dict(fam))
multi = 0
for row in rows:
    item = tokens.get(f"{row['id']}|{row['strategy']}", {})
    if int(item.get("none_intervals") or 0) >= 2:
        multi += 1
print("multi", multi)
print("k36", sum(row["k36_inside"] for row in rows), "k47", sum(row["k47_inside"] for row in rows))
by = Counter()
for row in rows:
    if row["k36_inside"]:
        by["k36"] += 1
    elif row["k47_inside"]:
        by["k47_only"] += 1
    else:
        by["neither"] += 1
print(dict(by))
ics_k36 = {row["id"] for row in rows if row["k36_inside"]}
ics_k47 = {row["id"] for row in rows if row["k47_inside"]}
print("ics_with_k36", len(ics_k36), "ics_with_k47", len(ics_k47))
print("neither_family", Counter(row["family"] for row in rows if not row["k47_inside"]))
