"""Summarize replay jsonl files written so far."""

import json
from collections import Counter
from pathlib import Path

def summarize(path):
    rows = []
    if not Path(path).is_file():
        print(path, "missing")
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(path, "rows", len(rows))
    print(" status", dict(Counter(row.get("status") for row in rows)))
    print(" label", dict(Counter(row.get("label") for row in rows)))
    played = [row for row in rows if row.get("status") in {"REPLAYED", "PRIOR_REPLAY"}]
    print(" played_success", sum(1 for row in played if row.get("primary_10s")), "/", len(played))
    fails = [row for row in rows if row.get("status") in {"REPLAYED", "PRIOR_REPLAY"} and not row.get("primary_10s")]
    print(" fails", [(row["id"], row["strategy"], row.get("label")) for row in fails][:40], "n", len(fails))

summarize("demo_results/diagnostics/raw/legal_recovery_v2/replay_k47_all.jsonl")
summarize("demo_results/diagnostics/raw/legal_recovery_v2/replay_k56.jsonl")
