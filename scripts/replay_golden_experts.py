#!/usr/bin/env python
"""Entry point: re-validate frozen Plan C golden experts (3x independent replay)."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_mod = runpy.run_path(str(ROOT / "scripts" / "consolidation" / "freeze_unified_experts.py"))
replay_plan_c = _mod["replay_plan_c"]
PLAN_C_ROOTS = _mod["PLAN_C_ROOTS"]


def main():
    ok_all = True
    summary = []
    for rid in PLAN_C_ROOTS:
        payload, _, rows = replay_plan_c(rid, n_replay=3)
        n_ok = payload["exact_replay"]["n_pass"]
        ok = n_ok == 3
        ok_all = ok_all and ok
        summary.append({"root_id": rid, "pass": f"{n_ok}/3", "ok": ok, "rows": rows})
        print(f"{rid}: {n_ok}/3 SUCCESS_NEW", flush=True)
    out = ROOT / "results" / "consolidation" / "GOLDEN_REPLAY_RERUN.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
