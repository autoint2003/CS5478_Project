"""Re-validate frozen Plan C golden experts (3 independent replays each)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.plan_c_golden_replay import PLAN_C_ROOTS, replay_plan_c


def main() -> None:
    ok_all = True
    summary = []
    for rid in PLAN_C_ROOTS:
        payload, _, _rows = replay_plan_c(rid, n_replay=3)
        n_ok = payload["exact_replay"]["n_pass"]
        ok = n_ok == 3
        ok_all = ok_all and ok
        summary.append({"root_id": rid, "pass": f"{n_ok}/3", "ok": ok})
        print(f"{rid}: {n_ok}/3 SUCCESS_NEW", flush=True)
    out = ROOT / "results" / "consolidation" / "GOLDEN_REPLAY_RERUN.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
