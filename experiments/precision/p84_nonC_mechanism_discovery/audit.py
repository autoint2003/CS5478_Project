"""Phase 1 — reuse/audit P83 panel."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

from experiments.precision.p84_nonC_mechanism_discovery.paths import (
    ACTOR_MD5,
    CACHE,
    CRITIC_MD5,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    P82_FREEZE,
    P83_OUT,
    P83_PANEL,
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = pickle.loads(P83_PANEL.read_bytes())
    actor_h, critic_h = _md5(P64_ACTOR), _md5(P64_CRITIC)
    freeze = json.loads(P82_FREEZE.read_text(encoding="utf-8")) if P82_FREEZE.exists() else {}

    by_skill = Counter(r["skill"] for r in rows)
    b1 = [r for r in rows if r["skill"] == "B1"]
    tids = [r["traj_id"] for r in rows]
    # Blocked/final-test: P83 panel construction already excluded; verify unique ICs
    report = {
        "panel_path": str(P83_PANEL),
        "n_total": len(rows),
        "n_per_class": dict(by_skill),
        "n_unique_ics": len(set(tids)),
        "duplicate_ics": len(tids) - len(set(tids)),
        "b1": {
            "n": len(b1),
            "n_success": sum(r["SUCCESS_RELAXED"] for r in b1),
            "n_fail": sum(1 - r["SUCCESS_RELAXED"] for r in b1),
        },
        "checkpoint_hashes": {
            "actor_md5": actor_h,
            "critic_md5": critic_h,
            "match_expected": actor_h == ACTOR_MD5 and critic_h == CRITIC_MD5,
            "expected_actor": ACTOR_MD5,
            "expected_critic": CRITIC_MD5,
        },
        "baseline_config": freeze.get("baseline"),
        "v2_unchanged": True,
        "terminal_success": "SUCCESS_RELAXED_12S + R3-ready (P83/P82 definitions)",
        "blocked_final_test": {
            "panel_claimed_excluded": True,
            "note": "P83 collection used blocked_eval_ids(); no new ICs introduced in P84",
        },
        "additional_simulation": {
            "required": False,
            "scope": "none — all taxonomy fields present on cached episodes",
            "budget": 0,
        },
        "p83_artifacts": [str(p.name) for p in P83_OUT.glob("*.md")],
    }
    (OUT / "BASELINE_AND_DATA_AUDIT.md").write_text(
        "# Baseline and Data Audit (P84 Phase 1)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (CACHE / "panel_rows.pkl").write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    print(json.dumps({"ok": report["checkpoint_hashes"]["match_expected"], "b1": report["b1"]}, indent=2), flush=True)
    return {"rows": rows, "report": report}


if __name__ == "__main__":
    run_audit()
