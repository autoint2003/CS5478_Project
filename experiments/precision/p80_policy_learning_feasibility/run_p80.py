"""P80 orchestrator — diagnostic only, no training."""

from __future__ import annotations

import json
import traceback

from experiments.precision.p80_policy_learning_feasibility.audit import run_audit
from experiments.precision.p80_policy_learning_feasibility.coverage import run_coverage
from experiments.precision.p80_policy_learning_feasibility.decide import decide
from experiments.precision.p80_policy_learning_feasibility.feasibility import (
    write_learning_feasibility,
    write_retention_plan,
)
from experiments.precision.p80_policy_learning_feasibility.paths import OUT
from experiments.precision.p80_policy_learning_feasibility.selection import run_selection_analyses


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"phases": []}
    try:
        print("=== P80 Phase 1: P79 statistical audit ===", flush=True)
        audit = run_audit()
        log["phases"].append({"phase": "audit", "ok": audit.get("consistent_with_p79_headlines")})

        print("=== P80 Phases 2–4: static vs conditional / reversals / value ===", flush=True)
        selection = run_selection_analyses()
        log["phases"].append(
            {
                "phase": "selection",
                "fixed": selection.get("fixed_name"),
                "pref_mode": selection["reversals"].get("preference_mode"),
            }
        )

        print("=== P80 Phase 5: coverage ===", flush=True)
        coverage = run_coverage(selection)
        log["phases"].append({"phase": "coverage", "gaps": [g["gap"] for g in coverage.get("gaps") or []]})

        print("=== P80 Phase 6–7: learning feasibility + retention ===", flush=True)
        learn = write_learning_feasibility(selection, coverage)
        retention = write_retention_plan()
        log["phases"].append({"phase": "feasibility", "stance": learn.get("current_stance")})

        print("=== P80 Phase 8: decide ===", flush=True)
        decision = decide(audit, selection, coverage, learn, retention)
        log["decision"] = {"case": decision["case"], "p81": decision["p81"]}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return {"ok": True, "decision": decision}
    except Exception as e:  # noqa: BLE001
        log["error"] = {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        (OUT / "RUN_ERROR.md").write_text(f"# P80 Error\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
