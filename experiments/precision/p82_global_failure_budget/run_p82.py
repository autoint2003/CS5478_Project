"""P82 orchestrator — diagnostic failure budget; no training; no P83 execution."""

from __future__ import annotations

import json
import traceback

from experiments.precision.p82_global_failure_budget.analyze import failure_budget, skill_transition_analysis
from experiments.precision.p82_global_failure_budget.baseline_freeze import freeze_baseline
from experiments.precision.p82_global_failure_budget.collect import collect_baseline
from experiments.precision.p82_global_failure_budget.decide import decide
from experiments.precision.p82_global_failure_budget.evidence import evidence_matrix, msp_opportunity, propose_p83
from experiments.precision.p82_global_failure_budget.paths import OUT


def main(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"phases": []}
    try:
        print("=== P82 Phase 1: baseline freeze ===", flush=True)
        freeze = freeze_baseline()
        log["phases"].append({"phase": "freeze", "actor_md5": freeze["baseline"]["actor"]["md5"]})

        print("=== P82 collect baseline A/B/C/D episodes ===", flush=True)
        col = collect_baseline(device=device)
        rows = col["rows"]
        log["phases"].append({"phase": "collect", "n": len(rows), "cached": col.get("cached")})

        print("=== P82 Phase 2: failure budget ===", flush=True)
        budget = failure_budget(rows)
        log["phases"].append({"phase": "budget", "msp": budget.get("msp_equal_class_weight")})

        print("=== P82 Phase 3: skill transitions ===", flush=True)
        skills = skill_transition_analysis(rows)
        log["phases"].append({"phase": "skills", "assoc": {k: v.get("association") for k, v in (skills.get("associations") or {}).items()}})

        print("=== P82 Phase 4: evidence matrix ===", flush=True)
        matrix = evidence_matrix()

        print("=== P82 Phase 5–6: MSP opportunity + P83 proposal ===", flush=True)
        opportunity = msp_opportunity(budget, skills)
        proposal = propose_p83(budget, opportunity, skills)
        log["phases"].append({"phase": "opportunity", "top": (opportunity.get("ranking") or [{}])[0]})

        print("=== P82 Phase 7: decide ===", flush=True)
        decision = decide(freeze, budget, skills, matrix, opportunity, proposal)
        log["decision"] = {"case": decision["case"], "p83": decision["p83"]}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "decision": decision}
    except Exception as e:  # noqa: BLE001
        log["error"] = {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        (OUT / "RUN_ERROR.md").write_text(f"# P82 Error\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
