"""P83 orchestrator — diagnostic only; no training; no P84 execution."""

from __future__ import annotations

import json
import traceback

from experiments.precision.p83_recoverable_failure_mass.analyze import (
    exclusive_attribution,
    failure_mass,
    msp_scenarios,
    reconstruct_funnel,
    recoverability_evidence,
    statistical_sensitivity,
)
from experiments.precision.p83_recoverable_failure_mass.decide import decide
from experiments.precision.p83_recoverable_failure_mass.panel import validate_and_expand
from experiments.precision.p83_recoverable_failure_mass.paths import N_STAGE1, N_STAGE2, OUT


def _pipeline(rows, panel_report, *, stage: int):
    print("=== P83 Phase 2: funnel ===", flush=True)
    funnel = reconstruct_funnel(rows)
    print("=== P83 Phase 3: exclusive attribution ===", flush=True)
    attr = exclusive_attribution(rows)
    print("=== P83 Phase 4: failure mass ===", flush=True)
    mass = failure_mass(rows, attr)
    print("=== P83 Phase 5: recoverability ===", flush=True)
    recov = recoverability_evidence(mass)
    print("=== P83 Phase 6: MSP scenarios ===", flush=True)
    scenarios = msp_scenarios(rows, mass, recov)
    print("=== P83 Phase 7: sensitivity ===", flush=True)
    sens = statistical_sensitivity(rows, mass, scenarios)
    print("=== P83 Phase 8–9: decide ===", flush=True)
    decision = decide(panel_report, funnel, attr, mass, recov, scenarios, sens, stage=stage)
    return {
        "funnel": funnel,
        "attr": attr,
        "mass": mass,
        "recov": recov,
        "scenarios": scenarios,
        "sens": sens,
        "decision": decision,
    }


def main(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"phases": []}
    try:
        print("=== P83 Stage 1 panel (≥32 ICs/class) ===", flush=True)
        stage1 = validate_and_expand(device=device, target=N_STAGE1)
        log["phases"].append({"phase": "panel_stage1", "report": stage1["report"]})
        out1 = _pipeline(stage1["rows"], stage1["report"], stage=1)
        decision = out1["decision"]

        if decision.get("need_stage2_to_48"):
            print("=== P83 Stage 2 panel (up to 48 ICs/class) — decision unresolved ===", flush=True)
            stage2 = validate_and_expand(device=device, target=N_STAGE2)
            log["phases"].append({"phase": "panel_stage2", "report": stage2["report"]})
            out2 = _pipeline(stage2["rows"], stage2["report"], stage=2)
            decision = out2["decision"]
            # If still D after stage2, force stop recommendation
            if decision["case"] == "D":
                decision["p84"] = "stop_consolidate"
                decision["reason"] = (
                    decision["reason"]
                    + " After stage-2 (48/class cap), still indistinguishable — stop broad diagnostics; "
                    "no credible low-cost causal test beyond consolidating the baseline."
                )
                decision["proposal"]["smallest_useful_experiment"] = "None — consolidate P64+V2"
                (OUT / "P84_DECISION.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        else:
            log["phases"].append({"phase": "panel_stage2", "skipped": True})

        log["decision"] = {"case": decision["case"], "p84": decision["p84"]}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "decision": decision}
    except Exception as e:  # noqa: BLE001
        log["error"] = {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        (OUT / "RUN_ERROR.md").write_text(f"# P83 Error\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
