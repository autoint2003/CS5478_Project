"""P84 orchestrator — B1 mechanism discovery; diagnostic only."""

from __future__ import annotations

import json
import traceback

from experiments.precision.p84_nonC_mechanism_discovery.audit import run_audit
from experiments.precision.p84_nonC_mechanism_discovery.decide import decide
from experiments.precision.p84_nonC_mechanism_discovery.feasibility import intervention_feasibility, msp_opportunity
from experiments.precision.p84_nonC_mechanism_discovery.paths import OUT
from experiments.precision.p84_nonC_mechanism_discovery.signature import analyze_signatures
from experiments.precision.p84_nonC_mechanism_discovery.taxonomy import build_taxonomy, concentration_gate


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"phases": []}
    try:
        print("=== P84 Phase 1: audit ===", flush=True)
        audited = run_audit()
        rows = audited["rows"]
        log["phases"].append({"phase": "audit", "b1": audited["report"]["b1"]})

        print("=== P84 Phase 2: B1 taxonomy ===", flush=True)
        tax = build_taxonomy(rows)
        log["phases"].append({"phase": "taxonomy", "exclusive": tax["exclusive_counts"]})

        print("=== P84 Phase 3: concentration gate ===", flush=True)
        gate = concentration_gate(tax)
        log["phases"].append({"phase": "gate", "passed": gate["gate_passed"], "top": gate.get("largest_mode")})

        print("=== P84 Phases 4–6: signatures (conditional) ===", flush=True)
        sig = analyze_signatures(rows, gate)
        log["phases"].append({"phase": "signature", "skipped": bool(sig.get("skipped"))})

        print("=== P84 Phases 7–8: feasibility + MSP ===", flush=True)
        feas = intervention_feasibility(gate, sig)
        msp = msp_opportunity(gate, tax)

        print("=== P84 Phase 9: decide ===", flush=True)
        decision = decide(audited, tax, gate, sig, feas, msp)
        log["decision"] = {"case": decision["case"], "p85": decision["p85"]}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "decision": decision}
    except Exception as e:  # noqa: BLE001
        log["error"] = {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        (OUT / "RUN_ERROR.md").write_text(f"# P84 Error\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
