"""P79 orchestrator — diagnostic matched CF coverage expansion."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from experiments.precision.p79_matched_cf_coverage.analyze import run_analysis
from experiments.precision.p79_matched_cf_coverage.audit import run_restore_audit
from experiments.precision.p79_matched_cf_coverage.cf_run import analyze_outcomes, run_matched_cf, write_action_protocol
from experiments.precision.p79_matched_cf_coverage.decide import decide
from experiments.precision.p79_matched_cf_coverage.mine import mine_snapshots
from experiments.precision.p79_matched_cf_coverage.paths import CACHE, N_EVAL_IC_MIN, N_PILOT_IC, OUT


def main(*, device: str = "cuda", skip_audit: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log = {"phases": []}

    def mark(name, blob):
        log["phases"].append({"phase": name, "ok": True, "summary": blob})
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    try:
        print("=== P79 Phase 1: restore audit ===", flush=True)
        if skip_audit and (OUT / "SNAPSHOT_RESTORE_AUDIT.json").exists():
            audit = json.loads((OUT / "SNAPSHOT_RESTORE_AUDIT.json").read_text(encoding="utf-8"))
        else:
            audit = run_restore_audit(device=device)
        mark("audit", {"ok": audit.get("ok"), "checks": audit.get("checks")})

        print("=== P79 Phase 2: mine FC snapshots ===", flush=True)
        bank = mine_snapshots(device=device)
        mark(
            "mine",
            {
                "n_ics": bank.get("n_independent_ics"),
                "n_snaps": bank.get("n_unique_snapshots"),
                "strata": bank.get("strata_counts"),
            },
        )

        write_action_protocol()

        print("=== P79 Phase 3–5: pilot CF ===", flush=True)
        pilot = run_matched_cf(device=device, pilot=True)
        rows = pilot["rows"]
        outcome_pilot = analyze_outcomes(rows)
        mark(
            "pilot_cf",
            {
                "n_ics": outcome_pilot.get("n_independent_ics"),
                "n_disagree": outcome_pilot.get("n_nontrivial_outcome_disagreements"),
                "determinism": outcome_pilot.get("determinism"),
            },
        )

        det_frac = (outcome_pilot.get("determinism") or {}).get("deterministic_fraction")
        expand = True
        reasons = []
        if not audit.get("ok"):
            expand = False
            reasons.append("audit_failed")
        if (outcome_pilot.get("n_independent_ics") or 0) < 5:
            expand = False
            reasons.append("too_few_pilot_ics")
        # Always expand toward coverage target when pilot produced valid rows
        if len(rows) < 10:
            expand = False
            reasons.append("too_few_pilot_rows")

        # If deterministic, expand with 1 seed (already N_SEEDS_EXPAND=1)
        if det_frac is not None and det_frac >= 0.95:
            reasons.append("deterministic_use_single_seed_on_expand")

        if expand:
            print("=== P79 expand CF to full IC coverage ===", flush=True)
            # Skip re-running pilot ICs wastefully: expand includes all primary
            exp = run_matched_cf(device=device, pilot=False)
            rows = exp["rows"]
            outcome = analyze_outcomes(rows)
            mark(
                "expand_cf",
                {
                    "n_ics": outcome.get("n_independent_ics"),
                    "n_disagree": outcome.get("n_nontrivial_outcome_disagreements"),
                    "expand_reasons": reasons,
                },
            )
        else:
            print(f"=== P79 skip expand ({reasons}) — analyze pilot ===", flush=True)
            outcome = outcome_pilot
            mark("expand_cf", {"skipped": True, "reasons": reasons})

        print("=== P79 Phases 6–9: ranking / rep / alias / validity ===", flush=True)
        analysis = run_analysis(rows, outcome)
        mark(
            "analysis",
            {
                "rank": (analysis.get("rep") or {}).get("summary"),
                "alias": {
                    "n": (analysis.get("alias") or {}).get("n_alias_pairs"),
                    "established": (analysis.get("alias") or {}).get("action_relevant_aliasing_established"),
                },
            },
        )

        print("=== P79 Phase 10: decide ===", flush=True)
        decision = decide(outcome, analysis, audit)
        mark("decide", {"case": decision.get("case"), "p80": decision.get("p80")})

        # Coverage note vs planning targets
        n_eval = len(set(bank.get("eval_ics") or []) & set(bank.get("by_ic") or {}))
        (OUT / "COVERAGE_VS_TARGETS.md").write_text(
            "# Coverage vs Planning Targets\n\n"
            f"- Independent ICs mined: {bank.get('n_independent_ics')} "
            f"(target ~80–120; available C-skill pool limited)\n"
            f"- Eval ICs in bank: {n_eval} (min planning {N_EVAL_IC_MIN})\n"
            f"- Pilot ICs: {N_PILOT_IC}\n"
            f"- CF ICs analyzed: {outcome.get('n_independent_ics')}\n"
            f"- Nontrivial disagreements: {outcome.get('n_nontrivial_outcome_disagreements')}\n",
            encoding="utf-8",
        )
        return {"ok": True, "decision": decision, "log": log}
    except Exception as e:  # noqa: BLE001
        log["error"] = {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        (OUT / "RUN_ERROR.md").write_text(f"# P79 Error\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
