"""P81 orchestrator — diagnostic only; no training; no overlay deploy."""

from __future__ import annotations

import json
import traceback

from experiments.precision.p81_static_contact_bias.candidate_bias import run_candidate_bias
from experiments.precision.p81_static_contact_bias.decide import decide
from experiments.precision.p81_static_contact_bias.distribution import run_distribution_audit
from experiments.precision.p81_static_contact_bias.full_episode import run_full_episode
from experiments.precision.p81_static_contact_bias.mechanism import write_cross_class, write_mechanism
from experiments.precision.p81_static_contact_bias.paths import OUT
from experiments.precision.p81_static_contact_bias.semantics import write_token56_semantics
from experiments.precision.p81_static_contact_bias.validate_cf import run_matched_validation, run_timing_analysis


def main(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    log: dict = {"phases": [], "budgets": {
        "N_DIST_ICS": 28,
        "N_CF_VALIDATE_ICS": 40,
        "N_CF_SEEDS": 2,
        "N_TIMING_SNAPS": 16,
        "N_FULL_EP_PER_CLASS": 8,
    }}
    try:
        print("=== P81 tok56 semantics ===", flush=True)
        sem = write_token56_semantics(device=device)
        log["phases"].append({"phase": "semantics", "family": (sem.get("codebook_decode") or {}).get("family")})

        print("=== P81 Phase 1: Actor distribution ===", flush=True)
        dist = run_distribution_audit(device=device)
        log["phases"].append({"phase": "distribution", "fc": dist.get("fc_decision")})

        print("=== P81 Phase 2: candidate-set bias ===", flush=True)
        bias = run_candidate_bias()
        log["phases"].append({"phase": "candidate_bias", "cats": bias.get("categories")})

        print("=== P81 Phase 3: matched validation ===", flush=True)
        val = run_matched_validation(device=device)
        log["phases"].append({"phase": "validate", "delta": val.get("paired_delta_succ_ic")})

        print("=== P81 Phase 4: timing ===", flush=True)
        timing = run_timing_analysis(device=device)
        log["phases"].append({"phase": "timing", "stages": list((timing.get("by_stage") or {}).keys())})

        print("=== P81 Phase 5: full-episode transfer (gated) ===", flush=True)
        # Run full-episode if mean delta > 0 even if CI includes 0 (diagnostic); skip if mean <= 0
        full = run_full_episode(device=device, skip_if_weak_cf=True, cf_report=val)
        # If skipped due to weak CF but we still want partial evidence when mean slightly positive:
        if full.get("skipped") and ((val.get("paired_delta_succ_ic") or {}).get("mean") or 0) > 0:
            print("=== P81 Phase 5 retry: mean Δ>0 — run bounded full-episode ===", flush=True)
            full = run_full_episode(device=device, skip_if_weak_cf=False, cf_report=val, force=False)
        log["phases"].append({"phase": "full_episode", "msp_delta": full.get("msp_delta"), "skipped": full.get("skipped")})

        print("=== P81 Phase 6–7: mechanism + cross-class ===", flush=True)
        mech = write_mechanism(sem, dist, bias, val, timing, full)
        cross = write_cross_class(full, val, timing)

        print("=== P81 Phase 8: decide ===", flush=True)
        decision = decide(sem, dist, bias, val, timing, full, mech, cross)
        log["decision"] = {"case": decision["case"], "p82": decision["p82"]}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "decision": decision}
    except Exception as e:  # noqa: BLE001
        log["error"] = {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()}
        (OUT / "RUN_LOG.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        (OUT / "RUN_ERROR.md").write_text(f"# P81 Error\n\n```\n{traceback.format_exc()}\n```\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
