# Post-Audit Decision Revision (2026-11-07)

**Mark:** post-audit decision revision. No retrain · no large panel recollect · no weight changes.

## What changed

| Item | Before (archived) | After (formal) |
|---|---|---|
| P83 case | **C** — non-C mechanism discovery | **D** — opportunities indistinguishable; stop and consolidate |
| Next action | `nonC_mechanism_discovery_diagnostic` → executed as P84 | `stop_consolidate_baseline` (no new probe/train/overlay) |
| Dominance comparator | nonC **sum** any-failure **0.5078125** vs C_cf **0.1640625** | nonC **mean** any-failure **0.16927** vs C_cf **0.16406** |
| Stage-2 | skipped (Case C) | not authorized |

Both mass forms are retained on the revised decision. They are **not** homogeneous recoverability metrics.

## Unchanged

- P83 `cache/panel_32.pkl` outcomes and MSP **0.2734375**
- P64 Actor/Critic weights and hashes
- P84 diagnostic files (taxonomy, concentration, signatures) — **not rewritten**
- P84 Case D / P85 `stop_consolidate_baseline` empirical stop remains valid

## V2 / MSP

P83 V2 activation **2/128**. Do not attribute MSP 0.2734375 to V2 closed-loop performance.

## Related marks

- **P69:** old soft PPO gate = false proceed (ΔSUCCESS=0.015625).
- **P55:** BC obs repaired 36/36; no retrain → BC performance claims not updated.

## Pointers

| Role | Path |
|---|---|
| Formal P83 decision | `../p83_recoverable_failure_mass/P83_DECISION.json` |
| Case C archive | `../p83_recoverable_failure_mass/P83_DECISION_ORIGINAL_CASE_C.json` |
| Diff inputs | `SHORT_RERUN_DIFF.json`, `../p83_recoverable_failure_mass/P84_DECISION_RERUN.json` |
| P84 provenance note | `../p84_nonC_mechanism_discovery/POST_AUDIT_NOTE.md` |
