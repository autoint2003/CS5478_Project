# Reproducibility Checklist

Audit performed during final consolidation. **No training or new evaluation campaign was launched.**

| Check | Status | Evidence |
|---|---|---|
| Final Actor checkpoint exists | Pass | `P64_step4_actor.pt` (370227 bytes) |
| Final Critic checkpoint exists | Pass | `P64_step4_critic.pt` (332359 bytes) |
| MD5 matches P82 freeze | Pass | Actor `9f4c5041…b6e2`; Critic `95de94e5…81cb` |
| Checkpoints load on CPU | Pass | `load_actor` / `load_critic` → book=111, params OK |
| Inference config unambiguous | Pass | Manifest: V2 + `rule=None` + StabilizeGrpoEnv + M111 from ckpt |
| Canonical eval script available | Pass | `experiments/precision/p82_global_failure_budget/collect.py` (`rollout_episode`) |
| P83 panel construction script available | Pass | `experiments/precision/p83_recoverable_failure_mass/panel.py` |
| Seeds documented | Pass | P83 `SEED=5483`, `N_SEEDS=1`, `rollout_seed(...)` formula |
| IC split / blocked final-test documented | Pass | `blocked_eval_ids()`; panel `final_test_excluded: true` |
| Panel episode cache present | Pass | `p83_recoverable_failure_mass/cache/panel_32.pkl` (130425 bytes) |
| Metrics traceable to saved results | Pass | Tables cite MD/JSON under `results/experiments/pXX_*` |
| Figures cite sources | Pass | `FIGURE_SOURCE_INDEX.md` |
| Obsolete overlays off inference path | Pass | Documented; frozen path forces `rule=None` |
| Final-test not auto-evaluated | Pass | Constraint honored in consolidation |
| P85 not launched | Pass | P84 decision `stop_consolidate_baseline` |

## Workflow audit

See `WORKFLOW_AUDIT.md` (2026-11-06 read-only review). No CRITICAL holes. **Must disclose:** P83 V2 inactive (2/128 forced); P83 MSP ≈ Actor-only; train-IC overlap; do not paste P70 V2 gains onto P83.

## Unresolved inconsistencies (reported, not silently repaired)

1. **P70 vs P73 Actor-only / V2 SUCCESS rates differ** on their respective n=96 held panels (e.g. V0 SUCCESS 0.083 vs 0.135). Treat as separate campaigns; do not average or difference across them.
2. **P81 full-episode MSP baseline (0.34375, n=32)** is not the P83 MSP (0.2734375, n=128). Different panels.
3. **Filename expectation:** some notes refer to `panel_32.pkl` at P83 root; on disk the panel is at `cache/panel_32.pkl` (P84 `paths.py` is correct).
4. **P73 wrapper naming:** frozen inference imports `select_token` from the P73 controller module even when overlays are disabled (`rule=None`). Behavior is V2-only, but the import path can confuse readers — document explicitly (see Manifest).
5. **P70 funnel n=64 vs closed-loop n=96:** both valid within P70; must not be merged into one table without labeling panel size.

## How to reproduce the P83 panel metrics (documentation only)

1. Confirm checkpoint MD5s.  
2. Run the existing P83 analyze path against `cache/panel_32.pkl` (or re-run `panel.py` with documented seeds — **not required for this consolidation**).  
3. Compare SUCCESS counts to `BASELINE_PANEL_VALIDATION.md`.

Do not treat a fresh re-collection with different seeds as the same panel.
