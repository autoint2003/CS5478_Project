# Figure Source Index

Figures live in `results/experiments/final_consolidation/figures/`. Regenerator: `experiments/precision/final_consolidation/generate_figures.py`.

| Figure | File | Claim shown | Source artifact(s) |
|---|---|---|---|
| 1 | `fig01_final_system_architecture.{png,pdf}` | Frozen inference path P64+V2+M111 | `BASELINE_FREEZE.json`; `FINAL_SYSTEM_MANIFEST.md` |
| 2 | `fig02_m111_action_representation.{png,pdf}` | Discrete skill-aware M111 vocabulary | Live Actor load `book_n=111`; V2 `approach_pool` |
| 3 | `fig03_train_eval_pipeline.{png,pdf}` | Train→freeze→P83 panel pipeline | P64 / P70 / P82 / P83 paths |
| 4 | `fig04_p83_per_class_success.{png,pdf}` | A1 8/32, B1 6/32, C 4/32, D 17/32, MSP 0.2734375 | `BASELINE_PANEL_VALIDATION.md` |
| 5 | `fig05_c_contact_formation_funnel.{png,pdf}` | C nested funnel counts 32→29→17→15→8→6→4→4 | `CROSS_CLASS_FUNNEL.md` class C |
| 6 | `fig06_cross_class_failure_attribution.{png,pdf}` | Exclusive earliest-decisive counts by class | `EXCLUSIVE_FAILURE_ATTRIBUTION.md` |
| 7 | `fig07_matched_vs_full_episode.{png,pdf}` | P73 ΔSUCCESS −0.09375; P81 matched +0.0875; P81 full MSP Δ −0.25 | `P74_DECISION.json`; P81 `FINAL_REPORT.json` / `FULL_EPISODE_TRANSFER.md` |

## Intentionally omitted from report figures

Per-experiment heatmaps for P75–P80, P69 snapshot Δe tables, and every intermediate funnel — available in source experiment folders but omitted to avoid overcrowding. Cite those MD/JSON files directly if needed.
