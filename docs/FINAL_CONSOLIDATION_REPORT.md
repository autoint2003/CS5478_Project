# Final Consolidation Report

Canonical copy of `results/consolidation/FINAL_REPORT.md` (that path is gitignored). See also `ARCHIVE_MANIFEST.md`, `ARCHITECTURE.md`, `REPRODUCIBILITY.md`, `DATASET_SCHEMA.md`.

## 1. Original branch and commit

- Branch: `cleanup/ground-scene-prep`
- HEAD at checkpoint: `7db29b3`
- Archival branch/tag: `archive/pre-consolidation-20261021` / `archive/pre-consolidation-20261021-src`
- Main at checkpoint: `d322383`

## 2–3. Integration

Completed work from `cleanup/ground-scene-prep` (including Plan C ground recovery) is merged into `main`. Exact final `main` commit is recorded at merge time in git history.

## 4. Archived experiments

See `ARCHIVE_MANIFEST.md`. Golden Plan C binaries: `archive/golden_experts/` (pkl/mp4 local-only) + originals under `results/diagnostics/`.

## 5–6. Active tree

Retain `controllers/ envs/ models/ data_generation/ training/ evaluation/ visualization/ scripts/ config/ experiments/ tests/ datasets/ archive/`.

Plan C: `data_generation/strategies/ground_*.py`, `scripts/run_ground_*.py`.
Dataset/K49: `scripts/build_unified_dataset.py`, `scripts/replay_golden_experts.py`, `scripts/audit_k49.py`.

## 7. Golden regression

| Root | SUCCESS_NEW continuous replay |
|---|---|
| gbi_010 | **3/3** |
| gbi_022 | **3/3** |

## 8. Unified dataset

| Plan | Count |
|---|---:|
| A | 84 |
| B | 121 |
| C | 2 |

Plan C roots are regression fixtures only.

## 9–10. K49

Vocabulary unchanged (49 tokens, HANDOFF=47, tok48 slow reverse). Naive nearest-token physical replay of Plan C experts **fails** at steps ~12–13 (`GROUND_DROP`). Continuous experts remain the golden reference.

## 11. Blockers before training

1. Plan C not K49-representable via quantization alone.
2. Only two Plan C roots.
3. Scene mix (GROUND vs elevated) must stay explicit in metadata.
4. SFT/GRPO not started.
