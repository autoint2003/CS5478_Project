# Reproducibility

## Recoverable checkpoint

| Item | Value |
|---|---|
| Pre-cleanup branch | `cleanup/ground-scene-prep` |
| Archival branch | `archive/pre-consolidation-20261021` |
| Archival tag | `archive/pre-consolidation-20261021-src` |
| Checkpoint note | `archive/checkpoints/PRE_CONSOLIDATION_CHECKPOINT.txt` |

```powershell
git checkout archive/pre-consolidation-20261021-src
```

## Golden Plan C experts

Sources (gitignored binaries; also copied under `archive/golden_experts/`):

- `results/diagnostics/ground_plan_c_final_success/raw/gbi_010__SUCCESS_NEW_expert.pkl`
- `results/diagnostics/ground_plan_c_final_success/raw/gbi_022__SUCCESS_NEW_expert.pkl`

Prefix tapes:

- `results/diagnostics/ground_dynamic_catch_oracle/raw/{root}__catch_expert.pkl`
- `results/diagnostics/ground_postcatch_endgrasp_recovery/raw/{root}__endgrasp_expert.pkl`

### Validate (required after any consolidation)

```powershell
python -u scripts/replay_golden_experts.py
# expect gbi_010: 3/3 and gbi_022: 3/3
```

Or:

```powershell
pytest -q tests/test_plan_c_golden_regression.py -m slow
```

### Rebuild unified dataset

```powershell
python -u scripts/build_unified_dataset.py
```

Writes tracked manifests to `datasets/unified_experts/` and heavy records to `data/unified_experts/records/` (gitignored).

### K49 audit

```powershell
python -u scripts/audit_k49.py
```

Reports under `results/consolidation/K49_*.md|csv|json`.

## Constraints that must not change

- Legal 7D action interface and 20 ms control period
- 39D runtime observation (`obs_gt_airborne_39`)
- SUCCESS_NEW / GROUND_DROP / nominal controller
- Frozen K49 IDs 0–48, HANDOFF=47, token 48 slow reverse wrist
- No privileged state in runtime Transformer observations
- Do not rewrite verified experts with approximate invented actions

## Archive

See `docs/ARCHIVE_MANIFEST.md`. Diagnostics originals remain under `results/diagnostics/` (gitignored). Do not `git clean -fdx` that tree.
