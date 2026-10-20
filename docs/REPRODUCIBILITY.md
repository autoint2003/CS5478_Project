# Reproducibility

## Recoverable source checkpoint

| Item | Value |
|---|---|
| Pre-cleanup branch | `cleanup/ground-scene-prep` |
| Git archival branch | `archive/pre-consolidation-20261021` |
| Git archival tag | `archive/pre-consolidation-20261021-src` |
| Checkpoint note | `docs/PRE_CONSOLIDATION_CHECKPOINT.txt` |

```powershell
git checkout archive/pre-consolidation-20261021-src
```

There is **no** filesystem `archive/` directory. Historical **code** lives under `experiments/`. All **results** stay under `results/` in their original paths.

## Golden Plan C experts

Leave these locations unchanged (do not move or copy):

- `results/diagnostics/ground_plan_c_final_success/raw/gbi_010__SUCCESS_NEW_expert.pkl`
- `results/diagnostics/ground_plan_c_final_success/raw/gbi_022__SUCCESS_NEW_expert.pkl`

Prefix tapes:

- `results/diagnostics/ground_dynamic_catch_oracle/raw/{root}__catch_expert.pkl`
- `results/diagnostics/ground_postcatch_endgrasp_recovery/raw/{root}__endgrasp_expert.pkl`

### Validate

```powershell
python -u scripts/replay_experts.py
# expect gbi_010: 3/3 and gbi_022: 3/3
```

Or:

```powershell
pytest -q tests/test_plan_c_golden_regression.py -m slow
```

### Rebuild unified dataset

```powershell
python -u scripts/build_expert_dataset.py
```

Tracked manifests: `datasets/unified_experts/`. Heavy records: `data/unified_experts/records/` (gitignored).

### K49 audit

```powershell
python -u scripts/audit_action_vocab.py
```

Reports under `results/consolidation/` (new outputs only; do not alter prior diagnostics).

## Constraints that must not change

- Legal 7D action interface and 20 ms control period
- 39D runtime observation (`obs_gt_airborne_39`)
- SUCCESS_NEW / GROUND_DROP / nominal controller
- Frozen K49 IDs 0–48, HANDOFF=47, token 48 slow reverse wrist
- No privileged state in runtime Transformer observations
- Do not rewrite verified experts with approximate invented actions
- Do not reorganize or delete existing `results/` trees

## Layout

See `docs/ARCHIVE_MANIFEST.md` and `docs/MODULE_ORGANIZATION.md`.
