# Naming refactor report

Checkpoint: git tag `refactor/pre-naming-20261021-src` / branch `refactor/pre-naming-20261021`.

## Validation

| Gate | Result |
|---|---|
| A Imports | Pass — Plan A/B/C oracles, datasets, transformer policy, pipeline, evaluation |
| B Physics invariants | Unchanged by design (moves/renames only; no algorithm edits) |
| C Plan C golden | `gbi_010` 3/3, `gbi_022` 3/3 (`scripts/replay_experts.py`) |
| D Dataset | `datasets/unified_experts/index.json` unchanged census A=84 B=121 C=2 |
| E K49 | 49×7 table, HANDOFF=47, config `K49_tok48_wy0p25` |

## Unresolved / intentional leftovers

1. **K48 checkpoint / policy filenames** kept (`models/k48_checkpoint.py`, `models/k48_policy.py`, `training/k48_config.py`, `training/k47_codebook.py`) — format-specific frozen interfaces.
2. **Observation module** still `envs/observations/airborne_obs.py` — renaming deferred to avoid silent feature-order risk; behavior unchanged.
3. **Historical experiments** under `experiments/` may still mention old paths in prose; executable imports were updated where `ROOT` depth required it.
4. **Empty** `data_generation/strategies/` retained as a deprecation stub pointing at `oracles/`.
5. Training not launched (known K49 Plan C quantization gap remains).

Full path map: `docs/RENAME_MAP.md`.
