# Historical layout manifest

There is **no** top-level `archive/` directory.

## Policy

| Location | Role |
|---|---|
| `experiments/` | **Only** archive for historical / superseded **experimental code** |
| `results/` | All diagnostic outputs, trajectories, reports, videos — **leave untouched** |
| Git tag `archive/pre-consolidation-20261021-src` | Recoverable pre-cleanup **source** history (git ref name only; not a filesystem folder) |

Do **not**:

- Create a new top-level `archive/` tree
- Move, rename, delete, or reorganize existing files under `results/`
- Copy result binaries into `experiments/`

## Source checkpoint

See `docs/PRE_CONSOLIDATION_CHECKPOINT.txt`.

```powershell
git checkout archive/pre-consolidation-20261021-src
```

Pre-cleanup branch tip: `7db29b325202d04e7a2b774b79ea1178a368fc64`  
Branch: `archive/pre-consolidation-20261021`

## Golden Plan C evidence (in situ under results/)

Leave these paths unchanged:

| Path | Verified roots |
|---|---|
| `results/diagnostics/ground_plan_c_final_success/` | `gbi_010`, `gbi_022` |
| `results/diagnostics/ground_dynamic_catch_oracle/` | prefix catch tapes |
| `results/diagnostics/ground_postcatch_endgrasp_recovery/` | brake / slow-lift tapes |

Replay: `python -u scripts/replay_experts.py`

## Historical experimental code

Obsolete scripts and heuristics belong under `experiments/` (including `experiments/legacy/`). Active Plan A/B/C entry points remain under `scripts/` and `data_generation/strategies/`.

## Other diagnostics

All other folders under `results/diagnostics/` remain in place (gitignored). They are not relocated into `experiments/`.
