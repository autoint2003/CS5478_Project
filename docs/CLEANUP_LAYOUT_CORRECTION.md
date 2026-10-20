# Cleanup layout correction

Corrects the unnecessary top-level `archive/` directory introduced in Phase I.
Does **not** change GROUND_SCENE / GROUND_DROP, physics, controllers, actions, observations, or training.

## Commits

| Role | Hash |
|---|---|
| Pre-cleanup rollback | `d3223833647a8dc7107dc7898bec0b7abef55a47` |
| Phase I (introduced `archive/`) | `4663816ccd78e7f7cb7bc693565e4de8d023104e` |
| Phase II GROUND_SCENE | `532640fa08d81fb88b2293f118c33f10746e846f` |

## Moves (89, 0 collisions)

| From | To | Count |
|---|---|---|
| `archive/results/diagnostics/<name>/` | `results/diagnostics/<name>/` (restored) | 22 |
| `archive/experiments/sliding/_tmp_*` | `experiments/sliding/_tmp_*` | 14 |
| `archive/experiments/recatch/k49/_tmp_*` | `experiments/recatch/k49/_tmp_*` | 15 |
| `archive/experiments/ball_impact/_tmp_*` | `experiments/ball_impact/_tmp_*` | 4 |
| `archive/scripts/tmp/_tmp_*` / `_continue_*` | `experiments/legacy/tmp/` | 33 |
| `archive/experiments/legacy/organize_k49_new_files.py` | `experiments/legacy/organize_k49_new_files.py` | 1 |

`results/` was otherwise left alone (gitignored; outside cleanup scope after this restore).

## Removed

- Top-level `archive/` (including redundant manifests under `archive/manifests/`).

## Documentation retained under `docs/`

- `docs/PRESERVED_EVIDENCE.md` — validated expert paths
- `docs/REPOSITORY_CLEANUP_REPORT.md` — Phase I summary (paths updated)
- This file — layout correction record
