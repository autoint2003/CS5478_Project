# Repository cleanup report (Phase I)

## Rollback point (verified)

| Field | Value |
|---|---|
| Commit | `d3223833647a8dc7107dc7898bec0b7abef55a47` |
| Branch | `cleanup/ground-scene-prep` |
| Critical hashes | `archive/manifests/ROLLBACK_POINT.json` |

Uncommitted user work on this branch was **not** discarded. Validated pilots, generators, and evaluation modules remain in the active tree. Only classified historical one-offs and superseded diagnostics were moved into `archive/`.

## What stayed ACTIVE_CORE

Maintained packages (unchanged layout from Round-3 reorganization):

- `controllers/` — nominal, Cartesian, gripper, recovery action map
- `envs/` — simulation, observations, contacts, outcomes, ballistic impact, `step_legal`
- `models/` — K48/K49 policy + historical checkpoints loaders
- `data_generation/` — official generators, strategies, oracles, validation
- `training/` — SFT/GRPO/unified K48, recovery episode, K49 vocab
- `evaluation/` — closed-loop eval + grasp-configuration / full-objective utilities
- `visualization/`, `config/`, `sensors/`, `tests/`, `assets/`

Round-3 package split was **not** repeated.

## What stayed VALIDATED_RESEARCH (active paths)

- `experiments/sliding/pilot_*.py`, diagnose/build helpers
- `experiments/adaptive_g/`
- `experiments/handoff/` audits and benchmarks (non-`_tmp_`)
- `scripts/generate_k49_*.py`, `scripts/run_*.py`, expand/merge helpers
- `evaluation/full_objective.py`, `grasp_configuration.py`, mechanism audits
- `results/expert_generation/*` (all Plan A/B/C expert trees; gitignored, on disk)
- Key `results/diagnostics/*` freefall / lateral / sliding / plan_c / table_drop evidence

See `archive/manifests/PRESERVED_EVIDENCE.md`.

## What was archived (HISTORICAL)

- 33 `scripts/_tmp_*` + `_continue_*` → `archive/scripts/tmp/`
- 34 `experiments/**/_tmp_*` → `archive/experiments/...`
- `data_generation/maintenance/organize_k49_new_files.py` → `archive/experiments/legacy/`
- 22 superseded `results/diagnostics/*` trees → `archive/results/diagnostics/`

Full rows: `archive/manifests/ARCHIVE_MANIFEST.csv`.

## What was deleted

**Nothing.** Regenerable videos and `_frames_*` caches are listed as `REDUNDANT_GENERATED` / `retained_pending_delete` only.

## Reference repairs

- Active runtime imports do not depend on archived `_tmp_*` scripts.
- `docs/MIGRATION.md` still documents the *prior* Round-3 `_tmp_*` destinations under `experiments/`; those files now live one level deeper under `archive/experiments/`. Historical reproduction should run from `archive/` with `PYTHONPATH` at repo root, not by importing archive modules into production paths.
- No global string rewrite of production code was performed.

## Regression tests (Phase I gate)

| Check | Command | Result |
|---|---|---|
| Physics regression (incl. snapshot / 39D / contacts) | `python -u tests/test_physics_regression.py` | `PHYSICS_REGRESSION_OK` |
| TABLE_DROP / elevated-table latch | `python -u tests/test_table_drop.py` | `TABLE_DROP_OK` |
| Grasp-configuration metrics | `pytest tests/test_grasp_configuration_metrics.py` | 28 related unit tests passed with adaptive-g suite |
| Adaptive independent-g | `pytest tests/test_adaptive_independent_g.py` | passed (included above) |
| Critical imports | action map, K49 vocab, table_drop, capture | `imports_ok` |

Plan A/B/C **full expert replay** from `results/expert_generation/` was not re-executed in this cleanup gate (artifacts are gitignored local evidence; generators and paths preserved). No claim of fresh expert triple-replay is made here.

Note: collecting `test_physics_regression.py` / `test_table_drop.py` via bare `pytest` errors on non-fixture parameters; run them as scripts as documented in `MODULE_ORGANIZATION.md`.

## Cleanup commit

Recorded on branch `cleanup/ground-scene-prep` after this document is committed (hash in git log / Part I final report).

## Final maintained structure (compact view)

```text
controllers/  envs/  models/  data_generation/  training/
evaluation/  visualization/  config/  scripts/  experiments/
  adaptive_g/  ball_impact/  handoff/  recatch/  sliding/  legacy/
assets/  tests/  docs/
archive/          # historical only
results/          # gitignored artifacts; experts + active diagnostics
```
