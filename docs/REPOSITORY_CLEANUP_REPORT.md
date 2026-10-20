# Repository cleanup report (Phase I + layout correction)

## Rollback point (verified)

| Field | Value |
|---|---|
| Commit | `d3223833647a8dc7107dc7898bec0b7abef55a47` |
| Branch | `cleanup/ground-scene-prep` |
| Phase I commit | `4663816ccd78e7f7cb7bc693565e4de8d023104e` |
| Layout correction | see `docs/CLEANUP_LAYOUT_CORRECTION.md` |

Uncommitted user work was **not** discarded. Validated pilots, generators, and evaluation modules remain in the active tree.

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
- `results/expert_generation/*` and `results/diagnostics/*` (gitignored, on disk; **not** reorganized)

See `docs/PRESERVED_EVIDENCE.md`.

## Historical one-offs (under `experiments/`, not a top-level `archive/`)

| Content | Location |
|---|---|
| Former `scripts/_tmp_*` / `_continue_*` | `experiments/legacy/tmp/` |
| Sliding / recatch / ball-impact `_tmp_*` | `experiments/sliding/`, `experiments/recatch/k49/`, `experiments/ball_impact/` |
| `organize_k49_new_files.py` | `experiments/legacy/organize_k49_new_files.py` |
| Older legacy CLIs | `experiments/legacy/` (pre-existing) |

The temporary top-level `archive/` directory was removed; see `docs/CLEANUP_LAYOUT_CORRECTION.md`.

## What was deleted

**Nothing** from validated experts or result trees. Regenerable videos/`_frames_*` were never deleted.

## Reference repairs

- Active runtime does not import `_tmp_*` one-offs.
- Round-3 destinations in `docs/MIGRATION.md` (`experiments/sliding/_tmp_*`, etc.) match the restored layout again.
- Do not import historical `_tmp_*` scripts into the production path.

## Regression tests (Phase I gate)

| Check | Command | Result |
|---|---|---|
| Physics regression | `python -u tests/test_physics_regression.py` | `PHYSICS_REGRESSION_OK` |
| TABLE_DROP | `python -u tests/test_table_drop.py` | `TABLE_DROP_TESTS_OK` |
| Grasp / adaptive-g units | `pytest tests/test_grasp_configuration_metrics.py tests/test_adaptive_independent_g.py` | passed |

## Final maintained structure (compact view)

```text
controllers/  envs/  models/  data_generation/  training/
evaluation/  visualization/  config/  scripts/  experiments/
  adaptive_g/  ball_impact/  handoff/  recatch/  sliding/  legacy/
    legacy/tmp/          # former scripts/_tmp_* probes
assets/  tests/  docs/
results/                 # gitignored; leave alone
```
