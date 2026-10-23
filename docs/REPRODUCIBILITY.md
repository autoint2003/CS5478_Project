# Reproducibility

## Environment

Commands run from the repository root on Windows, using the project Python environment with MuJoCo available. Robot XML comes from `assets/` and `mujoco_menagerie/` (do not edit the vendored menagerie). Active YAML lives in `config/`. Simulation timestep and the legal action interface stay at 20 ms and 7D.

## Git checkpoints

| Checkpoint | Commit | What it preserves |
|---|---|---|
| `main` before this refactor | `3c361e4bdc3d4a4e4a5531539cd45b9c1d001971` | Pre-refactor history on main |
| Pre-refactor freeze on `checkpoint/pre-six-branch-refactor-20261023` | `121324c9e57dde27c12fd220dfcb5517e22707ee` | Tree immediately before the six-branch reorganization, including the to_UP brake-before-vertical fix and removal of the erroneous root `datasets/` tree |
| Refactor merge on `main` | `2b140f9431b118b2d210668359c17e41fb2eab7a` | Precision module split, six-branch `experts_v2` pipeline, docs reduced to three active files. Checkpoint branch tip matches this commit; `121324c` remains reachable |

```powershell
git checkout checkpoint/pre-six-branch-refactor-20261023
```

Earlier consolidation tags remain `archive/pre-consolidation-20261021-src` and `refactor/pre-naming-20261021-src`. There is no filesystem `archive/` directory. Historical code is under `experiments/`. Results stay under `results/`.

The file-move map for this reorganization is `experiments/repository_refactor_20261023/MOVE_MAP.md`.

## Oracle regression

Scheduler tests (existing initial conditions, no dataset write):

```powershell
python -u -m pytest -q tests/test_precision_oracle_v23_scheduler.py tests/test_expert_branch_dispatch.py
```

Import check:

```powershell
python -c "import data_generation.oracles.precision as p; from data_generation.oracles.precision.state_machine import run_precision; print(p.run_precision is run_precision)"
```

Branch list (does not simulate):

```powershell
python -u scripts/generate_experts.py --list-branches
```

The to_UP brake-before-vertical regression script is historical and lives at `experiments/precision/run_to_up_brake_before_vertical.py`. Its last recorded result is `results/diagnostics/seq4_to_up_brake_before_vertical/`. Re-running it replays the same Seq4 initial condition and writes a new diagnostic under that results tree; it is not part of dataset generation.

## Six dataset branches

Authoritative IC pool: `results/data_generation/stage2_abc_final_v2/` (post-Stage-1 `capture`+`ref`). Generation config: `config/generation/experts_v2.yaml`. Output: `data/experts_v2/` with `dataset_version=experts_v2`. Does not overwrite `data/unified_experts/`.

```powershell
python -u scripts/generate_experts.py --list-branches
python -u scripts/generate_experts.py --list-ics
# Smoke (temporary output, one IC per branch):
python -u scripts/generate_experts.py --all-branches --limit 1 --output results/diagnostics/experts_v2_smoke_tmp
# Production (resume-safe; SUCCESS_NEW required to admit):
python -u scripts/generate_experts.py --all-branches
# Or one branch:
python -u scripts/generate_experts.py --branch B_STABLE_PRECISION
python -u scripts/validate_dataset.py --manifest data/experts_v2/manifest.jsonl
```

Branch IDs: `A_STABLE_PRECISION`, `B_STABLE_DIRECT_HANDOFF`, `B_STABLE_PRECISION`, `C_RECATCH_DIRECT_HANDOFF`, `C_RECATCH_PRECISION`, `D_PRECISION`.

First production generation (git `2b140f9`): 122 accepted experts under `data/experts_v2/` (A:0, B_direct:16, B_precision:21, C_direct:40, C_precision:42, D:3). Reports: `results/diagnostics/experts_v2_generation/GENERATION_REPORT.json` and `AUDIT_REPORT.json`. Historical `data/unified_experts/` was not overwritten.

## Historical Plan C golden experts

Leave these pickles in place:

- `results/diagnostics/ground_plan_c_final_success/raw/gbi_010__SUCCESS_NEW_expert.pkl`
- `results/diagnostics/ground_plan_c_final_success/raw/gbi_022__SUCCESS_NEW_expert.pkl`

Replay:

```powershell
python -u scripts/replay_experts.py
python -u -m pytest -q tests/test_plan_c_golden_regression.py -m slow
```

Expected result is `gbi_010: 3/3` and `gbi_022: 3/3` SUCCESS_NEW. These fixtures are direct recatch-plus-handoff tapes. They are not `experts_v2` Precision rows.

## Training and evaluation

Existing trainers are unchanged and still read their current configs:

```powershell
python -u scripts/train_policy.py --supervised --config config/training/sft.yaml
python -u scripts/evaluate_policy.py --eval --config config/training/unified_transformer.yaml --checkpoint <ckpt>
python -u scripts/render_trajectory.py
```

Do not point those commands at `data/unified_experts/` in order to train the regenerated Precision policy. After regeneration, build a manifest under `data/experts_v2/` and pass that path explicitly. `config/training/unified_plan_abc.yaml` documents the legacy `unified_experts_v1` layout and is not a loader for `experts_v2`.

## Historical experiments

| Location | Contents |
|---|---|
| `experiments/stage2_mining/` | Stage-2 mining, pilots, and K49 dataset builders |
| `experiments/plan_b_table_free/` | Table-free Plan B recovery and migration |
| `experiments/plan_c_research/` | Plan C sidewall, post-lift, and expert-search scripts |
| `experiments/precision/` | Precision Oracle development runs and frozen baselines |
| `experiments/precision_probes/` | `_probe_*`, `_fixup_*`, and one-off audits |
| `experiments/expert_freeze/` | Unified-expert freeze and K49 representability audit |
| `experiments/rendering_comparisons/` | Comparison videos |
| `experiments/docs_historical/` | Superseded architecture, schema, audit, and rename notes |
| `experiments/config/` | Old training YAML |
| `experiments/legacy/`, `experiments/diagnostics/`, `experiments/recatch/`, `experiments/sliding/` | Earlier recovery campaigns |

K49 codebook workflows in those folders are reproducible from their original entry scripts. They are not the command sequence for the current Oracle.
