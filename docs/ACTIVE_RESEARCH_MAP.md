# Active research map

Maintained work vs historical one-offs under `experiments/`. The `results/` tree is gitignored and is not part of code cleanup.

## Simulation and scenes

| Need | Location |
|---|---|
| Elevated-table official scene | `assets/scene.xml` via `envs/simulation/grasp_sim.py` |
| Ballistic / impact diagnostic scenes | `assets/scene_ballistic_impact.xml`, `scene_impact_diag.xml` |
| Ground scene | `GROUND_SCENE` — `envs/simulation/ground_scene.py` → `assets/scene_ground.xml`, sim `envs/simulation/ground_sim.py` |
| Snapshot capture/restore | `envs/simulation/capture.py` |
| Legal 7D step | `envs/step_legal.py` + `controllers/recovery/action_map.py` |
| 39D observation | `envs/observations/` |
| TABLE_DROP (elevated) | `envs/outcomes/table_drop.py` |
| GROUND_DROP | `envs/outcomes/ground_drop.py` |
| Success / horizon | `envs/outcomes/horizon_outcome.py`, `recovery_success.py`, `physical_recovery.py` |

## Legal Recovery execution

| Need | Location |
|---|---|
| Episode loop | `training/recovery_episode.py`, `recovery_execution.py` |
| Nominal + HANDOFF | `controllers/nominal.py`, horizon outcome |
| K49 vocabulary | `training/action_vocab_k49.py` |
| Unified transformer | `models/k48_policy.py`, `training/unified_k48.py` |

## Validated Plan A / B / C references

| Plan | Generators / pilots | Evidence |
|---|---|---|
| A sliding/pivoting/return | `experiments/sliding/pilot_*.py`, `scripts/generate_k49_sliding_recatch_dataset.py` | `results/expert_generation/sliding_*`, `return_aware_*`, wrist-tilt diagnostics |
| B release–recatch | `scripts/generate_k49_lost_grasp_recovery.py`, `generate_k49_*recatch*.py` | `results/expert_generation/k49_lost_grasp_*`, `k49_*recatch*` |
| C ball-impact / freefall | `scripts/run_freefall_*.py`, `run_*ball_impact*`, `run_plan_c_*`, `generate_k49_ball_impact_*` | `results/expert_generation/k49_ball_impact_*`, `results/diagnostics/freefall_*`, `plan_c_*` |

Full success inventory: `docs/PRESERVED_EVIDENCE.md`.

## Training and evaluation

| Need | Location |
|---|---|
| Configs | `config/training/k48/`, `config/simulation/`, `config/recovery/` |
| SFT / GRPO entry | `scripts/train.py`, `training/train_sft.py`, `train_grpo.py` |
| Eval | `scripts/evaluate.py`, `evaluation/` |
| Adaptive-g pilots | `experiments/adaptive_g/` |

## Ground-scene development

| Need | Location |
|---|---|
| Outcome semantics | `docs/GROUND_SCENE_OUTCOMES.md` |
| Validation report | `results/diagnostics/ground_scene_validation/` |
| Maintained runner | `scripts/run_ground_scene_validation.py` |
| Unit test | `tests/test_ground_drop.py` |

## Historical one-offs

| Need | Location |
|---|---|
| Former `scripts/_tmp_*` probes | `experiments/legacy/tmp/` |
| Sliding / K49 / ball-impact one-shots | `experiments/sliding/_tmp_*`, `experiments/recatch/k49/_tmp_*`, `experiments/ball_impact/_tmp_*` |
| Older legacy CLIs | `experiments/legacy/` |
| Layout correction notes | `docs/CLEANUP_LAYOUT_CORRECTION.md` |
