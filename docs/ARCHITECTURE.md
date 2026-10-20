# Architecture

## Layout policy

| Location | Role |
|---|---|
| `experiments/` | Historical experimental **code** only |
| `results/` | Diagnostics and outputs — **do not reorganize** |
| `datasets/` | Tracked unified expert manifests (contents preserved) |
| Active packages | Responsibility-named modules for Plan A/B/C + training |

Source checkpoint before this naming refactor: git tag `refactor/pre-naming-20261021-src`.

Rename map: `docs/RENAME_MAP.md`.

## Dependency map

| Capability | Modules |
|---|---|
| MuJoCo simulation | `envs/simulation/`, `assets/` |
| Legal 7D / 20 ms | `controllers/recovery/action_map.py`, `envs/step_legal.py` |
| Nominal controller | `controllers/nominal.py` |
| 39D observation | `envs/observations/airborne_obs.py` |
| SUCCESS_NEW | `evaluation/success_criteria.py`, `envs/outcomes/recovery_success.py` |
| GROUND_DROP | `envs/outcomes/ground_drop.py` |
| Plan A oracles | `data_generation/oracles/plan_a/` |
| Plan B oracles | `data_generation/oracles/plan_b/` |
| Plan C oracles | `data_generation/oracles/plan_c/` |
| Dataset construction | `data_generation/datasets/` |
| Recording | `data_generation/recording/` |
| Transformer policy | `models/transformer_policy.py` |
| K48 checkpoint format | `models/k48_checkpoint.py`, `models/k48_policy.py` |
| Frozen K49 vocab | `training/action_vocab_k49.py` |
| SFT / GRPO | `training/sft.py`, `training/grpo.py`, `training/pipeline.py` |
| Visualization | `visualization/` |

## Active entry points

```text
python -u scripts/generate_experts.py
python -u scripts/generate_plan_c_experts.py
python -u scripts/build_expert_dataset.py
python -u scripts/replay_experts.py
python -u scripts/audit_action_vocab.py
python -u scripts/train_policy.py
python -u scripts/evaluate_policy.py
python -u scripts/render_trajectory.py
```

Active training configs: `config/training/{unified_transformer,sft,grpo,unified_plan_abc}.yaml`.  
Historical K47/K48 YAMLs: `experiments/config/training/`.
