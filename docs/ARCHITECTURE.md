# Architecture

Active code generates and evaluates six expert branches on the current Precision Oracle. Historical experiments stay under `experiments/`. Diagnostic outputs stay under `results/` and are not reorganized. `mujoco_menagerie/` is third-party and is not modified.

## Repository boundaries

| Path | Role |
|---|---|
| `assets/` | MuJoCo XML and robot includes |
| `config/` | Active controller, recovery, simulation, and training YAML |
| `controllers/` | Nominal controller, Cartesian/gripper execution, legal 7D action map |
| `envs/` | Simulation, contacts, captures, observations, ground-drop and recovery outcomes |
| `sensors/` | Tactile and lateral-offset measurement |
| `data_generation/` | Initial conditions, Plan A/B/C stage-1 oracles, Precision Oracle, recording, dataset construction code |
| `evaluation/` | SUCCESS_NEW / SUCCESS_12S, handoff continuation, hand-frame kinematics, Plan C golden replay |
| `models/` | Transformer policy and checkpoint loaders |
| `training/` | Current SFT, GRPO, and the existing K49 action vocabulary used by those trainers |
| `visualization/` | Rendering and video |
| `scripts/` | User-facing commands |
| `tests/` | Regression tests |
| `experiments/` | Archived experiment code, old configs, and historical documents |
| `results/` | Existing outputs. Do not move or delete |
| `data/` | Local datasets (gitignored). Legacy manifests and the future `experts_v2` namespace |
| `coursework/` | Course PDFs, outside `docs/` |
| `mujoco_menagerie/` | Vendored robot models |

`data_generation/datasets/` is construction code. It is not a dataset. The erroneous repository-root `datasets/` tree was removed; historical unified manifests live only at `data/unified_experts/`.

## Precision Oracle

`data_generation/oracles/precision/state_machine.py` is the only scheduler. Correction skills and contact mechanics are the modules listed in `docs/ORACLE_STATE_MACHINE_SPEC.md` section 11. Plan A, B, and C stage-1 search stays in `data_generation/oracles/plan_a/`, `plan_b/`, and `plan_c/`.

## Seven expert branches

| Branch id | Stage 1 | Continuation |
|---|---|---|
| `A_STABLE_DIRECT_HANDOFF` | Plan A stable grasp | Direct nominal HANDOFF |
| `A_STABLE_PRECISION` | Plan A stable grasp | Precision Oracle |
| `B_STABLE_DIRECT_HANDOFF` | Plan B stable grasp | Direct nominal HANDOFF |
| `B_STABLE_PRECISION` | Plan B stable grasp | Precision Oracle |
| `C_RECATCH_DIRECT_HANDOFF` | Plan C recatch | Direct nominal HANDOFF |
| `C_RECATCH_PRECISION` | Plan C recatch | Precision Oracle |
| `D_PRECISION` | none | Precision Oracle |

Production Precision always uses support-transfer DX with `use_verified_lateral_fallback=false`. Trajectories that entered verified_lateral fallback are not production experts.

Selection and provenance live in `data_generation/expert_pipeline/`. Continuation dispatch calls `run_precision` or `probe_direct_handoff`. It does not implement a second scheduler and does not write a dataset on import. A continuation that is not physically successful is returned with `expert = None`.

Stage-1 search is still owned by the plan packages. The pipeline does not reimplement grasp, recatch, or contact mechanics.

## Dataset versions

| Namespace | Version | Training use |
|---|---|---|
| `data/unified_experts/` | `unified_experts_v1` | Historical. Not the source for regenerated Precision experts |
| `data/supervised_targets*.jsonl` | prior SFT manifests | Existing trainers. Not silently mixed with `experts_v2` |
| `data/experts_v2/` | `experts_v2` | Destination for regenerated experts. Empty until a later generation run |

Every `experts_v2` row carries `dataset_version`, `branch_id`, `continuation`, `source_ic_id`, `physical_success`, `success_criteria_version`, `generated_utc`, `git_commit`, and `config_id`. Precision rows also carry `oracle_implementation = precision_state_machine_v23`. Rows without that stamp, and any Precision row produced by an older Oracle, are rejected by `row_allowed_for_training`. Direct-HANDOFF history is not treated as invalid merely by age; it is admitted only when explicitly tagged with a direct-HANDOFF branch and `physical_success` true, and it is not copied into `experts_v2` automatically.

`SUCCESS_NEW` (position ≤ 12 mm, angle ≤ 10°, hold ≥ 0.5 s, legal HANDOFF, unchanged nominal continuation) is required to stamp an expert. `SUCCESS_12S` is recorded separately. Failures are never promoted.

## Physical outcomes

`GROUND_SCENE` uses the MuJoCo ground plane at `GROUND_Z = 0`. The geometric guard is `GROUND_DROP_Z = 0.04` m. `envs.outcomes.ground_drop.arm` arms the latch at recovery start. Contact support uses `SUPPORT_TIME_S = 0.02` and `BOUNCE_DV = 0.15`. Bilateral finger contact blocks the latch. A latched `GROUND_DROP` is terminal. The 39D observation still subtracts `TABLE_TOP = 0.40` as a calibration offset, not as a physical table. Hand-frame working target is `P_TGT` in `evaluation/success_criteria.py`.

## Active commands

```text
python -u scripts/generate_experts.py --list-branches
python -u scripts/generate_experts.py --list-ics
python -u scripts/generate_experts.py --all-branches
python -u scripts/generate_experts.py --branch B_STABLE_PRECISION
python -u scripts/validate_dataset.py --manifest data/experts_v2/manifest.jsonl
python -u scripts/replay_experts.py
python -u scripts/train_policy.py --supervised --config config/training/sft.yaml
python -u scripts/evaluate_policy.py --eval --config config/training/unified_transformer.yaml --checkpoint <ckpt>
python -u scripts/render_trajectory.py
```

Generation reads post-Stage-1 ICs from `config/generation/experts_v2.yaml`, runs the current Oracle or direct HANDOFF, and writes only under `data/experts_v2/`. Failures go to `data/experts_v2/diagnostics/failures/`.

## Dependency direction

`envs` and `controllers` do not import `experiments` or dataset writers. The Precision scheduler imports correction and mechanics modules and `evaluation` for handoff measurement. `data_generation.expert_pipeline` imports the scheduler and the direct-HANDOFF probe. `training` imports models, environments, and the existing action vocabulary. `scripts` are thin entry points. Active packages do not import `experiments/`.

Archived K49 mining, stage-2 pilots, and one-off probes remain importable from their experiment folders for reproduction. They are not the supported generation path.
