# CS5478 K48 grasp recovery

A MuJoCo Panda recovers a cylinder after a ball impact. Recovery is a legal 47-token motion codebook plus a bounded residual. Token 47 is HANDOFF: it ends recovery, consumes no physics step, and does not decode a residual. The recovery clock stops at 4 seconds. After handoff, the unchanged nominal controller lifts and must hold the grasp for 12 seconds. That `SUCCESS_12S` outcome is the only recovery success. A 3 mm geometric improvement is not a success gate.

The audited training set is the supervised and GRPO entry. `scripts/train.py` and `scripts/evaluate.py` call `training/unified_k48.py`. That file dispatches to `training/train_sft.py` and `training/train_grpo.py`. Formal modules do not import `experiments/`.

## Directory layout

Run commands from the repository root.

| Path | Role |
|---|---|
| `assets/` | MuJoCo XML, including `scene_ballistic_impact.xml` |
| `config/` | `simulation/`, `controllers/`, `recovery/`, `training/k48/` (active recovery v3), `training/k47/`, `training/legacy/` |
| `controllers/` | `nominal.py`, `cartesian/`, `gripper/`, `recovery/action_map.py` |
| `envs/` | `step_legal.py`, `ballistic_impact.py`, `simulation/`, `observations/`, `contacts/`, `outcomes/`, `initialization/` |
| `sensors/` | Finger-pad tactile array and the hand-x offset estimate `lateral_offset.py` |
| `models/` | K48 policy and checkpoint. `airborne_v3.py` is the 39-D policy and `step_v3`. `historical/` is K33/K36 |
| `data_generation/` | Official generators, `strategies/`, `oracles/`, `initial_conditions/`, `recording/`, `validation/`, `maintenance/` |
| `training/` | `train_sft.py`, `train_grpo.py`, `dataset.py`, `grpo_progress.py`, `k47_codebook.py`, `reward_v2.py`, `unified_k48.py` |
| `evaluation/` | Closed-loop K48 eval, data audit, recatch legality, handoff-target check |
| `visualization/` | Camera, captions, MP4 encoding, and tape replay |
| `experiments/` | `ball_impact/`, `sliding/`, `recatch/{airborne,natural_loss,legal,top_impact}/`, `handoff/`, `legacy/` |
| `scripts/` | Thin entry points. Package roles are in `docs/MODULE_ORGANIZATION.md`. |
| `tests/` | Regression tests. `tests/fixtures/physics_state_baseline.npz` is the later-change state baseline |
| `data/` | Manifests, demonstrations, and `data/snapshots/` |
| `results/` | New SFT, GRPO, evaluation, and diagnostic outputs |
| `results/archive/demo_results/` | Historical diagnostics and the K=47 checkpoint used to initialize K=48 |
| `mujoco_menagerie/` | Vendored robot models. Not project control code |
| `docs/` | Architecture notes, `docs/FINAL_REFACTOR_CHECK.md`, and the course PDFs in `docs/coursework/` |

`data/`, `results/archive/demo_results/`, and `results/` are gitignored. Training writes under `results/sft/`, `results/grpo/`, and `results/evaluations/`. File moves are listed in `docs/MIGRATION.md`.

## Dataset

| File | Rows |
|---|---:|
| `data/supervised_targets.jsonl` | 303 training demonstrations (296 recovery, 7 nominal-solvable controls with weight 0) |
| `data/valid_demonstrations.jsonl` | 22 validation demonstrations |
| `data/grpo_initial_conditions.jsonl` | 87 confirmed-recoverable training initial conditions |
| `data/nominal_solvable_control.jsonl` | 7 nominal-solvable controls, excluded from the GRPO recovery pool |
| `data/valid_recovery.jsonl` | 8 validation recovery initial conditions |

SFT reads `data/supervised_targets.jsonl` only. The 22 validation rows stay out of the supervised loss. Train and validation snapshots do not overlap. Every recovery row has `verified_success: true` and `nominal_class: RECOVERY_CANDIDATE`. The 7 control rows are `NOMINAL_SOLVABLE` with `sample_weight` 0. A recovery tape that ends before 4 seconds supervises token 47 on the observation after its last action. A tape that lands exactly on the 4 second limit does not add an extra HANDOFF target; the environment forces handoff before the next observation.

Supervised strategy counts: `chase_lost` 101, `inward_reseat` 65, `controlled_slide` 47, `release_long` 47, `follow_early` 26, `immediate_handoff` 7, `release_short` 6, `later_close` 4. Validation: `inward_reseat` 10, `chase_lost` 4, `controlled_slide` 4, `follow_early` 4. `controlled_slide` and `inward_reseat` are one Gravity-Assisted In-Hand Sliding family. Task success is `SUCCESS_12S` after a legal recovery and the unchanged nominal handoff. Finger slip and post-impact tilt are diagnostic fields, not extra success gates.

Independent nominal-fail snapshots in the current set, with at least one legal successful tape: sliding 77 train / 11 valid, horizontal recatch 22 / 6, vertical recatch 41 / 5. One snapshot can hold several tapes, so these counts are lower than the jsonl row counts. The quota targets 100/20, 50/10, and 50/10 are not filled. The census is `results/diagnostics/dataset_expansion/CORRECTION_REPORT.md`.

Category manifests: `data/gravity_assisted_in_hand_sliding/`, `data/controlled_sliding/`, `data/inward_sliding/`, `data/recatch/` (horizontal cylinder), `data/vertical_recatch/`, `data/regrasp/`, and `data/immediate_handoff/`. Initial states are in `data/snapshots/`. Two vertical rows, `ball_train_v2_0168` and `ball_train_v2_0261`, remain `SUCCESS_12S` but have `counts_as_recatch: false` because the stored-action replay shows only one both-off contact sample. They are not in the supervised jsonl as vertical recatch.

## Setup

From the repository root, with the audited environment (MuJoCo 3.12, PyTorch, NumPy):

```powershell
python -m pip install -r requirements.txt
```

The training entry points import PyTorch in addition to the packages in `requirements.txt`. Use `python -u` so logs flush.

## Dataset commands

Print the rebuilt category counts. This does not generate data:

```powershell
python -u scripts/generate_data.py --counts
```

Generation entry point, when new ballistic initial conditions are actually required:

```powershell
python -u scripts/generate_data.py --recatch --target-train 200 --target-valid 50 --max-search-trials 400
python -u scripts/generate_data.py --controlled-slide --target-train 200 --target-valid 50 --max-search-trials 400
python -u scripts/generate_data.py --inward-reseat --target-train 200 --target-valid 50 --max-search-trials 400
```

`--max-search-trials` is a cap, not a request to synthesize that many initial conditions. Static dataset checks:

```powershell
python -u evaluation/audit_pretrain_data.py static
```

`python -u evaluation/verify_handoff_targets.py` replays every recovery demonstration in physics and then checks training batches. It is the full physical check, not a quick import test.

## Training and evaluation

Run these from the repository root. Evaluation is deterministic argmax on CPU. Training uses the `device` in the config.

Check the codebook, the HANDOFF head, one supervised loss, and the reward order. This does not update parameters or write a checkpoint:

```powershell
python -u scripts/train.py --check
```

### Supervised training on the current set

40 epochs, batch 8, seed 1, learning rate 3e-4. The run reads the 303-row `data/supervised_targets.jsonl` and trains from scratch. `init: scratch` builds a new K48 network with PyTorch's default linear and transformer initialization. The residual head starts at zero. No K=47 or K=48 checkpoint is loaded. The 47 motion codes remain the fixed codebook; they are not learned parameters.

`config/training/k48/unified_k48_recovery_v3_current.yaml` writes `results/sft/current/checkpoint_epoch_0001.pt` through `checkpoint_epoch_0040.pt`. The previous K=48 file `results/sft/checkpoint_epoch_0040.pt` is left in place. Do not pass `--checkpoint`, and do not use `config/training/k48/unified_k48_recovery_v3.yaml` for this run. That older config still expands the historical K=47 checkpoint.

```powershell
python -u scripts/train.py --supervised --config config/training/k48/unified_k48_recovery_v3_current.yaml
```

After it finishes, evaluate the new weights on the valid snapshots from the current quota table: 11 sliding, 6 horizontal recatch, and 5 vertical recatch. Five vertical snapshots are also sliding snapshots, so the closed-loop run has 17 episodes. The list is `data/valid_skill_quotas.jsonl`. The summary goes to `results/evaluations/current/heldout_train.json` and includes `by_category`. It does not use the older 8-row `data/valid_recovery.jsonl`.

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3_current.yaml --checkpoint results/sft/current/checkpoint_epoch_0040.pt
```

The GRPO commands below still load `results/sft/checkpoint_epoch_0040.pt`. Point them at `results/sft/current/checkpoint_epoch_0040.pt` only after this SFT run has finished and that file exists.

### GRPO reward v1

Three passes over the 87 training recovery initial conditions, 261 updates. Writes `results/grpo/v1/`. The last file is `results/grpo/v1/checkpoint_u0261.pt`. The success terminal is only `SUCCESS_12S`.

```powershell
python -u scripts/train.py --grpo --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt
```

### Tree-augmented GRPO

`config/training/k48/unified_k48_recovery_v3_tree80.yaml` writes `results/grpo/tree80/`. One round visits each unique training snapshot once. `--checkpoint` selects the file to load. `--ic-epochs` is the number of new rounds; omit it to use `ic_epochs` in the config. A finished round is saved as `checkpoint_epoch_XXXX.pt` and includes the Adam first and second moments. The KL reference stays `results/sft/checkpoint_epoch_0040.pt`.

Start from supervised epoch 40. `results/grpo/tree80` must be empty:

```powershell
python -u scripts/train.py --adaptive-grpo --config config/training/k48/unified_k48_recovery_v3_tree80.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt --ic-epochs 1
```

Continue from a saved epoch that contains Adam moments. Replace the checkpoint path and the round count:

```powershell
python -u scripts/train.py --adaptive-grpo --config config/training/k48/unified_k48_recovery_v3_tree80.yaml --checkpoint results/grpo/tree80/checkpoint_epoch_0001.pt --ic-epochs 1
```

`results/grpo/tree80/history/checkpoint_epoch_0001.pt` is the earlier weights-only pass. It has no Adam moments, and training refuses to resume it.

### GRPO reward v2

`config/training/k48/unified_k48_recovery_v3_reward_v2.yaml` runs nine passes, 783 updates, and writes `results/grpo/v2/`. A checkpoint saved after update 261 is the end of the first three passes. This command continues that file from update 262 through 783 and appends `training_log.jsonl` and `rollouts.jsonl`. It does not overwrite `checkpoint_u0261.pt`. The frozen KL reference stays `results/sft/checkpoint_epoch_0040.pt`. The last new file is `results/grpo/v2/checkpoint_u0783.pt`.

```powershell
python -u scripts/train.py --grpo --config config/training/k48/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/grpo/v2/checkpoint_u0261.pt
```

### Evaluation

The default IC list is `data/valid_recovery.jsonl` (8 validation recovery initial conditions). The summary is `results/evaluations/heldout_train.json`.

SFT epoch 40:

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt
```

GRPO v1, same 8 validation initial conditions:

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/grpo/v1/checkpoint_u0261.pt
```

GRPO v2, same 8 validation initial conditions:

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/grpo/v2/checkpoint_u0261.pt
```

`--eval-data` selects another recovery list. The 87 training recovery initial conditions are `data/grpo_initial_conditions.jsonl`. Pass `--eval-output` so this does not overwrite `results/evaluations/heldout_train.json`. The rows are written to `closed_loop.json` in that directory.

```powershell
python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml --checkpoint results/grpo/v1/checkpoint_u0261.pt --eval-data data/grpo_initial_conditions.jsonl --eval-output results/evaluations/raw/grpo_v1_train_eval
```

Replace the checkpoint path to evaluate another run on the same 87 initial conditions. Reward v1 and reward v2 use the same rollout. The config selects the codebook and the 4 s / 12 s limits.

## Video

Replay a saved tape from its snapshot. The default is `ball_train_v2_0070` / `follow_early`, with recovery frames repeated 6 times and the official 12 s nominal hold at 25 fps. `--extra-hold 5` continues the nominal controller for another 5 seconds. This resimulates the initial condition; it does not interpolate poses.

```powershell
python -u scripts/render_video.py
python -u scripts/render_video.py --episode ball_train_v2_0070 --strategy follow_early --extra-hold 5
```

## Model

The policy is `models/k48_policy.py`: width 64, 2 layers, 4 heads, feed-forward 128, 160 position embeddings. Each step reads the 39-D observation and the previous 7-D action. The motion table is `(47, 7)`. The residual is `0.25 * tanh(residual_head)`. Token 47 is outside that table. Checkpoint loading is `models/k48_checkpoint.py`. `training/unified_k48.py` owns the supervised loss and GRPO loop. `training/unified_k47.py` remains the shared codebook and array loader.

## Historical experiments

One-off searches live under `experiments/`. The old `demo/` scripts are in `experiments/legacy/demo/`. Their previous outputs, reports, and raw traces are in `results/archive/demo_results/`. A stored path that still begins with `results/` is resolved to `results/archive/demo_results/` when that historical file exists, so old manifest provenance is left unchanged.

Older K=47 commands in this file's previous revision used `config/training/k47/unified_k47_horizon.yaml` and wrote under what is now `results/archive/demo_results/diagnostics/raw/unified_k47_horizon/`. Those configs still point at that historical tree. Package roles are in `docs/ARCHITECTURE.md`.
