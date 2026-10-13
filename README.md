# CS5478 K48 grasp recovery

A MuJoCo Panda recovers a cylinder after a ball impact. Recovery is a legal 47-token motion codebook plus a bounded residual. Token 47 is HANDOFF: it ends recovery, consumes no physics step, and does not decode a residual. The recovery clock stops at 4 seconds. After handoff, the unchanged nominal controller lifts and must hold the grasp for 12 seconds. That `SUCCESS_12S` outcome is the only recovery success. A 3 mm geometric improvement is not a success gate.

The audited training set is the supervised and GRPO entry. Commands below are the ones implemented by `training/unified_k48.py`.

## Directory layout

| Path | Role |
|---|---|
| `training/` | Dataset preparation, validation, supervised training, GRPO, and policy evaluation |
| `demo/` | Historical experiments, exploratory searches, and visualization scripts |
| `data/` | Training manifests, demonstrations, and initial-condition snapshots |
| `demo_results/` | Historical diagnostics, raw traces, search logs, and the K=47 checkpoint used to initialize K=48 |
| `results/` | New supervised checkpoints, GRPO checkpoints, and policy evaluations |
| `config/` | YAML configs. The active recovery configs are `unified_k48_recovery_v3.yaml` and `unified_k48_recovery_v3_reward_v2.yaml` |
| `docs/HISTORICAL_K32_README.md` | Earlier K=32 write-up |

Training writes under `results/sft/`, `results/grpo/`, and `results/evaluations/`. Historical files stay in `demo_results/`. `data/`, `demo_results/`, and `results/` are gitignored.

## Dataset

| File | Rows |
|---|---:|
| `data/supervised_targets.jsonl` | 251 training demonstrations (244 recovery, 7 nominal-solvable controls with weight 0) |
| `data/valid_demonstrations.jsonl` | 14 validation demonstrations |
| `data/grpo_initial_conditions.jsonl` | 87 confirmed-recoverable training initial conditions |
| `data/nominal_solvable_control.jsonl` | 7 nominal-solvable controls, excluded from the GRPO recovery pool |
| `data/valid_recovery.jsonl` | 8 validation recovery initial conditions |

Train and validation initial conditions do not overlap. All 258 recovery demonstrations replay to `SUCCESS_12S` through the K48 codebook. A recovery tape that ends before 4 seconds supervises token 47 on the observation after its last action. A tape that lands exactly on the 4 second limit does not add an extra HANDOFF target; the environment forces handoff before the next observation. The current `recorded_end` tapes all end before that limit.

Category manifests are `data/recatch/`, `data/controlled_sliding/` (50), and `data/inward_sliding/` (50). One initial condition may have more than one strategy, so category totals exceed the number of distinct initial conditions.

## Setup

From the repository root, with the audited environment (MuJoCo 3.12, PyTorch, NumPy):

```powershell
python -m pip install -r requirements.txt
```

The training entry points import PyTorch in addition to the packages in `requirements.txt`. Use `python -u` so logs flush.

## Dataset commands

Print the rebuilt category counts. This does not generate data:

```powershell
python -u training/rebuild_k48_recovery.py --counts
```

Generation entry point, when new ballistic initial conditions are actually required:

```powershell
python -u training/rebuild_k48_recovery.py --recatch --target-train 200 --target-valid 50 --max-search-trials 400
python -u training/rebuild_k48_recovery.py --controlled-slide --target-train 200 --target-valid 50 --max-search-trials 400
python -u training/rebuild_k48_recovery.py --inward-reseat --target-train 200 --target-valid 50 --max-search-trials 400
```

`--max-search-trials` is a cap, not a request to synthesize that many initial conditions. Static dataset checks:

```powershell
python -u training/audit_pretrain_data.py static
```

`python -u training/verify_handoff_targets.py` replays every recovery demonstration in physics and then checks training batches. It is the full physical check, not a quick import test.

## Training and evaluation

Run these from the repository root. Evaluation is deterministic argmax on CPU. Training uses the `device` in the config.

Check the codebook, the HANDOFF head, one supervised loss, and the reward order. This does not update parameters or write a checkpoint:

```powershell
python -u training/unified_k48.py --check
```

### Supervised training

40 epochs, batch 8, seed 1, learning rate 3e-4. The initialization checkpoint is the historical K=47 epoch-40 model. Checkpoints go to `results/sft/`. The last file is `results/sft/checkpoint_epoch_0040.pt`.

```powershell
python -u training/unified_k48.py --supervised --config config/unified_k48_recovery_v3.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

### GRPO reward v1

Three passes over the 87 training recovery initial conditions, 261 updates. Writes `results/grpo/v1/`. The last file is `results/grpo/v1/checkpoint_u0261.pt`. The success terminal is only `SUCCESS_12S`.

```powershell
python -u training/unified_k48.py --grpo --config config/unified_k48_recovery_v3.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt
```

### GRPO reward v2

`config/unified_k48_recovery_v3_reward_v2.yaml` runs nine passes, 783 updates, and writes `results/grpo/v2/`. A checkpoint saved after update 261 is the end of the first three passes. This command continues that file from update 262 through 783 and appends `training_log.jsonl` and `rollouts.jsonl`. It does not overwrite `checkpoint_u0261.pt`. The frozen KL reference stays `results/sft/checkpoint_epoch_0040.pt`. The last new file is `results/grpo/v2/checkpoint_u0783.pt`.

```powershell
python -u training/unified_k48.py --grpo --config config/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/grpo/v2/checkpoint_u0261.pt
```

### Evaluation

The default IC list is `data/valid_recovery.jsonl` (8 validation recovery initial conditions). The summary is `results/evaluations/heldout_train.json`.

SFT epoch 40:

```powershell
python -u training/unified_k48.py --eval --config config/unified_k48_recovery_v3.yaml --checkpoint results/sft/checkpoint_epoch_0040.pt
```

GRPO v1, same 8 validation initial conditions:

```powershell
python -u training/unified_k48.py --eval --config config/unified_k48_recovery_v3.yaml --checkpoint results/grpo/v1/checkpoint_u0261.pt
```

GRPO v2, same 8 validation initial conditions:

```powershell
python -u training/unified_k48.py --eval --config config/unified_k48_recovery_v3_reward_v2.yaml --checkpoint results/grpo/v2/checkpoint_u0261.pt
```

`--eval-data` selects another recovery list. The 87 training recovery initial conditions are `data/grpo_initial_conditions.jsonl`. Pass `--eval-output` so this does not overwrite `results/evaluations/heldout_train.json`. The rows are written to `closed_loop.json` in that directory.

```powershell
python -u training/unified_k48.py --eval --config config/unified_k48_recovery_v3.yaml --checkpoint results/grpo/v1/checkpoint_u0261.pt --eval-data data/grpo_initial_conditions.jsonl --eval-output results/evaluations/raw/grpo_v1_train_eval
```

Replace the checkpoint path to evaluate another run on the same 87 initial conditions. Reward v1 and reward v2 use the same rollout. The config selects the codebook and the 4 s / 12 s limits.

## Model

The policy is the causal Transformer in `training/action_representation_expansion.py`: width 64, 2 layers, 4 heads, feed-forward 128, 160 position embeddings. Each step reads the 39-D observation and the previous 7-D action. The motion table is `(47, 7)`. The residual is `0.25 * tanh(residual_head)`. Token 47 is outside that table. `training/unified_k48.py` is the K=48 entry point. `training/unified_k47.py` remains the shared codebook and array loader.

## Historical experiments

Scripts that are not on the current dataset, training, or evaluation path are in `demo/`. Their previous outputs, reports, and raw traces are in `demo_results/`. A stored path that still begins with `results/` is resolved to `demo_results/` when that historical file exists, so old manifest provenance is left unchanged.

Older K=47 commands in this file's previous revision used `config/unified_k47_horizon.yaml` and wrote under what is now `demo_results/diagnostics/raw/unified_k47_horizon/`. Those configs still point at that historical tree.
