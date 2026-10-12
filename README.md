# CS5478 unified recovery

A MuJoCo Panda recovers a cylinder after a genuine ball impact. The nominal grasp controller runs until the cylinder leaves the grasp. A causal Transformer then commands a legal 7-D recovery action for at most 2.4 s (120 control steps of 20 ms). The 7-D action has no handoff channel. The persistent bilateral basin remains only an optional early exit. If that exit does not occur, nominal handoff is forced at the recovery limit unless the object has already dropped or escaped. A missed 3 mm basin is not a failure and does not block the nominal controller.

After handoff the unchanged nominal lift runs for 12 s of nominal time, `sim.data.time - handoff_time`. The simulation clock is not reset. Success means the lift target is reached while both fingers still hold the cylinder, and that grasp lasts through the 12 s window. Diagnostic pose errors are recorded and are not success gates.

The earlier K=32 write-up is in `docs/HISTORICAL_K32_README.md`. The basin-gated 288-demonstration set remains in `results/diagnostics/raw/legal_recovery_v2/` and is not overwritten. The migrated labels are in `results/diagnostics/raw/horizon_migration/`.

## Architecture

The policy is the causal Transformer in `training/action_representation_expansion.py` (`build_policy`): width 64, 2 layers, 4 heads, feed-forward 128, 160 position embeddings. Each step reads the 39-D `observe_airborne` vector and the previous executed 7-D action. History is not cleared at release, recontact, a repeated recatch, or the handoff.

The action is `recovery7d_airborne_v3`: hand-frame velocity, wrist rate, and gripper command. It is a discrete prototype plus a continuous residual. The selected table has 47 prototypes, shape `(47, 7)`. Tokens 0–35 are the original K=36 table. Tokens 36–46 are eleven appended recovery centers. The residual is `0.25 * tanh(residual_head)`, so it stays inside ±0.25, and the executed command is clipped to ±1. The same decode is used for supervised targets, GRPO rollouts, evaluation, and the one-step physics check.

`training/formal_grpo.py` is the historical K=36 launcher. It still loads the K=36 `FuturePolicy` checkpoint and rejects a 36-wide token head on a 47-token model. K=47 training goes through `training/unified_k47.py` and `step_legal`, which does not park the ball.

## Dataset

| Split | What it contains | Used for gradients |
| --- | --- | --- |
| TRAIN | 312 ball-impact initial conditions: 28 v1 and 284 v2 | yes |
| VALIDATION | sealed v1 validation episodes | no |
| FINAL_TEST | sealed v1 final-test episodes | no |

A TRAIN initial condition is a restored MuJoCo snapshot at recovery entry (hand-frame position error above 6 mm for 40 ms). It is not itself a demonstration. Privileged search produced 359 continuous recoveries. Under the old basin gate, 288 decoded K=47 replays were called successes. Those files are unchanged. The migrated supervised set is rebuilt from decoded replays that still succeed after the independent 12 s nominal window. Counts are in `results/diagnostics/K47_INDEPENDENT_HORIZON_DATASET_MIGRATION.md`. `ball_train_v2_0021 / release_long` remains one 46-step episode.

All 312 TRAIN snapshots stay in the GRPO set, including initial conditions with no demonstration. The return is `1{physical success} - 0.05 * recovery_seconds`. The 12 s nominal window is not penalized. A recovery timeout is a forced handoff, not a zero-success label by itself.

## Training pipeline

The epoch-40 checkpoint in `results/diagnostics/raw/unified_k47/supervised/` was trained on the old 288-row basin-gated set. It has been re-evaluated under the new definition and has not been trained again. Counts, the 56-IC comparison, and the reward are in `results/diagnostics/K47_INDEPENDENT_HORIZON_DATASET_MIGRATION.md`.

1. Supervised initialization on the 296 migrated demonstrations (`config/unified_k47_horizon.yaml`).
2. Closed-loop evaluation on the same held-out TRAIN snapshots (`ball_train_v2` ids whose number ends in 0 or 5). The policy acts by itself. Recovery lasts at most 2.4 s, then nominal verification lasts 12 s. This does not touch FINAL_TEST. Write the result under `results/diagnostics/raw/unified_k47_horizon/eval/`, not into the historical eval file.
3. A 20-update GRPO pilot from that new supervised checkpoint, sampling all 312 TRAIN snapshots. The return uses physical success minus `0.05` per recovery second.
4. A 312-update GRPO run only after that pilot has been inspected. One update samples one TRAIN initial condition, so 312 updates visit each of the 312 snapshots once.

GRPO advantages are group-relative. Identical non-invalid returns get zero advantage and skip the surrogate. An invalid simulation is omitted. A forced handoff is scored by the nominal outcome. The KL term compares the two 47-way categorical distributions.

## Setup

```powershell
python -m pip install -r requirements.txt
```

```bash
python -m pip install -r requirements.txt
```

The validation environment used MuJoCo 3.12.0. Commands below are run from the repository root. Use `python -u` so the logs flush.

## Commands

Check data, the codebook, one forward pass, one supervised loss, and the GRPO log-probability. This does not update any parameter:

```powershell
python -u training/unified_k47.py --check
```

```bash
python -u training/unified_k47.py --check
```

Supervised initialization (40 epochs, batch 8, seed 1, learning rate 3e-4, checkpoint every epoch):

```powershell
python -u training/unified_k47.py --supervised --config config/unified_k47_horizon.yaml
```

```bash
python -u training/unified_k47.py --supervised --config config/unified_k47_horizon.yaml
```

Closed-loop evaluation. Use the horizon config so the historical `heldout_train.json` is not overwritten. After a new supervised run, pass that run's checkpoint instead of the epoch-40 file:

```powershell
python -u training/unified_k47.py --eval --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --eval --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

GRPO pilot (20 updates, group 4, learning rate 3e-5, KL coefficient 0.05, clip 0.2, seed 1). Start it from the supervised checkpoint produced by the horizon config:

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

Longer GRPO run, only after the pilot is inspected. 312 updates visit every TRAIN initial condition once. Pass the same supervised checkpoint so the KL reference stays at that policy:

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon_long.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon_long.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

## Paths

- Codebook: `results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy`
- Historical supervised rows: `results/diagnostics/raw/legal_recovery_v2/supervised_targets.jsonl` (288)
- Migrated supervised rows: `results/diagnostics/raw/horizon_migration/supervised_targets.jsonl` (296)
- Migrated GRPO starts: `results/diagnostics/raw/horizon_migration/grpo_initial_conditions.jsonl` (312)
- Horizon pilot config: `config/unified_k47_horizon.yaml`
- Horizon longer-run config: `config/unified_k47_horizon_long.yaml`
- Historical checkpoints: `results/diagnostics/raw/unified_k47/supervised/`
- Horizon outputs: `results/diagnostics/raw/unified_k47_horizon/`

## K=48 handoff expansion

The motion codebook stays 47 tokens. Token 47 is HANDOFF and has no residual. The recovery limit used for that set is 4 s, chosen because it is the shortest tested limit that contains the 122-step demonstration and a 0.20 s pause. Nominal verification is still 12 s from the handoff. Counts and the pause command are in `results/diagnostics/K48_HANDOFF_DATASET_EXPANSION.md`. No K=48 policy has been trained.

```powershell
python -u training/unified_k48.py --check
python -u training/unified_k48.py --supervised --config config/unified_k48.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --eval --config config/unified_k48.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_long.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_reward_v2.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_long_reward_v1.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_long_reward_v2.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k48.py --check
python -u training/unified_k48.py --supervised --config config/unified_k48.yaml --checkpoint results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --eval --config config/unified_k48.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_long.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_reward_v2.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_long_reward_v1.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
python -u training/unified_k48.py --grpo --config config/unified_k48_long_reward_v2.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

## Current status and limitations

The migrated supervised set has 296 demonstrations on 153 TRAIN initial conditions. All 312 TRAIN snapshots remain in GRPO. The K=47 table is unchanged. `python -u training/unified_k47.py --check` still checks the frozen 288-row manifest and also the migrated 296-row manifest. It does not take an optimizer step.

The epoch-40 checkpoint was trained under the old basin gate. On the same 56 TRAIN initial conditions its old score is 6/56 and its new physical score is 26/56. Those definitions differ. Sixty-three continuous recoveries stay outside the ±0.25 residual and are not demonstrations. No training run has been started on the migrated set.
