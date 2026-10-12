# K=47 independent recovery and nominal horizons

The recovery controller and the nominal lift now have separate clocks. A missed 3 mm basin no longer ends an episode and no longer blocks the nominal controller. Historical manifests, trajectories, and `checkpoint_epoch_0040.pt` were not overwritten. No optimizer step was taken on the migrated data.

## Before and after

Control period is 0.020 s (`config/recovery_success.yaml`). The policy recovery limit stays 120 steps, which is 2.4 s.

| Clock | Meaning |
| --- | --- |
| Simulator time | `sim.data.time`. It is not reset. |
| Recovery elapsed | `sim.data.time - t_recovery_start` |
| Handoff time | `sim.data.time` at `apply_handoff` |
| Nominal elapsed | `sim.data.time - handoff_sim_time` |
| Total simulated time | Absolute `sim.data.time` at the end. It includes time already on the clock when the snapshot was restored. |

`ball_train_v2_0021 / release_long` is the timing trace. The snapshot restores with qpos error 0. The saved 46 commands run to their recorded end. Handoff is at simulator time 4.140 s. Nominal verification ends at simulator time 16.140 s. Nominal elapsed is 12.000 s. Recovery elapsed is 0.920 s. The lift target is reached and the reason is `SUCCESS_12S`. A forced handoff after only the first 10 commands of that same episode drops on the next nominal step (`DROP_AFTER_HANDOFF`, nominal elapsed 0.002 s). That is a grasp that has not been recovered, not a clock error.

### Previous semantics

- The 7-D action has no handoff channel. Early exit was the persistent bilateral basin: both finger contacts, orientation error at most 0.117 rad, pinch at most 3.03 mm, relative speed at most 0.020 m/s, grip command above 0.5, held for 0.100 s (5 control steps).
- Reaching that basin was required before `apply_handoff`. Missing it at the 120-step cap produced `BASIN_NOT_REACHED` and skipped nominal verification.
- `nominal_validate` already measured 12 s from the handoff instant. A separate legacy flag asked whether absolute simulator time had passed 12 s. That flag was not the success label.
- `apply_handoff` copies the saved grasp orientation into `r_des`, moves `p_des` to the current hand, and zeroes the commanded velocities. It does not write `qpos` or `qvel`.
- `label_of` in `training/recovery_success.py` is unchanged, so older scripts still emit the basin labels.

### Corrected semantics

Implemented in `training/horizon_outcome.py`.

- Recovery ends on a physical failure, on the optional basin early-exit, or at 2.4 s.
- The basin check is only that optional early exit. It is not a success requirement. No handoff dimension was added.
- At 2.4 s, if the object has not already failed, nominal handoff is forced.
- Historical replays follow the saved command list to its recorded end, including the one 122-step episode. They are not cut at 120.
- Nominal verification then runs the unchanged lift. The endpoint is `nominal_elapsed >= 12.0`, not `sim.data.time >= 12.0`.
- `SUCCESS_12S` requires the lift target while both fingers contact the object, and bilateral contact through the 12 s window.
- Pose error, relative speed, world height, and a fixed contact count before handoff are diagnostics.

Physical failure during recovery uses the legal-search rule, not an instant crossing of 0.44 m: object height below 0.30 m, escape beyond 0.12 m in the horizontal plane, or 50 consecutive control steps (1.0 s) with no finger contact while height is below 0.44 m. A short release that is recaught is not a drop.

After handoff, height below the table or below 0.44 m, or an escape, is `DROP_AFTER_HANDOFF`. Loss of bilateral contact for 75 physics steps (0.15 s) is `LOST_GRASP`. A grasped 12 s window whose lift target was never reached is `NOMINAL_LIFT_NOT_COMPLETED`. A simulator exception is `INVALID` and is not given a physical outcome. A recovery timeout by itself is not a failure.

## Code

- `training/horizon_outcome.py`: recovery limit, forced and early handoff, nominal verification, physical reward.
- `training/unified_k47.py`: closed-loop evaluation and GRPO rollouts call that outcome. Invalid rollouts are omitted from the advantage group. `--check` still validates the frozen 288-row manifest and also the migrated 296-row manifest.
- `training/migrate_horizon_dataset.py`: replay, view build, and checkpoint re-evaluation.
- `config/unified_k47_horizon.yaml` and `config/unified_k47_horizon_long.yaml`: point at the migrated manifests and at new output directories.
- `training/recovery_success.py` `label_of` was not changed.
- The codebook `codebook_selected_k47.npy` was not rewritten. Residual bound remains ±0.25.

`lambda_time`, `nominal_seconds`, and `recovery_limit_s` in the horizon yaml repeat the constants in `horizon_outcome.py` (0.05, 12.0, 2.4). If a config sets one of those keys to a different number, loading it raises. The trainer does not override the constants.

`config/unified_k47.yaml` still points at the historical manifests and at `results/diagnostics/raw/unified_k47/eval`. Do not run `--eval` with that file. It would overwrite `heldout_train.json`.

## Migrated dataset

Directory: `results/diagnostics/raw/horizon_migration/`. Source files under `legal_recovery_v2/` were read and not rewritten.

Each of the 359 saved recoveries was restored from its TRAIN snapshot and replayed with the saved commands.

| View | Rows | Result |
| --- | --- | --- |
| Exact continuous commands | 359 | 359 `SUCCESS_12S` |
| K=47 decoded commands inside ±0.25 | 296 | 296 `SUCCESS_12S` |
| Not representable | 63 | `UNRESOLVED` |
| Supervised demonstrations | 296 | 153 distinct TRAIN initial conditions |
| GRPO initial conditions | 312 | all TRAIN snapshots, including unsolved ones |
| Unsuccessful searches without an action array | 1345 | unresolved |

Historical decoded label to new decoded label:

| Historical label | New label | Count |
| --- | --- | --- |
| `SUCCESS_12S` | `SUCCESS_12S` | 288 |
| `BASIN_NOT_REACHED` | `SUCCESS_12S` | 6 |
| `DROP_AFTER_HANDOFF` | `SUCCESS_12S` | 2 |
| not representable | `UNRESOLVED` | 63 |

Previously accepted demonstrations that now fail: none.

Newly accepted demonstrations, all eight former in-range replay failures:

| Id | Strategy | Old decoded label |
| --- | --- | --- |
| `ball_train_v2_0016` | `chase_lost` | `DROP_AFTER_HANDOFF` |
| `ball_train_v2_0020` | `chase_lost` | `BASIN_NOT_REACHED` |
| `ball_train_v2_0035` | `chase_lost` | `BASIN_NOT_REACHED` |
| `ball_train_v2_0068` | `release_long` | `BASIN_NOT_REACHED` |
| `ball_train_v2_0087` | `chase_lost` | `DROP_AFTER_HANDOFF` |
| `ball_train_v2_0235` | `release_long` | `BASIN_NOT_REACHED` |
| `ball_train_v2_0250` | `release_long` | `BASIN_NOT_REACHED` |
| `ball_train_v2_0272` | `chase_lost` | `BASIN_NOT_REACHED` |

`ball_train_v2_0016` and `ball_train_v2_0068` are the two newly covered initial conditions. The other six add a strategy on an initial condition that already had a demonstration. Covered initial conditions go from 151 to 153.

The 63 not-representable trajectories have at least one command whose nearest prototype is outside ±0.25. Their exact continuous commands replay as `SUCCESS_12S`, but a clipped token sequence is not that command list, so they are not supervised targets. Three clipped replays (`ball_train_v2_0073 / slide_50`, `ball_train_v2_0145 / slide_50`, `ball_train_v2_0161 / chase_lost`) dropped after handoff. Those clipped outcomes are recorded as `clipped_replay_reason` and are not labels of the legal actions.

`ball_train_v2_0177 / slide_45` is the 122-step episode. It was replayed in full and is `SUCCESS_12S`. `ball_train_v2_0021 / release_long` remains one 46-step episode with three loss intervals.

Every migrated supervised success hands off at `recorded_end`, because historical replay does not take the basin early-exit. Recovery length on those 296 rows is 26 to 122 steps (0.520 s to 2.440 s), median 0.580 s. Multi-stage rows (`none_intervals >= 2`): 61.

Mechanism counts and sample weights. Each group's weights sum to 1.

| Mechanism | Old count | New count | Weight |
| --- | --- | --- | --- |
| `contact_preserving_slide` | 61 | 61 | 1/61 |
| `airborne_chase` | 77 | 81 | 1/81 |
| `release_regrasp` | 92 | 93 | 1/93 |
| `sliding` | 45 | 45 | 1/45 |
| `repeated_recatch` | 13 | 16 | 1/16 |

The 1345 unresolved searches have labels only: 662 `DROP_DURING_RECOVERY`, 521 `BASIN_NOT_REACHED`, 162 `DROP_AFTER_HANDOFF`. Their action arrays were never saved, so they were not relabeled.

GRPO keeps all 312 TRAIN snapshots. None is a validation or final-test path. An initial condition is not removed because it lacks a demonstration.

## Reward

```
R = 1{SUCCESS_12S} - 0.05 * recovery_elapsed_seconds
```

The 12 s nominal window is not penalized. At the 2.4 s limit the time penalty is 0.12, which is smaller than the success term, so the slowest success (0.88) still beats every failure. A 0.4 s success scores 0.98. A 0.2 s failure scores -0.01. Earlier success is preferred to a longer success. A forced handoff is scored by the nominal outcome. Handing off before the grasp is recovered scores the resulting drop, not a success. Identical non-invalid returns get zero advantage and skip the surrogate. `INVALID` is omitted from the advantage group. The coefficient was not tuned on FINAL_TEST.

There is no separate intermediate shaping term. The policy log-probability covers recovery steps only.

## Epoch-40 checkpoint, same 56 TRAIN initial conditions

Weights were not modified. The old file `results/diagnostics/raw/unified_k47/eval/heldout_train.json` was not modified. The new rows are `results/diagnostics/raw/horizon_migration/checkpoint_eval.json`.

The old evaluator stopped recovery only for height below 0.30 m, required the basin before nominal control, and called a 120-step miss `BASIN_NOT_REACHED`. The new evaluator uses the search fall rule and always runs nominal verification after a non-failed recovery. The two success rates are not the same measurement.

| Definition | Success | Other outcomes |
| --- | --- | --- |
| Old basin gate | 6/56 `SUCCESS_12S` | 35 `BASIN_NOT_REACHED`, 14 `DROP_DURING_RECOVERY`, 1 `DROP_AFTER_HANDOFF` |
| New physical horizon | 26/56 `SUCCESS_12S` | 27 `DROP_DURING_RECOVERY`, 2 `DROP_AFTER_HANDOFF`, 1 `LOST_GRASP` |

All 6 old successes remain `SUCCESS_12S`, and all 6 take the basin early-exit. All 14 old drops during recovery remain drops. The old drop after handoff remains a drop after handoff.

Of the 35 old `BASIN_NOT_REACHED` cases:

| New reason | Count | What happened |
| --- | --- | --- |
| `SUCCESS_12S` | 20 | Forced handoff at 120 steps, lift completed, grasp held 12 nominal seconds |
| `DROP_DURING_RECOVERY` | 13 | Both fingers lost the object below 0.44 m for 50 control steps before the limit. Height at the stop is about 0.42 m. |
| `LOST_GRASP` | 1 | `ball_train_v2_0075`: forced handoff, lift reached, bilateral contact then lost at 1.186 nominal seconds |
| `DROP_AFTER_HANDOFF` | 1 | `ball_train_v2_0235`: forced handoff at 2.4 s, object dropped on the next nominal step |

The 20 new successes from that group are `ball_train_v2_0005`, `0025`, `0030`, `0045`, `0055`, `0080`, `0090`, `0095`, `0120`, `0130`, `0135`, `0190`, `0200`, `0210`, `0215`, `0220`, `0225`, `0255`, `0275`, and `0280`.

On all 26 successes, nominal elapsed is 12.000 s and the lift completed. Twenty of them are forced handoffs at 2.400 s. The six early handoffs are the six old successes.

## Why 13 of the old basin misses are now recovery losses

The old diagnosis and the migrated evaluator use the same epoch-40 checkpoint, the same TRAIN snapshots, empty history, a zero previous action, and argmax inference in eval mode. On one device, the executed 7-D actions match until the new rule stops.

The old loop stopped a recovery only for height below 0.30 m, or for the basin early-exit. None of these 13 episodes do either, so that loop always runs 120 steps and the old label is `BASIN_NOT_REACHED`. The migrated rule also stops after 50 control steps with no finger contact while the object is below 0.44 m. That is the legal-search fall rule. It is not a basin early-exit.

On CUDA, which is the device the diagnosis used when a GPU is present, 12 of the 13 hit that 50-step loss. Continuing the same commands through step 120 never restores bilateral contact. Their height stays near 0.42 m. The old label was a missed basin on an object that had already left the fingers. The new label is the corrected classification of that same trajectory.

`ball_train_v2_0015` is the exception. CPU and CUDA commands first differ at step 24, by 1.6e-4. On CPU the 50-step rule fires at step 77. On CUDA it never fires, and step 120 still has contacts on both fingers. The published 13-loss count comes from the CPU re-evaluation. The same checkpoint on CUDA does not call 0015 a recovery-stage loss. The codebook, physics, and task rule were not changed to hide that split.

The 296 demonstration labels come from replaying saved commands, not from this policy. That set is unchanged by the CPU/CUDA split.

## Checks

No optimizer step, no checkpoint write, and no GRPO update.

| Check | Result |
| --- | --- |
| Snapshot restore on `0021` | qpos error 0 |
| Handoff does not edit `qpos` or simulator time | enforced in `nominal_verify` |
| Full `0021` replay | handoff 4.140 s, nominal end 16.140 s, `SUCCESS_12S` |
| Forced handoff after 10 commands of `0021` | `DROP_AFTER_HANDOFF` |
| Continuous replay of 359 saved episodes | 359 `SUCCESS_12S` |
| Decoded replay inside the residual bound | 296 `SUCCESS_12S` |
| `0177 / slide_45` | 122 steps kept, `SUCCESS_12S` |
| `0021` supervised row | one 46-step episode |
| Mechanism weights | each group sums to 1 |
| GRPO manifest | 312 TRAIN rows, no validation or final-test path |
| Supervised loss on an untrained K=47 model | finite token and residual terms, then gradients cleared |
| `python -u training/unified_k47.py --check` | passed; reports frozen 288 and migrated 296 |
| Advantage on identical returns | surrogate skipped |
| Advantage with an invalid member | that return stays out of the group |
| Epoch-40 weights and old eval file | timestamps unchanged |

## Remaining blockers

- The epoch-40 weights were fit to the old 288-row basin objective. They have not been trained on the 296-row set.
- On the 56 held-out TRAIN initial conditions, 27 episodes still lose the object during recovery under the search fall rule.
- Sixty-three continuous successes cannot be represented inside ±0.25 and are not demonstrations.
- 1345 searches have no saved action array and stay unresolved.
- Early exit is still the old basin persistence, because the 7-D action has no handoff channel.

## Commands

From the repository root. These start training. They were not run for this migration.

Supervised initialization, 40 epochs:

```powershell
python -u training/unified_k47.py --supervised --config config/unified_k47_horizon.yaml
```

```bash
python -u training/unified_k47.py --supervised --config config/unified_k47_horizon.yaml
```

Closed-loop evaluation of a checkpoint produced by that run:

```powershell
python -u training/unified_k47.py --eval --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --eval --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

GRPO pilot, 20 updates, only after that supervised checkpoint exists. These 20 updates sample 20 of the 312 initial conditions:

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

Longer GRPO run, 312 updates, only after the pilot has been inspected. Each update uses one TRAIN snapshot, so this run visits all 312:

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon_long.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47_horizon_long.yaml --checkpoint results/diagnostics/raw/unified_k47_horizon/supervised/checkpoint_epoch_0040.pt
```

Non-training check:

```powershell
python -u training/unified_k47.py --check
```

```bash
python -u training/unified_k47.py --check
```

RECOVERY AND NOMINAL HORIZONS SEPARATED: YES
FORCED HANDOFF IMPLEMENTED: YES
PHYSICAL SUCCESS DEFINITION UPDATED: YES
HISTORICAL DATA PRESERVED: YES
SUPERVISED DATASET MIGRATED: YES
GRPO DATASET MIGRATED: YES
EXISTING CHECKPOINT RE-EVALUATED: YES
TRAINING PIPELINE READY: YES
POLICY TRAINING EXECUTED: NO
