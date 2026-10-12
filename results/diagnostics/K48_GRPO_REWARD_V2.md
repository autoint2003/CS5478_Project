# K=48 GRPO reward v2

Reward v2 replaces the GRPO rollout return only when `reward_version: v2` is set. `config/unified_k48.yaml` and `config/unified_k48_long.yaml` stay on v1, whose return is still `1{SUCCESS_12S} - 0.05 * recovery_seconds`. The new pilot config is `config/unified_k48_reward_v2.yaml`. It writes to `results/diagnostics/raw/unified_k48/grpo_pilot_reward_v2/` and does not overwrite `grpo_pilot/`.

The recovery limit remains 4 s. Nominal verification remains 12 s after handoff. Token 47 is still an immediate handoff with no residual. Motion tokens 0–46 still use the 7-D action plus residual. No training run was started for this change.

## Return

- `SUCCESS_12S`: `R = 2.0 - 0.05 * recovery_seconds`
- `DROP_DURING_RECOVERY`: `R = -1.0 + 0.10 * q_recovery`
- `DROP_AFTER_HANDOFF`, `LOST_GRASP`, `NOMINAL_LIFT_NOT_COMPLETED`, and `PHYSICAL_VERIFICATION_FAILED`: `R = -0.5 + 0.20 * q_nominal - 0.05 * recovery_seconds`
- `INVALID`: no return. The rollout is omitted from the group advantage.

`q_recovery = clip(recovery_elapsed / recovery_limit, 0, 1)` and is added only for a terminal recovery-stage loss. A brief release during recovery is not a terminal loss: the existing search rule still requires escape, `z < 0.30`, or 50 contact-free policy steps below 0.44 m.

`q_nominal` is the fraction of the 12 s nominal window during which both fingers are in contact after the nominal lift command has reached its target. Steps before that event, steps with only one finger in contact, and steps after the first drop, escape, or sustained loss of bilateral contact are not counted. If that lift event never occurs, `q_nominal` is 0. The integral is taken from the same bilateral-contact check and `fsm.p_des[2] >= z_tgt` check already used to decide lift completion. No new position, orientation, or contact-count threshold was added.

At the 4 s limit the bounds are:

| Outcome | Return |
| --- | --- |
| Slowest success | 1.80 |
| Best after-handoff failure (`q_nominal = 1`, no recovery time) | -0.30 |
| Worst after-handoff failure (`q_nominal = 0`, 4 s) | -0.70 |
| Latest recovery drop | -0.90 |
| Immediate recovery drop | -1.00 |

Every success outranks every failure. Every recovery-stage drop scores below every after-handoff failure.

## Tests

`training/test_reward_v2.py` checked early drop (-0.995), late drop (-0.90), immediate handoff failure (-0.50), delayed nominal failure (-0.50 for 3 s held and 1 s of recovery), successful early handoff (1.98), successful forced handoff (1.80), and an invalid rollout with no return. A premature handoff that then fails nominally scores -0.502, which is a failure return. A 10-step bilateral gap after lift still counts only the bilateral steps and does not end the hold. A drop on the lift step yields `q_nominal = 0`.

Group advantages were computed for four distinct failures, four identical failures, and a group containing one invalid rollout. The mixed failure group is not skipped. The identical failure group skips the surrogate. The invalid member is excluded and gets advantage 0. Selecting v1 still returns the historical scalar.

## Saved traces

The original 20-update pilot stored `checkpoint_u0005.pt` through `checkpoint_u0020.pt`. Each file contains `proto`, `model`, `res_cap`, `handoff`, and `update`. It does not contain rollout states, contact traces, or the historical returns, so those trajectories cannot be rescored.

The 56-IC physical evaluation at `results/diagnostics/raw/horizon_migration/checkpoint_eval.json` was read and left unchanged. It has `reason`, `recovery_elapsed`, `nominal_elapsed`, `lift_completed`, and one end-of-episode contact sample. It does not have `held_after_lift_s`, `q_nominal`, `t_lift`, or `t_grasp_failure`. Those four fields are required for `q_nominal`. `nominal_elapsed` includes time before the lift and, on a lost grasp, time after bilateral contact has already ended. The final contact sample does not reconstruct the hold. The rescored file is `results/diagnostics/raw/unified_k48/reward_v2/saved_trace_comparison.json`.

Under the 4 s v2 limit, the 26 successes score 1.88 to 1.972, against historical scores of 0.88 to 0.972. The 27 recovery drops score -0.998 to -0.957, against historical scores of -0.086 to -0.004. The one `LOST_GRASP` and two `DROP_AFTER_HANDOFF` rows stay unscored. Their v2 values can only be bounded, from -0.62 to -0.42 for the lost grasp and inside -0.62 to -0.342 across the two post-handoff drops.

New GRPO rollouts append `rollouts.jsonl` with the outcome, `q_recovery`, `q_nominal`, recovery duration, reward components, historical v1 return, and the return used for the update.

## Command

```powershell
python -u training/unified_k48.py --grpo --config config/unified_k48_reward_v2.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k48.py --grpo --config config/unified_k48_reward_v2.yaml --checkpoint results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```
