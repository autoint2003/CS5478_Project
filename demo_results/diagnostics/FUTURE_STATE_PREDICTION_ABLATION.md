# Future-state prediction ablation

Both arms start from the same K=35 continual initialization, with token 35 appended and unused. The deployed policy is unchanged: 39-D observation, previous command, causal Transformer, K=36, residual cap ±0.25, token plus residual. The auxiliary head reads `h_t` during training only. Its prediction is not an input to the action head. No intention labels, skill IDs, new tokens, new demonstrations, diffusion, SAC, DPO, or GRPO. `FINAL_TEST` was not rolled out.

## Targets

The 15-D target is taken from the active 39-D observation. Indices were checked against `OBS_NAMES_AIR` before training:

| channel | index |
|---|---|
| hand-frame object position `p_rel_h` | 27, 28, 29 |
| relative orientation `rot_rel` | 33, 34, 35 |
| relative linear velocity `v_rel_h` | 30, 31, 32 |
| relative angular velocity `w_rel_h` | 36, 37, 38 |
| aperture | 12 |
| contact L / R | 2, 3 |

`p_rel_h` is the object position in the hand frame already stored in the observation (`Rh.T @ (po - ph)`). Demonstrations do not store a separate nominal-pinch offset, so the target is that observation channel, not a privileged pinch error. Continuous channels are predicted as `(x - mean) / std`. Contact channels use binary cross-entropy. `p_rel_hy` and aperture have raw standard deviations 0.001 and 0.016; the standard deviation used in the loss is floored at 0.02 so a near-constant channel cannot dominate.

Horizons use the policy step of 0.02 s: 5, 10, 20, and 40 steps (0.10, 0.20, 0.40, 0.80 s). A horizon past the end of a demonstration is masked. The demonstrations are not the same length. Natural-loss traces are 8 steps, so only +0.10 s exists, and only near the start. Airborne traces are 9 and 10 steps, so +0.40 s and +0.80 s do not exist there. Those longer horizons come from the sliding trace (52 steps) and `demo_impact_zp6` (300 steps).

## Loss scale

On one initialization batch the action loss was 4.11 and the normalized future loss was 1.25 (ratio 0.31). The requested weights 0, 0.1, 0.5, and 1.0 were used with no extra multiplier. A zero-initialized future head sends no gradient into the Transformer, so an encoder-gradient ratio measured at that initialization is not a usable scale.

At `lambda_future = 0` the future term is multiplied by zero, so the auxiliary head stays at zero and its error is the error of predicting each channel's dataset mean. That error does not move across epochs.

## Training

Family-balanced batches, Adam at 1e-3, batch 16, 80 batches per epoch, the existing 4× action weights, and the full network trainable. Checkpoints at 40, 80, 120, 180, and 240 epochs. The union was lower at 240 than at 120, so training stopped at 240. Seed 0 ran all four weights. Seeds 1 and 2 repeated `lambda = 0` and the best future weight, `lambda = 0.1`.

Closed-loop success is the nominal basin, an explicit handoff, and the 10 s nominal-only hold, with the 12 s diagnostic included in `SUCCESS_12S`. Counts are slip out of 3 (`slip_a`, `slip_b`, `slip_d`), natural-loss out of 3, impact out of 2 (`impact_zp6`, `imp_c`), and airborne out of 8. The wrist / sliding / open-finger scene `impact_zm6` was already `BASIN_NOT_REACHED` on the initialization and is not part of the union.

Initialization, before any of this training: slip 3, natural-loss 3, impact 1 (`impact_zp6`), airborne 0, union 7.

## Seed 0

Future-state error is normalized RMSE at 0.10 / 0.20 / 0.40 / 0.80 s. Token-timing regression means the closed-loop token-34 run is shorter than the initialization's 3 steps.

| model | lambda | epoch | train token accuracy | future-state error | slip | natural-loss | impact | airborne | union | token-timing regressions |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| baseline | 0 | 40 | 0.989 | 1.72 / 1.34 / 1.41 / 0.57 | 1 | 0 | 0 | 7 | 8 | slip and natural release run 2 |
| baseline | 0 | 80 | 0.989 | 1.72 / 1.34 / 1.41 / 0.57 | 0 | 2 | 1 | 7 | 10 | slip and natural release run 1–2 |
| baseline | 0 | 120 | 0.993 | 1.72 / 1.34 / 1.41 / 0.57 | 2 | 0 | 1 | 8 | 11 | slip and natural release run 2 |
| baseline | 0 | 180 | 0.993 | 1.72 / 1.34 / 1.41 / 0.57 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 2 |
| baseline | 0 | 240 | 0.993 | 1.72 / 1.34 / 1.41 / 0.57 | 1 | 0 | 0 | 8 | 9 | slip and natural release run 2 |
| future | 0.1 | 40 | 0.989 | 0.50 / 0.39 / 0.59 / 0.05 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 2 |
| future | 0.1 | 80 | 0.989 | 0.46 / 0.34 / 0.43 / 0.05 | 1 | 0 | 1 | 7 | 9 | slip and natural release run 2 |
| future | 0.1 | 120 | 0.948 | 0.40 / 0.46 / 0.44 / 0.06 | 2 | 2 | 1 | 7 | 12 | slip and natural release run 2 |
| future | 0.1 | 180 | 0.993 | 0.25 / 0.22 / 0.33 / 0.04 | 0 | 0 | 1 | 8 | 9 | slip and natural release run 1–2 |
| future | 0.1 | 240 | 0.993 | 0.27 / 0.25 / 0.30 / 0.03 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 1 |
| future | 0.5 | 40 | 0.989 | 0.32 / 0.30 / 0.42 / 0.04 | 0 | 0 | 0 | 8 | 8 | slip and natural release run 1 |
| future | 0.5 | 80 | 0.989 | 0.24 / 0.22 / 0.36 / 0.04 | 1 | 0 | 1 | 7 | 9 | slip and natural release run 1–2 |
| future | 0.5 | 120 | 0.993 | 0.22 / 0.21 / 0.31 / 0.07 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 1 |
| future | 0.5 | 180 | 0.993 | 0.13 / 0.12 / 0.26 / 0.03 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 1 |
| future | 0.5 | 240 | 0.993 | 0.17 / 0.23 / 0.27 / 0.04 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 1 |
| future | 1.0 | 40 | 0.983 | 0.23 / 0.21 / 0.35 / 0.03 | 0 | 0 | 1 | 7 | 8 | slip and natural release run 2 |
| future | 1.0 | 80 | 0.989 | 0.18 / 0.15 / 0.29 / 0.05 | 0 | 0 | 1 | 7 | 8 | slip_a and nl_dvz_p release run 2 |
| future | 1.0 | 120 | 0.993 | 0.23 / 0.17 / 0.29 / 0.02 | 0 | 0 | 1 | 7 | 8 | slip and natural release run 2 |
| future | 1.0 | 180 | 0.993 | 0.13 / 0.12 / 0.25 / 0.03 | 0 | 0 | 0 | 7 | 7 | slip and natural release run 2 |
| future | 1.0 | 240 | 0.993 | 0.09 / 0.10 / 0.24 / 0.03 | 0 | 0 | 1 | 7 | 8 | slip and natural release run 1–2 |

Action loss is already below 0.03 by epoch 40 in every arm. Raising `lambda` lowers future error and does not lower action loss further. Weights 0.5 and 1.0 keep airborne and drop slip and natural-loss for the whole schedule.

## Learning curves

Old-skill count is slip + natural-loss + impact. Values are the mean of three seeds for `lambda = 0` and `lambda = 0.1`.

| epoch | old, lambda 0 | airborne, lambda 0 | union, lambda 0 | old, lambda 0.1 | airborne, lambda 0.1 | union, lambda 0.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1.67 | 7.00 | 8.67 | 2.33 | 7.00 | 9.33 |
| 80 | 3.00 | 7.00 | 10.00 | 1.33 | 7.00 | 8.33 |
| 120 | 3.00 | 8.00 | 11.00 | 2.00 | 7.00 | 9.00 |
| 180 | 1.00 | 7.00 | 8.00 | 1.00 | 7.33 | 8.33 |
| 240 | 1.67 | 7.67 | 9.33 | 2.33 | 7.00 | 9.33 |

The union peaks by epoch 120 and is lower at epoch 240. Extra epochs do not remove the tradeoff. At the matched epoch-120 point, next-action training has the higher mean union (11.0 versus 9.0).

## Seeds

Best checkpoint of each seed, chosen by union on that seed:

| model | lambda | seed | epoch | slip | natural-loss | impact | airborne | union |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 120 | 2 | 0 | 1 | 8 | 11 |
| baseline | 0 | 1 | 80 | 2 | 2 | 1 | 7 | 12 |
| baseline | 0 | 2 | 120 | 2 | 2 | 0 | 8 | 12 |
| future | 0.1 | 0 | 120 | 2 | 2 | 1 | 7 | 12 |
| future | 0.1 | 1 | 40 | 2 | 2 | 2 | 7 | 13 |
| future | 0.1 | 2 | 240 | 2 | 2 | 0 | 7 | 11 |

Mean best union is 11.7 for next-action training and 12.0 for future prediction. Seed 2 goes the other way (12 down to 11). No seed keeps slip 3, natural-loss 3, impact, and airborne together. `nl_dz14.6` fails at every one of these six checkpoints. `slip_d` fails on five of them.

## Token timing

The initialization's successful slip and natural-loss rollouts hold token 34 for 3 steps, then token 0. Example, `slip_a`: `[34, 34, 34, 0, 34, 33, ...]`. `nl_dz14.6`: `[34, 34, 34, 0, 34, 34, 0, 0, 0, 33, ...]`.

The saved natural-loss demonstrations are only 8 steps and already use a 2-step release: `[34, 34, 0, 9, 0, 33, 33, 33]`. Every trained checkpoint copies that demonstration. On `slip_a` and `slip_b` the 2-step run still often succeeds. On `nl_dz14.6` the same run drops. Future prediction does not restore the third release step, because that step is not a supervised target.

Airborne demonstrations and the trained rollouts agree: token 34 once, token 35 once, close at step 2 (`[34, 35, 0, ...]`). `air_a` and `air_b` succeed under both losses. At `lambda = 0.1`, epoch 240, seed 0, slip rollouts switch to that airborne sequence (`[34, 35, 0, ...]`) and drop. A larger future weight makes this substitution earlier.

`imp_c` on the initialization is `[32, 32, 0, 33, ...]` and drops. Some trained checkpoints succeed with `[32, 32, 17, ...]`, which is the `success_imp_c` demonstration, not a change produced by the future head.

## What the future head actually predicts

Teacher-forced, `lambda = 1`, epoch 240. Position is hand-frame object position in millimetres. The `lambda = 0` head predicts the same mean on every trace: position about `[-0.6, 0.1, 111.7]` mm, aperture 17.8 mm, contact probability 0.5.

| trace | step | horizon | true position mm | predicted position mm | true v m/s | predicted v m/s | true aperture mm | predicted aperture mm | true contact | predicted contact |
|---|---:|---:|---|---|---|---|---:|---:|---|---|
| nl_dz14.6 | 0 | 0.10 s | [-0.2, -0.4, 100.9] | [-0.1, -0.4, 102.2] | [0.00, 0.07, 0.00] | [0.01, 0.07, 0.00] | 15.9 | 15.9 | 1, 1 | 1.00, 1.00 |
| air_a | 0 | 0.10 s | [0.7, 0.2, 97.5] | [1.2, 0.1, 97.0] | [0.00, 0.02, 0.00] | [0.00, 0.02, 0.00] | 16.1 | 16.1 | 1, 1 | 1.00, 1.00 |
| air_a | 4 | 0.10 s | [0.7, 0.0, 97.5] | [0.7, 0.1, 96.6] | [0.00, 0.01, 0.00] | [0.00, 0.01, 0.00] | 17.8 | 17.8 | 1, 1 | 1.00, 1.00 |

On these short traces the trained head follows position within about 1 mm, relative velocity, aperture, and contact at +0.10 s. The natural-loss and airborne demonstrations do not contain +0.40 s or +0.80 s labels, so this check cannot say the head tracks those maneuvers over the longer horizons. The low 0.80 s RMSE in the main table comes from the long sliding and impact traces.

## Reading

Teacher-forced token accuracy stays at about 0.95–0.99 for every family, including airborne and natural-loss, while action loss is already small at epoch 40. The model can fit the action sequences it is shown. Closed loop still loses `nl_dz14.6` and shortens the slip release from the initialization's 3 steps to the demonstration's 2 steps. That is a coverage limit: the live state after two release steps is not the state on which the demonstration's next token was labeled.

The future head does make `h_t` informative. Normalized future error falls from the mean-predictor values near 1.7 / 1.3 / 1.4 / 0.6 to about 0.1–0.4 once `lambda > 0`, and the physical check above matches the short-horizon hand-object state. That change does not move the union. Across seeds the best future checkpoint is within one success of the best next-action checkpoint, and one seed is worse. Heavier future weights make slip select the airborne open/chase token and destroy the retained release.

FUTURE-STATE PREDICTION IMPROVES REPRESENTATION BUT CLOSED-LOOP COVERAGE REMAINS DATA-LIMITED
