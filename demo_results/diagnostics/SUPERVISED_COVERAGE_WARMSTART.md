# Supervised coverage warm-start

One new airborne-chase token, then a from-scratch 120-epoch supervised fit. No SAC, DPO, or GRPO update. Final test was not rolled out, searched, or used for selection. Impact privileged search was not continued.

Checkpoint: `demo_results/diagnostics/raw/supervised_coverage_warmstart/ckpt.pt`. It is the warm-start result. It is not a formal GRPO initialization, because retained natural-loss and slip successes were lost.

## K=35 to K=36

The two out-of-range steps are the same physical mode: grip open, hand-z command near 0.45, and a lateral `vy` correction. Their difference is 0.084 in L2, almost all of it in `vy` (0.079).

| trace | step | action (vx, vy, vz, wx, wy, wz, grip) | nearest K=35 token | residual |
|---|---:|---|---:|---|
| air_a | 1 | −0.1054, 0.5183, 0.4528, 0, 0, 0, −1 | 34 | −0.0407, +0.4974, −0.0735, 0, 0, 0, 0 |
| air_b | 1 | −0.0765, 0.5970, 0.4464, 0, 0, 0, −1 | 34 | −0.0119, +0.5761, −0.0799, 0, 0, 0, 0 |

Token 34 is the existing second-release prototype. Every other step of both traces already fit K=35 ±0.25.

The new prototype is the mean of those two legal actions:

`token 35 = [−0.0910, 0.5576, 0.4496, 0, 0, 0, −1]`

Residuals against it are ±0.014, ±0.039, and ±0.003. Both sit inside ±0.25. Tokens 0–34 are the K=35 rows, bit for bit. The residual cap is still ±0.25. Old centers were not moved. No old demonstration step is nearer to token 35 than to its previous token.

| trace | max L2 under K=35 | max L2 under K=36 | clipped components under K=36 | steps assigned to token 35 |
|---|---:|---:|---:|---:|
| wrist, slide, open-finger, three natural-loss demos, success_trajectory | 0 | 0 | 0 | 0 |
| demo_impact_zp6 | 0.0365 | 0.0365 | 102, unchanged from K=35 | 0 |
| success_air_a | 0.247 | 0 | 0 | 1 |
| success_air_b | 0.326 | 0 | 0 | 1 |
| success_imp_c | 0 | 0 | 0 | 0 |

The 102 clipped components on `demo_impact_zp6` were already present at K=35. Token 35 does not add any. Air_a, air_b, and imp_c reconstruct with no clipping. The codebook is frozen at K=36. No further tokens were added.

## Demonstration families

Equal draws per family (182–188 suffixes per epoch). Air_a and air_b share one airborne family. Imp_c is its own impact example, not a claim that every train impact is recoverable. Restoration and the two new short families start at the first step on 75% of draws, the same prefix rate the restoration trainer already used.

| family | trajectories |
|---|---|
| wrist | demo_wrist_reseat |
| sliding | demo_gravity_inward |
| open-finger recatch | demo_recatch_open20 |
| natural-loss recatch | demo_train_dz13.5, demo_nl_dz14.6, demo_nl_dvz_p |
| restoration | success_trajectory, demo_impact_zp6 |
| airborne | success_air_a, success_air_b |
| impact example | success_imp_c |

Training is the established causal Transformer (39-D, d=64, 2 layers, 4 heads, previous-command history, token plus ±0.25 tanh residual), from scratch, 120 epochs, Adam 1e-3. Loss was 0.051 at epoch 40, 0.038 at epoch 80, and 0.020 at epoch 120.

## Closed-loop airborne

Deterministic argmax from the saved live entries. Token 35 is selected once on each success. Both open at the first recovery step, chase for two steps, and reclose.

| episode | entry pinch | entry speed | open step | chase steps | close step | basin | handoff | 10 s | 12 s | token 35 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|---:|
| air_a | 6.012 mm | 0.0003 m/s | 0 | 2 | 2 | yes | 6.436 s | yes | yes | 1 |
| air_b | 6.009 mm | 0.0005 m/s | 0 | 2 | 2 | yes | 5.716 s | yes | yes | 1 |

## Local airborne coverage

Four train-side pulses, not the validation or final-test airborne cases. Each entry is still about 6.0–6.1 mm pinch. The policy repeats the same two-step open and one use of token 35, then holds.

| case | pulse | entry pinch | entry speed | open | chase | close | basin | handoff | 10 s | 12 s |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| near_a_slow | +0.20 m/s, 0.10 s | 6.013 mm | 0.0003 | 0 | 2 | 2 | yes | 5.496 s | yes | yes |
| near_a_fast | +0.30 m/s, 0.08 s | 6.013 mm | 0.0003 | 0 | 2 | 2 | yes | 7.476 s | yes | yes |
| near_b_slow | −0.45 m/s, 0.06 s | 6.010 mm | 0.0006 | 0 | 2 | 2 | yes | 6.016 s | yes | yes |
| near_b_short | −0.55 m/s, 0.04 s | 6.085 mm | 0.0023 | 0 | 2 | 3 | yes | 4.816 s | yes | yes |

`air_lat_pos` and `air_fast_pos`, which fell under the previous supervised policy, now also finish `SUCCESS_12S` and each selects token 35 once.

## Impact

| episode | result | token 35 | open / chase |
|---|---|---:|---|
| imp_a | drop during recovery | 0 | 0 / 2 |
| imp_b | basin not reached | 0 | 12 / 1 |
| imp_c | basin not reached | 0 | 0 / 2 |
| imp_d | drop during recovery | 0 | no open |

`imp_c` is in the supervised set and still does not hand off. The previous frozen policy had one `SUCCESS_12S` in four stochastic samples on this capture. That sample is gone: the new group is 0/4, all basin-not-reached. The other three train impacts stay failures. `impact_zp6`, the retained restoration, still reaches the basin, hands off at 3.480 s, and holds 10 s and 12 s.

## Old-skill regression

Final success rule: basin, explicit handoff, 10 s nominal-only, 12 s diagnostic.

| case | previous supervised policy | this warm-start |
|---|---|---|
| impact_zp6 | SUCCESS_12S | SUCCESS_12S |
| train_dz13.5 | SUCCESS_12S | SUCCESS_12S |
| nl_dz14.6 | SUCCESS_12S | drop during recovery |
| nl_dvz_p | SUCCESS_12S | drop during recovery |
| impact_zm6 (wrist / slide / open-finger scene) | basin not reached | drop during recovery |
| air_lat_pos | fell | SUCCESS_12S |
| air_fast_pos | fell | SUCCESS_12S |
| slip_a | SUCCESS_12S under the frozen K=35 policy | SUCCESS_12S |
| slip_b | SUCCESS_12S under the frozen K=35 policy | drop during recovery |
| slip_c | drop | drop |
| slip_d | SUCCESS_12S under the frozen K=35 policy | drop during recovery |

Wrist, sliding, and open-finger remain in the training set and still reconstruct exactly. Their shared source scene, `impact_zm6`, was already outside the basin under the previous supervised policy. Deterministic train slip success went from 3/4 to 1/4. Two of the three natural-loss closed loops that previously held for 12 s now drop.

## Train results and GRPO signal

Group of 4, same seeds as the coverage audit (`2000 + 17*g + sum of character codes`), no learning update. A group is useful when outcomes are mixed or the return span is at least 0.5.

| episode | family | deterministic | success / 4 | return span | mixed | useful | token 35 counts |
|---|---|---|---:|---:|---|---|---|
| imp_a | impact | drop | 0/4 | 0.189 | no | no | 0, 1, 0, 0 |
| imp_b | impact | basin not reached | 0/4 | 0.188 | no | no | 1, 0, 0, 0 |
| imp_c | impact | basin not reached | 0/4 | 0.023 | no | no | 0, 0, 0, 0 |
| imp_d | impact | drop | 0/4 | 0.142 | no | no | 0, 0, 0, 0 |
| slip_a | slip | SUCCESS_12S | 1/4 | 0.989 | yes | yes | 0, 0, 0, 0 |
| slip_b | slip | drop | 1/4 | 0.980 | yes | yes | 0, 0, 0, 0 |
| slip_c | slip | drop | 0/4 | 0.086 | no | no | 0, 0, 0, 0 |
| slip_d | slip | drop | 0/4 | 0.240 | no | no | 0, 0, 0, 0 |
| seat_a | mis-seat | drop | 1/4 | 0.993 | yes | yes | 0, 0, 0, 0 |
| seat_b | mis-seat | drop | 0/4 | 0.053 | no | no | 1, 1, 0, 2 |
| air_a | airborne | SUCCESS_12S | 3/4 | 0.925 | yes | yes | 1, 1, 1, 1 |
| air_b | airborne | SUCCESS_12S | 4/4 | 0.013 | no | no | 1, 1, 1, 1 |

Useful groups: `slip_a`, `slip_b`, `seat_a`, `air_a`. Airborne is no longer 0/2 all-failure. `air_a` is a mixed group, and every `air_b` sample succeeds, so that group has almost no relative advantage. Impact useful coverage did not grow. The previous `imp_c` mixed group disappeared.

By family, useful groups are slip 2/4, mis-seat 1/2, impact 0/4, airborne 1/2.

## Why formal GRPO does not start

Airborne behavior is in the policy: both train entries, four nearby pulses, and the two legacy diagnostic airborne cases hand off and hold, and token 35 is used in closed loop. Slip and airborne both supply group signal.

The same fit drops `nl_dz14.6` and `nl_dvz_p`, which previously held for 12 s, and it drops the deterministic recoveries of `slip_b` and `slip_d`. `slip_d` also loses its mixed group. Starting GRPO from this checkpoint would be training on top of that regression.

SUPERVISED WARM-START CAUSES SKILL INTERFERENCE
