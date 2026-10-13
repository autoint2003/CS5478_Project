# Learned nominal-grasp restoration

The from-scratch policy completes the full impact_zp6 sequence on the training impact: first recatch at 19.81 mm, second release, hand-+z reposition at 1.97 m/s, second recatch at 2.34 mm, explicit handoff at 3.480 s, then 10 s and 12 s of nominal-only hold. The same schedule does not adapt when the pinch offset changes. Eight of the 14 post-recatch perturbations succeed, and all eight use the same release step and the same reclose step. One of six nearby impacts reaches the basin.

Success means the nominal basin, an explicit recovery-to-nominal handoff, and nominal-only stability for 10 s and for 12 s. A retained grasp that never hands off is a failure.

Basin used for this study, unchanged: orientation 0.117 rad, pinch 3.03 mm, relative speed 0.020 m/s, five-step persistence. Checkpoint: `demo_results/diagnostics/raw/learned_nominal_restoration/ckpt.pt`. The lateral-token checkpoint was not overwritten.

## Action representation

The 29-step success trajectory does not fit in K=33 with a ±0.25 residual.

| phase | steps | action | nearest old token | old L2 | new token |
|---|---|---|---|---|---|
| lateral chase and first reclose | 0–6 | token 32, then closed hold | 32, 0, 29 | 0 | unchanged |
| closed lift | 7–17 | `[0, 0, -0.4, 0, 0, 0, +1]` | 0, residual on a2 clipped from -0.40 to -0.25 | 0.150 | 33 |
| second release | 18–19 | a2 = 0.526, grip = -1 | 15 (`a2 = 1`), residual -0.474 outside the cap | 0.224 | 34 |
| reclose and hold | 20–28 | closed trim | 0 | 0 | unchanged |

Token 15 would reconstruct the second release at 2.85 m/s rather than the demonstrated 2.0 m/s. Two prototypes were appended. Tokens 0–32 and the ±0.25 cap are unchanged.

- token 33 = `[0, 0, -0.4, 0, 0, 0, +1]`
- token 34 = `[-0.065, 0.021, 0.526, 0, 0, 0, -1]`

With K=35 the trajectory reconstruction error is numerical zero. Wrist reseating, inward sliding, and open-finger recatch stay at zero and do not use the new tokens. Five steps in each natural-loss demonstration move onto token 34, so their reconstruction error falls from about 0.18–0.22 to zero. The Transformer width, depth, and heads are unchanged. The logit head is 35-wide.

## Training

Five families, equal draws, 256 suffixes per family per epoch for 120 epochs, trained from scratch on the frozen 39-D causal Transformer:

- wrist reseating
- inward sliding
- open-finger recatch
- natural-loss recatch (three demonstrations)
- the 29-step impact_zp6 restoration only

The restoration family is one trajectory. Seventy-five percent of its suffixes start at the first step, so the lift and the second release are inside the training window. It does not receive more draws than the other families. Loss is 0.0254 at epoch 40, 0.0241 at epoch 80, and 0.0080 at epoch 120. Teacher-forced RMSE on the restoration trajectory is 0.0064.

## Closed-loop impact_zp6

Live geometry entry, v = 6.5 m/s, lateral offset +6 mm.

| event | result |
|---|---|
| first release | 2.920 s |
| lateral chase | yes |
| first recatch | 3.040 s, pinch 19.81 mm, orientation 0.008 rad |
| second release | 3.280 s |
| reposition | hand +z, mean 1.97 m/s, 2 steps, grip open |
| second recatch | pinch 2.34 mm, orientation 0.030 rad |
| nominal basin | yes |
| handoff | 3.480 s |
| 10 s nominal-only | yes |
| 12 s nominal-only | yes |
| pinch / orientation at the end of the 12 s window | 2.79 mm / 0.004 rad |

The video of this rollout is `demo_results/videos/impact_zp6_learned_restoration.mp4`. It runs from the nominal start through the handoff and four further seconds of nominal hold. The overlay shows time, mode, pinch, orientation, token, hand-frame velocity, grip, and the basin flag. At 3.042 s the policy is in recovery on token 33, pinch 19.81 mm, hand vz -0.44 m/s, grip closed. At 3.282 s it is on token 34, grip open, hand vz +1.94 m/s. At 3.482 s the mode is nominal, the token is cleared, pinch is 2.79 mm, and the basin flag is yes.

## State versus timing

The 14 perturbations are applied at the learned policy's own first stable reclose, then the same policy continues. A success still requires the basin, the handoff, and both nominal windows.

| perturbation | open step | reclose step | 12 s nominal |
|---|---|---|---|
| hand x +2 mm | 0 | 1 | no |
| hand x -2 mm | 7 | 8 | no |
| hand y +2 mm | 1 | 2 | no |
| hand y -2 mm | none | none | no |
| hand z +2 mm | 11 | 13 | no |
| hand z -2 mm | 12 | 14 | no |
| hand-x velocity +0.03 m/s | 11 | 13 | yes |
| hand-x velocity -0.03 m/s | 11 | 13 | yes |
| hand-z velocity +0.03 m/s | 11 | 13 | yes |
| hand-z velocity -0.03 m/s | 11 | 13 | yes |
| angular x +0.2 rad/s | 11 | 13 | yes |
| angular x -0.2 rad/s | 11 | 13 | yes |
| angular z +0.2 rad/s | 11 | 13 | yes |
| angular z -0.2 rad/s | 11 | 13 | yes |

Eight of 14 succeed. Every success opens on step 11 and recloses on step 13, which is a 0.22 s wait and a 0.04 s translation. Velocity and angular perturbations do not change release timing, translation duration, or reclose timing. All six position perturbations fail. Four of them move the release earlier or suppress it. The hand-z +2 mm case keeps the successful timing and still ends at 5.43 mm, outside the basin. The policy does not lengthen the translation when the pinch offset grows.

## Regression

| case | first recatch | pinch after first recatch | second release / regrasp | final pinch | basin | handoff | 10 s | 12 s |
|---|---|---|---|---|---|---|---|---|
| impact_zp6 | yes | 19.81 mm | yes | 2.79 mm | yes | 3.480 s | yes | yes |
| train_dz13.5 | yes | 1.57 mm | no | 1.59 mm | yes | 4.876 s | yes | yes |
| nl_dvz_p | yes | 1.55 mm | no | 1.58 mm | yes | 4.896 s | yes | yes |
| nl_dz14.6 | yes | 2.38 mm | no | 2.43 mm | yes | 4.916 s | yes | yes |
| dz13.2 | yes | 1.42 mm | no | 1.44 mm | yes | 4.876 s | yes | yes |
| dvz_m | yes | 1.58 mm | no | 1.61 mm | yes | 4.896 s | yes | yes |
| impact_zm6 | yes | 12.57 mm | no | 12.42 mm | no | no | no | no |
| impact_z3 | yes | 12.35 mm | no | 12.23 mm | no | no | no | no |
| air_lat_pos | yes | 6.10 mm | yes | 2.21 mm | no | no | no | no |
| air_fast_pos | yes | 5.82 mm | yes | 2.72 mm | no | no | no | no |

The five natural-loss rows already sit inside the pinch basin at the first recatch. The policy does not open again, and all five hand off and hold for 12 s. Teacher-forced RMSE on the three natural-loss demonstrations is 0.280, with a peak action error near 2.09. That imitation error does not remove the closed-loop skill.

impact_zm6 and impact_z3 recatch with about 12.4 mm of pinch and 1.62 rad of hand orientation. They do not release again and they do not hand off. Wrist reseating, inward sliding, and open-finger recatch were demonstrated on this impact. Their teacher-forced RMSE is 0.020, 0.031, and 0.009. The closed-loop impact_zm6 rollout is the physical regression of those three skills, and it does not meet the nominal-basin rule. Those scripted skills also failed that rule before this training.

Both air rollouts do execute a second release. Orientation at the second reclose is about 0.17 rad, outside the basin, and the object then falls. They were already failures under this success rule. The new failure mode is a drop after the extra release.

## Generalization

These impacts are full online episodes. None is the saved training trajectory. Training impact is +6 mm at 6.5 m/s.

| case | impact | first recatch | pinch after first recatch | second release / regrasp | pinch after second recatch | basin | handoff | 10 s | 12 s |
|---|---|---|---|---|---|---|---|---|---|
| near_z4 | +4 mm, 6.5 m/s | yes | 20.72 mm | yes, 2.03 m/s, 2 steps | 2.44 mm | yes | 3.520 s | yes | yes |
| near_z5 | +5 mm, 6.5 m/s | yes | 20.91 mm | yes, 1.93 m/s, 2 steps | 3.80 mm | no | no | no | no |
| near_z7 | +7 mm, 6.5 m/s | yes | 21.90 mm | yes, 2.01 m/s, 2 steps | 3.23 mm | no | no | no | no |
| near_z8 | +8 mm, 6.5 m/s | yes | 21.36 mm | yes, 2.00 m/s, 2 steps | 3.21 mm | no | no | no | no |
| near_v6 | +6 mm, 6.0 m/s | yes | 22.37 mm | no | — | no | no | no | no |
| near_v7 | +6 mm, 7.0 m/s | no | — | no | — | no | no | no | no |

near_z4 is a real neighboring impact and it completes the skill. near_z5, near_z7, and near_z8 fire the same second release at 3.300 s, with the same two-step hand-+z move, and reclose between 3.21 mm and 3.80 mm. That is outside the 3.03 mm basin, so there is no handoff. near_v6 recatches and then falls without a second release. near_v7 never recatches and falls. The release-reposition-regrasp pattern appears on nearby lateral offsets at the training speed, with a fixed duration, and that duration lands in the basin for only one of those offsets.

POLICY MEMORIZES THE RESTORATION SEQUENCE WITHOUT ROBUST GENERALIZATION
