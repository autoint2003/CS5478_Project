# Final 39-D balanced policy

The active policy reads a 39-D observation and the previous 7-D command. The observation is `observe_airborne` (`obs_gt_airborne_39`): indices 0–38 of the former 45-D vector. Indices 39–44, the wide-scale copy of the hand twist, are removed. The prefix and the object channels keep their previous scales, frames, and clipping.

| indices | group |
|---|---|
| 0–26 | tactile and proprioceptive prefix |
| 27–29 | object relative position, hand frame, scale 0.25 m |
| 30–32 | object relative linear velocity, scale 2 m/s |
| 33–35 | object relative orientation as a rotation vector, scale π |
| 36–38 | object relative angular velocity, scale 8 rad/s |

The previous command is concatenated by the Transformer. It is not one of the 39. Object pose and twist are exact MuJoCo body state, a noise-free proxy for an RGB-D tracker. No camera image is read.

## Checkpoint and training

Checkpoint: `results/diagnostics/raw/balanced_unified_recovery/ckpt_39d.pt`.

The archived 45-D weights stay at `results/diagnostics/raw/balanced_unified_recovery/ckpt.pt`. `load_airborne_policy` refuses that file because its input layer is 64×52 rather than 64×46. The 39-D weights are trained from scratch on the 39-D vector. They are not a slice of the 45-D network.

Training matches the balanced recipe and the A1 ablation: regenerated demonstrations (18 natural-loss windows, one wrist, one sliding, one open-finger recatch, no airborne-carry demonstration), equal batch weight 0.25 on the four families that are present, K=32 prototypes from the archived checkpoint, causal Transformer (d=64, 2 layers, 4 heads, feedforward 128), recovery7d_airborne_v3, the same tracking gains, torch seed 1, 120 epochs, 80 steps per epoch, batch 16, Adam at 1e-3, gradient clip 1.0, CUDA. Final training loss 0.0572. Wall time 242 s. Raw log: `results/diagnostics/raw/balanced_39d/summary.json`.

## Clean 24-case result

Success is retention through absolute t = 12. The run scores **15/24**.

| family | t=12 |
|---|---:|
| impact | 3/4 |
| sliding | 4/4 |
| wrist | 4/4 |
| open-finger recatch | 4/4 |
| airborne carry | 0/4 |
| natural-loss | 0/4 |

Retained cases: `impact_zm6`, `impact_phi90`, `impact_z3`, `slide_suffix_20`, `slide_suffix_36`, `slide_dy1`, `slide_dz1`, `wrist_suffix_3`, `wrist_suffix_6`, `wrist_suffix_9`, `wrist_suffix_12`, `recatch_suffix_8`, `recatch_suffix_16`, `recatch_suffix_24`, `recatch_suffix_28`.

`impact_zp6` is ESCAPE. The four airborne-carry cases and the four held-out natural-loss cases are DROP.

Every impact, sliding, wrist, and open-finger case that the 45-D balanced policy retained is retained here. The regression check lost none of those 15 cases.

## Moderate sensor profile

The profile perturbs only the observation: 2 mm object-position noise, 1 deg orientation noise, 0.03 m/s linear-velocity noise, 0.3 rad/s angular-velocity noise, 40 ms delay on channels 27–38, 50 ms causal smoothing, 0.5 mm center-of-pressure noise, and 1 N force noise. The simulator state is unchanged.

| seed | t=12 | impact | sliding | wrist | recatch | airborne | natural |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15/24 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 |
| 1 | 16/24 | 3/4 | 4/4 | 4/4 | 4/4 | 1/4 | 0/4 |
| 2 | 16/24 | 3/4 | 4/4 | 4/4 | 4/4 | 1/4 | 0/4 |

Seeds 1 and 2 add `air_lat_pos`. Contact is lost during the policy window, and the t = 12 label is the grasp after nominal continuation. Impact, sliding, wrist, and open-finger recatch stay at 3/4, 4/4, 4/4, and 4/4 on every seed. This profile does not reduce the clean benchmark.

## Comparison with the 45-D policy

The published 45-D balanced policy also scored 15/24, with the same family counts and the same retained cases. Under this sensor profile it scored 15, 15, and 15. The 39-D policy scores 15, 16, and 16. The 30-D compact candidate from the ablation, which kept only the prefix and object angular velocity, lost `impact_phi90` and scored 13, 14, and 13 under the same profile, so it is not the active observation.

## Reproducibility check

A second process loaded `ckpt_39d.pt`, refused `ckpt.pt`, built the seated impact world, read a live observation of shape `(39,)`, and confirmed the input layer is `(64, 46)`, which is the 39-D observation plus the 7-D previous command. A closed-loop rollout of `wrist_suffix_3` retained the object through t = 12.

## Limitations

Airborne carry is 0/4 on the clean benchmark, and held-out natural-loss is 0/4. The object channels are simulator state, not a measured RGB-D pose. Center-of-pressure and lateral-estimate noise are more sensitive than this combined profile: the 45-D sweep lost impact and one recatch case at 0.5–1 mm on those tactile channels, and those channels are still in the 39-D prefix. Delay, smoothing, and millimetre-scale pose error in the tested profile did not move the score.

FINAL 39-D BALANCED POLICY VERIFIED
