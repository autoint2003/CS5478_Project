# K=47 supervised closed-loop diagnosis

Checkpoint: `demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt`. Evaluation file: `demo_results/diagnostics/raw/unified_k47/eval/heldout_train.json`. No parameters were updated. Physics, success labels, and the codebook were not changed.

The evaluation file contains 56 v2 initial conditions and these labels: 6 `SUCCESS_12S`, 35 `BASIN_NOT_REACHED`, 14 `DROP_DURING_RECOVERY`, 1 `DROP_AFTER_HANDOFF`. The 35 basin misses are exactly the 35 rows marked `truncated: true`. They hit the 120-step recovery cap without a drop and without a nominal handoff. A recount of the saved file is 35 basin misses and 14 drops, rather than 34 and 15.

## 1. Evaluation set

Every evaluated id is `ball_train_v2_*` with a numeric suffix divisible by 5. `training/release_coverage_v3.py` fits the eleven appended centers on the complementary ids only. All 56 were therefore excluded from codebook fitting. They were not held out of supervised training. `train_supervised` reads every row of `supervised_targets.jsonl`. Fifty-five of those 288 rows, covering 30 of the 56 ids, are in the training set. The other 26 ids have no verified K=47 demonstration.

Autonomous outcome by whether a verified demonstration exists:

| | ICs | SUCCESS_12S | BASIN_NOT_REACHED | DROP_DURING_RECOVERY | DROP_AFTER_HANDOFF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Has a verified K=47 demonstration | 30 | 6 | 23 | 0 | 1 |
| No verified K=47 demonstration | 26 | 0 | 12 | 14 | 0 |

Of the 26 without a verified demonstration, 23 have no successful continuous search at all. Three (`0060`, `0210`, `0280`) have continuous holds that the K=47 replay marks `NOT_REPRESENTABLE`. None of those 26 succeeded autonomously.

The 55 verified demonstrations on the 30 covered ids were replayed from the same snapshots through `replay` / `step_legal`. All 55 again finish `SUCCESS_12S`, matching the stored `k47_replay_label`. The demonstrations still execute. The autonomous policy does not follow them.

Restoring any of these snapshots leaves an empty observation history and a zero previous action. The live 39-D observation matches the first saved demonstration observation with error 0. The closed-loop input at step 0 is the training input. There is no history-alignment bug and no pre-recovery context the trainer never showed.

## 2. Supervised objective

The training loss printed each epoch is the mean, over the trajectories in a batch, of

`sample_weight * (token cross-entropy + residual MSE)`.

`sample_weight` is `1 / count(mechanism)`, about 0.011 to 0.077. On the finished checkpoint the same expression over all 288 trajectories is 0.0038, consistent with the reported epoch-40 value 0.003919. The unweighted quantities on those same trajectories are larger:

| Quantity | Value |
| --- | ---: |
| Token top-1, all 11,171 steps | 0.937 |
| Mean per-trajectory cross-entropy | 0.221 |
| Mean per-trajectory residual MSE | 0.0016 |
| Mean per-step residual L2 | 0.085 |
| Mean per-step executed-action L2 | 0.123 |
| Mean per-step executed-action Linf | 0.111 |
| Executed-action Linf when the token is correct | 0.069 |
| Executed-action Linf when the token is wrong | 0.750 |
| Step-0 token accuracy, 288 trajectories | 0.590 |
| Step-0 token accuracy, 55 eval-id trajectories | 0.618 |
| No-contact steps (both contact channels ≤ 0.5) | 0.891 |
| Steps where the target token changes | 0.703 |
| Steps where the target grip sign changes | 0.424 |
| Grip-sign agreement of the executed command | 0.982 |

Decoding the stored token plus the stored residual reproduces the saved 7-D action with Linf 0. Targets are aligned with the demonstrations. Inference takes the argmax token and the network residual `0.25 * tanh(head)`, then clips the sum to ±1. That matches `decode_action` on every training trajectory.

Top-1 by mechanism, micro-averaged over steps:

| Mechanism | Trajectories | Top-1 | Cross-entropy | Transition top-1 | Grip-switch top-1 | No-contact top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contact_preserving_slide | 61 | 0.968 | 0.079 | 0.141 | 0.143 | 1.000 |
| sliding | 45 | 0.967 | 0.080 | 0.382 | 0.340 | 1.000 |
| repeated_recatch | 13 | 0.914 | 0.307 | 0.765 | 0.649 | 0.902 |
| release_regrasp | 92 | 0.908 | 0.285 | 0.756 | 0.554 | 0.890 |
| airborne_chase | 77 | 0.897 | 0.323 | 0.719 | 0.442 | 0.879 |

The eval-id trajectories score 0.941 top-1 and the remaining trajectories score 0.936. Holding those ids out of center fitting did not make them harder to imitate under teacher forcing.

Only 21 of the 47 tokens appear as targets. Predicted mass sits on the same tokens: token 12 (4,757 predicted / 4,738 target), 36 (2,801 / 2,664), 33 (1,785 / 1,853), then 39, 34, 0, 14, 37. Tokens 36–46 receive 3,680 predictions against 3,558 targets. Eight rare target tokens are never selected (9, 17, 28, 31, 35, 42, 44, 46), together 48 steps. The head is not ignoring the appended centers as a group.

The loss is small because of the mechanism weight, not because token accuracy is near perfect. Long runs of a repeated token make step-average accuracy 94% while the first command is right on only 59% of trajectories and grip-switch steps are right on 42%.

## 3. Representative autonomous trajectories

Traces are in `demo_results/diagnostics/raw/unified_k47/diagnosis/representative_traces.json`. The basin threshold used by the evaluator is bilateral contact, orientation error ≤ 0.117 rad, position error ≤ 3.03 mm, relative speed ≤ 0.020 m/s, and gripper command > 0.5, held for the usual persistence window.

`ball_train_v2_0020`, `SUCCESS_12S`, one verified `release_long` demonstration. The policy opens for 21 steps (tokens 36, 38, 34, 37), closes at step 21, makes bilateral contact at step 22, and hands off at step 29. Minimum position error 0.66 mm. Two contact phases: none, then both. The first autonomous token already disagrees with the demonstration at step 7, and the executed action differs by more than 0.05 at step 0. This one still reaches the basin.

`ball_train_v2_0005`, `BASIN_NOT_REACHED`, verified `chase_lost` and `release_long`. Opens until step 20, then closes and stays in bilateral contact for the remaining 99 steps. Position error never falls below 6.2 mm, and after the close it stays at least 9.7 mm. Orientation and relative speed do reach the basin limits. The hand keeps the cylinder and moves it to about z = 0.79, outside the nominal position. This is a missed basin after a recatch, not a drop and not a failure to close.

`ball_train_v2_0145`, `DROP_AFTER_HANDOFF`, verified `slide_45`. Opens for 34 steps, closes, bilateral contact at step 35, hands off at step 42 with position error down to 0.34 mm. The nominal continuation then drops. The recovery entered the basin. The failure is the physical hold after handoff. Forcing the demonstration token on this same id, with the predicted residual, does not hand off at all (`ep` minimum 4.1 mm).

`ball_train_v2_0010`, `DROP_DURING_RECOVERY`, no demonstration and no continuous hold. Ten steps, gripper stays at −1, no finger contact, relative speed already 0.93 m/s, object z falls through 0.30. The policy repeats the open token 39. This is an uncovered initial condition, not a mistimed recatch.

## 4. Teacher forcing versus autonomous control

On the 30 demonstrated ids the three comparisons are:

1. Exact saved-action replay: 55/55 `SUCCESS_12S`.
2. Teacher-forced network predictions on the saved observation and previous-action history: token top-1 0.941 on these 55 trajectories, but the decoded action differs from the saved action by mean Linf 0.107. Step 0 is the same observation the autonomous rollout sees.
3. Autonomous rollout: 6/30 `SUCCESS_12S`.

The first disagreement is at the first command, before the state has left the demonstration. For 28 of the 30 ids at least one saved demonstration already has executed-action Linf above 0.05 at step 0. Two demonstrations (`0025/release_long` and `0200/release_long`) match the teacher token at every step and still differ in the command at step 0, because the residual on the correct prototype is nonzero. Both autonomous rollouts are `BASIN_NOT_REACHED`. `0125/release_long` also matches every teacher token and does succeed, so a residual error of this size is not fatal on every id.

A further rollout on the 24 demonstrated ids that the autonomous policy missed used the demonstration token and the network residual. Four became `SUCCESS_12S` (`0055`, `0120`, `0190`, `0220`). The other 20 stayed `BASIN_NOT_REACHED`, including `0025`, `0045`, `0080`, `0140`, `0200`, and `0275`, whose teacher-forced token accuracy on the chosen demonstration is 1.0. They close the gripper and make bilateral contact, and the position error remains above 3 mm (`0025` minimum 6.4 mm, `0045` 27 mm, `0080` 22 mm). Replacing the token with the teacher's token does not restore the demonstration. The residual does.

This is not exposure bias in the usual sense. The history at step 0 is the expert history. The action distribution at that state is already off the saved command, and the contact then amplifies a residual of about 0.07 in Linf.

## 5. Cause and the next experiment

The failures are a mixture, with a clear order:

- On the 26 ids with no verified demonstration, the policy has nothing local to imitate. All 14 drops are in this group. Coverage is the cause there.
- On the 30 demonstrated ids, target alignment and teacher-forced token accuracy are not the limiting defect. Exact replay still succeeds. The first executed command already leaves the saved action, and for the perfect-token cases the residual alone is sufficient to miss the 3 mm basin after recontact. Four of 24 misses are recovered by substituting the teacher token, so token errors are real but secondary.
- One demonstrated id, `0145`, does enter the basin and then drops under the nominal hold.
- The reported loss near 0.004 is the mechanism-weighted sum. It is consistent with cross-entropy 0.22 and should not be read as a 99.6% action match.

Do not start GRPO from this checkpoint yet. GRPO on the 26 uncovered ids would be searching without a local demonstration, while the 23 demonstrated basin misses are still an imitation error on ids the supervised loss already trained on.

The smallest next experiment is a residual-tolerance sweep on one perfect-token failure, `ball_train_v2_0025 / release_long`, with no parameter update. Replay the saved action scaled as

`prototype[teacher_token] + stored_residual + alpha * (predicted_residual - stored_residual)`

for a few values of `alpha` in `[0, 1]`. `alpha = 0` is the demonstration and succeeds. `alpha = 1` is the teacher-token rollout above and misses by 6.4 mm. The sweep measures how large a residual error this recovery can absorb. If the tolerable `alpha` is far below 1, the following training change is a loss on the executed 7-D action, with extra weight on step 0 and on grip-switch steps, and then the same 56-id evaluation. If a large `alpha` still succeeds, the miss is the closed-loop use of teacher tokens after the state has drifted, and the next run should be a short dataset-aggregation pass on these 30 ids rather than GRPO.

## Per-IC evidence

Ids are `ball_train_v2_XXXX`. “In training” means at least one verified K=47 demonstration was in the 288-row supervised file. Continuous search is listed as `strategy:K47 replay label`. Teacher-forced columns are `strategy correct/steps`, the first autonomous step whose token disagrees with that demonstration (`None` means no disagreement over the overlap), and the first step whose executed action differs by Linf > 0.05. The last column is the closed-loop result when the demonstration token is forced and the network residual is kept. It was run only for autonomous failures that have a demonstration. All 55 verified demonstrations replayed `SUCCESS_12S` again.

| ID | Autonomous | In training | Demos | Mechanisms | Continuous search | Teacher-forced match | Teacher token + predicted residual |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| 0005 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 28/29 tok 8 act 3; release_long 28/29 tok 8 act 4 | BASIN_NOT_REACHED |
| 0010 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0015 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 24/28 tok 0 act 0; release_long 24/28 tok 0 act 0 | BASIN_NOT_REACHED |
| 0020 | SUCCESS_12S | yes | 1 | release_regrasp | chase_lost:BASIN_NOT_REACHED;release_long:SUCCESS_12S | release_long 29/30 tok 7 act 0 | - |
| 0025 | BASIN_NOT_REACHED | yes | 1 | release_regrasp | chase_lost:NOT_REPRESENTABLE;release_long:SUCCESS_12S | release_long 28/28 tok None act 0 | BASIN_NOT_REACHED |
| 0030 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0035 | BASIN_NOT_REACHED | yes | 1 | release_regrasp | chase_lost:BASIN_NOT_REACHED;release_long:SUCCESS_12S | release_long 24/28 tok 0 act 0 | BASIN_NOT_REACHED |
| 0040 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0045 | BASIN_NOT_REACHED | yes | 2 | sliding | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S | slide_50 28/28 tok 21 act 0; slide_45 108/110 tok 33 act 0 | BASIN_NOT_REACHED |
| 0050 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0055 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 25/27 tok 7 act 0; release_long 24/27 tok 7 act 0 | SUCCESS_12S |
| 0060 | BASIN_NOT_REACHED | no | 0 | - | chase_lost:NOT_REPRESENTABLE;release_long:NOT_REPRESENTABLE | - | - |
| 0065 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0070 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0075 | BASIN_NOT_REACHED | yes | 2 | contact_preserving_slide | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S | slide_50 58/60 tok 0 act 0; slide_45 76/78 tok 0 act 0 | BASIN_NOT_REACHED |
| 0080 | BASIN_NOT_REACHED | yes | 2 | contact_preserving_slide | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S | slide_50 28/28 tok 23 act 0; slide_45 36/37 tok 30 act 0 | BASIN_NOT_REACHED |
| 0085 | SUCCESS_12S | yes | 2 | release_regrasp,repeated_recatch | slide_50:NOT_REPRESENTABLE;chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 36/39 tok 4 act 0; release_long 23/29 tok 6 act 0 | - |
| 0090 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 23/27 tok 7 act 0; release_long 25/28 tok 7 act 0 | BASIN_NOT_REACHED |
| 0095 | BASIN_NOT_REACHED | yes | 1 | repeated_recatch | release_long:SUCCESS_12S | release_long 30/33 tok 6 act 0 | BASIN_NOT_REACHED |
| 0100 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0105 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0110 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0115 | SUCCESS_12S | yes | 1 | sliding | slide_50:SUCCESS_12S | slide_50 36/38 tok 0 act 0 | - |
| 0120 | BASIN_NOT_REACHED | yes | 1 | release_regrasp | chase_lost:NOT_REPRESENTABLE;release_long:SUCCESS_12S | release_long 23/26 tok 0 act 0 | SUCCESS_12S |
| 0125 | SUCCESS_12S | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 26/27 tok 19 act 0; release_long 28/28 tok None act 0 | - |
| 0130 | BASIN_NOT_REACHED | yes | 2 | sliding | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S | slide_50 27/28 tok 0 act 0; slide_45 42/44 tok 0 act 0 | BASIN_NOT_REACHED |
| 0135 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0140 | BASIN_NOT_REACHED | yes | 4 | airborne_chase,contact_preserving_slide,release_regrasp,sliding | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S;chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | slide_50 65/67 tok 0 act 0; slide_45 102/110 tok 0 act 0; chase_lost 28/28 tok 4 act 0; release_long 26/28 tok 4 act 0 | BASIN_NOT_REACHED |
| 0145 | DROP_AFTER_HANDOFF | yes | 1 | sliding | slide_50:NOT_REPRESENTABLE;slide_45:SUCCESS_12S | slide_45 38/40 tok 0 act 0 | BASIN_NOT_REACHED |
| 0150 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0155 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 23/27 tok 0 act 0; release_long 26/27 tok 4 act 0 | BASIN_NOT_REACHED |
| 0160 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0165 | SUCCESS_12S | yes | 2 | contact_preserving_slide,release_regrasp | slide_50:SUCCESS_12S;chase_lost:NOT_REPRESENTABLE;release_long:SUCCESS_12S | slide_50 37/38 tok 31 act 26; release_long 26/29 tok 0 act 0 | - |
| 0170 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0175 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0180 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0185 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0190 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 28/28 tok 4 act 0; release_long 27/28 tok 4 act 0 | SUCCESS_12S |
| 0195 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0200 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 26/28 tok 7 act 0; release_long 28/28 tok None act 0 | BASIN_NOT_REACHED |
| 0205 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0210 | BASIN_NOT_REACHED | no | 0 | - | slide_50:NOT_REPRESENTABLE;slide_45:NOT_REPRESENTABLE | - | - |
| 0215 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 26/29 tok 1 act 0; release_long 26/29 tok 1 act 0 | BASIN_NOT_REACHED |
| 0220 | BASIN_NOT_REACHED | yes | 4 | airborne_chase,contact_preserving_slide,release_regrasp,sliding | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S;chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | slide_50 65/66 tok 0 act 0; slide_45 84/86 tok 0 act 0; chase_lost 26/28 tok 8 act 3; release_long 26/29 tok 8 act 3 | SUCCESS_12S |
| 0225 | BASIN_NOT_REACHED | yes | 2 | airborne_chase,release_regrasp | chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | chase_lost 24/27 tok 3 act 0; release_long 22/27 tok 3 act 0 | BASIN_NOT_REACHED |
| 0230 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0235 | BASIN_NOT_REACHED | yes | 1 | sliding | slide_45:SUCCESS_12S;chase_lost:NOT_REPRESENTABLE;release_long:BASIN_NOT_REACHED | slide_45 78/80 tok 40 act 2 | BASIN_NOT_REACHED |
| 0240 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0245 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0250 | SUCCESS_12S | yes | 1 | sliding | slide_50:SUCCESS_12S;chase_lost:NOT_REPRESENTABLE;release_long:BASIN_NOT_REACHED | slide_50 34/35 tok 28 act 0 | - |
| 0255 | BASIN_NOT_REACHED | yes | 1 | release_regrasp | chase_lost:NOT_REPRESENTABLE;release_long:SUCCESS_12S | release_long 27/28 tok 0 act 0 | BASIN_NOT_REACHED |
| 0260 | DROP_DURING_RECOVERY | no | 0 | - | none | - | - |
| 0265 | BASIN_NOT_REACHED | yes | 2 | contact_preserving_slide | slide_50:SUCCESS_12S;slide_45:SUCCESS_12S | slide_50 61/62 tok 0 act 0; slide_45 75/77 tok 0 act 0 | BASIN_NOT_REACHED |
| 0270 | BASIN_NOT_REACHED | no | 0 | - | none | - | - |
| 0275 | BASIN_NOT_REACHED | yes | 3 | airborne_chase,contact_preserving_slide,release_regrasp | slide_50:SUCCESS_12S;chase_lost:SUCCESS_12S;release_long:SUCCESS_12S | slide_50 28/28 tok 23 act 0; chase_lost 23/32 tok 0 act 0; release_long 22/29 tok 0 act 0 | BASIN_NOT_REACHED |
| 0280 | BASIN_NOT_REACHED | no | 0 | - | chase_lost:NOT_REPRESENTABLE;release_long:NOT_REPRESENTABLE | - | - |

Raw numbers and traces: `demo_results/diagnostics/raw/unified_k47/diagnosis/summary.json`, `representative_traces.json`, and `teacher_token_rollout.json`. The audit script is `training/diagnose_k47_closed_loop.py`.
