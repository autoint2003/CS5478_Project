# Controlled sliding versus `slide_50`

No verified legal controlled-sliding recovery exists under the current physical definition. The historical `slide_50` demonstration is a real contact-preserving slide, and its saved commands are inside the K48 action interface, but the initial condition is already solved by immediate HANDOFF. The current `slide_50` search uses the same feedback law on nominal-failing states and records 0/142 `SUCCESS_12S`.

Physics, the success definition, the codebook, the 4 s recovery limit, and the existing manifests were not changed. No FINAL snapshot was opened. No training was started.

Raw replays are in `demo_results/diagnostics/raw/k48_sliding_audit/`. Historical trajectory files were only read.

## Answer

`ball_train_0003` sliding `slide_50` does produce controlled sliding, and the quantized command sequence holds for 12 s after handoff. Immediate HANDOFF on that same snapshot also holds for 12 s. It is a `NOMINAL_SOLVABLE_DEMO`, not a recovery.

Current `slide_50` calls the same `slide_action` specification (`grip=-0.50`, `kp=20`, `cap=0.45`, `brake=True`). It does not replay a historical success. On the 142 nominal-failing candidates it never reaches the basin handoff that ended the historical demos. When finger contact is already gone, that law sets translation to zero. The 0/142 count is that behavior on the hard pool, not a failure to quantize a working slide.

## 1. Historical inventory

No video or rendered demo file is in the repository. The claims below come from saved trajectories and the two search reports.

### Strict controlled-sliding demonstrations

These five are the trajectories the matched-search report labels `controlled_sliding` with continuous contact. Source report: `demo_results/diagnostics/SLIDING_VS_REGRASP_MATCHED_SEARCH.md`, section 8. The controller is `slide_action` in `training/sliding_vs_regrasp.py`. Actions are `recovery7d_airborne_v3` commands chosen with privileged contact and pinch-error feedback. The cylinder, hand, ball, and controller targets are not edited.

| ID | File | Scenario | Duration | Historical 10 s / 12 s |
| --- | --- | --- | --- | --- |
| `ball_train_0003` | `demo_results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0003__sliding_slide_50.npz` | +X, 6.7 m/s, z offset -0.013 m, phi 5°, delay 0 | 0.68 s, 34 steps | True / True |
| `ball_train_0007` | `.../ball_train_0007__sliding_slide_45.npz` | +X, 6.5 m/s, z offset -0.015 m, phi 0°, delay 0.4 s | 0.88 s | True / True |
| `ball_train_0013` | `.../ball_train_0013__sliding_slide_45.npz` | +Y, 7.5 m/s, z offset -0.015 m, phi 0°, delay 0.4 s | 1.22 s | True / True |
| `ball_train_0014` | `.../ball_train_0014__sliding_slide_58.npz` | +Y, 7.5 m/s, z offset -0.015 m, phi 40°, delay 0.4 s | 1.40 s | True / True |
| `ball_train_0018` | `.../ball_train_0018__sliding_slide_45.npz` | +Y, 8.5 m/s, z offset -0.015 m, phi 40°, delay 0 | 1.32 s | True / True |

Snapshots are `demo_results/diagnostics/raw/ball_impact_dataset/accepted/train/<id>.pkl`.

The historical strict test required continuous finger contact, grip above -0.85, pinch-error drop of at least 3 mm with translation, measured tangential slip, and two ablations (no unloaded translation, and grip forced closed) that each lost the old 10 s hold. Those ablations were not re-run in this audit. The contact, slip, pinch, and 12 s outcomes below were re-simulated.

### Broader sliding family

`demo_results/diagnostics/raw/sliding_vs_regrasp/trajectories/` contains 37 `*sliding_slide*.npz` files, including the five above and 19 action-noise seeds (`_s1`, `_s2`). Many are short contact-gap slides or `slide_58` variants. The report's section 8 also lists hybrid files. Hybrids that enter a release chase are not strict controlled slides. They were not reclassified as sliding recoveries.

### Earlier privileged label

`demo_results/diagnostics/PRIVILEGED_BALL_IMPACT_RECOVERY_SEARCH.md` marks 4 of 18 v1 states `controlled_sliding` (`ball_train_0004`, `0014`, `0016`, `0018`). That label used a different test: slip plus an ablation that zeros lateral channels. The saved best law on those states is a release or chase law, not `slide_action`. Several of those sequences were outside the old K=36 residual and their quantized 10 s replay failed. They are not the `slide_50` demonstration.

### K48 supervised slide rows

`demo_results/diagnostics/raw/k48_handoff/supervised_targets.jsonl` has 202 rows tagged `contact_preserving_slide` (109) or `sliding` (93), on 68 initial conditions, strategies `slide_45` (120) and `slide_50` (82). All 68 are in the earlier immediate-HANDOFF audit, and all 68 are nominal-solvable there. This audit did not re-simulate all 202 action sequences. It re-simulated three recorded-end `slide_50` sequences (`ball_train_v2_0009`, `0045`, `0106`).

## 2. Revalidation

Procedure for every `*sliding_slide*.npz` file: restore the v1 snapshot, run immediate HANDOFF for 12 s, then replay the commands for up to 4 s and run the same 12 s nominal check. If every command is within ±0.25 of a K47 prototype, the replay uses the quantized command. Otherwise it uses the saved command and records the gap. Contact samples are taken with `grasp_sample` on the recovery window.

Result for all 37 files: immediate HANDOFF `SUCCESS_12S`, action replay `SUCCESS_12S`, class `NOMINAL_SOLVABLE_DEMO`. Nominal elapsed on those holds is 12.0 s. Thirty-one sequences are exactly representable. Six noise seeds have max L-infinity 0.256–0.283 and were replayed as saved commands; they still hold, and their initial conditions are already nominal-solvable.

The five strict demonstrations, replayed as quantized K48 actions:

| ID | Quantized replay | Immediate HANDOFF | Continuous contact | Slip | Pinch drop | Max L-inf |
| --- | --- | --- | --- | --- | --- | --- |
| `ball_train_0003` `slide_50` | `SUCCESS_12S` at 0.68 s | `SUCCESS_12S` | yes, 26 bilateral / 34 | yes, v_tan 0.040 m/s, migration 0.020 m | 10.5 mm | 0.205 |
| `ball_train_0007` `slide_45` | `SUCCESS_12S` at 0.88 s | `SUCCESS_12S` | yes, 40/44 | yes | 12.4 mm | 0.181 |
| `ball_train_0013` `slide_45` | `SUCCESS_12S` at 1.22 s | `SUCCESS_12S` | yes, 55/61 | yes | 11.5 mm | 0.186 |
| `ball_train_0014` `slide_58` | `SUCCESS_12S` at 1.40 s | `SUCCESS_12S` | yes, 62/70 | yes | 6.4 mm | 0.199 |
| `ball_train_0018` `slide_45` | `SUCCESS_12S` at 1.32 s | `SUCCESS_12S` | yes, 59/66 | yes | 6.7 mm | 0.193 |

`ball_train_0003` grip on the saved commands runs from -0.50 to +1.0. That is the unloaded slide followed by the reclose inside `slide_action`, not an external grip override. Handoff time is the recorded end of that closed loop: the original search stopped when the pinch basin had persisted for 5 steps. This replay did not stop early on the basin. It played the saved commands and then handed off, and the 12 s hold still passed.

The three v2 `slide_50` examples also return immediate `SUCCESS_12S` and replay `SUCCESS_12S`. Their recovery windows have 3–5 zero-contact steps, so they are contact-gap slides on nominal-solvable states, not stricter than `ball_train_0003`.

Classification count: `VERIFIED_LEGAL_SLIDING_RECOVERY` 0, `NOMINAL_SOLVABLE_DEMO` 37, `PRIVILEGED_OR_INCOMPATIBLE` 0, `SHORT_HORIZON_ONLY` 0, `INSUFFICIENT_EVIDENCE` 0 for this file set.

The six out-of-cap noise seeds are still `NOMINAL_SOLVABLE_DEMO` because immediate HANDOFF succeeds. They are not an additional legal K48 token sequence.

## 3. What `slide_50` actually does

Current search: `SEARCH_PLAN` and `propose_actions` in `training/rebuild_k48_recovery.py`. The action law is `slide_action` with the `slide_50` spec from `training/sliding_vs_regrasp.py`.

Construction. Each step reads the pinch point, the saved nominal pinch `p_nom`, finger contact counts, normal force, and object velocity in the hand frame. The command is a hand-frame velocity toward that pinch, a small wrist return, and a grip torque, passed through `normalize_action(..., AIRBORNE_V3)`. Direction is the pinch error, not a fixed world axis. There is no separate open-loop slide script.

Contact and grip. With both fingers in contact and normal force at least 0.8 N, grip stays -0.50 and the full velocity is used, capped at 0.45 m/s. One contact raises grip to at least -0.40 and scales translation by 0.35. Zero contacts raises grip to at least -0.30 and sets translation scale to 0. The brake path, when normal force is below 0.8 N, moves grip by +0.10 and scales translation by 0.45. When pinch error is at most 3.2 mm and contact is bilateral, grip becomes +1.0 and a short lift is added if the object is below 0.49 m.

Feedback. The proposal is closed-loop. The verified trial is not. `legalize` snaps the recorded commands to the K47 codebook with residual ±0.25, and `legal_verify` replays that open-loop sequence.

Duration and handoff. The proposal runs until the recovery fall rule, 5 steps inside the old pinch basin, the 4.0 s limit, or 200 steps. In the 142 trials the stop was `recovery_failed` 141 times and the 200-step cap once. `basin_proposal` occurred zero times. If the proposal already fell and every command is inside ±0.25, the 12 s replay is skipped and the log reason is `proposal_recovery_failed`. Otherwise the decoded sequence is replayed and HANDOFF is the recorded end, or the 4 s limit, followed by `finish_recovery`.

Same mechanism as the historical demo: the feedback law and the `slide_50` numbers are the same. Different execution: historical successes were generated on v1 states the nominal controller already solves, and they stopped by entering the pinch basin. The current search applies that law to nominal-failing candidates and almost always dies, often with translation disabled, before any basin handoff.

## 4. The 142 failed trials

Log: `demo_results/diagnostics/raw/k48_recovery_v3/search.jsonl`, strategy `slide_50`. All 142 initial conditions are `RECOVERY_CANDIDATE` states (129 train, 13 valid). Speed median 8.8 m/s, 64 of 142 at or above 9 m/s. Azimuth covers all eight 45° bins. Entry state labels are dominated by airborne or unilateral grasp loss (57 airborne complete-loss, plus 9 delayed failures).

| Outcome | Count | What the log and traces show |
| --- | --- | --- |
| Proposal failed before a 12 s replay | 91 | Privileged loop already hit the fall rule, and every command was inside ±0.25, so legal replay was skipped. 75 of these last 4–9 steps (0.08–0.18 s). |
| Drop during recovery | 43 | Decoded replay ran because at least one command exceeded ±0.25. Object lost during that replay. Nominal elapsed 0. |
| Drop after HANDOFF | 8 | Recorded commands finished or the 4 s cap was hit, then the nominal phase dropped. |
| Lost grasp after a completed lift | 0 | Not present. |
| `SUCCESS_12S` | 0 | Not present. |

`proposal_stop` is `recovery_failed` for 141 and `cap` for `k48r_train_00042` only.

Saved contact traces exist for the 51 legal replays. None of those 51 has continuous contact. Seventeen have at least 5 bilateral steps, and nine have at least 10. The 91 skipped proposals have no saved contact array; five representative failures were re-simulated:

| Trial | Steps | Contacts | Commanded motion | Outcome |
| --- | --- | --- | --- | --- |
| `ball_train_0023` | 5 (0.10 s) | 0 bilateral, 5 zero-contact | grip -0.30, translation 0 | `DROP_DURING_RECOVERY`. The no-contact branch never slides. |
| `ball_train_v2_0049` | 54 (1.08 s) | 0 bilateral, 54 zero-contact | grip -0.30, translation 0 | `DROP_DURING_RECOVERY`. This is the contact-free fall counter, not a slide. |
| `k48r_train_00152` | 35 (0.70 s) | 19 bilateral, 11 zero | grip -0.50, translation mean 0.14, slip yes, pinch drop 0 | `DROP_DURING_RECOVERY`. Brief slip, pinch error not reduced. |
| `k48r_train_00032` | 115 (2.30 s) | 115 bilateral, 0 zero in the pre-step samples | grip -0.50, translation mean 0.18, slip yes, pinch drop 0.08 mm | `DROP_DURING_RECOVERY` at the fall rule. Contact continued and the pinch did not approach the nominal grasp. |
| `k48r_train_00042` | 200 (4.00 s) | 187 bilateral, 1 zero | grip -0.50, slip yes, pinch drop 7.0 mm, migration 0.022 m | `DROP_AFTER_HANDOFF`. Nominal elapsed 0.002 s. |

So one trial, `k48r_train_00042`, did keep bilateral contact for most of 4 s, did reduce pinch error by 7 mm, and did measure slip. It then lost the object at handoff. It is not `SUCCESS_12S`, and it had a one-step contact gap. `k48r_train_00032` kept contact and a little slip without moving the pinch, then failed during recovery. The short and 54-step exemplars never commanded a slide.

## 5. Side-by-side

| | Historical strict slides | Current `slide_50` on 142 candidates |
| --- | --- | --- |
| Initial conditions | 18 v1 states, ball speed 6.5–8.5 m/s, azimuth only +X or +Y. All five strict ICs pass immediate HANDOFF. | 142 nominal-failing states. Median speed 8.8 m/s, 64 at ≥9 m/s, all azimuth bins, many airborne or unilateral entries. |
| Slide direction | Hand-frame velocity toward the saved nominal pinch. | Same error signal. |
| Displacement | Strict replays: pinch error down 6–12 mm, migration 0.020–0.027 m, mean command translation 0.13–0.17. | Most exact proposals: translation scale 0. The 4 s cap case: pinch down 7 mm. The 2.3 s contact case: pinch down 0.08 mm. |
| Arm motion | Closed-loop while the demo was recorded; the npz is the open-loop record. Wrist return gain 0.30. | Same law while proposing; the scored trial is the quantized open-loop record. |
| Grip | Spec -0.50, -0.45, or -0.58, then +1.0 at the 3.2 mm bilateral settle. | Spec -0.50. On the zero-contact exemplars the law substitutes -0.30 and never reaches the reclose. |
| Contact | Five strict replays: continuous contact and slip. Other family files often have 2–5 zero-contact steps and still hold, because the IC is nominal-solvable. | 0/51 saved traces are continuous. 91 proposals were not given a 12 s replay. |
| Duration | 0.68–1.40 s to the basin stop. | 75 proposals die by 0.18 s. One reaches the 4.0 s cap. None stops on the basin. |
| HANDOFF | Historical generator: 5-step pinch basin, then nominal. This replay: recorded commands, then 12 s nominal. Both hold. | Fall rule, or the 4 s cap. No basin stop in the 142. |
| Legal K48 actions | 31/37 family files, including all five strict demos, quantize inside ±0.25 and the quantized replay holds. Six noise seeds sit at L-inf 0.256–0.283. | 91 failures are exactly representable and still fall. The codebook is not what stops them. |
| `SUCCESS_12S` | 37/37 family replays, and 37/37 immediate HANDOFF. | 0/142. |

The current search fails to reproduce the historical holds because those holds start from states the nominal controller already lifts. On the failing pool the same law sees no finger contact, zeros translation, and the fall rule ends the episode. Where contact happens to persist, the recorded sequences still do not produce a 12 s grasp after handoff.

## 6. Recommendations

Do not put the historical slide trajectories into the recovery-candidate supervised set. Immediate HANDOFF already solves those 68 K48 slide initial conditions and the 18 v1 states replayed here.

Do not treat 0/142 as a missing token or a residual-cap problem. Ninety-one of the failures are exactly representable.

Do not spend more open-loop `slide_50` trials on candidates that are already contact-free. The 4–9 step and 54-step groups are the no-contact branch and the contact-free fall counter.

Keep `k48r_train_00042` as a negative example: 4 s of mostly bilateral contact and a 7 mm pinch correction still dropped at handoff. One trial is not a recoverable slide, and this audit did not invent a replacement demonstration.

A recovery-needed controlled slide would have to show three facts together: immediate HANDOFF fails, the quantized K48 commands return `SUCCESS_12S`, and the recovery window shows continuous or otherwise measured sliding. None of the replayed historical files, and none of the 142 search trials, meet all three.
