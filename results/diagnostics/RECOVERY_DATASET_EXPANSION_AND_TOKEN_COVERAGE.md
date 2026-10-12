# Recovery dataset expansion and token coverage

No BC, GRPO, or SAC update was run. VALIDATION (26) and FINAL_TEST (44) were not read for search and were not rewritten. v1 TRAIN remains 28 pickles.

The per-trial table for all 234 v2 searches is `results/diagnostics/raw/legal_recovery_v2/table.md`, generated from `per_ic_table.json`. Counts below are taken from those rows, not from the earlier aggregate sentence.

## 1. What the earlier 6 and 11 meant

On the first 21 v2 captures the slide laws produced **11 successful trajectories on 6 initial conditions**. Five of those conditions have both `slide_50` and `slide_45`. `ball_train_v2_0008` has only `slide_45`. Each of those 11 is a separate action list. All 11 are inside the current K=36 codebook (largest residual 0.19) and the quantized replay also holds for 10 s and 12 s after handoff.

The sentence "six slide-law holds" counted initial conditions. The sentence "11 quantized slide trajectories" counted action lists. They are the same experiments.

After the later 18 captures, the same search has **17 slide trajectories on 9 initial conditions**, and all 17 quantized replays hold. Release trajectories are a different list: 44 of them hold in continuous actions, and none of those 44 is inside K=36.

| | initial conditions | trajectories |
|---|---:|---:|
| slide law, continuous and K=36 replay | 9 | 17 |
| release law, continuous only | 25 | 44 |
| any legal 10 s hold | 28 of 39 | 61 |
| no hold under this budget | 11 of 39 | — |

An initial condition can appear in both mechanism rows. 6 conditions have both a slide hold and a release hold. 3 have only slide. 19 have only release.

Every one of the 61 holds has `primary_10s`, post-handoff `held_12`, and `legacy_retained_to_12`. The 10 s and 12 s timers in `nominal_validate` start at the handoff. Absolute t = 12 is the separate `legacy_retained_to_12` flag, and it is also true on these 61 because the impact-plus-recovery clock has already passed 12 s by the end of the hold. A post-handoff 12 s success is not the same definition as retention at absolute t = 12; both happened to be true here.

`handoff_z=None` is what opens those handoffs. Forcing 0.455 m on the original 36 holds rejected 35 of them. That comparison was not repeated on the later 25 holds.

Historical `wrist_reseat` and `gravity_inward` lists do not hold on any of the 39 captures.

## 2. The K=33 film example is not on disk

`training/film_impact_zp6_k33.py` expects:

- `results/diagnostics/raw/action_representation/ckpt_lateral_token.pt`
- `results/videos/impact_zp6_k33_lateral_chase_trace.csv`
- `results/videos/impact_zp6_k33_summary.txt`
- `results/videos/impact_zp6_k33_lateral_chase.mp4`

All four are missing. The script stores only the first time both contacts are gone and the first later time both return (`t_loss`, `t_recontact`). Without the CSV, later losses cannot be reconstructed, and a trajectory was not invented.

The current K=36 prototype 32 is `[1.000, 0.002, -0.050, -0.072, 0.002, 0.003, -1]`, within 0.0005 of the documented lateral-chase token. Prototype 35 is `[-0.091, 0.558, 0.450, 0, 0, 0, -1]`, the later airborne chase, not a copy of token 32. The release gap is not a missing token 32.

Repeated contact loss does show up in the new legal traces. Eleven of the 61 holds lose both fingers, regain them, and lose them again. The clearest release example is `ball_train_v2_0021` `release_long`: no contact on steps 1–10, 13–26, and 29–39, with bilateral contact between those gaps, then a hold. That is one scripted chase with several openings, not a second network.

## 3. Why release actions miss K=36

On the steps whose nearest prototype is farther than 0.25, the excess is mostly hand-frame `vy` (hundreds of steps) and `vz`, plus a smaller set of `vx` and grip steps. Wrist channels are already within 0.01. The far steps are an open gripper (`grip` near −1) with translation that is not the large `vx` of token 32 and not the `vy`/`vz` pair of token 35.

One open-and-nearly-still command, about `[0, 0, 0, 0, 0, 0, -1]`, sits within 0.25 of 410 of the 461 distinct far actions from the first search. Token 32 is about 1.0 away from that command because its `vx` is 1. Token 35 is about 0.56 away because of its `vy`. Adding that one still-open token raises full-trajectory coverage of the original 36 holds from 11 to 13. Coverage of those 36 reaches 36 only after 11 added centers (K=47). Four farthest-point tokens do not cover the dense cluster.

Those 11 centers were appended. The original 36 centers were not moved. The 11 slide holds keep the same quantized actions. The 25 release holds from that first set fall inside ±0.25. Replaying those quantized actions from the snapshot:

- 24 return `SUCCESS_12S`
- `ball_train_v2_0002` `chase_lost` becomes `BASIN_NOT_REACHED` (continuous hold, quantized miss)

The candidate is not adopted. One physical regression is enough, and the later release traces reach `vy` about 0.59 and `vx` about −0.57, outside the set those 11 centers were chosen from. The residual cap was not raised. The policy is still K=36.

## 4. TRAIN expansion

`ball_impact_v2` append-only batches `b0001`–`b0003`: 216 simulated launches, 177 cylinder hits that never reached the 6 mm entry, 39 accepted. No launch was rejected for an earlier robot or environment contact. Restore errors stay under 1e-3. v1 counts stay 28 / 26 / 44.

Accepted azimuth bins of 30° have 3 or 4 captures each, except 180–210°, which still has none. Launches in that bin do hit the cylinder. They do not reach the entry detector. The detector threshold was not changed.

Combined TRAIN initial conditions: 28 + 39 = 67. The planning target of 300 is not met. Entry yield is 39/216.

## 5. Readiness

The executable demonstration set has 61 continuous holds on 28 of 39 new captures, plus the older v1 searches that were not rerun here. Seventeen of those holds are inside K=36 and survive quantized replay. Forty-four release holds are legal and are not trainable under the current codebook. A K=47 append reproduces 24 of 25 tested release holds and loses one. Eleven new captures have no hold. The 180–210° entry bin is empty. This is not a balanced training set for the current Transformer.

TRAIN IC DATASET READY: NO

EXECUTABLE DEMONSTRATION SET READY FOR K=36 TRAINING: NO

POLICY TRAINING HAS NOT BEEN STARTED
