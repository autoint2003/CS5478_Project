# Legal recovery trajectories on v2 TRAIN

Search file: `training/legal_recovery_v2_pilot.py`. Log: `results/diagnostics/raw/legal_recovery_v2/search.jsonl`. No FINAL_TEST snapshot was loaded. No policy was trained.

## Budget

For each of the 21 v2 TRAIN captures, six legal trials:

- `slide_50` and `slide_45` from `training/sliding_vs_regrasp.py`
- `chase_lost` and `release_long` from `training/privileged_ball_impact_search.py`
- the historical open-loop `wrist_reseat` and `gravity_inward` lists, rewritten with `v1_normalized_to_v3`

Actions go through `step_legal`. The cylinder, the hand, and the ball are not teleported. Handoff uses `handoff_z=None`. A second replay of each 10 s success forces `handoff_z=0.455` without changing the action list.

126 trials. 36 of them hold for 10 s and 12 s under nominal control after the basin. 17 of 21 initial conditions have at least one such hold. Four have none: `ball_train_v2_0003`, `0010`, `0011`, `0012`.

## Mechanisms

| outcome | initial conditions |
| --- | ---: |
| slide law holds, release does not | 3 |
| release holds, slide does not | 11 |
| both hold | 3 |
| neither holds | 4 |

The historical wrist list and the historical gravity-inward list do not reach the basin on these 21 states. They are the short secure-tilt schedules, not a single-sided support controller. That candidate was not added.

Of the 11 slide holds, only `ball_train_v2_0004` / `slide_45` keeps both fingers in contact on every recovery step. The other ten slide holds lose both contacts for 1–5 steps of 20 ms. They are basin holds from the slide law. They are not the strict continuous-contact slide from the matched-search report.

Release holds open the gripper and spend many steps with no finger contact. A short regrasp is not counted: the label requires the basin, the handoff, and the 10 s nominal hold.

## Height gate on these same actions

35 of the 36 holds become `BASIN_NOT_REACHED` when the same actions are replayed with `z >= 0.455`. One hold still passes. The unconstrained replay already passed `basin_ok`, so the added predicate is the height gate. That matches the audit on `ball_train_0003` and `ball_train_0018`.

## K=36

The 11 slide holds lie inside some prototype plus a residual of at most 0.19. Replaying the clipped token-plus-residual actions still holds for 10 s on all 11.

The 25 release holds do not. The largest gap is about 0.45–0.52, above the ±0.25 residual. They are stored as legal continuous demonstrations and are not trainable under the current codebook. The tokenizer was not changed.

Stored action files: `results/diagnostics/raw/legal_recovery_v2/trajectories/`.

## Readiness

v1 TRAIN 28 plus v2 TRAIN 21 is 49 initial conditions. Eleven K=36-verified holds cover six of the new states, and they are mostly the slide law. Release, which succeeds on more states, does not fit the codebook. One 30° azimuth bin has no v2 accept. Four new states have no hold under this budget.

TRAIN DATASET READY FOR GRPO: NO

LEGAL RECOVERY TRAJECTORY COLLECTION: PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
