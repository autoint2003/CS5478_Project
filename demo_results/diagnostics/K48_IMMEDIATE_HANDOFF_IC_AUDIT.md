# Immediate-HANDOFF initial-condition audit

Policy-free replay of the saved snapshot and controller state, then the unchanged nominal controller for 12 seconds. No recovery command is applied. `SUCCESS_12S` means the lift target is reached with both fingers in contact and the grasp lasts through that window. Snapshots, rewards, checkpoints, and the success rule were not modified. FINAL was not opened.

Raw tables:

- `demo_results/diagnostics/raw/k48_handoff/nominal_audit/train_immediate.jsonl` (312 TRAIN rows)
- `demo_results/diagnostics/raw/k48_handoff/nominal_audit/valid_immediate.jsonl` (26 VALID rows)
- `demo_results/diagnostics/raw/k48_handoff/nominal_audit/demo_replays.jsonl` (126 nominal-failing TRAIN rows)
- `demo_results/diagnostics/raw/k48_handoff/nominal_audit/summary.json`

Every immediate-handoff row has `recovery_elapsed = 0`. Every `SUCCESS_12S` row has `nominal_elapsed >= 12` and `lift_completed = true`.

## Confirmed counts

| Set | N | Nominal-solvable | Nominal-failing |
| --- | ---: | ---: | ---: |
| TRAIN | 312 | 186 | 126 |
| of which v1 | 28 | 24 | 4 |
| of which v2 | 284 | 162 | 122 |
| v1 VALID | 26 | 26 | 0 |
| Held-out TRAIN slice (v2 id divisible by 5) | 56 | 30 | 26 |

The 126 TRAIN failures are `DROP_AFTER_HANDOFF` 99 and `LOST_GRASP` 27. There is no `NOMINAL_LIFT_NOT_COMPLETED` and no invalid rollout.

The held-out 56 split is 30 `SUCCESS_12S`, 20 `DROP_AFTER_HANDOFF`, 6 `LOST_GRASP`. That is the same score and the same failure mix as both update-936 argmax evaluations. Those evaluations recorded recovery duration 0 on every episode. The 30/56 result is the nominal controller, measured again here without a policy.

## Impact family, timing, and generation parameters

v1 stores a named direction. v2 stores `direction = continuous` and a continuous azimuth, speed, height offset, pitch, and delay. Bins below are computed from `train_immediate.jsonl`.

**v1 direction** (azimuth assigned before generation: TRAIN 0 / 45 / 90):

| Direction | Azimuth | N | Solvable | Failing |
| --- | ---: | ---: | ---: | ---: |
| +X | 0 | 9 | 9 | 0 |
| +Y | 90 | 10 | 10 | 0 |
| oblique +X+Y | 45 | 9 | 5 | 4 |

All 4 v1 failures are the oblique family, all at delay 0.40 s. Delays 0.00 (7) and 0.20 (3) are solvable. v1 speeds are the discrete grid 6.3–9.5 m/s; failures are one at 6.5, one at 7.3, and two at 7.5. Both v1 cases with `z_off = 0` fail; the 16 cases with negative `z_off` all succeed.

**v2 delay** (continuous, about 0 to 0.40 s). Solvable fraction stays near 55–59% in every bin:

| Delay (s) | N | Solvable | Failing |
| --- | ---: | ---: | ---: |
| [0.0, 0.1) | 70 | 40 | 30 |
| [0.1, 0.2) | 74 | 40 | 34 |
| [0.2, 0.3) | 81 | 47 | 34 |
| [0.3, 0.4] | 59 | 35 | 24 |

**v2 speed.** Higher speed is associated with nominal failure:

| Speed (m/s) | N | Solvable | Failing |
| --- | ---: | ---: | ---: |
| 5–6 | 48 | 42 | 6 |
| 6–7 | 54 | 37 | 17 |
| 7–8 | 68 | 36 | 32 |
| 8–9 | 50 | 22 | 28 |
| 9–10 | 64 | 25 | 39 |

**v2 height offset and pitch:**

| Parameter | N | Solvable | Failing |
| --- | ---: | ---: | ---: |
| z < -0.005 m | 130 | 99 | 31 |
| \|z\| ≤ 0.005 m | 78 | 23 | 55 |
| z > 0.005 m | 76 | 40 | 36 |
| \|phi\| < 10° | 225 | 116 | 109 |
| 10–25° | 41 | 29 | 12 |
| 25–40° | 12 | 11 | 1 |
| \|phi\| ≥ 40° | 6 | 6 | 0 |

**v2 azimuth**, 45° sectors. Solvable fraction is 44–63% in every sector, so azimuth alone does not separate the two classes. Counts are in `summary.json` under the per-value maps; the sector totals are 0–45: 21/37, 45–90: 15/34, 90–135: 19/34, 135–180: 22/37, 180–225: 22/34, 225–270: 22/37, 270–315: 17/33, 315–360: 24/38.

**Logged contact states** (a row can carry more than one label):

| State present | N | Solvable |
| --- | ---: | ---: |
| airborne_object | 34 | 1 |
| delayed_grasp_failure | 8 | 0 |
| intermittent_contact | 3 | 0 |
| complete_grasp_loss | 57 | 19 |
| tangential_slip | 66 | 34 |
| asymmetric_or_unilateral | 262 | 148 |
| contact_migration | 135 | 99 |
| relative_rolling | 83 | 57 |
| no logged state | 13 | 12 |

The joint label `airborne_object, asymmetric_or_unilateral, complete_grasp_loss` is 0/25 solvable. These labels describe the impact that was accepted. They were not a nominal-success filter.

## Demonstrations

The 534 K48 supervised trajectories cover 153 TRAIN initial conditions.

| Demonstration origin | Trajectories | Initial conditions |
| --- | ---: | ---: |
| Nominal-solvable | 428 | 118 |
| Nominal-failing | 106 | 35 |
| Nominal-failing with no demonstration | 0 | 91 |

Of the 428 trajectories on solvable states, 190 are `early_handoff` and 238 are `recorded_end`. All 190 early-HANDOFF labels sit on states where immediate HANDOFF already succeeds. All 106 trajectories on nominal-failing states are `recorded_end`; none has a verified early HANDOFF. Mechanism counts on the failing 106: airborne chase 61, release-regrasp 38, repeated recatch 7. Contact-preserving slide (109) and sliding (93) occur only on nominal-solvable states.

## Replay of legal demonstrations on nominal-failing states

Each of the 126 failing TRAIN states was checked against the 534 trajectories. Replay used the saved commands, the same snapshot, `limit_s = 4.0`, basin early-exit off, and the same 12-second nominal check. The first historical trajectory that succeeded was enough; pause trajectories were not needed.

| Status | Initial conditions |
| --- | ---: |
| Confirmed recoverable | 35 |
| Demonstration replay failed | 0 |
| No demonstration (unknown recoverability) | 91 |

All 35 confirmations are the first historical trajectory tried, and each replay returned `SUCCESS_12S`. The 91 states have no successful legal K48 trajectory. They are not shown to be unrecoverable. No new search was run.

## Why nominal-solvable states were accepted

Nominal failure was never a selection criterion.

`classify` in `training/ball_impact_dataset.py` returns `valid_impact_recovery_ic` after a launched ball, a verified ball-cylinder collision, and a geometry entry that does not precede that collision. The snapshot is then rejected only when `basin_ok` is already true:

```595:598:training/ball_impact_dataset.py
    if success.basin_ok(row, closed_grip(env.sim)):
        evidence["category"] = "trivial_goal_state"
        evidence["reason"] = "entry state is already inside the nominal basin"
        return evidence, None
```

That basin is the historical pinch test (position, orientation, speed, closed grip). It is not `SUCCESS_12S`. A state can sit outside the 6 mm basin and still be finished by the nominal controller. Snapshot fidelity (`qpos` error ≤ 1e-3) is the other post-entry gate. v2 calls the same `run_episode` path, forces `split = train`, and adds no nominal-failure test.

v1 azimuths were assigned before generation: TRAIN 0/45/90, VALIDATION 135/180, FINAL_TEST 225/270/315. VALID used the same acceptance function on a different azimuth pair. Measured immediate HANDOFF on those 26 files is 26/26 `SUCCESS_12S`. The four TRAIN azimuths that fail are all oblique 45°. The validation azimuths happen to be states the nominal controller can finish.

## GRPO updates 1–936

Both logs contain 936 updates and 3744 rollouts (group size 4). Each initial condition is visited once per epoch of 312 updates. The training loop samples a token and, if it is token 47, breaks before `step_legal`. A logged row with `handoff = early` and `recovery_elapsed = 0` is a first-token HANDOFF. The logs do not store the token id; the elapsed time and handoff field are the control-flow record.

First-token HANDOFF, multi-step rollouts (`recovery_elapsed ≥ 0.02`, or a forced or dropped handoff), and successes:

| Version | Updates | First-token HANDOFF | Multi-step successes | Successes on solvable ICs | Successes on failing ICs |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 | 1–156 | 0/624 | 285 | — | 23 |
| V1 | 157–312 | 4/624 | 286 | — | 21 |
| V1 | 313–468 | 43/624 | 298 | — | 14 |
| V1 | 469–624 | 465/624 | 29 | — | 4 |
| V1 | 625–780 | 570/624 | 5 | — | 3 |
| V1 | 781–936 | 598/624 | 1 | — | 0 |
| V2 | 1–156 | 0/624 | 278 | — | 26 |
| V2 | 157–312 | 0/624 | 270 | — | 23 |
| V2 | 313–468 | 18/624 | 309 | — | 11 |
| V2 | 469–624 | 414/624 | 37 | — | 6 |
| V2 | 625–780 | 563/624 | 6 | — | 3 |
| V2 | 781–936 | 596/624 | 3 | — | 0 |

Epoch totals (744 rollouts on the 186 solvable states, 504 on the 126 failing states):

| Version | Epoch | First-token rate | Reward-std median | Updates with nonzero advantage | Updates with identical returns | Solvable success | Failing success | Failing multi-step rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 1 | 0.003 | 0.023 | 287/312 | 25/312 | 531/744 | 44/504 | 504/504 |
| V1 | 2 | 0.407 | 0.0009 | 206/312 | 106/312 | 674/744 | 18/504 | 310/504 |
| V1 | 3 | 0.936 | 0.0 | 58/312 | 254/312 | 735/744 | 3/504 | 40/504 |
| V2 | 1 | 0.000 | 0.209 | 294/312 | 18/312 | 499/744 | 49/504 | 504/504 |
| V2 | 2 | 0.346 | 0.0009 | 223/312 | 89/312 | 655/744 | 17/504 | 340/504 |
| V2 | 3 | 0.929 | 0.0 | 60/312 | 252/312 | 736/744 | 3/504 | 52/504 |

Successful recovery trajectories disappear from the samples. V1 multi-step successes fall from 571 in epoch 1 to 6 in epoch 3. V2 falls from 548 to 9. In updates 781–936, successes on nominal-failing states are zero in both runs.

The three epoch-3 successes on failing states are still short sampled recoveries, not first-token HANDOFF: V1 updates 709, 744, and 759 (`ball_train_v2_0085` at 0.86 s, `0120` and `0135` at 0.04 s); V2 updates 722, 744, and 759 (`0098` at 0.72 s, then the same two 0.04 s episodes).

Sampling can still take a recovery step on a nominal-failing state late in training. Epoch 3 has 40 such V1 rollouts and 52 V2 rollouts out of 504. In updates 781–936 the counts are 13 and 18 out of the rollouts in that window. Those samples no longer return `SUCCESS_12S`.

## Validation

The 26 v1 VALID snapshots are azimuth 135° (22, oblique −X+Y) and 180° (4, −X), delays 0.00 (7), 0.20 (7), and 0.40 (12). Immediate HANDOFF succeeds on all 26, with zero recovery time and a completed 12-second grasp. The update-936 policies scoring 26/26 are doing the same thing.

These states are a no-intervention control. They do not evaluate recovery: a policy that emits token 47 at the first step scores 26/26, and so does a policy that emits no action at all.

## Revised dataset protocol

Keep the three groups separate. Do not put them in one GRPO objective.

1. **Recovery pool (35 TRAIN states).** Nominal HANDOFF fails, and a saved legal trajectory replays to `SUCCESS_12S`. Use the 106 `recorded_end` trajectories on these states (58 historical, 48 pause). Do not supervise token 47 from a prefix that was never verified on these states.
2. **Unknown pool (91 TRAIN states).** Nominal HANDOFF fails and no legal K48 trajectory exists. Exclude them from the reward until a legal recovery is confirmed. This audit did not search for one, and did not label them unrecoverable.
3. **No-intervention control (186 TRAIN states, plus the 26 VALID states).** Report immediate-HANDOFF `SUCCESS_12S` on this group by itself. Do not sample recovery actions on it for a GRPO update. On these states the maximum of both Reward V1 and Reward V2 is immediate success, because both subtract `0.05 * recovery_seconds` from a successful return.

Do not use the 190 `early_handoff` demonstrations as recovery targets. Every one of them comes from a state the nominal controller already solves. The 56-state training slice should be reported as 30 control states and 26 recovery-relevant states, not as one success rate. A collapsed policy scores 30/56 on that slice by construction.

## Hypotheses

These are inferences, not stored measurements.

- Token 47 became the argmax because it is optimal on 186 of 312 training states, and both rewards pay more for a shorter success. By epoch 3, groups mostly share one return, so the advantage is skipped on 254/312 (V1) and 252/312 (V2) updates. The logs do not store logits or per-state gradient shares, so this does not identify which states moved the HANDOFF logit.
- The association of nominal failure with higher ball speed, near-zero height offset, and airborne or delayed-loss labels is descriptive. The generator did not select on those associations.
- The 91 undemonstrated failing states may include further recoverable states. That was not tested.
