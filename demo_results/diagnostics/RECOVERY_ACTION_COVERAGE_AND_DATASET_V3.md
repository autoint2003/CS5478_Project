# Recovery action coverage and TRAIN dataset v3

The active policy is unchanged: 39-D `observe_airborne`, `recovery7d_airborne_v3`, the existing 36 prototypes, and residual cap ±0.25. No training job was started. VALIDATION and FINAL_TEST were not read or written.

## 1. TRAIN IC count

312 TRAIN initial conditions.

- v1 TRAIN pickles: 28
- v2 TRAIN pickles (`ball_train_v2_0001` through `ball_train_v2_0284`): 284
- v1 VALIDATION: 26
- v1 FINAL_TEST: 44

v2 is append-only under `demo_results/diagnostics/raw/ball_impact_dataset_v2/`. The 6 mm / 40 ms entry detector and the MuJoCo model were not changed. Four launches that crossed the entry threshold were rejected as `invalid_snapshot` because the restore error exceeded 1e-3. They are not in the 284.

## 2. Real impact and entry statistics

616 logged v2 launches. Every simulated launch produced a ball-cylinder contact. `mj_contactForce` is a force in newtons. The log stores that force, the contact normal, and the resulting cylinder-hand pinch `max_e_pos_m`. It does not store a collision impulse.

| Proposal | Launches | Accepted entries | Entry rate |
| --- | ---: | ---: | ---: |
| Original (`b0001`–`b0003`) | 216 | 39 | 18% |
| Near-normal pre-impact bias (`b0004`–`b0006`) | 400 | 245 | 61% |
| All v2 | 616 | 284 | 46% |

On the first 216 launches, entry followed the impact direction relative to the grasp:

| Approach | Launches | Accepted |
| --- | ---: | ---: |
| Near-normal, absolute phi under 20° | 37 | 26 |
| Oblique, 20° to 45° | 82 | 12 |
| Tangential, absolute phi at least 45° | 97 | 1 |

Accepted impacts had median pinch 8.6 mm and median peak normal force 82 N. Misses had median pinch 2.1 mm and median peak normal force 54 N. Median tangential force was 7.7 N on entries and 24 N on misses. Launch speed was a weak split: under 6 m/s accepted 10/48, 6–7 m/s accepted 9/56, over 7 m/s accepted 20/112.

The empty 180–210° bin in that first set was 9 real cylinder hits and 0 entries. Median pinch on those nine was 2.4 mm and median peak normal force was 55 N. The ball reached the cylinder. The grasp did not move 6 mm. That is contact geometry for a tangential approach into the supported side of the grasp, not a missed contact and not a measurement bug.

The pre-impact proposal was then changed in `training/ball_impact_targeted.py`. Eighty percent of draws use phi inside ±12°. A bin is set aside only after eight near-normal attempts produce no entry. The detector threshold stayed 6 mm. After that change, near-normal launches accepted 240/327 and tangential launches accepted 5/73. The 180–210° bin now has 23 accepted entries. All twelve 30° bins have 23 or 24 v2 accepts. Accepted azimuths run from 1.7° to 359.1°. Accepted speeds run from 5.36 m/s to 9.65 m/s.

## 3. Successful trajectories

Legal search on all 284 v2 initial conditions: 1704 trials, six strategies each (`slide_50`, `slide_45`, `chase_lost`, `release_long`, `hist_wrist`, `hist_gravity_inward`). Handoff uses the nominal basin only.

359 trajectories hold for 10 s after handoff, on 170 of 284 initial conditions. 114 initial conditions have no hold under this menu.

| Family | 10 s holds |
| --- | ---: |
| Controlled slide (`slide_50`, `slide_45`) | 123 |
| Release / chase (`release_long`, `chase_lost`) | 236 |
| Historical wrist reseat | 0 |
| Historical `gravity_inward` | 0 |

The original 39 v2 initial conditions still account for 61 of those holds (17 slide, 44 release) on 28 initial conditions. The 245 initial conditions added in this expansion were all searched: 298 new holds on 142 initial conditions, and 103 with no hold. New-trial labels: `SUCCESS_12S` 298, `DROP_DURING_RECOVERY` 600, `BASIN_NOT_REACHED` 433, `DROP_AFTER_HANDOFF` 139.

Of the 123 slide holds, the original 17 are inside K=36 and their K=36 quantized replay holds for 10 s. 89 of the 106 newer slide holds are also inside K=36 and were replayed successfully by the search. 17 newer slide holds leave the ±0.25 cap.

81 of the 359 holds contain at least two complete-loss intervals. The search does not stop at the first bilateral return: `basin_ok` requires a closed grip, so an open-finger brush does not hand off.

Continuous actions and 39-D observations for every hold are in `demo_results/diagnostics/raw/legal_recovery_v2/trajectories/`. Those files were not rewritten. Token and residual arrays for the tested 47-prototype candidate are stored beside them as `*__candidate_k47.npz`.

## 4. Release action-gap analysis

`recovery7d_airborne_v3`, confirmed in `controllers/recovery_actions.py`:

- `vx` and `vy` are hand-frame metres per second at ±1 m/s.
- Positive `vz` is hand +z. On the horizontal grasp that direction is world-down, and +1 maps to 3.8 m/s.
- Negative `vz` maps to 1.0 m/s hand-frame, world-up on this grasp.
- `wx`, `wy`, `wz` are radians per second in the `r_des` frame, scaled by 4.
- Grip torque is `+2 + 0.5 * (a + 1) * (-20)` N·m. Action −1 is +2 N·m (open). Action 0 is −8 N·m. Action +1 is −18 N·m.

Phases below are assigned from the action and the contact channels. The nearest K=36 token is recorded and is not the label.

All 44 original release holds, every policy step:

| Phase from the action and contact | Steps | Steps with a K=36 residual outside ±0.25 |
| --- | ---: | ---: |
| Open gripper, no contact, translation under 0.35 | 912 | 793 |
| Reclose and stabilize, bilateral, grip above +0.5 | 292 | 0 |
| Closed reposition while contact is absent or unilateral | 97 | 1 |
| Open gripper, nearly stationary hand, before loss, bilateral | 35 | 35 |
| Open gripper, no contact, horizontal command at least 0.35 | 24 | 21 |
| Open gripper, small translation under 0.40 | 13 | 13 |
| Open gripper, other translation | 5 | 5 |
| Lateral chase with the gripper open while contact remains | 3 | 3 |

Residual components outside ±0.25, counted per step: `vy` 671, `vz` 348, `vx` 59, `grip` 51. Wrist channels do not drive the gap.

Timing: 863 of the uncovered open-no-contact steps are during the first loss. 49 are after a bilateral return, which is the repeated-release portion. All 292 reclose steps are after recontact and already inside K=36. The 35 open-still steps are before the first loss.

What is missing from the 36 prototypes, by the behavior rather than by the nearest token:

1. Open gripper with a nearly stationary hand. All 35 such steps are outside ±0.25. K=36 has no open-still center.
2. Open gripper with a small positive or negative translation. All 13 are outside. The excess is mostly `vy` and `vz`, not a large `vx`.
3. Lateral chase with an open gripper while a finger is still touching. Three steps, all outside. Token 32 is the large `vx = +1` open chase and does not match these smaller, differently signed commands.
4. Airborne chase. Twenty-four open steps with horizontal command at least 0.35 and no contact; 21 are outside. Token 35 is about `[−0.09, 0.56, 0.45, 0, 0, 0, −1]`. The demonstrated chases sit at moderate `vy` with a different `vz`, so the nearest token still leaves `vy` or `vz` beyond 0.25.
5. Reclose and stabilize. Covered by the existing prototypes. None of the 292 steps are outside the cap.
6. Secondary reposition after contact returns. Four open steps and ten closed steps sit after recontact with contact already gone again. One closed step is outside the cap.
7. Repeated release and regrasp. The same open-chase commands recur after a bilateral return. They are the same uncovered open-`vy`/`vz` pattern, not a new wrist pattern.

## 5. Candidate codebook

The existing 36 prototypes were kept as the first 36 centers. Additional centers were chosen only from release steps of TRAIN initial conditions whose id number is not divisible by 5. Held-out initial conditions are `ball_train_v2_0005`, `0015`, `0020`, `0025`, and `0035`. VALIDATION and FINAL_TEST were not used. The previously tested 11 centers were not reused.

Greedy set cover on the fit-set actions that sit outside K=36, rounded to 0.001, produced 11 centers. Candidate size is 47. Residual cap stays ±0.25. Centers, in selection order:

| Extra index | vx | vy | vz | wx | wy | wz | grip | Behavior |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 36 | −0.037 | −0.014 | 0.015 | −0.002 | 0 | 0 | −1 | Open, nearly stationary |
| 37 | −0.046 | −0.154 | −0.429 | −0.004 | 0 | 0 | −1 | Open, upward and slight −vy |
| 38 | −0.106 | 0.297 | 0.019 | −0.003 | 0 | 0 | −1 | Open, +vy |
| 39 | 0.242 | −0.030 | 0.139 | 0 | 0 | 0 | −1 | Open, +vx |
| 40 | −0.003 | 0.036 | −0.705 | −0.004 | 0 | 0 | −1 | Open, strong upward |
| 41 | −0.236 | −0.338 | 0.119 | −0.002 | 0.001 | 0 | −1 | Open, −vx and −vy |
| 42 | 0.576 | −0.006 | −0.283 | −0.004 | 0 | 0 | −1 | Open lateral, upward |
| 43 | 0.160 | 0.347 | −0.204 | −0.004 | 0 | 0 | −1 | Open, +vy and upward |
| 44 | −0.569 | 0.233 | −0.052 | −0.003 | 0 | 0 | −1 | Open, −vx and +vy |
| 45 | 0.075 | −0.291 | −0.241 | 0 | 0 | 0 | +1 | Closed reposition |
| 46 | 0.548 | 0.144 | 0.061 | −0.004 | 0 | 0 | −1 | Open, +vx and +vy |

Slide actions quantize to the same prototypes as under K=36. The extra centers do not move the existing slide tokens. The candidate was not written into `policy_B_init.pt` or any training config.

## 6. Full-sequence representation

On the 44 release holds, with centers fit only on the non-held-out initial conditions:

| Set | Sequences inside ±0.25 at every step |
| --- | --- |
| Fit release trajectories | 33 / 34 |
| Held-out release trajectories | 9 / 10 |
| All 44 | 42 / 44 |
| All 17 slide trajectories | 17 / 17, quantization unchanged |

Two sequences still contain one step outside the cap. Both are step 7 of `chase_lost`, gripper open, no contact:

- `ball_train_v2_0024` (fit): action `[0.101, 0.547, 0.143, −0.003, 0, 0, −1]`. Airborne lateral chase. `vy` residual 0.250. K=36 residual on that step is also outside.
- `ball_train_v2_0025` (held out): action `[0.079, 0.293, 0.316, −0.003, 0, 0, −1]`. Open chase with no contact. `vy` residual 0.264. This initial condition was not allowed to donate a center.

No smaller prefix of these 11 centers covers the fit set. Adding a center from the held-out step would not be a fit-set result.

## 7. Physical replay

Replay starts from the saved snapshot, uses `handoff_z = None`, then runs the nominal handoff and the 10 s and 12 s nominal-only holds. Absolute t=12 is recorded separately.

| Execution | Release holds | Slide holds |
| --- | --- | --- |
| Saved continuous actions | 44 / 44 `SUCCESS_12S` | 17 / 17 `SUCCESS_12S` |
| Existing K=36 | 0 / 44 representable, so none were replayed | 17 / 17 replayed `SUCCESS_12S` |
| Candidate K=47 | 39 / 44 primary 10 s holds | Quantization identical to K=36, so the 17 replays stand |

Of the 39 candidate successes, all 39 also hold for 12 s after handoff, and all 39 have absolute t=12 true. `ball_train_v2_0002 / chase_lost`, which failed under the earlier 11-center set, is `SUCCESS_12S` here (max L∞ 0.230), including the 12 s hold and absolute t=12.

Five release sequences do not come through as candidate successes:

| Trajectory | Candidate result | Notes |
| --- | --- | --- |
| `ball_train_v2_0024 / chase_lost` | Not replayed | One step outside ±0.25 |
| `ball_train_v2_0025 / chase_lost` | Not replayed | One step outside ±0.25. Held out |
| `ball_train_v2_0016 / chase_lost` | `DROP_AFTER_HANDOFF` | Inside the cap (max L∞ 0.248). Saved actions replay to `SUCCESS_12S` twice. `prototype + residual` differs from the saved action by 3.7e-9 and drops during the hold |
| `ball_train_v2_0020 / chase_lost` | `BASIN_NOT_REACHED` | Held out. Inside the cap (max L∞ 0.239). Saved actions replay to `SUCCESS_12S` |
| `ball_train_v2_0035 / chase_lost` | `BASIN_NOT_REACHED` | Held out. Inside the cap (max L∞ 0.244). Saved actions replay to `SUCCESS_12S` |

The three in-range failures are chase holds whose 10 s outcome changes when the decoded command is formed as `clip(prototype + residual)` in float32. The residual cap is already satisfied. A smaller reconstruction error does not make these holds stable. The candidate is not adopted.

Phases that remain unsupported for a discrete policy:

- One airborne open-chase step on `ball_train_v2_0024` (`vy` just over 0.25).
- One open no-contact chase step on held-out `ball_train_v2_0025` (`vy` residual 0.264).
- The 10 s chase holds on `0016`, `0020`, and `0035`, which are inside the cap and still change outcome under decoded replay.

`release_long` on all three of those initial conditions does replay to `SUCCESS_12S` under the candidate. The gap is the `chase_lost` phase, not the final reclose.

The newer 298 holds were not given a second quantized physical replay under this candidate. Their continuous actions are unchanged. 190 of the 236 release holds, old and new together, fall inside the candidate numerically. That count is not a physical-replay result.

## 8. Multi-stage recovery on `ball_train_v2_0021 / release_long`

The saved trajectory is one array of 46 policy steps (20 ms) and 46 observations of dimension 39. It continues through the final closed bilateral grasp. Nothing in the current training code reads these files, so no loader splits them. A later loader has to keep this array as one episode through the nominal handoff: the same observation history, no cut at the first bilateral return, and no new episode at the intermediate touch.

Contact and grip, from observation channels 2 and 3 and the grip action:

| Steps | Duration | Contact | Grip | What the hand is doing |
| --- | ---: | --- | --- | --- |
| 0 | 20 ms | Bilateral | −1 | Already opening |
| 1–10 | 200 ms | None | −1 | Open chase, `vz` rising to about 0.38 then reversing |
| 11–12 | 40 ms | Unilateral | −1 | Still open. A finger touch, not a grasp |
| 13–24 | 240 ms | None | −1 | Chase continues |
| 25 | 20 ms | None | +1 | One close with no contact |
| 26 | 20 ms | None | −1 | Opens again |
| 27–28 | 40 ms | Bilateral | −1 | Both fingers touch while the gripper is still open |
| 29–37 | 180 ms | None | −1 | Open again |
| 38–39 | 40 ms | None | +1 | Close before contact returns |
| 40–45 | 120 ms | Bilateral | +1 | Closed hold through the handoff |

The three losses are real intervals of 200 ms, then 240 ms, then 180 ms, with the gripper held at −1. They are not 20 ms contact blips. They are also not three completed release/regrasp attempts. The only bilateral return before the final hold is 40 ms long and happens while the grip action is still −1, so the basin cannot trigger and the handoff does not run. The close at step 25 never makes contact. The episode is one open chase with two failed contact interruptions and a final close that holds.

Candidate replay of this full sequence is `SUCCESS_12S` (max L∞ 0.238), with the 12 s post-handoff hold and absolute t=12 true.

## 9. New IC generation and physical coverage

Batches `b0004` (80 launches), `b0005` (120), and `b0006` (200) used the near-normal proposal. They added 245 accepted initial conditions. Combined with the earlier 39, v2 has 284. Combined with v1 TRAIN, the TRAIN set has 312.

Accepted v2 states cover all twelve azimuth bins with 23 or 24 entries each, speeds 5.36–9.65 m/s, and both near-normal and a thin tangential tail (6 tangential accepts out of 170 tangential launches). Median pinch is 13.9 mm on the later entries and 4.0 mm on the later misses. The 6 mm line was not moved to produce that coverage.

The legal menu on the new initial conditions found contact-preserving slides, release/regrasp, lateral and airborne chase inside `chase_lost` and `release_long`, and repeated contact loss on 81 holds. Historical wrist reseat and `gravity_inward` produced no 10 s hold. 103 of the 245 new initial conditions have no successful trajectory. Failed trials remain in `search.jsonl`.

## 10. Remaining blockers

- The tested 47-prototype candidate is not the active codebook. Two `chase_lost` steps are still outside ±0.25, and three in-range `chase_lost` replays change the 10 s outcome.
- Under the active K=36 codebook, none of the 44 original release holds are representable. The 17 original slide holds are representable and physically replayed. Seventeen newer slide holds are outside K=36.
- 114 of 284 v2 initial conditions have no 10 s hold from the six-strategy menu. Absence of a hold is not a proof that the initial condition is unrecoverable.
- Decoded `prototype + residual` is not bit-identical to the saved command. On `ball_train_v2_0016 / chase_lost` a 3.7e-9 difference is enough to drop the hold. Those demonstrations are a weak discrete-execution target until the hold survives that decode.
- The Transformer architecture and the K=36 checkpoint were not modified.

TRAIN IC DATASET READY: YES

EXECUTABLE DEMONSTRATION SET READY: NO

CANDIDATE CODEBOOK PHYSICALLY VERIFIED: NO

POLICY TRAINING HAS NOT BEEN STARTED
