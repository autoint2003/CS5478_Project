# Legal K48 controlled-sliding search

Success means one thing: a codebook-legal K48 recovery, then HANDOFF (token 47, no physics step), then the unchanged nominal controller completes the lift and holds the grasp for 12 seconds. Geometric error, wrist return, gravity magnitude, brake completion, and continuous bilateral contact were logged. None of them accepted or rejected a rollout.

This batch does not train, and it does not modify FINAL, checkpoints, physics, the K48 codebook, the nominal controller, `data/`, or `demo_results/diagnostics/raw/k48_recovery_v3`. The old `--controlled-slide` command still uses the one-trial key `id|controlled_slide|0` and the 3 mm `error_not_reduced` gate. This search is a separate resumable log, so those keys cannot skip a new schedule.

Command:

```
python -u training/k48_controlled_slide_search.py
```

Raw traces: `demo_results/diagnostics/raw/k48_controlled_slide_search_v1/search.jsonl`  
Summary: `demo_results/diagnostics/raw/k48_controlled_slide_search_v1/search_summary.json`  
Privileged reference, kept separate: `demo_results/diagnostics/raw/k48_controlled_slide_search_v1/historical_reference.json`  
Verified legal tapes: `demo_results/diagnostics/raw/k48_controlled_slide_search_v1/verified/`

Two resumable runs, wall clock 53.5 s then 33.2 s (about 87 s including environment startup and the historical launch). Search-stage time inside those processes: stage A 32.9 s, stage B 0 s, stage D 23.0 s.

## Coordinate audit

`controlled_slide_action` still steers on world-frame pinch error: world pinch x minus `ref["p_nom"][0]`. After a wrist pitch, world x is not the finger axis.

The new search measures both frames on every step:

- Hand-frame object position `r_h` from `physical_pack`. `r_h.x` is displacement along the finger axis. `r_h.z - 0.099` is depth relative to the nominal finger depth, not a lateral error.
- World pinch position, and `world_ex = pinch_x - p_nom_x`.
- Hand-frame gravity `g_h`, hand-frame relative linear and angular velocity, pad contact counts, normal forces, tangential contact-force sums, and up to two hand-frame contact points per pad.

On `ball_train_v2_0002`, before any recovery motion, `r_h.x` is 10.58 mm and `world_ex` is 10.40 mm. The absolute values already differ. Endpoint error reductions often agree after the wrist returns. They diverge while the wrist is still pitched. On the early-drop traces of `ball_train_v2_0035`, `ball_train_v2_0011`, and `ball_train_v2_0089`, the hand-frame and world-frame absolute-error changes differ by 0.37 mm (`12.97` vs `12.60`, `-22.96` vs `-23.33`, `-172.71` vs `-173.08`). Mechanism labels and schedule choice use the hand-frame quantity.

## What was searched

Each candidate is a nominal-failing ballistic recovery IC. Immediate HANDOFF already fails on these captures (`LOST_GRASP` or `DROP_AFTER_HANDOFF` on the 12 baseline rows that record it).

Legal schedules transcribe the CENTER-6 physical phases into codebook actions. Privileged `freeze`, recenter, latch, and controller-target writes are not used. `+3 rad/s` for 0.70 s with grip closed is the `+120°` secure rotation and is representable. The slip phase is wrist rate 0 and `grip_action(-5)` (about `-5 N·m`). The opposite family uses `-1 rad/s`, because `-3 rad/s` is outside the codebook. A schedule is dropped before rollout if any segment has L-inf distance above 0.25 to the codebook. None of the five seeds were illegal.

| Schedule | Phases |
|---|---|
| `center6_pos` | `+3 rad/s` 0.70 s, slip 0.70 s, brake 0.40 s, return `-1 rad/s` 0.70 s |
| `center6_neg` | `-1 rad/s` 1.20 s, slip 0.70 s, brake 0.40 s, no return |
| `short_pos` | `+3 rad/s` 0.40 s, grip action `-0.48` for 0.48 s, brake 0.16 s |
| `short_neg` | `-1 rad/s` 0.80 s, grip action `-0.48` for 0.48 s, retighten 0.16 s |
| `no_return_pos` | `+3 rad/s` 0.70 s, slip 0.70 s, retighten 0.16 s, no return to R0 |
| deterministic baseline | existing `controlled_slide_action`, full tape, 12 s check, 3 mm gate not applied |

Handoff is tested at the recorded phase cuts: after rotation (`reorient`), during or after the slip (`relax`), after the brake, after retightening, and after return. If the cylinder is already falling, the cut is one step before the fall. A real fall ends the recovery rollout. It does not skip the 12 s check of the prefix that still had the object.

Contact state orders the batch. Bilateral pad contact is searched first. Temporary unilateral or absent contact does not skip the IC. This batch still rolled `ball_train_v2_0010`, `k48r_valid_00019`, and `k48r_valid_00023`, which were not bilateral at the snapshot.

Trial keys include the schedule name and handoff step (`a|id|center6_pos|return|125`). They do not collide with `id|controlled_slide|0`.

## Results

42 log rows, 39 nominal verifications, 18 distinct ICs, 12 `SUCCESS_12S` rows, **11 distinct recovered ICs** (8 train, 3 valid). Zero failures during recovery. 27 outcomes failed after handoff (`DROP_AFTER_HANDOFF` 21, `LOST_GRASP` 6). Three baseline tapes had no actions. Stage B ran 0 local perturbations: every IC that had not already succeeded had dropped on its first full schedule, and perturbation only runs on a completed non-drop trace.

| IC | Split | Sign of `r_h.x` | Winning schedule | Handoff | Hand-frame correction | Mechanism |
|---|---|---|---|---|---|---|
| `ball_train_v2_0002` | train | +10.6 mm | `center6_pos` | return | 2.90 mm | controlled slide |
| `ball_train_v2_0012` | train | −12.0 mm | `short_neg` | retighten | 3.66 mm | controlled slide |
| `ball_train_v2_0031` | train | +11.3 mm | `short_pos` | relax | 2.25 mm | controlled slide |
| `ball_train_v2_0036` | train | +14.1 mm | `no_return_pos` | retighten | 2.09 mm | controlled slide |
| `ball_train_v2_0038` | train | positive | `center6_pos` transfer | return | 2.14 mm | controlled slide |
| `ball_train_v2_0053` | train | positive | `center6_pos` transfer | return | 2.90 mm | controlled slide |
| `ball_train_v2_0070` | train | positive | `center6_pos` transfer | return | 5.70 mm | controlled slide |
| `ball_train_v2_0087` | train | positive | `center6_pos` transfer | return | 1.85 mm | controlled slide |
| `k48r_valid_00004` | valid | −9.2 mm | deterministic baseline | recorded end | not stored on baseline row | baseline closed loop |
| `k48r_valid_00009` | valid | +10.6 mm | `center6_pos` | return | 2.84 mm | controlled slide |
| `k48r_valid_00036` | valid | +9.4 mm | deterministic baseline | recorded end | not stored on baseline row | baseline closed loop |

`ball_train_v2_0031` also succeeds on the transferred `center6_pos` return tape (1.33 mm). That is the same IC, counted once.

Accepted traces replay from the saved capture through `step_legal` and the codebook. On the success rows, `qpos_error` is 0, `phase_restored` is true, `fraction_within_0.25` is 1, and `nominal_elapsed` is 12 s.

### Signs

Positive finger-axis offset: 9 distinct successes (`center6_pos`, `short_pos`, `no_return_pos`, the positive-sign transfer, and the positive baseline `k48r_valid_00036`). Negative finger-axis offset: 2 distinct successes (`short_neg` on `ball_train_v2_0012`, and the deterministic baseline on `k48r_valid_00004`). The opposite-sign open-loop family often drives the cylinder outward by tens of millimetres. Sign is not recovered by negating a positive pitch command; the legal negative rate is `-1 rad/s`, not `-3 rad/s`.

### Controlled sliding versus other motion

Of 30 traced rollouts, 14 meet the measured sliding label: during the relax window both pads stay in contact at least half the time and `|r_h.x|` changes by at least 0.5 mm. 16 are `other`, almost all large outward motions or early drops. No success trace was labeled recapture or whole-hand transport. Every legal-phase success that stored a trace is `controlled_slide`. The two baseline successes are the old closed-loop law; their contact window was not written into the log row.

Non-contact and unilateral snapshots were not skipped. `ball_train_v2_0010`, `k48r_valid_00019`, and `k48r_valid_00023` lost the object within 4–5 steps (0.08–0.10 s). Handoff one step before that fall still failed the nominal 12 s hold. Those ICs were attempted and failed physically.

### Geometrically imperfect states that still hold for 12 s

The old 3 mm correction gate would have rejected these successes. They passed the nominal 12 s hold:

- `ball_train_v2_0002`, `center6_pos`, return, 2.90 mm
- `ball_train_v2_0031`, `short_pos`, handoff during the slip window, 2.25 mm
- `ball_train_v2_0031`, transferred `center6_pos`, 1.33 mm
- `ball_train_v2_0036`, `no_return_pos`, wrist not returned to R0, 2.09 mm
- `ball_train_v2_0038`, transfer, 2.14 mm
- `ball_train_v2_0053`, transfer, 2.90 mm
- `ball_train_v2_0087`, transfer, 1.85 mm
- `k48r_valid_00009`, `center6_pos`, 2.84 mm

No success in this batch had a worse finger-axis offset at the end of the tape than at the start. Restoring the wrist is not required: `no_return_pos` succeeds on `ball_train_v2_0036`. Handoff during the slip, before retighten and before return, succeeds on `ball_train_v2_0031`.

### Failures that stayed failures

`ball_train_v2_0035`, `ball_train_v2_0011`, `ball_train_v2_0077`, and `ball_train_v2_0089` were handed off on a prefix that had not yet fallen. Nominal still dropped or lost the grasp. Early contact loss on `ball_train_v2_0010`, `k48r_valid_00019`, and `k48r_valid_00023` likewise failed after handoff. Removing the 3 mm gate did not turn those trajectories into successes.

The deterministic baseline, with the 3 mm gate removed, still failed the 12 s hold on `ball_train_v2_0002` (`LOST_GRASP`), `ball_train_v2_0012`, `ball_train_v2_0031`, `ball_train_v2_0035`, `ball_train_v2_0036`, and `k48r_valid_00009`. The legal phase menu recovered five of those six. It succeeded on the two valid ICs where the baseline itself already held for 12 s (`k48r_valid_00004`, `k48r_valid_00036`).

## Historical CENTER-6 reference

`demo_results/diagnostics/ballistic_recovery_transfer/raw/snap_early.pkl` exists. It is the ballistic-demo snapshot (`phase`, `ctrl`, `p_des`, `mocap_pos`, and related keys). It has no K48 `capture` / `ref`, so it was not passed through `legal_verify`. That privileged tape is not evidence of K48 success.

A fresh K48-environment launch at 6 m/s, zero vertical offset, azimuth 0, was run as a diagnostic and labeled as such. Category: `impact_without_recovery_entry`. Reason: the cylinder was hit and the position entry did not persist. There was no recovery snapshot on which to replay the legal CENTER-6 schedule. The known historical IC was not inserted.

## Efficiency

| Stage | Verifications | New distinct successes | Time |
|---|---|---|---|
| A coarse phase menu, including the deterministic baseline and handoff cuts | 31 | 7 | 32.9 s |
| B local duration scaling | 0 | 0 | 0 s |
| D transfer of `center6_pos` onto other positive-offset train ICs, full replay | 8 | 4 | 23.0 s |

About 0.13 distinct recovered ICs per second of the two process walls, or about 0.20 per second of stage A+D verification time. Transfer produced four new train ICs that stage A had not opened (`ball_train_v2_0038`, `0053`, `0070`, `0087`) and one repeat success on `ball_train_v2_0031`. One transfer lost the grasp (`ball_train_v2_0077`) and one dropped after handoff (`ball_train_v2_0089`).

These 11 ICs are eligible in the sense used here: genuine ballistic recovery candidates, immediate handoff fails, and at least one legal tape reaches `SUCCESS_12S` on replay from the saved snapshot. They are not merged into `data/`. The previous controlled-sliding training entry under `data/controlled_sliding` is unchanged.
