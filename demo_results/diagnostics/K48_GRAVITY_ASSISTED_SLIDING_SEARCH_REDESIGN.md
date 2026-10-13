# K48 gravity-assisted sliding search

Success means one thing: after a legal K48 recovery and handoff, the unchanged nominal controller finishes the lift and holds the grasp for 12 seconds. A finger-axis correction of 0 mm, 1 mm, or 10 mm is the same kind of success when that hold occurs.

This run did not train, did not open FINAL, and did not edit `demo_results/diagnostics/raw/k48_recovery_v3` or the codebook. New files are under `demo_results/diagnostics/raw/k48_inward_sliding_search_v1`.

Command:

```
python -u training/k48_inward_sliding_search.py
```

The process wall time was 188 s. Replaying the 91 gated tapes took 129.7 s. The staged search inside that process took 35.9 s (stage A 16.0 s, stage B 1.6 s, stage C 3.6 s).

## 1. What the old search discarded

Source log: `demo_results/diagnostics/raw/k48_recovery_v3/inward_reseat/search.jsonl` (105 rows).

| Logged reason | Count | What happened |
| --- | ---: | --- |
| `error_not_reduced` | 91 | A tape was built, then rejected before the 12 s check because the hand-frame offset fell by less than 3 mm or the final contact was not bilateral |
| `proposal_recovery_failed` | 8 | The object fell during recovery. Physical stop, not a geometric gate |
| `SUCCESS_12S` | 6 | Already passed the 3 mm gate and the 12 s hold |

An earlier combined log, `k48_recovery_v3/search.jsonl`, has a different inward controller: 22 `not_applicable`, 7 `LOST_GRASP`, 6 `DROP_AFTER_HANDOFF`, 1 `proposal_recovery_failed`, and no success. Those rows have no saved action arrays. The counts below are from re-simulating the current gated controller on the 105 category-log states. Gate labels were recorded in `training/rebuild_k48_recovery.py` without changing the wrist or grip command. `k48r_valid_00004` still produced 77 steps, matching the logged tape.

Gates observed on those 105 rollouts:

| Gate | Trajectories |
| --- | ---: |
| Offset increased by 2 mm during pitch or relaxation | 56 |
| Gravity projection reached the 6 m/s² slip threshold | 48 |
| Relaxation stopped because the 0.70 s window ended | 46 |
| Both fingers clear | 1 |
| Unilateral contact for 0.10 s | 1 |
| Offset entered the 3 mm band during relaxation | 1 |
| Gravity threshold never reached before the 90° cap | 0 |

A rollout can carry more than one label. The 90° gravity cap did not fire. Retightening was a fixed 0.16 s (8 steps) on 101 of 105 tapes. It was not a separate rejection code. Every finished tape then faced `error_not_reduced` if the offset had not fallen by 3 mm.

## 2. Replaying the rejected tapes

All 91 `error_not_reduced` tapes were restored from the saved capture, replayed with `step_legal`, and handed to the unchanged nominal controller. The 3 mm rule was not applied. Log: `k48_inward_sliding_search_v1/old_gate_replay.jsonl`.

| Physical result | Count |
| --- | ---: |
| `SUCCESS_12S` | 32 |
| `DROP_AFTER_HANDOFF` | 53 |
| `LOST_GRASP` | 6 |
| Drop during recovery | 0 |

Every one of the 32 successes has `qpos_error = 0`, the phase restored, `fraction_within_0.25 = 1`, lift completed, and `nominal_elapsed = 12 s`. Every one had been rejected only after gravity was ready and the 0.70 s relaxation window finished. Logged finger-axis correction on those 32 is 0.64 mm to 2.63 mm. All 32 are under 3 mm.

The 56 tapes that aborted because the offset grew did not become successes. Their physical results are 50 `DROP_AFTER_HANDOFF` and 6 `proposal_recovery_failed`. Removing the 2 mm abort, the unilateral timer, or the gravity cap is not supported by these replays: the gravity cap never occurred, and the offset-growth tapes still fail the 12 s task when handed off as they stand.

The 32 new holds, by split: 29 train, 3 valid.

Valid: `k48r_valid_00004` (0.64 mm, 77 steps), `k48r_valid_00009` (1.56 mm, 61 steps), `k48r_valid_00036` (1.13 mm, 61 steps).

Train includes `ball_train_0027` (1.27 mm) and `k48r_train_00032`, `k48r_train_00033`, `k48r_train_00092`. The full list is the `SUCCESS_12S` rows in `old_gate_replay.jsonl`.

Six earlier train successes remain valid under the old log because they already cleared 3 mm and held for 12 s: `ball_train_v2_0012`, `ball_train_v2_0070`, `ball_train_v2_0196`, `ball_train_v2_0268`, `k48r_train_00071`, `k48r_train_00100`. They were not part of the 91.

These 32 were not copied into the training manifest.

## 3. Historical 6.5 m/s, −6 mm reference

A new impact was launched in the K48 environment: azimuth 42°, speed 6.5 m/s, axial offset −6 mm, tangential angle 0, delay 0. Snapshot: `k48_inward_sliding_search_v1/diagnostic/hist_6p5_m6mm.pkl`. It is not a dataset initial condition.

The impact is a valid recovery capture. Entry contact is 1 left / 4 right, position error 17.5 mm, rotation error 0.88 rad. Immediate handoff is `DROP_AFTER_HANDOFF`.

The historical phase structure, executed with legal K48 actions (`wx = 1.12 rad/s`, `wy = +2 rad/s`, grip `+1` for 0.40 s, then grip `−0.48` with the wrist stopped), loses both fingers at step 14 (0.28 s). Hand-frame x goes from 17.8 mm to 22.1 mm. The along-finger gravity component reaches only −5.0 m/s². Prefix handoffs at steps 4, 8, and 13 are also `DROP_AFTER_HANDOFF`.

Under this stack the historical command does not reseat that launch. The old visual-audit numbers are not a current K48 success.

## 4. Bounded search

Implementation: `training/k48_inward_sliding_search.py`.

The menu is eight phase structures, each with both pitch signs. A schedule is rolled out only when every segment is inside the K48 codebook at L∞ ≤ 0.25. Handoff is tried at the last completed phase before a fall, then the nominal 12 s check is the only accept/reject test. Ten nominal-failing ballistic ICs were used: the four valid and six train rows in the old inward log with the largest |r_h.x|. That choice spends the budget on large offsets. The 32 holds above sit at smaller offsets and were already decided by the replay.

Ten of the sixteen sign variants are illegal on this codebook, 100 skipped rows:

- `wy = −2 rad/s` (action −0.50) is outside ±0.25. Negative pitch at the historical rate cannot be the mirror of `wy = +2`.
- Support roll alone (`wx = 1.12`, `wy = 0`) is outside ±0.25, so roll-then-pitch and pitch-then-roll were skipped.
- Grip `−0.48` together with a nonzero wrist is outside ±0.25, so relaxation during the compound rotation was skipped.

Legal schedules that were rolled out: simultaneous support-roll plus positive pitch, the same pitch with grip `grip_action(−5)` or with a brake, slow pitch at ±1 rad/s, and a return at −1 rad/s after the positive compound.

Stage A verified 25 handoffs. Three succeeded, all valid, all under 3 mm, all after retightening:

| IC | Schedule | Inward change at handoff | Relaxation window |
| --- | --- | ---: | --- |
| `k48r_valid_00009` | simultaneous `wy = +2` | 2.06 mm | both pads on, r_h.x 9.80 → 8.61 mm, mean g_h.x = −6.81 |
| `k48r_valid_00036` | simultaneous `wy = +2` | 1.73 mm | both pads on, r_h.x 8.46 → 7.72 mm, mean g_h.x = −6.81 |
| `k48r_valid_00004` | slow pitch `wy = −1` | 1.46 mm | both pads on, r_h.x −9.12 → −7.79 mm, mean g_h.x = +7.04 |

On `k48r_valid_00004` the positive compound drives r_h.x the wrong way (about −135 mm in the trace) and drops after handoff. The legal negative pitch at 1 rad/s holds both pads and then succeeds. The two signs are not interchangeable.

Stage B lengthened and shortened pitch and relaxation on three large-offset train ICs that stayed in the hand long enough to hand off. All 8 checks are `DROP_AFTER_HANDOFF`.

Stage C copied the successful positive compound onto another valid IC, choosing the pitch sign from that IC’s hand-frame offset. `k48r_valid_00036` succeeds again (same 1.73 mm trace). Targets that need `wy = −2` were not executed, because that action is illegal.

No verified trajectory in this search succeeds while the finger-axis offset is getting worse. The tapes that worsen the offset fail after handoff.

Immediate handoff on all 10 search ICs fails (`LOST_GRASP` or `DROP_AFTER_HANDOFF`), from `candidates.jsonl`. Old `slide_50` on the six train ICs is `DROP_AFTER_HANDOFF` or `DROP_DURING_RECOVERY`. The four valid ICs have no `slide_50` row. `slide_50` is not this controller.

Failures in the 34 new 12 s checks: 30 after handoff, 0 during recovery. Recovery drops were avoided by handing off before the fall; those early handoffs still failed the nominal task.

Verified action files for the three search successes:

- `k48_inward_sliding_search_v1/verified/k48r_valid_00009__hist_simultaneous_pos__retighten.npz`
- `k48_inward_sliding_search_v1/verified/k48r_valid_00004__slow_pitch_neg__retighten.npz`
- `k48_inward_sliding_search_v1/verified/k48r_valid_00036__hist_simultaneous_pos__retighten.npz`

## 5. What is eligible, and what was not added

A training IC from this experiment needs a real ballistic capture, a failed immediate handoff, and one legal replay that holds for 12 s.

That bar is met by the 32 replay successes (29 train, 3 valid) and by the 6 train successes already in the old inward log. They were not written into `k48_recovery_v3` supervised or GRPO manifests.

The 6.5 m/s, −6 mm diagnostic launch fails both immediate handoff and the legal compound replay. It was not inserted.

Nominal-solvable states were not counted as recovery ICs.
