# Autonomous recovery entry and exit

The frozen 39-D policy is unchanged. The skill data, the v3 action map, and the 39-D observation are unchanged. The deterioration threshold was not changed. It is still `score > 0.85`.

Pipeline A is the published clean 39-D benchmark: restore a prepared recovery snapshot, run a fixed number of policy steps, then the nominal controller until absolute `t = 12`. That result is 15/24.

Pipelines B and C start in the nominal controller, apply the same disturbance online, and switch to the 39-D policy only when the existing `DeteriorationMeter` latches. The switch keeps `qpos`, `qvel`, `p_des`, `r_des`, and the simulation clock. The policy's previous-action input starts at zero, because the nominal controller does not emit a 7-D token. B then runs the benchmark's fixed step budget. C leaves recovery when the grasp has been continuation-capable for 100 ms:

- both pads in contact
- object height above 0.51 m
- hand-object relative speed at most 0.08 m/s
- no translational or wrist command larger than 0.35
- grip command at least 0

Height alone is not the handoff. After either exit, nominal control runs until absolute `t = 12`. Recovery uses the same clock.

The meter's latch is updated only after the lift reference has been captured. That is the only place the existing code calls `update_mode`. The score itself is recorded on every 20 ms control step, including before that capture.

Wrist, open-finger, and the sliding-suffix cases in the 24-case list are not separate physical disturbances. They are later start indices along a scripted recovery that begins from the `impact_zm6` ball. Autonomously they share that one ball. The 1 mm `slide_dy1` and `slide_dz1` shifts are applied at the verified post-park instant, so their switch is later than the raw impact switch. Airborne uses the sideways carry and the finger opening, including the extra 20 ms of opening after both pads clear. Natural loss uses the verified grasp offset on the seated nominal grasp.

## Detector

On two undisturbed nominal grasps the armed meter never entered recovery.

| Grasp | Max score | Latched | Outcome |
| --- | ---: | --- | --- |
| Approach and lift, no ball | 5.27 before capture; no latch after capture | no | Held, object z = 0.605 m |
| Seated hold, no offset | 0.705 | no | Held, both pads in contact, object z = 0.600 m |

False-entry rate on those grasps: 0/2. The pre-capture peak of 5.27 is an open hand during approach: the contact term is 1 because nothing is grasped yet, and the velocity term is about 4.2. Arming the latch there would enter recovery before a grasp exists. The existing lift-only update does not.

Every applied disturbance did latch. None were missed.

| Disturbance | Entry after the physical event | Score at entry | Contact at entry |
| --- | ---: | ---: | --- |
| Ball, all four impacts | 8–10 ms after contact | 2.09 to 21.83 | Often already lost |
| Ball plus 1 mm shift | 14 ms after the post-park shift; about 94 ms after contact | 1.87 to 1.93 | Contact present, relative speed small |
| Airborne open | 20 ms after the open begins | 2.59 to 3.08 | Both pads clear |
| Natural-loss offset | 20 ms after the offset | 0.95 to 1.01 | Contact still present |

Before each ball release the score sits near 0.17. On the next sample after contact it jumps. For `impact_zm6` the jump is 0.17 to 21.83, with the velocity term about 11 and the angular-velocity term about 9, and both pads off. The position term is only 0.31. The latch is the collision spike, not a settled mis-seat.

The natural-loss latch is the opposite shape. A 14 mm offset moves the position term to about 0.73 while contact remains. The score crosses 0.85 in one control step without a slip.

## Fixed duration and online exit

| Pipeline | Retained to t = 12 |
| --- | ---: |
| A, prepared snapshot | 15/24 |
| B, autonomous entry, fixed budget | 2/24 |
| C, autonomous entry, online exit | 5/24 |

Per family, pipeline C:

| Family | A | C |
| --- | ---: | ---: |
| Impact | 3/4 | 0/4 |
| Sliding | 4/4 | 2/4 |
| Wrist | 4/4 | 0/4 |
| Open-finger | 4/4 | 0/4 |
| Airborne | 0/4 | 3/4 |
| Natural loss | 0/4 | 0/4 |

The two sliding retentions are `slide_dy1` and `slide_dz1`. The three airborne retentions are `air_lat_pos`, `air_fast_pos`, and `air_slow_neg`.

B and C match on the ball cases. Where the switch happens 8–10 ms after contact, both drop during the policy: `impact_zm6` and `impact_z3` at step 29, `impact_phi90` at step 4, `impact_zp6` at step 5. The online rule never gets a stable grasp to hand back. Short fixed budgets that return a few steps earlier (`slide_suffix_36` at step 20, `recatch_suffix_28` at step 28) also lose the object under nominal control. The exit rule is not what loses those skills.

On the two shift cases the online rule returns after 20 policy steps (0.40 s) and the fixed 56-step budget returns later. Both are retained to t = 12.

On airborne, the exit is the difference. C returns after 5 policy steps (0.10 s) and retains three of the four releases. B keeps the policy for the fixed budget and drops all four, at steps 107, 7, 114, and 13. The snapshot benchmark also drops all four. Leaving the policy on after the early regrasp is what loses `air_lat_pos`, `air_fast_pos`, and `air_slow_neg`. `air_lat_neg` falls at step 7 under both B and C, before a stable handoff.

Natural loss enters while the pads are still in contact. C treats that high, contacted grasp as ready and returns after 5 steps. B runs the full 120 steps. Both then drop under nominal control, and the snapshot policy drops as well.

## Failure attribution

Each failed autonomous rollout has one primary label. Suffix rows that share the `zm6` ball share its entry; their B label changes only when the fixed budget hands back before the fall.

| Case | C label | Why this label |
| --- | --- | --- |
| `impact_zm6`, `impact_phi90`, `impact_z3`, sliding suffixes, all wrist, all open-finger | `RECOVERY_POLICY_FAILED` | The meter switched 8–10 ms after contact. The policy then fell through 0.44 m. The same weights retain the post-park shift states and the prepared snapshots. |
| `slide_suffix_36` and `recatch_suffix_28`, pipeline B only | `NOMINAL_AFTER_HANDOFF_FAILED` | The short fixed budget returned one step before that fall, and nominal control did not keep the object. C still falls inside the policy. |
| `impact_zp6` | `PHYSICALLY_UNRECOVERABLE / NO VERIFIED LEGAL RECOVERY` | The snapshot result is already `ESCAPE`. The online policy falls at step 5. |
| `air_lat_neg` | `PHYSICALLY_UNRECOVERABLE / NO VERIFIED LEGAL RECOVERY` | Snapshot and both autonomous pipelines drop. C falls at step 7, with no completed handoff. |
| `air_lat_pos`, `air_fast_pos`, `air_slow_neg`, pipeline B only | `PHYSICALLY_UNRECOVERABLE / NO VERIFIED LEGAL RECOVERY` | The fixed budget stays in the policy and eventually drops. C exits at step 5 and retains them, so this label is for B, not for the online-exit pipeline. |
| All four natural-loss cases | `PHYSICALLY_UNRECOVERABLE / NO VERIFIED LEGAL RECOVERY` | The snapshot drops. A 2.4 s fixed recovery also drops. The 100 ms online return also drops. |

No rollout is `ENTRY_MISSED`, `ENTRY_TOO_LATE`, or `FALSE_ENTRY`. The impact switch is early relative to the prepared snapshot, which is taken after the ball has parked. There is no separate label for an early switch. The direct event after that switch is the policy fall, so those rows are `RECOVERY_POLICY_FAILED`. They are one entry event, not fourteen independent policy failures.

## What would be the smallest change

The threshold is not the problem. Impact scores are far above 0.85, and the seated hold stays at 0.70.

The collision sample is the problem. A switch on that sample hands the policy a grasp with the pads off and a velocity term near 11, and the first policy step also parks the ball. The prepared snapshot, and the shift cases, start after the ball has left. Relative speed is then small, some contact has returned, and the score is still about 1.9 because the orientation term is about 1.15. From that later state the unchanged policy retains the object under both the fixed budget and the online exit.

The smallest change is a short persistence on the existing latch: do not switch on the collision sample. Keep the threshold, and require the score to stay above 0.85 for about 100 ms, with the velocity term no longer dominating, before the control switch. That is the gap between the failing 8 ms entry and the retaining post-park entry. It was not applied in this run.

The online exit should stay. On the two shift retentions it matches the fixed budget. On three airborne releases it keeps a grasp that the fixed 120-step budget later drops. It does not repair natural loss, and it was not adjusted to do so.

## Does 15/24 survive

No. Autonomous entry plus the online exit retains 5/24. Two of those, the 1 mm shifts, are also in the prepared-snapshot 15. The other thirteen prepared successes are the early ball entry, counted once per suffix, and they drop. Three airborne releases are new retentions relative to the snapshot, because the online exit returns in 100 ms instead of running the fixed budget. Natural loss stays at 0/4.

ENTRY DETECTOR IS THE MAIN BOTTLENECK
