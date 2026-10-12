# Nominal-pose return and long-horizon handoff

The K=33 lateral-chase policy was replayed with the frozen v3 action map. Nothing was trained. `config/geometry_switch.yaml` and `controllers/geometry_switch.py` were not edited. The basin below is a study measurement. It is smaller than the frozen 0.35 rad exit, and it is not yet the controller rule.

A recovery counts as successful only when the hand is back inside the original nominal orientation, the object is back inside the nominal pinch region, an explicit handoff restores the original orientation setpoint, and the unchanged nominal controller then holds the grasp for 10 s and 12 s after that handoff. Holding a tilted grasp until the recovery step cap is not success.

## Nominal reference

The reference is the settled lift grasp immediately before the disturbance. For an impact, that is the hand pose and `r_des` at ball release. For an offset or an air pulse, it is the seated grasp before the disturbance is applied. It is not recomputed from the recovered hand.

On an undisturbed lift, the geometry switch’s early capture and this settled grasp differ by 0.041 rad of hand orientation and 0.074 mm of pinch position. The audit uses the settled grasp. Over the next 2.00 s of nominal hold (101 samples):

| quantity | maximum | 95th percentile |
| --- | ---: | ---: |
| hand orientation error | 0.0292 rad (1.67 deg) | 0.0292 rad |
| object-in-hand orientation error | 0.0018 rad | — |
| pinch position error | 0.031 mm | 0.030 mm |
| object-hand relative speed | 4.0 mm/s | — |

The saved pinch point is `p_nom_H = (0.24, 0.00, 96.31) mm`.

## Chosen basin

The orientation gate is four times the hold peak:

`0.0292 + 3 × 0.0292 = 0.117 rad` (6.70 deg).

That contains the undisturbed hold and the four recatches that later held for 12 s (handoff error 0.056–0.059 rad). It rejects the `impact_zp6` hand at the step cap (1.747 rad, 100 deg) and the old impact handoffs (0.78 rad and 1.62 rad).

The position gate is the hold peak plus 3 mm:

`0.031 mm + 3 mm = 3.03 mm`.

The 3 mm term is the smallest fixed margin used in the rule. The hold itself moves 0.03 mm, so a bitwise return to the reset point is not required. The four long-horizon successes land at 2.2–2.5 mm. `impact_zp6` after recatch is at 19.6 mm. The old air handoffs are at 6.0 mm.

Relative speed must be at most 0.020 m/s, five times the 4.0 mm/s hold peak and below the frozen 0.08 m/s exit. Grip must be the secure closed command. All of these must persist for 0.100 s. That window only rejects a one-sample crossing. Success is the 10 s and 12 s nominal interval after handoff.

At a valid handoff, `r_des` is set back to the saved nominal setpoint. It is not copied from the current hand.

## impact_zp6, K=33

Same episode as the inspected video: 6.5 m/s, `z_off = +6 mm`, `phi = 0`.

| event | time | hand error | pinch error |
| --- | ---: | ---: | ---: |
| ball impact | 2.852 s | — | — |
| geometry entry | 2.900 s | — | — |
| both pads lost | 2.920 s | 0.015 rad | 11.2 mm |
| recatch and reclose | 2.960 s | 0.023 rad | 17.1 mm |
| first 0.100 s of closed bilateral contact | 3.040 s | 0.014 rad | 19.6 mm |
| hand error passes 0.117 rad | 3.32–3.72 s | 0.092 then 0.203 rad | 19.6 mm |
| marker used for the growing rotation | 3.840 s | 0.237 rad | 19.7 mm |
| 300-step cap, mode flag to nominal | 8.900 s | 1.747 rad | 20.7 mm |

The two lateral-chase tokens (token 32, grip open) are the steps at 2.900 s and 2.920 s. From 2.960 s onward the policy emits the closed hold, almost entirely token 0, with one token 29 at 2.980 s. The wrist component stays near −0.06 to −0.08, about −0.24 to −0.32 rad/s, on every one of those steps. `r_des` integrates that command. The hand follows it, from 0.014 rad at 3.040 s to 1.747 rad at 8.900 s, and the hand climbs to object height 0.924 m. The object-in-hand orientation stays near 0.23–0.25 rad the whole time, inside the old 0.35 rad relative check, so that check cannot see this drift.

A. The hand is inside the 0.117 rad orientation basin from recatch through about 3.32 s. It never enters the joint basin: the pinch error is already 19.6 mm, and the relative speed at 3.040 s is 0.029 m/s. After 3.72 s the hand leaves the orientation basin and does not return.

B. There is no valid handoff on this episode.

C. The post-recatch hold is what prevents restoration. Its repeated wrist command walks `r_des`, and the hand, off the saved orientation. The pinch error stays at 20 mm while that happens, so stopping the wrist command would still leave the object outside the 3.03 mm region.

At the cap the recovery wrist command is `w = (−0.009, −0.295, −0.052) rad/s` and `r_des` is already 1.774 rad from the saved setpoint. Twelve seconds of nominal control then sets the angular command to zero and leaves that setpoint where recovery put it. The object is still held at 20.900 s, height 0.920 m, hand error 1.752 rad.

## Matched branches from the same post-recatch state

The fork is the live state at 3.040 s. No teleport, no position reset, no clock reset.

| branch | what ran | orientation at the decision | pinch error | 10 s nominal hold | 12 s nominal hold | classification |
| --- | --- | --- | ---: | --- | --- | --- |
| A. Current policy | token-0 hold through the step cap, then nominal with the drifted `r_des` | 1.747 rad at 8.900 s | 20.7 mm | held, wrong pose | held, wrong pose | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| B. Nominal immediately | nominal from 3.040 s, `r_des` left untouched | 0.014 rad | 19.6 mm | held, pinch still 20 mm | held through 15.040 s | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| C. Orientation return, then handoff | legal v3 wrist servo, secure grip, no translation | 0.001 rad at 5.36 s | 19.8 mm | no handoff | no handoff | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| D. Privileged return | same legal actions, plus a hand-frame correction from the simulator pinch error | 0.010 rad at 7.04 s | 19.9 mm | no handoff | no handoff | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |

Branch B shows that the hand orientation is already nominal at recatch, and that nominal control can keep this grasp for 12 s. The object height rises from 0.478 m to 0.595 m under the normal lift. The pinch stays near 20 mm, so the handoff is not a return to the original grasp. The angular command at the end of that window is zero. `r_des` remains 0.035 rad from the saved setpoint; it was not overwritten with the recovered hand.

Branch C drives the hand error from 0.017 rad down to 0.001 rad. The pinch error does not move. The object descends to the 0.440 m abort height while both pads are still in contact, so the servo stops without a handoff.

Branch D also restores the hand orientation, to 0.010 rad, and the contacts stay on (6 and 7). The pinch error stays at 19.9 mm for the whole 4 s. Commanding a hand translation through the closed fingers moves the object with the hand. The original pinch point is not recovered.

## Re-evaluation under the new definition

Physical recatch means the object is bilaterally held after recovery has run. Pose restoration means the study basin persisted for 0.100 s. The 10 s and 12 s columns start at that handoff and use only the nominal controller.

| case | recovery success? | recatch time | nominal orientation restored? | orientation error at handoff | position error at handoff | handoff time | 10 s nominal-only stable? | 12 s nominal-only stable? | failure classification |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| impact_zm6 | no | — | no | — | — | — | no | no | DROP_DURING_RECOVERY |
| impact_z3 | no | 3.42 s | no | — | — | — | no | no | DROP_DURING_RECOVERY |
| impact_zp6 | no | 3.04 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| wrist reseating | no | 3.28 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| inward sliding | no | 3.96 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| open-finger recatch | no | 3.56 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| train_dz13.5 | yes | 4.70 s | yes | 0.057 rad | 2.46 mm | 4.876 s | yes | yes | SUCCESS_12S |
| nl_dz14.6 | no | 4.70 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| nl_dvz_p | yes | 4.70 s | yes | 0.058 rad | 2.45 mm | 4.876 s | yes | yes | SUCCESS_12S |
| dz13.2 | yes | 4.70 s | yes | 0.056 rad | 2.22 mm | 4.876 s | yes | yes | SUCCESS_12S |
| dz14.2 | no | 4.70 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| dvz_m | yes | 4.70 s | yes | 0.059 rad | 2.49 mm | 4.876 s | yes | yes | SUCCESS_12S |
| h_dz14.00 | no | 4.72 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| h_dz14.25 | no | 4.70 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| h_dy | no | 4.72 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| h_dx | no | 4.70 s | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| air_lat_pos | no | held, no release | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |
| air_fast_pos | no | held, no release | no | — | — | — | no | no | RECATCh_SUCCEEDED_BUT_POSE_NOT_RESTORED |

Separate from the table:

- Physical hold: the four `SUCCESS_12S` rows, `impact_zp6`, all three scripted skills, both air cases, and the six natural-loss rows labeled pose-not-restored. `impact_zm6` and `impact_z3` lose the object while recovery is still running.
- Pose restoration and a real handoff: only `train_dz13.5`, `nl_dvz_p`, `dz13.2`, and `dvz_m`.
- Those four then remain bilateral under nominal control for both 10 s and 12 s. No delayed drop appeared in any completed nominal window. The `impact_zp6` windows that were forced open also held for 12 s, including the window that starts with the hand already 1.75 rad away.

Where the old exit would have fired, the hand error was 0.78 rad on `impact_zm6` and 1.62 rad on `impact_z3`, with pinch errors of 12.6 mm and 12.7 mm. On `air_lat_pos` and `air_fast_pos` the hand error at that old exit was 0.016 rad and 0.017 rad, already inside the new orientation basin, while the pinch error was 6.0 mm. Continuing recovery after that point is what grows the air-case hand error to 0.33 rad and 0.38 rad. The same pattern appears on `nl_dz14.6`, `dz14.2`, and the four excluded natural-loss states: an early hand error of about 0.01 rad, a pinch error of 8–14 mm, then a later hand error near 0.31 rad because the new basin never opens and the hold token keeps turning the wrist.

The three scripted skills finish still holding, with hand errors 0.78 rad, 0.93 rad, and 1.60 rad and pinch errors 10–13 mm.

The frozen exit was left unchanged. The measured replacement for its 0.35 rad orientation term is 0.117 rad, tied to the settled nominal hold, and it still has to be paired with the 3.03 mm pinch region. Widening 0.35 rad would accept the failure this audit is measuring.

CURRENT RECOVERY SUCCESS DEFINITION REMAINS INADEQUATE
