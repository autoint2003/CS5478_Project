# Vertical velocity-cap threshold

The natural-loss recatch does not need an unlimited downward command. On the saved 13.5 mm snapshot a downward world-z command limit of 2.3 m/s is the first value that regrasps and holds. That value is one state. A cylinder that starts 1.5 mm or 3 mm higher is still missed at 3.5 m/s and is caught at 3.6 m/s, which is also where the hand's peak acceleration matches the uncapped reference. The proposed downward limit is 3.8 m/s. The hand never reaches that speed. The fastest downward speed on the uncapped catch is −1.77 m/s.

The action stays 7-D velocity. Nothing here is a force or acceleration command. `vx` and `vy` stay at ±1.0 m/s. Upward `vz` stays at +1.0 m/s. Wrist rate stays at the existing ±4 rad/s bound and the intercept sets it to zero. Grip stays +2 N·m while opening and −18 N·m to reclose.

## Setup

Every chase reloads `results/diagnostics/raw/uncapped_cartesian_airborne_chase/natural_loss_13_5.npz`. After reload, before any command: excess 25.44 mm, object `vz` −0.0459 m/s, aperture 13.77 mm, both pad counts 0, no robot-cylinder contact. The pinch used by the intercept is the seated pinch stored with that audit, `[0.22, 0.0, 97.58]` mm, not the loss-time relative pose. Floor and table contacts are off, so the window is not cut by the table.

The law is the one that caught this state with no Cartesian cap:

    p_des = p_object − R p_pinch
    v_des = v_object + 200 (p_des − p_hand)

Gains are unchanged (`kp_pos` 1400, `kd_pos` 80, and the same orientation and nullspace terms). Torques are clipped to the actuators, ±87 N·m on joints 1–4 and ±12 N·m on joints 5–7. The arm has position limits and no joint-velocity limit. The fingers open from the saved 13.8 mm under +2 N·m. Nothing is teleported and hand velocity is not written. Only the downward part of `v_des` changes between runs.

The uncapped-vz reference in this study still clips `vx` and `vy` to ±1 m/s and upward `vz` to +1 m/s. Downward `vz` is not clipped. Its peak command is −5.34 m/s. Peak hand acceleration is −47.2 m/s², the same actuator-limited spike as the earlier fully uncapped intercept.

A run counts only if the chase makes bilateral contact, the fingers reclose, and both pads stay on the cylinder for 1 s afterwards. Gap closure alone is not success. "Stay on" here means both pads are present for at least 95% of that second, the log covers the whole second, and at the end the excess is under 20 mm with `|v_rel,z|` under 0.08 m/s. The longest unbroken stretch is about 0.72 s on the successful runs, including the uncapped reference, because a few 2 ms contact samples drop out. The cylinder does not leave the hand. A zero-dropout rule would also reject the uncapped catch.

The loss snapshot is at absolute time 10.610 s. The 1 s hold ends near 11.7 s. These runs stop there. They are not an absolute-t=12 score.

## Nominal sweep

Peak hand `vz`, peak `a_hand,z`, and peak torque below are taken up to the closest approach, or up to recontact when that is earlier. After a miss the position error keeps growing and the hand later exceeds the command cap. That late speed is not the chase.

| vz cap (m/s) | peak hand vz (m/s) | peak hand az (m/s²) | min gap (mm) | recontact | v_rel,z at contact (m/s) | 1 s hold | peak torque (N·m) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | −0.73 | −22.5 | 22.7 | no | — | no | 33 | miss |
| 1.25 | −0.92 | −26.2 | 19.7 | no | — | no | 43 | miss |
| 1.50 | −1.10 | −30.0 | 15.6 | no | — | no | 53 | miss |
| 1.75 | −1.19 | −33.8 | 11.3 | yes, 0.12 s | −0.01 | no | 63 | touch, then lost |
| 2.00 | −1.35 | −37.5 | 4.9 | yes, 0.13 s | +0.07 | no | 73 | held 0.78 s, then slipped |
| 2.10 | −1.43 | −39.0 | 1.3 | no | — | no | 77 | near miss, no contact |
| 2.20 | −1.41 | −40.4 | 0.5 | no | — | no | 81 | near miss, no contact |
| 2.30 | −1.42 | −41.8 | −0.0 | yes, 0.12 s | +0.18 | yes | 85 | hold, this state only |
| 2.40 | −1.43 | −43.2 | −0.5 | yes, 0.12 s | +0.13 | yes | 87 | hold |
| 2.50 | −1.45 | −44.6 | −0.9 | yes, 0.12 s | +0.16 | yes | 87 | hold |
| 3.00 | −1.60 | −47.3 | −2.3 | yes, 0.12 s | +0.15 | yes | 87 | hold; accel matches |
| 3.60 | −1.70 | −47.3 | −3.1 | yes, 0.11 s | +0.10 | yes | 87 | hold, and the vertical shifts below |
| 3.80 | −1.73 | −47.3 | −3.3 | yes, 0.11 s | +0.10 | yes | 87 | proposed limit |
| uncapped | −1.77 | −47.2 | −3.5 | yes, 0.11 s | +0.07 | yes | 87 | reference |

Positive `v_rel,z` means the hand is moving downward faster than the cylinder. The raw intercept wants about −5.3 m/s at the first step on every cap. The tested limit clips that command. By the time the gap is smallest the position error has shrunk and the command is about −1.4 to −1.7 m/s, so the cap is an early-torque setting, not the speed at contact.

At 1.0 m/s the mean hand acceleration over the first 80 ms is −9.9 m/s². The cylinder is in free fall at −9.81 m/s², so the relative acceleration is about zero and the 25 mm gap stays above 22 mm. Shoulder torque peaks at 33 N·m of 87. The actuators are not the limit. After the object has left, the same position loop winds the hand up to about −2.5 m/s. That is the late speed, not a violation of the cap during the chase.

From 1.0 to 2.2 m/s the gap falls steadily: 22.7, 19.7, 15.6, 11.3, 4.9, 1.3, 0.5 mm. Fingers open on every cap (peak aperture 41–44 mm, from 13.8 mm). At 1.75 m/s both pads touch and the bilateral stretch lasts 0.08 s, then the grasp is lost and the arm runs into the stops. At 2.0 m/s the reclose holds for 0.78 s and then slips (end counts 1/1, excess 26 mm). At 2.1 and 2.2 m/s the hand reaches the pinch height and the fingers are open, but no pad touches the cylinder.

The first 1 s hold on this exact state is 2.3 m/s. End counts are 10/10, excess 1.7 mm, relative velocity about 0. It is not the uncapped catch. Peak acceleration is −42 m/s² rather than −47 m/s², the hand peaks at −1.42 m/s rather than −1.77 m/s, the excess only just crosses the pinch, and contact is at +0.18 m/s rather than +0.07 m/s.

Peak acceleration reaches the uncapped −47 m/s² at 3.0 m/s, because joints 2 and 4 are on the 87 N·m clip and joint 6 is on the 12 N·m clip. Achieved speed is still −1.60 m/s. The nominal trajectory becomes numerically the same as the uncapped reference at 5.0 m/s (same −3.54 mm minimum, same 0.106 s recontact). That is the point where a −5.3 m/s raw command is no longer clipped. It is not the smallest command that catches.

On the holds, joint 4's position margin reaches about −0.02 rad, including the uncapped run, so that joint touches its stop. Other margins stay above 1 rad. This is not a hardware-safety statement. The model is not enforcing a joint-velocity limit.

## Local neighborhood

The 2.3 m/s hold does not survive a small change of the same snapshot. Of seven shifts (object ±1.5 mm in z, object `vz` ±0.01 m/s, object ±2 mm in x, joint 2 by +0.005 rad), only the slower object (`vz` −0.01 m/s) still holds.

The misses split into two groups.

Vertical shifts are the cap. An object 1.5 mm or 3 mm higher still has no 1 s grasp at 3.5 m/s. At 3.5 m/s the hand does cross the pinch (minimum excess −2.6 mm and −2.4 mm) and the fingers are open, and it still does not regrasp. At 3.6, 3.7, 3.8, and 3.9 m/s both of those shifts hold, and the nominal state holds. The uncapped reference also holds them. A slower or faster object by 0.01 m/s, and a 3 mm lower object, already hold at 2.5 m/s.

Two shifts miss at every downward cap that was tried, including no downward cap at all: 2 mm toward −x, and joint 2 at +0.005 rad. The hand still crosses the pinch height. Both pads never take the cylinder. Raising `vz` does not create that grasp. Those misses belong to the intercept and the reclose gate, not to the vertical command limit.

## Lateral replay

The 0.30 m/s, sign +1, 0.08 s carry was seated again with the table on, then the table was removed and the hand opened. Release velocity is `[0.253, 0.002, −0.252]` m/s, both pads off, aperture 21 mm. The seated pinch of that grasp is the same `[0.22, 0.0, 97.58]` mm.

The v2 chase that previously caught this release still catches it. Peak commanded velocity is `[0.29, 0.00, −1.00]` m/s. End counts are 10/10. The downward command is exactly −1 m/s, so a higher downward limit is never used. `vx` and `vy` stay inside ±1 m/s.

The intercept used for the threshold, replayed on that same release, does not hold at 1.0, 2.5, 3.0, or with downward `vz` uncapped. All four end with no contact, minimum excess 2.2 mm, and a bilateral touch of about 0.09 s. The higher cap changes the peak command from −1.0 to −3.4 m/s and does not change the miss. It also does not disturb the v2 catch, because that catch never asks for more than 1 m/s downward.

## Proposed range

The snapshot starts to hold at 2.3 m/s. The same law needs 3.6 m/s before a 1.5–3 mm higher cylinder holds as well. 3.8 m/s is that threshold plus 0.2 m/s. It was run: nominal hold, both upward shifts hold, peak hand speed −1.73 m/s, peak acceleration −47.3 m/s², contact at +0.10 m/s, end counts 10/10.

5.0 m/s makes the nominal trace identical to the uncapped command. The catch is already there at 3.6 m/s, and the hand never travels at 5 m/s, so 5 m/s is not the limit this task is asking for.

For the v2 velocity action on this distribution: downward `vz` to 3.8 m/s, upward `vz` kept at 1.0 m/s, `vx` and `vy` kept at ±1.0 m/s, rotation and grip unchanged.

A FINITE VZ CAP REPRODUCES THE UNCAPPED AIRBORNE RECATCh
