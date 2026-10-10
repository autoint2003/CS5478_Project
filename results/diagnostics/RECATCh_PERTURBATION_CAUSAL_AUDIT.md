# Causal audit of recatch perturbations

On the verified dz −13.5 pre-loss pose, two components of the failed-policy twist are enough to make the legal v3 intercept miss a 1 s regrasp: hand-frame vy at the observed median deviation (−0.123 m/s) and hand-frame vz at the observed median deviation (+0.187 m/s). The same two components also defeat every law in the five-law family. The observed vx deviation (−0.012 m/s), the median wy (−0.048 rad/s), and both wz deviations (−0.127 and +0.536 rad/s) leave the 1 s regrasp intact. A wx deviation of −1.86 rad/s changes the release that a closed hold produces, and the intercept started immediately from that same perturbed state still regrasps.

These magnitudes are the componentwise deviations of the 52 failed policy visits from the reference twist. They are not a claim that every small change leaves the basin. The vx change that was actually observed does not.

## Reference

The injection state is the last bilateral frame of the dz −13.5 natural-loss trajectory, before the persistent one-sided release. Object angular velocity is stored in the world frame. The hand-frame twist there is

- v = [0.0009, 0.0020, 0.0308] m/s
- w = [−0.126, 0.032, 0.051] rad/s
- finger opening 13.92 mm

A closed hold from that frame loses one pad 20 ms later, still on the distal tip: clearance −0.014 mm, vertical seat error 25.47 mm, lateral error 0.02 mm, relative velocity [0.0019, −0.0001, 0.0484] m/s, fingers 13.76 mm, contacts [1, 0]. The v3 intercept then holds both pads for 1 s.

Positive hand-frame vz is motion along hand +z, which is world down, so it is the separating direction for this grasp.

## What the previous FULL3D_RECATCh_BASIN search was

That audit rolled out a finite family of closed-loop feedback laws. It was not an exhaustive reachability test. There is no action grid, no sampled open-loop sequence, and no numerical optimizer (no CEM, no MPC). Each law is recomputed every 20 ms from the current object and hand state, then clipped into recovery7d_airborne_v3:

- hand-frame vx, vy within ±1.0 m/s
- hand-frame vz from −1.0 m/s to +3.8 m/s (world vz from −3.8 m/s to +1.0 m/s)
- wrist rate within ±4 rad/s
- the existing grip map
- actuator torque limits left on

The horizon is 2.8 s (140 policy steps of 20 ms). A rollout stops early after a 1 s regrasp, if the object falls below z = −0.15 m, or if 1.5 s pass with the object more than 80 mm below the seat and still unregrasped. Success requires a recontact time, both-pad contact on at least 95% of the samples from recontact through recontact + 1.0 s, both pads in contact at the end of that window, and seat excess under 20 mm.

Five laws, 5 candidates per state, 260 rollouts on the 52 states:

1. Intercept. World velocity = object velocity + 200 × the pinch-target error, wrist rate 0, grip open while chasing and closed once the seat is aligned and slow.
2. Lateral. Hand-frame velocity = object velocity + (16 e_x, 16 e_y, k_z e_z), with k_z = 2 while the lateral error exceeds 6 mm and k_z = 80 inside that gate. Wrist rate 0.
3. Lateral, harder gate. Gains (40, 40), and k_z = 0.4 until the lateral error is under 4 mm.
4. Orientation. The intercept translation plus a wrist rate of 10 × the rotation vector toward the seated hand-object orientation, clipped to 4 rad/s.
5. Combined. Law 2 plus that wrist rate.

Feedback is used: every command is recomputed from the state at that 20 ms step. A miss by this family does not prove that no legal v3 sequence can regrasp. On the unperturbed reference itself, only law 1 succeeds. Laws 2–5 miss the reference, so a miss by those four laws is not evidence about the intercept.

The controller used after each injection below is the one that succeeds on this reference: hold the closed grip until the first persistent loss, then run the v3 intercept, and only if the loss clearance is still within 1 mm. The five-law family is run as a second instrument, started immediately from the perturbed state.

## Matched injections

Only the object velocity is written. The reference snapshot is restored first, so the object pose, finger joints, contact counts, controller setpoint, and model are the pre-injection values. After the velocity write and one kinematics update, every trial has qpos change 0, the same finger opening, and the same contact counts.

The added values are the componentwise median and the componentwise 90th percentile of (failed − reference) over the 52 visits, in the order vx, vy, vz, wx, wy, wz:

- median delta: [−0.0123, −0.1232, +0.1869, −1.863, −0.0479, −0.1270]
- 90th-percentile delta: [−0.0020, +0.2069, +0.6937, +8.098, +0.1989, +0.5364]

Linear entries are m/s and angular entries are rad/s, in the hand frame. A single-axis trial adds one entry and leaves the other five at the reference. The median combination adds the whole median vector. The 90th-percentile combination adds the whole 90th-percentile vector at once; that joint vector was not observed as one state.

Every injection stays in bilateral contact at the write, then reaches a persistent loss. The first-loss state is the first frame of that persistent run.

| injection | delta | contact at write | time to loss | 1 s regrasp after hold | intercept started immediately |
|---|---:|---|---:|---|---|
| unperturbed | 0 | unchanged | 0.020 s | yes | yes |
| vx median | −0.0123 m/s | unchanged | 0.020 s | yes | yes |
| vx 90th | −0.0020 m/s | unchanged | 0.020 s | yes | yes |
| vy median | −0.1232 m/s | unchanged | 0.020 s | no | no |
| vy 90th | +0.2069 m/s | unchanged | 0.020 s | no | no |
| vz median | +0.1869 m/s | unchanged | 0.020 s | no | no |
| vz 90th | +0.6937 m/s | unchanged | 0.020 s | no | no |
| wx median | −1.863 rad/s | unchanged | 0.020 s | no | yes |
| wx 90th | +8.098 rad/s | unchanged | 0.020 s | no | yes |
| wy median | −0.0479 rad/s | unchanged | 0.020 s | yes | yes |
| wy 90th | +0.1989 rad/s | unchanged | 0.046 s | no | yes |
| wz median | −0.1270 rad/s | unchanged | 0.020 s | yes | yes |
| wz 90th | +0.5364 rad/s | unchanged | 0.020 s | yes | yes |
| median combination | all six median entries | unchanged | 0.020 s | no | no |
| 90th combination | all six 90th entries | unchanged | 0.020 s | no | no |

First-loss geometry, compared with the unperturbed release:

| injection | clearance (mm) | vertical (mm) | lateral (mm) | v at loss (m/s) | fingers (mm) | contacts |
|---|---:|---:|---:|---|---:|---|
| unperturbed | −0.014 | 25.47 | 0.02 | [0.0019, −0.0001, 0.0484] | 13.76 | [1, 0] |
| vx median | −0.029 | 25.37 | 0.05 | [−0.0090, 0.0028, 0.0396] | 13.82 | [4, 0] |
| vx 90th | −0.028 | 25.37 | 0.02 | [−0.0002, 0.0021, 0.0396] | 13.82 | [4, 0] |
| vy median | −0.155 | 25.30 | 0.19 | [0.0034, −0.0984, 0.0428] | 13.87 | [0, 4] |
| vy 90th | −0.274 | 25.32 | 0.35 | [−0.0028, 0.1721, 0.0504] | 13.87 | [4, 0] |
| vz median | +0.202 | 25.65 | 0.01 | [0.0010, 0.0026, 0.2190] | 13.88 | [0, 0] |
| vz 90th | +0.890 | 26.66 | 0.01 | [0.0011, 0.0027, 0.7241] | 13.90 | [0, 0] |
| wx median | +0.025 | 25.70 | 0.02 | [−0.0017, −0.0071, 0.0968] | 13.59 | [0, 0] |
| wx 90th | −0.151 | 25.84 | 0.07 | [0.0109, 0.0036, 0.1158] | 13.52 | [0, 1] |
| wy median | −0.026 | 25.37 | 0.01 | [0.0014, 0.0019, 0.0397] | 13.82 | [5, 0] |
| wy 90th | +0.154 | 29.56 | 0.13 | [−0.0027, 0.0008, 0.3436] | 9.16 | [0, 0] |
| wz median | −0.028 | 25.37 | 0.02 | [0.0014, 0.0024, 0.0396] | 13.82 | [4, 0] |
| wz 90th | −0.006 | 25.47 | 0.01 | [0.0017, −0.0024, 0.0487] | 13.76 | [1, 0] |
| median combination | +0.054 | 25.65 | 0.20 | [−0.0113, −0.1039, 0.2187] | 13.88 | [0, 0] |
| 90th combination | +0.615 | 26.67 | 0.37 | [−0.0010, 0.1821, 0.7246] | 13.90 | [0, 0] |

Clearance stays inside the 1 mm chase gate in every row, so the intercept is actually run. The vy and vz misses are misses of that chase, not cases where the gate refused to chase.

What each component does:

- vx at −12 mm/s and at −2 mm/s keeps the release on the distal tip and keeps the 1 s regrasp. This is the small linear deviation in the failed set, and it is recoverable.
- vy at −0.123 m/s and at +0.207 m/s still loses on one side with negative clearance, but the lateral velocity at the loss is still −0.098 m/s or +0.172 m/s, and a spin about x of about 1 rad/s has appeared. Both the hold-then-intercept and the immediate intercept miss.
- vz at +0.187 m/s and at +0.694 m/s releases both pads before the loss frame. Clearance becomes +0.20 mm and +0.89 mm, and the separating speed is still 0.22 m/s and 0.72 m/s. Both controllers miss. Vertical position only moves from 25.5 mm to 25.7 mm and 26.7 mm; the change that removes the regrasp is the separating velocity at release, not a large extra drop.
- wx at −1.86 rad/s, and at the much larger +8.1 rad/s tail, makes the hold-then-intercept miss. The intercept started at the injection instant still regrasps. The spin changes the release the hold is willing to wait for. It does not, by itself, make the injection state unreachable for that intercept.
- wy at −0.048 rad/s keeps the reference release and the regrasp. wy at +0.199 rad/s holds for 46 ms, closes the fingers to 9.2 mm, and reaches a separating speed of 0.34 m/s; the hold-then-intercept misses, and the immediate intercept still catches.
- wz at −0.127 rad/s and at +0.536 rad/s keeps the reference release and the regrasp.
- The median combination misses because it contains the vy and vz values that each miss alone.
- The 90th-percentile combination also misses. It is a stacked tail, including +8.1 rad/s of wx, and it is not one of the recorded visits.

## Component resets on the 52 failed visits

Each failed snapshot is edited, then run through the same hold-until-loss intercept and through the five-law family.

- A keeps the failed object pose and finger joints, and sets the hand-frame linear and angular velocity to the reference twist.
- B keeps the failed twist, and moves the object pose in the hand and the finger joints to the reference.
- C replaces only the lateral relative velocity.
- D replaces only the angular velocity.
- E replaces only the vertical relative velocity.

| reset | stayed in contact for 4 s | lost, intercept missed | 1 s regrasp | five-law family |
|---|---:|---:|---:|---:|
| A, velocity to reference, geometry kept | 22 | 29 | 1/52 | 0/52 |
| B, geometry to reference, velocity kept | 0 | 51 | 1/52 | 1/52 |
| C, lateral velocity only | 21 | 31 | 0/52 | 0/52 |
| D, angular velocity only | 21 | 30 | 1/52 | 0/52 |
| E, vertical velocity only | 21 | 30 | 1/52 | 0/52 |

The 22 states that stay grasped for 4 s after A are 21 of the visits previously labelled no persistent loss, plus one chase failure. Their vertical seat error has median 1.2 mm (10th–90th percentile −9.1 to +8.7 mm). With the reference twist put back, those grasps do not release, so there is no recatch to score. The velocity they were carrying is what the 4 s hold responds to; the near-seat geometry with the reference twist does not force a loss.

The 29 states that still lose after A are all previous chase failures. Their objects are already lower in the hand (vertical error median 19.2 mm) and the fingers are already narrower (median 9.3 mm), with the original separating speed median 0.39 m/s. After the twist is replaced, the release that still occurs has clearance median −0.62 mm, so the intercept does run, and it misses. Loss separating speed median is 0.26 m/s. For this group the pose and the finger opening, with the successful twist installed, are sufficient for this controller to miss.

B misses on 51 of 52. The release stays inside the 1 mm gate (clearance median −0.15 mm, 90th percentile +0.20 mm) and the intercept misses, with loss separating speed median 0.26 m/s. That matches the injection result: the failed twist, placed on the successful pose and the successful fingers, is sufficient for the miss. The one B success is the same visit as the one A success.

No visit is restored by lateral velocity alone. One visit is restored by more than one of the partial resets. The other 51 are restored by none of C, D, or E.

That one visit is `dz-13.15_dy0.1`, previously a chase failure. At the visit it is already close to the reference twist: lateral error 0.1 mm, vertical error 20.0 mm, fingers 16.7 mm, relative velocity about [0, 0, 0.014] m/s, angular speed 0.14 rad/s, cylinder-axis tilt 0.04°, both pads in contact. Hold-then-intercept regrasps after A, after B, after D, and after E. It does not regrasp after C, which leaves the recorded spin in place; the release under C has separating speed 0.34 m/s. The immediate intercept regrasps only after B.

## Which components remove the regrasp

On the successful pose, with fingers, contact, and controller unchanged at the injection:

- The observed median and 90th-percentile vy remove the 1 s regrasp for the hold-then-intercept and for the immediate intercept.
- The observed median and 90th-percentile vz do the same. The first loss becomes a two-pad release while the object is still inside the 1 mm gate, and the separating speed is still the injected speed.
- The observed vx, the median wy, and both wz values do not. The first loss stays a one-sided distal-tip release near 25.4 mm and 13.8 mm of finger opening, and the intercept holds for 1 s.
- The observed wx, whose median is −1.86 rad/s, removes the regrasp of the hold-then-intercept and leaves the immediate intercept able to regrasp.

On the failed visits themselves, the split is by where the object already is:

- Near the seat (22 visits), replacing the whole twist with the reference stops the release for 4 s.
- Already ~19 mm down the hand with fingers near 9 mm (29 visits), replacing the whole twist does not restore the intercept. The geometry at the visit is sufficient for the miss.
- On 51 of 52 visits, putting the reference pose and fingers back while keeping the failed twist is also sufficient for the miss.

Lateral position error is not the component this experiment isolates. The vy injection that misses moves the loss laterally by 0.2–0.4 mm. The vz injection that misses moves it laterally by 0.01 mm.
