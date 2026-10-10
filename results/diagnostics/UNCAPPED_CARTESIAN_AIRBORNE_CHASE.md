# Uncapped Cartesian airborne chase

Removing the ±1 m/s translation command lets the same torque controller close the 25 mm natural-loss gap and regrasps the cylinder. The ±1 m/s command does not. The actuators have the authority: on the catch that works, joints 2 and 4 sit on the 87 N·m clip and joint 6 sits on the 12 N·m clip for a few steps, and the hand's downward acceleration peaks at −47 m/s².

The state is the 13.5 mm closed-hold loss. The separation audit had not written `qpos`. This run repeated that same hold, matched the published scalars (loss at 6.074 s, excess 25.44 mm, object `vz` −0.046 m/s, no robot-cylinder contact, aperture 13.8 mm), saved `results/diagnostics/raw/uncapped_cartesian_airborne_chase/natural_loss_13_5.npz`, and reloaded that file for every chase. Floor and table contacts are off. Nothing is teleported. The hand moves only through MuJoCo, with the tracking gains (`kp_pos` 1400, `kd_pos` 80) and the actuator clips. The arm model has position limits. It does not define a joint-velocity limit.

The interception command is privileged and uncapped:

    p_des = p_object − R p_pinch
    v_des = v_object + 200 (p_des − p_hand)

No component of `v_des` is clipped for the uncapped runs. The ±0.08 and ±1.0 comparisons use that same law and then clip `v_des`. Grip torque is +2 N·m while chasing and −18 N·m after the pinch window (excess under 12 mm, relative speed under 0.15 m/s vertical and 0.20 m/s horizontal, aperture above 36 mm).

## What closing 25 mm requires

At the snapshot the excess is 0.0254 m and `v_obj,z − v_hand,z` is −0.046 m/s, so the gap is still opening. If the cylinder stays in free fall, constant relative acceleration `a_rel = a_obj,z − a_hand,z` has to reach:

| horizon (s) | required `a_rel` (m/s²) | required `a_hand,z` if the cylinder is at −g |
| --- | --- | --- |
| 0.05 | +22.2 | −32.0 |
| 0.08 | +9.1 | −18.9 |
| 0.10 | +6.0 | −15.8 |
| 0.15 | +2.9 | −12.7 |
| 0.20 | +1.7 | −11.5 |
| 0.40 | +0.55 | −10.4 |

A hand that only matches gravity (`a_hand,z ≈ −9.8`) has `a_rel ≈ 0` and cannot close the gap on any of these horizons.

## Matched chases

| command | min excess | when | peak `a_hand,z` | time with `a_hand,z < −g` | peak hand `vz` | actuator steps at the clip | regrasp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. clip `v_des` to ±0.08 m/s | 25.5 mm | 0.002 s | −10.3 | 0.028 s | −2.35 m/s late, after the object is gone | 10, joint 6 only, late | no |
| B. clip `v_des` to ±1.0 m/s | 22.7 mm | 0.060 s | −22.5 | 0.030 s | −2.44 m/s late | none. Peak torque 33 N·m of 87 | no |
| C. no Cartesian clip, fingers opening from 13.8 mm | −3.8 mm | 0.074 s | −47.2 | 0.040 s | −1.78 m/s | joints 2 and 4 at 87 N·m, joint 6 at 12 N·m | yes, see below |
| C, fingers preset to 40 mm before the chase | −6.8 mm | 0.062 s | −48.8 | 0.028 s | −1.25 m/s | same three joints, fewer steps | 1 s bilateral, end 10/10 |

A and B keep `v_des,z` on the cap (−0.08 and −1.0) while the object is still near the hand. Their early acceleration is about −4 to −14 m/s², short of the −16 to −19 m/s² a 0.08–0.10 s close needs. The excess never falls below 22 mm. Later the falling intercept point leaves a large position error, the position gain speeds the hand past the velocity clip, and the hand does reach about −2.4 m/s. That is after the cylinder is already tens of centimetres below the pinch. Both end with 0/0 contacts and the object near `z = −0.15` m.

Uncapped interception, fingers opening under +2 N·m:

| time (s) | excess (mm) | hand `vz` | object `vz` | `a_hand,z` | `a_rel,z` | aperture (mm) | contacts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.020 | 22.8 | −0.94 | −0.67 | −46.5 | +18.4 | 16.6 | 2/0 |
| 0.040 | 12.0 | −1.72 | −0.99 | −19.8 | +10.0 | 24.4 | 0/0 |
| 0.060 | −1.0 | −1.62 | −1.18 | +19.5 | −29.4 | 34.2 | 0/0 |
| 0.074 | −3.8 | −1.3 |  |  |  |  | 0, `v_rel,z` +0.02 |
| 0.100 | 1.5 | −1.18 | −1.30 |  |  | 21.8 | 18/0 |
| 0.120 | 1.9 | −1.11 | −1.04 |  |  | 9.5 | 19/26 |
| 0.160 | 0.2 | −0.95 | −0.96 |  |  | 16.5 | 10/8 |
| 1.15 | 1.4 | −0.03 | −0.03 |  |  | 17.9 | 10/10 |

The first 40 ms are free of a sustained push. There is a brief left-finger brush at 20 ms while the fingers are still at 17 mm; from 40 ms to 80 ms the contact count is zero and the excess falls through the pinch. At the minimum, relative vertical velocity is +0.02 m/s and nothing is touching the cylinder. The hand then brakes (positive `a_hand,z`) because the intercept law drops `v_des` as the error shrinks. Peak hand speed on this catch is −1.78 m/s, not because a 1.78 m/s ceiling was chosen, but because the gap is gone before a higher speed is required.

Both pads are on from 0.108 s. The close torque is −18 N·m. Aperture settles at 17.9 mm, the same figure as the seated pinch on this 36 mm cylinder. From 0.16 s, `|v_rel,z|` is under 0.01 m/s. At 1.15 s the counts are still 10/10, both vertical speeds are about −0.03 m/s, and the fall has stopped, at object `z = 0.194` m with the table absent. Five samples in that second have a 2 ms one-sided dropout, so the longest unbroken bilateral stretch is 0.68 s rather than a full second. The cylinder does not leave the hand.

Joint 4's position margin reaches −0.02 rad on this dynamic-open catch, so that joint touches its stop. Joint rates peak at 5.0 rad/s on joint 4 and 3.6 rad/s on joint 6. No velocity limit is being enforced.

The diagnostic branch sets the finger joints to 0.04 m and zeros their velocity before the first chase step, and does not move the arm or the cylinder. The hand still accelerates through the actuators. Excess crosses zero by 0.04 s with the fingers already at 40 mm and with no contact. Bilateral contact starts at 0.084 s. The strict 1 s bilateral flag is true: the run ends at 1.12 s with 10/10, aperture 17.9 mm, relative velocity at 0, object `z = 0.465` m. Joint margins stay positive (smallest is 0.93 rad on joint 4). This branch is only there to separate opening time from arm reachability. The dynamic-open run is the one that starts from the saved loss state.

A saturated downward setpoint, `v_des,z = v_obj,z − 30` with no speed clip, is a different command. The hand reaches −5.0 m/s and about −42 m/s² over the first 80 ms, with joints 2, 4, and 6 clipped on essentially every step. The excess overshoots to −21 mm while the cylinder is hit at high relative speed. Both pads never hold, and the object is lost. Asking for maximum downward speed is not the same as intercepting.

## Lateral release

Vertical interception did close the gap, so the same uncapped law was run from one verified lateral release: the closed hand carried sideways at 0.30 m/s for 0.08 s and then opened. At release the object velocity is `[0.253, 0.002, −0.252]` m/s, both pads are off, and the aperture is 21 mm. The cylinder is already near the pinch, so this is not the 25 mm vertical deficit.

Uncapped interception makes bilateral contact by 0.08 s and the vertical speeds match near −0.9 m/s, but the excess stays near 9 mm and the grasp is gone by the end of the run (0/0, excess 98 mm, object `z = −0.15` m). The ±1 m/s clip also touches and also loses. Horizontal speed here is 0.25 m/s, inside the old cap, so this miss is not the Cartesian ceiling that blocked the natural-loss case.

## Reading

On the natural-loss snapshot the ±1 m/s command never asks for the torque the actuators can produce, and the gap stays above 22 mm. The same intercept with that clip removed drives `a_hand,z` to −47 m/s², closes the 25 mm excess in 74 ms at a relative velocity of 0.02 m/s, and leaves the cylinder in the hand. The limiting command was the Cartesian cap.

THE ±1 M/S CARTESIAN CAP WAS THE LIMITING FACTOR
