# Unconstrained airborne recatch

This is a feasibility check, not a trained policy. The 0.08 m/s and 4 rad/s recovery-action caps are not used. The arm is driven by the existing Cartesian torque controller, and those torques are clipped only by the simulated actuators: 87 N·m on joints 1–4 and 12 N·m on joints 5–7. The cylinder is never teleported and is never given a velocity. Gravity stays on. The table and the floor no longer collide with anything, so the cylinder cannot land on a support during the attempt.

The aim is privileged. Each step reads the cylinder pose and velocity from the simulator and commands a hand target from that. That is not a deployable observation. It is used here only to test whether a dynamically simulated trajectory can intercept the fall.

## Why the earlier attempts missed

Three limits sat on top of the robot, separate from the actuator limits.

The recovery action scaled hand translation to ±0.08 m/s and rotation to ±4 rad/s. In the 0.18 s the cylinder previously took to reach the table, that cap moves the hand about 14 mm. The cylinder falls about 160 mm in the same interval.

The table top is at 0.40 m. From the horizontal pinch near 0.59 m the free fall lasts about 0.18 s, and then the cylinder is on the table.

The horizontal grasp itself is not a bad starting pose for a downward chase. It is documented below and was kept.

## Initial pose

The trial starts from the settled horizontal grasp: cylinder axis along world Y, both pads on (10 and 10), grip −18 N·m, object height 0.593 m, hand height 0.690 m. Hand-frame cylinder position is `[0.2, 0.0, 97.6]` mm.

Arm joints, in radians:

`[0.000, -0.166, 0.000, -1.591, 0.001, 1.441, -0.785]`

Joint 4 is at −1.59 rad, about 1.48 rad clear of either stop in its range. Joint 2 is near the middle of its range. Every arm joint has more than 1.4 rad of margin. The hand Jacobian condition number is 12.2, with singular values from 1.84 down to 0.15. The elbow is bent, so joints 2 and 4 can accelerate the hand downward without starting on a singularity or a joint stop. Once the table and floor collisions are removed, the space under the hand is open.

A separate hard downward command, not the catching controller, drove this same pose to about −3 m/s at the hand within 0.15 s, with joint 4 passing 8 rad/s and shoulder torque near 80 N·m. The actuator limits allow a much faster dive than the catching trajectories actually used.

## What the catching controller does

The fingers open at +18 N·m. When the aperture passes about 34 mm the open torque drops to +4 N·m so the fingers stay open without being driven through the stop. The hand target is the pose that puts the cylinder back at the measured pinch, optionally shifted by about a centimetre along the finger direction, or aimed 40 mm further down for the below-the-object trials. The commanded hand velocity is the cylinder velocity plus a multiple of the position error, clipped to ±4.5 m/s. The Cartesian law turns that into joint torques. The fingers close at −18 N·m when the cylinder is inside the pinch window, the aperture is still open, and the relative velocity is under the trial threshold (0.25, 0.40, or 0.60 m/s). After both pads have been on for 40 ms the arm eases off and the grip stays closed. Success requires a contact-free interval, a simulated recontact, both pads on, and that grasp still holding 1 s later.

Fifty-four settings were run: chase from above, a lower aim point, and explicit velocity matching, each with two error gains and three pinch offsets.

## A catch that holds

Thirty of the fifty-four settings recontact the cylinder and still have both pads on 1 s later. Two of them are saved in full.

Chase from above, pinch aim 12 mm toward the fingertips. The pads are both off from 4 ms to 64 ms after the open command, and both are on again at 72 ms. During that gap the aperture reaches 49 mm. The hand moves 11 mm down while the cylinder moves 20 mm down. At recontact the hand velocity is `[0.04, 0.00, -0.45]` m/s and the cylinder velocity is `[-0.29, 0.00, -0.61]` m/s, so the relative velocity is `[-0.32, 0.00, -0.16]` m/s. Peak downward hand speed on this trial is 0.66 m/s. Peak joint speed is 1.80 rad/s on joint 4 and 1.46 rad/s on joint 6. Peak torque is 52 N·m on joint 2 and 46 N·m on joint 4. Wrist joint 6 reaches 11.5 N·m, next to its 12 N·m stop. Orientation error stays under 2.9 degrees and the position tracking error stays under 26 mm. One second later both pads are still on (10 and 10), the relative velocity is about `0.0001` m/s, and the cylinder is at 0.405 m, held in the hand. The floor is not there to catch it.

Get-below aim, with the hand target 40 mm lower. Both pads are off over the same 4–64 ms window and return at 72 ms. The hand and the cylinder each drop about 26–28 mm before contact. At recontact the vertical velocities match: hand −0.687 m/s, cylinder −0.682 m/s, relative vertical velocity +0.005 m/s. The lateral relative velocity is −0.34 m/s. Peak downward hand speed is 0.92 m/s. Joint 4 reaches 2.53 rad/s and joint 6 reaches 2.13 rad/s. Joint 2 torque peaks at 65 N·m and joint 6 sits on the 12 N·m limit. One second later the pads are still on (6 and 7), the relative velocity is about `0.0004` m/s, and the cylinder is at 0.35 m.

The contact-free interval is short because the hand is already around the cylinder and can catch it before the fall becomes long. It is not a one-step flicker: the fingers open well past the original 18 mm aperture, both counts stay at zero for about 60 ms, and the reclose is what brings the pads back.

## How far that success extends

The same above-catch, repeated from nearby initial states:

| change before release | still grasped 1 s later |
| --- | --- |
| cylinder 5 mm to either side | yes, both pads on |
| cylinder 5 mm along its axis | yes, 10 and 10 |
| cylinder 5 mm higher in the pinch | yes, 10 and 9 |
| open command delayed 30 ms | yes, 10 and 8 |
| open command delayed 60 ms | yes, 9 and 8 |
| joint 2 shifted by +0.01 rad | yes, 10 and 10 |
| joint 2 shifted by +0.03 rad | no |
| joint 4 shifted by −0.03 rad | no |
| cylinder 5 mm or 13 mm lower in the pinch | pads can touch again, then the cylinder leaves within the one-second hold |

The successful set is a neighborhood of the seated pinch and of the controller gains, not a single timestep. A deeper starting slip, of the kind measured at 13 mm in the slow-slip audit, is outside that neighborhood for this controller: the fingers meet the cylinder and then lose it, and on the harder dives the shoulder torque saturates at 87 N·m.

Videos, with the contact-free interval labeled, are `results/diagnostics/raw/unconstrained_airborne_recatch/videos/privileged_chase.mp4` and `below_velocity_match.mp4`. The matching trajectories, including joint position, joint velocity, torque, hand and cylinder velocity, and contacts, are the `.npz` files in that same directory.

## Separation

The earlier misses were the 0.08 m/s action cap, which could not cover the fall, and the table, which ended the fall in about 0.18 s. The horizontal grasp pose was already usable: bent elbow, joint margins above 1.4 rad, Jacobian condition about 12.

With those caps and the table removed, the simulated arm reaches the falling cylinder on actuator torque alone. A catching trajectory holds both pads for a full second afterward, including one in which the vertical relative velocity at contact is about 0.005 m/s. The aim uses privileged cylinder state, so this is evidence of physical reachability, not of a policy that could be run from the legal onboard observation.

UNCONSTRAINED PANDA CAN PHYSICALLY CHASE AND RECATCh A FALLING CYLINDER
