# Observation robustness check

The frozen balanced checkpoint is evaluated on the same 24-case t = 12 benchmark. Noise is added only to the observation that enters the Transformer. The MuJoCo state, the previous action, the v3 torque command, and the nominal continuation are unchanged. Stochastic conditions use seeds 0, 1, and 2, with an independent stream on each case. The unmodified policy scores 15/24, matching the published balanced result.

## How each sensor error is applied

Relative position, orientation, and twist are decoded with the airborne scales (0.25 m, π rad, 2 m/s, 8 rad/s), perturbed, then scaled and clipped again. Orientation noise is a random-axis rotation whose angle is drawn from a zero-mean normal, composed on the left of the hand-frame relative rotation and converted back through the rotation vector. White noise is independent across axes and policy steps. Slow bias is an AR(1) process with correlation time 0.20 s, matched to the stated standard deviation. Delay holds channels 27–38 from the earlier policy step. Until that step exists, the first object sample is repeated. Proprioception, tactile readings, the wide-scale hand twist (39–44), and the previous action stay at the current step. Low-pass filtering is causal. Position, linear velocity, and angular velocity use y ← y + α(x − y) with α = dt / (τ + dt) and dt = 20 ms. Orientation takes the same step along the relative rotation and is reprojected to SO(3). The filter starts at the first measurement. Track loss begins at policy step 1 and lasts 1, 2, or 5 steps. Hold repeats the last object sample. Propagate advances that sample's position by its linear velocity and its rotation by its angular velocity, in the stored hand frame, then reprojects the rotation. The combined profile low-passes the object channels at τ = 50 ms, adds the white pose and twist noise, then delays the result by 40 ms. Tactile center-of-pressure noise is applied in metres on the pad coordinates and the lateral estimate is recomputed from those pads. Force noise is applied in newtons. Neither changes the contact flags or the simulator contact.

## Results

| condition | level | t=12 success | per seed | impact | sliding | wrist | recatch | airborne | natural | failure |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| position noise | sigma 1 mm per axis | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| position noise | sigma 2 mm per axis | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| position noise | sigma 5 mm per axis | 44/72 | 14,15,15 | 9/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| position bias | +2 mm on hand-x | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| position bias | +2 mm on hand-y | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| position bias | +2 mm on hand-z | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| position bias | +5 mm on hand-x | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| position bias | +5 mm on hand-y | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| position bias | +5 mm on hand-z | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| orientation noise | sigma 0.5 deg | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| orientation noise | sigma 1 deg | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| orientation noise | sigma 2 deg | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| orientation noise | sigma 5 deg | 44/72 | 14,15,15 | 9/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| linear velocity white | sigma 0.01 m/s | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| linear velocity slow | AR(1) sigma 0.01 m/s, correlation 0.2 s | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| linear velocity white | sigma 0.03 m/s | 44/72 | 14,15,15 | 9/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| linear velocity slow | AR(1) sigma 0.03 m/s, correlation 0.2 s | 44/72 | 14,15,15 | 9/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| linear velocity white | sigma 0.05 m/s | 43/72 | 14,14,15 | 9/12 | 12/12 | 12/12 | 10/12 | 0/12 | 0/12 | DROP |
| linear velocity slow | AR(1) sigma 0.05 m/s, correlation 0.2 s | 43/72 | 14,14,15 | 9/12 | 12/12 | 12/12 | 10/12 | 0/12 | 0/12 | DROP |
| linear velocity white | sigma 0.10 m/s | 43/72 | 14,14,15 | 9/12 | 12/12 | 12/12 | 10/12 | 0/12 | 0/12 | DROP |
| linear velocity slow | AR(1) sigma 0.10 m/s, correlation 0.2 s | 42/72 | 14,14,14 | 8/12 | 12/12 | 12/12 | 10/12 | 0/12 | 0/12 | DROP |
| angular velocity white | sigma 0.1 rad/s | 43/72 | 14,14,15 | 8/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| angular velocity slow | AR(1) sigma 0.1 rad/s, correlation 0.2 s | 44/72 | 15,15,14 | 9/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| angular velocity white | sigma 0.3 rad/s | 40/72 | 13,14,13 | 5/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| angular velocity slow | AR(1) sigma 0.3 rad/s, correlation 0.2 s | 42/72 | 14,15,13 | 7/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| angular velocity white | sigma 0.5 rad/s | 40/72 | 13,14,13 | 5/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| angular velocity slow | AR(1) sigma 0.5 rad/s, correlation 0.2 s | 39/72 | 13,14,12 | 4/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| angular velocity white | sigma 1.0 rad/s | 40/72 | 13,14,13 | 4/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| angular velocity slow | AR(1) sigma 1.0 rad/s, correlation 0.2 s | 37/72 | 14,12,11 | 4/12 | 11/12 | 11/12 | 11/12 | 0/12 | 0/12 | DROP |
| tracker delay | 20 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| tracker delay | 40 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| tracker delay | 60 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| tracker delay | 100 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| low-pass | tau 20 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| low-pass | tau 50 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| low-pass | tau 100 ms | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| track loss | 1 frame, hold, starting at policy step 1 | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| track loss | 1 frame, propagate, starting at policy step 1 | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| track loss | 2 frames, hold, starting at policy step 1 | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| track loss | 2 frames, propagate, starting at policy step 1 | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| track loss | 5 frames, hold, starting at policy step 1 | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| track loss | 5 frames, propagate, starting at policy step 1 | 15/24 | 15 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | DROP |
| tactile CoP | sigma 0.5 mm | 41/72 | 12,14,15 | 6/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| tactile lateral | sigma 0.5 mm | 43/72 | 14,15,14 | 8/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| tactile CoP | sigma 1.0 mm | 42/72 | 14,13,15 | 7/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| tactile lateral | sigma 1.0 mm | 39/72 | 12,14,13 | 5/12 | 12/12 | 12/12 | 10/12 | 0/12 | 0/12 | DROP |
| tactile force | sigma 1 N | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |
| tactile force | sigma 2 N | 44/72 | 15,14,15 | 9/12 | 12/12 | 12/12 | 11/12 | 0/12 | 0/12 | DROP |
| combined | 2 mm position, 1 deg orientation, 0.03 m/s, 0.3 rad/s, 40 ms delay, tau 50 ms, CoP 0.5 mm, force 1 N | 45/72 | 15,15,15 | 9/12 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | DROP |

The failure column is the most common label among rollouts that are not retained through t = 12. That label is DROP in every row. Airborne carry and natural-loss are DROP on the clean policy, and every grasp the noise newly loses is also DROP. No condition produces an ESCAPE that the clean policy did not already produce, and no condition retains a case the clean policy lost. Conditions absent from the next section kept the same 15 cases.

## Case changes against the clean 15/24

**pos_white_5mm** (sigma 5 mm per axis). Lost: recatch_suffix_16@0:DROP. Gained: none.

**ori_5deg** (sigma 5 deg). Lost: recatch_suffix_16@0:DROP. Gained: none.

**vwhite_0.03** (sigma 0.03 m/s). Lost: recatch_suffix_16@0:DROP. Gained: none.

**vslow_0.03** (AR(1) sigma 0.03 m/s, correlation 0.2 s). Lost: recatch_suffix_16@0:DROP. Gained: none.

**vwhite_0.05** (sigma 0.05 m/s). Lost: recatch_suffix_16@0:DROP, recatch_suffix_16@1:DROP. Gained: none.

**vslow_0.05** (AR(1) sigma 0.05 m/s, correlation 0.2 s). Lost: recatch_suffix_16@0:DROP, recatch_suffix_16@1:DROP. Gained: none.

**vwhite_0.10** (sigma 0.10 m/s). Lost: recatch_suffix_16@0:DROP, recatch_suffix_16@1:DROP. Gained: none.

**vslow_0.10** (AR(1) sigma 0.10 m/s, correlation 0.2 s). Lost: recatch_suffix_16@0:DROP, recatch_suffix_16@1:DROP, impact_z3@2:DROP. Gained: none.

**wwhite_0.1** (sigma 0.1 rad/s). Lost: impact_phi90@0:DROP, recatch_suffix_16@2:DROP. Gained: none.

**wslow_0.1** (AR(1) sigma 0.1 rad/s, correlation 0.2 s). Lost: recatch_suffix_16@2:DROP. Gained: none.

**wwhite_0.3** (sigma 0.3 rad/s). Lost: impact_phi90@0:DROP, impact_z3@0:DROP, impact_phi90@1:DROP, impact_phi90@2:DROP, recatch_suffix_16@2:DROP. Gained: none.

**wslow_0.3** (AR(1) sigma 0.3 rad/s, correlation 0.2 s). Lost: impact_phi90@0:DROP, impact_z3@2:DROP, recatch_suffix_16@2:DROP. Gained: none.

**wwhite_0.5** (sigma 0.5 rad/s). Lost: impact_phi90@0:DROP, impact_z3@0:DROP, impact_phi90@1:DROP, impact_phi90@2:DROP, recatch_suffix_16@2:DROP. Gained: none.

**wslow_0.5** (AR(1) sigma 0.5 rad/s, correlation 0.2 s). Lost: impact_phi90@0:DROP, impact_z3@0:DROP, impact_phi90@1:DROP, impact_phi90@2:DROP, impact_z3@2:DROP, recatch_suffix_16@2:DROP. Gained: none.

**wwhite_1.0** (sigma 1.0 rad/s). Lost: impact_phi90@0:DROP, impact_z3@0:DROP, impact_phi90@1:DROP, impact_phi90@2:DROP, impact_z3@2:DROP. Gained: none.

**wslow_1.0** (AR(1) sigma 1.0 rad/s, correlation 0.2 s). Lost: impact_phi90@0:DROP, impact_zm6@1:DROP, impact_phi90@1:DROP, wrist_suffix_9@1:DROP, impact_phi90@2:DROP, impact_z3@2:DROP, slide_dy1@2:DROP, recatch_suffix_16@2:DROP. Gained: none.

**cop_0.5mm** (sigma 0.5 mm). Lost: impact_phi90@0:DROP, impact_z3@0:DROP, recatch_suffix_16@0:DROP, impact_phi90@1:DROP. Gained: none.

**ehat_0.5mm** (sigma 0.5 mm). Lost: impact_phi90@0:DROP, recatch_suffix_16@2:DROP. Gained: none.

**cop_1.0mm** (sigma 1.0 mm). Lost: impact_phi90@0:DROP, impact_phi90@1:DROP, recatch_suffix_16@1:DROP. Gained: none.

**ehat_1.0mm** (sigma 1.0 mm). Lost: impact_phi90@0:DROP, impact_z3@0:DROP, recatch_suffix_16@0:DROP, impact_phi90@1:DROP, impact_zm6@2:DROP, recatch_suffix_16@2:DROP. Gained: none.

**fn_2N** (sigma 2 N). Lost: recatch_suffix_16@1:DROP. Gained: none.

## Conclusion

Performance is stable under modest realistic object-tracking error. Position noise at 1 mm and 2 mm, a constant 2 mm or 5 mm position bias on each hand axis, orientation noise through 2 deg, linear-velocity noise at 0.01 m/s, tracker delay through 100 ms, causal smoothing through τ = 100 ms, and a 1-, 2-, or 5-frame track loss (hold or propagate) all score 15/24 on every rollout. Normal-force noise at 1 N does the same. The combined profile — 2 mm position, 1 deg orientation, 0.03 m/s linear velocity, 0.3 rad/s angular velocity, 40 ms delay, τ = 50 ms, 0.5 mm center-of-pressure noise, and 1 N force noise — scores 15/24 on seeds 0, 1, and 2 (45/72).

Higher object-twist noise and tactile position error are what move the score. Linear velocity from 0.03 m/s upward drops `recatch_suffix_16` on one or two of three seeds, and the 0.10 m/s slow bias also drops `impact_z3` once. Angular velocity at 0.3 rad/s and above removes `impact_phi90` and sometimes `impact_z3`. Only the 1.0 rad/s slow bias also touches sliding and wrist, and only on seed 2 (`slide_dy1`, `wrist_suffix_9`). Center-of-pressure noise at 0.5 mm is the least stable modest single factor (12, 14, and 15 out of 24). Lateral-estimate noise at 1 mm is similar (12, 14, 13). Those losses are seed-dependent and concentrated on `impact_phi90`, `impact_z3`, and `recatch_suffix_16`.

The policy is not highly dependent on exact simulator object state. Delay, smoothing, short track loss, and millimetre-scale pose error leave the clean 15/24 in place. Sliding stays 4/4 and wrist stays 4/4 except for that one high angular-velocity seed. Airborne carry and natural-loss stay 0/4.

