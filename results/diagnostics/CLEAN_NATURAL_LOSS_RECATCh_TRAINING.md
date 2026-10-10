# Clean natural-loss recatch training

The retrained Transformer completes a 1 s bilateral regrasp on **1 of 6** held-out windows. Both windows belong to conditions whose first persistent bilateral loss is still on the distal tip and whose privileged intercept regrasps. The same short prefix of the −14.0 mm offset is the only success. Longer prefixes of that offset, and every prefix of the held-out −13.4 mm offset, miss.

`d_center` is the object-center displacement from the seated hand-frame pose `[0.22, 0.0, 97.56]` mm. Positive `d_center` points along hand +z, which is world down. Physical separation is `d_clearance`: the minimum `mj_geomDistance` from the cylinder geom to the four distal tip boxes (local z = 50 mm, half-size 3 × 2 × 3 mm). A loss is clean when that distance is at most **+1.0 mm**. The geometry is taken at the first frame of a bilateral loss that stays lost for 40 ms. Penetration (a negative distance) is still contact.

The action map, observation, tokenizer, architecture, and controller are the ones already verified. `recovery7d_airborne_v3` keeps vx, vy in ±1.0 m/s, downward world vz at 3.8 m/s, upward vz at 1.0 m/s, wrist rate at ±4 rad/s, and the existing grip map, with actuator torque limits on. The observation is the 45-D ground-truth vector. The tokenizer is the frozen K=32 prototype set from the previous checkpoint. The causal Transformer is unchanged (width 64, 2 layers, 4 heads). Training used CUDA on an RTX 3060, mixed precision, 120 epochs, final loss 0.248. No skill ids were added.

## What entered the set

Twelve nearby grasps were slipped under a closed hold. Eight reach a first persistent loss with clearance between −0.023 and +0.007 mm, center displacement 25.44–25.52 mm, finger opening about 13.7 mm, and hand-frame relative velocity about +0.05 m/s. The privileged intercept regrasps all eight (end contact 9–10 and 10). Six of those were assigned to training before the oracle results were known: offsets −13.2, −13.5, −14.2, and −14.6 mm, plus the −13.5 mm offset with object vz −0.01 m/s and with +0.005 m/s. Each training trajectory starts 0.3 s, 1.0 s, and 2.0 s before loss, while both pads are still on, and continues through the intercept, recontact, reclose, and hold. That is 18 windows.

Two clean, oracle-recatchable conditions were held out in advance: −13.4 mm and −14.0 mm. They are the primary denominator (6 windows).

Two further losses are still on the distal corner (clearance +0.008 and +0.107 mm) but the intercept does not regrasp. The fingers have closed to about 9 mm, the center has moved to 29.4 mm, and the separating velocity is about +0.34 m/s. The hand never gets the center back. They are clean and oracle-unrecoverable, and they were not used as demonstrations.

Two starts, +0.5 mm lateral and joint-2 +0.001 rad, have already lost bilateral contact at the moment the offset is applied. They are outside the marginal grasp and were not slipped, trained, or scored.

No candidate in this sweep loses contact with a clearance of several millimetres. The previously measured closed-loop state at `d_center` 36.92 mm and `d_clearance` +5.51 mm (finger joints crossed, about −4.5 mm) is recorded below so it cannot be read back in as a natural-loss example. It is not in the training balance, the primary rate, or the held-out set.

The three verified local skills on the −6 mm impact — wrist reseating, the open-finger recatch, and inward sliding — stay in the same dataset, without skill ids. After retraining, each retains the cylinder through absolute t = 12, with normalized translation commands below 0.05.

## Audit

Relative velocity is the hand-frame z component. On every clean recatchable loss the other two components are under 0.005 m/s.

| case | initial offset | first-loss time (s) | d_center (mm) | d_clearance (mm) | v_rel,z (m/s) | oracle recatchable? | in training? | in primary eval? |
|---|---|---:|---:|---:|---:|---|---|---|
| dz13.2 | dz −13.2 mm | 7.80 | 25.52 | +0.007 | +0.051 | yes | yes | no |
| dz13.5 | dz −13.5 mm | 6.58 | 25.47 | −0.014 | +0.048 | yes | yes | no |
| dz14.2 | dz −14.2 mm | 4.48 | 25.45 | +0.004 | +0.046 | yes | yes | no |
| dz14.6 | dz −14.6 mm | 3.86 | 25.47 | +0.006 | +0.047 | yes | yes | no |
| dvz_m | dz −13.5 mm, object vz −0.01 m/s | 6.36 | 25.44 | −0.023 | +0.048 | yes | yes | no |
| dvz_p | dz −13.5 mm, object vz +0.005 m/s | 6.68 | 25.44 | −0.001 | +0.046 | yes | yes | no |
| dz13.4 | dz −13.4 mm | 6.98 | 25.45 | +0.001 | +0.046 | yes | no | yes |
| dz14.0 | dz −14.0 mm | 4.68 | 25.49 | +0.002 | +0.045 | yes | no | yes |
| dz13.8 | dz −13.8 mm | 5.38 | 29.44 | +0.008 | +0.340 | no | no | no |
| dz13.6 | dz −13.6 mm | 6.20 | 29.45 | +0.107 | +0.350 | no | no | no |
| dq2 | dz −13.5 mm, joint 2 +0.001 rad | — | 13.01 | −0.38 | — | — | no | no |
| dx05 | dz −13.5 mm, dx +0.5 mm | — | 13.50 | −0.61 | — | — | no | no |
| prior 36.9 mm state | not a candidate; closed-loop outcome from an earlier policy | — | 36.92 | +5.51 | +0.51 | no | no | no |

`dq2` and `dx05` are already without bilateral contact when the trial starts, so they have no first-loss time. The last row is the excluded separated state from the earlier geometry audit. Nothing in this file trains on it.

## Closed loop

Each held-out window starts during bilateral slip. The policy alone chooses every action. There is no phase switch and no intercept call.

Teacher forcing on these windows predicts the full downward token (`a2 = 1`) at the first chase step, with action error 0.01–0.07. The closed-loop misses are not a failure to represent that action on the demonstration.

| window | d_center at loss (mm) | d_clearance (mm) | v_rel,z (m/s) | first chase | high-vz duration (s) | recontact state | 1 s hold |
|---|---:|---:|---:|---|---:|---|---|
| dz14.0, 0.3 s | 25.43 | −0.034 | +0.039 | a2 = 1, world vz −3.8 m/s | 0.06 | hand-frame position [−0.81, 7.68, 91.41] mm, d_center −6.15 mm, v_rel [−0.010, −0.035, +0.348] m/s | yes, end 10/10 |
| dz14.0, 1.0 s | 31.68 | +0.166 | +0.049 | a2 = 1, world vz −3.77 m/s | 0.06 | no recontact | no |
| dz14.0, 2.0 s | 38.87 | +7.09 | +0.639 | a2 = 1, world vz −3.80 m/s | 0.14 | no recontact | no |
| dz13.4, 0.3 s | 33.95 | +2.09 | +0.475 | a2 = 1, world vz −3.8 m/s | 0.02 | no recontact | no |
| dz13.4, 1.0 s | 33.41 | +1.58 | +0.438 | a2 = 1, world vz −3.8 m/s | 0.02 | brush at d_center 33.7 mm, clearance +1.8 mm; not held | no |
| dz13.4, 2.0 s | 36.38 | +4.48 | +0.490 | a2 = 1, world vz −3.8 m/s | 0.06 | brush at d_center 37.7 mm, clearance +5.8 mm; not held | no |

On the successful window the loss is the same physical event as the demonstrations: center 25.43 mm, clearance −0.03 mm, finger opening 13.8 mm, contact on the left distal boxes, separating speed 0.04 m/s. The first chase command is the downward cap. The command stays at or below −3 m/s for 0.06 s. The step that restores both pads starts with the center 6.15 mm above the seated reference and a downward relative speed of 0.35 m/s. Both pads then stay on for the following second (end counts 10 and 10). Absolute t = 12 is not part of this natural-loss trial; the scored retention is the 1 s regrasp. The three local skills, scored separately, are `RETAINED_TO_12`.

The other five windows are failures of those same two clean starts. During the policy's own slip the fingers close through the marginal opening. By the time bilateral contact is gone, the −13.4 mm windows are already 1.6–4.5 mm clear of the distal tips, with separating speed about 0.45 m/s, and the 2.0 s prefix of −14.0 mm is 7.1 mm clear at a center displacement of 38.9 mm. Those realized states are the separated regime. They stay in the denominator because the rollout began on a clean, oracle-recatchable slip. They are not added to training.

The 1.0 s prefix of −14.0 mm is the intermediate case. Distal-box clearance is still +0.17 mm and the separating speed is only +0.05 m/s, but the finger joints have crossed (−4.4 mm) and the remaining contact is on the finger mesh, with the center at 31.7 mm. The chase command reaches −3.77 m/s and lasts 0.06 s. The center does not come back.

## Three groups

- **Clean and oracle-recatchable:** 8 conditions, all with clearance within 0.03 mm of the distal tip and separating speed about 0.05 m/s. Policy success on the held-out portion is 1/6 windows (the 0.3 s prefix of dz −14.0 mm only).
- **Clean and oracle-unrecoverable:** dz −13.8 mm and dz −13.6 mm. Clearance is about 0, the center is at 29.4 mm, and the separating speed is about 0.34 m/s. The intercept leaves the center there. They are not in the rate.
- **Post-loss separated:** the earlier 36.92 mm / +5.51 mm state, and any policy rollout that arrives already clear of the tip. None of these is a training target or a primary-eval start.

Cleaning the distribution removes the separated state from the learning target. It does not make the held-out first-loss task succeed. The one closed-loop regrasp is the short prefix that already succeeded before this retrain.

POLICY STILL FAILS ON CLEAN ORACLE-RECATChABLE FIRST-LOSS STATES
