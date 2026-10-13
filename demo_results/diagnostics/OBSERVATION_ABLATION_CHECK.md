# Observation ablation check

Each reduced observation retrains the balanced causal Transformer for 120 epochs from torch seed 1. The demonstrations, suffix sampler, family weights, Adam schedule, K=32 prototypes, and recovery7d_airborne_v3 action interface are the same as the published run. The training set is the regenerated set: 18 natural-loss windows and one trajectory each of wrist reseat, gravity-inward sliding, and open-finger recatch. Airborne-carry demonstrations are absent, as in the published training set. The input layer is resized to the kept channels. The previous 7-D action stays a controller-state input and is not one of the observation channels. A0 is the published checkpoint, not a second training run. Its clean score is 15/24 and its natural-loss validation loss is 0.2752, matching the published evaluation. Success is retention through absolute t = 12 on the frozen 24-case benchmark.

## Retained channels

| variant | what is removed | retained indices | dim |
|---|---|---|---:|
| A0 full 45-D | nothing | 0–44 | 45 |
| A1 drop wide-scale hand twist | 39–44 | 0–38 | 39 |
| A2 drop object angular velocity | 36–38 | 0–35, 39–44 | 42 |
| A3 drop object orientation | 33–35 | 0–32, 36–44 | 42 |
| A4 drop object linear and angular velocity | 30–32 and 36–38 | 0–29, 33–35, 39–44 | 39 |
| A5 drop object position and orientation | 27–29 and 33–35 | 0–26, 30–32, 36–44 | 39 |
| A6 tactile and proprioception only | 27–44 | 0–26 | 27 |
| compact | see below | 0–26, 36–38 | 30 |

Semantic groups: 0–26 tactile and proprioceptive prefix; 27–29 object relative position; 30–32 object relative linear velocity; 33–35 object relative orientation; 36–38 object relative angular velocity; 39–44 hand linear and angular velocity repeated at the wider airborne scale.

The compact candidate keeps a group only when the ablation that removes that group drops impact below 3/4, drops sliding, wrist, or open-finger recatch below 4/4, or drops the case total below 14/24. Under that rule only object angular velocity is kept, because A2 misses the skill bar. A4 also removes angular velocity, together with linear velocity, and still retains the same 15 cases. The compact therefore follows the single-group rule rather than the joint ablation. It is the one compact model trained in this study.

## t = 12

| variant | dim | t=12 | impact | sliding | wrist | recatch | airborne | natural | val loss | robustness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0 full 45-D | 45 | 15/24 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.2752 | combined 15, 15, 15 |
| A1 drop wide-scale hand twist | 39 | 15/24 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.1895 | combined 15, 16, 16 |
| A2 drop object angular velocity | 42 | 17/24 | 2/4 | 4/4 | 4/4 | 3/4 | 4/4 | 0/4 | 0.2511 | not run |
| A3 drop object orientation | 42 | 17/24 | 3/4 | 4/4 | 4/4 | 4/4 | 2/4 | 0/4 | 0.2809 | not run |
| A4 drop object twist | 39 | 15/24 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.3362 | not run |
| A5 drop object pose | 39 | 15/24 | 3/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.2226 | not run |
| A6 27-D prefix | 27 | 13/24 | 1/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.1696 | not run |
| compact prefix plus angular velocity | 30 | 14/24 | 2/4 | 4/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.2321 | combined 13, 14, 13 |

Validation loss is the teacher-forced loss on the natural-loss holdout. Final training loss on the retrained models is 0.043 to 0.057, against 0.0547 for the published 45-D run. A6 has the lowest validation loss in the table and the worst impact score, so the observation is not chosen by validation loss.

A1, A4, and A5 retain the same 15 cases as A0. No clean success is lost and no clean failure is gained. A2 loses `impact_z3` and `recatch_suffix_8` and gains all four airborne-carry cases. A3 loses nothing and gains `air_lat_pos` and `air_fast_pos`. A6 loses `impact_zm6` and `impact_phi90`. The compact loses `impact_phi90`. Natural-loss stays 0/4 on every variant. Sliding and wrist stay 4/4 on every variant.

## Behavior where the outcome changes

**A1.** The 15 retained cases are the same ones. `recatch_suffix_8` still finishes, with the recorded mechanism moving from a closed-grip wrist reseat to a recontact that uses a half-scale wrist command and a partly open grip.

**A2.** `impact_z3` still commands a wrist reseat: max |wy| stays 0.80 and the grip stays closed near +1. Mean |vx| and mean vz barely move. The object is not retained. `recatch_suffix_8` changes the grip and the vertical command. Min grip goes from +0.998 to −1, so the hand opens, and mean vz goes from about 0 to +0.14, a downward command. Max |wy| stays 0.80 and mean |vy| rises from 0.002 to 0.034. That case drops. The four new airborne retentions are contact-preserving holds. Grip stays near +1, max |wy| is 0, and mean |vx|, |vy|, and vz stay near 0. Contact is present for the whole policy window. The baseline opens and chases those same cases and drops them. These four successes are holds of a grasp that is still in the fingers at the start of the policy window, not lateral or vertical recatches.

**A3.** Impact, sliding, wrist, and open-finger recatch stay on the same cases. `air_lat_pos` and `air_fast_pos` become the same closed-grip hold: mean vz about 0, grip near +1, no wrist command. `air_lat_neg` and `air_slow_neg` still open, chase laterally, and drop.

**A6.** `impact_zm6` still issues the wrist command (max |wy| 0.80) and opens the grip, from min grip +0.998 to −1. Mean vz rises from about 0 to +0.053. The object drops. `impact_phi90`, which the baseline holds with a closed grip and almost no wrist, also opens (min grip −1) and mean vz rises from +0.011 to +0.093. It drops. Sliding, wrist, and open-finger recatch stay at 4/4, and `impact_z3` remains retained. Removing every object-relative channel changes impact by opening the hand.

**Compact.** `impact_phi90` opens (min grip −1) and mean vz rises from +0.011 to +0.069. Max |wy| stays near 0. The case drops. The other 14 clean successes remain, including sliding, wrist, and open-finger recatch.

## Combined sensor profile

The profile is the Part A moderate combination: 2 mm position noise, 1 deg orientation noise, 0.03 m/s linear velocity, 0.3 rad/s angular velocity, 40 ms object-channel delay, τ = 50 ms smoothing, 0.5 mm center-of-pressure noise, and 1 N force noise.

The full 45-D policy scores 15/24 on the clean benchmark and 15/24 on each of the three noise seeds. A1 scores 15/24 clean. Under the same profile it scores 15, 16, and 16. Impact stays 3/4, sliding 4/4, wrist 4/4, and open-finger recatch 4/4 on every seed. The extra success on seeds 1 and 2 is `air_lat_pos`. Contact is lost during the policy window, the grip opens, and the wrist command reaches 1, while the mean lateral command stays near 0.007. The t = 12 label is the grasp after the nominal continuation. The 30-D compact scores 14/24 clean and 13, 14, and 13 under the profile. Its impact count is 1/4, 2/4, and 1/4. Sliding, wrist, and recatch stay 4/4. The compact is less robust than the 45-D policy, and A1 is not.

A4 and A5 match the clean 15/24, so pose alone and twist alone are each sufficient for the current retained families on this seed. They were not the derived compact, and they were not run under the combined profile. The 30-D candidate that the group rule produced loses `impact_phi90` and loses further impact cases once the sensor profile is applied, so it is not a replacement for the 45-D input.

DUPLICATE HAND-TWIST CHANNELS CAN BE REMOVED WITHOUT LOSS
