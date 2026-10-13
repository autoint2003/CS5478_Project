# Geometry-based recovery switch

The 39-D policy, the v3 action map, and the recovery skills are unchanged. Entry and exit are a hysteresis rule on the object pose in the hand.

At the start of the stable lift the controller records the hand-frame pose `p0_H`, `R0_H`. Every 20 ms it computes

- `e_pos = ||p_obj_H - p0_H||`
- `e_rot = angle(R0_H^T R_obj_H)`

The velocity term of the old deterioration score is not an entry signal. The old score is logged only so its trigger time can be compared.

## What the signals do

Undisturbed grasps stay inside a small band. A full nominal lift peaks at 0.15 mm and 0.009 rad. A seated hold peaks at 0.99 mm and 0.055 rad. Neither crosses the entry line. False-entry count: 0/2.

The ball is different. On `impact_zm6` the contact sample, 8 ms after contact, has `e_pos` 5.2 mm and both pads off. That is the sample where the deterioration score jumps to 21.8 and the old switch fired. One control step later `e_pos` is 11.7 mm and some contact has returned. By the post-park snapshot it is 12.1 mm and 0.29 rad, and it stays there.

| Disturbance | Max `e_pos` before the post-park point | Max `e_rot` | Position trigger | Orientation trigger |
| --- | ---: | ---: | ---: | ---: |
| `zm6` ball, shared by the sliding suffixes, wrist suffixes, and open-finger suffixes | 12.3 mm | 0.289 rad | 2.92 s | 2.90 s |
| `zp6` ball | 13.0 mm | 0.244 rad | 2.90 s | 2.90 s |
| `phi90` ball | 0.79 mm | 0.039 rad | none | none |
| `z3` ball | 11.5 mm | 0.117 rad | 2.92 s | 2.92 s |
| 1 mm shifts | 13.3 mm | 0.289 rad | 2.92 s | 2.90 s |

Position alone crosses 6 mm on every ball except `phi90`. Orientation crosses 0.10 rad on those same balls and does not cross it on `phi90`. The wrist and open-finger suffix starts sit at about 12 mm and 0.29 rad, so position sees them too. Orientation is not required. `phi90` never leaves the undisturbed band: 0.8 mm and 0.039 rad, both below the seated hold. Nominal control keeps that grasp through `t = 12` with no recovery entry.

The entry line is therefore position only: `e_pos > 6 mm` for 40 ms (three control samples). The 5.2 mm collision sample does not qualify. The switch falls on the later sample, after the ball has parked.

## Exit region from the retained handoffs

The prepared-snapshot successes were replayed: impact, controlled sliding, wrist reseating, and open-finger recatch. At the handoff where nominal continuation already retains the object, the pose error is 0.5–12.7 mm and 0.027–0.302 rad, with relative speed at most 1 mm/s. The policy does not return to `p0_H`. It settles on an off-center grasp of about 12 mm and, for several of those skills, about 0.29 rad.

An exit ball smaller than the 6 mm entry line would reject every one of those handoffs. The exit region is the measured continuation envelope, not a tighter center:

- both pads in contact
- `e_pos <= 14.7 mm` (widest retained handoff, 12.7 mm, plus 2 mm)
- `e_rot <= 0.353 rad`
- relative speed `<= 0.08 m/s`

held for 100 ms. The handoff is one-way. After it, nominal control runs to absolute `t = 12` and does not re-enter, because the continuation pose is still outside the 6 mm line.

## Online result

| Pipeline | Retained to `t = 12` |
| --- | ---: |
| A, prepared snapshot | 15/24 |
| B, deterioration score online | 5/24 |
| C, geometry switch | 19/24 |

C keeps every case in the snapshot 15 and adds the four airborne releases. `impact_zp6` and the four natural-loss offsets still drop.

Wrist, open-finger, and the sliding suffixes are the same `zm6` ball. The geometry switch on that ball is 2.92 s, 68 ms after contact and 60 ms after the old score. One recovery covers those rows. The 1 mm shifts are triggered at the same 2.92 s, which is before the shift instant, so they follow that same recovery and also retain.

| Case | Disturbance | Old trigger | Geometry trigger | Reason | Exit | `e_pos` at exit | Speed at exit | `t = 12` |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `impact_zm6` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `impact_zp6` | zp6 | 2.860 s | 2.900 s | position | 3.000 s | 12.8 mm | 0.008 m/s | drop |
| `impact_phi90` | phi90 | 2.860 s | none |  |  |  |  | retained |
| `impact_z3` | z3 | 2.860 s | 2.920 s | position | 3.280 s | 12.0 mm | 0.012 m/s | retained |
| `slide_suffix_20` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `slide_suffix_36` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `slide_dy1` | dy1 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `slide_dz1` | dz1 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `wrist_suffix_3` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `wrist_suffix_6` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `wrist_suffix_9` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `wrist_suffix_12` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `recatch_suffix_8` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `recatch_suffix_16` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `recatch_suffix_24` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `recatch_suffix_28` | zm6 | 2.860 s | 2.920 s | position | 3.280 s | 12.5 mm | 0.013 m/s | retained |
| `air_lat_pos` | carry +0.30 | 4.596 s | 7.296 s | position | 7.396 s | 6.0 mm | 0.0005 m/s | retained |
| `air_lat_neg` | carry −0.30 | 4.596 s | 6.596 s | position | 6.696 s | 6.0 mm | 0.0008 m/s | retained |
| `air_fast_pos` | carry +0.45 | 4.596 s | 6.076 s | position | 6.176 s | 6.0 mm | 0.0007 m/s | retained |
| `air_slow_neg` | carry −0.20 | 4.596 s | 7.596 s | position | 7.696 s | 6.0 mm | 0.0006 m/s | retained |
| `nl_h_dz14.00` | offset −14.00 mm | 4.516 s | 4.556 s | position | 4.656 s | 14.1 mm | 0.0006 m/s | drop |
| `nl_h_dz14.25` | offset −14.25 mm | 4.516 s | 4.556 s | position | 4.656 s | 14.3 mm | 0.0008 m/s | drop |
| `nl_h_dy` | offset −14.10 mm, 0.12 mm y | 4.516 s | 4.556 s | position | 4.656 s | 14.2 mm | 0.0007 m/s | drop |
| `nl_h_dx` | offset −14.70 mm, 0.10 mm x | 4.516 s | 4.556 s | position | 7.136 s, fall | 12.8 mm | 0.94 m/s | drop |

On the retained `zm6` and `z3` impacts the policy runs 18 steps (0.36 s) and hands back a 12 mm grasp at about 0.01 m/s. Nominal control then holds it to `t = 12`.

## Impact timing

| Case | Contact | Old score | Geometry | Pose at the old sample | Pose at the geometry switch | Pads at the switch | `t = 12` |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `impact_zm6` | 2.852 s | 2.860 s (+8 ms) | 2.920 s (+68 ms) | 5.2 mm, 0.102 rad, pads off | 12.1 mm, 0.288 rad, speed 0.005 m/s | 1, 2 | retained |
| `impact_zp6` | 2.852 s | 2.860 s (+8 ms) | 2.900 s (+48 ms) | 6.0 mm, 0.107 rad | 12.6 mm, 0.243 rad | 2, 2 | drop |
| `impact_phi90` | 2.850 s | 2.860 s (+10 ms) | no switch | 0.6 mm, 0.031 rad, pads 9 and 4 |  |  | retained |
| `impact_z3` | 2.852 s | 2.860 s (+8 ms) | 2.920 s (+68 ms) | 5.4 mm, 0.052 rad | 11.2 mm, 0.116 rad | 3, 2 | retained |

The geometry rule does not switch on the 8–10 ms collision sample. For `zm6` and `z3` it switches after the ball has parked, on the settled offset the snapshot policy already knows how to hold. `phi90` produces no pose error above the healthy band, so the premature score is ignored and the nominal grasp continues.

`impact_zp6` is delayed as well, from 8 ms to 48 ms, and still drops. The snapshot benchmark already escapes on that ball. The geometry exit there returns after 100 ms, and nominal control does not keep the object.

Natural loss is the remaining hole. A 14 mm offset crosses 6 mm in 40 ms while the pads are still in contact. That pose is inside the 14.7 mm exit envelope taken from the impact handoffs, so three of the four offsets are handed back after 100 ms and then drop. The exit region is wide enough for the retained off-center grasps, and a 14 mm offset fits in the same ball without being a grasp nominal control can keep. The thresholds were not adjusted for that family.

GEOMETRY-BASED SWITCH IMPROVES AUTONOMOUS RECOVERY
