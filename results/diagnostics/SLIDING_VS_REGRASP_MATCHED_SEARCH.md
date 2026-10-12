# Matched sliding versus release-regrasp recovery

Privileged contact, pose, and force selected the recovery commands. Every MuJoCo command was a clipped `recovery7d_airborne_v3` 7-vector. Snapshots were restored from the saved TRAIN initial conditions. The cylinder, hand, ball, contact state, friction, and gravity were not edited as an intervention. No policy weights were updated. VALIDATION and FINAL_TEST were not searched. The sealed GRPO pilot was not opened for writing.

A family result is `VERIFIED_SUCCESS` only when replay from the original snapshot reaches the existing nominal basin, hands off, and holds for 10 s under nominal-only control. The same nominal call continues to 12 s and that diagnostic is reported separately. Sliding `VERIFIED_SUCCESS` adds three physical requirements: at least one finger stays in contact on every recovery step, the unloaded grip stays above action -0.85, and two replays lose the 10 s hold. One replay removes translation only on the unloaded steps and keeps the later close-and-lift. The other forces the grip closed. Tangential motion and contact migration must also be present. A 10 s hold that passes through a zero-contact step is `HOLD_BUT_CONTACT_LOST` and is not counted as controlled sliding.

## 1. Can contact-preserving sliding recover most of these 18 initial conditions?

Strict controlled sliding verified on 5 of 18 TRAIN initial conditions. Release-regrasp replay verified on 18 of 18. Hybrid search verified on 18 of 18. Sliding holds that lost both fingers for at least one step: 11. Continuous-contact holds whose ablations did not isolate sliding: 0. Drops under every sliding law: 2.

Under the continuous-contact definition, controlled sliding does not recover most of these 18 initial conditions. The 11 contact-gap holds did reach the nominal basin and stayed grasped for both 10 s and 12 s, with both fingers off for 0.04–0.10 s. Those runs are not counted as controlled sliding. Every 10 s success in this comparison also passed the 12 s diagnostic.

The sample has 18 initial conditions. A difference of a few successes is not a general ranking of the mechanisms.

## 2. Laws, legal actions, and budgets

The grip band was frozen after a short profile on `ball_train_0001`, `ball_train_0014`, and `ball_train_0012`. Grip actions at or below -0.75 lost both contacts within a few steps. Grip actions from -0.45 to -0.62, with hand speed capped between 0.30 and 0.45 m/s and a close-and-lift once the pinch error was inside 3.2 mm, were the sliding laws used for every TRAIN initial condition. A unilateral contact raises the grip only to -0.40. A force brake, used on three of the six laws, adds 0.10 to the grip action when the normal force falls below 0.8 N. Neither change clamps at the secure torque.

Hybrid search starts in that sliding law. It switches to the existing `chase_lost` release law when both fingers lose contact for two consecutive steps. It also switches when the pinch error stays above 6 mm and has moved less than 1 mm after 2.0 s without improvement, or when 1.60 s have passed with less than 2 mm of progress and the error is still above 8 mm. A slide that is already reducing the pinch error keeps running.

Release-regrasp is the saved verified trajectory from the previous privileged search, replayed once. Its budget is the full previous attempt log for that initial condition, including laws that ran after the first success. The new families stop after the first strict success in that family, then mark the remaining frozen laws skipped. Search effort is therefore reported in seconds and attempt counts, not as a claim that one family is cheaper.

```
{
  "version": "recovery7d_airborne_v3",
  "vx_vy_mps": 1.0,
  "vz_down_mps": 3.8,
  "vz_up_mps": 1.0,
  "wrist_rad_s": 4.0,
  "wrist_command_norm": 4.0,
  "grip_open_nm": 2.0,
  "grip_secure_nm": -18.0,
  "ball_parked": false,
  "qpos_written": false
}
```

Wall time in the completing process: 635.5 s. Logged attempt time, summed over families: 476.4 s.

## 3. Per-initial-condition comparison

| IC | direction | pinch m | object ori rad | contacts | spin rad/s | sliding | regrasp | hybrid |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| `ball_train_0001` | +X | 0.01475 | 0.15183 | 2/6 | 3.4014 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0002` | +X | 0.01564 | 0.14612 | 2/2 | 2.8233 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0003` | +X | 0.01091 | 0.16273 | 3/7 | 2.4904 | VERIFIED_SUCCESS | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0004` | +X | 0.01478 | 0.1546 | 2/6 | 3.443 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0005` | +X | 0.01567 | 0.14809 | 2/3 | 2.6844 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0006` | +X | 0.01099 | 0.16053 | 2/7 | 2.7788 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0007` | +X | 0.01498 | 0.15202 | 1/5 | 4.1661 | VERIFIED_SUCCESS | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0008` | +X | 0.01584 | 0.14759 | 2/2 | 3.1245 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0009` | +X | 0.0112 | 0.16054 | 2/8 | 2.8298 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0010` | oblique_+X_+Y | 0.01741 | 0.84989 | 5/3 | 0.5724 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0011` | +Y | 0.00946 | 0.40985 | 5/6 | 0.0361 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0012` | oblique_+X_+Y | 0.02519 | 1.0363 | 1/2 | 1.0661 | DROP_OR_ESCAPE | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0013` | +Y | 0.01418 | 0.57692 | 1/1 | 0.1014 | VERIFIED_SUCCESS | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0014` | +Y | 0.00749 | 0.49892 | 6/4 | 0.383 | VERIFIED_SUCCESS | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0015` | oblique_+X_+Y | 0.00803 | 0.29516 | 8/13 | 0.5324 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0016` | +Y | 0.01133 | 0.67073 | 6/2 | 0.6564 | HOLD_BUT_CONTACT_LOST | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0017` | oblique_+X_+Y | 0.01169 | 0.45861 | 2/5 | 1.3697 | DROP_OR_ESCAPE | VERIFIED_SUCCESS | VERIFIED_SUCCESS |
| `ball_train_0018` | +Y | 0.00936 | 0.58613 | 5/8 | 0.3398 | VERIFIED_SUCCESS | VERIFIED_SUCCESS | VERIFIED_SUCCESS |

| IC | slide s | slide z-loss m | slide contact-loss s | regrasp s | regrasp budget s | regrasp z-loss m | regrasp contact-loss s | hybrid s | hybrid z-loss m | hybrid contact-loss s | hybrid switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ball_train_0001` | 0.980 | 0.0533 | 0.080 | 0.620 | 10.5 | 0.0561 | 0.480 | 0.640 | 0.0532 | 0.060 | None |
| `ball_train_0002` | 0.620 | 0.0529 | 0.080 | 0.540 | 26.5 | 0.0554 | 0.400 | 0.660 | 0.0523 | 0.060 | None |
| `ball_train_0003` | 0.680 | 0.0567 | 0.000 | 0.560 | 28.2 | 0.0593 | 0.420 | 0.720 | 0.0565 | 0.080 | None |
| `ball_train_0004` | 0.700 | 0.0695 | 0.040 | 1.880 | 26.9 | 0.0724 | 0.900 | 0.680 | 0.0685 | 0.060 | None |
| `ball_train_0005` | 0.620 | 0.0684 | 0.060 | 0.560 | 20.2 | 0.0713 | 0.420 | 0.700 | 0.0677 | 0.060 | None |
| `ball_train_0006` | 0.800 | 0.0719 | 0.040 | 0.620 | 21.7 | 0.0770 | 0.500 | 0.840 | 0.0723 | 0.100 | 23 |
| `ball_train_0007` | 0.880 | 0.0833 | 0.000 | 0.620 | 26.9 | 0.0840 | 0.480 | 0.740 | 0.0834 | 0.060 | None |
| `ball_train_0008` | 0.920 | 0.0825 | 0.080 | 0.940 | 11.3 | 0.0866 | 0.720 | 0.900 | 0.0832 | 0.240 | 19 |
| `ball_train_0009` | 0.840 | 0.0875 | 0.060 | 0.940 | 21.6 | 0.0930 | 0.500 | 0.820 | 0.0871 | 0.060 | None |
| `ball_train_0010` | 1.000 | 0.0878 | 0.080 | 0.780 | 16.7 | 0.0869 | 0.660 | 1.000 | 0.0878 | 0.080 | None |
| `ball_train_0011` | 0.860 | 0.0938 | 0.100 | 0.640 | 16.9 | 0.0943 | 0.500 | 1.400 | 0.0927 | 0.200 | 48 |
| `ball_train_0012` | 1.240 | 0.1067 | 1.080 | 0.900 | 12.4 | 0.0844 | 0.520 | 0.940 | 0.1062 | 0.460 | 7 |
| `ball_train_0013` | 1.220 | 0.0906 | 0.000 | 0.940 | 18.6 | 0.1062 | 0.640 | 1.220 | 0.0906 | 0.000 | None |
| `ball_train_0014` | 1.400 | 0.0923 | 0.000 | 1.160 | 29.0 | 0.1084 | 0.640 | 1.400 | 0.0923 | 0.000 | None |
| `ball_train_0015` | 1.020 | 0.0944 | 0.060 | 0.620 | 28.3 | 0.1013 | 0.480 | 1.340 | 0.0938 | 0.040 | None |
| `ball_train_0016` | 0.800 | 0.0609 | 0.100 | 2.820 | 24.7 | 0.0747 | 1.200 | 0.820 | 0.0605 | 0.040 | None |
| `ball_train_0017` | 1.280 | 0.1127 | 1.120 | 0.600 | 26.4 | 0.0993 | 0.460 | 0.940 | 0.1152 | 0.440 | 8 |
| `ball_train_0018` | 1.320 | 0.0606 | 0.000 | 1.260 | 19.1 | 0.0762 | 0.920 | 1.060 | 0.0605 | 0.060 | None |

## 4. Where sliding fails while regrasp holds

- `ball_train_0001` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00048 m, contact loss 0.08 s, migration 0.02809 m). Entry pinch 0.01475 m, object orientation 0.15183 rad, spin 3.4014 rad/s, contacts 2/6.
- `ball_train_0002` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00112 m, contact loss 0.08 s, migration 0.02987 m). Entry pinch 0.01564 m, object orientation 0.14612 rad, spin 2.8233 rad/s, contacts 2/2.
- `ball_train_0004` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00161 m, contact loss 0.04 s, migration 0.0259 m). Entry pinch 0.01478 m, object orientation 0.1546 rad, spin 3.443 rad/s, contacts 2/6.
- `ball_train_0005` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00059 m, contact loss 0.06 s, migration 0.02865 m). Entry pinch 0.01567 m, object orientation 0.14809 rad, spin 2.6844 rad/s, contacts 2/3.
- `ball_train_0006` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00117 m, contact loss 0.04 s, migration 0.0269 m). Entry pinch 0.01099 m, object orientation 0.16053 rad, spin 2.7788 rad/s, contacts 2/7.
- `ball_train_0008` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00068 m, contact loss 0.08 s, migration 0.02833 m). Entry pinch 0.01584 m, object orientation 0.14759 rad, spin 3.1245 rad/s, contacts 2/2.
- `ball_train_0009` +X: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00208 m, contact loss 0.06 s, migration 0.02093 m). Entry pinch 0.0112 m, object orientation 0.16054 rad, spin 2.8298 rad/s, contacts 2/8.
- `ball_train_0010` oblique_+X_+Y: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00272 m, contact loss 0.08 s, migration 0.03 m). Entry pinch 0.01741 m, object orientation 0.84989 rad, spin 0.5724 rad/s, contacts 5/3.
- `ball_train_0011` +Y: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00034 m, contact loss 0.1 s, migration 0.039 m). Entry pinch 0.00946 m, object orientation 0.40985 rad, spin 0.0361 rad/s, contacts 5/6.
- `ball_train_0012` oblique_+X_+Y: sliding `DROP_OR_ESCAPE` (min pinch 0.02463 m, contact loss 1.08 s, migration 0.02085 m). Entry pinch 0.02519 m, object orientation 1.0363 rad, spin 1.0661 rad/s, contacts 1/2.
- `ball_train_0015` oblique_+X_+Y: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.0002 m, contact loss 0.06 s, migration 0.02563 m). Entry pinch 0.00803 m, object orientation 0.29516 rad, spin 0.5324 rad/s, contacts 8/13.
- `ball_train_0016` +Y: sliding `HOLD_BUT_CONTACT_LOST` (min pinch 0.00168 m, contact loss 0.1 s, migration 0.03616 m). Entry pinch 0.01133 m, object orientation 0.67073 rad, spin 0.6564 rad/s, contacts 6/2.
- `ball_train_0017` oblique_+X_+Y: sliding `DROP_OR_ESCAPE` (min pinch 0.01182 m, contact loss 1.12 s, migration 0.01681 m). Entry pinch 0.01169 m, object orientation 0.45861 rad, spin 1.3697 rad/s, contacts 2/5.

## 5. Hybrid contact loss and mechanism sequence

The release chase was entered on 5 initial conditions: ball_train_0006, ball_train_0008, ball_train_0011, ball_train_0012, ball_train_0017. On `ball_train_0012` and `ball_train_0017` that switch converted a sliding drop into a 10 s and 12 s hold, and cut the zero-contact interval from about 1.1 s to about 0.45 s. On the other switched cases the sliding controller had already held, and the chase sometimes lengthened the contact gap. Where the hybrid runner never switched, it stayed on its sliding law. That law can still have a short contact gap on a snapshot where a different sliding law had none. Hybrid search did not uniformly reduce contact loss.

- `ball_train_0001`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.08 s, z-loss hybrid 0.0532 m versus sliding 0.0533 m. Labels ['reclose'].
- `ball_train_0002`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.08 s, z-loss hybrid 0.0523 m versus sliding 0.0529 m. Labels ['reclose'].
- `ball_train_0003`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.08 s, sliding contact loss 0.0 s, z-loss hybrid 0.0565 m versus sliding 0.0567 m. Labels ['reclose'].
- `ball_train_0004`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.04 s, z-loss hybrid 0.0685 m versus sliding 0.0695 m. Labels ['reclose'].
- `ball_train_0005`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.06 s, z-loss hybrid 0.0677 m versus sliding 0.0684 m. Labels ['reclose'].
- `ball_train_0006`: hybrid `VERIFIED_SUCCESS`, switch step 23, contact loss 0.1 s, sliding contact loss 0.04 s, z-loss hybrid 0.0723 m versus sliding 0.0719 m. Labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose'].
- `ball_train_0007`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.0 s, z-loss hybrid 0.0834 m versus sliding 0.0833 m. Labels ['reclose'].
- `ball_train_0008`: hybrid `VERIFIED_SUCCESS`, switch step 19, contact loss 0.24 s, sliding contact loss 0.08 s, z-loss hybrid 0.0832 m versus sliding 0.0825 m. Labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose'].
- `ball_train_0009`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.06 s, z-loss hybrid 0.0871 m versus sliding 0.0875 m. Labels ['reclose'].
- `ball_train_0010`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.08 s, sliding contact loss 0.08 s, z-loss hybrid 0.0878 m versus sliding 0.0878 m. Labels [].
- `ball_train_0011`: hybrid `VERIFIED_SUCCESS`, switch step 48, contact loss 0.2 s, sliding contact loss 0.1 s, z-loss hybrid 0.0927 m versus sliding 0.0938 m. Labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose'].
- `ball_train_0012`: hybrid `VERIFIED_SUCCESS`, switch step 7, contact loss 0.46 s, sliding contact loss 1.08 s, z-loss hybrid 0.1062 m versus sliding 0.1067 m. Labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose'].
- `ball_train_0013`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.0 s, sliding contact loss 0.0 s, z-loss hybrid 0.0906 m versus sliding 0.0906 m. Labels [].
- `ball_train_0014`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.0 s, sliding contact loss 0.0 s, z-loss hybrid 0.0923 m versus sliding 0.0923 m. Labels [].
- `ball_train_0015`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.04 s, sliding contact loss 0.06 s, z-loss hybrid 0.0938 m versus sliding 0.0944 m. Labels [].
- `ball_train_0016`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.04 s, sliding contact loss 0.1 s, z-loss hybrid 0.0605 m versus sliding 0.0609 m. Labels ['reclose'].
- `ball_train_0017`: hybrid `VERIFIED_SUCCESS`, switch step 8, contact loss 0.44 s, sliding contact loss 1.12 s, z-loss hybrid 0.1152 m versus sliding 0.1127 m. Labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose'].
- `ball_train_0018`: hybrid `VERIFIED_SUCCESS`, switch step None, contact loss 0.06 s, sliding contact loss 0.0 s, z-loss hybrid 0.0605 m versus sliding 0.0606 m. Labels [].

## 6. Conditions on this set

Strict sliding successes: 5. Entry pinch 0.007 to 0.015 m, object orientation 0.152 to 0.586 rad, spin 0.101 to 4.166 rad/s.
TRAIN cases without strict sliding: 13. Entry pinch 0.008 to 0.025 m, object orientation 0.146 to 1.036 rad, spin 0.036 to 3.443 rad/s.
These ranges describe the 18 saved TRAIN snapshots only. Overlap between the two groups means no single entry threshold separates them here.
The two sliding drops are `ball_train_0012` (pinch 25.2 mm, object orientation 1.04 rad, contacts 1/2) and `ball_train_0017` (spin 1.37 rad/s). Unload lost both fingers for about a second and the pinch error never entered the basin. Strict sliding still occurred on a 4.2 rad/s +X snapshot (`ball_train_0007`) and on slower +Y snapshots, so spin alone does not decide the mechanism. Release-regrasp remains the fallback when the unloaded grip loses the cylinder. Continuous-contact sliding is preferable on the five verified cases because the zero-contact interval is 0 s there and 0.42–0.92 s on the matched release replay, while both reach the same 10 s and 12 s hold.

## 7. Action primitives and K=36

For each saved recovery action, `d_t` is the L-infinity distance to the nearest of the 36 prototypes in `policy_B_init.pt`. A step is inside the current residual bound when `d_t <= 0.25`. The quantized command is that prototype plus the residual clipped to ±0.25. No token was added and the residual bound was not changed.
- `ball_train_0001` regrasp `release_short`: fraction 0.355, max L-inf 0.482, mean absolute gap [0.0783, 0.2218, 0.2404, 0.0024, 0.0001, 0.0, 0.0], quantized 10 s not replayed.
- `ball_train_0001` hybrid `slide_58`: fraction 1.0, max L-inf 0.194, mean absolute gap [0.0017, 0.0066, 0.0774, 0.0023, 0.0001, 0.0, 0.0756], quantized 10 s True.
- `ball_train_0002` regrasp `lift_soft`: fraction 0.296, max L-inf 0.419, mean absolute gap [0.0831, 0.2352, 0.1917, 0.0033, 0.0001, 0.0, 0.0], quantized 10 s not replayed.
- `ball_train_0002` hybrid `slide_58`: fraction 1.0, max L-inf 0.18, mean absolute gap [0.0031, 0.0065, 0.0739, 0.0022, 0.0001, 0.0, 0.0764], quantized 10 s True.
- `ball_train_0003` sliding `slide_50`: fraction 1.0, max L-inf 0.205, mean absolute gap [0.0034, 0.004, 0.0746, 0.0022, 0.0001, 0.0, 0.0176], quantized 10 s True.
- `ball_train_0003` regrasp `lift_soft`: fraction 0.286, max L-inf 0.441, mean absolute gap [0.0793, 0.22, 0.2122, 0.0033, 0.0001, 0.0, 0.0], quantized 10 s not replayed.
- `ball_train_0003` hybrid `slide_58`: fraction 1.0, max L-inf 0.182, mean absolute gap [0.0014, 0.0039, 0.0675, 0.0022, 0.0001, 0.0, 0.0811], quantized 10 s True.
- `ball_train_0004` regrasp `release_long`: fraction 0.574, max L-inf 0.52, mean absolute gap [0.0821, 0.1531, 0.194, 0.0626, 0.0603, 0.0641, 0.0622], quantized 10 s not replayed.
- `ball_train_0004` hybrid `slide_58`: fraction 1.0, max L-inf 0.181, mean absolute gap [0.0018, 0.0059, 0.0788, 0.0024, 0.0001, 0.0, 0.0776], quantized 10 s True.
- `ball_train_0005` regrasp `release_long`: fraction 0.393, max L-inf 0.467, mean absolute gap [0.0752, 0.1975, 0.2371, 0.0018, 0.0, 0.0, 0.0], quantized 10 s not replayed.
- `ball_train_0005` hybrid `slide_58`: fraction 1.0, max L-inf 0.184, mean absolute gap [0.0027, 0.0058, 0.0778, 0.0024, 0.0001, 0.0, 0.0777], quantized 10 s True.
- `ball_train_0006` regrasp `lift_soft`: fraction 0.226, max L-inf 0.471, mean absolute gap [0.0871, 0.2481, 0.2153, 0.0033, 0.0001, 0.0, 0.0], quantized 10 s not replayed.
- `ball_train_0006` hybrid `slide_58`: fraction 0.952, max L-inf 0.414, mean absolute gap [0.0077, 0.0241, 0.0958, 0.0021, 0.0001, 0.0, 0.0605], quantized 10 s True.
- `ball_train_0007` sliding `slide_45`: fraction 1.0, max L-inf 0.181, mean absolute gap [0.0013, 0.003, 0.0657, 0.0025, 0.0, 0.0, 0.0243], quantized 10 s True.
- `ball_train_0007` regrasp `lift_soft`: fraction 0.419, max L-inf 0.431, mean absolute gap [0.0804, 0.1884, 0.209, 0.0037, 0.0001, 0.0, 0.0], quantized 10 s not replayed.
- `ball_train_0007` hybrid `slide_58`: fraction 1.0, max L-inf 0.186, mean absolute gap [0.0015, 0.0052, 0.0805, 0.0026, 0.0001, 0.0, 0.0795], quantized 10 s True.
- `ball_train_0008` regrasp `chase_lost`: fraction 0.34, max L-inf 0.561, mean absolute gap [0.0705, 0.2112, 0.2595, 0.0018, 0.0, 0.0, 0.0443], quantized 10 s not replayed.
- `ball_train_0008` hybrid `slide_58`: fraction 0.8, max L-inf 0.387, mean absolute gap [0.0272, 0.0781, 0.1297, 0.0021, 0.0001, 0.0, 0.0462], quantized 10 s True.
- `ball_train_0009` regrasp `release_short`: fraction 0.596, max L-inf 0.52, mean absolute gap [0.0825, 0.1553, 0.1882, 0.0722, 0.059, 0.0656, 0.0548], quantized 10 s not replayed.
- `ball_train_0009` hybrid `slide_58`: fraction 1.0, max L-inf 0.181, mean absolute gap [0.0016, 0.003, 0.0708, 0.0025, 0.0001, 0.0, 0.081], quantized 10 s True.
- `ball_train_0010` sliding `slide_58`: fraction 1.0, max L-inf 0.228, mean absolute gap [0.0833, 0.0028, 0.0613, 0.0022, 0.0004, 0.0001, 0.086], quantized 10 s True.
- `ball_train_0010` regrasp `chase_lost`: fraction 0.256, max L-inf 0.636, mean absolute gap [0.1132, 0.251, 0.2402, 0.0017, 0.0003, 0.0001, 0.0267], quantized 10 s not replayed.
- `ball_train_0010` hybrid `slide_58`: fraction 1.0, max L-inf 0.228, mean absolute gap [0.0833, 0.0028, 0.0613, 0.0022, 0.0004, 0.0001, 0.086], quantized 10 s True.
- `ball_train_0011` regrasp `release_long`: fraction 0.344, max L-inf 0.52, mean absolute gap [0.1032, 0.2077, 0.2189, 0.0014, 0.0004, 0.0001, 0.0163], quantized 10 s not replayed.
- `ball_train_0011` hybrid `slide_58`: fraction 0.929, max L-inf 0.394, mean absolute gap [0.0943, 0.0294, 0.0677, 0.0014, 0.0005, 0.0, 0.0734], quantized 10 s True.
- `ball_train_0012` regrasp `lift_soft`: fraction 0.4, max L-inf 0.451, mean absolute gap [0.122, 0.171, 0.189, 0.0618, 0.0716, 0.0618, 0.0551], quantized 10 s not replayed.
- `ball_train_0012` hybrid `slide_58`: fraction 0.553, max L-inf 0.52, mean absolute gap [0.09, 0.1418, 0.1583, 0.0016, 0.0003, 0.0001, 0.0285], quantized 10 s False.
- `ball_train_0013` sliding `slide_45`: fraction 1.0, max L-inf 0.186, mean absolute gap [0.1027, 0.0017, 0.0449, 0.0016, 0.0006, 0.0, 0.0251], quantized 10 s True.
- `ball_train_0013` regrasp `lift_soft`: fraction 0.404, max L-inf 0.447, mean absolute gap [0.1382, 0.1889, 0.204, 0.0564, 0.0601, 0.0611, 0.0422], quantized 10 s not replayed.
- `ball_train_0013` hybrid `slide_45`: fraction 1.0, max L-inf 0.186, mean absolute gap [0.1027, 0.0017, 0.0449, 0.0016, 0.0006, 0.0, 0.0251], quantized 10 s True.
- `ball_train_0014` sliding `slide_58`: fraction 1.0, max L-inf 0.199, mean absolute gap [0.0874, 0.0016, 0.0393, 0.0014, 0.0003, 0.0, 0.0854], quantized 10 s True.
- `ball_train_0014` regrasp `chase_lost`: fraction 0.534, max L-inf 0.518, mean absolute gap [0.1227, 0.1734, 0.1911, 0.0698, 0.0737, 0.0683, 0.0499], quantized 10 s not replayed.
- `ball_train_0014` hybrid `slide_58`: fraction 1.0, max L-inf 0.199, mean absolute gap [0.0874, 0.0016, 0.0393, 0.0014, 0.0003, 0.0, 0.0854], quantized 10 s True.
- `ball_train_0015` regrasp `lift_soft`: fraction 0.29, max L-inf 0.433, mean absolute gap [0.0786, 0.215, 0.2159, 0.0032, 0.0005, 0.0001, 0.0], quantized 10 s not replayed.
- `ball_train_0015` hybrid `slide_58`: fraction 1.0, max L-inf 0.186, mean absolute gap [0.0888, 0.0021, 0.0399, 0.002, 0.0004, 0.0, 0.0875], quantized 10 s True.
- `ball_train_0016` regrasp `release_long`: fraction 0.589, max L-inf 0.52, mean absolute gap [0.09, 0.1602, 0.1735, 0.0547, 0.0586, 0.0696, 0.0483], quantized 10 s not replayed.
- `ball_train_0016` hybrid `slide_58`: fraction 1.0, max L-inf 0.183, mean absolute gap [0.0944, 0.0033, 0.056, 0.0013, 0.0004, 0.0001, 0.0795], quantized 10 s True.
- `ball_train_0017` regrasp `lift_soft`: fraction 0.3, max L-inf 0.443, mean absolute gap [0.0787, 0.2264, 0.206, 0.0033, 0.0006, 0.0001, 0.0], quantized 10 s not replayed.
- `ball_train_0017` hybrid `slide_58`: fraction 0.596, max L-inf 0.52, mean absolute gap [0.11, 0.1088, 0.1567, 0.0016, 0.0003, 0.0001, 0.0315], quantized 10 s False.
- `ball_train_0018` sliding `slide_45`: fraction 1.0, max L-inf 0.193, mean absolute gap [0.0796, 0.0015, 0.0338, 0.0012, 0.0004, 0.0, 0.0255], quantized 10 s True.
- `ball_train_0018` regrasp `lift_soft`: fraction 0.27, max L-inf 0.682, mean absolute gap [0.1249, 0.2342, 0.2405, 0.0615, 0.0755, 0.0565, 0.0923], quantized 10 s not replayed.
- `ball_train_0018` hybrid `slide_58`: fraction 1.0, max L-inf 0.184, mean absolute gap [0.0934, 0.002, 0.0431, 0.0012, 0.0003, 0.0001, 0.0857], quantized 10 s True.
Mean of the per-trajectory mean absolute gaps, channels vx, vy, vz, wx, wy, wz, grip: 0.070, 0.098, 0.133, 0.012, 0.011, 0.011, 0.047.
Quantized physics replay preserved the 10 s hold on 22 of 24 new sliding and hybrid trajectories that were replayed. Release-regrasp fractions are computed from the saved actions; their physics replay was already reported in the previous search and was not repeated.
Every strict sliding trajectory is inside the ±0.25 residual bound at every step, and its quantized replay kept the 10 s hold. The mean absolute gap on those trajectories stays below 0.11 on every channel. No new prototype is recommended for the sliding primitive. The two quantized failures are the hybrid trajectories that entered the open chase (`ball_train_0012` and `ball_train_0017`), where the L-infinity gap reaches 0.52. That is the same open-phase vy/vz chase gap reported for release-regrasp. The pooled channel means above are pulled up by those release sequences. No token was added and the residual bound was not changed.

## 8. Trajectories to retain as skill demonstrations

- `ball_train_0001` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0001__hybrid_slide_58.npz`. Mechanism labels ['reclose']. Duration 0.64 s, contact loss 0.06 s, 10 s True, 12 s True.
- `ball_train_0001` regrasp `release_short`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0001__regrasp_release_short.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.62 s, contact loss 0.48 s, 10 s True, 12 s True.
- `ball_train_0002` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0002__hybrid_slide_58.npz`. Mechanism labels ['reclose']. Duration 0.66 s, contact loss 0.06 s, 10 s True, 12 s True.
- `ball_train_0002` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0002__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.54 s, contact loss 0.4 s, 10 s True, 12 s True.
- `ball_train_0003` sliding `slide_50`: continuous-contact sliding with causal ablations. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0003__sliding_slide_50.npz`. Mechanism labels ['controlled_sliding', 'reclose']. Duration 0.68 s, contact loss 0.0 s, 10 s True, 12 s True.
- `ball_train_0003` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0003__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.56 s, contact loss 0.42 s, 10 s True, 12 s True.
- `ball_train_0004` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0004__hybrid_slide_58.npz`. Mechanism labels ['reclose']. Duration 0.68 s, contact loss 0.06 s, 10 s True, 12 s True.
- `ball_train_0004` regrasp `release_long`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0004__regrasp_release_long.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 1.88 s, contact loss 0.9 s, 10 s True, 12 s True.
- `ball_train_0005` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0005__hybrid_slide_58.npz`. Mechanism labels ['reclose']. Duration 0.7 s, contact loss 0.06 s, 10 s True, 12 s True.
- `ball_train_0005` regrasp `release_long`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0005__regrasp_release_long.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.56 s, contact loss 0.42 s, 10 s True, 12 s True.
- `ball_train_0006` hybrid `slide_58`: slide followed by a state-based release and regrasp. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0006__hybrid_slide_58.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose']. Duration 0.84 s, contact loss 0.1 s, 10 s True, 12 s True.
- `ball_train_0006` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0006__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.62 s, contact loss 0.5 s, 10 s True, 12 s True.
- `ball_train_0007` sliding `slide_45`: continuous-contact sliding with causal ablations. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0007__sliding_slide_45.npz`. Mechanism labels ['controlled_sliding', 'reclose']. Duration 0.88 s, contact loss 0.0 s, 10 s True, 12 s True.
- `ball_train_0007` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0007__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.62 s, contact loss 0.48 s, 10 s True, 12 s True.
- `ball_train_0008` hybrid `slide_58`: slide followed by a state-based release and regrasp. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0008__hybrid_slide_58.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose']. Duration 0.9 s, contact loss 0.24 s, 10 s True, 12 s True.
- `ball_train_0008` regrasp `chase_lost`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0008__regrasp_chase_lost.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.94 s, contact loss 0.72 s, 10 s True, 12 s True.
- `ball_train_0009` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0009__hybrid_slide_58.npz`. Mechanism labels ['reclose']. Duration 0.82 s, contact loss 0.06 s, 10 s True, 12 s True.
- `ball_train_0009` regrasp `release_short`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0009__regrasp_release_short.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.94 s, contact loss 0.5 s, 10 s True, 12 s True.
- `ball_train_0010` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0010__hybrid_slide_58.npz`. Mechanism labels []. Duration 1.0 s, contact loss 0.08 s, 10 s True, 12 s True.
- `ball_train_0010` regrasp `chase_lost`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0010__regrasp_chase_lost.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp']. Duration 0.78 s, contact loss 0.66 s, 10 s True, 12 s True.
- `ball_train_0011` hybrid `slide_58`: slide followed by a state-based release and regrasp. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0011__hybrid_slide_58.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose']. Duration 1.4 s, contact loss 0.2 s, 10 s True, 12 s True.
- `ball_train_0011` regrasp `release_long`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0011__regrasp_release_long.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.64 s, contact loss 0.5 s, 10 s True, 12 s True.
- `ball_train_0012` hybrid `slide_58`: slide followed by a state-based release and regrasp. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0012__hybrid_slide_58.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose']. Duration 0.94 s, contact loss 0.46 s, 10 s True, 12 s True.
- `ball_train_0012` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0012__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.9 s, contact loss 0.52 s, 10 s True, 12 s True.
- `ball_train_0013` sliding `slide_45`: continuous-contact sliding with causal ablations. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0013__sliding_slide_45.npz`. Mechanism labels ['controlled_sliding']. Duration 1.22 s, contact loss 0.0 s, 10 s True, 12 s True.
- `ball_train_0013` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0013__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.94 s, contact loss 0.64 s, 10 s True, 12 s True.
- `ball_train_0014` sliding `slide_58`: continuous-contact sliding with causal ablations. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0014__sliding_slide_58.npz`. Mechanism labels ['controlled_sliding']. Duration 1.4 s, contact loss 0.0 s, 10 s True, 12 s True.
- `ball_train_0014` regrasp `chase_lost`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0014__regrasp_chase_lost.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 1.16 s, contact loss 0.64 s, 10 s True, 12 s True.
- `ball_train_0015` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0015__hybrid_slide_58.npz`. Mechanism labels []. Duration 1.34 s, contact loss 0.04 s, 10 s True, 12 s True.
- `ball_train_0015` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0015__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.62 s, contact loss 0.48 s, 10 s True, 12 s True.
- `ball_train_0016` hybrid `slide_58`: short contact-gap slide; the release chase was not entered. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0016__hybrid_slide_58.npz`. Mechanism labels ['reclose']. Duration 0.82 s, contact loss 0.04 s, 10 s True, 12 s True.
- `ball_train_0016` regrasp `release_long`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0016__regrasp_release_long.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 2.82 s, contact loss 1.2 s, 10 s True, 12 s True.
- `ball_train_0017` hybrid `slide_58`: slide followed by a state-based release and regrasp. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0017__hybrid_slide_58.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'hybrid_slide_then_regrasp', 'reclose']. Duration 0.94 s, contact loss 0.44 s, 10 s True, 12 s True.
- `ball_train_0017` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0017__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 0.6 s, contact loss 0.46 s, 10 s True, 12 s True.
- `ball_train_0018` sliding `slide_45`: continuous-contact sliding with causal ablations. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0018__sliding_slide_45.npz`. Mechanism labels ['controlled_sliding']. Duration 1.32 s, contact loss 0.0 s, 10 s True, 12 s True.
- `ball_train_0018` regrasp `lift_soft`: release, chase, reclose, and nominal handoff. File `results/diagnostics/raw/sliding_vs_regrasp/trajectories/ball_train_0018__regrasp_lift_soft.npz`. Mechanism labels ['partial_or_complete_release', 'recontact_regrasp', 'reclose']. Duration 1.26 s, contact loss 0.92 s, 10 s True, 12 s True.
Retained verified demonstrations listed above: 36.

Skill patterns present in those files are temporal pieces, not one token each: unloaded translation while contact continues, reclose and lift, full release, hand chase, recontact, and the nominal handoff. A trajectory that uses more than one of these is stored as the sequence, not as a single skill id.

## 9. Replay

Logs: `results/diagnostics/raw/sliding_vs_regrasp/search_log.jsonl`.
Summary: `results/diagnostics/raw/sliding_vs_regrasp/summary.json`.
Trajectories: `results/diagnostics/raw/sliding_vs_regrasp/trajectories/`.

Resume, skipping initial conditions already marked complete:

```
python -u training/sliding_vs_regrasp.py
```

MATCHED RECOVERY COMPARISON COMPLETE

POLICY TRAINING HAS NOT BEEN STARTED
