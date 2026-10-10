# Contacted velocity-matching recovery

Recovery starts at the velocity injection, while both pads are still in contact. The closed hold is not used as the trigger. The reference is the last bilateral frame of the dz −13.5 natural-loss trajectory, the same state as the perturbation audit. Object angular velocity is in the body frame. The physics step is 2 ms, and the first 100 ms are logged at that step.

The search is a finite family of legal recovery7d_airborne_v3 commands. Hand vx and vy stay within ±1 m/s, hand vz stays within −1 m/s to +3.8 m/s, wrist rate is 0, and the grip torque stays at the secure value −18 N·m. Tracking gains are unchanged. Piecewise-constant commands hold the initial match. Feedback commands are recomputed every 20 ms, and two members recompute every physics step for the first 40 ms. Gains of 0.5, 1, 2, and 4 times the measured object or relative velocity are included, and one member commands the v3 bound on the perturbed axis. After the match window the hand is commanded to the object's current hand-frame velocity. A miss by this family is not a proof that no legal v3 sequence can keep the grasp.

## Three timings

A is hold until the persistent loss, then the verified v3 intercept. B is that intercept started at the injection, with the grip opened for the chase. C is the velocity-matching family started at the injection, with the grip kept closed. A success is either bilateral contact kept for 1 s with relative speed under 0.08 m/s and clearance within 1 mm, or a persistent loss whose clearance is still within 1 mm and which the verified intercept then holds for 1 s.

| perturbation | hold loss | hold 1 s | immediate intercept 1 s | match contact 1 s | match then 1 s recatch |
|---|---:|---|---|---|---|
| reference | 0.02 s | yes | yes | no | no |
| vx_-0.012 | 0.02 s | yes | yes | no | no |
| vy_-0.123 | 0.02 s | no | no | no | no |
| vy_+0.207 | 0.02 s | no | no | no | no |
| vz_+0.187 | 0.02 s | no | no | no | no |
| vz_+0.694 | 0.02 s | no | no | no | no |

## Each perturbation

### reference

Injected hand-frame twist v = [0.0009, 0.0020, 0.0308] m/s, w = [-0.1256, 0.0321, 0.0512] rad/s. qpos change 0.0, contact counts unchanged: True [3, 4].

Hold reaches a persistent loss at 0.02 s. Loss clearance -0.014 mm, relative velocity [0.0019, -0.0001, 0.0484] m/s, contacts [1, 0]. The verified intercept after that loss retains both pads for 1 s: yes.

At 20 ms under the hold, relative velocity is [-0.0056, 0.0060, 0.2081] m/s, achieved hand velocity [0.0001, 0.0034, 0.0116] m/s, command [0.0000, 0.0000, 0.0000] m/s, clearance 0.058 mm, contacts [0, 0], normal force 0.0 N, torque-limit ratio 0.3.

The immediate intercept does hold both pads for 1 s. At 20 ms its relative velocity is [0.0154, -0.0350, -0.2686] m/s, achieved hand velocity [0.0115, -0.1487, 0.9359] m/s, command [-0.0012, -0.0219, 3.8000] m/s, clearance -0.007 mm, contacts [2, 0], torque-limit ratio 1.0.

Best contacted match: full_obj × 1.0, command update 0.02 s, match window 0.1 s. Minimum |relative component 0| while contact remains is 0.0004 m/s at 0.018 s. At 20 ms, relative velocity [-0.0002, 0.0085, 0.2071] m/s, achieved hand velocity [0.0004, 0.0021, 0.0242] m/s, command [0.0009, 0.0045, 0.0484] m/s, clearance 0.032 mm, contacts [0, 0], normal force 0.0 N. Arm torque reached 95% of a limit in the first 20 ms: no (peak ratio 0.283). Peak ratio over the rollout 0.283.

Contact retained for 1 s with relative speed under 0.08 m/s: no. Any family member did so: no. Persistent-loss time 0.028. Loss clearance -0.044 mm, loss relative velocity [-0.0003, -0.0528, 0.3346] m/s, contacts [0, 4]. Verified intercept from that loss state retains both pads for 1 s: no. Any family member reaches that 1 s recatch: no.

First 100 ms of the best match, every 10 ms:

| t (ms) | v_rel (m/s) | v_hand (m/s) | command (m/s) | clear (mm) | contacts | Fn (N) | sat ratio |
|---:|---|---|---|---:|---|---:|---:|
| 0 | [0.0009, 0.0020, 0.0308] | [-0.0001, 0.0024, 0.0176] | [0.0009, 0.0045, 0.0484] | -0.043 | [3, 4] | 12.952 | 0.0 |
| 10 | [-0.0001, -0.0422, 0.0923] | [0.0003, 0.0037, 0.0219] | [0.0009, 0.0045, 0.0484] | -0.297 | [0, 4] | 6.393 | 0.278 |
| 20 | [-0.0002, 0.0085, 0.2071] | [0.0004, 0.0021, 0.0242] | [0.0009, 0.0045, 0.0484] | 0.032 | [0, 0] | 0.0 | 0.283 |
| 30 | [-0.0003, 0.0310, 0.3700] | [0.0009, -0.0035, 0.0517] | [0.0002, 0.0106, 0.2313] | 0.225 | [0, 0] | 0.0 | 0.207 |
| 40 | [-0.0006, 0.0321, 0.4457] | [0.0011, -0.0039, 0.0741] | [0.0002, 0.0106, 0.2313] | 2.52 | [0, 0] | 0.0 | 0.228 |
| 60 | [-0.0020, 0.0394, 0.5345] | [0.0022, -0.0076, 0.1814] | [0.0005, 0.0282, 0.5198] | 12.384 | [0, 0] | 0.0 | 0.203 |

Family size 1. Members whose arm torque hit 95% of a limit in the first 20 ms: 0.

### vx_-0.012

Injected hand-frame twist v = [-0.0114, 0.0020, 0.0308] m/s, w = [-0.1256, 0.0321, 0.0512] rad/s. qpos change 0.0, contact counts unchanged: True [3, 4].

Hold reaches a persistent loss at 0.02 s. Loss clearance -0.029 mm, relative velocity [-0.0090, 0.0033, 0.0398] m/s, contacts [4, 0]. The verified intercept after that loss retains both pads for 1 s: yes.

At 20 ms under the hold, relative velocity is [-0.0051, 0.0116, 0.2061] m/s, achieved hand velocity [-0.0004, 0.0030, 0.0119] m/s, command [0.0000, 0.0000, 0.0000] m/s, clearance 0.026 mm, contacts [0, 0], normal force 0.0 N, torque-limit ratio 0.3.

The immediate intercept does hold both pads for 1 s. At 20 ms its relative velocity is [0.0141, -0.0350, -0.2684] m/s, achieved hand velocity [0.0075, -0.1488, 0.9358] m/s, command [-0.0135, -0.0219, 3.8000] m/s, clearance -0.013 mm, contacts [1, 0], torque-limit ratio 1.0.

Best contacted match: oppose_rel × 4.0, command update 0.02 s, match window 0.1 s. Minimum |relative component 0| while contact remains is 0.0003 m/s at 0.014 s. At 20 ms, relative velocity [0.0041, 0.0137, 0.2058] m/s, achieved hand velocity [-0.0145, 0.0026, 0.0117] m/s, command [-0.0455, 0.0000, 0.0000] m/s, clearance -0.0 mm, contacts [0, 0], normal force 0.0 N. Arm torque reached 95% of a limit in the first 20 ms: no (peak ratio 0.3). Peak ratio over the rollout 0.3.

Contact retained for 1 s with relative speed under 0.08 m/s: no. Any family member did so: no. Persistent-loss time 0.024. Loss clearance -0.059 mm, loss relative velocity [0.0016, 0.0075, 0.2793] m/s, contacts [2, 0]. Verified intercept from that loss state retains both pads for 1 s: no. Any family member reaches that 1 s recatch: no.

First 100 ms of the best match, every 10 ms:

| t (ms) | v_rel (m/s) | v_hand (m/s) | command (m/s) | clear (mm) | contacts | Fn (N) | sat ratio |
|---:|---|---|---|---:|---|---:|---:|
| 0 | [-0.0114, 0.0020, 0.0308] | [-0.0001, 0.0024, 0.0176] | [-0.0455, 0.0000, 0.0000] | -0.043 | [3, 4] | 12.947 | 0.0 |
| 10 | [-0.0027, 0.0310, 0.1116] | [-0.0085, 0.0012, 0.0154] | [-0.0455, 0.0000, 0.0000] | -0.389 | [4, 4] | 16.216 | 0.299 |
| 20 | [0.0041, 0.0137, 0.2058] | [-0.0145, 0.0026, 0.0117] | [-0.0455, 0.0000, 0.0000] | -0.0 | [0, 0] | 0.0 | 0.299 |
| 30 | [-0.0020, 0.0271, 0.3794] | [-0.0078, 0.0023, 0.0077] | [0.0165, 0.0000, 0.0000] | 0.201 | [0, 0] | 0.0 | 0.293 |
| 40 | [-0.0071, 0.0272, 0.4789] | [-0.0027, 0.0023, 0.0063] | [0.0165, 0.0000, 0.0000] | 3.102 | [0, 0] | 0.0 | 0.294 |
| 60 | [0.0011, 0.0278, 0.6774] | [-0.0108, 0.0017, 0.0041] | [-0.0284, 0.0000, 0.0000] | 15.044 | [0, 0] | 0.0 | 0.292 |

Family size 16. Members whose arm torque hit 95% of a limit in the first 20 ms: 0. The v3-bound member commands [-1.0000, 0.0000, 0.0000] m/s, achieves hand velocity [-0.3087, -0.0029, 0.0047] m/s at 20 ms, and its torque-limit ratio in that window is 0.587.

### vy_-0.123

Injected hand-frame twist v = [0.0009, -0.1212, 0.0308] m/s, w = [-0.1256, 0.0321, 0.0512] rad/s. qpos change 0.0, contact counts unchanged: True [3, 4].

Hold reaches a persistent loss at 0.02 s. Loss clearance -0.156 mm, relative velocity [0.0035, -0.0982, 0.0427] m/s, contacts [0, 4]. The verified intercept after that loss retains both pads for 1 s: no.

At 20 ms under the hold, relative velocity is [0.0030, -0.0084, 0.2811] m/s, achieved hand velocity [-0.0003, 0.0017, 0.0094] m/s, command [0.0000, 0.0000, 0.0000] m/s, clearance -0.037 mm, contacts [2, 4], normal force 16.082 N, torque-limit ratio 0.301.

The immediate intercept does not hold both pads for 1 s. At 20 ms its relative velocity is [0.0102, -0.0253, -0.1677] m/s, achieved hand velocity [0.0112, -0.1598, 0.9359] m/s, command [-0.0012, -0.1451, 3.8000] m/s, clearance -0.551 mm, contacts [4, 0], torque-limit ratio 1.0.

Best contacted match: bound × 1.0, command update 0.02 s, match window 0.1 s. Minimum |relative component 1| while contact remains is 0.0007 m/s at 0.01 s. At 20 ms, relative velocity [0.0016, -0.0749, 0.2065] m/s, achieved hand velocity [-0.0038, -0.1749, 0.0348] m/s, command [0.0000, -1.0000, 0.0000] m/s, clearance -0.103 mm, contacts [2, 4], normal force 15.131 N. Arm torque reached 95% of a limit in the first 20 ms: no (peak ratio 0.549). Peak ratio over the rollout 0.549.

Contact retained for 1 s with relative speed under 0.08 m/s: no. Any family member did so: no. Persistent-loss time 0.022. Loss clearance 0.056 mm, loss relative velocity [0.0022, -0.0455, 0.2399] m/s, contacts [0, 0]. Verified intercept from that loss state retains both pads for 1 s: no. Any family member reaches that 1 s recatch: no.

First 100 ms of the best match, every 10 ms:

| t (ms) | v_rel (m/s) | v_hand (m/s) | command (m/s) | clear (mm) | contacts | Fn (N) | sat ratio |
|---:|---|---|---|---:|---|---:|---:|
| 0 | [0.0009, -0.1212, 0.0308] | [-0.0001, 0.0024, 0.0176] | [0.0000, -1.0000, 0.0000] | -0.043 | [3, 4] | 13.378 | 0.0 |
| 10 | [0.0008, 0.0007, 0.0785] | [-0.0023, -0.1061, 0.0382] | [0.0000, -1.0000, 0.0000] | -0.222 | [4, 4] | 13.598 | 0.548 |
| 20 | [0.0016, -0.0749, 0.2065] | [-0.0038, -0.1749, 0.0348] | [0.0000, -1.0000, 0.0000] | -0.103 | [2, 4] | 15.131 | 0.543 |
| 30 | [0.0031, -0.1441, 0.3630] | [-0.0049, -0.2281, 0.0241] | [0.0000, -1.0000, 0.0000] | 0.044 | [0, 0] | 0.0 | 0.533 |
| 40 | [0.0035, -0.0948, 0.4728] | [-0.0053, -0.2733, 0.0145] | [0.0000, -1.0000, 0.0000] | 2.644 | [0, 0] | 0.0 | 0.51 |
| 60 | [0.0039, -0.0148, 0.6953] | [-0.0058, -0.3423, -0.0081] | [0.0000, -1.0000, 0.0000] | 14.711 | [0, 0] | 0.0 | 0.471 |

Family size 16. Members whose arm torque hit 95% of a limit in the first 20 ms: 0. The v3-bound member commands [0.0000, -1.0000, 0.0000] m/s, achieves hand velocity [-0.0038, -0.1749, 0.0348] m/s at 20 ms, and its torque-limit ratio in that window is 0.549.

### vy_+0.207

Injected hand-frame twist v = [0.0009, 0.2089, 0.0308] m/s, w = [-0.1256, 0.0321, 0.0512] rad/s. qpos change 0.0, contact counts unchanged: True [3, 4].

Hold reaches a persistent loss at 0.02 s. Loss clearance -0.273 mm, relative velocity [-0.0028, 0.1723, 0.0506] m/s, contacts [4, 0]. The verified intercept after that loss retains both pads for 1 s: no.

At 20 ms under the hold, relative velocity is [-0.0008, -0.0492, 0.3169] m/s, achieved hand velocity [-0.0000, 0.0102, 0.0073] m/s, command [0.0000, 0.0000, 0.0000] m/s, clearance 0.013 mm, contacts [0, 0], normal force 0.0 N, torque-limit ratio 0.299.

The immediate intercept does not hold both pads for 1 s. At 20 ms its relative velocity is [0.0164, 0.0536, -0.2096] m/s, achieved hand velocity [0.0116, -0.1318, 0.9284] m/s, command [-0.0012, 0.1850, 3.7998] m/s, clearance -0.649 mm, contacts [4, 0], torque-limit ratio 1.0.

Best contacted match: oppose_rel × 2.0, command update 0.002 s, match window 0.04 s. Minimum |relative component 1| while contact remains is 0.0016 m/s at 0.014 s. At 20 ms, relative velocity [-0.0000, 0.0176, 0.3202] m/s, achieved hand velocity [0.0001, 0.0180, 0.0087] m/s, command [0.0000, 0.0349, 0.0000] m/s, clearance 0.238 mm, contacts [0, 0], normal force 0.0 N. Arm torque reached 95% of a limit in the first 20 ms: no (peak ratio 0.507). Peak ratio over the rollout 0.507.

Contact retained for 1 s with relative speed under 0.08 m/s: no. Any family member did so: no. Persistent-loss time 0.016. Loss clearance 0.047 mm, loss relative velocity [-0.0000, 0.0173, 0.2810] m/s, contacts [0, 0]. Verified intercept from that loss state retains both pads for 1 s: no. Any family member reaches that 1 s recatch: no.

First 100 ms of the best match, every 10 ms:

| t (ms) | v_rel (m/s) | v_hand (m/s) | command (m/s) | clear (mm) | contacts | Fn (N) | sat ratio |
|---:|---|---|---|---:|---|---:|---:|
| 0 | [0.0009, 0.2089, 0.0308] | [-0.0001, 0.0024, 0.0176] | [0.0000, 0.4179, 0.0000] | -0.043 | [3, 4] | 13.847 | 0.0 |
| 10 | [-0.0010, -0.0421, 0.1934] | [0.0004, 0.0278, 0.0066] | [0.0000, -0.1400, 0.0000] | -0.539 | [4, 4] | 16.893 | 0.322 |
| 20 | [-0.0000, 0.0176, 0.3202] | [0.0001, 0.0180, 0.0087] | [0.0000, 0.0349, 0.0000] | 0.238 | [0, 0] | 0.0 | 0.285 |
| 30 | [-0.0001, 0.0178, 0.4184] | [0.0002, 0.0176, 0.0086] | [0.0000, 0.0356, 0.0000] | 1.22 | [0, 0] | 0.0 | 0.286 |
| 40 | [-0.0001, 0.0176, 0.5168] | [0.0002, 0.0175, 0.0083] | [0.0000, 0.0354, 0.0000] | 6.077 | [0, 0] | 0.0 | 0.286 |

Family size 16. Members whose arm torque hit 95% of a limit in the first 20 ms: 0. The v3-bound member commands [0.0000, 1.0000, 0.0000] m/s, achieves hand velocity [0.0037, 0.1875, -0.0127] m/s at 20 ms, and its torque-limit ratio in that window is 0.922.

### vz_+0.187

Injected hand-frame twist v = [0.0009, 0.0020, 0.2177] m/s, w = [-0.1256, 0.0321, 0.0512] rad/s. qpos change 0.0, contact counts unchanged: True [3, 4].

Hold reaches a persistent loss at 0.02 s. Loss clearance 0.202 mm, relative velocity [0.0010, 0.0029, 0.2190] m/s, contacts [0, 0]. The verified intercept after that loss retains both pads for 1 s: no.

At 20 ms under the hold, relative velocity is [0.0020, -0.0032, 0.4503] m/s, achieved hand velocity [-0.0002, 0.0038, 0.0099] m/s, command [0.0000, 0.0000, 0.0000] m/s, clearance 0.838 mm, contacts [0, 0], normal force 0.0 N, torque-limit ratio 0.3.

The immediate intercept does not hold both pads for 1 s. At 20 ms its relative velocity is [0.0132, -0.0124, -0.1827] m/s, achieved hand velocity [0.0114, -0.1488, 0.9391] m/s, command [-0.0012, -0.0222, 3.8000] m/s, clearance -0.64 mm, contacts [4, 0], torque-limit ratio 1.0.

Best contacted match: match_obj × 4.0, command update 0.02 s, match window 0.1 s. Minimum |relative component 2| while contact remains is 0.0018 m/s at 0.09 s. At 20 ms, relative velocity [0.0032, -0.0969, 0.2601] m/s, achieved hand velocity [0.0036, -0.0291, 0.2575] m/s, command [0.0000, 0.0000, 0.9414] m/s, clearance -0.066 mm, contacts [4, 0], normal force 3.687 N. Arm torque reached 95% of a limit in the first 20 ms: no (peak ratio 0.344). Peak ratio over the rollout 1.0.

Contact retained for 1 s with relative speed under 0.08 m/s: no. Any family member did so: no. Persistent-loss time 0.092. Loss clearance -0.085 mm, loss relative velocity [0.0116, 0.1259, -0.0076] m/s, contacts [0, 4]. Verified intercept from that loss state retains both pads for 1 s: no. Any family member reaches that 1 s recatch: no.

First 100 ms of the best match, every 10 ms:

| t (ms) | v_rel (m/s) | v_hand (m/s) | command (m/s) | clear (mm) | contacts | Fn (N) | sat ratio |
|---:|---|---|---|---:|---|---:|---:|
| 0 | [0.0009, 0.0020, 0.2177] | [-0.0001, 0.0024, 0.0176] | [0.0000, 0.0000, 0.9414] | -0.043 | [3, 4] | 12.98 | 0.0 |
| 10 | [0.0028, 0.0008, 0.1619] | [0.0030, -0.0274, 0.1601] | [0.0000, 0.0000, 0.9414] | -0.511 | [2, 4] | 15.979 | 0.097 |
| 20 | [0.0032, -0.0969, 0.2601] | [0.0036, -0.0291, 0.2575] | [0.0000, 0.0000, 0.9414] | -0.066 | [4, 0] | 3.687 | 0.085 |
| 30 | [0.0014, -0.0165, 0.2827] | [0.0069, -0.0659, 0.5105] | [0.0000, 0.0000, 2.0703] | 0.289 | [0, 0] | 0.0 | 0.372 |
| 40 | [0.0010, -0.0089, 0.1970] | [0.0062, -0.0626, 0.6951] | [0.0000, 0.0000, 2.0703] | 1.976 | [0, 0] | 0.0 | 0.199 |
| 60 | [0.0033, 0.0451, -0.1680] | [0.0052, -0.0841, 1.3479] | [0.0000, 0.0000, 3.5685] | 0.164 | [5, 0] | 7.571 | 0.407 |
| 80 | [0.0144, 0.0706, 0.0171] | [-0.0034, -0.0584, 1.7888] | [0.0000, 0.0000, 3.8000] | -0.381 | [1, 10] | 0.831 | 0.724 |
| 100 | [0.0127, 0.1596, -0.0055] | [-0.0101, -0.0760, 2.0811] | [0.0000, 0.0000, 3.8000] | 0.296 | [0, 0] | 0.0 | 0.91 |

Family size 16. Members whose arm torque hit 95% of a limit in the first 20 ms: 1. The v3-bound member commands [0.0000, 0.0000, 3.8000] m/s, achieves hand velocity [0.0117, -0.1428, 0.9298] m/s at 20 ms, and its torque-limit ratio in that window is 1.0.

### vz_+0.694

Injected hand-frame twist v = [0.0009, 0.0020, 0.7245] m/s, w = [-0.1256, 0.0321, 0.0512] rad/s. qpos change 0.0, contact counts unchanged: True [3, 4].

Hold reaches a persistent loss at 0.02 s. Loss clearance 0.890 mm, relative velocity [0.0011, 0.0030, 0.7241] m/s, contacts [0, 0]. The verified intercept after that loss retains both pads for 1 s: no.

At 20 ms under the hold, relative velocity is [0.0011, 0.0020, 0.9069] m/s, achieved hand velocity [-0.0001, 0.0035, 0.0115] m/s, command [0.0000, 0.0000, 0.0000] m/s, clearance 9.65 mm, contacts [0, 0], normal force 0.0 N, torque-limit ratio 0.3.

The immediate intercept does not hold both pads for 1 s. At 20 ms its relative velocity is [-0.0143, 0.1701, -0.0174] m/s, achieved hand velocity [0.0128, -0.1575, 0.9544] m/s, command [-0.0012, -0.0228, 3.8000] m/s, clearance 2.765 mm, contacts [0, 0], torque-limit ratio 1.0.

Best contacted match: match_obj × 4.0, command update 0.02 s, match window 0.1 s. Minimum |relative component 2| while contact remains is 0.7245 m/s at 0.0 s. At 20 ms, relative velocity [-0.0122, -0.0025, 0.2034] m/s, achieved hand velocity [0.0112, -0.1073, 0.8011] m/s, command [0.0000, 0.0000, 2.9686] m/s, clearance 0.528 mm, contacts [0, 0], normal force 0.0 N. Arm torque reached 95% of a limit in the first 20 ms: yes (peak ratio 1.0). Peak ratio over the rollout 1.0.

Contact retained for 1 s with relative speed under 0.08 m/s: no. Any family member did so: no. Persistent-loss time 0.002. Loss clearance 0.716 mm, loss relative velocity [-0.0020, 0.0062, 0.6270] m/s, contacts [0, 0]. Verified intercept from that loss state retains both pads for 1 s: no. Any family member reaches that 1 s recatch: no.

First 100 ms of the best match, every 10 ms:

| t (ms) | v_rel (m/s) | v_hand (m/s) | command (m/s) | clear (mm) | contacts | Fn (N) | sat ratio |
|---:|---|---|---|---:|---|---:|---:|
| 0 | [0.0009, 0.0020, 0.7245] | [-0.0001, 0.0024, 0.0176] | [0.0000, 0.0000, 2.9686] | -0.043 | [3, 4] | 14.058 | 0.0 |
| 10 | [-0.0099, 0.0806, 0.3492] | [0.0102, -0.0875, 0.4723] | [0.0000, 0.0000, 2.9686] | 0.88 | [0, 0] | 0.0 | 0.862 |
| 20 | [-0.0122, -0.0025, 0.2034] | [0.0112, -0.1073, 0.8011] | [0.0000, 0.0000, 2.9686] | 0.528 | [0, 0] | 0.0 | 0.585 |
| 30 | [-0.0146, 0.0303, -0.0335] | [0.0115, -0.1287, 1.1931] | [0.0000, 0.0000, 3.8000] | 0.741 | [4, 0] | 5.551 | 0.706 |
| 40 | [0.0097, 0.0397, -0.0256] | [0.0044, -0.0998, 1.4599] | [0.0000, 0.0000, 3.8000] | 0.372 | [0, 0] | 0.0 | 0.453 |

Family size 16. Members whose arm torque hit 95% of a limit in the first 20 ms: 4. The v3-bound member commands [0.0000, 0.0000, 3.8000] m/s, achieves hand velocity [0.0124, -0.1394, 0.9392] m/s at 20 ms, and its torque-limit ratio in that window is 1.0.

## What the first 100 ms shows

The injection keeps the reference spin. Free-joint angular velocity is stored in the body frame, and the written spin matches the reference, w = [−0.126, 0.032, 0.051] rad/s. Pose, finger joints, and contact counts are unchanged at the write.

On the reference and on vx = −0.012 m/s, both the hold-then-intercept and the immediate intercept retain both pads for 1 s. The velocity-matching family does not. Moving the hand during this slip changes the release: the loss separating speed rises from 0.05 m/s to about 0.3 m/s, and the verified intercept then misses. On a twist that is already recatchable, this matcher is the worse of the three timings.

On the harmful injections the hold-then-intercept and the immediate intercept both miss, as in the perturbation audit. The matcher changes the contacted interval, and it still does not finish in a 1 s grasp.

- vy = −0.123 m/s. At 2 ms the hold has already lost the left pad and vy is −0.098 m/s. The v3-bound lateral command (−1 m/s) brings relative vy to 0.0007 m/s at 10 ms with both pads still in contact, and the achieved hand vy is −0.11 m/s. Torque in that window stays at 55% of a limit. Contact is then lost at 22 ms. The loss has clearance 0.056 mm, relative velocity [0.002, −0.046, 0.240] m/s, and spin about −8.8 rad/s. The verified intercept does not hold it.
- vy = +0.207 m/s. A 2 ms feedback command of twice the relative vy brings |vy| to 0.0016 m/s at 14 ms, again with both pads in contact at 10 ms. Achieved hand vy at 10 ms is only 0.028 m/s; the contact itself removes most of the lateral speed. The loss at 16 ms has relative vy 0.017 m/s and separating vz 0.281 m/s. The verified intercept misses. The bound command reaches an achieved hand vy of 0.19 m/s at 20 ms with torque ratio 0.92, after the pads have already opened.
- vz = +0.187 m/s. The hold loses both pads at 2 ms (clearance +0.20 mm, vz still 0.22 m/s). Commanding four times the object's hand-frame vz keeps a pad until 92 ms and brings relative vz to 0.0018 m/s at 90 ms, while clearance is still negative. The hand's achieved vz at 20 ms is 0.26 m/s against a 0.94 m/s command, without torque saturation in that window. The loss at 92 ms is not the verified release: lateral error 12.7 mm, relative velocity [0.012, 0.126, −0.008] m/s, spin about −11.8 rad/s. The verified intercept misses. The bound command of +3.8 m/s does saturate (ratio 1.0) and achieves 0.93 m/s of hand vz by 20 ms.
- vz = +0.694 m/s. Both pads are gone at 2 ms under every member, including the +3.8 m/s bound. The best loss at that first step still has vz 0.63 m/s and clearance 0.72 mm. By 20 ms the hand has reached 0.80 m/s on a 2.97 m/s command, with the torque ratio at 1.0. The reduction of relative vz happens after the pads have separated, and the verified intercept misses.

No member keeps bilateral contact for 1 s with relative speed under 0.08 m/s. No member leaves a loss that the verified intercept holds for 1 s. The family that was run is 16 commands on each harmful twist: scales 0.5, 1, 2, and 4, object-velocity matching and relative-velocity opposition, a held command and 20 ms feedback, 2 ms feedback for two of the scales, one full linear match, and the v3 bound on the perturbed axis. Grip stays at −18 N·m and wrist rate stays 0. A miss by these commands is not a proof that no legal v3 sequence can save the twist.

EARLY RECOVERY IMPROVES THE LOSS STATE BUT DOES NOT FULLY RECOVER
