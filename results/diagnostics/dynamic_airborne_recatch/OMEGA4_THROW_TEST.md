# Omega=4 diagnostic throw / recapture test

Cylinder mass was **intentionally held fixed** at the current construction value.
Mass robustness is out of scope. No mass sweep, friction change, or disturbance-distribution change.

Mass = **0.2000 kg** (`CYL_MASS=0.2`). Friction pair μ = 1.0.

`RECOVERY4D_W_HY_MAX` was **not** changed. `omega_y_cmd = +4.0` is a diagnostic override via `tick_omega_override` (`w_world = r_des @ [0, 4, 0]`). Legal map still clips `v_x`, `v_z`, and gripper tau.

## 1. Rotation gate (secure swing, tau=-18)

| | omega_cmd | cruise ω_axis median | max | supported peak v_obj,z | angle of peak | ori_err max (°) | pos drift max (mm) | peak |q̇_5| | peak τ util j5 | sat steps |
|---|---|---|---|---|---|---|---|---|---|---|
| omega=3 | 3.0 | 2.970 | 2.999 | 0.2999 | 79.2 | 11.83 | 24.6 | 2.807 | 0.641 | 0 |
| omega=4 | 4.0 | 3.930 | 3.971 | 0.4375 | 79.5 | 15.80 | 33.2 | 3.713 | 0.836 | 0 |

omega=3 q̇ peak (j1–j7) = [0.346, 0.038, 0.574, 0.918, 2.807, 0.89, 0.448]
omega=4 q̇ peak (j1–j7) = [0.428, 0.038, 0.729, 1.23, 3.713, 1.159, 0.5]
omega=3 τ util peak = [0.028, 0.268, 0.041, 0.257, 0.641, 0.218, 0.044]
omega=4 τ util peak = [0.044, 0.269, 0.057, 0.259, 0.836, 0.238, 0.052]

**Gate passed:** cruise hand rate 2.97 → **3.93 rad/s**. Not class C. No torque saturation. Wrist joint5 carries the extra rate (`|q̇_5|` 2.81 → 3.71). Tracking lag grows (SO(3) 12°→16°, position 25→33 mm) but **actual ω does increase**.

omega=4 peak-supported snapshot (~79.5°):
- r_h ≈ [−0.78, −0.14, 102.8] mm
- topology nL/nR = 4/4, Fn ≈ 9.00 / 9.10, ρ ≈ 0.42
- v_rel ≈ 0.283, carry_err ≈ 0.126 (object still follows enough for v_z to scale)

## 2. Matched comparison (same parent IC)

Kinematic predictions (not required matches): ω=3 → ~0.30 m/s; ω=4 → ~0.40 m/s with r_eff≈0.099 m.

Matched OPEN command for omega=3 uses the same phase-A command angle as omega=4 recatch (~76.9°).

| | actual ω cruise | supported peak v_obj,z | peak angle | FIRST_BOTH_OFF v_obj,z | both-off angle | rise after both-off |
|---|---|---|---|---|---|---|
| omega=3 | 2.970 | 0.300 | 79.2° | 0.233 | 78.5° | 2.5 mm |
| omega=4 phase A (recatch) | 3.930 | 0.438 | 79.5° | **0.360** | 79.0° | **6.2 mm** |
| omega=4 phase B | 3.930 | 0.438 | 79.5° | 0.358 | 69.5° | 6.2 mm |

Supported peak at ω=4 (0.438 m/s) is close to the 0.40 m/s kinematic scale. FIRST_BOTH_OFF is lower (0.36) because opening removes contact before the supported peak is frozen into free flight, but it is still **+0.13 m/s vs matched ω=3**.

## 3. OPEN phases (rotation continues; at most two)

OPEN while `omega_y_cmd` stays +4. Physical release = FIRST_BOTH_OFF. Latency ≈ **12 ms** (≈ 2.7° at 4 rad/s). Not a 90° ritual. Supported v_z peaked at **79.5°**, so OPEN was commanded ~3° earlier.

### phase 1 (used for recatch): OPEN_COMMAND at 76.9°

- OPEN_COMMAND actual angle 76.3°; FIRST_BOTH_OFF 79.0° at t=5.574 s
- open latency 12 ms
- actual ω_axis at both-off **3.984 rad/s**
- v_obj = [−0.020, −0.021, **+0.360**] m/s
- w_obj ≈ [3.99, 0.11, −0.11]
- r_h ≈ [−1.2, −0.14, 103.0] mm
- v_rel ≈ 0.207, v_rel_h ≈ [0.206, 0.002, 0.023]
- aperture 0.0184 m
- last-contact finger: **right** (t=5.572: nL=0, nR=2)
- height while still supported at v_z peak: obj_z = 0.656 m at 79.5°
- **after** FIRST_BOTH_OFF: z 0.6544 → 0.6607, rise **6.2 mm**, time to apex **38 ms**

### phase 2: OPEN_COMMAND at 67.5°

- FIRST_BOTH_OFF 69.5°, v_obj,z = +0.358 m/s, ω_axis 3.970
- last topology none; last-contact not separately marked
- post-release rise 6.2 mm, dt_apex 38 ms

Contact-free duration before GT close (phase A): FIRST_BOTH_OFF 5.574 → CLOSE 5.612 = **38 ms** of free flight, then fingers still closing another 26 ms before recontact.

## 4. Classification

**Class A: EFFECTIVE AUTHORITY INCREASE.**

- hand omega 2.97 → 3.93
- supported v_obj,z 0.30 → 0.44
- FIRST_BOTH_OFF v_obj,z 0.23 → 0.36
- ballistic rise 2.5 mm → 6.2 mm, time-to-apex 38 ms vs a shorter ω=3 window

Not class B (cylinder still follows). Not class C (actual rate tracks the override).

This is **simulation diagnostic authority**, not a hardware-safe bound change. `RECOVERY4D_W_HY_MAX` remains 3.0.

## 5. Recapture

**PRIVILEGED** GT close on phase A. After FIRST_BOTH_OFF, `ω_cmd=0` (`r_des` frozen). CLOSE at GT apex (`vz` crossing ≤0). Wrist not returned to nominal. Mass 0.20 kg.

| event | t (s) | note |
|---|---|---|
| OPEN_COMMAND | 5.562 | ~76.3°, `ctrl[7]=+2`, rotation continues |
| FIRST_BOTH_OFF | 5.574 | 79.0°, v_obj,z=+0.360, ω_hand≈3.98, aperture 0.0184 |
| BALLISTIC_APEX = CLOSE_START | 5.612 | rise 6.2 mm, dt_apex=38 ms, vz≈0, aperture still 0.028 |
| FIRST_RECONTACT = FIRST_BILATERAL | 5.638 | 26 ms after close command; aperture 0.011 |
| last bilateral sample | 5.652 | then nL=nR=0 |
| HOLD_START = HOLD_LOST | 5.690 | software 50 ms timer; contact already gone |

Recapture **failed**. No ≥2 s bilateral hold. Do not increase omega.

### Why, from `raw/omega4_recatch.npz`

Not “too fast to meet the hand”: it **does** recontact. Not a floor miss. Failure is **unstable recontact after the object leaves the squeeze-axis corridor**.

1. **Corridor / relative pose, not a pure vertical miss.** At FIRST_BOTH_OFF, `r_h.x ≈ −1.2 mm`. At apex/CLOSE, `r_h.x ≈ −10 mm`. At first recontact, `r_h.x ≈ −18.6 mm`, `r_h.z ≈ 110 mm`. During 38 ms of flight the COM walks along the squeeze axis. At recontact `obj_z=0.657` vs `hand_z=0.675` (~18 mm below the palm, still along the fingers, not fallen through).

2. **Aperture vs closing time.** At CLOSE_START aperture is still **0.028 m**. Fingers reach ~0.011 m only when contact returns. Closing is not instantaneous.

3. **Unstable recontact / relative velocity.** True bilateral window is **~14 ms** (5.638–5.652). In that window `v_obj,z` is already **−0.27 to −0.47 m/s** and `|v_rel|` is **0.27→0.53**. Contact then vanishes while fingers keep closing. Clip, not capture.

4. **Hand still rotating after park.** `ω_cmd=0` after both-off, but SO(3) lag at release was ~15°. Actual hand angle continues **79° → ~90°** during flight, so the geometric corridor moves.

5. **Not claimed:** miss with no finger touch; object too fast to recontact; orientation so wrong that contact never occurs.

FIRST_RECONTACT is insufficient. Shared HOLD_START/HOLD_LOST timestamps are not a 2 s hold.

Did not increase omega again. No close-timing search, no ω=5, no CENTER-6 transfer.

## Viewer

```text
python training/demo_dynamic_recatch.py --mode throw_omega4
python training/demo_dynamic_recatch.py --mode recatch_omega4
```

SPACE pause, R restart, `[` `]` speed. No MP4.

Headless used for this report: `python training/demo_dynamic_recatch.py --mode omega4`.
