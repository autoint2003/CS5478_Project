# Release-state map at diagnostic ω_y = 4 rad/s

Mass **held fixed at 0.20 kg**. Friction, controller, grip map, noslip, and `v_x`/`v_z` bounds unchanged.
`RECOVERY4D_W_HY_MAX` **not** changed. `omega_y_cmd = +4.0` is a diagnostic override.
No recatch during mapping. Offset is evaluated in the **full release state**, not by `v_obj,z` alone.

Geometric corridor (open-gripper pocket, not a score): `|r_h.x|<22 mm`, `|r_h.y|<20 mm`, `50 < r_h.z < 130 mm`.

## 1. Initial-state provenance

### centered
- High-clearance physically lifted parent (same construction as omega=4 throw). Not teleported.
- t=5.163999999999653, mass=0.2, nL/nR=7/6 bilateral=True
- initial r_h mm = [0.2036324206718957, -0.0032617788333722934, 96.71459004672961]
- aperture=0.0176474697556497, v_rel=0.003724814910566812, obj_z=0.5583048137853303

### center6_early
- Physically generated CENTER-6 EARLY `D:\NUS\CS5478 Intelligent Robots\CS5478_Project\results\diagnostics\ballistic_recovery_transfer\raw\snap_early.pkl` from the impact/ballistic trajectory (documented r_h.x=10.87 mm). Not a synthetic offset grid.
- t=3.3219999999998557, mass=0.2, nL/nR=7/4 bilateral=True
- initial r_h mm = [10.870572177265952, -0.023633299379496247, 98.32954296062117]
- aperture=0.01760644389328352, v_rel=0.0006629673990185905, obj_z=0.5241285919100399

`snap_mid.pkl` exists on the same impact trajectory but was **not** added: the map is centered × EARLY only (six trials).

## 2. Supported trajectory near the upward-velocity peak (centered, tau=-18, ω=4)

| ang | v_obj,z | v_obj | r_h mm | v_rel_h | |w_obj| | nL/nR | ap | Fn L/R |
|---|---|---|---|---|---|---|---|---|
| 50.1 | +0.364 | [0.003, 0.216, 0.364] | [-0.18, -0.15, 100.29] | [0.27, 0.002, -0.016] | 3.90 | 8/8 | 0.0176 | 9.02/9.24 |
| 59.9 | +0.405 | [-0.006, 0.129, 0.405] | [-0.35, -0.16, 101.12] | [0.275, 0.001, -0.016] | 3.93 | 4/4 | 0.0176 | 9.01/9.14 |
| 69.9 | +0.430 | [-0.015, 0.04, 0.43] | [-0.56, -0.16, 101.98] | [0.279, 0.001, -0.015] | 3.96 | 4/4 | 0.0175 | 9.01/9.13 |
| 74.9 | +0.436 | [-0.02, -0.005, 0.436] | [-0.68, -0.15, 102.41] | [0.281, 0.001, -0.014] | 3.96 | 4/4 | 0.0175 | 9.01/9.12 |
| 79.5 | +0.438 | [-0.024, -0.045, 0.438] | [-0.78, -0.14, 102.81] | [0.283, 0.0, -0.013] | 3.97 | 4/4 | 0.0175 | 9.00/9.10 |
| 84.9 | +0.435 | [-0.029, -0.093, 0.435] | [-0.91, -0.13, 103.29] | [0.285, 0.0, -0.013] | 3.97 | 4/4 | 0.0175 | 9.01/9.09 |
| 89.9 | +0.427 | [-0.034, -0.136, 0.427] | [-1.03, -0.11, 103.73] | [0.287, 0.0, -0.012] | 3.97 | 4/4 | 0.0175 | 9.01/9.08 |

OPEN phases (from this log, not a dense sweep): current 76.9° (previous command, BOTH_OFF ~79° near peak), modestly earlier 70°, clearly earlier 60°.
12 ms latency ≈ 2.7° at 4 rad/s; contact dynamics may need a different lead, which is why earlier commands are included.

## 3–4. FIRST_BOTH_OFF complete states

| IC | phase | cmd | θ_OPEN | θ_SINGLE | θ_BOTH | lat s | last finger | r_h IC mm | r_h OPEN mm | r_h BOTH mm | v_z | v_obj | v_rel_h | |ω_obj| | rel° | ap | ω_hand | dt_apex | rise mm | r_h apex mm | corridor apex | Δhand° to apex |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| centered | current | 76.9 | 76.27133938907507 | 78.54997848554619 | 79.006860578047 | 0.011999999999998678 | left | [0.2, -0.0, 96.71] | [-0.71, -0.15, 102.53] | [-1.22, -0.14, 102.98] | +0.360 | [-0.02, -0.021, 0.36] | [0.206, 0.002, 0.023] | 3.99 | 179.7 | 0.0184 | 3.984 | 0.038 | 6.2 | [-9.988008695105039, -0.1640086869396359, 106.80344550994916] | True | 7.948942515908783 |
| centered | modestly_earlier | 70.0 | 69.45413000027166 | 71.27223686244683 | 72.18378544407722 | 0.011999999999998678 | left | [0.2, -0.0, 96.71] | [-0.55, -0.16, 101.94] | [-1.05, -0.15, 102.44] | +0.352 | [-0.012, 0.04, 0.352] | [0.205, 0.004, 0.032] | 4.05 | 179.7 | 0.0184 | 3.976 | 0.036 | 6.0 | [-9.153255495225004, -0.12097256004207771, 107.22062418475662] | True | 7.608849767968579 |
| centered | clearly_earlier | 60.0 | 59.49681764708905 | 61.30395375025369 | 62.21016907403666 | 0.011999999999998678 | left | [0.2, -0.0, 96.71] | [-0.34, -0.16, 101.09] | [-0.79, -0.16, 101.67] | +0.329 | [-0.002, 0.13, 0.329] | [0.208, 0.004, 0.044] | 4.04 | 179.8 | 0.0184 | 3.953 | 0.034 | 5.2 | [-7.840584709802924, -0.17637736835879206, 107.6460493761385] | True | 7.231482945067931 |
| center6_early | current | 76.9 | 76.13471293086495 | 78.8686854972077 | 79.32571417512469 | 0.013999999999998458 | left | [10.87, -0.02, 98.33] | [10.49, -0.08, 104.01] | [9.83, -0.08, 104.59] | +0.358 | [-0.016, -0.045, 0.358] | [0.2, 0.002, -0.014] | 3.92 | 179.9 | 0.0185 | 3.984 | 0.038 | 6.2 | [0.6475304098907047, 0.06269786879074565, 108.54447502522844] | True | 7.949246802687128 |
| center6_early | modestly_earlier | 70.0 | 69.32336802779378 | None | 72.05012634192931 | 0.011999999999998678 | left | [10.87, -0.02, 98.33] | [10.61, -0.08, 103.42] | [10.16, -0.08, 103.93] | +0.373 | [-0.009, 0.019, 0.373] | [0.22, 0.001, -0.013] | 3.91 | 180.0 | 0.0183 | 3.970 | 0.04 | 6.7 | [1.393631235978472, -0.014375265260391387, 109.45220934456385] | True | 8.240049756063627 |
| center6_early | clearly_earlier | 60.0 | 59.381172270072916 | 61.63703482201363 | 62.54260644596477 | 0.013999999999998458 | left | [10.87, -0.02, 98.33] | [10.75, -0.12, 102.59] | [10.19, -0.1, 103.34] | +0.332 | [0.003, 0.113, 0.332] | [0.204, 0.005, 0.012] | 3.99 | 180.0 | 0.0185 | 3.949 | 0.034 | 5.3 | [2.8040746241573062, 0.007943373971259462, 109.58188069211397] | True | 7.226208664860216 |

Do not rank by `v_obj,z` alone. Initial offset is **not** assumed equal to release offset.

## 5–8. World object vs hand vs relative (`r_h`) at apex

If `dp_obj` is small in a world axis but `r_h` changes, the corridor moved. Do not call that “the cylinder flew sideways” without world-frame evidence.

**Causal structure:** `r_h.x` decreases by about 8-9 mm from FIRST_BOTH_OFF to apex in every run. Centered starts near -1 mm and arrives at apex near -8 to -10 mm. CENTER-6 EARLY starts near +10 mm (still +10 mm at BOTH_OFF) and arrives near +0.6 to +2.8 mm. The same negative `r_h.x` drift therefore places the offset object near the grasp midline at apex, while the centered object is already walking off the squeeze axis. World-frame object motion to apex is millimetres (`dp_obj` mostly +z). The hand moves about -4 to -6 mm in world y and still rotates **7-8 deg** after PARK (`omega_hand` ~2.7-3.0 rad/s at apex). Packed `R_rel` ~180 deg is the cylinder-frame convention in `physical_pack`, not a free-flight tumble.

- **centered/current**: BOTH_OFF r_h mm=[-1.22, -0.14, 102.98]; apex r_h mm=[-9.988008695105039, -0.1640086869396359, 106.80344550994916]; dp_obj mm=[-0.77, -0.79, 6.21]; dp_hand mm=[-0.86, -6.05, 1.02]; hand Δθ=7.948942515908783; ω_hand apex=2.843655847547933; corridor apex=True; rise=6.2 mm; dt_apex=0.037999999999995815.
- **centered/modestly_earlier**: BOTH_OFF r_h mm=[-1.05, -0.15, 102.44]; apex r_h mm=[-9.153255495225004, -0.12097256004207771, 107.22062418475662]; dp_obj mm=[-0.42, 1.45, 5.96]; dp_hand mm=[-0.65, -5.24, 1.62]; hand Δθ=7.608849767968579; ω_hand apex=2.932661531268104; corridor apex=True; rise=6.0 mm; dt_apex=0.035999999999996035.
- **centered/clearly_earlier**: BOTH_OFF r_h mm=[-0.79, -0.16, 101.67]; apex r_h mm=[-7.840584709802924, -0.17637736835879206, 107.6460493761385]; dp_obj mm=[-0.08, 4.41, 5.17]; dp_hand mm=[-0.37, -4.04, 2.21]; hand Δθ=7.231482945067931; ω_hand apex=3.0130474365324496; corridor apex=True; rise=5.2 mm; dt_apex=0.033999999999996255.
- **center6_early/current**: BOTH_OFF r_h mm=[9.83, -0.08, 104.59]; apex r_h mm=[0.6475304098907047, 0.06269786879074565, 108.54447502522844]; dp_obj mm=[-0.59, -1.72, 6.15]; dp_hand mm=[-0.92, -5.57, 0.95]; hand Δθ=7.949246802687128; ω_hand apex=2.840518957422788; corridor apex=True; rise=6.2 mm; dt_apex=0.037999999999995815.
- **center6_early/modestly_earlier**: BOTH_OFF r_h mm=[10.16, -0.08, 103.93]; apex r_h mm=[1.393631235978472, -0.014375265260391387, 109.45220934456385]; dp_obj mm=[-0.38, 0.77, 6.68]; dp_hand mm=[-0.75, -5.34, 1.41]; hand Δθ=8.240049756063627; ω_hand apex=2.734972482799479; corridor apex=True; rise=6.7 mm; dt_apex=0.039999999999995595.
- **center6_early/clearly_earlier**: BOTH_OFF r_h mm=[10.19, -0.1, 103.34]; apex r_h mm=[2.8040746241573062, 0.007943373971259462, 109.58188069211397]; dp_obj mm=[0.1, 3.85, 5.29]; dp_hand mm=[-0.38, -3.54, 1.96]; hand Δθ=7.226208664860216; ω_hand apex=3.007570425162335; corridor apex=True; rise=5.3 mm; dt_apex=0.033999999999996255.

## 9. PARK transient (secure, tau=-18, ω_cmd→0 at current OPEN phase 76.9°)

- ang at PARK = 76.72632272315894
- ang after 100 ms = 87.6126415566413
- **Δθ in 100 ms = 10.886318833482349 deg**
- ω_axis at PARK = 3.9674946444566697, after 100 ms = 0.7939656832288446
- SO(3) err at PARK = 14.659555915799626°, after 100 ms = 4.108886349726617°
Controller not modified. This lag is part of the capture-corridor motion during the ~38 ms ballistic window.

## 10–11. Catchability (separate quantities, no weighted score)

| IC | phase | vz>0.20 | vz | dt_apex≥20ms | corridor apex | |r_h.x| apex ≤12 mm | clearly catchable |
|---|---|---|---|---|---|---|---|
| centered | current | True | +0.360 | True | True | True (-9.988008695105039) | True |
| centered | modestly_earlier | True | +0.352 | True | True | True (-9.153255495225004) | True |
| centered | clearly_earlier | True | +0.329 | True | True | True (-7.840584709802924) | True |
| center6_early | current | True | +0.358 | True | True | True (0.6475304098907047) | True |
| center6_early | modestly_earlier | True | +0.373 | True | True | True (1.393631235978472) | True |
| center6_early | clearly_earlier | True | +0.332 | True | True | True (2.8040746241573062) | True |

## 12. Selected candidate

Selected: **center6_early / modestly_earlier** (open cmd 70.0°).
Lexicographic: require clearly_catchable, then prefer larger `v_obj,z`. Not a blended score.

## 13. Privileged recatch

**PRIVILEGED.** CLOSE_COMMAND = mapped apex − 20 ms (aperture delay), **not** assumed equal to apex. Hold at catch orientation. No return-to-nominal.

| event | t (s) |
|---|---|
| OPEN_COMMAND | 3.690 |
| FIRST_BOTH_OFF | 3.702 |
| CLOSE_COMMAND | 3.722 |
| FIRST_RECONTACT = FIRST_BILATERAL | 3.738 |
| HOLD_START | 3.790 |
| HOLD end | 5.792 |

HOLD duration **2.002 s**, **1002/1002** samples bilateral (`nL=nR=8`), zero contact-loss samples. End `v_rel=4e-4` m/s, `r_h.x=−0.24 mm`, `r_h.z=110 mm`, `tau=-18`, `ω_cmd=0`.

The logs report a **candidate** successful dynamic throw-recapture construction. This is **not** yet user-verified. Do not treat it as a validated physical result until the viewer is watched.

## Viewer (exact claimed construction — watch this first)

Replay check against `raw/release_map_recatch.json` matched event times within 1 timestep:

BOTH_OFF 3.702, CLOSE_COMMAND 3.722, BILATERAL 3.738, HOLD_START 3.790, hold 2.002 s, final r_h.x −0.24 mm, |v_rel| 4e-4 m/s, tau −18.

```text
python training/demo_dynamic_recatch.py --mode offset_throw_recatch_omega4
```

CENTER-6 EARLY, diagnostic ω_y=+4, OPEN at 70°, CLOSE at saved apex−20 ms, τ=−18, catch orientation held. Overlay/console prints ROTATE, OPEN_COMMAND, FIRST_BOTH_OFF, APEX, CLOSE_COMMAND, FIRST_RECONTACT, FIRST_BILATERAL, HOLD_START, HOLD_COMPLETE. Default playback ~0.12×. SPACE pause, R restart, `[` `]` speed. No MP4.

Matched centered control (same ω=4, same OPEN 70°, PARK after both-off, **no CLOSE**):

```text
python training/demo_dynamic_recatch.py --mode centered_throw_omega4_70
```

Inspect visually: genuine both-off, contact-free interval, independent flight, close around the cylinder vs residual contact, 2 s hold. Event labels are not a substitute for that.

No ω>4, no mass/friction change, no synthetic offsets, no SAC, no permanent bound change, no MP4.
