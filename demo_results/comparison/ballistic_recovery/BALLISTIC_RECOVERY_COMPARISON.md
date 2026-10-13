# Ballistic-impact recovery comparison

Privileged **authority constructions**, not learned-policy results.
No SAC was trained. Controlled-slip wrist rate in this video is `3` rad/s. `RECOVERY4D_W_HY_MAX` (4 rad/s) was not edited.

## Benchmark definition

- Scene: `assets/scene_ballistic_impact.xml` (noslip=1)
- Grasped cylinder, impact ball mass `0.05` kg, CENTER site, `v_hit = 6.6` m/s
- Video starts at the nominal grasp, continues through the lift, then the ball impact
- Impact `t = 2.8499999999999077` s
- Branch point: EARLY drift `t = 3.322000` s on this same episode
- ZERO, CONTROLLED_SLIP, and AIRBORNE restore that one snapshot
- At 6.0 m/s this simulator holds. 6.6 m/s is the same scene with a faster ball so ZERO drops.

## Branch-point equivalence

The grasp, lift, and impact are one recording copied onto all three panels.
Each panel then restores the snapshot captured at the end of that recording.

- SHA256 after ZERO restore: `b52b9e4fd3188fb9dcebc1c0586c03155fd49c032cb6cd2971a14f7a5f5ccc4c`
- SHA256 after CONTROLLED_SLIP restore: `b52b9e4fd3188fb9dcebc1c0586c03155fd49c032cb6cd2971a14f7a5f5ccc4c`
- SHA256 after AIRBORNE restore: `b52b9e4fd3188fb9dcebc1c0586c03155fd49c032cb6cd2971a14f7a5f5ccc4c`
- max |Δ| zero vs slip: `0.000e+00`
- max |Δ| slip vs airborne: `0.000e+00`

**PASS:** the three branch states are numerically identical.

## ZERO semantics

No recovery primitive. The panel continues the same EARLY grasp as the other two panels.

- `p_des`: continues the nominal lift until `z_tgt`, then holds that height
- `r_des`: unchanged (nominal orientation from the lift controller)
- `v_cmd` / `w_cmd`: nominal lift velocity until `z_tgt`, then **zero** (`tick_vw` hold)
- grip: `tau = -18` (`ctrl[7]` secure) throughout
- **Not** the old stale-`p_des` STOP freeze

Implementation: `continue_nominal` in `training/demo_ballistic_recovery_comparison.py`
(same lift-then-hold semantics as `continue_zero`; does not abort the video at first table contact).

Table contact at t=6.658 s. End t=11.322 s, obj_z=0.4180 m, nL=0, nR=0.

## CONTROLLED_SLIP construction

**PRIVILEGED AUTHORITY CONSTRUCTION** (timing/angle from the validated CENTER-6 proof).

1. `freeze()` (copy current `p_des`/`r_des`, zero `v_cmd`/`w_cmd`)
2. Secure rotate `+120°` at `omega_y = 3` rad/s, `tau=-18`
3. Weak-grip slip `tau=-5` for `0.70` s (`v_x=v_z=omega=0`)
4. Brake `tau=-18` for `0.40` s
5. `return_to_nominal` (legal `omega_y` ≤ 3 rad/s)
6. Hold via `continue_zero`

Baseline duration `0.7` s is the retained sufficiency result (also in `(0.7, 1.0, 1.3, 1.6)`).
Not an observable policy.

## AIRBORNE_RECAPTURE construction

**PRIVILEGED AUTHORITY CONSTRUCTION** + **DIAGNOSTIC SIMULATION AUTHORITY** (`omega_y=+4`).

This is **not** the training action bound and **not** a hardware-safe Panda velocity.
`tick_omega_override` uses diagnostic `omega_y=+4` for the throw only. The return to upright uses `omega_y` ≤ 3 rad/s.

Retained construction (user-visually-verified candidate):

- physically generated EARLY impact-offset state
- diagnostic `omega_y = +4` rad/s
- OPEN command at `70°` (`tau = +2`)
- genuine `FIRST_BOTH_OFF` and nonzero contact-free interval
- privileged CLOSE at `t_close = 3.7219999999998095` (`tau = -18`)
- bilateral recapture, then ≥2 s hold at the tilted catch
- then `return_to_nominal` (upright, `omega_y` ≤ 3 rad/s, `tau=-18`)
- then nominal lift until `z_tgt`, then hold (`continue_nominal`)

This replay events:
- `OPEN_COMMAND`: 3.689999999999815
- `FIRST_BOTH_OFF`: 3.699999999999814
- `CLOSE_COMMAND`: 3.7219999999998117
- `FIRST_RECONTACT`: 3.73599999999981
- `FIRST_BILATERAL`: 3.73599999999981
- `HOLD_START`: 3.7879999999998044
- `HOLD_COMPLETE`: 5.789999999999584
- `APEX`: 3.7979999999998033

After the tilted hold: upright return arrived=True, hand error deg=0.12808494948301585, resumed lift_s=1.855093655692324, z_tgt=0.7129629021557073, end p_des_z=0.7130354097003473.
## Raw outcome table

| branch | outcome | contact mode | max_abs_rhx_mm | contact_loss | recapture | hold_s | t_table |
|---|---|---|---|---|---|---|---|
| zero | DROP | no_recovery | 76.50 | True | None | 0.000 | 6.657999999999489 |
| controlled_slip | HOLD | controlled_gravitational_slip | 9.96 | False | None | 5.042 | None |
| airborne_recatch | HOLD | airborne_recapture | 10.92 | True | 3.73599999999981 | 2.002 | None |

See `demo_results/comparison/ballistic_recovery/raw/` for npz trajectories.

## Videos

- `demo_results/videos/ballistic_zero.mp4`
- `demo_results/videos/ballistic_controlled_slip.mp4`
- `demo_results/videos/ballistic_airborne_recatch.mp4`
- `demo_results/videos/ballistic_recovery_comparison.mp4`

## Limitations

- Controlled slip and airborne recatch are **authority constructions**, not learned policies.
- Airborne recatch uses diagnostic `omega_y=4` and privileged CLOSE timing.
- Not hardware validated.
- Unified observable SAC has **not** been trained on this benchmark.

