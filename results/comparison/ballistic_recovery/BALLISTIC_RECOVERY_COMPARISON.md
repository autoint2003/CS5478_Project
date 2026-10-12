# Ballistic-impact recovery comparison

Privileged **authority constructions**, not learned-policy results.
No SAC was trained. Main training bound `RECOVERY4D_W_HY_MAX = 3 rad/s` was not changed.

## Benchmark definition

- Scene: `assets/scene_ballistic_impact.xml` (noslip=1)
- Cylinder + ball mass `0.05` kg, CENTER site, `v_hit = 6.0` m/s
- Nominal grasp/lift, then genuine ballistic collision
- Branch point: EARLY drift `t = 3.322000` s (after impact, before drop)
- Shared prefix recorded from `t = 0` (before disturbance)

## Branch-point equivalence

All three recovery branches restore the **same live prefix snapshot**.

- SHA256 of live branch state: `4738614b105e9388cc6a1ea26201c4a3bbebd0c85c1118eac1b9a5382f204e05`
- SHA256 after ZERO restore: `d6ae2597e01cd6729567f6e99ea4ccee370ae29b1028a6fdf81152fa087b99e1`
- SHA256 after SLIP restore: `d6ae2597e01cd6729567f6e99ea4ccee370ae29b1028a6fdf81152fa087b99e1`
- SHA256 after AIRBORNE restore: `d6ae2597e01cd6729567f6e99ea4ccee370ae29b1028a6fdf81152fa087b99e1`
- max |Δ| ZERO vs SLIP restore: `0.000e+00`
- max |Δ| ZERO vs AIRBORNE restore: `0.000e+00`
- max |Δ| live prefix vs saved `snap_early.pkl`: `1.534e-04` (sub-mm; video and branches use the **live** prefix snap, not a mix of pickle vs live)

The live-run hash and the restore hashes differ because the restore is a fresh `load_snapshot` on a new `MjData`. The **three branch restores** of that same snap are bit-identical.

**PASS:** branches are numerically identical at the branch point before recovery diverges.

## ZERO semantics

After the matched impact, **no recovery primitive** is applied.

- `p_des`: continues the nominal lift until `z_tgt`, then holds that height
- `r_des`: unchanged (nominal orientation from the lift controller)
- `v_cmd` / `w_cmd`: nominal lift velocity until `z_tgt`, then **zero** (`tick_vw` hold)
- grip: `tau = -18` (`ctrl[7]` secure) throughout
- **Not** the old stale-`p_des` STOP freeze

Implementation: `continue_nominal` in `training/demo_ballistic_recovery_comparison.py`
(same lift-then-hold semantics as `continue_zero`; does not abort the video at first table contact).

## CONTROLLED_SLIP construction

**PRIVILEGED AUTHORITY CONSTRUCTION** (timing/angle from the validated CENTER-6 proof).

1. `freeze()` (copy current `p_des`/`r_des`, zero `v_cmd`/`w_cmd`)
2. Secure rotate `+120°` at legal `omega_y = 3.0` rad/s, `tau=-18`
3. Weak-grip slip `tau=-5` for `0.70` s (`v_x=v_z=omega=0`)
4. Brake `tau=-18` for `0.40` s
5. `return_to_nominal` (legal `omega_y` ≤ 3 rad/s)
6. Hold via `continue_zero`

Baseline duration `0.7` s is the retained sufficiency result (also in `(0.7, 1.0, 1.3, 1.6)`).
Not an observable policy.

## AIRBORNE_RECAPTURE construction

**PRIVILEGED AUTHORITY CONSTRUCTION** + **DIAGNOSTIC SIMULATION AUTHORITY** (`omega_y=+4`).

This is **not** the training action bound and **not** a hardware-safe Panda velocity.
`tick_omega_override` bypasses `RECOVERY4D_W_HY_MAX=3` for this diagnostic only.

Retained construction (user-visually-verified candidate):

- physically generated EARLY impact-offset state
- diagnostic `omega_y = +4` rad/s
- OPEN command at `70°` (`tau = +2`)
- genuine `FIRST_BOTH_OFF` and nonzero contact-free interval
- privileged CLOSE at `t_close = 3.7219999999998095` (`tau = -18`)
- bilateral recapture, then ≥2 s hold

This replay events:
- `OPEN_COMMAND`: 3.689999999999815
- `FIRST_BOTH_OFF`: 3.701999999999814
- `CLOSE_COMMAND`: 3.7219999999998117
- `FIRST_RECONTACT`: 3.73599999999981
- `FIRST_BILATERAL`: 3.73799999999981
- `HOLD_START`: 3.789999999999804
- `HOLD_COMPLETE`: 5.791999999999584
- `APEX`: 3.799999999999803
- contact-free interval (`FIRST_BOTH_OFF` → `FIRST_RECONTACT`): **0.034 s** (nonzero; object in free flight)
- end hold: bilateral `nL=8 nR=8`, `|r_h.x|≈0.11 mm`, `v_rel≈4.6e-4` m/s, `ctrl[7]=-18`

## Raw outcome table

| branch | outcome | contact mode | max_abs_rhx_mm | contact_loss | recapture | hold_s | t_table |
|---|---|---|---|---|---|---|---|
| zero | DROP | no_recovery | 410.90 | True | None | 0.000 | 6.131999999999547 |
| controlled_slip | HOLD | controlled_gravitational_slip | 10.05 | False | None | 5.032 | None |
| airborne_recatch | HOLD | airborne_recapture | 10.95 | True | 3.73799999999981 | 2.002 | None |

See `results/comparison/ballistic_recovery/raw/` for npz trajectories.

## Videos

- `results/videos/ballistic_zero.mp4`
- `results/videos/ballistic_controlled_slip.mp4`
- `results/videos/ballistic_airborne_recatch.mp4`
- `results/videos/ballistic_recovery_comparison.mp4`

## Limitations

- Controlled slip and airborne recatch are **authority constructions**, not learned policies.
- Airborne recatch uses diagnostic `omega_y=4` and privileged CLOSE timing.
- Not hardware validated.
- Unified observable SAC has **not** been trained on this benchmark.

