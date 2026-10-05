# Airborne release → recapture authority

No ballistic impact. No CENTER-6. No SAC. No MP4.

If a recapture sequence is shown below, it is a **PRIVILEGED AUTHORITY CONSTRUCTION** (close timing uses `nL`/`nR`). It is **not** an observable recovery controller and **not** a learned policy.

---

## 1. Grip-action-space audit

Executed from `controllers/residual.py` `map_recovery4d` and the live MuJoCo model (not comments).

| item | value |
|---|---|
| a3 range | **[-1, +1]** |
| formula | `tau = TAU_OPEN + 0.5*(a3+1)*(TAU_SECURE-TAU_OPEN)` |
| a3=-1 → tau (module at static audit) | -1.0 |
| a3=0 → tau | -9.5 |
| a3=+1 → tau | -18.0 |
| sign | negative ctrl[7] closes/squeezes; positive opens |
| XML/runtime ctrlrange[7] | [-50.0, 50.0] |

Legal 4D **before** the correction below was entirely negative (`[-18, −1]`): no true active opening.

## 2. Active-opening audit (legal 4D **before** extension)

| tau | class | ap0→ap1 mm | nL/nR | both-off |
|---|---|---:|---|---|
| -18 | SECURE_CLOSE | 17.6 → 17.6 | 7/6 | None |
| -8 | REDUCED_SQUEEZE | 17.6 → 17.7 | 6/6 | None |
| -5 | REDUCED_SQUEEZE | 17.6 → 17.8 | 8/8 | None |
| -2 | REDUCED_SQUEEZE | 17.6 → 17.8 | 1/3 | 4.3179999999997465 |
| -1 | PASSIVE_RELEASE_INSUFFICIENT_FN | 17.6 → 16.3 | 0/0 | 4.131999999999767 |

**D. ACTIVE FINGER OPENING was absent** from legal a3. `tau=-1` is **C**: contacts can be lost while aperture does **not** command-open. That is an action-space design limit, not a recapture-physics failure.

## 3. Grip-range correction (same 4th dimension)

CURRENT 4D ACTION SPACE DID NOT INCLUDE ACTIVE OPENING (a3∈[-1,1] → tau∈[-1, -18]). Minimal extension of the **same** a3 dimension: `RECOVERY4D_TAU_OPEN = 2` so a3=-1 → ctrl[7]=2 (active open), a3=+1 → -18 (secure). Intermediate taus including -5 remain reachable. Positive sweep used tick_vw **outside** 4D only as an actuator diagnostic; recapture uses the frozen 4D map only.

Positive-ctrl diagnostic (`tick_vw`, not recapture commands):

| ctrl[7] | class | ap0→ap1 mm |
|---|---|---:|
| 0 | NEAR_ZERO_ACTUATION | 17.6 → 20.8 |
| 2 | ACTIVE_FINGER_OPENING | 17.6 → 42.6 |
| 4 | ACTIVE_FINGER_OPENING | 17.6 → 41.8 |
| 6 | ACTIVE_FINGER_OPENING | 17.6 → 41.9 |
| 8 | ACTIVE_FINGER_OPENING | 17.6 → 42.0 |

Frozen: `a3=-1 → +2` (open), `a3=+1 → -18` (secure). RULE yaml `tau_open: -1` unchanged.

Sanity after freeze:

| tau | class | ctrl[7] |
|---|---|---:|
| -18 | SECURE_CLOSE | -18.0 |
| -5 | REDUCED_SQUEEZE | -5.0 |
| 2 | ACTIVE_FINGER_OPENING | 2.0 |

## 4. Clean airborne parent

| | |
|---|---|
| t | 4.114 s |
| phase | lift |
| obj_z | 0.496 m |
| table clearance | 66 mm |
| nL/nR | 7 / 6 |
| v_rel | 0.35 mm/s |
| ω_rel | 0.001 rad/s |
| e_x | 0.22 mm |
| aperture | 17.6 mm |
| scene | 0 |
| ctrl[7] | -18.0 |
| noslip | 1 |

Snapshot: `raw/recapture_parent.pkl`.

## 5. Physical release

From the parent, legal `a3=-1` (`tau=+2`): aperture increases, contacts drop, object qpos/qvel are integrated (no teleport).

## 6–8. Contact-free interval, free-flight, hand follow

| open_s | vz | first both-off | persist s | ap mm | r_h.z mm | v_obj_z | table? |
|---:|---:|---|---:|---:|---:|---:|---|
| 0.06 | 0.00 | 4.123999999999768 | 0.04999999999999449 | 31.4 | 112.7 | -0.486 | no |
| 0.10 | 0.00 | 4.123999999999768 | 0.08999999999999009 | 43.5 | 142.0 | -0.878 | no |
| 0.14 | 0.00 | 4.123999999999768 | 0.1079999999999881 | 42.6 | 160.1 | -1.055 | 4.231999999999756 |
| 0.20 | 0.00 | 4.123999999999768 | 0.1079999999999881 | 42.6 | 160.1 | -1.055 | 4.231999999999756 |
| 0.14 | -0.08 | 4.123999999999768 | 0.1079999999999881 | 42.6 | 157.4 | -1.056 | 4.231999999999756 |

Legal `|v_z|≤0.08` cannot match `g t` after a long release. Arbitrary-duration free fall is not recoverable.

## 9–11. Recapture construction / recontact vs capture / hold

PRIVILEGED AUTHORITY CONSTRUCTION succeeded: open tau=2 until both-off persist 0.04 s, vz_follow=0.0, then tau=-18. Events {'OPEN_START': 4.113999999999769, 'FIRST_BOTH_OFF': 4.123999999999768, 'CLOSE_START': 4.161999999999764, 'FIRST_RECONTACT': 4.183999999999761, 'FIRST_BILATERAL': 4.183999999999761, 'SECURE_CAPTURE': 4.2359999999997555, 'HOLD_START': 4.2359999999997555}. both-off interval 0.037999999999995815 s. Not an observable controller.

| event | t (s) |
|---|---:|
| OPEN_START | 4.114 |
| FIRST_BOTH_OFF | 4.124 |
| CLOSE_START | 4.162 |
| FIRST_RECONTACT | 4.184 |
| FIRST_BILATERAL | 4.184 |
| SECURE_CAPTURE | 4.236 |
| HOLD_START | 4.236 |

Long hold survived: table=None end n=6/9 obj_z=0.461 v_rel=0.0000.

End of hold: nL/nR=6/9, obj_z=0.461 m, clear=0.030 m, scene=0, v_rel=0.00005 m/s, table=None.

First finger contact is **not** counted as success. Secure capture requires bilateral contact and relative-motion arrest, then a 12 s elevated hold.

## 12. Failure mechanism

Existence succeeded. Nearby extra=0.06 both-off hits the table after recontact (window too long; legal v_z cannot chase a long fall).

## 13. Interactive viewer commands

```text
python training/demo_airborne_recapture.py --mode zero
python training/demo_airborne_recapture.py --mode recapture
```

SPACE pause, R restart, `[` `]` speed. Camera initialized once. No MP4.
Headless: `python training/demo_airborne_recapture.py --mode sweep`

## 14. Limitations / privileged information

- Privileged nL/nR used to time CLOSE. Not a policy.
- Observation vector / reward / success_obs not changed. SCALE_TAU in physical_recovery still uses -1 as the old open end; that is an obs-scale leftover for the later obs/reward pass.
- RULE yaml tau_open=-1 was not changed (frozen RULE).
- No ballistic / CENTER-6 / SAC / MP4.
- Legal |v_z|≤0.08 m/s vs g=9.81 limits catchable free-fall duration.

Raw: `results/diagnostics/airborne_recapture/raw/`.