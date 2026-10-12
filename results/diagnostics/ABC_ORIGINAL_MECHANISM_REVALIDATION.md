# Original A/B/C mechanism revalidation

The A/B laws in `training/abc_recovery_validation.py` are not the historical CENTER-6 gravity sequence. Their 0/28 result does not show that the original large-angle slip is physically infeasible.

No policy was trained. Physics constants were not edited. Existing reports and trajectory files were not overwritten. The historical modules were restored from commit `696c74c` and given small import shims so they load against the current simulator. The historical markdown was not regenerated.

## 1. What the failed A/B search actually ran

`action_a` translates the hand along the pinch offset and adds a world-up term while the grip is already unloaded. `action_b` tilts about 0.4–1.0 rad and translates perpendicular to the offset while the grip is unloaded. On the ball-impact snapshots that tilt produced spins of several to tens of rad/s and a drop. Neither law is the published CENTER-6 sequence: secure grip, large positive hand-y rotation, then a moderate unload, then a secure brake, then a return toward the original wrist.

## 2. Historical A, reproduced

Source: `training/demo_ballistic_recovery.py` mode `gravity_large_angle`, implemented in `training/ballistic_large_angle.py`, report `results/diagnostics/ballistic_recovery_transfer/LARGE_ANGLE_GRAVITY_TRANSFER.md`, snapshot `results/diagnostics/ballistic_recovery_transfer/raw/snap_early.pkl`.

Restored EARLY state, this run:

| quantity | reproduced | historical report |
| --- | ---: | ---: |
| time | 3.322 s | 3.322 s |
| hand-frame pinch x | +10.87 mm | +10.87 mm |
| g_h.x | +0.013 m/s² | +0.013 m/s² |
| contacts L/R | 7/4 | 7/4 |
| axis tilt | 0.19° | 0.19° |

Published sequence, replayed by `apply_gravity_sequence` with `θ=+120°`, `τ=-5` for 0.70 s, `τ=-18` for 0.40 s, then wrist return and nominal continuation:

| event | this replay |
| --- | --- |
| slip start / end r_h.x | 9.50 mm → 8.39 mm (Δ −1.11 mm) |
| contacts through the slip | 10/10 |
| g_h.x during the slip | −8.59 m/s² |
| after the secure brake | 8.29 mm, contacts 10/10 |
| after wrist return | 7.61 mm, residual wrist 0.03°, hand error 0.18° |
| end | t = 23.21 s, no table contact, z = 0.587 m, contacts 9/9, r_h.x = 7.58 mm |

The historical writeup recorded Δ −1.62 mm (9.96 → 8.34 mm) and an end offset of 11.5 mm at t = 22.08 s. This replay keeps the same sign, bilateral contact, and survival past 22 s. The millimetre values are close on the slip window and differ on the long hold. It is a reproduction of the mechanism, not a bitwise copy of the old log.

Causal checks from the same snapshot:

- Zero gravity during the weak dwell: `SECURE_LIKE`, Δ r_h.x = +0.05 mm, contacts 5/4. The correcting slide disappears. The historical 0 g row was +0.09 mm and `SECURE_LIKE`.
- Commanded −120°: the sequence never entered the slip window (`slip_drhx_mm` absent). The historical −120° row lost both contacts. Wrong-sign rotation is not a correcting slip.

Level 1 for this episode: yes. Gravity-driven correcting slip under bilateral contact, removed by zero gravity.

Level 2: no. The offset stays near 7.6–8.4 mm, outside the 3.03 mm pinch gate.

Level 3: no under the current nominal-basin handoff. The object is still held at t = 23 s by the original lift continuation. That is the historical survival claim, not a 10 s nominal-only hold inside the 3.03 mm / 0.117 rad basin.

Raw: `results/diagnostics/raw/abc_original_revalidation/center6_replay.json`.

No MP4 was written. The historical note already said the user video was pending, and this replay did not add one.

## 3. Historical B was not recovered

The requested B is a near-horizontal single-sided support, with the cylinder sliding or rolling inward along one finger. The recovered CENTER-6 experiment keeps both fingers in contact (10/10 during the slip). It is not that mechanism.

A search of the restored ballistic reports does not find one trial that moves a grasp from about 12.3 mm to about 8.2 mm under a favorable tilt. The nearby figures are different rows:

- −60° secure rotation leaves r_h.x at 12.33 mm and points gravity the wrong way.
- The +120° brake in this replay ends at 8.29 mm, starting from 9.50 mm, still bilateral.

No snapshot or script for a unilateral support-plane reseat was found. A new single-sided controller was not built. Anything later that does that must be labeled a reconstruction.

## 4. Historical C, representative replay

Saved actions from `results/diagnostics/raw/sliding_vs_regrasp/trajectories/`, replayed with the original call `handoff_z=None`.

| IC | file | 10 s | 12 s | zero-contact steps | min pinch |
| --- | --- | --- | --- | ---: | ---: |
| `ball_train_0003` | `regrasp_lift_soft` | yes | yes | 21 | 0.51 mm |
| `ball_train_0014` | `regrasp_chase_lost` | yes | yes | 32 | 0.43 mm |
| `ball_train_0018` | `regrasp_lift_soft` | yes | yes | 46 | 0.33 mm |

The same three files with an added handoff height gate of 0.455 m kept the 10 s hold only on `ball_train_0014`. `ball_train_0003` and `ball_train_0018` reached sub-millimetre pinch and were labeled `BASIN_NOT_REACHED`. The later A/B/C script used that 0.455 m gate. The original C replay did not. The 27/28 figure was not re-simulated in full in this pass.

Level 1 and level 3 both hold on these three replays when the original handoff gate is used: the actions open, lose contact, regrasp, and the nominal controller holds 10 s and 12 s.

## 5. Matched ball-impact A/B/C

Not run. The restored A controller is the 4D ballistic stepper on `snap_early.pkl` (`ω_y`, `v_x`, `v_z`, grip torque), with the Cartesian pose frozen during the rotation. The 28 TRAIN captures live in the later training environment and were not substituted for that snapshot.

## 6. Next step

Port `apply_gravity_sequence` onto the training environment only as an explicitly labeled transcription: secure `ω_y = +3 rad/s` until the measured hand angle makes `g_h` point against the pinch error, then `τ=-5`, then `τ=-18`, then a wrist return. Run that on the 28 TRAIN captures with the original C handoff gate (`None`), and keep level 1 separate from the 3.03 mm basin. Do not reuse `action_a` / `action_b`.

ORIGINAL A/B/C MECHANISM REVALIDATION: PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
