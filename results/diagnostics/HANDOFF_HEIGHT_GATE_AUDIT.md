# Handoff height gate

`handoff_z` is an extra check in `execute` (`training/sliding_vs_regrasp.py`). It is not part of `basin_ok`.

## What the number is

`HANDOFF_Z = 0.455` is compared with `errors()["z"]`, which is the cylinder body origin in the world frame, in metres. It is not the hand height and not the bottom of the cylinder.

The thresholds already in the task are different:

| name | value | role |
| --- | ---: | --- |
| `TABLE_DROP` | 0.440 m | table top 0.40 m plus the 0.03 m half-height plus 0.01 m |
| `Z_ABORT` | 0.440 m | recovery abort and nominal-hold drop |
| `HANDOFF_Z` | 0.455 m | optional extra gate inside `execute` |
| `Z_AIR` | 0.480 m | height at which the impact generator treats the nominal lift as ready to launch |

`basin_ok` requires bilateral contact, orientation ≤ 0.117 rad, pinch ≤ 3.03 mm, relative speed ≤ 0.020 m/s, and a closed grip command, for 0.100 s. It does not read height.

When `handoff_z` is set, a state that already passes `basin_ok` is ignored until `z >= 0.455`. When `handoff_z` is `None`, the basin alone opens the handoff. `apply_handoff` then copies the saved nominal orientation into `r_des`, zeros the commanded twist, and sets `p_des` to the current hand position. It does not write `qpos` or `qvel`.

`nominal_validate` runs that nominal controller for 10 s and 12 s after the handoff. A drop is `z` below `TABLE_DROP` or `Z_ABORT`, an escape, or both contacts gone for 1.5 s. Reaching `Z_AIR` is not required for `SUCCESS_10S` / `SUCCESS_12S`. Absolute simulator time t = 12 is recorded separately as `legacy_retained_to_12`.

## Same actions, measured height

Saved release–regrasp actions were stepped again. The first 0.100 s that passed `basin_ok` had:

| IC | pinch | orientation | speed | object z | `z >= 0.455` | `z >= 0.480` |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `ball_train_0003` | 0.65 mm | 0.019 rad | 0.003 m/s | 0.445 m | no | no |
| `ball_train_0014` | 0.43 mm | 0.086 rad | 0.016 m/s | 0.470 m | yes | no |
| `ball_train_0018` | 0.33 mm | 0.101 rad | 0.010 m/s | 0.450 m | no | no |

Contacts were bilateral (8/8, 8/8, 10/10). An earlier replay of these same files held for 10 s and 12 s with `handoff_z=None`, and `0003` and `0018` were `BASIN_NOT_REACHED` when the 0.455 m gate was forced. The gate is what changed. The grasp was already inside the strict basin, 5–10 mm under 0.455 m and still above the 0.440 m drop line.

None of the three readiness states had reached the 0.480 m lift-ready height. That height is the pre-impact launch condition. It is not the recovery-success definition. The nominal controller is allowed to keep lifting after the handoff.

## Decision

`0.455 m` is a recovery heuristic sitting between the drop line and the lift-ready height. It is not required by `basin_ok` or by `nominal_validate`. Using it rejects grasps that the nominal controller has already held for 10 s and 12 s.

New TRAIN labels in this pass use `handoff_z=None` for stage A and stage B. Object z at readiness is stored. A later nominal hold still fails if the cylinder falls through 0.440 m. `Z_AIR` is reported when the continuation reaches it, and it is not an extra rejection at the handoff instant.

`HANDOFF_Z` in `sliding_vs_regrasp.py` is left at 0.455 so the published matched-search report stays tied to the code that produced it.

Raw: `results/diagnostics/raw/handoff_height_audit/replay.json`.

HANDOFF HEIGHT GATE AUDIT: COMPLETE

POLICY TRAINING HAS NOT BEEN STARTED
