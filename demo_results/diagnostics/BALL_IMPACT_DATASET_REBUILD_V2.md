# Ball-impact dataset rebuild v2

Audit of the dataset that already exists. No new launches were generated in this pass. The generator rewrites `demo_results/diagnostics/BALL_IMPACT_DATASET_REBUILD.md` on every batch, and that report was left untouched.

## Counts

| | launches | accepted |
| --- | ---: | ---: |
| Before the previous expansion | 229 | train 18, validation 26, final test 39 |
| Now | 449 | train 28, validation 26, final test 44 |
| Planning targets | — | 300 / 60 / 120 |

Funnel on the 449 rows: 351 `impact_without_recovery_entry`, 98 `valid_impact_recovery_ic`. Duplicate launch keys: 0. Azimuth overlap across splits: false. Maximum restore qpos error: 8.699e-4, tolerance 1e-3. Ball still touching at entry: 0.

## Why TRAIN has only three directions

`SPLIT_AZIMUTH` in `training/ball_impact_dataset.py` is fixed before generation:

- TRAIN: 0°, 45°, 90° (`+X`, `oblique_+X_+Y`, `+Y`)
- VALIDATION: 135°, 180°
- FINAL_TEST: 225°, 270°, 315°

The scheduler cannot give TRAIN another azimuth without editing that table. In the last expansion, 16 new `+X` launches contacted the cylinder and never reached the 6 mm entry, so even inside the allowed set the +X cell is thin. Existing validation and final-test episodes were not moved.

## Proposed leakage-safe revision

Do not reassign the current 28 / 26 / 44 ids.

All eight compass azimuths are already owned by a split. A later campaign that needs every direction inside TRAIN has to be a new split rule, not a copy of validation rows. One rule that keeps leakage closed: hash the full launch key, assign new episodes to train / validation / final test by that hash, and refuse any key already present in another split. Old episodes stay where they are. That protocol was not applied.

## Acceptance

Unchanged: real ball–cylinder contact, geometry entry after that contact, a restorable snapshot, and a non-basin entry. Mass and friction stay at the XML values. Post-impact cylinder pose is not edited.

## Status

The dataset is not ready for a claim of coverage. TRAIN is 28 accepted states on three approach directions, mostly unilateral contact, with little complete grasp loss.

BALL-IMPACT DATASET REBUILD: PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
