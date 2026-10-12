# Ball-impact dataset expansion

Generation only. No Transformer, BC, GRPO, or PPO update was run. Physics, the success definition, and the sealed GRPO pilot were not changed. Existing TRAIN, VALIDATION, and FINAL_TEST assignments were kept. Post-impact cylinder pose and velocity were not edited.

The pre-expansion document is `results/diagnostics/raw/ball_impact_dataset_expansion/BALL_IMPACT_DATASET_REBUILD_BEFORE.md`. The live rebuild summary remains `results/diagnostics/BALL_IMPACT_DATASET_REBUILD.md` and still ends `BALL-IMPACT DATASET INCOMPLETE`.

## Launches and acceptance

Baseline before this expansion: 229 launches, accepted train 18, validation 26, final test 39.

Two resumable batches were run:

| Batch | Command | Launches | New accepted |
| --- | --- | ---: | --- |
| Mixed scheduler | `python -u training/ball_impact_dataset.py --attempts 120` | 120 | train +2, validation +0, final test +5 |
| TRAIN only | `python -u training/ball_impact_dataset.py --attempts 100 --split train` | 100 | train +8, validation +0, final test +0 |

Expansion launches: 220. Expansion accepts: 15 (`valid_impact_recovery_ic`). Expansion rejections: 205, all `impact_without_recovery_entry`. No launch was rejected for a missing snapshot or for an already-nominal basin.

Totals now: 449 launches, 98 accepted.

| Split | Before | After | Target |
| --- | ---: | ---: | ---: |
| TRAIN | 18 | 28 | 300 |
| VALIDATION | 26 | 26 | 60 |
| FINAL_TEST | 39 | 44 | 120 |

New accepted ids: `ball_train_0019` through `ball_train_0028`, and `ball_final_test_0040` through `ball_final_test_0044`.

The first batch still let the existing scheduler spend attempts on under-target validation and final-test azimuths. The second batch was restricted to TRAIN. Acceptance was not relaxed. FINAL_TEST ids were not used for privileged recovery search.

## Launch parameters

TRAIN is still only azimuths 0°, 45°, and 90° (`+X`, `oblique_+X_+Y`, `+Y`). VALIDATION remains 135° and 180°. FINAL_TEST remains 225°, 270°, and 315°. Azimuth overlap across splits is false. That separation is why TRAIN cannot be filled by moving the other splits, and why the three TRAIN directions are a coverage limit rather than a sampling accident.

Expansion attempts by direction: `+Y` 64, `oblique_+X_+Y` 60, `-X` 40, `-Y` 23, `oblique_+X_-Y` 17, `+X` 16.

Expansion accepts by direction: `+Y` 5, `oblique_+X_+Y` 5, `oblique_+X_-Y` 3, `-Y` 2. The 16 new `+X` launches contacted the cylinder and did not reach the 6 mm geometry entry. No new TRAIN `+X` initial condition was accepted.

Accepted expansion speeds, in m/s: 7.3 (5), 6.3 (3), 7.7 (2), and one each of 6.5, 6.7, 7.5, 8.7, and 9.5. Most accepts are in the slower half of the launch grid. Faster cells were attempted and mostly failed the entry threshold.

Accepted expansion strike-height offsets: +13 mm (5), +2 mm (4), −2 mm (3), 0 (2), +15 mm (1). Tangential offsets: ±35° (4 each), −5° (4), 0° (2), +40° (1). Launch delay: 0.40 s on 13 of the 15 accepts, and 0 s on 2.

## Post-impact grasp states

Labels overlap. Rolling still requires at least 1 rad/s, at least 3 mm of contact migration, and a tangential speed consistent with radius times angular rate. Entry spin alone is not controlled rolling.

Accepted episodes now, out of 98:

| State | Count |
| --- | ---: |
| Unilateral or asymmetric contact | 91 |
| Contact migration | 61 |
| Rolling signature | 50 |
| Tangential slip | 13 |
| Complete grasp loss | 6 |
| Airborne before entry | 1 |

TRAIN subset, out of 28: unilateral 25, contact migration 15, rolling signature 9, tangential slip 3, complete grasp loss 2, airborne 1. The new airborne complete-loss TRAIN case is `ball_train_0023` (pinch 48.7 mm, object orientation 1.32 rad). Validation and final-test counts were not used to retarget the recovery search.

## Leakage and snapshot fidelity

Duplicate launch keys: 0. Azimuth overlap: false. Accepted episodes with the ball still touching the cylinder at entry: 0. Other ball contact at entry: 0.

Restore-and-continue qpos error over 10 nominal steps, maximum 8.699e-4, tolerance 1e-3. Episodes over tolerance: 0. Cylinder mass, ball mass, friction, and gravity were not randomized.

## Privileged search on the new TRAIN initial conditions

The original 18 TRAIN snapshots kept their replay-verified C trajectories. The 10 new TRAIN snapshots received the same four A laws, four B laws, and four release–chase–regrasp laws (`release_long`, `chase_lost`, `lift_soft`, `release_short`).

New verified successes: 9, all mechanism C, all with a 10 s nominal-only hold and a 12 s diagnostic. They are `ball_train_0019`, `0020`, `0021`, `0022`, `0024`, `0025`, `0026`, `0027`, and `0028`. No new verified A or B trajectory was found.

`ball_train_0023` dropped within 0.2 s under all four A laws, all four B laws, and all four C laws. It enters already airborne, with no finger contact. That is a search failure inside the budget, not a claim that no longer search could recover it.

## Unresolved mechanisms

Directional lift with a causal world-up component does not enter the nominal basin on this grasp: hand +z points world-down, so motion along the dominant pinch offset is downward. Gravity-assisted tilt produces large cylinder spin and a drop, and the pinch error does not fall. Those two mechanisms remain unresolved. Release, chase, and regrasp remains the verified family, except on the airborne large-offset entry `ball_train_0023`.

K=36 and the ±0.25 residual bound were not changed. The new C trajectories have the same open-phase hand-frame vy and vz gap. No verified B trajectory exists, so no rolling primitive is added.

## Resume

TRAIN is 28 / 300. Continue TRAIN-only generation, then search only the new TRAIN ids:

```
python -u training/ball_impact_dataset.py --attempts 150 --split train
python -u training/abc_recovery_validation.py --max-ics 28
```

Raise `--max-ics` to the current TRAIN count after the next generation batch. Do not search FINAL_TEST. Do not start `training/formal_grpo.py`.

Raw counts: `results/diagnostics/raw/ball_impact_dataset_expansion/batch_stats.json`.
Attempt log: `results/diagnostics/raw/ball_impact_dataset/attempts.jsonl`.

ABC MECHANISM VALIDATION COMPLETE

BALL-IMPACT DATASET EXPANSION PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
