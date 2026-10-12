# Targeted ball-impact generation

Protocol `ball_impact_v2`. New rows are only under `results/diagnostics/raw/ball_impact_dataset_v2`. The v1 attempts log, v1 pickles, and `BALL_IMPACT_DATASET_REBUILD.md` were not rewritten. The old generator now writes `BALL_IMPACT_DATASET_REBUILD_CURRENT.md` if it is run, instead of the frozen report.

## Why TRAIN had three azimuths

`SPLIT_AZIMUTH` in `training/ball_impact_dataset.py` assigns directions before any episode: TRAIN 0/45/90, VALIDATION 135/180, FINAL_TEST 225/270/315. The scheduler cannot emit another TRAIN azimuth. That is the split table, not a failed random draw. v1 validation and final test stay sealed. v2 does not move them.

## v2 sampling

Each draw picks the 30° bin with the fewest accepts, then the least-used speed in that bin, and aims the ball with the existing ballistic constructor: cylinder radius, ball radius, a 0.10 s flight, and gravity. MuJoCo has to produce a ball-cylinder contact. A robot or environment contact on an earlier timestep is rejected. The 6 mm entry rule is unchanged. Draw indices with `index % 10 >= 8` are not simulated; those residues are reserved for a later validation block and a later final-test block.

## Counts

| set | accepted pickles |
| --- | ---: |
| v1 TRAIN | 28 |
| v1 VALIDATION | 26 |
| v1 FINAL_TEST | 44 |
| v2 TRAIN | 21 |

v2 log: 96 launches. 75 hit the cylinder and did not reach the 6 mm entry. 21 were accepted. Robot-before-cylinder rejections: 0. Near-duplicate skips: 0. Every logged v2 launch that was simulated produced a ball-cylinder contact. Acceptance is 21/96.

Accepted azimuths, degrees: 16.8, 39.3, 46.9, 63.3, 72.3, 92.4, 112.1, 129.1, 138.6, 155.7, 167.5, 212.8, 216.8, 241.1, 259.1, 280.2, 296.0, 314.4, 317.2, 345.2, 349.7. Eleven of the twelve 30° bins have one or two accepts. The bin 180–210° has none. Speeds run from 5.48 to 9.44 m/s.

Entry-state tags on the 21 accepts: unilateral 17, contact migration 13, tangential slip 6, rolling 3, complete grasp loss 2, airborne 1.

Snapshot files match the 21 accepted ids. The largest restore qpos error is 5.07e-4, under the 1e-3 tolerance. Contact force in the log is the MuJoCo normal force in newtons.

Batches: `results/diagnostics/raw/ball_impact_dataset_v2/batches/b0001.md` and `b0002.md`.

Combined TRAIN initial conditions are 49. That is not a 300-episode training distribution. VALIDATION and FINAL_TEST were not expanded.

TARGETED BALL-IMPACT GENERATION: PARTIAL

POLICY TRAINING HAS NOT BEEN STARTED
