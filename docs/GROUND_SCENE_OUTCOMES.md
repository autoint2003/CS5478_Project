# GROUND_SCENE outcome definition (authoritative)

Scene identifier: **`GROUND_SCENE`**  
Irreversible airborne Recovery failure: **`GROUND_DROP`**

Elevated-table evaluation remains available separately (`TABLE_DROP`, `assets/scene.xml` / ballistic table scene). Do not merge counts across scenes.

## Geometry

| Quantity | Value |
|---|---|
| Ground plane | MuJoCo `plane` geom `ground` through world origin |
| `GROUND_Z` | `0.0` m |
| Panda `link0` | at world origin (unchanged) |
| Cylinder half-height | `0.03` m (unchanged) |
| Resting COM on ground | `0.03` m |
| Geometric guard `GROUND_DROP_Z` | `0.04` m (rest + 1 cm). **Not** the old table gate `0.44` |

Usable vertical workspace for airborne Recovery is approximately from grasp height down to the ground plane (~0.4 m more free-fall than the elevated table when starting from the historical ~0.43–0.55 m grasp band).

## GROUND_DROP latch

1. **Arming:** `envs.outcomes.ground_drop.arm(sim)` at Recovery start. Pre-Recovery ground support while disarmed is **not** a failure (cylinder may rest on the ground during setup).
2. **Trigger:** real contact impulse / sustained support against geom `ground`, same bounce/support timing as the historical table latch (`SUPPORT_TIME_S=0.02`, `BOUNCE_DV=0.15`).
3. **Suppression:** bilateral finger grasp blocks latching; a later regrasp does **not** clear a latched event.
4. **Irreversibility:** once latched, terminal reason is `GROUND_DROP`; later “recovery” cannot count as success.
5. **Secondary height guard:** while armed, COM below `GROUND_DROP_Z` may abort via geometric checks; contact latch remains primary.

## Success (unchanged substantive objective)

Hand-frame targets (working SUCCESS_NEW):

- `p_COM_hand ≈ (0, 0, 0.105)` m (calibration uses `evaluation/full_objective.py` `P_TGT`)
- Angular ≤ 10°, position ≤ 12 mm, hold ≥ 0.5 s
- Legal recatch, stable grasp, legal HANDOFF, unchanged nominal continuation

Do not weaken these tolerances for GROUND_SCENE.

## Observation note

The 39D observation still uses the historical hand-z reference constant `TABLE_TOP = 0.40` as a **calibration offset**, not as a physical table in GROUND_SCENE. Drop physics and success height guards use ground-relative quantities above.

## Modules

| Role | Path |
|---|---|
| Scene XML writer | `envs/simulation/ground_scene.py` |
| Simulator | `envs/simulation/ground_sim.py` |
| Latch | `envs/outcomes/ground_drop.py` |
| Facade | `envs/outcomes/support_drop.py` |
| Height gate | `envs/outcomes/drop_thresholds.py` |
| Validation | `scripts/run_ground_scene_validation.py` |
