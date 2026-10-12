# Privileged recatch demonstration generation

The learned policy was not changed. Every saved command is a `recovery7d_airborne_v3` action executed by `step_v3`, with actuator limits left on. The search could read object pose, twist, contacts, and finger state. None of those quantities were added to the 39-D observation. Rollouts were not teleported, and qpos was not edited after a rollout started. A file was written only when the continuation returned `RETAINED_TO_12`. No training step was run.

The oracle is a privileged state-feedback law. It chooses when to open, chases the cylinder in world velocity using vx, vy, and the asymmetric vz map, and can add a wrist rate. Once the hand-frame grasp error enters the close tolerance, or bilateral contact returns, the grip latches shut and the velocity command brakes instead of following the fall. Parameters were a fixed grid, then CEM only if that grid found no retained recatch. The grid's first law was enough for every natural-loss entry.

## Where the search started

Each case was run to its own geometry entry. The entry keeps the live qpos, qvel, contacts, `p_des` / `r_des`, the task clock, and a zero previous action. The pretrained policy was then rolled on that same capture. Search ran only when that roll dropped.

| Case | Split | Policy | Oracle |
| --- | --- | --- | --- |
| `train_dz13.5` | train | drop, 5 steps | retained |
| `nl_dz14.6` | validation | drop, 5 steps | retained |
| `nl_dvz_p` | validation | drop, 5 steps | retained |
| `h_dz14.00` | test | drop, 5 steps | retained |
| `h_dz14.25` | test | drop, 5 steps | retained |
| `h_dy` | test | drop, 5 steps | retained |
| `h_dx` | test | drop, 129 steps | retained |
| `impact_zp6` | test | drop, 5 steps | retained |
| air 0.60 m/s, +y | probe | retained, 5 steps | not searched |

The frozen airborne validation and test carries were already retained by the policy. The extra 0.60 m/s carry also retained, so this pass had no failed airborne entry to search.

## What the natural-loss recatches do

On every offset entry the same law retains. The policy's failure on these entries is a 5-step geometry handoff from a grasp that is still inside the 14.7 mm exit box and then falls under nominal control. The oracle opens immediately, loses both pads, drives the hand-frame error from about 14 mm to about 3 mm, recloses, and only then satisfies the stable-grasp exit.

Recorded on `train_dz13.5`, and the same shape on the other offsets:

| | |
| --- | --- |
| Steps | 8 (0.16 s of recovery) |
| Contact | bilateral, 40 ms with neither pad, bilateral again |
| Airborne time | 0.04 s |
| Minimum grasp error | 3.2 mm |
| Peak command | 0.10, 0.02, 2.16 m/s |
| Peak achieved hand speed | 0.14, 0.01, 0.76 m/s |
| Peak wrist command | 0.09 rad/s |
| Grip | full open, then secure |
| Torque saturation | 0 |
| Exit | geometry, then nominal to t = 12 |
| Outcome | `RETAINED_TO_12` |

The observation stored with the actions is 39-D, with the previous 7-D command. Actions lie in `[-1, 1]`. Lateral speed stays under 0.15 m/s, so these are labeled **existing recatch-like recovery**: an open, short chase, and reclose, not a wrist reseat and not a closed-finger slide. They are new state-action sequences for entries the current policy drops.

Eight perturbations of the entry were replayed, both with the same feedback law and with the recorded action sequence alone: ±2 mm lateral, ±2 mm vertical, ±0.03 m/s object velocity, ±0.4 rad/s object angular velocity. All 8 closed-loop rolls and all 8 open-loop replays retained. That is a locally robust basin. The demonstration the learner would imitate already survives those perturbations without the oracle in the loop.

## Impact zp6

The first two grid laws dropped. The third retained. It opens, loses contact for 40 ms, and recloses. The chase command reaches 1.56 m/s along y and 0.88 m/s vertically; the arm achieves about 0.50 m/s laterally. That lateral chase is why it is labeled **genuinely new recatch mechanism** rather than the vertical-intercept family.

This retention did not come from a stable-grasp exit. Recovery ran the 300-step cap (bilateral for 297 steps after recontact) and the nominal continuation still returned `RETAINED_TO_12`. Grasp error bottomed at 9.7 mm. Torque saturation was 0. Closed-loop perturbations retained 6/8. The two misses were −2 mm vertical and +0.03 m/s vertical velocity. The recorded action sequence retained all 8, so the saved trajectory itself is locally robust even though the feedback law is narrower.

## Learnable set

Existing skill files, all present, are unchanged:

- `demo_wrist_reseat.npz`
- `demo_gravity_inward.npz`
- `demo_recatch_open20.npz`

New files are under `results/diagnostics/raw/privileged_recatch/`.

Trainable pool, for a later family-balanced update, not used now:

| File | Why it is included |
| --- | --- |
| `demo_train_dz13.5.npz` | train disturbance, basin C, stable exit |
| `demo_nl_dz14.6.npz` | validation failure, basin C, stable exit |
| `demo_nl_dvz_p.npz` | validation failure, basin C, stable exit |

Held-out successes are saved and are not in that pool: `demo_h_dz14.00.npz`, `demo_h_dz14.25.npz`, `demo_h_dy.npz`, `demo_h_dx.npz`, and `demo_impact_zp6.npz`. They answer the existence question for the test entries. A later trainer can leave them out so the frozen test split stays untouched.

The natural-loss demonstrations are short enough to imitate: eight 39-D observations, eight legal actions, and a physical t = 12 label. The impact demonstration is a 300-step lateral recatch with the same interface, without a geometry exit.

PRIVILEGED CONTROL FOUND NEW LEGAL RECATCh DEMONSTRATIONS
