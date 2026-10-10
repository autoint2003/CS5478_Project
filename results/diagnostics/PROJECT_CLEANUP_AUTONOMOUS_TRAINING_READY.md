# Autonomous control flow frozen

The online path is nominal control, a geometry entry, the frozen 39-D policy, a stable-grasp exit, and nominal continuation to absolute t = 12. The 39-D network, the observation, and `recovery7d_airborne_v3` were not changed. Nothing was trained. The 19/24 geometry result remains the pre-training autonomous baseline, not a trained score.

## Frozen entry

The reference is the object pose in the hand frame at the start of the stable lift. Entry is position only:

- hand-frame position error above 0.006 m
- held for 0.040 s, which is three control samples at 0.020 s

Orientation is not an entry signal. `DeteriorationMeter.update_mode` is not called on this path. The meter object remains on the simulator because snapshots still store its fields, and `body_twist` is the rigid-body velocity helper. `run_to_lift_or_deterioration` has been removed.

## Frozen exit

Nominal resumes when all of these hold for 0.100 s, five control samples:

- both finger pads in contact
- hand-frame position error at most 0.0147 m
- orientation error at most 0.35 rad
- object-hand relative speed at most 0.08 m/s

The exit region is wider than the entry line because retained handoffs settle about 12 mm and up to 0.30 rad from the lift pose. The handoff is one-way for the rest of the episode. The task clock is `sim.data.time`. Entry, exit, and the return to nominal do not reset it.

Both rules are in `config/geometry_switch.yaml`. `controllers/geometry_switch.py` is the only reader.

## Active pipeline

`training/autonomous_episode.py` builds the episode: nominal lift, geometry switch, 39-D policy on the live state, stable exit, then `continue_long` to absolute t = 12. `make_training_env` is the constructor for the next training stage. Weight training, when it is started, remains `training/final_39d_balanced_policy.py` and `train_equal` in `training/balanced_unified_recovery.py`.

Checkpoint: `results/diagnostics/raw/balanced_unified_recovery/ckpt_39d.pt`. Input layer `(64, 46)`. The archived 45-D weights stay at `ckpt.pt` because the 39-D trainer reads its K=32 prototypes from that file and the 45-D report still cites it.

## Smoke test after cleanup

| Check | Result |
| --- | --- |
| `impact_zm6` entry | 2.920 s, position, 68 ms after contact at 2.852 s |
| state at entry | qpos and the clock were not reset; the first policy step advanced 0.020 s |
| exit | 3.280 s, bilateral grasp, 12.5 mm, 0.289 rad, 0.013 m/s |
| nominal continuation | retained through t = 12.000 s |
| `slide_dy1` | same entry, exit, and retention; the 6 mm line is crossed before the 1 mm shift |
| wrist and open-finger | the same `impact_zm6` episode; those benchmark rows are not a second disturbance |
| undisturbed lift | max position error 0.15 mm, max orientation error 0.009 rad, no entry, clock reaches 12 s |

A separate process imported `final_39d_balanced_policy`, `train_equal`, and `test_current_mainline`, constructed the environment, and read a 39-D observation from `ckpt_39d.pt`. Training was not started.

## Retained reports

- `BALANCED_UNIFIED_RECOVERY_EVALUATION.md`
- `FINAL_39D_BALANCED_POLICY.md`
- `OBSERVATION_REALIZABILITY_AUDIT.md`
- `OBSERVATION_ROBUSTNESS_CHECK.md`
- `OBSERVATION_ABLATION_CHECK.md`
- `GEOMETRY_BASED_RECOVERY_SWITCH.md`
- `AUTONOMOUS_RECOVERY_INTEGRATION.md` (the deterioration-score result this switch replaced)
- `RECOVERY_ENTRY_EXIT_HANDOFF_AUDIT.md`
- `RECOVERY_POLICY_PRETRAINING_PREPARATION.md`

Their raw directories are still present: `balanced_unified_recovery`, `balanced_39d`, `observation_robustness`, `observation_ablation`, `geometry_recovery_switch`, `autonomous_recovery`, `recovery_handoff_audit`, and `recovery_policy_pretraining_preparation`.

## Deleted

Reports and raw outputs from DAgger, natural-loss scaling, the 120-epoch natural-loss checkpoint, contacted velocity matching, the finite full-3D search, perturbation sweeps, the uncapped chase, the unconstrained-chase videos, the vertical-cap writeup, and the sliding comparison video. Also `training/autonomous_recovery_integration.py` and the `__pycache__` directories. No archive tree was created.

## Kept on purpose, not on the online path

These modules still supply a function the 39-D trainer imports (`world`, `kmeans`, `jsonable`, `target_hand`, the carry and natural-loss collectors). Their experiment reports and checkpoints are gone, and they are not the recovery trigger:

- `training/full_3d_airborne_recatch_dataset.py`
- `training/natural_loss_recatch_data_scaling.py`
- `training/clean_natural_loss_recatch.py`
- `training/natural_loss_recatch_coverage_expansion.py`
- `training/horizontal_grasp_offset_freefall_recatch.py`
- `training/phase_decoupled_recovery_pilot.py`
- `training/long_context_recovery_pilot.py`
- `training/single_step_action_token_policy.py`
- `training/uncapped_cartesian_airborne_chase.py`
- `training/multi_disturbance_airborne_feasibility.py`
- `training/external_airborne_recatch.py`

`training/geometry_recovery_switch.py` is the closed study script. Its `main` exits so it cannot retune the thresholds. `controllers/rule_based_recovery.py` stays because replay verification hashes that file. `training/test_current_mainline.py` remains the skill regression suite.

## Git

The cleanup is not committed. Tracked deletions are the ten exploratory reports above. Tracked edits include `README.md`, `config/nominal.yaml`, and `envs/grasp_sim.py`. The geometry config, the switch, and `training/autonomous_episode.py` are new. Checkpoints and raw logs are gitignored.

AUTONOMOUS CONTROL FLOW FROZEN — PROJECT READY FOR TRAINING
