# CS5478 recovery

MuJoCo Panda grasp recovery. Physical success is nominal continuation that still holds the cylinder at absolute t = 12. `physical_of` returns drop, escape, or else bilateral contact with object height above 0.51 m. The published success string is `RETAINED_TO_12`.

The policy is a causal Transformer. It reads the 45-D `observe_airborne` state and commands `recovery7d_airborne_v3` in `controllers/recovery_actions.py`. Tokens are K = 32 prototypes, shape `(32, 7)`. Training uses `torch.manual_seed(1)` inside `train_equal`, 120 epochs, 80 steps per epoch, batch 16, and equal weight across the families that are present. There is no viability classifier. The handoff head is not this policy.

## Evaluate the saved policy

From the repository root:

1. Load `results/diagnostics/raw/balanced_unified_recovery/ckpt.pt`.
2. Build the model with `training.natural_loss_recatch_transformer_v3.build_model` on that checkpoint's prototypes, then load `model`.
3. Build the 24 cases the same way as `main()` in `training/balanced_unified_recovery.py`: seated grasp, impact snapshots, schedule suffixes, airborne carries, and the held-out natural-loss windows.
4. Score each case with `eval_case`, then `continue_long` to absolute t = 12.

`python training/balanced_unified_recovery.py` retrains and overwrites both `ckpt.pt` and `results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`. Load the checkpoint when the goal is to reproduce the published evaluation.

| role | path |
| --- | --- |
| Balanced policy | `results/diagnostics/raw/balanced_unified_recovery/ckpt.pt` |
| Published 24-case log | `results/diagnostics/raw/balanced_unified_recovery/summary.json` |
| Seed-1 baseline used in that comparison | `results/diagnostics/raw/natural_loss_controlled_dagger/ckpt_round0.pt` |
| v3 checkpoint named by the v3 training report | `results/diagnostics/raw/natural_loss_recatch_transformer_v3/ckpt_120.pt` |

The balanced script reads `ckpt_round0.pt` for the baseline weights and the prototypes. The balanced checkpoint stores the same K = 32 prototypes.

## Training trajectories

`main()` regenerates the demonstrations. It does not read a trajectory pickle.

- Wrist, sliding, and open-finger recatch come from `schedules()` in `training/recovery_policy_pretraining_preparation.py`: `wrist_reseat`, `gravity_inward`, and `recatch_open20`. They are recorded with `record_local` on the canonical impact, 6.5 m/s with axial offset −6 mm. Recorded lengths are 18, 52, and 32.
- Natural-loss training uses the six core offsets in `TRAIN_NATURAL`: `dz` −13.2, −13.5, −14.2, −14.6, and `dvz` −0.01 / +0.005 on −13.5. The published run stored 18 windows. Collection is `training.natural_loss_recatch_data_scaling.collect`.
- The seated vertical release and the sideways carry did not yield a verified v3 recatch, so airborne carry demonstrations are not in the training set.

Held-out natural-loss evaluation windows, not training targets, are `EVAL_NATURAL`: `h_dz14.00`, `h_dz14.25`, `h_dy` (`dz` −14.10, `dy` 0.12), and `h_dx` (`dz` −14.70, `dx` 0.10).

## 24-case benchmark

The suite is built inside `main()`, not saved as its own pickle. Outcomes are in `results/diagnostics/raw/balanced_unified_recovery/summary.json` and `results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`.

Four cases each: impact, sliding, wrist, open-finger recatch, airborne carry, and held-out natural-loss.

## Three demonstration schedules

The recipes live in `schedules()` and are the demonstrations the balanced policy replays. The canonical impact above is the snapshot on which reseat, recatch, and inward slide were recorded through to t = 12.

| mechanism | recipe |
| --- | --- |
| wrist micro-reseating | embedded `[0.8, 0.6, 0, 1]` for 12 steps, then a secure hold |
| open-finger recatch | 19 secure wrist steps, one 20 ms open, 12-step reclose |
| gravity-assisted sliding | `wx=0.28`, `wy=+0.50` for 20 steps, grip −0.48 for 24, retighten 8 |

Grip in these recipes uses −1 for +2 N·m (open) and +1 for −18 N·m (secure). `python training/test_current_mainline.py` checks nominal impact capture, snapshot restore, one 7D step, the preparation history, the three mechanism replays, and nominal continuation of the wrist reseat to t = 12.

## Where the pieces live

| piece | where |
| --- | --- |
| nominal grasp / lift | `envs/grasp_sim.py`, `controllers/nominal.py`, `training/demo_ballistic_impact.py` |
| 45-D airborne observation | `envs/airborne_obs.py` |
| v3 action map | `controllers/recovery_actions.py` |
| t = 12 continuation | `continue_long` and `physical_of` in `training/recovery_runtime.py` |
| snapshot restore | `restore_dyn` |
| balanced training and the 24-case suite | `training/balanced_unified_recovery.py` |
| scene and gains | `config/nominal.yaml`, `config/sim.yaml`, `config/rule_based_recovery.yaml` |

`map_recovery4d` remains in `controllers/residual.py` so the wrist embedding can be checked against the older 4D map. Training rollouts do not call it.

## Diagnostics worth keeping

- `results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`
- `results/diagnostics/VERTICAL_VELOCITY_CAP_THRESHOLD.md` and `results/diagnostics/UNCAPPED_CARTESIAN_AIRBORNE_CHASE.md` (downward-velocity authority; the chase reloads `results/diagnostics/raw/uncapped_cartesian_airborne_chase/natural_loss_13_5.npz`)
- `results/diagnostics/UNCONSTRAINED_AIRBORNE_RECATCh_EXISTENCE.md` with `results/diagnostics/raw/unconstrained_airborne_recatch/`
- `results/diagnostics/RECOVERY_POLICY_PRETRAINING_PREPARATION.md` with `results/diagnostics/raw/recovery_policy_pretraining_preparation/`
- `results/diagnostics/NATURAL_LOSS_RECATCh_TRANSFORMER_V3.md` and `results/diagnostics/CLEAN_NATURAL_LOSS_RECATCh_TRAINING.md`
- `results/diagnostics/NATURAL_LOSS_RECATCh_DATA_SCALING.md` and `results/diagnostics/NATURAL_LOSS_RECATCh_CONTROLLED_DAGGER.md`
- `results/diagnostics/FULL3D_RECATCh_BASIN_AUDIT.md`, `results/diagnostics/RECATCh_PERTURBATION_CAUSAL_AUDIT.md`, and `results/diagnostics/CONTACTED_VELOCITY_MATCHING_RECOVERY.md`

Sliding comparison video and traces: `results/diagnostics/raw/compound_orientation_gravity_rolling_visual_audit/` (`comparison_inward_outward_secure.mp4`, `inward_trace.npz`, `outward_trace.npz`, `secure_trace.npz`, `mechanism.json`).
