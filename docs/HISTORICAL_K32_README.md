# CS5478 recovery

MuJoCo Panda grasp recovery. Physical success is nominal continuation that still holds the cylinder at absolute t = 12. `physical_of` returns drop, escape, or else bilateral contact with object height above 0.51 m. The published success string is `RETAINED_TO_12`.

Online control is nominal until the geometry switch in `config/geometry_switch.yaml`, then the policy, then a stable-grasp exit, then nominal again through absolute t = 12. Entry is a hand-frame position error above 6 mm for 40 ms. Exit is bilateral contact, position error below 14.7 mm, orientation error below 0.35 rad, and relative speed below 0.08 m/s, held for 100 ms. The handoff is one-way. The deterioration score is not the trigger. `training/autonomous_episode.py` is the episode constructor. `training/autonomous_smoke.py` checks one impact, one slide, and an undisturbed grasp.

The policy is a causal Transformer. It reads the 39-D `observe_airborne` state and commands `recovery7d_airborne_v3` in `controllers/recovery_actions.py`. The 39 channels are the 27-D tactile and proprioceptive prefix plus 12 object-relative pose and twist channels. The duplicate wide-scale hand twist that used to occupy indices 39–44 is not in the vector. The previous 7-D command is a separate controller-state input. Tokens are K = 32 prototypes, shape `(32, 7)`. Training uses `torch.manual_seed(1)` inside `train_equal`, 120 epochs, 80 steps per epoch, batch 16, and equal weight across the families that are present. There is no viability classifier. The handoff head is not this policy.

Object position, orientation, and twist in that vector are exact MuJoCo body state, used as a noise-free proxy for an RGB-D tracker. The project does not run a camera or a learned tracker. On the tested moderate sensing profile (2 mm position noise, 1 deg orientation noise, 0.03 m/s linear velocity, 0.3 rad/s angular velocity, 40 ms object-channel delay, 50 ms smoothing, 0.5 mm center-of-pressure noise, and 1 N force noise) the 39-D policy scores 15, 16, and 16 out of 24, so that profile does not reduce the clean 15/24. Center-of-pressure and lateral-estimate noise remain the more sensitive part of the observation: on the 45-D sweep, 0.5–1 mm perturbations there moved impact and one open-finger case. Those tactile channels are unchanged in the 39-D vector.

## Evaluate the saved policy

From the repository root:

1. Load `results/diagnostics/raw/balanced_unified_recovery/ckpt_39d.pt` with `training.natural_loss_recatch_transformer_v3.load_airborne_policy`. That loader refuses a checkpoint whose first layer was built for a different observation width, including the archived 45-D weights.
2. Build the 24 cases the same way as `build_cases` in `training/observation_study_common.py`: seated grasp, impact snapshots, schedule suffixes, airborne carries, and the held-out natural-loss windows.
3. Score each case through `continue_long` to absolute t = 12.

`python training/final_39d_balanced_policy.py` retrains the 39-D policy and writes `results/diagnostics/raw/balanced_39d/summary.json`. `python training/balanced_unified_recovery.py` does not train. It stays closed so it cannot overwrite `ckpt.pt` or `results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`.

| role | path |
| --- | --- |
| Active 39-D policy | `results/diagnostics/raw/balanced_unified_recovery/ckpt_39d.pt` |
| 39-D benchmark and robustness log | `results/diagnostics/raw/balanced_39d/summary.json` |
| Archived 45-D balanced weights | `results/diagnostics/raw/balanced_unified_recovery/ckpt.pt` |
| Archived 45-D 24-case log | `results/diagnostics/raw/balanced_unified_recovery/summary.json` |
| Geometry entry and exit | `config/geometry_switch.yaml` |
| Online episode | `training/autonomous_episode.py` |

The 39-D checkpoint stores the same K = 32 prototypes as the archived 45-D run. Its weights are not a resized copy of those weights.

## Training trajectories

`training/final_39d_balanced_policy.py` regenerates the demonstrations. It does not read a trajectory pickle. `train_equal` in `training/balanced_unified_recovery.py` is the trainer.

- Wrist, sliding, and open-finger recatch come from `schedules()` in `training/recovery_policy_pretraining_preparation.py`: `wrist_reseat`, `gravity_inward`, and `recatch_open20`. They are recorded with `record_local` on the canonical impact, 6.5 m/s with axial offset −6 mm. Recorded lengths are 18, 52, and 32.
- Natural-loss training uses the six core offsets in `TRAIN_NATURAL`: `dz` −13.2, −13.5, −14.2, −14.6, and `dvz` −0.01 / +0.005 on −13.5. The published run stored 18 windows. Collection is `training.natural_loss_recatch_data_scaling.collect`.
- The seated vertical release and the sideways carry did not yield a verified v3 recatch, so airborne carry demonstrations are not in the training set.

Held-out natural-loss evaluation windows, not training targets, are `EVAL_NATURAL`: `h_dz14.00`, `h_dz14.25`, `h_dy` (`dz` −14.10, `dy` 0.12), and `h_dx` (`dz` −14.70, `dx` 0.10).

## 24-case benchmark

The suite is built by `build_cases`, not saved as its own pickle. The active 39-D outcomes are in `results/diagnostics/raw/balanced_39d/summary.json` and `results/diagnostics/FINAL_39D_BALANCED_POLICY.md`. The 45-D comparison remains in `results/diagnostics/raw/balanced_unified_recovery/summary.json` and `results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`.

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
| 39-D airborne observation | `envs/airborne_obs.py` |
| v3 action map | `controllers/recovery_actions.py` |
| geometry entry and stable exit | `config/geometry_switch.yaml`, `controllers/geometry_switch.py` |
| online episode | `training/autonomous_episode.py` |
| t = 12 continuation | `continue_long` and `physical_of` in `training/recovery_runtime.py` |
| snapshot restore | `restore_dyn` |
| balanced training and the 24-case definitions | `training/balanced_unified_recovery.py` |
| 39-D train, benchmark, and sensor check | `training/final_39d_balanced_policy.py` |
| scene and gains | `config/nominal.yaml`, `config/sim.yaml`, `config/rule_based_recovery.yaml` |

`map_recovery4d` remains in `controllers/residual.py` so the wrist embedding can be checked against the older 4D map. Training rollouts do not call it.

## Diagnostics worth keeping

- `results/diagnostics/FINAL_39D_BALANCED_POLICY.md`
- `results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md` (the archived 45-D comparison)
- `results/diagnostics/OBSERVATION_REALIZABILITY_AUDIT.md`, `results/diagnostics/OBSERVATION_ROBUSTNESS_CHECK.md`, and `results/diagnostics/OBSERVATION_ABLATION_CHECK.md`
- `results/diagnostics/GEOMETRY_BASED_RECOVERY_SWITCH.md` (pre-training autonomous baseline, not a trained score)
- `results/diagnostics/AUTONOMOUS_RECOVERY_INTEGRATION.md` (the deterioration-score switch this path replaced)
- `results/diagnostics/RECOVERY_ENTRY_EXIT_HANDOFF_AUDIT.md`
- `results/diagnostics/RECOVERY_POLICY_PRETRAINING_PREPARATION.md` with `results/diagnostics/raw/recovery_policy_pretraining_preparation/`
