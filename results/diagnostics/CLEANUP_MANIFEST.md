# Cleanup manifest

Hard cleanup before formal GRPO. Superseded artifacts were deleted. Nothing was moved into an archive. Formal training was not started.

81 files removed, 104,901,642 bytes (100.0 MB).

## Why the retained evidence is enough

Each kept report states the design decision, and the compact JSON or npz next to it holds the table or the one canonical trace. Model weights were kept only when the launcher loads them or when the 40-update sanity conclusion needs the final policy. Intermediate weights are recoverable as numbers from those logs, not as checkpoints.

## Active files kept

The launch command is `python -u training/formal_grpo.py --config config/formal_grpo.yaml`.

- `training/formal_grpo.py`, `training/pre_grpo_sanity.py` (mean recovery-step GRPO loss), `config/formal_grpo.yaml`
- `training/recovery_success.py`, `config/recovery_success.yaml`
- `controllers/geometry_switch.py`, `config/geometry_switch.yaml`
- `controllers/recovery_actions.py` (`recovery7d_airborne_v3`)
- `envs/airborne_obs.py` (39-D observation)
- `training/final_rl_readiness.py` (train / validation / final-test catalog). Its `CKPT` constant now points at policy A.
- The rest of the import closure of `training.formal_grpo`, including the environment, nominal controller, and simulator configs `config/nominal.yaml`, `config/sim.yaml`, and `config/rule_based_recovery.yaml`

## Checkpoints kept

| file | role |
|---|---|
| `results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt` | K=35 old-skill reference. sha256 `f0beb2d8…181c7` |
| `results/diagnostics/raw/pre_grpo_sanity/policy_B_init.pt` | K=36 formal initialization. sha256 `904d183b…23bd9` |
| `results/diagnostics/raw/pre_grpo_sanity/policy_C_weak_u40.pt` | Policy after the 40-update sanity run |

`sanity.json` and `results/diagnostics/raw/grpo_norm_check/check.json` keep the logged updates, advantages, and the length-normalization gradient check. Update-0, 10, and 20 weights were deleted; those closed-loop counts are in `sanity.json` and `FINAL_PRE_GRPO_SANITY.md`.

## Evidence kept

Reports: observation realizability, the 39-D ablation, the verified 39-D policy, geometry entry, the nominal-basin success audit, privileged recatch demos, the supervised recatch retrain, the lateral-chase token, privileged and learned nominal restoration, readiness and recoverability coverage, supervised warm-start interference, continual skill interference, future-state ablation, the critic-ranking audit, SAC/DPO/GRPO selection, the pre-GRPO sanity check, and this launch preparation.

Canonical traces: wrist reseat, inward slide, open-finger recatch, `train_dz13.5`, `nl_dz14.6`, `nl_dvz_p`, `impact_zp6`, the nominal restoration trajectory, `success_air_a`, `success_air_b`, `success_imp_c`.

Frozen episode captures under `results/diagnostics/raw/final_rl_readiness/ics/`, including final test. `imp_f` and `imp_h` remain the no-entry cases and have no capture.

One video: `results/videos/impact_zp6_learned_restoration.mp4`.

Compact logs kept beside the reports above: `ablation.json`, `representation.json`, `algorithm_selection.json`, `continual.json`, `warmstart.json`, `train.json`, `retrain.json`, `audit.json`, `variants.json`, `summary.json` for geometry and the 39-D run, `readiness.json`, `expansion.json`, `chase.json`.

## Deleted

| category | files | bytes |
|---|---:|---:|
| Archive trees `results/archive/pre_final_training/` and `results/archive/pre_formal_grpo/` | 11 | 3,607,213 |
| Superseded raw weights, search dumps, held-out demos, critic logs | 38 | 8,556,731 |
| Duplicate K=33 impact videos, side view, and contact trace | 4 | 92,379,934 |
| One-shot experiment scripts not imported by formal GRPO | 16 | 256,936 |
| Transient or overturned reports | 11 | 95,328 |
| This cleanup's own scratch script | 1 | 5,500 |

Deleted experiment categories: future-state checkpoints other than the selected initialization, Stage A/B weights, the failed supervised warm-start weights, K=32/K=33/K=35 duplicate checkpoints, SAC actor-critic checkpoints, critic-only and history-critic logs, old algorithm-selection weights, `RETAINED_TO_12` video dumps, privileged search dumps, and the held-out `h_*` demonstrations.

Deleted reports: critic-only audit, history-aware critic ablation, inconclusive RL sanity, old dataset-ready note, balanced-unified evaluation, entry-integration bottleneck, handoff-issue audit, impact video audit, observation-robustness check, pretraining-preparation note, and the earlier "ready for training" status.

Empty archive directories were removed.
