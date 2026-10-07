# Current recovery mainline

Nominal grasp and lift, then a 7D recovery policy. Physical success is unchanged nominal continuation that is still holding the cylinder at absolute t = 12.

```
physical recovery environment
  -> three established recovery mechanisms
  -> 7D multimodal recovery policy pretraining
  -> future oracle-based handoff learning
  -> later RL fine-tuning
```

## Architecture

| piece | where |
| --- | --- |
| nominal grasp / lift | `envs/grasp_sim.py`, `controllers/nominal.py`, `training/demo_ballistic_impact.py` |
| 7D action | `training/recovery_runtime.py` `map7` / `step7` |
| snapshot restore | `restore_dyn` |
| legal observation | `envs/observable_obs.py` (27-D) |
| history | 4 past observations, 4 past actions, 4 valid bits (167-D) in `training/recovery_policy_pretraining_preparation.py` |
| t = 12 oracle | `continue_long` + `physical_of` |
| actor pretraining | family-balanced behavior cloning in the preparation script |
| handoff head | `HandoffHead` on the same 167-D history; `fit` raises |

Action `a` in `[-1, 1]^7`: hand-frame velocity (0.08 m/s), `r_des`-frame angular rate (4 rad/s), grip (`-1` → +2 N·m, `+1` → −18 N·m). `recovery4d` cannot command the support rotation `wx`. The wrist reseat and the airborne recatch embed into this 7D map. The gravity-assisted slide uses `wx = 0.28` and `wy = +0.50` together, then grip −0.48.

Mass and friction for the preparation grid are nominal, 0.16 kg, and the held-out cell 0.18 kg / 0.55. They are not policy inputs.

## Established mechanisms

All three are recipes in `schedules()` and run through `step7`.

| mechanism | recipe |
| --- | --- |
| wrist micro-reseating | embedded `[0.8, 0.6, 0, 1]` for 12 steps, then a secure hold |
| wrist airborne recatch | 19 secure wrist steps, one 20 ms open, 12-step reclose |
| compound-orientation gravity-assisted sliding | `wx=0.28`, `wy=+0.50` for 20 steps, grip −0.48 for 24, retighten 8 |

The canonical impact is 6.5 m/s with axial offset −6 mm. On that snapshot the preparation recorded retention to t = 12 for the reseat, the recatch, and the inward slide.

## Training sequence

| stage | status |
| --- | --- |
| A. Recovery-actor pretraining from demonstrations | code and dataset ready; the first multimodal BC pilot does not yet execute more than one skill in closed loop |
| B. Oracle-labelled handoff training with the actor frozen | not started |
| C. Joint off-policy fine-tuning | not started |

NO CLASSIFIER

NO SAC TRAINING YET

HANDOFF HEAD NOT TRAINED

## Retained

Source:

- `envs/` except the removed 4D gym environment and the old proxy reward
- `controllers/` except the removed heuristic recovery
- `sensors/`
- `config/nominal.yaml`, `config/sim.yaml`, `config/rule_based_recovery.yaml`
- `training/demo_ballistic_impact.py`, `impact_ball.py`, `replay_core.py`, `write_ballistic_impact_scene.py`, `write_impact_scene.py`
- `training/recovery_runtime.py`, `recovery_policy_pretraining_preparation.py`, `test_current_mainline.py`
- `assets/`

Reports:

- `CURRENT_RECOVERY4D_ACTION_AUTHORITY_AUDIT.md`
- `LEGACY_RECATCh_REPRODUCTION_AUDIT.md`
- `COMPOUND_ORIENTATION_GRAVITY_ROLLING.md`
- `COMPOUND_ORIENTATION_GRAVITY_ROLLING_VISUAL_AUDIT.md`
- `RECOVERY_POLICY_PRETRAINING_PREPARATION.md`

Raw evidence:

- `results/diagnostics/raw/recovery_policy_pretraining_preparation/` (action audit, imitation steps, summary)
- `comparison_inward_outward_secure.mp4`
- `inward_trace.npz`, `outward_trace.npz`, `secure_trace.npz`, `mechanism.json`

The t = 12 rule lives in `physical_of`: drop, escape, or else bilateral contact with object height above 0.51 m. The old XY < 0.08 box is not that label.

## Deleted

Removed rather than archived:

- classifier, viability, discriminant, and classifier-guided termination code, reports, and outputs
- old region-C, Phi, proxy, and short-horizon success paths, including `envs/task_reward.py` and `envs/recovery_env.py`
- failed SAC runs, termination-collapse pilots, persistent-behavior pilots, and the 4D training entry points
- search bulk: multimode expansion, targeted mechanism search, unilateral support, Cartesian feasibility grids, failed rolling variants, and their raw trees
- diagnostic videos other than the inward/outward/secure comparison
- one-off audit scripts under `training/` that the runtime no longer imports
- `evaluation/evaluate_recovery.py` (it called the deleted 4D environment)
- unused configs `randomized.yaml`, `no_contact.yaml`, `impact_demo.yaml`

`map_recovery4d` remains in `controllers/residual.py` so the wrist embedding can be checked against the old 4D map. Training rollouts do not call it.

## Smoke test

`python training/test_current_mainline.py` checks nominal impact capture, snapshot restore, one 7D step, the 167-D history, the three mechanism replays, nominal continuation of the wrist reseat to t = 12, the imitation dataset, and instantiation of the actor and the untrained handoff head.

AGGRESSIVE CLEANUP COMPLETE — CURRENT MAINLINE VERIFIED
