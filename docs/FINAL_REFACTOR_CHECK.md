# Final refactor check

This is the last architecture pass. Physics, the 7D legal action, the K48 codebook and residual, the 39D observation, the nominal controller, the reward formulas, the SFT and GRPO definitions, `SUCCESS_12S`, and `TABLE_DROP` were not edited. No demonstration was regenerated and no training run was started.

The rollback commit remains `85a117b`. Round 3 and this pass are uncommitted work on that commit.

## 1. Files changed and why

| Change | Why |
|---|---|
| `models/airborne_v3.py` | Keeps the 39D load, `build_model`, `step_v3`, and `intercept_action`. |
| `experiments/legacy/airborne_v3_demos.py` | Historical natural-loss demo generator. Only experiment scripts call it. |
| `envs/simulation/scene_state.py` | `TABLE_TOP`, ball radius, and ball parking. |
| `envs/contacts/scene_contacts.py` | Scene and object contact rows used by replay and the privileged pack. |
| `envs/observations/privileged_obs.py` | `obs_from_sim`. |
| `controllers/cartesian/velocity_step.py` | `tick_vw` and `freeze`, shared by RULE replay and the nominal hold. |
| `envs/simulation/rule_replay.py` | Frozen RULE replay only. |
| `training/k47_codebook.py` | K47 codebook, `decode_action`, `group_advantage`, and `categorical_kl`. |
| `experiments/legacy/k47_training.py` | Historical K47 supervised and GRPO loop. |
| `training/grpo_progress.py` | Renamed from `training/trainer.py`. Progress and timing only. |
| `tests/test_physics_regression.py` | Separates deterministic replay from a saved state baseline. |
| `tests/fixtures/physics_state_baseline.npz` | State tape for later edits. |
| `tests/fixtures/physics_state_baseline.json` | Provenance of that tape. |
| `tests/test_reward_v2.py`, `tests/test_table_drop.py` | Write their reports to a temporary directory. |

`training/unified_k47.py` was removed after the split. Callers import `training.k47_codebook` or `experiments.legacy.k47_training`. There is no compatibility shim.

## 2. K47, K48, and the 39D model

`training/k47_codebook.py` loads the 47 by 7 motion table and decodes a token plus a residual clipped to ±0.25. Token 47 is outside that table and raises. K48 is a different head: `models/k48_policy.py` and `models/k48_checkpoint.py` use 48 logits, and logit 47 is HANDOFF. The two tables are not merged.

`models/airborne_v3.py` is the separate 39D causal policy. `load_airborne_policy` still refuses a checkpoint whose first layer has a different observation width. `step_v3` still rebases the hand setpoint and holds the mapped velocity for 10 physics steps.

## 3. Frozen RULE baseline

`verify_frozen_hashes()` hashes only:

- `config/controllers/rule_based_recovery.yaml` = `6a0b8103e5b9d8c4a4a4e6634aca43adb0b37bd39242d39269f27785a4d574b8`
- `controllers/recovery/rule_based.py` = `778bd97db7ca19f64192d3d833abd05df77129a223a9683076e1a993701e858c`

Both values were recorded before this pass and matched again after it. Neither file was edited, so the pin did not need a new hash or a migration map. Moving helpers out of `rule_replay.py` changes import paths, not the hashed control law.

## 4. Deterministic replay and the independent baseline

Deterministic replay runs the same snapshot and the same action tape twice. It compares qpos, qvel, the 39D observation, contact geom, normal, and force, controller phase and setpoints, and, for the vertical tape, the nominal outcome. That shows the current process repeats itself.

The fixture is a separate comparison. The saved demonstration npz stores actions and 39D observations. It does not store qvel or contact frames, and no pre-refactor full-state tape was in the repository. The fixture is therefore the state of this verified tree, not a reconstruction of the code before the refactor.

## 5. Baseline location and use

Files:

- `tests/fixtures/physics_state_baseline.npz` (about 164 KB)
- `tests/fixtures/physics_state_baseline.json`

Recorded with the tape: git HEAD `85a117b8a467612d4e54456c0a578975307a425c`, dirty working tree, MuJoCo 3.12.0, `config/simulation/nominal.yaml`, `config/simulation/sim.yaml`, the snapshot and action-tape paths, controller phase and setpoints, generation time `2026-10-15T14:46:32Z`, state tolerance `1e-8`, and contact force rounded to 6 decimals. Contact rows are geom1, geom2, Fn, side, and `n_w`.

Cases:

| Key | Episode | Strategy |
|---|---|---|
| `controlled_sliding` | `ball_train_v2_0070` | `controlled_slide` |
| `inward_sliding` | `ball_train_0017` | `inward_reseat` |
| `horizontal_recatch` | `ref_s6_p10_a0_e0` | first manifest row |
| `vertical_recatch` | `ball_train_v2_0070` | `follow_early` |

`python -u tests/test_physics_regression.py` loads the fixture. `WRITE_PHYSICS_BASELINE=1` rewrites it and must not be used to hide a physics change. Nominal failure and `TABLE_DROP` stay as separate checks: an unrestored vertical capture is `LOST_GRASP`, and the latch cases do not depend on the state tape.

## 6. Tests that ran

| Command | Result |
|---|---|
| `python -u tests/test_physics_regression.py` | `PHYSICS_REGRESSION_OK`, including the four baseline tapes |
| `python -u tests/test_reward_v2.py` | Printed the reward cases. Report went to a temp directory. |
| `python -u tests/test_table_drop.py` | `TABLE_DROP_TESTS_OK`. Report went to a temp directory. |
| `scripts/train.py --help` | Exit 0 |
| `scripts/evaluate.py --help` | Exit 0 |
| `scripts/generate_data.py --help` | Exit 0 |
| `scripts/render_video.py --help` | Exit 0 |
| In-memory K48 smoke on `results/grpo/v2/checkpoint_u0783.pt` | `SMOKE_OK` |

The smoke loaded the checkpoint, collated two supervised rows, computed the SFT loss (`0.009695815853774548`), ran `loss.backward()` and one Adam step, evaluated the GRPO `step_logprob` (`-1.7989915609359741`) and a second Adam step, checked `assert_reward_order` and `grpo_return` for `SUCCESS_12S`, and confirmed token 47 is rejected by `decode_action`. Logits were `(48, 64)`. Nothing was saved.

`verify_frozen_hashes()` matched the hashes in section 3. A scan of `controllers`, `envs`, `models`, `sensors`, `training`, `evaluation`, `data_generation`, `visualization`, and `scripts` found no import of `experiments` or `legacy`.

The archive file `saved_trace_comparison.json` and `results/diagnostics/table_drop_fix/raw/results.json` kept their earlier modification times.

## 7. Tests not run

| Not run | Reason |
|---|---|
| `scripts/train.py --supervised` or `--grpo` | Official training was not allowed. |
| `scripts/generate_data.py` without `--help` | Would search and write demonstrations. |
| `scripts/render_video.py` without `--help` | Would write a video under `results/`. |
| `evaluation/verify_handoff_targets.py` | Full physical replay of every demonstration. The regression already replays the four reference tapes and checks HANDOFF decode. |
| `experiments/legacy/k47_training.py` | Historical loop. It writes checkpoints. |

## 8. Remaining issues that do not affect the formal path

- `experiments/ball_impact/privileged_ball_impact_search.py` defines `step_legal` as a delegate to `envs.step_legal`. Formal code does not call that wrapper.
- `experiments/legacy/demo/demonstrations/demo_ballistic_recovery.py` has an old contact helper. Formal extraction is `envs/contacts/frames.py`.
- `experiments/legacy/demo/build_k48_handoff_policy.py` is an old K48 expansion script. Formal load is `models/k48_checkpoint.py`.
- `data_generation/maintenance/` is not on the generate entry, but older experiment scripts can still call it.
- `results/diagnostics/` was not reorganized.

## Directory

```text
controllers/nominal.py
controllers/cartesian/{jacobian,bias_torque,tracking,velocity_step}.py
controllers/gripper/controller.py
controllers/recovery/{action_map,residual_limits,geometry_switch,rule_based,skill_schedules}.py
envs/{step_legal,ballistic_impact}.py
envs/simulation/{grasp_sim,rule_replay,scene_state,capture,...}.py
envs/observations/{observable_obs,airborne_obs,tactile_reward,privileged_obs}.py
envs/contacts/{touch,frames,scene_contacts}.py
envs/outcomes/{horizon_outcome,recovery_success,table_drop,physical_recovery,nominal_pose}.py
models/{k48_policy,k48_checkpoint,airborne_v3}.py
models/historical/{k36_policy,k36_checkpoint,k33_checkpoint}.py
training/{unified_k48,train_sft,train_grpo,dataset,grpo_progress,k47_codebook,reward_v2}.py
data_generation/rebuild_k48_recovery.py
evaluation/k48_eval.py
visualization/render_video.py
tests/test_physics_regression.py
tests/fixtures/physics_state_baseline.npz
```

## Formal entries

| Job | Entry |
|---|---|
| Data generation | `python -u scripts/generate_data.py` |
| SFT | `python -u scripts/train.py --supervised --config config/training/k48/unified_k48_recovery_v3.yaml` |
| GRPO | `python -u scripts/train.py --grpo --config config/training/k48/unified_k48_recovery_v3_reward_v2.yaml` |
| Closed-loop evaluation | `python -u scripts/evaluate.py --eval --config config/training/k48/unified_k48_recovery_v3.yaml` |
| Video | `python -u scripts/render_video.py` |
| Physics regression | `python -u tests/test_physics_regression.py` |
