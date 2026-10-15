# Module organization

Round 3 grouped code by job. Physics, the 7D action map, the 39D observation, the K48 codebook and residual head, the reward formulas, SFT, GRPO, nominal handoff, `SUCCESS_12S`, and `TABLE_DROP` were not changed. `data/` and saved checkpoints were not rewritten.

The rollback point before this move is commit `85a117b`.

## 1. Directory tree

```text
controllers/
  nominal.py
  cartesian/   jacobian.py  bias_torque.py  tracking.py  velocity_step.py
  gripper/     controller.py
  recovery/    action_map.py  residual_limits.py  geometry_switch.py
               rule_based.py  skill_schedules.py
envs/
  step_legal.py
  ballistic_impact.py
  simulation/  grasp_sim.py  mujoco_ids.py  object_dynamics.py  panda_xml.py
               yaml_config.py  grasp_quality.py  capture.py  support_contacts.py
               impact_ball.py  ballistic_scene.py  impact_diag_scene.py
               rule_replay.py  scene_state.py
  observations/  observable_obs.py  airborne_obs.py  tactile_reward.py  privileged_obs.py
  contacts/    touch.py  frames.py  scene_contacts.py
  outcomes/    horizon_outcome.py  recovery_success.py  table_drop.py
               physical_recovery.py  nominal_pose.py
  initialization/  grasp_seat.py
sensors/       spatial_tactile.py  lateral_offset.py
models/        k48_policy.py  k48_checkpoint.py  airborne_v3.py
  historical/  k36_policy.py  k36_checkpoint.py  k33_checkpoint.py
data_generation/
  generate.py  rebuild_k48_recovery.py  ball_impact_dataset.py
  airborne_recatch_dataset.py  autonomous_episodes.py  policy_rollout.py
  action_kmeans.py  jsonable.py
  initial_conditions/  oracles/  strategies/  recording/  validation/  maintenance/
training/
  unified_k48.py  train_sft.py  train_grpo.py  dataset.py  grpo_progress.py
  reward_v2.py  grpo_math.py  k48_config.py  k47_codebook.py
  recovery_episode.py  recovery_execution.py
evaluation/    k48_eval.py  audit_pretrain_data.py  audit_recatch_legality.py
               verify_handoff_targets.py
visualization/ render_video.py  video_recorder.py  camera.py  overlays.py  impact_view.py
scripts/       train.py  evaluate.py  generate_data.py  render_video.py
config/        simulation/  controllers/  recovery/  training/k48/  training/k47/  training/legacy/
experiments/   ball_impact/  handoff/  sliding/
               recatch/airborne/  recatch/natural_loss/  recatch/legal/  recatch/top_impact/
               legacy/  legacy/demo/{audit,replay,diagnose,demonstrations}/
```

## 2. What each subpackage does

| Package | Job |
|---|---|
| `controllers/cartesian/` | Jacobian Cartesian tracking, MuJoCo bias torque, and the fixed `TRACK` gains. |
| `controllers/gripper/` | Finger torque from the grasp phase. |
| `controllers/recovery/` | How a normalized recovery command becomes a physical 7D action, plus the geometry switch and the frozen rule baseline. |
| `envs/simulation/` | Load the Panda, step MuJoCo, resolve body ids, and snapshot a running episode. |
| `envs/observations/` | The 27D policy observation, the 39D airborne observation, and the tactile reward. |
| `envs/contacts/` | Touch summaries and per-contact frames. They are different jobs and stay separate. |
| `envs/outcomes/` | Handoff, the 12 s nominal hold, the table-drop latch, and pose error. |
| `envs/initialization/` | Build a seated grasp before a disturbance. |
| `models/historical/` | K33 and K36 networks used to reproduce old checkpoints. |
| `data_generation/strategies/` | The distinct recovery laws and their search scripts. |
| `data_generation/oracles/` | Codeword projection (`representability`). Search loops stay in `strategies/`. |
| `data_generation/maintenance/` | One-shot layout and path repair. `generate.py` does not import it. |
| `config/training/k48/` | K48 training YAML. Filenames still name the dataset and reward version. |
| `experiments/recatch/` | Research scripts grouped by airborne chase, natural loss, legal search, and top impact. |

## 3. Important files

| File | Duty |
|---|---|
| `envs/step_legal.py` | One 20 ms legal recovery step. |
| `envs/ballistic_impact.py` | Ballistic world, `make_sim`, `world`. |
| `envs/outcomes/horizon_outcome.py` | Recovery limit, nominal verify, `SUCCESS_12S`. |
| `envs/outcomes/table_drop.py` | Latched table-support termination. |
| `envs/outcomes/physical_recovery.py` | Height gate `TABLE_DROP = 0.44` and the physical pack. This is not the latch. |
| `controllers/recovery/action_map.py` | 7D normalized action to velocity and grip torque. |
| `controllers/recovery/residual_limits.py` | Clip a 7D residual and map the old 4D action. Not the K48 residual head. |
| `controllers/cartesian/tracking.py` | `TRACK` gains used by the verified catch. |
| `models/k48_policy.py` | K48 transformer. The residual head is `res`. |
| `models/k48_checkpoint.py` | Load a K48 checkpoint or expand a K47 checkpoint. Keys are unchanged. |
| `models/airborne_v3.py` | 39D policy load and `step_v3`. The demo generator is `experiments/legacy/airborne_v3_demos.py`. |
| `training/train_sft.py` | Supervised loss and the SFT loop. |
| `training/train_grpo.py` | GRPO objective and the GRPO loop. |
| `training/reward_v2.py` | Selected GRPO return. Configs call it `reward_version: v2`. |
| `training/k47_codebook.py` | K47 codebook load and `decode_action`. Token 47 is rejected. |
| `training/recovery_episode.py` | Closed-loop episode: nominal, policy, handoff, nominal. |
| `data_generation/rebuild_k48_recovery.py` | Official K48 demonstration generator. |
| `data_generation/oracles/representability.py` | Quantize a 7D command onto the codebook. |
| `sensors/lateral_offset.py` | `e_x`: object-origin offset along hand x, estimated from finger-pad centers of pressure. |
| `visualization/render_video.py` | Replay a saved tape. CLI defaults are the vertical-cylinder example. |
| `visualization/video_recorder.py` | Write frames to MP4. It does not run MuJoCo. |

## 4. Old path to new path

| Old | New | Why |
|---|---|---|
| `controllers/jacobian_controller.py` | `controllers/cartesian/jacobian.py` | Cartesian tracking, not a separate product. |
| `controllers/bias.py` | `controllers/cartesian/bias_torque.py` | `qfrc_bias` compensation. |
| `controllers/gripper_controller.py` | `controllers/gripper/controller.py` | Gripper torque. |
| `controllers/recovery_actions.py` | `controllers/recovery/action_map.py` | 7D action map. |
| `controllers/residual.py` | `controllers/recovery/residual_limits.py` | Limits, not the K48 residual head. |
| `controllers/geometry_switch.py` | `controllers/recovery/geometry_switch.py` | Recovery entry and exit. |
| `controllers/rule_based_recovery.py` | `controllers/recovery/rule_based.py` | Frozen rule baseline. |
| `controllers/skill_schedules.py` | `controllers/recovery/skill_schedules.py` | Fixed 7D command tapes. |
| `envs/ids.py` | `envs/simulation/mujoco_ids.py` | MuJoCo body and joint ids. |
| `envs/dynamics.py` | `envs/simulation/object_dynamics.py` | Mass and friction. |
| `envs/deterioration.py` | `envs/simulation/grasp_quality.py` | Grasp-quality meter `D_t` and `body_twist`. |
| `envs/config_util.py` | `envs/simulation/yaml_config.py` | YAML load and sim merge. |
| `envs/xml_build.py` | `envs/simulation/panda_xml.py` | Menagerie to `panda_torque.xml`. |
| `envs/grasp_sim.py` | `envs/simulation/grasp_sim.py` | Core simulator. |
| `envs/contact.py` | `envs/contacts/touch.py` | Summed fingertip force. |
| `envs/contacts.py` | `envs/contacts/frames.py` | Per-contact frames. Not merged with touch. |
| `envs/observable_obs.py` | `envs/observations/observable_obs.py` | 27D observation. |
| `envs/airborne_obs.py` | `envs/observations/airborne_obs.py` | 39D observation. |
| `envs/observable_reward.py` | `envs/observations/tactile_reward.py` | Tactile reward. |
| `envs/horizon_outcome.py` | `envs/outcomes/horizon_outcome.py` | Episode outcome. |
| `envs/recovery_success.py` | `envs/outcomes/recovery_success.py` | Basin, handoff, nominal hold. |
| `envs/table_drop.py` | `envs/outcomes/table_drop.py` | Latch. |
| `envs/physical_recovery.py` | `envs/outcomes/physical_recovery.py` | Physical pack and height gate. |
| `envs/grasp_seat.py` | `envs/initialization/grasp_seat.py` | Seated initial grasp. |
| `sensors/ex_estimator.py` | `sensors/lateral_offset.py` | `e_x` is the hand-x object offset. |
| `models/k36_policy.py` | `models/historical/k36_policy.py` | Historical network. |
| `models/action_kmeans.py` | `data_generation/action_kmeans.py` | Offline token clustering, not a network. |
| `training/k33_checkpoint.py` | `models/historical/k33_checkpoint.py` | Historical loader. |
| `training/k36_checkpoint.py` | `models/historical/k36_checkpoint.py` | Historical loader. |
| `training/natural_loss_recatch_transformer_v3.py` | `models/airborne_v3.py` | 39D policy used by the closed-loop episode. |
| `training/replay_core.py` | `envs/simulation/rule_replay.py` | Scene replay used by the simulator. |
| `training/impact_ball.py` | `envs/simulation/impact_ball.py` | Ball state used by the ballistic scene. |
| `training/write_ballistic_impact_scene.py` | `envs/simulation/ballistic_scene.py` | Scene XML and cylinder constants. |
| `training/write_impact_scene.py` | `envs/simulation/impact_diag_scene.py` | Diagnostic scene XML. |
| `training/autonomous_episode.py` | `training/recovery_episode.py` | Closed-loop episode, not a dataset. |
| `training/recovery_runtime.py` | `training/recovery_execution.py` | 7D execution, snapshots, long nominal hold. |
| `training/history_aware_critic_ablation.py` | `experiments/legacy/history_aware_critic.py` | Ablation. Capture moved out first. |
| `training/nominal_pose_audit.py` | `experiments/legacy/nominal_pose_audit.py` | Audit CLI. `errors` and `hand_R` moved out first. |
| `training/autonomous_rl_sanity.py` | `experiments/legacy/autonomous_rl_sanity.py` | Short RL pilot. |
| `training/autonomous_dataset.py` | `data_generation/autonomous_episodes.py` | Episode generation. |
| `training/full_3d_airborne_recatch_dataset.py` | `data_generation/airborne_recatch_dataset.py` | Airborne demonstration generator. |
| `training/impact_demo_core.py` | `experiments/legacy/impact_demo.py` | Impact demo driver. |
| `training/impact_visualization_utils.py` | `visualization/impact_view.py` | Viewer helpers. |
| `training/policy_rollout.py` | `data_generation/policy_rollout.py` | One policy step while recording. |
| `training/jsonable.py` | `data_generation/jsonable.py` | Numpy-to-JSON for logs. |
| `data_generation/oracle_search.py` | `data_generation/oracles/representability.py` | The file projects commands; it does not search. |
| `config/*.yaml` | `config/{simulation,controllers,recovery,training}/` | Grouped by what the file configures. Values unchanged. |

`envs/step_legal.py` and `envs/ballistic_impact.py` stayed at the package root so the legal step and the ballistic scene stay easy to find.

## 5. Mixed files that were split

| File | Split |
|---|---|
| `history_aware_critic_ablation.py` | `take_capture` / `restore_capture` are `envs/simulation/capture.py`. The ablation script is `experiments/legacy/history_aware_critic.py`. |
| `nominal_pose_audit.py` | `errors`, `hand_R`, `capture_ref`, and `closed_grip` are `envs/outcomes/nominal_pose.py`. The study CLI stayed historical. |
| `autonomous_rl_sanity.py` | `supports_of` / `restore_supports` are `envs/simulation/support_contacts.py`. The RL pilot stayed historical. |
| `full_3d_airborne_recatch_dataset.py` | The `TRACK` gain dict is `controllers/cartesian/tracking.py`. The generator stayed a dataset module. |

`rebuild_k48_recovery.py`, `ball_impact_dataset.py`, `recatch_oracle.py`, and the two sliding searches still search, check, and write in one file. Those loops share one rollout. They were not split.

## 6. Duplicates

No second copy of the K48 network, the SFT loss, the GRPO objective, contact extraction, or the legal step was introduced. `video_recorder.py` only encodes frames. `render_video.py` is the MuJoCo replay. `contacts/touch.py` and `contacts/frames.py` stay separate because one sums fingertip force and the other records contact frames.

The frozen rule-controller source pin hashes `controllers/recovery/rule_based.py` and `config/controllers/rule_based_recovery.yaml`. This cleanup did not edit either file. `verify_frozen_hashes()` still matches `778bd97d…` and `6a0b8103…`.

## 7. Names that were kept on purpose

| Name | Reason |
|---|---|
| `training/reward_v2.py` | `reward_version: v2` in the live GRPO configs. The filename is that contract. |
| `training/k47_codebook.py` | K47 codebook width and `decode_action`. The historical loop is `experiments/legacy/k47_training.py`. |
| `training/unified_k48.py` | Command-line entry for the K48 trainer. |
| `models/k48_policy.py` | Supported network. Left at `models/` rather than nested again. |
| `envs/outcomes/physical_recovery.py` | Existing name for the physical pack and the height gate. |
| `envs/simulation/rule_replay.py` | Frozen RULE replay and the source-hash pin. Scene helpers live next to it. |
| `training/grpo_progress.py` | GRPO progress log and update timing. It is not a shared trainer. |

## 8. Historical experiments

| Location | What it is |
|---|---|
| `experiments/legacy/` | K36 GRPO, the K47 training loop `k47_training.py`, the 39D demo generator `airborne_v3_demos.py`, and older pilots. |
| `experiments/legacy/demo/audit/` | Old coverage and physics audits. |
| `experiments/legacy/demo/replay/` | Codebook and hold replays. |
| `experiments/legacy/demo/diagnose/` | Closed-loop and handoff diagnoses. |
| `experiments/legacy/demo/demonstrations/` | Rendered historical demos. |
| `experiments/legacy/demo/build_k48_handoff_policy.py` | Superseded by `models/k48_checkpoint.py`. Not on the training path. |
| `experiments/legacy/demo/build_train_manifest.py` | Old manifest join. Not imported by formal packages. |
| `experiments/legacy/demo/expand_k48_handoff_dataset.py` | Old handoff expansion. Not imported by formal packages. |
| `experiments/legacy/demo/refactor_repository.py` | The earlier path-rewrite script. Not imported. |
| `experiments/recatch/airborne/` | Airborne and freefall recatch studies. |
| `experiments/recatch/natural_loss/` | Seated natural-loss studies. |
| `experiments/recatch/legal/` | `step_legal` recatch searches. |
| `experiments/recatch/top_impact/` | Top-impact recatch studies. |
| `models/historical/` | K33 and K36 loaders. K48 training does not run them. |
| `config/training/k47/` and `config/training/legacy/` | Older training configs. Numeric values were not changed. |

## 9. Regression

`python -u tests/test_physics_regression.py` printed `PHYSICS_REGRESSION_OK` after this cleanup.

The run checks determinism by replaying each tape twice, then compares qpos, qvel, the 39D observation, contact geom/normal/force, and controller state with `tests/fixtures/physics_state_baseline.npz`. That fixture is the current verified tree. It is not a pre-refactor ground truth. Details are in `docs/FINAL_REFACTOR_CHECK.md`.

## 10. Still open

- `data_generation/maintenance/` is isolated from `generate.py`, but old experiments can still call those scripts.
- Current `results/diagnostics/` directories were not moved. Live scripts still point at them.
- `experiments/ball_impact/privileged_ball_impact_search.py` keeps a one-line `step_legal` delegate. The implementation is `envs/step_legal.py`.
- `experiments/legacy/demo/build_k48_handoff_policy.py` still has a historical copy of the K48 expansion. Formal training uses `models/k48_checkpoint.py`.
