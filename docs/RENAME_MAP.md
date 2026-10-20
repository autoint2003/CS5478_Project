# Naming refactor map

| Old path | New path |
|---|---|
| `data_generation/strategies/ground_future_oracle.py` | `data_generation/oracles/plan_c/future_trajectory.py` |
| `data_generation/strategies/ground_early_recovery.py` | `data_generation/oracles/plan_c/interception.py` |
| `data_generation/strategies/ground_dynamic_catch.py` | `data_generation/oracles/plan_c/dynamic_catch.py` |
| `data_generation/strategies/ground_postcatch_endgrasp.py` | `data_generation/oracles/plan_c/catch_stabilization.py` |
| `data_generation/strategies/k48_controlled_slide_search.py` | `data_generation/oracles/plan_a/controlled_sliding.py` |
| `data_generation/strategies/k48_inward_sliding_search.py` | `data_generation/oracles/plan_a/inward_reseat.py` |
| `data_generation/strategies/capture_geometry.py` | `data_generation/oracles/plan_a/contact_geometry.py` |
| `data_generation/strategies/recovery_laws.py` | `data_generation/oracles/plan_a/recovery_laws.py` |
| `data_generation/strategies/recatch_oracle.py` | `data_generation/oracles/plan_b/release_recatch.py` |
| `data_generation/strategies/vertical_recatch_search.py` | `data_generation/oracles/plan_b/vertical_recatch.py` |
| `data_generation/strategies/build_open_overchase_recatch_dataset.py` | `data_generation/datasets/open_overchase_recatch.py` |
| `data_generation/strategies/build_vertical_recatch_dataset.py` | `data_generation/datasets/vertical_recatch.py` |
| `data_generation/airborne_recatch_dataset.py` | `data_generation/datasets/airborne_recatch.py` |
| `data_generation/ball_impact_dataset.py` | `data_generation/datasets/ball_impact.py` |
| `data_generation/autonomous_episodes.py` | `data_generation/datasets/autonomous_episodes.py` |
| `data_generation/expand_skill_quotas.py` | `data_generation/datasets/skill_quotas.py` |
| `data_generation/skill_filters.py` | `data_generation/datasets/skill_filters.py` |
| `data_generation/sliding_labels.py` | `data_generation/datasets/sliding_labels.py` |
| `data_generation/rebuild_k48_recovery.py` | `data_generation/datasets/recovery_rebuild.py` |
| `data_generation/action_kmeans.py` | `data_generation/datasets/action_kmeans.py` |
| `data_generation/policy_rollout.py` | `data_generation/datasets/policy_rollout.py` |
| `data_generation/maintenance/align_data_layout.py` | `experiments/maintenance/align_data_layout.py` |
| `data_generation/maintenance/consolidate_snapshots.py` | `experiments/maintenance/consolidate_snapshots.py` |
| `data_generation/maintenance/relocate_training_data.py` | `experiments/maintenance/relocate_training_data.py` |
| `models/airborne_v3.py` | `models/transformer_policy.py` |
| `training/train_sft.py` | `training/sft.py` |
| `training/train_grpo.py` | `training/grpo.py` |
| `training/unified_k48.py` | `training/pipeline.py` |
| `training/reward_v2.py` | `training/rewards.py` |
| `training/prefix_rollout.py` | `training/rollout.py` |
| `evaluation/full_objective.py` | `evaluation/success_criteria.py` |
| `evaluation/grasp_configuration.py` | `evaluation/grasp_geometry.py` |
| `evaluation/verify_handoff_targets.py` | `evaluation/handoff_validation.py` |
| `evaluation/k48_eval.py` | `evaluation/policy_evaluation.py` |
| `scripts/build_unified_dataset.py` | `scripts/build_expert_dataset.py` |
| `scripts/replay_golden_experts.py` | `scripts/replay_experts.py` |
| `scripts/audit_k49.py` | `scripts/audit_action_vocab.py` |
| `scripts/train.py` | `scripts/train_policy.py` |
| `scripts/evaluate.py` | `scripts/evaluate_policy.py` |
| `scripts/render_video.py` | `scripts/render_trajectory.py` |
| `scripts/generate_data.py` | `scripts/generate_experts.py` |
| `scripts/generate_plan_c_experts.py` | `scripts/generate_plan_c_experts.py` |
| `scripts/consolidation/freeze_experts.py` | `scripts/consolidation/freeze_experts.py` |
| `scripts/consolidation/audit_token_representability.py` | `scripts/consolidation/audit_token_representability.py` |

## Archived to experiments/

- `scripts/run_ground_ball_impact_capture_refinement.py` → `experiments/diagnostics/run_ground_ball_impact_capture_refinement.py`
- `scripts/run_ground_ball_impact_early_recovery.py` → `experiments/diagnostics/run_ground_ball_impact_early_recovery.py`
- `scripts/run_ground_ball_impact_oracle_interception.py` → `experiments/diagnostics/run_ground_ball_impact_oracle_interception.py`
- `scripts/run_ground_dynamic_catch_oracle.py` → `experiments/diagnostics/run_ground_dynamic_catch_oracle.py`
- `scripts/run_ground_postcatch_endgrasp_recovery.py` → `experiments/diagnostics/run_ground_postcatch_endgrasp_recovery.py`
- `scripts/run_ground_scene_validation.py` → `experiments/diagnostics/run_ground_scene_validation.py`
- `scripts/run_plan_a_ground_scene_divergence.py` → `experiments/diagnostics/run_plan_a_ground_scene_divergence.py`
- `scripts/run_plan_a_sliding_pivoting_legality_audit.py` → `experiments/diagnostics/run_plan_a_sliding_pivoting_legality_audit.py`
- `scripts/run_plan_c_horizontal_downward_end_recatch.py` → `experiments/diagnostics/run_plan_c_horizontal_downward_end_recatch.py`
- `scripts/run_active_recatch_generalization_discretization.py` → `experiments/diagnostics/run_active_recatch_generalization_discretization.py`
- `scripts/run_contact_geometry_table_window.py` → `experiments/diagnostics/run_contact_geometry_table_window.py`
- `scripts/run_controlled_slow_detachment.py` → `experiments/diagnostics/run_controlled_slow_detachment.py`
- `scripts/run_free_fall_cylinder_catching.py` → `experiments/diagnostics/run_free_fall_cylinder_catching.py`
- `scripts/run_freefall_e_cpp_unload_isolation.py` → `experiments/diagnostics/run_freefall_e_cpp_unload_isolation.py`
- `scripts/run_freefall_e_postcapture_stabilize_slide.py` → `experiments/diagnostics/run_freefall_e_postcapture_stabilize_slide.py`
- `scripts/run_freefall_e_stable_lift_contact_manip.py` → `experiments/diagnostics/run_freefall_e_stable_lift_contact_manip.py`
- `scripts/run_freefall_recatch_collision_aware_feedback.py` → `experiments/diagnostics/run_freefall_recatch_collision_aware_feedback.py`
- `scripts/run_freefall_recatch_grasp_point_alignment.py` → `experiments/diagnostics/run_freefall_recatch_grasp_point_alignment.py`
- `scripts/run_freefall_recatch_pose_feedback.py` → `experiments/diagnostics/run_freefall_recatch_pose_feedback.py`
- `scripts/run_horizontal_offset_freefall_recatch.py` → `experiments/diagnostics/run_horizontal_offset_freefall_recatch.py`
- `scripts/run_k49_action_authority_ablation.py` → `experiments/diagnostics/run_k49_action_authority_ablation.py`
- `scripts/run_lateral_force_full_pose_recovery.py` → `experiments/diagnostics/run_lateral_force_full_pose_recovery.py`
- `scripts/run_lateral_force_grasp_loss_recovery.py` → `experiments/diagnostics/run_lateral_force_grasp_loss_recovery.py`
- `scripts/run_lateral_force_mechanism_audit.py` → `experiments/diagnostics/run_lateral_force_mechanism_audit.py`
- `scripts/run_lateral_offset_falling_recatch.py` → `experiments/diagnostics/run_lateral_offset_falling_recatch.py`
- `scripts/run_lateral_recatch_handoff.py` → `experiments/diagnostics/run_lateral_recatch_handoff.py`
- `scripts/run_real_ball_impact_low_speed_recatch.py` → `experiments/diagnostics/run_real_ball_impact_low_speed_recatch.py`
- `scripts/run_velocity_matched_soft_catch.py` → `experiments/diagnostics/run_velocity_matched_soft_catch.py`
- `scripts/expand_k49_capture_brake_diagnostics.py` → `experiments/diagnostics/expand_k49_capture_brake_diagnostics.py`
- `scripts/expand_k49_hard_dynamic.py` → `experiments/diagnostics/expand_k49_hard_dynamic.py`
- `scripts/expand_k49_large_chase_dynamic.py` → `experiments/diagnostics/expand_k49_large_chase_dynamic.py`
- `scripts/expand_k49_unified_dataset.py` → `experiments/diagnostics/expand_k49_unified_dataset.py`
- `scripts/generate_grasp_recovery_pose_experts_v1.py` → `experiments/diagnostics/generate_grasp_recovery_pose_experts_v1.py`
- `scripts/generate_k49_ball_impact_dynamic_recovery.py` → `experiments/diagnostics/generate_k49_ball_impact_dynamic_recovery.py`
- `scripts/generate_k49_dynamic_recatch_experts.py` → `experiments/diagnostics/generate_k49_dynamic_recatch_experts.py`
- `scripts/generate_k49_horizontal_endcatch_recatch.py` → `experiments/diagnostics/generate_k49_horizontal_endcatch_recatch.py`
- `scripts/generate_k49_in_recovery_recatch.py` → `experiments/diagnostics/generate_k49_in_recovery_recatch.py`
- `scripts/generate_k49_lost_grasp_recovery.py` → `experiments/diagnostics/generate_k49_lost_grasp_recovery.py`
- `scripts/generate_k49_sliding_recatch_dataset.py` → `experiments/diagnostics/generate_k49_sliding_recatch_dataset.py`
- `scripts/generate_k49_type3_capture_brake.py` → `experiments/diagnostics/generate_k49_type3_capture_brake.py`
- `scripts/merge_grpo_ic_datasets.py` → `experiments/diagnostics/merge_grpo_ic_datasets.py`
- `evaluation/audit_grasp_configuration_migration.py` → `experiments/audits/audit_grasp_configuration_migration.py`
- `evaluation/audit_mechanism_selection_feasibility.py` → `experiments/audits/audit_mechanism_selection_feasibility.py`
- `evaluation/audit_pretrain_data.py` → `experiments/audits/audit_pretrain_data.py`
- `evaluation/audit_recatch_legality.py` → `experiments/audits/audit_recatch_legality.py`
- `evaluation/classify_validation_skills.py` → `experiments/audits/classify_validation_skills.py`
- `evaluation/skill_traces.py` → `experiments/audits/skill_traces.py`
- `evaluation/plan_a_mechanism_audit.py` → `experiments/audits/plan_a_mechanism_audit.py`
- `docs/FINAL_REFACTOR_CHECK.md` → `experiments/docs/FINAL_REFACTOR_CHECK.md`
- `docs/HISTORICAL_K32_README.md` → `experiments/docs/HISTORICAL_K32_README.md`
- `docs/MIGRATION.md` → `experiments/docs/MIGRATION.md`
- `docs/REFACTOR_AUDIT.md` → `experiments/docs/REFACTOR_AUDIT.md`
- `config/training/k47|k48|legacy` → `experiments/config/training/`
