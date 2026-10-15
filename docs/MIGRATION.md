# File moves

Round 3 moved formal modules into subpackages and renamed files whose names did not match their jobs. The current map is `docs/MODULE_ORGANIZATION.md`. The list below is the cumulative rename log; destinations are the paths after that move.

Git renames from this layout change. Imports were rewritten to the new modules.

- `training/align_data_layout.py` -> `data_generation/maintenance/align_data_layout.py`
- `training/ball_impact_dataset.py` -> `data_generation/ball_impact_dataset.py`
- `training/consolidate_snapshots.py` -> `data_generation/maintenance/consolidate_snapshots.py`
- `training/rebuild_k48_recovery.py` -> `data_generation/rebuild_k48_recovery.py`
- `training/relocate_training_data.py` -> `data_generation/maintenance/relocate_training_data.py`
- `training/build_open_overchase_recatch_dataset.py` -> `data_generation/strategies/build_open_overchase_recatch_dataset.py`
- `training/build_vertical_recatch_dataset.py` -> `data_generation/strategies/build_vertical_recatch_dataset.py`
- `training/capture_geometry.py` -> `data_generation/strategies/capture_geometry.py`
- `training/k48_controlled_slide_search.py` -> `data_generation/strategies/k48_controlled_slide_search.py`
- `training/k48_inward_sliding_search.py` -> `data_generation/strategies/k48_inward_sliding_search.py`
- `training/vertical_recatch_search.py` -> `data_generation/strategies/vertical_recatch_search.py`
- `CS5478_Preproposal_Meeting.pdf` -> `docs/coursework/CS5478_Preproposal_Meeting.pdf`
- `CS5478_Project_Proposal.pdf` -> `docs/coursework/CS5478_Project_Proposal.pdf`
- `CS5478_Project_Proposal_Presentation.pdf` -> `docs/coursework/CS5478_Project_Proposal_Presentation.pdf`
- `training/demo_ballistic_impact.py` -> `envs/ballistic_impact.py`
- `training/horizon_outcome.py` -> `envs/outcomes/horizon_outcome.py`
- `training/recovery_success.py` -> `envs/outcomes/recovery_success.py`
- `training/table_drop.py` -> `envs/outcomes/table_drop.py`
- `training/audit_pretrain_data.py` -> `evaluation/audit_pretrain_data.py`
- `training/audit_recatch_legality.py` -> `evaluation/audit_recatch_legality.py`
- `training/verify_handoff_targets.py` -> `evaluation/verify_handoff_targets.py`
- `training/ball_impact_handoff_causal_audit.py` -> `experiments/ball_impact/ball_impact_handoff_causal_audit.py`
- `training/ball_impact_recoverability.py` -> `experiments/ball_impact/ball_impact_recoverability.py`
- `training/ball_impact_targeted.py` -> `experiments/ball_impact/ball_impact_targeted.py`
- `training/privileged_ball_impact_search.py` -> `experiments/ball_impact/privileged_ball_impact_search.py`
- `training/benchmark_k48_prefix_branching.py` -> `experiments/handoff/benchmark_k48_prefix_branching.py`
- `training/diagnose_k48_counterfactual_credit.py` -> `experiments/handoff/diagnose_k48_counterfactual_credit.py`
- `training/balanced_unified_recovery.py` -> `experiments/legacy/balanced_unified_recovery.py`
- `training/continual_skill_addition.py` -> `experiments/legacy/continual_skill_addition.py`
- `demo/__init__.py` -> `experiments/legacy/demo/__init__.py`
- `demo/abc_recovery_validation.py` -> `experiments/legacy/demo/abc_recovery_validation.py`
- `demo/audit_controlled_sliding.py` -> `experiments/legacy/demo/audit/audit_controlled_sliding.py`
- `demo/audit_historical_skill_physics.py` -> `experiments/legacy/demo/audit/audit_historical_skill_physics.py`
- `demo/audit_nominal_handoff.py` -> `experiments/legacy/demo/audit/audit_nominal_handoff.py`
- `demo/ballistic_large_angle.py` -> `experiments/legacy/demo/ballistic_large_angle.py`
- `demo/ballistic_slip_sufficiency.py` -> `experiments/legacy/demo/ballistic_slip_sufficiency.py`
- `demo/build_k48_handoff_policy.py` -> `experiments/legacy/demo/build_k48_handoff_policy.py`
- `demo/build_train_manifest.py` -> `experiments/legacy/demo/build_train_manifest.py`
- `demo/codebook_prefix.py` -> `experiments/legacy/demo/codebook_prefix.py`
- `demo/compare_k48_policy_evaluations.py` -> `experiments/legacy/demo/compare_k48_policy_evaluations.py`
- `demo/compare_recovery_termination.py` -> `experiments/legacy/demo/compare_recovery_termination.py`
- `demo/coverage_all_holds.py` -> `experiments/legacy/demo/coverage_all_holds.py`
- `demo/coverage_counts.py` -> `experiments/legacy/demo/coverage_counts.py`
- `demo/demo_airborne_recapture.py` -> `experiments/legacy/demo/demonstrations/demo_airborne_recapture.py`
- `demo/demo_airborne_recovery.py` -> `experiments/legacy/demo/demonstrations/demo_airborne_recovery.py`
- `demo/demo_ballistic_recovery.py` -> `experiments/legacy/demo/demonstrations/demo_ballistic_recovery.py`
- `demo/demo_ballistic_recovery_comparison.py` -> `experiments/legacy/demo/demonstrations/demo_ballistic_recovery_comparison.py`
- `demo/demo_dynamic_recatch.py` -> `experiments/legacy/demo/demonstrations/demo_dynamic_recatch.py`
- `demo/demo_grav_reposition_comparison.py` -> `experiments/legacy/demo/demonstrations/demo_grav_reposition_comparison.py`
- `demo/demo_grav_reposition_recovery.py` -> `experiments/legacy/demo/demonstrations/demo_grav_reposition_recovery.py`
- `demo/diagnose_compound_orientation.py` -> `experiments/legacy/demo/diagnose/diagnose_compound_orientation.py`
- `demo/diagnose_grpo_v2_handoff_regression.py` -> `experiments/legacy/demo/diagnose/diagnose_grpo_v2_handoff_regression.py`
- `demo/diagnose_k47_closed_loop.py` -> `experiments/legacy/demo/diagnose/diagnose_k47_closed_loop.py`
- `demo/diagnose_omega4_throw.py` -> `experiments/legacy/demo/diagnose/diagnose_omega4_throw.py`
- `demo/diagnose_sft_epoch40_closed_loop.py` -> `experiments/legacy/demo/diagnose/diagnose_sft_epoch40_closed_loop.py`
- `demo/expand_k48_handoff_dataset.py` -> `experiments/legacy/demo/expand_k48_handoff_dataset.py`
- `demo/finalize_codebook_views.py` -> `experiments/legacy/demo/finalize_codebook_views.py`
- `demo/grpo_norm_check.py` -> `experiments/legacy/demo/grpo_norm_check.py`
- `demo/handoff_height_audit.py` -> `experiments/legacy/demo/handoff_height_audit.py`
- `demo/launch_entry_stats.py` -> `experiments/legacy/demo/launch_entry_stats.py`
- `demo/legal_recovery_v2_pilot.py` -> `experiments/legacy/demo/legal_recovery_v2_pilot.py`
- `demo/map_ballistic_disturbance.py` -> `experiments/legacy/demo/map_ballistic_disturbance.py`
- `demo/nominal_grasp_creep_root_cause.py` -> `experiments/legacy/demo/nominal_grasp_creep_root_cause.py`
- `demo/refactor_repository.py` -> `experiments/legacy/demo/refactor_repository.py`
- `demo/release_coverage_v3.py` -> `experiments/legacy/demo/release_coverage_v3.py`
- `demo/release_state_map_omega4.py` -> `experiments/legacy/demo/release_state_map_omega4.py`
- `demo/render_grav_reposition.py` -> `experiments/legacy/demo/demonstrations/render_grav_reposition.py`
- `demo/render_zero_vs_verified.py` -> `experiments/legacy/demo/demonstrations/render_zero_vs_verified.py`
- `demo/replay_all_holds.py` -> `experiments/legacy/demo/replay/replay_all_holds.py`
- `demo/replay_c_subset.py` -> `experiments/legacy/demo/replay/replay_c_subset.py`
- `demo/replay_candidate_codebook.py` -> `experiments/legacy/demo/replay/replay_candidate_codebook.py`
- `demo/replay_codebook_v3.py` -> `experiments/legacy/demo/replay/replay_codebook_v3.py`
- `demo/replay_divergence.py` -> `experiments/legacy/demo/replay/replay_divergence.py`
- `demo/replay_raw_vs_quant.py` -> `experiments/legacy/demo/replay/replay_raw_vs_quant.py`
- `demo/reproduce_center6_large_angle.py` -> `experiments/legacy/demo/reproduce_center6_large_angle.py`
- `demo/revise_codebook.py` -> `experiments/legacy/demo/revise_codebook.py`
- `demo/robustness_probe.py` -> `experiments/legacy/demo/robustness_probe.py`
- `demo/select_append_codebook.py` -> `experiments/legacy/demo/select_append_codebook.py`
- `demo/signed_decode_error.py` -> `experiments/legacy/demo/signed_decode_error.py`
- `demo/summarize_legal_v3.py` -> `experiments/legacy/demo/summarize_legal_v3.py`
- `demo/summarize_replays.py` -> `experiments/legacy/demo/summarize_replays.py`
- `demo/uncovered_release_steps.py` -> `experiments/legacy/demo/uncovered_release_steps.py`
- `training/demo_teleport_recovery_state.py` -> `experiments/legacy/demo_teleport_recovery_state.py`
- `training/film_impact_zp6_k33.py` -> `experiments/legacy/film_impact_zp6_k33.py`
- `training/final_rl_readiness.py` -> `experiments/legacy/final_rl_readiness.py`
- `training/future_state_ablation.py` -> `experiments/legacy/future_state_ablation.py`
- `training/long_context_recovery_pilot.py` -> `experiments/legacy/long_context_recovery_pilot.py`
- `training/migrate_horizon_dataset.py` -> `experiments/legacy/migrate_horizon_dataset.py`
- `training/nominal_grasp_restoration.py` -> `experiments/legacy/nominal_grasp_restoration.py`
- `training/observation_study_common.py` -> `experiments/legacy/observation_study_common.py`
- `training/phase_decoupled_recovery_pilot.py` -> `experiments/legacy/phase_decoupled_recovery_pilot.py`
- `training/recovery_policy_pretraining_preparation.py` -> `experiments/legacy/recovery_policy_pretraining_preparation.py`
- `training/sac_dpo_grpo_selection.py` -> `experiments/legacy/sac_dpo_grpo_selection.py`
- `training/single_step_action_token_policy.py` -> `experiments/legacy/single_step_action_token_policy.py`
- `training/supervised_coverage_warmstart.py` -> `experiments/legacy/supervised_coverage_warmstart.py`
- `training/vertical_slip_contact_mechanics_audit.py` -> `experiments/legacy/vertical_slip_contact_mechanics_audit.py`
- `training/airborne_intercept_recatch.py` -> `experiments/recatch/airborne/airborne_intercept_recatch.py`
- `training/clean_natural_loss_recatch.py` -> `experiments/recatch/natural_loss/clean_natural_loss_recatch.py`
- `training/contact_aware_recatch_search.py` -> `experiments/recatch/legal/contact_aware_recatch_search.py`
- `training/external_airborne_recatch.py` -> `experiments/recatch/airborne/external_airborne_recatch.py`
- `training/freefall_capture_diagnostic.py` -> `experiments/recatch/airborne/freefall_capture_diagnostic.py`
- `training/horizontal_grasp_offset_freefall_recatch.py` -> `experiments/recatch/airborne/horizontal_grasp_offset_freefall_recatch.py`
- `training/legal_recatch_k48_handoff.py` -> `experiments/recatch/legal/legal_recatch_k48_handoff.py`
- `training/legal_recatch_probe.py` -> `experiments/recatch/legal/legal_recatch_probe.py`
- `training/legal_recatch_speed_match.py` -> `experiments/recatch/legal/legal_recatch_speed_match.py`
- `training/legal_recatch_transfer.py` -> `experiments/recatch/legal/legal_recatch_transfer.py`
- `training/multi_disturbance_airborne_feasibility.py` -> `experiments/recatch/airborne/multi_disturbance_airborne_feasibility.py`
- `training/natural_loss_recatch_coverage_expansion.py` -> `experiments/recatch/natural_loss/natural_loss_recatch_coverage_expansion.py`
- `training/natural_loss_recatch_data_scaling.py` -> `experiments/recatch/natural_loss/natural_loss_recatch_data_scaling.py`
- `training/new_recatch_supervised.py` -> `experiments/recatch/new_recatch_supervised.py`
- `training/probe_airborne_capability.py` -> `experiments/recatch/airborne/probe_airborne_capability.py`
- `training/probe_gripper_unstick.py` -> `experiments/recatch/legal/probe_gripper_unstick.py`
- `training/recatch_braking_optimization.py` -> `experiments/recatch/recatch_braking_optimization.py`
- `training/recatch_interface_comparison.py` -> `experiments/recatch/airborne/recatch_interface_comparison.py`
- `training/recovery_ic_expansion.py` -> `experiments/recatch/top_impact/recovery_ic_expansion.py`
- `training/top_impact_open_overchase_recatch.py` -> `experiments/recatch/top_impact/top_impact_open_overchase_recatch.py`
- `training/top_impact_progressive_recatch.py` -> `experiments/recatch/top_impact/top_impact_progressive_recatch.py`
- `training/top_impact_recatch_geometry_fix.py` -> `experiments/recatch/top_impact/top_impact_recatch_geometry_fix.py`
- `training/uncapped_cartesian_airborne_chase.py` -> `experiments/recatch/airborne/uncapped_cartesian_airborne_chase.py`
- `training/expand_controlled_sliding.py` -> `experiments/sliding/expand_controlled_sliding.py`
- `training/expand_inward_training_data.py` -> `experiments/sliding/expand_inward_training_data.py`
- `training/export_inward_training_data.py` -> `experiments/sliding/export_inward_training_data.py`
- `training/generate_inward_until.py` -> `experiments/sliding/generate_inward_until.py`
- `training/sliding_vs_regrasp.py` -> `experiments/sliding/sliding_vs_regrasp.py`
- `training/test_grpo_progress.py` -> `tests/test_grpo_progress.py`
- `training/test_reward_v2.py` -> `tests/test_reward_v2.py`
- `training/test_table_drop.py` -> `tests/test_table_drop.py`
- `training/render_vertical_recatch.py` -> `visualization/render_video.py`

## Kept in training/

These files still implement the live training runtime:

- `unified_k48.py` is the CLI. The loops live in `train_sft.py`, `train_grpo.py`, `dataset.py`, and `trainer.py`.
- `unified_k47.py` keeps the K=47 codebook, decode, and historical K47 training loop. It is not merged into K48.
- `reward_v2.py`, `autonomous_episode.py`, `autonomous_dataset.py`, `autonomous_rl_sanity.py`
- `natural_loss_recatch_transformer_v3.py`
- `replay_core.py`, `recovery_runtime.py`, `nominal_pose_audit.py`
- `full_3d_airborne_recatch_dataset.py`, `history_aware_critic_ablation.py`
- `impact_ball.py`, `impact_demo_core.py`, `impact_visualization_utils.py`
- `write_ballistic_impact_scene.py`, `write_impact_scene.py`
- `grpo_math.py`, `k48_config.py`, `k33_checkpoint.py`, `k36_checkpoint.py`, `jsonable.py`, `policy_rollout.py`

## Round 2

Shared implementations moved out of experiment scripts. The experiment files import the new modules.

- Contact extraction: `envs/contacts/frames.py`
- Grasp seating and seated offsets: `envs/initialization/grasp_seat.py`
- Fixed 7D skill tapes: `controllers/recovery/skill_schedules.py`
- Slide and privileged chase laws: `data_generation/strategies/recovery_laws.py`
- Horizontal recatch oracle used by dataset generation: `data_generation/strategies/recatch_oracle.py`
- K=36 head: `models/historical/k36_policy.py`
- Historical K36 GRPO, pre-GRPO sanity, and the representation audit: `experiments/legacy/formal_grpo.py`, `experiments/legacy/pre_grpo_sanity.py`, `experiments/legacy/action_representation_expansion.py`
- Top-level `demo_results/` renamed to `results/archive/demo_results/` after a SHA-256 inventory. The inventory is `results/archive/demo_results_sha256.txt`.

Formal packages do not import `experiments/`. Details and the regression record are in `docs/REFACTOR_AUDIT.md`.
