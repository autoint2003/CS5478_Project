# Archive Manifest

Generated (UTC): 2026-10-20T16:29:22.563768+00:00
Source commit: `7db29b325202d04e7a2b774b79ea1178a368fc64`
Archival branch/tag: `archive/pre-consolidation-20261021` / `archive/pre-consolidation-20261021-src`

## Policy

- Git-ignored `results/diagnostics/` originals are **not deleted**.
- Golden Plan C experts are **copied** to `archive/golden_experts/` with SHA-256 checks.
- Other diagnostics are **indexed** under `archive/experiments/<name>/` with report copies.
- Large video frame dumps stay in-situ to avoid bloating git history.

## Golden experts

| Original | Archive | Verified roots | Bytes | Status |
|---|---|---|---:|---|
| `results/diagnostics/ground_plan_c_final_success` | `archive/golden_experts/ground_plan_c_final_success` | gbi_010,gbi_022 | 504020 | COPIED_VERIFIED |
| `results/diagnostics/ground_dynamic_catch_oracle` | `archive/golden_experts/ground_dynamic_catch_oracle` | gbi_010,gbi_022 | 618031 | COPIED_VERIFIED |
| `results/diagnostics/ground_postcatch_endgrasp_recovery` | `archive/golden_experts/ground_postcatch_endgrasp_recovery` | gbi_010,gbi_022 | 599949 | COPIED_VERIFIED |

## Indexed diagnostics

| Original | Kind | Bytes | Status |
|---|---|---:|---|
| `results/diagnostics/active_recatch_generalization_discretization` | historical | 28124108 | IN_SITU_INDEXED |
| `results/diagnostics/airborne_recatch_search` | historical | 15539945 | IN_SITU_INDEXED |
| `results/diagnostics/ball_impact_handoff_causal_audit` | historical | 282854855 | IN_SITU_INDEXED |
| `results/diagnostics/ball_impact_recoverability` | historical | 18533405 | IN_SITU_INDEXED |
| `results/diagnostics/checkpoint3_skill_audit` | historical | 778938 | IN_SITU_INDEXED |
| `results/diagnostics/contact_aware_recatch_search` | historical | 5672786 | IN_SITU_INDEXED |
| `results/diagnostics/contact_geometry_table_window` | historical | 9160296 | IN_SITU_INDEXED |
| `results/diagnostics/data_expansion` | historical | 275614 | IN_SITU_INDEXED |
| `results/diagnostics/dataset_expansion` | historical | 15161 | IN_SITU_INDEXED |
| `results/diagnostics/free_fall_cylinder_catching` | historical | 17890073 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_capture` | historical | 8115512 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_e_cpp_unload_isolation` | historical | 1260330 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_e_postcapture_stabilize_slide` | historical | 1204992 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_e_stable_lift_contact_manip` | historical | 97992934 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_recatch_collision_aware_feedback` | historical | 6559975 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_recatch_grasp_point_alignment` | historical | 293789940 | IN_SITU_INDEXED |
| `results/diagnostics/freefall_recatch_pose_feedback` | historical | 293551365 | IN_SITU_INDEXED |
| `results/diagnostics/grasp_configuration_migration` | historical | 285155 | IN_SITU_INDEXED |
| `results/diagnostics/gravity_assisted_pivoting_feasibility` | historical | 2122450 | IN_SITU_INDEXED |
| `results/diagnostics/ground_ball_impact_capture_refinement` | historical | 61179 | IN_SITU_INDEXED |
| `results/diagnostics/ground_ball_impact_early_recovery` | historical | 267906 | IN_SITU_INDEXED |
| `results/diagnostics/ground_ball_impact_oracle_interception` | historical | 596387 | IN_SITU_INDEXED |
| `results/diagnostics/ground_scene_validation` | active_keep | 14648 | IN_SITU_INDEXED |
| `results/diagnostics/grpo_adaptive_hybrid` | historical | 5690 | IN_SITU_INDEXED |
| `results/diagnostics/grpo_v2_regression` | historical | 3151489 | IN_SITU_INDEXED |
| `results/diagnostics/handoff_40ms_advantage_audit` | historical | 4057122 | IN_SITU_INDEXED |
| `results/diagnostics/hard_ic_mining` | historical | 3146319 | IN_SITU_INDEXED |
| `results/diagnostics/historical_recatch_recovery` | historical | 14000 | IN_SITU_INDEXED |
| `results/diagnostics/horizontal_recatch_root_cause` | historical | 2800027 | IN_SITU_INDEXED |
| `results/diagnostics/horizontal_recatch_sampling` | historical | 1064649 | IN_SITU_INDEXED |
| `results/diagnostics/k48_counterfactual_credit` | historical | 4864438 | IN_SITU_INDEXED |
| `results/diagnostics/k48_prefix_branching` | historical | 1109628 | IN_SITU_INDEXED |
| `results/diagnostics/k48_prefix_branching_seed42` | historical | 4099459 | IN_SITU_INDEXED |
| `results/diagnostics/k49_action_authority_ablation` | historical | 1348017 | IN_SITU_INDEXED |
| `results/diagnostics/k49_return_axial_slip_mechanism` | historical | 7414877 | IN_SITU_INDEXED |
| `results/diagnostics/k49_slow_reverse_token48` | historical | 76474 | IN_SITU_INDEXED |
| `results/diagnostics/large_tilt_eccentric_grasp_mining` | historical | 54908233 | IN_SITU_INDEXED |
| `results/diagnostics/lateral_force_mechanism_audit` | historical | 13234282 | IN_SITU_INDEXED |
| `results/diagnostics/lateral_offset_falling_recatch` | historical | 10116499 | IN_SITU_INDEXED |
| `results/diagnostics/lateral_recatch_handoff` | historical | 14256330 | IN_SITU_INDEXED |
| `results/diagnostics/legal_recatch_transfer` | historical | 20544567 | IN_SITU_INDEXED |
| `results/diagnostics/mechanism_selection_feasibility` | historical | 272407 | IN_SITU_INDEXED |
| `results/diagnostics/nontrivial_recovery_ic_mining` | historical | 576988 | IN_SITU_INDEXED |
| `results/diagnostics/open_overchase_recatch_dataset` | historical | 4867 | IN_SITU_INDEXED |
| `results/diagnostics/plan_a_ground_scene_divergence` | active_keep | 221794 | IN_SITU_INDEXED |
| `results/diagnostics/plan_a_sliding_pivoting_legality` | active_keep | 1620728 | IN_SITU_INDEXED |
| `results/diagnostics/plan_c_horizontal_downward_end_recatch` | historical | 2926064 | IN_SITU_INDEXED |
| `results/diagnostics/recatch_braking_optimization` | historical | 11294419 | IN_SITU_INDEXED |
| `results/diagnostics/recatch_interface_comparison` | historical | 9662493 | IN_SITU_INDEXED |
| `results/diagnostics/recatch_legality_audit` | historical | 442508 | IN_SITU_INDEXED |
| `results/diagnostics/recovery_ic_expansion` | historical | 123865 | IN_SITU_INDEXED |
| `results/diagnostics/return_to_nominal_pose_before_handoff` | historical | 16530743 | IN_SITU_INDEXED |
| `results/diagnostics/reward_time_handoff_audit` | historical | 771518 | IN_SITU_INDEXED |
| `results/diagnostics/sft_sliding_equal_weight` | historical | 14053123 | IN_SITU_INDEXED |
| `results/diagnostics/sft_sliding_root_cause` | historical | 1916449 | IN_SITU_INDEXED |
| `results/diagnostics/sliding_mechanism_audit` | historical | 500652 | IN_SITU_INDEXED |
| `results/diagnostics/table_drop_fix` | historical | 13921 | IN_SITU_INDEXED |
| `results/diagnostics/top_impact_open_overchase_recatch` | historical | 5492666 | IN_SITU_INDEXED |
| `results/diagnostics/top_impact_progressive_recatch` | historical | 0 | IN_SITU_INDEXED |
| `results/diagnostics/top_impact_recatch_geometry_fix` | historical | 5146724 | IN_SITU_INDEXED |
| `results/diagnostics/validation_skill_classification` | historical | 13327569 | IN_SITU_INDEXED |
| `results/diagnostics/velocity_matched_soft_catch` | historical | 3082386 | IN_SITU_INDEXED |
| `results/diagnostics/vertical_recatch` | historical | 575422 | IN_SITU_INDEXED |
| `results/diagnostics/wrist_tilt_gravity_pivoting` | historical | 2120562 | IN_SITU_INDEXED |

## Recoverability

Restore source code from tag `archive/pre-consolidation-20261021-src`.
Restore golden binaries from `archive/golden_experts/` or original `results/diagnostics/` paths.
