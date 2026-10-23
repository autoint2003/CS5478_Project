# Repository refactor map (2026-10-23)
Checkpoint before these moves: `121324c9e57dde27c12fd220dfcb5517e22707ee` on `checkpoint/pre-six-branch-refactor-20261023`.
Behavior of the Precision scheduler was not retuned. Paths below are the organizational moves.
## Precision Oracle
| Old path | New path |
|---|---|
| `data_generation/oracles/precision/precision_types.py` | `data_generation/oracles/precision/state.py` |
| `data_generation/oracles/precision/precision_handoff.py` | `data_generation/oracles/precision/handoff.py` |
| `data_generation/oracles/precision/unified_precision_oracle.py` | `data_generation/oracles/precision/state_machine.py` |
| `data_generation/oracles/precision/precision_audit.py` | `data_generation/oracles/precision/replay_audit.py` |
| `data_generation/oracles/precision/axis_aware.py` | `data_generation/oracles/precision/corrections/residual_directions.py` |
| `data_generation/oracles/precision/sign_aware_axial.py` | `data_generation/oracles/precision/corrections/axial_sign.py` |
| `data_generation/oracles/precision/orientation_planner.py` | `data_generation/oracles/precision/corrections/orientation.py` |
| `data_generation/oracles/precision/primitives.py` | `data_generation/oracles/precision/corrections/contact_primitives.py` |
| `data_generation/oracles/precision/precision_dx.py` | `data_generation/oracles/precision/corrections/lateral.py` |
| `data_generation/oracles/precision/endpoint_only_axial_roll.py` | `data_generation/oracles/precision/mechanics/open_axis.py` |
| `data_generation/oracles/precision/finger_surface_support.py` | `data_generation/oracles/precision/mechanics/finger_surface.py` |
| `data_generation/oracles/precision/finger_supported_vertical.py` | `data_generation/oracles/precision/mechanics/finger_vertical.py` |
| `data_generation/oracles/precision/force_slip_control.py` | `data_generation/oracles/precision/mechanics/dual_loop.py` |
| `data_generation/oracles/precision/precision_slip.py` | `data_generation/oracles/precision/mechanics/axial_sliding.py` |
| `data_generation/oracles/precision/precision_dx_support_transfer.py` | `data_generation/oracles/precision/mechanics/support_transfer.py` |
| `data_generation/oracles/precision/support_aware_rotation.py` | `data_generation/oracles/precision/mechanics/protected_rotation.py` |
| `data_generation/oracles/precision/rotation_speed_profile.py` | `data_generation/oracles/precision/mechanics/rotation_braking.py` |
| `data_generation/oracles/precision/contact_modes.py` | `data_generation/oracles/precision/mechanics/support_verification.py` |
| `data_generation/oracles/precision/stable_grasp.py` | `data_generation/oracles/precision/mechanics/grasp_stabilization.py` |
| `data_generation/oracles/precision/reverse_supported_rotation.py` | `data_generation/oracles/precision/mechanics/reverse_rotation.py` |
| `data_generation/oracles/precision/precision_goals.py` | split into `entry_conditions.py` and `residuals.py` |
| `data_generation/oracles/precision/precision_dz.py` | split into `corrections/axial_common.py`, `axial_inward.py`, `axial_outward.py` |
| `scripts/generate_plan_c_experts.py` `evaluate_handoff_from_state` | `evaluation/handoff_from_state.py` |
| `data_generation/oracles/precision/precision.zip` and `unified_precision_oracle.py.bak_pre_v23` | `experiments/precision/frozen_baselines/` |

## Scripts moved out of `scripts/`

### `experiments/expert_freeze/`

- `experiments/expert_freeze/__init__.py`
- `experiments/expert_freeze/build_expert_dataset.py`
- `experiments/expert_freeze/consolidation/__init__.py`
- `experiments/expert_freeze/consolidation/_apply_naming_refactor.py`
- `experiments/expert_freeze/consolidation/audit_token_representability.py`
- `experiments/expert_freeze/consolidation/freeze_experts.py`

### `experiments/stage2_mining/`

- `experiments/stage2_mining/__init__.py`
- `experiments/stage2_mining/_fixup_stage2_mining_k49_and_report.py`
- `experiments/stage2_mining/_fixup_stage2_pilot_k49_residuals.py`
- `experiments/stage2_mining/audit_stage2_plan_b_ic_sources.py`
- `experiments/stage2_mining/expand_stage2_bc_and_readiness_v2.py`
- `experiments/stage2_mining/finalize_stage2_abc_dataset.py`
- `experiments/stage2_mining/finalize_stage2_abc_final_v2.py`
- `experiments/stage2_mining/generate_stage2_abc_pilot.py`
- `experiments/stage2_mining/generate_stage2_abc_recovery_fullchain_v1.py`
- `experiments/stage2_mining/mine_stage2_abc_nontrivial.py`
- `experiments/stage2_mining/render_stage2_recovery_vs_nominal.py`

### `experiments/plan_b_table_free/`

- `experiments/plan_b_table_free/__init__.py`
- `experiments/plan_b_table_free/audit_plan_a_gravity_slip_910220.py`
- `experiments/plan_b_table_free/confirm_plan_ab_table_free_3x.py`
- `experiments/plan_b_table_free/develop_plan_b_tail_pivot_invert_910220.py`
- `experiments/plan_b_table_free/diagnose_plan_b_k49_divergence.py`
- `experiments/plan_b_table_free/diagnose_plan_b_posta_plan_a_correction.py`
- `experiments/plan_b_table_free/diagnose_plan_b_replay_contradiction.py`
- `experiments/plan_b_table_free/diagnose_plan_b_table_free.py`
- `experiments/plan_b_table_free/migrate_plan_ab_table_free.py`
- `experiments/plan_b_table_free/optimize_plan_b_slide_pivot_910220.py`
- `experiments/plan_b_table_free/recover_plan_b_table_free.py`
- `experiments/plan_b_table_free/regenerate_plan_b_b0_b1_b2.py`
- `experiments/plan_b_table_free/screen_plan_b_precision_candidates.py`
- `experiments/plan_b_table_free/update_unified_after_plan_b_recovery.py`
- `experiments/plan_b_table_free/update_unified_experts_table_free.py`

### `experiments/plan_c_research/`

- `experiments/plan_c_research/__init__.py`
- `experiments/plan_c_research/_probe_slow_orient_gbi014.py`
- `experiments/plan_c_research/_render_gbi014_recatch_video.py`
- `experiments/plan_c_research/generate_plan_c_experts.py`
- `experiments/plan_c_research/investigate_plan_c_contact_surface.py`
- `experiments/plan_c_research/investigate_plan_c_postlift_correction.py`
- `experiments/plan_c_research/investigate_plan_c_postlift_phase2.py`
- `experiments/plan_c_research/regression_plan_c_golden.py`
- `experiments/plan_c_research/research_plan_c_sidewall_gbi014.py`
- `experiments/plan_c_research/research_plan_c_sidewall_gbi014_handoff.py`
- `experiments/plan_c_research/run_plan_c_lift_to_precision_v2.py`
- `experiments/plan_c_research/scale_plan_c_sidewall_experts.py`

### `experiments/rendering_comparisons/`

- `experiments/rendering_comparisons/__init__.py`
- `experiments/rendering_comparisons/rebuild_mechanism_videos_v2.py`
- `experiments/rendering_comparisons/render_k49_vs_continuous_diag.py`
- `experiments/rendering_comparisons/render_mode_b_full_arm_reori_video.py`
- `experiments/rendering_comparisons/render_plan_abc_vs_nominal.py`

### `experiments/precision_probes/`

- `experiments/precision_probes/__init__.py`
- `experiments/precision_probes/_finish_vertical_slip_handoff.py`
- `experiments/precision_probes/_probe_abc_entry_heights.py`
- `experiments/precision_probes/_probe_mode_b_deep_tilt.py`
- `experiments/precision_probes/_probe_post_up_samples.py`
- `experiments/precision_probes/_probe_tilt_to_down.py`
- `experiments/precision_probes/_probe_to_up_brake_series.py`
- `experiments/precision_probes/_probe_vertical_handoff.py`
- `experiments/precision_probes/_probe_vertical_slip_params.py`
- `experiments/precision_probes/_split_precision_oracle_v23.py`
- `experiments/precision_probes/audit_action_vocab.py`

## Documents

Kept in `docs/`: `ARCHITECTURE.md`, `ORACLE_STATE_MACHINE_SPEC.md`, `REPRODUCIBILITY.md`.

- `docs/ARCHIVE_MANIFEST.md` → `experiments/docs_historical/ARCHIVE_MANIFEST.md`
- `docs/DATASET_SCHEMA.md` → `experiments/docs_historical/DATASET_SCHEMA.md`
- `docs/FINAL_CONSOLIDATION_REPORT.md` → `experiments/docs_historical/FINAL_CONSOLIDATION_REPORT.md`
- `docs/GROUND_SCENE_OUTCOMES.md` → `experiments/docs_historical/GROUND_SCENE_OUTCOMES.md`
- `docs/MODULE_ORGANIZATION.md` → `experiments/docs_historical/MODULE_ORGANIZATION.md`
- `docs/NAMING_REFACTOR_REPORT.md` → `experiments/docs_historical/NAMING_REFACTOR_REPORT.md`
- `docs/ORACLE_IMPLEMENTATION_GAP_AUDIT.md` → `experiments/docs_historical/ORACLE_IMPLEMENTATION_GAP_AUDIT.md`
- `docs/ORACLE_REMAINING_UNSUPPORTED.md` → `experiments/docs_historical/ORACLE_REMAINING_UNSUPPORTED.md`
- `docs/ORACLE_RESTORE_DELIVERABLES.md` → `experiments/docs_historical/ORACLE_RESTORE_DELIVERABLES.md`
- `docs/ORACLE_SOURCE_CHANGES.md` → `experiments/docs_historical/ORACLE_SOURCE_CHANGES.md`
- `docs/PRE_CONSOLIDATION_CHECKPOINT.txt` → `experiments/docs_historical/PRE_CONSOLIDATION_CHECKPOINT.txt`
- `docs/RENAME_MAP.md` → `experiments/docs_historical/RENAME_MAP.md`

Course PDFs: `docs/coursework/` → `coursework/`.

## Shared helpers pulled out of historical audits

Active Plan C no longer imports `experiments/`.

| Symbol | Now defined in | Historical callers |
|---|---|---|
| `sample_kinematics` | `evaluation/hand_frame_kinematics.py` | `experiments/audits/plan_a_mechanism_audit.py` re-exports it |
| `decode_token` | `training/action_vocab_k49.py` | same audit module re-exports it |
| `evaluate_handoff_from_state` | `evaluation/handoff_from_state.py` | Plan C research scripts |
| Plan C golden `replay_plan_c` | `evaluation/plan_c_golden_replay.py` | `experiments/expert_freeze/consolidation/freeze_experts.py` delegates to it |

## New active modules

- `data_generation/expert_pipeline/` — six branch ids, continuation dispatch, provenance filter
- `evaluation/plan_c_golden_replay.py` — frozen Plan C golden replay
- `scripts/validate_dataset.py`
