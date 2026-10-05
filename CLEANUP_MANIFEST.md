# Cleanup manifest (CS5478 consolidation)

Inventory-first. Classifications are from repository references (README, Python
imports, retained reports, snapshots, figures, evaluation scripts), **not** from
old Cursor verdicts or filename age.

`KEEP` = needed for the current project or to reproduce retained evidence.
`REMOVE` = unreferenced generated junk; deleted after this manifest.
`UNCERTAIN` = not required for the final ballistic comparison, but a retained
report/script/import may still depend on it, or it documents a negative/superseded
result. **UNCERTAIN items are kept.**

No `rm -rf training/*` or `rm -rf results/*` was used.

---

## Canonical KEEP (current project)

| Path | Class | Reason | References | Canonical replacement |
|---|---|---|---|---|
| `envs/grasp_sim.py`, `envs/xml_build.py`, `config/sim.yaml`, `assets/panda_torque.xml` | KEEP | noslip=1 contact model | README, `apply_solver_from_cfg` | — |
| `assets/scene.xml`, `assets/scene_ballistic_impact.xml` | KEEP | nominal + ballistic scenes | `write_ballistic_impact_scene.py`, demos | — |
| `controllers/nominal.py`, `jacobian_controller.py`, `gripper_controller.py` | KEEP | nominal grasp/lift | `validate_grasp.py` | — |
| `controllers/residual.py` (`RECOVERY4D_W_HY_MAX=3`) | KEEP | 4D action map | RecoveryEnv, all recovery ticks | diagnostic ω=4 is override-only |
| `controllers/rule_based_recovery.py`, `config/rule_based_recovery.yaml` | KEEP | frozen RULE (not current 4D policy) | eval/replay tests | not the comparison mechanism |
| `envs/recovery_env.py`, `envs/physical_recovery.py`, `envs/deterioration.py` | KEEP | RecoveryEnv / pack / drop defs | train/eval (not run this task) | — |
| `envs/observable_obs.py`, `envs/observable_reward.py` | KEEP | 27D obs (not changed) | temporal audit | — |
| `sensors/spatial_tactile.py` | KEEP | spatial tactile | slip sufficiency, obs | — |
| `training/write_ballistic_impact_scene.py`, `impact_ball.py`, `demo_ballistic_impact.py` | KEEP | ballistic family / CENTER-6 | comparison, map | — |
| `training/map_ballistic_disturbance.py` | KEEP | matched impact `run_episode` | comparison prefix | — |
| `training/demo_ballistic_recovery.py` | KEEP | CENTER-6 ZERO `continue_zero`, snaps | comparison ZERO | not stale STOP |
| `training/ballistic_large_angle.py` | KEEP | +120° gravity transfer | slip branch | — |
| `training/ballistic_slip_sufficiency.py` | KEEP | τ=-5 / 0.70 s / brake | slip branch | — |
| `training/omega4_throw_test.py`, `release_state_map_omega4.py`, `demo_dynamic_recatch.py` | KEEP | diagnostic ω=4 recatch | airborne branch | — |
| `training/audit_omega_authority.py` | KEEP | justifies why main bound is 3 and why 4 is diagnostic | OMEGA_AUTHORITY_AUDIT.md | — |
| `training/replay_core.py` | KEEP | freeze / tick_vw | ZERO/slip | — |
| `training/demo_ballistic_recovery_comparison.py` | KEEP | final 3-way comparison | README | — |
| `training/validate_*.py`, `train_recovery.py`, `eval_heldout.py`, `build_recovery_train_set.py` | KEEP | later SAC / interface | README | **do not train now** |
| `results/diagnostics/ballistic_recovery_transfer/raw/snap_early.pkl` | KEEP | EARLY branch IC | recatch, slip | live prefix snap must match |
| `results/diagnostics/dynamic_airborne_recatch/raw/release_map_recatch.json` | KEEP | claimed OPEN 70° / t_close | recatch | — |
| `results/diagnostics/noslip_calibration/` (+ creep audits) | KEEP | why noslip=1 | README | — |
| `results/eval_sets/airborne_offset_eval.npz` | KEEP | held-out (future SAC) | README | — |
| `results/comparison/ballistic_recovery/` | KEEP | this task’s report | README | — |
| `results/videos/ballistic_*.mp4` | KEEP | presentation videos | README | — |

---

## training/*.py (72 scripts)

All listed files below exist. Broad experimental scripts are **not** deleted
because they are imported by retained diagnostics (`demo_teleport_recovery_state`
is a hub) or they reproduce KEEP reports.

| Path | Class | Reason / deps |
|---|---|---|
| `demo_ballistic_recovery_comparison.py` | KEEP | this task |
| `demo_ballistic_impact.py`, `map_ballistic_disturbance.py`, `audit_ballistic_cliff.py`, `write_ballistic_impact_scene.py` | KEEP | ballistic family |
| `demo_ballistic_recovery.py`, `ballistic_large_angle.py`, `ballistic_slip_sufficiency.py` | KEEP | ZERO + slip authority |
| `demo_dynamic_recatch.py`, `omega4_throw_test.py`, `release_state_map_omega4.py`, `audit_omega_authority.py`, `dynamic_throw_phase.py` | KEEP | airborne / ω audit |
| `demo_airborne_recovery.py`, `demo_airborne_recapture.py` | KEEP | recatch helpers (`make_parent_sim`, masks) |
| `gravitational_reposition_primitive_v2.py`, `demo_grav_reposition_*.py`, `audit_grav_reposition_self_arrest.py`, `grav_reposition_v2_viz.py` | KEEP | gravitational slip evidence |
| `noslip_calibration.py`, `box_vs_cylinder_creep.py`, `nominal_grasp_creep_root_cause.py`, `contact_creep_sensitivity_audit.py`, `vertical_slip_contact_mechanics_audit.py` | KEEP | contact model |
| `replay_core.py`, `impact_ball.py`, `impact_visualization_utils.py` | KEEP | shared |
| `demo_teleport_recovery_state.py` | KEEP | imported by ~20 scripts (constants, helpers). Teleport *experiments* are not the final comparison, but deleting breaks imports. |
| `validate_grasp.py`, `validate_hold.py`, `validate_reach.py`, `validate_recovery_env.py`, `validate_observable_*`, `validate_tactile_ex_estimator.py`, `validate_optional_ablation.py` | KEEP | entry points |
| `test_recovery4d_interface.py`, `test_observable_*`, `test_mainline_replay.py`, `test_tactile_no_gt_leak.py` | KEEP | interface tests |
| `train_recovery.py`, `eval_heldout.py`, `build_recovery_train_set.py`, `collect_recovery_states.py` | KEEP | future SAC; not run this task |
| `calibrate_detector.py`, `log_contact.py` | KEEP | deterioration/detector infrastructure |
| `observable_temporal_audit.py`, `observable_signal_audit.py`, `demo_observable_audit.py` | KEEP | 27D evidence |
| `demo_dynamic_deterioration.py` | KEEP | deterioration demo |
| `wrist_sweep_recapture.py`, `wrist_recovery_cycle_audit.py`, `inward_slide_return_audit.py`, `reposition_regrasp_audit.py`, `active_regrasp_authority_audit.py`, `recovery_false_positive_audit.py`, `recenter_observability_audit.py` | UNCERTAIN | superseded-by-noslip **interpretations**, but scripts reproduce kept reports/artifacts (`results/REPOSITORY_CLEANUP.md` already marked KEEP scripts). |
| `coupled_ic_authority.py`, `demo_offset_task_consequence.py`, `offset_construct.py` | UNCERTAIN | offset/teleport IC construction; not the ballistic comparison. Imported/used by audits. |
| `impact_demo_core.py`, `visualize_impact_recovery.py`, `write_impact_scene.py`, `calibrate_impact_demo.py`, `demo_impact_recovery.py` | UNCERTAIN | old impact demo family; quarantined in `OLD_IMPACT_QUARANTINE.md` but still referenced there and by tests. |
| `test_impact_demo_ball_from_t0.py`, `test_impact_ball_flyin.py`, `test_impact_demo_playback.py` | UNCERTAIN | tests for old/new impact viz |
| `search_delayed_drop.py`, `audit_continue_lift.py` | UNCERTAIN | documents invalid STOP / delayed-drop searches; not imported by comparison. Kept so ZERO semantics remain auditable. |
| `measure_nominal_grasp_timeline.py` | UNCERTAIN | timeline helper |

---

## results/

| Path | Class | Reason |
|---|---|---|
| `diagnostics/ballistic_impact_noslip1/` | KEEP | CENTER-6 family, cliff, map, quarantine note |
| `diagnostics/ballistic_recovery_transfer/` | KEEP | EARLY snap, slip/large-angle reports |
| `diagnostics/dynamic_airborne_recatch/` | KEEP | ω=3/4 audit + claimed recatch JSON |
| `diagnostics/gravitational_reposition_primitive_v2/`, `grav_reposition_*` | KEEP | slip primitive |
| `diagnostics/noslip_calibration/`, `nominal_grasp_creep_root_cause/`, `box_vs_cylinder_creep/`, `contact_creep_sensitivity/` | KEEP | noslip freeze evidence |
| `diagnostics/airborne_recapture/`, `airborne_nontrivial_recovery/` | KEEP | earlier airborne constructions (privileged); still cited |
| `diagnostics/observable_*`, `tactile_ex_estimator/`, `recenter_observability/` | KEEP | obs/reward/tactile |
| `diagnostics/_SUPERSEDED_BY_NOSLIP.md`, `inward_slide_return/` | UNCERTAIN | historical creep-as-authority; keep marker + raw |
| `diagnostics/wrist_sweep_recapture/`, `wrist_recovery_cycle/`, `reposition_regrasp/`, `active_regrasp/`, `recovery_false_positive/` | UNCERTAIN | archive; keep |
| `eval_sets/airborne_offset_eval.npz` | KEEP | future eval |
| `eval_sets/impact_severity_four.npz` | UNCERTAIN | quarantined mass bug; keep file, do not reuse |
| `logs/`, `figures/` | UNCERTAIN | historical CSVs/stills; not proven unused by reports |
| `videos/ballistic_*.mp4` | KEEP | this task |
| `videos/impact_recovery/` (if present) | UNCERTAIN | old viz |
| `REPOSITORY_CLEANUP.md` | KEEP | prior inventory (2026-10-03); this file supersedes deletion policy |
| `comparison/ballistic_recovery/` | KEEP | this task |

---

## REMOVE (deleted)

Generated caches only. No scientific logs, snaps, or training scripts.

| Path | Class | Reason | Refs | Replacement |
|---|---|---|---|---|
| `**/__pycache__/` | REMOVE | bytecode | none | recreated by Python |
| `**/*.pyc` | REMOVE | bytecode | none | — |
| `results/videos/_frames_*` | REMOVE | ffmpeg staging dirs from video encode | none | regenerated if videos are re-encoded |

---

## Deletion rule applied

- UNCERTAIN: **kept**.
- Experimental training scripts: **kept** (imports + report reproduction).
- No obsolete STOP/SSR/noslip=0 *conclusions* were deleted; they remain labeled superseded.
- No SAC checkpoints were present to delete.
