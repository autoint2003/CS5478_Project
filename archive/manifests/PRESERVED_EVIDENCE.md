# Preserved evidence (validated experts and mechanisms)

Rollback commit: `d3223833647a8dc7107dc7898bec0b7abef55a47`  
Branch: `cleanup/ground-scene-prep`  
Critical file hashes: `archive/manifests/ROLLBACK_POINT.json`

All paths below remain on disk under `results/` (gitignored). None were deleted or rewritten.

## Distinction of success labels

| Label | Meaning | Treat as full expert? |
|---|---|---|
| `SUCCESS_NEW` | Full legal recovery + HANDOFF + nominal hold with ≤10° / ≤12 mm | Yes |
| `SUCCESS_12S` | Horizon/nominal hold without full NEW pose gate | No |
| `CATCH_HANDOFF` | Catch + handoff without full NEW | No |
| Partial capture | Temporary bilateral contact | No |

Do not merge independent-root counts with repeated exact replays of the same root.

## Plan A — controlled sliding / pivoting / return

| Artifact | Path | Replay |
|---|---|---|
| Sliding position correction v1 | `results/expert_generation/sliding_position_correction_v1/` | Yes (legal 7D / K49) |
| Return-aware sliding K49 | `results/expert_generation/return_aware_sliding_k49/` | Yes |
| Slow return position refinement | `results/expert_generation/k49_slow_return_position_refinement/` | Yes |
| Extended pose recovery | `results/expert_generation/extended_k49_k50_pose_recovery/` | Yes |
| Wrist-tilt gravity pivoting diagnostics | `results/diagnostics/wrist_tilt_gravity_pivoting/` | Mechanism evidence |
| Axial slip mechanism | `results/diagnostics/k49_return_axial_slip_mechanism/` | Mechanism evidence |
| Generators | `scripts/generate_k49_sliding_recatch_dataset.py`, `experiments/sliding/pilot_*.py` | Entry points retained |

## Plan B — release–recatch / lost grasp

| Artifact | Path | Replay |
|---|---|---|
| Lost-grasp recovery | `results/expert_generation/k49_lost_grasp_recovery/` | Yes |
| In-recovery recatch | `results/expert_generation/k49_in_recovery_recatch/` | Yes |
| Dynamic recatch experts | `results/expert_generation/k49_dynamic_recatch_experts/` | Yes |
| Type3 capture–brake | `results/expert_generation/k49_type3_capture_brake/` | Yes |
| Capture–brake diagnostics | `results/expert_generation/k49_capture_brake_diagnostics/` | Diagnostics |

## Plan C — active / ball-impact / freefall recatch

| Artifact | Path | Notes |
|---|---|---|
| Ball-impact dynamic recovery | `results/expert_generation/k49_ball_impact_dynamic_recovery/` | Retain; elevated-table scene |
| Real ball impact low-speed | `results/expert_generation/real_ball_impact_low_speed_recatch/` | Retain |
| Horizontal endcatch | `results/expert_generation/k49_horizontal_endcatch_recatch/` | 0 Type3 NEW reported historically |
| Horizontal-offset freefall | `results/expert_generation/horizontal_offset_freefall_recatch/` | Retain |
| Large chase dynamic | `results/expert_generation/k49_large_chase_dynamic_recovery/` | Retain |
| Lateral-force full-pose recovery | `results/expert_generation/lateral_force_full_pose_recovery/` | Retain |
| Lateral-force grasp-loss | `results/expert_generation/lateral_force_grasp_loss_recovery/` | Retain |
| Freefall grasp-point / CPP / unload | `results/diagnostics/freefall_*` | Mechanism audits; do not relabel partial as SUCCESS_NEW |
| Plan C horizontal downward end-recatch | `results/diagnostics/plan_c_horizontal_downward_end_recatch/` | 0 DIRECT_SUCCESS_NEW; 3× SUCCESS_12S ≠ NEW |
| Top-impact / open-overchase | `results/diagnostics/top_impact_*`, `open_overchase_recatch_dataset/` | Geometry evidence |

## Elevated-table benchmark reproduction

| Asset | Path |
|---|---|
| Official scene | `assets/scene.xml` (table top ≈ 0.40 m) |
| TABLE_DROP latch | `envs/outcomes/table_drop.py` |
| Height gate | `envs/outcomes/physical_recovery.py` (`TABLE_DROP = 0.44`) |
| Regression | `tests/test_table_drop.py`, `results/diagnostics/table_drop_fix/` |

## Demo archive

Historical demo trees live at `results/archive/demo_results/` (pre-existing; covered by `results/` gitignore).

## Exact replay requirements preserved

For each full expert directory, keep:

- Exact initial conditions / snapshots
- Complete legal action trajectories
- Controller / MuJoCo state
- Success verification records
- Root identity
- Replay scripts / generator entry points
- Representative videos (on disk; not versioned)
