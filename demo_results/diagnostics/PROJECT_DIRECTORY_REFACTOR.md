# Project directory refactor

File organization only. No training run, no new demonstrations, no physics, reward, loss, or architecture change.

## 1. Directory structure

Original layout:

```
training/          dataset, training, evaluation, and historical experiment scripts together
demo/              absent
results/           historical diagnostics, raw traces, videos, and checkpoints (5463 files)
data/              current supervised, validation, and GRPO manifests
config/            YAML configs, including the active K48 recovery configs
```

Final layout:

```
training/          63 modules on the dataset, supervised, GRPO, or evaluation import path
demo/              49 relocated scripts plus demo/__init__.py
demo_results/      the previous results/ tree, 5463 files
results/           empty future-output directories: sft/, grpo/v1/, grpo/v2/, evaluations/
data/              unchanged training entry
config/            historical reads retargeted to demo_results/; active training writes stay in results/
```

`results/` contains 0 files. `demo_results/` contains 5463 files, the same count as the tree before the move.

## 2. Moved and renamed source files

Git-tracked moves:

| From | To |
|---|---|
| `training/abc_recovery_validation.py` | `demo/abc_recovery_validation.py` |
| `training/audit_historical_skill_physics.py` | `demo/audit_historical_skill_physics.py` |
| `training/ballistic_large_angle.py` | `demo/ballistic_large_angle.py` |
| `training/ballistic_slip_sufficiency.py` | `demo/ballistic_slip_sufficiency.py` |
| `training/k48_policy.py` | `demo/build_k48_handoff_policy.py` |
| `training/build_train_manifest.py` | `demo/build_train_manifest.py` |
| `training/codebook_prefix.py` | `demo/codebook_prefix.py` |
| `training/compare_recovery_termination.py` | `demo/compare_recovery_termination.py` |
| `training/coverage_all_holds.py` | `demo/coverage_all_holds.py` |
| `training/coverage_counts.py` | `demo/coverage_counts.py` |
| `training/demo_airborne_recapture.py` | `demo/demo_airborne_recapture.py` |
| `training/demo_airborne_recovery.py` | `demo/demo_airborne_recovery.py` |
| `training/demo_ballistic_recovery.py` | `demo/demo_ballistic_recovery.py` |
| `training/demo_ballistic_recovery_comparison.py` | `demo/demo_ballistic_recovery_comparison.py` |
| `training/demo_dynamic_recatch.py` | `demo/demo_dynamic_recatch.py` |
| `training/demo_grav_reposition_comparison.py` | `demo/demo_grav_reposition_comparison.py` |
| `training/demo_grav_reposition_recovery.py` | `demo/demo_grav_reposition_recovery.py` |
| `training/diagnose_k47_closed_loop.py` | `demo/diagnose_k47_closed_loop.py` |
| `training/omega4_throw_test.py` | `demo/diagnose_omega4_throw.py` |
| `training/k48_expand.py` | `demo/expand_k48_handoff_dataset.py` |
| `training/finalize_codebook_views.py` | `demo/finalize_codebook_views.py` |
| `training/grpo_norm_check.py` | `demo/grpo_norm_check.py` |
| `training/handoff_height_audit.py` | `demo/handoff_height_audit.py` |
| `training/launch_entry_stats.py` | `demo/launch_entry_stats.py` |
| `training/legal_recovery_v2_pilot.py` | `demo/legal_recovery_v2_pilot.py` |
| `training/map_ballistic_disturbance.py` | `demo/map_ballistic_disturbance.py` |
| `training/nominal_grasp_creep_root_cause.py` | `demo/nominal_grasp_creep_root_cause.py` |
| `training/release_coverage_v3.py` | `demo/release_coverage_v3.py` |
| `training/release_state_map_omega4.py` | `demo/release_state_map_omega4.py` |
| `training/grav_reposition_v2_viz.py` | `demo/render_grav_reposition.py` |
| `training/replay_all_holds.py` | `demo/replay_all_holds.py` |
| `training/replay_c_subset.py` | `demo/replay_c_subset.py` |
| `training/replay_candidate_codebook.py` | `demo/replay_candidate_codebook.py` |
| `training/replay_codebook_v3.py` | `demo/replay_codebook_v3.py` |
| `training/replay_divergence.py` | `demo/replay_divergence.py` |
| `training/replay_raw_vs_quant.py` | `demo/replay_raw_vs_quant.py` |
| `training/reproduce_center6_large_angle.py` | `demo/reproduce_center6_large_angle.py` |
| `training/revise_codebook.py` | `demo/revise_codebook.py` |
| `training/robustness_probe.py` | `demo/robustness_probe.py` |
| `training/select_append_codebook.py` | `demo/select_append_codebook.py` |
| `training/signed_decode_error.py` | `demo/signed_decode_error.py` |
| `training/summarize_legal_v3.py` | `demo/summarize_legal_v3.py` |
| `training/summarize_replays.py` | `demo/summarize_replays.py` |
| `training/uncovered_release_steps.py` | `demo/uncovered_release_steps.py` |

Untracked moves, same rule:

| From | To |
|---|---|
| `training/audit_controlled_sliding.py` | `demo/audit_controlled_sliding.py` |
| `training/audit_nominal_handoff.py` | `demo/audit_nominal_handoff.py` |
| `training/eval_k48_compare.py` | `demo/compare_k48_policy_evaluations.py` |
| `training/compound_orientation_visual_audit.py` | `demo/diagnose_compound_orientation.py` |
| `training/refactor_repository.py` | `demo/refactor_repository.py` |

`demo/__init__.py` is new so `demo` is an importable package. No source file was deleted.

## 3. Files retained in `training/`

A module stayed when the current dataset-generation, validation, supervised, GRPO, or evaluation path imports it, directly or through that closure. Stable names such as `unified_k48.py` and `unified_k47.py` were not renamed.

Retained because they are the live workflow or are imported by it:

- Entry points: `unified_k48.py`, `rebuild_k48_recovery.py`, `audit_pretrain_data.py`, `verify_handoff_targets.py`
- Shared model, loss, reward, and rollout code: `unified_k47.py`, `action_representation_expansion.py`, `horizon_outcome.py`, `reward_v2.py`, `recovery_runtime.py`, `recovery_success.py`, `replay_core.py`
- Current dataset preparation: `expand_controlled_sliding.py`, `expand_inward_training_data.py`, `export_inward_training_data.py`, `generate_inward_until.py`, `relocate_training_data.py`
- Search modules that those preparation scripts import: `k48_controlled_slide_search.py`, `k48_inward_sliding_search.py`, `privileged_ball_impact_search.py`

The two search modules still write historical traces under `demo_results/diagnostics/raw/`. They remain in `training/` because moving them would break the dataset scripts that import them.

Other retained names (`demo_ballistic_impact.py`, `film_impact_zp6_k33.py`, the airborne-recatch and ablation modules, and the rest of the 63-file package) are in the same import closure. Renaming them would change public imports without making the training command clearer, so their filenames were left as they are.

## 4. Path and import updates

- Imports of relocated modules now use `demo.<module>`. Imports of retained modules stay `training.<module>`.
- Historical artifact paths in Python, YAML, and Markdown now use `demo_results/`. The replacement skipped `data/` so manifest provenance strings are unchanged.
- Active configs `config/unified_k48_recovery_v3.yaml` and `config/unified_k48_recovery_v3_reward_v2.yaml` read the codebook and baseline evaluation from `demo_results/` and write to `results/sft`, `results/evaluations`, and `results/grpo/v1` or `results/grpo/v2`.
- Older K47 and K48 configs still point their outputs at `demo_results/diagnostics/raw/...`. Those are historical runs. The README tells the current workflow to use the two recovery configs above.
- `training/rebuild_k48_recovery.py` still stores rebuilt diagnostic dumps under `demo_results/diagnostics/raw/k48_recovery_v3`. The YAML template it would emit for a future `--manifest` now sets supervised, evaluation, and GRPO outputs to `results/sft`, `results/evaluations`, and `results/grpo/v1` or `results/grpo/v2`. `--manifest` was not run.
- `resolve_stored_path()` in `training/rebuild_k48_recovery.py` opens an existing path as stored. If a relative `results/...` record is missing, it opens the same suffix under `demo_results/`. `expand_controlled_sliding.py` uses that resolver when copying a snapshot. JSONL provenance text was not rewritten.
- `.gitignore` ignores `data/**/*.npz`, `data/**/*.pkl`, `data/**/*.mp4`, and the same raw suffixes plus raw, figure, and video trees under `demo_results/`. It still ignores future logs, checkpoints, and videos under `results/`.

## 5. Verification

| Command | Outcome |
|---|---|
| `python -m compileall -q training demo` | `COMPILE_OK` |
| `python training/unified_k48.py --help` | flags `--check`, `--supervised`, `--eval`, `--eval-output`, `--grpo`, `--config`, `--checkpoint` |
| `python training/rebuild_k48_recovery.py --help` | flags include `--counts`, `--recatch`, `--controlled-slide`, `--inward-reseat`, `--max-search-trials` |
| `python -u training/unified_k48.py --check` | `K48_CHECK` logits 48, nominal_s 12.0; no checkpoint written |
| Dataset row load | supervised 251, validation 14, GRPO 87, nominal-solvable controls 7, train/valid id overlap 0 |
| Path existence for every supervised and validation `continuous_npz` and `snapshot` | 0 missing |
| `episode_tensors` on one `recorded_end` row and one `early_handoff` row | last token 47, residual at that step is zero (`ball_train_v2_0002` length 29, `ball_train_v2_0082` length 51) |
| `recorded_end_includes_handoff(75)` / `(200)` | true / false |
| `grpo_ic_schedule` on the 87 recovery ICs, 261 updates, 3 epochs | 87 distinct ICs; save updates `[87, 174, 261]` |
| GRPO file contents | split `train` only, pool `recovery` only, nominal class `RECOVERY_CANDIDATE` only, none of the 7 controls |
| `import demo.build_k48_handoff_policy`, `demo.compare_k48_policy_evaluations`, `demo.diagnose_compound_orientation` | imported |
| `resolve_stored_path("results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt")` | file exists under `demo_results/` |
| Codebook path in the active config | `demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy` exists |

`training/audit_pretrain_data.py static` was not executed. That mode writes `static.json` into `demo_results/diagnostics/raw/pretrain_audit/`, which is a historical diagnostic directory. The non-writing row, path, and label checks above cover the dataset load. `training/verify_handoff_targets.py` was not rerun; it is a full physics replay of all 258 demonstrations.

Temporary smoke scripts were written under the user temp directory and removed. Nothing was written into `results/`.

## 6. Dataset integrity

| Set | Count |
|---|---:|
| Supervised training rows | 251 |
| Validation demonstrations | 14 |
| GRPO recovery initial conditions | 87 |
| Nominal-solvable controls, excluded from GRPO | 7 |

Handoff labels across the 265 demonstrations: `recorded_end` 153, `early_handoff` 105, `immediate_handoff` 7. Train and validation ids do not overlap. No demonstration file or snapshot was regenerated. Motion-token and residual arrays inside the existing `.npz` files were not rewritten by this refactor.

## 7. Remaining limitations

- Several retained historical modules import names that were already absent from this tree before the move: `training.single_step_token_behavior_audit`, `value_guided_recovery_bootstrap`, `real_impact_recatch_over_slide`, `recatch_over_slide_counterexample`, `action_chunk_recovery_pilot`, `recovery_skill_temporal_persistence_audit`, `long_context_recovery_generalization_audit`, `action_tokenization_recovery_feasibility`, and `unsupervised_trajectory_latent_recovery`. The supervised and GRPO entry point does not import those names. `python -m compileall` compiles the files; executing those particular historical scripts will fail at the missing import, as it did before the move.
- Manifest fields such as `source_snapshot` still say `results/...` on purpose. Runtime opens them through `resolve_stored_path()`. A script that joins `ROOT / source_snapshot` and does not use the resolver will miss the moved file.
- Running `audit_pretrain_data.py` or `verify_handoff_targets.py` writes under `demo_results/diagnostics/raw/pretrain_audit/`. That is the historical audit directory, not `results/`.
- `rebuild_k48_recovery.py --manifest` would still refresh diagnostic dumps inside `demo_results/diagnostics/raw/k48_recovery_v3/`. Its emitted training-output paths now point at `results/`. The command was not run.
- Empty `results/` directories are not tracked by Git. They exist on disk.
- Ignored raw files moved with the directory on disk. Git has no rename record for them because they were ignored.

## 8. Git status

Not committed and not pushed. Tracked source moves are Git renames from `training/` to `demo/`, with content updates where imports and historical paths changed. `README.md`, `.gitignore`, and the historical YAML configs are modified. The active recovery YAML files and the current dataset scripts remain untracked, as they were before this refactor. `results/` has no files. `demo_results/` has 5463 files.

REFACTOR_COMPLETE: YES
SFT_READY_AFTER_REFACTOR: YES
GRPO_READY_AFTER_REFACTOR: YES
REMAINING_ISSUES: historical scripts that import modules never present in this tree still cannot run; stored `results/` provenance strings rely on `resolve_stored_path()`; full `audit_pretrain_data.py static` and `verify_handoff_targets.py` were not rerun because they write into the historical audit directory or replay all demonstrations.
