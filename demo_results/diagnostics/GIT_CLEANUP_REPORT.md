# Git cleanup report

No commit was created. Formal GRPO was not started.

## What changed on disk

81 files were deleted (100.0 MB). Checkpoints, raw logs, and videos are ignored, so they do not appear in `git status`. Tracked files that were deleted show as `D` and were not committed:

- `demo_results/diagnostics/AUTONOMOUS_RECOVERY_INTEGRATION.md`
- `demo_results/diagnostics/BALANCED_UNIFIED_RECOVERY_EVALUATION.md`
- `demo_results/diagnostics/OBSERVATION_ROBUSTNESS_CHECK.md`
- `demo_results/diagnostics/PROJECT_CLEANUP_AUTONOMOUS_TRAINING_READY.md`
- `demo_results/diagnostics/RECOVERY_ENTRY_EXIT_HANDOFF_AUDIT.md`
- `demo_results/diagnostics/RECOVERY_POLICY_PRETRAINING_PREPARATION.md`
- `training/autonomous_smoke.py`
- `training/final_39d_balanced_policy.py`
- `training/geometry_recovery_switch.py`
- `training/observation_ablation_check.py`
- `training/observation_robustness_check.py`
- `training/recovery_entry_exit_handoff_audit.py`
- `training/test_current_mainline.py`

`training/final_rl_readiness.py` now points `CKPT` at `demo_results/diagnostics/raw/pre_grpo_sanity/policy_A_reference.pt`. That is the K=35 reference the launcher uses. The previous file was a byte-identical copy and was deleted.

`training/autonomous_episode.py` is still the only modification to a tracked file, and this cleanup did not edit it.

## Active command

```
python -u training/formal_grpo.py --config config/formal_grpo.yaml
```

Policy A, policy B, the train captures, and the rollable validation captures are on disk. Final-test captures were not deleted and are not used by the launcher. `demo_results/archive/` is gone.

## Not done

No commit, no push, and no formal training run.
