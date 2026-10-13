# K=47 project finalization

No policy optimization was run. `python -u training/unified_k47.py --check` is the validation entry point. Supervised training, closed-loop evaluation, and GRPO were not launched.

## Bugs found and fixed

1. `training/formal_grpo.py` `load_models` required the initialization prototype to be exactly `(36, 7)` and always constructed `FuturePolicy`. A K=47 table could not be loaded. The loader now takes K from the checkpoint prototype, which must be `(K, 7)`, and requires residual cap 0.25. A checkpoint that still has `logit35_w` or `logits_old.weight` loads as the historical 36-token `FuturePolicy` and is rejected if the prototype is any other width. Any other checkpoint loads with `build_policy`. `_reject_head_mismatch` raises `incompatible token head` on an unexpected key, a missing key, or a shape mismatch. A 36-wide logit matrix is not copied into the 47-wide head.

2. The first supervised-loss draft multiplied the whole batch by the mean of the row weights. That scale does not change the relative gradient inside a batch, so the five mechanism weights had no effect. The loss is now the mean over trajectories of `sample_weight * (token cross-entropy + residual MSE)`. Each trajectory's own weight is used. Dividing by the sum of the weights would cancel that balance, so the loss does not do so. Padded steps stay out of both terms. The previous action at step 0 is zero and later previous actions are the preceding executed commands.

The K=36 GRPO objective in `training/pre_grpo_sanity.py` was not rewritten. Its retention KL still pads a 35-wide reference by one logit. The K=47 path does not use that pad.

### K=47 GRPO objective

For a sampled recovery step,

`log pi(a) = log softmax(logits)[token] + log Normal(pre; residual_mean, sigma=0.05 I)`.

`pre` is the residual before the ±0.25 clip. The prototype add and the ±1 command clip are deterministic. `sigma` is the existing `SIGMA` of 0.05.

A rollout that hits `rollout_cap_steps` without a drop and without a nominal handoff is truncated. Truncated members are removed before the group mean and standard deviation. They are not given a failure return. If fewer than two members remain, or the kept returns have standard deviation below `1e-8`, the advantage is zero and the policy surrogate is skipped. Otherwise the advantage is `(return - mean) / std` on the kept returns only. The return is 1 for `SUCCESS_10S` or `SUCCESS_12S` and 0 for every other completed label, including a drop during recovery, a drop after handoff, and a basin that never hands off.

The surrogate is the clipped ratio used by the existing GRPO pilot, with clip 0.2: the ratio is the mean per-step log-probability under the current policy divided by the behavior log-probability recorded at sampling. The KL term is the categorical KL between the current and reference token distributions on the first recorded step of each completed member. The two logit vectors must have the same width. The residual density is not inside the KL, matching the historical token-only retention KL.

## Modified source files

- `training/formal_grpo.py`: K-derived loader and incompatible-head rejection. The historical `FuturePolicy` path remains. `formal_grpo.py` is still the K=36 launcher and still steps ball-impact rollouts with `step_v3`. Do not point it at the K=47 table.
- `training/unified_k47.py`: new K=47 check, supervised, evaluation, and GRPO entry point. Decode is `clip(prototype[token] + clip(residual, -0.25, 0.25), -1, 1)` through `step_legal`.
- `config/unified_k47.yaml`: pilot configuration. Read by the new entry point.
- `config/unified_k47_long.yaml`: 120-update configuration. Not launched.
- `config/unified_recovery_codebook.yaml`: comment now points at `config/unified_k47.yaml`. The trainer does not read this file.
- `README.md`: rewritten for the K=47 project.
- `docs/HISTORICAL_K32_README.md`: previous root README, unchanged in content.

No MuJoCo model, contact parameter, reward, entry geometry, nominal controller, handoff rule, or success criterion was edited. `codebook_selected_k47.npy` was not rewritten. Tokens 0–35 still match `policy_B_init.pt`, and tokens 36–46 still match `candidate_extra.npy`. Shape `(47, 7)`. No training label was added or removed.

## Tests

| Command | Result |
| --- | --- |
| `python -m py_compile training/unified_k47.py training/formal_grpo.py` | exit 0 |
| `python -u training/unified_k47.py --help` | `--check`, `--supervised`, `--eval`, `--grpo`, `--config`, `--checkpoint` |
| `python -u training/unified_k47.py --check` | first run failed: the loss batch picked `ball_train_v2_0021/slide_50` instead of `release_long`. The selector was fixed. |
| `python -u training/unified_k47.py --check` | `CHECK_OK supervised 288` with each of the five mechanism weights summing to 1.0. After the per-row weight fix the same command passed again: `qpos_err 0.0`, `label_without_handoff BASIN_NOT_REACHED`. |

That check does all of the following and then stops. It does not call `opt.step`.

- Imports the training modules and checks the codebook shape and identity.
- Requires 288 supervised trajectories, 151 distinct ids, `supervisory_target` on every row, and no overlap with `fragile_trajectories.jsonl`.
- Requires 312 GRPO rows and rejects ids or snapshot paths containing `validation`, `final_test`, `ball_validation`, or `ball_final`.
- Requires `ball_train_v2_0021/release_long` to be one 46-step episode with three complete-loss intervals, previous action 0 at step 0, and later previous actions equal to the preceding command.
- Builds the causal Transformer, checks logits `(2, 3, 47)` and residual `(2, 3, 7)` inside ±0.25, saves a temporary state, reloads it, and deletes the file.
- Rejects a 36-wide `logits.weight` on the 47-token model.
- Reloads `policy_B_init.pt` as a 36-token `FuturePolicy` whose old head still has width 35.
- Computes one supervised loss on a real batch, including the 46 unmasked steps of `release_long`, and runs `loss.backward()` followed by `zero_grad`.
- Checks a synthetic group `[1, 1, 0]` plus a truncated member: the truncated advantage is 0 and the group is not skipped. An identical-return group is skipped with advantage 0.
- Computes one K=47 step log-probability and backpropagates it, then clears the gradient.
- Restores `ball_train_v2_0021.pkl`, checks qpos error, decodes one legal action, and executes one `step_legal` step. A basin miss is `BASIN_NOT_REACHED`, not `SUCCESS_12S`.

The temporary files were `demo_results/diagnostics/raw/legal_recovery_v2/_k47_check/roundtrip.pt` and `bad_head.pt`. Both were deleted, and the directory was removed. No directory `demo_results/diagnostics/raw/unified_k47/` was created.

## Dataset and codebook integrity

- Codebook: `demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy`, `(47, 7)`. Prefix equals the K=36 prototype. Suffix equals the eleven selected centers.
- Supervised demonstrations: 288 verified executable trajectories, 151 initial conditions.
- GRPO pool: 312 TRAIN snapshots (284 v2 and 28 v1). Held-out closed-loop evaluation uses the 56 v2 ids whose numeric suffix is divisible by 5. Those ids stay in the GRPO pool. VALIDATION and FINAL_TEST are not in either manifest.
- Eight in-range physical failures remain in `fragile_trajectories.jsonl` and are excluded from supervision.

## Repository cleanup manifest

| Path | Action | Reason |
| --- | --- | --- |
| `docs/HISTORICAL_K32_README.md` | copied from the previous root README | keep the K=32 / 39-D description available |
| `README.md` | replaced | current K=47 entry point |
| `demo_results/diagnostics/raw/legal_recovery_v2/_k47_check/` | created during `--check`, then deleted | temporary checkpoint round-trip, not experimental evidence |

Nothing under `demo_results/`, historical checkpoints, contact traces, continuous trajectories, or the 312 snapshots was deleted. No historical artifact was renamed.

## README

The root README now describes the ball-impact recovery task, the causal Transformer, the 39-D observation, the 7-D legal action, K=47 with residual ±0.25, the split between 312 initial conditions and 288 demonstrations, and the four-stage pipeline. It states that training has not been run. Commands match `training/unified_k47.py --help` and the keys in the two yaml files.

## Remaining limitations

- 63 continuous successes are outside ±0.25 and are not demonstrations.
- Eight in-range decoded replays fail because the contact solver responds to a float32 reconstruction on the order of 1e-9 to 1e-8. The codebook was not changed to chase those cases.
- 114 v2 initial conditions have no successful search. They remain GRPO starts.
- A 20-update pilot visits the first 20 manifest rows, one initial condition per update, four rollouts each. A 120-update run visits the first 120. Covering all 312 once takes 312 updates. Starting the long run as a new process begins again at row 0.
- `--checkpoint` is both the initial policy and the KL reference. Passing the pilot checkpoint moves the KL anchor to the pilot. Passing the supervised checkpoint keeps the anchor and does not resume pilot weights.
- The printed label of a step-cap stop is `BASIN_NOT_REACHED`. The separate truncated flag is what keeps that rollout out of the advantage group.
- Success still requires the nominal handoff and the physical continuation. `step_v3` parks the ball and is not used by this entry point.

## Commands

PowerShell and Bash use the same invocation. Run them from the repository root. Do not run the training commands until this report has been read.

Supervised initialization. Seed 1, device `cuda`, 40 epochs, batch 8, learning rate 3e-4. A checkpoint is written after every epoch.

```powershell
python -u training/unified_k47.py --supervised --config config/unified_k47.yaml
```

```bash
python -u training/unified_k47.py --supervised --config config/unified_k47.yaml
```

Output: `demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt`.

Closed-loop evaluation on the 56 held-out TRAIN v2 snapshots. The policy samples its own argmax token and mean residual. It does not replay teacher actions. Cap 120 recovery steps, then the real handoff and nominal continuation.

```powershell
python -u training/unified_k47.py --eval --config config/unified_k47.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --eval --config config/unified_k47.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

Output: `demo_results/diagnostics/raw/unified_k47/eval/heldout_train.json`.

GRPO pilot. All 312 TRAIN snapshots are in the manifest. This budget executes the first 20. Group 4, learning rate 3e-5, KL coefficient 0.05, clip 0.2, seed 1, rollout cap 80, checkpoint every 5 updates. The KL reference is the supervised checkpoint.

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

Output: `demo_results/diagnostics/raw/unified_k47/grpo_pilot/checkpoint_u0020.pt`.

Longer run, only after the pilot has been inspected. 120 updates, rollout cap 120, checkpoint every 12 updates. This command also starts from the supervised checkpoint, so the KL reference stays there.

```powershell
python -u training/unified_k47.py --grpo --config config/unified_k47_long.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k47.py --grpo --config config/unified_k47_long.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

Output: `demo_results/diagnostics/raw/unified_k47/grpo_long/checkpoint_u0120.pt`.

## Recommended execution order

1. `python -u training/unified_k47.py --check`
2. Supervised initialization.
3. Closed-loop evaluation of `checkpoint_epoch_0040.pt` on the held-out TRAIN subset. Inspect `heldout_train.json`.
4. The 20-update GRPO pilot. Inspect its labels.
5. The 120-update configuration only if that inspection is acceptable.

K=47 LOADER FIXED: YES
TRAINING PIPELINE VALIDATED WITHOUT OPTIMIZATION: YES
DATASET INTEGRITY VERIFIED: YES
README UPDATED: YES
SAFE CLEANUP COMPLETED: YES
TRAINING COMMANDS PREPARED: YES
POLICY TRAINING EXECUTED: NO
