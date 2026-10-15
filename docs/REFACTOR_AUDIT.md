# Second-round refactor audit

Formal packages (`controllers/`, `envs/`, `sensors/`, `models/`, `data_generation/`, `training/`, `evaluation/`, `visualization/`) do not import `experiments/` or `legacy/`. Experiment scripts call the formal modules. No dynamic import was left in those packages as a stand-in for that dependency.

## Wrong dependencies

- `envs/ballistic_impact.py` called `extract_contacts` from the vertical-slip audit.
- Training runtime called grasp seating, seated offsets, fixed skill tapes, JSON conversion, action k-means, and the GRPO residual likelihood from experiment scripts.
- `data_generation` called `step_legal`, representability, slide/chase laws, and the horizontal recatch helpers from experiment files.
- `training/unified_k48.py` and `training/unified_k47.py` called the privileged search only to reach `step_legal`.
- `training/formal_grpo.py`, `training/pre_grpo_sanity.py`, and `training/action_representation_expansion.py` were historical trainers living in the formal package.

## How they were removed

Each shared function now has one body. Experiment files import that body.

- Finger-cylinder contact rows and `pad_span_hand` are in `envs/contacts/frames.py`. Geom ids, contact frame, normal force, and the summary fields are the previous extractor.
- Seating, `apply_offset`, `apply_spec`, and `target_hand` are in `envs/initialization/grasp_seat.py`. The horizontal offset grid stays in the freefall experiment.
- The fixed 7D skill tapes are in `controllers/recovery/skill_schedules.py`. They are action sequences, not learning-rate schedules.
- `SIGMA`, `gaussian_logprob`, and `advantages` are in `training/grpo_math.py`.
- Slide and privileged chase commands are in `data_generation/strategies/recovery_laws.py`.
- The dataset builder's horizontal recatch helpers are in `data_generation/strategies/recatch_oracle.py`. Vertical `follow_early` stays in `vertical_recatch_search.py`.
- `step_legal` and representability are imported from `envs/step_legal.py` and `data_generation/oracles/representability.py`.

## Files split

`training/unified_k48.py` keeps argument parsing and re-exports. The bodies are:

- `training/train_sft.py` — supervised loss and the supervised loop
- `training/train_grpo.py` — GRPO log-probability and the GRPO loop
- `training/dataset.py` — episode tensors and collate
- `training/trainer.py` — progress, reward-order, and checkpoint-update helpers
- `training/k48_config.py` — shared paths and the 4 s limit

K47 was not merged into K48. `training/unified_k47.py` still owns the codebook, `decode_action`, and the historical K47 loop.

## Shared implementations extracted

`envs/contacts/frames.py`, `envs/initialization/grasp_seat.py`, `controllers/recovery/skill_schedules.py`, `training/grpo_math.py`, `data_generation/jsonable.py`, `data_generation/policy_rollout.py`, `models/historical/k33_checkpoint.py`, `models/historical/k36_checkpoint.py`, `models/historical/k36_policy.py`, `data_generation/action_kmeans.py`, `data_generation/initial_conditions/natural_specs.py`, `data_generation/validation/impact_labels.py`, `data_generation/strategies/recovery_laws.py`, `data_generation/strategies/recatch_oracle.py`.

## Duplicate implementations removed from the old sites

The old function bodies were deleted where the formal module replaced them. `build_policy` callers use `models/k48_policy.py`. `load_models` callers use `models/historical/k36_checkpoint.py`.

## Legacy kept, and why

- `experiments/legacy/formal_grpo.py`, `pre_grpo_sanity.py`, and `action_representation_expansion.py` reproduce the K36 GRPO pilot and the representation audit.
- `training/unified_k47.py` still loads the K47 codebook that K48 expands.
- Recatch experiment scripts keep their diagnostic scans. They import the stable helpers.
- `experiments/legacy/demo/` remains the historical reproduction scripts.

## demo_results

Before the rename, `demo_results/` had 5464 files and 570015212 bytes. SHA-256 of every file is `results/archive/demo_results_sha256.txt`. After `os.replace` onto the same volume, the file count, total size, relative paths, and hashes matched. The top-level directory is gone. The K47 epoch-40 checkpoint at `results/archive/demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt` loads; its token head is `(47, 64)`.

Code, YAML, and docs now use `results/archive/demo_results/`. Files under `data/` were not rewritten.

## Regression

`python -u tests/test_physics_regression.py` printed `PHYSICS_REGRESSION_OK`.

Compared on the fixed tapes `ball_train_v2_0070` (controlled slide and vertical recatch), `ball_train_0017` (inward), and `ref_s6_p10_a0_e0` (horizontal recatch):

- action versus the saved tape, maximum error 0
- observation versus the saved tape, maximum error 0
- qpos, qvel, contact geom/normal/side, and controller phase and commands agree across two replays of the same tape
- vertical nominal outcome is `LOST_GRASP`; the recovery tape ends `SUCCESS_12S`
- TABLE_DROP latch boundaries and reward routing pass
- K48 checkpoint `results/grpo/v2/checkpoint_u0783.pt` loads, logits are 48-wide, and two forwards match

A CPU smoke test with seed 0 ran `supervised_loss` and `step_logprob` backward plus one Adam step each. It did not write a checkpoint. The Adam group is all 31 parameters. Comparable quantities are the loss graph and the parameter step, not a trained weight file.

`data/recatch`, `data/vertical_recatch`, `data/regrasp`, and the sliding manifests were not opened for writing.

`python -u scripts/render_video.py --out results/videos/vertical_recatch_regression.mp4` wrote 571 frames: 45 recovery frames repeated 6 times, then 301 frames of the official 12 s nominal hold at 25 fps.

## Remaining

- `training/unified_k47.py` still contains the historical K47 training loop next to the codebook API.
- Contact and qvel checks compare two replays of a tape. The saved npz files store actions and 39-D observations, not a separate qvel series.
- Current diagnostic directories under `results/diagnostics/` were left in place because live scripts still point at them.
