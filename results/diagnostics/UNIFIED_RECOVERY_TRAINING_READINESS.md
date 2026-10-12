# Unified recovery training readiness

The selected codebook is the existing 47-prototype table. Tokens 0–35 are the K=36 prototypes, unchanged. Tokens 36–46 are the eleven appended centers. No center was moved. VALIDATION and FINAL_TEST were not used. No impact was generated, the legal search was not repeated, and training was not started.

The book is `results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy`. It was checked with `np.array_equal` against `policy_B_init.pt` on rows 0–35.

## Why this book, and not a larger one

K=36 represents 106/359 successes, all slides, on 68 initial conditions. K=47 represents 296/359 on 153 initial conditions and physically holds on 288 of those 296. Dropping any one of the eleven appended centers loses distinct initial conditions: from 1 (centers 5 and 9) up to 71 (center 0, the open near-still action). A smaller subset is not coverage-neutral.

The 63 out-of-range successes were inspected, with priority on the 19 v2 initial conditions that still have no physically verified discrete demonstration. That set is the 17 whose successes are outside the cap, plus `0016` and `0068`, which are inside the cap and fail decoded replay.

Most of those out-of-range trajectories miss by one step. The missing actions are open-gripper commands: small signed `vy`, a different `vz`, contact-free lateral chase, and the same open commands again after a failed recatch. Placing a new prototype on any of those single far steps also becomes the nearest token for verified actions. The quietest case, `ball_train_v2_0239 / chase_lost` at L∞ 0.305, still captures 2 verified steps. The largest, `ball_train_v2_0284 / release_long`, captures 135. Every such append was rejected. New tokens cannot cover those initial conditions without changing a replay that already holds.

The 56-prototype refit from the previous pass is not used. It moved tokens under verified slides and dropped five of them, and it dropped `ball_train_v2_0002 / chase_lost`.

## Selected prototypes 36–46

| Token | vx | vy | vz | wx | wy | wz | grip | Role |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 36 | −0.037 | −0.014 | 0.015 | −0.002 | 0 | 0 | −1 | Open, nearly still |
| 37 | −0.046 | −0.154 | −0.429 | −0.004 | 0 | 0 | −1 | Open, small −vy, upward |
| 38 | −0.106 | 0.297 | 0.019 | −0.003 | 0 | 0 | −1 | Open, +vy |
| 39 | 0.242 | −0.030 | 0.139 | 0 | 0 | 0 | −1 | Open, +vx |
| 40 | −0.003 | 0.036 | −0.705 | −0.004 | 0 | 0 | −1 | Open, strong upward |
| 41 | −0.236 | −0.338 | 0.119 | −0.002 | 0.001 | 0 | −1 | Open, −vx and −vy |
| 42 | 0.576 | −0.006 | −0.283 | −0.004 | 0 | 0 | −1 | Contact-free lateral, upward |
| 43 | 0.160 | 0.347 | −0.204 | −0.004 | 0 | 0 | −1 | Open, +vy and upward |
| 44 | −0.569 | 0.233 | −0.052 | −0.003 | 0 | 0 | −1 | Open, −vx and +vy |
| 45 | 0.075 | −0.291 | −0.241 | 0 | 0 | 0 | +1 | Closed reposition |
| 46 | 0.548 | 0.144 | 0.061 | −0.004 | 0 | 0 | −1 | Open lateral chase |

Residual cap remains ±0.25.

## Representability and physical replay

| | Trajectories inside ±0.25 | Decoded 10 s holds | Distinct ICs with a verified demonstration |
| --- | ---: | ---: | ---: |
| K=36 | 106 | 106 | 68 |
| Selected K=47 | 296 | 288 | 151 |

K=47 coverage of the 359 continuous successes:

| Mechanism group | Inside the cap and replayed to a 10 s hold |
| --- | ---: |
| Contact-preserving slide (fewer than two loss intervals) | 61 |
| Slide with repeated contact loss | 45 |
| Release / regrasp | 92 |
| Airborne or lateral chase | 77 |
| Repeated recatch on a release or chase | 13 |

Slide total 106. Release and chase total 182. Repeated-loss trajectories verified: 58 of 81. The 106 K=36 slide replays are unchanged under K=47, because every verified slide step still quantizes to tokens 0–35. There is no K=36 regression.

Eight in-range replays do not hold. Three were traced to a one-ulp contact-solver split: `0016`, `0020`, and `0035` chase. Five more failed the same decoded-replay test without a new token being added: `0068` release, `0087` chase, `0235` release, `0250` release, `0272` chase. Original labels stay `SUCCESS_12S`. None of the eight is a supervisory target.

63 successes stay outside ±0.25, on 17 initial conditions that therefore have no verified discrete demonstration. 114 v2 initial conditions have no continuous success. 27 of the 151 verified initial conditions have both a verified slide and a verified release. Those pairs are separate trajectories, not extra initial conditions.

## Fragile trajectories

`results/diagnostics/raw/legal_recovery_v2/fragile_trajectories.jsonl` lists the eight. Continuous files are unchanged.

Perturbation checks from the previous pass, three seeds, are the robustness evidence used here. No new sweep was run.

| Trajectory | Exact replay | σ = 1e-6 | σ = 1e-4 | σ = 1e-3 | Supervisory target |
| --- | --- | ---: | ---: | ---: | --- |
| `0016` chase | 3/3 | 3/3 | 3/3 | 2/3 | no |
| `0020` chase | 3/3 | 1/3 | 1/3 | 3/3 | no |
| `0035` chase | 3/3 | 3/3 | 3/3 | 3/3 | no |
| `0021` release, three loss intervals | 3/3 | 3/3 | 3/3 | 1/3 | yes |
| `0002` release | 3/3 | 3/3 | 3/3 | 3/3 | yes |

`0020` is the narrow case. `0021` is stable through 1e-4 and is kept, with the full 46-step episode. Its sample weight is not increased to hide the 1e-3 misses.

## Multi-stage sequences

`ball_train_v2_0021 / release_long` is in the supervised view as one episode: 3 loss intervals, mechanism `repeated_recatch`, K=47 replay `SUCCESS_12S`. The stored arrays are the original 39-D observations, the legal 7-D actions, and the token/residual file beside them. A brief bilateral touch with the gripper still open is not a terminal label. The other 57 verified repeated-loss trajectories are stored the same way. The 23 repeated-loss successes that are outside the cap or fragile stay in the IC manifest and are not given successful action targets.

## Dataset views

Source manifest: `results/diagnostics/raw/legal_recovery_v2/train_manifest.jsonl`.

| View | File | Records |
| --- | --- | ---: |
| GRPO initial conditions | `grpo_initial_conditions.jsonl` | 312 snapshots, including 28 v1 and 114 unsolved v2 |
| Supervised targets | `supervised_targets.jsonl` | 288 verified trajectories on 151 initial conditions |
| Fragile, not targets | `fragile_trajectories.jsonl` | 8 |

Each supervised row has `sample_weight = 1 / count(mechanism)`, so the five mechanism groups have equal total weight. Rows were not copied. Failed search trials stay on the IC manifest and have no action target.

## Decoder check

`config/unified_recovery_codebook.yaml` names this book, cap ±0.25, and `build_policy`. A CPU forward with K=47 produced logits `(4, 47)`, residual `(4, 7)` inside ±0.25, and action `(4, 7)`. The action is `clamp(prototype[token] + residual, -1, 1)` with residual `0.25 * tanh(head)`, the same formula `build_policy` uses for training and for a later physical rollout.

`ContinualPolicy` and `FuturePolicy` still emit 36 logits. `formal_grpo.load_models` was given this 47-row table and raised `expected K=36 initialization`. The pilot checkpoint directory and `policy_A_reference.pt` / `policy_B_init.pt` were not written.

## First experiment, not launched

1. Supervised initialization for 40 epochs on `supervised_targets.jsonl`, weighted by `sample_weight`, one full sequence per trajectory. History is the observation and previous action through the whole episode, including repeated recatch. Held-out TRAIN initial conditions are those whose id number is divisible by 5. They are not in the fit of this check; they may still appear as GRPO initial conditions later.
2. Physical rollout of that initialization on the held-out TRAIN initial conditions only. Record the autonomous token sequence, whether a nominal handoff occurred, and the unchanged 10 s / 12 s criteria. Do not read VALIDATION or FINAL_TEST.
3. Twenty GRPO updates on all 312 snapshots, KL coefficient 0.05, the mean sequence log-probability already used in the launcher. Output directory must not be `formal_grpo_pilot`.
4. Compare that short run with the sealed pilot and with a K=36 head on the same 106 slide targets. Success is physical hold after handoff, not the presence of several mechanisms in the supervised file.

The K=36 launcher cannot run this experiment until it is pointed at `build_policy` and this codebook. That edit is not part of this task.

## Remaining blockers

- 17 v2 initial conditions have continuous successes that no append-only center can represent without retargeting a verified trajectory.
- 8 in-range demonstrations are physically sensitive and are not supervised targets.
- 114 v2 initial conditions have no demonstration. They remain GRPO starts.
- The active GRPO launcher still constructs a 36-logit head. The prepared configuration is separate and was rejected by that launcher on purpose.

TRAIN IC DATASET READY: YES

CODEBOOK READY: YES

EXECUTABLE DEMONSTRATIONS READY: YES

TRAINING CONFIGURATION READY: YES

POLICY TRAINING HAS NOT BEEN STARTED
