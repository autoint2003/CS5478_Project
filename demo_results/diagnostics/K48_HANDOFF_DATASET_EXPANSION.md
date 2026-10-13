# K=48 handoff dataset expansion

No policy was trained. The K=47 codebook, the epoch-40 checkpoint, the 296-row migrated manifest, the 312 TRAIN snapshots, and the validation and final-test partitions were not modified.

## Recovery limit

`T_recovery_max` is configurable. The nominal window stays 12 s from the handoff instant, and simulator time is not reset. Saved action lists were not extended with invented commands.

| Limit | Saved sequences that fit | Additional sequences versus 2.4 s |
| --- | --- | --- |
| 2.4 s | 358 | — |
| 4 s | 359 | the 122-step `ball_train_v2_0177 / slide_45` |
| 6 s | 359 | none |
| 8 s | 359 | none |

That 122-step episode succeeds if handoff is forced at step 120 and also if all 122 commands run. At 4 s and above the handoff is the recorded end, at simulator recovery time 2.44 s, and nominal elapsed is 12.000 s. The 1345 unsuccessful searches have no action array, so a longer cap cannot replay them.

Selected maximum: **4.0 s**. It is the shortest tested limit that keeps the 122-step demonstration intact and still has room for a 0.20 s pause. Six and eight seconds add no saved trajectory. The time penalty remains `1{SUCCESS_12S} - 0.05 * recovery_seconds`. A success at 4.0 s scores 0.80, which is still above every failure.

## Historical attempts

The legal search has 1704 rows. Only 359 have saved actions, and every one of those was already a continuous hold. Under the migrated physical definition, 296 of them are inside the ±0.25 residual and are the current supervised set, including the eight former in-range replay failures. The other 63 continuous holds remain outside the residual and were not admitted.

| Historical search label | Saved actions | Replayed here |
| --- | --- | --- |
| `SUCCESS_12S` | 359 | already in the migrated set; 296 representable |
| `BASIN_NOT_REACHED` | 0 of 521 | unresolved, no action array |
| `DROP_DURING_RECOVERY` | 0 of 662 | unresolved, no action array |
| `DROP_AFTER_HANDOFF` | 0 of 162 | unresolved, no action array |

Newly recovered demonstrations beyond the 296: **0**. Distinct TRAIN ICs stay **153**. Failed and missing attempts are listed in `demo_results/diagnostics/raw/k48_handoff/unresolved_searches.json` and `pause_trials.jsonl`. They were not deleted.

## Pause command

A pause is a legal 20 ms recovery step, not a frozen simulator. The all-zero action is not a pause: grip action 0 maps to -8 N·m, between open (+2 N·m) and secure (-18 N·m). The pause command is

`[0, 0, 0, 0, 0, 0, previous grip]`

Translation and wrist commands are zero. The grip channel is copied from the previous command so the fingers are not opened. `step_legal` still integrates contact and motion.

Seeds are 4 demonstrations from distinct initial conditions in each of the five mechanisms (20 parents, 19 initial conditions). Each parent is paused at 25%, 50%, and 75% of its commands for 0.04, 0.08, 0.12, and 0.20 s. Pause time counts toward the 4 s budget. The original suffix is then replayed. A trial is admitted only if that execution succeeds and the K=47 decoding of the executed commands also succeeds.

| Pause result | Count |
| --- | --- |
| Trials | 240 |
| Admitted | 238 |
| `DROP_AFTER_HANDOFF` | 2 |

The two failures are `ball_train_v2_0250 / release_long` at the 25% location for 0.12 s and 0.20 s. No new continuation was searched after those failures. Admitted pauses come from the same 153 initial conditions; they do not add coverage. Each stored pause trajectory has the observations recorded during that execution, not the unperturbed history.

## HANDOFF token

Tokens 0–46 are the original prototypes. Token 47 is HANDOFF. Selecting it does not decode a residual, does not call `step_legal`, and hands off to the unchanged nominal controller. Forced handoff still occurs at the configured limit. `decode_action(token=47)` raises. Loading the epoch-40 checkpoint into the 48-wide head copies the 47 motion rows exactly and sets the new row to zero. A mismatched head still raises. Residual loss and residual log-probability are masked on HANDOFF.

## Handoff labels

Every admitted trajectory was tested at 25%, 50%, and 75% of its commands. A candidate is positive only when nominal lift completes and the grasp lasts 12 s from that instant. The earliest positive candidate that also survives a further 0.04 s grip-preserving pause is the supervised cut. A timeout is not given a HANDOFF label.

| Candidate outcome | Count |
| --- | --- |
| Candidates tested | 1602 |
| `SUCCESS_12S` | 310 |
| `DROP_AFTER_HANDOFF` | 1284 |
| `LOST_GRASP` | 8 |
| Trajectories with a verified early HANDOFF | 190 |
| Of which the selected cut survived the 0.04 s pause | 190 |
| Trajectories kept to the recorded end, with no HANDOFF token | 344 |

Early-handoff recovery durations are 0.12–1.20 s, median 0.28 s. No saved sequence reaches the 4 s cap, so there is no forced-timeout demonstration. The 344 recorded-end rows are the verified full command lists, not fabricated HANDOFF labels.

Historical early HANDOFF: 93 of 296. Pause early HANDOFF: 97 of 238.

## Supervised set

`demo_results/diagnostics/raw/k48_handoff/supervised_targets.jsonl`

| Quantity | Count |
| --- | --- |
| Trajectories | 534 |
| Historical | 296 |
| Pause-perturbed | 238 |
| Distinct TRAIN initial conditions | 153 |
| Independent source demonstrations | 296, of which 20 were used as pause parents |
| Early HANDOFF | 190 |
| Recorded-end, no HANDOFF token | 344 |
| GRPO initial conditions | 312 |

Mechanism totals, historical plus pause:

| Mechanism | Rows | Historical weight sum | Pause weight sum |
| --- | --- | --- | --- |
| `contact_preserving_slide` | 109 | 1 | 0.25 |
| `airborne_chase` | 129 | 1 | 0.25 |
| `release_regrasp` | 141 | 1 | 0.25 |
| `sliding` | 93 | 1 | 0.25 |
| `repeated_recatch` | 62 | 1 | 0.25 |

Within each mechanism, historical rows share weight 1 equally across source initial conditions, and pause rows share an extra 0.25 the same way. One initial condition with many pauses cannot take the historical budget. HANDOFF is one step at the end of 190 trajectories, so it does not outnumber motion steps.

One checked early-handoff row, `ball_train_v2_0001 / slide_50`, has 13 motion steps plus token 47. The handoff observation matches the saved post-prefix state, the previous action at that step is the last motion command, and the handoff residual is zero.

## Checks

| Check | Result |
| --- | --- |
| 47 motion prototypes unchanged | codebook file not rewritten; expanded head copies those rows |
| HANDOFF does not emit a motion | forced argmax on token 47 gives a zero command and is not decoded |
| Nominal timing on `0177` | 12.000 s after both the 2.4 s forced cut and the 2.44 s recorded end |
| Reward order at a 4 s success | 0.98 > 0.80 > 0 > -0.01 |
| Pause execution | 238 decoded successes, 2 physical failures retained |
| History after a pause | observations stored from the paused rollout |
| Early-handoff replay | each of the 190 cuts was a physical nominal success, and each survived a 0.04 s pause |
| Unsuccessful candidates | 1292 negative handoff tests and 2 pause failures kept |
| Weights | historical sums are 1; pause sums are 0.25 |
| 48-token loss and log-probability | `python -u training/unified_k48.py --check` passed, then gradients were cleared |
| K=47 `--check` after the head change | passed; frozen 288 and migrated 296 still load |
| Sealed splits | none in the 534 rows or the 312 GRPO rows |
| CPU/CUDA | `ball_train_v2_0015` still diverges at step 24. That split is reported, not treated as a robust label |

## Unresolved

- 1345 search attempts, including 521 old basin misses, have no saved actions and cannot be replayed.
- 63 continuous holds remain outside ±0.25 and are not demonstrations.
- 344 successful trajectories have no verified early handoff. Their supervised target is the recorded command list, without a HANDOFF token.
- Two pause trials on `ball_train_v2_0250 / release_long` drop after handoff.
- `ball_train_v2_0015` is contact-sensitive to CPU versus CUDA. Device-specific outcomes are not counted as robust.

## Commands

These start training. They were not run. The epoch-40 file is read and expanded in memory. Checkpoints are written under `demo_results/diagnostics/raw/unified_k48/`.

Non-training check:

```powershell
python -u training/unified_k48.py --check
```

```bash
python -u training/unified_k48.py --check
```

Supervised training from the expanded epoch-40 head:

```powershell
python -u training/unified_k48.py --supervised --config config/unified_k48.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k48.py --supervised --config config/unified_k48.yaml --checkpoint demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt
```

GRPO pilot, 20 updates, after that supervised checkpoint exists:

```powershell
python -u training/unified_k48.py --grpo --config config/unified_k48.yaml --checkpoint demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k48.py --grpo --config config/unified_k48.yaml --checkpoint demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

Longer GRPO run, 312 updates, one TRAIN initial condition per update:

```powershell
python -u training/unified_k48.py --grpo --config config/unified_k48_long.yaml --checkpoint demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

```bash
python -u training/unified_k48.py --grpo --config config/unified_k48_long.yaml --checkpoint demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt
```

The dataset and the 48-token trainer are ready for that supervised run. The open physical gaps are the searches that were never saved, the 63 commands outside the residual bound, and the CPU/CUDA contact split on `ball_train_v2_0015`.
