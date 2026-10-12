# Recovery replay robustness and training readiness

The active policy is still the 36-prototype codebook. K=47 and the revised 56-prototype candidate were evaluated and not loaded. Physics, the basin, and the ±0.25 residual cap were not changed. No training job was started.

The manifest is `results/diagnostics/raw/legal_recovery_v2/train_manifest.jsonl`: 312 TRAIN initial conditions, with snapshot paths, every v2 legal trial, continuous-trajectory paths, contact-loss counts, and replay status. A multi-stage recovery stays one trajectory. The 114 v2 initial conditions with no discovered success are in the manifest and are not dropped.

## 1. The five K=47 replay failures

Repeated restoration of the same snapshot matches exactly: qpos, qvel, ctrl, warmstart, `p_des`, `r_des`, `v_cmd`, `w_cmd`, grip command, time, and phase all differ by 0. Two traces that replay the saved action array through `step_legal` are bitwise identical. The continuous-action 10 s outcome is `SUCCESS_12S` on every exact repeat. The decoded failure repeats as well. The execution path for the saved array and the decoded array is the same function.

`prototype + clipped residual` in float32 is not bit-identical to the saved command. The difference is one float32 ulp, at most 7.5e-9, on 11–19 steps, and it is a small positive bias on `vx`, `vy`, and `vz`.

### A. `0016`, `0020`, `0035` chase

| Case | Saved vs decoded | First state divergence | Contact at that step | Exact replay | Decoded replay |
| --- | --- | --- | --- | --- | --- |
| `ball_train_v2_0016 / chase_lost` | max 3.7e-9, 17 steps | Step 0. `v_cmd` matches. qvel differs by 5.0e-3. Hand position differs by 4.7e-9 | Counts stay 1 / 0 | `SUCCESS_12S` twice, and 3/3 in the later repeat | `DROP_AFTER_HANDOFF` twice |
| `ball_train_v2_0020 / chase_lost` | max 7.5e-9, 11 steps | Step 0 has only a 1e-9 command difference. qpos stays under 1e-6 until step 22 | Step 22: 20/20 versus 21/21. Object velocity differs by 0.061 m/s | `SUCCESS_12S` twice, and 3/3 later | `BASIN_NOT_REACHED` twice |
| `ball_train_v2_0035 / chase_lost` | max 3.7e-9, 19 steps | Step 0. qvel differs by 0.73. Object position differs by 0.24 mm | Counts stay 0 / 0 | `SUCCESS_12S` twice, and 3/3 later | `BASIN_NOT_REACHED` twice |

These three are inside the ±0.25 cap (max L∞ 0.248, 0.239, 0.244). The tokenizer has a prototype within the cap. The long-horizon failure is reproducible and comes from contact-solver sensitivity to a one-ulp action bias, not from a missing token, a different controller path, or an incomplete snapshot.

Random legal perturbations of the saved command, three seeds each:

| Trajectory | Exact 3/3 | σ = 1e-6 | σ = 1e-4 | σ = 1e-3 |
| --- | --- | ---: | ---: | ---: |
| `0016` chase | yes | 3/3 | 3/3 | 2/3 |
| `0020` chase | yes | 1/3 | 1/3 | 3/3 |
| `0035` chase | yes | 3/3 | 3/3 | 3/3 |

`0020` is sensitive to noise at 1e-6. `0016` and `0035` tolerate isotropic noise through 1e-4 and still fail the specific decoded array. The success basin is not a single point, and it does not contain that particular ulp pattern. Classification: physically sensitive, correctly represented, reproducible decoded failure.

### B. `0024` and `0025` chase

Each has one open, no-contact step outside ±0.25. Both are step 7. Grip is −1.

- `ball_train_v2_0024`: action `[0.101, 0.547, 0.143, −0.003, 0, 0, −1]`. Open lateral chase. The excess is `vy`. Candidate L∞ 0.250.
- `ball_train_v2_0025`: action `[0.079, 0.293, 0.316, −0.003, 0, 0, −1]`. Open chase with no contact. The excess is `vy`. Candidate L∞ 0.264. This initial condition was held out of the K=47 fit.

Both saved sequences replay to `SUCCESS_12S` on 3/3 exact repeats and on 3/3 trials at σ = 1e-6, 1e-4, and 1e-3. They are coverage gaps, not fragile rollouts. The residual cap was not raised. A later 56-prototype fit that includes open-chase centers represents both steps (L∞ 0.208 and 0.208) and their decoded replays hold for 10 s. Those two prototypes are justified. The rest of that 56-prototype book is not, because it regresses slides (section 4).

## 2. Robustness

Exact replay is deterministic on the seven probed trajectories: 3/3 `SUCCESS_12S` for `0016`, `0020`, `0035`, `0024`, `0025` chase and for `0021` and `0002` release.

| Trajectory | σ = 1e-6 | σ = 1e-4 | σ = 1e-3 |
| --- | ---: | ---: | ---: |
| `0024` chase | 3/3 | 3/3 | 3/3 |
| `0025` chase | 3/3 | 3/3 | 3/3 |
| `0002` release | 3/3 | 3/3 | 3/3 |
| `0021` release, three loss intervals | 3/3 | 3/3 | 1/3 |
| `0035` chase | 3/3 | 3/3 | 3/3 |
| `0016` chase | 3/3 | 3/3 | 2/3 |
| `0020` chase | 1/3 | 1/3 | 3/3 |

`0021` keeps the full 46-step sequence, including both intermediate contact interruptions. It is stable through 1e-4 and drops on two of three trials at 1e-3. `0020` is the narrow case: exact replay holds, and 1e-6 noise holds on only one of three seeds. Numerical sensitivity here is a property of the contact sequence. It is not treated as invalid physics, and a single success is not treated as robustness.

## 3. All 359 successful v2 trajectories

359 continuous holds on 170 of 284 v2 initial conditions. 123 are slides, 236 are release or chase. 81 contain at least two complete-loss intervals. Counts use the full stored sequence.

| Codebook | Full sequence inside ±0.25 | Physically verified 10 s replay | Representable but replay failed | Not representable |
| --- | ---: | ---: | ---: | ---: |
| K=36 | 106 | 106, all slides, inherited from the search replay of the same quantized actions | 0 | 253 |
| K=47 | 296 | 288 | 8 | 63 |
| Revised K=56, numerical | 355 | not replayed on every trajectory | — | 4, all `chase_lost`, L∞ 0.250–0.253 |

K=47 replay of the 296 in-range sequences: 106 slides inherit the K=36 replay because the extra centers do not change their quantized actions; 42 original releases use the earlier K=47 replay; 148 further sequences were replayed in this pass. 182 of those 190 non-inherited replays hold. The 8 failures:

| Trajectory | Decoded label |
| --- | --- |
| `0016` chase | `DROP_AFTER_HANDOFF` |
| `0087` chase | `DROP_AFTER_HANDOFF` |
| `0020` chase | `BASIN_NOT_REACHED` |
| `0035` chase | `BASIN_NOT_REACHED` |
| `0272` chase | `BASIN_NOT_REACHED` |
| `0068` release | `BASIN_NOT_REACHED` |
| `0235` release | `BASIN_NOT_REACHED` |
| `0250` release | `BASIN_NOT_REACHED` |

No replay failed because the snapshot would not restore. The 8 are the same class as section 1.A: in range, and the decoded rollout does not hold.

The 63 sequences outside K=47 are 46 release/chase and 17 slides. Far steps, counted on the K=47 book, are mostly partial grip (83), open no-contact (24), open small translation (15), and airborne open chase (15). The excess dimensions are `vx` 103, `vy` 27, `vz` 25.

Of the 81 multi-stage holds, 58 are physically verified under K=47. `0021 / release_long` is one of them.

## 4. Codebook comparison

Held-out initial conditions are those whose id number is divisible by 5. They were not used to place new centers. VALIDATION and FINAL_TEST were not used.

| Book | How it was built | Fit sequences in range | Held-out sequences in range | Physical replay |
| --- | --- | ---: | ---: | --- |
| K=36 | Active codebook, unchanged | 86/286 | 20/73 | 106/106 representable slides hold. Releases are outside the cap |
| K=47 | Previous 11 centers, fit on the original release set only | — | 9/10 of the original held-out releases | 288/296 in-range holds. Does not move the 106 slide tokens. `0002` chase holds. `0016` chase does not |
| Revised prefixes fit on all non-held-out successes | 36 preserved, greedy centers from actions outside K=36 | see below | see below | The full 20-center book was replayed on held-out, original, named, and multi-stage sequences (176 trajectories), not on every fit sequence |

Revised prefix coverage of all 359 sequences:

| Prototypes | Extra centers | Sequences inside ±0.25 |
| ---: | ---: | ---: |
| 36 | 0 | 106 |
| 40 | 4 | 246 |
| 44 | 8 | 307 |
| 47 | 11 | 323 |
| 51 | 15 | 345 |
| 56 | 20 | 355 |

The revised 11-center book (323/359) is a different set from the unadopted K=47 (296/359). The first new center is open and nearly still, `[0, 0, 0.01, 0, 0, 0, −1]`, and it covers 1952 of 2438 rounded far actions. Later centers add open chase, an upward open command, and one partial grip near −0.5.

On the 176 revised-book replays: 66 slides still match K=36 and hold; 96 of 108 decoded replays hold; 2 sequences stay outside ±0.25. Twelve decoded replays fail. Five of those are slides whose tokens moved (`0073` and `0085` and `0145` and `0247` `slide_50`, `0210` `slide_45`). `0002` chase, which holds under K=47, is `BASIN_NOT_REACHED` under the revised book. `0024` and `0025` chase, which K=47 cannot represent, hold under the revised book. `0020` and `0035` chase also hold under the revised book. `0016` chase fails under both.

The revised book improves coverage and fixes two coverage gaps, and it regresses verified slides and `0002` chase. It is not adopted. K=47 is not adopted either: 63 sequences are outside the cap and 8 in-range replays do not hold. The active codebook remains K=36.

## 5. Training manifest

`train_manifest.jsonl` has one record per TRAIN initial condition.

- 284 v2 records link the impact snapshot, all six legal trials, continuous npz paths for successes, 10 s and 12 s outcomes, the separate absolute t=12 flag, loss-interval counts, K=36 and K=47 residual scores, K=47 replay status, and K=56 replay status where that replay was run.
- 28 v1 records link the snapshot only. They were not part of this v2 search.
- 170 v2 initial conditions have 359 successes between them. Multiple successes on one initial condition stay separate.
- 114 v2 initial conditions have trials and no success. They remain GRPO initial conditions.

K=47 token and residual arrays for every continuous success are in `*__candidate_k47.npz` next to the continuous file. The continuous file itself was not modified.

## 6. Training readiness

Among the 284 searched v2 initial conditions, under the unadopted K=47 book:

1. 151 have at least one representable, physically verified trajectory.
2. 17 have only continuous successes that K=47 cannot represent.
3. 8 trajectories are representable and fail decoded replay. On 2 initial conditions (`0016`, `0068`) that is the only success. On 6 others a second trajectory is verified.
4. 114 have no demonstrated success. The search budget did not prove them unrecoverable.
5. Contact-preserving slides: 106 verified under K=36 and under K=47, on 68 initial conditions. Release and chase: 182 verified under K=47. Multi-stage recatch: 58 of 81 verified under K=47. The active K=36 book verifies the 106 slides and does not represent the release or repeated-recatch sequences.
6. Neither candidate is suitable for supervised initialization of skill selection. K=36 has no executable release demonstrations. K=47 has 288 executable demonstrations and still fails 8 in-range replays, leaves 63 successes outside the cap, and is not the codebook the policy loads. The revised 56-prototype book regresses slides. GRPO over all 312 snapshots is not ready to start on top of an initialization that cannot execute the release prior.

Recommended configuration, not launched:

- Keep the Transformer, the 39-D observation, `recovery7d_airborne_v3`, K=36, and residual ±0.25.
- Do not load K=47 or K=56.
- When a codebook is eventually adopted, imitate only trajectories whose decoded replay held, and keep each multi-stage trajectory as one sequence.
- Use all 312 snapshots as future GRPO initial conditions, including the 114 with no demonstration.
- Do not start that training on the current books.

REPLAY DIVERGENCE ROOT CAUSE: IDENTIFIED

CODEBOOK READY: NO

EXECUTABLE DEMONSTRATIONS READY: NO

TRAIN IC DATASET READY: YES

POLICY TRAINING HAS NOT BEEN STARTED
