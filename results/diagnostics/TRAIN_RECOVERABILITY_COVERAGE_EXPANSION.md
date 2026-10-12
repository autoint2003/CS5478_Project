# Train recoverability coverage expansion

Privileged search on the six unsolved train episodes, then a frozen K=35 group-of-4 audit of all 12 train episodes. No GRPO update, no SAC, no DPO, no training on the new traces, and no final-test rollout. The 39-D observation, K=35 codebook, ±0.25 residual cap, `recovery7d_airborne_v3` bounds, geometry entry, nominal basin, handoff, success evaluator, and controller gains are unchanged.

A case is `VERIFIED_RECOVERABLE` only after live entry, a legal recovery, the nominal basin (orientation ≤ 0.117 rad, pinch ≤ 3.03 mm, relative speed ≤ 0.020 m/s, persistence 0.100 s), an explicit handoff, and 10 s of nominal-only hold. The 12 s hold is reported beside it. Finite search failure stays `SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE`.

Slip and mis-seat were not searched again. Their privileged labels are the readiness-menu results (`restoration_feedback` on `slip_a`–`slip_d` and `seat_a`).

## Search budget

Two recorded budgets, both started from the saved live geometry-entry captures in `results/diagnostics/raw/final_rl_readiness/ics/`.

1. Known schedules once each (`wrist_reseat`, `gravity_inward`, `recatch_open20`), then a shared open-pulse along the hand-frame pinch error: 8 pulse lengths and speeds, plus up to 2 speed refinements of the best pulse. Cap 16 trials. Impact used 13. Airborne used 11.
2. A shared chase that stays open at one commanded speed until the pinch error stops falling, then closes and allows one short nudge. Six settings, cap 12. Each impact used 6. Each airborne episode stopped on the second trial.

Raw logs: `results/diagnostics/raw/train_recoverability/expansion.json` and `chase.json`.

## Main table

| episode | family | privileged status | mechanism found | nominal basin reached? | handoff? | 10 s stable? | 12 s stable? | K35 representable? | frozen-policy success / 4 | return span | mixed GRPO group? | failure mode |
|---|---|---|---|---|---|---|---|---|---:|---:|---|---|
| imp_a | impact | SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE | — | no | no | no | no | no new trace | 0/4 | 0.073 | no | basin not reached; object retained |
| imp_b | impact | SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE | — | no | no | no | no | no new trace | 0/4 | 0.189 | no | basin not reached; object retained |
| imp_c | impact | SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE | — (policy member saved) | no under the privileged menu | no under the privileged menu | no under the privileged menu | no under the privileged menu | policy trace yes, max L2 3.7e-9, 0 clipped | 1/4 | 0.987 | yes | privileged menu: basin not reached |
| imp_d | impact | SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE | — | no | no | no | no | no new trace | 0/4 | 0.216 | no | basin not reached; object retained |
| slip_a | slip | VERIFIED_RECOVERABLE | restoration_feedback (prior) | yes | yes | yes | yes | not re-saved | 2/4 | 1.112 | yes | — |
| slip_b | slip | VERIFIED_RECOVERABLE | restoration_feedback (prior) | yes | yes | yes | yes | not re-saved | 1/4 | 1.029 | yes | — |
| slip_c | slip | VERIFIED_RECOVERABLE | restoration_feedback (prior) | yes | yes | yes | yes | not re-saved | 0/4 | 0.043 | no | — |
| slip_d | slip | VERIFIED_RECOVERABLE | restoration_feedback (prior) | yes | yes | yes | yes | not re-saved | 1/4 | 1.009 | yes | — |
| seat_a | mis-seat | VERIFIED_RECOVERABLE | restoration_feedback (prior) | yes | yes | yes | yes | not re-saved | 0/4 | 0.027 | no | — |
| seat_b | mis-seat | SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE | — (prior menu) | no | no | no | no | no new trace | 0/4 | 0.160 | no | prior menu did not hand off |
| air_a | airborne | VERIFIED_RECOVERABLE | chase at 1.8 m/s, close when pinch stops falling | yes | 6.456 s | yes | yes | no | 0/4 | 0.059 | no | — |
| air_b | airborne | VERIFIED_RECOVERABLE | chase at 1.8 m/s, close when pinch stops falling | yes | 5.736 s | yes | yes | no | 0/4 | 0.242 | no | — |

The group-of-4 uses the current NumPy sampler. Member `g` is seeded with `2000 + 17*g + sum(ord(c) for c in episode id)`. A group is useful when successes are mixed or the return span is at least 0.5. Returns are shaping plus 1.0 on a 10 s success.

## Verified train recoverability by family

| family | privileged verified | legal recovery under the full success rule |
|---|---|---|
| slip | 4/4, unchanged | 4/4 privileged |
| mis-seat | 1/2 (`seat_a`), unchanged | 1/2 privileged |
| impact | 0/4 in both new budgets | 1/4: frozen-policy member 2 of `imp_c` is `SUCCESS_12S` |
| airborne | 2/2, new | 2/2 privileged |

Impact entries are slow bilateral grasps, a few millimetres off in hand +x, near the nominal orientation. Wrist reseating, inward sliding, and the fixed open-finger schedule leave the pinch at the entry value. A short open translation along the pinch error moves it. The closest joint sample in the pulse budget is `imp_b` at 3.48 mm pinch, 0.020 rad, and 0.004 m/s (`refine_pulse_0.25_2.0_1_x1.25`). That sample did not persist for 0.100 s with a closed grip, and the episode did not fall. Chase trials later printed pinch minima of 1.8–2.8 mm on `imp_b`, `imp_c`, and `imp_d`, while the jointly closest sample on those trials stayed at 6.7–13.8 mm. Recatch occurred on the best impact trials. Handoff did not.

`imp_c` is recoverable inside the frozen representation. Replaying group member 2 reproduces `SUCCESS_12S`, handoff at 3.640 s, pinch 0.58 mm at the last recovery sample. The 36 actions use tokens 0, 14, 17, 29, 32, 33, and 34. Reconstruction error is numerical noise and no residual component exceeds ±0.25. The trace is `results/diagnostics/raw/train_recoverability/success_imp_c.npz`. The privileged menu on the same capture still ends `BASIN_NOT_REACHED`.

Airborne entries are bilateral grasps about 6.01 mm off in hand +z, at z ≈ 0.596 m. The same chase law, with no prelift and a 1.8 m/s open command, reaches the basin on both train episodes and holds through 12 s of nominal control. The first airborne setting (1.2 m/s) drops. Handoff is at 6.456 s on `air_a` and 5.736 s on `air_b`.

## K=35 audit of the new successes

| trace | steps | max L2 | clipped residual components | outside the frozen representation |
|---|---:|---:|---|---|
| `success_imp_c.npz` | 36 | 3.7e-9 | 0 | no |
| `success_air_a.npz` | 10 | 0.247 | 1 | yes, step 1, vy |
| `success_air_b.npz` | 9 | 0.326 | 1 | yes, step 1, vy |

On `air_a` step 1 the legal action has vy = 0.518. The nearest token is 34, whose vy is 0.021, so the residual is +0.497. On `air_b` step 1, vy = 0.597 and the residual is +0.576. Both exceed ±0.25. The hand-z and grip channels of those steps sit on token 34. No token was added. The clipped action was not re-executed.

## Frozen-policy baseline and GRPO signal by family

Deterministic baseline from the readiness study, unchanged: 3/12 train successes, all slip (`slip_a`, `slip_b`, `slip_d`).

This seeded group-of-4 rerun:

| family | episodes with a success | useful groups | returns |
|---|---|---|---|
| slip | 3/4 | 3/4 (`slip_a` 2/4, `slip_b` 1/4, `slip_d` 1/4) | spans 1.01–1.11 on the useful groups; `slip_c` span 0.043, all drops |
| mis-seat | 0/2 | 0/2 | `seat_a` span 0.027, all drops; `seat_b` span 0.160, one basin miss and three drops |
| impact | 1/4 | 1/4 (`imp_c`) | `imp_c` returns 0.078, 0.073, 1.060, 0.182. Labels: drop after handoff, drop during recovery, `SUCCESS_12S`, basin not reached. The other impact spans are 0.073–0.216 |
| airborne | 0/2 | 0/2 | `air_a` span 0.059, four drops; `air_b` span 0.242, one basin miss and three drops |

Useful groups are `slip_a`, `slip_b`, `slip_d`, and `imp_c`. Impact now has one mixed group. Airborne groups remain all failures. `seat_a` is all failures under this seed; the earlier torch-sampled audit had one success there.

## Saved demonstrations

Not used for training.

- `results/diagnostics/raw/train_recoverability/success_air_a.npz` — chase, handoff 6.456 s, 10 s and 12 s hold, one vy sample outside K=35
- `results/diagnostics/raw/train_recoverability/success_air_b.npz` — same law, handoff 5.736 s, 10 s and 12 s hold, one vy sample outside K=35
- `results/diagnostics/raw/train_recoverability/success_imp_c.npz` — frozen-policy member 2, handoff 3.640 s, 10 s and 12 s hold, inside K=35

## Whether formal GRPO may begin

Airborne coverage is now 2/2 verified legal recoveries. Impact privileged search is still 0/4, with one representable `SUCCESS_12S` found by the frozen policy on `imp_c`. The preferred impact coverage of 2 verified train episodes is unmet.

The frozen policy's airborne groups never reach the new chase basin. Those demonstrations also contain a vy command the frozen codebook cannot reconstruct inside ±0.25. A GRPO update from the current policy would be training against all-failure airborne groups, and it cannot express the saved airborne correction without a new token. That token was not added.

Slip still supplies most of the mixed groups. Impact supplies one (`imp_c`). That is real cross-family signal, and it is not enough to start formal GRPO while the airborne solutions sit outside both the policy's sampled outcomes and the frozen residual cap.

PRIVILEGED COVERAGE IMPROVED BUT SUPERVISED WARM-START IS STILL REQUIRED
