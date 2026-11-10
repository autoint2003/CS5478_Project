# PROTOCOL_COMPATIBILITY

Generated: 2026-11-06T16:40:28.033656+00:00

## Question

Are historical K48 “B” results and the current independent **B1** evaluation the same physical task under a comparable protocol?

## Short answer

**No — not equivalent.** Physical restore of current B1 captures into the K48 `roll_policy` wrapper **is** possible (obs 39-D + capture/ref), but IC distribution, action vocabulary, HANDOFF semantics, and success criteria differ. A paired closed-loop on the eight B1 ICs is therefore a **diagnostic transfer test**, not a reproduction of the historical 30/56 number.

## Side-by-side

| Axis | Historical K48 GRPO (`v2_u0312`) | Current independent B1 (P64) | Compatible? |
|---|---|---|---|
| Policy family | K48 Transformer (48-way) | M111 Transformer (112-way) P64 Actor | **No weight sharing** — separate loaders |
| Action decode | `k47_codebook.decode_action` + HANDOFF=47 | `decode_m111_action` + HANDOFF=111 | **Incompatible vocab** |
| Inference sampling | **argmax** | Actor sample (+ V2 gate, rule=None; V2 idle on these eps) | Different stochasticity |
| Observation | 39-D airborne | 39-D airborne | Yes (dim) |
| IC restore | snapshot `capture`+`ref` | MixedStart `kind=ep` with nested `capture`+`ref` | Yes for physics |
| IC corpus | `ball_train_v2_{5,10,…,280}` (56) | Plan-B heldout mix: k48r / slide_quota / inward_extra / one ball_train_v2 | **Different** |
| Exact ID overlap with hist 56 | — | Exact normalized overlap: **0** → ∅ | Near-zero |
| HANDOFF | Early token 47 or forced @ 4 s | R3F dwell / monitor → SUCCESS path | Different |
| Success criterion | `SUCCESS_12S` | `SUCCESS_RELAXED_12S` (15° / 15 mm on 12s path) | Related but **not equal** |
| Max recovery | 4.0 s | 80 control steps (~episode budget) + R3F t_max 3100 | Different clocks |
| Controller | `step_legal` recovery v3 | Same underlying `step_legal` inside StabilizeGrpoEnv | Shared low-level |

## Current B1 IC inventory (frozen independent set)

| traj_id | root_id | mechanism | source_corpus |
|---|---|---|---|
| `plan_b__k48r_train_00092` | `k48r_train_00092` | follow_early | unified_experts |
| `plan_b__slide_quota_400062` | `slide_quota_400062` | follow_early | unified_experts |
| `plan_b__slide_quota_400473` | `slide_quota_400473` | follow_early | unified_experts |
| `plan_b__k48r_train_00125` | `k48r_train_00125` | follow_early | unified_experts |
| `plan_b__slide_quota_400614` | `slide_quota_400614` | follow_early | unified_experts |
| `plan_b__inward_extra_250025` | `inward_extra_250025` | follow_early | unified_experts |
| `plan_b__ball_train_v2_0239` | `ball_train_v2_0239` | follow_early | unified_experts |
| `plan_b__k48r_train_00007` | `k48r_train_00007` | follow_early | unified_experts |


Mechanism counts: {'follow_early': 8}
Corpus counts: {'unified_experts': 8}

## Restore audit (Phase 2 prerequisite)

Capture→sim restore without policy: **8/8** OK.

See `RESTORE_AUDIT.csv`.

## Common protocol adopted for Phase 3

1. **Same eight frozen B1 MixedStarts** (nested capture/ref).
2. **K48:** `load_trained_k48` + `make_training_env` + instrumented argmax `roll_policy` (native SUCCESS_12S; post-hoc SUCCESS_NEW / SUCCESS_RELAXED probes).
3. **P64:** identical `StabilizeGrpoEnv` + `rollout_episode` path as final eval (native SUCCESS_RELAXED_12S); seeds from frozen manifest.
4. **Do not** load K48 weights into M111.
5. Report **both** native metrics; treat paired numbers as **post-hoc diagnostic**, not a new independent final-test estimate.

## Incompatibilities that remain

- Cannot share action-token sequences across policies (48 ≠ 111).
- Cannot force identical HANDOFF gating without changing a policy’s inference contract.
- Historical 30/56 is **not** an estimate on these eight ICs.


## Phase 3 paired outcome note (post-run)

| Checkpoint | SUCCESS_12S on 8 B1 ICs |
|---|---|
| K48 tree80_e3 | **8/8** |
| K48 v2_u0312 | **0/8** |
| P64 SUCCESS_12S | **0/8** |
| P64 SUCCESS_RELAXED_12S | **0/8** |

Common physical restore worked. Action vocabularies remain incompatible (separate decoders used). See `B1_REGRESSION_VERDICT.md`.
