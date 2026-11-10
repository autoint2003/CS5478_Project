# B1_REGRESSION_VERDICT

Generated: 2026-11-06T16:43:31.518982+00:00

## Verdict: **CASE A** — Historical K48 GRPO succeeds on B1 ICs where P64 fails

**Primary B-relevant checkpoint:** `results/grpo/tree80/checkpoint_epoch_0003.pt` (K48 adaptive GRPO tree80 epoch 3), the lineage used historically for Plan-B / Tree80 B expert generation.

On the exact eight frozen independent B1 MixedStarts, under the original K48 `roll_policy` wrapper (argmax + decode_action + HANDOFF=47 + SUCCESS_12S):

| Policy | Metric | Score |
|---|---|---|
| K48 tree80_e3 | SUCCESS_12S | **8/8** |
| K48 v2_u0312 (archive 30/56 ckpt) | SUCCESS_12S | **0/8** (premature HANDOFF → LOST_GRASP) |
| P64 (post-hoc replay) | SUCCESS_12S | **0/8** |
| P64 (post-hoc / frozen) | SUCCESS_RELAXED_12S | **0/8** / **0/8** |

This is evidence of **B1 policy regression of the frozen M111 P64 Actor relative to earlier K48 tree80 GRPO** on the current B1 IC set.

### Important secondary finding

Archive `v2_u0312` (**30/56** on `ball_train_v2` ids÷5) does **not** succeed on these Plan-B B1 ICs. That historical score is **not directly comparable** to current B1 (distribution shift + premature-HANDOFF transfer failure). Alone it would look like CASE B/C; tree80 paired results drive the overall verdict to **CASE A** for the B-class K48 GRPO lineage that mattered for Plan-B.

HANDOFF contracts still differ (K48 early token vs P64 R3F). SUCCESS_RELAXED_12S is N/A for K48. Paired scores are **post-hoc diagnostic**, not a replacement independent final-test estimate.

## Preserved independent final-test result

| Item | Value |
|---|---|
| Frozen P64 B1 | **0/8** SUCCESS_RELAXED_12S |
| Artifact | `results/experiments/final_independent_success_rate/EPISODE_RESULTS.csv` |
| Status | **Unchanged** |

## Phase 2 — is 0/8 an eval bug?

- Infra errors: none.
- Failures: `no_e_pos_le9` (4) / `no_r3_ready` (4); all have first_contact=1 and opposing_ever=1.
- Snapshot restore: **8/8** OK (`RESTORE_AUDIT.csv`).
- Post-hoc P64 replay matches frozen success bits: **8/8**.

**Conclusion: 0/8 is not an evaluation/restoration bug (not CASE D).**

## Phase 4 mechanism read

See `TRAJECTORY_DIVERGENCE.md`.

1. Policy skill regression (P64 vs tree80) — **primary**.
2. IC/distribution shift vs archive v2 30/56 — **also true** (explains v2_u0312 failure here).
3. Action representation K48≠M111 — structural continuity break.
4. Inference/eval mismatch — present; dual metrics reported.
5. Statistical uncertainty — n=8 diagnostic only.

## What this audit does **not** authorize

- Restoring K48 / tree80 as the final policy
- Retraining or checkpoint-selecting P64 from these eight ICs
- Replacing the independent 0/8 with post-hoc scores

## Artifacts

- `HISTORICAL_K48_RESULTS.md`
- `PROTOCOL_COMPATIBILITY.md`
- `PAIRED_B1_RESULTS.csv`
- `TRAJECTORY_DIVERGENCE.md`
- `B1_REGRESSION_VERDICT.md`
- `RESTORE_AUDIT.csv`
- `RAW_K48_ROLLS.json` / `RAW_P64_ROLLS.json`
