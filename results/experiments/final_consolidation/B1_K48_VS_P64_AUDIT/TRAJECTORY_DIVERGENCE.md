# TRAJECTORY_DIVERGENCE

Generated: 2026-11-06T16:43:31.518982+00:00

Scope: post-hoc paired closed-loop on the **same eight frozen B1 ICs**.
Independent final-test estimate remains **P64 0/8** unchanged.

## Aggregate

- K48 `tree80_e3` native SUCCESS_12S: **8/8** (B-relevant historical GRPO lineage)
- K48 `v2_u0312` native SUCCESS_12S: **0/8** (archive ball_train_v2 GRPO; premature HANDOFF)
- P64 post-hoc SUCCESS_12S: **0/8**
- P64 post-hoc SUCCESS_RELAXED_12S: **0/8** (frozen independent: **0/8**)
- K48 SUCCESS_RELAXED_12S: **N/A** (R3F-path metric)

## Checkpoint behavior contrast

- **v2_u0312:** on all 8 ICs emits token `12` ×4 then HANDOFF=`47` (~0.08 s) → `LOST_GRASP`.
- **tree80_e3:** ~36–37 motion steps (~0.72–0.74 s) with non-trivial token sequences ending in HANDOFF=`47` → **SUCCESS_12S on 8/8**.
- **P64:** first-contact + opposing on all 8; stalls at `no_e_pos_le9` (4) or `no_r3_ready` (4).

## Per-IC earliest meaningful divergence

### `plan_b__k48r_train_00092`
- Frozen P64 fail: `no_e_pos_le9`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=37 e9=1 tokens_head=[39, 39, 0, 0, 0, 0, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_e_pos_le9 ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__slide_quota_400062`
- Frozen P64 fail: `no_r3_ready`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=36 e9=1 tokens_head=[36, 36, 14, 33, 33, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_r3_ready ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__slide_quota_400473`
- Frozen P64 fail: `no_r3_ready`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=36 e9=1 tokens_head=[36, 36, 14, 33, 33, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_r3_ready ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__k48r_train_00125`
- Frozen P64 fail: `no_r3_ready`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=37 e9=1 tokens_head=[36, 36, 14, 9, 14, 14, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_r3_ready ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__slide_quota_400614`
- Frozen P64 fail: `no_e_pos_le9`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=36 e9=0 tokens_head=[36, 39, 0, 0, 0, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_e_pos_le9 ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__inward_extra_250025`
- Frozen P64 fail: `no_e_pos_le9`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=36 e9=1 tokens_head=[36, 36, 14, 33, 33, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_e_pos_le9 ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__ball_train_v2_0239`
- Frozen P64 fail: `no_e_pos_le9`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=37 e9=1 tokens_head=[39, 39, 0, 5, 5, 0, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_e_pos_le9 ep_len=80 family_tail=['GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

### `plan_b__k48r_train_00007`
- Frozen P64 fail: `no_r3_ready`
- v2_u0312: s12=0 reason=LOST_GRASP steps=4 tokens=[12, 12, 12, 12, 47]
- tree80: s12=1 reason=SUCCESS_12S steps=36 e9=1 tokens_head=[36, 36, 14, 33, 33, 0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47]
- P64: relaxed=0 12S=0 fail=no_r3_ready ep_len=80 family_tail=['GRIP', 'ORIENT', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP', 'GRIP']
- Contact ever fc/bil/opp/pad4: tree80=1/1/1/1; P64=1/1/1/1
- **Earliest meaningful divergence:** terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp

## Mechanism classes (diagnostic)

1. **Policy skill regression (P64 vs tree80 K48)** — **supported** on these 8 ICs.
2. **Task / IC distribution shift vs archive v2 30/56** — **also supported** (v2 does not transfer).
3. **Action representation** — K48≠M111; cannot weight-copy tree80 into P64.
4. **Inference / eval mismatch** — present; does not erase tree80 SUCCESS_12S.
5. **Statistical uncertainty** — n=8 post-hoc; not a new independent final-test estimate.
