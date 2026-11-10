# HISTORICAL_K48_RESULTS

Generated: 2026-11-06T16:40:27.637765+00:00
Status: **archive recovery only** — no retrain.

## Canonical historical K48 GRPO checkpoint (primary)

| Field | Value |
|---|---|
| Label | `v2_u0312` (GRPO long reward v2, update 312) |
| Path | `results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2/checkpoint_u0312.pt` |
| MD5 | `f48cf9eca491c5ed76ad79ddc4e62024` |
| SHA256 | `d91ddde0772629df26c47734382c4ce58599b550379dc8fb037ff095dfd9df65` |
| Report | `results/archive/demo_results/diagnostics/K48_GRPO_EVAL_COMPARISON.md` |
| Raw | `results/archive/demo_results/diagnostics/raw/unified_k48/eval_compare/v2_u0312/heldout_train.json` |

### Historical scores

| Split | Success | Denominator | Criterion |
|---|---:|---:|---|
| Held-out train (ids divisible by 5) | **30** | **56** | `SUCCESS_12S` |
| Validation | **26** | **26** | `SUCCESS_12S` (all early HANDOFF) |

Archive summary reasons (train): `{"SUCCESS_12S": 30, "DROP_DURING_RECOVERY": 5, "DROP_AFTER_HANDOFF": 16, "LOST_GRASP": 5}`

### Historical IC IDs (all 56)

Corpus: **`ball_train_v2_*` only**, numeric id divisible by 5 (`0005` … `0280`).

First 12: ball_train_v2_0005, ball_train_v2_0010, ball_train_v2_0015, ball_train_v2_0020, ball_train_v2_0025, ball_train_v2_0030, ball_train_v2_0035, ball_train_v2_0040, ball_train_v2_0045, ball_train_v2_0050, ball_train_v2_0055, ball_train_v2_0060
…
Last 4: ball_train_v2_0265, ball_train_v2_0270, ball_train_v2_0275, ball_train_v2_0280

Root IDs: identical to traj ids in this corpus (no Plan-B `plan_b__` prefix).

### Historical evaluation protocol

| Item | Value |
|---|---|
| Inference | **argmax** K48 logits (deterministic) |
| Action decode | K47 codebook + residual (`decode_action`); HANDOFF index **47** |
| Recovery limit | **4.0 s** (`SELECTED_LIMIT_S`) |
| HANDOFF | Early via HANDOFF token **or** forced at recovery limit; basin early-exit only if no handoff head |
| Success | Physical **`SUCCESS_12S`** after recovery + nominal hold to 12 s |
| Device (archive) | CPU |
| Controller / physics | Legal 7D `step_legal` / AIRBORNE_V3; 39-D airborne obs; MuJoCo Panda cylinder |

Source report excerpt path: `D:/NUS/CS5478 Intelligent Robots/CS5478_Project/results/archive/demo_results/diagnostics/K48_GRPO_EVAL_COMPARISON.md`

## Secondary K48 GRPO lineage (do not confuse)

| Label | Path | MD5 | Role |
|---|---|---|---|
| tree80 epoch 3 | `results/grpo/tree80/checkpoint_epoch_0003.pt` | `331abfa72dbc049fe9df582998ea5340` | Later adaptive GRPO; used as frozen Stage-1 generator for B experts (P46/P47). SHA256 `c37da4a40fb73fcef1d1b5e0f04e9bd2410b01fa1c9d2d4e9e062063a6725648` |
| P22 A/B SFT | M111 | — | **Not K48** |
| P28 unified SFT | M111 | — | **Not K48** |
| M111 GRPO / P53+ | M111 | — | **Not K48** |
| P64 Actor | `P64_step4_actor.pt` | `9f4c50413535e4c7db3d3a868bc1b6e2` | Frozen final M111 policy under audit |

## Equivalence warning vs current B1

Historical “strong B” evidence is **`SUCCESS_12S` on `ball_train_v2` ICs with K48 early/forced HANDOFF**.
Current independent B1 is **Plan-B MixedStart ICs**, M111 P64, **`SUCCESS_RELAXED_12S` + R3F**.
Matching the letter “B” / “B1” does **not** imply equivalent physical tasks or protocols.


## Paired B1 transfer (post-hoc diagnostic)

On the eight frozen independent B1 ICs (not the historical 56):

| Checkpoint | SUCCESS_12S |
|---|---|
| tree80_e3 | **8/8** |
| v2_u0312 | **0/8** |

tree80 is the B-relevant historical GRPO for Plan-B comparison; v2_u0312's archive 30/56 does not transfer to this IC set.
