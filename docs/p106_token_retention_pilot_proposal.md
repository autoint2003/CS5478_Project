# P106 — Action-Aware Token Retention: Controlled Pilot (Authorized)

**Status:** Authorized for one bounded exploratory pilot.  
**Baseline:** Immutable P64 `ec08f59816aa2fdd`.  
**Do not** replace P64, unfreeze backbone, train residuals, change M111/R3F, run PPO/GRPO, or open the sealed independent panel.

## Confirmed P105 mechanism (do not restate as small-KL)

P104-B A1 regression on `plan_a__ball_train_0017` was caused by:

- t=0 token argmax flip **27 → 8**
- P64 logit margin ≈ **3.42** (large-margin)
- Token KL ≈ **0.91** (not small)
- Executed-action L2 ≈ **1.80** under frozen residual
- Later physical divergence and DROP
- Non-degenerate token interventions: force-P64 rescues; reverse destroys

P105 also: **15/15** early C→feasible corrections reduced frozen-residual action error. Frozen-residual incompatibility is **not** the leading hypothesis.

## Exact hypothesis

Action-aware token retention that penalizes probability mass on tokens with large executed-action distance from the P64 argmax token will suppress destructive A1 flips more effectively than a **retention-strength-matched** KL-only control, while preserving early-C correction capability.

## Architecture

All trainable arms:

- Initialize independently from immutable P64
- Freeze Transformer backbone and **entire** P64 residual
- Train **only** existing M111 token head
- Unchanged decoder / official R3F
- One unified Actor; no router / extra head / skill labels at inference
- Verify init token / residual / executed-action parity

## Experimental arms

| Arm | Role |
|---|---|
| **A** | Frozen P64 |
| **B** | Existing P104 v2 Token-only checkpoint (fixed reference; **do not retrain**) |
| **C** | Token-only SFT + **stronger ordinary KL** retention (strength-matched control) |
| **D** | Token-only SFT + KL + **action-aware** retention (primary intervention) |

Arms C and D: identical correction/preservation data, optimizer, seed, updates, checkpoint-selection rule, and **comparable retention regularization strength**.

## Loss equations

Correction objective = P104 v2 token CE (ABCD; Category D masked; early / early-C Category-C weights unchanged).

On preservation histories H (HANDOFF masked consistently with inference):

```
z0 = argmax π_P64(z | H)
a_z = decode_clipped(z, r_P64(H))     # actual executed action, not raw proto
d_z = ||a_z - a_{z0}||_2² / D_norm    # D_norm = mean d over train preservation (frozen)

L_AA = mean_H  Σ_z  d_z · (π_new(z|H) - π_P64(z|H))²
```

`L_AA = 0` at P64 initialization. HANDOFF excluded via `mask_handoff_logits`.

```
L_D = L_correction + λ_KL · L_KL + λ_AA · L_AA
L_C = L_correction + λ_KL_control · L_KL
L_KL = mean_H KL(π_P64 || π_new)
```

## Coefficient calibration (training-only; no eval peeking)

1. Measure mean correction loss scale `L_corr` at init (B1+C pair), as in P104.
2. KL reference: temperature `T=2` softmax (never uniform logit+1).  
   `λ_KL = clip(STRENGTH_KL · L_corr / KL_ref, max=Λ_max)`.
3. AA reference: **fixed prior P104-B v2** Token-only distributions on preservation (documented nontrivial harmful deviation; temperature AA recorded as audit only).  
   `λ_AA = clip(STRENGTH_AA · L_corr / AA_ref_P104B, max=Λ_max)`.
4. **Strength match C↔D** (locked before training; uses intended scales):  
   `λ_KL_control = clip((STRENGTH_KL + STRENGTH_AA) · L_corr / KL_ref, max=Λ_max)`  
   so Arm C’s KL retention matches Arm D’s total intended retention scale.  
   Additionally record gradient-norm ratio on a fixed preservation minibatch for audit.
5. Locked a priori: `STRENGTH_KL=1.5`, `STRENGTH_AA=1.5` (total ≈ P104’s 3.0). `Λ_max=10000` (AA_ref under peaked token distributions is small; cap must not crush intended scale). No post-hoc sweeps.

## Data provenance

- Correction: P98 train-eligible B1/C (same as P104)
- Preservation: P98 A1/B1/D P64-success (full trajectories; root-balanced)
- Audit whether exposed A1 failure root `ball_train_0017` is already in legitimate preservation; **do not add** it solely to fix the observed failure
- Exclude sealed / banned / ind roots
- Freeze `TRAIN_MANIFEST.json` before training

## Budget

- Seed **0** for Arms C and D
- **2000** updates, LR=`1e-3`, same early weights as P104
- Checkpoint = final update (no CL-based selection; never select on exposed A1 failure root)
- No hyperparameter sweep / automatic retry

## Evaluation

**Mechanistic** (identical recorded histories): token KL, argmax agreement, margin, action-weighted deviation, executed-action deviation, large-action-distance flips, early C feas + action error. Compare whether D suppresses destructive flips better than C at matched strength.

**Closed-loop** `SUCCESS_RELAXED_12S`: A1 / B1-success / D retention; B1 and C-Feasible root-disjoint gains; C-train diagnostic. P104-B C-Feasible gain is a fixed reference, not a guaranteed target. C-Feasible n=2 is not decisive.

If any new P64-success root is lost: first-divergence (+ token CF when appropriate).

## Success criteria (exploratory, one seed)

- No protected A1/B1/D CL losses on predefined small panels
- ≥1 genuine root-disjoint B1/C recovery vs P64
- No provenance violations
- Mechanistic evidence AA protects damaging token changes
- For AA **superiority**: D outperforms strength-matched KL control (C) on the predefined retention/acquisition comparison

Outcome labels A–E as in the authorization brief.

## Stop

Stop after P106. No P107 / PPO / backbone unfreeze / ind panel / P64 replacement without approval.
