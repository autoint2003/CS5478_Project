# SFT Sliding root-cause diagnosis

**Verdict:** Sliding ICs and Contact-Preserving expert tapes **are present and physically real**, but they contribute only **~4% of effective SFT loss weight**. Under teacher forcing the model fits mid-trajectory Sliding tokens well, yet at **step 0** it almost always prefers Recatch open-grip tokens (34/36). Deterministic and stochastic free rollouts therefore never execute Contact-Preserving Sliding.

| Item | Path |
|---|---|
| Checkpoint | `results/sft/checkpoint_epoch_0040.pt` |
| SFT manifest | `data/supervised_targets_with_horizontal.jsonl` |
| Loss | `training/train_sft.py` `supervised_loss` |
| Dataset | `training/dataset.py` `episode_tensors` / `collate` |
| Runner | `visualization/diagnose_sft_sliding_root_cause.py` |
| Prior valid audit | `results/diagnostics/validation_skill_classification/` |

---

## Phase 1 — Dataset and effective loss weight

**Loss formula (from code, not inferred from counts):**

1. Per trajectory: mean CE over masked tokens + mean residual MSE on non-HANDOFF steps.
2. Batch: `mean(sample_weight * traj_loss)`.

Therefore **epoch contribution ∝ `sample_weight`**, not IC count and not token count.

### By skill (all 325 manifest rows)

| skill | tapes | unique ICs | tokens | motion | HANDOFF toks | weight sum | weight share | mean w/tape | first-token top |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sliding | 112 | **77** | 8496 | 8384 | 112 | **2.44** | **0.042** | 0.0218 | 0×43, 10×37, 4×32 |
| vertical_recatch | 30 | 30 | 1154 | 1125 | 29 | **30.0** | **0.522** | 1.0 | 36×27, 39×3 |
| horizontal_recatch | 22 | 22 | 1651 | 1629 | 22 | **22.0** | **0.383** | 1.0 | 34×22 |
| regrasp_other | 154 | 83 | 5139 | 4985 | 154 | 3.0 | 0.052 | 0.0195 | 39, 36, … |
| control | 7 | 7 | 7 | 0 | 7 | 0.0 | 0.0 | 0.0 | 47×7 |

- Unique global snapshots: **154** (one per IC).
- Sliding∩Vertical overlapping train IDs: **30** (every Vertical tape shares an ID with Sliding). User-cited 46 is **not** reproduced from this manifest.
- No train/valid leakage in this file: valid Sliding focus ICs (`k48r_valid_00009`, `inward_extra_250043`) are **absent** from SFT loss (held out; see `data/valid_skill_quotas.jsonl`).
- Sliding uses the same K48 encoding path as other skills (`episode_tensors` → `episode_arrays` + codebook). First Sliding tokens 0/4/10 have grip channel **+1** (closed); Recatch 34/36 have grip **−1** (open).
- HANDOFF (logit 47): one supervised HANDOFF token per recovery tape with early/recorded-end labeling; Sliding HANDOFF position mean ≈ 74.9 steps (~1.5 s recovery). Share of Sliding tokens that are HANDOFF: 112/8496 ≈ 1.3%.

Files: `dataset_audit.csv`, `dataset_audit_summary.json`, `weight_vs_ic_share.png`.

---

## Phase 2 — Expert Sliding physical authenticity

Replay used continuous expert actions via `step_legal` + full `finish_recovery` (same interface as prior audits).

| Sample set | n | Contact-Preserving | SUCCESS_12S |
|---|---:|---:|---:|
| Train Sliding (focus + stratified) | 12 | **12** | 12 |
| Valid focus experts (`00009` inward+controlled, `250043` inward) | 3 | **3** | 3 |

Example facts: bilateral contact every recovery sample; contact-centroid migration 1.4–5.8 mm; tokens often long runs of 0, 4, or 10 (closed-grip).

**Classification:** these are **A — real Contact-Preserving Sliding**, not label-only. No audited tape was Release–Recatch (C) or empty (D).

File: `expert_mechanism_audit.csv`.

---

## Phase 3 — Teacher-forced probability audit

Protocol: one causal forward over the expert observation/prev prefix (`models/k48_policy.py` causal mask). Target at step `t` uses only `obs[:t+1]`. Same normalization / residual head / HANDOFF handling as training.

| skill | n eps | motion top-1 | top-5 | **first top-1** | early-5 top-1 | mean expert p | first open-grip argmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| horizontal_recatch | 22 | 0.961 | 1.000 | **1.000** | 0.855 | 0.936 | 1.0 |
| vertical_recatch | 30 | 0.958 | 0.998 | **0.900** | 0.740 | 0.933 | 1.0 |
| sliding (train sample) | 24 | 0.948 | 1.000 | **0.000** | 0.756 | 0.854 | 1.0 |

Sliding first-step detail (n=27 first decisions):

- Expert→argmax pairs: (0→36)×9, (4→36)×9, (10→36)×7, (0→34)×2
- Expert token p: mean **0.023**, min 0.007; rank median **3** (in top-5 for 96%)
- Argmax p mean **0.70**

Valid held-out experts (not trained): same pattern — argmax 36; expert first p ≈ 0.02–0.13; rank 4–5.

**Interpretation:** Under true Sliding history, SFT still prefers release-open Recatch at the first action. Expert Sliding tokens remain in the top-5 (multimodal competition), but never win argmax. After the first step, TF accuracy recovers — the failure is concentrated at the opening decision, not total sequence unlearnability.

Residual: Sliding mean resid L2 ≈ 0.13 vs Vertical 0.046 / Horizontal 0.077 (worse but secondary to the discrete first-token conflict).

Files: `teacher_forcing_per_token.csv`, `teacher_forcing_summary.json`, `teacher_forcing_valid_focus.csv`, `tf_first_step_prob_scatter.png`, `tf_sliding_first_rank_hist.png`.

---

## Phase 4 — Matched-state free rollouts

Same snapshot, MuJoCo, legal decode, full 12 s nominal after handoff. Stochastic changes only policy sampling seeds.

| IC | expert | det first | det mechanism / result | stoch any CP |
|---|---|---:|---|---|
| ball_train_v2_0012 | CP / SUCCESS | 36 | Other / TABLE_DROP | False |
| ball_train_v2_0085 | CP / SUCCESS | 34 | Release–Recatch / TABLE_DROP | False |
| ball_train_v2_0070 | CP / SUCCESS | 36 | Hybrid / SUCCESS | False |
| ball_train_v2_0098 | CP / SUCCESS | 36 | Hybrid / DROP_AFTER_HANDOFF | False |
| ball_train_v2_0135 | CP / SUCCESS | 36 | Hybrid / SUCCESS | False |
| ball_train_v2_0155 | CP / SUCCESS | 36 | Hybrid / SUCCESS | False |
| k48r_valid_00009 | CP (expert tape) | 36 | Hybrid / LOST_GRASP | False |
| inward_extra_250043 | CP (expert tape) | 36 | Other / TABLE_DROP | False |

**Pattern:** Teacher forcing can track Sliding after step 0, but free rollout **chooses Recatch from the first action**. Stochastic decoding does **not** recover Contact-Preserving Sliding on these matched ICs.

This matches: *TF wrong at t=0 → free rollout Recatch/Hybrid*; not “TF correct then closed-loop drift alone.”

File: `matched_rollout_comparison.csv`.

---

## Phase 5 — Overlap IC supervision conflicts

| metric | value |
|---|---:|
| Overlap IDs in SFT manifest | 30 |
| Same snapshot path | 30/30 |
| obs₀ L∞ < 1e-4 | 30/30 |
| Meaningful first-token conflict (same obs, different closed vs open intent) | **30/30** |
| TF prefers Vertical first token over Sliding | **30/30** |
| Weight on overlap (Sliding / Vertical) | **0.65 / 30.0** |

No skill-conditioning input exists. CE does not average continuous actions, but deterministic decoding prefers the higher-mass Recatch mode.

File: `overlap_conflicts.csv`, `overlap_conflicts_summary.json`.

---

## Phase 6 — Answers

### Q1. Does Sliding dominate SFT supervision?

| view | Sliding advantage? |
|---|---|
| IC count (77) | Yes numerically |
| Tape count (112) | Yes |
| Motion tokens (8384) | Yes |
| **Effective loss weight (2.44 / 4.2%)** | **No — starved** |

### Q2. How many Sliding experts are truly Contact-Preserving?

Audited sample: **15/15** Contact-Preserving (12 train + 3 valid). Do not extrapolate to all 112 tapes without further replay; evidence so far does **not** support “fake Sliding labels.”

### Q3. Can SFT predict Sliding under teacher forcing?

- Mid/late motion: **yes** (top-1 ≈ 0.95).
- First action: **no** (top-1 = 0.0; expert p ≈ 0.02; argmax Recatch open).

### Q4. Recatch from step 0, or Sliding then diverge?

**From step 0.** First token ∈ {34, 36}; sustained both-off often begins by ~0.02–0.06 s.

### Q5. Does stochastic decoding recover Contact-Preserving?

**No** on matched ICs (seeds 11/22/33).

### Q6. Overlap multimodal conflict?

**Yes, severe:** 30/30 same-obs first-token conflicts; Vertical weight 46× Sliding on those IDs; TF always prefers Vertical token.

### Q7. Most likely root causes (ranked)

See `root_cause_evidence.md`. Short form:

1. **Confirmed:** Sliding weight starvation (~4% vs Recatch ~90%).
2. **Confirmed:** First-step multimodal bias to Recatch open-grip (TF + free rollout).
3. **Supported:** No skill conditioning → argmax collapses to Recatch; stochastic insufficient.

---

## Chinese summary — 最有证据支持的前三个原因

1. **Sliding 的有效监督权重极低（约 4.2%），Vertical+Horizontal 约占 90%。** IC 数量不能代表梯度贡献；`sample_weight` 使每条 Sliding tape ≈0.02、Recatch tape =1.0。**需要重新训练（重加权/重平衡数据）才能解决。**

2. **在相同初始观测下，Sliding（闭爪 0/4/10）与 Vertical Recatch（开爪 36/39）形成系统性首步冲突；SFT 在 teacher forcing 下首步 top-1=0，argmax 稳定选开爪 Recatch。** 专家 Sliding 仍常在 top-5，属多模态竞争而非完全未学。**需要重新训练（去冲突、技能条件化、或提高 Sliding 首步权重）才能解决；仅改解码温度不够。**

3. **确定性自由 rollout 从第 0 步就输出 Recatch；随机采样也无法恢复 Contact-Preserving。** 专家 tape 本身经审计确为接触保持滑动，问题不在标签造假。**需要重新训练才能改变首步模式偏好；仅靠闭环微调或采样通常不够。**

---

## Repro

```powershell
python -u visualization/diagnose_sft_sliding_root_cause.py --device cuda
```

Outputs under `results/diagnostics/sft_sliding_root_cause/`.
