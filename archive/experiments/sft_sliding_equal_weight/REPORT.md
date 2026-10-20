# Equal-weight Sliding SFT — controlled ablation

## Hypothesis under test

Sliding was not preserved by the original SFT primarily because its `sample_weight` was ~0.02 vs Recatch 1.0.
This run raises **only** Sliding per-tape weight to 1.0, trains from **scratch** with the same seed/epochs/hparams,
and compares Epoch 40 to the original.

## Facts about training setup

- Base config: `config/training/k48/unified_k48_recovery_v3_horizontal.yaml`
- Experiment config: `config/training/k48/sft_sliding_equal_weight.yaml`
- Independent manifest: `data/supervised_targets_sliding_equal_weight.jsonl` (325 rows, same tapes)
- Original manifest untouched: `data/supervised_targets_with_horizontal.jsonl`
- Init: `scratch` via `build_k48`; seed **1**; epochs **40**; batch **8**; Adam lr **3e-4**; **no scheduler**
- Sampling: each epoch `np.random.default_rng(seed).permutation` — every row once; **no class sampler**
- Loss: traj-mean CE+resid, then `mean(sample_weight * traj_loss)` (`training/train_sft.py`)
- No other class / length / IC normalization beyond that formula
- Output: `results/sft/sliding_equal_weight/` (does not overwrite `results/sft/checkpoint_epoch_0040.pt`)

### Effective weight mass

| skill | tapes | weight before | share before | weight after | share after |
|---|---:|---:|---:|---:|---:|
| control | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| horizontal_recatch | 22 | 22.0000 | 0.3830 | 22.0000 | 0.1317 |
| regrasp_other | 154 | 3.0000 | 0.0522 | 3.0000 | 0.0180 |
| sliding | 112 | 2.4400 | 0.0425 | 112.0000 | 0.6707 |
| vertical_recatch | 30 | 30.0000 | 0.5223 | 30.0000 | 0.1796 |

**Note (fact):** Per-tape Sliding weight is equalized to Recatch (1.0). Because there are 112 Sliding tapes,
Sliding **skill mass** rises to ~67% of total weight — not merely “equal skill share.”

## Validation (17 IC, deterministic)

| Metric | Original SFT | Equal-weight SFT |
|---|---:|---:|
| SUCCESS_12S | 8/17 | 17/17 |
| Contact-Preserving | 0/17 | 9/17 |
| Release–Recatch | 4/17 | 1/17 |
| Hybrid | 11/17 | 7/17 |
| Other | 2/17 | 0/17 |

### SUCCESS_12S by data label (overlapping; do not sum)

| label | Original | Equal-weight |
|---|---|---|
| `gravity_assisted_in_hand_sliding` | 2/11 | 11/11 |
| `horizontal_recatch` | 6/6 | 6/6 |
| `vertical_recatch` | 2/5 | 5/5 |

## Teacher forcing

| ckpt / skill | first top-1 | first expert p | motion top-1 | mean expert p | first closed argmax | first open argmax |
|---|---:|---:|---:|---:|---:|---:|
| original / sliding | 0.0 | 0.02075615266095037 | 0.9546943231441049 | 0.856908896288039 | 0.0 | 1.0 |
| original / vertical_recatch | 0.9 | 0.763863397638003 | 0.9582222222222222 | 0.9332639709048801 | 0.0 | 1.0 |
| original / horizontal_recatch | 1.0 | 0.9522294537587599 | 0.9607120933087784 | 0.9364650481314918 | 0.0 | 1.0 |
| equal_weight / sliding | 0.4782608695652174 | 0.3482992629642072 | 0.9890829694323144 | 0.9772022298036964 | 0.782608695652174 | 0.21739130434782608 |
| equal_weight / vertical_recatch | 0.36666666666666664 | 0.3466071248054504 | 0.9386666666666666 | 0.9061882759481669 | 0.6333333333333333 | 0.36666666666666664 |
| equal_weight / horizontal_recatch | 1.0 | 0.7070746340534904 | 0.9318600368324125 | 0.8948406207173984 | 0.0 | 1.0 |

## Matched-state free rollouts (Sliding expert ICs)

| IC | split | expert | original first/mech/result | equal-weight first/mech/result |
|---|---|---|---|---|
| `ball_train_v2_0012` | train | Contact-Preserving Adjustment/SUCCESS_12S | 36/Other/Unclear/TABLE_DROP | 0/Contact-Preserving Adjustment/DROP_AFTER_HANDOFF |
| `ball_train_v2_0085` | train | Contact-Preserving Adjustment/SUCCESS_12S | 34/Release-Recatch/TABLE_DROP | 0/Contact-Preserving Adjustment/DROP_AFTER_HANDOFF |
| `ball_train_v2_0070` | train | Contact-Preserving Adjustment/SUCCESS_12S | 36/Hybrid/SUCCESS_12S | 4/Contact-Preserving Adjustment/SUCCESS_12S |
| `ball_train_v2_0135` | train | Contact-Preserving Adjustment/SUCCESS_12S | 36/Hybrid/SUCCESS_12S | 4/Contact-Preserving Adjustment/SUCCESS_12S |
| `k48r_valid_00009` | valid | Contact-Preserving Adjustment/SUCCESS_12S | 36/Hybrid/LOST_GRASP | 10/Contact-Preserving Adjustment/SUCCESS_12S |
| `inward_extra_250043` | valid | Contact-Preserving Adjustment/SUCCESS_12S | 36/Other/Unclear/TABLE_DROP | 10/Contact-Preserving Adjustment/SUCCESS_12S |

## Q&A

### Q1. Did Sliding effective weight rise?

**Yes (fact).** Weight sum 2.44 → 112.0; share 4.25% → 67.07%. Per-tape Sliding = Horizontal = Vertical = 1.0.

### Q2. Did the new SFT learn Contact-Preserving Sliding?

**Yes (fact on validation + matched rollouts).** Contact-Preserving **0 → 9/17**. Matched free rollouts: all 4 train + both valid focus ICs become Contact-Preserving (first tokens 0/4/10 closed-grip). TF Sliding first expert p **0.021 → 0.348** (not only argmax flip); first top-1 **0 → 0.48**; motion mean expert p **0.857 → 0.977**.

### Q3. Sliding on held-out validation ICs?

**Yes.** Sliding-labeled valid SUCCESS **2/11 → 11/11**. Of 11 Sliding-tagged ICs, most equal-weight mechanisms are Contact-Preserving (e.g. `00009`, `250043`, several `slide_quota_*`). Two focus valid matched states: CP + SUCCESS_12S with first token 10.

### Q4. Did Recatch degrade?

**Horizontal: retained.** Label SUCCESS 6/6 → 6/6; TF first top-1 stays 1.0; free-rollout first token stays 34 (open). Horizontal first expert p dips 0.95 → 0.71 (still decisive).

**Vertical: mixed.** Label SUCCESS 2/5 → 5/5 (better recovery outcome), but TF first top-1 **0.90 → 0.37** and first closed-argmax fraction rises to 0.63 — on Vertical expert prefixes the policy often prefers Sliding closed tokens. Validation Vertical-tagged ICs still succeed, but mechanisms shift toward Hybrid/CP rather than pure Release–Recatch (Release–Recatch count 4 → 1 overall).

### Q5. Mode swap vs multi-skill retention?

**Partial multi-skill, not pure swap.** Sliding mode is restored and Horizontal Recatch is kept. Vertical Recatch **first-decision** preference is partly replaced by Sliding on overlapping/ambiguous states (expected given 67% Sliding weight mass). Overall SUCCESS rises; Recatch is not eliminated as a successful recovery family on Horizontal ICs.

### Q6. Does this support “severe downweighting was the main cause”?

**Supported (inference).** The only intentional change was Sliding `sample_weight` 0.02→1.0; from-scratch same seed/hparams. Sliding TF probability, free-rollout Contact-Preserving, and valid Sliding SUCCESS all rise sharply. Caveat: skill-level weight mass became Sliding-dominant (~67%), so this does **not** prove that a milder skill-balanced reweight would suffice — only that undoing the severe downweight restores Sliding.

## Interpretation classes

- **Fact:** numbers from this run’s CSVs and checkpoints.
- **Inference:** causal attribution to weight change (only intentional difference).
- **Unverified:** whether a milder reweight (skill-equal mass, not tape-equal×112) would keep Vertical first-token fidelity while retaining Sliding.

## Next step (GRPO)

**Worth continuing GRPO from this equal-weight Epoch 40**, with Contact-Preserving count as a hard monitor (original GRPO raised SUCCESS but stayed at 0 CP). Prefer starting from `results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt`, not the downweighted original. Risk: RL may again collapse toward Recatch/Hybrid — evaluate mechanism mix every epoch.

## 中文结论

- **Sliding 已恢复：** 验证集 Contact-Preserving 0→9/17，SUCCESS 8→17/17；Sliding 标签 IC 2/11→11/11；TF 首步专家概率与闭环闭爪首步均明显改善。
- **Recatch 基本保留：** Horizontal 6/6 仍成功且首步仍为开爪 34；Vertical 成功率升至 5/5，但首步 TF 更偏 Sliding，纯 Release–Recatch 变少——属权重过冲后的模式迁移，不是 Horizontal 崩溃。
- **下一步值得在新 SFT 上做 GRPO**，并强制监控 Contact-Preserving，避免再次只涨成功率、丢掉滑动技能。可选后续：做“技能质量相等”（而非每 tape=1 导致 Sliding 占 67%）的温和重加权对照。

## Repro

```powershell
python -u scripts/train.py --supervised --config config/training/k48/sft_sliding_equal_weight.yaml
python -u visualization/eval_sft_sliding_equal_weight.py --device cuda
```

