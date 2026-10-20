# 40 ms HANDOFF + Time-Induced Advantage Audit — REPORT

Read-only. See also `handoff_timing_semantics.md`, `advantage_implementation_audit.md`.

## Part A — Is recovery really ~40 ms?

- ICs with `recovery_elapsed ≈ n_motion×0.02` and ≈0.04 s: **7/7**
- GRPO motion token counts: [2, 2, 2, 2, 2, 2, 2]
- `R=1.998` is exactly `2 − 0.05×0.04`; not saturation/default.
- Absolute `t_handoff~3.28` is IC snapshot clock + 0.04 s; paired_results used absolute time (misleading Δ).

## Parts B–C — Counterfactual causal classes

- DIRECT_HANDOFF_SUFFICIENT: **0**
- LEGAL_PREFIX_REQUIRED: **7**
- REPRODUCTION_MISMATCH: **0**
| IC | class | B0 | B1 | B2 | B3 | B4 | tokens |
|---|---|---|---|---|---|---|---|
| hrv1_920047 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |
| hrv1_920064 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |
| hrv1_920106 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |
| hrv1_920015 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |
| hrv1_920069 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |
| hrv1_leftover_hard_910207 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |
| hrv1_leftover_hard_910231 | LEGAL_PREFIX_REQUIRED | False | False | True | False | False | `[36, 36, 47]` |

Mechanism labels: these short prefixes are **not** evidenced Contact-Preserving Sliding or Release–Recatch 
(migration/both-off thresholds unmet in prior audit; Other/Unclear).

### Physical pattern (all 7 ICs identical token pattern `[36,36,47]`)

From `seven_ic_physical_state.csv` / `reproduction_checks.csv`:

| Branch | Typical pre-HANDOFF contact | SUCCESS_12S |
|---|---|---|
| B0 immediate | bilateral (nL,nR>0), nonzero pad forces | **False** |
| B1 one×token36 | both_off (nL=nR=0) | **False** |
| B2 two×token36 | both_off (nL=nR=0) | **True** |
| B3 SFT same #steps | (SFT actions) | **False** |
| B4 SFT full | long recovery then fail | **False** |

So: **IC is not already handoff-ready** (B0 fails). **One** legal step is insufficient (B1 fails). **Two** GRPO steps of token 36 are necessary and sufficient under this protocol. Matched-time SFT actions do not substitute. Brief both-off at the HANDOFF instant is visible but below the sustained Release–Recatch threshold used elsewhere.

## Parts D–F — Advantage & strata

```json
{
  "n_ic_updates_logged": 635,
  "bucket_counts": {
    "mixed_25_75": 247,
    "mixed_ge_75": 186,
    "all_failure": 60,
    "mixed_lt_25": 91,
    "all_success": 51
  },
  "n_optimizer_updated": 632,
  "n_all_success": 51,
  "n_all_success_with_time_variance": 48,
  "frac_updates_all_success": 0.0759493670886076,
  "all_success_adv_corr_mean": 1.0
}
```

- Implementation: `A=(R−mean)/std` with **no ε**; zero-std skips update (`grpo_math.advantages`).
- All-success + time variance ⇒ `A ≈ −z(T)` independent of λ>0 (see sensitivity CSV).

## Part G — Skill competition

See `mechanism_competition_audit.md` → **INSUFFICIENT_SKILL_COMPETITION_EVIDENCE** 
(no per-trajectory mechanism labels in `ic_log.jsonl`).

## Final answers


1. **是否真约 40 ms recovery？**  
   是（相对 IC restore）。典型为 **2×20 ms `step_legal` + HANDOFF**；`recovery_elapsed=0.04`，`R=1.998`。绝对仿真钟约 3.24→3.28 s。依据：`step_legal.py`、`finish_recovery`、`seven_ic_timing.csv`、各 IC `traces/*/grpo_sequence.json`。

2. **多少 IC 直接 HANDOFF 即可成功？**  
   **0/7**（`DIRECT_HANDOFF_SUFFICIENT`）。见 `causal_classification.csv` / `seven_ic_counterfactual.csv` B0 列。

3. **多少必须经过 GRPO 合法前缀？**  
   **7/7**（`LEGAL_PREFIX_REQUIRED`）。Reproduction mismatches: 0.

4. **成功依赖什么物理变化？**  
   7/7：B0 仍有双侧接触却失败；两步 token 36 后 HANDOFF 瞬间多为 both_off，但完整 nominal 成功。B1（一步）失败 ⇒ **第二步必要**。B3（SFT 同步数）失败 ⇒ **动作内容必要，非仅时间长度**。详见 `seven_ic_physical_state.csv`。

5. **能否称为 Sliding / Recatch？**  
   **不能**作为标准 Contact-Preserving Sliding 或 Release–Recatch。40 ms、无计量接触迁移；both-off 短于分类阈值。应描述为 **短时合法微调整 + 及时 HANDOFF**（Other/Unclear）。

6. **历史 all-success groups 多少？**  
   `n_all_success=51` / logged=635；其中 time 方差>0：`48`；占 optimizer updates 比例≈`0.0759493670886076`.

7. **时间项是否主导这些 group 的 normalized advantage？**  
   对 all-success 且 σ_T>0：是，`A≈−z(T)`（与 λ>0 无关）。零方差则 skip。依据：`all_success_group_detail.csv`、`lambda_advantage_sensitivity.csv`、`advantage_implementation_audit.md`。

8. **是否有证据持续偏向 Recatch、压制 Sliding？**  
   **INSUFFICIENT_SKILL_COMPETITION_EVIDENCE**（训练 log 无逐轨迹机制标签）。Oracle matched 对仅说明成功之间 reward 偏好更快者（ΔR≤0.04）。

9. **下一步？**  
   **保持 reward λ 为主建议。** 这 7 个 IC 全是 `LEGAL_PREFIX_REQUIRED`（非“本来就能交权”），说明 GRPO 学到了 **极短合法前缀**，不是纯 HANDOFF 投机。  
   同时：历史 **51** 个 all-success group（约 **7.6%** updates）中，时间方差使 `A≡−z(T)`（相关≈1.0），会放大“更快更好”的排序——这是采样/目标层问题，不是证明应立刻改 λ。  
   若继续：优先记录逐轨迹机制标签、监控 all-success 组的时间竞争；**不要**把这 7 例写成 Sliding/Recatch 技能进步。
