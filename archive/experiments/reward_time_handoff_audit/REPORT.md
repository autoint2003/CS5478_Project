# Reward / Time / HANDOFF Audit — REPORT

Read-only. Checkpoints, reward_v2, physics, and hard_recovery_v1 manifests unchanged.

## Phase 1 — Formula

See `reward_formula_audit.md`. Key: `R_SUCCESS = 2.0 − 0.05·recovery_s`; terminal-only; config `lambda_time` unused by code.

Component sum tests: **PASS** (`tests/reward_component_sum.csv`).

## Phase 3 — Seven GRPO-only ICs

| IC | SFT | GRPO | Δhandoff | R_SFT | R_GRPO | ΔR | ΔR_time | ΔR_terminal |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hrv1_920047 | LOST_GRASP | SUCCESS_12S | 1.4999999999998352 | -0.5256999999999841 | 1.9980000000000002 | 2.5236999999999843 | 0.07499999999999177 | 2.5 |
| hrv1_920064 | LOST_GRASP | SUCCESS_12S | 0.6999999999999229 | -0.46756666666666724 | 1.9980000000000002 | 2.4655666666666676 | 0.034999999999996145 | 2.5 |
| hrv1_920106 | LOST_GRASP | SUCCESS_12S | 0.6999999999999229 | -0.3912333333333083 | 1.9980000000000002 | 2.3892333333333085 | 0.034999999999996145 | 2.5 |
| hrv1_920015 | LOST_GRASP | SUCCESS_12S | 1.4999999999998352 | -0.4874666666666375 | 1.9980000000000002 | 2.4854666666666376 | 0.07499999999999177 | 2.5 |
| hrv1_920069 | DROP_AFTER_HANDOFF | SUCCESS_12S | 0.6999999999999229 | -0.4950666666666672 | 1.9980000000000002 | 2.4930666666666674 | 0.034999999999996145 | 2.5 |
| hrv1_leftover_hard_910207 | LOST_GRASP | SUCCESS_12S | 0.6999999999999229 | -0.48216666666666863 | 1.9980000000000002 | 2.480166666666669 | 0.034999999999996145 | 2.5 |
| hrv1_leftover_hard_910231 | LOST_GRASP | SUCCESS_12S | 0.6999999999999229 | -0.4904333333333344 | 1.9980000000000002 | 2.488433333333335 | 0.034999999999996145 | 2.5 |

**Conclusion:** GRPO return advantage on these ICs is **primarily SUCCESS_12S terminal (+2 vs fail bases)**, not the ~0.03–0.08 time-term difference from earlier HANDOFF. Earlier HANDOFF enables SUCCESS; the time term only ranks among successes.

## Phase 4 — HANDOFF counterfactuals

| IC | SFT earlier SUCCESS prefixes | earliest SUCCESS t | actual handoff | actual SFT success |
|---|---:|---:|---:|---|
| hrv1_920047 | 0 | None | 1.5399999999998308 | False |
| hrv1_920064 | 0 | None | 0.7399999999999185 | False |
| hrv1_920106 | 0 | None | 0.7399999999999185 | False |
| hrv1_920015 | 0 | None | 1.5399999999998308 | False |
| hrv1_920069 | 0 | None | 0.7399999999999185 | False |
| hrv1_leftover_hard_910207 | 0 | None | 0.7399999999999185 | False |
| hrv1_leftover_hard_910231 | 0 | None | 0.7399999999999185 | False |
| s7_p25_a+0.000_e0.008 | 12 | 0.19999999999997797 | 2.4999999999997247 | True |

### s7 (detail)

- SFT actual: `SUCCESS_12S` @ recovery_t≈**2.50 s**
- GRPO actual: `DROP_AFTER_HANDOFF` @ recovery_t≈**1.72 s**
- Along **GRPO’s own trajectory**, forced HANDOFF at prefixes **t∈[0.2, 0.8] s succeeds**; prefixes **t≥1.0 s** (including actual 1.72) fail.
- So s7 is **not** “failed because HANDOFF was too early.” GRPO **passed through a success window**, continued recovery actions that left a worse state, then handed off and failed. Later-than-actual SUCCESS prefixes: **0**.
- Along **SFT’s trajectory**, forced HANDOFF also succeeds from ≈0.2 s onward (SFT could have handed off earlier too).

Full table: `handoff_counterfactual.csv`.

## Phase 5 — Matched Sliding vs Recatch

Matched dual SUCCESS pairs: **6** / 10 tried supervised multi-strategy ICs (same snapshot; legal Sliding oracle + Vertical Recatch oracle; both `SUCCESS_12S`).  
Duration for Recatch = `vertical_rollout` end relative time `result["end"]["t"]` (fixed; earlier CSV mistakenly used missing `recovery_elapsed`→0).

| IC | T_slide | T_recatch | ΔT (R−S) | R_slide | R_recatch | ΔR | ΔR_time |
|---|---:|---:|---:|---:|---:|---:|---:|
| ball_train_v2_0002 | 1.22 | 0.72 | −0.50 | 1.939 | 1.964 | +0.025 | +0.025 |
| ball_train_v2_0041 | 1.54 | 0.72 | −0.82 | 1.923 | 1.964 | +0.041 | +0.041 |
| ball_train_v2_0053 | 1.22 | 0.74 | −0.48 | 1.939 | 1.963 | +0.024 | +0.024 |
| ball_train_v2_0098 | 1.22 | 0.88 | −0.34 | 1.939 | 1.956 | +0.017 | +0.017 |
| ball_train_v2_0103 | 1.54 | 0.74 | −0.80 | 1.923 | 1.963 | +0.040 | +0.040 |
| ball_train_v2_0155 | 1.22 | 0.74 | −0.48 | 1.939 | 1.963 | +0.024 | +0.024 |

- Recatch faster in **6/6** matched pairs (ΔT ≈ −0.34…−0.82 s).
- ΔR equals ΔR_time exactly (both SUCCESS → only λ·Δt differs); magnitude **≤0.041**, ≪ success/fail gap (~2.5).
- This shows a mild time-term preference for the faster mechanism **among successes**, not that λ causes fail→success flips.

## Phase 6–7 — Sensitivity & judgment

```json
[
  {
    "lambda": 0.0,
    "worst_success": 2.0,
    "best_after_handoff_fail": -0.3,
    "best_drop_during_recovery": -0.9,
    "failure_can_outrank_success": false
  },
  {
    "lambda": 0.0125,
    "worst_success": 1.95,
    "best_after_handoff_fail": -0.3,
    "best_drop_during_recovery": -0.9,
    "failure_can_outrank_success": false
  },
  {
    "lambda": 0.025,
    "worst_success": 1.9,
    "best_after_handoff_fail": -0.3,
    "best_drop_during_recovery": -0.9,
    "failure_can_outrank_success": false
  },
  {
    "lambda": 0.05,
    "worst_success": 1.8,
    "best_after_handoff_fail": -0.3,
    "best_drop_during_recovery": -0.9,
    "failure_can_outrank_success": false
  },
  {
    "lambda": 0.1,
    "worst_success": 1.6,
    "best_after_handoff_fail": -0.3,
    "best_drop_during_recovery": -0.9,
    "failure_can_outrank_success": false
  }
]
```

**Primary: C + partial B (not A).**
- Among 7 GRPO-only ICs, return gap is dominated by **terminal SUCCESS vs failure** (7/7 with |Δterminal| > |Δtime|), not by the λ·Δt term alone.
- Accounting asymmetry **exists** (recovery clock stops at HANDOFF; nominal untaxed) — this is intentional efficiency shaping, not a SUCCESS-vs-FAIL inversion under λ₀.
- Matched-IC Sliding vs Recatch: **6** dual SUCCESS pairs; Recatch faster 6/6; time-term alone decides SUCCESS ranking (ΔR=ΔR_time ≤0.041).
- Under λ₀, failure cannot outrank worst SUCCESS: `True` (order bounds).
- **Do not modify λ this round.** Offline sensitivity only.

## 中文结论（Q1–Q9）


1. **时间惩罚如何计算？**  
   终端标量：成功时 `R = 2.0 − 0.05 × recovery_elapsed`（秒），见 `training/reward_v2.py` 的 `TIME_PENALTY` 与 `reward_components`。不是逐步固定扣分；不计 nominal 12s。YAML `lambda_time` **未接入**该模块。

2. **HANDOFF 是否造成时间成本不对称？**  
   **是。** `recovery_elapsed` 在交权时冻结；nominal 不再累加时间惩罚。成功轨迹之间，更早 HANDOFF 严格提高 return（ΔR = 0.05·Δt）。这是 recovery-efficiency 设计，在 λ₀ 下不会让失败压过最差成功（`assert_reward_order`：最差成功 1.800 > 最佳交权后失败 -0.300）。

3. **7 个 GRPO-only 是否主要由时间项驱动？**  
   **否。** 平均 |ΔR_terminal|≈2.500，|ΔR_time|≈0.046；7/7 个 IC 终端项主导。时间项只解释「同为成功时谁更高」，不能解释「失败→成功」的主差距（约 2.3–2.7）。

4. **SFT 是否存在更早合法成功窗口？**  
   GRPO-only 集合中，有更早于实际 HANDOFF 且反事实 SUCCESS 的 SFT 前缀的 IC 数：**0/7**（见报告 Phase 4 表与 CSV）。若某 IC 为 0，则 SFT 前缀上提前交权并不能成功——继续 recovery 未必是「毁掉已可交权状态」。

5. **s7 是否因过早交权失败？**  
   **不是“过早交权”这么简单。** GRPO 实际在 t≈1.72s 交权后 `DROP_AFTER_HANDOFF`；但沿其轨迹在 t∈[0.2,0.8]s **强制 HANDOFF 可 SUCCESS**，t≥1.0s 失败。属于**错过/驶出可交权窗口后仍交权**。SFT 在 t≈2.50s 成功，且其轨迹上从 ≈0.2s 起强制交权也可成功。

6. **Recatch 是否真比 Sliding 快？**  
   有 **6** 条 matched-IC 双成功证据（同 snapshot、合法 oracle）；Recatch 均更快（ΔT≈−0.34…−0.82s）。样本小，不可外推到全部 IC / GRPO_only 集。

7. **时间惩罚是否系统性压低成功 Sliding？**  
   在这 6 对中，Sliding 更长 → return 低 **0.017–0.041**（纯时间项）。存在方向性偏好，但幅度远小于成功/失败鸿沟（~2.5），**不足以**解释 GRPO_only 成败翻转，也不能单独称为“系统性摧毁 Sliding”。

8. **减小 λ 会否破坏 SUCCESS 优先？**  
   在测试的 λ∈{0,0.25,0.5,1,2}λ₀ 下，最差成功仍高于最佳失败（见 `sensitivity_summary.json`）。减小 λ **不会**在公式层破坏成功优先；但离线重排 ≠ 重训后行为。

9. **下一步改什么？**  
   **保持现状（不改 λ / 不改 HANDOFF accounting）作为默认建议。** 证据更支持：GRPO 增益来自更早进入可 SUCCESS 的交权时机（解释 C），外加存在但不致命的计时不对称（解释 B）。**不支持**「时间惩罚过强导致 Sliding 被系统性抑制」（解释 A）作为本轮主因。若继续研究，应优先做 HANDOFF 时机/可交权状态分类，而不是先改 λ。
