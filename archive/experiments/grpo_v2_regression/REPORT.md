# GRPO v2 技能回归与全失败组诊断（只读）

目录：`results/diagnostics/grpo_v2_regression/`  
脚本：`scripts/diagnose_grpo_v2_regression.py`  
不训练、不写回任何 checkpoint。

---

## Phase 1 — 评估一致性复核

| 项目 | 结果 |
|---|---|
| Checkpoint | SFT `sliding_equal_weight/checkpoint_epoch_0040.pt`；GRPO `full_run_v2_natural_sampling/checkpoint_epoch_0005.pt` |
| 协议 | `model.eval()`；argmax logits；`roll_policy_full` / ballistic 同源；`SUCCESS_12S` 未改 |
| 复现 | Heuristic/Hard/Merged 与报告一致：**20/23 vs 20/23**（内部互换） |

### 逐 IC 转换计数（复现）

| 类型 | 数量 |
|---|---:|
| both_ok | 19 |
| both_fail | 2 (`hard_910024`, `hard_910344`) |
| SFT_ok → GRPO_fail | **1** (`s7_p25_a+0.000_e0.008`) |
| SFT_fail → GRPO_ok | **1** (`hard_910242`) |

文件：`per_ic_comparison.csv`、`phase1_counts.json`

---

## Phase 2 — 转换 IC 轨迹

### `s7_p25_a+0.000_e0.008`（Heuristic 净退化）

| | SFT | GRPO E5 |
|---|---|---|
| Outcome | SUCCESS_12S | **DROP_AFTER_HANDOFF** |
| Mechanism | Hybrid（borderline） | Hybrid（borderline） |
| Handoff | 7.20 s | 6.42 s |

- 标签类别含 horizontal 构型；**机制分类为 Hybrid，不是纯 Contact-Preserving Sliding。**
- Token 序列在 **step 2 / t≈0.06 s** 分叉：SFT token **16** vs GRPO token **14**（仍在 recovery，双侧接触均在）。
- 失败发生在 **HANDOFF 之后的 nominal continuation**（`DROP_AFTER_HANDOFF`），不是 recovery 中途丢抓。
- 两侧都有极短 both-off（恰好 0.04 s 阈值），证据偏边界；不应解读为“学会了另一种清晰 skill”。
- 轨迹：`traces/s7_p25_a+0.000_e0.008__{sft,e5}/`（ballistic，无 capture 视频）。

### `hard_910242`（Hard 新增成功）

| | SFT | GRPO E5 |
|---|---|---|
| Outcome | LOST_GRASP | SUCCESS_12S |
| Mechanism | Other/Unclear | Other/Unclear（E1 曾为 Contact-Preserving） |
| Handoff | 3.98 s | 3.28 s |

- 分叉同样很早（step 2）：SFT 选 token **14**，GRPO 在该状态直接选 **HANDOFF (47)**。
- 固定历史状态下的离线分布（SFT 前缀回放到分叉点）：
  - \(D_{\mathrm{KL}}(\pi_{\mathrm{SFT}}\parallel\pi_{\mathrm{GRPO}})\approx 1.84\)（**关键状态 KL 远大于 epoch 平均 KL**）
  - \(p(\mathrm{HANDOFF})\)：SFT **0.016** → GRPO **0.804**
  - SFT argmax token 14 的概率被压到 ≈0.054
- 物理含义：GRPO 更早把控制权交给 nominal，而不是延长 recovery 动作序列。成功是否“新机制”证据不足（Other/Unclear）；更像 **handoff timing / 早交权** 变化。
- 轨迹：`traces/hard_910242__{sft,e5}/` + mp4。

详见 `transition_deep_analysis.json`、`kl_action_distribution_audit.md`。

---

## Phase 3 — 逐 Epoch 变化

文件：`epoch_transition.csv`

| IC | SFT | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|---|
| s7_p25_a+0.000_e0.008 | ✓ | ✗ DROP_AFTER_HANDOFF | ✗ | ✗ | ✗ | ✗ |
| hard_910242 | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| hard_910277（净不变） | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |

- Heuristic 净损失在 **E1 即出现**（当时 training `kl_mean≈0.005`）。
- Hard 新增成功也在 **E1 出现**，E3 短暂回退，E4–E5 恢复。
- 两种变化 **同阶段开始**，但并非稳定同向：E3 时 Hard 得而复失。
- Contact-Preserving / Hybrid / Recatch 在成功 IC 上有波动；不能把标签变化单独当成因果。

---

## Phase 4 — G64 全失败组（B）

全 5 epoch × 127 = **635** 次 IC 更新：

| Bucket | Groups | 占更新 | Rollouts | 占 rollout | Env steps | 占 env |
|---|---:|---:|---:|---:|---:|---:|
| A 有 SUCCESS_12S | 575 | **90.6%** | 5768 | 60.0% | 2.80M | **77.5%** |
| B 全失败（G64） | 60 | **9.4%** | 3840 | **40.0%** | 0.81M | 22.5% |

- B 组来源：**Hard 53 / Heuristic 7**（Hard 主导）。
- 全部 60 个 B 组均 `optimizer_updated=True`（有 reward 方差时仍更新）。
- B 组 `return_std` 均值 ≈0.20，**无零方差跳过**；仅 2 例 `|advantage|>5`（`hard_910289` E4；`hard_910265` E5）。
- “最高 reward”的全失败轨迹 reward 仍为负（约 −0.40～−0.44），标签多为 `LOST_GRASP` / `DROP_AFTER_HANDOFF`：组内相对“较好失败”获得 **正 advantage**，强化的是 **未成功的近失败**，不是 SUCCESS。

文件：`group_success_statistics.csv`、`group_bucket_summary.csv`、`all_failure_anomalies.json`

---

## Phase 5 — 梯度（诊断探针，非历史）

实现审计（未改代码）：`training/adaptive_grpo.py` → `apply_group_update`

- 轨迹内：action mean → 组内：trajectory mean → **G64 不因条数获得 8× surrogate 权重**
- 每个 IC **一次** Adam step（IC 等权更新次数）
- KL：组内 **action mean**；`kl_coef=0.05`；grad clip 1.0

只读探针（E5 权重 + 新采样 group，**不是**当时历史梯度）：

| 探针 | IC | 结果 |
|---|---|---|
| Heuristic 成功组 | slide_quota_400738 | G8，grad_norm≈16.3 |
| 意图 Hard 全失败 | hard_910013 | 探针采样偶然 **G64_success**（1/64），不可当作纯 B 组 |
| Hard 成功组 | hard_910267 | grad_norm≈4.5 |

- cosine(启发式成功, 上述探针) ≈ **+0.095**（弱同向，非强冲突）
- cosine(Hard 成功, 启发式成功) ≈ **−0.18**
- cosine(Hard 成功, 上述探针) ≈ **−0.10**

**不能**据此断言历史训练中 A/B 梯度互相抵消；只能说明在 E5 附近，不同来源 group 的诊断梯度夹角可以接近正交甚至略反号。

文件：`gradient_audit.md` / `gradient_audit.json`

---

## Phase 6 — Reward ↔ SUCCESS

- Mixed-success 组中：**失败轨迹 reward 高于成功轨迹的 inversion = 0**（`reward_inversions.json`）。
- 成功轨迹 advantage 为正、失败为负的方向与 `SUCCESS_12S` **一致**。
- All-failure 组：reward 排序区分的是失败模式好坏，**不会**凭空造出 SUCCESS；但会抬高“慢丢/晚掉”相对“快掉”的概率质量。

**本轮未发现 reward_v2 与 SUCCESS_12S 的系统性反序。**

---

## Phase 7 — KL 与关键状态

训练日志 KL：

- Reference = Balanced SFT（init）
- 每 IC update：参与 loss 的 action 上 categorical KL 再平均
- Epoch `kl_mean`：各 IC update KL 的均值（E5≈0.085）

关键状态（hard_910242 分叉点）KL≈**1.8**，HANDOFF 概率 0.02→0.80 —— **全局平均 KL 小 ≠ 关键状态变化小。**

s7 退化发生在 E1（kl_mean≈0.005），与“晚期 KL 失控”不同步。

---

## 中文问题回答

1. **成功/失败转换：** 仅两处净互换——`s7_p25_a+0.000_e0.008` 丢失；`hard_910242` 新增。另有 epoch 内 `hard_910277` 先丢后回，E5 相对 SFT 仍成功。
2. **Heuristic 净退化机制：** 分类为 **Hybrid（边界）**，失败为 **DROP_AFTER_HANDOFF**；不是清晰的 Sliding 丧失，也不是 Recatch 标签主导。
3. **Hard 新成功：** 关键状态下大幅提高 **HANDOFF(47)** 概率并更早交权；机制标签 Other/Unclear，证据不足称“新接触技能”。
4. **B 组预算：** 占更新 **9.4%**，占 rollout **40%**，占仿真步 **22.5%**；几乎全是 Hard。
5. **错误强化？** Mixed 组无 reward 反序；B 组强化“相对不差的失败”（负 reward 中的高者）。这是相对排序效应，不是把失败标成成功。
6. **梯度冲突？** 诊断探针显示不同来源 group 可接近正交/略负相关；**无历史梯度证明** A/B 直接冲突导致 s7 退化。
7. **G64 额外 loss 权重？** Surrogate **否**（trajectory mean）。但 rollout/计算预算 **是**（条数多）；更新次数仍每 IC 一次。
8. **原技能退化 ↔ 动作概率？** s7：早期 token 16→14，最终 handoff 后掉落。hard_910242：关键状态 HANDOFF 概率剧增。全局 KL 掩盖了局部大偏移。
9. **下一步最有价值的单变量消融？** 见下。

---

## 明确结论：**D**（证据不足，需补充特定诊断/消融）

**不选 A 作为“已证明因果”：**  
s7 在 **E1、KL≈0.005** 即失败；同时 Hard 也在 E1 获得成功。B 组虽占 40% rollout，但 **时间线不支持“G64 全失败更新是 Heuristic 退化的充分解释”**。

**不选 C：** mixed 组 **0** 次成功/失败 reward 反序。

**不选 B 作为首选：** 退化不发生在 KL 抬升阶段；关键状态 KL 问题真实，但是否靠加大 KL 系数修复 s7 **未验证**。

**选 D，并给出最高优先级的下一实验（仍属单变量）：**

1. **记录/冻结“历史纯 B 组”的梯度探针**（必须在采样后确认 `success_count==0`，本次 hard_910013 探针被随机成功污染）。
2. **单独消融 B 组更新（候选 A）**：只跳过 G64 全失败更新，其余不变——用于检验 rollout 浪费与近失败强化，**不要预期必然恢复 s7**。
3. **针对 s7 的固定状态 token 监督探针**：比较 E0→E1 在 step-2 状态上 token 16 vs 14 的概率轨迹（需保存或重放 E1 checkpoint 已有）。

在完成 (1)(2) 之前，把“过滤 G64 全失败”写成已证实根因是不成立的。

---

## 产物清单

- `per_ic_comparison.csv`
- `epoch_transition.csv`
- `group_success_statistics.csv` / `group_bucket_summary.csv`
- `reward_advantage_audit.csv` / `reward_inversions.json`
- `gradient_audit.md` / `gradient_audit.json`
- `kl_action_distribution_audit.md`
- `transition_deep_analysis.json`
- `traces/**`
- `all_failure_anomalies.json` / `meta.json`
