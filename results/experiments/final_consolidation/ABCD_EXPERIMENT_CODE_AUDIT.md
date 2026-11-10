# A/B/C/D 四方案提出后 — 实验代码全量核对

**范围起点：** `experiments/precision/p51_four_class_mixed_ppo`（四类 A1/B1/C/D 混合正式提出）  
**范围终点：** `p84_nonC_mechanism_discovery`（+ `final_consolidation`）  
**方法：** AST 语法、`experiments.*` 导入解析、入口/指标/门控扫描、关键路径人工阅读、结果目录存在性、已核实的面板统计。  
**约束：** 只读；不训练；不新开评估战役。

---

## 总判

| 级别 | 结论 |
|---|---|
| CRITICAL（会自动训错集 / 写坏 final-test / 导入断裂） | **未发现** |
| HIGH（会错误表述“测到了什么系统 / 门控误触发”） | **有：P83 V2 未激活；P69 噪声触发 PPO** |
| MAJOR（指标制度切换、样本重叠、入口不统一等） | **有，需在报告中显式披露** |
| 代码可运行性（语法 / 导入 / 产出目录） | **整体健康** |

静态扫描摘要：

- 包数量：**35**（P51–P84；含双 P63）
- Python 语法错误：**0**
- 解析到的 `experiments.*` 导入：**1281**；缺失模块：**0**
- `results/experiments/pXX_*`：**35/35 均有产出** 且含报告类文件
- decide/run 中 `auto_start=True` / `authorize_training=True` / `subprocess` 启动训练：**未扫到**

---

## 时代划分（便于阅读）

| 时代 | 包 | 角色 |
|---|---|---|
| **E1 立制** | P51–P54 | 四类混合 PPO → 独立采样 vs 树 → GRPO64 → 重平衡数据集 |
| **E2 稳态训练** | P55–P64 | Expert takeover / 续训 / 稳定化 / C 专项 / 可复现联合评估 / Critic-PPO → **P64 冻结 Actor** |
| **E3 诊断收束** | P65–P84 | 续训探针 → R3F/接触几何/序列/V2 → Critic/观测/动作 CF → 失败预算 → **P84 停止** |

---

## HIGH（必须写进报告）

### H1. P82/P83 面板上 V2 名存实亡

已在 `panel_32.pkl` 核实：`forced_steps>0` 仅 **2/128**（皆 B1、无接触、失败）。  
原因：V2 `require_no_contact`；面板 IC 起步多已接触。  
→ **不可把 P83 MSP=0.273 写成 “P64+V2 闭环成绩”**；应写 ≈ **P64 Actor-only**。  
P70 上 V2 的增益属于**另一分布**，不能外推到该面板。  
详见 `WORKFLOW_AUDIT.md`；与 [P65–P84 审计](26d23fa2-8861-4e5a-b8c0-7f768daf00b9) 一致。

### H2. P69 在噪声增量上触发 PPO（同上审计）

`fixed_intervention.py` 门控 `r3_up/geom_up` 阈值极低（报告 ΔSUCCESS≈**2/128=0.0156**）仍使 `proceed_to_ppo` 为真并实际跑了 Phase 4 pilot。  
另：门控使用特权 `has_pad_contact`，却标注 `"obs_only": True`；`--phase 4` 可读陈旧 gate 文件。  
→ **P69 PPO pilot 不应作为“序列干预已验证可训”的证据**。

---

## MAJOR（跨包）

### M1. 主成功指标制度切换（P51–P52 vs P53+）

| 阶段 | 环境 / 主指标 |
|---|---|
| P51–P52 | `MixedRecoveryEnv` / 树搜索；主看 **SUCCESS_12S**（及 SUCCESS_NEW） |
| P53+ | `SUCCESS_RELAXED_12S` → 分析键 `SUCCESS_RELAXED` / `MSP_RELAXED` |

**禁止**把 P51 SUCCESS_12S 与 P64/P83 MSP_RELAXED 直接纵向对比成同一曲线。

### M2. 诊断面板与 P64 训练 IC 重叠

P83 约 **120/128** traj 出现在 `IC_EXPOSURE.jsonl`。适合失败结构诊断，不是训练外泛化成绩。

### M3. `blocked_eval_ids()` 对 P54 池基本空操作

Blocked 集（P61/P62 heldout 命名）与 P54 schedule 零交集；P82/P83 过滤弱于训练用的 `_is_blocked`（缺 root）。当前面板无泄漏，但是脆弱代码。

### M4. P51–P59 入口不统一

有 `run_pXX.py`：P51–P54、P60–P84（多数）。  
**无统一 run_**：P55（多脚本 `__main__`）、P56–P59（`train_*` / `heldout_eval` / `write_*`）。  
复现需按各包 README/脚本名手动编排，不是单一 orchestrator。

### M5. C 专项包仍依赖四类回归护栏（设计如此，非 bug）

P58/P59/P66 等以 C 为主优化，但代码中保留 A/B/D MSP 灾难回归检查 — 核对通过；不要误读为“只训 C 却声称四类联合最优”。

### M6. 匹配 CF / 全回合符号不一致（P79/P81）

已知：匹配 Δ 正、全回合 MSP Δ 负。决策停 fixed-token 正确；勿把匹配增益写成可部署改进。

### M7. P83 Case C / “non-C dominates” 不对称（升级表述）→ **已正式修订**

`analyze`/`decide` 曾用 **单类 C 的 contact-formation 子集** 对比 **三类 A1+B1+D 的任意失败质量 SUM**（再乘 1.25 门槛）。  
→ “non-C dominates” 很大程度上是 **3 对 1 的聚合假象**。  
**2026-11-07 post-audit revision:** 正式决策改为 **Case D / `stop_consolidate_baseline`**，保留 SUM=0.508 与 mean=0.169 双记录；见 `P83_DECISION.json` 与 `POST_AUDIT_DECISION_REVISION.md`。

### M8. P55 BC 对照无效（[P51–P64 审计](7b0a85d3-c7e4-415a-ac15-0904852613c4)）

`p55_expert_takeover/train_comparison.py` 的 `_bc_loss` 对观测喂 **`zeros(T,39)` / `zeros(T,7)`**，再用末步 logit 扩成序列做 CE。  
→ **不能学习状态条件行为**；P55 “expert takeover vs on-policy” 结论作废。P56 若从 P55A 初始化，继承该弱点。已本地确认源码含 `obs = torch.zeros(...)`。

### M9. P60 盘上数字来自旧门控（同上审计）

历史 feasibility/label 缺陷见审计叙述；门控后续有补丁且 `test_p60_labels_gate` 可通过，但 **磁盘上旧 P60 数值不应再引用为有效证据**。

### M10. P62→P64 同一 16-IC dev 选模链（同上审计）

P62_epoch3 / P63A / P64_step4 均在同一小 dev 集上选优与报告；P64 **未**在 untouched final-test 上评估。  
→ “P64_step4 champion MSP≈0.27” 是 **dev 上多检查点最大值**，泛化未证实。物理 twin root（仅剥 `plan_b__`/长 hex）在 train↔holdout/final 间仍可能重叠。

### M11. 其他 P51–P64 中等项（汇总）

- `hash(traj_id)` 种子：P58/P59/P60（跨进程不可复现）；P61 起改 SHA-256。  
- `build_p54_dataset` 每次调用重写 manifest（含被 P62–P64 采样触发）。  
- P53 PID lock 在 Windows 上 `os.kill(pid,0)` 不安全。  
- P52 顶层 run 不完整；仅有 inconclusive 子报告。  
- 多包单 seed / 效应小于自报噪声。

### M12. P65–P84 补充 MAJOR（[P65–P84 审计](26d23fa2-8861-4e5a-b8c0-7f768daf00b9)）

- **P70/P71：** V2 调参/验证基本 **C-only**；A1/B1/D 未对 V2 重滚验证（P71 甚至写 “A/B/D unchanged, not re-rolled”）。  
- **P66：** docstring 称 pilot 门控，`--phase all` 可直跑短 PPO；4 IC/类评估噪声大。  
- **P72–P74/P79–P81：** 多处 C-only 快照外推四类；缺 artifact 时报告默认 Case E/retain；硬编码 “verified” 旗标。  
- **P73：** V0 vs V2/P73 的 ever_opp/e9 起算窗口不一致。  
- **P77 vs P78、P80 vs P81：** 相继 Case 互相打架（无 CI / 匹配≠全回合）— 以更严的后者为准停训是对的。  
- **P82–P84：** 不记录 V2 激活率；缓存不按 ckpt 哈希键控；P84 将可恢复性硬编码为 False 使 Case A 不可达（与 stop 一致，但是断言非验证）。  
- **final_consolidation 图：** 数值已与 panel 对齐，但架构图仍暗示 V2 在 P83 面板上活跃 — 表述需改（已在 `WORKFLOW_AUDIT` / Manifest 披露）。

---

## 分包核对表（P51–P84）

图例：`OK` = 无新发现重大实现漏洞；`MAJ` = 有 MAJOR 披露点；`HIGH` = 影响主结论表述。

| 包 | 入口 | SKILLS/四类 | 结果目录 | 状态 | 要点 |
|---|---|---|---|---|---|
| **p51_four_class_mixed_ppo** | `run_p51.py` | A1,B1,C,D | 有 | MAJ | 四类立制；SUCCESS_12S；holdout 模糊匹配 / twin root |
| **p52_independent_vs_tree_sampling** | `run_p52.py` | A1,B1,C,D | 有 | MAJ | 顶层 run 不完整；结论 inconclusive |
| **p53_independent_grpo64** | `run_p53.py` | A1,B1,C,D | 有 | MAJ | RELAXED 切换点；Windows PID lock；小 pilot |
| **p54_rebalanced_grpo64** | `run_eval.py` + `train_full_epoch` | A1,B1,C,D | 有 | MAJ | 每次 build 重写 manifest；A 配额不足有文档 |
| **p55_expert_takeover** | 多 `__main__` | A1,B1,C,D | 有 | **MAJ** | **BC stub 喂零观测 → 对照无效** |
| **p56_continued_grpo64** | `train_full_epoch` | A1,B1,C,D | 有 | MAJ | 承 P55A；单 run |
| **p57_post_recatch_stabilization** | `train_full_epoch` | A1,B1,C,D | 有 | OK | `StabilizeGrpoEnv` 标准环境 |
| **p58_fast_c_grpo** | `train_focused`/`write_reports` | C 主 + ABD 护栏 | 有 | MAJ | val 偏 in-sample；`hash()` 种子 |
| **p59_expanded_c_grpo** | `train_focused`/`heldout_eval` | 同上 | 有 | MAJ | 同上 |
| **p60_c_stable_state_grpo** | `run_p60.py` | C 状态机 | 有 | **MAJ** | **旧盘结果勿用**；CF 状态 API 仍复用 |
| **p61_reproducible_joint_eval** | `run_p61.py` | A/B/C/D heldout | 有 | OK | SHA-256 种子；MSP_RELAXED 标准 |
| **p62_grpo_optimization** | `run_p62.py` | 四类 heldout/final | 有 | **MAJ** | 16-IC dev 选模；C twin；定义 blocked 源 |
| **p63_transformer_critic_ppo** | `run_p63.py` | 四类 | 有 | OK/MAJ | PPO/GAE 结构 OK；与 63b IC 日程不可互换 |
| **p63_transformer_ppo_c_grasp** | `run_p63b.py` | C 侧重 | 有 | MAJ | 短 pilot；效应≈噪声却推 Branch A |
| **p64_ppo_multiround** | `run_p64.py` | 四类 stratified | 有 | **MAJ** | **冻结权重可用**；champion 声明弱（同 dev 最大） |
| **p65_ppo_continuation** | `run_p65.py` | 四类 | 有 | MAJ | 默认可训；integrity 旗标硬编码；选模噪声 |
| **p66_c_curriculum_ppo** | `run_p66.py` | C 课程 + 对比 | 有 | **MAJ** | phase all 绕过表述门控；4 IC/类噪声 |
| **p67_r3f_gate_diagnosis** | `run_p67.py` | 诊断 | 有 | OK | |
| **p68_precontact_geometry** | `run_p68.py` | 诊断 | 有 | OK | C 假设生成，非 held-out |
| **p69_skill_sequence** | `run_p69.py` | 序列假设 | 有 | **HIGH** | **噪声触发 PPO**；特权接触却标 obs_only |
| **p70_geometry_to_r3f** | `run_p70.py` | V2 变体 | 有 | **MAJ** | **C-only** 调/验；勿外推 P83 |
| **p71_contact_safe_alignment** | `run_p71.py` | V2 规格源 | 有 | **MAJ** | ABD 未重滚；`require_no_contact` 致 P82+ 静默 |
| **p72_postcontact_reliability** | `run_p72.py` | 诊断 | 有 | MAJ | 缺产物默认 Case E；C-only |
| **p73_first_contact_bilateral** | `run_p73.py` | overlay CF | 有 | MAJ | 指标窗口不一致；retain V2 仍成立 |
| **p74_contact_learning_readiness** | `run_p74.py` | 诊断 | 有 | MAJ | ownership 数据 C-only |
| **p75_critic_calibration** | `run_p75.py` | Critic | 有 | OK/MAJ | Critic-only 允许；策略标签含 V2 未核激活 |
| **p76_value_observability_audit** | `run_p76.py` | 观测 | 有 | OK | 死路径 `P73_CF` |
| **p77_contact_observability** | `run_p77.py` | 传感可行性 | 有 | MAJ | Case A 点估计、无 CI；被 P78 打脸 |
| **p78_action_relevant_representation** | `run_p78.py` | Case C | 有 | OK | 纠正 P77；CF n_pairs 极小 |
| **p79_matched_cf_coverage** | `run_p79.py` | CF | 有 | MAJ | 审计失败仍继续；C-only；V2 描述不符 |
| **p80_policy_learning_feasibility** | `run_p80.py` | Case B | 有 | MAJ | 硬编码 P79 数字；被 P81 全回合打脸 |
| **p81_static_contact_bias** | `run_p81.py` | Case B | 有 | MAJ | 匹配≠全回合；特权注入门；n 小 |
| **p82_global_failure_budget** | `run_p82.py` | A1,B1,C,D | 有 | HIGH | 未断言 V2 激活；标签语义异于 P72/73 |
| **p83_recoverable_failure_mass** | `run_p83.py` | A1,B1,C,D | 有 | HIGH | V2 未激活；non-C 聚合假象；训练重叠 |
| **p84_nonC_mechanism_discovery** | `run_p84.py` | B1 门控 | 有 | MAJ | Case D stop OK；建立在 Actor-only 面板上 |

---

## 已核实 OK 的横切项

1. **四类标签稳定：** 显式 `SKILLS=("A1","B1","C","D")` 出现在 P51–P57、P82–P83；其余包通过 heldout/schedule/`by_class` A/B/C/D 使用同一语义。  
2. **P53+ 成功键链路：** env `SUCCESS_RELAXED_12S` → eval/`_r3_terminal` → `SUCCESS_RELAXED` / `p_relaxed` / `MSP_RELAXED`。  
3. **冻结推理接线：** P82 collect `rule=None`；V2 **规格**与 `BASELINE_FREEZE.json` 一致（激活是另一回事）。  
4. **P64 选模：** 用 dev MSP；final-test IC 未用于选 step-4。  
5. **P70–P84 decide（默认叙事）：** 多数不自动开 Actor 训；例外/弱门控见 **H2（P69）**、P65/P66 默认可训入口。  
6. **导入图完整：** 1281 个 `experiments.*` 引用均可解析到文件/包（P76 一处未使用死路径）。

---

## 与“冻结结论”的关系

| 结论 | 核对后 |
|---|---|
| 停止 P85 / 不再 overlay·fixed-token | **成立**（负向全回合证据仍有效） |
| 保留 P64 权重作为可复现检查点 | **成立**（工程冻结） |
| “P64_step4 已在独立集上证实最优” | **不成立** → 同 16-IC dev 选模链（M10） |
| “P64+V2 在 P83 上达到 MSP 0.27” | **不成立（表述）** → Actor-dominated 面板 |
| “V2 经 P70 证明可提升” | **成立，但仅限 P70/C 分布**；未在 ABD 重验 |
| P69 “序列干预已验证可进入 PPO” | **不成立**（H2 噪声门控） |
| P55 expert-takeover 对照结论 | **作废**（M8 BC stub） |
| 旧 P60 盘上数值 | **勿引用**（M9） |
| 四类方案代码链从 P51 可追溯到 P84 | **成立**（入口不齐；产出齐全） |

---

## 建议的报告写法

1. 时间线从 **P51 四类立制** 写起，标明 **P53 指标切换**。  
2. 主表用 P83，脚注：**V2 激活 2/128；≈Actor-only；非 final-test。**  
3. V2 增益单独一表（P70），禁止与 P83 并成“系统升级曲线”。  
4. 负向实验（P73/P81/P84）保持“失败的优化尝试”，不要改写成成功。

---

## CODE FIXES APPLIED（本轮，不重训 / 不重开面板）

| ID | 修复 |
|---|---|
| **M8** | P55 `_bc_loss` 改为因果 CE + 真实 `obs`；无 `obs` 的样本跳过；`build_replay` 从 capture 回放附着观测 |
| **M3** | P75/P82/P83（+P69）用 `is_blocked_eval`（含 `root_id`） |
| **H1 / latent** | `p73` controller：接触后 `rule=None` 不再调用 V2（禁 latent 重激活）；P82/P83 记录 `v2_activation` 并告警 |
| **H2** | P69 门控收紧（需 ΔSUCCESS>0.05 + R3/geom）；激活用 obs 接触；拒绝无 `succ_up` 的陈旧 gate |
| **M7** | P83 增加 `B_mean_*`；`decide` 用均值质量做 dominance（非 3 类求和假象） |
| **M11** | P53 Windows PID lock；P58/P59/P60/P82 `hash()`→`stable_u32` |
| **M12** | P76 `P73_CF` 路径改正；final_consolidation 架构图标注 V2 在 P83 上近似不活跃 |

**未改历史磁盘结果数字**（P55 旧对照、P69 PPO pilot、P83 MSP 表）：代码正确后需显式重跑才刷新产物；报告仍按审计披露解读旧产物。

---

## 辅助脚本（本目录）

- `_audit_abcd_inventory.py` — 包/入口/SKILLS/语法  
- `_audit_abcd_imports.py` — 导入完整性  
- `_audit_abcd_gates.py` — 自动训练门控与成功键计数  

（只读审计用；非训练入口。）
