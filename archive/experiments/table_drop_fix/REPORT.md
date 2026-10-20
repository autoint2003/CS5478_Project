# TABLE_DROP 判定修复

撞击后的恢复和 nominal continuation 现在记录不可逆的 `TABLE_DROP`。圆柱脱离双侧夹持后，桌面或地面承托其重量，或者一次真实落桌碰撞改变竖直速度，即失败。之后重新夹住也不能再得到 `SUCCESS_12S`。

旧判定只在无夹爪接触且高度低于 0.44 m 持续 1 s，或高度低于 0.30 m 时失败。`ball_train_v2_0051` 的 `chase_lost` 在约 3.29 s 脱爪并被桌面承托 0.288 s 后重新抓起，因此仍被判为 `SUCCESS_12S`。审计中 158 条 `ILLEGAL_TABLE_REGRASP` 全部如此。

## 1. 修改的文件和函数

- `training/table_drop.py`：新增 `consider`、`observe`、`export_state`、`restore_state`、`terminal_reason`。
- `training/privileged_ball_impact_search.py`：`step_legal` 的每个 2 ms 子步在 `mj_step` 前后读取竖直速度并调用 `observe`。
- `training/recovery_success.py`：`nominal_step` 同样采样；`nominal_validate` 在 latch 后停止；`label_of` 对 latch 返回 `TABLE_DROP`。
- `training/horizon_outcome.py`：`recovery_failed` 立即失败；`nominal_verify` 在进入 12 s 窗口前和每一步检查 latch，成功分支只在 reason 仍为空时给出 `SUCCESS_12S`；`_fail_recovery` 使用 `TABLE_DROP`。
- `training/history_aware_critic_ablation.py`：`take_capture` / `restore_capture` 保存并恢复 latch 及未完成的接触累计。旧快照没有该字段时恢复为未落桌。
- `training/unified_k48.py`、`training/unified_k47.py`、`training/benchmark_k48_prefix_branching.py`：recovery 失败不再一律写成 `DROP_DURING_RECOVERY`。
- `training/sliding_vs_regrasp.py`：`execute` 在 latch 后停止，标签为 `TABLE_DROP`。
- `training/k48_controlled_slide_search.py`：搜索中的 recovery 失败保留具体原因。
- `training/reward_v2.py`：`TABLE_DROP` 复用已有失败公式。未改系数。
- `training/test_table_drop.py`：回归测试。原始结果在 `results/diagnostics/table_drop_fix/raw/results.json`。

未改 7D action、K48 codebook、Transformer、reward 系数、圆柱质量、摩擦、碰撞参数或桌面。未重训，未覆盖旧数据、checkpoint 或既有 benchmark。

## 2. 判定定义和阈值

监测只发生在 `step_legal`（恢复，10 个 2 ms 子步）和 `recovery_success.nominal_step`（handoff 之后的 nominal continuation）。初始抓取使用 `autonomous_episode` 的步进，不调用该监测。

每个子步读取物体–桌面和物体–地面接触：geom 身份、接触法向、`mj_contactForce` 法向力在世界竖直向上的分量、接触前和接触后的 `cvel` 竖直速度。双侧手指都有接触时视为有效抓握，清零未确认的累计，不 latch。

圆柱质量 0.20 kg，重量约 1.962 N。

- 承托：脱离双侧抓握后，向上力达到重量的一半并累计 0.02 s（10 个物理步）。0.02 s 内力中断则累计清零。这只覆盖“已经几乎静止、没有明显反弹”的落桌。
- 反弹：同一次接触过程中，竖直速度相对接触开始前上升至少 0.15 m/s，且向上冲量至少达到“半重量 × 一个物理步”（约 0.002 N·s）。在产生该速度变化的子步立即 latch，不等待 0.02 s。
- 向上力低于 0.001 N 的接触不开启接触段。单纯 `ncon > 0` 或单一高度阈值不触发。
- latch 后同一 episode 内不可清除。重新出现双侧接触也不清除。

0.15 m/s 低于审计中非法轨迹约 1.3–1.5 m/s 的反弹，高于数值噪声。`ball_train_v2_0051` 在半重量力出现后的 0.004 s 内 latch，峰值向上力 38.0 N，与审计峰值一致。不使用原来的 1 s 无接触等待，因此落桌后的重抓来不及恢复成功资格。4 s recovery timeout 仍是独立失败，不是 `TABLE_DROP`。

`reward_v2` 没有新系数。恢复阶段的 `TABLE_DROP` 与 `DROP_DURING_RECOVERY` 使用同一公式；handoff 且 nominal lift 已完成后，与 `DROP_AFTER_HANDOFF` 使用同一公式。

## 3. 调用路径

生成、验证、GRPO、评估和 prefix branching 都经过同一物理步和同一 latch。

| 路径 | 入口 | 落桌如何失败 |
| --- | --- | --- |
| 数据生成与 `legal_verify` | `rebuild_k48_recovery.legal_verify` | `step_legal` 采样，`recovery_failed` 后 `_fail_recovery` 写 `TABLE_DROP`；未失败则 `finish_recovery` |
| Handoff 后的 12 s | `nominal_verify` | `nominal_step` 采样；已 latch 或中途 latch 都返回 `TABLE_DROP`，不能变成 `SUCCESS_12S` |
| GRPO rollout | `unified_k48.train_grpo` | 同上。原因字符串是 `TABLE_DROP`，不再改写成 `DROP_DURING_RECOVERY` |
| 评估 | `unified_k48.evaluate` → `roll_policy` | `_fail_recovery` / `finish_recovery` |
| Prefix branching | `benchmark_k48_prefix_branching` | 同样的 `step_legal`、`recovery_failed` 和 `take_capture` |
| 旧 replay | `sliding_vs_regrasp.execute` | latch 后停止，`label_of` 返回 `TABLE_DROP` |

`privileged_ball_impact_search.command` 仍只生成动作。它通过 `step_legal` 进入上述判定，没有第二套成功条件。

## 4. 关键案例

旧结果来自合法性审计和 manifest：这些轨迹当时是 `SUCCESS_12S`。新结果由 `legal_verify` 重放，qpos 恢复误差为 0。

| 轨迹 | 旧结果 | 新结果 | 证据 |
| --- | --- | --- | --- |
| `ball_train_v2_0051` `chase_lost` | `SUCCESS_12S` | `TABLE_DROP` | bounce，t = 3.426 s，vz 从 -1.267 m/s 上升 0.341 m/s，冲量 0.135 N·s，峰值 38.0 N，z = 0.430 m |
| `ball_train_v2_0226` `release_long` | `SUCCESS_12S` | `TABLE_DROP` | bounce，t = 3.356 s，dv = 0.333 m/s，峰值 37.2 N |
| `k48r_train_00135` `release_short` | `SUCCESS_12S` | `TABLE_DROP` | bounce，t = 3.468 s，dv = 0.304 m/s，峰值 34.3 N |
| `inward_extra_250005` `inward_reseat` | `SUCCESS_12S` | `SUCCESS_12S` | 未 latch |
| `slide_extra_260013` `controlled_slide` | `SUCCESS_12S` | `SUCCESS_12S` | 未 latch |
| `ball_train_v2_0070` `controlled_slide` | `SUCCESS_12S` | `SUCCESS_12S` | 未 latch |
| `ball_train_0017` `inward_reseat` | `SUCCESS_12S` | `SUCCESS_12S` | 未 latch |

另外两项物理测试：

- 从 `inward_extra_250005` 的抓持状态张开夹爪，圆柱落桌并在 t = 3.436 s 以 bounce latch（dv = 0.295 m/s）。随后闭合夹爪，latch 仍在，`finish_recovery` 返回 `TABLE_DROP`。
- `slide_extra_260013` 的两条恢复动作本身不落桌。恢复结束后张开夹爪，t = 3.290 s bounce latch，再进入 nominal continuation 仍是 `TABLE_DROP`。

边界测试不依赖这两条成功轨迹：双侧抓持下 40 N 桌面力不 latch；0.018 s 半重量承托不 latch，0.020 s latch；力中断 0.02 s 会清掉未完成累计；单步 30 N 且速度由 -1.2 m/s 变为 0.3 m/s 立即以 bounce latch；只有竖直速度变化、没有向上力时不 latch。

当前 manifest 重放中没有可验证的 `LEGAL_RECATCH`。空中 recatch 的保护来自规则本身：双侧手指仍接触时忽略桌面力；未达到 0.15 m/s 反弹且半重量承托短于 0.02 s 的接触视为轻触。

## 5. Snapshot / restore

`take_capture` 写入 `table_drop` 的副本，包含 latch、接触段、承托时间和冲量。`restore_capture` 在 `mj_forward` 之后安装一份新的副本。Prefix branching 的 `take_snapshot` 保存这份 capture，`restore_snapshot` 先恢复 capture，再 `mj_copyData`。

测试结果：

- 旧的 `ball_train_v2_0051` IC 快照恢复后未落桌。
- 落桌后的 prefix snapshot 恢复后仍是 `TABLE_DROP`。
- 只复制 `mjData` 不会把 latch 写回来，也不会把已经 latch 的状态清掉。
- 从同一状态分出的未落桌快照和已落桌快照互不覆盖。
- 导出的字典被外部改成未落桌时，仿真实例中的 latch 不变。
- 没有 `table_drop` 字段的 capture 恢复为未落桌。

## 6. 未解决的问题和误判边界

- 脱离双侧抓握后，向上速度变化小于 0.15 m/s、且半重量承托短于 0.02 s 的接触仍视为轻触。审计中的非法落桌都在冲击步上超过该速度变化。
- 双侧手指一直保持接触时，圆柱底部碰到桌面不会 latch。这是为了保留 controlled slide 和 inward reseat。若一条轨迹在双侧夹持下把桌面当作主要支撑，当前规则不会把它标成 `TABLE_DROP`。
- 没有一条现成的合法空中 recatch 可供重放。不能用文件名证明这类轨迹存在。
- 一些历史诊断脚本仍把 recovery 失败写成 `DROP_DURING_RECOVERY`。当前生成、GRPO、评估和 prefix branching 不走那些脚本。

## 7. 数据和 checkpoint

需要重新生成 SFT 示范，并重新评估旧 checkpoint。现有 recatch 示范里，审计标为 `ILLEGAL_TABLE_REGRASP` 的 158 条在新规则下不再具有成功资格；其中 154 条训练行的样本权重合计为 3。`inward_reseat` 和 `controlled_slide` 在本次抽查中仍为 `SUCCESS_12S`。

本次没有重新生成数据，没有重训，也没有重跑旧 checkpoint 评估。旧的 verified npz、审计报告和 benchmark 结果保持原样。
