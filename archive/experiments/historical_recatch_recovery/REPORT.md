# 早期 privileged airborne recatch 的证据审计

审计只读 Git 对象和当前工作树。没有改控制器、K48、`TABLE_DROP` 或 reward，没有训练，没有覆盖旧轨迹，也没有新的大规模搜索。没有独立复现出历史成功。

## 找到的材料

工作树里搜不到 `0.687`、`30/54`、`privileged_chase`、`25.4 mm`。这些句子只存在于提交 `042b9c4`（2026-10-10，*Add the balanced unified recovery policy…*）。同日的 `98e679f` 把这些报告从工作树删掉了，对象仍可用 `git show 042b9c4:<path>` 读出。仓库里没有 10 月 8 日的提交。最近的前序提交是 `741c5ba` 和 `696c74c`（都是 2026-10-05）。

| 线索 | 提交中的报告 | 当前是否还在工作树 |
| --- | --- | --- |
| 实验 A，30/54，−0.687 / −0.682 m/s | `results/diagnostics/UNCONSTRAINED_AIRBORNE_RECATCh_EXISTENCE.md` | 否，仅 `042b9c4` |
| 实验 C，25.4 mm，74 ms，−47 m/s² | `results/diagnostics/UNCAPPED_CARTESIAN_AIRBORNE_CHASE.md` | 报告否；脚本 `training/uncapped_cartesian_airborne_chase.py` 仍在 |
| 实验 B 的近邻 | `results/diagnostics/FULL3D_RECATCh_BASIN_AUDIT.md` 与 `training/full_3d_airborne_recatch_dataset.py` | 报告否；脚本仍在 |

报告点名的原始产物都不在 Git，也不在磁盘：

- `results/diagnostics/raw/unconstrained_airborne_recatch/videos/privileged_chase.mp4`
- `below_velocity_match.mp4` 及同目录 `.npz`
- `results/diagnostics/raw/uncapped_cartesian_airborne_chase/natural_loss_13_5.npz`

`git log --all --full-history -- "*unconstrained*"` 没有生成实验 A 的 Python 文件。`results/diagnostics/predictive_airborne_capture/` 不存在。

因此实验 A 的 30/54 和两条速度数字**有报告文字支持，没有可重放轨迹**。不能把报告中的 “pads still on 1 s later” 当成现在的 `SUCCESS_12S`。

## 每个历史数字

| 数字 | 证据 | 结论 |
| --- | --- | --- |
| 54 组、30 组 1 秒后仍双侧接触 | `UNCONSTRAINED_…md` 第 33、37 行 | 报告如此写。生成脚本和 54 行日志不在仓库 |
| 脱离接触约 60 ms（4–64 ms 双侧为 0，72 ms 再接触） | 同文件第 39、41 行 | 仅报告 |
| 再捕获 hand vz −0.687、cylinder vz −0.682、峰值 0.92 m/s | 第 41 行，get-below 那一组 | 仅报告 |
| 桌面和地面碰撞关闭 | 第 3、67 行 | 报告明确写了。这不是当前场景 |
| 约 12 条 full-3D demonstration | 全仓库无 “12 demos” 原文 | **没有原始证据**。现存脚本会保存 `ok` 的 v2 轨迹，数量由运行决定，本审计没有跑 |
| 平移 ±1 m/s、旋转 ±4 rad/s | `full_3d_airborne_recatch_dataset.py` 使用 `AIRBORNE_V2` | 脚本如此映射。不是现在的 v3（向下可到 3.8 m/s） |
| 横向 +0.25 / −0.21 / +0.37 | 未在上述报告中作为一组速度出现。`FULL3D_…md` 表里的 0.25、0.37 是别的列 | **不能算已核实** |
| 间隙 25.44 mm，74 ms，a_z −47.2 m/s²，关节 2 和 4 力矩 87 N·m | `UNCAPPED_…md` 第 3、5、35 行 | 报告如此写。对应 `.npz` 不在仓库 |
| 保留 ±1 m/s 则失败 | 同文件第 34、71 行 | 报告写最小余隙仍有 22.7 mm，没有再抓 |

## 当时的控制器实际做了什么

### 实验 A（报告中的 privileged chase）

报告描述的执行链：

1. 起点是已经坐稳的水平抓握，物体高度约 0.593 m，手高约 0.690 m，手系相对位置 `[0.2, 0.0, 97.6]` mm。这不是现在 ball-impact 快照里沿指长偏出 15–25 mm 的状态。
2. 每步从仿真器读圆柱位姿和速度，把手的目标设到“圆柱回到测得的 pinch”的位置，get-below 再把目标下移 40 mm。
3. 速度指令是圆柱速度加上位置误差的倍数，再截到 ±4.5 m/s。力矩只受执行器限制（关节 1–4 为 87 N·m，5–7 为 12 N·m）。
4. 手指用 +18 N·m 张开，开口过约 34 mm 后降到 +4 N·m。进入 pinch 窗且相对速度低于阈值后用 −18 N·m 闭合。双侧接触保持 40 ms 后手臂放松。
5. 成功定义：中间有一段无接触，然后再接触，双侧都在，**再过 1 秒仍抓住**。物体高度可以落到 0.35–0.40 m，因为桌子和地板不再碰撞。
6. 报告写夹爪在 60 ms 窗口里张开到 49 mm。没有视频可核对画面。

这不是 7D action，也不是 K48。报告自己写明这是特权可达性，不是机载观测能跑的策略。

### 实验 C（脚本还在）

`training/uncapped_cartesian_airborne_chase.py`：

- `drop_supports` 把 `floor` 和 `table` 的 `contype/conaffinity` 置 0。
- 每 2 ms 把 `p_des` 设成特权拦截点 `object_pos - R_des @ pinch`，`v_des = v_object + 200 * 位置误差`。无 clip 的一组不截这个速度；对比组截到 ±0.08 或 ±1.0 m/s。
- 力矩走 `TRACK`（`kp_pos=1400`，`kd_pos=80`），再被执行器截断。
- `preset_fingers_open` 直接写手指 `qpos = 0.04`。这不是力矩张开。
- 报告中的成功是 0.074 s 时余隙 −3.8 mm、相对竖直速度 +0.02 m/s，并且随后抓住。同样不是 12 秒名义续跑。

### 实验 B（脚本还在，12 条演示未证实）

`training/full_3d_airborne_recatch_dataset.py`：

- 特权状态选出 7 维命令，经 `recovery7d_airborne_v2` 或 `local_v1` 映射后再进同一套力矩控制器。
- 追逐阶段把世界速度设成横向物体速度（截在 ±1 m/s）和竖直 −1 m/s，夹爪力矩 +2 N·m；保持阶段 −18 N·m。
- `step_mapped` **每 2 ms 积分** `p_des += v * dt`，并积分 `r_des`。
- 同样先 `drop_supports`。扰动分支还会改物体或手臂 `qpos`。
- `token_audit` 不是现在的 K48 codebook。

`FULL3D_RECATCh_BASIN_AUDIT.md` 写的是另一件事：52 个策略访问过的 pre-loss 状态，用 v3 动作范围做 1 秒再抓，直接拦截恢复数为 0。它不支持“12 条成功 full-3D 演示”。

## 和最近失败搜索的差别

最近的 `airborne_recatch_search` 和 `contact_aware_recatch_search` 都是 0 次 `SUCCESS_12S`。差别不是同一控制器调参失败，而是任务和接口都变了。

| 项目 | 10 月 10 日报告里的成功 | 现在的搜索 |
| --- | --- | --- |
| 起点 | 坐稳的水平 pinch，物体约 0.59 m，几乎无横向滑出 | ball-impact 快照，沿指长已偏 15–25 mm，自旋可达数 rad/s，自由落体窗口约 0.13 s |
| 桌子 / 地面 | 碰撞关掉，物体可以落到 0.35 m 仍算抓住 | 碰撞开着，`TABLE_DROP` 在支撑或反弹时锁存，之后不能再算成功 |
| 动作接口 | A/C：直接写 Cartesian `p_des`/`v_des`。C 的一种模式写手指 `qpos` | 只能 `recovery7d_airborne_v3` 的 7 维命令，经 `step_legal` |
| 设定点 | C 每 2 ms 把 `p_des` 放到特权拦截点。B 每步积分 `p_des` | `step_legal` 每个 20 ms 把 `p_des` **重置为当前手位置**，20 ms 内不再积分平移。位置弹簧会在同一步里刹车 |
| 速度指令 | A 截到 ±4.5 m/s；C 用增益 200，无 Cartesian 截断时加速度到约 −47 m/s² | v3 向下命令最高 3.8 m/s，实测峰值约 2.1 m/s 后振荡；先前保守搜索还把可用速度压到 0.03–0.08 m/s |
| 张开 | 报告：开口到 49 mm，超过圆柱直径后再合 | 快照开口约 13 mm，小于直径 36 mm。张开约要 80 ms，物体已经从短垫滑出 |
| 成功 | 双侧接触后再保持 1 秒 | `HANDOFF` 之后名义控制器再跑 12 秒，且中途不能 `TABLE_DROP` |

早期控制器做到、最近搜索器没有做的，是这三件事同时成立：桌子不接住圆柱、手的位置设定点跟着特权拦截点走、手指先开到大于直径再在速度对齐时合上。最近的搜索器在桌子仍有效、设定点每 20 ms 被钉回当前手位置、并且物体已经偏出指垫的快照上，用合法 7 维速度命令去追。

`contact_aware` 里最接近的 `ball_train_v2_0002` redirect 是在**还没松开**时把物体抬高，然后名义段约 3.5 秒后丢失。那不是报告里的空中再捕获。

## 可以借鉴、但不能直接当合法动作的部分

可以借鉴的时序：先把开口加到大于 36 mm，再让手的竖直速度贴近物体，相对速度小了再合拢。报告里匹配的一组在再接触时竖直相对速度只有约 0.005 m/s。

不能放进当前正式任务的部分：

- 关闭桌子或地面碰撞
- 直接写 `p_des`、手指 `qpos` 或物体 `qpos`
- 用 ±4.5 m/s 或增益 200 的特权速度，绕过 v3 和 K48
- 把“1 秒后双侧还在”记成 `SUCCESS_12S`

## 复现

没有复现。缺的是：

- 实验 A 的生成脚本、54 组日志、两条 `.npz`、两个 `.mp4`
- 实验 C 的 `natural_loss_13_5.npz`（脚本会自己再生成，但成功数字目前只有报告）
- 任何写着 12 条 full-3D 成功演示的清单

现存的 B/C 脚本一运行就会关掉桌子碰撞，C 还可以写手指关节。把它们再跑一遍会得到旧定义下的轨迹，不是当前合法任务，所以没有启动。

## 最小的下一步

若要在**当前**物理和 `TABLE_DROP` 下试历史时序，只改搜索器、不改正式控制器：每个 20 ms 仍发合法 7 维动作，但让平移设定点按该步的指令速度积分，而不是每步钉回当前手位置；同时在物体还被夹住时把夹爪开过直径，再在竖直速度接近时合上。先在一个已坐稳、横向偏差很小的快照上做几次物理步，确认开口和相对速度，再决定要不要回到 ball-impact 快照。不要为了复现旧报告去关桌子。
