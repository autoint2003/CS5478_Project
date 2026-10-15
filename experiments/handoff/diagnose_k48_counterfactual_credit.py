"""Offline counterfactual credit audit for GRPO v2 u174 versus u783.

Does not train and does not change reward, the codebook, or SUCCESS_12S.
Counterfactual continuations are real closed-loop rollouts from a restored
MuJoCo, controller, and Transformer history.
"""

from __future__ import annotations


import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from training.recovery_episode import make_training_env

from envs.outcomes.horizon_outcome import _contacts, finish_recovery, recovery_failed
from training.reward_v2 import grpo_return
from training.k47_codebook import decode_action, load_codebook
from training.unified_k48 import HANDOFF, load_trained_k48

OUT = ROOT / "results/diagnostics/k48_counterfactual_credit"
LIMIT = 4.0
CAP = 200
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_audit():
    text = """# K48 GRPO 实现审计

审计对象是当前仓库里的训练代码，不是评估报告。`training/action_representation_expansion.py` 是另一项历史监督表征实验，`training/unified_k48.py` 没有导入它，GRPO v2 更新不经过那个文件。

## 1. Advantage 是否整条轨迹共享

是。每个初始条件采样 `group: 4` 条轨迹。每条轨迹先得到一个标量回报，再在组内标准化。

`training/unified_k47.py` 的 `group_advantage` 丢掉 `truncated` 轨迹，对其余回报调用 `advantages`。`training/pre_grpo_sanity.py` 的 `advantages` 是 `(return - mean) / std`。标准差低于 `1e-8` 时整组 advantage 为 0，并标记跳过。

`training/unified_k48.py` 的 `train_grpo` 把这个标量用于该轨迹的全部已记录步。步对数概率先取平均，再变成一个 trajectory ratio：

```python
logp = torch.stack([... step_logprob ...]).mean()
ratio = exp(logp - behavior)
surrogate = -minimum(ratio * advantage, clip(ratio) * advantage)
```

因此同一条轨迹上的运动步和 early HANDOFF 步共用一个 advantage。Advantage 不是按步计算的。组内各轨迹的 surrogate 再取平均，所以归一化单位是轨迹，不是 token，也不是按轨迹长度加权。

## 2. Policy loss 的归一化

- 轨迹内部：各步 `step_logprob` 的算术平均。
- 组内：advantage 绝对值大于 `1e-8`、且未截断、且有记录步的轨迹，其 surrogate 再取平均。
- KL 不按步平均。它只用组里第一条非空轨迹的第一个记录状态，计算一次 48 路 categorical KL，乘 `kl_coef`（配置为 0.05）后加到 surrogate 上。
- 梯度裁剪范数上限是 1.0。没有 entropy bonus。

## 3. Token 47 是否进入 loss、KL 和 entropy

- Policy loss：early HANDOFF 被记入 `records`，`step_logprob` 对 token 47 只返回 `log_softmax` 的对应项，不加 residual 高斯项。它因此进入轨迹平均 log-probability。
- KL：`categorical_kl` 对完整 48 维 logit 做 `KL(pi || pi_ref)`，不屏蔽 token 47。参考策略是配置里的 `grpo_reference`，即 SFT epoch 40，不是上一个 GRPO checkpoint。
- Entropy：GRPO loss 里没有 entropy 项。

Forced HANDOFF 不是 token 47 的采样结果，见下一节。

## 4. Early HANDOFF 和 forced HANDOFF

恢复循环在 `training/unified_k48.py` 的 `train_grpo` 里：

- 当前仿真时间已经达到 `recovery_limit_s`（4 s）时，循环在读取下一个观测之前退出，`how = "forced"`。这一步没有 token、没有 log-probability，也不进入 `records`。
- 否则从 48 路概率采样 token。若 token 是 47，记入 `records` 后立即退出，`how = "early"`，不调用 `step_legal`，不解码 residual。
- 评估用的 `roll_policy` 用 argmax 代替采样，early 和 forced 的物理含义相同：early 是 argmax 为 47，forced 是 4 s 时钟。K48 模型带 `handoff_index`，不会再走 basin 提前交接。

训练采样还在 residual 均值上加 `N(0, SIGMA)`，`SIGMA = 0.05`（`training/final_rl_readiness.py`）。确定性评估只用均值，不加噪声。

## 5. Reward v2

`training/reward_v2.py` 的 `reward_components`：

- `SUCCESS_12S`：`2.0 - 0.05 * recovery_seconds`
- `DROP_DURING_RECOVERY`：`-1.0 + 0.10 * q_recovery`，`q_recovery = clip(elapsed / 4, 0, 1)`，没有时间惩罚
- 交接后的 `DROP_AFTER_HANDOFF`、`LOST_GRASP`、`NOMINAL_LIFT_NOT_COMPLETED`、`PHYSICAL_VERIFICATION_FAILED`：`-0.5 + 0.20 * q_nominal - 0.05 * recovery_seconds`

`q_nominal` 只统计名义抬起命令到达目标之后、双侧接触仍在、且尚未判定抓取失败的时间占 12 s 的比例。`grpo_return` 在 `reward_version: v2` 时采用这个标量。配置文件 `config/training/k48/unified_k48_recovery_v3_reward_v2.yaml` 的 `lambda_time` 是 0.05，`nominal_seconds` 是 12，`recovery_limit_s` 是 4。

## 6. 状态恢复接口

`take_capture` / `restore_capture`（`experiments/legacy/history_aware_critic.py`）保存并恢复：

- MuJoCo `mj_state`、`qpos`、`qvel`、`ctrl`、`time`、`qacc_warmstart`、actuator activation、equality、mocap
- 名义 FSM 与 switch 内部量
- support、观测滤波 `prev_e`
- Transformer 因果历史 `_obs_seq`、`_prev_seq` 和上一条命令 `_prev`

`restore_capture` 在 `mj_forward` 之后检查 `qpos` 误差，大于 `1e-6` 会抛出异常。

`roll_policy` 里的 `lost_low`（连续无双侧接触计数，满 50 步且高度条件满足则判定恢复期掉落）是循环局部变量，不在 capture 里。反事实续跑必须单独携带这个计数。不携带就会改变终止条件，不能当作匹配状态。

## 对信用分配假设的直接含义

实现确实把一个轨迹级 advantage 乘到该轨迹全部已记录步的平均对数概率上。失败轨迹里的前缀动作和 early HANDOFF 会收到同一个 advantage。这只说明更新规则的形式。它不说明 update 174 到 783 的退化就是由这个规则造成的，也不说明某条确定性评估前缀在训练采样里出现过。
"""
    _write(OUT / "AUDIT.md", text)


def _forward(model, window_obs, window_prev):
    with torch.no_grad():
        logits, mean, _act = model(window_obs, window_prev)
    last = logits[-1].float()
    probs = torch.softmax(last, dim=-1)
    order = torch.argsort(probs, descending=True)
    top = [(int(order[index].item()), float(probs[order[index]].item())) for index in range(min(5, int(order.shape[0])))]
    return {
        "token": int(torch.argmax(last).item()),
        "probs": probs.detach().cpu().numpy(),
        "p47": float(probs[HANDOFF].item()),
        "rank47": int((order == HANDOFF).nonzero()[0].item()) + 1,
        "mean": mean[-1].detach().float().cpu().numpy().astype(np.float32),
        "top": top,
    }


def _windows(env, device):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    window_obs = torch.tensor(np.stack(env._obs_seq[-160:]), device=device)
    window_prev = torch.tensor(np.stack(env._prev_seq[-160:]), device=device)
    return obs, window_obs, window_prev


def _score(outcome, limit_s):
    scored = grpo_return(outcome, limit_s, "v2", outcome.get("reward"))
    return {
        "reason": outcome.get("reason"),
        "success": bool(outcome.get("success")),
        "handoff": outcome.get("handoff"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
        "recovery_steps": outcome.get("recovery_steps"),
        "nominal_elapsed": outcome.get("nominal_elapsed"),
        "lift_completed": bool(outcome.get("lift_completed")),
        "q_nominal": outcome.get("q_nominal"),
        "return_v2": scored.get("return"),
    }


def rollout_pair(env, behavior, other, book, blob, device):
    """Deterministic behavior rollout. Also scores ``other`` on the same windows."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    how = None
    steps = 0
    rows = []
    captures = {}
    for _ in range(CAP):
        if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
            how = "forced"
            break
        capture = take_capture(env)
        obs, window_obs, window_prev = _windows(env, device)
        own = _forward(behavior, window_obs, window_prev)
        alt = _forward(other, window_obs, window_prev)
        n_l, n_r, z = _contacts(env.sim)
        obj = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], np.float64).copy()
        row = {
            "step": steps,
            "t": float(env.sim.data.time) - t0,
            "lost_low": int(lost_low),
            "token": own["token"],
            "p47": own["p47"],
            "rank47": own["rank47"],
            "top": own["top"],
            "mean": own["mean"],
            "other_token": alt["token"],
            "other_p47": alt["p47"],
            "other_rank47": alt["rank47"],
            "other_top": alt["top"],
            "other_mean": alt["mean"],
            "nL": int(n_l),
            "nR": int(n_r),
            "z": float(z),
            "obj": obj,
            "obs0": float(obs[0]),
        }
        rows.append(row)
        captures[steps] = capture
        if own["token"] == HANDOFF:
            how = "early"
            break
        _residual, action = decode_action(book, own["token"], own["mean"])
        executed = np.asarray(__import__("experiments.ball_impact.privileged_ball_impact_search", fromlist=["step_legal"]).step_legal(env.sim, action), np.float32)
        env._prev = executed
        row["residual"] = np.asarray(_residual, np.float32)
        row["grip"] = float(executed[6])
        row["qpos_after"] = np.asarray(env.sim.data.qpos, np.float64).copy()
        n_l, n_r, _z = _contacts(env.sim)
        row["nL_after"] = int(n_l)
        row["nR_after"] = int(n_r)
        steps += 1
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        row["lost_low_after"] = int(lost_low)
        if failed:
            outcome = {
                "reason": "DROP_DURING_RECOVERY",
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "recovery_steps": steps,
                "nominal_elapsed": 0.0,
                "lift_completed": False,
                "q_nominal": 0.0,
                "reward": None,
            }
            return outcome, rows, captures, t0
    if how is None:
        how = "forced"
    outcome = finish_recovery(env, blob["ref"], t0, how)
    outcome["recovery_steps"] = steps
    return outcome, rows, captures, t0


def roundtrip(env, book, row, capture):
    """Replay one recorded motion and compare qpos. HANDOFF has no physics step."""
    if row["token"] == HANDOFF:
        return None
    restore_capture(env, capture)
    _obs, _window_obs, _window_prev = _windows(env, torch.device("cpu"))
    _residual, action = decode_action(book, row["token"], row["mean"])
    from experiments.ball_impact.privileged_ball_impact_search import step_legal
    step_legal(env.sim, action)
    return float(np.max(np.abs(np.asarray(env.sim.data.qpos, np.float64) - row["qpos_after"])))


def continue_from(env, model, book, blob, device, capture, lost_low, t0, token, pre):
    restore_capture(env, capture)
    motion_steps = 0
    how = None
    if int(token) != HANDOFF:
        _obs, _window_obs, _window_prev = _windows(env, device)
        _residual, action = decode_action(book, int(token), pre)
        from experiments.ball_impact.privileged_ball_impact_search import step_legal
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = executed
        motion_steps += 1
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            outcome = {
                "reason": "DROP_DURING_RECOVERY",
                "success": False,
                "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "recovery_steps": motion_steps,
                "nominal_elapsed": 0.0,
                "lift_completed": False,
                "q_nominal": 0.0,
                "reward": None,
            }
            return _score(outcome, LIMIT)
    else:
        how = "early"
    if how is None:
        for _ in range(CAP):
            if float(env.sim.data.time) - t0 >= LIMIT - 1e-9:
                how = "forced"
                break
            _obs, window_obs, window_prev = _windows(env, device)
            decision = _forward(model, window_obs, window_prev)
            if decision["token"] == HANDOFF:
                how = "early"
                break
            _residual, action = decode_action(book, decision["token"], decision["mean"])
            from experiments.ball_impact.privileged_ball_impact_search import step_legal
            executed = np.asarray(step_legal(env.sim, action), np.float32)
            env._prev = executed
            motion_steps += 1
            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
            if failed:
                outcome = {
                    "reason": "DROP_DURING_RECOVERY",
                    "success": False,
                    "handoff": None,
                    "recovery_elapsed": float(env.sim.data.time) - t0,
                    "recovery_steps": motion_steps,
                    "nominal_elapsed": 0.0,
                    "lift_completed": False,
                    "q_nominal": 0.0,
                    "reward": None,
                }
                return _score(outcome, LIMIT)
        if how is None:
            how = "forced"
    outcome = finish_recovery(env, blob["ref"], t0, how)
    outcome["recovery_steps"] = motion_steps
    return _score(outcome, LIMIT)


def _contact_change(rows):
    if len(rows) < 2:
        return None
    previous = (rows[0].get("nL_after", rows[0]["nL"]) > 0, rows[0].get("nR_after", rows[0]["nR"]) > 0)
    for row in rows[1:]:
        if "nL_after" not in row:
            continue
        current = (row["nL_after"] > 0, row["nR_after"] > 0)
        if current != previous:
            return int(row["step"])
        previous = current
    return None


def _divergence(left, right):
    count = min(len(left), len(right))
    for index in range(count):
        if int(left[index]["token"]) != int(right[index]["token"]):
            return index, "token"
        if float(np.linalg.norm(left[index]["obj"] - right[index]["obj"])) > 0.005:
            return index, "object_position"
        if float(np.max(np.abs(left[index]["mean"] - right[index]["mean"]))) > 0.02:
            return index, "residual_mean"
    if len(left) != len(right):
        return max(count - 1, 0), "length"
    return None, None


def probes_for(rows, divergence_step):
    chosen = {}

    def add(step, label):
        if step is None or step < 0 or step >= len(rows):
            return
        chosen.setdefault(int(step), [])
        if label not in chosen[int(step)]:
            chosen[int(step)].append(label)

    add(0, "initial")
    add(divergence_step, "first_divergence")
    add(_contact_change(rows), "contact_change")
    if rows and rows[-1]["token"] == HANDOFF:
        add(rows[-1]["step"], "early_handoff")
    if rows:
        add(rows[-1]["step"], "before_outcome")
    return chosen


def candidates_for(row):
    found = []

    def add(name, token, pre):
        pre = np.zeros(7, np.float32) if int(token) == HANDOFF else np.asarray(pre, np.float32)
        key = (int(token), tuple(np.round(pre, 5)))
        if any(item["key"] == key for item in found):
            return
        found.append({"name": name, "token": int(token), "pre": pre, "key": key})

    add("original", row["token"], row["mean"])
    if int(row["other_token"]) != int(row["token"]) or float(np.max(np.abs(row["other_mean"] - row["mean"]))) > 1e-4:
        add("other_checkpoint_argmax", row["other_token"], row["other_mean"])
    for token, _prob in row["top"]:
        if int(token) == int(row["token"]) or int(token) == HANDOFF:
            continue
        add("behavior_next_motion", token, row["mean"])
        break
    if int(row["token"]) != HANDOFF:
        add("original_token_zero_residual", row["token"], np.zeros(7, np.float32))
        if float(np.max(np.abs(row["other_mean"] - row["mean"]))) > 1e-3:
            add("original_token_other_residual", row["token"], row["other_mean"])
    add("immediate_handoff", HANDOFF, np.zeros(7, np.float32))
    return found


def compact_row(row):
    item = {key: value for key, value in row.items() if key not in {"mean", "other_mean", "qpos_after", "probs"}}
    item["mean_abs"] = float(np.mean(np.abs(row["mean"])))
    item["other_mean_abs"] = float(np.mean(np.abs(row["other_mean"])))
    item["obj"] = [float(value) for value in row["obj"]]
    if "residual" in row:
        item["residual"] = [float(value) for value in row["residual"]]
    return _jsonable(item)


def regression_ids(prior):
    by = {name: {row["id"]: row for row in rows} for name, rows in prior.items()}
    selected = []
    for row in prior["u174"]:
        if row["split"] != "train":
            continue
        if row["success"] and not by["u783"][row["id"]]["success"]:
            selected.append(row["id"])
    return selected


def write_reports(baseline, branches, prior_ids, stopped):
    matched = [row for row in baseline if row["match"]]
    mismatched = [row for row in baseline if not row["match"]]
    lines = [
        "# 基线复现",
        "",
        "命令：`python -u training/diagnose_k48_counterfactual_credit.py`",
        "",
        "每个 TRAIN 退化 IC 用 CPU 确定性 argmax 重新执行 u174 和 u783，交接后使用原来的名义控制器直到 12 秒终点。对照文件是 `results/evaluations/raw/grpo_v2_u174_u783/closed_loop.json`。",
        "",
        f"先前报告中的 TRAIN 退化 IC 共 {len(prior_ids)} 个。本次重新执行 {len(baseline)} 条策略轨迹。",
        "",
        f"与已有评估的 `reason` 和 `recovery_steps` 一致：{len(matched)}。不一致：{len(mismatched)}。",
        "",
    ]
    if stopped:
        lines.append("存在不一致或状态往返误差，反事实续跑已停止。")
        lines.append("")
    lines.append("| IC | policy | 本次 reason | steps | handoff | 已有 reason | 已有 steps | qpos 往返最大误差 |")
    lines.append("| --- | --- | --- | ---: | --- | --- | ---: | ---: |")
    for row in baseline:
        lines.append(
            f"| {row['id']} | {row['policy']} | {row['reason']} | {row['recovery_steps']} | {row['handoff']} | "
            f"{row['prior_reason']} | {row['prior_steps']} | {row['roundtrip']} |"
        )
    lines.append("")
    lines.append("往返误差是：恢复该步决策前的 capture，执行记录中的同一个 motion token 和 residual，再比较 `qpos`。`None` 表示该检查点的第一步是 HANDOFF，没有物理步。")
    _write(OUT / "BASELINE_REPRODUCTION.md", "\n".join(lines) + "\n")

    by_ic = {}
    for row in branches:
        by_ic.setdefault(row["id"], []).append(row)
    summary = [
        "# 反事实续跑",
        "",
        "每个分支都从 `take_capture` 恢复 MuJoCo、FSM、switch、观测历史和上一条命令，并带上当时的 `lost_low`。然后只替换当前这一步的合法 K48 动作，再用固定的 continuation policy 执行到 12 秒终点或恢复期掉落。",
        "",
        "仿真是确定性的。每个 `(历史, 动作, continuation)` 只跑一次。这里的均值不是多个随机样本的平均，随机性来源数为 0。不同动作之间的回报差来自动作本身，不来自重复采样。",
        "",
        "Continuation 只取 u174 或 u783 之一，两个条件分开记录，不合成一个优势。",
        "",
        "候选动作只来自当前窗口的策略输出：原动作、另一 checkpoint 的 argmax、当前策略下一个 motion token、原 token 的零 residual、原 token 配另一 checkpoint 的 residual 均值、立即 HANDOFF。没有使用轨迹后半段的结果去构造动作。",
        "",
        f"分支记录 {len(branches)} 条。",
        "",
    ]
    for episode_id, rows in by_ic.items():
        summary.append(f"## {episode_id}")
        summary.append("")
        summary.append("| 轨迹 | 步 | t | 标签 | 动作 | token | continuation | reason | SUCCESS_12S | return v2 | handoff | recovery_s |")
        summary.append("| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | ---: | --- | ---: |")
        for row in rows:
            summary.append(
                f"| {row['history']} | {row['step']} | {row['t']:.3f} | {','.join(row['labels'])} | {row['action']} | "
                f"{row['token']} | {row['continuation']} | {row['reason']} | {row['success']} | {row['return_v2']} | "
                f"{row['handoff']} | {None if row['recovery_elapsed'] is None else round(float(row['recovery_elapsed']), 3)} |"
            )
        summary.append("")
    _write(OUT / "COUNTERFACTUAL_RESULTS.md", "\n".join(summary) + "\n")
    write_conclusions(baseline, branches, stopped)


def write_conclusions(baseline, branches, stopped):
    u174 = [row for row in baseline if row["policy"] == "u174"]
    preference_steps = 0
    preference_changed = 0
    handoff_rank_rise = 0
    for row in u174:
        for step in row["steps"]:
            preference_steps += 1
            if int(step["other_token"]) != int(step["token"]):
                preference_changed += 1
            if int(step["other_rank47"]) < int(step["rank47"]):
                handoff_rank_rise += 1
    recoverable = []
    handoff_only = []
    motion_also = []
    for row in branches:
        if row["history"] != "u783" or row["continuation"] != "u783":
            continue
        siblings = [
            item for item in branches
            if item["id"] == row["id"] and item["history"] == "u783" and item["step"] == row["step"] and item["continuation"] == "u783"
        ]
        original = next((item for item in siblings if item["action"] == "original"), None)
        if original is None or original["success"]:
            continue
        successes = [item for item in siblings if item["success"] and item["action"] != "original"]
        if not successes:
            continue
        record = {
            "id": row["id"],
            "step": row["step"],
            "t": row["t"],
            "labels": row["labels"],
            "original_reason": original["reason"],
            "success_actions": sorted({item["action"] for item in successes}),
        }
        if record not in recoverable:
            recoverable.append(record)
        if any(item["action"] != "immediate_handoff" for item in successes):
            motion_also.append(record)
        if any(item["action"] == "immediate_handoff" for item in successes) and original["token"] != HANDOFF:
            handoff_only.append(record)
    early = [row for row in baseline if row["policy"] == "u783" and row["handoff"] == "early"]
    motion_states = [record for record in recoverable if any(name != "immediate_handoff" for name in record["success_actions"])]
    handoff_states = [record for record in recoverable if "immediate_handoff" in record["success_actions"]]
    recoverable_ids = sorted({record["id"] for record in recoverable})
    covered = sorted({row["id"] for row in baseline if row["policy"] == "u783"})
    uncovered = [episode_id for episode_id in covered if episode_id not in recoverable_ids]
    divergence = []
    seen = set()
    for row in branches:
        if row["history"] != "u174" or "first_divergence" not in row["labels"] or row["id"] in seen:
            continue
        seen.add(row["id"])
        divergence.append((row["id"], row["divergence"], row["step"], row["t"]))
    lines = [
        "# 结论",
        "",
        "这些结论只覆盖本次重新执行的 TRAIN 退化 IC。反事实成功不是 u783 的原始成功率。",
        "",
        "## 1. u783 在 u174 成功历史上是否改变了动作偏好",
        "",
        f"u174 轨迹上的决策步共 {preference_steps}。其中 u783 在同一窗口上的 argmax token 与 u174 不同的有 {preference_changed} 步。",
        f"u783 的 token 47 排名高于 u174 的有 {handoff_rank_rise} 步。排名更高指 rank 数字更小。",
        "",
        "这是同一因果历史上的前向比较，历史由 u174 的动作生成。它说明条件动作映射是否改变，不说明闭环状态分布是否改变。",
        "",
        "## 2. 提前 HANDOFF 和运动策略",
        "",
        f"u783 基线里 early HANDOFF 的轨迹有 {len(early)} 条，forced HANDOFF 的轨迹有 {sum(1 for row in baseline if row['policy'] == 'u783' and row['handoff'] == 'forced')} 条。",
        "",
        "闭环轨迹的第一次有意义分歧：",
        "",
    ]
    token_div = sum(1 for item in divergence if item[1] == "token")
    residual_div = sum(1 for item in divergence if item[1] == "residual_mean")
    lines.extend([
        f"- 对齐步上首先是 token 不同：{token_div} 个 IC。",
        f"- token 相同，但 residual 均值的最大绝对差超过 0.02：{residual_div} 个 IC。",
        "",
    ])
    for episode_id, kind, step, time_s in divergence:
        lines.append(f"- `{episode_id}`：{kind}，step {step}，t={time_s:.3f}s")
    handoff_recovered = []
    grouped = {}
    for row in branches:
        if row["history"] == "u783" and row["continuation"] == "u783" and "early_handoff" in row["labels"]:
            grouped.setdefault((row["id"], row["step"]), []).append(row)
    for key, rows in grouped.items():
        original = next(item for item in rows if item["action"] == "original")
        recovered = [
            item["action"] for item in rows
            if item["success"] and item["action"] != "original" and item["token"] != HANDOFF
        ]
        if recovered:
            handoff_recovered.append((key[0], key[1], rows[0]["t"], original["reason"], sorted(set(recovered))))
    lines.extend([
        "",
        "在 u783 实际选出 token 47 的状态上，原动作是 HANDOFF。下面列出把该步换成候选 motion、并且 continuation 仍是 u783 后达到 `SUCCESS_12S` 的状态。没有列出的 early HANDOFF 状态，候选 motion 也失败。",
        "",
    ])
    if not handoff_recovered:
        lines.append("没有这样的状态。")
    for episode_id, step, time_s, reason, actions in handoff_recovered:
        lines.append(f"- `{episode_id}` t={time_s:.3f}s step={step} original={reason} motion={actions}")
    lines.extend([
        "",
        "在 u783 自己的历史上、并且 continuation 也是 u783 时，原动作失败而某个更早或不同的合法动作成功的状态列在下一节。这不能单独证明原动作是唯一原因，因为没有穷尽动作空间。",
        "",
        "## 3. 失败轨迹里是否有可恢复前缀",
        "",
        f"满足“同一 u783 continuation 下原动作失败、至少一个合法替代成功”的状态有 {len(recoverable)} 个，分布在 {len(recoverable_ids)} 个 IC。",
        f"其中替代 motion 或 residual 成功的状态有 {len(motion_states)} 个；立即 HANDOFF 这一支成功的状态有 {len(handoff_states)} 个。同一状态可以同时计入两类。",
        "",
        "本次候选里没有出现这种状态的 IC：",
        "",
    ]
    )
    for episode_id in uncovered:
        lines.append(f"- `{episode_id}`")
    lines.extend([
        "",
    ])
    if not recoverable:
        lines.append("本次候选集合里没有这样的状态。")
        lines.append("")
    for record in recoverable:
        lines.append(
            f"- `{record['id']}` t={record['t']:.3f}s step={record['step']} labels={record['labels']} "
            f"original={record['original_reason']} alternatives={record['success_actions']}"
        )
    lines.extend([
        "",
        "## 4. 这些前缀有没有被序列级负优势统一惩罚",
        "",
        "更新规则会这样做：一条被采样轨迹只有一个 advantage，它乘在全部已记录步的平均对数概率上，early HANDOFF 步包含在内。见 `AUDIT.md`。",
        "",
        "当前实验不能证明这 18 条确定性失败前缀在 update 175–783 的随机采样里出现过，也不能给出它们当时的 advantage 符号。训练日志没有逐步 token 和逐步 advantage。因此“这些前缀被负优势惩罚”仍然是机制上可能、样本上未证实。",
        "",
        "## 5. 是否足够支持改用 step-dependent advantage",
        "",
        "不够。即便某些替代动作在固定 continuation 下成功，也只说明当前这组候选里存在回报差。候选不是策略的完整动作分布，仿真没有独立随机重复，也没有用新的 advantage 重新训练。不能把这个差当成应该更换更新规则的充分证据。",
        "",
        "## 6. 当前实验识别不了的部分",
        "",
        "- 训练时 `N(0, 0.05)` residual 噪声下的动作分布。本次续跑是确定性均值。",
        "- token 47 在训练更新中的概率轨迹。日志没有这一项。",
        "- 未纳入候选的其他 K48 token 或 residual 是否也会成功。",
        "- u174 continuation 下的成功不能解释 u783 的闭环失败原因；那个条件回答的是另一套后续策略。",
        "- VALID IC 没有参与本次动作比较，也不应拿来选择干预。",
        "",
    ])
    if stopped:
        lines.append("基线或状态往返未通过，上面的反事实条目只包含停止前已经写出的分支。")
        lines.append("")
    lines.append("计数按 `(IC, step)` 去重。同一个状态上的多条 continuation 或多种动作不重复计成多个状态。")
    lines.append("")
    _write(OUT / "CREDIT_ASSIGNMENT_CONCLUSIONS.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ics", type=int, default=0, help="0 runs every rechecked regression IC")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    write_audit()
    if args.reports_only:
        baseline = json.loads((OUT / "raw" / "baseline.json").read_text(encoding="utf-8"))
        branches = json.loads((OUT / "raw" / "branches.json").read_text(encoding="utf-8"))
        prior = json.loads((ROOT / "results/evaluations/raw/grpo_v2_u174_u783/closed_loop.json").read_text(encoding="utf-8"))
        write_reports(baseline, branches, regression_ids(prior), False)
        return
    device = torch.device("cpu")
    book = load_codebook(CODEBOOK)
    u174, book174 = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0174.pt", device)
    u783, book783 = load_trained_k48(ROOT / "results/grpo/v2/checkpoint_u0783.pt", device)
    u174.eval()
    u783.eval()
    if not np.array_equal(book, book174) or not np.array_equal(book, book783):
        raise RuntimeError("checkpoint codebook differs from the configured K47 book")
    prior = json.loads((ROOT / "results/evaluations/raw/grpo_v2_u174_u783/closed_loop.json").read_text(encoding="utf-8"))
    ids = regression_ids(prior)
    if args.max_ics:
        ids = ids[: args.max_ics]
    manifest = {row["id"]: row for row in _rows(ROOT / "data/grpo_initial_conditions.jsonl")}
    blobs = {}
    for episode_id in ids:
        with (ROOT / manifest[episode_id]["snapshot"]).open("rb") as handle:
            blobs[episode_id] = pickle.load(handle)
    env = make_training_env(device, model=u174)
    prior_by = {name: {row["id"]: row for row in rows} for name, rows in prior.items()}
    baseline = []
    branches = []
    stopped = False
    models = {"u174": u174, "u783": u783}
    for episode_id in ids:
        traces = {}
        captures = {}
        starts = {}
        for name, model, other in (("u174", u174, u783), ("u783", u783, u174)):
            outcome, rows, caps, t0 = rollout_pair(env, model, other, book, blobs[episode_id], device)
            scored = _score(outcome, LIMIT)
            motion = next((row for row in rows if row["token"] != HANDOFF), None)
            error = None if motion is None else roundtrip(env, book, motion, caps[motion["step"]])
            prior_row = prior_by[name][episode_id]
            match = prior_row["reason"] == scored["reason"] and int(prior_row["recovery_steps"]) == int(scored["recovery_steps"])
            if error is not None and error > 1e-5:
                match = False
            record = {
                "id": episode_id,
                "policy": name,
                **scored,
                "prior_reason": prior_row["reason"],
                "prior_steps": prior_row["recovery_steps"],
                "match": bool(match),
                "roundtrip": error,
                "steps": [compact_row(row) for row in rows],
            }
            baseline.append(record)
            traces[name] = rows
            captures[name] = caps
            starts[name] = t0
            print("BASE", episode_id, name, scored["reason"], scored["recovery_steps"], "match", match, "rt", error, flush=True)
            if not match:
                stopped = True
        if stopped:
            break
        div_step, div_kind = _divergence(traces["u174"], traces["u783"])
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        for name in ("u174", "u783"):
            chosen = probes_for(traces[name], div_step)
            for step, labels in chosen.items():
                row = traces[name][step]
                for candidate in candidates_for(row):
                    for continuation in ("u174", "u783"):
                        if "contact_change" in labels and continuation != name and "early_handoff" not in labels and "first_divergence" not in labels and "initial" not in labels and "before_outcome" not in labels:
                            continue
                        result = continue_from(
                            env, models[continuation], book, blobs[episode_id], device,
                            captures[name][step], int(row["lost_low"]), starts[name],
                            candidate["token"], candidate["pre"],
                        )
                        item = {
                            "id": episode_id,
                            "history": name,
                            "step": int(step),
                            "t": float(row["t"]),
                            "labels": labels,
                            "divergence": div_kind,
                            "action": candidate["name"],
                            "token": int(candidate["token"]),
                            "pre_abs_mean": float(np.mean(np.abs(candidate["pre"]))),
                            "continuation": continuation,
                            **result,
                        }
                        branches.append(item)
                        print("CF", episode_id, name, step, candidate["name"], continuation, result["reason"], flush=True)
        del captures
    (OUT / "raw" / "baseline.json").write_text(json.dumps(_jsonable(baseline), indent=2), encoding="utf-8")
    (OUT / "raw" / "branches.json").write_text(json.dumps(_jsonable(branches), indent=2), encoding="utf-8")
    meta = {"ids": ids, "stopped": stopped, "divergence_note": "token, then 5 mm object position, then 0.02 residual"}
    (OUT / "raw" / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_reports(baseline, branches, regression_ids(prior), stopped)
    print("DONE", "stopped" if stopped else "ok", "ics", len(ids), "branches", len(branches), flush=True)


if __name__ == "__main__":
    main()
