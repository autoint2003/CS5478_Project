"""Fair SAC / DPO / GRPO comparison from the immutable 39-D checkpoint.

Regime 1 trains all three methods on one shared branched rollout pool.
Regime 2 lets each method collect from its own policy under the same
rollout cap. Held-out episodes are rolled only after validation selection.
The observation, actor architecture, action map, geometry switch, gains,
and absolute t=12 rule are unchanged.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.autonomous_rl_sanity as sanity
import training.natural_loss_recatch_data_scaling as scale
import training.natural_loss_recatch_transformer_v3 as v3
from controllers.geometry_switch import GeometrySwitch, hand_pose
from controllers.recovery_actions import AIRBORNE_V2, normalize_action
from envs.physical_recovery import physical_pack
from training.autonomous_dataset import _until_impact_entry
from training.autonomous_episode import (
    Z_ABORT,
    fresh_nominal_state,
    make_training_env,
    nominal_substep,
    seat_base,
)
from training.autonomous_rl_sanity import (
    ACTOR_LR,
    BC_COEF,
    CRITIC_LR,
    GAMMA,
    POLYAK,
    RL_COEF,
    SIGMA,
    TERMINAL_RETAIN,
    TwinQ,
    assert_allowed,
    demo_rmse,
    drop_supports,
    eval_impact,
    load_demos,
    restore_supports,
    reward_detail,
    sample_impact,
    supports_of,
)
from training.balanced_unified_recovery import CKPT_39, EVAL_NATURAL, TRAIN_NATURAL
from training.full_3d_airborne_recatch_dataset import TRACK, restore_to, step_mapped
from training.history_aware_critic_ablation import restore_capture, take_capture
from training.natural_loss_recatch_transformer_v3 import load_airborne_policy
from training.recovery_runtime import T_TASK

RAW = ROOT / "results" / "diagnostics" / "raw" / "algorithm_selection"
OUT = RAW / "algorithm_selection.json"
G = 8
SMALL_GROUPS = 13
MEDIUM_GROUPS = 32
ONLINE_GROUPS = 13
PAIR_GAP = 0.5
ADV_STD_MIN = 0.02
BETA = 0.1
CLIP_EPS = 0.2
KL_COEF = 0.01
N_STEP = 10
RMSE_STOP = 0.15
DPO_EPOCHS = 4
GRPO_EPOCHS = 4
MAX_H = 160


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write(payload):
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _clone(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load(model, state):
    model.load_state_dict(state)
    model.eval()


def _discount(rewards):
    value = 0.0
    for reward in reversed(rewards):
        value = float(reward) + GAMMA * value
    return float(value)


def _sim_steps(sim, t0):
    dt = float(sim.model.opt.timestep)
    return int(round(max(0.0, float(sim.data.time) - t0) / max(dt, 1e-9)))


def _policy_step(env, stochastic):
    obs = np.asarray(env.observation(), np.float32)
    prev = np.asarray(env._prev, np.float32).copy()
    env._obs_seq.append(obs.copy())
    env._prev_seq.append(prev.copy())
    window_o = np.stack(env._obs_seq[-MAX_H:])
    window_p = np.stack(env._prev_seq[-MAX_H:])
    device = env.device
    with torch.no_grad():
        logits, res, mean_act = env.model(
            torch.tensor(window_o, device=device).unsqueeze(0),
            torch.tensor(window_p, device=device).unsqueeze(0),
        )
        logits = logits[0, -1]
        res_mean = res[0, -1]
        if stochastic:
            token = int(torch.distributions.Categorical(logits=logits).sample().item())
            res_sample = res_mean + SIGMA * torch.randn(7, device=device)
            act = torch.clamp(env.model.proto[token] + res_sample, -1.0, 1.0)
        else:
            token = int(torch.argmax(logits).item())
            res_sample = res_mean
            act = mean_act[0, -1]
        act_np = act.detach().float().cpu().numpy().astype(np.float32)
        res_np = res_sample.detach().float().cpu().numpy().astype(np.float32)
    v3.step_v3(env.sim, act_np)
    env._prev = act_np.copy()
    return {
        "obs": obs,
        "prev": prev,
        "act": act_np,
        "token": int(token),
        "res_sample": res_np,
    }


def _gauss(sample, mean):
    var = SIGMA ** 2
    quad = ((sample - mean) / SIGMA) ** 2
    return -0.5 * (quad + math.log(2.0 * math.pi * var)).sum(-1)


def sequence_logprob(model, steps, device):
    """Sum of token log-prob and fixed-variance residual log-prob."""
    if not steps:
        return torch.zeros((), device=device)
    if len(steps) <= MAX_H:
        obs = torch.tensor(np.stack([s["obs"] for s in steps]), device=device).unsqueeze(0)
        prev = torch.tensor(np.stack([s["prev"] for s in steps]), device=device).unsqueeze(0)
        logits, res, _act = model(obs, prev)
        tokens = torch.tensor([s["token"] for s in steps], device=device)
        sampled = torch.tensor(np.stack([s["res_sample"] for s in steps]), device=device)
        log_tok = torch.log_softmax(logits[0], dim=-1)[torch.arange(len(steps), device=device), tokens]
        log_res = _gauss(sampled, res[0])
        return (log_tok + log_res).sum()
    total = torch.zeros((), device=device)
    for start in range(0, len(steps), 8):
        windows = []
        tokens = []
        sampled = []
        lengths = []
        for t in range(start, min(len(steps), start + 8)):
            a = max(0, t + 1 - MAX_H)
            windows.append(steps[a:t + 1])
            tokens.append(steps[t]["token"])
            sampled.append(steps[t]["res_sample"])
            lengths.append(t + 1 - a)
        length = max(lengths)
        obs = torch.zeros(len(windows), length, steps[0]["obs"].shape[0], device=device)
        prev = torch.zeros(len(windows), length, 7, device=device)
        pad = torch.ones(len(windows), length, dtype=torch.bool, device=device)
        for i, window in enumerate(windows):
            n = len(window)
            obs[i, :n] = torch.tensor(np.stack([s["obs"] for s in window]), device=device)
            prev[i, :n] = torch.tensor(np.stack([s["prev"] for s in window]), device=device)
            pad[i, :n] = False
        logits, res, _act = model(obs, prev, pad)
        index = torch.tensor([n - 1 for n in lengths], device=device)
        row = torch.arange(len(windows), device=device)
        chosen = torch.tensor(tokens, device=device)
        log_tok = torch.log_softmax(logits[row, index], dim=-1)[row, chosen]
        mean = res[row, index]
        sample = torch.tensor(np.stack(sampled), device=device)
        total = total + (log_tok + _gauss(sample, mean)).sum()
    return total


def _finish_branch(env, steps, rewards, kind):
    if kind == "fall_abort":
        t12 = "DROP"
    else:
        t12, _t = env.finish_nominal(TRACK)
    bonus = TERMINAL_RETAIN if t12 == "RETAINED_TO_12" else 0.0
    if not rewards:
        rewards = [bonus]
    else:
        rewards[-1] = float(rewards[-1]) + bonus
    if steps:
        steps[-1]["rew"] = float(rewards[-1])
        steps[-1]["done"] = True
    return {
        "steps": steps,
        "ret": _discount(rewards),
        "t12": t12,
        "success": t12 == "RETAINED_TO_12",
        "exit": kind,
        "n_steps": len(steps),
    }


def run_recovery(env, stochastic):
    steps = []
    rewards = []
    kind = "time_cap"
    while float(env.sim.data.time) < T_TASK - 1e-9:
        row = _policy_step(env, stochastic)
        measured = env.switch.measure(env.sim)
        reward = float(reward_detail(measured)["shaping"])
        row["rew"] = reward
        row["done"] = False
        steps.append(row)
        rewards.append(reward)
        if measured["z"] < Z_ABORT:
            kind = "fall_abort"
            break
        if env.switch.update_exit(env.sim):
            kind = "geometry"
            break
        if len(steps) > 800:
            break
    return _finish_branch(env, steps, rewards, kind)


def _enter_impact(env, spec):
    assert_allowed(spec)
    _until_impact_entry(env, spec)
    t_start = 0.0
    if not env.switch.entered:
        t12, _t = env.finish_nominal(env.gains)
        return False, t12, t_start
    env.start_recovery()
    return True, None, t_start


def _enter_offset(env, base, spec):
    sim = env.sim
    info = restore_to(sim, base, env.nominal, env.pads, env.pad_mu)
    drop_supports(sim)
    t_start = float(sim.data.time)
    if not info.get("pose_ok", False):
        raise RuntimeError("offset restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(sim)
    scale.apply_spec(sim, spec)
    st = fresh_nominal_state()
    switch = env.switch
    while float(sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            nominal_substep(sim, TRACK, st)
        row = switch.measure(sim)
        switch.update_entry(sim)
        if row["z"] < Z_ABORT:
            break
    if not switch.entered:
        t12, _t = env.finish_nominal(TRACK)
        return False, t12, t_start
    env.start_recovery()
    return True, None, t_start


def _enter_air(env, base, q_ref, spec):
    assert_allowed(spec)
    sim = env.sim
    info = restore_to(sim, base, env.nominal, env.pads, env.pad_mu)
    drop_supports(sim)
    t_start = float(sim.data.time)
    if not info.get("pose_ok", False):
        raise RuntimeError("carry restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(sim)
    switch = env.switch
    t0 = float(sim.data.time)
    open_at = t0 + float(spec["duration_s"])
    opened = False
    open_hold = 0
    st = fresh_nominal_state()
    while float(sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            t = float(sim.data.time)
            if open_hold < 10 and t < open_at + 0.20:
                v_h = np.zeros(3)
                if t < open_at:
                    v_h[1] = float(spec["sign"]) * float(spec["speed"])
                    tau = -18.0
                else:
                    opened = True
                    v_h[2] = 0.3
                    tau = 2.0
                action = normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V2)
                step_mapped(sim, action, AIRBORNE_V2, TRACK, q_ref)
                pack = physical_pack(sim)
                if opened and int(pack["nL"]) == 0 and int(pack["nR"]) == 0:
                    open_hold += 1
            else:
                nominal_substep(sim, TRACK, st)
        if switch.update_entry(sim):
            break
    if not switch.entered:
        t12, _t = env.finish_nominal(TRACK)
        return False, t12, t_start
    env.start_recovery()
    return True, None, t_start


def _annotate_logp(env, ref_model, branch):
    device = env.device
    was = env.model.training
    env.model.eval()
    with torch.no_grad():
        branch["logp_behavior"] = float(sequence_logprob(env.model, branch["steps"], device))
        branch["logp_ref"] = float(sequence_logprob(ref_model, branch["steps"], device))
    env.model.train(was)


def collect_group(env, ref_model, kind, spec, base, q_ref, table):
    wall0 = time.perf_counter()
    if kind == "impact":
        restore_supports(env.sim, table)
        entered, t12, t_start = _enter_impact(env, spec)
    elif kind == "offset":
        entered, t12, t_start = _enter_offset(env, base, spec)
    else:
        entered, t12, t_start = _enter_air(env, base, q_ref, spec)
    if not entered:
        steps_sim = _sim_steps(env.sim, t_start)
        return {
            "kind": kind,
            "entered": False,
            "branches": [{
                "steps": [],
                "ret": TERMINAL_RETAIN if t12 == "RETAINED_TO_12" else 0.0,
                "t12": t12,
                "success": t12 == "RETAINED_TO_12",
                "exit": "no_entry",
                "n_steps": 0,
                "logp_behavior": 0.0,
                "logp_ref": 0.0,
            }],
            "sim_steps": steps_sim,
            "rollouts": 1,
            "wall_s": time.perf_counter() - wall0,
        }
    cap = take_capture(env)
    branches = []
    sim_steps = _sim_steps(env.sim, t_start)
    for _b in range(G):
        restore_capture(env, cap)
        b0 = float(env.sim.data.time)
        branch = run_recovery(env, True)
        _annotate_logp(env, ref_model, branch)
        branches.append(branch)
        sim_steps += _sim_steps(env.sim, b0)
    return {
        "kind": kind,
        "entered": True,
        "branches": branches,
        "sim_steps": sim_steps,
        "rollouts": G,
        "wall_s": time.perf_counter() - wall0,
    }


def _spec_for(rng, index):
    slot = index % 6
    if slot < 4:
        return "impact", sample_impact(rng)
    if slot == 4:
        return "offset", {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.0}
    return "air", {"kind": "air", "speed": 0.35, "sign": -1.0, "duration_s": 0.08}


def pool_stats(groups):
    branches = [b for g in groups for b in g["branches"]]
    entered = [g for g in groups if g["entered"]]
    kinds = []
    stds = []
    useful = 0
    for g in entered:
        succ = [b["success"] for b in g["branches"]]
        rets = [b["ret"] for b in g["branches"]]
        std = float(np.std(rets)) if rets else 0.0
        span = float(max(rets) - min(rets)) if rets else 0.0
        stds.append(std)
        mixed = bool(succ) and any(succ) and not all(succ)
        same_spread = (not mixed) and span >= PAIR_GAP
        if mixed or same_spread:
            useful += 1
        if mixed:
            kinds.append("mixed")
        elif same_spread:
            kinds.append("same_outcome_spread")
        elif succ and all(succ):
            kinds.append("all_retained")
        else:
            kinds.append("all_dropped")
    n_b = len([b for g in entered for b in g["branches"]])
    return {
        "groups": len(groups),
        "entered_groups": len(entered),
        "no_entry_groups": len(groups) - len(entered),
        "branch_rollouts": int(sum(g["rollouts"] for g in groups)),
        "sim_steps": int(sum(g["sim_steps"] for g in groups)),
        "successful_branches": int(sum(b["success"] for b in branches)),
        "failed_branches": int(sum(not b["success"] for b in branches)),
        "mixed_groups": kinds.count("mixed"),
        "all_retained_groups": kinds.count("all_retained"),
        "all_dropped_groups": kinds.count("all_dropped"),
        "same_outcome_spread_groups": kinds.count("same_outcome_spread"),
        "useful_std_groups": useful,
        "useful_std_fraction": (useful / len(entered)) if entered else 0.0,
        "mean_group_std": float(np.mean(stds)) if stds else 0.0,
        "policy_steps": int(sum(b["n_steps"] for b in branches)),
    }


def build_pairs(groups):
    pairs = []
    possible = 0
    success_success = 0
    for g in groups:
        if not g["entered"]:
            continue
        br = g["branches"]
        cands = []
        for i, left in enumerate(br):
            for right in br:
                gap = float(left["ret"] - right["ret"])
                if left["success"] and right["success"] and gap > 0.05 and left is not right:
                    success_success += 1
                if left["success"] and (not right["success"]) and gap >= PAIR_GAP:
                    possible += 1
                    cands.append((gap, left, right))
        cands.sort(key=lambda item: -item[0])
        for gap, plus, minus in cands[:4]:
            pairs.append({"plus": plus, "minus": minus, "gap": gap})
    return pairs, {"retained_vs_drop_available": possible, "retained_vs_drop_used": len(pairs), "success_vs_success_gaps": success_success}


def _advantages(branches):
    """Nonzero only for mixed outcomes or a same-outcome return span of at least PAIR_GAP.

    All-drop and all-retained groups whose returns differ only by dense shaping
    get advantage 0. Near-zero standard deviation is handled the same way.
    """
    rets = np.array([b["ret"] for b in branches], np.float64)
    std = float(rets.std()) if len(rets) else 0.0
    span = float(rets.max() - rets.min()) if len(rets) else 0.0
    mixed = len({bool(b["success"]) for b in branches}) > 1
    if std < 1e-8 or not (mixed or span >= PAIR_GAP):
        return np.zeros(len(branches), np.float64), std
    return (rets - float(rets.mean())) / std, std


def _shift(model, ref, demos, device):
    model.eval()
    ref.eval()
    tok = []
    res = []
    act = []
    sat = []
    with torch.no_grad():
        for demo in demos:
            logits, residual, pred = model(demo["obs"].unsqueeze(0), demo["prev"].unsqueeze(0))
            r_logits, r_res, r_pred = ref(demo["obs"].unsqueeze(0), demo["prev"].unsqueeze(0))
            tok.append(float((logits.argmax(-1) != r_logits.argmax(-1)).float().mean()))
            res.append(float((residual - r_res).norm(dim=-1).mean()))
            act.append(float((pred - r_pred).norm(dim=-1).mean()))
            sat.append(float((pred.abs() > 0.98).float().mean()))
    return {
        "token_disagree": float(np.mean(tok)),
        "residual_l2": float(np.mean(res)),
        "action_l2": float(np.mean(act)),
        "saturation": float(np.mean(sat)),
    }


def _specs():
    return {row["tag"]: row for row in list(TRAIN_NATURAL) + list(EVAL_NATURAL)}


def evaluate(env, ref_model, base, q_ref, table, demos, include_test):
    env.model.eval()
    rows = []
    restore_supports(env.sim, table)
    zm6 = eval_impact(env, 6.5, -0.006, 0.0)
    rows.append({"id": "impact_zm6", "split": "train_skill", "t12": zm6["t12"], "exit": zm6["exit_kind"]})
    print("EVAL", rows[-1], flush=True)
    val_impacts = [("impact_z3", 6.5, -0.003, 0.0)]
    test_impacts = [("impact_zp6", 6.5, 0.006, 0.0), ("impact_phi90", 6.5, 0.0, 90.0)]
    for name, speed, z, phi in val_impacts:
        restore_supports(env.sim, table)
        row = eval_impact(env, speed, z, phi)
        rows.append({"id": name, "split": "validation", "t12": row["t12"], "exit": row["exit_kind"]})
        print("EVAL", rows[-1], flush=True)
    airs = [("air_lat_pos", "validation", 0.30, 1.0), ("air_fast_pos", "validation", 0.45, 1.0)]
    if include_test:
        for name, speed, z, phi in test_impacts:
            restore_supports(env.sim, table)
            row = eval_impact(env, speed, z, phi)
            rows.append({"id": name, "split": "test", "t12": row["t12"], "exit": row["exit_kind"]})
            print("EVAL", rows[-1], flush=True)
        airs += [("air_lat_neg", "test", 0.30, -1.0), ("air_slow_neg", "test", 0.20, -1.0)]
    for name, split, speed, sign in airs:
        row = sanity._roll_air(env, base, q_ref, name, split, speed, sign, "select")
        rows.append({"id": name, "split": split, "t12": row["t12"], "exit": row["exit"]})
        print("EVAL", rows[-1], flush=True)
    specs = _specs()
    offsets = [("nl_dz14.6", "validation", "dz14.6"), ("nl_dvz_p", "validation", "dvz_p")]
    if include_test:
        offsets += [
            ("h_dz14.00", "test", "h_dz14.00"),
            ("h_dz14.25", "test", "h_dz14.25"),
            ("h_dy", "test", "h_dy"),
            ("h_dx", "test", "h_dx"),
        ]
    for name, split, key in offsets:
        row = sanity._roll_offset(env, base, name, split, specs[key], "select")
        rows.append({"id": name, "split": split, "t12": row["t12"], "exit": row["exit"]})
        print("EVAL", rows[-1], flush=True)
    rmse = demo_rmse(env.model, demos)
    shift = _shift(env.model, ref_model, demos, env.device)
    def score(split):
        chosen = [r for r in rows if r["split"] == split]
        return {
            "n": len(chosen),
            "success": int(sum(r["t12"] == "RETAINED_TO_12" for r in chosen)),
            "rows": chosen,
        }
    return {
        "zm6": rows[0],
        "validation": score("validation"),
        "test": score("test") if include_test else None,
        "rmse": rmse,
        "shift": shift,
        "skill_physical": rows[0]["t12"] == "RETAINED_TO_12",
    }


def _classify(row, baseline_row, rmse, shift):
    if row["t12"] == "RETAINED_TO_12":
        return "retained"
    if shift["action_l2"] < 0.02:
        return "failed to change the pretrained behavior"
    if baseline_row and baseline_row["t12"] == "RETAINED_TO_12" and max(rmse.values()) > RMSE_STOP:
        return "destroyed an old skill"
    if shift["token_disagree"] > 0.3:
        return "selected the wrong action token"
    if shift["residual_l2"] > 0.05:
        return "produced a bad residual"
    if row["exit"] in ("fall_abort", "DROP"):
        return "entered an unrecoverable trajectory"
    return "entered an unrecoverable trajectory"


def _sac_batch(episodes, rng, device, model, critic_target):
    flat = [(ep, t) for ep in episodes for t in range(len(ep["steps"]))]
    pick = [flat[int(i)] for i in rng.choice(len(flat), size=min(64, len(flat)), replace=False)]
    obs, prev, act, y = [], [], [], []
    boot_o, boot_p, scales = [], [], []
    for ep, t in pick:
        step = ep["steps"][t]
        a0 = max(0, t + 1 - MAX_H)
        hist = ep["steps"][a0:t + 1]
        obs.append(np.stack([s["obs"] for s in hist]))
        prev.append(np.stack([s["prev"] for s in hist]))
        act.append(step["act"])
        reward, scale, boot_t = _nstep(ep, t, N_STEP)
        y.append(reward)
        scales.append(scale)
        if boot_t is None:
            boot_o.append(hist)
            boot_p.append(hist)
        else:
            b0 = max(0, boot_t + 1 - MAX_H)
            boot = ep["steps"][b0:boot_t + 1]
            boot_o.append(boot)
            boot_p.append(boot)
    return pick, obs, prev, act, y, scales, boot_o


def _nstep(ep, t, n):
    reward = 0.0
    scale = 1.0
    steps = ep["steps"]
    for k in range(n):
        reward += scale * float(steps[t + k]["rew"])
        scale *= GAMMA
        if steps[t + k]["done"] or t + k + 1 >= len(steps):
            return reward, 0.0, None
    return reward, scale, t + n


def _windows_to_batch(windows, device):
    length = max(len(w) for w in windows)
    obs = torch.zeros(len(windows), length, windows[0][0]["obs"].shape[0], device=device)
    prev = torch.zeros(len(windows), length, 7, device=device)
    pad = torch.ones(len(windows), length, dtype=torch.bool, device=device)
    for i, window in enumerate(windows):
        n = len(window)
        obs[i, :n] = torch.tensor(np.stack([s["obs"] for s in window]), device=device)
        prev[i, :n] = torch.tensor(np.stack([s["prev"] for s in window]), device=device)
        pad[i, :n] = False
    index = torch.tensor([len(w) - 1 for w in windows], device=device)
    return obs, prev, pad, index


def train_sac(env, demos, episodes, updates, device, critic_state=None):
    model = env.model
    critic = TwinQ().to(device)
    if critic_state is not None:
        critic.load_state_dict(critic_state)
    target = TwinQ().to(device)
    target.load_state_dict(critic.state_dict())
    opt_c = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    opt_a = torch.optim.Adam(model.parameters(), lr=ACTOR_LR)
    rng = np.random.default_rng(0)
    best = _clone(model)
    best_rmse = demo_rmse(model, demos)
    last = None
    stopped = None
    model.train()
    for update in range(1, updates + 1):
        flat = [(ep, t) for ep in episodes for t in range(len(ep["steps"]))]
        if not flat:
            break
        choice = rng.choice(len(flat), size=min(64, len(flat)), replace=False)
        picks = [flat[int(i)] for i in choice]
        hists, acts, rewards, scales, boots = [], [], [], [], []
        for ep, t in picks:
            a0 = max(0, t + 1 - MAX_H)
            hists.append(ep["steps"][a0:t + 1])
            acts.append(ep["steps"][t]["act"])
            reward, scale, boot_t = _nstep(ep, t, N_STEP)
            rewards.append(reward)
            scales.append(scale)
            if boot_t is None:
                boots.append(ep["steps"][a0:t + 1])
            else:
                b0 = max(0, boot_t + 1 - MAX_H)
                boots.append(ep["steps"][b0:boot_t + 1])
        obs_t = torch.tensor(np.stack([h[-1]["obs"] for h in hists]), device=device)
        act_t = torch.tensor(np.stack(acts), device=device)
        y = torch.tensor(np.asarray(rewards, np.float32), device=device)
        scale_t = torch.tensor(np.asarray(scales, np.float32), device=device)
        with torch.no_grad():
            b_obs, b_prev, b_pad, b_index = _windows_to_batch(boots, device)
            b_act = model(b_obs, b_prev, b_pad)[2][torch.arange(len(boots), device=device), b_index]
            q1_t, q2_t = target(torch.tensor(np.stack([b[-1]["obs"] for b in boots]), device=device), b_act)
            y = y + scale_t * torch.minimum(q1_t, q2_t)
        q1, q2 = critic(obs_t, act_t)
        loss_c = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
        opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt_c.step()
        with torch.no_grad():
            for param, tparam in zip(critic.parameters(), target.parameters()):
                tparam.data.mul_(1.0 - POLYAK).add_(param.data, alpha=POLYAK)
        h_obs, h_prev, h_pad, h_index = _windows_to_batch(hists, device)
        pi = model(h_obs, h_prev, h_pad)[2][torch.arange(len(hists), device=device), h_index]
        q_pi, _q2 = critic(obs_t, pi)
        bc = 0.0
        for demo in demos:
            pred = model(demo["obs"].unsqueeze(0), demo["prev"].unsqueeze(0))[2][0]
            bc = bc + (pred - demo["act"]).pow(2).mean()
        bc = bc / len(demos)
        loss_a = BC_COEF * bc + RL_COEF * (-q_pi.mean())
        opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_a.step()
        td = float((q1.detach() - y.detach()).abs().mean())
        last = {
            "update": update,
            "critic": float(loss_c.detach()),
            "bc": float(bc.detach()),
            "rl": float((-q_pi.mean()).detach()),
            "q_pi": float(q_pi.mean().detach()),
            "q_abs_mean": float(q1.detach().abs().mean()),
            "target_abs_max": float(y.detach().abs().max()),
            "td_mean": td,
        }
        if update % 10 == 0 or update == updates:
            rmse = demo_rmse(model, demos)
            print("SAC", update, {k: round(v, 4) for k, v in rmse.items()}, "td", round(td, 4), flush=True)
            if max(rmse.values()) > RMSE_STOP:
                stopped = f"skill RMSE exceeded {RMSE_STOP} at update {update}"
                _load(model, best)
                break
            best = _clone(model)
            best_rmse = rmse
    q_ret, q_drop = [], []
    model.eval()
    critic.eval()
    with torch.no_grad():
        for ep in episodes:
            if not ep["steps"]:
                continue
            obs = torch.tensor(np.stack([s["obs"] for s in ep["steps"]]), device=device)
            act = torch.tensor(np.stack([s["act"] for s in ep["steps"]]), device=device)
            q1, _q2 = critic(obs, act)
            bucket = q_ret if ep["success"] else q_drop
            bucket.extend(float(x) for x in q1.detach().cpu())
    rank_hit = 0
    rank_n = 0
    with torch.no_grad():
        for group_eps in _group_entry_sets(episodes):
            if not group_eps:
                continue
            good = [ep for ep in group_eps if ep["success"]]
            bad = [ep for ep in group_eps if not ep["success"]]
            if not good or not bad:
                continue
            obs0 = good[0]["steps"][0]["obs"]
            for g in good:
                for b in bad:
                    qg, _ = critic(torch.tensor(obs0, device=device).unsqueeze(0), torch.tensor(g["steps"][0]["act"], device=device).unsqueeze(0))
                    qb, _ = critic(torch.tensor(obs0, device=device).unsqueeze(0), torch.tensor(b["steps"][0]["act"], device=device).unsqueeze(0))
                    rank_n += 1
                    rank_hit += int(float(qg) > float(qb))
    return {
        "updates": 0 if last is None else last["update"],
        "stopped": stopped,
        "last": last,
        "rmse_at_stop": best_rmse,
        "q_retained_mean": float(np.mean(q_ret)) if q_ret else None,
        "q_dropped_mean": float(np.mean(q_drop)) if q_drop else None,
        "entry_action_rank": None if rank_n == 0 else {"correct": rank_hit, "n": rank_n},
        "state": _clone(model),
        "critic_state": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
    }


def _group_entry_sets(episodes):
    # Episodes were appended per branch; grouping is recovered by identical first obs bytes.
    groups = []
    current = []
    key = None
    for ep in episodes:
        if not ep["steps"]:
            continue
        mark = ep["steps"][0]["obs"].tobytes()
        if key is None or mark == key:
            current.append(ep)
            key = mark
        else:
            groups.append(current)
            current = [ep]
            key = mark
    if current:
        groups.append(current)
    return groups


def train_dpo(env, ref_model, demos, pairs, device, epochs=DPO_EPOCHS):
    model = env.model
    opt = torch.optim.Adam(model.parameters(), lr=ACTOR_LR)
    best = _clone(model)
    last = None
    stopped = None
    margins = []
    correct = 0
    steps = 0
    model.train()
    if not pairs:
        return {"updates": 0, "stopped": "no retained-vs-drop pairs", "last": None, "preference_accuracy": None, "margin": None, "state": _clone(model), "kl": None}
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(epoch).permutation(len(pairs))
        losses = []
        for index in order:
            pair = pairs[int(index)]
            log_plus = sequence_logprob(model, pair["plus"]["steps"], device)
            log_minus = sequence_logprob(model, pair["minus"]["steps"], device)
            ref_plus = pair["plus"]["logp_ref"]
            ref_minus = pair["minus"]["logp_ref"]
            argument = BETA * ((log_plus - ref_plus) - (log_minus - ref_minus))
            loss = -torch.nn.functional.logsigmoid(argument)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            losses.append(float(loss.detach()))
            margins.append(float((log_plus - log_minus).detach()))
            correct += int(float(log_plus.detach()) > float(log_minus.detach()))
        rmse = demo_rmse(model, demos)
        print("DPO", epoch, {k: round(v, 4) for k, v in rmse.items()}, "loss", round(float(np.mean(losses)), 4), flush=True)
        last = {"epoch": epoch, "loss": float(np.mean(losses)), "rmse": rmse}
        if max(rmse.values()) > RMSE_STOP:
            stopped = f"skill RMSE exceeded {RMSE_STOP} at epoch {epoch}"
            _load(model, best)
            break
        best = _clone(model)
    kl = _mean_kl(model, ref_model, pairs, device)
    _load(model, best)
    return {
        "updates": steps if stopped is None else steps,
        "stopped": stopped,
        "last": last,
        "preference_accuracy": correct / max(len(margins), 1),
        "margin": float(np.mean(margins)) if margins else None,
        "kl_per_trajectory": kl,
        "beta": BETA,
        "state": _clone(model),
    }


def _mean_kl(model, ref_model, pairs, device):
    if not pairs:
        return None
    vals = []
    model.eval()
    with torch.no_grad():
        for pair in pairs[:8]:
            for branch in (pair["plus"], pair["minus"]):
                n = max(len(branch["steps"]), 1)
                logp = float(sequence_logprob(model, branch["steps"], device))
                vals.append((logp - branch["logp_ref"]) / n)
    return float(np.mean(vals)) if vals else None


def train_grpo(env, ref_model, demos, groups, device, epochs=GRPO_EPOCHS):
    model = env.model
    opt = torch.optim.Adam(model.parameters(), lr=ACTOR_LR)
    best = _clone(model)
    stopped = None
    updates = 0
    clip_hits = 0
    clip_n = 0
    adv_nz = 0
    adv_n = 0
    stds = []
    last = None
    usable = [g for g in groups if g["entered"] and g["branches"] and g["branches"][0]["steps"]]
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for g in usable:
            adv, std = _advantages(g["branches"])
            stds.append(std)
            adv_n += len(g["branches"])
            if float(np.max(np.abs(adv))) < 1e-8:
                continue
            logps = []
            for branch in g["branches"]:
                logps.append(sequence_logprob(model, branch["steps"], device))
            loss = torch.zeros((), device=device)
            for i, branch in enumerate(g["branches"]):
                if abs(float(adv[i])) < 1e-8:
                    continue
                adv_nz += 1
                logp = logps[i]
                ratio = torch.exp(torch.clamp(logp - branch["logp_behavior"], -20.0, 20.0))
                unclipped = ratio * float(adv[i])
                clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * float(adv[i])
                loss = loss + (-torch.minimum(unclipped, clipped))
                kl = logp - branch["logp_ref"]
                loss = loss + KL_COEF * kl
                clip_n += 1
                clip_hits += int(float(ratio.detach()) < 1.0 - CLIP_EPS or float(ratio.detach()) > 1.0 + CLIP_EPS)
            if clip_n == 0 and abs(float(adv.sum())) < 1e-8:
                continue
            if float(loss.detach()) == 0.0 and adv_nz == 0:
                continue
            opt.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                updates += 1
            losses.append(float(loss.detach()))
        rmse = demo_rmse(model, demos)
        print("GRPO", epoch, {k: round(v, 4) for k, v in rmse.items()}, "loss", None if not losses else round(float(np.mean(losses)), 4), flush=True)
        last = {"epoch": epoch, "loss": None if not losses else float(np.mean(losses)), "rmse": rmse}
        if max(rmse.values()) > RMSE_STOP:
            stopped = f"skill RMSE exceeded {RMSE_STOP} at epoch {epoch}"
            _load(model, best)
            break
        best = _clone(model)
    _load(model, best)
    return {
        "updates": updates,
        "stopped": stopped,
        "last": last,
        "clip_fraction": (clip_hits / clip_n) if clip_n else 0.0,
        "nonzero_advantage_fraction": (adv_nz / adv_n) if adv_n else 0.0,
        "mean_group_std": float(np.mean(stds)) if stds else 0.0,
        "kl_coef": KL_COEF,
        "state": _clone(model),
    }


def _episodes_from(groups):
    eps = []
    for g in groups:
        for branch in g["branches"]:
            if branch["steps"]:
                eps.append(branch)
    return eps


def _sac_updates_for(groups):
    n = sum(b["n_steps"] for g in groups for b in g["branches"])
    return int(max(20, min(80, n // 64)))


def _run_method(name, env, ref_state, ref_model, demos, groups, base, q_ref, table, device):
    _load(env.model, ref_state)
    wall0 = time.perf_counter()
    if name == "sac":
        trained = train_sac(env, demos, _episodes_from(groups), _sac_updates_for(groups), device)
        trained.pop("critic_state", None)
    elif name == "dpo":
        pairs, pair_info = build_pairs(groups)
        trained = train_dpo(env, ref_model, demos, pairs, device)
        trained["pairs"] = pair_info
    else:
        trained = train_grpo(env, ref_model, demos, groups, device)
        trained["pool"] = pool_stats(groups)
    _load(env.model, trained["state"])
    metrics = evaluate(env, ref_model, base, q_ref, table, demos, include_test=False)
    trained["wall_train_s"] = time.perf_counter() - wall0
    trained["eval"] = metrics
    trained["pool_rollouts"] = int(sum(g["rollouts"] for g in groups))
    trained["pool_steps"] = int(sum(g["sim_steps"] for g in groups))
    state = trained.pop("state")
    _load(env.model, ref_state)
    return trained, state


def _strip(trained):
    return {k: v for k, v in trained.items() if k != "state"}


def main():
    payload = {
        "ckpt": str(CKPT_39),
        "beta": BETA,
        "pair_gap": PAIR_GAP,
        "adv_std_min": ADV_STD_MIN,
        "clip": CLIP_EPS,
        "kl_coef": KL_COEF,
        "sigma": SIGMA,
        "n_step": N_STEP,
        "group_size": G,
        "bc_coef": BC_COEF,
        "rl_coef": RL_COEF,
        "actor_lr": ACTOR_LR,
        "critic": "instantaneous twin MLP, 10-step TD, history-conditioned actor backup",
    }
    stamp = CKPT_39.stat().st_mtime_ns
    torch.manual_seed(0)
    env = make_training_env()
    device = env.device
    ref_model, _proto, _saved = load_airborne_policy(CKPT_39, device)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()
    ref_state = _clone(env.model)
    demos = load_demos(device)
    table = supports_of(env.sim)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    rng = np.random.default_rng(21)
    print("BASELINE", flush=True)
    _load(env.model, ref_state)
    baseline = evaluate(env, ref_model, base, q_ref, table, demos, include_test=False)
    payload["baseline_validation"] = baseline
    _write(payload)

    groups = []
    for index in range(MEDIUM_GROUPS):
        kind, spec = _spec_for(rng, index)
        _load(env.model, ref_state)
        group = collect_group(env, ref_model, kind, spec, base, q_ref, table)
        groups.append(group)
        succ = sum(b["success"] for b in group["branches"])
        print(
            "GROUP", index, kind, "entered", group["entered"],
            "succ", succ, "/", len(group["branches"]),
            "steps", group["sim_steps"], "wall", round(group["wall_s"], 1),
            flush=True,
        )
        if index + 1 == SMALL_GROUPS:
            payload["small_pool"] = pool_stats(groups)
            _write(payload)
    payload["medium_pool"] = pool_stats(groups)
    small = groups[:SMALL_GROUPS]
    _write(payload)

    fixed = {}
    states = {}
    for label, subset in (("small", small), ("medium", groups)):
        fixed[label] = {}
        for name in ("sac", "dpo", "grpo"):
            print("TRAIN", label, name, flush=True)
            trained, state = _run_method(name, env, ref_state, ref_model, demos, subset, base, q_ref, table, device)
            fixed[label][name] = trained
            states[(label, name)] = state
            payload["fixed"] = {k: {n: _strip(v) for n, v in block.items()} for k, block in fixed.items()}
            _write(payload)

    online = {}
    online_rng = np.random.default_rng(22)
    for name in ("sac", "dpo", "grpo"):
        print("ONLINE", name, flush=True)
        _load(env.model, ref_state)
        own = []
        replay_eps = []
        pairs = []
        critic_state = None
        wall0 = time.perf_counter()
        stopped = None
        last_trained = None
        for index in range(ONLINE_GROUPS):
            kind, spec = _spec_for(online_rng, index)
            group = collect_group(env, ref_model, kind, spec, base, q_ref, table)
            own.append(group)
            if name == "sac":
                replay_eps.extend(_episodes_from([group]))
                if replay_eps:
                    trained = train_sac(env, demos, replay_eps, updates=8, device=device, critic_state=critic_state)
                    critic_state = trained.pop("critic_state")
                    last_trained = {k: v for k, v in trained.items() if k not in ("state", "critic_state")}
                    if trained["stopped"]:
                        stopped = trained["stopped"]
                        _load(env.model, trained["state"])
                        break
                    _load(env.model, trained["state"])
            elif name == "dpo":
                new_pairs, _info = build_pairs([group])
                pairs.extend(new_pairs)
                if new_pairs:
                    trained = train_dpo(env, ref_model, demos, new_pairs, device, epochs=1)
                    last_trained = {k: v for k, v in trained.items() if k != "state"}
                    if trained["stopped"]:
                        stopped = trained["stopped"]
                        _load(env.model, trained["state"])
                        break
                    _load(env.model, trained["state"])
            else:
                trained = train_grpo(env, ref_model, demos, [group], device, epochs=1)
                last_trained = {k: v for k, v in trained.items() if k != "state"}
                if trained["stopped"]:
                    stopped = trained["stopped"]
                    _load(env.model, trained["state"])
                    break
                _load(env.model, trained["state"])
            rmse = demo_rmse(env.model, demos)
            print("ONLINE", name, index, {k: round(v, 4) for k, v in rmse.items()}, flush=True)
            if max(rmse.values()) > RMSE_STOP:
                stopped = f"skill RMSE exceeded {RMSE_STOP} after group {index}"
                break
        metrics = evaluate(env, ref_model, base, q_ref, table, demos, include_test=False)
        online[name] = {
            "stopped": stopped,
            "pool": pool_stats(own),
            "wall_train_s": time.perf_counter() - wall0,
            "eval": metrics,
            "pairs": build_pairs(own)[1],
            "last_update": last_trained,
        }
        states[("online", name)] = _clone(env.model)
        payload["online"] = online
        _write(payload)
        _load(env.model, ref_state)

    def val_key(block):
        ev = block["eval"]
        return (
            ev["validation"]["success"],
            1 if ev["skill_physical"] else 0,
            -float(sum(ev["rmse"].values())),
        )

    selected = {}
    for name in ("sac", "dpo", "grpo"):
        cands = []
        for label in ("small", "medium"):
            cands.append((val_key(fixed[label][name]), ("fixed", label), states[(label, name)], fixed[label][name]))
        cands.append((val_key(online[name]), ("online", "online"), states[("online", name)], online[name]))
        cands.sort(key=lambda item: item[0], reverse=True)
        selected[name] = {"regime": cands[0][1][0], "budget": cands[0][1][1], "val": cands[0][0][0]}
        states[("selected", name)] = cands[0][2]
    payload["selected_before_heldout"] = selected
    _write(payload)

    print("HELDOUT", flush=True)
    _load(env.model, ref_state)
    base_test = evaluate(env, ref_model, base, q_ref, table, demos, include_test=True)
    held = {"pretrained": base_test}
    for name in ("sac", "dpo", "grpo"):
        _load(env.model, states[("selected", name)])
        held[name] = evaluate(env, ref_model, base, q_ref, table, demos, include_test=True)
    payload["heldout"] = held
    payload["ckpt_mtime_unchanged"] = CKPT_39.stat().st_mtime_ns == stamp
    _write(payload)
    print("WROTE", OUT, "ckpt_same", payload["ckpt_mtime_unchanged"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
