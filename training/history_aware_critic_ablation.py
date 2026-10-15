"""History-aware critic ablation. The 40-update actor stays frozen.

The instantaneous critic is the existing twin MLP, Q(o_t, a_t).
The history critic is a separate causal Transformer, Q(h_t, a_t), with the
same observation and previous-command window the actor sees. It does not
share weights with the actor. Reward, geometry, the 39-D observation, and
recovery7d_airborne_v3 are unchanged. The held-out eight are not rolled.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.autonomous_rl_sanity as sanity
import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import OBS_DIM_AIR
from envs.observable_obs import ObservableObsState
from training.autonomous_dataset import IMPACT, SKILLS, _until_impact_entry
from training.autonomous_episode import Z_ABORT, make_training_env, seat_base
from training.autonomous_rl_sanity import (
    CRITIC_LR,
    GAMMA,
    POLYAK,
    SIGMA,
    TERMINAL_RETAIN,
    TwinQ,
    collect_air,
    collect_impact,
    collect_offset,
    drop_supports,
    restore_supports,
    reward_detail,
    sample_impact,
    supports_of,
)
from training.full_3d_airborne_recatch_dataset import TRACK
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import T_TASK

RAW = ROOT / "results" / "diagnostics" / "raw" / "autonomous_rl"
CKPT = RAW / "ckpt_u0040.pt"
DT = 0.02
BATCH = 64
UPDATES = 4000
PRINT_EVERY = 500
MAX_H = 160
SHORT_H = 8
N_STEP = 10


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
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _nparams(module):
    return int(sum(p.numel() for p in module.parameters()))


def _actor_clone(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _same_actor(model, saved):
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), saved[key]):
            return False
    return True


def _state_spec():
    names = (
        "mjSTATE_TIME",
        "mjSTATE_QPOS",
        "mjSTATE_QVEL",
        "mjSTATE_ACT",
        "mjSTATE_WARMSTART",
        "mjSTATE_CTRL",
        "mjSTATE_MOCAP_POS",
        "mjSTATE_MOCAP_QUAT",
        "mjSTATE_USERDATA",
        "mjSTATE_PLUGIN",
    )
    spec = 0
    for name in names:
        if hasattr(mujoco.mjtState, name):
            spec |= int(getattr(mujoco.mjtState, name))
    if spec == 0:
        spec = int(mujoco.mjtState.mjSTATE_PHYSICS)
    return spec


STATE_SPEC = None


def _physics_blob(sim):
    global STATE_SPEC
    if STATE_SPEC is None:
        STATE_SPEC = _state_spec()
    n = int(mujoco.mj_stateSize(sim.model, STATE_SPEC))
    state = np.empty(n, np.float64)
    mujoco.mj_getState(sim.model, sim.data, state, STATE_SPEC)
    blob = {
        "state": state,
        "qpos": np.array(sim.data.qpos, float).copy(),
        "qvel": np.array(sim.data.qvel, float).copy(),
        "ctrl": np.array(sim.data.ctrl, float).copy(),
        "time": float(sim.data.time),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, float).copy(),
        "captured": bool(sim.captured),
    }
    if int(sim.model.na):
        blob["act"] = np.array(sim.data.act, float).copy()
    if hasattr(sim, "guide_eq"):
        blob["eq_data"] = int(sim.data.eq_active[sim.guide_eq]) if hasattr(sim.data, "eq_active") else None
        blob["eq_model"] = int(sim.model.eq_active0[sim.guide_eq])
        blob["mocap_pos"] = np.array(sim.data.mocap_pos[sim.guide_mocap], float).copy()
        blob["mocap_quat"] = np.array(sim.data.mocap_quat[sim.guide_mocap], float).copy()
    return blob


def _fsm_blob(fsm):
    arrays = {}
    for name in ("p_des", "r_des", "v_des", "w_des", "v_cmd", "w_cmd", "grasp_xy", "grasp_offset"):
        value = getattr(fsm, name)
        arrays[name] = None if value is None else np.array(value, float).copy()
    return {
        "phase": str(fsm.phase),
        "t_phase": float(fsm.t_phase),
        "t_stable": float(fsm.t_stable),
        "t_held": float(fsm.t_held),
        "lift_started": bool(fsm.lift_started),
        "success": bool(fsm.success),
        "grasp_z": float(fsm.grasp_z),
        "fg_cmd": float(fsm.fg_cmd),
        "arrays": arrays,
    }


def _switch_blob(switch):
    return {
        "p0": None if switch.p0 is None else np.array(switch.p0, float).copy(),
        "r0": None if switch.r0 is None else np.array(switch.r0, float).copy(),
        "pos_run": int(switch.pos_run),
        "exit_run": int(switch.exit_run),
        "entered": bool(switch.entered),
        "exited": bool(switch.exited),
        "t_entry": None if switch.t_entry is None else float(switch.t_entry),
        "t_exit": None if switch.t_exit is None else float(switch.t_exit),
        "entry_reason": switch.entry_reason,
    }


def take_capture(env):
    from training.table_drop import export_state as export_table_drop

    hist = env._hist
    return {
        "physics": _physics_blob(env.sim),
        "fsm": _fsm_blob(env.sim.fsm),
        "switch": _switch_blob(env.switch),
        "supports": supports_of(env.sim),
        "hist_prev_e": None if hist is None else float(hist.prev_e),
        "hist_prev_valid": False if hist is None else bool(hist.prev_valid),
        "obs_seq": [np.asarray(x, np.float32).copy() for x in env._obs_seq],
        "prev_seq": [np.asarray(x, np.float32).copy() for x in env._prev_seq],
        "prev": np.asarray(env._prev, np.float32).copy(),
        "table_drop": export_table_drop(env.sim),
    }


def restore_capture(env, cap):
    global STATE_SPEC
    if STATE_SPEC is None:
        STATE_SPEC = _state_spec()
    sim = env.sim
    physics = cap["physics"]
    mujoco.mj_setState(sim.model, sim.data, physics["state"], STATE_SPEC)
    sim.data.qpos[:] = physics["qpos"]
    sim.data.qvel[:] = physics["qvel"]
    sim.data.ctrl[:] = physics["ctrl"]
    sim.data.time = physics["time"]
    sim.data.qacc_warmstart[:] = physics["qacc_warmstart"]
    if "act" in physics and int(sim.model.na):
        sim.data.act[:] = physics["act"]
    if "eq_model" in physics:
        sim.model.eq_active0[sim.guide_eq] = physics["eq_model"]
        if physics["eq_data"] is not None and hasattr(sim.data, "eq_active"):
            sim.data.eq_active[sim.guide_eq] = physics["eq_data"]
        sim.data.mocap_pos[sim.guide_mocap] = physics["mocap_pos"]
        sim.data.mocap_quat[sim.guide_mocap] = physics["mocap_quat"]
    restore_supports(sim, cap["supports"])
    fsm = sim.fsm
    blob = cap["fsm"]
    fsm.phase = blob["phase"]
    fsm.t_phase = blob["t_phase"]
    fsm.t_stable = blob["t_stable"]
    fsm.t_held = blob["t_held"]
    fsm.lift_started = blob["lift_started"]
    fsm.success = blob["success"]
    fsm.grasp_z = blob["grasp_z"]
    fsm.fg_cmd = blob["fg_cmd"]
    for name, value in blob["arrays"].items():
        setattr(fsm, name, None if value is None else np.array(value, float).copy())
    sw = env.switch
    sb = cap["switch"]
    sw.p0 = None if sb["p0"] is None else np.array(sb["p0"], float).copy()
    sw.r0 = None if sb["r0"] is None else np.array(sb["r0"], float).copy()
    sw.pos_run = sb["pos_run"]
    sw.exit_run = sb["exit_run"]
    sw.entered = sb["entered"]
    sw.exited = sb["exited"]
    sw.t_entry = sb["t_entry"]
    sw.t_exit = sb["t_exit"]
    sw.entry_reason = sb["entry_reason"]
    env._hist = ObservableObsState(v3.DT_POLICY)
    if cap["hist_prev_e"] is None or not cap["hist_prev_valid"]:
        env._hist.prev_e = float("nan")
        env._hist.prev_valid = False
    else:
        env._hist.prev_e = float(cap["hist_prev_e"])
        env._hist.prev_valid = True
    env._obs_seq = [x.copy() for x in cap["obs_seq"]]
    env._prev_seq = [x.copy() for x in cap["prev_seq"]]
    env._prev = cap["prev"].copy()
    sim.captured = physics["captured"]
    mujoco.mj_forward(sim.model, sim.data)
    from training.table_drop import restore_state
    restore_state(sim, cap.get("table_drop"))
    err = float(np.max(np.abs(np.array(sim.data.qpos, float) - physics["qpos"])))
    if err > 1e-6:
        raise RuntimeError(f"qpos restore error {err}")


def _discount(rewards):
    value = 0.0
    for reward in reversed(rewards):
        value = float(reward) + GAMMA * value
    return float(value)


def _after_action(env, use_exit):
    row = env.switch.measure(env.sim)
    reward = float(reward_detail(row)["shaping"])
    if row["z"] < Z_ABORT:
        return reward, True, "fall_abort"
    if use_exit and env.switch.update_exit(env.sim):
        return reward, True, "geometry"
    if float(env.sim.data.time) >= T_TASK - 1e-9:
        return reward, True, "time_cap"
    return reward, False, None


def _conclude(env, rewards, kind):
    if kind == "fall_abort":
        t12 = "DROP"
    else:
        t12, _t = env.finish_nominal(TRACK)
    bonus = TERMINAL_RETAIN if t12 == "RETAINED_TO_12" else 0.0
    if not rewards:
        rewards = [0.0]
    rewards[-1] = float(rewards[-1]) + bonus
    return {"t12": t12, "ret": _discount(rewards), "exit": kind, "steps": len(rewards)}


def roll_policy(env, cap, action):
    """One action, then the frozen mean actor until the physical outcome."""
    restore_capture(env, cap)
    v3.step_v3(env.sim, np.asarray(action, np.float32))
    env._prev = np.asarray(action, np.float32).copy()
    reward, stop, kind = _after_action(env, True)
    rewards = [reward]
    while not stop:
        env.recovery_step()
        reward, stop, kind = _after_action(env, True)
        rewards.append(reward)
        if len(rewards) > 800:
            kind = "time_cap" if kind is None else kind
            break
    if kind is None:
        kind = "time_cap"
    return _conclude(env, rewards, kind)


def roll_skill_suffix(env, cap, first, future):
    """One action, then the rest of the verified schedule, then nominal to t=12.

    Geometry exit is not applied during the schedule. That matches the
    demonstration, which plays the schedule open-loop and only then continues.
    """
    restore_capture(env, cap)
    rewards = []
    kind = None
    for act in [first, *future]:
        if float(env.sim.data.time) >= T_TASK - 1e-9:
            kind = "time_cap"
            break
        v3.step_v3(env.sim, np.asarray(act, np.float32))
        reward, stop, step_kind = _after_action(env, False)
        rewards.append(reward)
        if stop:
            kind = step_kind
            break
    if kind is None:
        kind = "schedule_done"
    return _conclude(env, rewards, kind)


def _alternatives(act):
    grip = np.asarray(act, np.float32).copy()
    grip[6] = np.float32(-1.0 if act[6] > -0.5 else 1.0)
    vertical = np.asarray(act, np.float32).copy()
    vertical[2] = np.float32(-1.0 if act[2] > -0.5 else 1.0)
    return (("grip", grip), ("vertical", vertical))


def _separated(a, b):
    retained = "RETAINED_TO_12"
    return (a["t12"] == retained) != (b["t12"] == retained)


class CaptureKeeper:
    def __init__(self, family):
        self.family = family
        self.rows = []

    def __call__(self, env, obs, prev, act):
        t = len(env._obs_seq) - 1
        self.rows.append({
            "family": self.family,
            "t": int(t),
            "recorded": np.asarray(act, np.float32).copy(),
            "capture": take_capture(env),
        })
        if t > 40:
            self.rows = [row for row in self.rows if row["t"] == 0 or row["t"] >= t - 5]

    def select(self, outcome, length, episode):
        if length <= 40:
            want = {0, length // 2, length - 1}
        else:
            want = {length - 1, max(0, length - 5)}
        chosen = []
        for row in self.rows:
            if row["t"] not in want:
                continue
            row["original_outcome"] = outcome
            row["episode_length"] = int(length)
            row["episode"] = int(episode)
            row["source"] = "replay_policy"
            chosen.append(row)
        return chosen


def _annotate(ep):
    discounted = 0.0
    for step in reversed(ep["data"]):
        discounted = float(step["rew"]) + GAMMA * discounted
        step["mc"] = float(discounted)
        step["t_to_go"] = (len(ep["data"]) - 1 - step["t"]) * DT
    ep["success"] = ep["t12"] == "RETAINED_TO_12"


def _remember(episodes, buf, start, row, family):
    data = []
    for local_t, step in enumerate(buf[start:]):
        step["family"] = family
        step["outcome"] = row["t12"]
        step["ep"] = len(episodes)
        step["t"] = local_t
        step["exit"] = row.get("exit")
        data.append(step)
    ep = {
        "i": len(episodes),
        "family": family,
        "t12": row["t12"],
        "exit": row.get("exit"),
        "entered": bool(row["entered"]),
        "split": row.get("split", "train"),
        "name": row.get("name", family),
        "data": data,
    }
    if data:
        _annotate(ep)
    episodes.append(ep)
    print(
        "COLLECT", ep["split"], ep["i"], family, row.get("name", ""),
        row["t12"], row.get("exit"), len(data), flush=True,
    )
    return ep


def _actor_spec(model):
    layer = model.enc.layers[0]
    return {
        "context_length": int(model.pos.num_embeddings),
        "observation_dim": int(OBS_DIM_AIR),
        "previous_action_dim": 7,
        "input_projection": [int(x) for x in model.in_proj.weight.shape],
        "positional_encoding": "learned embedding added to each frame, index 0..T-1 inside the window",
        "d_model": int(model.in_proj.out_features),
        "layers": len(model.enc.layers),
        "heads": int(getattr(layer.self_attn, "num_heads", 4)),
        "feedforward": int(layer.linear1.out_features),
        "tokenizer_actions": int(model.logits.out_features),
        "residual_head": [int(x) for x in model.res.weight.shape],
        "action_head": (
            "causal Transformer on concat(o_t, previous command); "
            "argmax prototype token plus 0.25*tanh residual; clamp to [-1, 1]"
        ),
        "window_rule": "recovery longer than 160 frames passes only the latest 160",
        "parameters": _nparams(model),
    }


def _critic_spec():
    net = TwinQ()
    linear = net.q1[0]
    return {
        "form": "Q(o_t, a_t)",
        "input": "concat(current 39-D observation, candidate 7-D action)",
        "input_dim": int(linear.in_features),
        "sees_current_observation": True,
        "sees_previous_command": False,
        "sees_history": False,
        "sees_transformer_context": False,
        "sees_candidate_action": True,
        "hidden": 128,
        "layers": "Linear-ReLU-Linear-ReLU-Linear, two separate heads",
        "parameters": _nparams(net),
    }


class HistoryQ(torch.nn.Module):
    """Causal Transformer critic. The candidate action is applied after the history latent."""

    def __init__(self):
        super().__init__()
        d = 64
        self.in_proj = torch.nn.Linear(OBS_DIM_AIR + 7, d)
        self.pos = torch.nn.Embedding(MAX_H, d)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128, dropout=0.0,
            batch_first=True, activation="relu",
        )
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(d + 7, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(d + 7, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, obs, prev, act, pad_mask=None):
        x = torch.cat([obs, prev], dim=-1)
        length = x.shape[1]
        if length > MAX_H:
            raise RuntimeError(f"history length {length} exceeds the {MAX_H}-frame window")
        h = self.in_proj(x) + self.pos(torch.arange(length, device=x.device))
        causal = torch.triu(torch.ones(length, length, device=x.device) * float("-inf"), diagonal=1)
        key_pad = None
        if pad_mask is not None:
            key_pad = torch.zeros(pad_mask.shape, device=x.device, dtype=h.dtype)
            key_pad = key_pad.masked_fill(pad_mask, float("-inf"))
        ctx = self.enc(h, mask=causal, src_key_padding_mask=key_pad)
        if pad_mask is None:
            last = ctx[:, -1]
        else:
            index = (~pad_mask).sum(dim=1) - 1
            last = ctx[torch.arange(ctx.shape[0], device=ctx.device), index]
        z = torch.cat([last, act], dim=-1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)


def _history_spec(module):
    return {
        "form": "Q(h_t, a_t)",
        "history": "latest min(t+1, 160) frames of (39-D observation, 7-D previous command)",
        "positional_encoding": "learned embedding, index 0 is the oldest frame in the window",
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "feedforward": 128,
        "candidate_action": "concatenated with the last valid latent, then a 64-unit head",
        "shared_with_actor": False,
        "heads_share_encoder": True,
        "parameters": _nparams(module),
    }


def _tail_dist(left, right, k, drop_last):
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    if drop_last:
        a = a[:-1]
        b = b[:-1]
    if len(a) == 0 or len(b) == 0:
        return None
    n = min(k, len(a), len(b))
    return float(np.mean(np.linalg.norm(a[-n:] - b[-n:], axis=1)))


def _alias_rows():
    rows = []
    for skill in SKILLS:
        path = sanity.DEMO / f"demo_{skill['name']}.npz"
        z = np.load(path)
        obs = np.asarray(z["obs"], np.float32)
        act = np.asarray(z["act"], np.float32)
        prev = np.asarray(z["prev"], np.float32)
        for t in range(len(obs)):
            rows.append({
                "skill": skill["name"],
                "pool": skill["pool"],
                "t": t,
                "length": len(obs),
                "obs": obs[t],
                "act": act[t],
                "prev": prev[t],
                "obs_seq": obs,
                "act_seq": act,
                "prev_seq": prev,
            })
    return rows


def _pair_record(left, right):
    obs_d = float(np.linalg.norm(left["obs"] - right["obs"]))
    act_d = float(np.linalg.norm(left["act"] - right["act"]))
    return {
        "skill_a": left["skill"],
        "t_a": int(left["t"]),
        "len_a": int(left["length"]),
        "skill_b": right["skill"],
        "t_b": int(right["t"]),
        "len_b": int(right["length"]),
        "obs_l2": obs_d,
        "action_l2": act_d,
        "prev_command_l2": float(np.linalg.norm(left["prev"] - right["prev"])),
        "past_obs_l2": _tail_dist(left["obs_seq"][: left["t"] + 1], right["obs_seq"][: right["t"] + 1], 8, True),
        "past_action_l2": _tail_dist(left["act_seq"][: left["t"]], right["act_seq"][: right["t"]], 8, False),
        "same_skill": left["skill"] == right["skill"],
    }


def aliasing_report():
    rows = _alias_rows()
    cross = []
    within = []
    for i, left in enumerate(rows):
        best_cross = None
        best_within = None
        for right in rows[i + 1:]:
            rec = _pair_record(left, right)
            if left["skill"] != right["skill"]:
                cross.append(rec)
                if best_cross is None or rec["obs_l2"] < best_cross["obs_l2"]:
                    best_cross = rec
            elif abs(left["t"] - right["t"]) >= 3:
                within.append(rec)
                if best_within is None or rec["obs_l2"] < best_within["obs_l2"]:
                    best_within = rec
        # nearest from earlier rows too
        for right in rows[:i]:
            rec = _pair_record(right, left)
            if left["skill"] != right["skill"]:
                if best_cross is None or rec["obs_l2"] < best_cross["obs_l2"]:
                    best_cross = rec
            elif abs(left["t"] - right["t"]) >= 3:
                if best_within is None or rec["obs_l2"] < best_within["obs_l2"]:
                    best_within = rec
    def pack(records, limit):
        ordered = sorted(records, key=lambda row: row["obs_l2"])[:limit]
        return ordered

    def nn_summary(records):
        if not records:
            return None
        obs = np.array([row["obs_l2"] for row in records])
        act = np.array([row["action_l2"] for row in records])
        past = np.array([row["past_obs_l2"] or 0.0 for row in records])
        return {
            "n": int(len(records)),
            "obs_l2_min": float(obs.min()),
            "obs_l2_median": float(np.median(obs)),
            "action_l2_at_min_obs": float(records[int(np.argmin(obs))]["action_l2"]),
            "action_l2_median": float(np.median(act)),
            "past_obs_l2_median": float(np.median(past)),
            "close_obs_1_and_action_0_5": int(np.sum((obs < 1.0) & (act > 0.5))),
            "close_obs_2_and_action_0_5": int(np.sum((obs < 2.0) & (act > 0.5))),
            "close_obs_0_5_and_action_0_5": int(np.sum((obs < 0.5) & (act > 0.5))),
        }

    # One nearest cross-skill neighbor per step, not every pair.
    nn_cross = []
    nn_within = []
    for left in rows:
        cross_best = None
        within_best = None
        for right in rows:
            if left is right:
                continue
            rec = _pair_record(left, right)
            if left["skill"] != right["skill"]:
                if cross_best is None or rec["obs_l2"] < cross_best["obs_l2"]:
                    cross_best = rec
            elif abs(left["t"] - right["t"]) >= 3:
                if within_best is None or rec["obs_l2"] < within_best["obs_l2"]:
                    within_best = rec
        if cross_best is not None:
            nn_cross.append(cross_best)
        if within_best is not None:
            nn_within.append(within_best)
    return {
        "n_steps": len(rows),
        "definition": {
            "obs_l2": "L2 of the current 39-D observation",
            "action_l2": "L2 of the verified action at that step",
            "past_obs_l2": "mean per-frame L2 of up to 8 preceding observations, aligned from the current step",
            "past_action_l2": "mean per-frame L2 of up to 8 actions executed before the current step",
            "within_excludes": "same-skill pairs closer than 3 steps",
        },
        "nearest_cross_skill": nn_summary(nn_cross),
        "nearest_within_skill": nn_summary(nn_within),
        "closest_cross_skill_pairs": pack(cross, 8),
        "closest_within_skill_pairs": pack(within, 6),
    }


def _window(ep, t):
    start = max(0, t + 1 - MAX_H)
    obs = np.stack([ep["data"][k]["obs"] for k in range(start, t + 1)])
    prev = np.stack([ep["data"][k]["prev"] for k in range(start, t + 1)])
    return obs, prev


def _sample_batch(episodes, rng):
    weights = np.array([len(ep["data"]) for ep in episodes], np.float64)
    weights /= weights.sum()
    pick = int(rng.choice(len(episodes), p=weights))
    ep = episodes[pick]
    t = int(rng.integers(0, len(ep["data"])))
    return ep["i"], t


def _make_batches(episodes, seed):
    rng = np.random.default_rng(seed)
    return [_sample_batch(episodes, rng) for _ in range(UPDATES * BATCH)]


def _collate(obs_list, prev_list, act, device):
    length = max(len(obs) for obs in obs_list)
    batch = len(obs_list)
    obs = np.zeros((batch, length, OBS_DIM_AIR), np.float32)
    prev = np.zeros((batch, length, 7), np.float32)
    pad = np.ones((batch, length), np.bool_)
    for i, (o, p) in enumerate(zip(obs_list, prev_list)):
        n = len(o)
        obs[i, :n] = o
        prev[i, :n] = p
        pad[i, :n] = False
    return (
        torch.tensor(obs, device=device),
        torch.tensor(prev, device=device),
        torch.tensor(np.asarray(act, np.float32), device=device),
        torch.tensor(pad, device=device),
    )


def _prepare_history(obs, prev, mode, rng):
    obs = np.asarray(obs, np.float32)[-MAX_H:]
    prev = np.asarray(prev, np.float32)[-MAX_H:]
    if mode == "current":
        return obs[-1:], prev[-1:]
    if mode == "short":
        return obs[-SHORT_H:], prev[-SHORT_H:]
    if mode == "shuffle":
        if len(obs) <= 2:
            return obs, prev
        order = np.arange(len(obs) - 1)
        rng.shuffle(order)
        out_o = obs.copy()
        out_p = prev.copy()
        out_o[:-1] = obs[order]
        out_p[:-1] = prev[order]
        return out_o, out_p
    return obs, prev


def _q_history(critic, obs_list, prev_list, act, device, mode):
    prepared_o = []
    prepared_p = []
    for i, (obs, prev) in enumerate(zip(obs_list, prev_list)):
        rng = np.random.default_rng(20_000 + i + (0 if mode != "shuffle" else 17))
        o, p = _prepare_history(obs, prev, mode, rng)
        prepared_o.append(o)
        prepared_p.append(p)
    obs_t, prev_t, act_t, pad = _collate(prepared_o, prepared_p, act, device)
    with torch.no_grad():
        q1, q2 = critic(obs_t, prev_t, act_t, pad)
        qmin = torch.minimum(q1, q2)
    return q1.detach().cpu().numpy(), qmin.detach().cpu().numpy()


def _q_mlp(critic, obs_list, act, device):
    last = np.stack([np.asarray(obs, np.float32)[-1] for obs in obs_list])
    with torch.no_grad():
        q1, q2 = critic(
            torch.tensor(last, device=device),
            torch.tensor(np.asarray(act, np.float32), device=device),
        )
        qmin = torch.minimum(q1, q2)
    return q1.detach().cpu().numpy(), qmin.detach().cpu().numpy()


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _predict_executed(critic, episodes, device, history):
    rows = [step for ep in episodes if ep["data"] for step in ep["data"]]
    q1_all = np.zeros(len(rows), np.float32)
    cursor = 0
    by_ep = {ep["i"]: ep for ep in episodes}
    for chunk in _chunks(rows, 64):
        obs, prev, act = [], [], []
        for step in chunk:
            o, p = _window(by_ep[step["ep"]], step["t"])
            obs.append(o)
            prev.append(p)
            act.append(step["act"])
        if history:
            q1, _qmin = _q_history(critic, obs, prev, act, device, "full")
        else:
            q1, _qmin = _q_mlp(critic, obs, act, device)
        q1_all[cursor:cursor + len(chunk)] = q1
        cursor += len(chunk)
    return rows, q1_all


def _summarize(values):
    if len(values) == 0:
        return None
    arr = np.asarray(values, np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _value_report(critic, episodes, device, history):
    rows, q1 = _predict_executed(critic, episodes, device, history)
    if not rows:
        return None
    mc = np.array([step["mc"] for step in rows], np.float64)
    success = np.array([step["outcome"] == "RETAINED_TO_12" for step in rows])
    terminal = np.array([bool(step["done"]) for step in rows])
    err = q1 - mc
    return {
        "retained": _summarize(q1[success]),
        "dropped": _summarize(q1[~success]),
        "terminal_retained": _summarize(q1[success & terminal]),
        "terminal_dropped": _summarize(q1[(~success) & terminal]),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "q_scale": _summarize(q1),
        "mc_retained": _summarize(mc[success]),
        "mc_dropped": _summarize(mc[~success]),
    }


def _rank_block(q_good, q_bad, pairs):
    margin = q_good - q_bad
    correct = margin > 1e-6

    def subset(mask):
        if not np.any(mask):
            return None
        return {
            "n": int(mask.sum()),
            "correct": int(np.sum(correct[mask])),
            "mean_margin": float(margin[mask].mean()),
            "median_margin": float(np.median(margin[mask])),
        }

    groups = {}
    for name in ("wrist_reseat", "gravity_inward", "recatch_open20", "impact", "offset", "air"):
        mask = np.array([pair["group"] == name for pair in pairs])
        block = subset(mask)
        if block is not None:
            groups[name] = block
    sources = {}
    for name in ("skill_suffix", "replay_policy"):
        mask = np.array([pair["source"] == name for pair in pairs])
        block = subset(mask)
        if block is not None:
            sources[name] = block
    long_h = np.array([pair["hist_len"] >= SHORT_H for pair in pairs])
    return {
        "pooled": subset(np.ones(len(pairs), np.bool_)),
        "groups": groups,
        "sources": sources,
        "history_at_least_8": subset(long_h),
    }


def _score_pairs(mlp, transformer, pairs, device):
    if not pairs:
        return {"n": 0}
    out = {"n": len(pairs), "variants": {}}
    specs = [("mlp", mlp, False, "full")]
    for mode in ("full", "current", "shuffle", "short"):
        specs.append((f"transformer_{mode}", transformer, True, mode))
    for name, critic, history, mode in specs:
        qg = np.zeros(len(pairs), np.float32)
        qb = np.zeros(len(pairs), np.float32)
        qg_min = np.zeros(len(pairs), np.float32)
        qb_min = np.zeros(len(pairs), np.float32)
        cursor = 0
        for chunk in _chunks(pairs, 32):
            obs = [pair["hist_obs"] for pair in chunk]
            prev = [pair["hist_prev"] for pair in chunk]
            good = [pair["good"] for pair in chunk]
            bad = [pair["bad"] for pair in chunk]
            if history:
                g1, gmin = _q_history(critic, obs, prev, good, device, mode)
                b1, bmin = _q_history(critic, obs, prev, bad, device, mode)
            else:
                g1, gmin = _q_mlp(critic, obs, good, device)
                b1, bmin = _q_mlp(critic, obs, bad, device)
            n = len(chunk)
            qg[cursor:cursor + n] = g1
            qb[cursor:cursor + n] = b1
            qg_min[cursor:cursor + n] = gmin
            qb_min[cursor:cursor + n] = bmin
            cursor += n
        block = _rank_block(qg, qb, pairs)
        block["qmin"] = _rank_block(qg_min, qb_min, pairs)["pooled"]
        out["variants"][name] = block
    return out


def _fit_mc(name, critic, episodes, batches, device, history):
    opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    by_ep = {ep["i"]: ep for ep in episodes}
    losses = []
    critic.train()
    for update in range(1, UPDATES + 1):
        chunk = batches[(update - 1) * BATCH:update * BATCH]
        obs_l, prev_l, act_l, target = [], [], [], []
        for ep_i, t in chunk:
            ep = by_ep[ep_i]
            step = ep["data"][t]
            obs, prev = _window(ep, t)
            obs_l.append(obs)
            prev_l.append(prev)
            act_l.append(step["act"])
            target.append(step["mc"])
        y = torch.tensor(np.asarray(target, np.float32), device=device)
        if history:
            obs_t, prev_t, act_t, pad = _collate(obs_l, prev_l, act_l, device)
            q1, q2 = critic(obs_t, prev_t, act_t, pad)
        else:
            last = np.stack([obs[-1] for obs in obs_l])
            q1, q2 = critic(
                torch.tensor(last, device=device),
                torch.tensor(np.asarray(act_l, np.float32), device=device),
            )
        loss = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()
        if update % PRINT_EVERY == 0 or update == UPDATES:
            losses.append({"update": update, "loss": float(loss.detach())})
            print(name, update, "loss", round(float(loss.detach()), 5), flush=True)
            if not math.isfinite(losses[-1]["loss"]):
                break
    critic.eval()
    return {"name": name, "loss_trace": losses, "final_loss": losses[-1]["loss"] if losses else None}


def _nstep(ep, t, n):
    reward = 0.0
    scale = 1.0
    data = ep["data"]
    for k in range(n):
        step = data[t + k]
        reward += scale * float(step["rew"])
        scale *= GAMMA
        if step["done"] or t + k + 1 >= len(data):
            return reward, 0.0, None
    return reward, scale, t + n


def _actor_on_windows(model, obs_list, prev_list, device):
    obs_t, prev_t, _act, pad = _collate(obs_list, prev_list, np.zeros((len(obs_list), 7), np.float32), device)
    with torch.no_grad():
        pred = model(obs_t, prev_t, pad)[2]
        index = (~pad).sum(dim=1) - 1
        act = pred[torch.arange(pred.shape[0], device=device), index]
    return act


def _fit_td(name, critic, episodes, batches, model, device, history):
    target = HistoryQ().to(device) if history else TwinQ().to(device)
    target.load_state_dict(critic.state_dict())
    target.eval()
    opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    by_ep = {ep["i"]: ep for ep in episodes}
    losses = []
    critic.train()
    for update in range(1, UPDATES + 1):
        chunk = batches[(update - 1) * BATCH:update * BATCH]
        obs_l, prev_l, act_l = [], [], []
        reward_l, scale_l = [], []
        boot_obs, boot_prev = [], []
        for ep_i, t in chunk:
            ep = by_ep[ep_i]
            step = ep["data"][t]
            obs, prev = _window(ep, t)
            obs_l.append(obs)
            prev_l.append(prev)
            act_l.append(step["act"])
            reward, scale, boot_t = _nstep(ep, t, N_STEP)
            reward_l.append(reward)
            scale_l.append(scale)
            if boot_t is None:
                boot_obs.append(obs[-1:])
                boot_prev.append(prev[-1:])
            else:
                b_obs, b_prev = _window(ep, boot_t)
                boot_obs.append(b_obs)
                boot_prev.append(b_prev)
        y = torch.tensor(np.asarray(reward_l, np.float32), device=device)
        scale_t = torch.tensor(np.asarray(scale_l, np.float32), device=device)
        with torch.no_grad():
            pi = _actor_on_windows(model, boot_obs, boot_prev, device)
            if history:
                b_obs_t, b_prev_t, pi_t, pad = _collate(boot_obs, boot_prev, pi.detach().cpu().numpy(), device)
                q1_t, q2_t = target(b_obs_t, b_prev_t, pi_t, pad)
            else:
                last = np.stack([obs[-1] for obs in boot_obs])
                q1_t, q2_t = target(
                    torch.tensor(last, device=device),
                    pi,
                )
            y = y + scale_t * torch.minimum(q1_t, q2_t)
        if history:
            obs_t, prev_t, act_t, pad = _collate(obs_l, prev_l, act_l, device)
            q1, q2 = critic(obs_t, prev_t, act_t, pad)
        else:
            last = np.stack([obs[-1] for obs in obs_l])
            q1, q2 = critic(
                torch.tensor(last, device=device),
                torch.tensor(np.asarray(act_l, np.float32), device=device),
            )
        loss = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for param, tparam in zip(critic.parameters(), target.parameters()):
                tparam.data.mul_(1.0 - POLYAK).add_(param.data, alpha=POLYAK)
        if update % PRINT_EVERY == 0 or update == UPDATES:
            losses.append({
                "update": update,
                "loss": float(loss.detach()),
                "target_abs_max": float(y.detach().abs().max()),
            })
            print(name, update, "loss", round(float(loss.detach()), 5), flush=True)
            if not math.isfinite(losses[-1]["loss"]):
                break
    critic.eval()
    return {"name": name, "n_step": N_STEP, "loss_trace": losses, "final_loss": losses[-1]["loss"] if losses else None}


def _phase_trace(critic_mlp, critic_tr, episodes, device, stride_limit=8):
    traces = []
    for ep in episodes:
        if not ep["data"]:
            continue
        data = ep["data"]
        if len(data) <= 12:
            picks = list(range(len(data)))
        else:
            stride = max(1, len(data) // stride_limit)
            picks = list(range(0, len(data), stride))
            if picks[-1] != len(data) - 1:
                picks.append(len(data) - 1)
        obs, prev, act = [], [], []
        for t in picks:
            o, p = _window(ep, t)
            obs.append(o)
            prev.append(p)
            act.append(data[t]["act"])
        q_m, _ = _q_mlp(critic_mlp, obs, act, device)
        q_t, _ = _q_history(critic_tr, obs, prev, act, device, "full")
        rows = []
        for i, t in enumerate(picks):
            step = data[t]
            rows.append({
                "t": int(t),
                "t_to_go_s": float(step["t_to_go"]),
                "reward": float(step["rew"]),
                "mc": float(step["mc"]),
                "q_mlp": float(q_m[i]),
                "q_transformer": float(q_t[i]),
                "grip": float(step["act"][6]),
                "vz": float(step["act"][2]),
            })
        traces.append({
            "name": ep.get("name"),
            "family": ep["family"],
            "t12": ep["t12"],
            "length": len(data),
            "rows": rows,
        })
    return traces


def _cross_values(mlp, transformer, device):
    rows = _alias_rows()
    by_skill = {}
    for row in rows:
        by_skill.setdefault(row["skill"], row)
    # Use the closest cross-skill pairs already defined by current observations.
    report = aliasing_report()
    examples = []
    for pair in report["closest_cross_skill_pairs"][:6]:
        left = next(row for row in rows if row["skill"] == pair["skill_a"] and row["t"] == pair["t_a"])
        right = next(row for row in rows if row["skill"] == pair["skill_b"] and row["t"] == pair["t_b"])
        h1 = left["obs_seq"][: left["t"] + 1][-MAX_H:]
        p1 = left["prev_seq"][: left["t"] + 1][-MAX_H:]
        h2 = right["obs_seq"][: right["t"] + 1][-MAX_H:]
        p2 = right["prev_seq"][: right["t"] + 1][-MAX_H:]
        action = left["act"]
        q_m, _ = _q_mlp(mlp, [h1, h2], [action, action], device)
        q_t, _ = _q_history(transformer, [h1, h2], [p1, p2], [action, action], device, "full")
        examples.append({
            **{k: pair[k] for k in ("skill_a", "t_a", "skill_b", "t_b", "obs_l2", "action_l2", "past_obs_l2", "past_action_l2")},
            "q_mlp_hist_a": float(q_m[0]),
            "q_mlp_hist_b": float(q_m[1]),
            "q_tr_hist_a": float(q_t[0]),
            "q_tr_hist_b": float(q_t[1]),
            "same_action": "action from the first state",
        })
    return examples


def _replay_skill(env, skill):
    spec = next(row for row in IMPACT if row["id"] == "impact_zm6")
    _until_impact_entry(env, spec)
    if not env.switch.entered:
        empty = {"name": skill["name"], "family": skill["pool"], "t12": "NO_ENTRY", "data": []}
        return {"name": skill["name"], "t12": "NO_ENTRY", "steps": 0, "obs_max_abs_vs_npz": None, "forks": 0}, [], empty
    actions = [v3.v1_normalized_to_v3(np.asarray(act, np.float32)) for act in schedules()[skill["schedule"]]]
    n = len(actions)
    stride = 3 if n <= 20 else 8
    picks = set(range(0, n, stride))
    picks.add(n - 1)
    env._hist = ObservableObsState(v3.DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    rewards = []
    jobs = []
    fell = False
    for t, act in enumerate(actions):
        if float(env.sim.data.time) >= T_TASK - 1e-9:
            break
        obs = env.observation()
        env._obs_seq.append(np.asarray(obs, np.float32))
        env._prev_seq.append(env._prev.copy())
        if t in picks:
            jobs.append({
                "family": skill["pool"],
                "group": skill["name"],
                "source": "skill_suffix",
                "t": int(t),
                "episode_length": n,
                "recorded": np.asarray(act, np.float32).copy(),
                "future": [np.asarray(a, np.float32).copy() for a in actions[t + 1:]],
                "capture": take_capture(env),
                "original_outcome": None,
            })
        v3.step_v3(env.sim, np.asarray(act, np.float32))
        env._prev = np.asarray(act, np.float32).copy()
        row = env.switch.measure(env.sim)
        rewards.append(float(reward_detail(row)["shaping"]))
        if row["z"] < Z_ABORT:
            fell = True
            break
    if fell:
        t12 = "DROP"
    else:
        t12, _t = env.finish_nominal(TRACK)
    bonus = TERMINAL_RETAIN if t12 == "RETAINED_TO_12" else 0.0
    if rewards:
        rewards[-1] += bonus
    mc = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + GAMMA * running
        mc.append(running)
    mc.reverse()
    ref = np.load(sanity.DEMO / f"demo_{skill['name']}.npz")
    got = np.stack(env._obs_seq) if env._obs_seq else np.zeros((0, OBS_DIM_AIR))
    ref_obs = np.asarray(ref["obs"], np.float32)
    if got.shape == ref_obs.shape:
        obs_err = float(np.max(np.abs(got - ref_obs)))
    else:
        obs_err = None
    trace = {
        "name": skill["name"],
        "family": skill["pool"],
        "t12": t12,
        "exit": "skill_schedule",
        "split": "train_skill",
        "data": [],
    }
    for t, reward in enumerate(rewards):
        trace["data"].append({
            "obs": got[t],
            "prev": env._prev_seq[t],
            "act": np.asarray(actions[t], np.float32),
            "rew": float(reward),
            "mc": float(mc[t]),
            "t": t,
            "done": t == len(rewards) - 1,
            "ep": -1,
            "outcome": t12,
            "t_to_go": (len(rewards) - 1 - t) * DT,
        })
    for job in jobs:
        job["original_outcome"] = t12
    print("SKILL", skill["name"], t12, "steps", len(rewards), "obs_err", obs_err, "forks", len(jobs), flush=True)
    return {
        "name": skill["name"],
        "t12": t12,
        "steps": len(rewards),
        "obs_max_abs_vs_npz": obs_err,
        "forks": len(jobs),
    }, jobs, trace


def _fork_jobs(env, jobs, kind):
    pairs = []
    tested = []
    for index, job in enumerate(jobs):
        if kind == "skill_suffix":
            recorded = roll_skill_suffix(env, job["capture"], job["recorded"], job["future"])
        else:
            recorded = roll_policy(env, job["capture"], job["recorded"])
        choice = None
        alt_out = None
        alt_name = None
        for name, alt in _alternatives(job["recorded"]):
            if float(np.linalg.norm(alt - job["recorded"])) < 0.25:
                continue
            if kind == "skill_suffix":
                rolled = roll_skill_suffix(env, job["capture"], alt, job["future"])
            else:
                rolled = roll_policy(env, job["capture"], alt)
            if _separated(recorded, rolled):
                choice = alt
                alt_out = rolled
                alt_name = name
                break
            alt_out = rolled
            alt_name = name
        cap = job["capture"]
        hist_obs = np.stack(cap["obs_seq"])
        hist_prev = np.stack(cap["prev_seq"])
        separated = choice is not None
        record = {
            "group": job.get("group", job["family"]),
            "source": job["source"],
            "t": int(job["t"]),
            "episode_length": int(job.get("episode_length", 0)),
            "hist_len": int(len(hist_obs)),
            "alt": alt_name,
            "separated": bool(separated),
            "recorded_t12": recorded["t12"],
            "alt_t12": None if alt_out is None else alt_out["t12"],
            "recorded_return": recorded["ret"],
            "alt_return": None if alt_out is None else alt_out["ret"],
            "original_outcome": job.get("original_outcome"),
        }
        tested.append(record)
        if separated:
            if recorded["t12"] == "RETAINED_TO_12":
                good, bad = job["recorded"], choice
                good_ret, bad_ret = recorded["ret"], alt_out["ret"]
            else:
                good, bad = choice, job["recorded"]
                good_ret, bad_ret = alt_out["ret"], recorded["ret"]
            pairs.append({
                "group": record["group"],
                "source": job["source"],
                "t": int(job["t"]),
                "hist_len": int(len(hist_obs)),
                "hist_obs": hist_obs,
                "hist_prev": hist_prev,
                "good": np.asarray(good, np.float32),
                "bad": np.asarray(bad, np.float32),
                "good_return": float(good_ret),
                "bad_return": float(bad_ret),
                "alt": alt_name,
            })
        job.pop("capture", None)
        job.pop("future", None)
        print(
            "FORK", kind, index, record["group"], "t", record["t"],
            record["recorded_t12"], record["alt_t12"], "sep", separated,
            flush=True,
        )
    return pairs, tested


def _composition(episodes):
    steps = [step for ep in episodes for step in ep["data"]]
    retained_eps = [ep for ep in episodes if ep["success"] and ep["data"]]
    dropped_eps = [ep for ep in episodes if (not ep["success"]) and ep["data"]]
    return {
        "episodes": len(episodes),
        "retained_episodes": len(retained_eps),
        "dropped_episodes": len(dropped_eps),
        "transitions": len(steps),
        "retained_transitions": sum(len(ep["data"]) for ep in retained_eps),
        "dropped_transitions": sum(len(ep["data"]) for ep in dropped_eps),
        "terminals": sum(1 for step in steps if step["done"]),
    }


def _accuracy(block):
    if not block or not block.get("pooled") or not block["pooled"]["n"]:
        return None
    return block["pooled"]["correct"] / block["pooled"]["n"]


def main():
    clock = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / "history_critic_ablation.json"
    alias = aliasing_report()
    print("ALIAS cross", alias["nearest_cross_skill"], flush=True)
    torch.manual_seed(0)
    env = make_training_env()
    device = env.device
    saved = torch.load(CKPT, map_location=device, weights_only=False)
    env.model.load_state_dict(saved["actor"])
    env.model.eval()
    for param in env.model.parameters():
        param.requires_grad_(False)
    actor_ref = _actor_clone(env.model)
    actor_info = _actor_spec(env.model)
    critic_info = _critic_spec()
    probe = HistoryQ()
    history_info = _history_spec(probe)
    del probe
    print("PARAMS", actor_info["parameters"], critic_info["parameters"], history_info["parameters"], flush=True)

    table = supports_of(env.sim)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    rng = np.random.default_rng(11)
    buf = []
    train = []
    jobs = []
    for _cycle in range(2):
        for _impact in range(4):
            restore_supports(env.sim, table)
            spec = sample_impact(rng)
            keeper = CaptureKeeper("impact")
            start = len(buf)
            row = collect_impact(env, spec, rng, buf, SIGMA, on_decision=keeper)
            ep = _remember(train, buf, start, row, "impact")
            if ep["data"]:
                jobs.extend(keeper.select(row["t12"], len(ep["data"]), ep["i"]))
        drop_supports(env.sim)
        keeper = CaptureKeeper("offset")
        start = len(buf)
        row = collect_offset(env, base, buf, SIGMA, rng, on_decision=keeper)
        ep = _remember(train, buf, start, row, "offset")
        if ep["data"]:
            jobs.extend(keeper.select(row["t12"], len(ep["data"]), ep["i"]))
        keeper = CaptureKeeper("air")
        start = len(buf)
        row = collect_air(env, base, q_ref, buf, rng, SIGMA, on_decision=keeper)
        ep = _remember(train, buf, start, row, "air")
        if ep["data"]:
            jobs.extend(keeper.select(row["t12"], len(ep["data"]), ep["i"]))

    terminals = [job for job in jobs if job["t"] == job["episode_length"] - 1 and job["episode_length"] <= 40]
    if not terminals:
        terminals = [job for job in jobs if job["t"] == job["episode_length"] - 1][:1]
    repro = []
    for job in terminals[:2]:
        rolled = roll_policy(env, job["capture"], job["recorded"])
        match = rolled["t12"] == job["original_outcome"]
        repro.append({
            "family": job["family"],
            "t": job["t"],
            "length": job["episode_length"],
            "original": job["original_outcome"],
            "reproduced": rolled["t12"],
            "match": bool(match),
        })
        print("REPRO", repro[-1], flush=True)
        if not match:
            _write(out_path, {"reproduction": repro, "alias": alias, "failed": "restore"})
            raise RuntimeError(f"terminal action was not reproduced: {repro[-1]}")

    val = []
    val_rng = np.random.default_rng(19)
    restore_supports(env.sim, table)
    start = len(buf)
    row = collect_impact(
        env,
        {"kind": "impact", "v_mps": 6.5, "z_off_m": -0.003, "phi_deg": 0.0},
        val_rng, buf, 0.0, check=False,
    )
    row["name"] = "impact_z3"
    row["split"] = "validation"
    _remember(val, buf, start, row, "impact")
    for name, speed, sign in (("air_lat_pos", 0.30, 1.0), ("air_fast_pos", 0.45, 1.0)):
        start = len(buf)
        row = collect_air(env, base, q_ref, buf, val_rng, 0.0, speed=speed, sign=sign, check=False)
        row["name"] = name
        row["split"] = "validation"
        _remember(val, buf, start, row, "air")
    drop_supports(env.sim)
    for name, spec in (
        ("nl_dz14.6", {"kind": "offset", "dz": -14.6, "dx": 0.0, "dy": 0.0, "dvz": 0.0}),
        ("nl_dvz_p", {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.005}),
    ):
        start = len(buf)
        row = collect_offset(env, base, buf, 0.0, val_rng, spec=spec)
        row["name"] = name
        row["split"] = "validation"
        _remember(val, buf, start, row, "offset")

    skill_meta = []
    skill_jobs = []
    skill_eps = []
    for skill in SKILLS:
        restore_supports(env.sim, table)
        meta, skill_job, trace = _replay_skill(env, skill)
        skill_meta.append(meta)
        skill_jobs.extend(skill_job)
        if trace["data"] and trace["t12"] != "RETAINED_TO_12":
            print("SKILL_NOT_RETAINED", skill["name"], trace["t12"], flush=True)
        skill_eps.append(trace)
    # Confirm a restored t=0 skill suffix still retains before spending the rest of the forks.
    if not any(job["t"] == 0 for job in skill_jobs):
        raise RuntimeError(f"skill replay produced no t=0 fork: {skill_meta}")
    skill_anchor = next(job for job in skill_jobs if job["t"] == 0)
    t0_roll = roll_skill_suffix(env, skill_anchor["capture"], skill_anchor["recorded"], skill_anchor["future"])
    print("SKILL_T0_RESTORE", skill_anchor["group"], t0_roll["t12"], flush=True)
    if t0_roll["t12"] != "RETAINED_TO_12":
        _write(out_path, {"skill_t0": t0_roll, "skill_meta": skill_meta, "failed": "skill_restore"})
        raise RuntimeError("restored skill suffix did not retain")

    print("FORK_COUNTS", "replay", len(jobs), "skill", len(skill_jobs), flush=True)
    skill_pairs, skill_tested = _fork_jobs(env, skill_jobs, "skill_suffix")
    replay_pairs, replay_tested = _fork_jobs(env, jobs, "replay_policy")
    pairs = skill_pairs + replay_pairs
    print("PAIRS", len(pairs), "tested", len(skill_tested) + len(replay_tested), flush=True)
    if not _same_actor(env.model, actor_ref):
        raise RuntimeError("actor changed during rollouts")

    train_only = [ep for ep in train if ep["data"]]
    batches = []
    batch_rng = np.random.default_rng(0)
    flat = []
    for _ in range(UPDATES * BATCH):
        flat.append(_sample_batch(train_only, batch_rng))
    batches = flat

    torch.manual_seed(0)
    mlp = TwinQ().to(device)
    mc_mlp = _fit_mc("mc_mlp", mlp, train_only, batches, device, False)
    torch.manual_seed(0)
    transformer = HistoryQ().to(device)
    mc_tr = _fit_mc("mc_transformer", transformer, train_only, batches, device, True)
    if not _same_actor(env.model, actor_ref):
        raise RuntimeError("actor changed during Monte Carlo fitting")

    rank = _score_pairs(mlp, transformer, pairs, device)
    values = {
        "train_mlp": _value_report(mlp, train_only, device, False),
        "train_transformer": _value_report(transformer, train_only, device, True),
        "val_mlp": _value_report(mlp, [ep for ep in val if ep["data"]], device, False),
        "val_transformer": _value_report(transformer, [ep for ep in val if ep["data"]], device, True),
    }
    mlp_acc = _accuracy(rank["variants"]["mlp"]) if rank.get("variants") else None
    tr_acc = _accuracy(rank["variants"]["transformer_full"]) if rank.get("variants") else None
    mlp_margin = None
    tr_margin = None
    if rank.get("variants"):
        mlp_margin = rank["variants"]["mlp"]["pooled"]["mean_margin"] if rank["variants"]["mlp"]["pooled"] else None
        tr_margin = rank["variants"]["transformer_full"]["pooled"]["mean_margin"] if rank["variants"]["transformer_full"]["pooled"] else None
    enough = rank.get("n", 0) >= 8
    history_better = False
    if enough and mlp_acc is not None and tr_acc is not None:
        history_better = tr_acc >= mlp_acc + 0.05 or (
            tr_margin is not None and mlp_margin is not None and tr_margin > mlp_margin + 0.02 and tr_acc >= mlp_acc
        )
    td = None
    td_rank = None
    td_values = None
    if history_better:
        torch.manual_seed(0)
        mlp_td = TwinQ().to(device)
        tr_td = HistoryQ().to(device)
        td = {
            "mlp": _fit_td("td10_mlp", mlp_td, train_only, batches, env.model, device, False),
            "transformer": _fit_td("td10_transformer", tr_td, train_only, batches, env.model, device, True),
        }
        td_rank = _score_pairs(mlp_td, tr_td, pairs, device)
        td_values = {
            "train_mlp": _value_report(mlp_td, train_only, device, False),
            "train_transformer": _value_report(tr_td, train_only, device, True),
            "val_mlp": _value_report(mlp_td, [ep for ep in val if ep["data"]], device, False),
            "val_transformer": _value_report(tr_td, [ep for ep in val if ep["data"]], device, True),
        }
        phase_tr = tr_td
        phase_mlp = mlp_td
    else:
        phase_tr = transformer
        phase_mlp = mlp
    chosen = []
    retained = [ep for ep in train_only if ep["success"]]
    dropped = [ep for ep in train_only if not ep["success"]]
    if retained:
        chosen.append(sorted(retained, key=lambda ep: len(ep["data"]))[0])
    if dropped:
        chosen.append(sorted(dropped, key=lambda ep: len(ep["data"]))[0])
    for ep in skill_eps:
        if ep["data"]:
            chosen.append(ep)
    phase = _phase_trace(phase_mlp, phase_tr, chosen, device)
    cross = _cross_values(phase_mlp, phase_tr, device)
    payload = {
        "actor_checkpoint": str(CKPT),
        "actor_unchanged": _same_actor(env.model, actor_ref),
        "held_out_evaluated": False,
        "gamma": GAMMA,
        "reward_changed": False,
        "updates": UPDATES,
        "batch": BATCH,
        "critic_lr": CRITIC_LR,
        "sampler": "transition-uniform, identical index sequence for every critic",
        "normalization": "none",
        "actor": actor_info,
        "instantaneous_critic": critic_info,
        "history_critic": history_info,
        "alias": alias,
        "reproduction": repro,
        "skill_replay": skill_meta,
        "replay_train": _composition(train_only),
        "replay_val": _composition([ep for ep in val if ep["data"]]),
        "fork_tested": {
            "skill": skill_tested,
            "replay": replay_tested,
            "separated": len(pairs),
        },
        "mc_mlp": mc_mlp,
        "mc_transformer": mc_tr,
        "mc_values": values,
        "mc_ranking": rank,
        "history_better_under_mc": bool(history_better),
        "history_better_rule": "transformer pooled accuracy at least 5 points above the MLP, or a larger mean margin by 0.02 with accuracy no worse; requires at least 8 separated pairs",
        "td10": td,
        "td10_ranking": td_rank,
        "td10_values": td_values,
        "phase_source": "10-step TD critics" if history_better else "Monte Carlo critics",
        "phase": phase,
        "same_action_different_history": cross,
        "seconds": time.perf_counter() - clock,
    }
    _write(out_path, payload)
    print("WROTE", out_path, "seconds", round(payload["seconds"], 1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
