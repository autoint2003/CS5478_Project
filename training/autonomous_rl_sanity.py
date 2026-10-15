"""Short off-policy fine-tune of the 39-D policy on autonomous episodes.

The actor is the existing causal Transformer, initialized from ckpt_39d.pt.
A separate twin critic is not part of that architecture. This script does not
change the geometry switch, the observation, the v3 map, or the tokenizer.
It does not sample the held-out test episodes, and it does not train for long.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.grasp_seat import apply_spec
import training.natural_loss_recatch_transformer_v3 as v3
from controllers.geometry_switch import GeometrySwitch, hand_pose
from controllers.recovery_actions import AIRBORNE_V2, normalize_action
from envs.airborne_obs import OBS_DIM_AIR, observe_airborne, relative_state
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import physical_pack
from training.autonomous_episode import (
    Z_ABORT,
    ball_substep,
    empty_ball,
    fresh_nominal_state,
    make_training_env,
    nominal_substep,
    seat_base,
)
from training.full_3d_airborne_recatch_dataset import TRACK, drop_supports, restore_to, step_mapped
from training.recovery_runtime import T_TASK

RAW = ROOT / "results" / "diagnostics" / "raw" / "autonomous_rl"
DEMO = ROOT / "results" / "diagnostics" / "raw" / "autonomous_training"
GAMMA = 0.99
SIGMA = 0.05
ACTOR_LR = 3e-5
CRITIC_LR = 1e-3
BC_COEF = 5.0
RL_COEF = 0.05
POLYAK = 0.005
# Shaping per 20 ms step is a few thousandths. A 1.0 retain bonus dominates.
SHAPE_CONTACT = 0.002
SHAPE_HEIGHT = 0.001
SHAPE_POS = 0.001
SHAPE_SPEED = 0.0005
TERMINAL_RETAIN = 1.0

# Exact held-out and validation conditions. Training samples must not hit them.
FORBIDDEN = (
    {"kind": "impact", "v": 6.5, "z": 0.006, "phi": 0.0},
    {"kind": "impact", "v": 6.5, "z": 0.0, "phi": 90.0},
    {"kind": "impact", "v": 6.5, "z": -0.003, "phi": 0.0},
    {"kind": "air", "speed": 0.30, "sign": 1.0},
    {"kind": "air", "speed": 0.45, "sign": 1.0},
    {"kind": "air", "speed": 0.30, "sign": -1.0},
    {"kind": "air", "speed": 0.20, "sign": -1.0},
)


def shape_of(row):
    return reward_detail(row)["shaping"]


def reward_detail(row):
    """Per-component shaping. The sum is the reward used by the update."""
    bilateral = 1.0 if row["nL"] > 0 and row["nR"] > 0 else 0.0
    height = float(np.clip((row["z"] - 0.44) / 0.20, 0.0, 1.0))
    pos = float(np.clip(row["e_pos"] / 0.05, 0.0, 1.0))
    speed = float(np.clip(row["vrel"] / 0.20, 0.0, 1.0))
    contact = SHAPE_CONTACT * bilateral
    height_t = SHAPE_HEIGHT * height
    pos_t = -SHAPE_POS * pos
    speed_t = -SHAPE_SPEED * speed
    return {
        "contact": contact,
        "height": height_t,
        "pos": pos_t,
        "speed": speed_t,
        "shaping": contact + height_t + pos_t + speed_t,
        "e_pos": float(row["e_pos"]),
        "e_rot": float(row["e_rot"]),
        "vrel": float(row["vrel"]),
        "z": float(row["z"]),
        "bilateral": bilateral,
    }


def supports_of(sim):
    saved = []
    for name in ("floor", "table"):
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        saved.append((int(gid), int(sim.model.geom_contype[gid]), int(sim.model.geom_conaffinity[gid])))
    return saved


def restore_supports(sim, saved):
    for gid, contype, conaffinity in saved:
        sim.model.geom_contype[gid] = contype
        sim.model.geom_conaffinity[gid] = conaffinity


class TwinQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = self._net()
        self.q2 = self._net()

    @staticmethod
    def _net():
        return torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM_AIR + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def sample_impact(rng):
    """Neighborhood of impact_zm6. The box excludes z3, zp6, and phi90."""
    return {
        "kind": "impact",
        "v_mps": float(rng.uniform(5.8, 7.2)),
        "z_off_m": float(rng.uniform(-0.009, -0.0045)),
        "phi_deg": float(rng.uniform(-20.0, 20.0)),
    }


def assert_allowed(spec):
    if spec["kind"] == "impact":
        for ban in FORBIDDEN:
            if ban["kind"] != "impact":
                continue
            if (
                abs(spec["v_mps"] - ban["v"]) < 1e-6
                and abs(spec["z_off_m"] - ban["z"]) < 1e-6
                and abs(spec["phi_deg"] - ban["phi"]) < 1e-6
            ):
                raise RuntimeError(f"training sample hit a held-out impact {spec}")
        if abs(spec["z_off_m"] - (-0.003)) < 5e-4 or spec["z_off_m"] > -0.004:
            raise RuntimeError(f"impact sample left the training box {spec}")
        if abs(spec["phi_deg"]) > 25:
            raise RuntimeError(f"azimuth left the training box {spec}")
    elif spec["kind"] == "air":
        for ban in FORBIDDEN:
            if ban["kind"] != "air":
                continue
            if abs(spec["speed"] - ban["speed"]) < 1e-6 and int(np.sign(spec["sign"])) == int(np.sign(ban["sign"])):
                raise RuntimeError(f"training sample hit a held-out carry {spec}")


def _policy_mean(env):
    obs = np.asarray(env.observation(), np.float32)
    prev = env._prev.copy()
    env._obs_seq.append(obs)
    env._prev_seq.append(prev)
    obs_in = env._obs_seq[-160:]
    prev_in = env._prev_seq[-160:]
    with torch.no_grad():
        pred = env.model(
            torch.tensor(np.stack(obs_in), device=env.device),
            torch.tensor(np.stack(prev_in), device=env.device),
        )[2][-1].float().cpu().numpy()
    return obs, prev, np.clip(pred, -1.0, 1.0).astype(np.float32)


def _push(buf, obs, prev, act, rew, done, truncated, detail=None):
    item = {
        "obs": np.asarray(obs, np.float32),
        "prev": np.asarray(prev, np.float32),
        "act": np.asarray(act, np.float32),
        "rew": float(rew),
        "done": bool(done),
        "truncated": bool(truncated),
        "terminal": 0.0,
    }
    if detail is not None:
        if abs(float(detail["shaping"]) - float(rew)) > 1e-6:
            raise RuntimeError("shaping components do not add up to the step reward")
        for key in ("contact", "height", "pos", "speed", "shaping", "e_pos", "e_rot", "vrel", "z", "bilateral"):
            item[key] = float(detail[key])
    buf.append(item)


def _finish_rewards(buf, start, t12):
    if start >= len(buf):
        return
    terminal = TERMINAL_RETAIN if t12 == "RETAINED_TO_12" else 0.0
    buf[-1]["rew"] += terminal
    buf[-1]["done"] = True
    buf[-1]["truncated"] = False
    buf[-1]["terminal"] = terminal


def collect_impact(env, spec, rng, buf, sigma, on_decision=None, check=True):
    if check:
        assert_allowed(spec)
    env.begin_nominal()
    ball = empty_ball()
    switch = env.switch
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            ball_substep(
                env.sim, env.gains, env.nominal, ball,
                spec["v_mps"], spec["z_off_m"], spec["phi_deg"], None,
            )
        row = switch.measure(env.sim)
        if ball["released"]:
            switch.update_entry(env.sim)
        if ball["released"] and row["z"] < Z_ABORT:
            break
        if (not ball["released"]) and float(env.sim.data.time) > 8.0:
            break
    if not switch.entered:
        t12, _t = env.finish_nominal(env.gains)
        return {"spec": spec, "entered": False, "t12": t12, "steps": 0, "exit": "no_entry"}
    env.start_recovery()
    start = len(buf)
    exit_kind = "time_cap"
    while float(env.sim.data.time) < T_TASK - 1e-9:
        obs, prev, mean = _policy_mean(env)
        noise = rng.normal(0.0, sigma, size=7).astype(np.float32)
        act = np.clip(mean + noise, -1.0, 1.0)
        if on_decision is not None:
            on_decision(env, obs, prev, act)
        v3.step_v3(env.sim, act)
        env._prev = act
        row = switch.measure(env.sim)
        detail = reward_detail(row)
        rew = detail["shaping"]
        done = False
        if row["z"] < Z_ABORT:
            exit_kind = "fall_abort"
            done = True
            _push(buf, obs, prev, act, rew, done, False, detail)
            break
        if switch.update_exit(env.sim):
            exit_kind = "geometry"
            _push(buf, obs, prev, act, rew, False, False, detail)
            break
        _push(buf, obs, prev, act, rew, False, False, detail)
    if exit_kind == "fall_abort":
        t12 = "DROP"
    else:
        t12, _t = env.finish_nominal(TRACK)
    _finish_rewards(buf, start, t12)
    return {
        "spec": spec, "entered": True, "t12": t12, "steps": len(buf) - start,
        "exit": exit_kind, "t_entry": switch.t_entry,
    }


def collect_offset(env, base, buf, sigma, rng, on_decision=None, spec=None):
    """Seated offset. The default is the training offset dz=-13.5 mm."""
    if spec is None:
        spec = {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.0}
    sim = env.sim
    info = restore_to(sim, base, env.nominal, env.pads, env.pad_mu)
    if not info.get("pose_ok", False):
        raise RuntimeError("offset restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(sim)
    apply_spec(sim, spec)
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
        return {"spec": spec, "entered": False, "t12": t12, "steps": 0, "exit": "no_entry"}
    env.start_recovery()
    start = len(buf)
    exit_kind = "time_cap"
    while float(sim.data.time) < T_TASK - 1e-9:
        obs, prev, mean = _policy_mean(env)
        act = np.clip(mean + rng.normal(0.0, sigma, size=7).astype(np.float32), -1.0, 1.0)
        if on_decision is not None:
            on_decision(env, obs, prev, act)
        v3.step_v3(sim, act)
        env._prev = act
        row = switch.measure(sim)
        detail = reward_detail(row)
        if row["z"] < Z_ABORT:
            _push(buf, obs, prev, act, detail["shaping"], True, False, detail)
            exit_kind = "fall_abort"
            break
        _push(buf, obs, prev, act, detail["shaping"], False, False, detail)
        if switch.update_exit(sim):
            exit_kind = "geometry"
            break
    t12 = "DROP" if exit_kind == "fall_abort" else env.finish_nominal(TRACK)[0]
    _finish_rewards(buf, start, t12)
    return {"spec": spec, "entered": True, "t12": t12, "steps": len(buf) - start, "exit": exit_kind}


def collect_air(env, base, q_ref, buf, rng, sigma, on_decision=None, speed=0.35, sign=-1.0, check=True):
    spec = {"kind": "air", "speed": float(speed), "sign": float(sign), "duration_s": 0.08}
    if check:
        assert_allowed(spec)
    sim = env.sim
    info = restore_to(sim, base, env.nominal, env.pads, env.pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        raise RuntimeError("carry restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(sim)
    switch = env.switch
    t0 = float(sim.data.time)
    open_at = t0 + spec["duration_s"]
    opened = False
    open_hold = 0
    st = fresh_nominal_state()
    while float(sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            t = float(sim.data.time)
            if open_hold < 10 and t < open_at + 0.20:
                v_h = np.zeros(3)
                if t < open_at:
                    v_h[1] = spec["sign"] * spec["speed"]
                    tau = -18.0
                else:
                    opened = True
                    v_h[2] = 0.3
                    tau = 2.0
                action = normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V2)
                step_mapped(sim, action, AIRBORNE_V2, TRACK, q_ref)
                o = physical_pack(sim)
                if opened and int(o["nL"]) == 0 and int(o["nR"]) == 0:
                    open_hold += 1
            else:
                nominal_substep(sim, TRACK, st)
        if switch.update_entry(sim):
            break
    if not switch.entered:
        t12, _t = env.finish_nominal(TRACK)
        return {"spec": spec, "entered": False, "t12": t12, "steps": 0, "exit": "no_entry"}
    env.start_recovery()
    start = len(buf)
    exit_kind = "time_cap"
    while float(sim.data.time) < T_TASK - 1e-9:
        obs, prev, mean = _policy_mean(env)
        act = np.clip(mean + rng.normal(0.0, sigma, size=7).astype(np.float32), -1.0, 1.0)
        if on_decision is not None:
            on_decision(env, obs, prev, act)
        v3.step_v3(sim, act)
        env._prev = act
        row = switch.measure(sim)
        detail = reward_detail(row)
        if row["z"] < Z_ABORT:
            _push(buf, obs, prev, act, detail["shaping"], True, False, detail)
            exit_kind = "fall_abort"
            break
        _push(buf, obs, prev, act, detail["shaping"], False, False, detail)
        if switch.update_exit(sim):
            exit_kind = "geometry"
            break
    t12 = "DROP" if exit_kind == "fall_abort" else env.finish_nominal(TRACK)[0]
    _finish_rewards(buf, start, t12)
    return {"spec": spec, "entered": True, "t12": t12, "steps": len(buf) - start, "exit": exit_kind}


def load_demos(device):
    demos = []
    for name in ("wrist_reseat", "gravity_inward", "recatch_open20"):
        z = np.load(DEMO / f"demo_{name}.npz")
        demos.append({
            "name": name,
            "obs": torch.tensor(z["obs"], device=device),
            "prev": torch.tensor(z["prev"], device=device),
            "act": torch.tensor(z["act"], device=device),
        })
    return demos


def demo_rmse(model, demos):
    model.eval()
    out = {}
    with torch.no_grad():
        for demo in demos:
            pred = model(demo["obs"].unsqueeze(0), demo["prev"].unsqueeze(0))[2][0]
            err = (pred - demo["act"]).pow(2).mean().sqrt()
            out[demo["name"]] = float(err)
    return out


def action_stats(actions):
    if not actions:
        return {"n": 0}
    a = np.stack(actions)
    return {
        "n": int(a.shape[0]),
        "mean_abs": round(float(np.mean(np.abs(a))), 4),
        "max_abs": round(float(np.max(np.abs(a))), 4),
        "frac_saturated": round(float(np.mean(np.abs(a) > 0.98)), 4),
        "mean": [round(float(x), 4) for x in a.mean(0)],
    }


def update_once(model, critic, target, opt_a, opt_c, buf, demos, device):
    """One SAC-style update. The last transition is terminal, so it is not bootstrapped.

    truncated is recorded and would bootstrap; this sanity never sets it, because
    every recovery episode is simulated through the physical t=12 outcome.
    """
    model.train()
    critic.train()
    idx = np.random.randint(0, len(buf) - 1) if len(buf) > 1 else 0
    # Prefer a non-final index when the next sample exists in the same sense:
    # bootstrap only when this transition is not done and not the buffer end.
    batch_i = np.random.choice(len(buf), size=min(64, len(buf)), replace=False)
    obs = torch.tensor(np.stack([buf[i]["obs"] for i in batch_i]), device=device)
    act = torch.tensor(np.stack([buf[i]["act"] for i in batch_i]), device=device)
    rew = torch.tensor([buf[i]["rew"] for i in batch_i], device=device)
    done = torch.tensor([buf[i]["done"] for i in batch_i], dtype=torch.float32, device=device)
    truncated = torch.tensor([buf[i]["truncated"] for i in batch_i], dtype=torch.float32, device=device)
    # Next state is the following buffer row only when it continues the episode.
    # A done/truncated row uses its own next fields stored as zeros and a mask.
    nxt_obs = []
    nxt_prev = []
    for i in batch_i:
        if i + 1 < len(buf) and not buf[i]["done"]:
            nxt_obs.append(buf[i + 1]["obs"])
            nxt_prev.append(buf[i + 1]["prev"])
        else:
            nxt_obs.append(buf[i]["obs"])
            nxt_prev.append(buf[i]["act"])
    nxt_obs = torch.tensor(np.stack(nxt_obs), device=device)
    nxt_prev = torch.tensor(np.stack(nxt_prev), device=device)
    with torch.no_grad():
        nxt_act = model(nxt_obs.unsqueeze(1), nxt_prev.unsqueeze(1))[2][:, 0]
        q1_t, q2_t = target(nxt_obs, nxt_act)
        boot = (1.0 - done) * (1.0 - truncated) + truncated
        # truncated would bootstrap; done does not. Here truncated is always 0,
        # so boot = 1 - done.
        y = rew + GAMMA * boot * torch.min(q1_t, q2_t)
    q1, q2 = critic(obs, act)
    loss_c = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
    opt_c.zero_grad(set_to_none=True)
    loss_c.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opt_c.step()
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), target.parameters()):
            tp.data.mul_(1.0 - POLYAK).add_(p.data, alpha=POLYAK)
    prev = torch.tensor(np.stack([buf[i]["prev"] for i in batch_i]), device=device)
    pi = model(obs.unsqueeze(1), prev.unsqueeze(1))[2][:, 0]
    q_pi, _q2 = critic(obs, pi)
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
    del idx
    with torch.no_grad():
        td_abs = (q1.detach() - y).abs()
        q1_std = float(q1.detach().std(unbiased=False)) if q1.numel() > 1 else 0.0
        y_std = float(y.std(unbiased=False)) if y.numel() > 1 else 0.0
        td_p90 = float(torch.quantile(td_abs, 0.9)) if td_abs.numel() > 1 else float(td_abs.mean())
    return {
        "critic": float(loss_c.detach()),
        "bc": float(bc.detach()),
        "q_pi": float(q_pi.mean().detach()),
        "target_mean": float(y.mean().detach()),
        "target_abs": float(y.abs().mean().detach()),
        "target_max": float(y.abs().max().detach()),
        "q1_abs": float(q1.abs().mean().detach()),
        "q1_mean": float(q1.detach().mean()),
        "q1_std": q1_std,
        "target_std": y_std,
        "td_mean": float(td_abs.mean()),
        "td_median": float(td_abs.median()),
        "td_p90": td_p90,
    }


def eval_impact(env, v, z, phi):
    from training.autonomous_episode import run_ball_episode
    env.model.eval()
    return run_ball_episode(env, v, z, phi, None)


def eval_seated(env, base, q_ref, tag):
    """Validation and held-out carries and grasp offsets. Not used as training data."""
    from data_generation.natural_specs import EVAL_NATURAL, TRAIN_NATURAL
    rows = []
    jobs = [
        ("air_lat_pos", "validation", "air", 0.30, 1.0),
        ("air_fast_pos", "validation", "air", 0.45, 1.0),
        ("air_lat_neg", "test", "air", 0.30, -1.0),
        ("air_slow_neg", "test", "air", 0.20, -1.0),
    ]
    for name, split, _kind, speed, sign in jobs:
        rows.append(_roll_air(env, base, q_ref, name, split, speed, sign, tag))
        print("EVAL", rows[-1], flush=True)
    specs = {s["tag"]: s for s in list(TRAIN_NATURAL) + list(EVAL_NATURAL)}
    for name, split, key in (
        ("nl_dz14.6", "validation", "dz14.6"),
        ("nl_dvz_p", "validation", "dvz_p"),
        ("nl_h_dz14.00", "test", "h_dz14.00"),
        ("nl_h_dz14.25", "test", "h_dz14.25"),
        ("nl_h_dy", "test", "h_dy"),
        ("nl_h_dx", "test", "h_dx"),
    ):
        rows.append(_roll_offset(env, base, name, split, specs[key], tag))
        print("EVAL", rows[-1], flush=True)
    return rows


def _arm(env, base):
    info = restore_to(env.sim, base, env.nominal, env.pads, env.pad_mu)
    drop_supports(env.sim)
    if not info.get("pose_ok", False):
        raise RuntimeError("seat restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(env.sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(env.sim)
    env.model.eval()


def _policy_until(env):
    env.start_recovery()
    switch = env.switch
    steps = 0
    exit_kind = "time_cap"
    while float(env.sim.data.time) < T_TASK - 1e-9:
        _obs, _prev, act = _policy_mean(env)
        v3.step_v3(env.sim, act)
        env._prev = act
        steps += 1
        row = switch.measure(env.sim)
        if row["z"] < Z_ABORT:
            exit_kind = "fall_abort"
            break
        if switch.update_exit(env.sim):
            exit_kind = "geometry"
            break
    if exit_kind == "fall_abort":
        return "DROP", exit_kind, steps
    t12, _t = env.finish_nominal(TRACK)
    return t12, exit_kind, steps


def _roll_air(env, base, q_ref, name, split, speed, sign, tag):
    _arm(env, base)
    sim = env.sim
    switch = env.switch
    t0 = float(sim.data.time)
    open_at = t0 + 0.08
    opened = False
    open_hold = 0
    st = fresh_nominal_state()
    while float(sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            t = float(sim.data.time)
            if open_hold < 10 and t < open_at + 0.20:
                v_h = np.zeros(3)
                if t < open_at:
                    v_h[1] = float(sign) * float(speed)
                    tau = -18.0
                else:
                    opened = True
                    v_h[2] = 0.3
                    tau = 2.0
                action = normalize_action(v_h, np.zeros(3), tau, AIRBORNE_V2)
                step_mapped(sim, action, AIRBORNE_V2, TRACK, q_ref)
                o = physical_pack(sim)
                if opened and int(o["nL"]) == 0 and int(o["nR"]) == 0:
                    open_hold += 1
            else:
                nominal_substep(sim, TRACK, st)
        if switch.update_entry(sim):
            break
    if not switch.entered:
        t12, _t = env.finish_nominal(TRACK)
        return {"id": name, "split": split, "tag": tag, "t12": t12, "exit": "no_entry", "steps": 0}
    t12, exit_kind, steps = _policy_until(env)
    return {"id": name, "split": split, "tag": tag, "t12": t12, "exit": exit_kind, "steps": steps}


def _roll_offset(env, base, name, split, spec, tag):
    _arm(env, base)
    apply_spec(env.sim, spec)
    st = fresh_nominal_state()
    switch = env.switch
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, TRACK, st)
        row = switch.measure(env.sim)
        switch.update_entry(env.sim)
        if row["z"] < Z_ABORT:
            break
    if not switch.entered:
        t12, _t = env.finish_nominal(TRACK)
        return {"id": name, "split": split, "tag": tag, "t12": t12, "exit": "no_entry", "steps": 0}
    t12, exit_kind, steps = _policy_until(env)
    return {"id": name, "split": split, "tag": tag, "t12": t12, "exit": exit_kind, "steps": steps}


def main():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    env = make_training_env()
    device = env.device
    init_state = {k: v.detach().cpu().clone() for k, v in env.model.state_dict().items()}
    demos = load_demos(device)
    before = demo_rmse(env.model, demos)
    print("SKILL_RMSE_BEFORE", before, flush=True)
    table = supports_of(env.sim)

    def impacts(tag):
        rows = []
        for name, v, z, phi in (
            ("impact_zm6", 6.5, -0.006, 0.0),
            ("impact_z3", 6.5, -0.003, 0.0),
            ("impact_zp6", 6.5, 0.006, 0.0),
            ("impact_phi90", 6.5, 0.0, 90.0),
        ):
            restore_supports(env.sim, table)
            row = eval_impact(env, v, z, phi)
            rec = {"id": name, "tag": tag, "t12": row["t12"], "exit": row["exit_kind"], "steps": row["steps"], "t_entry": row["t_entry"]}
            rows.append(rec)
            print("EVAL", rec, flush=True)
        return rows

    # Baseline closed-loop on the training impact and the validation/test impacts
    # is collected before any gradient step. Air and offset evals follow seating.
    base_impacts = impacts("A")
    print("BASE_IMPACTS_DONE", flush=True)
    # The remaining baseline numbers for air and natural are measured after the
    # sanity updates as well. Both policies see the same seated initial condition.
    critic = TwinQ().to(device)
    target = TwinQ().to(device)
    target.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(env.model.parameters(), lr=ACTOR_LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    buf = []
    train_log = []
    audits = []
    executed = []
    restore_supports(env.sim, table)
    for i in range(4):
        spec = sample_impact(rng)
        row = collect_impact(env, spec, rng, buf, SIGMA)
        train_log.append(row)
        executed.extend(b["act"] for b in buf[-max(row["steps"], 0):])
        print("TRAIN", i, row["t12"], row["exit"], row["steps"], spec, flush=True)
        if len(buf) < 2:
            continue
        for _ in range(8):
            audits.append(update_once(env.model, critic, target, opt_a, opt_c, buf, demos, device))
    after_imp = demo_rmse(env.model, demos)
    print("SKILL_RMSE_AFTER_IMPACT_UPDATES", after_imp, flush=True)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    off = collect_offset(env, base, buf, SIGMA, rng)
    air = collect_air(env, base, q_ref, buf, rng, SIGMA)
    train_log.extend([off, air])
    print("TRAIN_OFFSET", off, flush=True)
    print("TRAIN_AIR", air, flush=True)
    if len(buf) >= 2:
        for _ in range(8):
            audits.append(update_once(env.model, critic, target, opt_a, opt_c, buf, demos, device))
    after = demo_rmse(env.model, demos)
    print("SKILL_RMSE_AFTER", after, flush=True)
    print("ACTIONS", action_stats(executed), flush=True)
    if audits:
        last = audits[-1]
        print("CRITIC_LAST", {k: round(v, 4) for k, v in last.items()}, flush=True)
        print("CRITIC_MAX_TARGET", round(max(a["target_max"] for a in audits), 4), flush=True)
    restore_supports(env.sim, table)
    ft_impacts = impacts("B")
    ft_seated = eval_seated(env, base, q_ref, "B")
    env.model.load_state_dict(init_state)
    base_seated = eval_seated(env, base, q_ref, "A")
    RAW.mkdir(parents=True, exist_ok=True)
    payload = {
        "skill_rmse_before": before,
        "skill_rmse_after": after,
        "train": [
            {k: (v if not isinstance(v, dict) else {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in v.items()}) for k, v in row.items()}
            for row in train_log
        ],
        "base_impacts": base_impacts,
        "ft_impacts": ft_impacts,
        "base_seated": base_seated,
        "ft_seated": ft_seated,
        "actions": action_stats(executed),
        "critic_last": None if not audits else {k: round(v, 5) for k, v in audits[-1].items()},
        "n_updates": len(audits),
        "n_transitions": len(buf),
        "reward": {
            "shape_contact": SHAPE_CONTACT,
            "shape_height": SHAPE_HEIGHT,
            "shape_pos": SHAPE_POS,
            "shape_speed": SHAPE_SPEED,
            "terminal_retain": TERMINAL_RETAIN,
            "gamma": GAMMA,
        },
    }
    (RAW / "sanity_partial.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("PARTIAL", RAW / "sanity_partial.json", flush=True)
    return env, base, q_ref, table


if __name__ == "__main__":
    main()
