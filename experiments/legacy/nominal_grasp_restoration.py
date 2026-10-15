"""Privileged existence test for object-in-hand recentering after impact_zp6 recatch.

Starts from the first sustained reclose. Every command is a legal v3 action.
Does not train and does not edit the frozen observation, codebook, or exit rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.recatch.natural_loss_recatch_data_scaling as scale
import training.natural_loss_recatch_transformer_v3 as v3
from controllers.geometry_switch import GeometrySwitch, hand_pose
from controllers.recovery_actions import AIRBORNE_V3, grip_tau, normalize_action
from training.autonomous_episode import T_TASK, Z_ABORT, fresh_nominal_state, make_training_env, nominal_substep, seat_base
from training.autonomous_rl_sanity import restore_supports, supports_of
from experiments.legacy.film_impact_zp6_k33 import load_policy
from training.full_3d_airborne_recatch_dataset import TRACK, drop_supports, restore_to
from training.history_aware_critic_ablation import restore_capture, take_capture
from training.nominal_pose_audit import (
    HORIZON,
    PERSIST,
    SPEC_ZP6,
    capture_ref,
    choose_tol,
    errors,
    nominal_hold,
    policy_action,
    rotvec,
    undisturbed,
    until_impact,
)
from experiments.legacy.recovery_policy_pretraining_preparation import schedules

RAW = ROOT / "results" / "diagnostics" / "raw" / "nominal_grasp_restoration"
TOL_FALLBACK = {"ori": 0.117, "pos": 0.00303, "speed": 0.020, "persist_steps": 5}


def clip_vh(v):
    v = np.array(v, float).reshape(3)
    v[0] = float(np.clip(v[0], -1.0, 1.0))
    v[1] = float(np.clip(v[1], -1.0, 1.0))
    v[2] = float(np.clip(v[2], -1.0, 3.8))
    return v


def pinch_error(sim, ref):
    p, _rel = hand_pose(sim)
    return p - ref["p_nom"], p


def act_of(sim, ref, v_h, grip, wrist=0.0):
    rd = np.array(sim.fsm.r_des, float).reshape(3, 3)
    w_world = np.zeros(3)
    if wrist:
        w_world = rotvec(ref["R_hand"] @ rd.T)
        mag = float(np.linalg.norm(w_world))
        if mag > 1e-8:
            w_world = w_world / mag * min(wrist, mag * 2.0)
    return normalize_action(clip_vh(v_h), rd.T @ w_world, grip_tau(grip), AIRBORNE_V3)


def reach_fork(env, ref):
    env.start_recovery()
    seen_loss = False
    stable = 0
    pre = []
    while float(env.sim.data.time) < T_TASK:
        obs = np.asarray(env.observation(), np.float32).copy()
        prev = np.asarray(env._prev, np.float32).copy()
        token, act = policy_action(env)
        row = errors(env.sim, ref)
        dp, _p = pinch_error(env.sim, ref)
        pre.append({
            "obs": obs, "prev": prev, "act": act.copy(), "token": token,
            "t": row["t"], "ep": row["ep"], "eR": row["eR"], "vrel": row["vrel"],
            "nL": row["nL"], "nR": row["nR"], "z": row["z"], "dp": dp.copy(),
        })
        if row["nL"] == 0 and row["nR"] == 0:
            seen_loss = True
            stable = 0
        elif seen_loss and float(act[6]) > 0.5 and row["nL"] > 0 and row["nR"] > 0:
            stable += 1
            if stable >= PERSIST:
                cap = take_capture(env)
                cap["t"] = row["t"]
                return cap, pre, row, dp
        else:
            stable = 0
        if row["z"] < Z_ABORT:
            break
    raise RuntimeError("post-recatch fork was not reached")


def summarize(kind, mechanism, row0, dp0, trace, hold):
    eps = [float(r["ep"]) for r in trace] or [float(row0["ep"])]
    ers = [float(r["eR"]) for r in trace] or [float(row0["eR"])]
    shrunk = float(min(eps)) < float(row0["ep"]) - 0.002
    return {
        "kind": kind,
        "mechanism": mechanism,
        "ep0": float(row0["ep"]),
        "eR0": float(row0["eR"]),
        "dp0": [round(float(x), 5) for x in dp0],
        "ep_min": float(min(eps)),
        "eR_min": float(min(ers)),
        "ep_end": float(eps[-1]),
        "eR_end": float(ers[-1]),
        "relative_motion": bool(shrunk),
        "basin": bool(hold is not None),
        "handoff": None if hold is None else hold["t_handoff"],
        "held_10": None if hold is None else hold["held_10"],
        "held_12": None if hold is None else hold["held_12"],
        "drop_t": None if hold is None else hold["drop_t"],
        "n_steps": len(trace),
        "end_z": None if not trace else trace[-1]["z"],
        "end_nL": None if not trace else trace[-1]["nL"],
        "end_nR": None if not trace else trace[-1]["nR"],
    }


def play(env, cap, ref, tol, kind, mechanism, commands, record=False, after=None):
    restore_capture(env, cap)
    if after is not None:
        after()
    row0 = errors(env.sim, ref)
    dp0, _p = pinch_error(env.sim, ref)
    trace = []
    actions = []
    frames = []
    run = 0
    hold = None
    for cmd in commands:
        if float(env.sim.data.time) and errors(env.sim, ref)["z"] < Z_ABORT:
            break
        obs = np.asarray(env.observation(), np.float32).copy()
        prev = np.asarray(env._prev, np.float32).copy()
        act = np.asarray(cmd(env.sim, ref), np.float32)
        v3.step_v3(env.sim, act)
        env._prev = act.copy()
        row = errors(env.sim, ref)
        dp, _p = pinch_error(env.sim, ref)
        row["dp"] = [round(float(x), 5) for x in dp]
        row["grip"] = round(float(act[6]), 3)
        trace.append(row)
        actions.append(act.copy())
        if record:
            frames.append({"obs": obs, "prev": prev, "act": act.copy(), "row": row})
        ok = (
            row["nL"] > 0 and row["nR"] > 0 and row["eR"] <= tol["ori"]
            and row["ep"] <= tol["pos"] and row["vrel"] <= tol["speed"] and float(act[6]) > 0.5
        )
        run = run + 1 if ok else 0
        if row["z"] < Z_ABORT:
            break
        if run >= PERSIST:
            hold = nominal_hold(env, ref, HORIZON, restore_orientation=True)
            break
    out = summarize(kind, mechanism, row0, dp0, trace, hold)
    out["actions"] = actions
    out["commands"] = commands
    out["frames"] = frames
    out["trace_tail"] = [
        {k: r[k] for k in ("t", "ep", "eR", "vrel", "nL", "nR", "z", "grip", "dp")}
        for r in (trace[:4] + trace[-4:])
    ]
    return out


def fixed_commands(vh, grip, n, wrist=0.0):
    vh = np.array(vh, float)

    def _one(_sim, ref, vh=vh, grip=grip, wrist=wrist):
        return act_of(_sim, ref, vh, grip, wrist)

    return [_one for _ in range(int(n))]


def skill_commands(name):
    acts = [v3.v1_normalized_to_v3(np.asarray(a, np.float32)) for a in schedules()[name]]

    def _at(index):
        def _one(_sim, _ref, index=index):
            return acts[index]
        return _one

    return [_at(i) for i in range(len(acts))]


def snap_commands(grip_open, n_open, n_close=20):
    cmds = []

    def _open(sim, ref):
        dp, _p = pinch_error(sim, ref)
        return act_of(sim, ref, dp / 0.02, grip_open, 0.0)

    def _close(sim, ref):
        dp, _p = pinch_error(sim, ref)
        return act_of(sim, ref, np.clip(4.0 * dp, -0.12, 0.12), 1.0, 1.0)

    for _ in range(int(n_open)):
        cmds.append(_open)
    for _ in range(int(n_close)):
        cmds.append(_close)
    return cmds


def chase_commands(prelift, open_s, kp, vcap, grip, limit=90):
    cmds = []

    def _make(mode):
        def _one(sim, ref, mode=mode):
            dp, _p = pinch_error(sim, ref)
            if mode == "lift":
                return act_of(sim, ref, np.array([0.0, 0.0, -0.35]), 1.0, 0.0)
            if mode == "open":
                step = np.clip(kp * dp, -vcap, vcap)
                return act_of(sim, ref, step, grip, 0.0)
            return act_of(sim, ref, np.clip(3.0 * dp, -0.15, 0.15), 1.0, 1.0)
        return _one

    n_lift = int(round(prelift / 0.02))
    n_open = int(round(open_s / 0.02))
    n_close = max(0, int(limit) - n_lift - n_open)
    for _ in range(n_lift):
        cmds.append(_make("lift"))
    for _ in range(n_open):
        cmds.append(_make("open"))
    for _ in range(n_close):
        cmds.append(_make("close"))
    return cmds


def perturb(sim, dp_h=None, dv_h=None, w_h=None):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    qadr = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    if dp_h is not None:
        sim.data.qpos[qadr:qadr + 3] += rh @ np.asarray(dp_h, float)
    if dv_h is not None:
        sim.data.qvel[dadr:dadr + 3] += rh @ np.asarray(dv_h, float)
    if w_h is not None:
        sim.data.qvel[dadr + 3:dadr + 6] += rh @ np.asarray(w_h, float)
    mujoco.mj_forward(sim.model, sim.data)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_training_env()
    table = supports_of(env.sim)
    model, _proto, _cap = load_policy(device)
    env.model = model.to(env.device)
    env.model.eval()
    nom = undisturbed(env)
    tol = choose_tol(nom, 1.75)
    print("TOL", {k: round(tol[k], 5) if isinstance(tol[k], float) else tol[k] for k in tol}, flush=True)
    restore_supports(env.sim, table)
    ref, _ball = until_impact(env, SPEC_ZP6)
    if ref is None or not env.switch.entered:
        raise RuntimeError("impact_zp6 did not enter")
    cap, pre, row, dp = reach_fork(env, ref)
    print("FORK", row["t"], "eR", round(row["eR"], 4), "ep", round(row["ep"], 4), "dp", np.round(dp, 4), flush=True)
    if not (0.012 < row["ep"] < 0.028 and row["eR"] < 0.05):
        raise RuntimeError(f"fork is not the inspected reclose: ep={row['ep']} eR={row['eR']}")

    trials = []

    def run(kind, mechanism, commands, record=False):
        out = play(env, cap, ref, tol, kind, mechanism, commands, record=record)
        brief = {k: out[k] for k in (
            "mechanism", "ep_min", "eR_min", "ep_end", "relative_motion", "basin", "held_10", "held_12", "end_z"
        )}
        print(kind, brief, flush=True)
        trials.append(out)
        return out

    run("A_closed", "closed grip, hand velocity along the pinch error", fixed_commands(np.zeros(3), 1.0, 1)[:0] or chase_commands(0.0, 0.8, 12.0, 0.5, 1.0, limit=40), False)
    trials[-1]["mechanism"] = "closed grip slides along the pinch error"
    trials[-1]["kind"] = "A_closed"

    for name, label in (
        ("gravity_inward", "inward sliding"),
        ("gravity_outward", "outward sliding"),
        ("wrist_reseat", "wrist reseating"),
        ("recatch_open20", "open-finger recatch"),
        ("recatch_open60", "open-finger recatch, longer opening"),
    ):
        run("existing", label, skill_commands(name))
    run("existing", "inward sliding then wrist reseating", skill_commands("gravity_inward") + skill_commands("wrist_reseat"))
    run("existing", "open-finger recatch then wrist reseating", skill_commands("recatch_open20") + skill_commands("wrist_reseat"))
    run("existing", "wrist reseating then inward sliding", skill_commands("wrist_reseat") + skill_commands("gravity_inward"))

    for n_open, grip in ((1, -1.0), (2, -1.0), (3, -1.0), (4, -0.48), (6, -0.48), (4, 0.0)):
        run("release", f"open {n_open} steps at grip {grip:+.2f}, then reclose", snap_commands(grip, n_open))
    for prelift in (0.20, 0.40):
        run("hybrid", f"closed lift {prelift:.2f}s, then 2-step release and reclose", chase_commands(prelift, 0.04, 50.0, 1.0, -1.0, limit=40))
    for grip, kp, open_s, prelift in (
        (-1.0, 8.0, 0.30, 0.0),
        (-1.0, 20.0, 0.50, 0.0),
        (-1.0, 8.0, 0.40, 0.25),
        (-0.48, 6.0, 0.80, 0.0),
        (-0.48, 6.0, 0.80, 0.25),
        (0.0, 10.0, 0.60, 0.20),
    ):
        run(
            "hybrid",
            f"grip {grip:+.2f}, kp {kp:.0f}, open {open_s:.2f}s, prelift {prelift:.2f}s",
            chase_commands(prelift, open_s, kp, 0.8, grip, limit=80),
        )

    # If a partial reduction appeared, refine around the best non-basin trial.
    best = min(trials, key=lambda r: r["ep_min"])
    print("BEST", best["mechanism"], round(best["ep_min"], 4), flush=True)
    if best["ep_min"] < row["ep"] - 0.004 and not any(r["basin"] for r in trials):
        for grip in (-1.0, -0.48, 0.2):
            for open_s in (0.10, 0.20, 0.40, 0.70):
                for kp in (4.0, 12.0, 30.0):
                    run(
                        "refine",
                        f"refine grip {grip:+.2f} kp {kp:.0f} open {open_s:.2f}s",
                        chase_commands(0.20, open_s, kp, 0.9, grip, limit=70),
                    )

    success = next((r for r in trials if r["held_12"]), None)
    if success is None:
        success = next((r for r in trials if r["basin"]), None)
    robust = []
    if success is not None and success["held_12"]:
        print("ROBUST", success["mechanism"], flush=True)
        offsets = []
        for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
            dp_h = np.zeros(3)
            dp_h[axis] = 0.002 * sign
            offsets.append((f"pos_axis{axis}_{sign:+.0f}", dp_h, None, None))
        for axis, sign in ((0, 1.0), (0, -1.0), (2, 1.0), (2, -1.0)):
            dv = np.zeros(3)
            dv[axis] = 0.03 * sign
            offsets.append((f"vel_axis{axis}_{sign:+.0f}", None, dv, None))
        for axis, sign in ((0, 1.0), (0, -1.0), (2, 1.0), (2, -1.0)):
            ww = np.zeros(3)
            ww[axis] = 0.2 * sign
            offsets.append((f"ang_axis{axis}_{sign:+.0f}", None, None, ww))
        for name, dp_h, dv_h, w_h in offsets:
            def _shift(dp_h=dp_h, dv_h=dv_h, w_h=w_h):
                perturb(env.sim, dp_h, dv_h, w_h)

            shifted = play(env, cap, ref, tol, "robust", name, success["commands"], after=_shift)
            item = {
                "name": name,
                "ep0": shifted["ep0"],
                "ep_min": shifted["ep_min"],
                "basin": shifted["basin"],
                "held_10": shifted["held_10"],
                "held_12": shifted["held_12"],
            }
            robust.append(item)
            print(" pert", item, flush=True)

    controls = []
    print("CONTROLS", flush=True)
    base, _q = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    from training.nominal_pose_audit import roll_policy

    def enter_offset(spec):
        info = restore_to(env.sim, base, env.nominal, env.pads, env.pad_mu)
        drop_supports(env.sim)
        if not info.get("pose_ok", False):
            raise RuntimeError("offset restore failed")
        env.switch = GeometrySwitch()
        env.switch.capture_if_ready(env.sim)
        if env.switch.p0 is None:
            env.switch.p0, env.switch.r0 = hand_pose(env.sim)
        ref_c = capture_ref(env.sim, env.switch)
        scale.apply_spec(env.sim, spec)
        st = fresh_nominal_state()
        while float(env.sim.data.time) < T_TASK - 1e-9 and not env.switch.entered:
            for _ in range(v3.N_SUB):
                nominal_substep(env.sim, TRACK, st)
            row_c = env.switch.measure(env.sim)
            env.switch.update_entry(env.sim)
            if row_c["z"] < Z_ABORT:
                break
        return ref_c, bool(env.switch.entered)

    for name, dz, dvz in (
        ("train_dz13.5", -13.5, 0.0),
        ("nl_dvz_p", -13.5, 0.005),
        ("dz13.2", -13.2, 0.0),
        ("dvz_m", -13.5, -0.01),
    ):
        ref_c, entered = enter_offset({"kind": "offset", "dz": dz, "dx": 0.0, "dy": 0.0, "dvz": dvz})
        if not entered:
            controls.append({"case": name, "entered": False})
            print(name, "no entry", flush=True)
            continue
        out = roll_policy(env, ref_c, tol, 80, stop_on_basin=True)
        hold = None
        if out["basin"] is not None:
            hold = nominal_hold(env, ref_c, HORIZON, restore_orientation=True)
        controls.append({
            "case": name,
            "entered": True,
            "basin": out["basin"] is not None,
            "eR": None if out["basin"] is None else out["basin"]["row"]["eR"],
            "ep": None if out["basin"] is None else out["basin"]["row"]["ep"],
            "held_10": None if hold is None else hold["held_10"],
            "held_12": None if hold is None else hold["held_12"],
            "fell": out["fell"],
        })
        print(name, controls[-1], flush=True)

    winner = next((r for r in trials if r.get("held_12")), None)
    if winner is not None:
        restore_capture(env, cap)
        # Rebuild the saved episode: policy prefix through the fork, then the winning actions.
        # The winning actions were computed closed-loop, so replay those executed actions.
        acts = [np.asarray(a, np.float32) for a in winner["actions"]]
        obs_l, prev_l, act_l, ep_l, eR_l, v_l, nL_l, nR_l = [], [], [], [], [], [], [], []
        for frame in pre:
            obs_l.append(frame["obs"])
            prev_l.append(frame["prev"])
            act_l.append(frame["act"])
            ep_l.append(frame["ep"])
            eR_l.append(frame["eR"])
            v_l.append(frame["vrel"])
            nL_l.append(frame["nL"])
            nR_l.append(frame["nR"])
        for act in acts:
            obs_l.append(np.asarray(env.observation(), np.float32))
            prev_l.append(np.asarray(env._prev, np.float32).copy())
            v3.step_v3(env.sim, act)
            env._prev = act.copy()
            cur = errors(env.sim, ref)
            act_l.append(act)
            ep_l.append(cur["ep"])
            eR_l.append(cur["eR"])
            v_l.append(cur["vrel"])
            nL_l.append(cur["nL"])
            nR_l.append(cur["nR"])
        np.savez(
            RAW / "success_trajectory.npz",
            obs=np.stack(obs_l),
            prev=np.stack(prev_l),
            act=np.stack(act_l),
            ep=np.asarray(ep_l),
            eR=np.asarray(eR_l),
            vrel=np.asarray(v_l),
            nL=np.asarray(nL_l),
            nR=np.asarray(nR_l),
            mechanism=winner["mechanism"],
        )

    def public(row):
        return {k: row[k] for k in row if k not in ("actions", "frames", "trace_tail", "commands")}

    payload = {
        "fork": {"t": row["t"], "eR": row["eR"], "ep": row["ep"], "dp": [float(x) for x in dp], "vrel": row["vrel"], "nL": row["nL"], "nR": row["nR"], "z": row["z"]},
        "tolerance": tol,
        "nominal": {k: nom[k] for k in ("eR_max", "ep_max", "v_max")},
        "trials": [{**public(r), "trace_tail": r["trace_tail"]} for r in trials],
        "robust": robust,
        "controls": controls,
        "winner": None if winner is None else winner["mechanism"],
    }
    (RAW / "search.json").write_text(json.dumps(payload), encoding="utf-8")
    print("WROTE", RAW / "search.json", "winner", payload["winner"], flush=True)


if __name__ == "__main__":
    main()
