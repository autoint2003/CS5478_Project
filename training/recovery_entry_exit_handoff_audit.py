"""Handoff measurement for the entry/exit audit. Does not change the controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import physical_pack
from training.demo_ballistic_impact import TAU_SEC, z_tgt_of
from training.full_3d_airborne_recatch_dataset import restore_to, drop_supports
from training.observation_study_common import build_cases, load_balanced
from training.recovery_runtime import (
    T_TASK,
    Z_AIR,
    _dropped,
    _escaped,
    continue_long,
    physical_of,
    restore_dyn,
    stamp,
)
from training.replay_core import park_impact_ball, tick_vw

CASES = (
    "impact_zm6",
    "slide_suffix_20",
    "wrist_suffix_3",
    "recatch_suffix_8",
    "air_lat_pos",
    "nl_h_dz14.00",
)
OUT = ROOT / "results" / "diagnostics" / "raw" / "recovery_handoff_audit" / "summary.json"


def rot_err_deg(rd, rh):
    r = np.asarray(rd, float).reshape(3, 3).T @ np.asarray(rh, float).reshape(3, 3)
    c = float(np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def pose_row(sim, t0, mode):
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    obj = np.array(sim.data.xpos[sim.ids.object_body], float)
    p_des = np.asarray(sim.fsm.p_des, float)
    o = physical_pack(sim)
    return {
        "t": round(float(sim.data.time) - t0, 4),
        "abs_t": round(float(sim.data.time), 4),
        "mode": mode,
        "phase": str(sim.fsm.phase),
        "p_err_mm": round(float(np.linalg.norm(p_des - hand)) * 1e3, 3),
        "z_err_mm": round(float(p_des[2] - hand[2]) * 1e3, 3),
        "r_err_deg": round(rot_err_deg(sim.fsm.r_des, sim.data.xmat[sim.ids.hand_body]), 3),
        "obj_z": round(float(obj[2]), 4),
        "hand_z": round(float(hand[2]), 4),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "grip": round(float(sim.data.ctrl[7]), 3),
        "v_cmd": [round(float(x), 4) for x in np.asarray(sim.fsm.v_cmd, float)],
    }


def run_policy(sim, model, case):
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    device = next(model.parameters()).device
    t0 = float(sim.data.time)
    bilateral_steps = 0
    min_z = 10.0
    dropped = False
    with torch.no_grad():
        for _k in range(int(case["n_steps"])):
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(np.asarray(obs, np.float32))
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            v3.step_v3(sim, act)
            prev = act
            o = physical_pack(sim)
            min_z = min(min_z, float(o["obj_z"]))
            if int(o["nL"]) > 0 and int(o["nR"]) > 0:
                bilateral_steps += 1
            if float(o["obj_z"]) < 0.44:
                dropped = True
                break
    return {
        "t0": t0,
        "t1": float(sim.data.time),
        "steps": len(obs_seq),
        "early_drop": dropped,
        "min_z": round(min_z, 4),
        "bilateral_steps": bilateral_steps,
        "prev0": [0.0] * 7,
        "obs0": int(np.asarray(obs_seq[0]).shape[0]) if obs_seq else None,
    }


def traced(sim, gains):
    dt = float(sim.model.opt.timestep)
    t0 = float(sim.data.time)
    t0_z = float(physical_pack(sim)["obj_z"])
    t_drop = None
    t_esc = None
    airborne_seen = bool(t0_z >= Z_AIR)
    n = int(round(max(0.0, T_TASK - t0) / dt)) + 50
    rows = []
    next_log = 0.0
    first_mode = None
    for _ in range(n):
        z_now = float(z_tgt_of(sim))
        lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9
        mode = "lift_hold" if lift_hold else "nominal_fsm"
        if first_mode is None:
            first_mode = mode
        if lift_hold:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
        elapsed = float(sim.data.time) - t0
        if elapsed + 1e-9 >= next_log and next_log <= 0.20:
            rows.append(pose_row(sim, t0, mode))
            next_log += 0.02
        o = physical_pack(sim)
        if float(o["obj_z"]) >= Z_AIR:
            airborne_seen = True
        t = float(sim.data.time)
        if airborne_seen and _dropped(o) and t_drop is None:
            t_drop = t
        if _escaped(sim) and t_esc is None:
            t_esc = t
        if t_drop is not None and t >= t_drop + 0.15:
            break
        if t_esc is not None and t >= t_esc + 0.15:
            break
        if t >= T_TASK - 1e-9:
            break
    y, kind = physical_of(_end(sim), t_drop, t_esc)
    return {
        "t0": round(t0, 4),
        "t_end": round(float(sim.data.time), 4),
        "first_mode": first_mode,
        "t12": "RETAINED_TO_12" if y == 1 else kind,
        "t_drop": None if t_drop is None else round(float(t_drop), 4),
        "t_escape": None if t_esc is None else round(float(t_esc), 4),
        "trace": rows,
    }


def _end(sim):
    from training.recovery_runtime import _end_state
    return _end_state(sim)


def restore_case(ctx, case, snap):
    sim = ctx["sim_i"] if case["scene"] == "impact" else ctx["sim"]
    nominal = ctx["nominal_i"] if case["scene"] == "impact" else ctx["nominal"]
    pads = ctx["pads_i"] if case["scene"] == "impact" else ctx["pads"]
    pad_mu = ctx["pad_mu_i"] if case["scene"] == "impact" else ctx["pad_mu"]
    if case["scene"] == "impact":
        restore_dyn(sim, snap, nominal, pads, pad_mu, None, None)
        park_impact_ball(sim)
    else:
        restore_to(sim, snap, nominal, pads, pad_mu)
        drop_supports(sim)
    return sim


def handoff_picture(sim):
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    obj = np.array(sim.data.xpos[sim.ids.object_body], float)
    p_des = np.asarray(sim.fsm.p_des, float)
    z_tgt = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(p_des[2]) >= z_tgt - 1e-9
    o = physical_pack(sim)
    return {
        "time": round(float(sim.data.time), 4),
        "phase": str(sim.fsm.phase),
        "lift_started": bool(sim.fsm.lift_started),
        "z_tgt": round(z_tgt, 4),
        "lift_hold": bool(lift_hold),
        "p_err_mm": round(float(np.linalg.norm(p_des - hand)) * 1e3, 3),
        "z_gap_mm": round(float(p_des[2] - z_tgt) * 1e3, 3),
        "r_err_deg": round(rot_err_deg(sim.fsm.r_des, sim.data.xmat[sim.ids.hand_body]), 3),
        "v_cmd": [round(float(x), 4) for x in np.asarray(sim.fsm.v_cmd, float)],
        "w_cmd": [round(float(x), 4) for x in np.asarray(sim.fsm.w_cmd, float)],
        "grip": round(float(sim.data.ctrl[7]), 3),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": round(float(obj[2]), 4),
        "hand_z": round(float(hand[2]), 4),
    }


def rebase_to_hand(sim):
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    sim.fsm.r_des = np.array(sim.data.xmat[sim.ids.hand_body], float).reshape(3, 3).copy()
    sim.fsm.v_cmd[:] = 0.0
    sim.fsm.w_cmd[:] = 0.0
    sim.fsm.v_des[:] = 0.0
    sim.fsm.w_des[:] = 0.0


def main():
    ctx = build_cases()
    model, _proto, _saved, _device = load_balanced()
    by_id = {c["id"]: c for c in ctx["cases"]}
    rows = []
    for name in CASES:
        case = by_id[name]
        sim = restore_case(ctx, case, case["snap"])
        pre = handoff_picture(sim)
        meta = run_policy(sim, model, case)
        if meta["early_drop"]:
            rows.append({
                "id": name,
                "family": case["family"],
                "n_steps_budget": int(case["n_steps"]),
                "policy": meta,
                "at_policy_start": pre,
                "handoff": None,
                "A": None,
                "B": None,
            })
            print("EARLY", name, meta, flush=True)
            continue
        picture = handoff_picture(sim)
        snap = stamp(sim)
        from training.full_3d_airborne_recatch_dataset import TRACK
        sim_a = restore_case(ctx, case, snap)
        rec = continue_long(sim_a, TRACK, T_TASK, "A")
        y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        official = "RETAINED_TO_12" if y == 1 else kind
        sim_a = restore_case(ctx, case, snap)
        from training.full_3d_airborne_recatch_dataset import TRACK
        traced_a = traced(sim_a, TRACK)
        sim_b = restore_case(ctx, case, snap)
        rebase_to_hand(sim_b)
        traced_b = traced(sim_b, TRACK)
        rows.append({
            "id": name,
            "family": case["family"],
            "n_steps_budget": int(case["n_steps"]),
            "policy": meta,
            "at_policy_start": pre,
            "handoff": picture,
            "official_continue_long": official,
            "A": traced_a,
            "B": traced_b,
        })
        print(
            name,
            "official", official,
            "A", traced_a["t12"], traced_a["first_mode"],
            "B", traced_b["t12"], traced_b["first_mode"],
            "perr", picture["p_err_mm"],
            "rerr", picture["r_err_deg"],
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
