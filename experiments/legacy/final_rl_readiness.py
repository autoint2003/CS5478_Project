"""Freeze the success rule, rebuild live initial conditions, and audit GRPO signal.

Does not train. Does not roll out FINAL_TEST recoveries. The historical
diagnostic cases are not reused as this split.
"""

from __future__ import annotations


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
from envs.outcomes.nominal_pose import capture_ref, closed_grip, errors, hand_R, rotvec
from controllers.cartesian.tracking import TRACK
from envs.simulation.support_contacts import restore_supports, supports_of
import experiments.recatch.natural_loss.natural_loss_recatch_data_scaling as scale
import models.airborne_v3 as v3
import envs.outcomes.recovery_success as success
from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from controllers.recovery.action_map import AIRBORNE_V2, AIRBORNE_V3, normalize_action
from envs.outcomes.physical_recovery import physical_pack
from models.k48_policy import build_policy
from training.recovery_episode import (
    T_TASK,
    Z_ABORT,
    ball_substep,
    empty_ball,
    fresh_nominal_state,
    make_training_env,
    nominal_substep,
    seat_base,
)
from experiments.legacy.autonomous_rl_sanity import reward_detail
from data_generation.airborne_recatch_dataset import drop_supports, restore_to, step_mapped

from experiments.legacy.nominal_grasp_restoration import act_of, pinch_error
from experiments.legacy.nominal_pose_audit import until_impact
from experiments.legacy.recovery_policy_pretraining_preparation import schedules
from training.grpo_math import SIGMA

RAW = ROOT / "results" / "diagnostics" / "raw" / "final_rl_readiness"
CKPT = ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity" / "policy_A_reference.pt"
SKILL = ROOT / "results" / "diagnostics" / "raw" / "autonomous_training"
DEMO = ROOT / "results" / "diagnostics" / "raw" / "privileged_recatch"
TRAJ = ROOT / "results" / "diagnostics" / "raw" / "nominal_grasp_restoration" / "success_trajectory.npz"
CAP = 0.25
GROUP = 4
POLICY_CAP = 100

# Assigned before any rollout. None of these match the historical diagnostic IDs.
CATALOG = (
    {"id": "imp_a", "split": "train", "family": "impact", "v_mps": 5.5, "z_off_m": 0.009, "phi_deg": 0.0},
    {"id": "imp_b", "split": "train", "family": "impact", "v_mps": 5.5, "z_off_m": -0.009, "phi_deg": 0.0},
    {"id": "imp_c", "split": "train", "family": "impact", "v_mps": 7.5, "z_off_m": 0.004, "phi_deg": 20.0},
    {"id": "imp_d", "split": "train", "family": "impact", "v_mps": 6.0, "z_off_m": -0.010, "phi_deg": 0.0},
    {"id": "imp_e", "split": "validation", "family": "impact", "v_mps": 7.5, "z_off_m": -0.007, "phi_deg": 0.0},
    {"id": "imp_f", "split": "validation", "family": "impact", "v_mps": 5.0, "z_off_m": 0.010, "phi_deg": 40.0},
    {"id": "imp_g", "split": "final_test", "family": "impact", "v_mps": 8.0, "z_off_m": 0.003, "phi_deg": 0.0},
    {"id": "imp_h", "split": "final_test", "family": "impact", "v_mps": 5.5, "z_off_m": -0.012, "phi_deg": 15.0},
    {"id": "slip_a", "split": "train", "family": "slip", "dz": -12.6, "dx": 0.0, "dy": 0.0, "dvz": 0.0},
    {"id": "slip_b", "split": "train", "family": "slip", "dz": -13.8, "dx": 0.0, "dy": 0.0, "dvz": 0.008},
    {"id": "slip_c", "split": "train", "family": "slip", "dz": -15.2, "dx": 0.0, "dy": 0.0, "dvz": 0.0},
    {"id": "slip_d", "split": "train", "family": "slip", "dz": -14.0, "dx": 0.05, "dy": 0.0, "dvz": 0.0},
    {"id": "slip_e", "split": "validation", "family": "slip", "dz": -12.2, "dx": 0.0, "dy": 0.0, "dvz": -0.008},
    {"id": "slip_f", "split": "validation", "family": "slip", "dz": -15.6, "dx": 0.0, "dy": 0.0, "dvz": 0.0},
    {"id": "slip_g", "split": "final_test", "family": "slip", "dz": -13.0, "dx": 0.0, "dy": 0.08, "dvz": 0.0},
    {"id": "slip_h", "split": "final_test", "family": "slip", "dz": -14.9, "dx": 0.0, "dy": 0.0, "dvz": 0.012},
    {"id": "seat_a", "split": "train", "family": "mis_seat", "dz": -8.0, "dx": 6.0, "dy": 0.0, "dvz": 0.0},
    {"id": "seat_b", "split": "train", "family": "mis_seat", "dz": -9.0, "dx": 0.0, "dy": -6.0, "dvz": 0.0},
    {"id": "seat_c", "split": "validation", "family": "mis_seat", "dz": -7.5, "dx": -5.0, "dy": 4.0, "dvz": 0.0},
    {"id": "seat_d", "split": "final_test", "family": "mis_seat", "dz": -8.5, "dx": 0.0, "dy": 7.0, "dvz": 0.0},
    {"id": "air_a", "split": "train", "family": "airborne", "speed": 0.25, "sign": 1.0, "duration_s": 0.10},
    {"id": "air_b", "split": "train", "family": "airborne", "speed": 0.55, "sign": -1.0, "duration_s": 0.06},
    {"id": "air_c", "split": "validation", "family": "airborne", "speed": 0.35, "sign": 1.0, "duration_s": 0.12},
    {"id": "air_d", "split": "final_test", "family": "airborne", "speed": 0.50, "sign": -1.0, "duration_s": 0.08},
)

LEGACY = (
    "impact_zm6", "impact_z3", "impact_zp6", "impact_phi90",
    "air_lat_pos", "air_lat_neg", "air_fast_pos", "air_slow_neg",
    "nl_dz13.2", "nl_dz13.5", "nl_dz14.2", "nl_dz14.6", "nl_dvz_m", "nl_dvz_p",
    "h_dz14.00", "h_dz14.25", "h_dy", "h_dx",
)


def reconstruct(act, proto):
    dist = ((act[:, None, :] - proto[None]) ** 2).sum(-1)
    lab = dist.argmin(1)
    resid = act - proto[lab]
    clipped = np.clip(resid, -CAP, CAP)
    pred = np.clip(proto[lab] + clipped, -1.0, 1.0)
    err = np.linalg.norm(act - pred, axis=1)
    n_clip = int(np.sum(np.abs(resid) > CAP + 1e-8))
    return {
        "steps": int(len(act)),
        "max_l2": float(err.max()) if len(err) else 0.0,
        "median_l2": float(np.median(err)) if len(err) else 0.0,
        "p95_l2": float(np.quantile(err, 0.95)) if len(err) else 0.0,
        "clipped_components": n_clip,
        "tokens": sorted(set(int(x) for x in lab)),
        "token_counts": {str(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True))},
    }


def audit_representation(proto):
    files = [
        (SKILL / "demo_wrist_reseat.npz", "wrist"),
        (SKILL / "demo_gravity_inward.npz", "sliding"),
        (SKILL / "demo_recatch_open20.npz", "open_finger"),
        (DEMO / "demo_train_dz13.5.npz", "natural_loss"),
        (DEMO / "demo_nl_dz14.6.npz", "natural_loss"),
        (DEMO / "demo_nl_dvz_p.npz", "natural_loss"),
        (DEMO / "demo_impact_zp6.npz", "lateral_chase"),
        (TRAJ, "nominal_restoration"),
    ]
    rows = []
    for path, family in files:
        act = np.asarray(np.load(path)["act"], np.float32)
        rows.append({"name": path.stem, "family": family, **reconstruct(act, proto)})
    new_tokens = {
        "33_lift": [round(float(x), 4) for x in proto[33]],
        "34_second_release": [round(float(x), 4) for x in proto[34]],
        "32_lateral_chase": [round(float(x), 4) for x in proto[32]],
    }
    worst = max(row["max_l2"] for row in rows)
    return {"k": int(len(proto)), "cap": CAP, "worst_l2": worst, "frozen": bool(worst <= 0.05), "tokens": new_tokens, "rows": rows}


def entry_record(ref, sim):
    row = errors(sim, ref)
    pack = physical_pack(sim)
    grip = bool(sim.data.ctrl.size > 7 and float(sim.data.ctrl[7]) < -8.0)
    trivial = success.basin_ok(row, grip)
    return {
        "ep_mm": round(row["ep"] * 1000.0, 3),
        "eR_rad": round(row["eR"], 4),
        "vrel": round(row["vrel"], 4),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "bilateral": bool(pack["nL"] > 0 and pack["nR"] > 0),
        "z": round(float(pack["obj_z"]), 4),
        "trivial": bool(trivial),
    }


def enter_offset(env, base, spec):
    info = restore_to(env.sim, base, env.nominal, env.pads, env.pad_mu)
    drop_supports(env.sim)
    if not info.get("pose_ok", False):
        raise RuntimeError("offset restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(env.sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(env.sim)
    ref = capture_ref(env.sim, env.switch)
    scale.apply_spec(env.sim, spec)
    st = fresh_nominal_state()
    while float(env.sim.data.time) < T_TASK - 1e-9 and not env.switch.entered:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, TRACK, st)
        row = env.switch.measure(env.sim)
        env.switch.update_entry(env.sim)
        if row["z"] < Z_ABORT:
            break
    return ref, bool(env.switch.entered)


def enter_air(env, base, q_ref, spec):
    info = restore_to(env.sim, base, env.nominal, env.pads, env.pad_mu)
    drop_supports(env.sim)
    if not info.get("pose_ok", False):
        raise RuntimeError("air restore failed")
    env.switch = GeometrySwitch()
    env.switch.capture_if_ready(env.sim)
    if env.switch.p0 is None:
        env.switch.p0, env.switch.r0 = hand_pose(env.sim)
    ref = capture_ref(env.sim, env.switch)
    t0 = float(env.sim.data.time)
    open_at = t0 + float(spec["duration_s"])
    opened = False
    open_hold = 0
    st = fresh_nominal_state()
    while float(env.sim.data.time) < T_TASK - 1e-9 and not env.switch.entered:
        for _ in range(v3.N_SUB):
            t = float(env.sim.data.time)
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
                step_mapped(env.sim, action, AIRBORNE_V2, TRACK, q_ref)
                pack = physical_pack(env.sim)
                if opened and int(pack["nL"]) == 0 and int(pack["nR"]) == 0:
                    open_hold += 1
            else:
                nominal_substep(env.sim, TRACK, st)
        if env.switch.update_entry(env.sim):
            break
    return ref, bool(env.switch.entered)


def enter_episode(env, table, base, q_ref, spec):
    if spec["family"] == "impact":
        restore_supports(env.sim, table)
        ref, _ball = until_impact(env, spec)
        return ref, bool(ref is not None and env.switch.entered)
    if spec["family"] in ("slip", "mis_seat"):
        return enter_offset(env, base, spec)
    if spec["family"] == "airborne":
        return enter_air(env, base, q_ref, spec)
    raise KeyError(spec["family"])


def save_ic(spec, cap, ref):
    path = RAW / "ics" / f"{spec['id']}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"spec": spec, "capture": cap, "ref": ref}
    with path.open("wb") as handle:
        pickle.dump(blob, handle)
    return str(path)


def load_ic(episode_id):
    with (RAW / "ics" / f"{episode_id}.pkl").open("rb") as handle:
        return pickle.load(handle)


def mechanism(tokens, grips):
    names = []
    if any(t == 32 for t in tokens):
        names.append("lateral_chase")
    if any(t in (33, 34) for t in tokens):
        names.append("release_regrasp")
    if any(g < -0.5 for g in grips) and "release_regrasp" not in names:
        names.append("open_finger")
    if not names:
        names.append("closed_hold")
    return "+".join(names)


def rollout(env, ref, action_fn, cap, store_audit):
    basin_run = 0
    tokens, grips = [], []
    shaping = 0.0
    actions = []
    fell = False
    legacy = None
    ep_end = None
    eR_end = None
    for _ in range(cap):
        token, act = action_fn()
        tokens.append(int(token) if token is not None else -1)
        grips.append(float(act[6]))
        actions.append(np.asarray(act, np.float32).copy())
        row = errors(env.sim, ref)
        ep_end, eR_end = row["ep"], row["eR"]
        geo = {
            "nL": row["nL"], "nR": row["nR"], "e_pos": row["ep"], "e_rot": row["eR"],
            "vrel": row["vrel"], "z": row["z"],
        }
        shaping += float(reward_detail(geo)["shaping"])
        if legacy is None and float(env.sim.data.time) >= 12.0 - 1e-9:
            legacy = row["z"] >= Z_ABORT
        if row["z"] < Z_ABORT:
            fell = True
            break
        inside = success.basin_ok(row, float(act[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS:
            audit = success.apply_handoff(env.sim, ref)
            if store_audit["audit"] is None:
                store_audit["audit"] = audit
            hold = success.nominal_validate(env, ref)
            if hold["legacy_retained_to_12"] is not None:
                legacy = hold["legacy_retained_to_12"]
            label = success.label_of(False, True, True, hold)
            return {
                "label": label,
                "success": success.is_success(label),
                "basin": True,
                "handoff": True,
                "held_10": hold["held_10"],
                "held_12": hold["held_12"],
                "legacy_retained_to_12": legacy,
                "mechanism": mechanism(tokens, grips),
                "ep_end": None if hold["end"] is None else hold["end"]["ep"],
                "eR_end": None if hold["end"] is None else hold["end"]["eR"],
                "shaping": shaping,
                "return": shaping + (1.0 if success.is_success(label) else 0.0),
                "actions": actions,
                "handoff_audit": audit,
            }
    label = success.label_of(fell, False, False, None)
    return {
        "label": label,
        "success": False,
        "basin": False,
        "handoff": False,
        "held_10": False,
        "held_12": False,
        "legacy_retained_to_12": legacy,
        "mechanism": mechanism(tokens, grips),
        "ep_end": ep_end,
        "eR_end": eR_end,
        "shaping": shaping,
        "return": shaping,
        "actions": actions,
        "handoff_audit": None,
    }


def policy_fn(env, rng):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    with torch.no_grad():
        logits, res, act = env.model(
            torch.tensor(np.stack(env._obs_seq[-160:]), device=env.device),
            torch.tensor(np.stack(env._prev_seq[-160:]), device=env.device),
        )
    if rng is None:
        token = int(torch.argmax(logits[-1]).item())
        clipped = np.clip(act[-1].float().cpu().numpy(), -1.0, 1.0).astype(np.float32)
    else:
        probs = torch.softmax(logits[-1].float(), dim=-1).cpu().numpy()
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        token = int(rng.choice(len(probs), p=probs))
        resid = res[-1].float().cpu().numpy() + rng.normal(0.0, SIGMA, size=7).astype(np.float32)
        resid = np.clip(resid, -CAP, CAP)
        proto = env.model.proto[token].float().cpu().numpy()
        clipped = np.clip(proto + resid, -1.0, 1.0).astype(np.float32)
    v3.step_v3(env.sim, clipped)
    env._prev = clipped
    return token, clipped


def scripted_fn(env, commands, cursor):
    act = np.asarray(commands[min(cursor[0], len(commands) - 1)], np.float32)
    cursor[0] += 1
    v3.step_v3(env.sim, act)
    return None, act


def feedback_commands():
    state = {"t0": None, "freed": False}

    def one(sim, ref):
        if state["t0"] is None:
            state["t0"] = float(sim.data.time)
        dp, _p = pinch_error(sim, ref)
        age = float(sim.data.time) - state["t0"]
        pack = physical_pack(sim)
        if int(pack["nL"]) == 0 and int(pack["nR"]) == 0:
            state["freed"] = True
        if age < 0.20:
            return act_of(sim, ref, np.array([0.0, 0.0, -0.40]), 1.0, 0.0)
        err = float(np.linalg.norm(dp))
        if err > 0.004:
            z_cmd = 2.0 if dp[2] > 0.008 else float(np.clip(dp[2] / 0.04, -0.8, 0.8))
            step = np.array([
                float(np.clip(dp[0] / 0.04, -0.6, 0.6)),
                float(np.clip(dp[1] / 0.04, -0.6, 0.6)),
                z_cmd,
            ])
            use = -1.0 if not (state["freed"] and err < 0.008) else 1.0
            vel = step if use < 0 else np.clip(4.0 * dp, -0.1, 0.1)
            return act_of(sim, ref, vel, use, 0.0 if use < 0 else 1.0)
        return act_of(sim, ref, np.clip(3.0 * dp, -0.08, 0.08), 1.0, 1.0)

    return one


def privileged_probe(env, spec):
    blob = load_ic(spec["id"])
    ref = blob["ref"]
    probes = []
    audits = []
    law = feedback_commands()

    def run_feedback():
        restore_capture(env, blob["capture"])
        box = {"audit": None}

        def step():
            act = np.asarray(law(env.sim, ref), np.float32)
            v3.step_v3(env.sim, act)
            return None, act

        out = rollout(env, ref, step, 80, box)
        if box["audit"] is not None:
            audits.append(box["audit"])
        return out

    probes.append(("restoration_feedback", run_feedback()))
    if probes[-1][1]["success"]:
        return _priv_summary(probes)
    book = schedules()
    for name in ("recatch_open20", "wrist_reseat", "gravity_inward"):
        actions = [v3.v1_normalized_to_v3(np.asarray(a, np.float32)) for a in book[name]]
        restore_capture(env, blob["capture"])
        cursor = [0]

        def step(actions=actions, cursor=cursor):
            return scripted_fn(env, actions, cursor)

        box = {"audit": None}
        probes.append((name, rollout(env, ref, step, len(actions) + 5, box)))
        if box["audit"] is not None:
            audits.append(box["audit"])
        if probes[-1][1]["success"]:
            break
    summary = _priv_summary(probes)
    summary["handoff_audit"] = audits[0] if audits else None
    return summary


def _priv_summary(probes):
    slim = []
    verified = False
    winner = None
    for name, row in probes:
        slim.append({
            "probe": name,
            "label": row["label"],
            "success": row["success"],
            "basin": row["basin"],
            "held_10": row["held_10"],
            "held_12": row["held_12"],
        })
        if row["success"] and not verified:
            verified = True
            winner = name
    return {
        "class": "VERIFIED_RECOVERABLE" if verified else "SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE",
        "winner": winner,
        "probes": slim,
    }


def public_row(row):
    skip = {"actions", "handoff_audit"}
    return {k: v for k, v in row.items() if k not in skip}


def action_distance(group):
    seqs = [np.stack(row["actions"]) if row["actions"] else np.zeros((1, 7), np.float32) for row in group]
    dists = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            n = min(len(seqs[i]), len(seqs[j]))
            dists.append(float(np.linalg.norm(seqs[i][:n] - seqs[j][:n])))
    return 0.0 if not dists else float(np.mean(dists))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    proto = np.asarray(saved["proto"], np.float32)
    if proto.shape != (35, 7) or abs(float(saved["res_cap"]) - CAP) > 1e-9:
        raise RuntimeError(f"expected K=35 cap 0.25, got {proto.shape} {saved['res_cap']}")
    representation = audit_representation(proto)
    print("REPR", representation["worst_l2"], representation["frozen"], flush=True)
    model = build_policy(proto, device, CAP)
    model.load_state_dict(saved["model"])
    model.eval()
    env = make_training_env()
    table = supports_of(env.sim)
    env.model = model.to(env.device)
    env.model.eval()
    print("SEAT", flush=True)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    episodes = []
    handoff_audit = None
    for spec in CATALOG:
        ref, entered = enter_episode(env, table, base, q_ref, spec)
        row = {
            "id": spec["id"],
            "split": spec["split"],
            "family": spec["family"],
            "entered": bool(entered),
            "episode_identity": spec["id"],
        }
        if not entered or ref is None:
            row["class"] = "INVALID_IC"
            row["entry"] = None
            print(spec["id"], "no entry", flush=True)
        else:
            entry = entry_record(ref, env.sim)
            row["entry"] = entry
            if entry["trivial"]:
                row["class"] = "VERIFIED_ALREADY_NOMINAL/TRIVIAL"
                print(spec["id"], "trivial", entry["ep_mm"], flush=True)
            elif entry["z"] < Z_ABORT:
                row["class"] = "INVALID_IC"
                print(spec["id"], "fallen", flush=True)
            else:
                env.start_recovery()
                row["capture"] = save_ic(spec, take_capture(env), ref)
                row["class"] = "CANDIDATE"
                print(spec["id"], spec["split"], entry["ep_mm"], entry["eR_rad"], flush=True)
        episodes.append(row)

    candidates = [r for r in episodes if r["class"] == "CANDIDATE" and r["split"] != "final_test"]
    print("PROBE", len(candidates), flush=True)
    for row in candidates:
        spec = next(s for s in CATALOG if s["id"] == row["id"])
        priv = privileged_probe(env, spec)
        row["privileged"] = {k: v for k, v in priv.items() if k != "handoff_audit"}
        if priv.get("handoff_audit") is not None and handoff_audit is None:
            handoff_audit = priv["handoff_audit"]
        row["class"] = priv["class"]
        print(" priv", row["id"], priv["class"], priv["winner"], flush=True)

    print("BASELINE", flush=True)
    store = {"audit": None}
    baseline = []
    for row in candidates:
        blob = load_ic(row["id"])
        restore_capture(env, blob["capture"])
        out = rollout(env, blob["ref"], lambda: policy_fn(env, None), POLICY_CAP, store)
        packed = {"id": row["id"], "split": row["split"], "family": row["family"], **public_row(out)}
        baseline.append(packed)
        if store["audit"] is not None and handoff_audit is None:
            handoff_audit = store["audit"]
        print(" base", row["id"], out["label"], out["mechanism"], flush=True)

    print("GRPO", flush=True)
    groups = []
    train_ids = [r["id"] for r in candidates if r["split"] == "train"]
    det_check = None
    for episode_id in train_ids:
        blob = load_ic(episode_id)
        members = []
        for g in range(GROUP):
            restore_capture(env, blob["capture"])
            rng = np.random.default_rng(1000 + 17 * g + sum(ord(c) for c in episode_id))
            out = rollout(env, blob["ref"], lambda rng=rng: policy_fn(env, rng), 80, store)
            members.append(out)
        if det_check is None:
            restore_capture(env, blob["capture"])
            rng_a = np.random.default_rng(4242)
            first = rollout(env, blob["ref"], lambda: policy_fn(env, rng_a), 20, {"audit": None})
            restore_capture(env, blob["capture"])
            rng_b = np.random.default_rng(4242)
            second = rollout(env, blob["ref"], lambda: policy_fn(env, rng_b), 20, {"audit": None})
            restore_capture(env, blob["capture"])
            rng_c = np.random.default_rng(4243)
            third = rollout(env, blob["ref"], lambda: policy_fn(env, rng_c), 20, {"audit": None})
            same = np.stack(first["actions"]) if first["actions"] else np.zeros((1, 7))
            other = np.stack(second["actions"]) if second["actions"] else np.zeros((1, 7))
            differ = np.stack(third["actions"]) if third["actions"] else np.zeros((1, 7))
            n = min(len(same), len(other), len(differ))
            det_check = {
                "episode": episode_id,
                "repeat_l2": float(np.linalg.norm(same[:n] - other[:n])),
                "other_seed_l2": float(np.linalg.norm(same[:n] - differ[:n])),
            }
        returns = [m["return"] for m in members]
        successes = int(sum(m["success"] for m in members))
        span = float(max(returns) - min(returns)) if returns else 0.0
        mixed = 0 < successes < GROUP
        useful = bool(mixed or span >= 0.5)
        groups.append({
            "id": episode_id,
            "family": next(r["family"] for r in candidates if r["id"] == episode_id),
            "successes": successes,
            "failures": GROUP - successes,
            "return_span": span,
            "mixed": mixed,
            "useful_advantage": useful,
            "action_distance": action_distance(members),
            "labels": [m["label"] for m in members],
        })
        print(" group", episode_id, successes, round(span, 4), useful, flush=True)

    def counts(split):
        rows = [r for r in episodes if r["split"] == split]
        out = {}
        for fam in ("impact", "slip", "mis_seat", "airborne"):
            sub = [r for r in rows if r["family"] == fam]
            out[fam] = {
                "generated": len(sub),
                "candidate": sum(1 for r in sub if "capture" in r),
                "invalid": sum(1 for r in sub if r["class"] == "INVALID_IC"),
                "trivial": sum(1 for r in sub if str(r["class"]).startswith("VERIFIED_ALREADY")),
                "verified": sum(1 for r in sub if r["class"] == "VERIFIED_RECOVERABLE"),
                "search_failed": sum(1 for r in sub if r["class"] == "SEARCH_FAILED_NOT_PROVEN_IMPOSSIBLE"),
            }
            if split == "final_test":
                out[fam]["candidate"] = sum(r["class"] == "CANDIDATE" for r in sub)
                out[fam]["verified"] = None
                out[fam]["search_failed"] = None
        return out

    payload = {
        "checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "k": 35,
        "res_cap": CAP,
        "success_config": "config/recovery/recovery_success.yaml",
        "legacy_diagnostic_set": list(LEGACY),
        "representation": representation,
        "handoff_audit": handoff_audit,
        "episodes": [{k: v for k, v in r.items() if k != "capture"} | {"saved": "capture" in r} for r in episodes],
        "family_counts": {s: counts(s) for s in ("train", "validation", "final_test")},
        "baseline": baseline,
        "groups": groups,
        "determinism": det_check,
        "group_size": GROUP,
    }
    (RAW / "readiness.json").write_text(json.dumps(payload), encoding="utf-8")
    print("WROTE", RAW / "readiness.json", flush=True)


if __name__ == "__main__":
    main()
