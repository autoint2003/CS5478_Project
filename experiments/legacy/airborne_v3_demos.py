"""Historical 39-D natural-loss demo generator.

This is not the K48 trainer and not the formal data rebuild. It records the
old airborne demos and trains the 39-D checkpoint used by recovery_episode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.tracking import TRACK
from controllers.recovery.action_map import (
    AIRBORNE_V2,
    AIRBORNE_V3,
    V_Z_DOWN,
    V_Z_UP,
    hand_translation,
    map_action,
    normalize_action,
    v1_normalized_to_v3,
    v2_normalized_to_v3,
)
from envs.observations.airborne_obs import (
    OBS_DIM_AIR,
    OBS_NAMES_AIR,
    SCALE_PREL,
    SCALE_ROT,
    SCALE_VREL,
    SCALE_WREL,
    observe_airborne,
    relative_state,
)
from envs.simulation.grasp_quality import body_twist
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.physical_recovery import physical_pack
from envs.ballistic_impact import assert_noslip, world
from data_generation.airborne_recatch_dataset import (
    begin,
    capture_nominal,
    capture_pads,
    carry_then_release,
    drop_supports,
    grasp_settled,
    hold_step,
    limits_of,
    restore_to,
    run_chase,
)
from data_generation.jsonable import jsonable
from controllers.recovery.skill_schedules import schedules
from training.recovery_execution import (
    T_TASK,
    continue_long,
    physical_of,
    restore_dyn,
    run_impact,
    stamp,
)
from envs.simulation.scene_state import park_impact_ball
from data_generation.action_kmeans import kmeans
from envs.initialization.grasp_seat import target_hand
from models.airborne_v3 import (
    DT_POLICY,
    HOLD,
    K_INT,
    N_SUB,
    assert_map,
    build_model,
    intercept_action,
    load_airborne_policy,
    step_v3,
)

OUT = ROOT / "results" / "diagnostics" / "raw" / "natural_loss_recatch_transformer_v3"
EPOCHS = 120
MAX_LEN = 96


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cuda_info():
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": "cpu",
        "gpu": None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["device"] = "cuda"
    print(
        "CUDA", info["torch"], "avail", info["cuda_available"],
        "gpu", info["gpu"], "cuda", info["cuda_version"], "cudnn", info["cudnn"],
        "device", info["device"],
        flush=True,
    )
    return info


def obs_match(sim, hist):
    obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
    rel = relative_state(sim.model, sim.data, sim.ids)
    raw = np.array([
        *rel["p_rel_h"] / SCALE_PREL, *rel["v_rel_h"] / SCALE_VREL,
        *rel["rot_rel"] / SCALE_ROT, *rel["w_rel_h"] / SCALE_WREL,
    ], np.float32)
    clipped = np.clip(raw, -1.0, 1.0)
    extra = obs[27:]
    return obs, {
        "finite": bool(np.isfinite(obs).all()) and obs.shape[0] == OBS_DIM_AIR,
        "match": float(np.max(np.abs(extra - clipped))),
        "absmax": float(np.max(np.abs(extra))),
        "saturated": [OBS_NAMES_AIR[27 + i] for i in range(extra.shape[0]) if abs(float(extra[i])) > 0.999],
        "raw_absmax": float(np.max(np.abs(raw))),
    }


def natural_rollout(sim, pinch):
    hist = ObservableObsState(DT_POLICY)
    phase = "slip"
    free = False
    both_hold = 0.0
    re_t = None
    hold_t = None
    prev_vz = None
    frames = []
    t0 = float(sim.data.time)
    o0 = physical_pack(sim)
    if int(o0["nL"]) == 0 or int(o0["nR"]) == 0:
        return frames, {"ok": False, "why": "already separated"}
    for _ in range(int(8.5 / DT_POLICY)):
        o = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        rh = rel["p_rel_h"]
        excess = float(rh[2] - pinch[2])
        bilateral = int(o["nL"]) > 0 and int(o["nR"]) > 0
        if not bilateral:
            free = True
        if phase == "slip" and not bilateral and len(frames) >= 3:
            phase = "chase"
        aper = float(o["aperture"])
        vrel = rel["v_obj_w"] - rel["v_hand_w"]
        slow = abs(float(vrel[2])) < 0.15 and float(np.linalg.norm(vrel[:2])) < 0.20
        near = (
            abs(excess) < 0.012
            and abs(float(rh[0] - pinch[0])) < 0.012
            and abs(float(rh[1] - pinch[1])) < 0.012
        )
        if phase == "chase" and aper > 0.036 and near and slow:
            phase = "close"
        if phase == "close":
            if bilateral and free:
                if re_t is None:
                    re_t = float(sim.data.time) - t0
                both_hold += DT_POLICY
                if both_hold >= 0.04 and hold_t is None:
                    hold_t = float(sim.data.time) - t0
                    phase = "hold"
            else:
                both_hold = 0.0
        if phase == "slip":
            action = HOLD.copy()
            v_world = map_action(action, sim.fsm.r_des, sim.data.xmat[sim.ids.hand_body].reshape(3, 3), AIRBORNE_V3)["v_world"]
        else:
            action, v_world = intercept_action(sim, pinch, phase)
        obs, audit = obs_match(sim, hist)
        vh = float(body_twist(sim.model, sim.data, sim.ids.hand_body)[0][2])
        az = 0.0 if prev_vz is None else (vh - prev_vz) / DT_POLICY
        prev_vz = vh
        lim = np.array(sim.model.actuator_ctrlrange[:7, 1], float)
        tau = np.array(sim.data.ctrl[:7], float)
        frames.append({
            "snap": stamp(sim),
            "obs": obs,
            "act": np.asarray(action, np.float32).copy(),
            "phase": phase,
            "excess": excess,
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "aper": aper,
            "cmd_vz": float(v_world[2]),
            "vh_z": vh,
            "vo_z": float(rel["v_obj_w"][2]),
            "vrel_z": float(vrel[2]),
            "az": float(az),
            "tau_peak": float(np.max(np.abs(tau))),
            "sat": bool(np.any(np.abs(tau) >= 0.95 * lim)),
            "audit": audit,
            "t": float(sim.data.time) - t0,
        })
        step_v3(sim, action, stop_on_loss=(phase == "slip"))
        if hold_t is not None and frames[-1]["t"] >= hold_t + 1.02:
            break
        z = float(sim.data.xpos[sim.ids.object_body][2])
        if phase != "slip" and re_t is None and z < -0.15:
            break
        if phase != "slip" and re_t is None and lost_age(frames) > 0.70 and excess > 0.05:
            if len(frames) > 8 and frames[-1]["excess"] > frames[-8]["excess"]:
                break
    end = frames[-1] if frames else None
    lost = next((f for f in frames if f["phase"] != "slip"), None)
    window = []
    if re_t is not None:
        window = [f for f in frames if re_t - 1e-9 <= f["t"] <= re_t + 1.0]
    both_frac = None
    if window:
        both_frac = float(np.mean([(f["nL"] > 0 and f["nR"] > 0) for f in window]))
    covered = bool(window) and window[-1]["t"] >= re_t + 0.98
    ok = bool(
        lost is not None
        and re_t is not None
        and covered
        and both_frac is not None
        and both_frac >= 0.95
        and end is not None
        and end["nL"] > 0
        and end["nR"] > 0
        and abs(end["excess"]) < 0.02
    )
    why = "ok" if ok else "no loss" if lost is None else "no 1s hold"
    summary = {
        "ok": ok,
        "why": why,
        "t_loss": None if lost is None else round(float(lost["t"]), 3),
        "gap_loss_mm": None if lost is None else round(float(lost["excess"]) * 1e3, 2),
        "cmd_after_loss": None if lost is None else round(float(lost["cmd_vz"]), 3),
        "re_t": None if re_t is None else round(float(re_t), 3),
        "both_frac": None if both_frac is None else round(both_frac, 3),
        "end_n": None if end is None else [end["nL"], end["nR"]],
        "end_excess_mm": None if end is None else round(float(end["excess"]) * 1e3, 2),
        "min_excess_mm": None if not frames else round(float(min(f["excess"] for f in frames)) * 1e3, 2),
        "n_frames": len(frames),
    }
    return frames, summary


def lost_age(frames):
    loss = next((f for f in frames if f["phase"] != "slip"), None)
    if loss is None or not frames:
        return 0.0
    return float(frames[-1]["t"] - loss["t"])


def windows_of(frames, summary, tag, split):
    if not summary["ok"]:
        return []
    loss_i = next(i for i, f in enumerate(frames) if f["phase"] != "slip")
    out = []
    for look, name in ((50, "w1.0"), (15, "w0.3")):
        start = max(0, loss_i - look)
        if frames[start]["nL"] == 0 or frames[start]["nR"] == 0:
            continue
        sl = frames[start:]
        if len(sl) < 8:
            continue
        prev = np.zeros((len(sl), 7), np.float32)
        prev[1:] = np.stack([f["act"] for f in sl[:-1]])
        out.append({
            "family": f"nl_{tag}_{name}",
            "split": split,
            "domain": "natural",
            "tag": tag,
            "obs": np.stack([f["obs"] for f in sl]).astype(np.float32),
            "act": np.stack([f["act"] for f in sl]).astype(np.float32),
            "prev": prev,
            "snap": sl[0]["snap"],
            "pinch_gap0_mm": round(float(sl[0]["excess"]) * 1e3, 2),
            "t_loss": summary["t_loss"],
            "gap_loss_mm": summary["gap_loss_mm"],
            "cmd_after_loss": summary["cmd_after_loss"],
            "min_excess_mm": summary["min_excess_mm"],
            "frames_light": [{k: f[k] for k in ("t", "phase", "excess", "cmd_vz", "vh_z", "vrel_z", "az", "nL", "nR", "sat", "aper")} for f in sl],
        })
    return out


def phase_audit(frames):
    loss_i = next((i for i, f in enumerate(frames) if f["phase"] != "slip"), None)
    if loss_i is None:
        return None
    contact = next((f for f in frames if f["nL"] > 0 and f["nR"] > 0 and f["excess"] > 0.008), frames[0])
    chase = next((f for f in frames[loss_i:] if f["phase"] == "chase" and f["t"] > frames[loss_i]["t"] + 0.04), frames[loss_i])
    re = next((f for f in frames if f["phase"] in ("close", "hold") and f["nL"] > 0 and f["nR"] > 0), None)
    picked = {"contact": contact, "natural_loss": frames[loss_i], "chase": chase}
    if re is not None:
        picked["recontact"] = re
    return {
        name: {
            "finite": item["audit"]["finite"],
            "match": round(item["audit"]["match"], 6),
            "absmax": round(item["audit"]["absmax"], 3),
            "saturated": item["audit"]["saturated"],
            "raw_absmax": round(item["audit"]["raw_absmax"], 3),
            "n": [item["nL"], item["nR"]],
            "phase": item["phase"],
        }
        for name, item in picked.items()
    }


def run_natural(sim, base, nominal, pads, pad_mu, pinch, spec, split):
    info = restore_to(sim, base, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        return [], {"tag": spec["tag"], "ok": False, "why": "restore", "split": split}, []
    from envs.initialization.grasp_seat import apply_offset

    apply_offset(sim, {"dx": spec.get("dx", 0.0), "dy": 0.0, "dz": spec["dz"], "pitch": 0.0})
    if spec.get("dq2"):
        sim.data.qpos[sim.ids.arm_jnt[1]] += float(spec["dq2"])
    if spec.get("dvz"):
        dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
        sim.data.qvel[dadr + 2] += float(spec["dvz"])
    mujoco.mj_forward(sim.model, sim.data)
    frames, summary = natural_rollout(sim, pinch)
    summary["tag"] = spec["tag"]
    summary["split"] = split
    print(
        "NL", spec["tag"], split, summary["why"],
        "loss", summary.get("t_loss"), "gap", summary.get("gap_loss_mm"),
        "cmd", summary.get("cmd_after_loss"), "min", summary.get("min_excess_mm"),
        "end", summary.get("end_n"),
        flush=True,
    )
    return windows_of(frames, summary, spec["tag"], split), summary, frames


def record_local(sim, name, actions, snap, nominal, pad_ids, pad_mu, mass):
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, mass, None)
    park_impact_ball(sim)
    if not info["pose_ok"]:
        return None
    hist = ObservableObsState(DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_l, act_l, prev_l = [], [], []
    for act in actions:
        obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
        a3 = v1_normalized_to_v3(act)
        obs_l.append(obs)
        act_l.append(a3)
        prev_l.append(prev.copy())
        step_v3(sim, a3)
        prev = a3
        if float(physical_pack(sim)["obj_z"]) < 0.44:
            return None
    rec = continue_long(sim, TRACK, T_TASK, name)
    y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
    if y != 1:
        print("local fail", name, mass, kind, flush=True)
        return None
    return {
        "family": name,
        "mass": mass,
        "domain": "local",
        "obs": np.stack(obs_l).astype(np.float32),
        "act": np.stack(act_l).astype(np.float32),
        "prev": np.stack(prev_l).astype(np.float32),
        "snap": snap,
        "outcome": "RETAINED_TO_12",
        "end_n": [int(rec["end"]["nL"]), int(rec["end"]["nR"])],
    }


def record_air(sim, spec, snap, nominal, pads, pad_mu, rh0, split):
    """Recompute the v2 oracle under the v3 rebase stepper. Do not replay an integrated trace."""
    from data_generation.airborne_recatch_dataset import oracle_action

    info = restore_to(sim, snap, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        return None
    hist = ObservableObsState(DT_POLICY)
    o0 = physical_pack(sim)
    phase = "chase" if int(o0["nL"]) == 0 and int(o0["nR"]) == 0 and float(o0["aperture"]) > 0.028 else "open"
    free_run = DT_POLICY if phase == "chase" else 0.0
    free_t = None
    re_t = None
    hold_t = None
    both_hold = 0.0
    t0 = float(sim.data.time)
    prev = np.zeros(7, np.float32)
    obs_l, act_l, prev_l = [], [], []
    for _ in range(int(2.6 / DT_POLICY)):
        o = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        both = int(o["nL"]) > 0 and int(o["nR"]) > 0
        none = int(o["nL"]) == 0 and int(o["nR"]) == 0
        aper = float(o["aperture"])
        rh = rel["p_rel_h"]
        vrel = rel["v_obj_w"] - rel["v_hand_w"]
        t = float(sim.data.time) - t0
        if none:
            free_run += DT_POLICY
            if free_t is None and free_run >= 0.02:
                free_t = t
        elif free_t is None:
            free_run = 0.0
        aligned = (
            abs(float(rh[0])) < 0.012
            and abs(float(rh[1])) < 0.012
            and abs(float(rh[2]) - float(rh0[2])) < 0.012
        )
        slow = abs(float(vrel[2])) < 0.12 and float(np.linalg.norm(vrel[:2])) < 0.15
        if phase == "open" and aper >= 0.034:
            phase = "chase"
        if phase == "chase" and free_t is not None and aligned and slow and aper > 0.038:
            phase = "close"
        if phase in ("close", "hold") and free_t is not None and re_t is None and both:
            re_t = t
        if phase == "close" and both:
            both_hold += DT_POLICY
            if both_hold >= 0.04 and hold_t is None:
                hold_t = t
                phase = "hold"
        elif phase == "close":
            both_hold = 0.0
        raw, *_rest = oracle_action(sim, sim.fsm.r_des, rh0, spec, phase, AIRBORNE_V2)
        act = v2_normalized_to_v3(raw)
        obs_l.append(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist))
        act_l.append(act)
        prev_l.append(prev.copy())
        step_v3(sim, act)
        prev = act
        if hold_t is not None and t >= hold_t + 1.0:
            break
        if float(sim.data.xpos[sim.ids.object_body][2]) < -0.15 and re_t is None and t > 0.4:
            break
    end = physical_pack(sim)
    ok = re_t is not None and hold_t is not None and int(end["nL"]) > 0 and int(end["nR"]) > 0
    if not ok:
        print("air fail", spec.get("tag"), "end", int(end["nL"]), int(end["nR"]), "re", re_t, flush=True)
        return None
    return {
        "family": "air_" + spec["tag"],
        "split": split,
        "domain": "air",
        "obs": np.stack(obs_l).astype(np.float32),
        "act": np.stack(act_l).astype(np.float32),
        "prev": np.stack(prev_l).astype(np.float32),
        "snap": snap,
        "end_n": [int(end["nL"]), int(end["nR"])],
        "air_s": None if free_t is None or re_t is None else round(float(re_t - free_t), 3),
        "pinch": np.array(rh0, float),
    }


def phys_of(actions):
    v = np.stack([hand_translation(a, AIRBORNE_V3) for a in actions])
    w = np.asarray(actions[:, 3:6], float) * 4.0
    g = np.asarray(actions[:, 6:7], float)
    return np.concatenate([v, w, g], axis=1).astype(np.float32)


def token_fit(actions, families, k):
    phys = phys_of(actions)
    # Local translation is weighted above large vz. Grip and wrist stay separate.
    weight = np.array([25.0, 25.0, 2.0, 2.0, 2.0, 2.0, 3.0], np.float32)
    feat = phys * weight
    _inertia, _centers, labels = kmeans(feat, k, 0, restarts=6, iters=30)
    proto = np.zeros((k, 7), np.float32)
    occ = []
    for j in range(k):
        m = labels == j
        occ.append(int(m.sum()))
        if m.any():
            proto[j] = actions[m].mean(0)
    recon = proto[labels]
    err = np.linalg.norm(actions - recon, axis=1)
    v_err = np.abs(phys_of(actions)[:, 0:3] - phys_of(recon)[:, 0:3])

    def mask_err(mask, cols):
        if not np.any(mask):
            return None
        return round(float(np.mean(v_err[mask][:, cols])), 4)

    local = np.array([f.startswith("wrist") or f.startswith("recatch") or f.startswith("gravity") for f in families])
    high = phys[:, 2] > 1.2
    lateral = np.array(["lat" in f for f in families])
    opened = actions[:, 6] < -0.3
    closed = actions[:, 6] > 0.5
    wrist = (np.abs(actions[:, 3]) > 0.4) | (np.abs(actions[:, 4]) > 0.4)
    hold = (np.abs(actions[:, 2]) < 0.05) & (actions[:, 6] > 0.5)

    def overlap(a, b):
        if not np.any(a) or not np.any(b):
            return []
        return sorted(int(x) for x in (set(labels[a].tolist()) & set(labels[b].tolist())))

    return {
        "k": k,
        "occupancy": occ,
        "empty": int(sum(n == 0 for n in occ)),
        "recon_l2_median": round(float(np.median(err)), 4),
        "local_v_mae_mps": mask_err(local, [0, 1, 2]),
        "high_vz_mae_mps": mask_err(high, [2]),
        "lateral_v_mae_mps": mask_err(lateral, [0, 1, 2]),
        "open_grip_mae": round(float(np.mean(np.abs(actions[opened, 6] - recon[opened, 6]))), 4) if opened.any() else None,
        "close_grip_mae": round(float(np.mean(np.abs(actions[closed, 6] - recon[closed, 6]))), 4) if closed.any() else None,
        "wrist_mae": round(float(np.mean(np.abs(actions[wrist, 3:6] - recon[wrist, 3:6]))), 4) if wrist.any() else None,
        "open_close_overlap": overlap(opened, closed),
        "high_vz_hold_overlap": overlap(high, hold),
        "fast_wrist_overlap": overlap(high, wrist),
        "proto": [[round(float(x), 3) for x in p] for p in proto],
        "labels": labels,
        "centers": proto,
    }


def train_model(trajs, proto, device, use_amp):
    import torch

    model = build_model(proto, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    rng = np.random.default_rng(0)
    bank = []
    for tr in trajs:
        n = len(tr["act"])
        bank.append((
            torch.tensor(tr["obs"]),
            torch.tensor(tr["prev"]),
            torch.tensor(tr["act"]),
        ))
    OUT.mkdir(parents=True, exist_ok=True)
    last = None
    amp_on = use_amp and device.type == "cuda"
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        order = np.repeat(np.arange(len(bank)), 8)
        rng.shuffle(order)
        running = 0.0
        nbat = 0
        batch = []
        def flush():
            nonlocal running, nbat, amp_on
            if not batch:
                return
            T = min(MAX_LEN, max(b[0].shape[0] for b in batch))
            B = len(batch)
            obs = torch.zeros(B, T, OBS_DIM_AIR)
            prev = torch.zeros(B, T, 7)
            act = torch.zeros(B, T, 7)
            valid = torch.zeros(B, T, dtype=torch.bool)
            for i, (o, p, a) in enumerate(batch):
                n = min(T, o.shape[0])
                obs[i, :n] = o[:n]
                prev[i, :n] = p[:n]
                act[i, :n] = a[:n]
                valid[i, :n] = True
            obs = obs.pin_memory().to(device, non_blocking=True)
            prev = prev.pin_memory().to(device, non_blocking=True)
            act = act.pin_memory().to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            pad = ~valid
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits, res, _pred = model(obs, prev, pad)
                dist = ((act.unsqueeze(2) - model.proto.view(1, 1, -1, 7)) ** 2).sum(-1)
                lab = dist.argmin(-1)
                ce_tok = torch.nn.functional.cross_entropy(
                    logits[valid].float(), lab[valid], reduction="none")
                weight = torch.ones_like(lab, dtype=torch.float32)
                weight = torch.where(act[:, :, 2] > 0.4, weight * 4.0, weight)
                weight = torch.where(act[:, :, 6] < -0.2, weight * 4.0, weight)
                weight = torch.where(act[:, :, 4].abs() > 0.4, weight * 4.0, weight)
                w = weight[valid]
                ce = (ce_tok * w).sum() / w.sum().clamp_min(1.0)
                target = act - model.proto[lab]
                mse = ((res - target)[valid] ** 2).mean()
                loss = ce + mse
            if not torch.isfinite(loss):
                amp_on = False
                print("AMP off, non-finite loss", flush=True)
                opt.zero_grad(set_to_none=True)
                batch.clear()
                return
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running += float(loss.detach())
            nbat += 1
            batch.clear()

        for idx in order:
            o, p, a = bank[int(idx)]
            n = o.shape[0]
            s0 = int(rng.integers(0, n))
            batch.append((o[s0:], p[s0:], a[s0:]))
            if len(batch) == 16:
                flush()
        flush()
        if nbat == 0:
            continue
        last = running / nbat
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            print("epoch", epoch + 1, "loss", round(last, 5), "amp", amp_on, flush=True)
            torch.save(
                {"epoch": epoch + 1, "loss": last, "proto": proto, "model": model.state_dict(), "amp": amp_on},
                OUT / f"ckpt_{epoch + 1:03d}.pt",
            )
    elapsed = time.perf_counter() - t0
    model.eval()
    return model, last, elapsed, amp_on


def teacher_forced(model, trajs, device):
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for tr in trajs:
            obs = torch.tensor(tr["obs"], device=device)
            prev = torch.tensor(tr["prev"], device=device)
            _logits, _res, pred = model(obs, prev)
            pred = pred.float().cpu().numpy()
            err = np.linalg.norm(pred - tr["act"], axis=1)
            hi = np.where(tr["act"][:, 2] > 0.5)[0]
            row = {
                "family": tr["family"],
                "split": tr.get("split"),
                "mae": round(float(err.mean()), 4),
                "vz_mae": round(float(np.mean(np.abs(pred[:, 2] - tr["act"][:, 2]))), 4),
                "grip_mae": round(float(np.mean(np.abs(pred[:, 6] - tr["act"][:, 6]))), 4),
            }
            if len(hi):
                row["a2_hat_chase"] = round(float(pred[hi[0], 2]), 3)
                row["a2_true_chase"] = round(float(tr["act"][hi[0], 2]), 3)
            rows.append(row)
    return rows


def closed_loop(sim, model, snap, nominal, pads, pad_mu, mass, n_steps, kind, pinch=None):
    import torch

    device = next(model.parameters()).device
    if kind == "local":
        info = restore_dyn(sim, snap, nominal, pads, pad_mu, mass, None)
        park_impact_ball(sim)
        if not info["pose_ok"]:
            return {"pose_ok": False}
    else:
        info = restore_to(sim, snap, nominal, pads, pad_mu)
        drop_supports(sim)
        if not info.get("pose_ok", False):
            return {"pose_ok": False}
    hist = ObservableObsState(DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    rows = []
    prev_vz = None
    model.eval()
    with torch.no_grad():
        for _k in range(n_steps):
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(obs)
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            cmd = step_v3(sim, act)
            o = physical_pack(sim)
            rel = relative_state(sim.model, sim.data, sim.ids)
            excess = None if pinch is None else float(rel["p_rel_h"][2] - pinch[2])
            vh = float(rel["v_hand_w"][2])
            az = 0.0 if prev_vz is None else (vh - prev_vz) / DT_POLICY
            prev_vz = vh
            lim = np.array(sim.model.actuator_ctrlrange[:7, 1], float)
            tau = np.array(sim.data.ctrl[:7], float)
            rows.append({
                "act": act,
                "cmd_vz": float(cmd["v_world"][2]),
                "vh_z": vh,
                "az": float(az),
                "vrel_z": float(rel["v_obj_w"][2] - vh),
                "excess": excess,
                "nL": int(o["nL"]),
                "nR": int(o["nR"]),
                "aper": float(o["aperture"]),
                "sat": bool(np.any(np.abs(tau) >= 0.95 * lim)),
                "tau": float(np.max(np.abs(tau))),
                "z": float(o["obj_z"]),
            })
            prev = act
            if kind == "natural" and excess is not None and excess > 0.12 and _k > 30 and rows[-1]["nL"] == 0 and rows[-1]["nR"] == 0:
                break
            if kind == "local" and float(o["obj_z"]) < 0.44:
                break
    out = {"pose_ok": True, "steps": len(rows)}
    if kind == "local" and rows and rows[-1]["z"] >= 0.44:
        rec = continue_long(sim, TRACK, T_TASK, "policy")
        y, kind_y = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        out["t12"] = "RETAINED_TO_12" if y == 1 else kind_y
        out["t12_n"] = [int(rec["end"]["nL"]), int(rec["end"]["nR"])]
    if pinch is not None and rows:
        loss_i = next((i for i, r in enumerate(rows) if r["nL"] == 0 and r["nR"] == 0), None)
        re_i = None
        if loss_i is not None:
            re_i = next((i for i in range(loss_i, len(rows)) if rows[i]["nL"] > 0 and rows[i]["nR"] > 0), None)
        held = False
        if re_i is not None and re_i + 50 <= len(rows):
            tail = rows[re_i:re_i + 50]
            held = float(np.mean([(r["nL"] > 0 and r["nR"] > 0) for r in tail])) >= 0.95 and tail[-1]["nL"] > 0 and tail[-1]["nR"] > 0
        i_min = int(np.argmin([r["excess"] for r in rows]))
        chase = [r for r in rows if r["act"][2] > 0.4]
        out.update({
            "loss_step": loss_i,
            "gap_loss_mm": None if loss_i is None else round(float(rows[loss_i]["excess"]) * 1e3, 2),
            "cmd_after_loss": None if loss_i is None else round(float(rows[min(loss_i, len(rows) - 1)]["cmd_vz"]), 3),
            "a2_after_loss": None if loss_i is None else round(float(rows[loss_i]["act"][2]), 3),
            "min_gap_mm": round(float(rows[i_min]["excess"]) * 1e3, 2),
            "vrel_at_min": round(float(rows[i_min]["vrel_z"]), 3),
            "recontact": re_i is not None,
            "reclose": any(r["act"][6] > 0.5 and r["nL"] > 0 for r in rows[re_i:]) if re_i is not None else False,
            "held_1s": bool(held),
            "peak_cmd_vz": round(float(min(r["cmd_vz"] for r in rows)), 3),
            "peak_ach_vz": round(float(min(r["vh_z"] for r in rows[:i_min + 1])), 3),
            "peak_az": round(float(min(r["az"] for r in rows[:i_min + 1])), 2),
            "sat_steps": int(sum(r["sat"] for r in rows)),
            "peak_tau": round(float(max(r["tau"] for r in rows)), 1),
            "cmd_at_min_gap": round(float(rows[i_min]["cmd_vz"]), 3),
            "a2_at_min_gap": round(float(rows[i_min]["act"][2]), 3),
            "end_n": [rows[-1]["nL"], rows[-1]["nR"]],
            "opened": any(r["act"][6] < -0.3 for r in rows),
            "max_a2": round(float(max(r["act"][2] for r in rows)), 3),
            "chase_steps": len(chase),
        })
    elif rows:
        out["end_n"] = [rows[-1]["nL"], rows[-1]["nR"]]
        out["max_abs_v"] = [round(float(x), 3) for x in np.max(np.abs(np.stack([r["act"][:3] for r in rows])), axis=0)]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = cuda_info()
    mapping = assert_map()
    print("MAP", mapping["v2_full_down_rewritten_a2"], flush=True)

    sim, cfg_gains, nominal, pad_ids, pad_mu, pads_l, pads_r = world()
    assert_noslip(sim)
    # Natural-loss scene: seat under the tracking gains used by the cap study.
    base_seat = grasp_settled(sim, TRACK)
    if base_seat is None or not begin(sim, base_seat, nominal, pad_ids, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    probe = map_action([0, 0, 1, 0, 0, 0, 1], sim.fsm.r_des, rh, AIRBORNE_V3)
    print(
        "SEAT pinch_mm", [round(float(x) * 1e3, 2) for x in pinch],
        "hand_z_world", round(float(rh[2, 2]), 3),
        "full_down_world_vz", round(float(probe["v_world"][2]), 3),
        flush=True,
    )
    drop_supports(sim)
    base = stamp(sim)
    q_ref, _m = limits_of(sim)

    train_specs = [
        {"tag": "dz13.5", "dz": -13.5},
        {"tag": "dz13.3", "dz": -13.3},
        {"tag": "dz13.8", "dz": -13.8},
        {"tag": "dz14.2", "dz": -14.2},
        {"tag": "dx1", "dz": -13.5, "dx": 1.0},
        {"tag": "dxm1", "dz": -13.5, "dx": -1.0},
        {"tag": "q2", "dz": -13.5, "dq2": 0.002},
        {"tag": "dvz", "dz": -13.5, "dvz": -0.01},
    ]
    hold_specs = [
        {"tag": "dz13.6", "dz": -13.6},
        {"tag": "dz14.0", "dz": -14.0},
        {"tag": "dx1p", "dz": -13.5, "dx": 1.5},
        {"tag": "q2m", "dz": -13.5, "dq2": -0.002},
    ]
    natural, failed, audit = [], [], None
    for spec, split in [(s, "train") for s in train_specs] + [(s, "hold") for s in hold_specs]:
        wins, summary, frames = run_natural(sim, base, nominal, pad_ids, pad_mu, pinch, spec, split)
        if summary["ok"] and audit is None:
            audit = phase_audit(frames)
            print("OBS", audit, flush=True)
        if summary["ok"]:
            natural.extend(wins)
        else:
            failed.append({k: v for k, v in summary.items()})

    print("NL traj", len(natural), "fail", len(failed), flush=True)
    (OUT / "natural_status.json").write_text(
        json.dumps(jsonable({"ok": [t["family"] for t in natural], "failed": failed, "obs": audit, "mapping": mapping}), indent=2),
        encoding="utf-8",
    )

    air_specs = [
        ({"tag": "vertical", "mode": "above", "lead": 0.0, "k": 4.0, "rz": -0.012, "vrel": 0.45}, "train"),
        ({"tag": "lat_pos", "mode": "below", "lead": 0.03, "k": 5.0, "rz": 0.0, "vrel": 0.55, "carry": (0.30, 1.0)}, "train"),
        ({"tag": "lat_neg", "mode": "below", "lead": 0.03, "k": 5.0, "rz": 0.0, "vrel": 0.55, "carry": (0.30, -1.0)}, "hold"),
    ]
    airborne = []
    for spec, split in air_specs:
        spec = dict(spec)
        use_snap, use_rh = base, pinch
        if "carry" in spec:
            speed, sign = spec.pop("carry")
            made = carry_then_release(sim, base, nominal, pad_ids, pad_mu, q_ref, speed, sign, 0.08)
            if made is None:
                print("carry fail", spec["tag"], flush=True)
                continue
            use_snap = made["snap"]
            use_rh = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
        tr = record_air(sim, spec, use_snap, nominal, pad_ids, pad_mu, use_rh, split)
        if tr is None:
            continue
        tr["pinch"] = use_rh
        print("AIR", tr["family"], split, "steps", len(tr["act"]), "end", tr["end_n"], flush=True)
        airborne.append(tr)

    # Local skills on a fresh seat of the impact task. Table stays enabled.
    sim_l, gains_l, nominal_l, pads_l_ids, pad_mu_l, pads_left, pads_right = world()
    assert_noslip(sim_l)
    snaps = {}
    for name, z in (("zm6", -0.006), ("zp6", 0.006)):
        impact = run_impact(sim_l, gains_l, 6.5, z, 0.0, pads_left, pads_right, nominal_l, None, None)
        snaps[name] = impact["snap"]
        print("impact", name, flush=True)
    sched = schedules()
    local_jobs = [
        ("wrist_reseat", "zm6", None, "train"),
        ("recatch_open20", "zm6", None, "train"),
        ("gravity_inward", "zm6", None, "train"),
        ("recatch_open20", "zp6", None, "hold"),
        ("gravity_inward", "zp6", None, "hold"),
    ]
    local = []
    for name, ic, mass, split in local_jobs:
        tr = record_local(sim_l, name, sched[name], snaps[ic], nominal_l, pads_l_ids, pad_mu_l, mass)
        if tr is None:
            print("SKIP", name, ic, flush=True)
            continue
        tr["split"] = split
        tr["ic"] = ic
        print("LOCAL", name, ic, split, "steps", len(tr["act"]), flush=True)
        local.append(tr)

    train_set = [t for t in natural + airborne + local if t["split"] == "train"]
    hold_set = [t for t in natural + airborne + local if t["split"] == "hold"]
    print("TRAIN", len(train_set), "HOLD", len(hold_set), "nl", sum(t["domain"] == "natural" for t in train_set), flush=True)
    if not any(t["domain"] == "natural" for t in train_set):
        blob = {"mapping": mapping, "cuda": info, "failed": failed, "obs": audit, "note": "no successful natural-loss demo"}
        (OUT / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")
        print("NO_NATURAL_DEMOS", flush=True)
        return

    actions = np.concatenate([t["act"] for t in train_set])
    families = [t["family"] for t in train_set for _ in range(len(t["act"]))]
    reports = []
    chosen = None
    for k in (16, 32):
        rep = token_fit(actions, families, k)
        print(
            "K", k, "open/close", rep["open_close_overlap"], "vz/hold", rep["high_vz_hold_overlap"],
            "local", rep["local_v_mae_mps"], "vz", rep["high_vz_mae_mps"], "empty", rep["empty"],
            flush=True,
        )
        reports.append({k2: v for k2, v in rep.items() if k2 not in ("labels", "centers")})
        good = (
            not rep["open_close_overlap"]
            and not rep["high_vz_hold_overlap"]
            and rep["empty"] <= 2
            and (rep["local_v_mae_mps"] is None or rep["local_v_mae_mps"] < 0.03)
        )
        if chosen is None and (good or k == 32):
            chosen = rep
        if good and k == 16:
            break
    proto = chosen["centers"]
    print("CHOSEN", chosen["k"], flush=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loss, elapsed, amp_on = train_model(train_set, proto, device, use_amp=True)
    tf = teacher_forced(model, train_set + hold_set, device)
    print("TEACHER", [(r["family"], r["mae"], r["vz_mae"]) for r in tf], flush=True)

    closed = []
    for t in natural:
        if t["split"] == "train" and not t["family"].endswith("w1.0") and "dz13.5" not in t["family"]:
            continue
        row = closed_loop(sim, model, t["snap"], nominal, pad_ids, pad_mu, None, 140, "natural", pinch)
        row.update({"family": t["family"], "split": t["split"]})
        closed.append(row)
        print("CL", t["family"], row.get("held_1s"), "gap", row.get("min_gap_mm"), "cmd", row.get("cmd_after_loss"), "a2", row.get("a2_after_loss"), "end", row.get("end_n"), flush=True)
    for t in airborne:
        row = closed_loop(sim, model, t["snap"], nominal, pad_ids, pad_mu, None, 80, "air", t.get("pinch"))
        row.update({"family": t["family"], "split": t["split"], "end_demo": t["end_n"]})
        # Air catches are scored by 1s bilateral inside the rollout when pinch is set.
        closed.append(row)
        print("AIRCL", t["family"], row.get("held_1s"), row.get("t12"), row.get("end_n"), flush=True)
    for t in local:
        row = closed_loop(sim_l, model, t["snap"], nominal_l, pads_l_ids, pad_mu_l, t["mass"], 56, "local")
        row.update({"family": t["family"], "split": t["split"], "ic": t["ic"]})
        closed.append(row)
        print("LOCL", t["family"], t["ic"], row.get("t12"), flush=True)

    stats_act = np.concatenate([t["act"] for t in train_set])
    blob = {
        "mapping": mapping,
        "cuda": info,
        "amp": amp_on,
        "train_s": round(elapsed, 1),
        "epochs": EPOCHS,
        "loss": loss,
        "obs_dim": OBS_DIM_AIR,
        "obs_names": OBS_NAMES_AIR,
        "obs_audit": audit,
        "natural_failed": failed,
        "natural_train": [t["family"] for t in natural if t["split"] == "train"],
        "natural_hold": [t["family"] for t in natural if t["split"] == "hold"],
        "n_train": len(train_set),
        "n_hold": len(hold_set),
        "steps": {t["family"]: int(len(t["act"])) for t in train_set + hold_set},
        "tokenizer": reports,
        "chosen_k": chosen["k"],
        "teacher": tf,
        "closed": closed,
        "train_vz_max": round(float(stats_act[:, 2].max()), 3),
        "train_vz_p95": round(float(np.percentile(stats_act[:, 2], 95)), 3),
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

