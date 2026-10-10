"""Scale the clean natural-loss recatch set and retrain the same Transformer.

Held-out offsets are chosen by a fixed rule before any training run. A
trajectory enters the train or held-out pool only when the first persistent
bilateral loss still lies on the distal tip and the v3 intercept regrasps
for 1 s. Architecture, tokenizer, action map, and the local-skill trajectories
stay fixed. Dataset size and epoch count are the only changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import observe_airborne, relative_state
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import physical_pack
from training.clean_natural_loss_recatch import (
    CLEAR_TOL,
    PERSIST_STEPS,
    distal_geoms,
    measure,
    slip_until_loss,
    windows_from,
)
from training.demo_ballistic_impact import assert_noslip
from training.full_3d_airborne_recatch_dataset import (
    TRACK,
    begin,
    drop_supports,
    grasp_settled,
    hold_step,
    restore_to,
)
from training.horizontal_grasp_offset_freefall_recatch import apply_offset
from training.long_context_recovery_pilot import jsonable
from training.natural_loss_recatch_coverage_expansion import oracle_chase
from training.phase_decoupled_recovery_pilot import world
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import run_impact

RAW = ROOT / "results" / "diagnostics" / "raw" / "natural_loss_data_scaling"
V3_CKPT = (
    ROOT / "results" / "diagnostics" / "raw"
    / "natural_loss_recatch_transformer_v3" / "ckpt_120.pt"
)
TARGET_N = (6, 20, 50, 100)
# The original six clean trajectories. They occupy the front of every subset.
CORE = [
    {"tag": "dz13.2", "dz": -13.2},
    {"tag": "dz13.5", "dz": -13.5},
    {"tag": "dz14.2", "dz": -14.2},
    {"tag": "dz14.6", "dz": -14.6},
    {"tag": "dvz_m", "dz": -13.5, "dvz": -0.01},
    {"tag": "dvz_p", "dz": -13.5, "dvz": 0.005},
]
# Extra held-out conditions, fixed before any of these runs.
HOLD_EXTRA = [
    {"tag": "h_dy", "dz": -14.10, "dy": 0.12},
    {"tag": "h_dvz", "dz": -13.90, "dvz": -0.008},
    {"tag": "h_dx", "dz": -14.70, "dx": 0.10},
]
PERTURBS = [
    {"dvz": -0.012}, {"dvz": -0.006}, {"dvz": 0.006}, {"dvz": 0.012},
    {"dy": -0.20}, {"dy": -0.10}, {"dy": 0.10}, {"dy": 0.20},
    {"dx": -0.10}, {"dx": 0.10},
    {"dq2": -0.0003}, {"dq2": 0.0003},
    {"df": -0.0004}, {"df": 0.0004},
    {"dq4": -0.001}, {"dq4": 0.001},
    {"pitch": -0.4}, {"pitch": 0.4},
]


def dz_grid():
    return [round(-13.0 - 0.05 * i, 2) for i in range(41)]


def hold_dz(dz):
    """Every fifth grid point, plus the two offsets already used as held-out.

    dz -13.5 stays in the original training core even though it lands on a
    fifth-grid point.
    """
    if abs(dz - (-13.4)) < 1e-6 or abs(dz - (-14.0)) < 1e-6:
        return True
    if abs(dz - (-13.5)) < 1e-6:
        return False
    i = int(round((-13.0 - dz) / 0.05))
    return i % 5 == 0


def tag_of(spec):
    if "tag" in spec:
        return spec["tag"]
    parts = [f"dz{spec['dz']:.2f}"]
    for key in ("dx", "dy", "dvz", "dq2", "df", "dq4", "pitch"):
        if abs(float(spec.get(key, 0.0))) > 1e-12:
            parts.append(f"{key}{spec[key]}")
    return "_".join(parts)


def apply_spec(sim, spec):
    apply_offset(sim, {
        "dx": spec.get("dx", 0.0),
        "dy": spec.get("dy", 0.0),
        "dz": spec["dz"],
        "pitch": spec.get("pitch", 0.0),
    })
    if spec.get("dq2"):
        j = int(sim.ids.arm_jnt[1])
        sim.data.qpos[int(sim.model.jnt_qposadr[j])] += float(spec["dq2"])
    if spec.get("dq4"):
        j = int(sim.ids.arm_jnt[3])
        sim.data.qpos[int(sim.model.jnt_qposadr[j])] += float(spec["dq4"])
    if spec.get("df"):
        for j in sim.ids.finger_jnt:
            sim.data.qpos[int(sim.model.jnt_qposadr[int(j)])] += float(spec["df"])
    if spec.get("dvz"):
        dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
        sim.data.qvel[dadr + 2] += float(spec["dvz"])
    mujoco.mj_forward(sim.model, sim.data)


def collect(sim, base, spec, pinch, distal, nominal, pads, pad_mu, split):
    info = restore_to(sim, base, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        return None, {"tag": tag_of(spec), "split": split, "cls": "RESTORE_FAIL"}
    apply_spec(sim, spec)
    start = measure(sim, pinch, distal)
    meta = {
        "tag": tag_of(spec),
        "split": split,
        "dz": spec["dz"],
        "dx": spec.get("dx", 0.0),
        "dy": spec.get("dy", 0.0),
        "dvz": spec.get("dvz", 0.0),
        "dq2": spec.get("dq2", 0.0),
        "df": spec.get("df", 0.0),
        "dq4": spec.get("dq4", 0.0),
        "pitch": spec.get("pitch", 0.0),
    }
    if start["n"][0] == 0 or start["n"][1] == 0:
        meta.update({
            "cls": "INITIAL_ALREADY_SEPARATED",
            "d_center_mm": start["d_center_mm"],
            "d_clearance_mm": start["d_clearance_mm"],
            "oracle": False,
        })
        return None, meta
    slip, loss_snap, why = slip_until_loss(sim, pinch, distal)
    if loss_snap is None:
        meta.update({"cls": why, "oracle": False})
        return None, meta
    info = restore_to(sim, loss_snap, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        meta.update({"cls": "LOSS_RESTORE_FAIL", "oracle": False})
        return None, meta
    loss = measure(sim, pinch, distal)
    meta.update({
        "t_loss": round(slip[-1]["t"] + v3.DT_POLICY, 3) if slip else None,
        "d_center_mm": loss["d_center_mm"],
        "d_clearance_mm": loss["d_clearance_mm"],
        "v_rel_z": loss["v_rel_h"][2],
        "finger_mm": loss["finger_mm"],
        "nearest_geom": loss["nearest_geom"],
    })
    if loss["d_clearance_mm"] > CLEAR_TOL * 1e3:
        meta.update({"cls": "POST_LOSS_SEPARATED", "oracle": False})
        return None, meta
    oframes, osum = oracle_chase(sim, loss_snap, nominal, pads, pad_mu, pinch)
    ok = bool(osum.get("ok"))
    meta.update({
        "cls": "CLEAN_FIRST_LOSS",
        "oracle": ok,
        "oracle_end": osum.get("end_n"),
        "oracle_min_d_center_mm": osum.get("min_excess_mm"),
    })
    if not ok or not oframes:
        meta["cls"] = "CLEAN_ORACLE_UNRECATChABLE"
        return None, meta
    made = windows_from(slip, oframes, meta["tag"], split, loss)
    meta["n_windows"] = len(made)
    if not made:
        meta["cls"] = "NO_WINDOW"
        return None, meta
    return {"meta": meta, "windows": made}, meta


def predict_scores(model, trajs, device):
    """Teacher-forced loss on the suffix that still contains the chase."""
    import torch

    if not trajs:
        return {"loss": None, "token_acc": None, "residual_l2": None, "action_mae": None}
    model.eval()
    losses, accs, rerr, maes = [], [], [], []
    with torch.no_grad():
        for tr in trajs:
            n = len(tr["act"])
            s0 = max(0, n - v3.MAX_LEN)
            obs = torch.tensor(tr["obs"][s0:], device=device)
            prev = torch.tensor(tr["prev"][s0:], device=device)
            act = torch.tensor(tr["act"][s0:], device=device)
            logits, res, pred = model(obs, prev)
            dist = ((act.unsqueeze(1) - model.proto.view(1, -1, 7)) ** 2).sum(-1)
            lab = dist.argmin(-1)
            ce = torch.nn.functional.cross_entropy(logits.float(), lab)
            target = act - model.proto[lab]
            mse = ((res - target) ** 2).mean()
            losses.append(float(ce + mse))
            accs.append(float((logits.argmax(-1) == lab).float().mean()))
            rerr.append(float(torch.linalg.norm(res - target, dim=-1).mean()))
            maes.append(float(torch.linalg.norm(pred.float() - act, dim=-1).mean()))
    return {
        "loss": round(float(np.mean(losses)), 4),
        "token_acc": round(float(np.mean(accs)), 4),
        "residual_l2": round(float(np.mean(rerr)), 4),
        "action_mae": round(float(np.mean(maes)), 4),
    }


def train_one(trajs, proto, device, epochs, hold_windows):
    import torch

    model = v3.build_model(proto, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)
    bank = []
    for tr in trajs:
        bank.append((
            torch.tensor(tr["obs"]),
            torch.tensor(tr["prev"]),
            torch.tensor(tr["act"]),
        ))
    curve = []
    last = None
    amp_on = use_amp
    for epoch in range(epochs):
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
            T = min(v3.MAX_LEN, max(b[0].shape[0] for b in batch))
            B = len(batch)
            obs = torch.zeros(B, T, v3.OBS_DIM_AIR)
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
            s0 = int(rng.integers(0, o.shape[0]))
            batch.append((o[s0:], p[s0:], a[s0:]))
            if len(batch) == 16:
                flush()
        flush()
        last = None if nbat == 0 else running / nbat
        if (epoch + 1) in (120, 240, 480) or (epoch + 1) == epochs or (epoch + 1) % 40 == 0:
            model.eval()
            val = predict_scores(model, hold_windows, device)
            curve.append({"epoch": epoch + 1, "train_loss": None if last is None else round(last, 4), **{f"val_{k}": v for k, v in val.items()}})
            print("epoch", epoch + 1, "train", None if last is None else round(last, 4), "val", val["loss"], "acc", val["token_acc"], flush=True)
            model.train()
    model.eval()
    return model, last, curve, amp_on


def rollout(sim, model, snap, nominal, pads, pad_mu, pinch, distal):
    import torch

    device = next(model.parameters()).device
    info = restore_to(sim, snap, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        return {"pose_ok": False}
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    rows = []
    loss_i = None
    lost_run = 0
    pinch_mm = np.asarray(pinch, float) * 1e3
    model.eval()
    with torch.no_grad():
        for k in range(160):
            meas = measure(sim, pinch, distal)
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(obs)
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            cmd = v3.step_v3(sim, act)
            o = physical_pack(sim)
            rel = np.asarray(meas["p_rel_mm"], float)
            rows.append({
                "act": act,
                "cmd_vz": float(cmd["v_world"][2]),
                "clear": meas["d_clearance_mm"],
                "dc": meas["d_center_mm"],
                "vz": meas["v_rel_h"][2],
                "rel": float(np.linalg.norm(rel - pinch_mm)),
                "post_n": [int(o["nL"]), int(o["nR"])],
                "bilateral": meas["n"][0] > 0 and meas["n"][1] > 0,
            })
            prev = act
            if not rows[-1]["bilateral"]:
                lost_run += 1
                if lost_run >= PERSIST_STEPS and loss_i is None:
                    loss_i = k - PERSIST_STEPS + 1
            else:
                lost_run = 0
            if k > 20 and meas["d_center_mm"] > 80 and int(o["nL"]) == 0 and int(o["nR"]) == 0:
                break
            if float(o["obj_z"]) < -0.05:
                break
    re = None
    if loss_i is not None:
        re = next((i for i in range(loss_i, len(rows)) if rows[i]["post_n"][0] > 0 and rows[i]["post_n"][1] > 0), None)
    held = False
    if re is not None and re + 50 <= len(rows):
        tail = rows[re:re + 50]
        held = float(np.mean([r["post_n"][0] > 0 and r["post_n"][1] > 0 for r in tail])) >= 0.95
        held = held and tail[-1]["post_n"][0] > 0 and tail[-1]["post_n"][1] > 0
    chase = [r for r in (rows[loss_i:] if loss_i is not None else []) if r["act"][2] > 0.4]
    pre = rows[:loss_i] if loss_i is not None else [r for r in rows if r["bilateral"]]
    pre_err = None
    if pre:
        pre_err = float(np.mean([np.linalg.norm(r["act"] - v3.HOLD) for r in pre]))
    after = rows[loss_i:] if loss_i is not None else rows
    return {
        "pose_ok": True,
        "held_1s": bool(held),
        "recontact": re is not None,
        "reclose": bool(re is not None and any(r["act"][6] > 0.5 and r["post_n"][0] > 0 for r in rows[re:])),
        "d_clearance_mm": None if loss_i is None else rows[loss_i]["clear"],
        "v_rel_z": None if loss_i is None else rows[loss_i]["vz"],
        "d_center_mm": None if loss_i is None else rows[loss_i]["dc"],
        "first_a2": None if not chase else round(float(chase[0]["act"][2]), 3),
        "first_cmd": None if not chase else round(float(chase[0]["cmd_vz"]), 3),
        "min_rel_mm": None if not after else round(float(min(r["rel"] for r in after)), 2),
        "min_d_center_mm": None if not after else round(float(min(r["dc"] for r in after)), 2),
        "preloss_mae": None if pre_err is None else round(pre_err, 4),
        "end_n": rows[-1]["post_n"] if rows else None,
    }


def save_summary(blob):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    info = v3.cuda_info()
    v3.assert_map()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
    proto = np.asarray(ckpt["proto"], np.float32)

    sim, _g, nominal, pad_ids, pad_mu, _pl, _pr = world()
    assert_noslip(sim)
    seat = grasp_settled(sim, TRACK)
    if seat is None or not begin(sim, seat, nominal, pad_ids, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    distal = distal_geoms(sim)
    drop_supports(sim)
    base = v3.stamp(sim)
    print("PINCH_MM", [round(float(x) * 1e3, 2) for x in pinch], flush=True)

    audit = []
    hold = []
    train = []

    def accept(spec, split, bucket):
        rec, meta = collect(sim, base, spec, pinch, distal, nominal, pad_ids, pad_mu, split)
        audit.append(meta)
        flag = "KEEP" if rec is not None else meta.get("cls")
        print(split, meta["tag"], flag, "clear", meta.get("d_clearance_mm"), "v", meta.get("v_rel_z"), "n", len(bucket) + (1 if rec else 0), flush=True)
        if rec is not None:
            bucket.append(rec)

    hold_specs = []
    for dz in dz_grid():
        if hold_dz(dz):
            hold_specs.append({"tag": f"h_dz{dz:.2f}", "dz": dz})
    hold_specs.extend(HOLD_EXTRA)
    for spec in hold_specs:
        accept(spec, "hold", hold)
    print("HOLD_READY", len(hold), flush=True)
    (RAW / "hold_manifest.json").write_text(json.dumps(jsonable([r["meta"] for r in hold]), indent=2), encoding="utf-8")

    for spec in CORE:
        accept(spec, "train", train)
    for dz in dz_grid():
        if hold_dz(dz):
            continue
        if any(abs(dz - c["dz"]) < 1e-6 and len(c) == 2 for c in CORE):
            continue
        accept({"dz": dz}, "train", train)
    goods = [r for r in train if abs(r["meta"].get("dx", 0)) < 1e-12 and abs(r["meta"].get("dy", 0)) < 1e-12
             and abs(r["meta"].get("dvz", 0)) < 1e-12 and abs(r["meta"].get("dq2", 0)) < 1e-12
             and abs(r["meta"].get("df", 0)) < 1e-12 and abs(r["meta"].get("dq4", 0)) < 1e-12
             and abs(r["meta"].get("pitch", 0)) < 1e-12]
    for pert in PERTURBS:
        if len(train) >= 100:
            break
        for g in goods:
            if len(train) >= 100:
                break
            spec = {"dz": g["meta"]["dz"], **pert}
            accept(spec, "train", train)
    print("TRAIN_READY", len(train), "HOLD", len(hold), flush=True)

    sim_l, _gl, nominal_l, pads_l, pad_mu_l, pads_left, pads_right = world()
    assert_noslip(sim_l)
    local = []
    impact = run_impact(sim_l, _gl, 6.5, -0.006, 0.0, pads_left, pads_right, nominal_l, None, None)
    sched = schedules()
    for skill in ("wrist_reseat", "recatch_open20", "gravity_inward"):
        tr = v3.record_local(sim_l, skill, sched[skill], impact["snap"], nominal_l, pads_l, pad_mu_l, None)
        if tr is None:
            print("LOCAL_FAIL", skill, flush=True)
            continue
        tr["family"] = skill
        local.append(tr)
        print("LOCAL", skill, len(tr["act"]), flush=True)

    hold_windows = []
    for rec in hold:
        # One window per prefix length. windows_from already built 0.3 / 1.0 / 2.0.
        hold_windows.extend(rec["windows"])
    # Validation uses the 1.0 s window when it exists, so the chase is in view
    # and the three prefixes are not triple-counted.
    hold_val = []
    for rec in hold:
        ones = [w for w in rec["windows"] if w["family"].endswith("p1.0")]
        hold_val.append(ones[0] if ones else rec["windows"][0])

    available = [n for n in TARGET_N if n <= len(train)]
    if len(train) >= 6 and len(train) not in available:
        available.append(len(train))
    epoch_sizes = []
    if 6 in available:
        epoch_sizes.append(6)
    larger = [n for n in available if n >= 20]
    if larger:
        epoch_sizes.append(larger[min(1, len(larger) - 1)] if 50 in larger else larger[-1])
        if 50 in larger and 50 not in epoch_sizes:
            epoch_sizes.append(50)
    epoch_sizes = list(dict.fromkeys(epoch_sizes))[:2]
    runs = []
    for n in available:
        if n in epoch_sizes:
            for epochs in (120, 240, 480):
                runs.append((n, epochs))
        else:
            runs.append((n, 120))
    print("RUNS", runs, "EPOCH_SIZES", epoch_sizes, flush=True)

    results = []
    blob = {
        "cuda": info,
        "n_train_available": len(train),
        "n_hold": len(hold),
        "hold": [r["meta"] for r in hold],
        "train_order": [r["meta"] for r in train],
        "audit_counts": {},
        "runs": results,
    }
    from collections import Counter
    blob["audit_counts"] = dict(Counter(a.get("cls") for a in audit))
    save_summary(blob)

    for n, epochs in runs:
        natural = []
        for rec in train[:n]:
            natural.extend(rec["windows"])
        print("TRAIN_START", n, epochs, "windows", len(natural), flush=True)
        model, train_loss, curve, amp_on = train_one(natural + local, proto, device, epochs, hold_val)
        train_pred = predict_scores(model, [rec["windows"][0] for rec in train[:n]], device)
        closed = []
        for w in hold_windows:
            row = rollout(sim, model, w["snap"], nominal, pad_ids, pad_mu, pinch, distal)
            oracle_clear = w["loss_meas"]["d_clearance_mm"]
            if row.get("d_clearance_mm") is not None:
                row["clearance_err_mm"] = round(float(row["d_clearance_mm"]) - float(oracle_clear), 3)
            else:
                row["clearance_err_mm"] = None
            row["family"] = w["family"]
            row["tag"] = w["tag"]
            closed.append(row)
        n_ok = sum(1 for r in closed if r.get("held_1s"))
        errs = [abs(r["clearance_err_mm"]) for r in closed if r.get("clearance_err_mm") is not None]
        pres = [r["preloss_mae"] for r in closed if r.get("preloss_mae") is not None]
        skills = []
        for tr in local:
            srow = v3.closed_loop(sim_l, model, tr["snap"], nominal_l, pads_l, pad_mu_l, None, 56, "local")
            skills.append({"family": tr["family"], "t12": srow.get("t12")})
        rec = {
            "n": n,
            "epochs": epochs,
            "steps_per_epoch": int(np.ceil(len(natural + local) * 8 / 16)),
            "amp": amp_on,
            "train_loss": None if train_loss is None else round(float(train_loss), 4),
            "train_pred": train_pred,
            "curve": curve,
            "held_out_success": n_ok,
            "held_out_n": len(closed),
            "mean_abs_clearance_err_mm": None if not errs else round(float(np.mean(errs)), 3),
            "mean_preloss_mae": None if not pres else round(float(np.mean(pres)), 4),
            "recontact": sum(1 for r in closed if r.get("recontact")),
            "reclose": sum(1 for r in closed if r.get("reclose")),
            "skills": skills,
            "closed": closed,
        }
        results.append(rec)
        print(
            "DONE_RUN", n, epochs, "success", n_ok, "/", len(closed),
            "clear_err", rec["mean_abs_clearance_err_mm"],
            "pre", rec["mean_preloss_mae"],
            "skills", skills,
            flush=True,
        )
        save_summary(blob)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
