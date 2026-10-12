"""Supervised retraining on the new legal recatch demonstrations.

Fine-tune ckpt_39d.pt and train one model from scratch on the same
family-balanced set. The architecture, K=32 codebook, observation, action
map, geometry switch, and t=12 rule stay as they are. Held-out privileged
successes are not training data and are not used to pick the checkpoint.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
import training.sac_dpo_grpo_selection as sel
from training.autonomous_dataset import _until_impact_entry
from training.autonomous_episode import (
    T_TASK,
    Z_ABORT,
    fresh_nominal_state,
    make_training_env,
    nominal_substep,
    seat_base,
)
from training.autonomous_rl_sanity import restore_supports, supports_of
from training.balanced_unified_recovery import CKPT_39, EPOCHS, STEPS_PER_EPOCH
from training.full_3d_airborne_recatch_dataset import TRACK, drop_supports, restore_to, step_mapped
from training.natural_loss_recatch_transformer_v3 import AIRBORNE_V2, build_model, load_airborne_policy
from controllers.geometry_switch import GeometrySwitch, hand_pose
from controllers.recovery_actions import normalize_action
from envs.physical_recovery import physical_pack

RAW = ROOT / "results" / "diagnostics" / "raw" / "new_recatch_supervised"
SKILL = ROOT / "results" / "diagnostics" / "raw" / "autonomous_training"
DEMO = ROOT / "results" / "diagnostics" / "raw" / "privileged_recatch"
FAMILIES = ("wrist", "sliding", "recatch", "natural_recatch")
RES_CAP = 0.25


def _load_pair(path, pool):
    data = np.load(path)
    obs = np.asarray(data["obs"], np.float32)
    prev = np.asarray(data["prev"], np.float32)
    act = np.asarray(data["act"], np.float32)
    if obs.ndim != 2 or obs.shape[1] != 39 or act.shape != obs.shape[:1] + (7,):
        raise RuntimeError(f"{path} has obs {obs.shape} act {act.shape}")
    phase = None if "phase" not in data.files else [str(x) for x in data["phase"]]
    return {"pool": pool, "name": path.stem, "obs": obs, "prev": prev, "act": act, "phase": phase}


def load_train():
    rows = [
        _load_pair(SKILL / "demo_wrist_reseat.npz", "wrist"),
        _load_pair(SKILL / "demo_gravity_inward.npz", "sliding"),
        _load_pair(SKILL / "demo_recatch_open20.npz", "recatch"),
        _load_pair(DEMO / "demo_train_dz13.5.npz", "natural_recatch"),
        _load_pair(DEMO / "demo_nl_dz14.6.npz", "natural_recatch"),
        _load_pair(DEMO / "demo_nl_dvz_p.npz", "natural_recatch"),
    ]
    return rows


def _represent(act, proto):
    dist = ((act[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
    lab = dist.argmin(1)
    recon = proto[lab]
    resid = act - recon
    pred = np.clip(recon + np.clip(resid, -RES_CAP, RES_CAP), -1.0, 1.0)
    return lab, float(np.linalg.norm(act - recon, axis=-1).mean()), float(np.linalg.norm(act - pred, axis=-1).mean()), resid


def token_report(rows, proto, extra=None):
    out = []
    for row in list(rows) + ([] if extra is None else [extra]):
        lab, proto_err, capped_err, resid = _represent(row["act"], proto)
        open_tok = sorted({int(t) for t, a in zip(lab, row["act"]) if float(a[6]) < -0.5})
        close_tok = sorted({int(t) for t, a in zip(lab, row["act"]) if float(a[6]) > 0.5})
        lateral = np.abs(row["act"][:, 1]) > 0.4
        lat_over = None
        if lateral.any():
            lat_over = float((np.abs(resid[lateral, 1]) > RES_CAP).mean())
        phases = row["phase"] or []
        phase_tokens = {}
        for phase, token in zip(phases, lab):
            phase_tokens.setdefault(phase, set()).add(int(token))
        out.append({
            "name": row["name"],
            "pool": row["pool"],
            "steps": int(len(lab)),
            "tokens": [int(t) for t in lab],
            "unique_tokens": sorted({int(t) for t in lab}),
            "proto_l2": proto_err,
            "capped_residual_l2": capped_err,
            "residual_abs_mean": float(np.abs(resid).mean()),
            "residual_abs_max": float(np.abs(resid).max()),
            "open_tokens": open_tok,
            "close_tokens": close_tok,
            "open_close_disjoint": bool(open_tok and close_tok and set(open_tok).isdisjoint(close_tok)),
            "phase_tokens": {k: sorted(v) for k, v in phase_tokens.items()},
            "lateral_steps": int(lateral.sum()),
            "lateral_residual_over_cap_fraction": lat_over,
        })
    return out


def train_families(trajs, proto, device, init_state):
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)
    model = build_model(proto, device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)
    groups = {name: [] for name in FAMILIES}
    bank = []
    for tr in trajs:
        groups[tr["pool"]].append(len(bank))
        bank.append((
            torch.tensor(tr["obs"]),
            torch.tensor(tr["prev"]),
            torch.tensor(tr["act"]),
        ))
    n_suffix = STEPS_PER_EPOCH * 16
    counts = {name: n_suffix // len(FAMILIES) for name in FAMILIES}
    counts[FAMILIES[0]] += n_suffix - sum(counts.values())
    losses = []
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        order = []
        for name in FAMILIES:
            choice = rng.choice(groups[name], size=counts[name], replace=True)
            order.extend(int(i) for i in choice)
        rng.shuffle(order)
        running = 0.0
        nbat = 0
        batch = []

        def flush():
            nonlocal running, nbat, use_amp
            if not batch:
                return
            length = min(v3.MAX_LEN, max(item[0].shape[0] for item in batch))
            obs = torch.zeros(len(batch), length, 39)
            prev = torch.zeros(len(batch), length, 7)
            act = torch.zeros(len(batch), length, 7)
            valid = torch.zeros(len(batch), length, dtype=torch.bool)
            for i, (o, p, a) in enumerate(batch):
                n = min(length, o.shape[0])
                obs[i, :n] = o[:n]
                prev[i, :n] = p[:n]
                act[i, :n] = a[:n]
                valid[i, :n] = True
            obs = obs.to(device)
            prev = prev.to(device)
            act = act.to(device)
            valid = valid.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, res, _pred = model(obs, prev, ~valid)
                dist = ((act.unsqueeze(2) - model.proto.view(1, 1, -1, 7)) ** 2).sum(-1)
                lab = dist.argmin(-1)
                ce_tok = torch.nn.functional.cross_entropy(logits[valid].float(), lab[valid], reduction="none")
                wgt = torch.ones_like(lab, dtype=torch.float32)
                wgt = torch.where(act[:, :, 2] > 0.4, wgt * 4.0, wgt)
                wgt = torch.where(act[:, :, 6] < -0.2, wgt * 4.0, wgt)
                wgt = torch.where(act[:, :, 4].abs() > 0.4, wgt * 4.0, wgt)
                w = wgt[valid]
                ce = (ce_tok * w).sum() / w.sum().clamp_min(1.0)
                target = act - model.proto[lab]
                loss = ce + ((res - target)[valid] ** 2).mean()
            if not torch.isfinite(loss):
                use_amp = False
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
            o, p, a = bank[idx]
            s0 = int(rng.integers(0, o.shape[0]))
            batch.append((o[s0:], p[s0:], a[s0:]))
            if len(batch) == 16:
                flush()
        flush()
        last = None if nbat == 0 else running / nbat
        losses.append(last)
        if (epoch + 1) % 40 == 0 or epoch + 1 == EPOCHS:
            print("epoch", epoch + 1, "loss", None if last is None else round(last, 4), flush=True)
    model.eval()
    return model, {
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "family_draws_per_epoch": counts,
        "seconds": round(time.perf_counter() - t0, 1),
        "final_loss": losses[-1],
        "loss_epoch_40": losses[39],
        "loss_epoch_80": losses[79],
    }


def _rmse(model, demo, device):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(demo["obs"], device=device).unsqueeze(0),
            torch.tensor(demo["prev"], device=device).unsqueeze(0),
        )[2][0]
    err = pred - torch.tensor(demo["act"], device=device)
    return float(err.pow(2).mean().sqrt())


def _enter_impact(env, table, spec):
    restore_supports(env.sim, table)
    _until_impact_entry(env, spec)
    if not env.switch.entered:
        t12, _t = env.finish_nominal(env.gains)
        return False, t12
    env.start_recovery()
    return True, None


def _enter_air(env, base, q_ref, spec):
    saved = sel.assert_allowed
    sel.assert_allowed = lambda _spec: None
    try:
        return sel._enter_air(env, base, q_ref, spec)
    finally:
        sel.assert_allowed = saved


def _roll(env, cap_steps):
    acts = []
    contacts = []
    exited = False
    kind = "time_cap"
    while float(env.sim.data.time) < T_TASK - 1e-9 and len(acts) < cap_steps:
        act = env.recovery_step()
        row = env.switch.measure(env.sim)
        acts.append(np.asarray(act, np.float32))
        contacts.append((int(row["nL"]), int(row["nR"])))
        if row["z"] < Z_ABORT:
            kind = "fall_abort"
            break
        if env.switch.update_exit(env.sim):
            kind = "geometry"
            exited = True
            break
    capped = kind == "time_cap" and len(acts) >= cap_steps and float(env.sim.data.time) < T_TASK - 1e-9
    if kind == "fall_abort":
        t12 = "DROP"
    else:
        t12, _t = env.finish_nominal(TRACK)
    both_lost = any(nL == 0 and nR == 0 for nL, nR in contacts)
    reclosed = False
    seen_open = False
    for nL, nR in contacts:
        if nL == 0 and nR == 0:
            seen_open = True
        elif seen_open and nL > 0 and nR > 0:
            reclosed = True
    grip = np.stack(acts)[:, 6] if acts else np.zeros(1)
    return {
        "t12": t12,
        "exit": kind,
        "stable_exit": bool(exited),
        "steps": len(acts),
        "hit_step_cap": bool(capped),
        "cap_benign": bool(capped and t12 == "RETAINED_TO_12"),
        "opened": bool(float(grip.min()) < -0.3) if len(acts) else False,
        "reclosed_grip": bool(float(grip.min()) < -0.3 and float(grip[-1]) > 0.3) if len(acts) else False,
        "lost_both": bool(both_lost),
        "recontact": bool(reclosed),
        "mechanism": (
            "open-chase-reclose"
            if both_lost and reclosed and float(grip.min()) < -0.3
            else "no open-reclose"
        ),
    }


def evaluate(env, model, base, q_ref, table, skills, device):
    env.model.load_state_dict(model.state_dict())
    env.model.eval()
    skill_fit = {row["pool"]: _rmse(model, row, device) for row in skills}
    cases = []

    def add(case_id, split, entered, nominal, roll):
        cases.append({
            "id": case_id,
            "split": split,
            "entered": bool(entered),
            "nominal_t12": nominal,
            **({} if roll is None else roll),
        })
        shown = nominal if roll is None else roll["t12"]
        print(" ", case_id, shown, flush=True)

    print(" zm6", flush=True)
    entered, nominal = _enter_impact(env, table, {"v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0})
    add("impact_zm6", "train_skill", entered, nominal, None if not entered else _roll(env, 300))
    print(" z3", flush=True)
    entered, nominal = _enter_impact(env, table, {"v_mps": 6.5, "z_off_m": -0.003, "phi_deg": 0.0})
    add("impact_z3", "validation_old", entered, nominal, None if not entered else _roll(env, 300))
    for name, speed in (("air_lat_pos", 0.30), ("air_fast_pos", 0.45)):
        print(" ", name, flush=True)
        entered, nominal, _t0 = _enter_air(env, base, q_ref, {"kind": "air", "speed": speed, "sign": 1.0, "duration_s": 0.08})
        add(name, "validation_old", entered, nominal, None if not entered else _roll(env, 300))
    offsets = [
        ("train_dz13.5", "train_recatch", -13.5, 0.0),
        ("nl_dz14.6", "train_recatch", -14.6, 0.0),
        ("nl_dvz_p", "train_recatch", -13.5, 0.005),
        ("dz13.2", "validation_new", -13.2, 0.0),
        ("dz14.2", "validation_new", -14.2, 0.0),
        ("dvz_m", "validation_new", -13.5, -0.01),
    ]
    for name, split, dz, dvz in offsets:
        print(" ", name, flush=True)
        spec = {"kind": "offset", "dz": dz, "dx": 0.0, "dy": 0.0, "dvz": dvz}
        entered, nominal, _t0 = sel._enter_offset(env, base, spec)
        add(name, split, entered, nominal, None if not entered else _roll(env, 300))
    return {"skill_rmse": skill_fit, "cases": cases}


def evaluate_heldout(env, model, base, q_ref, table):
    env.model.load_state_dict(model.state_dict())
    env.model.eval()
    rows = []
    offsets = [
        ("h_dz14.00", -14.00, 0.0, 0.0, 0.0),
        ("h_dz14.25", -14.25, 0.0, 0.0, 0.0),
        ("h_dy", -14.10, 0.0, 0.12, 0.0),
        ("h_dx", -14.70, 0.10, 0.0, 0.0),
    ]
    for name, dz, dx, dy, dvz in offsets:
        print(" held", name, flush=True)
        spec = {"kind": "offset", "dz": dz, "dx": dx, "dy": dy, "dvz": dvz}
        entered, nominal, _t0 = sel._enter_offset(env, base, spec)
        roll = None if not entered else _roll(env, 300)
        rows.append({"id": name, "entered": bool(entered), "nominal_t12": nominal, **({} if roll is None else roll)})
    print(" held impact_zp6", flush=True)
    entered, nominal = _enter_impact(env, table, {"v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0})
    roll = None if not entered else _roll(env, 300)
    rows.append({"id": "impact_zp6", "entered": bool(entered), "nominal_t12": nominal, **({} if roll is None else roll)})
    return rows


def _success(cases, split):
    chosen = [row for row in cases if row["split"] == split and row.get("t12")]
    return int(sum(row["t12"] == "RETAINED_TO_12" for row in chosen)), len(chosen)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = CKPT_39.stat().st_mtime_ns
    trajs = load_train()
    zp6 = _load_pair(DEMO / "demo_impact_zp6.npz", "analysis_only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, proto, saved = load_airborne_policy(CKPT_39, device)
    tokens = token_report(trajs, proto, extra=zp6)
    inadequate = any(row["capped_residual_l2"] > 0.15 and row["pool"] != "analysis_only" for row in tokens if row["name"] != "demo_impact_zp6")
    print("TOKEN", [{k: row[k] for k in ("name", "proto_l2", "capped_residual_l2", "open_close_disjoint")} for row in tokens], flush=True)
    if inadequate:
        raise RuntimeError("K=32 cannot represent a training demonstration; K was not changed")
    init = {k: v.detach().cpu().clone() for k, v in saved["model"].items()}
    print("TRAIN finetune", flush=True)
    fine, fine_info = train_families(trajs, proto, device, init)
    print("TRAIN scratch", flush=True)
    scratch, scratch_info = train_families(trajs, proto, device, None)
    torch.save({"model": fine.state_dict(), "proto": proto, "obs_dim": 39}, RAW / "ckpt_finetune.pt")
    torch.save({"model": scratch.state_dict(), "proto": proto, "obs_dim": 39}, RAW / "ckpt_scratch.pt")
    env = make_training_env()
    table = supports_of(env.sim)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    skills = [row for row in trajs if row["pool"] in ("wrist", "sliding", "recatch")]
    pretrained, _p, _s = load_airborne_policy(CKPT_39, env.device)
    report = {}
    for name, model in (("pretrained", pretrained), ("finetune", fine.to(env.device)), ("scratch", scratch.to(env.device))):
        print("EVAL", name, flush=True)
        block = evaluate(env, model, base, q_ref, table, skills, env.device)
        block["train_recatch"] = _success(block["cases"], "train_recatch")
        block["validation_new"] = _success(block["cases"], "validation_new")
        block["validation_old"] = _success(block["cases"], "validation_old")
        report[name] = block
    ranked = []
    for name in ("finetune", "scratch"):
        block = report[name]
        zm6 = next(row for row in block["cases"] if row["id"] == "impact_zm6")
        ranked.append((
            block["validation_new"][0] + block["validation_old"][0],
            1 if zm6.get("t12") == "RETAINED_TO_12" else 0,
            block["train_recatch"][0],
            -float(sum(block["skill_rmse"].values())),
            name,
        ))
    ranked.sort(reverse=True)
    selected = ranked[0][3]
    print("SELECTED", selected, "before held-out", flush=True)
    held = {}
    models = {
        "pretrained": pretrained,
        "finetune": fine.to(env.device),
        "scratch": scratch.to(env.device),
    }
    for name, model in models.items():
        print("HELDOUT", name, flush=True)
        held[name] = evaluate_heldout(env, model, base, q_ref, table)
    payload = {
        "families": FAMILIES,
        "training_names": [row["name"] for row in trajs],
        "excluded": ["demo_h_dz14.00", "demo_h_dz14.25", "demo_h_dy", "demo_h_dx", "demo_impact_zp6"],
        "k_changed": False,
        "token_inadequate": False,
        "tokens": tokens,
        "finetune_train": fine_info,
        "scratch_train": scratch_info,
        "selected_before_heldout": selected,
        "eval": report,
        "heldout": held,
        "ckpt_mtime_unchanged": CKPT_39.stat().st_mtime_ns == stamp,
    }
    (RAW / "retrain.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", RAW / "retrain.json", "ckpt_same", payload["ckpt_mtime_unchanged"], flush=True)


if __name__ == "__main__":
    main()
