"""Representation audit for the privileged impact_zp6 chase, then a
supervised retrain of the smallest variants that can express that action.

The Transformer width, depth, and heads stay as they are. The physical
recovery7d_airborne_v3 map is not touched. A wider residual scale, or one
appended codebook row, is the only representation change under test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
import experiments.legacy.sac_dpo_grpo_selection as sel
from training.autonomous_dataset import _until_impact_entry
from training.autonomous_episode import T_TASK, Z_ABORT, make_training_env, seat_base
from training.autonomous_rl_sanity import restore_supports, supports_of
from experiments.legacy.balanced_unified_recovery import CKPT_39, EPOCHS, STEPS_PER_EPOCH
from training.full_3d_airborne_recatch_dataset import TRACK
from experiments.recatch.new_recatch_supervised import _enter_air, _enter_impact

RAW = ROOT / "results" / "diagnostics" / "raw" / "action_representation"
SKILL = ROOT / "results" / "diagnostics" / "raw" / "autonomous_training"
DEMO = ROOT / "results" / "diagnostics" / "raw" / "privileged_recatch"
PRIOR = ROOT / "results" / "diagnostics" / "raw" / "new_recatch_supervised" / "ckpt_scratch.pt"
ACCURATE_L2 = 0.05
FAMILIES = ("wrist", "sliding", "recatch", "natural_recatch", "impact")


def _load(path, pool):
    data = np.load(path)
    obs = np.asarray(data["obs"], np.float32)
    prev = np.asarray(data["prev"], np.float32)
    act = np.asarray(data["act"], np.float32)
    phase = None if "phase" not in data.files else [str(x) for x in data["phase"]]
    return {"pool": pool, "name": path.stem, "obs": obs, "prev": prev, "act": act, "phase": phase}


def load_rows():
    rows = [
        _load(SKILL / "demo_wrist_reseat.npz", "wrist"),
        _load(SKILL / "demo_gravity_inward.npz", "sliding"),
        _load(SKILL / "demo_recatch_open20.npz", "recatch"),
        _load(DEMO / "demo_train_dz13.5.npz", "natural_recatch"),
        _load(DEMO / "demo_nl_dz14.6.npz", "natural_recatch"),
        _load(DEMO / "demo_nl_dvz_p.npz", "natural_recatch"),
        _load(DEMO / "demo_impact_zp6.npz", "impact"),
    ]
    return rows


def reconstruct(act, proto, cap):
    dist = ((act[:, None, :] - proto[None]) ** 2).sum(-1)
    lab = dist.argmin(1)
    nearest = proto[lab]
    resid = act - nearest
    clipped = np.clip(resid, -cap, cap)
    pred = np.clip(nearest + clipped, -1.0, 1.0)
    err = act - pred
    return lab, resid, clipped, err


def _round_row(values):
    return [round(float(x), 4) for x in values]


def family_summary(rows, proto, cap):
    out = []
    for row in rows:
        _lab, resid, _clipped, err = reconstruct(row["act"], proto, cap)
        l2 = np.linalg.norm(err, axis=1)
        out.append({
            "name": row["name"],
            "pool": row["pool"],
            "steps": int(len(l2)),
            "mean_l2": float(l2.mean()),
            "max_l2": float(l2.max()),
            "max_abs": float(np.abs(err).max()),
            "steps_l2_above_0.05": int((l2 > ACCURATE_L2).sum()),
            "max_abs_residual": float(np.abs(resid).max()),
        })
    return out


def zp6_steps(act, phase, proto, cap):
    lab, resid, clipped, err = reconstruct(act, proto, cap)
    rows = []
    for i in range(len(act)):
        rows.append({
            "i": i,
            "phase": phase[i],
            "token": int(lab[i]),
            "action": _round_row(act[i]),
            "residual": _round_row(resid[i]),
            "clipped": _round_row(clipped[i]),
            "error": _round_row(err[i]),
            "error_l2": round(float(np.linalg.norm(err[i])), 4),
        })
    return rows


def build_policy(proto, device, res_cap, handoff=False):
    from models.k48_policy import build_policy as build

    return build(proto, device, res_cap, handoff)


def train_variant(trajs, proto, device, res_cap):
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)
    model = build_policy(proto, device, res_cap)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)
    groups = {name: [] for name in FAMILIES}
    bank = []
    pools = []
    for tr in trajs:
        groups[tr["pool"]].append(len(bank))
        pools.append(tr["pool"])
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
            if pools[idx] == "impact" and rng.random() < 0.5:
                s0 = 0
            else:
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
        "res_cap": res_cap,
        "k": int(len(proto)),
        "seconds": round(time.perf_counter() - t0, 1),
        "final_loss": losses[-1],
        "loss_epoch_40": losses[39],
        "loss_epoch_80": losses[79],
        "impact_prefix_fraction": 0.5,
    }


def sequence_fit(model, demo, device):
    model.eval()
    obs = demo["obs"]
    prev = demo["prev"]
    act = demo["act"]
    pred = np.zeros_like(act)
    with torch.no_grad():
        for t in range(len(act)):
            s = max(0, t + 1 - 160)
            out = model(
                torch.tensor(obs[s:t + 1], device=device),
                torch.tensor(prev[s:t + 1], device=device),
            )[2][-1].float().cpu().numpy()
            pred[t] = out
    err = pred - act
    l2 = np.linalg.norm(err, axis=1)
    chase = np.zeros(len(act), dtype=bool) if not demo["phase"] else np.array([p == "chase" for p in demo["phase"]])
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_l2": float(l2.max()),
        "chase_rmse": None if not chase.any() else float(np.sqrt(np.mean(err[chase] ** 2))),
        "chase_pred": None if not chase.any() else [_round_row(x) for x in pred[chase]],
        "chase_act": None if not chase.any() else [_round_row(x) for x in act[chase]],
    }


def roll(env):
    acts = []
    contacts = []
    exited = False
    kind = "time_cap"
    while float(env.sim.data.time) < T_TASK - 1e-9 and len(acts) < 300:
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
    capped = kind == "time_cap" and len(acts) >= 300 and float(env.sim.data.time) < T_TASK - 1e-9
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
    stacked = np.stack(acts) if acts else np.zeros((1, 7), np.float32)
    grip = stacked[:, 6]
    return {
        "t12": t12,
        "exit": kind,
        "stable_exit": bool(exited),
        "steps": len(acts),
        "hit_step_cap": bool(capped),
        "cap_benign": bool(capped and t12 == "RETAINED_TO_12"),
        "opened": bool(float(grip.min()) < -0.3),
        "reclosed_grip": bool(float(grip.min()) < -0.3 and float(grip[-1]) > 0.3),
        "lost_both": bool(both_lost),
        "recontact": bool(reclosed),
        "max_abs_ax": float(np.abs(stacked[:, 0]).max()),
        "max_abs_ay": float(np.abs(stacked[:, 1]).max()),
        "first_actions": [_round_row(a) for a in stacked[:4]],
        "mechanism": (
            "open-chase-reclose"
            if both_lost and reclosed and float(grip.min()) < -0.3
            else "no open-reclose"
        ),
    }


def evaluate(env, model, base, q_ref, table, rows, device):
    env.model = model.to(env.device)
    env.model.eval()
    fit = {}
    for row in rows:
        if row["pool"] in ("wrist", "sliding", "recatch", "natural_recatch", "impact"):
            fit[row["name"]] = sequence_fit(env.model, row, env.device)
    cases = []

    def add(case_id, split, entered, nominal, outcome):
        cases.append({
            "id": case_id,
            "split": split,
            "entered": bool(entered),
            "nominal_t12": nominal,
            **({} if outcome is None else outcome),
        })
        shown = nominal if outcome is None else outcome["t12"]
        exit_name = None if outcome is None else outcome["exit"]
        stable = None if outcome is None else outcome["stable_exit"]
        print(" ", case_id, shown, exit_name, "stable" if stable else "", flush=True)

    print(" zm6", flush=True)
    entered, nominal = _enter_impact(env, table, {"v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0})
    add("impact_zm6", "train_skill", entered, nominal, None if not entered else roll(env))
    print(" z3", flush=True)
    entered, nominal = _enter_impact(env, table, {"v_mps": 6.5, "z_off_m": -0.003, "phi_deg": 0.0})
    add("impact_z3", "validation_old", entered, nominal, None if not entered else roll(env))
    for name, speed in (("air_lat_pos", 0.30), ("air_fast_pos", 0.45)):
        print(" ", name, flush=True)
        entered, nominal, _t0 = _enter_air(env, base, q_ref, {"kind": "air", "speed": speed, "sign": 1.0, "duration_s": 0.08})
        add(name, "validation_old", entered, nominal, None if not entered else roll(env))
    offsets = [
        ("train_dz13.5", "train_recatch", -13.5, 0.0, 0.0, 0.0),
        ("nl_dz14.6", "train_recatch", -14.6, 0.0, 0.0, 0.0),
        ("nl_dvz_p", "train_recatch", -13.5, 0.0, 0.0, 0.005),
        ("dz13.2", "validation_new", -13.2, 0.0, 0.0, 0.0),
        ("dz14.2", "validation_new", -14.2, 0.0, 0.0, 0.0),
        ("dvz_m", "validation_new", -13.5, 0.0, 0.0, -0.01),
        ("h_dz14.00", "heldout_natural", -14.00, 0.0, 0.0, 0.0),
        ("h_dz14.25", "heldout_natural", -14.25, 0.0, 0.0, 0.0),
        ("h_dy", "heldout_natural", -14.10, 0.0, 0.12, 0.0),
        ("h_dx", "heldout_natural", -14.70, 0.10, 0.0, 0.0),
    ]
    for name, split, dz, dx, dy, dvz in offsets:
        print(" ", name, flush=True)
        spec = {"kind": "offset", "dz": dz, "dx": dx, "dy": dy, "dvz": dvz}
        entered, nominal, _t0 = sel._enter_offset(env, base, spec)
        add(name, split, entered, nominal, None if not entered else roll(env))
    print(" zp6", flush=True)
    entered, nominal = _enter_impact(env, table, {"v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0})
    add("impact_zp6", "impact", entered, nominal, None if not entered else roll(env))
    return {"fit": fit, "cases": cases}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = CKPT_39.stat().st_mtime_ns
    rows = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT_39, map_location="cpu", weights_only=False)
    proto = np.asarray(saved["model"]["proto"] if "proto" not in saved else saved["proto"], np.float32)
    if proto.ndim != 2:
        proto = np.asarray(saved["model"]["proto"], np.float32)
    zp6 = next(row for row in rows if row["name"] == "demo_impact_zp6")
    chase = np.array([p == "chase" for p in zp6["phase"]])
    added = zp6["act"][chase].mean(0).astype(np.float32)
    proto_token = np.concatenate([proto, added[None]], 0)
    options = {
        "cap_0.25": {"proto": proto, "cap": 0.25, "added_tokens": 0},
        "cap_0.5": {"proto": proto, "cap": 0.5, "added_tokens": 0},
        "cap_1.0": {"proto": proto, "cap": 1.0, "added_tokens": 0},
        "lateral_token": {"proto": proto_token, "cap": 0.25, "added_tokens": 1},
    }
    audit = {
        "added_lateral_prototype": _round_row(added),
        "token28": _round_row(proto[28]),
        "token0": _round_row(proto[0]),
        "options": {},
    }
    base_skills = None
    for name, spec in options.items():
        summary = family_summary(rows, spec["proto"], spec["cap"])
        by_name = {row["name"]: row for row in summary}
        if name == "cap_0.25":
            base_skills = by_name
            steps = zp6_steps(zp6["act"], zp6["phase"], spec["proto"], spec["cap"])
            audit["zp6_steps_cap_0.25"] = steps
        else:
            steps = None
        skill_worse = []
        for row in summary:
            if row["pool"] == "impact":
                continue
            before = base_skills[row["name"]]["mean_l2"]
            if row["mean_l2"] > before + 0.01:
                skill_worse.append(row["name"])
        zp6_row = by_name["demo_impact_zp6"]
        accurate = zp6_row["max_l2"] <= ACCURATE_L2
        audit["options"][name] = {
            "k": int(spec["proto"].shape[0]),
            "res_cap": spec["cap"],
            "added_tokens": spec["added_tokens"],
            "families": summary,
            "zp6_accurate": bool(accurate),
            "skill_reconstruction_worse": skill_worse,
            "viable": bool(accurate and not skill_worse),
        }
        if steps is None:
            chase_idx = np.where(chase)[0]
            lab, resid, clipped, err = reconstruct(zp6["act"], spec["proto"], spec["cap"])
            audit["options"][name]["zp6_chase_steps"] = [
                {
                    "i": int(i),
                    "token": int(lab[i]),
                    "action": _round_row(zp6["act"][i]),
                    "residual": _round_row(resid[i]),
                    "clipped": _round_row(clipped[i]),
                    "error": _round_row(err[i]),
                    "error_l2": round(float(np.linalg.norm(err[i])), 4),
                }
                for i in chase_idx
            ]
        print(
            name,
            "zp6 maxL2", round(zp6_row["max_l2"], 4),
            "accurate", accurate,
            "worse", skill_worse,
            flush=True,
        )
    viable = [name for name, row in audit["options"].items() if row["viable"]]
    print("VIABLE", viable, flush=True)
    trained = {}
    models = {}
    for name in viable:
        spec = options[name]
        print("TRAIN", name, flush=True)
        model, info = train_variant(rows, spec["proto"], device, spec["cap"])
        trained[name] = info
        models[name] = model
        torch.save(
            {"model": model.state_dict(), "proto": spec["proto"], "res_cap": spec["cap"], "obs_dim": 39},
            RAW / f"ckpt_{name.replace('.', '_')}.pt",
        )
    prior = torch.load(PRIOR, map_location="cpu", weights_only=False)
    prior_proto = np.asarray(prior["proto"], np.float32)
    prior_model = build_policy(prior_proto, device, 0.25)
    prior_model.load_state_dict(prior["model"])
    prior_model.eval()
    models = {"prior_scratch": prior_model, **models}
    env = make_training_env()
    table = supports_of(env.sim)
    base, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    evals = {}
    for name, model in models.items():
        print("EVAL", name, flush=True)
        evals[name] = evaluate(env, model, base, q_ref, table, rows, env.device)
    payload = {
        "accurate_l2": ACCURATE_L2,
        "audit": audit,
        "trained": trained,
        "eval": evals,
        "ckpt_mtime_unchanged": CKPT_39.stat().st_mtime_ns == stamp,
    }
    out = RAW / "representation.json"
    out.write_text(json.dumps(_jsonable(payload)), encoding="utf-8")
    print("WROTE", out, "ckpt_same", payload["ckpt_mtime_unchanged"], flush=True)


if __name__ == "__main__":
    main()
