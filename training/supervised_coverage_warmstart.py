"""Supervised coverage warm-start. One airborne token, then a from-scratch fit.

Does not run SAC, DPO, or GRPO updates, and does not touch final test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
import training.recovery_success as success
from training.action_representation_expansion import build_policy
from training.autonomous_episode import Z_ABORT, make_training_env, seat_base
from training.autonomous_rl_sanity import restore_supports, supports_of
from training.balanced_unified_recovery import EPOCHS, STEPS_PER_EPOCH
from training.final_rl_readiness import (
    CAP,
    CATALOG,
    CKPT,
    DEMO,
    GROUP,
    SKILL,
    TRAJ,
    load_ic,
    policy_fn,
    reconstruct,
    rollout,
)
from training.history_aware_critic_ablation import restore_capture
from training.new_recatch_supervised import _enter_air
from training.nominal_pose_audit import capture_ref, errors, until_impact

RAW = ROOT / "results" / "diagnostics" / "raw" / "supervised_coverage_warmstart"
RECOV = ROOT / "results" / "diagnostics" / "raw" / "train_recoverability"
FAMILIES = (
    "wrist",
    "sliding",
    "recatch",
    "natural_recatch",
    "restoration",
    "airborne",
    "impact_example",
)
PREFIX_FAMILIES = ("restoration", "airborne", "impact_example")


def load_traj(path, pool):
    data = np.load(path)
    return {
        "pool": pool,
        "name": path.stem,
        "obs": np.asarray(data["obs"], np.float32),
        "prev": np.asarray(data["prev"], np.float32),
        "act": np.asarray(data["act"], np.float32),
    }


def outlier_steps(act, proto):
    dist = ((act[:, None, :] - proto[None]) ** 2).sum(-1)
    lab = dist.argmin(1)
    resid = act - proto[lab]
    rows = []
    for i in range(len(act)):
        if np.any(np.abs(resid[i]) > CAP + 1e-8):
            rows.append({
                "index": int(i),
                "action": [round(float(x), 6) for x in act[i]],
                "token": int(lab[i]),
                "residual": [round(float(x), 6) for x in resid[i]],
                "l2": float(np.linalg.norm(act[i] - np.clip(proto[lab[i]] + np.clip(resid[i], -CAP, CAP), -1, 1))),
            })
    return rows


def audit_actions(base):
    traces = {
        "air_a": np.asarray(np.load(RECOV / "success_air_a.npz")["act"], np.float32),
        "air_b": np.asarray(np.load(RECOV / "success_air_b.npz")["act"], np.float32),
    }
    outliers = {name: outlier_steps(act, base) for name, act in traces.items()}
    if any(len(rows) != 1 for rows in outliers.values()):
        return {"covered": False, "reason": "expected exactly one out-of-range step on each airborne trace", "outliers": outliers}
    a = np.asarray(outliers["air_a"][0]["action"], np.float32)
    b = np.asarray(outliers["air_b"][0]["action"], np.float32)
    gap = b - a
    proto_new = ((a + b) * 0.5).astype(np.float32)
    resid_a = a - proto_new
    resid_b = b - proto_new
    covered = bool(np.max(np.abs(resid_a)) <= CAP + 1e-8 and np.max(np.abs(resid_b)) <= CAP + 1e-8)
    return {
        "covered": covered,
        "outliers": outliers,
        "distance_l2": float(np.linalg.norm(gap)),
        "distance_components": [round(float(x), 6) for x in gap],
        "prototype": [round(float(x), 6) for x in proto_new],
        "residual_air_a": [round(float(x), 6) for x in resid_a],
        "residual_air_b": [round(float(x), 6) for x in resid_b],
        "proto_vector": proto_new,
    }


def dataset():
    rows = [
        load_traj(SKILL / "demo_wrist_reseat.npz", "wrist"),
        load_traj(SKILL / "demo_gravity_inward.npz", "sliding"),
        load_traj(SKILL / "demo_recatch_open20.npz", "recatch"),
        load_traj(DEMO / "demo_train_dz13.5.npz", "natural_recatch"),
        load_traj(DEMO / "demo_nl_dz14.6.npz", "natural_recatch"),
        load_traj(DEMO / "demo_nl_dvz_p.npz", "natural_recatch"),
        load_traj(TRAJ, "restoration"),
        load_traj(DEMO / "demo_impact_zp6.npz", "restoration"),
        load_traj(RECOV / "success_air_a.npz", "airborne"),
        load_traj(RECOV / "success_air_b.npz", "airborne"),
        load_traj(RECOV / "success_imp_c.npz", "impact_example"),
    ]
    return rows


def representation_table(trajs, proto):
    rows = []
    stolen = []
    for tr in trajs:
        info = reconstruct(tr["act"], proto)
        base_info = reconstruct(tr["act"], proto[:35])
        dist = ((tr["act"][:, None, :] - proto[None]) ** 2).sum(-1)
        lab = dist.argmin(1)
        moved = [int(i) for i in range(len(lab)) if int(lab[i]) == 35]
        if moved:
            stolen.append({"name": tr["name"], "steps": moved})
        rows.append({
            "name": tr["name"],
            "pool": tr["pool"],
            "steps": int(len(tr["act"])),
            "max_l2_k35": base_info["max_l2"],
            "max_l2_k36": info["max_l2"],
            "clipped_k35": base_info["clipped_components"],
            "clipped_k36": info["clipped_components"],
            "uses_token_35": int(np.sum(lab == 35)),
        })
    new_names = {"success_air_a", "success_air_b", "success_imp_c"}
    ok = True
    for row in rows:
        if row["name"] in new_names:
            ok = ok and row["max_l2_k36"] <= 0.05 and row["clipped_k36"] == 0
        else:
            ok = ok and row["max_l2_k36"] <= max(row["max_l2_k35"], 0.05) + 1e-8
            ok = ok and row["clipped_k36"] <= row["clipped_k35"]
    for item in stolen:
        ok = ok and item["name"] in ("success_air_a", "success_air_b")
    return rows, stolen, bool(ok)


def train(trajs, proto, device):
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)
    model = build_policy(proto, device, CAP)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)
    groups = {name: [] for name in FAMILIES}
    bank, pools = [], []
    for tr in trajs:
        groups[tr["pool"]].append(len(bank))
        pools.append(tr["pool"])
        bank.append((torch.tensor(tr["obs"]), torch.tensor(tr["prev"]), torch.tensor(tr["act"])))
    n_suffix = STEPS_PER_EPOCH * 16
    counts = {name: n_suffix // len(FAMILIES) for name in FAMILIES}
    counts[FAMILIES[0]] += n_suffix - sum(counts.values())
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        order = []
        for name in FAMILIES:
            order.extend(int(i) for i in rng.choice(groups[name], size=counts[name], replace=True))
        rng.shuffle(order)
        running, nbat, batch = 0.0, 0, []

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
            obs, prev, act, valid = (t.to(device) for t in (obs, prev, act, valid))
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
                target = act - model.proto[lab]
                loss = (ce_tok * w).sum() / w.sum().clamp_min(1.0) + ((res - target)[valid] ** 2).mean()
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
            if pools[idx] in PREFIX_FAMILIES and rng.random() < 0.75:
                s0 = 0
            else:
                s0 = int(rng.integers(0, max(1, o.shape[0])))
            batch.append((o[s0:], p[s0:], a[s0:]))
            if len(batch) == 16:
                flush()
        flush()
        last = None if nbat == 0 else running / nbat
        losses.append(last)
        if (epoch + 1) % 40 == 0 or epoch + 1 == EPOCHS:
            print("epoch", epoch + 1, "loss", None if last is None else round(last, 4), flush=True)
    model.eval()
    return model, counts, losses


def policy_step(env):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    with torch.no_grad():
        logits, _res, act = env.model(
            torch.tensor(np.stack(env._obs_seq[-160:]), device=env.device),
            torch.tensor(np.stack(env._prev_seq[-160:]), device=env.device),
        )
    token = int(torch.argmax(logits[-1]).item())
    clipped = np.clip(act[-1].float().cpu().numpy(), -1.0, 1.0).astype(np.float32)
    v3.step_v3(env.sim, clipped)
    env._prev = clipped
    return token, clipped


def det_roll(env, ref, cap=120):
    entry = errors(env.sim, ref)
    tokens, grips = [], []
    basin_run = 0
    fell = False
    hold = None
    open_i = None
    close_i = None
    for _ in range(cap):
        token, act = policy_step(env)
        row = errors(env.sim, ref)
        tokens.append(token)
        grips.append(float(act[6]))
        if open_i is None and act[6] < -0.5:
            open_i = len(tokens) - 1
        if open_i is not None and close_i is None and act[6] > 0.5 and row["nL"] > 0 and row["nR"] > 0:
            close_i = len(tokens) - 1
        if row["z"] < Z_ABORT:
            fell = True
            break
        inside = success.basin_ok(row, float(act[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        if basin_run >= success.PERSIST_STEPS:
            success.apply_handoff(env.sim, ref)
            hold = success.nominal_validate(env, ref)
            break
    label = success.label_of(fell, hold is not None, hold is not None, hold)
    chase = 0
    if open_i is not None:
        stop = close_i if close_i is not None else len(grips)
        chase = int(sum(g < -0.5 for g in grips[open_i:stop]))
    return {
        "label": label,
        "success": bool(success.is_success(label)),
        "basin": hold is not None,
        "handoff": None if hold is None else hold["t_handoff"],
        "held_10": False if hold is None else bool(hold["held_10"]),
        "held_12": False if hold is None else bool(hold["held_12"]),
        "entry_ep_mm": round(entry["ep"] * 1000.0, 3),
        "entry_vrel": round(entry["vrel"], 4),
        "entry_eR": round(entry["eR"], 4),
        "open_step": open_i,
        "close_step": close_i,
        "chase_steps": chase,
        "token_35": int(sum(t == 35 for t in tokens)),
        "tokens": tokens[:24],
        "steps": len(tokens),
    }


def roll_ic(env, episode_id):
    blob = load_ic(episode_id)
    restore_capture(env, blob["capture"])
    out = det_roll(env, blob["ref"])
    out["id"] = episode_id
    print(" det", episode_id, out["label"], "tok35", out["token_35"], "open", out["open_step"], "chase", out["chase_steps"], flush=True)
    return out


def roll_impact_spec(env, table, name, spec):
    restore_supports(env.sim, table)
    ref, _ball = until_impact(env, spec)
    if ref is None or not env.switch.entered:
        print(name, "no entry", flush=True)
        return {"id": name, "entered": False}
    env.start_recovery()
    out = det_roll(env, ref)
    out["id"] = name
    out["entered"] = True
    print(" det", name, out["label"], "tok35", out["token_35"], flush=True)
    return out


def roll_offset(env, base, name, spec):
    from training.final_rl_readiness import enter_offset
    ref, entered = enter_offset(env, base, spec)
    if not entered:
        print(name, "no entry", flush=True)
        return {"id": name, "entered": False}
    env.start_recovery()
    out = det_roll(env, ref)
    out["id"] = name
    out["entered"] = True
    print(" det", name, out["label"], flush=True)
    return out


def roll_legacy_air(env, base, q_ref, name, speed):
    entered, _n, _t = _enter_air(env, base, q_ref, {"kind": "air", "speed": speed, "sign": 1.0, "duration_s": 0.08})
    if not entered or env.switch.p0 is None:
        print(name, "no entry", flush=True)
        return {"id": name, "entered": False}
    ref = capture_ref(env.sim, env.switch)
    env.start_recovery()
    out = det_roll(env, ref)
    out["id"] = name
    out["entered"] = True
    print(" det", name, out["label"], "tok35", out["token_35"], flush=True)
    return out


def roll_near_air(env, base, q_ref, name, spec):
    from training.final_rl_readiness import enter_air
    ref, entered = enter_air(env, base, q_ref, spec)
    if not entered:
        print(name, "no entry", flush=True)
        return {"id": name, "entered": False, "spec": spec}
    env.start_recovery()
    out = det_roll(env, ref)
    out["id"] = name
    out["entered"] = True
    out["spec"] = spec
    print(" near", name, out["label"], "ep", out["entry_ep_mm"], "vrel", out["entry_vrel"], "tok35", out["token_35"], flush=True)
    return out


def grpo(env):
    groups = []
    for spec in CATALOG:
        if spec["split"] != "train":
            continue
        blob = load_ic(spec["id"])
        members = []
        for g in range(GROUP):
            restore_capture(env, blob["capture"])
            rng = np.random.default_rng(2000 + 17 * g + sum(ord(c) for c in spec["id"]))
            out = rollout(env, blob["ref"], lambda rng=rng: policy_fn(env, rng), 100, {"audit": None})
            acts = np.asarray(out["actions"], np.float32) if out["actions"] else np.zeros((0, 7), np.float32)
            if len(acts):
                dist = ((acts[:, None, :] - env.model.proto.detach().float().cpu().numpy()[None]) ** 2).sum(-1)
                tok35 = int(np.sum(dist.argmin(1) == 35))
            else:
                tok35 = 0
            members.append({
                "label": out["label"],
                "success": bool(out["success"]),
                "return": round(float(out["return"]), 4),
                "token_35": tok35,
            })
        returns = [m["return"] for m in members]
        successes = int(sum(m["success"] for m in members))
        span = float(max(returns) - min(returns)) if returns else 0.0
        mixed = 0 < successes < GROUP
        row = {
            "id": spec["id"],
            "family": spec["family"],
            "successes": successes,
            "returns": returns,
            "return_span": span,
            "mixed": mixed,
            "useful_advantage": bool(mixed or span >= 0.5),
            "labels": [m["label"] for m in members],
            "token_35": [m["token_35"] for m in members],
        }
        groups.append(row)
        print(" group", spec["id"], successes, round(span, 4), mixed, flush=True)
    return groups


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    base = np.asarray(saved["proto"], np.float32)
    if base.shape != (35, 7) or abs(float(saved["res_cap"]) - CAP) > 1e-9:
        raise RuntimeError(f"expected K=35 cap 0.25, got {base.shape} {saved['res_cap']}")
    audit = audit_actions(base)
    proto_new = audit.pop("proto_vector")
    print("AIR gap", audit["covered"], "L2", None if "distance_l2" not in audit else round(audit["distance_l2"], 4), flush=True)
    if not audit["covered"]:
        (RAW / "warmstart.json").write_text(json.dumps({"audit": audit, "stopped": True}), encoding="utf-8")
        print("STOP representation", flush=True)
        return
    proto = np.vstack([base, proto_new]).astype(np.float32)
    unchanged = bool(np.array_equal(proto[:35], base))
    trajs = dataset()
    table, stolen, frozen = representation_table(trajs, proto)
    print("REPR frozen", frozen, "stolen", stolen, "unchanged", unchanged, flush=True)
    for row in table:
        print(" ", row["name"], "k35", round(row["max_l2_k35"], 4), "k36", round(row["max_l2_k36"], 4), "clip", row["clipped_k36"], "t35", row["uses_token_35"], flush=True)
    payload = {
        "k_before": 35,
        "k_after": 36,
        "tokens_0_34_unchanged": unchanged,
        "residual_cap": CAP,
        "audit": audit,
        "representation": table,
        "reassigned_to_token_35": stolen,
        "frozen": bool(frozen and unchanged),
    }
    if not payload["frozen"]:
        payload["stopped"] = True
        (RAW / "warmstart.json").write_text(json.dumps(payload), encoding="utf-8")
        print("STOP freeze", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("TRAIN", device, flush=True)
    model, counts, losses = train(trajs, proto, device)
    ckpt = RAW / "ckpt.pt"
    torch.save({
        "proto": proto,
        "res_cap": CAP,
        "model": model.state_dict(),
        "k": 36,
        "token_35": proto_new,
    }, ckpt)
    print("CKPT", ckpt, flush=True)
    env = make_training_env()
    table_supports = supports_of(env.sim)
    env.model = model.to(env.device)
    env.model.eval()
    base_q, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    print("AIR", flush=True)
    air = [roll_ic(env, "air_a"), roll_ic(env, "air_b")]
    near_specs = [
        ("near_a_slow", {"speed": 0.20, "sign": 1.0, "duration_s": 0.10}),
        ("near_a_fast", {"speed": 0.30, "sign": 1.0, "duration_s": 0.08}),
        ("near_b_slow", {"speed": 0.45, "sign": -1.0, "duration_s": 0.06}),
        ("near_b_short", {"speed": 0.55, "sign": -1.0, "duration_s": 0.04}),
    ]
    near = [roll_near_air(env, base_q, q_ref, name, spec) for name, spec in near_specs]
    print("IMPACT", flush=True)
    impacts = [roll_ic(env, name) for name in ("imp_a", "imp_b", "imp_c", "imp_d")]
    print("TRAIN REST", flush=True)
    others = [roll_ic(env, name) for name in ("slip_a", "slip_b", "slip_c", "slip_d", "seat_a", "seat_b")]
    print("LEGACY", flush=True)
    legacy = [
        roll_impact_spec(env, table_supports, "impact_zp6", {"v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0}),
        roll_impact_spec(env, table_supports, "impact_zm6", {"v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0}),
        roll_offset(env, base_q, "train_dz13.5", {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.0}),
        roll_offset(env, base_q, "nl_dz14.6", {"kind": "offset", "dz": -14.6, "dx": 0.0, "dy": 0.0, "dvz": 0.0}),
        roll_offset(env, base_q, "nl_dvz_p", {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.005}),
        roll_legacy_air(env, base_q, q_ref, "air_lat_pos", 0.30),
        roll_legacy_air(env, base_q, q_ref, "air_fast_pos", 0.45),
    ]
    print("GRPO", flush=True)
    groups = grpo(env)
    payload.update({
        "stopped": False,
        "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
        "epochs": EPOCHS,
        "family_draws": counts,
        "loss_40": losses[39],
        "loss_80": losses[79],
        "loss_120": losses[-1],
        "families": {
            name: [tr["name"] for tr in trajs if tr["pool"] == name]
            for name in FAMILIES
        },
        "air": air,
        "near_air": near,
        "impacts": impacts,
        "train_rest": others,
        "legacy": legacy,
        "groups": groups,
    })
    (RAW / "warmstart.json").write_text(json.dumps(payload), encoding="utf-8")
    print("WROTE", RAW / "warmstart.json", flush=True)


if __name__ == "__main__":
    main()
