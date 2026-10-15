"""Future-state auxiliary ablation on the continual K=36 initialization.

The action interface stays token plus residual. An auxiliary head reads h_t
and predicts later observation channels. It is not used at execution.
No GRPO, SAC, DPO, new tokens, or final-test rollouts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import OBS_NAMES_AIR, SCALE_PREL, SCALE_VREL
from envs.physical_recovery import SCALE_AP
from training.autonomous_episode import make_training_env, seat_base
from training.autonomous_rl_sanity import supports_of
from experiments.legacy.balanced_unified_recovery import STEPS_PER_EPOCH
from experiments.legacy.continual_skill_addition import ContinualPolicy, copy_base, windows
from experiments.legacy.final_rl_readiness import CAP, CKPT
from models.k36_policy import FuturePolicy, G_NAMES, HORIZONS
from experiments.legacy.supervised_coverage_warmstart import (
    dataset,
    roll_ic,
    roll_impact_spec,
    roll_legacy_air,
    roll_near_air,
    roll_offset,
)

RAW = ROOT / "results" / "diagnostics" / "raw" / "future_state_ablation"
RECOV = ROOT / "results" / "diagnostics" / "raw" / "train_recoverability"
EPOCHS = 240
CKPT_EPOCHS = (40, 80, 120, 180, 240)
LAMBDAS = (0.0, 0.1, 0.5, 1.0)
DT = float(v3.DT_POLICY)
N_CONT = 13
SLIP_IDS = ("slip_a", "slip_b", "slip_d")
NL_IDS = ("train_dz13.5", "nl_dz14.6", "nl_dvz_p")
IMPACT_IDS = ("impact_zp6", "imp_c")
AIR_IDS = (
    "air_a", "air_b",
    "near_a_slow", "near_a_fast", "near_b_slow", "near_b_short",
    "air_lat_pos", "air_fast_pos",
)
TIME_IDS = ("slip_a", "slip_b", "nl_dz14.6", "nl_dvz_p", "air_a", "air_b", "imp_c")
NEAR = (
    ("near_a_slow", {"speed": 0.20, "sign": 1.0, "duration_s": 0.10}),
    ("near_a_fast", {"speed": 0.30, "sign": 1.0, "duration_s": 0.08}),
    ("near_b_slow", {"speed": 0.45, "sign": -1.0, "duration_s": 0.06}),
    ("near_b_short", {"speed": 0.55, "sign": -1.0, "duration_s": 0.04}),
)


def channel_index():
    idx = [OBS_NAMES_AIR.index(name) for name in G_NAMES]
    if idx[:3] != [27, 28, 29] or idx[12] != 12 or idx[-2:] != [2, 3]:
        raise RuntimeError(f"observation layout changed: {list(zip(G_NAMES, idx))}")
    return idx


G_IDX = channel_index()




def build_init(proto, state, device):
    model = FuturePolicy(proto).to(device)
    copy_base(model, {k: v.to(device) for k, v in state.items() if torch.is_tensor(v)})
    model.logit35_b.data.fill_(-100.0)
    torch.nn.init.zeros_(model.logit35_w)
    torch.nn.init.zeros_(model.res35.weight)
    torch.nn.init.zeros_(model.res35.bias)
    torch.nn.init.zeros_(model.fut.weight)
    torch.nn.init.zeros_(model.fut.bias)
    return model


def channel_norm(trajs):
    cols = []
    for tr in trajs:
        g = tr["obs"][:, G_IDX[:N_CONT]]
        n = len(tr["obs"])
        for h in HORIZONS:
            if n > h:
                cols.append(g[h:])
    stacked = np.concatenate(cols, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    raw = stacked.std(axis=0).astype(np.float32)
    # A near-constant channel must not dominate the normalized loss.
    std = np.maximum(raw, np.float32(0.02))
    return mean, raw, std


def pack_batch(batch, device):
    length = min(v3.MAX_LEN, max(len(item[0]) for item in batch))
    obs = torch.zeros(len(batch), length, 39, device=device)
    prev = torch.zeros(len(batch), length, 7, device=device)
    act = torch.zeros(len(batch), length, 7, device=device)
    valid = torch.zeros(len(batch), length, dtype=torch.bool, device=device)
    for i, (o, p, a) in enumerate(batch):
        n = min(length, len(o))
        obs[i, :n] = torch.tensor(o[:n], device=device)
        prev[i, :n] = torch.tensor(p[:n], device=device)
        act[i, :n] = torch.tensor(a[:n], device=device)
        valid[i, :n] = True
    return obs, prev, act, valid


def action_loss(model, obs, prev, act, valid):
    ctx = model.encode(obs, prev, ~valid)
    logits, res_old, res35 = model.heads(ctx)
    dist = ((act.unsqueeze(2) - model.proto.view(1, 1, -1, 7)) ** 2).sum(-1)
    lab = dist.argmin(-1)
    ce = torch.nn.functional.cross_entropy(logits[valid].float(), lab[valid], reduction="none")
    weight = torch.ones_like(lab, dtype=torch.float32)
    weight = torch.where(act[:, :, 2] > 0.4, weight * 4.0, weight)
    weight = torch.where(act[:, :, 6] < -0.2, weight * 4.0, weight)
    weight = torch.where(act[:, :, 4].abs() > 0.4, weight * 4.0, weight)
    w = weight[valid]
    target = act - model.proto[lab]
    use35 = (lab == 35) & valid
    use_old = (lab != 35) & valid
    res_loss = logits.new_zeros(())
    nres = 0
    if bool(use35.any()):
        res_loss = res_loss + ((res35 - target)[use35] ** 2).mean()
        nres += 1
    if bool(use_old.any()):
        res_loss = res_loss + ((res_old - target)[use_old] ** 2).mean()
        nres += 1
    loss = (ce * w).sum() / w.sum().clamp_min(1.0) + res_loss / max(nres, 1)
    return loss, ctx


def future_loss(model, ctx, obs, valid, mean, std):
    pred = model.fut(ctx).view(*obs.shape[:-1], len(HORIZONS), len(G_NAMES))
    g = obs[..., G_IDX]
    pieces, bces = [], []
    for i, h in enumerate(HORIZONS):
        if obs.shape[1] <= h:
            continue
        mask = valid[:, :-h] & valid[:, h:]
        if not bool(mask.any()):
            continue
        target = (g[:, h:, :N_CONT] - mean) / std
        err = pred[:, :-h, i, :N_CONT] - target
        pieces.append((err[mask] ** 2).mean())
        bces.append(torch.nn.functional.binary_cross_entropy_with_logits(
            pred[:, :-h, i, N_CONT:][mask].float(),
            g[:, h:, N_CONT:][mask].float(),
        ))
    if not pieces:
        return pred.new_zeros(())
    return 0.5 * torch.stack(pieces).mean() + 0.5 * torch.stack(bces).mean()


def grad_norm(params):
    total = 0.0
    for parameter in params:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5


def measure_scale(model, trajs, mean, std, device):
    groups = {}
    for i, tr in enumerate(trajs):
        groups.setdefault(tr["pool"], []).append(i)
    rng = np.random.default_rng(0)
    batch = []
    for name in groups:
        for _ in range(2):
            tr = trajs[int(rng.choice(groups[name]))]
            batch.append((tr["obs"], tr["prev"], tr["act"]))
    obs, prev, act, valid = pack_batch(batch, device)
    saved_w = model.fut.weight.detach().clone()
    saved_b = model.fut.bias.detach().clone()
    # A zero weight blocks the encoder gradient. Probe with a small map, then restore.
    torch.manual_seed(0)
    torch.nn.init.normal_(model.fut.weight, std=0.02)
    torch.nn.init.zeros_(model.fut.bias)
    model.train()
    loss_a, ctx = action_loss(model, obs, prev, act, valid)
    loss_f = future_loss(model, ctx, obs, valid, mean, std)
    model.zero_grad(set_to_none=True)
    loss_a.backward(retain_graph=True)
    g_a = grad_norm(model.enc.parameters())
    model.zero_grad(set_to_none=True)
    loss_f.backward()
    g_f = grad_norm(model.enc.parameters())
    model.zero_grad(set_to_none=True)
    model.fut.weight.data.copy_(saved_w)
    model.fut.bias.data.copy_(saved_b)
    ratio = g_f / max(g_a, 1e-8)
    loss_ratio = float(loss_f.detach()) / max(float(loss_a.detach()), 1e-8)
    # Encoder gradient through a freshly random head scales with that head's
    # weight norm, so it is not a stable tuning signal at initialization.
    # The normalized future loss is already the same order as the action loss.
    scale = 1.0 if 0.2 <= loss_ratio <= 5.0 else float(1.0 / max(loss_ratio, 1e-8))
    return {
        "loss_action": float(loss_a.detach()),
        "loss_future": float(loss_f.detach()),
        "loss_ratio_future_over_action": loss_ratio,
        "grad_action": g_a,
        "grad_future_at_probe_std_0.02": g_f,
        "grad_ratio_future_over_action": ratio,
        "applied_scale": scale,
        "note": "lambda values are unscaled because normalized future loss and action loss already match",
    }


def sample_order(trajs, rng):
    groups = {}
    for i, tr in enumerate(trajs):
        groups.setdefault(tr["pool"], []).append(i)
    names = list(groups)
    n_suffix = STEPS_PER_EPOCH * 16
    counts = {name: n_suffix // len(names) for name in names}
    counts[names[0]] += n_suffix - sum(counts.values())
    order = []
    for name in names:
        order.extend(int(i) for i in rng.choice(groups[name], size=counts[name], replace=True))
    rng.shuffle(order)
    return order


def train_one(model, trajs, mean, std, device, lam, scale, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    for parameter in model.parameters():
        parameter.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    losses = []
    snapshots = {}
    for epoch in range(EPOCHS):
        model.train()
        running_a, running_f, nbat = 0.0, 0.0, 0
        order = sample_order(trajs, rng)
        batch = []

        def flush():
            nonlocal running_a, running_f, nbat, use_amp
            if not batch:
                return
            obs, prev, act, valid = pack_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_a, ctx = action_loss(model, obs, prev, act, valid)
                # lambda 0 keeps a detached probe so future error is measurable
                # without changing the Transformer or the action head.
                ctx_f = ctx.detach() if float(lam) == 0.0 else ctx
                loss_f = future_loss(model, ctx_f, obs, valid, mean, std)
                loss = loss_a + float(lam) * float(scale) * loss_f
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
            running_a += float(loss_a.detach())
            running_f += float(loss_f.detach())
            nbat += 1
            batch.clear()

        for idx in order:
            tr = trajs[idx]
            o, p, a = tr["obs"], tr["prev"], tr["act"]
            if tr["pool"] in ("restoration", "airborne", "impact_example") and rng.random() < 0.75:
                s0 = 0
            else:
                s0 = int(rng.integers(0, max(1, len(o))))
            batch.append((o[s0:], p[s0:], a[s0:]))
            if len(batch) == 16:
                flush()
        flush()
        last_a = None if nbat == 0 else running_a / nbat
        last_f = None if nbat == 0 else running_f / nbat
        losses.append({"epoch": epoch + 1, "action": last_a, "future": last_f})
        if (epoch + 1) in CKPT_EPOCHS:
            snapshots[epoch + 1] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(" ckpt", lam, "seed", seed, "epoch", epoch + 1, "L_a", round(last_a, 4), "L_f", round(last_f, 4), flush=True)
    model.eval()
    return losses, snapshots


def teacher_metrics(model, trajs, mean, std, device):
    model.eval()
    per_pool = {}
    fut_err = {str(int(round(h * DT * 1000))): [] for h in HORIZONS}
    with torch.no_grad():
        for tr in trajs:
            for clip in windows(tr):
                obs = torch.tensor(clip["obs"][:v3.MAX_LEN], device=device)
                prev = torch.tensor(clip["prev"][:v3.MAX_LEN], device=device)
                act = torch.tensor(clip["act"][:v3.MAX_LEN], device=device)
                logits, _res, _act = model(obs, prev)
                dist = ((act.unsqueeze(1) - model.proto) ** 2).sum(-1)
                lab = dist.argmin(-1)
                hit = (logits.argmax(-1) == lab).float().cpu().numpy()
                per_pool.setdefault(tr["pool"], []).append(hit)
                pred = model.future_of(obs, prev)
                g = obs[..., G_IDX]
                for i, h in enumerate(HORIZONS):
                    if obs.shape[0] <= h:
                        continue
                    err = pred[:-h, i, :N_CONT] - (g[h:, :N_CONT] - mean) / std
                    fut_err[str(int(round(h * DT * 1000)))].append(float((err ** 2).mean().sqrt()))
    accuracy = {pool: float(np.concatenate(rows).mean()) for pool, rows in per_pool.items()}
    future = {k: (None if not v else float(np.mean(v))) for k, v in fut_err.items()}
    return accuracy, future


def physical_check(model, traj, mean, std, device):
    model.eval()
    obs_np = traj["obs"][:v3.MAX_LEN]
    obs = torch.tensor(obs_np, device=device)
    prev = torch.tensor(traj["prev"][:v3.MAX_LEN], device=device)
    mean_np = mean.detach().float().cpu().numpy()
    std_np = std.detach().float().cpu().numpy()
    with torch.no_grad():
        pred = model.future_of(obs, prev).float().cpu().numpy()
    g = obs_np[:, G_IDX]
    rows = []
    for step in (0, 4, 8):
        if step >= len(g):
            continue
        for i, h in enumerate(HORIZONS):
            if step + h >= len(g):
                continue
            true = g[step + h]
            hat = pred[step, i].copy()
            hat[:N_CONT] = hat[:N_CONT] * std_np + mean_np
            rows.append({
                "step": int(step),
                "horizon_s": round(h * DT, 2),
                "true_p_mm": [round(float(x) * SCALE_PREL * 1000.0, 2) for x in true[:3]],
                "pred_p_mm": [round(float(x) * SCALE_PREL * 1000.0, 2) for x in hat[:3]],
                "true_v_mps": [round(float(x) * SCALE_VREL, 3) for x in true[6:9]],
                "pred_v_mps": [round(float(x) * SCALE_VREL, 3) for x in hat[6:9]],
                "true_ap_mm": round(float(true[12]) * SCALE_AP * 1000.0, 2),
                "pred_ap_mm": round(float(hat[12]) * SCALE_AP * 1000.0, 2),
                "true_contact": [int(true[13] > 0.5), int(true[14] > 0.5)],
                "pred_contact": [
                    round(float(1.0 / (1.0 + np.exp(-hat[13]))), 2),
                    round(float(1.0 / (1.0 + np.exp(-hat[14]))), 2),
                ],
            })
    return rows


def expert_prefix(trajs, proto):
    proto_t = torch.tensor(np.asarray(proto, np.float32))
    rows = []
    for tr in trajs:
        act = torch.tensor(tr["act"][:24])
        lab = ((act.unsqueeze(1) - proto_t) ** 2).sum(-1).argmin(-1).tolist()
        rows.append({
            "name": tr["name"],
            "pool": tr["pool"],
            "steps": int(len(tr["act"])),
            "tokens": [int(t) for t in lab],
            "token34_run": run34(lab),
        })
    return rows


def run34(tokens):
    n = 0
    for token in tokens or []:
        if int(token) == 34:
            n += 1
        else:
            break
    return n


def evaluate(env, table, base_q, q_ref):
    rows = []
    for episode_id in ("air_a", "air_b", "slip_a", "slip_b", "slip_d", "imp_c"):
        rows.append(roll_ic(env, episode_id))
    for name, spec in NEAR:
        rows.append(roll_near_air(env, base_q, q_ref, name, spec))
    rows.append(roll_legacy_air(env, base_q, q_ref, "air_lat_pos", 0.30))
    rows.append(roll_legacy_air(env, base_q, q_ref, "air_fast_pos", 0.45))
    rows.append(roll_impact_spec(env, table, "impact_zp6", {"v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0}))
    rows.append(roll_impact_spec(env, table, "impact_zm6", {"v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0}))
    for name, spec in (
        ("train_dz13.5", {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.0}),
        ("nl_dz14.6", {"kind": "offset", "dz": -14.6, "dx": 0.0, "dy": 0.0, "dvz": 0.0}),
        ("nl_dvz_p", {"kind": "offset", "dz": -13.5, "dx": 0.0, "dy": 0.0, "dvz": 0.005}),
    ):
        rows.append(roll_offset(env, base_q, name, spec))
    slim = []
    for row in rows:
        tokens = row.get("tokens")
        slim.append({
            "id": row.get("id"),
            "label": row.get("label"),
            "success": bool(row.get("success")),
            "open_step": row.get("open_step"),
            "close_step": row.get("close_step"),
            "chase_steps": row.get("chase_steps"),
            "token_35": row.get("token_35"),
            "tokens": tokens,
            "token34_run": run34(tokens),
        })
    return slim


def counts(rows):
    by = {row["id"]: row for row in rows}

    def n_ok(ids):
        return int(sum(bool(by[i]["success"]) for i in ids if i in by))

    slip = n_ok(SLIP_IDS)
    natural = n_ok(NL_IDS)
    impact = n_ok(IMPACT_IDS)
    airborne = n_ok(AIR_IDS)
    return {
        "slip": slip,
        "natural": natural,
        "impact": impact,
        "airborne": airborne,
        "union": slip + natural + impact + airborne,
        "zm6": None if "impact_zm6" not in by else by["impact_zm6"]["label"],
    }


def main():
    if abs(DT - 0.02) > 1e-9:
        raise RuntimeError(f"policy dt {DT}")
    RAW.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    base_proto = np.asarray(saved["proto"], np.float32)
    air_a = np.asarray(np.load(RECOV / "success_air_a.npz")["act"], np.float32)
    air_b = np.asarray(np.load(RECOV / "success_air_b.npz")["act"], np.float32)
    token35 = ((air_a[1] + air_b[1]) * 0.5).astype(np.float32)
    proto = np.vstack([base_proto, token35])
    trajs = dataset()
    mean_np, raw_np, std_np = channel_norm(trajs)
    mean = torch.tensor(mean_np, device=device)
    std = torch.tensor(std_np, device=device)
    probe = build_init(proto, saved["model"], device)
    scale_info = measure_scale(probe, trajs, mean, std, device)
    del probe
    print("SCALE", {k: round(v, 4) if isinstance(v, float) else v for k, v in scale_info.items()}, flush=True)
    print("CHANNELS", list(zip(G_NAMES, G_IDX)), flush=True)
    env = make_training_env()
    table = supports_of(env.sim)
    base_q, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    init_model = build_init(proto, saved["model"], device)
    init_model.eval()
    env.model = init_model
    init_closed = evaluate(env, table, base_q, q_ref)
    payload_init = {
        "counts": counts(init_closed),
        "timing": [
            {
                "id": row["id"],
                "tokens": row["tokens"],
                "token34_run": row["token34_run"],
                "label": row["label"],
            }
            for row in init_closed
            if row["id"] in TIME_IDS
        ],
    }
    del init_model
    path = RAW / "ablation.json"
    payload = {
        "base_checkpoint": str(Path(CKPT).relative_to(ROOT)).replace("\\", "/"),
        "channels": list(G_NAMES),
        "channel_index": G_IDX,
        "horizons_s": [round(h * DT, 2) for h in HORIZONS],
        "std_raw": [round(float(x), 4) for x in raw_np],
        "std_continuous": [round(float(x), 4) for x in std_np],
        "mean_continuous": [round(float(x), 4) for x in mean_np],
        "scales": {
            "p_rel_m": SCALE_PREL,
            "v_rel_mps": SCALE_VREL,
            "aperture_m": SCALE_AP,
        },
        "scale": scale_info,
        "init": payload_init,
        "expert_tokens": expert_prefix(trajs, proto),
        "runs": [],
        "physical": [],
    }

    def dump():
        path.write_text(json.dumps(payload), encoding="utf-8")

    def run_job(seed, lam):
        print("RUN", lam, "seed", seed, flush=True)
        model = build_init(proto, saved["model"], device)
        losses, snapshots = train_one(model, trajs, mean, std, device, lam, scale_info["applied_scale"], seed)
        for epoch, snap in snapshots.items():
            model.load_state_dict({k: v.to(device) for k, v in snap.items()})
            model.eval()
            acc, fut = teacher_metrics(model, trajs, mean, std, device)
            env.model = model
            closed = evaluate(env, table, base_q, q_ref)
            summary = counts(closed)
            timing = []
            for row in closed:
                if row["id"] in TIME_IDS:
                    timing.append({
                        "id": row["id"],
                        "tokens": row["tokens"],
                        "token34_run": row["token34_run"],
                        "open_step": row["open_step"],
                        "chase_steps": row["chase_steps"],
                        "close_step": row["close_step"],
                        "token_35": row["token_35"],
                        "label": row["label"],
                    })
            payload["runs"].append({
                "lambda": lam,
                "seed": seed,
                "epoch": epoch,
                "action_loss": losses[epoch - 1]["action"],
                "future_loss": losses[epoch - 1]["future"],
                "token_accuracy": acc,
                "future_rmse": fut,
                "counts": summary,
                "timing": timing,
                "closed": [{k: v for k, v in row.items() if k != "tokens"} for row in closed],
            })
            dump()
            print(" EVAL", lam, seed, epoch, summary, flush=True)
        if seed == 0 and lam in (0.0, 1.0):
            model.load_state_dict({k: v.to(device) for k, v in snapshots[240].items()})
            checks = []
            for tr in trajs:
                if tr["name"] in ("success_air_a", "demo_nl_dz14.6", "demo_train_dz13.5"):
                    checks.append({"name": tr["name"], "rows": physical_check(model, tr, mean, std, device)})
            payload["physical"].append({"lambda": lam, "seed": seed, "epoch": 240, "checks": checks})
            dump()
        torch.save(
            {"proto": proto, "res_cap": CAP, "lambda": lam, "seed": seed, "model": snapshots[240]},
            RAW / f"ckpt_l{lam}_s{seed}.pt",
        )
        del snapshots

    for lam in LAMBDAS:
        run_job(0, lam)
    future = [rec for rec in payload["runs"] if rec["seed"] == 0 and rec["lambda"] > 0]
    best = max(future, key=lambda rec: (rec["counts"]["union"], rec["counts"]["airborne"], rec["counts"]["slip"], -rec["lambda"]))
    payload["best_future_lambda"] = float(best["lambda"])
    dump()
    print("BEST", payload["best_future_lambda"], flush=True)
    for seed in (1, 2):
        run_job(seed, 0.0)
        if payload["best_future_lambda"] != 0.0:
            run_job(seed, payload["best_future_lambda"])
    dump()
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
