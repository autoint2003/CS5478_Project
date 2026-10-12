"""Add the airborne token on top of the K=35 policy without a from-scratch fit.

Old token centers and old weights are copied. Stage A trains only the new token
head. Later stages unfreeze outputs, then the last Transformer layer, and stop
at the first stage whose closed loop recovers airborne. No GRPO update.
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
from training.autonomous_episode import make_training_env, seat_base
from training.autonomous_rl_sanity import supports_of
from training.final_rl_readiness import CAP, CATALOG, CKPT, GROUP, load_ic, rollout
from training.history_aware_critic_ablation import restore_capture
from training.supervised_coverage_warmstart import (
    RECOV,
    dataset,
    load_traj,
    roll_ic,
    roll_impact_spec,
    roll_legacy_air,
    roll_near_air,
    roll_offset,
)

RAW = ROOT / "results" / "diagnostics" / "raw" / "continual_skill_addition"
SCRATCH = ROOT / "results" / "diagnostics" / "raw" / "supervised_coverage_warmstart" / "ckpt.pt"
EPOCHS = 60
BATCHES = 40
MARGIN = 2.0
NEAR = (
    ("near_a_slow", {"speed": 0.20, "sign": 1.0, "duration_s": 0.10}),
    ("near_a_fast", {"speed": 0.30, "sign": 1.0, "duration_s": 0.08}),
    ("near_b_slow", {"speed": 0.45, "sign": -1.0, "duration_s": 0.06}),
    ("near_b_short", {"speed": 0.55, "sign": -1.0, "duration_s": 0.04}),
)


class ContinualPolicy(torch.nn.Module):
    """K=36 policy. Token 35 has its own logit and residual. Tokens 0-34 share the copied head."""

    def __init__(self, proto):
        super().__init__()
        self.register_buffer("proto", torch.tensor(np.asarray(proto, np.float32)))
        d = 64
        self.in_proj = torch.nn.Linear(39 + 7, d)
        self.pos = torch.nn.Embedding(160, d)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128, dropout=0.0,
            batch_first=True, activation="relu",
        )
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.logits_old = torch.nn.Linear(d, 35)
        self.logit35_w = torch.nn.Parameter(torch.zeros(d))
        self.logit35_b = torch.nn.Parameter(torch.tensor(-100.0))
        self.res = torch.nn.Linear(d, 7)
        self.res35 = torch.nn.Linear(d, 7)
        torch.nn.init.zeros_(self.res.weight)
        torch.nn.init.zeros_(self.res.bias)
        torch.nn.init.zeros_(self.res35.weight)
        torch.nn.init.zeros_(self.res35.bias)
        self.res_cap = float(CAP)

    def encode(self, obs, prev, pad_mask=None):
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[-2]
        h = self.in_proj(x) + self.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
        if x.dim() == 2:
            ctx = self.enc(h.unsqueeze(0), mask=causal)[0]
        else:
            ctx = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        return ctx

    def heads(self, ctx):
        old = self.logits_old(ctx)
        extra = (ctx * self.logit35_w).sum(-1) + self.logit35_b
        logits = torch.cat([old, extra.unsqueeze(-1)], dim=-1)
        res_old = self.res_cap * torch.tanh(self.res(ctx))
        res35 = self.res_cap * torch.tanh(self.res35(ctx))
        return logits, res_old, res35

    def forward(self, obs, prev, pad_mask=None):
        ctx = self.encode(obs, prev, pad_mask)
        logits, res_old, res35 = self.heads(ctx)
        tok = torch.argmax(logits, dim=-1)
        res = torch.where(tok.unsqueeze(-1) == 35, res35, res_old)
        act = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
        self.last_res_old = res_old
        self.last_res35 = res35
        return logits, res, act


def copy_base(model, state):
    own = model.state_dict()
    mapped = {
        "in_proj.weight": "in_proj.weight",
        "in_proj.bias": "in_proj.bias",
        "pos.weight": "pos.weight",
        "logits_old.weight": "logits.weight",
        "logits_old.bias": "logits.bias",
        "res.weight": "res.weight",
        "res.bias": "res.bias",
    }
    with torch.no_grad():
        for dst, src in mapped.items():
            own[dst].copy_(state[src])
        for key, value in state.items():
            if key.startswith("enc."):
                own[key].copy_(value)
    missing = [k for k in state if k not in ("logits.weight", "logits.bias", "proto") and k not in own and not k.startswith("enc.")]
    if missing:
        raise RuntimeError(f"unmapped base weights {missing}")


def set_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.logit35_w.requires_grad = True
    model.logit35_b.requires_grad = True
    for parameter in model.res35.parameters():
        parameter.requires_grad = True
    if stage in ("B", "C"):
        for parameter in model.logits_old.parameters():
            parameter.requires_grad = True
        for parameter in model.res.parameters():
            parameter.requires_grad = True
    if stage == "C":
        for parameter in model.enc.layers[1].parameters():
            parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def windows(tr):
    n = len(tr["obs"])
    spans = [(0, min(n, 160))]
    if n > 160:
        spans.append((n - 160, n))
    clips = []
    for start, end in spans:
        name = tr["name"] if start == 0 else tr["name"] + "_tail"
        clips.append({
            "name": name,
            "pool": tr["pool"],
            "obs": tr["obs"][start:end],
            "prev": tr["prev"][start:end],
            "act": tr["act"][start:end],
        })
    return clips


def action_gap(base, extended, trajs, device):
    rows = []
    base.eval()
    extended.eval()
    for tr in trajs:
        obs = torch.tensor(tr["obs"], device=device)
        prev = torch.tensor(tr["prev"], device=device)
        with torch.no_grad():
            logits_b, res_b, act_b = base(obs, prev)
            logits_e, res_e, act_e = extended(obs, prev)
        tok_b = logits_b.argmax(-1)
        tok_e = logits_e.argmax(-1)
        diff = (act_b - act_e).abs()
        rows.append({
            "name": tr["name"],
            "max_abs": float(diff.max()),
            "mean_l2": float(torch.linalg.norm(act_b - act_e, dim=-1).mean()),
            "token_match": float((tok_b == tok_e).float().mean()),
            "token_35_frac": float((tok_e == 35).float().mean()),
            "residual_l2": float(torch.linalg.norm(res_b - res_e, dim=-1).mean()),
        })
    return rows


def drift_report(base, extended, trajs, device):
    rows = []
    base.eval()
    extended.eval()
    for tr in trajs:
        obs = torch.tensor(tr["obs"], device=device)
        prev = torch.tensor(tr["prev"], device=device)
        with torch.no_grad():
            logits_b, res_b, act_b = base(obs, prev)
            logits_e, _res_e, act_e = extended(obs, prev)
            prob_b = torch.softmax(logits_b.float(), dim=-1)
            prob_e = torch.softmax(logits_e.float(), dim=-1)
            padded = torch.zeros_like(prob_e)
            padded[..., :35] = prob_b
            kl = (padded * (padded.clamp_min(1e-8).log() - prob_e.clamp_min(1e-8).log())).sum(-1)
        tok_b = logits_b.argmax(-1)
        tok_e = logits_e.argmax(-1)
        rows.append({
            "name": tr["name"],
            "pool": tr["pool"],
            "token_match": float((tok_b == tok_e).float().mean()),
            "token_35_frac": float((tok_e == 35).float().mean()),
            "mean_kl": float(kl.mean()),
            "action_mean_l2": float(torch.linalg.norm(act_b - act_e, dim=-1).mean()),
            "action_max_l2": float(torch.linalg.norm(act_b - act_e, dim=-1).max()),
            "residual_mean_l2": float(torch.linalg.norm(res_b - extended.last_res_old, dim=-1).mean()),
        })
    return rows


def train_stage(model, base, air, old, device, stage):
    names = set_stage(model, stage)
    opt = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3 if stage == "A" else 3e-4)
    rng = np.random.default_rng(0)
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        running = 0.0
        for _batch in range(BATCHES):
            kind = "air" if _batch % 2 == 0 else "old"
            pool = air if kind == "air" else old
            batch = []
            for _ in range(16):
                tr = pool[int(rng.integers(0, len(pool)))]
                n = len(tr["act"])
                s0 = 0 if n <= 40 or rng.random() < 0.75 else int(rng.integers(0, n))
                batch.append((tr["obs"][s0:], tr["prev"][s0:], tr["act"][s0:]))
            length = min(v3.MAX_LEN, max(item[0].shape[0] for item in batch))
            obs = torch.zeros(len(batch), length, 39, device=device)
            prev = torch.zeros(len(batch), length, 7, device=device)
            act = torch.zeros(len(batch), length, 7, device=device)
            valid = torch.zeros(len(batch), length, dtype=torch.bool, device=device)
            for i, (o, p, a) in enumerate(batch):
                n = min(length, o.shape[0])
                obs[i, :n] = torch.tensor(o[:n], device=device)
                prev[i, :n] = torch.tensor(p[:n], device=device)
                act[i, :n] = torch.tensor(a[:n], device=device)
                valid[i, :n] = True
            opt.zero_grad(set_to_none=True)
            ctx = model.encode(obs, prev, ~valid)
            logits, res_old, res35 = model.heads(ctx)
            if kind == "air":
                dist = ((act.unsqueeze(2) - model.proto.view(1, 1, -1, 7)) ** 2).sum(-1)
                lab = dist.argmin(-1)
                ce = torch.nn.functional.cross_entropy(logits[valid].float(), lab[valid])
                target = act - model.proto[lab]
                use35 = (lab == 35) & valid
                use_old = (lab != 35) & valid
                parts = []
                if bool(use35.any()) and model.res35.weight.requires_grad:
                    parts.append(((res35 - target)[use35] ** 2).mean())
                if bool(use_old.any()) and model.res.weight.requires_grad:
                    parts.append(((res_old - target)[use_old] ** 2).mean())
                loss = ce + (parts[0] if parts else 0.0)
                if len(parts) == 2:
                    loss = ce + parts[0] + parts[1]
            else:
                with torch.no_grad():
                    base_logits, base_res, _base_act = base(obs, prev)
                    base_tok = base_logits.argmax(-1)
                ce = torch.nn.functional.cross_entropy(logits[:, :, :35][valid].float(), base_tok[valid])
                hinge = torch.relu(logits[:, :, 35] - logits[:, :, :35].max(-1).values + MARGIN)
                hinge = hinge[valid].mean()
                res_pen = ((res_old - base_res)[valid] ** 2).mean() if model.res.weight.requires_grad else 0.0
                loss = ce + hinge + res_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            opt.step()
            running += float(loss.detach())
        losses.append(running / BATCHES)
        if (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(" ", stage, "epoch", epoch + 1, "loss", round(losses[-1], 4), flush=True)
    model.eval()
    return names, losses


def suite(env, table, base_q, q_ref, tag):
    print(tag, flush=True)
    rows = []
    for episode_id in ("air_a", "air_b", "imp_a", "imp_b", "imp_c", "imp_d", "slip_a", "slip_b", "slip_c", "slip_d", "seat_a", "seat_b"):
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
    return [{k: v for k, v in row.items() if k != "tokens" or True} for row in rows]


def air_ok(rows):
    labels = {row["id"]: row.get("label") for row in rows}
    return bool(success.is_success(labels.get("air_a", "")) and success.is_success(labels.get("air_b", "")))


def retained(base_rows, rows):
    keys = ("nl_dz14.6", "nl_dvz_p", "train_dz13.5", "slip_a", "slip_b", "slip_c", "slip_d", "imp_c", "impact_zp6", "impact_zm6")
    base_map = {row["id"]: row.get("label") for row in base_rows}
    new_map = {row["id"]: row.get("label") for row in rows}
    lost = []
    for key in keys:
        if success.is_success(base_map.get(key, "")) and not success.is_success(new_map.get(key, "")):
            lost.append(key)
    return lost


def trace_case(env, model, kind, name):
    """First 24 recovery steps, for the interference comparison."""
    from training.nominal_pose_audit import errors
    from training.autonomous_episode import Z_ABORT
    import training.natural_loss_recatch_transformer_v3 as v3
    env.model = model.to(env.device)
    env.model.eval()
    if kind == "ic":
        blob = load_ic(name)
        restore_capture(env, blob["capture"])
    else:
        raise KeyError(kind)
    tokens, acts = [], []
    for _ in range(24):
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, res, act = env.model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=env.device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=env.device),
            )
        token = int(torch.argmax(logits[-1]).item())
        clipped = np.clip(act[-1].float().cpu().numpy(), -1.0, 1.0)
        top = torch.topk(logits[-1].float(), k=min(3, logits.shape[-1]))
        tokens.append({
            "token": token,
            "margin": float(top.values[0] - top.values[1]) if top.values.numel() > 1 else 0.0,
            "top": [int(x) for x in top.indices.cpu()],
            "res_l2": float(torch.linalg.norm(res[-1].float()).cpu()),
        })
        acts.append(clipped.copy())
        v3.step_v3(env.sim, clipped.astype(np.float32))
        env._prev = clipped.astype(np.float32)
        if errors(env.sim, blob["ref"])["z"] < Z_ABORT:
            break
    return {"tokens": tokens, "act": np.stack(acts)}


def compare_traces(env, models):
    out = []
    for name in ("slip_a", "slip_b", "slip_d", "imp_c"):
        traces = {key: trace_case(env, model, "ic", name) for key, model in models.items()}
        n = min(len(traces[k]["act"]) for k in traces)
        def first_diff(a, b):
            for i in range(n):
                if traces[a]["tokens"][i]["token"] != traces[b]["tokens"][i]["token"]:
                    return {"step": i, "token_a": traces[a]["tokens"][i]["token"], "token_b": traces[b]["tokens"][i]["token"]}
                if float(np.linalg.norm(traces[a]["act"][i] - traces[b]["act"][i])) > 0.05:
                    return {"step": i, "token_a": traces[a]["tokens"][i]["token"], "token_b": traces[b]["tokens"][i]["token"], "action_l2": float(np.linalg.norm(traces[a]["act"][i] - traces[b]["act"][i]))}
            return None
        out.append({
            "id": name,
            "base_tokens": [row["token"] for row in traces["base"]["tokens"][:12]],
            "scratch_tokens": [row["token"] for row in traces["scratch"]["tokens"][:12]],
            "continual_tokens": [row["token"] for row in traces["continual"]["tokens"][:12]],
            "base_vs_scratch": first_diff("base", "scratch"),
            "base_vs_continual": first_diff("base", "continual"),
        })
        print(" trace", name, out[-1]["base_vs_scratch"], out[-1]["base_vs_continual"], flush=True)
    return out


def grpo_continual(env):
    from training.final_rl_readiness import SIGMA
    groups = []
    for spec in CATALOG:
        if spec["split"] != "train":
            continue
        blob = load_ic(spec["id"])
        members = []
        for g in range(GROUP):
            restore_capture(env, blob["capture"])
            rng = np.random.default_rng(2000 + 17 * g + sum(ord(c) for c in spec["id"]))

            def action_fn(rng=rng):
                obs = np.asarray(env.observation(), np.float32)
                env._obs_seq.append(obs)
                env._prev_seq.append(env._prev.copy())
                with torch.no_grad():
                    logits, _res, _act = env.model(
                        torch.tensor(np.stack(env._obs_seq[-160:]), device=env.device),
                        torch.tensor(np.stack(env._prev_seq[-160:]), device=env.device),
                    )
                probs = torch.softmax(logits[-1].float(), dim=-1).cpu().numpy()
                probs = np.clip(probs, 0.0, None)
                probs = probs / probs.sum()
                token = int(rng.choice(len(probs), p=probs))
                resid = env.model.last_res35[-1] if token == 35 else env.model.last_res_old[-1]
                resid = resid.float().cpu().numpy() + rng.normal(0.0, SIGMA, size=7).astype(np.float32)
                resid = np.clip(resid, -CAP, CAP)
                proto = env.model.proto[token].float().cpu().numpy()
                clipped = np.clip(proto + resid, -1.0, 1.0).astype(np.float32)
                v3.step_v3(env.sim, clipped)
                env._prev = clipped
                return token, clipped

            out = rollout(env, blob["ref"], action_fn, 100, {"audit": None})
            members.append({"label": out["label"], "success": bool(out["success"]), "return": round(float(out["return"]), 4)})
        returns = [m["return"] for m in members]
        successes = int(sum(m["success"] for m in members))
        span = float(max(returns) - min(returns)) if returns else 0.0
        mixed = 0 < successes < GROUP
        groups.append({
            "id": spec["id"],
            "family": spec["family"],
            "successes": successes,
            "returns": returns,
            "return_span": span,
            "mixed": mixed,
            "useful_advantage": bool(mixed or span >= 0.5),
            "labels": [m["label"] for m in members],
        })
        print(" group", spec["id"], successes, round(span, 4), mixed, flush=True)
    return groups


def slim(rows):
    keep = ("id", "label", "success", "basin", "handoff", "held_10", "held_12", "entry_ep_mm", "entry_vrel", "open_step", "close_step", "chase_steps", "token_35", "entered")
    out = []
    for row in rows:
        out.append({k: row.get(k) for k in keep if k in row or row.get(k) is not None})
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    base_proto = np.asarray(saved["proto"], np.float32)
    if base_proto.shape != (35, 7):
        raise RuntimeError(f"expected K=35, got {base_proto.shape}")
    air_a = np.asarray(np.load(RECOV / "success_air_a.npz")["act"], np.float32)
    air_b = np.asarray(np.load(RECOV / "success_air_b.npz")["act"], np.float32)
    token35 = ((air_a[1] + air_b[1]) * 0.5).astype(np.float32)
    proto = np.vstack([base_proto, token35])
    if not np.array_equal(proto[:35], base_proto):
        raise RuntimeError("token centers 0-34 changed")
    base = build_policy(base_proto, device, CAP)
    base.load_state_dict(saved["model"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    model = ContinualPolicy(proto).to(device)
    copy_base(model, {k: v.to(device) for k, v in saved["model"].items() if torch.is_tensor(v)})
    model.eval()
    trajs = dataset()
    old = [tr for tr in trajs if tr["pool"] != "airborne"]
    air = [tr for tr in trajs if tr["pool"] == "airborne"]
    before = action_gap(base, model, [clip for tr in old for clip in windows(tr)], device)
    max_abs = max(row["max_abs"] for row in before)
    print("IDENTITY", max_abs, flush=True)
    payload = {
        "base_checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "token_35": [round(float(x), 6) for x in token35],
        "identity_before_training": before,
        "identity_max_abs": max_abs,
    }
    if max_abs > 1e-5:
        payload["stopped"] = "append_changes_actions"
        (RAW / "continual.json").write_text(json.dumps(payload), encoding="utf-8")
        print("STOP identity", flush=True)
        return
    env = make_training_env()
    table = supports_of(env.sim)
    env.model = base
    base_q, q_ref = seat_base(env.sim, env.nominal, env.pads, env.pad_mu)
    print("BASE", flush=True)
    base_rows = suite(env, table, base_q, q_ref, "base")
    payload["base_closed_loop"] = slim(base_rows)
    stages = []
    selected = None
    for stage in ("A", "B", "C"):
        print("TRAIN", stage, flush=True)
        names, losses = train_stage(model, base, air, old, device, stage)
        drift = drift_report(base, model, [clip for tr in old for clip in windows(tr)], device)
        env.model = model.to(env.device)
        env.model.eval()
        rows = suite(env, table, base_q, q_ref, stage)
        lost = retained(base_rows, rows)
        record = {
            "stage": stage,
            "trainable": names,
            "loss_last": losses[-1],
            "drift": drift,
            "drift_max_action_l2": max(row["action_max_l2"] for row in drift),
            "closed_loop": slim(rows),
            "air_success": air_ok(rows),
            "lost_successes": lost,
        }
        stages.append(record)
        torch.save({"proto": proto, "res_cap": CAP, "model": model.state_dict(), "k": 36, "stage": stage}, RAW / f"ckpt_stage_{stage}.pt")
        print(stage, "air", record["air_success"], "lost", lost, "drift", round(record["drift_max_action_l2"], 4), flush=True)
        if record["air_success"]:
            selected = stage
            break
    payload["stages"] = stages
    payload["selected_stage"] = selected
    if selected is not None and not stages[-1]["lost_successes"]:
        print("GRPO", flush=True)
        env.model = model.to(env.device)
        payload["groups"] = grpo_continual(env)
    else:
        payload["groups"] = None
    scratch = torch.load(SCRATCH, map_location="cpu", weights_only=False)
    scratch_proto = np.asarray(scratch["proto"], np.float32)
    scratch_model = build_policy(scratch_proto, device, CAP)
    scratch_model.load_state_dict(scratch["model"])
    scratch_model.eval()
    print("TRACE", flush=True)
    payload["traces"] = compare_traces(env, {"base": base, "scratch": scratch_model, "continual": model})
    (RAW / "continual.json").write_text(json.dumps(payload), encoding="utf-8")
    print("WROTE", RAW / "continual.json", flush=True)


if __name__ == "__main__":
    main()
