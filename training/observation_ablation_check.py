"""Retrain the balanced Transformer on observation subsets. Same data, seed, and 24 cases."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import training.natural_loss_recatch_data_scaling as scale
import training.natural_loss_recatch_transformer_v3 as v3
from training.balanced_unified_recovery import EPOCHS, POOLS, STEPS_PER_EPOCH
from training.observation_robustness_check import conditions
from training.observation_study_common import (
    build_cases,
    collect_train,
    family_counts,
    load_balanced,
    run_suite,
)

OUT = ROOT / "results" / "diagnostics" / "raw" / "observation_ablation"
REPORT = ROOT / "results" / "diagnostics" / "OBSERVATION_ABLATION_CHECK.md"
FAMS = ("impact", "sliding", "wrist", "recatch", "airborne", "natural")

GROUPS = {
    "0-26 tactile and proprioception": list(range(0, 27)),
    "27-29 relative position": [27, 28, 29],
    "30-32 relative linear velocity": [30, 31, 32],
    "33-35 relative orientation": [33, 34, 35],
    "36-38 relative angular velocity": [36, 37, 38],
    "39-44 wide-scale hand twist": [39, 40, 41, 42, 43, 44],
}


def indices_except(drop):
    drop = set(drop)
    return [i for i in range(45) if i not in drop]


VARIANTS = [
    ("A0", "full 45-D", list(range(45))),
    ("A1", "drop wide-scale hand twist", indices_except(range(39, 45))),
    ("A2", "drop object angular velocity", indices_except(range(36, 39))),
    ("A3", "drop object orientation", indices_except(range(33, 36))),
    ("A4", "drop object linear and angular velocity", indices_except(list(range(30, 33)) + list(range(36, 39)))),
    ("A5", "drop object position and orientation", indices_except(list(range(27, 30)) + list(range(33, 36)))),
    ("A6", "27-D tactile and proprioception only", list(range(27))),
]


def build_policy(proto, device, obs_dim):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("proto", torch.tensor(np.asarray(proto, np.float32)))
            d = 64
            self.in_proj = torch.nn.Linear(obs_dim + 7, d)
            self.pos = torch.nn.Embedding(160, d)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=128, dropout=0.0,
                batch_first=True, activation="relu",
            )
            self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.logits = torch.nn.Linear(d, len(proto))
            self.res = torch.nn.Linear(d, 7)
            torch.nn.init.zeros_(self.res.weight)
            torch.nn.init.zeros_(self.res.bias)

        def forward(self, obs, prev, pad_mask=None):
            x = torch.cat([obs, prev], dim=-1)
            t = x.shape[-2]
            h = self.in_proj(x) + self.pos(torch.arange(t, device=x.device))
            causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
            if x.dim() == 2:
                ctx = self.enc(h.unsqueeze(0), mask=causal)[0]
            else:
                ctx = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
            logits = self.logits(ctx)
            res = 0.25 * torch.tanh(self.res(ctx))
            tok = torch.argmax(logits, dim=-1)
            pred = torch.clamp(self.proto[tok] + res, -1.0, 1.0)
            return logits, res, pred

    return Policy().to(device)


def train_variant(trajs, proto, device, obs_dim):
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)
    model = build_policy(proto, device, obs_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)
    groups = {k: [] for k in POOLS}
    bank = []
    for tr in trajs:
        groups[tr["pool"]].append(len(bank))
        bank.append((torch.tensor(tr["obs"]), torch.tensor(tr["prev"]), torch.tensor(tr["act"])))
    present = {k: v for k, v in groups.items() if v}
    weight = 1.0 / len(present)
    n_suffix = STEPS_PER_EPOCH * 16
    counts = {k: int(round(weight * n_suffix)) for k in present}
    counts[max(counts, key=counts.get)] += n_suffix - sum(counts.values())
    amp_on = use_amp
    last = None
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        order = []
        for name, n_draw in counts.items():
            choice = rng.choice(groups[name], size=n_draw, replace=True)
            order.extend(int(i) for i in choice)
        rng.shuffle(order)
        running = 0.0
        nbat = 0
        batch = []

        def flush():
            nonlocal running, nbat, amp_on
            if not batch:
                return
            length = min(v3.MAX_LEN, max(b[0].shape[0] for b in batch))
            nbatch = len(batch)
            obs = torch.zeros(nbatch, length, obs_dim)
            prev = torch.zeros(nbatch, length, 7)
            act = torch.zeros(nbatch, length, 7)
            valid = torch.zeros(nbatch, length, dtype=torch.bool)
            for i, (o, p, a) in enumerate(batch):
                n = min(length, o.shape[0])
                obs[i, :n] = o[:n]
                prev[i, :n] = p[:n]
                act[i, :n] = a[:n]
                valid[i, :n] = True
            obs = obs.pin_memory().to(device, non_blocking=True)
            prev = prev.pin_memory().to(device, non_blocking=True)
            act = act.pin_memory().to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
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
            o, p, a = bank[idx]
            s0 = int(rng.integers(0, o.shape[0]))
            batch.append((o[s0:], p[s0:], a[s0:]))
            if len(batch) == 16:
                flush()
        flush()
        last = None if nbat == 0 else running / nbat
        if (epoch + 1) % 40 == 0 or epoch + 1 == EPOCHS:
            print("epoch", epoch + 1, "train", None if last is None else round(last, 4), flush=True)
    model.eval()
    return model, last, round(time.perf_counter() - t0, 1)


def slice_trajs(trajs, columns):
    cols = list(columns)
    out = []
    for tr in trajs:
        item = dict(tr)
        item["obs"] = np.asarray(tr["obs"], np.float32)[:, cols]
        out.append(item)
    return out


def keeps_skills(fam, total):
    return (
        fam["sliding"] >= 4
        and fam["wrist"] >= 4
        and fam["recatch"] >= 4
        and fam["impact"] >= 3
        and total >= 14
    )


def behavior_lines(base_rows, rows):
    base = {r["id"]: r for r in base_rows}
    lines = []
    for r in rows:
        b = base[r["id"]]
        if b["t12"] == "RETAINED_TO_12" and r["t12"] != "RETAINED_TO_12":
            lines.append(
                f"{r['id']} ({r['family']}) {b['t12']} -> {r['t12']}. "
                f"Clean mechanism {b['mechanism']}. "
                f"mean |vx| {b['mean_abs_vx']:.3f}->{r['mean_abs_vx']:.3f}, "
                f"mean |vy| {b['mean_abs_vy']:.3f}->{r['mean_abs_vy']:.3f}, "
                f"mean vz {b['mean_vz']:.3f}->{r['mean_vz']:.3f}, "
                f"max |wy| {b['max_abs_wy']:.3f}->{r['max_abs_wy']:.3f}, "
                f"min grip {b['min_grip']:.3f}->{r['min_grip']:.3f}."
            )
    return lines


def choose_compact(by_name):
    """Drop a group only when the ablation that removes it still holds the established skills."""
    a1 = by_name["A1"]
    a2 = by_name["A2"]
    a3 = by_name["A3"]
    a4 = by_name["A4"]
    a5 = by_name["A5"]
    keep_hand = not keeps_skills(a1["families"], a1["ok"])
    keep_w = not keeps_skills(a2["families"], a2["ok"])
    keep_rot = not keeps_skills(a3["families"], a3["ok"])
    keep_v = not keeps_skills(a4["families"], a4["ok"])
    keep_p = not keeps_skills(a5["families"], a5["ok"])
    cols = list(range(27))
    if keep_p:
        cols += [27, 28, 29]
    cols += [30, 31, 32] if keep_v else []
    if keep_rot:
        cols += [33, 34, 35]
    if keep_w:
        cols += [36, 37, 38]
    if keep_hand:
        cols += [39, 40, 41, 42, 43, 44]
    cols = sorted(cols)
    reason = {
        "keep_position": keep_p,
        "keep_linear_velocity": keep_v,
        "keep_orientation": keep_rot,
        "keep_angular_velocity": keep_w,
        "keep_wide_hand_twist": keep_hand,
    }
    return cols, reason


def main():
    raise SystemExit(
        "This ablation is a finished study of the retired 45-D observation. "
        "The active policy observation is 39-D. "
        "training/final_39d_balanced_policy.py is the trainer."
    )
    print("build cases and training trajectories", flush=True)
    ctx = build_cases()
    trajs = collect_train(ctx)
    print("trajs", len(trajs), {p: sum(t["pool"] == p for t in trajs) for p in POOLS}, flush=True)
    model0, proto, _saved, device = load_balanced()
    hold = ctx["eval_natural"]
    results = {}
    cache_path = OUT / "variants.json"
    if cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))

    def remember(tag, payload):
        results[tag] = payload
        cache_path.write_text(json.dumps(results), encoding="utf-8")

    if "A0" not in results:
        ev = run_suite(ctx, model0, {}, 0)
        val = scale.predict_scores(model0, hold, device)
        remember("A0", {
            "label": "full 45-D",
            "columns": list(range(45)),
            "dim": 45,
            "ok": ev["ok"],
            "families": family_counts(ev["rows"]),
            "val_loss": val["loss"],
            "train_loss": None,
            "seconds": 0,
            "rows": ev["rows"],
            "retrained": False,
        })
        print("A0", ev["ok"], val["loss"], flush=True)
    if results["A0"]["ok"] != 15:
        raise SystemExit(f"A0 scored {results['A0']['ok']}/24")

    for tag, label, cols in VARIANTS:
        if tag == "A0" or tag in results:
            continue
        print("TRAIN", tag, len(cols), flush=True)
        sliced = slice_trajs(trajs, cols)
        model, loss, seconds = train_variant(sliced, proto, device, len(cols))
        ev = run_suite(ctx, model, {}, 0, columns=cols)
        val = scale.predict_scores(model, slice_trajs(hold, cols), device)
        torch.save({"columns": cols, "model": model.state_dict(), "train_loss": loss}, OUT / f"{tag}.pt")
        remember(tag, {
            "label": label,
            "columns": cols,
            "dim": len(cols),
            "ok": ev["ok"],
            "families": family_counts(ev["rows"]),
            "val_loss": val["loss"],
            "train_loss": None if loss is None else round(float(loss), 4),
            "seconds": seconds,
            "rows": ev["rows"],
            "retrained": True,
        })
        print(tag, ev["ok"], family_counts(ev["rows"]), "val", val["loss"], flush=True)

    cols, reason = choose_compact(results)
    print("compact", cols, reason, flush=True)
    known = {tuple(results[k]["columns"]): k for k in results}
    if tuple(cols) in known:
        results["compact"] = dict(results[known[tuple(cols)]])
        results["compact"]["label"] = "compact candidate (same as " + known[tuple(cols)] + ")"
        results["compact"]["alias"] = known[tuple(cols)]
    elif "compact" not in results:
        print("TRAIN compact", len(cols), flush=True)
        model, loss, seconds = train_variant(slice_trajs(trajs, cols), proto, device, len(cols))
        ev = run_suite(ctx, model, {}, 0, columns=cols)
        val = scale.predict_scores(model, slice_trajs(hold, cols), device)
        torch.save({"columns": cols, "model": model.state_dict(), "train_loss": loss}, OUT / "compact.pt")
        remember("compact", {
            "label": "compact candidate",
            "columns": cols,
            "dim": len(cols),
            "ok": ev["ok"],
            "families": family_counts(ev["rows"]),
            "val_loss": val["loss"],
            "train_loss": None if loss is None else round(float(loss), 4),
            "seconds": seconds,
            "rows": ev["rows"],
            "retrained": True,
        })
    results["compact_reason"] = reason
    cache_path.write_text(json.dumps(results), encoding="utf-8")

    # Full-model combined profile is the Part A result. Rerun it only if that file is absent.
    combined = next(c for c in conditions() if c["name"] == "combined_moderate")
    part_a = ROOT / "results" / "diagnostics" / "raw" / "observation_robustness" / "summary.json"
    if "robust_full" not in results and part_a.exists():
        summary = json.loads(part_a.read_text(encoding="utf-8"))
        row = next(r for r in summary if r["name"] == "combined_moderate")
        results["robust_full"] = [
            {"seed": int(s), "ok": int(ok), "source": "observation_robustness"}
            for s, ok in zip(row["seeds"], row["per_seed"])
        ]
        cache_path.write_text(json.dumps(results), encoding="utf-8")
        print("full noisy from part A", results["robust_full"], flush=True)
    if "robust_full" not in results:
        full_rows = []
        for seed in combined["seeds"]:
            ev = run_suite(ctx, model0, combined["spec"], seed)
            full_rows.append({"seed": seed, "ok": ev["ok"], "families": family_counts(ev["rows"])})
            print("full noisy", seed, ev["ok"], flush=True)
        results["robust_full"] = full_rows
        cache_path.write_text(json.dumps(results), encoding="utf-8")
    if results["compact"]["dim"] < 45 and "robust_compact" not in results:
        alias = results["compact"].get("alias")
        ckpt_name = f"{alias}.pt" if alias and alias != "A0" else "compact.pt"
        ckpt = torch.load(OUT / ckpt_name, map_location=device, weights_only=False)
        comp_model = build_policy(proto, device, len(cols))
        comp_model.load_state_dict(ckpt["model"])
        comp_model.eval()
        crow = []
        for seed in combined["seeds"]:
            ev = run_suite(ctx, comp_model, combined["spec"], seed, columns=cols)
            crow.append({"seed": seed, "ok": ev["ok"], "families": family_counts(ev["rows"])})
            print("compact noisy", seed, ev["ok"], flush=True)
        results["robust_compact"] = crow
        cache_path.write_text(json.dumps(results), encoding="utf-8")
    write_report(results)
    print("REPORT", REPORT, flush=True)


def write_report(results):
    lines = []
    lines.append("# Observation ablation check")
    lines.append("")
    lines.append(
        "Each reduced observation is the balanced causal Transformer retrained for 120 epochs "
        "from torch seed 1 on the same regenerated demonstrations, the same suffix sampler, "
        "and the same K=32 action prototypes. The input layer is resized to the kept channels. "
        "A0 is the published checkpoint, not a second training run. "
        "Success is retention through absolute t = 12 on the frozen 24-case benchmark."
    )
    lines.append("")
    lines.append("## Channel groups")
    lines.append("")
    for name, idx in GROUPS.items():
        lines.append(f"- {name}: {idx[0]}–{idx[-1]}")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    for tag, label, cols in VARIANTS:
        lines.append(f"- {tag}, {label}: indices {cols[0]}–{cols[-1]}" if cols == list(range(cols[0], cols[-1] + 1)) and len(cols) == cols[-1] - cols[0] + 1 else f"- {tag}, {label}: {cols}")
    comp = results.get("compact")
    if comp:
        lines.append(f"- compact: {comp['columns']}")
        lines.append("")
        lines.append(f"Compact rule: {json.dumps(results.get('compact_reason'))}. A group is kept when removing it drops impact below 3/4 or drops sliding, wrist, or open-finger recatch below 4/4, or drops the case total below 14/24.")
    lines.append("")
    lines.append("## t = 12")
    lines.append("")
    lines.append("| variant | dim | t=12 | impact | sliding | wrist | recatch | airborne | natural | val loss | train loss |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = [t for t, _l, _c in VARIANTS] + (["compact"] if "compact" in results else [])
    for tag in order:
        row = results[tag]
        fam = row["families"]
        lines.append(
            f"| {tag} {row['label']} | {row['dim']} | {row['ok']}/24 | "
            f"{fam['impact']}/4 | {fam['sliding']}/4 | {fam['wrist']}/4 | {fam['recatch']}/4 | "
            f"{fam['airborne']}/4 | {fam['natural']}/4 | {row['val_loss']} | {row['train_loss']} |"
        )
    lines.append("")
    lines.append("## Behavior on regressions")
    lines.append("")
    base_rows = results["A0"]["rows"]
    any_reg = False
    for tag in order:
        if tag == "A0":
            continue
        notes = behavior_lines(base_rows, results[tag]["rows"])
        if not notes:
            continue
        any_reg = True
        lines.append(f"**{tag}.**")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    if not any_reg:
        lines.append("No retained clean case was lost by a listed variant.")
        lines.append("")
    lines.append("## Combined sensor profile")
    lines.append("")
    full = results.get("robust_full")
    if full:
        lines.append(
            "Full 45-D clean "
            f"{results['A0']['ok']}/24. Full 45-D with the moderate combined profile: "
            + ", ".join(f"seed {r['seed']} {r['ok']}/24" for r in full)
            + "."
        )
        lines.append("")
    crow = results.get("robust_compact")
    if crow:
        lines.append(
            "Compact clean "
            f"{results['compact']['ok']}/24. Compact with the same profile: "
            + ", ".join(f"seed {r['seed']} {r['ok']}/24" for r in crow)
            + "."
        )
        lines.append("")
    elif comp and comp["dim"] == 45:
        lines.append("The compact candidate is the full 45-D vector, so it has no separate noisy run.")
        lines.append("")
    lines.append(_verdict(results))
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verdict(results):
    comp = results["compact"]
    a6 = results["A6"]
    a1 = results["A1"]
    skills_lost = not keeps_skills(comp["families"], comp["ok"]) and comp["dim"] < 45
    full_noisy = results.get("robust_full") or []
    comp_noisy = results.get("robust_compact") or []
    full_mean = None if not full_noisy else float(np.mean([r["ok"] for r in full_noisy]))
    comp_mean = None if not comp_noisy else float(np.mean([r["ok"] for r in comp_noisy]))
    robust_ok = comp["dim"] == 45 or comp_mean is None or (full_mean is not None and comp_mean + 1 >= full_mean)
    if comp["dim"] < 45 and keeps_skills(comp["families"], comp["ok"]) and comp["ok"] >= 14 and robust_ok:
        if comp["dim"] == 39 and comp["columns"] == list(range(39)) and keeps_skills(a1["families"], a1["ok"]):
            return "DUPLICATE HAND-TWIST CHANNELS CAN BE REMOVED WITHOUT LOSS"
        return "A COMPACT OBSERVATION PRESERVES BALANCED RECOVERY PERFORMANCE"
    if keeps_skills(a1["families"], a1["ok"]) and a1["ok"] >= 14 and not keeps_skills(results["A2"]["families"], results["A2"]["ok"]):
        return "DUPLICATE HAND-TWIST CHANNELS CAN BE REMOVED WITHOUT LOSS"
    if a6["ok"] <= 10 and comp["dim"] == 45:
        return "AIRBORNE RELATIVE-STATE CHANNELS ARE NECESSARY FOR BALANCED RECOVERY"
    if skills_lost or any(not keeps_skills(results[t]["families"], results[t]["ok"]) for t, _l, _c in VARIANTS if t not in ("A0",)):
        if comp["dim"] == 45 and a6["ok"] < results["A0"]["ok"]:
            return "OBSERVATION REDUCTION DEGRADES IMPORTANT RECOVERY SKILLS"
    if comp["dim"] == 45:
        return "45-D OBSERVATION IS JUSTIFIED"
    return "INCONCLUSIVE"


if __name__ == "__main__":
    main()
