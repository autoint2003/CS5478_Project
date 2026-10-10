"""Balanced unified recovery. Airborne recatch is one family, not the objective.

The observation, tokenizer, Transformer, v3 action map, and tracking gains stay
fixed. Family names balance the dataloader and never enter the policy.
The score is physical retention through absolute t = 12.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_data_scaling as scale
import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import observe_airborne, relative_state
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import physical_pack
from training.clean_natural_loss_recatch import distal_geoms, measure
from training.demo_ballistic_impact import assert_noslip
from training.full_3d_airborne_recatch_dataset import (
    TRACK,
    begin,
    carry_then_release,
    drop_supports,
    grasp_settled,
    hold_step,
    limits_of,
    restore_to,
)
from training.long_context_recovery_pilot import jsonable
from training.phase_decoupled_recovery_pilot import world
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import T_TASK, continue_long, physical_of, restore_dyn, run_impact
from training.replay_core import park_impact_ball

RAW = ROOT / "results" / "diagnostics" / "raw" / "balanced_unified_recovery"
REPORT = ROOT / "results" / "diagnostics" / "BALANCED_UNIFIED_RECOVERY_EVALUATION.md"
CKPT = ROOT / "results" / "diagnostics" / "raw" / "natural_loss_controlled_dagger" / "ckpt_round0.pt"
EPOCHS = 120
STEPS_PER_EPOCH = 80
POOLS = ("wrist", "sliding", "recatch", "airborne", "natural")
TRAIN_NATURAL = (
    {"tag": "dz13.2", "dz": -13.2},
    {"tag": "dz13.5", "dz": -13.5},
    {"tag": "dz14.2", "dz": -14.2},
    {"tag": "dz14.6", "dz": -14.6},
    {"tag": "dvz_m", "dz": -13.5, "dvz": -0.01},
    {"tag": "dvz_p", "dz": -13.5, "dvz": 0.005},
)
EVAL_NATURAL = (
    {"tag": "h_dz14.00", "dz": -14.00},
    {"tag": "h_dz14.25", "dz": -14.25},
    {"tag": "h_dy", "dz": -14.10, "dy": 0.12},
    {"tag": "h_dx", "dz": -14.70, "dx": 0.10},
)


def train_equal(trajs, proto, device):
    """Same loss and batch as the current Transformer. Families share equal weight."""
    import torch

    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)
    model = v3.build_model(proto, device)
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
    drift = n_suffix - sum(counts.values())
    counts[max(counts, key=counts.get)] += drift
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
    total = max(1, sum(counts.values()))
    return model, last, {
        "epochs": EPOCHS,
        "steps": STEPS_PER_EPOCH * EPOCHS,
        "seconds": round(time.perf_counter() - t0, 1),
        "seed": 1,
        "n_traj": {k: len(groups[k]) for k in POOLS},
        "fractions": {k: round(counts.get(k, 0) / total, 3) for k in POOLS},
    }


def mechanism_of(rows, t12):
    if t12 != "RETAINED_TO_12" or not rows:
        return "none"
    lost = any(r["nL"] == 0 or r["nR"] == 0 for r in rows)
    end_both = rows[-1]["nL"] > 0 and rows[-1]["nR"] > 0
    max_a2 = max(float(r["act"][2]) for r in rows)
    max_wy = max(abs(float(r["act"][4])) for r in rows)
    if lost and end_both and max_a2 > 0.4:
        return "airborne recatch"
    if (not lost) and max_wy > 0.4:
        return "wrist reseat"
    if not lost:
        return "contact-preserving"
    if lost and end_both:
        return "recontact"
    return "contact lost during recovery"


def eval_case(sim, model, case, nominal, pads, pad_mu):
    import torch

    kind = case["scene"]
    if kind == "impact":
        info = restore_dyn(sim, case["snap"], nominal, pads, pad_mu, None, None)
        park_impact_ball(sim)
    else:
        info = restore_to(sim, case["snap"], nominal, pads, pad_mu)
        drop_supports(sim)
    if not info.get("pose_ok", False):
        return {"t12": "RESTORE_FAIL", "mechanism": "none", "t_start": None}
    t_start = float(sim.data.time)
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    rows = []
    model.eval()
    dropped = False
    with torch.no_grad():
        for _k in range(int(case["n_steps"])):
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(obs)
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=next(model.parameters()).device),
                torch.tensor(np.stack(prev_seq), device=next(model.parameters()).device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            v3.step_v3(sim, act)
            o = physical_pack(sim)
            rows.append({"act": act, "nL": int(o["nL"]), "nR": int(o["nR"]), "z": float(o["obj_z"])})
            prev = act
            if float(o["obj_z"]) < 0.44:
                dropped = True
                break
    if dropped:
        t12, kind_y = "DROP", "DROP"
        end = physical_pack(sim)
    else:
        rec = continue_long(sim, TRACK, T_TASK, "policy")
        y, kind_y = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        t12 = "RETAINED_TO_12" if y == 1 else kind_y
        end = rec["end"]
    return {
        "t12": t12,
        "kind": kind_y,
        "mechanism": mechanism_of(rows, t12),
        "end_n": [int(end["nL"]), int(end["nR"])],
        "t_start": round(t_start, 3),
        "policy_steps": len(rows),
        "lost_contact": any(r["nL"] == 0 or r["nR"] == 0 for r in rows),
    }


def longest(windows):
    return max(windows, key=lambda w: len(w["obs"]))


def mode(items):
    items = [x for x in items if x and x != "none"]
    if not items:
        return "none"
    return max(set(items), key=items.count)


def family_table(rows, key):
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        ok = [r for r in sub if r[key]["t12"] == "RETAINED_TO_12"]
        fails = [r[key]["t12"] for r in sub if r[key]["t12"] != "RETAINED_TO_12"]
        out[fam] = {
            "n": len(sub),
            "ok": len(ok),
            "mechanism": mode([r[key]["mechanism"] for r in ok]),
            "failure": mode(fails) if fails else "none",
        }
    return out


def macro(table):
    if not table:
        return 0.0
    return float(np.mean([v["ok"] / v["n"] for v in table.values()]))


def write_report(blob):
    base, bal = blob["baseline_families"], blob["balanced_families"]
    lines = []
    lines.append("# Balanced unified recovery")
    lines.append("")
    lines.append(
        "The policy interface is unchanged: the 45-D observation, the frozen K=32 tokenizer, "
        "the causal Transformer, recovery7d_airborne_v3, and the tracking gains. "
        "Family labels are used only to give each verified recovery family an equal share of the "
        f"training batches. The baseline is the seed-1 120-epoch checkpoint. "
        f"The balanced model trains for {blob['train']['epochs']} epochs, {blob['train']['steps']} optimizer steps, "
        f"in {blob['train']['seconds']} s, from torch seed {blob['train']['seed']}."
    )
    lines.append("")
    lines.append(
        "A case succeeds only when the recovery policy acts, the nominal controller then continues unchanged, "
        "and the object is still in a bilateral grasp above the lift height at absolute t = 12, "
        "with no drop and no escape. A 1 s recatch is not the score."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        f"Training trajectories by family: {blob['train']['n_traj']}. "
        f"Batch fractions: {blob['train']['fractions']}. "
        f"Final training loss {None if blob['train_loss'] is None else round(float(blob['train_loss']), 4)}. "
        f"Family-balanced token accuracy on the training trajectories {blob['train_acc']}. "
        f"Held-out natural-loss token accuracy {blob['val_natural']}."
    )
    lines.append("")
    lines.append(
        "Natural-loss demonstrations are the six core offsets whose first distal-tip loss the v3 intercept regrasps. "
        "They receive the same batch share as wrist reseating, inward sliding, the open-finger recatch, "
        "and the airborne carry-and-release demos. Held-out natural-loss offsets are not in this training set."
    )
    lines.append("")
    lines.append("## t = 12 by disturbance family")
    lines.append("")
    lines.append("| disturbance family | cases | baseline | balanced policy | dominant successful mechanism | failure mode |")
    lines.append("|---|---:|---:|---:|---|---|")
    fams = list(base)
    for fam in fams:
        b, n = base[fam], bal[fam]
        fail = n["failure"] if n["ok"] < n["n"] else "none"
        lines.append(
            f"| {fam} | {n['n']} | {b['ok']}/{b['n']} | {n['ok']}/{n['n']} | {n['mechanism']} | {fail} |"
        )
    lines.append("")
    lines.append(
        f"Macro-average family success is {blob['baseline_macro']:.3f} for the baseline and "
        f"{blob['balanced_macro']:.3f} for the balanced policy. "
        f"Case totals are {blob['baseline_cases']}/{blob['n_cases']} and {blob['balanced_cases']}/{blob['n_cases']}."
    )
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| case | family | baseline t=12 | balanced t=12 | balanced mechanism |")
    lines.append("|---|---|---|---|---|")
    for row in blob["rows"]:
        lines.append(
            f"| {row['id']} | {row['family']} | {row['baseline']['t12']} | {row['balanced']['t12']} | {row['balanced']['mechanism']} |"
        )
    lines.append("")
    lines.append(
        "Wrist, sliding, and open-finger rows are states along those verified demonstrations, "
        "including the centered −6 mm impact and the +6 mm and lateral impacts. "
        "Airborne rows start after a sideways carry and an opening, with both pads already clear. "
        "Natural-loss rows are held-out distal-tip slips. The same snaps are used for both policies."
    )
    lines.append("")
    skill = ("sliding", "wrist", "recatch")
    regressed = [f for f in skill if f in bal and bal[f]["ok"] < base[f]["ok"]]
    natural_up = "natural" in bal and bal["natural"]["ok"] > base["natural"]["ok"]
    overall_up = blob["balanced_macro"] > blob["baseline_macro"] + 1e-9
    overall_down = blob["balanced_macro"] + 1e-9 < blob["baseline_macro"]
    if natural_up and overall_down:
        verdict = "RECATCh SPECIALIZATION HURTS OVERALL RECOVERY"
    elif overall_up and not regressed:
        verdict = "BALANCED UNIFIED RECOVERY IMPROVES OVERALL PHYSICAL SUCCESS"
    elif regressed:
        verdict = "UNIFIED POLICY STILL LACKS SUFFICIENT CLOSED-LOOP ROBUSTNESS"
    elif not overall_up:
        verdict = "BALANCED TRAINING PRESERVES SKILLS BUT DOES NOT IMPROVE OVERALL SUCCESS"
    else:
        verdict = "INCONCLUSIVE"
    lines.append(verdict)
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return verdict


def main():
    import torch

    RAW.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CKPT, map_location=device, weights_only=False)
    if int(saved.get("seed", -1)) != 1:
        raise RuntimeError("baseline checkpoint is not seed 1")
    proto = np.asarray(saved["proto"], np.float32)
    baseline = v3.build_model(proto, device)
    baseline.load_state_dict(saved["model"])
    baseline.eval()
    print("BASELINE seed 1 proto", tuple(proto.shape), "device", device, flush=True)

    sim, _g, nominal, pads, pad_mu, _pl, _pr = world()
    assert_noslip(sim)
    seat = grasp_settled(sim, TRACK)
    if seat is None or not begin(sim, seat, nominal, pads, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    distal = distal_geoms(sim)
    q_ref, _margin = limits_of(sim)
    drop_supports(sim)
    base = v3.stamp(sim)
    print("SEAT t", round(float(sim.data.time), 3), flush=True)

    natural = []
    for spec in TRAIN_NATURAL:
        rec, meta = scale.collect(sim, base, spec, pinch, distal, nominal, pads, pad_mu, "train")
        print("TRAIN_NL", spec["tag"], "KEEP" if rec else meta.get("cls"), flush=True)
        if rec is None:
            continue
        for w in rec["windows"]:
            w["pool"] = "natural"
            natural.append(w)
    if not natural:
        raise RuntimeError("no verified natural-loss trajectories")

    eval_natural = []
    for spec in EVAL_NATURAL:
        rec, meta = scale.collect(sim, base, spec, pinch, distal, nominal, pads, pad_mu, "hold")
        print("EVAL_NL", spec["tag"], "KEEP" if rec else meta.get("cls"), flush=True)
        if rec is None:
            continue
        window = longest(rec["windows"])
        window["pool"] = "natural"
        eval_natural.append(window)

    air_train = []
    vertical = v3.record_air(
        sim, {"tag": "vertical", "mode": "above", "lead": 0.0, "k": 4.0, "rz": -0.012, "vrel": 0.45},
        base, nominal, pads, pad_mu, pinch, "train",
    )
    if vertical is not None:
        vertical["pool"] = "airborne"
        air_train.append(vertical)
        print("AIR vertical", len(vertical["act"]), flush=True)
    else:
        print("AIR vertical fail", flush=True)

    sim_i, gains_i, nominal_i, pads_i, pad_mu_i, pads_l, pads_r = world()
    assert_noslip(sim_i)
    sched = schedules()
    print("IMPACT zm6", flush=True)
    zm6 = run_impact(sim_i, gains_i, 6.5, -0.006, 0.0, pads_l, pads_r, nominal_i, None, None)["snap"]
    print("IMPACT zp6", flush=True)
    zp6 = run_impact(sim_i, gains_i, 6.5, 0.006, 0.0, pads_l, pads_r, nominal_i, None, None)["snap"]
    print("IMPACT lateral", flush=True)
    lat = run_impact(sim_i, gains_i, 6.5, 0.0, 90.0, pads_l, pads_r, nominal_i, None, None)["snap"]
    print("IMPACT z3", flush=True)
    z3 = run_impact(sim_i, gains_i, 6.5, -0.003, 0.0, pads_l, pads_r, nominal_i, None, None)["snap"]

    local = []
    for skill, pool in (
        ("wrist_reseat", "wrist"),
        ("gravity_inward", "sliding"),
        ("recatch_open20", "recatch"),
    ):
        tr = v3.record_local(sim_i, skill, sched[skill], zm6, nominal_i, pads_i, pad_mu_i, None)
        if tr is None:
            raise RuntimeError(f"verified local skill failed: {skill}")
        tr["pool"] = pool
        local.append(tr)
        print("LOCAL", skill, len(tr["act"]), flush=True)

    def suffix(actions, index):
        info = restore_dyn(sim_i, zm6, nominal_i, pads_i, pad_mu_i, None, None)
        park_impact_ball(sim_i)
        if not info["pose_ok"]:
            return None
        for i, act in enumerate(actions):
            if i == index:
                return v3.stamp(sim_i)
            v3.step_v3(sim_i, v3.v1_normalized_to_v3(np.asarray(act, np.float32)))
            if float(physical_pack(sim_i)["obj_z"]) < 0.44:
                return None
        return None

    def shift(delta):
        import copy
        j = int(sim_i.ids.object_jnt)
        adr = int(sim_i.model.jnt_qposadr[j])
        out = copy.deepcopy(zm6)
        q = np.array(out["qpos"], float)
        q[adr:adr + 3] = q[adr:adr + 3] + np.asarray(delta, float)
        out["qpos"] = q
        return out

    cases = [
        {"id": "impact_zm6", "family": "impact", "scene": "impact", "snap": zm6, "n_steps": 56},
        {"id": "impact_zp6", "family": "impact", "scene": "impact", "snap": zp6, "n_steps": 56},
        {"id": "impact_phi90", "family": "impact", "scene": "impact", "snap": lat, "n_steps": 56},
        {"id": "impact_z3", "family": "impact", "scene": "impact", "snap": z3, "n_steps": 56},
    ]
    for index in (20, 36):
        snap_i = suffix(sched["gravity_inward"], index)
        if snap_i is not None:
            cases.append({
                "id": f"slide_suffix_{index}", "family": "sliding", "scene": "impact",
                "snap": snap_i, "n_steps": max(12, 56 - index),
            })
    cases.append({"id": "slide_dy1", "family": "sliding", "scene": "impact", "snap": shift((0, 0.001, 0)), "n_steps": 56})
    cases.append({"id": "slide_dz1", "family": "sliding", "scene": "impact", "snap": shift((0, 0, 0.001)), "n_steps": 56})
    for index in (3, 6, 9, 12):
        snap_i = suffix(sched["wrist_reseat"], index)
        if snap_i is not None:
            cases.append({
                "id": f"wrist_suffix_{index}", "family": "wrist", "scene": "impact",
                "snap": snap_i, "n_steps": max(12, 56 - index),
            })
    for index in (8, 16, 24, 28):
        snap_i = suffix(sched["recatch_open20"], index)
        if snap_i is not None:
            cases.append({
                "id": f"recatch_suffix_{index}", "family": "recatch", "scene": "impact",
                "snap": snap_i, "n_steps": max(12, 56 - index),
            })

    for tag, speed, sign in (
        ("air_lat_pos", 0.30, 1.0),
        ("air_lat_neg", 0.30, -1.0),
        ("air_fast_pos", 0.45, 1.0),
        ("air_slow_neg", 0.20, -1.0),
    ):
        made = carry_then_release(sim, base, nominal, pads, pad_mu, q_ref, speed, sign, 0.08)
        print("CARRY", tag, None if made is None else made["n"], flush=True)
        if made is None:
            continue
        if tag == "air_lat_pos":
            tr = v3.record_air(
                sim,
                {"tag": "lat_pos", "mode": "below", "lead": 0.03, "k": 5.0, "rz": 0.0, "vrel": 0.55},
                made["snap"], nominal, pads, pad_mu,
                np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float),
                "train",
            )
            if tr is not None:
                tr["pool"] = "airborne"
                air_train.append(tr)
                print("AIR lat_pos", len(tr["act"]), flush=True)
        cases.append({
            "id": tag, "family": "airborne", "scene": "natural",
            "snap": made["snap"], "n_steps": 120,
        })
    for w in eval_natural:
        cases.append({
            "id": "nl_" + w["tag"], "family": "natural", "scene": "natural",
            "snap": w["snap"], "n_steps": 120,
        })
    print("CASES", len(cases), [c["family"] for c in cases], flush=True)

    trajs = natural + local + air_train
    model, train_loss, train_info = train_equal(trajs, proto, device)
    torch.save({"seed": 1, "proto": proto, "model": model.state_dict(), "train": train_info}, RAW / "ckpt.pt")

    acc = {}
    for pool in POOLS:
        group = [tr for tr in trajs if tr["pool"] == pool]
        acc[pool] = None if not group else scale.predict_scores(model, group, device)["token_acc"]
    present_acc = [v for v in acc.values() if v is not None]
    val = scale.predict_scores(model, eval_natural, device) if eval_natural else {"loss": None, "token_acc": None}
    print("ACC", acc, "val", val, flush=True)

    rows = []
    for case in cases:
        sim_use = sim_i if case["scene"] == "impact" else sim
        nom = nominal_i if case["scene"] == "impact" else nominal
        pad = pads_i if case["scene"] == "impact" else pads
        mu = pad_mu_i if case["scene"] == "impact" else pad_mu
        b = eval_case(sim_use, baseline, case, nom, pad, mu)
        n = eval_case(sim_use, model, case, nom, pad, mu)
        rows.append({"id": case["id"], "family": case["family"], "baseline": b, "balanced": n})
        print("CASE", case["id"], b["t12"], n["t12"], n["mechanism"], flush=True)

    base_fam = family_table(rows, "baseline")
    bal_fam = family_table(rows, "balanced")
    blob = {
        "train": train_info,
        "train_loss": None if train_loss is None else float(train_loss),
        "train_acc": None if not present_acc else round(float(np.mean(present_acc)), 4),
        "train_acc_families": acc,
        "val_natural": val,
        "baseline_families": base_fam,
        "balanced_families": bal_fam,
        "baseline_macro": macro(base_fam),
        "balanced_macro": macro(bal_fam),
        "baseline_cases": sum(1 for r in rows if r["baseline"]["t12"] == "RETAINED_TO_12"),
        "balanced_cases": sum(1 for r in rows if r["balanced"]["t12"] == "RETAINED_TO_12"),
        "n_cases": len(rows),
        "rows": rows,
    }
    (RAW / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")
    verdict = write_report(blob)
    print("VERDICT", verdict, flush=True)
    print("REPORT", REPORT, flush=True)


if __name__ == "__main__":
    main()
