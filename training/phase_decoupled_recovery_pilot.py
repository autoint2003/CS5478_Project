"""Phase-decoupled recovery pilot. Same causal Transformer, continuous 7-D actions.

Every successful trajectory is trained as a suffix from every intermediate step, with
the position index reset to zero. The same physical state therefore sits at many
token positions. Absolute recovery-step index is not an input. No handoff, skill
id, or classifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.long_context_recovery_pilot import CausalTransformer

OUT = ROOT / "results" / "diagnostics" / "raw" / "phase_decoupled_recovery_pilot"
EPOCHS = 200
EVENT_BOOST = 4.0
GH = slice(15, 18)


def train_suffixes(trajs, epochs=EPOCHS, seed=0):
    import torch
    torch.manual_seed(seed)
    model = CausalTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pieces = []
    for tr in trajs:
        w = np.asarray(tr["w"], np.float64)
        obs = torch.tensor(tr["obs"])
        prev = torch.tensor(tr["prev"])
        target = torch.tensor(tr["act"])
        for s in range(len(w)):
            idx = np.arange(s, len(w))
            pieces.append((obs[s:], prev[s:], target[s:], w[idx] / (idx + 1.0)))
    denom = float(sum(p[3].sum() for p in pieces))
    print("suffixes", len(pieces), "weight", round(denom, 2), flush=True)
    model.modules_train()
    last = None
    for epoch in range(epochs):
        opt.zero_grad()
        running = 0.0
        for obs, prev, target, scales in pieces:
            pred, _ = model.forward_seq(obs, prev)
            err = ((pred - target) ** 2).mean(dim=-1)
            term = (torch.tensor(scales, dtype=torch.float32) * err).sum() / denom
            term.backward()
            running += float(term.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = running
        if epoch % 40 == 39 or epoch == 0:
            print("epoch", epoch + 1, "loss", round(last, 5), flush=True)
    model.modules_eval()
    npar = sum(p.numel() for p in model.parameters())
    return model, last, npar


def predict(model, obs, prev):
    import torch
    with torch.no_grad():
        pred, _ = model.forward_seq(torch.tensor(obs), torch.tensor(prev))
    return pred.numpy()


def recatch_open_index(wy, grip):
    for i, (w, g) in enumerate(zip(wy, grip)):
        if g < -0.30 and w > 0.40:
            return i
    return None


def reclose_after(grip, open_i):
    if open_i is None:
        return None
    for i in range(open_i + 1, len(grip)):
        if grip[i] > 0.50:
            return i
    return None


def relax_index(grip):
    for i, g in enumerate(grip):
        if g < -0.20:
            return i
    return None


def retighten_after(grip, relax_i):
    if relax_i is None:
        return None
    for i in range(relax_i + 1, len(grip)):
        if grip[i] > 0.50:
            return i
    return None


def pack_pred(pred):
    return {
        "wy": [round(float(x), 4) for x in pred[:, 4]],
        "grip": [round(float(x), 4) for x in pred[:, 6]],
        "vx": [round(float(x), 4) for x in pred[:, 0]],
        "wx": [round(float(x), 4) for x in pred[:, 3]],
        "a0": [round(float(x), 4) for x in pred[0]],
    }


def eval_restart(model, tr, k, kind):
    pred = predict(model, tr["obs"][k:], tr["prev"][k:])
    row = pack_pred(pred)
    row["k"] = int(k)
    row["kind"] = kind
    row["name"] = tr["name"]
    row["tag"] = tr.get("tag", tr["name"])
    row["gh"] = [round(float(x), 4) for x in tr["obs"][k, GH]]
    wy = np.asarray(row["wy"])
    grip = np.asarray(row["grip"])
    if kind == "recatch":
        op = recatch_open_index(wy, grip)
        row["open_step"] = op
        row["reclose_step"] = reclose_after(grip, op)
        row["wy_before_open"] = None if not op else round(float(wy[:op].mean()), 4)
        row["pass"] = bool(op is not None and 2 <= op <= 5 and row["reclose_step"] is not None and row["reclose_step"] <= op + 3)
        # The decisive k=16 case wants the open about three steps later. Callers override `expect`.
    elif kind == "wrist":
        hold = wy[4:10] if len(wy) >= 10 else wy[4:]
        early_open = recatch_open_index(wy[:12], grip[:12])
        row["move_wy"] = round(float(wy[:4].mean()), 4) if len(wy) >= 4 else None
        row["hold_wy"] = round(float(np.mean(np.abs(hold))), 4) if len(hold) else None
        row["early_open"] = early_open
        row["pass"] = bool(
            row["move_wy"] is not None and row["move_wy"] >= 0.50
            and row["hold_wy"] is not None and row["hold_wy"] <= 0.40
            and early_open is None
            and float(grip[:12].min()) > -0.30
        )
    else:
        rel = relax_index(grip)
        ret = retighten_after(grip, rel)
        row["relax_step"] = rel
        row["retighten_step"] = ret
        row["early_grip"] = round(float(grip[:5].mean()), 4) if len(grip) >= 5 else None
        row["pass"] = False
    return row


def phase_gap(tr):
    obs = tr["obs"]
    marks = [k for k in (0, 8, 16, 18, 19, 20, 40, 44) if k < len(obs)]
    gaps = {}
    for i, a in enumerate(marks):
        for b in marks[i + 1:]:
            gaps[f"{a}_vs_{b}"] = round(float(np.linalg.norm(obs[a] - obs[b])), 4)
    gh = {str(k): [round(float(x), 4) for x in obs[k, GH]] for k in marks}
    return {"obs_l2": gaps, "g_h": gh}


def world():
    from envs.dynamics import finger_pad_geom_ids
    from training.demo_ballistic_impact import make_sim
    from training.recovery_runtime import capture_nominal, capture_pads

    sim, cfg = make_sim()
    from controllers.jacobian_controller import gains_from_cfg
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pad_ids, pad_mu = capture_pads(sim)
    pads_l, pads_r = finger_pad_geom_ids(sim.model, sim.ids)
    return sim, gains, nominal, pad_ids, pad_mu, pads_l, pads_r


def record_sequence(sim, actions, snap, nominal, pad_ids, pad_mu, mass, friction, dx, dv, prev0):
    from training.value_guided_recovery_bootstrap import DT_POLICY
    from envs.observable_obs import ObservableObsState
    from envs.physical_recovery import physical_pack
    from training.long_context_recovery_generalization_audit import prepare
    from training.recovery_policy_pretraining_preparation import legal_obs
    from training.recovery_runtime import _dropped, _escaped, step_cmd

    info, _o0 = prepare(sim, snap, nominal, pad_ids, pad_mu, mass, friction, dx, dv)
    if _o0 is None:
        return None
    hist = ObservableObsState(DT_POLICY)
    prev = np.asarray(prev0, np.float32).copy()
    obs_l, prev_l = [], []
    acted = []
    for act in np.asarray(actions, np.float32):
        obs_l.append(np.asarray(legal_obs(sim, hist), np.float32))
        prev_l.append(prev.copy())
        step_cmd(sim, "7", act)
        acted.append(act.copy())
        prev = act.copy()
        if _dropped(physical_pack(sim)) or _escaped(sim):
            return None
    return {
        "obs": np.stack(obs_l),
        "act": np.stack(acted),
        "prev": np.stack(prev_l),
    }


def oracle_now(sim, gains, nominal, pad_ids, pad_mu, mass, friction):
    from training.long_context_recovery_generalization_audit import oracle_of
    return oracle_of(sim, gains, nominal, pad_ids, pad_mu, mass, friction)


def closed_loop(sim, gains, model, stamp, nominal, pad_ids, pad_mu, mass, friction, prev0, n_steps=48):
    from training.long_context_recovery_generalization_audit import rollout_policy
    return rollout_policy(
        sim, gains, model, stamp, nominal, pad_ids, pad_mu, mass, friction, prev0, n_steps,
    )


def main() -> int:
    from training.action_chunk_recovery_pilot import advantages, event_indices
    from training.long_context_recovery_pilot import ACT_DIM, build_success, jsonable, step_weights
    from training.recovery_skill_temporal_persistence_audit import load_buffer, match_name, split_trajs
    from training.value_guided_recovery_bootstrap import SUCCESS, all_schedules, fit_iql

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_buffer()
    print("fit critic", flush=True)
    _actor, q1, q2, vnet, _ = fit_iql(data, steps=2000, seed=0, actor_start=500)
    buf = split_trajs(data)
    scheds = all_schedules()
    labels = [match_name(tr["a"], scheds) for tr in buf]
    success = build_success(buf, labels, advantages(q1, q2, vnet, data["s"], data["a"]))
    copy_of = {
        "wrist_reseat": ["impact_zm6", "mass_0.16"],
        "recatch_open20": ["impact_zm6", "impact_zp6", "mass_0.16"],
        "gravity_inward": ["impact_zm6", "impact_zp6", "mass_0.16"],
    }
    seen = {}
    for tr in success:
        k = seen.get(tr["name"], 0)
        tr["copy"] = copy_of[tr["name"]][k]
        seen[tr["name"]] = k + 1
    _base, conflicts = step_weights(success)
    for tr in success:
        for t in event_indices(tr["act"]):
            tr["w"][t] *= EVENT_BOOST
    weight_of = {(tr["name"], tr["copy"]): tr["w"].copy() for tr in success}

    sim, gains, nominal, pad_ids, pad_mu, pads_l, pads_r = world()
    from training.recovery_runtime import run_impact, stamp
    from training.long_context_recovery_generalization_audit import record_frames

    snaps = {}
    for name, z in (("impact_zm6", -0.006), ("impact_zp6", 0.006), ("impact_zp14", 0.014)):
        impact = run_impact(sim, gains, 6.5, z, 0.0, pads_l, pads_r, nominal, None, None)
        snaps[name] = impact["snap"]
        print("impact", name, impact["meta"]["early"]["rh_mm"], flush=True)
    sched = {k: np.asarray(v, np.float32) for k, v in scheds.items() if k in SUCCESS}

    recorded = []
    frames = {}
    jobs = [
        ("wrist_reseat", "impact_zm6", None, None, "impact_zm6"),
        ("wrist_reseat", "impact_zm6", 0.16, None, "mass_0.16"),
        ("recatch_open20", "impact_zm6", None, None, "impact_zm6"),
        ("recatch_open20", "impact_zp6", None, None, "impact_zp6"),
        ("recatch_open20", "impact_zm6", 0.16, None, "mass_0.16"),
        ("gravity_inward", "impact_zm6", None, None, "impact_zm6"),
        ("gravity_inward", "impact_zp6", None, None, "impact_zp6"),
        ("gravity_inward", "impact_zm6", 0.16, None, "mass_0.16"),
    ]
    for name, ic, mass, fr, copy in jobs:
        seq = record_sequence(
            sim, sched[name], snaps[ic], nominal, pad_ids, pad_mu, mass, fr, 0.0, 0.0, np.zeros(ACT_DIM),
        )
        if seq is None:
            print("CANONICAL FAIL", name, copy, flush=True)
            continue
        seq.update({
            "name": name,
            "family": success[0]["family"] if False else None,
            "copy": copy,
            "tag": f"{name}|{copy}",
            "train": True,
            "w": np.asarray(weight_of[(name, copy)], np.float64),
        })
        fam = {"wrist_reseat": "wrist_micro_reseat", "recatch_open20": "wrist_airborne_recatch", "gravity_inward": "gravity_slide"}
        seq["family"] = fam[name]
        recorded.append(seq)
        if copy == "impact_zm6":
            frames[name] = record_frames(sim, sched[name], snaps[ic], nominal, pad_ids, pad_mu, None, None)
        print("recorded", name, copy, len(seq["act"]), flush=True)

    # Nearby physical perturbations. Targets are the remaining demonstration.
    # A continuation is kept only when the t=12 oracle retains.
    perturb_jobs = [
        ("recatch_open20", 16, -0.002, 0.0, "train"),
        ("recatch_open20", 16, 0.0, 0.02, "train"),
        ("recatch_open20", 8, -0.002, 0.0, "train"),
        ("gravity_inward", 20, -0.002, 0.0, "train"),
        ("gravity_inward", 20, 0.002, 0.0, "train"),
        ("gravity_inward", 20, 0.0, -0.02, "train"),
        ("gravity_inward", 36, -0.002, 0.0, "train"),
        ("wrist_reseat", 6, -0.002, 0.0, "train"),
        ("recatch_open20", 16, 0.003, 0.0, "test"),
        ("gravity_inward", 20, -0.004, 0.0, "test"),
    ]
    mean_w = float(np.mean([tr["w"].mean() for tr in recorded]))
    perturb_report = []
    for name, k, dx, dv, split in perturb_jobs:
        frs = frames[name]
        if k >= len(frs):
            continue
        remain = sched[name][k:]
        seq = record_sequence(
            sim, remain, frs[k]["stamp"], nominal, pad_ids, pad_mu, None, None, dx, dv, frs[k]["prev"],
        )
        lab = None
        kept = False
        if seq is not None:
            lab = oracle_now(sim, gains, nominal, pad_ids, pad_mu, None, None)
            kept = lab.get("outcome") == "RETAINED_TO_12" and split == "train"
            seq.update({
                "name": name,
                "family": recorded[0]["family"],
                "copy": f"perturb_k{k}",
                "tag": f"{name}|k={k}|dx={dx}|dv={dv}|{split}",
                "train": kept,
                "w": np.full(len(seq["act"]), mean_w, np.float64),
            })
            fam = {"wrist_reseat": "wrist_micro_reseat", "recatch_open20": "wrist_airborne_recatch", "gravity_inward": "gravity_slide"}
            seq["family"] = fam[name]
            if kept:
                recorded.append(seq)
            elif split == "test":
                seq["train"] = False
                recorded.append(seq)
        perturb_report.append({
            "name": name, "k": k, "dx": dx, "dv": dv, "split": split,
            "recorded": seq is not None, "kept_for_train": kept, "oracle": None if lab is None else lab.get("outcome"),
        })
        print("perturb", name, k, dx, dv, split, None if lab is None else lab.get("outcome"), "train" if kept else "skip", flush=True)

    # Held-out impact: scripted recatch on zp14, not used for training.
    zp14 = record_sequence(
        sim, sched["recatch_open20"], snaps["impact_zp14"], nominal, pad_ids, pad_mu,
        None, None, 0.0, 0.0, np.zeros(ACT_DIM),
    )
    zp14_oracle = oracle_now(sim, gains, nominal, pad_ids, pad_mu, None, None) if zp14 is not None else None
    if zp14 is not None:
        zp14.update({
            "name": "recatch_open20", "family": "wrist_airborne_recatch", "copy": "impact_zp14",
            "tag": "recatch_open20|impact_zp14|heldout", "train": False,
            "w": np.full(len(zp14["act"]), mean_w),
        })
    print("zp14 script", None if zp14_oracle is None else zp14_oracle.get("outcome"), flush=True)

    train_set = [tr for tr in recorded if tr.get("train")]
    gaps = {}
    for tr in train_set:
        if tr["copy"] == "impact_zm6":
            gaps[tr["name"]] = phase_gap(tr)
            print("gap", tr["name"], gaps[tr["name"]], flush=True)

    model, loss, npar = train_suffixes(train_set)
    print("trained", npar, "loss", loss, flush=True)

    by_tag = {tr["tag"]: tr for tr in recorded}
    zm6 = {tr["name"]: tr for tr in train_set if tr["copy"] == "impact_zm6"}

    restarts = []
    # Expected relative event: recatch open is 19-k, slide retighten is 44-k, wrist hold starts at 12-k.
    specs = [
        ("recatch_open20", 16, "recatch", 3),
        ("recatch_open20", 12, "recatch", 7),
        ("recatch_open20", 18, "recatch", 1),
        ("wrist_reseat", 8, "wrist", 4),
        ("wrist_reseat", 10, "wrist", 2),
        ("gravity_inward", 20, "slide", 24),
        ("gravity_inward", 40, "slide", 4),
        ("gravity_inward", 32, "slide", 12),
    ]
    for name, k, kind, expect in specs:
        row = eval_restart(model, zm6[name], k, kind)
        row["expect_offset"] = expect
        if kind == "recatch":
            row["pass"] = bool(row["open_step"] is not None and abs(row["open_step"] - expect) <= 2 and row["reclose_step"] is not None and row["reclose_step"] <= row["open_step"] + 3)
        elif kind == "wrist":
            wy = np.asarray(row["wy"])
            grip = np.asarray(row["grip"])
            move = wy[:expect]
            hold = wy[expect:expect + 6]
            early = recatch_open_index(wy[:12], grip[:12])
            row["move_wy"] = round(float(move.mean()), 4) if len(move) else None
            row["hold_wy"] = round(float(np.mean(np.abs(hold))), 4) if len(hold) else None
            row["early_open"] = early
            row["pass"] = bool(
                row["move_wy"] is not None and row["move_wy"] >= 0.50
                and row["hold_wy"] is not None and row["hold_wy"] <= 0.40
                and early is None
            )
        elif kind == "slide" and k == 40:
            grip_k = np.asarray(row["grip"])
            hit = next((i for i, g in enumerate(grip_k) if g > 0.50), None)
            row["retighten_step"] = hit
            row["pass"] = bool(hit is not None and hit <= 6 and abs(row["a0"][4]) < 0.55)
        elif kind == "slide":
            row["pass"] = bool(
                row["early_grip"] is not None and row["early_grip"] <= -0.15
                and row["retighten_step"] is not None and abs(row["retighten_step"] - expect) <= 6
                and abs(row["a0"][4]) < 0.75
            )
        restarts.append(row)
        print("TF", name, "k", k, "pass", row["pass"], {kk: row[kk] for kk in row if kk not in ("wy", "grip", "vx", "wx")}, flush=True)

    heldout_rows = []
    if zp14 is not None:
        row = eval_restart(model, zp14, 16, "recatch")
        row["expect_offset"] = 3
        row["pass"] = bool(row["open_step"] is not None and abs(row["open_step"] - 3) <= 2)
        row["oracle_script"] = None if zp14_oracle is None else zp14_oracle.get("outcome")
        heldout_rows.append(row)
        print("TF heldout zp14", row["pass"], row.get("open_step"), row.get("reclose_step"), flush=True)
    for tr in recorded:
        if tr.get("train"):
            continue
        if "dx=0.003" in tr["tag"]:
            row = eval_restart(model, tr, 0, "recatch")
            row["expect_offset"] = 3
            row["pass"] = bool(row["open_step"] is not None and abs(row["open_step"] - 3) <= 2)
            heldout_rows.append(row)
            print("TF heldout perturb", tr["tag"], row["pass"], row.get("open_step"), flush=True)
        if "dx=-0.004" in tr["tag"]:
            row = eval_restart(model, tr, 0, "slide")
            row["expect_offset"] = 24
            row["pass"] = bool(row["early_grip"] is not None and row["early_grip"] <= -0.15 and row["retighten_step"] is not None and abs(row["retighten_step"] - 24) <= 6)
            heldout_rows.append(row)
            print("TF heldout slide", tr["tag"], row["pass"], row.get("relax_step"), row.get("retighten_step"), flush=True)

    gate_keys = {("recatch_open20", 16), ("wrist_reseat", 8), ("gravity_inward", 20), ("gravity_inward", 40)}
    gate = [r for r in restarts if (r["name"], r["k"]) in gate_keys]
    gate_pass = all(r["pass"] for r in gate)
    print("GATE", gate_pass, [(r["name"], r["k"], r["pass"]) for r in gate], flush=True)

    closed = None
    if gate_pass:
        closed = []
        tests = [
            ("recatch_open20", 16, 3),
            ("recatch_open20", 12, 7),
            ("recatch_open20", 18, 1),
            ("recatch_open20", 22, None),
            ("gravity_inward", 20, 24),
            ("gravity_inward", 32, 12),
            ("gravity_inward", 40, 4),
            ("wrist_reseat", 8, 4),
        ]
        for name, k, expect in tests:
            fr = frames[name][k]
            row = closed_loop(
                sim, gains, model, fr["stamp"], nominal, pad_ids, pad_mu, None, None, fr["prev"],
            )
            wy = np.asarray(row.get("wy") or [])
            grip = np.asarray(row.get("grip") or [])
            if name.startswith("recatch"):
                op = recatch_open_index(wy, grip) if len(wy) else None
                row["open_step"] = op
                row["reclose_step"] = reclose_after(grip, op) if len(grip) else None
                if expect is None:
                    row["restarted_recatch"] = bool(op == 19 or (len(wy) > 10 and float(wy[:10].mean()) > 0.85 and (op is None or op >= 15)))
                    row["pass"] = bool(not row["restarted_recatch"] and (op is None or op > 8) and float(grip[:8].mean()) > 0.5 if len(grip) >= 8 else False)
                else:
                    row["pass"] = bool(op is not None and abs(op - expect) <= 2)
            elif name.startswith("gravity"):
                rel = relax_index(grip) if len(grip) else None
                ret = retighten_after(grip, rel) if len(grip) else None
                row["relax_step"] = rel
                row["retighten_step"] = ret
                row["restarted_recatch"] = bool(len(wy) > 10 and float(wy[:10].mean()) > 0.85)
                if k >= 40:
                    hit = next((i for i, g in enumerate(grip) if g > 0.50), None)
                    row["retighten_step"] = hit
                    row["pass"] = bool(hit is not None and hit <= 6 and not row["restarted_recatch"])
                else:
                    row["pass"] = bool(
                        not row["restarted_recatch"]
                        and len(grip) >= 5 and float(grip[:5].mean()) <= -0.15
                        and ret is not None and abs(ret - expect) <= 6
                    )
            else:
                row["pass"] = bool(len(wy) >= 8 and float(wy[:4].mean()) >= 0.5 and float(np.mean(np.abs(wy[4:10]))) <= 0.45)
            row.update({"skill": name, "demo_k": k, "expect_offset": expect})
            # Drop long traces from the printed line; keep them in the row.
            closed.append(row)
            print(
                "CL", name, k, "pass", row["pass"], "oracle", row.get("outcome"),
                "open", row.get("open_step"), "reclose", row.get("reclose_step"),
                "relax", row.get("relax_step"), "retighten", row.get("retighten_step"),
                "a0", row.get("a0"), flush=True,
            )
    else:
        print("skip closed loop", flush=True)

    summary = {
        "architecture": "same causal Transformer, position index reset on every suffix",
        "absolute_step_not_an_input": True,
        "epochs": EPOCHS,
        "event_boost": EVENT_BOOST,
        "params": npar,
        "loss": loss,
        "n_train_trajectories": len(train_set),
        "conflicts": conflicts,
        "phase_gaps": gaps,
        "perturbations": perturb_report,
        "teacher_forced": restarts,
        "heldout_teacher_forced": heldout_rows,
        "gate_pass": gate_pass,
        "closed_loop": closed,
        "learned_handoff_trained": False,
        "no_skill_id": True,
        "no_classifier": True,
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
