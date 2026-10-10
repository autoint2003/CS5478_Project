"""Single-step k-means action tokens. No chunks, no skill labels, no RL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "diagnostics" / "raw" / "single_step_action_token_policy"
DIMS = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]
EPOCHS = 80
STEPS = 40


def collect(sim, gains, nominal, pad_ids, pad_mu, snaps):
    from training.long_context_recovery_pilot import ACT_DIM
    from training.phase_decoupled_recovery_pilot import oracle_now, record_sequence
    from training.recovery_policy_pretraining_preparation import schedules
    from training.value_guided_recovery_bootstrap import SUCCESS

    sched = {k: np.asarray(v, np.float32) for k, v in schedules().items() if k in SUCCESS}
    trajs = []
    jobs = [
        ("wrist_reseat", "zm6", None),
        ("wrist_reseat", "zm6", 0.16),
        ("recatch_open20", "zm6", None),
        ("recatch_open20", "zp6", None),
        ("recatch_open20", "zm6", 0.16),
        ("recatch_open20", "z10", None),
        ("gravity_inward", "zm6", None),
        ("gravity_inward", "zp6", None),
        ("gravity_inward", "zm6", 0.16),
        ("gravity_inward", "z10", None),
    ]
    for name, ic, mass in jobs:
        seq = record_sequence(
            sim, sched[name], snaps[ic], nominal, pad_ids, pad_mu, mass, None, 0.0, 0.0, np.zeros(ACT_DIM),
        )
        if seq is None or (
            ic == "z10" and oracle_now(sim, gains, nominal, pad_ids, pad_mu, None, None).get("outcome") != "RETAINED_TO_12"
        ):
            print("skip", name, ic, flush=True)
            continue
        seq["name"] = name
        seq["ic"] = ic
        trajs.append(seq)
    rng = np.random.default_rng(1)
    for name, ic in (
        ("recatch_open20", "zp6"),
        ("recatch_open20", "z10"),
        ("gravity_inward", "zp6"),
        ("gravity_inward", "z10"),
    ):
        noisy = np.clip(sched[name] + rng.normal(0.0, 0.08, size=sched[name].shape).astype(np.float32), -1.0, 1.0)
        seq = record_sequence(
            sim, noisy, snaps[ic], nominal, pad_ids, pad_mu, None, None, 0.0, 0.0, np.zeros(ACT_DIM),
        )
        if seq is None or oracle_now(sim, gains, nominal, pad_ids, pad_mu, None, None).get("outcome") != "RETAINED_TO_12":
            continue
        seq["name"] = name
        seq["ic"] = ic + "_noise"
        trajs.append(seq)
    return trajs, sched


def action_stats(actions):
    a = np.asarray(actions, np.float32)
    rounded = np.round(a, 3)
    keys, counts = np.unique(rounded, axis=0, return_counts=True)
    order = np.argsort(-counts)
    top = []
    for i in order[:12]:
        top.append({"a": [round(float(x), 3) for x in keys[i]], "n": int(counts[i])})
    return {
        "n": int(len(a)),
        "min": [round(float(x), 4) for x in a.min(0)],
        "max": [round(float(x), 4) for x in a.max(0)],
        "mean": [round(float(x), 4) for x in a.mean(0)],
        "unique_round3": int(len(keys)),
        "grip_open": int((a[:, 6] < -0.30).sum()),
        "grip_relax": int(((a[:, 6] < -0.20) & (a[:, 6] >= -0.70)).sum()),
        "wy_high": int((a[:, 4] > 0.85).sum()),
        "wx_twist": int((a[:, 3] > 0.15).sum()),
        "top": top,
    }


def kmeans(x, k, seed, restarts=8, iters=40):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = np.empty((k, x.shape[1]), np.float32)
        choice = rng.integers(0, len(x))
        centers[0] = x[choice]
        for j in range(1, k):
            d = ((x[:, None, :] - centers[None, :j, :]) ** 2).sum(-1).min(1)
            probs = d / max(float(d.sum()), 1e-8)
            centers[j] = x[rng.choice(len(x), p=probs)]
        labels = np.zeros(len(x), np.int64)
        for _it in range(iters):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = dist.argmin(1)
            nxt = centers.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    nxt[j] = x[mask].mean(0)
            if np.allclose(nxt, centers, atol=1e-6):
                centers = nxt
                break
            centers = nxt
        inertia = float(((x - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers.astype(np.float32), labels.copy())
    return best


def assign(centers, actions):
    a = np.asarray(actions, np.float32)
    dist = ((a[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return dist.argmin(1), np.sqrt(dist.min(1))


def nearest_proto(centers, action):
    a = np.asarray(action, np.float32)
    dist = np.linalg.norm(centers - a, axis=1)
    i = int(dist.argmin())
    return i, round(float(dist[i]), 4), [round(float(x), 3) for x in centers[i]]


def chunk_collapse(trajs, rec, slide):
    from training.action_tokenization_recovery_feasibility import gather_chunks, train_tokenizer

    import torch
    model = train_tokenizer(gather_chunks(trajs, 8), 8, 16, 3)
    cr = torch.tensor(np.asarray(rec[:8], np.float32)[None])
    cs = torch.tensor(np.asarray(slide[:8], np.float32)[None])
    with torch.no_grad():
        ir, zer = model.ids_of(cr)
        iz, zes = model.ids_of(cs)
        dr = model.decode_ids(ir)[0].numpy()
        ds = model.decode_ids(iz)[0].numpy()
        gap = float(torch.norm(zer - zes))
        dcode = torch.cdist(zer, model.code.weight)[0].numpy()
    return {
        "id_recatch": int(ir[0]),
        "id_slide": int(iz[0]),
        "encoder_l2": round(gap, 4),
        "decode_l2": round(float(np.linalg.norm(dr - ds)), 4),
        "decode_wy8": [round(float(dr[:, 4].mean()), 3), round(float(ds[:, 4].mean()), 3)],
        "decode_wx8": [round(float(dr[:, 3].mean()), 3), round(float(ds[:, 3].mean()), 3)],
        "dist_to_codes_recatch": [round(float(x), 3) for x in np.sort(dcode)[:4]],
    }


class TokenHead:
    def __init__(self, n_codes):
        import torch
        self.logits = torch.nn.Linear(32, n_codes)
        self.res = torch.nn.Sequential(
            torch.nn.Linear(32 + ACT_DIM, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, ACT_DIM),
        )

    def parameters(self):
        return list(self.logits.parameters()) + list(self.res.parameters())


def train_policy(encoder, head, trajs, centers, labels_of):
    from training.unsupervised_trajectory_latent_recovery import encode_hist, encoder_parameters, pad_batch, window

    import torch
    index = [(ti, s) for ti, seq in enumerate(trajs) for s in range(len(seq["act"]))]
    opt = torch.optim.Adam(
        [
            {"params": encoder_parameters(encoder), "lr": 1e-4},
            {"params": head.parameters(), "lr": 1e-3},
        ]
    )
    rng = np.random.default_rng(0)
    encoder.modules_train()
    centers_t = torch.tensor(centers)
    last = None
    for epoch in range(EPOCHS):
        pick = [index[int(i)] for i in rng.integers(0, len(index), size=48)]
        rows, ys, acts = [], [], []
        for ti, s in pick:
            rows.append(window(trajs[ti], s + 1))
            ys.append(int(labels_of[(ti, s)]))
            acts.append(trajs[ti]["act"][s])
        obs, prev, lengths = pad_batch(rows)
        h = encode_hist(encoder, obs, prev, lengths)
        y = torch.tensor(ys)
        proto = centers_t[y]
        delta = head.res(torch.cat([h, proto], dim=-1))
        pred = (proto + delta).clamp(-1.0, 1.0)
        ce = torch.nn.functional.cross_entropy(head.logits(h), y)
        rec = (pred - torch.tensor(np.stack(acts))).pow(2).mean()
        loss = ce + rec
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_parameters(encoder) + head.parameters(), 1.0)
        opt.step()
        last = (float(ce.detach()), float(rec.detach()), float(delta.detach().pow(2).mean().sqrt()))
        if epoch % 20 == 19 or epoch == 0:
            print("policy", epoch + 1, "ce", round(last[0], 4), "rec", round(last[1], 4), "drms", round(last[2], 4), flush=True)
    encoder.modules_eval()
    return last


def probs_at(encoder, head, obs, prev):
    from training.unsupervised_trajectory_latent_recovery import encode_hist

    import torch
    obs_t = torch.tensor(np.asarray(obs, np.float32)[None])
    prev_t = torch.tensor(np.asarray(prev, np.float32)[None])
    with torch.no_grad():
        h = encode_hist(encoder, obs_t, prev_t, torch.tensor([obs_t.shape[1]]))
        prob = torch.softmax(head.logits(h), dim=-1)[0].numpy()
    return prob


def rollout(sim, gains, snap, encoder, head, centers, nominal, pad_ids, pad_mu, steps=STEPS, mass=None, friction=None, prev0=None):
    from training.long_context_recovery_pilot import ACT_DIM
    from training.long_context_recovery_generalization_audit import event_steps
    from training.phase_decoupled_recovery_pilot import oracle_now
    from training.unsupervised_trajectory_latent_recovery import encode_hist

    import torch
    from envs.observable_obs import ObservableObsState
    from envs.physical_recovery import physical_pack
    from training.long_context_recovery_generalization_audit import prepare
    from training.recovery_policy_pretraining_preparation import legal_obs
    from training.recovery_runtime import _dropped, _escaped, step_cmd
    from training.value_guided_recovery_bootstrap import DT_POLICY

    info, _o = prepare(sim, snap, nominal, pad_ids, pad_mu, mass, friction, 0.0, 0.0)
    if _o is None:
        return {"pose_ok": False}
    state = ObservableObsState(DT_POLICY)
    prev = np.zeros(ACT_DIM, np.float32) if prev0 is None else np.asarray(prev0, np.float32).copy()
    obs_l, prev_l, acts, tokens, deltas, deltas_v = [], [], [], [], [], []
    stopped = None
    top2 = []
    centers_t = torch.tensor(centers)
    for _k in range(steps):
        obs_l.append(np.asarray(legal_obs(sim, state), np.float32))
        prev_l.append(prev.copy())
        sl = slice(max(0, len(obs_l) - 48), None)
        obs_t = torch.tensor(np.stack(obs_l)[sl][None])
        prev_t = torch.tensor(np.stack(prev_l)[sl][None])
        with torch.no_grad():
            h = encode_hist(encoder, obs_t, prev_t, torch.tensor([obs_t.shape[1]]))
            prob = torch.softmax(head.logits(h), dim=-1)[0].numpy()
            z = int(prob.argmax())
            proto = centers_t[z][None]
            delta = head.res(torch.cat([h, proto], dim=-1))[0].numpy()
        if not tokens:
            order = np.argsort(-prob)
            top2 = [(int(order[0]), round(float(prob[order[0]]), 4)), (int(order[1]), round(float(prob[order[1]]), 4))]
        act = np.clip(centers[z] + delta, -1.0, 1.0).astype(np.float32)
        step_cmd(sim, "7", act)
        acts.append(act)
        tokens.append(z)
        deltas.append(float(np.linalg.norm(delta)))
        deltas_v.append(np.asarray(delta, np.float32))
        prev = act.copy()
        if _dropped(physical_pack(sim)) or _escaped(sim):
            stopped = "DROP" if _dropped(physical_pack(sim)) else "ESCAPE"
            break
    lab = oracle_now(sim, gains, nominal, pad_ids, pad_mu, mass, friction)
    A = np.stack(acts)
    events = event_steps(A[:, 6])
    early = min(8, len(A))
    return {
        "outcome": lab.get("outcome"),
        "y": int(lab.get("y", 0)),
        "stopped": stopped,
        "tokens": tokens,
        "n_unique_tokens": int(len(set(tokens))),
        "delta_mean": round(float(np.mean(deltas)), 4),
        "wy8": round(float(A[:early, 4].mean()), 3),
        "wx8": round(float(A[:early, 3].mean()), 3),
        "vx8": round(float(A[:early, 0].mean()), 3),
        "grip_min": round(float(A[:, 6].min()), 3),
        "open_step": events["open_step"],
        "reclose_step": events["reclose_step"],
        "relax_step": events["relax_step"],
        "a0": [round(float(x), 3) for x in A[0]],
        "delta0": [round(float(x), 3) for x in deltas_v[0]],
        "top2": top2,
        "tokens_head": tokens[:12],
    }


def main() -> int:
    from training.long_context_recovery_generalization_audit import record_frames
    from training.long_context_recovery_pilot import jsonable
    from training.recovery_policy_pretraining_preparation import schedules
    from training.sac_skill_forgetting_attribution import CKPT
    from training.unsupervised_trajectory_latent_recovery import load_encoder
    from training.value_guided_recovery_bootstrap import SUCCESS

    OUT.mkdir(parents=True, exist_ok=True)
    from training.phase_decoupled_recovery_pilot import world
    from training.recovery_runtime import run_impact
    from training.unsupervised_trajectory_latent_recovery import initial_history

    sim, gains, nominal, pad_ids, pad_mu, pads_l, pads_r = world()
    snaps = {}
    for name, mm in (("zm6", -6), ("zp6", 6), ("z10", 10), ("z3", 3), ("z4", 4), ("z8", 8), ("z12", 12)):
        impact = run_impact(sim, gains, 6.5, mm / 1000.0, 0.0, pads_l, pads_r, nominal, None, None)
        if impact["snap"] is None:
            raise RuntimeError("impact snapshot missing " + name)
        snaps[name] = impact["snap"]
        print("impact", name, flush=True)
    sched = {k: np.asarray(v, np.float32) for k, v in schedules().items() if k in SUCCESS}
    frames = {}
    for name in ("wrist_reseat", "recatch_open20", "gravity_inward"):
        frames[name] = record_frames(sim, sched[name], snaps["zm6"], nominal, pad_ids, pad_mu, None, None)
        print("frames", name, len(frames[name]), flush=True)
    trajs, sched = collect(sim, gains, nominal, pad_ids, pad_mu, snaps)
    actions = np.concatenate([seq["act"] for seq in trajs], 0).astype(np.float32)
    stats = action_stats(actions)
    print("DATA", len(trajs), stats["n"], "unique", stats["unique_round3"], "range", stats["min"], stats["max"], flush=True)
    by = {(seq["name"], seq["ic"]): seq for seq in trajs}
    rec = by[("recatch_open20", "zp6")]["act"]
    slide = by[("gravity_inward", "zp6")]["act"]
    print("A0 recatch", [round(float(x), 3) for x in rec[0]], "slide", [round(float(x), 3) for x in slide[0]], flush=True)
    collapse = chunk_collapse(trajs, rec, slide)
    print("CHUNK", collapse, flush=True)

    probes = {
        "recatch_hold": rec[0],
        "slide_hold": slide[0],
        "recatch_open": rec[19],
        "recatch_reclose": rec[20],
        "slide_relax": slide[20],
        "wrist_move": by[("wrist_reseat", "zm6")]["act"][0],
    }
    fits = {}
    separable = {}
    for k in (8, 16, 32):
        inertia, centers, labels = kmeans(actions, k, seed=0)
        occ = np.bincount(labels, minlength=k).tolist()
        assigned = {}
        for name, act in probes.items():
            i, dist, proto = nearest_proto(centers, act)
            assigned[name] = {"id": i, "dist": dist, "proto": proto, "n": int(occ[i])}
        ok = len({assigned[n]["id"] for n in ("recatch_hold", "slide_hold", "recatch_open", "slide_relax", "wrist_move")}) == 5
        fits[str(k)] = {
            "inertia": round(inertia, 3),
            "occupied": int(sum(n > 0 for n in occ)),
            "empty": int(sum(n == 0 for n in occ)),
            "probes": assigned,
            "separates_five": ok,
            "proto_l2_recatch_slide": round(float(np.linalg.norm(
                centers[assigned["recatch_hold"]["id"]] - centers[assigned["slide_hold"]["id"]]
            )), 4),
        }
        separable[k] = (ok, centers, labels)
        print("KM", k, "sep", ok, {n: assigned[n]["id"] for n in assigned}, "empty", fits[str(k)]["empty"], flush=True)

    chosen_k = next((k for k in (8, 16, 32) if separable[k][0]), None)
    summary = {
        "n_traj": len(trajs),
        "action_stats": stats,
        "chunk_collapse": collapse,
        "kmeans": fits,
        "chosen_k": chosen_k,
        "policy_trained": False,
        "rl_trained": False,
        "selector_trained": False,
        "skill_labels_in_loss": False,
    }
    if chosen_k is None:
        (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
        print("STOP merged", flush=True)
        print("DONE", flush=True)
        return 0

    _ok, centers, labels = separable[chosen_k]
    label_of = {}
    cursor = 0
    for ti, seq in enumerate(trajs):
        for s in range(len(seq["act"])):
            label_of[(ti, s)] = int(labels[cursor])
            cursor += 1
    proto_err = float(np.linalg.norm(actions - centers[labels], axis=1).mean())
    summary["prototype_mean_l2"] = round(proto_err, 4)
    summary["prototype_l2_modes"] = fits[str(chosen_k)]["proto_l2_recatch_slide"]

    import torch
    torch.manual_seed(0)
    encoder = load_encoder(CKPT / "init.pt")
    head = TokenHead(chosen_k)
    last = train_policy(encoder, head, trajs, centers, label_of)
    summary["policy_trained"] = True
    summary["policy_ce"] = last[0]
    summary["policy_rec"] = last[1]
    summary["policy_delta_rms"] = last[2]

    ids = {name: fits[str(chosen_k)]["probes"][name]["id"] for name in fits[str(chosen_k)]["probes"]}
    dist_rows = []
    for cell in ("zp6", "z10"):
        hist = initial_history(sim, snaps[cell], nominal, pad_ids, pad_mu)
        prob = probs_at(encoder, head, hist[0], hist[1])
        order = np.argsort(-prob)
        top = [{"id": int(i), "p": round(float(prob[i]), 4), "proto": [round(float(x), 3) for x in centers[i]]} for i in order[:6]]
        row = {
            "cell": cell,
            "top": top,
            "p_recatch": round(float(prob[ids["recatch_hold"]]), 4),
            "p_slide": round(float(prob[ids["slide_hold"]]), 4),
            "p_open": round(float(prob[ids["recatch_open"]]), 4),
            "p_relax": round(float(prob[ids["slide_relax"]]), 4),
            "p_wrist": round(float(prob[ids["wrist_move"]]), 4),
        }
        dist_rows.append(row)
        print("DIST", cell, row["p_recatch"], row["p_slide"], row["p_open"], row["p_relax"], "top", top[:3], flush=True)
    summary["distributions"] = dist_rows

    jobs = [
        ("z3", snaps["z3"], {}),
        ("z4", snaps["z4"], {}),
        ("zp6", snaps["zp6"], {}),
        ("z8", snaps["z8"], {}),
        ("z10", snaps["z10"], {}),
        ("z12", snaps["z12"], {}),
        ("wrist", frames["wrist_reseat"][8]["stamp"], {"prev0": frames["wrist_reseat"][8]["prev"]}),
        ("recatch", frames["recatch_open20"][16]["stamp"], {"prev0": frames["recatch_open20"][16]["prev"]}),
        ("slide", frames["gravity_inward"][20]["stamp"], {"prev0": frames["gravity_inward"][20]["prev"]}),
        ("held", snaps["zm6"], {"mass": 0.18, "friction": 0.55}),
    ]
    rolls = []
    for tag, snap, kw in jobs:
        row = rollout(
            sim, gains, snap, encoder, head, centers, nominal, pad_ids, pad_mu,
            mass=kw.get("mass"), friction=kw.get("friction"), prev0=kw.get("prev0"),
        )
        row["tag"] = tag
        rolls.append(row)
        print(
            "CASE", tag, row.get("outcome"), "wy", row.get("wy8"), "wx", row.get("wx8"),
            "g", row.get("grip_min"), "open", row.get("open_step"), "tok", row.get("n_unique_tokens"),
            "d", row.get("delta_mean"), flush=True,
        )
    summary["rollouts"] = rolls
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
