"""Long-context recurrent recovery pilot. No handoff, SAC, classifier, or skill id.

Input at each step is the legal 27-D observation and the previous 7-D action.
The GRU hidden state or the causal Transformer context carries the rest of the episode.
Advantage weights are applied only inside clusters that share that input and disagree
on the action, so the shared impact state is not regressed to the mean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "diagnostics" / "raw" / "long_context_recovery_pilot"
OBS_DIM = 27
ACT_DIM = 7
EPOCHS = 400
EVENT_BOOST = 8.0
WY_SUSTAIN_MIN = 0.75
OPEN_GRIP_MAX = -0.30
RECLOSE_GRIP_MIN = 0.50
RECLOSE_WY_MAX = 0.35


def nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


class GRUPolicy:
    def __init__(self, hidden=64):
        import torch
        self.torch = torch
        self.gru = torch.nn.GRU(OBS_DIM + ACT_DIM, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, ACT_DIM)
        torch.nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        torch.nn.init.zeros_(self.head.bias)
        self.hidden = hidden

    def parameters(self):
        return list(self.gru.parameters()) + list(self.head.parameters())

    def modules_eval(self):
        self.gru.eval()
        self.head.eval()

    def modules_train(self):
        self.gru.train()
        self.head.train()

    def forward_seq(self, obs, prev):
        torch = self.torch
        x = torch.cat([obs, prev], dim=-1).unsqueeze(0)
        out, _ = self.gru(x)
        act = torch.tanh(self.head(out[0]))
        return act, out[0]


class CausalTransformer:
    def __init__(self, d_model=32, nhead=4, layers=2, ff=64, max_len=64):
        import torch
        self.torch = torch
        self.max_len = max_len
        self.in_proj = torch.nn.Linear(OBS_DIM + ACT_DIM, d_model)
        self.pos = torch.nn.Embedding(max_len, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=0.0, batch_first=True, activation="relu",
        )
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.head = torch.nn.Linear(d_model, ACT_DIM)
        torch.nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        torch.nn.init.zeros_(self.head.bias)
        self.d_model = d_model

    def parameters(self):
        return (
            list(self.in_proj.parameters()) + list(self.pos.parameters())
            + list(self.enc.parameters()) + list(self.head.parameters())
        )

    def modules_eval(self):
        self.in_proj.eval()
        self.pos.eval()
        self.enc.eval()
        self.head.eval()

    def modules_train(self):
        self.in_proj.train()
        self.pos.train()
        self.enc.train()
        self.head.train()

    def forward_seq(self, obs, prev):
        torch = self.torch
        x = torch.cat([obs, prev], dim=-1)
        t = x.shape[0]
        h = self.in_proj(x) + self.pos(torch.arange(t))
        mask = torch.triu(torch.ones(t, t) * float("-inf"), diagonal=1)
        ctx = self.enc(h.unsqueeze(0), mask=mask)[0]
        return torch.tanh(self.head(ctx)), ctx


def prev_actions(actions: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(actions)
    if len(actions) > 1:
        prev[1:] = actions[:-1]
    return prev


def build_success(trajs, labels, adv_all):
    from training.action_chunk_recovery_pilot import event_indices
    from training.value_guided_recovery_bootstrap import SUCCESS

    cursor = 0
    chosen = []
    for tr, name in zip(trajs, labels):
        n = len(tr["a"])
        adv = adv_all[cursor:cursor + n]
        cursor += n
        if name in SUCCESS and int(tr["y"][0]) == 1:
            obs = tr["s"][:, :OBS_DIM].astype(np.float32)
            acts = tr["a"].astype(np.float32)
            chosen.append({
                "name": name,
                "family": str(tr["family"][0]),
                "obs": obs,
                "act": acts,
                "prev": prev_actions(acts),
                "adv": adv.astype(np.float32),
                "events": event_indices(acts),
            })
    return chosen


def step_weights(trajs) -> tuple[np.ndarray, list]:
    """Family-balanced weights. Conflicting contexts get a softmax over advantage."""
    from training.value_guided_recovery_bootstrap import BETA

    rows = []
    for ti, tr in enumerate(trajs):
        for t in range(len(tr["act"])):
            rows.append((ti, t))
    n = len(rows)
    fam_count = {}
    for tr in trajs:
        fam_count[tr["family"]] = fam_count.get(tr["family"], 0) + len(tr["act"])
    base = np.zeros(n, np.float64)
    inputs = np.zeros((n, OBS_DIM + ACT_DIM), np.float32)
    actions = np.zeros((n, ACT_DIM), np.float32)
    adv = np.zeros(n, np.float64)
    for i, (ti, t) in enumerate(rows):
        tr = trajs[ti]
        base[i] = 1.0 / fam_count[tr["family"]]
        if t in tr["events"]:
            base[i] *= EVENT_BOOST
        inputs[i, :OBS_DIM] = tr["obs"][t]
        inputs[i, OBS_DIM:] = tr["prev"][t]
        actions[i] = tr["act"][t]
        adv[i] = float(tr["adv"][t])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(inputs[i] - inputs[j])) < 0.08:
                union(i, j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    conflict_info = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        aa = actions[members]
        spread = float(np.max(np.linalg.norm(aa[:, None, :] - aa[None, :, :], axis=-1)))
        families = {trajs[rows[i][0]]["family"] for i in members}
        # Same-skill phase changes can share a context while the command changes.
        # Those steps stay fully supervised. Only a cross-family disagreement,
        # such as the shared impact state, is resolved by the value weights.
        if spread <= 0.35 or len(families) < 2:
            continue
        logits = adv[members] / BETA
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        total = base[members].sum()
        base[members] = total * p
        conflict_info.append({
            "n": len(members),
            "names": [trajs[rows[i][0]]["name"] for i in members],
            "t": [int(rows[i][1]) for i in members],
            "softmax": [round(float(x), 4) for x in p],
            "adv": [round(float(adv[i]), 4) for i in members],
            "actions_wy": [round(float(actions[i, 4]), 3) for i in members],
            "actions_vx": [round(float(actions[i, 0]), 3) for i in members],
        })
    base *= n / max(base.sum(), 1e-8)
    for tr in trajs:
        tr["w"] = np.zeros(len(tr["act"]), np.float64)
    for i, (ti, t) in enumerate(rows):
        trajs[ti]["w"][t] = base[i]
    return base, conflict_info


def train_model(kind: str, trajs, epochs=EPOCHS, seed=0):
    import torch
    torch.manual_seed(seed)
    model = GRUPolicy() if kind == "gru" else CausalTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.modules_train()
    last = None
    for epoch in range(epochs):
        opt.zero_grad()
        total = 0.0
        denom = 0.0
        for tr in trajs:
            obs = torch.tensor(tr["obs"])
            prev = torch.tensor(tr["prev"])
            target = torch.tensor(tr["act"])
            weight = torch.tensor(tr["w"], dtype=torch.float32)
            pred, _ = model.forward_seq(obs, prev)
            err = ((pred - target) ** 2).mean(dim=-1)
            total = total + (weight * err).sum()
            denom += float(weight.sum())
        loss = total / max(denom, 1e-6)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach())
        if epoch % 100 == 99:
            print(kind, "epoch", epoch + 1, "loss", round(last, 5), flush=True)
    model.modules_eval()
    return model, last, nparams_of(model)


def nparams_of(model) -> int:
    return sum(p.numel() for p in model.parameters())


def predict(model, tr):
    import torch
    with torch.no_grad():
        pred, ctx = model.forward_seq(torch.tensor(tr["obs"]), torch.tensor(tr["prev"]))
    return pred.numpy(), ctx.numpy()


def phase_metrics(name: str, pred: np.ndarray, demo: np.ndarray) -> dict:
    def mae(sl, col):
        if sl.stop > len(pred) or sl.start >= len(pred):
            return None
        return round(float(np.abs(pred[sl, col] - demo[sl, col]).mean()), 4)

    def mean(sl, col):
        if sl.stop > len(pred) or sl.start >= len(pred):
            return None
        return round(float(pred[sl, col].mean()), 4)

    out = {
        "wy": [round(float(x), 4) for x in pred[:, 4]],
        "grip": [round(float(x), 4) for x in pred[:, 6]],
        "vx": [round(float(x), 4) for x in pred[:, 0]],
        "wx": [round(float(x), 4) for x in pred[:, 3]],
    }
    if name == "recatch_open20":
        out.update({
            "sustain_wy_mean": mean(slice(0, 19), 4),
            "sustain_wy_mae": mae(slice(0, 19), 4),
            "open_grip": None if len(pred) <= 19 else round(float(pred[19, 6]), 4),
            "open_wy": None if len(pred) <= 19 else round(float(pred[19, 4]), 4),
            "reclose_grip": None if len(pred) <= 20 else round(float(pred[20, 6]), 4),
            "reclose_wy": None if len(pred) <= 20 else round(float(pred[20, 4]), 4),
        })
        out["pass"] = bool(
            out["sustain_wy_mean"] is not None
            and out["sustain_wy_mean"] >= WY_SUSTAIN_MIN
            and out["sustain_wy_mae"] <= 0.25
            and out["open_grip"] is not None and out["open_grip"] <= OPEN_GRIP_MAX
            and out["reclose_grip"] is not None and out["reclose_grip"] >= RECLOSE_GRIP_MIN
            and abs(out["reclose_wy"]) <= RECLOSE_WY_MAX
        )
    elif name == "wrist_reseat":
        # Step 0 is the shared branch. The wrist prefix is steps 1-11.
        out.update({
            "step0_vx": round(float(pred[0, 0]), 4),
            "step0_wy": round(float(pred[0, 4]), 4),
            "move_vx_mae": mae(slice(1, 12), 0),
            "move_wy_mae": mae(slice(1, 12), 4),
            "hold_wy_mae": mae(slice(12, 18), 4),
        })
        out["pass"] = bool(
            out["move_vx_mae"] is not None and out["move_vx_mae"] <= 0.25
            and out["move_wy_mae"] is not None and out["move_wy_mae"] <= 0.25
            and out["hold_wy_mae"] is not None and out["hold_wy_mae"] <= 0.25
        )
    elif name == "gravity_inward":
        out.update({
            "step0_wx": round(float(pred[0, 3]), 4),
            "step0_wy": round(float(pred[0, 4]), 4),
            "orient_wx_mae": mae(slice(1, 20), 3),
            "orient_wy_mae": mae(slice(1, 20), 4),
            "relax_grip_mean": mean(slice(20, 44), 6),
            "retighten_grip": None if len(pred) <= 44 else round(float(pred[44, 6]), 4),
        })
        out["pass"] = bool(
            out["orient_wx_mae"] is not None and out["orient_wx_mae"] <= 0.15
            and out["orient_wy_mae"] is not None and out["orient_wy_mae"] <= 0.20
            and out["relax_grip_mean"] is not None and out["relax_grip_mean"] <= -0.20
            and out["retighten_grip"] is not None and out["retighten_grip"] >= 0.50
        )
    return out


def context_phases(ctx: np.ndarray, pred: np.ndarray) -> dict:
    """Distances among recatch phases, plus the action at those positions."""
    def center(a, b):
        return ctx[a:b].mean(axis=0)

    marks = {
        "early_wrist": center(0, 5),
        "late_wrist": center(14, 19),
        "open_step": ctx[19],
        "post_reclose": center(22, 27),
    }
    names = list(marks)
    dist = {}
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            dist[f"{na}_vs_{nb}"] = round(float(np.linalg.norm(marks[na] - marks[nb])), 4)
    within = round(float(np.linalg.norm(ctx[2] - ctx[3])), 4)
    return {
        "context_l2": dist,
        "within_early_l2": within,
        "action_at": {
            "early": [round(float(x), 4) for x in pred[2]],
            "late_wrist": [round(float(x), 4) for x in pred[16]],
            "open": [round(float(x), 4) for x in pred[19]],
            "reclose": [round(float(x), 4) for x in pred[20]],
            "post_reclose": [round(float(x), 4) for x in pred[24]],
        },
    }


def branch_at_shared(model, trajs) -> dict:
    from training.value_guided_recovery_bootstrap import SUCCESS
    import torch
    rec = next(tr for tr in trajs if tr["name"] == "recatch_open20")
    modes = {}
    for name in SUCCESS:
        tr = next(t for t in trajs if t["name"] == name and np.linalg.norm(t["obs"][0] - rec["obs"][0]) < 0.08)
        modes[name] = tr["act"][0]
    with torch.no_grad():
        obs = torch.tensor(rec["obs"][:1])
        prev = torch.tensor(rec["prev"][:1])
        # Single-step context is the same object the sequence model uses at t=0.
        pred, _ = model.forward_seq(obs, prev)
        a = pred[0].numpy()
    dists = {name: round(float(np.linalg.norm(a - modes[name])), 4) for name in SUCCESS}
    mean = np.mean(np.stack(list(modes.values())), axis=0)
    dists["mean_of_modes"] = round(float(np.linalg.norm(a - mean)), 4)
    dists["nearest"] = min(SUCCESS, key=lambda n: dists[n])
    dists["action"] = [round(float(x), 4) for x in a]
    dists["closer_to_nearest_than_mean"] = bool(dists[dists["nearest"]] + 0.05 < dists["mean_of_modes"])
    return dists


def loss_share(trajs) -> dict:
    ev = 0.0
    sustain = 0.0
    total = 0.0
    for tr in trajs:
        for t, w in enumerate(tr["w"]):
            total += w
            if t in tr["events"]:
                ev += w
            elif tr["name"] == "recatch_open20" and t < 19:
                sustain += w
    return {
        "event_fraction": round(float(ev / max(total, 1e-8)), 4),
        "recatch_sustain_fraction": round(float(sustain / max(total, 1e-8)), 4),
        "n_trajectories": len(trajs),
        "n_steps": int(sum(len(tr["act"]) for tr in trajs)),
        "per_family": {
            fam: int(sum(len(tr["act"]) for tr in trajs if tr["family"] == fam))
            for fam in sorted({tr["family"] for tr in trajs})
        },
    }


def gate_passed(reports) -> bool:
    return all(r["pass"] for r in reports)


def closed_loop(models):
    from envs.dynamics import finger_pad_geom_ids
    from envs.observable_obs import ObservableObsState
    from envs.physical_recovery import physical_pack
    from training.demo_ballistic_impact import make_sim
    from training.recovery_policy_pretraining_preparation import legal_obs
    from training.recovery_runtime import (
        _dropped,
        _escaped,
        capture_nominal,
        capture_pads,
        oracle_snap,
        restore_dyn,
        run_impact,
        step_cmd,
        stamp,
    )
    from training.replay_core import park_impact_ball
    from training.value_guided_recovery_bootstrap import DT_POLICY, HELD_DYN
    import torch

    sim, cfg = make_sim()
    from controllers.jacobian_controller import gains_from_cfg
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pad_ids, pad_mu = capture_pads(sim)
    pads_l, pads_r = finger_pad_geom_ids(sim.model, sim.ids)
    snaps = {}
    for name, z in (("impact_zm6", -0.006), ("impact_zp6", 0.006), ("impact_zp14", 0.014)):
        impact = run_impact(sim, gains, 6.5, z, 0.0, pads_l, pads_r, nominal, None, None)
        snaps[name] = impact["snap"]
        print("impact", name, impact["meta"]["early"]["rh_mm"], flush=True)
    dyns = {"nominal": (None, None), HELD_DYN: (0.18, 0.55)}
    jobs = [
        ("impact_zm6", "nominal"),
        ("impact_zp6", "nominal"),
        ("impact_zp14", "nominal"),
        ("impact_zm6", HELD_DYN),
    ]

    def rollout(kind, model, snap, mass, fr, n_steps=52):
        info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, mass, fr)
        park_impact_ball(sim)
        hist = ObservableObsState(DT_POLICY)
        if not info["pose_ok"]:
            return {"pose_ok": False}
        obs_seq = []
        prev_seq = []
        prev = np.zeros(ACT_DIM, np.float32)
        acts = []
        query = {}
        for k in range(n_steps):
            obs = np.asarray(legal_obs(sim, hist), np.float32)
            obs_seq.append(obs)
            prev_seq.append(prev.copy())
            with torch.no_grad():
                pred, _ = model.forward_seq(torch.tensor(np.stack(obs_seq)), torch.tensor(np.stack(prev_seq)))
                act = np.clip(pred[-1].numpy(), -1.0, 1.0)
            acts.append(act)
            step_cmd(sim, "7", act)
            prev = act.astype(np.float32)
            if (k + 1) in (16, 32, n_steps) or _dropped(physical_pack(sim)) or _escaped(sim):
                query[k + 1] = stamp(sim)
            if _dropped(physical_pack(sim)) or _escaped(sim):
                break
        A = np.stack(acts)
        labels = []
        for kq in sorted(query):
            lab = oracle_snap(sim, gains, query[kq], nominal, pad_ids, pad_mu, mass, fr)
            labels.append({
                "k": int(kq),
                "where": "endpoint" if kq == max(query) else "during",
                "label": "HANDOFF_SUCCESS" if lab.get("y") == 1 else "HANDOFF_FAILURE",
                "outcome": lab.get("outcome"),
            })
        n19 = min(19, len(A))
        return {
            "pose_ok": True,
            "n": int(len(A)),
            "sustain_wy_mean": round(float(A[:n19, 4].mean()), 4),
            "wy": [round(float(x), 4) for x in A[:min(32, len(A)), 4]],
            "grip": [round(float(x), 4) for x in A[:min(32, len(A)), 6]],
            "grip_at_19": None if len(A) <= 19 else round(float(A[19, 6]), 4),
            "wy_at_19": None if len(A) <= 19 else round(float(A[19, 4]), 4),
            "grip_at_20": None if len(A) <= 20 else round(float(A[20, 6]), 4),
            "labels": labels,
            "any_success": any(x["label"] == "HANDOFF_SUCCESS" for x in labels),
            "endpoint": next((x["label"] for x in labels if x["where"] == "endpoint"), None),
        }

    evals = []
    for kind, model in models:
        for ic, dyn_name in jobs:
            mass, fr = dyns[dyn_name]
            row = rollout(kind, model, snaps[ic], mass, fr)
            row.update({"policy": kind, "ic": ic, "dyn": dyn_name})
            evals.append(row)
            print("EVAL", kind, ic, dyn_name, row.get("sustain_wy_mean"), row.get("grip_at_19"), row.get("endpoint"), flush=True)
    return evals


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    from training.action_chunk_recovery_pilot import advantages
    from training.recovery_skill_temporal_persistence_audit import load_buffer, match_name, split_trajs
    from training.value_guided_recovery_bootstrap import BETA, all_schedules, fit_iql

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_buffer()
    print("fit critic for branch weights", flush=True)
    _actor, q1, q2, vnet, _ = fit_iql(data, steps=2000, seed=0, actor_start=500)
    trajs_all = split_trajs(data)
    scheds = all_schedules()
    labels = [match_name(tr["a"], scheds) for tr in trajs_all]
    adv = advantages(q1, q2, vnet, data["s"], data["a"])
    success = build_success(trajs_all, labels, adv)
    copy_of = {"wrist_reseat": ["impact_zm6", "mass_0.16"],
               "recatch_open20": ["impact_zm6", "impact_zp6", "mass_0.16"],
               "gravity_inward": ["impact_zm6", "impact_zp6", "mass_0.16"]}
    seen = {}
    for tr in success:
        k = seen.get(tr["name"], 0)
        tr["copy"] = copy_of[tr["name"]][k]
        seen[tr["name"]] = k + 1
    _base, conflicts = step_weights(success)
    share = loss_share(success)
    print("success", [(tr["name"], len(tr["act"])) for tr in success], flush=True)
    print("conflicts", conflicts, flush=True)
    print("loss share", share, flush=True)

    results = {}
    trained = []
    passing = []
    for kind in ("gru", "transformer"):
        model, loss, npar = train_model(kind, success)
        reports = []
        phase = None
        for tr in success:
            pred, ctx = predict(model, tr)
            rep = phase_metrics(tr["name"], pred, tr["act"])
            rep["name"] = tr["name"]
            rep["copy"] = tr["copy"]
            rep["n"] = int(len(tr["act"]))
            reports.append(rep)
            if tr["name"] == "recatch_open20" and phase is None:
                phase = context_phases(ctx, pred)
        branch = branch_at_shared(model, success)
        ok = all(r["pass"] for r in reports)
        if ok:
            passing.append(kind)
        results[kind] = {
            "params": npar,
            "loss": loss,
            "teacher_forced_pass": ok,
            "reports": reports,
            "recatch_context": phase,
            "shared_step0": branch,
        }
        rec = [r for r in reports if r["name"] == "recatch_open20"]
        print(
            kind, "params", npar, "pass", ok,
            "wy", [r.get("sustain_wy_mean") for r in rec],
            "open", [r.get("open_grip") for r in rec],
            "reclose", [r.get("reclose_grip") for r in rec],
            "branch", branch["nearest"], branch["action"], flush=True,
        )
        trained.append((kind, model))

    summary = {
        "input": "current legal 27-D observation and previous 7-D action",
        "event_boost": EVENT_BOOST,
        "beta": BETA,
        "epochs": EPOCHS,
        "loss_share": share,
        "conflict_clusters": conflicts,
        "models": results,
        "passing": passing,
        "closed_loop": None,
        "learned_handoff_trained": False,
        "no_classifier": True,
        "no_skill_id": True,
        "online_sac": False,
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    evals = None
    if passing:
        print("closed loop", passing, flush=True)
        evals = closed_loop([(k, m) for k, m in trained if k in passing])
    else:
        print("skip closed loop", flush=True)

    summary = {
        "input": "current legal 27-D observation and previous 7-D action",
        "event_boost": EVENT_BOOST,
        "beta": BETA,
        "epochs": EPOCHS,
        "loss_share": share,
        "conflict_clusters": conflicts,
        "models": results,
        "passing": passing,
        "closed_loop": evals,
        "learned_handoff_trained": False,
        "no_classifier": True,
        "no_skill_id": True,
        "online_sac": False,
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
