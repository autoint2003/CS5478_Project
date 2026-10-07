"""Stage A preparation: 7D recovery action, legal history, balanced demos, BC pilot.

No viability classifier. No learned handoff. No joint SAC.
Physical labels are nominal continuation to t=12.
Mechanism names are analysis columns, never policy inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamics import finger_pad_geom_ids
from envs.observable_obs import OBS_NAMES, ObservableObsState, observe_observable
from envs.observable_reward import read_tactile
from envs.physical_recovery import physical_pack
from training.demo_ballistic_impact import make_sim
from training.recovery_runtime import (
    capture_nominal,
    capture_pads,
    map7,
    oracle_snap,
    restore_dyn,
    run_impact,
    step_cmd,
)
from controllers.residual import map_recovery4d

OUT = ROOT / "results" / "diagnostics" / "raw" / "recovery_policy_pretraining_preparation"
HIST = 4
DT_POLICY = 0.02
HELD_IC = "impact_zp14"
HELD_DYN = "held_0.18_0.55"
SUCCESS_SCHEDULES = ("wrist_reseat", "recatch_open20", "gravity_inward")
FAMILIES = ("wrist_micro_reseat", "wrist_airborne_recatch", "gravity_slide")


def embed4(a4) -> list:
    """4D [wy, vhx, vz_world, grip] -> 7D. Refuses a nonzero world-z channel."""
    a = np.asarray(a4, float).reshape(4)
    if abs(float(a[2])) > 1e-8:
        raise ValueError("4D world-z is not a hand-frame vz; this skill does not use it")
    return [float(a[1]), 0.0, 0.0, 0.0, float(a[0]), 0.0, float(a[3])]


def rep(a, n):
    return [list(a) for _ in range(int(n))]


def schedules() -> dict:
    return {
        "wrist_reseat": rep(embed4([0.8, 0.6, 0.0, 1.0]), 12) + rep(embed4([0, 0, 0, 1]), 6),
        "wrist_opposite": rep(embed4([-0.8, -0.6, 0.0, 1.0]), 12) + rep(embed4([0, 0, 0, 1]), 6),
        "recatch_open20": rep(embed4([1, 0, 0, 1]), 19) + rep(embed4([1, 0, 0, -1]), 1) + rep(embed4([0, 0, 0, 1]), 12),
        "recatch_open60": rep(embed4([1, 0, 0, 1]), 19) + rep(embed4([1, 0, 0, -1]), 3) + rep(embed4([0, 0, 0, 1]), 12),
        "gravity_inward": rep([0, 0, 0, 0.28, 0.50, 0, 1.0], 20) + rep([0, 0, 0, 0, 0, 0, -0.48], 24) + rep([0, 0, 0, 0, 0, 0, 1.0], 8),
        "gravity_outward": rep([0, 0, 0, 0.28, -0.50, 0, 1.0], 20) + rep([0, 0, 0, 0, 0, 0, -0.48], 24) + rep([0, 0, 0, 0, 0, 0, 1.0], 8),
        "gravity_secure": rep([0, 0, 0, 0.28, 0.50, 0, 1.0], 20) + rep([0, 0, 0, 0, 0, 0, 1.0], 24) + rep([0, 0, 0, 0, 0, 0, 1.0], 8),
    }


def family_of(name: str) -> str:
    if name.startswith("wrist"):
        return "wrist_micro_reseat"
    if name.startswith("recatch"):
        return "wrist_airborne_recatch"
    return "gravity_slide"


def action_audit() -> dict:
    eye = np.eye(3)
    a4 = np.array([0.8, 0.6, 0.0, 1.0])
    m4 = map_recovery4d(a4, eye, eye)
    v7, w7, tau7, vh, wb = map7(embed4(a4), eye, eye)
    grav = np.array([0, 0, 0, 0.28, 0.50, 0, 1.0])
    _, _, _, _, wb_g = map7(grav, eye, eye)
    return {
        "name": "recovery7",
        "components": ["vx_hand", "vy_hand", "vz_hand", "wx_rdes", "wy_rdes", "wz_rdes", "grip"],
        "v_max_m_s": 0.08,
        "w_max_rad_s": 4.0,
        "tau": "a=-1 -> +2 N*m, a=+1 -> -18 N*m",
        "frame": "v = R_hand @ a[0:3]*0.08; w = R_des @ a[3:6]*4",
        "embed4_matches_wrist": bool(
            np.allclose(m4["v_world"], v7) and np.allclose(m4["w_world"], w7) and abs(m4["tau"] - tau7) < 1e-6
        ),
        "recovery4d_wx": 0.0,
        "gravity_wx_rad_s": float(wb_g[0]),
        "gravity_wy_rad_s": float(wb_g[1]),
        "note": "recovery4d cannot command wx. The gravity slide uses wx and wy together. Stage A actions are 7D.",
    }


def stamp(sim) -> dict:
    snap = make_snap(sim)
    snap["obj_z_ref"] = float(sim.data.xpos[sim.ids.object_body][2])
    snap["hand_ref"] = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    return snap


def legal_obs(sim, hist) -> np.ndarray:
    _, tac = read_tactile(sim.model, sim.data, sim.ids)
    return np.asarray(observe_observable(sim.model, sim.data, sim.ids, sim.fsm, tac, hist), np.float32)


class HandoffHead:
    """Same 167-D history as the actor. Logit means nominal handoff is safe now.

    Stage B fits this with the actor frozen. Stage A does not call fit.
    """

    def __init__(self, dim: int):
        import torch
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
        self.dim = dim

    def fit(self, *_args, **_kwargs):
        raise RuntimeError("Stage A does not train the handoff head")


def blank_hist():
    return [np.zeros(27, np.float32) for _ in range(HIST)], [np.zeros(7, np.float32) for _ in range(HIST)], [0.0] * HIST


def push(past_obs, past_act, valid, obs, act):
    past_obs = [obs] + past_obs[:-1]
    past_act = [np.asarray(act, np.float32)] + past_act[:-1]
    valid = [1.0] + valid[:-1]
    return past_obs, past_act, valid


def feat_from(obs, past_obs, past_act, valid) -> np.ndarray:
    parts = [np.asarray(obs, np.float32)]
    for p in past_obs:
        parts.append(np.asarray(p, np.float32))
    for a in past_act:
        parts.append(np.asarray(a, np.float32))
    parts.append(np.asarray(valid, np.float32))
    return np.concatenate(parts)


def rollout_actions(sim, snap, actions, nominal, pad_ids, pad_mu, mass, friction):
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, mass, friction)
    park_impact_ball(sim)
    hist = ObservableObsState(DT_POLICY)
    past_obs, past_act, valid = blank_hist()
    rows = []
    if not info["pose_ok"]:
        return rows, None, info
    for k, act in enumerate(actions):
        obs = legal_obs(sim, hist)
        o = physical_pack(sim)
        rows.append({
            "k": k,
            "obs": obs.copy(),
            "feat": feat_from(obs, past_obs, past_act, valid),
            "act": np.asarray(act, np.float32),
            "rh": np.asarray(o["rh"], float).copy(),
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "g_hx": float(np.asarray(o["g_h"], float)[0]),
        })
        step_cmd(sim, "7", act)
        past_obs, past_act, valid = push(past_obs, past_act, valid, obs, act)
    end = stamp(sim)
    return rows, end, info


def mechanism_flags(name: str, rows: list) -> dict:
    if not rows:
        return {"executed": False}
    if name == "gravity_inward" and len(rows) > 43:
        dx = 1e3 * float(rows[43]["rh"][0] - rows[19]["rh"][0])
        held = rows[43]["nL"] > 0 and rows[43]["nR"] > 0
        return {"executed": bool(dx <= -2.0 and held), "relax_dx_mm": round(dx, 2)}
    if name == "wrist_reseat":
        a = np.stack([r["act"] for r in rows[:12]])
        on = all(r["nL"] > 0 and r["nR"] > 0 for r in rows[:12])
        return {"executed": bool(on and float(a[:, 4].mean()) > 0.4 and float(a[:, 0].mean()) > 0.3)}
    if name == "recatch_open20":
        off = [i for i, r in enumerate(rows) if r["nL"] == 0 and r["nR"] == 0]
        back = False
        if off:
            back = any(r["nL"] > 0 and r["nR"] > 0 for r in rows[off[0] + 1:])
        return {"executed": bool(off and back), "air_steps": int(len(off))}
    return {"executed": False}


def pattern_of(actions: np.ndarray) -> str:
    """Coarse closed-loop signature over the first 0.40 s. Analysis only."""
    a = np.asarray(actions[:20], float)
    if len(a) == 0:
        return "none"
    wx, wy, vx, grip = float(np.mean(a[:, 3])), float(np.mean(a[:, 4])), float(np.mean(a[:, 0])), float(np.min(a[:, 6]))
    if abs(wx) > 0.12 and abs(wy) > 0.15:
        return "gravity_orientation"
    if abs(wy) > 0.45 and abs(vx) < 0.15 and grip < 0.0:
        return "recatch_like"
    if abs(wy) > 0.25 and abs(vx) > 0.25:
        return "wrist_reseat_like"
    return "unstructured"


def fit_actor(X, Y, weights, seed=0):
    import torch
    torch.manual_seed(seed)
    d, da = X.shape[1], Y.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, 128), torch.nn.Tanh(),
        torch.nn.Linear(128, 128), torch.nn.Tanh(),
        torch.nn.Linear(128, da), torch.nn.Tanh(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(Y, dtype=torch.float32)
    wt = torch.tensor(weights, dtype=torch.float32)
    net.train()
    for _ in range(120):
        pred = net(xt)
        err = ((pred - yt) ** 2).mean(dim=1)
        loss = (wt * err).sum() / max(float(wt.sum()), 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(xt).numpy()
    mae = float(np.mean(np.abs(pred - Y)))
    return net, mae


def act_fn(net, feat):
    import torch
    with torch.no_grad():
        a = net(torch.tensor(feat, dtype=torch.float32)[None]).numpy()[0]
    return np.clip(a, -1.0, 1.0)


def family_weights(families: np.ndarray) -> np.ndarray:
    w = np.zeros(len(families), float)
    for fam in FAMILIES:
        idx = np.where(families == fam)[0]
        if len(idx):
            w[idx] = 1.0 / len(idx)
    return w


def actor_rollout(sim, net, snap, nominal, pad_ids, pad_mu, mass, friction, n_steps, query_at):
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, mass, friction)
    park_impact_ball(sim)
    hist = ObservableObsState(DT_POLICY)
    past_obs, past_act, valid = blank_hist()
    actions = []
    snaps = {}
    contact_k = None
    seen_both = False
    if not info["pose_ok"]:
        return {"pose_ok": False, "actions": [], "snaps": {}, "pattern": "none"}
    for k in range(n_steps):
        obs = legal_obs(sim, hist)
        o = physical_pack(sim)
        if k in query_at:
            snaps[k] = stamp(sim)
        both = int(o["nL"]) > 0 and int(o["nR"]) > 0
        if both:
            seen_both = True
        if contact_k is None and seen_both and not both:
            contact_k = k
            if k not in snaps:
                snaps[k] = stamp(sim)
        feat = feat_from(obs, past_obs, past_act, valid)
        act = act_fn(net, feat)
        actions.append(act)
        step_cmd(sim, "7", act)
        past_obs, past_act, valid = push(past_obs, past_act, valid, obs, act)
    snaps[n_steps] = stamp(sim)
    early_dead = False
    return {
        "pose_ok": True,
        "actions": np.stack(actions),
        "snaps": snaps,
        "pattern": pattern_of(np.stack(actions)),
        "contact_k": contact_k,
        "early_dead": early_dead,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = action_audit()
    (OUT / "action_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("ACTION", json.dumps(audit), flush=True)
    if not audit["embed4_matches_wrist"] or audit["gravity_wx_rad_s"] == 0.0:
        raise RuntimeError("action audit failed")

    sim, cfg = make_sim()
    from controllers.jacobian_controller import gains_from_cfg
    gains = gains_from_cfg(cfg)
    nominal = capture_nominal(sim)
    pad_ids, pad_mu = capture_pads(sim)
    pads_l, pads_r = finger_pad_geom_ids(sim.model, sim.ids)
    sched = schedules()

    print("capture impacts", flush=True)
    snaps = {}
    for name, z in (("impact_zm6", -0.006), ("impact_zp6", 0.006), ("impact_zp14", 0.014)):
        impact = run_impact(sim, gains, 6.5, z, 0.0, pads_l, pads_r, nominal, None, None)
        snaps[name] = impact["snap"]
        print(name, impact["meta"]["early"]["rh_mm"], "g", impact["meta"]["early"]["g_h"], flush=True)

    dyns = {
        "nominal": (None, None),
        "mass_0.16": (0.16, None),
        HELD_DYN: (0.18, 0.55),
    }
    # Precommitted grid. Held-out IC and held-out dynamics are named before outcomes.
    jobs = []
    for ic in ("impact_zm6", "impact_zp6", "impact_zp14"):
        for sch in SUCCESS_SCHEDULES:
            jobs.append((ic, "nominal", sch))
        if ic == "impact_zm6":
            jobs.append((ic, "nominal", "wrist_opposite"))
            jobs.append((ic, "nominal", "recatch_open60"))
            jobs.append((ic, "nominal", "gravity_outward"))
            jobs.append((ic, "nominal", "gravity_secure"))
            for sch in SUCCESS_SCHEDULES:
                jobs.append((ic, "mass_0.16", sch))
                jobs.append((ic, HELD_DYN, sch))

    records = []
    bank = []  # imitate rows
    for ic, dyn_name, sch in jobs:
        mass, fr = dyns[dyn_name]
        split = "train"
        if ic == HELD_IC:
            split = "heldout_ic"
        elif dyn_name == HELD_DYN:
            split = "heldout_dyn"
        rows, end, info = rollout_actions(sim, snaps[ic], sched[sch], nominal, pad_ids, pad_mu, mass, fr)
        if end is None:
            lab = {"y": 0, "outcome": "INVALID_RESTORE", "pose_ok": False}
        else:
            lab = oracle_snap(sim, gains, end, nominal, pad_ids, pad_mu, mass, fr)
        flags = mechanism_flags(sch, rows)
        imitate = bool(
            split == "train" and sch in SUCCESS_SCHEDULES and lab.get("y") == 1 and flags.get("executed")
        )
        rec = {
            "ic": ic, "dyn": dyn_name, "schedule": sch, "family": family_of(sch),
            "split": split, "n": len(rows), "outcome": lab.get("outcome"),
            "y": int(lab.get("y", 0)), "executed": bool(flags.get("executed")),
            "imitate": imitate, "pose_ok": bool(info.get("pose_ok")),
            "mass": lab.get("mass"), "friction": lab.get("friction"),
            "detail": {k: v for k, v in flags.items() if k != "executed"},
        }
        records.append(rec)
        print("DEMO", rec["split"], ic, dyn_name, sch, rec["outcome"], "exec", rec["executed"], "imitate", imitate, flush=True)
        if imitate:
            for r in rows:
                bank.append((r["feat"], r["act"], family_of(sch)))

    if len(bank) < 20:
        summary = {"records": records, "n_imitate_steps": len(bank), "actor": None}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("TOO_FEW", len(bank), flush=True)
        return 0

    X = np.stack([b[0] for b in bank])
    Y = np.stack([b[1] for b in bank])
    fam = np.array([b[2] for b in bank])
    w = family_weights(fam)
    net, mae = fit_actor(X, Y, w)
    shares = {f: int(np.sum(fam == f)) for f in FAMILIES}
    print("BC mae", round(mae, 4), "steps", shares, flush=True)

    # Leave-one-family-out actors. Same loss, family column is not a feature.
    ablations = {}
    for left in FAMILIES:
        keep = fam != left
        if keep.sum() < 20 or len(set(fam[keep])) < 2:
            ablations[left] = None
            continue
        net_l, mae_l = fit_actor(X[keep], Y[keep], family_weights(fam[keep]), seed=1)
        ablations[left] = (net_l, mae_l)
        print("ABLATION", left, "mae", round(mae_l, 4), "n", int(keep.sum()), flush=True)

    def queries_for(n):
        base = [0, 16, 32, n]
        if n >= 60:
            base.append(60)
        return sorted(set(k for k in base if 0 <= k <= n))

    evals = []

    def run_eval(tag, net_e, ic, dyn_name, n_steps):
        mass, fr = dyns[dyn_name]
        # Before-recovery label is the IC itself, queried once per (ic, dyn).
        rolled = actor_rollout(
            sim, net_e, snaps[ic], nominal, pad_ids, pad_mu, mass, fr, n_steps, queries_for(n_steps)
        )
        if not rolled["pose_ok"]:
            evals.append({"tag": tag, "ic": ic, "dyn": dyn_name, "pose_ok": False})
            print("EVAL bad restore", tag, ic, flush=True)
            return
        labels = []
        # Drop adjacent duplicates: keep queries at least 4 steps apart, plus contact.
        chosen = []
        for k in sorted(rolled["snaps"]):
            if k == rolled.get("contact_k") or not chosen or k - chosen[-1] >= 4:
                chosen.append(k)
        for k in chosen:
            lab = oracle_snap(sim, gains, rolled["snaps"][k], nominal, pad_ids, pad_mu, mass, fr)
            kind = "HANDOFF_SUCCESS" if lab.get("y") == 1 else "HANDOFF_FAILURE"
            if k == 0:
                where = "before"
            elif k == rolled.get("contact_k"):
                where = "contact_transition"
            elif k > 52:
                where = "overcontrol"
            elif k == n_steps:
                where = "endpoint"
            else:
                where = "during"
            labels.append({"k": int(k), "where": where, "label": kind, "outcome": lab.get("outcome")})
        row = {
            "tag": tag, "ic": ic, "dyn": dyn_name, "pattern": rolled["pattern"],
            "n_steps": n_steps, "labels": labels,
            "any_success": any(x["label"] == "HANDOFF_SUCCESS" and x["k"] > 0 for x in labels),
            "endpoint": next((x["label"] for x in labels if x["where"] == "endpoint"), None),
        }
        evals.append(row)
        print("EVAL", tag, ic, dyn_name, rolled["pattern"], row["endpoint"], "any", row["any_success"], flush=True)

    horizon = 52
    run_eval("full", net, "impact_zm6", "nominal", horizon)
    run_eval("full", net, "impact_zp6", "nominal", horizon)
    run_eval("full", net, "impact_zp14", "nominal", horizon)
    run_eval("full", net, "impact_zm6", HELD_DYN, horizon)
    run_eval("full_over", net, "impact_zm6", "nominal", 60)
    for left, pack in ablations.items():
        if pack is None:
            continue
        run_eval("without_" + left, pack[0], "impact_zm6", "nominal", horizon)

    # Handoff dataset is the actor queries above, not a classifier fit.
    handoff_rows = []
    for ev in evals:
        if not ev.get("labels"):
            continue
        if not str(ev["tag"]).startswith("full"):
            continue
        for lab in ev["labels"]:
            handoff_rows.append({
                "ic": ev["ic"], "dyn": ev["dyn"], "k": lab["k"], "where": lab["where"],
                "label": lab["label"], "outcome": lab["outcome"], "pattern": ev["pattern"],
            })

    summary = {
        "action": audit,
        "obs_names": list(OBS_NAMES),
        "history": {"past_obs": HIST, "past_actions": HIST, "feat_dim": int(X.shape[1])},
        "records": records,
        "imitate_steps": shares,
        "bc_mae": mae,
        "ablation_mae": {k: None if v is None else v[1] for k, v in ablations.items()},
        "evals": evals,
        "handoff_queries": handoff_rows,
        "no_classifier": True,
        "learned_handoff_trained": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(OUT / "imitate_steps.npz", X=X, Y=Y, family=fam)
    print("DONE", shares, "handoff", len(handoff_rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
