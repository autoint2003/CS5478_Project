"""Phase 3–4 — bounded P73 CF replay + action-outcome discrimination."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p73_first_contact_bilateral.controller import EventRule
from experiments.precision.p73_first_contact_bilateral.paths import MAX_EP_STEPS, T_MAX
from experiments.precision.p73_first_contact_bilateral.phase23_cf import _continue
from experiments.precision.p77_contact_observability.metrics import eval_binary_probe
from experiments.precision.p78_action_relevant_representation.features import (
    obs27,
    rep_A_current,
    rep_B_current_plus_eng,
    rep_C_history,
    rep_D_history_plus_eng,
)
from experiments.precision.p78_action_relevant_representation.paths import (
    CACHE,
    CF_KIND,
    CF_MAX_SNAPS,
    CF_RULES,
    CF_SEEDS_PER,
    ENG_HIST,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    P73_CF,
    P73_SNAP,
    SEED,
)


def _rules() -> dict[str, EventRule | None]:
    return {
        "baseline": None,
        "soft_approach": EventRule(name="soft_approach", sequence=[56, 50, 71, 50, 56], max_steps=5),
        "grip_then_approach": EventRule(
            name="grip_then_approach", sequence=[82, 82, 56, 50, 71], max_steps=5
        ),
        "A_continue_h5": EventRule(name="A_continue_h5", sequence=[40, 56, 40, 40, 83], max_steps=5),
    }


def _extract_rep(env) -> dict:
    """Causal reps from env.obs_seq at restored snap (before CF action)."""
    frames = [obs27(np.asarray(o, float)) for o in env.obs_seq]
    if not frames:
        frames = [np.zeros(27)]
    hh = frames[-ENG_HIST:]
    hist = frames[-max(8, ENG_HIST) :]
    return {
        "A": rep_A_current(frames[-1]),
        "B": rep_B_current_plus_eng(hh),
        "C": rep_C_history(hist),
        "D": rep_D_history_plus_eng(hist),
        "n_hist": len(frames),
    }


def run_bounded_cf(*, device: str = "cuda") -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cf_diag_rows.pkl"
    if path.exists():
        rows = pickle.loads(path.read_bytes())
        return {"rows": rows, "cached": True}

    bank = json.loads(P73_SNAP.read_text(encoding="utf-8"))
    metas = list(bank.get(CF_KIND) or [])[:CF_MAX_SNAPS]
    if not metas:
        return {"rows": [], "error": "no snaps"}

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    schedule = build_p54_dataset(seed=5478)["schedule"]
    lookup = {s.traj_id: s for s in schedule}
    rules = _rules()
    rows = []
    for mi, sm in enumerate(metas):
        tid = sm["traj_id"]
        if tid not in lookup:
            continue
        start = lookup[tid]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        # Extract representation once per snap
        seed_all(int(sm.get("seed") or SEED), deterministic_torch=True)
        restore_full_state(env, state, start)
        try:
            ensure_inner_capture_buffers(env)
        except Exception:  # noqa: BLE001
            pass
        g0 = relative_geometry(env.env.sim)
        rep = _extract_rep(env)
        for rname in CF_RULES:
            if rname not in rules:
                continue
            for si in range(CF_SEEDS_PER):
                seed = int(sm.get("seed") or SEED) + 17 * si + 101 * hash(rname) % 1000
                out = _continue(
                    env, actor, critic, book, start, state, device_t, rule=rules[rname], seed=seed
                )
                rows.append(
                    {
                        "snap_id": sm["id"],
                        "traj_id": tid,
                        "rule": rname,
                        "seed": seed,
                        "e0": float(g0["e_pos_mm"]),
                        "sw0": int(g0["correct_sidewall"]),
                        "bil0": int(g0["bilateral"]),
                        **{f"rep_{k}": v for k, v in rep.items() if k in ("A", "B", "C", "D")},
                        "reached_opposing": out["reached_opposing"],
                        "reached_bilateral": out["reached_bilateral"],
                        "reached_e9": out["reached_e9"],
                        "r3_ready": out["r3_ready"],
                        "SUCCESS_RELAXED": out["SUCCESS_RELAXED"],
                        # contact preservation proxy: opposing or bilateral without early loss — use e9/opp
                        "contact_ok": int(out["reached_bilateral"] or out["reached_opposing"]),
                    }
                )
        if (mi + 1) % 4 == 0 or mi == 0:
            print(f"[cf] {mi+1}/{len(metas)} snaps rows={len(rows)}", flush=True)

    path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return {"rows": rows, "cached": False, "n_snaps": len(metas)}


def analyze_existing_cf_aggregates() -> dict:
    if not P73_CF.exists():
        return {"available": False}
    cf = json.loads(P73_CF.read_text(encoding="utf-8"))
    fc = cf.get("cf_results", {}).get("first_contact", {})
    base = fc.get("baseline", {})
    deltas = []
    for rule, blob in fc.items():
        if rule == "baseline":
            continue
        deltas.append(
            {
                "rule": rule,
                "d_opp": float(blob.get("reached_opposing", 0) - base.get("reached_opposing", 0)),
                "d_r3": float(blob.get("r3_ready", 0) - base.get("r3_ready", 0)),
                "d_e9": float(blob.get("reached_e9", 0) - base.get("reached_e9", 0)),
                "d_success": float(blob.get("SUCCESS_RELAXED", 0) - base.get("SUCCESS_RELAXED", 0)),
            }
        )
    return {
        "available": True,
        "baseline": base,
        "deltas": deltas,
        "note": (
            "Aggregate CF shows action sequences change R3/success from matched FC states. "
            "Per-snap outcomes require diagnostic replay (below)."
        ),
    }


def action_outcome_discrimination(rows: list[dict]) -> dict:
    """Predict CF outcomes from (state_rep ⊕ action_id); compare reps."""
    if len(rows) < 20:
        return {"insufficient": True, "n": len(rows)}
    rules = sorted({r["rule"] for r in rows})
    rule_idx = {r: i for i, r in enumerate(rules)}
    # IC split by traj
    ics = sorted({r["traj_id"] for r in rows})
    rng = np.random.default_rng(SEED + 9)
    rng.shuffle(ics)
    n_val = max(1, len(ics) // 3)
    eval_ics, fit_ics = set(ics[:n_val]), set(ics[n_val:])

    def onehot(rule):
        v = np.zeros(len(rules), float)
        v[rule_idx[rule]] = 1.0
        return v

    report = {"n": len(rows), "n_fit_ics": len(fit_ics), "n_eval_ics": len(eval_ics), "rules": rules, "tasks": {}}
    for ykey in ("reached_opposing", "contact_ok", "r3_ready", "SUCCESS_RELAXED", "reached_e9"):
        report["tasks"][ykey] = {}
        for rep in ("A", "B", "C", "D"):
            X = []
            y = []
            tids = []
            for r in rows:
                X.append(np.concatenate([np.asarray(r[f"rep_{rep}"], float), onehot(r["rule"])]))
                y.append(int(r[ykey]))
                tids.append(r["traj_id"])
            X = np.stack(X)
            y = np.asarray(y, np.int8)
            fit = np.asarray([t in fit_ics for t in tids])
            ev = np.asarray([t in eval_ics for t in tids])
            # baselines: action-only / state-only
            X_act = np.stack([onehot(r["rule"]) for r in rows])
            X_state = np.stack([np.asarray(r[f"rep_{rep}"], float) for r in rows])
            report["tasks"][ykey][rep] = {
                "state_plus_action": eval_binary_probe(X, y, fit, ev),
                "action_only": eval_binary_probe(X_act, y, fit, ev),
                "state_only": eval_binary_probe(X_state, y, fit, ev),
            }
    return report


def matched_action_ranking(rows: list[dict]) -> dict:
    """Within-snap: does higher predicted score rank the better rule?"""
    by_snap = defaultdict(list)
    for r in rows:
        by_snap[r["snap_id"]].append(r)
    # Prefer soft_approach over grip_then_approach when they differ on r3 (from P73 knowledge)
    # Ranking metric: for each snap with ≥2 rules, pairwise accuracy using state_only score of B
    # Fit a simple model on fit snaps: score = w·rep_B predicting r3_ready, then rank rules by... 
    # Actually for ranking we need score(state, action). Use fitted state+action probe.
    if len(by_snap) < 6:
        return {"insufficient": True, "n_snaps": len(by_snap), "reason": "too few snaps for held-out ranking"}

    snaps = sorted(by_snap.keys())
    rng = np.random.default_rng(SEED + 11)
    rng.shuffle(snaps)
    n_te = max(2, len(snaps) // 3)
    te_snaps, tr_snaps = set(snaps[:n_te]), set(snaps[n_te:])
    rules = sorted({r["rule"] for r in rows})
    rule_idx = {r: i for i, r in enumerate(rules)}

    def pack(rep):
        Xs, ys = [], []
        for r in rows:
            if r["snap_id"] not in tr_snaps:
                continue
            oh = np.zeros(len(rules))
            oh[rule_idx[r["rule"]]] = 1
            Xs.append(np.concatenate([np.asarray(r[f"rep_{rep}"], float), oh]))
            ys.append(float(r["r3_ready"]))
        return np.stack(Xs), np.asarray(ys)

    ranking = {}
    for rep in ("A", "B", "C", "D"):
        Xtr, ytr = pack(rep)
        if len(np.unique((ytr > 0.5).astype(int))) < 2 or len(Xtr) < 8:
            ranking[rep] = {"insufficient": True}
            continue
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        A = np.column_stack([(Xtr - mu) / sd, np.ones(len(Xtr))])
        coef = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ ytr)
        # pairwise on test snaps
        correct = total = 0
        for sid in te_snaps:
            group = by_snap[sid]
            # all pairs of rules with different outcomes
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ri, rj = group[i], group[j]
                    if ri["r3_ready"] == rj["r3_ready"]:
                        continue
                    def score(r):
                        oh = np.zeros(len(rules))
                        oh[rule_idx[r["rule"]]] = 1
                        x = np.concatenate([np.asarray(r[f"rep_{rep}"], float), oh])
                        xz = (x - mu) / sd
                        return float(np.concatenate([xz, [1.0]]) @ coef)

                    si, sj = score(ri), score(rj)
                    better = ri if ri["r3_ready"] > rj["r3_ready"] else rj
                    worse = rj if better is ri else ri
                    pred_better = ri if si >= sj else rj
                    correct += int(pred_better is better)
                    total += 1
        ranking[rep] = {
            "pairwise_accuracy": float(correct / total) if total else None,
            "n_pairs": total,
            "n_test_snaps": len(te_snaps),
        }
    # Chance = 0.5 for pairs with different labels
    ranking["note"] = (
        "Held-out within-snap pairwise ranking on r3_ready. "
        "If n_pairs small or accuracy≈0.5, mark action relevance inconclusive."
    )
    return ranking


def run_cf_pipeline(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    agg = analyze_existing_cf_aggregates()
    print("=== Bounded CF diagnostic replay ===", flush=True)
    cf = run_bounded_cf(device=device)
    rows = cf.get("rows") or []
    # numpy arrays in rows — convert for JSON later
    disc = action_outcome_discrimination(rows)
    rank = matched_action_ranking(rows)

    # Serialize rows summary without huge arrays
    summary_rows = [
        {
            "snap_id": r["snap_id"],
            "traj_id": r["traj_id"],
            "rule": r["rule"],
            "r3_ready": r["r3_ready"],
            "reached_opposing": r["reached_opposing"],
            "SUCCESS_RELAXED": r["SUCCESS_RELAXED"],
            "reached_e9": r["reached_e9"],
        }
        for r in rows
    ]
    report = {
        "existing_aggregates": agg,
        "bounded_replay": {
            "n_rows": len(rows),
            "n_snaps": len({r["snap_id"] for r in rows}),
            "rules": list(CF_RULES),
            "cached": cf.get("cached"),
        },
        "action_outcome_discrimination": disc,
        "matched_action_ranking": rank,
        "row_summary": summary_rows[:50],
    }
    # JSON-safe: convert numpy in disc already via eval_binary_probe floats
    (OUT / "COUNTERFACTUAL_ACTION_ANALYSIS.json").write_text(
        json.dumps({"existing_aggregates": agg, "bounded_replay": report["bounded_replay"], "row_summary": summary_rows}, indent=2),
        encoding="utf-8",
    )
    (OUT / "COUNTERFACTUAL_ACTION_ANALYSIS.md").write_text(
        "# Counterfactual Action Analysis (P78 Phase 3)\n\n```json\n"
        + json.dumps({"aggregates": agg, "replay": report["bounded_replay"]}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "ACTION_OUTCOME_DISCRIMINATION.json").write_text(
        json.dumps({"discrimination": disc, "ranking": rank}, indent=2),
        encoding="utf-8",
    )
    # Compact AUROCs
    compact = {}
    for task, reps in (disc.get("tasks") or {}).items():
        compact[task] = {
            rep: {
                "state_plus_action": (blob.get("state_plus_action") or {}).get("auroc"),
                "action_only": (blob.get("action_only") or {}).get("auroc"),
                "state_only": (blob.get("state_only") or {}).get("auroc"),
            }
            for rep, blob in reps.items()
        }
    (OUT / "ACTION_OUTCOME_DISCRIMINATION.md").write_text(
        "# Action-Outcome Discrimination (P78 Phase 4)\n\n```json\n"
        + json.dumps({"auroc_compact": compact, "ranking": rank}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"n_rows": len(rows), "ranking": rank, "r3_B": compact.get("r3_ready", {}).get("B")}, indent=2), flush=True)
    # stash rows for aliasing
    (CACHE / "cf_rows_meta.json").write_text(json.dumps({"n": len(rows)}), encoding="utf-8")
    return {"rows": rows, "disc": disc, "rank": rank, "agg": agg}


if __name__ == "__main__":
    run_cf_pipeline()
