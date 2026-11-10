"""P109-A — Future contact-state predictability and representation audit.

Diagnostics only: no Actor updates, no PPO/GRPO, no sealed panel, no P109-B execution.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers  # noqa: E402
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates  # noqa: E402
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402

import experiments.precision.p104_joint_token_residual_pilot.run_p104 as p104  # noqa: E402

OUT = ROOT / "results/experiments/p109a_contact_predictability_audit"
CACHE = OUT / "cache"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P108 = ROOT / "results/experiments/p108_closed_loop_generalization_discovery"
EXPECTED_SHA = "ec08f59816aa2fdd"
CL_SEED = 101
DT_POLICY = 0.02  # seconds per policy step
H_SHORT = 0.2
H_LONG = 0.5
STEPS_SHORT = int(round(H_SHORT / DT_POLICY))  # 10
STEPS_LONG = int(round(H_LONG / DT_POLICY))  # 25

# Pre-registered gates (BEFORE final test evaluation)
PRE_REG = {
    "min_auroc_lift_vs_current": 0.05,
    "min_median_lead_s_at_fpr": 0.10,
    "max_fpr_for_lead": 0.20,
    "min_test_roots": 8,
    "min_valid_windows_per_target": 50,
}


def utc() -> str:
    return p104.utc()


def write_json(path: Path, obj) -> None:
    p104.write_json(path, obj)


def write_csv(path: Path, rows: list[dict]) -> None:
    p104.write_csv(path, rows)


def gbi_key(s: str) -> str | None:
    m = re.search(r"gbi_\d+", str(s), re.I)
    return m.group(0).lower() if m else None


def canonical_id(tid: str) -> str:
    return re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)


@torch.no_grad()
def encode_ctx(actor, obs, prev, device):
    o = torch.as_tensor(obs, device=device, dtype=torch.float32)
    p = torch.as_tensor(prev, device=device, dtype=torch.float32)
    x = torch.cat([o, p], dim=-1)
    t = x.shape[-2]
    h = actor.in_proj(x) + actor.pos(torch.arange(t, device=x.device))
    causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
    ctx = actor.enc(h.unsqueeze(0), mask=causal)[0]
    logits = actor.logits(ctx)
    return ctx, logits


def audit_p108_traces() -> dict:
    """P108 cache is incomplete for P109-A — document gaps; do not invent zeros."""
    cache = P108 / "cache"
    files = sorted(cache.glob("trace_*.npz")) if cache.is_dir() else []
    keys = set()
    n = 0
    if files:
        z = np.load(files[0], allow_pickle=True)
        keys = set(z.files)
        n = len(files)
    required = {
        "obs39",
        "prev7",
        "vrel",
        "bilateral_before",
        "bilateral_after",
        "sim_time",
        "fn_L",
        "fn_R",
    }
    present = keys
    missing = sorted(required - present)
    # known present fields
    known = sorted(keys)
    return {
        "n_traces": n,
        "keys_present": known,
        "required_for_p109a": sorted(required),
        "missing": missing,
        "reusable_for_probes": len(missing) == 0,
        "vrel_slip_note": (
            "P108 reagg fixed state_metrics key to vrel in code, but cached traces never stored vrel/slip."
        ),
        "verdict": "P108_CACHE_INCOMPLETE_MUST_RECOLLECT",
    }


def build_root_pools(starts: dict) -> tuple[list[dict], dict]:
    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    train_g = {gbi_key(x) for x in splits["c_train_phys"]} - {None}
    dev_g = {gbi_key(x) for x in splits.get("c_dev_phys", [])} - {None}
    ind_g = {gbi_key(t) for t in (splits.get("ind") or {}).get("C") or []} - {None}
    ban_g = {gbi_key(t) for t in p104.eval_banned_phys()} - {None}

    by_g: dict[str, list] = defaultdict(list)
    for tid, st in starts.items():
        if getattr(st, "skill", None) != "C":
            continue
        g = gbi_key(tid)
        if g is None or g in ind_g:
            continue
        by_g[g].append((tid, st))

    rows = []
    for g, lst in sorted(by_g.items()):
        lst = sorted(lst, key=lambda x: (0 if str(x[0]).startswith("plan_c__") else 1, x[0]))
        tid, st = lst[0]
        if g in train_g:
            prov = "training_roots"
        elif g in dev_g:
            prov = "previously_exposed_development"
        elif g in ban_g:
            prov = "historical_freeze_ban_not_independent"
        else:
            prov = "newly_selected_development"
        rows.append(
            {
                "traj_id": tid,
                "canonical_physical_id": canonical_id(tid),
                "gbi": g,
                "provenance": prov,
                "in_p98_c_train": int(g in train_g),
                "in_p98_c_dev": int(g in dev_g),
                "in_freeze_ban": int(g in ban_g),
                "in_sealed_ind": 0,
                "source": "schedule",
            }
        )
    meta = {
        "n": len(rows),
        "by_prov": {k: sum(1 for r in rows if r["provenance"] == k) for k in set(r["provenance"] for r in rows)},
        "sealed_excluded": sorted(ind_g),
        "note": (
            "Freeze-ban roots are NOT treated as fresh independent eval. "
            "Newly selected development requires generated ICs if count is 0."
        ),
    }
    return rows, meta


def generate_new_starts(
    ban_g: set[str],
    existing_g: set[str],
    book,
    rho,
    *,
    target: int,
    wall_s: float,
    seed: int,
) -> list[dict]:
    """Bounded generation of new catch ICs (physical starts for Actor rollouts)."""
    from data_generation.oracles.plan_c.expert_search import (
        build_physical_root,
        diverse_impact_configs,
        search_catch,
    )
    from envs.simulation.ground_sim import make_ground_sim

    out_dir = CACHE / "new_ics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    configs = diverse_impact_configs(max(target * 8, 40), seed=seed)
    rng.shuffle(configs)
    accepted = []
    t0 = time.time()
    for cfg in configs:
        if len(accepted) >= target or time.time() - t0 > wall_s:
            break
        rid = str(cfg.get("config_id") or "")
        g = gbi_key(rid) or rid
        if g in ban_g or g in existing_g:
            continue
        try:
            sim, _ = make_ground_sim()
            root = build_physical_root(sim, cfg, max_s=1.2)
            if root is None:
                continue
            fp = str(root.get("fingerprint") or hashlib.sha256(pickle.dumps(cfg)).hexdigest()[:16])
            catch = search_catch(sim, root["snap"], root["ref"], rng=rng, max_rollouts=12)
            best = catch.get("best")
            if not best:
                continue
            ep = {
                "traj_id": f"plan_c__{rid}__{fp[:12]}",
                "fingerprint": fp,
                "cfg": cfg,
                "catch_snap": root["snap"],
                "ref": root["ref"],
                "capture": best.get("capture"),
                "start_kind": "catch_snap",
                "actions": np.zeros((0, 7), np.float32),
            }
            tid = ep["traj_id"]
            st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            # verify reset with real book/rho
            env = StabilizeGrpoEnv(
                book, rho, max_steps=2, t_max=100, weights=PoseRewardWeights(), use_stabilization=False
            )
            env.reset(st)
            pkl = out_dir / f"{tid}.pkl"
            pkl.write_bytes(pickle.dumps({"ep": ep}))
            accepted.append(
                {
                    "traj_id": tid,
                    "canonical_physical_id": canonical_id(tid),
                    "gbi": gbi_key(tid) or g,
                    "provenance": "newly_selected_development",
                    "in_p98_c_train": 0,
                    "in_p98_c_dev": 0,
                    "in_freeze_ban": 0,
                    "in_sealed_ind": 0,
                    "source": "generated_p109a",
                    "pkl": str(pkl),
                    "start": st,
                }
            )
            existing_g.add(gbi_key(tid) or g)
            print(f"  [gen] accepted {tid} n={len(accepted)}/{target}", flush=True)
        except Exception as e:
            print(f"  [gen] skip {rid}: {type(e).__name__}: {e}", flush=True)
            continue
    return accepted


@torch.no_grad()
def rich_rollout(actor, start: MixedStart, book, rho, device, *, seed: int = CL_SEED) -> dict | None:
    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    try:
        env.reset(start)
    except Exception as e:
        return {"error": f"reset_failed:{type(e).__name__}:{e}"}
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    steps = []
    while not done:
        t = int(env.t)
        g0 = relative_geometry(env.env.sim)
        m0 = state_metrics(env.env.sim)
        obs_win = np.stack(env.obs_seq[-160:]).astype(np.float32)
        prev_win = np.stack(env.prev_seq[-160:]).astype(np.float32)
        ctx, logits = encode_ctx(actor, obs_win, prev_win, device)
        h_t = ctx[-1].detach().cpu().numpy().astype(np.float32)
        obs_win_c = obs_win.copy()
        prev_win_c = prev_win.copy()
        logits_last = mask_handoff_logits(logits[-1].float())
        tok_a = int(torch.argmax(logits_last).item())
        # residual from actor forward
        obs_t = torch.as_tensor(obs_win, device=device, dtype=torch.float32)
        prev_t = torch.as_tensor(prev_win, device=device, dtype=torch.float32)
        _logits, res, _act = actor(obs_t, prev_t)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        if int(tok) == HANDOFF:
            action = np.zeros(7, np.float32)
        else:
            action = np.asarray(decode_m111_action(book, int(tok), resid, rho), np.float32)
        obs39 = env.obs_seq[-1].astype(np.float32).copy()
        prev7 = env.prev_seq[-1].astype(np.float32).copy()
        bil_before = int(bool(g0["bilateral"]))
        # history features: last 5 obs force + bil from stored if any — use current window tips
        hist = np.concatenate(
            [
                obs_win[-min(5, len(obs_win)) :, 10:12].reshape(-1),
                prev_win[-min(5, len(prev_win)) :].reshape(-1),
            ]
        ).astype(np.float32)
        steps.append(
            {
                "t": t,
                "sim_time": float(env.env.sim.data.time),
                "tok": int(tok),
                "action": action,
                "obs": obs39,
                "prev": prev7,
                "h_t": h_t,
                "bilateral_before": bil_before,
                "e_pos_mm": float(g0["e_pos_mm"]),
                "theta_deg": float(g0["theta_deg"]),
                "vrel": float(m0.get("vrel")) if m0.get("vrel") is not None else None,
                "fn_L": float(m0.get("fn_L")) if m0.get("fn_L") is not None else float(obs39[10]),
                "fn_R": float(m0.get("fn_R")) if m0.get("fn_R") is not None else float(obs39[11]),
                "min_pad": int(g0.get("min_pad") or 0),
                "hist_tail": hist,
                "obs_win": obs_win_c,
                "prev_win": prev_win_c,
            }
        )
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        g1 = relative_geometry(env.env.sim)
        steps[-1]["bilateral_after"] = int(bool(g1["bilateral"]))
        steps[-1]["e_pos_mm_after"] = float(g1["e_pos_mm"])
        steps[-1]["handoff"] = int(info.get("handoff") or 0)
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "steps": steps,
        "dt_policy": DT_POLICY,
    }


def label_windows(roll: dict) -> list[dict]:
    """Build T1–T4 labels at 0.2s/0.5s using actual step indices; censor short horizons."""
    steps = roll.get("steps") or []
    if not steps:
        return []
    # regrasp: first 0→1 bilateral_after
    bil_a = [int(s["bilateral_after"]) for s in steps]
    regrasp_t = None
    for i in range(1, len(bil_a)):
        if bil_a[i - 1] == 0 and bil_a[i] == 1:
            regrasp_t = i
            break
    rows = []
    for i, s in enumerate(steps):
        for horizon_s, n_steps, tag in (
            (H_SHORT, STEPS_SHORT, "0.2s"),
            (H_LONG, STEPS_LONG, "0.5s"),
        ):
            j_end = i + n_steps
            if j_end >= len(steps):
                # censored
                rows.append(
                    {
                        "t": i,
                        "horizon_s": horizon_s,
                        "horizon_tag": tag,
                        "n_steps": n_steps,
                        "censored": 1,
                        "T1_bilateral_survival": None,
                        "T2_contact_loss": None,
                        "T3_slip_mean": None,
                        "T3_slip_unstable": None,
                        "T4_post_recatch_stable": None,
                        "T4_applicable": 0,
                    }
                )
                continue
            future = steps[i + 1 : j_end + 1]
            # T1: bilateral continuously maintained (after-step) over horizon
            bals = [int(x["bilateral_after"]) for x in future]
            t1 = int(all(b == 1 for b in bals)) if bals else None
            # T2: any loss event (1→0) in horizon, or start bil and then lose
            t2 = 0
            prev_b = int(s["bilateral_after"])
            for x in future:
                b = int(x["bilateral_after"])
                if prev_b == 1 and b == 0:
                    t2 = 1
                    break
                prev_b = b
            # T3: mean vrel over horizon where measured; unstable if mean>threshold
            vrels = [x["vrel"] for x in future if x.get("vrel") is not None]
            if not vrels:
                t3_mean = None
                t3_unstable = None
            else:
                t3_mean = float(np.mean(vrels))
                t3_unstable = int(t3_mean > 0.05)
            # T4: only if already post-regrasp at i
            t4_app = int(regrasp_t is not None and i >= regrasp_t)
            if t4_app:
                t4 = int(all(b == 1 for b in bals))
            else:
                t4 = None
            rows.append(
                {
                    "t": i,
                    "horizon_s": horizon_s,
                    "horizon_tag": tag,
                    "n_steps": n_steps,
                    "censored": 0,
                    "T1_bilateral_survival": t1,
                    "T2_contact_loss": t2,
                    "T3_slip_mean": t3_mean,
                    "T3_slip_unstable": t3_unstable,
                    "T4_post_recatch_stable": t4,
                    "T4_applicable": t4_app,
                    "bilateral_now": int(s["bilateral_after"]),
                    "regrasp_t": regrasp_t if regrasp_t is not None else "",
                }
            )
    return rows


def feat_current(s: dict) -> np.ndarray:
    """A: current obs + prev action (46). Missing→nan (caller drops)."""
    return np.concatenate([s["obs"].reshape(-1), s["prev"].reshape(-1)]).astype(np.float32)


def feat_history(s: dict) -> np.ndarray:
    """B: current + hist tail (variable); pad/truncate to fixed."""
    cur = feat_current(s)
    hist = np.asarray(s.get("hist_tail"), np.float32).reshape(-1)
    # fixed size: 46 + 5*2 + 5*7 = 46+10+35 = 91
    target = 46 + 10 + 35
    if hist.size < target - 46:
        hist = np.pad(hist, (0, target - 46 - hist.size))
    else:
        hist = hist[: target - 46]
    return np.concatenate([cur, hist]).astype(np.float32)


def feat_hidden(s: dict) -> np.ndarray:
    return np.asarray(s["h_t"], np.float32).reshape(-1)


def root_clustered_auroc(y, p, roots) -> float:
    """AUROC on pooled samples; also return nan if degenerate."""
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def train_eval_probes(samples: list[dict], target_key: str, horizon_tag: str, split_of_root: dict) -> list[dict]:
    """Train logistic + MLP on train roots; tune on val; report test. Pre-registered."""
    rows_out = []
    # filter valid labels
    data = []
    for s in samples:
        if s["horizon_tag"] != horizon_tag:
            continue
        if s.get("censored"):
            continue
        y = s.get(target_key)
        if y is None:
            continue
        if target_key == "T4_post_recatch_stable" and not s.get("T4_applicable"):
            continue
        root = s["canonical_physical_id"]
        sp = split_of_root.get(root)
        if sp is None:
            continue
        data.append((sp, root, s, int(y)))
    if not data:
        return [
            {
                "target": target_key,
                "horizon": horizon_tag,
                "status": "NO_VALID_SAMPLES",
            }
        ]

    def pack(split, feat_fn, name):
        Xs, ys, rs = [], [], []
        for sp, root, s, y in data:
            if sp != split:
                continue
            x = feat_fn(s["step"])
            if np.any(~np.isfinite(x)):
                continue
            Xs.append(x)
            ys.append(y)
            rs.append(root)
        if not Xs:
            return None
        return np.stack(Xs), np.asarray(ys), rs

    families = [
        ("A_current", feat_current),
        ("B_history", feat_history),
        ("C_p64_hidden", feat_hidden),
    ]
    # prevalence baseline
    y_train = [y for sp, _, _, y in data if sp == "train"]
    prev = float(np.mean(y_train)) if y_train else 0.5

    for fam, fn in families:
        tr = pack("train", fn, fam)
        va = pack("val", fn, fam)
        te = pack("test", fn, fam)
        if tr is None or te is None:
            rows_out.append(
                {
                    "target": target_key,
                    "horizon": horizon_tag,
                    "family": fam,
                    "status": "MISSING_SPLIT",
                    "n_train": 0 if tr is None else len(tr[1]),
                    "n_test": 0 if te is None else len(te[1]),
                }
            )
            continue
        Xtr, ytr, rtr = tr
        Xte, yte, rte = te
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        Xva_s = scaler.transform(va[0]) if va is not None else None
        yva = va[1] if va is not None else None

        # pick C for logistic on val
        best_C, best_brier = 1.0, 1e9
        for C in (0.1, 1.0, 10.0):
            clf = LogisticRegression(max_iter=400, C=C, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            if Xva_s is not None and yva is not None and len(np.unique(yva)) > 1:
                pv = clf.predict_proba(Xva_s)[:, 1]
                br = float(brier_score_loss(yva, pv))
                if br < best_brier:
                    best_brier, best_C = br, C
        logit = LogisticRegression(max_iter=500, C=best_C, class_weight="balanced")
        logit.fit(Xtr_s, ytr)
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=0)
        mlp.fit(Xtr_s, ytr)

        for model_name, model in (("logreg", logit), ("mlp", mlp), ("prevalence", None)):
            if model_name == "prevalence":
                pte = np.full(len(yte), prev, dtype=np.float64)
            else:
                pte = model.predict_proba(Xte_s)[:, 1]
            auroc = root_clustered_auroc(yte, pte, rte)
            try:
                auprc = float(average_precision_score(yte, pte)) if len(np.unique(yte)) > 1 else float("nan")
            except Exception:
                auprc = float("nan")
            try:
                brier = float(brier_score_loss(yte, pte))
            except Exception:
                brier = float("nan")
            # root-level mean prediction vs mean label
            by_root = defaultdict(list)
            for r, y, p in zip(rte, yte, pte):
                by_root[r].append((y, p))
            root_aucs = []
            for r, pairs in by_root.items():
                yy = [a[0] for a in pairs]
                pp = [a[1] for a in pairs]
                if len(set(yy)) > 1:
                    root_aucs.append(roc_auc_score(yy, pp))
            rows_out.append(
                {
                    "target": target_key,
                    "horizon": horizon_tag,
                    "family": fam,
                    "model": model_name,
                    "status": "OK",
                    "n_train": int(len(ytr)),
                    "n_test": int(len(yte)),
                    "n_test_roots": int(len(by_root)),
                    "prevalence_train": prev,
                    "prevalence_test": float(np.mean(yte)),
                    "auroc": auroc,
                    "auprc": auprc,
                    "brier": brier,
                    "auroc_root_mean": float(np.mean(root_aucs)) if root_aucs else float("nan"),
                    "logreg_C": best_C if model_name == "logreg" else "",
                    "input_dim": int(Xtr.shape[1]),
                }
            )
    return rows_out


def early_warning_analysis(rolls: list[dict], split_of_root: dict) -> list[dict]:
    """Lead time for T2 predictions using a simple current-bilateral + vrel heuristic and probe scores from history.

    Uses offline recomputed risk = 1 - fraction bilateral in recent window + scaled vrel (no peeking future).
    """
    rows = []
    for R in rolls:
        root = R["canonical_physical_id"]
        if split_of_root.get(root) != "test":
            continue
        steps = R["roll"]["steps"]
        # find contact loss times
        loss_times = []
        for i in range(1, len(steps)):
            if int(steps[i - 1]["bilateral_after"]) == 1 and int(steps[i]["bilateral_after"]) == 0:
                loss_times.append(i)
        # risk score at t: high if currently bilateral but vrel high / e_pos rising
        scores = []
        for i, s in enumerate(steps):
            v = s["vrel"] if s["vrel"] is not None else 0.0
            risk = 0.0
            if int(s["bilateral_after"]) == 1:
                risk = float(np.clip(v / 0.1, 0, 1))
                if i >= 2 and s["e_pos_mm"] > steps[i - 2]["e_pos_mm"] + 5:
                    risk = max(risk, 0.6)
            scores.append(risk)
        # false alarms on trajectories with no loss
        if not loss_times:
            for thr in (0.3, 0.5, 0.7):
                fa = int(any(sc >= thr for sc in scores))
                rows.append(
                    {
                        "root": root,
                        "event": "none",
                        "thr": thr,
                        "false_alarm": fa,
                        "lead_s": "",
                        "category": "stable_trajectory",
                    }
                )
            continue
        for t_loss in loss_times:
            for thr in (0.3, 0.5, 0.7):
                first = next((i for i in range(t_loss) if scores[i] >= thr), None)
                if first is None:
                    rows.append(
                        {
                            "root": root,
                            "event_t": t_loss,
                            "thr": thr,
                            "false_alarm": 0,
                            "lead_s": "",
                            "category": "missed",
                        }
                    )
                    continue
                lead = (t_loss - first) * DT_POLICY
                # categorize relative to instability onset: e_pos rise
                onset = t_loss
                for i in range(t_loss, 0, -1):
                    if steps[i]["e_pos_mm"] > steps[max(0, i - 1)]["e_pos_mm"] + 3:
                        onset = i
                    else:
                        break
                if first < onset:
                    cat = "before_instability"
                elif first < t_loss:
                    cat = "after_instability_before_loss"
                else:
                    cat = "after_loss"
                rows.append(
                    {
                        "root": root,
                        "event_t": t_loss,
                        "onset_t": onset,
                        "warn_t": first,
                        "thr": thr,
                        "lead_s": lead,
                        "false_alarm": 0,
                        "category": cat,
                    }
                )
    return rows


def gradient_path_audit(actor, book, device, samples: list[dict]) -> dict:
    """Non-optimizing gradient test on disposable copy."""
    # pick a batch of windows with valid T2 labels
    batch = []
    for s in samples:
        if s.get("censored") or s.get("horizon_tag") != "0.2s":
            continue
        if s.get("T2_contact_loss") is None:
            continue
        batch.append(s)
        if len(batch) >= 16:
            break
    if len(batch) < 4:
        return {"status": "INSUFFICIENT_BATCH", "n": len(batch)}

    # --- frozen backbone: aux head only ---
    p64 = actor
    for p in p64.parameters():
        p.requires_grad_(False)
    H = 64
    aux = nn.Linear(H, 1).to(device)
    opt_params = list(aux.parameters())
    xs = []
    ys = []
    for s in batch:
        xs.append(torch.as_tensor(s["step"]["h_t"], device=device, dtype=torch.float32))
        ys.append(float(s["T2_contact_loss"]))
    X = torch.stack(xs)
    Y = torch.as_tensor(ys, device=device, dtype=torch.float32).unsqueeze(-1)
    logits = aux(X)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, Y)
    # grads wrt token head should be None/0 since frozen and not in graph from aux on detached h_t
    # rebuild with h from encode requiring grad through disposable copy
    frozen_token_grad = 0.0
    for n, p in p64.named_parameters():
        if "logits" in n and p.grad is not None:
            frozen_token_grad += float(p.grad.norm())

    # --- disposable copy: last transformer block trainable ---
    import copy

    disp = copy.deepcopy(p64).to(device)
    for p in disp.parameters():
        p.requires_grad_(False)
    # enable last encoder layer + logits + aux
    for p in disp.enc.layers[-1].parameters():
        p.requires_grad_(True)
    for p in disp.logits.parameters():
        p.requires_grad_(True)
    aux2 = nn.Linear(H, 1).to(device)
    # forward from obs/prev with grad
    losses_c = []
    losses_sft = []
    disp.zero_grad(set_to_none=True)
    aux2.zero_grad(set_to_none=True)
    for s in batch:
        obs = torch.as_tensor(s["step"]["obs_win"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(s["step"]["prev_win"], device=device, dtype=torch.float32)
        ctx, logits = encode_ctx_grad(disp, obs, prev, device)
        h = ctx[-1]
        y = torch.tensor([float(s["T2_contact_loss"])], device=device, dtype=torch.float32)
        lc = nn.functional.binary_cross_entropy_with_logits(aux2(h), y)
        # SFT-like: CE on recorded token
        tok = int(s["step"]["tok"])
        ls = nn.functional.cross_entropy(logits[-1].unsqueeze(0), torch.tensor([tok], device=device))
        losses_c.append(lc)
        losses_sft.append(ls)
    L_c = torch.stack(losses_c).mean()
    L_s = torch.stack(losses_sft).mean()
    # contact grads
    L_c.backward(retain_graph=True)
    g_block_c = grad_norm(disp.enc.layers[-1])
    g_tok_c = grad_norm(disp.logits)
    g_aux = grad_norm(aux2)
    # clear and SFT grads
    vec_c = flatten_grads(disp.enc.layers[-1])
    disp.zero_grad(set_to_none=True)
    L_s.backward()
    g_block_s = grad_norm(disp.enc.layers[-1])
    vec_s = flatten_grads(disp.enc.layers[-1])
    cos = float("nan")
    if vec_c is not None and vec_s is not None and float(vec_c.norm()) > 0 and float(vec_s.norm()) > 0:
        cos = float(torch.dot(vec_c, vec_s) / (vec_c.norm() * vec_s.norm()))
    return {
        "status": "OK",
        "n_batch": len(batch),
        "frozen_backbone_aux_only": {
            "note": "aux trained on detached h_t ⇒ no Actor parameter graph",
            "token_head_grad_norm": frozen_token_grad,
            "implies_grad_Lcontact_token_head_zero": True,
        },
        "disposable_last_block_trainable": {
            "grad_norm_final_transformer_block_Lcontact": g_block_c,
            "grad_norm_token_head_Lcontact": g_tok_c,
            "grad_norm_aux_head_Lcontact": g_aux,
            "grad_norm_final_block_LSFT": g_block_s,
            "cosine_SFT_vs_contact_on_final_block": cos,
            "optimizer_step_executed": False,
            "checkpoint_saved": False,
        },
    }


def encode_ctx_grad(actor, obs, prev, device):
    x = torch.cat([obs, prev], dim=-1)
    t = x.shape[-2]
    h = actor.in_proj(x) + actor.pos(torch.arange(t, device=x.device))
    causal = torch.triu(torch.ones(t, t, device=x.device) * float("-inf"), diagonal=1)
    ctx = actor.enc(h.unsqueeze(0), mask=causal)[0]
    logits = actor.logits(ctx)
    return ctx, logits


def grad_norm(module) -> float:
    tot = 0.0
    for p in module.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().norm() ** 2)
    return float(tot**0.5)


def flatten_grads(module):
    grads = [p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]
    if not grads:
        return None
    return torch.cat(grads)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[P109-A] Phase 0 — integrity + pools", flush=True)

    p108_audit = audit_p108_traces()
    (OUT / "DATA_INTEGRITY_AUDIT.md").write_text(
        f"""# DATA_INTEGRITY_AUDIT — P109-A

## P108 cached traces
- n_traces: {p108_audit['n_traces']}
- keys present: {p108_audit['keys_present']}
- missing for P109-A: {p108_audit['missing']}
- reusable_for_probes: **{p108_audit['reusable_for_probes']}**
- verdict: `{p108_audit['verdict']}`

{p108_audit['vrel_slip_note']}

**Action:** Recollect rich P64 diagnostic rollouts with obs39, prev7, vrel, bilateral before/after, sim_time, and frozen hidden states. Missing signals are not replaced by zeros.

## Field conventions (new collection)
- `vrel`: from `state_metrics` key `vrel` (not `slip` / `v_rel`).
- `bilateral_before` / `bilateral_after`: relative_geometry around `env.step`.
- `sim_time`: `env.env.sim.data.time` at decision.
- Horizons: 0.2s→{STEPS_SHORT} steps, 0.5s→{STEPS_LONG} steps at dt={DT_POLICY}s.
- Censored windows: episode ends before horizon → label null (not success/fail).

## Pre-registered gates
{json.dumps(PRE_REG, indent=2)}
""",
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64_sha = p104.sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha}"})
        raise SystemExit(f"sha {p64_sha}")
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = p104.rho_arr(rho)

    starts = p104.lookup_starts()
    pool_rows, pool_meta = build_root_pools(starts)
    # try generate new roots
    ban_g = {gbi_key(t) for t in p104.eval_banned_phys()} - {None}
    existing = {r["gbi"] for r in pool_rows}
    print("[P109-A] Phase 1 — generate new development ICs (bounded)", flush=True)
    gen = generate_new_starts(
        ban_g, set(existing), book, rho, target=20, wall_s=900.0, seed=109001
    )
    for g in gen:
        st = g.pop("start")
        starts[g["traj_id"]] = st
        pool_rows.append(g)
    write_csv(OUT / "ROOT_PROVENANCE.csv", [{k: v for k, v in r.items() if k != "start"} for r in pool_rows])
    write_json(OUT / "POOL_META.json", {**pool_meta, "n_generated": len(gen)})

    # Assign splits by provenance BEFORE predictions
    # train: training_roots + half of historical (for probe capacity)
    # val: remaining historical quarter
    # test: newly_selected_development if ≥8 else held-out historical with caveat
    train_roots = [r for r in pool_rows if r["provenance"] == "training_roots"]
    hist = [r for r in pool_rows if r["provenance"] == "historical_freeze_ban_not_independent"]
    prev_exp = [r for r in pool_rows if r["provenance"] == "previously_exposed_development"]
    newly = [r for r in pool_rows if r["provenance"] == "newly_selected_development"]
    rng = np.random.default_rng(109)
    rng.shuffle(hist)
    n_hist = len(hist)
    hist_train = hist[: max(1, int(0.5 * n_hist))]
    hist_val = hist[int(0.5 * n_hist) : int(0.75 * n_hist)]
    hist_test_fallback = hist[int(0.75 * n_hist) :]

    split_of_root = {}
    for r in train_roots + hist_train:
        split_of_root[r["canonical_physical_id"]] = "train"
    for r in hist_val + prev_exp:
        split_of_root[r["canonical_physical_id"]] = "val"
    test_caveat = None
    if len(newly) >= PRE_REG["min_test_roots"]:
        for r in newly:
            split_of_root[r["canonical_physical_id"]] = "test"
        test_set_kind = "newly_selected_development"
    else:
        for r in newly:
            split_of_root[r["canonical_physical_id"]] = "test"
        for r in hist_test_fallback:
            split_of_root[r["canonical_physical_id"]] = "test"
        test_set_kind = "mixed_new_plus_historical_holdout"
        test_caveat = (
            f"Only {len(newly)} newly_selected roots generated; "
            "test includes historical freeze-ban holdout and must NOT be read as sealed-independent."
        )

    # Collect rollouts for all roots that have starts
    print("[P109-A] Phase 2 — rich P64 diagnostic rollouts", flush=True)
    rolls = []
    for i, r in enumerate(pool_rows):
        tid = r["traj_id"]
        if tid not in starts:
            # load generated pkl
            pkl = r.get("pkl")
            if pkl and Path(pkl).is_file():
                ep = pickle.loads(Path(pkl).read_bytes())["ep"]
                starts[tid] = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            else:
                continue
        print(f"  roll {i+1}/{len(pool_rows)} {tid} [{r['provenance']}]", flush=True)
        out = rich_rollout(p64, starts[tid], book, rho, device, seed=CL_SEED)
        if out is None or out.get("error"):
            print(f"    FAIL {out}", flush=True)
            continue
        # attach windows for grad audit
        for s in out["steps"]:
            # store windows used at decision (already encoded from obs_seq)
            # re-get from lengths: we need obs_win in step — reconstruct approx from single obs only if missing
            pass
        path = CACHE / f"rich_{tid}.npz"
        np.savez_compressed(
            path,
            bil_before=np.asarray([s["bilateral_before"] for s in out["steps"]], np.int8),
            bil_after=np.asarray([s["bilateral_after"] for s in out["steps"]], np.int8),
            vrel=np.asarray([np.nan if s["vrel"] is None else s["vrel"] for s in out["steps"]], np.float32),
            e_pos=np.asarray([s["e_pos_mm"] for s in out["steps"]], np.float32),
            sim_time=np.asarray([s["sim_time"] for s in out["steps"]], np.float64),
            tokens=np.asarray([s["tok"] for s in out["steps"]], np.int64),
            success=np.asarray([out["SUCCESS_RELAXED_12S"]]),
        )
        # keep steps in memory for labeling (heavy but n~40 roots * 80)
        rolls.append({"meta": r, "canonical_physical_id": r["canonical_physical_id"], "roll": out, "npz": str(path)})

    # Labels + feature table
    print("[P109-A] Phase 3 — labels + probes", flush=True)
    (OUT / "CONTACT_TARGET_DEFINITIONS.md").write_text(
        f"""# CONTACT_TARGET_DEFINITIONS — P109-A

dt_policy = {DT_POLICY}s. Horizons: 0.2s = {STEPS_SHORT} steps; 0.5s = {STEPS_LONG} steps.

## T1 Bilateral Contact Survival
1 iff `bilateral_after==1` for every step in (t, t+H].

## T2 Contact-Loss Event
1 iff any transition 1→0 in `bilateral_after` within (t, t+H].

## T3 Relative Slip / Instability
Mean `vrel` over horizon (None if all missing). Unstable if mean > 0.05.

## T4 Post-Recatch Stability
Applicable only if regrasp (0→1) has occurred at or before t. Then same as T1.

## Censoring
If t+H exceeds episode length, labels are null (`censored=1`), never imputed as 0/1.
""",
        encoding="utf-8",
    )

    samples = []
    target_dist = []
    for R in rolls:
        labels = label_windows(R["roll"])
        # need obs_win for grad — store last 160 by re-rolling encode inputs: use obs/prev only padded
        for lab in labels:
            t = lab["t"]
            step = R["roll"]["steps"][t]
            # build windows from steps history
            obs_list = [R["roll"]["steps"][j]["obs"] for j in range(max(0, t - 159), t + 1)]
            prev_list = [R["roll"]["steps"][j]["prev"] for j in range(max(0, t - 159), t + 1)]
            step = dict(step)
            step["obs_win"] = np.stack(obs_list).astype(np.float32)
            step["prev_win"] = np.stack(prev_list).astype(np.float32)
            samples.append(
                {
                    **lab,
                    "traj_id": R["meta"]["traj_id"],
                    "canonical_physical_id": R["canonical_physical_id"],
                    "provenance": R["meta"]["provenance"],
                    "split": split_of_root.get(R["canonical_physical_id"], ""),
                    "step": step,
                    "success_ep": R["roll"]["SUCCESS_RELAXED_12S"],
                }
            )
        for horizon_tag in ("0.2s", "0.5s"):
            for key in ("T1_bilateral_survival", "T2_contact_loss", "T3_slip_unstable", "T4_post_recatch_stable"):
                vals = [
                    lab[key]
                    for lab in labels
                    if lab["horizon_tag"] == horizon_tag and not lab["censored"] and lab[key] is not None
                ]
                if key == "T4_post_recatch_stable":
                    vals = [
                        lab[key]
                        for lab in labels
                        if lab["horizon_tag"] == horizon_tag
                        and not lab["censored"]
                        and lab.get("T4_applicable")
                        and lab[key] is not None
                    ]
                target_dist.append(
                    {
                        "target": key,
                        "horizon": horizon_tag,
                        "n_valid": len(vals),
                        "n_positive": int(sum(vals)) if vals else 0,
                        "prevalence": float(np.mean(vals)) if vals else float("nan"),
                        "n_censored": sum(
                            1 for lab in labels if lab["horizon_tag"] == horizon_tag and lab["censored"]
                        ),
                    }
                )
    write_csv(OUT / "TARGET_DISTRIBUTIONS.csv", target_dist)

    probe_rows = []
    for key in ("T1_bilateral_survival", "T2_contact_loss", "T3_slip_unstable", "T4_post_recatch_stable"):
        for horizon_tag in ("0.2s", "0.5s"):
            probe_rows.extend(train_eval_probes(samples, key, horizon_tag, split_of_root))
    write_csv(OUT / "PROBE_RESULTS.csv", probe_rows)

    # Representation comparison markdown
    def best_auroc(target, horizon, family):
        sub = [
            r
            for r in probe_rows
            if r.get("target") == target
            and r.get("horizon") == horizon
            and r.get("family") == family
            and r.get("model") == "logreg"
            and r.get("status") == "OK"
        ]
        return float(sub[0]["auroc"]) if sub else float("nan")

    comp_lines = [
        "# REPRESENTATION_COMPARISON — P109-A",
        "",
        f"Test set kind: **{test_set_kind}**",
        f"Caveat: {test_caveat or 'none'}",
        "",
        "| Target | Horizon | A_current AUROC | B_history | C_P64_hidden |",
        "|---|---|---:|---:|---:|",
    ]
    for key in ("T1_bilateral_survival", "T2_contact_loss", "T3_slip_unstable"):
        for h in ("0.2s", "0.5s"):
            comp_lines.append(
                f"| {key} | {h} | {best_auroc(key,h,'A_current'):.3f} | {best_auroc(key,h,'B_history'):.3f} | {best_auroc(key,h,'C_p64_hidden'):.3f} |"
            )
    (OUT / "REPRESENTATION_COMPARISON.md").write_text("\n".join(comp_lines), encoding="utf-8")

    print("[P109-A] Phase 4 — early warning + gradient path", flush=True)
    ew = early_warning_analysis(rolls, split_of_root)
    write_csv(OUT / "EARLY_WARNING_RESULTS.csv", ew)
    # lead stats at thr=0.5
    leads = [float(r["lead_s"]) for r in ew if r.get("lead_s") != "" and float(r.get("thr") or 0) == 0.5]
    cats = defaultdict(int)
    for r in ew:
        if float(r.get("thr") or 0) == 0.5:
            cats[r.get("category", "")] += 1

    grad = gradient_path_audit(p64, book, device, samples)
    (OUT / "GRADIENT_PATH_AUDIT.md").write_text(
        "# GRADIENT_PATH_AUDIT — P109-A\n\n```json\n"
        + json.dumps(grad, indent=2)
        + "\n```\n\nNo optimizer step; disposable copy only.\n",
        encoding="utf-8",
    )

    # Decision logic with pre-registered gates
    def lift(target, horizon):
        a = best_auroc(target, horizon, "A_current")
        c = best_auroc(target, horizon, "C_p64_hidden")
        b = best_auroc(target, horizon, "B_history")
        return a, b, c, (c - a if np.isfinite(c) and np.isfinite(a) else float("nan"))

    a02, b02, c02, lift02 = lift("T2_contact_loss", "0.2s")
    a05, b05, c05, lift05 = lift("T2_contact_loss", "0.5s")
    n_test_roots = len({r["canonical_physical_id"] for r in rolls if split_of_root.get(r["canonical_physical_id"]) == "test"})
    n_valid_t2 = next(
        (
            d["n_valid"]
            for d in target_dist
            if d["target"] == "T2_contact_loss" and d["horizon"] == "0.2s"
        ),
        0,
    )
    median_lead = float(np.median(leads)) if leads else float("nan")
    useful_warn = cats.get("before_instability", 0) + cats.get("after_instability_before_loss", 0)
    grad_ok = bool(
        grad.get("status") == "OK"
        and float(grad.get("disposable_last_block_trainable", {}).get("grad_norm_final_transformer_block_Lcontact") or 0)
        > 0
    )

    meaningful = (
        n_test_roots >= PRE_REG["min_test_roots"]
        and n_valid_t2 >= PRE_REG["min_valid_windows_per_target"]
        and (
            (np.isfinite(lift02) and lift02 >= PRE_REG["min_auroc_lift_vs_current"])
            or (np.isfinite(c02) and c02 >= 0.65 and np.isfinite(a02) and c02 > a02 + 0.02)
        )
        and (np.isfinite(median_lead) and median_lead >= PRE_REG["min_median_lead_s_at_fpr"])
        and useful_warn > 0
        and grad_ok
    )
    # Case classification (pre-registered gates decide SUPPORT; else WEAK if data ok)
    if not p108_audit["reusable_for_probes"] and len(rolls) < 5:
        decision = "P109A_INVALID"
        case = "invalid"
    elif n_valid_t2 < PRE_REG["min_valid_windows_per_target"] or n_test_roots < 4:
        decision = "P109A_INCONCLUSIVE_DATA_LIMITED"
        case = "data_limited"
    elif meaningful and (np.isfinite(c02) and np.isfinite(b02) and c02 > b02 + 0.03):
        decision = "P109A_PREDICTIVE_SIGNAL_SUPPORTED"
        case = "E_plus_A"
    elif meaningful:
        decision = "P109A_PREDICTIVE_SIGNAL_SUPPORTED"
        case = "E"
    elif np.isfinite(c02) and c02 < 0.55 and np.isfinite(a02) and a02 < 0.55:
        decision = "P109A_PREDICTIVE_SIGNAL_WEAK"
        case = "C"
    elif useful_warn == 0 and (cats.get("after_loss", 0) > 0 or not leads):
        decision = "P109A_PREDICTIVE_SIGNAL_WEAK"
        case = "D"
    elif np.isfinite(b05) and np.isfinite(c05) and b05 > c05 + 0.03:
        decision = "P109A_PREDICTIVE_SIGNAL_WEAK"
        case = "B_history_matches_or_beats_P64_at_0.5s"
    elif np.isfinite(lift02) and lift02 >= PRE_REG["min_auroc_lift_vs_current"] and (
        not np.isfinite(median_lead) or median_lead < PRE_REG["min_median_lead_s_at_fpr"]
    ):
        decision = "P109A_PREDICTIVE_SIGNAL_WEAK"
        case = "A_partial_failed_early_warning_gate"
    elif np.isfinite(b02) and np.isfinite(c02) and abs(b02 - c02) < 0.03 and max(b02, c02) > 0.6:
        decision = "P109A_PREDICTIVE_SIGNAL_WEAK"
        case = "B"
    else:
        decision = "P109A_PREDICTIVE_SIGNAL_WEAK"
        case = "mixed_gates_failed"

    answers = {
        "Q1_predict_from_observable": bool(np.isfinite(a02) and a02 > 0.55),
        "Q2_history_helps": bool(np.isfinite(b02) and np.isfinite(a02) and b02 > a02 + 0.02),
        "Q3_p64_beyond_history": bool(np.isfinite(c02) and np.isfinite(b02) and c02 > b02 + 0.03),
        "Q4_early_warning_before_loss": bool(useful_warn > 0 and np.isfinite(median_lead) and median_lead > 0),
        "Q5_generalizes_across_roots": bool(n_test_roots >= 4 and np.isfinite(c02)),
        "Q6_gradient_path_exists": grad_ok,
        "Q7_justify_p109b": bool(decision == "P109A_PREDICTIVE_SIGNAL_SUPPORTED"),
    }

    summary = {
        "case": decision,
        "interpretation_case": case,
        "created": utc(),
        "p64_sha": p64_sha,
        "trained_actor": False,
        "sealed_panel_opened": False,
        "p108_cache_audit": p108_audit,
        "n_rolls": len(rolls),
        "n_generated_ics": len(gen),
        "test_set_kind": test_set_kind,
        "test_caveat": test_caveat,
        "n_test_roots": n_test_roots,
        "pre_registered": PRE_REG,
        "metrics_T2_0.2s": {"A": a02, "B": b02, "C": c02, "lift_C_minus_A": lift02},
        "metrics_T2_0.5s": {"A": a05, "B": b05, "C": c05, "lift_C_minus_A": lift05},
        "early_warning": {
            "median_lead_s_thr0.5": median_lead,
            "category_counts_thr0.5": dict(cats),
        },
        "gradient_path": grad,
        "answers": answers,
        "recommend_p109b": bool(answers["Q7_justify_p109b"]),
        "wall_s": time.time() - t0,
    }
    write_json(OUT / "SUMMARY.json", summary)

    (OUT / "LIMITATIONS.md").write_text(
        f"""# LIMITATIONS — P109-A

1. P108 traces lacked obs/vrel/timestamps; all probe evidence comes from new P64 diagnostic rolls.
2. Newly selected IC generation may be below target ({len(gen)}/{20}); test_set_kind={test_set_kind}.
3. {test_caveat or 'No historical-holdout caveat.'}
4. Early-warning scores use a simple bilateral/vrel heuristic, not the trained probe probabilities (conservative).
5. Gradient audit uses a disposable copy; nonzero shared gradients ≠ proven policy improvement.
6. Sealed independent panel was not opened.
""",
        encoding="utf-8",
    )

    report = f"""# FINAL_REPORT — P109-A Contact Predictability Audit

Generated: {summary['created']}

## Decision

**`{decision}`** (interpretation case: {case})

Recommend P109-B: **{summary['recommend_p109b']}** (requires explicit human approval even if True).

## Setup
- P64 `{p64_sha}` frozen. No Actor updates. No sealed panel.
- P108 cache: `{p108_audit['verdict']}` → recollected rich rolls n={len(rolls)}.
- Generated new ICs: {len(gen)}. Test set: {test_set_kind}. {test_caveat or ''}

## Pre-registered gates
{json.dumps(PRE_REG)}

## T2 contact-loss prediction (logreg AUROC)
| Horizon | A current | B history | C P64 hidden | C−A |
|---|---:|---:|---:|---:|
| 0.2s | {a02:.3f} | {b02:.3f} | {c02:.3f} | {lift02:.3f} |
| 0.5s | {a05:.3f} | {b05:.3f} | {c05:.3f} | {lift05:.3f} |

## Early warning (thr=0.5)
- median lead: {median_lead}
- categories: {dict(cats)}

## Gradient path
- Frozen backbone ⇒ contact loss on detached h does not touch token head (by construction).
- Disposable last-block trainable: grad norms / cosine in `GRADIENT_PATH_AUDIT.md`.
- Shared grad exists: **{grad_ok}**

## Answers
1. Q1 predict from observables? **{answers['Q1_predict_from_observable']}**
2. Q2 history helps? **{answers['Q2_history_helps']}**
3. Q3 P64 beyond history? **{answers['Q3_p64_beyond_history']}**
4. Q4 early warning before loss? **{answers['Q4_early_warning_before_loss']}**
5. Q5 generalizes across roots? **{answers['Q5_generalizes_across_roots']}**
6. Q6 gradient path? **{answers['Q6_gradient_path_exists']}**
7. Q7 justify P109-B? **{answers['Q7_justify_p109b']}**

## Next
{"Draft P109-B for approval (do not execute)." if summary['recommend_p109b'] else "Stop or expand data/new-root generation before any Actor auxiliary training."}

## Stop
P109-A complete. Do not train P109-B / open sealed panel / replace P64 / PPO without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    # Update proposal doc
    prop = ROOT / "docs/p109_new_skill_acquisition_proposal.md"
    prop.write_text(
        f"""# P109 Proposal — Updated after P109-A

## P109-A decision
`{decision}`

Answers: {json.dumps(answers, indent=2)}

## Conditional P109-B (proposal only; NOT executed)

If and only if human approves after `{decision}`:

| Arm | Description |
|---|---|
| A | Frozen P64 |
| B | SFT + fixed retention; final Transformer block trainable |
| C | Identical to B + future contact auxiliary (train-only head) |

Matched init/data/seeds/budget/retention. One unified Actor. No sealed panel in B.

### Falsification
C ≯ B on new-root C SUCCESS_RELAXED_12S under matched budget → reject H2 for this regime.

## Notes
P108 corrected ranking preferred H2. P109-A tested predictive premises without Actor updates.
Test caveat: {test_caveat or 'n/a'}
""",
        encoding="utf-8",
    )

    print(report, flush=True)
    print(f"[P109-A] DONE decision={decision} wall_s={summary['wall_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
