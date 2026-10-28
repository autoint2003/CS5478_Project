"""P45 Phase 2 — Targeted data aggregation.

P45-A: upsample early-horizon chunks from P44 expert cuts (no new sim).
P45-B: on train ICs only, policy rollout → at early divergence/pre-drop,
       restore capture and try expert-suffix continuation; keep if SUCCESS_12S.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paths import (  # noqa: E402
    OUT,
    P44,
    P44_CKPT,
    P44_DATA,
    P44_EVAL,
    SEED,
    T_MAX,
)
from models.m111_checkpoint import HANDOFF, MOTION_K, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402

EARLY_HORIZON = 80
EARLY_WEIGHT = 3.0
SKILLS = ("A1", "B1", "C", "D")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def holdout_ids() -> set[tuple[str, str]]:
    rows = list(csv.DictReader(P44_EVAL.open(encoding="utf-8")))
    return {(r["skill"], r["traj_id"]) for r in rows if r.get("tag") == "holdout"}


def load_p44_chunks() -> dict[str, list]:
    out = {}
    for sk in SKILLS:
        p = P44_DATA / f"chunks_{sk}.pkl"
        out[sk] = pickle.loads(p.read_bytes()) if p.exists() else []
    return out


def build_p45a(base: dict[str, list]) -> dict[str, list]:
    """Upsample chunks that cover early absolute steps via chunk_index==0 and boost weight."""
    out = {s: [] for s in SKILLS}
    stats = Counter()
    for sk, chs in base.items():
        for ch in chs:
            c = dict(ch)
            c["sample_weight"] = float(c.get("sample_weight") or 1.0)
            c["source"] = "p44_expert"
            out[sk].append(c)
            # early chunk: first window often starts at 0
            if int(c.get("chunk_index") or 0) == 0:
                boost = dict(c)
                boost["sample_weight"] = float(EARLY_WEIGHT)
                boost["source"] = "p45a_early_upsample"
                out[sk].append(boost)
                stats[sk] += 1
    return out, stats


def policy_tok(model, book, rho, obs_seq, prev_seq, device):
    tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, WINDOW)
    if action is None or tok == HANDOFF:
        with torch.no_grad():
            lg, resid, _ = model(
                torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device),
                torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device),
            )
            lg = lg[-1].clone()
            lg[HANDOFF] = -1e9
            tok = int(torch.argmax(lg).item())
            rr = resid[-1].float().cpu().numpy().astype(np.float32)
            action = decode_m111_action(book, tok, rr, rho=rho)
    return int(tok), np.asarray(action, np.float32)


def attempt_suffix(
    env,
    job,
    book,
    rho,
    *,
    cut_step: int,
    policy_obs: list,
    policy_prev: list,
    t_max: int = T_MAX,
) -> dict | None:
    """From current sim state (already at cut_step after policy), run expert tokens[cut:]."""
    toks = np.asarray(job["tokens"], np.int64)
    res = np.asarray(job["residuals"], np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    if cut_step >= n:
        return None
    if latched(env.sim):
        return None

    # Build episode: policy prefix obs/prev already collected; expert actions from cut
    obs_all = list(policy_obs)
    prev_all = list(policy_prev)
    tok_all = []
    res_all = []
    # We need tokens for prefix too — use whatever policy took; stored separately by caller
    return {
        "n": n,
        "cut": cut_step,
    }


def collect_p45b(
    *,
    max_train_per_skill: int = 24,
    max_attempts: int = 80,
) -> tuple[dict[str, list], list[dict]]:
    held = holdout_ids()
    jobs = [j for j in collect_jobs() if (j["skill"], j["traj_id"]) not in held]
    # prefer train split
    train_jobs = [j for j in jobs if j.get("split") == "train"] or jobs
    by_sk: dict[str, list] = {s: [] for s in SKILLS}
    for j in train_jobs:
        by_sk[j["skill"]].append(j)
    selected = []
    for s in SKILLS:
        selected.extend(by_sk[s][:max_train_per_skill])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(P44_CKPT, device)
    model.eval()
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    chunks: dict[str, list] = {s: [] for s in SKILLS}
    meta_rows = []
    attempts = 0
    accepted = 0

    print(f"P45-B collect train_jobs={len(selected)} holdout_blocked={len(held)}", flush=True)
    for job in selected:
        if attempts >= max_attempts:
            break
        skill = job["skill"]
        if skill in ("A1", "B1", "C"):
            ep = job["ep"]
            ref = ep["ref"]

            def restore(ep=ep):
                restore_start(env, ep)

        else:
            cap0, ref = job["capture"], job["ref"]

            def restore(cap0=cap0):
                restore_capture(env, cap0)

        toks = np.asarray(job["tokens"], np.int64)
        res = np.asarray(job["residuals"], np.float32)
        n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)

        restore()
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        obs_seq, prev_seq = [], []
        pol_toks, pol_res = [], []
        cut = None
        reason_cut = None

        for t in range(min(n, EARLY_HORIZON + 40)):
            if latched(env.sim):
                break
            pb = pad_bilateral(env.sim)
            obs = np.asarray(
                observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
                np.float32,
            )
            obs_seq.append(obs.copy())
            prev_seq.append(env._prev.copy())
            pred, action = policy_tok(model, book, rho, obs_seq, prev_seq, device)
            exp = int(toks[t]) if t < n else -1
            # trigger collection: first token mismatch OR bilateral about to fail (asym/missing)
            trigger = False
            if exp >= 0 and exp != HANDOFF and pred != exp and t <= EARLY_HORIZON:
                trigger = True
                reason_cut = "token_mismatch"
            if not pb["bilateral"] and t > 0:
                trigger = True
                reason_cut = "bilateral_lost"
            if trigger and cut is None and t >= 1:
                # back up one step: use capture BEFORE this action — we already need prior state
                # Instead: cut at t (current state before action), expert takes over from t
                cut = t
                cap = take_capture(env)
                break
            # execute policy action
            rr = np.zeros(7, np.float32)  # residual from decode already in action
            # store residual approx zeros for packaging; token is what matters for CE
            pol_toks.append(pred)
            pol_res.append(rr)
            step_legal(env.sim, action)
            env._prev = action.reshape(7)

        if cut is None or cut <= 0:
            continue
        attempts += 1
        # obs_seq includes current frame at cut; pol_toks are actions already executed (len==cut)
        if len(pol_toks) != cut or len(obs_seq) < cut + 1:
            meta_rows.append(
                {
                    "traj_id": job["traj_id"],
                    "skill": skill,
                    "cut_step": cut,
                    "accepted": 0,
                    "fail": "prefix_len_mismatch",
                }
            )
            continue

        # Expert suffix from capture
        restore_capture(env, cap)
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = prev_seq[cut].copy()
        mon = R3FOnline.create(t_max=T_MAX)
        obs_suf, prev_suf, tok_suf, res_suf = [], [], [], []
        ok_hand = False
        for t in range(cut, n + 1):
            if latched(env.sim):
                break
            st = mon.observe(env.sim, t - cut, tape_ended=(t >= n))
            if st["ready"]:
                ok_hand = True
                break
            if t >= n:
                break
            exp = int(toks[t])
            if exp == HANDOFF:
                continue
            rr = res[t] if t < len(res) else np.zeros(7, np.float32)
            obs = np.asarray(
                observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
                np.float32,
            )
            obs_suf.append(obs)
            prev_suf.append(env._prev.copy())
            tok_suf.append(exp)
            res_suf.append(rr)
            action = decode_m111_action(book, exp, rr, rho=rho)
            step_legal(env.sim, action)
            env._prev = np.asarray(action, np.float32).reshape(7)

        row = {
            "traj_id": job["traj_id"],
            "skill": skill,
            "cut_step": cut,
            "reason_cut": reason_cut,
            "suffix_len": len(tok_suf),
            "handoff_ready": int(ok_hand),
            "SUCCESS_12S": 0,
            "SUCCESS_NEW": 0,
            "accepted": 0,
        }
        if not ok_hand or not tok_suf:
            meta_rows.append(row)
            print(f"  reject {skill} {job['traj_id'][:36]} cut={cut} no_handoff", flush=True)
            continue

        hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
        s12 = int(bool(hand.get("SUCCESS_12S")))
        snew = int(bool(hand.get("SUCCESS_NEW")))
        row.update({"SUCCESS_12S": s12, "SUCCESS_NEW": snew, "eval_reason": hand.get("reason")})
        if not s12:
            meta_rows.append(row)
            print(f"  reject {skill} {job['traj_id'][:36]} cut={cut} 12S=0", flush=True)
            continue

        # Supervise verified expert suffix from the policy-visited cut state
        ep = {
            "observations": np.asarray(obs_suf, np.float32),
            "prev_actions": np.asarray(prev_suf, np.float32),
            "tokens": np.asarray(tok_suf, np.int64),
            "residuals": np.asarray(res_suf, np.float32),
            "loss_mask": np.ones(len(tok_suf), dtype=np.bool_),
        }
        ep["loss_mask"] &= (ep["tokens"] != HANDOFF) & (ep["tokens"] >= 0) & (ep["tokens"] < MOTION_K)
        if not bool(ep["loss_mask"].any()):
            meta_rows.append({**row, "accepted": 0})
            continue

        for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
            mask = np.asarray(ch["loss_mask"], dtype=np.bool_)
            mask &= (np.asarray(ch["tokens"]) != HANDOFF)
            if not bool(mask.any()):
                continue
            chunks[skill].append(
                {
                    **ch,
                    "loss_mask": mask,
                    "traj_id": job["traj_id"],
                    "skill": skill,
                    "split": "train",
                    "chunk_index": ci,
                    "n_tokens": int(mask.sum()),
                    "sample_weight": 2.5,
                    "source": "p45b_expert_suffix",
                    "cut_step": cut,
                    "handoff_reason": reason_cut,
                    "SUCCESS_NEW": snew,
                }
            )
        accepted += 1
        row["accepted"] = 1
        meta_rows.append(row)
        write_csv(OUT / "P45B_ATTEMPTS.csv", meta_rows)
        print(
            f"  ACCEPT {skill} {job['traj_id'][:36]} cut={cut} suf={len(tok_suf)} NEW={snew}",
            flush=True,
        )

    return chunks, meta_rows


def run(*, max_train_per_skill: int = 24, max_attempts: int = 80) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base = load_p44_chunks()
    p45a, early_stats = build_p45a(base)
    for sk, chs in p45a.items():
        (OUT / "dataset" / f"chunks_{sk}_P45A.pkl").write_bytes(pickle.dumps(chs))
    print("P45-A early upsamples:", dict(early_stats), {s: len(p45a[s]) for s in SKILLS}, flush=True)

    p45b_chunks, meta = collect_p45b(
        max_train_per_skill=max_train_per_skill, max_attempts=max_attempts
    )
    # Mix A = p45a only; B = p45a + p45b
    mixed = {s: list(p45a[s]) + list(p45b_chunks[s]) for s in SKILLS}
    for sk, chs in p45b_chunks.items():
        (OUT / "dataset" / f"chunks_{sk}_P45B_only.pkl").write_bytes(pickle.dumps(chs))
    for sk, chs in mixed.items():
        (OUT / "dataset" / f"chunks_{sk}_P45B.pkl").write_bytes(pickle.dumps(chs))

    report = {
        "utc": utc(),
        "p45a_sizes": {s: len(p45a[s]) for s in SKILLS},
        "p45a_early_extra": dict(early_stats),
        "p45b_attempts": len(meta),
        "p45b_accepted": sum(int(r.get("accepted") or 0) for r in meta),
        "p45b_chunk_sizes": {s: len(p45b_chunks[s]) for s in SKILLS},
        "p45b_mixed_sizes": {s: len(mixed[s]) for s in SKILLS},
        "holdout_excluded": len(holdout_ids()),
        "early_horizon": EARLY_HORIZON,
        "early_weight": EARLY_WEIGHT,
    }
    (OUT / "DATA_AGGREGATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# DATA_AGGREGATION_REPORT",
        "",
        f"Generated: {utc()}",
        "",
        "## P45-A (early expert upsample)",
        "",
        f"Early chunk_index==0 cloned with weight={EARLY_WEIGHT}.",
        f"Sizes: `{report['p45a_sizes']}` extras=`{report['p45a_early_extra']}`",
        "",
        "## P45-B (policy-state → expert suffix, train ICs only)",
        "",
        f"Attempts={report['p45b_attempts']} accepted={report['p45b_accepted']} "
        f"holdout_excluded={report['holdout_excluded']}",
        f"B-only chunks: `{report['p45b_chunk_sizes']}`",
        f"Mixed A+B sizes: `{report['p45b_mixed_sizes']}`",
        "",
        "Only SUCCESS_12S continuations admitted. Unsupported/failed attempts logged in P45B_ATTEMPTS.csv.",
        "",
    ]
    (OUT / "DATA_AGGREGATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-per-skill", type=int, default=24)
    ap.add_argument("--max-attempts", type=int, default=80)
    args = ap.parse_args()
    run(max_train_per_skill=args.max_train_per_skill, max_attempts=args.max_attempts)


if __name__ == "__main__":
    main()
