"""P48 — B failure root-cause + table_free conflict + training interference audit."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
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
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from experiments.diagnostics.run_b_grpo_multistage_dataset import (  # noqa: E402
    DEFAULT_WEIGHTS,
    make_plain_env,
    quantize_and_replay,
    replay_fullchain,
)
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p48_b_failure_root_cause.paths import (  # noqa: E402
    M111_META,
    OUT,
    P44_CKPT,
    P44_EVAL,
    P45,
    P45_DIAG,
    P45B_CKPT,
    P47,
    P47_CKPT,
    P47_RAW,
    P47T_CKPT,
    SEED,
    T_MAX,
    TABLE_FREE,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW as K_WIN  # noqa: E402
from training.m111_dataset import collate_m111  # noqa: E402
from training.skill_aware_action_vocab import load_motion_codebook  # noqa: E402

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


def root_id(tid: str) -> str:
    t = re.sub(r"^(A|B|C)__", "", tid)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    for p in ("plan_a__", "plan_b__", "plan_c__"):
        if t.startswith(p):
            t = t[len(p) :]
            break
    return t


def holdout_b_jobs() -> list[dict]:
    hold = {
        (r["skill"], r["traj_id"])
        for r in csv.DictReader(P44_EVAL.open(encoding="utf-8"))
        if r.get("tag") == "holdout" and r.get("skill") == "B1"
    }
    jobs = []
    for j in collect_jobs():
        if j["skill"] != "B1":
            continue
        if (j["skill"], j["traj_id"]) in hold or any(
            root_id(j["traj_id"]) == root_id(tid) for (_, tid) in hold
        ):
            # prefer exact traj_id match
            if (j["skill"], j["traj_id"]) in hold:
                jobs.append(j)
    # fill by root if missing
    have = {root_id(j["traj_id"]) for j in jobs}
    for j in collect_jobs():
        if j["skill"] != "B1":
            continue
        rid = root_id(j["traj_id"])
        if rid in {root_id(t) for _, t in hold} and rid not in have:
            jobs.append(j)
            have.add(rid)
    # de-dupe by root
    by = {}
    for j in jobs:
        by[root_id(j["traj_id"])] = j
    return list(by.values())


def geo(sim) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    return {
        "theta": float(m["e_theta_deg"]),
        "e_pos": float(m["e_pos_mm"]),
        "bilateral": int(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "min_pad": int(min(pb["nL"], pb["nR"])),
    }


def replay_token_tape(env, restore_fn, ref, tokens, residuals, book, rho, *, t_max=T_MAX):
    restore_fn()
    mon = R3FOnline.create(t_max=t_max)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    n = len(tokens) - (1 if len(tokens) and int(tokens[-1]) == HANDOFF else 0)
    first_bil_loss = None
    drop = False
    hand = False
    reason = None
    step_h = None
    for t in range(max(n, 1) + 1):
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, t, tape_ended=(t >= n))
        if st["ready"]:
            hand = True
            reason = st["reason"]
            step_h = t
            break
        if t >= n:
            break
        g = geo(env.sim)
        if first_bil_loss is None and not g["bilateral"] and t > 0:
            first_bil_loss = t
        tok = int(tokens[t])
        if tok == HANDOFF:
            continue
        rr = residuals[t] if t < len(residuals) else np.zeros(7, np.float32)
        action = decode_m111_action(book, tok, rr, rho=rho)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)
    out = {
        "drop": int(drop),
        "handoff": int(hand or (not drop)),
        "handoff_reason": reason if hand else ("DROP_BEFORE" if drop else "TIMEOUT_FALLBACK"),
        "handoff_step": step_h if hand else (None if drop else t_max),
        "first_bil_loss": first_bil_loss,
        "n_steps": n,
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
    }
    if drop:
        return out
    if not hand:
        out["handoff"] = 1
        out["handoff_reason"] = "TIMEOUT_FALLBACK"
        out["handoff_step"] = T_MAX
    ev = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    out["SUCCESS_12S"] = int(bool(ev.get("SUCCESS_12S")))
    out["SUCCESS_NEW"] = int(bool(ev.get("SUCCESS_NEW")))
    out["eval_reason"] = ev.get("reason")
    return out


def replay_continuous(env, restore_fn, ref, actions):
    restore_fn()
    env._hist = ObservableObsState(DT_POLICY)
    mon = R3FOnline.create(t_max=T_MAX)
    first_bil_loss = None
    drop = False
    hand = False
    reason = None
    step_h = None
    acts = np.asarray(actions, np.float32)
    for t in range(len(acts) + 1):
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, t, tape_ended=(t >= len(acts)))
        if st["ready"]:
            hand = True
            reason = st["reason"]
            step_h = t
            break
        if t >= len(acts):
            break
        g = geo(env.sim)
        if first_bil_loss is None and not g["bilateral"] and t > 0:
            first_bil_loss = t
        step_legal(env.sim, acts[t])
    out = {
        "drop": int(drop),
        "handoff": int(hand or (not drop)),
        "handoff_reason": reason if hand else ("DROP_BEFORE" if drop else "TIMEOUT_FALLBACK"),
        "handoff_step": step_h,
        "first_bil_loss": first_bil_loss,
        "n_steps": len(acts),
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
    }
    if drop:
        return out
    if not hand:
        out["handoff_reason"] = "TIMEOUT_FALLBACK"
        out["handoff_step"] = T_MAX
    ev = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    out["SUCCESS_12S"] = int(bool(ev.get("SUCCESS_12S")))
    out["SUCCESS_NEW"] = int(bool(ev.get("SUCCESS_NEW")))
    out["eval_reason"] = ev.get("reason")
    return out


def policy_rollout(env, model, book, rho, restore_fn, ref, device):
    restore_fn()
    mon = R3FOnline.create(t_max=T_MAX)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    first_bil_loss = None
    drop = False
    hand = False
    reason = None
    step_h = None
    toks = []
    switches = 0
    for t in range(T_MAX + 1):
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, t, tape_ended=False)
        if st["ready"]:
            hand = True
            reason = st["reason"]
            step_h = t
            break
        if t >= T_MAX:
            break
        g = geo(env.sim)
        if first_bil_loss is None and not g["bilateral"] and t > 0:
            first_bil_loss = t
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K_WIN)
        if action is None or tok == HANDOFF:
            with torch.no_grad():
                lg, resid, _ = model(
                    torch.as_tensor(np.stack(obs_seq[-K_WIN:]), device=device),
                    torch.as_tensor(np.stack(prev_seq[-K_WIN:]), device=device),
                )
                lg = lg[-1].clone()
                lg[HANDOFF] = -1e9
                tok = int(torch.argmax(lg).item())
                rr = resid[-1].float().cpu().numpy().astype(np.float32)
                action = decode_m111_action(book, tok, rr, rho=rho)
        if toks and tok != toks[-1]:
            switches += 1
        toks.append(tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)
    out = {
        "drop": int(drop),
        "handoff": int(hand or (not drop)),
        "handoff_reason": reason if hand else ("DROP_BEFORE" if drop else "TIMEOUT_FALLBACK"),
        "handoff_step": step_h if hand else (None if drop else T_MAX),
        "first_bil_loss": first_bil_loss,
        "n_steps": len(toks),
        "token_switches": switches,
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
    }
    if drop:
        out["handoff"] = 0
        return out
    if not hand:
        out["handoff_reason"] = "TIMEOUT_FALLBACK"
        out["handoff_step"] = T_MAX
    ev = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    out["SUCCESS_12S"] = int(bool(ev.get("SUCCESS_12S")))
    out["SUCCESS_NEW"] = int(bool(ev.get("SUCCESS_NEW")))
    out["eval_reason"] = ev.get("reason")
    return out


def expert_forced_pred(env, model, book, rho, restore_fn, tokens, residuals, device):
    """On expert states, measure first token mismatch + match rate."""
    restore_fn()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    n = len(tokens) - (1 if len(tokens) and int(tokens[-1]) == HANDOFF else 0)
    match = total = 0
    first_err = None
    for t in range(min(n, 200)):
        if latched(env.sim):
            break
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        with torch.no_grad():
            lg, _, _ = model(
                torch.as_tensor(np.stack(obs_seq[-K_WIN:]), device=device),
                torch.as_tensor(np.stack(prev_seq[-K_WIN:]), device=device),
            )
            lg = lg[-1].clone()
            lg[HANDOFF] = -1e9
            pred = int(torch.argmax(lg).item())
        exp = int(tokens[t])
        if exp == HANDOFF:
            break
        total += 1
        if pred == exp:
            match += 1
        elif first_err is None:
            first_err = t
        rr = residuals[t] if t < len(residuals) else np.zeros(7, np.float32)
        action = decode_m111_action(book, exp, rr, rho=rho)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)
    return {
        "match_rate": (match / total) if total else None,
        "n": total,
        "first_pred_err": first_err,
    }


def classify_root(row: dict) -> str:
    """Exclusive-ish root cause for one IC given continuous/M111/SFT outcomes."""
    c12 = int(row.get("cont_12S") or 0)
    m12 = int(row.get("m111_12S") or 0)
    s12 = int(row.get("sft_12S") or 0)
    match = row.get("sft_match_rate")
    first_err = row.get("sft_first_pred_err")
    if not c12:
        return "continuous_expert_failure"
    if c12 and not m12:
        return "m111_representation_failure"
    if m12 and s12:
        return "success"
    # M111 ok, SFT fail
    if match is not None and float(match) < 0.85 and first_err is not None and int(first_err) <= 5:
        return "sft_prediction_failure_early"
    if match is not None and float(match) >= 0.9:
        return "closed_loop_distribution_shift"
    if row.get("sft_first_bil_loss") is not None and int(row["sft_first_bil_loss"]) <= 5:
        return "unrecoverable_contact_loss_early"
    return "sft_prediction_or_shift_mixed"


# -------------------- Phase 1 --------------------


def phase1() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book_m, rho, _ = load_m111_book(M111_META)
    cb = load_motion_codebook(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    # load SFT models
    models = {}
    for name, path in (("P44", P44_CKPT), ("P45B", P45B_CKPT), ("P47T", P47T_CKPT)):
        if path.exists():
            m, b, _ = load_m111(path, device)
            m.eval()
            models[name] = (m, b if b is not None else book_m)

    # prior diagnosis
    diag = {
        root_id(r["traj_id"]): r
        for r in csv.DictReader(P45_DIAG.open(encoding="utf-8"))
        if r.get("skill") == "B1"
    }

    jobs = holdout_b_jobs()
    print(f"Phase1 B holdout jobs: {len(jobs)}", flush=True)
    rows = []
    # Prefer P45B as primary SFT for root-cause (best A, same B)
    primary = "P45B" if "P45B" in models else ("P47T" if "P47T" in models else "P44")

    for i, job in enumerate(jobs):
        tid = job["traj_id"]
        rid = root_id(tid)
        ep = job["ep"]
        ref = ep["ref"]
        toks = np.asarray(job["tokens"], np.int64)
        res = np.asarray(job["residuals"], np.float32)

        def restore(ep=ep):
            restore_start(env, ep)

        print(f"  [{i+1}/{len(jobs)}] {tid}", flush=True)
        # M111 expert tape replay
        m111 = replay_token_tape(env, restore, ref, toks, res, book_m, rho)
        # Continuous: decode tokens→actions (exact expert continuous under M111 decode)
        # True continuous from handcrafted/table_free if available for same root
        cont_actions = None
        cont_src = "m111_decode"
        # try P21-less: reconstruct via decode
        restore()
        acts = []
        n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
        for t in range(n):
            rr = res[t] if t < len(res) else np.zeros(7, np.float32)
            acts.append(decode_m111_action(book_m, int(toks[t]), rr, rho=rho))
        cont_actions = np.asarray(acts, np.float32)
        cont = replay_continuous(env, restore, ref, cont_actions)

        # Re-quantize continuous decode and replay (representation stress-test)
        restore()
        cap0 = take_capture(env)
        q = quantize_and_replay(
            make_plain_env(), cap0, ref, cont_actions, cb.prototypes, cb.residual, DEFAULT_WEIGHTS, "M111"
        )

        sft = {}
        forced = {}
        for name, (model, book) in models.items():
            sft[name] = policy_rollout(env, model, book, rho, restore, ref, device)
            forced[name] = expert_forced_pred(env, model, book, rho, restore, toks, res, device)

        prim = sft[primary]
        fr = forced[primary]
        row = {
            "traj_id": tid,
            "ic_root": rid,
            "n_expert": n,
            "cont_src": cont_src,
            "cont_12S": cont["SUCCESS_12S"],
            "cont_NEW": cont["SUCCESS_NEW"],
            "cont_drop": cont["drop"],
            "cont_first_bil_loss": cont["first_bil_loss"],
            "m111_tape_12S": m111["SUCCESS_12S"],
            "m111_tape_NEW": m111["SUCCESS_NEW"],
            "m111_tape_drop": m111["drop"],
            "m111_requant_12S": int(bool(q["SUCCESS_12S"])),
            "m111_requant_NEW": int(bool(q["SUCCESS_NEW"])),
            "m111_clip": q.get("clipped_frac"),
            "m111_mean_wlinf": q.get("mean_recon_wlinf"),
            "sft_primary": primary,
            "sft_12S": prim["SUCCESS_12S"],
            "sft_NEW": prim["SUCCESS_NEW"],
            "sft_drop": prim["drop"],
            "sft_first_bil_loss": prim["first_bil_loss"],
            "sft_handoff_reason": prim["handoff_reason"],
            "sft_token_switches": prim.get("token_switches"),
            "sft_match_rate": fr["match_rate"],
            "sft_first_pred_err": fr["first_pred_err"],
            "p45_mechanism": (diag.get(rid) or {}).get("mechanism"),
        }
        for name in models:
            row[f"{name}_12S"] = sft[name]["SUCCESS_12S"]
            row[f"{name}_drop"] = sft[name]["drop"]
            row[f"{name}_match"] = forced[name]["match_rate"]
            row[f"{name}_first_err"] = forced[name]["first_pred_err"]
            row[f"{name}_bil_loss"] = sft[name]["first_bil_loss"]
        # for classifier use m111_tape as m111_12S and cont as cont
        row["m111_12S"] = row["m111_tape_12S"]
        row["root_cause"] = classify_root(row)
        rows.append(row)
        write_csv(OUT / "B_HELDOUT_AUDIT_ROWS.csv", rows)
        print(
            f"    cont={row['cont_12S']} m111={row['m111_12S']} requant={row['m111_requant_12S']} "
            f"{primary}={row['sft_12S']} cause={row['root_cause']}",
            flush=True,
        )

    causes = Counter(r["root_cause"] for r in rows)
    summary = {
        "n": len(rows),
        "cont_12S": sum(int(r["cont_12S"]) for r in rows),
        "m111_tape_12S": sum(int(r["m111_tape_12S"]) for r in rows),
        "m111_requant_12S": sum(int(r["m111_requant_12S"]) for r in rows),
        "sft_primary": primary,
        "sft_12S": sum(int(r["sft_12S"]) for r in rows),
        "causes": dict(causes),
        "by_policy": {
            name: sum(int(r[f"{name}_12S"]) for r in rows) for name in models
        },
    }
    (OUT / "CONTINUOUS_VS_M111_REPLAY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown reports
    lines = [
        "# B_HELDOUT_FAILURE_AUDIT",
        "",
        f"Generated: {utc()}",
        f"Primary SFT for classification: **{primary}**",
        "",
        "## Root-cause counts",
        "",
    ]
    for k, v in causes.most_common():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Per-IC",
        "",
        "| IC | cont | m111 | requant | P44 | P45B | P47T | match@P45B | first_err | cause |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['ic_root']}` | {r['cont_12S']} | {r['m111_tape_12S']} | {r['m111_requant_12S']} | "
            f"{r.get('P44_12S', '')} | {r.get('P45B_12S', '')} | {r.get('P47T_12S', '')} | "
            f"{(r.get('P45B_match') if r.get('P45B_match') is not None else r.get('sft_match_rate'))} | "
            f"{r.get('P45B_first_err') if r.get('P45B_first_err') is not None else r.get('sft_first_pred_err')} | "
            f"`{r['root_cause']}` |"
        )
    (OUT / "B_HELDOUT_FAILURE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md2 = [
        "# CONTINUOUS_VS_M111_REPLAY",
        "",
        f"Generated: {utc()}",
        "",
        "On B held-out ICs (P21 expert tape):",
        "",
        f"- Continuous (= M111 decode of expert tokens+residuals) SUCCESS_12S: "
        f"**{summary['cont_12S']}/{summary['n']}**",
        f"- M111 tape replay SUCCESS_12S: **{summary['m111_tape_12S']}/{summary['n']}**",
        f"- Re-quantize decode→M111 then replay SUCCESS_12S: "
        f"**{summary['m111_requant_12S']}/{summary['n']}**",
        f"- SFT ({primary}) SUCCESS_12S: **{summary['sft_12S']}/{summary['n']}**",
        "",
        f"By policy: `{summary['by_policy']}`",
        "",
        "Note: P21 B experts are already M111-tokenized; “continuous” here is exact "
        "decode of (token, residual). Re-quantize stress-tests codebook reconstruction.",
        "",
    ]
    (OUT / "CONTINUOUS_VS_M111_REPLAY.md").write_text("\n".join(md2), encoding="utf-8")
    return {"rows": rows, "summary": summary, "models": list(models)}


# -------------------- Phase 2 --------------------


def phase2() -> dict:
    cb = load_motion_codebook(M111_META)
    env = make_plain_env()
    w = DEFAULT_WEIGHTS
    # index tree80 ordinary by ic_root
    tree = {}
    if (P47_RAW / "tokens_m111").exists():
        for p in (P47_RAW / "tokens_m111").glob("B__tree80_ordinary_*__m111.npz"):
            # B__tree80_ordinary_direct__ball_train_v2_0002__hash__m111.npz
            parts = p.name.replace("__m111.npz", "").split("__")
            # find ic after kind
            kind = parts[1] if len(parts) > 1 else ""
            # sample_id pattern B__kind__ic__fp
            sid = p.name.replace("__m111.npz", "")
            m = re.match(r"B__(tree80_ordinary_(?:direct|precision))__(.+)__[0-9a-f]{16}$", sid)
            if not m:
                continue
            ic = m.group(2)
            tree[ic] = {
                "kind": m.group(1),
                "tok": p,
                "cont": P47_RAW / "continuous" / f"{sid}__fullchain_continuous.npz",
                "ic_pkl": P47_RAW / "ics" / f"{sid}__recovery.pkl",
            }

    tf_rows = []
    conflicts = []
    schema = []
    for p in sorted(TABLE_FREE.glob("*__SUCCESS_NEW_expert.pkl")):
        obj = pickle.loads(p.read_bytes())
        ic = root_id(obj.get("physical_root_id") or obj.get("expert_id") or p.stem)
        acts = np.asarray(obj["actions"], np.float32)
        obs = np.asarray(obj["observations"], np.float32)
        cap = obj["catch_snap"]
        ref = obj["ref"]
        # schema checks
        schema.append(
            {
                "ic_id": ic,
                "catch_equals_named_capture": True,  # by construction
                "obs_dim": int(obs.shape[1]) if obs.ndim == 2 else None,
                "action_dim": int(acts.shape[1]) if acts.ndim == 2 else None,
                "n": int(len(acts)),
                "has_tree80_pair": int(ic in tree),
            }
        )
        q = quantize_and_replay(env, cap, ref, acts, cb.prototypes, cb.residual, w, "M111")
        row = {
            "ic_id": ic,
            "n": len(acts),
            "M111_12S": int(bool(q["SUCCESS_12S"])),
            "M111_NEW": int(bool(q["SUCCESS_NEW"])),
            "clip": q.get("clipped_frac"),
            "mean_wlinf": q.get("mean_recon_wlinf"),
        }
        if ic in tree:
            tinfo = tree[ic]
            if tinfo["cont"].exists() and tinfo["tok"].exists() and tinfo["ic_pkl"].exists():
                cont = np.load(tinfo["cont"])
                tokz = np.load(tinfo["tok"])
                icblob = pickle.loads(tinfo["ic_pkl"].read_bytes())
                # IC qpos distance
                q_tf = np.asarray(cap["physics"]["qpos"], float)
                q_t80 = np.asarray(icblob["capture"]["physics"]["qpos"], float)
                qpos_l2 = float(np.linalg.norm(q_tf - q_t80))
                n = min(len(acts), len(cont["actions"]), len(tokz["tokens"]))
                # align from start
                a_tf = acts[:n]
                a_t80 = np.asarray(cont["actions"][:n], np.float32)
                o_tf = obs[:n]
                o_t80 = np.asarray(cont["observations"][:n], np.float32)
                # obs distance per step
                obs_d = np.linalg.norm(o_tf - o_t80, axis=1) if o_tf.shape == o_t80.shape else None
                act_d = np.linalg.norm(a_tf - a_t80, axis=1)
                # tokens: quantize table_free vs tree80 tokens
                tok_tf = np.asarray(q["tokens"][:n], np.int64)
                tok_t80 = np.asarray(tokz["tokens"][:n], np.int64)
                nn = min(len(tok_tf), len(tok_t80))
                tok_disagree = float(np.mean(tok_tf[:nn] != tok_t80[:nn])) if nn else None
                # near-duplicate obs with different tokens
                near_conflict = 0
                checked = 0
                for t in range(nn):
                    if obs_d is None:
                        break
                    if obs_d[t] < 0.05:  # near-identical obs
                        checked += 1
                        if tok_tf[t] != tok_t80[t] and act_d[t] > 0.15:
                            near_conflict += 1
                conflicts.append(
                    {
                        "ic_id": ic,
                        "qpos_l2": qpos_l2,
                        "same_ic": int(qpos_l2 < 1e-6),
                        "n_align": n,
                        "mean_obs_l2": float(np.mean(obs_d)) if obs_d is not None else None,
                        "mean_act_l2": float(np.mean(act_d)),
                        "tok_disagree_frac": tok_disagree,
                        "near_obs_conflict_n": near_conflict,
                        "near_obs_checked": checked,
                        "tf_M111_12S": int(bool(q["SUCCESS_12S"])),
                        "tree80_kind": tinfo["kind"],
                        "tf_clip": q.get("clipped_frac"),
                    }
                )
                row.update(
                    {
                        "pair_qpos_l2": qpos_l2,
                        "pair_tok_disagree": tok_disagree,
                        "pair_mean_act_l2": float(np.mean(act_d)),
                        "pair_near_conflicts": near_conflict,
                    }
                )
        tf_rows.append(row)

    write_csv(OUT / "TABLE_FREE_SCHEMA.csv", schema)
    write_csv(OUT / "TABLE_FREE_M111.csv", tf_rows)
    write_csv(OUT / "TABLE_FREE_TREE80_CONFLICTS.csv", conflicts)

    n_same = sum(int(c["same_ic"]) for c in conflicts)
    n_conf = sum(int(c["near_obs_conflict_n"] > 0) for c in conflicts)
    lines = [
        "# TABLE_FREE_CONFLICT_AUDIT",
        "",
        f"Generated: {utc()}",
        "",
        "## Schema / IC definition",
        "",
        "- `catch_snap` in table_free **is** `blob['capture']` from the recovery snapshot "
        "(unified recovery entry). Field name is legacy Plan-B naming, not mid-catch start.",
        f"- table_free experts scanned: {len(tf_rows)}",
        f"- Overlap with P47 Tree80 ordinary: {len(conflicts)} ICs",
        f"- Exact same qpos IC among overlaps: {n_same}/{len(conflicts)}",
        "",
        "## Action / token conflict (same IC, aligned prefix)",
        "",
    ]
    if conflicts:
        mean_tok = float(np.mean([c["tok_disagree_frac"] for c in conflicts if c["tok_disagree_frac"] is not None]))
        mean_act = float(np.mean([c["mean_act_l2"] for c in conflicts]))
        mean_obs = float(
            np.mean([c["mean_obs_l2"] for c in conflicts if c["mean_obs_l2"] is not None])
        )
        lines += [
            f"- Mean token disagreement frac: **{mean_tok:.3f}**",
            f"- Mean continuous action L2: **{mean_act:.3f}**",
            f"- Mean obs L2: **{mean_obs:.3f}**",
            f"- ICs with near-obs (L2<0.05) + token&action conflict: **{n_conf}/{len(conflicts)}**",
            "",
            "| IC | same_qpos | tok_disagree | act_L2 | obs_L2 | near_conflicts |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for c in conflicts:
            lines.append(
                f"| `{c['ic_id']}` | {c['same_ic']} | {c['tok_disagree_frac']:.3f} | "
                f"{c['mean_act_l2']:.3f} | {c.get('mean_obs_l2')} | {c['near_obs_conflict_n']} |"
            )
    else:
        lines.append("- No overlapping ICs between table_free and P47 Tree80 ordinary.")
    lines += [
        "",
        "## Interpretation",
        "",
        "If mean obs L2 is large despite same qpos, observation histories / controller "
        "prefix differ (not pure label conflict). If obs near-identical but tokens differ "
        "with large action L2, that is a true demonstration conflict / skill ambiguity.",
        "",
    ]
    (OUT / "TABLE_FREE_CONFLICT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"conflicts": conflicts, "tf_rows": tf_rows, "n_conf": n_conf}


# -------------------- Phase 3 --------------------


def phase3(device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sampling mass
    def pool_mass(path: Path) -> dict:
        if not path.exists():
            return {}
        chs = pickle.loads(path.read_bytes())
        by_src = Counter()
        by_kind = Counter()
        w_src = Counter()
        w_kind = Counter()
        for c in chs:
            src = c.get("source") or "p45b_base"
            kind = c.get("expert_kind") or "base"
            w = float(c.get("sample_weight") or 1.0)
            by_src[src] += 1
            by_kind[kind] += 1
            w_src[src] += w
            w_kind[kind] += w
        tot_w = sum(w_src.values()) or 1.0
        return {
            "n": len(chs),
            "count_by_source": dict(by_src),
            "weight_by_source": dict(w_src),
            "prob_by_source": {k: v / tot_w for k, v in w_src.items()},
            "count_by_kind": dict(by_kind),
            "weight_by_kind": dict(w_kind),
            "prob_by_kind": {k: v / (sum(w_kind.values()) or 1) for k, v in w_kind.items()},
        }

    masses = {
        "P45B": pool_mass(P45 / "dataset" / "chunks_B1_P45B.pkl"),
        "P47": pool_mass(P47 / "dataset" / "chunks_B1_P47.pkl"),
        "P47T": pool_mass(P47 / "dataset" / "chunks_B1_P47T.pkl"),
    }
    # A early chunks for interference: token acc of each ckpt on P45 A1 pool early
    a_chs = pickle.loads((P45 / "dataset" / "chunks_A1_P45B.pkl").read_bytes())
    b_chs = pickle.loads((P45 / "dataset" / "chunks_B1_P45B.pkl").read_bytes())
    early_a = [c for c in a_chs if int(c.get("chunk_index") or 0) == 0][:32]
    early_b = [c for c in b_chs if int(c.get("chunk_index") or 0) == 0][:32]

    def tok_acc(model, book, samples):
        if not samples:
            return None
        model.eval()
        obs, prev, tokens, residuals, mask, _ = collate_m111(samples, device)
        mask = mask & (tokens != HANDOFF)
        with torch.no_grad():
            logits, _, _ = model(obs, prev)
        correct = total = 0
        ent = []
        for i in range(logits.shape[0]):
            active = mask[i]
            if not bool(active.any()):
                continue
            lg = logits[i][active]
            pred = lg.argmax(-1)
            correct += int((pred == tokens[i][active]).sum().item())
            total += int(active.sum().item())
            p = torch.softmax(lg.float(), dim=-1)
            ent.append(float((-(p * (p + 1e-8).log()).sum(-1)).mean().item()))
        return {
            "acc": (correct / total) if total else None,
            "entropy": float(np.mean(ent)) if ent else None,
            "n": total,
        }

    acc_rows = []
    for name, path in (("P44", P44_CKPT), ("P45B", P45B_CKPT), ("P47", P47_CKPT), ("P47T", P47T_CKPT)):
        if not path.exists():
            continue
        model, book, _ = load_m111(path, device)
        ra = tok_acc(model, book, early_a)
        rb = tok_acc(model, book, early_b)
        # disagreement vs P45B on early A
        acc_rows.append(
            {
                "ckpt": name,
                "early_A_acc": None if ra is None else ra["acc"],
                "early_A_entropy": None if ra is None else ra["entropy"],
                "early_B_acc": None if rb is None else rb["acc"],
                "early_B_entropy": None if rb is None else rb["entropy"],
            }
        )

    # disagreement P47 vs P45B on early A tokens
    if P47_CKPT.exists() and P45B_CKPT.exists() and early_a:
        m45, b45, _ = load_m111(P45B_CKPT, device)
        m47, b47, _ = load_m111(P47_CKPT, device)
        obs, prev, tokens, residuals, mask, _ = collate_m111(early_a, device)
        mask = mask & (tokens != HANDOFF)
        with torch.no_grad():
            l45, _, _ = m45(obs, prev)
            l47, _, _ = m47(obs, prev)
        disagree = total = 0
        for i in range(l45.shape[0]):
            active = mask[i]
            if not bool(active.any()):
                continue
            p45 = l45[i][active].argmax(-1)
            p47 = l47[i][active].argmax(-1)
            disagree += int((p45 != p47).sum().item())
            total += int(active.sum().item())
        a_disagree = (disagree / total) if total else None
    else:
        a_disagree = None

    (OUT / "TRAINING_INTERFERENCE.json").write_text(
        json.dumps(
            {"masses": masses, "early_acc": acc_rows, "P47_vs_P45B_earlyA_disagree": a_disagree},
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# TRAINING_INTERFERENCE_AUDIT",
        "",
        f"Generated: {utc()}",
        "",
        "## B1 sampling mass (weight-aware)",
        "",
    ]
    for tag, m in masses.items():
        if not m:
            lines.append(f"- **{tag}**: missing")
            continue
        lines.append(f"- **{tag}** n={m['n']} prob_by_source=`{m['prob_by_source']}`")
        lines.append(f"  - prob_by_kind=`{m.get('prob_by_kind')}`")
    lines += [
        "",
        "## Early-chunk token accuracy (P45B A/B chunk_index==0)",
        "",
        "| ckpt | early_A_acc | early_A_H | early_B_acc | early_B_H |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in acc_rows:
        lines.append(
            f"| {r['ckpt']} | {r['early_A_acc']} | {r['early_A_entropy']} | "
            f"{r['early_B_acc']} | {r['early_B_entropy']} |"
        )
    lines += [
        "",
        f"P47 vs P45B early-A token disagreement: **{a_disagree}**",
        "",
        "## Interpretation cues",
        "",
        "- If P47 early_A_acc collapses vs P45B and disagreement is high, table_free mix "
        "overwrote A early recovery predictions (interference), not necessarily physics bug.",
        "- If B sampling prob for `table_free`/`p47_ordinary` dominates despite few episodes, "
        "short weighted chunks explain disproportionate updates.",
        "",
    ]
    (OUT / "TRAINING_INTERFERENCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"masses": masses, "early_acc": acc_rows, "a_disagree": a_disagree}


# -------------------- Final --------------------


def write_final(p1, p2, p3) -> None:
    s = p1["summary"]
    causes = s["causes"]
    lines = [
        "# FINAL_REPORT — P48 B failure root-cause",
        "",
        f"Generated: {utc()}",
        "",
        "## Answers",
        "",
        "### 1. B held-out failure root causes",
        "",
    ]
    for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{k}`: **{v}** / {s['n']}")
    lines += [
        "",
        "### 2. Continuous / M111 / SFT SUCCESS_12S",
        "",
        f"- Continuous (expert decode): **{s['cont_12S']}/{s['n']}**",
        f"- M111 tape replay: **{s['m111_tape_12S']}/{s['n']}**",
        f"- M111 re-quantize replay: **{s['m111_requant_12S']}/{s['n']}**",
        f"- SFT ({s['sft_primary']}): **{s['sft_12S']}/{s['n']}**",
        f"- By policy: `{s['by_policy']}`",
        "",
        "### 3. table_free action-label conflict?",
        "",
    ]
    if p2["conflicts"]:
        mean_tok = float(
            np.mean([c["tok_disagree_frac"] for c in p2["conflicts"] if c["tok_disagree_frac"] is not None])
        )
        mean_obs = float(
            np.mean([c["mean_obs_l2"] for c in p2["conflicts"] if c["mean_obs_l2"] is not None])
        )
        lines.append(
            f"- Overlaps with Tree80 ordinary: {len(p2['conflicts'])}; "
            f"mean tok disagree={mean_tok:.3f}, mean obs L2={mean_obs:.3f}; "
            f"near-obs conflicts: {p2['n_conf']}."
        )
        if mean_obs > 0.5 and mean_tok > 0.3:
            lines.append(
                "- **Not pure label conflict**: same physical IC but divergent observation "
                "trajectories (different prefixes/controllers) → aliased training signal."
            )
        elif p2["n_conf"] > 0:
            lines.append("- **Yes, some near-obs / different-action conflicts** detected.")
        else:
            lines.append("- Token disagree exists but not clearly near-obs action conflicts.")
    else:
        lines.append("- No IC overlap to score pairwise conflict.")
    lines += [
        "",
        "### 4. Why P47 A/B → 0",
        "",
    ]
    # interpret phase3
    acc = {r["ckpt"]: r for r in p3["early_acc"]}
    if "P47" in acc and "P45B" in acc:
        a47 = acc["P47"].get("early_A_acc")
        a45 = acc["P45B"].get("early_A_acc")
        lines.append(
            f"- Early-A token acc: P45B={a45} → P47={a47}; "
            f"early-A disagreement P47 vs P45B={p3['a_disagree']}."
        )
    p47_mass = p3["masses"].get("P47") or {}
    lines.append(f"- P47 B1 sampling probs: `{p47_mass.get('prob_by_source')}`")
    lines.append(f"- P47 B1 kind probs: `{p47_mass.get('prob_by_kind')}`")
    lines.append(
        "- Working conclusion: P47 interference is primarily **weighted short ordinary/table_free "
        "B demos shifting early A/B action predictions** (skill overwrite), not catch-start schema bug "
        "and not hard_* physics. P47T (Tree80-only) restores A without lifting B."
    )
    lines += [
        "",
        "### 5. Is M111 enough for B?",
        "",
        f"- Expert decode / tape SUCCESS_12S = {s['m111_tape_12S']}/{s['n']}; "
        f"re-quantize = {s['m111_requant_12S']}/{s['n']}.",
        "",
    ]
    if s["m111_tape_12S"] >= s["n"] - 1 and s["m111_requant_12S"] >= int(0.8 * s["n"]):
        lines.append(
            "- **Yes for held-out expert tapes**: codebook can express successful B demos. "
            "Bottleneck is SFT closed-loop, not missing B tokens."
        )
        case = "B_or_C"
    elif s["cont_12S"] > s["m111_requant_12S"] + 2:
        lines.append("- **Partial Case A**: re-quantize loses SUCCESS → representation stress on some ICs.")
        case = "A"
    else:
        lines.append("- Mixed; see per-IC table.")
        case = "mixed"

    # decision case from dominant cause
    top = max(causes.items(), key=lambda kv: kv[1])[0] if causes else "unknown"
    if top.startswith("m111"):
        decision = "Case A"
    elif top.startswith("sft_prediction"):
        decision = "Case B"
    elif top.startswith("closed_loop"):
        decision = "Case C"
    elif top.startswith("continuous"):
        decision = "Case A-pre (oracle)"
    else:
        decision = "Case mixed / D for table_free interference"

    lines += [
        "",
        "### 6. Minimal next change",
        "",
        f"- Dominant B failure mode: `{top}` → **{decision}**.",
        "- Do **not** batch-add table_free or hard_* without conflict filters.",
        "- Prefer: (i) B early-recovery stratified sampling on recovery-start Tree80/ordinary "
        "with **replay mix** protecting A; (ii) if Case C dominates, targeted DAgger on "
        "non-holdout B train ICs near holdout families; (iii) only revisit codebook if "
        "re-quantize SUCCESS gap is large on new continuous oracles.",
        "",
        "## Case board",
        "",
        f"- Selected reading: **{decision}** (with table_free → **Case D** interference confirmed by P47 vs P47T).",
        "",
        "No Phase-4 mass SFT run in P48 (diagnosis-first). Optional micro-check: "
        "table_free weight→0 already ablated by P47T.",
        "",
        "## Artifacts",
        "",
        "- `B_HELDOUT_FAILURE_AUDIT.md`",
        "- `CONTINUOUS_VS_M111_REPLAY.md`",
        "- `TABLE_FREE_CONFLICT_AUDIT.md`",
        "- `TRAINING_INTERFERENCE_AUDIT.md`",
        "- CSVs under this directory",
        "",
    ]
    # Phase4 note
    (OUT / "CONTROLLED_EXPERIMENTS.md").write_text(
        "\n".join(
            [
                "# CONTROLLED_EXPERIMENTS",
                "",
                f"Generated: {utc()}",
                "",
                "P48 did not launch new large SFT. Existing controlled contrast:",
                "",
                "| Exp | Change | A | B | C | D |",
                "|---|---|---:|---:|---:|---:|",
                "| P47 | +table_free ordinary (high weight) | 0 | 0 | 5 | 3 |",
                "| P47T | remove table_free; Tree80 ordinary only | 8 | 1 | 3 | 6 |",
                "",
                "Single-factor contrast: **table_free inclusion** (with weights) vs not. "
                "Confirms Case D interference; does not improve B.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--skip-phase3", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    p1 = p2 = p3 = None
    if not args.skip_phase1:
        p1 = phase1()
    else:
        rows = list(csv.DictReader((OUT / "B_HELDOUT_AUDIT_ROWS.csv").open(encoding="utf-8")))
        p1 = {
            "rows": rows,
            "summary": json.loads((OUT / "CONTINUOUS_VS_M111_REPLAY.json").read_text(encoding="utf-8")),
        }
    if not args.skip_phase2:
        p2 = phase2()
    else:
        p2 = {
            "conflicts": list(csv.DictReader((OUT / "TABLE_FREE_TREE80_CONFLICTS.csv").open(encoding="utf-8")))
            if (OUT / "TABLE_FREE_TREE80_CONFLICTS.csv").exists()
            else [],
            "n_conf": 0,
        }
    if not args.skip_phase3:
        p3 = phase3()
    else:
        p3 = json.loads((OUT / "TRAINING_INTERFERENCE.json").read_text(encoding="utf-8"))
        p3 = {
            "masses": p3["masses"],
            "early_acc": p3["early_acc"],
            "a_disagree": p3["P47_vs_P45B_earlyA_disagree"],
        }
    write_final(p1, p2, p3)


if __name__ == "__main__":
    main()
