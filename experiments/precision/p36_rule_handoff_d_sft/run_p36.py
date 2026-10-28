"""P36 — Rule-based geometric HANDOFF + D-only motion SFT.

HANDOFF is not learned. Geometric targets = original Precision θ/dx/dz tolerances.
STRICT 12s validation remains mandatory after HANDOFF.
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
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import (  # noqa: E402
    policy_step_K,
)
from experiments.precision.p36_rule_handoff_d_sft.geometry import (  # noqa: E402
    DX_TOL_MM,
    DZ_TOL_MM,
    THETA_TOL_DEG,
    geometric_target_reached,
)
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book, save_m111  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402
from training.m111_sft import eval_token_metrics, train_m111  # noqa: E402

OUT = ROOT / "results/diagnostics/p36_rule_handoff_d_sft"
P35 = ROOT / "results/diagnostics/p35_precision_sft"
P35_MAN = P35 / "PACKAGED_MANIFEST.csv"
D1_CKPT = P35 / "checkpoints/D1_p1_only/checkpoint_best.pt"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
SEED = 36
MAX_STEPS = 2500


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


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def abs_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_p1_manifest() -> list[dict]:
    rows = list(csv.DictReader(P35_MAN.open(encoding="utf-8")))
    return [r for r in rows if r.get("class") == "P1"]


def classify_ic(env, row: dict) -> dict:
    blob = pickle.loads(abs_path(row["ic_path"]).read_bytes())
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    ok, g = geometric_target_reached(env.sim)
    return {
        "traj_id": row["traj_id"],
        "split": row["split"],
        "initially_target_satisfied": int(ok),
        "nontrivial": int(not ok),
        **{f"init_{k}": v for k, v in g.items() if k in ("dtheta_deg", "dx_mm", "dz_mm", "e_pos_mm", "bilateral")},
    }


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def rollout(
    env,
    model,
    book,
    rho,
    row: dict,
    *,
    device,
    mode: str,
    max_steps: int = MAX_STEPS,
    K: int = WINDOW,
) -> dict[str, Any]:
    """mode ∈ {learned_handoff, rule_handoff}.

    rule_handoff: never execute learned HANDOFF; terminate when geometry ready, then STRICT.
    """
    blob = pickle.loads(abs_path(row["ic_path"]).read_bytes())
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    ref = blob["ref"]
    hist = env._hist

    init_ok, init_g = geometric_target_reached(env.sim)
    obs_seq: list = []
    prev_seq: list = []
    tokens: list[int] = []
    actions: list = []
    n_motion = 0
    termination = "timeout"
    geo_reach_step = None
    drop = False
    contact_loss = False

    for t in range(max_steps):
        if latched(env.sim):
            drop = True
            termination = "drop"
            break
        geo_ok, geo = geometric_target_reached(env.sim)
        if mode == "rule_handoff" and geo_ok:
            geo_reach_step = t
            termination = "rule_handoff"
            break

        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K)
        tokens.append(tok)

        if mode == "learned_handoff":
            if action is None:
                termination = "learned_handoff"
                # record whether geometry was ready at learned stop
                geo_ok, geo = geometric_target_reached(env.sim)
                if geo_ok:
                    geo_reach_step = t
                break
        else:
            # rule mode: if policy wants HANDOFF, ignore and keep last motion / skip
            if action is None:
                # remap: treat as no-op motion skip — force continue with zero wrist hold grip
                # Better: re-sample argmax over motion logits only
                with torch.no_grad():
                    logits, resid, _ = model(
                        torch.as_tensor(np.stack(obs_seq[-K:]), device=device),
                        torch.as_tensor(np.stack(prev_seq[-K:]), device=device),
                    )
                    logits = logits[-1].clone()
                    logits[HANDOFF] = -1e9
                    tok = int(torch.argmax(logits).item())
                    from models.m111_checkpoint import decode_m111_action

                    rr = resid[-1].float().cpu().numpy().astype(np.float32)
                    action = decode_m111_action(book, tok, rr, rho=rho)
                    tokens[-1] = tok

        step_legal(env.sim, action)
        env._prev = action
        actions.append(action)
        n_motion += 1

        # rough contact loss probe
        geo_ok2, geo2 = geometric_target_reached(env.sim)
        if float(geo2.get("bilateral") or 0) < 0.5:
            contact_loss = True

    # Final geometry
    geo_ok_f, geo_f = geometric_target_reached(env.sim)
    if geo_ok_f and geo_reach_step is None and termination in ("rule_handoff", "learned_handoff", "timeout"):
        if termination == "rule_handoff" or (termination == "learned_handoff" and geo_ok_f):
            geo_reach_step = n_motion

    # STRICT only when we intended HANDOFF (learned or rule). Timeout → still evaluate
    # physical state but do NOT count as successful recovery via HANDOFF.
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    intended_handoff = termination in ("rule_handoff", "learned_handoff")
    strict_ok = bool(hand.get("SUCCESS_NEW")) and intended_handoff

    # Failure category
    if strict_ok:
        fail_cat = "SUCCESS"
    elif drop:
        fail_cat = "DROP"
    elif termination == "timeout":
        fail_cat = "TIMEOUT_NO_HANDOFF"
    elif intended_handoff and not geo_ok_f:
        fail_cat = "HANDOFF_WITHOUT_GEOMETRY"
    elif intended_handoff and geo_ok_f and not hand.get("SUCCESS_NEW"):
        if hand.get("SUCCESS_12S"):
            fail_cat = "GEOMETRY_OK_STRICT_HOLD_FAIL"
        else:
            fail_cat = f"GEOMETRY_OK_STRICT_FAIL:{hand.get('reason')}"
    elif contact_loss:
        fail_cat = "CONTACT_LOSS"
    else:
        fail_cat = f"OTHER:{termination}"

    return {
        "traj_id": row["traj_id"],
        "lineage_id": row.get("lineage_id"),
        "split": row.get("split"),
        "mode": mode,
        "initially_target_satisfied": int(init_ok),
        "nontrivial": int(not init_ok),
        "init_dtheta_deg": init_g["dtheta_deg"],
        "init_dx_mm": init_g["dx_mm"],
        "init_dz_mm": init_g["dz_mm"],
        "n_motion": n_motion,
        "termination": termination,
        "geometric_reach": int(geo_ok_f or geo_reach_step is not None),
        "geometric_reach_step": geo_reach_step,
        "handoff_triggered": int(intended_handoff),
        "STRICT_SUCCESS": int(strict_ok),
        "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
        "SUCCESS_NEW_raw": int(bool(hand.get("SUCCESS_NEW"))),
        "eval_reason": hand.get("reason"),
        "timeout": int(termination == "timeout"),
        "drop": int(drop),
        "contact_loss": int(contact_loss),
        "fail_category": fail_cat,
        "final_dtheta_deg": geo_f["dtheta_deg"],
        "final_dx_mm": geo_f["dx_mm"],
        "final_dz_mm": geo_f["dz_mm"],
        "tokens_head": tokens[:16],
        "theta_tol_deg": THETA_TOL_DEG,
        "dx_tol_mm": DX_TOL_MM,
        "dz_tol_mm": DZ_TOL_MM,
    }


def eval_pool(
    label, model, book, rho, rows, device, env, mode: str, *, max_steps: int = MAX_STEPS
) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        row = rollout(env, model, book, rho, r, device=device, mode=mode, max_steps=max_steps)
        row["condition"] = label
        out.append(row)
        print(
            f"  [{label}] {i+1}/{len(rows)} {r['traj_id'][:44]} "
            f"init_sat={row['initially_target_satisfied']} "
            f"geo={row['geometric_reach']} hand={row['handoff_triggered']} "
            f"STRICT={row['STRICT_SUCCESS']} cat={row['fail_category']}",
            flush=True,
        )
    return out


def summarize(rows: list[dict], *, nontrivial_only: bool = False) -> dict:
    rs = [r for r in rows if (r.get("nontrivial") == 1 if nontrivial_only else True)]
    if nontrivial_only is False:
        pass
    n = len(rs)
    return {
        "n": n,
        "geometric_reach": sum(int(r["geometric_reach"]) for r in rs),
        "handoff": sum(int(r["handoff_triggered"]) for r in rs),
        "STRICT": sum(int(r["STRICT_SUCCESS"]) for r in rs),
        "timeout": sum(int(r["timeout"]) for r in rs),
        "drop": sum(int(r["drop"]) for r in rs),
        "contact_loss": sum(int(r["contact_loss"]) for r in rs),
        "initially_satisfied": sum(int(r["initially_target_satisfied"]) for r in rs),
        "fail_cats": dict(Counter(r["fail_category"] for r in rs)),
    }


# ---------------------------------------------------------------------------
# Motion-only dataset (no HANDOFF supervision)
# ---------------------------------------------------------------------------
def motion_samples(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        z = np.load(P35 / r["npz_relpath"], allow_pickle=False)
        toks = np.asarray(z["tokens"], np.int64)
        # drop terminal HANDOFF and matching obs/prev/resid
        if int(toks[-1]) == HANDOFF:
            n = len(toks) - 1
        else:
            n = len(toks)
        if n <= 0:
            continue
        ep = {
            "observations": np.asarray(z["observations"][:n], np.float32),
            "prev_actions": np.asarray(z["prev_actions"][:n], np.float32),
            "tokens": toks[:n].astype(np.int64),
            "residuals": np.asarray(z["residuals"][:n], np.float32),
            "loss_mask": np.ones(n, dtype=np.bool_),
        }
        # exclude any remaining HANDOFF tokens from loss
        ep["loss_mask"] = ep["loss_mask"] & (ep["tokens"] != HANDOFF)
        if not bool(ep["loss_mask"].any()):
            continue
        for ch in chunk_episode(ep, window=WINDOW):
            # zero-out HANDOFF positions if any slipped in
            mask = np.asarray(ch["loss_mask"], dtype=np.bool_) & (np.asarray(ch["tokens"]) != HANDOFF)
            if not bool(mask.any()):
                continue
            ch = {**ch, "loss_mask": mask}
            out.append(
                {
                    **ch,
                    "traj_id": r["traj_id"],
                    "lineage_id": r["lineage_id"],
                    "skill": "P1",
                    "skill_group": "P1",
                    "family": "D_motion",
                    "sample_weight": 1.0,
                    "split": r["split"],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_classify(env, p1: list[dict]) -> list[dict]:
    rows = []
    for r in p1:
        rows.append({**r, **classify_ic(env, r)})
    write_csv(OUT / "IC_GEOMETRY_CLASS.csv", rows)
    print(
        f"IC class: nontrivial={sum(r['nontrivial'] for r in rows)}/{len(rows)} "
        f"init_sat={sum(r['initially_target_satisfied'] for r in rows)}",
        flush=True,
    )
    return rows


def phase_ablation(device, book, rho, env, p1: list[dict]) -> list[dict]:
    if not D1_CKPT.is_file():
        raise FileNotFoundError(D1_CKPT)
    model, _, _ = load_m111(D1_CKPT, device)
    all_rows = []
    for mode, tag in (
        ("learned_handoff", "D1_learned_HANDOFF"),
        ("rule_handoff", "D1_rule_HANDOFF"),
    ):
        print(f"=== Ablation {tag} n={len(p1)} ===", flush=True)
        # Eval all splits; report nontrivial separately
        er = eval_pool(tag, model, book, rho, p1, device, env, mode=mode)
        all_rows.extend(er)
        for split in ("train", "val", "test"):
            sub = [r for r in er if r["split"] == split]
            s_all = summarize(sub)
            s_nt = summarize(sub, nontrivial_only=True)
            print(
                f"  {split} ALL STRICT={s_all['STRICT']}/{s_all['n']} "
                f"geo={s_all['geometric_reach']} hand={s_all['handoff']} to={s_all['timeout']}",
                flush=True,
            )
            print(
                f"  {split} NT  STRICT={s_nt['STRICT']}/{s_nt['n']} "
                f"geo={s_nt['geometric_reach']} hand={s_nt['handoff']} to={s_nt['timeout']}",
                flush=True,
            )
    write_csv(OUT / "ABLATION_D1.csv", all_rows)
    del model
    return all_rows


def phase_train(device, book, rho, env, p1: list[dict], *, skip_fit: bool = False) -> dict:
    train = [r for r in p1 if r["split"] == "train"]
    val = [r for r in p1 if r["split"] == "val"]
    test = [r for r in p1 if r["split"] == "test"]
    val_nt = [r for r in val if int(r.get("nontrivial") or 0) == 1]
    # Checkpoint selection on val nontrivial closed-loop — never test
    tr_s = motion_samples(train)
    va_s = motion_samples(val)
    out_dir = OUT / "checkpoints/D_motion_rule_handoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    if skip_fit and history_path.is_file():
        print(f"=== Skip fit; resume CL-select from {out_dir} ===", flush=True)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        result = {"history": history, "best": json.loads((out_dir / "best.json").read_text(encoding="utf-8")) if (out_dir / "best.json").is_file() else {}}
    else:
        print(f"=== D-only motion SFT train_chunks={len(tr_s)} val_chunks={len(va_s)} ===", flush=True)
        cfg = {
            "name": "D_motion_rule_handoff",
            "device": str(device),
            "seed": SEED,
            "epochs": 24,
            "batch_size": 8,
            "lr": 3e-4,
        }
        result = train_m111(
            train_samples=tr_s, val_samples=va_s, cfg=cfg, out_dir=out_dir, book=book, rho=rho
        )
        history = result.get("history") or []

    # Closed-loop selection: prioritize mid/late epochs (early usually all-timeout).
    # Never use test. Criterion = val NT STRICT then geo.
    last_ep = int(history[-1]["epoch"]) if history else 24
    probe_epochs = sorted(
        {
            ep
            for ep in range(6, last_ep + 1, 2)
        }
        | ({last_ep} if last_ep else set())
    )
    cl_sel_rows: list[dict] = []
    best_cl = {"STRICT": -1, "geo": -1, "epoch": 0, "path": None}
    for ep in probe_epochs:
        path = out_dir / f"checkpoint_epoch_{ep:04d}.pt"
        if not path.is_file():
            continue
        model_ep, _, _ = load_m111(path, device)
        print(f"=== Closed-loop val NT @ epoch {ep} n={len(val_nt)} ===", flush=True)
        er = eval_pool(
            f"CL_select_ep{ep}",
            model_ep,
            book,
            rho,
            val_nt,
            device,
            env,
            mode="rule_handoff",
            max_steps=1500,  # selection horizon; timeouts ≃ never-reach-geo
        )
        s = summarize(er, nontrivial_only=True)
        row = {
            "epoch": ep,
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "n_nontrivial": s["n"],
            "STRICT": s["STRICT"],
            "geo": s["geometric_reach"],
            "handoff": s["handoff"],
            "timeout": s["timeout"],
            "drop": s["drop"],
            "fail_cats": s["fail_cats"],
            "val_token_acc": next(
                (h.get("val_token_acc") for h in history if int(h["epoch"]) == ep), None
            ),
        }
        cl_sel_rows.append(row)
        print(
            f"  CL-select ep={ep} STRICT={s['STRICT']}/{s['n']} geo={s['geometric_reach']}",
            flush=True,
        )
        key = (s["STRICT"], s["geometric_reach"], -ep)
        best_key = (best_cl["STRICT"], best_cl["geo"], -(best_cl["epoch"] or 0))
        if key > best_key:
            best_cl = {
                "STRICT": s["STRICT"],
                "geo": s["geometric_reach"],
                "epoch": ep,
                "path": path,
                "summary": s,
            }
        del model_ep

    write_csv(OUT / "CLOSED_LOOP_VAL_SELECTION.csv", cl_sel_rows)
    if best_cl["path"] is None:
        ckpt = out_dir / "checkpoint_best.pt"
        selection = "fallback_val_token_acc"
    else:
        ckpt = Path(best_cl["path"])
        # Promote closed-loop winner
        import shutil

        shutil.copy2(ckpt, out_dir / "checkpoint_best_closed_loop.pt")
        selection = "val_nontrivial_STRICT_then_geo"
        atomic_json(
            out_dir / "best_closed_loop.json",
            {
                "epoch": best_cl["epoch"],
                "STRICT": best_cl["STRICT"],
                "geo": best_cl["geo"],
                "path": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                "selection": selection,
            },
        )

    model, _, _ = load_m111(ckpt, device)
    off = eval_token_metrics(model, tr_s, device, max_batches=40)

    eval_rows = []
    curve = []
    for split_name, pool in (("train", train), ("val", val), ("test", test)):
        er = eval_pool(
            f"D_motion_rule/{split_name}",
            model,
            book,
            rho,
            pool,
            device,
            env,
            mode="rule_handoff",
        )
        eval_rows.extend(er)
        s_all = summarize(er)
        s_nt = summarize(er, nontrivial_only=True)
        s_sat = summarize([r for r in er if r["initially_target_satisfied"] == 1])
        curve.append(
            {
                "condition": "D_motion_rule",
                "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                "selection": selection,
                "best_epoch": best_cl.get("epoch") or (result.get("best") or {}).get("epoch"),
                "eval_split": split_name,
                "train_token_acc": off.get("token_acc"),
                "n_all": s_all["n"],
                "STRICT_all": s_all["STRICT"],
                "geo_all": s_all["geometric_reach"],
                "hand_all": s_all["handoff"],
                "timeout_all": s_all["timeout"],
                "n_nontrivial": s_nt["n"],
                "STRICT_nontrivial": s_nt["STRICT"],
                "geo_nontrivial": s_nt["geometric_reach"],
                "hand_nontrivial": s_nt["handoff"],
                "timeout_nontrivial": s_nt["timeout"],
                "drop_nontrivial": s_nt["drop"],
                "n_init_sat": s_sat["n"],
                "STRICT_init_sat": s_sat["STRICT"],
                "fail_cats_nt": s_nt["fail_cats"],
                "fail_cats_init_sat": s_sat["fail_cats"],
            }
        )
        print(
            f"  SUMMARY {split_name} NT STRICT={s_nt['STRICT']}/{s_nt['n']} "
            f"geo={s_nt['geometric_reach']} hand={s_nt['handoff']} | "
            f"init_sat STRICT={s_sat['STRICT']}/{s_sat['n']}",
            flush=True,
        )

    write_csv(OUT / "TRAIN_EVAL.csv", eval_rows)
    write_csv(OUT / "LEARNING_SUMMARY.csv", curve)
    atomic_json(
        OUT / "train_meta.json",
        {
            "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
            "selection": selection,
            "best_epoch": best_cl.get("epoch"),
            "offline": off,
            "closed_loop_val_selection": cl_sel_rows,
            "thresholds": {
                "theta_tol_deg": THETA_TOL_DEG,
                "dx_tol_mm": DX_TOL_MM,
                "dz_tol_mm": DZ_TOL_MM,
            },
            "ic_counts": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "train_nt": sum(int(r.get("nontrivial") or 0) for r in train),
                "val_nt": len(val_nt),
                "test_nt": sum(int(r.get("nontrivial") or 0) for r in test),
            },
        },
    )
    del model
    return {"ckpt": ckpt, "curve": curve, "evals": eval_rows, "offline": off, "best_cl": best_cl}


def write_final(ablation: list[dict], train_info: dict, p1: list[dict]) -> None:
    def pick(cond, split, nontrivial=True):
        rows = [
            r
            for r in ablation
            if r.get("condition") == cond and r.get("split") == split
        ]
        return summarize(rows, nontrivial_only=nontrivial)

    d1_learn_te = pick("D1_learned_HANDOFF", "test")
    d1_rule_te = pick("D1_rule_HANDOFF", "test")
    d1_learn_tr = pick("D1_learned_HANDOFF", "train")
    d1_rule_tr = pick("D1_rule_HANDOFF", "train")

    curve = train_info.get("curve") or []
    tr_nt = next((c for c in curve if c["eval_split"] == "train"), {})
    va_nt = next((c for c in curve if c["eval_split"] == "val"), {})
    te_nt = next((c for c in curve if c["eval_split"] == "test"), {})

    # Scientific conclusion
    te_strict = int(te_nt.get("STRICT_nontrivial") or 0)
    te_n = int(te_nt.get("n_nontrivial") or 0)
    tr_strict = int(tr_nt.get("STRICT_nontrivial") or 0)
    tr_n = int(tr_nt.get("n_nontrivial") or 0)
    rate_te = te_strict / max(1, te_n)
    rate_tr = tr_strict / max(1, tr_n)

    if rate_tr >= 0.5 and rate_te >= 0.4:
        conclusion = (
            "D-only motion SFT + rule HANDOFF shows substantial held-out STRICT recovery. "
            "Skill D is promising but claim 'solved' only if rates stay high across seeds."
        )
        solved = False  # still require explicit high bar
    elif rate_tr >= 0.4 and rate_te < 0.25:
        conclusion = (
            "Training STRICT is moderate but held-out is poor → generalization / coverage bottleneck."
        )
        solved = False
    elif rate_tr < 0.25:
        conclusion = (
            "Training STRICT remains poor under rule HANDOFF → corrective closed-loop control "
            "(prediction drift / coverage), not merely termination, is still limiting."
        )
        solved = False
    else:
        conclusion = "Mixed results; see tables. Skill D not solved."
        solved = False

    # Ablation interpretation
    if d1_rule_te["STRICT"] > d1_learn_te["STRICT"] + 2:
        term_bottleneck = (
            "YES — rule HANDOFF substantially raises STRICT vs learned HANDOFF on the same D1 weights "
            "→ termination was a primary bottleneck for that checkpoint."
        )
    elif d1_rule_te["geometric_reach"] > d1_learn_te["geometric_reach"] + 2:
        term_bottleneck = (
            "PARTIAL — rule HANDOFF improves geometric reach / handoff rate; STRICT gains limited."
        )
    else:
        term_bottleneck = (
            "NO clear termination-only win — frozen D1 still fails to reach geometry or STRICT "
            "even with oracle geometric stop."
        )

    text = f"""# FINAL_REPORT — P36 Rule-based HANDOFF + D-only SFT

Generated: {utc()}

## Frozen geometric HANDOFF rule

Not tuned on evaluation data. Original PrecisionConfig / AXIS_V1 defaults:

| Quantity | Threshold | Frame |
|---|---:|---|
| dθ | ≤ **{THETA_TOL_DEG}°** | grasp θ error |
| dx | ≤ **{DX_TOL_MM} mm** | signed hand-frame lateral |
| dz | ≤ **{DZ_TOL_MM} mm** | signed hand-frame axial |

HANDOFF triggers only when **all three** hold. Then run unchanged 12s STRICT.
Reaching geometry ≠ recovery success.

## Dataset

- Source: P35 packaged R1b P1 (`PACKAGED_MANIFEST.csv`)
- Splits preserved: train/val/test
- IC geometry class: `IC_GEOMETRY_CLASS.csv`
- Nontrivial = not initially target-satisfied under the same thresholds
- Checkpoint selection: **val nontrivial closed-loop STRICT** (then geometric reach); never test
- IC geometry: ~86/139 initially AXIS_V1-satisfied (control); ~53 nontrivial (mostly θ>10°)

## 1. Frozen D1 ablation (checkpoint `{D1_CKPT.relative_to(ROOT).as_posix()}`)

Nontrivial test:

| Mode | n | Geo reach | HANDOFF | STRICT | Timeout | Drop |
|---|---:|---:|---:|---:|---:|---:|
| Learned HANDOFF | {d1_learn_te['n']} | {d1_learn_te['geometric_reach']} | {d1_learn_te['handoff']} | **{d1_learn_te['STRICT']}** | {d1_learn_te['timeout']} | {d1_learn_te['drop']} |
| Rule HANDOFF | {d1_rule_te['n']} | {d1_rule_te['geometric_reach']} | {d1_rule_te['handoff']} | **{d1_rule_te['STRICT']}** | {d1_rule_te['timeout']} | {d1_rule_te['drop']} |

Nontrivial train:

| Mode | n | Geo | HANDOFF | STRICT | Timeout |
|---|---:|---:|---:|---:|---:|
| Learned | {d1_learn_tr['n']} | {d1_learn_tr['geometric_reach']} | {d1_learn_tr['handoff']} | {d1_learn_tr['STRICT']} | {d1_learn_tr['timeout']} |
| Rule | {d1_rule_tr['n']} | {d1_rule_tr['geometric_reach']} | {d1_rule_tr['handoff']} | {d1_rule_tr['STRICT']} | {d1_rule_tr['timeout']} |

Failure cats (test rule): {d1_rule_te['fail_cats']}
Failure cats (test learned): {d1_learn_te['fail_cats']}

**Termination bottleneck?** {term_bottleneck}

## 2. D-only motion SFT (no learned HANDOFF)

- Checkpoint: `{train_info.get('ckpt')}`
- Best epoch: {te_nt.get('best_epoch')}
- Offline motion token_acc (train probe): {train_info.get('offline', {}).get('token_acc')}

Nontrivial closed-loop (rule HANDOFF):

| Split | n | Geo reach | HANDOFF | STRICT | Timeout | Drop |
|---|---:|---:|---:|---:|---:|---:|
| Train | {tr_nt.get('n_nontrivial')} | {tr_nt.get('geo_nontrivial')} | {tr_nt.get('hand_nontrivial')} | **{tr_nt.get('STRICT_nontrivial')}** | {tr_nt.get('timeout_nontrivial')} | {tr_nt.get('drop_nontrivial')} |
| Val | {va_nt.get('n_nontrivial')} | {va_nt.get('geo_nontrivial')} | {va_nt.get('hand_nontrivial')} | **{va_nt.get('STRICT_nontrivial')}** | {va_nt.get('timeout_nontrivial')} | — |
| Test (held-out) | {te_nt.get('n_nontrivial')} | {te_nt.get('geo_nontrivial')} | {te_nt.get('hand_nontrivial')} | **{te_nt.get('STRICT_nontrivial')}** | {te_nt.get('timeout_nontrivial')} | — |

Initially target-satisfied (safety/control, test): STRICT={te_nt.get('STRICT_init_sat')}/{te_nt.get('n_init_sat')}

Fail cats (train NT): {tr_nt.get('fail_cats_nt')}

## 3. Scientific conclusion

{conclusion}

**Is Skill D solved?** **{"NO" if not solved else "NO — rates still require multi-seed confirmation"}** — require substantial held-out STRICT on nontrivial ICs; current test rate = {te_strict}/{te_n}.

## Artifacts

- `IC_GEOMETRY_CLASS.csv`
- `ABLATION_D1.csv` (per-root)
- `TRAIN_EVAL.csv` (per-root)
- `LEARNING_SUMMARY.csv`
- `FINAL_REPORT.md`
- `checkpoints/D_motion_rule_handoff/checkpoint_best.pt`
"""
    write_md(OUT / "FINAL_REPORT.md", text)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-only", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--resume-select",
        action="store_true",
        help="Skip SFT fit; run closed-loop selection + eval on existing epoch checkpoints",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    p1 = load_p1_manifest()
    if args.limit > 0:
        # keep split balance roughly
        p1 = p1[: args.limit]
    print(
        f"=== P36 device={device} n_p1={len(p1)} "
        f"θ≤{THETA_TOL_DEG} |dx|≤{DX_TOL_MM} |dz|≤{DZ_TOL_MM} ===",
        flush=True,
    )

    p1 = phase_classify(env, p1)
    # merge class flags onto rows used later
    by_id = {r["traj_id"]: r for r in p1}

    ablation = []
    if not args.skip_ablation:
        ablation = phase_ablation(device, book, rho, env, p1)
    if args.ablation_only:
        write_final(ablation, {"curve": [], "offline": {}, "ckpt": ""}, p1)
        return

    train_info = {"curve": [], "offline": {}, "ckpt": ""}
    if not args.skip_train:
        train_info = phase_train(
            device, book, rho, env, p1, skip_fit=bool(args.resume_select)
        )
        train_info["ckpt"] = str(train_info["ckpt"].relative_to(ROOT)).replace("\\", "/")

    if args.skip_ablation and (OUT / "ABLATION_D1.csv").is_file():
        ablation = list(csv.DictReader((OUT / "ABLATION_D1.csv").open(encoding="utf-8")))
        for r in ablation:
            for k in (
                "geometric_reach",
                "handoff_triggered",
                "STRICT_SUCCESS",
                "timeout",
                "drop",
                "nontrivial",
                "initially_target_satisfied",
            ):
                if k in r and r[k] != "":
                    r[k] = int(float(r[k]))

    write_final(ablation, train_info, p1)


if __name__ == "__main__":
    main()
