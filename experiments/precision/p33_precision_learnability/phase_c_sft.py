"""P33 Phase C — Precision-only SFT learnability (after Phase B dataset exists)."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import rollout_corrected  # noqa: E402
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book, save_m111  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW, build_samples, chunk_episode, collate_m111  # noqa: E402
from training.m111_sft import eval_token_metrics, supervised_loss_m111, train_m111  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/diagnostics/p33_precision_learnability"
NPZ = OUT / "dataset" / "npz"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft/checkpoints/A_natural/checkpoint_best.pt"
P28 = ROOT / "results/diagnostics/p28_multiskill_distillation/checkpoints/C_distill_corr_K32/checkpoint_best.pt"
P31 = ROOT / "results/diagnostics/p31_skill_interference/checkpoints/M2_pcgrad/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
SEED = 33


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def package_ic_to_npz(env, ic_path: Path, book, rho, *, split: str, cls: str) -> dict | None:
    blob = pickle.loads(ic_path.read_bytes())
    acts = np.asarray(blob.get("actions"), np.float32).reshape(-1, 7)
    capture, ref = blob["capture"], blob["ref"]
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hist = env._hist
    obs_list, prev_list = [], []
    for a in acts:
        obs = np.asarray(observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist), np.float32)
        obs_list.append(obs)
        prev_list.append(env._prev.copy())
        step_legal(env.sim, a)
        env._prev = a
    hob = np.asarray(observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist), np.float32)
    obs_list.append(hob)
    prev_list.append(env._prev.copy())
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    if cls == "P1" and not hand.get("SUCCESS_NEW"):
        return None
    if cls == "P0":
        # zero-action HANDOFF tape
        restore_capture(env, capture)
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        obs0 = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist), np.float32
        )
        tokens = np.array([HANDOFF], np.int64)
        residuals = np.zeros((1, 7), np.float32)
        obs_arr = obs0[None, :]
        prev_arr = np.zeros((1, 7), np.float32)
    else:
        q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
        tokens = np.concatenate([np.asarray(q["tokens"], np.int64), np.array([HANDOFF], np.int64)])
        residuals = np.concatenate([np.asarray(q["residuals"], np.float32), np.zeros((1, 7), np.float32)], axis=0)
        obs_arr = np.stack(obs_list).astype(np.float32)
        prev_arr = np.stack(prev_list).astype(np.float32)
    stem = hashlib.sha1(str(ic_path).encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    outp = OUT / rel
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp,
        observations=obs_arr,
        prev_actions=prev_arr,
        tokens=tokens,
        residuals=residuals,
        loss_mask=np.ones(len(tokens), dtype=np.bool_),
        episode_boundary=np.array([False] * (len(tokens) - 1) + [True]),
        handoff_id=np.int64(HANDOFF),
    )
    return {
        "traj_id": blob.get("tag") or ic_path.stem,
        "lineage_id": blob.get("lineage_id"),
        "physical_root": blob.get("physical_root"),
        "split": split,
        "class": cls,
        "npz_relpath": rel,
        "n_tokens": int(len(tokens)),
        "ic_path": str(ic_path.relative_to(ROOT)).replace("\\", "/"),
    }


def package_p0_from_immediate(env, capture, ref, book, rho, *, tag: str, lineage: str, split: str) -> dict | None:
    """Verified immediate-HANDOFF success → single HANDOFF token episode."""
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    if not hand.get("SUCCESS_NEW"):
        return None
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs0 = np.asarray(observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist), np.float32)
    tokens = np.array([HANDOFF], np.int64)
    residuals = np.zeros((1, 7), np.float32)
    stem = hashlib.sha1(("p0_" + tag).encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    outp = OUT / rel
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp,
        observations=obs0[None, :],
        prev_actions=np.zeros((1, 7), np.float32),
        tokens=tokens,
        residuals=residuals,
        loss_mask=np.ones(1, dtype=np.bool_),
        episode_boundary=np.array([True]),
        handoff_id=np.int64(HANDOFF),
    )
    return {
        "traj_id": tag,
        "lineage_id": lineage,
        "physical_root": lineage.split("__")[1] if "__" in lineage else lineage,
        "split": split,
        "class": "P0",
        "npz_relpath": rel,
        "n_tokens": 1,
    }


def load_episode(row: dict) -> list:
    z = np.load(OUT / row["npz_relpath"], allow_pickle=False)
    # Only sequence fields — scalars break chunk_episode slicing
    keys = ("observations", "prev_actions", "tokens", "residuals", "loss_mask")
    ep = {}
    for k in keys:
        if k not in z.files:
            continue
        arr = np.asarray(z[k])
        if arr.ndim == 0:
            continue
        ep[k] = arr
    if "loss_mask" not in ep and "tokens" in ep:
        ep["loss_mask"] = np.ones(len(ep["tokens"]), dtype=np.bool_)
    samples = []
    for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
        samples.append(
            {
                **ch,
                "traj_id": row["traj_id"],
                "lineage_id": row["lineage_id"],
                "skill": "P1" if row["class"] == "P1" else "P0",
                "skill_group": row["class"],
                "family": row["class"],
                "sample_weight": 1.0,
                "split": row["split"],
                "n_tokens": int(ch["tokens"].shape[0]),
            }
        )
    return samples


def balance_samples(p1_s: list, p0_s: list) -> list:
    """Equal skill-group weight via resampling minority."""
    if not p1_s and not p0_s:
        return []
    if not p0_s:
        return list(p1_s)
    if not p1_s:
        return list(p0_s)
    rng = np.random.default_rng(SEED)
    n = max(len(p1_s), len(p0_s))
    out = []
    for pool in (p1_s, p0_s):
        idx = rng.choice(len(pool), size=n, replace=True)
        out.extend(pool[int(i)] for i in idx)
    rng.shuffle(out)
    return out


def eval_pool(label, model, book, rho, rows, device, env) -> list[dict]:
    out = []
    for r in rows:
        ic = ROOT / r["ic_path"] if r.get("ic_path") else None
        if ic is None or not ic.is_file():
            continue
        blob = pickle.loads(ic.read_bytes())
        ep = {
            "traj_id": r["traj_id"],
            "lineage_id": r["lineage_id"],
            "capture": blob["capture"],
            "ref": blob["ref"],
            "actions": blob.get("actions"),
        }
        # zero HANDOFF
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        zh = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
        # policy
        pr = rollout_corrected(env, model, book, rho, ep, device=device, K=160, start_kind="d2")
        out.append(
            {
                "condition": label,
                "traj_id": r["traj_id"],
                "lineage_id": r["lineage_id"],
                "split": r.get("split"),
                "true_class": r.get("class"),
                "zero_SUCCESS_NEW": int(bool(zh.get("SUCCESS_NEW"))),
                "pol_SUCCESS_NEW": int(pr["SUCCESS_NEW"]),
                "pol_SUCCESS_NEW_handoff": int(pr["SUCCESS_NEW_policy_handoff"]),
                "pol_immediate_handoff": int(pr["immediate_handoff"]),
                "pol_n_motion": int(pr["n_motion"]),
                "pol_termination": pr["termination_kind"],
                "pol_reason": pr.get("eval_reason"),
                "tokens_head": pr.get("tokens_head"),
                # P0 metrics
                "p0_ok": int(r.get("class") == "P0" and pr["SUCCESS_NEW_policy_handoff"] and pr["immediate_handoff"]),
                "p0_overcontrol": int(r.get("class") == "P0" and (not pr["immediate_handoff"]) and not pr["SUCCESS_NEW_policy_handoff"]),
                # P1 metrics
                "p1_ok": int(r.get("class") == "P1" and pr["SUCCESS_NEW_policy_handoff"] and pr["n_motion"] > 0),
                "p1_handoff_shortcut": int(r.get("class") == "P1" and pr["immediate_handoff"]),
            }
        )
    return out


def run_phase_c() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gen_path = OUT / "GENERATED_DATASET.json"
    if not gen_path.is_file():
        raise RuntimeError("Phase B GENERATED_DATASET.json missing — do not start Phase C")
    gen = json.loads(gen_path.read_text(encoding="utf-8"))
    if int(gen.get("p1_total") or 0) < 1:
        _write_md(
            OUT / "FAILURE_ANALYSIS.md",
            f"# FAILURE_ANALYSIS\n\nGenerated: {_utc()}\n\nPhase C blocked: Phase B produced **0** P1 ICs. "
            f"See GENERATION_REPORT.md bottleneck. No Precision-only SFT was launched.\n",
        )
        _write_csv(OUT / "SFT_LEARNING_CURVE.csv", [])
        _write_csv(OUT / "CLOSED_LOOP_EVAL.csv", [])
        print("Phase C skipped — no P1 data", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    print(f"=== P33 Phase C device={device} ===", flush=True)

    # Package all P1 ICs
    packaged = []
    for split in ("train", "val", "test"):
        for r in gen.get(split) or []:
            ic = ROOT / r["ic_path"]
            row = package_ic_to_npz(env, ic, book, rho, split=split, cls="P1")
            if row:
                packaged.append(row)
                # also keep ic_path for eval
                packaged[-1]["ic_path"] = r["ic_path"]
                packaged[-1]["class"] = "P1"

    # P0 pool: side products + re-verify a few generated P0 if any; also package from GENERATED_P0 if we saved ICs
    # Minimal: create P0 from same donors' unperturbed captures where immediate STRICT works
    p0_rows = []
    # Use Phase A P0 official if not holdout — optional small set
    man = OUT / "DATASET_MANIFEST.csv"
    if man.is_file():
        with man.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("class") != "P0" or r.get("holdout_eval") in ("1", "True"):
                    continue
                if r.get("source") != "experts_v2_official":
                    continue
                # skip packaging heavy official without path — limit
                if len(p0_rows) >= 20:
                    break

    # Build nested train subsets by unique lineage count
    train_p1 = [r for r in packaged if r["split"] == "train"]
    val_p1 = [r for r in packaged if r["split"] == "val"]
    test_p1 = [r for r in packaged if r["split"] == "test"]
    # Also use val+test as held-out eval; if empty, hold out 20% of train by root
    if not val_p1 and not test_p1 and len(train_p1) >= 5:
        roots = sorted({r["physical_root"] for r in train_p1})
        rng = np.random.default_rng(SEED)
        rng.shuffle(roots)
        n_h = max(1, len(roots) // 5)
        hold_roots = set(roots[:n_h])
        val_p1 = [r for r in train_p1 if r["physical_root"] in hold_roots]
        train_p1 = [r for r in train_p1 if r["physical_root"] not in hold_roots]
        for r in val_p1:
            r["split"] = "val"

    def samples_for(rows):
        out = []
        for r in rows:
            out.extend(load_episode(r))
        return out

    # Nested sizes
    train_by_root = {}
    for r in train_p1:
        train_by_root.setdefault(r["physical_root"], []).append(r)
    roots = list(train_by_root.keys())
    # Nested subsets by trajectory count (actual yield may be below 25/50/100/200)
    n_tr = len(train_p1)
    sizes = sorted({max(1, n_tr // 3), max(1, (2 * n_tr) // 3), n_tr})
    sizes = [s for s in sizes if s <= n_tr]

    curve_rows = []
    eval_rows = []

    # C0 frozen baselines on held-out P1 (+ train for fit check)
    for tag, path, K in (("C0_P22", P22, 160), ("C0_P28", P28, 32), ("C0_P31", P31, 160)):
        if not path.is_file():
            continue
        m, _, _ = load_m111(path, device)
        for pool_name, pool in (("train", train_p1), ("val", val_p1), ("test", test_p1)):
            if not pool:
                continue
            # attach class
            for r in pool:
                r.setdefault("class", "P1")
            er = eval_pool(f"{tag}/{pool_name}", m, book, rho, pool, device, env)
            eval_rows.extend(er)
            curve_rows.append(
                {
                    "condition": tag,
                    "n_p1_train_target": 0,
                    "n_p1_train_actual": 0,
                    "eval_split": pool_name,
                    "n_eval": len(er),
                    "p1_closed_loop": sum(r["p1_ok"] for r in er),
                    "handoff_strict": sum(r["pol_SUCCESS_NEW_handoff"] for r in er),
                    "immediate_rate": sum(r["pol_immediate_handoff"] for r in er) / max(1, len(er)),
                    "mean_n_motion": float(np.mean([r["pol_n_motion"] for r in er])) if er else 0,
                }
            )
        del m

    # Learning curve C2–C5
    for target_n in sizes:
        # take first k trajectories (nested), preferring diverse roots
        chosen = []
        for root in roots:
            if len(chosen) >= target_n:
                break
            chosen.extend(train_by_root[root])
        chosen = chosen[: max(1, min(target_n, len(train_p1)))]
        name = f"C_P1n{len(chosen)}"
        print(f"=== train {name} n={len(chosen)} ===", flush=True)
        tr_s = samples_for(chosen)
        va_s = samples_for(val_p1) if val_p1 else samples_for(chosen[: max(1, len(chosen) // 5)])
        # balance with synthetic P0 from same ICs' immediate? skip if no P0 — use weight only P1
        out_dir = OUT / "checkpoints" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "name": name,
            "device": str(device),
            "seed": SEED,
            "epochs": 20,
            "batch_size": 8,
            "lr": 3e-4,
        }
        result = train_m111(train_samples=tr_s, val_samples=va_s, cfg=cfg, out_dir=out_dir, book=book, rho=rho)
        m, _, _ = load_m111(out_dir / "checkpoint_best.pt", device)
        # offline metrics
        tr_met = eval_token_metrics(m, tr_s, device, max_batches=50)
        # closed-loop train + held-out
        for pool_name, pool in (("train", chosen), ("val", val_p1), ("test", test_p1)):
            if not pool:
                continue
            for r in pool:
                r.setdefault("class", "P1")
            er = eval_pool(f"{name}/{pool_name}", m, book, rho, pool, device, env)
            eval_rows.extend(er)
            curve_rows.append(
                {
                    "condition": name,
                    "n_p1_train_target": target_n,
                    "n_p1_train_actual": len(chosen),
                    "eval_split": pool_name,
                    "n_eval": len(er),
                    "p1_closed_loop": sum(r["p1_ok"] for r in er),
                    "handoff_strict": sum(r["pol_SUCCESS_NEW_handoff"] for r in er),
                    "immediate_rate": sum(r["pol_immediate_handoff"] for r in er) / max(1, len(er)),
                    "mean_n_motion": float(np.mean([r["pol_n_motion"] for r in er])) if er else 0,
                    "train_token_acc": tr_met.get("token_acc"),
                    "best_epoch": result.get("best_epoch"),
                }
            )
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(OUT / "SFT_LEARNING_CURVE.csv", curve_rows)
    _write_csv(OUT / "CLOSED_LOOP_EVAL.csv", eval_rows)

    # Failure analysis
    train_fit = [r for r in eval_rows if "/train" in r["condition"] and r["condition"].startswith("C_P1")]
    val_gen = [r for r in eval_rows if "/val" in r["condition"] and r["condition"].startswith("C_P1")]
    fa = f"""# FAILURE_ANALYSIS

Generated: {_utc()}

## Train fit vs held-out

- Train-set closed-loop P1 successes (latest curve rows): see `SFT_LEARNING_CURVE.csv`.
- If train P1 closed-loop ≪ offline token acc → fit/control divergence (obs encoding, discretization, PE, HANDOFF mask).
- If train OK but val/test fail → coverage / lineage shift.
- If immediate HANDOFF on P1 dominates → HANDOFF shortcut.
- If P0 over-control high → unnecessary correction (need more P0 balance).

## Counts in this run

- Packaged train P1: {len(train_p1)}
- Val P1: {len(val_p1)}
- Test P1: {len(test_p1)}
- Curve conditions: {sorted({r['condition'] for r in curve_rows})}
"""
    _write_md(OUT / "FAILURE_ANALYSIS.md", fa)
    print("=== Phase C done ===", flush=True)


if __name__ == "__main__":
    run_phase_c()
