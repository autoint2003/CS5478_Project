"""P35 — D/Precision-only SFT learnability (R1b dataset).

SFT only. No unified A/B/C/D, no RL.
"""

from __future__ import annotations

import argparse
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
from experiments.precision.p28_multiskill_distillation.harness import (  # noqa: E402
    rollout_corrected,
)
from experiments.precision.p35_precision_sft.paths import (  # noqa: E402
    ENCODER_VERSION,
    M111_META,
    MAX_ROLLOUT_STEPS,
    OUT,
    P0_MANIFEST,
    P22,
    P28,
    P31,
    R1B_GRIP_WEIGHT,
    R1B_MANIFEST,
    R1B_PAIRED,
    ROOT as PROJ,
    SEED,
)
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book, save_m111  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode, collate_m111  # noqa: E402
from training.m111_sft import eval_token_metrics, train_m111  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402


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
    return path if path.is_absolute() else PROJ / path


def r1b_weights() -> np.ndarray:
    w = DEFAULT_WEIGHTS.copy()
    w[6] = float(R1B_GRIP_WEIGHT)
    return w


def load_r1b_manifest() -> list[dict]:
    rows = list(csv.DictReader(R1B_MANIFEST.open(encoding="utf-8")))
    return [r for r in rows if str(r.get("m111_trainable_r1b")) in ("1", "True", "true")]


# ---------------------------------------------------------------------------
# Phase 0 — audit + package
# ---------------------------------------------------------------------------
def package_p1(env, row: dict, book, rho) -> dict | None:
    """Build obs history by replaying R1b recon actions; tokens from R1b encode."""
    ic = abs_path(row["ic_path"])
    ep_path = abs_path(row["episode_path"])
    blob = pickle.loads(ic.read_bytes())
    z = np.load(ep_path, allow_pickle=True)
    tokens_full = np.asarray(z["tokens"], np.int64)  # includes terminal HANDOFF
    recon = np.asarray(z["recon_actions"], np.float32).reshape(-1, 7)
    residuals = np.asarray(z["residuals"], np.float32).reshape(-1, 7)
    if int(tokens_full[-1]) != HANDOFF:
        raise RuntimeError(f"{row.get('tag')}: last token not HANDOFF")
    if len(recon) != len(tokens_full) - 1:
        # re-encode if mismatch
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        q = quantize_trajectory(book, acts, rho, w=r1b_weights())
        recon = np.asarray(q["recon_actions"], np.float32)
        residuals = np.asarray(q["residuals"], np.float32)
        tokens_full = np.concatenate([np.asarray(q["tokens"], np.int64), [HANDOFF]])

    # Fresh STRICT verify under R1b recon
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in recon:
        step_legal(env.sim, a)
        env._prev = a
    h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
    if not h.get("SUCCESS_NEW"):
        return None

    # Obs tape aligned with expert: obs_t before action_t; final obs before HANDOFF
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hist = env._hist
    obs_list, prev_list = [], []
    for a in recon:
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_list.append(obs)
        prev_list.append(env._prev.copy())
        step_legal(env.sim, a)
        env._prev = a
    # HANDOFF step observation (state after last motion = policy-visible pre-handoff)
    hob = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
        np.float32,
    )
    obs_list.append(hob)
    prev_list.append(env._prev.copy())

    residuals_h = np.concatenate([residuals, np.zeros((1, 7), np.float32)], axis=0)
    if len(obs_list) != len(tokens_full):
        raise RuntimeError(
            f"obs/token length mismatch {len(obs_list)} vs {len(tokens_full)} for {row.get('tag')}"
        )

    stem = hashlib.sha1(f"p35_p1_{row.get('tag')}".encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    outp = OUT / rel
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp,
        observations=np.stack(obs_list).astype(np.float32),
        prev_actions=np.stack(prev_list).astype(np.float32),
        tokens=tokens_full.astype(np.int64),
        residuals=residuals_h.astype(np.float32),
        loss_mask=np.ones(len(tokens_full), dtype=np.bool_),
        episode_boundary=np.array([False] * (len(tokens_full) - 1) + [True]),
        handoff_id=np.int64(HANDOFF),
    )
    return {
        "traj_id": row.get("tag"),
        "lineage_id": row.get("lineage_id"),
        "physical_root": row.get("physical_root"),
        "split": row.get("split"),
        "class": "P1",
        "source": row.get("source"),
        "family": row.get("family") or "",
        "npz_relpath": rel,
        "ic_path": row.get("ic_path"),
        "n_tokens": int(len(tokens_full)),
        "n_motion": int(len(tokens_full) - 1),
        "encoder_version": ENCODER_VERSION,
        "strict_SUCCESS_NEW": 1,
        "strict_reason": h.get("reason"),
    }


def package_p0(env, row: dict) -> dict | None:
    ic = abs_path(row["ic_path"])
    if not ic.is_file():
        return None
    blob = pickle.loads(ic.read_bytes())
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
    if not h.get("SUCCESS_NEW"):
        return None
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs0 = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
        np.float32,
    )
    stem = hashlib.sha1(f"p35_p0_{row.get('tag')}".encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    outp = OUT / rel
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp,
        observations=obs0[None, :],
        prev_actions=np.zeros((1, 7), np.float32),
        tokens=np.array([HANDOFF], np.int64),
        residuals=np.zeros((1, 7), np.float32),
        loss_mask=np.ones(1, dtype=np.bool_),
        episode_boundary=np.array([True]),
        handoff_id=np.int64(HANDOFF),
    )
    # Assign P0 to train by default (no lineage conflict with P1 splits); hold out 10% val
    return {
        "traj_id": row.get("tag"),
        "lineage_id": row.get("lineage_id"),
        "physical_root": row.get("physical_root"),
        "split": "train",  # reassigned below
        "class": "P0",
        "source": row.get("source"),
        "family": row.get("family") or "P0_DONOR",
        "npz_relpath": rel,
        "ic_path": row.get("ic_path"),
        "n_tokens": 1,
        "n_motion": 0,
        "encoder_version": ENCODER_VERSION,
        "strict_SUCCESS_NEW": 1,
    }


def phase0_audit_and_package(*, limit_p1: int = 0, limit_p0: int = 0) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    r1b = load_r1b_manifest()
    if limit_p1 > 0:
        r1b = r1b[:limit_p1]
    print(f"=== P35 Phase0 package P1 n={len(r1b)} ===", flush=True)
    p1_rows = []
    fail_verify = []
    for i, r in enumerate(r1b):
        print(f"  P1 [{i+1}/{len(r1b)}] {r.get('tag')}", flush=True)
        row = package_p1(env, r, book, rho)
        if row is None:
            fail_verify.append(r.get("tag"))
            print(f"    FAIL re-verify STRICT", flush=True)
        else:
            p1_rows.append(row)

    p0_src = list(csv.DictReader(P0_MANIFEST.open(encoding="utf-8"))) if P0_MANIFEST.is_file() else []
    if limit_p0 > 0:
        p0_src = p0_src[:limit_p0]
    print(f"=== P35 Phase0 package P0 n={len(p0_src)} ===", flush=True)
    p0_rows = []
    for i, r in enumerate(p0_src):
        row = package_p0(env, r)
        if row:
            p0_rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  P0 [{i+1}/{len(p0_src)}] ok={len(p0_rows)}", flush=True)

    # Assign P0 splits by root: 80% train / 10% val / 10% test, disjoint from nothing critical
    rng = np.random.default_rng(SEED)
    roots = sorted({r["physical_root"] for r in p0_rows})
    rng.shuffle(roots)
    n = len(roots)
    n_te = max(1, n // 10) if n >= 10 else 0
    n_va = max(1, n // 10) if n >= 10 else 0
    te_r, va_r = set(roots[:n_te]), set(roots[n_te : n_te + n_va])
    for r in p0_rows:
        pr = r["physical_root"]
        r["split"] = "test" if pr in te_r else "val" if pr in va_r else "train"

    all_rows = p1_rows + p0_rows
    write_csv(OUT / "PACKAGED_MANIFEST.csv", all_rows)
    atomic_json(
        OUT / "phase0_summary.json",
        {
            "encoder_version": ENCODER_VERSION,
            "p1_r1b_trainable_in": len(r1b),
            "p1_packaged": len(p1_rows),
            "p1_reverify_fail": fail_verify,
            "p0_packaged": len(p0_rows),
            "splits_p1": dict(Counter(r["split"] for r in p1_rows)),
            "splits_p0": dict(Counter(r["split"] for r in p0_rows)),
            "families_p1": dict(Counter(r.get("family") or "?" for r in p1_rows)),
            "decoder_unchanged": True,
            "r1b_grip_weight": R1B_GRIP_WEIGHT,
            "max_rollout_steps": MAX_ROLLOUT_STEPS,
            "window": WINDOW,
        },
    )
    audit = f"""# DATASET_AUDIT — P35

Generated: {utc()}

## Encoder

- Version: `{ENCODER_VERSION}` (grip NN weight={R1B_GRIP_WEIGHT})
- Decoder / M111 IDs 0..110 / HANDOFF={HANDOFF}: **unchanged**
- R1b affects expert encoding only

## P1 packaging

- R1b trainable input: {len(r1b)}
- Re-verified STRICT + packaged: **{len(p1_rows)}**
- Re-verify failures: {len(fail_verify)} {fail_verify[:10]}
- Splits: {dict(Counter(r['split'] for r in p1_rows))}
- Families: {dict(Counter(r.get('family') or '?' for r in p1_rows))}

Obs history built by replaying **R1b recon actions**; terminal obs is the
policy-visible state immediately before expert HANDOFF. Tokens match R1b + HANDOFF.

## P0 packaging

- Packaged (immediate HANDOFF STRICT): **{len(p0_rows)}**
- Splits: {dict(Counter(r['split'] for r in p0_rows))}

## Holdouts

- P21/P22/P32 eval lineages: excluded upstream in P34 donor discovery (not reintroduced).
- P1 train/val/test: preserved from P34.6 manifest.

## Rollout horizon

`MAX_ROLLOUT_STEPS={MAX_ROLLOUT_STEPS}` (P33 used 800; expert tapes often ~1.7–2.1k).
"""
    write_md(OUT / "DATASET_AUDIT.md", audit)
    print(audit, flush=True)
    return {"p1": p1_rows, "p0": p0_rows}


def load_packaged() -> tuple[list[dict], list[dict]]:
    man = OUT / "PACKAGED_MANIFEST.csv"
    if not man.is_file():
        raise RuntimeError("PACKAGED_MANIFEST.csv missing — run phase0")
    rows = list(csv.DictReader(man.open(encoding="utf-8")))
    p1 = [r for r in rows if r.get("class") == "P1"]
    p0 = [r for r in rows if r.get("class") == "P0"]
    return p1, p0


def samples_for(rows: list[dict], *, handoff_boost: float = 40.0, handoff_dup: int = 8) -> list[dict]:
    """Build windowed samples. Boost/dup HANDOFF-terminal chunks (rare vs long motion).

    P35 diagnosis: default CE is motion-dominated; teacher-forced HANDOFF recall → 0
    and closed-loop always times out despite high motion token accuracy.
    """
    out = []
    for r in rows:
        z = np.load(OUT / r["npz_relpath"], allow_pickle=False)
        ep = {}
        for k in ("observations", "prev_actions", "tokens", "residuals", "loss_mask"):
            if k in z.files:
                arr = np.asarray(z[k])
                if arr.ndim > 0:
                    ep[k] = arr
        if "loss_mask" not in ep:
            ep["loss_mask"] = np.ones(len(ep["tokens"]), dtype=np.bool_)
        for ch in chunk_episode(ep, window=WINDOW):
            toks = np.asarray(ch["tokens"])
            has_h = bool(np.any(toks == HANDOFF))
            sample = {
                **ch,
                "traj_id": r["traj_id"],
                "lineage_id": r["lineage_id"],
                "skill": r["class"],
                "skill_group": r["class"],
                "family": r.get("family") or r["class"],
                "sample_weight": float(handoff_boost if has_h else 1.0),
                "split": r["split"],
                "has_handoff": int(has_h),
            }
            reps = handoff_dup if has_h else 1
            for _ in range(reps):
                out.append(dict(sample))
    return out


def balance_root_level(p1_rows: list[dict], p0_rows: list[dict], rng) -> list[dict]:
    """Balance by resampling chunk pools so P0 and P1 get equal sample counts."""
    s1 = samples_for(p1_rows)
    s0 = samples_for(p0_rows) if p0_rows else []
    if not s0:
        return s1
    if not s1:
        return s0
    n = max(len(s1), len(s0))
    out = []
    for pool in (s1, s0):
        idx = rng.choice(len(pool), size=n, replace=True)
        out.extend(pool[int(i)] for i in idx)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def eval_closed_loop(label, model, book, rho, rows, device, env) -> list[dict]:
    out = []
    for r in rows:
        ic = abs_path(r["ic_path"])
        if not ic.is_file():
            continue
        blob = pickle.loads(ic.read_bytes())
        ep = {
            "traj_id": r["traj_id"],
            "lineage_id": r["lineage_id"],
            "capture": blob["capture"],
            "ref": blob["ref"],
            "actions": blob.get("actions"),
        }
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        zh = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
        pr = rollout_corrected(
            env,
            model,
            book,
            rho,
            ep,
            device=device,
            K=WINDOW,
            max_steps=MAX_ROLLOUT_STEPS,
            start_kind="d2",
        )
        out.append(
            {
                "condition": label,
                "traj_id": r["traj_id"],
                "lineage_id": r["lineage_id"],
                "split": r.get("split"),
                "true_class": r.get("class"),
                "family": r.get("family"),
                "zero_SUCCESS_NEW": int(bool(zh.get("SUCCESS_NEW"))),
                "pol_SUCCESS_NEW": int(pr["SUCCESS_NEW"]),
                "pol_SUCCESS_NEW_handoff": int(pr["SUCCESS_NEW_policy_handoff"]),
                "pol_immediate_handoff": int(pr["immediate_handoff"]),
                "pol_n_motion": int(pr["n_motion"]),
                "pol_termination": pr["termination_kind"],
                "pol_reason": pr.get("eval_reason"),
                "tokens_head": pr.get("tokens_head"),
                "p0_ok": int(
                    r.get("class") == "P0"
                    and pr["SUCCESS_NEW_policy_handoff"]
                    and pr["immediate_handoff"]
                ),
                "p0_overcontrol": int(
                    r.get("class") == "P0"
                    and (not pr["immediate_handoff"])
                    and not pr["SUCCESS_NEW_policy_handoff"]
                ),
                "p1_ok": int(
                    r.get("class") == "P1"
                    and pr["SUCCESS_NEW_policy_handoff"]
                    and pr["n_motion"] > 0
                ),
                "p1_handoff_shortcut": int(r.get("class") == "P1" and pr["immediate_handoff"]),
                "timeout": int(pr["termination_kind"] == "timeout"),
            }
        )
        print(
            f"  [{label}] {r['traj_id'][:44]} class={r.get('class')} "
            f"term={pr['termination_kind']} NEW_h={pr['SUCCESS_NEW_policy_handoff']} "
            f"n_mot={pr['n_motion']}",
            flush=True,
        )
    return out


def handoff_counterfactual(env, model, book, rho, row, device, *, probe_every: int = 100) -> dict:
    """Offline CF: record policy actions, then test immediate HANDOFF after prefixes."""
    from experiments.precision.p28_multiskill_distillation.harness import policy_step_K

    ic = abs_path(row["ic_path"])
    blob = pickle.loads(ic.read_bytes())
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    hist = env._hist
    obs_seq, prev_seq, actions, tokens = [], [], [], []
    actual_handoff_step = None
    for t in range(MAX_ROLLOUT_STEPS):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, WINDOW)
        tokens.append(tok)
        if action is None:
            actual_handoff_step = t
            break
        step_legal(env.sim, action)
        env._prev = action
        actions.append(action)

    probe_steps = list(range(0, len(actions) + 1, max(1, probe_every)))
    if len(actions) not in probe_steps:
        probe_steps.append(len(actions))
    safe_steps = []
    for n in probe_steps:
        restore_capture(env, blob["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        for a in actions[:n]:
            step_legal(env.sim, a)
            env._prev = a
        h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
        if h.get("SUCCESS_NEW"):
            safe_steps.append(n)

    first_safe = safe_steps[0] if safe_steps else None
    last_safe = safe_steps[-1] if safe_steps else None
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in actions:
        step_legal(env.sim, a)
        env._prev = a
    h_end = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
    end_ok = bool(h_end.get("SUCCESS_NEW"))

    if first_safe is None and actual_handoff_step is None:
        diag = "A_NEVER_SAFE_TIMEOUT"
    elif first_safe is None and actual_handoff_step is not None:
        diag = "D_HANDOFF_WITHOUT_EVER_SAFE" if not end_ok else "OK_HANDOFF_NO_MID_PROBE"
    elif first_safe is not None and actual_handoff_step is None:
        diag = (
            "C_SAFE_THEN_DESTROYED_NO_HANDOFF"
            if last_safe is not None and last_safe < len(actions)
            else "B_REACHED_SAFE_NO_HANDOFF"
        )
    elif first_safe is not None and actual_handoff_step is not None:
        if end_ok:
            diag = "OK_CORRECT_AND_TERMINATE"
        elif actual_handoff_step > (last_safe or 0):
            diag = "C_SAFE_THEN_DESTROYED_THEN_HANDOFF"
        else:
            diag = "D_HANDOFF_STRICT_FAIL"
    else:
        diag = "UNRESOLVED"

    return {
        "traj_id": row["traj_id"],
        "n_motion": len(actions),
        "actual_handoff_step": actual_handoff_step,
        "first_safe_step": first_safe,
        "last_safe_step": last_safe,
        "n_safe_probes": len(safe_steps),
        "end_SUCCESS_NEW": int(end_ok),
        "diagnosis": diag,
        "tokens_head": tokens[:12],
    }


# ---------------------------------------------------------------------------
# Phase 1 — overfit
# ---------------------------------------------------------------------------
def select_overfit_roots(p1_train: list[dict], n: int = 8) -> list[dict]:
    """Diverse short+long P1 train roots."""
    by_fam: dict[str, list] = {}
    for r in p1_train:
        by_fam.setdefault(r.get("family") or "?", []).append(r)
    # prefer shorter within each family for learnability, but keep diversity
    chosen = []
    for fam, rs in sorted(by_fam.items()):
        rs = sorted(rs, key=lambda x: int(x.get("n_motion") or 10**9))
        chosen.append(rs[0])
        if len(chosen) >= n:
            break
    # fill with more short ones
    rest = sorted(
        [r for r in p1_train if r not in chosen],
        key=lambda x: int(x.get("n_motion") or 10**9),
    )
    for r in rest:
        if len(chosen) >= n:
            break
        if r["physical_root"] in {c["physical_root"] for c in chosen}:
            continue
        chosen.append(r)
    return chosen[:n]


def phase1_overfit(p1_rows: list[dict], device, book, rho, env) -> dict:
    train_p1 = [r for r in p1_rows if r["split"] == "train"]
    chosen = select_overfit_roots(train_p1, n=8)
    print(
        f"=== Phase1 overfit n={len(chosen)} motions={[r['n_motion'] for r in chosen]} ===",
        flush=True,
    )
    tr_s = samples_for(chosen)
    # val = same set for overfit (intentional)
    out_dir = OUT / "checkpoints" / "overfit_p1_8"
    cfg = {
        "name": "overfit_p1_8",
        "device": str(device),
        "seed": SEED,
        "epochs": 60,
        "batch_size": 4,
        "lr": 3e-4,
    }
    result = train_m111(
        train_samples=tr_s, val_samples=tr_s, cfg=cfg, out_dir=out_dir, book=book, rho=rho
    )
    m, _, _ = load_m111(out_dir / "checkpoint_best.pt", device)
    offline = eval_token_metrics(m, tr_s, device)
    cl = eval_closed_loop("overfit/train", m, book, rho, chosen, device, env)
    # HANDOFF CF on failures / all
    cf_rows = []
    for r in chosen:
        print(f"  CF {r['traj_id']}", flush=True)
        cf_rows.append(handoff_counterfactual(env, m, book, rho, r, device, probe_every=100))
    write_csv(OUT / "OVERFIT_CLOSED_LOOP.csv", cl)
    write_csv(OUT / "OVERFIT_HANDOFF_CF.csv", cf_rows)
    n_ok = sum(r["p1_ok"] for r in cl)
    md = f"""# OVERFIT_TEST — P35

Generated: {utc()}

## Setup

- Roots: {len(chosen)} independent train P1
- Trajectories: {[r['traj_id'] for r in chosen]}
- n_motion: {[r['n_motion'] for r in chosen]}
- Epochs: {cfg['epochs']}, window={WINDOW}, max_rollout={MAX_ROLLOUT_STEPS}
- Encoder: {ENCODER_VERSION}

## Offline (teacher-forced)

- token_acc: {offline.get('token_acc')}
- motion_token_acc: {offline.get('motion_token_acc')}
- handoff_precision/recall: {offline.get('handoff_precision')} / {offline.get('handoff_recall')}
- best_epoch: {(result.get('best') or {}).get('epoch')}

## Closed-loop on same training roots

- P1 true correction success: **{n_ok}/{len(cl)}**
- Policy HANDOFF STRICT: {sum(r['pol_SUCCESS_NEW_handoff'] for r in cl)}/{len(cl)}
- Immediate HANDOFF shortcuts: {sum(r['p1_handoff_shortcut'] for r in cl)}
- Timeouts: {sum(r['timeout'] for r in cl)}
- Terminations: {dict(Counter(r['pol_termination'] for r in cl))}

## HANDOFF counterfactual diagnoses

{dict(Counter(r['diagnosis'] for r in cf_rows))}

## Gate

{"PROCEED to full SFT — closed-loop fit demonstrated on overfit set." if n_ok > 0 else "DIAGNOSE before scaling — offline may be high but closed-loop is zero; see CF diagnoses and first divergence."}
"""
    write_md(OUT / "OVERFIT_TEST.md", md)
    print(md, flush=True)
    del m
    return {"n_ok": n_ok, "n": len(cl), "offline": offline, "chosen": chosen, "cf": cf_rows}


# ---------------------------------------------------------------------------
# Phase 2–4 full SFT
# ---------------------------------------------------------------------------
def phase_full_sft(p1_rows, p0_rows, device, book, rho, env, *, do_scale: bool) -> None:
    train_p1 = [r for r in p1_rows if r["split"] == "train"]
    val_p1 = [r for r in p1_rows if r["split"] == "val"]
    test_p1 = [r for r in p1_rows if r["split"] == "test"]
    train_p0 = [r for r in p0_rows if r["split"] == "train"]
    val_p0 = [r for r in p0_rows if r["split"] == "val"]
    test_p0 = [r for r in p0_rows if r["split"] == "test"]

    curve, evals = [], []
    rng = np.random.default_rng(SEED)

    # D0 frozen baselines
    for tag, path in (("D0_P22", P22), ("D0_P28", P28), ("D0_P31", P31)):
        if not path.is_file():
            continue
        print(f"=== {tag} ===", flush=True)
        m, _, _ = load_m111(path, device)
        for pool_name, pool in (
            ("train_p1", train_p1[:15]),
            ("val_p1", val_p1),
            ("test_p1", test_p1),
            ("val_p0", val_p0),
        ):
            if not pool:
                continue
            er = eval_closed_loop(f"{tag}/{pool_name}", m, book, rho, pool, device, env)
            evals.extend(er)
            curve.append(
                {
                    "condition": tag,
                    "n_p1_train": 0,
                    "eval_split": pool_name,
                    "n_eval": len(er),
                    "p1_closed_loop": sum(r["p1_ok"] for r in er),
                    "p0_ok": sum(r["p0_ok"] for r in er),
                    "handoff_strict": sum(r["pol_SUCCESS_NEW_handoff"] for r in er),
                    "timeout_rate": sum(r["timeout"] for r in er) / max(1, len(er)),
                    "immediate_rate": sum(r["pol_immediate_handoff"] for r in er) / max(1, len(er)),
                }
            )
        del m

    def train_and_eval(name: str, p1_subset: list[dict], use_p0: bool, epochs: int = 25):
        nonlocal curve, evals
        print(f"=== train {name} p1={len(p1_subset)} p0={use_p0} ===", flush=True)
        if use_p0:
            tr_s = balance_root_level(p1_subset, train_p0, rng)
            va_s = balance_root_level(val_p1 or p1_subset[:2], val_p0 or train_p0[:5], rng)
        else:
            tr_s = samples_for(p1_subset)
            va_s = samples_for(val_p1) if val_p1 else samples_for(p1_subset[: max(1, len(p1_subset) // 5)])
        out_dir = OUT / "checkpoints" / name
        cfg = {
            "name": name,
            "device": str(device),
            "seed": SEED,
            "epochs": epochs,
            "batch_size": 8,
            "lr": 3e-4,
        }
        result = train_m111(
            train_samples=tr_s, val_samples=va_s, cfg=cfg, out_dir=out_dir, book=book, rho=rho
        )
        m, _, _ = load_m111(out_dir / "checkpoint_best.pt", device)
        off = eval_token_metrics(m, tr_s, device, max_batches=40)
        for pool_name, pool in (
            ("train_p1", p1_subset),
            ("val_p1", val_p1),
            ("test_p1", test_p1),
            ("train_p0", train_p0[:20] if use_p0 else []),
            ("val_p0", val_p0 if use_p0 else []),
            ("test_p0", test_p0 if use_p0 else []),
        ):
            if not pool:
                continue
            er = eval_closed_loop(f"{name}/{pool_name}", m, book, rho, pool, device, env)
            evals.extend(er)
            curve.append(
                {
                    "condition": name,
                    "n_p1_train": len(p1_subset),
                    "use_p0": int(use_p0),
                    "eval_split": pool_name,
                    "n_eval": len(er),
                    "p1_closed_loop": sum(r["p1_ok"] for r in er),
                    "p0_ok": sum(r["p0_ok"] for r in er),
                    "p0_overcontrol": sum(r["p0_overcontrol"] for r in er),
                    "handoff_strict": sum(r["pol_SUCCESS_NEW_handoff"] for r in er),
                    "timeout_rate": sum(r["timeout"] for r in er) / max(1, len(er)),
                    "immediate_rate": sum(r["pol_immediate_handoff"] for r in er) / max(1, len(er)),
                    "mean_n_motion": float(np.mean([r["pol_n_motion"] for r in er])),
                    "train_token_acc": off.get("token_acc"),
                    "train_handoff_recall": off.get("handoff_recall"),
                    "best_epoch": (result.get("best") or {}).get("epoch"),
                }
            )
        # CF on a few val failures
        fail_val = [
            r
            for r in evals
            if r["condition"] == f"{name}/val_p1" and not r["p1_ok"]
        ][:5]
        cf = []
        id_map = {r["traj_id"]: r for r in val_p1}
        for fr in fail_val:
            row = id_map.get(fr["traj_id"])
            if row:
                cf.append(handoff_counterfactual(env, m, book, rho, row, device, probe_every=100))
        if cf:
            write_csv(OUT / f"HANDOFF_CF_{name}.csv", cf)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # D1: all train P1
    train_and_eval("D1_p1_only", train_p1, use_p0=False, epochs=25)
    # D2: P1 + P0
    train_and_eval("D2_p1_p0", train_p1, use_p0=True, epochs=25)

    if do_scale:
        by_root: dict[str, list] = {}
        for r in train_p1:
            by_root.setdefault(r["physical_root"], []).append(r)
        roots = list(by_root.keys())
        for target in (25, 50, 100, len(train_p1)):
            if target > len(train_p1):
                continue
            chosen = []
            for root in roots:
                if len(chosen) >= target:
                    break
                chosen.extend(by_root[root])
            chosen = chosen[:target]
            if len(chosen) < 5:
                continue
            train_and_eval(f"S_p1n{len(chosen)}", chosen, use_p0=True, epochs=20)

    write_csv(OUT / "TRAINING_SUMMARY.csv", curve)
    write_csv(OUT / "CLOSED_LOOP_EVAL.csv", evals)
    write_csv(OUT / "LEARNING_CURVE.csv", curve)

    # HANDOFF diagnostics aggregate
    cf_files = list(OUT.glob("HANDOFF_CF_*.csv")) + (
        [OUT / "OVERFIT_HANDOFF_CF.csv"] if (OUT / "OVERFIT_HANDOFF_CF.csv").is_file() else []
    )
    all_cf = []
    for f in cf_files:
        all_cf.extend(csv.DictReader(f.open(encoding="utf-8")))
    write_md(
        OUT / "HANDOFF_DIAGNOSTICS.md",
        f"""# HANDOFF_DIAGNOSTICS — P35

Generated: {utc()}

## Method

Offline counterfactual: replay policy action prefixes from the IC snapshot, then
run immediate HANDOFF + 12s STRICT. Labels A–D as specified.

## Aggregate (available CF rows)

{dict(Counter(r.get('diagnosis') for r in all_cf)) if all_cf else 'none yet'}

See `HANDOFF_CF_*.csv` and `OVERFIT_HANDOFF_CF.csv`.
""",
    )


def write_final(overfit: dict | None) -> None:
    curve = list(csv.DictReader((OUT / "LEARNING_CURVE.csv").open(encoding="utf-8"))) if (OUT / "LEARNING_CURVE.csv").is_file() else []
    evals = list(csv.DictReader((OUT / "CLOSED_LOOP_EVAL.csv").open(encoding="utf-8"))) if (OUT / "CLOSED_LOOP_EVAL.csv").is_file() else []

    def best_row(cond_prefix, split):
        rows = [r for r in curve if str(r.get("condition", "")).startswith(cond_prefix) and r.get("eval_split") == split]
        if not rows:
            return None
        return max(rows, key=lambda r: int(float(r.get("p1_closed_loop") or 0)))

    d1_tr = best_row("D1", "train_p1")
    d1_te = best_row("D1", "test_p1")
    d2_tr = best_row("D2", "train_p1")
    d2_te = best_row("D2", "test_p1")
    d2_p0 = best_row("D2", "test_p0") or best_row("D2", "val_p0")
    d0 = best_row("D0_P22", "test_p1") or best_row("D0_P31", "test_p1")

    ov_ok = int(overfit.get("n_ok") or 0) if overfit else 0
    ov_n = int(overfit.get("n") or 0) if overfit else 0

    def g(row, k):
        return int(float(row.get(k) or 0)) if row else 0

    def gn(row):
        return int(float(row.get("n_eval") or 0)) if row else 0

    ready = (
        ov_ok > 0
        and g(d2_te, "p1_closed_loop") >= max(1, gn(d2_te) // 5)
        and (d2_p0 is None or g(d2_p0, "p0_ok") >= max(1, gn(d2_p0) // 2))
    )

    text = f"""# FINAL_REPORT — P35 Precision-only SFT (R1b)

Generated: {utc()}

## Answers

1. **Overfit small P1 set?** Offline token_acc={overfit.get('offline',{}).get('token_acc') if overfit else 'n/a'};
   closed-loop **{ov_ok}/{ov_n}** on overfit roots (see `OVERFIT_TEST.md`).

2. **Closed-loop on those roots?** {ov_ok}/{ov_n} true P1 correction (explicit HANDOFF + motion + STRICT).

3. **R1b vs P33 baseline?** D0 frozen (P22/P28/P31) test P1: see curve.
   D1/D2 use R1b labels. Compare D0 vs D2 test_p1 closed-loop in `LEARNING_CURVE.csv`.
   D0 test p1_ok={g(d0,'p1_closed_loop')}/{gn(d0)}; D2 test p1_ok={g(d2_te,'p1_closed_loop')}/{gn(d2_te)}.

4. **One policy for P1 + P0?** D2 test/val P0 ok={g(d2_p0,'p0_ok')}/{gn(d2_p0)};
   P1 test={g(d2_te,'p1_closed_loop')}/{gn(d2_te)}.

5. **Held-out generalization?** train P1 D2={g(d2_tr,'p1_closed_loop')}/{gn(d2_tr)} vs test={g(d2_te,'p1_closed_loop')}/{gn(d2_te)}.

6. **Remaining bottleneck?** See `HANDOFF_DIAGNOSTICS.md` and termination mixes in `CLOSED_LOOP_EVAL.csv`
   (timeout vs no-HANDOFF vs STRICT fail). Horizon raised to {MAX_ROLLOUT_STEPS} vs P33's 800.

7. **Ready for unified A/B/C/D SFT?** {"YES — Precision closed-loop learning evidenced; still isolate skill interference in a later unified run." if ready else "NO / PARTIAL — Precision not yet sufficiently closed-loop reliable for unified merge."}

## Artifacts

- `DATASET_AUDIT.md`
- `OVERFIT_TEST.md`
- `TRAINING_SUMMARY.csv` / `LEARNING_CURVE.csv`
- `CLOSED_LOOP_EVAL.csv`
- `HANDOFF_DIAGNOSTICS.md`
- `FINAL_REPORT.md`
"""
    write_md(OUT / "FINAL_REPORT.md", text)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-only", action="store_true")
    ap.add_argument("--skip-phase0", action="store_true")
    ap.add_argument("--overfit-only", action="store_true")
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-scale", action="store_true")
    ap.add_argument("--limit-p1", type=int, default=0)
    ap.add_argument("--limit-p0", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    print(f"=== P35 device={device} WINDOW={WINDOW} max_steps={MAX_ROLLOUT_STEPS} ===", flush=True)

    if not args.skip_phase0:
        phase0_audit_and_package(limit_p1=args.limit_p1, limit_p0=args.limit_p0)
    if args.phase0_only:
        return

    p1_rows, p0_rows = load_packaged()
    overfit = None
    if not args.skip_overfit:
        overfit = phase1_overfit(p1_rows, device, book, rho, env)
        atomic_json(
            OUT / "overfit_summary.json",
            {
                "n_ok": overfit["n_ok"],
                "n": overfit["n"],
                "offline": overfit.get("offline"),
            },
        )
    elif (OUT / "overfit_summary.json").is_file():
        overfit = json.loads((OUT / "overfit_summary.json").read_text(encoding="utf-8"))
        print(f"Loaded prior overfit summary: {overfit}", flush=True)
    if args.overfit_only:
        write_final(overfit)
        return

    # Gate: still run full SFT for evidence even if overfit=0, but note diagnosis
    if overfit and overfit["n_ok"] == 0:
        write_md(
            OUT / "GATE_NOTE.md",
            f"# GATE_NOTE\n\nOverfit closed-loop was 0/{overfit['n']}. "
            f"Proceeding to full SFT for complete evidence tables, but unified merge is blocked "
            f"until closed-loop fit is demonstrated.\n",
        )

    if not args.skip_full:
        phase_full_sft(
            p1_rows,
            p0_rows,
            device,
            book,
            rho,
            env,
            do_scale=not args.skip_scale and bool(overfit and overfit["n_ok"] > 0),
        )
    write_final(overfit)


if __name__ == "__main__":
    main()
