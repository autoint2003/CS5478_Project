"""E0/E1/E2 — skill-balanced motion-only SFT (HANDOFF never supervised / never executed)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.precision.p38_unified_motion_only_sft.paths import (
    M111_META,
    OUT,
    P31_CKPT,
    P36_CKPT,
    SEED,
)
from models.m111_checkpoint import (
    HANDOFF,
    LOGIT_K,
    MOTION_K,
    build_m111,
    load_m111,
    load_m111_book,
    save_m111,
)
from training.m111_dataset import assert_no_target_in_inputs, collate_m111

SKILLS = ("A1", "B1", "C", "D")
# Matched budget across E0/E1/E2
UPDATES = 800
BATCH_PER_SKILL = 4
LR = 3e-4
EVAL_EVERY = 50
CKPT_DIR = OUT / "checkpoints"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skill_batch_loss(model, samples: list[dict], device) -> tuple[torch.Tensor, dict]:
    """Mean CE+resid over this skill's batch (motion tokens only via loss_mask)."""
    if not samples:
        z = torch.zeros((), device=device)
        return z, {"token": 0.0, "resid": 0.0, "n": 0}
    obs, prev, tokens, residuals, mask, _w = collate_m111(samples, device)
    # Force HANDOFF out of mask
    mask = mask & (tokens != HANDOFF)
    logits, pred, _ = model(obs, prev)
    if logits.shape[-1] != LOGIT_K:
        raise RuntimeError(f"logits {logits.shape[-1]} != {LOGIT_K}")
    assert_no_target_in_inputs(obs, prev, tokens)
    per_tok = []
    per_res = []
    for i in range(logits.shape[0]):
        active = mask[i]
        if not bool(active.any()):
            per_tok.append(logits.new_zeros(()))
            per_res.append(logits.new_zeros(()))
            continue
        per_tok.append(F.cross_entropy(logits[i][active], tokens[i][active]))
        per_res.append(((pred[i][active] - residuals[i][active]) ** 2).mean())
    token_loss = torch.stack(per_tok).mean()
    resid_loss = torch.stack(per_res).mean()
    loss = token_loss + resid_loss
    return loss, {
        "token": float(token_loss.detach()),
        "resid": float(resid_loss.detach()),
        "n": len(samples),
    }


def balanced_step(model, opt, pools: dict[str, list], device, rng: random.Random) -> dict:
    """L = (L_A1 + L_B1 + L_C + L_D) / 4 with equal batch draws per skill."""
    losses = []
    stats = {}
    for sk in SKILLS:
        pool = pools[sk]
        if not pool:
            raise RuntimeError(f"empty pool {sk}")
        batch = [pool[rng.randrange(len(pool))] for _ in range(BATCH_PER_SKILL)]
        loss_sk, st = skill_batch_loss(model, batch, device)
        losses.append(loss_sk)
        stats[sk] = st
    total = torch.stack(losses).mean()
    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(total.detach()),
        "per_skill": {sk: float(losses[i].detach()) for i, sk in enumerate(SKILLS)},
        "stats": stats,
    }


@torch.no_grad()
def val_skill_loss(model, pools: dict[str, list], device, *, max_batches: int = 20) -> dict:
    model.eval()
    out = {}
    for sk in SKILLS:
        pool = pools.get(sk) or []
        if not pool:
            out[sk] = {"loss": None, "token_acc": None, "n": 0}
            continue
        losses = []
        correct = total = 0
        for b in range(max_batches):
            start = (b * BATCH_PER_SKILL) % max(1, len(pool))
            batch = [pool[(start + i) % len(pool)] for i in range(BATCH_PER_SKILL)]
            loss, _ = skill_batch_loss(model, batch, device)
            losses.append(float(loss))
            obs, prev, tokens, _r, mask, _w = collate_m111(batch, device)
            mask = mask & (tokens != HANDOFF)
            logits, _, _ = model(obs, prev)
            pred = torch.argmax(logits, dim=-1)
            for i in range(len(batch)):
                active = mask[i]
                if not bool(active.any()):
                    continue
                total += int(active.sum())
                correct += int((pred[i][active] == tokens[i][active]).sum())
        out[sk] = {
            "loss": float(np.mean(losses)) if losses else None,
            "token_acc": (correct / total) if total else None,
            "n": total,
        }
    model.train()
    return out


def init_model(kind: str, book, rho, device):
    """kind ∈ {E0, E1, E2}."""
    info: dict[str, Any] = {"kind": kind}
    if kind == "E0":
        model = build_m111(book, device, rho=rho)
        info["init"] = "random_build_m111"
        info["source_ckpt"] = None
    elif kind == "E1":
        if not P31_CKPT.is_file():
            raise FileNotFoundError(P31_CKPT)
        model, _, saved = load_m111(P31_CKPT, device)
        info["init"] = "P31_M2_pcgrad"
        info["source_ckpt"] = str(P31_CKPT)
        info["source_sha256"] = sha256_file(P31_CKPT)
        # HANDOFF logit retained but never supervised; zero its bias to discourage
        with torch.no_grad():
            if hasattr(model, "logits") and model.logits.bias is not None:
                model.logits.bias[HANDOFF] = -20.0
        info["handoff_bias_suppressed"] = True
        info["saved_keys"] = list(saved.keys())[:12] if isinstance(saved, dict) else None
    elif kind == "E2":
        if not P36_CKPT.is_file():
            raise FileNotFoundError(P36_CKPT)
        model, _, saved = load_m111(P36_CKPT, device)
        info["init"] = "P36_D_motion_ep10"
        info["source_ckpt"] = str(P36_CKPT)
        info["source_sha256"] = sha256_file(P36_CKPT)
        with torch.no_grad():
            if hasattr(model, "logits") and model.logits.bias is not None:
                model.logits.bias[HANDOFF] = -20.0
        info["handoff_bias_suppressed"] = True
    else:
        raise ValueError(kind)
    # Architecture sanity
    info["logits_out"] = int(model.logits.out_features)
    info["n_motion"] = int(model.n_motion)
    info["handoff_index"] = int(model.handoff_index)
    info["obs_dim"] = int(getattr(model, "obs_dim", -1)) if hasattr(model, "obs_dim") else None
    return model, info


def train_experiment(
    kind: str,
    train_pools: dict[str, list],
    val_pools: dict[str, list],
    *,
    device,
    book,
    rho,
    updates: int = UPDATES,
    seed: int = SEED,
) -> dict:
    rng = random.Random(seed + hash(kind) % 10007)
    torch.manual_seed(seed + (0 if kind == "E0" else 1 if kind == "E1" else 2))
    np.random.seed(seed + (0 if kind == "E0" else 1 if kind == "E1" else 2))

    model, init_info = init_model(kind, book, rho, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    out_dir = CKPT_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "init.json").write_text(json.dumps(init_info, indent=2, default=str), encoding="utf-8")

    history = []
    best = {"val_mean_loss": 1e9, "update": 0, "path": None}
    model.train()
    for upd in range(1, updates + 1):
        step = balanced_step(model, opt, train_pools, device, rng)
        row = {"update": upd, "train_loss": step["loss"], **{f"train_{k}": v for k, v in step["per_skill"].items()}}
        if upd % EVAL_EVERY == 0 or upd == updates:
            v = val_skill_loss(model, val_pools, device)
            losses = [v[sk]["loss"] for sk in SKILLS if v[sk]["loss"] is not None]
            mean_val = float(np.mean(losses)) if losses else 1e9
            row["val_mean_loss"] = mean_val
            for sk in SKILLS:
                row[f"val_loss_{sk}"] = v[sk]["loss"]
                row[f"val_acc_{sk}"] = v[sk]["token_acc"]
            path = out_dir / f"checkpoint_update_{upd:04d}.pt"
            save_m111(
                path,
                model,
                book,
                epoch=upd,
                rho=rho,
                extra={"kind": kind, "update": upd, "val_mean_loss": mean_val, "p38": True},
            )
            if mean_val < best["val_mean_loss"]:
                best = {"val_mean_loss": mean_val, "update": upd, "path": str(path)}
                save_m111(
                    out_dir / "checkpoint_best.pt",
                    model,
                    book,
                    epoch=upd,
                    rho=rho,
                    extra={"kind": kind, "update": upd, "val_mean_loss": mean_val, "selection": "val_mean_loss"},
                )
            print(
                f"  [{kind}] upd={upd}/{updates} train={step['loss']:.4f} val={mean_val:.4f} "
                + " ".join(f"{sk}={v[sk]['token_acc']:.3f}" if v[sk]['token_acc'] is not None else f"{sk}=NA" for sk in SKILLS),
                flush=True,
            )
        history.append(row)

    # curves
    import csv

    with (out_dir / "LEARNING_CURVE.csv").open("w", encoding="utf-8", newline="") as f:
        keys: list[str] = []
        seen: set[str] = set()
        for r in history:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in history:
            w.writerow(r)
    meta = {
        "kind": kind,
        "updates": updates,
        "batch_per_skill": BATCH_PER_SKILL,
        "lr": LR,
        "seed": seed,
        "best": best,
        "init": init_info,
        "best_sha256": sha256_file(Path(best["path"])) if best["path"] else None,
        "utc": utc(),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"=== {kind} done best_upd={best['update']} val={best['val_mean_loss']:.4f} ===", flush=True)
    return meta


def train_all(train_pools, val_pools, device, book, rho, *, only: str | None = None):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    kinds = [only] if only else ["E0", "E1", "E2"]
    results = {}
    # Merge prior completed runs when resuming a subset
    if (OUT / "TRAIN_SUMMARY.json").is_file():
        try:
            results = json.loads((OUT / "TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
        except Exception:
            results = {}
    for kind in kinds:
        meta_path = CKPT_DIR / kind / "train_meta.json"
        if meta_path.is_file() and only is None:
            print(f"=== Skipping {kind} (train_meta exists) ===", flush=True)
            results[kind] = json.loads(meta_path.read_text(encoding="utf-8"))
            continue
        print(f"=== Training {kind} updates={UPDATES} ===", flush=True)
        results[kind] = train_experiment(kind, train_pools, val_pools, device=device, book=book, rho=rho)
    (OUT / "TRAIN_SUMMARY.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results
