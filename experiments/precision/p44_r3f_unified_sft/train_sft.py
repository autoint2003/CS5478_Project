"""Balanced A/B/C/D motion-only SFT baseline under R3F-packaged data."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p38_unified_motion_only_sft.paths import M111_META, P31_CKPT  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.paths import OUT, SEED  # noqa: E402
from models.m111_checkpoint import (  # noqa: E402
    HANDOFF,
    LOGIT_K,
    build_m111,
    load_m111,
    load_m111_book,
    save_m111,
)
from training.m111_dataset import assert_no_target_in_inputs, collate_m111  # noqa: E402

SKILLS = ("A1", "B1", "C", "D")
DATA = OUT / "dataset"
CKPT_DIR = OUT / "checkpoints"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def skill_batch_loss(model, samples, device):
    if not samples:
        z = torch.zeros((), device=device)
        return z, {"token": 0.0, "resid": 0.0, "n": 0}
    obs, prev, tokens, residuals, mask, _w = collate_m111(samples, device)
    mask = mask & (tokens != HANDOFF)
    logits, pred, _ = model(obs, prev)
    if logits.shape[-1] != LOGIT_K:
        raise RuntimeError(f"logits {logits.shape[-1]} != {LOGIT_K}")
    assert_no_target_in_inputs(obs, prev, tokens)
    per_tok, per_res = [], []
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
    return token_loss + resid_loss, {
        "token": float(token_loss.item()),
        "resid": float(resid_loss.item()),
        "n": int(mask.sum().item()),
    }


@torch.no_grad()
def eval_pools(model, pools, device, n_per=32):
    model.eval()
    out = {}
    for sk, chs in pools.items():
        if not chs:
            out[sk] = {"loss": None, "acc": None}
            continue
        samp = random.sample(chs, min(n_per, len(chs)))
        obs, prev, tokens, residuals, mask, _ = collate_m111(samp, device)
        mask = mask & (tokens != HANDOFF)
        logits, pred, _ = model(obs, prev)
        losses, correct, total = [], 0, 0
        for i in range(logits.shape[0]):
            active = mask[i]
            if not bool(active.any()):
                continue
            losses.append(float(F.cross_entropy(logits[i][active], tokens[i][active]).item()))
            pred_t = logits[i][active].argmax(-1)
            correct += int((pred_t == tokens[i][active]).sum().item())
            total += int(active.sum().item())
        out[sk] = {
            "loss": float(np.mean(losses)) if losses else None,
            "acc": (correct / total) if total else None,
            "n_tok": total,
        }
    model.train()
    return out


def run(*, updates: int = 400, batch_per_skill: int = 4, lr: float = 3e-4) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    pools = {}
    for sk in SKILLS:
        p = DATA / f"chunks_{sk}.pkl"
        pools[sk] = pickle.loads(p.read_bytes()) if p.exists() else []
        print(f"load {sk}: {len(pools[sk])} chunks", flush=True)
        if not pools[sk]:
            raise RuntimeError(f"empty chunk pool for {sk}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book, rho, _ = load_m111_book(M111_META)
    if P31_CKPT.exists():
        model, book, saved = load_m111(P31_CKPT, device)
        with torch.no_grad():
            if hasattr(model, "logits") and model.logits.bias is not None:
                model.logits.bias[HANDOFF] = -20.0
        print(f"init from {P31_CKPT}", flush=True)
    else:
        model = build_m111(book, device, rho=rho)
        print("init fresh from meta", flush=True)
        saved = {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    best = {"mean_acc": -1.0, "path": None}
    model.train()
    for u in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        stats = {}
        for sk in SKILLS:
            chs = pools[sk]
            batch = [chs[random.randrange(len(chs))] for _ in range(batch_per_skill)]
            loss_sk, st = skill_batch_loss(model, batch, device)
            losses.append(loss_sk)
            stats[sk] = st
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if u % 50 == 0 or u == 1 or u == updates:
            ev = eval_pools(model, pools, device)
            mean_acc = float(np.mean([ev[s]["acc"] for s in SKILLS if ev[s]["acc"] is not None]))
            row = {
                "update": u,
                "train_loss": float(loss.item()),
                "per_skill_train": stats,
                "eval": ev,
                "mean_acc": mean_acc,
            }
            history.append(row)
            print(
                f"u={u} loss={loss.item():.4f} mean_acc={mean_acc:.3f} "
                + " ".join(f"{s}:{ev[s]['acc']:.3f}" for s in SKILLS if ev[s]["acc"] is not None),
                flush=True,
            )
            ckpt = CKPT_DIR / f"p44_u{u:04d}.pt"
            save_m111(ckpt, model, book, epoch=u, rho=rho, extra={"p44": True})
            if mean_acc >= best["mean_acc"]:
                best = {"mean_acc": mean_acc, "path": str(ckpt), "update": u}
                save_m111(CKPT_DIR / "p44_best.pt", model, book, epoch=u, rho=rho, extra={"p44_best": True})

    report = {
        "utc": utc(),
        "updates": updates,
        "batch_per_skill": batch_per_skill,
        "lr": lr,
        "seed": SEED,
        "best": best,
        "history": history,
        "pool_sizes": {s: len(pools[s]) for s in SKILLS},
    }
    (OUT / "SFT_TRAINING_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# SFT_TRAINING_REPORT",
        "",
        f"Generated: {utc()}",
        f"Updates: {updates}, batch/skill: {batch_per_skill}, lr: {lr}, seed: {SEED}",
        f"Best mean token acc: {best['mean_acc']:.4f} @ {best.get('path')}",
        "",
        "## Pool sizes",
        "",
        *(f"- {s}: {len(pools[s])} chunks" for s in SKILLS),
        "",
        "## Final eval acc",
        "",
    ]
    if history:
        ev = history[-1]["eval"]
        for s in SKILLS:
            lines.append(f"- {s}: acc={ev[s]['acc']} loss={ev[s]['loss']}")
    (OUT / "SFT_TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--batch-per-skill", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    run(updates=args.updates, batch_per_skill=args.batch_per_skill, lr=args.lr)


if __name__ == "__main__":
    main()
