"""P45 Phase 3 — Controlled SFT: P44 baseline reuse vs P45-A vs P45-B."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paths import OUT, P44_CKPT, SEED  # noqa: E402
from models.m111_checkpoint import HANDOFF, LOGIT_K, load_m111, load_m111_book, save_m111  # noqa: E402
from training.m111_dataset import assert_no_target_in_inputs, collate_m111  # noqa: E402

SKILLS = ("A1", "B1", "C", "D")
CKPT_DIR = OUT / "checkpoints"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def skill_batch_loss(model, samples, device):
    if not samples:
        z = torch.zeros((), device=device)
        return z, {"n": 0}
    # weighted: repeat high-weight samples in batch construction instead
    obs, prev, tokens, residuals, mask, _w = collate_m111(samples, device)
    mask = mask & (tokens != HANDOFF)
    logits, pred, _ = model(obs, prev)
    per_tok, per_res = [], []
    for i in range(logits.shape[0]):
        active = mask[i]
        if not bool(active.any()):
            per_tok.append(logits.new_zeros(()))
            per_res.append(logits.new_zeros(()))
            continue
        per_tok.append(F.cross_entropy(logits[i][active], tokens[i][active]))
        per_res.append(((pred[i][active] - residuals[i][active]) ** 2).mean())
    return torch.stack(per_tok).mean() + torch.stack(per_res).mean(), {"n": int(mask.sum().item())}


def sample_batch(chs, n, rng):
    if not chs:
        return []
    # weight-aware (numpy RNG; `rng` kept for call-site compatibility)
    _ = rng
    weights = np.array([float(c.get("sample_weight") or 1.0) for c in chs], dtype=np.float64)
    weights = weights / weights.sum()
    idx = np.random.choice(len(chs), size=n, replace=True, p=weights)
    return [chs[int(i)] for i in idx]


@torch.no_grad()
def eval_acc(model, pools, device, rng, n_per=24):
    model.eval()
    out = {}
    for sk, chs in pools.items():
        if not chs:
            out[sk] = None
            continue
        samp = sample_batch(chs, min(n_per, len(chs)), rng)
        obs, prev, tokens, residuals, mask, _ = collate_m111(samp, device)
        mask = mask & (tokens != HANDOFF)
        logits, _, _ = model(obs, prev)
        correct = total = 0
        for i in range(logits.shape[0]):
            active = mask[i]
            if not bool(active.any()):
                continue
            correct += int((logits[i][active].argmax(-1) == tokens[i][active]).sum().item())
            total += int(active.sum().item())
        out[sk] = (correct / total) if total else None
    model.train()
    return out


def train_one(tag: str, pool_suffix: str, *, updates: int, batch_per_skill: int, lr: float) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + hash(tag) % 10007)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    pools = {}
    for sk in SKILLS:
        p = OUT / "dataset" / f"chunks_{sk}_{pool_suffix}.pkl"
        if not p.exists():
            raise FileNotFoundError(p)
        pools[sk] = pickle.loads(p.read_bytes())
        print(f"{tag} load {sk}: {len(pools[sk])}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(P44_CKPT, device)
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    with torch.no_grad():
        if hasattr(model, "logits") and model.logits.bias is not None:
            model.logits.bias[HANDOFF] = -20.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    history = []
    best = {"mean_acc": -1.0, "path": None}
    for u in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        for sk in SKILLS:
            batch = sample_batch(pools[sk], batch_per_skill, rng)
            loss, _ = skill_batch_loss(model, batch, device)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if u % 50 == 0 or u == 1 or u == updates:
            acc = eval_acc(model, pools, device, rng)
            mean_acc = float(np.mean([a for a in acc.values() if a is not None]))
            print(
                f"{tag} u={u} loss={loss.item():.4f} mean_acc={mean_acc:.3f} "
                + " ".join(f"{s}:{acc[s]:.3f}" for s in SKILLS if acc[s] is not None),
                flush=True,
            )
            path = CKPT_DIR / f"{tag}_u{u:04d}.pt"
            save_m111(path, model, book, epoch=u, rho=rho, extra={"tag": tag})
            history.append({"update": u, "loss": float(loss.item()), "acc": acc, "mean_acc": mean_acc})
            if mean_acc >= best["mean_acc"]:
                best = {"mean_acc": mean_acc, "path": str(path), "update": u}
                save_m111(CKPT_DIR / f"{tag}_best.pt", model, book, epoch=u, rho=rho, extra={"tag": tag, "best": True})
    return {"tag": tag, "best": best, "history": history, "updates": updates}


def run(*, updates: int = 600, batch_per_skill: int = 4, lr: float = 1e-4) -> None:
    # Ensure P45A/P45B datasets exist
    results = []
    results.append(train_one("P45A", "P45A", updates=updates, batch_per_skill=batch_per_skill, lr=lr))
    results.append(train_one("P45B", "P45B", updates=updates, batch_per_skill=batch_per_skill, lr=lr))
    (OUT / "TRAINING_COMPARISON.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# TRAINING_COMPARISON",
        "",
        f"Generated: {utc()}",
        f"Init: P44 best; updates={updates}; lr={lr}; batch/skill={batch_per_skill}; seed={SEED}",
        "",
    ]
    for r in results:
        lines.append(
            f"- **{r['tag']}**: best mean_acc={r['best']['mean_acc']:.4f} @ `{r['best']['path']}` (u={r['best'].get('update')})"
        )
    lines.append("")
    lines.append("Closed-loop SUCCESS_12S is the primary selection metric (see PAIRED_EVAL_REPORT).")
    (OUT / "TRAINING_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--batch-per-skill", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    run(updates=args.updates, batch_per_skill=args.batch_per_skill, lr=args.lr)


if __name__ == "__main__":
    main()
