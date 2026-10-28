"""P47 — Train from P45-B on ordinary-B mix; paired eval vs P44/P45B."""

from __future__ import annotations

import argparse
import csv
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
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paired_eval import (  # noqa: E402
    eval_ckpt,
    summarize,
    write_csv,
)
from experiments.precision.p47_b_ordinary_expansion.paths import (  # noqa: E402
    OUT,
    P44_EVAL,
    P45,
    P45B_CKPT,
    SEED,
)
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book, save_m111  # noqa: E402
from training.m111_dataset import collate_m111  # noqa: E402

SKILLS = ("A1", "B1", "C", "D")
CKPT_DIR = OUT / "checkpoints"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_batch(chs, n, rng):
    if not chs:
        return []
    _ = rng
    weights = np.array([float(c.get("sample_weight") or 1.0) for c in chs], dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    idx = np.random.choice(len(chs), size=n, replace=True, p=weights)
    return [chs[int(i)] for i in idx]


def skill_batch_loss(model, samples, device):
    if not samples:
        return torch.zeros((), device=device)
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
    return torch.stack(per_tok).mean() + torch.stack(per_res).mean()


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


def train(*, updates: int, batch_per_skill: int, lr: float, pool: str = "P47", tag: str = "P47") -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pools = {
        sk: pickle.loads((OUT / "dataset" / f"chunks_{sk}_{pool}.pkl").read_bytes()) for sk in SKILLS
    }
    for sk, chs in pools.items():
        print(f"{tag} load {sk}: {len(chs)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(P45B_CKPT, device)
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    with torch.no_grad():
        if hasattr(model, "logits") and model.logits.bias is not None:
            model.logits.bias[HANDOFF] = -20.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    best = {"mean_acc": -1.0, "b_acc": -1.0, "path": None}
    history = []
    for u in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        losses = [skill_batch_loss(model, sample_batch(pools[sk], batch_per_skill, rng), device) for sk in SKILLS]
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if u % 50 == 0 or u == 1 or u == updates:
            acc = eval_acc(model, pools, device, rng)
            mean_acc = float(np.mean([a for a in acc.values() if a is not None]))
            b_acc = float(acc.get("B1") or 0.0)
            print(
                f"{tag} u={u} loss={loss.item():.4f} mean_acc={mean_acc:.3f} "
                + " ".join(f"{s}:{acc[s]:.3f}" for s in SKILLS if acc[s] is not None),
                flush=True,
            )
            path = CKPT_DIR / f"{tag}_u{u:04d}.pt"
            save_m111(path, model, book, epoch=u, rho=rho, extra={"tag": tag})
            history.append({"update": u, "loss": float(loss.item()), "acc": acc, "mean_acc": mean_acc})
            score = mean_acc + 0.08 * b_acc
            if score >= best["mean_acc"] + 0.08 * float(best.get("b_acc") or 0):
                best = {"mean_acc": mean_acc, "b_acc": b_acc, "path": str(path), "update": u}
                save_m111(CKPT_DIR / f"{tag}_best.pt", model, book, epoch=u, rho=rho, extra={"tag": tag, "best": True})
    report = {"generated": utc(), "best": best, "updates": updates, "history": history, "pool": pool, "tag": tag}
    (OUT / f"TRAINING_REPORT_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / f"TRAINING_REPORT_{tag}.md").write_text(
        f"# TRAINING_REPORT — {tag}\n\nInit P45-B; pool={pool}; best mean={best['mean_acc']:.4f} "
        f"B={best['b_acc']:.4f} @ u={best.get('update')}\n",
        encoding="utf-8",
    )
    return best


def evaluate(*, tag: str = "P47") -> None:
    hold = [r for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")) if r.get("tag") == "holdout"]
    uniq = {(r["skill"], r["traj_id"]): r for r in hold}
    hold = list(uniq.values())
    jobs_by_key = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}
    all_rows = []
    for r in hold:
        drop = int(r.get("fail_category") == "DROP_BEFORE")
        all_rows.append(
            {
                "tag": "P44",
                "skill": r["skill"],
                "traj_id": r["traj_id"],
                "SUCCESS_12S": int(r.get("SUCCESS_12S") or 0),
                "SUCCESS_NEW": int(r.get("SUCCESS_NEW") or 0),
                "drop_before": drop,
                "handoff": int(not drop),
                "handoff_reason": r.get("handoff_reason") or ("DROP_BEFORE" if drop else "OK"),
                "fail_category": r.get("fail_category"),
            }
        )
    p45 = P45 / "PAIRED_EVAL_ROLLOUTS.csv"
    if p45.exists():
        all_rows.extend([r for r in csv.DictReader(p45.open(encoding="utf-8")) if r.get("tag") == "P45B"])
    ckpt = OUT / "checkpoints" / f"{tag}_best.pt"
    all_rows.extend(eval_ckpt(tag, ckpt, jobs_by_key, hold))
    write_csv(OUT / f"PAIRED_EVAL_ROLLOUTS_{tag}.csv", all_rows)
    summaries = {t: summarize(all_rows, t) for t in ("P44", "P45B", tag)}
    lines = [
        f"# PAIRED_EVAL_REPORT — {tag} ordinary B",
        "",
        f"Generated: {utc()}",
        "",
        f"| Skill | P44 | P45-B | {tag} | P44 DROP | {tag} DROP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sk in SKILLS:
        a, b, c = summaries["P44"][sk], summaries.get("P45B", {}).get(sk, {}), summaries[tag][sk]
        lines.append(
            f"| {sk} | {a['SUCCESS_12S']}/{a['n']} | {b.get('SUCCESS_12S', '?')}/{b.get('n', '?')} | "
            f"{c['SUCCESS_12S']}/{c['n']} | {a['DROP_BEFORE']} | {c['DROP_BEFORE']} |"
        )
    (OUT / f"PAIRED_EVAL_REPORT_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"PAIRED_EVAL_SUMMARY_{tag}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    # also refresh primary names for latest tag
    (OUT / "PAIRED_EVAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final = [
        f"# FINAL_SUMMARY — {tag} ordinary B expansion",
        "",
        f"Generated: {utc()}",
        "",
        "Ordinary non-hard, non-holdout B. Prefer Tree80 full-chain (recovery-start); "
        "table_free catch-start demos hurt A/B in P47-full mix.",
        "",
    ] + lines[4:]
    (OUT / "FINAL_SUMMARY.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--batch-per-skill", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--pool", default="P47T", help="dataset suffix, e.g. P47 or P47T")
    ap.add_argument("--tag", default="P47T")
    args = ap.parse_args()
    if not args.eval_only:
        train(
            updates=args.updates,
            batch_per_skill=args.batch_per_skill,
            lr=args.lr,
            pool=args.pool,
            tag=args.tag,
        )
    evaluate(tag=args.tag)


if __name__ == "__main__":
    main()
