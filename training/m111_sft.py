"""M111 supervised fine-tuning (112-way). Does not touch legacy K48 SFT."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.m111_checkpoint import (
    HANDOFF,
    LOGIT_K,
    MOTION_K,
    build_m111,
    load_m111_book,
    save_m111,
)
from training.m111_dataset import assert_no_target_in_inputs, collate_m111

ROOT = Path(__file__).resolve().parents[1]


def supervised_loss_m111(model, obs, prev, tokens, residuals, weight, mask):
    logits, pred, _act = model(obs, prev)
    if logits.shape[-1] != LOGIT_K:
        raise RuntimeError(f"token head is not {LOGIT_K}-wide (got {logits.shape[-1]})")
    assert_no_target_in_inputs(obs, prev, tokens)
    per_token = []
    per_resid = []
    for index in range(logits.shape[0]):
        active = mask[index]
        if not bool(active.any()):
            per_token.append(logits.new_zeros(()))
            per_resid.append(logits.new_zeros(()))
            continue
        per_token.append(F.cross_entropy(logits[index][active], tokens[index][active]))
        motion = active & (tokens[index] != HANDOFF)
        if bool(motion.any()):
            per_resid.append(((pred[index][motion] - residuals[index][motion]) ** 2).mean())
        else:
            per_resid.append(pred.new_zeros(()))
    token_vec = torch.stack(per_token)
    resid_vec = torch.stack(per_resid)
    w = weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    loss = (w * (token_vec + resid_vec)).mean()
    return loss, token_vec.mean().detach(), resid_vec.mean().detach()


@torch.no_grad()
def eval_token_metrics(model, samples: list[dict], device, *, max_batches: int | None = None) -> dict:
    model.eval()
    if not samples:
        return {"n_steps": 0}
    # Evaluate in batches of 8
    batch = 8
    total_correct = 0
    total = 0
    motion_correct = 0
    motion_total = 0
    hand_tp = hand_fp = hand_fn = hand_tn = 0
    pred_counts: Counter = Counter()
    skill_stats: dict[str, dict] = {}
    n_batches = 0
    for start in range(0, len(samples), batch):
        if max_batches is not None and n_batches >= max_batches:
            break
        chunk = samples[start : start + batch]
        obs, prev, tokens, residuals, mask, _w = collate_m111(chunk, device)
        logits, _pred, _act = model(obs, prev)
        pred = torch.argmax(logits, dim=-1)
        for i in range(len(chunk)):
            active = mask[i]
            t_true = tokens[i][active]
            t_pred = pred[i][active]
            total += int(active.sum())
            total_correct += int((t_true == t_pred).sum())
            motion = t_true != HANDOFF
            motion_total += int(motion.sum())
            motion_correct += int((t_true[motion] == t_pred[motion]).sum())
            for p in t_pred.tolist():
                pred_counts[int(p)] += 1
            # HANDOFF PR at step level
            for yt, yp in zip(t_true.tolist(), t_pred.tolist()):
                y_h = yt == HANDOFF
                p_h = yp == HANDOFF
                if y_h and p_h:
                    hand_tp += 1
                elif (not y_h) and p_h:
                    hand_fp += 1
                elif y_h and (not p_h):
                    hand_fn += 1
                else:
                    hand_tn += 1
            # Per-skill coarse tag from sample metadata
            sk = chunk[i].get("skill_group") or "unknown"
            st = skill_stats.setdefault(sk, {"correct": 0, "total": 0})
            st["total"] += int(active.sum())
            st["correct"] += int((t_true == t_pred).sum())
        n_batches += 1
    prec = hand_tp / max(1, hand_tp + hand_fp)
    rec = hand_tp / max(1, hand_tp + hand_fn)
    return {
        "n_steps": total,
        "token_acc": total_correct / max(1, total),
        "motion_token_acc": motion_correct / max(1, motion_total),
        "handoff_precision": prec,
        "handoff_recall": rec,
        "handoff_tp": hand_tp,
        "handoff_fp": hand_fp,
        "handoff_fn": hand_fn,
        "pred_token_hist": dict(pred_counts),
        "per_skill_acc": {k: v["correct"] / max(1, v["total"]) for k, v in skill_stats.items()},
        "n_batches": n_batches,
    }


def train_m111(
    *,
    train_samples: list[dict],
    val_samples: list[dict],
    cfg: dict[str, Any],
    out_dir: Path,
    book: np.ndarray,
    rho,
    init_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(cfg.get("seed", 20))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    init_path = init_checkpoint or cfg.get("init_checkpoint")
    if init_path:
        from models.m111_checkpoint import load_m111

        model, book_loaded, _saved = load_m111(Path(init_path), device)
        if not np.array_equal(np.asarray(book, np.float32), np.asarray(book_loaded, np.float32)):
            raise RuntimeError(f"init checkpoint book mismatch: {init_path}")
        print(f"M111_SFT init from {init_path}", flush=True)
    else:
        model = build_m111(book, device, rho=rho)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epochs", 20))
    rng = np.random.default_rng(seed)

    history: list[dict] = []
    best = {"val_token_acc": -1.0, "epoch": 0, "path": None}

    # Log effective weight / token frequency once
    w_hist = Counter()
    tok_hist: Counter = Counter()
    for s in train_samples:
        w_hist[round(float(s["sample_weight"]), 3)] += 1
        tok_hist.update(int(t) for t in s["tokens"].tolist())
    (out_dir / "effective_sample_weights.json").write_text(
        json.dumps(
            {
                "n_train_samples": len(train_samples),
                "n_val_samples": len(val_samples),
                "weight_round3_hist": dict(w_hist),
                "mean_weight": float(np.mean([s["sample_weight"] for s in train_samples])),
                "train_token_hist": {str(k): v for k, v in tok_hist.most_common()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_samples))
        total_loss = 0.0
        total_tok = 0.0
        total_res = 0.0
        seen = 0
        for start in range(0, len(train_samples), batch):
            idxs = order[start : start + batch]
            chunk = [train_samples[int(i)] for i in idxs]
            obs, prev, tokens, residuals, mask, weight = collate_m111(chunk, device)
            opt.zero_grad(set_to_none=True)
            loss, tok_l, res_l = supervised_loss_m111(
                model, obs, prev, tokens, residuals, weight, mask
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            total_tok += float(tok_l)
            total_res += float(res_l)
            seen += 1
        train_loss = total_loss / max(1, seen)

        val_metrics = eval_token_metrics(model, val_samples, device)
        # Also train-set probe (capped) for logging
        train_probe = eval_token_metrics(model, train_samples, device, max_batches=20)

        ckpt_path = out_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        save_m111(
            ckpt_path,
            model,
            book,
            epoch=epoch,
            rho=rho,
            extra={
                "train_loss": train_loss,
                "val_token_acc": val_metrics.get("token_acc"),
                "config_name": cfg.get("name"),
                "seed": seed,
            },
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_loss": total_tok / max(1, seen),
            "train_resid_loss": total_res / max(1, seen),
            "val_token_acc": val_metrics.get("token_acc"),
            "val_motion_token_acc": val_metrics.get("motion_token_acc"),
            "val_handoff_precision": val_metrics.get("handoff_precision"),
            "val_handoff_recall": val_metrics.get("handoff_recall"),
            "train_probe_token_acc": train_probe.get("token_acc"),
            "checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
        }
        history.append(row)
        print(
            f"M111_SFT {cfg.get('name')} epoch={epoch} loss={train_loss:.5f} "
            f"val_acc={row['val_token_acc']:.4f} motion={row['val_motion_token_acc']:.4f} "
            f"H_p={row['val_handoff_precision']:.3f} H_r={row['val_handoff_recall']:.3f}",
            flush=True,
        )
        # Selection: validation token accuracy (not train loss alone)
        if float(val_metrics.get("token_acc") or -1) >= float(best["val_token_acc"]):
            best = {
                "val_token_acc": float(val_metrics["token_acc"]),
                "epoch": epoch,
                "path": ckpt_path,
                "metrics": val_metrics,
            }
            save_m111(
                out_dir / "checkpoint_best.pt",
                model,
                book,
                epoch=epoch,
                rho=rho,
                extra={"selection": "val_token_acc", "val_metrics": {k: v for k, v in val_metrics.items() if k != "pred_token_hist"}},
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "best.json").write_text(
        json.dumps(
            {
                "epoch": best["epoch"],
                "val_token_acc": best["val_token_acc"],
                "path": str(Path(best["path"]).relative_to(ROOT)).replace("\\", "/") if best["path"] else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"history": history, "best": best, "out_dir": out_dir}


def smoke_train_tiny(
    samples: list[dict],
    book: np.ndarray,
    rho,
    device,
    *,
    steps: int = 30,
) -> dict:
    """Overfit a tiny subset; verify shapes, finite grads, loss drop, ckpt roundtrip."""
    from models.m111_checkpoint import load_m111, decode_m111_action

    tiny = samples[: min(4, len(samples))]
    model = build_m111(book, device, rho=rho)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        obs, prev, tokens, residuals, mask, weight = collate_m111(tiny, device)
        logits, pred, act = model(obs, prev)
        if logits.shape[-1] != LOGIT_K:
            raise RuntimeError("smoke: bad logit width")
        if logits.dim() != 3 or logits.shape[0] != len(tiny):
            raise RuntimeError(f"smoke: logits shape {tuple(logits.shape)}")
        loss, _, _ = supervised_loss_m111(model, obs, prev, tokens, residuals, weight, mask)
        if not torch.isfinite(loss):
            raise RuntimeError("smoke: non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError("smoke: non-finite or missing grads")
        opt.step()
        losses.append(float(loss.detach()))

    if losses[0] > 0.05 and losses[-1] >= losses[0]:
        raise RuntimeError(f"smoke: loss did not decrease {losses[0]} -> {losses[-1]}")

    tmp = ROOT / "results/diagnostics/p20_m111_sft_baseline/_smoke_ckpt.pt"
    save_m111(tmp, model, book, epoch=0, rho=rho, extra={"smoke": True})
    model2, book2, _ = load_m111(tmp, device)
    if not np.array_equal(book, book2):
        raise RuntimeError("smoke: book mismatch on reload")
    model2.eval()
    obs, prev, tokens, residuals, mask, weight = collate_m111(tiny[:1], device)
    with torch.no_grad():
        logits, resid, act = model2(obs, prev)
    tok = int(torch.argmax(logits[0, 0]).item())
    if tok != HANDOFF:
        _ = decode_m111_action(book2, tok, resid[0, 0].cpu().numpy(), rho=rho)
    return {"loss_start": losses[0], "loss_end": losses[-1], "steps": steps, "ckpt": str(tmp)}
