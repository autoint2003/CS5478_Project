"""P49 Phase 4–6: P49-L/M fine-tune from P47T + paired eval."""

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
from experiments.precision.p49_b_targeted_dagger.paths import (  # noqa: E402
    OUT,
    P44_EVAL,
    P45,
    P45B_DATA,
    P47T_CKPT,
    SEED,
)
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book, save_m111  # noqa: E402
from training.m111_dataset import collate_m111  # noqa: E402

SKILLS = ("A1", "B1", "C", "D")
CKPT_DIR = OUT / "checkpoints"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_mixed_pools(dagger_frac: float) -> dict[str, list]:
    """Replay = P45B A/C/D + P45B B; B batch mixes dagger_frac from dagger-only."""
    dagger = pickle.loads((OUT / "dataset" / "chunks_B1_dagger_only.pkl").read_bytes())
    pools = {}
    for sk in SKILLS:
        pools[sk] = pickle.loads((P45B_DATA / f"chunks_{sk}_P45B.pkl").read_bytes())
    # annotate dagger into B pool with relative mass via sample_weight scaling
    # Effective: when sampling B, with prob dagger_frac draw from dagger (renormalize weights)
    replay_b = pools["B1"]
    # set weights so expected mass matches dagger_frac
    # w_d / (W_r + W_d) = dagger_frac => W_d = dagger_frac/(1-dagger_frac) * W_r
    wr = sum(float(c.get("sample_weight") or 1.0) for c in replay_b) or 1.0
    if dagger and dagger_frac > 0:
        target_wd = dagger_frac / max(1e-6, 1.0 - dagger_frac) * wr
        cur = sum(float(c.get("sample_weight") or 1.0) for c in dagger) or 1.0
        scale = target_wd / cur
        dagger = [{**c, "sample_weight": float(c.get("sample_weight") or 1.0) * scale} for c in dagger]
    pools["B1"] = list(replay_b) + list(dagger)
    return pools


def sample_batch(chs, n, rng):
    if not chs:
        return []
    _ = rng
    w = np.array([float(c.get("sample_weight") or 1.0) for c in chs], dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    idx = np.random.choice(len(chs), size=n, replace=True, p=w)
    return [chs[int(i)] for i in idx]


def skill_loss(model, samples, device):
    if not samples:
        return torch.zeros((), device=device)
    obs, prev, tokens, residuals, mask, _ = collate_m111(samples, device)
    mask = mask & (tokens != HANDOFF)
    logits, pred, _ = model(obs, prev)
    losses = []
    for i in range(logits.shape[0]):
        active = mask[i]
        if not bool(active.any()):
            losses.append(logits.new_zeros(()))
            continue
        losses.append(
            F.cross_entropy(logits[i][active], tokens[i][active])
            + ((pred[i][active] - residuals[i][active]) ** 2).mean()
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def eval_acc(model, pools, device, rng):
    model.eval()
    out = {}
    for sk, chs in pools.items():
        if not chs:
            out[sk] = None
            continue
        samp = sample_batch(chs, min(24, len(chs)), rng)
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


def train_one(tag: str, dagger_frac: float, *, updates: int, batch: int, lr: float) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + hash(tag) % 10007)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pools = build_mixed_pools(dagger_frac)
    # measure actual B dagger mass
    b = pools["B1"]
    wd = sum(float(c.get("sample_weight") or 1.0) for c in b if c.get("source") == "p49_dagger")
    wt = sum(float(c.get("sample_weight") or 1.0) for c in b) or 1.0
    actual_frac = wd / wt
    for sk in SKILLS:
        print(f"{tag} load {sk}: {len(pools[sk])}", flush=True)
    print(f"{tag} target_dagger_frac={dagger_frac} actual_B_mass={actual_frac:.3f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(P47T_CKPT, device)
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    with torch.no_grad():
        if hasattr(model, "logits") and model.logits.bias is not None:
            model.logits.bias[HANDOFF] = -20.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    best = {"mean_acc": -1.0, "path": None}
    hist = []
    for u in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        loss = torch.stack(
            [skill_loss(model, sample_batch(pools[sk], batch, rng), device) for sk in SKILLS]
        ).mean()
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
            save_m111(path, model, book, epoch=u, rho=rho, extra={"tag": tag, "dagger_frac": dagger_frac})
            hist.append({"update": u, "loss": float(loss.item()), "acc": acc, "mean_acc": mean_acc})
            if mean_acc >= best["mean_acc"]:
                best = {"mean_acc": mean_acc, "path": str(path), "update": u, "actual_frac": actual_frac}
                save_m111(
                    CKPT_DIR / f"{tag}_best.pt",
                    model,
                    book,
                    epoch=u,
                    rho=rho,
                    extra={"tag": tag, "best": True, "dagger_frac": dagger_frac},
                )
    return {"tag": tag, "dagger_frac": dagger_frac, "actual_frac": actual_frac, "best": best, "history": hist}


def evaluate(tags: list[str]) -> None:
    hold = [r for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")) if r.get("tag") == "holdout"]
    uniq = {(r["skill"], r["traj_id"]): r for r in hold}
    hold = list(uniq.values())
    jobs = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}
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
                "handoff_reason": r.get("handoff_reason") or "",
                "fail_category": r.get("fail_category"),
            }
        )
    # reuse P47T if available
    p47t_csv = OUT.parent / "p47_b_ordinary_expansion/PAIRED_EVAL_ROLLOUTS_P47T.csv"
    if p47t_csv.exists():
        all_rows.extend([r for r in csv.DictReader(p47t_csv.open(encoding="utf-8")) if r.get("tag") == "P47T"])
    for tag in tags:
        ckpt = CKPT_DIR / f"{tag}_best.pt"
        if not ckpt.exists():
            print(f"SKIP {ckpt}", flush=True)
            continue
        all_rows.extend(eval_ckpt(tag, ckpt, jobs, hold))
    write_csv(OUT / "PAIRED_EVAL_ROLLOUTS.csv", all_rows)
    tags_all = ["P44", "P47T"] + tags
    summaries = {t: summarize(all_rows, t) for t in tags_all}
    lines = [
        "# PAIRED_EVAL_REPORT — P49",
        "",
        f"Generated: {utc()}",
        "",
        "| Skill | P44 | P47T | " + " | ".join(tags) + " |",
        "|---|" + "|".join(["---:"] * (2 + len(tags))) + "|",
    ]
    for sk in SKILLS:
        cells = []
        for t in tags_all:
            s = summaries.get(t, {}).get(sk, {})
            cells.append(f"{s.get('SUCCESS_12S', '?')}/{s.get('n', '?')}")
        lines.append(f"| {sk} | " + " | ".join(cells) + " |")
    # drops
    lines += ["", "## DROP_BEFORE", ""]
    for sk in SKILLS:
        cells = []
        for t in tags_all:
            s = summaries.get(t, {}).get(sk, {})
            cells.append(str(s.get("DROP_BEFORE", "?")))
        lines.append(f"- {sk}: " + " / ".join(f"{t}={c}" for t, c in zip(tags_all, cells)))
    # acceptance vs P47T
    lines += ["", "## Gates vs P47T", ""]
    p47 = summaries.get("P47T", {})
    for tag in tags:
        s = summaries.get(tag, {})
        if not s:
            continue
        ok = True
        for sk, floor in (("A1", 8), ("B1", 1), ("C", 3), ("D", 6)):
            # floors from P47T baseline counts
            base = p47.get(sk, {}).get("SUCCESS_12S", 0)
            cur = s.get(sk, {}).get("SUCCESS_12S", 0)
            if sk == "B1":
                tgt = 3
                msg = f"{tag} B {cur}/16 (target≥3, baseline={base})"
            else:
                tgt = base
                msg = f"{tag} {sk} {cur} vs baseline {base}"
            if sk == "B1":
                if cur < 1:
                    ok = False
                lines.append(f"- {msg}")
            else:
                if cur < tgt:
                    ok = False
                    lines.append(f"- WARN {msg}")
                else:
                    lines.append(f"- OK {msg}")
        if s.get("B1", {}).get("SUCCESS_12S", 0) >= 3 and ok:
            lines.append(f"- **{tag} meets soft B target without A/C/D soft-floor break**")
    (OUT / "PAIRED_EVAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "PAIRED_EVAL_SUMMARY.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summaries


def write_final(train_results, summaries) -> None:
    rec = json.loads((OUT / "ORACLE_RECOVERABILITY.json").read_text(encoding="utf-8"))
    # pick best: B max among those with A/C/D >= P47T
    p47 = summaries.get("P47T", {})
    best_tag = None
    best_b = -1
    for tag in ("P49L", "P49M"):
        s = summaries.get(tag)
        if not s:
            continue
        if s["A1"]["SUCCESS_12S"] < p47.get("A1", {}).get("SUCCESS_12S", 0):
            continue
        if s["C"]["SUCCESS_12S"] < p47.get("C", {}).get("SUCCESS_12S", 0):
            continue
        if s["D"]["SUCCESS_12S"] < p47.get("D", {}).get("SUCCESS_12S", 0):
            continue
        if s["B1"]["SUCCESS_12S"] > best_b:
            best_b = s["B1"]["SUCCESS_12S"]
            best_tag = tag
    if best_tag is None:
        # fall back highest B without hard zero classes
        for tag in ("P49L", "P49M"):
            s = summaries.get(tag)
            if not s:
                continue
            if min(s[sk]["SUCCESS_12S"] for sk in SKILLS) <= 0:
                continue
            if s["B1"]["SUCCESS_12S"] > best_b:
                best_b = s["B1"]["SUCCESS_12S"]
                best_tag = tag
    keep = best_tag or "P47T"
    lines = [
        "# FINAL_REPORT — P49 B targeted DAgger",
        "",
        f"Generated: {utc()}",
        "",
        "## Recoverability (Phase 2)",
        "",
        f"- Probes: {rec['n_probes']}; status `{rec['status_counts']}`",
        f"- Usable DAgger: {rec['usable_dagger_probes']}; chunks={rec['n_dagger_chunks']}",
        f"- By cut_t: `{rec['recoverability_by_cut_t']}`",
        f"- Intervention window hint: `{rec['intervention_window_hint']}`",
        "",
        "## Closed-loop",
        "",
        "| Class | P47T | P49-L | P49-M |",
        "|---|---:|---:|---:|",
    ]
    for sk, lab in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D", "D")):
        def cell(tag):
            s = summaries.get(tag, {}).get(sk, {})
            return f"{s.get('SUCCESS_12S', '?')}/{s.get('n', '?')}"

        lines.append(f"| {lab} | {cell('P47T')} | {cell('P49L')} | {cell('P49M')} |")
    lines += [
        "",
        f"**Selected:** `{keep}`",
        "",
        "## Answers",
        "",
        "1. Recoverable policy states: see ORACLE_RECOVERABILITY (precision/Tree80 from early cuts).",
        "2. Intervention window: see hint (prefer early cut_t before bilateral loss).",
        f"3. DROP_BEFORE: compare P47T vs selected in PAIRED_EVAL_REPORT.",
        f"4. B lift: P47T {summaries.get('P47T',{}).get('B1',{}).get('SUCCESS_12S','?')}/16 → "
        f"best {best_b if best_b>=0 else '?'}/16.",
        "5. A/C/D: see table; selection required no soft-floor break when possible.",
        "6. Next: if B still 1/16 with few usable dagger → earlier cuts / more families; "
        "if oracle mostly UNRECOVERABLE → intervention too late (Case C timing); "
        "if B up but A down → lower dagger frac.",
        "",
        f"Train configs: `{json.dumps(train_results, default=str)[:500]}...`",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "SFT_COMPARISON.md").write_text(
        "# SFT_COMPARISON\n\n"
        + "\n".join(
            f"- **{r['tag']}**: target_frac={r['dagger_frac']} actual_B_mass={r['actual_frac']:.3f} "
            f"best_acc={r['best']['mean_acc']:.4f} @ u={r['best'].get('update')}"
            for r in train_results
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    dagger = OUT / "dataset" / "chunks_B1_dagger_only.pkl"
    if not dagger.exists():
        raise FileNotFoundError(dagger)
    n = len(pickle.loads(dagger.read_bytes()))
    if n == 0:
        print("No dagger chunks — skip train; write failure FINAL_REPORT", flush=True)
        (OUT / "FINAL_REPORT.md").write_text(
            "# FINAL_REPORT — P49\n\nNo usable RECOVERABLE DAgger chunks. "
            "Stop before SFT; move intervention earlier (Case C timing).\n",
            encoding="utf-8",
        )
        return
    results = []
    if not args.eval_only:
        results.append(train_one("P49L", 0.10, updates=args.updates, batch=args.batch, lr=args.lr))
        results.append(train_one("P49M", 0.20, updates=args.updates, batch=args.batch, lr=args.lr))
        (OUT / "SFT_COMPARISON.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summaries = evaluate(["P49L", "P49M"])
    if not results and (OUT / "SFT_COMPARISON.json").exists():
        results = json.loads((OUT / "SFT_COMPARISON.json").read_text(encoding="utf-8"))
    write_final(results, summaries)


if __name__ == "__main__":
    main()
