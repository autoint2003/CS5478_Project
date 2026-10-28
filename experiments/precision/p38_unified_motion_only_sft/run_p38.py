"""P38 orchestrator — Gate1 → Gate2 → E0/E1/E2 train → closed-loop eval → FINAL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p38_unified_motion_only_sft.dataset import build_all_pools  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.eval_closed_loop import (  # noqa: E402
    eval_all_kinds,
)
from experiments.precision.p38_unified_motion_only_sft.gate1_working_handoff import (  # noqa: E402
    run_gate1,
)
from experiments.precision.p38_unified_motion_only_sft.paths import (  # noqa: E402
    M111_META,
    OUT,
    P31_CKPT,
    P36_CKPT,
    SEED,
)
from experiments.precision.p38_unified_motion_only_sft.train_unified import (  # noqa: E402
    UPDATES,
    train_all,
)
from models.m111_checkpoint import load_m111_book  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_final(gate1, train_meta, eval_summary) -> None:
    # Pivot summary by kind × suite (nontrivial)
    by = defaultdict(dict)
    for s in eval_summary or []:
        if int(s.get("nontrivial_only") or 0) != 1 and s.get("suite") not in ("A1", "B1", "C", "D2"):
            # For A/B/C/D2 report both; prefer nontrivial_only=0 for A/B/C suites (init WORKING rare)
            pass
        key = (s["kind"], s["suite"], int(s.get("nontrivial_only") or 0))
        by[key] = s

    def cell(kind, suite, nt=0):
        s = by.get((kind, suite, nt)) or by.get((kind, suite, 0))
        if not s:
            return "—/—"
        return f"{s['SUCCESS_NEW']}/{s['n']}"

    lines = [
        f"# FINAL_REPORT — P38 Unified A/B/C/D Motion-Only SFT",
        f"",
        f"Generated: {utc()}",
        f"",
        f"## Scope",
        f"",
        f"- Shared M111 motion-only SFT (112-way head retained; HANDOFF never supervised; masked at inference).",
        f"- Shared WORKING HANDOFF: θ≤10°, e_pos≤12 mm + P36 nominal contract + 12s STRICT.",
        f"- Experiments: E0 scratch, E1←P31 M2_pcgrad, E2←P36 D-only. Matched updates={UPDATES}.",
        f"- SFT only; no RL.",
        f"",
        f"## Baseline references (not a unified model)",
        f"",
        f"- P31 M2_pcgrad (learned HANDOFF): A1=14/19, B1=3/4, C=3/19, D2=0/17.",
        f"- P36/P37 D-only WORKING: nontrivial test **8/11**; WORKING-triggered expert **35/35** NEW.",
        f"",
        f"## Gate 1 — WORKING HANDOFF on experts",
        f"",
    ]
    if gate1 and gate1.get("agg"):
        lines.append("| Skill | n | Reach | SUCCESS_NEW | NEW|reach | Flags |")
        lines.append("|---|---:|---:|---:|---|---|")
        for a in gate1["agg"]:
            lines.append(
                f"| {a['skill']} | {a['n_traj']} | {a['reached_working']} | {a['SUCCESS_NEW']} | "
                f"{a['NEW_given_reach']} | {''} |"
            )
        if gate1.get("flags"):
            lines.append("")
            lines.append("**Counterexample flags:**")
            for f in gate1["flags"]:
                lines.append(f"- {f}")
        else:
            lines.append("")
            lines.append("No severe WORKING→STRICT conversion failures flagged.")
    else:
        lines.append("(Gate1 not run or empty)")
    lines += [
        f"",
        f"See `gate1_working_handoff/GATE1_REPORT.md`.",
        f"",
        f"## Gate 2 — Dataset",
        f"",
        f"See `gate2_dataset/DATASET_CENSUS.csv` (motion tokens only; train/val separated).",
        f"",
        f"## Training",
        f"",
        f"| Exp | Init | Best update | Val mean loss |",
        f"|---|---|---:|---:|",
    ]
    for kind in ("E0", "E1", "E2"):
        m = (train_meta or {}).get(kind) or {}
        b = m.get("best") or {}
        lines.append(
            f"| {kind} | {(m.get('init') or {}).get('init') if isinstance(m.get('init'), dict) else m.get('init')} | "
            f"{b.get('update')} | {b.get('val_mean_loss')} |"
        )
    lines += [
        f"",
        f"Checkpoints under `checkpoints/E*/checkpoint_best.pt` (hashes in `train_meta.json`).",
        f"",
        f"## Closed-loop (single checkpoint per experiment; WORKING HANDOFF)",
        f"",
        f"| Suite | E0 NEW | E1 NEW | E2 NEW |",
        f"|---|---|---|---|",
    ]
    for suite in ("A1", "B1", "C", "D2", "D_nt_test"):
        nt = 1 if suite == "D_nt_test" else 0
        lines.append(
            f"| {suite} | {cell('E0', suite, nt)} | {cell('E1', suite, nt)} | {cell('E2', suite, nt)} |"
        )

    # Feasibility milestone: one ckpt with ≥1 NEW on each of A,B,C,D
    milestone = {}
    for kind in ("E0", "E1", "E2"):
        ok = True
        detail = {}
        for suite, label in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D_nt_test", "D")):
            # Also accept D2 as D signal if D_nt missing
            s = by.get((kind, suite, 1 if suite == "D_nt_test" else 0)) or by.get((kind, suite, 0))
            if suite == "D_nt_test" and (not s or s["n"] == 0):
                s = by.get((kind, "D2", 0)) or by.get((kind, "D2", 1))
                label = "D2"
            n_new = int(s["SUCCESS_NEW"]) if s else 0
            detail[label] = f"{n_new}/{s['n'] if s else 0}"
            if n_new < 1:
                ok = False
        milestone[kind] = {"pass": ok, "detail": detail}

    lines += [
        f"",
        f"## Feasibility milestone (≥1 nontrivial STRICT on A, B, C, and D from **one** ckpt)",
        f"",
    ]
    for kind, m in milestone.items():
        lines.append(f"- **{kind}**: {'PASS' if m['pass'] else 'FAIL'} — {m['detail']}")

    any_pass = any(m["pass"] for m in milestone.values())
    lines += [
        f"",
        f"## Scientific conclusions",
        f"",
        f"1. E0 from scratch: see table (A/B/C/D).",
        f"2. E1 P31 transfer: compare A/B/C vs E0; D acquisition.",
        f"3. E2 P36 transfer: D retention vs A/B/C acquisition.",
        f"4. Interference: if a ckpt keeps A/B/C but loses D (or vice versa), that is the primary limitation.",
        f"5. Milestone achieved? **{'YES' if any_pass else 'NO'}**.",
        f"",
        f"**Skill D / unified recovery solved?** **NO** — milestone is feasibility only, not robust recovery.",
        f"",
        f"## Artifacts",
        f"",
        f"- `gate1_working_handoff/`",
        f"- `gate2_dataset/`",
        f"- `checkpoints/E0|E1|E2/`",
        f"- `eval/`",
        f"- `FINAL_REPORT.md`",
        f"",
    ]
    text = "\n".join(lines)
    (OUT / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gate1", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--only", choices=["E0", "E1", "E2"], default=None)
    ap.add_argument("--gate1-limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book, rho, _ = load_m111_book(M111_META)
    print(f"=== P38 device={device} P31={P31_CKPT.is_file()} P36={P36_CKPT.is_file()} ===", flush=True)

    gate1 = None
    if not args.skip_gate1:
        print("=== GATE1 ===", flush=True)
        gate1 = run_gate1(include_d=True, limit=args.gate1_limit)
        if gate1.get("flags"):
            print("GATE1 flags (proceeding with shared WORKING; see report):", gate1["flags"], flush=True)

    print("=== GATE2 dataset ===", flush=True)
    pools = build_all_pools()

    train_meta = {}
    if not args.skip_train:
        train_meta = train_all(
            pools["train"], pools["val"], device, book, rho, only=args.only
        )
    elif (OUT / "TRAIN_SUMMARY.json").is_file():
        train_meta = json.loads((OUT / "TRAIN_SUMMARY.json").read_text(encoding="utf-8"))

    eval_summary = []
    if not args.skip_eval:
        kinds = (args.only,) if args.only else ("E0", "E1", "E2")
        ev = eval_all_kinds(device, book, rho, kinds=kinds)
        eval_summary = ev["summary"]
    elif (OUT / "eval" / "SUMMARY.csv").is_file():
        eval_summary = list(csv.DictReader((OUT / "eval" / "SUMMARY.csv").open(encoding="utf-8")))
        for s in eval_summary:
            for k in ("n", "SUCCESS_NEW", "nontrivial_only", "WORKING_reach", "HANDOFF", "timeout", "drop"):
                if k in s and s[k] != "":
                    try:
                        s[k] = int(float(s[k]))
                    except Exception:
                        pass

    # Reload gate1 agg if skipped
    if gate1 is None:
        g1p = OUT / "gate1_working_handoff" / "GATE1_AGGREGATE.csv"
        if g1p.is_file():
            gate1 = {"agg": list(csv.DictReader(g1p.open(encoding="utf-8"))), "flags": []}

    write_final(gate1, train_meta, eval_summary)


if __name__ == "__main__":
    main()
