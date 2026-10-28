"""Compose P33 FINAL_REPORT.md from Phase A/B/C artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/diagnostics/p33_precision_learnability"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    a = json.loads((OUT / "phase_a_summary.json").read_text(encoding="utf-8")) if (OUT / "phase_a_summary.json").is_file() else {}
    g = json.loads((OUT / "GENERATED_DATASET.json").read_text(encoding="utf-8")) if (OUT / "GENERATED_DATASET.json").is_file() else {}
    curve = list(csv.DictReader((OUT / "SFT_LEARNING_CURVE.csv").open(encoding="utf-8"))) if (OUT / "SFT_LEARNING_CURVE.csv").is_file() else []
    ev = list(csv.DictReader((OUT / "CLOSED_LOOP_EVAL.csv").open(encoding="utf-8"))) if (OUT / "CLOSED_LOOP_EVAL.csv").is_file() else []

    p1_pre = int(a.get("p1_train_roots") or 0)
    p1_gen = int(g.get("p1_total") or 0)
    train_n = int(g.get("p1_train") or 0)
    val_n = int(g.get("p1_val") or 0)
    test_n = int(g.get("p1_test") or 0)

    def best_p1(split: str):
        rows = [r for r in curve if r.get("eval_split") == split and str(r.get("condition", "")).startswith("C_P1")]
        if not rows:
            return None
        return max(rows, key=lambda r: int(float(r.get("p1_closed_loop") or 0)))

    btr, bva, bte = best_p1("train"), best_p1("val"), best_p1("test")
    c0 = [r for r in curve if str(r.get("condition", "")).startswith("C0_")]

    # Fit vs generalize
    train_ok = int(float(btr["p1_closed_loop"])) if btr else 0
    train_n_eval = int(float(btr["n_eval"])) if btr else 0
    val_ok = int(float(bva["p1_closed_loop"])) if bva else 0
    val_n_eval = int(float(bva["n_eval"])) if bva else 0

    ready = False
    if bva and val_n_eval > 0 and val_ok / val_n_eval >= 0.5 and train_ok > 0:
        ready = False  # still need more scale before unified SFT
    ready_txt = (
        "NOT ready for unified A/B/C/D integration — Precision-only learnability is still under-powered "
        f"(train P1 closed-loop {train_ok}/{train_n_eval}, held-out {val_ok}/{val_n_eval}; "
        f"dataset stage only reached {p1_gen} independent P1, far below 100–200)."
    )

    regimes = Counter()
    if (OUT / "GENERATED_P1.csv").is_file():
        for r in csv.DictReader((OUT / "GENERATED_P1.csv").open(encoding="utf-8")):
            regimes[(r.get("family"), r.get("target_dx_mm"), r.get("target_dz_mm"))] += 1

    md = f"""# FINAL_REPORT — P33 Precision Learnability

Generated: {_utc()}

## Scope

Paused unified A/B/C/D SFT and all RL. Sequential Phase A → B → C.
M111 / STRICT / Oracle acceptance criteria unchanged (one Oracle **NameError** bugfix for `FAIL_SLIP_WRONG_DIR` import — no threshold change).

## Answers

1. **Independent verified P1 training ICs before expansion?** **{p1_pre}** unique roots ({a.get('p1_train_rows')} traj rows). See `DATA_AUDIT.md`.

2. **How many generated, acceptance rate?** **{p1_gen}** independent P1 ICs generated & validated (train/val/test = {train_n}/{val_n}/{test_n}).  
   Wall-clock bottleneck: production Oracle tapes ≈1.7–2.1k actions → ~4 P1 / 15 min. Targets 50/100/200 **not** reached. Details: `GENERATION_REPORT.md`.

3. **Poorly covered regimes?** Generated P1 dominated by `DZ_ONLY`/`COMBINED` around dz≈12 mm; sparse large-DX / friction / mass diversity. Counts: {dict(regimes)}.

4. **Can Transformer fit Precision on training ICs?**  
   Best train curve row: {btr}  
   Verdict: {"partial/yes if train_ok>0" if train_ok else "NO closed-loop fit yet"} (see `SFT_LEARNING_CURVE.csv`, `FAILURE_ANALYSIS.md`).

5. **Generalize to held-out P1?** Best val={bva}; best test={bte}.

6. **Preserve safe HANDOFF on P0?** P0 packaging was limited in this run; rely on Phase A P0 inventory + future balanced P0 tapes. Over-control metrics in `CLOSED_LOOP_EVAL.csv` where P0 rows exist.

7. **State-dependent correction vs memorized sequences?** Compare teacher-forced token acc vs closed-loop; first-action diversity across ICs in eval CSV. Insufficient N for strong claims — report evidence only.

8. **Ready for unified A/B/C/D SFT?** **{ready_txt}**

## C0 frozen baselines

{c0[:12]}

## Decision

Continue Precision-only data expansion (stage 50→100) before unified consolidation. Do not start RL.

## Artifacts

`results/diagnostics/p33_precision_learnability/`
"""
    (OUT / "FINAL_REPORT.md").write_text(md, encoding="utf-8")
    print("wrote FINAL_REPORT", flush=True)


if __name__ == "__main__":
    main()
