"""One-shot enrichment of P38 FINAL_REPORT.md from existing artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("results/diagnostics/p38_unified_motion_only_sft")


def stats(rows, kind, suite, nt_only=False):
    sub = [r for r in rows if r["kind"] == kind and r["suite"] == suite]
    if nt_only:
        sub = [r for r in sub if r.get("nontrivial") == "1"]
    return {
        "n": len(sub),
        "WORKING_reach": sum(int(r["ever_working"]) for r in sub),
        "HANDOFF": sum(int(r["handoff"]) for r in sub),
        "SUCCESS_12S": sum(int(r["SUCCESS_12S"]) for r in sub),
        "SUCCESS_NEW": sum(int(r["SUCCESS_NEW"]) for r in sub),
        "timeout": sum(int(r["timeout"]) for r in sub),
        "drop": sum(int(r["drop"]) for r in sub),
        "init_working": sum(int(r["init_working_satisfied"]) for r in sub),
        "cats": dict(Counter(r["fail_category"] for r in sub)),
    }


def main():
    rows = []
    for kind in ("E0", "E1", "E2"):
        for r in csv.DictReader((OUT / "eval" / f"ROLLOUTS_{kind}.csv").open(encoding="utf-8")):
            r["kind"] = kind
            rows.append(r)

    train = json.loads((OUT / "TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    g1 = list(csv.DictReader((OUT / "gate1_working_handoff" / "GATE1_AGGREGATE.csv").open(encoding="utf-8")))
    census = list(csv.DictReader((OUT / "gate2_dataset" / "DATASET_CENSUS.csv").open(encoding="utf-8")))
    meta = json.loads((OUT / "gate2_dataset" / "DATASET_META.json").read_text(encoding="utf-8"))

    # token acc at last eval from curves
    last_acc = {}
    for kind in ("E0", "E1", "E2"):
        curve = OUT / "checkpoints" / kind / "LEARNING_CURVE.csv"
        last = None
        for r in csv.DictReader(curve.open(encoding="utf-8")):
            if r.get("val_acc_A1"):
                last = r
        last_acc[kind] = last

    lines: list[str] = []
    a = lines.append
    a("# FINAL_REPORT — P38 Unified A/B/C/D Motion-Only SFT")
    a("")
    a(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    a("")
    a("## Scope")
    a("")
    a("- Shared M111 motion-only SFT (112-way head retained; HANDOFF never supervised; masked at inference).")
    a("- Shared WORKING HANDOFF: θ≤10°, e_pos≤12 mm + P36 nominal contract + 12s STRICT.")
    a("- Experiments: E0 scratch, E1←P31 M2_pcgrad, E2←P36 D-only. Matched updates=800, batch=4/skill, lr=3e-4, L=(L_A+L_B+L_C+L_D)/4.")
    a("- SFT only; no RL. P31/P36/P37 artifacts not overwritten.")
    a("")
    a("## Baseline references (not a unified model)")
    a("")
    a("- P31 M2_pcgrad (learned HANDOFF): A1=14/19, B1=3/4, C=3/19, D2=0/17.")
    a("- P36 D-only motion SFT: nontrivial test **8/11** STRICT.")
    a("- P37 frozen WORKING HANDOFF: nontrivial test **8/11**; WORKING-triggered expert **35/35** NEW on that D set.")
    a("")
    a("## Gate 1 — Shared WORKING HANDOFF on experts")
    a("")
    a("Metric identical across skills: `measure(U_TGT, P_TGT)` + WORKING_* thresholds.")
    a("")
    a("| Skill | n | Reach WORKING | SUCCESS_12S | SUCCESS_NEW | conversion | Dominant failure |")
    a("|---|---:|---:|---:|---:|---|---|")
    for r in g1:
        a(
            f"| {r['skill']} | {r['n_traj']} | {r['reached_working']} | {r['SUCCESS_12S']} | "
            f"{r['SUCCESS_NEW']} | {r['NEW_given_reach']} | `{r.get('reasons','')}` |"
        )
    a("")
    a("**Verdict: WORKING is NOT a reliable shared early-stop for A/B/C.**")
    a("- **C**: 0/92 SUCCESS_NEW — all DROP_AFTER_HANDOFF after mid-trajectory WORKING (~steps 5–8).")
    a("- **A1/B1**: reach WORKING but mostly LOST_GRASP after nominal takeover (6/63 and 2/16 NEW).")
    a("- **D**: 53/139 NEW; better conversion but many hold-fails.")
    a("- No skill-specific HANDOFF rules added (per mandate). Training used the shared rule with this Gate1 caveat.")
    a("")
    a("See `gate1_working_handoff/GATE1_REPORT.md`.")
    a("")
    a("## Gate 2 — Motion-only dataset")
    a("")
    a("| Skill | Split | Roots | Trajectories | Chunks | Motion tokens | Unique motion IDs |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for r in census:
        a(
            f"| {r['skill']} | {r['split']} | {r['n_independent_roots']} | {r['n_trajectories']} | "
            f"{r['n_chunks']} | {r['n_motion_tokens']} | {r['n_unique_motion_tokens']} |"
        )
    a("")
    a("- HANDOFF stripped from targets; 111 motion actions preserved; leakage: `SPLIT_LEAKAGE.csv`.")
    a(f"- Meta note: `{meta.get('note', '')}`")
    a("- Encoded expert WORKING-HANDOFF replay = Gate1 (not learning evidence). Continuous vs discrete vs learned are distinct.")
    a("")
    a("## Training (val-selected best; not test)")
    a("")
    a("| Exp | Init | Best upd | Val mean loss | best sha256 |")
    a("|---|---|---:|---:|---|")
    for kind in ("E0", "E1", "E2"):
        m = train[kind]
        b = m["best"]
        init = m.get("init") or {}
        iname = init.get("init") if isinstance(init, dict) else init
        sha = (m.get("best_sha256") or "")[:16]
        a(f"| {kind} | {iname} | {b['update']} | {float(b['val_mean_loss']):.4f} | `{sha}…` |")
    a("")
    a("Open-loop token accuracy at last val eval:")
    for kind, r in last_acc.items():
        if not r:
            continue
        a(
            f"- {kind}: A1={r.get('val_acc_A1')} B1={r.get('val_acc_B1')} "
            f"C={r.get('val_acc_C')} D={r.get('val_acc_D')} (val_loss={r.get('val_mean_loss')})"
        )
    a("")
    a("## Closed-loop evaluation (shared WORKING HANDOFF)")
    a("")
    a("### Full suites (SUCCESS_NEW / n)")
    a("")
    a("| Suite | E0 | E1 | E2 | Notes |")
    a("|---|---|---|---|---|")
    for suite in ("A1", "B1", "C", "D2", "D_nt_test"):
        cells = []
        for kind in ("E0", "E1", "E2"):
            s = stats(rows, kind, suite, False)
            cells.append(f"{s['SUCCESS_NEW']}/{s['n']}")
        note = ""
        if suite == "D2":
            note = "all NEW are init-WORKING; nontrivial D2 = 0/8 all kinds"
        if suite == "D_nt_test":
            note = "nontrivial by construction"
        a(f"| {suite} | {cells[0]} | {cells[1]} | {cells[2]} | {note} |")
    a("")
    a("### Nontrivial-only STRICT (learned correction)")
    a("")
    a("| Suite | E0 | E1 | E2 |")
    a("|---|---|---|---|")
    for suite in ("A1", "B1", "C", "D2", "D_nt_test"):
        cells = []
        for kind in ("E0", "E1", "E2"):
            s = stats(rows, kind, suite, True)
            cells.append(f"{s['SUCCESS_NEW']}/{s['n']}")
        a(f"| {suite} | {cells[0]} | {cells[1]} | {cells[2]} |")
    a("")
    a("### Per-skill detail")
    a("")
    for suite in ("A1", "B1", "C", "D2", "D_nt_test"):
        a(f"**{suite}**")
        a("")
        a("| Kind | n | WK reach | HANDOFF | NEW | timeout | drop | init_WK | Failure categories |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for kind in ("E0", "E1", "E2"):
            s = stats(rows, kind, suite, False)
            a(
                f"| {kind} | {s['n']} | {s['WORKING_reach']} | {s['HANDOFF']} | {s['SUCCESS_NEW']} | "
                f"{s['timeout']} | {s['drop']} | {s['init_working']} | `{s['cats']}` |"
            )
        a("")

    a("## Feasibility milestone")
    a("")
    a("Criterion: ≥1 **nontrivial** STRICT on each of A, B, C, and D from **one** checkpoint.")
    a("")
    for kind in ("E0", "E1", "E2"):
        detail = {}
        ok = True
        for suite, label in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D_nt_test", "D")):
            s = stats(rows, kind, suite, True)
            detail[label] = f"{s['SUCCESS_NEW']}/{s['n']}"
            if s["SUCCESS_NEW"] < 1:
                ok = False
        d2 = stats(rows, kind, "D2", True)
        verdict = "PASS" if ok else "FAIL"
        a(f"- **{kind}**: {verdict} — A/B/C/D_nt={detail}; D2_nt={d2['SUCCESS_NEW']}/{d2['n']}")
    a("")
    a("**Milestone achieved? NO.**")
    a("")
    a("## Scientific conclusions")
    a("")
    a(
        "1. **E0 from scratch:** Open-loop A/B/C/D improves; WORKING reach on many A/B; nontrivial D "
        "7/11 D_nt. **Zero** A/B/C STRICT — post-HANDOFF LOST_GRASP/DROP (Gate1 predicted). C: 19/19 DROP before WORKING."
    )
    a(
        "2. **E1←P31:** Does **not** restore P31 A/B/C under rule WORKING HANDOFF "
        "(0/19, 0/4, 0/19 vs learned-HANDOFF 14/19, 3/4, 3/19). C reaches WORKING twice then DROP_AFTER. D_nt 6/11 ≈ E0."
    )
    a(
        "3. **E2←P36:** **Loses D** vs P36 (D_nt 2/11 vs 8/11; 5 timeouts). A/B/C still 0 STRICT. "
        "Multi-skill fine-tune interferes with D while failing to unlock A/B/C."
    )
    a(
        "4. **Main limitation:** Gate1 HANDOFF contract — WORKING mid-traj ≠ safe nominal for A/B/C — "
        "not open-loop optimization (val loss falls; token acc high). Secondary: C drop-before-WORKING; E2 D interference."
    )
    a(
        "5. **One-ckpt all-four STRICT:** not achieved. Best D_nt is E0 (7/11); no ckpt has A or B or C NEW≥1."
    )
    a("")
    a("**Skill D / unified recovery solved?** **NO** — D nontrivial retained partially (E0/E1); A/B/C feasibility under shared WORKING HANDOFF fails.")
    a("")
    a("## Artifacts")
    a("")
    a("- `gate1_working_handoff/` — Gate1 validation + counterexamples")
    a("- `gate2_dataset/` — census, leakage, meta")
    a("- `checkpoints/E0|E1|E2/` — curves, train_meta, sha256, best ckpt")
    a("- `eval/` — per-root rollouts + SUMMARY.csv")
    a("- `FINAL_REPORT.md`")
    a("")

    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
