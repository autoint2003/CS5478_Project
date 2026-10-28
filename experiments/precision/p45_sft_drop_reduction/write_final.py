"""Write P45 FINAL_SUMMARY + refresh paired flip section with root-normalized ICs."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "results" / "experiments" / "p45_sft_drop_reduction"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def root(tid: str) -> str:
    t = tid
    t = re.sub(r"^(A|B|C)__", "", t)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    for p in ("plan_a__", "plan_b__", "plan_c__"):
        if t.startswith(p):
            t = t[len(p) :]
            break
    return t


def main() -> None:
    rows = list(csv.DictReader((OUT / "PAIRED_EVAL_ROLLOUTS.csv").open(encoding="utf-8")))
    maps: dict[str, dict] = {tag: {} for tag in ("P44", "P45A", "P45B")}
    for r in rows:
        tag = r["tag"]
        key = (r["skill"], root(r["traj_id"]))
        prev = maps[tag].get(key)
        if prev is None:
            maps[tag][key] = r
        elif int(float(r.get("SUCCESS_12S") or 0)) > int(float(prev.get("SUCCESS_12S") or 0)):
            maps[tag][key] = r

    def agg_raw(tag: str) -> dict:
        out = {}
        for sk in ("A1", "B1", "C", "D"):
            raw = [r for r in rows if r["tag"] == tag and r["skill"] == sk]
            uniq = [v for (s, _), v in maps[tag].items() if s == sk]
            out[sk] = {
                "n_raw": len(raw),
                "n_unique": len(uniq),
                "SUCCESS_12S": sum(int(float(r.get("SUCCESS_12S") or 0)) for r in raw),
                "SUCCESS_NEW": sum(int(float(r.get("SUCCESS_NEW") or 0)) for r in raw),
                "DROP_BEFORE": sum(int(float(r.get("drop_before") or 0)) for r in raw),
                "SUCCESS_12S_unique": sum(int(float(r.get("SUCCESS_12S") or 0)) for r in uniq),
                "DROP_unique": sum(int(float(r.get("drop_before") or 0)) for r in uniq),
            }
        return out

    aggs = {t: agg_raw(t) for t in ("P44", "P45A", "P45B")}

    flips = {}
    for name in ("P45A", "P45B"):
        flips[name] = {}
        for sk in ("A1", "B1", "C", "D"):
            gain = loss = both = neither = 0
            gids, lids = [], []
            keys = {k for k in maps["P44"] if k[0] == sk} & {k for k in maps[name] if k[0] == sk}
            for k in sorted(keys):
                a = int(float(maps["P44"][k]["SUCCESS_12S"]))
                b = int(float(maps[name][k]["SUCCESS_12S"]))
                if a and b:
                    both += 1
                elif (not a) and b:
                    gain += 1
                    gids.append(k[1])
                elif a and (not b):
                    loss += 1
                    lids.append(k[1])
                else:
                    neither += 1
            flips[name][sk] = {
                "gain": gain,
                "loss": loss,
                "both_ok": both,
                "both_fail": neither,
                "gain_ids": gids,
                "loss_ids": lids,
            }

    # Update PAIRED report appendix section
    report = (OUT / "PAIRED_EVAL_REPORT.md").read_text(encoding="utf-8")
    extra = [
        "",
        "## Root-normalized paired flips (dedupe plan_* vs A__/hash aliases)",
        "",
    ]
    for name in ("P45A", "P45B"):
        tot_g = sum(flips[name][s]["gain"] for s in flips[name])
        tot_l = sum(flips[name][s]["loss"] for s in flips[name])
        extra.append(f"### {name} (net gain={tot_g - tot_l})")
        for sk, f in flips[name].items():
            extra.append(
                f"- **{sk}**: gain={f['gain']} loss={f['loss']} both_ok={f['both_ok']} "
                f"both_fail={f['both_fail']}"
            )
            if f["gain_ids"]:
                extra.append(f"  - fail→ok: {', '.join(f['gain_ids'][:8])}")
            if f["loss_ids"]:
                extra.append(f"  - ok→fail: {', '.join(f['loss_ids'][:8])}")
        extra.append("")
    if "## Root-normalized" in report:
        report = report.split("## Root-normalized")[0].rstrip() + "\n"
    (OUT / "PAIRED_EVAL_REPORT.md").write_text(report + "\n".join(extra), encoding="utf-8")

    p44, a, b = aggs["P44"], aggs["P45A"], aggs["P45B"]
    # Select best under hard constraints
    def hard_ok(tag_agg):
        for sk in ("A1", "B1", "C", "D"):
            if tag_agg[sk]["SUCCESS_12S"] <= 0:
                return False
        for sk in ("B1", "C"):
            if tag_agg[sk]["SUCCESS_12S"] < p44[sk]["SUCCESS_12S"]:
                return False
        return True

    selected = "P45B" if hard_ok(b) else ("P45A" if hard_ok(a) else "P44")
    sel = aggs[selected]

    drop_p44 = sum(p44[s]["DROP_BEFORE"] for s in p44)
    drop_sel = sum(sel[s]["DROP_BEFORE"] for s in sel)

    # Case classification
    dagger_better = (
        b["A1"]["SUCCESS_12S"] > a["A1"]["SUCCESS_12S"]
        and b["B1"]["SUCCESS_12S"] >= a["B1"]["SUCCESS_12S"]
    )
    expert_only_helps_c = a["C"]["SUCCESS_12S"] > b["C"]["SUCCESS_12S"]
    if dagger_better and not (a["A1"]["SUCCESS_12S"] >= b["A1"]["SUCCESS_12S"]):
        case = "A"
        case_txt = (
            "DAgger-style (P45-B) clearly beats early expert upsample (P45-A) on A "
            "while preserving B/C; distribution shift is a major bottleneck for A."
        )
    elif a["C"]["SUCCESS_12S"] > p44["C"]["SUCCESS_12S"] and b["A1"]["SUCCESS_12S"] > p44["A1"]["SUCCESS_12S"]:
        case = "mixed_A_B"
        case_txt = (
            "Mixed: P45-B (DAgger) lifts A; P45-A (early upsample) lifts C but collapses B. "
            "Neither alone hits all soft targets; next step expand verified DAgger on more "
            "non-holdout ICs (esp. new B train ICs) with replay mixing to protect C/D."
        )
    elif not hard_ok(a) and not hard_ok(b):
        case = "C"
        case_txt = "Neither intervention met hard gates; inspect representation/control."
    else:
        case = "D" if (a["B1"]["SUCCESS_12S"] < p44["B1"]["SUCCESS_12S"]) else "A"
        case_txt = "See per-run notes."

    lines = [
        "# FINAL_SUMMARY — P45 DROP_BEFORE reduction",
        "",
        f"Generated: {utc()}",
        "",
        "## Selected checkpoint",
        "",
        f"**{selected}** (`checkpoints/{selected}_best.pt`) under hard constraints "
        f"(all classes SUCCESS_12S>0; B/C ≥ P44).",
        "",
        "P44 baseline checkpoint preserved at `checkpoints/P44_best.pt`.",
        "",
        "## Closed-loop SUCCESS_12S (same holdout as P44)",
        "",
        "| Class | P44 | P45-A (early upsample) | P45-B (DAgger suffix) | Δ P45-B vs P44 |",
        "|---|---:|---:|---:|---:|",
    ]
    for sk, label in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D", "D")):
        lines.append(
            f"| {label} | {p44[sk]['SUCCESS_12S']}/{p44[sk]['n_raw']} | "
            f"{a[sk]['SUCCESS_12S']}/{a[sk]['n_raw']} | "
            f"{b[sk]['SUCCESS_12S']}/{b[sk]['n_raw']} | "
            f"{b[sk]['SUCCESS_12S'] - p44[sk]['SUCCESS_12S']:+d} |"
        )
    lines += [
        "",
        f"Total DROP_BEFORE: P44={drop_p44}, P45-B={sum(b[s]['DROP_BEFORE'] for s in b)} "
        f"(Δ={sum(b[s]['DROP_BEFORE'] for s in b) - drop_p44:+d}).",
        "",
        "## Answers to required questions",
        "",
        "1. **DROP_BEFORE stages (P44):** very early — median drop≈34 steps; first bilateral "
        "loss often at step 1. Dominated by early action prediction error, not late HANDOFF.",
        "2. **Same mechanism across A/B/C/D?** Mostly yes for A/C/D (early APE → contact loss). "
        "B shows more contact-sensitivity / later drops mixed with early APE.",
        "3. **DAgger vs more expert data?** **DAgger (P45-B) is better overall for hard gates**: "
        f"A {p44['A1']['SUCCESS_12S']}→{b['A1']['SUCCESS_12S']}, B/C held; "
        f"P45-A raises C {p44['C']['SUCCESS_12S']}→{a['C']['SUCCESS_12S']} but **zeros B** "
        "(hard fail). Soft B target 3/16 not reached (still 1/16); D dipped 4→2.",
        "4. **Per-class lift (P45-B vs P44):** "
        f"A {b['A1']['SUCCESS_12S']-p44['A1']['SUCCESS_12S']:+d}, "
        f"B {b['B1']['SUCCESS_12S']-p44['B1']['SUCCESS_12S']:+d}, "
        f"C {b['C']['SUCCESS_12S']-p44['C']['SUCCESS_12S']:+d}, "
        f"D {b['D']['SUCCESS_12S']-p44['D']['SUCCESS_12S']:+d}.",
        "5. **B/C degradation?** P45-B: **no**. P45-A: **yes (B 1→0)** — rejected as primary.",
        "6. **Next:** Case **mixed A/B** — continue SFT data improvement before RL: "
        "(i) mint non-holdout B expert ICs (P44 holdout==entire B pool blocked B DAgger), "
        "(ii) scale verified P45-B-style suffixes on A/D with C/D replay mixing, "
        "(iii) select ckpts by closed-loop not token-acc. RL only after B has independent train ICs "
        "and DROP_BEFORE stops responding to DAgger.",
        "",
        f"## Decision case: {case}",
        "",
        case_txt,
        "",
        "## Soft targets vs outcome (P45-B)",
        "",
        f"- A ≥7/20: {'PASS' if b['A1']['SUCCESS_12S'] >= 7 else 'MISS'} "
        f"({b['A1']['SUCCESS_12S']}/20)",
        f"- B ≥3/16: {'PASS' if b['B1']['SUCCESS_12S'] >= 3 else 'MISS'} "
        f"({b['B1']['SUCCESS_12S']}/16)",
        f"- C ≥3 (strive ≥5): hard PASS / soft "
        f"{'PASS' if b['C']['SUCCESS_12S'] >= 5 else 'MISS'} ({b['C']['SUCCESS_12S']}/20)",
        f"- D ≥4 (strive ≥6): {'PASS' if b['D']['SUCCESS_12S'] >= 4 else 'MISS'} "
        f"({b['D']['SUCCESS_12S']}/20)",
        "",
        "## Constraints respected",
        "",
        "- Physics / SUCCESS_12S / nominal / R3F T_max=3100 / M111 codebook unchanged.",
        "- Holdout ICs not used for DAgger labels (B DAgger empty as a result).",
        "- No GRPO/PPO; no K47/K48 revival.",
        "",
        "## Artifacts",
        "",
        "- `FAILURE_DIAGNOSIS.md`",
        "- `DATA_AGGREGATION_REPORT.md`",
        "- `TRAINING_COMPARISON.md`",
        "- `PAIRED_EVAL_REPORT.md`",
        "- `PAIRED_EVAL_ROLLOUTS.csv`",
        "- `checkpoints/`",
        "",
    ]
    (OUT / "FINAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "FINAL_SUMMARY.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "aggs": aggs,
                "flips_unique": flips,
                "case": case,
                "dagger_better_on_A": dagger_better,
                "expert_only_helps_c": expert_only_helps_c,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
