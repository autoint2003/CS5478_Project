"""P46 paired closed-loop eval on P44 holdouts; reuse P44/P45B CSV baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p45_sft_drop_reduction.paired_eval import (  # noqa: E402
    eval_ckpt,
    summarize,
    write_csv,
)
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p46_b_tree80_expansion.paths import (  # noqa: E402
    OUT,
    P44_EVAL,
    P45,
    SEED,
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_baseline_tag(path: Path, tag: str) -> list[dict]:
    if not path.exists():
        return []
    return [{**r, "tag": tag} for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("tag") == tag]


def run(*, only_p46: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    hold = [r for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")) if r.get("tag") == "holdout"]
    uniq = {}
    for r in hold:
        uniq[(r["skill"], r["traj_id"])] = r
    hold = list(uniq.values())
    jobs_by_key = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}

    all_rows: list[dict] = []
    # P44 from eval CSV
    for r in hold:
        all_rows.append(
            {
                "tag": "P44",
                "skill": r["skill"],
                "traj_id": r["traj_id"],
                "SUCCESS_12S": int(r.get("SUCCESS_12S") or 0),
                "SUCCESS_NEW": int(r.get("SUCCESS_NEW") or 0),
                "drop_before": int(r.get("fail_category") == "DROP_BEFORE"),
                "handoff": int(r.get("fail_category") != "DROP_BEFORE"),
                "handoff_reason": r.get("handoff_reason"),
                "fail_category": r.get("fail_category"),
            }
        )
    # Reuse P45B rollouts if present
    p45_rows = load_baseline_tag(P45 / "PAIRED_EVAL_ROLLOUTS.csv", "P45B")
    if p45_rows:
        # keep only holdout keys
        hold_keys = {(r["skill"], r["traj_id"]) for r in hold}
        # P45B used job traj_ids — include all P45B rows for skill-level aggregate
        all_rows.extend(p45_rows)
        print(f"Reused P45B rollouts: {len(p45_rows)}", flush=True)
    else:
        print("WARN: no P45B rollouts to reuse", flush=True)

    ckpt = OUT / "checkpoints/P46_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    all_rows.extend(eval_ckpt("P46", ckpt, jobs_by_key, hold))

    write_csv(OUT / "PAIRED_EVAL_ROLLOUTS.csv", all_rows)
    summaries = {tag: summarize(all_rows, tag) for tag in ("P44", "P45B", "P46")}

    lines = [
        "# PAIRED_EVAL_REPORT — P46",
        "",
        f"Generated: {utc()}",
        "Same P44 holdout ICs. R3F T_max=3100 unchanged. Tree80 used only as frozen B data source.",
        "",
        "## Aggregate SUCCESS_12S",
        "",
        "| Skill | P44 | P45-B | P46 ( +Tree80 B) | P44 DROP | P45-B DROP | P46 DROP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sk in ("A1", "B1", "C", "D"):
        a, b, c = summaries.get("P44", {}).get(sk, {}), summaries.get("P45B", {}).get(sk, {}), summaries.get("P46", {}).get(sk, {})
        lines.append(
            f"| {sk} | {a.get('SUCCESS_12S', 0)}/{a.get('n', 0)} | "
            f"{b.get('SUCCESS_12S', 0)}/{b.get('n', 0)} | "
            f"{c.get('SUCCESS_12S', 0)}/{c.get('n', 0)} | "
            f"{a.get('DROP_BEFORE', 0)} | {b.get('DROP_BEFORE', 0)} | {c.get('DROP_BEFORE', 0)} |"
        )

    # flips vs P44 and P45B for B
    def mmap(tag):
        return {(r["skill"], r["traj_id"]): r for r in all_rows if r["tag"] == tag}

    p44m, p46m = mmap("P44"), mmap("P46")
    lines += ["", "## Paired flips P44 → P46 (SUCCESS_12S)", ""]
    for sk in ("A1", "B1", "C", "D"):
        gain = loss = 0
        gids, lids = [], []
        for k, a in p44m.items():
            if k[0] != sk or k not in p46m:
                continue
            aok, bok = int(float(a["SUCCESS_12S"])), int(float(p46m[k]["SUCCESS_12S"]))
            if (not aok) and bok:
                gain += 1
                gids.append(k[1][:48])
            elif aok and (not bok):
                loss += 1
                lids.append(k[1][:48])
        lines.append(f"- **{sk}**: gain={gain} loss={loss}")
        if gids:
            lines.append(f"  - fail→ok: {', '.join(gids[:8])}")
        if lids:
            lines.append(f"  - ok→fail: {', '.join(lids[:8])}")

    lines += ["", "## Acceptance", ""]
    p44, p46 = summaries["P44"], summaries["P46"]
    for sk in ("A1", "B1", "C", "D"):
        if p46.get(sk, {}).get("SUCCESS_12S", 0) <= 0:
            lines.append(f"- FAIL {sk} SUCCESS_12S=0")
        else:
            lines.append(f"- OK {sk} >0 ({p46[sk]['SUCCESS_12S']}/{p46[sk]['n']})")
    for sk in ("B1", "C"):
        if p46[sk]["SUCCESS_12S"] < p44[sk]["SUCCESS_12S"]:
            lines.append(f"- WARN {sk} below P44")
        else:
            lines.append(f"- OK {sk} ≥ P44")
    if p46["B1"]["SUCCESS_12S"] > p44["B1"]["SUCCESS_12S"]:
        lines.append(
            f"- B improved vs P44: {p44['B1']['SUCCESS_12S']} → {p46['B1']['SUCCESS_12S']}"
        )
    if "P45B" in summaries and summaries["P45B"].get("B1"):
        lines.append(
            f"- B vs P45-B: {summaries['P45B']['B1']['SUCCESS_12S']} → {p46['B1']['SUCCESS_12S']}"
        )

    (OUT / "PAIRED_EVAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "PAIRED_EVAL_SUMMARY.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)

    # FINAL_SUMMARY
    final = [
        "# FINAL_SUMMARY — P46 B Tree80 expansion",
        "",
        f"Generated: {utc()}",
        "",
        "Added non-holdout Tree80 (K48 GRPO frozen) B experts with M111_12S into P45-B mix; "
        "fine-tuned from P45-B. No K48/GRPO retrain.",
        "",
        "| Class | P44 | P45-B | P46 |",
        "|---|---:|---:|---:|",
    ]
    for sk, lab in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D", "D")):
        final.append(
            f"| {lab} | {p44[sk]['SUCCESS_12S']}/{p44[sk]['n']} | "
            f"{summaries.get('P45B', {}).get(sk, {}).get('SUCCESS_12S', '?')}/"
            f"{summaries.get('P45B', {}).get(sk, {}).get('n', '?')} | "
            f"{p46[sk]['SUCCESS_12S']}/{p46[sk]['n']} |"
        )
    final += [
        "",
        f"B DROP_BEFORE: P44={p44['B1']['DROP_BEFORE']} → P46={p46['B1']['DROP_BEFORE']}",
        "",
        "See DATA_PACK_REPORT.md and PAIRED_EVAL_REPORT.md.",
        "",
    ]
    (OUT / "FINAL_SUMMARY.md").write_text("\n".join(final), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-p46", action="store_true", default=True)
    args = ap.parse_args()
    run(only_p46=args.only_p46)


if __name__ == "__main__":
    main()
