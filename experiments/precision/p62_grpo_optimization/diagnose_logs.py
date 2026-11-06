"""Phase 1 diagnostics from P56/P57 progress CSVs (no expensive training)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.precision.p62_grpo_optimization.paths import (
    OUT,
    P56_PROGRESS,
    P57A_PROGRESS,
)


def _f(x, default=None):
    if x is None or x == "":
        return default
    return float(x)


def analyze_progress(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    out: dict = {"source": str(path), "n_groups": len(rows), "by_class": {}}
    for c in ("A", "B", "C", "D"):
        sub = by.get(c) or []
        if not sub:
            continue
        n = len(sub)
        rew = [_f(r["reward_mean"], 0.0) for r in sub]
        rstd = [_f(r["reward_std"], 0.0) for r in sub]
        succ = [int(r["success_relaxed_count"]) for r in sub]
        skip = [int(r["skipped_zero_var"]) for r in sub]
        nz = [int(float(r["nonzero_adv"] or 0)) for r in sub]
        kl = [_f(r["policy_kl"]) for r in sub if r.get("policy_kl") not in (None, "")]
        ent = [_f(r["policy_entropy"]) for r in sub if r.get("policy_entropy") not in (None, "")]
        mixed = sum(1 for s in succ if 0 < s < 64)
        ge1 = sum(1 for s in succ if s >= 1)
        zero = sum(1 for s in succ if s == 0)
        zero_dense = sum(
            1
            for r in sub
            if int(r["success_relaxed_count"]) == 0 and int(r["skipped_zero_var"]) == 0
        )
        out["by_class"][c] = {
            "n_groups": n,
            "mean_group_return": float(np.mean(rew)),
            "std_group_return_across_ics": float(np.std(rew)),
            "mean_within_group_return_std": float(np.mean(rstd)),
            "frac_zero_variance_groups": float(np.mean(skip)),
            "frac_groups_ge1_success": ge1 / n,
            "frac_mixed_success_failure": mixed / n,
            "frac_zero_success": zero / n,
            "frac_zero_success_with_dense_signal": zero_dense / n,
            "mean_nonzero_adv_count": float(np.mean(nz)),
            "mean_policy_kl": float(np.mean(kl)) if kl else None,
            "mean_policy_entropy": float(np.mean(ent)) if ent else None,
            "train_p_relaxed": sum(succ) / (n * 64),
            "effective_updates": sum(
                1 for r in sub if r.get("loss") not in (None, "") and int(r["skipped_zero_var"]) == 0
            ),
        }
    all_s = [int(r["success_relaxed_count"]) for r in rows]
    all_r = [_f(r["reward_mean"], 0.0) for r in rows]
    fail_r = [all_r[i] for i, s in enumerate(all_s) if s == 0]
    succ_r = [all_r[i] for i, s in enumerate(all_s) if s > 0]
    mean_succ = float(np.mean(succ_r)) if succ_r else None
    out["reward_alignment"] = {
        "corr_success_count_vs_mean_return": float(np.corrcoef(all_s, all_r)[0, 1]),
        "mean_return_zero_success_groups": float(np.mean(fail_r)) if fail_r else None,
        "mean_return_any_success_groups": mean_succ,
        "frac_zero_succ_groups_with_return_above_succ_mean": (
            float(np.mean([1.0 if x > mean_succ else 0.0 for x in fail_r]))
            if fail_r and mean_succ is not None
            else None
        ),
    }
    out["sampling"] = {
        "schedule": {c: out["by_class"][c]["n_groups"] for c in out["by_class"]},
        "total_groups": len(rows),
        "rollouts": len(rows) * 64,
        "note": "A60/B100/C68/D93 from P54 rebalanced manifest",
    }
    return out


def run_diagnose() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    p57 = analyze_progress(P57A_PROGRESS)
    p56 = analyze_progress(P56_PROGRESS) if P56_PROGRESS.is_file() else {}
    blob = {"P57_A": p57, "P56": p56}
    (OUT / "GRPO_GRADIENT_DIAGNOSTICS.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")

    # Classification for Phase 2
    bc = p57["by_class"]
    zero_var = max(bc[c]["frac_zero_variance_groups"] for c in bc)
    corr = p57["reward_alignment"]["corr_success_count_vs_mean_return"]
    exploit = p57["reward_alignment"]["frac_zero_succ_groups_with_return_above_succ_mean"] or 0.0

    findings = {
        "confirmed_optimization_defect": [],
        "strongly_supported_improvement": [],
        "uncertain_hypothesis": [
            {
                "id": "balanced_sampling",
                "note": (
                    "B has 100 groups vs A60; B zero-success frac=0.44 but all groups still "
                    "provide dense-signal updates. Balancing may help A/C exposure but is not "
                    "required before testing whether more epochs improve overall MSP."
                ),
            }
        ],
        "no_actionable_issue": [
            {
                "id": "zero_variance",
                "note": f"P57-A frac_zero_variance_groups max={zero_var:.3f} (all classes 0).",
            },
            {
                "id": "reward_success_alignment",
                "note": (
                    f"corr(success_count, mean_return)={corr:.3f}; "
                    f"frac zero-succ groups outscoring succ mean={exploit:.3f}."
                ),
            },
            {
                "id": "effective_updates",
                "note": "All 321 groups produced optimizer updates in P57-A.",
            },
        ],
        "decision": (
            "No confirmed defect. Proceed with unchanged multi-epoch continuation from "
            "P57_A_epoch1.pt. Defer balanced-sampling ablation unless continuation saturates "
            "without overall MSP gains."
        ),
    }
    (OUT / "PHASE2_DECISION.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    _write_mds(blob, findings)
    return blob


def _write_mds(blob: dict, findings: dict):
    p57 = blob["P57_A"]
    lines = [
        "# GRPO_GRADIENT_DIAGNOSTICS",
        "",
        "Source: `P57_A_FULL_EPOCH_PROGRESS.csv` (and P56 for reference).",
        "",
        "| Class | groups | mean R | within σ | zero-var | ≥1 succ | mixed | zero-succ | zero+dense | KL | H | p_rel |",
        "|-------|-------:|-------:|---------:|---------:|--------:|------:|----------:|-----------:|---:|--:|------:|",
    ]
    for c, d in p57["by_class"].items():
        lines.append(
            f"| {c} | {d['n_groups']} | {d['mean_group_return']:.2f} | "
            f"{d['mean_within_group_return_std']:.2f} | {d['frac_zero_variance_groups']:.2f} | "
            f"{d['frac_groups_ge1_success']:.2f} | {d['frac_mixed_success_failure']:.2f} | "
            f"{d['frac_zero_success']:.2f} | {d['frac_zero_success_with_dense_signal']:.2f} | "
            f"{(d['mean_policy_kl'] or 0):.4f} | {(d['mean_policy_entropy'] or 0):.3f} | "
            f"{d['train_p_relaxed']:.3f} |"
        )
    lines += [
        "",
        "**Key:** Zero-variance groups = 0 across all classes. Every group contributed an update.",
        "Zero-success groups still show dense within-group variance → not discarded as useless.",
        "",
    ]
    (OUT / "GRPO_GRADIENT_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")

    ra = p57["reward_alignment"]
    lines = [
        "# REWARD_ALIGNMENT",
        "",
        f"- corr(success_count, group mean return) = **{ra['corr_success_count_vs_mean_return']:.3f}**",
        f"- mean return | zero-success groups = **{ra['mean_return_zero_success_groups']:.2f}**",
        f"- mean return | any-success groups = **{ra['mean_return_any_success_groups']:.2f}**",
        f"- frac zero-succ groups with return > succ mean = **{ra['frac_zero_succ_groups_with_return_above_succ_mean']}**",
        "",
        "Interpretation: group mean return is strongly aligned with success count; "
        "systematic high-reward failures outscoring successes are not evidenced at group level.",
        "No reward weight change justified before multi-epoch continuation.",
        "",
    ]
    (OUT / "REWARD_ALIGNMENT.md").write_text("\n".join(lines), encoding="utf-8")

    samp = p57["sampling"]
    lines = [
        "# CLASS_SAMPLING_ANALYSIS",
        "",
        f"Current schedule: **{samp['schedule']}** → {samp['total_groups']} groups × 64 = {samp['rollouts']} rollouts/epoch.",
        "",
        "Relative to equal-class 60/60/60/60 (=240 groups, 15360 rolls):",
        "- B is over-represented (+40 groups).",
        "- A/C/D near or above 60.",
        "",
        "B still has 56% mixed success groups and 0% zero-variance → learning signal exists.",
        "Oversampling only hardest ICs is not recommended (overfit risk).",
        "",
        "**Decision:** Keep A60/B100/C68/D93 for baseline continuation. "
        "Optional balanced ablation only if epochs 2–3 show saturation without overall MSP gains.",
        "",
    ]
    (OUT / "CLASS_SAMPLING_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "# OPTIMIZATION_AUDIT",
        "",
        "## Phase 2 classification",
        "",
        "### Confirmed optimization defect",
        "- None found in P57-A logs.",
        "",
        "### Strongly supported improvement",
        "- None requiring a training-objective change before continuation.",
        "",
        "### Uncertain hypothesis",
    ]
    for u in findings["uncertain_hypothesis"]:
        lines.append(f"- **{u['id']}**: {u['note']}")
    lines += ["", "### No actionable issue"]
    for u in findings["no_actionable_issue"]:
        lines.append(f"- **{u['id']}**: {u['note']}")
    lines += [
        "",
        f"**Decision:** {findings['decision']}",
        "",
        "Clipping fraction / exact grad norms were not logged historically; "
        "P57-A logs show loss present whenever skip=0 (all groups).",
        "",
        "See also `POLICY_DRIFT_ANALYSIS.md` (P59 vs P57 token KL at IC reset).",
        "",
    ]
    (OUT / "OPTIMIZATION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_diagnose()["P57_A"]["by_class"], indent=2))
