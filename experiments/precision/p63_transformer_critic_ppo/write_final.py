"""Assemble FINAL_REPORT.md from pilot / pretrain / test artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p63_transformer_critic_ppo.paths import OUT


def write_final() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = {}
    if (OUT / "PILOT_RESULTS.json").is_file():
        pilot = json.loads((OUT / "PILOT_RESULTS.json").read_text(encoding="utf-8"))
    tests = {}
    if (OUT / "PPO_CORRECTNESS_TESTS.json").is_file():
        tests = json.loads((OUT / "PPO_CORRECTNESS_TESTS.json").read_text(encoding="utf-8"))
    warm = {}
    if (OUT / "CRITIC_PRETRAIN_REPORT.json").is_file():
        warm = json.loads((OUT / "CRITIC_PRETRAIN_REPORT.json").read_text(encoding="utf-8"))
    csum = {}
    if (OUT / "C_DATA_EXPANSION_SUMMARY.json").is_file():
        csum = json.loads((OUT / "C_DATA_EXPANSION_SUMMARY.json").read_text(encoding="utf-8"))

    a = pilot.get("A") or {}
    b = pilot.get("B") or {}
    lines = [
        "# FINAL_REPORT — P63 Transformer Critic PPO + C Expansion",
        "",
        "## Pipeline status",
        "",
        f"- Correctness tests: **{'PASS' if tests.get('all_ok') else 'FAIL / pending'}**",
        f"- Critic warm-start gates: **{(warm.get('gates') or {}).get('pass')}**",
        f"- Expanded C ICs: **{csum.get('n_selected', 'n/a')}** (see `C_DATA_MANIFEST.jsonl`)",
        "",
        "## Pilot vs P62_epoch3",
        "",
        "Short pilots only — no long PPO run was launched from this report alone.",
        "",
    ]
    if a:
        lines += [
            f"- Branch A MSP: {((a.get('pilot') or {}).get('MSP_RELAXED'))} "
            f"(Δ={a.get('delta_msp')}, wall_h={((a.get('wall_s') or 0)/3600):.2f})",
            f"- Branch B MSP: {((b.get('pilot') or {}).get('MSP_RELAXED'))} "
            f"(Δ={b.get('delta_msp')}, wall_h={((b.get('wall_s') or 0)/3600):.2f})",
            f"- A ΔMSP/GPU-h: {a.get('msp_per_gpu_hour')}",
            f"- B ΔMSP/GPU-h: {b.get('msp_per_gpu_hour')}",
            "",
        ]
    lines += [
        "## Promotion decision",
        "",
        "Promote to long PPO only if:",
        "1. All critical tests pass,",
        "2. Critic warm-start gates pass,",
        "3. Pilot is numerically stable (finite losses, bounded KL),",
        "4. Overall MSP improves **or** efficiency is clearly better at comparable MSP,",
        "5. Gains are not C-only with overall MSP regression.",
        "",
        "Policy anchoring remains a **separate ablation** if A/B/D drift is severe.",
        "",
        "Artifacts: `results/experiments/p63_transformer_critic_ppo/`.",
        "",
    ]
    if (OUT / "PILOT_RESULTS.md").is_file():
        lines += ["## Pilot table", ""]
        lines.extend(
            (OUT / "PILOT_RESULTS.md").read_text(encoding="utf-8").splitlines()[8:20]
        )
        lines.append("")
    path = OUT / "FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_final())
