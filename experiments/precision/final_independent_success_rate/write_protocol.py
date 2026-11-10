"""Write EVAL_PROTOCOL.md and DATA_LEAKAGE_AUDIT.md from frozen manifest."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.final_independent_success_rate.paths import OUT, TARGET_PER_CLASS


def write_protocol(manifest: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fs = manifest.get("frozen_system") or {}
    lines = [
        "# EVAL_PROTOCOL — Small-Scale Independent Evaluation",
        "",
        "## Purpose",
        "",
        "One-time, preregistered **small-scale independent evaluation** of the frozen CS5478 grasp-recovery system.",
        "Primary metric: **Episode Success Rate** = successes / independently evaluated episodes.",
        "**Not** MSP-64. **Not** repeated rollouts of the same IC as independent trials.",
        "",
        "## Scale",
        "",
        f"- Target: **{TARGET_PER_CLASS} independent ICs / class** × 4 classes × **1 rollout / IC**",
        f"- Planned episodes in this run: **{manifest.get('n_episodes_planned')}**",
        f"- Shortfalls (no train padding): `{json.dumps(manifest.get('shortfalls'))}`",
        "",
        "## Frozen system",
        "",
        f"- Actor: `{fs.get('actor_path')}` MD5 `{fs.get('actor_md5')}`",
        f"- Critic: `{fs.get('critic_path')}` MD5 `{fs.get('critic_md5')}`",
        "- Controller: P70 V2 (`require_no_contact`, ready_score), `rule=None`",
        "- Vocabulary: original M111 from Actor checkpoint",
        "- Success: `SUCCESS_RELAXED_12S` (full episode / 12s criterion as implemented in env)",
        f"- Max steps: {fs.get('max_ep_steps')}; R3F t_max: {fs.get('t_max')}",
        f"- Inference entry: `{fs.get('inference')}`",
        "",
        "## IC generation / selection",
        "",
        "1. Source pool = **heldout** starts from `load_heldout_starts` (P44 holdout tags + expert jobs).",
        "2. De-duplicate by `root_id`.",
        "3. Exclude: P54 train schedule (+roots), P64 IC_EXPOSURE, P82/P83 diagnostic panel ICs,",
        "   P61/P62 dev + final-test slices, `blocked_eval_ids`.",
        "4. Sample up to 10/class with `manifest_seed` (no replacement).",
        "5. If a class has fewer than 10 eligible ICs, **report shortfall** and use the actual count — never recycle train ICs.",
        "",
        "## Rollout rules",
        "",
        "- Exactly one full-episode rollout per selected IC.",
        "- Predefined seed via `rollout_seed(split=final_independent_small, traj_id, rollout_i=0)`.",
        "- No checkpoint selection on these outcomes; no mid-run retuning.",
        "- Infra errors logged; not silently replaced with different ICs.",
        "",
        "## Primary metrics",
        "",
        "- Per-class Episode Success Rate + 95% Wilson CI",
        "- Overall = equal-weight mean of class SRs + stratified bootstrap CI",
        "",
        "## Hard constraints",
        "",
    ]
    for c in manifest.get("constraints") or []:
        lines.append(f"- {c}")
    lines.append("")
    (OUT / "EVAL_PROTOCOL.md").write_text("\n".join(lines), encoding="utf-8")


def write_leakage_audit(manifest: dict) -> None:
    lines = [
        "# DATA_LEAKAGE_AUDIT",
        "",
        "## Exclusion sets (sizes)",
        "",
        "```json",
        json.dumps(manifest.get("exclusion_set_sizes"), indent=2),
        "```",
        "",
        "## Pool summary by class",
        "",
        "```json",
        json.dumps(manifest.get("pool_summary"), indent=2),
        "```",
        "",
        "## Shortfalls vs target 10/class",
        "",
        "```json",
        json.dumps(manifest.get("gaps"), indent=2),
        "```",
        "",
    ]
    if not manifest.get("can_meet_10_per_class"):
        lines += [
            "## Gap report (blocking padding)",
            "",
            "At least one class cannot supply 10 genuinely independent unused heldout ICs.",
            "**B1** heldout loadable unique roots after exclusions are insufficient for target 10.",
            "We **do not** recycle training ICs or diagnostic-panel ICs to pad.",
            "Evaluation proceeds with **actual available denominators** and labels the study as",
            "**small-scale independent evaluation** with sample-size limits.",
            "",
            "Root cause for B1: only 16 loadable heldout starts exist; 8 were consumed by P61/P62",
            "dev/final-test slices (`blocked_eval`), leaving **8** eligible — shortfall **2**.",
            "",
        ]
    lines += [
        "## Selected episode IDs (frozen)",
        "",
        "```json",
        json.dumps(
            [
                {"skill": e["skill"], "traj_id": e["traj_id"], "root_id": e["root_id"], "seed": e["seed"]}
                for e in manifest.get("episodes") or []
            ],
            indent=2,
        ),
        "```",
        "",
        "## Duplicate root check",
        "",
    ]
    roots = [e["root_id"] for e in manifest.get("episodes") or []]
    lines.append(f"- n_episodes={len(roots)}; n_unique_roots={len(set(roots))}")
    lines.append(
        "- PASS"
        if len(roots) == len(set(roots))
        else "- FAIL: duplicate roots in manifest"
    )
    lines.append("")
    (OUT / "DATA_LEAKAGE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
