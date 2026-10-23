#!/usr/bin/env python
"""One-shot naming refactor: git-mv + import path rewrites (active tree only).

Does not touch results/. Does not modify algorithm bodies.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# old relative path → new relative path
MOVES: list[tuple[str, str]] = [
    # Plan C oracles
    ("data_generation/strategies/ground_future_oracle.py", "data_generation/oracles/plan_c/future_trajectory.py"),
    ("data_generation/strategies/ground_early_recovery.py", "data_generation/oracles/plan_c/interception.py"),
    ("data_generation/strategies/ground_dynamic_catch.py", "data_generation/oracles/plan_c/dynamic_catch.py"),
    ("data_generation/strategies/ground_postcatch_endgrasp.py", "data_generation/oracles/plan_c/catch_stabilization.py"),
    # Plan A oracles
    ("data_generation/strategies/k48_controlled_slide_search.py", "data_generation/oracles/plan_a/controlled_sliding.py"),
    ("data_generation/strategies/k48_inward_sliding_search.py", "data_generation/oracles/plan_a/inward_reseat.py"),
    ("data_generation/strategies/capture_geometry.py", "data_generation/oracles/plan_a/contact_geometry.py"),
    ("data_generation/strategies/recovery_laws.py", "data_generation/oracles/plan_a/recovery_laws.py"),
    # Plan B oracles
    ("data_generation/strategies/recatch_oracle.py", "data_generation/oracles/plan_b/release_recatch.py"),
    ("data_generation/strategies/vertical_recatch_search.py", "data_generation/oracles/plan_b/vertical_recatch.py"),
    # Dataset construction
    ("data_generation/strategies/build_open_overchase_recatch_dataset.py", "data_generation/datasets/open_overchase_recatch.py"),
    ("data_generation/strategies/build_vertical_recatch_dataset.py", "data_generation/datasets/vertical_recatch.py"),
    ("data_generation/airborne_recatch_dataset.py", "data_generation/datasets/airborne_recatch.py"),
    ("data_generation/ball_impact_dataset.py", "data_generation/datasets/ball_impact.py"),
    ("data_generation/autonomous_episodes.py", "data_generation/datasets/autonomous_episodes.py"),
    ("data_generation/expand_skill_quotas.py", "data_generation/datasets/skill_quotas.py"),
    ("data_generation/skill_filters.py", "data_generation/datasets/skill_filters.py"),
    ("data_generation/sliding_labels.py", "data_generation/datasets/sliding_labels.py"),
    ("data_generation/rebuild_k48_recovery.py", "data_generation/datasets/recovery_rebuild.py"),
    ("data_generation/action_kmeans.py", "data_generation/datasets/action_kmeans.py"),
    ("data_generation/policy_rollout.py", "data_generation/datasets/policy_rollout.py"),
    # Maintenance → experiments
    ("data_generation/maintenance/align_data_layout.py", "experiments/maintenance/align_data_layout.py"),
    ("data_generation/maintenance/consolidate_snapshots.py", "experiments/maintenance/consolidate_snapshots.py"),
    ("data_generation/maintenance/relocate_training_data.py", "experiments/maintenance/relocate_training_data.py"),
    # Models / training (active functional names; keep K48 format loaders truthful)
    ("models/airborne_v3.py", "models/transformer_policy.py"),
    ("training/train_sft.py", "training/sft.py"),
    ("training/train_grpo.py", "training/grpo.py"),
    ("training/unified_k48.py", "training/pipeline.py"),
    ("training/reward_v2.py", "training/rewards.py"),
    ("training/prefix_rollout.py", "training/rollout.py"),
    # Evaluation reusable
    ("evaluation/full_objective.py", "evaluation/success_criteria.py"),
    ("evaluation/grasp_configuration.py", "evaluation/grasp_geometry.py"),
    ("evaluation/verify_handoff_targets.py", "evaluation/handoff_validation.py"),
    ("evaluation/k48_eval.py", "evaluation/policy_evaluation.py"),
    # Scripts: active entry points
    ("scripts/build_unified_dataset.py", "scripts/build_expert_dataset.py"),
    ("scripts/replay_golden_experts.py", "scripts/replay_experts.py"),
    ("scripts/audit_k49.py", "scripts/audit_action_vocab.py"),
    ("scripts/train.py", "scripts/train_policy.py"),
    ("scripts/evaluate.py", "scripts/evaluate_policy.py"),
    ("scripts/render_video.py", "scripts/render_trajectory.py"),
    ("scripts/generate_data.py", "scripts/generate_experts.py"),
    ("scripts/generate_plan_c_experts.py", "scripts/generate_plan_c_experts.py"),
    # Consolidation helpers stay but rename freeze entry clarity
    ("scripts/consolidation/freeze_experts.py", "scripts/consolidation/freeze_experts.py"),
    ("scripts/consolidation/audit_token_representability.py", "scripts/consolidation/audit_token_representability.py"),
]

# Archive experimental runners / historical audits / configs
ARCHIVE_SCRIPTS = [
    "scripts/run_ground_ball_impact_capture_refinement.py",
    "scripts/run_ground_ball_impact_early_recovery.py",
    "scripts/run_ground_ball_impact_oracle_interception.py",
    "scripts/run_ground_dynamic_catch_oracle.py",
    "scripts/run_ground_postcatch_endgrasp_recovery.py",
    "scripts/run_ground_scene_validation.py",
    "scripts/run_plan_a_ground_scene_divergence.py",
    "scripts/run_plan_a_sliding_pivoting_legality_audit.py",
    "scripts/run_plan_c_horizontal_downward_end_recatch.py",
    "scripts/run_active_recatch_generalization_discretization.py",
    "scripts/run_contact_geometry_table_window.py",
    "scripts/run_controlled_slow_detachment.py",
    "scripts/run_free_fall_cylinder_catching.py",
    "scripts/run_freefall_e_cpp_unload_isolation.py",
    "scripts/run_freefall_e_postcapture_stabilize_slide.py",
    "scripts/run_freefall_e_stable_lift_contact_manip.py",
    "scripts/run_freefall_recatch_collision_aware_feedback.py",
    "scripts/run_freefall_recatch_grasp_point_alignment.py",
    "scripts/run_freefall_recatch_pose_feedback.py",
    "scripts/run_horizontal_offset_freefall_recatch.py",
    "scripts/run_k49_action_authority_ablation.py",
    "scripts/run_lateral_force_full_pose_recovery.py",
    "scripts/run_lateral_force_grasp_loss_recovery.py",
    "scripts/run_lateral_force_mechanism_audit.py",
    "scripts/run_lateral_offset_falling_recatch.py",
    "scripts/run_lateral_recatch_handoff.py",
    "scripts/run_real_ball_impact_low_speed_recatch.py",
    "scripts/run_velocity_matched_soft_catch.py",
    "scripts/expand_k49_capture_brake_diagnostics.py",
    "scripts/expand_k49_hard_dynamic.py",
    "scripts/expand_k49_large_chase_dynamic.py",
    "scripts/expand_k49_unified_dataset.py",
    "scripts/generate_grasp_recovery_pose_experts_v1.py",
    "scripts/generate_k49_ball_impact_dynamic_recovery.py",
    "scripts/generate_k49_dynamic_recatch_experts.py",
    "scripts/generate_k49_horizontal_endcatch_recatch.py",
    "scripts/generate_k49_in_recovery_recatch.py",
    "scripts/generate_k49_lost_grasp_recovery.py",
    "scripts/generate_k49_sliding_recatch_dataset.py",
    "scripts/generate_k49_type3_capture_brake.py",
    "scripts/merge_grpo_ic_datasets.py",
]

ARCHIVE_EVAL = [
    "evaluation/audit_grasp_configuration_migration.py",
    "evaluation/audit_mechanism_selection_feasibility.py",
    "evaluation/audit_pretrain_data.py",
    "evaluation/audit_recatch_legality.py",
    "evaluation/classify_validation_skills.py",
    "evaluation/skill_traces.py",
    "evaluation/plan_a_mechanism_audit.py",
]

ARCHIVE_DOCS = [
    "docs/FINAL_REFACTOR_CHECK.md",
    "docs/HISTORICAL_K32_README.md",
    "docs/MIGRATION.md",
    "docs/REFACTOR_AUDIT.md",
]

ARCHIVE_CONFIG_DIRS = [
    ("config/training/k47", "experiments/config/training/k47"),
    ("config/training/legacy", "experiments/config/training/legacy"),
]

# Import string replacements (longest first)
IMPORT_REPLACEMENTS: list[tuple[str, str]] = [
    ("data_generation.oracles.plan_c.future_trajectory", "data_generation.oracles.plan_c.future_trajectory"),
    ("data_generation.oracles.plan_c.interception", "data_generation.oracles.plan_c.interception"),
    ("data_generation.oracles.plan_c.dynamic_catch", "data_generation.oracles.plan_c.dynamic_catch"),
    ("data_generation.oracles.plan_c.catch_stabilization", "data_generation.oracles.plan_c.catch_stabilization"),
    ("data_generation.oracles.plan_a.controlled_sliding", "data_generation.oracles.plan_a.controlled_sliding"),
    ("data_generation.oracles.plan_a.inward_reseat", "data_generation.oracles.plan_a.inward_reseat"),
    ("data_generation.oracles.plan_a.contact_geometry", "data_generation.oracles.plan_a.contact_geometry"),
    ("data_generation.oracles.plan_a.recovery_laws", "data_generation.oracles.plan_a.recovery_laws"),
    ("data_generation.oracles.plan_b.release_recatch", "data_generation.oracles.plan_b.release_recatch"),
    ("data_generation.oracles.plan_b.vertical_recatch", "data_generation.oracles.plan_b.vertical_recatch"),
    ("data_generation.datasets.open_overchase_recatch", "data_generation.datasets.open_overchase_recatch"),
    ("data_generation.datasets.vertical_recatch", "data_generation.datasets.vertical_recatch"),
    ("data_generation.datasets.airborne_recatch", "data_generation.datasets.airborne_recatch"),
    ("data_generation.datasets.ball_impact", "data_generation.datasets.ball_impact"),
    ("data_generation.datasets.autonomous_episodes", "data_generation.datasets.autonomous_episodes"),
    ("data_generation.datasets.skill_quotas", "data_generation.datasets.skill_quotas"),
    ("data_generation.datasets.skill_filters", "data_generation.datasets.skill_filters"),
    ("data_generation.datasets.sliding_labels", "data_generation.datasets.sliding_labels"),
    ("data_generation.datasets.recovery_rebuild", "data_generation.datasets.recovery_rebuild"),
    ("data_generation.datasets.action_kmeans", "data_generation.datasets.action_kmeans"),
    ("data_generation.datasets.policy_rollout", "data_generation.datasets.policy_rollout"),
    ("models.transformer_policy", "models.transformer_policy"),
    ("training.sft", "training.sft"),
    ("training.grpo", "training.grpo"),
    ("training.pipeline", "training.pipeline"),
    ("training.rewards", "training.rewards"),
    ("training.rollout", "training.rollout"),
    ("evaluation.success_criteria", "evaluation.success_criteria"),
    ("evaluation.grasp_geometry", "evaluation.grasp_geometry"),
    ("evaluation.handoff_validation", "evaluation.handoff_validation"),
    ("evaluation.policy_evaluation", "evaluation.policy_evaluation"),
    ("experiments.audits.plan_a_mechanism_audit", "experiments.audits.plan_a_mechanism_audit"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def ensure_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.py"
    if not init.exists():
        init.write_text('"""Package."""\n', encoding="utf-8")
        run(["git", "add", str(init.relative_to(ROOT))])


def git_mv(src: str, dst: str) -> None:
    sp, dp = ROOT / src, ROOT / dst
    if not sp.exists():
        print(f"SKIP missing {src}", flush=True)
        return
    dp.parent.mkdir(parents=True, exist_ok=True)
    if (ROOT / dst).exists():
        print(f"SKIP exists {dst}", flush=True)
        return
    run(["git", "mv", src, dst])


def rewrite_imports() -> int:
    n = 0
    text_ext = {".py", ".md", ".yaml", ".yml", ".toml", ".txt"}
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in text_ext:
            continue
        if "results" in p.parts or ".git" in p.parts:
            continue
        if p.name.startswith("_apply_naming"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        orig = text
        for old, new in sorted(IMPORT_REPLACEMENTS, key=lambda x: -len(x[0])):
            text = text.replace(old, new)
        # script path string updates
        text = text.replace("scripts/generate_plan_c_experts.py", "scripts/generate_plan_c_experts.py")
        text = text.replace("scripts/build_unified_dataset.py", "scripts/build_expert_dataset.py")
        text = text.replace("scripts/replay_golden_experts.py", "scripts/replay_experts.py")
        text = text.replace("scripts/audit_k49.py", "scripts/audit_action_vocab.py")
        text = text.replace("scripts/train.py", "scripts/train_policy.py")
        text = text.replace("scripts/evaluate.py", "scripts/evaluate_policy.py")
        text = text.replace("scripts/render_video.py", "scripts/render_trajectory.py")
        text = text.replace("scripts/generate_data.py", "scripts/generate_experts.py")
        text = text.replace("scripts/consolidation/freeze_experts.py", "scripts/consolidation/freeze_experts.py")
        text = text.replace(
            "scripts/consolidation/audit_token_representability.py",
            "scripts/consolidation/audit_token_representability.py",
        )
        text = text.replace("models/airborne_v3.py", "models/transformer_policy.py")
        text = text.replace("evaluation/full_objective.py", "evaluation/success_criteria.py")
        if text != orig:
            p.write_text(text, encoding="utf-8")
            n += 1
    return n


def main() -> None:
    # Package inits
    for rel in [
        "data_generation/oracles/plan_a",
        "data_generation/oracles/plan_b",
        "data_generation/oracles/plan_c",
        "data_generation/datasets",
        "experiments/maintenance",
        "experiments/audits",
        "experiments/diagnostics",
        "experiments/config/training",
        "experiments/docs",
    ]:
        ensure_init(ROOT / rel)

    for src, dst in MOVES:
        git_mv(src, dst)

    for src in ARCHIVE_SCRIPTS:
        name = Path(src).name
        git_mv(src, f"experiments/diagnostics/{name}")

    for src in ARCHIVE_EVAL:
        name = Path(src).name
        git_mv(src, f"experiments/audits/{name}")

    for src in ARCHIVE_DOCS:
        name = Path(src).name
        git_mv(src, f"experiments/docs/{name}")

    for src, dst in ARCHIVE_CONFIG_DIRS:
        if (ROOT / src).exists() and not (ROOT / dst).exists():
            (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
            run(["git", "mv", src, dst])

    # Historical k48 yamls except keep a pointer note — move all k48 experimental to experiments
    k48 = ROOT / "config/training/k48"
    if k48.is_dir():
        dest = ROOT / "experiments/config/training/k48"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            run(["git", "mv", "config/training/k48", "experiments/config/training/k48"])

    n = rewrite_imports()
    print(f"rewrote imports in {n} files", flush=True)

    # Remove empty maintenance package if hollow
    maint = ROOT / "data_generation/maintenance"
    if maint.is_dir() and not any(p.suffix == ".py" and p.name != "__init__.py" for p in maint.rglob("*.py")):
        init = maint / "__init__.py"
        if init.exists():
            run(["git", "rm", "-f", "data_generation/maintenance/__init__.py"])
        # leave dir removal to git

    mapping = ROOT / "docs" / "RENAME_MAP.md"
    lines = ["# Naming refactor map", "", "| Old path | New path |", "|---|---|"]
    for a, b in MOVES:
        lines.append(f"| `{a}` | `{b}` |")
    lines.append("")
    lines.append("## Archived to experiments/")
    lines.append("")
    for s in ARCHIVE_SCRIPTS:
        lines.append(f"- `{s}` → `experiments/diagnostics/{Path(s).name}`")
    for s in ARCHIVE_EVAL:
        lines.append(f"- `{s}` → `experiments/audits/{Path(s).name}`")
    for s in ARCHIVE_DOCS:
        lines.append(f"- `{s}` → `experiments/docs/{Path(s).name}`")
    lines.append("- `config/training/k47|k48|legacy` → `experiments/config/training/`")
    mapping.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {mapping}", flush=True)


if __name__ == "__main__":
    main()
