"""One-shot: classify post-cleanup new files into package folders.

Keeps durable generators in scripts/. Moves pilots/diagnoses/audits/tmp
into experiments/ or evaluation/ per docs/MODULE_ORGANIZATION.md.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# dest relative to ROOT -> list of filenames currently under scripts/
MOVES: dict[str, list[str]] = {
    "evaluation": [
        "audit_grasp_configuration_migration.py",
        "audit_mechanism_selection_feasibility.py",
    ],
    "experiments/handoff": [
        "audit_handoff_40ms_advantage.py",
        "audit_reward_time_handoff.py",
        "benchmark_return_to_nominal_pose_before_handoff.py",
        "benchmark_pose_correction_counterfactual.py",
        "diagnose_grpo_v2_regression.py",
    ],
    "experiments/sliding": [
        "pilot_sliding_position_correction_v1.py",
        "pilot_return_aware_sliding_k49.py",
        "pilot_k49_slow_return_position_refinement.py",
        "pilot_extended_k49_pose_recovery.py",
        "pilot_large_tilt_eccentric_grasp_mining.py",
        "pilot_gravity_assisted_pivoting.py",
        "pilot_wrist_tilt_gravity_pivoting.py",
        "pilot_nontrivial_recovery_ic_mining.py",
        "diagnose_k49_return_axial_slip.py",
        "diagnose_k49_slow_reverse_token48.py",
        "build_hard_recovery_benchmark.py",
        "_tmp_axial_overcomp_a2.py",
        "_tmp_axial_slip_extra_cf.py",
        "_tmp_mid_interleaved_return.py",
        "_tmp_partial_return_search.py",
        "_tmp_write_slow_refine_report.py",
        "_tmp_list_slide_recatch_ics.py",
        "_tmp_finalize_sliding_recatch_dataset.py",
        "_tmp_second_pass_near_threshold.py",
        "_tmp_pass3_recatch.py",
        "_tmp_pass4_mine_unilateral_snapshots.py",
        "_tmp_list_short_handoff_ics.py",
        "_tmp_verify_large_tilt_cpp.py",
    ],
    "experiments/recatch/k49": [
        "_tmp_finalize_in_recovery_recatch.py",
        "_tmp_verify_inrec_contact.py",
        "_tmp_finalize_dynamic_recatch.py",
        "_tmp_probe_type2_hard.py",
        "_tmp_finalize_horiz_endcatch.py",
        "_tmp_mine_horizontal_ics.py",
        "_tmp_probe_horiz_gen.py",
        "_tmp_analyze_horiz_results.py",
        "_tmp_finalize_capture_brake.py",
        "_tmp_verify_capture_brake.py",
        "_tmp_probe_brake_318.py",
        "_tmp_strong_rot_capture.py",
        "_tmp_lost_grasp_pass2.py",
        "_tmp_verify_lost_grasp.py",
        "_tmp_fix_lost_q.py",
    ],
    "experiments/ball_impact": [
        "_tmp_finalize_ball_impact_report.py",
        "_tmp_verify_ball_impact_experts.py",
    ],
    "experiments/adaptive_g": [
        "pilot_adaptive_g.py",
        "run_adaptive_g_full.py",
        "smoke_adaptive_g.py",
        "verify_adaptive_g_schedule.py",
        "eval_adaptive_g_baseline.py",
    ],
}

# After move: rewrite imports in relocated files
IMPORT_REWRITES = [
    ("from scripts.pilot_return_aware_sliding_k49 import",
     "from experiments.sliding.pilot_return_aware_sliding_k49 import"),
    ("from scripts.pilot_large_tilt_eccentric_grasp_mining import",
     "from experiments.sliding.pilot_large_tilt_eccentric_grasp_mining import"),
    ("from scripts.diagnose_k49_return_axial_slip import",
     "from experiments.sliding.diagnose_k49_return_axial_slip import"),
]


def ensure_init(dir_path: Path) -> None:
    init = dir_path / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


def main():
    moved = []
    missing = []
    for dest_rel, names in MOVES.items():
        dest = ROOT / dest_rel
        dest.mkdir(parents=True, exist_ok=True)
        ensure_init(dest)
        for name in names:
            src = ROOT / "scripts" / name
            if not src.exists():
                missing.append(str(src))
                continue
            dst = dest / name
            if dst.exists():
                print(f"SKIP exists: {dst}")
                continue
            shutil.move(str(src), str(dst))
            moved.append((f"scripts/{name}", f"{dest_rel}/{name}"))
            print(f"MOVE {name} -> {dest_rel}/")

    # rewrite imports inside newly moved trees
    scan_dirs = [
        ROOT / "experiments" / "sliding",
        ROOT / "experiments" / "recatch" / "k49",
        ROOT / "experiments" / "ball_impact",
        ROOT / "experiments" / "adaptive_g",
        ROOT / "experiments" / "handoff",
        ROOT / "evaluation",
    ]
    n_rewrite = 0
    for d in scan_dirs:
        if not d.exists():
            continue
        for py in d.glob("*.py"):
            text = py.read_text(encoding="utf-8")
            new = text
            for old, repl in IMPORT_REWRITES:
                if old in new:
                    new = new.replace(old, repl)
            if new != text:
                py.write_text(new, encoding="utf-8")
                n_rewrite += 1
                print(f"REWRITE imports: {py.relative_to(ROOT)}")

    # append migration log
    mig = ROOT / "docs" / "MIGRATION.md"
    lines = ["", "## K49 expert / adaptive-g cleanup (post Round 3)", ""]
    for a, b in moved:
        lines.append(f"- `{a}` -> `{b}`")
    lines.append("")
    with mig.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # update MODULE_ORGANIZATION tree snippet note
    org = ROOT / "docs" / "MODULE_ORGANIZATION.md"
    note = (
        "\n## 11. Post–Round-3 K49 / adaptive-g placement\n\n"
        "- Durable expert generators stay in `scripts/generate_k49_*.py` "
        "(and `generate_grasp_recovery_pose_experts_v1.py`).\n"
        "- Sliding pilots / K49 return diagnoses / slide-recatch one-shots: "
        "`experiments/sliding/`.\n"
        "- K49 Recatch / Type2–3 / lost-grasp one-shots: `experiments/recatch/k49/`.\n"
        "- Ball-impact verify/finalize one-shots: `experiments/ball_impact/`.\n"
        "- Handoff audits / pose-handoff benchmarks: `experiments/handoff/`.\n"
        "- Adaptive independent-g pilots: `experiments/adaptive_g/`.\n"
        "- Formal metric / migration audits: `evaluation/` "
        "(`full_objective.py`, `grasp_configuration.py`, "
        "`audit_grasp_configuration_migration.py`, "
        "`audit_mechanism_selection_feasibility.py`).\n"
        "- Training modules: `training/action_vocab_k49.py`, "
        "`training/adaptive_independent_g.py`; configs under "
        "`config/training/k48/*adaptive_g*.yaml`.\n"
    )
    text = org.read_text(encoding="utf-8")
    if "## 11. Post–Round-3" not in text:
        org.write_text(text.rstrip() + "\n" + note, encoding="utf-8")

    print(f"\nMoved {len(moved)} files; rewrote {n_rewrite}; missing {len(missing)}")
    for m in missing:
        print("  MISSING", m)


if __name__ == "__main__":
    main()
