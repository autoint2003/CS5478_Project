"""Move historical scripts and the results tree. Does not train or commit."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "training"
DEMO = ROOT / "demo"
KEEP_SEEDS = {
    "unified_k48",
    "unified_k47",
    "rebuild_k48_recovery",
    "verify_handoff_targets",
    "audit_pretrain_data",
    "test_reward_v2",
    "test_grpo_progress",
    "expand_controlled_sliding",
    "generate_inward_until",
    "expand_inward_training_data",
    "export_inward_training_data",
    "relocate_training_data",
}
# Historical comparison, not the --eval entry point.
FORCE_MOVE = {"eval_k48_compare"}
RENAMES = {
    "k48_expand": "expand_k48_handoff_dataset",
    "k48_policy": "build_k48_handoff_policy",
    "grav_reposition_v2_viz": "render_grav_reposition",
    "omega4_throw_test": "diagnose_omega4_throw",
    "eval_k48_compare": "compare_k48_policy_evaluations",
    "compound_orientation_visual_audit": "diagnose_compound_orientation",
}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".ps1", ".sh"}
PATH_RE = re.compile(r"(?<!demo_)results/archive/demo_results/")


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("training"):
            rest = node.module.split(".", 1)[1] if node.module != "training" else ""
            if rest:
                found.add(rest.split(".")[0])
            else:
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("training."):
                    found.add(alias.name.split(".", 1)[1].split(".")[0])
    return found


def closure() -> set[str]:
    files = {path.stem: path for path in TRAIN.glob("*.py")}
    graph = {stem: imports_of(path) & set(files) for stem, path in files.items()}
    keep = set(KEEP_SEEDS) & set(files)
    changed = True
    while changed:
        changed = False
        for name in list(keep):
            for dep in graph.get(name, ()):
                if dep not in keep and dep not in FORCE_MOVE:
                    keep.add(dep)
                    changed = True
    return keep


def git_tracked(path: Path) -> bool:
    probe = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT)).replace("\\", "/")],
        cwd=ROOT, capture_output=True, text=True,
    )
    return probe.returncode == 0


def move_sources(keep: set[str]) -> list[tuple[str, str]]:
    DEMO.mkdir(exist_ok=True)
    (DEMO / "__init__.py").write_text('"""Historical experiments and visualization scripts."""\n', encoding="utf-8")
    moved = []
    for path in sorted(TRAIN.glob("*.py")):
        if path.stem in keep:
            continue
        new_stem = RENAMES.get(path.stem, path.stem)
        dest = DEMO / f"{new_stem}.py"
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        if git_tracked(path):
            subprocess.run(["git", "mv", str(path.relative_to(ROOT)), str(dest.relative_to(ROOT))], cwd=ROOT, check=True)
        else:
            shutil.move(str(path), str(dest))
        moved.append((path.name, dest.name))
    return moved


def rewrite_imports(moved: list[tuple[str, str]]) -> int:
    mapping = {Path(old).stem: Path(new).stem for old, new in moved}
    changed = 0
    for path in list(ROOT.rglob("*.py")):
        if ".git" in path.parts or path.name == "refactor_repository.py":
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
            updated = re.sub(rf"\btraining\.{re.escape(old)}\b", f"demo.{new}", updated)
            updated = updated.replace(f"training/{old}.py", f"demo/{new}.py")
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def rewrite_historical_paths() -> int:
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if ".git" in path.parts or "data" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        updated = PATH_RE.sub("results/archive/demo_results/", text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def point_future_outputs() -> None:
    replacements = {
        ROOT / "config/training/k48/unified_k48_recovery_v3.yaml": {
            "supervised_output: results/archive/demo_results/diagnostics/raw/k48_recovery_v3/supervised": "supervised_output: results/archive/demo_results/sft",
            "eval_output: results/archive/demo_results/diagnostics/raw/k48_recovery_v3/eval": "eval_output: results/archive/demo_results/evaluations",
            "grpo_output: results/archive/demo_results/diagnostics/raw/k48_recovery_v3/grpo_v1": "grpo_output: results/archive/demo_results/grpo/v1",
        },
        ROOT / "config/training/k48/unified_k48_recovery_v3_reward_v2.yaml": {
            "supervised_output: results/archive/demo_results/diagnostics/raw/k48_recovery_v3/supervised": "supervised_output: results/archive/demo_results/sft",
            "eval_output: results/archive/demo_results/diagnostics/raw/k48_recovery_v3/eval": "eval_output: results/archive/demo_results/evaluations",
            "grpo_output: results/archive/demo_results/diagnostics/raw/k48_recovery_v3/grpo_v2": "grpo_output: results/archive/demo_results/grpo/v2",
        },
    }
    for path, mapping in replacements.items():
        text = path.read_text(encoding="utf-8")
        for old, new in mapping.items():
            if old not in text:
                raise SystemExit(f"missing output path in {path.name}: {old}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def move_results() -> tuple[int, int]:
    source = ROOT / "results"
    dest = ROOT / "results/archive/demo_results"
    if dest.exists():
        raise SystemExit("results/archive/demo_results already exists")
    before = sum(1 for _ in source.rglob("*") if _.is_file())
    shutil.move(str(source), str(dest))
    after = sum(1 for _ in dest.rglob("*") if _.is_file())
    fresh = ROOT / "results"
    for folder in ("sft", "grpo/v1", "grpo/v2", "evaluations"):
        (fresh / folder).mkdir(parents=True, exist_ok=True)
    return before, after


def main():
    keep = closure()
    print("KEEP", len(keep), flush=True)
    moved = move_sources(keep)
    print("MOVED", len(moved), flush=True)
    for old, new in moved:
        print(f"  {old} -> demo/{new}", flush=True)
    print("IMPORTS", rewrite_imports(moved), flush=True)
    print("PATHS", rewrite_historical_paths(), flush=True)
    point_future_outputs()
    before, after = move_results()
    print("RESULTS", before, after, flush=True)


if __name__ == "__main__":
    main()
