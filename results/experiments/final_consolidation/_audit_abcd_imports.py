"""AST-check experiments.* imports under P51-P84 resolve to existing modules."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREC = ROOT / "experiments" / "precision"


def module_path(mod: str) -> Path | None:
    # experiments.precision.p51_... -> file or package
    parts = mod.split(".")
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    return None


def main() -> None:
    pkgs = sorted(
        [
            p
            for p in PREC.iterdir()
            if p.is_dir() and re.match(r"^p(5[1-9]|6[0-9]|7[0-9]|8[0-4])", p.name)
        ],
        key=lambda p: int(re.match(r"p(\d+)", p.name).group(1)),
    )
    missing: list[str] = []
    checked = 0
    for p in pkgs:
        for f in p.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            tree = ast.parse(f.read_text(encoding="utf-8"))
            mods: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("experiments."):
                        mods.add(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("experiments."):
                            mods.add(alias.name)
            for m in mods:
                checked += 1
                # For from experiments.x.y import z, module may be package or submodule
                if module_path(m) is None:
                    # try parent if importing name from package
                    missing.append(f"{p.name}/{f.name}: missing {m}")
    print("checked_import_modules", checked)
    print("missing", len(missing))
    for line in missing[:40]:
        print(line)


if __name__ == "__main__":
    main()
