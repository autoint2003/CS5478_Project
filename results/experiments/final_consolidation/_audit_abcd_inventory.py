"""Read-only inventory of P51-P84 experiment packages."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREC = ROOT / "experiments" / "precision"


def main() -> None:
    pkgs = sorted(
        [
            p
            for p in PREC.iterdir()
            if p.is_dir() and re.match(r"^p(5[1-9]|6[0-9]|7[0-9]|8[0-4])", p.name)
        ],
        key=lambda p: int(re.match(r"p(\d+)", p.name).group(1)),
    )
    print("packages", len(pkgs))
    for p in pkgs:
        pys = [x for x in p.rglob("*.py") if "__pycache__" not in str(x)]
        entry = [x.name for x in pys if x.name.startswith("run_") or x.name == "main.py"]
        has_skills = False
        skills_vals: set[str] = set()
        syntax_err: list[str] = []
        for f in pys:
            try:
                t = f.read_text(encoding="utf-8")
                ast.parse(t)
            except SyntaxError as e:
                syntax_err.append(f"{f.name}:{e}")
                continue
            if "SKILLS" in t and "A1" in t and "B1" in t:
                has_skills = True
            for pat in (r"SKILLS\s*=\s*\(([^)]+)\)", r"SKILLS\s*=\s*\[([^\]]+)\]"):
                m = re.search(pat, t)
                if m:
                    skills_vals.add(re.sub(r"['\"\s]", "", m.group(1)))
        print(
            f"{p.name}\tpy={len(pys)}\tentry={entry or ['-']}\t"
            f"skills={has_skills}\tvals={skills_vals or '-'}\terr={syntax_err[:1]}"
        )


if __name__ == "__main__":
    main()
