"""Scan P51-P84 for auto-train launches and SKILLS / success-key usage."""

from __future__ import annotations

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
    print("=== auto launch / authorize patterns in decide/run ===")
    for p in pkgs:
        for f in p.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            if not any(k in f.name for k in ("decide", "run_p", "write_report", "write_final")):
                continue
            t = f.read_text(encoding="utf-8")
            hits = []
            for pat in (
                r"auto_start\s*=\s*True",
                r"authorize_training\s*=\s*True",
                r"recommend_train\s*=\s*True",
                r"subprocess\.(run|Popen|call)",
                r"os\.system\(",
            ):
                if re.search(pat, t):
                    hits.append(pat)
            if hits:
                print(f"{p.name}/{f.name}: {hits}")

    print("\n=== SUCCESS key mix (env write vs analysis read) ===")
    for p in pkgs:
        keys = {"SUCCESS_RELAXED_12S": 0, "SUCCESS_RELAXED": 0, "SUCCESS_NEW": 0, "SUCCESS_12S": 0}
        for f in p.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            t = f.read_text(encoding="utf-8")
            for k in keys:
                keys[k] += t.count(k)
        if any(keys.values()):
            print(f"{p.name}: {keys}")

    print("\n=== packages missing run_p* (other entry candidates) ===")
    for p in pkgs:
        pys = [x for x in p.glob("*.py")]
        runs = [x.name for x in pys if x.name.startswith("run_")]
        if not runs:
            cands = [
                x.name
                for x in pys
                if x.name.startswith(("train_", "eval_", "finalize_", "heldout_", "write_"))
            ]
            print(f"{p.name}: entry_candidates={cands}")


if __name__ == "__main__":
    main()
