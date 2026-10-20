"""Move control to immediate_handoff and put every snapshot in data/snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def rename_control() -> None:
    source = DATA / "control"
    destination = DATA / "immediate_handoff"
    if source.exists():
        if destination.exists():
            raise SystemExit("data/immediate_handoff already exists")
        source.rename(destination)
    for path in DATA.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "data/control/" not in text:
            continue
        path.write_text(text.replace("data/control/", "data/immediate_handoff/"), encoding="utf-8")


def _place(path: Path, split: str) -> None:
    destination = DATA / "snapshots" / split / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != path.read_bytes():
            raise SystemExit(f"snapshot conflict {destination}")
        path.unlink()
        return
    shutil.move(str(path), str(destination))


def consolidate(folder: Path) -> None:
    for path in list(folder.rglob("*.pkl")):
        _place(path, path.parent.name)
    if folder.exists():
        for path in sorted(folder.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        folder.rmdir()


def retarget(manifest: Path) -> None:
    rows = _rows(manifest)
    for row in rows:
        name = Path(row["snapshot"]).name
        row["snapshot"] = f"data/snapshots/{row['split']}/{name}"
        source = str(row.get("source_snapshot") or "")
        if source.startswith("data/inward_sliding/snapshots/") or source.startswith("data/recatch/snapshots/"):
            row["source_snapshot"] = row["snapshot"]
    _dump(manifest, rows)


def main() -> None:
    rename_control()
    consolidate(DATA / "inward_sliding" / "snapshots")
    consolidate(DATA / "recatch" / "snapshots")
    retarget(DATA / "inward_sliding" / "manifest.jsonl")
    retarget(DATA / "recatch" / "manifest.jsonl")
    entry_path = DATA / "entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry.setdefault("categories", {})
    entry["categories"].pop("control", None)
    entry["categories"]["immediate_handoff"] = "data/immediate_handoff/manifest.jsonl"
    entry_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print("moved")


if __name__ == "__main__":
    main()
