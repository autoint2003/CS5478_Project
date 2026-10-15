"""Physical mechanism labels for the two in-hand sliding oracles.

``controlled_slide`` and ``inward_reseat`` stay as oracle names. Both belong
to one physical mechanism. Historical category directories are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MECHANISM = "gravity_assisted_in_hand_sliding"
UNIFIED = ROOT / "data" / "gravity_assisted_in_hand_sliding" / "manifest.jsonl"
ORACLE_CATEGORY = {
    "controlled_slide": "controlled_sliding",
    "inward_reseat": "inward_sliding",
}
CATEGORY_ORACLE = {value: key for key, value in ORACLE_CATEGORY.items()}


def mechanism_of(strategy_or_category: str) -> str | None:
    name = str(strategy_or_category or "")
    if name == MECHANISM or name in ORACLE_CATEGORY or name in CATEGORY_ORACLE:
        return MECHANISM
    return None


def oracle_of(row: dict) -> str:
    if row.get("oracle_strategy"):
        return str(row["oracle_strategy"])
    return str(row.get("strategy") or "")


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_unified(status_path: Path | None = None) -> list[dict]:
    status = {}
    path = status_path or (ROOT / "results" / "diagnostics" / "sliding_mechanism_audit" / "all_tape_slip.json")
    if path.is_file():
        for row in json.loads(path.read_text(encoding="utf-8")):
            status[(row["id"], row["strategy"])] = row["mechanism_status"]
    unified = []
    for category, oracle in CATEGORY_ORACLE.items():
        for row in load_rows(ROOT / "data" / category / "manifest.jsonl"):
            item = dict(row)
            item["strategy"] = oracle
            item["oracle_strategy"] = oracle
            item["mechanism"] = MECHANISM
            item["source_category"] = category
            item["mechanism_status"] = status.get((row["id"], oracle), "unscanned")
            unified.append(item)
    UNIFIED.parent.mkdir(parents=True, exist_ok=True)
    with UNIFIED.open("w", encoding="utf-8") as handle:
        for row in unified:
            handle.write(json.dumps(row) + "\n")
    return unified


def independent_ics(rows: list[dict], confirmed_only: bool = False) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        if confirmed_only and row.get("mechanism_status") != "confirmed":
            continue
        grouped.setdefault(row["snapshot"], set()).add(row["split"])
    return grouped
