"""Align data/recatch and data/inward_sliding with the category manifest layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def align_recatch() -> None:
    folder = ROOT / "data" / "recatch"
    updated = []
    for row in _rows(folder / "manifest.jsonl"):
        stem = Path(row["continuous_npz"]).stem
        split = row["split"]
        source = ROOT / row["continuous_npz"]
        blob = np.load(source)
        if "tokens" not in blob.files:
            updated.append(row)
            continue
        tokens = np.asarray(blob["tokens"], np.int32)
        residual = np.asarray(blob["residual"], np.float32)
        np.savez_compressed(
            source,
            actions=np.asarray(blob["actions"], np.float32),
            observations=np.asarray(blob["observations"], np.float32),
            handoff_obs=np.asarray(blob["handoff_obs"], np.float32),
        )
        token_path = folder / split / f"{stem}__tokens.npz"
        np.savez_compressed(token_path, tokens=tokens, residuals=residual)
        snap_dir = folder / "snapshots" / split
        snap_dir.mkdir(parents=True, exist_ok=True)
        destination = snap_dir / f"{row['id']}.pkl"
        current = ROOT / row["snapshot"]
        if current.exists() and current.resolve() != destination.resolve():
            current.replace(destination)
        updated.append({
            "id": row["id"],
            "split": split,
            "strategy": row["strategy"],
            "family": "recatch",
            "impact_family": row.get("impact_family", row.get("family")),
            "mechanism": row["mechanism"],
            "continuous_npz": f"data/recatch/{split}/{stem}.npz",
            "token_residual_npz": f"data/recatch/{split}/{stem}__tokens.npz",
            "snapshot": f"data/snapshots/{split}/{row['id']}.pkl",
            "source_snapshot": f"data/snapshots/{split}/{row['id']}.pkl",
            "handoff_label": "SUCCESS_12S",
            "nominal_class": "RECOVERY_CANDIDATE",
            "provenance": "top_impact_open_overchase",
            "source_kind": "generated",
            "sample_weight": 1.0,
            "verified_success": True,
            "speed": row["speed"],
            "pitch": row["pitch"],
            "axial": row["axial"],
            "ecc": row["ecc"],
            "mass": row["mass"],
            "friction": row["friction"],
            "k48_linf": row["k48_linf"],
            "extension": row["extension"],
            "formal": row["formal"],
            "dz": row["dz"],
            "re_cz": row["re_cz"],
            "min_bottom": row["min_bottom"],
            "t_open": row["t_open"],
            "t_re": row["t_re"],
            "t_brake": row["t_brake"],
        })
    _dump(folder / "manifest.jsonl", updated)


def align_inward() -> None:
    path = ROOT / "data" / "inward_sliding" / "manifest.jsonl"
    rows = _rows(path)
    for row in rows:
        row["continuous_npz"] = row["trajectory"]
        row["token_residual_npz"] = row["tokens"]
        row.setdefault("family", "inward_sliding")
        row.setdefault("mechanism", "inward_reseat")
        row.setdefault("handoff_label", "SUCCESS_12S")
        row.setdefault("nominal_class", "RECOVERY_CANDIDATE")
        row.setdefault("provenance", "data/inward_sliding")
        row.setdefault("source_kind", "inward_sliding")
        row.setdefault("sample_weight", 1.0)
        row["verified_success"] = bool(row.get("success", row.get("verified_success", False)))
    _dump(path, rows)


def main() -> None:
    align_recatch()
    align_inward()
    print("aligned")


if __name__ == "__main__":
    main()
