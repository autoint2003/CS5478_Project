#!/usr/bin/env python
"""Archive historical diagnostics with checksums; preserve golden Plan C evidence.

Does not delete originals until integrity is verified. Large frame dumps stay
in-place under results/diagnostics (gitignored) and are recorded as in-situ.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DIAG = ROOT / "results" / "diagnostics"
ARCHIVE = ROOT / "archive"
GOLDEN = ARCHIVE / "golden_experts"
EXP = ARCHIVE / "experiments"
MANIFEST = ROOT / "docs" / "ARCHIVE_MANIFEST.md"
MANIFEST_JSON = ARCHIVE / "archive_manifest.json"

# Golden Plan C evidence (must remain recoverable).
GOLDEN_DIRS = {
    "ground_plan_c_final_success": {
        "description": "Plan C final SUCCESS_NEW experts gbi_010 / gbi_022",
        "verified_experts": ["gbi_010", "gbi_022"],
        "reproducible": True,
    },
    "ground_dynamic_catch_oracle": {
        "description": "Plan C dynamic airborne catch tapes (prefix for final experts)",
        "verified_experts": ["gbi_010", "gbi_022"],
        "reproducible": True,
    },
    "ground_postcatch_endgrasp_recovery": {
        "description": "Plan C brake/slow-lift end-grasp recovery tapes",
        "verified_experts": ["gbi_010", "gbi_022"],
        "reproducible": True,
    },
}

# Essential suffixes to copy into archive/golden (skip per-frame PNGs).
GOLDEN_COPY_GLOBS = [
    "*.csv",
    "*.md",
    "*.json",
    "raw/*.pkl",
    "raw/*.json",
    "videos/*.mp4",
]

# Plan A legality remains active regression evidence (copy, do not relocate).
ACTIVE_KEEP = {
    "plan_a_sliding_pivoting_legality",
    "plan_a_ground_scene_divergence",
    "ground_scene_validation",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def copy_glob(src: Path, dst: Path, pattern: str) -> list[Path]:
    copied = []
    for p in src.glob(pattern):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied.append(out)
    return copied


def archive_golden(commit: str) -> list[dict]:
    rows = []
    for name, meta in GOLDEN_DIRS.items():
        src = DIAG / name
        if not src.is_dir():
            rows.append({
                "original_path": str(src.relative_to(ROOT)).replace("\\", "/"),
                "archive_path": None,
                "status": "MISSING",
                **meta,
            })
            continue
        dst = GOLDEN / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        files = []
        for g in GOLDEN_COPY_GLOBS:
            files.extend(copy_glob(src, dst, g))
        # Integrity: checksum essential pkls / reports
        checksums = {}
        for p in sorted(files):
            if p.suffix.lower() in {".pkl", ".csv", ".md", ".json", ".mp4"}:
                checksums[str(p.relative_to(ARCHIVE)).replace("\\", "/")] = {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
        # Verify copy matches source for SUCCESS experts
        for rid in meta["verified_experts"]:
            src_pkl = src / "raw" / f"{rid}__SUCCESS_NEW_expert.pkl"
            if not src_pkl.exists():
                # catch / endgrasp naming
                alt = list((src / "raw").glob(f"{rid}__*.pkl"))
                for a in alt:
                    dst_p = dst / "raw" / a.name
                    if dst_p.exists():
                        assert sha256_file(a) == sha256_file(dst_p)
                continue
            dst_pkl = dst / "raw" / src_pkl.name
            assert dst_pkl.exists(), dst_pkl
            assert sha256_file(src_pkl) == sha256_file(dst_pkl)
        rows.append({
            "original_path": str(src.relative_to(ROOT)).replace("\\", "/"),
            "archive_path": str(dst.relative_to(ROOT)).replace("\\", "/"),
            "experiment_description": meta["description"],
            "git_commit": commit,
            "file_size_bytes": dir_size(dst),
            "checksums": checksums,
            "reproducible": meta["reproducible"],
            "contains_verified_expert": True,
            "verified_experts": meta["verified_experts"],
            "replay_deps": [
                "envs/simulation/ground_sim.py",
                "envs/step_legal.py",
                "data_generation/strategies/ground_*.py",
                "scripts/run_ground_plan_c_final_success.py",
            ],
            "status": "COPIED_VERIFIED",
            "in_situ_original_preserved": True,
        })
    return rows


def index_diagnostics(commit: str) -> list[dict]:
    rows = []
    if not DIAG.is_dir():
        return rows
    for d in sorted(DIAG.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        kind = "golden" if name in GOLDEN_DIRS else (
            "active_keep" if name in ACTIVE_KEEP else "historical"
        )
        # Pointer archive entry (no mass binary move — preserves gitignored evidence)
        pointer = EXP / name
        pointer.mkdir(parents=True, exist_ok=True)
        meta = {
            "original_path": str(d.relative_to(ROOT)).replace("\\", "/"),
            "archive_path": str(pointer.relative_to(ROOT)).replace("\\", "/"),
            "experiment_description": f"Diagnostics folder '{name}' ({kind})",
            "git_commit": commit,
            "file_size_bytes": dir_size(d),
            "reproducible": kind in {"golden", "active_keep"},
            "contains_verified_expert": name in GOLDEN_DIRS,
            "status": "IN_SITU_INDEXED",
            "kind": kind,
            "note": (
                "Originals preserved under results/diagnostics (gitignored). "
                "Binary artifacts not duplicated into git history."
            ),
        }
        # Copy lightweight reports into archive/experiments/<name>/
        for pat in ("REPORT.md", "*.csv", "VERDICTS.csv", "FULL_EXPERTS.csv"):
            for p in d.glob(pat):
                if p.is_file() and p.stat().st_size < 5_000_000:
                    out = pointer / p.name
                    shutil.copy2(p, out)
                    meta.setdefault("checksums", {})[p.name] = {
                        "sha256": sha256_file(out),
                        "bytes": out.stat().st_size,
                    }
        (pointer / "ORIGINAL_LOCATION.txt").write_text(
            f"original={meta['original_path']}\nkind={kind}\ncommit={commit}\n",
            encoding="utf-8",
        )
        rows.append(meta)
    return rows


def write_manifest(entries: list[dict], commit: str) -> None:
    lines = [
        "# Archive Manifest",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Source commit: `{commit}`",
        f"Archival branch/tag: `archive/pre-consolidation-20261021` / `archive/pre-consolidation-20261021-src`",
        "",
        "## Policy",
        "",
        "- Git-ignored `results/diagnostics/` originals are **not deleted**.",
        "- Golden Plan C experts are **copied** to `archive/golden_experts/` with SHA-256 checks.",
        "- Other diagnostics are **indexed** under `archive/experiments/<name>/` with report copies.",
        "- Large video frame dumps stay in-situ to avoid bloating git history.",
        "",
        "## Golden experts",
        "",
        "| Original | Archive | Verified roots | Bytes | Status |",
        "|---|---|---|---:|---|",
    ]
    for e in entries:
        if e.get("contains_verified_expert") or e.get("kind") == "golden" or e.get("status") == "COPIED_VERIFIED":
            roots = ",".join(e.get("verified_experts") or [])
            lines.append(
                f"| `{e.get('original_path')}` | `{e.get('archive_path')}` | {roots} | "
                f"{e.get('file_size_bytes', 0)} | {e.get('status')} |"
            )
    lines += [
        "",
        "## Indexed diagnostics",
        "",
        "| Original | Kind | Bytes | Status |",
        "|---|---|---:|---|",
    ]
    for e in entries:
        if e.get("status") == "IN_SITU_INDEXED":
            lines.append(
                f"| `{e['original_path']}` | {e.get('kind')} | {e.get('file_size_bytes', 0)} | {e['status']} |"
            )
    lines += [
        "",
        "## Recoverability",
        "",
        "Restore source code from tag `archive/pre-consolidation-20261021-src`.",
        "Restore golden binaries from `archive/golden_experts/` or original `results/diagnostics/` paths.",
        "",
    ]
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    MANIFEST_JSON.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def main():
    commit = "7db29b3"
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    golden_rows = archive_golden(commit)
    index_rows = index_diagnostics(commit)
    # Prefer golden detail entries over duplicate index for same path
    merged = golden_rows + [
        r for r in index_rows
        if Path(r["original_path"]).name not in GOLDEN_DIRS
    ]
    write_manifest(merged, commit)
    print(f"Wrote {MANIFEST}")
    print(f"Golden copies: {sum(1 for r in golden_rows if r.get('status')=='COPIED_VERIFIED')}")
    print(f"Indexed diagnostics: {len(index_rows)}")


if __name__ == "__main__":
    main()
