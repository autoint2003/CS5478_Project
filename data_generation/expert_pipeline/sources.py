"""Discover post-Stage-1 ICs for each of the seven expert branches.

ICs without Stage-1 SUCCESS_12S evidence (registry) are excluded from discovery
when ``require_stage1_evidence`` is True (default for production generation).

Exception: ``D_PRECISION`` (HARD_IC) has no Stage-1 A/B/C prefix — the Stage-1
registry gate is skipped for that branch. Live D2 eligibility
(``d2_eligibility.check_d2_eligible``) runs at generate time instead.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data_generation.expert_pipeline.branches import BRANCHES, select_branch
from data_generation.oracles.precision.stage1_eligibility import (
    capture_fingerprint,
    lookup_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "generation" / "experts_v2.yaml"


@dataclass(frozen=True)
class SourceIC:
    sample_id: str
    branch_id: str
    plan_provenance: str
    ic_path: Path
    final_category: str | None
    source_pool: str | None

    @property
    def source_ic_id(self) -> str:
        return self.sample_id


def load_generation_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid generation config: {cfg_path}")
    return raw


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _ic_path(ics_dir: Path, sample_id: str) -> Path | None:
    direct = ics_dir / f"{sample_id}.pkl"
    if direct.is_file():
        return direct
    hits = sorted(ics_dir.glob(f"{sample_id}*.pkl"))
    return hits[0] if hits else None


def discover_branch_ics(
    branch_id: str,
    config: dict[str, Any] | None = None,
    *,
    require_stage1_evidence: bool | None = None,
) -> list[SourceIC]:
    """Return deterministic IC list for ``branch_id`` from the authoritative manifest.

    When ``require_stage1_evidence`` is True (default unless config sets
    ``require_stage1_evidence: false``), A/B/C ICs whose capture fingerprint
    lacks SUCCESS_12S=True registry evidence are skipped. ``D_PRECISION`` always
    skips the Stage-1 registry (no A/B/C prefix). Live revalidation still occurs
    at generate time (Stage-1 for A/B/C, D2 gate for D).
    """
    cfg = config or load_generation_config()
    branch = select_branch(branch_id)
    spec = (cfg.get("branches") or {}).get(branch.branch_id)
    if not spec:
        raise KeyError(f"branch {branch.branch_id} missing from generation config")
    if require_stage1_evidence is None:
        require_stage1_evidence = bool(cfg.get("require_stage1_evidence", True))
    # D2: no Stage-1 A/B/C recovery — registry evidence is conceptually wrong.
    if branch.stage1 is None:
        require_stage1_evidence = False
    allowed = set(spec.get("plan_provenance") or [])
    manifest = _resolve(str(spec.get("source_manifest") or cfg["source_manifest"]))
    ics_dir = _resolve(str(spec.get("source_ics") or cfg["source_ics"]))
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    out: list[SourceIC] = []
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        plan = str(row.get("plan_provenance") or "")
        if plan not in allowed:
            continue
        sid = str(row["sample_id"])
        if sid in seen:
            continue
        path = _ic_path(ics_dir, sid)
        if path is None:
            continue
        if require_stage1_evidence:
            try:
                blob = pickle.loads(path.read_bytes())
                fp = capture_fingerprint(blob["capture"])
                ev = lookup_evidence(fp)
                if ev is None or not bool(ev.get("SUCCESS_12S")):
                    continue
            except Exception:
                continue
        seen.add(sid)
        out.append(
            SourceIC(
                sample_id=sid,
                branch_id=branch.branch_id,
                plan_provenance=plan,
                ic_path=path,
                final_category=row.get("final_category"),
                source_pool=row.get("source_pool"),
            )
        )
    out.sort(key=lambda s: s.sample_id)
    return out


def discover_all(config: dict[str, Any] | None = None) -> dict[str, list[SourceIC]]:
    cfg = config or load_generation_config()
    return {bid: discover_branch_ics(bid, cfg) for bid in BRANCHES}
