"""IC cohort selection for LAP mechanism audit and paired A/B/C/D runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"

# Known successful Pivot / Precision controls (stable after Stage-1)
KNOWN_SUCCESS = [
    "C__postlift__gbi_112__30425714c8e115fe",
    "C__postlift__gbi_063__08584501db98836d",
    "B__B_post_lateral__a4bb242b95fcf0ae",
    "B__hard_910220__0412621ee978627d",
    "B__hard_910026__post_b1_v2__e3a32a342c39f52b",
]


def load_audit() -> list[dict[str, Any]]:
    return [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]


def e_signed(row: dict) -> tuple[float, float, float]:
    e = (row.get("physics_entry") or {}).get("e_signed_mm") or [0.0, 0.0, 0.0]
    return float(e[0]), float(e[1]), float(e[2])


def dtheta_failures(rows: list[dict] | None = None) -> list[dict]:
    rows = rows or load_audit()
    return [
        r
        for r in rows
        if any(str(f).startswith("dtheta:") for f in ((r.get("precision") or {}).get("failures") or []))
    ]


def large_dz_st_cases(rows: list[dict] | None = None, *, min_abs_dz_mm: float = 6.0) -> list[dict]:
    """Large |dz| with ST/dx activity — candidates for DX-induced rotation audit."""
    rows = rows or load_audit()
    out = []
    for r in rows:
        dx, _, dz = e_signed(r)
        if abs(dz) < min_abs_dz_mm:
            continue
        fails = [str(f) for f in ((r.get("precision") or {}).get("failures") or [])]
        fp = (r.get("precision") or {}).get("first_failed_primitive")
        st_ish = (
            fp in ("support_transfer", "DX1", "dx_support_transfer", "dz", "dtheta")
            or any("support_transfer" in f or "dtheta" in f for f in fails)
            or r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL"
        )
        if not st_ish and abs(dx) < 3.0:
            continue
        out.append(r)
    out.sort(key=lambda r: -abs(e_signed(r)[2]))
    return out


def success_controls(rows: list[dict] | None = None) -> list[dict]:
    rows = rows or load_audit()
    by_id = {r["sample_id"]: r for r in rows}
    out = [by_id[k] for k in KNOWN_SUCCESS if k in by_id]
    if len(out) < 5:
        succ = [r for r in rows if r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL"]
        for r in succ:
            if r["sample_id"] not in {x["sample_id"] for x in out}:
                out.append(r)
            if len(out) >= 8:
                break
    return out


def paired_cohort(
    *,
    n_dtheta: int = 10,
    n_large_dz: int = 8,
    n_success: int = 5,
) -> list[dict]:
    rows = load_audit()
    seen: set[str] = set()
    out: list[dict] = []
    for r in dtheta_failures(rows)[:n_dtheta] + large_dz_st_cases(rows)[:n_large_dz] + success_controls(rows)[
        :n_success
    ]:
        sid = r["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        dx, dy, dz = e_signed(r)
        r = dict(r)
        r["_cohort_dx_mm"] = dx
        r["_cohort_dy_mm"] = dy
        r["_cohort_dz_mm"] = dz
        r["_cohort_dtheta_deg"] = float((r.get("physics_entry") or {}).get("dtheta_deg") or 0.0)
        out.append(r)
    return out
