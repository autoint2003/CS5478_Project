"""IC selection for OAR — reuse Precision audit pools; do not swap hard ICs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"

KNOWN_SUCCESS = [
    "C__postlift__gbi_112__30425714c8e115fe",
    "C__postlift__gbi_063__08584501db98836d",
    "B__B_post_lateral__a4bb242b95fcf0ae",
    "B__hard_910220__0412621ee978627d",
]


def load_audit() -> list[dict[str, Any]]:
    return [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]


def e_signed(row: dict) -> tuple[float, float, float]:
    e = (row.get("physics_entry") or {}).get("e_signed_mm") or [0.0, 0.0, 0.0]
    return float(e[0]), float(e[1]), float(e[2])


def oar_cohort(*, n_dtheta: int = 6, n_large_dz: int = 4, n_success: int = 3) -> list[dict]:
    rows = load_audit()
    dth = [
        r
        for r in rows
        if any(str(f).startswith("dtheta:") for f in ((r.get("precision") or {}).get("failures") or []))
    ]
    large = []
    for r in rows:
        dx, _, dz = e_signed(r)
        dtheta = float((r.get("physics_entry") or {}).get("dtheta_deg") or 0)
        if abs(dz) >= 8.0 and (abs(dx) >= 3.0 or dtheta >= 5.0):
            large.append(r)
    large.sort(key=lambda r: -abs(e_signed(r)[2]))
    by_id = {r["sample_id"]: r for r in rows}
    succ = [by_id[k] for k in KNOWN_SUCCESS if k in by_id]
    if len(succ) < n_success:
        for r in rows:
            if r.get("class") == "PRECISION_REQUIRED_AND_SUCCESSFUL" and r["sample_id"] not in {
                x["sample_id"] for x in succ
            }:
                succ.append(r)
            if len(succ) >= n_success:
                break

    out, seen = [], set()
    for r in dth[:n_dtheta] + large[:n_large_dz] + succ[:n_success]:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        dx, dy, dz = e_signed(r)
        rr = dict(r)
        rr["_dx"] = dx
        rr["_dy"] = dy
        rr["_dz"] = dz
        rr["_dtheta"] = float((r.get("physics_entry") or {}).get("dtheta_deg") or 0)
        out.append(rr)
    return out
