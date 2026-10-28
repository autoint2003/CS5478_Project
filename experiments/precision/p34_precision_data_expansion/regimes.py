"""Stratified Precision IC regimes for P34 (construction specs only).

Exploration quota prevents collapse onto DZ≈12 / COMBINED(dx4,dz12).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegimeSpec:
    regime_id: str
    family: str  # DX_ONLY | DZ_ONLY | COMBINED | DTHETA | DX_DTHETA | VEL | MASS_FRIC
    target_dx_mm: float = 0.0
    target_dy_mm: float = 0.0
    target_dz_mm: float = 0.0
    dtheta_deg: float = 0.0
    v_lin_hand_mms: tuple[float, float, float] = (0.0, 0.0, 0.0)
    w_hand_dps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    bucket: str = "explore"  # productive | explore | underrepresented

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["v_lin_hand_mms"] = list(self.v_lin_hand_mms)
        d["w_hand_dps"] = list(self.w_hand_dps)
        return d


# Known easy / overrepresented cells from P33 — still allowed but capped by quota.
OVERREPRESENTED = {
    ("DZ_ONLY", 0.0, 12.0),
    ("COMBINED", 4.0, 12.0),
}


def build_regime_catalog() -> list[RegimeSpec]:
    specs: list[RegimeSpec] = []

    def add(family, dx, dz, *, dy=0.0, th=0.0, bucket="explore", **kw):
        rid = f"{family}_dx{dx:g}_dy{dy:g}_dz{dz:g}_th{th:g}"
        if kw.get("mass_scale", 1.0) != 1.0 or kw.get("friction_scale", 1.0) != 1.0:
            rid += f"_m{kw.get('mass_scale', 1.0):g}_f{kw.get('friction_scale', 1.0):g}"
        vlin = kw.get("v_lin_hand_mms") or (0.0, 0.0, 0.0)
        wdeg = kw.get("w_hand_dps") or (0.0, 0.0, 0.0)
        if any(abs(float(x)) > 1e-12 for x in vlin) or any(abs(float(x)) > 1e-12 for x in wdeg):
            rid += (
                f"_v{float(vlin[0]):g}_{float(vlin[1]):g}_{float(vlin[2]):g}"
                f"_w{float(wdeg[0]):g}_{float(wdeg[1]):g}_{float(wdeg[2]):g}"
            )
        if (
            (family, float(dx), float(dz)) in OVERREPRESENTED
            and abs(th) < 1e-9
            and abs(dy) < 1e-9
            and abs(kw.get("mass_scale", 1.0) - 1.0) < 1e-12
            and abs(kw.get("friction_scale", 1.0) - 1.0) < 1e-12
            and not any(abs(float(x)) > 1e-12 for x in vlin)
            and not any(abs(float(x)) > 1e-12 for x in wdeg)
        ):
            bucket = "productive"
        specs.append(
            RegimeSpec(
                regime_id=rid,
                family=family,
                target_dx_mm=float(dx),
                target_dy_mm=float(dy),
                target_dz_mm=float(dz),
                dtheta_deg=float(th),
                bucket=bucket,
                **kw,
            )
        )

    # --- DX / DZ / COMBINED position residuals (P14-scale eligibility band) ---
    for v in (8.0, 10.0, 12.0, 14.0, 16.0, -8.0, -10.0, -12.0, -14.0, -16.0):
        add("DX_ONLY", v, 0.0, bucket="underrepresented")
    for v in (8.0, 10.0, 12.0, 14.0, 16.0, -8.0, -10.0, -12.0, -14.0, -16.0):
        b = "productive" if abs(v) == 12.0 else "underrepresented"
        add("DZ_ONLY", 0.0, v, bucket=b)
    for dx, dz in (
        (4.0, 12.0),
        (4.0, 16.0),
        (8.0, 4.0),
        (8.0, 8.0),
        (8.0, -8.0),
        (10.0, 8.0),
        (10.0, -8.0),
        (12.0, -4.0),
        (-4.0, -12.0),
        (-8.0, -4.0),
        (-8.0, 8.0),
        (-10.0, -8.0),
        (-12.0, 4.0),
        (12.0, 8.0),
        (-12.0, -8.0),
        (6.0, 14.0),
        (-6.0, 14.0),
        (14.0, 6.0),
        (-14.0, -6.0),
        (9.0, -12.0),
        (-9.0, 12.0),
    ):
        b = "productive" if (dx, dz) == (4.0, 12.0) else "underrepresented"
        add("COMBINED", dx, dz, bucket=b)

    # Mild lateral (Y) with DZ — contact-margin diversity
    for dy, dz in ((3.0, 12.0), (-3.0, 12.0), (4.0, -12.0), (-4.0, 10.0)):
        add("COMBINED", 0.0, dz, dy=dy, bucket="explore")

    # Orientation residuals
    for th in (8.0, 12.0, 16.0, -8.0, -12.0, -16.0):
        add("DTHETA", 0.0, 0.0, th=th, bucket="underrepresented")
    for th, dx in ((10.0, 8.0), (-10.0, -8.0), (12.0, 4.0), (-12.0, 6.0)):
        add("DX_DTHETA", dx, 0.0, th=th, bucket="underrepresented")
    for th, dz in ((8.0, 12.0), (-8.0, 12.0), (10.0, -12.0), (12.0, 10.0)):
        add("COMBINED", 2.0, dz, th=th, bucket="explore")

    # Mild relative velocity (hand-frame linear mm/s → set after settle, before freeze)
    for v in ((20.0, 0.0, 0.0), (-20.0, 0.0, 0.0), (0.0, 0.0, 25.0), (0.0, 0.0, -25.0), (15.0, 0.0, 15.0)):
        add(
            "VEL",
            4.0,
            12.0,
            v_lin_hand_mms=v,
            bucket="explore",
        )
    for w in ((30.0, 0.0, 0.0), (0.0, 25.0, 0.0), (-25.0, 0.0, 0.0)):
        add("VEL", 0.0, 12.0, w_hand_dps=w, bucket="explore")

    # Mass / friction scales applied pre-freeze; stored as dynamics_override for restore
    for ms, fs in ((0.85, 1.0), (1.15, 1.0), (1.0, 0.85), (1.0, 1.15), (0.9, 1.1), (1.1, 0.9)):
        add(
            "MASS_FRIC",
            4.0,
            12.0,
            mass_scale=ms,
            friction_scale=fs,
            bucket="explore",
        )
        add(
            "MASS_FRIC",
            0.0,
            12.0,
            mass_scale=ms,
            friction_scale=fs,
            bucket="explore",
        )

    # Dedup by regime_id
    seen: set[str] = set()
    out: list[RegimeSpec] = []
    for s in specs:
        if s.regime_id in seen:
            continue
        seen.add(s.regime_id)
        out.append(s)
    return out


def adaptive_schedule(
    catalog: list[RegimeSpec],
    *,
    regime_p1_counts: dict[str, int],
    rng: np.random.Generator,
    exploration_frac: float = 0.30,
    productive_frac: float = 0.40,
) -> list[RegimeSpec]:
    """Mix productive / underrepresented / explore with fixed quotas.

    Productive cells (historically yielding P1) keep a floor quota so the
    sampler does not starve DZ≈12 while hunting diversity. Exploration and
    underrepresented cells retain dedicated share so we do not collapse onto
    only the easiest family.
    """
    by_bucket: dict[str, list[RegimeSpec]] = {"productive": [], "underrepresented": [], "explore": []}
    for s in catalog:
        by_bucket.setdefault(s.bucket, []).append(s)

    def score(s: RegimeSpec) -> float:
        n = float(regime_p1_counts.get(s.regime_id, 0))
        base = 1.0 / (1.0 + n)
        if s.bucket == "productive":
            # Soft cap: still sample, but less once a cell is saturated
            if n >= 8:
                return 0.15 * base
            return 3.0 * base
        if s.bucket == "underrepresented":
            return 2.0 * base
        return base

    prod = sorted(by_bucket.get("productive", []), key=score, reverse=True)
    under = sorted(by_bucket.get("underrepresented", []), key=score, reverse=True)
    explore = list(by_bucket.get("explore", []))
    rng.shuffle(explore)

    n = len(catalog)
    n_prod = max(len(prod), int(round(n * productive_frac)))
    n_expl = max(1, int(round(n * exploration_frac)))
    # Build round-robin mix: P, U, P, E, U, ...
    prod_cycle = (prod * ((n_prod // max(1, len(prod))) + 2))[: max(n_prod, len(prod))]
    under_cycle = under
    expl_cycle = explore[:n_expl]

    out: list[RegimeSpec] = []
    pi = ui = ei = 0
    # Target mix ratios in the emitted order
    pattern = ["P", "U", "P", "E", "U", "P", "U", "E"]
    guard = 0
    while len(out) < n and guard < n * 4:
        guard += 1
        kind = pattern[guard % len(pattern)]
        if kind == "P" and pi < len(prod_cycle):
            out.append(prod_cycle[pi])
            pi += 1
        elif kind == "U" and ui < len(under_cycle):
            out.append(under_cycle[ui])
            ui += 1
        elif kind == "E" and ei < len(expl_cycle):
            out.append(expl_cycle[ei])
            ei += 1
        elif ui < len(under_cycle):
            out.append(under_cycle[ui])
            ui += 1
        elif pi < len(prod_cycle):
            out.append(prod_cycle[pi])
            pi += 1
        elif ei < len(expl_cycle):
            out.append(expl_cycle[ei])
            ei += 1
        else:
            break
    # Dedup preserving order
    seen: set[str] = set()
    dedup: list[RegimeSpec] = []
    for s in out:
        if s.regime_id in seen:
            continue
        seen.add(s.regime_id)
        dedup.append(s)
    for s in catalog:
        if s.regime_id not in seen:
            dedup.append(s)
    return dedup
