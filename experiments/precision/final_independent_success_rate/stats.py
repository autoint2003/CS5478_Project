"""Wilson CI + stratified bootstrap for equal-weight overall SR."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> dict:
    if n <= 0:
        return {"low": None, "high": None, "center": None, "n": 0, "k": int(k)}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return {
        "low": float(max(0.0, center - half)),
        "high": float(min(1.0, center + half)),
        "center": float(center),
        "n": int(n),
        "k": int(k),
        "method": "wilson",
        "z": z,
    }


def class_sr(successes: Iterable[int]) -> dict:
    arr = [int(x) for x in successes]
    n = len(arr)
    k = int(sum(arr))
    rate = (k / n) if n else None
    return {
        "n_independent_episodes": n,
        "n_success": k,
        "n_fail": n - k,
        "success_rate": rate,
        "success_rate_pct": (100.0 * rate) if rate is not None else None,
        "ci95_wilson": wilson_interval(k, n),
    }


def equal_weight_overall(by_class: dict[str, list[int]], *, n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean of per-class SRs; stratified bootstrap CI resampling ICs within class."""
    skills = list(by_class.keys())
    rates = []
    total_n = total_k = 0
    for sk in skills:
        s = class_sr(by_class[sk])
        rates.append(s["success_rate"] if s["success_rate"] is not None else 0.0)
        total_n += s["n_independent_episodes"]
        total_k += s["n_success"]
    overall = float(np.mean(rates)) if rates else None
    # Stratified bootstrap
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        br = []
        for sk in skills:
            arr = np.asarray(by_class[sk], dtype=float)
            if len(arr) == 0:
                br.append(0.0)
                continue
            samp = rng.choice(arr, size=len(arr), replace=True)
            br.append(float(samp.mean()))
        boots.append(float(np.mean(br)))
    ci = None
    if boots:
        ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return {
        "n_independent_episodes_total": total_n,
        "n_success_total": total_k,
        "equal_weight_success_rate": overall,
        "equal_weight_success_rate_pct": (100.0 * overall) if overall is not None else None,
        "pooled_success_rate": (total_k / total_n) if total_n else None,
        "ci95_stratified_bootstrap": ci,
        "n_boot": n_boot,
        "note": (
            "Primary overall = mean of class SRs (equal-weight). "
            "When class n equal, equal-weight == pooled. With unequal n, report both."
        ),
    }
