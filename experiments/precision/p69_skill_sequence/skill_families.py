"""Heuristic M111 skill-family labels (no official token→skill map exists).

Families are derived from codebook prototype magnitudes only — deployable
without privileged physics. Not Expert skill labels.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Family = Literal["APPROACH", "ORIENT", "GRIP", "MIXED"]


def family_of_token(book: np.ndarray, tok: int) -> Family:
    a = np.asarray(book[int(tok)], float).reshape(7)
    lin = float(np.linalg.norm(a[:3]))
    ang = float(np.linalg.norm(a[3:6]))
    grip = abs(float(a[6]))
    m = max(lin, ang, grip)
    if m < 1e-6:
        return "MIXED"
    if lin >= ang and lin >= grip:
        return "APPROACH"
    if ang >= lin and ang >= grip:
        return "ORIENT"
    if grip >= lin and grip >= ang:
        return "GRIP"
    return "MIXED"


def family_sequence(book: np.ndarray, tokens: list[int]) -> list[str]:
    return [family_of_token(book, t) for t in tokens]


def n_family_switches(fams: list[str]) -> int:
    if len(fams) < 2:
        return 0
    return sum(1 for a, b in zip(fams, fams[1:]) if a != b)
