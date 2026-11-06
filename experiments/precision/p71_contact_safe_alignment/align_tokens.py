"""Select alignment-capable M111 tokens (no new actions)."""

from __future__ import annotations

import numpy as np

from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p71_contact_safe_alignment.paths import ALIGN_FALLBACK, TOKEN_56


def alignment_token_pool(book: np.ndarray, *, exclude_56: bool = True, k: int = 8) -> list[int]:
    """Rank APPROACH/ORIENT tokens by linear+angular magnitude; optionally drop token 56."""
    scored = []
    for i in range(len(book)):
        if exclude_56 and int(i) == int(TOKEN_56):
            continue
        fam = family_of_token(book, i)
        if fam not in ("APPROACH", "ORIENT"):
            continue
        a = np.asarray(book[i], float).reshape(7)
        # Prefer alignment motion; penalize strong closing (grip>0 large)
        score = float(np.linalg.norm(a[:3]) + 0.5 * np.linalg.norm(a[3:6]) - 0.35 * max(float(a[6]), 0.0))
        scored.append((score, int(i)))
    scored.sort(key=lambda x: -x[0])
    toks = [t for _, t in scored[:k]]
    return toks if toks else list(ALIGN_FALLBACK)


def decode_token(book: np.ndarray, tok: int) -> dict:
    a = np.asarray(book[int(tok)], float).reshape(7)
    return {
        "token": int(tok),
        "family": family_of_token(book, tok),
        "lin": a[:3].tolist(),
        "ang": a[3:6].tolist(),
        "grip": float(a[6]),
        "lin_norm": float(np.linalg.norm(a[:3])),
        "ang_norm": float(np.linalg.norm(a[3:6])),
    }
