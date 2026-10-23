"""K49 action vocabulary: K48 motions 0–46 + HANDOFF 47 + secure reverse-wrist 48.

Does not modify the original K48 codebook file or checkpoints.
New motion rows are appended at ID 48 (and optionally 49 for K50).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K48_MOTION_NPY = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"

# Stable IDs
HANDOFF = 47
SECURE_REVERSE_WRIST = 48  # K49 append
# Optional K50 mid-grip reverse (only if justified)
MID_REVERSE_WRIST = 49

CAP = 0.25

# Dense table indexed by token ID; row 47 is unused (HANDOFF).
K49_TABLE_ROWS = 49  # indices 0..48
K50_TABLE_ROWS = 50  # indices 0..49


def _mirror_wy(row: np.ndarray, wy_scale: float = 1.0) -> np.ndarray:
    """Secure reverse of tok4: flip wy sign, optionally scale |wy| (grip/trans/other unchanged)."""
    out = np.asarray(row, np.float32).copy()
    out[4] = np.float32(-float(out[4]) * float(wy_scale))
    return out


# Named K49 configs: tokens 0–47 identical; only tok48 |wy| differs.
# Historical default = full-speed mirror (wy_scale=1.0).
K49_TOK48_SPEED_CONFIGS: dict[str, dict] = {
    "K49_tok48_wy1p0": {
        "wy_scale": 1.0,
        "description": "Original K49: tok48 = exact secure -wy mirror of tok4 (|wy|=|tok4|)",
        "historical_default": True,
    },
    "K49_tok48_wy0p5": {
        "wy_scale": 0.5,
        "description": "tok48 secure -wy at 50% of original |wy|; grip/trans/other unchanged",
        "historical_default": False,
    },
    "K49_tok48_wy0p25": {
        "wy_scale": 0.25,
        "description": "tok48 secure -wy at 25% of original |wy|; grip/trans/other unchanged",
        "historical_default": False,
    },
}
DEFAULT_K49_CONFIG_ID = "K49_tok48_wy1p0"


def build_k49_table(
    base: np.ndarray | None = None,
    *,
    tok48_wy_scale: float = 1.0,
    config_id: str | None = None,
) -> np.ndarray:
    """Return shape (49, 7): rows 0–46 original, 47 zeros, 48 = scaled secure -wy mirror of tok4.

    Default ``tok48_wy_scale=1.0`` preserves the historical full-speed K49 mapping.
    Pass ``config_id`` (see ``K49_TOK48_SPEED_CONFIGS``) to select a named speed variant.
    """
    if config_id is not None:
        if config_id not in K49_TOK48_SPEED_CONFIGS:
            raise ValueError(f"unknown K49 config_id {config_id!r}; known={list(K49_TOK48_SPEED_CONFIGS)}")
        tok48_wy_scale = float(K49_TOK48_SPEED_CONFIGS[config_id]["wy_scale"])
    if base is None:
        base = np.load(K48_MOTION_NPY).astype(np.float32)
    if base.shape != (47, 7):
        raise RuntimeError(f"K48 motion book must be (47,7), got {base.shape}")
    table = np.zeros((K49_TABLE_ROWS, 7), np.float32)
    table[:47] = base
    # row 47 unused (HANDOFF id)
    table[SECURE_REVERSE_WRIST] = _mirror_wy(base[4], wy_scale=tok48_wy_scale)
    return table


def describe_k49_config(config_id: str, base: np.ndarray | None = None) -> dict:
    """Metadata + concrete tok48 action for a named config."""
    if config_id not in K49_TOK48_SPEED_CONFIGS:
        raise ValueError(config_id)
    if base is None:
        base = np.load(K48_MOTION_NPY).astype(np.float32)
    table = build_k49_table(base, config_id=config_id)
    cfg = K49_TOK48_SPEED_CONFIGS[config_id]
    tok4 = np.asarray(base[4], float)
    tok48 = np.asarray(table[SECURE_REVERSE_WRIST], float)
    return {
        "config_id": config_id,
        "wy_scale": float(cfg["wy_scale"]),
        "description": cfg["description"],
        "historical_default": bool(cfg["historical_default"]),
        "handoff_id": HANDOFF,
        "token_4_action": tok4.tolist(),
        "token_48_action": tok48.tolist(),
        "token_48_wy": float(tok48[4]),
        "token_4_wy": float(tok4[4]),
        "preserved_ids_0_47": True,
        "grip_a6": float(tok48[6]),
        "translation_a012": tok48[:3].tolist(),
        "other_ang_a35": [float(tok48[3]), float(tok48[5])],
    }


def save_k49_speed_configs(out_dir: Path, base: np.ndarray | None = None) -> dict:
    """Write all named tok48-speed tables without overwriting a single default file silently."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if base is None:
        base = np.load(K48_MOTION_NPY).astype(np.float32)
    catalog = {}
    for cid in K49_TOK48_SPEED_CONFIGS:
        table = build_k49_table(base, config_id=cid)
        assert_k48_prefix_identical(table, base)
        np.save(out_dir / f"{cid}_table.npy", table)
        meta = describe_k49_config(cid, base)
        (out_dir / f"{cid}_meta.json").write_text(
            __import__("json").dumps(meta, indent=2), encoding="utf-8"
        )
        catalog[cid] = meta
    # Explicit historical alias
    hist = build_k49_table(base, config_id=DEFAULT_K49_CONFIG_ID)
    np.save(out_dir / "codebook_k49_table_historical_wy1p0.npy", hist)
    (out_dir / "CONFIG_CATALOG.json").write_text(
        __import__("json").dumps(catalog, indent=2), encoding="utf-8"
    )
    return catalog


def build_k50_table(base: np.ndarray | None = None) -> np.ndarray:
    """K49 + mid-grip reverse of tok4 (a6=-0.48, -wy). Only if experiments require it."""
    table49 = build_k49_table(base)
    table = np.zeros((K50_TABLE_ROWS, 7), np.float32)
    table[:K49_TABLE_ROWS] = table49
    mid = table49[SECURE_REVERSE_WRIST].copy()
    mid[6] = np.float32(-0.48)
    table[MID_REVERSE_WRIST] = mid
    return table


def save_vocab_artifacts(out_dir: Path, include_k50: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = np.load(K48_MOTION_NPY).astype(np.float32)
    # Historical default: full-speed tok48 (wy_scale=1.0)
    k49 = build_k49_table(base, config_id=DEFAULT_K49_CONFIG_ID)
    np.save(out_dir / "codebook_k49_table.npy", k49)
    meta = {
        "vocab": "K49",
        "config_id": DEFAULT_K49_CONFIG_ID,
        "n_action_ids": 49,
        "ids": "0..48",
        "handoff_id": HANDOFF,
        "new_motion_ids": [SECURE_REVERSE_WRIST],
        "token_48_source": "exact mirror of K48 token 4 (secure +wy → secure -wy)",
        "token_48_action": k49[SECURE_REVERSE_WRIST].tolist(),
        "token_4_action": base[4].tolist(),
        "preserved_motion_0_46": True,
        "tok48_wy_scale": 1.0,
        "k48_source": str(K48_MOTION_NPY.relative_to(ROOT)),
        "note": "Slow tok48 variants live under named config_ids; see save_k49_speed_configs()",
    }
    if include_k50:
        k50 = build_k50_table(base)
        np.save(out_dir / "codebook_k50_table.npy", k50)
        meta["vocab"] = "K50"
        meta["n_action_ids"] = 50
        meta["ids"] = "0..49"
        meta["new_motion_ids"] = [SECURE_REVERSE_WRIST, MID_REVERSE_WRIST]
        meta["token_49_action"] = k50[MID_REVERSE_WRIST].tolist()
        meta["token_49_note"] = "mid-grip reverse wy; only if secure reverse insufficient"
    return meta


def is_handoff(token: int) -> bool:
    return int(token) == HANDOFF


def decode_token(table: np.ndarray, book48: np.ndarray, tok: int, residual: np.ndarray | None = None) -> np.ndarray:
    """Frozen legal decode. Token 47 is HANDOFF (not a motion). Residuals default to 0."""
    from training.k47_codebook import decode_action as decode_k48

    tok = int(tok)
    pre = np.zeros(7, np.float32) if residual is None else np.asarray(residual, np.float32)
    if tok == HANDOFF:
        raise RuntimeError("HANDOFF token must not be executed as a motion action")
    if tok <= 46:
        _residual, action = decode_k48(book48, tok, pre)
        return np.asarray(action, np.float32)
    _residual, action = decode_action(table, tok, pre)
    return np.asarray(action, np.float32)


def decode_action(table: np.ndarray, token: int, pre) -> tuple[np.ndarray, np.ndarray]:
    """Decode a motion token from a K49/K50 ID-indexed table.

    Token 47 (HANDOFF) is rejected here — callers must treat it as terminal.
    """
    tok = int(token)
    if tok == HANDOFF:
        raise RuntimeError("token 47 is HANDOFF, not a motion code")
    if tok < 0 or tok >= len(table):
        raise RuntimeError(f"token {tok} outside vocabulary table len={len(table)}")
    if tok == HANDOFF:
        raise RuntimeError("invalid")
    # unused HANDOFF row must not be sampled as motion
    residual = np.clip(np.asarray(pre, np.float32), -CAP, CAP)
    action = np.clip(np.asarray(table[tok], np.float32) + residual, -1.0, 1.0)
    return residual, action


def motion_token_ids(vocab: str = "K49") -> list[int]:
    ids = list(range(47))  # 0..46
    ids.append(SECURE_REVERSE_WRIST)
    if vocab.upper() == "K50":
        ids.append(MID_REVERSE_WRIST)
    return ids


def logit_layout(vocab: str = "K49") -> dict:
    """Document intended future Transformer logit layout (not trained in this task)."""
    if vocab.upper() == "K48":
        return {
            "logit_dim": 48,
            "motion_ids": list(range(47)),
            "handoff_id": 47,
            "note": "existing K48: logits[0:47]=motion, logits[47]=HANDOFF",
        }
    if vocab.upper() == "K49":
        return {
            "logit_dim": 49,
            "motion_ids": list(range(47)) + [48],
            "handoff_id": 47,
            "note": "logits[0:47]=motion0-46, logits[47]=HANDOFF, logits[48]=secure reverse",
            "proto_rows": 49,
            "proto_row_47": "unused placeholder",
        }
    if vocab.upper() == "K50":
        return {
            "logit_dim": 50,
            "motion_ids": list(range(47)) + [48, 49],
            "handoff_id": 47,
            "note": "K49 + logits[49]=mid reverse",
            "proto_rows": 50,
        }
    raise ValueError(vocab)


def assert_k48_prefix_identical(table: np.ndarray, base: np.ndarray | None = None) -> None:
    if base is None:
        base = np.load(K48_MOTION_NPY).astype(np.float32)
    if not np.array_equal(table[:47], base):
        raise RuntimeError("K49/K50 table corrupted original tokens 0-46")
