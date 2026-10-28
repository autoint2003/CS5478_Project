"""M111 policy build / load. 111 motion tokens + HANDOFF=111 (112 logits).

Refuses K48 / K47 checkpoints. Does not remap legacy action tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.k48_policy import build_policy
from training.skill_aware_action_vocab import ResidualBudget, load_motion_codebook

MOTION_K = 111
HANDOFF = 111
LOGIT_K = 112
# Frozen residual budget from SELECTED M111 meta (ρ=0.35).
DEFAULT_RHO = 0.35
M111_META = Path(
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
)


def load_m111_book(meta_path: Path | None = None):
    """Load frozen M111 prototypes + residual budget. Never regenerates the table."""
    path = Path(meta_path) if meta_path is not None else M111_META
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        path = root / path
    cb = load_motion_codebook(path)
    book = np.asarray(cb.prototypes, np.float32)
    if book.shape != (MOTION_K, 7):
        raise RuntimeError(f"M111 book shape {book.shape}, expected {(MOTION_K, 7)}")
    if int(cb.handoff_id) != HANDOFF:
        raise RuntimeError(f"HANDOFF id {cb.handoff_id}, expected {HANDOFF}")
    rho = ResidualBudget.from_spec(cb.residual).as_array()
    return book, rho, cb


def build_m111(book: np.ndarray, device, *, rho: float | np.ndarray = DEFAULT_RHO):
    book = np.asarray(book, np.float32)
    if book.shape != (MOTION_K, 7):
        raise RuntimeError(f"build_m111 expects {(MOTION_K, 7)}, got {book.shape}")
    # Policy residual head uses scalar res_cap * tanh; match max ρ dim.
    r = ResidualBudget.from_spec(rho).as_array()
    res_cap = float(np.max(r))
    model = build_policy(book, device, res_cap, handoff=True)
    if model.logits.out_features != LOGIT_K or model.handoff_index != HANDOFF:
        raise RuntimeError(
            f"M111 head mismatch: logits={model.logits.out_features} "
            f"handoff_index={model.handoff_index}"
        )
    if int(model.n_motion) != MOTION_K:
        raise RuntimeError(f"n_motion={model.n_motion}, expected {MOTION_K}")
    return model


def assert_m111_checkpoint_payload(saved: dict[str, Any], path: Path | str = "") -> np.ndarray:
    """Reject incompatible K48/K47 or wrong-width checkpoints."""
    label = str(path) or "checkpoint"
    if "vocab" in saved and str(saved["vocab"]).upper() not in ("M111", "SKILL_AWARE_HYBRID"):
        raise RuntimeError(f"{label}: vocab={saved['vocab']} is not M111")
    book = np.asarray(saved.get("proto"), np.float32)
    if book.shape != (MOTION_K, 7):
        raise RuntimeError(
            f"{label}: proto shape {book.shape} is incompatible with M111 "
            f"(refusing K48 remapping; expected {(MOTION_K, 7)})"
        )
    state = saved.get("model")
    if not isinstance(state, dict) or "logits.weight" not in state:
        raise RuntimeError(f"{label}: missing model.logits.weight")
    logits = state["logits.weight"]
    if tuple(logits.shape)[:1] != (LOGIT_K,):
        raise RuntimeError(
            f"{label}: logits rows {tuple(logits.shape)[:1]} != ({LOGIT_K},); "
            "refusing to load K48 (48) or other heads into M111"
        )
    handoff = saved.get("handoff")
    if handoff is not None and int(handoff) != HANDOFF:
        raise RuntimeError(f"{label}: handoff={handoff}, expected {HANDOFF}")
    return book


def save_m111(
    path: Path,
    model: torch.nn.Module,
    book: np.ndarray,
    *,
    epoch: int,
    rho: float | list | np.ndarray,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab": "M111",
        "proto": np.asarray(book, np.float32),
        "res_cap": float(np.max(ResidualBudget.from_spec(rho).as_array())),
        "residual_rho": ResidualBudget.from_spec(rho).to_list(),
        "handoff": HANDOFF,
        "motion_k": MOTION_K,
        "logit_k": LOGIT_K,
        "obs_dim": 39,
        "epoch": int(epoch),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_m111(path: Path | str, device):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    book = assert_m111_checkpoint_payload(saved, path)
    rho = saved.get("residual_rho", saved.get("res_cap", DEFAULT_RHO))
    model = build_m111(book, device, rho=rho)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, book, saved


def decode_m111_action(
    book: np.ndarray,
    token: int,
    residual: np.ndarray,
    rho: float | np.ndarray = DEFAULT_RHO,
) -> np.ndarray:
    """Physical decode for motion tokens. HANDOFF must not be decoded."""
    tok = int(token)
    if tok == HANDOFF:
        raise RuntimeError("HANDOFF is terminal; do not decode as a control vector")
    if tok < 0 or tok >= MOTION_K:
        raise RuntimeError(f"token {tok} outside M111 motion range [0, {MOTION_K})")
    rbudget = ResidualBudget.from_spec(rho).as_array().astype(np.float32)
    res = np.clip(np.asarray(residual, np.float32).reshape(7), -rbudget, rbudget)
    return np.clip(np.asarray(book[tok], np.float32) + res, -1.0, 1.0).astype(np.float32)
