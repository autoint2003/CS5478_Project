"""Skill-aware hybrid action vocabulary (motion prototypes + bounded residual + HANDOFF).

Does not overwrite K49 artifacts. Hybrid decoder matches the deployed policy form:

    a = clip(c_k + clip(r, -rho, rho), -1, 1)

where rho may be a scalar or a length-7 vector. HANDOFF is outside the motion book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Dimension names / default assignment weights (emphasize vz, wx; de-emphasize grip).
ACT_DIMS = ("vx", "vy", "vz", "wx", "wy", "wz", "grip")
DEFAULT_WEIGHTS = np.asarray([1.0, 1.0, 1.5, 2.0, 1.0, 1.0, 0.5], dtype=np.float64)
M_MAX = 128
DT = 0.02
EARLY_CATCH_S = 0.20
EARLY_CATCH_N = int(round(EARLY_CATCH_S / DT))  # 10
SETTLE_N = EARLY_CATCH_N

SKILLS = ("A", "B", "C", "STABLE_PRECISION")


@dataclass
class ResidualBudget:
    """Per-dimension residual caps in normalized action units."""

    rho: np.ndarray  # shape (7,)

    @classmethod
    def scalar(cls, v: float) -> "ResidualBudget":
        return cls(rho=np.full(7, float(v), dtype=np.float64))

    @classmethod
    def from_spec(cls, spec: Any) -> "ResidualBudget":
        if isinstance(spec, ResidualBudget):
            return spec
        if isinstance(spec, (int, float)):
            return cls.scalar(float(spec))
        arr = np.asarray(spec, dtype=np.float64).reshape(7)
        return cls(rho=arr.copy())

    def as_array(self) -> np.ndarray:
        return np.asarray(self.rho, dtype=np.float64).reshape(7)

    def label(self) -> str:
        r = self.as_array()
        if float(np.max(r) - np.min(r)) < 1e-12:
            return f"rho={float(r[0]):.2f}"
        parts = [f"{d}={float(r[i]):.2f}" for i, d in enumerate(ACT_DIMS)]
        return "rho_dim[" + ",".join(parts) + "]"

    def to_list(self) -> list[float]:
        return [float(x) for x in self.as_array()]


@dataclass
class MotionCodebook:
    """Dense motion prototype table. HANDOFF is not a row."""

    prototypes: np.ndarray  # (M, 7)
    residual: ResidualBudget
    weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    meta: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def handoff_id(self) -> int:
        """HANDOFF token id = M (first id after motion rows)."""
        return self.M

    def save(self, out_dir: Path, stem: str = "skill_aware_hybrid") -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npy = out_dir / f"{stem}_prototypes.npy"
        np.save(npy, self.prototypes.astype(np.float32))
        meta = {
            "vocab": "skill_aware_hybrid",
            "M": self.M,
            "handoff_id": self.handoff_id,
            "residual_rho": self.residual.to_list(),
            "residual_label": self.residual.label(),
            "weights": [float(x) for x in np.asarray(self.weights).reshape(7)],
            "act_dims": list(ACT_DIMS),
            "prototypes_path": str(npy.relative_to(ROOT)).replace("\\", "/"),
            **(self.meta or {}),
        }
        (out_dir / f"{stem}_meta.json").write_text(
            __import__("json").dumps(meta, indent=2), encoding="utf-8"
        )
        return meta


def weighted_linf(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    w = DEFAULT_WEIGHTS if w is None else np.asarray(w, dtype=np.float64).reshape(7)
    d = np.abs(np.asarray(a, dtype=np.float64).reshape(7) - np.asarray(b, dtype=np.float64).reshape(7))
    return float(np.max(w * d))


def nearest_proto(
    book: np.ndarray,
    a: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[int, np.ndarray, float]:
    """Nearest motion prototype under weighted L∞. book shape (M,7), no HANDOFF row."""
    book = np.asarray(book, dtype=np.float64).reshape(-1, 7)
    a = np.asarray(a, dtype=np.float64).reshape(7)
    w = DEFAULT_WEIGHTS if w is None else np.asarray(w, dtype=np.float64).reshape(7)
    if len(book) == 0:
        raise RuntimeError("empty motion book")
    d = np.max(w * np.abs(book - a), axis=1)
    best_i = int(np.argmin(d))
    return best_i, book[best_i].astype(np.float32).copy(), float(d[best_i])


def batch_nearest_wlinf(
    book: np.ndarray,
    acts: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (token_ids, wlinf_dists) for a batch of actions."""
    book = np.asarray(book, dtype=np.float64).reshape(-1, 7)
    acts = np.asarray(acts, dtype=np.float64).reshape(-1, 7)
    w = DEFAULT_WEIGHTS if w is None else np.asarray(w, dtype=np.float64).reshape(7)
    # (N,M,7) — OK for N~1e3, M~128
    d = np.max(w * np.abs(acts[:, None, :] - book[None, :, :]), axis=2)
    tok = np.argmin(d, axis=1).astype(np.int64)
    return tok, d[np.arange(len(acts)), tok]


def clip_residual(raw: np.ndarray, rho: ResidualBudget | np.ndarray | float) -> np.ndarray:
    r = ResidualBudget.from_spec(rho).as_array()
    return np.clip(np.asarray(raw, dtype=np.float32).reshape(7), -r, r).astype(np.float32)


def hybrid_decode(
    proto: np.ndarray,
    residual: np.ndarray,
    rho: ResidualBudget | np.ndarray | float,
) -> np.ndarray:
    """Deployed hybrid decoder: clip(proto + clip(r, ±rho), ±1)."""
    res = clip_residual(residual, rho)
    return np.clip(np.asarray(proto, dtype=np.float32).reshape(7) + res, -1.0, 1.0).astype(np.float32)


def oracle_reconstruct(
    book: np.ndarray,
    a: np.ndarray,
    rho: ResidualBudget | np.ndarray | float,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Oracle token+residual from expert action (diagnostic, not a learned policy)."""
    ti, proto, dist = nearest_proto(book, a, w=w)
    raw = np.asarray(a, dtype=np.float32).reshape(7) - proto
    res = clip_residual(raw, rho)
    recon = hybrid_decode(proto, res, rho)
    err = recon - np.asarray(a, dtype=np.float32).reshape(7)
    rbudget = ResidualBudget.from_spec(rho).as_array()
    beyond = np.maximum(np.abs(raw) - rbudget, 0.0)
    return recon, {
        "token": ti,
        "proto": proto,
        "residual": res,
        "raw_residual": raw.astype(np.float32),
        "proto_wlinf": dist,
        "recon_l2": float(np.linalg.norm(err)),
        "recon_linf": float(np.max(np.abs(err))),
        "recon_wlinf": weighted_linf(recon, a, w),
        "clipped": bool(np.any(beyond > 1e-8)),
        "beyond": beyond.astype(np.float32),
        "saturated_dims": [ACT_DIMS[i] for i in range(7) if beyond[i] > 1e-8],
        "err": err.astype(np.float32),
    }


def quantize_trajectory(
    book: np.ndarray,
    actions: np.ndarray,
    rho: ResidualBudget | np.ndarray | float,
    w: np.ndarray | None = None,
) -> dict:
    acts = np.asarray(actions, dtype=np.float32)
    if acts.size == 0:
        return {
            "tokens": np.zeros((0,), dtype=np.int64),
            "residuals": np.zeros((0, 7), dtype=np.float32),
            "recon_actions": np.zeros((0, 7), dtype=np.float32),
            "clipped_frac": 0.0,
            "mean_recon_wlinf": 0.0,
            "max_recon_wlinf": 0.0,
            "mean_abs_residual": np.zeros(7, dtype=np.float32),
            "sat_frac_per_dim": np.zeros(7, dtype=np.float32),
            "infos": [],
        }
    book = np.asarray(book, dtype=np.float32).reshape(-1, 7)
    w_arr = DEFAULT_WEIGHTS if w is None else np.asarray(w, dtype=np.float64).reshape(7)
    rho_b = ResidualBudget.from_spec(rho)
    rbudget = rho_b.as_array()
    tokens, _ = batch_nearest_wlinf(book, acts, w=w_arr)
    proto = book[tokens]
    raw = acts - proto
    residuals = np.clip(raw, -rbudget, rbudget).astype(np.float32)
    recons = np.clip(proto + residuals, -1.0, 1.0).astype(np.float32)
    err = recons - acts
    beyond = np.maximum(np.abs(raw) - rbudget, 0.0)
    wl = np.max(w_arr * np.abs(err), axis=1)
    clipped_mask = np.any(beyond > 1e-8, axis=1)
    n = len(acts)
    infos = [
        {
            "token": int(tokens[i]),
            "proto": proto[i],
            "residual": residuals[i],
            "raw_residual": raw[i].astype(np.float32),
            "recon_wlinf": float(wl[i]),
            "recon_l2": float(np.linalg.norm(err[i])),
            "recon_linf": float(np.max(np.abs(err[i]))),
            "clipped": bool(clipped_mask[i]),
            "beyond": beyond[i].astype(np.float32),
            "err": err[i].astype(np.float32),
        }
        for i in range(n)
    ]
    return {
        "tokens": tokens.astype(np.int64),
        "residuals": residuals,
        "recon_actions": recons,
        "clipped_frac": float(np.mean(clipped_mask)),
        "mean_recon_wlinf": float(np.mean(wl)),
        "max_recon_wlinf": float(np.max(wl)),
        "mean_abs_residual": np.mean(np.abs(residuals), axis=0).astype(np.float32),
        "sat_frac_per_dim": np.mean(beyond > 1e-8, axis=0).astype(np.float32),
        "infos": infos,
    }


def decode_tokens(
    book: np.ndarray,
    tokens: np.ndarray,
    residuals: np.ndarray,
    rho: ResidualBudget | np.ndarray | float,
    handoff_id: int | None = None,
) -> np.ndarray:
    """Decode motion tokens with bounded residual. Rejects HANDOFF ids."""
    hid = int(len(book) if handoff_id is None else handoff_id)
    out = []
    res = np.asarray(residuals, dtype=np.float32)
    for i, t in enumerate(tokens):
        ti = int(t)
        if ti == hid:
            raise RuntimeError("HANDOFF is a control switch, not a motion decode")
        if ti < 0 or ti >= len(book):
            raise RuntimeError(f"token {ti} outside motion book M={len(book)}")
        r = res[i] if i < len(res) else np.zeros(7, dtype=np.float32)
        out.append(hybrid_decode(book[ti], r, rho))
    return np.asarray(out, dtype=np.float32)


def phase_labels(
    n: int,
    stage1_end: int,
    skill: str,
    *,
    n_precision: int | None = None,
) -> list[str]:
    """Per-step phase tags for stratified coverage."""
    skill = str(skill)
    s1 = int(stage1_end)
    labels = []
    for i in range(n):
        if skill == "C":
            if i < s1:
                labels.append("P_catch" if i < EARLY_CATCH_N else "P_lift")
            else:
                labels.append("P_precision" if n_precision and i < s1 + int(n_precision) else "P_settle")
            continue
        if skill in ("A", "B"):
            if i < s1:
                labels.append("P_s1_short")
            elif n_precision and i < s1 + int(n_precision):
                labels.append("P_precision")
            else:
                # settle: last SETTLE_N of traj, else post
                labels.append("P_settle" if i >= max(0, n - SETTLE_N) else "P_post")
            continue
        if skill == "STABLE_PRECISION":
            if n_precision and i < int(n_precision):
                labels.append("P_precision")
            else:
                labels.append("P_settle" if i >= max(0, n - SETTLE_N) else "P_post")
            continue
        labels.append("P_other")
    # mark settle for C/A if near end
    for i in range(max(0, n - SETTLE_N), n):
        if labels[i] in ("P_lift", "P_post", "P_precision") and skill != "STABLE_PRECISION":
            # keep precision label if still in precision body; only rewrite lift/post
            if labels[i] != "P_precision":
                labels[i] = "P_settle"
    return labels


def action_key(a: np.ndarray, nd: int = 3) -> str:
    import hashlib

    r = np.round(np.asarray(a, dtype=np.float64), nd)
    return hashlib.sha1(r.tobytes()).hexdigest()[:12]


def dedupe_prototypes(protos: list[np.ndarray], min_dist: float = 0.08, w=None) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for p in protos:
        p = np.asarray(p, dtype=np.float32).reshape(7)
        if any(weighted_linf(p, q, w) < min_dist for q in kept):
            continue
        kept.append(p.copy())
    return kept


def load_motion_codebook(meta_path: Path) -> MotionCodebook:
    """Load a frozen skill-aware codebook from its meta JSON."""
    meta_path = Path(meta_path)
    meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
    proto_path = ROOT / meta["prototypes_path"]
    if not proto_path.exists():
        proto_path = meta_path.parent / Path(meta["prototypes_path"]).name
    protos = np.load(proto_path).astype(np.float32)
    return MotionCodebook(
        prototypes=protos,
        residual=ResidualBudget.from_spec(meta["residual_rho"]),
        weights=np.asarray(meta.get("weights", DEFAULT_WEIGHTS), dtype=np.float64),
        meta=meta,
    )
