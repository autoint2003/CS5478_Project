"""P19 M111 supervised dataset: NPZ episodes → padded batches.

Policy inputs: observations (39), prev_actions (7).
Targets: tokens, residuals. Targets are never concatenated into inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.m111_checkpoint import HANDOFF, MOTION_K
from envs.observations.airborne_obs import OBS_DIM_AIR

ROOT = Path(__file__).resolve().parents[1]
P19_DEFAULT = ROOT / "results/diagnostics/p19_sft_packaging"
P21_DEFAULT = ROOT / "results/diagnostics/p21_full_stage1_experts"
WINDOW = 160


def load_p19_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = Path(manifest_path) if manifest_path else P19_DEFAULT / "SFT_DATASET_MANIFEST.json"
    if not path.is_file():
        raise RuntimeError(f"P19 manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_p21_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Full-horizon recovery SFT manifest (P21)."""
    path = Path(manifest_path) if manifest_path else P21_DEFAULT / "SFT_FULL_HORIZON_MANIFEST.json"
    if not path.is_file():
        raise RuntimeError(f"P21 manifest missing: {path}")
    man = json.loads(path.read_text(encoding="utf-8"))
    if str(man.get("start_distribution") or "") != "recovery_ic_pre_stage1":
        raise RuntimeError(
            f"{path}: start_distribution={man.get('start_distribution')!r}; "
            "expected recovery_ic_pre_stage1 (refusing continuation-only corpus as primary)"
        )
    return man


def load_episode_npz(root: Path, relpath: str) -> dict[str, np.ndarray]:
    path = root / relpath
    blob = np.load(path)
    obs = np.asarray(blob["observations"], np.float32)
    prev = np.asarray(blob["prev_actions"], np.float32)
    tokens = np.asarray(blob["tokens"], np.int64)
    residuals = np.asarray(blob["residuals"], np.float32)
    mask = np.asarray(blob["loss_mask"], np.bool_)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM_AIR:
        raise RuntimeError(f"{path}: obs shape {obs.shape}, expected (*, {OBS_DIM_AIR})")
    n = obs.shape[0]
    if prev.shape != (n, 7) or tokens.shape != (n,) or residuals.shape != (n, 7):
        raise RuntimeError(f"{path}: tensor length mismatch")
    if mask.shape != (n,):
        raise RuntimeError(f"{path}: loss_mask shape {mask.shape}")
    if int(tokens.min()) < 0 or int(tokens.max()) > HANDOFF:
        raise RuntimeError(f"{path}: token out of [0, {HANDOFF}]")
    if np.any((tokens == HANDOFF)[:-1]):  # HANDOFF only allowed at end for P19 packs
        # Allow only terminal HANDOFF (P19 guarantee); still valid if all HANDOFF for T=1
        if not (np.all(tokens == HANDOFF) and n == 1) and np.any(tokens[:-1] == HANDOFF):
            raise RuntimeError(f"{path}: non-terminal HANDOFF in targets")
    if int(tokens[-1]) != HANDOFF:
        raise RuntimeError(f"{path}: last token must be HANDOFF")
    # Target leakage guard: prev must not equal "current" expert continuous target encoding
    # We only check structural: prev[0]==0 and prev[t] is prior executed action.
    if float(np.max(np.abs(prev[0]))) > 1e-6:
        raise RuntimeError(f"{path}: prev[0] must be zero")
    return {
        "observations": obs,
        "prev_actions": prev,
        "tokens": tokens,
        "residuals": residuals,
        "loss_mask": mask,
    }


def chunk_episode(
    ep: dict[str, np.ndarray],
    *,
    window: int = WINDOW,
) -> list[dict[str, np.ndarray]]:
    """Split long episodes into ≤window supervised segments.

    Non-overlapping chunks cover ``[0, n)`` so Stage-1 steps are never dropped.
    The final chunk always contains the terminal HANDOFF (last index).
    If the remainder is shorter than ``window``, optionally prepend context so
    the HANDOFF chunk has up to ``window`` steps of history (overlap only on
    the last chunk; earlier chunks stay disjoint and cover their ranges).
    """
    n = int(ep["observations"].shape[0])
    if n <= window:
        return [ep]
    chunks: list[dict[str, np.ndarray]] = []
    covered = np.zeros(n, dtype=np.bool_)
    start = 0
    while start < n:
        end = min(start + window, n)
        # Last chunk: give HANDOFF up to `window` of context if remainder is short.
        if end == n and (end - start) < window and start > 0:
            hist_start = max(0, n - window)
            chunk = {k: v[hist_start:n] for k, v in ep.items()}
            chunks.append(chunk)
            covered[start:n] = True  # only mark yet-uncovered tail as required
            # Overlap region hist_start:start was already covered by prior chunks.
            break
        chunk = {k: v[start:end] for k, v in ep.items()}
        chunks.append(chunk)
        covered[start:end] = True
        if end >= n:
            break
        start = end
    if not bool(covered.all()):
        missing = int(np.sum(~covered))
        raise RuntimeError(f"chunk_episode dropped {missing}/{n} timesteps")
    return chunks


def episode_weight(row: dict, scheme: str) -> float:
    scheme = str(scheme).lower()
    if scheme in ("natural", "a", "baseline_a"):
        return 1.0
    if scheme in ("lineage_length", "b", "baseline_b"):
        w_lin = float(row.get("sample_weight_inv_lineage") or 1.0)
        w_len = float(row.get("sample_weight_inv_sqrt_len") or 1.0)
        return max(1e-6, w_lin * w_len)
    if scheme in ("lineage_skill", "skill_aware", "baseline_b_skill"):
        w_lin = float(row.get("sample_weight_inv_lineage") or 1.0)
        w_len = float(row.get("sample_weight_inv_sqrt_len") or 1.0)
        w_fam = float(row.get("sample_weight_inv_sqrt_family") or 1.0)
        # Mild boost for Recovery→Precision without dominating
        w_prec = float(row.get("sample_weight_precision_boost") or 1.0)
        return max(1e-6, w_lin * w_len * w_fam * w_prec)
    raise RuntimeError(f"unknown weight scheme {scheme}")


def _annotate_p21_weights(rows: list[dict]) -> None:
    """Fill inverse-frequency weights on P21 traj rows (in-place)."""
    from collections import Counter

    lin_c = Counter(r.get("lineage_id") or r.get("source_lineage") for r in rows)
    fam_c = Counter(r.get("family") for r in rows)
    for r in rows:
        lid = r.get("lineage_id") or r.get("source_lineage")
        fam = r.get("family")
        n_lin = max(1, int(lin_c[lid]))
        n_fam = max(1, int(fam_c[fam]))
        n_steps = max(1, int(r.get("n_steps_with_handoff") or r.get("n_motion") or 1))
        r["sample_weight_inv_lineage"] = float(1.0 / n_lin)
        r["sample_weight_inv_sqrt_len"] = float(1.0 / np.sqrt(n_steps))
        r["sample_weight_inv_sqrt_family"] = float(1.0 / np.sqrt(n_fam))
        r["sample_weight_precision_boost"] = 1.25 if r.get("includes_precision") else 1.0
        r["source_lineage"] = lid


def build_samples(
    manifest: dict[str, Any],
    *,
    split: str,
    scheme: str = "natural",
    dataset_root: Path | None = None,
    window: int = WINDOW,
) -> list[dict[str, Any]]:
    """Expand traj rows into supervised window samples for one split.

    Works for P19 continuation and P21 full-horizon manifests (same NPZ schema).
    Long trajectories are chunked to ``window`` without dropping Stage-1 steps.
    """
    if dataset_root is not None:
        root = Path(dataset_root)
    elif str(manifest.get("dataset_id") or "").startswith("p21"):
        root = P21_DEFAULT
    elif str(manifest.get("start_distribution") or "") == "recovery_ic_pre_stage1":
        root = P21_DEFAULT
    else:
        root = P19_DEFAULT
    rows = [r for r in manifest["trajectories"] if r.get("split") == split]
    if rows and (
        "sample_weight_inv_lineage" not in rows[0]
        or scheme in ("lineage_skill", "skill_aware", "baseline_b_skill")
    ):
        _annotate_p21_weights(rows)
    samples: list[dict[str, Any]] = []
    for row in rows:
        ep = load_episode_npz(root, row["npz_relpath"])
        # Coverage assert: motion steps == n_stage1 + n_precision when provided
        n_s1 = row.get("n_stage1")
        n_pr = row.get("n_precision")
        if n_s1 is not None and n_pr is not None:
            expect = int(n_s1) + int(n_pr)
            n_motion = int(ep["tokens"].shape[0]) - 1  # exclude terminal HANDOFF
            if expect > 0 and n_motion != expect:
                raise RuntimeError(
                    f"{row.get('traj_id')}: motion steps {n_motion} != n_stage1+n_precision {expect}"
                )
        w = episode_weight(row, scheme)
        chunks = chunk_episode(ep, window=window)
        for ci, chunk in enumerate(chunks):
            samples.append(
                {
                    "traj_id": row["traj_id"],
                    "split": split,
                    "chunk_index": ci,
                    "n_chunks": len(chunks),
                    "source_lineage": row.get("lineage_id") or row.get("source_lineage"),
                    "family": row.get("family"),
                    "horizon_kind": row.get("horizon_kind"),
                    "includes_precision": row.get("includes_precision"),
                    "skill_group": row.get("skill_group") or row.get("family"),
                    "sample_weight": w,
                    "n_steps": int(chunk["observations"].shape[0]),
                    "meta_recovery_letter": row.get("meta_recovery_letter") or row.get("family"),
                    "meta_executes_dx": row.get("meta_executes_dx"),
                    "meta_executes_dz": row.get("meta_executes_dz"),
                    "meta_executes_coupled": row.get("meta_executes_coupled"),
                    "meta_is_direct_handoff_only": row.get("meta_is_direct_handoff_only"),
                    "meta_precision_executed_corrections": row.get("includes_precision"),
                    **chunk,
                }
            )
    # Cap extreme weights then renormalize to mean 1
    if samples:
        raw = np.asarray([s["sample_weight"] for s in samples], np.float64)
        med = float(np.median(raw)) if len(raw) else 1.0
        cap = max(med * 5.0, 1e-6)
        for s in samples:
            s["sample_weight"] = float(min(s["sample_weight"], cap))
        mean_w = float(np.mean([s["sample_weight"] for s in samples]))
        if mean_w > 0:
            for s in samples:
                s["sample_weight"] = float(s["sample_weight"] / mean_w)
    return samples


def collate_m111(samples: list[dict[str, Any]], device) -> tuple:
    """Pad to batch max length. Pad positions have mask=False (excluded from loss)."""
    if not samples:
        raise RuntimeError("empty collate")
    length = max(int(s["observations"].shape[0]) for s in samples)
    if length > WINDOW:
        raise RuntimeError(f"collate length {length} exceeds WINDOW={WINDOW}")
    b = len(samples)
    obs = torch.zeros(b, length, OBS_DIM_AIR, device=device)
    prev = torch.zeros(b, length, 7, device=device)
    tokens = torch.zeros(b, length, dtype=torch.long, device=device)
    residuals = torch.zeros(b, length, 7, device=device)
    mask = torch.zeros(b, length, dtype=torch.bool, device=device)
    weights = []
    for i, s in enumerate(samples):
        n = int(s["observations"].shape[0])
        obs[i, :n] = torch.as_tensor(s["observations"], device=device)
        prev[i, :n] = torch.as_tensor(s["prev_actions"], device=device)
        tokens[i, :n] = torch.as_tensor(s["tokens"], device=device)
        residuals[i, :n] = torch.as_tensor(s["residuals"], device=device)
        # Combine stored loss_mask with pad mask; never supervise pads.
        lm = np.asarray(s["loss_mask"], np.bool_)
        mask[i, :n] = torch.as_tensor(lm, device=device)
        weights.append(float(s["sample_weight"]))
        # Safety: input last dim is obs||prev only (checked by model in_proj).
        if s["observations"].shape[1] != OBS_DIM_AIR:
            raise RuntimeError("obs dim leak check failed")
    return (
        obs,
        prev,
        tokens,
        residuals,
        mask,
        torch.tensor(weights, device=device, dtype=torch.float32),
    )


def assert_no_target_in_inputs(obs: torch.Tensor, prev: torch.Tensor, tokens: torch.Tensor) -> None:
    """Sanity: model input rank/shape cannot carry a token channel."""
    if obs.shape[-1] != OBS_DIM_AIR:
        raise RuntimeError(f"obs last dim {obs.shape[-1]} != {OBS_DIM_AIR}")
    if prev.shape[-1] != 7:
        raise RuntimeError(f"prev last dim {prev.shape[-1]} != 7")
    if tokens.dim() != 2:
        raise RuntimeError("tokens must be (B,T) targets, not concatenated into inputs")
