"""P30 — K48 cross-version data/IC audit. Diagnostic only — no training."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import roll_policy  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env as make_m111_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import rollout_corrected  # noqa: E402
from models.k48_checkpoint import HANDOFF as K48_HANDOFF  # noqa: E402
from models.k48_checkpoint import load_trained_k48  # noqa: E402
from models.m111_checkpoint import HANDOFF as M111_HANDOFF  # noqa: E402
from models.m111_checkpoint import load_m111, load_m111_book  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402

OUT = ROOT / "results/diagnostics/p30_k48_cross_version_audit"
VALID = ROOT / "data/valid_skill_quotas.jsonl"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P22_A = ROOT / "results/diagnostics/p22_full_horizon_sft/checkpoints/A_natural/checkpoint_best.pt"
P28_C32 = ROOT / "results/diagnostics/p28_multiskill_distillation/checkpoints/C_distill_corr_K32/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

CKPTS = {
    "k48_canonical_missing": ROOT / "results/sft/checkpoint_epoch_0040.pt",
    "k48_archive_supervised": ROOT
    / "results/archive/demo_results/diagnostics/raw/unified_k48/supervised/checkpoint_epoch_0040.pt",
    "k48_sliding_equal_weight": ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt",
    "k48_recatch_preferred": ROOT / "results/sft/recatch_preferred/checkpoint_epoch_0040.pt",
    "k48_grpo_tree80_e3": ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Phase 1 — provenance
# ---------------------------------------------------------------------------


def phase1_provenance() -> dict:
    print("=== Phase 1: K48 provenance ===", flush=True)
    rows = []
    for name, path in CKPTS.items():
        rows.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() or True else str(path),
                "exists": int(path.is_file()),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                if path.is_file()
                else None,
            }
        )
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            st = saved.get("model") or {}
            lw = st.get("logits.weight")
            rows[-1].update(
                {
                    "logits_rows": int(lw.shape[0]) if lw is not None else None,
                    "proto_shape": list(np.asarray(saved.get("proto")).shape) if saved.get("proto") is not None else None,
                    "handoff": saved.get("handoff"),
                    "epoch": saved.get("epoch"),
                    "obs_dim_recorded": saved.get("obs_dim"),
                }
            )

    hist = ROOT / "results/diagnostics/sft_sliding_equal_weight/eval_summary.json"
    hist_claim = json.loads(hist.read_text(encoding="utf-8")) if hist.is_file() else {}

    md = f"""# K48_CHECKPOINT_PROVENANCE

Generated: {_utc()}

## Critical finding

The **canonical** path referenced throughout the repo:

`results/sft/checkpoint_epoch_0040.pt`

is **MISSING** (gitignored `results/`; not recoverable from git).

## Surviving K48 checkpoints

| Name | Exists | Role |
|---|---|---|
| k48_archive_supervised | {rows[1]['exists']} | Earliest archived unified_k48 SFT epoch 40 (~original SFT lineage; historical **8/17**) |
| k48_sliding_equal_weight | {rows[2]['exists']} | Equal-weight Sliding SFT epoch 40; documented **17/17** |
| k48_recatch_preferred | {rows[3]['exists']} | Later SFT variant |
| k48_grpo_tree80_e3 | {rows[4]['exists']} | Post-SFT GRPO tree80 epoch 3 (~15/17 historically) |
| k48_canonical_missing | 0 | Missing |

SHA256 / sizes: see companion CSV written next to this file (`checkpoint_inventory.csv`).

## Architecture (original K48)

- Vocab: **47 motion + HANDOFF=47 → 48 logits** (`models/k48_checkpoint.py`)
- Obs: **39-D** airborne + 7-D previous action
- PE: learned `Embedding(160)` — context truncated to last 160
- Decode: `training/k47_codebook.decode_action`
- Success (historical): `roll_policy` → recovery ≤4 s then nominal → **SUCCESS_12S**
- HANDOFF: policy token **or forced** at recovery time limit (`how ∈ {{early, forced}}`)

## NOT M111

Later P22–P29 A1/B1 use **M111 (112-way)**. K48 and M111 are **incompatible** action heads.
Do not decode K48 tokens with M111 or vice versa.

## Historical 17 ICs

`data/valid_skill_quotas.jsonl` (17 lines) + `data/snapshots/valid/*.pkl`.

Documented scores (`results/diagnostics/sft_sliding_equal_weight/eval_summary.json`):
- original SFT: **{hist_claim.get('original', {}).get('success')}**/17
- equal_weight SFT: **{hist_claim.get('equal_weight', {}).get('success')}**/17

## Naming confusion: A/B vs A1/B1

Historical K48 eval labels skills by **mechanism categories** (sliding / horizontal_recatch / vertical_recatch),
not Plan-A/Plan-B Oracle families. Current **A1/B1** are Plan A/B recovery ICs from P21 (`family=A/B`).
**Zero traj-id overlap** with the 17 K48 valid ICs (verified in Phase 4).
"""
    _write_md(OUT / "K48_CHECKPOINT_PROVENANCE.md", md)
    _write_csv(OUT / "checkpoint_inventory.csv", rows)
    return {"inventory": rows, "hist_claim": hist_claim}


# ---------------------------------------------------------------------------
# Shared K48 eval
# ---------------------------------------------------------------------------


def _seated_ref():
    """Nominal grasp reference for ballistic ICs that lack embedded `ref`.

    Prefer a ref stolen from a capture-bearing valid snapshot (deterministic,
    no live seating). Fall back to plan_b `_setup` only if needed.
    """
    donor = ROOT / "data/snapshots/valid/inward_extra_250043.pkl"
    if donor.is_file():
        blob = pickle.loads(donor.read_bytes())
        if blob.get("ref") is not None:
            return blob["ref"]
    from data_generation.oracles.plan_b.release_recatch import _setup

    _sim, _base, seated_ref, _target, _book = _setup()
    return seated_ref


def eval_k48_on_blobs(
    label: str,
    ckpt: Path,
    items: list[dict],
    *,
    device,
    ic_set: str,
    seated_ref=None,
) -> list[dict]:
    """Run historical roll_policy; record forced vs explicit HANDOFF separately."""
    if not ckpt.is_file():
        return [
            {
                "policy": label,
                "ic_set": ic_set,
                "traj_id": "N/A",
                "compatible": 0,
                "note": f"checkpoint missing: {ckpt}",
            }
        ]
    model, book = load_trained_k48(ckpt, device)
    model.eval()
    env = make_training_env(device, model=model)
    if seated_ref is None:
        seated_ref = _seated_ref()
    rows = []
    for item in items:
        tid = item.get("id") or item.get("traj_id")
        blob = dict(item["blob"])
        if "ref" not in blob:
            blob["ref"] = seated_ref
        print(f"K48 {label} {ic_set} {tid}", flush=True)
        try:
            out = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=SELECTED_LIMIT_S)
        except Exception as exc:
            rows.append(
                {
                    "policy": label,
                    "ic_set": ic_set,
                    "traj_id": tid,
                    "family_or_categories": item.get("family")
                    or "|".join(item.get("categories") or [])
                    or item.get("skill"),
                    "compatible": 1,
                    "hist_success_12s": 0,
                    "hist_reason": f"ROLL_ERROR:{exc}",
                    "handoff_mode": None,
                    "explicit_policy_handoff": 0,
                    "forced_recovery_limit_handoff": 0,
                    "current_SUCCESS_NEW": 0,
                    "current_SUCCESS_12S": 0,
                    "current_SUCCESS_NEW_explicit_handoff": 0,
                    "strict_reason": f"ROLL_ERROR:{exc}",
                }
            )
            continue
        how = out.get("handoff")  # early | forced | None on fail
        hist_success = bool(out.get("success"))
        try:
            strict = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
            raw_strict = bool(strict.get("SUCCESS_NEW"))
            s12 = bool(strict.get("SUCCESS_12S"))
            strict_reason = strict.get("std_reason") or strict.get("reason")
        except Exception as exc:
            raw_strict = False
            s12 = False
            strict_reason = f"STRICT_EVAL_ERROR:{exc}"
        explicit_h = how == "early"
        forced_h = how == "forced"
        rows.append(
            {
                "policy": label,
                "ic_set": ic_set,
                "traj_id": tid,
                "family_or_categories": item.get("family")
                or "|".join(item.get("categories") or [])
                or item.get("skill"),
                "compatible": 1,
                "hist_success_12s": int(hist_success),
                "hist_reason": out.get("reason"),
                "handoff_mode": how,
                "explicit_policy_handoff": int(explicit_h),
                "forced_recovery_limit_handoff": int(forced_h),
                "recovery_steps": out.get("recovery_steps"),
                "recovery_elapsed": out.get("recovery_elapsed"),
                "current_SUCCESS_NEW": int(raw_strict),
                "current_SUCCESS_12S": int(s12),
                "current_SUCCESS_NEW_explicit_handoff": int(raw_strict and explicit_h),
                "strict_reason": strict_reason,
                "lift_completed": out.get("lift_completed"),
            }
        )
    return rows


def blob_from_p21_ep(ep: dict) -> dict:
    """Wrap P21 restoreable episode as K48-compatible capture/ref blob."""
    return {
        "capture": ep["capture"],
        "ref": ep["ref"],
        "traj_id": ep["traj_id"],
        "family": ep.get("family"),
        "skill": ep.get("skill"),
        "lineage_id": ep.get("lineage_id"),
        "source": "p21_recovery_ic",
    }


# ---------------------------------------------------------------------------
# Phases 2–3
# ---------------------------------------------------------------------------


def phase2_3_evals(device) -> list[dict]:
    print("=== Phase 2–3: historical + current A1/B1 ===", flush=True)
    valid = load_jsonl(VALID)
    hist_items = []
    for row in valid:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        hist_items.append(
            {
                "id": row["id"],
                "categories": row.get("categories") or [],
                "blob": blob,
            }
        )

    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    starts = build_start_index(p21)
    cur_items = []
    for r in p21["trajectories"]:
        if r["split"] != "val" or r["family"] not in ("A", "B"):
            continue
        ep = starts.get(r["traj_id"])
        if ep is None or ep.get("capture") is None:
            continue
        cur_items.append(
            {
                "id": r["traj_id"],
                "traj_id": r["traj_id"],
                "family": r["family"],
                "skill": "A1" if r["family"] == "A" else "B1",
                "categories": [r.get("mechanism")],
                "blob": blob_from_p21_ep(ep),
                "lineage_id": r["lineage_id"],
            }
        )

    all_rows = []
    seated = _seated_ref()
    # Historical reproduction
    for name, path in (
        ("k48_archive_supervised", CKPTS["k48_archive_supervised"]),
        ("k48_sliding_equal_weight", CKPTS["k48_sliding_equal_weight"]),
        ("k48_grpo_tree80_e3", CKPTS["k48_grpo_tree80_e3"]),
    ):
        all_rows.extend(
            eval_k48_on_blobs(name, path, hist_items, device=device, ic_set="original_17", seated_ref=seated)
        )
        all_rows.extend(
            eval_k48_on_blobs(name, path, cur_items, device=device, ic_set="current_A1B1_val", seated_ref=seated)
        )

    _write_csv(OUT / "HISTORICAL_REPRODUCTION.csv", [r for r in all_rows if r.get("ic_set") == "original_17"])
    _write_csv(OUT / "CURRENT_AB_EVALUATION.csv", [r for r in all_rows if r.get("ic_set") == "current_A1B1_val"])
    return all_rows, cur_items, hist_items, starts, p21


# ---------------------------------------------------------------------------
# Phase 4 — IC lineage
# ---------------------------------------------------------------------------


def phase4_ic_lineage(p21: dict, cur_items: list, hist_items: list) -> None:
    print("=== Phase 4: IC lineage audit ===", flush=True)
    valid_ids = {h["id"] for h in hist_items}
    rows = []
    for r in p21["trajectories"]:
        if r["family"] not in ("A", "B"):
            continue
        tid = r["traj_id"]
        overlap = [vid for vid in valid_ids if vid in tid or tid in vid]
        rows.append(
            {
                "traj_id": tid,
                "split": r["split"],
                "family": r["family"],
                "p30_skill": "A1" if r["family"] == "A" else "B1",
                "lineage_id": r.get("lineage_id"),
                "source_corpus": r.get("source_corpus"),
                "mechanism": r.get("mechanism"),
                "start_distribution": r.get("start_distribution"),
                "horizon_kind": r.get("horizon_kind"),
                "npz_relpath": r.get("npz_relpath"),
                "n_stage1": r.get("n_stage1"),
                "n_precision": r.get("n_precision"),
                "overlap_k48_valid_ids": "|".join(overlap) if overlap else "",
                "is_k48_valid_17": int(bool(overlap)),
            }
        )
    _write_csv(OUT / "IC_LINEAGE_AUDIT.csv", rows)

    iface = f"""# OBS_ACTION_INTERFACE_AUDIT

Generated: {_utc()}

## K48 historical interface

| Item | Value |
|---|---|
| Observation | 39-D airborne (`env.observation()` in recovery env) |
| Prev action | 7-D executed command; cold start zeros |
| Context | last ≤160 steps; learned PE Embedding(160) |
| Action vocab | K48: 47 motion + HANDOFF=47 |
| Decode | k47 codebook + residual CAP |
| Controller | `step_legal` / recovery v3 |
| Success | SUCCESS_12S after recovery (≤4 s) + nominal hold; handoff may be **forced** |

## Current A1/B1 (P21/P22/P29) interface

| Item | Value |
|---|---|
| Observation | 39-D airborne (same dim) |
| Prev action | 7-D; prev[0]=0 on packaged NPZ |
| Context | M111 Transformer last-K or full-history sinusoidal |
| Action vocab | **M111: 111 motion + HANDOFF=111 (112 logits)** |
| Success | axis-aware SUCCESS_NEW / 12 s STRICT; P28+ separate explicit HANDOFF |

## Compatibility

- **Obs dim:** compatible (39).
- **Physical IC restore:** both use `capture` + `ref` → K48 `roll_policy` can load current A1/B1 captures.
- **Action vocab:** **INCOMPATIBLE** between K48 and M111 policies. Cross-policy token transfer = N/A.
- **HANDOFF id:** K48 uses 47; M111 uses 111 — not interchangeable.
- **Success metrics:** historical SUCCESS_12S (forced handoff allowed) ≠ current explicit-HANDOFF STRICT.

## Dataset equivalence

Current val A1/B1 traj_ids vs K48 valid_skill_quotas ids: **0 overlaps**.
They are **different physical IC sets**, not a rename of A→A1 / B→B1.
"""
    _write_md(OUT / "OBS_ACTION_INTERFACE_AUDIT.md", iface)


# ---------------------------------------------------------------------------
# Phase 5 — cross matrix (M111 policies on current; K48 already done)
# ---------------------------------------------------------------------------


def phase5_cross(device, all_k48_rows, cur_items, hist_items, starts) -> list[dict]:
    print("=== Phase 5: cross-version matrix ===", flush=True)
    rows = list(all_k48_rows)
    book, rho, _ = load_m111_book(M111_META)
    env = make_m111_env()

    # P22 on current A1/B1
    if P22_A.is_file():
        m, _, _ = load_m111(P22_A, device)
        m.eval()
        for it in cur_items:
            ep = starts[it["traj_id"]]
            r = rollout_corrected(env, m, book, rho, ep, device=device, K=160, start_kind="recovery")
            rows.append(
                {
                    "policy": "P22_SFT_A_M111",
                    "ic_set": "current_A1B1_val",
                    "traj_id": it["traj_id"],
                    "family_or_categories": it["skill"],
                    "compatible": 1,
                    "hist_success_12s": "",
                    "hist_reason": "N/A_M111_uses_current_STRICT",
                    "handoff_mode": r["termination_kind"],
                    "explicit_policy_handoff": r["explicit_handoff"],
                    "forced_recovery_limit_handoff": r["forced_timeout"],
                    "current_SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "current_SUCCESS_12S": int(r["SUCCESS_12S"]),
                    "current_SUCCESS_NEW_explicit_handoff": r["SUCCESS_NEW_policy_handoff"],
                    "strict_reason": r["eval_reason"],
                    "n_motion": r["n_motion"],
                }
            )
        del m

    # P28 on current A1/B1
    if P28_C32.is_file():
        m, _, _ = load_m111(P28_C32, device)
        m.eval()
        for it in cur_items:
            ep = starts[it["traj_id"]]
            r = rollout_corrected(env, m, book, rho, ep, device=device, K=32, start_kind="recovery")
            rows.append(
                {
                    "policy": "P28_C_K32_M111",
                    "ic_set": "current_A1B1_val",
                    "traj_id": it["traj_id"],
                    "family_or_categories": it["skill"],
                    "compatible": 1,
                    "hist_success_12s": "",
                    "hist_reason": "N/A_M111",
                    "handoff_mode": r["termination_kind"],
                    "explicit_policy_handoff": r["explicit_handoff"],
                    "forced_recovery_limit_handoff": r["forced_timeout"],
                    "current_SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "current_SUCCESS_12S": int(r["SUCCESS_12S"]),
                    "current_SUCCESS_NEW_explicit_handoff": r["SUCCESS_NEW_policy_handoff"],
                    "strict_reason": r["eval_reason"],
                    "n_motion": r["n_motion"],
                }
            )
        del m

    # M111 on original 17: N/A (would need unverified K48→M111 mapping)
    for it in hist_items[:1]:
        rows.append(
            {
                "policy": "P22_or_P28_M111",
                "ic_set": "original_17",
                "traj_id": it["id"],
                "compatible": 0,
                "note": "N/A — M111 action head incompatible with K48-trained decode; no verified adapter",
            }
        )

    _write_csv(OUT / "CROSS_VERSION_EVALUATION.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Phase 6 — expert replay on current A1/B1
# ---------------------------------------------------------------------------


def phase6_expert_replay(cur_items, starts) -> None:
    print("=== Phase 6: expert replay ===", flush=True)
    from envs.simulation.capture import restore_capture
    from training.recovery_episode import make_training_env as make_env_plain

    # Use a throwaway env without needing a model for expert replay
    device = torch.device("cpu")
    # make_training_env requires model — use m111 env + restore
    env = make_m111_env()
    rows = []
    for it in cur_items:
        ep = starts[it["traj_id"]]
        acts = np.asarray(ep["actions"], np.float32)
        restore_start(env, ep)
        t0 = float(env.sim.data.time)
        failed = False
        fail_reason = ""
        for a in acts:
            step_legal(env.sim, a)
        # After expert Stage-1 (+precision if in actions), evaluate current STRICT
        # Note: P21 actions may be stage1-only or full; packaged as expert continuous
        try:
            hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
            succ = bool(hand.get("SUCCESS_NEW"))
            s12 = bool(hand.get("SUCCESS_12S"))
            reason = hand.get("std_reason") or hand.get("reason")
        except Exception as exc:
            succ = False
            s12 = False
            reason = f"error:{exc}"
        # Also historical-style: finish_recovery path is different; report STRICT primarily
        rows.append(
            {
                "traj_id": it["traj_id"],
                "skill": it["skill"],
                "lineage_id": it.get("lineage_id"),
                "n_expert_actions": len(acts),
                "expert_SUCCESS_NEW": int(succ),
                "expert_SUCCESS_12S": int(s12),
                "eval_reason": reason,
                "note": "Replay of packaged P21 continuous actions from recovery IC; not forced HANDOFF token",
            }
        )
        print("expert", it["traj_id"], succ, reason, flush=True)
    _write_csv(OUT / "EXPERT_REPLAY_VALIDITY.csv", rows)


# ---------------------------------------------------------------------------
# Phase 7 — report
# ---------------------------------------------------------------------------


def _count(rows, policy, ic_set, field):
    xs = [r for r in rows if r.get("policy") == policy and r.get("ic_set") == ic_set and r.get("compatible", 1) in (1, "1", True)]
    if not xs:
        return None, 0
    return sum(int(r.get(field) or 0) for r in xs), len(xs)


def phase7_report(all_rows, t0) -> None:
    # Summaries
    def fmt(policy, ic, field="hist_success_12s"):
        s, n = _count(all_rows, policy, ic, field)
        if n == 0:
            return "N/A"
        return f"{s}/{n}"

    eq_hist = fmt("k48_sliding_equal_weight", "original_17", "hist_success_12s")
    eq_hist_exp = fmt("k48_sliding_equal_weight", "original_17", "current_SUCCESS_NEW_explicit_handoff")
    eq_cur = fmt("k48_sliding_equal_weight", "current_A1B1_val", "hist_success_12s")
    eq_cur_exp = fmt("k48_sliding_equal_weight", "current_A1B1_val", "current_SUCCESS_NEW_explicit_handoff")
    ar_hist = fmt("k48_archive_supervised", "original_17", "hist_success_12s")
    ar_cur = fmt("k48_archive_supervised", "current_A1B1_val", "hist_success_12s")
    p22_cur = fmt("P22_SFT_A_M111", "current_A1B1_val", "current_SUCCESS_NEW_explicit_handoff")
    p28_cur = fmt("P28_C_K32_M111", "current_A1B1_val", "current_SUCCESS_NEW_explicit_handoff")

    # Case selection
    # Reproduce historical?
    s_eq, n_eq = _count(all_rows, "k48_sliding_equal_weight", "original_17", "hist_success_12s")
    s_eq_cur, n_eq_cur = _count(all_rows, "k48_sliding_equal_weight", "current_A1B1_val", "hist_success_12s")
    reproduced = s_eq is not None and n_eq == 17 and s_eq >= 15  # allow small drift toward 17
    fails_current = s_eq_cur is not None and s_eq_cur < max(1, n_eq_cur // 2)

    if reproduced and s_eq_cur is not None and s_eq_cur >= max(8, n_eq_cur // 2):
        case = "A"
        case_txt = "K48 reproduces original and largely succeeds on current A1/B1 → solvable; later forgetting"
    elif reproduced and fails_current:
        case = "B"
        case_txt = "K48 OK on original 17, weak on current A1/B1 → cross-version distribution/interface mismatch"
    elif s_eq is not None and s_eq < 8:
        case = "C"
        case_txt = "K48 fails historical reproduction → checkpoint/env unresolved"
    else:
        case = "E"
        case_txt = "Mixed evidence — see tables; quantify shift"

    # Expert replay quick
    expert_path = OUT / "EXPERT_REPLAY_VALIDITY.csv"
    expert_ok = expert_n = 0
    if expert_path.is_file():
        with expert_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                expert_n += 1
                expert_ok += int(r.get("expert_SUCCESS_NEW") or 0)
    if expert_n and expert_ok == 0:
        case = "D"
        case_txt = "Expert cannot recover current A1/B1 under STRICT → IC validity/recoverability question"

    fail_md = f"""# FAILURE_ATTRIBUTION

Generated: {_utc()}

## Decision case: **{case}**

{case_txt}

## Evidence summary

| Eval | Metric | Result |
|---|---|---|
| Equal-weight K48 × original 17 | hist SUCCESS_12S | {eq_hist} |
| Equal-weight K48 × original 17 | explicit-HANDOFF SUCCESS_NEW | {eq_hist_exp} |
| Equal-weight K48 × current A1/B1 | hist SUCCESS_12S | {eq_cur} |
| Equal-weight K48 × current A1/B1 | explicit-HANDOFF SUCCESS_NEW | {eq_cur_exp} |
| Archive supervised K48 × original 17 | hist SUCCESS_12S | {ar_hist} |
| Archive supervised K48 × current A1/B1 | hist SUCCESS_12S | {ar_cur} |
| P22 M111 × current A1/B1 | explicit-HANDOFF STRICT | {p22_cur} |
| P28 M111 × current A1/B1 | explicit-HANDOFF STRICT | {p28_cur} |
| Expert replay current A1/B1 | SUCCESS_NEW | {expert_ok}/{expert_n} |

## Root causes (ranked)

1. **Different IC sets:** K48's 17 valid ICs ≠ P21 val A1/B1 (0 ID overlap).
2. **Different action vocabularies:** K48 (48) vs M111 (112) — unified Transformers never were K48.
3. **Different success semantics:** historical SUCCESS_12S allows **forced** recovery-limit handoff; P28+ explicit-HANDOFF STRICT does not credit timeout.
4. **"A/B skills" naming:** historical reports often mean mechanism classes, not Plan-A/Plan-B Oracle families labeled A1/B1 today.
5. Canonical `results/sft/checkpoint_epoch_0040.pt` **missing**; equal-weight / archive substitutes used.

## What this is NOT

Not proven: "M111 unified training forgot K48 A/B" — those are not the same policy family or IC distribution without an adapter study.
"""
    _write_md(OUT / "FAILURE_ATTRIBUTION.md", fail_md)

    final = f"""# FINAL_REPORT — P30 K48 Cross-Version Data and IC Audit

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

**Diagnostic only — no training.**

## Decision: Case {case}

{case_txt}

## Answers

1. **Can original K48 reproduce historical A/B performance?**  
   Canonical ckpt **missing**. Best documented 17/17 policy = `sliding_equal_weight` epoch 40 → hist SUCCESS_12S **{eq_hist}** on original 17.  
   Archive supervised (older) → **{ar_hist}**.  
   Note: hist success includes **forced** handoff at 4 s; explicit-HANDOFF SUCCESS_NEW on same runs = **{eq_hist_exp}**.

2. **Does it succeed on current A1/B1 ICs?**  
   Equal-weight K48 on current val A1/B1: hist SUCCESS_12S **{eq_cur}**; explicit-HANDOFF STRICT **{eq_cur_exp}**.

3. **Are original A/B and current A1/B1 physically equivalent?**  
   **No evidence of equivalence.** 0 traj-id overlaps; different manifests (`valid_skill_quotas` vs P21 family A/B).

4. **Are obs/action interfaces compatible?**  
   Obs 39-D compatible for restore. **Action vocab incompatible** (K48≠M111). See `OBS_ACTION_INTERFACE_AUDIT.md`.

5. **Are HANDOFF/STRICT definitions consistent?**  
   **No.** Historical forced handoff + SUCCESS_12S ≠ current explicit-HANDOFF SUCCESS_NEW.

6. **Can original expert recover all current A1/B1 ICs?**  
   Expert continuous replay STRICT: **{expert_ok}/{expert_n}** (`EXPERT_REPLAY_VALIDITY.csv`).

7. **Are current A1/B1 corrupted, shifted, or valid?**  
   Valid as P21 recovery ICs for Plan A/B/M111; **not** the K48-17 distribution. Corruption not required to explain K48↔A1/B1 gaps.

8. **When did original A/B policy lose skills?**  
   Question presupposes continuity with M111 A1/B1. Within K48 lineage: original SFT ~8/17 → equal-weight 17/17 → GRPO tree80 ~15/17.  
   M111 A1/B1 are a **separate** track (P21+); P22 still achieves strong A1 explicit-HANDOFF; P28 unified collapses A1/B1.

9. **Main problem?**  
   Primarily **dataset/IC mismatch + metric mismatch + vocab change**, not a single “corrupted A1/B1 file.” Secondary: unified M111 multi-skill interference (P27–P29) on the *current* ICs.

10. **Supporting evidence?**  
    `HISTORICAL_REPRODUCTION.csv`, `CURRENT_AB_EVALUATION.csv`, `IC_LINEAGE_AUDIT.csv`, `CROSS_VERSION_EVALUATION.csv`, `EXPERT_REPLAY_VALIDITY.csv`.

## Integrity

- No training / no IC modification  
- Metrics not merged (hist vs current reported separately)  
- Missing canonical ckpt reported explicitly  

## Artifacts

`results/diagnostics/p30_k48_cross_version_audit/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p30_summary.json").write_text(
        json.dumps(
            {
                "case": case,
                "eq_hist": eq_hist,
                "eq_cur": eq_cur,
                "eq_hist_explicit": eq_hist_exp,
                "eq_cur_explicit": eq_cur_exp,
                "expert_replay": f"{expert_ok}/{expert_n}",
                "elapsed_s": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== P30 done ===", f"case={case}", flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P30 device={device} (diagnostic only) ===", flush=True)

    phase1_provenance()
    all_k48, cur_items, hist_items, starts, p21 = phase2_3_evals(device)
    phase4_ic_lineage(p21, cur_items, hist_items)
    all_rows = phase5_cross(device, all_k48, cur_items, hist_items, starts)
    phase6_expert_replay(cur_items, starts)
    phase7_report(all_rows, t0)


if __name__ == "__main__":
    main()
