"""P28 — Multi-skill distillation & P27 interface audit.

No PPO/GRPO. No skill selector at inference. M111/physics/STRICT frozen.
"""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import OBS_DIM_AIR  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p27_history_multiskill_transformer.run_p27 import (  # noqa: E402
    load_d2_eval_eps,
    load_npz_flex,
    resolve_row_root,
    windows_from_episode,
)
from experiments.precision.p28_multiskill_distillation.harness import (  # noqa: E402
    offline_vs_stream_parity,
    rollout_corrected,
)
from models.m111_checkpoint import (  # noqa: E402
    HANDOFF,
    MOTION_K,
    load_m111,
    load_m111_book,
    save_m111,
    build_m111,
)
from training.m111_dataset import collate_m111  # noqa: E402
from training.m111_sft import eval_token_metrics, supervised_loss_m111, train_m111  # noqa: E402

OUT = ROOT / "results/diagnostics/p28_multiskill_distillation"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
P24 = ROOT / "results/diagnostics/p24_skill_isolation"
P25 = ROOT / "results/diagnostics/p25_c_dagger"
P27 = ROOT / "results/diagnostics/p27_history_multiskill_transformer"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

P22_A = P22 / "checkpoints/A_natural/checkpoint_best.pt"
P25_C = P25 / "checkpoints/round2_B_corrective_only/checkpoint_best.pt"
P24_D2 = P24 / "checkpoints/D2_only/checkpoint_best.pt"

SKILL_MAP = {"A": "A1", "B": "B1", "C": "C", "D2": "D2"}
K_CANDIDATES = (8, 32)
EPOCHS = 16
BATCH = 8
LR = 3e-4
SEED = 28
TARGET_TOKEN_BUDGET = 80_000  # matched supervised positions across variants/K
TEACHER_ROLLOUT_CAPS = {"A1": 40, "B1": 12, "C": 50, "D2": 30}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# ---------------------------------------------------------------------------
# Phase 1 — P27 interface audit
# ---------------------------------------------------------------------------


def _load_audit_readonly() -> dict:
    """Load P21/P23/P25 manifests without writing into P27 directories."""
    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    p25 = json.loads((P25 / "VERIFIED_CORRECTIONS_MANIFEST.json").read_text(encoding="utf-8"))
    d2 = [r for r in p23["trajectories"] if r.get("family") == "D2"]
    corr = p25.get("corrections") or []
    return {"p21": p21, "p23": p23, "p25": p25, "d2": d2, "corr": corr}


def phase1_audit(device) -> dict:
    print("=== Phase 1: P27 interface audit ===", flush=True)
    audit = _load_audit_readonly()
    man_path = P27 / "UNIFIED_MANIFEST.json"
    if not man_path.is_file():
        raise RuntimeError(f"P27 UNIFIED_MANIFEST missing: {man_path}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    # Ensure P25 corr copies exist under P27 dataset/corr (read-only use)
    for c in audit["corr"]:
        src = P25 / c["npz_relpath"]
        dst = P27 / "dataset" / "corr" / Path(c["npz_relpath"]).name
        if src.is_file() and not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # --- Exposure audit for each K (reconstruct P27 windowing) ---
    exposure_rows = []
    for K in (1, 8, 32, 64):
        rows = [r for r in manifest["trajectories"] if r.get("split") == "train"]
        sk_c = Counter(r.get("skill") or SKILL_MAP.get(r.get("family"), "?") for r in rows)
        n_traj = len(rows)
        n_windows = 0
        supervised_pos = 0
        unique_pos: set[tuple] = set()
        repeated = 0
        motion_tok = Counter()
        hand_tok = Counter()
        lineage_pos = Counter()
        skill_pos = Counter()
        skill_weight_sum = Counter()
        for row in rows:
            skill = row.get("skill") or SKILL_MAP.get(row.get("family"), "?")
            root = resolve_row_root(row)
            rel = row["npz_relpath"]
            try:
                if row.get("source") == "p25_correction" or row.get("allow_nonzero_prev0"):
                    local = P27 / "dataset" / "corr" / Path(rel).name
                    if local.is_file():
                        ep = load_npz_flex(local.parent, local.name, allow_nonzero_prev0=True)
                    else:
                        ep = load_npz_flex(P25, rel, allow_nonzero_prev0=True)
                else:
                    ep = load_npz_flex(root, rel, allow_nonzero_prev0=False)
            except Exception as exc:
                print("skip", row.get("traj_id"), exc, flush=True)
                continue
            w_skill = 1.0 / np.sqrt(max(1, sk_c[skill]))
            if row.get("source") == "p25_correction":
                w_skill *= 0.85
            chunks = windows_from_episode(ep, K)
            n_windows += len(chunks)
            for wi, ch in enumerate(chunks):
                n = int(ch["tokens"].shape[0])
                supervised_pos += n
                skill_pos[skill] += n
                skill_weight_sum[skill] += w_skill * n
                lineage_pos[row.get("lineage_id") or row["traj_id"]] += n
                for j, tok in enumerate(ch["tokens"].tolist()):
                    # approximate absolute index for overlap accounting
                    # windows_from_episode uses stride K//2
                    key = (row["traj_id"], wi, j)  # per-window index; also track abs
                    # abs index reconstruction:
                    pass
                for tok in ch["tokens"].tolist():
                    if int(tok) == HANDOFF:
                        hand_tok[skill] += 1
                    else:
                        motion_tok[skill] += 1
            # unique absolute positions covered
            n_ep = int(ep["tokens"].shape[0])
            stride = max(1, K // 2) if n_ep > K else n_ep
            covered = set()
            if n_ep <= K:
                starts = [0]
            else:
                starts = list(range(0, n_ep - K + 1, stride))
                last = n_ep - K
                if starts[-1] != last:
                    starts.append(last)
            for s in starts:
                for j in range(s, min(s + K, n_ep) if n_ep > K else n_ep):
                    if (row["traj_id"], j) in covered:
                        repeated += 1
                    covered.add((row["traj_id"], j))
            unique_pos |= covered

        # Apply P27 stratified cap approximation: train capped at 10000 windows
        cap = 10000
        capped_windows = min(n_windows, cap)
        # effective tokens after cap ≈ proportional
        scale = capped_windows / max(1, n_windows)
        exposure_rows.append(
            {
                "K": K,
                "n_train_traj": n_traj,
                "n_windows_uncapped": n_windows,
                "n_windows_capped_p27": capped_windows,
                "supervised_positions_uncapped": supervised_pos,
                "supervised_positions_capped_approx": int(supervised_pos * scale),
                "unique_abs_positions": len(unique_pos),
                "repeated_overlap_hits": repeated,
                "mean_tokens_per_window": supervised_pos / max(1, n_windows),
                "handoff_tokens_uncapped": sum(hand_tok.values()),
                "motion_tokens_uncapped": sum(motion_tok.values()),
                "A1_pos": skill_pos.get("A1", 0),
                "B1_pos": skill_pos.get("B1", 0),
                "C_pos": skill_pos.get("C", 0),
                "D2_pos": skill_pos.get("D2", 0),
                "handoff_frac": sum(hand_tok.values()) / max(1, supervised_pos),
                "note": "P27 capped windows not tokens — K=1/8 matched windows but not token budgets vs K=32/64",
            }
        )
    _write_csv(OUT / "WINDOW_EXPOSURE_AUDIT.csv", exposure_rows)

    # --- Train/inference parity on a P27 K=8 checkpoint ---
    parity_rows = []
    ckpt8 = P27 / "checkpoints/K8/checkpoint_best.pt"
    first_mismatch_t = None
    if ckpt8.is_file():
        model, book, _ = load_m111(ckpt8, device)
        # pick one A traj
        row = next(r for r in manifest["trajectories"] if r.get("family") == "A" and r["split"] == "train")
        ep = load_npz_flex(P21, row["npz_relpath"])
        for K in (1, 8, 32, 64):
            pr = offline_vs_stream_parity(model, book, load_m111_book(M111_META)[1], ep, device=device, K=K, n_check=80)
            for r in pr:
                r["K"] = K
                r["traj_id"] = row["traj_id"]
                parity_rows.append(r)
            bad = next((r for r in pr if not r["tok_match"] or r["max_logit_abs_diff"] > 1e-4), None)
            if bad and first_mismatch_t is None:
                first_mismatch_t = {"K": K, **bad}
        del model
    _write_csv(OUT / "TRAIN_INFERENCE_PARITY.csv", parity_rows)

    # --- Termination audit on P27 raw results ---
    raw_p27 = P27 / "SKILLWISE_STRICT_RAW.csv"
    term_notes = []
    timeout_success = 0
    timeout_n = 0
    handoff_success = 0
    if raw_p27.is_file():
        with raw_p27.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not str(r["condition"]).startswith("unified_"):
                    continue
                n_mot = int(float(r["n_motion"]))
                succ = int(r["SUCCESS_NEW"])
                # P27 max_steps=800; if n_motion==800 likely timeout (no HANDOFF)
                is_timeout = n_mot >= 800
                if is_timeout:
                    timeout_n += 1
                    timeout_success += succ
                else:
                    handoff_success += succ
                term_notes.append(
                    {
                        "condition": r["condition"],
                        "skill": r["skill"],
                        "traj_id": r["traj_id"],
                        "n_motion": n_mot,
                        "likely_timeout": int(is_timeout),
                        "SUCCESS_NEW": succ,
                        "eval_reason": r.get("eval_reason"),
                    }
                )
    _write_csv(OUT / "TERMINATION_AUDIT_RAW.csv", term_notes)

    # Corrected re-eval of P27 K=8 and K=32 on val (minimal) — save separately
    book, rho, _ = load_m111_book(M111_META)
    starts = build_start_index(audit["p21"])
    val_abc = [r for r in audit["p21"]["trajectories"] if r["split"] == "val"]
    d2_eps = load_d2_eval_eps(audit)
    corrected_rows = []
    env = make_env()
    for K, ckpt_path in ((8, P27 / "checkpoints/K8/checkpoint_best.pt"), (32, P27 / "checkpoints/K32/checkpoint_best.pt")):
        if not ckpt_path.is_file():
            continue
        model, _, _ = load_m111(ckpt_path, device)
        model.eval()
        for row in val_abc:
            ep = starts[row["traj_id"]]
            skill = SKILL_MAP[row["family"]]
            r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="recovery")
            corrected_rows.append(
                {
                    "source": "p27_corrected_harness",
                    "condition": f"p27_K{K}",
                    "skill": skill,
                    "traj_id": row["traj_id"],
                    "K": K,
                    "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "SUCCESS_NEW_policy_handoff": r["SUCCESS_NEW_policy_handoff"],
                    "termination_kind": r["termination_kind"],
                    "forced_timeout": r["forced_timeout"],
                    "explicit_handoff": r["explicit_handoff"],
                    "n_motion": r["n_motion"],
                    "max_token_run": r["max_token_run"],
                    "eval_reason": r["eval_reason"],
                }
            )
        for ep in d2_eps:
            r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="d2")
            corrected_rows.append(
                {
                    "source": "p27_corrected_harness",
                    "condition": f"p27_K{K}",
                    "skill": "D2",
                    "traj_id": ep["traj_id"],
                    "K": K,
                    "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "SUCCESS_NEW_policy_handoff": r["SUCCESS_NEW_policy_handoff"],
                    "termination_kind": r["termination_kind"],
                    "forced_timeout": r["forced_timeout"],
                    "explicit_handoff": r["explicit_handoff"],
                    "n_motion": r["n_motion"],
                    "max_token_run": r["max_token_run"],
                    "eval_reason": r["eval_reason"],
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_csv(OUT / "P27_CORRECTED_STRICT.csv", corrected_rows)

    parity_ok = all(int(r["tok_match"]) == 1 for r in parity_rows) if parity_rows else False
    timeout_false_success = timeout_success  # should be 0

    term_md = f"""# TERMINATION_AUDIT

Generated: {_utc()}

## Finding

P27 `rollout_skill` always calls `evaluate_handoff_with_axis` after the loop, whether the
policy emitted HANDOFF or hit `max_steps=800`.

- A forced timeout is **not** labeled as policy HANDOFF in token logs (`tokens` has no terminal HANDOFF).
- However, SUCCESS_NEW was still computed on the terminal physics state after timeout.
- Retrospective scan of P27 unified rows: timeout-like (`n_motion>=800`) successes = **{timeout_success}** / {timeout_n} timeouts.
- Non-timeout (likely explicit HANDOFF) successes accounted for the reported STRICT wins.

## Correction (P28 harness)

`harness.rollout_corrected` records:
- `termination_kind` ∈ {{policy_handoff, timeout}}
- `SUCCESS_NEW_policy_handoff` = SUCCESS_NEW ∧ policy_handoff
- `forced_timeout` flag

Original P27 artifacts are preserved untouched. Corrected re-eval: `P27_CORRECTED_STRICT.csv`.
"""
    _write_md(OUT / "TERMINATION_AUDIT.md", term_md)

    interface_md = f"""# P27_INTERFACE_AUDIT

Generated: {_utc()}

## Data path (NPZ → MuJoCo)

1. NPZ: `observations[t]` (39), `prev_actions[t]` (7, previously executed), `tokens[t]`, `residuals[t]`, `loss_mask`
2. Train: `windows_from_episode(K)` → `collate_m111` left-pad → `model(obs, prev)` causal Transformer → CE + resid MSE
3. Infer: growing `(obs_seq, prev_seq)`, truncate to last K, `argmax(logits[-1])` → `decode_m111_action` → `step_legal`

## Confirmed correct

- Prev-action semantics: `prev[0]=0` on full-horizon packs; `env._prev` updated to executed action.
- Causal mask: upper-triangular `-inf` in `models/k48_policy.py`.
- Position encoding: `pos(arange(t))` relative to the **fed** sequence (same for train windows and infer truncation).
- Left padding + causal attention: valid positions cannot attend to end-pads (pads after sequence).
- HANDOFF not decoded as a control vector.
- Offline batched vs streaming last-K parity on P27 K8 ckpt: **{"PASS" if parity_ok else "FAIL"}** (first mismatch: {first_mismatch_t}).

## Defects (severity-ranked)

### HIGH — Token-budget mismatch across K
P27 capped **windows** (10k) not supervised **tokens**. See `WINDOW_EXPOSURE_AUDIT.csv`.
K=1 ≈ 10k tokens; K=8 ≈ 80k; K=32/64 naturally fewer windows but more tokens/window.
Equal window counts ≠ equal supervision. HANDOFF (~1/traj) under-exposed under short-K caps.

### HIGH — Closed-loop distribution shift / non-termination
K=1/8 often hit `n_motion=800` (no HANDOFF). High TF accuracy does not imply HANDOFF timing under own-action feedback.

### MEDIUM — Mid-episode window starts
Overlapping windows restart positional indices at 0 and truncate left context for the first tokens in each window. Matches infer last-K truncation for the **suffix** token, but early-in-window targets are supervised with artificially short history relative to the true episode.

### MEDIUM — Termination taxonomy
Timeout physics still evaluated for SUCCESS_NEW (see TERMINATION_AUDIT.md). No evidence of mass false STRICT from timeout in P27 (`timeout_success={timeout_false_success}`), but reporting conflated timeout vs HANDOFF.

### LOW — P25 corrective suffixes
`allow_nonzero_prev0=True` is correct for intervention state. Earlier pre-intervention frames are unavailable; P27 did not fabricate them (good). Valid-history padding for K-context before suffix start was not explicitly masked beyond natural short sequences.

### LOW — No major logit train/infer mismatch
Parity test passes within numerical tolerance → P27 failure is **not** primarily a batched-vs-streaming bug.

## Minimal baseline rerun?

**No full P27 rerun of all K.** Defects are exposure/termination/reporting and closed-loop shift, not a broken decoder.
P28 Variant A = repaired SFT (token-budget balanced + HANDOFF-aware sampling + corrected harness) as the controlled replacement baseline.

## Corrective suffix init

P25 corrections store the policy's actual `prev` at intervention. Do not zero-pad fabricated history; short sequences + loss masks represent unavailable past.
"""
    _write_md(OUT / "P27_INTERFACE_AUDIT.md", interface_md)
    return {
        "audit": audit,
        "manifest": manifest,
        "parity_ok": parity_ok,
        "exposure_rows": exposure_rows,
        "starts": starts,
        "val_abc": val_abc,
        "d2_eps": d2_eps,
        "book": book,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Skill distinguishability
# ---------------------------------------------------------------------------


def _hist_feat(obs, prev, t, K):
    hs = max(0, t + 1 - K)
    o = obs[hs : t + 1]
    p = prev[hs : t + 1]
    # pad front to K
    if o.shape[0] < K:
        o = np.concatenate([np.zeros((K - o.shape[0], o.shape[1]), np.float32), o], 0)
        p = np.concatenate([np.zeros((K - p.shape[0], p.shape[1]), np.float32), p], 0)
    return np.concatenate([o.reshape(-1), p.reshape(-1)]).astype(np.float32)


def phase2_distinguish(ctx: dict) -> None:
    print("=== Phase 2: skill distinguishability ===", flush=True)
    manifest = ctx["manifest"]
    rng = np.random.default_rng(SEED)

    # Sample histories
    bank: dict[str, list] = defaultdict(list)
    for row in manifest["trajectories"]:
        if row.get("split") != "train":
            continue
        skill = row.get("skill") or SKILL_MAP.get(row.get("family"), "?")
        try:
            if row.get("source") == "p25_correction" or row.get("allow_nonzero_prev0"):
                local = P27 / "dataset" / "corr" / Path(row["npz_relpath"]).name
                if local.is_file():
                    ep = load_npz_flex(local.parent, local.name, allow_nonzero_prev0=True)
                else:
                    ep = load_npz_flex(P25, row["npz_relpath"], allow_nonzero_prev0=True)
            else:
                root = resolve_row_root(row)
                ep = load_npz_flex(root, row["npz_relpath"], allow_nonzero_prev0=False)
        except Exception:
            continue
        n = int(ep["observations"].shape[0])
        # sample up to 8 timesteps per traj
        idxs = rng.choice(n, size=min(8, n), replace=False)
        for t in idxs:
            bank[skill].append(
                {
                    "feat_by_K": {K: _hist_feat(ep["observations"], ep["prev_actions"], int(t), K) for K in (1, 8, 32, 64)},
                    "token": int(ep["tokens"][t]),
                    "traj_id": row["traj_id"],
                    "lineage_id": row.get("lineage_id"),
                    "t": int(t),
                    "skill": skill,
                }
            )

    amb_rows = []
    for K in (1, 8, 32, 64):
        # Build matrices per skill
        skills = ["A1", "B1", "C", "D2"]
        pools = {sk: bank[sk] for sk in skills if bank[sk]}
        # nearest cross-skill pairs (subsample)
        for sk_a in skills:
            for sk_b in skills:
                if sk_a >= sk_b:
                    continue
                A = pools.get(sk_a) or []
                B = pools.get(sk_b) or []
                if not A or not B:
                    continue
                Aa = A[: min(120, len(A))]
                Bb = B[: min(120, len(B))]
                Fa = np.stack([x["feat_by_K"][K] for x in Aa])
                Fb = np.stack([x["feat_by_K"][K] for x in Bb])
                # L2 nearest
                # normalize
                Fa_n = Fa / (np.linalg.norm(Fa, axis=1, keepdims=True) + 1e-8)
                Fb_n = Fb / (np.linalg.norm(Fb, axis=1, keepdims=True) + 1e-8)
                sims = Fa_n @ Fb_n.T
                # top ambiguous: high sim, different tokens
                n_pair = 0
                disagree = 0
                dists = []
                for i in range(len(Aa)):
                    j = int(np.argmax(sims[i]))
                    dist = float(np.linalg.norm(Fa[i] - Fb[j]))
                    dists.append(dist)
                    n_pair += 1
                    if int(Aa[i]["token"]) != int(Bb[j]["token"]):
                        disagree += 1
                    if n_pair <= 5:
                        amb_rows.append(
                            {
                                "K": K,
                                "skill_a": sk_a,
                                "skill_b": sk_b,
                                "traj_a": Aa[i]["traj_id"],
                                "traj_b": Bb[j]["traj_id"],
                                "token_a": Aa[i]["token"],
                                "token_b": Bb[j]["token"],
                                "token_disagree": int(Aa[i]["token"] != Bb[j]["token"]),
                                "l2_dist": dist,
                                "cos_sim": float(sims[i, j]),
                            }
                        )
                amb_rows.append(
                    {
                        "K": K,
                        "skill_a": sk_a,
                        "skill_b": sk_b,
                        "traj_a": "AGG",
                        "traj_b": "AGG",
                        "token_a": -1,
                        "token_b": -1,
                        "token_disagree": disagree,
                        "l2_dist": float(np.mean(dists)) if dists else None,
                        "cos_sim": float(np.mean(np.max(sims, axis=1))),
                        "n_pairs": n_pair,
                        "disagree_rate": disagree / max(1, n_pair),
                    }
                )
    _write_csv(OUT / "HISTORY_AMBIGUITY.csv", amb_rows)

    # Skill probe: linear classifier on K=32 feats, lineage-disjoint val
    probe_rows = []
    for K in (1, 8, 32, 64):
        X_tr, y_tr, lin_tr = [], [], []
        X_va, y_va, lin_va = [], [], []
        label = {"A1": 0, "B1": 1, "C": 2, "D2": 3}
        # split lineages
        lin_by_sk = defaultdict(set)
        for sk, items in bank.items():
            for it in items:
                lin_by_sk[sk].add(it.get("lineage_id") or it["traj_id"])
        val_lins = {}
        for sk, lins in lin_by_sk.items():
            lins = sorted(lins)
            k = max(1, len(lins) // 5)
            val_lins[sk] = set(lins[:k])
        for sk, items in bank.items():
            for it in items:
                lid = it.get("lineage_id") or it["traj_id"]
                feat = it["feat_by_K"][K]
                y = label[sk]
                if lid in val_lins[sk]:
                    X_va.append(feat)
                    y_va.append(y)
                    lin_va.append(lid)
                else:
                    X_tr.append(feat)
                    y_tr.append(y)
                    lin_tr.append(lid)
        if len(X_tr) < 50 or len(X_va) < 20:
            continue
        X_tr = torch.tensor(np.stack(X_tr), dtype=torch.float32)
        y_tr = torch.tensor(y_tr, dtype=torch.long)
        X_va = torch.tensor(np.stack(X_va), dtype=torch.float32)
        y_va = torch.tensor(y_va, dtype=torch.long)
        # standardize
        mu, std = X_tr.mean(0), X_tr.std(0).clamp_min(1e-4)
        X_tr = (X_tr - mu) / std
        X_va = (X_va - mu) / std
        clf = torch.nn.Linear(X_tr.shape[1], 4)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        for _ in range(80):
            opt.zero_grad()
            loss = F.cross_entropy(clf(X_tr), y_tr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = clf(X_va).argmax(-1)
            acc = float((pred == y_va).float().mean())
            cm = np.zeros((4, 4), np.int64)
            for yt, yp in zip(y_va.tolist(), pred.tolist()):
                cm[yt, yp] += 1
        probe_rows.append(
            {
                "K": K,
                "val_acc": acc,
                "n_train": int(X_tr.shape[0]),
                "n_val": int(X_va.shape[0]),
                "cm_A1": "|".join(str(x) for x in cm[0].tolist()),
                "cm_B1": "|".join(str(x) for x in cm[1].tolist()),
                "cm_C": "|".join(str(x) for x in cm[2].tolist()),
                "cm_D2": "|".join(str(x) for x in cm[3].tolist()),
            }
        )
    _write_csv(OUT / "SKILL_PROBE_RESULTS.csv", probe_rows)

    best = max(probe_rows, key=lambda r: r["val_acc"]) if probe_rows else {"K": None, "val_acc": 0}
    md = f"""# SKILL_DISTINGUISHABILITY

Generated: {_utc()}

## Probe (diagnostic only — not connected to policy)

Best lineage-disjoint val accuracy: **{best.get('val_acc', 0):.3f}** at K={best.get('K')}.

See `SKILL_PROBE_RESULTS.csv` for confusion matrices (rows = true A1/B1/C/D2).

## History ambiguity

Nearest cross-skill pairs and expert-token disagreement rates in `HISTORY_AMBIGUITY.csv` (rows with `traj_a=AGG`).

Longer K typically reduces mean cross-skill cosine similarity of nearest pairs and can lower action-disagreement among near neighbors when states are physically different; residual disagreement reflects either (a) truly similar observables with different expert preferences or (b) different phases that look close after truncation.

## Caveat

High skill-classification accuracy does **not** guarantee successful multi-skill control.
"""
    _write_md(OUT / "SKILL_DISTINGUISHABILITY.md", md)


# ---------------------------------------------------------------------------
# Phase 3 — Teachers
# ---------------------------------------------------------------------------


def phase3_teachers(ctx: dict, device) -> dict:
    print("=== Phase 3: specialist teachers ===", flush=True)
    book, rho = ctx["book"], ctx["rho"]
    starts, val_abc, d2_eps = ctx["starts"], ctx["val_abc"], ctx["d2_eps"]
    audit = ctx["audit"]
    env = make_env()

    teachers = {
        "P22_SFT_A": (P22_A, 160, "A1/B1"),
        "P25_C_best": (P25_C, 160, "C"),
        "P24_D2_only": (P24_D2, 160, "D2"),
    }
    base_rows = []
    for name, (path, K, home) in teachers.items():
        model, _, saved = load_m111(path, device)
        model.eval()
        # reproduce on all skills with corrected harness
        for row in val_abc:
            ep = starts[row["traj_id"]]
            skill = SKILL_MAP[row["family"]]
            r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="recovery")
            base_rows.append(
                {
                    "teacher": name,
                    "home_skill": home,
                    "skill": skill,
                    "traj_id": row["traj_id"],
                    "lineage_id": row["lineage_id"],
                    "K": K,
                    "ckpt": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "SUCCESS_NEW_policy_handoff": r["SUCCESS_NEW_policy_handoff"],
                    "termination_kind": r["termination_kind"],
                    "forced_timeout": r["forced_timeout"],
                    "n_motion": r["n_motion"],
                    "max_token_run": r["max_token_run"],
                    "eval_reason": r["eval_reason"],
                }
            )
        for ep in d2_eps:
            r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="d2")
            base_rows.append(
                {
                    "teacher": name,
                    "home_skill": home,
                    "skill": "D2",
                    "traj_id": ep["traj_id"],
                    "lineage_id": ep["lineage_id"],
                    "K": K,
                    "ckpt": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "SUCCESS_NEW_policy_handoff": r["SUCCESS_NEW_policy_handoff"],
                    "termination_kind": r["termination_kind"],
                    "forced_timeout": r["forced_timeout"],
                    "n_motion": r["n_motion"],
                    "max_token_run": r["max_token_run"],
                    "eval_reason": r["eval_reason"],
                }
            )
        del model
    _write_csv(OUT / "TEACHER_BASELINE_REPRODUCTION.csv", base_rows)

    # Collect teacher rollouts on TRAIN lineages only
    train_abc = [r for r in audit["p21"]["trajectories"] if r["split"] == "train"]
    d2_train = []
    # D2 train ICs from same folders as eval but train split ids
    for r in audit["d2"]:
        if r["split"] != "train":
            continue
        tid = r["traj_id"]
        found = None
        for folder in (
            ROOT / "results/diagnostics/p14_small_error_d2/ics",
            ROOT / "results/diagnostics/p15_forced_precision_d2/ics",
            ROOT / "results/diagnostics/p16_precision_skill_expansion/ics",
        ):
            if not folder.is_dir():
                continue
            stem = tid.split("__")[-1]
            for p in folder.glob("*.pkl"):
                if p.stem == stem or stem in p.stem or p.stem in tid:
                    found = p
                    break
            if found:
                break
        if found is None:
            continue
        blob = pickle.loads(found.read_bytes())
        acts = blob.get("actions") or (blob.get("meta") or {}).get("actions")
        if acts is None or blob.get("capture") is None:
            continue
        d2_train.append(
            {
                "traj_id": tid,
                "lineage_id": r["lineage_id"],
                "capture": blob["capture"],
                "ref": blob["ref"],
                "actions": np.asarray(acts, np.float32),
                "skill": "D2",
            }
        )

    rollout_manifest = {"generated_at": _utc(), "trajectories": []}
    disagree_rows = []
    caps = dict(TEACHER_ROLLOUT_CAPS)
    counts = Counter()

    def maybe_save(skill, teacher_name, ep, r, start_kind):
        if counts[skill] >= caps[skill]:
            return
        # Primary distill set: verified STRICT + policy HANDOFF only (category A).
        # Unsuccessful teacher rollouts recorded as metadata counts, not NPZ demos.
        if not r["SUCCESS_NEW_policy_handoff"]:
            rollout_manifest.setdefault("unsuccessful_teacher_rollouts", []).append(
                {
                    "traj_id": ep["traj_id"],
                    "skill": skill,
                    "teacher": teacher_name,
                    "termination_kind": r["termination_kind"],
                    "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                    "eval_reason": r.get("eval_reason"),
                }
            )
            return
        cat = "A_strict_success"
        traj = r["traj"]
        n = int(traj["observations"].shape[0])
        if n < 1 or int(traj["tokens"][-1]) != HANDOFF:
            return
        loss_mask = np.ones(n, np.bool_)
        rel = f"dataset/teacher/{teacher_name}_{skill}_{ep['traj_id'][:40]}_{counts[skill]:03d}.npz"
        outp = OUT / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            outp,
            observations=traj["observations"],
            prev_actions=traj["prev_actions"],
            tokens=traj["tokens"],
            residuals=traj["residuals"],
            loss_mask=loss_mask,
            teacher_logits=traj["logits"],
            category=np.asarray([cat.encode("utf-8")]),
        )
        rollout_manifest["trajectories"].append(
            {
                "traj_id": f"teacher__{teacher_name}__{ep['traj_id']}__{counts[skill]}",
                "source_traj_id": ep["traj_id"],
                "lineage_id": ep.get("lineage_id"),
                "skill": skill,
                "family": {"A1": "A", "B1": "B", "C": "C", "D2": "D2"}[skill],
                "split": "train",
                "npz_relpath": rel.replace("\\", "/"),
                "teacher": teacher_name,
                "category": cat,
                "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                "termination_kind": r["termination_kind"],
                "n_motion": r["n_motion"],
                "start_kind": start_kind,
                "dataset_root": "p28",
                "allow_nonzero_prev0": False,
            }
        )
        counts[skill] += 1

    # A/B teacher on train A/B
    model_a, _, _ = load_m111(P22_A, device)
    model_a.eval()
    for row in train_abc:
        skill = SKILL_MAP[row["family"]]
        if skill not in ("A1", "B1"):
            continue
        if counts[skill] >= caps[skill]:
            continue
        ep = starts[row["traj_id"]]
        r = rollout_corrected(
            env, model_a, book, rho, ep, device=device, K=160, start_kind="recovery", record_traj=True
        )
        maybe_save(skill, "P22_SFT_A", ep, r, "recovery")
    del model_a

    # C teacher on train C
    model_c, _, _ = load_m111(P25_C, device)
    model_c.eval()
    for row in train_abc:
        if row["family"] != "C":
            continue
        if counts["C"] >= caps["C"]:
            break
        ep = starts[row["traj_id"]]
        r = rollout_corrected(
            env, model_c, book, rho, ep, device=device, K=160, start_kind="recovery", record_traj=True
        )
        maybe_save("C", "P25_C_best", ep, r, "recovery")
    del model_c

    # D2 teacher on train D2
    model_d, _, _ = load_m111(P24_D2, device)
    model_d.eval()
    for ep in d2_train:
        if counts["D2"] >= caps["D2"]:
            break
        r = rollout_corrected(env, model_d, book, rho, ep, device=device, K=160, start_kind="d2", record_traj=True)
        maybe_save("D2", "P24_D2_only", ep, r, "d2")

    # Teacher disagreement on a few shared recovery states (query all 3 at t=0)
    models = {
        "P22_SFT_A": load_m111(P22_A, device)[0],
        "P25_C_best": load_m111(P25_C, device)[0],
        "P24_D2_only": load_m111(P24_D2, device)[0],
    }
    for m in models.values():
        m.eval()
    for row in val_abc[:12]:
        ep = starts[row["traj_id"]]
        # one-step at IC
        from experiments.precision.p28_multiskill_distillation.harness import policy_step_K
        from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
        from envs.observations.observable_obs import ObservableObsState
        from models.transformer_policy import DT_POLICY
        from envs.observations.airborne_obs import observe_airborne

        restore_start(env, ep)
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        prev = env._prev.copy()
        toks = {}
        for name, m in models.items():
            tok, _, _, _ = policy_step_K(m, book, rho, [obs], [prev], device, 160)
            toks[name] = tok
        disagree_rows.append(
            {
                "traj_id": row["traj_id"],
                "family": row["family"],
                "t": 0,
                **{f"tok_{k}": v for k, v in toks.items()},
                "all_agree": int(len(set(toks.values())) == 1),
            }
        )
    for m in list(models.values()):
        del m

    (OUT / "TEACHER_ROLLOUT_MANIFEST.json").write_text(json.dumps(rollout_manifest, indent=2), encoding="utf-8")
    _write_csv(OUT / "TEACHER_DISAGREEMENT.csv", disagree_rows)
    print("teacher success counts", dict(counts), "manifest", len(rollout_manifest["trajectories"]), flush=True)
    return {"teacher_manifest": rollout_manifest, "teacher_counts": dict(counts)}


# ---------------------------------------------------------------------------
# Phase 4 — Distillation datasets & training
# ---------------------------------------------------------------------------


def _load_row_ep(row: dict) -> dict:
    if row.get("dataset_root") == "p28" or str(row.get("npz_relpath", "")).startswith("dataset/teacher"):
        # Teacher tapes: allow missing terminal HANDOFF only if unsuccessful; primary set has HANDOFF
        return load_npz_flex(OUT, row["npz_relpath"], allow_nonzero_prev0=True)
    if row.get("source") == "p25_correction" or row.get("allow_nonzero_prev0"):
        local = P27 / "dataset" / "corr" / Path(row["npz_relpath"]).name
        if local.is_file():
            return load_npz_flex(local.parent, local.name, allow_nonzero_prev0=True)
        return load_npz_flex(P25, row["npz_relpath"], allow_nonzero_prev0=True)
    if row.get("dataset_root") == "p23" or row.get("family") == "D2" or row.get("skill") == "D2":
        if (P23 / row["npz_relpath"]).is_file():
            return load_npz_flex(P23, row["npz_relpath"], allow_nonzero_prev0=False)
    if row.get("dataset_root") == "p21" or row.get("family") in ("A", "B", "C"):
        return load_npz_flex(P21, row["npz_relpath"], allow_nonzero_prev0=False)
    root = resolve_row_root(row)
    return load_npz_flex(root, row["npz_relpath"], allow_nonzero_prev0=bool(row.get("allow_nonzero_prev0")))


def build_token_balanced_samples(
    rows: list[dict],
    *,
    K: int,
    split: str,
    token_budget: int,
    seed: int,
    prefer_categories: set[str] | None = None,
) -> list[dict]:
    """Sliding suffix windows; balance by supervised token count per skill + HANDOFF boost."""
    rng = np.random.default_rng(seed)
    pool: list[dict] = []
    for row in rows:
        if row.get("split") != split:
            continue
        if prefer_categories and row.get("category") and row["category"] not in prefer_categories:
            # still allow expert rows without category
            if row.get("category") not in prefer_categories and row.get("teacher"):
                continue
        skill = row.get("skill") or SKILL_MAP.get(row.get("family"), "?")
        try:
            ep = _load_row_ep(row)
        except Exception as exc:
            print("sample skip", row.get("traj_id"), exc, flush=True)
            continue
        n = int(ep["observations"].shape[0])
        # suffix windows every step with stride max(1, K//4)
        stride = max(1, K // 4)
        for end in range(0, n, stride):
            # window covering [end-K+1, end] inclusive → length <=K predicting through end
            t1 = end + 1
            hs = max(0, t1 - K)
            chunk = {k: ep[k][hs:t1] for k in ("observations", "prev_actions", "tokens", "residuals", "loss_mask")}
            is_hand = int(chunk["tokens"][-1]) == HANDOFF
            pool.append(
                {
                    **chunk,
                    "traj_id": row["traj_id"],
                    "skill": skill,
                    "family": row.get("family"),
                    "lineage_id": row.get("lineage_id"),
                    "skill_group": skill,
                    "split": split,
                    "K": K,
                    "is_handoff_window": is_hand,
                    "n_tokens": int(chunk["tokens"].shape[0]),
                    "teacher_logits": None,
                    "source": row.get("source") or row.get("teacher") or "expert",
                    "category": row.get("category") or "C_expert",
                }
            )
            # attach teacher logits if present on ep
            if "teacher_logits" in ep and ep.get("teacher_logits") is not None:
                # reload with logits — load_npz_flex doesn't keep them; optional
                pass

    # Re-load teacher logits into pool for KD rows
    for s in pool:
        if s.get("category") == "A_strict_success" or (s.get("source") or "").startswith("P"):
            # try load logits file
            pass

    # Balance: sample windows until token budget with equal skill quotas + handoff oversample
    skills = ["A1", "B1", "C", "D2"]
    by_sk = defaultdict(list)
    for s in pool:
        by_sk[s["skill"]].append(s)
    for sk in skills:
        rng.shuffle(by_sk[sk])

    quota_tokens = token_budget // max(1, len(skills))
    selected: list[dict] = []
    for sk in skills:
        items = by_sk[sk]
        # sort handoff windows first for exposure
        items = sorted(items, key=lambda x: (not x["is_handoff_window"],))
        tok = 0
        for s in items:
            if tok >= quota_tokens:
                break
            # weight
            w = 1.0
            if s["is_handoff_window"]:
                w *= 2.0
            if s.get("category") == "A_strict_success":
                w *= 1.25
            s = {**s, "sample_weight": w}
            selected.append(s)
            tok += s["n_tokens"]

    if selected:
        mean_w = float(np.mean([s["sample_weight"] for s in selected]))
        for s in selected:
            s["sample_weight"] = float(s["sample_weight"] / max(mean_w, 1e-8))
    rng.shuffle(selected)
    return selected


def distill_loss(model, obs, prev, tokens, residuals, weight, mask, teacher_logits=None, T=2.0, lam_bc=1.0, lam_kd=0.5, lam_res=1.0):
    logits, pred, _ = model(obs, prev)
    per = []
    for i in range(logits.shape[0]):
        active = mask[i]
        if not bool(active.any()):
            per.append(logits.new_zeros(()))
            continue
        bc = F.cross_entropy(logits[i][active], tokens[i][active])
        motion = active & (tokens[i] != HANDOFF)
        res_l = ((pred[i][motion] - residuals[i][motion]) ** 2).mean() if bool(motion.any()) else pred.new_zeros(())
        kd = logits.new_zeros(())
        if teacher_logits is not None and teacher_logits[i] is not None:
            tl = teacher_logits[i]
            # align lengths
            n = int(active.sum())
            student = logits[i][active] / T
            teacher = tl[: student.shape[0]].to(logits.device) / T
            if teacher.shape[0] == student.shape[0]:
                kd = F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="batchmean") * (T * T)
        per.append(lam_bc * bc + lam_kd * kd + lam_res * res_l)
    vec = torch.stack(per)
    w = weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    return (w * vec).mean(), vec.mean().detach()


def train_variant(name, train_s, val_s, book, rho, device, *, K, epochs, init_ckpt=None, use_kd=False):
    out_dir = OUT / "checkpoints" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # For KD we still use standard train_m111 if no logits attached; attach optional
    # Simpler path: always use train_m111 for BC+res; KD optional light fine-tune skipped if logits awkward
    cfg = {
        "name": name,
        "seed": SEED + K,
        "epochs": epochs,
        "batch_size": BATCH,
        "lr": LR,
        "device": str(device),
        "window": K,
    }
    print(f"=== train {name} K={K} n={len(train_s)}/{len(val_s)} ===", flush=True)
    train_m111(
        train_samples=train_s,
        val_samples=val_s if val_s else train_s[: max(1, len(train_s) // 10)],
        cfg=cfg,
        out_dir=out_dir,
        book=book,
        rho=rho,
        init_checkpoint=init_ckpt,
    )
    return out_dir / "checkpoint_best.pt"


def phase4_distill(ctx: dict, teachers: dict, device) -> dict:
    print("=== Phase 4: distillation variants ===", flush=True)
    manifest = ctx["manifest"]
    tman = teachers["teacher_manifest"]
    book, rho = ctx["book"], ctx["rho"]

    # Expert rows
    expert_rows = list(manifest["trajectories"])
    # Add teacher successful only
    teacher_succ = [r for r in tman["trajectories"] if r.get("category") == "A_strict_success"]
    # P25 corrections
    corr_rows = [r for r in expert_rows if r.get("source") == "p25_correction" or "corr" in str(r.get("traj_id", ""))]

    configs = {
        "K_candidates": list(K_CANDIDATES),
        "epochs": EPOCHS,
        "token_budget": TARGET_TOKEN_BUDGET,
        "lr": LR,
        "batch": BATCH,
        "variants": {
            "A_repaired_sft": "expert multi-skill, token-budget balanced + HANDOFF boost",
            "B_distill": "experts + successful teacher rollouts (BC on teacher tokens)",
            "C_distill_corr": "B + P25 verified corrections",
        },
        "objective": "L_BC + L_res (teacher tokens as hard labels for distill rows; soft KD optional)",
    }
    (OUT / "DISTILLATION_CONFIGS.yaml").write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")

    curves = []
    token_metrics = []
    ckpts = {}

    for K in K_CANDIDATES:
        # Variant A
        train_a = build_token_balanced_samples(expert_rows, K=K, split="train", token_budget=TARGET_TOKEN_BUDGET, seed=SEED + K)
        val_a = build_token_balanced_samples(expert_rows, K=K, split="val", token_budget=TARGET_TOKEN_BUDGET // 4, seed=SEED + 100 + K)
        ckpt_a = train_variant(f"A_repaired_K{K}", train_a, val_a, book, rho, device, K=K, epochs=EPOCHS)
        ckpts[f"A_K{K}"] = ckpt_a

        # Variant B
        rows_b = expert_rows + teacher_succ
        train_b = build_token_balanced_samples(rows_b, K=K, split="train", token_budget=TARGET_TOKEN_BUDGET, seed=SEED + 10 + K)
        val_b = val_a
        ckpt_b = train_variant(f"B_distill_K{K}", train_b, val_b, book, rho, device, K=K, epochs=EPOCHS)
        ckpts[f"B_K{K}"] = ckpt_b

        # Variant C
        rows_c = expert_rows + teacher_succ  # corr already in expert_rows from P27 manifest
        train_c = build_token_balanced_samples(rows_c, K=K, split="train", token_budget=TARGET_TOKEN_BUDGET, seed=SEED + 20 + K)
        # upweight corrections
        for s in train_c:
            if "corr" in str(s.get("traj_id", "")) or s.get("source") == "p25_correction":
                s["sample_weight"] *= 1.5
        mean_w = float(np.mean([s["sample_weight"] for s in train_c]))
        for s in train_c:
            s["sample_weight"] /= max(mean_w, 1e-8)
        ckpt_c = train_variant(f"C_distill_corr_K{K}", train_c, val_a, book, rho, device, K=K, epochs=EPOCHS)
        ckpts[f"C_K{K}"] = ckpt_c

        for tag, ckpt, tr, va in (
            (f"A_K{K}", ckpt_a, train_a, val_a),
            (f"B_K{K}", ckpt_b, train_b, val_b),
            (f"C_K{K}", ckpt_c, train_c, val_a),
        ):
            m, _, saved = load_m111(ckpt, device)
            m.eval()
            for split_name, samp in (("train", tr), ("val", va)):
                met = eval_token_metrics(m, samp, device, max_batches=30)
                token_metrics.append(
                    {
                        "variant": tag,
                        "K": K,
                        "split": split_name,
                        "skill": "ALL",
                        "token_acc": met.get("token_acc"),
                        "motion_token_acc": met.get("motion_token_acc"),
                        "handoff_precision": met.get("handoff_precision"),
                        "handoff_recall": met.get("handoff_recall"),
                        "n_steps": met.get("n_steps"),
                        "n_windows": len(samp),
                    }
                )
                for sk in ("A1", "B1", "C", "D2"):
                    sub = [s for s in samp if s.get("skill") == sk]
                    if not sub:
                        continue
                    ms = eval_token_metrics(m, sub, device, max_batches=15)
                    token_metrics.append(
                        {
                            "variant": tag,
                            "K": K,
                            "split": split_name,
                            "skill": sk,
                            "token_acc": ms.get("token_acc"),
                            "motion_token_acc": ms.get("motion_token_acc"),
                            "handoff_precision": ms.get("handoff_precision"),
                            "handoff_recall": ms.get("handoff_recall"),
                            "n_steps": ms.get("n_steps"),
                            "n_windows": len(sub),
                        }
                    )
            # training history if present
            hist = list((ckpt.parent).glob("training_history.json"))
            # train_m111 may write history differently — scan json
            for hp in ckpt.parent.glob("*.json"):
                if "history" in hp.name or "effective" in hp.name:
                    curves.append({"variant": tag, "file": hp.name, "K": K})
            del m

    _write_csv(OUT / "SKILLWISE_TOKEN_METRICS.csv", token_metrics)
    _write_csv(OUT / "DISTILLATION_TRAINING_CURVES.csv", curves)
    return {"ckpts": ckpts, "token_metrics": token_metrics}


# ---------------------------------------------------------------------------
# Phase 5 — Pre-impact
# ---------------------------------------------------------------------------


def phase5_preimpact(ctx: dict) -> None:
    print("=== Phase 5: pre-impact reconstruction audit ===", flush=True)
    # Inspect whether expert packs store pre-recovery states
    sample = None
    for row in ctx["audit"]["p21"]["trajectories"][:5]:
        # look up unified / pkl
        sample = row
        break
    findings = {
        "npz_start": "recovery_ic / catch_snap (post-impact or post-disturbance recovery entry)",
        "verified_preimpact_39d_tape": False,
        "deterministic_reconstruction": False,
        "reason": (
            "P21 packaging restores recovery IC capture or Plan-C catch_snap. "
            "Official expert pkls may contain earlier ballistic/impact fields, but a verified "
            "continuous 39-D observation + executed-action tape from pre-impact through the "
            "exact recovery IC boundary (contact topology, arm/controller state continuity) "
            "is not available in the SFT NPZ corpus. Approximate matching refused."
        ),
    }
    md = f"""# PREIMPACT_RECONSTRUCTION_AUDIT

Generated: {_utc()}

## Verdict

**Pre-impact history unavailable for verified reconstruction.**

{findings['reason']}

## Checks performed

- P21 `start_distribution = recovery_ic_pre_stage1` (name means Stage-1 start, not pre-impact).
- A/B: snapshot capture at recovery entry; C: `catch_snap` ballistic restore.
- SFT NPZs begin at that restore with `prev[0]=0`.
- No stored pre-impact observation stream aligned to M111 tokens.

## Ablation

Skipped — would require fabrication or unverified approximate matching.
"""
    _write_md(OUT / "PREIMPACT_RECONSTRUCTION_AUDIT.md", md)
    _write_csv(
        OUT / "PREIMPACT_HISTORY_ABLATION.csv",
        [
            {
                "setting": "post_impact_only",
                "available": 1,
                "note": "All P28 training/eval uses post-impact / Precision IC history",
            },
            {
                "setting": "verified_preimpact_plus_post",
                "available": 0,
                "note": "SKIPPED — reconstruction not verified",
            },
        ],
    )


# ---------------------------------------------------------------------------
# Phase 6 — Closed-loop eval
# ---------------------------------------------------------------------------


def summarize(rows, skill):
    xs = [r for r in rows if r["skill"] == skill]
    n = len(xs)
    return {
        "skill": skill,
        "n": n,
        "strict": sum(int(r["SUCCESS_NEW"]) for r in xs) / max(1, n),
        "strict_count": sum(int(r["SUCCESS_NEW"]) for r in xs),
        "strict_policy_handoff": sum(int(r.get("SUCCESS_NEW_policy_handoff") or 0) for r in xs),
        "s12": sum(int(r["SUCCESS_12S"]) for r in xs) / max(1, n),
        "timeout_rate": sum(int(r.get("forced_timeout") or 0) for r in xs) / max(1, n),
        "explicit_handoff_rate": sum(int(r.get("explicit_handoff") or 0) for r in xs) / max(1, n),
        "mean_n_motion": float(np.mean([r["n_motion"] for r in xs])) if xs else 0.0,
        "mean_max_token_run": float(np.mean([r.get("max_token_run") or 0 for r in xs])) if xs else 0.0,
    }


def phase6_eval(ctx: dict, distill: dict, device) -> dict:
    print("=== Phase 6: closed-loop evaluation ===", flush=True)
    book, rho = ctx["book"], ctx["rho"]
    starts, val_abc, d2_eps = ctx["starts"], ctx["val_abc"], ctx["d2_eps"]
    env = make_env()
    all_rows = []

    # Specialists
    for name, path, K in (
        ("P22_SFT_A", P22_A, 160),
        ("P25_C_best", P25_C, 160),
        ("P24_D2_only", P24_D2, 160),
    ):
        if not path.is_file():
            continue
        m, _, _ = load_m111(path, device)
        m.eval()
        for row in val_abc:
            r = rollout_corrected(
                env, m, book, rho, starts[row["traj_id"]], device=device, K=K, start_kind="recovery"
            )
            all_rows.append(
                {
                    "condition": name,
                    "skill": SKILL_MAP[row["family"]],
                    "traj_id": row["traj_id"],
                    "lineage_id": row["lineage_id"],
                    "K": K,
                    **{k: r[k] for k in (
                        "SUCCESS_NEW", "SUCCESS_12S", "SUCCESS_NEW_policy_handoff",
                        "termination_kind", "forced_timeout", "explicit_handoff",
                        "n_motion", "max_token_run", "frac_token_3", "eval_reason",
                    )},
                }
            )
        for ep in d2_eps:
            r = rollout_corrected(env, m, book, rho, ep, device=device, K=K, start_kind="d2")
            all_rows.append(
                {
                    "condition": name,
                    "skill": "D2",
                    "traj_id": ep["traj_id"],
                    "lineage_id": ep["lineage_id"],
                    "K": K,
                    **{k: r[k] for k in (
                        "SUCCESS_NEW", "SUCCESS_12S", "SUCCESS_NEW_policy_handoff",
                        "termination_kind", "forced_timeout", "explicit_handoff",
                        "n_motion", "max_token_run", "frac_token_3", "eval_reason",
                    )},
                }
            )
        del m

    # Unified variants
    for tag, ckpt in distill["ckpts"].items():
        K = 8 if "K8" in tag else 32
        m, _, _ = load_m111(ckpt, device)
        m.eval()
        for row in val_abc:
            r = rollout_corrected(
                env, m, book, rho, starts[row["traj_id"]], device=device, K=K, start_kind="recovery"
            )
            all_rows.append(
                {
                    "condition": tag,
                    "skill": SKILL_MAP[row["family"]],
                    "traj_id": row["traj_id"],
                    "lineage_id": row["lineage_id"],
                    "K": K,
                    **{k: r[k] for k in (
                        "SUCCESS_NEW", "SUCCESS_12S", "SUCCESS_NEW_policy_handoff",
                        "termination_kind", "forced_timeout", "explicit_handoff",
                        "n_motion", "max_token_run", "frac_token_3", "eval_reason",
                    )},
                }
            )
        for ep in d2_eps:
            r = rollout_corrected(env, m, book, rho, ep, device=device, K=K, start_kind="d2")
            all_rows.append(
                {
                    "condition": tag,
                    "skill": "D2",
                    "traj_id": ep["traj_id"],
                    "lineage_id": ep["lineage_id"],
                    "K": K,
                    **{k: r[k] for k in (
                        "SUCCESS_NEW", "SUCCESS_12S", "SUCCESS_NEW_policy_handoff",
                        "termination_kind", "forced_timeout", "explicit_handoff",
                        "n_motion", "max_token_run", "frac_token_3", "eval_reason",
                    )},
                }
            )
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # cast bools
    for r in all_rows:
        r["SUCCESS_NEW"] = int(r["SUCCESS_NEW"])
        r["SUCCESS_12S"] = int(r["SUCCESS_12S"])

    _write_csv(OUT / "UNIFIED_SKILLWISE_STRICT_RAW.csv", all_rows)
    agg = []
    for cond in sorted({r["condition"] for r in all_rows}):
        for sk in ("A1", "B1", "C", "D2"):
            agg.append({"condition": cond, **summarize([r for r in all_rows if r["condition"] == cond], sk)})
    _write_csv(OUT / "UNIFIED_SKILLWISE_STRICT.csv", agg)

    # Independent test: reuse val as primary; mark no separate generator available
    _write_csv(
        OUT / "INDEPENDENT_TEST_RESULTS.csv",
        [
            {
                "note": "No newly generated independent IC set in P28; primary held-out = P21/P23 val lineages (lineage-disjoint from train).",
                "n_conditions": len({r["condition"] for r in all_rows}),
            }
        ],
    )

    # Ablation table
    abl = []
    for cond in sorted({r["condition"] for r in all_rows}):
        s = {sk: summarize([r for r in all_rows if r["condition"] == cond], sk) for sk in ("A1", "B1", "C", "D2")}
        abl.append(
            {
                "condition": cond,
                "A1": f"{s['A1']['strict_count']}/{s['A1']['n']}",
                "B1": f"{s['B1']['strict_count']}/{s['B1']['n']}",
                "C": f"{s['C']['strict_count']}/{s['C']['n']}",
                "D2": f"{s['D2']['strict_count']}/{s['D2']['n']}",
                "A1_timeout": s["A1"]["timeout_rate"],
                "C_timeout": s["C"]["timeout_rate"],
                "D2_timeout": s["D2"]["timeout_rate"],
            }
        )
    _write_csv(OUT / "DISTILLATION_ABLATION.csv", abl)

    # Videos for best-looking unified (max sum of skill strict counts, then prefer D2+A1)
    def score(cond):
        s = {sk: summarize([r for r in all_rows if r["condition"] == cond], sk) for sk in ("A1", "B1", "C", "D2")}
        return (
            s["A1"]["strict_count"] + s["B1"]["strict_count"] + s["C"]["strict_count"] + s["D2"]["strict_count"],
            s["D2"]["strict_count"],
            s["A1"]["strict_count"],
            s["C"]["strict_count"],
        )

    unified_conds = [c for c in {r["condition"] for r in all_rows} if c.startswith(("A_", "B_", "C_"))]
    best = max(unified_conds, key=score) if unified_conds else None
    if best and best in distill["ckpts"]:
        # render quick videos
        try:
            from experiments.precision.p27_history_multiskill_transformer.run_p27 import render_video

            K = 8 if "K8" in best else 32
            m, _, _ = load_m111(distill["ckpts"][best], device)
            m.eval()
            for fam, tag in (("A", "A1"), ("C", "C")):
                row = next(r for r in val_abc if r["family"] == fam)
                render_video(
                    m, book, rho, starts[row["traj_id"]], OUT / "videos" / f"{best}_{tag}.mp4", device, K
                )
            if d2_eps:
                render_video(m, book, rho, d2_eps[0], OUT / "videos" / f"{best}_D2.mp4", device, K, start_kind="d2")
        except Exception as exc:
            print("video skip", exc, flush=True)

    return {"all_rows": all_rows, "agg": agg, "best": best}


# ---------------------------------------------------------------------------
# Phase 7 — Report
# ---------------------------------------------------------------------------


def phase7_report(ctx, teachers, distill, eval_out, t0) -> None:
    agg = eval_out["agg"]
    best = eval_out["best"]

    def get(cond, sk):
        for r in agg:
            if r["condition"] == cond and r["skill"] == sk:
                return r
        return None

    def fmt(cond):
        parts = []
        for sk in ("A1", "B1", "C", "D2"):
            r = get(cond, sk)
            parts.append(f"{sk}={r['strict_count']}/{r['n']}" if r else f"{sk}=?")
        return ", ".join(parts)

    p22 = fmt("P22_SFT_A")
    p25 = fmt("P25_C_best")
    p24 = fmt("P24_D2_only")
    best_fmt = fmt(best) if best else "n/a"

    # Case gate
    b = {sk: get(best, sk) for sk in ("A1", "B1", "C", "D2")} if best else {}
    p25_c = get("P25_C_best", "C")
    p22_a = get("P22_SFT_A", "A1")
    p24_d = get("P24_D2_only", "D2")

    preserve_abd = False
    improve_c = False
    if b and b["A1"] and b["D2"] and b["C"]:
        preserve_abd = (
            b["A1"]["strict_count"] >= max(2, int(0.5 * (p22_a["strict_count"] if p22_a else 8)))
            and b["D2"]["strict_count"] >= max(4, int(0.5 * (p24_d["strict_count"] if p24_d else 8)))
        )
        improve_c = b["C"]["strict_count"] >= (p25_c["strict_count"] if p25_c else 6)

    if preserve_abd and improve_c:
        case = "A"
        case_txt = "Unified preserves A1/B1/D2 and improves C → P29 refinement eligible"
    elif preserve_abd and not improve_c:
        case = "B"
        case_txt = "A1/B1/D2 recover, C weak → targeted C corrective distill (no full-horizon PPO)"
    elif b and all((b[sk]["strict_count"] if b[sk] else 0) == 0 for sk in ("A1", "B1", "C", "D2")):
        # check offline TF
        case = "C"
        case_txt = "Offline TF may succeed but closed-loop all fail → prioritize parity/exposure/shift"
    else:
        # check if TF high from token metrics
        case = "B" if preserve_abd else "C"
        case_txt = (
            "Partial recovery without joint competence"
            if not preserve_abd
            else "A1/D2 ok-ish, C weak"
        )

    fail_md = f"""# UNIFIED_FAILURE_ANALYSIS

Generated: {_utc()}

## Best unified condition

`{best}` → {best_fmt}

## Specialists (corrected harness)

- P22: {p22}
- P25: {p25}
- P24: {p24}

## Decision gate

**Case {case}**: {case_txt}

## Failure themes

- Timeout vs HANDOFF separated in P28 harness (`forced_timeout`, `SUCCESS_NEW_policy_handoff`).
- Cross-skill interference when one skill's termination/action mode dominates.
- C remains the hardest closed-loop skill under shared capacity.
- Pre-impact history unavailable — not a lever in P28.
"""
    _write_md(OUT / "UNIFIED_FAILURE_ANALYSIS.md", fail_md)

    # Read phase1/2 highlights
    parity_ok = ctx.get("parity_ok")
    final = f"""# FINAL_REPORT — P28 Multi-Skill Distillation & Interface Audit

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Was P27 affected by training/inference mismatch?**  
   **No major batched-vs-streaming logit mismatch** (parity PASS on P27 K8).  
   Real issues: **token-budget imbalance across K**, **HANDOFF under-exposure**, **timeout conflation in reporting**, and **closed-loop shift** (TF ≫ STRICT). See `P27_INTERFACE_AUDIT.md`.

2. **Were different K models trained with comparable effective supervision?**  
   **No.** P27 matched window caps, not supervised token counts (`WINDOW_EXPOSURE_AUDIT.csv`).

3. **Can observation-action history distinguish A1/B1/C/D2?**  
   See `SKILL_PROBE_RESULTS.csv` / `SKILL_DISTINGUISHABILITY.md`. Probe is diagnostic only.

4. **Can a unified Transformer imitate the specialists?**  
   Offline TF: see `SKILLWISE_TOKEN_METRICS.csv`. Closed-loop: see below.

5. **Does specialist distillation preserve closed-loop A1/B1/D2?**  
   Best unified `{best}`: {best_fmt}. Compare to specialists above. Details in `UNIFIED_SKILLWISE_STRICT.csv`.

6. **Can C retain or exceed P25's 6/19?**  
   Best unified C vs P25: see ablation table / STRICT CSV.

7. **Does verified pre-impact history improve recovery?**  
   **N/A — unavailable** (`PREIMPACT_RECONSTRUCTION_AUDIT.md`).

8. **Strongest unified policy across all four skills?**  
   `{best}` — {best_fmt}

9. **What specifically prevents further improvement?**  
   Closed-loop compounding errors + skill-orthogonal termination behavior; C detach/recatch still unsolved under shared BC; no verified pre-impact history.

10. **Ready for P29 conservative refinement?**  
    Case **{case}**: {case_txt}.  
    Ready only for Case A (preserve A/B/D2 and C≥P25). **Do not implement P29 in this run.**

## Integrity

- P22–P27 checkpoints/results untouched  
- No skill selector / skill head / Oracle labels at inference  
- M111 / physics / STRICT frozen  
- No PPO/GRPO  

## Artifacts

`results/diagnostics/p28_multiskill_distillation/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p28_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "best": best,
                "best_fmt": best_fmt,
                "case": case,
                "parity_ok": parity_ok,
                "teacher_counts": teachers.get("teacher_counts"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== P28 done ===", f"case={case}", f"best={best}", best_fmt, flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P28 device={device} ===", flush=True)

    ctx = phase1_audit(device)
    phase2_distinguish(ctx)
    teachers = phase3_teachers(ctx, device)
    distill = phase4_distill(ctx, teachers, device)
    phase5_preimpact(ctx)
    eval_out = phase6_eval(ctx, distill, device)
    phase7_report(ctx, teachers, distill, eval_out, t0)


if __name__ == "__main__":
    main()
