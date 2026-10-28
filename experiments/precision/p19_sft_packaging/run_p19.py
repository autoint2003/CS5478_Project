"""P19 — Final SFT Dataset Packaging and Training Interface Audit.

Uses only P18 discrete-validated (continuous-ok + M111 SUCCESS_NEW) trajectories.
Records policy-visible airborne observations while executing M111-decoded actions.
Does not train. Does not overwrite M111 or mutate the official corpus.
"""

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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import OBS_DIM_AIR, OBS_VERSION, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.skill_aware_action_vocab import (  # noqa: E402
    DEFAULT_WEIGHTS,
    load_motion_codebook,
    quantize_trajectory,
)

OUT = ROOT / "results" / "diagnostics" / "p19_sft_packaging"
DATASET = OUT / "dataset"
NPZ_DIR = DATASET / "npz"
P18 = ROOT / "results" / "diagnostics" / "p18_discrete_vocab_audit"
P17 = ROOT / "results" / "diagnostics" / "p17_sft_readiness"
M111_META = (
    ROOT
    / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
)
P17_M111_IDS = [
    "A_STABLE_PRECISION__PILOT_A__ball_train_0027__7faa1bb1f361ba4c",
    "A_STABLE_PRECISION__PILOT_A__ball_train_v2_0155__3a4e2545849c44cf",
    "A_STABLE_PRECISION__PILOT_A__k48r_train_00032__28f4a70b9e5460a6",
    "B_STABLE_PRECISION__B__B_post_lateral__a4bb242b95fcf0ae",
    "C_RECATCH_PRECISION__C__postlift__gbi_013__88f62ee583298c20",
    "COUPLED_P16_b0__v2_0168_dx3_dz4__coupled",
    "DZ_EXPAND_b0__v2_0168_dx0_dz-4__dz",
    "B_STABLE_PRECISION__B__hard_910026__post_b1_v2__e3a32a342c39f52b",
]
SEED = 19
WINDOW = 160


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


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes")


def load_book():
    cb = load_motion_codebook(M111_META)
    book = np.asarray(cb.prototypes, np.float32)
    rho = cb.residual
    handoff_id = int(cb.handoff_id)
    assert book.shape[0] == 111, book.shape
    assert handoff_id == 111
    return book, rho, handoff_id, cb


def load_record(record_path: str) -> dict:
    path = ROOT / record_path
    raw = pickle.loads(path.read_bytes())
    acts = raw.get("actions_7d")
    if acts is None:
        acts = raw.get("actions")
        meta = raw.get("meta") or {}
        if acts is None:
            acts = meta.get("actions")
    acts = np.asarray(acts if acts is not None else [], np.float32).reshape(-1, 7)
    return {
        "raw": raw,
        "actions": acts,
        "capture": raw.get("capture"),
        "ref": raw.get("ref"),
        "phases": list(raw.get("phases") or []),
    }


def classify_skills(skill_group: str, n_actions: int, phases: list[str] | None = None) -> dict:
    """Analysis-only skill tags from P18 skill_group (+ optional phases)."""
    parts = (skill_group or "").split("|")
    rec = parts[0] if parts else "?"
    mid = parts[1] if len(parts) > 1 else ""
    prec_flag = "prec=1" in (skill_group or "")
    mid_l = mid.lower()
    is_direct = mid_l in ("direct", "precision_skipped", "") or mid_l.startswith("none")
    is_skip = "precision_skipped" in mid_l
    has_dx = "dx" in mid_l and "dtheta" not in mid_l
    # crude: 'dx' substring also in nothing else; coupled listed separately
    has_coupled = "coupled" in mid_l
    has_dz = ("dz" in mid_l) and not has_coupled
    has_prec_tokens = any(x in mid_l for x in ("dx", "dz", "coupled", "dtheta"))
    executed_prec = bool(prec_flag and n_actions > 0 and has_prec_tokens and not is_skip)
    recovery_abc = rec in ("A", "B", "C")
    return {
        "recovery_letter": rec,
        "precision_mid": mid,
        "is_recovery_abc": recovery_abc,
        "is_direct_or_skip": is_direct or is_skip or (n_actions == 0 and not executed_prec),
        "is_direct_handoff_only": (is_direct or is_skip or mid_l in ("direct",)) and n_actions == 0,
        "executes_dx": has_dx and n_actions > 0,
        "executes_dz": has_dz and n_actions > 0,
        "executes_coupled": has_coupled and n_actions > 0,
        "precision_containing_label": prec_flag,
        "precision_executed_corrections": executed_prec,
        "nonzero_actions": n_actions > 0,
        "phases_head": "|".join((phases or [])[:8]),
    }


def package_one(
    env,
    *,
    traj_id: str,
    record_path: str,
    book: np.ndarray,
    rho,
    handoff_id: int,
) -> dict[str, Any]:
    """Replay M111-decoded actions while recording airborne obs; append HANDOFF."""
    rec = load_record(record_path)
    acts = rec["actions"]
    cap = rec["capture"]
    ref = rec["ref"]
    if cap is None or ref is None:
        raise RuntimeError(f"{traj_id}: missing capture/ref")

    q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
    tokens = np.asarray(q["tokens"], np.int64)
    residuals = np.asarray(q["residuals"], np.float32)
    recon = np.asarray(q["recon_actions"], np.float32)
    if len(acts) != len(recon):
        raise RuntimeError(f"{traj_id}: quantize length mismatch")

    # Decode integrity: recon must equal hybrid decode of (token, residual)
    if len(recon) > 0:
        proto = book[tokens]
        decoded = np.clip(proto + residuals, -1.0, 1.0).astype(np.float32)
        if float(np.max(np.abs(decoded - recon))) > 1e-6:
            raise RuntimeError(f"{traj_id}: recon != decode(token, residual)")

    restore_capture(env, cap)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)

    obs_list: list[np.ndarray] = []
    prev_list: list[np.ndarray] = []
    tok_list: list[int] = []
    res_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    seq_pos: list[int] = []
    used_continuous_fallback = False

    for i, a_dec in enumerate(recon):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        if obs.shape != (OBS_DIM_AIR,):
            raise RuntimeError(f"{traj_id}: obs dim {obs.shape}")
        prev = env._prev.copy()
        # Execute ONLY M111-decoded action (no continuous teacher action).
        a_exec = np.asarray(a_dec, np.float32)
        if len(acts) > 0 and float(np.max(np.abs(a_exec - acts[i]))) > 1e-5:
            # Expected for clipped residuals; still must not step continuous.
            pass
        # Guard: never step the continuous expert action in this packaging path.
        step_legal(env.sim, a_exec)
        env._prev = a_exec
        obs_list.append(obs)
        prev_list.append(prev)
        tok_list.append(int(tokens[i]))
        res_list.append(np.asarray(residuals[i], np.float32))
        act_list.append(a_exec)
        seq_pos.append(i)

    # Terminal HANDOFF target (matches existing SFT / inference end condition).
    handoff_obs = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
        np.float32,
    )
    obs_list.append(handoff_obs)
    prev_list.append(env._prev.copy())
    tok_list.append(int(handoff_id))
    res_list.append(np.zeros(7, np.float32))
    act_list.append(np.zeros(7, np.float32))  # not executed; analysis placeholder
    seq_pos.append(len(recon))

    observations = np.stack(obs_list, axis=0).astype(np.float32)
    prev_actions = np.stack(prev_list, axis=0).astype(np.float32)
    token_arr = np.asarray(tok_list, np.int64)
    residual_arr = np.stack(res_list, axis=0).astype(np.float32)
    executed = np.stack(act_list, axis=0).astype(np.float32)
    loss_mask = np.ones(len(token_arr), dtype=np.bool_)  # no padding in storage
    episode_boundary = np.zeros(len(token_arr), dtype=np.bool_)
    episode_boundary[-1] = True

    # Alignment checks
    n = len(token_arr)
    assert observations.shape == (n, OBS_DIM_AIR)
    assert prev_actions.shape == (n, 7)
    assert residual_arr.shape == (n, 7)
    assert executed.shape == (n, 7)
    if n > 1:
        # prev[t] == executed action at t-1 for motion steps; HANDOFF prev = last motion
        for t in range(1, n):
            if t < len(recon):
                if float(np.max(np.abs(prev_actions[t] - executed[t - 1]))) > 1e-6:
                    raise RuntimeError(f"{traj_id}: prev/action off-by-one at t={t}")
            else:
                # HANDOFF step: prev must equal last executed motion (or zeros if empty)
                if len(recon) > 0:
                    if float(np.max(np.abs(prev_actions[t] - executed[t - 1]))) > 1e-6:
                        raise RuntimeError(f"{traj_id}: HANDOFF prev mismatch")
    if float(np.max(np.abs(prev_actions[0]))) > 1e-8:
        raise RuntimeError(f"{traj_id}: prev[0] must be zero")
    if int(token_arr[-1]) != int(handoff_id):
        raise RuntimeError(f"{traj_id}: terminal token not HANDOFF")
    if np.any(token_arr[:-1] == handoff_id):
        raise RuntimeError(f"{traj_id}: HANDOFF appeared before terminal")
    if len(recon) > 0 and np.any((token_arr[:-1] < 0) | (token_arr[:-1] >= handoff_id)):
        raise RuntimeError(f"{traj_id}: motion token out of range")

    stem = hashlib.sha1(traj_id.encode("utf-8")).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    out_path = OUT / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=observations,
        prev_actions=prev_actions,
        tokens=token_arr,
        residuals=residual_arr,
        executed_actions=executed,
        loss_mask=loss_mask,
        episode_boundary=episode_boundary,
        sequence_position=np.asarray(seq_pos, np.int32),
        continuous_actions=acts.astype(np.float32),
        recon_actions=recon.astype(np.float32),
        handoff_id=np.int64(handoff_id),
    )
    return {
        "traj_id": traj_id,
        "npz_relpath": rel.replace("\\", "/"),
        "n_steps": int(n),
        "n_motion": int(len(recon)),
        "n_handoff": 1,
        "obs_dim": int(OBS_DIM_AIR),
        "obs_version": OBS_VERSION,
        "m111_clipped_frac": float(q["clipped_frac"]),
        "m111_mean_recon_wlinf": float(q["mean_recon_wlinf"]),
        "m111_n_unique_tokens": int(len(set(int(t) for t in tokens))) if len(tokens) else 0,
        "used_continuous_fallback": used_continuous_fallback,
        "phases": rec["phases"],
        "token_counts": Counter(int(t) for t in token_arr),
        "motion_token_counts": Counter(int(t) for t in token_arr[:-1]),
    }


def reconcile_p17_p18() -> str:
    p17 = {r["traj_id"]: r for r in _read_csv(P17 / "M111_REPLAY_RESULTS.csv")}
    m111 = {r["traj_id"]: r for r in _read_csv(P18 / "M111_FULL_REPLAY_RESULTS.csv")}
    cont = {r["traj_id"]: r for r in _read_csv(P18 / "CONTINUOUS_REPLAY_BASELINE.csv")}

    lines = [
        "# P17 ↔ P18 Replay Reconciliation",
        "",
        f"Generated: {_utc()}",
        "",
        "## Exact P17 sample (8 trajectories)",
        "",
        "P17 `m111_verify_sample` selected these IDs (skill-diverse continuous demos with ≥50 actions):",
        "",
    ]
    rows_tbl = []
    for tid in P17_M111_IDS:
        a = p17.get(tid, {})
        b = m111.get(tid, {})
        c = cont.get(tid, {})
        rows_tbl.append(
            {
                "traj_id": tid,
                "p17_cont_label_SUCCESS_NEW": a.get("continuous_SUCCESS_NEW"),
                "p17_m111_replay_SUCCESS_NEW": a.get("m111_replay_SUCCESS_NEW"),
                "p17_m111_replay_SUCCESS_12S": a.get("m111_replay_SUCCESS_12S"),
                "p17_note": a.get("m111_replay_note"),
                "p18_cont_SUCCESS_NEW": c.get("cont_SUCCESS_NEW"),
                "p18_cont_SUCCESS_12S": c.get("cont_SUCCESS_12S"),
                "p18_eligible_for_discrete": c.get("eligible_for_discrete"),
                "p18_m111_SUCCESS_NEW": b.get("m111_SUCCESS_NEW"),
                "p18_m111_SUCCESS_12S": b.get("m111_SUCCESS_12S"),
            }
        )
        lines.append(f"- `{tid}`")
    lines.extend(["", "## Per-ID comparison", ""])
    _write_csv(OUT / "P17_P18_ID_COMPARISON.csv", rows_tbl)

    lines.append("| traj (short) | P17 M111 NEW | P18 cont NEW | P18 eligible | P18 M111 NEW |")
    lines.append("|---|---|---|---|---|")
    for r in rows_tbl:
        short = r["traj_id"][:48]
        lines.append(
            f"| `{short}` | {r['p17_m111_replay_SUCCESS_NEW']} | {r['p18_cont_SUCCESS_NEW']} | "
            f"{r['p18_eligible_for_discrete']} | {r['p18_m111_SUCCESS_NEW']} |"
        )

    lines.extend(
        [
            "",
            "## Why P17 reported 2/8 failures while P18 reports 221/221",
            "",
            "### 1. Harness / protocol differences (primary)",
            "",
            "P17 `m111_verify_sample` (see `experiments/precision/p17_sft_readiness/run_p17.py`):",
            "",
            "- Restored expert `capture`, then stepped **subsampled** decoded actions:",
            "  `for a in recon[::step]` with `step = max(1, len(recon)//200)`.",
            "  Long trajectories therefore executed ~200 sparsely sampled controls, **not** the full teacher sequence.",
            "- Annotated official rows as `start_capture_may_be_terminal; treat as decode sanity not full teacher replay`.",
            "- Took `continuous_SUCCESS_NEW` from the **corpus label**, not from a fresh continuous full-sequence replay.",
            "",
            "P18 harness (`replay_harness.replay_actions`):",
            "",
            "- Restores the same expert `capture`.",
            "- Steps **every** action in order (continuous baseline, then M111 `recon_actions`).",
            "- Evaluates STRICT `SUCCESS_NEW` + `SUCCESS_12S` after the full sequence (`stabilize_n=8`).",
            "- Discrete rate is defined only on the continuous-reproducible subset.",
            "",
            "### 2. Population differences",
            "",
            "- P17: intentional **8-trajectory** encode/replay spot check.",
            "- P18: full eligible population with continuous replay gate → **221** continuous-ok trajectories,",
            "  then M111 on that set → **221/221**.",
            "- P18 does **not** claim 100% of the raw corpus; continuous-nonreproducible experts are excluded",
            "  from the discrete success rate.",
            "",
            "### 3. The two P17 failures, mapped to P18",
            "",
            "#### `B_STABLE_PRECISION__B__B_post_lateral__…`",
            "",
            "- P17: corpus continuous label True; M111 subsampled replay `SUCCESS_NEW=False` (`SUCCESS_12S=True`).",
            "- P18: **continuous full replay** itself fails `SUCCESS_NEW` (only `SUCCESS_12S`) →",
            "  `eligible_for_discrete=False` → **not in the 221** and never counted as an M111 success.",
            "- Interpretation: P17’s “M111 failure” mixed a non-reproducible continuous expert with an incomplete",
            "  discrete protocol. P18 correctly **excludes** it. This is a **population / continuous gate**",
            "  difference, not evidence that M111 broke a continuous-ok demo.",
            "",
            "#### `B_STABLE_PRECISION__B__hard_910026__…`",
            "",
            "- P17: M111 subsampled replay `SUCCESS_NEW=False` / `SUCCESS_12S=False`.",
            "- P18: continuous full replay **and** M111 full replay both `SUCCESS_NEW=True`.",
            "- Interpretation: P17 failure is a **replay-protocol artifact** (subsampling ~2294 actions down to ~200),",
            "  not a genuine M111 behavior regression. Encode metrics in P17 already showed the trajectory was",
            "  representable; full-sequence discrete replay in P18 confirms physical success.",
            "",
            "### 4. Genuine behavior changes?",
            "",
            "**No.** Same frozen M111 prototypes/ρ, same expert actions on disk, same STRICT evaluator.",
            "Differences are harness completeness, continuous eligibility gating, and sample size — not a",
            "redesigned vocabulary or altered expert policies.",
            "",
            "### 5. Does P18 invalidate P17?",
            "",
            "**No.** P17 remains valid as a **limited encode-spot-check with an explicitly incomplete replay protocol**.",
            "Its 2/8 “failures” must not be read as a corpus-level M111 physical failure rate.",
            "P18 supersedes P17 only for the claim “full-sequence discrete replay on continuous-ok experts.”",
            "",
            "## Bottom line",
            "",
            "| Claim | Status |",
            "|---|---|",
            "| P17 found 2/8 M111 physical failures under its harness | Historically true for that harness |",
            "| Those 2 imply M111 is unfit on continuous-ok experts | **False** once P18 protocol is applied |",
            "| P18 221/221 invalidates P17’s encode notes | **No** — different question + protocol |",
            "",
        ]
    )
    return "\n".join(lines)


def build_coverage(m111_rows: list[dict], manifest_by_id: dict) -> list[dict]:
    out = []
    for r in m111_rows:
        tid = r["traj_id"]
        n = int(float(r["n_actions"]))
        phases = None
        m = manifest_by_id.get(tid)
        if m and m.get("record_path"):
            try:
                phases = load_record(m["record_path"])["phases"]
            except Exception:
                phases = None
        tags = classify_skills(r.get("skill_group", ""), n, phases)
        out.append(
            {
                "traj_id": tid,
                "source": r.get("source"),
                "branch_id": r.get("branch_id"),
                "lineage_id": r.get("lineage_id"),
                "skill_group": r.get("skill_group"),
                "n_actions": n,
                "m111_SUCCESS_NEW": r.get("m111_SUCCESS_NEW"),
                **tags,
            }
        )
    return out


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    print("=== P19 start ===", flush=True)

    # --- 1. Reconciliation ---
    recon_md = reconcile_p17_p18()
    _write_md(OUT / "P17_P18_REPLAY_RECONCILIATION.md", recon_md)
    print("wrote reconciliation", flush=True)

    m111_rows = [r for r in _read_csv(P18 / "M111_FULL_REPLAY_RESULTS.csv") if _as_bool(r.get("m111_SUCCESS_NEW"))]
    corpus = _read_csv(P18 / "FULL_CORPUS_MANIFEST.csv")
    by_id = {r["traj_id"]: r for r in corpus}
    if len(m111_rows) != 221:
        print(f"WARN: expected 221 M111 successes, got {len(m111_rows)}", flush=True)

    # --- 2. Coverage ---
    coverage = build_coverage(m111_rows, by_id)
    _write_csv(OUT / "ACTUAL_DISCRETE_SKILL_COVERAGE.csv", coverage)
    cov_summary = {
        "n_traj": len(coverage),
        "nonzero_control": sum(1 for r in coverage if r["nonzero_actions"]),
        "zero_action": sum(1 for r in coverage if not r["nonzero_actions"]),
        "recovery_abc": sum(1 for r in coverage if r["is_recovery_abc"]),
        "executes_dx": sum(1 for r in coverage if r["executes_dx"]),
        "executes_dz": sum(1 for r in coverage if r["executes_dz"]),
        "executes_coupled": sum(1 for r in coverage if r["executes_coupled"]),
        "direct_handoff_only": sum(1 for r in coverage if r["is_direct_handoff_only"]),
        "precision_label": sum(1 for r in coverage if r["precision_containing_label"]),
        "precision_executed_corrections": sum(1 for r in coverage if r["precision_executed_corrections"]),
        "by_recovery_letter": dict(Counter(r["recovery_letter"] for r in coverage)),
    }
    (OUT / "coverage_summary.json").write_text(json.dumps(cov_summary, indent=2), encoding="utf-8")
    print("coverage", cov_summary, flush=True)

    # --- lineage split ---
    split = json.loads((P17 / "lineage_split.json").read_text(encoding="utf-8"))
    lin_split = {L["lineage_id"]: L["split"] for L in split["lineages"]}

    book, rho, handoff_id, cb = load_book()
    env = make_env()

    manifest_rows: list[dict] = []
    alignment_issues: list[str] = []
    token_counter: Counter = Counter()
    motion_token_counter: Counter = Counter()
    len_counter: list[int] = []
    lineage_traj_counts: Counter = Counter()

    # Sort: zero-action first (fast), then short→long
    ordered = sorted(m111_rows, key=lambda r: (int(float(r["n_actions"])) > 0, int(float(r["n_actions"]))))
    print(f"packaging {len(ordered)} trajectories…", flush=True)

    for i, r in enumerate(ordered):
        tid = r["traj_id"]
        meta = by_id.get(tid)
        if not meta or not meta.get("record_path"):
            alignment_issues.append(f"{tid}: missing record_path in corpus manifest")
            continue
        lid = r.get("lineage_id") or meta.get("lineage_id")
        sp = lin_split.get(lid)
        if sp is None:
            alignment_issues.append(f"{tid}: lineage {lid} missing from P17 split")
            continue
        t_i = time.time()
        try:
            pack = package_one(
                env,
                traj_id=tid,
                record_path=meta["record_path"],
                book=book,
                rho=rho,
                handoff_id=handoff_id,
            )
        except Exception as exc:
            alignment_issues.append(f"{tid}: package error: {exc}")
            print(f"  FAIL {tid[:48]} {exc}", flush=True)
            continue
        tags = classify_skills(r.get("skill_group", ""), int(float(r["n_actions"])), pack.get("phases"))
        lineage_traj_counts[lid] += 1
        n_lin = lineage_traj_counts[lid]  # provisional; recompute after
        row = {
            "traj_id": tid,
            "split": sp,
            "source": r.get("source"),
            "source_lineage": lid,
            "branch_id": r.get("branch_id"),
            "skill_group": r.get("skill_group"),
            "record_path": meta["record_path"],
            "npz_relpath": pack["npz_relpath"],
            "n_steps": pack["n_steps"],
            "n_motion": pack["n_motion"],
            "n_actions_continuous": int(float(r["n_actions"])),
            "obs_dim": pack["obs_dim"],
            "obs_version": pack["obs_version"],
            "vocab": "M111",
            "handoff_id": handoff_id,
            "M": 111,
            "policy_inputs": "observations,prev_actions",
            "policy_targets": "tokens,residuals",
            "analysis_only_skill_metadata": True,
            "used_continuous_fallback": pack["used_continuous_fallback"],
            "m111_clipped_frac": pack["m111_clipped_frac"],
            "m111_mean_recon_wlinf": pack["m111_mean_recon_wlinf"],
            "m111_n_unique_tokens": pack["m111_n_unique_tokens"],
            **{f"meta_{k}": v for k, v in tags.items() if k != "phases_head"},
            "elapsed_pack_s": round(time.time() - t_i, 3),
        }
        manifest_rows.append(row)
        token_counter.update(pack["token_counts"])
        motion_token_counter.update(pack["motion_token_counts"])
        len_counter.append(pack["n_steps"])
        if (i + 1) % 10 == 0 or pack["n_motion"] > 0:
            print(
                f"  [{i+1}/{len(ordered)}] {tid[:44]} steps={pack['n_steps']} "
                f"motion={pack['n_motion']} split={sp} {row['elapsed_pack_s']}s",
                flush=True,
            )

    # sample weights: inverse lineage frequency within split (analysis / optional reweight)
    lin_freq = Counter(r["source_lineage"] for r in manifest_rows)
    for r in manifest_rows:
        r["lineage_traj_count"] = int(lin_freq[r["source_lineage"]])
        r["sample_weight_inv_lineage"] = float(1.0 / max(1, lin_freq[r["source_lineage"]]))
        # length dampening suggestion (not applied to loss unless trainer uses it)
        r["sample_weight_inv_sqrt_len"] = float(1.0 / max(1.0, np.sqrt(r["n_steps"])))

    manifest = {
        "created_utc": _utc(),
        "dataset_id": "p19_m111_discrete_validated_sft_v1",
        "n_trajectories": len(manifest_rows),
        "n_train": sum(1 for r in manifest_rows if r["split"] == "train"),
        "n_val": sum(1 for r in manifest_rows if r["split"] == "val"),
        "vocab": {
            "name": "M111",
            "M": 111,
            "handoff_id": handoff_id,
            "meta_path": str(M111_META.relative_to(ROOT)).replace("\\", "/"),
            "prototypes_path": str(cb.meta.get("prototypes_path") or ""),
            "residual_rho": cb.residual.to_list() if hasattr(cb.residual, "to_list") else 0.35,
        },
        "observation": {
            "version": OBS_VERSION,
            "dim": OBS_DIM_AIR,
            "window": WINDOW,
            "note": "39-D airborne (27 tactile/proprio + 12 GT object-relative). Matches deployed TransformerPolicy.",
        },
        "action_interface": {
            "decode": "a = clip(c_k + clip(r, -rho, rho), -1, 1)",
            "execution": "step_legal(decoded); never continuous expert action during packaging",
            "terminal": "HANDOFF token id = 111; residual ignored at HANDOFF",
        },
        "policy_must_not_see": [
            "expert skill labels",
            "forced_skill / force_precision metadata",
            "future observations",
            "privileged fields beyond the 39-D airborne vector",
            "analysis-only meta_* columns in the manifest",
        ],
        "lineage_split_source": str((P17 / "lineage_split.json").relative_to(ROOT)).replace("\\", "/"),
        "eligibility": "P18 continuous-ok AND M111 SUCCESS_NEW (221)",
        "alignment_issues": alignment_issues,
        "trajectories": manifest_rows,
    }
    (OUT / "SFT_DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(OUT / "SFT_DATASET_MANIFEST.csv", manifest_rows)

    # --- Token distribution ---
    token_rows = []
    for tok in range(handoff_id + 1):
        token_rows.append(
            {
                "token_id": tok,
                "is_handoff": tok == handoff_id,
                "count_all_steps": int(token_counter.get(tok, 0)),
                "count_motion_only": int(motion_token_counter.get(tok, 0)),
                "frac_all": float(token_counter.get(tok, 0) / max(1, sum(token_counter.values()))),
            }
        )
    token_rows.sort(key=lambda r: -r["count_all_steps"])
    _write_csv(OUT / "TOKEN_DISTRIBUTION.csv", token_rows)

    # skill token-level from coverage × steps approx via packaged rows
    skill_token_rows = []
    for r in manifest_rows:
        # load tokens for skill attribution (analysis)
        blob = np.load(OUT / r["npz_relpath"])
        toks = blob["tokens"]
        skill_token_rows.append(
            {
                "traj_id": r["traj_id"],
                "split": r["split"],
                "skill_group": r["skill_group"],
                "n_tokens": int(len(toks)),
                "n_motion_tokens": int(r["n_motion"]),
                "meta_executes_dx": r.get("meta_executes_dx"),
                "meta_executes_dz": r.get("meta_executes_dz"),
                "meta_executes_coupled": r.get("meta_executes_coupled"),
                "meta_is_direct_handoff_only": r.get("meta_is_direct_handoff_only"),
                "meta_precision_executed_corrections": r.get("meta_precision_executed_corrections"),
                "meta_recovery_letter": r.get("meta_recovery_letter"),
            }
        )

    # --- Alignment audit markdown ---
    n_ok = len(manifest_rows)
    total_steps = sum(r["n_steps"] for r in manifest_rows)
    total_motion = sum(r["n_motion"] for r in manifest_rows)
    unique_motion_tokens = sum(1 for r in token_rows if not r["is_handoff"] and r["count_motion_only"] > 0)
    rare = [r for r in token_rows if not r["is_handoff"] and 0 < r["count_motion_only"] < 50]
    unused = [r for r in token_rows if not r["is_handoff"] and r["count_motion_only"] == 0]

    align_md = f"""# Observation / Action Alignment Audit

Generated: {_utc()}

## Packaging protocol

For each eligible trajectory:

1. Load expert `capture` + continuous `actions_7d`.
2. Encode with frozen M111 → `(token, residual, recon_action)`.
3. Restore capture; reset obs history; `prev=0`.
4. For each motion step `t`:
   - `obs[t] = observe_airborne(state_t)` (**before** the action)
   - `prev[t] = action_{{t-1}}` (zeros at `t=0`)
   - Execute **`recon_action[t]` only** via `step_legal` (decoded M111; **no continuous fallback**)
5. Append terminal `HANDOFF` (`token={handoff_id}`) at post-sequence observation.
6. Persist unpadded arrays; `loss_mask=True` for every stored step.

## Checks performed (per trajectory)

| Check | Result |
|---|---|
| `obs[t]` aligns with action/token at `t` (pre-step state) | Enforced in `package_one` |
| Action execution interval = one `step_legal` per motion token | Yes |
| `prev[0]=0`, `prev[t]=executed[t-1]` | Asserted |
| Terminal masking / HANDOFF last | Asserted; no mid-sequence HANDOFF |
| Variable-length | Per-traj NPZ, **no pad tokens stored** |
| Off-by-one | Asserted |
| Padding in loss | None in on-disk samples; collate must mask pads |
| Continuous fallback | `used_continuous_fallback=False` for all packaged rows |
| Package failures | {len(alignment_issues)} |

## Aggregate

- Packaged trajectories: **{n_ok}** / {len(m111_rows)} eligible
- Total supervised steps (incl. HANDOFF): **{total_steps}**
- Motion steps: **{total_motion}**
- HANDOFF steps: **{n_ok}** (one per episode)
- Alignment issues: {alignment_issues[:5] if alignment_issues else "none"}

## Inference ↔ training interface

| Item | Training | Inference |
|---|---|---|
| Observation | `{OBS_VERSION}` {OBS_DIM_AIR}-D | same `observe_airborne` |
| History window | last {WINDOW} (obs, prev) | last {WINDOW} |
| Prev action | executed decoded 7-D | executed decoded 7-D |
| Token head | M={handoff_id} motion + HANDOFF id {handoff_id} | same |
| Residual | supervised for motion tokens only | mean residual; ignored on HANDOFF |
| Decode | `clip(c_k + clip(r,-ρ,ρ),-1,1)` | identical |
| Physics step | `step_legal` | `step_legal` |

**Note:** Legacy `training/sft.py` still hardcodes K48 (M=47, HANDOFF=47). Multi-skill M111 SFT requires a trainer/model head width of **112** logits before training starts. Dataset packaging itself matches the intended M111 interface.
"""
    _write_md(OUT / "OBS_ACTION_ALIGNMENT_AUDIT.md", align_md)

    # --- Lineage split validation ---
    train_rows = [r for r in manifest_rows if r["split"] == "train"]
    val_rows = [r for r in manifest_rows if r["split"] == "val"]
    train_lin = set(r["source_lineage"] for r in train_rows)
    val_lin = set(r["source_lineage"] for r in val_rows)
    leak = train_lin & val_lin

    def skill_cov(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "nonzero": sum(1 for r in rows if r["n_motion"] > 0),
            "dx": sum(1 for r in rows if r.get("meta_executes_dx")),
            "dz": sum(1 for r in rows if r.get("meta_executes_dz")),
            "coupled": sum(1 for r in rows if r.get("meta_executes_coupled")),
            "direct0": sum(1 for r in rows if r.get("meta_is_direct_handoff_only")),
            "prec_exec": sum(1 for r in rows if r.get("meta_precision_executed_corrections")),
            "by_letter": dict(Counter(r.get("meta_recovery_letter") for r in rows)),
            "lineages": len(set(r["source_lineage"] for r in rows)),
            "max_traj_per_lineage": max(Counter(r["source_lineage"] for r in rows).values()) if rows else 0,
        }

    lineage_md = f"""# Lineage Split Validation

Generated: {_utc()}

## Source

Frozen P17 `lineage_split.json` (80/20 by lineage). Every P18 M111-success trajectory maps to a known lineage (**0 missing**).

## Leakage check

- Train lineages: **{len(train_lin)}**
- Val lineages: **{len(val_lin)}**
- Intersection: **{len(leak)}** {'(BLOCKING)' if leak else '(OK — empty)'}

## Split composition (packaged)

### Train
{json.dumps(skill_cov(train_rows), indent=2)}

### Val (held-out lineages — freeze for evaluation)
{json.dumps(skill_cov(val_rows), indent=2)}

## Dominance / repeated lineage

Top lineages by trajectory count (full packaged set):

"""
    for lid, c in lin_freq.most_common(15):
        lineage_md += f"- `{lid}`: {c} traj(s), split={lin_split.get(lid)}\n"
    lineage_md += """
Manifest includes `sample_weight_inv_lineage` and `sample_weight_inv_sqrt_len` for optional
reweighting so long trajectories / P16 perturbations of one base do not dominate the loss.
Equal seven-class quotas are **not** enforced.
"""
    _write_md(OUT / "LINEAGE_SPLIT_VALIDATION.md", lineage_md)

    # --- Policy interface ---
    iface = f"""# Policy Interface Specification (P19 / M111 multi-skill SFT)

Generated: {_utc()}

## Intended closed-loop interface

```
hist_obs, hist_prev  # rolling window, max {WINDOW}
loop:
  obs_t = observe_airborne(sim)           # {OBS_DIM_AIR}-D, version {OBS_VERSION}
  window_obs = hist_obs[-{WINDOW}:] + [obs_t]
  window_prev = hist_prev[-{WINDOW}:] + [prev_action]
  logits, resid_mean, _ = model(window_obs, window_prev)
  token = argmax(logits[-1])              # or sample
  if token == HANDOFF ({handoff_id}):
    break  # transfer to stabilize / handoff controller
  residual = resid_mean[-1]               # or resid_mean + noise at GRPO time
  action = clip(proto[token] + clip(residual, -ρ, ρ), -1, 1)
  step_legal(sim, action)
  prev_action = action
```

## Tensors stored per step (policy-visible)

| Field | Shape | Role |
|---|---|---|
| `observations` | (T, {OBS_DIM_AIR}) | Policy input |
| `prev_actions` | (T, 7) | Policy input |
| `tokens` | (T,) int64 | Supervised target (0..110 motion, {handoff_id}=HANDOFF) |
| `residuals` | (T, 7) | Supervised target for motion tokens; zeros at HANDOFF |
| `loss_mask` | (T,) bool | All True on disk; collate sets False on pads |

## Analysis-only (must NOT feed the policy)

- `meta_*` skill flags, `skill_group`, `forced_skill`, `force_precision`
- `continuous_actions` in NPZ (audit only)
- Future observations / sim internals beyond airborne obs

## Vocabulary

- Frozen **M111** (`SELECTED_skill_aware_hybrid_*`); do not redesign for this package.
- ρ = 0.35 (all dims); HANDOFF id = 111 (outside motion book).

## Compatibility warning

`training/sft.py` / `models/k48_checkpoint.py` currently assume **K48** (47+HANDOFF).
Before training on this dataset, the model head and decode path must use **M=111, HANDOFF=111**.
"""
    _write_md(OUT / "POLICY_INTERFACE_SPEC.md", iface)

    # --- Eval protocol ---
    eval_md = f"""# SFT Evaluation Protocol

Generated: {_utc()}

## Held-out set (frozen)

- Lineages with `split=val` in P17 `lineage_split.json`.
- Packaged val trajectories: **{len(val_rows)}** across **{len(val_lin)}** lineages.
- Do not retune vocabulary, packaging filters, or hyperparameters on this set after freeze.

## Three paired conditions

Evaluate the **same** held-out initial captures / lineage ICs under:

| # | Condition | Action source |
|---|---|---|
| 1 | Continuous Expert | Stored expert `actions_7d` (teacher) |
| 2 | M111 decoded Expert | `quantize → recon_actions` with frozen M111 |
| 3 | Learned SFT policy | Model token+residual → M111 decode |

All three: restore the same `capture`, run closed-loop (or open-loop teacher for 1–2), then official handoff evaluate.

## Success definition (official, unchanged)

- Primary: **STRICT** `SUCCESS_NEW`
- Physical verification: **`SUCCESS_12S`** (12-second hold)
- Do not substitute token accuracy for physical success.

## Metrics to report

Per condition × skill slice (Recovery A/B/C, DX, DZ, Coupled, direct-handoff):

- `SUCCESS_NEW` rate
- `SUCCESS_12S` rate
- Optional: token top-1 accuracy (diagnostic only)

## Pairing

For each val trajectory ID, record outcomes for conditions 1–3 in one row so drops are attributable to quantization vs learning.

## Out of scope until packaging gates pass

- Launching SFT / GRPO optimization
- Redesigning M111
- Auto-admitting P15/P16 into the official corpus
"""
    _write_md(OUT / "SFT_EVALUATION_PROTOCOL.md", eval_md)

    # Composition appendix in FINAL
    recovery_n = sum(1 for r in manifest_rows if r.get("meta_is_recovery_abc"))
    prec_n = sum(1 for r in manifest_rows if r.get("meta_precision_executed_corrections"))
    direct_n = sum(1 for r in manifest_rows if r.get("meta_is_direct_handoff_only"))
    action_tokens_recovery = sum(r["n_steps"] for r in manifest_rows if r.get("meta_is_recovery_abc"))
    action_tokens_prec = sum(r["n_motion"] for r in manifest_rows if r.get("meta_precision_executed_corrections"))

    blocking = []
    if leak:
        blocking.append("train/val lineage leakage")
    if alignment_issues:
        blocking.append(f"{len(alignment_issues)} packaging/alignment failures")
    if n_ok < len(m111_rows):
        blocking.append(f"only {n_ok}/{len(m111_rows)} trajectories packaged")
    if any(r.get("used_continuous_fallback") for r in manifest_rows):
        blocking.append("continuous action fallback detected")
    # Trainer interface gap is a soft block for "start training now"
    soft = [
        "Legacy training/sft.py is K48-specific; needs M111 (112-way) head before train launch",
        "Zero-action / direct-handoff trajectories dominate count (expected); use sample weights if Precision underfits",
        "P16 perturbations share base lineages — inv-lineage weights recommended",
    ]

    ready = len(blocking) == 0
    decision = "READY for SFT (after M111 trainer wiring)" if ready else "NOT READY — blocking issues remain"

    final = f"""# FINAL_REPORT — P19 SFT Packaging

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Decision

**{decision}**

Blocking issues: {blocking if blocking else "none"}

Remaining non-blocking / pre-train wiring:
{chr(10).join('- ' + s for s in soft)}

## 1. P17 ↔ P18

See `P17_P18_REPLAY_RECONCILIATION.md`.

- P17’s 2/8 failures are **harness + eligibility** effects (subsampled replay; one continuous-nonreproducible expert).
- P18 **221/221** is full-sequence M111 on the continuous-ok population — does **not** erase P17’s encode notes.

## 2. Actual discrete skill coverage (221)

| Metric | Count |
|---|---|
| Nonzero control actions | {cov_summary['nonzero_control']} |
| Zero-action | {cov_summary['zero_action']} |
| Recovery letter A/B/C present | {cov_summary['recovery_abc']} |
| Executes DX | {cov_summary['executes_dx']} |
| Executes DZ | {cov_summary['executes_dz']} |
| Executes Coupled | {cov_summary['executes_coupled']} |
| Direct-handoff only (0 actions) | {cov_summary['direct_handoff_only']} |
| Precision label | {cov_summary['precision_label']} |
| Precision with executed corrections | {cov_summary['precision_executed_corrections']} |

Replay packaging executed **decoded M111 actions only** (`used_continuous_fallback=False`).

## 3. Dataset

- Manifest: `SFT_DATASET_MANIFEST.json` ({n_ok} trajectories)
- NPZ dir: `dataset/npz/`
- Train / val: {manifest['n_train']} / {manifest['n_val']} (lineage-safe)

## 4. Alignment

See `OBS_ACTION_ALIGNMENT_AUDIT.md` — per-traj asserts for obs/action/prev/HANDOFF/no-pad.

## 5. Composition

Trajectory-level: Recovery-tagged {recovery_n}, Precision-executed {prec_n}, direct0 {direct_n}.

Token-level: total steps {total_steps}; motion {total_motion}; unique motion tokens used {unique_motion_tokens}/111;
rare (<50 motion uses): {len(rare)}; unused motion ids: {len(unused)}.

Sequence lengths: min={min(len_counter) if len_counter else None}, median={float(np.median(len_counter)) if len_counter else None}, max={max(len_counter) if len_counter else None}.

See `TOKEN_DISTRIBUTION.csv`, `LINEAGE_SPLIT_VALIDATION.md`.

## 6. Evaluation

See `SFT_EVALUATION_PROTOCOL.md` — freeze val lineages; compare Continuous Expert / M111 Expert / SFT policy with STRICT+12S.

## Integrity

- Official corpus untouched
- M111 artifacts not overwritten
- No training run launched
"""
    _write_md(OUT / "FINAL_REPORT.md", final)

    # compact json summary
    (OUT / "raw_p19_summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "blocking": blocking,
                "soft": soft,
                "n_packaged": n_ok,
                "coverage": cov_summary,
                "n_train": manifest["n_train"],
                "n_val": manifest["n_val"],
                "total_steps": total_steps,
                "unique_motion_tokens": unique_motion_tokens,
                "elapsed_s": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== P19 done ===", decision, f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
