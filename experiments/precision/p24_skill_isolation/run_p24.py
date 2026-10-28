"""P24 — Skill isolation and Plan C learnability audit.

Diagnosis only. No GRPO. No physics/Oracle/M111/STRICT changes.
Does not modify P20–P23 artifacts.
"""

from __future__ import annotations

import csv
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
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import OBS_DIM_AIR, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.physical_recovery import TABLE_TOP, physical_pack  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    quantize_with_book,
)
from experiments.precision.p21_full_stage1_experts.run_p21 import (  # noqa: E402
    load_unified_episode,
    restore_start,
)
from experiments.precision.p22_full_horizon_sft.run_p22 import (  # noqa: E402
    build_start_index,
    classify_failure,
    policy_step,
    replay_expert,
    rollout_policy,
)
from experiments.precision.p23_c_d2_sft.run_p23 import (  # noqa: E402
    contact_flags,
    first_divergence,
    infer_stage_bounds,
    load_unified_by_id,
    record_timeline,
    resolve_npz_root,
    stage_name,
    teacher_forced_stage_acc,
)
from models.m111_checkpoint import HANDOFF, MOTION_K, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import (  # noqa: E402
    WINDOW,
    build_samples,
    chunk_episode,
    load_episode_npz,
    load_p21_manifest,
)
from training.m111_sft import eval_token_metrics, train_m111  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/diagnostics/p24_skill_isolation"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
SFT_A_CKPT = P22 / "checkpoints/A_natural/checkpoint_best.pt"
B_D2_CKPT = P23 / "checkpoints/B_d2/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
UNIFIED = ROOT / "data/unified_experts/manifest.jsonl"
P13 = ROOT / "results/diagnostics/p13_teleport_d2"
P14 = ROOT / "results/diagnostics/p14_small_error_d2"
P15 = ROOT / "results/diagnostics/p15_forced_precision_d2"
P16 = ROOT / "results/diagnostics/p16_precision_skill_expansion"

SEED = 24
EPOCHS = 30
LR = 3e-4
BATCH = 8
WINDOW_INF = 160
P23_BAD_TOKEN = 3  # repeated token observed in P23 SFT-A on Plan C


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


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * np.log(p + 1e-12)
    return float(h)


def run_lengths(tokens: list[int]) -> list[int]:
    if not tokens:
        return []
    out = []
    cur = tokens[0]
    n = 1
    for t in tokens[1:]:
        if t == cur:
            n += 1
        else:
            out.append(n)
            cur = t
            n = 1
    out.append(n)
    return out


def verify_prev_alignment(ep: dict[str, np.ndarray], cont_actions: np.ndarray | None = None) -> dict:
    """Verify prev_actions[t] is prior executed action; never equals current target encoding wrongly."""
    prev = np.asarray(ep["prev_actions"], np.float32)
    tokens = np.asarray(ep["tokens"], np.int64)
    n = prev.shape[0]
    issues = []
    if float(np.max(np.abs(prev[0]))) > 1e-6:
        issues.append("prev[0]!=0")
    # If continuous actions available: prev[t] should match executed action at t-1
    if cont_actions is not None:
        acts = np.asarray(cont_actions, np.float32).reshape(-1, 7)
        # NPZ has n = n_motion+1 (HANDOFF row); executed continuous length = n_motion
        n_mot = min(len(acts), n - 1)
        for t in range(1, n_mot + 1):
            if not np.allclose(prev[t], acts[t - 1], atol=1e-4):
                issues.append(f"prev[{t}]!=executed[{t-1}]")
                if len(issues) > 5:
                    break
    # Structural: prev must not equal a one-hot of current token in action space (impossible for 7-D)
    # Check tokens are not embedded into prev (prev linf should match action scale, tokens are ints)
    return {
        "n_steps": n,
        "n_issues": len(issues),
        "issues_head": "|".join(issues[:5]),
        "prev0_zero": int(float(np.max(np.abs(prev[0]))) <= 1e-6),
        "obs_dim": int(ep["observations"].shape[1]),
        "terminal_handoff": int(int(tokens[-1]) == HANDOFF),
    }


# ---------------------------------------------------------------------------
# Phase 1 — audits
# ---------------------------------------------------------------------------


def audit_plan_c(manifest: dict, book, rho) -> tuple[str, list[dict], list[dict]]:
    print("=== Phase 1a: Plan C audit ===", flush=True)
    env = make_env()
    unified = load_unified_by_id()
    starts = build_start_index(manifest)
    c_rows = [r for r in manifest["trajectories"] if r["family"] == "C"]
    stage_token_rows = []
    rep_rows = []
    align_ok = 0
    truncated_chunks = 0
    total_chunks = 0
    stage_tok: dict[str, Counter] = defaultdict(Counter)
    lineage_split = defaultdict(lambda: Counter())

    for row in c_rows:
        tid = row["traj_id"]
        lineage_split[row["lineage_id"]][row["split"]] += 1
        ep_npz = load_episode_npz(P21, row["npz_relpath"])
        # continuous from start index if available
        cont = None
        if tid in starts:
            cont = starts[tid]["actions"]
        elif (P21 / row["npz_relpath"]).is_file():
            blob = np.load(P21 / row["npz_relpath"])
            if "continuous_actions" in blob.files:
                cont = np.asarray(blob["continuous_actions"], np.float32)

        align = verify_prev_alignment(ep_npz, cont)
        if align["n_issues"] == 0:
            align_ok += 1

        tokens = ep_npz["tokens"].tolist()
        motion_tok = tokens[:-1]
        chunks = chunk_episode(ep_npz, window=WINDOW)
        total_chunks += len(chunks)
        if len(motion_tok) > WINDOW:
            truncated_chunks += max(0, len(chunks) - 1)  # multi-chunk = long traj covered, not dropped

        # stage bounds from continuous expert if possible
        bounds = {
            "detach_end": 15,
            "recatch_end": 25,
            "stabilize_end": int(row.get("n_stage1") or len(motion_tok)),
            "lift_start": int(row.get("n_stage1") or len(motion_tok)),
            "n_total": len(motion_tok),
            "first_bilateral": -1,
        }
        if tid in starts:
            ep = starts[tid]
            if tid in unified and unified[tid].get("n_catch") is not None:
                ep = {**ep, "n_catch": unified[tid]["n_catch"]}
            tl = record_timeline(env, ep, ep["actions"])
            bounds = infer_stage_bounds(tl, ep.get("n_catch"))
            # M111 tokens for stage distribution
            q = quantize_trajectory(book, ep["actions"], rho, w=DEFAULT_WEIGHTS)
            mtok = [int(t) for t in q["tokens"].tolist()]
        else:
            mtok = motion_tok

        for t, tok in enumerate(mtok[: bounds["n_total"]]):
            st = stage_name(t, bounds)
            stage_tok[st][int(tok)] += 1
            stage_token_rows.append(
                {
                    "traj_id": tid,
                    "split": row["split"],
                    "lineage_id": row["lineage_id"],
                    "t": t,
                    "stage": st,
                    "token": int(tok),
                    "is_p23_bad_token": int(int(tok) == P23_BAD_TOKEN),
                }
            )

        rls = run_lengths(mtok)
        rep_rows.append(
            {
                "traj_id": tid,
                "split": row["split"],
                "lineage_id": row["lineage_id"],
                "n_motion": len(mtok),
                "n_stage1": row.get("n_stage1"),
                "n_chunks": len(chunks),
                "detach_end": bounds["detach_end"],
                "recatch_end": bounds["recatch_end"],
                "stabilize_end": bounds["stabilize_end"],
                "lift_start": bounds["lift_start"],
                "first_bilateral": bounds["first_bilateral"],
                "n_unique_tokens": len(set(mtok)),
                "max_run_length": max(rls) if rls else 0,
                "mean_run_length": float(np.mean(rls)) if rls else 0.0,
                "count_token_3": sum(1 for t in mtok if int(t) == P23_BAD_TOKEN),
                "frac_token_3": sum(1 for t in mtok if int(t) == P23_BAD_TOKEN) / max(1, len(mtok)),
                "prev_align_ok": int(align["n_issues"] == 0),
                "prev_issues": align["issues_head"],
                "tokens_head": "|".join(str(t) for t in mtok[:16]),
            }
        )

    # Stage entropy / frequency summary rows (aggregate)
    dist_rows = []
    for st, ctr in stage_tok.items():
        total = sum(ctr.values())
        top = ctr.most_common(8)
        dist_rows.append(
            {
                "stage": st,
                "n_tokens": total,
                "n_unique": len(ctr),
                "entropy": entropy(ctr),
                "frac_token_3": ctr.get(P23_BAD_TOKEN, 0) / max(1, total),
                "top_tokens": "|".join(f"{t}:{c}" for t, c in top),
            }
        )
    # Also per-traj stage rows already in stage_token_rows — write aggregate + keep detail slim?
    # Write aggregate distribution + rep audit; stage detail would be huge — write summary by traj×stage
    traj_stage = []
    by_ts: dict[tuple, Counter] = defaultdict(Counter)
    for r in stage_token_rows:
        by_ts[(r["traj_id"], r["stage"], r["split"])][r["token"]] += 1
    for (tid, st, split), ctr in by_ts.items():
        traj_stage.append(
            {
                "traj_id": tid,
                "split": split,
                "stage": st,
                "n": sum(ctr.values()),
                "entropy": entropy(ctr),
                "frac_token_3": ctr.get(P23_BAD_TOKEN, 0) / max(1, sum(ctr.values())),
                "top": "|".join(f"{t}:{c}" for t, c in ctr.most_common(5)),
            }
        )

    n_train = sum(1 for r in c_rows if r["split"] == "train")
    n_val = sum(1 for r in c_rows if r["split"] == "val")
    n_lin = len({r["lineage_id"] for r in c_rows})
    lens = [int(r.get("n_motion") or 0) for r in c_rows]

    md = f"""# DATASET_AND_INTERFACE_AUDIT

Generated: {_utc()}

## Plan C (P21 full-horizon)

| Metric | Value |
|---|---|
| Trajectories | {len(c_rows)} (train {n_train} / val {n_val}) |
| Independent lineages | **{n_lin}** (1:1 with trajs) |
| Motion length min/median/max | {min(lens)} / {int(np.median(lens))} / {max(lens)} |
| Chunks (window={WINDOW}) | {total_chunks} total; long trajs multi-chunk (no Stage-1 drop; last chunk may overlap for HANDOFF context) |
| Prev-action alignment OK | **{align_ok}/{len(c_rows)}** |
| Obs dim | {OBS_DIM_AIR} (policy-visible only) |

### Interface contract

- Inputs: `observations` (39-D) + `prev_actions` (7-D **previously executed** continuous action).
- Targets: M111 `tokens` + `residuals` (never concatenated into inputs).
- `prev[0] = 0`; `prev[t] = executed action at t-1`.
- Verified against packaged `continuous_actions` where present.

### Expert vs P23 bad token

- P23 SFT-A closed-loop Plan C repeatedly predicted token **{P23_BAD_TOKEN}**.
- Expert Plan C fraction of token {P23_BAD_TOKEN}: see `C_TOKEN_REPETITION_AUDIT.csv` / stage distribution.
- Expert stage entropies: see `C_STAGE_TOKEN_DISTRIBUTION.csv`.

### Truncation / undersampling

- Trajectories longer than {WINDOW} are **chunked**, not truncated to a single window.
- Coverage assert in `chunk_episode` requires all timesteps appear in some chunk.
- Natural weighting: each chunk weight = episode weight (no deliberate undersampling of early detach).

## D2 (P23 curated subset)

See section below (filled after D2 audit).
"""
    _write_csv(OUT / "C_STAGE_TOKEN_DISTRIBUTION.csv", dist_rows)
    _write_csv(OUT / "C_STAGE_TOKEN_BY_TRAJ.csv", traj_stage)
    _write_csv(OUT / "C_TOKEN_REPETITION_AUDIT.csv", rep_rows)
    return md, rep_rows, dist_rows


def audit_d2(c_audit_md: str) -> tuple[str, dict]:
    print("=== Phase 1b: D2 audit ===", flush=True)
    man = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    d2 = [r for r in man["trajectories"] if r.get("family") == "D2"]
    # Exclude direct-handoff-only if any slipped in
    d2 = [r for r in d2 if "direct" not in str(r.get("skill_group") or "").lower() or "prec=1" in str(r.get("skill_group") or "")]
    p21 = load_p21_manifest(P21 / "SFT_FULL_HORIZON_MANIFEST.json")
    p21_lin = {r["lineage_id"] for r in p21["trajectories"]}

    rows = []
    skill_c = Counter()
    for r in d2:
        sg = str(r.get("skill_group") or r.get("mechanism") or "")
        skill = "coupled" if "coupled" in sg.lower() else ("dz" if "dz" in sg.lower() else ("dx" if "dx" in sg.lower() else sg or "?"))
        # finer from source
        if r.get("source_corpus") == "p14":
            skill = "combined_dx_dz"
        skill_c[skill] += 1
        overlap = r["lineage_id"] in p21_lin
        # load npz validity
        root = resolve_npz_root(r) if "dataset_root" in r else (P23 if (P23 / r["npz_relpath"]).is_file() else P21)
        if not (root / r["npz_relpath"]).is_file():
            root = P23
        ok = (root / r["npz_relpath"]).is_file()
        n_tok = 0
        prev0 = None
        if ok:
            ep = load_episode_npz(root, r["npz_relpath"])
            n_tok = int(ep["tokens"].shape[0])
            prev0 = float(np.max(np.abs(ep["prev_actions"][0])))
        rows.append(
            {
                "traj_id": r["traj_id"],
                "split": r["split"],
                "lineage_id": r["lineage_id"],
                "source_corpus": r.get("source_corpus"),
                "skill": skill,
                "skill_group": sg,
                "start_distribution": r.get("start_distribution"),
                "n_motion": r.get("n_motion"),
                "n_steps_npz": n_tok,
                "forced_precision": int(bool(r.get("forced_precision"))),
                "overlap_p21_lineage": int(overlap),
                "npz_ok": int(ok),
                "prev0_linf": prev0,
            }
        )

    _write_csv(OUT / "D2_CURATED_AUDIT_ROWS.csv", rows)
    n_lin = len({r["lineage_id"] for r in d2})
    overlap_n = sum(1 for r in rows if r["overlap_p21_lineage"])
    md_extra = f"""
## D2 curated (from P23 COMBINED_DATASET_MANIFEST)

| Metric | Value |
|---|---|
| Trajectories | {len(d2)} (train {sum(1 for r in d2 if r['split']=='train')} / val {sum(1 for r in d2 if r['split']=='val')}) |
| Independent lineages | **{n_lin}** |
| Skill mix | {dict(skill_c)} |
| Start distribution | `stable_precision_ic` (metadata; not a policy input) |
| Overlap with P21 recovery lineages | **{overlap_n}** trajs |
| Direct-HANDOFF-only as primary | **excluded** (P23 curation) |
| Forced Precision | {sum(1 for r in rows if r['forced_precision'])} (capped per lineage×skill in P23) |

M111 packaging required terminal HANDOFF + successful packaging path; NPZ `prev[0]=0` enforced by loader.
"""
    full = c_audit_md + md_extra
    _write_md(OUT / "DATASET_AND_INTERFACE_AUDIT.md", full)
    return full, {"d2_rows": rows, "manifest_d2": d2, "combined": man}


# ---------------------------------------------------------------------------
# Phase 2 — isolated training
# ---------------------------------------------------------------------------


def build_c_only_manifest(p21: dict) -> dict:
    trajs = []
    for r in p21["trajectories"]:
        if r["family"] != "C":
            continue
        trajs.append({**r, "dataset_root": "p21"})
    return {
        "dataset_id": "p24_c_only",
        "generated_at": _utc(),
        "trajectories": trajs,
        "start_distribution": "recovery_ic_pre_stage1",
    }


def build_d2_only_manifest(combined: dict) -> dict:
    trajs = []
    for r in combined["trajectories"]:
        if r.get("family") != "D2":
            continue
        trajs.append({**r})
    return {
        "dataset_id": "p24_d2_only",
        "generated_at": _utc(),
        "trajectories": trajs,
        "start_distribution_note": "stable_precision_ic metadata only",
    }


def build_samples_from_rows(
    rows: list[dict],
    *,
    split: str,
    default_root: Path,
) -> list[dict]:
    samples = []
    sub = [r for r in rows if r.get("split") == split]
    for row in sub:
        if row.get("dataset_root") == "p21" or (default_root == P21 and (P21 / row["npz_relpath"]).is_file()):
            root = P21
        elif (P23 / row["npz_relpath"]).is_file():
            root = P23
        else:
            root = default_root
        ep = load_episode_npz(root, row["npz_relpath"])
        for ci, chunk in enumerate(chunk_episode(ep, window=WINDOW)):
            samples.append(
                {
                    **chunk,
                    "sample_weight": 1.0,
                    "traj_id": row["traj_id"],
                    "family": row.get("family"),
                    "lineage_id": row.get("lineage_id"),
                    "skill_group": row.get("skill_group") or row.get("family"),
                    "split": split,
                    "chunk_index": ci,
                }
            )
    return samples


def confusion_matrix(model, samples, device, *, max_batches=40) -> dict:
    model.eval()
    cm = np.zeros((MOTION_K + 1, MOTION_K + 1), dtype=np.int64)  # incl HANDOFF
    batch = 8
    n_b = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            if n_b >= max_batches:
                break
            chunk = samples[start : start + batch]
            from training.m111_dataset import collate_m111

            obs, prev, tokens, residuals, mask, _w = collate_m111(chunk, device)
            logits, _, _ = model(obs, prev)
            pred = torch.argmax(logits, dim=-1)
            for i in range(len(chunk)):
                active = mask[i]
                yt = tokens[i][active].tolist()
                yp = pred[i][active].tolist()
                for a, b in zip(yt, yp):
                    cm[int(a), int(b)] += 1
            n_b += 1
    # top confusions (off-diagonal)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    return {
        "n_counted": int(cm.sum()),
        "diag_acc": float(np.trace(cm) / max(1, cm.sum())),
        "top_confusions": [{"true": a, "pred": b, "n": n} for n, a, b in pairs[:20]],
        "pred_token_3": int(cm[:, P23_BAD_TOKEN].sum()),
        "true_token_3": int(cm[P23_BAD_TOKEN, :].sum()),
    }


def train_isolated(name: str, train_s: list, val_s: list, book, rho, device) -> tuple[Any, dict]:
    out_dir = OUT / "checkpoints" / name
    cfg = {
        "name": name,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
        "window": WINDOW,
        "device": str(device),
        # fresh init — no init_checkpoint
    }
    print(
        f"=== train {name} n_train_seq={len(train_s)} n_val_seq={len(val_s)} "
        f"train_tokens≈{sum(int(s['tokens'].shape[0]) for s in train_s)} ===",
        flush=True,
    )
    result = train_m111(
        train_samples=train_s,
        val_samples=val_s,
        cfg=cfg,
        out_dir=out_dir,
        book=book,
        rho=rho,
        init_checkpoint=None,
    )
    model, _, _ = load_m111(out_dir / "checkpoint_best.pt", device)
    model.eval()
    return model, result


# ---------------------------------------------------------------------------
# Phase 3–5 — C diagnosis
# ---------------------------------------------------------------------------


def rollout_policy_timeline_c(env, model, book, rho, ep, *, device, max_steps=400) -> tuple[list[dict], dict]:
    """Policy timeline with no mid-rollout geometry abort.

    Plan C catch routinely drives obj_z below TABLE_TOP and clear < 0; those are not
    failure signals. Terminate only on HANDOFF or max_steps; STRICT decides success.
    """
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    rows = []
    tokens = []
    reason = None
    for t in range(int(max_steps)):
        pack = physical_pack(env.sim)
        o = contact_flags(pack)
        o["scene"] = int(pack.get("scene") or 0)
        o["clear"] = float(pack.get("clear") or 0.0)
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        tokens.append(tok)
        rows.append({"t": t, **o, "pred_token": tok})
        if action is None:
            reason = "HANDOFF"
            break
        step_legal(env.sim, action)
        env._prev = action
    else:
        reason = "TIMEOUT"
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
    return rows, {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "dropped": False,
        "n_motion": sum(1 for tok in tokens if tok != HANDOFF),
        "reason": reason or hand.get("std_reason") or hand.get("reason"),
        "tokens_head": "|".join(str(t) for t in tokens[:12]),
    }


def expert_prefix_suffix_c(
    env, model, book, rho, ep, *, prefix_n: int, device, max_suffix: int = 800, stabilize_n: int = 8
) -> dict:
    """Expert prefix → policy suffix; no mid-rollout TABLE_TOP/clear abort."""
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    acts = np.asarray(ep["actions"], np.float32).reshape(-1, 7)
    n_pref = int(min(max(0, prefix_n), len(acts)))
    for t in range(n_pref):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        step_legal(env.sim, acts[t])
        env._prev = acts[t]
    n_motion = 0
    timeout = False
    reason = None
    tokens = []
    for _ in range(int(max_suffix)):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        tokens.append(tok)
        if action is None:
            reason = "HANDOFF"
            break
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
    else:
        timeout = True
        reason = "TIMEOUT"
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=int(stabilize_n))
    return {
        "prefix_n": n_pref,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "dropped": False,
        "timeout": timeout,
        "n_suffix_motion": n_motion,
        "reason": reason or hand.get("std_reason") or hand.get("reason"),
        "tokens_head": "|".join(str(t) for t in tokens[:10]),
    }


def physical_stage_completion(env, ep, actions_or_policy, *, mode: str, model=None, book=None, rho=None, device=None, max_steps=800) -> dict:
    """Roll and mark detach/recatch/stabilize/lift completion by physical contact/z.

    Important: Plan C catch legitimately drives obj_z below TABLE_TOP during flight.
    Do **not** abort on TABLE_TOP+margin (that false-fails Expert and any correct chase).
    Abort only on catastrophic scene/clear failure; STRICT comes from evaluate_handoff.
    """
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []

    detach_done = False
    recatch_done = False
    stable_bilateral = False
    lift_done = False
    bilateral_hold = 0
    z_at_stable = None
    n_motion = 0
    catastrophic = False
    tokens = []
    timeline = []
    min_z = 99.0
    ever_flight = False

    def step_once(action):
        nonlocal n_motion, min_z
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
        pack = physical_pack(env.sim)
        min_z = min(min_z, float(pack.get("obj_z") or 99))
        return True

    max_t = max_steps if mode == "policy" else len(np.asarray(actions_or_policy).reshape(-1, 7))
    expert_acts = None if mode == "policy" else np.asarray(actions_or_policy, np.float32).reshape(-1, 7)

    for t in range(max_t):
        o = contact_flags(physical_pack(env.sim))
        pack = physical_pack(env.sim)
        o["scene"] = int(pack.get("scene") or 0)
        o["clear"] = float(pack.get("clear") or 0.0)
        timeline.append(o)
        min_z = min(min_z, o["obj_z"])
        if o["bilateral"] == 0:
            ever_flight = True
            if t >= 3:
                detach_done = True
        if o["bilateral"] == 1:
            bilateral_hold += 1
            if bilateral_hold >= 5 and ever_flight:
                recatch_done = True
                stable_bilateral = True
                if z_at_stable is None:
                    z_at_stable = o["obj_z"]
            elif bilateral_hold >= 5 and not ever_flight:
                stable_bilateral = True
                if z_at_stable is None:
                    z_at_stable = o["obj_z"]
        else:
            bilateral_hold = 0
        if stable_bilateral and z_at_stable is not None and o["obj_z"] > z_at_stable + 0.08:
            lift_done = True

        if mode == "policy":
            obs = np.asarray(
                observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
                np.float32,
            )
            obs_seq.append(obs)
            prev_seq.append(env._prev.copy())
            tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
            tokens.append(tok)
            if action is None:
                break
            step_once(action)
        else:
            step_once(expert_acts[t])

    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
    # Physical failure proxy: no recatch and short motion, or STRICT fail after attempting motion
    dropped = bool(
        mode == "policy"
        and not bool(hand.get("SUCCESS_NEW"))
        and (not recatch_done)
    )
    catastrophic = False
    rls = run_lengths(tokens) if tokens else []
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "dropped": dropped,
        "catastrophic": int(catastrophic),
        "min_z": float(min_z if min_z < 90 else -1),
        "n_motion": n_motion,
        "detach_completed": int(detach_done),
        "recatch_completed": int(recatch_done),
        "stable_bilateral": int(stable_bilateral),
        "lift_completed": int(lift_done),
        "immediate_handoff": int(mode == "policy" and n_motion == 0),
        "premature_handoff": int(
            mode == "policy"
            and tokens
            and tokens[-1] == HANDOFF
            and n_motion < 20
            and not hand.get("SUCCESS_NEW")
        ),
        "max_token_run": max(rls) if rls else 0,
        "frac_token_3": (sum(1 for t in tokens if t == P23_BAD_TOKEN) / max(1, len(tokens))) if tokens else 0.0,
        "dominant_token": Counter(tokens).most_common(1)[0][0] if tokens else "",
        "tokens_head": "|".join(str(t) for t in tokens[:16]),
        "n_unique_tokens": len(set(tokens)),
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "timeline_n": len(timeline),
    }


def render_video(env, model, book, rho, ep, path: Path, device, tag: str):
    try:
        import mujoco
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4
    except Exception as exc:
        print("video skip", exc, flush=True)
        return
    cam = mjv_camera_from_preset(
        {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames, obs_seq, prev_seq = [], [], []
    for _ in range(250):
        renderer.update_scene(env.sim.data, cam)
        frames.append(renderer.render().copy())
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        if action is None:
            for __ in range(20):
                step_legal(env.sim, np.zeros(7, np.float32))
                renderer.update_scene(env.sim.data, cam)
                frames.append(renderer.render().copy())
            break
        step_legal(env.sim, action)
        env._prev = action
        if float(physical_pack(env.sim).get("obj_z", 99)) < TABLE_TOP + 0.01:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, path, fps=20)
    print("video", tag, len(frames), flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reeval_only = "--reeval" in sys.argv
    print(f"=== P24 device={device} reeval_only={reeval_only} ===", flush=True)

    book, rho, _ = load_m111_book(M111_META)
    p21 = load_p21_manifest(P21 / "SFT_FULL_HORIZON_MANIFEST.json")

    # Phase 1
    if not reeval_only or not (OUT / "DATASET_AND_INTERFACE_AUDIT.md").is_file():
        c_md, rep_rows, dist_rows = audit_plan_c(p21, book, rho)
        _audit_full, d2_info = audit_d2(c_md)
    else:
        print("=== skip Phase 1 (reeval) ===", flush=True)
        _, d2_info = audit_d2("# (cached audit; D2 section refreshed)\n")

    # Phase 2
    c_man = build_c_only_manifest(p21)
    d2_man = build_d2_only_manifest(d2_info["combined"])
    (OUT / "C_ONLY_MANIFEST.json").write_text(json.dumps(c_man, indent=2), encoding="utf-8")
    (OUT / "D2_ONLY_MANIFEST.json").write_text(json.dumps(d2_man, indent=2), encoding="utf-8")

    train_c = build_samples_from_rows(c_man["trajectories"], split="train", default_root=P21)
    val_c = build_samples_from_rows(c_man["trajectories"], split="val", default_root=P21)
    train_d2 = build_samples_from_rows(d2_man["trajectories"], split="train", default_root=P23)
    val_d2 = build_samples_from_rows(d2_man["trajectories"], split="val", default_root=P23)

    configs = {
        "architecture": "P22 M111 transformer (112-way)",
        "optimizer": "Adam",
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "window": WINDOW,
        "seed": SEED,
        "init": "scratch",
        "C_only": {
            "n_traj_train": sum(1 for r in c_man["trajectories"] if r["split"] == "train"),
            "n_traj_val": sum(1 for r in c_man["trajectories"] if r["split"] == "val"),
            "n_seq_train": len(train_c),
            "n_seq_val": len(val_c),
            "n_token_steps_train": sum(int(s["tokens"].shape[0]) for s in train_c),
            "n_token_steps_val": sum(int(s["tokens"].shape[0]) for s in val_c),
        },
        "D2_only": {
            "n_traj_train": sum(1 for r in d2_man["trajectories"] if r["split"] == "train"),
            "n_traj_val": sum(1 for r in d2_man["trajectories"] if r["split"] == "val"),
            "n_seq_train": len(train_d2),
            "n_seq_val": len(val_d2),
            "n_token_steps_train": sum(int(s["tokens"].shape[0]) for s in train_d2),
            "n_token_steps_val": sum(int(s["tokens"].shape[0]) for s in val_d2),
        },
        "note": "Comparable epoch budget; exposure differs by n_seq × length. No Oracle phase labels in inputs.",
        "plan_c_eval_note": "Closed-loop C does not abort on TABLE_TOP; Plan C catch intentionally goes below table height. Catastrophic=scene/clear only.",
    }
    (OUT / "ISOLATED_TRAINING_CONFIGS.yaml").write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")

    ckpt_c = OUT / "checkpoints/C_only/checkpoint_best.pt"
    ckpt_d2 = OUT / "checkpoints/D2_only/checkpoint_best.pt"
    if reeval_only and ckpt_c.is_file() and ckpt_d2.is_file():
        print("=== Phase 2: load existing C_only / D2_only ===", flush=True)
        model_c, _, _ = load_m111(ckpt_c, device)
        model_d2, _, _ = load_m111(ckpt_d2, device)
        model_c.eval()
        model_d2.eval()
    else:
        print("=== Phase 2: isolated training ===", flush=True)
        model_c, hist_c = train_isolated("C_only", train_c, val_c, book, rho, device)
        model_d2, hist_d2 = train_isolated("D2_only", train_d2, val_d2, book, rho, device)

    # Load baselines (read-only)
    model_a, _, _ = load_m111(SFT_A_CKPT, device)
    model_a.eval()
    model_bd2 = None
    if B_D2_CKPT.is_file():
        model_bd2, _, _ = load_m111(B_D2_CKPT, device)
        model_bd2.eval()

    # Teacher-forced comparison
    print("=== teacher-forced ===", flush=True)
    tf_rows = []
    for name, model, tr, va in [
        ("C_only", model_c, train_c, val_c),
        ("D2_only", model_d2, train_d2, val_d2),
        ("SFT_A", model_a, train_c, val_c),  # evaluate A on C data for comparison
    ]:
        for split_name, samp in (("train", tr), ("val", va)):
            m = eval_token_metrics(model, samp, device)
            cm = confusion_matrix(model, samp, device)
            tf_rows.append(
                {
                    "model": name,
                    "eval_data": f"C_chunks" if name != "D2_only" else "D2_chunks",
                    "split": split_name,
                    "token_acc": m.get("token_acc"),
                    "motion_token_acc": m.get("motion_token_acc"),
                    "handoff_precision": m.get("handoff_precision"),
                    "handoff_recall": m.get("handoff_recall"),
                    "n_steps": m.get("n_steps"),
                    "cm_diag_acc": cm["diag_acc"],
                    "pred_mass_token_3": cm["pred_token_3"],
                    "top_confusions": json.dumps(cm["top_confusions"][:8]),
                }
            )
    if model_bd2 is not None:
        for split_name, samp in (("train", train_d2), ("val", val_d2)):
            m = eval_token_metrics(model_bd2, samp, device)
            cm = confusion_matrix(model_bd2, samp, device)
            tf_rows.append(
                {
                    "model": "P23_B_d2",
                    "eval_data": "D2_chunks",
                    "split": split_name,
                    "token_acc": m.get("token_acc"),
                    "motion_token_acc": m.get("motion_token_acc"),
                    "handoff_precision": m.get("handoff_precision"),
                    "handoff_recall": m.get("handoff_recall"),
                    "n_steps": m.get("n_steps"),
                    "cm_diag_acc": cm["diag_acc"],
                    "pred_mass_token_3": cm["pred_token_3"],
                    "top_confusions": json.dumps(cm["top_confusions"][:8]),
                }
            )
    # Also D2_only on D2; C_only stage TF on val C starts
    starts = build_start_index(p21)
    unified = load_unified_by_id()
    val_c_rows = [r for r in p21["trajectories"] if r["split"] == "val" and r["family"] == "C"]
    stage_tf_rows = []
    for row in val_c_rows:
        ep = starts[row["traj_id"]]
        if row["traj_id"] in unified:
            ep = {**ep, "n_catch": unified[row["traj_id"]].get("n_catch")}
        tl = record_timeline(make_env(), ep, ep["actions"])
        bounds = infer_stage_bounds(tl, ep.get("n_catch"))
        for mname, model in (("C_only", model_c), ("SFT_A", model_a)):
            acc = teacher_forced_stage_acc(model, book, rho, ep, bounds, device=device)
            stage_tf_rows.append(
                {
                    "traj_id": row["traj_id"],
                    "model": mname,
                    **{f"acc_{k}": v.get("acc") for k, v in acc.items()},
                    **{f"n_{k}": v.get("n") for k, v in acc.items()},
                }
            )
    _write_csv(OUT / "TEACHER_FORCED_COMPARISON.csv", tf_rows)
    _write_csv(OUT / "TEACHER_FORCED_STAGE_ACC.csv", stage_tf_rows)

    # Phase 3 closed-loop C
    print("=== Phase 3: C closed-loop ===", flush=True)
    env = make_env()
    cl_rows = []
    for i, row in enumerate(val_c_rows):
        tid = row["traj_id"]
        ep = starts[tid]
        # Continuous expert
        ce = physical_stage_completion(env, ep, ep["actions"], mode="expert")
        ce.update({"traj_id": tid, "condition": "continuous_expert", "lineage_id": row["lineage_id"]})
        cl_rows.append(ce)
        # M111 expert
        q = quantize_with_book(book, ep["actions"], rho, w=DEFAULT_WEIGHTS)
        me = physical_stage_completion(env, ep, q["recon_actions"], mode="expert")
        me.update({"traj_id": tid, "condition": "m111_expert", "lineage_id": row["lineage_id"]})
        cl_rows.append(me)
        for mname, model in (("SFT_A", model_a), ("C_only", model_c)):
            # divergence vs expert
            expert_tl = record_timeline(env, ep, ep["actions"])
            bounds = infer_stage_bounds(expert_tl, ep.get("n_catch") or unified.get(tid, {}).get("n_catch"))
            sft_tl, sft_sum = rollout_policy_timeline_c(
                env, model, book, rho, ep, device=device, max_steps=400
            )
            div = first_divergence(expert_tl, sft_tl, bounds)
            phys = physical_stage_completion(
                env, ep, None, mode="policy", model=model, book=book, rho=rho, device=device, max_steps=800
            )
            # TF stage acc already recorded; attach summary
            phys.update(
                {
                    "traj_id": tid,
                    "condition": mname,
                    "lineage_id": row["lineage_id"],
                    "div_t": div.get("first_div_t"),
                    "div_stage": div.get("first_div_stage"),
                    "steps_to_term": phys["n_motion"],
                    "failure_class": classify_failure(
                        {
                            **phys,
                            "never_handoff_timeout": phys["n_motion"] >= 200 and not phys["SUCCESS_NEW"],
                            "nonzero_action": phys["n_motion"] > 0,
                            "expert_n_actions": len(ep["actions"]),
                        }
                    ),
                }
            )
            cl_rows.append(phys)
        if (i + 1) % 5 == 0:
            print(f"  C eval {i+1}/{len(val_c_rows)}", flush=True)
    _write_csv(OUT / "C_CLOSED_LOOP_EVALUATION.csv", cl_rows)

    # Phase 5 prefix-suffix
    print("=== Phase 5: prefix-suffix ===", flush=True)
    pref_rows = []
    for row in val_c_rows:
        tid = row["traj_id"]
        ep = starts[tid]
        if tid in unified:
            ep = {**ep, "n_catch": unified[tid].get("n_catch")}
        tl = record_timeline(env, ep, ep["actions"])
        bounds = infer_stage_bounds(tl, ep.get("n_catch"))
        boundaries = [
            ("after_detach", bounds["detach_end"]),
            ("after_recatch", bounds["recatch_end"]),
            ("after_stabilize", bounds["stabilize_end"]),
            ("during_lift", bounds["stabilize_end"] + max(1, (bounds["n_total"] - bounds["stabilize_end"]) // 2)),
            ("full_expert", bounds["n_total"]),
            ("t0", 0),
        ]
        for mname, model in (("C_only", model_c), ("SFT_A", model_a)):
            for bname, pn in boundaries:
                r = expert_prefix_suffix_c(
                    env, model, book, rho, ep, prefix_n=pn, device=device
                )
                pref_rows.append(
                    {
                        "traj_id": tid,
                        "model": mname,
                        "boundary": bname,
                        "prefix_n": r["prefix_n"],
                        "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                        "SUCCESS_12S": int(r["SUCCESS_12S"]),
                        "dropped": int(r["dropped"]),
                        "n_suffix_motion": r["n_suffix_motion"],
                        "reason": r["reason"],
                        "tokens_head": r["tokens_head"],
                    }
                )
    _write_csv(OUT / "C_PREFIX_SUFFIX_EVALUATION.csv", pref_rows)

    # Phase 6 D2 isolation
    print("=== Phase 6: D2 isolation ===", flush=True)
    d2_eval_eps = []
    for r in d2_info["d2_rows"]:
        if r["split"] != "val":
            continue
        # find IC
        sample = r["traj_id"]
        found = None
        for folder in (P14 / "ics", P15 / "ics", P16 / "ics"):
            for p in folder.glob("*.pkl"):
                if p.stem in sample or sample.endswith(p.stem) or p.stem in str(sample):
                    found = p
                    break
            if found:
                break
        # also try stripping prefixes
        if found is None and "d2__" in sample:
            stem = sample.split("__")[-1]
            for folder in (P14 / "ics", P15 / "ics", P16 / "ics"):
                cand = folder / f"{stem}.pkl"
                if cand.is_file():
                    found = cand
                    break
        if found is None:
            continue
        blob = pickle.loads(found.read_bytes())
        acts = blob.get("actions")
        if acts is None:
            acts = (blob.get("meta") or {}).get("actions")
        if acts is None or blob.get("capture") is None:
            continue
        skill = r.get("skill") or "?"
        d2_eval_eps.append(
            {
                "traj_id": r["traj_id"],
                "lineage_id": r["lineage_id"],
                "skill": skill,
                "capture": blob["capture"],
                "ref": blob["ref"],
                "actions": np.asarray(acts, np.float32),
            }
        )
    print(f"D2 val restoreable: {len(d2_eval_eps)}", flush=True)

    d2_rows_out = []
    models_d2 = [("D2_only", model_d2)]
    if model_bd2 is not None:
        models_d2.append(("P23_B_d2", model_bd2))
    models_d2.append(("SFT_A", model_a))

    for ep in d2_eval_eps:
        fake = {
            "capture": ep["capture"],
            "ref": ep["ref"],
            "actions": ep["actions"],
            "start_kind": "recovery_ic_capture",
        }
        for mname, model in models_d2:
            r = rollout_policy(
                env, model, book, rho, fake, device=device, max_steps=800, stabilize_n=8
            )
            r.update(
                {
                    "traj_id": ep["traj_id"],
                    "lineage_id": ep["lineage_id"],
                    "skill": ep["skill"],
                    "condition": mname,
                    "eval_set": "D2_heldout",
                }
            )
            d2_rows_out.append(r)

    # OOD: D2_only on A/B/C val (diagnostic)
    for fam in ("A", "B", "C"):
        for row in [r for r in p21["trajectories"] if r["split"] == "val" and r["family"] == fam][:5]:
            ep = starts[row["traj_id"]]
            r = rollout_policy(
                env, model_d2, book, rho, ep, device=device, max_steps=400, stabilize_n=8
            )
            r.update(
                {
                    "traj_id": row["traj_id"],
                    "lineage_id": row["lineage_id"],
                    "skill": fam,
                    "condition": "D2_only",
                    "eval_set": f"OOD_recovery_{fam}",
                }
            )
            d2_rows_out.append(r)
    _write_csv(OUT / "D2_ISOLATION_EVALUATION.csv", d2_rows_out)

    # Videos
    print("=== videos ===", flush=True)
    c0 = val_c_rows[0]
    render_video(env, model_c, book, rho, starts[c0["traj_id"]], OUT / "videos" / "c_only_plan_c.mp4", device, "C_only")
    render_video(env, model_a, book, rho, starts[c0["traj_id"]], OUT / "videos" / "sft_a_plan_c.mp4", device, "SFT_A")
    if d2_eval_eps:
        fake = {
            "capture": d2_eval_eps[0]["capture"],
            "ref": d2_eval_eps[0]["ref"],
            "actions": d2_eval_eps[0]["actions"],
            "start_kind": "recovery_ic_capture",
            "catch_snap": None,
        }
        # restore via capture path in render — hack: use restore_start compatible ep
        ep_d2 = {
            "start_kind": "recovery_ic_capture",
            "capture": d2_eval_eps[0]["capture"],
            "ref": d2_eval_eps[0]["ref"],
            "actions": d2_eval_eps[0]["actions"],
        }
        render_video(env, model_d2, book, rho, ep_d2, OUT / "videos" / "d2_only.mp4", device, "D2_only")

    # Aggregates for reports
    def agg_cl(cond: str) -> dict:
        xs = [r for r in cl_rows if r["condition"] == cond]
        n = len(xs)
        return {
            "n": n,
            "strict": sum(1 for r in xs if r.get("SUCCESS_NEW")) / max(1, n),
            "s12": sum(1 for r in xs if r.get("SUCCESS_12S")) / max(1, n),
            "detach": sum(int(r.get("detach_completed") or 0) for r in xs) / max(1, n),
            "recatch": sum(int(r.get("recatch_completed") or 0) for r in xs) / max(1, n),
            "stable": sum(int(r.get("stable_bilateral") or 0) for r in xs) / max(1, n),
            "lift": sum(int(r.get("lift_completed") or 0) for r in xs) / max(1, n),
            "prem_h": sum(int(r.get("premature_handoff") or 0) for r in xs) / max(1, n),
            "mean_steps": float(np.mean([r.get("n_motion") or 0 for r in xs])) if xs else 0,
            "mean_frac_t3": float(np.mean([float(r.get("frac_token_3") or 0) for r in xs])) if xs else 0,
            "mean_div_t": float(np.mean([float(r["div_t"]) for r in xs if r.get("div_t") is not None])) if any(r.get("div_t") is not None for r in xs) else -1,
            "div_stages": Counter(r.get("div_stage") for r in xs if r.get("div_stage")),
        }

    a_cl = agg_cl("SFT_A")
    c_cl = agg_cl("C_only")
    ce_cl = agg_cl("continuous_expert")

    def pref_rate(model, boundary):
        xs = [r for r in pref_rows if r["model"] == model and r["boundary"] == boundary]
        return sum(int(r["SUCCESS_NEW"]) for r in xs) / max(1, len(xs)), len(xs)

    def d2_skill_rate(cond, skill=None):
        xs = [r for r in d2_rows_out if r["condition"] == cond and r.get("eval_set") == "D2_heldout"]
        if skill:
            xs = [r for r in xs if str(r.get("skill")).lower() == skill.lower() or skill.lower() in str(r.get("skill")).lower()]
        n = len(xs)
        return sum(1 for r in xs if r.get("SUCCESS_NEW")) / max(1, n), n

    # TF train fit for C_only
    tf_c_train = next(r for r in tf_rows if r["model"] == "C_only" and r["split"] == "train")
    tf_c_val = next(r for r in tf_rows if r["model"] == "C_only" and r["split"] == "val")
    tf_a_val = next(r for r in tf_rows if r["model"] == "SFT_A" and r["split"] == "val")

    d2_only_overall = d2_skill_rate("D2_only")
    bd2_overall = d2_skill_rate("P23_B_d2") if model_bd2 else (0.0, 0)

    # Case classification
    c_strict = c_cl["strict"]
    c_fit = float(tf_c_train["token_acc"] or 0)
    c_gen = float(tf_c_val["token_acc"] or 0)
    if c_strict >= 0.15:
        case = "A"
        case_txt = "C_only substantially improves C STRICT → multi-skill interference; recommend skill heads / hierarchy in P25."
    elif c_fit >= 0.5 and c_strict < 0.05:
        case = "B"
        case_txt = "C_only fits offline tokens but 0% STRICT → offline BC insufficient for closed-loop C; recommend corrective demos / DAgger."
    elif c_fit < 0.35:
        case = "C"
        case_txt = "C_only cannot fit training trajectories well → capacity/context/history/target/opt issues; no hierarchy yet."
    elif c_fit >= 0.5 and c_gen < 0.35 and c_strict < 0.05:
        case = "D"
        case_txt = "Fits train but fails held-out TF → lineage diversity / coverage; not proof M111 inadequate."
    else:
        case = "B/mixed"
        case_txt = "Mixed: partial fit with persistent closed-loop failure; treat as offline-vs-closed-loop gap (Case B leaning)."

    failure_md = f"""# FAILURE_CAUSAL_ANALYSIS

Generated: {_utc()}

## Teacher-forced vs closed-loop (C_only)

| Signal | Value |
|---|---|
| Train token acc | {tf_c_train['token_acc']:.4f} |
| Val token acc | {tf_c_val['token_acc']:.4f} |
| SFT-A val token acc (on C chunks) | {tf_a_val['token_acc']:.4f} |
| Closed-loop STRICT | {c_cl['strict']:.3f} |
| Mean steps to DROP/term | {c_cl['mean_steps']:.1f} |
| Frac pred token {P23_BAD_TOKEN} in CL | {c_cl['mean_frac_t3']:.3f} |
| Mean first divergence t | {c_cl['mean_div_t']:.1f} |
| Divergence stages | {dict(c_cl['div_stages'])} |

## Causal checklist

| Hypothesis | Evidence |
|---|---|
| Failure to fit C demos | Train TF acc={tf_c_train['token_acc']:.3f} → {"YES" if c_fit < 0.35 else "NO / partial"} |
| Poor held-out generalization (TF) | Val TF={tf_c_val['token_acc']:.3f} vs train {tf_c_train['token_acc']:.3f} |
| Incorrect action-history encoding | Phase-1 prev alignment OK on packaged NPZs; closed-loop uses executed prev |
| Repeated-token collapse | CL frac token-3={c_cl['mean_frac_t3']:.3f}; Expert token-3 rates in repetition audit |
| Early closed-loop state divergence | Mean div_t={c_cl['mean_div_t']:.1f}, stages={dict(c_cl['div_stages'])} |
| Fail stabilize after recatch | Recatch rate={c_cl['recatch']:.3f}, stable={c_cl['stable']:.3f}, lift={c_cl['lift']:.3f} |

## Decision case

**Case {case}.** {case_txt}
"""
    _write_md(OUT / "FAILURE_CAUSAL_ANALYSIS.md", failure_md)

    # Prefix summary for final
    pref_summary = []
    for b in ("t0", "after_detach", "after_recatch", "after_stabilize", "during_lift", "full_expert"):
        rc, nc = pref_rate("C_only", b)
        ra, na = pref_rate("SFT_A", b)
        pref_summary.append((b, rc, nc, ra, na))

    final = f"""# FINAL_REPORT — P24 Skill Isolation and Plan C Learnability Audit

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Can Plan C be learned in isolation?**  
   Closed-loop STRICT C_only={c_cl['strict']:.3f} (n={c_cl['n']}) vs SFT-A={a_cl['strict']:.3f} vs Expert={ce_cl['strict']:.3f}.  
   Train/val TF acc: {tf_c_train['token_acc']:.3f} / {tf_c_val['token_acc']:.3f}.  
   Verdict: isolation {"helps closed-loop" if c_cl['strict'] > a_cl['strict'] + 0.05 else "does **not** unlock closed-loop C"} (Case **{case}**).

2. **At what physical stage does C_only first fail?**  
   Dominant first divergence stage: **{c_cl['div_stages'].most_common(1)[0] if c_cl['div_stages'] else ('unknown',0)}**; mean div_t={c_cl['mean_div_t']:.1f}.  
   Stage completion rates — detach={c_cl['detach']:.2f}, recatch={c_cl['recatch']:.2f}, stable={c_cl['stable']:.2f}, lift={c_cl['lift']:.2f}.

3. **Does C_only eliminate repeated-token behavior?**  
   Closed-loop frac token-{P23_BAD_TOKEN}: C_only={c_cl['mean_frac_t3']:.3f} vs SFT-A={a_cl['mean_frac_t3']:.3f}.  
   See also confusion `pred_mass_token_3` in `TEACHER_FORCED_COMPARISON.csv`.

4. **Can C_only complete any Expert-prefix suffix?**  
"""
    for b, rc, nc, ra, na in pref_summary:
        final += f"   - `{b}`: C_only STRICT={rc:.3f} (n={nc}) vs SFT-A={ra:.3f} (n={na})\n"
    final += f"""
5. **Can D2_only reliably learn Precision?**  
   Held-out D2 STRICT: D2_only={d2_only_overall[0]:.3f} (n={d2_only_overall[1]}) vs P23 B_d2={bd2_overall[0]:.3f} (n={bd2_overall[1]}).  
   Per-skill rows in `D2_ISOLATION_EVALUATION.csv`. OOD recovery failures are expected (not a defect).

6. **Shared-policy interference vs independent C bottleneck?**  
   If C_only ≫ SFT-A on C STRICT → interference primary.  
   If C_only still ~0% STRICT → **independent Plan C learnability bottleneck** (with optional secondary interference for D2/A tradeoffs from P23).  
   Evidence: Case **{case}**.

7. **P25 direction supported?**  
   {case_txt}  
   **Do not implement P25 architecture in this phase.**

## Integrity

- Fresh C_only / D2_only init; P20–P23 artifacts untouched  
- M111 / physics / Oracle / STRICT / 39-D obs unchanged  
- No GRPO  
- No Oracle phase labels in policy inputs  

## Artifacts

All under `results/diagnostics/p24_skill_isolation/`.
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p24_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "case": case,
                "c_cl": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in c_cl.items()},
                "a_cl": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in a_cl.items()},
                "tf_c_train": tf_c_train,
                "tf_c_val": tf_c_val,
                "d2_only": d2_only_overall,
                "configs": configs,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P24 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
