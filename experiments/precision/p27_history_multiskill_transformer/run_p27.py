"""P27 — History-conditioned unified multi-skill Transformer (A1/B1/C/D2).

No skill labels / selector / privileged phase at inference.
M111 frozen. No full-horizon PPO.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
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

from envs.observations.airborne_obs import OBS_DIM_AIR  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.physical_recovery import TABLE_TOP, physical_pack  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import (  # noqa: E402
    build_start_index,
    classify_failure,
)
from experiments.precision.p23_c_d2_sft.run_p23 import resolve_npz_root  # noqa: E402
from models.m111_checkpoint import (  # noqa: E402

    HANDOFF,
    MOTION_K,
    decode_m111_action,
    load_m111,
    load_m111_book,
    save_m111,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import assert_no_target_in_inputs, collate_m111, load_episode_npz  # noqa: E402
from training.m111_sft import eval_token_metrics, supervised_loss_m111, train_m111  # noqa: E402

OUT = ROOT / "results/diagnostics/p27_history_multiskill_transformer"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
P24 = ROOT / "results/diagnostics/p24_skill_isolation"
P25 = ROOT / "results/diagnostics/p25_c_dagger"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

P22_A = P22 / "checkpoints/A_natural/checkpoint_best.pt"
P25_C = P25 / "checkpoints/round2_B_corrective_only/checkpoint_best.pt"
P24_D2 = P24 / "checkpoints/D2_only/checkpoint_best.pt"

# A1 := Plan A recovery, B1 := Plan B recovery (project naming)
SKILL_MAP = {"A": "A1", "B": "B1", "C": "C", "D2": "D2"}
K_VALUES = (1, 8, 32, 64)
EPOCHS = 18
LR = 3e-4
BATCH = 8
SEED = 27
# Matched optimizer budget across K (prior SFT used ~150–300 traj windows).
TARGET_TRAIN_WINDOWS = 10000
TARGET_VAL_WINDOWS = 2500


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


def load_npz_flex(root: Path, rel: str, *, allow_nonzero_prev0: bool = False) -> dict:
    path = root / rel
    blob = np.load(path)
    obs = np.asarray(blob["observations"], np.float32)
    prev = np.asarray(blob["prev_actions"], np.float32)
    tokens = np.asarray(blob["tokens"], np.int64)
    residuals = np.asarray(blob["residuals"], np.float32)
    mask = np.asarray(blob["loss_mask"], np.bool_) if "loss_mask" in blob.files else np.ones(len(tokens), np.bool_)
    if obs.shape[1] != OBS_DIM_AIR:
        raise RuntimeError(f"bad obs dim {path}")
    if not allow_nonzero_prev0 and float(np.max(np.abs(prev[0]))) > 1e-6:
        raise RuntimeError(f"prev[0]!=0 {path}")
    if int(tokens[-1]) != HANDOFF:
        raise RuntimeError(f"no terminal HANDOFF {path}")
    return {
        "observations": obs,
        "prev_actions": prev,
        "tokens": tokens,
        "residuals": residuals,
        "loss_mask": mask,
    }


# ---------------------------------------------------------------------------
# Phase 0 — audit
# ---------------------------------------------------------------------------


def phase0_audit() -> dict:
    print("=== Phase 0: dataset audit ===", flush=True)
    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    p25 = json.loads((P25 / "VERIFIED_CORRECTIONS_MANIFEST.json").read_text(encoding="utf-8"))

    by_fam = Counter(r["family"] for r in p21["trajectories"])
    by_split = Counter((r["family"], r["split"]) for r in p21["trajectories"])
    d2 = [r for r in p23["trajectories"] if r.get("family") == "D2"]
    corr = p25.get("corrections") or []

    # lineage disjoint
    lin = defaultdict(lambda: {"train": set(), "val": set()})
    for r in p21["trajectories"]:
        lin[r["family"]][r["split"]].add(r["lineage_id"])
    for r in d2:
        lin["D2"][r["split"]].add(r["lineage_id"])
    leaks = {sk: len(lin[sk]["train"] & lin[sk]["val"]) for sk in lin}

    # alignment spot-check
    align_ok = 0
    align_n = 0
    for r in p21["trajectories"][:8]:
        ep = load_episode_npz(P21, r["npz_relpath"])
        align_n += 1
        if float(np.max(np.abs(ep["prev_actions"][0]))) <= 1e-6 and int(ep["tokens"][-1]) == HANDOFF:
            align_ok += 1

    # Pre-impact availability: P21 starts at recovery_ic / catch_snap (post-impact)
    preimpact = {
        "A1_B1_C_recovery": "UNAVAILABLE — packaged NPZs start at recovery IC / catch_snap (post-impact). "
        "Expert pkl may contain `snap` root state, but no verified deterministic pre-impact 39-D "
        "observation tape is stored on the SFT NPZ. Do not fabricate frames.",
        "D2_precision": "N/A by design — stable Precision IC starts; no impact history required.",
        "reconstructed_count": 0,
    }

    md = f"""# DATASET_AUDIT

Generated: {_utc()}

## Skill name mapping

| P27 name | Source family | Corpus |
|---|---|---|
| **A1** | Plan A recovery (`family=A`) | P21 full-horizon |
| **B1** | Plan B recovery (`family=B`) | P21 full-horizon |
| **C** | Plan C recovery (`family=C`) | P21 + P25 verified corrective suffixes |
| **D2** | Precision-required (`family=D2`) | P23 curated D2 (not bulk HANDOFF) |

No literal `A1`/`B1` tags in manifests; A1/B1 denote Plan A / Plan B Stage-1 recovery skills.

## Counts

| Skill | Train traj | Val traj | Lineages (approx) |
|---|---:|---:|---:|
| A1 | {by_split[('A','train')]} | {by_split[('A','val')]} | {len(lin['A']['train']|lin['A']['val'])} |
| B1 | {by_split[('B','train')]} | {by_split[('B','val')]} | {len(lin['B']['train']|lin['B']['val'])} |
| C | {by_split[('C','train')]} | {by_split[('C','val')]} | {len(lin['C']['train']|lin['C']['val'])} |
| C corrections | {sum(1 for c in corr if c.get('split')=='train')} | {sum(1 for c in corr if c.get('split')=='val')} | — |
| D2 | {sum(1 for r in d2 if r['split']=='train')} | {sum(1 for r in d2 if r['split']=='val')} | {len(lin['D2']['train']|lin['D2']['val'])} |

Train∩val lineage leaks: {leaks}

## Interface

- Obs: 39-D airborne; prev = **previously executed** 7-D action; targets = M111 token+residual.
- Spot-check prev[0]=0 + terminal HANDOFF: {align_ok}/{align_n}
- Full-horizon from recovery IC (A/B/C) or stable Precision IC (D2).
"""
    _write_md(OUT / "DATASET_AUDIT.md", md)

    hist_md = f"""# HISTORY_AVAILABILITY

Generated: {_utc()}

## Pre-impact history

{preimpact['A1_B1_C_recovery']}

{preimpact['D2_precision']}

Reconstructed pre-impact trajectories in this run: **{preimpact['reconstructed_count']}**.

## Post-impact / Precision history

- A1/B1/C NPZs: full Stage-1 (+ optional Precision) from recovery IC with causal (obs, prev) streams.
- P25 corrective suffixes: mid-recovery snapshots with nonzero prev[0] allowed (intervention history).
- D2: Precision correction tapes from stable Precision ICs.

## Policy context

Causal window K ∈ {list(K_VALUES)}: input (o_{{t-K+1}}, a_{{t-K}}, …, o_t) with prev-action channel; **no future** tokens.
Padding + loss masks for short prefixes.
"""
    _write_md(OUT / "HISTORY_AVAILABILITY.md", hist_md)
    return {
        "p21": p21,
        "p23": p23,
        "p25": p25,
        "d2": d2,
        "corr": corr,
        "preimpact": preimpact,
        "leaks": leaks,
    }


def build_unified_manifest(audit: dict) -> dict:
    rows = []
    # P21 A/B/C
    for r in audit["p21"]["trajectories"]:
        skill = SKILL_MAP[r["family"]]
        rows.append(
            {
                **r,
                "skill": skill,
                "dataset_root": "p21",
                "allow_nonzero_prev0": False,
                "source": f"p21_{r['family']}",
            }
        )
    # P25 corrections (C)
    for c in audit["corr"]:
        rows.append(
            {
                **c,
                "skill": "C",
                "family": "C",
                "dataset_root": "p25",
                "allow_nonzero_prev0": True,
                "source": "p25_correction",
                "npz_relpath": c["npz_relpath"],
            }
        )
        # ensure file accessible under OUT or P25
        src = P25 / c["npz_relpath"]
        if src.is_file():
            dst = OUT / "dataset" / "corr" / Path(c["npz_relpath"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_file():
                shutil.copy2(src, dst)
    # D2 from P23
    for r in audit["d2"]:
        rows.append(
            {
                **r,
                "skill": "D2",
                "dataset_root": "p23",
                "allow_nonzero_prev0": False,
                "source": r.get("source_corpus") or "d2",
            }
        )

    man = {
        "dataset_id": "p27_unified_a1_b1_c_d2",
        "generated_at": _utc(),
        "skill_map": SKILL_MAP,
        "n": len(rows),
        "trajectories": rows,
        "preimpact_available": False,
    }
    (OUT / "UNIFIED_MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def resolve_row_root(row: dict) -> Path:
    if row.get("dataset_root") == "p21":
        return P21
    if row.get("dataset_root") == "p25" or row.get("source") == "p25_correction":
        # prefer copied
        local = OUT / "dataset" / "corr" / Path(row["npz_relpath"]).name
        if local.is_file():
            return local.parent
        return P25
    if row.get("dataset_root") == "p23":
        if (P23 / row["npz_relpath"]).is_file():
            return P23
        return OUT
    return P21


def windows_from_episode(ep: dict, K: int) -> list[dict]:
    """Causal windows of length ≤K. Stride ≈ K/2 so longer K still covers the tape."""
    keys = ("observations", "prev_actions", "tokens", "residuals", "loss_mask")
    n = int(ep["observations"].shape[0])
    if n <= 0:
        return []
    if n <= K:
        return [{k: ep[k] for k in keys}]
    stride = max(1, K // 2)
    starts = list(range(0, n - K + 1, stride))
    # always include a suffix window ending at n
    last = n - K
    if starts[-1] != last:
        starts.append(last)
    return [{k: ep[k][s : s + K] for k in keys} for s in starts]


def _stratified_cap(samples: list[dict], cap: int, *, seed: int) -> list[dict]:
    """Skill-balanced subsample to match training budget across K."""
    if len(samples) <= cap:
        return samples
    by_sk: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_sk[str(s.get("skill") or "UNK")].append(s)
    rng = np.random.default_rng(seed)
    # equal quota per skill present, leftover by residual pool
    skills = sorted(by_sk)
    base = cap // max(1, len(skills))
    picked: list[dict] = []
    leftover: list[dict] = []
    for sk in skills:
        pool = by_sk[sk]
        rng.shuffle(pool)
        take = min(base, len(pool))
        picked.extend(pool[:take])
        leftover.extend(pool[take:])
    need = cap - len(picked)
    if need > 0 and leftover:
        rng.shuffle(leftover)
        picked.extend(leftover[:need])
    rng.shuffle(picked)
    return picked[:cap]


def build_samples_for_K(manifest: dict, *, K: int, split: str) -> list[dict]:
    rows = [r for r in manifest["trajectories"] if r.get("split") == split]
    # skill counts for inverse-sqrt balancing
    sk_c = Counter(r["skill"] for r in rows)
    samples = []
    for row in rows:
        root = resolve_row_root(row)
        rel = row["npz_relpath"]
        if row.get("source") == "p25_correction":
            # local copy uses basename only under dataset/corr
            local = OUT / "dataset" / "corr" / Path(rel).name
            if local.is_file():
                ep = load_npz_flex(local.parent, local.name, allow_nonzero_prev0=True)
            else:
                ep = load_npz_flex(P25, rel, allow_nonzero_prev0=True)
        else:
            ep = load_npz_flex(root, rel, allow_nonzero_prev0=bool(row.get("allow_nonzero_prev0")))
        w_skill = 1.0 / np.sqrt(max(1, sk_c[row["skill"]]))
        # mild downweight repeated corrections
        if row.get("source") == "p25_correction":
            w_skill *= 0.85
        for wi, chunk in enumerate(windows_from_episode(ep, K)):
            samples.append(
                {
                    **chunk,
                    "sample_weight": float(w_skill),
                    "traj_id": row["traj_id"],
                    "skill": row["skill"],
                    "family": row.get("family"),
                    "lineage_id": row.get("lineage_id"),
                    "skill_group": row["skill"],
                    "split": split,
                    "window_index": wi,
                    "K": K,
                }
            )
    cap = TARGET_TRAIN_WINDOWS if split == "train" else TARGET_VAL_WINDOWS
    samples = _stratified_cap(samples, cap, seed=SEED + K + (0 if split == "train" else 1000))
    if samples:
        mean_w = float(np.mean([s["sample_weight"] for s in samples]))
        for s in samples:
            s["sample_weight"] = float(s["sample_weight"] / max(mean_w, 1e-8))
    return samples


def train_K(K: int, train_s: list, val_s: list, book, rho, device) -> Path:
    out_dir = OUT / "checkpoints" / f"K{K}"
    # Temporarily allow collate length up to max(K, batch max) — collate checks WINDOW=160
    assert K <= 160
    cfg = {
        "name": f"unified_K{K}",
        "seed": SEED + K,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
        "device": str(device),
        "window": K,
    }
    print(f"=== train K={K} n_train={len(train_s)} n_val={len(val_s)} ===", flush=True)
    train_m111(
        train_samples=train_s,
        val_samples=val_s if val_s else train_s[: max(1, len(train_s) // 10)],
        cfg=cfg,
        out_dir=out_dir,
        book=book,
        rho=rho,
        init_checkpoint=None,
    )
    return out_dir / "checkpoint_best.pt"


def policy_step_K(model, book, rho, obs_seq, prev_seq, device, K: int):
    obs_in = obs_seq[-K:]
    prev_in = prev_seq[-K:]
    with torch.no_grad():
        logits, resid, _ = model(
            torch.as_tensor(np.stack(obs_in), device=device),
            torch.as_tensor(np.stack(prev_in), device=device),
        )
    tok = int(torch.argmax(logits[-1]).item())
    if tok == HANDOFF:
        return tok, np.zeros(7, np.float32), None
    r = resid[-1].float().cpu().numpy().astype(np.float32)
    a = decode_m111_action(book, tok, r, rho=rho)
    return tok, r, a


def rollout_skill(env, model, book, rho, ep, *, device, K: int, max_steps: int, start_kind: str):
    if start_kind == "d2":
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        ref = ep["ref"]
    else:
        restore_start(env, ep)
        ref = ep["ref"]
    hist = env._hist
    obs_seq: list = []
    prev_seq: list = []
    tokens = []
    n_motion = 0
    for _ in range(max_steps):
        from envs.observations.airborne_obs import observe_airborne

        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K)
        tokens.append(tok)
        if action is None:
            break
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    rls = []
    if tokens:
        cur, n = tokens[0], 1
        for t in tokens[1:]:
            if t == cur:
                n += 1
            else:
                rls.append(n)
                cur, n = t, 1
        rls.append(n)
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "n_motion": n_motion,
        "immediate_handoff": int(n_motion == 0),
        "frac_token_3": sum(1 for t in tokens if t == 3) / max(1, len(tokens)),
        "max_token_run": max(rls) if rls else 0,
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "tokens_head": "|".join(str(t) for t in tokens[:12]),
    }


def load_d2_eval_eps(audit: dict) -> list[dict]:
    import pickle

    eps = []
    for r in audit["d2"]:
        if r["split"] != "val":
            continue
        # try IC restore like P23
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
        acts = blob.get("actions")
        if acts is None:
            acts = (blob.get("meta") or {}).get("actions")
        if acts is None or blob.get("capture") is None:
            continue
        eps.append(
            {
                "traj_id": tid,
                "lineage_id": r["lineage_id"],
                "capture": blob["capture"],
                "ref": blob["ref"],
                "actions": np.asarray(acts, np.float32),
                "skill": "D2",
            }
        )
    return eps


def eval_multiskill(label, model, book, rho, starts, val_abc, d2_eps, device, K: int) -> list[dict]:
    env = make_env()
    rows = []
    # A/B/C with physical stage completion for C; STRICT for all
    for row in val_abc:
        ep = starts[row["traj_id"]]
        skill = SKILL_MAP[row["family"]]
        # Always use K-matched rollout (no TABLE_TOP mid-abort; STRICT at end).
        r = rollout_skill(env, model, book, rho, ep, device=device, K=K, max_steps=800, start_kind="recovery")
        rows.append(
            {
                "condition": label,
                "skill": skill,
                "traj_id": row["traj_id"],
                "lineage_id": row["lineage_id"],
                "K": K,
                "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                "SUCCESS_12S": int(r["SUCCESS_12S"]),
                "n_motion": r["n_motion"],
                "immediate_handoff": r["immediate_handoff"],
                "frac_token_3": r["frac_token_3"],
                "max_token_run": r["max_token_run"],
                "eval_reason": r.get("eval_reason"),
            }
        )
    for ep in d2_eps:
        r = rollout_skill(env, model, book, rho, ep, device=device, K=K, max_steps=800, start_kind="d2")
        rows.append(
            {
                "condition": label,
                "skill": "D2",
                "traj_id": ep["traj_id"],
                "lineage_id": ep["lineage_id"],
                "K": K,
                "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                "SUCCESS_12S": int(r["SUCCESS_12S"]),
                "n_motion": r["n_motion"],
                "immediate_handoff": r["immediate_handoff"],
                "frac_token_3": r["frac_token_3"],
                "max_token_run": r["max_token_run"],
                "eval_reason": r.get("eval_reason"),
            }
        )
    return rows


def summarize_skill(rows: list[dict], skill: str) -> dict:
    xs = [r for r in rows if r["skill"] == skill]
    n = len(xs)
    return {
        "skill": skill,
        "n": n,
        "strict": sum(int(r["SUCCESS_NEW"]) for r in xs) / max(1, n),
        "strict_count": sum(int(r["SUCCESS_NEW"]) for r in xs),
        "s12": sum(int(r["SUCCESS_12S"]) for r in xs) / max(1, n),
        "imm_h": sum(int(r.get("immediate_handoff") or 0) for r in xs) / max(1, n),
        "mean_n_motion": float(np.mean([r["n_motion"] for r in xs])) if xs else 0.0,
    }


def render_video(model, book, rho, ep, path: Path, device, K: int, start_kind="recovery"):
    try:
        import mujoco
        from envs.observations.airborne_obs import observe_airborne
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4
    except Exception as exc:
        print("video skip", exc, flush=True)
        return
    env = make_env()
    if start_kind == "d2":
        restore_capture(env, ep["capture"])
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
    else:
        restore_start(env, ep)
    hist = env._hist
    cam = mjv_camera_from_preset(
        {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames, obs_seq, prev_seq = [], [], []
    for _ in range(220):
        renderer.update_scene(env.sim.data, cam)
        frames.append(renderer.render().copy())
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K)
        if action is None:
            for __ in range(20):
                step_legal(env.sim, np.zeros(7, np.float32))
                renderer.update_scene(env.sim.data, cam)
                frames.append(renderer.render().copy())
            break
        step_legal(env.sim, action)
        env._prev = action
    path.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, path, fps=20)
    print("video", path.name, len(frames), flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P27 device={device} ===", flush=True)

    book, rho, _ = load_m111_book(M111_META)
    audit = phase0_audit()
    manifest = build_unified_manifest(audit)

    configs = {
        "skills": ["A1", "B1", "C", "D2"],
        "K_values": list(K_VALUES),
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH,
        "seed": SEED,
        "architecture": "shared M111 Transformer (d=64, 2 layers, 112-way); no skill head",
        "sampling": "inverse-sqrt skill frequency; corrections mildly downweighted; "
        f"stratified cap train≤{TARGET_TRAIN_WINDOWS} val≤{TARGET_VAL_WINDOWS} for matched K budgets",
        "preimpact": "unavailable — post-impact / Precision history only",
        "baselines": {
            "P22_SFT_A": str(P22_A.relative_to(ROOT)).replace("\\", "/"),
            "P25_C": str(P25_C.relative_to(ROOT)).replace("\\", "/"),
            "P24_D2": str(P24_D2.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (OUT / "MULTISKILL_TRAINING_CONFIGS.yaml").write_text(
        yaml.safe_dump(configs, sort_keys=False), encoding="utf-8"
    )

    # Phase 3 — train K ablations
    tf_rows = []
    ckpts = {}
    for K in K_VALUES:
        train_s = build_samples_for_K(manifest, K=K, split="train")
        val_s = build_samples_for_K(manifest, K=K, split="val")
        ckpt = train_K(K, train_s, val_s, book, rho, device)
        ckpts[K] = ckpt
        model, _, _ = load_m111(ckpt, device)
        model.eval()
        for split_name, samp in (("train", train_s), ("val", val_s)):
            # overall
            m = eval_token_metrics(model, samp, device, max_batches=40)
            tf_rows.append(
                {
                    "K": K,
                    "split": split_name,
                    "skill": "ALL",
                    "token_acc": m.get("token_acc"),
                    "motion_token_acc": m.get("motion_token_acc"),
                    "n_steps": m.get("n_steps"),
                    "n_windows": len(samp),
                }
            )
            # per skill
            for sk in ("A1", "B1", "C", "D2"):
                sub = [s for s in samp if s.get("skill") == sk]
                if not sub:
                    continue
                ms = eval_token_metrics(model, sub, device, max_batches=20)
                tf_rows.append(
                    {
                        "K": K,
                        "split": split_name,
                        "skill": sk,
                        "token_acc": ms.get("token_acc"),
                        "motion_token_acc": ms.get("motion_token_acc"),
                        "n_steps": ms.get("n_steps"),
                        "n_windows": len(sub),
                    }
                )
        # free
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(OUT / "SKILLWISE_TF_ACCURACY.csv", tf_rows)

    # Pick best K by val motion acc overall (not test STRICT)
    val_all = [r for r in tf_rows if r["split"] == "val" and r["skill"] == "ALL"]
    best_K = max(val_all, key=lambda r: float(r["token_acc"] or 0))["K"]
    print(f"best K by val TF = {best_K}", flush=True)

    # Phase 4 — pre-impact ablation (unavailable)
    pre_rows = [
        {
            "K": best_K,
            "setting": "post_impact_only",
            "available": 1,
            "note": "Primary corpus; packaged recovery IC / Precision IC starts",
            "C_strict": None,
            "A1_strict": None,
            "B1_strict": None,
            "D2_strict": None,
        },
        {
            "K": best_K,
            "setting": "full_preimpact_plus_post",
            "available": 0,
            "note": "SKIPPED — no verified pre-impact 39-D tapes; refusing fabrication",
            "C_strict": None,
            "A1_strict": None,
            "B1_strict": None,
            "D2_strict": None,
        },
    ]
    _write_csv(OUT / "PREIMPACT_ABLATION.csv", pre_rows)
    _write_csv(
        OUT / "HISTORY_ABLATION.csv",
        [
            {
                "K": K,
                "val_token_acc": next(
                    float(r["token_acc"]) for r in tf_rows if r["K"] == K and r["split"] == "val" and r["skill"] == "ALL"
                ),
                "n_train_windows": next(
                    int(r["n_windows"]) for r in tf_rows if r["K"] == K and r["split"] == "train" and r["skill"] == "ALL"
                ),
            }
            for K in K_VALUES
        ],
    )

    # Phase 5 — closed-loop
    print("=== Phase 5: closed-loop multi-skill eval ===", flush=True)
    starts = build_start_index(audit["p21"])
    val_abc = [r for r in audit["p21"]["trajectories"] if r["split"] == "val"]
    d2_eps = load_d2_eval_eps(audit)
    print(f"D2 val restoreable={len(d2_eps)}", flush=True)

    all_cl = []
    # Baselines
    for name, path, Kref in (
        ("P22_SFT_A", P22_A, 160),
        ("P25_C_best", P25_C, 160),
        ("P24_D2_only", P24_D2, 160),
    ):
        if not path.is_file():
            continue
        m, _, _ = load_m111(path, device)
        m.eval()
        # evaluate each baseline on all skills with its native context (160)
        rows = eval_multiskill(name, m, book, rho, starts, val_abc, d2_eps, device, K=min(Kref, 160))
        all_cl.extend(rows)

    # Unified K models
    for K, ckpt in ckpts.items():
        m, _, _ = load_m111(ckpt, device)
        m.eval()
        rows = eval_multiskill(f"unified_K{K}", m, book, rho, starts, val_abc, d2_eps, device, K=K)
        all_cl.extend(rows)

    _write_csv(OUT / "SKILLWISE_STRICT_RAW.csv", all_cl)

    # Aggregate STRICT table
    strict_rows = []
    conditions = sorted({r["condition"] for r in all_cl})
    for cond in conditions:
        for sk in ("A1", "B1", "C", "D2"):
            s = summarize_skill([r for r in all_cl if r["condition"] == cond], sk)
            strict_rows.append({"condition": cond, **s})
    _write_csv(OUT / "SKILLWISE_STRICT.csv", strict_rows)

    # Fill preimpact ablation with best_K unified numbers (post-impact only)
    best_rows = [r for r in all_cl if r["condition"] == f"unified_K{best_K}"]
    for pr in pre_rows:
        if pr["setting"] == "post_impact_only":
            for sk, key in (("C", "C_strict"), ("A1", "A1_strict"), ("B1", "B1_strict"), ("D2", "D2_strict")):
                pr[key] = summarize_skill(best_rows, sk)["strict"]
    _write_csv(OUT / "PREIMPACT_ABLATION.csv", pre_rows)

    # Videos
    best_model, _, _ = load_m111(ckpts[best_K], device)
    best_model.eval()
    for fam, tag in (("A", "A1"), ("B", "B1"), ("C", "C")):
        row = next(r for r in val_abc if r["family"] == fam)
        render_video(
            best_model, book, rho, starts[row["traj_id"]], OUT / "videos" / f"unified_K{best_K}_{tag}.mp4", device, best_K
        )
    if d2_eps:
        render_video(
            best_model, book, rho, d2_eps[0], OUT / "videos" / f"unified_K{best_K}_D2.mp4", device, best_K, start_kind="d2"
        )

    # Decide Phase 6
    def rate(cond, sk):
        return next(r for r in strict_rows if r["condition"] == cond and r["skill"] == sk)

    u = {sk: rate(f"unified_K{best_K}", sk) for sk in ("A1", "B1", "C", "D2")}
    p22_a = rate("P22_SFT_A", "A1") if any(r["condition"] == "P22_SFT_A" for r in strict_rows) else None
    p25_c = rate("P25_C_best", "C") if any(r["condition"] == "P25_C_best" for r in strict_rows) else None
    p24_d2 = rate("P24_D2_only", "D2") if any(r["condition"] == "P24_D2_only" for r in strict_rows) else None

    preserve_ab = u["A1"]["strict"] >= 0.5 * (p22_a["strict"] if p22_a else 0.3) and u["B1"]["strict"] >= 0.0
    preserve_d2 = u["D2"]["strict"] >= 0.5 * (p24_d2["strict"] if p24_d2 else 0.5)
    do_phase6 = preserve_ab and preserve_d2 and u["C"]["strict"] < 0.5

    phase6_note = "skipped"
    if do_phase6 and (P25 / "VERIFIED_CORRECTIONS_MANIFEST.json").is_file():
        print("=== Phase 6: light C corrective FT with A/B/D2 replay ===", flush=True)
        # Fine-tune best_K from unified ckpt with upweighted C corrections + replay
        train_s = build_samples_for_K(manifest, K=best_K, split="train")
        for s in train_s:
            if s.get("skill") == "C" and "corr" in str(s.get("traj_id", "")):
                s["sample_weight"] *= 2.0
            elif s.get("skill") in ("A1", "B1", "D2"):
                s["sample_weight"] *= 1.25  # replay retention
        mean_w = float(np.mean([s["sample_weight"] for s in train_s]))
        for s in train_s:
            s["sample_weight"] /= max(mean_w, 1e-8)
        val_s = build_samples_for_K(manifest, K=best_K, split="val")
        out6 = OUT / "checkpoints" / f"K{best_K}_c_corrective_ft"
        train_m111(
            train_samples=train_s,
            val_samples=val_s,
            cfg={
                "name": f"K{best_K}_c_ft",
                "seed": SEED,
                "epochs": 8,
                "batch_size": BATCH,
                "lr": 1e-4,
                "device": str(device),
            },
            out_dir=out6,
            book=book,
            rho=rho,
            init_checkpoint=ckpts[best_K],
        )
        m6, _, _ = load_m111(out6 / "checkpoint_best.pt", device)
        m6.eval()
        rows6 = eval_multiskill(f"unified_K{best_K}_Cft", m6, book, rho, starts, val_abc, d2_eps, device, K=best_K)
        all_cl.extend(rows6)
        for sk in ("A1", "B1", "C", "D2"):
            s = summarize_skill(rows6, sk)
            strict_rows.append({"condition": f"unified_K{best_K}_Cft", **s})
        _write_csv(OUT / "SKILLWISE_STRICT.csv", strict_rows)
        phase6_note = f"ran; C STRICT={summarize_skill(rows6,'C')['strict_count']}/{summarize_skill(rows6,'C')['n']}"
    else:
        phase6_note = f"skipped (preserve_ab={preserve_ab}, preserve_d2={preserve_d2}, C_strict={u['C']['strict']:.3f})"

    # Failure attribution
    fail_md = f"""# FAILURE_ATTRIBUTION

Generated: {_utc()}

## Best context length

K* = **{best_K}** by validation token accuracy (not test STRICT).

## Cross-skill interference signals

| Skill | Unified K{best_K} STRICT | Specialist reference |
|---|---:|---|
| A1 | {u['A1']['strict_count']}/{u['A1']['n']} | P22 SFT-A {p22_a['strict_count'] if p22_a else '?'}/{p22_a['n'] if p22_a else '?'} |
| B1 | {u['B1']['strict_count']}/{u['B1']['n']} | P22 SFT-A B slice (see raw) |
| C | {u['C']['strict_count']}/{u['C']['n']} | P25 {p25_c['strict_count'] if p25_c else '?'}/{p25_c['n'] if p25_c else '?'} |
| D2 | {u['D2']['strict_count']}/{u['D2']['n']} | P24 D2_only {p24_d2['strict_count'] if p24_d2 else '?'}/{p24_d2['n'] if p24_d2 else '?'} |

Interference hypothesized when unified ≪ specialist on a skill while another skill improves.

## Pre-impact

Pre-impact history **not available** without fabrication — cannot attribute C failures to missing pre-impact frames from this ablation.

## Phase 6

{phase6_note}
"""
    _write_md(OUT / "FAILURE_ATTRIBUTION.md", fail_md)

    exceeds_p25 = u["C"]["strict_count"] > (p25_c["strict_count"] if p25_c else 6)
    # ready for conservative RL?
    ready_rl = (
        u["A1"]["strict"] >= 0.25
        and u["D2"]["strict"] >= 0.5
        and u["C"]["strict_count"] >= (p25_c["strict_count"] if p25_c else 6)
    )

    final = f"""# FINAL_REPORT — P27 History-Conditioned Unified Multi-Skill Transformer

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Can one Transformer learn A1/B1/C/D2 without explicit skill labels?**  
   Unified K={best_K}: A1={u['A1']['strict_count']}/{u['A1']['n']}, B1={u['B1']['strict_count']}/{u['B1']['n']}, C={u['C']['strict_count']}/{u['C']['n']}, D2={u['D2']['strict_count']}/{u['D2']['n']}.  
   See `SKILLWISE_STRICT.csv` / TF acc in `SKILLWISE_TF_ACCURACY.csv`.

2. **Does history improve C closed-loop STRICT?**  
   Compare unified_K1 vs K8/32/64 C rows in `SKILLWISE_STRICT.csv` and val TF vs K in `HISTORY_ABLATION.csv`.

3. **Does pre-impact history add value beyond post-impact?**  
   **Not tested — unavailable.** No verified pre-impact 39-D tapes; fabrication refused (`HISTORY_AVAILABILITY.md`).

4. **Does multi-skill training preserve A1/B1 and D2?**  
   A1/D2 vs specialists above; B1 n_val is small (4). Details in FAILURE_ATTRIBUTION.

5. **Does C exceed P25’s 6/19?**  
   Unified C={u['C']['strict_count']}/{u['C']['n']} vs P25={p25_c['strict_count'] if p25_c else 6}/{p25_c['n'] if p25_c else 19}.  
   Exceeds? **{"YES" if exceeds_p25 else "NO"}**.

6. **Cross-skill interference cause?**  
   Shared capacity + imbalanced supervision (C longest / most tokens; B scarce). See skill-wise STRICT gaps vs specialists.

7. **Ready for conservative RL refinement?**  
   **{"YES — cautiously" if ready_rl else "NOT YET"}** — require non-collapsed A1/D2 and C ≥ P25 before RL. No PPO in P27. Phase6: {phase6_note}

## Integrity

- No skill IDs / selector / privileged phase labels at inference  
- M111 / physics / STRICT unchanged  
- Pre-impact not fabricated  
- Aggregate STRICT not used alone (skill-wise tables required)  

## Artifacts

`results/diagnostics/p27_history_multiskill_transformer/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p27_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "best_K": best_K,
                "unified": u,
                "exceeds_p25": exceeds_p25,
                "ready_rl": ready_rl,
                "phase6": phase6_note,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P27 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
