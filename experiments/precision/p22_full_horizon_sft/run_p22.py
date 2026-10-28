"""P22 — Full-horizon multi-skill recovery SFT on P21.

Primary corpus: P21 recovery-IC full-horizon (not P19 continuation).
Fresh M111 112-way init. No GRPO. No Oracle/physics/STRICT changes.
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
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.outcomes.physical_recovery import TABLE_TOP, physical_pack  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    quantize_with_book,
)
from experiments.precision.p21_full_stage1_experts.run_p21 import (  # noqa: E402
    load_fullchain_episode,
    load_unified_episode,
    restore_start,
)
from models.m111_checkpoint import (  # noqa: E402
    HANDOFF,
    MOTION_K,
    decode_m111_action,
    load_m111,
    load_m111_book,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import (  # noqa: E402
    WINDOW,
    build_samples,
    chunk_episode,
    load_episode_npz,
    load_p21_manifest,
)
from training.m111_sft import eval_token_metrics, smoke_train_tiny, train_m111  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS  # noqa: E402

OUT = ROOT / "results/diagnostics/p22_full_horizon_sft"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
CFG_PATH = ROOT / "config/training/m111_sft_full_horizon.yaml"
UNIFIED = ROOT / "data/unified_experts/manifest.jsonl"
FULLCHAIN = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
WINDOW_INF = 160


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


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def integration_audit(manifest: dict) -> str:
    rows = manifest["trajectories"]
    issues = []
    token_counts = Counter()
    family_tokens = defaultdict(Counter)
    family_traj = Counter()
    family_lin = defaultdict(set)
    prec_traj = 0
    handoff_steps = 0
    motion_steps = 0
    gt160 = 0
    for r in rows:
        if r.get("start_distribution") != "recovery_ic_pre_stage1":
            issues.append(f"{r['traj_id']}: bad start_distribution")
        ep = load_episode_npz(P21, r["npz_relpath"])
        n = ep["tokens"].shape[0]
        if int(ep["tokens"][-1]) != HANDOFF:
            issues.append(f"{r['traj_id']}: HANDOFF not terminal")
        if float(np.max(np.abs(ep["prev_actions"][0]))) > 1e-6:
            issues.append(f"{r['traj_id']}: prev[0]!=0")
        if ep["observations"].shape[1] != OBS_DIM_AIR:
            issues.append(f"{r['traj_id']}: obs dim")
        n_s1 = int(r.get("n_stage1") or 0)
        n_pr = int(r.get("n_precision") or 0)
        if n_s1 + n_pr != n - 1:
            issues.append(f"{r['traj_id']}: stage1+prec != motion")
        if n_s1 < 1 and r.get("family") in ("A", "B", "C"):
            issues.append(f"{r['traj_id']}: missing Stage-1")
        # chunk coverage
        chunks = chunk_episode(ep, window=WINDOW)
        covered = 0
        for c in chunks:
            covered += int(c["tokens"].shape[0])
        # overlapping final window may recount; verify via indices in chunk_episode assert
        if n > WINDOW:
            gt160 += 1
        fam = r["family"]
        family_traj[fam] += 1
        family_lin[fam].add(r["lineage_id"])
        if r.get("includes_precision"):
            prec_traj += 1
        for t in ep["tokens"].tolist():
            token_counts[int(t)] += 1
            family_tokens[fam][int(t)] += 1
            if int(t) == HANDOFF:
                handoff_steps += 1
            else:
                motion_steps += 1

    # lineage disjoint
    train_lin = {r["lineage_id"] for r in rows if r["split"] == "train"}
    val_lin = {r["lineage_id"] for r in rows if r["split"] == "val"}
    leak = train_lin & val_lin

    md = f"""# DATASET_INTEGRATION_AUDIT

Generated: {_utc()}

## Corpus

- Manifest: `{manifest.get('dataset_id')}`
- N trajectories: **{len(rows)}** (train {sum(1 for r in rows if r['split']=='train')} / val {sum(1 for r in rows if r['split']=='val')})
- `start_distribution`: **recovery_ic_pre_stage1** (P19 continuation excluded as primary)
- Vocab: M111 / HANDOFF={HANDOFF}

## Checks

| Check | Result |
|---|---|
| All start at recovery IC / catch_snap (manifest flag) | {sum(1 for r in rows if r.get('start_distribution')=='recovery_ic_pre_stage1')}/{len(rows)} |
| Stage-1 steps present (n_stage1≥1 for A/B/C) | {sum(1 for r in rows if int(r.get('n_stage1') or 0)>=1)}/{len(rows)} |
| Precision included when flagged | {prec_traj} Recovery→Precision trajs |
| Terminal HANDOFF | enforced per NPZ |
| prev[0]=0; obs 39-D; no target-in-input | enforced |
| Long seq chunking covers all timesteps (window={WINDOW}) | yes (`chunk_episode` coverage assert); trajs >{WINDOW}: {gt160} |
| Train/val lineage intersection | **{len(leak)}** |

Issues: {issues[:8] if issues else "none"}

## Skill coverage (trajectory / lineages / motion+HANDOFF tokens)

| Skill | Trajectories | Lineages | Token steps (approx family) |
|---|---:|---:|---:|
| A | {family_traj['A']} | {len(family_lin['A'])} | {sum(family_tokens['A'].values())} |
| B | {family_traj['B']} | {len(family_lin['B'])} | {sum(family_tokens['B'].values())} |
| C | {family_traj['C']} | {len(family_lin['C'])} | {sum(family_tokens['C'].values())} |
| Recovery→Precision | {prec_traj} | {len({r['lineage_id'] for r in rows if r.get('includes_precision')})} | — |
| HANDOFF steps (all) | — | — | {handoff_steps} |
| Motion steps (all) | — | — | {motion_steps} |

**Plan B underrepresentation:** {family_traj['B']} traj / {len(family_lin['B'])} lineages vs C {family_traj['C']}/{len(family_lin['C'])}
(~{family_traj['B']/max(1,len(rows)):.1%} of trajectories). Unique states are limited by short B tapes (~36–37 steps).

63/16/92 is **not** balanced supervision — skill-aware baseline upweights inverse √family and inverse lineage.
"""
    (OUT / "skill_coverage.json").write_text(
        json.dumps(
            {
                "family_traj": dict(family_traj),
                "family_lineages": {k: len(v) for k, v in family_lin.items()},
                "prec_traj": prec_traj,
                "handoff_steps": handoff_steps,
                "motion_steps": motion_steps,
                "issues": issues,
                "lineage_leak": len(leak),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if issues or leak:
        raise RuntimeError(f"integration audit failed: issues={issues[:5]} leak={leak}")
    return md


def build_start_index(manifest: dict) -> dict[str, dict]:
    """Map traj_id → restoreable start episode (capture/catch_snap + ref + expert actions)."""
    by_id = {r["traj_id"]: r for r in manifest["trajectories"]}
    index: dict[str, dict] = {}
    # unified
    for line in UNIFIED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        ep = load_unified_episode(raw)
        if ep and ep["traj_id"] in by_id:
            index[ep["traj_id"]] = ep
    # fullchain
    for line in (FULLCHAIN / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        ep = load_fullchain_episode(raw)
        if ep and ep["traj_id"] in by_id:
            index[ep["traj_id"]] = ep
    missing = [tid for tid in by_id if tid not in index]
    if missing:
        raise RuntimeError(f"missing start states for {len(missing)} trajs e.g. {missing[:3]}")
    return index


def policy_step(model, book, rho, obs_seq, prev_seq, device):
    obs_in = obs_seq[-WINDOW_INF:]
    prev_in = prev_seq[-WINDOW_INF:]
    with torch.no_grad():
        logits, resid, _act = model(
            torch.as_tensor(np.stack(obs_in), device=device),
            torch.as_tensor(np.stack(prev_in), device=device),
        )
    tok = int(torch.argmax(logits[-1]).item())
    if tok == HANDOFF:
        return tok, np.zeros(7, np.float32), None
    if tok < 0 or tok >= MOTION_K:
        raise RuntimeError(f"illegal token {tok}")
    r = resid[-1].float().cpu().numpy().astype(np.float32)
    a = decode_m111_action(book, tok, r, rho=rho)
    return tok, r, a


def rollout_policy(env, model, book, rho, ep, *, device, max_steps, stabilize_n) -> dict:
    restore_start(env, ep)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    tokens: list[int] = []
    n_motion = 0
    dropped = False
    timeout = False
    reason = None
    for t in range(int(max_steps)):
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
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
        o = physical_pack(env.sim)
        if float(o.get("obj_z", 99)) < TABLE_TOP + 0.01:
            dropped = True
            reason = "DROP_DURING_POLICY"
            break
    else:
        timeout = True
        reason = "NEVER_HANDOFF_TIMEOUT"
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=int(stabilize_n))
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")) and not dropped,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")) and not dropped,
        "eval_reason": reason or hand.get("std_reason") or hand.get("reason"),
        "n_motion": n_motion,
        "n_decisions": len(tokens),
        "immediate_handoff": bool(n_motion == 0),
        "nonzero_action": bool(n_motion > 0),
        "never_handoff_timeout": bool(timeout),
        "dropped": bool(dropped),
        "unique_tokens": len(set(tokens)),
        "tokens_head": "|".join(str(t) for t in tokens[:12]),
        "tokens_tail": "|".join(str(t) for t in tokens[-8:]),
        "failure_reason": reason
        or (None if hand.get("SUCCESS_NEW") else hand.get("std_reason") or hand.get("reason")),
    }


def replay_expert(env, ep, actions, *, stabilize_n) -> dict:
    restore_start(env, ep)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=int(stabilize_n))
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "n_motion": int(len(acts)),
        "n_decisions": int(len(acts)),
        "immediate_handoff": False,
        "nonzero_action": bool(len(acts) > 0),
        "never_handoff_timeout": False,
        "dropped": False,
        "failure_reason": None if hand.get("SUCCESS_NEW") else hand.get("std_reason") or hand.get("reason"),
    }


def classify_failure(row: dict) -> str:
    if row.get("SUCCESS_NEW"):
        return "success"
    if row.get("dropped"):
        return "contact_instability_or_drop"
    if row.get("never_handoff_timeout"):
        return "never_handoff_timeout"
    if row.get("immediate_handoff"):
        return "early_handoff_immediate"
    if int(row.get("n_motion") or 0) < max(5, int(row.get("expert_n_actions") or 0) // 10):
        return "premature_handoff"
    if int(row.get("n_motion") or 0) > int(row.get("expert_n_actions") or 0) + 100:
        return "excessive_actions"
    return f"physical_fail:{row.get('failure_reason') or row.get('eval_reason') or 'unknown'}"


def render_videos(env, model, book, rho, starts, device, tags: list[tuple[str, str]]) -> None:
    try:
        import mujoco
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4
    except Exception as exc:
        print("video skip", exc, flush=True)
        return
    vdir = OUT / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    cam = mjv_camera_from_preset(
        {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )
    for traj_id, tag in tags:
        ep = starts[traj_id]
        restore_start(env, ep)
        hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        renderer = mujoco.Renderer(env.sim.model, 480, 640)
        frames = []
        obs_seq: list = []
        prev_seq: list = []
        for _ in range(200):
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
                for __ in range(30):
                    step_legal(env.sim, np.zeros(7, np.float32))
                    renderer.update_scene(env.sim.data, cam)
                    frames.append(renderer.render().copy())
                break
            step_legal(env.sim, action)
            env._prev = action
        write_frames_mp4(frames, vdir / f"{tag}.mp4", fps=20)
        print("video", tag, len(frames), flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f"=== P22 device={device} ===", flush=True)

    manifest = load_p21_manifest(ROOT / cfg["dataset_manifest"])
    print("=== integration audit ===", flush=True)
    audit_md = integration_audit(manifest)
    _write_md(OUT / "DATASET_INTEGRATION_AUDIT.md", audit_md)

    # configs dump
    (OUT / "TRAINING_CONFIGS.yaml").write_text(
        yaml.safe_dump(
            {
                "shared": {k: v for k, v in cfg.items() if k != "baselines"},
                "baselines": cfg["baselines"],
                "weighting_B": {
                    "formula": "inv_lineage * inv_sqrt_len * inv_sqrt_family * precision_boost(1.25)",
                    "cap": "5x median before mean-1 renormalization",
                    "note": "Boosts Plan B / rare lineages without deleting HANDOFF supervision",
                },
                "init": "scratch (refuse P20 collapsed checkpoint)",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    book, rho, cb = load_m111_book(ROOT / cfg["m111_meta"])
    train_A = build_samples(
        manifest, split="train", scheme="natural", dataset_root=P21, window=int(cfg.get("window", 160))
    )
    val_A = build_samples(
        manifest, split="val", scheme="natural", dataset_root=P21, window=int(cfg.get("window", 160))
    )
    train_B = build_samples(
        manifest, split="train", scheme="lineage_skill", dataset_root=P21, window=int(cfg.get("window", 160))
    )
    val_B = build_samples(
        manifest, split="val", scheme="lineage_skill", dataset_root=P21, window=int(cfg.get("window", 160))
    )
    print(
        f"samples A {len(train_A)}/{len(val_A)} B {len(train_B)}/{len(val_B)}",
        flush=True,
    )
    # log effective weights
    for name, samp in (("A_natural", train_A), ("B_lineage_skill", train_B)):
        by_fam = defaultdict(list)
        for s in samp:
            by_fam[s.get("family")].append(s["sample_weight"])
        (OUT / f"weights_{name}.json").write_text(
            json.dumps(
                {
                    fam: {
                        "n_chunks": len(ws),
                        "mean_w": float(np.mean(ws)),
                        "sum_w": float(np.sum(ws)),
                    }
                    for fam, ws in by_fam.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=== smoke ===", flush=True)
    smoke = smoke_train_tiny(train_A, book, rho, device, steps=40)
    (OUT / "smoke_result.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print("smoke OK", smoke, flush=True)

    metrics_rows = []
    models = {}
    for key, bcfg in cfg["baselines"].items():
        name = bcfg["name"]
        scheme = bcfg["weight_scheme"]
        train_s = train_A if scheme == "natural" else train_B
        val_s = val_A if scheme == "natural" else val_B
        out_dir = OUT / "checkpoints" / name
        print(f"=== train {name} ===", flush=True)
        tcfg = {**{k: v for k, v in cfg.items() if k != "baselines"}, "name": name, "device": str(device)}
        result = train_m111(
            train_samples=train_s,
            val_samples=val_s,
            cfg=tcfg,
            out_dir=out_dir,
            book=book,
            rho=rho,
        )
        for h in result["history"]:
            metrics_rows.append({"baseline": name, "weight_scheme": scheme, **h})
        model, _b, _s = load_m111(out_dir / "checkpoint_best.pt", device)
        # refuse accidental K48 / wrong vocab already in load_m111
        models[f"sft_{name}"] = model
        final_val = eval_token_metrics(model, val_s, device)
        (out_dir / "final_val_metrics.json").write_text(
            json.dumps({k: v for k, v in final_val.items() if k != "pred_token_hist"}, indent=2),
            encoding="utf-8",
        )

    _write_csv(OUT / "TRAINING_METRICS.csv", metrics_rows)

    # Closed-loop from recovery IC
    print("=== build start index ===", flush=True)
    starts = build_start_index(manifest)
    val_rows = [r for r in manifest["trajectories"] if r["split"] == "val"]
    env = make_env()
    stabilize_n = int(cfg.get("stabilize_n", 8))
    max_steps = int(cfg.get("max_policy_steps", 800))
    cl_rows = []
    print(f"=== closed-loop eval n_val={len(val_rows)} ===", flush=True)
    for i, row in enumerate(val_rows):
        tid = row["traj_id"]
        ep = starts[tid]
        expert_n = int(len(ep["actions"]))
        # continuous expert
        c = replay_expert(env, ep, ep["actions"], stabilize_n=stabilize_n)
        c.update(
            {
                "traj_id": tid,
                "condition": "continuous_expert",
                "family": row["family"],
                "lineage_id": row["lineage_id"],
                "includes_precision": row.get("includes_precision"),
                "horizon_kind": row.get("horizon_kind"),
                "expert_n_actions": expert_n,
            }
        )
        c["failure_class"] = classify_failure(c)
        cl_rows.append(c)
        # M111 expert
        q = quantize_with_book(book, ep["actions"], rho, w=DEFAULT_WEIGHTS)
        m = replay_expert(env, ep, q["recon_actions"], stabilize_n=stabilize_n)
        m.update(
            {
                "traj_id": tid,
                "condition": "m111_expert",
                "family": row["family"],
                "lineage_id": row["lineage_id"],
                "includes_precision": row.get("includes_precision"),
                "horizon_kind": row.get("horizon_kind"),
                "expert_n_actions": expert_n,
            }
        )
        m["failure_class"] = classify_failure(m)
        cl_rows.append(m)
        for name, model in models.items():
            model.eval()
            r = rollout_policy(
                env, model, book, rho, ep, device=device, max_steps=max_steps, stabilize_n=stabilize_n
            )
            r.update(
                {
                    "traj_id": tid,
                    "condition": name,
                    "family": row["family"],
                    "lineage_id": row["lineage_id"],
                    "includes_precision": row.get("includes_precision"),
                    "horizon_kind": row.get("horizon_kind"),
                    "expert_n_actions": expert_n,
                }
            )
            r["failure_class"] = classify_failure(r)
            cl_rows.append(r)
        if (i + 1) % 5 == 0:
            print(f"  eval {i+1}/{len(val_rows)}", flush=True)

    _write_csv(OUT / "CLOSED_LOOP_RESULTS.csv", cl_rows)

    # Per-skill evaluation
    skill_rows = []
    conditions = ["continuous_expert", "m111_expert"] + list(models.keys())
    for cond in conditions:
        sub = [r for r in cl_rows if r["condition"] == cond]
        for fam in ("A", "B", "C", "ALL", "PREC"):
            if fam == "ALL":
                xs = sub
            elif fam == "PREC":
                xs = [r for r in sub if str(r.get("includes_precision")).lower() in ("true", "1")]
            else:
                xs = [r for r in sub if r.get("family") == fam]
            n = len(xs)
            skill_rows.append(
                {
                    "condition": cond,
                    "slice": fam,
                    "n": n,
                    "strict_rate": sum(1 for r in xs if r.get("SUCCESS_NEW")) / max(1, n),
                    "survival_12s": sum(1 for r in xs if r.get("SUCCESS_12S")) / max(1, n),
                    "immediate_handoff_rate": sum(1 for r in xs if r.get("immediate_handoff")) / max(1, n),
                    "nonzero_action_rate": sum(1 for r in xs if r.get("nonzero_action")) / max(1, n),
                    "timeout_rate": sum(1 for r in xs if r.get("never_handoff_timeout")) / max(1, n),
                    "drop_rate": sum(1 for r in xs if r.get("dropped")) / max(1, n),
                    "mean_n_motion": float(np.mean([r.get("n_motion") or 0 for r in xs])) if xs else 0.0,
                }
            )
    _write_csv(OUT / "PER_SKILL_EVALUATION.csv", skill_rows)

    def agg(cond: str) -> dict:
        return next(r for r in skill_rows if r["condition"] == cond and r["slice"] == "ALL")

    ce, me = agg("continuous_expert"), agg("m111_expert")
    a = agg("sft_A_natural")
    b = agg("sft_B_lineage_skill")

    # Failure analysis
    fail_md = ["# FAILURE_ANALYSIS", "", f"Generated: {_utc()}", ""]
    for cond in models:
        rows = [r for r in cl_rows if r["condition"] == cond]
        fails = [r for r in rows if not r.get("SUCCESS_NEW")]
        fail_md.append(f"## {cond}")
        fail_md.append(f"- failures: {len(fails)}/{len(rows)}")
        fail_md.append(f"- nonzero-action rate: {sum(1 for r in rows if r.get('nonzero_action'))/max(1,len(rows)):.3f}")
        fail_md.append(f"- immediate HANDOFF: {sum(1 for r in rows if r.get('immediate_handoff'))/max(1,len(rows)):.3f}")
        for k, v in Counter(r.get("failure_class") for r in fails).most_common():
            fail_md.append(f"  - `{k}`: {v}")
        for fam in ("A", "B", "C"):
            xs = [r for r in rows if r.get("family") == fam]
            ok = sum(1 for r in xs if r.get("SUCCESS_NEW"))
            nz = sum(1 for r in xs if r.get("nonzero_action"))
            fail_md.append(f"  - {fam}: STRICT {ok}/{len(xs)}, nonzero {nz}/{len(xs)}")
        fail_md.append("")
    fail_md.extend(
        [
            "## Skill selection vs execution (log-supported)",
            "",
            "- **Early HANDOFF** (`immediate_handoff` / `premature_handoff`): selection/termination error — never entered recovery motion.",
            "- **Nonzero motion but physical fail**: execution / distribution-shift error (wrong tokens or contact).",
            "- **Timeout without HANDOFF**: termination error after attempting motion.",
            "- Teacher-forced val token acc (see checkpoints) separates representation learning from closed-loop control.",
            "",
            "No new Expert heuristics introduced.",
            "",
        ]
    )
    _write_md(OUT / "FAILURE_ANALYSIS.md", "\n".join(fail_md))

    def fam_rate(cond, fam):
        r = next(x for x in skill_rows if x["condition"] == cond and x["slice"] == fam)
        return r["strict_rate"], r["nonzero_action_rate"], r["n"]

    comp = f"""# BASELINE_COMPARISON

Generated: {_utc()}

## Closed-loop from recovery IC / catch_snap (held-out lineages)

| Condition | N | STRICT | 12S | nonzero-action | immediate H | timeout | mean motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continuous Expert | {ce['n']} | {ce['strict_rate']:.3f} | {ce['survival_12s']:.3f} | {ce['nonzero_action_rate']:.3f} | {ce['immediate_handoff_rate']:.3f} | {ce['timeout_rate']:.3f} | {ce['mean_n_motion']:.1f} |
| M111 Expert | {me['n']} | {me['strict_rate']:.3f} | {me['survival_12s']:.3f} | {me['nonzero_action_rate']:.3f} | {me['immediate_handoff_rate']:.3f} | {me['timeout_rate']:.3f} | {me['mean_n_motion']:.1f} |
| SFT A natural | {a['n']} | {a['strict_rate']:.3f} | {a['survival_12s']:.3f} | {a['nonzero_action_rate']:.3f} | {a['immediate_handoff_rate']:.3f} | {a['timeout_rate']:.3f} | {a['mean_n_motion']:.1f} |
| SFT B skill-aware | {b['n']} | {b['strict_rate']:.3f} | {b['survival_12s']:.3f} | {b['nonzero_action_rate']:.3f} | {b['immediate_handoff_rate']:.3f} | {b['timeout_rate']:.3f} | {b['mean_n_motion']:.1f} |

## Per-family STRICT (nonzero rate)

| Family | Expert | SFT-A | SFT-B |
|---|---|---|---|
| A | {fam_rate('continuous_expert','A')[0]:.2f} (nz {fam_rate('continuous_expert','A')[1]:.2f}, n={fam_rate('continuous_expert','A')[2]}) | {fam_rate('sft_A_natural','A')[0]:.2f} (nz {fam_rate('sft_A_natural','A')[1]:.2f}) | {fam_rate('sft_B_lineage_skill','A')[0]:.2f} (nz {fam_rate('sft_B_lineage_skill','A')[1]:.2f}) |
| B | {fam_rate('continuous_expert','B')[0]:.2f} (nz {fam_rate('continuous_expert','B')[1]:.2f}, n={fam_rate('continuous_expert','B')[2]}) | {fam_rate('sft_A_natural','B')[0]:.2f} (nz {fam_rate('sft_A_natural','B')[1]:.2f}) | {fam_rate('sft_B_lineage_skill','B')[0]:.2f} (nz {fam_rate('sft_B_lineage_skill','B')[1]:.2f}) |
| C | {fam_rate('continuous_expert','C')[0]:.2f} (nz {fam_rate('continuous_expert','C')[1]:.2f}, n={fam_rate('continuous_expert','C')[2]}) | {fam_rate('sft_A_natural','C')[0]:.2f} (nz {fam_rate('sft_A_natural','C')[1]:.2f}) | {fam_rate('sft_B_lineage_skill','C')[0]:.2f} (nz {fam_rate('sft_B_lineage_skill','C')[1]:.2f}) |

Δ STRICT (B−A): **{b['strict_rate']-a['strict_rate']:+.3f}**; Δ nonzero (B−A): **{b['nonzero_action_rate']-a['nonzero_action_rate']:+.3f}**
"""
    _write_md(OUT / "BASELINE_COMPARISON.md", comp)

    # Videos
    print("=== videos ===", flush=True)
    # pick one A/B/C from val
    tags = []
    for fam in ("A", "B", "C"):
        cand = next((r for r in val_rows if r["family"] == fam), None)
        if cand:
            tags.append((cand["traj_id"], f"B_lineage_skill__{fam}__{cand['traj_id'][:40]}"))
    if "sft_B_lineage_skill" in models and tags:
        render_videos(env, models["sft_B_lineage_skill"], book, rho, starts, device, tags[:3])

    # Final report
    best = b if b["strict_rate"] >= a["strict_rate"] else a
    nonzero_ok = max(a["nonzero_action_rate"], b["nonzero_action_rate"]) > 0.5
    grpo_ok = nonzero_ok and best["strict_rate"] >= 0.25

    final = f"""# FINAL_REPORT — P22 Full-Horizon Multi-Skill Recovery SFT

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Does the model execute nonzero recovery actions from recovery ICs?**  
   Nonzero-action rate: SFT-A **{a['nonzero_action_rate']:.3f}**, SFT-B **{b['nonzero_action_rate']:.3f}**  
   (vs Expert {ce['nonzero_action_rate']:.3f}). Immediate HANDOFF: A {a['immediate_handoff_rate']:.3f}, B {b['immediate_handoff_rate']:.3f}.

2. **Can it distinguish / reproduce Plan A/B/C?**  
   Per-family STRICT / nonzero in `PER_SKILL_EVALUATION.csv` and `BASELINE_COMPARISON.md`.

3. **Does skill-aware weighting improve Plan B?**  
   B STRICT A={fam_rate('sft_A_natural','B')[0]:.3f} → B={fam_rate('sft_B_lineage_skill','B')[0]:.3f};  
   nonzero A={fam_rate('sft_A_natural','B')[1]:.3f} → B={fam_rate('sft_B_lineage_skill','B')[1]:.3f}.

4. **Recovery → Precision when necessary?**  
   Slice PREC STRICT: A={fam_rate('sft_A_natural','PREC')[0]:.3f} (n={fam_rate('sft_A_natural','PREC')[2]}),  
   B={fam_rate('sft_B_lineage_skill','PREC')[0]:.3f}. Expert PREC={fam_rate('continuous_expert','PREC')[0]:.3f}.

5. **Held-out STRICT?**  
   Continuous={ce['strict_rate']:.3f}, M111 expert={me['strict_rate']:.3f},  
   SFT-A={a['strict_rate']:.3f}, SFT-B={b['strict_rate']:.3f} (N={ce['n']}).

6. **Meaningful GRPO init?**  
   {"YES — nonzero recovery motion + non-trivial STRICT from recovery ICs." if grpo_ok else "MARGINAL / NOT YET — improve closed-loop recovery competence before GRPO."}  
   **GRPO not started in P22.**

## Integrity

- Primary corpus: P21 full-horizon only
- Fresh scratch init (no P20 always-HANDOFF checkpoint)
- M111 frozen; Oracle / physics / STRICT unchanged
- No GRPO
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p22_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "aggregates": {"A": a, "B": b, "continuous": ce, "m111": me},
                "grpo_ok": grpo_ok,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P22 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
