"""P25 — Plan C closed-loop correction via DAgger.

Baseline: P24 C_only. No physics/Oracle corpora/M111/STRICT/GRPO/hierarchy changes.
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
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.catch_stabilization import fast_world_lift_sequence  # noqa: E402
from data_generation.oracles.plan_c.expert_search import lift_and_validate, search_catch  # noqa: E402
from data_generation.oracles.plan_c.future_trajectory import (  # noqa: E402
    hold_action,
    pad_bilateral,
    pose_correct_and_handoff,
)
from envs.observations.airborne_obs import OBS_DIM_AIR, OBS_VERSION, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import (  # noqa: E402
    build_start_index,
    policy_step,
)
from experiments.precision.p23_c_d2_sft.run_p23 import (  # noqa: E402
    contact_flags,
    first_divergence,
    infer_stage_bounds,
    load_unified_by_id,
    record_timeline,
    stage_name,
)
from experiments.precision.p24_skill_isolation.run_p24 import (  # noqa: E402
    expert_prefix_suffix_c,
    physical_stage_completion,
    rollout_policy_timeline_c,
    run_lengths,
)
from models.m111_checkpoint import HANDOFF, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402
from training.m111_sft import eval_token_metrics, train_m111  # noqa: E402
from training.recovery_execution import make_snap, restore_ballistic  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/diagnostics/p25_c_dagger"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P24 = ROOT / "results/diagnostics/p24_skill_isolation"
C_ONLY_CKPT = P24 / "checkpoints/C_only/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

SEED = 25
EPOCHS_FT = 12
LR_FT = 1e-4
BATCH = 8
MAX_ORACLE_ROLLOUTS = 24
N_TRAIN_TRAJ_COLLECT = 24  # subsample train lineages for state collection
MAX_CORR_ATTEMPTS_PER_ROUND = 72
MAX_POLICY_STEPS_COLLECT = 120
P23_BAD_TOKEN = 3
N_DAGGER_ROUNDS = 2


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


def load_p21() -> dict:
    return json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))


def load_npz_flexible(root: Path, relpath: str, *, allow_nonzero_prev0: bool = False) -> dict:
    path = root / relpath
    blob = np.load(path)
    obs = np.asarray(blob["observations"], np.float32)
    prev = np.asarray(blob["prev_actions"], np.float32)
    tokens = np.asarray(blob["tokens"], np.int64)
    residuals = np.asarray(blob["residuals"], np.float32)
    mask = np.asarray(blob["loss_mask"], np.bool_)
    if obs.shape[1] != OBS_DIM_AIR:
        raise RuntimeError(f"{path}: bad obs dim")
    if not allow_nonzero_prev0 and float(np.max(np.abs(prev[0]))) > 1e-6:
        raise RuntimeError(f"{path}: prev[0] must be zero")
    if int(tokens[-1]) != HANDOFF:
        raise RuntimeError(f"{path}: last token must be HANDOFF")
    return {
        "observations": obs,
        "prev_actions": prev,
        "tokens": tokens,
        "residuals": residuals,
        "loss_mask": mask,
    }


def phase_of(o: dict, t: int, bounds: dict | None) -> str:
    if bounds is not None:
        return stage_name(t, bounds)
    if o.get("bilateral"):
        return "bilateral_hold"
    if o.get("any_contact"):
        return "unilateral"
    return "flight_detach"


# ---------------------------------------------------------------------------
# Phase 1 — baseline
# ---------------------------------------------------------------------------


def reproduce_baseline(model, book, rho, starts, val_c, device) -> list[dict]:
    print("=== Phase 1: baseline reproduction ===", flush=True)
    env = make_env()
    unified = load_unified_by_id()
    rows = []
    for i, row in enumerate(val_c):
        tid = row["traj_id"]
        ep = starts[tid]
        if tid in unified:
            ep = {**ep, "n_catch": unified[tid].get("n_catch")}
        expert_tl = record_timeline(env, ep, ep["actions"])
        bounds = infer_stage_bounds(expert_tl, ep.get("n_catch"))
        sft_tl, _ = rollout_policy_timeline_c(env, model, book, rho, ep, device=device, max_steps=400)
        div = first_divergence(expert_tl, sft_tl, bounds)
        phys = physical_stage_completion(
            env, ep, None, mode="policy", model=model, book=book, rho=rho, device=device, max_steps=800
        )
        # Expert reference once
        if i == 0:
            ce = physical_stage_completion(env, ep, ep["actions"], mode="expert")
            assert ce["SUCCESS_NEW"], "Expert must succeed without TABLE_TOP/clear mid-abort"
        rows.append(
            {
                "traj_id": tid,
                "lineage_id": row["lineage_id"],
                "condition": "C_only_baseline",
                "SUCCESS_NEW": int(phys["SUCCESS_NEW"]),
                "SUCCESS_12S": int(phys["SUCCESS_12S"]),
                "detach_completed": phys["detach_completed"],
                "recatch_completed": phys["recatch_completed"],
                "stable_bilateral": phys["stable_bilateral"],
                "lift_completed": phys["lift_completed"],
                "n_motion": phys["n_motion"],
                "frac_token_3": phys["frac_token_3"],
                "dominant_token": phys.get("dominant_token"),
                "max_token_run": phys["max_token_run"],
                "div_t": div.get("first_div_t"),
                "div_stage": div.get("first_div_stage"),
                "eval_reason": phys.get("eval_reason"),
                "tokens_head": phys.get("tokens_head"),
                "no_table_top_abort": 1,
            }
        )
        print(f"  baseline {i+1}/{len(val_c)} STRICT={phys['SUCCESS_NEW']} n={phys['n_motion']}", flush=True)
    n_ok = sum(r["SUCCESS_NEW"] for r in rows)
    md = f"""# BASELINE_REPRODUCTION

Generated: {_utc()}

Checkpoint: `{C_ONLY_CKPT.relative_to(ROOT).as_posix()}`

| Check | Result |
|---|---|
| Held-out STRICT | **{n_ok}/{len(rows)}** (expect ~2/19) |
| Mid-abort on clear/TABLE_TOP | **disabled** (Plan C catch geometry) |
| Policy input | obs 39-D + prev executed 7-D |
| Action decode | frozen M111 |
| Controller restore | `restore_start` / catch_snap → ballistic |

P22/P23 artifacts untouched.
"""
    _write_md(OUT / "BASELINE_REPRODUCTION.md", md)
    _write_csv(OUT / "BASELINE_HELD_OUT.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Phase 2 — state collection
# ---------------------------------------------------------------------------


def collect_policy_states(
    model, book, rho, starts, train_c, device, *, round_id: int, rng: np.random.Generator
) -> list[dict]:
    print(f"=== Phase 2: collect states round={round_id} ===", flush=True)
    env = make_env()
    unified = load_unified_by_id()
    # subsample train trajs
    order = list(range(len(train_c)))
    rng.shuffle(order)
    chosen = [train_c[i] for i in order[:N_TRAIN_TRAJ_COLLECT]]
    rows = []
    snap_dir = OUT / "snapshots" / f"round{round_id}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    for ti, row in enumerate(chosen):
        tid = row["traj_id"]
        ep = starts[tid]
        if tid in unified:
            ep = {**ep, "n_catch": unified[tid].get("n_catch")}
        expert_tl = record_timeline(env, ep, ep["actions"])
        bounds = infer_stage_bounds(expert_tl, ep.get("n_catch"))

        restore_start(env, ep)
        arm(env.sim)
        hist = ObservableObsState(DT_POLICY)
        env._hist = hist
        env._prev = np.zeros(7, np.float32)
        obs_seq: list = []
        prev_seq: list = []
        tokens: list[int] = []
        priority_ts = {0, 1, 2, 3, 5, 8, 12, 16, 20, 28, 40}
        priority_ts.add(max(0, int(bounds.get("detach_end", 10)) - 1))
        priority_ts.add(int(bounds.get("detach_end", 10)))
        priority_ts.add(int(bounds.get("first_bilateral", 14)))
        saw_uni = False
        saw_bil = False
        tok_run = 0
        last_tok = None
        outcome = None

        for t in range(MAX_POLICY_STEPS_COLLECT):
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
            tok, resid, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
            tokens.append(tok)
            if last_tok == tok:
                tok_run += 1
            else:
                tok_run = 1
                last_tok = tok

            if o["any_contact"] and not o["bilateral"]:
                saw_uni = True
            if o["bilateral"]:
                saw_bil = True

            take = t in priority_ts or tok_run >= 4 or (saw_uni and t not in {r["t"] for r in rows if r["traj_id"] == tid})
            # first unilateral / bilateral
            if o["any_contact"] and not o["bilateral"] and t <= 60:
                take = True
            if o["bilateral"] and t <= 80:
                take = True

            if take and action is not None:
                snap = make_snap(env.sim)
                sid = f"r{round_id}_{hashlib.sha1(f'{tid}:{t}:{tok}'.encode()).hexdigest()[:12]}"
                spath = snap_dir / f"{sid}.pkl"
                blob = {
                    "snap": snap,
                    "ref": ep["ref"],
                    "traj_id": tid,
                    "lineage_id": row["lineage_id"],
                    "t": t,
                    "obs": obs.copy(),
                    "prev_action": env._prev.copy(),
                    "pred_token": int(tok),
                    "pred_resid": None if resid is None else np.asarray(resid, np.float32),
                    "pred_action": np.asarray(action, np.float32),
                    "contact": o,
                    "phase": phase_of(o, t, bounds),
                    "tok_run": tok_run,
                    "round": round_id,
                }
                spath.write_bytes(pickle.dumps(blob))
                rows.append(
                    {
                        "state_id": sid,
                        "round": round_id,
                        "traj_id": tid,
                        "lineage_id": row["lineage_id"],
                        "split_source": "train",
                        "t": t,
                        "phase": blob["phase"],
                        "pred_token": int(tok),
                        "tok_run": tok_run,
                        "bilateral": o["bilateral"],
                        "any_contact": o["any_contact"],
                        "nL": o["nL"],
                        "nR": o["nR"],
                        "obj_z": o["obj_z"],
                        "clear": o["clear"],
                        "action_linf": float(np.max(np.abs(action))),
                        "snap_relpath": str(spath.relative_to(OUT)).replace("\\", "/"),
                        "priority": int(
                            t <= 5
                            or tok_run >= 4
                            or blob["phase"] in ("flight_detach", "unilateral", "detach", "recatch")
                        ),
                    }
                )

            if action is None:
                outcome = "HANDOFF"
                break
            step_legal(env.sim, action)
            env._prev = action
        else:
            outcome = "TIMEOUT"

        # annotate outcome on this traj's rows
        for r in rows:
            if r["traj_id"] == tid and r.get("eventual_outcome") is None:
                r["eventual_outcome"] = outcome

        print(f"  collect {ti+1}/{len(chosen)} {tid[:36]} states_so_far={len(rows)} out={outcome}", flush=True)

    _write_csv(OUT / "POLICY_STATE_COLLECTION.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Phase 3 — correction feasibility
# ---------------------------------------------------------------------------


def attempt_correction(env, blob: dict, *, rng: np.random.Generator) -> dict:
    """Replan from snapshot with Plan C oracle; require MuJoCo-executed STRICT success."""
    snap = blob["snap"]
    ref = blob["ref"]
    restore_ballistic(env.sim, snap)
    restore_drop(env.sim, None)
    arm(env.sim)
    bil = pad_bilateral(env.sim)

    if bil.get("bilateral"):
        # Already captured: lift + pose/handoff (no replay of original Expert future)
        prefix: list[np.ndarray] = []
        brake = make_snap(env.sim)
        lift = fast_world_lift_sequence(env.sim, brake, vz_up=0.12, target_z=0.55, max_s=8.0)
        for a in lift.get("actions") or []:
            a = np.asarray(a, np.float32)
            prefix.append(a.copy())
            step_legal(env.sim, a)
            if latched(env.sim) or not pad_bilateral(env.sim)["bilateral"]:
                return {
                    "ok": False,
                    "reason": "LOST_LIFT",
                    "stage": "lift",
                    "oracle_path": "bilateral_lift",
                    "n_actions": len(prefix),
                }
        pose_acts, meta = pose_correct_and_handoff(env.sim, ref, max_s=4.0)
        for a in pose_acts:
            prefix.append(np.asarray(a, np.float32).copy())
        # Verify STRICT via evaluate_handoff (same as SUCCESS_NEW path)
        hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
        ok = bool(hand.get("SUCCESS_NEW"))
        return {
            "ok": ok,
            "SUCCESS_NEW": ok,
            "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
            "reason": hand.get("std_reason") or hand.get("reason") or meta.get("reason"),
            "stage": "handoff" if ok else "pose_handoff_fail",
            "oracle_path": "bilateral_lift",
            "prefix_actions": prefix,
            "n_actions": len(prefix),
        }

    # Flight / unilateral: replan catch (do not reuse original Expert future)
    search = search_catch(env.sim, snap, ref, rng=rng, max_rollouts=MAX_ORACLE_ROLLOUTS)
    best = search.get("best")
    if best is None or not best.get("end_bilateral"):
        return {
            "ok": False,
            "reason": "NO_BILATERAL_CATCH",
            "stage": "catch_search",
            "oracle_path": "search_catch",
            "funnel": search.get("funnel"),
            "n_actions": 0,
            "best_score": None if best is None else best.get("score"),
        }
    out = lift_and_validate(env.sim, snap, ref, best)
    ok = bool(out.get("SUCCESS_NEW"))
    # Extra verify: replay prefix from snap and STRICT
    if ok and out.get("prefix_actions"):
        restore_ballistic(env.sim, snap)
        restore_drop(env.sim, None)
        arm(env.sim)
        for a in out["prefix_actions"]:
            step_legal(env.sim, np.asarray(a, np.float32))
            if latched(env.sim):
                ok = False
                break
        if ok:
            hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
            ok = bool(hand.get("SUCCESS_NEW"))
            out["verify_reason"] = hand.get("std_reason") or hand.get("reason")
            out["SUCCESS_12S"] = bool(hand.get("SUCCESS_12S"))
    return {
        "ok": ok,
        "SUCCESS_NEW": ok,
        "SUCCESS_12S": bool(out.get("SUCCESS_12S")),
        "reason": out.get("reason") or out.get("verify_reason"),
        "stage": out.get("stage") or "catch_lift",
        "oracle_path": "search_catch+lift_and_validate",
        "prefix_actions": list(out.get("prefix_actions") or []),
        "n_actions": len(out.get("prefix_actions") or []),
        "funnel": search.get("funnel"),
        "mechanism": out.get("mechanism") or best.get("mechanism"),
        "best_score": best.get("score"),
    }


def run_corrections(state_rows: list[dict], *, round_id: int, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    print(f"=== Phase 3: corrections round={round_id} ===", flush=True)
    env = make_env()
    # Prioritize early / flight / repeated-token states
    cands = [r for r in state_rows if r.get("round") == round_id]
    cands.sort(key=lambda r: (-int(r.get("priority") or 0), int(r["t"]), r["state_id"]))
    # Dedup near-identical (traj, phase, t-bin)
    seen = set()
    selected = []
    for r in cands:
        key = (r["traj_id"], r["phase"], int(r["t"]) // 3, int(r["pred_token"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(r)
        if len(selected) >= MAX_CORR_ATTEMPTS_PER_ROUND:
            break

    feas_rows = []
    verified = []
    for i, r in enumerate(selected):
        blob = pickle.loads((OUT / r["snap_relpath"]).read_bytes())
        t0 = time.time()
        try:
            res = attempt_correction(env, blob, rng=rng)
        except Exception as exc:
            res = {"ok": False, "reason": f"EXC:{exc}", "stage": "exception", "n_actions": 0}
        elapsed = time.time() - t0
        feas_rows.append(
            {
                "state_id": r["state_id"],
                "round": round_id,
                "traj_id": r["traj_id"],
                "lineage_id": r["lineage_id"],
                "t": r["t"],
                "phase": r["phase"],
                "pred_token": r["pred_token"],
                "ok": int(bool(res.get("ok"))),
                "reason": res.get("reason"),
                "stage": res.get("stage"),
                "oracle_path": res.get("oracle_path"),
                "n_actions": res.get("n_actions"),
                "best_score": res.get("best_score"),
                "elapsed_s": round(elapsed, 2),
                "funnel_bilateral": (res.get("funnel") or {}).get("end_bilateral"),
            }
        )
        if res.get("ok") and res.get("prefix_actions"):
            vpath = OUT / "snapshots" / f"round{round_id}" / f"corr_{r['state_id']}.pkl"
            corr_blob = {
                **blob,
                "corrective_actions": [np.asarray(a, np.float32) for a in res["prefix_actions"]],
                "original_policy_action": blob["pred_action"],
                "original_policy_token": blob["pred_token"],
                "correction_stage": res.get("stage"),
                "oracle_path": res.get("oracle_path"),
                "mechanism": res.get("mechanism"),
                "SUCCESS_NEW": True,
                "SUCCESS_12S": bool(res.get("SUCCESS_12S")),
            }
            vpath.write_bytes(pickle.dumps(corr_blob))
            verified.append(
                {
                    "state_id": r["state_id"],
                    "round": round_id,
                    "traj_id": r["traj_id"],
                    "lineage_id": r["lineage_id"],
                    "t": r["t"],
                    "phase": r["phase"],
                    "n_corrective_actions": len(res["prefix_actions"]),
                    "corr_relpath": str(vpath.relative_to(OUT)).replace("\\", "/"),
                    "oracle_path": res.get("oracle_path"),
                    "original_pred_token": r["pred_token"],
                }
            )
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  corr {i+1}/{len(selected)} ok={res.get('ok')} reason={res.get('reason')} "
                f"{elapsed:.1f}s verified={len(verified)}",
                flush=True,
            )

    _write_csv(OUT / f"CORRECTION_FEASIBILITY_round{round_id}.csv", feas_rows)
    # merge all rounds into main CSV
    all_feas = []
    for p in sorted(OUT.glob("CORRECTION_FEASIBILITY_round*.csv")):
        all_feas.extend(csv.DictReader(p.open(encoding="utf-8")))
    _write_csv(OUT / "CORRECTION_FEASIBILITY.csv", all_feas)
    return feas_rows, verified


# ---------------------------------------------------------------------------
# Phase 4 — package dataset
# ---------------------------------------------------------------------------


def package_correction(env, corr_meta: dict, book, rho, *, split: str) -> dict | None:
    blob = pickle.loads((OUT / corr_meta["corr_relpath"]).read_bytes())
    acts = [np.asarray(a, np.float32) for a in blob["corrective_actions"]]
    if not acts:
        return None
    restore_ballistic(env.sim, blob["snap"])
    restore_drop(env.sim, None)
    arm(env.sim)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    # DAgger: keep the policy's prev at intervention (may be nonzero)
    env._prev = np.asarray(blob["prev_action"], np.float32).copy()
    obs_list, prev_list = [], []
    for a in acts:
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_list.append(obs)
        prev_list.append(env._prev.copy())
        step_legal(env.sim, a)
        env._prev = a
    # terminal HANDOFF obs
    hob = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
        np.float32,
    )
    obs_list.append(hob)
    prev_list.append(env._prev.copy())
    hand = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
    if not hand.get("SUCCESS_NEW"):
        return None
    cont = np.stack(acts).astype(np.float32)
    q = quantize_trajectory(book, cont, rho, w=DEFAULT_WEIGHTS)
    # M111 replay check
    restore_ballistic(env.sim, blob["snap"])
    restore_drop(env.sim, None)
    arm(env.sim)
    for a in q["recon_actions"]:
        step_legal(env.sim, np.asarray(a, np.float32))
    hand_m = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
    m111_ok = bool(hand_m.get("SUCCESS_NEW"))
    tokens = np.concatenate([np.asarray(q["tokens"], np.int64), np.array([HANDOFF], np.int64)])
    residuals = np.concatenate(
        [np.asarray(q["residuals"], np.float32), np.zeros((1, 7), np.float32)], axis=0
    )
    stem = hashlib.sha1(f"corr_{corr_meta['state_id']}".encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.stack(obs_list).astype(np.float32),
        prev_actions=np.stack(prev_list).astype(np.float32),
        tokens=tokens,
        residuals=residuals,
        continuous_actions=cont,
        executed_actions=np.concatenate(
            [np.asarray(q["recon_actions"], np.float32), np.zeros((1, 7), np.float32)], axis=0
        ),
        loss_mask=np.ones(len(tokens), dtype=np.bool_),
        episode_boundary=np.array([False] * (len(tokens) - 1) + [True]),
        handoff_id=np.int64(HANDOFF),
        corrective=np.int32(1),
        allow_nonzero_prev0=np.int32(1),
    )
    return {
        "traj_id": f"corr__{corr_meta['state_id']}",
        "split": split,
        "family": "C",
        "lineage_id": corr_meta["lineage_id"],
        "source": "verified_correction",
        "npz_relpath": rel.replace("\\", "/"),
        "n_motion": int(len(acts)),
        "n_steps_with_handoff": int(len(tokens)),
        "phase": corr_meta["phase"],
        "round": corr_meta["round"],
        "state_id": corr_meta["state_id"],
        "m111_replay_ok": int(m111_ok),
        "original_pred_token": corr_meta["original_pred_token"],
        "allow_nonzero_prev0": True,
        "dataset_root": "p25",
    }


def build_training_sets(verified_all: list[dict], book, rho, p21: dict) -> dict:
    print("=== Phase 4: build corrective dataset ===", flush=True)
    env = make_env()
    # Hold out 20% of corrective lineages for val (still train-lineages only — not P24 val)
    lin = sorted({v["lineage_id"] for v in verified_all})
    val_lin = {lid for lid in lin if int(hashlib.sha1(lid.encode()).hexdigest()[:8], 16) % 100 >= 80}
    if not val_lin and lin:
        val_lin = {lin[0]}

    corr_rows = []
    for v in verified_all:
        split = "val" if v["lineage_id"] in val_lin else "train"
        packed = package_correction(env, v, book, rho, split=split)
        if packed and packed.get("m111_replay_ok"):
            corr_rows.append(packed)
        elif packed:
            # keep continuous-verified even if M111 slightly fails? Spec requires M111 replay verification
            print(f"  skip M111-fail {v['state_id']}", flush=True)

    # Original Plan C from P21
    orig = []
    for r in p21["trajectories"]:
        if r["family"] != "C":
            continue
        orig.append(
            {
                **r,
                "source": "original_p21_c",
                "dataset_root": "p21",
                "allow_nonzero_prev0": False,
            }
        )

    manifest = {
        "dataset_id": "p25_c_dagger",
        "generated_at": _utc(),
        "n_original_c": len(orig),
        "n_verified_corrections": len(corr_rows),
        "correction_val_lineages": sorted(val_lin),
        "original": orig,
        "corrections": corr_rows,
        "note": "Corrections only from train-lineage C_only states; held-out eval lineages excluded.",
    }
    (OUT / "VERIFIED_CORRECTIONS_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"packaged corrections={len(corr_rows)} original_c={len(orig)}", flush=True)
    return manifest


def build_samples_variant(manifest: dict, *, mode: str, split: str) -> list[dict]:
    """mode: corrective_only | mixed | original_probe"""
    samples = []
    rows = []
    if mode in ("corrective_only", "mixed"):
        rows.extend([r for r in manifest["corrections"] if r["split"] == split])
    if mode in ("mixed", "original_probe"):
        rows.extend([r for r in manifest["original"] if r["split"] == split])
    # weight: in mixed, upweight corrections
    n_corr = sum(1 for r in rows if r.get("source") == "verified_correction")
    n_orig = sum(1 for r in rows if r.get("source") == "original_p21_c")
    for row in rows:
        root = P21 if row.get("dataset_root") == "p21" else OUT
        allow = bool(row.get("allow_nonzero_prev0"))
        ep = load_npz_flexible(root, row["npz_relpath"], allow_nonzero_prev0=allow)
        if row.get("source") == "verified_correction":
            w = 2.0 if mode == "mixed" else 1.0
        else:
            w = 1.0 if n_orig == 0 else max(0.5, n_corr / max(1, n_orig)) if mode == "mixed" else 1.0
        for ci, chunk in enumerate(chunk_episode(ep, window=WINDOW)):
            samples.append(
                {
                    **chunk,
                    "sample_weight": float(w),
                    "traj_id": row["traj_id"],
                    "family": "C",
                    "lineage_id": row["lineage_id"],
                    "skill_group": "C_corr" if row.get("source") == "verified_correction" else "C",
                    "split": split,
                    "chunk_index": ci,
                }
            )
    if samples:
        mean_w = float(np.mean([s["sample_weight"] for s in samples]))
        if mean_w > 0:
            for s in samples:
                s["sample_weight"] = float(s["sample_weight"] / mean_w)
    return samples


# ---------------------------------------------------------------------------
# Phase 5–6 — train + eval
# ---------------------------------------------------------------------------


def heldout_eval(name, model, book, rho, starts, val_c, device) -> list[dict]:
    env = make_env()
    unified = load_unified_by_id()
    rows = []
    for row in val_c:
        tid = row["traj_id"]
        ep = starts[tid]
        if tid in unified:
            ep = {**ep, "n_catch": unified[tid].get("n_catch")}
        expert_tl = record_timeline(env, ep, ep["actions"])
        bounds = infer_stage_bounds(expert_tl, ep.get("n_catch"))
        sft_tl, _ = rollout_policy_timeline_c(env, model, book, rho, ep, device=device, max_steps=400)
        div = first_divergence(expert_tl, sft_tl, bounds)
        phys = physical_stage_completion(
            env, ep, None, mode="policy", model=model, book=book, rho=rho, device=device, max_steps=800
        )
        rows.append(
            {
                "condition": name,
                "traj_id": tid,
                "lineage_id": row["lineage_id"],
                "SUCCESS_NEW": int(phys["SUCCESS_NEW"]),
                "SUCCESS_12S": int(phys["SUCCESS_12S"]),
                "detach_completed": phys["detach_completed"],
                "recatch_completed": phys["recatch_completed"],
                "stable_bilateral": phys["stable_bilateral"],
                "lift_completed": phys["lift_completed"],
                "n_motion": phys["n_motion"],
                "frac_token_3": phys["frac_token_3"],
                "max_token_run": phys["max_token_run"],
                "dominant_token": phys.get("dominant_token"),
                "div_t": div.get("first_div_t"),
                "div_stage": div.get("first_div_stage"),
                "premature_handoff": phys["premature_handoff"],
                "eval_reason": phys.get("eval_reason"),
                "tokens_head": phys.get("tokens_head"),
            }
        )
    return rows


def render_pair(env, models, book, rho, ep, stem: str, device):
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
    for name, model in models.items():
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
        path = OUT / "videos" / f"{stem}__{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_frames_mp4(frames, path, fps=20)
        print("video", path.name, len(frames), flush=True)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P25 device={device} ===", flush=True)
    rng = np.random.default_rng(SEED)

    book, rho, _ = load_m111_book(M111_META)
    if not C_ONLY_CKPT.is_file():
        raise RuntimeError(f"missing {C_ONLY_CKPT}")
    model_a, _, _ = load_m111(C_ONLY_CKPT, device)
    model_a.eval()

    p21 = load_p21()
    starts = build_start_index(p21)
    val_c = [r for r in p21["trajectories"] if r["family"] == "C" and r["split"] == "val"]
    train_c = [r for r in p21["trajectories"] if r["family"] == "C" and r["split"] == "train"]

    # Phase 1
    base_rows = reproduce_baseline(model_a, book, rho, starts, val_c, device)
    base_strict = sum(r["SUCCESS_NEW"] for r in base_rows)

    all_states = []
    all_verified = []
    all_feas = []
    round_models = {"A_baseline": model_a}
    held_all = list(base_rows)
    round_cmp = []

    policy = model_a
    for round_id in range(1, N_DAGGER_ROUNDS + 1):
        states = collect_policy_states(
            policy, book, rho, starts, train_c, device, round_id=round_id, rng=rng
        )
        all_states.extend(states)
        feas, verified = run_corrections(states, round_id=round_id, rng=rng)
        all_feas.extend(feas)
        all_verified.extend(verified)

        recover_rate = sum(1 for f in feas if int(f["ok"])) / max(1, len(feas))
        print(f"round {round_id} recover_rate={recover_rate:.3f} n_verified={len(verified)}", flush=True)

        if not verified and round_id == 1:
            print("No verified corrections in round 1 — still building empty manifest + baseline-only report", flush=True)

        manifest = build_training_sets(all_verified, book, rho, p21)

        configs = {
            "init_checkpoint": str(C_ONLY_CKPT.relative_to(ROOT)).replace("\\", "/"),
            "epochs": EPOCHS_FT,
            "lr": LR_FT,
            "batch_size": BATCH,
            "seed": SEED,
            "max_oracle_rollouts": MAX_ORACLE_ROLLOUTS,
            "n_train_traj_collect": N_TRAIN_TRAJ_COLLECT,
            "max_corr_attempts_per_round": MAX_CORR_ATTEMPTS_PER_ROUND,
            "round": round_id,
            "n_verified_cumulative": len(all_verified),
            "variants": {
                "A_baseline": "no additional training",
                "B_corrective_only": "FT on verified corrections",
                "C_mixed": "FT on corrections + original Plan C (reweighted)",
            },
        }
        (OUT / "CORRECTIVE_TRAINING_CONFIGS.yaml").write_text(
            yaml.safe_dump(configs, sort_keys=False), encoding="utf-8"
        )

        # Train B and C if we have corrections
        models_round = {"A_baseline": model_a}
        if manifest["corrections"]:
            for vname, mode in (("B_corrective_only", "corrective_only"), ("C_mixed", "mixed")):
                train_s = build_samples_variant(manifest, mode=mode, split="train")
                val_s = build_samples_variant(manifest, mode=mode, split="val")
                if not train_s:
                    # fallback val from train chunks
                    val_s = train_s[: max(1, len(train_s) // 5)] or train_s
                out_dir = OUT / "checkpoints" / f"round{round_id}_{vname}"
                print(f"=== train {vname} n={len(train_s)} ===", flush=True)
                train_m111(
                    train_samples=train_s,
                    val_samples=val_s or train_s[:1],
                    cfg={
                        "name": vname,
                        "seed": SEED + round_id,
                        "epochs": EPOCHS_FT,
                        "batch_size": BATCH,
                        "lr": LR_FT,
                        "device": str(device),
                    },
                    out_dir=out_dir,
                    book=book,
                    rho=rho,
                    init_checkpoint=C_ONLY_CKPT if round_id == 1 else (
                        OUT / "checkpoints" / f"round{round_id-1}_C_mixed" / "checkpoint_best.pt"
                        if (OUT / "checkpoints" / f"round{round_id-1}_C_mixed" / "checkpoint_best.pt").is_file()
                        else C_ONLY_CKPT
                    ),
                )
                m, _, _ = load_m111(out_dir / "checkpoint_best.pt", device)
                m.eval()
                models_round[vname] = m
                # TF metrics
                if train_s:
                    tr = eval_token_metrics(m, train_s, device, max_batches=30)
                    print(f"  {vname} train_token_acc={tr.get('token_acc'):.4f}", flush=True)

        # Held-out eval
        print(f"=== Phase 6: held-out eval round={round_id} ===", flush=True)
        for name, model in models_round.items():
            tag = f"round{round_id}_{name}" if name != "A_baseline" or round_id > 1 else "A_baseline"
            if name == "A_baseline" and round_id > 1:
                continue  # already have baseline
            ev = heldout_eval(tag if name != "A_baseline" else "A_baseline", model, book, rho, starts, val_c, device)
            if name == "A_baseline" and round_id == 1:
                pass  # already in held_all from phase1; refresh
                held_all = [r for r in held_all if r["condition"] != "A_baseline" and r["condition"] != "C_only_baseline"]
                for r in ev:
                    r["condition"] = "A_baseline"
                held_all.extend(ev)
            else:
                held_all.extend(ev)
            n = len(ev)
            strict = sum(r["SUCCESS_NEW"] for r in ev) / max(1, n)
            recatch = sum(int(r["recatch_completed"]) for r in ev) / max(1, n)
            lift = sum(int(r["lift_completed"]) for r in ev) / max(1, n)
            div_t = float(np.mean([float(r["div_t"]) for r in ev if r.get("div_t") is not None]))
            round_cmp.append(
                {
                    "round": round_id,
                    "condition": tag if name != "A_baseline" else "A_baseline",
                    "n": n,
                    "strict_rate": strict,
                    "strict_count": sum(r["SUCCESS_NEW"] for r in ev),
                    "recatch_rate": recatch,
                    "lift_rate": lift,
                    "detach_rate": sum(int(r["detach_completed"]) for r in ev) / max(1, n),
                    "stable_rate": sum(int(r["stable_bilateral"]) for r in ev) / max(1, n),
                    "mean_div_t": div_t,
                    "mean_frac_token_3": float(np.mean([float(r["frac_token_3"]) for r in ev])),
                    "mean_n_motion": float(np.mean([float(r["n_motion"]) for r in ev])),
                    "n_verified_corrections_used": len(all_verified),
                    "oracle_recover_rate_this_round": recover_rate,
                }
            )
            print(f"  {tag} STRICT={strict:.3f} recatch={recatch:.3f}", flush=True)

        # Pick best for next DAgger round (prefer mixed then corrective)
        best_name = "A_baseline"
        best_s = base_strict / max(1, len(val_c))
        for name in ("C_mixed", "B_corrective_only"):
            if name in models_round:
                tag = f"round{round_id}_{name}"
                row = next(r for r in round_cmp if r["condition"] == tag)
                if row["strict_rate"] >= best_s:
                    best_s = row["strict_rate"]
                    best_name = name
                    policy = models_round[name]
        round_models[f"round{round_id}_best"] = policy

        # Stop if no improvement or low recoverability
        improved = best_s > (base_strict / max(1, len(val_c))) + 1e-9
        if round_id == 1 and (recover_rate < 0.05 or not verified):
            print("Stopping DAgger: insufficient recoverable corrections", flush=True)
            break
        if round_id >= 1 and not improved and round_id == N_DAGGER_ROUNDS:
            break
        if round_id == 1 and not improved:
            print("Round 1 did not improve STRICT; skipping round 2", flush=True)
            break

    _write_csv(OUT / "HELDOUT_C_EVALUATION.csv", held_all)
    _write_csv(OUT / "ROUND_COMPARISON.csv", round_cmp)
    _write_csv(OUT / "POLICY_STATE_COLLECTION.csv", all_states)

    # Videos
    env = make_env()
    ep0 = starts[val_c[0]["traj_id"]]
    vid_models = {"A_baseline": model_a}
    for k, m in round_models.items():
        if "C_mixed" in k or "B_corrective" in str(m):
            pass
    # load last mixed if exists
    for p in sorted(OUT.glob("checkpoints/round*_C_mixed/checkpoint_best.pt")):
        m, _, _ = load_m111(p, device)
        m.eval()
        vid_models[p.parent.name] = m
    for p in sorted(OUT.glob("checkpoints/round*_B_corrective_only/checkpoint_best.pt")):
        m, _, _ = load_m111(p, device)
        m.eval()
        vid_models[p.parent.name] = m
    render_pair(env, vid_models, book, rho, ep0, "heldout_c0", device)

    # Phase 8 attribution + final
    n_attempt = len(all_feas)
    n_recov = sum(1 for f in all_feas if int(f.get("ok") or 0))
    frac_recov = n_recov / max(1, n_attempt)

    def strict_of(cond):
        xs = [r for r in held_all if r["condition"] == cond]
        if not xs:
            return 0.0, 0
        return sum(int(r["SUCCESS_NEW"]) for r in xs) / len(xs), len(xs)

    a_s, a_n = strict_of("A_baseline")
    # best FT
    best_ft = max(round_cmp, key=lambda r: r["strict_rate"]) if round_cmp else {"condition": "A_baseline", "strict_rate": a_s, "strict_count": int(a_s * a_n)}

    fail_md = f"""# FAILURE_ATTRIBUTION

Generated: {_utc()}

## Oracle coverage vs policy learning

| Bucket | Count |
|---|---:|
| Correction attempts | {n_attempt} |
| Oracle recovered (verified STRICT) | {n_recov} |
| Oracle failed (unrecoverable / search fail) | {n_attempt - n_recov} |
| Recoverability fraction | {frac_recov:.3f} |

### Oracle failure reasons
"""
    fail_md += "\n".join(
        f"- `{k}`: {v}" for k, v in Counter(f.get("reason") for f in all_feas if not int(f.get("ok") or 0)).most_common()
    )
    fail_md += f"""

## Interpretation

- If recoverability is low, **Oracle coverage failure** dominates — DAgger cannot label most C_only states.
- If recoverability is high but held-out STRICT stays near {base_strict}/{a_n}, **policy learning failure**.
- States successfully recovered by updated policy: see STRICT gains in `ROUND_COMPARISON.csv`.

Best FT condition: **{best_ft.get('condition')}** STRICT={best_ft.get('strict_rate'):.3f} ({best_ft.get('strict_count')}/{best_ft.get('n', a_n)}).
"""
    _write_md(OUT / "FAILURE_ATTRIBUTION.md", fail_md)

    another_round = bool(
        frac_recov >= 0.15
        and best_ft.get("strict_rate", 0) > a_s + 0.02
        and best_ft.get("strict_rate", 0) < 0.5
    )

    final = f"""# FINAL_REPORT — P25 Plan C DAgger

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Fraction of C_only-visited failure states physically recoverable?**  
   Verified STRICT corrections / attempts = **{n_recov}/{n_attempt} ({frac_recov:.3f})**.

2. **Can existing Expert Oracle produce valid corrective M111 sequences?**  
   {"Yes, for a subset" if n_recov else "Rarely / not in this sample"} — path `search_catch` + `lift_and_validate` (or bilateral lift+handoff), with MuJoCo replay + M111 quantize checks. Failures tracked in `CORRECTION_FEASIBILITY.csv`.

3. **Does corrective supervision reduce early closed-loop divergence?**  
   See mean `div_t` / stage completion in `ROUND_COMPARISON.csv` vs A_baseline.

4. **Does C STRICT improve beyond 2/19?**  
   Baseline reproduction: **{base_strict}/{a_n}**. Best FT: **{best_ft.get('strict_count')}/{best_ft.get('n', a_n)}** (`{best_ft.get('condition')}`).

5. **Does mixed training preserve original Plan C behavior?**  
   Compare C_mixed vs B_corrective_only retention via stage completion + STRICT; original demos remain in mixed loss (`CORRECTIVE_TRAINING_CONFIGS.yaml`).

6. **Is another DAgger round justified?**  
   **{"YES" if another_round else "NO / not strongly"}** (need sufficient oracle recoverability and STRICT gains; capped at 2 rounds here).

7. **Remaining obstacle to reliable Plan C recatch?**  
   {"Oracle cannot recover most C_only states (coverage)" if frac_recov < 0.2 else "Policy still mishandles recoverable states / residual early divergence"} — details in `FAILURE_ATTRIBUTION.md`.

## Integrity

- Init: P24 C_only; M111 / physics / STRICT / 39-D unchanged  
- No GRPO; no hierarchy  
- No held-out lineages in corrective training  
- No TABLE_TOP/clear mid-abort in C eval  

## Artifacts

`results/diagnostics/p25_c_dagger/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p25_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "base_strict": base_strict,
                "n_attempt": n_attempt,
                "n_recovered": n_recov,
                "frac_recoverable": frac_recov,
                "best_ft": best_ft,
                "another_round_justified": another_round,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P25 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
