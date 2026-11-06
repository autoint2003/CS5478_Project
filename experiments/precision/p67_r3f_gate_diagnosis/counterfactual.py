"""Phase 3 — Expert/Actor counterfactual from Actor post-stable states."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from data_generation.oracles.plan_c.expert_search import run_stage1_catch_and_lift, search_catch
from data_generation.oracles.precision.state_machine import run_precision
from envs.simulation.capture import take_capture
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers, step_raw
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p67_r3f_gate_diagnosis.instrument import rollout_instrumented
from experiments.precision.p67_r3f_gate_diagnosis.paths import (
    MAX_EP_STEPS,
    N_ACTOR_CF,
    N_SNAPSHOTS_TARGET,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    STATE_DIR,
    T_MAX,
)


def _lookup() -> dict:
    return {s.traj_id: s for s in build_p54_dataset(seed=SEED)["schedule"] if s.skill == "C"}


def _train_c(n: int = 20) -> list:
    blocked = blocked_eval_ids()
    return [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ][:n]


def mine_post_stable_snapshots(*, device: str = "cuda", target: int = N_SNAPSHOTS_TARGET) -> dict:
    """Harvest Actor states: stable_hold & ¬R3-ready, or near-ready then break, or sidewall unstable."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    starts = _train_c(24)
    manifest = []
    seen = set()

    for st in starts:
        if len(manifest) >= target:
            break
        for i in range(8):
            if len(manifest) >= target:
                break
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=68000 + i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ensure_inner_capture_buffers(env)
            done = False
            info = {}
            captured_this = False
            peak_ok = 0.0
            while not done:
                # Capture candidates BEFORE action when post-stable / near
                tr = env.mon.tracker
                ok_s = float(tr.ok_s) if tr else 0.0
                peak_ok = max(peak_ok, ok_s)
                preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
                stable = bool(env.diag.get("stable_hold"))
                near = bool(preds["frame_ok"] and ok_s > 0.05)
                r3 = bool(preds["dwell_complete"] and preds["frame_ok"])
                kind = None
                if stable and not r3 and env.t + 12 <= env.max_steps:
                    kind = "stable_not_r3"
                elif near and not r3 and env.t + 12 <= env.max_steps:
                    kind = "near_ready"
                elif preds["correct_sidewall"] and not stable and env.t + 12 <= env.max_steps:
                    kind = "sidewall_unstable"

                if kind and not captured_this:
                    key = (st.traj_id, kind, int(env.t) // 4)
                    if key not in seen:
                        seen.add(key)
                        ensure_inner_capture_buffers(env)
                        state = take_full_state(
                            env,
                            meta={
                                "kind": kind,
                                "traj_id": st.traj_id,
                                "ok_s": ok_s,
                                "preds": {k: preds[k] for k in (
                                    "frame_ok", "bilateral", "min_pad_ok", "margin_ok",
                                    "correct_sidewall", "theta", "e_pos", "nL", "nR",
                                )},
                            },
                        )
                        sid = f"{st.traj_id}__{kind}_t{env.t}".replace("/", "_")[:170]
                        path = STATE_DIR / f"{sid}.pkl"
                        path.write_bytes(pickle.dumps(state))
                        entry = {
                            "id": sid,
                            "path": str(path),
                            "traj_id": st.traj_id,
                            "kind": kind,
                            "t": int(env.t),
                            "remaining": int(env.max_steps - env.t),
                            "ok_s": ok_s,
                            "preds": state["meta"]["preds"],
                        }
                        manifest.append(entry)
                        captured_this = True
                        print(f"[mine] {kind} t={env.t} ok_s={ok_s:.3f} {st.traj_id[-28:]}", flush=True)

                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
                _r, done, info = env.step(tok, resid)

    man = {
        "n_states": len(manifest),
        "states": manifest,
        "policy": str(P64_STEP4_ACTOR),
    }
    (OUT / "POST_STABLE_SNAPSHOTS.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def _actor_continue(env, actor, critic, start, state, device, *, n: int, seed0: int) -> dict:
    stats = {"n": 0, "r3": 0, "hand": 0, "succ": 0, "lost_bil": 0, "returns": []}
    for i in range(n):
        seed_all(seed0 + i, deterministic_torch=True)
        restore_full_state(env, state, start)
        ever_bil = True
        lost = False
        done = False
        info = {}
        ep_r = 0.0
        while not done:
            tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            r, done, info = env.step(tok, resid)
            ep_r += float(r)
            m = info.get("metrics") or {}
            if ever_bil and not int(m.get("bilateral") or 0):
                lost = True
        stats["n"] += 1
        stats["r3"] += int(info.get("handoff") or 0)  # R3_EARLY or ready
        # finer: check reason
        if str(info.get("handoff_reason") or "") == "R3_EARLY":
            pass
        stats["hand"] += int(info.get("handoff") or 0)
        stats["succ"] += int(info.get("SUCCESS_RELAXED_12S") or 0)
        stats["lost_bil"] += int(lost)
        stats["returns"].append(ep_r)
    return stats


def _expert_replan_continue(env, start, state) -> dict:
    """Re-plan Expert from restored Actor state (P55-style); do NOT replay Expert tape."""
    from training.recovery_execution import make_snap

    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    ref = (start.ep or {}).get("ref")
    inner = env.env
    m0 = state_metrics(inner.sim)
    t0 = time.perf_counter()
    actions: list[np.ndarray] = []
    status = "NONE"
    try:
        # Prefer precision when already bilateral (post-stable); else catch+lift then precision
        if int(m0.get("bilateral") or 0) == 1:
            cap = take_capture(inner)
            blob = {"capture": cap, "ref": ref, "ic_id": start.traj_id}
            res = run_precision(inner, blob, plan_id=f"p67_{start.traj_id[-24:]}")
            status = str(getattr(res, "status", None) or "PRECISION")
            actions = [np.asarray(a, np.float32) for a in (getattr(res, "actions", None) or [])]
        else:
            snap = make_snap(inner.sim)
            rng = np.random.default_rng(SEED)
            search = search_catch(inner.sim, snap, ref, rng=rng, max_rollouts=12)
            best = search.get("best")
            if not best or not best.get("end_bilateral"):
                return {
                    "accepted": False,
                    "status": "CATCH_NO_BILATERAL",
                    "SUCCESS_RELAXED_12S": False,
                    "handoff": False,
                    "r3_ready": False,
                    "lost_bil": False,
                    "wall_s": time.perf_counter() - t0,
                    "n_actions": 0,
                    "compatible": True,
                }
            stage = run_stage1_catch_and_lift(inner.sim, snap, ref, best)
            actions = [np.asarray(a, np.float32) for a in (stage.get("stage1_actions") or [])]
            status = "CATCH_STAGE1_OK" if stage.get("ok") else str(stage.get("reason") or "CATCH_FAIL")
            # then precision from resulting state via fresh capture after applying catch on Stabilize env
            restore_full_state(env, state, start)
            ensure_inner_capture_buffers(env)

        # Apply planned actions through StabilizeGrpoEnv bookkeeping (horizon preserved)
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        done = False
        info: dict = {}
        ever_bil = int(m0.get("bilateral") or 0) == 1
        lost = False
        for a in actions:
            if done or env.t >= env.max_steps:
                break
            _r, done, info = step_raw(env, a)
            bil = int((info.get("metrics") or {}).get("bilateral") or 0)
            if ever_bil and not bil:
                lost = True
            if bil:
                ever_bil = True
        # If precision returned no actions but gate_ok, settle grip and let R3 observe
        if not actions and not done:
            for _ in range(min(8, env.max_steps - env.t)):
                a = np.zeros(7, np.float32)
                a[6] = 1.0
                _r, done, info = step_raw(env, a)
                if done:
                    break
        succ = int((info or {}).get("SUCCESS_RELAXED_12S") or 0)
        hand = int((info or {}).get("handoff") or 0)
        reason = str((info or {}).get("handoff_reason") or "")
        r3 = hand and reason == "R3_EARLY"
        return {
            "accepted": bool(actions) or succ or hand,
            "status": status,
            "SUCCESS_RELAXED_12S": bool(succ),
            "handoff": bool(hand),
            "r3_ready": bool(r3 or (hand and reason != "TIMEOUT")),
            "lost_bil": bool(lost),
            "wall_s": time.perf_counter() - t0,
            "n_actions": len(actions),
            "compatible": True,
            "handoff_reason": reason,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "accepted": False,
            "status": f"EXCEPTION:{type(e).__name__}",
            "error": str(e)[:240],
            "SUCCESS_RELAXED_12S": False,
            "handoff": False,
            "r3_ready": False,
            "lost_bil": False,
            "wall_s": time.perf_counter() - t0,
            "n_actions": 0,
            "compatible": False,
        }


def run_counterfactuals(*, device: str = "cuda") -> dict:
    man_path = OUT / "POST_STABLE_SNAPSHOTS.json"
    if not man_path.is_file():
        mine_post_stable_snapshots(device=device)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    lookup = _lookup()
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    rows = []
    buckets = {"expert_ok_actor_fail": 0, "both_ok": 0, "neither": 0, "incompatible": 0}
    for ent in man["states"]:
        start = lookup.get(ent["traj_id"])
        if start is None:
            continue
        state = pickle.loads(Path(ent["path"]).read_bytes())
        # quick restore sanity
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        q0 = np.asarray(env.env.sim.data.qpos, float).copy()
        restore_full_state(env, state, start)
        q1 = np.asarray(env.env.sim.data.qpos, float).copy()
        restore_ok = bool(np.allclose(q0, q1, atol=1e-7))

        actor_stats = _actor_continue(
            env, actor, critic, start, state, device_t, n=N_ACTOR_CF, seed0=SEED + hash(ent["id"]) % 10000
        )
        expert = _expert_replan_continue(env, start, state)

        actor_succ = actor_stats["succ"] / max(1, actor_stats["n"])
        actor_r3 = actor_stats["hand"] / max(1, actor_stats["n"])
        exp_ok = bool(expert.get("SUCCESS_RELAXED_12S") or expert.get("r3_ready"))
        act_fail = actor_succ < 0.25 and actor_r3 < 0.25

        if not expert.get("compatible", True):
            bucket = "incompatible"
        elif exp_ok and act_fail:
            bucket = "expert_ok_actor_fail"
        elif exp_ok and not act_fail:
            bucket = "both_ok"
        elif (not exp_ok) and act_fail:
            bucket = "neither"
        else:
            bucket = "neither" if not exp_ok else "both_ok"
        buckets[bucket] += 1

        row = {
            "id": ent["id"],
            "kind": ent["kind"],
            "traj_id": ent["traj_id"],
            "restore_ok": restore_ok,
            "actor": {
                "p_succ": actor_succ,
                "p_hand": actor_r3,
                "p_lost_bil": actor_stats["lost_bil"] / max(1, actor_stats["n"]),
                "mean_return": float(np.mean(actor_stats["returns"])) if actor_stats["returns"] else None,
            },
            "expert": expert,
            "bucket": bucket,
            "preds_at_snap": ent.get("preds"),
        }
        rows.append(row)
        print(
            f"[cf] {ent['kind']} bucket={bucket} act_succ={actor_succ:.2f} "
            f"exp={expert.get('SUCCESS_RELAXED_12S')} status={expert.get('status')} "
            f"{ent['id'][-40:]}",
            flush=True,
        )

    report = {
        "n": len(rows),
        "buckets": buckets,
        "rows": rows,
        "key_question": (
            "Are post-stable C states recoverable with better control, "
            "or already physically unfavorable?"
        ),
    }
    (OUT / "EXPERT_COUNTERFACTUAL.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Expert Counterfactual (P67 Phase 3)\n"]
    md.append("Expert continuation = **re-plan** from Actor state (not Expert tape replay).\n")
    md.append(f"Snapshots: **{len(rows)}**\n")
    md.append("## Buckets\n")
    for k, v in buckets.items():
        md.append(f"- **{k}**: {v}")
    n_rec = buckets["expert_ok_actor_fail"] + buckets["both_ok"]
    n_fail = buckets["neither"]
    md.append(
        f"\n**Recoverable fraction (Expert R3/success):** "
        f"{n_rec}/{max(1,len(rows))} ; neither={n_fail}; incompatible={buckets['incompatible']}\n"
    )
    (OUT / "EXPERT_COUNTERFACTUAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return report


if __name__ == "__main__":
    mine_post_stable_snapshots()
    run_counterfactuals()
