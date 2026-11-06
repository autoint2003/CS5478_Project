"""Phase 2 — Restoration consistency + Expert feasibility + Actor probe on snapshots."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.continuation import rollout_continuation
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p66_c_curriculum_ppo.expert_drive import expert_actions, step_raw
from experiments.precision.p66_c_curriculum_ppo.geometry_validate import _r3f_ready
from experiments.precision.p66_c_curriculum_ppo.paths import (
    MAX_EP_STEPS,
    N_ACTOR_PROBE,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


def _lookup_starts() -> dict[str, Any]:
    ds = build_p54_dataset(seed=SEED)
    return {s.traj_id: s for s in ds["schedule"] if s.skill == "C"}


def _load_manifest() -> dict:
    path = OUT / "CURRICULUM_SNAPSHOT_MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_restoration_tests(*, n_states: int = 8, n_actions: int = 12) -> dict:
    man = _load_manifest()
    states = man["states"][:n_states]
    lookup = _lookup_starts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    critic = load_critic(P64_STEP4_CRITIC, actor, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    results = []
    all_ok = True
    for ent in states:
        start = lookup.get(ent["traj_id"])
        if start is None:
            results.append({"id": ent["id"], "ok": False, "reason": "missing_start"})
            all_ok = False
            continue
        state = pickle.loads(Path(ent["path"]).read_bytes())
        restore_full_state(env, state, start)
        q0 = np.asarray(env.env.sim.data.qpos, float).copy()
        t0 = int(env.t)
        ms0 = int(env.max_steps)
        # collect fixed actions from actor
        actions = []
        seed_all(SEED + hash(ent["id"]) % 100000, deterministic_torch=True)
        # fresh restore for action collection
        restore_full_state(env, state, start)
        done = False
        info = {}
        for _ in range(n_actions):
            if done:
                break
            tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            actions.append((int(tok), np.asarray(resid, np.float32).copy()))
            _r, done, info = env.step(tok, resid)
        q_a = np.asarray(env.env.sim.data.qpos, float).copy()
        rew_a = float(info.get("t") or 0)

        # restore and replay same actions
        restore_full_state(env, state, start)
        assert int(env.t) == t0 and int(env.max_steps) == ms0
        q0b = np.asarray(env.env.sim.data.qpos, float).copy()
        qpos_ok = bool(np.allclose(q0, q0b, atol=1e-8))
        done = False
        for tok, resid in actions:
            if done:
                break
            _r, done, info = env.step(int(tok), resid)
        q_b = np.asarray(env.env.sim.data.qpos, float).copy()
        replay_ok = bool(np.allclose(q_a, q_b, atol=1e-6))
        ok = qpos_ok and replay_ok and (ms0 - t0) >= 1
        all_ok = all_ok and ok
        results.append(
            {
                "id": ent["id"],
                "ok": ok,
                "qpos_restore_ok": qpos_ok,
                "replay_ok": replay_ok,
                "t": t0,
                "max_steps": ms0,
                "remaining": ms0 - t0,
                "n_actions_replayed": len(actions),
            }
        )
        print(f"[restore] {ent['id'][-50:]} ok={ok}", flush=True)

    report = {"all_ok": all_ok, "n": len(results), "results": results}
    (OUT / "RESTORATION_TESTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Restoration Tests (P66 Phase 2)\n"]
    md.append(f"Overall OK: **{all_ok}** ({sum(r['ok'] for r in results)}/{len(results)})\n")
    for r in results:
        md.append(
            f"- `{r['id'][-60:]}`: ok={r['ok']} restore={r.get('qpos_restore_ok')} "
            f"replay={r.get('replay_ok')} rem={r.get('remaining')}"
        )
    md.append("\nRemaining policy step / max_steps preserved (episode clock not reset).\n")
    (OUT / "RESTORATION_TESTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_expert_feasibility(*, max_states: int | None = None, actor_probe_n: int | None = None) -> dict:
    """Replay Expert continuation from each snapshot; require SUCCESS_RELAXED or Expert label path."""
    man = _load_manifest()
    lookup = _lookup_starts()
    device = torch.device("cpu")
    _, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    # Also Actor probes (sampled subset for speed)
    device_a = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book2, rho2, _ = load_actor(P64_STEP4_ACTOR, device_a)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_a)
    env_a = StabilizeGrpoEnv(
        book2, rho2, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    states = list(man["states"])
    if max_states is not None:
        states = states[: int(max_states)]
    n_probe = int(actor_probe_n if actor_probe_n is not None else N_ACTOR_PROBE)
    # Actor-probe every K-th state to keep wall-time reasonable
    probe_stride = max(1, len(states) // 40)

    rows = []
    keep_ids = []
    drop_ids = set()
    for si, ent in enumerate(states):
        start = lookup[ent["traj_id"]]
        state = pickle.loads(Path(ent["path"]).read_bytes())
        acts = expert_actions(start.ep or {})
        # Expert continuation: restore, then apply remaining expert actions from snapshot_t
        snap_t = int(ent["snapshot_t"])
        restore_full_state(env, state, start)
        done = False
        info: dict = {}
        ever_sw = False
        for a in acts[snap_t:]:
            if done or env.t >= env.max_steps:
                break
            _r, done, info = step_raw(env, a)
            if is_correct_sidewall(classify_contacts(env.env.sim)):
                ever_sw = True
        expert_succ = int((info or {}).get("SUCCESS_RELAXED_12S") or 0)
        expert_ok = bool(expert_succ or (ever_sw and int((info or {}).get("c_diag", {}).get("recatch") or 0)))

        actor_stats = {"n": 0, "sw": 0, "recatch": 0, "r3f": 0, "hand": 0, "succ": 0}
        do_probe = (si % probe_stride == 0)
        if do_probe:
            for i in range(n_probe):
                seed_all(SEED + 9000 + i + hash(ent["id"]) % 10000, deterministic_torch=True)
                restore_full_state(env_a, state, start)
                done = False
                info = {}
                ever_sw_a = ever_r3f = False
                while not done:
                    tok, resid, _, _ = act_with_value(
                        actor, critic, env_a.obs_seq, env_a.prev_seq, device_a
                    )
                    _r, done, info = env_a.step(tok, resid)
                    if is_correct_sidewall(classify_contacts(env_a.env.sim)):
                        ever_sw_a = True
                    if _r3f_ready(env_a):
                        ever_r3f = True
                cd = info.get("c_diag") or {}
                actor_stats["n"] += 1
                actor_stats["sw"] += int(ever_sw_a)
                actor_stats["recatch"] += int(cd.get("recatch") or 0)
                actor_stats["r3f"] += int(ever_r3f)
                actor_stats["hand"] += int(info.get("handoff") or 0)
                actor_stats["succ"] += int(info.get("SUCCESS_RELAXED_12S") or 0)

        feasible = bool(expert_ok)
        row = {
            "id": ent["id"],
            "traj_id": ent["traj_id"],
            "stage": ent["stage"],
            "expert_ok": expert_ok,
            "expert_succ_relaxed": expert_succ,
            "actor_probe": actor_stats,
            "actor_probed": do_probe,
            "feasible": feasible,
        }
        rows.append(row)
        if feasible:
            keep_ids.append(ent["id"])
        else:
            drop_ids.add(ent["id"])
        print(
            f"[feas] {ent['stage']} expert_ok={expert_ok} probed={do_probe} "
            f"actor_sw={actor_stats['sw']}/{actor_stats['n']} "
            f"succ={actor_stats['succ']} {ent['id'][-40:]}",
            flush=True,
        )

    # Drop only explicitly infeasible; keep untested-as-failed out of the set
    new_states = [s for s in man["states"] if s["id"] not in drop_ids]
    # If we only tested a prefix, also keep untested
    tested_ids = {r["id"] for r in rows}
    for s in man["states"]:
        if s["id"] not in tested_ids and s not in new_states:
            new_states.append(s)
    # Dedup by id
    seen = set()
    dedup = []
    for s in new_states:
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        dedup.append(s)
    man["states"] = dedup
    man["n_states"] = len(dedup)
    man["feasibility_filter"] = True
    man["n_dropped_infeasible"] = len(drop_ids)
    (OUT / "CURRICULUM_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )

    report = {
        "n_tested": len(rows),
        "n_feasible": len(keep_ids),
        "rows": rows,
    }
    (OUT / "EXPERT_FEASIBILITY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Expert Feasibility (P66 Phase 2)\n"]
    md.append(f"Feasible snapshots: **{len(keep_ids)}** / {len(rows)}\n")
    md.append("Expert success establishes physical recoverability under at least one controller.\n")
    by_stage: dict[str, list] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)
    for stage, rs in by_stage.items():
        n_ok = sum(1 for r in rs if r["feasible"])
        md.append(f"- {stage}: {n_ok}/{len(rs)} feasible")
    (OUT / "EXPERT_FEASIBILITY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    run_restoration_tests()
    run_expert_feasibility()
