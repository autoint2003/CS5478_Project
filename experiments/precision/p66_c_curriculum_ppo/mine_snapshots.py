"""Phase 2 — Mine physically valid pre-recatch curriculum snapshots from train C Experts."""

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
from experiments.precision.p60_c_stable_state_grpo.full_state import take_full_state
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (
    ensure_inner_capture_buffers,
    expert_actions,
    step_raw,
)
from experiments.precision.p66_c_curriculum_ppo.geometry_validate import _r3f_ready
from experiments.precision.p66_c_curriculum_ppo.paths import (
    MAX_EP_STEPS,
    MIN_REMAINING_STEPS,
    OUT,
    P64_STEP4_ACTOR,
    SEED,
    STAGE_OFFSETS,
    STATE_DIR,
    T_MAX,
)


def _train_c_starts() -> list:
    blocked = blocked_eval_ids()
    return [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ]


def mine_pre_recatch_snapshots(*, n_experts: int = 40) -> dict:
    """For each Expert, find first correct sidewall; capture states at STAGE_OFFSETS before it."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    _, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    # Policy horizon for curriculum (Actor continues within 80 steps from catch)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    starts = _train_c_starts()[:n_experts]
    manifest: list[dict[str, Any]] = []
    skipped = []

    for st in starts:
        ep = st.ep or {}
        acts = expert_actions(ep)
        # Drive only within policy horizon; sidewall is typically at t≈4–13
        env.max_steps = MAX_EP_STEPS
        env.reset(st)
        ensure_inner_capture_buffers(env)
        # Buffer states keyed by policy step AFTER each expert action (env.t)
        # Also keep pre-action state at t=0 (right after reset)
        buffers: dict[int, dict] = {}
        # t=0 snapshot (early intercept candidate)
        if not _r3f_ready(env):
            buffers[0] = take_full_state(
                env,
                meta={"stage_hint": "reset", "traj_id": st.traj_id},
            )

        first_sw: int | None = None
        done = False
        info: dict = {}
        for a in acts:
            if done or env.t >= MAX_EP_STEPS:
                break
            # Capture state BEFORE this action if we might need it
            t_before = int(env.t)
            if t_before not in buffers and not _r3f_ready(env):
                buffers[t_before] = take_full_state(
                    env, meta={"stage_hint": "pre_action", "traj_id": st.traj_id}
                )
            _r, done, info = step_raw(env, a)
            audit = classify_contacts(env.env.sim)
            if first_sw is None and is_correct_sidewall(audit):
                first_sw = int(env.t)
                break  # stop mining after first sidewall; keep buffers up to here

        if first_sw is None:
            skipped.append({"traj_id": st.traj_id, "reason": "no_sidewall_in_horizon"})
            print(f"[mine] skip {st.traj_id[-36:]}: no sidewall in {MAX_EP_STEPS}", flush=True)
            continue

        for stage, off in STAGE_OFFSETS.items():
            # Snapshot at policy step = first_sw + off, but "before" the event:
            # first_sw is AFTER the action that created sidewall, so the pre-recatch
            # state for offset 0 is the state at t=first_sw-1 (before that action),
            # and negative offsets go further back.
            # STAGE_OFFSETS["before_bilateral_recatch"]=0 → take at first_sw-1
            target_t = int(first_sw - 1 + off)
            if target_t < 0:
                continue
            if target_t not in buffers:
                skipped.append(
                    {"traj_id": st.traj_id, "stage": stage, "reason": f"missing_t={target_t}"}
                )
                continue
            state = buffers[target_t]
            rem = int(state["remaining_steps"])
            if rem < MIN_REMAINING_STEPS:
                skipped.append(
                    {"traj_id": st.traj_id, "stage": stage, "reason": f"remaining={rem}"}
                )
                continue
            # Exclude already R3F-ready / already stable
            if int(state.get("recatch", {}).get("stable_hold") or 0):
                skipped.append({"traj_id": st.traj_id, "stage": stage, "reason": "already_stable"})
                continue
            # Check r3f blob
            r3f = state.get("r3f") or {}
            # conservative: skip if export says ready
            if bool((r3f.get("tracker") or {}).get("ready")):
                skipped.append({"traj_id": st.traj_id, "stage": stage, "reason": "r3f_ready"})
                continue

            sid = f"{st.traj_id}__{stage}_t{target_t}"
            # sanitize filename
            safe = sid.replace("/", "_").replace("\\", "_")[:180]
            path = STATE_DIR / f"{safe}.pkl"
            blob = dict(state)
            blob["meta"] = {
                **(state.get("meta") or {}),
                "traj_id": st.traj_id,
                "stage": stage,
                "offset": off,
                "first_sidewall_step": first_sw,
                "snapshot_t": target_t,
                "remaining_steps": rem,
            }
            path.write_bytes(pickle.dumps(blob))
            entry = {
                "id": safe,
                "path": str(path),
                "traj_id": st.traj_id,
                "stage": stage,
                "offset": off,
                "first_sidewall_step": first_sw,
                "snapshot_t": target_t,
                "remaining_steps": rem,
                "policy_step": int(state["policy_step"]),
                "max_steps": int(state["max_steps"]),
                "sim_time": float(state.get("sim_time") or 0),
            }
            manifest.append(entry)
            print(
                f"[mine] {stage} t={target_t} sw={first_sw} rem={rem} {st.traj_id[-28:]}",
                flush=True,
            )

    man = {
        "version": 1,
        "seed": SEED,
        "init_checkpoint": str(P64_STEP4_ACTOR),
        "n_states": len(manifest),
        "n_experts_scanned": len(starts),
        "stage_offsets": STAGE_OFFSETS,
        "min_remaining_steps": MIN_REMAINING_STEPS,
        "states": manifest,
        "skipped": skipped[:200],
        "n_skipped": len(skipped),
    }
    (OUT / "CURRICULUM_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )
    print(f"[mine] wrote {len(manifest)} snapshots, skipped={len(skipped)}", flush=True)
    return man


if __name__ == "__main__":
    mine_pre_recatch_snapshots()
