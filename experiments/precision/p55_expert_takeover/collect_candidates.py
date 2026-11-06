"""Phase 2 — Harvest diverse mid-trajectory states from policy rollouts on 0/64 ICs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import take_branch_state
from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p55_expert_takeover.audit_zero import load_zero_ics
from experiments.precision.p55_expert_takeover.ic_resolve import resolve_starts
from experiments.precision.p55_expert_takeover.paths import (
    CAPTURE_DIR,
    CAPTURE_STRIDE,
    MAX_CANDIDATES_PER_IC,
    MAX_EP_STEPS,
    N_PROBE_ROLLOUTS,
    OUT,
    P54_CKPT,
    SEED,
    T_MAX,
    TEMPERATURE,
)


def _phase_bucket(step: int, max_steps: int) -> str:
    f = step / max(max_steps, 1)
    if f < 0.33:
        return "early"
    if f < 0.66:
        return "mid"
    return "late"


def _dedupe_key(m: dict) -> tuple:
    """Coarse physical fingerprint for near-duplicate rejection."""
    return (
        int(m.get("bilateral") or 0),
        int(round(float(m.get("e_pos_mm") or 0) / 5.0)),
        int(round(float(m.get("theta") or 0) / 5.0)),
        int(m.get("min_pad") or 0),
        int(bool(m.get("latched"))),
    )


def collect_for_ic(
    env: GrpoRecoveryEnv,
    actor,
    start,
    device,
    *,
    n_rollouts: int,
    max_cand: int,
    stride: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Return up to max_cand diverse branch-state dicts (in-memory; snap pickled separately)."""
    harvested: list[dict] = []
    for ri in range(n_rollouts):
        env.reset(start)
        done = False
        step = 0
        caps = []
        while not done:
            if step > 0 and step % stride == 0 and env.mon is not None:
                m = state_metrics(env.env.sim)
                # Prefer before irreversible drop
                if not m.get("latched"):
                    st = take_branch_state(env.env, env.mon, step=step, entropy=0.0, token=-1)
                    caps.append(
                        {
                            "rollout_i": ri,
                            "step": step,
                            "phase": _phase_bucket(step, env.max_steps),
                            "metrics": m,
                            "branch": st,
                            "dedupe": _dedupe_key(m),
                        }
                    )
            tok, resid, _, _ = sample_step(
                actor, env.obs_seq, env.prev_seq, device, temperature=TEMPERATURE
            )
            _r, done, info = env.step(tok, resid)
            step += 1
            if info.get("drop"):
                break
        harvested.extend(caps)

    # Diversity: one per phase when possible, cover different rollouts / dedupe keys
    by_phase: dict[str, list] = {"early": [], "mid": [], "late": []}
    for h in harvested:
        by_phase[h["phase"]].append(h)
    for ph in by_phase:
        rng.shuffle(by_phase[ph])

    selected = []
    seen = set()
    # Round-robin phases
    while len(selected) < max_cand:
        progressed = False
        for ph in ("early", "mid", "late"):
            while by_phase[ph]:
                h = by_phase[ph].pop()
                if h["dedupe"] in seen:
                    continue
                # For C prioritize low bilateral / mid-late; still keep diversity
                seen.add(h["dedupe"])
                selected.append(h)
                progressed = True
                break
            if len(selected) >= max_cand:
                break
        if not progressed:
            break
    return selected


def run_collect(*, device: str = "cuda", skills: list[str] | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    zeros = load_zero_ics()
    if skills:
        zeros = [r for r in zeros if r["skill"] in skills]
    ic_ids = [r["ic_id"] for r in zeros]
    starts = resolve_starts(ic_ids)
    missing = [t for t in ic_ids if t not in starts]
    if missing:
        print("WARN missing starts", len(missing), missing[:5], flush=True)

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P54_CKPT, device_t)
    actor.eval()
    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )
    rng = np.random.default_rng(SEED)

    candidates = []
    for i, row in enumerate(zeros):
        tid = row["ic_id"]
        if tid not in starts:
            continue
        start = starts[tid]
        print(f"[{i+1}/{len(zeros)}] collect {start.skill} {tid}", flush=True)
        sel = collect_for_ic(
            env,
            actor,
            start,
            device_t,
            n_rollouts=N_PROBE_ROLLOUTS,
            max_cand=MAX_CANDIDATES_PER_IC,
            stride=CAPTURE_STRIDE,
            rng=rng,
        )
        for j, h in enumerate(sel):
            cap_path = CAPTURE_DIR / f"{start.skill}__{tid.replace('/', '_')}__c{j}.pkl"
            # Persist branch without huge duplication in JSON
            pickle.dump(h["branch"], cap_path.open("wb"))
            candidates.append(
                {
                    "candidate_id": f"{tid}::c{j}",
                    "ic_id": tid,
                    "skill": start.skill,
                    "class": {"A1": "A", "B1": "B", "C": "C", "D": "D"}[start.skill],
                    "rollout_i": h["rollout_i"],
                    "step": h["step"],
                    "phase": h["phase"],
                    "metrics": {
                        k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer, bool, np.bool_)) else v)
                        for k, v in h["metrics"].items()
                    },
                    "capture_path": str(cap_path),
                    "dedupe": list(h["dedupe"]),
                }
            )

    out = {
        "n_zero_ics": len(zeros),
        "n_candidates": len(candidates),
        "n_probe_rollouts": N_PROBE_ROLLOUTS,
        "max_per_ic": MAX_CANDIDATES_PER_IC,
        "stride": CAPTURE_STRIDE,
        "ckpt": str(P54_CKPT),
        "candidates": candidates,
        "missing_starts": missing,
    }
    (OUT / "TAKEOVER_CANDIDATES.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("candidates", len(candidates), flush=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skills", nargs="*", default=None)
    args = ap.parse_args()
    run_collect(device=args.device, skills=args.skills)
