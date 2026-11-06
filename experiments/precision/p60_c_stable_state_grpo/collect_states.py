"""Phase 2 — Collect stable-but-failed C intermediate states from train ICs only."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import take_full_state
from experiments.precision.p60_c_stable_state_grpo.state_labels import classify_capture_and_future
from experiments.precision.p60_c_stable_state_grpo.paths import (
    COLLECT_ROLLOUTS_PER_IC,
    MAX_EP_STEPS,
    MAX_STATES_PER_IC,
    MIN_REMAINING_STEPS,
    N_STATES_MAX,
    N_STATES_MIN,
    N_STATES_TARGET,
    OUT,
    P59_MANIFEST,
    P59_PASS1,
    SEED,
    STATE_DIR,
    T_MAX,
    TEMPERATURE,
)
from experiments.precision.p60_c_stable_state_grpo.select_checkpoint import select_init_checkpoint


def _dedupe_key(m: dict, step: int) -> tuple:
    return (
        int(round(float(m.get("e_pos_mm") or 0) / 8.0)),
        int(round(float(m.get("theta") or 0) / 8.0)),
        int(m.get("min_pad") or 0),
        int(round(float(m.get("vrel") or 0) / 0.1)),
        int(step // 8),
    )


# Post-stable offsets (policy steps after first stable_hold) for multi-time capture
_POST_STABLE_OFFSETS = (0, 4, 10)


def _r3f_ready(env: StabilizeGrpoEnv) -> bool:
    """True if R3F dwell already satisfies ready (exclude from training starts)."""
    if env.mon is None or env.mon.tracker is None:
        return False
    # Peek without mutating: use tracker ok_s vs dwell_s
    t = env.mon.tracker
    return bool(t._active and float(t.ok_s) + 1e-12 >= float(t.dwell_s))


def _rollout_harvest(
    env: StabilizeGrpoEnv,
    actor,
    start,
    device,
    *,
    temperature: float,
) -> list[dict[str, Any]]:
    """One stochastic rollout; if fails after stable hold, return multi-time capture points."""
    env.reset(start)
    done = False
    first_stable_step: int | None = None
    captures: dict[int, dict[str, Any]] = {}
    while not done:
        if (
            env.diag.get("stable_hold")
            and not env.diag.get("lost_after_recatch")
            and env.t + MIN_REMAINING_STEPS <= env.max_steps
        ):
            if first_stable_step is None:
                first_stable_step = int(env.t)
            if first_stable_step is not None:
                for off in _POST_STABLE_OFFSETS:
                    target = first_stable_step + int(off)
                    if int(env.t) == target and target not in captures:
                        # Filter states already fully R3F-ready
                        if _r3f_ready(env):
                            continue
                        captures[target] = take_full_state(env)
        tok, resid, _, _ = sample_step(
            actor, env.obs_seq, env.prev_seq, device, temperature=temperature
        )
        _r, done, info = env.step(tok, resid)

    if not captures:
        return []
    success = bool(info.get("SUCCESS_RELAXED_12S"))
    if success:
        return []
    cd = info.get("c_diag") or {}
    if not cd.get("recatch") or not cd.get("stable_hold"):
        return []
    handoff = bool(info.get("handoff"))
    if handoff and success:
        return []

    cands = []
    for step_t, cap in captures.items():
        labels = classify_capture_and_future(
            cap["metrics"],
            lost_after=bool(cd.get("lost_after_recatch")),
            truncated=bool(info.get("truncated")),
            handoff=handoff,
            success=False,
            drop=bool(info.get("drop")),
        )
        # Primary training bucket = capture geometry (not future outcome)
        cap["failure_pattern"] = labels["primary_capture_label"]
        cap["capture_labels"] = labels["capture_labels"]
        cap["future_labels"] = labels["future_labels"]
        cap["primary_future_label"] = labels["primary_future_label"]
        cap["label_bundle"] = labels
        cap["meta"] = {
            "final_handoff": int(handoff),
            "final_drop": int(info.get("drop") or 0),
            "final_truncated": int(info.get("truncated") or 0),
            "final_success_relaxed": 0,
            "first_stable_step": first_stable_step,
            "capture_step": int(step_t),
            "post_stable_offset": int(step_t) - int(first_stable_step or step_t),
            "traj_fail_reason": info.get("fail_reason") or info.get("handoff_reason"),
            "r3_geometry_ok": labels["r3_geometry_ok"],
        }
        cands.append(cap)
    return cands


def collect_states(*, device: str = "cuda", ckpt: Path | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    sel = select_init_checkpoint()
    ckpt_path = Path(ckpt) if ckpt else Path(sel["path"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    man = json.loads(P59_MANIFEST.read_text(encoding="utf-8"))
    train_ids = list(man["all"])
    # Exclude nothing from val here — train ICs only (manifest already train)
    ds = build_p54_dataset(seed=SEED)
    lookup = {s.traj_id: s for s in ds["schedule"] if s.skill == "C"}
    starts = [lookup[tid] for tid in train_ids if tid in lookup]

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(ckpt_path, device_t)
    actor.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    rng = np.random.default_rng(SEED + 60)

    pool: list[dict[str, Any]] = []
    per_ic_count: dict[str, int] = {}
    for i, st in enumerate(starts):
        print(f"[collect {i+1}/{len(starts)}] {st.traj_id}", flush=True)
        got_ic = 0
        for ri in range(COLLECT_ROLLOUTS_PER_IC):
            if got_ic >= MAX_STATES_PER_IC:
                break
            torch.manual_seed(SEED + 6000 + i * 100 + ri)
            cands = _rollout_harvest(env, actor, st, device_t, temperature=TEMPERATURE)
            for c in cands:
                if got_ic >= MAX_STATES_PER_IC:
                    break
                if int(c["remaining_steps"]) < MIN_REMAINING_STEPS:
                    continue
                # Must have recatch+stable at capture
                if not c["diag"].get("recatch") or not c["diag"].get("stable_hold"):
                    continue
                pool.append(
                    {
                        "ic_id": st.traj_id,
                        "rollout_i": ri,
                        "state": c,
                        "dedupe": _dedupe_key(c["metrics"], c["policy_step"]),
                        "pattern": c["failure_pattern"],
                    }
                )
                got_ic += 1
        per_ic_count[st.traj_id] = got_ic

    # Diversity selection across patterns / ICs / dedupe
    by_pattern: dict[str, list] = {}
    for p in pool:
        by_pattern.setdefault(p["pattern"], []).append(p)
    for ph in by_pattern:
        rng.shuffle(by_pattern[ph])

    selected: list[dict] = []
    seen_dedupe: set = set()
    seen_ic: dict[str, int] = {}
    patterns_order = [
        "angular_error",
        "position_error",
        "contact_quality",
        "r3_geometry_ok_dwell_incomplete",
        "other_gate_fail",
    ]
    # Round-robin until target
    while len(selected) < N_STATES_TARGET:
        progressed = False
        for pat in patterns_order:
            bucket = by_pattern.get(pat) or []
            while bucket:
                h = bucket.pop()
                if h["dedupe"] in seen_dedupe:
                    continue
                if seen_ic.get(h["ic_id"], 0) >= MAX_STATES_PER_IC:
                    continue
                seen_dedupe.add(h["dedupe"])
                seen_ic[h["ic_id"]] = seen_ic.get(h["ic_id"], 0) + 1
                selected.append(h)
                progressed = True
                break
            if len(selected) >= N_STATES_TARGET:
                break
        if not progressed:
            break

    # If short, fill from remaining pool ignoring pattern order
    if len(selected) < N_STATES_MIN:
        rng.shuffle(pool)
        for h in pool:
            if len(selected) >= N_STATES_MIN:
                break
            if h["dedupe"] in seen_dedupe:
                continue
            if seen_ic.get(h["ic_id"], 0) >= MAX_STATES_PER_IC:
                continue
            seen_dedupe.add(h["dedupe"])
            seen_ic[h["ic_id"]] = seen_ic.get(h["ic_id"], 0) + 1
            selected.append(h)

    selected = selected[:N_STATES_MAX]

    manifest_states = []
    for j, h in enumerate(selected):
        sid = f"S{j:02d}"
        path = STATE_DIR / f"{sid}.pkl"
        pickle.dump(h["state"], path.open("wb"))
        m = h["state"]["metrics"]
        manifest_states.append(
            {
                "state_id": sid,
                "ic_id": h["ic_id"],
                "path": str(path),
                "policy_step": h["state"]["policy_step"],
                "remaining_steps": h["state"]["remaining_steps"],
                "sim_time": h["state"]["sim_time"],
                "failure_pattern": h["pattern"],
                "primary_capture_label": h["pattern"],
                "capture_labels": h["state"].get("capture_labels"),
                "future_labels": h["state"].get("future_labels"),
                "primary_future_label": h["state"].get("primary_future_label"),
                "e_pos_mm": m.get("e_pos_mm"),
                "theta": m.get("theta"),
                "min_pad": m.get("min_pad"),
                "vrel": m.get("vrel"),
                "bilateral": m.get("bilateral"),
                "rollout_i": h["rollout_i"],
                "capture_step": (h["state"].get("meta") or {}).get("capture_step"),
                "post_stable_offset": (h["state"].get("meta") or {}).get("post_stable_offset"),
            }
        )
        print(
            f"  keep {sid} ic={h['ic_id'][:40]} pat={h['pattern']} "
            f"step={h['state']['policy_step']} rem={h['state']['remaining_steps']}",
            flush=True,
        )

    by_pat_count = {}
    for s in manifest_states:
        by_pat_count[s["failure_pattern"]] = by_pat_count.get(s["failure_pattern"], 0) + 1

    man_out = {
        "init_checkpoint": sel,
        "n_states": len(manifest_states),
        "n_pool_candidates": len(pool),
        "train_ics_scanned": len(starts),
        "per_ic_harvested": per_ic_count,
        "pattern_counts": by_pat_count,
        "min_remaining_steps": MIN_REMAINING_STEPS,
        "states": manifest_states,
        "note": "Train ICs only (P59 TRAIN_IC_MANIFEST). No val/held-out ICs. No synthetic ICs.",
    }
    (OUT / "STATE_SELECTION_MANIFEST.json").write_text(json.dumps(man_out, indent=2), encoding="utf-8")
    print(f"selected {len(manifest_states)} states; patterns={by_pat_count}", flush=True)
    return man_out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    collect_states(device=args.device)
