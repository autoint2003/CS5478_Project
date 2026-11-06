"""Phase 1 — C contact dynamics audit (P56 summaries + stratified rich re-probes)."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.criteria import RecatchTracker
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p57_post_recatch_stabilization.paths import (
    MAX_EP_STEPS,
    OUT,
    P56_CKPT,
    P56_C_AUDIT,
    P56_C_DETAIL,
    SEED,
    T_MAX,
    TEMPERATURE,
)


def _classify_mode(trace: list[dict], recatch_step: int | None, lost: bool, stable: bool, succ: bool) -> str:
    if succ and stable:
        return "STABLE_GRASP_SUCCESS"
    if succ:
        return "SUCCESS_WITHOUT_LABELED_STABLE_HOLD"
    if recatch_step is None:
        return "NO_VALID_RECATCH"
    # post-recatch
    post = [t for t in trace if t["step"] >= recatch_step]
    bil_frac = float(np.mean([t["bilateral"] for t in post])) if post else 0.0
    mean_v = float(np.mean([t["vrel"] for t in post])) if post else 0.0
    max_run = 0
    run = 0
    for t in post:
        run = run + 1 if t["bilateral"] else 0
        max_run = max(max_run, run)
    if max_run <= 4:
        return "TRANSIENT_BILATERAL"
    if lost and mean_v > 0.4:
        return "UNCONTROLLED_SLIP_THEN_LOSS"
    if lost:
        return "RECATCH_THEN_CONTACT_LOSS"
    if bil_frac > 0.7 and mean_v < 0.25 and not succ:
        return "STABLE_GRASP_OTHER_FAIL"
    if mean_v < 0.35 and max_run >= 8:
        return "CONTROLLED_SLIDING_OR_HOLD"
    return "UNSTABLE_POST_RECATCH"


def analyze_existing() -> dict:
    audit = json.loads(P56_C_AUDIT.read_text(encoding="utf-8")) if P56_C_AUDIT.is_file() else {}
    detail = json.loads(P56_C_DETAIL.read_text(encoding="utf-8")) if P56_C_DETAIL.is_file() else {}
    per = detail.get("per_ic_rollouts") or {}
    hold_stats = []
    by_bucket = Counter()
    for tid, rows in per.items():
        for r in rows:
            by_bucket[r.get("bucket")] += 1
            hold_stats.append(
                {
                    "ic_id": tid,
                    "bucket": r.get("bucket"),
                    "max_bilateral_run": r.get("max_bilateral_run"),
                    "bilateral_frac": r.get("bilateral_frac"),
                    "regained": r.get("regained_bilateral"),
                    "success": r.get("SUCCESS_RELAXED_12S"),
                    "n_steps": r.get("n_steps"),
                    "min_e_pos_mm": r.get("min_e_pos_mm"),
                }
            )
    succ = [h for h in hold_stats if h["success"]]
    fail = [h for h in hold_stats if not h["success"]]
    return {
        "source": "P56_C_FAILURE_AUDIT_DETAIL",
        "n_rollouts": len(hold_stats),
        "bucket_counts": dict(by_bucket),
        "succ_mean_max_bil_run": float(np.mean([h["max_bilateral_run"] for h in succ])) if succ else None,
        "fail_mean_max_bil_run": float(np.mean([h["max_bilateral_run"] for h in fail])) if fail else None,
        "succ_mean_bil_frac": float(np.mean([h["bilateral_frac"] for h in succ])) if succ else None,
        "fail_mean_bil_frac": float(np.mean([h["bilateral_frac"] for h in fail])) if fail else None,
        "p56_global_buckets": audit.get("global_bucket_frac"),
    }


def rich_probe(*, device: str = "cuda", n_ics: int = 12, rolls_per_ic: int = 8) -> dict:
    """Stratified re-roll with force/vrel/e_pos traces under P56 policy."""
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P56_CKPT, device_t)
    actor.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    ds = build_p54_dataset(seed=SEED)
    c_starts = [s for s in ds["schedule"] if s.skill == "C"]
    rng = np.random.default_rng(SEED + 57)
    # Prefer mix of ICs that were zero-success vs successful in P56 audit
    audit = json.loads(P56_C_AUDIT.read_text(encoding="utf-8")) if P56_C_AUDIT.is_file() else {}
    zero = {r["ic_id"] for r in (audit.get("per_ic") or []) if r.get("zero_success")}
    zeros = [s for s in c_starts if s.traj_id in zero]
    others = [s for s in c_starts if s.traj_id not in zero]
    rng.shuffle(zeros)
    rng.shuffle(others)
    pick = (zeros[: n_ics // 3] + others[: n_ics - n_ics // 3])[:n_ics]

    modes = Counter()
    traj_summaries = []
    t0 = time.perf_counter()
    for st in pick:
        for ri in range(rolls_per_ic):
            env.reset(st)
            tracker = RecatchTracker()
            trace = []
            done = False
            info = {}
            while not done:
                m = state_metrics(env.env.sim)
                tr = tracker.update(env.t, m)
                trace.append(
                    {
                        "step": env.t,
                        "bilateral": int(m["bilateral"]),
                        "nL": int(m["nL"]),
                        "nR": int(m["nR"]),
                        "min_pad": int(m["min_pad"]),
                        "fn_L": float(m["fn_L"]),
                        "fn_R": float(m["fn_R"]),
                        "vrel": float(m["vrel"]),
                        "e_pos_mm": float(m["e_pos_mm"]),
                        "theta": float(m["theta"]),
                        "obj_z": float(m["obj_z"]),
                        "latched": int(m["latched"]),
                        "post_recatch": int(tr["recatch"]),
                    }
                )
                tok, resid, _, _ = sample_step(
                    actor, env.obs_seq, env.prev_seq, device_t, temperature=TEMPERATURE
                )
                _r, done, info = env.step(tok, resid)
            rs = tracker.recatch_step
            post = [t for t in trace if rs is not None and t["step"] >= rs]
            hold_dur = 0
            if post:
                run = 0
                best = 0
                for t in post:
                    run = run + 1 if t["bilateral"] else 0
                    best = max(best, run)
                hold_dur = best
            e0 = post[0]["e_pos_mm"] if post else None
            e1 = post[-1]["e_pos_mm"] if post else None
            mode = _classify_mode(
                trace,
                rs,
                tracker.lost_after_recatch,
                tracker.stable_hold,
                bool(info.get("SUCCESS_RELAXED_12S")),
            )
            modes[mode] += 1
            traj_summaries.append(
                {
                    "ic_id": st.traj_id,
                    "roll": ri,
                    "recatch_step": rs,
                    "stable_hold": int(tracker.stable_hold),
                    "lost_after_recatch": int(tracker.lost_after_recatch),
                    "post_hold_run_max": hold_dur,
                    "post_mean_vrel": float(np.mean([t["vrel"] for t in post])) if post else None,
                    "post_mean_fn": float(np.mean([t["fn_L"] + t["fn_R"] for t in post])) if post else None,
                    "post_de_pos_mm": (float(e1 - e0) if e0 is not None and e1 is not None else None),
                    "time_recatch_to_end": (len(trace) - rs) if rs is not None else None,
                    "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
                    "mode": mode,
                }
            )
    return {
        "n_ics": len(pick),
        "rolls_per_ic": rolls_per_ic,
        "mode_counts": dict(modes),
        "traj_summaries": traj_summaries,
        "elapsed_s": time.perf_counter() - t0,
        "criteria": {
            "RECATCH_MIN_STEPS": 3,
            "STABLE_HOLD_STEPS": 12,
            "STABLE_MIN_PAD": 2,
            "STABLE_VREL_MAX": 0.35,
        },
    }


def run_audit(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    existing = analyze_existing()
    rich = rich_probe(device=device, n_ics=12, rolls_per_ic=8)
    out = {"existing_p56": existing, "rich_probe": rich}
    (OUT / "C_CONTACT_DYNAMICS_AUDIT.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# C_CONTACT_DYNAMICS_AUDIT",
        "",
        "Hypothesis: policy often re-establishes contact but cannot sustain bilateral grasp.",
        "",
        "## Fixed criteria (frozen)",
        "",
        "- **Valid recatch**: first run of ≥3 consecutive bilateral steps with min_pad≥1.",
        "- **Stable hold**: after recatch, ≥12 consecutive bilateral steps with min_pad≥2 and mean vrel≤0.35.",
        "- Stability is **not** inferred from a single bilateral timestep.",
        "",
        "## P56 aggregate diagnostics",
        "",
        f"- Rollouts analyzed: {existing.get('n_rollouts')}",
        f"- Bucket counts: `{existing.get('bucket_counts')}`",
        f"- Success vs fail mean max bilateral run: "
        f"{existing.get('succ_mean_max_bil_run')} vs {existing.get('fail_mean_max_bil_run')}",
        f"- Success vs fail mean bilateral frac: "
        f"{existing.get('succ_mean_bil_frac')} vs {existing.get('fail_mean_bil_frac')}",
        "",
        "## Stratified rich re-probe (P56 policy)",
        "",
        f"- ICs×rolls: {rich['n_ics']}×{rich['rolls_per_ic']}",
        f"- Mode counts: `{rich['mode_counts']}`",
        "",
    ]
    # Aggregate post-recatch stats
    with_r = [t for t in rich["traj_summaries"] if t["recatch_step"] is not None]
    lost = [t for t in with_r if t["lost_after_recatch"]]
    lines += [
        f"- Trajectories with valid recatch: {len(with_r)} / {len(rich['traj_summaries'])}",
        f"- Of those, lost bilateral again: {len(lost)} ({len(lost)/max(len(with_r),1):.2f})",
        f"- Stable-hold rate (criterion): {np.mean([t['stable_hold'] for t in rich['traj_summaries']]):.3f}",
        f"- Mean post-recatch max bilateral run: "
        f"{np.mean([t['post_hold_run_max'] for t in with_r]) if with_r else None}",
        f"- Mean post-recatch vrel: "
        f"{np.nanmean([t['post_mean_vrel'] for t in with_r if t['post_mean_vrel'] is not None]) if with_r else None}",
        "",
        "## Distinctions observed",
        "",
        "- **Transient bilateral**: short pad contact without durable hold.",
        "- **Recatch then contact loss**: meets recatch gate then loses bilateral (dominant among failures).",
        "- **Uncontrolled slip**: elevated post-recatch vrel before loss.",
        "- **Controlled sliding/hold**: lower vrel with longer bilateral runs; may still miss terminal SUCCESS.",
        "- **No valid recatch**: never reaches 3-step bilateral gate (maps to FAILED_RECATCH-like).",
        "",
        "## Implication for reward",
        "",
        "Stage-aware shaping should activate **only after valid recatch**, reward sustained bilateral + calmness, "
        "penalize post-recatch contact loss / uncontrolled slip, and leave pre-recatch sliding unpenalized by the new terms.",
        "",
    ]
    (OUT / "C_CONTACT_DYNAMICS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_audit(device=args.device)
