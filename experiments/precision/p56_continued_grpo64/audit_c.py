"""Phase 2 — C-class failure audit: 64 independent rollouts on every C train IC."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p56_continued_grpo64.paths import (
    GROUP,
    MAX_EP_STEPS,
    OUT,
    P55A_CKPT,
    SEED,
    T_MAX,
    TEMPERATURE,
)

EARLY_STEPS = 8


def _classify(trace: list[dict], info: dict) -> dict[str, Any]:
    """Map one C rollout into a failure/success bucket + first irreversible step."""
    srel = bool(info.get("SUCCESS_RELAXED_12S"))
    s12 = bool(info.get("SUCCESS_12S"))
    snew = bool(info.get("SUCCESS_NEW"))
    drop = bool(info.get("drop"))
    hand = bool(info.get("handoff"))
    fail = str(info.get("fail_reason") or "")
    n = len(trace)
    bil = [int(t.get("bilateral") or 0) for t in trace]
    latch_steps = [i for i, t in enumerate(trace) if t.get("latched")]
    first_latch = int(latch_steps[0]) if latch_steps else None

    ever_bil = any(bil)
    # Recatch: bilateral after a stretch of no bilateral
    regained = False
    saw_gap = False
    for b in bil:
        if b == 0:
            saw_gap = True
        elif saw_gap and b == 1:
            regained = True
            break
    # Unstable after contact: had bilateral then lost then dropped
    lost_after_bil = False
    had = False
    for b in bil:
        if b:
            had = True
        elif had and b == 0:
            lost_after_bil = True
            break

    if srel:
        bucket = "SUCCESS_RELAXED"
        irrev = None
    elif hand and s12 and not srel:
        bucket = "STABLE_INVALID_GEOMETRY"  # 12s ok under strict, fails RELAXED pose band
        irrev = n - 1
    elif hand and not s12:
        bucket = "FAILED_HANDOFF"
        irrev = n - 1
    elif fail == "TIMEOUT" or (not drop and not hand and n >= 1):
        # truncated without hand/drop
        if not drop and not hand:
            bucket = "TIMEOUT"
            irrev = n - 1
        elif drop and first_latch is not None and first_latch <= EARLY_STEPS and not ever_bil:
            bucket = "IMMEDIATE_CONTACT_LOSS"
            irrev = first_latch
        elif drop and regained and lost_after_bil:
            bucket = "UNSTABLE_AFTER_RECATCH"
            irrev = first_latch
        elif drop and ever_bil and lost_after_bil and not regained:
            bucket = "UNSTABLE_AFTER_RECATCH"
            irrev = first_latch
        elif drop and not ever_bil:
            if first_latch is not None and first_latch <= EARLY_STEPS:
                bucket = "IMMEDIATE_CONTACT_LOSS"
            else:
                bucket = "FAILED_RECATCH"
            irrev = first_latch
        elif drop:
            bucket = "FAILED_RECATCH"
            irrev = first_latch
        else:
            bucket = "TIMEOUT"
            irrev = n - 1
    elif drop:
        if first_latch is not None and first_latch <= EARLY_STEPS and (not ever_bil or sum(bil[: EARLY_STEPS + 1]) == 0):
            bucket = "IMMEDIATE_CONTACT_LOSS"
            irrev = first_latch
        elif regained and (lost_after_bil or (bil and bil[-1] == 0)):
            bucket = "UNSTABLE_AFTER_RECATCH"
            irrev = first_latch
        elif ever_bil and lost_after_bil:
            bucket = "UNSTABLE_AFTER_RECATCH"
            irrev = first_latch
        else:
            bucket = "FAILED_RECATCH"
            irrev = first_latch
    else:
        bucket = "TIMEOUT"
        irrev = n - 1

    # Partial-progress proxies
    min_e = float(min((t.get("e_pos_mm") or 1e9) for t in trace)) if trace else None
    max_bil_run = 0
    run = 0
    for b in bil:
        run = run + 1 if b else 0
        max_bil_run = max(max_bil_run, run)
    return {
        "bucket": bucket,
        "first_irreversible_step": irrev,
        "n_steps": n,
        "ever_bilateral": int(ever_bil),
        "regained_bilateral": int(regained),
        "lost_after_bilateral": int(lost_after_bil),
        "min_e_pos_mm": min_e,
        "max_bilateral_run": max_bil_run,
        "bilateral_frac": float(np.mean(bil)) if bil else 0.0,
        "SUCCESS_RELAXED_12S": int(srel),
        "SUCCESS_12S": int(s12),
        "SUCCESS_NEW": int(snew),
        "drop": int(drop),
        "handoff": int(hand),
        "fail_reason": fail or None,
        "final_e_pos_mm": float(trace[-1]["e_pos_mm"]) if trace else None,
        "final_theta": float(trace[-1]["theta"]) if trace else None,
        "final_bilateral": int(bil[-1]) if bil else 0,
    }


def rollout_traced(env, actor, start, device) -> dict:
    env.reset(start)
    trace = []
    done = False
    info: dict = {}
    while not done:
        m = state_metrics(env.env.sim)
        trace.append(
            {
                "step": env.t,
                "e_pos_mm": float(m["e_pos_mm"]),
                "theta": float(m["theta"]),
                "bilateral": int(m["bilateral"]),
                "nL": int(m["nL"]),
                "nR": int(m["nR"]),
                "min_pad": int(m["min_pad"]),
                "vrel": float(m["vrel"]),
                "obj_z": float(m["obj_z"]),
                "latched": int(m["latched"]),
            }
        )
        tok, resid, _, _ = sample_step(
            actor, env.obs_seq, env.prev_seq, device, temperature=TEMPERATURE
        )
        _r, done, info = env.step(tok, resid)
    # append terminal metrics
    m = state_metrics(env.env.sim)
    trace.append(
        {
            "step": env.t,
            "e_pos_mm": float(m["e_pos_mm"]),
            "theta": float(m["theta"]),
            "bilateral": int(m["bilateral"]),
            "nL": int(m["nL"]),
            "nR": int(m["nR"]),
            "min_pad": int(m["min_pad"]),
            "vrel": float(m["vrel"]),
            "obj_z": float(m["obj_z"]),
            "latched": int(m["latched"]),
        }
    )
    cls = _classify(trace, info)
    return {**cls, "info_keys": list(info.keys())}


def run_audit(*, device: str = "cuda", ckpt: Path | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = Path(ckpt or P55A_CKPT)
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(ckpt, device_t)
    actor.eval()
    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )
    ds = build_p54_dataset(seed=SEED)
    c_starts = [s for s in ds["schedule"] if s.skill == "C"]
    print(f"C train ICs: {len(c_starts)} × {GROUP} from {ckpt}", flush=True)

    per_ic = []
    global_buckets = Counter()
    t0 = time.perf_counter()
    for i, st in enumerate(c_starts):
        rows = []
        for _ in range(GROUP):
            rows.append(rollout_traced(env, actor, st, device_t))
        buckets = Counter(r["bucket"] for r in rows)
        global_buckets.update(buckets)
        n_succ = sum(r["SUCCESS_RELAXED_12S"] for r in rows)
        irrev = [r["first_irreversible_step"] for r in rows if r["first_irreversible_step"] is not None]
        # Partial progress among failures
        fails = [r for r in rows if not r["SUCCESS_RELAXED_12S"]]
        succ = [r for r in rows if r["SUCCESS_RELAXED_12S"]]
        partial = {
            "n_fail_ever_bilateral": int(sum(r["ever_bilateral"] for r in fails)),
            "n_fail_regained": int(sum(r["regained_bilateral"] for r in fails)),
            "n_fail_any_handoff": int(sum(r["handoff"] for r in fails)),
            "fail_min_e_pos_best": float(min((r["min_e_pos_mm"] or 1e9) for r in fails)) if fails else None,
            "fail_max_bil_run_best": int(max((r["max_bilateral_run"] for r in fails), default=0)),
            "succ_mean_e_final": float(np.mean([r["final_e_pos_mm"] for r in succ])) if succ else None,
            "fail_mean_e_final": float(np.mean([r["final_e_pos_mm"] for r in fails if r["final_e_pos_mm"] is not None]))
            if fails
            else None,
        }
        useful_partial = int(
            n_succ == 0
            and (
                partial["n_fail_regained"] > 0
                or partial["n_fail_any_handoff"] > 0
                or partial["fail_max_bil_run_best"] >= 5
                or (partial["fail_min_e_pos_best"] is not None and partial["fail_min_e_pos_best"] < 40.0)
            )
        )
        row = {
            "ic_id": st.traj_id,
            "n_rollouts": GROUP,
            "n_success_relaxed": n_succ,
            "zero_success": int(n_succ == 0),
            "p_relaxed": n_succ / GROUP,
            "buckets": dict(buckets),
            "mean_first_irreversible": float(np.mean(irrev)) if irrev else None,
            "partial_progress": partial,
            "useful_partial_despite_zero": useful_partial,
            "rollouts": rows,  # full detail
        }
        per_ic.append(row)
        print(
            f"[{i+1}/{len(c_starts)}] {st.traj_id}: succ={n_succ}/{GROUP} "
            f"buckets={dict(buckets)} partial0={useful_partial}",
            flush=True,
        )

    zero_ics = [r for r in per_ic if r["zero_success"]]
    summary = {
        "ckpt": str(ckpt),
        "n_c_ics": len(c_starts),
        "n_rollouts_total": len(c_starts) * GROUP,
        "global_buckets": dict(global_buckets),
        "global_bucket_frac": {k: v / max(len(c_starts) * GROUP, 1) for k, v in global_buckets.items()},
        "n_zero_success_ics": len(zero_ics),
        "n_zero_with_useful_partial": int(sum(r["useful_partial_despite_zero"] for r in zero_ics)),
        "mean_p_relaxed": float(np.mean([r["p_relaxed"] for r in per_ic])) if per_ic else 0.0,
        "elapsed_s": time.perf_counter() - t0,
        "per_ic": [{k: v for k, v in r.items() if k != "rollouts"} for r in per_ic],
    }
    # Compact JSON without every rollout; store detailed separately
    (OUT / "C_FAILURE_AUDIT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detail = {"per_ic_rollouts": {r["ic_id"]: r["rollouts"] for r in per_ic}}
    (OUT / "C_FAILURE_AUDIT_DETAIL.json").write_text(json.dumps(detail), encoding="utf-8")

    lines = [
        "# C_FAILURE_AUDIT",
        "",
        f"Policy: `{ckpt.name}` (P55-A). C train ICs from P54 schedule: **{len(c_starts)}** × 64.",
        f"Wall time: {summary['elapsed_s']:.0f}s. Total rollouts: {summary['n_rollouts_total']}.",
        "",
        f"- Mean p_relaxed: **{summary['mean_p_relaxed']:.4f}**",
        f"- Zero-success C ICs: **{summary['n_zero_success_ics']}** / {len(c_starts)}",
        f"- Zero-success ICs with useful partial progress: **{summary['n_zero_with_useful_partial']}**",
        "",
        "## Failure distribution (all C rollouts)",
        "",
        "| Bucket | Count | Fraction |",
        "|--------|------:|---------:|",
    ]
    for k, v in sorted(global_buckets.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} | {v / max(summary['n_rollouts_total'], 1):.3f} |")
    lines += [
        "",
        "## Bucket definitions",
        "",
        "- **IMMEDIATE_CONTACT_LOSS**: drop by step ≤8 with no useful bilateral contact.",
        "- **FAILED_RECATCH**: drop without durable bilateral recovery.",
        "- **UNSTABLE_AFTER_RECATCH**: bilateral regained (or held then lost) then drop.",
        "- **FAILED_HANDOFF**: R3F HANDOFF fired but SUCCESS_12S failed.",
        "- **STABLE_INVALID_GEOMETRY**: SUCCESS_12S but not SUCCESS_RELAXED_12S.",
        "- **TIMEOUT**: max steps without drop/handoff.",
        "- **SUCCESS_RELAXED**: terminal SUCCESS_RELAXED_12S.",
        "",
        "## Zero-success ICs with partial progress",
        "",
    ]
    for r in zero_ics:
        if not r["useful_partial_despite_zero"]:
            continue
        p = r["partial_progress"]
        lines.append(
            f"- `{r['ic_id']}`: regained={p['n_fail_regained']}, handoff_fail={p['n_fail_any_handoff']}, "
            f"best_bil_run={p['fail_max_bil_run_best']}, best_min_e={p['fail_min_e_pos_best']}"
        )
    if summary["n_zero_with_useful_partial"] == 0:
        lines.append("_None flagged under the partial-progress heuristics._")
    lines += [
        "",
        "## Implication for next intervention",
        "",
        "See FINAL_REPORT after training+eval. Dominant bucket among failures guides whether to target "
        "early exploration, recatch execution, handoff, terminal pose, or C IC coverage.",
        "",
    ]
    (OUT / "C_FAILURE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    run_audit(device=args.device, ckpt=Path(args.ckpt) if args.ckpt else None)
