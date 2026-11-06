"""Phase 5 — Sequential short pilots A → B → C (matched sim budget)."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    init_critic_from_actor,
    load_actor,
    load_critic,
    save_pair,
)
from experiments.precision.p63_transformer_critic_ppo.eval_dev import (
    eval_actor,
    eval_p62_baseline,
    paired_delta,
)
from experiments.precision.p63_transformer_critic_ppo.policy_drift import measure_drift
from experiments.precision.p63_transformer_critic_ppo.ppo import make_optimizers, ppo_update
from experiments.precision.p63_transformer_ppo_c_grasp.dataset import (
    schedule_branch_a,
    schedule_branch_b,
    schedule_branch_c,
)
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p63_transformer_ppo_c_grasp.paths import (
    CKPT_DIR,
    CRITIC_WARM,
    GROUP,
    OUT,
    P62_EPOCH3,
    PILOT_ICS,
    PILOT_UPDATES,
)
from experiments.precision.p63_transformer_ppo_c_grasp.rollout import collect_schedule


def _load_warm_or_init(device):
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    if CRITIC_WARM.is_file():
        critic = load_critic(CRITIC_WARM, actor, device)
        note = f"loaded {CRITIC_WARM.name}"
    else:
        critic = init_critic_from_actor(actor).to(device)
        note = "fresh critic from actor (warm-start missing)"
    return actor, critic, book, rho, note


def run_branch(
    *,
    branch: str,
    schedule: list,
    enable_c_grasp_reward: bool,
    device: str,
    n_updates: int = PILOT_UPDATES,
    n_traj: int = GROUP,
) -> dict:
    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, critic, book, rho, warm_note = _load_warm_or_init(device_t)
    opt_a, opt_c = make_optimizers(actor, critic)
    env = make_env(book, rho, enable_c_grasp_reward=enable_c_grasp_reward)

    history = []
    t_all = time.perf_counter()
    total_steps = 0
    total_eps = 0

    for u in range(n_updates):
        print(
            f"=== Branch {branch} update {u+1}/{n_updates} "
            f"({len(schedule)} ICs × {n_traj}, c_reward={enable_c_grasp_reward}) ===",
            flush=True,
        )
        batch, summary = collect_schedule(
            env, actor, critic, schedule, device_t, n_traj=n_traj
        )
        metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
        total_steps += summary["n_steps"]
        total_eps += summary["n_episodes"]
        row = {"update": u + 1, "rollout": summary, "ppo": metrics}
        history.append(row)
        print(
            json.dumps(
                {
                    "branch": branch,
                    "update": u + 1,
                    "p_rel": summary["overall_p_relaxed"],
                    "ev": summary["explained_var"],
                    "mean_r_c": summary.get("mean_r_c"),
                    "approx_kl": metrics.get("approx_kl"),
                    "clipfrac": metrics.get("clipfrac"),
                },
                indent=2,
            ),
            flush=True,
        )
        if not np.isfinite(metrics.get("pg", 0)) or not np.isfinite(metrics.get("vf", 0)):
            print("ABORT: non-finite PPO loss", flush=True)
            break
        if metrics.get("approx_kl", 0) > 0.5:
            print("ABORT: approx_kl exploded", flush=True)
            break

    wall = time.perf_counter() - t_all
    actor_path = CKPT_DIR / f"P63b_pilot_{branch}_actor.pt"
    critic_path = CKPT_DIR / f"P63b_pilot_{branch}_critic.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_pair(
        actor_path,
        critic_path,
        actor,
        critic,
        book,
        rho,
        epoch=n_updates,
        extra={
            "branch": branch,
            "pilot": True,
            "enable_c_grasp_reward": enable_c_grasp_reward,
            "experiment": "p63_transformer_ppo_c_grasp",
        },
    )

    print(f"=== Dev eval Branch {branch} ===", flush=True)
    base = eval_p62_baseline(device=device)
    cur = eval_actor(f"P63b_pilot_{branch}", actor_path, device=device)
    paired = paired_delta(base, cur)
    drift = measure_drift(actor_path, device=device, tag=f"p63b_pilot_{branch}")

    gpu_h = wall / 3600.0
    delta_msp = float(cur["MSP_RELAXED"] - base["MSP_RELAXED"])
    result = {
        "branch": branch,
        "enable_c_grasp_reward": enable_c_grasp_reward,
        "warm_note": warm_note,
        "n_ics": len(schedule),
        "n_traj_per_ic": n_traj,
        "n_updates": len(history),
        "total_episodes": total_eps,
        "total_steps": total_steps,
        "wall_s": wall,
        "gpu_hours_approx": gpu_h,
        "delta_msp": delta_msp,
        "msp_per_gpu_hour": delta_msp / gpu_h if gpu_h > 0 else None,
        "msp_per_1k_traj": delta_msp / (total_eps / 1000.0) if total_eps else None,
        "baseline": {k: v for k, v in base.items() if k != "per_ic"},
        "pilot": {k: v for k, v in cur.items() if k != "per_ic"},
        "paired_vs_p62": paired,
        "drift": drift,
        "history": history,
        "actor": str(actor_path),
        "critic": str(critic_path),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"PILOT_{branch}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _stable(res: dict) -> bool:
    if not res.get("history"):
        return False
    for h in res["history"]:
        ppo = h.get("ppo") or {}
        if not np.isfinite(ppo.get("pg", 0)) or not np.isfinite(ppo.get("vf", 0)):
            return False
        if ppo.get("approx_kl", 0) > 0.5:
            return False
    # Soft: MSP should not collapse vs baseline by >0.08 absolute
    base = float(res["baseline"].get("MSP_RELAXED") or 0)
    cur = float(res["pilot"].get("MSP_RELAXED") or 0)
    if cur < base - 0.08:
        return False
    return True


def run_pilots(
    *,
    device: str = "cuda",
    n_updates: int = PILOT_UPDATES,
    max_ics: int = PILOT_ICS,
    only: str | None = None,
) -> dict:
    """Run Branch A first; B/C only if A is stable. ``only`` in {A,B,C,None}."""
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    summary: dict = {
        "matched_n_ics": max_ics,
        "n_traj": GROUP,
        "n_updates": n_updates,
        "matched_episodes_per_branch": max_ics * GROUP * n_updates,
    }

    if only in (None, "A"):
        sched_a = schedule_branch_a(max_ics=max_ics)
        print("=== Branch A: PPO baseline (original reward, original C) ===", flush=True)
        results["A"] = run_branch(
            branch="A",
            schedule=sched_a,
            enable_c_grasp_reward=False,
            device=device,
            n_updates=n_updates,
        )
        summary["A_stable"] = _stable(results["A"])
        if only == "A":
            _write_summary(results, summary)
            return {"results": {k: _slim(v) for k, v in results.items()}, "summary": summary}
        if not summary["A_stable"]:
            print("Branch A unstable — skipping B/C per decision rules.", flush=True)
            _write_summary(results, summary)
            return {"results": {k: _slim(v) for k, v in results.items()}, "summary": summary}

    if only in (None, "B"):
        sched_b, n_new = schedule_branch_b(max_ics=max_ics)
        summary["branch_b_n_new_c"] = n_new
        print("=== Branch B: +new C ICs, original reward ===", flush=True)
        results["B"] = run_branch(
            branch="B",
            schedule=sched_b,
            enable_c_grasp_reward=False,
            device=device,
            n_updates=n_updates,
        )

    if only in (None, "C"):
        sched_c, n_new = schedule_branch_c(max_ics=max_ics)
        summary["branch_c_n_new_c"] = n_new
        print("=== Branch C: +new C ICs + C grasp event reward ===", flush=True)
        results["C"] = run_branch(
            branch="C",
            schedule=sched_c,
            enable_c_grasp_reward=True,
            device=device,
            n_updates=n_updates,
        )

    _write_summary(results, summary)
    return {"results": {k: _slim(v) for k, v in results.items()}, "summary": summary}


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "history"}


def _write_summary(results: dict, summary: dict):
    # Merge with any prior branch results so --only A/B/C does not wipe siblings.
    prev_path = OUT / "PILOT_RESULTS.json"
    prev_branches: dict = {}
    prev_summary: dict = {}
    if prev_path.is_file():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            prev_branches = dict(prev.get("branches") or {})
            prev_summary = dict(prev.get("summary") or {})
        except Exception:
            pass
    slim = {**prev_branches, **{k: _slim(v) for k, v in results.items()}}
    merged_summary = {**prev_summary, **summary}
    payload = {"summary": merged_summary, "branches": slim}
    (OUT / "PILOT_RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Use merged branches for markdown table
    results = {**{k: prev_branches[k] for k in prev_branches if k not in results}, **results}
    summary = merged_summary

    def row(name, r):
        if r is None:
            return f"| {name} | — | — | — | — | — | — | — | — |"
        p = r["pilot"]
        bc = p.get("by_class") or {}
        return (
            f"| {name} | {p.get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{p.get('zero_success_ics', '')} | {r.get('wall_s', 0)/3600:.2f} | "
            f"{r.get('msp_per_gpu_hour')} |"
        )

    base = next(iter(results.values()))["baseline"]
    bbc = base.get("by_class") or {}
    lines = [
        "# BRANCH_COMPARISON / PILOT_RESULTS",
        "",
        f"Matched budget: **{summary.get('matched_n_ics')}** ICs × "
        f"{summary.get('n_traj')} traj × {summary.get('n_updates')} updates "
        f"(≈{summary.get('matched_episodes_per_branch')} traj/branch).",
        f"Branch B/C new C slots: {summary.get('branch_b_n_new_c', summary.get('branch_c_n_new_c', '—'))}",
        f"A_stable={summary.get('A_stable')}",
        "",
        "| Branch | MSP | A | B | C | D | Zero | GPU-h | ΔMSP/GPU-h |",
        "|--------|----:|--:|--:|--:|--:|-----:|------:|-----------:|",
        (
            f"| P62_epoch3 | {base.get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bbc.get('A') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bbc.get('B') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bbc.get('C') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{(bbc.get('D') or {}).get('MSP_RELAXED', float('nan')):.4f} | "
            f"{base.get('zero_success_ics', '')} | — | — |"
        ),
        row("PPO-A", results.get("A")),
        row("PPO-B (+C data)", results.get("B")),
        row("PPO-C (+C reward)", results.get("C")),
        "",
        "## Stability notes",
        "",
        "- No long-run training in this phase.",
        "- Promote only if pilot is stable and sample-efficiency competitive with GRPO-64.",
        "",
    ]
    (OUT / "BRANCH_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "PILOT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=PILOT_UPDATES)
    ap.add_argument("--max-ics", type=int, default=PILOT_ICS)
    ap.add_argument("--only", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    out = run_pilots(
        device=args.device,
        n_updates=args.updates,
        max_ics=args.max_ics,
        only=args.only,
    )
    print(json.dumps({"summary": out["summary"], "branches": list(out["results"])}, indent=2))
