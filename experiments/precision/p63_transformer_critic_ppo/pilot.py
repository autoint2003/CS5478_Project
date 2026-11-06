"""Phase 5 — Controlled short PPO pilots (Branch A baseline, Branch B + expanded C)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    init_critic_from_actor,
    load_actor,
    load_critic,
    save_pair,
)
from experiments.precision.p63_transformer_critic_ppo.eval_dev import eval_actor, eval_p62_baseline, paired_delta
from experiments.precision.p63_transformer_critic_ppo.expand_c import load_expanded_c_starts
from experiments.precision.p63_transformer_critic_ppo.paths import (
    CKPT_DIR,
    GROUP_PILOT,
    OUT,
    P62_EPOCH3,
    PILOT_UPDATES,
    SEED,
)
from experiments.precision.p63_transformer_critic_ppo.policy_drift import measure_drift
from experiments.precision.p63_transformer_critic_ppo.ppo import make_optimizers, ppo_update
from experiments.precision.p63_transformer_critic_ppo.rollout import collect_schedule, make_env


def _load_warm_or_init(device):
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    warm = CKPT_DIR / "P63_critic_warm.pt"
    if warm.is_file():
        critic = load_critic(warm, actor, device)
        note = "loaded P63_critic_warm.pt"
    else:
        critic = init_critic_from_actor(actor).to(device)
        note = "fresh critic from actor (warm-start missing)"
    return actor, critic, book, rho, note


def run_branch(
    *,
    branch: str,
    schedule: list,
    device: str,
    n_updates: int = PILOT_UPDATES,
    n_traj: int = GROUP_PILOT,
    max_ics: int | None = None,
) -> dict:
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, critic, book, rho, warm_note = _load_warm_or_init(device_t)
    opt_a, opt_c = make_optimizers(actor, critic)
    env = make_env(book, rho)

    # Match budgets: use same max_ics * n_traj across branches when comparing
    ics = schedule if max_ics is None else schedule[:max_ics]
    history = []
    t_all = time.perf_counter()
    total_steps = 0
    total_eps = 0

    for u in range(n_updates):
        print(f"=== Branch {branch} update {u+1}/{n_updates} ({len(ics)} ICs × {n_traj}) ===", flush=True)
        batch, summary = collect_schedule(
            env, actor, critic, ics, device_t, n_traj=n_traj
        )
        metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
        total_steps += summary["n_steps"]
        total_eps += summary["n_episodes"]
        row = {
            "update": u + 1,
            "rollout": summary,
            "ppo": metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "branch": branch,
                    "update": u + 1,
                    "p_rel": summary["overall_p_relaxed"],
                    "ev": summary["explained_var"],
                    "approx_kl": metrics.get("approx_kl"),
                    "clipfrac": metrics.get("clipfrac"),
                },
                indent=2,
            ),
            flush=True,
        )
        # Early abort on NaN / explosion
        if not np.isfinite(metrics.get("pg", 0)) or not np.isfinite(metrics.get("vf", 0)):
            break
        if metrics.get("approx_kl", 0) > 0.5:
            print("ABORT: approx_kl exploded", flush=True)
            break

    wall = time.perf_counter() - t_all
    actor_path = CKPT_DIR / f"P63_pilot_{branch}_actor.pt"
    critic_path = CKPT_DIR / f"P63_pilot_{branch}_critic.pt"
    save_pair(
        actor_path,
        critic_path,
        actor,
        critic,
        book,
        rho,
        epoch=n_updates,
        extra={"branch": branch, "pilot": True},
    )

    print(f"=== Dev eval Branch {branch} ===", flush=True)
    base = eval_p62_baseline(device=device)
    cur = eval_actor(f"P63_pilot_{branch}", actor_path, device=device)
    paired = paired_delta(base, cur)
    drift = measure_drift(actor_path, device=device, tag=f"pilot_{branch}")

    gpu_h = wall / 3600.0
    delta_msp = float(cur["MSP_RELAXED"] - base["MSP_RELAXED"])
    result = {
        "branch": branch,
        "warm_note": warm_note,
        "n_ics": len(ics),
        "n_traj_per_ic": n_traj,
        "n_updates": len(history),
        "total_episodes": total_eps,
        "total_steps": total_steps,
        "wall_s": wall,
        "gpu_hours_approx": gpu_h,
        "delta_msp": delta_msp,
        "msp_per_gpu_hour": delta_msp / gpu_h if gpu_h > 0 else None,
        "baseline": {k: v for k, v in base.items() if k != "per_ic"},
        "pilot": {k: v for k, v in cur.items() if k != "per_ic"},
        "paired_vs_p62": paired,
        "drift": drift,
        "history": history,
        "actor": str(actor_path),
        "critic": str(critic_path),
    }
    (OUT / f"PILOT_{branch}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_pilots(*, device: str = "cuda", n_updates: int = PILOT_UPDATES, max_ics: int | None = 48) -> dict:
    """Short stability pilots before any long run.

    Default ``max_ics=48`` keeps the pilot affordable while covering all classes.
    Matched budget: Branch B uses the same IC count (replace/append C within budget).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ds = build_p54_dataset(seed=SEED)
    base_schedule = list(ds["schedule"])
    if max_ics is not None:
        # stratified subset for pilot
        by = {"A1": [], "B1": [], "C": [], "D": []}
        for st in base_schedule:
            by[st.skill].append(st)
        per = max(1, max_ics // 4)
        schedule_a = []
        for sk in ("A1", "B1", "C", "D"):
            schedule_a.extend(by[sk][:per])
        schedule_a = schedule_a[:max_ics]
    else:
        schedule_a = base_schedule

    print("=== Branch A: PPO baseline ===", flush=True)
    res_a = run_branch(
        branch="A",
        schedule=schedule_a,
        device=device,
        n_updates=n_updates,
        max_ics=None,
    )

    # Branch B: same total IC count; swap in expanded C
    print("=== Branch B: PPO + expanded C (matched IC budget) ===", flush=True)
    expanded = load_expanded_c_starts()
    # Keep non-C from schedule_a; replace C with mix of original C + expanded
    non_c = [s for s in schedule_a if s.skill != "C"]
    orig_c = [s for s in schedule_a if s.skill == "C"]
    n_c = len(orig_c)
    # half expanded if available
    n_new = min(len(expanded), max(1, n_c // 2))
    c_mix = orig_c[: max(0, n_c - n_new)] + expanded[:n_new]
    schedule_b = non_c + c_mix
    # match length
    schedule_b = schedule_b[: len(schedule_a)]
    while len(schedule_b) < len(schedule_a) and expanded:
        schedule_b.append(expanded[len(schedule_b) % len(expanded)])

    res_b = run_branch(
        branch="B",
        schedule=schedule_b,
        device=device,
        n_updates=n_updates,
        max_ics=None,
    )

    summary = {
        "A": {k: res_a[k] for k in res_a if k != "history"},
        "B": {k: res_b[k] for k in res_b if k != "history"},
        "matched_n_ics": len(schedule_a),
        "branch_b_n_new_c": n_new,
    }
    (OUT / "PILOT_RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_pilot_md(res_a, res_b, summary)
    return summary


def _write_pilot_md(res_a, res_b, summary):
    def row(name, r):
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

    base = res_a["baseline"]
    bbc = base.get("by_class") or {}
    lines = [
        "# PILOT_RESULTS",
        "",
        "Short PPO pilots from P62_epoch3 (not a full training run).",
        "",
        f"Matched IC budget: **{summary['matched_n_ics']}** ICs × {GROUP_PILOT} traj × {PILOT_UPDATES} updates.",
        f"Branch B inserted **{summary['branch_b_n_new_c']}** expanded C ICs within that budget.",
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
        row("PPO-A", res_a),
        row("PPO-B (+C)", res_b),
        "",
        "## Efficiency / stability",
        "",
        f"- A wall_s={res_a['wall_s']:.0f}, steps={res_a['total_steps']}, paired Δ={res_a['paired_vs_p62']}",
        f"- B wall_s={res_b['wall_s']:.0f}, steps={res_b['total_steps']}, paired Δ={res_b['paired_vs_p62']}",
        "",
        "Policy anchoring ablation was **not** auto-enabled; see drift JSON if A/B/D regress.",
        "",
        "**Decision gate:** only promote to long PPO if pilot is numerically stable and shows "
        "credible MSP gain or clear efficiency advantage at comparable performance.",
        "",
    ]
    (OUT / "PILOT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=PILOT_UPDATES)
    ap.add_argument("--max-ics", type=int, default=48)
    args = ap.parse_args()
    print(json.dumps(run_pilots(device=args.device, n_updates=args.updates, max_ics=args.max_ics), indent=2)[:3000])
