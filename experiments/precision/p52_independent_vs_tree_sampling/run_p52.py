"""P52 — Independent vs Tree80-style branching under equal candidate budgets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import (  # noqa: E402
    restore_branch_state,
    take_branch_state,
)
from experiments.precision.p52_independent_vs_tree_sampling.paths import (  # noqa: E402
    BUDGET_CHECKPOINTS,
    CLASS_LABEL,
    M111_META,
    N_CANDIDATES,
    N_ENTROPY_PER_TRUNK,
    N_TRUNKS,
    N_UNIFORM_PER_TRUNK,
    OUT,
    P44_EVAL,
    P47T_CKPT,
    SEED,
    SKILLS,
    T_MAX,
)
from experiments.precision.p52_independent_vs_tree_sampling.search import (  # noqa: E402
    continue_from_branch,
    reset_ic,
    rollout_independent,
    sample_m111_token,
    search_independent,
    search_tree,
)
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from models.m111_checkpoint import load_m111, load_m111_book  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in ((kk, r.get(kk)) for kk in keys)})


def family_of_b(tid: str) -> str:
    t = tid.lower()
    for fam in ("ball", "slide", "k48r", "inward"):
        if fam in t:
            return fam
    return "other"


def write_audit() -> None:
    md = f"""# IMPLEMENTATION_AUDIT

Generated: {utc()}

## Tree80 (source of truth)

Code: `training/adaptive_grpo.py`, `training/adaptive_hybrid.py`, `training/rollout.py`.

| Aspect | Tree80 behavior |
|---|---|
| Trunks | Stage A 8 + Stage B 8 = **16** independent stochastic rollouts with per-step captures |
| Branch points | Per trunk: **2 uniform** + **2 entropy** positions among captured pre-action states |
| Branch count | 16×4 = **64** suffixes |
| Group size | 16 trunks + 64 branches = **G=80** (GRPO update) |
| Restore | `take_snapshot` / `restore_snapshot` (MjData copy + capture: FSM, hist, drop latch, obs windows) |
| Child actions | Fresh RNG; categorical token + residual noise (K48) |
| Terminal | `finish_recovery` / SUCCESS_12S in GRPO reward path |
| Policy | **K48**, not M111; trained GRPO weights historically |

Tree80 does **not** mean 80 independent rollouts. Branching only runs after independent stages fail (in training loop); for this experiment we always build trunks then branches.

## P52 adaptations (documented deltas)

| Item | Tree80 | P52 |
|---|---|---|
| Policy | K48 GRPO | **Frozen P47T M111** |
| HANDOFF | Sampled as token | **Masked**; **R3F** decides handoff |
| Residual | mean + N(0,σ) | M111 residual head (no extra noise) |
| Terminal eval | finish_recovery | `evaluate_handoff_with_axis` (SUCCESS_12S) |
| T_max | ~4s recovery limit | **R3F T_max={T_MAX}** |
| Candidate budget | G=80 | **64 terminals each method** |
| Tree composition | 16 trunks + 64 branches | **16 trunks + 48 branches** (2 uniform + 1 entropy / trunk) = 64 |
| Branch state | snapshot | snapshot **+ R3F dwell timers** |

Search mechanism (uniform/entropy branch allocation, prefix reuse, suffix resampling) follows Tree80. Weights and action space follow P47T M111.
"""
    write_md(OUT / "IMPLEMENTATION_AUDIT.md", md)


def _p47t_fail_ids() -> dict[str, set[str]]:
    """Held-out traj_ids where frozen P47T has SUCCESS_12S=0."""
    path = ROOT / "results/experiments/p47_b_ordinary_expansion/PAIRED_EVAL_ROLLOUTS_P47T.csv"
    out: dict[str, set[str]] = {s: set() for s in SKILLS}
    if not path.is_file():
        # fallback: P44 eval rows with no SUCCESS
        for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")):
            if r.get("tag") != "holdout":
                continue
            sk = r["skill"]
            if sk in out and int(float(r.get("SUCCESS_12S") or 0)) == 0:
                out[sk].add(r["traj_id"])
        return out
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r.get("tag") != "P47T":
            continue
        sk = r["skill"]
        if sk in out and int(float(r.get("SUCCESS_12S") or 0)) == 0:
            out[sk].add(r["traj_id"])
    return out


def select_ics(*, per_class: int = 4, seed: int = SEED, p47t_fails_only: bool = False) -> tuple[list[dict], dict]:
    hold_by_skill: dict[str, set[str]] = {s: set() for s in SKILLS}
    for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")):
        if r.get("tag") != "holdout":
            continue
        sk = r["skill"]
        if sk in hold_by_skill:
            hold_by_skill[sk].add(r["traj_id"])
    fail_ids = _p47t_fail_ids() if p47t_fails_only else None
    jobs = collect_jobs()
    by_skill: dict[str, list] = {s: [] for s in SKILLS}
    for j in jobs:
        sk = j["skill"]
        if sk not in hold_by_skill:
            continue
        tid = j["traj_id"]
        held = hold_by_skill[sk]
        if not (tid in held or any(tid in h or h in tid for h in held)):
            continue
        if fail_ids is not None:
            fails = fail_ids[sk]
            if not (tid in fails or any(tid in f or f in tid for f in fails)):
                continue
        by_skill[sk].append(j)

    rng = np.random.default_rng(seed)
    selected = []
    meta = {
        "seed": seed,
        "per_class": per_class,
        "p47t_fails_only": bool(p47t_fails_only),
        "source": "P44 holdout" + (" ∩ P47T SUCCESS_12S=0" if p47t_fails_only else "") + " via collect_jobs",
        "by_skill": {},
    }
    for sk in SKILLS:
        pool = by_skill[sk]
        if not pool:
            meta["by_skill"][sk] = []
            continue
        if sk == "B1" and per_class > 1:
            buckets: dict[str, list] = defaultdict(list)
            for j in pool:
                buckets[family_of_b(j["traj_id"])].append(j)
            fams = sorted(buckets.keys())
            picks = []
            while len(picks) < per_class and any(buckets.values()):
                for fam in fams:
                    if len(picks) >= per_class:
                        break
                    if buckets[fam]:
                        idx = int(rng.integers(0, len(buckets[fam])))
                        picks.append(buckets[fam].pop(idx))
            chosen = picks
        else:
            order = np.arange(len(pool))
            rng.shuffle(order)
            chosen = [pool[int(i)] for i in order[:per_class]]
        meta["by_skill"][sk] = [
            {
                "traj_id": j["traj_id"],
                "family": family_of_b(j["traj_id"]) if sk == "B1" else j.get("family"),
                "split": j.get("split"),
                "p47t_failed": True if p47t_fails_only else None,
            }
            for j in chosen
        ]
        selected.extend(chosen)
    return selected, meta


def validate_branch_restore(env, model, book, rho, job, device) -> dict:
    import pickle

    from envs.observations.airborne_obs import observe_airborne
    from envs.step_legal import step_legal
    from models.m111_checkpoint import decode_m111_action

    rng = np.random.default_rng(123)
    raw = rollout_independent(env, model, book, rho, job, device, rng, keep_captures=True)
    if not raw["captures"]:
        return {"passed": False, "reason": "no_captures", "n_steps": raw["n_policy_steps"]}
    keys = sorted(raw["captures"])
    mid = keys[min(3, len(keys) - 1)]
    cap = raw["captures"][mid]
    ref = raw["ref"]
    bg = np.random.default_rng(999).bit_generator.state

    def one_step(state_bg):
        restore_branch_state(env, cap)
        rng2 = np.random.default_rng(0)
        rng2.bit_generator.state = pickle.loads(pickle.dumps(state_bg))
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        env._obs_seq.append(obs)
        env._prev_seq.append(np.asarray(env._prev, np.float32).copy())
        tok, resid, _ = sample_m111_token(model, env._obs_seq, env._prev_seq, device, rng2)
        a = decode_m111_action(book, tok, resid, rho=rho)
        step_legal(env.sim, a)
        return tok, np.asarray(env.sim.data.qpos, float).copy(), float(env.sim.data.time)

    tok_a, q_a, t_a = one_step(bg)
    tok_b, q_b, t_b = one_step(bg)
    report = {
        "passed": bool(tok_a == tok_b and np.max(np.abs(q_a - q_b)) <= 1e-6 and abs(t_a - t_b) <= 1e-9),
        "token_match": tok_a == tok_b,
        "qpos_max_abs": float(np.max(np.abs(q_a - q_b))),
        "time_abs": abs(t_a - t_b),
        "branch_step": int(mid),
        "trunk_steps": int(raw["n_policy_steps"]),
        "ic": job["traj_id"],
        "skill": job["skill"],
    }
    rng3 = np.random.default_rng(7)
    cont = continue_from_branch(env, model, book, rho, ref, device, rng3, cap)
    report["suffix_ok"] = cont["n_policy_steps"] >= int(mid)
    report["passed"] = bool(report["passed"] and report["suffix_ok"])
    return report


def smoke_sampling(env, model, book, rho, job, device) -> dict:
    ind = search_independent(env, model, book, rho, job, device, search_seed=1, n=2)
    tree = search_tree(env, model, book, rho, job, device, search_seed=1, n=4)
    return {
        "independent_n": len(ind),
        "tree_n": len(tree),
        "independent_origins": [c.origin for c in ind],
        "tree_origins": [c.origin for c in tree],
        "stochastic_tokens_differ": ind[0].tokens != ind[1].tokens or len(ind[0].tokens) != len(ind[1].tokens),
        "ok": len(ind) == 2 and len(tree) == 4,
    }


def coverage_at(cands: list, k: int) -> int:
    return int(any(c.SUCCESS_12S for c in cands[:k]))


def analyze(all_rows: list[dict], ic_jobs: list[dict], seeds: list[int]) -> dict:
    # Primary seed analysis (first seed) for paired table
    seed0 = seeds[0]
    rows0 = [r for r in all_rows if int(r["search_seed"]) == seed0]
    per_ic = []
    for job in ic_jobs:
        tid = job["traj_id"]
        sk = job["skill"]
        ind = [r for r in rows0 if r["ic_id"] == tid and r["method"] == "independent"]
        tre = [r for r in rows0 if r["ic_id"] == tid and r["method"] == "tree"]
        ind = sorted(ind, key=lambda r: int(r["cand_index"]))
        tre = sorted(tre, key=lambda r: int(r["cand_index"]))
        n_ind = sum(int(r["SUCCESS_12S"]) for r in ind)
        n_tre = sum(int(r["SUCCESS_12S"]) for r in tre)
        first_ind = next((int(r["cand_index"]) for r in ind if int(r["SUCCESS_12S"])), None)
        first_tre = next((int(r["cand_index"]) for r in tre if int(r["SUCCESS_12S"])), None)
        if n_ind > 0 and n_tre > 0:
            winner = "both"
        elif n_ind > 0:
            winner = "independent"
        elif n_tre > 0:
            winner = "tree"
        else:
            winner = "neither"
        per_ic.append(
            {
                "ic_id": tid,
                "skill": sk,
                "class": CLASS_LABEL[sk],
                "independent_successes": n_ind,
                "tree_successes": n_tre,
                "independent_first_success": first_ind,
                "tree_first_success": first_tre,
                "winner": winner,
                "independent_sim_steps": sum(int(r["sim_steps"]) for r in ind),
                "tree_sim_steps": sum(int(r["sim_steps"]) for r in tre),
                "independent_wall_s": sum(float(r["wall_s"]) for r in ind),
                "tree_wall_s": sum(float(r["wall_s"]) for r in tre),
                "independent_policy_calls": sum(int(r["policy_calls"]) for r in ind),
                "tree_policy_calls": sum(int(r["policy_calls"]) for r in tre),
                "tree_snapshots": sum(int(r["snapshots"]) for r in tre),
                "tree_restores": sum(int(r["restores"]) for r in tre),
            }
        )

    budget_rows = []
    for k in BUDGET_CHECKPOINTS:
        for method in ("independent", "tree"):
            cov = 0
            by_sk = {s: 0 for s in SKILLS}
            for job in ic_jobs:
                tid = job["traj_id"]
                sk = job["skill"]
                sub = sorted(
                    [r for r in rows0 if r["ic_id"] == tid and r["method"] == method],
                    key=lambda r: int(r["cand_index"]),
                )
                hit = any(int(r["SUCCESS_12S"]) for r in sub[:k])
                cov += int(hit)
                by_sk[sk] += int(hit)
            budget_rows.append(
                {
                    "k": k,
                    "method": method,
                    "coverage": cov,
                    "coverage_A": by_sk["A1"],
                    "coverage_B": by_sk["B1"],
                    "coverage_C": by_sk["C"],
                    "coverage_D": by_sk["D"],
                }
            )

    # seed stability
    seed_cov = {}
    for sd in seeds:
        rows = [r for r in all_rows if int(r["search_seed"]) == sd]
        for method in ("independent", "tree"):
            cov = 0
            for job in ic_jobs:
                sub = [r for r in rows if r["ic_id"] == job["traj_id"] and r["method"] == method]
                cov += int(any(int(r["SUCCESS_12S"]) for r in sub))
            seed_cov.setdefault(method, {})[sd] = cov

    return {"per_ic": per_ic, "budget": budget_rows, "seed_cov": seed_cov, "seed0": seed0}


def write_final(analysis: dict, restore_rep: dict, smoke: dict, config: dict) -> None:
    per = analysis["per_ic"]
    n_ic = len(per)
    both = sum(1 for r in per if r["winner"] == "both")
    ind_only = sum(1 for r in per if r["winner"] == "independent")
    tree_only = sum(1 for r in per if r["winner"] == "tree")
    neither = sum(1 for r in per if r["winner"] == "neither")
    cov_ind = both + ind_only
    cov_tree = both + tree_only
    sim_ind = sum(r["independent_sim_steps"] for r in per)
    sim_tree = sum(r["tree_sim_steps"] for r in per)
    wall_ind = sum(r["independent_wall_s"] for r in per)
    wall_tree = sum(r["tree_wall_s"] for r in per)

    by_cls = {}
    for sk in SKILLS:
        sub = [r for r in per if r["skill"] == sk]
        by_cls[sk] = {
            "independent": sum(1 for r in sub if r["independent_successes"] > 0),
            "tree": sum(1 for r in sub if r["tree_successes"] > 0),
            "n": len(sub),
        }

    if cov_ind > cov_tree:
        winner = "independent"
    elif cov_tree > cov_ind:
        winner = "tree"
    else:
        winner = "tie"

    seed_cov = analysis["seed_cov"]
    stable = True
    if len(config["search_seeds"]) > 1:
        dirs = []
        for sd in config["search_seeds"]:
            di = seed_cov["independent"][sd] - seed_cov["tree"][sd]
            dirs.append(np.sign(di))
        stable = len(set(dirs)) == 1

    margin = abs(cov_ind - cov_tree)
    conclusive = margin >= 2 and (len(config["search_seeds"]) == 1 or stable)

    lines = [
        "# FINAL_REPORT — P52 Independent vs Tree Branching",
        "",
        f"Generated: {utc()}",
        "",
        f"Policy: frozen P47T M111. Budget: {config.get('n_candidates', N_CANDIDATES)} terminal candidates / IC / method.",
        f"ICs: {n_ic} ({config.get('ic_note', 'held-out')}). Seeds: {config['search_seeds']}.",
        "",
        f"## Primary: Coverage@{config.get('n_candidates', 64)}",
        "",
        f"| Method | Coverage |",
        f"|---|---:|",
        f"| Independent | {cov_ind}/{n_ic} |",
        f"| Tree branching | {cov_tree}/{n_ic} |",
        f"| **Winner** | **{winner}** |",
        "",
        f"Paired: both={both}, independent-only={ind_only}, tree-only={tree_only}, neither={neither}.",
        "",
        "## Per-class coverage",
        "",
        "| Class | Independent | Tree |",
        "|---|---:|---:|",
    ]
    for sk in SKILLS:
        b = by_cls[sk]
        if b["n"] == 0:
            continue
        lines.append(f"| {CLASS_LABEL[sk]} | {b['independent']}/{b['n']} | {b['tree']}/{b['n']} |")

    lines += [
        "",
        f"## Compute (sum over {n_ic} ICs, seed0)",
        "",
        f"| Metric | Independent | Tree |",
        f"|---|---:|---:|",
        f"| Sim steps (policy) | {sim_ind} | {sim_tree} |",
        f"| Wall-clock (s) | {wall_ind:.1f} | {wall_tree:.1f} |",
        f"| Snapshots | 0 | {sum(r['tree_snapshots'] for r in per)} |",
        f"| Restores | 0 | {sum(r['tree_restores'] for r in per)} |",
        "",
        f"Branch restore validation: **{'PASS' if restore_rep.get('passed') else 'FAIL'}**.",
        f"Smoke: **{'PASS' if smoke.get('ok') else 'FAIL'}**.",
        "",
        "## Answers",
        "",
        f"1. **More ICs covered?** {winner} (Indep {cov_ind}/{n_ic} vs Tree {cov_tree}/{n_ic}).",
        f"2. **Per class?** See table above.",
        f"3. **Exclusive solves?** Independent-only ICs={ind_only}, Tree-only={tree_only}.",
        f"4. **Stable across seeds?** {stable if len(config['search_seeds'])>1 else 'N/A (single seed pilot)'}; seed coverages `{seed_cov}`.",
        f"5. **Sim steps?** Tree {'saves' if sim_tree < sim_ind else 'consumes more or equal'} vs independent "
        f"({sim_tree} vs {sim_ind} policy steps; prefixes reused but trunks still fully simulated).",
        f"6. **Meaningful branching advantage?** "
        f"{'Yes' if conclusive and winner=='tree' else ('No clear advantage' if winner!='tree' else 'Narrow — treat as inconclusive')}.",
        f"7. **Next GRPO sampling?** "
        + (
            "Prefer **independent** sampling as the default; keep tree as optional augmentation only if compute-normalized gains appear."
            if winner == "independent"
            else (
                "Tree branching covers more ICs under equal candidates — consider tree-augmented GRPO, but monitor step cost."
                if winner == "tree" and conclusive
                else "Evidence inconclusive — default to independent sampling for next GRPO; re-test tree with more seeds/ICs before committing."
            )
        ),
        "",
        f"Pilot on {n_ic} ICs. Do not treat as definitive statistical proof.",
    ]
    write_md(OUT / "FINAL_REPORT.md", "\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--n-cand", type=int, default=N_CANDIDATES)
    ap.add_argument("--seeds", default="52", help="comma-separated search seeds")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument(
        "--p47t-fails-only",
        action="store_true",
        help="Restrict to held-out ICs where P47T has SUCCESS_12S=0 (use --per-class 1 for one fail/class)",
    )
    ap.add_argument("--out-subdir", default="", help="Optional subdirectory under OUT for this run")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    global OUT
    if args.out_subdir:
        OUT = OUT / args.out_subdir
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        write_audit()

    print("Selecting ICs...", flush=True)
    jobs, sel_meta = select_ics(
        per_class=args.per_class, seed=SEED, p47t_fails_only=bool(args.p47t_fails_only)
    )
    (OUT / "IC_SELECTION.json").write_text(
        json.dumps({"generated": utc(), **sel_meta, "n": len(jobs)}, indent=2), encoding="utf-8"
    )
    print(f"Selected {len(jobs)} ICs (p47t_fails_only={args.p47t_fails_only})", flush=True)
    for j in jobs:
        print(f"  {j['skill']} {j['traj_id']}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, saved = load_m111(P47T_CKPT, device)
    model.eval()
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    rho = saved.get("residual_rho", rho)

    config = {
        "generated": utc(),
        "checkpoint": str(P47T_CKPT),
        "n_candidates": args.n_cand,
        "n_trunks": N_TRUNKS,
        "uniform_per_trunk": N_UNIFORM_PER_TRUNK,
        "entropy_per_trunk": N_ENTROPY_PER_TRUNK,
        "t_max": T_MAX,
        "search_seeds": seeds,
        "temperature": 1.0,
        "handoff": "R3F_masked_HANDOFF_token",
        "ic_seed": SEED,
        "p47t_fails_only": bool(args.p47t_fails_only),
        "ic_note": (
            f"{len(jobs)} P47T-fail held-out ICs (1/class)"
            if args.p47t_fails_only
            else f"{len(jobs)} stratified held-out"
        ),
    }
    (OUT / "EXPERIMENT_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    print("Branch restore validation...", flush=True)
    restore_rep = validate_branch_restore(env, model, book, rho, jobs[0], device)
    write_md(
        OUT / "BRANCH_RESTORE_VALIDATION.md",
        "# BRANCH_RESTORE_VALIDATION\n\n```json\n" + json.dumps(restore_rep, indent=2) + "\n```\n",
    )
    print(f"  restore passed={restore_rep.get('passed')}", flush=True)
    if not restore_rep.get("passed"):
        write_md(OUT / "FINAL_REPORT.md", f"# FINAL_REPORT — aborted\n\nBranch restore failed.\n\n{restore_rep}\n")
        return

    print("Smoke...", flush=True)
    smoke = smoke_sampling(env, model, book, rho, jobs[0], device)
    (OUT / "SMOKE.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print(f"  smoke ok={smoke.get('ok')}", flush=True)
    if args.smoke_only or not smoke.get("ok"):
        return

    all_rows = []
    t_run = time.perf_counter()
    for sd in seeds:
        for ji, job in enumerate(jobs):
            print(f"=== seed={sd} IC {ji+1}/{len(jobs)} {job['skill']} {job['traj_id'][:40]} independent ===", flush=True)
            ind = search_independent(env, model, book, rho, job, device, search_seed=sd, n=args.n_cand)
            print(f"    successes={sum(c.SUCCESS_12S for c in ind)}/{len(ind)}", flush=True)
            print(f"=== seed={sd} IC {ji+1}/{len(jobs)} tree ===", flush=True)
            tre = search_tree(env, model, book, rho, job, device, search_seed=sd, n=args.n_cand)
            print(f"    successes={sum(c.SUCCESS_12S for c in tre)}/{len(tre)} origins={Counter(c.origin for c in tre)}", flush=True)
            for c in ind + tre:
                all_rows.append(c.to_row())
            write_csv(OUT / "TRAJECTORY_RESULTS.csv", all_rows)

    analysis = analyze(all_rows, jobs, seeds)
    write_csv(OUT / "PER_IC_COMPARISON.csv", analysis["per_ic"])
    write_csv(OUT / "BUDGET_CURVES.csv", analysis["budget"])
    (OUT / "ANALYSIS.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    write_final(analysis, restore_rep, smoke, config)
    print(f"Done in {(time.perf_counter()-t_run)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
