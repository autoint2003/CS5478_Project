"""P64 orchestrator: integrity → sequential steps with early-stop → reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.precision.p62_grpo_optimization.eval_dev import paired_delta
from experiments.precision.p64_ppo_multiround.eval_p64 import eval_baselines, eval_tag
from experiments.precision.p64_ppo_multiround.integrity import run_integrity
from experiments.precision.p64_ppo_multiround.paths import (
    CKPT_DIR,
    CLASS_REGRESS_ABS,
    ENTROPY_COLLAPSE,
    EV_WARN,
    KL_WARN,
    MAX_STEPS,
    MSP_DECLINE_STREAK,
    OUT,
    P63_A_ACTOR,
    P63_A_CRITIC,
    P63_A_CUMULATIVE,
    TRAJ_PER_STEP,
)
from experiments.precision.p64_ppo_multiround.train_step import run_step
from experiments.precision.p64_ppo_multiround.write_reports import write_all_reports


def _early_stop(step_result: dict, best_msp: float, decline_streak: int) -> tuple[bool, str, int]:
    if step_result.get("aborted"):
        return True, f"aborted:{step_result['aborted']}", decline_streak
    hist = step_result.get("history") or []
    for h in hist:
        ppo = h.get("ppo") or {}
        if float(ppo.get("approx_kl") or 0) > KL_WARN * 5:
            # soft warn only unless extreme (handled by aborted)
            pass
        if float(ppo.get("ent") or 1.0) < ENTROPY_COLLAPSE:
            return True, "entropy_collapse", decline_streak
        if float(h.get("rollout", {}).get("explained_var") or 0) < -0.5:
            return True, "critic_ev_collapse", decline_streak

    msp = float(step_result["eval"]["MSP_RELAXED"])
    # Do not stop solely for 0.002–0.005 dips
    if msp < best_msp - 0.01:
        decline_streak += 1
    else:
        decline_streak = 0
    if decline_streak >= MSP_DECLINE_STREAK:
        return True, "credible_msp_decline_streak", decline_streak

    # Severe multi-class regression vs P63-A reference stored in step
    prev_bc = (step_result.get("ref_p63a_by_class") or {})
    cur_bc = (step_result["eval"].get("by_class") or {})
    bad = 0
    for c in ("A", "B", "D"):
        if c in prev_bc and c in cur_bc:
            if float(cur_bc[c]["MSP_RELAXED"]) < float(prev_bc[c]["MSP_RELAXED"]) - CLASS_REGRESS_ABS:
                bad += 1
    if bad >= 2:
        return True, "multi_class_regression", decline_streak
    return False, "", decline_streak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--start-step", type=int, default=1)
    ap.add_argument("--skip-integrity", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_integrity:
        print("=== Phase 0 integrity ===", flush=True)
        integ = run_integrity()
        if not integ.get("all_ok"):
            print("INTEGRITY ISSUES — see P63_INTEGRITY_CHECK.md", flush=True)
            # Optimizer missing is documented inside a PASS finding; hard-fail only if all_ok false
            # Continue if only soft issues: still require all_ok
            return

    print("=== Baseline evals (P62 + P63-A) ===", flush=True)
    bases = eval_baselines(device=args.device)
    (OUT / "BASELINES.json").write_text(
        json.dumps(
            {
                "P62_epoch3": {k: v for k, v in bases["P62_epoch3"].items() if k != "per_ic"},
                "P63_PPO_A": {k: v for k, v in bases["P63_PPO_A"].items() if k != "per_ic"},
                "paired_p63a_vs_p62": bases["paired_p63a_vs_p62"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    p62 = bases["P62_epoch3"]
    p63a = bases["P63_PPO_A"]

    if args.skip_train:
        write_all_reports()
        return

    actor_in = P63_A_ACTOR
    critic_in = P63_A_CRITIC
    opt_in = None
    prev_eval = p63a
    best = {"msp": float(p63a["MSP_RELAXED"]), "step": 0, "tag": "P63_PPO_A", "ckpt": str(P63_A_ACTOR)}
    decline_streak = 0
    step_results = []

    for step in range(args.start_step, args.max_steps + 1):
        # resume from last completed if start_step > 1
        if step > 1 and args.start_step == step:
            prev_tag = f"P64_step{step-1}"
            actor_in = CKPT_DIR / f"{prev_tag}_actor.pt"
            critic_in = CKPT_DIR / f"{prev_tag}_critic.pt"
            opt_in = CKPT_DIR / f"{prev_tag}_optim.pt"
            prev_path = OUT / f"{prev_tag}.json"
            if prev_path.is_file():
                prev_blob = json.loads(prev_path.read_text(encoding="utf-8"))
                # re-eval not required; use stored if present
                prev_eval = eval_tag(prev_tag, actor_in, device=args.device)

        print(f"\n######## P64 STEP {step} (+{TRAJ_PER_STEP} traj) ########\n", flush=True)
        res = run_step(
            step=step,
            device=args.device,
            actor_in=actor_in,
            critic_in=critic_in,
            opt_in=opt_in,
            prev_eval=prev_eval,
            baseline_eval=p62,
        )
        res["paired_vs_p63a"] = paired_delta(p63a, {"per_ic": res["eval_full_per_ic"], **res["eval"]})
        res["ref_p63a_by_class"] = p63a.get("by_class")
        # rewrite step json with paired_vs_p63a
        (OUT / f"P64_step{step}.json").write_text(
            json.dumps({k: v for k, v in res.items()}, indent=2), encoding="utf-8"
        )

        msp = float(res["eval"]["MSP_RELAXED"])
        ref_best = float(best["msp"])
        if msp > ref_best:
            best = {
                "msp": msp,
                "step": step,
                "tag": res["tag"],
                "ckpt": res["actor"],
            }
            decline_streak = 0
        stop, reason, decline_streak = _early_stop(res, ref_best, decline_streak)
        res["early_stop_check"] = {"stop": stop, "reason": reason, "decline_streak": decline_streak}
        step_results.append(res)

        # prepare next
        actor_in = Path(res["actor"])
        critic_in = Path(res["critic"])
        opt_in = Path(res["optim"])
        prev_eval = {"per_ic": res["eval_full_per_ic"], **res["eval"]}

        if stop:
            print(f"EARLY STOP after step {step}: {reason}", flush=True)
            (OUT / "EARLY_STOP.json").write_text(
                json.dumps({"step": step, "reason": reason, "best": best}, indent=2),
                encoding="utf-8",
            )
            break

        # Soft diagnostics
        last_ppo = (res.get("history") or [{}])[-1].get("ppo") or {}
        last_ev = (res.get("history") or [{}])[-1].get("rollout", {}).get("explained_var")
        if last_ev is not None and last_ev < EV_WARN:
            print(f"WARN: low explained variance {last_ev}", flush=True)
        if float(last_ppo.get("approx_kl") or 0) > KL_WARN:
            print(f"WARN: approx_kl {last_ppo.get('approx_kl')}", flush=True)

    (OUT / "BEST_CHECKPOINT.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_all_reports(best=best)
    print(json.dumps({"best": best, "n_steps_done": len(step_results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
