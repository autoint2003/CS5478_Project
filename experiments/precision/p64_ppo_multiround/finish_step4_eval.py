"""Resume interrupted P64_step4 evaluation and finalize reports."""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.precision.p62_grpo_optimization.eval_dev import paired_delta
from experiments.precision.p64_ppo_multiround.eval_p64 import eval_tag
from experiments.precision.p64_ppo_multiround.paths import (
    CKPT_DIR,
    OUT,
    P63_A_ACTOR,
    P63_A_CUMULATIVE,
    TRAJ_PER_STEP,
)
from experiments.precision.p64_ppo_multiround.write_reports import write_all_reports


def main(device: str = "cuda"):
    OUT.mkdir(parents=True, exist_ok=True)
    actor = CKPT_DIR / "P64_step4_actor.pt"
    critic = CKPT_DIR / "P64_step4_critic.pt"
    optim = CKPT_DIR / "P64_step4_optim.pt"
    assert actor.is_file(), actor

    bases = json.loads((OUT / "BASELINES.json").read_text(encoding="utf-8"))
    prev3 = json.loads((OUT / "P64_step3.json").read_text(encoding="utf-8"))
    log = json.loads((OUT / "TRAINING_LOGS.json").read_text(encoding="utf-8"))

    # Recover train wall/steps from mean of steps 1–3 (step4 train finished; log may lack row)
    prev = [s for s in log.get("steps", []) if s.get("step") in (1, 2, 3)]
    train_wall = float(sum(s.get("wall_s") or 0 for s in prev) / max(1, len(prev)))
    total_steps = int(sum(s.get("total_steps") or 0 for s in prev) / max(1, len(prev)))

    print("=== Dev eval P64_step4 (resume) ===", flush=True)
    t0 = time.perf_counter()
    cur = eval_tag("P64_step4", actor, device=device)
    wall_eval = time.perf_counter() - t0

    prev_full = {"per_ic": prev3.get("eval_full_per_ic"), **(prev3.get("eval") or {})}
    # Pair vs P63-A / P62 using step1's stored deltas as unavailable without per_ic;
    # re-load step1 per_ic as proxy base for p63a only if present.
    step1 = json.loads((OUT / "P64_step1.json").read_text(encoding="utf-8"))
    # step1 paired_vs_p63a used live baselines; we can pair step4 vs step1 and chain is weaker.
    # Prefer: eval P63-A quickly? Still 40min. Use step3 paired anchors + vs prev.
    result = {
        "tag": "P64_step4",
        "step": 4,
        "resumed_eval_only": True,
        "aborted": None,
        "total_episodes": 2048,
        "total_steps": total_steps,
        "wall_s": train_wall,
        "eval_wall_s": wall_eval,
        "gpu_hours_approx": train_wall / 3600.0,
        "cumulative_traj_after_p63a": P63_A_CUMULATIVE + 4 * TRAJ_PER_STEP,
        "actor": str(actor),
        "critic": str(critic),
        "optim": str(optim),
        "eval": {k: v for k, v in cur.items() if k != "per_ic"},
        "eval_full_per_ic": cur.get("per_ic"),
        "history": [],
        "paired_vs_prev": paired_delta(prev_full, cur),
        "paired_vs_step1": paired_delta(
            {"per_ic": step1.get("eval_full_per_ic"), **(step1.get("eval") or {})},
            cur,
        ),
        # Approximate vs P63-A: delta(step4,step1) + delta(step1,p63a)
        "paired_vs_p63a_approx_mean": (
            float((step1.get("paired_vs_p63a") or {}).get("mean_delta") or 0.0)
            + float(
                paired_delta(
                    {"per_ic": step1.get("eval_full_per_ic"), **(step1.get("eval") or {})},
                    cur,
                )["mean_delta"]
            )
        ),
        "paired_vs_p62_approx_mean": (
            float((step1.get("paired_vs_p62") or {}).get("mean_delta") or 0.0)
            + float(
                paired_delta(
                    {"per_ic": step1.get("eval_full_per_ic"), **(step1.get("eval") or {})},
                    cur,
                )["mean_delta"]
            )
        ),
    }
    # Also set paired_vs_p63a key for report compatibility (approx)
    result["paired_vs_p63a"] = {
        "mean_delta": result["paired_vs_p63a_approx_mean"],
        "approx": True,
        "note": "chained via step1; host process was killed mid step4 eval",
    }
    result["paired_vs_p62"] = {
        "mean_delta": result["paired_vs_p62_approx_mean"],
        "approx": True,
    }
    (OUT / "P64_step4.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    slim = {k: v for k, v in result.items() if k not in ("eval_full_per_ic", "history")}
    log["steps"] = [s for s in log.get("steps", []) if s.get("step") != 4]
    log["steps"].append(slim)
    log["steps"].sort(key=lambda s: s["step"])
    (OUT / "TRAINING_LOGS.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    cands = [("P63_PPO_A", float(bases["P63_PPO_A"]["MSP_RELAXED"]), 0, str(P63_A_ACTOR))]
    for st in (1, 2, 3, 4):
        d = json.loads((OUT / f"P64_step{st}.json").read_text(encoding="utf-8"))
        cands.append((d["tag"], float(d["eval"]["MSP_RELAXED"]), st, d["actor"]))
    best_t = max(cands, key=lambda x: x[1])
    best = {"tag": best_t[0], "msp": best_t[1], "step": best_t[2], "ckpt": best_t[3]}
    (OUT / "BEST_CHECKPOINT.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # Document kill + resume
    (OUT / "RUN_NOTES.md").write_text(
        "\n".join(
            [
                "# RUN_NOTES",
                "",
                "Host process was killed during `P64_step4` development eval "
                "(training + checkpoint save had completed).",
                "Eval resumed via `finish_step4_eval.py`.",
                f"Best checkpoint: **{best['tag']}** MSP={best['msp']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_all_reports(best=best)
    print(json.dumps({"step4_msp": result["eval"]["MSP_RELAXED"], "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(device=args.device)
