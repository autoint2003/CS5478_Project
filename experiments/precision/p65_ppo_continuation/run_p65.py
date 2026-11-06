"""P65 orchestrator: integrity → diagnose → step5 → eval → optional step6 → reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.precision.p62_grpo_optimization.eval_dev import paired_delta
from experiments.precision.p64_ppo_multiround.eval_p64 import eval_tag
from experiments.precision.p65_ppo_continuation.dev_panel import build_panel_manifest, eval_on_panel
from experiments.precision.p65_ppo_continuation.diagnose_a import run_a_diagnosis
from experiments.precision.p65_ppo_continuation.diagnose_c import run_c_diagnosis
from experiments.precision.p65_ppo_continuation.exposure import run_exposure
from experiments.precision.p65_ppo_continuation.integrity import run_integrity
from experiments.precision.p65_ppo_continuation.paths import (
    CKPT_DIR,
    OUT,
    P64_CUMULATIVE,
    P64_OUT,
    P64_STEP1_ACTOR,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P64_STEP4_OPTIM,
    P65_STEP5_CUM,
    TRAJ_PER_STEP,
)
from experiments.precision.p65_ppo_continuation.train_step import run_p65_step
from experiments.precision.p65_ppo_continuation.write_reports import write_all


def _decide_step6(step5: dict, p64_step4_eval: dict) -> tuple[bool, str]:
    if step5.get("aborted"):
        return False, f"aborted:{step5['aborted']}"
    hist = step5.get("history") or []
    for h in hist:
        ppo = h.get("ppo") or {}
        if float(ppo.get("ent") or 1) < 0.05:
            return False, "entropy_collapse"
        if float(ppo.get("approx_kl") or 0) > 0.5:
            return False, "kl_explosion"
    m5 = float(step5["eval"]["MSP_RELAXED"])
    m4 = float(p64_step4_eval.get("MSP_RELAXED") or 0)
    # Allow small regression within noise; pause on broad severe drop
    if m5 < m4 - 0.04:
        return False, "severe_overall_regression"
    bc5 = step5["eval"].get("by_class") or {}
    bc4 = p64_step4_eval.get("by_class") or {}
    bad = 0
    for c in ("A", "B", "D"):
        if c in bc5 and c in bc4:
            if float(bc5[c]["MSP_RELAXED"]) < float(bc4[c]["MSP_RELAXED"]) - 0.10:
                bad += 1
    if bad >= 2:
        return False, "multi_class_regression"
    return True, "stable_continue_ok"


def _select_best(candidates: list[tuple[str, dict, str]], panel: dict | None) -> dict:
    # Primary: overall MSP; secondary panel MSP if available
    scored = []
    for tag, ev, ckpt in candidates:
        score = float(ev.get("MSP_RELAXED") or -1)
        panel_msp = None
        if panel and tag in panel:
            panel_msp = float(panel[tag].get("MSP_RELAXED") or 0)
            score += 0.15 * panel_msp  # light secondary weight
        zeros = int(ev.get("zero_success_ics") or 99)
        scored.append((score, -zeros, tag, ev.get("MSP_RELAXED"), ckpt, panel_msp))
    scored.sort(reverse=True)
    best = scored[0]
    return {
        "tag": best[2],
        "msp": best[3],
        "ckpt": best[4],
        "panel_msp": best[5],
        "selection_rule": "primary overall MSP_RELAXED; secondary panel MSP & zeros",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-integrity", action="store_true")
    ap.add_argument("--skip-diagnose", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--force-step6", action="store_true")
    ap.add_argument("--no-step6", action="store_true")
    ap.add_argument("--c-rollouts", type=int, default=16)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_integrity:
        print("=== Phase 0 integrity ===", flush=True)
        integ = run_integrity(device=args.device)
        if not integ.get("all_ok"):
            print("INTEGRITY FAILED", flush=True)
            return

    print("=== Pre-register extra dev panel ===", flush=True)
    build_panel_manifest()

    if not args.skip_diagnose:
        print("=== Exposure audit ===", flush=True)
        run_exposure()
        print("=== A diagnosis ===", flush=True)
        a_rep = run_a_diagnosis(device=args.device)
        # merge A critic into shared analysis
        ca_path = OUT / "CRITIC_ADVANTAGE_ANALYSIS.json"
        blob = json.loads(ca_path.read_text(encoding="utf-8")) if ca_path.is_file() else {}
        blob["A"] = a_rep.get("critic_advantage_p64_step4")
        ca_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        (OUT / "CRITIC_ADVANTAGE_ANALYSIS.md").write_text(
            "# CRITIC_ADVANTAGE_ANALYSIS\n\n"
            + "```json\n"
            + json.dumps(blob, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        print("=== C diagnosis ===", flush=True)
        run_c_diagnosis(device=args.device, n_per_ic=args.c_rollouts)

    p64_4 = json.loads((P64_OUT / "P64_step4.json").read_text(encoding="utf-8"))
    p64_1 = json.loads((P64_OUT / "P64_step1.json").read_text(encoding="utf-8"))

    step5 = step6 = None
    if not args.skip_train:
        print("=== P65_step5 (+2048) ===", flush=True)
        step5 = run_p65_step(
            step_name="P65_step5",
            step_index=5,
            device=args.device,
            actor_in=P64_STEP4_ACTOR,
            critic_in=P64_STEP4_CRITIC,
            opt_in=P64_STEP4_OPTIM,
            cumulative_before=P64_CUMULATIVE,
        )
        step5["paired_vs_p64_step4"] = paired_delta(
            {"per_ic": p64_4.get("eval_full_per_ic"), **(p64_4.get("eval") or {})},
            {"per_ic": step5["eval_full_per_ic"], **step5["eval"]},
        )
        step5["paired_vs_p64_step1"] = paired_delta(
            {"per_ic": p64_1.get("eval_full_per_ic"), **(p64_1.get("eval") or {})},
            {"per_ic": step5["eval_full_per_ic"], **step5["eval"]},
        )
        (OUT / "P65_step5.json").write_text(json.dumps(step5, indent=2), encoding="utf-8")

        cont, reason = _decide_step6(step5, p64_4.get("eval") or {})
        (OUT / "STEP6_DECISION.json").write_text(
            json.dumps({"continue": cont, "reason": reason}, indent=2), encoding="utf-8"
        )
        print(f"Step6 decision: continue={cont} ({reason})", flush=True)
        if args.force_step6:
            cont = True
        if args.no_step6:
            cont = False
        if cont:
            print("=== P65_step6 (+2048) ===", flush=True)
            step6 = run_p65_step(
                step_name="P65_step6",
                step_index=6,
                device=args.device,
                actor_in=Path(step5["actor"]),
                critic_in=Path(step5["critic"]),
                opt_in=Path(step5["optim"]),
                cumulative_before=P65_STEP5_CUM,
            )
            step6["paired_vs_p64_step4"] = paired_delta(
                {"per_ic": p64_4.get("eval_full_per_ic"), **(p64_4.get("eval") or {})},
                {"per_ic": step6["eval_full_per_ic"], **step6["eval"]},
            )
            (OUT / "P65_step6.json").write_text(json.dumps(step6, indent=2), encoding="utf-8")

    # Panel evals for selection candidates
    panel = {}
    if not args.skip_panel:
        cands_ckpts = [
            ("P64_step1", P64_STEP1_ACTOR),
            ("P64_step4", P64_STEP4_ACTOR),
        ]
        if (OUT / "P65_step5_actor.pt").is_file() or (CKPT_DIR / "P65_step5_actor.pt").is_file():
            cands_ckpts.append(("P65_step5", CKPT_DIR / "P65_step5_actor.pt"))
        if (CKPT_DIR / "P65_step6_actor.pt").is_file():
            cands_ckpts.append(("P65_step6", CKPT_DIR / "P65_step6_actor.pt"))
        for tag, ck in cands_ckpts:
            print(f"=== Panel eval {tag} ===", flush=True)
            panel[tag] = {
                k: v
                for k, v in eval_on_panel(tag, ck, device=args.device).items()
                if k != "per_ic"
            }

    candidates = [
        ("P64_step1", p64_1.get("eval") or {}, str(P64_STEP1_ACTOR)),
        ("P64_step4", p64_4.get("eval") or {}, str(P64_STEP4_ACTOR)),
    ]
    if (OUT / "P65_step5.json").is_file():
        d = json.loads((OUT / "P65_step5.json").read_text(encoding="utf-8"))
        candidates.append(("P65_step5", d.get("eval") or {}, d.get("actor")))
    if (OUT / "P65_step6.json").is_file():
        d = json.loads((OUT / "P65_step6.json").read_text(encoding="utf-8"))
        candidates.append(("P65_step6", d.get("eval") or {}, d.get("actor")))

    best = _select_best(candidates, panel if panel else None)
    (OUT / "BEST_CHECKPOINT.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_all(best=best)
    _write_final(best, panel)
    print(json.dumps({"best": best, "panel_tags": list(panel)}, indent=2), flush=True)


def _write_final(best: dict, panel: dict):
    a_path = OUT / "A_REGRESSION_DIAGNOSIS.json"
    c_path = OUT / "C_BOTTLENECK_DIAGNOSIS.json"
    a = json.loads(a_path.read_text(encoding="utf-8")) if a_path.is_file() else {}
    c = json.loads(c_path.read_text(encoding="utf-8")) if c_path.is_file() else {}
    s5 = OUT / "P65_step5.json"
    s6 = OUT / "P65_step6.json"
    e5 = json.loads(s5.read_text(encoding="utf-8"))["eval"] if s5.is_file() else None
    e6 = json.loads(s6.read_text(encoding="utf-8"))["eval"] if s6.is_file() else None
    p64_4 = json.loads((P64_OUT / "P64_step4.json").read_text(encoding="utf-8"))["eval"]
    p64_1 = json.loads((P64_OUT / "P64_step1.json").read_text(encoding="utf-8"))["eval"]

    lines = [
        "# FINAL_REPORT — P65",
        "",
        f"**Best checkpoint:** `{best.get('tag')}` MSP={best.get('msp')} → `{best.get('ckpt')}`",
        "",
        "## Final questions",
        "",
        f"1. **Why A declined after step1?** See `A_REGRESSION_DIAGNOSIS.md`. "
        f"Summary: mean Δ(step4−step1)={((a.get('regression_summary') or {}).get('mean_delta_step4_vs_step1'))}; "
        f"concentrated={((a.get('regression_summary') or {}).get('concentrated_in_one_ic'))}; "
        f"drift vs step1 in diagnosis JSON. Not claiming KL causes regression.",
        "",
        f"2. **Why C ~0.02?** See `C_BOTTLENECK_DIAGNOSIS.md`. "
        f"Buckets={((c.get('diagnostic') or {}).get('bucket_rates'))}; "
        f"hypotheses={((c.get('diagnostic') or {}).get('hypotheses'))}.",
        "",
        f"3. **Critic useful for C?** EV/bias in `CRITIC_ADVANTAGE_ANALYSIS.json` / C diagnosis critic block.",
        "",
        "4. **Exposure appropriate?** See `TRAINING_EXPOSURE.md` (stratified 16/class; no weight changes).",
        "",
        f"5. **Additional PPO improve overall?** "
        f"P64_step4={p64_4.get('MSP_RELAXED')}; step5={None if not e5 else e5.get('MSP_RELAXED')}; "
        f"step6={None if not e6 else e6.get('MSP_RELAXED')}.",
        "",
        "6. **Saturating?** Compare step4→5→6 deltas vs earlier gains; see CHECKPOINT_COMPARISON.",
        "",
        f"7. **Zeros?** step1={p64_1.get('zero_success_ics')} step4={p64_4.get('zero_success_ics')} "
        f"step5={None if not e5 else e5.get('zero_success_ics')} step6={None if not e6 else e6.get('zero_success_ics')}.",
        "",
        f"8. **Best?** `{best.get('tag')}`.",
        "",
        "9. **Top P66 improvement?** See `P66_RECOMMENDATIONS.md` — prefer C exploration/curriculum.",
        "",
        f"Panel MSP (secondary): {panel}",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
