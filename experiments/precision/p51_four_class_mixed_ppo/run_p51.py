"""P51 orchestrator: P50 audit → mixed PPO → smoke → pilot → val → held-out eval."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p26_c_curriculum_rl.ppo_m111 import load_actor_critic  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paired_eval import (  # noqa: E402
    eval_ckpt,
    summarize,
    write_csv,
)
from experiments.precision.p50_ppo_reward_redesign.ppo_train import (  # noqa: E402
    P50PPOConfig,
    act_masked,
    load_sft_pools,
)
from experiments.precision.p50_ppo_reward_redesign.reward import RewardWeights, weights_dict  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedRecoveryEnv  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import build_pools  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.mixed_ppo_train import (  # noqa: E402
    token_acc_per_skill,
    train_mixed_ppo,
)
from experiments.precision.p51_four_class_mixed_ppo.paths import (  # noqa: E402
    CLASS_LABEL,
    OUT,
    P44_EVAL,
    P47T_CKPT,
    P50,
    P50_CKPT,
    SEED,
    SKILLS,
    T_MAX,
)
from models.m111_checkpoint import MOTION_K, load_m111  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_p50_audit() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pools = load_sft_pools("P47T")
    import random

    rng = random.Random(SEED)
    ac47 = load_actor_critic(P47T_CKPT, None, device, None)
    acc47 = token_acc_per_skill(ac47, pools, device, rng)
    acc50 = {sk: None for sk in SKILLS}
    if P50_CKPT.is_file():
        ac50 = load_actor_critic(P50_CKPT, None, device, None)
        acc50 = token_acc_per_skill(ac50, pools, device, rng)

    # P50 failure flips
    rows = list(csv.DictReader((P50 / "PPO_PAIRED_EVAL_ROLLOUTS.csv").open(encoding="utf-8")))
    p47 = {(r["skill"], r["traj_id"]): r for r in rows if r.get("tag") == "P47T"}
    p50 = {(r["skill"], r["traj_id"]): r for r in rows if r.get("tag") == "P50B"}
    flips = {sk: {"gain": [], "loss": []} for sk in SKILLS}
    for key, a in p47.items():
        b = p50.get(key)
        if not b:
            continue
        sk = key[0]
        aok, bok = int(float(a.get("SUCCESS_12S") or 0)), int(float(b.get("SUCCESS_12S") or 0))
        if bok and not aok:
            flips[sk]["gain"].append(key[1])
        if aok and not bok:
            flips[sk]["loss"].append(key[1])

    fail50 = {}
    for sk in SKILLS:
        sub = [r for r in rows if r.get("tag") == "P50B" and r.get("skill") == sk]
        fail50[sk] = dict(Counter(r.get("fail_category") for r in sub))

    split = {}
    sp = P50 / "TRAIN_VAL_SPLIT.json"
    if sp.is_file():
        split = json.loads(sp.read_text(encoding="utf-8"))

    md = f"""# P50_REGRESSION_AUDIT

Generated: {utc()}

## Held-out outcome (P50 vs P47T)

| Class | P47T | P50 | Δ | P50 fail modes |
|---|---:|---:|---:|---|
| A | 8/20 | 6/20 | −2 | `{fail50.get('A1')}` |
| B | 1/16 | 5/16 | +4 | `{fail50.get('B1')}` |
| C | 3/20 | 1/20 | −2 | `{fail50.get('C')}` |
| D | 6/20 | 9/20 | +3 | `{fail50.get('D')}` |

## 1. P50 rollout task distribution

P50 trained **B-only** on ordinary non-holdout ball/slide/k48r/inward ICs.

- Train ICs: {len(split.get('train') or [])}
- Val ICs: {len(split.get('val') or [])}
- Families: ball / slide / k48r / inward (equal family quota in loader)
- **A/C/D had 0% of PPO episode sampling** (only via SFT anchor batches)

## 2. PPO diagnostics (from P50 training report)

- LR 1e-5, clip 0.2, GAE λ=0.95, γ=0.99, epochs=3, minibatch=32
- KL→P47T: rose ~0.001→0.05 then ~0.02; entropy fell ~0.6→0.29
- SFT anchor: A/B/C/D chunks present; coef=0.1
- Value loss high/unstable (10–170); explained var noisy
- Short-horizon val SUCCESS_12S≈0 (max_ep_steps=80) — selection tracked drop mainly

## 3. Reward

Identical P50 potential-based reward for all classes (no class-specific terms).
C controlled sliding relies on slip penalty only punishing **increases** in v_rel.

## 4. A/C failure concentration

- A P50: mostly **DROP_BEFORE** (14/20); lost 2 prior successes; no new SUCCESS flips compensating.
- C P50: **DROP_BEFORE** 19/20; only 1 SUCCESS remains.
- Mechanism: early contact loss under closed-loop (same family as P45 DROP_BEFORE), amplified by B-focused policy shift.

Flips (traj_id counts): A gain={len(flips['A1']['gain'])} loss={len(flips['A1']['loss'])}; 
C gain={len(flips['C']['gain'])} loss={len(flips['C']['loss'])}; 
B gain={len(flips['B1']['gain'])} loss={len(flips['B1']['loss'])}; 
D gain={len(flips['D']['gain'])} loss={len(flips['D']['loss'])}.

## 5. Per-class token accuracy (SFT chunks, P47T vs P50)

| Skill | P47T acc | P50 acc |
|---|---:|---:|
"""
    for sk in SKILLS:
        a = acc47.get(sk)
        b = acc50.get(sk)
        md += f"| {sk} | {a if a is None else f'{a:.3f}'} | {b if b is None else f'{b:.3f}'} |\n"
    md += """
SFT token accuracy remaining high does **not** prevent A/C closed-loop regression (matches P48/P49 lesson).

## Verdict for P51

Primary hypothesis: **A/C regression is driven by B-only PPO sampling**, not reward definition.
P51 tests this by balanced 25% A/B/C/D rollouts from P47T with **unchanged** P50 reward/hyperparams.
"""
    write_md(OUT / "P50_REGRESSION_AUDIT.md", md)
    blob = {
        "generated": utc(),
        "acc47": acc47,
        "acc50": acc50,
        "flips": flips,
        "fail50": fail50,
        "p50_split_sizes": {"train": len(split.get("train") or []), "val": len(split.get("val") or [])},
    }
    (OUT / "P50_REGRESSION_AUDIT.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    print("Wrote P50_REGRESSION_AUDIT.md", flush=True)
    return blob


def write_config(pools_meta: dict) -> None:
    w = weights_dict(RewardWeights())
    md = f"""# MIXED_PPO_CONFIG

Generated: {utc()}

## Hypothesis

P50 A/C regression came from **B-focused rollout distribution**. P51 changes only sampling.

## Init

- Checkpoint: `{P47T_CKPT}` (not P50)
- Frozen reference: P47T
- Reward: **exact P50** weights
- Hyperparams: **exact P50** (lr, clip, GAE, KL, SFT, epochs, minibatch, rollout_steps, total_steps)

## Sampling

| Class | Target episode share | Train pool | Val pool |
|---|---:|---:|---:|
| A | 25% | {pools_meta['counts']['train'].get('A1', 0)} | {pools_meta['counts']['val'].get('A1', 0)} |
| B | 25% | {pools_meta['counts']['train'].get('B1', 0)} | {pools_meta['counts']['val'].get('B1', 0)} |
| C | 25% | {pools_meta['counts']['train'].get('C', 0)} | {pools_meta['counts']['val'].get('C', 0)} |
| D | 25% | {pools_meta['counts']['train'].get('D', 0)} | {pools_meta['counts']['val'].get('D', 0)} |

- Episode sample: uniform over nonempty classes, then uniform IC within class.
- Transition share logged separately (`tr_frac`) to detect long-trajectory bias.
- B uses ordinary ball/slide/k48r/inward (P21 B train ⊆ holdout).
- A/C/D use P21/P35 train minus held-out traj_ids.

## Reward JSON (unchanged)

```json
{json.dumps(w, indent=2)}
```

## Skill preservation

- KL(π ‖ π_P47T) coef=0.05
- SFT anchor stratified A/B/C/D coef=0.1
- Entropy 0.02, grad clip 1.0
"""
    write_md(OUT / "MIXED_PPO_CONFIG.md", md)


def run_smoke(pools: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = load_actor_critic(P47T_CKPT, None, device, None)
    _, book, saved = load_m111(P47T_CKPT, device)
    rho = saved.get("residual_rho", saved.get("res_cap"))
    env = MixedRecoveryEnv(book, rho, max_steps=15)
    checks = []

    def ok(name, cond, detail=""):
        checks.append({"check": name, "pass": bool(cond), "detail": detail})

    for sk in SKILLS:
        starts = (pools["train"].get(sk) or []) + (pools["val"].get(sk) or [])
        ok(f"reset_{sk}", bool(starts), f"n={len(starts)}")
        if not starts:
            continue
        env.reset(starts[0])
        tok, resid, logp, val = act_masked(ac, env.obs_seq, env.prev_seq, device)
        ok(f"token_{sk}", 0 <= tok < MOTION_K, f"tok={tok}")
        r, done, info = env.step(tok, resid)
        ok(f"step_{sk}", np.isfinite(r) and "reward_breakdown" in info, f"r={r:.3f} done={done}")
        ok(f"logprob_{sk}", np.isfinite(logp), f"logp={logp:.3f}")

    sft = load_sft_pools("P47T")
    ok("sft_anchor_all_classes", all(len(sft.get(sk) or []) > 0 for sk in SKILLS), {k: len(v) for k, v in sft.items()})
    ok("r3f_t_max", env.t_max == T_MAX, f"t_max={env.t_max}")

    report = {"generated": utc(), "checks": checks, "all_pass": all(c["pass"] for c in checks)}
    lines = ["# SMOKE_TEST_REPORT", "", f"Generated: {utc()}", ""]
    for c in checks:
        lines.append(f"- [{'PASS' if c['pass'] else 'FAIL'}] **{c['check']}**: {c['detail']}")
    lines += ["", f"## Verdict: {'PASS' if report['all_pass'] else 'FAIL'}", ""]
    write_md(OUT / "SMOKE_TEST_REPORT.md", "\n".join(lines))
    (OUT / "SMOKE_TEST_REPORT.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Smoke all_pass={report['all_pass']}", flush=True)
    return report


def write_training_report(train_out: dict) -> None:
    hist = train_out.get("history") or []
    best = train_out.get("best") or {}
    lines = [
        "# PPO_TRAINING_REPORT",
        "",
        f"Generated: {utc()}",
        "",
        f"- Init: `{train_out.get('init')}`",
        f"- Best: `{best.get('path')}` @ steps={best.get('steps')} score={best.get('score')}",
        "",
        "## Sampling + metrics (tail)",
        "",
        "| steps | ret | drop | kl | ep_A | ep_B | ep_C | ep_D | tr_A | tr_B | tr_C | tr_D | score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hist:
        ef, tf = row.get("ep_frac") or {}, row.get("tr_frac") or {}
        lines.append(
            f"| {row.get('steps')} | {row.get('mean_return', float('nan')):.2f} | "
            f"{row.get('drop_rate', float('nan')):.2f} | {row.get('kl_ref', float('nan')):.4f} | "
            f"{ef.get('A1', 0):.2f} | {ef.get('B1', 0):.2f} | {ef.get('C', 0):.2f} | {ef.get('D', 0):.2f} | "
            f"{tf.get('A1', 0):.2f} | {tf.get('B1', 0):.2f} | {tf.get('C', 0):.2f} | {tf.get('D', 0):.2f} | "
            f"{row.get('ckpt_score', float('nan'))} |"
        )
    write_md(OUT / "PPO_TRAINING_REPORT.md", "\n".join(lines))
    (OUT / "PPO_TRAINING_REPORT.json").write_text(json.dumps(train_out, indent=2, default=str), encoding="utf-8")


def write_validation_report(train_out: dict) -> None:
    cands = train_out.get("candidates") or []
    best = train_out.get("best") or {}
    lines = [
        "# VALIDATION_REPORT",
        "",
        f"Generated: {utc()}",
        "",
        "Non-holdout validation only. Selection uses preservation-weighted score "
        "(A/C drop+SFT prioritized; not held-out SUCCESS).",
        "",
        f"Selected: `{best.get('path')}` score={best.get('score')}",
        "",
        "## Candidates",
        "",
    ]
    for c in cands:
        lines.append(f"### steps={c.get('steps')} score={c.get('score'):.3f}")
        lines.append(f"- path: `{c.get('path')}`")
        lines.append(f"- sft_acc: `{c.get('sft_acc')}`")
        lines.append(f"- val: `{c.get('val')}`")
        lines.append("")
    # Pareto note
    lines += [
        "## Selection rationale",
        "",
        "Prefer checkpoints that keep A/C SFT accuracy and lower A/C val drop, "
        "while not ignoring B/D. If no checkpoint meets all promotion targets on held-out, "
        "report trade-offs — do not pick highest mean alone.",
        "",
    ]
    write_md(OUT / "VALIDATION_REPORT.md", "\n".join(lines))


def skill_label(sk: str) -> str:
    return CLASS_LABEL.get(sk, sk)


def run_heldout(tag: str, ckpt: Path) -> dict:
    hold = [r for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")) if r.get("tag") == "holdout"]
    uniq = {(r["skill"], r["traj_id"]): r for r in hold}
    hold = list(uniq.values())
    jobs_by_key = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}
    rows = []
    p47_csv = ROOT / "results/experiments/p47_b_ordinary_expansion/PAIRED_EVAL_ROLLOUTS_P47T.csv"
    if p47_csv.is_file():
        for r in csv.DictReader(p47_csv.open(encoding="utf-8")):
            if r.get("tag") == "P47T":
                rows.append(dict(r))
    p50_csv = P50 / "PPO_PAIRED_EVAL_ROLLOUTS.csv"
    if p50_csv.is_file():
        for r in csv.DictReader(p50_csv.open(encoding="utf-8")):
            if r.get("tag") == "P50B":
                rows.append(dict(r))
    rows.extend(eval_ckpt(tag, ckpt, jobs_by_key, hold))
    write_csv(OUT / "PAIRED_EVAL_ROLLOUTS.csv", rows)
    summaries = {t: summarize(rows, t) for t in ("P47T", "P50B", tag)}

    # transitions
    by = {t: {(r["skill"], r["traj_id"]): r for r in rows if r["tag"] == t} for t in ("P47T", "P50B", tag)}
    transitions = []
    for key, a in by["P47T"].items():
        c = by[tag].get(key)
        if not c:
            continue
        aok = int(float(a.get("SUCCESS_12S") or 0))
        cok = int(float(c.get("SUCCESS_12S") or 0))
        if aok != cok:
            transitions.append(
                {
                    "skill": key[0],
                    "traj_id": key[1],
                    "P47T": aok,
                    "P51": cok,
                    "direction": "gain" if cok and not aok else "loss",
                    "P47T_fail": a.get("fail_category"),
                    "P51_fail": c.get("fail_category"),
                }
            )

    lines = [
        "# PAIRED_EVAL_REPORT",
        "",
        f"Generated: {utc()}",
        "",
        f"Checkpoint: `{ckpt}`",
        "",
        "| Class | P47T | P50 | P51 | Target | P47T DROP | P51 DROP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    targets = {"A1": 8, "B1": 5, "C": 3, "D": 9}
    for sk in SKILLS:
        a, b, c = summaries["P47T"][sk], summaries.get("P50B", {}).get(sk, {}), summaries[tag][sk]
        lines.append(
            f"| {skill_label(sk)} | {a['SUCCESS_12S']}/{a['n']} | "
            f"{b.get('SUCCESS_12S', '?')}/{b.get('n', '?')} | {c['SUCCESS_12S']}/{c['n']} | "
            f"≥{targets[sk]}/{a['n']} | {a['DROP_BEFORE']} | {c['DROP_BEFORE']} |"
        )
    lines += ["", "## IC transitions vs P47T (SUCCESS_12S)", ""]
    for sk in SKILLS:
        sub = [t for t in transitions if t["skill"] == sk]
        gains = [t for t in sub if t["direction"] == "gain"]
        losses = [t for t in sub if t["direction"] == "loss"]
        lines.append(f"### {skill_label(sk)}: +{len(gains)} / −{len(losses)}")
        for t in gains[:8]:
            lines.append(f"- GAIN `{t['traj_id'][:48]}`")
        for t in losses[:8]:
            lines.append(f"- LOSS `{t['traj_id'][:48]}` ({t.get('P51_fail')})")
        lines.append("")
    write_md(OUT / "PAIRED_EVAL_REPORT.md", "\n".join(lines))
    (OUT / "PAIRED_EVAL_SUMMARY.json").write_text(
        json.dumps({"summaries": summaries, "transitions": transitions}, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n".join(lines[:20]), flush=True)
    return {"summaries": summaries, "transitions": transitions, "tag": tag}


def write_final(audit: dict, smoke: dict, train_out: dict, eval_out: dict) -> None:
    tag = eval_out.get("tag") or "P51"
    summ = eval_out.get("summaries") or {}
    p47, p50, p51 = summ.get("P47T") or {}, summ.get("P50B") or {}, summ.get(tag) or {}

    def s12(block, sk):
        return (block.get(sk) or {}).get("SUCCESS_12S", 0), (block.get(sk) or {}).get("n", 0)

    def drop(block, sk):
        return (block.get(sk) or {}).get("DROP_BEFORE", 0)

    targets = {"A1": 8, "B1": 5, "C": 3, "D": 9}
    rates = {}
    for sk in SKILLS:
        a0, n = s12(p47, sk)
        a1, _ = s12(p51, sk)
        rates[sk] = {"p47": a0, "p51": a1, "n": n, "target": targets[sk], "ok": a1 >= targets[sk]}

    promoted = all(rates[sk]["ok"] for sk in SKILLS)
    ac_ok = rates["A1"]["p51"] >= rates["A1"]["p47"] and rates["C"]["p51"] >= rates["C"]["p47"]
    bd_ok = rates["B1"]["p51"] >= 5 and rates["D"]["p51"] >= 9
    # looser: B/D vs P47T improved
    bd_vs47 = rates["B1"]["p51"] > rates["B1"]["p47"] and rates["D"]["p51"] > rates["D"]["p47"]

    if promoted:
        case = "A"
        nxt = "Expand PPO budget and verify multi-seed stability."
    elif ac_ok and not bd_vs47:
        case = "B"
        nxt = "Balanced sampling diluted B/D — try dynamic / class-aware sampling."
    elif bd_vs47 and not ac_ok:
        case = "C"
        nxt = "B/D kept gains but A/C still regress — investigate reward/gradient conflict (not more B-only)."
    else:
        case = "D"
        nxt = "Limited overall lift — check credit assignment / value generalization / KL-SFT strength."

    best = train_out.get("best") or {}
    if promoted:
        prom_path = OUT / "checkpoints" / "P51_promoted.pt"
        src = Path(best.get("path") or "")
        if src.is_file():
            import shutil

            shutil.copy2(src, prom_path)
            if src.with_suffix(".value.pt").is_file():
                shutil.copy2(src.with_suffix(".value.pt"), prom_path.with_suffix(".value.pt"))

    lines = [
        "# FINAL_REPORT — P51 Four-Class Mixed PPO",
        "",
        f"Generated: {utc()}",
        "",
        "## Held-out SUCCESS_12S",
        "",
        "| Class | P47T | P50 | P51 | Target | Met? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for sk in SKILLS:
        a0, n = s12(p47, sk)
        b0, _ = s12(p50, sk)
        a1, _ = s12(p51, sk)
        lines.append(
            f"| {skill_label(sk)} | {a0}/{n} | {b0}/{n} | {a1}/{n} | ≥{targets[sk]}/{n} | "
            f"{'YES' if rates[sk]['ok'] else 'NO'} |"
        )
    lines += [
        "",
        f"B DROP: P47T={drop(p47,'B1')} → P51={drop(p51,'B1')}; "
        f"A DROP: {drop(p47,'A1')}→{drop(p51,'A1')}; C DROP: {drop(p47,'C')}→{drop(p51,'C')}",
        "",
        f"**Promotion:** {'YES → P51_promoted' if promoted else 'NO — keep P47T baseline; P50/P51 experimental only'}",
        f"**Decision case:** {case}",
        "",
        "## Answers",
        "",
        f"1. **Mixed PPO ran successfully?** {'Yes' if smoke.get('all_pass') else 'Partial/smoke issues'}; "
        f"balanced ep sampling logged in training report.",
        f"2. **A/C recovered?** A {rates['A1']['p47']}→{rates['A1']['p51']}, "
        f"C {rates['C']['p47']}→{rates['C']['p51']} "
        f"({'floors held vs P47T' if ac_ok else 'still below / not fully recovered'}).",
        f"3. **B/D gains retained?** B {rates['B1']['p47']}→{rates['B1']['p51']} (P50=5), "
        f"D {rates['D']['p47']}→{rates['D']['p51']} (P50=9).",
        "4. **Reward/KL differ by class?** See training `per_class` components and ep/tr fractions in "
        "`PPO_TRAINING_REPORT.json` — reward definition identical; empirical returns may differ.",
        f"5. **Promotion rule met?** {'Yes' if promoted else 'No'}.",
        f"6. **Next?** {nxt}",
        "",
        f"Best experimental ckpt: `{best.get('path')}`",
        "Baselines retained: P47T (official), P50B (B-focused PPO control).",
    ]
    write_md(OUT / "FINAL_REPORT.md", "\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--total-steps", type=int, default=4096)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--per-class-train", type=int, default=24)
    ap.add_argument("--tag", default="P51")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    audit = {}
    if not args.skip_audit:
        audit = write_p50_audit()

    print("Building IC pools...", flush=True)
    pools = build_pools(per_class_train=args.per_class_train, per_class_val=4, b_per_family=8)
    (OUT / "IC_POOLS.json").write_text(
        json.dumps(
            {
                "counts": pools["counts"],
                "train_ids": {sk: [s.traj_id for s in pools["train"][sk]] for sk in SKILLS},
                "val_ids": {sk: [s.traj_id for s in pools["val"][sk]] for sk in SKILLS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_config(pools)
    smoke = run_smoke(pools)
    if not smoke["all_pass"]:
        write_md(OUT / "FINAL_REPORT.md", f"# FINAL_REPORT — P51 aborted\n\nSmoke failed.\nGenerated: {utc()}\n")
        return
    if args.smoke_only:
        return

    train_out = {"best": {"path": str(P47T_CKPT)}, "history": [], "candidates": [], "init": str(P47T_CKPT)}
    if not args.skip_train:
        cfg = P50PPOConfig(
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"=== Mixed PPO pilot steps={cfg.total_steps} ===", flush=True)
        train_out = train_mixed_ppo(
            train_pools=pools["train"],
            val_pools=pools["val"],
            cfg=cfg,
            tag=args.tag,
        )
        write_training_report(train_out)
        write_validation_report(train_out)

    ckpt = Path((train_out.get("best") or {}).get("path") or P47T_CKPT)
    if not ckpt.is_file():
        ckpt = P47T_CKPT
    eval_out = {"tag": args.tag, "summaries": {}, "transitions": []}
    if not args.skip_eval:
        eval_out = run_heldout(args.tag, ckpt)
        write_final(audit, smoke, train_out, eval_out)


if __name__ == "__main__":
    main()
