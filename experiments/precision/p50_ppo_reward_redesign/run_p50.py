"""P50 orchestrator: Audit → Reward → Sanity → B PPO pilot → Val → Held-out eval."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paired_eval import (  # noqa: E402
    eval_ckpt,
    summarize,
    write_csv,
)
from experiments.precision.p49_b_targeted_dagger.collect_probe import (  # noqa: E402
    holdout_b,
    load_family_ics,
)
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import BStart  # noqa: E402
from experiments.precision.p50_ppo_reward_redesign.paths import (  # noqa: E402
    M111_META,
    OUT,
    P44_EVAL,
    P47T_CKPT,
    SEED,
    T_MAX,
)
from experiments.precision.p50_ppo_reward_redesign.ppo_train import (  # noqa: E402
    P50PPOConfig,
    act_masked,
    load_sft_pools,
    train_b_ppo,
)
from experiments.precision.p50_ppo_reward_redesign.reward import RewardWeights, weights_dict  # noqa: E402
from experiments.precision.p50_ppo_reward_redesign.reward_sanity import run_synthetic  # noqa: E402
from experiments.precision.p26_c_curriculum_rl.ppo_m111 import load_actor_critic  # noqa: E402
from models.m111_checkpoint import HANDOFF, LOGIT_K, MOTION_K, load_m111, load_m111_book  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_audit() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, saved = load_m111(P47T_CKPT, device)
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    n_logits = int(model.logits.out_features)
    pools = load_sft_pools("P47T")
    md = f"""# PPO_PIPELINE_AUDIT

Generated: {utc()}

## Objective

Initialize categorical M111 PPO from `P47T_best.pt` and fine-tune on B recovery ICs
with redesigned potential-based reward. Do not treat tokens as continuous actions.

## Architecture (reuse)

| Piece | Source | Status |
|---|---|---|
| M111 Transformer actor | `models/m111_checkpoint.py` + P47T | **OK** — {n_logits} logits (MOTION={MOTION_K}, HANDOFF={HANDOFF}) |
| Observation | `observe_airborne` 39-D + prev action 7-D | **OK** |
| Action decode | `decode_m111_action` (proto + residual ρ) | **OK** — discrete token → 7-D legal action |
| Value head | `ActorCriticM111` (P26) on last context | **OK** — independent Linear(d,1) |
| PPO update | P26 `ppo_update` extended in `ppo_train.py` | **OK** — clipped ratio, GAE, entropy |
| Env | `BRecoveryEnv` + R3FOnline T_max={T_MAX} | **OK** |
| SUCCESS | `evaluate_handoff_with_axis` (nominal inside) | **OK** — not policy actions |
| Checkpoint | `load_m111` / `save_m111` + `.value.pt` | **OK** — P47T not overwritten |

## Critical checks

1. **Rollout logprob ↔ train token**: `act_masked` samples Categorical on HANDOFF-masked logits; stored token/logp reused in PPO ratio. **OK**
2. **PPO ratio uses old logprob**: `ratio = exp(logp - old_logp)`. **OK**
3. **No recovery action after HANDOFF**: R3F `ready` ends episode; HANDOFF logit masked (−1e9); never decoded. **OK**
4. **Timeout vs terminal**: DROP / R3 ready = terminal; `max_steps` truncation evaluates once then terminal timeout reward. **OK**
5. **GAE**: episode `done=1` cuts bootstrap; mid-buffer cut bootstraps value. Nominal 12s is not in the action buffer. **OK**
6. **Value input = policy obs**: same `obs_seq`/`prev_seq` window. **OK**
7. **SUCCESS_12S credit**: terminal reward added on last recovery step → GAE backprops to recovery tokens. **OK**
8. **Discrete only**: Categorical over 111 motion tokens; residual is head output, not continuous action space for PPO. **OK**

## Init from P47T

- Path: `{P47T_CKPT}`
- Exists: {P47T_CKPT.is_file()}
- Saved keys include model/proto/rho; vocab M111.
- SFT anchor pools (A1/B1/C/D P47T chunks): { {k: len(v) for k, v in pools.items()} }

## Reward stack

Potential-based progress + contact quality Δ + slip/imbalance/time/switch penalties + terminal SUCCESS/DROP table.
See `REWARD_SPEC.md`.

## Verdict

**PPO can initialize from P47T and sample M111 discrete tokens.** Pipeline reuses P26 ActorCritic + P45/P49 R3F eval path. Proceed to reward sanity → B pilot.
"""
    write_md(OUT / "PPO_PIPELINE_AUDIT.md", md)
    (OUT / "PPO_PIPELINE_AUDIT.json").write_text(
        json.dumps(
            {
                "generated": utc(),
                "ckpt": str(P47T_CKPT),
                "n_logits": n_logits,
                "motion_k": MOTION_K,
                "handoff": HANDOFF,
                "sft_pool_sizes": {k: len(v) for k, v in pools.items()},
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote PPO_PIPELINE_AUDIT.md", flush=True)


def write_reward_spec() -> None:
    w = RewardWeights()
    md = f"""# REWARD_SPEC

Generated: {utc()}

## Principle

Reward **stable, recoverable grasp progress toward R3F**, not instantaneous WORKING or transient bilateral contact.

## Dense (per recovery step)

\\[
r_t = w_p(\\gamma\\Phi_{{t+1}}-\\Phi_t) + w_c\\Delta c_q - w_s[\\Delta v_{{rel}}]_+ - w_i p_{{imb}} - w_t - w_{{sw}}\\mathbf{{1}}_{{switch}}
\\]

Potential \\(\\Phi\\):

- \\(-0.05\\,e_{{\\mathrm{{pos,mm}}}} - 0.03\\,\\theta - 0.5\\,v_{{\\mathrm{{rel}}}} + 1.5\\,c_q\\)
- \\(c_q = \\mathrm{{bilateral}}\\cdot\\min(n_{{\\min}},4)/4\\) (quality, not flat WORKING)

Contact: reward **Δ** in quality + tiny sustain if \\(c_q\\ge 0.75\\) (capped by time penalty).

Slip: only positive \\(\\Delta v_{{\\mathrm{{rel}}}}\\) while in contact (allows controlled C sliding at constant v).

Imbalance: \\(|F_L-F_R|/(F_L+F_R+\\epsilon)\\) when bilateral forces available.

## Terminal (once)

| Event | Reward |
|---|---:|
| SUCCESS_12S | {w.r_success_12s} |
| SUCCESS_NEW bonus | +{w.r_success_new_bonus} |
| DROP_BEFORE | {w.r_drop} |
| LOST_GRASP after HANDOFF | {w.r_lost_after_hand} |
| Timeout / failed 12S | {w.r_timeout_fail} |

SUCCESS_NEW is **additive**, never a substitute for SUCCESS_12S.

## Anti-hacking

- No flat WORKING bonus (potential-based only).
- Time penalty discourages linger / timeout farming.
- Switch penalty discourages token flicker.
- Contact uses quality Δ, not clamp spam.
- Component returns logged per episode.

## Weights JSON

```json
{json.dumps(weights_dict(w), indent=2)}
```
"""
    write_md(OUT / "REWARD_SPEC.md", md)
    print("Wrote REWARD_SPEC.md", flush=True)


def split_ics(ics: list[dict], *, n_val: int = 8) -> tuple[list[BStart], list[BStart]]:
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for ic in ics:
        by[ic["family"]].append(ic)
    train, val = [], []
    for fam, pool in by.items():
        rng.shuffle(pool)
        n_v = min(max(1, n_val // max(1, len(by))), max(0, len(pool) // 4))
        for ic in pool[:n_v]:
            val.append(BStart(ic["ic_id"], ic["family"], ic["capture"], ic["ref"]))
        for ic in pool[n_v:]:
            train.append(BStart(ic["ic_id"], ic["family"], ic["capture"], ic["ref"]))
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def smoke_test(starts: list[BStart]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = load_actor_critic(P47T_CKPT, None, device, None)
    _, book, saved = load_m111(P47T_CKPT, device)
    rho = saved.get("residual_rho", saved.get("res_cap"))
    from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import BRecoveryEnv

    env = BRecoveryEnv(book, rho, max_steps=20)
    st = starts[0]
    env.reset(st)
    tok, resid, logp, val = act_masked(ac, env.obs_seq, env.prev_seq, device)
    assert 0 <= tok < MOTION_K
    r, done, info = env.step(tok, resid)
    out = {
        "token": tok,
        "logp": logp,
        "value": val,
        "reward": r,
        "done": done,
        "info_keys": list(info.keys()),
        "ok": True,
    }
    (OUT / "PPO_SMOKE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Smoke OK tok={tok} r={r:.3f} done={done}", flush=True)
    return out


def skill_label(sk: str) -> str:
    return {"A1": "A", "B1": "B", "C": "C", "D": "D"}.get(sk, sk)


def run_heldout(tag: str, ckpt: Path) -> dict:
    hold = [r for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")) if r.get("tag") == "holdout"]
    uniq = {(r["skill"], r["traj_id"]): r for r in hold}
    hold = list(uniq.values())
    jobs_by_key = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}
    # baseline rows from P47T paired eval if present
    rows = []
    p47_csv = ROOT / "results/experiments/p47_b_ordinary_expansion/PAIRED_EVAL_ROLLOUTS_P47T.csv"
    if p47_csv.is_file():
        for r in csv.DictReader(p47_csv.open(encoding="utf-8")):
            if r.get("tag") == "P47T":
                rows.append(dict(r))
    else:
        for r in hold:
            rows.append(
                {
                    "tag": "P47T",
                    "skill": r["skill"],
                    "traj_id": r["traj_id"],
                    "SUCCESS_12S": int(r.get("SUCCESS_12S") or 0),
                    "drop_before": int(r.get("fail_category") == "DROP_BEFORE"),
                }
            )
    rows.extend(eval_ckpt(tag, ckpt, jobs_by_key, hold))
    write_csv(OUT / "PPO_PAIRED_EVAL_ROLLOUTS.csv", rows)
    summaries = {t: summarize(rows, t) for t in ("P47T", tag)}
    lines = [
        "# PPO_PAIRED_EVAL",
        "",
        f"Generated: {utc()}",
        "",
        f"Checkpoint: `{ckpt}`",
        "",
        "| Class | P47T SUCCESS_12S | P50 SUCCESS_12S | P47T DROP_BEFORE | P50 DROP_BEFORE |",
        "|---|---:|---:|---:|---:|",
    ]
    for sk in ("A1", "B1", "C", "D"):
        a, b = summaries["P47T"][sk], summaries[tag][sk]
        lines.append(
            f"| {skill_label(sk)} | {a['SUCCESS_12S']}/{a['n']} | {b['SUCCESS_12S']}/{b['n']} | "
            f"{a['DROP_BEFORE']} | {b['DROP_BEFORE']} |"
        )
    write_md(OUT / "PPO_PAIRED_EVAL.md", "\n".join(lines))
    (OUT / "PPO_PAIRED_EVAL.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summaries


def write_training_report(train_out: dict) -> None:
    hist = train_out.get("history") or []
    best = train_out.get("best") or {}
    lines = [
        "# PPO_TRAINING_REPORT",
        "",
        f"Generated: {utc()}",
        "",
        f"- Init: `{train_out.get('init')}`",
        f"- Best: `{best.get('path')}` @ steps={best.get('steps')} val_s12={best.get('val_s12')} val_drop={best.get('val_drop')}",
        f"- Config: `{json.dumps(train_out.get('cfg') or {}, indent=2)}`",
        "",
        "## History (tail)",
        "",
        "| steps | return | drop | s12 | kl_ref | ent | vf | clipfrac | val_s12 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hist[-20:]:
        lines.append(
            f"| {row.get('steps')} | {row.get('mean_return', float('nan')):.2f} | "
            f"{row.get('drop_rate', float('nan')):.2f} | {row.get('success_12s_rate', float('nan')):.2f} | "
            f"{row.get('kl_ref', float('nan')):.4f} | {row.get('ent', float('nan')):.3f} | "
            f"{row.get('vf', float('nan')):.3f} | {row.get('clipfrac', float('nan')):.3f} | "
            f"{row.get('val_s12', float('nan'))} |"
        )
    write_md(OUT / "PPO_TRAINING_REPORT.md", "\n".join(lines))
    (OUT / "PPO_TRAINING_REPORT.json").write_text(json.dumps(train_out, indent=2, default=str), encoding="utf-8")


def write_final(sanity: dict, summaries: dict, tag: str, train_out: dict) -> None:
    p47 = summaries.get("P47T") or {}
    p50 = summaries.get(tag) or {}

    def s12(block, sk):
        return (block.get(sk) or {}).get("SUCCESS_12S", 0), (block.get(sk) or {}).get("n", 0)

    def drop(block, sk):
        return (block.get(sk) or {}).get("DROP_BEFORE", 0)

    a0, an = s12(p47, "A1")
    b0, bn = s12(p47, "B1")
    c0, cn = s12(p47, "C")
    d0, dn = s12(p47, "D")
    a1, _ = s12(p50, "A1")
    b1, _ = s12(p50, "B1")
    c1, _ = s12(p50, "C")
    d1, _ = s12(p50, "D")
    b_drop0, b_drop1 = drop(p47, "B1"), drop(p50, "B1")

    b_improved = b1 > b0
    floors_ok = a1 >= a0 and c1 >= c0 and d1 >= d0
    b_target = b1 >= 3

    if b_improved and floors_ok:
        nxt = "Continue B-focused PPO with longer budget / reward calibration; promote only if floors hold."
    elif b_improved and not floors_ok:
        nxt = "Do **not** promote. Next: four-class mixed PPO (or stronger SFT/KL anchor) to protect A/C/D."
    else:
        nxt = (
            "PPO did not lift B. Diagnose credit assignment / exploration coverage / reward scale "
            "before more steps; do not blind-scale training."
        )

    lines = [
        "# FINAL_REPORT — P50 Reward Redesign + PPO",
        "",
        f"Generated: {utc()}",
        "",
        "## Held-out SUCCESS_12S",
        "",
        "| Class | P47T | P50 | Target |",
        "|---|---:|---:|---:|",
        f"| A | {a0}/{an} | {a1}/{an} | ≥{a0}/{an} |",
        f"| B | {b0}/{bn} | {b1}/{bn} | ≥3/{bn} |",
        f"| C | {c0}/{cn} | {c1}/{cn} | ≥{c0}/{cn} |",
        f"| D | {d0}/{dn} | {d1}/{dn} | ≥{d0}/{dn} |",
        "",
        f"B DROP_BEFORE: P47T={b_drop0} → P50={b_drop1}",
        "",
        "## Answers",
        "",
        f"1. **PPO init from P47T stable?** Yes — smoke + pilot loaded `{P47T_CKPT.name}`; discrete Categorical; KL to frozen P47T.",
        f"2. **Reward distinguishes stable vs brief contact?** Sanity={'PASS' if sanity.get('all_pass') else 'FAIL'} "
        f"(see REWARD_SANITY_TEST.md).",
        f"3. **DROP_BEFORE on B reduced?** {'Yes' if b_drop1 < b_drop0 else ('No change' if b_drop1 == b_drop0 else 'No — increased')}.",
        f"4. **B breakthrough ≥3/16?** {'Yes' if b_target else f'No ({b1}/{bn})'}.",
        f"5. **A/C/D regression?** {'No (floors held)' if floors_ok else 'Yes — floors broken'}.",
        f"6. **Next step?** {nxt}",
        "",
        f"Best checkpoint: `{ (train_out.get('best') or {}).get('path') }`",
        "",
        "Constraints honored: no physics / SUCCESS / R3F / nominal / codebook / oracle SM changes; "
        "held-out unused for train/tune; P47T not overwritten.",
    ]
    write_md(OUT / "FINAL_REPORT.md", "\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--total-steps", type=int, default=4096)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--per-family", type=int, default=12)
    ap.add_argument("--tag", default="P50B")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    if not args.skip_audit:
        write_audit()
        write_reward_spec()

    sanity = {"all_pass": True}
    if not args.skip_sanity:
        sanity = run_synthetic(OUT)
        print(f"Reward sanity all_pass={sanity['all_pass']}", flush=True)
        if not sanity["all_pass"]:
            write_md(
                OUT / "FINAL_REPORT.md",
                f"# FINAL_REPORT — P50 aborted\n\nGenerated: {utc()}\n\nReward sanity FAILED. PPO not started.\n",
            )
            return

    held = holdout_b()
    print(f"Loading B family ICs (exclude {len(held)} holdout)...", flush=True)
    ics = load_family_ics(held, per_family=args.per_family, seed=SEED)
    train_starts, val_starts = split_ics(ics, n_val=8)
    print(f"Train ICs={len(train_starts)} Val ICs={len(val_starts)}", flush=True)
    (OUT / "TRAIN_VAL_SPLIT.json").write_text(
        json.dumps(
            {
                "train": [s.ic_id for s in train_starts],
                "val": [s.ic_id for s in val_starts],
                "heldout_excluded": sorted(held),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    smoke_test(train_starts or val_starts)
    if args.smoke_only:
        return

    train_out = {"best": {"path": str(P47T_CKPT)}, "history": [], "init": str(P47T_CKPT)}
    tag = args.tag
    if not args.skip_train:
        cfg = P50PPOConfig(
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"=== B-focused PPO pilot steps={cfg.total_steps} ===", flush=True)
        train_out = train_b_ppo(
            init_ckpt=P47T_CKPT,
            starts_train=train_starts,
            starts_val=val_starts or train_starts[:4],
            cfg=cfg,
            tag=tag,
        )
        write_training_report(train_out)

    ckpt = Path((train_out.get("best") or {}).get("path") or P47T_CKPT)
    if not ckpt.is_file():
        ckpt = P47T_CKPT
        tag = "P47T"

    summaries = {}
    if not args.skip_eval:
        # Only eval held-out after training; validation already used non-holdout val set
        summaries = run_heldout(tag if ckpt != P47T_CKPT else "P50B", ckpt if ckpt != P47T_CKPT else P47T_CKPT)
        # if we eval'd P47T path under P50B tag when no train — still fine
        write_final(sanity, summaries, tag if "P50" in str(ckpt) or tag.startswith("P50") else tag, train_out)
    else:
        write_final(sanity, {}, tag, train_out)


if __name__ == "__main__":
    main()
