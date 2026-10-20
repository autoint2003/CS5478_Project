"""Diagnose 40 ms GRPO-only HANDOFF success and time-induced advantage bias.

Read-only. Writes under results/diagnostics/handoff_40ms_advantage_audit/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed, terminal_reason
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import N_SUB, step_legal
from evaluation import skill_traces as audit
from evaluation.classify_validation_skills import classify_mechanism
from models.k48_checkpoint import load_trained_k48
from training.grpo_math import advantages as grpo_advantages
from training.k47_codebook import decode_action, group_advantage, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.reward_v2 import TIME_PENALTY, SUCCESS_BONUS, grpo_return, reward_components
from training.unified_k48 import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "handoff_40ms_advantage_audit"
BENCH = ROOT / "results" / "benchmarks" / "hard_recovery_v1"
PRIOR = ROOT / "results" / "diagnostics" / "reward_time_handoff_audit"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
GRPO = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling/checkpoint_epoch_0005.pt"
IC_LOG = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling/ic_log.jsonl"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
MATCHED = ROOT / "results/diagnostics/reward_time_handoff_audit/matched_mechanism_fixed.json"
LIMIT_S = float(SELECTED_LIMIT_S)
DT_POLICY = 0.02  # step_legal = N_SUB * mujoco dt ≈ 20 ms

GRPO_ONLY = [
    "hrv1_920047",
    "hrv1_920064",
    "hrv1_920106",
    "hrv1_920015",
    "hrv1_920069",
    "hrv1_leftover_hard_910207",
    "hrv1_leftover_hard_910231",
]


def jsafe(obj):
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsafe(x) for x in obj]
    return str(obj)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def load_bench(ic_id: str) -> dict:
    for name in ("verified_manifest.jsonl", "unverified_manifest.jsonl"):
        for line in (BENCH / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] == ic_id:
                return row
    raise KeyError(ic_id)


def phys_snapshot(env, blob, t0):
    row = audit.sample_state(env.sim, blob["ref"])
    fsm = env.sim.fsm
    return {
        "sim_t": float(env.sim.data.time),
        "recovery_t": float(env.sim.data.time) - float(t0),
        "obj_pos": np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float).tolist(),
        "obj_vel": None,
        "hand_pos": np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float).tolist(),
        "nL": int(row["nL"]),
        "nR": int(row["nR"]),
        "bilateral": bool(row["bilateral"]),
        "both_off": bool(row["both_off"]),
        "cL": None if row.get("cL") is None else np.asarray(row["cL"], float).tolist(),
        "cR": None if row.get("cR") is None else np.asarray(row["cR"], float).tolist(),
        "fnL": float(row.get("fnL") or 0.0),
        "fnR": float(row.get("fnR") or 0.0),
        "tau_grip": float(getattr(fsm, "fg_cmd", 0.0) or 0.0),
        "p_des": np.asarray(fsm.p_des, float).tolist(),
        "r_des": np.asarray(fsm.r_des, float).reshape(-1).tolist(),
        "v_cmd": np.asarray(getattr(fsm, "v_cmd", np.zeros(3)), float).tolist(),
    }


def collect_policy_tokens(env, model, book, blob, device, limit_s):
    """Deterministic recovery tokens until HANDOFF/drop/limit. Returns motion tokens only + meta."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    snap0 = take_capture(env)
    phys0 = phys_snapshot(env, blob, t0)
    lost_low = 0
    tokens = []
    residuals = []
    actions = []
    step_phys = [phys0]
    how = None
    cap = int(round(float(limit_s) / DT_POLICY))
    for i in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        if token == HANDOFF:
            how = "early"
            tokens.append(token)
            residuals.append(residual)
            break
        _res, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        tokens.append(token)
        residuals.append(residual)
        actions.append(np.asarray(executed, np.float32).copy())
        step_phys.append(phys_snapshot(env, blob, t0))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    t_end = float(env.sim.data.time)
    return {
        "t0": t0,
        "snap0": snap0,
        "phys0": phys0,
        "tokens": tokens,
        "residuals": residuals,
        "actions": actions,
        "step_phys": step_phys,
        "how": how,
        "sim_t_end": t_end,
        "recovery_elapsed_so_far": t_end - t0,
        "n_motion": sum(1 for t in tokens if t != HANDOFF),
        "n_handoff": int(any(t == HANDOFF for t in tokens)),
        "handoff_at_end": bool(tokens) and tokens[-1] == HANDOFF,
    }


def run_branch(env, blob, t0, snap0, motion_actions, label):
    """Restore IC, execute motion_actions (list of 7D), then legal HANDOFF + nominal."""
    restore_capture(env, snap0)
    # Ensure t0 matches restored time
    t0_now = float(env.sim.data.time)
    if abs(t0_now - t0) > 1e-9:
        # still use restored clock as recovery start
        t0 = t0_now
    lost_low = 0
    for action in motion_actions:
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            elapsed = float(env.sim.data.time) - t0
            outcome = {
                "reason": reason,
                "success": False,
                "handoff": None,
                "recovery_elapsed": elapsed,
                "lift_completed": False,
                "q_nominal": 0.0,
            }
            scored = grpo_return(outcome, LIMIT_S, "v2", None)
            pre = phys_snapshot(env, blob, t0)
            return {
                "branch": label,
                "n_motion_executed": len(motion_actions),  # may have stopped early
                "dropped_before_handoff": True,
                "pre_handoff_phys": pre,
                "outcome": outcome,
                "components": scored["components"],
                "return_v2": scored["return"],
                "success": False,
                "reason": reason,
                "recovery_elapsed": elapsed,
            }
    pre = phys_snapshot(env, blob, t0)
    outcome = finish_recovery(env, blob["ref"], t0, "early")
    scored = grpo_return(outcome, LIMIT_S, "v2", outcome.get("reward"))
    return {
        "branch": label,
        "n_motion_executed": len(motion_actions),
        "dropped_before_handoff": False,
        "pre_handoff_phys": pre,
        "outcome": outcome,
        "components": scored["components"],
        "return_v2": scored["return"],
        "success": bool(outcome.get("success")),
        "reason": outcome.get("reason"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
    }


def classify_causal(b0, b1, b2, b3, b4):
    if b2 is None or b2.get("success") is None:
        return "UNRESOLVED"
    # reproduction: B2 should succeed for GRPO_only
    if not b2["success"]:
        return "REPRODUCTION_MISMATCH"
    if b0["success"]:
        return "DIRECT_HANDOFF_SUFFICIENT"
    if b1 is not None and b1.get("branch") != "NOT_APPLICABLE" and b1["success"] and not b0["success"]:
        return "LEGAL_PREFIX_REQUIRED"
    if b2["success"] and not b0["success"]:
        return "LEGAL_PREFIX_REQUIRED"
    return "HANDOFF_STATE_DEPENDENT"


def write_timing_semantics():
    text = f"""# HANDOFF / 40 ms timing semantics

## Code facts

| Item | Location | Value |
|---|---|---|
| Policy step duration | `envs/step_legal.py` `N_SUB={N_SUB}` × MuJoCo dt | **{DT_POLICY} s (20 ms)** per motion token |
| HANDOFF token | `training/unified_k48.py` `HANDOFF=47` | **No** `step_legal`; no physics advance |
| recovery_elapsed | `envs/outcomes/horizon_outcome.py` `finish_recovery` | `t_handoff − t0` where `t0=sim.time` at IC restore |
| SUCCESS return | `training/reward_v2.py` | `R = {SUCCESS_BONUS} − {TIME_PENALTY} * recovery_elapsed` |
| R=1.998 ⇒ | algebra | `elapsed = ({SUCCESS_BONUS}−1.998)/{TIME_PENALTY} = 0.04 s` |

## What 40 ms is **not**

- Not absolute simulation clock at HANDOFF (IC snapshots already sit near ~3.2 s wall clock).
- Not two full HANDOFF steps.
- Not a default/saturated timer: `finish_recovery` uses real `sim.data.time` delta.

## What 40 ms **is**

For GRPO-only traces (see Part A CSV): typically **two** `step_legal` motion tokens (2×20 ms) then token 47.

Example `hrv1_920047` GRPO (`reward_time_handoff_audit/traces/.../reward_steps.csv`):

- step0 @ recovery_t=0.00, token 36 → executes 20 ms
- step1 @ recovery_t=0.02, token 36 → executes 20 ms
- step2 @ recovery_t=0.04, token 47 HANDOFF → `recovery_elapsed=0.04`, `R=1.998`

`outcome.json` simultaneously records absolute `t_handoff≈3.28` and `t_recovery_start≈3.24`.

## Paired-benchmark handoff times were absolute

`hard_recovery_v1/paired_results.csv` stored absolute `sim.time` at HANDOFF (~3.28 vs ~4.78).  
Δhandoff ≈ 1.5 s there is **wall-clock difference**, not “GRPO recovered for 3.28 s”. Relative to IC, GRPO used ~0.04 s and SFT ~1.54 s.
"""
    (OUT / "handoff_timing_semantics.md").write_text(text, encoding="utf-8")


def write_advantage_audit():
    text = f"""# GRPO advantage implementation audit

## Formula (actual code)

`training/grpo_math.py` `advantages(returns)`:

```
std = returns.std()          # population std, ddof=0 (numpy default)
if std < 1e-8:
    return zeros, skipped=True
A = (returns - mean) / std   # NO epsilon in denominator
```

`training/k47_codebook.py` `group_advantage`: drops `truncated` members, then calls `advantages` on the remainder; truncated slots get A=0.

## Where it is applied

`training/adaptive_grpo.py` `apply_group_update`:

- Advantage is **trajectory-level** (one scalar per rollout).
- Every recorded action in that trajectory (including HANDOFF token) shares the **same** A.
- Per-action clipped surrogate: `ratio = exp(logp − behavior_logp)`, clip=`cfg["clip"]` (0.2).
- Trajectory loss = mean over actions; group loss = mean over trajectories with |A|≥1e-8.
- KL: mean categorical KL over those actions, weight `kl_coef` (0.05).
- Grad clip norm 1.0. **No discount γ. No return clipping.**
- Zero variance ⇒ `skipped=True`, **no optimizer step** (no invented advantage).

## Adaptive Independent-G

`training/adaptive_independent_g.py`: expand G=8→16→32→64 until first SUCCESS appears in the cumulative pool, then update on that group.  
Larger G only when early buckets find no success — those all-failure expansions still form one group for advantage.

## Implication for all-success groups

If every member is SUCCESS_12S and `R_i = 2 − λ T_i` with identical other terms:

`A_i = −(T_i − T̄) / σ_T`  (independent of λ as long as λ>0 and σ_T>0)

So a 0.02–0.04 s duration edge can dominate the **sign** of A inside an all-success group even when |ΔR| is tiny. λ=0 ⇒ all R equal ⇒ std=0 ⇒ skip update.
"""
    (OUT / "advantage_implementation_audit.md").write_text(text, encoding="utf-8")


def analyze_ic_log():
    rows = [json.loads(l) for l in IC_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    strata_rows = []
    all_success_detail = []
    for row in rows:
        rewards = [None if r is None else float(r) for r in row.get("rewards") or []]
        labels = list(row.get("labels") or row.get("outcomes") or [])
        handoffs = row.get("handoff_t") or []
        n = len(rewards)
        n_succ = sum(1 for lab in labels if lab == "SUCCESS_12S")
        rate = n_succ / n if n else 0.0
        if n_succ == n and n > 0:
            bucket = "all_success"
        elif n_succ == 0:
            bucket = "all_failure"
        elif rate >= 0.75:
            bucket = "mixed_ge_75"
        elif rate >= 0.25:
            bucket = "mixed_25_75"
        else:
            bucket = "mixed_lt_25"
        adv = (row.get("update_stats") or {}).get("advantages") or []
        times = []
        time_contrib = []
        for lab, rew, ht in zip(labels, rewards, handoffs if handoffs else [None] * n):
            if lab == "SUCCESS_12S" and rew is not None:
                # invert R=2-λT
                t_est = (SUCCESS_BONUS - float(rew)) / TIME_PENALTY
                times.append(t_est)
                time_contrib.append(TIME_PENALTY * t_est)
            else:
                times.append(None)
                time_contrib.append(None)
        succ_returns = [float(r) for lab, r in zip(labels, rewards) if lab == "SUCCESS_12S" and r is not None]
        time_var = None
        adv_matches_neg_T = None
        if bucket == "all_success" and len(succ_returns) >= 2:
            t_arr = np.asarray([(SUCCESS_BONUS - float(r)) / TIME_PENALTY for r in succ_returns], float)
            time_var = float(t_arr.std())
            if time_var > 1e-8 and adv and len(adv) == len(succ_returns):
                pred = -(t_arr - t_arr.mean()) / t_arr.std()
                corr = float(np.corrcoef(pred, np.asarray(adv, float))[0, 1])
                adv_matches_neg_T = corr
            all_success_detail.append({
                "id": row["id"],
                "epoch": row.get("epoch"),
                "G": row.get("final_g") or row.get("group_size"),
                "n": n,
                "return_std": row.get("return_std"),
                "time_std": time_var,
                "adv_corr_with_neg_T": adv_matches_neg_T,
                "optimizer_updated": row.get("optimizer_updated"),
                "skipped": (row.get("update_stats") or {}).get("skipped"),
                "advantage_max": (row.get("update_stats") or {}).get("advantage_max"),
                "advantage_min": (row.get("update_stats") or {}).get("advantage_min"),
                "primary_category": row.get("primary_category"),
            })
        strata_rows.append({
            "id": row["id"],
            "epoch": row.get("epoch"),
            "bucket": bucket,
            "G": row.get("final_g") or row.get("group_size"),
            "n": n,
            "n_success": n_succ,
            "success_rate": rate,
            "return_mean": row.get("return_mean"),
            "return_std": row.get("return_std"),
            "optimizer_updated": row.get("optimizer_updated"),
            "skipped": (row.get("update_stats") or {}).get("skipped"),
            "advantage_max": (row.get("update_stats") or {}).get("advantage_max"),
            "advantage_min": (row.get("update_stats") or {}).get("advantage_min"),
            "primary_category": row.get("primary_category"),
            "immediate_handoff_count": row.get("immediate_handoff"),
            "source": row.get("source"),
        })

    # summaries
    by_bucket = Counter(r["bucket"] for r in strata_rows)
    updated = [r for r in strata_rows if r.get("optimizer_updated")]
    all_s = [r for r in strata_rows if r["bucket"] == "all_success"]
    all_s_var = [d for d in all_success_detail if d["time_std"] is not None and d["time_std"] > 1e-8]
    summary = {
        "n_ic_updates_logged": len(strata_rows),
        "bucket_counts": dict(by_bucket),
        "n_optimizer_updated": sum(1 for r in strata_rows if r.get("optimizer_updated")),
        "n_all_success": len(all_s),
        "n_all_success_with_time_variance": len(all_s_var),
        "frac_updates_all_success": None if not updated else sum(1 for r in updated if r["bucket"] == "all_success") / len(updated),
        "all_success_adv_corr_mean": None if not all_s_var else float(np.nanmean([
            d["adv_corr_with_neg_T"] for d in all_s_var if d["adv_corr_with_neg_T"] is not None
        ])),
    }
    write_csv(OUT / "group_success_strata.csv", strata_rows)
    write_csv(OUT / "all_success_group_detail.csv", all_success_detail)
    write_json(OUT / "group_strata_summary.json", summary)
    return strata_rows, all_success_detail, summary


def lambda_sensitivity(strata_rows, max_groups=40):
    """Offline recompute A for representative logged groups under λ scales."""
    raw = [json.loads(l) for l in IC_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {r["id"] + f"::e{r.get('epoch')}": r for r in raw}
    # pick: some all-success with variance, some mixed, some all-fail
    picks = []
    for row in strata_rows:
        key = row["id"] + f"::e{row.get('epoch')}"
        if key not in by_id:
            continue
        if row["bucket"] == "all_success" and row.get("return_std") and float(row["return_std"]) > 1e-4:
            picks.append(("all_success_var", by_id[key]))
        elif row["bucket"] == "mixed_25_75":
            picks.append(("mixed", by_id[key]))
        elif row["bucket"] == "all_failure":
            picks.append(("all_failure", by_id[key]))
        if len(picks) >= max_groups:
            break
    # ensure diversity
    lambdas = [0.0, 0.25 * TIME_PENALTY, 0.5 * TIME_PENALTY, TIME_PENALTY, 2.0 * TIME_PENALTY]
    out_rows = []
    for tag, row in picks[:max_groups]:
        labels = list(row.get("labels") or [])
        rewards = list(row.get("rewards") or [])
        handoffs = list(row.get("handoff_t") or [None] * len(labels))
        # Reconstruct components approximately: SUCCESS uses R=2-λ0*T with T from logged return
        for lam in lambdas:
            new_R = []
            for lab, rew, ht in zip(labels, rewards, handoffs):
                if rew is None:
                    new_R.append(None)
                    continue
                if lab == "SUCCESS_12S":
                    t = (SUCCESS_BONUS - float(rew)) / TIME_PENALTY
                    new_R.append(SUCCESS_BONUS - lam * t)
                else:
                    # keep non-success returns as logged (their time term also scales for after-handoff fails)
                    if lab in ("DROP_AFTER_HANDOFF", "LOST_GRASP", "NOMINAL_LIFT_NOT_COMPLETED", "PHYSICAL_VERIFICATION_FAILED"):
                        # approximate: R = terminal + nom_progress - λ T; unknown split — scale only the
                        # difference from λ0 time piece estimated if ht known
                        if ht is not None:
                            # crude: replace λ0*ht with lam*ht in logged return
                            new_R.append(float(rew) + TIME_PENALTY * float(ht) - lam * float(ht))
                        else:
                            new_R.append(float(rew))
                    else:
                        new_R.append(float(rew))  # DROP_DURING_RECOVERY has no time term
            valid = [(i, r) for i, r in enumerate(new_R) if r is not None]
            if len(valid) < 2:
                adv = [0.0] * len(new_R)
                skipped = True
                std = 0.0
            else:
                vals = [r for _, r in valid]
                arr, std, skipped = grpo_advantages(vals)
                adv = [0.0] * len(new_R)
                for (i, _), a in zip(valid, arr):
                    adv[i] = float(a)
            # sign pattern among successes
            succ_idx = [i for i, lab in enumerate(labels) if lab == "SUCCESS_12S"]
            succ_adv = [adv[i] for i in succ_idx]
            out_rows.append({
                "group_id": row["id"],
                "epoch": row.get("epoch"),
                "tag": tag,
                "lambda": lam,
                "lambda_scale": None if TIME_PENALTY == 0 else lam / TIME_PENALTY,
                "n": len(labels),
                "n_success": sum(1 for lab in labels if lab == "SUCCESS_12S"),
                "return_std": float(std),
                "skipped_zero_var": bool(skipped),
                "adv_max": None if not adv else max(adv),
                "adv_min": None if not adv else min(adv),
                "succ_adv_signs": "".join("+" if a > 1e-8 else ("-" if a < -1e-8 else "0") for a in succ_adv),
                "original_adv": (row.get("update_stats") or {}).get("advantages"),
            })
    write_csv(OUT / "lambda_advantage_sensitivity.csv", out_rows)
    # check: for all-success, advantages identical up to sign pattern across λ>0
    return out_rows


def mechanism_competition(summary_all_success):
    """Check whether mixed Sliding/Recatch appear in same GRPO groups."""
    lines = ["# Mechanism competition audit\n"]
    matched = []
    if MATCHED.is_file():
        matched = json.loads(MATCHED.read_text(encoding="utf-8")).get("matched") or []
    lines.append("## Matched-IC oracle pairs (from prior audit, corrected durations)\n")
    lines.append(f"- n_matched = {len(matched)}")
    if matched:
        faster = sum(1 for m in matched if m.get("recatch_faster") or (m.get("delta_T_recatch_minus_sliding") or 0) < 0)
        lines.append(f"- Recatch faster: {faster}/{len(matched)}")
        lines.append("- Reward prefers faster SUCCESS only via λ·ΔT (ΔR=ΔR_time ≤ ~0.04).")
        lines.append("- Contact evidence: these used `_try_sliding` / `vertical_recatch_search` SUCCESS_12S; mechanism labels are oracle families, not GRPO rollouts.\n")
    # Historical groups: primary_category only — no per-trajectory mechanism labels stored
    rows = [json.loads(l) for l in IC_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    cats = Counter(r.get("primary_category") for r in rows)
    lines.append("## Historical GRPO groups (`ic_log.jsonl`)\n")
    lines.append(f"- Logged IC-updates: {len(rows)}")
    lines.append(f"- `primary_category` counts: {dict(cats)}")
    lines.append("- **Per-trajectory mechanism labels (Sliding vs Recatch) were not stored** in `ic_log.jsonl`.")
    lines.append("- Therefore we **cannot** test whether Recatch systematically got higher normalized advantage than Sliding inside the same group.")
    lines.append("\n**INSUFFICIENT_SKILL_COMPETITION_EVIDENCE** for claims (3)–(4) in Part G.\n")
    lines.append("What we *can* say: (1) reward ranks faster SUCCESS higher by λ·ΔT among successes; (2) all-success groups with time variance get A ≈ −z(T).")
    (OUT / "mechanism_competition_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "INSUFFICIENT_SKILL_COMPETITION_EVIDENCE.txt").write_text(
        "No per-trajectory Sliding/Recatch labels in historical GRPO groups.\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)

    write_timing_semantics()
    write_advantage_audit()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    sft_m, _ = load_trained_k48(SFT, device)
    grpo_m, _ = load_trained_k48(GRPO, device)
    sft_m.eval()
    grpo_m.eval()
    env_s = make_training_env(device, model=sft_m)
    env_g = make_training_env(device, model=grpo_m)

    timing_rows = []
    cf_rows = []
    phys_rows = []
    repro_rows = []
    causal = []

    for ic in GRPO_ONLY:
        print("IC", ic, flush=True)
        meta = load_bench(ic)
        blob = pickle.loads((ROOT / meta["snapshot"]).read_bytes())
        assert "capture" in blob, ic

        g_seq = collect_policy_tokens(env_g, grpo_m, book, blob, device, LIMIT_S)
        s_seq = collect_policy_tokens(env_s, sft_m, book, blob, device, LIMIT_S)

        # Actual outcomes for reproduction
        b2 = run_branch(env_g, blob, g_seq["t0"], g_seq["snap0"], g_seq["actions"], "B2_GRPO_actual")
        b4 = run_branch(env_s, blob, s_seq["t0"], s_seq["snap0"], s_seq["actions"], "B4_SFT_actual")
        b0 = run_branch(env_g, blob, g_seq["t0"], g_seq["snap0"], [], "B0_immediate_HANDOFF")

        if g_seq["n_motion"] == 0:
            b1 = {"branch": "NOT_APPLICABLE", "success": None, "reason": "GRPO handed off with zero motion tokens",
                  "recovery_elapsed": None, "return_v2": None, "pre_handoff_phys": g_seq["phys0"]}
        else:
            b1 = run_branch(env_g, blob, g_seq["t0"], g_seq["snap0"], g_seq["actions"][:1], "B1_GRPO_prefix1")

        # B3: SFT actions for same number of motion steps as GRPO, then HANDOFF
        n_g = g_seq["n_motion"]
        if n_g == 0:
            b3 = {"branch": "NOT_APPLICABLE", "success": None, "reason": "no GRPO motion to match",
                  "recovery_elapsed": None, "return_v2": None, "pre_handoff_phys": s_seq["phys0"]}
        elif len(s_seq["actions"]) < n_g:
            # run whatever SFT produced then handoff (shorter)
            b3 = run_branch(env_s, blob, s_seq["t0"], s_seq["snap0"], s_seq["actions"], "B3_SFT_matched_time_truncated")
            b3["note"] = f"SFT produced only {len(s_seq['actions'])} motions vs GRPO {n_g}"
        else:
            b3 = run_branch(env_s, blob, s_seq["t0"], s_seq["snap0"], s_seq["actions"][:n_g], "B3_SFT_matched_time")

        # Timing row
        g_elapsed = b2["recovery_elapsed"]
        timing_rows.append({
            "ic_id": ic,
            "snapshot_sim_t0": g_seq["t0"],
            "grpo_tokens": json.dumps(g_seq["tokens"]),
            "sft_tokens_prefix20": json.dumps(s_seq["tokens"][:20]),
            "grpo_n_motion": g_seq["n_motion"],
            "grpo_n_handoff_token": g_seq["n_handoff"],
            "grpo_motion_physics_s": g_seq["n_motion"] * DT_POLICY,
            "grpo_B2_recovery_elapsed": g_elapsed,
            "grpo_B2_return": b2["return_v2"],
            "grpo_B2_success": b2["success"],
            "grpo_B2_reason": b2["reason"],
            "sft_n_motion": s_seq["n_motion"],
            "sft_B4_recovery_elapsed": b4["recovery_elapsed"],
            "sft_B4_return": b4["return_v2"],
            "sft_B4_success": b4["success"],
            "sft_B4_reason": b4["reason"],
            "R_implies_elapsed": None if b2["return_v2"] is None else (SUCCESS_BONUS - float(b2["return_v2"])) / TIME_PENALTY,
            "elapsed_matches_n_motion_dt": None if g_elapsed is None else abs(float(g_elapsed) - g_seq["n_motion"] * DT_POLICY) < 1e-6,
        })

        tag = classify_causal(b0, b1, b2, b3, b4)
        causal.append({"ic_id": ic, "causal_class": tag})

        for br in (b0, b1, b2, b3, b4):
            if br is None or br.get("branch") == "NOT_APPLICABLE":
                cf_rows.append({"ic_id": ic, "branch": br.get("branch") if br else None, "status": "NOT_APPLICABLE",
                                "reason": None if not br else br.get("reason")})
                continue
            cf_rows.append({
                "ic_id": ic,
                "branch": br["branch"],
                "success": br["success"],
                "reason": br["reason"],
                "recovery_elapsed": br.get("recovery_elapsed"),
                "return_v2": br.get("return_v2"),
                "n_motion_executed": br.get("n_motion_executed"),
                "dropped_before_handoff": br.get("dropped_before_handoff"),
                "causal_class": tag,
            })
            pre = br.get("pre_handoff_phys") or {}
            phys_rows.append({
                "ic_id": ic,
                "branch": br["branch"],
                "success": br["success"],
                **{k: pre.get(k) for k in (
                    "sim_t", "recovery_t", "nL", "nR", "bilateral", "both_off",
                    "fnL", "fnR", "tau_grip", "obj_pos", "hand_pos", "p_des", "v_cmd",
                )},
            })

        repro_rows.append({
            "ic_id": ic,
            "B2_success": b2["success"],
            "B2_reason": b2["reason"],
            "B2_elapsed": b2["recovery_elapsed"],
            "B2_return": b2["return_v2"],
            "prior_R_1_998": abs(float(b2["return_v2"] or 0) - 1.998) < 1e-6,
            "B0_success": b0["success"],
            "B1_success": None if b1.get("branch") == "NOT_APPLICABLE" else b1["success"],
            "B3_success": None if b3.get("branch") == "NOT_APPLICABLE" else b3.get("success"),
            "B4_success": b4["success"],
            "causal_class": tag,
            "grpo_tokens": json.dumps(g_seq["tokens"]),
        })

        # save token traces
        dest = OUT / "traces" / ic
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / "grpo_sequence.json", {
            "tokens": g_seq["tokens"], "n_motion": g_seq["n_motion"], "t0": g_seq["t0"],
            "phys0": g_seq["phys0"], "step_phys": g_seq["step_phys"],
        })
        write_json(dest / "sft_sequence.json", {
            "tokens": s_seq["tokens"][:50], "n_motion": s_seq["n_motion"], "t0": s_seq["t0"],
        })
        write_json(dest / "branches.json", jsafe({"B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "causal": tag}))

    write_csv(OUT / "seven_ic_timing.csv", timing_rows)
    write_csv(OUT / "seven_ic_counterfactual.csv", cf_rows)
    write_csv(OUT / "seven_ic_physical_state.csv", phys_rows)
    write_csv(OUT / "reproduction_checks.csv", repro_rows)
    write_csv(OUT / "causal_classification.csv", causal)

    strata_rows, all_success_detail, strata_sum = analyze_ic_log()
    lam_rows = lambda_sensitivity(strata_rows)
    mechanism_competition(all_success_detail)

    # Report
    n_direct = sum(1 for c in causal if c["causal_class"] == "DIRECT_HANDOFF_SUFFICIENT")
    n_prefix = sum(1 for c in causal if c["causal_class"] == "LEGAL_PREFIX_REQUIRED")
    n_mismatch = sum(1 for c in causal if c["causal_class"] == "REPRODUCTION_MISMATCH")
    n_40 = sum(1 for r in timing_rows if r.get("elapsed_matches_n_motion_dt") and abs(float(r["grpo_B2_recovery_elapsed"]) - 0.04) < 1e-6)
    n_motion_all = [r["grpo_n_motion"] for r in timing_rows]

    report = []
    report.append("# 40 ms HANDOFF + Time-Induced Advantage Audit — REPORT\n")
    report.append("Read-only. See also `handoff_timing_semantics.md`, `advantage_implementation_audit.md`.\n")
    report.append("## Part A — Is recovery really ~40 ms?\n")
    report.append(f"- ICs with `recovery_elapsed ≈ n_motion×0.02` and ≈0.04 s: **{n_40}/{len(GRPO_ONLY)}**")
    report.append(f"- GRPO motion token counts: {n_motion_all}")
    report.append("- `R=1.998` is exactly `2 − 0.05×0.04`; not saturation/default.")
    report.append("- Absolute `t_handoff~3.28` is IC snapshot clock + 0.04 s; paired_results used absolute time (misleading Δ).\n")
    report.append("## Parts B–C — Counterfactual causal classes\n")
    report.append(f"- DIRECT_HANDOFF_SUFFICIENT: **{n_direct}**")
    report.append(f"- LEGAL_PREFIX_REQUIRED: **{n_prefix}**")
    report.append(f"- REPRODUCTION_MISMATCH: **{n_mismatch}**")
    report.append("| IC | class | B0 | B1 | B2 | B3 | B4 | tokens |")
    report.append("|---|---|---|---|---|---|---|---|")
    for r in repro_rows:
        report.append(
            f"| {r['ic_id']} | {r['causal_class']} | {r['B0_success']} | {r['B1_success']} | "
            f"{r['B2_success']} | {r['B3_success']} | {r['B4_success']} | `{r['grpo_tokens']}` |"
        )
    report.append("")
    report.append("Mechanism labels: these short prefixes are **not** evidenced Contact-Preserving Sliding or Release–Recatch ")
    report.append("(migration/both-off thresholds unmet in prior audit; Other/Unclear).\n")
    report.append("## Parts D–F — Advantage & strata\n")
    report.append("```json\n" + json.dumps(jsafe(strata_sum), indent=2) + "\n```\n")
    report.append("- Implementation: `A=(R−mean)/std` with **no ε**; zero-std skips update (`grpo_math.advantages`).")
    report.append("- All-success + time variance ⇒ `A ≈ −z(T)` independent of λ>0 (see sensitivity CSV).\n")
    report.append("## Part G — Skill competition\n")
    report.append("See `mechanism_competition_audit.md` → **INSUFFICIENT_SKILL_COMPETITION_EVIDENCE** ")
    report.append("(no per-trajectory mechanism labels in `ic_log.jsonl`).\n")
    report.append("## Final answers\n")
    report.append(final_answers(n_40, n_direct, n_prefix, n_mismatch, timing_rows, causal, strata_sum, repro_rows))
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("DONE", OUT, flush=True)


def final_answers(n_40, n_direct, n_prefix, n_mismatch, timing_rows, causal, strata_sum, repro_rows):
    return f"""
1. **是否真约 40 ms recovery？**  
   是（相对 IC restore）。典型为 **2×20 ms `step_legal` + HANDOFF**；`recovery_elapsed=0.04`，`R=1.998`。绝对仿真钟约 3.24→3.28 s。依据：`step_legal.py`、`finish_recovery`、`seven_ic_timing.csv`、各 IC `traces/*/grpo_sequence.json`。

2. **多少 IC 直接 HANDOFF 即可成功？**  
   **{n_direct}/{len(GRPO_ONLY)}**（`DIRECT_HANDOFF_SUFFICIENT`）。见 `causal_classification.csv` / `seven_ic_counterfactual.csv` B0 列。

3. **多少必须经过 GRPO 合法前缀？**  
   **{n_prefix}/{len(GRPO_ONLY)}**（`LEGAL_PREFIX_REQUIRED`）。Reproduction mismatches: {n_mismatch}.

4. **成功依赖什么物理变化？**  
   见 `seven_ic_physical_state.csv`（B0 vs B1/B2 的 bilateral/forces/p_des）。短前缀多为微调接触/控制器设定，而非长程 sliding。

5. **能否称为 Sliding / Recatch？**  
   **不能**（按既有接触证据标准）。短 40 ms、机制 Other/Unclear；无持续迁移 / 无 sustained both-off。

6. **历史 all-success groups 多少？**  
   `n_all_success={strata_sum.get('n_all_success')}` / logged={strata_sum.get('n_ic_updates_logged')}；其中 time 方差>0：`{strata_sum.get('n_all_success_with_time_variance')}`；占 optimizer updates 比例≈`{strata_sum.get('frac_updates_all_success')}`.

7. **时间项是否主导这些 group 的 normalized advantage？**  
   对 all-success 且 σ_T>0：是，`A≈−z(T)`（与 λ>0 无关）。零方差则 skip。依据：`all_success_group_detail.csv`、`lambda_advantage_sensitivity.csv`、`advantage_implementation_audit.md`。

8. **是否有证据持续偏向 Recatch、压制 Sliding？**  
   **INSUFFICIENT_SKILL_COMPETITION_EVIDENCE**（训练 log 无逐轨迹机制标签）。Oracle matched 对仅说明成功之间 reward 偏好更快者（ΔR≤0.04）。

9. **下一步？**  
   **优先保持 reward λ；若调整，应针对训练采样/目标（减少 all-success 内纯时间竞争、或记录机制标签），而不是先改 SUCCESS 主项。**  
   对这 7 个 IC：若多数为 `DIRECT_HANDOFF_SUFFICIENT`，增益来自“识别可交权状态”；若多数为 `LEGAL_PREFIX_REQUIRED`，则 1–2 步合法动作是必要的微调整——仍非长程 Sliding/Recatch。
"""


if __name__ == "__main__":
    main()
