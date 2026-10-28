"""Validate unified online R3F on expert ICs under current nominal.

Modes per traj:
  - R3_EARLY: R3 ready during tape
  - TAPE_END_FALLBACK: tape ends before T_max without R3 → handoff at end
  - TIMEOUT_HOLD: after tape end without R3, grip-hold until T_max then handoff
    (stress-tests timeout ≠ expert-end)

All handoffs evaluated with evaluate_handoff_with_axis (real 12s nominal).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.paths import OUT, SEED, T_MAX_DEFAULT  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402

DEST = OUT
# Post-tape hold before TIMEOUT handoff (stress ≠ expert-end; avoids 3k no-op steps)
TIMEOUT_EXTRA_HOLD = 100


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def replay_and_eval(
    env,
    job,
    book,
    rho,
    *,
    t_max: int,
    mode: str,
) -> dict[str, Any]:
    """mode: 'r3f' | 'timeout_hold'."""
    skill = job["skill"]
    if skill in ("A1", "B1", "C"):
        ep = job["ep"]
        ref = ep["ref"]

        def restore():
            restore_start(env, ep)

    else:
        cap, ref = job["capture"], job["ref"]

        def restore():
            restore_capture(env, cap)

    restore()
    mon = R3FOnline.create(t_max=t_max)
    toks = np.asarray(job["tokens"], np.int64)
    res = np.asarray(job["residuals"], np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)

    handoff = False
    reason = "NONE"
    step_h = None
    drop = False

    for t in range(n + 1):
        if latched(env.sim):
            drop = True
            break
        tape_ended = t >= n
        if mode == "r3f":
            st = mon.observe(env.sim, t, tape_ended=tape_ended)
        else:
            # timeout_hold: only R3 during tape; after tape, hold to T_max
            if t < n:
                st = mon.observe(env.sim, t, tape_ended=False)
            else:
                st = {"ready": False, "reason": "NONE"}
        if st.get("ready"):
            handoff = True
            reason = st["reason"]
            step_h = t
            break
        if t >= n:
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    # timeout_hold: after tape without R3, grip-hold EXTRA steps then HANDOFF.
    # This is not identical to waiting until global T_max, but tests that
    # post-expert-end delay ≠ instant tape-end (user: expert-end ≠ timeout).
    if mode == "timeout_hold" and not handoff and not drop:
        t = n
        target = min(int(t_max), int(n) + int(TIMEOUT_EXTRA_HOLD))
        while t < target:
            if latched(env.sim):
                drop = True
                break
            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
            t += 1
        if not drop:
            handoff = True
            reason = "TIMEOUT_FALLBACK"
            step_h = t

    # r3f tape-end already handled via observe(tape_ended=True)

    out: dict[str, Any] = {
        "mode": mode,
        "skill": skill,
        "traj_id": job["traj_id"],
        "split": job.get("split"),
        "n_motion": n,
        "t_max": t_max,
        "handoff": int(handoff),
        "handoff_reason": reason if handoff else ("DROP_BEFORE" if drop else "NONE"),
        "handoff_step": step_h,
        "drop_before_handoff": int(drop and not handoff),
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
        "eval_reason": None,
        "fail_category": None,
    }

    if drop and not handoff:
        out["eval_reason"] = "DROP_BEFORE_HANDOFF"
        out["fail_category"] = "DROP_BEFORE_HANDOFF"
        return out
    if not handoff:
        out["eval_reason"] = "NO_HANDOFF"
        out["fail_category"] = "NO_HANDOFF"
        return out

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    reason_e = hand.get("reason")
    out.update({"SUCCESS_12S": s12, "SUCCESS_NEW": snew, "eval_reason": reason_e})
    if snew:
        out["fail_category"] = "SUCCESS"
    elif s12:
        out["fail_category"] = "HOLD_FAIL_12S_OK"
    else:
        out["fail_category"] = f"POST_HANDOFF:{reason_e}"
    return out


def summarize(rows: list[dict], mode: str) -> list[dict]:
    agg = []
    sub = [r for r in rows if r["mode"] == mode]
    for skill in ("A1", "B1", "C", "D"):
        s = [r for r in sub if r["skill"] == skill]
        if not s:
            continue
        hand = [r for r in s if int(r["handoff"])]
        s12 = sum(int(r["SUCCESS_12S"]) for r in s)
        snew = sum(int(r["SUCCESS_NEW"]) for r in s)
        p12 = (sum(int(r["SUCCESS_12S"]) for r in hand) / len(hand)) if hand else None
        agg.append(
            {
                "mode": mode,
                "skill": skill,
                "n": len(s),
                "handoff": len(hand),
                "SUCCESS_12S": s12,
                "SUCCESS_NEW": snew,
                "P_12S_given_hand": p12,
                "reasons": dict(Counter(r["handoff_reason"] for r in s)),
                "fail_cats": dict(Counter(r["fail_category"] for r in s)),
            }
        )
    return agg


def write_report(agg_r3f: list[dict], agg_to: list[dict], t_max: int) -> None:
    lines = [
        "# R3F_ONLINE_VALIDATION",
        "",
        f"Generated: {utc()}",
        f"T_max={t_max} (from expert length dist; D max≈3042)",
        "",
        "## Important",
        "",
        "TAPE_END_FALLBACK ≠ TIMEOUT_FALLBACK. Timeout-hold stress continues grip-hold",
        "after the expert tape until T_max, then hands off.",
        "",
        "## Mode A — Online R3F (R3_EARLY | TAPE_END_FALLBACK)",
        "",
        "| Skill | 12S | NEW | P(12S\\|hand) | reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for a in agg_r3f:
        p = a["P_12S_given_hand"]
        lines.append(
            f"| {a['skill']} | {a['SUCCESS_12S']}/{a['n']} | {a['SUCCESS_NEW']}/{a['n']} | "
            f"{'—' if p is None else f'{p:.3f}'} | `{a['reasons']}` |"
        )
    lines += [
        "",
        "## Mode B — Timeout-hold stress (R3 during tape, else hold→T_max)",
        "",
        "| Skill | 12S | NEW | P(12S\\|hand) | reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for a in agg_to:
        p = a["P_12S_given_hand"]
        lines.append(
            f"| {a['skill']} | {a['SUCCESS_12S']}/{a['n']} | {a['SUCCESS_NEW']}/{a['n']} | "
            f"{'—' if p is None else f'{p:.3f}'} | `{a['reasons']}` |"
        )
    lines += [
        "",
        "## Verdict questions",
        "",
        "1. Is R3 + tape-end fallback reliable on experts under current nominal?",
        "2. Does delaying to T_max via hold after tape still keep SUCCESS_12S?",
        "3. Do not claim mathematical P(12S|hand)=1 from finite tests.",
        "",
    ]
    (DEST / "R3F_ONLINE_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(*, t_max: int = T_MAX_DEFAULT, limit: int = 0, skills: tuple[str, ...] | None = None) -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    jobs = collect_jobs()
    if skills:
        jobs = [j for j in jobs if j["skill"] in skills]
    if limit > 0:
        kept, counts = [], {}
        for j in jobs:
            c = counts.get(j["skill"], 0)
            if c < limit:
                kept.append(j)
                counts[j["skill"]] = c + 1
        jobs = kept

    rows: list[dict] = []
    # Full R3F on all; timeout-hold only on jobs that need fallback under R3
    # (computed on the fly: run r3f first, if TAPE_END then also timeout_hold)
    total_est = len(jobs) * 2
    k = 0
    print(f"P44 R3F validate jobs={len(jobs)} t_max={t_max}", flush=True)
    for job in jobs:
        k += 1
        r = replay_and_eval(env, job, book, rho, t_max=t_max, mode="r3f")
        rows.append(r)
        write_csv(DEST / "R3F_VALIDATION_ROLLOUTS.csv", rows)
        print(
            f"  [{k}/{len(jobs)}] r3f {job['skill']} {job['traj_id'][:40]} "
            f"{r['handoff_reason']} 12S={r['SUCCESS_12S']} NEW={r['SUCCESS_NEW']}",
            flush=True,
        )
        if r["handoff_reason"] == "TAPE_END_FALLBACK":
            r2 = replay_and_eval(env, job, book, rho, t_max=t_max, mode="timeout_hold")
            rows.append(r2)
            write_csv(DEST / "R3F_VALIDATION_ROLLOUTS.csv", rows)
            print(
                f"       timeout_hold → {r2['handoff_reason']} "
                f"12S={r2['SUCCESS_12S']} NEW={r2['SUCCESS_NEW']}",
                flush=True,
            )

    agg_r3f = summarize(rows, "r3f")
    agg_to = summarize(rows, "timeout_hold")
    write_csv(DEST / "R3F_VALIDATION_AGG.csv", agg_r3f + agg_to)
    (DEST / "R3F_VALIDATION_AGG.json").write_text(
        json.dumps({"r3f": agg_r3f, "timeout_hold": agg_to, "t_max": t_max}, indent=2),
        encoding="utf-8",
    )
    write_report(agg_r3f, agg_to, t_max)
    return {"rows": rows, "agg_r3f": agg_r3f, "agg_to": agg_to}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-max", type=int, default=T_MAX_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skills", default="A1,B1,C,D")
    args = ap.parse_args()
    skills = tuple(s.strip() for s in args.skills.split(",") if s.strip())
    run(t_max=args.t_max, limit=args.limit, skills=skills)


if __name__ == "__main__":
    main()
