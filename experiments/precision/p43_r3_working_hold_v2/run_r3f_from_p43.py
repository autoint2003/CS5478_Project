"""Build unified R3F results by reusing P43 R2/R3 rows + terminal fallback evals.

R3F(traj) = R3 result if R3 handed off, else HANDOFF at expert terminal.
"""

from __future__ import annotations

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

from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.paths import OUT  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.r3_rules import metric_meta  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402

P43_CSV = OUT / "EXPERT_ROLLOUTS.csv"
DEST = OUT / "r3f_unified"


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


def eval_terminal(env, restore_fn: Callable, ref, tokens, residuals, book, rho) -> dict[str, Any]:
    restore_fn()
    toks = np.asarray(tokens, np.int64)
    res = np.asarray(residuals, np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    for t in range(n):
        if latched(env.sim):
            return {
                "handoff": 0,
                "handoff_step": None,
                "used_fallback": 1,
                "never_satisfied": 0,
                "drop_before_handoff": 1,
                "SUCCESS_12S": 0,
                "SUCCESS_NEW": 0,
                "eval_reason": "DROP_BEFORE_HANDOFF",
                "fail_category": "DROP_BEFORE_HANDOFF",
            }
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    reason = hand.get("reason")
    if snew:
        cat = "SUCCESS"
    elif s12 and not snew:
        cat = "HOLD_FAIL_12S_OK"
    else:
        cat = f"POST_HANDOFF:{reason}"
    return {
        "handoff": 1,
        "handoff_step": n,
        "used_fallback": 1,
        "never_satisfied": 0,
        "drop_before_handoff": 0,
        "SUCCESS_12S": s12,
        "SUCCESS_NEW": snew,
        "eval_reason": reason,
        "fail_category": cat,
        "hand_theta": None,
        "hand_e_pos": None,
        "hand_nL": None,
        "hand_nR": None,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    agg = []
    for rule in ("R2", "R3", "R3F"):
        for skill in ("A1", "B1", "C", "D"):
            sub = [r for r in rows if r["rule"] == rule and r["skill"] == skill]
            if not sub:
                continue
            n = len(sub)
            hand = sum(int(r["handoff"]) for r in sub)
            s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
            snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
            lost = sum(
                1
                for r in sub
                if "LOST_GRASP" in str(r["fail_category"]) or "DROP" in str(r["fail_category"])
            )
            never = sum(int(float(r["never_satisfied"] or 0)) for r in sub)
            fb = sum(int(float(r.get("used_fallback") or 0)) for r in sub)
            p12 = (s12 / hand) if hand else None
            agg.append(
                {
                    "rule": rule,
                    "skill": skill,
                    "n": n,
                    "handoff": hand,
                    "SUCCESS_12S": s12,
                    "SUCCESS_NEW": snew,
                    "P_12S_given_hand": p12,
                    "LOST_or_DROP": lost,
                    "never_satisfied": never,
                    "fallback_used": fb,
                    "fail_cats": dict(Counter(r["fail_category"] for r in sub)),
                }
            )
    return agg


def constraints_ok(agg: list[dict]) -> dict[str, Any]:
    by = {(a["rule"], a["skill"]): a for a in agg}
    checks = {}
    p12_ok = True
    for s in ("A1", "B1", "C", "D"):
        a = by.get(("R3F", s))
        if not a:
            continue
        ok = a["handoff"] == 0 or abs((a["P_12S_given_hand"] or 0) - 1.0) < 1e-9
        checks[f"R3F_{s}_P12S"] = {"ok": ok, "value": a["P_12S_given_hand"], "hand": a["handoff"]}
        p12_ok = p12_ok and ok
    bc_ok = True
    for s in ("B1", "C"):
        r2, r3f = by.get(("R2", s)), by.get(("R3F", s))
        if r2 and r3f:
            ok = r3f["SUCCESS_NEW"] >= r2["SUCCESS_NEW"]
            checks[f"R3F_{s}_NEW_ge_R2"] = {"ok": ok, "R2": r2["SUCCESS_NEW"], "R3F": r3f["SUCCESS_NEW"]}
            bc_ok = bc_ok and ok
    return {"P12S_all": p12_ok, "BC_no_degrade": bc_ok, "pass": p12_ok and bc_ok, "checks": checks}


def write_final(agg: list[dict], verdict: dict) -> None:
    meta = metric_meta()
    lines = [
        "# FINAL_REPORT — Unified R3F (R3 + terminal fallback)",
        "",
        f"Generated: {utc()}",
        "",
        "## Goal",
        "",
        "- Unified rule for A/B/C/D",
        "- Trigger ⇒ SUCCESS_12S always",
        "- B/C SUCCESS_NEW ≥ R2",
        "- A/D may miss NEW if 12S holds",
        "",
        f"R3: {meta['rules']['R3']['desc']}",
        "R3F: R3 if ready during tape; else HANDOFF at expert terminal",
        "",
        "## Aggregate",
        "",
        "| Rule | Skill | NEW | 12S | P(12S\\|hand) | never | fallback | lost/drop |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in agg:
        p = a["P_12S_given_hand"]
        ps = "—" if p is None else f"{p:.3f}"
        lines.append(
            f"| {a['rule']} | {a['skill']} | {a['SUCCESS_NEW']}/{a['n']} | "
            f"{a['SUCCESS_12S']}/{a['n']} | {ps} | {a['never_satisfied']} | "
            f"{a['fallback_used']} | {a['LOST_or_DROP']} |"
        )
    lines += [
        "",
        "## Constraint checks (R3F)",
        "",
        f"- **PASS**: `{verdict['pass']}`",
        f"- P(12S|hand)=1: `{verdict['P12S_all']}`",
        f"- B/C NEW ≥ R2: `{verdict['BC_no_degrade']}`",
        f"- details: `{json.dumps(verdict['checks'])}`",
        "",
    ]
    (DEST / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    p43 = list(csv.DictReader(P43_CSV.open(encoding="utf-8")))
    by = {}
    for r in p43:
        by.setdefault((r["skill"], r["traj_id"]), {})[r["rule"]] = r

    jobs = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    rows: list[dict] = []
    # Keep R2/R3 as-is
    for r in p43:
        rows.append({**r, "used_fallback": 0})

    need_fb = []
    for (skill, tid), m in by.items():
        r3 = m.get("R3")
        if r3 is None:
            continue
        if int(r3["handoff"]) == 1:
            row = {**r3, "rule": "R3F", "used_fallback": 0}
            rows.append(row)
        else:
            need_fb.append((skill, tid, r3))

    print(f"R3F reuse R3 handoffs; terminal fallback needed: {len(need_fb)}", flush=True)
    for i, (skill, tid, r3) in enumerate(need_fb):
        job = jobs.get((skill, tid))
        if job is None:
            # fuzzy
            cands = [j for (sk, t), j in jobs.items() if sk == skill and (t == tid or tid in t or t in tid)]
            job = cands[0] if cands else None
        if job is None:
            print(f"  SKIP missing job {skill} {tid}", flush=True)
            rows.append(
                {
                    **r3,
                    "rule": "R3F",
                    "used_fallback": 1,
                    "handoff": 0,
                    "never_satisfied": 1,
                    "SUCCESS_12S": 0,
                    "SUCCESS_NEW": 0,
                    "fail_category": "JOB_NOT_FOUND",
                    "eval_reason": "JOB_NOT_FOUND",
                }
            )
            continue

        if skill in ("A1", "B1", "C"):
            ep = job["ep"]

            def restore(ep=ep):
                restore_start(env, ep)

            ref = ep["ref"]
        else:
            cap, ref = job["capture"], job["ref"]

            def restore(cap=cap):
                restore_capture(env, cap)

        ev = eval_terminal(env, restore, ref, job["tokens"], job["residuals"], book, rho)
        rows.append(
            {
                "traj_id": tid,
                "skill": skill,
                "family": r3.get("family"),
                "split": r3.get("split"),
                "source": r3.get("source"),
                "lineage_id": r3.get("lineage_id"),
                "rule": "R3F",
                **ev,
            }
        )
        write_csv(DEST / "ROLLOUTS.csv", rows)
        print(
            f"  [{i+1}/{len(need_fb)}] {skill} {tid[:48]} "
            f"12S={ev['SUCCESS_12S']} NEW={ev['SUCCESS_NEW']} cat={ev['fail_category']}",
            flush=True,
        )

    write_csv(DEST / "ROLLOUTS.csv", rows)
    agg = aggregate(rows)
    verdict = constraints_ok(agg)
    write_csv(DEST / "AGGREGATE.csv", agg)
    (DEST / "AGGREGATE.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (DEST / "VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_final(agg, verdict)


if __name__ == "__main__":
    main()
