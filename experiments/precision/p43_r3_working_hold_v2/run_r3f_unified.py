"""P43b — Unified R3F: R3 readiness + expert-terminal fallback.

Constraints:
  - One rule for A/B/C/D
  - P(SUCCESS_12S | handoff) == 1 (no drop / LOST_GRASP after trigger)
  - B/C SUCCESS_NEW must not degrade vs R2 (16/16, 92/92)
  - A/D may miss SUCCESS_NEW if still SUCCESS_12S
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

from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.paths import OUT  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.r3_rules import (  # noqa: E402
    DwellTracker,
    metric_meta,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

DEST = OUT / "r3f_unified"
RULES = ("R2", "R3", "R3F")  # R3F = R3 + terminal fallback


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


def replay(
    env,
    *,
    restore_fn: Callable,
    ref,
    tokens,
    residuals,
    book,
    rho,
    rule: str,
) -> dict[str, Any]:
    restore_fn()
    # R2 / R3 trackers from r3_rules (R2 = working+bilateral; R3 = +pad+margin)
    base = "R3" if rule in ("R3", "R3F") else "R2"
    tracker = DwellTracker.from_rule(base, dt=float(DT_POLICY))
    toks = np.asarray(tokens, np.int64)
    res = np.asarray(residuals, np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)

    handoff = False
    handoff_step = None
    used_fallback = False
    drop = False
    last: dict[str, Any] = {}

    for t in range(n + 1):
        if latched(env.sim):
            drop = True
            break
        st = tracker.observe(env.sim, t)
        last = st
        if st["ready"]:
            handoff = True
            handoff_step = t
            break
        if t >= n:
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    if not handoff and not drop and rule == "R3F":
        handoff = True
        handoff_step = n
        used_fallback = True
        last = tracker.observe(env.sim, n)

    out: dict[str, Any] = {
        "rule": rule,
        "handoff": int(handoff),
        "handoff_step": handoff_step,
        "used_fallback": int(used_fallback),
        "never_satisfied": int(not handoff and not drop),
        "drop_before_handoff": int(drop and not handoff),
        "hand_theta": last.get("theta") if handoff else None,
        "hand_e_pos": last.get("e_pos") if handoff else None,
        "hand_nL": last.get("nL") if handoff else None,
        "hand_nR": last.get("nR") if handoff else None,
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
        out["eval_reason"] = "NEVER_SATISFIED_RULE"
        out["fail_category"] = "NEVER_SATISFIED_RULE"
        return out

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    reason = hand.get("reason")
    out.update({"SUCCESS_12S": s12, "SUCCESS_NEW": snew, "eval_reason": reason})
    if snew:
        out["fail_category"] = "SUCCESS"
    elif s12 and not snew:
        out["fail_category"] = "HOLD_FAIL_12S_OK"
    else:
        out["fail_category"] = f"POST_HANDOFF:{reason}"
    return out


def aggregate(rows: list[dict]) -> list[dict]:
    agg = []
    for rule in RULES:
        for skill in ("A1", "B1", "C", "D"):
            sub = [r for r in rows if r["rule"] == rule and r["skill"] == skill]
            if not sub:
                continue
            n = len(sub)
            hand = sum(int(r["handoff"]) for r in sub)
            s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
            snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
            lost = sum(1 for r in sub if "LOST_GRASP" in str(r["fail_category"]) or "DROP" in str(r["fail_category"]))
            never = sum(int(r["never_satisfied"]) for r in sub)
            fb = sum(int(r["used_fallback"]) for r in sub)
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
    r2 = {s: by.get(("R2", s)) for s in ("A1", "B1", "C", "D")}
    r3f = {s: by.get(("R3F", s)) for s in ("A1", "B1", "C", "D")}
    checks = {}
    # P(12S|hand)==1 for R3F all skills with handoffs
    p12_ok = True
    for s, a in r3f.items():
        if a is None:
            continue
        ok = a["handoff"] == 0 or abs((a["P_12S_given_hand"] or 0) - 1.0) < 1e-9
        checks[f"R3F_{s}_P12S"] = {"ok": ok, "value": a["P_12S_given_hand"], "hand": a["handoff"]}
        p12_ok = p12_ok and ok
    # B/C NEW not degrade vs R2
    bc_ok = True
    for s in ("B1", "C"):
        if r2[s] and r3f[s]:
            ok = r3f[s]["SUCCESS_NEW"] >= r2[s]["SUCCESS_NEW"]
            checks[f"R3F_{s}_NEW_ge_R2"] = {
                "ok": ok,
                "R2": r2[s]["SUCCESS_NEW"],
                "R3F": r3f[s]["SUCCESS_NEW"],
            }
            bc_ok = bc_ok and ok
    return {"P12S_all": p12_ok, "BC_no_degrade": bc_ok, "pass": p12_ok and bc_ok, "checks": checks}


def write_final(agg: list[dict], verdict: dict) -> None:
    meta = metric_meta()
    lines = [
        "# FINAL_REPORT — Unified R3F Trial",
        "",
        f"Generated: {utc()}",
        "",
        "## Goal",
        "",
        "- One rule for all skills",
        "- Trigger ⇒ SUCCESS_12S always (no drop / LOST_GRASP)",
        "- B/C SUCCESS_NEW ≥ R2 (no degrade)",
        "- A/D may miss NEW if 12S holds",
        "",
        f"R3 = {meta['rules']['R3']['desc']}",
        "R3F = R3, else HANDOFF at expert terminal",
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
    lines += ["", "## Constraint checks (R3F)", ""]
    lines.append(f"- **PASS**: {verdict['pass']}")
    lines.append(f"- P(12S|hand)=1 all skills: {verdict['P12S_all']}")
    lines.append(f"- B/C NEW ≥ R2: {verdict['BC_no_degrade']}")
    lines.append(f"- details: `{json.dumps(verdict['checks'])}`")
    lines += ["", "## Artifacts", "", "- `ROLLOUTS.csv`", "- `AGGREGATE.json`", ""]
    (DEST / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(*, limit: int = 0, skills: tuple[str, ...] | None = None) -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
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
    total = len(jobs) * len(RULES)
    k = 0
    print(f"R3F unified jobs={len(jobs)} rules={RULES} total={total}", flush=True)

    for job in jobs:
        skill = job["skill"]
        if skill in ("A1", "B1", "C"):
            ep = job["ep"]

            def restore(ep=ep):
                restore_start(env, ep)

            ref = ep["ref"]
        else:
            cap, ref = job["capture"], job["ref"]

            def restore(cap=cap):
                restore_capture(env, cap)

        for rule in RULES:
            k += 1
            ev = replay(
                env,
                restore_fn=restore,
                ref=ref,
                tokens=job["tokens"],
                residuals=job["residuals"],
                book=book,
                rho=rho,
                rule=rule,
            )
            row = {
                "traj_id": job["traj_id"],
                "skill": skill,
                "split": job["split"],
                **ev,
            }
            rows.append(row)
            write_csv(DEST / "ROLLOUTS.csv", rows)
            print(
                f"  [{k}/{total}] {skill} {rule} {job['traj_id'][:40]} "
                f"hand={ev['handoff']} fb={ev['used_fallback']} "
                f"12S={ev['SUCCESS_12S']} NEW={ev['SUCCESS_NEW']} cat={ev['fail_category']}",
                flush=True,
            )

    agg = aggregate(rows)
    verdict = constraints_ok(agg)
    write_csv(DEST / "AGGREGATE.csv", agg)
    (DEST / "AGGREGATE.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (DEST / "VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_final(agg, verdict)
    return {"agg": agg, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skills", default="A1,B1,C,D")
    args = ap.parse_args()
    skills = tuple(s.strip() for s in args.skills.split(",") if s.strip())
    run(limit=args.limit, skills=skills)


if __name__ == "__main__":
    main()
