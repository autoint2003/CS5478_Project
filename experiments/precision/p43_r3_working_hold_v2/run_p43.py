"""P43 — Expert validation of R3 WORKING_HOLD_V2 vs R2.

Diagnosis / rule trial only: does not change physics, datasets, or training.
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
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.paths import OUT  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.r3_rules import (  # noqa: E402
    RULE_IDS,
    DwellTracker,
    metric_meta,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402


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


def replay_expert_rule(
    env,
    *,
    restore_fn: Callable,
    ref,
    tokens: np.ndarray,
    residuals: np.ndarray,
    book,
    rho,
    rule_id: str,
) -> dict[str, Any]:
    restore_fn()
    tracker = DwellTracker.from_rule(rule_id, dt=float(DT_POLICY))
    init_st = tracker.observe(env.sim, 0)
    tracker = DwellTracker.from_rule(rule_id, dt=float(DT_POLICY))

    toks = np.asarray(tokens, np.int64)
    res = np.asarray(residuals, np.float32)
    n = len(toks)
    if n and int(toks[-1]) == HANDOFF:
        n -= 1

    handoff = False
    handoff_step = None
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

    out: dict[str, Any] = {
        "rule": rule_id,
        "init_working": int(init_st.get("working_ok", False)),
        "init_bilateral": int(init_st.get("bilateral", False)),
        "init_theta": init_st.get("theta"),
        "init_e_pos": init_st.get("e_pos"),
        "n_motion_available": n,
        "handoff": int(handoff),
        "handoff_step": handoff_step,
        "dwell_ok_s_at_end": last.get("ok_s"),
        "n_dwell_resets": tracker.n_resets,
        "ever_partial_ok": int(tracker.ever_partial),
        "drop_before_handoff": int(drop and not handoff),
        "never_satisfied": int(not handoff and not drop),
        "hand_theta": last.get("theta") if handoff else None,
        "hand_e_pos": last.get("e_pos") if handoff else None,
        "hand_bilateral": int(last.get("bilateral", False)) if handoff else None,
        "hand_nL": last.get("nL") if handoff else None,
        "hand_nR": last.get("nR") if handoff else None,
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
        "eval_reason": None,
        "fail_category": None,
        "win_mean_pos": None,
        "win_mean_theta": None,
        "win_working_hold": None,
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
    win = hand.get("window_SUCCESS_NEW") or {}
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    reason = hand.get("reason")
    out.update(
        {
            "SUCCESS_12S": s12,
            "SUCCESS_NEW": snew,
            "eval_reason": reason,
            "win_mean_pos": win.get("mean_pos"),
            "win_mean_theta": win.get("mean_theta"),
            "win_working_hold": int(bool(win.get("working_SUCCESS_NEW_hold"))),
        }
    )
    if snew:
        out["fail_category"] = "SUCCESS"
    elif s12 and not snew:
        out["fail_category"] = "HOLD_FAIL_12S_OK"
    else:
        out["fail_category"] = f"POST_HANDOFF:{reason}"
    return out


def aggregate(rows: list[dict]) -> list[dict]:
    agg = []
    for rule in RULE_IDS:
        for skill in ("A1", "B1", "C", "D"):
            sub = [r for r in rows if r["rule"] == rule and r["skill"] == skill]
            if not sub:
                continue
            n = len(sub)
            n_hand = sum(int(r["handoff"]) for r in sub)
            s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
            snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
            drop = sum(int(r["drop_before_handoff"]) for r in sub)
            never = sum(int(r["never_satisfied"]) for r in sub)
            steps = [int(r["handoff_step"]) for r in sub if r["handoff_step"] is not None]
            conv = (snew / n_hand) if n_hand else None
            agg.append(
                {
                    "rule": rule,
                    "skill": skill,
                    "n_traj": n,
                    "handoff": n_hand,
                    "handoff_rate": n_hand / n if n else None,
                    "mean_handoff_step": float(np.mean(steps)) if steps else None,
                    "median_handoff_step": float(np.median(steps)) if steps else None,
                    "SUCCESS_12S": s12,
                    "SUCCESS_NEW": snew,
                    "NEW_over_all": snew / n if n else None,
                    "NEW_given_handoff": f"{snew}/{n_hand}",
                    "conversion_rate": conv,
                    "drop_before_handoff": drop,
                    "never_satisfied": never,
                    "fail_cats": dict(Counter(r["fail_category"] for r in sub)),
                }
            )
    return agg


def paired_delta(rows: list[dict]) -> list[dict]:
    """Per-traj R2 vs R3 outcome delta."""
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        key = (r["skill"], r["traj_id"])
        by.setdefault(key, {})[r["rule"]] = r
    out = []
    for (skill, tid), m in sorted(by.items()):
        r2, r3 = m.get("R2"), m.get("R3")
        if not r2 or not r3:
            continue
        out.append(
            {
                "skill": skill,
                "traj_id": tid,
                "r2_handoff": r2["handoff"],
                "r3_handoff": r3["handoff"],
                "r2_NEW": r2["SUCCESS_NEW"],
                "r3_NEW": r3["SUCCESS_NEW"],
                "r2_12S": r2["SUCCESS_12S"],
                "r3_12S": r3["SUCCESS_12S"],
                "r2_step": r2["handoff_step"],
                "r3_step": r3["handoff_step"],
                "r2_fail": r2["fail_category"],
                "r3_fail": r3["fail_category"],
                "delta_NEW": int(r3["SUCCESS_NEW"]) - int(r2["SUCCESS_NEW"]),
                "r2_ok_r3_fail": int(int(r2["SUCCESS_NEW"]) == 1 and int(r3["SUCCESS_NEW"]) == 0),
                "r2_fail_r3_ok": int(int(r2["SUCCESS_NEW"]) == 0 and int(r3["SUCCESS_NEW"]) == 1),
                "both_ok": int(int(r2["SUCCESS_NEW"]) == 1 and int(r3["SUCCESS_NEW"]) == 1),
                "both_fail": int(int(r2["SUCCESS_NEW"]) == 0 and int(r3["SUCCESS_NEW"]) == 0),
            }
        )
    return out


def write_final(agg: list[dict], paired: list[dict], meta: dict) -> None:
    lines = [
        "# FINAL_REPORT — P43 R3 WORKING_HOLD_V2 Trial",
        "",
        f"Generated: {utc()}",
        "",
        "## Rule",
        "",
        f"- R2: {meta['rules']['R2']['desc']}",
        f"- R3: {meta['rules']['R3']['desc']}",
        f"- R3 constants: min_pad={meta['R3_MIN_PAD_CONTACTS']}, "
        f"θ≤{meta['R3_MARGIN_THETA_DEG']}°, e_pos≤{meta['R3_MARGIN_POS_MM']} mm, dwell={meta['HOLD_S']}s",
        "",
        "## Aggregate SUCCESS_NEW",
        "",
        "| Skill | R2 NEW/all | R2 NEW|hand | R3 NEW/all | R3 NEW|hand | R3 never | R3 handoff_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by = {(a["rule"], a["skill"]): a for a in agg}
    for skill in ("A1", "B1", "C", "D"):
        a2, a3 = by.get(("R2", skill)), by.get(("R3", skill))
        if not a2 or not a3:
            continue
        lines.append(
            f"| {skill} | {a2['SUCCESS_NEW']}/{a2['n_traj']} | {a2['NEW_given_handoff']} | "
            f"{a3['SUCCESS_NEW']}/{a3['n_traj']} | {a3['NEW_given_handoff']} | "
            f"{a3['never_satisfied']} | {a3['handoff_rate']:.3f} |"
        )
    lines += ["", "## Paired flips (same IC)", ""]
    lines.append("| Skill | both_ok | R2→R3 gain | R2→R3 loss | both_fail |")
    lines.append("|---|---:|---:|---:|---:|")
    for skill in ("A1", "B1", "C", "D"):
        sub = [p for p in paired if p["skill"] == skill]
        if not sub:
            continue
        lines.append(
            f"| {skill} | {sum(p['both_ok'] for p in sub)} | "
            f"{sum(p['r2_fail_r3_ok'] for p in sub)} | "
            f"{sum(p['r2_ok_r3_fail'] for p in sub)} | "
            f"{sum(p['both_fail'] for p in sub)} |"
        )

    # Verdict
    a1_2 = by.get(("R2", "A1"))
    a1_3 = by.get(("R3", "A1"))
    d_2 = by.get(("R2", "D"))
    d_3 = by.get(("R3", "D"))
    lines += ["", "## Verdict", ""]
    if a1_2 and a1_3 and d_2 and d_3:
        a_gain = a1_3["SUCCESS_NEW"] - a1_2["SUCCESS_NEW"]
        d_gain = d_3["SUCCESS_NEW"] - d_2["SUCCESS_NEW"]
        lines.append(
            f"- A1: R2 {a1_2['SUCCESS_NEW']}/{a1_2['n_traj']} → R3 {a1_3['SUCCESS_NEW']}/{a1_3['n_traj']} "
            f"(Δ={a_gain:+d}); never_satisfied R3={a1_3['never_satisfied']}."
        )
        lines.append(
            f"- D: R2 {d_2['SUCCESS_NEW']}/{d_2['n_traj']} → R3 {d_3['SUCCESS_NEW']}/{d_3['n_traj']} "
            f"(Δ={d_gain:+d}); never_satisfied R3={d_3['never_satisfied']}."
        )
        if a_gain > 0 and d_gain >= 0:
            lines.append("- **R3 helps** on the target skills without collapsing D.")
        elif a_gain > 0 and d_gain < 0:
            lines.append("- **R3 helps A1 but hurts D** — consider skill-split gates.")
        elif a_gain <= 0 and d_gain <= 0:
            lines.append("- **R3 does not improve** NEW/all on A1/D under these constants.")
        else:
            lines.append("- Mixed: inspect paired flips and fail_cats.")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `EXPERT_ROLLOUTS.csv`",
        "- `AGGREGATE.csv`",
        "- `PAIRED_R2_VS_R3.csv`",
        "- `RULE_METRIC.json`",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(*, limit: int = 0, skills: tuple[str, ...] | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = metric_meta()
    (OUT / "RULE_METRIC.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    jobs = collect_jobs()
    if skills:
        jobs = [j for j in jobs if j["skill"] in skills]
    if limit > 0:
        # per-skill limit
        kept = []
        counts: dict[str, int] = {}
        for j in jobs:
            c = counts.get(j["skill"], 0)
            if c < limit:
                kept.append(j)
                counts[j["skill"]] = c + 1
        jobs = kept

    rows: list[dict] = []
    total = len(jobs) * len(RULE_IDS)
    k = 0
    print(f"P43 jobs={len(jobs)} rules={RULE_IDS} total={total}", flush=True)
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

        for rule_id in RULE_IDS:
            k += 1
            ev = replay_expert_rule(
                env,
                restore_fn=restore,
                ref=ref,
                tokens=job["tokens"],
                residuals=job["residuals"],
                book=book,
                rho=rho,
                rule_id=rule_id,
            )
            row = {
                "traj_id": job["traj_id"],
                "skill": skill,
                "family": job["family"],
                "split": job["split"],
                "source": job["source"],
                "lineage_id": job.get("lineage_id"),
                **ev,
            }
            rows.append(row)
            write_csv(OUT / "EXPERT_ROLLOUTS.csv", rows)
            print(
                f"  [{k}/{total}] {skill} {rule_id} {job['traj_id'][:48]} "
                f"hand={ev['handoff']} NEW={ev['SUCCESS_NEW']} cat={ev['fail_category']} "
                f"step={ev['handoff_step']}",
                flush=True,
            )

    agg = aggregate(rows)
    paired = paired_delta(rows)
    write_csv(OUT / "AGGREGATE.csv", agg)
    write_csv(OUT / "PAIRED_R2_VS_R3.csv", paired)
    (OUT / "AGGREGATE.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
    write_final(agg, paired, meta)
    return {"rows": rows, "agg": agg, "paired": paired}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Per-skill traj limit (0=all)")
    ap.add_argument("--skills", default="A1,B1,C,D", help="Comma-separated skills")
    args = ap.parse_args()
    skills = tuple(s.strip() for s in args.skills.split(",") if s.strip())
    run(limit=args.limit, skills=skills)


if __name__ == "__main__":
    main()
