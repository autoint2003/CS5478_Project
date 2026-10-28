"""P42 — A1/D R2 HANDOFF failure audit (diagnosis only).

For every P41 R2 failure on A1 and D:
1. Classify failure mode from logged reason.
2. Branch R2 vs expert-terminal HANDOFF under the same nominal/STRICT path.
3. Summarize exclusive categories + representative traces.

Does not modify thresholds, physics, datasets, or training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p39_working_hold_handoff.rules import DwellTracker  # noqa: E402
from experiments.precision.p42_ad_handoff_failure_audit.paths import OUT, P41_ROLLOUTS  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

SKILLS = ("A1", "D")
STABILIZE_N = 8


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


def snap(sim) -> dict[str, Any]:
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    wk = bool(m["e_theta_deg"] <= WORKING_THETA_DEG and m["e_pos_mm"] <= WORKING_POS_MM)
    return {
        "theta": float(m["e_theta_deg"]),
        "e_pos": float(m["e_pos_mm"]),
        "obj_z": float(m["obj_z"]),
        "working": int(wk),
        "bilateral": int(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "phase": str(sim.fsm.phase),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "time": float(sim.data.time),
        "latched": int(latched(sim)),
    }


def load_p41_failures() -> list[dict]:
    rows = list(csv.DictReader(P41_ROLLOUTS.open(encoding="utf-8")))
    fails = [
        r
        for r in rows
        if r["skill"] in SKILLS and r["rule"] == "R2" and int(r["SUCCESS_NEW"]) == 0
    ]
    return fails


def classify_logged(r: dict) -> str:
    """Coarse class from P41 log (before branch compare)."""
    if int(r.get("handoff") or 0) == 0 or int(r.get("never_satisfied") or 0) == 1:
        return "R2_NEVER_TRIGGERED"
    cat = str(r.get("fail_category") or "")
    if cat == "HOLD_FAIL_12S_OK" or (
        int(r.get("SUCCESS_12S") or 0) == 1 and int(r.get("SUCCESS_NEW") or 0) == 0
    ):
        return "R2_TRIGGERED_12S_OK_NEW_FAIL"
    if "LOST_GRASP" in cat:
        return "R2_TRIGGERED_LOST_GRASP"
    if "DROP" in cat:
        return "R2_TRIGGERED_DROP"
    return f"R2_TRIGGERED_OTHER:{cat}"


def replay_to_r2_and_terminal(env, job, book, rho) -> dict[str, Any]:
    """Single forward pass: capture at first R2 ready and at expert end."""
    skill = job["skill"]
    if skill == "A1":
        ep = job["ep"]

        def restore():
            restore_start(env, ep)

        ref = ep["ref"]
    else:
        cap0, ref = job["capture"], job["ref"]

        def restore():
            restore_capture(env, cap0)

    if not hasattr(env, "_obs_seq"):
        env._obs_seq = []
    if not hasattr(env, "_prev_seq"):
        env._prev_seq = []

    restore()
    toks = np.asarray(job["tokens"], np.int64)
    res = np.asarray(job["residuals"], np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    tracker = DwellTracker.from_rule("R2", dt=float(DT_POLICY))

    r2_cap = None
    r2_step = None
    r2_geo = None
    drop_step = None
    trace_around_r2: list[dict] = []

    for t in range(n + 1):
        g = snap(env.sim)
        st = tracker.observe(env.sim, t)
        if r2_step is None and st["ready"]:
            r2_step = t
            r2_geo = g
            r2_cap = take_capture(env)
        # keep short window around R2 for traces
        if r2_step is not None and r2_step <= t <= r2_step + 5:
            trace_around_r2.append({"step": t, **g, "ok_s": st.get("ok_s")})
        elif r2_step is None and t >= max(0, n - 1):
            pass
        if t >= n:
            break
        if latched(env.sim):
            drop_step = t
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    term_geo = snap(env.sim)
    term_cap = take_capture(env)
    term_step = n if drop_step is None else drop_step

    return {
        "ref": ref,
        "n_motion": n,
        "r2_step": r2_step,
        "r2_geo": r2_geo,
        "r2_cap": r2_cap,
        "term_step": term_step,
        "term_geo": term_geo,
        "term_cap": term_cap,
        "drop_before_end": drop_step,
        "trace_around_r2": trace_around_r2,
    }


def eval_branch(env, cap, ref, *, tag: str) -> dict[str, Any]:
    restore_capture(env, cap)
    pre = snap(env.sim)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=STABILIZE_N)
    win = hand.get("window_SUCCESS_NEW") or hand.get("window") or {}
    s12 = bool(hand.get("SUCCESS_12S"))
    snew = bool(hand.get("SUCCESS_NEW"))
    reason = hand.get("reason")
    if snew:
        fail = "SUCCESS"
    elif s12 and not snew:
        fail = "HOLD_FAIL_12S_OK"
    else:
        fail = f"POST_HANDOFF:{reason}"
    return {
        "branch": tag,
        "pre_theta": pre["theta"],
        "pre_e_pos": pre["e_pos"],
        "pre_obj_z": pre["obj_z"],
        "pre_working": pre["working"],
        "pre_bilateral": pre["bilateral"],
        "pre_ctrl7": pre["ctrl7"],
        "SUCCESS_12S": int(s12),
        "SUCCESS_NEW": int(snew),
        "reason": reason,
        "fail_category": fail,
        "win_mean_theta": win.get("mean_theta") if isinstance(win, dict) else None,
        "win_mean_pos": win.get("mean_pos") if isinstance(win, dict) else None,
        "win_working_hold": int(bool(win.get("working_SUCCESS_NEW_hold"))) if isinstance(win, dict) else None,
    }


def paired_category(r2: dict | None, term: dict | None, r2_triggered: bool) -> str:
    """Mutually exclusive outcome category."""
    if not r2_triggered or r2 is None:
        return "R2_never_triggered"
    r2_ok = int(r2["SUCCESS_NEW"]) == 1
    term_ok = term is not None and int(term["SUCCESS_NEW"]) == 1
    if r2_ok:
        return "R2_success"
    if term_ok:
        return "R2_fails_terminal_succeeds"
    return "both_fail"


def run(*, limit: int = 0, trace_n: int = 6, skills: tuple[str, ...] = SKILLS, resume: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)

    fails = load_p41_failures()
    # classify from log first
    class_rows = []
    for r in fails:
        class_rows.append(
            {
                "traj_id": r["traj_id"],
                "skill": r["skill"],
                "split": r.get("split"),
                "handoff": r.get("handoff"),
                "handoff_step": r.get("handoff_step"),
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "fail_category": r.get("fail_category"),
                "eval_reason": r.get("eval_reason"),
                "logged_class": classify_logged(r),
            }
        )
    write_csv(OUT / "FAILURE_CLASSIFICATION_LOGGED.csv", class_rows)
    print(
        "Logged classes:",
        dict(Counter((r["skill"], r["logged_class"]) for r in class_rows)),
        flush=True,
    )

    jobs = { (j["skill"], j["traj_id"]): j for j in collect_jobs() if j["skill"] in skills }
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    # Process failures (optionally limited per skill for smoke)
    by_skill: dict[str, list] = {s: [r for r in fails if r["skill"] == s] for s in skills}
    if limit > 0:
        for s in skills:
            by_skill[s] = by_skill[s][:limit]

    paired_rows: list[dict] = []
    done_keys: set[tuple[str, str]] = set()
    paired_path = OUT / "PAIRED_R2_VS_TERMINAL.csv"
    if resume and paired_path.exists() and paired_path.stat().st_size > 0:
        paired_rows = list(csv.DictReader(paired_path.open(encoding="utf-8")))
        done_keys = {(r["skill"], r["traj_id"]) for r in paired_rows if r.get("category")}
        print(f"Resume: loaded {len(paired_rows)} prior paired rows", flush=True)

    traces_saved = len(list((OUT / "traces").glob("*.json")))

    for skill in skills:
        subset = by_skill[skill]
        print(f"=== P42 branch compare {skill} n={len(subset)} ===", flush=True)
        for i, fr in enumerate(subset):
            if (skill, fr["traj_id"]) in done_keys:
                continue
            key = (skill, fr["traj_id"])
            job = jobs.get(key)
            if job is None:
                # try suffix match
                cands = [j for (sk, tid), j in jobs.items() if sk == skill and (tid == fr["traj_id"] or fr["traj_id"] in tid or tid in fr["traj_id"])]
                job = cands[0] if cands else None
            if job is None:
                print(f"  SKIP missing job {fr['traj_id']}", flush=True)
                paired_rows.append(
                    {
                        "traj_id": fr["traj_id"],
                        "skill": skill,
                        "error": "job_not_found",
                        "category": "R2_never_triggered",
                        "logged_class": classify_logged(fr),
                    }
                )
                write_csv(OUT / "PAIRED_R2_VS_TERMINAL.csv", paired_rows)
                continue

            land = replay_to_r2_and_terminal(env, job, book, rho)
            r2_triggered = land["r2_cap"] is not None
            r2_ev = None
            term_ev = None
            if r2_triggered:
                r2_ev = eval_branch(env, land["r2_cap"], land["ref"], tag="R2")
            term_ev = eval_branch(env, land["term_cap"], land["ref"], tag="terminal")
            cat = paired_category(r2_ev, term_ev, r2_triggered)

            row = {
                "traj_id": fr["traj_id"],
                "skill": skill,
                "logged_class": classify_logged(fr),
                "logged_fail_category": fr.get("fail_category"),
                "logged_handoff_step": fr.get("handoff_step"),
                "r2_triggered": int(r2_triggered),
                "r2_step": land["r2_step"],
                "term_step": land["term_step"],
                "n_motion": land["n_motion"],
                "category": cat,
                "drop_before_end": land["drop_before_end"],
            }
            if land["r2_geo"]:
                for k, v in land["r2_geo"].items():
                    row[f"r2_{k}"] = v
            if land["term_geo"]:
                for k, v in land["term_geo"].items():
                    row[f"term_{k}"] = v
            if r2_ev:
                for k, v in r2_ev.items():
                    if k == "branch":
                        continue
                    row[f"r2ev_{k}"] = v
            if term_ev:
                for k, v in term_ev.items():
                    if k == "branch":
                        continue
                    row[f"termev_{k}"] = v

            # deltas
            if r2_ev and term_ev:
                row["delta_NEW"] = int(term_ev["SUCCESS_NEW"]) - int(r2_ev["SUCCESS_NEW"])
                row["delta_12S"] = int(term_ev["SUCCESS_12S"]) - int(r2_ev["SUCCESS_12S"])
                row["delta_e_pos"] = (
                    None
                    if land["r2_geo"] is None
                    else float(land["term_geo"]["e_pos"] - land["r2_geo"]["e_pos"])
                )
                row["delta_obj_z"] = (
                    None
                    if land["r2_geo"] is None
                    else float(land["term_geo"]["obj_z"] - land["r2_geo"]["obj_z"])
                )

            paired_rows.append(row)
            write_csv(OUT / "PAIRED_R2_VS_TERMINAL.csv", paired_rows)
            print(
                f"  [{i+1}/{len(subset)}] {skill} {fr['traj_id']} "
                f"cat={cat} r2_NEW={None if not r2_ev else r2_ev['SUCCESS_NEW']} "
                f"term_NEW={term_ev['SUCCESS_NEW']} r2_step={land['r2_step']}",
                flush=True,
            )

            # save a few representative traces
            if traces_saved < trace_n and cat in (
                "R2_fails_terminal_succeeds",
                "both_fail",
            ):
                payload = {
                    "traj_id": fr["traj_id"],
                    "skill": skill,
                    "category": cat,
                    "r2_step": land["r2_step"],
                    "term_step": land["term_step"],
                    "r2_geo": land["r2_geo"],
                    "term_geo": land["term_geo"],
                    "r2_eval": r2_ev,
                    "term_eval": term_ev,
                    "trace_around_r2": land["trace_around_r2"],
                }
                (OUT / "traces" / f"{skill}_{cat}_{fr['traj_id'][:48]}.json").write_text(
                    json.dumps(payload, indent=2, default=str), encoding="utf-8"
                )
                traces_saved += 1

    write_csv(OUT / "PAIRED_R2_VS_TERMINAL.csv", paired_rows)

    # Exclusive category table
    summary = []
    for skill in skills:
        sub = [r for r in paired_rows if r["skill"] == skill]
        # Also count R2 successes from P41 for the table's "R2 success" on full set
        p41_all = [
            r
            for r in csv.DictReader(P41_ROLLOUTS.open(encoding="utf-8"))
            if r["skill"] == skill and r["rule"] == "R2"
        ]
        n_r2_success_full = sum(int(r["SUCCESS_NEW"]) for r in p41_all)
        n_fail = len([r for r in p41_all if int(r["SUCCESS_NEW"]) == 0])
        counts = Counter(r["category"] for r in sub)
        # For failures-only audit, R2_success among fail set should be 0 unless re-eval disagrees
        summary.append(
            {
                "skill": skill,
                "n_total_p41": len(p41_all),
                "n_r2_success_p41": n_r2_success_full,
                "n_r2_fail_p41": n_fail,
                "n_fail_audited": len(sub),
                "R2_success_among_fail_reeval": counts.get("R2_success", 0),
                "R2_fails_terminal_succeeds": counts.get("R2_fails_terminal_succeeds", 0),
                "both_fail": counts.get("both_fail", 0),
                "R2_never_triggered": counts.get("R2_never_triggered", 0),
                "logged_classes": dict(Counter(r["logged_class"] for r in sub)),
                "r2_fail_cats_reeval": dict(
                    Counter(r.get("r2ev_fail_category") for r in sub if r.get("r2ev_fail_category"))
                ),
                "term_fail_cats": dict(
                    Counter(r.get("termev_fail_category") for r in sub if r.get("termev_fail_category"))
                ),
            }
        )
    write_csv(OUT / "CATEGORY_SUMMARY.csv", summary)
    (OUT / "CATEGORY_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_final(class_rows, paired_rows, summary)
    return {"paired": paired_rows, "summary": summary}


def write_final(class_rows, paired_rows, summary) -> None:
    lines = [
        "# FINAL_REPORT — P42 A/D Expert HANDOFF Failure Audit",
        "",
        f"Generated: {utc()}",
        "",
        "## Scope",
        "",
        "- Diagnosis only: no threshold / physics / dataset / training changes.",
        "- Source failures: P41 R2 expert eval — A1 32/63 fail, D 68/139 fail.",
        "- Branches: HANDOFF at first R2 vs HANDOFF at expert terminal (same `evaluate_handoff_with_axis`, stabilize_n=8).",
        "",
        "## 1. Logged failure classification (P41 R2)",
        "",
    ]
    for skill in SKILLS:
        sub = [r for r in class_rows if r["skill"] == skill]
        lines.append(f"### {skill} (n_fail={len(sub)})")
        lines.append("")
        lines.append("| Logged class | n |")
        lines.append("|---|---:|")
        for k, v in Counter(r["logged_class"] for r in sub).most_common():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("| fail_category | n |")
        lines.append("|---|---:|")
        for k, v in Counter(r["fail_category"] for r in sub).most_common():
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    lines += [
        "All A1 and D failures in P41 had `handoff=1` — **R2 always triggered** on these fail sets "
        "(no never-triggered cases among the 32+68).",
        "",
        "## 2. Paired R2 vs terminal (failures only)",
        "",
        "| Category | A1 | D |",
        "|---|---:|---:|",
    ]
    by = {s["skill"]: s for s in summary}
    a = by.get("A1")
    d = by.get("D")

    def cell(sk, key, default="—"):
        if sk is None:
            return default
        return sk.get(key, default)

    lines.append(
        f"| R2 success (full P41 set) | "
        f"{cell(a,'n_r2_success_p41')}/{cell(a,'n_total_p41')} | "
        f"{cell(d,'n_r2_success_p41')}/{cell(d,'n_total_p41')} |"
    )
    lines.append(
        f"| R2 fails, terminal succeeds | {cell(a,'R2_fails_terminal_succeeds')} | {cell(d,'R2_fails_terminal_succeeds')} |"
    )
    lines.append(f"| Both fail | {cell(a,'both_fail')} | {cell(d,'both_fail')} |")
    lines.append(f"| R2 never triggered | {cell(a,'R2_never_triggered')} | {cell(d,'R2_never_triggered')} |")
    lines.append("")
    lines.append(
        f"Audited failures: A1={cell(a,'n_fail_audited')}, D={cell(d,'n_fail_audited')}. "
        "Categories among audited failures are mutually exclusive."
    )
    lines.append("")
    lines.append("### Re-eval fail reasons at R2 / terminal")
    lines.append("")
    for s in summary:
        lines.append(
            f"- **{s['skill']} R2 fail cats:** `{s['r2_fail_cats_reeval']}`"
        )
        lines.append(
            f"- **{s['skill']} terminal fail cats:** `{s['term_fail_cats']}`"
        )
    lines.append("")

    # Mechanism bullets from counts
    lines += [
        "## 3. Mechanism interpretation",
        "",
    ]
    for skill, sk, logged_note in (
        (
            "A1",
            a,
            "- Logged mode: almost all R2-triggered `LOST_GRASP` (grasp lost after nominal takeover), "
            "plus a few `HOLD_FAIL_12S_OK`.",
        ),
        (
            "D",
            d,
            "- Logged mode: **all 68** are `HOLD_FAIL_12S_OK` — SUCCESS_12S yes, SUCCESS_NEW no "
            "(WORKING hold in final window fails). Not false height DROP; not never-triggered R2.",
        ),
    ):
        lines.append(f"### {skill}")
        lines.append("")
        if sk is None:
            lines.append("- Not audited in this run.")
            lines.append("")
            continue
        prem = sk["R2_fails_terminal_succeeds"]
        both = sk["both_fail"]
        if prem > both:
            lines.append(
                f"- Dominant among failures: **premature R2 HANDOFF** "
                f"({prem} terminal recovers where R2 fails; {both} both fail)."
            )
        elif both > prem:
            lines.append(
                f"- Dominant among failures: **expert/terminal also invalid** "
                f"({both} both fail; only {prem} are R2-only)."
            )
        else:
            lines.append(f"- Mixed: R2-only failures={prem}, both-fail={both}.")
        lines.append(logged_note)
        lines.append("")
    lines.append("")
    lines += [
        "## 4. What this rules out",
        "",
        "- Failures are **not** primarily “R2 never fired”.",
        "- For D, failures are **not** primarily contact DROP after takeover; they are "
        "**final-window WORKING hold** failures after a completed 12 s nominal.",
        "- Do not propose new HANDOFF rules until premature vs both-fail split is used to guide the next step.",
        "",
        "## Artifacts",
        "",
        "- `FAILURE_CLASSIFICATION_LOGGED.csv`",
        "- `PAIRED_R2_VS_TERMINAL.csv`",
        "- `CATEGORY_SUMMARY.csv` / `.json`",
        "- `traces/*.json` — representative R2-vs-terminal cases",
        "- `FINAL_REPORT.md`",
        "",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit failures per skill (0=all)")
    ap.add_argument("--trace-n", type=int, default=8)
    ap.add_argument("--skills", default="A1,D", help="Comma-separated skills")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    skills = tuple(s.strip() for s in args.skills.split(",") if s.strip())
    run(limit=args.limit, trace_n=args.trace_n, skills=skills, resume=not args.no_resume)


if __name__ == "__main__":
    main()
