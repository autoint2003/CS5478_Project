"""Gate 1 — Validate shared WORKING HANDOFF on A/B/C(/D) expert discrete replays.

Replay encoded M111 motion tokens until first WORKING-satisfied state, then
apply P36 nominal HANDOFF contract + 12s STRICT (evaluate_handoff_with_axis).

Includes all packaged trajectories (successful and failed experts).
Does not retrain. Does not modify source datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p27_history_multiskill_transformer.run_p27 import (  # noqa: E402
    load_npz_flex,
)
from experiments.precision.p37_working_handoff_d.handoff_rules import (  # noqa: E402
    working_geometry,
)
from experiments.precision.p38_unified_motion_only_sft.paths import (  # noqa: E402
    M111_META,
    OUT,
    P21,
    P21_MAN,
    P23,
    P23_MAN,
    P35,
    P35_MAN,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402

GATE1 = OUT / "gate1_working_handoff"


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


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def abs_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def working_ok(sim) -> tuple[bool, dict]:
    g = working_geometry(sim)
    return bool(g["working_ok"]), g


def replay_until_working(
    env,
    *,
    restore_fn,
    ref,
    tokens: np.ndarray,
    residuals: np.ndarray,
    book,
    rho,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Discrete expert replay; HANDOFF at first WORKING frame (not learned)."""
    restore_fn()
    init_ok, init_g = working_ok(env.sim)
    toks = np.asarray(tokens, np.int64)
    res = np.asarray(residuals, np.float32)
    n = len(toks)
    if n and int(toks[-1]) == HANDOFF:
        n -= 1
    if max_steps is not None:
        n = min(n, int(max_steps))

    reached = False
    reach_step = None
    drop = False
    hand_g = None
    for t in range(n):
        if latched(env.sim):
            drop = True
            break
        ok, g = working_ok(env.sim)
        if ok:
            reached = True
            reach_step = t
            hand_g = g
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        action = decode_m111_action(book, tok, rr, rho=rho)
        step_legal(env.sim, action)

    if not reached and not drop:
        ok, g = working_ok(env.sim)
        if ok:
            reached = True
            reach_step = n
            hand_g = g

    out: dict[str, Any] = {
        "init_working": int(init_ok),
        "init_theta": init_g["dtheta_deg"],
        "init_e_pos": init_g["e_pos_mm"],
        "n_motion_available": n,
        "reached_working": int(reached),
        "reach_step": reach_step,
        "drop_before_working": int(drop),
        "hand_theta": None if hand_g is None else hand_g["dtheta_deg"],
        "hand_e_pos": None if hand_g is None else hand_g["e_pos_mm"],
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
        "eval_reason": None,
        "win_mean_pos": None,
        "win_mean_theta": None,
    }
    if drop or not reached:
        out["eval_reason"] = "DROP_BEFORE_WORKING" if drop else "NEVER_REACHED_WORKING"
        return out

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    win = hand.get("window_SUCCESS_NEW") or {}
    out.update(
        {
            "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
            "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
            "eval_reason": hand.get("reason"),
            "win_mean_pos": win.get("mean_pos"),
            "win_mean_theta": win.get("mean_theta"),
            "win_working_hold": int(bool(win.get("working_SUCCESS_NEW_hold"))),
        }
    )
    return out


def collect_abc_jobs() -> list[dict]:
    man = json.loads(P21_MAN.read_text(encoding="utf-8"))
    index = build_start_index(man)
    jobs = []
    for r in man["trajectories"]:
        fam = r["family"]
        if fam not in ("A", "B", "C"):
            continue
        ep = index[r["traj_id"]]
        z = load_npz_flex(P21, r["npz_relpath"])
        jobs.append(
            {
                "skill": {"A": "A1", "B": "B1", "C": "C"}[fam],
                "family": fam,
                "traj_id": r["traj_id"],
                "split": r["split"],
                "lineage_id": r.get("lineage_id"),
                "source": "p21",
                "ep": ep,
                "tokens": np.asarray(z["tokens"], np.int64),
                "residuals": np.asarray(z["residuals"], np.float32),
            }
        )
    return jobs


def collect_d_jobs() -> list[dict]:
    """P35 P1 Precision trajectories (D skill used in P36)."""
    rows = [
        r
        for r in csv.DictReader(P35_MAN.open(encoding="utf-8"))
        if r.get("class") == "P1"
    ]
    jobs = []
    for r in rows:
        z = np.load(P35 / r["npz_relpath"], allow_pickle=False)
        blob = pickle.loads(Path(r["ic_path"]).read_bytes())
        jobs.append(
            {
                "skill": "D",
                "family": "D",
                "traj_id": r["traj_id"],
                "split": r["split"],
                "lineage_id": r.get("lineage_id"),
                "source": "p35_p1",
                "capture": blob["capture"],
                "ref": blob["ref"],
                "tokens": np.asarray(z["tokens"], np.int64),
                "residuals": np.asarray(z["residuals"], np.float32),
            }
        )
    return jobs


def collect_d2_jobs() -> list[dict]:
    """Optional D2 packaged rows (eval suite corpus)."""
    man = json.loads(P23_MAN.read_text(encoding="utf-8"))
    jobs = []
    for r in man["trajectories"]:
        if r.get("family") != "D2":
            continue
        z = load_npz_flex(P23, r["npz_relpath"])
        # D2 ICs resolved at eval time; for Gate1 try npz-only mid-traj is insufficient
        # without IC — skip if no start. Mark as needs_ic.
        jobs.append(
            {
                "skill": "D2",
                "family": "D2",
                "traj_id": r["traj_id"],
                "split": r["split"],
                "lineage_id": r.get("lineage_id"),
                "source": "p23_d2",
                "npz_relpath": r["npz_relpath"],
                "tokens": np.asarray(z["tokens"], np.int64),
                "residuals": np.asarray(z["residuals"], np.float32),
                "needs_external_ic": True,
            }
        )
    return jobs


def run_gate1(*, include_d: bool = True, limit: int = 0) -> dict:
    GATE1.mkdir(parents=True, exist_ok=True)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()

    # Document WORKING metric identity
    meta = {
        "WORKING_THETA_DEG": WORKING_THETA_DEG,
        "WORKING_POS_MM": WORKING_POS_MM,
        "U_TGT": np.asarray(U_TGT, float).tolist(),
        "P_TGT": np.asarray(P_TGT, float).tolist(),
        "measure": "evaluation.grasp_geometry.measure",
        "handoff_eval": "evaluation.axis_aware_success.evaluate_handoff_with_axis",
        "nominal_contract": "envs.outcomes.recovery_success.apply_handoff (P36)",
        "utc": utc(),
    }
    (GATE1 / "WORKING_METRIC.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    jobs = collect_abc_jobs()
    if include_d:
        jobs.extend(collect_d_jobs())
    if limit > 0:
        jobs = jobs[:limit]

    rows = []
    for i, job in enumerate(jobs):
        skill = job["skill"]
        if skill in ("A1", "B1", "C"):
            ep = job["ep"]

            def _restore(ep=ep):
                from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start

                restore_start(env, ep)

            ref = ep["ref"]
        else:
            cap, ref = job["capture"], job["ref"]

            def _restore(cap=cap):
                restore_capture(env, cap)

        # Spot-check reference pose: WORKING uses fixed U_TGT/P_TGT (not ref r_des)
        m0 = None
        _restore()
        m0 = measure(env.sim, U_TGT, P_TGT)

        res = replay_until_working(
            env,
            restore_fn=_restore,
            ref=ref,
            tokens=job["tokens"],
            residuals=job["residuals"],
            book=book,
            rho=rho,
        )
        row = {
            "traj_id": job["traj_id"],
            "skill": skill,
            "family": job["family"],
            "split": job["split"],
            "source": job["source"],
            "lineage_id": job.get("lineage_id"),
            "pre_theta_measure": float(m0["e_theta_deg"]),
            "pre_e_pos_measure": float(m0["e_pos_mm"]),
            **res,
        }
        rows.append(row)
        print(
            f"  [{i+1}/{len(jobs)}] {skill} {job['traj_id'][:40]} "
            f"reach={row['reached_working']} NEW={row['SUCCESS_NEW']} "
            f"12S={row['SUCCESS_12S']} reason={row['eval_reason']}",
            flush=True,
        )

    write_csv(GATE1 / "EXPERT_WORKING_HANDOFF.csv", rows)

    # Aggregates
    agg = []
    for skill in sorted({r["skill"] for r in rows}):
        sub = [r for r in rows if r["skill"] == skill]
        n = len(sub)
        reach = sum(int(r["reached_working"]) for r in sub)
        s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
        snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
        drop = sum(int(r["drop_before_working"]) for r in sub)
        never = sum(1 for r in sub if not int(r["reached_working"]) and not int(r["drop_before_working"]))
        hold_fail = sum(
            1
            for r in sub
            if int(r["reached_working"]) and int(r["SUCCESS_12S"]) and not int(r["SUCCESS_NEW"])
        )
        post_drop = sum(
            1
            for r in sub
            if int(r["reached_working"]) and not int(r["SUCCESS_12S"]) and int(r["SUCCESS_NEW"]) == 0
            and r.get("eval_reason") and "DROP" in str(r.get("eval_reason")).upper()
        )
        # Among those that reached WORKING
        reach_rows = [r for r in sub if int(r["reached_working"])]
        conv = sum(int(r["SUCCESS_NEW"]) for r in reach_rows)
        agg.append(
            {
                "skill": skill,
                "n_traj": n,
                "reached_working": reach,
                "SUCCESS_12S": s12,
                "SUCCESS_NEW": snew,
                "drop_before_working": drop,
                "never_reached_working": never,
                "hold_fail_12s_ok_new_fail": hold_fail,
                "NEW_given_reach": f"{conv}/{len(reach_rows)}",
                "conversion_rate": (conv / len(reach_rows)) if reach_rows else None,
                "reasons": dict(Counter(r["eval_reason"] for r in sub)),
            }
        )
    write_csv(GATE1 / "GATE1_AGGREGATE.csv", agg)

    # Counterexamples: reached WORKING but not SUCCESS_NEW
    bad = [
        r
        for r in rows
        if int(r["reached_working"]) and not int(r["SUCCESS_NEW"])
    ]
    write_csv(GATE1 / "WORKING_BUT_NOT_STRICT.csv", bad)

    reliable = all(
        (a["conversion_rate"] is None or a["conversion_rate"] >= 0.85)
        or a["reached_working"] == 0
        for a in agg
    )
    # Stricter: if any skill has many reach-but-fail, flag
    flags = []
    for a in agg:
        if a["reached_working"] >= 5 and (a["conversion_rate"] or 0) < 0.8:
            flags.append(
                f"{a['skill']}: WORKING→NEW only {a['NEW_given_reach']} — WORKING may be unreliable"
            )

    report = f"""# GATE1 — Shared WORKING HANDOFF Validation

Generated: {utc()}

## Metric (identical for all skills)

- θ ≤ **{WORKING_THETA_DEG}°**, e_pos ≤ **{WORKING_POS_MM} mm**
- From `evaluation.grasp_geometry.measure(U_TGT, P_TGT)` + SUCCESS_NEW WORKING constants
- Single-step trigger; STRICT 12s via `evaluate_handoff_with_axis` (P36 nominal contract)

## Results (discrete expert replay → first WORKING → HANDOFF)

| Skill | n | Reach WORKING | SUCCESS_12S | SUCCESS_NEW | NEW\\|reach | Drop before | Never reach |
|---|---:|---:|---:|---:|---|---:|---:|
"""
    for a in agg:
        report += (
            f"| {a['skill']} | {a['n_traj']} | {a['reached_working']} | {a['SUCCESS_12S']} | "
            f"**{a['SUCCESS_NEW']}** | {a['NEW_given_reach']} | {a['drop_before_working']} | "
            f"{a['never_reached_working']} |\n"
        )
    report += f"""
Hold-fail (12S=1, NEW=0) after WORKING: see `WORKING_BUT_NOT_STRICT.csv` (n={len(bad)}).

## Reliability verdict

"""
    if flags:
        report += "**CAUTION — skill-specific issues:**\n" + "\n".join(f"- {f}" for f in flags) + "\n"
        report += "Do not silently add skill-specific HANDOFF rules; investigate counterexamples first.\n"
    else:
        report += (
            "**WORKING appears sufficiently reliable** as a shared termination trigger for "
            "skills with nontrivial reach counts (conversion ≥ ~0.85 or few reaches).\n"
        )
    report += f"\nAuto flag reliable={reliable and not flags}.\n"
    report += "\n## Artifacts\n\n`EXPERT_WORKING_HANDOFF.csv`, `GATE1_AGGREGATE.csv`, `WORKING_BUT_NOT_STRICT.csv`, `WORKING_METRIC.json`\n"
    write_md(GATE1 / "GATE1_REPORT.md", report)
    print(report, flush=True)
    return {"rows": rows, "agg": agg, "flags": flags, "n_bad": len(bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-d", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_gate1(include_d=not args.no_d, limit=args.limit)


if __name__ == "__main__":
    main()
