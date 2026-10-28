"""Expert discrete replay under R0/R1/R2 — all ICs, A/B/C/D."""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index
from experiments.precision.p27_history_multiskill_transformer.run_p27 import load_npz_flex
from experiments.precision.p39_working_hold_handoff.paths import (
    OUT,
    P21,
    P21_MAN,
    P35,
    P35_MAN,
    M111_META,
)
from experiments.precision.p39_working_hold_handoff.rules import (
    RULE_IDS,
    DwellTracker,
    metric_meta,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book
from models.transformer_policy import DT_POLICY

EXPERT_OUT = OUT / "expert_validation"


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


def collect_jobs() -> list[dict]:
    man = json.loads(P21_MAN.read_text(encoding="utf-8"))
    index = build_start_index(man)
    jobs: list[dict] = []
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
    for r in csv.DictReader(P35_MAN.open(encoding="utf-8")):
        if r.get("class") != "P1":
            continue
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
    # reset tracker after peeking init (R0 ready at step 0 is valid; re-observe cleanly)
    tracker = DwellTracker.from_rule(rule_id, dt=float(DT_POLICY))

    toks = np.asarray(tokens, np.int64)
    res = np.asarray(residuals, np.float32)
    n = len(toks)
    if n and int(toks[-1]) == HANDOFF:
        n -= 1

    handoff = False
    handoff_step = None
    drop = False
    last = {}
    # Check state before each action; if ready → HANDOFF without applying further motion.
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
        action = decode_m111_action(book, tok, rr, rho=rho)
        step_legal(env.sim, action)

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
        "ever_partial_ok": int(tracker.ever_partial or (rule_id == "R0" and handoff)),
        "drop_before_handoff": int(drop and not handoff),
        "never_satisfied": int(not handoff and not drop),
        "hand_theta": last.get("theta") if handoff else None,
        "hand_e_pos": last.get("e_pos") if handoff else None,
        "hand_bilateral": int(last.get("bilateral", False)) if handoff else None,
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
        for skill in sorted({r["skill"] for r in rows}):
            sub = [r for r in rows if r["rule"] == rule and r["skill"] == skill]
            n = len(sub)
            n_hand = sum(int(r["handoff"]) for r in sub)
            s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
            snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
            drop = sum(int(r["drop_before_handoff"]) for r in sub)
            never = sum(int(r["never_satisfied"]) for r in sub)
            steps = [int(r["handoff_step"]) for r in sub if r["handoff_step"] is not None]
            hand_rows = [r for r in sub if int(r["handoff"])]
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
                    "reasons": dict(Counter(r["eval_reason"] for r in sub)),
                }
            )
    return agg


def r2_reliable(agg: list[dict]) -> tuple[bool, list[str]]:
    """Conservative Gate: R2 conversion ≥0.8 when ≥5 handoffs, and C NEW/all ≥0.2."""
    flags: list[str] = []
    ok = True
    for a in agg:
        if a["rule"] != "R2":
            continue
        skill = a["skill"]
        conv = a["conversion_rate"]
        n_hand = a["handoff"]
        new_all = a["NEW_over_all"] or 0.0
        if n_hand >= 5 and (conv is None or conv < 0.8):
            ok = False
            flags.append(f"R2 {skill}: NEW|handoff={a['NEW_given_handoff']} (<0.8)")
        if skill == "C" and new_all < 0.2:
            ok = False
            flags.append(f"R2 C: NEW/all={a['SUCCESS_NEW']}/{a['n_traj']} (<0.2)")
        if n_hand == 0 and a["n_traj"] >= 5:
            ok = False
            flags.append(f"R2 {skill}: zero handoffs")
    return ok, flags


def run_expert_validation(*, limit: int = 0, rules: tuple[str, ...] = RULE_IDS) -> dict:
    EXPERT_OUT.mkdir(parents=True, exist_ok=True)
    (EXPERT_OUT / "RULE_METRIC.json").write_text(
        json.dumps(metric_meta(), indent=2), encoding="utf-8"
    )
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    jobs = collect_jobs()
    if limit > 0:
        jobs = jobs[:limit]

    rows: list[dict] = []
    total = len(jobs) * len(rules)
    k = 0
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

        restore()
        m0 = measure(env.sim, U_TGT, P_TGT)

        for rule_id in rules:
            k += 1
            res = replay_expert_rule(
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
                "pre_theta": float(m0["e_theta_deg"]),
                "pre_e_pos": float(m0["e_pos_mm"]),
                **res,
            }
            rows.append(row)
            print(
                f"  [{k}/{total}] {rule_id} {skill} {str(job['traj_id'])[:36]} "
                f"hand={row['handoff']} step={row['handoff_step']} "
                f"NEW={row['SUCCESS_NEW']} cat={row['fail_category']}",
                flush=True,
            )

    write_csv(EXPERT_OUT / "EXPERT_ROLLOUTS.csv", rows)
    agg = aggregate(rows)
    write_csv(EXPERT_OUT / "AGGREGATE.csv", agg)

    # Per-root wide compare
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r["skill"], r["traj_id"])
        by_key.setdefault(key, {"skill": r["skill"], "traj_id": r["traj_id"], "split": r["split"]})
        rid = r["rule"]
        by_key[key][f"{rid}_handoff"] = r["handoff"]
        by_key[key][f"{rid}_step"] = r["handoff_step"]
        by_key[key][f"{rid}_NEW"] = r["SUCCESS_NEW"]
        by_key[key][f"{rid}_12S"] = r["SUCCESS_12S"]
        by_key[key][f"{rid}_cat"] = r["fail_category"]
    write_csv(EXPERT_OUT / "PER_ROOT_COMPARE.csv", list(by_key.values()))

    reliable, flags = r2_reliable(agg)
    verdict = {
        "r2_reliable": reliable,
        "flags": flags,
        "utc": utc(),
        "n_jobs": len(jobs),
        "n_rows": len(rows),
    }
    (EXPERT_OUT / "R2_RELIABILITY.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Failure-focused dump
    bad = [r for r in rows if int(r["SUCCESS_NEW"]) == 0]
    write_csv(EXPERT_OUT / "FAILURES.csv", bad)

    print(f"=== Expert done R2_reliable={reliable} flags={flags} ===", flush=True)
    return {"rows": rows, "agg": agg, "verdict": verdict}
