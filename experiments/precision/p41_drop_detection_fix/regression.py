"""Regression tests for grasp-aware `_dropped` (P41)."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from envs.outcomes.horizon_outcome import _dropped, _monitor
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META, P35, P35_MAN
from experiments.precision.p40_c_handoff_investigation.run_p40 import (
    load_c_jobs,
    replay_to_landmarks,
    snap_geo,
)
from experiments.precision.p41_drop_detection_fix.paths import OUT
from models.m111_checkpoint import load_m111_book
from pathlib import Path as P


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


def unit_dropped_semantics(env) -> list[dict]:
    """Synthetic monitor rows: grasped low-z must not drop; free low-z must."""
    rows = []
    # No sim path with contacts
    for nL, nR, z, expect in (
        (1, 1, 0.16, False),
        (1, 0, 0.16, False),
        (0, 0, 0.16, True),
        (0, 0, 0.50, False),  # high free — may still be False via Z_ABORT only if z<0.44
    ):
        mon = {"nL": nL, "nR": nR, "z": z}
        # without sim: z=0.50 → not dropped; z=0.16 no contact → dropped
        got = bool(_dropped(mon, None))
        # Fix expect for high free
        if z >= 0.44 and nL + nR == 0:
            expect = False
        rows.append(
            {
                "case": f"legacy_nL{nL}_nR{nR}_z{z}",
                "expect_dropped": int(expect),
                "got_dropped": int(got),
                "pass": int(got == expect),
            }
        )
    # With live sim at default pose — just ensure API works
    mon = _monitor(env.sim)
    mon_low_grasp = {"nL": 5, "nR": 5, "z": 0.16}
    mon_low_free = {"nL": 0, "nR": 0, "z": 0.16}
    g1 = bool(_dropped(mon_low_grasp, env.sim))
    g2 = bool(_dropped(mon_low_free, env.sim))
    rows.append({"case": "sim_grasped_z0.16", "expect_dropped": 0, "got_dropped": int(g1), "pass": int(not g1)})
    rows.append({"case": "sim_free_z0.16", "expect_dropped": 1, "got_dropped": int(g2), "pass": int(g2)})
    return rows


def c_branch_regression(*, repr_n: int = 8) -> list[dict]:
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    jobs = load_c_jobs()
    picks = sorted(jobs, key=lambda j: j["traj_id"])[:repr_n]
    out = []
    for ji, job in enumerate(picks):
        print(f"  [regress C {ji+1}/{len(picks)}] {job['traj_id'][:48]}", flush=True)
        land = replay_to_landmarks(env, job, book, rho)
        ref = job["ep"]["ref"]
        for tag in ("first_working", "r2", "mid", "terminal"):
            if tag not in land["captures"]:
                continue
            cap = land["captures"][tag]
            restore_capture(env, cap)
            pre = snap_geo(env.sim)
            mon = _monitor(env.sim)
            pre_dropped = bool(_dropped(mon, env.sim))
            hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
            out.append(
                {
                    "traj_id": job["traj_id"],
                    "branch": tag,
                    "branch_step": land["landmarks"][tag]["step"],
                    "pre_obj_z": pre["obj_z"],
                    "pre_bilateral": pre["bilateral"],
                    "pre_working": pre["working"],
                    "pre_dropped_flag": int(pre_dropped),
                    "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
                    "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
                    "reason": hand.get("reason"),
                }
            )
            print(
                f"    {tag} z={pre['obj_z']:.3f} bil={pre['bilateral']} "
                f"pre_drop={pre_dropped} NEW={int(bool(hand.get('SUCCESS_NEW')))} "
                f"reason={hand.get('reason')}",
                flush=True,
            )
    return out


def d_terminal_smoke(*, limit: int = 12) -> list[dict]:
    """Replay a few P35 P1 D expert terminals through evaluate_handoff."""
    from models.m111_checkpoint import HANDOFF, decode_m111_action

    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    book, rho, _ = load_m111_book(M111_META)
    rows_m = [r for r in csv.DictReader(P35_MAN.open(encoding="utf-8")) if r.get("class") == "P1"]
    p39 = OUT.parent / "p39_working_hold_handoff" / "expert_validation" / "EXPERT_ROLLOUTS.csv"
    ok_ids: set[str] = set()
    if p39.is_file():
        for r in csv.DictReader(p39.open(encoding="utf-8")):
            if r.get("skill") == "D" and r.get("rule") == "R2" and r.get("SUCCESS_NEW") == "1":
                ok_ids.add(r["traj_id"])
    preferred = [r for r in rows_m if r["traj_id"] in ok_ids][:limit]
    if len(preferred) < min(limit, len(rows_m)):
        for r in rows_m:
            if r not in preferred:
                preferred.append(r)
            if len(preferred) >= limit:
                break
    out = []
    for r in preferred:
        blob = pickle.loads(P(r["ic_path"]).read_bytes())
        z = np.load(P35 / r["npz_relpath"], allow_pickle=False)
        toks = np.asarray(z["tokens"], np.int64)
        restore_capture(env, blob["capture"])
        n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
        res = np.asarray(z["residuals"], np.float32)
        for t in range(n):
            if latched(env.sim):
                break
            tok = int(toks[t])
            if tok == HANDOFF:
                continue
            step_legal(
                env.sim,
                decode_m111_action(book, tok, res[t] if t < len(res) else np.zeros(7), rho=rho),
            )
        pre = snap_geo(env.sim)
        hand = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
        out.append(
            {
                "traj_id": r["traj_id"],
                "pre_obj_z": pre["obj_z"],
                "pre_bilateral": pre["bilateral"],
                "pre_working": pre["working"],
                "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
                "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
                "reason": hand.get("reason"),
                "was_p39_r2_success": int(r["traj_id"] in ok_ids),
            }
        )
    return out


def genuine_fall_smoke(env) -> dict:
    """Zero finger contacts + low z → _dropped True; latched still separate."""
    mon = {"nL": 0, "nR": 0, "z": 0.10}
    return {
        "case": "free_fall_z0.10",
        "dropped": int(_dropped(mon, env.sim)),
        "pass": int(_dropped(mon, env.sim) is True),
    }


def run_regression() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    print("=== P41 unit _dropped semantics ===", flush=True)
    unit = unit_dropped_semantics(env)
    unit.append(genuine_fall_smoke(env))
    write_csv(OUT / "REGRESSION_UNIT_DROPPED.csv", unit)

    print("=== P41 C branch regression (P40 reps) ===", flush=True)
    c_rows = c_branch_regression(repr_n=8)
    write_csv(OUT / "REGRESSION_C_BRANCHES.csv", c_rows)

    print("=== P41 D terminal smoke ===", flush=True)
    d_rows = d_terminal_smoke(limit=12)
    write_csv(OUT / "REGRESSION_D_TERMINAL.csv", d_rows)

    summary = {
        "unit_pass": all(int(r["pass"]) for r in unit if "pass" in r),
        "c_first_working_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "first_working"),
        "c_r2_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "r2"),
        "c_mid_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "mid"),
        "c_terminal_NEW": sum(int(r["SUCCESS_NEW"]) for r in c_rows if r["branch"] == "terminal"),
        "c_n_per_branch": 8,
        "d_terminal_NEW": sum(int(r["SUCCESS_NEW"]) for r in d_rows),
        "d_n": len(d_rows),
        "d_p39_success_retained": sum(
            1 for r in d_rows if int(r["was_p39_r2_success"]) and int(r["SUCCESS_NEW"])
        ),
        "d_p39_success_n": sum(int(r["was_p39_r2_success"]) for r in d_rows),
    }
    (OUT / "REGRESSION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("REGRESSION", summary, flush=True)
    return {"unit": unit, "c": c_rows, "d": d_rows, "summary": summary}
